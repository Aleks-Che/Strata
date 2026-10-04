// GLM single-owner pipe engine. All model state is discarded between requests.
#include "runtime.hpp"
#include "runtime_memory.hpp"
#include "protocol.hpp"
#include "nlohmann/json.hpp"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <mutex>
#include <thread>

using namespace strata_glm;
using Clock=std::chrono::steady_clock;
static double ms(Clock::time_point start) { return std::chrono::duration<double,std::milli>(Clock::now()-start).count(); }
struct Options { std::string model; int context=2048,batch=16,threads=4,ram_percent=0,vram_percent=0; };
static Options options(int argc,char ** argv) {
    Options o;
    for (int i=1;i<argc;++i) {
        const std::string key=argv[i];
        if (key=="--serve") continue;
        require(i+1<argc,"missing value for "+key); const std::string v=argv[++i];
        if (key=="--native") o.model=v;
        else if (key=="--max-context") o.context=integer(v);
        else if (key=="--batch-size") o.batch=integer(v);
        else if (key=="--threads") o.threads=integer(v);
        else if (key=="--ram-target-percent") o.ram_percent=integer(v);
        else if (key=="--vram-target-percent") o.vram_percent=integer(v);
        else throw std::runtime_error("unsupported GLM option: "+key);
    }
    require(!o.model.empty() && o.context>=32 && o.context<=1048576 && o.batch>=1 && o.batch<=4096 && o.batch<=o.context && o.threads>=1 && o.threads<=256,
        "invalid model/context/batch/thread settings");
    return o;
}
using Sampler=std::unique_ptr<llama_sampler,decltype(&llama_sampler_free)>;
static Sampler sampler(const Request & r,int vocab) {
    Sampler s(llama_sampler_chain_init(llama_sampler_chain_default_params()),llama_sampler_free);
    require(bool(s),"sampler allocation failed");
    llama_sampler_chain_add(s.get(),llama_sampler_init_penalties(vocab,int(r.value("penalty_last_n",64)),
        float(r.value("penalty_repeat",1)),float(r.value("penalty_freq",0)),float(r.value("penalty_present",0))));
    if (r.value("temperature",0)==0) llama_sampler_chain_add(s.get(),llama_sampler_init_greedy());
    else {
        llama_sampler_chain_add(s.get(),llama_sampler_init_top_k(int(r.value("top_k",64))));
        llama_sampler_chain_add(s.get(),llama_sampler_init_top_p(float(r.value("top_p",1)),1));
        llama_sampler_chain_add(s.get(),llama_sampler_init_min_p(float(r.value("min_p",0)),1));
        llama_sampler_chain_add(s.get(),llama_sampler_init_temp(float(r.value("temperature",1))));
        llama_sampler_chain_add(s.get(),llama_sampler_init_dist(uint32_t(r.value("seed",LLAMA_DEFAULT_SEED))));
    }
    for (auto t:r.tokens) llama_sampler_accept(s.get(),t);
    return s;
}
struct Command { std::string line; std::atomic<bool> cancel{false}; explicit Command(std::string text):line(std::move(text)){} };
static void generate(llama_context * ctx,const llama_vocab * vocab,const Options & o,Command & command,RuntimeMemory & memory) {
    int generated=0; size_t prompt=0; double prompt_ms=0,decode_ms=0;
    const char * finish="length";
    try {
        const auto r=request(command.line,o.context,llama_vocab_n_tokens(vocab)); prompt=r.tokens.size();
        auto smpl=sampler(r,llama_vocab_n_tokens(vocab));
        clear(ctx); memory.refresh(); strata_glm_sync_reset();
        auto start=Clock::now();
        for (size_t i=0;i<prompt && !command.cancel.load();i+=o.batch) {
            const int n=int(std::min(size_t(o.batch),prompt-i));
            decode(ctx,r.tokens,int(i),n,int(i));
            std::cout<<"PP "<<i+n<<" "<<prompt<<"\n"<<std::flush;
        }
        prompt_ms=ms(start);
        start=Clock::now();
        std::vector<llama_token> token(1);
        for (int i=0;i<r.count && !command.cancel.load();++i) {
            const auto * logits=llama_get_logits_ith(ctx,-1); require(logits!=nullptr,"missing logits");
            for (int j=0;j<llama_vocab_n_tokens(vocab);++j) require(std::isfinite(logits[j]),"non-finite output logits");
            token[0]=llama_sampler_sample(smpl.get(),ctx,-1);
            ++generated; std::cout<<"T "<<token[0]<<"\n"<<std::flush;
            if (llama_vocab_is_eog(vocab,token[0])) { finish="stop"; break; }
            if (i+1<r.count && !command.cancel.load()) decode(ctx,token,0,1,int(prompt)+i);
        }
        decode_ms=ms(start);
        if (command.cancel.load()) finish="cancel";
        // STOP never interrupts a partially executed graph. Bound latency to a
        // microbatch, synchronize and clear all hybrid components before DONE.
        clear(ctx);
        auto stats=strata_glm_sync_snapshot();
        std::cerr<<"STRATA_GLM_REQUEST source_bytes="<<stats.source_bytes<<" h2d_bytes="<<stats.h2d_bytes
            <<" source_ms="<<stats.source_ms<<" h2d_ms="<<stats.h2d_ms<<" gpu_nodes="<<stats.gpu_nodes
            <<" expert_nodes="<<stats.expert_nodes<<" cpu_nodes="<<stats.rejected_cpu_nodes<<" finish="<<finish<<"\n";
        std::cerr<<"STRATA_GLM_MEMORY "<<memory.snapshot().dump()<<"\n";
        std::cout<<"DONE "<<generated<<" "<<prompt<<" "<<prompt_ms<<" "<<decode_ms<<" "<<finish<<" 0 0 0\n"<<std::flush;
    } catch(const std::exception & e) {
        clear(ctx);
        if (command.cancel.load()) std::cout<<"DONE "<<generated<<" "<<prompt<<" "<<prompt_ms<<" "<<decode_ms<<" cancel 0 0 0\n"<<std::flush;
        else std::cout<<"ERR "<<e.what()<<"\n"<<std::flush;
    }
}
static void serve(llama_context * ctx,const llama_vocab * vocab,const Options & o,RuntimeMemory & memory) {
    std::mutex mutex; std::condition_variable cv;
    std::deque<std::shared_ptr<Command>> pending;
    std::shared_ptr<Command> active;
    bool ended=false;
    std::thread reader([&] {
        std::string line;
        while (std::getline(std::cin,line)) {
            if (!line.empty() && line.back()=='\r') line.pop_back();
            if (line=="QUIT") break;
            std::lock_guard lock(mutex);
            if (line=="STOP") {
                if (active) active->cancel.store(true);
                for (const auto & task:pending) task->cancel.store(true);
            } else {
                if (pending.size()>=16) break;
                pending.push_back(std::make_shared<Command>(std::move(line)));
            }
            cv.notify_one();
        }
        std::lock_guard lock(mutex); ended=true;
        if (active) active->cancel.store(true);
        cv.notify_one();
    });
    while (true) {
        {
            std::unique_lock lock(mutex); cv.wait(lock,[&]{return ended || !pending.empty();});
            if (ended) break;
            active=pending.front(); pending.pop_front();
        }
        generate(ctx,vocab,o,*active,memory);
        { std::lock_guard lock(mutex); active.reset(); }
    }
    reader.join();
}
int main(int argc,char ** argv) {
    if (argc==2 && std::string(argv[1])=="--version") {
        std::cout<<nlohmann::json({{"architecture","glm5next"},{"protocol_version",1},
            {"engine","glm5next-sync-p1"},{"source_sha",STRATA_GLM_SOURCE_SHA},{"patch_set",STRATA_GLM_PATCH_SET}}).dump()<<"\n";
        return 0;
    }
    try {
        const auto o=options(argc,argv);
        environment(); ggml_backend_load_all(); llama_backend_init(); strata_glm_sync_enable(true);
        {
            auto model=load(o.model); auto ctx=context(model.get(),o.context,o.batch,o.threads);
            RuntimeMemory memory(model,o.model,o.ram_percent,o.vram_percent); memory.warm();
            const auto usage=memory.snapshot(); std::cerr<<"STRATA_GLM_MEMORY "<<usage.dump()<<"\n";
            auto * vocab=llama_model_get_vocab(model.get());
            std::cout<<"INFO engine=glm5next-sync-p1 architecture=glm5next backend=llama.cpp mtp=0 spec=0 speculative=none"
                <<" expert_storage=mmap expert_compute=gpu gpu_only=1 expert_cache_mib="<<usage["cache_resident_bytes"].get<uint64_t>()/(1<<20)
                <<" ram_target_percent="<<o.ram_percent<<" vram_target_percent="<<o.vram_percent<<" expert_stage_mib=16"
                <<" expert_pipeline=0 expert_read_mode=mmap kv=fp16 flash_attention=0 tf32=0 conversation_cache=0\n"
                <<"READY "<<llama_n_ctx(ctx.get())<<" stop session-id\n"<<std::flush;
            serve(ctx.get(),vocab,o,memory);
        }
        strata_glm_sync_release(); llama_backend_free(); return 0;
    } catch (const std::exception & e) {
        strata_glm_sync_release(); llama_backend_free(); std::cerr<<"strata-glm5next: "<<e.what()<<"\n"; return 1;
    }
}
