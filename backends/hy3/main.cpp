// Hy3 pipe baseline: one model owner, fresh KV/sampler for every request.
#include "runtime.hpp"
#include "protocol.hpp"
#include "tokenizer_protocol.hpp"
#include "nlohmann/json.hpp"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <thread>

using namespace hy3;
using Clock = std::chrono::steady_clock;
using json = nlohmann::ordered_json;
static double ms(Clock::time_point start) { return std::chrono::duration<double,std::milli>(Clock::now()-start).count(); }
struct Options {
    std::string model, logits_file;
    int context=2048, batch=17, copy_mode=2;
    ggml_type kv=GGML_TYPE_F32;
    bool fixture=false, inspect_only=false;
};
static Options options(int argc,char ** argv) {
    Options out;
    for(int i=1;i<argc;++i) {
        const std::string key=argv[i];
        if(key=="--serve") continue;
        if(key=="--fixture") {out.fixture=true;continue;}
        if(key=="--inspect-only") {out.inspect_only=true;continue;}
        require(i+1<argc,"missing option value: "+key);
        const std::string value=argv[++i];
        if(key=="--native") out.model=value;
        else if(key=="--max-context") out.context=integer(value);
        else if(key=="--batch-size") out.batch=integer(value);
        else if(key=="--logits-file") out.logits_file=value;
        else if(key=="--copy-mode") {
            require(value=="native" || value=="pinned","copy-mode must be native or pinned");
            out.copy_mode=value=="native" ? 1 : 2;
        } else if(key=="--kv") {
            require(value=="f32","only validated F32 KV is admitted by this baseline");
        } else throw std::runtime_error("unsupported Hy3 option: "+key);
    }
    require(!out.model.empty() && out.context>=32 && out.context<=2048 && out.batch>=1 && out.batch<=32 && out.batch<=out.context,"invalid model/context/batch");
    if(!out.logits_file.empty()) {
        const auto p=std::filesystem::absolute(out.logits_file);
        require(p.extension()==".f32","logits output must end with .f32");
        require(!std::filesystem::exists(p) || !std::filesystem::equivalent(p,out.model),"logits output aliases model");
    }
    return out;
}
using Sampler = std::unique_ptr<llama_sampler,decltype(&llama_sampler_free)>;
static Sampler sampler(const Request & request, int vocab) {
    Sampler out(llama_sampler_chain_init(llama_sampler_chain_default_params()),llama_sampler_free);
    require(bool(out), "sampler allocation failed");
    llama_sampler_chain_add(out.get(),llama_sampler_init_penalties(vocab,int(request.value("penalty_last_n",64)),
        float(request.value("penalty_repeat",1)),float(request.value("penalty_freq",0)),float(request.value("penalty_present",0))));
    if (request.value("temperature",0) == 0) llama_sampler_chain_add(out.get(),llama_sampler_init_greedy());
    else {
        llama_sampler_chain_add(out.get(),llama_sampler_init_top_k(int(request.value("top_k",64))));
        llama_sampler_chain_add(out.get(),llama_sampler_init_top_p(float(request.value("top_p",1)),1));
        llama_sampler_chain_add(out.get(),llama_sampler_init_min_p(float(request.value("min_p",0)),1));
        llama_sampler_chain_add(out.get(),llama_sampler_init_temp(float(request.value("temperature",1))));
        llama_sampler_chain_add(out.get(),llama_sampler_init_dist(uint32_t(request.value("seed",LLAMA_DEFAULT_SEED))));
    }
    for (auto token : request.tokens) llama_sampler_accept(out.get(),token);
    return out;
}
struct Command { std::string line; std::atomic<bool> cancel{false}; explicit Command(std::string s):line(std::move(s)) {} };
static void execute(llama_context * ctx, const llama_vocab * vocab, const Options & options, Command & command) {
    struct CancelScope {
        CancelScope(const std::atomic<bool> & value) {strata_hy3_sync_cancel(&value);}
        ~CancelScope() {strata_hy3_sync_cancel(nullptr);}
    } cancel_scope(command.cancel);
    size_t prompt = 0;
    int generated = 0, decode_steps = 0;
    double prompt_ms = 0, decode_ms = 0;
    try {
        if (command.line.rfind("ENC ",0) == 0) {
            require(llama_vocab_type(vocab) != LLAMA_VOCAB_TYPE_NONE, "fixture has no text tokenizer");
            const auto input = parse_encode(command.line);
            int count = llama_tokenize(vocab,input.text.data(),int(input.text.size()),nullptr,0,false,input.parse_special);
            require(count <= 0,"unexpected tokenizer size result");
            std::vector<llama_token> ids(size_t(-count),0);
            if (count) require(llama_tokenize(vocab,input.text.data(),int(input.text.size()),ids.data(),int(ids.size()),false,input.parse_special) == int(ids.size()),"tokenizer failed");
            std::cout << format_ids(ids) << '\n' << std::flush; return;
        }
        const auto input = request(command.line,options.context,llama_vocab_n_tokens(vocab));
        prompt = input.tokens.size();
        auto smpl = sampler(input,llama_vocab_n_tokens(vocab));
        clear(ctx); strata_hy3_sync_reset();
        std::ofstream logits_file;
        if (!options.logits_file.empty()) {
            logits_file.open(options.logits_file,std::ios::binary | std::ios::trunc);
            require(bool(logits_file),"cannot open logits output");
        }

        auto start = Clock::now();
        for (size_t i = 0; i < prompt && !command.cancel.load(); i += options.batch) {
            const int count = int(std::min(size_t(options.batch),prompt-i));
            decode(ctx,input.tokens,i,count,int(i));
            std::cout << "PP " << i+count << " " << prompt << '\n' << std::flush;
        }
        prompt_ms = ms(start);
        const auto prefill_counters=strata_hy3_sync_snapshot();
        start = Clock::now();
        std::string finish = "length";
        for (int i = 0; i < input.count && !command.cancel.load(); ++i) {
            auto * logits = llama_get_logits_ith(ctx,-1);
            require(logits != nullptr,"missing Hy3 logits");
            for (int j = 0; j < llama_vocab_n_tokens(vocab); ++j)
                if (!std::isfinite(logits[j])) throw std::runtime_error("non-finite Hy3 logits");
            if (logits_file.is_open()) {
                logits_file.write(reinterpret_cast<const char *>(logits),llama_vocab_n_tokens(vocab)*sizeof(float));
                require(bool(logits_file),"logits write failed");
            }
            const auto token = llama_sampler_sample(smpl.get(),ctx,-1);
            ++generated; std::cout << "T " << token << '\n' << std::flush;
            if (llama_vocab_is_eog(vocab,token)) { finish = "stop"; break; }
            if (i+1 < input.count && !command.cancel.load()) {
                const auto step = Clock::now();
                decode(ctx,std::vector<llama_token>{token},0,1,int(prompt)+i);
                decode_ms += ms(step); ++decode_steps;
            }
        }
        const double wall_ms = ms(start);
        if (command.cancel.load()) finish = "cancel";
        const auto s = strata_hy3_sync_snapshot();
        clear(ctx);
        std::cerr << "STRATA_HY3_REQUEST " << json({{"prompt_tokens",prompt},{"generated",generated},
            {"prefill_ms",prompt_ms},{"generation_wall_ms",wall_ms},{"decode_steps",decode_steps},{"decode_forward_ms",decode_ms},
            {"source_bytes",s.source_bytes},{"h2d_bytes",s.h2d_bytes},{"source_ms",s.source_ms},{"h2d_ms",s.h2d_ms},
            {"prefill_h2d_bytes",prefill_counters.h2d_bytes},{"decode_h2d_bytes",s.h2d_bytes-prefill_counters.h2d_bytes},
            {"staging_bytes",s.staging_bytes},{"ranges",s.ranges},{"chunks",s.chunks},
            {"gpu_nodes",s.gpu_nodes},{"expert_nodes",s.expert_nodes},
            {"rejected_cpu_nodes",s.rejected_cpu_nodes},{"rejected_full_copies",s.rejected_full_copies},
            {"gpu_free",s.gpu_free},{"gpu_total",s.gpu_total},{"ram_free",s.ram_free},{"ram_total",s.ram_total},
            {"working_set_limit",s.working_set_limit},{"ram_target_percent",93},{"memory_ceiling_percent",95},
            {"finish",finish}}).dump() << '\n';
        std::cout << "DONE " << generated << " " << prompt << " " << prompt_ms << " " << wall_ms << " " << finish << " 0 0 0\n" << std::flush;
    } catch (const std::exception & error) {
        clear(ctx);
        if(command.cancel.load()) {
            std::cout << "DONE " << generated << " " << prompt << " " << prompt_ms << " 0 cancel 0 0 0\n" << std::flush;
            return;
        }
        std::cerr << "STRATA_HY3_ERROR " << error.what() << '\n';
        std::cout << "ERR invalid request or Hy3 execution failed\n" << std::flush;
    }
}
static void serve(llama_context * ctx, const llama_vocab * vocab, const Options & options) {
    std::mutex mutex; std::condition_variable cv;
    std::deque<std::shared_ptr<Command>> queue;
    std::shared_ptr<Command> active;
    bool ended = false;
    std::thread reader([&] {
        std::string line;
        while (std::getline(std::cin,line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line == "QUIT" || line.size() > 2*1024*1024+16) break;
            std::lock_guard lock(mutex);
            if (line == "STOP") {
                if (active) active->cancel.store(true);
                for (auto & pending : queue) pending->cancel.store(true);
            } else {
                if (queue.size() >= 16) break;
                queue.push_back(std::make_shared<Command>(std::move(line)));
            }
            cv.notify_one();
        }
        std::lock_guard lock(mutex); ended = true;
        if (active) active->cancel.store(true);
        cv.notify_one();
    });
    while (true) {
        {
            std::unique_lock lock(mutex); cv.wait(lock,[&] { return ended || !queue.empty(); });
            if (ended) break;
            active = queue.front(); queue.pop_front();
        }
        execute(ctx,vocab,options,*active);
        { std::lock_guard lock(mutex); active.reset(); }
    }
    reader.join();
}
int main(int argc,char ** argv) {
    if(argc==2 && std::string(argv[1])=="--version") {
        std::cout<<json({{"architecture","hy_v3"},{"engine","hy3-native"},{"protocol_version",1},
            {"source_sha",STRATA_HY3_SOURCE_SHA},{"patch_set",STRATA_HY3_PATCH_SET}}).dump()<<'\n';return 0;
    }
    try {
        const auto config=options(argc,argv);
        const auto admitted=inspect(config.model,config.fixture);
        if(config.inspect_only) {
            std::cout<<json({{"status","pass"},{"blocks",admitted.blocks},{"vocab",admitted.vocab},
                {"main_fixed_bytes",admitted.fixed_bytes},{"main_routed_bytes",admitted.routed_bytes},
                {"mtp_bytes",admitted.mtp_bytes},{"weight_allocations",false}}).dump()<<'\n';return 0;
        }
        environment();ggml_backend_load_all();strata_hy3_sync_mode(config.copy_mode);
        {
            auto model=load(config.model,false,false,config.fixture);
            auto ctx=context(model.get(),config.context,config.batch,config.kv);
            strata_hy3_memory_check();
            std::cout<<"INFO engine=hy3-native architecture=hy_v3 backend=llama.cpp gpu_only=1"
                <<" mtp=0 spec=0 speculative=none expert_storage=mmap expert_compute=gpu expert_pipeline=0"
                <<" expert_cache_mib=0 memory_target_percent=95 ram_target_percent=93 expert_readers=0 expert_stage_mib="<<(config.copy_mode==2 ? 16 : 0)
                <<" expert_copy="<<(config.copy_mode==2 ? "pinned-sync-file" : "native-reference")
                <<" kv=f32 flash_attention=0 tf32=0 cuda_fusion=0 conversation_cache=0\n"
                <<"READY "<<config.context<<" stop session-id\n"<<std::flush;
            serve(ctx.get(),llama_model_get_vocab(model.get()),config);
            strata_hy3_sync_release();
        }
        llama_backend_free();return 0;
    } catch(const std::exception & e) {
        strata_hy3_sync_release();llama_backend_free();std::cerr<<"strata-hy3: "<<e.what()<<'\n';return 1;
    }
}
