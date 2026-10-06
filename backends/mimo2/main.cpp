// Fresh KV for each request. CPU handles tokenization, greedy sampling and I/O only.
#include "runtime.hpp"
#include "../step35/tokenizer_protocol.hpp"
#include "nlohmann/json.hpp"
#include <chrono>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
using namespace mimo2;
using json=nlohmann::ordered_json;
using Clock=std::chrono::steady_clock;
static double ms(Clock::time_point start) {return std::chrono::duration<double,std::milli>(Clock::now()-start).count();}
static int integer(const std::string &s) {size_t end=0;int v=std::stoi(s,&end);require(end==s.size(),"invalid integer");return v;}
struct Options {
    std::string model,logits,trace;int context=512,batch=8,mode=2,readers=1,chunk=8,trace_graphs=0;
    size_t cache=14ull<<30;bool prefill=false,mmap=true;
};
static Options options(int argc,char **argv) {
    Options out;bool cache_explicit=false,readers_explicit=false;
    for(int i=1;i<argc;++i) {
        const std::string key=argv[i];if(key=="--serve")continue;
        require(i+1<argc,"missing option value: "+key);const std::string value=argv[++i];
        if(key=="--native")out.model=value;
        else if(key=="--max-context")out.context=integer(value);
        else if(key=="--batch-size")out.batch=integer(value);
        else if(key=="--logits-file")out.logits=value;
        else if(key=="--expert-readers") {out.readers=integer(value);readers_explicit=true;}
        else if(key=="--expert-chunk-mib")out.chunk=integer(value);
        else if(key=="--trace-graphs")out.trace_graphs=integer(value);
        else if(key=="--trace-file")out.trace=value;
        else if(key=="--expert-cache-mib") {int n=integer(value);require(n>=0 && n<=32768,"invalid expert cache size");out.cache=size_t(n)<<20;cache_explicit=true;}
        else if(key=="--expert-cache-prefill") {require(value=="on" || value=="off","invalid prefill cache policy");out.prefill=value=="on";}
        else if(key=="--expert-reader") {require(value=="file" || value=="mmap","invalid expert reader");out.mmap=value=="mmap";}
        else if(key=="--copy-mode") {require(value=="native" || value=="pinned","invalid copy mode");out.mode=value=="native"?1:2;}
        else throw std::runtime_error("unsupported MiMo option: "+key);
    }
    require(!out.model.empty() && out.context>=256 && out.context<=4096 && out.batch>=1 && out.batch<=16,"invalid model/context/batch");
    if(out.mode==1 && !cache_explicit)out.cache=0;
    if(!readers_explicit && (out.mode==1 || !out.cache))out.readers=0;
    require(!out.cache || out.mode==2,"expert cache requires pinned copy mode");
    require(out.readers>=0 && out.readers<=2 && (out.chunk==4 || out.chunk==8 || out.chunk==16) &&
        out.trace_graphs>=0 && out.trace_graphs<=128 && (!out.readers || (out.mode==2 && out.cache)) &&
        (!out.trace_graphs || (out.readers && !out.trace.empty())),"invalid pipeline/trace configuration");
    if(!out.trace.empty())require(out.trace_graphs && std::filesystem::path(out.trace).extension()==".json" &&
        !std::filesystem::exists(out.trace) && out.trace!=out.logits,"trace requires a fresh .json output");
    if(!out.logits.empty()) {
        require(std::filesystem::path(out.logits).extension()==".f32","logits output requires .f32 extension");
        require(!std::filesystem::exists(out.logits),"logits output must not exist (protect input and prior results)");
    }
    return out;
}
struct Command {std::string line;std::atomic<bool> cancel{false};explicit Command(std::string s):line(std::move(s)) {}};
struct CancelScope {
    explicit CancelScope(const std::atomic<bool> &flag) {strata_mimo_cancel(&flag);}
    ~CancelScope() {strata_mimo_cancel(nullptr);}
};
static void execute(llama_context *ctx,const llama_vocab *vocab,const Options &o,Command &command) {
    CancelScope cancel(command.cancel);int generated=0,steps=0;size_t prompt=0;
    double prefill_ms=0,decode_ms=0,ttft_ms=0;auto start=Clock::now();auto generation=start;
    std::string finish="length";uint64_t prefill_h2d=0;
    strata_mimo_reset(); // Invalid requests must not inherit previous request counters.
    try {
        if(command.line.rfind("ENC ",0)==0) {
            require(llama_vocab_type(vocab)!=LLAMA_VOCAB_TYPE_NONE,"fixture has no tokenizer");
            const auto input=step35::parse_encode(command.line);
            int n=llama_tokenize(vocab,input.text.data(),int(input.text.size()),nullptr,0,false,input.parse_special);
            require(n<=0,"unexpected tokenize size");std::vector<llama_token> ids(size_t(-n),0);
            if(n)require(llama_tokenize(vocab,input.text.data(),int(input.text.size()),ids.data(),int(ids.size()),false,input.parse_special)==int(ids.size()),"tokenizer failed");
            std::cout<<step35::format_ids(ids)<<'\n'<<std::flush;return;
        }
        std::istringstream in(command.line);std::string verb,length,word,extra;
        require(bool(in>>verb>>length>>word) && verb=="GEN" && !(in>>extra),"expected GEN <count> <comma-separated-ids>");
        const int count=integer(length),nv=llama_vocab_n_tokens(vocab);
        require(count>=1 && count<=o.context && !word.empty() && word.back()!=',',"invalid generation length/ids");
        std::vector<llama_token> tokens;std::istringstream list(word);
        while(std::getline(list,word,',')) {
            int id=integer(word);require(id>=0 && id<nv,"token outside vocabulary");tokens.push_back(id);
            require(tokens.size()<=size_t(o.context),"prompt exceeds context");
        }
        prompt=tokens.size();require(prompt && prompt+count<=size_t(o.context),"prompt plus generation exceeds context");
        clear(ctx);strata_mimo_reset();strata_mimo_phase(true);start=Clock::now();
        for(size_t i=0;i<prompt && !command.cancel.load();i+=o.batch) {
            decode(ctx,tokens,i,int(std::min(size_t(o.batch),prompt-i)),int(i));
            std::cout<<"PP "<<std::min(i+o.batch,prompt)<<' '<<prompt<<'\n'<<std::flush;
        }
        prefill_ms=ms(start);prefill_h2d=strata_mimo_snapshot().h2d_bytes;generation=Clock::now();
        strata_mimo_phase(false);
        // Append one row per sampled token across requests; the invocation owns a fresh output file.
        std::ofstream logits;
        if(!o.logits.empty()) {logits.open(o.logits,std::ios::binary|std::ios::app);require(bool(logits),"cannot open logits output");}
        for(int i=0;i<count && !command.cancel.load();++i) {
            const auto *values=llama_get_logits_ith(ctx,-1);require(values,"missing logits");
            for(int j=0;j<nv;++j)require(std::isfinite(values[j]),"non-finite logits");
            if(logits.is_open()) {logits.write(reinterpret_cast<const char *>(values),nv*sizeof(float));require(bool(logits),"logits write failed");}
            const auto token=int(std::max_element(values,values+nv)-values);
            if(!generated)ttft_ms=ms(start);
            ++generated;std::cout<<"T "<<token<<'\n'<<std::flush;
            // The exported EOS is authoritative. PAD/FIM/native fallback EOG are not chat stops.
            if(is_stop(token)) {finish="stop";break;}
            if(i+1<count && !command.cancel.load()) {
                const auto step=Clock::now();decode(ctx,std::vector<llama_token>{token},0,1,int(prompt)+i);
                decode_ms+=ms(step);++steps;
            }
        }
        if(command.cancel.load())finish="cancel";
    } catch(const std::exception &e) {
        finish=command.cancel.load()?"cancel":"error";
        std::cerr<<"STRATA_MIMO_ERROR "<<e.what()<<'\n';
    }
    const double generation_ms=ms(generation),request_ms=ms(start);
    // Cancellation or partial decode never survives into the next request.
    clear(ctx);const auto s=strata_mimo_snapshot();
    std::cerr<<"STRATA_MIMO_REQUEST "<<json({{"prompt_tokens",prompt},{"generated",generated},{"finish",finish},
        {"prefill_ms",prefill_ms},{"ttft_ms",ttft_ms},{"generation_ms",generation_ms},{"request_ms",request_ms},
        {"decode_steps",steps},{"decode_forward_ms",decode_ms},{"decode_tokens_per_second",decode_ms?1000*steps/decode_ms:0},
        {"ranges",s.ranges},{"chunks",s.chunks},{"source_bytes",s.source_bytes},{"h2d_bytes",s.h2d_bytes},
        {"requested_bytes",s.requested_bytes},{"cache_hits",s.cache_hits},{"cache_misses",s.cache_misses},
        {"cache_hit_bytes",s.cache_hit_bytes},{"cache_fill_bytes",s.cache_fill_bytes},{"d2d_ms",s.d2d_ms},
        {"cache_bytes",s.cache_bytes},{"cache_payload_bytes",s.cache_payload_bytes},{"cache_limit",s.cache_limit},{"cache_evictions",s.cache_evictions},
        {"cache_allocations",s.cache_allocations},{"cache_reuses",s.cache_reuses},{"cache_bypasses",s.cache_bypasses},{"cache_oom",s.cache_oom},
        {"host_working_set_limit",s.host_working_set_limit},
        {"pipeline_groups",s.pipeline_groups},{"pipeline_chunks",s.pipeline_chunks},{"pipeline_device_bytes",s.pipeline_device_bytes},
        {"pipeline_unused_bytes",s.pipeline_unused_bytes},{"pipeline_delivered_bytes",s.pipeline_delivered_bytes},
        {"pipeline_read_peak",s.pipeline_read_peak},{"pipeline_reader_owned",s.pipeline_reader_owned},{"pipeline_queued",s.pipeline_queued},
        {"pipeline_consumer_wait_us",s.pipeline_consumer_wait_us},{"pipeline_slot_wait_us",s.pipeline_slot_wait_us},
        {"pipeline_submit_us",s.pipeline_submit_us},
        {"pipeline_file_bytes",s.pipeline_file_bytes},{"pipeline_mmap_bytes",s.pipeline_mmap_bytes},{"pipeline_d2d_bytes",s.pipeline_d2d_bytes},
        {"prefill_h2d_bytes",prefill_h2d},{"decode_h2d_bytes",s.h2d_bytes-prefill_h2d},
        {"source_ms",s.source_ms},{"h2d_ms",s.h2d_ms},{"staging_bytes",s.staging_bytes},
        {"gpu_nodes",s.gpu_nodes},{"expert_nodes",s.expert_nodes},{"rejected_cpu_nodes",s.rejected_cpu_nodes},
        {"rejected_full_copies",s.rejected_full_copies},{"gpu_free",s.gpu_free},{"gpu_total",s.gpu_total},
        {"ram_free",s.ram_free},{"ram_total",s.ram_total}}).dump()<<'\n';
    if(finish=="error")std::cout<<"ERR invalid request or MiMo execution failed\n";
    else std::cout<<"DONE "<<generated<<' '<<prompt<<' '<<prefill_ms<<' '<<generation_ms<<' '<<finish<<" 0 0 0\n";
    std::cout<<std::flush;
}
static void serve(llama_context *ctx,const llama_vocab *vocab,const Options &o) {
    std::mutex mutex;std::condition_variable cv;std::deque<std::shared_ptr<Command>> queue;
    std::shared_ptr<Command> active;bool ended=false;
    std::thread reader([&] {
        std::string line;
        while(std::getline(std::cin,line)) {
            if(!line.empty() && line.back()=='\r')line.pop_back();
            if(line=="QUIT" || line.size()>2*1024*1024+16)break;
            std::lock_guard lock(mutex);
            if(line=="STOP") {if(active)active->cancel.store(true);for(auto &p:queue)p->cancel.store(true);}
            else {if(queue.size()>=16)break;queue.push_back(std::make_shared<Command>(std::move(line)));}
            cv.notify_one();
        }
        std::lock_guard lock(mutex);ended=true;if(active)active->cancel.store(true);cv.notify_one();
    });
    while(true) {
        {std::unique_lock lock(mutex);cv.wait(lock,[&] {return ended || !queue.empty();});
         if(ended)break;active=queue.front();queue.pop_front();}
        execute(ctx,vocab,o,*active);
        {std::lock_guard lock(mutex);active.reset();}
    }
    reader.join();
}
int main(int argc,char **argv) {
    if(argc==2 && std::string(argv[1])=="--version") {
        std::cout<<json({{"architecture","mimo2"},{"engine","mimo2-native"},{"protocol_version",1},
            {"source_sha",STRATA_MIMO_SOURCE_SHA},{"patch_set",STRATA_MIMO_PATCH_SET}}).dump()<<'\n';return 0;
    }
    try {
        const auto o=options(argc,argv);environment();ggml_backend_load_all();strata_mimo_mode(o.mode);
        const auto start=Clock::now();
        {
#ifdef STRATA_MIMO_TEST_FIXTURE
            auto model=load(o.model,false,true);
#else
            auto model=load(o.model);
#endif
            auto ctx=context(model.get(),o.context,o.batch);
            strata_mimo_cache(o.cache);strata_mimo_cache_prefill(o.prefill);
            strata_mimo_reader(o.mode==2 && o.mmap);
            strata_mimo_pipeline_config(o.readers,o.chunk,o.trace_graphs);
            std::cerr<<"STRATA_MIMO_LOAD_MS "<<ms(start)<<'\n';
            std::cout<<"INFO engine=mimo2-native architecture=mimo2 gpu_only=1 text_only=1 mtp=0 spec=0"
                <<" expert_compute=gpu expert_storage=mmap expert_pipeline="<<bool(o.readers)<<" expert_cache_mib="<<(strata_mimo_snapshot().cache_limit>>20)
                <<" expert_cache_requested_mib="<<(o.cache>>20)<<" expert_cache_prefill="<<(o.prefill?"on":"off")
                <<" expert_copy="<<(o.readers?(o.mmap?"mmap-pipeline":"file-pipeline"):o.mode==2?(o.mmap?"mmap-cache-sync":"pinned-file-sync"):"native-reference")
                <<" expert_readers="<<o.readers<<" expert_chunk_mib="<<o.chunk
                <<" expert_stage_mib="<<(o.readers?4*o.chunk:o.mode==2 && !o.mmap?16:0)<<" memory_target_percent=95 kv=f32 flash_attention=1"
                <<" host_working_set_target_percent="<<(o.mode==2 && o.mmap?94:0)
                <<" tf32=0 cuda_graphs=0 conversation_cache=0 sampling=greedy add_bos=0 stop_ids=151645\n"
                <<"READY "<<o.context<<" stop\n"<<std::flush;
            serve(ctx.get(),llama_model_get_vocab(model.get()),o);strata_mimo_trace_write(o.trace.c_str());strata_mimo_release();
        }
        llama_backend_free();return 0;
    } catch(const std::exception &e) {
        strata_mimo_release();llama_backend_free();std::cerr<<"strata-mimo2: "<<e.what()<<'\n';return 1;
    }
}
