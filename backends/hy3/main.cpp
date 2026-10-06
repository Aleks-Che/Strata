// Hy3 pipe baseline: one model owner, fresh KV/sampler for every request.
#include "runtime.hpp"
#include "mtp.hpp"
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
    std::string model, logits_file, trace_file;
    std::string ram_cache_policy="frequency";
    std::string gpu_cache_policy="all";
    int context=2048, batch=17, copy_mode=2, cache_mib=0;
    int pipeline_readers=0, pipeline_chunk=4, trace_graphs=0;
    int pipeline_batch=0;
    int profile_delivery=0, mtp=0, ram_cache_mib=0;
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
        else if(key=="--expert-cache-mib") out.cache_mib=integer(value);
        else if(key=="--ram-cache-policy") out.ram_cache_policy=value;
        else if(key=="--gpu-cache-policy") out.gpu_cache_policy=value;
        else if(key=="--ram-cache-mib") {
            require(value=="auto" || (!value.empty() && value[0]!='-'),"RAM cache must be auto or 0..1048576 MiB");
            out.ram_cache_mib=value=="auto"?-1:integer(value);
        }
        else if(key=="--pipeline-readers") out.pipeline_readers=integer(value);
        else if(key=="--pipeline-chunk-mib") out.pipeline_chunk=integer(value);
        else if(key=="--pipeline-batch") out.pipeline_batch=integer(value);
        else if(key=="--mtp") out.mtp=integer(value);
        else if(key=="--profile-delivery") out.profile_delivery=integer(value);
        else if(key=="--trace-graphs") out.trace_graphs=integer(value);
        else if(key=="--trace-file") out.trace_file=value;
        else if(key=="--logits-file") out.logits_file=value;
        else if(key=="--copy-mode") {
            require(value=="native" || value=="pinned","copy-mode must be native or pinned");
            out.copy_mode=value=="native" ? 1 : 2;
        } else if(key=="--kv") {
            require(value=="f32","only validated F32 KV is admitted by this baseline");
        } else throw std::runtime_error("unsupported Hy3 option: "+key);
    }
    require(!out.model.empty() && out.context>=32 && out.context<=2048 && out.batch>=1 && out.batch<=32 && out.batch<=out.context,"invalid model/context/batch");
    require(out.mtp>=0 && out.mtp<=3 && (!out.mtp || out.batch>=out.mtp+1),"MTP requires depth 0..3 and batch >= depth+1");
    require(out.cache_mib>=0 && out.cache_mib<=16384 && (!out.cache_mib || out.copy_mode==2),"cache must be 0..16384 MiB and requires pinned copy mode");
    require(out.ram_cache_mib>=-1 && out.ram_cache_mib<=1048576 && (!out.ram_cache_mib || out.copy_mode==2),"RAM cache requires auto or 0..1048576 MiB and pinned copy mode");
    require(out.ram_cache_policy=="frequency" || out.ram_cache_policy=="lru","RAM cache policy must be frequency or lru");
    require(out.gpu_cache_policy=="all" || out.gpu_cache_policy=="decode","GPU cache policy must be all or decode");
    require(out.pipeline_readers>=0 && out.pipeline_readers<=2 &&
        (out.pipeline_chunk==4 || out.pipeline_chunk==8 || out.pipeline_chunk==16) &&
        (!out.pipeline_readers || (out.cache_mib && out.copy_mode==2)),"pipeline requires readers0..2, chunk4/8/16 MiB and pinned cache");
    require(out.trace_graphs>=0 && out.trace_graphs<=128 &&
        (out.trace_graphs>0)==!out.trace_file.empty() && (!out.trace_graphs || out.pipeline_readers),"trace requires pipeline, trace-file and graphs1..128");
    require((out.pipeline_batch==0 || out.pipeline_batch==1) && (!out.pipeline_batch || out.pipeline_readers),"pipeline-batch must be0/1 and requires pipeline readers");
    require((out.profile_delivery==0 || out.profile_delivery==1) && (!out.profile_delivery || out.pipeline_readers),"profile-delivery must be 0/1 and requires pipeline readers");
    if(!out.trace_file.empty()) {
        const auto p=std::filesystem::absolute(out.trace_file);
        require(p.extension()==".json" && !std::filesystem::exists(p),"trace output must be a new .json file");
    }
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
static json delivery_metrics(const strata_hy3_sync_stats & s,const strata_hy3_sync_stats & before={}) {
    return {{"source_read_cpu_ms",s.source_ms-before.source_ms},
        {"producer_wait_cpu_ms",double(s.pipeline_wait_us-before.pipeline_wait_us)/1000},
        {"scratch_wait_cpu_ms",s.pipeline_scratch_wait_ms-before.pipeline_scratch_wait_ms},
        {"delivery_wait_cpu_ms",s.pipeline_delivery_wait_ms-before.pipeline_delivery_wait_ms},
        {"cache_get_cpu_ms",s.cache_get_ms-before.cache_get_ms},
        {"cache_admit_cpu_ms",s.cache_admit_ms-before.cache_admit_ms},
        {"cache_victim_cpu_ms",s.cache_victim_ms-before.cache_victim_ms},
        {"cache_allocate_cpu_ms",s.cache_allocate_ms-before.cache_allocate_ms},
        {"cache_free_cpu_ms",s.cache_free_ms-before.cache_free_ms},
        {"cache_refresh_cpu_ms",s.cache_refresh_ms-before.cache_refresh_ms},
        {"cache_probe_cpu_ms",s.cache_probe_ms-before.cache_probe_ms},
        {"cache_protect_cpu_ms",s.cache_protect_ms-before.cache_protect_ms},
        {"cache_trim_cpu_ms",s.cache_trim_ms-before.cache_trim_ms},
        {"cache_allocations",s.cache_allocations-before.cache_allocations},
        {"cache_victim_candidates",s.cache_victim_candidates-before.cache_victim_candidates},
        {"cache_reuses",s.cache_reuses-before.cache_reuses},
        {"cache_fill_bytes",s.cache_fill_bytes-before.cache_fill_bytes}};
}
static void execute(llama_context * ctx, const llama_vocab * vocab, const Options & options, Command & command, Mtp * mtp) {
    struct CancelScope {
        CancelScope(const std::atomic<bool> & value) {strata_hy3_sync_cancel(&value);}
        ~CancelScope() {strata_hy3_sync_cancel(nullptr);}
    } cancel_scope(command.cancel);
    size_t prompt = 0;
    int generated = 0, decode_steps = 0, delivered_drafts = 0;
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
        clear(ctx); if(mtp) mtp->reset(); strata_hy3_sync_reset();
        // Stochastic requests retain the existing target-only sampler path.
        Mtp * active_mtp=input.value("temperature",0)==0 ? mtp : nullptr;
        std::ofstream logits_file;
        if (!options.logits_file.empty()) {
            logits_file.open(options.logits_file,std::ios::binary | std::ios::trunc);
            require(bool(logits_file),"cannot open logits output");
        }

        strata_hy3_cache_prefill(true);
        auto start = Clock::now();
        for (size_t i = 0; i < prompt && !command.cancel.load(); i += options.batch) {
            const int count = int(std::min(size_t(options.batch),prompt-i));
            decode(ctx,input.tokens,i,count,int(i));
            if(active_mtp) active_mtp->prefill(input.tokens,int(i),count,int(i));
            std::cout << "PP " << i+count << " " << prompt << '\n' << std::flush;
        }
        prompt_ms = ms(start);
        strata_hy3_cache_prefill(false);
        const auto prefill_counters=strata_hy3_sync_snapshot();
        start = Clock::now();
        std::string finish = "length";
        auto sample_at = [&](int row) {
            auto * logits = llama_get_logits_ith(ctx,row);
            require(logits != nullptr,"missing Hy3 logits");
            for (int j = 0; j < llama_vocab_n_tokens(vocab); ++j)
                if(!std::isfinite(logits[j])) throw std::runtime_error("non-finite Hy3 logits");
            if (logits_file.is_open()) {
                logits_file.write(reinterpret_cast<const char *>(logits),llama_vocab_n_tokens(vocab)*sizeof(float));
                require(bool(logits_file),"logits write failed");
            }
            return llama_sampler_sample(smpl.get(),ctx,row);
        };
        auto stop = [&](llama_token token) {return llama_vocab_is_eog(vocab,token);};
        auto cancelled = [&] {return command.cancel.load();};
        auto emit = [&](llama_token token) {
            ++generated; std::cout << "T " << token << '\n' << std::flush;
            if(stop(token)) finish="stop";
        };
        if(active_mtp && !cancelled()) {
            auto carry=sample_at(-1); emit(carry); int position=int(prompt);
            while(generated<input.count && !stop(carry) && !cancelled()) {
                auto round=active_mtp->advance(carry,position,input.count-generated,sample_at,stop,cancelled);
                if(round.cancelled || cancelled()) break;
                require(!round.tokens.empty(),"empty Hy3 MTP round");
                for(size_t i=0;i<round.tokens.size() && !cancelled();++i) {
                    emit(round.tokens[i]); if(int(i)<round.accepted) ++delivered_drafts;
                }
                carry=round.tokens.back(); position=round.next_position;
            }
            decode_steps=int(active_mtp->counters.rounds);
            decode_ms=active_mtp->counters.verify_ms;
        } else if(!active_mtp) {
            for(int i=0;i<input.count && !cancelled();++i) {
                auto token=sample_at(-1); emit(token);
                if(stop(token)) break;
                if(i+1<input.count && !cancelled()) {
                    const auto step=Clock::now();
                    decode(ctx,std::vector<llama_token>{token},0,1,int(prompt)+i);
                    decode_ms+=ms(step); ++decode_steps;
                }
            }
        }
        const double wall_ms = ms(start);
        if (command.cancel.load()) finish = "cancel";
        const auto s = strata_hy3_sync_snapshot();
        const auto mt=mtp ? mtp->counters : Mtp::Counters{};
        clear(ctx); if(mtp) mtp->reset();
        std::cerr << "STRATA_HY3_REQUEST " << json({{"prompt_tokens",prompt},{"generated",generated},
            {"mtp_depth",active_mtp ? options.mtp : 0},
            {"mtp",{{"proposed",mt.proposed},{"accepted",mt.accepted},{"delivered",delivered_drafts},{"rounds",mt.rounds},
                {"reject_first",mt.reject_first},{"reject_middle",mt.reject_middle},{"accept_all",mt.accept_all},
                {"prefill_ms",mt.prefill_ms},{"draft_ms",mt.draft_ms},{"verify_ms",mt.verify_ms},{"repair_ms",mt.repair_ms}}},
            {"prefill_ms",prompt_ms},{"generation_wall_ms",wall_ms},{"decode_steps",decode_steps},{"decode_forward_ms",decode_ms},
            {"source_bytes",s.source_bytes},{"h2d_bytes",s.h2d_bytes},{"source_ms",s.source_ms},{"h2d_ms",s.h2d_ms},
            {"prefill_h2d_bytes",prefill_counters.h2d_bytes},{"decode_h2d_bytes",s.h2d_bytes-prefill_counters.h2d_bytes},
            {"staging_bytes",s.staging_bytes},{"ranges",s.ranges},{"chunks",s.chunks},
            {"cache_bytes",s.cache_bytes},{"cache_budget",s.cache_budget},{"cache_entries",s.cache_entries},
            {"gpu_cache_policy",options.gpu_cache_policy},{"gpu_prefill_bypasses",s.cache_prefill_bypasses},
            {"prefill_gpu_fill_bytes",prefill_counters.cache_fill_bytes},
            {"decode_gpu_fill_bytes",s.cache_fill_bytes-prefill_counters.cache_fill_bytes},
            {"ram_cache",{{"bytes",s.ram_cache_bytes},{"budget",s.ram_cache_budget},{"entries",s.ram_cache_entries},
                {"peak_bytes",s.ram_cache_peak_bytes},{"pending_bytes",s.ram_cache_pending},
                {"hits",s.ram_cache_hits},{"misses",s.ram_cache_misses},{"hit_bytes",s.ram_cache_hit_bytes},
                {"file_bytes",s.ram_cache_file_bytes},{"fill_bytes",s.ram_cache_fill_bytes},
                {"evictions",s.ram_cache_evictions},{"rejected",s.ram_cache_rejected},{"oom",s.ram_cache_oom},
                {"allocations",s.ram_cache_allocations},{"reuses",s.ram_cache_reuses},
                {"policy",options.ram_cache_policy},{"prefill_bypasses",s.ram_cache_prefill_bypasses},
                {"frequency_bypasses",s.ram_cache_frequency_bypasses},{"victim_candidates",s.ram_cache_victim_candidates},
                {"history_entries",s.ram_cache_history_entries}}},
            {"ram_reused_payload_bytes",s.ram_cache_reused_payload_bytes},
            {"ram_unreused_payload_bytes",s.ram_cache_bytes-s.ram_cache_reused_payload_bytes},
            {"prefill_ram_fill_bytes",prefill_counters.ram_cache_fill_bytes},
            {"decode_ram_fill_bytes",s.ram_cache_fill_bytes-prefill_counters.ram_cache_fill_bytes},
            {"prefill_ram_hit_bytes",prefill_counters.ram_cache_hit_bytes},
            {"decode_ram_hit_bytes",s.ram_cache_hit_bytes-prefill_counters.ram_cache_hit_bytes},
            {"prefill_file_bytes",prefill_counters.ram_cache_file_bytes},
            {"decode_file_bytes",s.ram_cache_file_bytes-prefill_counters.ram_cache_file_bytes},
            {"cache_generation",s.cache_generation},{"cache_hits",s.cache_hits},{"cache_misses",s.cache_misses},
            {"cache_evictions",s.cache_evictions},{"cache_reuses",s.cache_reuses},{"cache_oom",s.cache_oom},
            {"cache_rejected",s.cache_rejected},{"cache_fill_bytes",s.cache_fill_bytes},{"d2d_bytes",s.d2d_bytes},
            {"pipeline_groups",s.pipeline_groups},{"pipeline_chunks",s.pipeline_chunks},
            {"pipeline_unused_bytes",s.pipeline_unused_bytes},{"pipeline_device_bytes",s.pipeline_device_bytes},
            {"pipeline_wait_us",s.pipeline_wait_us},{"pipeline_slot_wait_us",s.pipeline_slot_wait_us},
            {"pipeline_submit_us",s.pipeline_submit_us},{"pipeline_read_peak",s.pipeline_read_peak},
            {"pipeline_reader_owned",s.pipeline_reader_owned},{"pipeline_queued",s.pipeline_queued},
            {"pipeline_copy_fences",s.pipeline_copy_fences},
            {"pipeline_copy_batches",s.pipeline_copy_batches},{"pipeline_pending_fills_peak",s.pipeline_pending_fills_peak},
            {"pipeline_batch_ms",s.pipeline_batch_ms},
            {"profile_delivery",options.profile_delivery!=0},
            {"prefill_delivery",options.profile_delivery ? delivery_metrics(prefill_counters) : json(nullptr)},
            {"decode_delivery",options.profile_delivery ? delivery_metrics(s,prefill_counters) : json(nullptr)},
            {"prefill_copy_fences",prefill_counters.pipeline_copy_fences},
            {"decode_copy_fences",s.pipeline_copy_fences-prefill_counters.pipeline_copy_fences},
            {"decode_cache_hits",s.cache_hits-prefill_counters.cache_hits},{"decode_cache_misses",s.cache_misses-prefill_counters.cache_misses},
            {"gpu_nodes",s.gpu_nodes},{"expert_nodes",s.expert_nodes},
            {"rejected_cpu_nodes",s.rejected_cpu_nodes},{"rejected_full_copies",s.rejected_full_copies},
            {"gpu_free",s.gpu_free},{"gpu_total",s.gpu_total},{"ram_free",s.ram_free},{"ram_total",s.ram_total},
            {"working_set_limit",s.working_set_limit},{"ram_target_percent",93},{"memory_ceiling_percent",95},
            {"finish",finish}}).dump() << '\n';
        std::cout << "DONE " << generated << " " << prompt << " " << prompt_ms << " " << wall_ms << " " << finish << " " << delivered_drafts << " " << mt.proposed << " 0\n" << std::flush;
    } catch (const std::exception & error) {
        const auto mt=mtp ? mtp->counters : Mtp::Counters{};
        clear(ctx); if(mtp) mtp->reset();
        if(command.cancel.load()) {
            std::cout << "DONE " << generated << " " << prompt << " " << prompt_ms << " 0 cancel "
                << delivered_drafts << " " << mt.proposed << " 0\n" << std::flush;
            return;
        }
        std::cerr << "STRATA_HY3_ERROR " << error.what() << '\n';
        std::cout << "ERR invalid request or Hy3 execution failed\n" << std::flush;
    }
}
static void serve(llama_context * ctx, const llama_vocab * vocab, const Options & options, Mtp * mtp) {
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
        execute(ctx,vocab,options,*active,mtp);
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
            auto model=load(config.model,false,false,config.fixture,config.mtp>0);
            auto ctx=context(model.get(),config.context,config.batch,config.kv);
            std::unique_ptr<Mtp> mtp;
            if(config.mtp) mtp=std::make_unique<Mtp>(model.get(),ctx.get(),config.context,config.batch,config.mtp);
            configure_cache(model.get(),size_t(config.cache_mib)<<20);
            strata_hy3_gpu_cache_policy(config.gpu_cache_policy=="all");
            strata_hy3_ram_cache_config(config.ram_cache_mib<0?SIZE_MAX:size_t(config.ram_cache_mib)<<20,config.ram_cache_policy=="frequency");
            strata_hy3_delivery_profile(config.profile_delivery!=0);
            strata_hy3_pipeline_config(config.pipeline_readers,config.pipeline_chunk,config.trace_graphs,config.pipeline_batch!=0);
            strata_hy3_memory_check();
            std::cout<<"INFO engine=hy3-native architecture=hy_v3 backend=llama.cpp gpu_only=1"
                <<" mtp="<<(config.mtp>0)<<" spec="<<config.mtp<<" speculative="<<(config.mtp ? "mtp" : "none")
                <<" mtp_storage="<<(config.mtp ? "resident" : "none")<<" expert_storage=mmap expert_compute=gpu expert_pipeline="<<(config.pipeline_readers>0)
                <<" expert_cache_mib="<<config.cache_mib<<" ram_cache_mib="<<(config.ram_cache_mib<0?"auto":std::to_string(config.ram_cache_mib))
                <<" ram_cache_policy="<<config.ram_cache_policy
                <<" gpu_cache_policy="<<config.gpu_cache_policy
                <<" memory_target_percent=95 ram_target_percent=93 expert_readers="<<config.pipeline_readers
                <<" expert_stage_mib="<<(config.pipeline_readers ? 4*config.pipeline_chunk : config.copy_mode==2 ? 16 : 0)
                <<" expert_batch_copy="<<config.pipeline_batch
                <<" profile_delivery="<<config.profile_delivery
                <<" expert_copy="<<(config.pipeline_readers ? "pinned-pipeline-file" : config.copy_mode==2 ? "pinned-sync-file" : "native-reference")
                <<" kv=f32 flash_attention=0 tf32=0 cuda_fusion=0 cuda_graphs=0 conversation_cache=0\n"
                <<"READY "<<config.context<<" stop session-id\n"<<std::flush;
            serve(ctx.get(),llama_model_get_vocab(model.get()),config,mtp.get());
            strata_hy3_trace_write(config.trace_file.c_str());
            strata_hy3_sync_release();
        }
        llama_backend_free();return 0;
    } catch(const std::exception & e) {
        strata_hy3_sync_release();llama_backend_free();std::cerr<<"strata-hy3: "<<e.what()<<'\n';return 1;
    }
}
