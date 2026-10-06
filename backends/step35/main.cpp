// Step pipe baseline: one model owner, fresh KV/sampler for every request.
#include "runtime.hpp"
#include "cache_registry.hpp"
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

using namespace step35;
using Clock = std::chrono::steady_clock;
using json = nlohmann::ordered_json;
static double ms(Clock::time_point start) { return std::chrono::duration<double,std::milli>(Clock::now()-start).count(); }
struct Options {
    std::string model, logits_file, trace_file;
    int context = 2048, batch = 16, copy_mode = 2;
    ggml_type kv = GGML_TYPE_F32;
    size_t cache_cap=0;
    bool prefill_admission=true;
    bool cache_reuse=false,pipeline_batch=false;
    int readers=0,chunk_mib=8,trace_graphs=8;
};
static Options options(int argc, char ** argv) {
    Options out;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--serve") continue;
        require(i+1 < argc, "missing option value: " + key);
        const std::string value = argv[++i];
        if (key == "--native") out.model = value;
        else if (key == "--max-context") out.context = integer(value);
        else if (key == "--batch-size") out.batch = integer(value);
        else if (key == "--logits-file") out.logits_file = value;
        else if (key == "--expert-pipeline-readers") out.readers=integer(value);
        else if (key == "--expert-pipeline-chunk-mib") out.chunk_mib=integer(value);
        else if (key == "--trace-file") out.trace_file=value;
        else if (key == "--trace-graphs") out.trace_graphs=integer(value);
        else if (key == "--expert-cache-prefill") {
            require(value=="on" || value=="off","expert-cache-prefill must be on or off");
            out.prefill_admission=value=="on";
        }
        else if (key == "--expert-cache-reuse" || key == "--expert-pipeline-batch") {
            require(value=="on" || value=="off",key+" must be on or off");
            (key=="--expert-cache-reuse" ? out.cache_reuse : out.pipeline_batch)=value=="on";
        }
        else if (key == "--expert-cache-mib") {
            if (value=="auto") out.cache_cap=SIZE_MAX;
            else {const int mib=integer(value);require(mib>=0 && mib<=1048576,"invalid expert-cache-mib");out.cache_cap=size_t(mib)<<20;}
        }
        else if (key == "--copy-mode") {
            require(value == "native" || value == "pinned", "copy-mode must be native or pinned");
            out.copy_mode = value == "native" ? 1 : 2;
        } else if (key == "--kv") {
            require(value == "f32" || value == "f16", "kv must be f32 or f16");
            out.kv = value == "f32" ? GGML_TYPE_F32 : GGML_TYPE_F16;
        } else throw std::runtime_error("unsupported Step option: " + key);
    }
    require(!out.model.empty() && out.context >= 32 && out.context <= 32768 &&
        out.batch >= 1 && out.batch <= 256 && out.batch <= out.context, "invalid Step model/context/batch");
    require(!out.cache_cap || out.copy_mode==2,"expert cache requires pinned copy mode");
    require(out.prefill_admission || out.copy_mode==2,"prefill admission policy requires pinned copy mode");
    require(!out.cache_reuse || (out.cache_cap && out.copy_mode==2),"cache reuse requires a nonzero pinned expert cache");
    require(!out.pipeline_batch || out.readers>0,"pipeline batching requires pipeline readers");
    require(out.readers>=0 && out.readers<=2 && (!out.readers || out.copy_mode==2) &&
        (out.chunk_mib==4 || out.chunk_mib==8 || out.chunk_mib==16) && out.trace_graphs>=1 && out.trace_graphs<=128,"invalid Step pipeline/trace settings");
    if (!out.trace_file.empty()) {
        const auto output=std::filesystem::absolute(out.trace_file);
        require(output.extension()==".json","trace output must have .json extension");
        if (std::filesystem::exists(output)) for (const auto & entry:std::filesystem::directory_iterator(std::filesystem::absolute(out.model).parent_path()))
            if (entry.is_regular_file() && entry.path().extension()==".gguf")
                require(!std::filesystem::equivalent(output,entry.path()),"trace output aliases model input");
    }
    if (!out.logits_file.empty()) {
        const auto output = std::filesystem::absolute(out.logits_file);
        require(output.extension() == ".f32", "logits output must have .f32 extension");
        if (std::filesystem::exists(output)) {
            for (const auto & entry : std::filesystem::directory_iterator(std::filesystem::absolute(out.model).parent_path()))
                if (entry.is_regular_file() && entry.path().extension() == ".gguf")
                    require(!std::filesystem::equivalent(output,entry.path()), "logits output aliases model input");
        }
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
        clear(ctx); strata_step_sync_reset();
        std::ofstream logits_file;
        if (!options.logits_file.empty()) {
            logits_file.open(options.logits_file,std::ios::binary | std::ios::trunc);
            require(bool(logits_file),"cannot open logits output");
        }
        RequestPhase phase(1);
        auto start = Clock::now();
        for (size_t i = 0; i < prompt && !command.cancel.load(); i += options.batch) {
            const int count = int(std::min(size_t(options.batch),prompt-i));
            decode(ctx,input.tokens,i,count,int(i));
            std::cout << "PP " << i+count << " " << prompt << '\n' << std::flush;
        }
        prompt_ms = ms(start);
        const auto prefill_counters=strata_step_sync_snapshot();
        phase.decode();
        start = Clock::now();
        std::string finish = "length";
        for (int i = 0; i < input.count && !command.cancel.load(); ++i) {
            auto * logits = llama_get_logits_ith(ctx,-1);
            require(logits != nullptr,"missing Step logits");
            for (int j = 0; j < llama_vocab_n_tokens(vocab); ++j)
                if (!std::isfinite(logits[j])) throw std::runtime_error("non-finite Step logits");
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
        const auto s = strata_step_sync_snapshot();
        clear(ctx);
        std::cerr << "STRATA_STEP_REQUEST " << json({{"prompt_tokens",prompt},{"generated",generated},
            {"prefill_ms",prompt_ms},{"generation_wall_ms",wall_ms},{"decode_steps",decode_steps},{"decode_forward_ms",decode_ms},
            {"source_bytes",s.source_bytes},{"h2d_bytes",s.h2d_bytes},{"source_ms",s.source_ms},{"h2d_ms",s.h2d_ms},
            {"staging_bytes",s.staging_bytes},{"gpu_nodes",s.gpu_nodes},{"expert_nodes",s.expert_nodes},
            {"cache_hits",s.cache_hits},{"cache_misses",s.cache_misses},{"cache_evictions",s.cache_evictions},
            {"cache_bypasses",s.cache_bypasses},{"cache_oom",s.cache_oom},{"cache_rejected",s.cache_rejected},
            {"cache_allocations",s.cache_allocations},{"cache_reuses",s.cache_reuses},
            {"pipeline_copy_batches",s.pipeline_copy_batches},{"pipeline_copy_fences",s.pipeline_copy_fences},
            {"cache_bytes",s.cache_bytes},{"cache_limit",s.cache_limit},{"d2d_bytes",s.d2d_bytes},
            {"cache_fill_bytes",s.cache_fill_bytes},{"d2d_ms",s.d2d_ms},{"requested_bytes",s.requested_bytes},
            {"prefill_cache_fill_bytes",prefill_counters.cache_fill_bytes},{"decode_cache_fill_bytes",s.cache_fill_bytes-prefill_counters.cache_fill_bytes},
            {"prefill_admission_skips",s.prefill_admission_skips},{"prefill_admission_skip_bytes",s.prefill_admission_skip_bytes},
            {"gpu_free",s.gpu_free},{"gpu_total",s.gpu_total},{"ram_free",s.ram_free},{"ram_total",s.ram_total},{"memory_samples",s.memory_samples},
            {"prefill_source_bytes",prefill_counters.source_bytes},{"decode_source_bytes",s.source_bytes-prefill_counters.source_bytes},
            {"prefill_h2d_bytes",prefill_counters.h2d_bytes},{"decode_h2d_bytes",s.h2d_bytes-prefill_counters.h2d_bytes},
            {"prefill_cache_hits",prefill_counters.cache_hits},{"decode_cache_hits",s.cache_hits-prefill_counters.cache_hits},
            {"prefill_cache_misses",prefill_counters.cache_misses},{"decode_cache_misses",s.cache_misses-prefill_counters.cache_misses},
            {"pipeline_groups",s.pipeline_groups},{"pipeline_chunks",s.pipeline_chunks},{"pipeline_unused_bytes",s.pipeline_unused_bytes},
            {"pipeline_device_bytes",s.pipeline_device_bytes},{"pipeline_wait_us",s.pipeline_wait_us},
            {"pipeline_slot_wait_us",s.pipeline_slot_wait_us},{"pipeline_submit_us",s.pipeline_submit_us},{"pipeline_read_peak",s.pipeline_read_peak},
            {"rejected_cpu_nodes",s.rejected_cpu_nodes},{"finish",finish}}).dump() << '\n';
        std::cout << "DONE " << generated << " " << prompt << " " << prompt_ms << " " << wall_ms << " " << finish << " 0 0 0\n" << std::flush;
    } catch (const std::exception & error) {
        clear(ctx);
        std::cerr << "STRATA_STEP_ERROR " << error.what() << '\n';
        std::cout << "ERR invalid request or Step execution failed\n" << std::flush;
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
int main(int argc, char ** argv) {
    if (argc == 2 && std::string(argv[1]) == "--version") {
        std::cout << json({{"architecture","step35"},{"engine","step35-native"},{"protocol_version",1},
            {"source_sha",STRATA_STEP_SOURCE_SHA},{"patch_set",STRATA_STEP_PATCH_SET}}).dump() << '\n'; return 0;
    }
    try {
        const auto config = options(argc,argv);
        environment(); ggml_backend_load_all(); strata_step_sync_mode(config.copy_mode);
        strata_step_cache_begin(config.model.c_str(),0); // Global admission before fixed weights.
        {
            auto model = load(config.model);
            auto ctx = context(model.get(),config.context,config.batch,config.kv);
            register_cache(model.get(),config.model,config.cache_cap);
            strata_step_cache_prefill(config.prefill_admission);
            strata_step_cache_reuse(config.cache_reuse);
            strata_step_pipeline_config_ex(config.readers,config.chunk_mib,config.trace_file.empty()?0:config.trace_graphs,false,config.pipeline_batch);
            const auto memory=strata_step_sync_snapshot();
            const auto * vocab = llama_model_get_vocab(model.get());
            std::cout << "INFO engine=step35-native architecture=step35 backend=llama.cpp gpu_only=1"
                << " mtp=0 spec=0 speculative=none expert_storage=mmap expert_compute=gpu expert_pipeline=" << (config.readers>0)
                << " expert_readers=" << config.readers << " expert_chunk_mib=" << config.chunk_mib
                << " expert_cache_mib=" << (memory.cache_limit>>20) << " memory_target_percent=95"
                << " expert_cache_prefill=" << (config.prefill_admission?"on":"off")
                << " expert_cache_reuse=" << int(config.cache_reuse) << " expert_pipeline_batch=" << int(config.pipeline_batch)
                << " expert_stage_mib=" << (config.readers ? 4*config.chunk_mib : config.copy_mode == 2 ? 16 : 0)
                << " expert_copy=" << (config.readers ? "pinned-pipeline" : config.copy_mode == 2 ? "pinned-sync" : "native-reference")
                << " kv=" << ggml_type_name(config.kv) << " flash_attention=0 tf32=0 conversation_cache=0\n"
                << "READY " << config.context << " stop session-id\n" << std::flush;
            serve(ctx.get(),vocab,config);
            if (!config.trace_file.empty()) strata_step_trace_write(config.trace_file.c_str());
            strata_step_sync_release(); // Release cached data before model mappings.
        }
        strata_step_sync_release(); llama_backend_free(); return 0;
    } catch (const std::exception & error) {
        strata_step_sync_release(); llama_backend_free();
        std::cerr << "strata-step35: " << error.what() << '\n'; return 1;
    }
}
