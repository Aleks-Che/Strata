// Explicit P1 baseline runner. This is not the server protocol or an MTP engine.
#include "sync_runtime.h"
#include "runtime.hpp"
#include "runtime_memory.hpp"
#include "host_pages.hpp"
#include "synthetic_glm.hpp"
#include "llama.h"
#include "ggml-backend.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#endif

using json = nlohmann::json;
using Clock = std::chrono::steady_clock;
using Model = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
using Context = std::unique_ptr<llama_context, decltype(&llama_free)>;
static void require(bool ok, const std::string & message) { if (!ok) throw std::runtime_error(message); }
static double elapsed(Clock::time_point start) { return std::chrono::duration<double>(Clock::now()-start).count(); }
static std::string read(const std::string & path) {
    std::ifstream in(path, std::ios::binary); require(bool(in), "cannot open "+path);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}
static json counters() {
    auto s = strata_glm_sync_snapshot();
    return {{"source_bytes",s.source_bytes},{"h2d_bytes",s.h2d_bytes},{"ranges",s.ranges},{"chunks",s.chunks},
        {"staging_bytes",s.staging_bytes},{"source_seconds",s.source_ms/1000},{"h2d_seconds",s.h2d_ms/1000},
        {"compute_calls",s.compute_calls},{"gpu_compute_nodes",s.gpu_nodes},
        {"gpu_expert_matmuls",s.expert_nodes},{"rejected_cpu_nodes",s.rejected_cpu_nodes}};
}
static json memory(const char * phase) {
    size_t free=0,total=0; ggml_backend_dev_memory(ggml_backend_dev_by_name("CUDA0"), &free, &total);
    json m={{"phase",phase},{"cuda_view_used_bytes",total-free},{"cuda_view_total_bytes",total}};
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX p{}; p.cb=sizeof(p);
    require(GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS *>(&p),sizeof(p))!=0,"memory query failed");
    m["working_set_bytes"]=p.WorkingSetSize; m["peak_working_set_bytes"]=p.PeakWorkingSetSize;
    m["private_commit_bytes"]=p.PrivateUsage;
    MEMORYSTATUSEX s{}; s.dwLength=sizeof(s); require(GlobalMemoryStatusEx(&s)!=0,"RAM query failed");
    m["available_system_ram_bytes"]=s.ullAvailPhys;
#endif
    return m;
}
struct Options {
    std::string model, prompt_file, report, logits_file;
    std::vector<llama_token> ids;
    int predict=64, context=2048, batch=16, ram_percent=0, vram_percent=0;
    bool resident=false, cpu_embedding=false, candidate_copy=false, pipeline=false;
    int chunk_mib=4;
};
using strata_glm::decode;
static void run(const Options & o, json & report, std::vector<float> * capture=nullptr) {
    strata_glm_sync_enable(true); strata_glm_sync_candidate_copy(o.candidate_copy); strata_glm_sync_reset();
    report["configuration"]={{"model",o.model},{"resident_experts",o.resident},{"n_ctx",o.context},
        {"n_batch",o.batch},{"n_ubatch",o.batch},{"n_predict",o.predict},{"kv_type","F16"},
        {"flash_attention","off"},{"mtp",false},{"NVIDIA_TF32_OVERRIDE","0"},
        {"GGML_OP_OFFLOAD_MIN_BATCH",1},{"sampling","greedy"},{"expert_cache",o.vram_percent!=0},
        {"ram_target_percent",o.ram_percent},{"vram_target_percent",o.vram_percent},
        {"expert_pipeline",o.pipeline},{"expert_chunk_mib",o.chunk_mib},
        {"transfer",o.candidate_copy ? "candidate selected-range reference (not instrumented)" : o.vram_percent ? "synchronous cache plus 16 MiB pinned staging" : "synchronous 16 MiB pinned staging"}};
    auto & samples=report["memory_samples"]; samples=json::array(); samples.push_back(memory("before_load"));
    auto start=Clock::now(); report["phase"]="load";
    strata_glm::HostWorkingSetBudget load_budget;
    if(strata_glm::ram_experts())load_budget.apply(o.ram_percent);
    auto model=strata_glm::load(o.model,o.resident,o.cpu_embedding);
    report["load_seconds"]=elapsed(start);
    samples.push_back(memory("after_load"));
    std::cerr << "STRATA_GLM loaded in " << report["load_seconds"] << " s\n";
    const auto * vocab=llama_model_get_vocab(model.get());
    const int n_vocab=llama_vocab_n_tokens(vocab);
    auto ids=o.ids;
    if (ids.empty()) {
        const auto prompt=read(o.prompt_file);
        int count=llama_tokenize(vocab,prompt.data(),int(prompt.size()),nullptr,0,false,true);
        require(count<0,"empty/invalid prompt"); ids.resize(-count);
        require(llama_tokenize(vocab,prompt.data(),int(prompt.size()),ids.data(),int(ids.size()),false,true)==int(ids.size()),"tokenization failed");
    }
    require(!ids.empty() && ids.size()+o.predict<=size_t(o.context),"prompt plus generation must fit context");
    for (auto id:ids) require(id>=0 && id<n_vocab,"input token outside vocabulary");
    report["prompt_ids"]=ids; report["prompt_tokens"]=ids.size();
    report["phase"]="context"; start=Clock::now();
    auto ctx=strata_glm::context(model.get(),o.context,o.batch);
    report["context_seconds"]=elapsed(start);
    samples.push_back(memory("after_context"));
    strata_glm::RuntimeMemory runtime_memory(model,o.model,o.ram_percent,o.vram_percent,o.pipeline,o.chunk_mib);
    runtime_memory.warm();
    report["runtime_memory_after_warm"]=runtime_memory.snapshot();
    report["global_memory_samples"]=json::array();
    std::cerr<<"STRATA_GLM_MEMORY "<<report["runtime_memory_after_warm"].dump()<<"\n";
    report["phase"]="prefill"; strata_glm_sync_reset(); start=Clock::now();
    for (size_t i=0;i<ids.size();i+=o.batch) {
        const int count=std::min(size_t(o.batch),ids.size()-i);
        decode(ctx.get(),ids,int(i),count,int(i));
        std::cerr << "STRATA_GLM prefill " << i+count << "/" << ids.size() << "\n";
    }
    const double prefill=elapsed(start); report["prefill_seconds"]=prefill;
    report["prefill_tokens_per_second"]=ids.size()/prefill; report["prefill_transport"]=counters();
    samples.push_back(memory("after_prefill"));
    std::ofstream logits_file;
    if (!o.logits_file.empty()) { logits_file.open(o.logits_file,std::ios::binary); require(bool(logits_file),"cannot create logits file"); }
    report["logits_format"]={{"vocabulary",n_vocab},{"dtype","float32_le"},{"layout","one vocabulary row per sampled token"}};
    std::vector<llama_token> generated;
    std::string output;
    double decode_seconds=0; int decode_steps=0;
    report["phase"]="decode"; strata_glm_sync_reset();
    for (int i=0;i<o.predict;++i) {
        auto * logits=llama_get_logits_ith(ctx.get(),-1); require(logits!=nullptr,"missing logits");
        for (int j=0;j<n_vocab;++j) require(std::isfinite(logits[j]),"non-finite logits at token "+std::to_string(i));
        if (capture) capture->insert(capture->end(),logits,logits+n_vocab);
        if (logits_file.is_open()) { logits_file.write(reinterpret_cast<const char *>(logits),n_vocab*sizeof(float)); require(bool(logits_file),"logits write failed"); }
        const auto token=llama_token(std::max_element(logits,logits+n_vocab)-logits);
        generated.push_back(token);
        if (!capture) {
            std::vector<char> piece(256); int n=llama_token_to_piece(vocab,token,piece.data(),int(piece.size()),0,true);
            if (n<0) { piece.resize(-n); n=llama_token_to_piece(vocab,token,piece.data(),int(piece.size()),0,true); }
            require(n>=0,"detokenization failed"); output.append(piece.data(),n);
            std::cerr.write(piece.data(),n); std::cerr.flush();
        }
        if (!capture && llama_vocab_is_eog(vocab,token)) { report["stop_reason"]="eog"; break; }
        if (i+1==o.predict) break;
        start=Clock::now(); decode(ctx.get(),generated,i,1,int(ids.size())+i);
        decode_seconds+=elapsed(start); ++decode_steps;
        samples.push_back(memory("decode"));
        report["global_memory_samples"].push_back(runtime_memory.snapshot());
    }
    std::cerr << "\n";
    report["generated_ids"]=generated; report["generated_tokens"]=generated.size(); report["output_text"]=output;
    report["decode_steps"]=decode_steps; report["decode_seconds"]=decode_seconds;
    report["decode_tokens_per_second"]=decode_steps ? decode_steps/decode_seconds : 0;
    report["decode_timing_definition"]="single-token llama_decode plus synchronize; excludes first token from prefill, sampling, memory queries and output I/O";
    report["decode_transport"]=counters(); report["all_logits_finite"]=true;
    if (!report.contains("stop_reason")) report["stop_reason"]="length";
    samples.push_back(memory("after_decode"));
    report["runtime_memory_after_decode"]=runtime_memory.snapshot();
    report["status"]="pass"; report["phase"]="complete";
    std::cerr << "STRATA_GLM decode " << report["decode_tokens_per_second"] << " tok/s (" << decode_steps << " steps)\n";
}
static json self_test(const std::string & directory) {
    std::filesystem::create_directories(directory);
    json tests=json::array();
    for (int top_k : {8,512}) {
        Options o; o.model=directory+"/stream-"+std::to_string(top_k)+".gguf";
        o.context=512; o.batch=16; o.predict=8;
        for (int i=0;i<33;++i) o.ids.push_back((7*i+11)%64);
        write_synthetic_glm(o.model,top_k);
        json resident, streamed; std::vector<float> a,b;
        o.resident=true; run(o,resident,&a);
        o.resident=false; run(o,streamed,&b);
        require(a.size()==b.size(),"logit size mismatch");
        double error=0,energy=0,max_abs=0;
        for (size_t i=0;i<a.size();++i) { const double d=double(a[i])-b[i]; error+=d*d; energy+=double(a[i])*a[i]; max_abs=std::max(max_abs,std::abs(d)); }
        auto s=strata_glm_sync_snapshot();
        require(s.ranges>0 && s.source_bytes==s.h2d_bytes && s.staging_bytes<=16*1024*1024 && s.rejected_cpu_nodes==0,"streaming counters invalid");
        require(max_abs<=5e-4 && error/std::max(energy,1e-30)<=1e-7,"streamed/resident logits mismatch");
        require(resident["generated_ids"]==streamed["generated_ids"],"streamed/resident tokens differ");
        tests.push_back({{"top_k",top_k},{"status","pass"},{"elements",a.size()},
            {"max_abs",max_abs},{"nmse",error/std::max(energy,1e-30)},{"resident",resident},{"streamed",streamed}});
        json reference; std::vector<float> c; o.candidate_copy=true; run(o,reference,&c);
        require(b==c && reference["generated_ids"]==streamed["generated_ids"],"candidate selected-copy parity failed");
        tests.push_back({{"name","candidate selected-copy parity"},{"top_k",top_k},{"status","pass"},{"elements",c.size()}});
        json cached; std::vector<float> d; o.candidate_copy=false; o.vram_percent=90;
        run(o,cached,&d);
        require(b==d && cached["generated_ids"]==streamed["generated_ids"],"cached selected-copy parity failed");
        require(cached["runtime_memory_after_decode"]["cache"]["hits"].get<uint64_t>()>0,"cache did not serve hits");
        tests.push_back({{"name","cached selected-copy exact logits parity"},{"top_k",top_k},{"status","pass"},{"cached",cached}});
        for (int budget:{0,90}) {
            json piped; std::vector<float> e; o.pipeline=true; o.vram_percent=budget;
            run(o,piped,&e);
            require(b==e && piped["generated_ids"]==streamed["generated_ids"],"pipeline logits parity failed");
            if (!budget) require(piped["runtime_memory_after_decode"]["pipeline"]["chunks"].get<uint64_t>()>0,"pipeline transferred no misses");
            require(piped["runtime_memory_after_decode"]["pipeline"]["unused_bytes"]==0,"pipeline abandoned selected expert bytes");
            tests.push_back({{"name","router lookahead pipeline exact logits parity"},{"top_k",top_k},{"cache_percent",budget},{"status","pass"},{"pipeline",piped}});
        }
    }
    // Deliberately put embeddings on CPU: the complete split audit must reject
    // the graph before any expert transfer or computation, not merely log it.
    Options bad; bad.model=directory+"/stream-8.gguf"; bad.context=512; bad.ids={11}; bad.predict=1;
    bad.resident=true; bad.cpu_embedding=true; json result; std::vector<float> unused;
    bool rejected=false;
    try { run(bad,result,&unused); } catch (const std::exception &) {
        auto s=strata_glm_sync_snapshot(); rejected=s.rejected_cpu_nodes==1 && s.compute_calls==0 && s.source_bytes==0;
    }
    require(rejected,"CPU embedding graph was not rejected before computation");
    tests.push_back({{"name","CPU embedding rejected before computation"},{"status","pass"},{"audit",counters()}});
    return {{"status","pass"},{"scope","synthetic streamed/resident F16-KV parity and negative GPU audit"},{"checks",tests}};
}
int main(int argc,char ** argv) {
    json report={{"status","error"},{"scope","streamed CLI validation with optional expert cache/pipeline; MTP off"},
        {"requested_revision",STRATA_GLM_SOURCE_SHA},{"archive_sha256",STRATA_GLM_ARCHIVE_SHA256},{"patch_set",STRATA_GLM_PATCH_SET}};
    Options o; int exit_code=1;
    try {
        std::string test_dir;
        for (int i=1;i<argc;++i) {
            const std::string arg=argv[i];
            if (arg=="--resident-experts") { o.resident=true; continue; }
            if (arg=="--candidate-copy") { o.candidate_copy=true; continue; }
            require(i+1<argc,"missing value for "+arg); const std::string value=argv[++i];
            if (arg=="--model") o.model=value;
            else if (arg=="--prompt-file") o.prompt_file=value;
            else if (arg=="--report") o.report=value;
            else if (arg=="--logits") o.logits_file=value;
            else if (arg=="--self-test") test_dir=value;
            else if (arg=="--n-predict") o.predict=std::stoi(value);
            else if (arg=="--ctx") o.context=std::stoi(value);
            else if (arg=="--batch") o.batch=std::stoi(value);
            else if (arg=="--ram-target-percent") o.ram_percent=std::stoi(value);
            else if (arg=="--vram-target-percent") o.vram_percent=std::stoi(value);
            else if (arg=="--expert-pipeline") {const int n=std::stoi(value); require(n==0 || n==1,"pipeline must be 0/1");o.pipeline=n!=0;}
            else if (arg=="--expert-chunk-mib") o.chunk_mib=std::stoi(value);
            else if (arg=="--input-ids") { std::istringstream in(value); std::string id; while(std::getline(in,id,',')) o.ids.push_back(std::stoi(id)); }
            else throw std::runtime_error("unknown option "+arg);
        }
        require(o.predict>0 && o.context>0 && o.batch>0,"positive predict/context/batch required");
        strata_glm::environment();
        ggml_backend_load_all(); llama_backend_init();
        require(ggml_backend_dev_by_name("CUDA0")!=nullptr,"CUDA0 missing");
        if (!test_dir.empty()) report["validation"]=self_test(test_dir);
        else { require(!o.model.empty() && (!o.prompt_file.empty() || !o.ids.empty()),"--model and --prompt-file (or --input-ids) required"); run(o,report); }
        report["status"]="pass"; exit_code=0;
    } catch (const std::exception & e) { report["error"]=e.what(); report["last_counters"]=counters(); std::cerr << "STRATA_GLM error: " << e.what() << "\n"; }
    strata_glm_sync_release(); llama_backend_free();
    const auto output=report.dump(2,' ',false,json::error_handler_t::replace)+"\n";
    if (!o.report.empty()) { std::ofstream out(o.report,std::ios::binary); out << output; if(!out) { std::cerr << "cannot write report\n"; return 1; } }
    else std::cout << output;
    return exit_code;
}
