// Explicit P1 baseline runner. This is not the server protocol or an MTP engine.
#include "sync_runtime.h"
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
static void env(const char * key, const char * value) {
#ifdef _WIN32
    _putenv_s(key, value);
#else
    setenv(key, value, 1);
#endif
}
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
    json m={{"phase",phase},{"device_used_bytes_including_other_processes",total-free},{"device_total_bytes",total}};
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
    int predict=64, context=2048, batch=16;
    bool resident=false, cpu_embedding=false, candidate_copy=false;
};
static void decode(llama_context * ctx, const std::vector<llama_token> & ids, int start, int count, int position) {
    auto b=llama_batch_init(count,0,1); b.n_tokens=count;
    for (int i=0;i<count;++i) {
        b.token[i]=ids[start+i]; b.pos[i]=position+i; b.n_seq_id[i]=1; b.seq_id[i][0]=0; b.logits[i]=i==count-1;
    }
    const int rc=llama_decode(ctx,b); llama_batch_free(b);
    require(rc==0,"llama_decode failed: "+std::to_string(rc));
    llama_synchronize(ctx);
}
static void run(const Options & o, json & report, std::vector<float> * capture=nullptr) {
    strata_glm_sync_enable(true); strata_glm_sync_candidate_copy(o.candidate_copy); strata_glm_sync_reset();
    report["configuration"]={{"model",o.model},{"resident_experts",o.resident},{"n_ctx",o.context},
        {"n_batch",o.batch},{"n_ubatch",o.batch},{"n_predict",o.predict},{"kv_type","F16"},
        {"flash_attention","off"},{"mtp",false},{"NVIDIA_TF32_OVERRIDE","0"},
        {"GGML_OP_OFFLOAD_MIN_BATCH",1},{"sampling","greedy"},{"expert_cache",false},
        {"transfer",o.candidate_copy ? "candidate selected-range reference (not instrumented)" : "synchronous 16 MiB pinned staging"}};
    auto & samples=report["memory_samples"]; samples=json::array(); samples.push_back(memory("before_load"));
    auto * gpu=ggml_backend_dev_by_name("CUDA0"); auto * cpu=ggml_backend_dev_by_name("CPU");
    require(gpu && cpu,"CUDA0 and CPU buffer backends required");
    auto mp=llama_model_default_params();
    ggml_backend_dev_t devices[]={gpu,nullptr}; mp.devices=devices;
    llama_model_tensor_buft_override overrides[]={
        {"token_embd\\.weight",ggml_backend_dev_buffer_type(o.cpu_embedding ? cpu : gpu)},
        {"blk\\.[0-9]+\\.ffn_(gate|up|down)_exps\\.weight",ggml_backend_dev_buffer_type(cpu)},
        {nullptr,nullptr}};
    if (o.resident) overrides[1]={nullptr,nullptr};
    mp.tensor_buft_overrides=overrides; mp.n_gpu_layers=-1; mp.split_mode=LLAMA_SPLIT_MODE_NONE;
    mp.load_mode=LLAMA_LOAD_MODE_MMAP; mp.load_mtp=false; mp.use_extra_bufts=false;
    auto start=Clock::now(); report["phase"]="load";
    Model model(llama_model_load_from_file(o.model.c_str(),mp),llama_model_free);
    require(bool(model),"model load failed"); report["load_seconds"]=elapsed(start);
    char arch[64]{}; llama_model_meta_val_str(model.get(),"general.architecture",arch,sizeof(arch));
    require(std::string(arch)=="glm5next","runner accepts glm5next only");
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
    auto cp=llama_context_default_params();
    cp.n_ctx=o.context; cp.n_batch=cp.n_ubatch=o.batch; cp.n_seq_max=1; cp.n_rs_seq=0;
    cp.n_threads=cp.n_threads_batch=4; cp.type_k=cp.type_v=GGML_TYPE_F16;
    cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED; cp.offload_kqv=cp.op_offload=true; cp.no_perf=false;
    report["phase"]="context"; start=Clock::now();
    Context ctx(llama_init_from_model(model.get(),cp),llama_free);
    require(bool(ctx),"context allocation failed"); report["context_seconds"]=elapsed(start);
    samples.push_back(memory("after_context"));
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
    }
    std::cerr << "\n";
    report["generated_ids"]=generated; report["generated_tokens"]=generated.size(); report["output_text"]=output;
    report["decode_steps"]=decode_steps; report["decode_seconds"]=decode_seconds;
    report["decode_tokens_per_second"]=decode_steps ? decode_steps/decode_seconds : 0;
    report["decode_timing_definition"]="single-token llama_decode plus synchronize; excludes first token from prefill, sampling, memory queries and output I/O";
    report["decode_transport"]=counters(); report["all_logits_finite"]=true;
    if (!report.contains("stop_reason")) report["stop_reason"]="length";
    samples.push_back(memory("after_decode"));
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
    json report={{"status","error"},{"scope","P1 synchronous CLI baseline; no server, MTP or expert cache"},
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
            else if (arg=="--input-ids") { std::istringstream in(value); std::string id; while(std::getline(in,id,',')) o.ids.push_back(std::stoi(id)); }
            else throw std::runtime_error("unknown option "+arg);
        }
        require(o.predict>0 && o.context>0 && o.batch>0,"positive predict/context/batch required");
        env("NVIDIA_TF32_OVERRIDE","0"); env("GGML_OP_OFFLOAD_MIN_BATCH","1");
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
