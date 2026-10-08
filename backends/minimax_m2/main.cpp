// Experimental synchronous baseline. JSONL pipe accepts already-rendered prompts or token IDs.
#include "runtime.hpp"
#include "nlohmann/json.hpp"
#include <chrono>
#include <csignal>
#include <fstream>
#include <iostream>
using namespace minimax_m2;
using json=nlohmann::ordered_json;
using Clock=std::chrono::steady_clock;
static std::atomic<bool> cancelled{false};
static void interrupt(int) {cancelled.store(true);}
static double elapsed(Clock::time_point start) {return std::chrono::duration<double,std::milli>(Clock::now()-start).count();}
static json stats() {
    const auto s=strata_mm27_snapshot();
    return {{"compute_calls",s.compute_calls},{"gpu_nodes",s.gpu_nodes},{"expert_nodes",s.expert_nodes},
        {"rejected_cpu_nodes",s.rejected_cpu_nodes},{"rejected_full_copies",s.rejected_full_copies},
        {"ranges",s.ranges},{"chunks",s.chunks},{"source_bytes",s.source_bytes},{"h2d_bytes",s.h2d_bytes},
        {"file_bytes",s.file_bytes},{"mmap_bytes",s.mmap_bytes},{"host_working_set_limit",s.host_working_set_limit},
        {"host_budget_updates",s.host_budget_updates},
        {"pipeline_plans",s.pipeline_plans},{"pipeline_matrices",s.pipeline_matrices},
        {"pipeline_plan_peak",s.pipeline_plan_peak},{"pipeline_lookahead_plans",s.pipeline_lookahead_plans},
        {"pipeline_copy_batches",s.pipeline_copy_batches},{"pipeline_copy_fences",s.pipeline_copy_fences},{"pipeline_scratch_fences",s.pipeline_scratch_fences},{"pipeline_copy_submissions",s.pipeline_copy_submissions},{"pipeline_pending_fills_peak",s.pipeline_pending_fills_peak},{"pipeline_abort_fences",s.pipeline_abort_fences},
        {"pipeline_groups",s.pipeline_groups},{"pipeline_chunks",s.pipeline_chunks},{"pipeline_h2d_bytes",s.pipeline_h2d_bytes},
        {"pipeline_d2d_bytes",s.pipeline_d2d_bytes},{"pipeline_unused_bytes",s.pipeline_unused_bytes},{"pipeline_device_bytes",s.pipeline_device_bytes},
        {"pipeline_queued_bytes",s.pipeline_queued_bytes},{"pipeline_reader_owned_bytes",s.pipeline_reader_owned_bytes},{"pipeline_read_peak",s.pipeline_read_peak},
        {"pipeline_wait_us",s.pipeline_wait_us},{"pipeline_slot_wait_us",s.pipeline_slot_wait_us},{"pipeline_read_us",s.pipeline_read_us},
        {"pipeline_submit_us",s.pipeline_submit_us},{"pipeline_delivery_ms",s.pipeline_delivery_ms},
        {"memory_checks",s.memory_checks},{"pressure_rejections",s.pressure_rejections},
        {"sampled_ram_used_peak",s.sampled_ram_used_peak},{"sampled_vram_used_peak",s.sampled_vram_used_peak},
        {"sampled_private_peak",s.sampled_private_peak},{"sampled_working_set_peak",s.sampled_working_set_peak},
        {"selected_bytes",s.selected_bytes},{"cache_hits",s.cache_hits},{"cache_misses",s.cache_misses},
        {"cache_hit_bytes",s.cache_hit_bytes},{"cache_fill_bytes",s.cache_fill_bytes},{"cache_guard_bytes",s.cache_guard_bytes},
        {"cache_evictions",s.cache_evictions},{"cache_reuses",s.cache_reuses},{"cache_oom",s.cache_oom},
        {"cache_bypasses",s.cache_bypasses},{"cache_resident",s.cache_resident},{"cache_limit",s.cache_limit},
        {"arena_reserved",s.arena_reserved},{"arena_live",s.arena_live},{"arena_blocks",s.arena_blocks},
        {"arena_allocations",s.arena_allocations},{"arena_frees",s.arena_frees},{"arena_rejects",s.arena_rejects},
        {"staging_bytes",s.staging_bytes},{"source_ms",s.source_ms},{"h2d_ms",s.h2d_ms},{"compute_ms",s.compute_ms}};
}
static json memory() {
    const auto m=strata_mm27_memory();
    return {{"ram_total",m.ram_total},{"ram_available",m.ram_available},{"vram_total",m.vram_total},
        {"vram_available",m.vram_available},{"process_private",m.process_private},{"process_working_set",m.process_working_set}};
}
static json request(llama_model *model,llama_context *ctx,const json &r,std::ostream *logits,bool stream) {
    const auto *vocab=llama_model_get_vocab(model);const int nv=llama_vocab_n_tokens(vocab);
    std::vector<llama_token> tokens;
    require(r.is_object() && r.contains("tokens")!=r.contains("prompt"),"provide exactly one of tokens or rendered prompt");
    if(r.contains("tokens")) {
        require(r.at("tokens").is_array() && r.at("tokens").size()<=llama_n_ctx(ctx),"invalid token array");
        for(const auto &t:r.at("tokens")) {
            require(t.is_number_integer() && t>=0 && t<nv,"invalid token ID");tokens.push_back(t.get<llama_token>());
        }
    }
    else {
        const auto prompt=r.at("prompt").get<std::string>();require(prompt.size()<=1u<<20,"prompt too large");
        const int n=llama_tokenize(vocab,prompt.data(),int(prompt.size()),nullptr,0,false,true);
        require(n<0,"empty prompt");tokens.resize(-n);
        require(llama_tokenize(vocab,prompt.data(),int(prompt.size()),tokens.data(),int(tokens.size()),false,true)==-n,"tokenization failed");
    }
    require(!r.contains("max_tokens") || (r.at("max_tokens").is_number_integer() &&
        r.at("max_tokens")>=1 && r.at("max_tokens")<=256),"invalid max_tokens");
    const int count=r.value("max_tokens",8);
    require(count>=1 && count<=256 && !tokens.empty() && tokens.size()+size_t(count)<=llama_n_ctx(ctx),"invalid request/context size");
    for(auto t:tokens)require(t>=0 && t<nv,"token outside vocabulary");
    clear(ctx);cancelled.store(false);strata_mm27_reset();strata_mm27_cache_decode(false);
    json samples=json::array();samples.push_back(memory());
    const auto start=Clock::now();
    for(size_t i=0;i<tokens.size();i+=llama_n_batch(ctx)) {
        decode(ctx,tokens,i,int(std::min<size_t>(llama_n_batch(ctx),tokens.size()-i)),int(i));samples.push_back(memory());
    }
    const double prefill_ms=elapsed(start);const auto prefill=stats();strata_mm27_reset();
    json generated=json::array();std::string text;double decode_ms=0,ttft_ms=0;bool eos=false;
    for(int i=0;i<count;++i) {
        const float *l=llama_get_logits_ith(ctx,-1);require(l,"missing logits");
        require(std::all_of(l,l+nv,[](float x){return std::isfinite(x);}),"non-finite logits");
        if(logits) {logits->write(reinterpret_cast<const char *>(l),nv*sizeof(float));require(bool(*logits),"cannot write logits");}
        const llama_token token=llama_token(std::max_element(l,l+nv)-l);generated.push_back(token);
        std::vector<char> piece(256);int n=llama_token_to_piece(vocab,token,piece.data(),int(piece.size()),0,true);
        if(n<0) {piece.resize(-n);n=llama_token_to_piece(vocab,token,piece.data(),int(piece.size()),0,true);}
        require(n>=0,"cannot decode token piece");text.append(piece.data(),size_t(n));
        if(i==0)ttft_ms=elapsed(start);
        if(stream)std::cout<<json({{"event","token"},{"id",token}}).dump()<<std::endl;
        eos=is_stop(token);if(eos || i+1==count)break;
        const auto step=Clock::now();decode(ctx,{token},0,1,int(tokens.size())+i);decode_ms+=elapsed(step);samples.push_back(memory());
        // Let the first serial decode finish allocating its workspace before
        // admitting experts. Existing hits remain usable during warmup/prefill.
        strata_mm27_cache_decode(true);
    }
    const double request_ms=elapsed(start);
    return {{"event","result"},{"prompt_tokens",tokens.size()},{"generated_tokens",generated.size()},{"token_ids",generated},
        {"text",text},{"stop_reason",eos?"eos":"length"},{"prefill_ms",prefill_ms},{"ttft_ms",ttft_ms},
        {"decode_forward_tokens",generated.size()-1},{"decode_ms",decode_ms},
        {"decode_tokens_per_second",decode_ms>0?1000.*(generated.size()-1)/decode_ms:0.},
        {"request_ms",request_ms},{"prefill",prefill},{"decode",stats()},{"memory_samples",samples}};
}
int main(int argc,char **argv) {
    try {
        std::string path,input,output,logit_path;int size=512,batch=8,mode=2;bool pipe=false;
        uint64_t cache_mib=0;
        std::string cache_allocator="cuda";
        std::string reader="file";
        int pipeline_readers=0,pipeline_chunk=8;bool pipeline_lookahead=false,pipeline_d2d_batch=false;
        std::string pipeline_trace;
        for(int i=1;i<argc;++i) {
            const std::string arg=argv[i];
            if(arg=="--version") {std::cout<<json({{"architecture","minimax-m2"},{"source_revision",STRATA_MM27_SOURCE_SHA},
                {"patches",STRATA_MM27_PATCH_SET}}).dump()<<'\n';return 0;}
            if(arg=="--pipe") {pipe=true;continue;}
            require(i+1<argc,"missing argument: "+arg);const std::string value=argv[++i];
            if(arg=="--gguf")path=value;else if(arg=="--request")input=value;else if(arg=="--output")output=value;
            else if(arg=="--logits")logit_path=value;else if(arg=="--ctx")size=std::stoi(value);
            else if(arg=="--batch")batch=std::stoi(value);else if(arg=="--mode")mode=std::stoi(value);
            else if(arg=="--gpu-cache-mib") {
                require(!value.empty() && value.find_first_not_of("0123456789")==std::string::npos,"invalid GPU cache MiB");
                cache_mib=std::stoull(value);require(cache_mib<=131072,"GPU cache MiB too large");
            }
            else if(arg=="--gpu-cache-allocator") {
                require(value=="cuda" || value=="arena","invalid cache allocator");cache_allocator=value;
            }
            else if(arg=="--expert-reader") {
                require(value=="file" || value=="mmap" || (value=="mmap-direct" || value=="mmap-decode"),"invalid expert reader");reader=value;
            }
            else if(arg=="--pipeline-readers") {require(value=="0" || value=="1" || value=="2","invalid pipeline readers");pipeline_readers=std::stoi(value);}
            else if(arg=="--pipeline-chunk-mib") {require(value=="4" || value=="8" || value=="16","invalid pipeline chunk");pipeline_chunk=std::stoi(value);}
            else if(arg=="--pipeline-lookahead") {require(value=="0" || value=="1","invalid pipeline chunk");pipeline_lookahead=value=="1";}
            else if(arg=="--pipeline-d2d-batch") {require(value=="0" || value=="1","invalid pipeline chunk");pipeline_d2d_batch=value=="1";}
            else if(arg=="--pipeline-trace")pipeline_trace=value;
            else throw std::runtime_error("unknown argument: "+arg);
        }
        require(!path.empty() && (pipe?input.empty() && output.empty() && logit_path.empty():!input.empty()),
            "usage: --gguf MODEL (--pipe | --request JSON [--output JSON] [--logits F32]) [--ctx 512] [--batch 8] [--mode 2] [--gpu-cache-mib 0]");
        require(size>=256 && size<=4096 && batch>=1 && batch<=16 && (mode==1 || mode==2),"invalid context/batch/mode");
        // Parse input and open outputs before loading weights.
        json r;std::ofstream out,logit;
        if(!pipe) {
            std::ifstream in(input);require(bool(in),"cannot read request");in>>r;
            if(!output.empty()) {out.open(output);require(bool(out),"cannot write report");}
            if(!logit_path.empty()) {logit.open(logit_path,std::ios::binary);require(bool(logit),"cannot write logits");}
        }
        environment();ggml_backend_load_all();strata_mm27_mode(mode);
        struct Release {~Release(){strata_mm27_release();}} release;
        strata_mm27_reader(reader=="file"?0:reader=="mmap"?1:reader=="mmap-direct"?2:3);
        strata_mm27_cache_configure(cache_mib<<20,cache_allocator=="arena");
        strata_mm27_pipeline(pipeline_readers,pipeline_chunk,pipeline_lookahead,pipeline_d2d_batch);
        if(!pipeline_trace.empty())strata_mm27_pipeline_trace(true);
        signal(SIGINT,interrupt);strata_mm27_cancel(&cancelled);
        const auto before=memory();const auto start=Clock::now();
        auto model=load(path);auto ctx=context(model.get(),size,batch);const double load_ms=elapsed(start);
        json header={{"architecture","minimax-m2"},{"source_revision",STRATA_MM27_SOURCE_SHA},{"patches",STRATA_MM27_PATCH_SET},
            {"model",path},{"mode",mode},{"context",size},{"batch",batch},{"kv","F32"},{"flash_attention",false},
            {"strict_f32",true},{"graphs",false},{"cache",cache_mib>0},{"gpu_cache_mib",cache_mib},
            {"gpu_cache_allocator",cache_allocator},
            {"expert_reader",reader},{"host_working_set_target_percent",reader=="file"?0:94},
            {"pipeline_readers",pipeline_readers},{"pipeline_chunk_mib",pipeline_chunk},{"pipeline_lookahead",pipeline_lookahead},{"pipeline_d2d_batch",pipeline_d2d_batch},
            {"pipeline_trace",pipeline_trace},
            {"cache_policy","decode admission after first serial step; prefill hits only; global 95% minus 256 MiB"},
            {"mtp",false},{"load_ms",load_ms},{"memory_before",before},{"memory_loaded",memory()}};
        if(pipe) {
            header["event"]="ready";std::cout<<header.dump()<<std::endl;std::string line;
            while(std::getline(std::cin,line))try {
                require(line.size()<=2u<<20,"pipe request too large");
                std::cout<<request(model.get(),ctx.get(),json::parse(line),nullptr,true).dump(-1,' ',false,json::error_handler_t::replace)<<std::endl;
            } catch(const std::exception &e) {
                clear(ctx.get());std::cout<<json({{"event","error"},{"message",e.what()}}).dump()<<std::endl;
            }
        } else {
            if(r.is_array()) {
                require(!r.empty() && r.size()<=32,"invalid request corpus");header["results"]=json::array();
                for(const auto &item:r)header["results"].push_back(request(model.get(),ctx.get(),item,logit.is_open()?&logit:nullptr,false));
            } else header["result"]=request(model.get(),ctx.get(),r,logit.is_open()?&logit:nullptr,false);
            const auto report=header.dump(2,' ',false,json::error_handler_t::replace);
            if(out.is_open())out<<report<<'\n';std::cout<<report<<'\n';
        }
        if(!pipeline_trace.empty())strata_mm27_pipeline_trace_write(pipeline_trace.c_str());
        return 0;
    } catch(const std::exception &e) {std::cerr<<"MiniMax: "<<e.what()<<'\n';return 2;}
}
