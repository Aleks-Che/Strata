// Experimental synchronous baseline. JSONL pipe accepts already-rendered prompts or token IDs.
#include "runtime.hpp"
#include "sampling.hpp"
#include "prefix.hpp"
#include "session_runtime.hpp"
#include "nlohmann/json.hpp"
#include <chrono>
#include <csignal>
#include <fstream>
#include <iostream>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif
using namespace minimax_m2;
using json=nlohmann::ordered_json;
using Clock=std::chrono::steady_clock;
static std::atomic<bool> cancelled{false};
#ifdef _WIN32
// MSVC signal(SIGBREAK, ...) is reset to SIG_DFL after its first delivery.
// A resident JSONL server needs a persistent handler for every request.
static BOOL WINAPI console_interrupt(DWORD event) {
    if(event!=CTRL_BREAK_EVENT && event!=CTRL_C_EVENT)return FALSE;
    cancelled.store(true);return TRUE;
}
#else
static void interrupt(int) {cancelled.store(true);}
#endif
static double elapsed(Clock::time_point start) {return std::chrono::duration<double,std::milli>(Clock::now()-start).count();}
static json stats() {
    const auto s=strata_mm27_snapshot();
    return {{"compute_calls",s.compute_calls},{"gpu_nodes",s.gpu_nodes},{"expert_nodes",s.expert_nodes},
        {"rejected_cpu_nodes",s.rejected_cpu_nodes},{"rejected_full_copies",s.rejected_full_copies},
        {"ranges",s.ranges},{"chunks",s.chunks},{"source_bytes",s.source_bytes},{"h2d_bytes",s.h2d_bytes},
        {"file_bytes",s.file_bytes},{"mmap_bytes",s.mmap_bytes},{"host_working_set_limit",s.host_working_set_limit},
        {"host_budget_updates",s.host_budget_updates},
        {"ram_cache_bytes",s.ram_cache_bytes},{"ram_cache_budget",s.ram_cache_budget},{"ram_cache_cap",s.ram_cache_cap},
        {"ram_cache_entries",s.ram_cache_entries},{"ram_cache_readers",s.ram_cache_readers},{"ram_cache_peak_bytes",s.ram_cache_peak_bytes},
        {"ram_cache_hits",s.ram_cache_hits},{"ram_cache_misses",s.ram_cache_misses},
        {"ram_cache_mapped_bytes",s.ram_cache_mapped_bytes},{"ram_cache_file_bytes",s.ram_cache_file_bytes},
        {"ram_cache_admissions",s.ram_cache_admissions},{"ram_cache_evictions",s.ram_cache_evictions},
        {"ram_cache_gpu_drops",s.ram_cache_gpu_drops},{"ram_cache_gpu_drop_bytes",s.ram_cache_gpu_drop_bytes},
        {"ram_cache_prefill_bypasses",s.ram_cache_prefill_bypasses},{"ram_cache_rejected",s.ram_cache_rejected},
        {"ram_cache_gpu_waits",s.ram_cache_gpu_waits},{"ram_cache_history_entries",s.ram_cache_history_entries},
        {"pipeline_plans",s.pipeline_plans},{"pipeline_matrices",s.pipeline_matrices},
        {"pipeline_plan_peak",s.pipeline_plan_peak},{"pipeline_lookahead_plans",s.pipeline_lookahead_plans},
        {"pipeline_copy_batches",s.pipeline_copy_batches},{"pipeline_copy_fences",s.pipeline_copy_fences},{"pipeline_scratch_fences",s.pipeline_scratch_fences},{"pipeline_copy_submissions",s.pipeline_copy_submissions},{"pipeline_pending_fills_peak",s.pipeline_pending_fills_peak},{"pipeline_abort_fences",s.pipeline_abort_fences},
        {"pipeline_scratch_events",s.pipeline_scratch_events},{"pipeline_copy_events",s.pipeline_copy_events},
        {"pipeline_retire_checks",s.pipeline_retire_checks},{"pipeline_retire_waits",s.pipeline_retire_waits},
        {"pipeline_scheduler_waits_skipped",s.pipeline_scheduler_waits_skipped},{"pipeline_observer_fences",s.pipeline_observer_fences},
        {"pipeline_pending_copy",s.pipeline_pending_copy},{"async_compute_calls",s.async_compute_calls},{"graph_exit_fences",s.graph_exit_fences},
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
        {"cache_pressure_trims",s.cache_pressure_trims},{"cache_pressure_groups",s.cache_pressure_groups},
        {"cache_pressure_evicted_bytes",s.cache_pressure_evicted_bytes},{"cache_pressure_released_bytes",s.cache_pressure_released_bytes},
        {"cache_bypasses",s.cache_bypasses},{"cache_group_experts",s.cache_group_experts},{"cache_expert_groups",s.cache_expert_groups},{"cache_partial_expert_groups",s.cache_partial_expert_groups},
        {"cache_ready_matrices",s.cache_ready_matrices},{"cache_pending_matrices",s.cache_pending_matrices},{"cache_ready_bytes",s.cache_ready_bytes},
        {"cache_group_admissions",s.cache_group_admissions},{"cache_group_plan_pins_peak",s.cache_group_plan_pins_peak},{"cache_resident",s.cache_resident},{"cache_limit",s.cache_limit},
        {"arena_reserved",s.arena_reserved},{"arena_live",s.arena_live},{"arena_blocks",s.arena_blocks},
        {"arena_allocations",s.arena_allocations},{"arena_frees",s.arena_frees},{"arena_rejects",s.arena_rejects},
        {"staging_bytes",s.staging_bytes},{"source_ms",s.source_ms},{"h2d_ms",s.h2d_ms},{"compute_ms",s.compute_ms}};
}
static json memory() {
    const auto m=strata_mm27_memory();
    return {{"ram_total",m.ram_total},{"ram_available",m.ram_available},{"vram_total",m.vram_total},
        {"vram_available",m.vram_available},{"process_private",m.process_private},{"process_working_set",m.process_working_set}};
}
static json request(llama_model *model,llama_context *ctx,const json &r,std::ostream *logits,bool stream,
                    bool prefix_cache,ResidentPrefix &prefix,SessionRuntime &sessions) {
    cancelled.store(false);
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
        r.at("max_tokens")>=1 && r.at("max_tokens")<=llama_n_ctx(ctx)),"invalid max_tokens");
    const int count=r.value("max_tokens",8);
    require(count>=1 && !tokens.empty() && tokens.size()+size_t(count)<=llama_n_ctx(ctx),"invalid request/context size");
    for(auto t:tokens)require(t>=0 && t<nv,"token outside vocabulary");
    const auto sampling=SamplingConfig::parse(r.value("sampling",json::object()),nv);
    RequestSampler sampler(sampling); // Fresh seed/state for every request, including after cancellation.
    std::string session;
    if(r.contains("session_key")) {
        require(prefix_cache && r.at("session_key").is_string(),"session_key requires prefix cache");
        session=r.at("session_key").get<std::string>();
        require(session.size()==64 && session.find_first_not_of("0123456789abcdef")==std::string::npos,
                "session_key must be a lowercase SHA-256 digest");
    }
    const auto start=Clock::now();
    const bool restored=sessions.switch_to(prefix,session,tokens);
    const size_t reused=prefix.reusable(session,tokens,llama_n_batch(ctx));
    if(reused) {
        llama_synchronize(ctx);
        require(llama_memory_seq_pos_min(llama_get_memory(ctx),0)==0 &&
                llama_memory_seq_pos_max(llama_get_memory(ctx),0)==int(prefix.computed)-1,
                "resident prefix KV positions mismatch");
        require(llama_memory_seq_rm(llama_get_memory(ctx),0,int(reused),-1),"cannot trim resident KV prefix");
        require(llama_memory_seq_pos_max(llama_get_memory(ctx),0)==int(reused)-1,"KV prefix trim mismatch");
    } else clear(ctx);
    prefix.invalidate(); // No partially computed request is eligible for reuse.
    const double session_ms=elapsed(start);
    strata_mm27_reset();strata_mm27_cache_decode(false);
    json samples=json::array();samples.push_back(memory());
    const auto prefill_start=Clock::now();
    for(size_t i=reused;i<tokens.size();i+=llama_n_batch(ctx)) {
        require(!cancelled.load(),"MiniMax request cancelled");
        sessions.trim();
        decode(ctx,tokens,i,int(std::min<size_t>(llama_n_batch(ctx),tokens.size()-i)),int(i));samples.push_back(memory());
    }
    const double prefill_ms=elapsed(prefill_start);const auto prefill=stats();strata_mm27_reset();
    json generated=json::array();std::string text;double decode_ms=0,ttft_ms=0,sampling_ms=0;bool eos=false;
    for(int i=0;i<count;++i) {
        require(!cancelled.load(),"MiniMax request cancelled");
        const float *l=llama_get_logits_ith(ctx,-1);require(l,"missing logits");
        require(std::all_of(l,l+nv,[](float x){return std::isfinite(x);}),"non-finite logits");
        if(logits) {logits->write(reinterpret_cast<const char *>(l),nv*sizeof(float));require(bool(*logits),"cannot write logits");}
        const auto draw=Clock::now();const llama_token token=sampler.sample(l,nv);sampling_ms+=elapsed(draw);generated.push_back(token);
        std::vector<char> piece(256);int n=llama_token_to_piece(vocab,token,piece.data(),int(piece.size()),0,true);
        if(n<0) {piece.resize(-n);n=llama_token_to_piece(vocab,token,piece.data(),int(piece.size()),0,true);}
        require(n>=0,"cannot decode token piece");text.append(piece.data(),size_t(n));
        if(i==0)ttft_ms=elapsed(start);
        if(stream)std::cout<<json({{"event","token"},{"id",token}}).dump()<<std::endl;
        eos=is_stop(token);if(eos || i+1==count)break;
        const auto step=Clock::now();sessions.trim();decode(ctx,{token},0,1,int(tokens.size())+i);decode_ms+=elapsed(step);samples.push_back(memory());
        // Let the first serial decode finish allocating its workspace before
        // admitting experts. Existing hits remain usable during warmup/prefill.
        strata_mm27_cache_decode(true);
    }
    const double request_ms=elapsed(start);
    const size_t computed=tokens.size()+generated.size()-1; // Last sampled token has no KV yet.
    require(!cancelled.load(),"MiniMax request cancelled");
    require(llama_memory_seq_pos_min(llama_get_memory(ctx),0)==0 &&
            llama_memory_seq_pos_max(llama_get_memory(ctx),0)==int(computed)-1,"completed KV positions mismatch");
    json result={{"event","result"},{"prompt_tokens",tokens.size()},{"generated_tokens",generated.size()},{"token_ids",generated},
        {"reused_tokens",reused},{"evaluated_prompt_tokens",tokens.size()-reused},{"kv_tokens",computed},
        {"session_restore",restored},{"session_ms",session_ms},
        {"session_saved_bytes",sessions.saved_bytes},{"session_restored_bytes",sessions.restored_bytes},
        {"session_archive_bytes",sessions.archive.used},{"session_archive_entries",sessions.archive.count()},
        {"session_archive_evictions",sessions.archive.evictions},{"session_archive_rejected",sessions.archive.rejected},
        {"text",text},{"stop_reason",eos?"eos":"length"},{"max_tokens",count},
        {"stop_token_id",eos?json(generated.back()):json(nullptr)},{"prefill_ms",prefill_ms},{"ttft_ms",ttft_ms},
        {"sampling",sampling.json()},{"sampling_algorithm",sampling.algorithm()},{"sampling_ms",sampling_ms},
        {"decode_forward_tokens",generated.size()-1},{"decode_ms",decode_ms},
        {"decode_tokens_per_second",decode_ms>0?1000.*(generated.size()-1)/decode_ms:0.},
        {"request_ms",request_ms},{"prefill",prefill},{"decode",stats()},{"memory_samples",samples}};
    prefix.commit(session,tokens,llama_n_batch(ctx),computed);
    return result;
}
int main(int argc,char **argv) {
    try {
        std::string path,input,output,logit_path;int size=512,batch=8,mode=2;bool pipe=false,prefix_cache=false;
        uint64_t cache_mib=0,ram_cache_mib=0,session_cache_mib=0,session_cache_slots=4,cache_decay=65536;
        std::string cache_allocator="cuda";
        uint32_t arena_block_mib=64,arena_growth_reserve_mib=0;bool group_experts=false;
        std::string reader="file";
        int pipeline_readers=0,pipeline_chunk=8,pipeline_events=0;bool pipeline_lookahead=false,pipeline_d2d_batch=false;
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
            else if(arg=="--prefix-cache") {require(value=="0" || value=="1","invalid prefix cache flag");prefix_cache=value=="1";}
            else if(arg=="--session-cache-mib" || arg=="--session-cache-slots") {
                require(!value.empty() && value.find_first_not_of("0123456789")==std::string::npos,"invalid session cache limit");
                const auto n=std::stoull(value);
                if(arg=="--session-cache-mib"){require(n<=131072,"session cache MiB too large");session_cache_mib=n;}
                else {require(n>=1 && n<=64,"session cache slots must be 1..64");session_cache_slots=n;}
            }
            else if(arg=="--gpu-cache-mib") {
                require(!value.empty() && value.find_first_not_of("0123456789")==std::string::npos,"invalid GPU cache MiB");
                cache_mib=std::stoull(value);require(cache_mib<=131072,"GPU cache MiB too large");
            }
            else if(arg=="--cache-decay-period") {
                require(!value.empty() && value.find_first_not_of("0123456789")==std::string::npos,"invalid cache decay period");
                cache_decay=std::stoull(value);require(cache_decay>=1 && cache_decay<=UINT32_MAX,"cache decay period must be 1..4294967295");
            }
            else if(arg=="--gpu-cache-allocator") {
                require(value=="cuda" || value=="arena","invalid cache allocator");cache_allocator=value;
            }
            else if(arg=="--ram-cache-mib") {
                require(!value.empty() && value.find_first_not_of("0123456789")==std::string::npos,"invalid RAM cache MiB");
                ram_cache_mib=std::stoull(value);require(ram_cache_mib<=131072,"RAM cache MiB too large");
            }
            else if(arg=="--cache-group-experts") {require(value=="0" || value=="1","invalid group flag");group_experts=value=="1";}
            else if(arg=="--arena-growth-reserve-mib") {
                require(!value.empty() && value.find_first_not_of("0123456789")==std::string::npos,"invalid arena growth reserve MiB");
                const auto reserve=std::stoull(value);require(reserve<=1024,"arena growth reserve MiB too large");arena_growth_reserve_mib=uint32_t(reserve);
            }
            else if(arg=="--arena-block-mib") {
                require(value=="8" || value=="16" || value=="32" || value=="64","invalid arena block MiB");arena_block_mib=uint32_t(std::stoul(value));
            }
            else if(arg=="--expert-reader") {
                require(value=="file" || value=="mmap" || (value=="mmap-direct" || value=="mmap-decode"),"invalid expert reader");reader=value;
            }
            else if(arg=="--pipeline-readers") {require(value=="0" || value=="1" || value=="2","invalid pipeline readers");pipeline_readers=std::stoi(value);}
            else if(arg=="--pipeline-chunk-mib") {require(value=="4" || value=="8" || value=="16","invalid pipeline chunk");pipeline_chunk=std::stoi(value);}
            else if(arg=="--pipeline-lookahead") {require(value=="0" || value=="1","invalid pipeline chunk");pipeline_lookahead=value=="1";}
            else if(arg=="--pipeline-d2d-batch") {require(value=="0" || value=="1","invalid pipeline chunk");pipeline_d2d_batch=value=="1";}
            else if(arg=="--pipeline-events") {require(value=="0" || value=="1" || value=="2","invalid pipeline events mode");pipeline_events=std::stoi(value);}
            else if(arg=="--pipeline-trace")pipeline_trace=value;
            else throw std::runtime_error("unknown argument: "+arg);
        }
        require(!path.empty() && (pipe?input.empty() && output.empty() && logit_path.empty():!input.empty()),
            "usage: --gguf MODEL (--pipe | --request JSON [--output JSON] [--logits F32]) [--ctx 512] [--batch 8] [--mode 2] [--gpu-cache-mib 0]");
        require(size>=256 && size<=4096 && batch>=1 && batch<=16 && (mode==1 || mode==2),"invalid context/batch/mode");
        require(!session_cache_mib || prefix_cache,"session cache requires prefix cache");
        require(!pipeline_events || (pipeline_readers && pipeline_d2d_batch),"pipeline events require readers and D2D batch");
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
        strata_mm27_cache_configure(cache_mib<<20,cache_allocator=="arena",arena_block_mib,arena_growth_reserve_mib,group_experts,cache_decay);
        strata_mm27_pipeline(pipeline_readers,pipeline_chunk,pipeline_lookahead,pipeline_d2d_batch,pipeline_events);
        strata_mm27_ram_cache_configure(ram_cache_mib<<20);
        if(!pipeline_trace.empty())strata_mm27_pipeline_trace(true);
#ifdef _WIN32
        // A supervisor can address CTRL_BREAK to this process group without
        // interrupting the console that launched it.
        require(SetConsoleCtrlHandler(console_interrupt,TRUE)!=0,"cannot install console cancellation handler");
#else
        signal(SIGINT,interrupt);
#endif
        strata_mm27_cancel(&cancelled);
        const auto before=memory();const auto start=Clock::now();
        auto model=load(path);auto ctx=context(model.get(),size,batch);const double load_ms=elapsed(start);
        ResidentPrefix prefix;
        SessionRuntime sessions(ctx.get(),size_t(session_cache_mib)<<20,size_t(session_cache_slots));
        json header={{"architecture","minimax-m2"},{"source_revision",STRATA_MM27_SOURCE_SHA},{"patches",STRATA_MM27_PATCH_SET},
            {"model",path},{"mode",mode},{"context",size},{"batch",batch},{"kv","F32"},{"flash_attention",false},
            {"strict_f32",true},{"prefix_cache",prefix_cache},{"graphs",false},{"cache",cache_mib>0},{"gpu_cache_mib",cache_mib},{"ram_cache_mib",ram_cache_mib},
            {"session_cache_mib",session_cache_mib},{"session_cache_slots",session_cache_slots},
            {"gpu_cache_allocator",cache_allocator},{"cache_decay_period",cache_decay},
            {"cache_group_experts",group_experts},{"arena_block_mib",arena_block_mib},{"arena_growth_reserve_mib",arena_growth_reserve_mib},
            {"expert_reader",reader},{"host_working_set_target_percent",reader=="file" && !ram_cache_mib?0:94},
            {"pipeline_readers",pipeline_readers},{"pipeline_chunk_mib",pipeline_chunk},{"pipeline_lookahead",pipeline_lookahead},{"pipeline_d2d_batch",pipeline_d2d_batch},
            {"pipeline_events",pipeline_events},{"compute_ms_scope",pipeline_events==2?"split CPU submission (async); graph exit drains":"synchronous split completion"},
            {"pipeline_trace",pipeline_trace},
            {"cache_policy","decode admission after first serial step; prefill hits only; global 95% minus 256 MiB"},
            {"mtp",false},{"load_ms",load_ms},{"memory_before",before},{"memory_loaded",memory()}};
        if(pipe) {
            header["event"]="ready";std::cout<<header.dump()<<std::endl;std::string line;
            while(std::getline(std::cin,line))try {
                require(line.size()<=2u<<20,"pipe request too large");
                std::cout<<request(model.get(),ctx.get(),json::parse(line),nullptr,true,prefix_cache,prefix,sessions).dump(-1,' ',false,json::error_handler_t::replace)<<std::endl;
            } catch(const std::exception &e) {
                prefix.invalidate();clear(ctx.get());std::cout<<json({{"event","error"},{"message",e.what()}}).dump()<<std::endl;
            }
        } else {
            if(r.is_array()) {
                require(!r.empty() && r.size()<=32,"invalid request corpus");header["results"]=json::array();
                for(const auto &item:r)header["results"].push_back(request(model.get(),ctx.get(),item,logit.is_open()?&logit:nullptr,false,prefix_cache,prefix,sessions));
            } else header["result"]=request(model.get(),ctx.get(),r,logit.is_open()?&logit:nullptr,false,prefix_cache,prefix,sessions);
            const auto report=header.dump(2,' ',false,json::error_handler_t::replace);
            if(out.is_open())out<<report<<'\n';std::cout<<report<<'\n';
        }
        if(!pipeline_trace.empty())strata_mm27_pipeline_trace_write(pipeline_trace.c_str());
        return 0;
    } catch(const std::exception &e) {std::cerr<<"MiniMax: "<<e.what()<<'\n';return 2;}
}
