#include "runtime.hpp"
#include "sync_test.h"
#include "synthetic_minimax_m2.hpp"
#include "../common/expert_file.hpp"
#include "nlohmann/json.hpp"
#include <cuda_runtime_api.h>
#include <chrono>
#include <fstream>
#include <iostream>
using namespace minimax_m2;
using json=nlohmann::ordered_json;
using Floats=std::vector<float>;
using Clock=std::chrono::steady_clock;
static double elapsed(Clock::time_point start) {return std::chrono::duration<double,std::milli>(Clock::now()-start).count();}
static json memory(const strata_mm27_memory_info &m) {
    return {{"ram_total",m.ram_total},{"ram_available",m.ram_available},{"vram_total",m.vram_total},
        {"vram_available",m.vram_available},{"private_bytes",m.process_private},{"working_set",m.process_working_set}};
}
static json stats() {
    const auto s=strata_mm27_snapshot();
    return {{"gpu_nodes",s.gpu_nodes},{"expert_nodes",s.expert_nodes},{"compute_calls",s.compute_calls},
        {"rejected_cpu_nodes",s.rejected_cpu_nodes},{"rejected_full_copies",s.rejected_full_copies},
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
        {"source_bytes",s.source_bytes},{"h2d_bytes",s.h2d_bytes},{"staging_bytes",s.staging_bytes},
        {"memory_checks",s.memory_checks},{"pressure_rejections",s.pressure_rejections},
        {"sampled_ram_used_peak",s.sampled_ram_used_peak},{"sampled_vram_used_peak",s.sampled_vram_used_peak},
        {"sampled_private_peak",s.sampled_private_peak},{"sampled_working_set_peak",s.sampled_working_set_peak},
        {"selected_bytes",s.selected_bytes},{"cache_hit_bytes",s.cache_hit_bytes},{"cache_fill_bytes",s.cache_fill_bytes},
        {"cache_resident",s.cache_resident},{"cache_evictions",s.cache_evictions},{"cache_oom",s.cache_oom},
        {"arena_reserved",s.arena_reserved},{"arena_live",s.arena_live},{"arena_blocks",s.arena_blocks}};
}
static json compare(const std::string &name,const Floats &a,const Floats &b,bool exact=true) {
    require(!a.empty() && a.size()==b.size(),"invalid logit comparison");double max_abs=0,error=0,energy=0;
    bool finite=true;
    for(size_t i=0;i<a.size();++i) {
        finite&=std::isfinite(a[i]) && std::isfinite(b[i]);const double d=double(a[i])-b[i];
        max_abs=std::max(max_abs,std::abs(d));error+=d*d;energy+=double(b[i])*b[i];
    }
    const bool bits=std::memcmp(a.data(),b.data(),a.size()*sizeof(float))==0;
    const double nmse=error/std::max(energy,1e-30);
    return {{"name",name},{"pass",finite && max_abs<=5e-4 && nmse<=1e-7 && (!exact || bits)},
        {"bit_exact",bits},{"max_abs",max_abs},{"nmse",nmse},{"elements",a.size()},{"exact_required",exact}};
}
struct Fault {
    enum Kind {cancel,ram,vram} kind;
    bool decode_only=false,enabled=false,triggered=false;
    int block=0;std::atomic<bool> cancelled{false};
    static void observe(ggml_backend_t,const ggml_tensor *,const ggml_tensor *src,size_t,size_t,void *owner) {
        auto &f=*static_cast<Fault *>(owner);
        if(!f.enabled || f.triggered || std::string(src->name).rfind("blk."+std::to_string(f.block)+".",0)!=0)return;
        f.triggered=true;
        if(f.kind==cancel)f.cancelled.store(true);
        else strata_mm27_test_memory_limits(f.kind==ram?strata_mm27_test_limits{0,UINT64_MAX}:strata_mm27_test_limits{UINT64_MAX,0});
    }
    void detach() {strata_mm27_observe(nullptr,nullptr);strata_mm27_cancel(nullptr);strata_mm27_test_memory_limits({});}
    ~Fault() {detach();}
};
struct Run {Floats logits;std::vector<llama_token> generated;json report;};
static Run evaluate(llama_context *ctx,int vocab,int count,int steps,int token,Fault *fault=nullptr) {
    clear(ctx);strata_mm27_reset();strata_mm27_cache_decode(false);std::vector<llama_token> prompt(count,token);Run r;
    // The repeated prefix bounds expert I/O; the varied suffix exercises real
    // attention over different values at the far end of the context window.
    if(count>64)for(int i=count-33;i<count;++i)prompt[i]=(token+(i*7)%32)%vocab;
    if(fault)fault->enabled=!fault->decode_only;
    const auto start=Clock::now();
    for(int i=0;i<count;i+=llama_n_batch(ctx))decode(ctx,prompt,i,std::min<int>(llama_n_batch(ctx),count-i),i);
    const double prefill_ms=elapsed(start);double decode_ms=0;
    if(fault)fault->enabled=true;
    for(int i=0;i<steps;++i) {
        const auto *l=llama_get_logits_ith(ctx,-1);require(l,"missing logits");
        require(std::all_of(l,l+vocab,[](float a){return std::isfinite(a);}),"non-finite logits");
        r.logits.insert(r.logits.end(),l,l+vocab);const auto next=llama_token(std::max_element(l,l+vocab)-l);
        r.generated.push_back(next);
        if(i+1<steps) {const auto step=Clock::now();decode(ctx,{next},0,1,count+i);decode_ms+=elapsed(step);strata_mm27_cache_decode(true);}
    }
    r.report={{"prompt_tokens",count},{"continuation_steps",steps},{"generated",r.generated},
        {"prefill_ms",prefill_ms},{"decode_ms",decode_ms},{"decode_tokens_per_second",decode_ms>0?1000.*(steps-1)/decode_ms:0.},
        {"stats",stats()},{"memory_after",memory(strata_mm27_memory())}};
    return r;
}
// Controlled real pressure, capped at 90% of both global physical totals.
// RAM uses a separate read-only mapping of the existing checkpoint: committing
// anonymous RAM would exhaust this PC's commit limit before physical RAM fills.
// Only allocations/mappings owned here are released; model mappings stay intact.
struct Pressure {
    std::vector<void *> gpu;uint64_t ram_bytes=0,gpu_bytes=0;
    HANDLE file=INVALID_HANDLE_VALUE,mapping=nullptr;void *view=nullptr;
    uint8_t checksum=0;
    ~Pressure() {
        for(auto p:gpu)cudaFree(p);
        if(view)UnmapViewOfFile(view);if(mapping)CloseHandle(mapping);if(file!=INVALID_HANDLE_VALUE)CloseHandle(file);
    }
    void allocate(const std::string &path,bool cached=false) {
        const size_t chunk=cached?128u<<20:256u<<20;
        for(;;) {
            const auto m=strata_mm27_memory();
            // Replace up to 12 GiB of a warm cache with separate GPU pressure.
            // Each step stays below 95%; the live cache guard reclaims backing
            // before the next allocation. Non-cache runs retain the 90% target.
            if(cached && gpu_bytes>=(12ull<<30))break;
            if(m.vram_available<(cached?m.vram_total/20+(32u<<20):m.vram_total/10)+chunk)break;
            void *p=nullptr;if(cudaMalloc(&p,chunk)!=cudaSuccess) {cudaGetLastError();break;}
            gpu.push_back(p);gpu_bytes+=chunk;
            require(cudaMemset(p,0,chunk)==cudaSuccess && cudaDeviceSynchronize()==cudaSuccess,"pressure buffer initialization failed");
        }
        file=CreateFileW(std::filesystem::path(path).wstring().c_str(),GENERIC_READ,
            FILE_SHARE_READ|FILE_SHARE_WRITE|FILE_SHARE_DELETE,nullptr,OPEN_EXISTING,FILE_FLAG_RANDOM_ACCESS,nullptr);
        require(file!=INVALID_HANDLE_VALUE,"cannot open pressure source");LARGE_INTEGER length{};
        require(GetFileSizeEx(file,&length)!=0,"cannot size pressure source");
        mapping=CreateFileMappingW(file,nullptr,PAGE_READONLY,0,0,nullptr);require(mapping,"cannot create pressure mapping");
        view=MapViewOfFile(mapping,FILE_MAP_READ,0,0,0);require(view,"cannot map pressure source");
        for(uint64_t offset=0;offset<uint64_t(length.QuadPart);offset+=chunk) {
            const auto m=strata_mm27_memory();if(m.ram_available<m.ram_total/10+chunk)break;
            const size_t count=size_t(std::min<uint64_t>(chunk,uint64_t(length.QuadPart)-offset));
            const auto *source=static_cast<const volatile uint8_t *>(view)+offset;
            for(size_t i=0;i<count;i+=4096)checksum^=source[i];
            ram_bytes+=count;
        }
    }
};
int main(int argc,char **argv) {
    json report={{"architecture","minimax-m2"},{"source_revision",STRATA_MM27_SOURCE_SHA},{"patches",STRATA_MM27_PATCH_SET},
        {"strict_f32",true},{"flash_attention",false},{"tests",json::array()},{"runs",json::array()}};
    json tests=json::array(); // ordered_json object insertion can invalidate references to its values.
    std::filesystem::path directory;bool created=false;
    try {
        std::string path;bool fixture=false,long_only=false,pressure=false;int size=512,count=31,batch=8,steps=8;
        uint64_t cache_mib=0;std::string allocator="cuda",reader="file";
        int pipeline_readers=0,pipeline_chunk=8;bool pipeline_lookahead=false,pipeline_d2d_batch=false;
        for(int i=1;i<argc;++i) {
            std::string arg=argv[i];
            if(arg=="--fixture") {fixture=true;continue;}
            if(arg=="--long-only") {long_only=true;continue;}
            if(arg=="--pressure") {pressure=true;continue;}
            require(i+1<argc,"missing argument: "+arg);std::string value=argv[++i];
            if(arg=="--gguf")path=value;else if(arg=="--out")directory=value;
            else if(arg=="--ctx")size=std::stoi(value);else if(arg=="--tokens")count=std::stoi(value);
            else if(arg=="--batch")batch=std::stoi(value);else if(arg=="--steps")steps=std::stoi(value);
            else if(arg=="--gpu-cache-mib") {require(value.find_first_not_of("0123456789")==std::string::npos,"invalid cache cap");cache_mib=std::stoull(value);require(cache_mib<=131072,"cache cap too large");}
            else if(arg=="--gpu-cache-allocator") {require(value=="cuda" || value=="arena","invalid allocator");allocator=value;}
            else if(arg=="--expert-reader") {require(value=="file" || value=="mmap" || (value=="mmap-direct" || value=="mmap-decode"),"invalid reader");reader=value;}
            else if(arg=="--pipeline-readers") {require(value=="0" || value=="1" || value=="2","invalid pipeline readers");pipeline_readers=std::stoi(value);}
            else if(arg=="--pipeline-chunk-mib") {require(value=="4" || value=="8" || value=="16","invalid pipeline chunk");pipeline_chunk=std::stoi(value);}
            else if(arg=="--pipeline-lookahead") {require(value=="0" || value=="1","invalid pipeline chunk");pipeline_lookahead=value=="1";}
            else if(arg=="--pipeline-d2d-batch") {require(value=="0" || value=="1","invalid pipeline chunk");pipeline_d2d_batch=value=="1";}
            else throw std::runtime_error("unknown argument: "+arg);
        }
        require((fixture?path.empty():!path.empty()) && !directory.empty(),"specify --fixture or --gguf MODEL, and --out NEW_DIRECTORY");
        require(size>=256 && size<=4096 && batch>=1 && batch<=16 && steps>=2 && steps<=64 && count>0 && count<=size-steps,"invalid context/count/batch/steps");
        require(!pressure || (!fixture && !long_only),"real pressure requires a full-model lifecycle run");
        require(!std::filesystem::exists(directory),"output directory exists");
        std::filesystem::create_directories(directory);created=true;environment();ggml_backend_load_all();
        struct Release {~Release(){strata_mm27_release();}} release;
        if(fixture) {path=(directory/"fixture.gguf").string();write_synthetic_minimax_m2(path,true);}
        const auto contract=inspect(path,fixture);const int token=fixture?11:758;
        report["configuration"]={{"model",path},{"fixture",fixture},{"context",size},{"batch",batch},{"prompt_tokens",count},
            {"gpu_cache_mib",cache_mib},{"gpu_cache_allocator",allocator},
            {"expert_reader",reader},
            {"pipeline_readers",pipeline_readers},{"pipeline_chunk_mib",pipeline_chunk},{"pipeline_lookahead",pipeline_lookahead},{"pipeline_d2d_batch",pipeline_d2d_batch},
            {"continuation_steps",steps},{"input_pattern","repeated token; for length >64, final 33 tokens vary; numerical stress, not natural-language quality"},{"token_id",token}};
        Run file,reference;
        for(int mode:{2,1}) {
            std::cerr<<"Lifecycle: context "<<size<<", tokens "<<count<<", mode "<<mode<<'\n';
            strata_mm27_mode(mode);strata_mm27_cache_configure(mode==2?cache_mib<<20:0,allocator=="arena");
            strata_mm27_reader(mode==1 || reader=="file"?0:reader=="mmap"?1:reader=="mmap-direct"?2:3);
            strata_mm27_pipeline(mode==2?pipeline_readers:0,pipeline_chunk,mode==2 && pipeline_lookahead,mode==2 && pipeline_d2d_batch);
            auto model=load(path,false,fixture);auto ctx=context(model.get(),size,batch);
            auto run=evaluate(ctx.get(),contract.vocab,count,steps,token);
            run.report["mode"]=mode;report["runs"].push_back(run.report);
            std::ofstream binary(directory/("mode"+std::to_string(mode)+".f32"),std::ios::binary);
            binary.write(reinterpret_cast<const char *>(run.logits.data()),run.logits.size()*sizeof(float));require(bool(binary),"cannot save logits");
            const auto s=strata_mm27_snapshot();
            tests.push_back({{"name","gpu_and_budget_audit/mode"+std::to_string(mode)},
                {"pass",s.gpu_nodes && s.expert_nodes && !s.rejected_cpu_nodes && !s.rejected_full_copies && !s.pressure_rejections &&
                    s.memory_checks>2 && (mode==1 || s.source_bytes==s.h2d_bytes)}});
            if(mode==2)file=std::move(run);else reference=std::move(run);
            if(mode==1 && cache_mib && fixture) {
                strata_mm27_mode(2);
                const auto replay=evaluate(ctx.get(),contract.vocab,count,steps,token);
                tests.push_back(compare("diagnostic_same_context_file_vs_native",replay.logits,reference.logits));
                tests.push_back(compare("same_context_vs_initial_file",replay.logits,file.logits));
                strata_mm27_mode(1);
            }
            if(mode==2 && cache_mib)tests.push_back({{"name","cache_admitted_with_bounded_backing"},
                {"pass",s.cache_fill_bytes>0 && s.cache_resident>0 && s.cache_resident<=(cache_mib<<20) &&
                    s.arena_reserved<=(cache_mib<<20) && (allocator!="arena" || s.arena_live==s.cache_resident)}});
            if(mode==2 && !long_only) {
                const auto fresh=evaluate(ctx.get(),contract.vocab,17,3,token);
                for(int fault=0;fault<4;++fault) {
                    Fault f;f.kind=fault<2?Fault::cancel:fault==2?Fault::ram:Fault::vram;
                    f.decode_only=fault==1 || fault==3;f.block=int(contract.blocks)/2;
                    strata_mm27_observe(Fault::observe,&f);strata_mm27_cancel(&f.cancelled);
                    bool failed=false;std::string error;
                    try {evaluate(ctx.get(),contract.vocab,17,3,token,&f);} catch(const std::exception &e) {failed=true;error=e.what();}
                    const auto s=strata_mm27_snapshot();const auto captured=stats();f.detach();
                    const std::string label=fault==0?"cancel_prefill":fault==1?"cancel_decode":fault==2?"RAM_pressure_injected":"VRAM_pressure_injected";
                    // A pipeline may hold future GPU hits when an injected
                    // VRAM cap arrives. The arena rejects this case before the
                    // final 95% guard; drain then releases these residency pins.
                    const bool pinned_vram=fault==3 && error.find("MiniMax pinned cache cannot shrink under memory pressure; drain plan first")!=std::string::npos;
                    const bool reason=fault<2?error.find("cancelled")!=std::string::npos:pinned_vram ||
                        error.find("95")!=std::string::npos && (fault==2?
                            error.find("RAM")!=std::string::npos && error.find("VRAM")==std::string::npos:error.find("VRAM")!=std::string::npos);
                    tests.push_back({{"name",label},{"pass",failed && f.triggered && s.selected_bytes>0 && reason &&
                        (fault<2?s.pressure_rejections==0:s.pressure_rejections==1)},{"error",error},{"stats",captured}});
                    tests.push_back(compare(label+"/fresh_recovery",evaluate(ctx.get(),contract.vocab,17,3,token).logits,fresh.logits));
                }
                if(pressure) {
                    if(cache_mib)report["pressure_warmup"]=evaluate(ctx.get(),contract.vocab,17,8,token).report;
                    std::cerr<<"Lifecycle: independent read-only RAM mapping and GPU pressure, within budget95\n";
                    Pressure hold;const auto before=strata_mm27_memory();const auto cache_before=strata_mm27_snapshot();
                    hold.allocate(path,cache_mib>0);const auto during=strata_mm27_memory();const auto cache_during=strata_mm27_snapshot();
                    if(cache_mib)tests.push_back({{"name","real_GPU_pressure_reclaims_warm_cache"},
                        {"pass",hold.gpu_bytes>0 && cache_during.cache_resident<cache_before.cache_resident &&
                            (allocator!="arena" || cache_during.arena_reserved<cache_before.arena_reserved)},
                        {"before_resident",cache_before.cache_resident},{"during_resident",cache_during.cache_resident},
                        {"before_backing",cache_before.arena_reserved},{"during_backing",cache_during.arena_reserved}});
                    auto stressed=evaluate(ctx.get(),contract.vocab,17,3,token);
                    tests.push_back(compare("real_pressure/logits",stressed.logits,fresh.logits));
                    tests.push_back({{"name","real_pressure/at_least_85_percent_both"},
                        {"pass",during.ram_available<during.ram_total*15/100 && during.vram_available<during.vram_total*15/100},
                        {"touched_mapped_bytes",hold.ram_bytes},{"held_gpu_bytes",hold.gpu_bytes},{"before",memory(before)},{"during",memory(during)}});
                    report["pressure_run"]=stressed.report;
                }
                tests.push_back(compare("after_pressure_or_fault_release",evaluate(ctx.get(),contract.vocab,17,3,token).logits,fresh.logits));
                clear(ctx.get());ctx.reset();model.reset();strata_mm27_release();
                {
                    auto &r=strata_expert_file::registry();std::lock_guard<std::mutex> lock(r.mutex);
                    tests.push_back({{"name","full_unload_mapping_release"},{"pass",r.files.empty() && !strata_mm27_snapshot().staging_bytes && !strata_mm27_snapshot().arena_reserved}});
                }
                strata_mm27_mode(2);strata_mm27_cache_configure(cache_mib<<20,allocator=="arena");
                strata_mm27_reader(reader=="file"?0:reader=="mmap"?1:reader=="mmap-direct"?2:3);
                strata_mm27_pipeline(pipeline_readers,pipeline_chunk,pipeline_lookahead,pipeline_d2d_batch);model=load(path,false,fixture);ctx=context(model.get(),size,batch);
                tests.push_back(compare("full_unload_reload",evaluate(ctx.get(),contract.vocab,17,3,token).logits,fresh.logits));
            }
            ctx.reset();model.reset();strata_mm27_release();
            report["tests"]=tests;
            std::ofstream(directory/"lifecycle-report.json")<<report.dump(2)<<'\n';
        }
        tests.push_back(compare("file_vs_native/long_context",file.logits,reference.logits));
        tests.push_back({{"name","file_vs_native/continuation_ids"},{"pass",file.generated==reference.generated}});
        require(tests.size()==size_t(long_only?4:pressure?17:15)+(cache_mib?1:0)+(cache_mib && pressure?1:0)+(cache_mib && fixture?2:0),"incomplete lifecycle coverage");
        int failures=0;for(const auto &t:tests)if(!t.at("pass").get<bool>())++failures;
        report["failures"]=failures;report["cases"]=tests.size();report["pass"]=failures==0;
        std::cout<<json({{"cases",tests.size()},{"failures",failures},{"pass",failures==0}}).dump()<<'\n';
    } catch(const std::exception &e) {report["pass"]=false;report["error"]=e.what();std::cerr<<e.what()<<'\n';}
    report["tests"]=tests;
    if(created)std::ofstream(directory/"lifecycle-report.json")<<report.dump(2)<<'\n';
    return report.value("pass",false)?0:1;
}
