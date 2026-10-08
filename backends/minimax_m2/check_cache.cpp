#include "runtime.hpp"
#include "sync_test.h"
#include "synthetic_minimax_m2.hpp"
#include "../common/expert_file.hpp"
#include "nlohmann/json.hpp"
#include <fstream>
#include <iostream>
using namespace minimax_m2;
using json=nlohmann::ordered_json;
using Floats=std::vector<float>;
static Floats run(llama_context *ctx,int salt=0) {
    clear(ctx);Floats out;
    for(int i=0;i<8;++i) {
        decode(ctx,{(i*7+11+salt)%64},0,1,i);
        const auto *l=llama_get_logits_ith(ctx,-1);out.insert(out.end(),l,l+64);
    }
    return out;
}
static bool equal(const Floats &a,const Floats &b) {
    return a.size()==b.size() && std::memcmp(a.data(),b.data(),a.size()*sizeof(float))==0 &&
        std::all_of(a.begin(),a.end(),[](float x){return std::isfinite(x);});
}
struct Observer {
    uint64_t bytes=0,pressure_vram=0;std::atomic<bool> *cancel=nullptr;
    static void check(ggml_backend_t,const ggml_tensor *dst,const ggml_tensor *src,size_t offset,size_t count,void *owner) {
        auto &o=*static_cast<Observer *>(owner);std::vector<uint8_t> data(count);
        ggml_backend_tensor_get(dst,data.data(),offset,count);
        require(!std::memcmp(data.data(),static_cast<const char *>(src->data)+offset,count),"cached delivery bytes differ");
        o.bytes+=count;if(o.cancel)o.cancel->store(true);
        if(o.pressure_vram) {
            strata_mm27_test_limits limits;limits.vram_available=o.pressure_vram;
            strata_mm27_test_memory_limits(limits);strata_mm27_memory();
        }
    }
};
int main(int argc,char **argv) {
    try {
        require(argc>=2,"usage: cache-check NEW_DIRECTORY [--pipeline-readers N --pipeline-chunk-mib N]");const std::filesystem::path dir=argv[1];
        int pipeline_readers=0,pipeline_chunk=8;bool pipeline_lookahead=false,pipeline_d2d_batch=false;
        for(int i=2;i<argc;++i) {
            const std::string arg=argv[i];require(i+1<argc,"missing pipeline argument");const std::string v=argv[++i];
            if(arg=="--pipeline-readers") {require(v=="0" || v=="1" || v=="2","invalid readers");pipeline_readers=std::stoi(v);}
            else if(arg=="--pipeline-chunk-mib") {require(v=="4" || v=="8" || v=="16","invalid chunk");pipeline_chunk=std::stoi(v);}
            else if(arg=="--pipeline-lookahead") {require(v=="0" || v=="1","invalid chunk");pipeline_lookahead=v=="1";}
            else if(arg=="--pipeline-d2d-batch") {require(v=="0" || v=="1","invalid chunk");pipeline_d2d_batch=v=="1";}
            else throw std::runtime_error("unknown cache argument: "+arg);
        }
        require(!std::filesystem::exists(dir),"output directory exists");std::filesystem::create_directories(dir);
        environment();ggml_backend_load_all();json tests=json::array(),runs=json::array();
        auto check=[&](const std::string &name,bool ok){tests.push_back({{"name",name},{"pass",ok}});};
        SIZE_T old_min=0,old_max=0;DWORD old_flags=0;
        require(GetProcessWorkingSetSizeEx(GetCurrentProcess(),&old_min,&old_max,&old_flags)!=0,"read original working set");
        for(int reader:(pipeline_readers?std::vector<int>{0}:std::vector<int>{0,1,2,3}))for(bool arena:{false,true})for(bool mixed:{false,true}) {
            const std::string label=std::to_string(reader)+"/"+(arena?"arena/":"cuda/")+(mixed?"mixed/":"f32/");
            const auto path=(dir/(std::to_string(reader)+"-"+(arena?"arena-":"cuda-")+(mixed?"mixed.gguf":"f32.gguf"))).string();
            write_synthetic_minimax_m2(path,mixed);strata_mm27_mode(2);
            auto model=load(path,false,true);auto ctx=context(model.get());
            const auto reference=run(ctx.get()),topic=run(ctx.get(),3);
            strata_mm27_reader(reader);
            strata_mm27_pipeline(pipeline_readers,pipeline_chunk,pipeline_lookahead,pipeline_d2d_batch);
            if(pipeline_d2d_batch) {
                strata_mm27_reset();
                check(label+"batch_uncached_exact",equal(run(ctx.get()),reference));
                const auto s=strata_mm27_snapshot();
                check(label+"batch_uncached_fences",s.pipeline_copy_batches>0 && s.pipeline_copy_batches==s.pipeline_copy_fences &&
                    s.pipeline_copy_fences==s.pipeline_scratch_fences && s.pipeline_copy_batches==s.pipeline_matrices && !s.pipeline_abort_fences);
            }
            Observer observer;strata_mm27_observe(Observer::check,&observer);
            for(uint64_t cap:{2ull<<20,256ull<<20}) {
                strata_mm27_cache_configure(cap,arena);strata_mm27_cache_decode(true);strata_mm27_reset();
                check(label+"cold_exact/"+std::to_string(cap),equal(run(ctx.get()),reference));
                const auto cold=strata_mm27_snapshot();
                if(pipeline_d2d_batch)check(label+"batch_fences_and_fill_pins/"+std::to_string(cap),
                    cold.pipeline_copy_batches==cold.pipeline_matrices && cold.pipeline_copy_fences==cold.pipeline_copy_batches &&
                    cold.pipeline_scratch_fences==cold.pipeline_copy_batches && cold.pipeline_copy_fences<cold.pipeline_copy_submissions &&
                    cold.pipeline_pending_fills_peak>=2 && !cold.pipeline_abort_fences);
                if(pipeline_lookahead)check(label+"lookahead_plan_bound/"+std::to_string(cap),
                    cold.pipeline_lookahead_plans>0 && cold.pipeline_plan_peak==3 &&
                    cold.pipeline_matrices==3*cold.pipeline_plans);
                if(pipeline_readers)check(label+"pipeline_capacity_and_drain/"+std::to_string(cap),
                    cold.pipeline_groups && cold.pipeline_h2d_bytes==cold.h2d_bytes && cold.pipeline_d2d_bytes==cold.h2d_bytes &&
                    !cold.pipeline_unused_bytes && !cold.pipeline_queued_bytes && !cold.pipeline_reader_owned_bytes &&
                    cold.staging_bytes==(uint64_t(pipeline_chunk)<<22) && cold.pipeline_device_bytes==cold.staging_bytes);
                check(label+"reader_accounting/"+std::to_string(cap),cold.file_bytes+cold.mmap_bytes==cold.source_bytes &&
                    (reader?cold.mmap_bytes>0 && !cold.file_bytes && cold.host_working_set_limit>0:
                        cold.file_bytes>0 && !cold.mmap_bytes && !cold.host_working_set_limit));
                strata_mm27_reset();observer.bytes=0;
                check(label+"warm_exact/"+std::to_string(cap),equal(run(ctx.get()),reference));
                const auto warm=strata_mm27_snapshot();
                check(label+"bounded/"+std::to_string(cap),warm.cache_resident<=cap && warm.cache_resident<=warm.cache_limit &&
                    warm.arena_reserved<=cap && (!arena || warm.arena_live==warm.cache_resident));
                check(label+"byte_accounting/"+std::to_string(cap),observer.bytes==warm.selected_bytes &&
                    warm.h2d_bytes==warm.source_bytes && warm.h2d_bytes+warm.cache_hit_bytes==warm.selected_bytes+warm.cache_guard_bytes);
                check(label+"eviction_or_all_hits/"+std::to_string(cap),cap<(256ull<<20)?
                    cold.cache_evictions>0 && cold.cache_reuses>0:warm.cache_hits>0 && warm.h2d_bytes==0);
                check(label+"new_topic_exact/"+std::to_string(cap),equal(run(ctx.get(),3),topic));
                if(pipeline_d2d_batch) {
                    strata_mm27_observe(nullptr,nullptr);
                    check(label+"batch_without_observer_exact/"+std::to_string(cap),equal(run(ctx.get()),reference));
                    strata_mm27_observe(Observer::check,&observer);
                }
                runs.push_back({{"fixture",label},{"cap",cap},{"resident",warm.cache_resident},{"hits",warm.cache_hits},
                    {"misses",warm.cache_misses},{"h2d_bytes",warm.h2d_bytes},{"hit_bytes",warm.cache_hit_bytes},
                    {"cold_evictions",cold.cache_evictions},{"cold_reuses",cold.cache_reuses},{"arena_reserved",warm.arena_reserved}});
            }
            strata_mm27_cache_decode(false);strata_mm27_reset();
            check(label+"prefill_hits_exact",equal(run(ctx.get()),reference));
            check(label+"prefill_no_fill",strata_mm27_snapshot().cache_fill_bytes==0);
            if(reader==3) {
                strata_mm27_cache_clear();strata_mm27_reset();
                const bool same=equal(run(ctx.get()),reference);const auto p=strata_mm27_snapshot();
                check(label+"hybrid_prefill_file_exact",same && p.file_bytes>0 && !p.mmap_bytes && !p.cache_fill_bytes);
                strata_mm27_cache_decode(true);strata_mm27_reset();
                const bool decode_same=equal(run(ctx.get()),reference);const auto d=strata_mm27_snapshot();
                check(label+"hybrid_decode_mmap_exact",decode_same && d.mmap_bytes>0 && !d.file_bytes && d.cache_fill_bytes>0);
            }
            strata_mm27_cache_decode(true);
            std::atomic<bool> cancel{false};observer.cancel=&cancel;strata_mm27_cancel(&cancel);
            bool failed=false;try {run(ctx.get());}catch(const std::exception &){failed=true;}
            check(label+"cancel_cache_hit",failed && cancel.load());
            observer.cancel=nullptr;strata_mm27_cancel(nullptr);
            // An aborted asynchronous plan invalidates its cache. Rewarm
            // explicitly; pressure coverage must not depend on abort policy.
            strata_mm27_cache_decode(true);
            check(label+"cancel_recovery",equal(run(ctx.get()),reference));
            if(pipeline_readers && arena) {
                const auto m=strata_mm27_memory();const auto s=strata_mm27_snapshot();
                require(s.arena_reserved && s.arena_reserved<=(256ull<<20),"missing warm arena for pinned pressure");
                // Force a 1 MiB physical arena budget while a tensor plan pins
                // future hits. A partial slab cannot be reclaimed until drain.
                observer.pressure_vram=m.vram_total/20+(256ull<<20)-s.arena_reserved+(1ull<<20);
                bool rejected=false;std::string error;
                try {run(ctx.get());}catch(const std::exception &e) {rejected=true;error=e.what();}
                observer.pressure_vram=0;strata_mm27_test_memory_limits({});
                const auto after=strata_mm27_snapshot();
                check(label+"pinned_pressure_drains",rejected && error.find("pinned cache cannot shrink")!=std::string::npos &&
                    !after.arena_reserved && !after.cache_resident && !after.pipeline_device_bytes && !after.pipeline_queued_bytes);
                strata_mm27_cache_decode(true);
                check(label+"pinned_pressure_recovery",equal(run(ctx.get()),reference));
            }
            const auto before=strata_mm27_snapshot().cache_resident;const auto mem=strata_mm27_memory();
            strata_mm27_test_limits limits;limits.vram_available=mem.vram_total/20+(255ull<<20);
            strata_mm27_test_memory_limits(limits);strata_mm27_memory();
            check(label+"live_pressure_eviction",strata_mm27_snapshot().cache_resident<before);
            limits={};limits.ram_available=0;strata_mm27_test_memory_limits(limits);
            failed=false;try {strata_mm27_memory();}catch(const std::exception &){failed=true;}
            check(label+"RAM_pressure_clears_cache",failed && strata_mm27_snapshot().cache_resident==0 && !strata_mm27_snapshot().arena_reserved);
            limits={};limits.cache_oom=true;strata_mm27_test_memory_limits(limits);
            strata_mm27_cache_decode(true);strata_mm27_reset();
            check(label+"OOM_bypass_exact",equal(run(ctx.get()),reference));
            check(label+"OOM_no_residency",strata_mm27_snapshot().cache_oom>0 && strata_mm27_snapshot().cache_resident==0);
            limits={};limits.cache_fill_failure=true;strata_mm27_test_memory_limits(limits);
            failed=false;try {run(ctx.get());}catch(const std::exception &){failed=true;}
            check(label+"partial_fill_invalidated",failed && strata_mm27_snapshot().cache_resident==0);
            strata_mm27_test_memory_limits({});strata_mm27_cache_decode(true);
            check(label+"fill_failure_recovery",equal(run(ctx.get()),reference));
            if(pipeline_d2d_batch)for(int fault=0;fault<3;++fault) {
                // No observer fence can hide missing completion. Fail only
                // after two experts and their cache fills have been enqueued.
                strata_mm27_observe(nullptr,nullptr);strata_mm27_cache_clear();strata_mm27_reset();
                limits={};cancel.store(false);
                if(fault==0)limits.batch_fail_after=2;
                if(fault==1) {limits.batch_cancel_after=2;limits.batch_cancel=&cancel;strata_mm27_cancel(&cancel);}
                if(fault==2)limits.batch_pressure_after=2;
                strata_mm27_test_memory_limits(limits);strata_mm27_cache_decode(true);
                bool refused=false;try {run(ctx.get());}catch(const std::exception &) {refused=true;}
                const auto s=strata_mm27_snapshot();
                check(label+"pending_batch_fault_drains/"+std::to_string(fault),refused && s.pipeline_abort_fences>0 &&
                    s.pipeline_pending_fills_peak>=2 && s.pipeline_copy_submissions>=4 &&
                    !s.cache_resident && !s.arena_reserved && !s.pipeline_queued_bytes && !s.pipeline_reader_owned_bytes && !s.pipeline_device_bytes);
                strata_mm27_test_memory_limits({});strata_mm27_cancel(nullptr);strata_mm27_cache_decode(true);
                check(label+"pending_batch_fault_recovery/"+std::to_string(fault),equal(run(ctx.get()),reference));
                strata_mm27_observe(Observer::check,&observer);
            }
            if(pipeline_readers)for(int fault=0;fault<(pipeline_lookahead?3:2);++fault) {
                strata_mm27_cache_clear();limits={};limits.pipeline_read_failure=fault==0;limits.pipeline_copy_failure=fault==1;
                limits.pipeline_compute_failure=fault==2;
                strata_mm27_test_memory_limits(limits);strata_mm27_cache_decode(true);
                bool refused=false;try {run(ctx.get());}catch(const std::exception &){refused=true;}
                const auto s=strata_mm27_snapshot();
                check(label+"worker_fault_drains/"+std::to_string(fault),refused && !s.cache_resident && !s.pipeline_queued_bytes &&
                    !s.pipeline_reader_owned_bytes && !s.pipeline_device_bytes);
                strata_mm27_test_memory_limits({});strata_mm27_cache_decode(true);
                check(label+"worker_fault_recovery/"+std::to_string(fault),equal(run(ctx.get()),reference));
            }
            auto *weight=model->layers[0].ffn_gate_exps;const std::string saved=weight->name;
            ggml_set_name(weight,"blk.99.ffn_gate_exps.weight");failed=false;
            try {run(ctx.get());}catch(const std::exception &){failed=true;}
            check(label+"metadata_identity_rejected",failed && strata_mm27_snapshot().cache_resident==0);
            ggml_set_name(weight,saved.c_str());strata_mm27_cache_decode(true);
            check(label+"identity_recovery",equal(run(ctx.get()),reference));
            strata_mm27_observe(nullptr,nullptr);ctx.reset();model.reset();
            check(label+"unload_no_entries",strata_mm27_snapshot().cache_resident==0 && !strata_mm27_snapshot().arena_reserved);
            {
                auto &r=strata_expert_file::registry();std::lock_guard<std::mutex> lock(r.mutex);
                check(label+"unload_no_mapping",r.files.empty());
            }
            model=load(path,false,true);ctx=context(model.get());strata_mm27_cache_decode(true);
            check(label+"reload_exact",equal(run(ctx.get()),reference));
            auto alternate=context(model.get(),512,16);
            check(label+"new_context_reclaims_workspace",strata_mm27_snapshot().cache_resident==0);
            alternate.reset();ctx.reset();model.reset();strata_mm27_release();
            SIZE_T restored_min=0,restored_max=0;DWORD restored_flags=0;
            check(label+"working_set_restored",GetProcessWorkingSetSizeEx(GetCurrentProcess(),&restored_min,&restored_max,&restored_flags) &&
                old_min==restored_min && old_max==restored_max && old_flags==restored_flags && !strata_mm27_snapshot().host_working_set_limit);
        }
        size_t failures=0;for(const auto &t:tests)if(!t.at("pass").get<bool>())++failures;
        json report={{"pipeline_readers",pipeline_readers},{"pipeline_chunk_mib",pipeline_chunk},{"pipeline_lookahead",pipeline_lookahead},{"pipeline_d2d_batch",pipeline_d2d_batch},
            {"tests",tests},{"runs",runs},{"cases",tests.size()},{"failures",failures},{"pass",!failures}};
        std::ofstream(dir/"cache-report.json")<<report.dump(2)<<'\n';std::cout<<report.dump(2)<<'\n';return failures?1:0;
    }catch(const std::exception &e){std::cerr<<e.what()<<'\n';strata_mm27_release();return 2;}
}
