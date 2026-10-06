#include "runtime.hpp"
#include "synthetic_hy3.hpp"
#include "sync_test.h"
#include "../common/expert_file.hpp"
#include "llama-context.h"
#include "nlohmann/json.hpp"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <chrono>
#include <algorithm>
using json = nlohmann::ordered_json;
using namespace hy3;
using Floats = std::vector<float>;
static json stats() {
    const auto s = strata_hy3_sync_snapshot();
    return {{"source_bytes",s.source_bytes},{"h2d_bytes",s.h2d_bytes},{"ranges",s.ranges},
        {"chunks",s.chunks},{"staging_bytes",s.staging_bytes},{"compute_calls",s.compute_calls},
        {"gpu_nodes",s.gpu_nodes},{"expert_nodes",s.expert_nodes},{"rejected_cpu_nodes",s.rejected_cpu_nodes},
        {"rejected_full_copies",s.rejected_full_copies},{"source_ms",s.source_ms},{"h2d_ms",s.h2d_ms}};
}
struct Observer {
    uint64_t bytes = 0, ranges = 0;
    static void copy(void * owner, ggml_backend *, ggml_tensor * dst, const ggml_tensor * src, size_t offset, size_t size) {
        auto & out = *static_cast<Observer *>(owner);
        std::vector<uint8_t> bytes(size);
        ggml_backend_tensor_get(dst, bytes.data(), offset, size);
        require(std::memcmp(bytes.data(), static_cast<const uint8_t *>(src->data)+offset, size) == 0, "expert GPU bytes differ from mmap");
        ++out.ranges; out.bytes += size;
    }
};
static Floats run(llama_context * ctx, int prompt, int batch, strata_hy3_sync_stats * prefill=nullptr) {
    clear(ctx);
    std::vector<llama_token> tokens(prompt);
    for (int i = 0; i < prompt; ++i) tokens[i] = (i*7+11)%64;
    Floats logits;
    for (int i = 0; i < prompt; i += batch) {
        const int count = std::min(batch,prompt-i);
        decode(ctx, tokens, i, count, i, true);
        for (int row = 0; row < count; ++row) {
            auto * p = llama_get_logits_ith(ctx, row); logits.insert(logits.end(),p,p+64);
        }

    }
    if (prefill) *prefill=strata_hy3_sync_snapshot();
    for (int i = 0; i < 8; ++i) {
        decode(ctx, std::vector<llama_token>{(prompt+i)%64}, 0, 1, prompt+i);
        auto * p = llama_get_logits_ith(ctx, -1); logits.insert(logits.end(),p,p+64);
    }
    for (float value : logits) require(std::isfinite(value), "non-finite logits");
    return logits;
}
static json compare(const std::string & name, const Floats & a, const Floats & b) {
    require(a.size() == b.size() && !a.empty(), "logit size mismatch");
    double max_abs = 0;
    for (size_t i = 0; i < a.size(); ++i) max_abs = std::max(max_abs,std::abs(double(a[i])-b[i]));
    const bool exact = std::memcmp(a.data(),b.data(),a.size()*sizeof(float)) == 0;
    json result={{"name",name},{"pass",exact},{"elements",a.size()},{"max_abs",max_abs},{"bit_exact_required",true}};
    if(!exact) {result["actual_logits"]=a;result["reference_logits"]=b;}
    return result;
}
// Invoked only after an uncaptured mismatch. Callbacks change graph splitting;
// their results are diagnostic evidence, not a substitute for the original run.
struct Capture {
    std::vector<std::pair<json,Floats>> tensors;
    static bool callback(ggml_tensor * t,bool ask,void * owner) {
        if(ask) return t->type==GGML_TYPE_F32 && t->op!=GGML_OP_NONE && t->op!=GGML_OP_VIEW &&
            t->op!=GGML_OP_RESHAPE && t->op!=GGML_OP_PERMUTE && t->op!=GGML_OP_TRANSPOSE;
        auto & self=*static_cast<Capture *>(owner);
        std::vector<uint8_t> bytes(ggml_nbytes(t));ggml_backend_tensor_get(t,bytes.data(),0,bytes.size());
        Floats values;values.reserve(ggml_nelements(t));
        for(int64_t d=0;d<t->ne[3];++d) for(int64_t c=0;c<t->ne[2];++c)
            for(int64_t b=0;b<t->ne[1];++b) for(int64_t a=0;a<t->ne[0];++a) {
                float value;std::memcpy(&value,bytes.data()+a*t->nb[0]+b*t->nb[1]+c*t->nb[2]+d*t->nb[3],4);
                values.push_back(value);
            }
        self.tensors.push_back({{{"name",t->name},{"op",ggml_op_name(t->op)},
            {"shape",{t->ne[0],t->ne[1],t->ne[2],t->ne[3]}}},std::move(values)});
        return true;
    }
};
static json diagnose(llama_model * resident,llama_model * mapped,llama_context * a,llama_context * b,
                     int batch,const Floats & reference,const Floats & candidate) {
    json out={{"scope","post-failure weights, uncaptured repeats, then fresh-context callback capture"},{"weights",json::array()}};
    for(const auto & entry:resident->tensors_by_name) {
        const auto found=std::find_if(mapped->tensors_by_name.begin(),mapped->tensors_by_name.end(),
            [&](const auto & other){return entry.first==other.first;});
        require(found!=mapped->tensors_by_name.end(),"diagnostic weight missing");
        const auto bytes=ggml_nbytes(entry.second);std::vector<uint8_t> x(bytes),y(bytes);
        ggml_backend_tensor_get(entry.second,x.data(),0,bytes);ggml_backend_tensor_get(found->second,y.data(),0,bytes);
        out["weights"].push_back({{"name",entry.first},{"bytes",bytes},{"exact",x==y}});
    }
    strata_hy3_sync_mode(1);strata_hy3_sync_observer(nullptr,nullptr);
    auto ra=run(a,19,batch),rb=run(b,19,batch);
    out["resident_repeat"]=compare("resident_repeat",ra,reference);
    out["native_repeat"]=compare("native_repeat",rb,candidate);
    out["repeated_pair"]=compare("repeated_pair",ra,rb);
    Capture ca,cb;
    auto ac=context(resident,2048,batch,GGML_TYPE_F32,Capture::callback,&ca);
    auto bc=context(mapped,2048,batch,GGML_TYPE_F32,Capture::callback,&cb);
    std::vector<llama_token> tokens(batch);for(int i=0;i<batch;++i) tokens[i]=(i*7+11)%64;
    clear(ac.get());clear(bc.get());
    decode(ac.get(),tokens,0,batch,0,true);decode(bc.get(),tokens,0,batch,0,true);
    Floats captured_a,captured_b;
    for(int row=0;row<batch;++row) {
        auto * p=llama_get_logits_ith(ac.get(),row);captured_a.insert(captured_a.end(),p,p+64);
        p=llama_get_logits_ith(bc.get(),row);captured_b.insert(captured_b.end(),p,p+64);
    }
    out["captured_resident_vs_original"]=compare("captured_resident_vs_original",captured_a,Floats(reference.begin(),reference.begin()+batch*64));
    out["captured_native_vs_original"]=compare("captured_native_vs_original",captured_b,Floats(candidate.begin(),candidate.begin()+batch*64));
    out["captured_nodes_resident"]=ca.tensors.size();out["captured_nodes_native"]=cb.tensors.size();
    out["capture_comparisons"]=json::array();
    for(size_t i=0;i<std::min(ca.tensors.size(),cb.tensors.size());++i) {
        const auto & x=ca.tensors[i];const auto & y=cb.tensors[i];
        if(x.first!=y.first) {out["capture_structure_mismatch"]={{"index",i},{"resident",x.first},{"native",y.first}};break;}
        auto c=compare(x.first["name"].get<std::string>(),x.second,y.second);c["tensor"]=x.first;
        out["capture_comparisons"].push_back(std::move(c));
    }
    return out;
}
struct InterruptedCopy {
    std::shared_ptr<strata_expert_file::Source> source;
    HANDLE original=INVALID_HANDLE_VALUE;
    std::atomic<bool> cancel{false};
    bool read_error;
    size_t uploaded=0;
    explicit InterruptedCopy(bool fail_read):read_error(fail_read) {}
    void restore() {
        if(source) {source->file=original;source.reset();}
        strata_hy3_sync_cancel(nullptr);strata_hy3_sync_observer(nullptr,nullptr);
    }
    ~InterruptedCopy() {restore();}
    static void copy(void * owner,ggml_backend *,ggml_tensor *,const ggml_tensor * src,size_t,size_t bytes) {
        auto & test=*static_cast<InterruptedCopy *>(owner);
        test.uploaded+=bytes;
        if(test.read_error && !test.source) {
            // Trigger a real Win32 ReadFile failure on the next expert range.
            // Keep the original handle alive and restore it before recovery.
            test.source=strata_expert_file::find(src->data,ggml_nbytes(src));
            require(bool(test.source),"missing source for read failure fixture");
            test.original=test.source->file;test.source->file=INVALID_HANDLE_VALUE;
        } else if(!test.read_error) test.cancel.store(true);
    }
};
static void failures(json & cases,const std::string & path) {
    std::cerr<<"HY3_FAILURE_CHECK reference\n";
    Floats reference;
    {
        auto model=load(path,false,false,true);auto ctx=context(model.get());
        strata_hy3_sync_mode(2);reference=run(ctx.get(),19,17);
    }
    strata_hy3_sync_release(); // Next request must allocate a new staging buffer.
    std::cerr<<"HY3_FAILURE_CHECK pinned allocation\n";
    {
        auto model=load(path,false,false,true);auto ctx=context(model.get());
        strata_hy3_sync_mode(2);strata_hy3_sync_reset();
        strata_hy3_test_limits limits;limits.fail_next_pinned_allocation=true;
        strata_hy3_test_set_limits(limits);
        bool rejected=false;try {run(ctx.get(),19,17);}catch(const std::runtime_error &) {rejected=true;}
        const auto s=strata_hy3_sync_snapshot();
        cases.push_back({{"name","injected_pinned_oom_before_upload"},
            {"pass",rejected && s.h2d_bytes==0 && s.staging_bytes==0}});
        strata_hy3_test_set_limits({});
        cases.push_back(compare("recovery_after_pinned_oom",run(ctx.get(),19,17),reference));
        for(bool fail_read:{true,false}) {
            std::cerr<<"HY3_FAILURE_CHECK partial copy "<<fail_read<<'\n';
            strata_hy3_sync_reset();InterruptedCopy test(fail_read);
            strata_hy3_sync_observer(InterruptedCopy::copy,&test);
            if(!fail_read) strata_hy3_sync_cancel(&test.cancel);
            const auto start=std::chrono::steady_clock::now();
            rejected=false;try {run(ctx.get(),19,17);}catch(const std::runtime_error &) {rejected=true;}
            const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            const auto partial=strata_hy3_sync_snapshot();
            test.restore();
            const std::string name=fail_read ? "read_error" : "cancel";
            cases.push_back({{"name",name+"_after_partial_upload"},
                {"pass",rejected && test.uploaded>0 && partial.h2d_bytes==test.uploaded && elapsed<5},
                {"completed_upload_bytes",test.uploaded},{"seconds_until_failure",elapsed}});
            cases.push_back(compare("recovery_after_partial_"+name,run(ctx.get(),19,17),reference));
        }
        for(bool ram:{true,false}) {
            std::cerr<<"HY3_FAILURE_CHECK pressure "<<ram<<'\n';
            strata_hy3_sync_reset();strata_hy3_test_limits low;
            if(ram) low.ram_available=0;else low.gpu_available=0;
            strata_hy3_test_set_limits(low);
            rejected=false;std::string error;
            try {run(ctx.get(),19,17);}catch(const std::runtime_error & e) {rejected=true;error=e.what();}
            strata_hy3_test_set_limits({});
            const auto s=strata_hy3_sync_snapshot();
            const std::string name=ram ? "ram" : "vram";
            cases.push_back({{"name","simulated_"+name+"_pressure_before_compute"},
                {"pass",rejected && s.compute_calls==0 && s.h2d_bytes==0 && error.find("95% limit")!=std::string::npos}});
            cases.push_back(compare("recovery_after_"+name+"_pressure",run(ctx.get(),19,17),reference));
        }
    }
    bool rejected=false;
    std::cerr<<"HY3_FAILURE_CHECK unload\n";
    try {strata_hy3_memory_check(std::numeric_limits<size_t>::max());}
    catch(const std::runtime_error &) {rejected=true;}
    cases.push_back({{"name","reject_impossible_gpu_reserve_without_allocation"},{"pass",rejected}});
    strata_hy3_sync_release();
    {
        auto & r=strata_expert_file::registry();std::lock_guard lock(r.mutex);
        cases.push_back({{"name","unload_releases_registered_sources"},{"pass",r.files.empty()}});
    }
    {
        auto model=load(path,false,false,true);auto ctx=context(model.get());
        strata_hy3_sync_mode(2);
        cases.push_back(compare("reload_after_failures",run(ctx.get(),19,17),reference));
    }
}
static void cache_checks(json & cases,const std::filesystem::path & directory) {
    for(bool mixed:{false,true}) for(int batch:{1,17}) {
        const std::string path=(directory/(mixed ? "hy3-mixed.gguf" : "hy3-f32.gguf")).string();
        const std::string name=std::string("cache/")+(mixed ? "mixed" : "f32")+"/batch="+std::to_string(batch);
        strata_hy3_sync_release();strata_hy3_sync_mode(2);
        auto model=load(path,false,false,true);auto ctx=context(model.get(),2048,batch);
        const auto reference=run(ctx.get(),19,batch);
        configure_cache(model.get(),64<<20);
        Observer observer;strata_hy3_sync_observer(Observer::copy,&observer);
        strata_hy3_sync_reset();cases.push_back(compare(name+"/cold_exact",run(ctx.get(),19,batch),reference));
        auto cold=strata_hy3_sync_snapshot();
        require(cold.cache_bytes && cold.cache_misses && cold.cache_fill_bytes,"cache did not fill");
        strata_hy3_sync_reset();observer={};
        cases.push_back(compare(name+"/warm_exact",run(ctx.get(),19,batch),reference));
        auto warm=strata_hy3_sync_snapshot();
        cases.push_back({{"name",name+"/warm_all_hits_bytes"},{"pass",warm.cache_hits>0 && !warm.cache_misses &&
            !warm.h2d_bytes && !warm.source_bytes && observer.bytes==warm.d2d_bytes && warm.cache_bytes<=warm.cache_budget},
            {"cold_h2d_bytes",cold.h2d_bytes},{"warm_h2d_bytes",warm.h2d_bytes},{"warm_d2d_bytes",warm.d2d_bytes},
            {"cache_bytes",warm.cache_bytes},{"hits",warm.cache_hits}});
        // Same pointers after a new generation must never hit previous entries.
        configure_cache(model.get(),64<<20);strata_hy3_sync_reset();
        cases.push_back(compare(name+"/new_generation_exact",run(ctx.get(),19,batch),reference));
        auto fresh=strata_hy3_sync_snapshot();
        cases.push_back({{"name",name+"/generation_invalidates"},{"pass",fresh.cache_generation!=warm.cache_generation &&
            fresh.cache_misses==cold.cache_misses && fresh.h2d_bytes==cold.h2d_bytes}});
        configure_cache(model.get(),1<<20);strata_hy3_sync_reset();
        cases.push_back(compare(name+"/eviction_exact",run(ctx.get(),19,batch),reference));
        auto tiny=strata_hy3_sync_snapshot();
        cases.push_back({{"name",name+"/bounded_eviction"},{"pass",tiny.cache_evictions>0 && tiny.cache_reuses>0 &&
            tiny.cache_bytes<=1<<20},{"evictions",tiny.cache_evictions},{"reuses",tiny.cache_reuses}});
        strata_hy3_sync_observer(nullptr,nullptr);
        for(bool fill:{false,true}) {
            configure_cache(model.get(),64<<20);strata_hy3_sync_reset();
            strata_hy3_test_limits limits;
            limits.fail_next_cache_allocation=!fill;limits.fail_next_cache_fill=fill;
            strata_hy3_test_set_limits(limits);
            bool refused=false;Floats result;
            try {result=run(ctx.get(),19,batch);}catch(const std::runtime_error &) {refused=true;}
            strata_hy3_test_set_limits({});auto s=strata_hy3_sync_snapshot();
            if(fill) cases.push_back({{"name",name+"/incomplete_fill_invalidated"},{"pass",refused && !s.cache_bytes}});
            else {
                require(!refused && s.cache_oom==1,"cache OOM did not bypass safely");
                cases.push_back(compare(name+"/allocation_oom_exact",result,reference));
            }
            cases.push_back(compare(name+(fill ? "/fill_recovery" : "/oom_recovery"),run(ctx.get(),19,batch),reference));
        }
        for(bool fail_read:{true,false}) {
            configure_cache(model.get(),64<<20);InterruptedCopy test(fail_read);
            strata_hy3_sync_observer(InterruptedCopy::copy,&test);
            if(!fail_read) strata_hy3_sync_cancel(&test.cancel);
            bool refused=false;try {run(ctx.get(),19,batch);}catch(const std::runtime_error &) {refused=true;}
            test.restore();require(refused && test.uploaded>0,"cache partial failure not exercised");
            cases.push_back(compare(name+(fail_read ? "/read_recovery" : "/cancel_recovery"),run(ctx.get(),19,batch),reference));
        }
        configure_cache(model.get(),64<<20);run(ctx.get(),19,batch);
        for(bool ram:{true,false}) {
            strata_hy3_test_limits limits;
            if(ram) limits.ram_available=0;else limits.gpu_available=0;
            strata_hy3_test_set_limits(limits);
            bool refused=false;try {run(ctx.get(),19,batch);}catch(const std::runtime_error &) {refused=true;}
            strata_hy3_test_set_limits({});
            cases.push_back({{"name",name+(ram ? "/ram_pressure_trim" : "/gpu_pressure_trim")},
                {"pass",refused && !strata_hy3_sync_snapshot().cache_bytes}});
            cases.push_back(compare(name+(ram ? "/ram_recovery" : "/gpu_recovery"),run(ctx.get(),19,batch),reference));
        }
        strata_hy3_sync_release();
        require(!strata_hy3_sync_snapshot().cache_bytes,"cache leaked after release");
    }
}
static void pipeline_checks(json & cases,const std::filesystem::path & directory,bool batch_copy=false) {
    for(bool mixed:{false,true}) for(int batch:{1,17}) for(int readers:{1,2}) {
        const auto path=(directory/(mixed ? "hy3-mixed.gguf" : "hy3-f32.gguf")).string();
        const auto name=std::string(batch_copy?"tensor_batch/":"pipeline/")+(mixed?"mixed":"f32")+"/batch="+std::to_string(batch)+"/readers="+std::to_string(readers);
        strata_hy3_sync_release();strata_hy3_sync_mode(2);
        auto model=load(path,false,false,true);auto ctx=context(model.get(),2048,batch);
        const auto reference=run(ctx.get(),19,batch);
        auto configure=[&](size_t cap=64<<20,int trace=0) {
            configure_cache(model.get(),cap);strata_hy3_pipeline_config(readers,4,trace,batch_copy);strata_hy3_sync_reset();
        };
        configure();Observer observer;strata_hy3_sync_observer(Observer::copy,&observer);
        cases.push_back(compare(name+"/cold_exact",run(ctx.get(),19,batch),reference));
        auto cold=strata_hy3_sync_snapshot();
        cases.push_back({{"name",name+"/bounded_ring_bytes"},{"pass",cold.pipeline_groups>0 &&
            cold.pipeline_chunks>0 && cold.source_bytes==cold.h2d_bytes && !cold.pipeline_unused_bytes &&
            cold.staging_bytes==16<<20 && cold.pipeline_device_bytes==16<<20 && cold.pipeline_read_peak<=uint64_t(readers) &&
            !cold.pipeline_queued && !cold.pipeline_reader_owned && observer.bytes==cold.d2d_bytes},
            {"h2d_bytes",cold.h2d_bytes},{"d2d_bytes",cold.d2d_bytes},{"read_peak",cold.pipeline_read_peak}});
        strata_hy3_sync_reset();
        cases.push_back(compare(name+"/warm_exact",run(ctx.get(),19,batch),reference));
        auto warm=strata_hy3_sync_snapshot();
        cases.push_back({{"name",name+"/warm_no_reads"},{"pass",warm.cache_hits>0 && !warm.cache_misses && !warm.h2d_bytes && !warm.source_bytes}});
        if(batch_copy) cases.push_back({{"name",name+"/one_delivery_fence_per_tensor"},
            {"pass",cold.pipeline_copy_batches>0 && cold.pipeline_copy_fences==cold.pipeline_copy_batches &&
                warm.pipeline_copy_fences==warm.pipeline_copy_batches && warm.pipeline_copy_fences<warm.ranges &&
                cold.pipeline_pending_fills_peak>=2},{"cold_fences",cold.pipeline_copy_fences},
            {"warm_fences",warm.pipeline_copy_fences},{"matrices",warm.ranges},{"pending_fills_peak",cold.pipeline_pending_fills_peak}});
        configure(1<<20);
        cases.push_back(compare(name+"/pinned_eviction_exact",run(ctx.get(),19,batch),reference));
        auto tiny=strata_hy3_sync_snapshot();
        cases.push_back({{"name",name+"/eviction_bounded"},{"pass",tiny.cache_evictions>0 && tiny.cache_bytes<=1<<20}});
        strata_hy3_sync_observer(nullptr,nullptr);
        // STOP after a completed consumer transfer while other slots may prefetch.
        configure();InterruptedCopy cancel(false);strata_hy3_sync_cancel(&cancel.cancel);
        strata_hy3_sync_observer(InterruptedCopy::copy,&cancel);
        auto start=std::chrono::steady_clock::now();bool refused=false;
        try {run(ctx.get(),19,batch);}catch(const std::runtime_error &) {refused=true;}
        cancel.restore();auto stopped=strata_hy3_sync_snapshot();
        const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        cases.push_back({{"name",name+"/partial_cancel_drained"},{"pass",refused && cancel.uploaded>0 &&
            !stopped.pipeline_queued && !stopped.pipeline_reader_owned && seconds<5},{"seconds",seconds},{"unused_bytes",stopped.pipeline_unused_bytes}});
        cases.push_back(compare(name+"/cancel_recovery",run(ctx.get(),19,batch),reference));
        // Fault the file before workers start; never mutate a handle concurrently.
        configure();std::shared_ptr<strata_expert_file::Source> source;
        for(const auto & t:model->tensors_by_name) if(expert(t.first)) {source=strata_expert_file::find(t.second->data,ggml_nbytes(t.second));break;}
        require(bool(source),"pipeline test source missing");auto original=source->file;source->file=INVALID_HANDLE_VALUE;
        refused=false;try {run(ctx.get(),19,batch);}catch(const std::runtime_error &) {refused=true;}
        source->file=original;
        auto failed=strata_hy3_sync_snapshot();
        cases.push_back({{"name",name+"/real_readfile_error_drained"},{"pass",refused && !failed.pipeline_queued && !failed.pipeline_reader_owned}});
        cases.push_back(compare(name+"/reader_error_auto_recovery",run(ctx.get(),19,batch),reference));
        for(bool fill:{false,true}) {
            configure_cache(model.get(),64<<20);strata_hy3_test_limits limits;
            if(fill) limits.fail_next_cache_fill=true;else limits.fail_pipeline_submission_after=2;
            strata_hy3_test_set_limits(limits);strata_hy3_pipeline_config(readers,4,0,batch_copy);strata_hy3_sync_reset();
            refused=false;try {run(ctx.get(),19,batch);}catch(const std::runtime_error &) {refused=true;}
            strata_hy3_test_set_limits({});auto s=strata_hy3_sync_snapshot();
            cases.push_back({{"name",name+(fill?"/incomplete_fill":"/partial_submission_error")},
                {"pass",refused && s.h2d_bytes>0 && !s.cache_bytes && !s.pipeline_queued && !s.pipeline_reader_owned}});
            cases.push_back(compare(name+(fill?"/fill_auto_recovery":"/submission_auto_recovery"),run(ctx.get(),19,batch),reference));
        }
        for(bool ram:{true,false}) {
            strata_hy3_test_limits limits;if(ram) limits.ram_available=0;else limits.gpu_available=0;
            strata_hy3_test_set_limits(limits);refused=false;
            try {run(ctx.get(),19,batch);}catch(const std::runtime_error &) {refused=true;}
            strata_hy3_test_set_limits({});
            cases.push_back({{"name",name+(ram?"/ram_pressure":"/vram_pressure")},{"pass",refused && !strata_hy3_sync_snapshot().cache_bytes}});
            cases.push_back(compare(name+(ram?"/ram_recovery":"/vram_recovery"),run(ctx.get(),19,batch),reference));
        }
        if(batch_copy) {
            for(bool warm_cache:{false,true}) for(bool cancel_queue:{false,true}) {
                configure();if(warm_cache) run(ctx.get(),19,batch);strata_hy3_sync_reset();
                std::atomic<bool> stop{false};strata_hy3_test_limits limits;
                if(cancel_queue) {limits.cancel_batch_enqueue_after=2;limits.batch_cancel=&stop;strata_hy3_sync_cancel(&stop);}
                else limits.fail_batch_enqueue_after=2;
                strata_hy3_test_set_limits(limits);refused=false;start=std::chrono::steady_clock::now();
                try {run(ctx.get(),19,batch);}catch(const std::runtime_error &) {refused=true;}
                strata_hy3_test_set_limits({});strata_hy3_sync_cancel(nullptr);auto s=strata_hy3_sync_snapshot();
                const auto elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
                const auto suffix=std::string(warm_cache?"/warm":"/cold")+(cancel_queue?"_queued_cancel":"_queued_failure");
                cases.push_back({{"name",name+suffix},{"pass",refused && s.ranges==2 &&
                    !s.pipeline_reader_owned && !s.pipeline_queued && elapsed<5 &&
                    (cancel_queue ? stop.load() : s.cache_bytes==0)},
                    {"queued_matrices",s.ranges},{"pending_fills_peak",s.pipeline_pending_fills_peak},{"seconds",elapsed}});
                cases.push_back(compare(name+suffix+"_recovery",run(ctx.get(),19,batch),reference));
            }
            configure();strata_hy3_test_limits limits;limits.fail_next_cache_allocation=true;
            strata_hy3_test_set_limits(limits);
            cases.push_back(compare(name+"/allocation_oom_exact",run(ctx.get(),19,batch),reference));
            require(strata_hy3_sync_snapshot().cache_oom==1,"batched cache OOM did not bypass");
            strata_hy3_test_set_limits({});
        }
        if(mixed && batch==17 && readers==1) {
            configure(64<<20,4);cases.push_back(compare(name+"/trace_exact",run(ctx.get(),19,batch),reference));
            strata_hy3_trace_write((directory/(batch_copy?"tensor-batch-trace.json":"pipeline-trace.json")).string().c_str());
        }
        strata_hy3_sync_release();
    }
    // One matrix spans several ring jobs, including an uneven final chunk.
    strata_hy3_sync_mode(2);
    auto model=load((directory/"hy3-wide.gguf").string(),false,false,true);auto ctx=context(model.get(),2048,1);
    const auto reference=run(ctx.get(),3,1);
    configure_cache(model.get(),1<<20);strata_hy3_pipeline_config(1,4,0,batch_copy);strata_hy3_sync_reset();
    Observer observer;strata_hy3_sync_observer(Observer::copy,&observer);
    cases.push_back(compare(batch_copy?"tensor_batch/wide_multichunk_exact":"pipeline/wide_multichunk_exact",run(ctx.get(),3,1),reference));
    strata_hy3_sync_release();
}
static void reference_probe(json & cases,const std::filesystem::path & directory) {
    strata_hy3_sync_release();strata_hy3_sync_mode(1);
    const auto path=(directory/"hy3-mixed.gguf").string();
    write_synthetic_hy3(path,true);
    for(int reload=0;reload<4;++reload) {
        auto resident=load(path,true,false,true),mapped=load(path,false,false,true);
        auto a=context(resident.get(),2048,4),b=context(mapped.get(),2048,4);
        Floats first;
        for(int repeat=0;repeat<8;++repeat) {
            auto actual=run(a.get(),19,4),reference=run(b.get(),19,4);
            const auto name="reference_probe/reload="+std::to_string(reload)+"/repeat="+std::to_string(repeat);
            cases.push_back(compare(name+"/resident_native",actual,reference));
            if(repeat) cases.push_back(compare(name+"/native_repeat",reference,first));
            else first=reference;
        }
    }
}
static void delivery_profile_checks(json & cases,const std::filesystem::path & directory) {
    strata_hy3_sync_release();strata_hy3_sync_mode(2);
    auto model=load((directory/"hy3-mixed.gguf").string(),false,false,true);auto ctx=context(model.get(),2048,17);
    const auto reference=run(ctx.get(),19,17);
    for(bool batch:{false,true}) for(bool enabled:{false,true}) {
        const auto name=std::string("delivery_profile/batch=")+(batch?"1":"0")+"/enabled="+(enabled?"1":"0");
        strata_hy3_delivery_profile(enabled);configure_cache(model.get(),1<<20);
        strata_hy3_pipeline_config(2,4,0,batch);strata_hy3_sync_reset();
        cases.push_back(compare(name+"/exact",run(ctx.get(),19,17),reference));
        const auto s=strata_hy3_sync_snapshot();
        cases.push_back({{"name",name+"/counters"},{"pass",s.cache_allocations>0 && s.cache_evictions>0 &&
            (enabled ? (s.cache_get_ms>0 && s.cache_admit_ms>0 && s.cache_victim_ms>0 &&
                s.cache_allocate_ms>0 && s.cache_refresh_ms>0 && s.cache_probe_ms>0 && s.cache_protect_ms>0 &&
                s.cache_victim_candidates>0 && s.pipeline_scratch_wait_ms>0 && s.pipeline_delivery_wait_ms>0) :
                (s.cache_get_ms==0 && s.cache_admit_ms==0 && s.cache_victim_ms==0 && s.cache_allocate_ms==0 &&
                s.cache_refresh_ms==0 && s.cache_probe_ms==0 && s.cache_protect_ms==0 && s.cache_victim_candidates==0 &&
                s.pipeline_scratch_wait_ms==0 && s.pipeline_delivery_wait_ms==0))},
            {"admit_ms",s.cache_admit_ms},{"allocate_ms",s.cache_allocate_ms},{"victim_ms",s.cache_victim_ms}});
        strata_hy3_sync_reset();const auto zero=strata_hy3_sync_snapshot();
        cases.push_back({{"name",name+"/reset"},{"pass",!zero.cache_allocations && !zero.cache_victim_candidates &&
            zero.cache_get_ms==0 && zero.cache_admit_ms==0 && zero.pipeline_delivery_wait_ms==0 && zero.cache_bytes>0}});
    }
    strata_hy3_sync_release();
}
static void host_cache_checks(json &cases,const std::filesystem::path &directory) {
    for(bool frequency:{false,true}) for(bool mixed:{false,true}) for(int batch:{1,17}) for(int readers:{0,1,2}) {
        const auto path=(directory/(mixed?"hy3-mixed.gguf":"hy3-f32.gguf")).string();
        const auto name=std::string("ram/")+(frequency?"frequency/":"lru/")+(mixed?"mixed":"f32")+"/batch="+std::to_string(batch)+"/readers="+std::to_string(readers);
        strata_hy3_sync_release();strata_hy3_sync_mode(2);
        auto model=load(path,false,false,true);auto ctx=context(model.get(),2048,batch);
        const auto reference=run(ctx.get(),19,batch);
        // No matrix fits in the tiny GPU cache; every selected copy exercises
        // the RAM tier while retaining the normal pipeline/scheduler path.
        configure_cache(model.get(),1024);strata_hy3_ram_cache_config(64<<20,frequency);
        strata_hy3_pipeline_config(readers,4,0,readers==2);
        strata_hy3_sync_reset();Observer observer;strata_hy3_sync_observer(Observer::copy,&observer);
        if(frequency) {
            strata_hy3_cache_prefill(true);
            std::shared_ptr<strata_expert_file::Source> source;
            for(const auto &t:model->tensors_by_name) if(expert(t.first)) {
                source=strata_expert_file::find(t.second->data,ggml_nbytes(t.second));break;
            }
            require(bool(source),"RAM phase test source missing");
            auto original=source->file;source->file=INVALID_HANDLE_VALUE;
            bool refused=false;try {run(ctx.get(),19,batch);}catch(const std::runtime_error &) {refused=true;}
            source->file=original;
            auto failed=strata_hy3_sync_snapshot();
            cases.push_back({{"name",name+"/phase_error_drained"},{"pass",refused && !failed.ram_cache_pending &&
                !failed.pipeline_queued && !failed.pipeline_reader_owned}});
            // This is the next request's phase transition, before begin_plan.
            // It must recover a latched native read error as well as STOP.
            strata_hy3_cache_prefill(true);
            cases.push_back(compare(name+"/prefill_exact",run(ctx.get(),19,batch),reference));
            auto p=strata_hy3_sync_snapshot();
            cases.push_back({{"name",name+"/prefill_bypass"},{"pass",p.ram_cache_prefill_bypasses>0 &&
                !p.ram_cache_bytes && !p.ram_cache_history_entries && !p.ram_cache_fill_bytes}});
            strata_hy3_cache_prefill(false);strata_hy3_sync_reset();
        }
        cases.push_back(compare(name+"/cold_exact",run(ctx.get(),19,batch),reference));
        auto cold=strata_hy3_sync_snapshot();
        cases.push_back({{"name",name+"/fill_accounting"},{"pass",cold.ram_cache_bytes>0 && cold.ram_cache_bytes<=64<<20 &&
            cold.ram_cache_file_bytes>0 && cold.ram_cache_hit_bytes+cold.ram_cache_file_bytes==cold.h2d_bytes && !cold.ram_cache_pending}});
        if(frequency)cases.push_back(compare(name+"/promotion_exact",run(ctx.get(),19,batch),reference));
        strata_hy3_sync_reset();observer.bytes=observer.ranges=0;
        cases.push_back(compare(name+"/warm_exact",run(ctx.get(),19,batch),reference));
        auto warm=strata_hy3_sync_snapshot();
        cases.push_back({{"name",name+"/warm_without_file_io"},{"pass",warm.ram_cache_hits>0 && !warm.ram_cache_file_bytes &&
            warm.ram_cache_hit_bytes==warm.h2d_bytes && observer.bytes==warm.h2d_bytes && !warm.ram_cache_pending},
            {"verified_gpu_bytes",observer.bytes},{"ram_bytes",warm.ram_cache_bytes}});
        strata_hy3_sync_observer(nullptr,nullptr);
        configure_cache(model.get(),1024);
        cases.push_back({{"name",name+"/new_generation_clears_ram"},{"pass",!strata_hy3_sync_snapshot().ram_cache_bytes}});
        strata_hy3_sync_release();
    }
}
static void gpu_prefill_policy_checks(json &cases,const std::filesystem::path &directory) {
    for(bool ram:{false,true})for(bool mixed:{false,true})for(int batch:{1,17})for(int readers:{0,1,2}) {
        const auto name=std::string("gpu_decode_policy/")+(ram?"ram/":"file/")+(mixed?"mixed/":"f32/")+
            "batch="+std::to_string(batch)+"/readers="+std::to_string(readers);
        strata_hy3_sync_release();strata_hy3_sync_mode(2);
        auto model=load((directory/(mixed?"hy3-mixed.gguf":"hy3-f32.gguf")).string(),false,false,true);
        auto ctx=context(model.get(),2048,batch);const auto reference=run(ctx.get(),19,batch);
        configure_cache(model.get(),64<<20);strata_hy3_gpu_cache_policy(false);
        if(ram)strata_hy3_ram_cache_config(8<<20);
        strata_hy3_pipeline_config(readers,4,0,readers==2);
        Observer observer;strata_hy3_sync_observer(Observer::copy,&observer);
        strata_hy3_cache_prefill(true);strata_hy3_sync_reset();
        cases.push_back(compare(name+"/cold_prefill_exact",run(ctx.get(),19,batch),reference));
        auto p=strata_hy3_sync_snapshot();
        cases.push_back({{"name",name+"/prefill_without_admission"},{"pass",p.h2d_bytes>0 &&
            p.cache_prefill_bypasses>0 && !p.cache_bytes && !p.cache_fill_bytes && !p.cache_allocations && !p.ram_cache_bytes}});
        strata_hy3_cache_prefill(false);strata_hy3_sync_reset();
        cases.push_back(compare(name+"/decode_admission_exact",run(ctx.get(),19,batch),reference));
        auto d=strata_hy3_sync_snapshot();
        cases.push_back({{"name",name+"/batched_decode_admitted"},{"pass",d.cache_bytes>0 && d.cache_fill_bytes>0 &&
            !d.cache_prefill_bypasses && d.cache_bytes<=64<<20}});
        strata_hy3_cache_prefill(true);strata_hy3_sync_reset();
        cases.push_back(compare(name+"/prefill_hit_exact",run(ctx.get(),19,batch),reference));
        auto warm=strata_hy3_sync_snapshot();
        cases.push_back({{"name",name+"/prefill_preserves_existing_hits"},{"pass",warm.cache_bytes==d.cache_bytes &&
            warm.cache_hits>0 && !warm.cache_misses && !warm.h2d_bytes && !warm.cache_fill_bytes && !warm.cache_evictions}});
        strata_hy3_sync_observer(nullptr,nullptr);
        strata_hy3_sync_release();
    }
}
int main(int argc,char ** argv) {
    json report={{"status","error"},{"scope","synthetic resident/native-selected/pinned-file byte and logit parity"},
        {"source_sha",STRATA_HY3_SOURCE_SHA},{"patch_set",STRATA_HY3_PATCH_SET},{"cuda_fusion",false},{"cases",json::array()}};
    std::filesystem::path directory;bool created=false;
    try {
        const bool reference_only=argc==3 && std::string(argv[2])=="--reference-probe";
        const bool graph_history=argc==3 && std::string(argv[2])=="--history-graphs-probe";
        const bool history_only=graph_history || (argc==3 && std::string(argv[2])=="--history-probe");
        require(argc==2 || reference_only || history_only,"usage: strata-hy3-runtime-check NEW_DIRECTORY [--reference-probe|--history-probe|--history-graphs-probe]");directory=argv[1];
        require(!std::filesystem::exists(directory),"directory must be new");
        require(std::filesystem::create_directories(directory),"cannot create output directory");created=true;
        environment();
        // Test executable only: retain a reproducer for the quarantined mode.
        if(graph_history) {
#ifdef _WIN32
            _putenv_s("GGML_CUDA_DISABLE_GRAPHS", "");
#else
            unsetenv("GGML_CUDA_DISABLE_GRAPHS");
#endif
        }
        report["cuda_graphs"]=graph_history;
        ggml_backend_load_all();
        if(reference_only) reference_probe(report["cases"],directory);
        else {
        for(int replay=0;replay<(history_only?32:1);++replay) {
        for(int variant:{0,1,2}) {
            const bool quant=variant==1,wide=variant==2;
            const auto path=(directory/(wide ? "hy3-wide.gguf" : quant ? "hy3-mixed.gguf" : "hy3-f32.gguf")).string();
            write_synthetic_hy3(path,quant,wide);
            auto resident=load(path,true,false,true),mapped=load(path,false,false,true);
            for(int batch:{1,4,17}) {
                if(wide && batch!=1) continue;
                auto a=context(resident.get(),2048,batch),b=context(mapped.get(),2048,batch);
                strata_hy3_sync_mode(1);strata_hy3_sync_reset();
                const auto reference=run(a.get(),19,batch),candidate=run(b.get(),19,batch);
                Observer observer;strata_hy3_sync_observer(Observer::copy,&observer);
                strata_hy3_sync_mode(2);strata_hy3_sync_reset();
                const auto actual=run(b.get(),19,batch);
                const std::string name=(history_only?"replay="+std::to_string(replay)+"/":"")+
                    std::string(wide ? "wide" : quant ? "mixed" : "f32")+"/batch="+std::to_string(batch);
                const auto native=compare(name+"/native_vs_resident",candidate,reference);
                report["cases"].push_back(native);
                auto c=compare(name+"/pinned_file_vs_native",actual,candidate);c["transport"]=stats();c["verified_gpu_bytes"]=observer.bytes;
                const auto s=strata_hy3_sync_snapshot();
                require(observer.bytes==s.h2d_bytes && observer.ranges==s.ranges && observer.bytes>0,"observer byte coverage mismatch");
                require(s.source_bytes==s.h2d_bytes && s.staging_bytes==16*1024*1024 && !s.rejected_cpu_nodes && !s.rejected_full_copies,"staging budget/audit failed");
                if(wide) require(s.chunks>s.ranges,"fixture must cross staging boundary");
                report["cases"].push_back(c);strata_hy3_sync_observer(nullptr,nullptr);
                if(history_only && !native["pass"].get<bool>()) {
                    report["diagnostic"]=diagnose(resident.get(),mapped.get(),a.get(),b.get(),batch,reference,candidate);
                    report["status"]="fail";
                    throw std::runtime_error("history replay mismatch captured");
                }
            }
        }
        }
        if(!history_only) {
        const auto path=(directory/"hy3-mixed.gguf").string();
        {
            auto model=load(path,false,false,true);auto ctx=context(model.get());
            strata_hy3_sync_mode(2);auto reference=run(ctx.get(),19,17);
            std::atomic<bool> cancel{true};strata_hy3_sync_cancel(&cancel);
            bool rejected=false;try {run(ctx.get(),19,17);}catch(const std::runtime_error &){rejected=true;}
            strata_hy3_sync_cancel(nullptr);
            report["cases"].push_back({{"name","cancel_before_compute"},{"pass",rejected}});
            report["cases"].push_back(compare("recovery_after_cancel",run(ctx.get(),19,17),reference));
        }
        {
            auto model=load(path,false,true,true);auto ctx=context(model.get());
            strata_hy3_sync_mode(2);strata_hy3_sync_reset();bool rejected=false;
            try {run(ctx.get(),19,17);}catch(const std::runtime_error &){rejected=true;}
            const auto s=strata_hy3_sync_snapshot();
            report["cases"].push_back({{"name","reject_cpu_tensor_math_before_copy"},
                {"pass",rejected && s.rejected_cpu_nodes>0 && s.h2d_bytes==0}});
        }
        bool rejected=false;try {inspect(path,false);}catch(const std::runtime_error &){rejected=true;}
        report["cases"].push_back({{"name","fixture_requires_explicit_flag"},{"pass",rejected}});
        const auto truncated=directory/"truncated.gguf";
        std::filesystem::copy_file(path,truncated);std::filesystem::resize_file(truncated,std::filesystem::file_size(truncated)-32);
        rejected=false;try {inspect(truncated.string(),true);}catch(const std::runtime_error &){rejected=true;}
        report["cases"].push_back({{"name","reject_truncated_before_weight_allocation"},{"pass",rejected}});
        failures(report["cases"],path);
        cache_checks(report["cases"],directory);
        pipeline_checks(report["cases"],directory);
        pipeline_checks(report["cases"],directory,true);
        delivery_profile_checks(report["cases"],directory);
        host_cache_checks(report["cases"],directory);
        gpu_prefill_policy_checks(report["cases"],directory);
        }
        }
        bool pass=true;for(const auto & c:report["cases"]) pass&=c["pass"].get<bool>();
        report["status"]=pass ? "pass" : "fail";
    } catch(const std::exception & e) {report["error"]=e.what();}
    strata_hy3_sync_release();ggml_quantize_free();llama_backend_free();
    report["case_count"]=report["cases"].size();
    if(created) {std::ofstream out(directory/"runtime-report.json");out<<report.dump(2)<<'\n';if(!out) return 1;}
    std::cout<<report.dump(2)<<'\n';return report["status"]=="pass" ? 0 : 1;
}
