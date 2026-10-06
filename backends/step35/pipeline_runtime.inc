// Included after the sync baseline, before the generated scheduler function.
struct StepPlanInput {const ggml_tensor * tensor; int expert;};
struct StepPlanMatrix {
    const ggml_tensor * tensor;
    step35::MatrixKey key;
    size_t offset,bytes;
    bool hit;
};
static thread_local std::vector<StepPlanMatrix> step_plan;
static thread_local size_t step_plan_next=0;
static thread_local std::unique_ptr<step35::ExpertCache::PlanPins> step_plan_pins;
static thread_local bool step_pipeline_batch=false;
static thread_local bool step_pipeline_early=false;
static thread_local bool step_pipeline_write_combined=false;
static thread_local int step_pipeline_readers=0,step_pipeline_chunk=0;
static void step_account_pipeline() {
    if (!step_pipeline) return;
    if (step_host_copy) {
        const auto c=step_host_copy->snapshot();auto &a=step_host_accounted;auto &s=step_sync_stats;
        s.host_copy_calls+=c.copies-a.copies;s.host_copy_bytes+=c.bytes-a.bytes;
        s.host_copy_wall_ns+=c.wall_ns-a.wall_ns;s.host_copy_cycles+=c.cycles-a.cycles;
        s.host_copy_slow+=c.slow-a.slow;a=c;
    }
    const auto c=step_pipeline->counters();auto & s=step_sync_stats;auto & a=step_pipeline_accounted;
    s.source_bytes+=(c.file_bytes+c.mmap_bytes)-(a.file_bytes+a.mmap_bytes);
    s.h2d_bytes+=c.h2d_bytes-a.h2d_bytes;s.d2d_bytes+=c.d2d_bytes-a.d2d_bytes;
    s.source_ms+=double(c.read_us-a.read_us)/1000;s.chunks+=c.chunks-a.chunks;
    s.pipeline_groups+=c.groups-a.groups;s.pipeline_chunks+=c.chunks-a.chunks;
    s.pipeline_unused_bytes+=c.unused_bytes-a.unused_bytes;
    s.pipeline_wait_us+=c.consumer_wait_us-a.consumer_wait_us;
    s.pipeline_slot_wait_us+=c.slot_wait_us-a.slot_wait_us;s.pipeline_submit_us+=c.submit_us-a.submit_us;
    s.staging_bytes=c.pinned_bytes;s.pipeline_device_bytes=c.device_ring_bytes;
    s.pipeline_read_peak=std::max(s.pipeline_read_peak,c.read_peak);
    a=c;
}
static void step_end_plan(bool cancelled) {
    step35::CpuTimer timer(step_cache_profiling?&step_sync_stats.plan_end_ms:nullptr);
    const bool incomplete=!cancelled && step_plan_next!=step_plan.size();
    // Workers must release every source view before model lifetime can end.
    // Cache reads on step_cache_stream have completed before each copy returns.
    if (step_pipeline) step_pipeline->finish(!cancelled);
    step_plan_pins.reset();step_plan.clear();step_plan_next=0;
    if (step_cache) step_cache->trim(step_cache->budget());
    step_account_pipeline();
    if (incomplete) throw std::runtime_error("incomplete Step router plan");
}
extern "C" void strata_step_pipeline_config(int readers,int chunk_mib,int trace_graphs) noexcept(false) {
    strata_step_pipeline_config_ex(readers,chunk_mib,trace_graphs,false,false);
}
extern "C" void strata_step_pipeline_config_ex(int readers,int chunk_mib,int trace_graphs,bool early_refill,bool batch_copy) noexcept(false) {
    strata_step_pipeline_config_staging(readers,chunk_mib,trace_graphs,early_refill,batch_copy,false);
}
extern "C" void strata_step_pipeline_config_staging(int readers,int chunk_mib,int trace_graphs,bool early_refill,bool batch_copy,bool write_combined) noexcept(false) {
    strata_step_pipeline_config_copy(readers,chunk_mib,trace_graphs,early_refill,batch_copy,write_combined,0,false);
}
extern "C" void strata_step_pipeline_config_copy(int readers,int chunk_mib,int trace_graphs,bool early_refill,bool batch_copy,bool write_combined,int host_copy,bool profile) noexcept(false) {
    if (readers<0 || readers>2 || (chunk_mib!=4 && chunk_mib!=8 && chunk_mib!=16) || trace_graphs<0 || trace_graphs>128)
        throw std::runtime_error("invalid Step pipeline configuration");
    if (readers && (step_sync_mode!=2 || !step_cache || !step_cache_stream))
        throw std::runtime_error("Step pipeline requires a registered pinned-copy runtime");
    if (!readers && (early_refill || batch_copy || write_combined)) throw std::runtime_error("Step copy options require pipeline readers");
    if (host_copy<0 || host_copy>1 || (host_copy==1 && !step35::host_copy_avx2_available()) || (!readers && (host_copy || profile)))
        throw std::runtime_error("unsupported Step host copy configuration");
    step_end_plan(true);
    if (step_pipeline && !step_trace && !trace_graphs && readers==step_pipeline_readers &&
        chunk_mib==step_pipeline_chunk && early_refill==step_pipeline_early &&
        write_combined==step_pipeline_write_combined) {
        // Batch delivery changes no buffer or producer state. Preserve the ring
        // and cache when switching this option between requests.
        step_host_copy->configure(host_copy,profile);step_pipeline_batch=batch_copy;return;
    }
    const bool replacing=bool(step_pipeline);
    step_pipeline.reset();step_trace.reset();step_pipeline_accounted={};step_pipeline_batch=false;
    step_host_copy.reset();step_host_accounted={};
    if (trace_graphs) step_trace=std::make_unique<step35::GpuTrace>(trace_graphs);
    if (readers) {
        const size_t bytes=size_t(chunk_mib)<<20,capacity=StrataExpertPipeline::slots*bytes;
        auto m=step35::memory_sample();
        const size_t needed=m.gpu_total/20+(256<<20)+capacity;
        if (replacing && m.gpu_free<needed && step_cache->resident_bytes()) {
            // A full cache may occupy the ring's replacement headroom. Drain
            // has already released all pins; trim only the shortfall plus a
            // bounded margin for allocation rounding/global-sample lag.
            const size_t release=std::min(step_cache->resident_bytes(),needed-m.gpu_free+(64<<20));
            step_cache->trim(step_cache->resident_bytes()-release);
            for (int attempt=0;attempt<10;++attempt) {
                m=step35::memory_sample();if(m.gpu_free>=needed)break;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
        if (m.gpu_free<m.gpu_total/20+(256<<20)+capacity || m.ram_free<m.ram_total/20+(64<<20)+capacity)
            throw std::runtime_error("Step pipeline ring exceeds available memory budget");
        int gpu=-1;step35::cuda_check(cudaGetDevice(&gpu));
        auto * trace=step_trace.get();
        StrataExpertPipeline::CopyObserver observer;
        if (trace) observer=[trace](cudaStream_t stream,bool begin,size_t){trace->record(stream,nullptr,begin);};
        step_host_copy=std::make_shared<step35::HostCopyState>();step_host_copy->configure(host_copy,profile);
        auto host=step_host_copy;
        step_pipeline=std::make_unique<StrataExpertPipeline>(gpu,bytes,write_combined,readers,0,observer,readers,early_refill,
            [host](void *dst,const void *src,size_t n){(*host)(dst,src,n);});
        ggml_backend_buffer_free(step_sync_staging);step_sync_staging=nullptr;
        step_cache->refresh();
        step_pipeline_batch=batch_copy;
        step_pipeline_early=early_refill;step_pipeline_readers=readers;step_pipeline_chunk=chunk_mib;
        step_pipeline_write_combined=write_combined;
    }
}
extern "C" void strata_step_trace_write(const char * path) noexcept(false) {
    if (!step_trace || !path || !*path) return;
    std::ofstream out(path);if (!out) throw std::runtime_error("cannot open Step trace output");
    out<<step_trace->snapshot().dump(2)<<'\n';if(!out)throw std::runtime_error("cannot write Step trace output");
}
static bool step_begin_plan(const std::vector<StepPlanInput> & inputs,bool decode) {
    try {
        step_end_plan(false);
        step35::CpuTimer timer(step_cache_profiling?&step_sync_stats.plan_build_ms:nullptr);
        if (inputs.empty() || inputs.size()>4096) throw std::runtime_error("invalid Step router plan size");
        std::set<step35::MatrixKey> unique;
        std::vector<step35::MatrixKey> keys;
        std::vector<StrataExpertSlice> misses;
        for (const auto & input:inputs) {
            const auto * t=input.tensor;
            const auto registered=step_tensors.find(t->data);
            if (registered==step_tensors.end()) throw std::runtime_error("unregistered Step plan tensor");
            const auto & r=registered->second;
            if (!step_expert_weight(t) || r.name!=t->name || r.type!=t->type || !ggml_is_contiguous(t) ||
                r.columns!=t->ne[0] || r.rows!=t->ne[1] || r.experts!=t->ne[2] || t->ne[3]!=1 || r.stride!=t->nb[2] ||
                input.expert<0 || input.expert>=r.experts) throw std::runtime_error("invalid Step planned expert");
            const step35::MatrixKey key{step_generation,r.id,uint32_t(input.expert)};
            if (!unique.insert(key).second) throw std::runtime_error("duplicate Step planned matrix");
            const size_t offset=size_t(input.expert)*r.stride;
            const size_t bytes=r.stride+(input.expert+1<r.experts?std::min<size_t>(512,r.stride):0);
            if (offset>ggml_nbytes(t) || bytes>ggml_nbytes(t)-offset) throw std::runtime_error("Step plan outside tensor");
            const bool hit=step_cache->contains(key,bytes);
            step_plan.push_back({t,key,offset,bytes,hit});keys.push_back(key);
            if (!hit) misses.push_back({static_cast<const uint8_t *>(t->data)+offset,bytes,decode});
        }
        step_plan_pins=step_cache->protect(keys);
        step_pipeline->start(misses);return true;
    } catch (const std::exception & error) {
        step_end_plan(true);GGML_LOG_ERROR("STRATA_STEP plan failed: %s\n",error.what());return false;
    }
}
static bool step_pipeline_copy(ggml_backend_t backend,ggml_tensor * dst,const ggml_tensor * src,size_t offset,size_t bytes) {
    try {
        if (!ggml_are_same_shape(src,dst) || src->type!=dst->type || !ggml_is_contiguous(dst) ||
            !src->nb[2] || offset%src->nb[2] || offset>ggml_nbytes(src) || bytes>ggml_nbytes(src)-offset)
            throw std::runtime_error("invalid Step pipeline destination/range");
        const size_t count=bytes/src->nb[2],tail=bytes%src->nb[2];
        if (!count || (tail && tail!=std::min<size_t>(512,src->nb[2]))) throw std::runtime_error("invalid Step pipeline padding");
        // Keep native scratch dependency and cache-read synchronization. The
        // independent ring can still prepare up/down while gate is computing.
        ggml_backend_synchronize(backend);
        for (size_t i=0;i<count;++i) {
            if (step_plan_next>=step_plan.size()) throw std::runtime_error("Step pipeline has no next matrix");
            const auto & p=step_plan[step_plan_next];
            if (p.tensor->data!=src->data || p.offset!=offset+i*src->nb[2] || p.bytes>ggml_nbytes(dst)-p.offset)
                throw std::runtime_error("Step copy order differs from router plan");
            auto * dest=static_cast<uint8_t *>(dst->data)+p.offset;
            auto * cached=step_cache->get(p.key,p.bytes);
            if (bool(cached)!=p.hit) throw std::runtime_error("Step planned cache residency changed");
            auto start=std::chrono::steady_clock::now();
            if (cached) {
                step35::cuda_check(cudaMemcpyAsync(dest,cached,p.bytes,cudaMemcpyDeviceToDevice,step_cache_stream));
                step_sync_stats.d2d_bytes+=p.bytes;
            } else {
                if (!step_pipeline->transfer(dest,static_cast<const uint8_t *>(src->data)+p.offset,p.bytes,step_cache_stream))
                    throw std::runtime_error("Step pipeline source differs from plan");
            }
            step35::cuda_check(cudaStreamSynchronize(step_cache_stream));
            ++step_sync_stats.pipeline_copy_fences;
            if (cached) step_sync_stats.d2d_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            if (!cached) if (auto * slot=step_cache_admit(p.key,p.bytes)) {
                start=std::chrono::steady_clock::now();
                step35::cuda_check(cudaMemcpyAsync(slot,dest,p.bytes,cudaMemcpyDeviceToDevice,step_cache_stream));
                step35::cuda_check(cudaStreamSynchronize(step_cache_stream));
                ++step_sync_stats.pipeline_copy_fences;
                step_sync_stats.d2d_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
                step_sync_stats.cache_fill_bytes+=p.bytes;
            }
            ++step_sync_stats.ranges;++step_plan_next;
            if (step_sync_observer) step_sync_observer(step_sync_owner,backend,dst,src,p.offset,p.bytes);
        }
        step_account_pipeline();return true;
    } catch(const std::exception & error) {
        cudaStreamSynchronize(step_cache_stream);step_end_plan(true);
        if (step_cache) step_cache->trim(0);
        GGML_LOG_ERROR("STRATA_STEP pipeline copy failed: %s\n",error.what());return false;
    }
}
// Copies all selected experts of ONE scheduler input before its compute split.
// Native scratch may alias previous activations, so keep the backend fence.
// PlanPins protect all cached sources; local pins protect newly admitted fills
// from subsequent admissions/refresh/trim until the single final stream fence.
static bool step_pipeline_copy_tensor(ggml_backend_t backend,ggml_tensor * dst,const ggml_tensor * src) {
    struct Pending {
        std::vector<std::unique_ptr<step35::ExpertCache::PlanPins>> fills;
        bool complete=false;
        ~Pending() {
            // Also on exceptions: drain before releasing pins and trimming.
            if (!complete) cudaStreamSynchronize(step_cache_stream);
        }
    };
    try {
        if (!step_pipeline || !step_plan_pins || !ggml_are_same_shape(src,dst) || src->type!=dst->type ||
            !ggml_is_contiguous(dst) || !src->nb[2] || step_plan_next>=step_plan.size() ||
            step_plan[step_plan_next].tensor->data!=src->data)
            throw std::runtime_error("invalid Step batched tensor/plan");
        ggml_backend_synchronize(backend);
        Pending pending;
        const size_t begin=step_plan_next;
        const auto start=std::chrono::steady_clock::now();
        size_t previous_end=0;
        while (step_plan_next<step_plan.size() && step_plan[step_plan_next].tensor->data==src->data) {
            const auto & p=step_plan[step_plan_next];
            if (p.offset>ggml_nbytes(dst) || p.bytes>ggml_nbytes(dst)-p.offset)
                throw std::runtime_error("Step batched destination outside tensor");
            auto * dest=static_cast<uint8_t *>(dst->data)+p.offset;
            auto * cached=step_cache->get(p.key,p.bytes);
            if (bool(cached)!=p.hit) throw std::runtime_error("Step batched cache residency changed");
            if (cached) {
                step35::cuda_check(cudaMemcpyAsync(dest,cached,p.bytes,cudaMemcpyDeviceToDevice,step_cache_stream));
                step_sync_stats.d2d_bytes+=p.bytes;
            } else {
                if (!step_pipeline->transfer(dest,static_cast<const uint8_t *>(src->data)+p.offset,p.bytes,step_cache_stream))
                    throw std::runtime_error("Step batched source differs from plan");
                if (auto * slot=step_cache_admit(p.key,p.bytes)) {
                    pending.fills.push_back(step_cache->protect({p.key}));
                    // Same stream: the ring-to-scratch copy completes first.
                    step35::cuda_check(cudaMemcpyAsync(slot,dest,p.bytes,cudaMemcpyDeviceToDevice,step_cache_stream));
                    step_sync_stats.cache_fill_bytes+=p.bytes;
                }
            }
            // Match native consecutive-range accounting (overlapping tails).
            step_sync_stats.requested_bytes+=p.bytes-(previous_end>p.offset ? previous_end-p.offset : 0);
            previous_end=p.offset+p.bytes;
            ++step_sync_stats.ranges;++step_plan_next;
        }
        step35::cuda_check(cudaStreamSynchronize(step_cache_stream));
        pending.complete=true;
        ++step_sync_stats.pipeline_copy_batches;++step_sync_stats.pipeline_copy_fences;
        // Batched wall time includes source waits/admission; not GPU D2D time.
        step_sync_stats.pipeline_batch_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        if (step_sync_observer) for (size_t i=begin;i<step_plan_next;++i) {
            const auto & p=step_plan[i];
            step_sync_observer(step_sync_owner,backend,dst,src,p.offset,p.bytes);
        }
        step_account_pipeline();return true;
    } catch (const std::exception & error) {
        cudaStreamSynchronize(step_cache_stream);step_end_plan(true);
        if (step_cache) step_cache->trim(0);
        GGML_LOG_ERROR("STRATA_STEP batched copy failed: %s\n",error.what());return false;
    }
}
static ggml_status step_graph_compute(ggml_backend_t backend,ggml_cgraph * graph) {
    if (step_trace) step_trace->record(nullptr,backend,true);
    const auto result=ggml_backend_graph_compute_async(backend,graph);
    if (step_trace) step_trace->record(nullptr,backend,false);
    return result;
}
struct StepPlanScope {
    bool closed=false;
    bool finish() {
        try {step_end_plan(false);if(step_trace)step_trace->finish(false);closed=true;return true;}
        catch(const std::exception & e){GGML_LOG_ERROR("STRATA_STEP plan finish: %s\n",e.what());return false;}
    }
    ~StepPlanScope() {
        if (!closed) try {step_end_plan(true);if(step_trace)step_trace->finish(true);}
        catch(const std::exception & e){GGML_LOG_ERROR("STRATA_STEP plan cleanup: %s\n",e.what());}
    }
};
