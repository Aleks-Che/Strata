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
static void step_account_pipeline() {
    if (!step_pipeline) return;
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
    const bool incomplete=!cancelled && step_plan_next!=step_plan.size();
    // Workers must release every source view before model lifetime can end.
    // Cache reads on step_cache_stream have completed before each copy returns.
    if (step_pipeline) step_pipeline->finish(!cancelled);
    step_plan_pins.reset();step_plan.clear();step_plan_next=0;
    if (step_cache) step_cache->trim(step_cache->budget());
    step_account_pipeline();
    if (incomplete) throw std::runtime_error("incomplete Step router plan");
}
extern "C" void strata_step_pipeline_config(int readers,int chunk_mib,int trace_graphs) {
    step_end_plan(true);step_pipeline.reset();step_trace.reset();step_pipeline_accounted={};
    if (readers<0 || readers>2 || (chunk_mib!=4 && chunk_mib!=8 && chunk_mib!=16) || trace_graphs<0 || trace_graphs>128)
        throw std::runtime_error("invalid Step pipeline configuration");
    if (readers && (step_sync_mode!=2 || !step_cache || !step_cache_stream))
        throw std::runtime_error("Step pipeline requires a registered pinned-copy runtime");
    if (trace_graphs) step_trace=std::make_unique<step35::GpuTrace>(trace_graphs);
    if (readers) {
        const size_t bytes=size_t(chunk_mib)<<20,capacity=StrataExpertPipeline::slots*bytes;
        const auto m=step35::memory_sample();
        if (m.gpu_free<m.gpu_total/20+(256<<20)+capacity || m.ram_free<m.ram_total/20+(64<<20)+capacity)
            throw std::runtime_error("Step pipeline ring exceeds available memory budget");
        int gpu=-1;step35::cuda_check(cudaGetDevice(&gpu));
        auto * trace=step_trace.get();
        StrataExpertPipeline::CopyObserver observer;
        if (trace) observer=[trace](cudaStream_t stream,bool begin,size_t){trace->record(stream,nullptr,begin);};
        step_pipeline=std::make_unique<StrataExpertPipeline>(gpu,bytes,false,readers,0,observer,readers,false);
        ggml_backend_buffer_free(step_sync_staging);step_sync_staging=nullptr;
        step_cache->refresh();
    }
}
extern "C" void strata_step_trace_write(const char * path) {
    if (!step_trace || !path || !*path) return;
    std::ofstream out(path);if (!out) throw std::runtime_error("cannot open Step trace output");
    out<<step_trace->snapshot().dump(2)<<'\n';if(!out)throw std::runtime_error("cannot write Step trace output");
}
static bool step_begin_plan(const std::vector<StepPlanInput> & inputs,bool decode) {
    try {
        step_end_plan(false);
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
            if (cached) step_sync_stats.d2d_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            if (!cached) if (auto * slot=step_cache_admit(p.key,p.bytes)) {
                start=std::chrono::steady_clock::now();
                step35::cuda_check(cudaMemcpyAsync(slot,dest,p.bytes,cudaMemcpyDeviceToDevice,step_cache_stream));
                step35::cuda_check(cudaStreamSynchronize(step_cache_stream));
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
