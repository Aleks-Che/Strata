// Hy3 scheduler integration; transport, cache policy and CUDA event tracing are
// reused unchanged. Plans contain only actual router IDs for one live route.
#include "../common/expert_pipeline.hpp"
#include "../step35/gpu_trace.hpp"
#include <set>
struct Hy3PlanInput { const ggml_tensor * tensor; int expert; };
struct Hy3PlanMatrix {
    const ggml_tensor * tensor;
    step35::MatrixKey key;
    size_t offset,bytes;
    bool hit;
};
static thread_local std::unique_ptr<StrataExpertPipeline> hy3_pipeline;
static thread_local StrataExpertPipeline::Counters hy3_pipeline_accounted;
static thread_local std::unique_ptr<step35::GpuTrace> hy3_trace;
static thread_local std::vector<Hy3PlanMatrix> hy3_plan;
static thread_local size_t hy3_plan_next=0;
static thread_local std::unique_ptr<step35::ExpertCache::PlanPins> hy3_plan_pins;
static thread_local int hy3_pipeline_readers=0,hy3_pipeline_chunk=4;
static thread_local bool hy3_pipeline_recreate=false;
static thread_local bool hy3_pipeline_batch=false;
static bool hy3_pipeline_enabled() {return bool(hy3_pipeline);}
static void hy3_account_pipeline() {
    if(!hy3_pipeline) return;
    const auto c=hy3_pipeline->counters();auto & a=hy3_pipeline_accounted;auto & s=hy3_sync_stats;
    s.source_bytes+=c.file_bytes+c.mmap_bytes-a.file_bytes-a.mmap_bytes;
    s.h2d_bytes+=c.h2d_bytes-a.h2d_bytes;s.d2d_bytes+=c.d2d_bytes-a.d2d_bytes;
    s.source_ms+=double(c.read_us-a.read_us)/1000;s.chunks+=c.chunks-a.chunks;
    s.pipeline_groups+=c.groups-a.groups;s.pipeline_chunks+=c.chunks-a.chunks;
    s.pipeline_unused_bytes+=c.unused_bytes-a.unused_bytes;
    s.pipeline_wait_us+=c.consumer_wait_us-a.consumer_wait_us;
    s.pipeline_slot_wait_us+=c.slot_wait_us-a.slot_wait_us;
    s.pipeline_submit_us+=c.submit_us-a.submit_us;
    s.staging_bytes=c.pinned_bytes;s.pipeline_device_bytes=c.device_ring_bytes;
    s.pipeline_read_peak=std::max(s.pipeline_read_peak,c.read_peak);
    s.pipeline_reader_owned=c.reader_owned_bytes;s.pipeline_queued=c.queued_bytes;a=c;
}
static void hy3_end_plan(bool cancelled) {
    if(!hy3_pipeline && hy3_plan.empty()) return;
    const bool incomplete=!cancelled && hy3_plan_next!=hy3_plan.size();
    std::exception_ptr failure;
    // Finish even on reader errors; release pins only after all cache consumers.
    if(hy3_pipeline) try {hy3_pipeline->finish(!cancelled);}catch(...) {failure=std::current_exception();}
    if(hy3_cache_stream) hy3_cuda(cudaStreamSynchronize(hy3_cache_stream));
    hy3_plan_pins.reset();hy3_plan.clear();hy3_plan_next=0;
    if(hy3_cache) hy3_cache->trim(hy3_cache->budget());
    hy3_account_pipeline();
    if(failure) std::rethrow_exception(failure);
    if(incomplete) throw std::runtime_error("incomplete Hy3 router plan");
}
void strata_hy3_pipeline_config(int readers,int chunk_mib,int trace_graphs,bool batch_copy) {
    if(readers<0 || readers>2 || (chunk_mib!=4 && chunk_mib!=8 && chunk_mib!=16) ||
        trace_graphs<0 || trace_graphs>128 || (!readers && (trace_graphs || batch_copy)))
        throw std::runtime_error("invalid Hy3 pipeline configuration");
    if(readers && (hy3_sync_mode!=2 || !hy3_cache || !hy3_cache_stream || hy3_tensors.empty()))
        throw std::runtime_error("Hy3 pipeline requires registered pinned cache runtime");
    hy3_end_plan(true);hy3_pipeline.reset();hy3_trace.reset();hy3_pipeline_accounted={};
    hy3_pipeline_recreate=false;hy3_pipeline_readers=readers;hy3_pipeline_chunk=chunk_mib;
    hy3_pipeline_batch=batch_copy;
    if(!readers) return;
    const size_t bytes=size_t(chunk_mib)<<20,capacity=StrataExpertPipeline::slots*bytes;
    strata_hy3_memory_check(capacity+(256<<20));
    auto m=step35::memory_sample();
    if(std::min(m.ram_free,hy3_test_limits.ram_available)<m.ram_total/20+(64<<20)+capacity)
        throw std::runtime_error("Hy3 pipeline pinned ring exceeds RAM budget");
    if(hy3_test_limits.fail_next_pinned_allocation) {
        hy3_test_limits.fail_next_pinned_allocation=false;hy3_cuda(cudaErrorMemoryAllocation);
    }
    if(trace_graphs) hy3_trace=std::make_unique<step35::GpuTrace>(trace_graphs);
    auto * trace=hy3_trace.get();
    // Capture by value: producer threads must never access the owner's TLS.
    auto fail_after=std::make_shared<int>(hy3_test_limits.fail_pipeline_submission_after);
    hy3_test_limits.fail_pipeline_submission_after=-1;
    StrataExpertPipeline::CopyObserver observer=[trace,fail_after](cudaStream_t stream,bool begin,size_t) {
        if(begin && *fail_after>=0) {
            if(!*fail_after) throw std::runtime_error("injected Hy3 pipeline submission failure");
            --*fail_after;
        }
        if(trace) trace->record(stream,nullptr,begin);
    };
    int gpu=-1;hy3_cuda(cudaGetDevice(&gpu));
    // Explicit native-file mode matches the synchronous baseline's source path.
    hy3_pipeline=std::make_unique<StrataExpertPipeline>(gpu,bytes,false,readers,1,observer,readers);
    if(hy3_staging) {hy3_cuda(cudaFreeHost(hy3_staging));hy3_staging=nullptr;}
    hy3_cache->refresh();hy3_account_pipeline();
}
void strata_hy3_trace_write(const char * path) {
    if(!hy3_trace || !path || !*path) return;
    std::ofstream out(path);if(!out) throw std::runtime_error("cannot open Hy3 trace output");
    out<<hy3_trace->snapshot().dump(2)<<'\n';if(!out) throw std::runtime_error("cannot write Hy3 trace output");
}
static bool hy3_begin_plan(const std::vector<Hy3PlanInput> & inputs,bool decode) {
    try {
        // Shared transport retains reader errors. Recreate only after drain.
        if(hy3_pipeline_recreate) strata_hy3_pipeline_config(hy3_pipeline_readers,hy3_pipeline_chunk,0,hy3_pipeline_batch);
        hy3_end_plan(false);
        if(inputs.empty() || inputs.size()>4096) throw std::runtime_error("invalid Hy3 router plan size");
        std::set<step35::MatrixKey> unique;
        std::vector<step35::MatrixKey> keys;
        std::vector<StrataExpertSlice> misses;
        for(const auto & input:inputs) {
            const auto * src=input.tensor;
            const auto entry=hy3_tensors.find(src->data);
            if(entry==hy3_tensors.end()) throw std::runtime_error("unregistered Hy3 plan tensor");
            const auto & t=entry->second;
            const auto file=strata_expert_file::find(src->data,ggml_nbytes(src));
            if(!file || file!=t.file || file->offset_of(src->data,ggml_nbytes(src))!=t.offset ||
                t.name!=src->name || t.type!=src->type || !ggml_is_contiguous(src) ||
                !std::equal(t.shape.begin(),t.shape.end(),src->ne) ||
                !std::equal(t.strides.begin(),t.strides.end(),src->nb) || input.expert<0 || input.expert>=t.shape[2])
                throw std::runtime_error("Hy3 plan tensor identity changed");
            const size_t offset=size_t(input.expert)*t.strides[2];
            const size_t bytes=t.strides[2]+(input.expert+1<t.shape[2]?std::min<size_t>(512,t.strides[2]):0);
            if(offset>ggml_nbytes(src) || bytes>ggml_nbytes(src)-offset) throw std::runtime_error("Hy3 plan outside tensor");
            const step35::MatrixKey key{hy3_generation,t.id,uint32_t(input.expert)};
            if(!unique.insert(key).second) throw std::runtime_error("duplicate Hy3 planned matrix");
            const bool hit=hy3_cache->contains(key,bytes);
            hy3_plan.push_back({src,key,offset,bytes,hit});keys.push_back(key);
            if(!hit) misses.push_back({static_cast<const uint8_t *>(src->data)+offset,bytes,decode});
        }
        hy3_plan_pins=hy3_cache->protect(keys);hy3_pipeline->start(misses);return true;
    } catch(const std::exception & error) {
        hy3_end_plan(true);hy3_pipeline_recreate=true;
        GGML_LOG_ERROR("STRATA_HY3 plan: %s\n",error.what());return false;
    }
}
static bool hy3_pipeline_copy(ggml_backend_t backend,ggml_tensor * dst,const ggml_tensor * src,size_t offset,size_t bytes) {
    try {
        if(!ggml_are_same_shape(src,dst) || src->type!=dst->type || !ggml_is_contiguous(dst) ||
            ggml_backend_buffer_is_host(dst->buffer) || !src->nb[2] || offset%src->nb[2] ||
            offset>ggml_nbytes(src) || bytes>ggml_nbytes(src)-offset)
            throw std::runtime_error("invalid Hy3 pipeline destination/range");
        const size_t count=bytes/src->nb[2],tail=bytes%src->nb[2];
        if(!count || (tail && tail!=std::min<size_t>(512,src->nb[2]))) throw std::runtime_error("invalid Hy3 pipeline padding");
        // Scratch can alias preceding activations. Ring prefetch never writes it.
        ggml_backend_synchronize(backend);
        for(size_t i=0;i<count;++i) {
            if(hy3_cancelled()) {hy3_end_plan(true);return false;}
            if(hy3_plan_next>=hy3_plan.size()) throw std::runtime_error("Hy3 pipeline has no next matrix");
            const auto & p=hy3_plan[hy3_plan_next];
            if(p.tensor!=src || p.offset!=offset+i*src->nb[2] || p.bytes>ggml_nbytes(dst)-p.offset)
                throw std::runtime_error("Hy3 copy differs from router plan");
            auto * dest=static_cast<uint8_t *>(dst->data)+p.offset;
            auto * cached=hy3_cache->get(p.key,p.bytes);
            if(bool(cached)!=p.hit) throw std::runtime_error("Hy3 planned cache residency changed");
            if(cached) {
                hy3_cuda(cudaMemcpyAsync(dest,cached,p.bytes,cudaMemcpyDeviceToDevice,hy3_cache_stream));
                hy3_sync_stats.d2d_bytes+=p.bytes;
            } else if(!hy3_pipeline->transfer(dest,static_cast<const uint8_t *>(src->data)+p.offset,p.bytes,hy3_cache_stream))
                throw std::runtime_error("Hy3 pipeline source differs from plan");
            hy3_cuda(cudaStreamSynchronize(hy3_cache_stream));++hy3_sync_stats.pipeline_copy_fences;
            if(!cached) if(auto * slot=hy3_cache->admit(p.key,p.bytes)) {
                if(hy3_test_limits.fail_next_cache_fill) {
                    hy3_test_limits.fail_next_cache_fill=false;throw std::runtime_error("injected Hy3 cache fill failure");
                }
                hy3_cuda(cudaMemcpyAsync(slot,dest,p.bytes,cudaMemcpyDeviceToDevice,hy3_cache_stream));
                hy3_cuda(cudaStreamSynchronize(hy3_cache_stream));++hy3_sync_stats.pipeline_copy_fences;
                hy3_sync_stats.cache_fill_bytes+=p.bytes;
            }
            ++hy3_sync_stats.ranges;++hy3_plan_next;
            if(hy3_observer) hy3_observer(hy3_owner,backend,dst,src,p.offset,p.bytes);
        }
        hy3_account_pipeline();return true;
    } catch(const std::exception & error) {
        hy3_end_plan(true);hy3_pipeline_recreate=true;
        if(hy3_cache) hy3_cache->trim(0);
        GGML_LOG_ERROR("STRATA_HY3 pipeline copy: %s\n",error.what());return false;
    }
}
// One scheduler tensor per batch. All cache sources are plan-pinned; newly
// admitted destinations remain pinned until this consumer stream has drained.
static bool hy3_pipeline_copy_tensor(ggml_backend_t backend,ggml_tensor * dst,const ggml_tensor * src) {
    struct Cancelled {};
    struct Pending {
        std::vector<std::unique_ptr<step35::ExpertCache::PlanPins>> fills;
        bool complete=false;
        ~Pending() {
            // Destructor body runs before fills releases its pins, also during
            // exception unwinding. Scratch and cache copies share this stream.
            if(!complete) cudaStreamSynchronize(hy3_cache_stream);
        }
    };
    try {
        if(!hy3_pipeline || !hy3_plan_pins || !ggml_are_same_shape(src,dst) || src->type!=dst->type ||
            !ggml_is_contiguous(dst) || ggml_backend_buffer_is_host(dst->buffer) || !src->nb[2] ||
            hy3_plan_next>=hy3_plan.size() || hy3_plan[hy3_plan_next].tensor!=src)
            throw std::runtime_error("invalid Hy3 batched tensor/plan");
        ggml_backend_synchronize(backend); // Scratch may alias preceding activations.
        Pending pending;
        const size_t begin=hy3_plan_next;
        const auto start=std::chrono::steady_clock::now();
        while(hy3_plan_next<hy3_plan.size() && hy3_plan[hy3_plan_next].tensor==src) {
            if(hy3_cancelled()) throw Cancelled{};
            const auto & p=hy3_plan[hy3_plan_next];
            if(p.offset>ggml_nbytes(dst) || p.bytes>ggml_nbytes(dst)-p.offset)
                throw std::runtime_error("Hy3 batched destination outside tensor");
            auto * dest=static_cast<uint8_t *>(dst->data)+p.offset;
            auto * cached=hy3_cache->get(p.key,p.bytes);
            if(bool(cached)!=p.hit) throw std::runtime_error("Hy3 batched cache residency changed");
            if(cached) {
                hy3_cuda(cudaMemcpyAsync(dest,cached,p.bytes,cudaMemcpyDeviceToDevice,hy3_cache_stream));
                hy3_sync_stats.d2d_bytes+=p.bytes;
            } else {
                if(!hy3_pipeline->transfer(dest,static_cast<const uint8_t *>(src->data)+p.offset,p.bytes,hy3_cache_stream))
                    throw std::runtime_error("Hy3 batched source differs from plan");
                if(auto * slot=hy3_cache->admit(p.key,p.bytes)) {
                    pending.fills.push_back(hy3_cache->protect({p.key}));
                    hy3_sync_stats.pipeline_pending_fills_peak=std::max<uint64_t>(
                        hy3_sync_stats.pipeline_pending_fills_peak,pending.fills.size());
                    if(hy3_test_limits.fail_next_cache_fill) {
                        hy3_test_limits.fail_next_cache_fill=false;throw std::runtime_error("injected Hy3 cache fill failure");
                    }
                    // Queue fill after ring-to-scratch D2D on the same stream.
                    hy3_cuda(cudaMemcpyAsync(slot,dest,p.bytes,cudaMemcpyDeviceToDevice,hy3_cache_stream));
                    hy3_sync_stats.cache_fill_bytes+=p.bytes;
                }
            }
            ++hy3_sync_stats.ranges;++hy3_plan_next;
            const int queued=int(hy3_plan_next-begin);
            if(hy3_test_limits.fail_batch_enqueue_after==queued) {
                hy3_test_limits.fail_batch_enqueue_after=-1;
                throw std::runtime_error("injected Hy3 batched enqueue failure");
            }
            if(hy3_test_limits.batch_cancel && hy3_test_limits.cancel_batch_enqueue_after==queued) {
                hy3_test_limits.cancel_batch_enqueue_after=-1;
                hy3_test_limits.batch_cancel->store(true);
            }
        }
        if(hy3_cancelled()) throw Cancelled{};
        hy3_cuda(cudaStreamSynchronize(hy3_cache_stream));pending.complete=true;
        ++hy3_sync_stats.pipeline_copy_batches;++hy3_sync_stats.pipeline_copy_fences;
        hy3_sync_stats.pipeline_batch_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        if(hy3_observer) for(size_t i=begin;i<hy3_plan_next;++i) {
            const auto & p=hy3_plan[i];hy3_observer(hy3_owner,backend,dst,src,p.offset,p.bytes);
        }
        hy3_account_pipeline();return true;
    } catch(const Cancelled &) {
        hy3_end_plan(true);return false;
    } catch(const std::exception & error) {
        hy3_end_plan(true);hy3_pipeline_recreate=true;
        if(hy3_cache) hy3_cache->trim(0);
        GGML_LOG_ERROR("STRATA_HY3 batched copy: %s\n",error.what());return false;
    }
}
static ggml_status hy3_graph_compute(ggml_backend_t backend,ggml_cgraph * graph) {
    if(hy3_trace) hy3_trace->record(nullptr,backend,true);
    const auto result=ggml_backend_graph_compute_async(backend,graph);
    if(hy3_trace) hy3_trace->record(nullptr,backend,false);
    return result;
}
struct Hy3PlanScope {
    bool closed=false;
    bool finish() {
        try {hy3_end_plan(false);if(hy3_trace)hy3_trace->finish(false);closed=true;return true;}
        catch(const std::exception & e) {GGML_LOG_ERROR("STRATA_HY3 plan finish: %s\n",e.what());return false;}
    }
    ~Hy3PlanScope() {
        if(!closed) try {hy3_end_plan(true);if(hy3_trace)hy3_trace->finish(true);}
        catch(const std::exception & e) {GGML_LOG_ERROR("STRATA_HY3 plan cleanup: %s\n",e.what());}
    }
};
