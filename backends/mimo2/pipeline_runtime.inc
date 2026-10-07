// True router lookahead, never prediction of a future layer. Shared transport
// writes only its own bounded device ring, not scheduler scratch/activations.
#include "../common/expert_pipeline.hpp"
#include "gpu_trace.hpp"
#include <fstream>
#include <set>
#include <optional>
struct MimoPlanInput {const ggml_tensor *tensor;int expert;};
struct MimoPlanMatrix {const ggml_tensor *tensor;mimo2::MatrixKey key;size_t offset,bytes,tail;bool hit,tail_hit;};
static thread_local std::unique_ptr<StrataExpertPipeline> mimo_pipeline;
static thread_local StrataExpertPipeline::Counters mimo_pipeline_accounted;
static thread_local std::unique_ptr<mimo2::GpuTrace> mimo_trace;
static thread_local std::vector<MimoPlanMatrix> mimo_plan;
static thread_local size_t mimo_plan_next=0;
static thread_local std::unique_ptr<mimo2::ExpertCache::PlanPins> mimo_plan_pins;
static thread_local int mimo_readers=0,mimo_chunk_mib=8;
static thread_local bool mimo_pipeline_recreate=false;
static thread_local bool mimo_pipeline_batch=false;
static thread_local bool mimo_fill_batch=false;
static thread_local int mimo_fill_failure=-1;
int strata_mimo_cache_fill_batch_mode() {
    const char *value=std::getenv("STRATA_MIMO_CACHE_FILL_BATCH");
    if(!value || std::strcmp(value,"0")==0)return 0;
    if(std::strcmp(value,"1")==0)return 1;
    throw std::runtime_error("STRATA_MIMO_CACHE_FILL_BATCH must be 0 or 1");
}
int strata_mimo_pipeline_batch_mode() {
    const char *value=std::getenv("STRATA_MIMO_PIPELINE_BATCH");
    if(!value)return 1;
    if(std::strcmp(value,"0")==0)return 0;
    if(std::strcmp(value,"1")==0)return 1;
    throw std::runtime_error("STRATA_MIMO_PIPELINE_BATCH must be 0 or 1");
}
static thread_local std::shared_ptr<int> mimo_submission_failure;
static bool mimo_pipeline_enabled() {return bool(mimo_pipeline);}
static void mimo_pipeline_account() {
    if(!mimo_pipeline)return;
    mimo_stats.pipeline_batch=mimo_pipeline_batch;
    mimo_stats.cache_fill_batch=mimo_fill_batch;
    const auto c=mimo_pipeline->counters();auto &a=mimo_pipeline_accounted;
    mimo_stats.source_bytes+=c.file_bytes+c.mmap_bytes-a.file_bytes-a.mmap_bytes;
    mimo_stats.pipeline_file_bytes+=c.file_bytes-a.file_bytes;
    mimo_stats.pipeline_mmap_bytes+=c.mmap_bytes-a.mmap_bytes;
    mimo_stats.pipeline_d2d_bytes+=c.d2d_bytes-a.d2d_bytes;
    mimo_stats.h2d_bytes+=c.h2d_bytes-a.h2d_bytes;
    mimo_stats.source_ms+=double(c.read_us-a.read_us)/1000;
    mimo_stats.chunks+=c.chunks-a.chunks;
    mimo_stats.pipeline_groups+=c.groups-a.groups;mimo_stats.pipeline_chunks+=c.chunks-a.chunks;
    mimo_stats.pipeline_unused_bytes+=c.unused_bytes-a.unused_bytes;
    mimo_stats.pipeline_consumer_wait_us+=c.consumer_wait_us-a.consumer_wait_us;
    mimo_stats.pipeline_slot_wait_us+=c.slot_wait_us-a.slot_wait_us;
    mimo_stats.pipeline_submit_us+=c.submit_us-a.submit_us;
    mimo_stats.pipeline_read_peak=std::max(mimo_stats.pipeline_read_peak,c.read_peak);
    mimo_stats.staging_bytes=c.pinned_bytes+(mimo_staging?mimo_staging_limit:0);mimo_stats.pipeline_device_bytes=c.device_ring_bytes;
    mimo_stats.pipeline_reader_owned=c.reader_owned_bytes;mimo_stats.pipeline_queued=c.queued_bytes;a=c;
}
static void mimo_pipeline_reset_counters() {
    if(mimo_pipeline) {mimo_pipeline_accounted=mimo_pipeline->counters();mimo_pipeline_account();}
}
static void mimo_end_plan(bool aborted) {
    const bool incomplete=!aborted && mimo_plan_next!=mimo_plan.size();
    std::exception_ptr error;
    if(mimo_pipeline)try {mimo_pipeline->finish(!aborted);}catch(...) {error=std::current_exception();}
    if(mimo_cache_stream)mimo_cuda(cudaStreamSynchronize(mimo_cache_stream));
    mimo_plan_pins.reset();mimo_plan.clear();mimo_plan_next=0;
    if(mimo_cache)mimo_cache->trim(mimo_cache->budget());
    mimo_pipeline_account();
    if(error)std::rethrow_exception(error);
    if(incomplete)throw std::runtime_error("incomplete MiMo router plan");
}
static void mimo_pipeline_shutdown() {
    mimo_end_plan(true);mimo_pipeline.reset();mimo_trace.reset();
    mimo_pipeline_accounted={};mimo_pipeline_recreate=false;mimo_readers=0;mimo_submission_failure.reset();mimo_pipeline_batch=false;
    mimo_fill_batch=false;mimo_fill_failure=-1;
}
static void mimo_create_pipeline() {
    mimo_pipeline.reset();mimo_pipeline_accounted={};
    const size_t capacity=StrataExpertPipeline::slots*(size_t(mimo_chunk_mib)<<20);
    strata_mimo_memory(capacity+(64ull<<20),capacity+(64ull<<20));
    auto *trace=mimo_trace.get();auto fail=mimo_submission_failure;
    StrataExpertPipeline::CopyObserver observer=[trace,fail](cudaStream_t stream,bool begin,size_t) {
        // Serialized by the transport's copy_mutex; no producer touches owner TLS.
        if(begin && *fail>=0) {if(!(*fail)--)throw std::runtime_error("injected MiMo pipeline submission error");}
        if(trace)trace->record(stream,nullptr,begin);
    };
    int device=0;mimo_cuda(cudaGetDevice(&device));
    mimo_pipeline=std::make_unique<StrataExpertPipeline>(device,size_t(mimo_chunk_mib)<<20,false,
        mimo_readers,mimo_mmap_reader?0:1,observer,mimo_readers,false);
    if(mimo_staging) {mimo_cuda(cudaFreeHost(mimo_staging));mimo_staging=nullptr;}
    mimo_pipeline_recreate=false;strata_mimo_memory();mimo_pipeline_account();
}
void strata_mimo_pipeline_config(int readers,int chunk_mib,int trace_graphs) {
    if(readers<0 || readers>2 || (chunk_mib!=4 && chunk_mib!=8 && chunk_mib!=16) ||
       trace_graphs<0 || trace_graphs>128 || (!readers && trace_graphs))throw std::runtime_error("invalid MiMo pipeline configuration");
    if(readers && (mimo_mode!=2 || !mimo_cache || !mimo_cache_stream || mimo_tensors.empty()))
        throw std::runtime_error("MiMo pipeline requires registered managed cache runtime");
    mimo_pipeline_shutdown();mimo_readers=readers;mimo_chunk_mib=chunk_mib;
    if(!readers)return;
    mimo_pipeline_batch=strata_mimo_pipeline_batch_mode()!=0;
    mimo_fill_batch=mimo_pipeline_batch && strata_mimo_cache_fill_batch_mode()!=0;
    if(trace_graphs)mimo_trace=std::make_unique<mimo2::GpuTrace>(trace_graphs);
    mimo_submission_failure=std::make_shared<int>(-1);mimo_create_pipeline();
}
void strata_mimo_test_pipeline_failure(int submissions) {
    if(!mimo_pipeline || !mimo_plan.empty())throw std::runtime_error("inject failure only between graphs");
    *mimo_submission_failure=submissions;
}
void strata_mimo_test_fill_failure(int submissions) {
    if(!mimo_pipeline || !mimo_fill_batch || !mimo_plan.empty())throw std::runtime_error("inject fill failure only between batched graphs");
    mimo_fill_failure=submissions;
}
void strata_mimo_trace_write(const char *path) {
    if(!mimo_trace || !path || !*path)return;
    std::ofstream file(path);if(!file)throw std::runtime_error("cannot open MiMo trace");
    file<<mimo_trace->snapshot().dump(2)<<'\n';if(!file)throw std::runtime_error("cannot write MiMo trace");
}
static bool mimo_begin_plan(const std::vector<MimoPlanInput> &inputs,bool decode) {
    try {
        if(mimo_pipeline_recreate) {mimo_end_plan(true);mimo_create_pipeline();}
        else mimo_end_plan(false);
        if(inputs.empty() || inputs.size()>4096)throw std::runtime_error("invalid MiMo route plan size");
        // Trim before taking pins, so future cache hits cannot change residency.
        strata_mimo_memory();
        std::set<mimo2::MatrixKey> unique;std::vector<mimo2::MatrixKey> keys;
        std::vector<StrataExpertSlice> misses;
        for(size_t index=0;index<inputs.size();++index) {
            const auto &input=inputs[index];
            const auto *src=input.tensor;auto found=mimo_tensors.find(src);
            if(found==mimo_tensors.end())throw std::runtime_error("unregistered MiMo plan tensor");
            const auto &id=found->second;
            if(src->data!=id.data || src->type!=id.type || id.name!=src->name || src->nb[2]!=id.stride ||
               src->ne[2]!=id.experts || ggml_nbytes(src)!=id.bytes || !ggml_is_contiguous(src) ||
               input.expert<0 || input.expert>=id.experts)throw std::runtime_error("MiMo plan identity changed");
            const mimo2::MatrixKey key{mimo_generation,id.id,uint32_t(input.expert)};
            if(!unique.insert(key).second)throw std::runtime_error("duplicate MiMo planned matrix");
            const size_t offset=size_t(input.expert)*id.stride;
            const bool hit=mimo_cache->contains(key,id.stride);
            const bool consecutive=index+1<inputs.size() && inputs[index+1].tensor==src && inputs[index+1].expert==input.expert+1;
            const size_t tail=!consecutive && input.expert+1<id.experts?std::min<size_t>(512,id.stride):0;
            const mimo2::MatrixKey next{mimo_generation,id.id,uint32_t(input.expert+1)};
            const bool tail_hit=hit && tail && mimo_cache->contains(next,id.stride);
            mimo_plan.push_back({src,key,offset,id.stride,tail,hit,tail_hit});keys.push_back(key);
            if(tail_hit)keys.push_back(next);
            auto *address=static_cast<const uint8_t *>(src->data)+offset;
            if(!hit)misses.push_back({address,id.stride+tail,decode});
            else if(tail && !tail_hit)misses.push_back({address+id.stride,tail,decode});
        }
        mimo_plan_pins=mimo_cache->protect(keys);mimo_pipeline->start(misses);return true;
    } catch(const std::exception &e) {
        mimo_end_plan(true);mimo_pipeline_recreate=true;
        GGML_LOG_ERROR("STRATA_MIMO_PLAN: %s\n",e.what());return false;
    }
}
static bool mimo_pipeline_copy(ggml_backend_t backend,ggml_tensor *dst,const ggml_tensor *src,size_t offset,size_t bytes) {
    try {
        if(mimo_cancelled()) {mimo_end_plan(true);return false;}
        // begin_plan samples/clamps once per route. The fixed ring and cache
        // allocations remain within that cap; resampling each expert range
        // adds driver calls without changing the plan's allocation bound.
        if(!mimo_plan_pins || !ggml_are_same_shape(src,dst) || src->type!=dst->type ||
           !ggml_is_contiguous(dst) || ggml_backend_buffer_is_host(dst->buffer) || !src->nb[2] ||
           offset%src->nb[2] || offset>ggml_nbytes(dst) || bytes>ggml_nbytes(dst)-offset)
            throw std::runtime_error("invalid MiMo pipeline destination");
        const size_t count=bytes/src->nb[2],tail=bytes%src->nb[2];
        if(!count || tail>512)throw std::runtime_error("invalid MiMo pipeline padding");
        ++mimo_stats.ranges;mimo_stats.requested_bytes+=bytes;
        // Scratch aliases preceding activations. Prefetch never writes scratch;
        // wait for its previous consumer before delivering from the device ring.
        ++mimo_stats.pipeline_scratch_fences;ggml_backend_synchronize(backend);
        for(size_t n=0;n<count;++n) {
            if(mimo_cancelled()) {mimo_end_plan(true);return false;}
            if(mimo_plan_next>=mimo_plan.size())throw std::runtime_error("MiMo copy exceeds plan");
            const auto &p=mimo_plan[mimo_plan_next];
            if(p.tensor!=src || p.offset!=offset+n*src->nb[2] || p.tail!=(n+1==count?tail:0))
                throw std::runtime_error("MiMo copy differs from route plan");
            auto *dest=static_cast<uint8_t *>(dst->data)+p.offset;
            if(!mimo_prefill || mimo_prefill_admit)mimo_cache->record(p.key);
            void *hit=mimo_cache->get(p.key,p.bytes,!mimo_prefill || mimo_prefill_admit);
            if(bool(hit)!=p.hit)throw std::runtime_error("MiMo planned residency changed");
            if(hit) {
                mimo_cuda(cudaMemcpyAsync(dest,hit,p.bytes,cudaMemcpyDeviceToDevice,mimo_cache_stream));
                mimo_stats.cache_hit_bytes+=p.bytes;
                if(p.tail) {
                    if(p.tail_hit) {
                        auto *guard=mimo_cache->get({p.key.generation,p.key.tensor,p.key.expert+1},p.bytes,false);
                        if(!guard)throw std::runtime_error("MiMo guard residency changed");
                        mimo_cuda(cudaMemcpyAsync(dest+p.bytes,guard,p.tail,cudaMemcpyDeviceToDevice,mimo_cache_stream));
                        mimo_stats.cache_hit_bytes+=p.tail;
                    } else {
                        if(!mimo_pipeline->transfer(dest+p.bytes,static_cast<const uint8_t *>(src->data)+p.offset+p.bytes,p.tail,mimo_cache_stream))
                            throw std::runtime_error("MiMo guard differs from plan");
                        mimo_stats.pipeline_delivered_bytes+=p.tail;
                    }
                }
                ++mimo_stats.pipeline_copy_fences;mimo_cuda(cudaStreamSynchronize(mimo_cache_stream));
            }
            else {
                if(!mimo_pipeline->transfer(dest,static_cast<const uint8_t *>(src->data)+p.offset,p.bytes+p.tail,mimo_cache_stream))
                    throw std::runtime_error("MiMo pipeline source differs from plan");
                ++mimo_stats.pipeline_copy_fences;mimo_cuda(cudaStreamSynchronize(mimo_cache_stream));
                mimo_stats.pipeline_delivered_bytes+=p.bytes+p.tail;
                if((!mimo_prefill || mimo_prefill_admit) && !mimo_cancelled())mimo_cache->store(p.key,p.bytes,[&](void *slot) {
                    mimo_d2d(slot,dest,p.bytes);mimo_stats.cache_fill_bytes+=p.bytes;
                });
            }
            ++mimo_plan_next;
            if(mimo_observer)mimo_observer(backend,dst,src,p.offset,p.bytes+p.tail,mimo_owner);
        }
        mimo_pipeline_account();return !mimo_cancelled();
    } catch(const std::exception &e) {
        mimo_end_plan(true);mimo_pipeline_recreate=true;ggml_backend_synchronize(backend);
        GGML_LOG_ERROR("STRATA_MIMO_PIPELINE: %s\n",e.what());return false;
    }
}
// One scheduler input, one delivery fence. All cache sources are route-pinned.
// No cache admission/eviction or scratch consumer can run until that fence.
// Replay admission order afterwards. Grouped fills reserve invisible slots,
// enqueue only survivors, then fence before publication or failure cleanup.
static bool mimo_pipeline_copy_tensor(ggml_backend_t backend,ggml_tensor *dst,const ggml_tensor *src) {
    try {
        if(mimo_cancelled()) {mimo_end_plan(true);return false;}
        if(!mimo_pipeline || !mimo_plan_pins || !ggml_are_same_shape(src,dst) || src->type!=dst->type ||
           !ggml_is_contiguous(dst) || !dst->data || ggml_backend_buffer_is_host(dst->buffer) ||
           !src->nb[2] || mimo_plan_next>=mimo_plan.size() || mimo_plan[mimo_plan_next].tensor!=src)
            throw std::runtime_error("invalid MiMo batched tensor/plan");
        ++mimo_stats.pipeline_scratch_fences;ggml_backend_synchronize(backend);
        const auto start=std::chrono::steady_clock::now();
        const size_t begin=mimo_plan_next;size_t end=begin,previous_end=0;
        while(end<mimo_plan.size() && mimo_plan[end].tensor==src) {
            if(mimo_cancelled())throw std::runtime_error("MiMo batch cancelled");
            const auto &p=mimo_plan[end];
            if(p.bytes!=src->nb[2] || p.offset%p.bytes || p.tail>512 ||
               p.offset>ggml_nbytes(dst) || p.bytes>ggml_nbytes(dst)-p.offset ||
               p.tail>ggml_nbytes(dst)-p.offset-p.bytes || (end>begin && p.offset<previous_end))
                throw std::runtime_error("MiMo batched destination outside tensor");
            if(end==begin || p.offset!=previous_end)++mimo_stats.ranges;
            previous_end=p.offset+p.bytes;mimo_stats.requested_bytes+=p.bytes+p.tail;
            auto *dest=static_cast<uint8_t *>(dst->data)+p.offset;
            auto *source=static_cast<const uint8_t *>(src->data)+p.offset;
            auto *hit=mimo_cache->get(p.key,p.bytes,false);
            if(bool(hit)!=p.hit)throw std::runtime_error("MiMo batched residency changed");
            if(hit) {
                mimo_cuda(cudaMemcpyAsync(dest,hit,p.bytes,cudaMemcpyDeviceToDevice,mimo_cache_stream));
                mimo_stats.cache_hit_bytes+=p.bytes;
                if(p.tail) {
                    if(p.tail_hit) {
                        auto *guard=mimo_cache->get({p.key.generation,p.key.tensor,p.key.expert+1},p.bytes,false);
                        if(!guard)throw std::runtime_error("MiMo batched guard residency changed");
                        mimo_cuda(cudaMemcpyAsync(dest+p.bytes,guard,p.tail,cudaMemcpyDeviceToDevice,mimo_cache_stream));
                        mimo_stats.cache_hit_bytes+=p.tail;
                    } else {
                        if(!mimo_pipeline->transfer(dest+p.bytes,source+p.bytes,p.tail,mimo_cache_stream))
                            throw std::runtime_error("MiMo batched guard differs from plan");
                        mimo_stats.pipeline_delivered_bytes+=p.tail;
                    }
                }
            } else {
                if(!mimo_pipeline->transfer(dest,source,p.bytes+p.tail,mimo_cache_stream))
                    throw std::runtime_error("MiMo batched source differs from plan");
                mimo_stats.pipeline_delivered_bytes+=p.bytes+p.tail;
            }
            ++end;
        }
        ++mimo_stats.pipeline_copy_fences;mimo_cuda(cudaStreamSynchronize(mimo_cache_stream));
        ++mimo_stats.pipeline_copy_batches;
        const bool admit=!mimo_prefill || mimo_prefill_admit;
        if(admit) {
            std::optional<mimo2::ExpertCache::FillBatch> fills;
            if(mimo_fill_batch)fills.emplace(*mimo_cache,[] {
                ++mimo_stats.pipeline_copy_fences;++mimo_stats.pipeline_fill_batches;
                mimo_cuda(cudaStreamSynchronize(mimo_cache_stream));
            },end-begin);
            for(size_t i=begin;i<end && !mimo_cancelled();++i) {
                const auto &p=mimo_plan[i];
                mimo_cache->record(p.key);
                if(p.hit)mimo_cache->touch(p.key);
                else if(fills) {
                    const auto *source=static_cast<uint8_t *>(dst->data)+p.offset;const size_t bytes=p.bytes;
                    fills->store(p.key,p.bytes,[source,bytes](void *slot) {
                        if(mimo_cancelled())throw std::runtime_error("MiMo cache fill cancelled");
                        mimo_cuda(cudaMemcpyAsync(slot,source,bytes,cudaMemcpyDeviceToDevice,mimo_cache_stream));
                        ++mimo_stats.pipeline_fill_submissions;mimo_stats.cache_fill_bytes+=bytes;
                        if(mimo_fill_failure>=0 && !mimo_fill_failure--)throw std::runtime_error("injected MiMo cache fill error");
                    });
                } else mimo_cache->store(p.key,p.bytes,[&](void *slot) {
                    mimo_d2d(slot,static_cast<uint8_t *>(dst->data)+p.offset,p.bytes);mimo_stats.cache_fill_bytes+=p.bytes;
                });
            }
            if(fills && !mimo_cancelled()) {
                const auto fill_start=std::chrono::steady_clock::now();fills->finish();
                mimo_stats.d2d_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-fill_start).count();
            }
        }
        mimo_plan_next=end;
        // Diagnostics see only completed data, just as in per-matrix mode.
        if(mimo_observer)for(size_t i=begin;i<end && !mimo_cancelled();++i) {
            const auto &p=mimo_plan[i];mimo_observer(backend,dst,src,p.offset,p.bytes+p.tail,mimo_owner);
        }
        mimo_stats.pipeline_batch_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        mimo_pipeline_account();return !mimo_cancelled();
    } catch(const std::exception &e) {
        // Drain delivery before plan pins or scratch can be released on failure.
        mimo_end_plan(true);mimo_pipeline_recreate=true;ggml_backend_synchronize(backend);
        GGML_LOG_ERROR("STRATA_MIMO_BATCH: %s\n",e.what());return false;
    }
}
static ggml_status mimo_graph_compute(ggml_backend_t backend,ggml_cgraph *graph) {
    if(mimo_trace)mimo_trace->record(nullptr,backend,true);
    const auto result=ggml_backend_graph_compute_async(backend,graph);
    if(mimo_trace)mimo_trace->record(nullptr,backend,false);return result;
}
struct MimoPlanScope {
    bool closed=false;
    bool finish() {
        try {mimo_end_plan(false);if(mimo_trace)mimo_trace->finish(false);closed=true;return true;}
        catch(const std::exception &e) {mimo_pipeline_recreate=true;GGML_LOG_ERROR("STRATA_MIMO_FINISH: %s\n",e.what());return false;}
    }
    ~MimoPlanScope() {
        if(!closed)try {mimo_end_plan(true);if(mimo_trace)mimo_trace->finish(true);}
        catch(const std::exception &e) {GGML_LOG_ERROR("STRATA_MIMO_DRAIN: %s\n",e.what());}
    }
};
