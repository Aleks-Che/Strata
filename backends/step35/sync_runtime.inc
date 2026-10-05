// Included only in the generated Step ggml-backend.cpp.
#include "sync_runtime.h"
#include "expert_cache.hpp"
#include "../common/expert_pipeline.hpp"
#include "gpu_trace.hpp"
#include <set>
#include <chrono>
#include <cstring>
#include <stdexcept>
static thread_local int step_sync_mode = 0;
static thread_local int step_request_phase = 0;
static thread_local bool step_prefill_admission = true;
static thread_local strata_step_sync_stats step_sync_stats{};
static thread_local ggml_backend_buffer_t step_sync_staging = nullptr;
static thread_local strata_step_copy_observer step_sync_observer = nullptr;
static thread_local void * step_sync_owner = nullptr;
static constexpr size_t step_staging_limit = 16 * 1024 * 1024;
struct StepTensorIdentity {
    uint32_t id;
    std::string name, shard;
    uint64_t offset;
    ggml_type type;
    int64_t columns, rows, experts;
    size_t stride;
};
static thread_local std::unique_ptr<step35::ExpertCache> step_cache;
static thread_local std::map<const void *,StepTensorIdentity> step_tensors;
static thread_local std::string step_model_identity;
static thread_local uint64_t step_generation=0;
static thread_local cudaStream_t step_cache_stream=nullptr;
static thread_local std::unique_ptr<StrataExpertPipeline> step_pipeline;
static thread_local std::unique_ptr<step35::GpuTrace> step_trace;
static thread_local StrataExpertPipeline::Counters step_pipeline_accounted{};
static void step_end_plan(bool cancelled);
static void step_account_pipeline();
static bool step_pipeline_copy(ggml_backend_t,ggml_tensor *,const ggml_tensor *,size_t,size_t);
extern "C" void strata_step_cache_begin(const char * identity,size_t cap) {
    step_request_phase=0;step_prefill_admission=true;
    step_end_plan(true);step_pipeline.reset();step_trace.reset();step_pipeline_accounted={};
    if (step_cache_stream) step35::cuda_check(cudaStreamSynchronize(step_cache_stream));
    step_cache.reset(); step_tensors.clear(); step_model_identity=identity?identity:"";
    ++step_generation;
    if (cap && step_model_identity.empty()) throw std::runtime_error("Step cache needs a live model identity");
    // A zero-cap controller still checks global RAM/VRAM at decode boundaries.
    step_cache=std::make_unique<step35::ExpertCache>(cap);
    if (!step_cache_stream) step35::cuda_check(cudaStreamCreateWithFlags(&step_cache_stream,cudaStreamNonBlocking));
    step_cache->refresh();
}
extern "C" void strata_step_cache_register(const ggml_tensor * tensor,const char * shard,uint64_t offset) {
    if (!step_cache || !tensor || !tensor->data || !shard || !*shard ||
        !ggml_is_contiguous(tensor) || tensor->ne[3]!=1 || tensor->ne[2]<=0 ||
        offset>UINT64_MAX-ggml_nbytes(tensor)) throw std::runtime_error("invalid Step cache tensor registration");
    const auto id=uint32_t(step_tensors.size());
    if (!step_tensors.emplace(tensor->data,StepTensorIdentity{id,tensor->name,shard,offset,tensor->type,
            tensor->ne[0],tensor->ne[1],tensor->ne[2],tensor->nb[2]}).second)
        throw std::runtime_error("duplicate Step expert tensor registration");
}
extern "C" void strata_step_cache_refresh() {if (step_cache) step_cache->refresh();}
extern "C" void strata_step_cache_trim(size_t bytes) {if (step_cache) step_cache->trim(bytes);}
extern "C" void strata_step_cache_prefill(bool enabled) {step_prefill_admission=enabled;}
extern "C" void strata_step_cache_reuse(bool enabled) {
    step_end_plan(false);
    if (step_cache) step_cache->set_reuse_allocations(enabled);
}
extern "C" int strata_step_request_phase(int phase) {
    if (phase<0 || phase>2) throw std::invalid_argument("invalid Step request phase");
    const int previous=step_request_phase;step_request_phase=phase;return previous;
}
static void * step_cache_admit(const step35::MatrixKey & key,size_t bytes) {
    // get() has already recorded frequency and checked existing entries.
    // Only allocation/eviction/fill is suppressed; hits remain usable.
    if (step_request_phase==1 && !step_prefill_admission) {
        ++step_sync_stats.prefill_admission_skips;
        step_sync_stats.prefill_admission_skip_bytes+=bytes;
        return nullptr;
    }
    return step_cache->admit(key,bytes);
}
extern "C" void strata_step_sync_mode(int mode) {
    if (mode < 0 || mode > 2) throw std::invalid_argument("invalid Step copy mode");
    step_sync_mode = mode;
}
extern "C" void strata_step_sync_reset() {
    step_sync_stats = {};
    if (step_sync_staging) step_sync_stats.staging_bytes = ggml_backend_buffer_get_size(step_sync_staging);
    if (step_cache) step_cache->reset_counters();
    if (step_pipeline) step_pipeline_accounted=step_pipeline->counters();
}
extern "C" strata_step_sync_stats strata_step_sync_snapshot() {
    step_account_pipeline();
    auto s=step_sync_stats;
    if (step_cache) {
        auto c=step_cache->counters(); auto m=step_cache->memory();
        s.cache_hits=c.hits;s.cache_misses=c.misses;s.cache_evictions=c.evictions;s.cache_bypasses=c.bypasses;
        s.cache_oom=c.oom;s.cache_rejected=c.rejected;s.cache_bytes=step_cache->resident_bytes();s.cache_limit=step_cache->budget();
        s.cache_allocations=c.allocations;s.cache_reuses=c.reuses;
        s.gpu_free=m.gpu_free;s.gpu_total=m.gpu_total;s.ram_free=m.ram_free;s.ram_total=m.ram_total;s.memory_samples=c.samples;
    }
    return s;
}
extern "C" void strata_step_sync_observer(strata_step_copy_observer observer, void * owner) {
    step_sync_observer = observer; step_sync_owner = owner;
}
extern "C" void strata_step_sync_release() {
    step_end_plan(true);step_pipeline.reset();step_trace.reset();step_pipeline_accounted={};
    if (step_cache_stream) {cudaStreamSynchronize(step_cache_stream);cudaStreamDestroy(step_cache_stream);step_cache_stream=nullptr;}
    step_cache.reset();step_tensors.clear();step_model_identity.clear();
    ggml_backend_buffer_free(step_sync_staging); step_sync_staging = nullptr;
    step_sync_mode = 0; step_sync_observer = nullptr; step_sync_owner = nullptr;
    step_request_phase=0;step_prefill_admission=true;
}
static bool step_expert_weight(const ggml_tensor * tensor) {
    const char * suffix = "_exps.weight";
    const size_t length = std::strlen(tensor->name), tail = std::strlen(suffix);
    return length >= tail && std::strcmp(tensor->name + length - tail, suffix) == 0;
}
static bool step_copy_uncached(ggml_backend_t backend, ggml_tensor * dst, const ggml_tensor * src,
                            size_t offset, size_t bytes) {
    try {
        if (!step_expert_weight(src) || !src->data || src->type != dst->type ||
            offset > ggml_nbytes(src) || bytes > ggml_nbytes(src)-offset ||
            offset > ggml_nbytes(dst) || bytes > ggml_nbytes(dst)-offset)
            throw std::runtime_error("invalid Step selected expert range");
        if (!step_sync_staging) {
            auto * type = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(backend));
            if (!type) throw std::runtime_error("Step requires pinned host staging");
            step_sync_staging = ggml_backend_buft_alloc_buffer(type, step_staging_limit);
            if (!step_sync_staging || !ggml_backend_buffer_is_host(step_sync_staging) ||
                ggml_backend_buffer_get_type(step_sync_staging) != type) {
                ggml_backend_buffer_free(step_sync_staging); step_sync_staging = nullptr;
                throw std::runtime_error("Step pinned allocation failed; pageable fallback rejected");
            }
            step_sync_stats.staging_bytes = ggml_backend_buffer_get_size(step_sync_staging);
        }
        // Fence both scratch consumers and each upload before reusing staging.
        ggml_backend_synchronize(backend);
        auto * staging = static_cast<uint8_t *>(ggml_backend_buffer_get_base(step_sync_staging));
        using clock = std::chrono::steady_clock;
        const auto elapsed = [](clock::time_point start) {
            return std::chrono::duration<double, std::milli>(clock::now()-start).count();
        };
        ++step_sync_stats.ranges;
        for (size_t done = 0; done < bytes;) {
            const size_t count = std::min(bytes-done, step_staging_limit);
            auto start = clock::now();
            std::memcpy(staging, static_cast<const uint8_t *>(src->data)+offset+done, count);
            step_sync_stats.source_ms += elapsed(start);
            start = clock::now();
            ggml_backend_tensor_set_async(backend, dst, staging, offset+done, count);
            ggml_backend_synchronize(backend);
            step_sync_stats.h2d_ms += elapsed(start);
            step_sync_stats.source_bytes += count; step_sync_stats.h2d_bytes += count;
            ++step_sync_stats.chunks; done += count;
        }
        if (step_sync_observer) step_sync_observer(step_sync_owner, backend, dst, src, offset, bytes);
        return true;
    } catch (const std::exception & error) {
        GGML_LOG_ERROR("STRATA_STEP selected-copy failed: %s\n", error.what());
        return false;
    }
}
static bool step_copy_range(ggml_backend_t backend,ggml_tensor * dst,const ggml_tensor * src,size_t offset,size_t bytes) {
    step_sync_stats.requested_bytes+=bytes;
    if (step_pipeline) return step_pipeline_copy(backend,dst,src,offset,bytes);
    if (!step_cache || !step_cache->budget()) return step_copy_uncached(backend,dst,src,offset,bytes);
    try {
        const auto registered=step_tensors.find(src->data);
        if (registered==step_tensors.end()) throw std::runtime_error("unregistered Step expert tensor");
        const auto & t=registered->second;
        if (!step_expert_weight(src) || t.name!=src->name || t.type!=src->type || dst->type!=src->type ||
            t.columns!=src->ne[0] || t.rows!=src->ne[1] || t.experts!=src->ne[2] || t.stride!=src->nb[2] ||
            src->ne[3]!=1 || !ggml_are_same_shape(src,dst) || !ggml_is_contiguous(dst) ||
            !t.stride || offset%t.stride || offset>ggml_nbytes(src) || bytes>ggml_nbytes(src)-offset ||
            offset>ggml_nbytes(dst) || bytes>ggml_nbytes(dst)-offset)
            throw std::runtime_error("Step selected range identity/bounds mismatch");
        const size_t count=bytes/t.stride, tail=bytes%t.stride;
        if (!count || (tail && tail!=std::min<size_t>(512,t.stride))) throw std::runtime_error("invalid Step selected range padding");
        ggml_backend_synchronize(backend); // Native scratch may alias preceding activations.
        for (size_t i=0;i<count;++i) {
            const size_t expert=offset/t.stride+i, at=expert*t.stride;
            const size_t stored=t.stride+(expert+1<size_t(t.experts)?std::min<size_t>(512,t.stride):0);
            const step35::MatrixKey key{step_generation,t.id,uint32_t(expert)};
            auto * cached=step_cache->get(key,stored);
            using clock=std::chrono::steady_clock;
            if (cached) {
                const auto start=clock::now();
                step35::cuda_check(cudaMemcpyAsync(static_cast<uint8_t *>(dst->data)+at,cached,stored,cudaMemcpyDeviceToDevice,step_cache_stream));
                step35::cuda_check(cudaStreamSynchronize(step_cache_stream));
                step_sync_stats.d2d_ms+=std::chrono::duration<double,std::milli>(clock::now()-start).count();
                step_sync_stats.d2d_bytes+=stored;++step_sync_stats.ranges;
                if (step_sync_observer) step_sync_observer(step_sync_owner,backend,dst,src,at,stored);
            } else {
                if (!step_copy_uncached(backend,dst,src,at,stored)) throw std::runtime_error("Step cache miss transfer failed");
                if (auto * slot=step_cache_admit(key,stored)) {
                    const auto start=clock::now();
                    step35::cuda_check(cudaMemcpyAsync(slot,static_cast<const uint8_t *>(dst->data)+at,stored,cudaMemcpyDeviceToDevice,step_cache_stream));
                    step35::cuda_check(cudaStreamSynchronize(step_cache_stream));
                    step_sync_stats.d2d_ms+=std::chrono::duration<double,std::milli>(clock::now()-start).count();
                    step_sync_stats.cache_fill_bytes+=stored;
                }
            }
        }
        return true;
    } catch (const std::exception & error) {
        // Do not leave partially filled entries valid after a failed copy.
        if (step_cache_stream) cudaStreamSynchronize(step_cache_stream);
        if (step_cache) step_cache->trim(0);
        GGML_LOG_ERROR("STRATA_STEP cache transfer failed: %s\n",error.what());return false;
    }
}
#include "pipeline_runtime.inc"
