// Included in the private generated ggml-backend.cpp, not the source archive.
#include "sync_runtime.h"
#include "sync_test.h"
#include "../common/expert_file.hpp"
#include "../common/device_memory.hpp"
#include "../glm5next/host_pages.hpp"
#include "host_cache.hpp"
#include <cuda_runtime_api.h>
#include <chrono>
#include <cstring>
#include <stdexcept>
static thread_local int hy3_sync_mode=0;
static thread_local strata_hy3_sync_stats hy3_sync_stats{};
static thread_local void * hy3_staging=nullptr;
static constexpr size_t hy3_staging_limit=16*1024*1024;
static thread_local strata_hy3_copy_observer hy3_observer=nullptr;
static thread_local void * hy3_owner=nullptr;
static thread_local const std::atomic<bool> * hy3_cancel=nullptr;
static thread_local std::unique_ptr<strata_glm::HostWorkingSetBudget> hy3_ram_budget;
static thread_local std::shared_ptr<hy3::HostCache> hy3_host_cache;
static thread_local strata_hy3_test_limits hy3_test_limits;
extern "C" void strata_hy3_test_set_limits(strata_hy3_test_limits limits) {hy3_test_limits=limits;}
static const std::atomic<bool> hy3_never_cancel{false};
static void hy3_cuda(cudaError_t rc) {if(rc!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(rc));}
#include "cache_runtime.inc"
void strata_hy3_sync_mode(int mode) {
    if(mode<0 || mode>2) throw std::invalid_argument("invalid Hy3 copy mode");
    hy3_sync_mode=mode;
}
void strata_hy3_sync_cancel(const std::atomic<bool> * cancel) {hy3_cancel=cancel;}
void strata_hy3_sync_observer(strata_hy3_copy_observer fn,void * owner) {hy3_observer=fn;hy3_owner=owner;}
void strata_hy3_sync_reset() {
    hy3_account_pipeline();
    hy3_sync_stats={};hy3_sync_stats.staging_bytes=hy3_staging ? hy3_staging_limit : 0;
    if(hy3_cache) hy3_cache->reset_counters();
    if(hy3_host_cache) hy3_host_cache->reset_counters();
}
strata_hy3_sync_stats strata_hy3_sync_snapshot() {
    hy3_account_pipeline();
    auto s=hy3_sync_stats;s.cache_generation=hy3_generation;
    if(hy3_host_cache) {
        const auto c=hy3_host_cache->snapshot();
        s.ram_cache_bytes=c.bytes;s.ram_cache_budget=c.budget;s.ram_cache_entries=c.entries;
        s.ram_cache_peak_bytes=c.peak_bytes;s.ram_cache_pending=c.pending;
        s.ram_cache_hits=c.hits;s.ram_cache_misses=c.misses;s.ram_cache_hit_bytes=c.hit_bytes;
        s.ram_cache_file_bytes=c.file_bytes;s.ram_cache_fill_bytes=c.fill_bytes;
        s.ram_cache_evictions=c.evictions;s.ram_cache_rejected=c.rejected;s.ram_cache_oom=c.oom;
        s.ram_cache_allocations=c.allocations;s.ram_cache_reuses=c.reuses;
        s.ram_cache_prefill_bypasses=c.prefill_bypasses;s.ram_cache_frequency_bypasses=c.frequency_bypasses;
        s.ram_cache_victim_candidates=c.victim_candidates;s.ram_cache_history_entries=c.history_entries;
        s.ram_cache_reused_payload_bytes=c.reused_payload_bytes;
    }
    if(hy3_cache) {
        const auto c=hy3_cache->counters();
        s.cache_bytes=hy3_cache->resident_bytes();s.cache_budget=hy3_cache->budget();s.cache_entries=hy3_cache->size();
        s.cache_hits=c.hits;s.cache_misses=c.misses;s.cache_evictions=c.evictions;s.cache_oom=c.oom;
        s.cache_rejected=c.rejected;s.cache_reuses=c.reuses;
        s.cache_allocations=c.allocations;s.cache_victim_candidates=c.victim_candidates;
        s.cache_get_ms=c.get_ms;s.cache_admit_ms=c.admit_ms;s.cache_victim_ms=c.victim_ms;
        s.cache_allocate_ms=c.allocate_ms;s.cache_free_ms=c.free_ms;
        s.cache_refresh_ms=c.refresh_ms;s.cache_probe_ms=c.probe_ms;
        s.cache_protect_ms=c.protect_ms;s.cache_trim_ms=c.trim_ms;
    }
    return s;
}
void strata_hy3_memory_check(size_t reserve) {
    // Decode boundaries are synchronized: reclaim cache before the global guard.
    if(hy3_host_cache) hy3_host_cache->refresh();
    if(hy3_cache) hy3_cache->refresh();
    size_t free=0,total=0;
#ifdef _WIN32
    static StrataGlobalMemory gpu;
    int device=-1;hy3_cuda(cudaGetDevice(&device));
    if(!gpu.sample(device,free,total)) throw std::runtime_error("Hy3 global VRAM query failed");
    if(!hy3_ram_budget) hy3_ram_budget=std::make_unique<strata_glm::HostWorkingSetBudget>();
    // Bound this process's mapped resident pages before the global 95% guard.
    // The extra 2% margin absorbs driver allocations and other-process changes.
    hy3_sync_stats.working_set_limit=hy3_ram_budget->apply(93);
    MEMORYSTATUSEX ram{};ram.dwLength=sizeof(ram);
    if(!GlobalMemoryStatusEx(&ram)) throw std::runtime_error("cannot query global RAM");
    hy3_sync_stats.ram_total=ram.ullTotalPhys;hy3_sync_stats.ram_free=ram.ullAvailPhys;
    if(std::min<size_t>(ram.ullAvailPhys,hy3_test_limits.ram_available)<ram.ullTotalPhys/20+hy3_staging_limit)
        throw std::runtime_error("Hy3 global RAM 95% limit reached");
#else
    throw std::runtime_error("Hy3 native-file baseline currently requires Windows");
#endif
    hy3_sync_stats.gpu_free=free;hy3_sync_stats.gpu_total=total;
    ++hy3_sync_stats.memory_samples;
    const size_t available=std::min(free,hy3_test_limits.gpu_available);
    if(available<total/20 || reserve>available-total/20) throw std::runtime_error("Hy3 global VRAM 95% limit reached");
}
void strata_hy3_sync_release() {
    strata_hy3_cache_begin(0);
    if(hy3_staging) cudaFreeHost(hy3_staging);
    hy3_staging=nullptr;hy3_observer=nullptr;hy3_owner=nullptr;hy3_cancel=nullptr;hy3_sync_mode=0;
    hy3_ram_budget.reset();
    hy3_profile_delivery=false;
    hy3_prefill=false;hy3_admit_prefill=true;
    hy3_test_limits={};
}
static bool hy3_expert_weight(const ggml_tensor * t) {
    const std::string name=t->name;
    return name.rfind("blk.",0)==0 && name.size()>12 && name.compare(name.size()-12,12,"_exps.weight")==0;
}
static bool hy3_cancelled() {return hy3_cancel && hy3_cancel->load();}
static bool hy3_copy_uncached(ggml_backend_t backend,ggml_tensor * dst,const ggml_tensor * src,size_t offset,size_t bytes) {
    try {
        if(hy3_cancelled()) return false;
        if(!hy3_expert_weight(src) || !src->data || src->type!=dst->type || !ggml_are_same_shape(src,dst) ||
            !ggml_is_contiguous(src) || !ggml_is_contiguous(dst) || src->ne[3]!=1 ||
            offset%src->nb[2] || offset>ggml_nbytes(src) || bytes>ggml_nbytes(src)-offset ||
            offset>ggml_nbytes(dst) || bytes>ggml_nbytes(dst)-offset)
            throw std::runtime_error("invalid Hy3 selected expert range");
        const size_t tail=bytes%src->nb[2];
        if(bytes<src->nb[2] || (tail && tail!=std::min<size_t>(512,src->nb[2])))
            throw std::runtime_error("invalid Hy3 selected expert padding");
        auto file=strata_expert_file::find(src->data,ggml_nbytes(src));
        if(!file) throw std::runtime_error("unregistered Hy3 expert source mapping");
        if(!hy3_staging) {
            strata_hy3_memory_check();
            if(hy3_test_limits.fail_next_pinned_allocation) {
                hy3_test_limits.fail_next_pinned_allocation=false;
                hy3_cuda(cudaErrorMemoryAllocation);
            }
            hy3_cuda(cudaHostAlloc(&hy3_staging,hy3_staging_limit,cudaHostAllocDefault));
            hy3_sync_stats.staging_bytes=hy3_staging_limit;
        }
        // Fence scratch consumers and every upload before staging reuse.
        ggml_backend_synchronize(backend);
        strata_expert_file::Request read;
        using clock=std::chrono::steady_clock;
        const auto elapsed=[](clock::time_point start){return std::chrono::duration<double,std::milli>(clock::now()-start).count();};
        const uint64_t file_offset=file->offset_of(src->data,ggml_nbytes(src))+offset;
        ++hy3_sync_stats.ranges;
        for(size_t done=0;done<bytes;) {
            const size_t count=std::min(bytes-done,hy3_staging_limit);
            auto start=clock::now();
            const auto &cancel=hy3_cancel ? *hy3_cancel : hy3_never_cancel;
            const bool ok=hy3_host_cache?hy3_host_cache->read(file,file_offset+done,hy3_staging,count,cancel,read):
                read.read_at(*file,file_offset+done,hy3_staging,count,cancel);
            if(!ok) return false;
            hy3_sync_stats.source_ms+=elapsed(start);
            start=clock::now();
            ggml_backend_tensor_set_async(backend,dst,hy3_staging,offset+done,count);
            ggml_backend_synchronize(backend);
            hy3_sync_stats.h2d_ms+=elapsed(start);
            hy3_sync_stats.source_bytes+=count;hy3_sync_stats.h2d_bytes+=count;
            ++hy3_sync_stats.chunks;done+=count;
        }
        if(hy3_observer) hy3_observer(hy3_owner,backend,dst,src,offset,bytes);
        return true;
    } catch(const std::exception & e) {GGML_LOG_ERROR("STRATA_HY3 selected-copy: %s\n",e.what());return false;}
}
#include "pipeline_runtime.inc"
