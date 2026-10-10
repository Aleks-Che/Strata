// Compiled only in MiniMax's private generated ggml-backend.cpp.
#include "sync_runtime.h"
#include "sync_test.h"
#include "router_ids.hpp"
#include "sort_table.hpp"
#include "../common/expert_file.hpp"
#include "../common/device_memory.hpp"
#include "../glm5next/host_pages.hpp"
#include <chrono>
#include <cstring>
#include <stdexcept>
static thread_local int mm27_mode=0;
static thread_local strata_mm27_stats mm27_stats;
static thread_local minimax_m2::RouterIds mm27_router_ids;
static thread_local bool mm27_router_host_ids=false;
static thread_local bool mm27_sort_table_async=false;
static thread_local minimax_m2::SortTableRing mm27_sort_table;
void strata_mm27_sort_table_async(bool enabled) {
    mm27_sort_table.drain();
    if(!enabled)mm27_sort_table.close();
    mm27_sort_table_async=enabled;
}
bool strata_mm27_sort_table_copy(void *dst,const void *src,size_t bytes,cudaStream_t stream) {
    if(!mm27_sort_table_async || mm27_mode!=2)return false;
    if(!mm27_sort_table.pinned_bytes())strata_mm27_memory(0,minimax_m2::SortTableRing::capacity);
    if(mm27_sort_table.copy(dst,src,bytes,stream))return true;
    ++mm27_stats.sort_table_fallbacks;return false;
}
void strata_mm27_router_host_ids(bool enabled) {
    mm27_router_ids.clear();mm27_router_host_ids=enabled;
}
bool strata_mm27_router_ids_take(const ggml_tensor *node,void *data,size_t bytes) {
    if(!mm27_router_host_ids || mm27_mode!=2)return false;
    if(!mm27_router_ids.take(node,data,bytes)) {++mm27_stats.router_ids_misses;return false;}
    ++mm27_stats.router_ids_hits;mm27_stats.router_ids_bytes+=bytes;return true;
}
static thread_local void *mm27_staging=nullptr;
static thread_local int mm27_reader=0;
static thread_local std::unique_ptr<strata_glm::HostWorkingSetBudget> mm27_host_budget;
static thread_local std::chrono::steady_clock::time_point mm27_host_refresh{};
static constexpr size_t mm27_staging_limit=16u<<20;
static thread_local const std::atomic<bool> *mm27_cancel=nullptr;
static const std::atomic<bool> mm27_never_cancel{false};
static thread_local strata_mm27_observer mm27_observer=nullptr;
static thread_local void *mm27_owner=nullptr;
using mm27_clock=std::chrono::steady_clock;
static thread_local mm27_clock::time_point mm27_next_memory_check{};
static thread_local strata_mm27_test_limits mm27_test_limits;
static thread_local std::string mm27_error;
#include "cache_runtime.inc"
#include "pipeline_state.inc"
void strata_mm27_test_memory_limits(strata_mm27_test_limits limits) {
    mm27_test_limits=limits;mm27_next_memory_check={};
}
const char *strata_mm27_last_error() {return mm27_error.c_str();}
static double mm27_elapsed(mm27_clock::time_point start) {
    return std::chrono::duration<double,std::milli>(mm27_clock::now()-start).count();
}
void strata_mm27_mode(int mode) {
    if(mode<0 || mode>2)throw std::invalid_argument("invalid MiniMax copy mode");
    if(mode!=2 && mm27_cache)throw std::invalid_argument("disable MiniMax cache before changing copy mode");
    if(mode!=2 && mm27_ram_cache)throw std::invalid_argument("disable MiniMax RAM cache before changing copy mode");
    if(mode!=2 && mm27_pipeline_readers)throw std::invalid_argument("disable MiniMax pipeline before changing copy mode");
    mm27_mode=mode;
}
void strata_mm27_cancel(const std::atomic<bool> *flag) {mm27_cancel=flag;}
void strata_mm27_reader(int reader) {
    if(reader<0 || reader>3 || (reader && mm27_mode!=2))throw std::invalid_argument("invalid MiniMax reader/mode");
    if(reader && mm27_pipeline_readers)throw std::invalid_argument("MiniMax pipeline requires file reader");
    if(reader && !mm27_host_budget) {
        auto budget=std::make_unique<strata_glm::HostWorkingSetBudget>();
        mm27_stats.host_working_set_limit=budget->apply(94);
        ++mm27_stats.host_budget_updates;mm27_host_refresh=mm27_clock::now();mm27_host_budget=std::move(budget);
    } else if(!reader && !mm27_ram_cache) {mm27_host_budget.reset();mm27_stats.host_working_set_limit=0;}
    mm27_reader=reader;
}
void strata_mm27_observe(strata_mm27_observer fn,void *owner) {mm27_observer=fn;mm27_owner=owner;}
void strata_mm27_reset() {
    mm27_sort_table.reset_counters();
    mm27_router_ids.clear();
    mm27_pipeline_account();
    const auto host_limit=mm27_stats.host_working_set_limit;
    mm27_stats={};mm27_stats.staging_bytes=mm27_staging?mm27_staging_limit:0;
    mm27_stats.host_working_set_limit=mm27_host_budget?host_limit:0;
    mm27_error.clear();mm27_next_memory_check={};
    if(mm27_cache)mm27_cache->reset_counters();
    if(mm27_ram_cache)mm27_ram_cache->reset_counters();
    if(mm27_arena)mm27_arena->reset_counters();
    mm27_pipeline_account();
}
strata_mm27_stats strata_mm27_snapshot() {
    mm27_pipeline_account();auto s=mm27_stats;s.pipeline_pending_copy=mm27_pending_copy;mm27_cache_snapshot(s);
    const auto &t=mm27_sort_table.counters();
    s.sort_table_copies=t.copies;s.sort_table_bytes=t.bytes;s.sort_table_reuse_waits=t.reuse_waits;
    s.sort_table_drain_waits=t.drain_waits;s.sort_table_peak_pending=t.peak_pending;
    s.sort_table_pinned_bytes=mm27_sort_table.pinned_bytes();s.sort_table_pending=mm27_sort_table.pending();
    return s;
}
void strata_mm27_release() {
    strata_mm27_sort_table_async(false);
    strata_mm27_router_host_ids(false);
    mm27_pipeline_dispose();mm27_pipeline_trace.reset();mm27_pipeline_readers=0;mm27_pipeline_lookahead=false;
    mm27_pipeline_d2d_batch=false;mm27_pipeline_events=0;
    mm27_ram_cache.reset();
    mm27_cache.reset();mm27_arena.reset();mm27_cache_admit=false;mm27_group_experts=false;mm27_tensors.clear();mm27_bound_model=nullptr;
    if(mm27_staging)cudaFreeHost(mm27_staging);
    mm27_staging=nullptr;mm27_cancel=nullptr;mm27_observer=nullptr;mm27_owner=nullptr;mm27_mode=0;
    mm27_reader=0;mm27_host_budget.reset();
    mm27_test_limits={};
    strata_mm27_reset();
}
strata_mm27_memory_info strata_mm27_memory(uint64_t gpu_reserve,uint64_t ram_reserve) {
    MEMORYSTATUSEX ram{};ram.dwLength=sizeof(ram);
    if(!GlobalMemoryStatusEx(&ram))throw std::runtime_error("cannot sample global RAM");
    PROCESS_MEMORY_COUNTERS_EX process{};process.cb=sizeof(process);
    if(!GetProcessMemoryInfo(GetCurrentProcess(),reinterpret_cast<PROCESS_MEMORY_COUNTERS *>(&process),sizeof(process)))
        throw std::runtime_error("cannot sample process memory");
    if(mm27_host_budget && (mm27_clock::now()-mm27_host_refresh>std::chrono::milliseconds(250) ||
        ram.ullAvailPhys<ram.ullTotalPhys/20+(512ull<<20))) {
        // Cap only this process. Clean mapped pages can be reclaimed by Windows;
        // no duplicate heap cache, locking, or changes to the machine's pagefile.
        mm27_stats.host_working_set_limit=mm27_host_budget->apply(94,ram.ullTotalPhys,ram.ullAvailPhys,process.WorkingSetSize);
        ++mm27_stats.host_budget_updates;mm27_host_refresh=mm27_clock::now();
        if(!GlobalMemoryStatusEx(&ram) || !GetProcessMemoryInfo(GetCurrentProcess(),
            reinterpret_cast<PROCESS_MEMORY_COUNTERS *>(&process),sizeof(process)))
            throw std::runtime_error("cannot resample MiniMax RAM after working set trim");
    }
    // Reclaim this process's mapped pages before GPU cache policy rejects RAM
    // pressure. Both caches get a chance to shrink before the hard 95% guard.
    if(mm27_ram_cache) {
        mm27_ram_cache->refresh();
        if(!GlobalMemoryStatusEx(&ram) || !GetProcessMemoryInfo(GetCurrentProcess(),
            reinterpret_cast<PROCESS_MEMORY_COUNTERS *>(&process),sizeof(process)))
            throw std::runtime_error("cannot resample MiniMax RAM after view reclamation");
    }
    try {mm27_cache_refresh();} catch(const std::exception &e) {
        ++mm27_stats.pressure_rejections;mm27_error=std::string("MiniMax cache memory admission: ")+e.what();
        throw std::runtime_error(mm27_error);
    }
    static StrataGlobalMemory global;
    int device=0;size_t free=0,total=0;
    if(cudaGetDevice(&device)!=cudaSuccess || !global.sample(device,free,total))
        throw std::runtime_error("cannot sample global MiniMax GPU memory");
    ++mm27_stats.memory_checks;
    mm27_stats.sampled_ram_used_peak=std::max<uint64_t>(mm27_stats.sampled_ram_used_peak,ram.ullTotalPhys-ram.ullAvailPhys);
    mm27_stats.sampled_vram_used_peak=std::max<uint64_t>(mm27_stats.sampled_vram_used_peak,total-free);
    mm27_stats.sampled_private_peak=std::max<uint64_t>(mm27_stats.sampled_private_peak,process.PrivateUsage);
    mm27_stats.sampled_working_set_peak=std::max<uint64_t>(mm27_stats.sampled_working_set_peak,process.WorkingSetSize);
    // Subtraction avoids overflow with rejected UINT64_MAX reservations.
    // Test ceilings do not change reported real memory or sampled peaks.
    const auto gpu_free=std::min<uint64_t>(free,mm27_test_limits.vram_available);
    const auto ram_free=std::min<uint64_t>(ram.ullAvailPhys,mm27_test_limits.ram_available);
    const char *error=nullptr;
    if(gpu_free<total/20 || gpu_reserve>gpu_free-total/20)error="MiniMax global VRAM 95% limit";
    else if(ram_free<ram.ullTotalPhys/20 || ram_reserve>ram_free-ram.ullTotalPhys/20)error="MiniMax global RAM 95% limit";
    if(error) {++mm27_stats.pressure_rejections;mm27_error=error;throw std::runtime_error(error);}
    return {ram.ullTotalPhys,ram.ullAvailPhys,total,free,process.PrivateUsage,process.WorkingSetSize};
}
static bool mm27_expert(const ggml_tensor *t) {
    const std::string n=t->name;
    return n.rfind("blk.",0)==0 && n.size()>12 && n.compare(n.size()-12,12,"_exps.weight")==0;
}
static bool mm27_cancelled() {
    if(mm27_cancel && mm27_cancel->load()) {mm27_error="MiniMax request cancelled";return true;}
    return false;
}
static bool mm27_live_memory() {
    // Poll at copy/compute boundaries, at most once per 25 ms. This does not
    // interrupt a running kernel or prevent concurrent external allocations.
    if(mm27_clock::now()<mm27_next_memory_check)return true;
    try {
        strata_mm27_memory();mm27_next_memory_check=mm27_clock::now()+std::chrono::milliseconds(25);return true;
    } catch(const std::exception &e) {
        mm27_error=e.what();GGML_LOG_ERROR("STRATA_MM27 memory: %s\n",e.what());return false;
    }
}
static bool mm27_upload(ggml_backend_t backend,ggml_tensor *dst,
    const strata_expert_file::Source &file,uint64_t base,size_t offset,size_t bytes,bool defer=false) {
    if(!file.contains(base,bytes))throw std::runtime_error("MiniMax reader range outside mapping");
    if(mm27_pipeline_active) {
        if(mm27_cancelled() || !mm27_live_memory())return false;
        const auto start=mm27_clock::now();
        StrataExpertPipeline::CopyObserver observer;
        if(mm27_pipeline_trace)observer=[](cudaStream_t stream,bool begin,size_t) {mm27_pipeline_trace->record(stream,false,begin);};
        if(!mm27_pipeline_queue->transfer(static_cast<char *>(dst->data)+offset,
            reinterpret_cast<const char *>(file.base)+base,bytes,mm27_pipeline_stream,observer))
            throw std::runtime_error("MiniMax pipeline range differs from plan");
        ++mm27_stats.pipeline_copy_submissions;
        if(!defer) {++mm27_stats.pipeline_copy_fences;step35::cuda_check(cudaStreamSynchronize(mm27_pipeline_stream));}
        mm27_stats.pipeline_delivery_ms+=mm27_elapsed(start);
        mm27_stats.file_bytes+=bytes;mm27_stats.source_bytes+=bytes;mm27_stats.h2d_bytes+=bytes;
        return !mm27_cancelled();
    }
    const int reader=mm27_reader==3?(mm27_cache_admit?1:0):mm27_reader;
    strata_expert_file::Request read;
    for(size_t done=0;done<bytes;) {
        if(mm27_cancelled() || !mm27_live_memory())return false;
        const size_t count=std::min(bytes-done,mm27_staging_limit);
        auto start=mm27_clock::now();
        const auto *mapped=reinterpret_cast<const char *>(file.base)+base+done;
        if(reader==0) {
            if(!read.read_at(file,base+done,mm27_staging,count,mm27_cancel?*mm27_cancel:mm27_never_cancel)) {
                mm27_cancelled();return false;
            }
            mm27_stats.file_bytes+=count;
        } else if(reader==1) {
            std::memcpy(mm27_staging,mapped,count);mm27_stats.mmap_bytes+=count;
        }
        if(reader!=2)mm27_stats.source_ms+=mm27_elapsed(start);
        start=mm27_clock::now();
        ggml_backend_tensor_set_async(backend,dst,reader==2?mapped:mm27_staging,offset+done,count);
        ggml_backend_synchronize(backend);
        if(reader==2)mm27_stats.mmap_bytes+=count; // H2D timer includes faults/driver staging.
        mm27_stats.source_bytes+=count;
        mm27_stats.h2d_ms+=mm27_elapsed(start);mm27_stats.h2d_bytes+=count;
        ++mm27_stats.chunks;done+=count;
    }
    return true;
}
static void mm27_cache_copy(void *dst,const void *src,size_t bytes,bool defer=false) {
    // CUDA's "synchronous" D2D cudaMemcpy does not synchronize the host.
    // Explicitly complete copies before graph execution or cache eviction;
    // the ggml compute stream is non-blocking and cannot rely on stream 0.
    const auto stream=mm27_pipeline_active?mm27_pipeline_stream:nullptr;
    step35::cuda_check(cudaMemcpyAsync(dst,src,bytes,cudaMemcpyDeviceToDevice,stream));
    if(mm27_pipeline_active)++mm27_stats.pipeline_copy_submissions;
    if(!defer) {
        if(mm27_pipeline_active)++mm27_stats.pipeline_copy_fences;
        step35::cuda_check(cudaStreamSynchronize(stream));
    }
}
static bool mm27_copy_range(ggml_backend_t backend,ggml_tensor *dst,const ggml_tensor *src,size_t offset,size_t bytes) {
    try {
        if(mm27_cancelled() || !mm27_live_memory())return false;
        if(!mm27_expert(src) || !src->data || !src->nb[2] || src->type!=dst->type || !ggml_are_same_shape(src,dst) ||
            !ggml_is_contiguous(src) || !ggml_is_contiguous(dst) || src->ne[3]!=1 ||
            offset%src->nb[2] || offset>ggml_nbytes(src) || bytes>ggml_nbytes(src)-offset ||
            offset>ggml_nbytes(dst) || bytes>ggml_nbytes(dst)-offset)
            throw std::runtime_error("invalid MiniMax selected range");
        const auto tail=bytes%src->nb[2];
        if(bytes<src->nb[2] || (tail && tail!=std::min<size_t>(512,src->nb[2])))
            throw std::runtime_error("invalid MiniMax expert padding");
        auto file=strata_expert_file::find(src->data,ggml_nbytes(src));
        if(!file)throw std::runtime_error("unregistered MiniMax expert mapping");
        if(!mm27_staging && mm27_reader!=2 && !mm27_pipeline_active) {
            strata_mm27_memory(0,mm27_staging_limit);
            const auto rc=cudaHostAlloc(&mm27_staging,mm27_staging_limit,cudaHostAllocDefault);
            if(rc!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(rc));
            mm27_stats.staging_bytes=mm27_staging_limit;
        }
        // No previous scratch consumer or upload may outlive buffer reuse.
        if(mm27_pipeline_active)++mm27_stats.pipeline_scratch_fences;
        ggml_backend_synchronize(backend);
        const uint64_t base=file->offset_of(src->data,ggml_nbytes(src))+offset;
        ++mm27_stats.ranges;mm27_stats.selected_bytes+=bytes;
        if(!mm27_cache || (!mm27_cache_admit && !mm27_cache->size())) {
            if(!mm27_upload(backend,dst,*file,base,offset,bytes))return false;
        } else {
            const auto id=mm27_tensor_id(src);const size_t stride=src->nb[2],count=bytes/stride;
            for(size_t e=0;e<count;++e) {
                if(mm27_cancelled() || !mm27_live_memory())return false;
                const size_t at=offset+e*stride;
                // Cache real following bytes, including the scheduler's guard.
                // Never synthesize zeros or read beyond the last tensor expert.
                const size_t stored=stride+std::min<size_t>(512,ggml_nbytes(src)-at-stride);
                const size_t needed=stride+(e+1==count?tail:0);
                const step35::MatrixKey key{mm27_generation,id,uint32_t(at/stride)};
                if(void *hit=mm27_cache->get(key,stored,mm27_cache_admit)) {
                    mm27_cache_copy(static_cast<char *>(dst->data)+at,hit,needed);
                    mm27_stats.cache_hit_bytes+=needed;
                } else {
                    const size_t delivered=mm27_cache_admit?stored:needed;
                    if(!mm27_upload(backend,dst,*file,base+e*stride,at,delivered))return false;
                    mm27_stats.cache_guard_bytes+=delivered-needed;
                    if(mm27_cache_admit)if(void *slot=mm27_cache->admit(key,stored)) {
                        if(mm27_pipeline_active && mm27_cache->grouped()) {
                            mm27_plan_fill_pins.push_back(mm27_cache->protect({key}));
                            mm27_stats.cache_group_plan_pins_peak=std::max<uint64_t>(mm27_stats.cache_group_plan_pins_peak,mm27_plan_fill_pins.size());
                        }
                        // Admission becomes visible only after this synchronous
                        // fill. Exceptions invalidate every possibly partial slot.
                        if(mm27_test_limits.cache_fill_failure)throw std::runtime_error("injected MiniMax cache fill failure");
                        mm27_cache_copy(slot,static_cast<char *>(dst->data)+at,stored);
                        mm27_cache->commit(key,stored);
                        if(mm27_ram_cache)mm27_ram_cache->drop_gpu(file,base+e*stride,stored);
                        mm27_stats.cache_fill_bytes+=stored;
                    } else if(mm27_ram_cache)mm27_ram_cache->gpu_bypass(file,base+e*stride,stored);
                }
            }
        }
        if(mm27_observer)mm27_observer(backend,dst,src,offset,bytes,mm27_owner);
        return !mm27_cancelled();
    } catch(const std::exception &e) {
        strata_mm27_cache_clear();mm27_error=e.what();GGML_LOG_ERROR("STRATA_MM27 selected-copy: %s\n",e.what());return false;
    }
}
#include "pipeline_runtime.inc"
static ggml_status mm27_graph_compute(ggml_backend_t backend,ggml_cgraph *graph) {
    if(!mm27_mode)return ggml_backend_graph_compute_async(backend,graph);
    if(mm27_cancelled())return GGML_STATUS_ABORTED;
    if(!mm27_live_memory())return GGML_STATUS_FAILED;
    const auto start=mm27_clock::now();
    if(mm27_pipeline_active && mm27_pipeline_trace)mm27_pipeline_trace->record_compute(backend,true);
    ggml_status rc;
    try {
    if(mm27_pipeline_events==2 && std::strstr(ggml_backend_name(backend),"CUDA")) {
        if(std::find(mm27_async_backends.begin(),mm27_async_backends.end(),backend)==mm27_async_backends.end())
            mm27_async_backends.push_back(backend);
        ++mm27_stats.async_compute_calls;rc=ggml_backend_graph_compute_async(backend,graph);
    } else rc=ggml_backend_graph_compute(backend,graph);
    }catch(const std::exception &e) {
        mm27_error=e.what();GGML_LOG_ERROR("STRATA_MM27 compute: %s\n",e.what());return GGML_STATUS_FAILED;
    }
    if(mm27_pipeline_active && mm27_pipeline_trace)mm27_pipeline_trace->record_compute(backend,false);
    mm27_stats.compute_ms+=mm27_elapsed(start);
    if(mm27_pipeline_active && mm27_plan_next<mm27_plan.size() && mm27_test_limits.pipeline_compute_failure) {
        mm27_error="injected MiniMax compute failure with pending lookahead";return GGML_STATUS_FAILED;
    }
    return rc;
}
