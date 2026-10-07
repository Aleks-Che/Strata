// Compiled only into the private generated MiMo scheduler.
#include "sync_runtime.h"
#include "../common/expert_file.hpp"
#include "../common/device_memory.hpp"
#include "../glm5next/host_pages.hpp"
#include "expert_cache.hpp"
#include "expert_slab.hpp"
#include "gpu_trace.hpp"
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <stdexcept>
static thread_local int mimo_mode=0;
static thread_local strata_mimo_stats mimo_stats{};
static thread_local std::unique_ptr<mimo2::GpuTrace> mimo_trace;
static thread_local void * mimo_staging=nullptr;
static thread_local const std::atomic<bool> * mimo_cancel=nullptr;
static thread_local strata_mimo_observer mimo_observer=nullptr;
static thread_local void * mimo_owner=nullptr;
static constexpr size_t mimo_staging_limit=16*1024*1024;
static thread_local std::unique_ptr<mimo2::ExpertSlab> mimo_slab;
// Cache returns its slots before the slab owner is destroyed, including TLS exit.
static thread_local std::unique_ptr<mimo2::ExpertCache> mimo_cache;
int strata_mimo_cache_slab_mib() {
    const char *value=std::getenv("STRATA_MIMO_CACHE_SLAB_MIB");
    if(!value)return 16;
    if(std::strcmp(value,"0")==0)return 0;
    if(std::strcmp(value,"16")==0)return 16;
    if(std::strcmp(value,"32")==0)return 32;
    throw std::runtime_error("STRATA_MIMO_CACHE_SLAB_MIB must be 0, 16 or 32");
}
static thread_local cudaStream_t mimo_cache_stream=nullptr;
int strata_mimo_cache_decay() {
    const char *value=std::getenv("STRATA_MIMO_CACHE_DECAY");
    if(!value)return 65536;
    if(std::strcmp(value,"0")==0)return 0;
    if(std::strcmp(value,"16384")==0)return 16384;
    if(std::strcmp(value,"65536")==0)return 65536;
    if(std::strcmp(value,"131072")==0)return 131072;
    throw std::runtime_error("STRATA_MIMO_CACHE_DECAY must be 0, 16384, 65536 or 131072");
}
static thread_local uint64_t mimo_generation=0;
static thread_local bool mimo_prefill=false,mimo_prefill_admit=true;
static thread_local bool mimo_mmap_reader=false;
static thread_local bool mimo_workspace_pending=true;
static thread_local std::unique_ptr<strata_glm::HostWorkingSetBudget> mimo_host_budget;
static thread_local std::chrono::steady_clock::time_point mimo_host_refresh;
struct MimoTensorIdentity {
    uint32_t id;std::string name;ggml_type type;const void *data;
    size_t bytes,stride;int64_t experts;
    std::shared_ptr<strata_expert_file::Source> file;uint64_t offset;
};
static thread_local std::map<const ggml_tensor *,MimoTensorIdentity> mimo_tensors;
static bool mimo_pipeline_enabled();
static void mimo_pipeline_shutdown();
static void mimo_pipeline_account();
static void mimo_pipeline_reset_counters();
static bool mimo_pipeline_copy(ggml_backend_t,ggml_tensor *,const ggml_tensor *,size_t,size_t);
static void mimo_cuda(cudaError_t rc) {if(rc!=cudaSuccess)throw std::runtime_error(std::string("MiMo cache CUDA: ")+cudaGetErrorString(rc));}
static bool mimo_cancelled() {return mimo_cancel && mimo_cancel->load();}
static bool mimo_expert(const ggml_tensor *t) {
    const std::string name=t->name,suffix="_exps.weight";
    return name.size()>=suffix.size() && name.compare(name.size()-suffix.size(),suffix.size(),suffix)==0;
}
void strata_mimo_mode(int mode) {
    if(mode<0 || mode>2)throw std::runtime_error("invalid MiMo copy mode");
    mimo_mode=mode;
}
void strata_mimo_cancel(const std::atomic<bool> *flag) {mimo_cancel=flag;}
void strata_mimo_observe(strata_mimo_observer observer,void *owner) {mimo_observer=observer;mimo_owner=owner;}
void strata_mimo_reset() {
    const auto host_limit=mimo_stats.host_working_set_limit;mimo_stats={};
    mimo_stats.staging_bytes=mimo_staging?mimo_staging_limit:0;
    if(mimo_host_budget)mimo_stats.host_working_set_limit=host_limit;
    if(mimo_cache)mimo_cache->reset_stats();
    if(mimo_slab)mimo_slab->reset_stats();
    mimo_pipeline_reset_counters();
}
strata_mimo_stats strata_mimo_snapshot() {
    mimo_pipeline_account();
    auto s=mimo_stats;
    if(mimo_cache) {
        auto c=mimo_cache->stats();s.cache_hits=c.hits;s.cache_misses=c.misses;s.cache_evictions=c.evictions;
        s.cache_allocations=c.allocations;s.cache_reuses=c.reuses;s.cache_bypasses=c.bypasses;s.cache_oom=c.oom;
        s.cache_bytes=mimo_cache->bytes();s.cache_payload_bytes=mimo_cache->payload_bytes();s.cache_limit=mimo_cache->budget();
        s.cache_slot_bytes=mimo_cache->slot_bytes();
        s.cache_pending=mimo_cache->pending();
        s.cache_decay=mimo_cache->decay_period();s.cache_history_keys=mimo_cache->history_keys();
        s.cache_frequency_updates=c.frequency_updates;s.cache_frequency_rejected=c.frequency_rejected;
        s.cache_frequency_candidates=c.frequency_candidates;
    }
    if(mimo_slab) {
        auto a=mimo_slab->stats();s.cache_slab_mib=strata_mimo_cache_slab_mib();s.cache_slab_blocks=a.blocks;
        s.cache_slab_allocations=a.allocations;s.cache_slab_reuses=a.reuses;s.cache_slab_denied=a.denied;s.cache_slab_oom=a.oom;
    }
    return s;
}
void strata_mimo_memory(uint64_t extra_gpu,uint64_t extra_ram) {
    static StrataGlobalMemory gpu;
    size_t free=0,total=0;
    if(!gpu.sample(0,free,total))throw std::runtime_error("MiMo requires global NVML memory accounting");
    if(mimo_cache) {
        // FP32 cuBLAS can allocate weight conversion pools at first prefill.
        // Do not let a cache fill consume that yet-unallocated working space.
        const size_t workspace=mimo_workspace_pending?size_t(3)<<30:0;
        const size_t before=mimo_cache->bytes();mimo_cache->constrain(free>workspace?free-workspace:0,total);
        if(mimo_cache->bytes()!=before && !gpu.sample(0,free,total))throw std::runtime_error("MiMo memory resample failed after cache trim");
    }
    MEMORYSTATUSEX ram{};ram.dwLength=sizeof(ram);
    if(!GlobalMemoryStatusEx(&ram))throw std::runtime_error("cannot sample global RAM");
    if(mimo_host_budget) {
        const auto now=std::chrono::steady_clock::now();
        if(now-mimo_host_refresh>std::chrono::milliseconds(250) || ram.ullAvailPhys<ram.ullTotalPhys/20+(512ull<<20)) {
            // Leave one additional percent for page-fault bursts and other processes.
            // Only this process is changed; clean mapped weights may be re-faulted.
            mimo_stats.host_working_set_limit=mimo_host_budget->apply(94);
            mimo_host_refresh=now;
            if(!GlobalMemoryStatusEx(&ram))throw std::runtime_error("cannot resample RAM after working set admission");
        }
    }
    mimo_stats.gpu_free=free;mimo_stats.gpu_total=total;
    mimo_stats.ram_free=ram.ullAvailPhys;mimo_stats.ram_total=ram.ullTotalPhys;
    const uint64_t gpu_reserve=(total+19)/20,ram_reserve=(ram.ullTotalPhys+19)/20;
    if(free<gpu_reserve || extra_gpu>free-gpu_reserve ||
       ram.ullAvailPhys<ram_reserve || extra_ram>ram.ullAvailPhys-ram_reserve)
        throw std::runtime_error("MiMo 95% global RAM/VRAM budget exceeded");
}
void strata_mimo_release() {
    mimo_pipeline_shutdown(); // Drain readers/ring before cache identities or mapping owners die.
    // Every copy fences before returning, including error/cancel paths.
    if(mimo_staging)cudaFreeHost(mimo_staging);
    mimo_staging=nullptr;mimo_cancel=nullptr;mimo_observer=nullptr;mimo_owner=nullptr;
    if(mimo_cache_stream)cudaStreamSynchronize(mimo_cache_stream);
    mimo_cache.reset();mimo_slab.reset();mimo_tensors.clear();
    if(mimo_cache_stream)cudaStreamDestroy(mimo_cache_stream);
    mimo_cache_stream=nullptr;
    mimo_mmap_reader=false;
    mimo_host_budget.reset();
}
void strata_mimo_model_begin() {strata_mimo_release();++mimo_generation;mimo_workspace_pending=true;}
void strata_mimo_register(const ggml_tensor *t) {
    if(!mimo_expert(t) || !ggml_is_contiguous(t) || t->ne[2]<=0 || t->ne[3]!=1 || !t->data ||
       t->nb[2]*size_t(t->ne[2])!=ggml_nbytes(t))throw std::runtime_error("invalid MiMo cache tensor registration");
    auto file=strata_expert_file::find(t->data,ggml_nbytes(t));
    if(!file)throw std::runtime_error("MiMo tensor mapping is not registered");
    MimoTensorIdentity id{uint32_t(mimo_tensors.size()),t->name,t->type,t->data,ggml_nbytes(t),t->nb[2],t->ne[2],file,file->offset_of(t->data,ggml_nbytes(t))};
    if(!mimo_tensors.emplace(t,std::move(id)).second)throw std::runtime_error("duplicate MiMo tensor registration");
}
void strata_mimo_cache(size_t bytes) {
    if(mimo_pipeline_enabled())throw std::runtime_error("configure MiMo cache before pipeline readers");
    if(bytes && mimo_mode!=2)throw std::runtime_error("MiMo cache requires pinned copy mode");
    if(mimo_cache_stream)mimo_cuda(cudaStreamSynchronize(mimo_cache_stream));
    mimo_cache.reset();mimo_slab.reset();
    if(bytes) {
        if(!mimo_cache_stream)mimo_cuda(cudaStreamCreateWithFlags(&mimo_cache_stream,cudaStreamNonBlocking));
        auto allocate=[](size_t n)->void * {
            void *p=nullptr;const auto rc=cudaMalloc(&p,n);
            if(rc==cudaErrorMemoryAllocation) {cudaGetLastError();return nullptr;}
            mimo_cuda(rc);return p;
        };
        auto release=[](void *p) {cudaFree(p);};
        const int slab=strata_mimo_cache_slab_mib();
        if(slab) {
            mimo_slab=std::make_unique<mimo2::ExpertSlab>(size_t(slab)<<20,2u<<20,allocate,release);
            mimo_cache=std::make_unique<mimo2::ExpertCache>(bytes,
                [](size_t n) {return mimo_slab->get(n,mimo_cache->budget());},
                [](void *p) {mimo_slab->put(p);},256,
                [] {return mimo_slab->reserved();},[](size_t n) {return mimo_slab->growth(n);},strata_mimo_cache_decay());
        } else mimo_cache=std::make_unique<mimo2::ExpertCache>(bytes,allocate,release,2u<<20,
            std::function<size_t()>{},std::function<size_t(size_t)>{},strata_mimo_cache_decay());
        // NVML clamps whole blocks, including free slots, not just payload.
        strata_mimo_memory();
    }
}
void strata_mimo_cache_prefill(bool admit) {mimo_prefill_admit=admit;}
void strata_mimo_phase(bool prefill) {mimo_prefill=prefill;}
void strata_mimo_reader(bool mmap) {
    if(mimo_pipeline_enabled())throw std::runtime_error("configure MiMo source before pipeline readers");
    mimo_mmap_reader=mmap;
    if(mmap) {
        mimo_host_budget=std::make_unique<strata_glm::HostWorkingSetBudget>();
        mimo_stats.host_working_set_limit=mimo_host_budget->apply(94);mimo_host_refresh=std::chrono::steady_clock::now();
    } else mimo_host_budget.reset();
}
void strata_mimo_workspace_ready() {mimo_workspace_pending=false;}
static void mimo_d2d(void *dst,const void *src,size_t bytes) {
    const auto start=std::chrono::steady_clock::now();
    if(mimo_trace)mimo_trace->record(mimo_cache_stream,nullptr,true,4);
    const auto rc=cudaMemcpyAsync(dst,src,bytes,cudaMemcpyDeviceToDevice,mimo_cache_stream);
    if(mimo_trace)mimo_trace->record(mimo_cache_stream,nullptr,false,4);
    ++mimo_stats.pipeline_copy_fences;
    const auto fence=cudaStreamSynchronize(mimo_cache_stream); // also drain on submission errors
    mimo_cuda(rc);mimo_cuda(fence);
    mimo_stats.d2d_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
}
static bool mimo_copy_file(ggml_backend_t backend,ggml_tensor *dst,const ggml_tensor *src,size_t offset,size_t bytes) {
    try {
        if(mimo_cancelled())return false;
        if(!mimo_expert(src) || !src->data || src->type!=dst->type ||
           offset>ggml_nbytes(src) || bytes>ggml_nbytes(src)-offset ||
           offset>ggml_nbytes(dst) || bytes>ggml_nbytes(dst)-offset)
            throw std::runtime_error("invalid MiMo selected expert range");
        if(mimo_mmap_reader) {
            // Preserve the native pageable-mmap delivery, now at a cache miss only.
            // This wall timer includes page faults plus the driver's H2D staging.
            ggml_backend_synchronize(backend);const auto start=std::chrono::steady_clock::now();
            ggml_backend_tensor_set_async(backend,dst,static_cast<const char *>(src->data)+offset,offset,bytes);
            ggml_backend_synchronize(backend);
            mimo_stats.h2d_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            mimo_stats.h2d_bytes+=bytes;++mimo_stats.chunks;
            if(mimo_observer)mimo_observer(backend,dst,src,offset,bytes,mimo_owner);
            return !mimo_cancelled();
        }
        if(!mimo_staging && cudaHostAlloc(&mimo_staging,mimo_staging_limit,cudaHostAllocDefault)!=cudaSuccess)
            throw std::runtime_error("MiMo pinned staging allocation failed");
        mimo_stats.staging_bytes=mimo_staging_limit;
        const auto *address=static_cast<const char *>(src->data)+offset;
        auto source=strata_expert_file::find(address,bytes);
        if(!source)throw std::runtime_error("MiMo expert is outside registered file mapping");
        static const std::atomic<bool> never{false};
        strata_expert_file::Request reader;
        ggml_backend_synchronize(backend);
        for(size_t done=0;done<bytes;) {
            const size_t count=std::min(bytes-done,mimo_staging_limit);
            auto start=std::chrono::steady_clock::now();
            if(!reader.read(*source,address+done,mimo_staging,count,mimo_cancel?*mimo_cancel:never))return false;
            mimo_stats.source_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            mimo_stats.source_bytes+=count;
            start=std::chrono::steady_clock::now();
            ggml_backend_tensor_set_async(backend,dst,mimo_staging,offset+done,count);
            ggml_backend_synchronize(backend); // protects both the source slot and destination lifetime
            mimo_stats.h2d_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            mimo_stats.h2d_bytes+=count;++mimo_stats.chunks;
            if(mimo_observer)mimo_observer(backend,dst,src,offset+done,count,mimo_owner);
            done+=count;
            if(mimo_cancelled())return false;
        }
        return true;
    } catch(const std::exception &e) {
        ggml_backend_synchronize(backend);
        GGML_LOG_ERROR("STRATA_MIMO_COPY: %s\n",e.what());return false;
    }
}
static bool mimo_copy_range(ggml_backend_t backend,ggml_tensor *dst,const ggml_tensor *src,size_t offset,size_t bytes) {
    if(mimo_pipeline_enabled())return mimo_pipeline_copy(backend,dst,src,offset,bytes);
    try {
        if(mimo_cancelled())return false;
        strata_mimo_memory(0,mimo_staging || mimo_mmap_reader?0:mimo_staging_limit);
        ++mimo_stats.ranges;mimo_stats.requested_bytes+=bytes;
        if(!mimo_cache)return mimo_copy_file(backend,dst,src,offset,bytes);
        auto found=mimo_tensors.find(src);
        if(found==mimo_tensors.end())throw std::runtime_error("unregistered cached MiMo tensor");
        const auto &id=found->second;
        if(src->data!=id.data || src->type!=id.type || id.name!=src->name || src->nb[2]!=id.stride ||
           src->ne[2]!=id.experts || ggml_nbytes(src)!=id.bytes || !ggml_is_contiguous(src) ||
           dst->type!=src->type || !dst->data || !ggml_is_contiguous(dst) || ggml_backend_buffer_is_host(dst->buffer) ||
           offset%id.stride || offset>id.bytes || bytes>id.bytes-offset || offset>ggml_nbytes(dst) || bytes>ggml_nbytes(dst)-offset)
            throw std::runtime_error("MiMo cached tensor/range identity changed");
        const size_t matrices=bytes/id.stride,tail=bytes%id.stride;
        if(!matrices || tail>512)throw std::runtime_error("unexpected selected expert padding");
        ggml_backend_synchronize(backend);
        const bool admit=!mimo_prefill || mimo_prefill_admit;
        std::vector<void *> existing;
        if(!admit) {
            bool any=false;existing.reserve(matrices);
            for(size_t n=0;n<matrices;++n) {
                auto *p=mimo_cache->get({mimo_generation,id.id,uint32_t(offset/id.stride+n)},id.stride,false);
                existing.push_back(p);any|=p!=nullptr;
            }
            // Preserve the native grouped range (including padding) on a prefill miss.
            // No admission/eviction occurs while these hit pointers are retained.
            if(!any)return mimo_copy_file(backend,dst,src,offset,bytes);
        }
        for(size_t n=0;n<matrices;++n) {
            if(mimo_cancelled())return false;
            const size_t at=offset+n*id.stride;
            const mimo2::MatrixKey key{mimo_generation,id.id,uint32_t(at/id.stride)};
            if(admit)mimo_cache->record(key);
            void *cached=admit?mimo_cache->get(key,id.stride):existing[n];
            auto *device=static_cast<char *>(dst->data)+at;
            if(cached) {
                mimo_d2d(device,cached,id.stride);mimo_stats.cache_hit_bytes+=id.stride;
                if(mimo_observer)mimo_observer(backend,dst,src,at,id.stride,mimo_owner);
            } else {
                if(!mimo_copy_file(backend,dst,src,at,id.stride))return false;
                if(admit && !mimo_cancelled())mimo_cache->store(key,id.stride,[&](void *p) {
                    mimo_d2d(p,device,id.stride);mimo_stats.cache_fill_bytes+=id.stride;
                });
            }
        }
        // Scheduler's <=512-byte overread guard belongs to the next matrix;
        // never publish that partial data as a cache entry.
        if(tail) {
            const size_t at=offset+matrices*id.stride;
            const mimo2::MatrixKey key{mimo_generation,id.id,uint32_t(at/id.stride)};
            if(void *cached=mimo_cache->get(key,id.stride,false)) {
                mimo_d2d(static_cast<char *>(dst->data)+at,cached,tail);mimo_stats.cache_hit_bytes+=tail;
                if(mimo_observer)mimo_observer(backend,dst,src,at,tail,mimo_owner);
            } else if(!mimo_copy_file(backend,dst,src,at,tail))return false;
        }
        return !mimo_cancelled();
    } catch(const std::exception &e) {
        if(mimo_cache_stream)cudaStreamSynchronize(mimo_cache_stream);
        ggml_backend_synchronize(backend);GGML_LOG_ERROR("STRATA_MIMO_CACHE: %s\n",e.what());return false;
    }
}
