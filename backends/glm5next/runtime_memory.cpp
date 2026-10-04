#include "runtime_memory.hpp"
#include "expert_memory.hpp"
#include "llama-model.h"
#include <atomic>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#ifdef _WIN32
#include <psapi.h>
#endif

namespace strata_glm {
namespace {
using Clock=std::chrono::steady_clock;
constexpr size_t MiB=1ULL<<20, staging_size=16*MiB;
double milliseconds(Clock::time_point start) {return std::chrono::duration<double,std::milli>(Clock::now()-start).count();}
void cuda_check(cudaError_t e) {require(e==cudaSuccess,cudaGetErrorString(e));}
struct Ram {uint64_t total=0,available=0,working_set=0;};
Ram ram() {
    Ram r;
#ifdef _WIN32
    MEMORYSTATUSEX s{}; s.dwLength=sizeof(s);
    require(GlobalMemoryStatusEx(&s)!=0,"global RAM query failed");
    PROCESS_MEMORY_COUNTERS p{}; p.cb=sizeof(p);
    require(GetProcessMemoryInfo(GetCurrentProcess(),&p,sizeof(p))!=0,"working set query failed");
    r={s.ullTotalPhys,s.ullAvailPhys,p.WorkingSetSize};
#else
    std::ifstream in("/proc/meminfo"); std::string line;
    while (std::getline(in,line)) {
        std::istringstream s(line); std::string key,unit; uint64_t kib=0; s>>key>>kib>>unit;
        if (key=="MemTotal:") r.total=kib*1024;
        if (key=="MemAvailable:") r.available=kib*1024;
    }
    require(r.total && r.available<=r.total,"global RAM query failed");
#endif
    return r;
}
}
struct RuntimeMemory::Impl {
    Model model;
    std::string identity;
    uint64_t generation;
    int ram_percent,vram_percent;
    std::vector<std::pair<const ggml_tensor *,ExpertKey>> tensors;
    std::map<const ggml_tensor *,size_t> tensor_index;
    std::unique_ptr<ExpertCache> cache;
    std::unique_ptr<ExpertMemoryController> controller;
    ExpertMemoryController::Probe probe=make_global_memory_probe();
    cudaStream_t stream=nullptr;
    cudaMemPool_t pool=nullptr;
    void * staging=nullptr;
    uint64_t ram_touched=0,working_set_limit=0,d2d_bytes=0;
    double warm_ms=0;
    bool hook=false;
#ifdef _WIN32
    SIZE_T old_min=0,old_max=0; DWORD old_flags=0; bool ws_changed=false;
#endif
    Impl(Model m,const std::string & path,int rp,int vp):model(std::move(m)),
        identity(std::filesystem::weakly_canonical(path).u8string()),ram_percent(rp),vram_percent(vp) {
        static std::atomic<uint64_t> next{0}; generation=++next;
    }
    ~Impl() {
        if (hook) strata_glm_sync_copy_hook(nullptr,nullptr);
        if (stream) cudaStreamSynchronize(stream);
        controller.reset(); cache.reset(); // Drain leases before releasing stream/source.
        if (stream) cudaStreamSynchronize(stream);
        if (pool) cudaMemPoolDestroy(pool);
        if (staging) cudaFreeHost(staging);
        if (stream) cudaStreamDestroy(stream);
#ifdef _WIN32
        if (ws_changed) SetProcessWorkingSetSizeEx(GetCurrentProcess(),old_min,old_max,old_flags);
#endif
    }
    ExpertKey key(const ggml_tensor * t,int expert) const {
        auto k=tensors.at(tensor_index.at(t)).second;
        k.expert=expert; k.offset=size_t(expert)*t->nb[2];
        // Match upstream MMQ's overlap padding. It contains real source bytes,
        // not zeroes, and is charged to the cache along with the matrix.
        k.bytes=t->nb[2]+(expert+1<t->ne[2]?std::min(size_t(512),t->nb[2]):0);
        return k;
    }
    void ram_limit() {
        if (!ram_percent) return;
        const auto r=ram();
#ifdef _WIN32
        const uint64_t used=r.total-r.available, other=used-std::min(used,r.working_set);
        const uint64_t target=r.total/100*ram_percent;
        working_set_limit=std::max<uint64_t>(512*MiB,target-std::min(target,other));
        if (!ws_changed) require(GetProcessWorkingSetSizeEx(GetCurrentProcess(),&old_min,&old_max,&old_flags)!=0,"read working set limit failed");
        // Raising the minimum requires SeIncreaseWorkingSetPrivilege. Keep the
        // existing minimum; a maximum and page warming need no such privilege.
        const bool limited=SetProcessWorkingSetSizeEx(GetCurrentProcess(),old_min,working_set_limit,
            QUOTA_LIMITS_HARDWS_MIN_DISABLE|QUOTA_LIMITS_HARDWS_MAX_ENABLE)!=0;
        require(limited,"set RAM working set limit failed: "+std::to_string(GetLastError()));
        ws_changed=true;
#endif
    }
    void initialize() {
        require((!ram_percent || (ram_percent>=10 && ram_percent<=95)) &&
                (!vram_percent || (vram_percent>=10 && vram_percent<=95)),"memory targets must be 0 (off) or 10..95 percent");
        if (!ram_percent && !vram_percent) return;
        std::regex pattern("blk\\.([0-9]+)\\.ffn_(gate|up|down)_exps\\.weight");
        for (const auto & item:model->tensors_by_name) {
            std::smatch match; const auto * t=item.second;
            if (!std::regex_match(item.first,match,pattern)) continue;
            require(t->data && ggml_backend_buffer_is_host(t->buffer),"memory targets require host expert tensors");
            require(t->nb[2]*size_t(t->ne[2])==ggml_nbytes(t),"expert tensor must be contiguous");
            const auto projection=match[2]=="gate"?Projection::gate:match[2]=="up"?Projection::up:Projection::down;
            // Runtime namespace uses tensor-relative offsets, never claims to be
            // GGUF file offsets. The retained model owns the underlying mapping.
            ExpertKey k{identity,generation,Branch::main,std::stoi(match[1]),0,projection,
                ggml_type_name(t->type),uint64_t(t->ne[0]),uint64_t(t->ne[1]),"runtime-tensor:"+item.first,0,0};
            tensor_index[t]=tensors.size(); tensors.push_back({t,std::move(k)});
        }
        require(!tensors.empty(),"no routed expert tensors found");
        ram_limit();
        if (!vram_percent) return;
        size_t free=0,total=0; require(probe(0,free,total),"global VRAM query unavailable; refusing per-process WDDM estimate");
        StrataVramPolicy policy; policy.mode=2;
        policy.target_mib=(total/100*vram_percent)/MiB;
        policy.reserve_mib=std::max<uint64_t>(128,(total-total/100*vram_percent+MiB-1)/MiB);
        ExpertCache::Admission admission; admission.frequency=true;
        cuda_check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        int supports_pool=0; cuda_check(cudaDeviceGetAttribute(&supports_pool,cudaDevAttrMemoryPoolsSupported,0));
        if (supports_pool) {
            cudaMemPoolProps props{}; props.allocType=cudaMemAllocationTypePinned;
            props.location.type=cudaMemLocationTypeDevice; props.location.id=0;
            cuda_check(cudaMemPoolCreate(&pool,&props));
            // Default release threshold zero: unused physical pages are returned
            // at synchronization. Eviction followed by allocation can reuse them.
            cache=std::make_unique<ExpertCache>(0,admission,
                [this](void **p,size_t n){return cudaMallocFromPoolAsync(p,n,pool,stream);},
                [this](void *p){return cudaFreeAsync(p,stream);});
        } else cache=std::make_unique<ExpertCache>(0,admission);
        controller=std::make_unique<ExpertMemoryController>(*cache,total,policy,probe);
        cuda_check(cudaHostAlloc(&staging,staging_size,cudaHostAllocDefault));
        require(controller->refresh().sample_valid,"initial global VRAM sample invalid");
        strata_glm_sync_copy_hook([](void * owner,ggml_backend * backend,ggml_tensor * dst,const ggml_tensor * src,
                                    size_t offset,size_t bytes,strata_glm_sync_stats * stats) {
            static_cast<Impl *>(owner)->copy(backend,dst,src,offset,bytes,*stats);
        },this); hook=true;
    }
    void upload(void * dst,const uint8_t * src,size_t bytes,strata_glm_sync_stats & stats) {
        stats.staging_bytes=staging_size;
        for (size_t done=0;done<bytes;) {
            const auto n=std::min(bytes-done,staging_size);
            auto start=Clock::now(); memcpy(staging,src+done,n); stats.source_ms+=milliseconds(start);
            start=Clock::now();
            cuda_check(cudaMemcpyAsync(static_cast<uint8_t *>(dst)+done,staging,n,cudaMemcpyHostToDevice,stream));
            cuda_check(cudaStreamSynchronize(stream)); stats.h2d_ms+=milliseconds(start);
            stats.source_bytes+=n; stats.h2d_bytes+=n; ++stats.chunks; done+=n;
        }
    }
    void copy(ggml_backend * backend,ggml_tensor * dst,const ggml_tensor * src,size_t offset,size_t bytes,strata_glm_sync_stats & stats) {
        ggml_backend_synchronize(backend);
        controller->refresh_if_due(std::chrono::milliseconds(500));
        const auto stride=src->nb[2],padding=bytes%stride;
        require(offset%stride==0 && bytes/stride>0 && (padding==0 || padding==std::min(size_t(512),stride)),"invalid selected expert span");
        const size_t first=offset/stride,count=bytes/stride;
        // When stride<=512 the padding is a whole extra matrix. This tiny-fixture
        // case may safely copy that matrix too, but must stay inside the tensor.
        std::vector<ExpertKey> plan;
        for (size_t i=first;i<first+count;++i) plan.push_back(key(src,int(i)));
        auto pins=cache->protect_plan(plan); ++stats.ranges; stats.staging_bytes=staging_size;
        for (const auto & k:plan) {
            const auto * source=static_cast<const uint8_t *>(src->data)+k.offset;
            auto lease=cache->get(k,std::shared_ptr<const void>(model,source),stream,
                [&](void * dest,size_t n,cudaStream_t) {upload(dest,source,n,stats);});
            const auto n=std::min(size_t(k.bytes),offset+bytes-size_t(k.offset));
            auto * dest=static_cast<uint8_t *>(dst->data)+k.offset;
            if (lease) {
                cuda_check(cudaMemcpyAsync(dest,lease.data(),n,cudaMemcpyDeviceToDevice,stream)); d2d_bytes+=n;
            } else upload(dest,source,n,stats);
            lease.release();
        }
        cuda_check(cudaStreamSynchronize(stream));
    }
    void warm() {
        if (!ram_percent && !vram_percent) return;
        auto start=Clock::now();
        if (cache) {
            strata_glm_sync_stats unused{};
            // Round-robin layers/projections to avoid filling only early layers.
            for (int expert=0;;++expert) {
                bool eligible=false,full=false;
                for (const auto & item:tensors) {
                    const auto * t=item.first; if (expert>=t->ne[2]) continue; eligible=true;
                    controller->refresh_if_due(std::chrono::milliseconds(100));
                    const auto k=key(t,expert);
                    if (cache->resident_bytes()+k.bytes>cache->byte_budget()) {full=true;break;}
                    const auto * source=static_cast<const uint8_t *>(t->data)+k.offset;
                    auto lease=cache->get(k,std::shared_ptr<const void>(model,source),stream,
                        [&](void * dest,size_t n,cudaStream_t) {upload(dest,source,n,unused);});
                    if (!lease) {full=true;break;}
                    lease.release(); cuda_check(cudaStreamSynchronize(stream));
                }
                if (!eligible || full) break;
            }
            std::cerr<<"STRATA_GLM cache warm MiB="<<cache->resident_bytes()/MiB<<"\n";
        }
        if (ram_percent) {
            ram_limit();
            volatile uint8_t sink=0;
            bool full=false;
            for (const auto & item:tensors) {
                const auto * t=item.first; auto * source=static_cast<const volatile uint8_t *>(t->data);
                for (size_t offset=0;offset<ggml_nbytes(t);) {
                    const auto r=ram();
                    if (r.total-r.available>=r.total/100*ram_percent) {full=true;break;}
                    const auto end=std::min(offset+32*MiB,ggml_nbytes(t));
                    for (size_t p=offset;p<end;p+=4096) sink=uint8_t(sink^source[p]);
                    ram_touched+=end-offset; offset=end;
                }
                if (full) break;
            }
            (void)sink;
        }
        warm_ms=milliseconds(start);
    }
    nlohmann::json snapshot() const {
        const auto r=ram(); size_t free=0,total=0; const bool valid=probe(0,free,total);
        nlohmann::json result={{"ram_target_percent",ram_percent},{"vram_target_percent",vram_percent},
            {"ram_total_bytes",r.total},{"ram_used_bytes",r.total-r.available},{"working_set_bytes",r.working_set},
            {"working_set_limit_bytes",working_set_limit},{"ram_warm_touched_bytes",ram_touched},{"warm_ms",warm_ms},
            {"global_vram_valid",valid},{"global_vram_total_bytes",total},{"global_vram_used_bytes",total-free},
            {"cache_resident_bytes",cache?cache->resident_bytes():0},{"cache_budget_bytes",cache?cache->byte_budget():0},
            {"cache_d2d_bytes",d2d_bytes},{"cuda_memory_pool",pool!=nullptr}};
        if (cache) {
            const auto c=cache->counters();
            result["cache"]={{"hits",c.hits},{"misses",c.misses},{"bypasses",c.bypasses},{"evictions",c.evictions},
                {"admissions",c.admissions},{"allocation_bypasses",c.allocation_bypasses},{"admission_rejects",c.admission_rejects}};
            const auto s=controller->status();
            result["controller"]={{"failed_samples",s.failed_samples},{"samples",s.samples},{"deferred_bytes",s.deferred}};
        }
        return result;
    }
};
RuntimeMemory::RuntimeMemory(Model model,const std::string & path,int ram_percent,int vram_percent)
    :impl(std::make_unique<Impl>(std::move(model),path,ram_percent,vram_percent)) {impl->initialize();}
RuntimeMemory::~RuntimeMemory()=default;
void RuntimeMemory::warm() {impl->warm();}
void RuntimeMemory::refresh() {impl->ram_limit(); if (impl->controller) impl->controller->refresh();}
nlohmann::json RuntimeMemory::snapshot() const {return impl->snapshot();}
}
