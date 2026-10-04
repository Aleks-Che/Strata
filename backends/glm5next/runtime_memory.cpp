#include "runtime_memory.hpp"
#include "expert_memory.hpp"
#include "expert_dispatch.hpp"
#include "expert_slab.hpp"
#include "expert_warm_profile.hpp"
#include "host_pages.hpp"
#include "gpu_trace.hpp"
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
struct Ram {uint64_t total=0,available=0,working_set=0,page_faults=0;};
Ram ram() {
    Ram r;
#ifdef _WIN32
    MEMORYSTATUSEX s{}; s.dwLength=sizeof(s);
    require(GlobalMemoryStatusEx(&s)!=0,"global RAM query failed");
    PROCESS_MEMORY_COUNTERS p{}; p.cb=sizeof(p);
    require(GetProcessMemoryInfo(GetCurrentProcess(),&p,sizeof(p))!=0,"working set query failed");
    r={s.ullTotalPhys,s.ullAvailPhys,p.WorkingSetSize,p.PageFaultCount};
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
    bool pipeline_enabled; int chunk_mib; size_t mtp_cache_mib;
    std::unique_ptr<ExpertTransport> transport;
    std::unique_ptr<GpuTrace> trace;
    std::unique_ptr<ExpertDispatch> dispatch;
    std::vector<ExpertKey> dispatch_keys; size_t dispatch_next=0;
    StrataExpertPipeline::Counters accounted{};
    std::vector<std::pair<const ggml_tensor *,ExpertKey>> tensors;
    std::map<const ggml_tensor *,size_t> tensor_index;
    std::unique_ptr<ExpertCache> cache;
    std::unique_ptr<ExpertMemoryController> controller;
    ExpertMemoryController::Probe probe=make_global_memory_probe();
    cudaStream_t stream=nullptr;
    cudaEvent_t scratch_released=nullptr,copy_ready=nullptr;
    int event_copy=0; // 0 host waits; 1 per range; 2 one fence per expert tensor.
    int decode_readers=1;
    bool write_combined=false;
    uint64_t main_cache_decay=4096;
    bool pool_reclaim=false;
    uint64_t pool_reclaims=0;
    double pool_reclaim_ms=0;
    uint64_t copy_fences=0;
    cudaMemPool_t pool=nullptr;
    std::shared_ptr<ExpertSlabAllocator> slab;
    size_t slab_mib=0;
    std::filesystem::path profile_path;
    nlohmann::json profile_identity;
    WarmProfile warm_profile;
    bool profile_read_only=false;
    size_t profile_loaded=0,profile_saved=0;
    double profile_save_ms=0;
    std::map<std::string,size_t> profile_tensor_index;
    int ram_warm_mode=0; // 1: bounded startup scan of CPU-required experts.
    bool ram_diagnostics=false;
    uint64_t ram_skipped_gpu_bytes=0,ram_rotation_checks=0,essential_ws_estimate=0;
    bool ram_rotation_allowed=false;
    double residency_ms=0;
    nlohmann::json residency_before=nullptr,residency_after=nullptr;
    void * staging=nullptr;
    uint64_t ram_touched=0,working_set_limit=0,d2d_bytes=0;
    std::string ram_warm_stop="off";
    double warm_ms=0;
    bool hook=false;
#ifdef _WIN32
    SIZE_T old_min=0,old_max=0; DWORD old_flags=0; bool ws_changed=false;
#endif
    Impl(Model m,const std::string & path,int rp,int vp,bool pipe,int chunk,size_t mtp_limit):model(std::move(m)),
        identity(std::filesystem::weakly_canonical(path).u8string()),ram_percent(rp),vram_percent(vp),pipeline_enabled(pipe),chunk_mib(chunk),mtp_cache_mib(mtp_limit) {
        static std::atomic<uint64_t> next{0}; generation=++next;
    }
    ~Impl() {
        if (hook) {strata_glm_sync_event_copy(false);strata_glm_sync_compute_hook(nullptr);strata_glm_sync_plan_hooks(nullptr,nullptr);strata_glm_sync_copy_hook(nullptr,nullptr);}
        dispatch.reset(); transport.reset();
        if (stream) cudaStreamSynchronize(stream);
        controller.reset(); cache.reset(); // Drain leases before releasing stream/source.
        slab.reset();
        if (stream) cudaStreamSynchronize(stream);
        if (pool) cudaMemPoolDestroy(pool);
        if (staging) cudaFreeHost(staging);
        if (scratch_released) cudaEventDestroy(scratch_released);
        if (copy_ready) cudaEventDestroy(copy_ready);
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
        if (!ram_percent && !vram_percent && !pipeline_enabled) return;
        std::regex pattern("blk\\.([0-9]+)\\.ffn_(gate|up|down)_exps\\.weight");
        for (const auto & item:model->tensors_by_name) {
            std::smatch match; const auto * t=item.second;
            if (!std::regex_match(item.first,match,pattern)) continue;
            require(t->data && ggml_backend_buffer_is_host(t->buffer),"memory targets require host expert tensors");
            require(t->nb[2]*size_t(t->ne[2])==ggml_nbytes(t),"expert tensor must be contiguous");
            const auto projection=match[2]=="gate"?Projection::gate:match[2]=="up"?Projection::up:Projection::down;
            // Runtime namespace uses tensor-relative offsets, never claims to be
            // GGUF file offsets. The retained model owns the underlying mapping.
            const int layer=std::stoi(match[1]);
            ExpertKey k{identity,generation,layer<int(model->hparams.n_layer())?Branch::main:Branch::mtp,layer,0,projection,
                ggml_type_name(t->type),uint64_t(t->ne[0]),uint64_t(t->ne[1]),"runtime-tensor:"+item.first,0,0};
            tensor_index[t]=tensors.size(); tensors.push_back({t,std::move(k)});
        }
        require(!tensors.empty(),"no routed expert tensors found");
        if(const auto *value=std::getenv("STRATA_GLM_EXPERT_PROFILE")) {
            if(*value) {
                profile_path=std::filesystem::u8path(value);
                profile_identity={{"model",identity},{"file_bytes",std::filesystem::file_size(path_from_identity())},
                    {"file_time",std::filesystem::last_write_time(path_from_identity()).time_since_epoch().count()},
                    {"tensors",nlohmann::json::array()}};
                std::vector<size_t> counts;
                for(size_t i=0;i<tensors.size();++i) {
                    const auto *t=tensors[i].first;const auto &k=tensors[i].second;
                    profile_tensor_index[k.shard]=i;counts.push_back(size_t(t->ne[2]));
                    profile_identity["tensors"].push_back({k.shard,k.quant,t->ne[0],t->ne[1],t->ne[2],t->nb[2]});
                }
                warm_profile=read_warm_profile(profile_path,profile_identity,counts);
            }
        }
        if(const auto *value=std::getenv("STRATA_GLM_EXPERT_PROFILE_READ_ONLY")) {
            require(std::strcmp(value,"0")==0 || std::strcmp(value,"1")==0,"STRATA_GLM_EXPERT_PROFILE_READ_ONLY must be 0 or 1");
            profile_read_only=*value=='1';
        }
        if(const auto *value=std::getenv("STRATA_GLM_RAM_WARM_MODE")) {
            require(std::strcmp(value,"0")==0 || std::strcmp(value,"1")==0,"STRATA_GLM_RAM_WARM_MODE must be 0 or 1");
            ram_warm_mode=*value-'0';
        }
        if(const auto *value=std::getenv("STRATA_GLM_RAM_DIAGNOSTICS")) {
            require(std::strcmp(value,"0")==0 || std::strcmp(value,"1")==0,"STRATA_GLM_RAM_DIAGNOSTICS must be 0 or 1");
            ram_diagnostics=*value=='1';
        }
#ifndef _WIN32
        require(!ram_warm_mode && !ram_diagnostics,"GLM bounded host scan/diagnostics currently require Windows");
#endif
        ram_limit();
        if (!vram_percent && !pipeline_enabled) return;
        require(chunk_mib>=1 && chunk_mib<=16,"expert chunk must be 1..16 MiB");
        size_t free=0,total=0; require(probe(0,free,total),"global VRAM query unavailable; refusing per-process WDDM estimate");
        StrataVramPolicy policy; policy.mode=2;
        policy.target_mib=(total/100*vram_percent)/MiB;
        policy.reserve_mib=std::max<uint64_t>(128,(total-total/100*vram_percent+MiB-1)/MiB);
        ExpertCache::Admission admission; admission.frequency=true; admission.separate_branches=true;
        if (const auto * value=std::getenv("STRATA_GLM_MAIN_CACHE_DECAY")) {
            const auto n=std::strlen(value);
            require(n>0 && n<=7 && std::all_of(value,value+n,[](char c){return c>='0' && c<='9';}),"STRATA_GLM_MAIN_CACHE_DECAY must be 1..1048576");
            main_cache_decay=std::stoul(value);
            require(main_cache_decay>=1 && main_cache_decay<=1048576,"STRATA_GLM_MAIN_CACHE_DECAY must be 1..1048576");
        }
        admission.decay_period=main_cache_decay;
        admission.mtp_decay_period=4096; // One draft layer has a much shorter access stream.
        if (const auto * value=std::getenv("STRATA_GLM_POOL_RECLAIM")) {
            require(std::strcmp(value,"0")==0 || std::strcmp(value,"1")==0,"STRATA_GLM_POOL_RECLAIM must be 0 or 1");
            pool_reclaim=value[0]=='1';
        }
        cuda_check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        int supports_pool=0; cuda_check(cudaDeviceGetAttribute(&supports_pool,cudaDevAttrMemoryPoolsSupported,0));
        if (const auto * value=std::getenv("STRATA_GLM_MEMORY_POOL")) {
            require(std::strcmp(value,"0")==0 || std::strcmp(value,"1")==0,"STRATA_GLM_MEMORY_POOL must be 0 or 1");
            if (value[0]=='0') supports_pool=0;
        }
        if (const auto * value=std::getenv("STRATA_GLM_CACHE_SLAB_MIB")) {
            const auto n=std::strlen(value);
            require(n>0 && n<=3 && std::all_of(value,value+n,[](char c){return c>='0' && c<='9';}),"STRATA_GLM_CACHE_SLAB_MIB must be 0 or 4..256");
            slab_mib=std::stoul(value);
            require(slab_mib==0 || (slab_mib>=4 && slab_mib<=256),"STRATA_GLM_CACHE_SLAB_MIB must be 0 or 4..256");
        }
        if (slab_mib) {
            policy.reserve_mib+=2; // Same physical margin as the slab growth check.
            slab=std::make_shared<ExpertSlabAllocator>(slab_mib*MiB,[this](size_t bytes) {
                if(!vram_percent)return bytes;
                size_t available=0,capacity=0;
                if(!probe(0,available,capacity) || !capacity || available>capacity)return size_t(0);
                const size_t reserve=capacity-capacity/100*vram_percent;
                // A new block consumes more than the first entry's logical
                // bytes. Check physical headroom before committing the block;
                // retain 2 MiB for driver allocation granularity/sample lag.
                return available>=reserve && available-reserve>=2*MiB?std::min(bytes,available-reserve-2*MiB):size_t(0);
            });
            cache=std::make_unique<ExpertCache>(0,admission,
                [allocator=slab](void **p,size_t n){return allocator->allocate(p,n);},
                [allocator=slab](void *p){return allocator->release(p);});
        } else if (supports_pool) {
            cudaMemPoolProps props{}; props.allocType=cudaMemAllocationTypePinned;
            props.location.type=cudaMemLocationTypeDevice; props.location.id=0;
            cuda_check(cudaMemPoolCreate(&pool,&props));
            // Default release threshold zero: unused physical pages are returned
            // at synchronization. Eviction followed by allocation can reuse them.
            cache=std::make_unique<ExpertCache>(0,admission,
                [this](void **p,size_t n){return cudaMallocFromPoolAsync(p,n,pool,stream);},
                [this](void *p){return cudaFreeAsync(p,stream);});
        } else cache=std::make_unique<ExpertCache>(0,admission);
        cache->set_branch_budgets(std::numeric_limits<size_t>::max(),mtp_cache_mib*MiB);
        if (pipeline_enabled) {
            if (const auto * value=std::getenv("STRATA_GLM_DECODE_READERS")) {
                require(std::strlen(value)==1 && value[0]>='1' && value[0]<='4',"STRATA_GLM_DECODE_READERS must be 1..4");
                decode_readers=value[0]-'0';
            }
            if (const auto * value=std::getenv("STRATA_GLM_WRITE_COMBINED")) {
                require(std::strcmp(value,"0")==0 || std::strcmp(value,"1")==0,"STRATA_GLM_WRITE_COMBINED must be 0 or 1");
                write_combined=value[0]=='1';
            }
            if (const auto * value=std::getenv("STRATA_GLM_COPY_EVENTS")) {
                require(std::strcmp(value,"0")==0 || std::strcmp(value,"1")==0 || std::strcmp(value,"2")==0,"STRATA_GLM_COPY_EVENTS must be 0, 1 or 2");
                event_copy=std::atoi(value);
            }
            if (event_copy) {
                cuda_check(cudaEventCreateWithFlags(&scratch_released,cudaEventDisableTiming));
                cuda_check(cudaEventCreateWithFlags(&copy_ready,cudaEventDisableTiming));
            }
            StrataExpertPipeline::CopyObserver observer;
            if (const auto * n=std::getenv("STRATA_GLM_TRACE_GRAPHS")) {
                trace=std::make_unique<GpuTrace>(std::stoi(n));
                observer=[this](cudaStream_t stream,bool begin,size_t) {trace->record(stream,nullptr,begin);};
            }
            transport=std::make_unique<ExpertTransport>(0,size_t(chunk_mib)*MiB,write_combined,std::max(2,decode_readers),0,std::move(observer),decode_readers);
        }
        if (vram_percent) controller=std::make_unique<ExpertMemoryController>(*cache,total,policy,
            [this](int device,size_t &free,size_t &total) {
                if (pool && pool_reclaim) {
                    // Only on actual budget samples, at safe dispatch boundaries
                    // (500 ms cadence in decode). Host-observe async frees before
                    // trimming; otherwise unused pool pages count as non-cache
                    // pressure and repeatedly shrink the logical cache budget.
                    const auto started=Clock::now();
                    cuda_check(cudaStreamSynchronize(stream));
                    cuda_check(cudaMemPoolTrimTo(pool,0));
                    ++pool_reclaims;pool_reclaim_ms+=milliseconds(started);
                }
                return probe(device,free,total);
            });
        cuda_check(cudaHostAlloc(&staging,staging_size,cudaHostAllocDefault));
        if (controller) require(controller->refresh().sample_valid,"initial global VRAM sample invalid");
        strata_glm_sync_copy_hook([](void * owner,ggml_backend * backend,ggml_tensor * dst,const ggml_tensor * src,
                                    size_t offset,size_t bytes,size_t count,strata_glm_sync_stats * stats) {
            static_cast<Impl *>(owner)->copy(backend,dst,src,offset,bytes,count,*stats);
        },this); hook=true;
        strata_glm_sync_event_copy(event_copy);
        if (pipeline_enabled) strata_glm_sync_plan_hooks(
            [](void * owner,const strata_glm_expert * entries,size_t n,bool decode,strata_glm_sync_stats * stats) {
                static_cast<Impl *>(owner)->begin_plan(entries,n,decode,*stats);
            },[](void * owner,bool cancel,strata_glm_sync_stats * stats) {
                auto * self=static_cast<Impl *>(owner);self->end_plan(cancel,stats);
                if (self->trace) self->trace->finish(cancel);
            });
        if (trace) strata_glm_sync_compute_hook([](void * owner,ggml_backend * backend,bool begin) {
            static_cast<Impl *>(owner)->trace->record(nullptr,backend,begin);
        });
    }
    void account(strata_glm_sync_stats & stats) {
        if (!transport) return;
        const auto c=transport->counters();
        stats.source_bytes+=(c.file_bytes+c.mmap_bytes)-(accounted.file_bytes+accounted.mmap_bytes);
        stats.h2d_bytes+=c.h2d_bytes-accounted.h2d_bytes;
        stats.source_ms+=double(c.read_us-accounted.read_us)/1000;
        stats.chunks+=c.chunks-accounted.chunks;
        stats.staging_bytes=c.pinned_bytes;
        // H2D submission/consumer wait are exposed separately in snapshot().
        // They are not the synchronous H2D completion timer from the baseline.
        accounted=c;
    }
    void end_plan(bool cancel,strata_glm_sync_stats * stats) {
        if (dispatch) {
            if (cancel) dispatch->cancel(); else dispatch->finish();
            dispatch.reset(); dispatch_keys.clear(); dispatch_next=0;
        }
        // A failed copy may not have published its final compute dependency.
        // Drain before error recovery can clear/reuse either context's scratch.
        if (cancel && event_copy) cuda_check(cudaStreamSynchronize(stream));
        if (stats) account(*stats);
    }
    void begin_plan(const strata_glm_expert * entries,size_t n,bool decode,strata_glm_sync_stats & stats) {
        end_plan(false,&stats);
        if (controller) controller->refresh_if_due(std::chrono::milliseconds(500));
        std::vector<ExpertSourceView> sources; std::set<const ggml_tensor *> bound;
        for (size_t i=0;i<n;++i) {
            const auto * t=entries[i].tensor; auto k=key(t,entries[i].expert);
            if (bound.insert(t).second) sources.push_back({k.model,k.generation,k.shard,
                static_cast<const uint8_t *>(t->data),ggml_nbytes(t),std::shared_ptr<const void>(model,t->data)});
            dispatch_keys.push_back(std::move(k));
        }
        dispatch=std::make_unique<ExpertDispatch>(*cache,*transport,dispatch_keys,std::move(sources),decode);
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
    void copy(ggml_backend * backend,ggml_tensor * dst,const ggml_tensor * src,size_t offset,size_t bytes,size_t count,strata_glm_sync_stats & stats) {
        const auto stride=src->nb[2];
        require(offset%stride==0 && count>0 && count<=size_t(src->ne[2])-offset/stride,"invalid selected expert span");
        const size_t first=offset/stride;
        // The scheduler copies all selected ranges of one tensor consecutively,
        // then submits its split. Gate/up/down use distinct runtime namespaces.
        // A single pair of events can therefore protect the entire tensor.
        const bool first_range=event_copy!=2 || dispatch_next==0 ||
            dispatch_keys[dispatch_next-1].shard!=tensors.at(tensor_index.at(src)).second.shard;
        if (event_copy && first_range) {
            require(std::strstr(ggml_backend_name(backend),"CUDA")!=nullptr,"event copy requires the audited CUDA backend");
            // Events are recorded/waited immediately on the real backend stream.
            // Re-recording is safe: each wait captures the preceding record.
            // The pinned CUDA backend stores cudaEvent_t in event.context.
            ggml_backend_event released{ggml_backend_get_device(backend),scratch_released};
            ggml_backend_event_record(&released,backend);
            cuda_check(cudaStreamWaitEvent(stream,scratch_released,0));
        } else if (!event_copy) ggml_backend_synchronize(backend);
        if (pipeline_enabled) {
            require(bool(dispatch),"selected copy has no router lookahead plan");
            ++stats.ranges;
            for (size_t i=first;i<first+count;++i) {
                const auto k=key(src,int(i));
                require(dispatch_next<dispatch_keys.size() && dispatch_keys[dispatch_next]==k,"selected copy order differs from router plan");
                dispatch->copy(dispatch_next++,static_cast<uint8_t *>(dst->data)+k.offset,stream);
                d2d_bytes+=k.bytes;
            }
            const bool last_range=event_copy!=2 || dispatch_next==dispatch_keys.size() ||
                dispatch_keys[dispatch_next].shard!=tensors.at(tensor_index.at(src)).second.shard;
            if (event_copy && last_range) {
                cuda_check(cudaEventRecord(copy_ready,stream));
                ggml_backend_event ready{ggml_backend_get_device(backend),copy_ready};
                ggml_backend_event_wait(backend,&ready);++copy_fences;
            } else if (!event_copy) cuda_check(cudaStreamSynchronize(stream));
            account(stats); return;
        }
        if (controller) controller->refresh_if_due(std::chrono::milliseconds(500));
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
    std::filesystem::path path_from_identity() const {return std::filesystem::u8path(identity);}
    nlohmann::json sample_residency() {
        const auto started=Clock::now();HostResidency groups[2];
        for(const auto &item:tensors) {
            const auto *t=item.first;
            for(int expert=0;expert<t->ne[2];++expert) {
                const auto k=key(t,expert);const bool gpu=cache && cache->resident(k);
                const auto sample=sample_host_residency(static_cast<const uint8_t *>(t->data)+k.offset,t->nb[2]);
                auto &g=groups[gpu?1:0];g.bytes+=sample.bytes;g.samples+=sample.samples;
                g.valid_samples+=sample.valid_samples;g.estimated_resident_bytes+=sample.estimated_resident_bytes;g.valid=g.valid && sample.valid;
            }
        }
        nlohmann::json result;
        for(int i=0;i<2;++i) {
            const auto &g=groups[i];result[i?"gpu_cached":"cpu_required"]={{"valid",g.valid},{"bytes",g.bytes},
                {"samples",g.samples},{"valid_samples",g.valid_samples},{"estimated_resident_bytes",g.estimated_resident_bytes}};
        }
        result["probe_stride_bytes"]=1<<20;result["sample_ms"]=milliseconds(started);residency_ms+=milliseconds(started);
        return result;
    }
    void checkpoint() {
        if(profile_path.empty() || profile_read_only || !warm_profile.writable || !cache)return;
        auto start=Clock::now();
        try {
            std::vector<WarmExpert> entries;
            for(const auto &item:cache->warm_entries()) {
                const auto index=profile_tensor_index.at(item.first.shard);
                entries.push_back({index,item.first.expert,item.second});
            }
            if(!entries.empty()) {write_warm_profile(profile_path,profile_identity,entries);profile_saved=entries.size();}
        }catch(const std::exception &e) {
            warm_profile.writable=false;
            std::cerr<<"STRATA_GLM expert profile save disabled: "<<e.what()<<"\n";
        }
        profile_save_ms+=milliseconds(start);
    }
    void warm_ram() {
        if(!ram_percent)return;
        require(!dispatch && (!transport || !transport->in_progress()),"host warmup requires an idle expert transport");
        ram_limit();auto last_limit=Clock::now();
        const auto initial=ram();
        const auto before=(ram_diagnostics || ram_warm_mode)?sample_residency():nlohmann::json(nullptr);
        if(ram_warm_mode) {
            HostResidency cpu,gpu;
            cpu.bytes=before["cpu_required"]["bytes"].get<uint64_t>();
            cpu.estimated_resident_bytes=before["cpu_required"]["estimated_resident_bytes"].get<uint64_t>();
            gpu.estimated_resident_bytes=before["gpu_cached"]["estimated_resident_bytes"].get<uint64_t>();
            essential_ws_estimate=required_host_working_set(initial.working_set,cpu,gpu);
            ram_rotation_allowed=before["cpu_required"]["valid"].get<bool>() && before["gpu_cached"]["valid"].get<bool>() && host_scan_fits(essential_ws_estimate,working_set_limit);
        }
        if(ram_diagnostics)residency_before=before;
        ram_warm_stop="all_experts";volatile uint8_t sink=0;bool full=false;size_t until_check=0;
        for(const auto &item:tensors) {
            const auto *t=item.first;auto *source=static_cast<const volatile uint8_t *>(t->data);
            for(size_t offset=0;offset<ggml_nbytes(t);) {
                const size_t expert_end=ram_warm_mode?offset+(t->nb[2]-offset%t->nb[2]):ggml_nbytes(t);
                if(ram_warm_mode && cache && cache->resident(key(t,int(offset/t->nb[2])))) {
                    ram_skipped_gpu_bytes+=expert_end-offset;offset=expert_end;continue;
                }
                if(!ram_warm_mode || !until_check) {
                    auto r=ram();const auto target=r.total/100*ram_percent;
                    if(ram_rotation_allowed && r.total-r.available>=target && Clock::now()-last_limit>=std::chrono::milliseconds(500)) {
                        // Recalibrate the existing HARDWS maximum to account for
                        // other processes. No page discard/unlock and no raised cap.
                        ram_limit();last_limit=Clock::now();r=ram();
                        ram_rotation_allowed=host_scan_fits(essential_ws_estimate,working_set_limit);
                    }
                    if(r.total-r.available>=target && !ram_rotation_allowed) {ram_warm_stop="global_target";full=true;break;}
                    if(working_set_limit && r.working_set>=working_set_limit-std::min<uint64_t>(working_set_limit,32*MiB)) {
                        if(!ram_rotation_allowed) {ram_warm_stop="working_set_limit";full=true;break;}
                        ++ram_rotation_checks;
                    }
                    until_check=32*MiB;
                }
                const auto end=std::min(offset+until_check,expert_end);
                for(size_t p=offset;p<end;p+=4096)sink=uint8_t(sink^source[p]);
                if(ram_warm_mode)sink=uint8_t(sink^source[end-1]);
                until_check-=end-offset;ram_touched+=end-offset;offset=end;
            }
            if(full)break;
        }
        (void)sink;
        if(ram_diagnostics)residency_after=sample_residency();
    }
    void warm() {
        if (!ram_percent && !vram_percent && !pipeline_enabled) return;
        auto start=Clock::now();
        if (cache) {
            strata_glm_sync_stats unused{};
            // Load learned resident entries in descending frequency order.
            // These are placement hints; all weights still come from this model.
            for(const auto &entry:warm_profile.entries) {
                const auto *t=tensors[entry.tensor].first;const auto k=key(t,entry.expert);
                if(controller)controller->refresh_if_due(std::chrono::milliseconds(100));
                if(k.bytes>cache->byte_budget()-std::min(cache->byte_budget(),cache->resident_bytes()) ||
                   k.bytes>cache->byte_budget(k.branch)-std::min(cache->byte_budget(k.branch),cache->resident_bytes(k.branch)))continue;
                const auto *source=static_cast<const uint8_t *>(t->data)+k.offset;
                auto lease=cache->get(k,std::shared_ptr<const void>(model,source),stream,
                    [&](void *dest,size_t n,cudaStream_t){upload(dest,source,n,unused);});
                if(lease) {cache->seed_frequency(k,entry.score);++profile_loaded;}
                lease.release();cuda_check(cudaStreamSynchronize(stream));
            }
            // Round-robin layers/projections to avoid filling only early layers.
            for (int expert=0;;++expert) {
                bool eligible=false,full=false;
                for (const auto & item:tensors) {
                    const auto * t=item.first; if (expert>=t->ne[2]) continue; eligible=true;
                    if (controller) controller->refresh_if_due(std::chrono::milliseconds(100));
                    const auto k=key(t,expert);
                    if(cache->resident(k))continue;
                    if (cache->resident_bytes()+k.bytes>cache->byte_budget()) {full=true;break;}
                    if (k.bytes>cache->byte_budget(k.branch)-std::min(cache->byte_budget(k.branch),cache->resident_bytes(k.branch))) continue;
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
        warm_ram();
        if (pipeline_enabled && staging) {cuda_check(cudaFreeHost(staging));staging=nullptr;}
        warm_ms=milliseconds(start);
    }
    nlohmann::json snapshot() const {
        const auto r=ram(); size_t free=0,total=0; const bool valid=probe(0,free,total);
        nlohmann::json result={{"ram_target_percent",ram_percent},{"vram_target_percent",vram_percent},
            {"ram_total_bytes",r.total},{"ram_used_bytes",r.total-r.available},{"working_set_bytes",r.working_set},
            {"working_set_limit_bytes",working_set_limit},{"ram_warm_touched_bytes",ram_touched},{"warm_ms",warm_ms},
            {"ram_warm_stop",ram_warm_stop},
            {"process_page_faults",r.page_faults},
            {"global_vram_valid",valid},{"global_vram_total_bytes",total},{"global_vram_used_bytes",total-free},
            {"cache_resident_bytes",cache?cache->resident_bytes():0},{"cache_budget_bytes",cache?cache->byte_budget():0},
            {"cache_d2d_bytes",d2d_bytes},{"cuda_memory_pool",pool!=nullptr}};
        result["expert_cache_slab_mib"]=slab_mib;
        result["ram_warm_mode"]=ram_warm_mode;result["ram_diagnostics"]=ram_diagnostics;
        result["ram_warm_skipped_gpu_bytes"]=ram_skipped_gpu_bytes;
        result["ram_host_scan"]={{"rotation_allowed",ram_rotation_allowed},{"rotation_checks",ram_rotation_checks},
            {"essential_ws_estimate_bytes",essential_ws_estimate}};
        result["ram_residency"]={{"before_warm",residency_before},{"after_warm",residency_after},{"ms",residency_ms}};
        result["expert_warm_profile"]={{"status",profile_path.empty()?"off":warm_profile.status},
            {"read_only",profile_read_only},{"writable",!profile_path.empty() && warm_profile.writable && !profile_read_only},
            {"candidates",warm_profile.entries.size()},{"loaded",profile_loaded},{"saved",profile_saved},
            {"save_ms",profile_save_ms}};
        if(slab) {
            const auto s=slab->status();
            result["cache_slab"]={{"reserved_bytes",s.reserved},{"requested_bytes",s.requested},
                {"slot_bytes",s.slot_bytes},{"unused_bytes",s.reserved-s.requested},{"blocks",s.blocks},
                {"allocations",s.allocations},{"block_allocations",s.block_allocations},
                {"reuses",s.reuses},{"growth_denied",s.growth_denied},{"partial_blocks",s.partial_blocks}};
        }
        if (pool) {
            uint64_t reserved=0,used=0;
            cuda_check(cudaMemPoolGetAttribute(pool,cudaMemPoolAttrReservedMemCurrent,&reserved));
            cuda_check(cudaMemPoolGetAttribute(pool,cudaMemPoolAttrUsedMemCurrent,&used));
            result["cache_pool_reserved_bytes"]=reserved;
            result["cache_pool_used_bytes"]=used;
            result["cache_pool_unused_bytes"]=reserved-std::min(reserved,used);
        }
        if (cache) {
            result["main_cache_bytes"]=cache->resident_bytes(Branch::main);
            result["mtp_cache_bytes"]=cache->resident_bytes(Branch::mtp);
            result["mtp_cache_limit_bytes"]=cache->byte_budget(Branch::mtp);
            const auto c=cache->counters();
            result["cache"]={{"hits",c.hits},{"misses",c.misses},{"bypasses",c.bypasses},{"evictions",c.evictions},
                {"admissions",c.admissions},{"allocation_bypasses",c.allocation_bypasses},{"admission_rejects",c.admission_rejects}};
            if (controller) {const auto s=controller->status();
            result["controller"]={{"failed_samples",s.failed_samples},{"samples",s.samples},{"deferred_bytes",s.deferred}};}
        }
        result["expert_pipeline"]=pipeline_enabled;
        result["expert_copy_events"]=event_copy;
        result["expert_decode_readers"]=decode_readers;
        result["expert_write_combined"]=int(write_combined);
        result["main_cache_decay"]=main_cache_decay;
        result["mtp_cache_decay"]=4096;
        result["expert_pool_reclaim"]=int(pool_reclaim);
        result["pool_reclaims"]=pool_reclaims;
        result["pool_reclaim_ms"]=pool_reclaim_ms;
        result["expert_copy_fences"]=copy_fences;
        if (trace) result["gpu_trace"]=trace->snapshot();
        if (transport) {
            const auto c=transport->counters();
            result["pipeline"]={{"groups",c.groups},{"chunks",c.chunks},{"source_bytes",c.file_bytes+c.mmap_bytes},
                {"h2d_bytes",c.h2d_bytes},{"d2d_bytes",c.d2d_bytes},{"read_us",c.read_us},
                {"submit_us",c.submit_us},{"consumer_wait_us",c.consumer_wait_us},{"slot_wait_us",c.slot_wait_us},
                {"unused_bytes",c.unused_bytes},{"pinned_bytes",c.pinned_bytes},{"device_ring_bytes",c.device_ring_bytes},
                {"queued_peak",c.queued_peak},{"read_peak",c.read_peak}};
        }
        return result;
    }
};
RuntimeMemory::RuntimeMemory(Model model,const std::string & path,int ram_percent,int vram_percent,bool pipeline,int chunk_mib,size_t mtp_cache_mib)
    :impl(std::make_unique<Impl>(std::move(model),path,ram_percent,vram_percent,pipeline,chunk_mib,mtp_cache_mib)) {impl->initialize();}
RuntimeMemory::~RuntimeMemory()=default;
void RuntimeMemory::warm() {impl->warm();}
void RuntimeMemory::refresh() {
    impl->ram_limit();
    if (impl->controller) impl->controller->refresh();
}
void RuntimeMemory::checkpoint() {impl->checkpoint();}
nlohmann::json RuntimeMemory::snapshot() const {return impl->snapshot();}
}
