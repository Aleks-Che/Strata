#include "runtime_memory.hpp"
#include "expert_memory.hpp"
#include "expert_dispatch.hpp"
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
    uint64_t copy_fences=0;
    cudaMemPool_t pool=nullptr;
    void * staging=nullptr;
    uint64_t ram_touched=0,working_set_limit=0,d2d_bytes=0;
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
        ram_limit();
        if (!vram_percent && !pipeline_enabled) return;
        require(chunk_mib>=1 && chunk_mib<=16,"expert chunk must be 1..16 MiB");
        size_t free=0,total=0; require(probe(0,free,total),"global VRAM query unavailable; refusing per-process WDDM estimate");
        StrataVramPolicy policy; policy.mode=2;
        policy.target_mib=(total/100*vram_percent)/MiB;
        policy.reserve_mib=std::max<uint64_t>(128,(total-total/100*vram_percent+MiB-1)/MiB);
        ExpertCache::Admission admission; admission.frequency=true; admission.separate_branches=true;
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
        cache->set_branch_budgets(std::numeric_limits<size_t>::max(),mtp_cache_mib*MiB);
        if (pipeline_enabled) {
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
            transport=std::make_unique<ExpertTransport>(0,size_t(chunk_mib)*MiB,false,2,0,std::move(observer));
        }
        if (vram_percent) controller=std::make_unique<ExpertMemoryController>(*cache,total,policy,probe);
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
    void warm() {
        if (!ram_percent && !vram_percent && !pipeline_enabled) return;
        auto start=Clock::now();
        if (cache) {
            strata_glm_sync_stats unused{};
            // Round-robin layers/projections to avoid filling only early layers.
            for (int expert=0;;++expert) {
                bool eligible=false,full=false;
                for (const auto & item:tensors) {
                    const auto * t=item.first; if (expert>=t->ne[2]) continue; eligible=true;
                    if (controller) controller->refresh_if_due(std::chrono::milliseconds(100));
                    const auto k=key(t,expert);
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
        if (pipeline_enabled && staging) {cuda_check(cudaFreeHost(staging));staging=nullptr;}
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
void RuntimeMemory::refresh() {impl->ram_limit(); if (impl->controller) impl->controller->refresh();}
nlohmann::json RuntimeMemory::snapshot() const {return impl->snapshot();}
}
