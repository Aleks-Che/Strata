#pragma once
#include "gpu_fetch_queue.cuh"
#include "host_cache.hpp"
#include "expert_file.hpp"
#include "gpu_fetch_reader.hpp"

// Immutable GPU/RAM snapshot for one MoE branch. The service may recycle an
// unleased RAM block; payload allocation and publication stay on the owner.
struct StrataGpuFetchLane {
    bool enabled=false,active=false;
    bool early_retire=false;
    bool async_park=false;
    uint64_t early_retire_branches=0,early_retire_fallbacks=0,retire_tail_pending=0,retire_overlap_us=0;
    bool submitted[3]={},published[3]={},early_sources=false;
    uint64_t epoch=0,branches=0,matrices=0,ram_bytes=0,source_bytes=0,h2d_bytes=0;
    uint64_t demotion_epoch=0;
    strata_ds4::GpuFetchPlan plan[3]{};
    std::shared_ptr<strata_expert_file::Source> files[3];
    uint64_t offsets[3]={};
    std::array<std::array<strata_ds4::HostCache::Lease,256>,3> leases;
    std::array<std::array<strata_ds4::HostCache::Lease,256>,3> epoch_leases;
    bool ram_hit[3][256]={};
    bool release_ram=false;
    std::function<void(const bool *)> release_cpu_candidates;
    uint64_t released_unused=0,released_used=0,released_bytes=0,release_deferred=0,release_us=0;
    std::array<std::array<std::unique_ptr<strata_ds4::HostCache::Fill>,64>,3> fills;
    bool filled[3][64]={};
    std::shared_ptr<strata_ds4::HostCache> ram;
    std::unique_ptr<strata_ds4::HostCache::RecycleWindow> recycle_window;
    const uint8_t *weights[3]={};
    size_t reserve_limit=0;
    int cache_slots=0;
    int cache_keep_ram=0; // 0: off, 1: ready triple, 2: assigned in this branch
    bool cache_current_victims=false;
    bool cache_tail_limit=false;
    bool cache_demand=false;
    size_t cache_demand_max_bytes=0; // zero: all cache budgets
    bool cache_rank=false;
    size_t cache_rank_min_bytes=0;
    unsigned cache_rank_margin=1;
    int cache_rank_scan=64;
    uint64_t cache_rank_plans=0,cache_rank_rejected=0,cache_rank_admitted=0,cache_rank_cost=0;
    uint64_t cache_demand_plans=0,cache_demand_misses=0,cache_demand_scanned=0;
    uint64_t cache_tail_offered=0,cache_tail_skipped=0;
    uint64_t cache_current_offered=0,cache_current_admitted=0,cache_current_protected=0;
    uint64_t cache_ram_bypasses=0,route_ready=0,route_partial=0,route_allgpu=0,route_allmiss=0,route_cpu_only=0;
    uint64_t cache_ram_unblocked=0,cache_ram_promoted=0;
    uint64_t cache_offered=0,cache_admitted=0,cache_restored=0,cache_bytes=0,cache_prepare_us=0,cache_retire_us=0;
    uint64_t cache_duplicates=0;
    size_t pending_bytes=0;
    uint64_t reserved=0,completed=0,unused=0,reserved_bytes=0,completed_bytes=0;
    uint64_t prepare_us=0,retire_us=0,owner_wait_us=0,read_us=0;
    strata_ds4::GpuFetchReader reader;
    std::unique_ptr<strata_ds4::GpuFetchQueue> queue;
    StrataGpuFetchLane() {
        const char *v=std::getenv("STRATA_DS4_GPU_FETCH_QUEUE");enabled=v && std::strcmp(v,"1")==0;
        // The shadow route recorder requires every original host route. Keep
        // its existing semantics instead of silently writing a partial trace.
        if(std::getenv("STRATA_DS4_ROUTE_TRACE"))enabled=false;
        if(const char *v=std::getenv("STRATA_DS4_GPU_FETCH_RAM_MIB"))
            reserve_limit=size_t(std::max(0,std::min(128,std::atoi(v))))<<20;
        if(const char *v=std::getenv("STRATA_DS4_GPU_FETCH_CACHE_SLOTS"))
            cache_slots=std::max(0,std::min(strata_ds4::gpu_fetch_cache_slots,std::atoi(v)));
        if(const char *v=std::getenv("STRATA_DS4_GPU_FETCH_KEEP_RAM"))cache_keep_ram=std::atoi(v)==2?2:std::atoi(v)!=0;
        if(const char *v=std::getenv("STRATA_DS4_GPU_FETCH_CURRENT_VICTIMS"))cache_current_victims=std::atoi(v)!=0;
        if(const char *v=std::getenv("STRATA_DS4_GPU_FETCH_TAIL_LIMIT"))cache_tail_limit=std::atoi(v)!=0;
        if(const char *v=std::getenv("STRATA_DS4_GPU_FETCH_DEMAND_CACHE"))cache_demand=std::atoi(v)!=0;
        if(const char *v=std::getenv("STRATA_DS4_GPU_FETCH_DEMAND_MAX_CACHE_MIB"))
            cache_demand_max_bytes=size_t(std::max(0,std::min(65536,std::atoi(v))))<<20;
        if(const char *v=std::getenv("STRATA_DS4_GPU_FETCH_DEMAND_RANK"))cache_rank=std::atoi(v)!=0;
        if(const char *v=std::getenv("STRATA_DS4_GPU_FETCH_DEMAND_RANK_MIN_CACHE_MIB"))
            cache_rank_min_bytes=size_t(std::max(0,std::min(65536,std::atoi(v))))<<20;
        if(const char *v=std::getenv("STRATA_DS4_GPU_FETCH_DEMAND_RANK_MARGIN"))cache_rank_margin=unsigned(std::max(0,std::min(16,std::atoi(v))));
        if(const char *v=std::getenv("STRATA_DS4_GPU_FETCH_DEMAND_RANK_SCAN"))cache_rank_scan=std::max(8,std::min(1024,std::atoi(v)));
        if(const char *v=std::getenv("STRATA_DS4_GPU_FETCH_RELEASE_RAM"))release_ram=std::atoi(v)!=0;
        if(const char *v=std::getenv("STRATA_DS4_GPU_FETCH_EARLY_RETIRE"))early_retire=std::atoi(v)!=0;
        if(const char *v=std::getenv("STRATA_DS4_QUEUE_ASYNC_PARK"))async_park=std::atoi(v)!=0;
        if(const char *v=std::getenv("STRATA_DS4_GPU_FETCH_EARLY_SOURCES"))early_sources=std::atoi(v)!=0;
    }
    void allocate(size_t cache_budget) {
        if(queue)return;
        queue=std::make_unique<strata_ds4::GpuFetchQueue>([this](int m,auto &slot,auto &service){execute(m,slot,service);});
        // Mapped consumers have no owner fence at which destinations can change.
        cache_demand=cache_demand && queue->owner_dma && (!cache_demand_max_bytes || cache_budget<=cache_demand_max_bytes);
        cache_rank=cache_rank && cache_demand && cache_budget>=cache_rank_min_bytes;
        queue->owner_admission=cache_demand;
    }
    void prepare_fills() {
        if(!reserve_limit || !ram)return;
        const auto start=std::chrono::steady_clock::now();
        ram->collect();
        pending_bytes=0;
        for(int m=0;m<3;++m) {
            for(int i=0;i<64;++i) {filled[m][i]=false;if(!fills[m][i])fills[m][i]=ram->make_fill();}
            ram->prime_spare(files[m],plan[m].width+std::min<size_t>(plan[m].width,512));
            ram->prime_spare(files[m],plan[m].width);
        }
        std::vector<size_t> sizes;
        for(const auto &p:plan){sizes.push_back(p.width);sizes.push_back(p.width+std::min<size_t>(p.width,512));}
        recycle_window=ram->reserve_recycling(sizes,reserve_limit);
        prepare_us+=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-start).count();
    }
    void execute(int m,strata_ds4::GpuFetchSlot &slot,strata_ds4::GpuFetchQueue &service) {
        const auto &job=slot.job;const auto &p=plan[m];
        if(!active || job.epoch!=epoch || job.epoch!=p.epoch || job.count<0 || job.count>64 || job.used_count<0 || job.used_count>64) {
            slot.error=strata_ds4::cpu_bad_request;return;
        }
        bool selected[256]={},used[256]={},claimed[strata_ds4::gpu_fetch_cache_slots]={};
        for(int i=0;i<job.used_count;++i) {
            const int id=job.used_ids[i];
            if(id<0 || id>=p.experts || selected[id]) {slot.error=strata_ds4::cpu_bad_request;return;}
            selected[id]=true;
        }
        for(int k=0;k<p.cache_count;++k) {
            const int victim=p.cache[k].victim;
            if(victim < -1 || victim>=p.experts) {slot.error=strata_ds4::cpu_bad_request;return;}
            if(victim>=0 && selected[victim])++cache_current_protected;
        }
        for(int i=0;i<job.count;++i) {
            const int id=job.ids[i];
            if(id<0 || id>=p.experts || !selected[id] || used[id] || p.cached[id] || (p.compact && i>=p.compact)) {
                slot.error=strata_ds4::cpu_bad_request;return;
            }
            used[id]=true;
            const auto offset=p.compact?size_t(i)*(p.width+512):size_t(id)*p.width;
            const int loan=job.cache_slots[i];void *expected=(uint8_t *)p.scratch+offset;
            if(loan>=0) {
                const size_t n=p.width+(id<p.experts-1?std::min<size_t>(p.width,512):0);
                if(loan>=p.cache_count || claimed[loan] || strata_ds4::gpu_fetch_keep_ram(p,job,id) || p.cache[loan].bytes!=n || p.scores[id]<p.cache[loan].score) {
                    slot.error=strata_ds4::cpu_bad_request;return;
                }
                if(p.cache[loan].victim>=0 && selected[p.cache[loan].victim]) {slot.error=strata_ds4::cpu_bad_request;return;}
                claimed[loan]=true;expected=p.cache[loan].address;
            } else if(loan!=-1) {slot.error=strata_ds4::cpu_bad_request;return;}
            if(job.destinations[i]!=expected) {slot.error=strata_ds4::cpu_bad_request;return;}
        }
        if(release_ram) {
            const auto start=std::chrono::steady_clock::now();
            if(!m && release_cpu_candidates)release_cpu_candidates(job.cpu_selected);
            for(int id=0;id<p.experts;++id)if(!used[id])release_lease(m,id,false);
            release_us+=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-start).count();
        }
        strata_expert_file::Request local_reader;
        for(int i=0;i<job.count;++i) {
            const int id=job.ids[i];const size_t n=p.width+(id<p.experts-1?std::min<size_t>(p.width,512):0);
            if(service.cancelled.load()) {slot.error=strata_ds4::cpu_cancelled;return;}
            if(leases[m][id]) {
                service.upload(job.destinations[i],leases[m][id].data(),n);ram_bytes+=n;
            } else {
                // Actual GPU demand can take a spare or an unleased victim of
                // the same size, but never allocate/free its pinned payload.
                const auto offset=offsets[m]+size_t(id)*p.width;
                void *retained=nullptr;
                if(reserve_limit && n<=reserve_limit-pending_bytes && ram->recycle_fill(*fills[m][i],files[m],offset,n,recycle_window.get())) {
                    pending_bytes+=n;++reserved;reserved_bytes+=n;retained=fills[m][i]->data();
                }
                service.read_upload(job.destinations[i],n,[&](size_t off,void *dest,size_t bytes) {
                    const auto start=std::chrono::steady_clock::now();
                    const bool ok=reader.read(local_reader,*files[m],offset+off,
                        weights[m]+size_t(id)*p.width+off,dest,bytes,service.cancelled);
                    read_us+=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-start).count();return ok;
                },retained);source_bytes+=n;
                filled[m][i]=retained!=nullptr;
            }
            h2d_bytes+=n;++matrices;
        }
        if(release_ram && job.count) {
            service.finish_uploads(); // includes the last upload, not just earlier ones
            const auto start=std::chrono::steady_clock::now();
            for(int i=0;i<job.count;++i)release_lease(m,job.ids[i],true);
            release_us+=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-start).count();
        }
    }
    void release_lease(int m,int id,bool used) {
        auto &lease=leases[m][id];if(!lease)return;
        if(const auto bytes=ram->release_cached(lease)) {
            if(used)++released_used;else ++released_unused;released_bytes+=bytes;
        } else ++release_deferred;
    }
    template<class Retire> void finish(cudaStream_t owner,bool complete_branch,Retire retire) {
        if(!active)return;
        auto start=std::chrono::steady_clock::now();
        // Partial branches can leave early requests waiting for consumers
        // that will never run. Drain producers, then stop those requests.
        bool partial=false;for(int m=0;m<3;++m)partial|=published[m] && !submitted[m];
        if(partial) {
            strata_ds4::cpu_queue_check(cudaStreamSynchronize(owner));queue->cancelled.store(true);
            for(int m=0;m<3;++m)if(published[m] && !submitted[m])queue->abandon(m);
        }
        const auto retirement=queue->retire(owner,early_retire && complete_branch,!async_park || !complete_branch);
        if(partial)for(int m=0;m<3;++m)if(published[m] && !submitted[m]) {
            auto &slot=queue->host[m];
            slot.chunk_done.store(slot.chunk.load(cuda::memory_order_acquire),cuda::memory_order_release);
            slot.consumed.store(slot.request.load(cuda::memory_order_acquire),cuda::memory_order_release);
        }
        if(retirement.early)++early_retire_branches;else if(early_retire)++early_retire_fallbacks;
        retire_tail_pending+=retirement.tail_pending;
        owner_wait_us+=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-start).count();
        start=std::chrono::steady_clock::now();
        for(int m=0;m<3;++m)if(submitted[m]) {
            const auto &slot=queue->host[m];
            const auto failure=slot.failure.load(cuda::memory_order_acquire);
            if(failure || slot.job.epoch!=epoch ||
               slot.consumed.load(cuda::memory_order_acquire)!=slot.request.load(cuda::memory_order_acquire))
                GGML_ABORT("GPU fetch queue failed: matrix=%d epoch=%llu status=%u",m,(unsigned long long)epoch,failure);
        }
        for(int m=0;m<3;++m)if(submitted[m]) {
            const auto &job=queue->host[m].job;const auto &p=plan[m];
            if(reserve_limit && ram)for(int i=0;i<job.count;++i) {
                const int id=job.ids[i];const auto n=p.width+(id<p.experts-1?std::min<size_t>(p.width,512):0);
                const auto offset=offsets[m]+size_t(id)*p.width;
                if(ram_hit[m][id]) {
                    if(release_ram)ram->observe_ram(files[m],offset,n);else ram->acquire(files[m],offset,n);
                } else ram->observe_file(files[m],offset,n);
            }
            retire(m,job);
        }
        for(int m=0;m<3;++m)for(int i=0;i<64;++i)if(fills[m][i] && fills[m][i]->data()) {
            const int id=queue->host[m].job.ids[i];
            if(filled[m][i]) {++completed;completed_bytes+=plan[m].width+(id<plan[m].experts-1?std::min<size_t>(plan[m].width,512):0);}
            else ++unused;
            fills[m][i]->complete(filled[m][i]);filled[m][i]=false;
        }
        active=false;
        recycle_window.reset();
        for(auto &matrix:leases)for(auto &lease:matrix)lease={};
        for(auto &matrix:epoch_leases)for(auto &lease:matrix)lease={};
        for(auto &file:files)file.reset();
        const auto elapsed=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-start).count();
        retire_us+=elapsed;
        // Upper bound on work available for overlap, not measured GPU time:
        // the tail may finish at any point during this host bookkeeping.
        if(retirement.tail_pending)retire_overlap_us+=elapsed;
    }
    ~StrataGpuFetchLane() {
        if(enabled)GGML_LOG_INFO("STRATA_GPU_FETCH_DEMAND_CACHE enabled=%d plans=%llu misses=%llu scanned=%llu\n",
            int(cache_demand),(unsigned long long)cache_demand_plans,
            (unsigned long long)cache_demand_misses,(unsigned long long)cache_demand_scanned);
        if(enabled)GGML_LOG_INFO("STRATA_GPU_FETCH_CURRENT_VICTIMS enabled=%d offered=%llu admitted=%llu protected=%llu\n",
            int(cache_current_victims),(unsigned long long)cache_current_offered,
            (unsigned long long)cache_current_admitted,(unsigned long long)cache_current_protected);
        if(queue)GGML_LOG_INFO("STRATA_GPU_FETCH_READER mode=%s file_bytes=%llu mmap_bytes=%llu resident_checks=%llu resident_misses=%llu\n",
            reader.name(),(unsigned long long)reader.file_bytes,(unsigned long long)reader.mmap_bytes,
            (unsigned long long)reader.resident_checks,(unsigned long long)reader.resident_misses);
        if(queue)GGML_LOG_INFO("STRATA_GPU_FETCH_TRANSPORT dma=%d early_sources=%d early_reads=%llu early_read_bytes=%llu route_wait_us=%llu dma_wait_us=%llu depth=%d copies=%llu peak_inflight=%llu stage_reuse_waits=%llu stage_slots=%d\n",
            int(queue->owner_dma),int(early_sources),(unsigned long long)queue->early_reads,
            (unsigned long long)queue->early_read_bytes,(unsigned long long)queue->route_wait_us,(unsigned long long)queue->dma_wait_us,
            queue->dma_depth,(unsigned long long)queue->dma_copies,(unsigned long long)queue->dma_peak_inflight,(unsigned long long)queue->stage_reuse_waits,queue->dma_stage_slots);
        if(queue)GGML_LOG_INFO("STRATA_GPU_FETCH_PARK enabled=%d async_disarms=%llu waits=%llu wait_us=%llu\n",
            int(async_park),(unsigned long long)queue->async_disarms,(unsigned long long)queue->park_waits,(unsigned long long)queue->park_us);
        if(enabled)GGML_LOG_INFO("STRATA_GPU_FETCH_RETIRE enabled=%d early=%llu fallback=%llu tail_pending=%llu candidate_overlap_us=%llu\n",
            int(early_retire),(unsigned long long)early_retire_branches,(unsigned long long)early_retire_fallbacks,
            (unsigned long long)retire_tail_pending,(unsigned long long)retire_overlap_us);
        if(enabled)GGML_LOG_INFO("STRATA_GPU_FETCH_RAM_RELEASE enabled=%d unused=%llu used=%llu lease_bytes=%llu deferred=%llu release_us=%llu\n",
            int(release_ram),(unsigned long long)released_unused,(unsigned long long)released_used,
            (unsigned long long)released_bytes,(unsigned long long)release_deferred,(unsigned long long)release_us);
        if(enabled)GGML_LOG_INFO("STRATA_GPU_FETCH_PLACEMENT keep_ram=%d ram_bypasses=%llu ready=%llu partial=%llu allgpu=%llu allmiss=%llu cpu_only=%llu\n",
            int(cache_keep_ram),(unsigned long long)cache_ram_bypasses,(unsigned long long)route_ready,(unsigned long long)route_partial,
            (unsigned long long)route_allgpu,(unsigned long long)route_allmiss,(unsigned long long)route_cpu_only);
        if(enabled && cache_slots)GGML_LOG_INFO("STRATA_GPU_FETCH_CACHE slots=%d offered=%llu admitted=%llu restored=%llu bytes=%llu prepare_us=%llu retire_us=%llu duplicates=%llu\n",
            cache_slots,(unsigned long long)cache_offered,(unsigned long long)cache_admitted,(unsigned long long)cache_restored,
            (unsigned long long)cache_bytes,(unsigned long long)cache_prepare_us,(unsigned long long)cache_retire_us,(unsigned long long)cache_duplicates);
        if(enabled)GGML_LOG_INFO("STRATA_GPU_FETCH_QUEUE branches=%llu matrices=%llu ram_bytes=%llu source_bytes=%llu h2d_bytes=%llu stage_bytes=%zu copy_blocks=%d ram_reserve_bytes=%zu reservations=%llu completed=%llu unused=%llu reserved_bytes=%llu completed_bytes=%llu prepare_us=%llu retire_us=%llu owner_wait_us=%llu read_us=%llu transport=%s early_sources=%d\n",
            (unsigned long long)branches,(unsigned long long)matrices,(unsigned long long)ram_bytes,
            (unsigned long long)source_bytes,(unsigned long long)h2d_bytes,queue?queue->staging_bytes():0,queue?queue->copy_blocks:0,
            reserve_limit,(unsigned long long)reserved,(unsigned long long)completed,(unsigned long long)unused,
            (unsigned long long)reserved_bytes,(unsigned long long)completed_bytes,(unsigned long long)prepare_us,(unsigned long long)retire_us,
            (unsigned long long)owner_wait_us,(unsigned long long)read_us,queue && queue->owner_dma?"owner_dma":"mapped",int(early_sources));
    }
};
