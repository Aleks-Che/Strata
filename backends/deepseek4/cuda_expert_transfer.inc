// Included only in the private CUDA translation unit of strata-deepseek4.
#include "expert_transfer.h"
#include <list>
#include <map>
#include <memory>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <limits>
#include <unordered_set>
#include "device_memory.inc"
#include "expert_pipeline.hpp"
#include "expert_reuse.hpp"
#include "expert_frequency.hpp"
#include "host_copy.hpp"
#include "expert_tuning.hpp"
#include "expert_slab.hpp"
#include "expert_copy_batch.cuh"
#include "expert_direct_cache.cuh"
#include "expert_hash.cuh"
#include "expert_early_hits.cuh"
#include "host_cache.hpp"
#include "expert_residency.cuh"
#include "cpu_lane.cuh"
#include "cpu_expert_lane.cuh"
#include "gpu_fetch_lane.cuh"
#ifdef STRATA_DS4_COST_EXPERIMENT
#include "expert_transport_profile.cuh"
#endif

// Weak ownership avoids keeping CUDA allocations alive after the last backend.
// Registry access is covered by strata_expert_transfer_mutex below.
static std::map<int,std::weak_ptr<strata_ds4::ExpertSlabAllocator>> strata_expert_slabs;

static std::weak_ptr<strata_ds4::HostCache> strata_host_cache;
static thread_local int strata_expert_thread_budget=-1;
static thread_local int strata_expert_thread_decay=0;
static thread_local bool strata_cpu_decode=true;
static void strata_cuda_cpu_expert_phase(bool decode){strata_cpu_decode=decode;}
static void strata_cuda_expert_budget(int mib) {
    GGML_ASSERT(mib>=0 && mib<=65536);
    strata_expert_thread_budget=mib;
}
static void strata_cuda_expert_decay(int accesses) {
    GGML_ASSERT(accesses>=0 && accesses<=strata_ds4::max_frequency_decay);
    strata_expert_thread_decay=accesses;
}
static uint64_t strata_expert_touch_clock(int device);

struct StrataExpertTransfer {
    struct Entry {
        size_t offset, bytes; std::list<const void *>::iterator lru;
        uint8_t *allocation=nullptr;
        uint64_t touched=0;
        std::shared_ptr<StrataExpertResidency::Upload> residency;
        bool empty=false; // charged allocation reserved by the GPU-demand path
    };
    int device;
    ggml_backend_cuda_context *context;
    cudaStream_t stream;
    uint8_t *arena=nullptr, *stage[2]={nullptr,nullptr};
    cudaEvent_t ready[2]={nullptr,nullptr};
    bool pending[2]={false,false};
    size_t capacity=0, stage_size=0, turn=0;
    std::map<size_t,size_t> free_blocks;
    std::list<const void *> lru;
    using Entries=std::unordered_map<const void *,Entry>;
    Entries entries;
    // Extracted nodes retain their LRU positions and allocation budget. Their
    // keys are protected, but absent from lookup until the branch is drained.
    struct FetchLoan { Entries::node_type node;const void *replacement=nullptr; };

    StrataExpertCounters stats;
    // Tensor properties do not include the optional mask/scatter/join kernels.
    // Track their captured addresses separately for each GGML split graph.
    using CpuGraphSignature=std::vector<std::array<uintptr_t,6>>;
    std::map<const void *,CpuGraphSignature> cpu_graph_signatures;
    bool managed=false;
    size_t allocated=0;
    std::shared_ptr<strata_ds4::ExpertSlabAllocator> slab;
    std::unique_ptr<StrataExpertPipeline> pipeline;

    StrataExpertReuseFence reuse_fence;
    std::unique_ptr<StrataExpertFrequency> frequency;
    bool match_size=false;
    bool prefill_cache_hits=false;
    StrataExpertCopyBatch copy_batch;
    StrataExpertDirect direct;
    StrataHashScratch hash_scratch;
    StrataExpertEarlyHitScratch early_hits;
    StrataCpuLane cpu_lane;
    struct BranchContext {
        StrataCpuExpertLane cpu;
        StrataGpuFetchLane fetch;
        FetchLoan loans[3][strata_ds4::gpu_fetch_cache_slots];
        size_t loan_count=0;
        std::unordered_set<const void *> protected_keys,deferred_keys;
    };
    std::array<BranchContext,2> branch_contexts;
    size_t current_context=0;
    uint64_t context_rotations=0,context_reuse_waits=0;
    uint64_t demotion_clock=0;
    auto &cpu_whole(){return branch_contexts[current_context].cpu;}
    auto &gpu_fetch(){return branch_contexts[current_context].fetch;}
    auto &fetch_loans(){return branch_contexts[current_context].loans;}
    auto &fetch_loan_count(){return branch_contexts[current_context].loan_count;}
    auto &protected_keys(){return branch_contexts[current_context].protected_keys;}
    auto &deferred_keys(){return branch_contexts[current_context].deferred_keys;}
    bool protected_in_flight(const void *key)const {
        for(const auto &c:branch_contexts)if(c.protected_keys.count(key) || c.deferred_keys.count(key))return true;
        return false;
    }
    std::shared_ptr<strata_ds4::HostCache> host_cache;
    StrataExpertResidency residency;
    bool ram_drop_gpu=false,ram_on_bypass=false;
    void promoted(const void *key,Entry &entry,bool already_copied=false) {
        if(!residency.cache && !ram_drop_gpu)return;
        if(auto file=strata_expert_file::find(key,entry.bytes)) {
            const auto offset=file->offset_of(key,entry.bytes);
            if(residency.cache)entry.residency=residency.uploaded(file,offset,entry.bytes,ram_drop_gpu,stream,already_copied);
            else host_cache->discard(file,offset,entry.bytes);
        }
    }
    std::vector<StrataExpertSlice> early_slices;
    void retire_direct() {
        if(!direct.tensor)return;
        ggml_cuda_set_device(device);
        early_hits.clear(stream);
        // An abandoned prepared tensor may still be consumed by a caller after
        // changing the cache policy. Materialize it before releasing its leases.
        if(!direct.consumed) {
            for(auto &h:direct.held) {
                copy_cached(h.destination,h.source,h.bytes);
                stats.d2d_bytes+=h.bytes;stats.direct_materialized_bytes+=h.bytes;
            }
            flush_copies();direct.fallback_table();direct.publish(stream);++stats.direct_tables;
        }
        direct.clear();
    }
    bool read_direct(const ggml_tensor *dest,int id,const void *key,const void *source,size_t bytes,bool was_hit) {
        if((!was_hit && !direct.admissions) || !direct.bind(dest,id,key,source,bytes))return false;
        if(was_hit)++stats.direct_hits;else ++stats.direct_admissions;
        stats.direct_bytes+=bytes;return true;
    }
    void flush_copies() {
        if(!copy_batch.count)return;
        ggml_cuda_set_device(device);
#ifdef STRATA_DS4_COST_EXPERIMENT
        const auto event=transport_profile.begin(stream,"cache_batch_d2d",copy_batch.bytes());
#endif
        copy_batch.flush(stream);
#ifdef STRATA_DS4_COST_EXPERIMENT
        transport_profile.end(event,stream);
#endif
    }
    void copy_cached(void *dst,const void *src,size_t bytes) {
        if(copy_batch.append(dst,src,bytes))return;
        flush_copies();
        if(!copy_batch.append(dst,src,bytes))copy_device(dst,src,bytes,cudaMemcpyDeviceToDevice);
    }
#ifdef STRATA_DS4_COST_EXPERIMENT
    StrataExpertTransportProfile transport_profile;
    StrataExpertPipeline::Counters previous_profile_counters;
    void profile_drain() {
        if(!transport_profile.enabled())return;
        transport_profile.drain();
        if(pipeline) {
            const auto now=pipeline->counters();
            transport_profile.host_delta("host_read",now.read_us-previous_profile_counters.read_us);
            transport_profile.host_delta("host_consumer_wait",now.consumer_wait_us-previous_profile_counters.consumer_wait_us);
            transport_profile.host_delta("host_slot_wait",now.slot_wait_us-previous_profile_counters.slot_wait_us);
            previous_profile_counters=now;
        }
    }
#endif
    StrataExpertPipeline::CopyObserver h2d_observer,delivery_observer;
    void copy_device(void *dst,const void *src,size_t bytes,cudaMemcpyKind kind) {
#ifdef STRATA_DS4_COST_EXPERIMENT
        const auto event=transport_profile.begin(stream,kind==cudaMemcpyDeviceToDevice?"cache_d2d":"fallback_h2d",bytes);
#endif
        CUDA_CHECK(cudaMemcpyAsync(dst,src,bytes,kind,stream));
#ifdef STRATA_DS4_COST_EXPERIMENT
        transport_profile.end(event,stream);
#endif
    }

    bool prefill_hit(void *dest,const void *key,size_t bytes) {
        if(!prefill_cache_hits)return false;
        auto found=entries.find(key);
        if(found==entries.end())return false;
        GGML_ASSERT(found->second.bytes==bytes);
        auto *source=managed?found->second.allocation:arena+found->second.offset;
        copy_device(dest,source,bytes,cudaMemcpyDeviceToDevice);
        ++stats.hits;++stats.prefill_hits;stats.d2d_bytes+=bytes;
        // Read-only prefill: no admission or frequency/LRU training.
        return true;
    }

    bool admits(const void *key,const StrataExpertTransfer &owner,const void *victim) {
        if(!frequency || !owner.frequency || frequency->score(key)>=owner.frequency->score(victim))return true;
        ++stats.admission_rejects;return false;
    }

    Entries::iterator reserve_fetch_slot(size_t bytes);
#include "gpu_fetch_cache.inc"

    void finish_context(bool keep_whole=false,const std::function<void()> &prepare_next={},bool overlap_context=false) {
        // A handoff receipt precedes the current fetch on this owner stream.
        // Reuse its existing copy fence, not an extra wait at the branch entry.
        const bool receipt_at_fetch=cpu_whole().handoff_pending && gpu_fetch().active;
        if(!receipt_at_fetch)cpu_whole().finish_consumer();
        deferred_keys().clear();
        // Only a complete GPU-fetch branch proves that all route producers
        // have finished at the copy fence. Cancellation/legacy/demotion retain
        // the ordinary boundary. The callback owns a separate host snapshot.
        const bool early_context=!keep_whole && cpu_whole().gpu_queue_active && cpu_whole().completion &&
            gpu_fetch().active && gpu_fetch().early_retire && gpu_fetch().queue->has_copy_fence() &&
            !(host_cache && host_cache->demotion_enabled());
        const bool can_prepare=prepare_next && early_context;
        const bool defer_consumer=overlap_context && cpu_whole().context_overlap && early_context;
        // Optional copy fence permits RAM/cache bookkeeping during the final
        // GPU/CPU compute. Deferred retirement retains VRAM keys and the mapped
        // result through finish_consumer(); otherwise clear() waits here.
        gpu_fetch().finish(stream,!keep_whole && cpu_whole().completion,[&](int m,const auto &job) {
            const auto &p=gpu_fetch().plan[m];
            if(!m)for(int i=0;i<job.used_count;++i) {
                const int id=job.used_ids[i];int hits=0;
                for(int j=0;j<3;++j)hits+=gpu_fetch().plan[j].cached[id]!=nullptr;
                if(!hits)++gpu_fetch().route_allmiss;else if(hits==3)++gpu_fetch().route_allgpu;else ++gpu_fetch().route_partial;
                const auto &e=cpu_whole().queue_table.expert[id];
                gpu_fetch().route_ready+=e.ready && e.eligible;gpu_fetch().route_cpu_only+=job.cpu_only[i];
            }
            if(!gpu_fetch().reserve_limit && !gpu_fetch().cache_slots)return; // original untrained mode
            bool admitted[256]={};
            for(int i=0;i<job.count;++i)if(p.cache_count && p.keep_ram[job.ids[i]])++gpu_fetch().cache_ram_bypasses;
            for(int i=0;i<job.count;++i)if(job.cache_slots[i]>=0) {
                const int id=job.ids[i],k=job.cache_slots[i];
                GGML_ASSERT(k<p.cache_count && !fetch_loans()[m][k].node.empty() && !fetch_loans()[m][k].replacement);
                fetch_loans()[m][k].replacement=gpu_fetch().weights[m]+size_t(id)*p.width;admitted[id]=true;
            }
            for(int i=0;i<job.used_count;++i) {
                const int id=job.used_ids[i];const void *key=gpu_fetch().weights[m]+size_t(id)*p.width;
                const auto bytes=p.width+(id<p.experts-1?std::min<size_t>(p.width,512):0);
                if(frequency)frequency->record(key);
                if(job.cpu_only[i]) {cpu_whole().saved_bytes+=bytes;continue;}
                if(p.cached[id]) {
                    if(host_cache && host_cache->demotion_enabled())host_cache->observe_gpu(gpu_fetch().files[m],gpu_fetch().offsets[m]+size_t(id)*p.width,bytes);
                    const auto it=entries.find(key);GGML_ASSERT(it!=entries.end());
                    lru.splice(lru.begin(),lru,it->second.lru);
                    if(managed)it->second.touched=strata_expert_touch_clock(device);
                    ++stats.hits;++stats.direct_hits;stats.direct_bytes+=bytes;
                } else {++stats.misses;if(!admitted[id])++stats.bypass;}
            }
        });
        if(gpu_fetch().demotion_epoch){residency.retire(gpu_fetch().demotion_epoch);gpu_fetch().demotion_epoch=0;}
        if(receipt_at_fetch)cpu_whole().finish_consumer(true);
        finish_fetch_cache();
        if(can_prepare) {
            if(!cpu_whole().queue_adopted){uint8_t selected[256];cpu_whole().queue_adopt(selected);}
            prepare_next();
        }
        cpu_lane.clear();cpu_lane.planned.clear();
        if(!keep_whole)cpu_whole().clear(defer_consumer);
        early_slices.clear();
        retire_direct();
        flush_copies();
        if(pipeline)pipeline->finish(false);
        // Keep the previous branch protected while constructing the next
        // cache-loan plan. Its final fence is checked before next publication.
        if(cpu_whole().consumer_pending)deferred_keys().swap(protected_keys());
        else protected_keys().clear();
        if(residency.cache)residency.poll();
        if(host_cache)host_cache->collect();
    }
    // Advance only between complete GPU-fetch branches. Shared GPU scratch,
    // packet and output reuse is ordered on the owner stream; all mutable host
    // work/table/trace/leases and service slots belong to the separate context.
    void next_context(const StrataCpuExpertBranch &next) {
        auto &old=cpu_whole();auto &fetch=gpu_fetch();
        if(!old.context_overlap || !old.gpu_queue_active || !old.completion || !fetch.active)return;
        // A repeated weight set could name an admission not yet retired into
        // entries. Drain before publishing another snapshot of those keys.
        for(auto &c:branch_contexts)if(c.cpu.configured)
            for(auto *w:c.cpu.branch.weights)for(auto *n:next.weights)
                if(w->data==n->data){finish_plan();return;}
        retire_direct();flush_copies();
        const auto next_index=(current_context+1)%branch_contexts.size();
        current_context=next_index;
        if(cpu_whole().configured || gpu_fetch().active) {
            const auto state=cudaEventQuery(cpu_whole().consumed);
            if(state==cudaErrorNotReady)++context_reuse_waits;else CUDA_CHECK(state);
            finish_context();
        }
        auto &fresh=cpu_whole();fresh.ram=host_cache;
        fresh.pool=old.pool;fresh.graph_storage=old.graph_storage;fresh.costs=old.costs;
        ++context_rotations;
    }
    void finish_plan(bool keep_whole=false,const std::function<void()> &prepare_next={},bool overlap_context=false) {
        const auto current=current_context;
        // FIFO retirement makes frequency/LRU training independent of which
        // service happens to finish first. Cancellation drains every context.
        for(size_t i=1;i<branch_contexts.size();++i) {
            current_context=(current+i)%branch_contexts.size();
            if(cpu_whole().configured || gpu_fetch().active)finish_context();
        }
        current_context=current;finish_context(keep_whole,prepare_next,overlap_context);
    }
    StrataExpertCounters counters() {
        auto result=stats;
        result.cpu_lane_ops=cpu_lane.operations;result.cpu_lane_experts=cpu_lane.experts;
        result.cpu_lane_saved_bytes=cpu_lane.saved_bytes;
        const auto current=current_context;
        for(current_context=0;current_context<branch_contexts.size();++current_context) {
        result.cpu_lane_ops+=cpu_whole().operations;result.cpu_lane_experts+=cpu_whole().experts;
        result.cpu_lane_saved_bytes+=cpu_whole().saved_bytes;
        result.cpu_queue_jobs+=cpu_whole().queue_jobs;result.cpu_queue_rows+=cpu_whole().queue_rows;
        result.gpu_fetch_branches+=gpu_fetch().branches;result.gpu_fetch_matrices+=gpu_fetch().matrices;
        result.gpu_fetch_ram_reservations+=gpu_fetch().reserved;result.gpu_fetch_ram_completed+=gpu_fetch().completed;
        result.gpu_fetch_ram_bytes+=gpu_fetch().ram_bytes;
        result.gpu_fetch_cache_offered+=gpu_fetch().cache_offered;result.gpu_fetch_cache_admitted+=gpu_fetch().cache_admitted;
        result.gpu_fetch_cache_restored+=gpu_fetch().cache_restored;result.gpu_fetch_cache_bytes+=gpu_fetch().cache_bytes;
        result.gpu_fetch_cache_duplicates+=gpu_fetch().cache_duplicates;
        result.gpu_fetch_cache_ram_bypasses+=gpu_fetch().cache_ram_bypasses;
        result.gpu_fetch_ram_released_unused+=gpu_fetch().released_unused;result.gpu_fetch_ram_released_used+=gpu_fetch().released_used;
        result.cpu_snapshot_released+=cpu_whole().snapshot_released;
        result.gpu_fetch_early_retire+=gpu_fetch().early_retire_branches;result.gpu_fetch_retire_tail_pending+=gpu_fetch().retire_tail_pending;
        result.cpu_weight_release_gate_up+=cpu_whole().weight_release_gate_up.load(std::memory_order_relaxed);
        result.cpu_weight_release_down+=cpu_whole().weight_release_down.load(std::memory_order_relaxed);
        result.cpu_weight_release_deferred+=cpu_whole().weight_release_deferred.load(std::memory_order_relaxed);
        result.cpu_weight_release_jobs+=cpu_whole().weight_release_jobs.load(std::memory_order_relaxed);
        result.cpu_weight_capture_jobs+=cpu_whole().weight_capture_jobs.load(std::memory_order_relaxed);
        result.cpu_snapshot_prepared+=cpu_whole().snapshot_prepared;
        result.cpu_snapshot_tail_pending+=cpu_whole().snapshot_tail_pending;
        result.cpu_snapshot_fallbacks+=cpu_whole().snapshot_fallbacks;
        result.cpu_context_deferred+=cpu_whole().context_deferred;
        result.cpu_context_tail_pending+=cpu_whole().context_tail_pending;
        result.cpu_consumer_handoffs+=cpu_whole().handoffs;
        result.cpu_consumer_handoff_fenced+=cpu_whole().handoff_fenced;
        result.cpu_consumer_handoff_waits+=cpu_whole().handoff_waits;
        result.cpu_consumer_handoff_ram_fallbacks+=cpu_whole().handoff_ram_fallbacks;
        result.h2d_bytes+=gpu_fetch().h2d_bytes;
        result.file_read_bytes+=gpu_fetch().file_bytes;
        }
        current_context=current;
        result.cpu_context_rotations=context_rotations;result.cpu_context_reuse_waits=context_reuse_waits;
        result.gpu_demotion_bytes=residency.d2h_bytes;
        if(pipeline) {
            auto p=pipeline->counters();
            result.pipeline_groups=p.groups;result.pipeline_chunks=p.chunks;result.pipeline_unused=p.unused;
            result.h2d_bytes+=p.h2d_bytes;result.d2d_bytes+=p.d2d_bytes;result.stage_bytes+=p.h2d_bytes-p.borrowed_bytes;
            result.stage_us+=p.read_us+p.retain_us;result.wait_us+=p.wait_us;result.submit_us+=p.submit_us;
            result.file_read_bytes+=p.file_bytes;result.mmap_read_bytes=p.mmap_bytes;result.read_peak=p.read_peak;
        }
        return result;
    }
    void plan(const StrataExpertSlice *slices,size_t count) {
        if(!early_slices.empty()) {
            bool same=early_slices.size()==count;
            for(size_t i=0;same && i<count;++i)
                same=early_slices[i].data==slices[i].data && early_slices[i].bytes==slices[i].bytes &&
                    early_slices[i].cacheable==slices[i].cacheable && early_slices[i].host_compute==slices[i].host_compute;
            if(same) {early_slices.clear();++stats.early_reuses;return;}
            ++stats.early_mismatches;
        }
        finish_plan(cpu_whole().pending);cpu_whole().pending=false;
        if(!pipeline)return;
        if(host_cache)host_cache->refresh();
        std::vector<StrataExpertSlice> misses;
        for(size_t i=0;i<count;++i) {
            if(slices[i].host_compute && cpu_lane.threads) {cpu_lane.planned.insert(slices[i].data);continue;}
            if((slices[i].cacheable || prefill_cache_hits) && entries.find(slices[i].data)!=entries.end())
                protected_keys().insert(slices[i].data);
            else misses.push_back(slices[i]);
        }
        pipeline->start(misses);
    }
    auto evictable(size_t prefer_bytes=0) {
        auto oldest=lru.rend();
        size_t scanned=0;
        for(auto it=lru.rbegin();it!=lru.rend();++it) {
            if(direct.protects(*it))continue;
            if(protected_in_flight(*it))continue;
            if(oldest==lru.rend())oldest=it;
            if(!prefer_bytes || entries.find(*it)->second.bytes==prefer_bytes)return it;
            // Keep selection near the LRU tail. Prefer an existing allocation
            // of the right size over a synchronous free/malloc replacement.
            if(++scanned==64)break;
        }
        return oldest;
    }

    static size_t env_mib(const char *name) {
        const char *s=std::getenv(name);
        if (!s) return 0;
        char *end=nullptr; long long value=std::strtoll(s,&end,10);
        GGML_ASSERT(end != s && *end == 0 && value >= 0 && value <= 65536);
        return size_t(value)<<20;
    }
    explicit StrataExpertTransfer(ggml_backend_cuda_context *ctx,bool dynamic=false):device(ctx->device),context(ctx),stream(ctx->stream()),managed(dynamic) {
        ggml_cuda_set_device(device);
        const auto slab_bytes=env_mib("STRATA_EXPERT_SLAB_MIB");
        if(slab_bytes) {
            slab=strata_expert_slabs[device].lock();
            if(!slab) {
                slab=std::make_shared<strata_ds4::ExpertSlabAllocator>(slab_bytes);
                strata_expert_slabs[device]=slab;
            }
        }
        capacity=strata_expert_thread_budget<0?env_mib("STRATA_EXPERT_CACHE_MIB"):size_t(strata_expert_thread_budget)<<20;
        const char *policy=std::getenv("STRATA_EXPERT_CACHE_POLICY");
        const char *match=std::getenv("STRATA_EXPERT_CACHE_MATCH_SIZE");
        match_size=match && std::strcmp(match,"1")==0;
        if(policy && std::strcmp(policy,"frequency")==0) {
            stats.frequency_decay=strata_ds4::frequency_decay(capacity,strata_expert_thread_decay);
            frequency=std::make_unique<StrataExpertFrequency>(stats.frequency_decay);
        }
        stage_size=env_mib("STRATA_EXPERT_STAGE_MIB");
        const char *wc=std::getenv("STRATA_EXPERT_STAGE_WC");
        bool write_combined=wc && std::strcmp(wc,"1")==0;
        const char *early=std::getenv("STRATA_EXPERT_EARLY_REFILL");
        const char *host=std::getenv("STRATA_EXPERT_HOST_COPY");
        const char *prefill=std::getenv("STRATA_EXPERT_PREFILL_CACHE_HITS");
        const bool early_refill=early && std::strcmp(early,"1")==0;
        const bool avx2=host && std::strcmp(host,"avx2")==0;
        prefill_cache_hits=prefill && std::strcmp(prefill,"1")==0;
        if(avx2 && !strata_ds4_host_copy::available())throw std::runtime_error("AVX2 host copy requires AVX2 CPU/OS support");
        if(capacity && !managed) {
            CUDA_CHECK(cudaMalloc((void **)&arena,capacity));
            free_blocks.emplace(0,capacity);
        }
        if(stage_size) for(int i=0;i<2;++i) {
            CUDA_CHECK(cudaHostAlloc((void **)&stage[i],stage_size,write_combined?cudaHostAllocWriteCombined:cudaHostAllocDefault));
            CUDA_CHECK(cudaEventCreateWithFlags(&ready[i],cudaEventDisableTiming|cudaEventBlockingSync));
        }
        const char *pipe=std::getenv("STRATA_EXPERT_PIPELINE");
        const char *readers=std::getenv("STRATA_EXPERT_READERS"),*mode=std::getenv("STRATA_EXPERT_READ_MODE");
        int n_readers=readers?std::atoi(readers):2;
        int read_mode=mode && std::strcmp(mode,"file")==0?1:mode && std::strcmp(mode,"auto")==0?2:0;
        try {
#ifdef STRATA_DS4_COST_EXPERIMENT
            if(transport_profile.enabled()) {
                h2d_observer=[this](cudaStream_t s,bool start,size_t n){transport_profile.observe(s,start,n,true);};
                delivery_observer=[this](cudaStream_t s,bool start,size_t n){transport_profile.observe(s,start,n,false);};
            }
#endif
            StrataExpertPipeline::SourceCopy source_copy;
            StrataExpertPipeline::SourceBorrow source_borrow;
            StrataExpertPipeline::SourceRetain source_retain;
            const auto enabled=[](const char *name){const char *v=std::getenv(name);return v && std::strcmp(v,"1")==0;};
            const bool ram_direct=enabled("STRATA_DS4_RAM_DIRECT_H2D");
            const bool ram_async=enabled("STRATA_DS4_RAM_ASYNC_RETAIN");
            size_t ram_pool=env_mib("STRATA_DS4_RAM_POOL_MIB");
            const size_t ram_demote=env_mib("STRATA_DS4_RAM_DEMOTE_MIB");
            ram_on_bypass=enabled("STRATA_DS4_RAM_ON_BYPASS");
            ram_drop_gpu=ram_on_bypass || enabled("STRATA_DS4_RAM_DROP_GPU");
            const bool ram_residency=ram_drop_gpu || ram_demote || enabled("STRATA_DS4_RAM_RESIDENCY");
            if(ram_async && !ram_on_bypass)throw std::runtime_error("RAM async retention requires STRATA_DS4_RAM_ON_BYPASS=1");
            if(const size_t budget=env_mib("STRATA_DS4_RAM_CACHE_MIB")) {
                if(gpu_fetch().enabled)ram_pool=std::max(ram_pool,gpu_fetch().reserve_limit);
                if(!pipe || std::strcmp(pipe,"1"))throw std::runtime_error("DeepSeek RAM cache requires pipeline");
                host_cache=strata_host_cache.lock();
                if(!host_cache) {
                    strata_ds4::HostCache::Storage storage;
                    if(cpu_whole().shared_ram || ram_direct || ram_demote) {
                        storage.allocate=[dev=device](size_t n)->uint8_t * {
                            ggml_cuda_set_device(dev);void *p=nullptr;
                            if(cudaMallocHost(&p,n)!=cudaSuccess){cudaGetLastError();return nullptr;}
                            return (uint8_t *)p;
                        };
                        storage.release=[](uint8_t *p){CUDA_CHECK(cudaFreeHost(p));};
                    }
                    host_cache=std::make_shared<strata_ds4::HostCache>(budget,strata_ds4::HostCache::memory,
                        strata_ds4::HostCache::Policy::Frequency,131072,131072,std::move(storage),ram_pool,ram_demote);strata_host_cache=host_cache;
                }
                cpu_whole().ram=host_cache;
                if(ram_residency)residency.cache=host_cache;
                if(ram_direct)source_borrow=[cache=host_cache](const auto &file,const void *src,size_t n,bool cacheable) {
                    return cache->acquire(file,file->offset_of(src,n),n,!cacheable).shared_data();
                };
                if(ram_on_bypass)source_retain=[cache=host_cache](const auto &file,const void *src,const void *payload,
                        size_t n,const std::atomic<bool> &cancel) {
                    cache->retain(file,file->offset_of(src,n),payload,n,cancel);
                };
                source_copy=[cache=host_cache,defer=ram_on_bypass](const auto &file,const void *src,void *dst,size_t n,
                        const std::atomic<bool> &cancel,strata_expert_file::Request &request,bool cacheable,bool native) {
                    return cache->read(file,file->offset_of(src,n),dst,n,cancel,[&](void *p,size_t bytes) {
                        if(native)return request.read(*file,src,p,bytes,cancel);
                        if(cancel.load())return false;
                        std::memcpy(p,src,bytes);return !cancel.load();
                    },!cacheable,defer);
                };
            }
            if((ram_direct || ram_residency || ram_pool) && !host_cache)throw std::runtime_error("RAM transfer tuning requires STRATA_DS4_RAM_CACHE_MIB > 0");
            GGML_LOG_INFO("STRATA_RAM_TRANSFER direct_h2d=%d drop_gpu=%d on_bypass=%d pinned=%d async_retain=%d pool_mib=%zu\n",int(ram_direct),int(ram_drop_gpu),int(ram_on_bypass),int(bool(host_cache) && (ram_direct || cpu_whole().shared_ram || ram_demote)),int(ram_async),ram_pool>>20);
            GGML_LOG_INFO("STRATA_RAM_RESIDENCY enabled=%d demote_mib=%zu\n",int(ram_residency),ram_demote>>20);
            if(cpu_whole().shared_ram && !host_cache)throw std::runtime_error("CPU shared RAM requires STRATA_DS4_RAM_CACHE_MIB > 0");
            if(pipe && std::strcmp(pipe,"1")==0)pipeline=std::make_unique<StrataExpertPipeline>(device,stage_size,write_combined,n_readers,read_mode,
                h2d_observer,1,early_refill,
                avx2?StrataExpertPipeline::HostCopy(strata_ds4_host_copy::copy):StrataExpertPipeline::HostCopy{},
                StrataExpertPipeline::SourceRead{},source_copy,source_borrow,source_retain,ram_async?2:0);
        } catch(...) {
            // A failed ring allocation or worker creation must not leak the
            // already allocated cache and legacy staging buffers.
            for(int i=0;i<2;++i) {
                if(ready[i])cudaEventDestroy(ready[i]);
                if(stage[i])cudaFreeHost(stage[i]);
            }
            if(arena)cudaFree(arena);
            throw;
        }
        GGML_LOG_INFO("STRATA_EXPERT_TRANSFER cache_mib=%zu stage_mib=%zu buffers=2 policy=%s prefill_bypass=1 stage_wc=%d pipeline=%d pipeline_slots=%zu readers=%d read_mode=%d\n",capacity>>20,stage_size>>20,frequency?"frequency":"lru",int(write_combined),int(bool(pipeline)),pipeline?StrataExpertPipeline::slots:0,n_readers,read_mode);
        GGML_LOG_INFO("STRATA_EXPERT_TUNING host_copy=%s early_refill=%d prefill_cache_hits=%d\n",avx2?"avx2":"crt",int(early_refill),int(prefill_cache_hits));
        GGML_LOG_INFO("STRATA_EXPERT_FREQUENCY decay=%llu\n",(unsigned long long)stats.frequency_decay);
        GGML_LOG_INFO("STRATA_EXPERT_SLAB mib=%zu dynamic=%d\n",slab_bytes>>20,int(managed));
        GGML_LOG_INFO("STRATA_EXPERT_DIRECT_CONFIG enabled=%d admissions=%d compact_leases=%d\n",int(direct.enabled),int(direct.admissions),int(direct.compact_leases));
        if(cpu_lane.threads) {
            GGML_ASSERT(direct.enabled && pipeline && !early_hits.enabled);
            GGML_LOG_INFO("STRATA_CPU_LANE_CONFIG threads=%d max_experts=%d prefill=GPU mode=%s policy=%s ram_mib=%zu shared_ram=%d graphs=%s\n",
                cpu_lane.threads,cpu_lane.max_experts,cpu_whole().legacy?"matrix_legacy":"whole_expert",
                cpu_whole().force?"force":"auto",cpu_whole().cache_limit>>20,int(cpu_whole().shared_ram),cpu_whole().legacy?"GPU_only_splits":"external_event");
        }
    }
    ~StrataExpertTransfer() {
        ggml_cuda_set_device(device);
        finish_plan();
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if(residency.cache)residency.poll(true);
        if(host_cache)host_cache->collect();
#ifdef STRATA_DS4_COST_EXPERIMENT
        profile_drain();
#endif
        if(pipeline) {
            const auto p=pipeline->counters();
            GGML_LOG_INFO("STRATA_RAM_H2D chunks=%llu bytes=%llu retain_chunks=%llu retain_bytes=%llu retain_us=%llu\n",
                (unsigned long long)p.borrowed_chunks,(unsigned long long)p.borrowed_bytes,
                (unsigned long long)p.retain_chunks,(unsigned long long)p.retain_bytes,(unsigned long long)p.retain_us);
            GGML_LOG_INFO("STRATA_RAM_ASYNC queued=%llu dropped=%llu cancelled=%llu pending_peak=%llu drain_us=%llu\n",
                (unsigned long long)p.retain_queued,(unsigned long long)p.retain_dropped,(unsigned long long)p.retain_cancelled,
                (unsigned long long)p.retain_pending_peak,(unsigned long long)p.retain_wait_us);
        }
        stats=counters();pipeline.reset();
        if(cpu_lane.threads)GGML_LOG_INFO("STRATA_CPU_LANE ops=%llu experts=%llu skipped_weight_bytes=%llu wait_us=%llu input_us=%llu\n",
            (unsigned long long)stats.cpu_lane_ops,(unsigned long long)stats.cpu_lane_experts,
            (unsigned long long)stats.cpu_lane_saved_bytes,(unsigned long long)cpu_lane.wait_us,(unsigned long long)cpu_lane.input_us);
        for(size_t index=0;index<branch_contexts.size();++index) {
            const auto &c=branch_contexts[index].cpu;
        if(c.threads && !c.legacy)GGML_LOG_INFO("STRATA_CPU_WHOLE branches=%llu declined=%llu ops=%llu expert_rows=%llu compute_us=%llu pack_us=%llu select_us=%llu retire_us=%llu input_bytes=%llu output_bytes=%llu ram_hits=%llu ram_misses=%llu ram_bytes=%zu context=%zu\n",
            (unsigned long long)c.branches,(unsigned long long)c.declined,
            (unsigned long long)c.operations,(unsigned long long)c.experts,
            (unsigned long long)c.compute_us,(unsigned long long)c.pack_us,
            (unsigned long long)c.select_us,(unsigned long long)c.retire_us,
            (unsigned long long)c.input_bytes,(unsigned long long)c.output_bytes,
            (unsigned long long)c.ram_hits,(unsigned long long)c.ram_misses,c.cache_bytes,index);
        }
        if(host_cache) {
            auto h=host_cache->snapshot();
            GGML_LOG_INFO("STRATA_RAM_TIERS gpu_entries=%llu gpu_bytes=%llu gpu_pending=%llu publications=%llu duplicate_fills=%llu stale_fills=%llu gpu_bypasses=%llu demotion_pending=%llu demotion_jobs=%llu demotion_admitted=%llu demotion_published=%llu demotion_bytes=%llu d2h_bytes=%llu d2h_jobs=%llu\n",
                (unsigned long long)h.gpu_entries,(unsigned long long)h.gpu_bytes,(unsigned long long)h.gpu_pending,
                (unsigned long long)h.gpu_publications,(unsigned long long)h.duplicate_fills,(unsigned long long)h.stale_fills,(unsigned long long)h.gpu_bypasses,
                (unsigned long long)h.demotion_pending,(unsigned long long)h.demotion_jobs,(unsigned long long)h.demotion_admitted,
                (unsigned long long)h.demotion_published,(unsigned long long)h.demotion_bytes,
                (unsigned long long)residency.d2h_bytes,(unsigned long long)residency.d2h_jobs);
            GGML_LOG_INFO("STRATA_RAM_POOL bytes=%llu peak=%llu puts=%llu reuses=%llu frees=%llu allocations=%llu\n",
                (unsigned long long)h.pool_bytes,(unsigned long long)h.pool_peak,(unsigned long long)h.pool_puts,
                (unsigned long long)h.pool_reuses,(unsigned long long)h.pool_frees,(unsigned long long)h.allocations);
            GGML_LOG_INFO("STRATA_RAM_CACHE scope=shared bytes=%llu budget=%llu peak=%llu hits=%llu misses=%llu hit_bytes=%llu source_bytes=%llu fill_bytes=%llu evictions=%llu prefill_bypasses=%llu frequency_bypasses=%llu oom=%llu pending=%llu lease_hits=%llu lease_misses=%llu lease_bytes=%llu retired_bytes=%llu gpu_drop_bytes=%llu gpu_drop_entries=%llu deferred_reads=%llu retain_calls=%llu retain_hits=%llu\n",
                (unsigned long long)h.bytes,(unsigned long long)h.budget,(unsigned long long)h.peak_bytes,
                (unsigned long long)h.hits,(unsigned long long)h.misses,(unsigned long long)h.hit_bytes,
                (unsigned long long)h.file_bytes,(unsigned long long)h.fill_bytes,(unsigned long long)h.evictions,
                (unsigned long long)h.prefill_bypasses,(unsigned long long)h.frequency_bypasses,
                (unsigned long long)h.oom,(unsigned long long)h.pending,
                (unsigned long long)h.lease_hits,(unsigned long long)h.lease_misses,(unsigned long long)h.lease_bytes,
                (unsigned long long)h.retired_bytes,(unsigned long long)h.gpu_drop_bytes,(unsigned long long)h.gpu_drop_entries,
                (unsigned long long)h.deferred_reads,(unsigned long long)h.retain_calls,(unsigned long long)h.retain_hits);
        }
        GGML_LOG_INFO("STRATA_CPU_CONTEXT_RING contexts=2 rotations=%llu reuse_waits=%llu\n",
            (unsigned long long)context_rotations,(unsigned long long)context_reuse_waits);
        GGML_LOG_INFO("STRATA_EARLY_HITS ops=%llu hits=%llu misses=%llu\n",
            (unsigned long long)stats.early_hit_ops,(unsigned long long)stats.early_hit_experts,
            (unsigned long long)stats.early_miss_experts);
        GGML_LOG_INFO("STRATA_HASH_PREFETCH reads=%llu plans=%llu reused=%llu mismatches=%llu\n",
            (unsigned long long)stats.hash_reads,(unsigned long long)stats.early_plans,
            (unsigned long long)stats.early_reuses,(unsigned long long)stats.early_mismatches);
        GGML_LOG_INFO("STRATA_EXPERT_DIRECT hits=%llu skipped_d2d_bytes=%llu materialized_bytes=%llu table_updates=%llu admissions=%llu\n",
            (unsigned long long)stats.direct_hits,(unsigned long long)stats.direct_bytes,
            (unsigned long long)stats.direct_materialized_bytes,(unsigned long long)stats.direct_tables,(unsigned long long)stats.direct_admissions);
        GGML_LOG_INFO("STRATA_EXPERT_PREFILL hits=%llu\n",(unsigned long long)stats.prefill_hits);
        for(int i=0;i<2;++i) {
            if(ready[i]) CUDA_CHECK(cudaEventDestroy(ready[i]));
            if(stage[i]) CUDA_CHECK(cudaFreeHost(stage[i]));
        }
        for(auto &entry:entries)entry.second.residency.reset();
        if(arena) CUDA_CHECK(cudaFree(arena));
        for(auto &entry:entries) if(entry.second.allocation) free_allocation(entry.second.allocation);
        GGML_LOG_INFO("STRATA_EXPERT_TOTAL hits=%llu misses=%llu bypass=%llu evictions=%llu h2d_bytes=%llu d2d_bytes=%llu stage_us=%llu wait_us=%llu submit_us=%llu pipeline_groups=%llu pipeline_chunks=%llu pipeline_unused=%llu pipeline_fallbacks=%llu\n",
            (unsigned long long)stats.hits,(unsigned long long)stats.misses,(unsigned long long)stats.bypass,(unsigned long long)stats.evictions,
            (unsigned long long)stats.h2d_bytes,(unsigned long long)stats.d2d_bytes,(unsigned long long)stats.stage_us,
            (unsigned long long)stats.wait_us,(unsigned long long)stats.submit_us,
            (unsigned long long)stats.pipeline_groups,(unsigned long long)stats.pipeline_chunks,
            (unsigned long long)stats.pipeline_unused,(unsigned long long)stats.pipeline_fallbacks);
        GGML_LOG_INFO("STRATA_EXPERT_IO file_bytes=%llu mmap_bytes=%llu read_peak=%llu ordered_reuses=%llu reuse_events=%llu eviction_syncs=%llu admission_rejects=%llu\n",
            (unsigned long long)stats.file_read_bytes,(unsigned long long)stats.mmap_read_bytes,(unsigned long long)stats.read_peak,
            (unsigned long long)stats.ordered_reuses,(unsigned long long)stats.reuse_events,(unsigned long long)stats.eviction_syncs,
            (unsigned long long)stats.admission_rejects);
    }
    void free_allocation(uint8_t *allocation) {
        CUDA_CHECK(slab?slab->release(allocation):cudaFree(allocation));
    }
    void manage() {
        if(managed)return;
        finish_plan();
        // finish_plan retires direct readers (or materializes an abandoned
        // preparation); wait before returning arena allocations to the driver.
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        entries.clear();
        if(arena)CUDA_CHECK(cudaFree(arena));
        arena=nullptr;lru.clear();free_blocks.clear();managed=true;
    }
    uint8_t *evict(bool reuse=false,cudaStream_t consumer=nullptr,size_t prefer_bytes=0) {
        auto candidate=evictable(prefer_bytes);
        if(candidate==lru.rend())return nullptr;
        ggml_cuda_set_device(device);
        flush_copies();
        auto it=entries.find(*candidate);
        residency.evict(it->second.residency,it->second.allocation,stream,!strata_cpu_decode);
        it->second.residency.reset();
        if(reuse) {
            stats.reuse_events+=reuse_fence.order(stream,consumer);
            ++stats.ordered_reuses;
        } else {
            CUDA_CHECK(cudaStreamSynchronize(stream));++stats.eviction_syncs;
        }
        auto *ptr=it->second.allocation;
        allocated-=it->second.bytes;
        if(!reuse) {free_allocation(ptr);ptr=nullptr;}
        lru.erase(it->second.lru);entries.erase(it);++stats.evictions;
        return ptr;
    }
    void release(size_t offset,size_t bytes) {
        auto next=free_blocks.lower_bound(offset);
        if(next!=free_blocks.begin()) {
            auto prev=std::prev(next);
            if(prev->first+prev->second==offset) {offset=prev->first;bytes+=prev->second;free_blocks.erase(prev);}
        }
        if(next!=free_blocks.end() && offset+bytes==next->first) {bytes+=next->second;free_blocks.erase(next);}
        free_blocks.emplace(offset,bytes);
    }
    size_t allocate(size_t bytes,const void *key) {
        for(;;) {
            for(auto it=free_blocks.begin();it!=free_blocks.end();++it) if(it->second>=bytes) {
                size_t offset=it->first, remaining=it->second-bytes;
                free_blocks.erase(it);
                if(remaining)free_blocks.emplace(offset+bytes,remaining);
                return offset;
            }
            // An exact-size victim creates a usable hole in one eviction.
            // Stay within the existing 64-entry LRU window, preserve planned
            // readers, and apply frequency admission to the actual victim.
            auto candidate=evictable(match_size?bytes:0);
            if(candidate==lru.rend())return SIZE_MAX;
            if(!admits(key,*this,*candidate))return SIZE_MAX;
            flush_copies(); // enqueue reads before an arena slot can be overwritten
            auto old=entries.find(*candidate);
            residency.evict(old->second.residency,arena+old->second.offset,stream,!strata_cpu_decode);
            release(old->second.offset,old->second.bytes);
            lru.erase(old->second.lru);entries.erase(old);++stats.evictions;
        }
    }
    void upload(void *dst,const void *src,size_t bytes,bool retain_ram=false) {
        flush_copies(); // retain copy/padding order across cache misses and bypass
        if(pipeline) {
            if(pipeline->transfer(dst,src,bytes,stream,delivery_observer,ram_on_bypass && retain_ram && strata_cpu_decode))return;
            ++stats.pipeline_fallbacks;finish_plan();
        }
        stats.h2d_bytes+=bytes;
        if(!stage_size) {
            int64_t t=ggml_time_us();
            copy_device(dst,src,bytes,cudaMemcpyHostToDevice);
            stats.submit_us+=ggml_time_us()-t;
            return;
        }
        size_t pos=0;
        while(pos<bytes) {
            size_t i=turn++%2, n=std::min(stage_size,bytes-pos);
            int64_t t=ggml_time_us();
            if(pending[i]) CUDA_CHECK(cudaEventSynchronize(ready[i]));
            stats.wait_us+=ggml_time_us()-t;
            t=ggml_time_us();
            std::memcpy(stage[i],(const uint8_t *)src+pos,n);
            stats.stage_us+=ggml_time_us()-t;stats.stage_bytes+=n;
            t=ggml_time_us();
            copy_device((uint8_t *)dst+pos,stage[i],n,cudaMemcpyHostToDevice);
            CUDA_CHECK(cudaEventRecord(ready[i],stream));pending[i]=true;
            stats.submit_us+=ggml_time_us()-t;
            pos+=n;
        }
    }
    void copy(const ggml_tensor *source,ggml_tensor *dest,int first,int last,int tokens) {
        ggml_cuda_set_device(device);
        size_t width=source->nb[2];
        int count=int(source->ne[2]);
        GGML_ASSERT(first>=0 && last>=first && last<count && width>0);
        // The quantized bytes and their original indices stay unchanged. The
        // extra 512 bytes preserve GGML's MMQ padding contract at expert edges.
        if((!capacity || (tokens>8 && !prefill_cache_hits)) && !pipeline) {
            size_t offset=size_t(first)*width,bytes=size_t(last-first+1)*width+(last<count-1?std::min<size_t>(width,512):0);
            stats.bypass+=last-first+1;
            upload((uint8_t *)dest->data+offset,(const uint8_t *)source->data+offset,bytes);
            return;
        }
        for(int id=first;id<=last;++id) {
            size_t offset=size_t(id)*width,bytes=width+(id<count-1?std::min<size_t>(width,512):0);
            const void *key=(const uint8_t *)source->data+offset;
            if(cpu_lane.selected(dest,id) || cpu_whole().selected(dest,id)) {
                if(frequency)frequency->record(key);
                if(cpu_whole().selected(dest,id))cpu_whole().saved_bytes+=bytes;else cpu_lane.saved_bytes+=bytes;
                continue;
            }
            void *destination=direct.destination(dest,id);
            if(tokens>8 && prefill_hit(destination,key,bytes))continue;
            if(bytes>capacity || tokens>8) {++stats.bypass;upload(destination,key,bytes,tokens<=8);continue;}
            if(frequency)frequency->record(key);
            auto found=entries.find(key);
            const bool was_hit=found!=entries.end();
            if(found!=entries.end()) {
                GGML_ASSERT(found->second.bytes==bytes);
                ++stats.hits;lru.splice(lru.begin(),lru,found->second.lru);
                if(pipeline)pipeline->transfer(nullptr,key,bytes,stream);
            } else {
                ++stats.misses;
                size_t slot=allocate(bytes,key);
                if(slot==SIZE_MAX) {++stats.bypass;upload(destination,key,bytes,true);continue;}
                upload(arena+slot,key,bytes);
                lru.push_front(key);
                found=entries.emplace(key,Entry{slot,bytes,lru.begin()}).first;
                ++stats.gpu_admissions;
                promoted(key,found->second);
            }
            if(!read_direct(dest,id,key,arena+found->second.offset,bytes,was_hit)) {
                copy_cached(destination,arena+found->second.offset,bytes);
                stats.d2d_bytes+=bytes;
            }
        }
    }
};

// Access is serial per backend, matching GGML's graph execution contract. State
// is released before the CUDA context/stream, never shared across model loads.
static std::unordered_map<ggml_backend_t,std::unique_ptr<StrataExpertTransfer>> strata_expert_transfers;
static std::mutex strata_expert_transfer_mutex;
#ifdef STRATA_DS4_COST_EXPERIMENT
static void strata_cuda_expert_profile_drain() {
    std::lock_guard<std::mutex> lock(strata_expert_transfer_mutex);
    for(auto &entry:strata_expert_transfers)entry.second->profile_drain();
}
#endif
struct StrataDeviceCacheControl {
    bool enabled=false;
    StrataVramPolicy policy;
    StrataVramStatus status;
    uint64_t clock=0;
};
static std::map<int,StrataDeviceCacheControl> strata_device_cache_controls;
static uint64_t strata_expert_touch_clock(int device){return ++strata_device_cache_controls[device].clock;}

static StrataExpertTransfer &strata_expert_state(ggml_backend_t backend) {
    auto found=strata_expert_transfers.find(backend);
    if(found==strata_expert_transfers.end()) {
        auto *ctx=(ggml_backend_cuda_context *)backend->context;
        auto state=std::make_unique<StrataExpertTransfer>(ctx,strata_device_cache_controls[ctx->device].enabled);
        // Publish only a fully constructed state. Control/stats may be called
        // after an allocation failure and must never encounter a null entry.
        found=strata_expert_transfers.emplace(backend,std::move(state)).first;
    }
    return *found->second;
}

static void strata_cache_totals(int device,StrataVramStatus &out) {
    out.cache_bytes=out.matrices=out.cache_reserved_bytes=out.slab_blocks=out.slab_allocations=0;
    std::unordered_set<strata_ds4::ExpertSlabAllocator *> pools;
    for(auto &kv:strata_expert_transfers) if(kv.second->device==device) {
        auto &s=*kv.second;
        out.cache_bytes+=s.managed?s.allocated:s.capacity;
        out.matrices+=s.entries.size();for(const auto &context:s.branch_contexts)out.matrices+=context.loan_count;
        if(s.managed && s.slab) {
            if(pools.insert(s.slab.get()).second) {
                auto status=s.slab->status();
                out.cache_reserved_bytes+=status.reserved;
                out.slab_blocks+=status.blocks;out.slab_allocations+=status.block_allocations;
            }
        } else out.cache_reserved_bytes+=s.managed?s.allocated:s.capacity;
    }
}
static StrataExpertTransfer *strata_oldest_cache(int device,size_t prefer_bytes=0) {
    StrataExpertTransfer *oldest=nullptr;
    uint64_t stamp=std::numeric_limits<uint64_t>::max();
    bool matched=false;
    for(auto &kv:strata_expert_transfers) {
        auto &s=*kv.second;
        if(s.device!=device || s.lru.empty())continue;
        auto candidate=s.evictable(prefer_bytes);if(candidate==s.lru.rend())continue;
        const auto &entry=s.entries.find(*candidate)->second;
        bool match=prefer_bytes && entry.bytes==prefer_bytes;
        if(!oldest || (match && !matched) || (match==matched && entry.touched<stamp)) {
            stamp=entry.touched;oldest=&s;matched=match;
        }
    }
    return oldest;
}
static bool strata_cuda_expert_control(int device,const StrataVramPolicy *policy,StrataVramStatus *out) {
    std::lock_guard<std::mutex> lock(strata_expert_transfer_mutex);
    auto &control=strata_device_cache_controls[device];
    if(policy) {
        if(!policy->valid())return false;
        control.enabled=true;control.policy=*policy;
        for(auto &kv:strata_expert_transfers) if(kv.second->device==device)kv.second->manage();
    }
    ggml_cuda_set_device(device);
    control.status.enabled=control.enabled;
    size_t free=0,total=0;
    control.status.telemetry_ok=strata_device_memory(device,free,total);
    strata_cache_totals(device,control.status);
    // Manual release works even if the memory provider is temporarily down.
    if(control.enabled) {
        const auto max_count=control.policy.mode==1?control.policy.matrices:UINT64_MAX;
        while(control.status.matrices>max_count) {
            auto *oldest=strata_oldest_cache(device);
            if(!oldest)break;
            oldest->evict();strata_cache_totals(device,control.status);
        }
        if(control.policy.mode==0)for(auto &kv:strata_expert_transfers) {
            auto &s=*kv.second;
            if(s.device==device)while(s.allocated>s.capacity)s.evict();
        }
        // Account for releases when deriving the non-cache portion below.
        strata_cache_totals(device,control.status);
        control.status.telemetry_ok=strata_device_memory(device,free,total);
    }
    if(control.status.telemetry_ok) {
        control.status.free_bytes=free;control.status.total_bytes=total;
        control.status.limit_bytes=control.policy.byte_limit(free,total,control.status.cache_reserved_bytes);
        if(control.enabled) {
            const auto max_count=control.policy.mode==1?control.policy.matrices:UINT64_MAX;
            while(control.status.cache_reserved_bytes>control.status.limit_bytes || control.status.matrices>max_count) {
                auto *oldest=strata_oldest_cache(device);
                if(!oldest)break;
                oldest->evict();strata_cache_totals(device,control.status);
            }
            // A restored configured budget is also enforced per model/backend.
            if(control.policy.mode==0)for(auto &kv:strata_expert_transfers) {
                auto &s=*kv.second;
                if(s.device==device)while(s.allocated>s.capacity)s.evict();
            }
            strata_cache_totals(device,control.status);
            control.status.telemetry_ok=strata_device_memory(device,free,total);
            if(control.status.telemetry_ok) {control.status.free_bytes=free;control.status.total_bytes=total;}
        }
        auto ceiling=total-std::min<uint64_t>(total,control.policy.reserve_mib<<20);
        if(control.policy.mode==2)ceiling=std::min(ceiling,control.policy.target_mib<<20);
        control.status.target_unreachable=control.enabled && total-free>ceiling && !control.status.cache_bytes;
    } else {
        // No reliable free-memory reading: keep existing copies, admit no more.
        control.status.limit_bytes=control.status.cache_reserved_bytes;
    }
    *out=control.status;
    return true;
}

StrataExpertTransfer::Entries::iterator StrataExpertTransfer::reserve_fetch_slot(size_t bytes) {
    if(!bytes || bytes>capacity)return entries.end();
    size_t offset=0;uint8_t *pointer=nullptr;
    if(!managed) {
        auto block=free_blocks.begin();
        while(block!=free_blocks.end() && block->second<bytes)++block;
        if(block==free_blocks.end())return entries.end();
        offset=block->first;const auto left=block->second-bytes;free_blocks.erase(block);
        if(left)free_blocks.emplace(offset+bytes,left);pointer=arena+offset;
    } else {
        auto &c=strata_device_cache_controls[device];strata_cache_totals(device,c.status);
        const uint64_t count=c.policy.mode==1?c.policy.matrices:UINT64_MAX;
        if(!c.status.telemetry_ok || c.status.matrices>=count ||
           (c.policy.mode==0 && allocated+bytes>capacity) ||
           bytes>c.status.limit_bytes-std::min(c.status.limit_bytes,c.status.cache_bytes))return entries.end();
        size_t free=0,total=0;uint64_t growth=0;
        if(strata_device_memory(device,free,total)) {
            const auto limit=std::min(c.status.limit_bytes,c.policy.byte_limit(free,total,c.status.cache_reserved_bytes));
            growth=limit-std::min<uint64_t>(limit,c.status.cache_reserved_bytes+(64ULL<<20));
        }
        bool denied=false;
        const auto error=slab?slab->allocate((void **)&pointer,bytes,size_t(growth),&denied):
            growth>=bytes?cudaMalloc((void **)&pointer,bytes):cudaSuccess;
        if(error!=cudaSuccess){cudaGetLastError();if(!denied)++c.status.allocation_failures;return entries.end();}
        if(!pointer)return entries.end();allocated+=bytes;
    }
    // Empty reservations have an allocation identity, never a model key, and
    // participate in the same LRU/budget/release mechanism as ordinary entries.
    lru.push_back(pointer);
    auto result=entries.emplace(pointer,Entry{offset,bytes,std::prev(lru.end()),managed?pointer:nullptr,0,{},true});
    GGML_ASSERT(result.second);return result.first;
}

// Dynamic matrices use individual allocations or shared slabs. Equal-size LRU
// replacements retain their slot after ordering its reads against the consumer.
static void strata_managed_copy(StrataExpertTransfer &s,const ggml_tensor *source,ggml_tensor *dest,int first,int last,int tokens) {
    auto &c=strata_device_cache_controls[s.device];
    const size_t width=source->nb[2];
    const int count=int(source->ne[2]);
    GGML_ASSERT(first>=0 && last>=first && last<count && width>0);
    if(tokens>8 && !s.prefill_cache_hits && !s.pipeline) {
        size_t offset=size_t(first)*width,bytes=size_t(last-first+1)*width+(last<count-1?std::min<size_t>(width,512):0);
        s.stats.bypass+=last-first+1;s.upload((uint8_t *)dest->data+offset,(const uint8_t *)source->data+offset,bytes);return;
    }
    for(int id=first;id<=last;++id) {
        size_t offset=size_t(id)*width,bytes=width+(id<count-1?std::min<size_t>(width,512):0);
        const void *key=(const uint8_t *)source->data+offset;
        if(s.cpu_lane.selected(dest,id) || s.cpu_whole().selected(dest,id)) {
            if(s.frequency)s.frequency->record(key);
            if(s.cpu_whole().selected(dest,id))s.cpu_whole().saved_bytes+=bytes;else s.cpu_lane.saved_bytes+=bytes;
            continue;
        }
        void *destination=s.direct.destination(dest,id);
        if(tokens>8 && s.prefill_hit(destination,key,bytes))continue;
        if(tokens>8) {++s.stats.bypass;s.upload(destination,key,bytes);continue;}
        if(s.frequency)s.frequency->record(key);
        auto found=s.entries.find(key);
        const bool was_hit=found!=s.entries.end();
        if(found!=s.entries.end()) {
            ++s.stats.hits;s.lru.splice(s.lru.begin(),s.lru,found->second.lru);
            if(s.pipeline)s.pipeline->transfer(nullptr,key,bytes,s.stream);
        } else {
            strata_cache_totals(s.device,c.status);
            const uint64_t max_count=c.policy.mode==1?c.policy.matrices:UINT64_MAX;
            uint8_t *allocation=nullptr;
            bool fits=c.status.telemetry_ok && bytes<=c.status.limit_bytes && max_count>0 &&
                (c.policy.mode!=0 || bytes<=s.capacity);
            if(fits) {
                bool rejected=false;
                while(c.status.cache_bytes+bytes>c.status.limit_bytes || c.status.matrices>=max_count ||
                      (c.policy.mode==0 && s.allocated+bytes>s.capacity)) {
                    size_t prefer=allocation?0:bytes;
                    auto *oldest=c.policy.mode==0 && s.allocated+bytes>s.capacity?&s:strata_oldest_cache(s.device,prefer);
                    if(!oldest)break;
                    auto candidate=oldest->evictable(prefer);if(candidate==oldest->lru.rend())break;
                    if(!allocation && !s.admits(key,*oldest,*candidate)) {rejected=true;break;}
                    auto old_bytes=oldest->entries.find(*candidate)->second.bytes;
                    bool reuse=!allocation && old_bytes==bytes && oldest->slab==s.slab;
                    auto *released=oldest->evict(reuse,s.stream,prefer);
                    if(reuse)allocation=released;
                    strata_cache_totals(s.device,c.status);
                }
                fits=!rejected && c.status.cache_bytes+bytes<=c.status.limit_bytes && c.status.matrices<max_count &&
                    (c.policy.mode!=0 || s.allocated+bytes<=s.capacity);
                if(!fits && allocation) {
                    // A reused slot may still have reads ordered on this stream.
                    CUDA_CHECK(cudaStreamSynchronize(s.stream));
                    s.free_allocation(allocation);allocation=nullptr;
                }
                ggml_cuda_set_device(s.device);
                if(fits && !allocation) {
                    // Recheck at allocation time: another process can allocate
                    // after the last periodic sample. Keep a growth margin.
                    size_t free=0,total=0;
                    const uint64_t reserved=c.status.cache_reserved_bytes;
                    uint64_t growth=0;
                    // Use the same memory domain as the controller. Under
                    // WDDM CUDA's process view can disagree with global NVML
                    // usage and otherwise deny growth after pressure clears.
                    if(strata_device_memory(s.device,free,total)) {
                        const auto limit=std::min(c.status.limit_bytes,c.policy.byte_limit(free,total,reserved));
                        const auto charged=reserved+(64ULL<<20);
                        growth=limit-std::min<uint64_t>(limit,charged);
                    }
                    // Slab holes remain usable even when physical growth stops.
                    bool growth_denied=false;
                    const auto error=s.slab?s.slab->allocate((void **)&allocation,bytes,size_t(growth),&growth_denied):
                        growth>=bytes?cudaMalloc((void **)&allocation,bytes):cudaSuccess;
                    if(error!=cudaSuccess) {
                        cudaGetLastError();if(!growth_denied)++c.status.allocation_failures;allocation=nullptr;
                    }
                    if(!allocation) {
                        // Admission headroom can prevent growth before the
                        // logical budget is full. Keep the cache adaptive by
                        // replacing an equal-size entry without allocating.
                        auto *oldest=strata_oldest_cache(s.device,bytes);
                        if(oldest) {
                            auto candidate=oldest->evictable(bytes);
                            if(oldest->slab==s.slab && candidate!=oldest->lru.rend() && oldest->entries.find(*candidate)->second.bytes==bytes &&
                               s.admits(key,*oldest,*candidate))
                                allocation=oldest->evict(true,s.stream,bytes);
                        }
                    }
                }
            }
            if(!allocation) {++s.stats.bypass;s.upload(destination,key,bytes,true);continue;}
            ++s.stats.misses;s.upload(allocation,key,bytes);s.lru.push_front(key);
            found=s.entries.emplace(key,StrataExpertTransfer::Entry{0,bytes,s.lru.begin(),allocation,0}).first;
            ++s.stats.gpu_admissions;
            s.allocated+=bytes;s.promoted(key,found->second);
        }
        found->second.touched=++c.clock;
        if(!s.read_direct(dest,id,key,found->second.allocation,bytes,was_hit)) {
            s.copy_cached(destination,found->second.allocation,bytes);
            s.stats.d2d_bytes+=bytes;
        }
    }
}
static void strata_cuda_expert_copy(ggml_backend_t backend,const ggml_tensor *source,ggml_tensor *dest,int first,int last,int tokens) {
    std::lock_guard<std::mutex> lock(strata_expert_transfer_mutex);
    auto *ctx=(ggml_backend_cuda_context *)backend->context;
    auto &state=strata_expert_state(backend);
    GGML_ASSERT(state.stream==ctx->stream());
    GGML_ASSERT(strata_expert_thread_budget<0 || state.capacity==(size_t(strata_expert_thread_budget)<<20));
    if(state.managed)strata_managed_copy(state,source,dest,first,last,tokens);
    else state.copy(source,dest,first,last,tokens);
}
static void strata_cuda_expert_stats(ggml_backend_t backend,StrataExpertCounters *out) {
    std::lock_guard<std::mutex> lock(strata_expert_transfer_mutex);
    auto it=strata_expert_transfers.find(backend);
    *out=it==strata_expert_transfers.end()?StrataExpertCounters{}:it->second->counters();
}
static void strata_cuda_expert_snapshot(uint64_t request,const char *phase) {
    std::lock_guard<std::mutex> lock(strata_expert_transfer_mutex);
    GGML_LOG_INFO("STRATA_EXPERT_BOUNDARY request=%llu phase=%s\n",(unsigned long long)request,phase);
    for(auto &entry:strata_expert_transfers) {
        auto &s=*entry.second;const auto c=s.counters();
        uint64_t ram_hits=0,branches=0;
        for(auto &b:s.branch_contexts){ram_hits+=b.cpu.ram_hits;branches+=b.cpu.queue_branches;}
        const auto borrowed=s.pipeline?s.pipeline->counters().borrowed_bytes:0;
        GGML_LOG_INFO("STRATA_EXPERT_SNAPSHOT request=%llu phase=%s backend=%llu cache_bytes=%llu hits=%llu misses=%llu admissions=%llu h2d_bytes=%llu d2d_bytes=%llu d2h_bytes=%llu file_bytes=%llu mmap_bytes=%llu ram_cpu_hits=%llu ram_gpu_bytes=%llu cpu_branches=%llu cpu_jobs=%llu cpu_rows=%llu fetch_branches=%llu graph_changes=%llu graph_captures=%llu graph_replays=%llu\n",
            (unsigned long long)request,phase,(unsigned long long)uintptr_t(entry.first),(unsigned long long)s.capacity,
            (unsigned long long)c.hits,(unsigned long long)c.misses,(unsigned long long)c.gpu_admissions,
            (unsigned long long)c.h2d_bytes,(unsigned long long)c.d2d_bytes,(unsigned long long)c.gpu_demotion_bytes,
            (unsigned long long)c.file_read_bytes,(unsigned long long)c.mmap_read_bytes,(unsigned long long)ram_hits,
            (unsigned long long)(c.gpu_fetch_ram_bytes+borrowed),(unsigned long long)branches,
            (unsigned long long)c.cpu_queue_jobs,(unsigned long long)c.cpu_queue_rows,(unsigned long long)c.gpu_fetch_branches,
            (unsigned long long)c.cpu_graph_changes,(unsigned long long)c.cuda_graph_captures,(unsigned long long)c.cuda_graph_replays);
    }
}
static void strata_cuda_expert_plan(ggml_backend_t backend,const StrataExpertSlice *slices,size_t count) {
    std::lock_guard<std::mutex> lock(strata_expert_transfer_mutex);
    strata_expert_state(backend).plan(slices,count);
}
static void strata_cuda_expert_early_plan(ggml_backend_t backend,const StrataExpertSlice *slices,size_t count) {
    if(!count)return;
    std::lock_guard<std::mutex> lock(strata_expert_transfer_mutex);
    auto &s=strata_expert_state(backend);
    // Starting a new early plan never treats an older prediction as validated.
    s.finish_plan();s.plan(slices,count);
    if(s.pipeline) {s.early_slices.assign(slices,slices+count);++s.stats.early_plans;}
}
static bool strata_cuda_expert_hash_read(ggml_backend_t backend,StrataHashRoutes *routes) {
    auto *ctx=(ggml_backend_cuda_context *)backend->context;
    if(!routes || routes->count<1 || routes->count>3 || routes->n_tokens<1 || routes->n_tokens>8 ||
       ctx->curr_stream_no!=0 || !GGML_CUDA_CC_IS_NVIDIA(ggml_cuda_info().devices[ctx->device].cc))return false;
    StrataHashLookup args{};args.count=routes->count;args.n_tokens=routes->n_tokens;
    for(int t=0;t<args.n_tokens;++t)args.tokens[t]=routes->tokens[t];
    for(int i=0;i<args.count;++i) {
        const auto *table=routes->tables[i];
        if(!table || table->op!=GGML_OP_NONE || (table->flags&GGML_TENSOR_FLAG_INPUT) ||
           table->type!=GGML_TYPE_I32 || table->view_src || !table->data || !table->buffer ||
           table->buffer->buft!=ggml_backend_cuda_buffer_type(ctx->device) ||
           ggml_backend_buffer_get_usage(table->buffer)!=GGML_BACKEND_BUFFER_USAGE_WEIGHTS ||
           !ggml_is_contiguous(table) || table->ne[0]<1 || table->ne[0]>8 ||
           table->ne[1]<1 || table->ne[2]!=1 || table->ne[3]!=1)return false;
        for(int t=0;t<args.n_tokens;++t)if(args.tokens[t]<0 || args.tokens[t]>=table->ne[1])return false;
        args.tables[i]=(const int32_t *)table->data;args.widths[i]=int(table->ne[0]);
    }
    std::lock_guard<std::mutex> lock(strata_expert_transfer_mutex);
    auto &s=strata_expert_state(backend);
    if(!s.pipeline)return false;
    ggml_cuda_set_device(s.device);
    if(!s.hash_scratch.allocate())return false;
    strata_hash_lookup<<<1,192,0,s.stream>>>(args,s.hash_scratch.device);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(s.hash_scratch.host,s.hash_scratch.device,sizeof(routes->ids),cudaMemcpyDeviceToHost,s.stream));
    CUDA_CHECK(cudaStreamSynchronize(s.stream));
    for(int i=0;i<args.count;++i)
        std::memcpy(routes->ids[i],s.hash_scratch.host+i*64,args.widths[i]*args.n_tokens*sizeof(int32_t));
    ++s.stats.hash_reads;return true;
}
static void strata_cuda_expert_finish(ggml_backend_t backend) {
    std::lock_guard<std::mutex> lock(strata_expert_transfer_mutex);
    auto it=strata_expert_transfers.find(backend);
    if(it!=strata_expert_transfers.end())it->second->finish_plan();
}
static void strata_cuda_cpu_cost_boundary(int operation) {
    if(!strata_ds4::cpu_cost_auto_enabled())return;
    std::lock_guard<std::mutex> lock(strata_expert_transfer_mutex);
    // All backend instances share one request generation. Retire target and
    // draft rings (including service replies and deferred GPU consumers) before
    // the registry can publish any successor tables.
    for(auto &entry:strata_expert_transfers) {
        auto &s=*entry.second;ggml_cuda_set_device(s.device);s.finish_plan();
    }
    strata_ds4::cpu_cost_request_boundary(operation);
}
static void strata_cuda_expert_flush(ggml_backend_t backend) {
    std::lock_guard<std::mutex> lock(strata_expert_transfer_mutex);
    auto it=strata_expert_transfers.find(backend);
    if(it!=strata_expert_transfers.end()) {
        auto &s=*it->second;s.flush_copies();s.early_hits.join(s.stream);
        if(s.direct.dirty) {ggml_cuda_set_device(s.device);s.direct.publish(s.stream);++s.stats.direct_tables;}
    }
}
static bool strata_cuda_expert_ordered(ggml_backend_t backend,const ggml_tensor *node) {
    auto *ctx=(ggml_backend_cuda_context *)backend->context;
    if(ctx->curr_stream_no!=0 || !node || node->op!=GGML_OP_MUL_MAT_ID)return false;
    if(!GGML_CUDA_CC_IS_NVIDIA(ggml_cuda_info().devices[ctx->device].cc))return false;
    const auto *w=node->src[0],*x=node->src[1],*ids=node->src[2];
    if(!w || !x || !ids || w->view_src || !w->buffer ||
       w->buffer->buft!=ggml_backend_cuda_buffer_type(ctx->device))return false;
    if(w->type!=GGML_TYPE_IQ3_XXS && w->type!=GGML_TYPE_MXFP4 && w->type!=GGML_TYPE_Q8_0)return false;
    if(!ggml_is_contiguous(w) || w->ne[3]!=1 || w->ne[2]<1 || w->ne[2]>256 ||
       w->ne[0]%256 || w->ne[1]%128)return false;
    if(x->type!=GGML_TYPE_F32 || ids->type!=GGML_TYPE_I32 || ids->nb[0]!=sizeof(int32_t) || node->type!=GGML_TYPE_F32)return false;
    // CUDA graph evaluation joins auxiliary streams back to stream 0. Cached
    // reads also join their readers explicitly in expert_complete(). Selected
    // copies, table publication and subsequent consumers use this same stream.
    // Thus GPU ordering is retained while the host can prepare the next copy.
    return ids->ne[1]>=1 && ids->ne[1]<=8 && node->ne[2]>=1 && node->ne[2]<=8;
}
static int strata_cuda_expert_compact(ggml_backend_t backend,const ggml_tensor *node) {
    const char *flag=std::getenv("STRATA_DS4_COMPACT_EXPERTS");
    const char *direct=std::getenv("STRATA_DS4_DIRECT_CACHE"),*pipe=std::getenv("STRATA_EXPERT_PIPELINE");
    if(!flag || std::strcmp(flag,"1") || !direct || (std::strcmp(direct,"1") && std::strcmp(direct,"2")) ||
       !pipe || std::strcmp(pipe,"1") || !node || node->op!=GGML_OP_MUL_MAT_ID)return 0;
    if(const char *mmq=std::getenv("STRATA_DS4_ROUTED_MMQ"))if(std::strcmp(mmq,"0"))return 0;
    auto *ctx=(ggml_backend_cuda_context *)backend->context;
    const auto *w=node->src[0],*x=node->src[1],*ids=node->src[2];
    const int cc=ggml_cuda_info().devices[ctx->device].cc;
    if(!GGML_CUDA_CC_IS_NVIDIA(cc) || !w || !x || !ids || w->view_src || !ggml_is_contiguous(w) ||
       w->ne[3]!=1 || w->ne[2]>256 || w->ne[0]%256 || w->ne[1]%128 ||
       (w->type!=GGML_TYPE_IQ3_XXS && w->type!=GGML_TYPE_MXFP4 && w->type!=GGML_TYPE_Q8_0) ||
       x->type!=GGML_TYPE_F32 || ids->type!=GGML_TYPE_I32 || ids->nb[0]!=4 || node->type!=GGML_TYPE_F32 ||
       node->ne[2]<1 || node->ne[2]>8 || node->ne[2]>get_mmvq_mmid_max_batch(w->type,cc))return 0;
    const int64_t slots=ids->ne[0]*ids->ne[1];
    return slots>0 && slots<w->ne[2]?int(slots):0;
}
static bool strata_cuda_expert_prepare(ggml_backend_t backend,const ggml_tensor *node) {
    std::lock_guard<std::mutex> lock(strata_expert_transfer_mutex);
    auto &s=strata_expert_state(backend);
    s.retire_direct();
    if(!s.direct.enabled || !node || node->op!=GGML_OP_MUL_MAT_ID)return false;
    const auto *w=node->src[0],*x=node->src[1],*ids=node->src[2];
    const int cc=ggml_cuda_info().devices[s.device].cc;
    if(!GGML_CUDA_CC_IS_NVIDIA(cc) || !w || !x || !ids)return false;
    if(w->type!=GGML_TYPE_IQ3_XXS && w->type!=GGML_TYPE_MXFP4 && w->type!=GGML_TYPE_Q8_0)return false;
    if(w->view_src || !ggml_is_contiguous(w) || w->ne[3]!=1 || w->ne[2]<1 || w->ne[2]>256)return false;
    // MMVQ may read complete row tiles before masking output rows. Cache
    // entries only carry 512 bytes of edge padding, not another whole row.
    if(w->ne[0]%256 || w->ne[1]%128)return false;
    if(x->type!=GGML_TYPE_F32 || ids->type!=GGML_TYPE_I32 || ids->nb[0]!=sizeof(int32_t) || node->type!=GGML_TYPE_F32)return false;
    if(node->ne[2]<1 || node->ne[2]>8 || node->ne[2]>get_mmvq_mmid_max_batch(w->type,cc))return false;
    // A separately enabled research dispatch can replace MMVQ with MMQ.
    const char *mmq=std::getenv("STRATA_DS4_ROUTED_MMQ");
    if(mmq && (std::strcmp(mmq,"1")==0 || std::strcmp(mmq,"2")==0))return false;
    ggml_cuda_set_device(s.device);s.direct.prepare(w);s.early_hits.prepare(s.stream);return true;
}
#include "cpu_lane_hooks.inc"
static bool strata_cuda_expert_early_hits(ggml_backend_t backend,const ggml_tensor *source,
        ggml_tensor *node,const ggml_tensor *route,const int32_t *selected,size_t count) {
#if !defined(STRATA_DS4_DIRECT_CACHE_COMPILED) || !defined(STRATA_DS4_EARLY_HITS_COMPILED)
    GGML_UNUSED(backend);GGML_UNUSED(source);GGML_UNUSED(node);GGML_UNUSED(route);GGML_UNUSED(selected);GGML_UNUSED(count);
    return false;
#else
    if(!strata_cuda_expert_ordered(backend,node) || !source || !route || !count || count>256)return false;
    auto *ctx=(ggml_backend_cuda_context *)backend->context;
    auto local_gpu=[&](const ggml_tensor *t) {return t && t->buffer && t->data &&
        t->buffer->buft==ggml_backend_cuda_buffer_type(ctx->device);};
    if(!local_gpu(node) || !local_gpu(node->src[1]) || !local_gpu(route) ||
       !ggml_is_contiguous(node) || node->ne[3]!=1 || ggml_nbytes(node)>StrataExpertEarlyHitScratch::output_bytes ||
       source->ne[2]!=node->src[0]->ne[2] || source->nb[2]!=node->src[0]->nb[2] ||
       !ggml_are_same_shape(route,node->src[2]) || route->type!=GGML_TYPE_I32 || route->nb[0]!=4)return false;
    std::lock_guard<std::mutex> lock(strata_expert_transfer_mutex);
    auto &s=strata_expert_state(backend);
    if(!s.early_hits.enabled || !s.pipeline || s.direct.tensor!=node->src[0] || !s.early_hits.pointers)return false;
    const auto *x=node->src[1];
    const size_t quantized_bytes=size_t(x->ne[3])*x->ne[2]*x->ne[1]*
        GGML_PAD(x->ne[0],MATRIX_ROW_PADDING)*sizeof(block_q8_1)/QK8_1;
    if(s.early_hits.separate && quantized_bytes>StrataExpertEarlyHitScratch::input_bytes)return false;
    StrataExpertDirect::Packet packet{};
    size_t hits=0;
    for(size_t i=0;i<count;++i) {
        int id=selected[i];if(id<0 || id>=source->ne[2])return false;
        const void *key=(const uint8_t *)source->data+size_t(id)*source->nb[2];
        auto found=s.entries.find(key);
        if(found==s.entries.end())continue;
        // The current layer plan retains these allocations throughout all
        // selected copies. The owner stream joins separate early work before
        // retirement, so cross-context eviction fences also include its reads.
        if(s.protected_keys().find(key)==s.protected_keys().end())return false;
        if(!packet.pointers[id])++hits;
        packet.pointers[id]=s.managed?found->second.allocation:s.arena+found->second.offset;
    }
    if(!hits || hits==count)return false;
    ggml_cuda_set_device(s.device);s.flush_copies();
    auto stream=s.early_hits.begin(s.stream);s.early_hits.publish(stream,packet);
    auto output=*node;output.data=s.early_hits.output;
    strata_expert_early_matvec(*ctx,node->src[0],node->src[1],route,&output,s.early_hits.pointers,
        stream,s.early_hits.input);
    s.early_hits.end();
    CUDA_CHECK(cudaGetLastError());
    ++s.stats.early_hit_ops;s.stats.early_hit_experts+=hits;s.stats.early_miss_experts+=count-hits;
    return true;
#endif
}
void strata_expert_early_result(ggml_backend_cuda_context &ctx,const ggml_tensor *weight,
        const void *const **pointers,const float **output) {
    std::lock_guard<std::mutex> lock(strata_expert_transfer_mutex);
    for(auto &entry:strata_expert_transfers) {
        auto &s=*entry.second;
        if(s.context==&ctx && s.direct.tensor==weight && s.direct.data==weight->data && s.early_hits.enabled) {
            *pointers=s.early_hits.pointers;*output=s.early_hits.output;return;
        }
    }
}
const void *const *strata_expert_direct_table(ggml_backend_cuda_context &ctx,const ggml_tensor *weight) {
    std::lock_guard<std::mutex> lock(strata_expert_transfer_mutex);
    for(auto &entry:strata_expert_transfers) {
        auto &s=*entry.second;
        if(s.context==&ctx && s.direct.tensor==weight && s.direct.data==weight->data) {
            GGML_ASSERT(!s.direct.dirty);s.direct.reader(ctx.stream());return s.direct.table;
        }
    }
    return nullptr;
}
static void strata_cuda_expert_complete(ggml_backend_t backend,ggml_backend_cuda_context *ctx) {
    std::lock_guard<std::mutex> lock(strata_expert_transfer_mutex);
    auto it=strata_expert_transfers.find(backend);
    if(it!=strata_expert_transfers.end()) {
        auto &s=*it->second;s.direct.complete(s.stream,ctx->stream());
        if(s.cpu_whole().configured && s.direct.tensor==s.cpu_whole().branch.nodes[2]->src[0]) {
            CUDA_CHECK(cudaEventRecord(s.cpu_whole().consumed,s.stream));s.cpu_whole().completion=true;
        }
    }
}
static void strata_cuda_expert_free(ggml_backend_t backend) {
    std::lock_guard<std::mutex> lock(strata_expert_transfer_mutex);
    auto it=strata_expert_transfers.find(backend);
    if(it!=strata_expert_transfers.end() && it->second->cpu_whole().stream) {
        auto &s=*it->second;ggml_cuda_set_device(s.device);s.finish_plan();
        CUDA_CHECK(cudaStreamSynchronize(s.stream));
#ifdef USE_CUDA_GRAPH
        // External event handles and stable packet/output buffers must outlive
        // every graph which references them, not only the last graph launch.
        s.context->cuda_graphs.clear();
#endif
    }
    strata_expert_transfers.erase(backend);
}
