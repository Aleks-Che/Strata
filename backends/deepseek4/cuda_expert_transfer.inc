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

// Weak ownership avoids keeping CUDA allocations alive after the last backend.
// Registry access is covered by strata_expert_transfer_mutex below.
static std::map<int,std::weak_ptr<strata_ds4::ExpertSlabAllocator>> strata_expert_slabs;

static thread_local int strata_expert_thread_budget=-1;
static thread_local int strata_expert_thread_decay=0;
static void strata_cuda_expert_budget(int mib) {
    GGML_ASSERT(mib>=0 && mib<=65536);
    strata_expert_thread_budget=mib;
}
static void strata_cuda_expert_decay(int accesses) {
    GGML_ASSERT(accesses>=0 && accesses<=strata_ds4::max_frequency_decay);
    strata_expert_thread_decay=accesses;
}

struct StrataExpertTransfer {
    struct Entry {
        size_t offset, bytes; std::list<const void *>::iterator lru;
        uint8_t *allocation=nullptr;
        uint64_t touched=0;
    };
    int device;
    cudaStream_t stream;
    uint8_t *arena=nullptr, *stage[2]={nullptr,nullptr};
    cudaEvent_t ready[2]={nullptr,nullptr};
    bool pending[2]={false,false};
    size_t capacity=0, stage_size=0, turn=0;
    std::map<size_t,size_t> free_blocks;
    std::list<const void *> lru;
    std::unordered_map<const void *,Entry> entries;
    StrataExpertCounters stats;
    bool managed=false;
    size_t allocated=0;
    std::shared_ptr<strata_ds4::ExpertSlabAllocator> slab;
    std::unique_ptr<StrataExpertPipeline> pipeline;
    std::unordered_set<const void *> protected_keys;
    StrataExpertReuseFence reuse_fence;
    std::unique_ptr<StrataExpertFrequency> frequency;
    bool match_size=false;
    bool prefill_cache_hits=false;

    bool prefill_hit(void *dest,const void *key,size_t bytes) {
        if(!prefill_cache_hits)return false;
        auto found=entries.find(key);
        if(found==entries.end())return false;
        GGML_ASSERT(found->second.bytes==bytes);
        auto *source=managed?found->second.allocation:arena+found->second.offset;
        CUDA_CHECK(cudaMemcpyAsync(dest,source,bytes,cudaMemcpyDeviceToDevice,stream));
        ++stats.hits;++stats.prefill_hits;stats.d2d_bytes+=bytes;
        // Read-only prefill: no admission or frequency/LRU training.
        return true;
    }

    bool admits(const void *key,const StrataExpertTransfer &owner,const void *victim) {
        if(!frequency || !owner.frequency || frequency->score(key)>=owner.frequency->score(victim))return true;
        ++stats.admission_rejects;return false;
    }

    void finish_plan() {
        if(pipeline)pipeline->finish(false);
        protected_keys.clear();
    }
    StrataExpertCounters counters() {
        auto result=stats;
        if(pipeline) {
            auto p=pipeline->counters();
            result.pipeline_groups=p.groups;result.pipeline_chunks=p.chunks;result.pipeline_unused=p.unused;
            result.h2d_bytes+=p.h2d_bytes;result.d2d_bytes+=p.d2d_bytes;result.stage_bytes+=p.h2d_bytes;
            result.stage_us+=p.read_us;result.wait_us+=p.wait_us;result.submit_us+=p.submit_us;
            result.file_read_bytes=p.file_bytes;result.mmap_read_bytes=p.mmap_bytes;result.read_peak=p.read_peak;
        }
        return result;
    }
    void plan(const StrataExpertSlice *slices,size_t count) {
        finish_plan();
        if(!pipeline)return;
        std::vector<StrataExpertSlice> misses;
        for(size_t i=0;i<count;++i) {
            if((slices[i].cacheable || prefill_cache_hits) && entries.find(slices[i].data)!=entries.end())
                protected_keys.insert(slices[i].data);
            else misses.push_back(slices[i]);
        }
        pipeline->start(misses);
    }
    auto evictable(size_t prefer_bytes=0) {
        auto oldest=lru.rend();
        size_t scanned=0;
        for(auto it=lru.rbegin();it!=lru.rend();++it) {
            if(protected_keys.find(*it)!=protected_keys.end())continue;
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
    explicit StrataExpertTransfer(ggml_backend_cuda_context *ctx,bool dynamic=false):device(ctx->device),stream(ctx->stream()),managed(dynamic) {
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
            if(pipe && std::strcmp(pipe,"1")==0)pipeline=std::make_unique<StrataExpertPipeline>(device,stage_size,write_combined,n_readers,read_mode,
                StrataExpertPipeline::CopyObserver{},1,early_refill,
                avx2?StrataExpertPipeline::HostCopy(strata_ds4_host_copy::copy):StrataExpertPipeline::HostCopy{});
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
    }
    ~StrataExpertTransfer() {
        ggml_cuda_set_device(device);
        finish_plan();
        CUDA_CHECK(cudaStreamSynchronize(stream));
        stats=counters();pipeline.reset();
        GGML_LOG_INFO("STRATA_EXPERT_PREFILL hits=%llu\n",(unsigned long long)stats.prefill_hits);
        for(int i=0;i<2;++i) {
            if(ready[i]) CUDA_CHECK(cudaEventDestroy(ready[i]));
            if(stage[i]) CUDA_CHECK(cudaFreeHost(stage[i]));
        }
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
        // Cached copies are never referenced directly by model graphs. Wait for
        // queued D2D reads before returning their allocation to the driver.
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if(arena)CUDA_CHECK(cudaFree(arena));
        arena=nullptr;entries.clear();lru.clear();free_blocks.clear();managed=true;
    }
    uint8_t *evict(bool reuse=false,cudaStream_t consumer=nullptr,size_t prefer_bytes=0) {
        auto candidate=evictable(prefer_bytes);
        if(candidate==lru.rend())return nullptr;
        ggml_cuda_set_device(device);
        if(reuse) {
            stats.reuse_events+=reuse_fence.order(stream,consumer);
            ++stats.ordered_reuses;
        } else {
            CUDA_CHECK(cudaStreamSynchronize(stream));++stats.eviction_syncs;
        }
        auto it=entries.find(*candidate);
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
            auto old=entries.find(*candidate);
            release(old->second.offset,old->second.bytes);
            lru.erase(old->second.lru);entries.erase(old);++stats.evictions;
        }
    }
    void upload(void *dst,const void *src,size_t bytes) {
        if(pipeline) {
            if(pipeline->transfer(dst,src,bytes,stream))return;
            ++stats.pipeline_fallbacks;finish_plan();
        }
        stats.h2d_bytes+=bytes;
        if(!stage_size) {
            int64_t t=ggml_time_us();
            CUDA_CHECK(cudaMemcpyAsync(dst,src,bytes,cudaMemcpyHostToDevice,stream));
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
            CUDA_CHECK(cudaMemcpyAsync((uint8_t *)dst+pos,stage[i],n,cudaMemcpyHostToDevice,stream));
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
            if(tokens>8 && prefill_hit((uint8_t *)dest->data+offset,key,bytes))continue;
            if(bytes>capacity || tokens>8) {++stats.bypass;upload((uint8_t *)dest->data+offset,key,bytes);continue;}
            if(frequency)frequency->record(key);
            auto found=entries.find(key);
            if(found!=entries.end()) {
                GGML_ASSERT(found->second.bytes==bytes);
                ++stats.hits;lru.splice(lru.begin(),lru,found->second.lru);
                if(pipeline)pipeline->transfer(nullptr,key,bytes,stream);
            } else {
                ++stats.misses;
                size_t slot=allocate(bytes,key);
                if(slot==SIZE_MAX) {++stats.bypass;upload((uint8_t *)dest->data+offset,key,bytes);continue;}
                upload(arena+slot,key,bytes);
                lru.push_front(key);
                found=entries.emplace(key,Entry{slot,bytes,lru.begin()}).first;
            }
            CUDA_CHECK(cudaMemcpyAsync((uint8_t *)dest->data+offset,arena+found->second.offset,bytes,cudaMemcpyDeviceToDevice,stream));
            stats.d2d_bytes+=bytes;
        }
    }
};

// Access is serial per backend, matching GGML's graph execution contract. State
// is released before the CUDA context/stream, never shared across model loads.
static std::unordered_map<ggml_backend_t,std::unique_ptr<StrataExpertTransfer>> strata_expert_transfers;
static std::mutex strata_expert_transfer_mutex;
struct StrataDeviceCacheControl {
    bool enabled=false;
    StrataVramPolicy policy;
    StrataVramStatus status;
    uint64_t clock=0;
};
static std::map<int,StrataDeviceCacheControl> strata_device_cache_controls;

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
        out.matrices+=s.entries.size();
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
        if(tokens>8 && s.prefill_hit((uint8_t *)dest->data+offset,key,bytes))continue;
        if(tokens>8) {++s.stats.bypass;s.upload((uint8_t *)dest->data+offset,key,bytes);continue;}
        if(s.frequency)s.frequency->record(key);
        auto found=s.entries.find(key);
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
                    if(cudaMemGetInfo(&free,&total)==cudaSuccess) {
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
            if(!allocation) {++s.stats.bypass;s.upload((uint8_t *)dest->data+offset,key,bytes);continue;}
            ++s.stats.misses;s.upload(allocation,key,bytes);s.lru.push_front(key);
            found=s.entries.emplace(key,StrataExpertTransfer::Entry{0,bytes,s.lru.begin(),allocation,0}).first;
            s.allocated+=bytes;
        }
        found->second.touched=++c.clock;
        CUDA_CHECK(cudaMemcpyAsync((uint8_t *)dest->data+offset,found->second.allocation,bytes,cudaMemcpyDeviceToDevice,s.stream));
        s.stats.d2d_bytes+=bytes;
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
static void strata_cuda_expert_plan(ggml_backend_t backend,const StrataExpertSlice *slices,size_t count) {
    std::lock_guard<std::mutex> lock(strata_expert_transfer_mutex);
    strata_expert_state(backend).plan(slices,count);
}
static void strata_cuda_expert_finish(ggml_backend_t backend) {
    std::lock_guard<std::mutex> lock(strata_expert_transfer_mutex);
    auto it=strata_expert_transfers.find(backend);
    if(it!=strata_expert_transfers.end())it->second->finish_plan();
}
static void strata_cuda_expert_free(ggml_backend_t backend) {
    std::lock_guard<std::mutex> lock(strata_expert_transfer_mutex);
    strata_expert_transfers.erase(backend);
}
