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

static thread_local int strata_expert_thread_budget=-1;
static void strata_cuda_expert_budget(int mib) {
    GGML_ASSERT(mib>=0 && mib<=65536);
    strata_expert_thread_budget=mib;
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
    std::unique_ptr<StrataExpertPipeline> pipeline;
    std::unordered_set<const void *> protected_keys;

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
        }
        return result;
    }
    void plan(const StrataExpertSlice *slices,size_t count) {
        finish_plan();
        if(!pipeline)return;
        std::vector<StrataExpertSlice> misses;
        for(size_t i=0;i<count;++i) {
            if(slices[i].cacheable && entries.find(slices[i].data)!=entries.end())
                protected_keys.insert(slices[i].data);
            else misses.push_back(slices[i]);
        }
        pipeline->start(misses);
    }
    auto evictable() {
        return std::find_if(lru.rbegin(),lru.rend(),[&](const void *key){return protected_keys.find(key)==protected_keys.end();});
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
        capacity=strata_expert_thread_budget<0?env_mib("STRATA_EXPERT_CACHE_MIB"):size_t(strata_expert_thread_budget)<<20;
        stage_size=env_mib("STRATA_EXPERT_STAGE_MIB");
        const char *wc=std::getenv("STRATA_EXPERT_STAGE_WC");
        bool write_combined=wc && std::strcmp(wc,"1")==0;
        if(capacity && !managed) {
            CUDA_CHECK(cudaMalloc((void **)&arena,capacity));
            free_blocks.emplace(0,capacity);
        }
        if(stage_size) for(int i=0;i<2;++i) {
            CUDA_CHECK(cudaHostAlloc((void **)&stage[i],stage_size,write_combined?cudaHostAllocWriteCombined:cudaHostAllocDefault));
            CUDA_CHECK(cudaEventCreateWithFlags(&ready[i],cudaEventDisableTiming|cudaEventBlockingSync));
        }
        const char *pipe=std::getenv("STRATA_EXPERT_PIPELINE");
        try {
            if(pipe && std::strcmp(pipe,"1")==0)pipeline=std::make_unique<StrataExpertPipeline>(device,stage_size,write_combined);
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
        GGML_LOG_INFO("STRATA_EXPERT_TRANSFER cache_mib=%zu stage_mib=%zu buffers=2 policy=lru prefill_bypass=1 stage_wc=%d pipeline=%d pipeline_slots=%zu\n",capacity>>20,stage_size>>20,int(write_combined),int(bool(pipeline)),pipeline?StrataExpertPipeline::slots:0);
    }
    ~StrataExpertTransfer() {
        ggml_cuda_set_device(device);
        finish_plan();
        CUDA_CHECK(cudaStreamSynchronize(stream));
        stats=counters();pipeline.reset();
        for(int i=0;i<2;++i) {
            if(ready[i]) CUDA_CHECK(cudaEventDestroy(ready[i]));
            if(stage[i]) CUDA_CHECK(cudaFreeHost(stage[i]));
        }
        if(arena) CUDA_CHECK(cudaFree(arena));
        for(auto &entry:entries) if(entry.second.allocation) CUDA_CHECK(cudaFree(entry.second.allocation));
        GGML_LOG_INFO("STRATA_EXPERT_TOTAL hits=%llu misses=%llu bypass=%llu evictions=%llu h2d_bytes=%llu d2d_bytes=%llu stage_us=%llu wait_us=%llu submit_us=%llu pipeline_groups=%llu pipeline_chunks=%llu pipeline_unused=%llu pipeline_fallbacks=%llu\n",
            (unsigned long long)stats.hits,(unsigned long long)stats.misses,(unsigned long long)stats.bypass,(unsigned long long)stats.evictions,
            (unsigned long long)stats.h2d_bytes,(unsigned long long)stats.d2d_bytes,(unsigned long long)stats.stage_us,
            (unsigned long long)stats.wait_us,(unsigned long long)stats.submit_us,
            (unsigned long long)stats.pipeline_groups,(unsigned long long)stats.pipeline_chunks,
            (unsigned long long)stats.pipeline_unused,(unsigned long long)stats.pipeline_fallbacks);
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
    uint8_t *evict(bool reuse=false) {
        auto candidate=evictable();
        if(candidate==lru.rend())return nullptr;
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto it=entries.find(*candidate);
        auto *ptr=it->second.allocation;
        allocated-=it->second.bytes;
        if(!reuse) {CUDA_CHECK(cudaFree(ptr));ptr=nullptr;}
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
    size_t allocate(size_t bytes) {
        for(;;) {
            for(auto it=free_blocks.begin();it!=free_blocks.end();++it) if(it->second>=bytes) {
                size_t offset=it->first, remaining=it->second-bytes;
                free_blocks.erase(it);
                if(remaining)free_blocks.emplace(offset+bytes,remaining);
                return offset;
            }
            auto candidate=evictable();
            if(candidate==lru.rend())return SIZE_MAX;
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
        if((!capacity || tokens>8) && !pipeline) {
            size_t offset=size_t(first)*width,bytes=size_t(last-first+1)*width+(last<count-1?std::min<size_t>(width,512):0);
            stats.bypass+=last-first+1;
            upload((uint8_t *)dest->data+offset,(const uint8_t *)source->data+offset,bytes);
            return;
        }
        for(int id=first;id<=last;++id) {
            size_t offset=size_t(id)*width,bytes=width+(id<count-1?std::min<size_t>(width,512):0);
            const void *key=(const uint8_t *)source->data+offset;
            if(bytes>capacity || tokens>8) {++stats.bypass;upload((uint8_t *)dest->data+offset,key,bytes);continue;}
            auto found=entries.find(key);
            if(found!=entries.end()) {
                GGML_ASSERT(found->second.bytes==bytes);
                ++stats.hits;lru.splice(lru.begin(),lru,found->second.lru);
                if(pipeline)pipeline->transfer(nullptr,key,bytes,stream);
            } else {
                ++stats.misses;
                size_t slot=allocate(bytes);
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
    out.cache_bytes=out.matrices=0;
    for(auto &kv:strata_expert_transfers) if(kv.second->device==device) {
        auto &s=*kv.second;
        out.cache_bytes+=s.managed?s.allocated:s.capacity;
        out.matrices+=s.entries.size();
    }
}
static StrataExpertTransfer *strata_oldest_cache(int device) {
    StrataExpertTransfer *oldest=nullptr;
    uint64_t stamp=std::numeric_limits<uint64_t>::max();
    for(auto &kv:strata_expert_transfers) {
        auto &s=*kv.second;
        if(s.device!=device || s.lru.empty())continue;
        auto candidate=s.evictable();if(candidate==s.lru.rend())continue;
        auto touched=s.entries.find(*candidate)->second.touched;
        if(!oldest || touched<stamp) {stamp=touched;oldest=&s;}
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
        control.status.limit_bytes=control.policy.byte_limit(free,total,control.status.cache_bytes);
        if(control.enabled) {
            const auto max_count=control.policy.mode==1?control.policy.matrices:UINT64_MAX;
            while(control.status.cache_bytes>control.status.limit_bytes || control.status.matrices>max_count) {
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
        control.status.limit_bytes=control.status.cache_bytes;
    }
    *out=control.status;
    return true;
}

// Individually owned allocations allow pressure to release a subset of cached
// matrices. Equal-sized LRU replacements reuse the allocation after its reads
// finish; a failed allocation falls back to the normal streaming path.
static void strata_managed_copy(StrataExpertTransfer &s,const ggml_tensor *source,ggml_tensor *dest,int first,int last,int tokens) {
    auto &c=strata_device_cache_controls[s.device];
    const size_t width=source->nb[2];
    const int count=int(source->ne[2]);
    GGML_ASSERT(first>=0 && last>=first && last<count && width>0);
    if(tokens>8 && !s.pipeline) {
        size_t offset=size_t(first)*width,bytes=size_t(last-first+1)*width+(last<count-1?std::min<size_t>(width,512):0);
        s.stats.bypass+=last-first+1;s.upload((uint8_t *)dest->data+offset,(const uint8_t *)source->data+offset,bytes);return;
    }
    for(int id=first;id<=last;++id) {
        size_t offset=size_t(id)*width,bytes=width+(id<count-1?std::min<size_t>(width,512):0);
        const void *key=(const uint8_t *)source->data+offset;
        if(tokens>8) {++s.stats.bypass;s.upload((uint8_t *)dest->data+offset,key,bytes);continue;}
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
                while(c.status.cache_bytes+bytes>c.status.limit_bytes || c.status.matrices>=max_count ||
                      (c.policy.mode==0 && s.allocated+bytes>s.capacity)) {
                    auto *oldest=c.policy.mode==0 && s.allocated+bytes>s.capacity?&s:strata_oldest_cache(s.device);
                    if(!oldest)break;
                    auto candidate=oldest->evictable();if(candidate==oldest->lru.rend())break;
                    auto old_bytes=oldest->entries.find(*candidate)->second.bytes;
                    bool reuse=!allocation && old_bytes==bytes;
                    auto *released=oldest->evict(reuse);
                    if(reuse)allocation=released;
                    strata_cache_totals(s.device,c.status);
                }
                fits=c.status.cache_bytes+bytes<=c.status.limit_bytes && c.status.matrices<max_count &&
                    (c.policy.mode!=0 || s.allocated+bytes<=s.capacity);
                if(!fits && allocation) {CUDA_CHECK(cudaFree(allocation));allocation=nullptr;}
                ggml_cuda_set_device(s.device);
                if(fits && !allocation) {
                    // Recheck at allocation time: another process can allocate
                    // after the last periodic sample. Keep a growth margin.
                    size_t free=0,total=0;
                    fits=c.status.cache_bytes+bytes+(64ULL<<20)<=c.status.limit_bytes &&
                        cudaMemGetInfo(&free,&total)==cudaSuccess &&
                        c.policy.byte_limit(free,total,c.status.cache_bytes)>=c.status.cache_bytes+bytes+(64ULL<<20);
                    if(fits && cudaMalloc((void **)&allocation,bytes)!=cudaSuccess) {
                        cudaGetLastError();++c.status.allocation_failures;allocation=nullptr;
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
