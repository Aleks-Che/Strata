// Included only in the private CUDA translation unit of strata-deepseek4.
#include "expert_transfer.h"
#include <list>
#include <map>
#include <memory>
#include <cstdlib>
#include <cstring>
#include <mutex>

struct StrataExpertTransfer {
    struct Entry { size_t offset, bytes; std::list<const void *>::iterator lru; };
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

    static size_t env_mib(const char *name) {
        const char *s=std::getenv(name);
        if (!s) return 0;
        char *end=nullptr; long long value=std::strtoll(s,&end,10);
        GGML_ASSERT(end != s && *end == 0 && value >= 0 && value <= 65536);
        return size_t(value)<<20;
    }
    explicit StrataExpertTransfer(ggml_backend_cuda_context *ctx):device(ctx->device),stream(ctx->stream()) {
        ggml_cuda_set_device(device);
        capacity=env_mib("STRATA_EXPERT_CACHE_MIB");
        stage_size=env_mib("STRATA_EXPERT_STAGE_MIB");
        if(capacity) {
            CUDA_CHECK(cudaMalloc((void **)&arena,capacity));
            free_blocks.emplace(0,capacity);
        }
        if(stage_size) for(int i=0;i<2;++i) {
            CUDA_CHECK(cudaMallocHost((void **)&stage[i],stage_size));
            CUDA_CHECK(cudaEventCreateWithFlags(&ready[i],cudaEventDisableTiming|cudaEventBlockingSync));
        }
        GGML_LOG_INFO("STRATA_EXPERT_TRANSFER cache_mib=%zu stage_mib=%zu buffers=2 policy=lru prefill_bypass=1\n",capacity>>20,stage_size>>20);
    }
    ~StrataExpertTransfer() {
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for(int i=0;i<2;++i) {
            if(ready[i]) CUDA_CHECK(cudaEventDestroy(ready[i]));
            if(stage[i]) CUDA_CHECK(cudaFreeHost(stage[i]));
        }
        if(arena) CUDA_CHECK(cudaFree(arena));
        GGML_LOG_INFO("STRATA_EXPERT_TOTAL hits=%llu misses=%llu bypass=%llu evictions=%llu h2d_bytes=%llu d2d_bytes=%llu stage_us=%llu wait_us=%llu submit_us=%llu\n",
            (unsigned long long)stats.hits,(unsigned long long)stats.misses,(unsigned long long)stats.bypass,(unsigned long long)stats.evictions,
            (unsigned long long)stats.h2d_bytes,(unsigned long long)stats.d2d_bytes,(unsigned long long)stats.stage_us,
            (unsigned long long)stats.wait_us,(unsigned long long)stats.submit_us);
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
            GGML_ASSERT(!lru.empty());
            auto old=entries.find(lru.back());
            release(old->second.offset,old->second.bytes);
            lru.pop_back();entries.erase(old);++stats.evictions;
        }
    }
    void upload(void *dst,const void *src,size_t bytes) {
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
        if(!capacity || tokens>8) {
            size_t offset=size_t(first)*width,bytes=size_t(last-first+1)*width+(last<count-1?std::min<size_t>(width,512):0);
            stats.bypass+=last-first+1;
            upload((uint8_t *)dest->data+offset,(const uint8_t *)source->data+offset,bytes);
            return;
        }
        for(int id=first;id<=last;++id) {
            size_t offset=size_t(id)*width,bytes=width+(id<count-1?std::min<size_t>(width,512):0);
            const void *key=(const uint8_t *)source->data+offset;
            if(bytes>capacity) {++stats.bypass;upload((uint8_t *)dest->data+offset,key,bytes);continue;}
            auto found=entries.find(key);
            if(found!=entries.end()) {
                GGML_ASSERT(found->second.bytes==bytes);
                ++stats.hits;lru.splice(lru.begin(),lru,found->second.lru);
            } else {
                ++stats.misses;
                size_t slot=allocate(bytes);
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
static void strata_cuda_expert_copy(ggml_backend_t backend,const ggml_tensor *source,ggml_tensor *dest,int first,int last,int tokens) {
    std::lock_guard<std::mutex> lock(strata_expert_transfer_mutex);
    auto *ctx=(ggml_backend_cuda_context *)backend->context;
    auto &state=strata_expert_transfers[backend];
    if(!state)state=std::make_unique<StrataExpertTransfer>(ctx);
    GGML_ASSERT(state->stream==ctx->stream());
    state->copy(source,dest,first,last,tokens);
}
static void strata_cuda_expert_stats(ggml_backend_t backend,StrataExpertCounters *out) {
    std::lock_guard<std::mutex> lock(strata_expert_transfer_mutex);
    auto it=strata_expert_transfers.find(backend);
    *out=it==strata_expert_transfers.end()?StrataExpertCounters{}:it->second->stats;
}
static void strata_cuda_expert_free(ggml_backend_t backend) {
    std::lock_guard<std::mutex> lock(strata_expert_transfer_mutex);
    strata_expert_transfers.erase(backend);
}
