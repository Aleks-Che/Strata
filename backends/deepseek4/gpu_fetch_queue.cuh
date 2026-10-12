#pragma once
#include "cpu_gpu_queue.cuh"
#include <algorithm>
#include <array>
#include <cstring>
#include <cstdlib>
#include <cooperative_groups.h>

namespace strata_ds4 {
// Three matrix slots, one unfused branch per backend. GPU publication is after
// the previous reader of the shared weight scratch. The I/O service never
// allocates CUDA memory, edits a cache, or calls back into the scheduler.
struct GpuFetchPlan {
    uint64_t epoch;
    const void *cached[256];
    void *scratch;
    size_t width;
    int experts,topk,tokens,compact;
};
struct GpuFetchRequest {
    uint64_t epoch;
    int count;
    int ids[64];
    void *destinations[64];
};
struct alignas(64) GpuFetchSlot {
    CpuDoorbell request{0},reply{0},consumed{0},failure{0};
    CpuDoorbell chunk{0},chunk_done{0};
    const uint8_t *source=nullptr;
    uint8_t *destination=nullptr;
    size_t bytes=0;
    uint32_t error=0;
    GpuFetchRequest job;
};
struct GpuFetchConsumer {uint32_t sequence,chunk,error,done;uint64_t start;};

class GpuFetchQueue {
    std::thread service;
    std::mutex mutex;
    std::condition_variable wake;
    bool idle=false;
    std::atomic<bool> stop{false},armed{false};
    std::function<void(int,GpuFetchSlot &,GpuFetchQueue &)> execute;
    uint8_t *stage[2]={};
    GpuFetchSlot *current=nullptr;
    size_t turn=0;
    void wait_chunk() {
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(2);
        while(current->chunk.load(cuda::memory_order_relaxed)!=current->chunk_done.load(cuda::memory_order_acquire)) {
            if(current->failure.load(cuda::memory_order_acquire) || std::chrono::steady_clock::now()>deadline)
                throw std::runtime_error("GPU fetch chunk consumer failed");
            std::this_thread::yield();
        }
    }
    void run() noexcept {
        for(;;) {
            {std::unique_lock<std::mutex> lock(mutex);idle=true;wake.notify_all();
                wake.wait(lock,[&]{return stop.load() || armed.load();});idle=false;}
            if(stop.load())return;
            bool found=false;
            for(int m=0;m<3;++m) {
                auto &s=host[m];const auto seq=s.request.load(cuda::memory_order_acquire);
                if(!seq || seq==s.reply.load(cuda::memory_order_relaxed))continue;
                found=true;s.error=cpu_ok;current=&s;
                if(cancelled.load())s.error=cpu_cancelled;
                else try {execute(m,s,*this);wait_chunk();}catch(...) {s.error=cpu_exception;}
                // No CUDA calls here: GGML may synchronously allocate while a
                // consumer is queued. A CUDA-API service would then deadlock.
                s.reply.store(seq,cuda::memory_order_release);
            }
            if(!found)std::this_thread::yield();
        }
    }
public:
    static constexpr size_t stage_bytes=2ULL<<20;
    GpuFetchSlot *host=nullptr,*mapped=nullptr;
    GpuFetchConsumer *consumer=nullptr;
    int copy_blocks=1;
    std::atomic<bool> cancelled{false};
    explicit GpuFetchQueue(std::function<void(int,GpuFetchSlot &,GpuFetchQueue &)> fn):execute(std::move(fn)) {
        try {
            int device=0,cooperative=0,sms=0;cpu_queue_check(cudaGetDevice(&device));
            cpu_queue_check(cudaDeviceGetAttribute(&cooperative,cudaDevAttrCooperativeLaunch,device));
            cpu_queue_check(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device));
            int requested=16;if(const char *v=std::getenv("STRATA_DS4_GPU_FETCH_BLOCKS"))requested=std::max(1,std::min(16,std::atoi(v)));
            copy_blocks=cooperative?std::min(sms,requested):1;
            cpu_queue_check(cudaMalloc((void **)&consumer,sizeof(GpuFetchConsumer)*3));
            cpu_queue_check(cudaHostAlloc((void **)&host,sizeof(GpuFetchSlot)*3,cudaHostAllocMapped));
            for(int m=0;m<3;++m)new(host+m) GpuFetchSlot;
            cpu_queue_check(cudaHostGetDevicePointer((void **)&mapped,host,0));
            for(int i=0;i<2;++i) {
                cpu_queue_check(cudaHostAlloc((void **)&stage[i],stage_bytes,cudaHostAllocMapped));
                void *mapped_stage=nullptr;cpu_queue_check(cudaHostGetDevicePointer(&mapped_stage,stage[i],0));
                if(mapped_stage!=stage[i])throw std::runtime_error("GPU fetch requires identical UVA pinned addresses");
            }
            service=std::thread([this]{run();});
        } catch(...) {release();throw;}
    }
    GpuFetchQueue(const GpuFetchQueue &)=delete;
    ~GpuFetchQueue() {
        // Caller has drained the producer and parked this service first.
        {std::lock_guard<std::mutex> lock(mutex);stop.store(true);}
        wake.notify_all();if(service.joinable())service.join();release();
    }
    void release() {
        if(consumer)cudaFree(consumer);
        for(int i=0;i<2;++i)if(stage[i])cudaFreeHost(stage[i]);
        if(host)cudaFreeHost(host);
    }
    void arm() {
        {std::lock_guard<std::mutex> lock(mutex);cancelled.store(false);armed.store(true);}wake.notify_one();
    }
    void wait(int matrix,cudaStream_t owner,uint64_t timeout_ns=500000000ULL);
    // Only the I/O service calls these. A RAM hit uses the retained pinned
    // allocation directly; native reads alternate two preallocated buffers.
    void upload(void *dest,const void *source,size_t bytes) {
        wait_chunk();const auto seq=current->chunk.load(cuda::memory_order_relaxed);
        if(seq==UINT32_MAX)throw std::runtime_error("GPU fetch chunk sequence exhausted");
        current->source=(const uint8_t *)source;current->destination=(uint8_t *)dest;current->bytes=bytes;
        current->chunk.store(seq+1,cuda::memory_order_release);
    }
    template<class Read> void read_upload(void *dest,size_t bytes,Read read) {
        for(size_t offset=0;offset<bytes;) {
            if(cancelled.load())throw std::runtime_error("GPU fetch cancelled");
            const auto i=turn++%2,n=std::min(stage_bytes,bytes-offset);
            // The previous upload uses the other buffer. Publish waits for its
            // GPU acknowledgement, allowing this native read to overlap it.
            if(!read(offset,stage[i],n))throw std::runtime_error("GPU fetch read failed");
            upload((uint8_t *)dest+offset,stage[i],n);
            offset+=n;
        }
    }
    // Called only after all producer/consumer kernels have finished, including
    // graph cancellation. No timeout may release sources of a live GPU read.
    void drain() {
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(10);
        for(;;) {
            bool done=true;for(int m=0;m<3;++m)
                done&=host[m].request.load(cuda::memory_order_acquire)==host[m].reply.load(cuda::memory_order_acquire);
            if(done)break;
            if(std::chrono::steady_clock::now()>deadline)throw std::runtime_error("GPU fetch drain timed out");
            std::this_thread::yield();
        }
        std::unique_lock<std::mutex> lock(mutex);armed.store(false);
        if(!wake.wait_until(lock,deadline,[&]{return idle;}))throw std::runtime_error("GPU fetch service did not park");
    }
};

// No route IDs return to the main host thread. Cache hits directly populate the
// normal MMVQ pointer table; only unique GPU-required misses reach the service.
static __global__ void gpu_fetch_publish(GpuFetchSlot *slot,GpuFetchPlan plan,
        const int32_t *ids,int stride,const CpuQueuePacket *cpu,const void **table) {
    const int id=threadIdx.x;
    if(id<256)table[id]=id<plan.experts?plan.cached[id]:nullptr;
    __syncthreads();if(id)return;
    slot->failure.store(cpu_ok,cuda::memory_order_release);
    const auto previous=slot->request.load(cuda::memory_order_acquire);
    if(!plan.epoch || previous==UINT32_MAX || previous!=slot->reply.load(cuda::memory_order_acquire) ||
       previous!=slot->consumed.load(cuda::memory_order_acquire)) {
        slot->failure.store(cpu_bad_request,cuda::memory_order_release);return;
    }
    auto &job=slot->job;job.epoch=plan.epoch;job.count=0;
    bool used[256]={};
    for(int t=0;t<plan.tokens;++t)for(int j=0;j<plan.topk;++j) {
        const int e=ids[t*stride+j];
        if(e<0 || e>=plan.experts) {slot->failure.store(cpu_bad_request,cuda::memory_order_release);return;}
        if(used[e])continue;used[e]=true;
        if((cpu && cpu->mask[e]) || plan.cached[e])continue;
        if(job.count>=64 || (plan.compact && job.count>=plan.compact)) {
            slot->failure.store(cpu_bad_request,cuda::memory_order_release);return;
        }
        const auto offset=plan.compact?size_t(job.count)*(plan.width+512):size_t(e)*plan.width;
        auto *destination=(uint8_t *)plan.scratch+offset;
        table[e]=destination;job.ids[job.count]=e;job.destinations[job.count++]=destination;
    }
    __threadfence_system();slot->request.store(previous+1,cuda::memory_order_release);
}
static __device__ inline void gpu_fetch_barrier() {
    if(gridDim.x>1)cooperative_groups::this_grid().sync();else __syncthreads();
}
static __global__ void gpu_fetch_wait(GpuFetchSlot *slot,GpuFetchConsumer *s,uint64_t timeout_ns) {
    const auto rank=threadIdx.x+blockIdx.x*blockDim.x,threads=blockDim.x*gridDim.x;
    if(!rank) {
        s->sequence=slot->request.load(cuda::memory_order_acquire);
        s->error=slot->failure.load(cuda::memory_order_acquire);
        uint64_t start;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(start));s->start=start;
    }
    gpu_fetch_barrier();if(s->error)return;
    for(;;) {
        if(!rank) {
            s->chunk=slot->chunk.load(cuda::memory_order_acquire);
            if(s->chunk==slot->chunk_done.load(cuda::memory_order_relaxed))s->chunk=0;
            s->done=!s->chunk && slot->reply.load(cuda::memory_order_acquire)==s->sequence;
            uint64_t now;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(now));
            if(now-s->start>timeout_ns)s->error=cpu_timeout;
        }
        gpu_fetch_barrier();if(s->error || s->done)break;
        if(s->chunk) {
            // Cooperative launch guarantees residency for the grid barrier.
            // Devices without it use one block and ordinary block barriers.
            const auto n=slot->bytes;
            if(!(uintptr_t(slot->source)%16) && !(uintptr_t(slot->destination)%16)) {
                for(size_t i=rank;i<n/16;i+=threads)
                    ((uint4 *)slot->destination)[i]=((const uint4 *)slot->source)[i];
                for(size_t i=n/16*16+rank;i<n;i+=threads)slot->destination[i]=slot->source[i];
            } else for(size_t i=rank;i<n;i+=threads)slot->destination[i]=slot->source[i];
            __threadfence_system();gpu_fetch_barrier();
            if(!rank)slot->chunk_done.store(s->chunk,cuda::memory_order_release);
        } else __nanosleep(256);
        gpu_fetch_barrier();
    }
    if(!rank) {
        if(!s->error)s->error=slot->error;
        if(s->error)slot->failure.store(s->error,cuda::memory_order_release);
        slot->consumed.store(s->sequence,cuda::memory_order_release);
    }
}
inline void GpuFetchQueue::wait(int matrix,cudaStream_t owner,uint64_t timeout_ns) {
    auto *slot=mapped+matrix;auto *state=consumer+matrix;
    if(copy_blocks>1) {
        void *args[]={&slot,&state,&timeout_ns};
        cpu_queue_check(cudaLaunchCooperativeKernel((void *)gpu_fetch_wait,copy_blocks,256,args,0,owner));
    } else gpu_fetch_wait<<<1,256,0,owner>>>(slot,state,timeout_ns);
    cpu_queue_check(cudaGetLastError());
}
}
