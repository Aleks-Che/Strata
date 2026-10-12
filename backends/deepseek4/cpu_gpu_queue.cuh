#pragma once
#include <cuda_runtime.h>
#include <cuda/atomic>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <new>
#include <stdexcept>
#include <thread>

namespace strata_ds4 {
// SPSC, one producer CUDA stream and one persistent CPU service per backend.
// All cross-processor atomics are aligned 32-bit loads/stores: no mapped-host
// RMW, which would require hostNativeAtomicSupported. Payload ownership:
// GPU publish -> CPU reply -> GPU consume -> next GPU publish.
using CpuDoorbell=cuda::atomic<uint32_t,cuda::thread_scope_system>;
constexpr int cpu_queue_depth=2, cpu_queue_floats=(2<<20)/sizeof(float);
enum CpuQueueError : uint32_t { cpu_ok, cpu_bad_request, cpu_cancelled, cpu_exception, cpu_timeout };
struct CpuQueueMatrix { uint64_t address,generation; };
struct CpuQueueRow { CpuQueueMatrix matrix[3];int expert,token,slot; };
struct CpuQueuePacket { uint8_t mask[256];int slots[64]; };
struct CpuQueueRequest {
    uint64_t backend,epoch,layer;
    int columns,tokens,topk,rows;
    int32_t route[64];
    CpuQueueRow row[64];
    CpuQueuePacket packet;
};
struct alignas(64) CpuQueueSlot {
    CpuDoorbell request{0},reply{0},consumed{0};
    CpuDoorbell gpu_failure{0}; // ring[0], written by the producer/consumer stream
    uint32_t error=cpu_ok;
    CpuQueueRequest job;
    float input[cpu_queue_floats],output[cpu_queue_floats];
};
struct CpuQueueDevice {
    uint32_t sequence=0,active=0,error=cpu_ok;
    uint64_t full=0,published=0;
};
inline void cpu_queue_check(cudaError_t error) {
    if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));
}
class CpuGpuQueue {
    std::thread service;
    std::mutex mutex;
    std::condition_variable wake;
    bool idle=false;
    std::atomic<bool> stop{false},armed{false};
    std::atomic<uint64_t> cancelled_epoch{0};
    std::function<void(CpuQueueSlot &)> compute;
    void run() noexcept {
        for(;;) {
            {std::unique_lock<std::mutex> lock(mutex);idle=true;wake.notify_all();
                wake.wait(lock,[&]{return stop.load() || armed.load();});idle=false;}
            if(stop.load())return;
            bool found=false;
            for(int i=0;i<cpu_queue_depth;++i) {
                auto &slot=host[i];const auto seq=slot.request.load(cuda::memory_order_acquire);
                if(!seq || seq==slot.reply.load(cuda::memory_order_relaxed))continue;
                found=true;
                slot.error=cpu_ok;
                if(!slot.job.epoch || slot.job.epoch<=cancelled_epoch.load())slot.error=cpu_cancelled;
                else try {compute(slot);}catch(...) {slot.error=cpu_exception;}
                slot.reply.store(seq,cuda::memory_order_release);
                // No CUDA calls, cache changes or lease destruction here.
            }
            if(!found)std::this_thread::yield();
        }
    }
public:
    CpuQueueSlot *host=nullptr,*mapped=nullptr;
    CpuQueueDevice *device=nullptr;
    explicit CpuGpuQueue(std::function<void(CpuQueueSlot &)> fn):compute(std::move(fn)) {
        int dev=0,map=0,uva=0;cpu_queue_check(cudaGetDevice(&dev));
        cpu_queue_check(cudaDeviceGetAttribute(&map,cudaDevAttrCanMapHostMemory,dev));
        cpu_queue_check(cudaDeviceGetAttribute(&uva,cudaDevAttrUnifiedAddressing,dev));
        if(!map || !uva)throw std::runtime_error("CPU GPU queue requires mapped host memory and UVA");
        try {
            cpu_queue_check(cudaHostAlloc((void **)&host,sizeof(CpuQueueSlot)*cpu_queue_depth,cudaHostAllocMapped));
            for(int i=0;i<cpu_queue_depth;++i)new(host+i) CpuQueueSlot;
            cpu_queue_check(cudaHostGetDevicePointer((void **)&mapped,host,0));
            cpu_queue_check(cudaMalloc((void **)&device,sizeof(CpuQueueDevice)));
            cpu_queue_check(cudaMemset(device,0,sizeof(CpuQueueDevice)));
            service=std::thread([this]{run();});
        }catch(...) {if(device)cudaFree(device);if(host)cudaFreeHost(host);throw;}
    }
    CpuGpuQueue(const CpuGpuQueue &)=delete;
    ~CpuGpuQueue() {
        // Caller drains producer/consumer streams before releasing the queue.
        {std::lock_guard<std::mutex> lock(mutex);stop.store(true);}
        wake.notify_all();if(service.joinable())service.join();
        cudaFree(device);for(int i=0;i<cpu_queue_depth;++i)host[i].~CpuQueueSlot();cudaFreeHost(host);
    }
    void arm() {{std::lock_guard<std::mutex> lock(mutex);armed.store(true);}wake.notify_one();}
    void cancel(uint64_t epoch) {cancelled_epoch.store(epoch);arm();}
    // Cancellation may skip the down graph. Only after both stream and CPU
    // drain may the host acknowledge that no GPU reader remains for these slots.
    void abandon_after_drain() {
        for(int i=0;i<cpu_queue_depth;++i) {
            const auto request=host[i].request.load(cuda::memory_order_acquire);
            if(request!=host[i].reply.load(cuda::memory_order_acquire))throw std::runtime_error("abandoning a live CPU job");
            host[i].consumed.store(request,cuda::memory_order_release);
        }
    }
    // Producer must have finished before this call (including cancellation).
    // A timeout never releases a live snapshot or silently accepts stale data.
    void drain() {
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(10);
        for(;;) {
            bool done=true;
            for(int i=0;i<cpu_queue_depth;++i)
                done&=host[i].request.load(cuda::memory_order_acquire)==host[i].reply.load(cuda::memory_order_acquire);
            if(done)break;
            if(std::chrono::steady_clock::now()>deadline)throw std::runtime_error("CPU service drain timed out");
            std::this_thread::yield();
        }
        std::unique_lock<std::mutex> lock(mutex);armed.store(false);
        if(!wake.wait_until(lock,deadline,[&]{return idle;}))throw std::runtime_error("CPU service did not park");
    }
};

// Only thread zero calls reserve. Full -> no CPU decision has been published;
// caller clears the CPU mask and executes the entire branch on GPU.
static __device__ inline uint32_t cpu_queue_reserve(CpuQueueSlot *ring,CpuQueueDevice *state) {
    state->active=0;state->error=cpu_ok;
    if(state->sequence==UINT32_MAX){++state->full;return 0;} // never alias a wrapped reply
    const uint32_t seq=state->sequence+1;auto &slot=ring[seq%cpu_queue_depth];
    const auto previous=slot.request.load(cuda::memory_order_acquire);
    if(previous!=slot.reply.load(cuda::memory_order_acquire) || previous!=slot.consumed.load(cuda::memory_order_acquire)) {
        ++state->full;return 0;
    }
    state->sequence=state->active=seq;++state->published;return seq;
}
static __global__ void cpu_queue_reset(CpuQueueSlot *ring,CpuQueueDevice *state) {
    state->active=0;state->error=cpu_ok;ring[0].gpu_failure.store(cpu_ok,cuda::memory_order_release);
}
// Late join copies the complete CPU output into ordinary device memory. No
// weighted reduction here: the existing scatter preserves original top-k rows.
static __global__ void cpu_queue_join(CpuQueueSlot *ring,CpuQueueDevice *state,float *out,uint64_t timeout_ns) {
    __shared__ uint32_t seq,error;
    if(threadIdx.x==0) {
        seq=state->active;error=cpu_ok;
        if(seq) {
            auto &s=ring[seq%cpu_queue_depth];uint64_t start,now;
            asm volatile("mov.u64 %0, %%globaltimer;":"=l"(start));
            while(s.reply.load(cuda::memory_order_acquire)!=seq) {
                asm volatile("mov.u64 %0, %%globaltimer;":"=l"(now));
                if(now-start>timeout_ns){error=cpu_timeout;break;}
                __nanosleep(256);
            }
            if(!error)error=s.error;state->error=error;
            if(error)ring[0].gpu_failure.store(error,cuda::memory_order_release);
        }
    }
    __syncthreads();if(!seq)return;
    auto &s=ring[seq%cpu_queue_depth];
    for(int i=threadIdx.x;i<s.job.rows*s.job.columns;i+=blockDim.x)out[i]=error?__int_as_float(0x7fffffff):s.output[i];
    __syncthreads();
    // A timed-out CPU still owns its output; reserve also checks reply.
    if(threadIdx.x==0)s.consumed.store(seq,cuda::memory_order_release);
}
}
