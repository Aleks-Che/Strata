#pragma once
#include "cpu_gpu_queue.cuh"
#include <algorithm>
#include <array>
#include <cstring>
#include <cstdlib>
#include <cooperative_groups.h>

namespace strata_ds4 {
// Three matrix slots, one unfused branch per backend context. Publication may
// prepare sources early; wait()/bind runs after the previous scratch reader.
// The I/O service never
// allocates/releases CUDA memory or calls back into the scheduler. RAM metadata
// recycling is permitted only when it cannot release a payload allocation.
static constexpr int gpu_fetch_cache_slots=8;
static constexpr int gpu_fetch_dma_capacity=16;
struct GpuFetchDmaChunk {const void *source;void *destination;size_t bytes;};
struct GpuFetchCacheSlot {
    void *address;size_t bytes;unsigned score;
    int victim=-1; // current-branch expert ID, or an unrelated/empty allocation
};
struct GpuFetchPlan {
    uint64_t epoch;
    const void *cached[256];
    void *scratch;
    size_t width;
    int experts,topk,tokens,compact;
    int cache_count;
    GpuFetchCacheSlot cache[gpu_fetch_cache_slots];
    uint8_t scores[256];
    bool keep_ram[256];
    bool keep_ram_assigned; // protect only actual CPU occurrences in this route
};
struct GpuFetchRequest {
    uint64_t epoch;
    int count;
    int ids[64];
    void *destinations[64];
    int cache_slots[64];
    // Retired after the branch, never awaited to schedule its GPU work.
    int used_count,used_ids[64];
    bool cpu_only[64];
    bool cpu_selected[256]; // includes mixed CPU/GPU occurrences, unlike cpu_only
};
static __host__ __device__ inline bool gpu_fetch_keep_ram(const GpuFetchPlan &plan,
        const GpuFetchRequest &job,int id) {
    return plan.keep_ram[id] && (!plan.keep_ram_assigned || job.cpu_selected[id]);
}
struct alignas(64) GpuFetchSlot {
    CpuDoorbell request{0},reply{0},consumed{0},failure{0};
    CpuDoorbell started{0};
    CpuDoorbell admitted{0}; // owner publishes final destinations before service reads them
    CpuDoorbell chunk{0},chunk_done{0};
    const uint8_t *source=nullptr;
    uint8_t *destination=nullptr;
    size_t bytes=0;
    GpuFetchDmaChunk dma[gpu_fetch_dma_capacity]; // CPU service -> owner, fenced by chunk/chunk_done
    uint32_t error=0;
    GpuFetchRequest job;
};
struct GpuFetchConsumer {uint32_t sequence,chunk,error,done;uint64_t start,progress;};

class GpuFetchQueue {
    std::thread service;
    std::mutex mutex;
    std::condition_variable wake;
    bool idle=false;
    std::atomic<bool> stop{false},armed{false};
    bool service_armed=false; // owner only; callbacks are drained before rearm
    uint64_t progress_generation=0; // protected by mutex; owner notifications cannot be lost
    std::atomic<int> source_turn{-1};
    int source_order[3]={0,1,2};
    std::function<void(int,GpuFetchSlot &,GpuFetchQueue &)> execute;
    uint8_t *stage[2]={};
    size_t stage_buffer_bytes=stage_bytes; // fixed for queue lifetime
    GpuFetchSlot *current=nullptr;
    size_t turn=0;
    cudaEvent_t copies_done=nullptr;
    cudaEvent_t dma_done=nullptr;
    cudaEvent_t dma_events[gpu_fetch_dma_capacity]={};
    GpuFetchSlot *stage_owner[gpu_fetch_dma_capacity]={};
    uint32_t stage_sequence[gpu_fetch_dma_capacity]={};
    bool copies_recorded=false; // owner only, one complete branch per arm
    void wait_chunk() {
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(2);
        while(current->chunk.load(cuda::memory_order_relaxed)!=current->chunk_done.load(cuda::memory_order_acquire)) {
            if(current->failure.load(cuda::memory_order_acquire) || std::chrono::steady_clock::now()>deadline)
                throw std::runtime_error("GPU fetch chunk consumer failed");
            std::this_thread::yield();
        }
    }
    void wait_source(GpuFetchSlot *slot,uint32_t sequence) {
        if(!slot || slot->chunk_done.load(cuda::memory_order_acquire)>=sequence)return;
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(2);
        while(slot->chunk_done.load(cuda::memory_order_acquire)<sequence) {
            if(cancelled.load() || current->failure.load(cuda::memory_order_acquire) || std::chrono::steady_clock::now()>deadline)
                throw std::runtime_error("DMA source retirement failed");
            std::this_thread::yield();
        }
    }
    bool run_ready() noexcept {
        bool found=false;
        for(int m=0;m<3;++m) {
            const int next=source_turn.load(std::memory_order_acquire);
            if(ordered_sources && (next<0 || next>=3 || source_order[next]!=m))continue;
            auto &s=host[m];const auto seq=s.request.load(cuda::memory_order_acquire);
            if(!seq || seq==s.reply.load(cuda::memory_order_relaxed))continue;
            if(owner_admission && !cancelled.load() && s.admitted.load(cuda::memory_order_acquire)!=seq)continue;
            if(owner_empty && !cancelled.load() && !s.job.count)continue;
            found=true;s.error=cpu_ok;current=&s;
            if(cancelled.load())s.error=cpu_cancelled;
            else try {execute(m,s,*this);wait_chunk();}catch(...) {s.error=cpu_exception;}
            // No CUDA API calls on the service, including while the owner allocates.
            if(ordered_sources)source_turn.store(next+1,std::memory_order_release);
            s.reply.store(seq,cuda::memory_order_release);
        }
        return found;
    }
    void run_parked() noexcept {
        std::unique_lock<std::mutex> lock(mutex);
        for(;;) {
            idle=true;wake.notify_all();
            wake.wait(lock,[&]{return stop.load() || armed.load();});idle=false;
            if(stop.load())return;
            while(armed.load() && !stop.load()) {
                const auto seen=progress_generation;
                lock.unlock();
                const auto until=std::chrono::steady_clock::now()+std::chrono::microseconds(service_spin_us);
                bool found;
                do {
                    found=run_ready();
                    if(found || !service_spin_us || !armed.load() || stop.load())break;
                    std::this_thread::yield();
                } while(std::chrono::steady_clock::now()<until);
                lock.lock();
                if(found)continue; // already admitted future matrices remain eligible for early reads
                if(stop.load() || !armed.load() || progress_generation!=seen)continue;
                ++service_waits;const auto start=std::chrono::steady_clock::now();
                // GPU writes do not wake this condition variable. The owner
                // signals after its existing DMA fence/admission, or on cancel.
                wake.wait(lock,[&]{return stop.load() || !armed.load() || progress_generation!=seen;});
                service_wait_us+=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-start).count();
            }
            if(stop.load())return;
        }
    }
    void run() noexcept {
        if(service_park){run_parked();return;}
        for(;;) {
            {std::unique_lock<std::mutex> lock(mutex);idle=true;wake.notify_all();
                wake.wait(lock,[&]{return stop.load() || armed.load();});idle=false;}
            if(stop.load())return;
            if(!run_ready())std::this_thread::yield();
        }
    }
public:
    // Experimental owner-driven DMA. The service still performs no CUDA API
    // calls. Publication and each preceding scratch reader retire before the
    // owner submits a copy; chunk acknowledgement retires its pinned source.
    bool owner_dma=false;
    bool owner_admission=false; // set while parked; requires owner DMA
    bool owner_empty=false; // empty requests retired by owner at existing DMA fence
    bool service_park=false; // opt-in; owner DMA supplies explicit publication/cancel wakes
    int service_spin_us=0;
    std::atomic<uint64_t> service_waits{0},service_wait_us{0}; // may change just after a reply
    uint64_t empty_requests=0;
    uint64_t service_wakes=0,service_skips=0;
    std::function<void(int)> admit; // owner only, after the existing route fence
    std::function<void()> route_ready; // same fence; may wake actual CPU work
    int dma_depth=1; // experimental; bounded independently of staging payload
    int dma_stage_slots=2; // subdivisions of the two pinned allocations
    bool ordered_sources=false; // set only while parked, before arm()
    uint64_t async_disarms=0,park_waits=0,park_us=0; // owner counters
    uint64_t route_wait_us=0,dma_wait_us=0; // owner; DMA path only
    uint64_t early_reads=0,early_read_bytes=0; // service; read after retirement
    uint64_t dma_copies=0,dma_peak_inflight=0; // owner
    uint64_t stage_reuse_waits=0; // service
    static constexpr size_t stage_bytes=2ULL<<20;
    static constexpr size_t max_stage_bytes=8ULL<<20;
    size_t staging_bytes() const {return 2*stage_buffer_bytes;}
    GpuFetchSlot *host=nullptr,*mapped=nullptr;
    GpuFetchConsumer *consumer=nullptr;
    int copy_blocks=1;
    std::atomic<bool> cancelled{false};
    explicit GpuFetchQueue(std::function<void(int,GpuFetchSlot &,GpuFetchQueue &)> fn):execute(std::move(fn)) {
        try {
            if(const char *v=std::getenv("STRATA_DS4_GPU_FETCH_DMA"))owner_dma=std::atoi(v)!=0;
            if(const char *v=std::getenv("STRATA_DS4_GPU_FETCH_EMPTY_FAST"))owner_empty=owner_dma && std::atoi(v)!=0;
            if(const char *v=std::getenv("STRATA_DS4_GPU_FETCH_SERVICE_PARK"))service_park=owner_dma && std::atoi(v)!=0;
            if(const char *v=std::getenv("STRATA_DS4_GPU_FETCH_SERVICE_SPIN_US"))service_spin_us=std::max(0,std::min(1000,std::atoi(v)));
            if(const char *v=std::getenv("STRATA_DS4_GPU_FETCH_DMA_DEPTH"))dma_depth=std::max(1,std::min(gpu_fetch_dma_capacity,std::atoi(v)));
            if(const char *v=std::getenv("STRATA_DS4_GPU_FETCH_DMA_STAGE_SLOTS")) {
                const int n=std::atoi(v);dma_stage_slots=n>=16?16:n>=8?8:n>=4?4:2;
            }
            if(const char *v=std::getenv("STRATA_DS4_GPU_FETCH_STAGE_MIB")) {
                const int n=std::atoi(v);stage_buffer_bytes=size_t(n>=16?8:n>=8?4:2)<<20;
            }
            if(const char *v=std::getenv("STRATA_DS4_GPU_FETCH_EARLY_SOURCES"))ordered_sources=std::atoi(v)!=0;
            int device=0,cooperative=0,sms=0;cpu_queue_check(cudaGetDevice(&device));
            cpu_queue_check(cudaDeviceGetAttribute(&cooperative,cudaDevAttrCooperativeLaunch,device));
            cpu_queue_check(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device));
            int requested=16;if(const char *v=std::getenv("STRATA_DS4_GPU_FETCH_BLOCKS"))requested=std::max(1,std::min(16,std::atoi(v)));
            copy_blocks=cooperative?std::min(sms,requested):1;
            cpu_queue_check(cudaMalloc((void **)&consumer,sizeof(GpuFetchConsumer)*3));
            cpu_queue_check(cudaEventCreateWithFlags(&copies_done,cudaEventDisableTiming));
            cpu_queue_check(cudaEventCreateWithFlags(&dma_done,cudaEventDisableTiming));
            for(auto &event:dma_events)cpu_queue_check(cudaEventCreateWithFlags(&event,cudaEventDisableTiming));
            cpu_queue_check(cudaHostAlloc((void **)&host,sizeof(GpuFetchSlot)*3,cudaHostAllocMapped));
            for(int m=0;m<3;++m)new(host+m) GpuFetchSlot;
            cpu_queue_check(cudaHostGetDevicePointer((void **)&mapped,host,0));
            for(int i=0;i<2;++i) {
                cpu_queue_check(cudaHostAlloc((void **)&stage[i],stage_buffer_bytes,cudaHostAllocMapped));
                void *mapped_stage=nullptr;cpu_queue_check(cudaHostGetDevicePointer(&mapped_stage,stage[i],0));
                if(mapped_stage!=stage[i])throw std::runtime_error("GPU fetch requires identical UVA pinned addresses");
            }
            service=std::thread([this]{run();});
        } catch(...) {release();throw;}
    }
    GpuFetchQueue(const GpuFetchQueue &)=delete;
    ~GpuFetchQueue() {
        // Caller has drained the producer and disarmed this service first.
        {std::lock_guard<std::mutex> lock(mutex);stop.store(true);}
        wake.notify_all();if(service.joinable())service.join();release();
    }
    void release() {
        if(copies_done)cudaEventDestroy(copies_done);
        if(dma_done)cudaEventDestroy(dma_done);
        for(auto event:dma_events)if(event)cudaEventDestroy(event);
        if(consumer)cudaFree(consumer);
        for(int i=0;i<2;++i)if(stage[i])cudaFreeHost(stage[i]);
        if(host)cudaFreeHost(host);
    }
    bool has_copy_fence() const {return copies_recorded;}
    void arm() {
        copies_recorded=false;
        source_turn.store(-1,std::memory_order_relaxed);
        cancelled.store(false);service_armed=false;
        if(!owner_empty)start_service();
    }
    void start_service() {
        if(service_armed)return;
        {std::lock_guard<std::mutex> lock(mutex);armed.store(true);++progress_generation;}
        service_armed=true;++service_wakes;wake.notify_one();
    }
    // Owner only, after acquiring publication. Also used after cancellation:
    // waking a CV is not a substitute for draining outstanding GPU readers.
    void notify_progress() {
        if(!service_park)return;
        {std::lock_guard<std::mutex> lock(mutex);++progress_generation;}
        wake.notify_one();
    }
    // Gate and up can occur in either graph order. A future request must never
    // hold the single I/O service while the owner waits for an earlier one.
    // Publish this order before publishing any of the three GPU requests.
    void order_sources(int first) {
        if(first<0 || first>1)throw std::runtime_error("invalid first expert matrix");
        source_order[0]=first;source_order[1]=1-first;source_order[2]=2;
        source_turn.store(0,std::memory_order_release);
    }
    // Owner records this only after ALL branch fetch waits were enqueued.
    // It covers RAM/staging reads; it does not retire VRAM weight consumers.
    void mark_copies(cudaStream_t owner) {
        cpu_queue_check(cudaEventRecord(copies_done,owner));copies_recorded=true;
    }
    struct Retirement {bool early=false,tail_pending=false;};
    Retirement retire(cudaStream_t owner,bool early,bool wait_idle=true) {
        Retirement result;result.early=early && copies_recorded;
        if(result.early) {
            cpu_queue_check(cudaEventSynchronize(copies_done));
            const auto status=cudaStreamQuery(owner);
            if(status!=cudaErrorNotReady)cpu_queue_check(status);
            result.tail_pending=status==cudaErrorNotReady;
        } else cpu_queue_check(cudaStreamSynchronize(owner));
        drain(wait_idle);return result;
    }
    void wait(int matrix,cudaStream_t owner,uint64_t timeout_ns=2000000000ULL);
    void wait_dma(int matrix,cudaStream_t owner,uint64_t timeout_ns);
    void pump_dma(GpuFetchSlot &slot,cudaStream_t owner,uint32_t sequence,uint64_t timeout_ns);
    // The GPU must acknowledge the final mapped RAM read before its lease can
    // be released. This is a service-side doorbell wait, never a CUDA API call.
    void finish_uploads() {wait_chunk();}
    // Owner only, after draining route producers. Never acknowledge a live
    // GPU read; this stops only requests that have no submitted consumer.
    void abandon(int matrix) {host[matrix].failure.store(cpu_cancelled,cuda::memory_order_release);}
    // Only the I/O service calls these. A RAM hit uses the retained pinned
    // allocation directly; native reads reuse bounded preallocated segments.
    void upload(void *dest,const void *source,size_t bytes) {
        const auto seq=current->chunk.load(cuda::memory_order_relaxed);
        if(seq==UINT32_MAX)throw std::runtime_error("GPU fetch chunk sequence exhausted");
        if(owner_dma && dma_depth>1) {
            if(seq>=uint32_t(dma_depth))wait_source(current,seq-dma_depth+1);
            if(cancelled.load() || current->failure.load(cuda::memory_order_acquire))throw std::runtime_error("DMA upload cancelled");
            current->dma[(seq+1)%dma_depth]={source,dest,bytes};
        } else {
            wait_chunk();current->source=(const uint8_t *)source;current->destination=(uint8_t *)dest;current->bytes=bytes;
        }
        current->chunk.store(seq+1,cuda::memory_order_release);
    }
    template<class Read> void read_upload(void *dest,size_t bytes,Read read,void *retained=nullptr) {
        for(size_t offset=0;offset<bytes;) {
            if(cancelled.load())throw std::runtime_error("GPU fetch cancelled");
            const size_t slots=owner_dma && dma_depth>1?dma_stage_slots:2;
            const auto i=turn++%slots,segment=staging_bytes()/slots,n=std::min(segment,bytes-offset);
            if(owner_dma && dma_depth>1 && !retained) {
                if(stage_owner[i] && stage_owner[i]->chunk_done.load(cuda::memory_order_acquire)<stage_sequence[i])++stage_reuse_waits;
                wait_source(stage_owner[i],stage_sequence[i]);
            }
            // A windowed read owns this segment after its previous DMA retires.
            // The serial path alternates two buffers and upload waits for the
            // previous copy, so preparing this buffer can overlap that copy.
            auto *buffer=retained?(uint8_t *)retained+offset:stage[i/(slots/2)]+(i%(slots/2))*segment;
            const bool early=current->started.load(cuda::memory_order_acquire)!=current->request.load(cuda::memory_order_relaxed);
            if(!read(offset,buffer,n))throw std::runtime_error("GPU fetch read failed");
            if(early){++early_reads;early_read_bytes+=n;}
            upload((uint8_t *)dest+offset,buffer,n);
            if(owner_dma && dma_depth>1 && !retained) {
                stage_owner[i]=current;stage_sequence[i]=current->chunk.load(cuda::memory_order_relaxed);
            }
            offset+=n;
        }
    }
    // Called only after all fetch producer/consumer kernels have finished,
    // including cancellation. Later model compute may still read VRAM weights.
    // No timeout may release sources of a live GPU read.
    // An asynchronous disarm is allowed only when no later publication can
    // occur before the next arm(). Replies retire all callback state access.
    void drain(bool wait_idle=true) {
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(10);
        notify_progress(); // cancellation may bypass all owner-DMA fences
        for(;;) {
            bool done=true;for(int m=0;m<3;++m)
                done&=host[m].request.load(cuda::memory_order_acquire)==host[m].reply.load(cuda::memory_order_acquire);
            if(done)break;
            // Cancellation may abandon publication before its first owner
            // DMA boundary. Wake the service to retire cancelled packets.
            start_service();
            if(std::chrono::steady_clock::now()>deadline)throw std::runtime_error("GPU fetch drain timed out");
            std::this_thread::yield();
        }
        if(!service_armed){++service_skips;return;}
        std::unique_lock<std::mutex> lock(mutex);armed.store(false);service_armed=false;
        if(service_park)wake.notify_all();
        if(!wait_idle){++async_disarms;return;}
        ++park_waits;const auto start=std::chrono::steady_clock::now();
        if(!wake.wait_until(lock,deadline,[&]{return idle;}))throw std::runtime_error("GPU fetch service did not park");
        park_us+=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-start).count();
    }
};

// Cache hits directly populate the normal MMVQ pointer table. The optional
// owner admission pass inspects these published misses at the DMA route fence.
static __global__ void gpu_fetch_publish(GpuFetchSlot *slot,GpuFetchPlan plan,
        const int32_t *ids,int stride,const CpuQueuePacket *cpu,const void **table) {
    const int id=threadIdx.x;
    if(table && id<256)table[id]=id<plan.experts?plan.cached[id]:nullptr;
    __syncthreads();if(id)return;
    slot->failure.store(cpu_ok,cuda::memory_order_release);
    const auto previous=slot->request.load(cuda::memory_order_acquire);
    if(!plan.epoch || plan.cache_count<0 || plan.cache_count>gpu_fetch_cache_slots || previous==UINT32_MAX || previous!=slot->reply.load(cuda::memory_order_acquire) ||
       previous!=slot->consumed.load(cuda::memory_order_acquire)) {
        slot->failure.store(cpu_bad_request,cuda::memory_order_release);return;
    }
    auto &job=slot->job;job.epoch=plan.epoch;job.count=0;job.used_count=0;
    for(int i=0;i<256;++i)job.cpu_selected[i]=false;
    bool used[256]={},claimed[gpu_fetch_cache_slots]={};
    for(int t=0;t<plan.tokens;++t)for(int j=0;j<plan.topk;++j) {
        const int e=ids[t*stride+j];
        if(e<0 || e>=plan.experts) {slot->failure.store(cpu_bad_request,cuda::memory_order_release);return;}
        if(cpu && cpu->slots[t*plan.topk+j]>=0)job.cpu_selected[e]=true;
        if(used[e])continue;used[e]=true;
        if(job.used_count>=64) {slot->failure.store(cpu_bad_request,cuda::memory_order_release);return;}
        job.used_ids[job.used_count]=e;job.cpu_only[job.used_count++]=cpu && cpu->mask[e];
    }
    // A victim can occur later in the packet. Collect the COMPLETE route
    // before admitting anything, including CPU-only and mixed occurrences.
    for(int i=0;i<job.used_count;++i) {
        const int e=job.used_ids[i];
        if((cpu && cpu->mask[e]) || plan.cached[e])continue;
        if(job.count>=64 || (plan.compact && job.count>=plan.compact)) {
            slot->failure.store(cpu_bad_request,cuda::memory_order_release);return;
        }
        const auto offset=plan.compact?size_t(job.count)*(plan.width+512):size_t(e)*plan.width;
        auto *destination=(uint8_t *)plan.scratch+offset;
        const size_t bytes=plan.width+(e<plan.experts-1?(plan.width<512?plan.width:512):0);
        int loan=-1;
        for(int k=0;!gpu_fetch_keep_ram(plan,job,e) && k<plan.cache_count;++k)if(!claimed[k] && plan.cache[k].bytes==bytes && plan.scores[e]>=plan.cache[k].score &&
                (plan.cache[k].victim==-1 || (plan.cache[k].victim>=0 && plan.cache[k].victim<plan.experts && !used[plan.cache[k].victim]))) {
            loan=k;claimed[k]=true;destination=(uint8_t *)plan.cache[k].address;break;
        }
        if(table)table[e]=destination;
        job.ids[job.count]=e;job.cache_slots[job.count]=loan;job.destinations[job.count++]=destination;
    }
    __threadfence_system();slot->request.store(previous+1,cuda::memory_order_release);
}
// Bind at the original matrix boundary, after earlier users of the stable
// pointer table and scratch. Early publication only prepares host sources.
static __global__ void gpu_fetch_bind(const GpuFetchSlot *slot,GpuFetchPlan plan,const void **table) {
    const int id=threadIdx.x;
    if(id<256)table[id]=id<plan.experts?plan.cached[id]:nullptr;
    __syncthreads();
    if(!slot->failure.load(cuda::memory_order_acquire) && id<slot->job.count)
        table[slot->job.ids[id]]=slot->job.destinations[id];
}
static __device__ inline void gpu_fetch_barrier() {
    if(gridDim.x>1)cooperative_groups::this_grid().sync();else __syncthreads();
}
static __global__ void gpu_fetch_wait(GpuFetchSlot *slot,GpuFetchConsumer *s,uint64_t timeout_ns) {
    const auto rank=threadIdx.x+blockIdx.x*blockDim.x,threads=blockDim.x*gridDim.x;
    if(!rank) {
        s->sequence=slot->request.load(cuda::memory_order_acquire);
        slot->started.store(s->sequence,cuda::memory_order_release);
        s->error=slot->failure.load(cuda::memory_order_acquire);
        uint64_t start;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(start));s->start=s->progress=start;
    }
    gpu_fetch_barrier();if(s->error)return;
    for(;;) {
        if(!rank) {
            s->chunk=slot->chunk.load(cuda::memory_order_acquire);
            if(s->chunk==slot->chunk_done.load(cuda::memory_order_relaxed))s->chunk=0;
            s->done=!s->chunk && slot->reply.load(cuda::memory_order_acquire)==s->sequence;
            uint64_t now;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(now));
            if(!s->done && (now-s->progress>timeout_ns || now-s->start>8*timeout_ns))s->error=cpu_timeout;
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
            if(!rank) {
                slot->chunk_done.store(s->chunk,cuda::memory_order_release);
                uint64_t now;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(now));s->progress=now;
            }
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
    if(owner_dma){wait_dma(matrix,owner,timeout_ns);return;}
    auto *slot=mapped+matrix;auto *state=consumer+matrix;
    if(copy_blocks>1) {
        void *args[]={&slot,&state,&timeout_ns};
        cpu_queue_check(cudaLaunchCooperativeKernel((void *)gpu_fetch_wait,copy_blocks,256,args,0,owner));
    } else gpu_fetch_wait<<<1,256,0,owner>>>(slot,state,timeout_ns);
    cpu_queue_check(cudaGetLastError());
}
inline void GpuFetchQueue::wait_dma(int matrix,cudaStream_t owner,uint64_t timeout_ns) {
    cudaStreamCaptureStatus capture;
    cpu_queue_check(cudaStreamIsCapturing(owner,&capture));
    if(capture!=cudaStreamCaptureStatusNone)
        throw std::runtime_error("owner DMA must run between captured compute graphs");
    const auto boundary=std::chrono::steady_clock::now();
    // This owner-thread fence both acquires the published route/mask and waits
    // for older readers of the shared scratch. Never queue a GPU wait for the
    // service: allocation on another owner must remain able to make progress.
    cpu_queue_check(cudaEventRecord(dma_done,owner));
    cpu_queue_check(cudaEventSynchronize(dma_done));
    route_wait_us+=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-boundary).count();
    auto &slot=host[matrix];
    if(slot.failure.load(cuda::memory_order_acquire))return;
    const auto sequence=slot.request.load(cuda::memory_order_acquire);
    slot.started.store(sequence,cuda::memory_order_release);
    if(route_ready)route_ready();
    if(owner_admission) {
        if(!admit)throw std::runtime_error("GPU fetch owner admission callback missing");
        admit(matrix);
    }
    notify_progress();
    if(owner_empty)for(int m=0;m<3;++m) {
        const auto &pending=host[m];const auto request=pending.request.load(cuda::memory_order_acquire);
        if(request!=pending.reply.load(cuda::memory_order_acquire) && (cancelled.load() || pending.job.count)) {start_service();break;}
    }
    if(owner_empty && !cancelled.load() && !slot.job.count) {
        slot.error=cpu_ok;
        // Preserve RAM lease release and epoch validation without I/O work.
        try {execute(matrix,slot,*this);}catch(...) {slot.error=cpu_exception;}
        if(ordered_sources) {
            const int next=source_turn.load(std::memory_order_acquire);
            if(next<0 || next>=3 || source_order[next]!=matrix)throw std::runtime_error("empty fetch source order mismatch");
            source_turn.store(next+1,std::memory_order_release);
            notify_progress(); // a following already-admitted matrix can now start
        }
        if(slot.error)slot.failure.store(slot.error,cuda::memory_order_release);
        slot.reply.store(sequence,cuda::memory_order_release);
        slot.consumed.store(sequence,cuda::memory_order_release);++empty_requests;return;
    }
    const auto start=std::chrono::steady_clock::now();auto progress=start;
    if(dma_depth>1) {
        pump_dma(slot,owner,sequence,timeout_ns);
        dma_wait_us+=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-start).count();return;
    }
    for(;;) {
        const auto chunk=slot.chunk.load(cuda::memory_order_acquire);
        if(chunk!=slot.chunk_done.load(cuda::memory_order_relaxed)) {
            cpu_queue_check(cudaMemcpyAsync(slot.destination,slot.source,slot.bytes,cudaMemcpyHostToDevice,owner));
            ++dma_copies;dma_peak_inflight=std::max<uint64_t>(dma_peak_inflight,1);
            cpu_queue_check(cudaEventRecord(dma_done,owner));
            // No timeout may acknowledge a live DMA read or release its source.
            cpu_queue_check(cudaEventSynchronize(dma_done));
            slot.chunk_done.store(chunk,cuda::memory_order_release);
            progress=std::chrono::steady_clock::now();
        } else if(slot.reply.load(cuda::memory_order_acquire)==sequence &&
                  slot.chunk.load(cuda::memory_order_acquire)==slot.chunk_done.load(cuda::memory_order_relaxed)) {
            if(slot.error)slot.failure.store(slot.error,cuda::memory_order_release);
            break;
        } else {
            const auto now=std::chrono::steady_clock::now();
            if(now-progress>std::chrono::nanoseconds(timeout_ns) || now-start>std::chrono::nanoseconds(8*timeout_ns)) {
                slot.failure.store(cpu_timeout,cuda::memory_order_release);break;
            }
            std::this_thread::yield();
        }
    }
    slot.consumed.store(sequence,cuda::memory_order_release);
    dma_wait_us+=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-start).count();
}
inline void GpuFetchQueue::pump_dma(GpuFetchSlot &slot,cudaStream_t owner,uint32_t sequence,uint64_t timeout_ns) {
    uint32_t completed=slot.chunk_done.load(cuda::memory_order_relaxed),submitted=completed;
    const auto start=std::chrono::steady_clock::now();auto progress=start;
    for(;;) {
        const auto published=slot.chunk.load(cuda::memory_order_acquire);
        while(submitted<published) {
            const auto index=(++submitted)%dma_depth;const auto &chunk=slot.dma[index];
            cpu_queue_check(cudaMemcpyAsync(chunk.destination,chunk.source,chunk.bytes,cudaMemcpyHostToDevice,owner));
            cpu_queue_check(cudaEventRecord(dma_events[index],owner));++dma_copies;
            dma_peak_inflight=std::max<uint64_t>(dma_peak_inflight,submitted-completed);
        }
        bool advanced=false;
        while(completed<submitted) {
            const auto status=cudaEventQuery(dma_events[(completed+1)%dma_depth]);
            if(status==cudaErrorNotReady)break;
            cpu_queue_check(status);++completed;advanced=true;
            slot.chunk_done.store(completed,cuda::memory_order_release);
        }
        if(advanced)progress=std::chrono::steady_clock::now();
        if(completed==submitted && slot.reply.load(cuda::memory_order_acquire)==sequence &&
           completed==slot.chunk.load(cuda::memory_order_acquire)) {
            if(slot.error)slot.failure.store(slot.error,cuda::memory_order_release);
            break;
        }
        const auto now=std::chrono::steady_clock::now();
        if(now-progress>std::chrono::nanoseconds(timeout_ns) || now-start>std::chrono::nanoseconds(8*timeout_ns)) {
            slot.failure.store(cpu_timeout,cuda::memory_order_release);
            // Stop publication, but retain every submitted source until DMA
            // really completes. Never release memory merely because of timeout.
            cpu_queue_check(cudaStreamSynchronize(owner));
            slot.chunk_done.store(submitted,cuda::memory_order_release);break;
        }
        if(!advanced)std::this_thread::yield();
    }
    slot.consumed.store(sequence,cuda::memory_order_release);
}
}
