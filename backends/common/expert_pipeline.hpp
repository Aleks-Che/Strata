#pragma once
#include "expert_slice.hpp"
#include "expert_file.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// SSD/mmap -> pinned RAM -> independent H2D stream -> consumer D2D/compute.
// A bounded reader queue per backend; memory is bounded independently of the
// model and prompt size. The producer never writes scheduler-owned tensors.
class StrataExpertPipeline {
    struct Chunk { const uint8_t *source; size_t bytes; std::shared_ptr<strata_expert_file::Source> file; };
    struct Slot {
        uint8_t *host=nullptr,*device=nullptr;
        cudaEvent_t ready=nullptr,used=nullptr;
        bool available=true,submitted=false,has_use=false;
        size_t job=0;
        size_t bytes=0;
    };
public:
    static constexpr size_t slots=4;
    // Optional diagnostic observer, immutable for the pipeline's lifetime.
    // Called around each H2D submission on its actual stream, under copy_mutex.
    using CopyObserver=std::function<void(cudaStream_t,bool,size_t)>;
    struct Counters {
        uint64_t groups=0,chunks=0,unused=0,h2d_bytes=0,d2d_bytes=0;
        uint64_t read_us=0,wait_us=0,submit_us=0;
        uint64_t file_bytes=0,mmap_bytes=0,read_peak=0;
        // CPU wall-time sums, not CUDA execution durations. wait_us retains
        // its existing meaning and equals these two components added together.
        uint64_t slot_wait_us=0,consumer_wait_us=0;
        // Fixed allocated capacities; exclude CUDA events/allocator overhead.
        uint64_t pinned_bytes=0,device_ring_bytes=0;
        // Logical payload bytes owned by readers (including slot/event waits)
        // or published for consumers, not physical resident-memory estimates.
        uint64_t reader_owned_bytes=0,reader_owned_peak=0;
        uint64_t queued_bytes=0,queued_peak=0,unused_bytes=0;
    };
private:
    int device;
    size_t chunk_bytes;
    cudaStream_t copy=nullptr;
    std::array<Slot,slots> ring{};
    std::vector<Chunk> jobs;
    size_t consumed=0,next_job=0,busy=0;
    int read_mode,reader_limit,decode_reader_limit,active_readers=0;
    std::mutex mutex,copy_mutex;
    std::condition_variable cv;
    bool active=false,quit=false;
    bool early_host_refill=false;
    std::atomic<bool> cancel_reads{true};
    std::exception_ptr error;
    Counters totals;
    std::vector<std::thread> producers;
    CopyObserver copy_observer;

    static void check(cudaError_t status) {
        if(status!=cudaSuccess)throw std::runtime_error(std::string("expert pipeline: ")+cudaGetErrorString(status));
    }
    static uint64_t now_us() {
        return uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
    }
    void produce(int reader) noexcept {
        bool claimed=false;
        size_t claimed_bytes=0;
        try {
            check(cudaSetDevice(device));
            strata_expert_file::Request request;
            for(;;) {
                std::unique_lock<std::mutex> lock(mutex);
                cv.wait(lock,[&]{return quit || (active && reader<active_readers && next_job<jobs.size() && ring[next_job%slots].available);});
                if(quit)break;
                size_t index=next_job++;
                Slot &slot=ring[index%slots];
                slot.available=false;
                Chunk job=jobs[index];
                bool has_use=slot.has_use;
                ++busy;claimed=true;totals.read_peak=std::max<uint64_t>(totals.read_peak,busy);
                claimed_bytes=job.bytes;totals.reader_owned_bytes+=job.bytes;
                totals.reader_owned_peak=std::max(totals.reader_owned_peak,totals.reader_owned_bytes);
                cv.notify_all();lock.unlock();
                uint64_t started=now_us();
                // Host reuse needs only the previous H2D. Optionally prepare
                // the next payload while the GPU still consumes the device
                // slot. Device reuse remains protected by used below.
                if(has_use)check(cudaEventSynchronize(early_host_refill?slot.ready:slot.used));
                uint64_t waited=now_us()-started;started=now_us();
                bool file_read=job.file && (read_mode==1 || !strata_expert_file::resident(job.source,job.bytes));
                bool read_ok=!cancel_reads.load();
                if(read_ok) {
                    if(file_read)read_ok=request.read(*job.file,job.source,slot.host,job.bytes,cancel_reads);
                    else std::memcpy(slot.host,job.source,job.bytes);
                }
                uint64_t read=now_us()-started;
                lock.lock();
                totals.read_us+=read;totals.wait_us+=waited;totals.slot_wait_us+=waited;
                if(read_ok) (file_read?totals.file_bytes:totals.mmap_bytes)+=job.bytes;
                if(!active || !read_ok) {
                    slot.available=true;--busy;claimed=false;totals.reader_owned_bytes-=claimed_bytes;
                    cv.notify_all();continue;
                }
                lock.unlock();started=now_us();
                if(has_use && early_host_refill)check(cudaEventSynchronize(slot.used));
                const uint64_t device_wait=early_host_refill?now_us()-started:0;
                started=now_us();
                {
                    // Keep copy/event pairs together while readers finish in
                    // arbitrary order. Publication follows event recording.
                    std::lock_guard<std::mutex> submit(copy_mutex);
                    if(copy_observer)copy_observer(copy,true,job.bytes);
                    check(cudaMemcpyAsync(slot.device,slot.host,job.bytes,cudaMemcpyHostToDevice,copy));
                    if(copy_observer)copy_observer(copy,false,job.bytes);
                    check(cudaEventRecord(slot.ready,copy));
                }
                uint64_t submitted=now_us()-started;
                lock.lock();slot.job=index;slot.bytes=job.bytes;slot.submitted=true;
                totals.wait_us+=device_wait;totals.slot_wait_us+=device_wait;
                totals.queued_bytes+=job.bytes;totals.queued_peak=std::max(totals.queued_peak,totals.queued_bytes);
                ++totals.chunks;totals.h2d_bytes+=job.bytes;totals.submit_us+=submitted;
                --busy;claimed=false;totals.reader_owned_bytes-=claimed_bytes;cv.notify_all();
            }
        } catch(...) {
            std::lock_guard<std::mutex> lock(mutex);
            if(claimed) {--busy;totals.reader_owned_bytes-=claimed_bytes;}
            error=std::current_exception();active=false;cancel_reads.store(true);cv.notify_all();
        }
    }
    void free_buffers() noexcept {
        cudaSetDevice(device);
        if(copy)cudaStreamSynchronize(copy);
        for(auto &slot:ring) {
            if(slot.has_use)cudaEventSynchronize(slot.used);
            if(slot.ready)cudaEventDestroy(slot.ready);
            if(slot.used)cudaEventDestroy(slot.used);
            if(slot.host)cudaFreeHost(slot.host);
            if(slot.device)cudaFree(slot.device);
        }
        if(copy)cudaStreamDestroy(copy);
    }
public:
    StrataExpertPipeline(int gpu,size_t bytes,bool write_combined,int readers=2,int mode=0,CopyObserver observer={},int decode_readers=1,bool refill_early=false)
        :device(gpu),chunk_bytes(bytes),read_mode(mode),reader_limit(readers),decode_reader_limit(decode_readers),early_host_refill(refill_early),copy_observer(std::move(observer)) {
        if(!bytes)throw std::runtime_error("expert pipeline requires pinned staging");
        if(bytes>std::numeric_limits<size_t>::max()/slots)throw std::runtime_error("expert staging capacity overflow");
        if(readers<1 || readers>int(slots) || decode_readers<1 || decode_readers>readers || mode<0 || mode>2)
            throw std::runtime_error("invalid expert reader configuration");
        try {
            check(cudaSetDevice(device));
            check(cudaStreamCreateWithFlags(&copy,cudaStreamNonBlocking));
            for(auto &slot:ring) {
                check(cudaHostAlloc((void **)&slot.host,bytes,write_combined?cudaHostAllocWriteCombined:cudaHostAllocDefault));
                check(cudaMalloc((void **)&slot.device,bytes));
                check(cudaEventCreateWithFlags(&slot.ready,cudaEventDisableTiming));
                check(cudaEventCreateWithFlags(&slot.used,cudaEventDisableTiming|cudaEventBlockingSync));
            }
            totals.pinned_bytes=totals.device_ring_bytes=bytes*slots;
            for(int i=0;i<readers;++i)producers.emplace_back([this,i]{produce(i);});
        } catch(...) {
            {std::lock_guard<std::mutex> lock(mutex);quit=true;cv.notify_all();}
            for(auto &thread:producers)if(thread.joinable())thread.join();
            free_buffers();throw;
        }
    }
    ~StrataExpertPipeline() {
        {std::lock_guard<std::mutex> lock(mutex);quit=true;active=false;cancel_reads.store(true);cv.notify_all();}
        for(auto &thread:producers)if(thread.joinable())thread.join();
        free_buffers();
    }
    StrataExpertPipeline(const StrataExpertPipeline&)=delete;
    // Safe even if a graph stopped before consuming the planned suffix. The
    // producer can only be waiting on events of already released slots.
    void finish(bool propagate=true) {
        std::unique_lock<std::mutex> lock(mutex);
        if(jobs.empty() && !busy) {
            if(propagate && error)std::rethrow_exception(error);
            return;
        }
        active=false;cancel_reads.store(true);cv.notify_all();
        cv.wait(lock,[&]{return !busy;});
        // Consumed slots are protected by their consumer events. A host-side
        // stream drain is needed only for an abandoned, uploaded suffix.
        bool abandoned=false;
        for(auto &slot:ring)abandoned|=slot.submitted;
        lock.unlock();auto status=abandoned?cudaStreamSynchronize(copy):cudaSuccess;lock.lock();
        for(auto &slot:ring) {
            if(slot.submitted) {
                ++totals.unused;totals.unused_bytes+=slot.bytes;totals.queued_bytes-=slot.bytes;
            }
            slot.bytes=0;
            slot.available=true;slot.submitted=false;
        }
        jobs.clear();consumed=next_job=0;
        if(propagate) {if(error)std::rethrow_exception(error);check(status);}
    }
    void start(const std::vector<StrataExpertSlice>& slices) {
        finish();
        std::lock_guard<std::mutex> lock(mutex);
        // Keep the historical one-worker decode default. Backends can opt in
        // to more readers after measuring their matrix sizes and host memory.
        bool decode=std::all_of(slices.begin(),slices.end(),[](const auto &s){return s.cacheable;});
        active_readers=decode && read_mode!=1?decode_reader_limit:reader_limit;
        for(auto &slice:slices) {
            // Native cached reads do not fault the mmap view into the working
            // set. Using residency alone would keep warm decode on ReadFile
            // forever. Auto queues cold prefill slices and leaves decode on
            // mmap; explicit file mode remains available for measurement.
            bool native=read_mode==1 || (read_mode==2 && !slice.cacheable);
            auto file=native?strata_expert_file::find(slice.data,slice.bytes):nullptr;
            for(size_t off=0;off<slice.bytes;off+=chunk_bytes)
                jobs.push_back({(const uint8_t *)slice.data+off,std::min(chunk_bytes,slice.bytes-off),file});
        }
        if(jobs.empty())return;
        ++totals.groups;cancel_reads.store(false);active=true;cv.notify_all();
    }
    // False means the source was not in the bounded lookahead plan. The
    // caller cancels that plan and uses the original ordered transfer path.
    bool transfer(void *destination,const void *source,size_t bytes,cudaStream_t consumer) {
        std::unique_lock<std::mutex> lock(mutex);
        if(error)std::rethrow_exception(error);
        size_t count=0;
        for(size_t off=0;off<bytes;off+=chunk_bytes,++count) {
            if(consumed+count>=jobs.size())return false;
            const auto &job=jobs[consumed+count];
            if(job.source!=(const uint8_t *)source+off || job.bytes!=std::min(chunk_bytes,bytes-off))return false;
        }
        for(size_t off=0;off<bytes;) {
            Slot &slot=ring[consumed%slots];
            uint64_t started=now_us();
            cv.wait(lock,[&]{return error || (slot.submitted && slot.job==consumed);});
            if(error)std::rethrow_exception(error);
            const auto waited=now_us()-started;totals.wait_us+=waited;totals.consumer_wait_us+=waited;
            size_t n=jobs[consumed].bytes;
            lock.unlock();
            // ready is recorded before publication; waiting on an unrecorded
            // CUDA event would otherwise be a no-op.
            check(cudaStreamWaitEvent(consumer,slot.ready,0));
            if(destination)check(cudaMemcpyAsync((uint8_t *)destination+off,slot.device,n,cudaMemcpyDeviceToDevice,consumer));
            check(cudaEventRecord(slot.used,consumer));
            lock.lock();
            if(destination)totals.d2d_bytes+=n;
            totals.queued_bytes-=n;slot.bytes=0;
            slot.has_use=true;slot.submitted=false;slot.available=true;
            ++consumed;off+=n;cv.notify_all();
        }
        return true;
    }
    Counters counters() {std::lock_guard<std::mutex> lock(mutex);return totals;}
    // Diagnostic: completed H2D chunks waiting for a consumer, not just queued
    // copies. Used by the overlap test with a deliberately blocked consumer.
    size_t ready_chunks() {
        std::lock_guard<std::mutex> lock(mutex);
        if(error)std::rethrow_exception(error);
        size_t n=0;
        for(auto &slot:ring)if(slot.submitted) {
            auto status=cudaEventQuery(slot.ready);
            if(status==cudaSuccess)++n;
            else if(status!=cudaErrorNotReady)check(status);
        }
        return n;
    }
};
