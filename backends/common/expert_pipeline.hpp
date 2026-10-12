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
    struct Chunk { const uint8_t *source; size_t bytes; std::shared_ptr<strata_expert_file::Source> file; bool cacheable,native; };
    struct Slot {
        uint8_t *host=nullptr,*device=nullptr;
        cudaEvent_t ready=nullptr,used=nullptr;
        bool available=true,submitted=false,has_use=false,retaining=false;
        // Keep borrowed pinned RAM alive even if submission fails after enqueue.
        std::shared_ptr<const void> borrowed;
        size_t job=0;
        size_t bytes=0;
    };
public:
    static constexpr size_t slots=4;
    // Optional diagnostic observer, immutable for the pipeline's lifetime.
    // Called around each H2D submission on its actual stream, under copy_mutex.
    using CopyObserver=std::function<void(cudaStream_t,bool,size_t)>;
    // Optional backend-specific host copy. Must finish all writes before return;
    // the slot's existing ready/used ownership rules still apply. Default: memcpy.
    using HostCopy=std::function<void(void *,const void *,size_t)>;
    // Optional managed source cache. Called only for native-file jobs, on a
    // producer thread, with its own Request and the pipeline cancellation flag.
    using SourceRead=std::function<bool(const std::shared_ptr<strata_expert_file::Source> &,
        const void *,void *,size_t,const std::atomic<bool> &,strata_expert_file::Request &)>;
    // Optional cache for both mmap and native-file sources. The final booleans
    // retain the plan's decode admission hint and the ordinary read decision.
    using SourceCopy=std::function<bool(const std::shared_ptr<strata_expert_file::Source> &,
        const void *,void *,size_t,const std::atomic<bool> &,strata_expert_file::Request &,bool,bool)>;
    // Return immutable, pinned host bytes with shared ownership, or an empty
    // pointer for ordinary staging. Never perform I/O or wait for a future fill.
    using SourceBorrow=std::function<std::shared_ptr<const void>(
        const std::shared_ptr<strata_expert_file::Source> &,const void *,size_t,bool)>;
    // Optional synchronous retention of immutable, delivered host bytes. The
    // slot cannot be refilled until this returns. Called only for consumed
    // cacheable chunks explicitly selected by the consumer, never abandoned work.
    using SourceRetain=std::function<void(const std::shared_ptr<strata_expert_file::Source> &,
        const void *,const void *,size_t,const std::atomic<bool> &)>;
    struct Counters {
        uint64_t groups=0,chunks=0,unused=0,h2d_bytes=0,d2d_bytes=0;
        uint64_t read_us=0,wait_us=0,submit_us=0;
        uint64_t file_bytes=0,mmap_bytes=0,read_peak=0;
        uint64_t borrowed_chunks=0,borrowed_bytes=0;
        uint64_t retain_chunks=0,retain_bytes=0,retain_us=0;
        uint64_t retain_queued=0,retain_dropped=0,retain_cancelled=0,retain_pending_peak=0,retain_wait_us=0;
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
    std::atomic<bool> cancel_reads{true},cancel_retains{true};
    struct RetainJob {size_t slot=0;Chunk chunk{};};
    std::array<RetainJob,slots> retain_queue{};
    size_t retain_head=0,retain_tail=0,retain_count=0,retain_pending=0;
    int retain_limit=0;
    std::thread retainer;
    std::exception_ptr error;
    Counters totals;
    std::vector<std::thread> producers;
    CopyObserver copy_observer;
    HostCopy host_copy;
    SourceRead source_read;
    SourceCopy source_copy;
    SourceBorrow source_borrow;
    SourceRetain source_retain;

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
                slot.borrowed.reset(); // previous H2D has completed (or finish drained it)
                std::shared_ptr<const void> borrowed;
                if(!cancel_reads.load() && source_borrow && job.file)
                    borrowed=source_borrow(job.file,job.source,job.bytes,job.cacheable);
                bool file_read=!borrowed && job.file && job.native && (read_mode==1 || !strata_expert_file::resident(job.source,job.bytes));
                bool read_ok=!cancel_reads.load();
                if(read_ok && !borrowed) {
                    if(source_copy && job.file)read_ok=source_copy(job.file,job.source,slot.host,job.bytes,cancel_reads,request,job.cacheable,file_read);
                    else if(file_read)read_ok=source_read?source_read(job.file,job.source,slot.host,job.bytes,cancel_reads,request):
                        request.read(*job.file,job.source,slot.host,job.bytes,cancel_reads);
                    else if(host_copy)host_copy(slot.host,job.source,job.bytes);
                    else std::memcpy(slot.host,job.source,job.bytes);
                }
                uint64_t read=now_us()-started;
                lock.lock();
                totals.read_us+=read;totals.wait_us+=waited;totals.slot_wait_us+=waited;
                if(read_ok && !borrowed) (file_read?totals.file_bytes:totals.mmap_bytes)+=job.bytes;
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
                    slot.borrowed=std::move(borrowed);
                    if(copy_observer)copy_observer(copy,true,job.bytes);
                    check(cudaMemcpyAsync(slot.device,slot.borrowed?slot.borrowed.get():slot.host,job.bytes,cudaMemcpyHostToDevice,copy));
                    if(copy_observer)copy_observer(copy,false,job.bytes);
                    check(cudaEventRecord(slot.ready,copy));
                }
                uint64_t submitted=now_us()-started;
                lock.lock();slot.job=index;slot.bytes=job.bytes;slot.submitted=true;
                totals.wait_us+=device_wait;totals.slot_wait_us+=device_wait;
                totals.queued_bytes+=job.bytes;totals.queued_peak=std::max(totals.queued_peak,totals.queued_bytes);
                ++totals.chunks;totals.h2d_bytes+=job.bytes;totals.submit_us+=submitted;
                if(slot.borrowed){++totals.borrowed_chunks;totals.borrowed_bytes+=job.bytes;}
                --busy;claimed=false;totals.reader_owned_bytes-=claimed_bytes;cv.notify_all();
            }
        } catch(...) {
            std::lock_guard<std::mutex> lock(mutex);
            if(claimed) {--busy;totals.reader_owned_bytes-=claimed_bytes;}
            error=std::current_exception();active=false;cancel_reads.store(true);cv.notify_all();
        }
    }
    void retain_worker() noexcept {
        // Ring ownership, including the host source, lasts through the callback.
        // No additional payload allocation is needed for this bounded queue.
        try {check(cudaSetDevice(device));}
        catch(...) {
            std::lock_guard<std::mutex> lock(mutex);error=std::current_exception();
            active=false;cancel_reads.store(true);cancel_retains.store(true);cv.notify_all();
        }
        for(;;) {
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait(lock,[&]{return quit || retain_count;});
            if(!retain_count && quit)return;
            auto job=std::move(retain_queue[retain_head]);retain_queue[retain_head]={};
            retain_head=(retain_head+1)%slots;--retain_count;
            auto &slot=ring[job.slot];
            const bool run=!cancel_retains.load() && !error;
            lock.unlock();const auto started=now_us();std::exception_ptr failure;
            if(run)try {source_retain(job.chunk.file,job.chunk.source,slot.host,job.chunk.bytes,cancel_retains);}
            catch(...) {failure=std::current_exception();}
            lock.lock();
            if(run){++totals.retain_chunks;totals.retain_bytes+=job.chunk.bytes;totals.retain_us+=now_us()-started;}
            else ++totals.retain_cancelled;
            if(failure) {error=failure;active=false;cancel_reads.store(true);cancel_retains.store(true);}
            slot.retaining=false;slot.available=true;--retain_pending;cv.notify_all();
        }
    }
    void free_buffers() noexcept {
        cudaSetDevice(device);
        if(copy)cudaStreamSynchronize(copy);
        for(auto &slot:ring) {
            if(slot.has_use)cudaEventSynchronize(slot.used);
            slot.borrowed.reset();
            if(slot.ready)cudaEventDestroy(slot.ready);
            if(slot.used)cudaEventDestroy(slot.used);
            if(slot.host)cudaFreeHost(slot.host);
            if(slot.device)cudaFree(slot.device);
        }
        if(copy)cudaStreamDestroy(copy);
    }
public:
    StrataExpertPipeline(int gpu,size_t bytes,bool write_combined,int readers=2,int mode=0,CopyObserver observer={},int decode_readers=1,bool refill_early=false,HostCopy host_copy_fn={},SourceRead source_read_fn={},SourceCopy source_copy_fn={},SourceBorrow source_borrow_fn={},SourceRetain source_retain_fn={},int retain_queue_limit=0)
        :device(gpu),chunk_bytes(bytes),read_mode(mode),reader_limit(readers),decode_reader_limit(decode_readers),early_host_refill(refill_early),copy_observer(std::move(observer)),host_copy(std::move(host_copy_fn)),source_read(std::move(source_read_fn)),source_copy(std::move(source_copy_fn)),source_borrow(std::move(source_borrow_fn)),source_retain(std::move(source_retain_fn)) {
        retain_limit=retain_queue_limit;
        if(!bytes)throw std::runtime_error("expert pipeline requires pinned staging");
        if(bytes>std::numeric_limits<size_t>::max()/slots)throw std::runtime_error("expert staging capacity overflow");
        if(readers<1 || readers>int(slots) || decode_readers<1 || decode_readers>readers || mode<0 || mode>2)
            throw std::runtime_error("invalid expert reader configuration");
        if(retain_limit<0 || retain_limit>=int(slots) || (retain_limit && !source_retain))
            throw std::runtime_error("invalid expert retention queue");
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
            if(retain_limit)retainer=std::thread([this]{retain_worker();});
        } catch(...) {
            {std::lock_guard<std::mutex> lock(mutex);quit=true;cv.notify_all();}
            for(auto &thread:producers)if(thread.joinable())thread.join();
            if(retainer.joinable())retainer.join();
            free_buffers();throw;
        }
    }
    ~StrataExpertPipeline() {
        {std::lock_guard<std::mutex> lock(mutex);quit=true;active=false;cancel_reads.store(true);cancel_retains.store(true);cv.notify_all();}
        for(auto &thread:producers)if(thread.joinable())thread.join();
        if(retainer.joinable())retainer.join();
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
        active=false;cancel_reads.store(true);
        // Complete consumed work at a normal plan boundary. An abandoned suffix
        // or sticky failure cancels optional retention instead. A new plan may
        // never reuse the host slots or reset cancellation before this drains.
        if(consumed!=jobs.size() || error)cancel_retains.store(true);
        cv.notify_all();cv.wait(lock,[&]{return !busy;});
        const auto retain_started=now_us();const bool waiting=retain_pending!=0;
        cv.wait(lock,[&]{return !retain_pending;});
        if(waiting)totals.retain_wait_us+=now_us()-retain_started;
        cancel_retains.store(true);
        // Consumed slots are protected by their consumer events. A host-side
        // stream drain is needed only for an abandoned, uploaded suffix.
        bool abandoned=false;
        for(auto &slot:ring)abandoned|=slot.submitted;
        lock.unlock();auto status=abandoned?cudaStreamSynchronize(copy):cudaSuccess;lock.lock();
        for(auto &slot:ring) {
            if(slot.submitted) {
                ++totals.unused;totals.unused_bytes+=slot.bytes;totals.queued_bytes-=slot.bytes;
            }
            if(abandoned && status==cudaSuccess)slot.borrowed.reset();
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
            auto file=(native || source_copy || source_borrow || source_retain)?strata_expert_file::find(slice.data,slice.bytes):nullptr;
            for(size_t off=0;off<slice.bytes;off+=chunk_bytes)
                jobs.push_back({(const uint8_t *)slice.data+off,std::min(chunk_bytes,slice.bytes-off),file,slice.cacheable,native});
        }
        if(jobs.empty())return;
        ++totals.groups;cancel_reads.store(false);cancel_retains.store(false);active=true;cv.notify_all();
    }
    // False means the source was not in the bounded lookahead plan. The
    // caller cancels that plan and uses the original ordered transfer path.
    bool transfer(void *destination,const void *source,size_t bytes,cudaStream_t consumer,const CopyObserver &delivery_observer={},bool retain_source=false) {
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
            const auto job=jobs[consumed];size_t n=job.bytes;
            lock.unlock();
            // ready is recorded before publication; waiting on an unrecorded
            // CUDA event would otherwise be a no-op.
            check(cudaStreamWaitEvent(consumer,slot.ready,0));
            if(destination && delivery_observer)delivery_observer(consumer,true,n);
            if(destination)check(cudaMemcpyAsync((uint8_t *)destination+off,slot.device,n,cudaMemcpyDeviceToDevice,consumer));
            if(destination && delivery_observer)delivery_observer(consumer,false,n);
            check(cudaEventRecord(slot.used,consumer));
            lock.lock();slot.has_use=true; // protects queued consumer work even if retention throws
            if(destination && retain_source && source_retain && job.cacheable && job.file && !slot.borrowed) {
                if(retain_limit) {
                    if(retain_pending<size_t(retain_limit)) {
                        retain_queue[retain_tail]={consumed%slots,job};retain_tail=(retain_tail+1)%slots;
                        ++retain_count;++retain_pending;slot.retaining=true;++totals.retain_queued;
                        totals.retain_pending_peak=std::max<uint64_t>(totals.retain_pending_peak,retain_pending);
                    } else ++totals.retain_dropped; // optional cache fill must have a hard bound
                } else {
                    lock.unlock();const auto retain_started=now_us();
                    try {source_retain(job.file,job.source,slot.host,n,cancel_reads);}
                    catch(...) {
                        lock.lock();error=std::current_exception();active=false;cancel_reads.store(true);cv.notify_all();
                        throw;
                    }
                    lock.lock();++totals.retain_chunks;totals.retain_bytes+=n;totals.retain_us+=now_us()-retain_started;
                }
            }
            if(destination)totals.d2d_bytes+=n;
            totals.queued_bytes-=n;slot.bytes=0;
            slot.has_use=true;slot.submitted=false;slot.available=!slot.retaining;
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
