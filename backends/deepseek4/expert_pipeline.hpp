#pragma once
#include "expert_transfer.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// SSD/mmap -> pinned RAM -> independent H2D stream -> consumer D2D/compute.
// One persistent producer per backend; memory is bounded independently of the
// model and prompt size. The producer never writes scheduler-owned tensors.
class StrataExpertPipeline {
    struct Chunk { const uint8_t *source; size_t bytes; };
    struct Slot {
        uint8_t *host=nullptr,*device=nullptr;
        cudaEvent_t ready=nullptr,used=nullptr;
        bool available=true,submitted=false,has_use=false;
        size_t job=0;
    };
public:
    static constexpr size_t slots=4;
    struct Counters {
        uint64_t groups=0,chunks=0,unused=0,h2d_bytes=0,d2d_bytes=0;
        uint64_t read_us=0,wait_us=0,submit_us=0;
    };
private:
    int device;
    size_t chunk_bytes;
    cudaStream_t copy=nullptr;
    std::array<Slot,slots> ring{};
    std::vector<Chunk> jobs;
    size_t consumed=0;
    std::mutex mutex;
    std::condition_variable cv;
    bool active=false,busy=false,quit=false;
    std::exception_ptr error;
    Counters totals;
    std::thread producer;

    static void check(cudaError_t status) {
        if(status!=cudaSuccess)throw std::runtime_error(std::string("expert pipeline: ")+cudaGetErrorString(status));
    }
    static uint64_t now_us() {
        return uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
    }
    void produce() noexcept {
        try {
            check(cudaSetDevice(device));
            std::unique_lock<std::mutex> lock(mutex);
            while(!quit) {
                cv.wait(lock,[&]{return quit || active;});
                if(quit)break;
                busy=true;cv.notify_all();
                for(size_t index=0;index<jobs.size() && active;++index) {
                    Slot &slot=ring[index%slots];
                    cv.wait(lock,[&]{return !active || slot.available;});
                    if(!active)break;
                    slot.available=false;
                    Chunk job=jobs[index];
                    bool has_use=slot.has_use;
                    lock.unlock();
                    uint64_t started=now_us();
                    // A previous consumer may still be reading this device
                    // slot. Its event also implies the old H2D has completed.
                    if(has_use)check(cudaEventSynchronize(slot.used));
                    uint64_t waited=now_us()-started;started=now_us();
                    std::memcpy(slot.host,job.source,job.bytes);
                    uint64_t read=now_us()-started;started=now_us();
                    check(cudaMemcpyAsync(slot.device,slot.host,job.bytes,cudaMemcpyHostToDevice,copy));
                    check(cudaEventRecord(slot.ready,copy));
                    uint64_t submitted=now_us()-started;
                    lock.lock();
                    slot.job=index;slot.submitted=true;
                    ++totals.chunks;totals.h2d_bytes+=job.bytes;
                    totals.read_us+=read;totals.wait_us+=waited;totals.submit_us+=submitted;
                    cv.notify_all();
                }
                active=false;busy=false;cv.notify_all();
            }
        } catch(...) {
            std::lock_guard<std::mutex> lock(mutex);
            error=std::current_exception();active=false;busy=false;cv.notify_all();
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
    StrataExpertPipeline(int gpu,size_t bytes,bool write_combined):device(gpu),chunk_bytes(bytes) {
        if(!bytes)throw std::runtime_error("expert pipeline requires pinned staging");
        try {
            check(cudaSetDevice(device));
            check(cudaStreamCreateWithFlags(&copy,cudaStreamNonBlocking));
            for(auto &slot:ring) {
                check(cudaHostAlloc((void **)&slot.host,bytes,write_combined?cudaHostAllocWriteCombined:cudaHostAllocDefault));
                check(cudaMalloc((void **)&slot.device,bytes));
                check(cudaEventCreateWithFlags(&slot.ready,cudaEventDisableTiming));
                check(cudaEventCreateWithFlags(&slot.used,cudaEventDisableTiming|cudaEventBlockingSync));
            }
            producer=std::thread([this]{produce();});
        } catch(...) {free_buffers();throw;}
    }
    ~StrataExpertPipeline() {
        {std::lock_guard<std::mutex> lock(mutex);quit=true;active=false;cv.notify_all();}
        if(producer.joinable())producer.join();
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
        active=false;cv.notify_all();
        cv.wait(lock,[&]{return !busy;});
        lock.unlock();auto status=cudaStreamSynchronize(copy);lock.lock();
        for(auto &slot:ring) {
            if(slot.submitted)++totals.unused;
            slot.available=true;slot.submitted=false;
        }
        jobs.clear();consumed=0;
        if(propagate) {if(error)std::rethrow_exception(error);check(status);}
    }
    void start(const std::vector<StrataExpertSlice>& slices) {
        finish();
        std::lock_guard<std::mutex> lock(mutex);
        for(auto &slice:slices)for(size_t off=0;off<slice.bytes;off+=chunk_bytes)
            jobs.push_back({(const uint8_t *)slice.data+off,std::min(chunk_bytes,slice.bytes-off)});
        if(jobs.empty())return;
        ++totals.groups;active=true;cv.notify_all();
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
            totals.wait_us+=now_us()-started;
            size_t n=jobs[consumed].bytes;
            lock.unlock();
            // ready is recorded before publication; waiting on an unrecorded
            // CUDA event would otherwise be a no-op.
            check(cudaStreamWaitEvent(consumer,slot.ready,0));
            if(destination)check(cudaMemcpyAsync((uint8_t *)destination+off,slot.device,n,cudaMemcpyDeviceToDevice,consumer));
            check(cudaEventRecord(slot.used,consumer));
            lock.lock();
            if(destination)totals.d2d_bytes+=n;
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
