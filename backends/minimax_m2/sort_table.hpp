#pragma once
#include "ggml.h"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace minimax_m2 {
// Own the H2D source until its event completes. Device table/scratch ownership
// stays with the upstream stream-ordered CUDA pool. No graph capture is used.
class SortTableRing {
public:
    static constexpr size_t slots=4, slot_bytes=4096, capacity=slots*slot_bytes;
    struct Counters {uint64_t copies=0,bytes=0,reuse_waits=0,drain_waits=0,peak_pending=0;};
private:
    struct Slot {cudaEvent_t done=nullptr;cudaStream_t stream=nullptr;bool pending=false,recorded=false;};
    std::array<Slot,slots> ring_{};
    char *host_=nullptr;
    size_t next_=0;
    int device_=-1;
    Counters counts_{};
    static void check(cudaError_t rc) {
        if(rc!=cudaSuccess)throw std::runtime_error(std::string("MiniMax sort table: ")+cudaGetErrorString(rc));
    }
    void retire(Slot &s,bool reuse) {
        if(!s.pending)return;
        const auto rc=s.recorded?cudaEventQuery(s.done):cudaErrorNotReady;
        if(rc==cudaErrorNotReady) {
            ++(reuse?counts_.reuse_waits:counts_.drain_waits);
            // If event recording failed after enqueue, retain the source and
            // use its stream for cleanup; never free an in-flight host buffer.
            check(s.recorded?cudaEventSynchronize(s.done):cudaStreamSynchronize(s.stream));
        } else check(rc);
        s.pending=false;s.recorded=false;s.stream=nullptr;
    }
public:
    SortTableRing()=default;
    SortTableRing(const SortTableRing &)=delete;
    SortTableRing &operator=(const SortTableRing &)=delete;
    ~SortTableRing() noexcept {
        try {close();}catch(...) {GGML_ABORT("MiniMax sort table cleanup failed");}
    }
    bool copy(void *dst,const void *src,size_t bytes,cudaStream_t stream) {
        if(!dst || !src || !bytes || bytes>slot_bytes)return false;
        int current;check(cudaGetDevice(&current));
        if(host_ && current!=device_)return false;
        if(!host_) {
            device_=current;
            check(cudaMallocHost(reinterpret_cast<void **>(&host_),capacity));
            try {for(auto &s:ring_)check(cudaEventCreateWithFlags(&s.done,cudaEventDisableTiming));}
            catch(...) {close();throw;}
        }
        auto &s=ring_[next_];retire(s,true);
        auto *source=host_+next_*slot_bytes;
        std::memcpy(source,src,bytes);
        check(cudaMemcpyAsync(dst,source,bytes,cudaMemcpyHostToDevice,stream));
        s.pending=true;s.recorded=false;s.stream=stream;
        check(cudaEventRecord(s.done,stream));s.recorded=true;
        next_=(next_+1)%slots;++counts_.copies;counts_.bytes+=bytes;
        counts_.peak_pending=std::max<uint64_t>(counts_.peak_pending,pending());
        return true;
    }
    void drain() {for(auto &s:ring_)retire(s,false);}
    void reset_counters() {drain();counts_={};}
    void close() {
        if(!host_)return;
        int previous;check(cudaGetDevice(&previous));
        if(previous!=device_)check(cudaSetDevice(device_));
        drain();
        for(auto &s:ring_)if(s.done) {check(cudaEventDestroy(s.done));s.done=nullptr;}
        check(cudaFreeHost(host_));host_=nullptr;next_=0;device_=-1;
        if(previous>=0)check(cudaSetDevice(previous));
    }
    uint64_t pending() const {uint64_t n=0;for(const auto &s:ring_)n+=s.pending;return n;}
    uint64_t pinned_bytes() const {return host_?capacity:0;}
    const Counters &counters() const {return counts_;}
};
}

// MiniMax-private ggml-base bridge, called from the strict Q4_K/Q6_K dispatch.
bool strata_mm27_sort_table_copy(void *dst,const void *src,size_t bytes,cudaStream_t stream);
