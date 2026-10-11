#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>

// A bounded materialization batch, not direct matrix-kernel access to the cache.
// Source allocations remain owned until flush() has enqueued their reads.
struct StrataExpertCopyBatch {
    static constexpr unsigned capacity=32;
    struct Item { void *dst; const void *src; size_t bytes; };
    struct Packet { Item items[capacity]; } packet{};
    unsigned count=0;
    bool enabled=false;

    StrataExpertCopyBatch() {
        const char *v=std::getenv("STRATA_DS4_CACHE_COPY_BATCH");
        enabled=v && std::strcmp(v,"1")==0;
    }
    static bool aligned(const void *dst,const void *src,size_t bytes) {
        return ((uintptr_t(dst)|uintptr_t(src)|bytes)&15)==0;
    }
    bool append(void *dst,const void *src,size_t bytes) {
        if(!enabled || count==capacity || !aligned(dst,src,bytes))return false;
        if(count) {
            auto &previous=packet.items[count-1];
            const auto begin=uintptr_t(dst),old=uintptr_t(previous.dst);
            // The scheduler enumerates experts in increasing order. Preserve
            // last-writer semantics for their 512-byte MMQ padding; never run
            // overlapping stores in separate blocks of the same launch.
            if(begin<=old)return false;
            if(begin<old+previous.bytes)previous.bytes=begin-old;
        }
        packet.items[count++]={dst,src,bytes};return true;
    }
    size_t bytes() const {
        size_t result=0;for(unsigned i=0;i<count;++i)result+=packet.items[i].bytes;
        return result;
    }
    void flush(cudaStream_t stream);
};

static __global__ void strata_expert_copy_batch_kernel(StrataExpertCopyBatch::Packet packet) {
    const auto item=packet.items[blockIdx.y];
    const auto *src=static_cast<const uint4 *>(item.src);
    auto *dst=static_cast<uint4 *>(item.dst);
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<item.bytes/sizeof(uint4);
        i+=size_t(gridDim.x)*blockDim.x)dst[i]=src[i];
}

inline void StrataExpertCopyBatch::flush(cudaStream_t stream) {
    if(!count)return;
    size_t largest=0;for(unsigned i=0;i<count;++i)largest=std::max(largest,packet.items[i].bytes);
    const unsigned blocks=unsigned(std::min<size_t>(128,(largest+4095)/4096));
    strata_expert_copy_batch_kernel<<<dim3(blocks,count),256,0,stream>>>(packet);
    CUDA_CHECK(cudaGetLastError());count=0;
}
