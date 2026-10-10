#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <vector>

namespace strata_ds4 {
// Adapted from GLM's expert_slab.hpp. One pool per device, shared by target and
// draft; the caller serializes host access and finishes GPU reads before release.
// Slots never move. Empty blocks return to CUDA immediately. Partially occupied
// blocks remain charged in full to the device's VRAM policy.
class ExpertSlabAllocator {
    struct Block {
        void *data=nullptr;
        size_t stride=0,bytes=0;
        std::vector<size_t> requested,free;
    };
    int device;
    size_t slab_bytes;
    std::map<uintptr_t,std::unique_ptr<Block>> blocks;
    size_t reserved=0,requested=0;
    uint64_t block_allocations=0;
    cudaError_t device_check() const {
        int actual=-1;auto error=cudaGetDevice(&actual);
        return error!=cudaSuccess?error:actual==device?cudaSuccess:cudaErrorInvalidDevice;
    }
public:
    struct Status {size_t reserved,requested,blocks;uint64_t block_allocations;};
    explicit ExpertSlabAllocator(size_t bytes):slab_bytes(bytes) {
        if(bytes<256)throw std::invalid_argument("DeepSeek slab must hold at least 256 bytes");
        auto error=cudaGetDevice(&device);
        if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));
    }
    ~ExpertSlabAllocator() {
        int previous=device;cudaGetDevice(&previous);cudaSetDevice(device);
        for(auto &item:blocks)cudaFree(item.second->data);
        cudaSetDevice(previous);
    }
    ExpertSlabAllocator(const ExpertSlabAllocator&)=delete;
    ExpertSlabAllocator &operator=(const ExpertSlabAllocator&)=delete;
    Status status() const {return {reserved,requested,blocks.size(),block_allocations};}
    // Existing holes need no growth allowance. New blocks may be smaller than
    // slab_bytes, so a large target slab never forces growth beyond the policy.
    cudaError_t allocate(void **out,size_t bytes,size_t max_growth,bool *growth_denied=nullptr) {
        if(growth_denied)*growth_denied=false;
        if(!out)return cudaErrorInvalidValue;
        *out=nullptr;
        auto error=device_check();if(error!=cudaSuccess)return error;
        if(!bytes || bytes>std::numeric_limits<size_t>::max()-255)return cudaErrorInvalidValue;
        const size_t stride=(bytes+255)&~size_t(255);
        Block *selected=nullptr;
        for(auto &item:blocks) {
            auto &b=*item.second;
            if(b.stride==stride && !b.free.empty() && (!selected || b.free.size()<selected->free.size()))selected=&b;
        }
        if(!selected) {
            const size_t wanted=std::max(size_t(1),std::min(size_t(4096),slab_bytes/stride));
            const size_t slots=std::min(wanted,max_growth/stride),capacity=slots*stride;
            if(!slots) {if(growth_denied)*growth_denied=true;return cudaErrorMemoryAllocation;}
            auto b=std::make_unique<Block>();b->stride=stride;b->bytes=capacity;
            b->requested.resize(slots);b->free.reserve(slots);
            for(size_t i=slots;i>0;--i)b->free.push_back(i-1);
            error=cudaMalloc(&b->data,capacity);if(error!=cudaSuccess)return error;
            const auto address=reinterpret_cast<uintptr_t>(b->data);
            void *allocation=b->data;selected=b.get();
            try {blocks.emplace(address,std::move(b));}
            catch(...) {cudaFree(allocation);throw;}
            reserved+=capacity;++block_allocations;
        }
        const auto slot=selected->free.back();selected->free.pop_back();
        selected->requested[slot]=bytes;requested+=bytes;
        *out=static_cast<uint8_t *>(selected->data)+slot*stride;
        return cudaSuccess;
    }
    cudaError_t release(void *pointer) {
        auto error=device_check();if(error!=cudaSuccess)return error;
        const auto address=reinterpret_cast<uintptr_t>(pointer);
        auto it=blocks.upper_bound(address);
        if(it==blocks.begin())return cudaErrorInvalidValue;
        --it;auto &b=*it->second;
        const auto offset=address-it->first;
        if(offset>=b.bytes || offset%b.stride)return cudaErrorInvalidValue;
        const auto slot=offset/b.stride,bytes=b.requested[slot];
        if(!bytes)return cudaErrorInvalidValue;
        const bool empty=b.free.size()+1==b.requested.size();
        if(empty) {error=cudaFree(b.data);if(error!=cudaSuccess)return error;}
        requested-=bytes;
        if(empty) {reserved-=b.bytes;blocks.erase(it);}
        else {b.requested[slot]=0;b.free.push_back(slot);}
        return cudaSuccess;
    }
};
}
