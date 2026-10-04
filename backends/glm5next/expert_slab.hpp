#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <vector>

namespace strata_glm {
// Single host owner/device. Equal-size slots share a CUDA allocation; pointers
// never move. ExpertCache calls release only after ready/consumer events finish.
// An empty slab is returned immediately, so no unused whole blocks are hidden
// from the global memory controller. Partial-slab waste remains charged by NVML.
class ExpertSlabAllocator {
public:
    // Return the allowed physical growth, at most the requested block size.
    // The last slab may be smaller than the target; zero closes new growth.
    using GrowthLimit=std::function<size_t(size_t)>;
    struct Status {
        size_t reserved=0,requested=0,slot_bytes=0,blocks=0;
        uint64_t allocations=0,block_allocations=0,reuses=0,growth_denied=0,partial_blocks=0;
    };
private:
    struct Block {
        void *data=nullptr;
        size_t stride=0,bytes=0;
        std::vector<size_t> requested,free;
    };
    int device;
    size_t slab_bytes;
    GrowthLimit growth_limit;
    std::map<uintptr_t,std::unique_ptr<Block>> blocks;
    Status counts;
    cudaError_t device_check() const {
        int actual=-1;auto error=cudaGetDevice(&actual);
        return error!=cudaSuccess?error:actual==device?cudaSuccess:cudaErrorInvalidDevice;
    }
public:
    explicit ExpertSlabAllocator(size_t bytes,GrowthLimit limit={})
        :slab_bytes(bytes),growth_limit(std::move(limit)) {
        if(bytes<256)throw std::invalid_argument("GLM slab must hold at least 256 bytes");
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
    Status status() const {auto s=counts;s.blocks=blocks.size();return s;}
    cudaError_t allocate(void **out,size_t bytes) {
        if(!out)return cudaErrorInvalidValue;
        *out=nullptr;
        auto error=device_check();if(error!=cudaSuccess)return error;
        if(!bytes || bytes>std::numeric_limits<size_t>::max()-255)return cudaErrorInvalidValue;
        const size_t stride=(bytes+255)&~size_t(255);
        Block *selected=nullptr;
        // Fill the fullest compatible block first, leaving other blocks easier
        // to release under pressure. Slots may differ in their final padding.
        for(auto &item:blocks) {
            auto &b=*item.second;
            if(b.stride==stride && !b.free.empty() && (!selected || b.free.size()<selected->free.size()))selected=&b;
        }
        if(selected)++counts.reuses;
        else {
            const size_t wanted_slots=std::max(size_t(1),std::min(size_t(4096),slab_bytes/stride));
            const size_t wanted=wanted_slots*stride;
            const size_t allowed=growth_limit?std::min(wanted,growth_limit(wanted)):wanted;
            const size_t slots=allowed/stride,capacity=slots*stride;
            if(!slots) {++counts.growth_denied;return cudaErrorMemoryAllocation;}
            auto b=std::make_unique<Block>();b->stride=stride;b->bytes=capacity;
            b->requested.resize(slots);b->free.reserve(slots);
            for(size_t i=slots;i>0;--i)b->free.push_back(i-1);
            error=cudaMalloc(&b->data,capacity);if(error!=cudaSuccess)return error;
            const auto address=reinterpret_cast<uintptr_t>(b->data);
            void *allocation=b->data;selected=b.get();
            try {blocks.emplace(address,std::move(b));}
            catch(...) {cudaFree(allocation);throw;}
            counts.reserved+=capacity;++counts.block_allocations;
            if(slots<wanted_slots)++counts.partial_blocks;
        }
        const auto slot=selected->free.back();selected->free.pop_back();
        selected->requested[slot]=bytes;
        counts.requested+=bytes;counts.slot_bytes+=stride;++counts.allocations;
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
        counts.requested-=bytes;counts.slot_bytes-=b.stride;
        if(empty) {counts.reserved-=b.bytes;blocks.erase(it);}
        else {b.requested[slot]=0;b.free.push_back(slot);} // Capacity reserved at creation.
        return cudaSuccess;
    }
};
}
