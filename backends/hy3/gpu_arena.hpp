#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <vector>

namespace hy3 {
// Dense slots inside bounded CUDA slabs. Separate size classes avoid a large
// matrix needing several unrelated evictions to make a contiguous hole.
// The owner must finish all GPU consumers before release (ExpertCache contract).
// Empty slabs return to CUDA immediately; no retained empty allocation pool.
class GpuArena {
public:
    struct Memory {size_t free,total;};
    using Probe=std::function<Memory()>;
    struct Stats {size_t reserved=0,live=0,blocks=0;uint64_t allocations=0,frees=0,budget_rejects=0;};
private:
    struct Slab {
        void *base=nullptr;
        size_t bytes=0,stride=0,used=0;
        std::vector<size_t> free;
        std::vector<uint8_t> live;
    };
    std::map<uintptr_t,std::unique_ptr<Slab>> slabs;
    size_t cap;
    Probe probe;
    Stats stats;
    static constexpr size_t page=2<<20,block=64<<20;
    static void *take(Slab &s) {
        const auto index=s.free.back();s.free.pop_back();s.live[index]=1;++s.used;
        return static_cast<uint8_t*>(s.base)+index*s.stride;
    }
public:
    GpuArena(size_t limit,Probe read):cap(limit),probe(std::move(read)) {}
    ~GpuArena() {for(auto &p:slabs)cudaFree(p.second->base);}
    GpuArena(const GpuArena&)=delete;
    GpuArena&operator=(const GpuArena&)=delete;
    Stats snapshot() const {auto s=stats;s.blocks=slabs.size();return s;}
    void reset_counters() {stats.allocations=stats.frees=stats.budget_rejects=0;}
    cudaError_t allocate(void **out,size_t n) {
        *out=nullptr;
        if(!n || n>cap || n%65536)return cudaErrorInvalidValue;
        for(auto &p:slabs) {
            auto &s=*p.second;
            if(s.stride==n && !s.free.empty()) {
                *out=take(s);stats.live+=n;return cudaSuccess;
            }
        }
        // One global probe per new slab, not per matrix. Reserved physical
        // bytes include all empty slots and class padding in the memory guard.
        const size_t bytes=(std::max(n,std::min(cap,block))+page-1)/page*page;
        const auto m=probe();
        if(!m.total || m.free>m.total)throw std::runtime_error("invalid arena memory sample");
        const size_t reserve=m.total/20+(256<<20);
        if(m.free<reserve || bytes>m.free-reserve) {++stats.budget_rejects;return cudaErrorMemoryAllocation;}
        try {
            auto s=std::make_unique<Slab>();s->bytes=bytes;s->stride=n;
            const size_t count=bytes/n;
            s->free.reserve(count);s->live.assign(count,0);
            for(size_t i=count;i>0;--i)s->free.push_back(i-1);
            auto error=cudaMalloc(&s->base,bytes);
            if(error!=cudaSuccess)return error;
            void *base=s->base;
            try {
                auto inserted=slabs.emplace(reinterpret_cast<uintptr_t>(base),std::move(s));
                *out=take(*inserted.first->second);
            } catch(...) {cudaFree(base);throw;}
            stats.reserved+=bytes;stats.live+=n;++stats.allocations;
            return cudaSuccess;
        } catch(const std::bad_alloc&) {return cudaErrorMemoryAllocation;}
    }
    cudaError_t release(void *ptr) {
        const auto address=reinterpret_cast<uintptr_t>(ptr);
        auto it=slabs.upper_bound(address);
        if(it==slabs.begin())return cudaErrorInvalidValue;
        --it;auto &s=*it->second;
        const size_t offset=address-it->first,index=offset/s.stride;
        if(offset>=s.bytes || offset%s.stride || index>=s.live.size() || !s.live[index])
            return cudaErrorInvalidValue;
        if(s.used==1) {
            const auto error=cudaFree(s.base);if(error!=cudaSuccess)return error;
            stats.reserved-=s.bytes;stats.live-=s.stride;++stats.frees;slabs.erase(it);
        } else {
            // Capacity was reserved when the slab was made: release cannot
            // fail due to allocating CPU metadata while reclaiming VRAM.
            s.free.push_back(index);s.live[index]=0;--s.used;stats.live-=s.stride;
        }
        return cudaSuccess;
    }
};
} // namespace hy3
