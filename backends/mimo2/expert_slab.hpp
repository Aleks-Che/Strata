#pragma once
// Single owner, stable slots. The cache fences all readers/fills before release.
// Physical blocks, including unused slots and page padding, remain charged.
#include <algorithm>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <vector>

namespace mimo2 {
class ExpertSlab {
public:
    using Allocate=std::function<void *(size_t)>;
    using Free=std::function<void(void *)>;
    struct Stats {size_t reserved=0,slots=0,blocks=0;uint64_t allocations=0,reuses=0,denied=0,oom=0;};
private:
    struct Block {
        void *data;size_t stride,bytes;std::vector<size_t> free;std::vector<uint8_t> live;
        Block *next=nullptr,*previous=nullptr,**head=nullptr;
    };
    size_t target,page;Allocate allocate;Free release;
    std::map<uintptr_t,std::unique_ptr<Block>> blocks;
    // Intrusive lists avoid allocations while releasing a slot or handling OOM.
    // Only blocks with holes are searched, within the requested size class.
    std::map<size_t,Block *> partial;
    Stats counts;
    static size_t rounded(size_t bytes,size_t alignment) {
        if(!bytes || bytes>SIZE_MAX-(alignment-1))throw std::runtime_error("invalid slab size");
        return ((bytes+alignment-1)/alignment)*alignment;
    }
    Block *available(size_t stride) const {
        Block *result=nullptr;
        const auto found=partial.find(stride);if(found==partial.end())return nullptr;
        // Pack fullest compatible blocks first; no relocation of live pointers.
        for(auto *b=found->second;b;b=b->next) {
            if(!result || b->free.size()<result->free.size() ||
               (b->free.size()==result->free.size() && reinterpret_cast<uintptr_t>(b->data)<reinterpret_cast<uintptr_t>(result->data)))result=b;
        }
        return result;
    }
    static void link(Block *b) {
        b->previous=nullptr;b->next=*b->head;
        if(b->next)b->next->previous=b;
        *b->head=b;
    }
    static void unlink(Block *b) {
        if(b->previous)b->previous->next=b->next;else *b->head=b->next;
        if(b->next)b->next->previous=b->previous;
        b->previous=b->next=nullptr;
    }
public:
    ExpertSlab(size_t target,size_t page,Allocate allocate,Free release)
        :target(target),page(page),allocate(std::move(allocate)),release(std::move(release)) {
        if(page<256 || page%256 || target<page || target%page)throw std::runtime_error("invalid slab geometry");
    }
    ~ExpertSlab() {for(const auto &b:blocks)release(b.second->data);}
    ExpertSlab(const ExpertSlab&)=delete;ExpertSlab &operator=(const ExpertSlab&)=delete;
    Stats stats() const {auto out=counts;out.blocks=blocks.size();return out;}
    void reset_stats() {counts.allocations=counts.reuses=counts.denied=counts.oom=0;}
    size_t reserved() const {return counts.reserved;}
    size_t growth(size_t bytes) const {
        if(!bytes)return 0;
        const size_t stride=rounded(bytes,256);
        return available(stride)?0:rounded(stride,page);
    }
    void *get(size_t bytes,size_t physical_limit) {
        const size_t stride=rounded(bytes,256);
        if(counts.reserved>physical_limit) {++counts.denied;return nullptr;}
        auto *b=available(stride);
        if(b)++counts.reuses;
        else {
            const size_t remaining=physical_limit-counts.reserved;
            const size_t allowed=(std::min(std::max(target,rounded(stride,page)),remaining)/page)*page;
            const size_t slots=std::min(size_t(4096),allowed/stride);
            if(!slots) {++counts.denied;return nullptr;}
            const size_t capacity=rounded(slots*stride,page);
            auto block=std::make_unique<Block>();block->stride=stride;block->bytes=capacity;
            block->head=&partial.try_emplace(stride,nullptr).first->second;
            block->live.resize(slots,0);block->free.reserve(slots);
            for(size_t i=slots;i>0;--i)block->free.push_back(i-1);
            block->data=allocate(capacity);
            if(!block->data) {++counts.oom;return nullptr;}
            const auto address=reinterpret_cast<uintptr_t>(block->data);auto *data=block->data;b=block.get();
            try {blocks.emplace(address,std::move(block));}catch(...) {release(data);throw;}
            link(b);
            counts.reserved+=capacity;++counts.allocations;
        }
        const size_t index=b->free.back();b->free.pop_back();b->live[index]=1;
        if(b->free.empty())unlink(b);
        counts.slots+=stride;
        return static_cast<uint8_t *>(b->data)+index*stride;
    }
    void put(void *pointer) {
        const auto address=reinterpret_cast<uintptr_t>(pointer);auto it=blocks.upper_bound(address);
        if(it==blocks.begin())throw std::runtime_error("foreign slab pointer");
        --it;auto &b=*it->second;const size_t offset=address-it->first,index=offset/b.stride;
        if(offset%b.stride || index>=b.live.size() || !b.live[index])throw std::runtime_error("invalid/double slab release");
        const bool full=b.free.empty();b.live[index]=0;b.free.push_back(index);counts.slots-=b.stride;
        if(full)link(&b);
        if(b.free.size()==b.live.size()) {unlink(&b);release(b.data);counts.reserved-=b.bytes;blocks.erase(it);}
    }
};
} // namespace mimo2
