#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>

// Coalesce the host state IO descriptors, keeping their serialized order.
// No format or tensor layout changes. File/device IO retain their native paths.
namespace minimax_m2_state {
constexpr size_t scratch_limit = 16u << 20;
struct Stats { size_t gets=0, sets=0, groups=0, scratch_bytes=0; };
// Thread-local diagnostic counters also let the fixture prove path coverage.
inline thread_local Stats totals;

template<bool Restore, class Infos, class Get, class Set, class Size>
Stats flush(const Infos & infos, bool enabled, Get get, Set set, Size size,
            size_t capacity=scratch_limit) {
    Stats stats;
    capacity=std::min(capacity,scratch_limit);
    std::unique_ptr<uint8_t[]> scratch;
    bool allocation_failed=false;
    for(size_t i=0;i<infos.size();) {
        const auto & first=infos[i];
        size_t end=i+1, span=first.size, packed=first.size;
        const size_t tensor_bytes=size(first.tensor);
        // Subtractions guard overflow, including malicious or unusual strides.
        if(enabled && first.size && first.offset<=tensor_bytes &&
                first.size<=tensor_bytes-first.offset && first.size<=capacity) {
            while(end<infos.size()) {
                const auto & next=infos[end];
                if(next.tensor!=first.tensor || !next.size || next.offset<first.offset ||
                        next.offset-first.offset<span || next.offset>tensor_bytes ||
                        next.size>tensor_bytes-next.offset || next.offset-first.offset>capacity ||
                        next.size>capacity-(next.offset-first.offset) || next.ptr!=first.ptr+packed) break;
                span=next.offset-first.offset+next.size;
                packed+=next.size; // nonoverlapping ranges within capacity
                ++end;
            }
        }
        // Sparse unrelated ranges should not cause disproportionate transfers.
        const bool group=end>i+1 && packed>=span/4;
        if(group && !scratch && !allocation_failed) {
            scratch.reset(new(std::nothrow) uint8_t[capacity]);
            allocation_failed=!scratch;
            if(scratch)stats.scratch_bytes=capacity;
        }
        if(group && scratch) {
            ++stats.groups;
            if constexpr(Restore) {
                // Preserve all gaps and other sequences; dense ranges need no read.
                if(packed!=span) {get(first.tensor,scratch.get(),first.offset,span);++stats.gets;}
                for(size_t j=i;j<end;++j) {
                    const auto & r=infos[j];
                    std::memcpy(scratch.get()+r.offset-first.offset,r.ptr,r.size);
                }
                set(first.tensor,scratch.get(),first.offset,span);++stats.sets;
            } else {
                get(first.tensor,scratch.get(),first.offset,span);++stats.gets;
                for(size_t j=i;j<end;++j) {
                    const auto & w=infos[j];
                    std::memcpy(w.ptr,scratch.get()+w.offset-first.offset,w.size);
                }
            }
            i=end;
        } else {
            // Consume a rejected plan once, rather than rescanning sparse ranges.
            for(;i<end;++i) {
                const auto & r=infos[i];
                if constexpr(Restore) {set(r.tensor,r.ptr,r.offset,r.size);++stats.sets;}
                else {get(r.tensor,r.ptr,r.offset,r.size);++stats.gets;}
            }
        }
    }
    totals.gets+=stats.gets;totals.sets+=stats.sets;totals.groups+=stats.groups;
    totals.scratch_bytes=std::max(totals.scratch_bytes,stats.scratch_bytes);
    return stats;
}
} // namespace minimax_m2_state
