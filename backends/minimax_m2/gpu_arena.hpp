#pragma once
#include "../hy3/gpu_arena.hpp"
namespace minimax_m2 {
// Hy3 supplies aligned size-class slots and immediate release of empty slabs.
// MiniMax additionally caps ALL backing bytes (including holes/class padding),
// and checks host commit for the whole new slab, not just its first matrix.
class GpuArena {
public:
    struct Memory {size_t free,total,commit_free;};
    using Probe=std::function<Memory()>;
private:
    size_t cap;
    Probe probe;
    hy3::GpuArena pool;
    hy3::GpuArena::Memory available() const {
        const auto m=probe();
        if(!m.total || m.free>m.total)throw std::runtime_error("invalid MiniMax arena memory sample");
        const auto used=pool.snapshot().reserved;
        const size_t room=cap-std::min(cap,used);
        const size_t commit=m.commit_free-std::min(m.commit_free,size_t(1)<<30);
        const size_t reserve=m.total/20+(256<<20); // same reserve as the slot allocator
        return {std::min({m.free,reserve+std::min(room,m.total),reserve+std::min(commit,m.total)}),m.total};
    }
public:
    GpuArena(size_t bytes,Probe reader):cap(bytes),probe(std::move(reader)),pool(bytes,[this]{return available();}) {}
    GpuArena(const GpuArena&)=delete;
    GpuArena&operator=(const GpuArena&)=delete;
    cudaError_t allocate(void **p,size_t n) {return pool.allocate(p,n);}
    cudaError_t release(void *p) {return pool.release(p);}
    auto snapshot() const {return pool.snapshot();}
    void reset_counters() {pool.reset_counters();}
};
} // namespace minimax_m2
