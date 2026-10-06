#pragma once
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#if defined(_MSC_VER) && defined(_M_X64)
#include <intrin.h>
#endif

namespace step35 {
// Only host_copy.cpp is compiled for AVX2; dispatch and all other code retain
// the original ISA baseline. Never call the specialized function before this.
inline bool host_copy_avx2_available() {
#if defined(_MSC_VER) && defined(_M_X64)
    int cpu[4];__cpuid(cpu,0);if(cpu[0]<7)return false;
    __cpuid(cpu,1);
    if(!(cpu[2]&(1<<27)) || !(cpu[2]&(1<<28)) || (_xgetbv(0)&6)!=6)return false;
    __cpuidex(cpu,7,0);return (cpu[1]&(1<<5))!=0;
#elif defined(__GNUC__) && defined(__x86_64__)
    return __builtin_cpu_supports("avx2");
#else
    return false;
#endif
}
void host_copy_avx2_cached(void * dst,const void * src,size_t bytes);

// One shared object per ring. Configuration is owning-thread-only AFTER drain;
// counters are atomic because readers can run while the owner takes a snapshot.
class HostCopyState {
    int mode=0;
    bool profile=false;
    std::atomic<uint64_t> copies{0},bytes{0},wall_ns{0},cycles{0},slow{0};
    static uint64_t thread_cycles() {
#ifdef _WIN32
        ULONG64 value=0;
        if(!QueryThreadCycleTime(GetCurrentThread(),&value))throw std::runtime_error("Step reader cycle sample failed");
        return uint64_t(value);
#else
        return 0; // No cross-platform conversion of cycles into CPU time.
#endif
    }
    void copy(void * dst,const void * src,size_t size) const {
        if(mode==1)host_copy_avx2_cached(dst,src,size);
        else std::memcpy(dst,src,size);
    }
public:
    struct Counters {uint64_t copies,bytes,wall_ns,cycles,slow;};
    void configure(int selected,bool profiling) {
        if(selected<0 || selected>1 || (selected==1 && !host_copy_avx2_available()))
            throw std::runtime_error("unsupported Step host copy mode");
        mode=selected;profile=profiling;
    }
    void operator()(void * dst,const void * src,size_t size) {
        if(!profile){copy(dst,src,size);return;}
        const auto before=thread_cycles();
        const auto start=std::chrono::steady_clock::now();
        copy(dst,src,size);
        const auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();
        const auto after=thread_cycles();
        copies.fetch_add(1,std::memory_order_relaxed);bytes.fetch_add(size,std::memory_order_relaxed);
        wall_ns.fetch_add(uint64_t(ns),std::memory_order_relaxed);cycles.fetch_add(after-before,std::memory_order_relaxed);
        if(ns>=500000)slow.fetch_add(1,std::memory_order_relaxed);
    }
    Counters snapshot() const {
        return {copies.load(),bytes.load(),wall_ns.load(),cycles.load(),slow.load()};
    }
};
}
