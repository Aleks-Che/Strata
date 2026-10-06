#pragma once
// Adapted from Step's measured cacheable AVX2 staging copy. Keep dispatch at
// the baseline ISA; only host_copy.cpp is compiled with AVX2 enabled.
#include <cstddef>
#if defined(_MSC_VER) && defined(_M_X64)
#include <intrin.h>
#endif
namespace strata_ds4_host_copy {
inline bool available() {
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
void copy(void *destination,const void *source,size_t bytes);
}
