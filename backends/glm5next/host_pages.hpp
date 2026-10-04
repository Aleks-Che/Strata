#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#endif

namespace strata_glm {
struct HostResidency {uint64_t bytes=0,samples=0,valid_samples=0,estimated_resident_bytes=0;bool valid=true;};
// One probe per <=1 MiB of payload. Does not dereference/fault the source pages.
// The weighted result is an estimate, not a count of all 4 KiB pages.
inline HostResidency sample_host_residency(const void *ptr,size_t bytes) {
    HostResidency result;result.bytes=bytes;
    if(bytes>UINTPTR_MAX-uintptr_t(ptr))throw std::invalid_argument("host sample overflow");
#ifdef _WIN32
    constexpr size_t stride=1<<20,batch=256;
    PSAPI_WORKING_SET_EX_INFORMATION probes[batch]{};
    size_t weights[batch]{};
    for(size_t offset=0;offset<bytes;) {
        size_t count=0;
        while(count<batch && offset<bytes) {
            const auto n=std::min(stride,bytes-offset);
            probes[count].VirtualAddress=reinterpret_cast<void *>(uintptr_t(ptr)+offset+n/2);
            probes[count].VirtualAttributes.Flags=0;weights[count++]=n;offset+=n;
        }
        if(!QueryWorkingSetEx(GetCurrentProcess(),probes,DWORD(count*sizeof(probes[0])))) {result.valid=false;return result;}
        result.samples+=count;
        for(size_t i=0;i<count;++i)if(probes[i].VirtualAttributes.Valid) {
            ++result.valid_samples;result.estimated_resident_bytes+=weights[i];
        }
    }
#else
    result.valid=false;
#endif
    return result;
}
// Required payload plus the currently resident non-expert memory. Residency is
// sampled, so reserve a separate margin before allowing a full bounded scan.
inline uint64_t required_host_working_set(uint64_t working_set,const HostResidency &cpu,const HostResidency &gpu) {
    if(cpu.estimated_resident_bytes>UINT64_MAX-gpu.estimated_resident_bytes)throw std::invalid_argument("host residency overflow");
    const auto other=working_set-std::min(working_set,cpu.estimated_resident_bytes+gpu.estimated_resident_bytes);
    if(cpu.bytes>UINT64_MAX-other)throw std::invalid_argument("host budget overflow");
    return cpu.bytes+other;
}
inline bool host_scan_fits(uint64_t required,uint64_t limit) {
    constexpr uint64_t margin=256ULL<<20;
    return required<=limit && limit-required>=margin;
}

}
