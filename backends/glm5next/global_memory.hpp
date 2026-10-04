#pragma once
#include "../common/device_memory.hpp"
#include <functional>
#include <memory>

namespace strata_glm {
// The returned closure owns the reader; no borrowed NVML/library lifetime.
inline std::function<bool(int,size_t &,size_t &)> make_global_memory_probe() {
#ifdef _WIN32
    auto reader=std::make_shared<StrataGlobalMemory>();
    return [reader](int device,size_t &free,size_t &total) {return reader->sample(device,free,total);};
#else
    // CUDA's device-wide view on non-WDDM platforms. The controller already
    // checks the current device; do not change another caller's CUDA context.
    return [](int device,size_t &free,size_t &total) {
        free=total=0;int current=-1;
        if(cudaGetDevice(&current)!=cudaSuccess || current!=device)return false;
        size_t f=0,t=0;
        if(cudaMemGetInfo(&f,&t)!=cudaSuccess || !t || f>t)return false;
        free=f;total=t;return true;
    };
#endif
}
}
