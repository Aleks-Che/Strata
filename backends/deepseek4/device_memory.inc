// CUDA under WDDM can report a per-process view that does not change when
// another process allocates VRAM. Use the driver's global NVML reading on
// Windows, matching the web monitor. Match PCI identity, not NVML/CUDA indices
// (CUDA_VISIBLE_DEVICES can reorder them). No NVML SDK or link dependency.
#include "../common/device_memory.hpp"
static bool strata_device_memory(int device,size_t &free,size_t &total) {
#ifdef _WIN32
    static StrataGlobalMemory global;
    return global.sample(device,free,total);
#else
    ggml_cuda_set_device(device);
    return cudaMemGetInfo(&free,&total)==cudaSuccess;
#endif
}
