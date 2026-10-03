// CUDA under WDDM can report a per-process view that does not change when
// another process allocates VRAM. Use the driver's global NVML reading on
// Windows, matching the web monitor. Match PCI identity, not NVML/CUDA indices
// (CUDA_VISIBLE_DEVICES can reorder them). No NVML SDK or link dependency.
#ifdef _WIN32
#include <windows.h>
struct StrataGlobalMemory {
    struct Memory { unsigned long long total, free, used; };
    using Init=int (*)();
    using Handle=int (*)(const char *,void **);
    using Read=int (*)(void *,Memory *);
    HMODULE library=nullptr;
    Init shutdown=nullptr;
    Handle handle=nullptr;
    Read read=nullptr;
    bool initialized=false;
    std::map<int,void *> devices;
    StrataGlobalMemory() {
        library=LoadLibraryExW(L"nvml.dll",nullptr,LOAD_LIBRARY_SEARCH_SYSTEM32);
        if(!library)return;
        auto init=reinterpret_cast<Init>(GetProcAddress(library,"nvmlInit_v2"));
        shutdown=reinterpret_cast<Init>(GetProcAddress(library,"nvmlShutdown"));
        handle=reinterpret_cast<Handle>(GetProcAddress(library,"nvmlDeviceGetHandleByPciBusId_v2"));
        read=reinterpret_cast<Read>(GetProcAddress(library,"nvmlDeviceGetMemoryInfo"));
        initialized=init && shutdown && handle && read && init()==0;
    }
    ~StrataGlobalMemory() {
        if(initialized)shutdown();
        if(library)FreeLibrary(library);
    }
    bool sample(int device,size_t &free,size_t &total) {
        if(!initialized)return false;
        auto &gpu=devices[device];
        if(!gpu) {
            char pci[32]={};
            if(cudaDeviceGetPCIBusId(pci,sizeof(pci),device)!=cudaSuccess || handle(pci,&gpu)!=0)return false;
        }
        Memory memory{};
        if(read(gpu,&memory)!=0 || !memory.total || memory.free>memory.total)return false;
        free=size_t(memory.free);total=size_t(memory.total);return true;
    }
};
#endif
static bool strata_device_memory(int device,size_t &free,size_t &total) {
#ifdef _WIN32
    static StrataGlobalMemory global;
    return global.sample(device,free,total);
#else
    ggml_cuda_set_device(device);
    return cudaMemGetInfo(&free,&total)==cudaSuccess;
#endif
}
