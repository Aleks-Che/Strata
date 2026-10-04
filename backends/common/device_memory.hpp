#pragma once
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif
#include <cuda_runtime.h>
#ifdef _WIN32
#include <windows.h>
#include <limits>
#include <map>
#include <mutex>

// WDDM needs the driver's global view, matched by PCI identity, not CUDA/NVML
// ordinal. No NVML SDK/link dependency; never fall back to per-process CUDA data.
class StrataGlobalMemory {
public:
    struct Memory {unsigned long long total,free,used;};
    using Init=int (*)();
    using Handle=int (*)(const char *,void **);
    using Read=int (*)(void *,Memory *);
    using Pci=int (*)(char *,int,int);
    struct Api {Init init=nullptr,shutdown=nullptr;Handle handle=nullptr;Read read=nullptr;Pci pci=nullptr;};
    enum class Stage {ready,library,symbols,init,pci,handle,read,invalid};
    struct Diagnostic {Stage stage;int code;};
private:
    HMODULE library=nullptr;
    Api api;
    bool initialized=false;
    std::map<int,void *> devices;
    mutable std::mutex mutex;
    Diagnostic last{Stage::symbols,0};
    void initialize() {
        if(!api.init || !api.shutdown || !api.handle || !api.read || !api.pci) {
            last={Stage::symbols,ERROR_PROC_NOT_FOUND};return;
        }
        const int error=api.init();
        initialized=error==0;last={initialized?Stage::ready:Stage::init,error};
    }
public:
    StrataGlobalMemory() {
        library=LoadLibraryExW(L"nvml.dll",nullptr,LOAD_LIBRARY_SEARCH_SYSTEM32);
        if(!library) {last={Stage::library,int(GetLastError())};return;}
        api.init=reinterpret_cast<Init>(GetProcAddress(library,"nvmlInit_v2"));
        api.shutdown=reinterpret_cast<Init>(GetProcAddress(library,"nvmlShutdown"));
        api.handle=reinterpret_cast<Handle>(GetProcAddress(library,"nvmlDeviceGetHandleByPciBusId_v2"));
        api.read=reinterpret_cast<Read>(GetProcAddress(library,"nvmlDeviceGetMemoryInfo"));
        api.pci=[](char *out,int bytes,int device)->int {return int(cudaDeviceGetPCIBusId(out,bytes,device));};
        initialize();
    }
    // Test seam: supplied symbols remain alive for this object's lifetime.
    explicit StrataGlobalMemory(Api functions):api(functions) {initialize();}
    StrataGlobalMemory(const StrataGlobalMemory&)=delete;
    StrataGlobalMemory &operator=(const StrataGlobalMemory&)=delete;
    ~StrataGlobalMemory() {
        if(initialized)api.shutdown();
        if(library)FreeLibrary(library);
    }
    Diagnostic diagnostic() const {std::lock_guard<std::mutex> lock(mutex);return last;}
    bool sample(int device,size_t &free,size_t &total) {
        std::lock_guard<std::mutex> lock(mutex);
        free=total=0; // Failure must never expose a previous successful sample.
        if(!initialized)return false;
        auto found=devices.find(device);
        void *gpu=found==devices.end()?nullptr:found->second;
        if(!gpu) {
            char pci[32]={};
            int error=api.pci(pci,sizeof(pci),device);
            if(error) {last={Stage::pci,error};return false;}
            error=api.handle(pci,&gpu);
            if(error || !gpu) {last={Stage::handle,error};return false;}
            devices.emplace(device,gpu);
        }
        Memory memory{};
        const int error=api.read(gpu,&memory);
        if(error) {devices.erase(device);last={Stage::read,error};return false;}
        if(!memory.total || memory.free>memory.total || memory.total>std::numeric_limits<size_t>::max()) {
            devices.erase(device);last={Stage::invalid,0};return false;
        }
        free=size_t(memory.free);total=size_t(memory.total);last={Stage::ready,0};return true;
    }
};
#endif
