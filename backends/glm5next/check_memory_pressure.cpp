// Explicit opt-in Windows/NVML integration check. Synthetic weights, no model.
#include "expert_memory.hpp"
#include <chrono>
#include <cstdio>
#include <iostream>
#include <thread>

using namespace strata_glm;
static constexpr size_t mib=1ULL<<20,weight_bytes=32*mib,cap=2*weight_bytes,pressure_bytes=256*mib;
static void require(bool ok,const char *message) {if(!ok)throw std::runtime_error(message);}
static void check(cudaError_t error) {if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));}
struct Allocation {
    void *data=nullptr;
    ~Allocation() {if(data)cudaFree(data);}
    void release() {if(data) {check(cudaFree(data));data=nullptr;}}
};
static void ready() {
    check(cudaSetDevice(0));check(cudaFree(nullptr));
    char pci[32]{};check(cudaDeviceGetPCIBusId(pci,sizeof(pci),0));
    int runtime=0,driver=0;check(cudaRuntimeGetVersion(&runtime));check(cudaDriverGetVersion(&driver));
    std::printf("{\"phase\":\"ready\",\"pci\":\"%s\",\"cuda_runtime\":%d,\"cuda_driver\":%d}\n",pci,runtime,driver);
}
static void holder() {
    Allocation allocation;ready();std::string command;
    while(std::getline(std::cin,command)) {
        if(command=="ALLOC") {
            require(!allocation.data,"holder already allocated");
            check(cudaMalloc(&allocation.data,pressure_bytes));
            check(cudaMemset(allocation.data,0x71,pressure_bytes));check(cudaDeviceSynchronize());
            std::printf("{\"phase\":\"allocated\",\"bytes\":%llu}\n",(unsigned long long)pressure_bytes);
        }else if(command=="FREE") {
            require(allocation.data!=nullptr,"holder has no allocation");allocation.release();
            std::puts("{\"phase\":\"freed\"}");
        }else if(command=="QUIT")return;
        else throw std::runtime_error("unexpected holder command");
    }
}
static ExpertKey key(bool mtp) {
    return {"pressure-fixture",1,mtp?Branch::mtp:Branch::main,mtp?45:3,0,Projection::gate,
            "IQ2_S",4096,2048,"fixture",0,weight_bytes};
}
static void verify(ExpertCache::Lease &lease,unsigned char value) {
    require(bool(lease),"unexpected cache bypass");check(cudaDeviceSynchronize());
    std::vector<unsigned char> bytes(weight_bytes);
    check(cudaMemcpy(bytes.data(),lease.data(),bytes.size(),cudaMemcpyDeviceToHost));
    require(std::all_of(bytes.begin(),bytes.end(),[&](unsigned char b){return b==value;}),"cache bytes changed under pressure");
}
static void load(ExpertCache &cache,const ExpertKey &key,unsigned char value) {
    auto lease=cache.get(key,std::make_shared<int>(1),nullptr,[&](void *dest,size_t bytes,cudaStream_t stream) {
        check(cudaMemsetAsync(dest,value,bytes,stream));
    });
    verify(lease,value);lease.release();check(cudaDeviceSynchronize());
}
static void report(const char *phase,ExpertMemoryController &controller,ExpertCache &cache,const StrataVramPolicy &policy) {
    const auto s=controller.status();const auto c=cache.counters();
    std::printf("{\"phase\":\"%s\",\"free_bytes\":%llu,\"total_bytes\":%llu,\"target_bytes\":%llu,"
                "\"resident_bytes\":%llu,\"deferred_bytes\":%llu,\"sample_valid\":%d,\"trim_complete\":%d,"
                "\"hits\":%llu,\"misses\":%llu,\"evictions\":%llu,\"device_target_mib\":%llu}\n",phase,
                (unsigned long long)s.free,(unsigned long long)s.total,(unsigned long long)s.target,
                (unsigned long long)cache.resident_bytes(),(unsigned long long)s.deferred,int(s.sample_valid),int(s.trim_complete),
                (unsigned long long)c.hits,(unsigned long long)c.misses,(unsigned long long)c.evictions,
                (unsigned long long)policy.target_mib);
}
template<class Predicate> static void await_sample(ExpertMemoryController &controller,Predicate matches) {
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(8);
    do {
        const auto status=controller.refresh();
        require(status.sample_valid,"global memory unavailable: external pressure check requires NVML");
        if(matches(status))return;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }while(std::chrono::steady_clock::now()<deadline);
    throw std::runtime_error("global memory did not reach expected pressure/recovery target within 8 seconds");
}
static void controller(bool frequency) {
    ready();std::string command;
    require(bool(std::getline(std::cin,command)) && command=="START","expected START after holder context initialization");
    auto probe=make_global_memory_probe();size_t free=0,total=0;
    require(probe(0,free,total),"global memory unavailable: external pressure check requires NVML");
    require(free>=pressure_bytes+cap+256*mib,"insufficient free VRAM for bounded pressure fixture");
    // Set a usage ceiling above current fixed/context usage; actual free VRAM
    // remains plentiful. The holder crosses this ceiling without causing OOM.
    StrataVramPolicy policy;policy.mode=2;policy.reserve_mib=128;
    policy.target_mib=(total-free+mib-1)/mib+128;
    ExpertCache cache(cap,{frequency,100000,32});ExpertMemoryController control(cache,cap,policy,probe);
    require(control.refresh().target==cap,"initial fixture budget does not fit");
    const auto main=key(false),mtp=key(true);
    load(cache,main,0x31);load(cache,mtp,0x42);
    auto pins=cache.protect_plan({main});
    await_sample(control,[](auto s){return s.target==cap && s.resident==cap;});
    report("warm",control,cache,policy);
    require(bool(std::getline(std::cin,command)) && command=="PRESSURE","expected PRESSURE");
    await_sample(control,[](auto s){return s.target==0;});
    require(cache.resident(main) && !cache.resident(mtp) && cache.resident_bytes()==weight_bytes &&
            control.status().deferred==weight_bytes && !control.status().trim_complete,
            "pressure failed to evict idle MTP or preserve main pin");
    {
        auto hit=cache.get(main,nullptr,nullptr,[](void *,size_t,cudaStream_t){throw std::runtime_error("pinned hit reuploaded");});
        verify(hit,0x31);hit.release();check(cudaDeviceSynchronize());
    }
    report("protected",control,cache,policy);
    require(bool(std::getline(std::cin,command)) && command=="TRIM","expected TRIM");
    pins.release();control.refresh();
    require(control.status().target==0 && control.status().trim_complete && cache.resident_bytes()==0,
            "released pin did not trim under pressure");
    report("trimmed",control,cache,policy);
    require(bool(std::getline(std::cin,command)) && command=="RECOVER","expected RECOVER");
    await_sample(control,[](auto s){return s.target==cap;});
    load(cache,main,0x53);load(cache,mtp,0x64);control.refresh();
    require(control.status().target==cap && cache.resident_bytes()==cap,"cache did not recover after holder release");
    report("recovered",control,cache,policy);
    require(bool(std::getline(std::cin,command)) && command=="QUIT","expected QUIT");
}
int main(int argc,char **argv) {
    std::setvbuf(stdout,nullptr,_IONBF,0);
    try {
        require(argc==2,"expected --holder, --controller-lru or --controller-frequency");
        const std::string mode=argv[1];
        if(mode=="--holder")holder();
        else if(mode=="--controller-lru" || mode=="--controller-frequency")controller(mode=="--controller-frequency");
        else throw std::runtime_error("invalid role");
        return 0;
    }catch(const std::exception &error) {std::fprintf(stderr,"FAIL: %s\n",error.what());return 1;}
}
