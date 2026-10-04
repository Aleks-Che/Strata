#ifdef STRATA_TEST_DEEPSEEK_MEMORY
#include "../deepseek4/device_memory.inc"
#else
#include "../common/device_memory.hpp"
#endif
#include "expert_memory.hpp"
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

static void require(bool ok,const char *message) {if(!ok)throw std::runtime_error(message);}
static void cuda_ok(cudaError_t error) {if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));}
using Reader=StrataGlobalMemory;
struct Fixture {
    int init_error=0,pci_error=0,handle_error=0,read_error=0;
    int inits=0,shutdowns=0,pci_calls=0,handle_calls=0,reads=0;
    bool null_handle=false;
    Reader::Memory memory{1024,512,512};
    std::string bus;
    void *expected=nullptr;
};
static Fixture *fixture=nullptr;
static int initialize() {++fixture->inits;return fixture->init_error;}
static int shutdown() {++fixture->shutdowns;return 0;}
static int pci(char *out,int bytes,int device) {
    ++fixture->pci_calls;
    if(fixture->pci_error)return fixture->pci_error;
    std::snprintf(out,size_t(bytes),"0000:%02x:00.0",device+3);return 0;
}
static int handle(const char *bus,void **out) {
    ++fixture->handle_calls;fixture->bus=bus;
    if(fixture->handle_error)return fixture->handle_error;
    fixture->expected=std::strcmp(bus,"0000:0a:00.0")==0?reinterpret_cast<void *>(77):reinterpret_cast<void *>(88);
    *out=fixture->null_handle?nullptr:fixture->expected;return 0;
}
static int read(void *gpu,Reader::Memory *memory) {
    ++fixture->reads;
    // The CUDA device 7 must use the handle resolved from PCI bus 0a, not 7.
    require(gpu==reinterpret_cast<void *>(77) || gpu==reinterpret_cast<void *>(88),"ordinal used as NVML handle");
    *memory=fixture->memory;
    if(gpu==reinterpret_cast<void *>(88))memory->free/=2; // Distinguish the two fake GPUs.
    return fixture->read_error;
}
static Reader::Api api() {return {initialize,shutdown,handle,read,pci};}

static void test_mapping_and_retry() {
    Fixture f;fixture=&f;
    {
        Reader reader(api());size_t free=123,total=456;
        require(reader.sample(7,free,total) && free==512 && total==1024 && f.bus=="0000:0a:00.0",
                "PCI identity mapping/bytes failed");
        require(reader.sample(7,free,total) && f.pci_calls==1 && f.handle_calls==1,"handle not cached");
        require(reader.sample(0,free,total) && free==256 && f.pci_calls==2 && f.bus=="0000:03:00.0","device identities mixed");
        f.read_error=999;
        require(!reader.sample(7,free,total) && free==0 && total==0 &&
                reader.diagnostic().stage==Reader::Stage::read && reader.diagnostic().code==999,"read failure returned stale bytes");
        f.read_error=0;
        require(reader.sample(7,free,total) && f.pci_calls==3,"failed handle was not resolved again");
        std::vector<std::thread> workers;
        for(int i=0;i<8;++i)workers.emplace_back([&] {
            for(int n=0;n<50;++n) {
                size_t a=0,b=0;require(reader.sample(7,a,b) && a==512 && b==1024,"concurrent sample");
            }
        });
        for(auto &worker:workers)worker.join();
        require(f.pci_calls==3 && f.reads==405,"concurrent reader state/counters");
    }
    require(f.inits==1 && f.shutdowns==1,"NVML lifetime not balanced");
    std::puts("PASS: PCI mapping, cached handles, rebind after failure, no stale bytes, 400 concurrent samples and shutdown");
}
static void test_errors() {
    Fixture f;fixture=&f;
    for(int missing=0;missing<5;++missing) {
        auto symbols=api();
        if(missing==0)symbols.init=nullptr;if(missing==1)symbols.shutdown=nullptr;
        if(missing==2)symbols.handle=nullptr;if(missing==3)symbols.read=nullptr;if(missing==4)symbols.pci=nullptr;
        Reader reader(symbols);size_t free=1,total=2;
        require(!reader.sample(7,free,total) && free==0 && total==0 &&
                reader.diagnostic().stage==Reader::Stage::symbols,"missing symbols accepted");
    }
    require(f.inits==0 && f.shutdowns==0,"partial API initialized or shut down NVML");
    f.init_error=4;
    {Reader reader(api());size_t a=1,b=2;require(!reader.sample(7,a,b) && a==0 && b==0 &&
        reader.diagnostic().stage==Reader::Stage::init && reader.diagnostic().code==4,"init failure ignored");}
    require(f.shutdowns==0,"failed initialization shut down another owner");f.init_error=0;
    {
        Reader reader(api());size_t free=1,total=2;
        f.pci_error=10;
        require(!reader.sample(7,free,total) && reader.diagnostic().stage==Reader::Stage::pci,"PCI error ignored");
        f.pci_error=0;f.handle_error=6;
        require(!reader.sample(7,free,total) && reader.diagnostic().stage==Reader::Stage::handle,"handle error ignored");
        f.handle_error=0;f.null_handle=true;
        require(!reader.sample(7,free,total),"null NVML handle accepted");f.null_handle=false;
        for(auto memory:std::vector<Reader::Memory>{{0,0,0},{1024,1025,0},{1024,~0ULL,0}}) {
            f.memory=memory;
            require(!reader.sample(7,free,total) && free==0 && total==0 &&
                    reader.diagnostic().stage==Reader::Stage::invalid,"invalid/NVML unsupported memory accepted");
        }
        f.memory={1024,0,1024};require(reader.sample(7,free,total) && free==0 && total==1024,"full device rejected");
    }
    require(f.shutdowns==1,"successful reader lifetime leaked");
    std::puts("PASS: missing symbols, init/PCI/handle/read failures, unsupported memory sentinels and recovery");
}
static void test_live() {
    cudaDeviceProp gpu{};cuda_ok(cudaGetDeviceProperties(&gpu,0));
    Reader reader;size_t free=123,total=456;const bool valid=reader.sample(0,free,total);
    const auto diagnostic=reader.diagnostic();
    require(valid?(total>0 && free<=total):(free==0 && total==0),"live reader returned inconsistent data");
    std::printf("LIVE_GLOBAL_MEMORY gpu=%s available=%d stage=%d code=%d free=%llu total=%llu\n",gpu.name,int(valid),
                int(diagnostic.stage),diagnostic.code,(unsigned long long)free,(unsigned long long)total);
#ifdef STRATA_TEST_DEEPSEEK_MEMORY
    free=total=123;const bool compat=strata_device_memory(0,free,total);
    require(compat?(total>0 && free<=total):(free==0 && total==0),"DeepSeek compatibility sample invalid");
    std::printf("LIVE_DEEPSEEK_MEMORY available=%d free=%llu total=%llu\n",int(compat),
                (unsigned long long)free,(unsigned long long)total);
#endif
    // Exercise the production factory/owning closure through the default controller.
    strata_glm::ExpertCache cache(1<<20);StrataVramPolicy policy;policy.reserve_mib=128;
    strata_glm::ExpertMemoryController controller(cache,1<<20,policy);
    const auto state=controller.refresh();
    require(state.samples==1 && state.target<=1<<20,"live controller cap/status");
    const strata_glm::ExpertKey key{"live-probe-fixture",1,strata_glm::Branch::main,3,0,
        strata_glm::Projection::gate,"IQ2_S",4096,2048,"fixture",0,64};
    bool uploaded=false;
    auto lease=cache.get(key,std::make_shared<int>(1),nullptr,[&](void *dest,size_t bytes,cudaStream_t stream) {
        uploaded=true;cuda_ok(cudaMemsetAsync(dest,0x5a,bytes,stream));
    });
    if(!state.sample_valid) {
        require(!lease && !uploaded && cache.counters().paused_bypasses==1,"unavailable live sample admitted cache bytes");
    }else if(state.target>=64) {
        require(bool(lease) && uploaded,"available live sample did not admit within cap");
        cuda_ok(cudaDeviceSynchronize());unsigned char bytes[64]{};
        cuda_ok(cudaMemcpy(bytes,lease.data(),64,cudaMemcpyDeviceToHost));
        for(auto byte:bytes)require(byte==0x5a,"live cache upload bytes mismatch");
    }else require(!lease && !uploaded,"small live budget exceeded");
    std::printf("LIVE_GLM_CONTROLLER valid=%d target=%llu resident=%llu paused_bypasses=%llu\n",int(state.sample_valid),
                (unsigned long long)state.target,(unsigned long long)cache.resident_bytes(),
                (unsigned long long)cache.counters().paused_bypasses);
}
int main() {
    try {test_mapping_and_retry();test_errors();test_live();return 0;}
    catch(const std::exception &error) {std::fprintf(stderr,"FAIL: %s\n",error.what());return 1;}
}
