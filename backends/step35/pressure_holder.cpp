// Opt-in diagnostic process. Never launched by the engine or normal CTest.
#include "expert_cache.hpp"
#include "nlohmann/json.hpp"
#include <iostream>

using namespace step35;
using json=nlohmann::ordered_json;
int main() {
    void * gpu=nullptr;std::vector<uint8_t> ram;
    const size_t gpu_bytes=128<<20,ram_bytes=size_t(2)<<30;
    auto release=[&]{if(gpu){cudaFree(gpu);gpu=nullptr;}std::vector<uint8_t>().swap(ram);};
    auto emit=[&](const char * phase){
        const auto m=memory_sample();char pci[32]{};cuda_check(cudaDeviceGetPCIBusId(pci,sizeof(pci),0));
        std::cout<<json({{"phase",phase},{"pci",pci},{"gpu_bytes",gpu?gpu_bytes:0},{"ram_bytes",ram.size()},
            {"gpu_free",m.gpu_free},{"gpu_total",m.gpu_total},{"ram_free",m.ram_free},{"ram_total",m.ram_total}}).dump()<<'\n'<<std::flush;
    };
    try {
        cuda_check(cudaSetDevice(0));cuda_check(cudaFree(nullptr));emit("ready");
        std::string command;
        while(std::getline(std::cin,command)) {
            if(command=="QUIT")break;
            if(command=="ALLOC") {
                if(gpu || !ram.empty())throw std::runtime_error("pressure already active");
                const auto m=memory_sample();
                if(m.gpu_free<m.gpu_total/20+gpu_bytes+(64<<20) ||
                   m.ram_free<m.ram_total/20+ram_bytes+(512<<20))
                    throw std::runtime_error("insufficient headroom for bounded pressure");
                cuda_check(cudaMalloc(&gpu,gpu_bytes));cuda_check(cudaMemset(gpu,0x6B,gpu_bytes));
                ram.resize(ram_bytes,0xA7); // Commit/touch RAM, not just reserve address space.
                const auto after=memory_sample();
                if(after.gpu_free<after.gpu_total/20+(32<<20) || after.ram_free<after.ram_total/20+(128<<20))
                    throw std::runtime_error("pressure allocation crossed safety margin");
                emit("allocated");
            } else if(command=="FREE") {release();emit("freed");}
            else if(command=="SAMPLE")emit("sample");
            else throw std::runtime_error("invalid holder command");
        }
        release();return 0;
    } catch(const std::exception & e) {
        release();std::cout<<json({{"phase","error"},{"error",e.what()}}).dump()<<'\n'<<std::flush;return 1;
    }
}
