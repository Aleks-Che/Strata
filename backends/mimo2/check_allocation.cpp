#include "../common/device_memory.hpp"
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
static void check(cudaError_t rc) {if(rc!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(rc));}
int main() {
    try {
        check(cudaFree(nullptr));StrataGlobalMemory memory;
        auto used=[&] {size_t free=0,total=0;if(!memory.sample(0,free,total))throw std::runtime_error("NVML unavailable");
            if(free<total/20+(2ull<<30))throw std::runtime_error("insufficient allocation probe budget");return total-free;};
        std::cout<<"[\n";bool first=true;
        for(size_t bytes:{size_t(2752512),size_t(3604480),size_t(4456448),size_t(4)<<20,size_t(6)<<20,size_t(64)<<20}) {
            const size_t count=bytes==(64u<<20)?8:128;const auto before=used();std::vector<void *> pointers;
            for(size_t i=0;i<count;++i) {void *p=nullptr;check(cudaMalloc(&p,bytes));pointers.push_back(p);}
            check(cudaDeviceSynchronize());std::this_thread::sleep_for(std::chrono::milliseconds(300));const auto after=used();
            for(auto *p:pointers)check(cudaFree(p));
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
            if(!first)std::cout<<",\n";first=false;
            std::cout<<"{\"request_per_allocation\":"<<bytes<<",\"count\":"<<count<<",\"requested\":"<<bytes*count<<",\"global_delta\":"<<after-before<<"}";
        }
        std::cout<<"\n]\n";return 0;
    } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
