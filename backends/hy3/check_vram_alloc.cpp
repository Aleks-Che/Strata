// Standalone allocation diagnostic. No model computation or speed claim.
#include "../common/device_memory.hpp"
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

static void check(cudaError_t e) {
    if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
}
struct Sample {size_t used,total,cuda_used;};
static StrataGlobalMemory global;
static Sample sample() {
    size_t free=0,total=0,cf=0,ct=0;
    if(!global.sample(0,free,total))throw std::runtime_error("global NVML unavailable");
    check(cudaMemGetInfo(&cf,&ct));return {total-free,total,ct-cf};
}
static Sample settled() {
    check(cudaDeviceSynchronize());
    std::this_thread::sleep_for(std::chrono::milliseconds(400));
    return sample();
}
int main() {
    std::vector<void*> allocations;
    try {
        check(cudaSetDevice(0));check(cudaFree(nullptr));
        // Actual 64-KiB-charged Hy3 matrix sizes. 237 tensors per routed trunk,
        // repeated four times. This is a size-distribution probe, not cache replay.
        std::vector<size_t> sizes;
        for(int repeat=0;repeat<4;++repeat) {
            for(int i=0;i<156;++i)sizes.push_back(2424832);
            for(int i=0;i<77;++i)sizes.push_back(3407872);
            sizes.push_back(4390912);
            for(int i=0;i<3;++i)sizes.push_back(5177344);
        }
        size_t bytes=0;for(auto n:sizes)bytes+=n;
        std::cout<<"{\"status\":\"pass\",\"matrix_count\":"<<sizes.size()<<",\"charged_bytes\":"<<bytes<<",\"runs\":[";
        for(int mode=0;mode<3;++mode) {
            const auto before=settled();
            auto start=std::chrono::steady_clock::now();
            auto allocate=[&](size_t n) {
                auto m=sample();
                if(m.total-m.used<m.total/20+(256ULL<<20)+n)
                    throw std::runtime_error("global VRAM budget95 refused");
                void *p=nullptr;check(cudaMalloc(&p,n));allocations.push_back(p);
                check(cudaMemset(p,0,n));
            };
            if(mode!=1)for(auto n:sizes)allocate(n);else allocate(bytes);
            check(cudaDeviceSynchronize());
            const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            const auto after=settled();
            if(after.used>after.total*0.95)throw std::runtime_error("global VRAM budget95 exceeded");
            for(auto p:allocations)check(cudaFree(p));allocations.clear();
            const auto released=settled();
            if(mode)std::cout<<',';
            std::cout<<"{\"mode\":\""<<(mode==1?"arena":"individual")<<"\",\"global_before\":"<<before.used
                <<",\"global_after\":"<<after.used<<",\"global_released\":"<<released.used
                <<",\"cuda_before\":"<<before.cuda_used<<",\"cuda_after\":"<<after.cuda_used
                <<",\"allocate_touch_ms\":"<<ms<<'}';
        }
        std::cout<<"]}\n";return 0;
    } catch(const std::exception &e) {
        for(auto p:allocations)cudaFree(p);
        std::cerr<<e.what()<<'\n';return 1;
    }
}
