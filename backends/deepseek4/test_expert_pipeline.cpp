#include "expert_pipeline.hpp"
#include <atomic>
#include <cstdio>

static void check(bool value) {if(!value)throw std::runtime_error("pipeline contract failed");}
static void cuda_ok(cudaError_t code) {if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));}
struct Gate {
    cudaStream_t stream;
    std::atomic<bool> release{false},expired{false};
    explicit Gate(cudaStream_t s):stream(s) {}
    ~Gate() {release.store(true);cudaStreamSynchronize(stream);}
};
static void CUDART_CB blocked(void *ptr) {
    auto &gate=*(Gate *)ptr;
    auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(3);
    while(!gate.release.load()) {
        if(std::chrono::steady_clock::now()>deadline) {gate.expired.store(true);break;}
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}
int main() {
    try {
        int count=0;
        if(cudaGetDeviceCount(&count)!=cudaSuccess || !count)return 77;
        constexpr size_t size=(1<<20)+17,chunk=256<<10;
        std::vector<uint8_t> source(size),actual(size);
        for(size_t i=0;i<size;++i)source[i]=uint8_t(i*17+(i>>7));
        uint8_t *dest=nullptr;cuda_ok(cudaMalloc((void **)&dest,size));
        cudaStream_t compute;cuda_ok(cudaStreamCreateWithFlags(&compute,cudaStreamNonBlocking));
        {
            StrataExpertPipeline pipeline(0,chunk,false);
            Gate gate(compute);
            cuda_ok(cudaLaunchHostFunc(compute,blocked,&gate));
            pipeline.start({{source.data(),size,false}});
            auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(2);
            while(pipeline.ready_chunks()<StrataExpertPipeline::slots && std::chrono::steady_clock::now()<deadline)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            bool overlapped=pipeline.ready_chunks()==StrataExpertPipeline::slots && !gate.expired.load();
            bool blocked_compute=cudaStreamQuery(compute)==cudaErrorNotReady;
            gate.release.store(true);
            check(overlapped && blocked_compute);
            check(pipeline.transfer(dest,source.data(),size,compute));
            pipeline.finish();cuda_ok(cudaStreamSynchronize(compute));
            cuda_ok(cudaMemcpy(actual.data(),dest,size,cudaMemcpyDeviceToHost));check(actual==source);
            // A full ring with an unconsumed suffix must cancel without needing
            // consumer events that were never recorded. Then restart safely.
            for(int repeat=0;repeat<20;++repeat) {
                pipeline.start({{source.data(),size,false},{source.data(),size,false}});
                check(pipeline.transfer(dest,source.data(),size,compute));
                pipeline.finish();
                pipeline.start({{source.data(),size,false}});
                check(pipeline.transfer(dest,source.data(),size,compute));
                pipeline.finish();cuda_ok(cudaStreamSynchronize(compute));
                cuda_ok(cudaMemcpy(actual.data(),dest,size,cudaMemcpyDeviceToHost));check(actual==source);
            }
        }
        cuda_ok(cudaStreamDestroy(compute));cuda_ok(cudaFree(dest));
        std::puts("Pipeline overlap, byte parity, ring reuse and cancellation passed");
        return 0;
    } catch(const std::exception &e) {std::fprintf(stderr,"%s\n",e.what());return 1;}
}
