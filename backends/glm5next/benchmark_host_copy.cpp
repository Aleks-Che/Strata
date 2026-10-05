// Bounded diagnostic: >L3 source, pinned staging, separate CPU/H2D/serial times.
// Not a model throughput benchmark. Does not change any runtime defaults.
#include "../common/host_staging_copy.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using Clock=std::chrono::steady_clock;
static void check(cudaError_t e) {if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
static double seconds(Clock::time_point start) {return std::chrono::duration<double>(Clock::now()-start).count();}
int main(int argc,char **argv) {
    try {
        if(!strata_host_copy::supported)throw std::runtime_error("streaming-copy benchmark requires x86-64");
        const size_t source_offset=argc==1?0:argc==2?std::stoul(argv[1]):64;
        if(source_offset>63)throw std::runtime_error("optional source byte offset must be 0..63");
        constexpr size_t MiB=1ULL<<20,GiB=1ULL<<30,max_chunk=4*MiB;
        std::vector<uint8_t> source(2*GiB+64),actual(max_chunk);
        for(size_t i=0;i<source.size();++i)source[i]=uint8_t(i*17+i/251);
        cudaStream_t queue;cudaEvent_t begin,end;uint8_t *device=nullptr;
        check(cudaSetDevice(0));check(cudaStreamCreateWithFlags(&queue,cudaStreamNonBlocking));
        check(cudaEventCreate(&begin));check(cudaEventCreate(&end));check(cudaMalloc((void **)&device,max_chunk));
        std::cout<<"source_offset,write_combined,chunk_bytes,round,stream_copy,cpu_GiB_s,h2d_GiB_s,serial_GiB_s\n";
        for(bool wc:{false,true}) {
            uint8_t *host=nullptr;check(cudaHostAlloc((void **)&host,max_chunk,wc?cudaHostAllocWriteCombined:cudaHostAllocDefault));
            for(size_t chunk:{size_t(64*1024),size_t(256*1024),MiB,3*MiB,max_chunk}) {
                const size_t blocks=(source.size()-64)/chunk;
                for(int round=0;round<6;++round) {
                    // Alternate AB/BA order to reduce persistent order bias.
                    const bool nt=round==1 || round==2 || round==5;
                    size_t offset=0;
                    const size_t cpu_count=2*GiB/chunk,serial_count=256*MiB/chunk;
                    auto start=Clock::now();
                    for(size_t i=0;i<cpu_count;++i) {
                        offset=((i*67+round*13)%blocks)*chunk+source_offset;
                        strata_host_copy::copy(host,source.data()+offset,chunk,nt);
                        std::atomic_signal_fence(std::memory_order_seq_cst); // Retain every write in the CPU-only timing.
                    }
                    const double cpu=seconds(start);
                    check(cudaEventRecord(begin,queue));
                    for(size_t i=0;i<serial_count;++i)check(cudaMemcpyAsync(device,host,chunk,cudaMemcpyHostToDevice,queue));
                    check(cudaEventRecord(end,queue));check(cudaEventSynchronize(end));float ms=0;check(cudaEventElapsedTime(&ms,begin,end));
                    start=Clock::now();
                    for(size_t i=0;i<serial_count;++i) {
                        offset=((i*67+round*13)%blocks)*chunk+source_offset;
                        strata_host_copy::copy(host,source.data()+offset,chunk,nt);
                        check(cudaMemcpyAsync(device,host,chunk,cudaMemcpyHostToDevice,queue));
                        check(cudaStreamSynchronize(queue));
                    }
                    const double serial=seconds(start);
                    check(cudaMemcpy(actual.data(),device,chunk,cudaMemcpyDeviceToHost));
                    if(std::memcmp(actual.data(),source.data()+offset,chunk))throw std::runtime_error("GPU received different bytes");
                    const double cpu_gib=double(cpu_count)*chunk/GiB,serial_gib=double(serial_count)*chunk/GiB;
                    std::cout<<source_offset<<','<<int(wc)<<','<<chunk<<','<<round<<','<<int(nt)<<','<<cpu_gib/cpu<<','<<serial_gib/(ms/1000)<<','<<serial_gib/serial<<'\n'<<std::flush;
                }
            }
            check(cudaFreeHost(host));
        }
        cudaFree(device);cudaEventDestroy(begin);cudaEventDestroy(end);cudaStreamDestroy(queue);return 0;
    }catch(const std::exception &e) {std::cerr<<e.what()<<"\n";return 1;}
}
