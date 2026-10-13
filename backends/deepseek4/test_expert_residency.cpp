#include "expert_residency.cuh"
#include "expert_reuse.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>
#include <fcntl.h>

using Cache=strata_ds4::HostCache;
static int checks=0;
static void require(bool b,const char *s){++checks;if(!b)throw std::runtime_error(s);}
static void cuda_check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
static void CUDART_CB gate(void *p){auto &flag=*static_cast<std::atomic<bool> *>(p);while(!flag.load())std::this_thread::yield();}
int main(int argc,char **argv) {
    int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices)return 77;
    try {
        auto directory=std::filesystem::path(argc>1?argv[1]:"residency-test");
        std::filesystem::create_directories(directory);
        std::vector<uint8_t> original(32768);for(size_t i=0;i<original.size();++i)original[i]=uint8_t(i*17+(i>>8));
        auto path=directory/"source.bin";{std::ofstream f(path,std::ios::binary);f.write((const char *)original.data(),original.size());}
        auto fd=_wopen(path.c_str(),_O_RDONLY|_O_BINARY);require(fd>=0,"source opened");
        auto source=std::make_shared<strata_expert_file::Source>((void *)0x10000,original.size(),fd);_close(fd);
        Cache::Storage storage;
        storage.allocate=[](size_t n){void *p=nullptr;cuda_check(cudaMallocHost(&p,n));return (uint8_t *)p;};
        storage.release=[](uint8_t *p){cuda_check(cudaFreeHost(p));};
        auto cache=std::make_shared<Cache>(16384,[]{return Cache::Memory{8ULL<<30,6ULL<<30,16ULL<<30};},
            Cache::Policy::Lru,131072,131072,storage,8192,8192);
        StrataExpertResidency target,draft;target.cache=draft.cache=cache;
        cudaStream_t first,second;cuda_check(cudaStreamCreateWithFlags(&first,cudaStreamNonBlocking));
        cuda_check(cudaStreamCreateWithFlags(&second,cudaStreamNonBlocking));
        uint8_t *device=nullptr;cuda_check(cudaMalloc((void **)&device,4096));
        std::atomic<bool> cancel{false};
        cache->retain(source,0,original.data(),4096,cancel);auto cpu=cache->acquire(source,0,4096);
        std::atomic<bool> released{false};
        cuda_check(cudaLaunchHostFunc(first,gate,&released));
        cuda_check(cudaMemcpyAsync(device,cpu.data(),4096,cudaMemcpyHostToDevice,first));
        auto upload=target.uploaded(source,0,4096,true,first);target.poll();
        const bool pending=cache->residency(source,0,4096).gpu==Cache::State::Transferring;
        // Never leave a gated CUDA callback blocked when an assertion throws.
        released=true;cuda_check(cudaStreamSynchronize(first));require(pending,"GPU publication waits for upload completion");
        target.poll();require(cache->residency(source,0,4096).ram==Cache::State::Retired,"completed upload retires RAM");
        require(!std::memcmp(cpu.data(),original.data(),4096),"live CPU lease unchanged by GPU promotion");
        cpu={};cache->snapshot();
        // Evict target, then immediately reuse the same GPU address from draft.
        // The stream fence must include D2H, not just the earlier compute work.
        target.evict(upload,device,first,false);const auto old=upload->token;upload.reset();
        StrataExpertReuseFence fence;require(fence.order(first,second),"cross-backend reuse has a stream dependency");
        cuda_check(cudaMemsetAsync(device,0xAA,4096,second));
        auto replacement=draft.uploaded(source,4096,4096,true,second);
        cuda_check(cudaStreamSynchronize(second));target.poll();draft.poll();
        auto restored=cache->acquire(source,0,4096);
        require(restored && !std::memcmp(restored.data(),original.data(),4096),"demotion precedes cross-backend GPU overwrite");
        require(!cache->ready_gpu(old) && replacement->token.generation>old.generation,"reused GPU address cannot revive old token");
        require(target.d2h_bytes==4096 && target.d2h_jobs==1 && cache->snapshot().demotion_published==1,"D2H bytes and publication counted separately");
        replacement.reset();restored={};cache->discard(source,0,4096);cache->snapshot();
        {
            cuda_check(cudaMemcpyAsync(device,original.data(),4096,cudaMemcpyHostToDevice,first));
            cuda_check(cudaStreamSynchronize(first));
            auto known=target.uploaded(source,16384,4096,false,first,true);
            require(cache->residency(source,16384,4096).gpu==Cache::State::Ready,"known copy fence created a timing-dependent publication");
            target.evict(known,device,first,false,41);known.reset();
            cuda_check(cudaStreamSynchronize(first));target.poll();
            require(!cache->acquire(source,16384,4096),"completed younger demotion escaped epoch retirement");
            target.retire(40);require(!cache->acquire(source,16384,4096),"wrong epoch published RAM");
            target.retire(41);auto bytes=cache->acquire(source,16384,4096);
            require(bytes && !std::memcmp(bytes.data(),original.data(),4096),"epoch retirement lost copied RAM bytes");
            bytes={};cache->discard(source,16384,4096);cache->collect();
        }
        // Same-stream arena reuse and a new promotion invalidate a pending
        // demotion of the same matrix without exposing partial host bytes.
        for(int round=0;round<32;++round) {
            cuda_check(cudaMemsetAsync(device,round,4096,first));
            auto a=target.uploaded(source,8192,4096,true,first);target.poll(true);
            target.evict(a,device,first,false);a.reset();
            auto b=draft.uploaded(source,8192,4096,true,first);
            target.poll(true);draft.poll(true);
            require(!cache->acquire(source,8192,4096),"re-promotion suppresses late demotion publication");b.reset();
        }
        // Teardown drains an outstanding D2H before releasing its pinned bytes.
        cuda_check(cudaMemsetAsync(device,0x39,4096,first));
        {
            StrataExpertResidency temporary;temporary.cache=cache;
            auto a=temporary.uploaded(source,12288,4096,true,first);temporary.poll(true);
            temporary.evict(a,device,first,false);a.reset();
        }
        auto s=cache->snapshot();require(!s.pending && !s.demotion_jobs && !s.gpu_entries,"teardown cancels unpublished demotion after DMA drains");
        cuda_check(cudaFree(device));cuda_check(cudaStreamDestroy(first));cuda_check(cudaStreamDestroy(second));
        std::cout<<"PASS "<<checks<<" residency checks\n";return 0;
    } catch(const std::exception &e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
