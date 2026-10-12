#include "gpu_fetch_queue.cuh"
#include <cstdio>
#include <vector>
using namespace strata_ds4;
static int checks=0;
static void check(bool ok,const char *text){++checks;if(!ok)throw std::runtime_error(text);}
static __global__ void gather(const void **table,const int32_t *ids,const CpuQueuePacket *cpu,uint8_t *out) {
    const int i=threadIdx.x,e=ids[i];if(i<8)out[i]=cpu->mask[e]?255:*(const uint8_t *)table[e];
}
int main(){try {
    int count=0;if(cudaGetDeviceCount(&count)!=cudaSuccess || !count)return 77;
    cudaStream_t owner;cpu_queue_check(cudaStreamCreateWithFlags(&owner,cudaStreamNonBlocking));
    uint8_t *weights,*scratch,*cached,*output;const void **table;int32_t *ids;CpuQueuePacket *mask;
    constexpr size_t width=1024;
    cpu_queue_check(cudaMallocHost((void **)&weights,8*width+512));
    cpu_queue_check(cudaMalloc((void **)&scratch,8*(width+512)));
    cpu_queue_check(cudaMalloc((void **)&cached,width));cpu_queue_check(cudaMemset(cached,77,width));
    cpu_queue_check(cudaMalloc((void **)&output,8));cpu_queue_check(cudaMalloc((void **)&table,256*sizeof(void *)));
    cpu_queue_check(cudaMalloc((void **)&ids,8*4));cpu_queue_check(cudaMalloc((void **)&mask,sizeof(CpuQueuePacket)));
    int jobs=0;bool fail=false;GpuFetchPlan p{};p.epoch=1;p.experts=8;p.topk=4;p.tokens=2;p.width=width;p.scratch=scratch;p.compact=8;p.cached[1]=cached;
    {
        GpuFetchQueue q([&](int m,GpuFetchSlot &slot,GpuFetchQueue &service) {
            check(m==0 && slot.job.epoch==1,"wrong source epoch");++jobs;
            if(fail)throw std::runtime_error("injected read failure");
            bool seen[8]={};for(int i=0;i<slot.job.count;++i) {
                int e=slot.job.ids[i];check(e>=0 && e<8 && e!=1 && e!=3 && !seen[e],"duplicate/cached/CPU-only fetch");seen[e]=true;
                check(slot.job.destinations[i]==scratch+size_t(i)*(width+512),"compact destination mismatch");
                service.read_upload(slot.job.destinations[i],width+512,[&](size_t off,void *dst,size_t n) {
                    std::memcpy(dst,weights+e*width+off,n);return true;
                });
            }
        });
        CpuQueuePacket packet{};packet.mask[3]=1;cpu_queue_check(cudaMemcpy(mask,&packet,sizeof(packet),cudaMemcpyHostToDevice));
        int32_t route[8]={1,2,3,2,6,2,7,0};cpu_queue_check(cudaMemcpy(ids,route,sizeof(route),cudaMemcpyHostToDevice));
        q.arm();cudaGraph_t graph;cudaGraphExec_t exec;
        cpu_queue_check(cudaStreamBeginCapture(owner,cudaStreamCaptureModeGlobal));
        gpu_fetch_publish<<<1,256,0,owner>>>(q.mapped,p,ids,4,mask,table);
        q.wait(0,owner);
        gather<<<1,8,0,owner>>>(table,ids,mask,output);
        cpu_queue_check(cudaStreamEndCapture(owner,&graph));cpu_queue_check(cudaGraphInstantiate(&exec,graph,0));
        for(int r=0;r<400;++r) {
            for(int e=0;e<8;++e)std::memset(weights+e*width,(e+r)%127,width);
            route[1]=r%2?4:2;cpu_queue_check(cudaMemcpyAsync(ids,route,sizeof(route),cudaMemcpyHostToDevice,owner));
            cpu_queue_check(cudaGraphLaunch(exec,owner));cpu_queue_check(cudaStreamSynchronize(owner));
            check(!q.host[0].failure.load(cuda::memory_order_acquire),"GPU fetch failed");
            uint8_t result[8];cpu_queue_check(cudaMemcpy(result,output,8,cudaMemcpyDeviceToHost));
            for(int i=0;i<8;++i)check(result[i]==(route[i]==1?77:route[i]==3?255:(route[i]+r)%127),"stale fetched matrix");
            check(q.host[0].consumed.load(cuda::memory_order_acquire)==uint32_t(r+1),"replay sequence stale");
        }
        q.drain();check(jobs==400,"replay lost I/O request");
        fail=true;q.arm();cpu_queue_check(cudaGraphLaunch(exec,owner));cpu_queue_check(cudaStreamSynchronize(owner));q.drain();
        check(q.host[0].failure.load(cuda::memory_order_acquire)==cpu_exception,"lost I/O failure");
        fail=false;q.arm();q.cancelled.store(true);
        cpu_queue_check(cudaGraphLaunch(exec,owner));cpu_queue_check(cudaStreamSynchronize(owner));q.drain();
        check(q.host[0].failure.load(cuda::memory_order_acquire)==cpu_cancelled,"lost cancellation");
        q.arm();cpu_queue_check(cudaGraphLaunch(exec,owner));cpu_queue_check(cudaStreamSynchronize(owner));q.drain();
        check(!q.host[0].failure.load(cuda::memory_order_acquire),"restore after cancellation failed");
        cpu_queue_check(cudaGraphExecDestroy(exec));cpu_queue_check(cudaGraphDestroy(graph));
        // A cancelled graph may publish without running its wait node. Refuse
        // reuse, even though its cancelled service request has already replied.
        q.arm();q.cancelled.store(true);gpu_fetch_publish<<<1,256,0,owner>>>(q.mapped,p,ids,4,mask,table);
        cpu_queue_check(cudaStreamSynchronize(owner));q.drain();
        gpu_fetch_publish<<<1,256,0,owner>>>(q.mapped,p,ids,4,mask,table);
        cpu_queue_check(cudaStreamSynchronize(owner));
        check(q.host[0].failure.load(cuda::memory_order_acquire)==cpu_bad_request,"reused unconsumed destination");
    }
    {
        // Cross both staging-buffer boundaries and the vectorized tail. A new
        // CUDA allocation after publication must not block the I/O service.
        constexpr size_t bytes=3*GpuFetchQueue::stage_bytes+1337;
        uint8_t *large;cpu_queue_check(cudaMalloc((void **)&large,bytes));
        GpuFetchPlan big{};big.epoch=42;big.experts=big.topk=big.tokens=1;big.width=bytes;big.scratch=large;
        GpuFetchQueue q([&](int m,GpuFetchSlot &slot,GpuFetchQueue &service) {
            check(!m && slot.job.epoch==42 && slot.job.count==1,"large transfer request");
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            service.read_upload(slot.job.destinations[0],bytes,[](size_t off,void *dst,size_t n) {
                for(size_t i=0;i<n;++i)((uint8_t *)dst)[i]=uint8_t((off+i)*37);return true;
            });
        });
        int32_t zero=0;cpu_queue_check(cudaMemcpy(ids,&zero,4,cudaMemcpyHostToDevice));q.arm();
        gpu_fetch_publish<<<1,256,0,owner>>>(q.mapped,big,ids,1,nullptr,table);
        q.wait(0,owner);
        void *probe;cpu_queue_check(cudaMalloc(&probe,1<<20));
        cpu_queue_check(cudaStreamSynchronize(owner));q.drain();
        check(!q.host[0].failure.load(cuda::memory_order_acquire),"allocation blocked mapped service");
        std::vector<uint8_t> result(bytes);cpu_queue_check(cudaMemcpy(result.data(),large,bytes,cudaMemcpyDeviceToHost));
        for(size_t i=0;i<bytes;++i)check(result[i]==uint8_t(i*37),"staging wrap/tail corruption");
        check(q.host[0].chunk_done.load(cuda::memory_order_acquire)==4,"large transfer did not wrap staging");
        int32_t invalid=1;cpu_queue_check(cudaMemcpy(ids,&invalid,4,cudaMemcpyHostToDevice));
        gpu_fetch_publish<<<1,256,0,owner>>>(q.mapped,big,ids,1,nullptr,table);
        q.wait(0,owner);
        cpu_queue_check(cudaStreamSynchronize(owner));
        check(q.host[0].failure.load(cuda::memory_order_acquire)==cpu_bad_request,"invalid route accepted");
        cudaFree(probe);cudaFree(large);
    }
    cudaFreeHost(weights);cudaFree(scratch);cudaFree(cached);cudaFree(output);cudaFree(table);cudaFree(ids);cudaFree(mask);cudaStreamDestroy(owner);
    std::printf("GPU fetch queue: %d checks, 400 graph replays, mapped transfer, compact pointers, deduplication, errors and cancellation\n",checks);return 0;
}catch(const std::exception &e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
