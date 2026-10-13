#include "gpu_fetch_queue.cuh"
#include <cstdio>
#include <vector>
using namespace strata_ds4;
static int checks=0;
static void check(bool ok,const char *text){++checks;if(!ok)throw std::runtime_error(text);}
static __global__ void gather(const void **table,const int32_t *ids,const CpuQueuePacket *cpu,uint8_t *out) {
    const int i=threadIdx.x,e=ids[i];if(i<8)out[i]=cpu->mask[e]?255:*(const uint8_t *)table[e];
}
static __global__ void hold_compute_tail(CpuDoorbell *gate) {
    uint64_t start,now;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(start));
    gate[0].store(1,cuda::memory_order_release);
    while(!gate[1].load(cuda::memory_order_acquire)) {
        asm volatile("mov.u64 %0, %%globaltimer;":"=l"(now));
        if(now-start>500000000ULL){gate[2].store(1,cuda::memory_order_release);break;}
        __nanosleep(256);
    }
}
int main(){try {
    int count=0;if(cudaGetDeviceCount(&count)!=cudaSuccess || !count)return 77;
    cudaStream_t owner;cpu_queue_check(cudaStreamCreateWithFlags(&owner,cudaStreamNonBlocking));
    uint8_t *weights,*scratch,*cached,*output,*loans;const void **table;int32_t *ids;CpuQueuePacket *mask;
    constexpr size_t width=1024;
    cpu_queue_check(cudaMallocHost((void **)&weights,8*width+512));
    cpu_queue_check(cudaMalloc((void **)&scratch,8*(width+512)));
    cpu_queue_check(cudaMalloc((void **)&cached,width));cpu_queue_check(cudaMemset(cached,77,width));
    cpu_queue_check(cudaMalloc((void **)&loans,4*(width+512)));cpu_queue_check(cudaMemset(loans,171,4*(width+512)));
    cpu_queue_check(cudaMalloc((void **)&output,8));cpu_queue_check(cudaMalloc((void **)&table,256*sizeof(void *)));
    cpu_queue_check(cudaMalloc((void **)&ids,8*4));cpu_queue_check(cudaMalloc((void **)&mask,sizeof(CpuQueuePacket)));
    int jobs=0;bool fail=false;GpuFetchPlan p{};p.epoch=1;p.experts=8;p.topk=4;p.tokens=2;p.width=width;p.scratch=scratch;p.compact=8;p.cached[1]=cached;
    p.cache_count=4;p.cache[0]={loans,width+512,3};p.cache[1]={loans+width+512,width,0};
    p.cache[2]={loans+2*(width+512),width+512,2};p.cache[3]={loans+3*(width+512),width+512,255};
    p.scores[0]=5;p.scores[2]=1;p.scores[4]=2;
    p.keep_ram[4]=true; // fits slot 2 and its score, but retains the RAM triple
    {
        GpuFetchQueue q([&](int m,GpuFetchSlot &slot,GpuFetchQueue &service) {
            check(m==0 && slot.job.epoch==1,"wrong source epoch");++jobs;
            if(fail)throw std::runtime_error("injected read failure");
            check(slot.job.used_count>=6 && slot.job.used_count<=7,"retired route lost IDs");
            bool routed[8]={};for(int i=0;i<slot.job.used_count;++i) {
                int e=slot.job.used_ids[i];check(e>=0 && e<8 && !routed[e],"retired route not unique");routed[e]=true;
                check(slot.job.cpu_only[i]==(e==3),"retired CPU mask mismatch");
            }
            check(routed[1] && routed[3],"retired route lost cache/CPU hit");
            for(int e=0;e<256;++e)check(slot.job.cpu_selected[e]==(e==2 || e==3),"mixed CPU occurrence lost its RAM protection");
            bool seen[8]={};for(int i=0;i<slot.job.count;++i) {
                int e=slot.job.ids[i];check(e>=0 && e<8 && e!=1 && e!=3 && !seen[e],"duplicate/cached/CPU-only fetch");seen[e]=true;
                const int want=e==0?0:e==7?1:-1;
                check(slot.job.cache_slots[i]==want,"cache admission ignored size/frequency/deduplication");
                check(slot.job.destinations[i]==(want<0?scratch+size_t(i)*(width+512):(uint8_t *)p.cache[want].address),"cache/compact destination mismatch");
                service.read_upload(slot.job.destinations[i],width+(e<7?512:0),[&](size_t off,void *dst,size_t n) {
                    std::memcpy(dst,weights+e*width+off,n);return true;
                });
            }
        });
        CpuQueuePacket packet{};std::fill(std::begin(packet.slots),std::end(packet.slots),-1);
        packet.mask[3]=1;packet.slots[2]=0;packet.slots[5]=1; // expert 2 also has GPU rows
        cpu_queue_check(cudaMemcpy(mask,&packet,sizeof(packet),cudaMemcpyHostToDevice));
        int32_t route[8]={1,2,3,2,6,2,7,0};cpu_queue_check(cudaMemcpy(ids,route,sizeof(route),cudaMemcpyHostToDevice));
        q.arm();cudaGraph_t graph;cudaGraphExec_t exec;
        cpu_queue_check(cudaStreamBeginCapture(owner,cudaStreamCaptureModeGlobal));
        gpu_fetch_publish<<<1,256,0,owner>>>(q.mapped,p,ids,4,mask,table);
        q.wait(0,owner);
        gather<<<1,8,0,owner>>>(table,ids,mask,output);
        cpu_queue_check(cudaStreamEndCapture(owner,&graph));cpu_queue_check(cudaGraphInstantiate(&exec,graph,0));
        for(int r=0;r<400;++r) {
            if(r){q.drain(false);q.arm();}
            for(int e=0;e<8;++e)std::memset(weights+e*width,(e+r)%127,width);
            route[1]=r%2?4:2;cpu_queue_check(cudaMemcpyAsync(ids,route,sizeof(route),cudaMemcpyHostToDevice,owner));
            cpu_queue_check(cudaGraphLaunch(exec,owner));cpu_queue_check(cudaStreamSynchronize(owner));
            check(!q.host[0].failure.load(cuda::memory_order_acquire),"GPU fetch failed");
            uint8_t result[8];cpu_queue_check(cudaMemcpy(result,output,8,cudaMemcpyDeviceToHost));
            for(int i=0;i<8;++i)check(result[i]==(route[i]==1?77:route[i]==3?255:(route[i]+r)%127),"stale fetched matrix");
            check(q.host[0].consumed.load(cuda::memory_order_acquire)==uint32_t(r+1),"replay sequence stale");
        }
        q.drain();check(jobs==400 && q.async_disarms==399,"replay/rearm lost I/O request");
        std::vector<uint8_t> unused(width+512);cpu_queue_check(cudaMemcpy(unused.data(),loans+3*(width+512),unused.size(),cudaMemcpyDeviceToHost));
        for(auto v:unused)check(v==171,"unused loan was overwritten");
        cpu_queue_check(cudaMemcpy(unused.data(),loans+2*(width+512),unused.size(),cudaMemcpyDeviceToHost));
        for(auto v:unused)check(v==171,"RAM triple protection still wrote its GPU loan");
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
        uint8_t *large,*retained;cpu_queue_check(cudaMalloc((void **)&large,bytes));
        cpu_queue_check(cudaMallocHost((void **)&retained,bytes));
        bool retain=false;
        GpuFetchPlan big{};big.epoch=42;big.experts=big.topk=big.tokens=1;big.width=bytes;big.scratch=large;
        GpuFetchQueue q([&](int m,GpuFetchSlot &slot,GpuFetchQueue &service) {
            check(!m && slot.job.epoch==42 && slot.job.count==1,"large transfer request");
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            service.read_upload(slot.job.destinations[0],bytes,[](size_t off,void *dst,size_t n) {
                for(size_t i=0;i<n;++i)((uint8_t *)dst)[i]=uint8_t((off+i)*37);return true;
            },retain?retained:nullptr);
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
        retain=true;q.arm();gpu_fetch_publish<<<1,256,0,owner>>>(q.mapped,big,ids,1,nullptr,table);q.wait(0,owner);
        cpu_queue_check(cudaStreamSynchronize(owner));q.drain();
        check(!q.host[0].failure.load(cuda::memory_order_acquire),"retained chunk transfer failed");
        cpu_queue_check(cudaMemcpy(result.data(),large,bytes,cudaMemcpyDeviceToHost));
        check(!std::memcmp(result.data(),retained,bytes),"retained bytes differ from GPU");
        for(size_t i=0;i<bytes;++i)check(retained[i]==uint8_t(i*37),"retained chunk/tail corruption");
        int32_t invalid=1;cpu_queue_check(cudaMemcpy(ids,&invalid,4,cudaMemcpyHostToDevice));
        gpu_fetch_publish<<<1,256,0,owner>>>(q.mapped,big,ids,1,nullptr,table);
        q.wait(0,owner);
        cpu_queue_check(cudaStreamSynchronize(owner));
        check(q.host[0].failure.load(cuda::memory_order_acquire)==cpu_bad_request,"invalid route accepted");
        big.cache_count=gpu_fetch_cache_slots+1;
        cpu_queue_check(cudaMemcpy(ids,&zero,4,cudaMemcpyHostToDevice));
        gpu_fetch_publish<<<1,256,0,owner>>>(q.mapped,big,ids,1,nullptr,table);q.wait(0,owner);
        cpu_queue_check(cudaStreamSynchronize(owner));
        check(q.host[0].failure.load(cuda::memory_order_acquire)==cpu_bad_request,"invalid cache slot count accepted");
        cudaFree(probe);cudaFree(large);cudaFreeHost(retained);
    }
    {
        // Slow but progressing native reads must not share a 500 ms whole-job
        // deadline. A stopped reader must still terminate within the idle cap.
        bool stalled=false;
        GpuFetchQueue q([&](int,GpuFetchSlot &slot,GpuFetchQueue &service) {
            for(int i=0;i<6;++i) {
                std::this_thread::sleep_for(std::chrono::milliseconds(stalled?180:30));
                if(slot.failure.load(cuda::memory_order_acquire))return;
                service.upload(scratch,weights,width);service.finish_uploads();
            }
        });
        GpuFetchPlan slow{};slow.epoch=1;slow.experts=slow.tokens=slow.topk=1;slow.width=width;slow.scratch=scratch;
        int32_t zero=0;cpu_queue_check(cudaMemcpy(ids,&zero,4,cudaMemcpyHostToDevice));
        for(int pass=0;pass<2;++pass) {
            stalled=pass!=0;q.arm();gpu_fetch_publish<<<1,256,0,owner>>>(q.mapped,slow,ids,1,nullptr,table);
            q.wait(0,owner,120000000ULL);cpu_queue_check(cudaStreamSynchronize(owner));q.drain();
            check(q.host[0].failure.load(cuda::memory_order_acquire)==(stalled?cpu_timeout:cpu_ok),
                "progressing read timed out or stopped reader escaped idle deadline");
        }
    }
    {
        // Hold unrelated compute after the copy fence. Retirement must return
        // while this GPU reader is still live, without an owner stream drain.
        CpuDoorbell *gate,*mapped_gate;
        cpu_queue_check(cudaHostAlloc((void **)&gate,3*sizeof(CpuDoorbell),cudaHostAllocMapped));
        for(int i=0;i<3;++i)new(gate+i) CpuDoorbell(0);
        cpu_queue_check(cudaHostGetDevicePointer((void **)&mapped_gate,gate,0));
        GpuFetchQueue q([&](int,GpuFetchSlot &slot,GpuFetchQueue &service){service.upload(slot.job.destinations[0],weights,width);});
        GpuFetchPlan once{};once.epoch=1;once.experts=once.tokens=once.topk=1;once.width=width;once.scratch=scratch;
        int32_t zero=0;cpu_queue_check(cudaMemcpyAsync(ids,&zero,4,cudaMemcpyHostToDevice,owner));
        q.arm();gpu_fetch_publish<<<1,256,0,owner>>>(q.mapped,once,ids,1,nullptr,table);q.wait(0,owner);q.mark_copies(owner);
        hold_compute_tail<<<1,1,0,owner>>>(mapped_gate);cpu_queue_check(cudaGetLastError());
        const auto retired=q.retire(owner,true);
        const bool early=retired.early && retired.tail_pending && cudaStreamQuery(owner)==cudaErrorNotReady;
        gate[1].store(1,cuda::memory_order_release);cpu_queue_check(cudaStreamSynchronize(owner));
        check(early && !gate[2].load(cuda::memory_order_acquire),"copy retirement waited for final GPU compute");
        // A new arm invalidates the previous fence. A partial/cancelled branch
        // must use the ordinary stream fence, not its old completion event.
        q.arm();const auto partial=q.retire(owner,true);
        check(!partial.early,"partial branch reused the preceding copy fence");
        q.arm();q.mark_copies(owner);const auto disabled=q.retire(owner,false);
        check(!disabled.early,"disabled early retirement ignored full stream fence");
        for(int i=0;i<3;++i)gate[i].~CpuDoorbell();cudaFreeHost(gate);
    }
    cudaFreeHost(weights);cudaFree(scratch);cudaFree(cached);cudaFree(loans);cudaFree(output);cudaFree(table);cudaFree(ids);cudaFree(mask);cudaStreamDestroy(owner);
    std::printf("GPU fetch queue: %d checks, 400 graph replays, mapped transfer, compact pointers, deduplication, errors and cancellation\n",checks);return 0;
}catch(const std::exception &e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
