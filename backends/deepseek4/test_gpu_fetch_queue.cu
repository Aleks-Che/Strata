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
    p.keep_ram_assigned=std::getenv("STRATA_DS4_TEST_ASSIGNED_RAM")!=nullptr;
    if(p.keep_ram_assigned){p.keep_ram[2]=true;p.scores[2]=2;} // mixed expert fits too; its CPU occurrence must protect it
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
                const int want=e==0?0:e==7?1:e==4 && p.keep_ram_assigned?2:-1;
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
        q.arm();cudaGraph_t graph,compute=nullptr;cudaGraphExec_t exec,compute_exec=nullptr;
        cpu_queue_check(cudaStreamBeginCapture(owner,cudaStreamCaptureModeGlobal));
        gpu_fetch_publish<<<1,256,0,owner>>>(q.mapped,p,ids,4,mask,table);
        if(!q.owner_dma) {
            q.wait(0,owner);gather<<<1,8,0,owner>>>(table,ids,mask,output);
        }
        cpu_queue_check(cudaStreamEndCapture(owner,&graph));cpu_queue_check(cudaGraphInstantiate(&exec,graph,0));
        if(q.owner_dma) {
            cpu_queue_check(cudaStreamBeginCapture(owner,cudaStreamCaptureModeGlobal));
            gather<<<1,8,0,owner>>>(table,ids,mask,output);
            cpu_queue_check(cudaStreamEndCapture(owner,&compute));cpu_queue_check(cudaGraphInstantiate(&compute_exec,compute,0));
        }
        auto replay=[&] {
            cpu_queue_check(cudaGraphLaunch(exec,owner));
            if(q.owner_dma){q.wait(0,owner);cpu_queue_check(cudaGraphLaunch(compute_exec,owner));}
            cpu_queue_check(cudaStreamSynchronize(owner));
        };
        for(int r=0;r<400;++r) {
            if(r){q.drain(false);q.arm();}
            for(int e=0;e<8;++e)std::memset(weights+e*width,(e+r)%127,width);
            route[1]=r%2?4:2;cpu_queue_check(cudaMemcpyAsync(ids,route,sizeof(route),cudaMemcpyHostToDevice,owner));
            replay();
            check(!q.host[0].failure.load(cuda::memory_order_acquire),"GPU fetch failed");
            uint8_t result[8];cpu_queue_check(cudaMemcpy(result,output,8,cudaMemcpyDeviceToHost));
            for(int i=0;i<8;++i)check(result[i]==(route[i]==1?77:route[i]==3?255:(route[i]+r)%127),"stale fetched matrix");
            check(q.host[0].consumed.load(cuda::memory_order_acquire)==uint32_t(r+1),"replay sequence stale");
        }
        q.drain();check(jobs==400 && q.async_disarms==399,"replay/rearm lost I/O request");
        std::vector<uint8_t> unused(width+512);cpu_queue_check(cudaMemcpy(unused.data(),loans+3*(width+512),unused.size(),cudaMemcpyDeviceToHost));
        for(auto v:unused)check(v==171,"unused loan was overwritten");
        cpu_queue_check(cudaMemcpy(unused.data(),loans+2*(width+512),unused.size(),cudaMemcpyDeviceToHost));
        if(p.keep_ram_assigned)check(!std::memcmp(unused.data(),weights+4*width,unused.size()),"rejected CPU candidate was not admitted");
        else for(auto v:unused)check(v==171,"RAM triple protection still wrote its GPU loan");
        fail=true;q.arm();replay();q.drain();
        check(q.host[0].failure.load(cuda::memory_order_acquire)==cpu_exception,"lost I/O failure");
        fail=false;q.arm();q.cancelled.store(true);
        replay();q.drain();
        check(q.host[0].failure.load(cuda::memory_order_acquire)==cpu_cancelled,"lost cancellation");
        q.arm();replay();q.drain();
        check(!q.host[0].failure.load(cuda::memory_order_acquire),"restore after cancellation failed");
        cpu_queue_check(cudaGraphExecDestroy(exec));cpu_queue_check(cudaGraphDestroy(graph));
        if(compute_exec){cpu_queue_check(cudaGraphExecDestroy(compute_exec));cpu_queue_check(cudaGraphDestroy(compute));}
        // A cancelled graph may publish without running its wait node. Refuse
        // reuse, even though its cancelled service request has already replied.
        q.arm();q.cancelled.store(true);gpu_fetch_publish<<<1,256,0,owner>>>(q.mapped,p,ids,4,mask,table);
        cpu_queue_check(cudaStreamSynchronize(owner));q.drain();
        gpu_fetch_publish<<<1,256,0,owner>>>(q.mapped,p,ids,4,mask,table);
        cpu_queue_check(cudaStreamSynchronize(owner));
        check(q.host[0].failure.load(cuda::memory_order_acquire)==cpu_bad_request,"reused unconsumed destination");
    }
    {
        // Wrap even the largest configurable staging ring, including its tail. A new
        // CUDA allocation after publication must not block the I/O service.
        constexpr size_t bytes=3*GpuFetchQueue::max_stage_bytes+1337;
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
        const auto segment=q.staging_bytes()/(q.owner_dma && q.dma_depth>1?q.dma_stage_slots:2);
        check(q.host[0].chunk_done.load(cuda::memory_order_acquire)==(bytes+segment-1)/segment,"large transfer did not wrap staging");
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
    {
        // All three requests share scratch/table, but only the next source is
        // allowed to become ready during the preceding matrix's GPU reader.
        CpuDoorbell *gate,*mapped_gate;
        cpu_queue_check(cudaHostAlloc((void **)&gate,3*sizeof(CpuDoorbell),cudaHostAllocMapped));
        for(int i=0;i<3;++i)new(gate+i) CpuDoorbell(0);
        cpu_queue_check(cudaHostGetDevicePointer((void **)&mapped_gate,gate,0));
        std::atomic<bool> prepared[3]{};
        GpuFetchQueue q([&](int m,GpuFetchSlot &slot,GpuFetchQueue &service) {
            service.read_upload(slot.job.destinations[0],width,[&](size_t,void *dst,size_t n) {
                std::memset(dst,31+m,n);prepared[m].store(true);return true;
            });
        });
        GpuFetchPlan p{};p.epoch=1;p.experts=p.tokens=p.topk=1;p.width=width;p.scratch=scratch;
        q.ordered_sources=true;
        int32_t zero=0;cpu_queue_check(cudaMemcpy(ids,&zero,4,cudaMemcpyHostToDevice));
        for(int pass=0;pass<3;++pass) {
            for(auto &ready:prepared)ready.store(false);
            for(int i=0;i<3;++i)gate[i].store(0,cuda::memory_order_release);
            q.arm();q.order_sources(0);
            for(int m=0;m<3;++m)gpu_fetch_publish<<<1,256,0,owner>>>(q.mapped+m,p,ids,1,nullptr,nullptr);
            gpu_fetch_bind<<<1,256,0,owner>>>(q.mapped,p,table);q.wait(0,owner);
            hold_compute_tail<<<1,1,0,owner>>>(mapped_gate);
            // Flush queued work on WDDM before waiting only on host atomics.
            const auto pending=cudaStreamQuery(owner);
            if(pending!=cudaErrorNotReady)cpu_queue_check(pending);
            const auto deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(300);
            while((!prepared[1].load() || !gate[0].load(cuda::memory_order_acquire)) && std::chrono::steady_clock::now()<deadline)
                std::this_thread::yield();
            const bool overlap=prepared[1].load() && gate[0].load(cuda::memory_order_acquire) && !gate[2].load(cuda::memory_order_acquire);
            if(!overlap)std::fprintf(stderr,"early overlap: pass=%d dma=%d prepared=%d active=%u expired=%u\n",
                pass,int(q.owner_dma),int(prepared[1].load()),gate[0].load(),gate[2].load());
            gate[1].store(1,cuda::memory_order_release);cpu_queue_check(cudaStreamSynchronize(owner));
            check(overlap,"next matrix was not read during current GPU compute");
            uint8_t value=0;cpu_queue_check(cudaMemcpy(&value,scratch,1,cudaMemcpyDeviceToHost));
            check(value==31,"early read overwrote the preceding matrix");
            if(pass==1) {
                q.cancelled.store(true);q.abandon(1);q.abandon(2);q.drain();
                for(int m=1;m<3;++m) {
                    auto &s=q.host[m];check(s.request.load()==s.reply.load(),"abandoned service still owns source");
                    s.chunk_done.store(s.chunk.load());s.consumed.store(s.request.load());
                }
            } else {
                for(int m=1;m<3;++m) {
                    gpu_fetch_bind<<<1,256,0,owner>>>(q.mapped+m,p,table);q.wait(m,owner);
                    cpu_queue_check(cudaStreamSynchronize(owner));
                    cpu_queue_check(cudaMemcpy(&value,scratch,1,cudaMemcpyDeviceToHost));
                    check(value==31+m && !q.host[m].failure.load(),"early request lost source or scratch ownership");
                }
                q.drain();
            }
        }
        for(int i=0;i<3;++i)gate[i].~CpuDoorbell();cudaFreeHost(gate);
    }
    {
        // Fill the descriptor ring before allowing the owner to copy. This
        // proves bounded backpressure and >1 genuinely outstanding DMA, not
        // merely a configuration value. Failure after publication must still
        // retire every submitted source before reuse on the next request.
        constexpr size_t destination_bytes=(gpu_fetch_dma_capacity+3)*width;
        uint8_t *dest;cpu_queue_check(cudaMalloc((void **)&dest,destination_bytes));
        bool fail_after_publish=false;std::atomic<int> issued{0};
        GpuFetchQueue q([&](int,GpuFetchSlot &,GpuFetchQueue &service) {
            const int count=fail_after_publish?service.dma_depth:service.dma_depth+3;
            for(int i=0;i<count;++i) {
                service.upload(dest+i*width,weights+(i%8)*width,width);issued.store(i+1);
            }
            if(fail_after_publish)throw std::runtime_error("failure after DMA publication");
        });
        if(q.owner_dma && q.dma_depth>1) {
            GpuFetchPlan p{};p.epoch=1;p.experts=p.tokens=p.topk=1;p.width=width;p.scratch=dest;
            int32_t zero=0;cpu_queue_check(cudaMemcpy(ids,&zero,4,cudaMemcpyHostToDevice));
            q.arm();gpu_fetch_publish<<<1,256,0,owner>>>(q.mapped,p,ids,1,nullptr,table);
            cpu_queue_check(cudaStreamSynchronize(owner));
            q.start_service(); // this fixture deliberately fills DMA before submitting its consumer
            q.notify_progress(); // producer was fenced above; parked service has a host wake
            const auto deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(300);
            while(issued.load()<q.dma_depth && std::chrono::steady_clock::now()<deadline)std::this_thread::yield();
            check(issued.load()==q.dma_depth && q.host[0].chunk_done.load()==0,"DMA ring did not bound outstanding sources");
            q.wait(0,owner);q.drain();
            check(q.dma_peak_inflight==uint64_t(q.dma_depth),"DMA window was serialized or exceeded capacity");
            std::vector<uint8_t> bytes(destination_bytes);
            auto verify=[&](int count) {
                cpu_queue_check(cudaMemcpy(bytes.data(),dest,bytes.size(),cudaMemcpyDeviceToHost));
                for(int i=0;i<count;++i)check(!std::memcmp(bytes.data()+i*width,weights+(i%8)*width,width),"DMA ring payload corrupt");
                check(q.host[0].chunk.load()==q.host[0].chunk_done.load(),"DMA source released before all copies completed");
            };
            verify(q.dma_depth+3);
            for(int pass=0;pass<40;++pass) {
                fail_after_publish=pass%2==0;issued.store(0);q.arm();
                cpu_queue_check(cudaMemsetAsync(dest,0xff,destination_bytes,owner));
                gpu_fetch_publish<<<1,256,0,owner>>>(q.mapped,p,ids,1,nullptr,table);q.wait(0,owner);q.drain();
                check(q.host[0].failure.load()==(fail_after_publish?cpu_exception:cpu_ok),"DMA failure or recovery lost");
                verify(fail_after_publish?q.dma_depth:q.dma_depth+3);
            }
        }
        cudaFree(dest);
    }
    {
        // The first miss must not overwrite a cached expert selected only at
        // the END of the packet. Include CPU-only and mixed-row occurrences.
        GpuFetchPlan current{};current.epoch=7;current.experts=8;current.topk=4;current.tokens=2;
        current.width=width;current.scratch=scratch;current.compact=8;current.cache_count=3;
        current.cached[1]=loans;current.cached[5]=loans+width+512;current.cached[3]=loans+2*(width+512);
        current.cache[0]={loans,width+512,0,1};current.cache[1]={loans+width+512,width+512,0,5};
        current.cache[2]={loans+2*(width+512),width+512,0,3};
        GpuFetchQueue q([&](int,GpuFetchSlot &s,GpuFetchQueue &service) {
            check(s.job.count==3,"current-victim route lost misses");
            for(int i=0;i<s.job.count;++i) {
                const int e=s.job.ids[i],k=s.job.cache_slots[i];
                check(k==(e==0?1:-1),"selected current-layer victim overwritten");
                service.read_upload(s.job.destinations[i],width+(e<7?512:0),[&](size_t off,void *dst,size_t n) {
                    std::memcpy(dst,weights+e*width+off,n);return true;
                });
            }
        });
        int32_t route[8]={0,2,3,0,7,3,1,1};
        for(int pass=0;pass<24;++pass) {
            CpuQueuePacket packet{};std::fill(std::begin(packet.slots),std::end(packet.slots),-1);
            packet.mask[3]=pass%2;packet.slots[6]=0; // expert 1 also occurs on GPU
            cpu_queue_check(cudaMemcpy(mask,&packet,sizeof(packet),cudaMemcpyHostToDevice));
            cpu_queue_check(cudaMemcpy(ids,route,sizeof(route),cudaMemcpyHostToDevice));
            cpu_queue_check(cudaMemset(loans,77,3*(width+512)));
            for(int e=0;e<8;++e)std::memset(weights+e*width,20+e,width);
            q.arm();gpu_fetch_publish<<<1,256,0,owner>>>(q.mapped,current,ids,4,mask,table);
            q.wait(0,owner);gather<<<1,8,0,owner>>>(table,ids,mask,output);
            cpu_queue_check(cudaStreamSynchronize(owner));q.drain();
            check(!q.host[0].failure.load(),"current-victim transfer failed");
            uint8_t result[8];cpu_queue_check(cudaMemcpy(result,output,8,cudaMemcpyDeviceToHost));
            for(int i=0;i<8;++i)check(result[i]==(packet.mask[route[i]]?255:route[i]==1||route[i]==3?77:20+route[i]),
                "current-victim table consumer corrupt");
            std::vector<uint8_t> bytes(width+512);
            for(int k:{0,2}) {
                cpu_queue_check(cudaMemcpy(bytes.data(),loans+k*(width+512),bytes.size(),cudaMemcpyDeviceToHost));
                for(auto v:bytes)check(v==77,"selected victim padding overwritten");
            }
            cpu_queue_check(cudaMemcpy(bytes.data(),loans+width+512,bytes.size(),cudaMemcpyDeviceToHost));
            check(!std::memcmp(bytes.data(),weights,bytes.size()),"unselected victim admission lost padding");
        }
    }
    {
        // Owner admission must precede service access even on rearm. Publish
        // all three early requests, then cancel some before any approval.
        std::atomic<int> executed{0};
        GpuFetchQueue q([&](int m,GpuFetchSlot &slot,GpuFetchQueue &service) {
            check(slot.admitted.load(cuda::memory_order_acquire)==slot.request.load(),"service bypassed owner admission");
            check(slot.job.destinations[0]==loans+m*(width+512),"service used pre-admission scratch address");
            ++executed;service.upload(slot.job.destinations[0],weights,width+512);
        });
        q.owner_dma=true;q.owner_admission=true;q.ordered_sources=true;
        q.admit=[&](int) {
            for(int m=0;m<3;++m) {
                auto &slot=q.host[m];if(slot.admitted.load()==slot.request.load())continue;
                slot.job.destinations[0]=loans+m*(width+512);
                slot.admitted.store(slot.request.load(),cuda::memory_order_release);
            }
        };
        GpuFetchPlan demand{};demand.epoch=9;demand.experts=8;demand.topk=demand.tokens=1;
        demand.width=width;demand.scratch=scratch;
        int32_t zero=0;cpu_queue_check(cudaMemcpy(ids,&zero,4,cudaMemcpyHostToDevice));
        for(int pass=0;pass<12;++pass) {
            const int before=executed.load();q.arm();q.order_sources(0);
            for(int m=0;m<3;++m)gpu_fetch_publish<<<1,256,0,owner>>>(q.mapped+m,demand,ids,1,nullptr,nullptr);
            cpu_queue_check(cudaStreamSynchronize(owner));
            // A previous sequence must not authorize a freshly published job.
            for(int m=0;m<3;++m)check(q.host[m].admitted.load()!=q.host[m].request.load(),"stale admission reused");
            check(executed.load()==before,"unapproved request executed");
            if(pass%3==0) {
                q.cancelled.store(true);for(int m=0;m<3;++m)q.abandon(m);q.drain();
                for(int m=0;m<3;++m)q.host[m].consumed.store(q.host[m].request.load());
                check(executed.load()==before,"cancelled unapproved request executed");
            } else {
                for(int m=0;m<3;++m)q.wait(m,owner);q.drain();
                check(executed.load()==before+3,"early approval lost a matrix");
                std::vector<uint8_t> bytes(width+512);
                for(int m=0;m<3;++m) {
                    check(!q.host[m].failure.load(),"approved DMA failed");
                    cpu_queue_check(cudaMemcpy(bytes.data(),loans+m*(width+512),bytes.size(),cudaMemcpyDeviceToHost));
                    check(!std::memcmp(bytes.data(),weights,bytes.size()),"approved destination payload corrupt");
                }
            }
        }
    }
    {
        // Empty/hit-only packets retain epochs and source order without a
        // service callback. Nonempty neighbours still run on the I/O thread.
        const auto owner_thread=std::this_thread::get_id();int empty=0,nonempty=0;bool inject=false;
        GpuFetchQueue q([&](int,GpuFetchSlot &slot,GpuFetchQueue &service) {
            if(!slot.job.count) {
                if(std::this_thread::get_id()!=owner_thread)throw std::runtime_error("empty callback ran on I/O service");
                ++empty;if(inject)throw std::runtime_error("empty metadata failure");
            } else {
                if(std::this_thread::get_id()==owner_thread)throw std::runtime_error("owner performed I/O callback");
                ++nonempty;service.upload(slot.job.destinations[0],weights,width);
            }
        });
        q.owner_dma=q.owner_empty=q.owner_admission=q.ordered_sources=true;
        q.admit=[&](int){for(int m=0;m<3;++m)q.host[m].admitted.store(q.host[m].request.load(),cuda::memory_order_release);};
        GpuFetchPlan hit{};hit.epoch=77;hit.experts=hit.topk=hit.tokens=1;hit.width=width;hit.scratch=scratch;
        int32_t zero=0;cpu_queue_check(cudaMemcpy(ids,&zero,4,cudaMemcpyHostToDevice));
        for(int pass=0;pass<20;++pass) {
            q.arm();q.order_sources(0);const int before_empty=empty,before_nonempty=nonempty;
            const bool all_hit=pass%4==2;
            for(int m=0;m<3;++m) {hit.cached[0]=m==1 && !all_hit?nullptr:cached;gpu_fetch_publish<<<1,256,0,owner>>>(q.mapped+m,hit,ids,1,nullptr,nullptr);}
            cpu_queue_check(cudaStreamSynchronize(owner));inject=pass%4==1;
            if(pass%4==0) {
                q.cancelled.store(true);for(int m=0;m<3;++m)q.abandon(m);q.drain();
                check(empty==before_empty && nonempty==before_nonempty,"cancelled empty packet executed");
                for(int m=0;m<3;++m)q.host[m].consumed.store(q.host[m].request.load());
            } else {
                for(int m=0;m<3;++m)q.wait(m,owner);q.drain();
                check(empty==before_empty+2+int(all_hit) && nonempty==before_nonempty+int(!all_hit),"mixed empty/nonempty source order lost a packet");
                for(int m=0;m<3;++m)check(q.host[m].failure.load()==(inject && m!=1?cpu_exception:cpu_ok),"empty error/recovery protocol failed");
            }
        }
        check(q.empty_requests==35 && q.service_wakes==15 && q.service_skips==5,"empty completion/wake count mismatch");
    }
    {
        GpuFetchQueue q([&](int,GpuFetchSlot &slot,GpuFetchQueue &service) {
            service.upload(slot.job.destinations[0],weights,width);
        });
        if(q.service_park) {
            q.owner_empty=false;q.owner_admission=true;q.ordered_sources=false;
            q.admit=[&](int m){q.host[m].admitted.store(q.host[m].request.load(),cuda::memory_order_release);};
            GpuFetchPlan p{};p.epoch=99;p.experts=p.topk=p.tokens=1;p.width=width;p.scratch=scratch;
            int32_t zero=0;cpu_queue_check(cudaMemcpy(ids,&zero,4,cudaMemcpyHostToDevice));
            for(int pass=0;pass<32;++pass) {
                const auto before=q.service_waits.load();q.arm();
                gpu_fetch_publish<<<1,256,0,owner>>>(q.mapped,p,ids,1,nullptr,nullptr);
                cpu_queue_check(cudaStreamSynchronize(owner));
                const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(1);
                while(q.service_waits.load()==before && std::chrono::steady_clock::now()<deadline)std::this_thread::yield();
                check(q.service_waits.load()>before,"service did not park while admission was withheld");
                if(pass%4==0) {
                    q.cancelled.store(true);q.abandon(0);q.drain();
                    check(q.host[0].error==cpu_cancelled,"parked cancellation did not wake");
                    q.host[0].consumed.store(q.host[0].request.load());
                } else {
                    q.wait(0,owner);q.drain(pass%2==0);
                    check(q.host[0].error==cpu_ok && q.host[0].request.load()==q.host[0].consumed.load(),"parked service lost an owner wake");
                    uint8_t result[8];cpu_queue_check(cudaMemcpy(result,scratch,sizeof(result),cudaMemcpyDeviceToHost));
                    check(!std::memcmp(result,weights,sizeof(result)),"parked service copied wrong source");
                }
            }
        }
    }
    cudaFreeHost(weights);cudaFree(scratch);cudaFree(cached);cudaFree(loans);cudaFree(output);cudaFree(table);cudaFree(ids);cudaFree(mask);cudaStreamDestroy(owner);
    std::printf("GPU fetch queue: %d checks, 400 graph replays, mapped transfer, compact pointers, deduplication, errors and cancellation\n",checks);return 0;
}catch(const std::exception &e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
