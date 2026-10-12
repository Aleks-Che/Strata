#include "cpu_gpu_router.cuh"
#include <cstdio>
#include <vector>
#include <cstring>
using namespace strata_ds4;
static int checks=0;
static void check(bool ok,const char *message){++checks;if(!ok)throw std::runtime_error(message);}
static __global__ void activate(CpuQueueDevice *s,uint32_t seq){s->active=seq;}
int main()try {
    int count=0;if(cudaGetDeviceCount(&count)!=cudaSuccess || !count)return 77;
    cudaStream_t stream;cpu_queue_check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    CpuGpuTable table;table.backend=123;table.epoch=1;table.layer=7;table.experts=4;
    table.columns=32;table.tokens=3;table.topk=2;table.cap=1;table.force=true;
    for(int i=0;i<4;++i) {
        auto &e=table.expert[i];e.ready=e.eligible=true;e.missing=1024;
        for(int m=0;m<3;++m)e.matrix[m]={uint64_t(100+i*3+m),uint64_t(10+i*3+m)};
    }
    std::atomic<int> calls{0},mode{0};
    CpuGpuQueue queue([&](CpuQueueSlot &s) {
        ++calls;auto &job=s.job;
        if(mode.load()==1)throw std::runtime_error("injected CPU failure");
        if(job.backend!=123 || job.epoch!=1 || job.layer!=7){s.error=cpu_bad_request;return;}
        for(int r=0;r<job.rows;++r) {
            const auto &e=job.row[r];
            for(int m=0;m<3;++m)if(e.matrix[m].generation!=uint64_t(10+e.expert*3+m)) {s.error=cpu_bad_request;return;}
            for(int j=0;j<job.columns;++j)s.output[r*job.columns+j]=s.input[e.token*job.columns+j]+float(e.expert*10+e.slot);
        }
    });
    CpuGpuTable *dt;CpuQueuePacket *packet;int32_t *ids;float *input,*output;
    cpu_queue_check(cudaMalloc((void **)&dt,sizeof(table)));cpu_queue_check(cudaMalloc((void **)&packet,sizeof(*packet)));
    cpu_queue_check(cudaMalloc((void **)&ids,6*4));cpu_queue_check(cudaMalloc((void **)&input,96*4));cpu_queue_check(cudaMalloc((void **)&output,64*32*4));
    auto upload=[&]{cpu_queue_check(cudaMemcpyAsync(dt,&table,sizeof(table),cudaMemcpyHostToDevice,stream));};upload();
    const int32_t route[6]={0,1,1,0,2,1};cpu_queue_check(cudaMemcpyAsync(ids,route,sizeof(route),cudaMemcpyHostToDevice,stream));
    std::vector<float> x(96),out(96);
    auto publish=[&]{cpu_gpu_route<<<1,256,0,stream>>>(queue.mapped,queue.device,dt,ids,2,input,packet);cpu_queue_check(cudaGetLastError());};
    auto join=[&](uint64_t timeout=500000000){cpu_queue_join<<<1,256,0,stream>>>(queue.mapped,queue.device,output,timeout);cpu_queue_check(cudaGetLastError());};
    auto state=[&]{CpuQueueDevice s;cpu_queue_check(cudaMemcpy(&s,queue.device,sizeof(s),cudaMemcpyDeviceToHost));return s;};
    auto sync=[&]{cpu_queue_check(cudaStreamSynchronize(stream));};
    // The graph holds no host job number; producer advances its device counter
    // on every replay, even though all captured node arguments are identical.
    sync();cudaGraph_t graph;cudaGraphExec_t executable;
    cpu_queue_check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));publish();join();
    cpu_queue_check(cudaStreamEndCapture(stream,&graph));cpu_queue_check(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
    queue.arm();
    for(int run=0;run<1200;++run) {
        for(int i=0;i<96;++i)x[i]=float(run*128+i);
        cpu_queue_check(cudaMemcpyAsync(input,x.data(),96*4,cudaMemcpyHostToDevice,stream));
        cpu_queue_check(cudaGraphLaunch(executable,stream));sync();
        cpu_queue_check(cudaMemcpy(out.data(),output,96*4,cudaMemcpyDeviceToHost));
        check(state().sequence==uint32_t(run+1) && state().error==cpu_ok,"unique replay generation or reply");
        for(int t=0;t<3;++t)for(int j=0;j<32;++j)
            check(out[t*32+j]==x[t*32+j]+route[t*2+1]*10+(t*2+1),"mapped input/output visibility");
    }
    check(calls==1200,"duplicate or dropped service job");
    CpuQueuePacket p;cpu_queue_check(cudaMemcpy(&p,packet,sizeof(p),cudaMemcpyDeviceToHost));
    check(!p.mask[0] && !p.mask[1] && !p.mask[2],"mixed CPU/GPU occurrence must retain GPU fetch");
    check(p.slots[1]==0 && p.slots[3]==1 && p.slots[5]==2 && p.slots[0]==-1,"original top-k row order");
    const auto prior=state().published;table.force=false;table.compute_us=100;table.copy_us_per_byte=0.001;
    upload();publish();join();sync();check(state().active==0 && state().published==prior,"unprofitable work was published");
    table.compute_us=0.1;upload();publish();join();sync();check(state().published==prior+1,"profitable prefix was not published");
    table.force=true;table.tokens=1;upload();publish();join();sync();
    std::vector<float> first(32);cpu_queue_check(cudaMemcpy(first.data(),output,32*4,cudaMemcpyDeviceToHost));
    table.tokens=3;upload();
    for(int variant=0;variant<8;++variant) {
        int32_t future[6]={0,1,variant%4,(variant+1)%4,(variant+2)%4,(variant+3)%4};
        cpu_queue_check(cudaMemcpyAsync(ids,future,sizeof(future),cudaMemcpyHostToDevice,stream));publish();join();sync();
        cpu_queue_check(cudaMemcpy(out.data(),output,32*4,cudaMemcpyDeviceToHost));
        check(!std::memcmp(first.data(),out.data(),32*4),"future routes changed the prefix decision or result");
    }
    cpu_queue_check(cudaMemcpyAsync(ids,route,sizeof(route),cudaMemcpyHostToDevice,stream));
    // A retained snapshot is checked by the service before its addresses are used.
    table.expert[1].matrix[0].generation=999;upload();publish();join();sync();
    check(state().error==cpu_bad_request,"stale matrix generation accepted");
    check(queue.host[0].gpu_failure.load(cuda::memory_order_acquire)==cpu_bad_request,"host error doorbell omitted GPU failure");
    table.expert[1].matrix[0].generation=13;upload();mode=1;publish();join();sync();
    check(state().error==cpu_exception,"CPU exception not propagated");mode=0;
    queue.drain();
    // Fill both slots without arming the service. The third decision must be
    // all-GPU; no overwrite, dropped reply, or post-publication fallback.
    publish();publish();publish();sync();auto full=state();
    check(full.active==0 && full.full==1,"bounded ring did not reject a full queue");
    cpu_queue_check(cudaMemcpy(&p,packet,sizeof(p),cudaMemcpyDeviceToHost));
    check(p.slots[1]==-1 && !p.mask[1],"full ring retained a CPU decision");
    queue.arm();
    for(uint32_t seq=full.sequence-1;seq<=full.sequence;++seq){activate<<<1,1,0,stream>>>(queue.device,seq);join();sync();check(state().error==cpu_ok,"full ring corrupted an older job");}
    queue.drain();
    queue.arm();publish();sync();queue.drain();queue.abandon_after_drain();
    queue.arm();publish();join();sync();check(state().error==cpu_ok && state().active,"cancel without down consumer leaked a ring slot");queue.drain();
    check(queue.host[0].gpu_failure.load(cuda::memory_order_acquire)==cpu_ok,"successful generation retained an old error");
    // No CPU consumer: late join must stop, emit an error, and retain the slot
    // until CPU acknowledges cancellation. Never read its previous output.
    publish();join(2000000);sync();check(state().error==cpu_timeout,"missing service did not time out");
    queue.cancel(1);queue.drain();
    activate<<<1,1,0,stream>>>(queue.device,state().sequence);join();sync();
    check(state().error==cpu_cancelled,"cancelled job returned a stale successful answer");
    cudaGraphExecDestroy(executable);cudaGraphDestroy(graph);sync();queue.drain();
    cudaFree(dt);cudaFree(packet);cudaFree(ids);cudaFree(input);cudaFree(output);cudaStreamDestroy(stream);
    std::printf("CPU GPU queue: %d checks, 1200 graph replays, full/error/cancel/timeout passed\n",checks);return 0;
}catch(const std::exception &e){std::fprintf(stderr,"%s\n",e.what());return 1;}
