#pragma once
#include "cpu_gpu_queue.cuh"
namespace strata_ds4 {
struct CpuGpuCandidate {
    CpuQueueMatrix matrix[3];
    uint64_t missing=0;
    unsigned score=0;
    bool ready=false,eligible=false;
};
struct CpuGpuTable {
    CpuGpuCandidate expert[256];
    uint64_t backend=0,epoch=0,layer=0;
    int experts=0,columns=0,tokens=0,topk=0,cap=0;
    bool force=false;
    double compute_us=0,copy_us_per_byte=0,overhead_us=0;
};
static __global__ void cpu_gpu_route(CpuQueueSlot *ring,CpuQueueDevice *state,
        const CpuGpuTable *table,const int32_t *ids,int stride,const float *input,CpuQueuePacket *packet) {
    __shared__ CpuQueueRequest request;
    __shared__ uint32_t seq;
    if(threadIdx.x==0) {
        seq=0;state->active=0;state->error=cpu_ok;
        ring[0].gpu_failure.store(cpu_ok,cuda::memory_order_release);
        request.backend=table->backend;request.epoch=table->epoch;request.layer=table->layer;
        request.columns=table->columns;request.tokens=table->tokens;request.topk=table->topk;request.rows=0;
        for(int i=0;i<64;++i)request.packet.slots[i]=-1;
        for(int i=0;i<256;++i)request.packet.mask[i]=0;
        bool gpu_required[256]={},used[256]={};uint64_t prefix_missing=0;double prefix_cpu=0;
        for(int t=0;t<table->tokens;++t) {
            int candidates[8],orders[8],count=0;bool seen[256]={};uint64_t total=prefix_missing;
            for(int j=0;j<table->topk;++j) {
                const int id=ids[t*stride+j];
                request.route[t*table->topk+j]=id;
                if(id<0 || id>=table->experts){state->error=cpu_bad_request;break;}
                used[id]=true;if(seen[id])continue;seen[id]=true;
                const auto &e=table->expert[id];if(!gpu_required[id])total+=e.missing;
                if(!e.ready || !e.eligible)continue;
                int k=count++;
                while(k>0 && (e.score<table->expert[candidates[k-1]].score ||
                       (e.score==table->expert[candidates[k-1]].score && j>orders[k-1]))) {
                    candidates[k]=candidates[k-1];orders[k]=orders[k-1];--k;
                }
                candidates[k]=id;orders[k]=j;
            }
            if(state->error)break;
            int take=min(table->cap,count);
            if(take && !table->force) {
                int best_n=0;double cpu=prefix_cpu,gpu=double(total)*table->copy_us_per_byte;
                double best=fmax(cpu,gpu);if(!request.rows)cpu+=table->overhead_us;
                for(int i=0;i<take;++i) {
                    const int id=candidates[i];cpu+=table->compute_us;
                    if(!gpu_required[id])gpu-=double(table->expert[id].missing)*table->copy_us_per_byte;
                    const double next=fmax(cpu,fmax(0.0,gpu));
                    if(next<best*0.98){best=next;best_n=i+1;}
                }
                take=best_n;
            }
            if(take && !table->force) {
                if(!request.rows)prefix_cpu+=table->overhead_us;
                prefix_cpu+=take*table->compute_us;
            }
            for(int j=0;j<table->topk;++j) {
                const int id=ids[t*stride+j];bool cpu=false;
                for(int i=0;i<take;++i)cpu|=candidates[i]==id;
                if(!cpu) {if(!gpu_required[id])prefix_missing+=table->expert[id].missing;gpu_required[id]=true;continue;}
                auto &row=request.row[request.rows];row.expert=id;row.token=t;row.slot=t*table->topk+j;
                for(int m=0;m<3;++m)row.matrix[m]=table->expert[id].matrix[m];
                request.packet.slots[row.slot]=request.rows++;
            }
        }
        if(request.rows && !state->error) {
            seq=cpu_queue_reserve(ring,state);
            if(seq)for(int i=0;i<256;++i)request.packet.mask[i]=used[i] && !gpu_required[i];
        }
        if(!seq) {request.rows=0;for(int i=0;i<64;++i)request.packet.slots[i]=-1;}
        *packet=request.packet;
        if(state->error)ring[0].gpu_failure.store(state->error,cuda::memory_order_release);
        if(seq)ring[seq%cpu_queue_depth].job=request;
    }
    __syncthreads();if(!seq)return;
    auto &slot=ring[seq%cpu_queue_depth];
    for(int i=threadIdx.x;i<table->tokens*table->columns;i+=blockDim.x)slot.input[i]=input[i];
    // Every writer fences before thread zero publishes the doorbell.
    __threadfence_system();__syncthreads();
    if(threadIdx.x==0)slot.request.store(seq,cuda::memory_order_release);
}
}
