#pragma once
#include "expert_direct.cuh"

// Stable addresses are essential: CUDA graph replay observes the current hit
// mask through this device table, including transitions to zero/all cache hits.
struct StrataExpertEarlyHitScratch {
    static constexpr size_t output_bytes=2ULL<<20;
    static constexpr size_t input_bytes=1ULL<<20;
    const void **pointers=nullptr;
    float *output=nullptr;
    void *input=nullptr;
    cudaStream_t compute=nullptr;
    cudaEvent_t inputs_ready=nullptr,results_ready=nullptr;
    bool enabled=false,separate=false,pending=false;
    StrataExpertEarlyHitScratch() {
        const char *value=std::getenv("STRATA_DS4_EARLY_HITS");
        separate=value && std::strcmp(value,"2")==0;
        enabled=separate || (value && std::strcmp(value,"1")==0);
#ifndef STRATA_DS4_EARLY_HITS_COMPILED
        if(enabled)GGML_ABORT("STRATA_DS4_EARLY_HITS requires STRATA_DEEPSEEK_EARLY_HITS_EXPERIMENT=ON");
#endif
    }
    ~StrataExpertEarlyHitScratch() {
        if(compute)CUDA_CHECK(cudaStreamDestroy(compute));
        if(inputs_ready)CUDA_CHECK(cudaEventDestroy(inputs_ready));
        if(results_ready)CUDA_CHECK(cudaEventDestroy(results_ready));
        if(input)CUDA_CHECK(cudaFree(input));
        if(pointers)CUDA_CHECK(cudaFree(pointers));
        if(output)CUDA_CHECK(cudaFree(output));
    }
    void prepare(cudaStream_t stream) {
        if(!enabled)return;
        if(!pointers) {
            CUDA_CHECK(cudaMalloc((void **)&pointers,sizeof(StrataExpertDirect::Packet)));
            clear(stream);
        }
        if(!output)CUDA_CHECK(cudaMalloc((void **)&output,output_bytes));
        if(separate && !compute) {
            CUDA_CHECK(cudaMalloc(&input,input_bytes));
            CUDA_CHECK(cudaStreamCreateWithFlags(&compute,cudaStreamNonBlocking));
            CUDA_CHECK(cudaEventCreateWithFlags(&inputs_ready,cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&results_ready,cudaEventDisableTiming));
        }
    }
    cudaStream_t begin(cudaStream_t owner) {
        if(!separate)return owner;
        GGML_ASSERT(!pending);
        CUDA_CHECK(cudaEventRecord(inputs_ready,owner));
        CUDA_CHECK(cudaStreamWaitEvent(compute,inputs_ready,0));return compute;
    }
    void end() {
        if(!separate)return;
        CUDA_CHECK(cudaEventRecord(results_ready,compute));pending=true;
    }
    void join(cudaStream_t owner) {
        if(!pending)return;
        CUDA_CHECK(cudaStreamWaitEvent(owner,results_ready,0));pending=false;
    }
    void clear(cudaStream_t stream) {
        join(stream);
        if(pointers)CUDA_CHECK(cudaMemsetAsync(pointers,0,sizeof(StrataExpertDirect::Packet),stream));
    }
    void publish(cudaStream_t stream,const StrataExpertDirect::Packet &packet) {
        strata_direct_publish<<<1,StrataExpertDirect::max_experts,0,stream>>>(pointers,packet);
        CUDA_CHECK(cudaGetLastError());
    }
};
