#pragma once
#include <cuda_runtime.h>
#include <stdexcept>

// Cache replacements stay ordered on the GPU. Same-stream reuse needs no
// barrier; another context waits for the source stream through an event.
class StrataExpertReuseFence {
    cudaEvent_t event=nullptr;
    static void check(cudaError_t value) {
        if(value!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(value));
    }
public:
    StrataExpertReuseFence()=default;
    StrataExpertReuseFence(const StrataExpertReuseFence&)=delete;
    StrataExpertReuseFence &operator=(const StrataExpertReuseFence&)=delete;
    ~StrataExpertReuseFence() {if(event)cudaEventDestroy(event);}
    bool order(cudaStream_t source,cudaStream_t destination) {
        if(source==destination)return false;
        if(!event)check(cudaEventCreateWithFlags(&event,cudaEventDisableTiming));
        check(cudaEventRecord(event,source));
        check(cudaStreamWaitEvent(destination,event,0));
        return true;
    }
};
