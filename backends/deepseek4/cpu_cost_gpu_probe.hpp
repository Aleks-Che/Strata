#pragma once
#include "ggml.h"
#include <cuda_runtime.h>
// Calibration only. Uses the same private MMVQ dispatch, Q8_1 and per-token
// skip mask as the engine. No host routing, scheduler or CPU join is inserted.
void strata_cpu_cost_matvec(ggml_type type,const void *weights,const float *input,void *quantized,float *output,
    const int32_t *ids,const int *slots,const uint8_t *mask,int columns,int rows,int experts,int topk,int tokens,
    int input_channels,cudaStream_t stream);
void strata_cpu_cost_swiglu(const float *gate,const float *up,float *hidden,int columns,int count,float clamp,cudaStream_t stream);
