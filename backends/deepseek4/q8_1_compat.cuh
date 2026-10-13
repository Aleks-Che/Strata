#pragma once
#include "ggml-cuda/quantize.cuh"
// Same pinned Q8_1 kernel and packing, with explicit RN division. Used only
// by the optional short routed MMVQ numerical contract.
void strata_quantize_row_q8_1_compat_cuda(
    const float *x,const int32_t *ids,void *vy,ggml_type type,
    int64_t columns,int64_t s01,int64_t s02,int64_t s03,
    int64_t padded,int64_t n1,int64_t n2,int64_t n3,cudaStream_t stream);
