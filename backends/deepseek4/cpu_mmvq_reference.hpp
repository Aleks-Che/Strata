#pragma once
#include "ggml.h"
#include <cstdint>
#include <vector>

namespace strata_ds4 {
struct CpuExpertWork;
class CpuLanePool;
// Diagnostic oracle for NVIDIA generic MMVQ, not a production CPU kernel.
// No weight conversion: the same IQ3_XXS/MXFP4/Q8_0 bytes are consumed.
struct CpuQ8_1 { uint16_t d, s; int8_t qs[32]; };
static_assert(sizeof(CpuQ8_1)==36, "Q8_1 layout");
std::vector<CpuQ8_1> cpu_mmvq_quantize(const float *x,int columns);
std::vector<float> cpu_mmvq_reference(ggml_type type,int columns,int rows,
                                    const void *weights,const CpuQ8_1 *q,int warps=4);
}
