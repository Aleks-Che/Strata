#pragma once
#include "cpu_mmvq_reference.hpp"
#include "cpu_lane.hpp"

namespace strata_ds4 {
bool cpu_mmvq_fast_enabled();
bool cpu_mmvq_avx2_available();
// Same four-warp lane/FMA/reduction order as the generic NVIDIA oracle.
// One weight stream is shared by 1..4 quantized activation vectors.
void cpu_mmvq_fast(ggml_type type,int columns,int rows,const void *weights,
                   const CpuQ8_1 *const *q,float *const *out,int tokens);
void cpu_mmvq_fast_compute(CpuExpertWork &,CpuLanePool &,CpuExpertTiming *,const CpuWeightRetire &);
}
