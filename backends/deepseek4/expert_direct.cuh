#pragma once

// Only the private generated MMVQ translation unit uses this extended argument.
// The GPU table is stable across graph capture/replay; its contents are refreshed
// before each selected-expert tensor is consumed.
struct StrataDirectFusion : ggml_cuda_mm_fusion_args_device {
    const void *const *expert_ptrs=nullptr;
    const void *const *gate_ptrs=nullptr;
#ifdef STRATA_DS4_EARLY_HITS_COMPILED
    const void *const *early_ptrs=nullptr;
    const float *early_output=nullptr;
    bool early_phase=false;
#endif
};
const void *const *strata_expert_direct_table(ggml_backend_cuda_context &ctx,const ggml_tensor *weight);
void strata_expert_early_result(ggml_backend_cuda_context &ctx,const ggml_tensor *weight,
    const void *const **pointers,const float **output);
void strata_expert_early_matvec(ggml_backend_cuda_context &ctx,const ggml_tensor *weight,
    const ggml_tensor *input,const ggml_tensor *ids,ggml_tensor *output,const void *const *pointers,
    cudaStream_t stream,void *quantized_input);
