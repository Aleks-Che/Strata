#pragma once

// Only the private generated MMVQ translation unit uses this extended argument.
// The GPU table is stable across graph capture/replay; its contents are refreshed
// before each selected-expert tensor is consumed.
struct StrataDirectFusion : ggml_cuda_mm_fusion_args_device {
    const void *const *expert_ptrs=nullptr;
    const void *const *gate_ptrs=nullptr;
};
const void *const *strata_expert_direct_table(ggml_backend_cuda_context &ctx,const ggml_tensor *weight);
