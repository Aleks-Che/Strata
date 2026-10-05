#pragma once
#include <cstdlib>
#include <cstring>

// Experimental: keep the one-token MMVQ reduction, but launch all verification
// tokens together. The normal path remains available for a same-binary A/B.
static inline bool strata_mmvq_token_batch_enabled() {
    const char * value = std::getenv("STRATA_GLM_MMVQ_TOKEN_BATCH");
    return value && std::strcmp(value, "1") == 0 &&
           std::getenv("STRATA_GLM_TOKENWISE_MATMUL");
}

static inline bool strata_mmvq_token_batch_supported(
        const ggml_tensor * a, const ggml_tensor * b, const ggml_tensor * dst, bool routed) {
    if (!strata_mmvq_token_batch_enabled() || !ggml_is_quantized(a->type) ||
        b->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
        a->ne[3] != 1 || b->ne[3] != 1 || dst->ne[3] != 1) return false;
    const auto tokens = routed ? dst->ne[2] : dst->ne[1];
    if (tokens < 2 || tokens > 4) return false;
    if (!routed && (a->ne[2] != 1 || b->ne[2] != 1 || dst->ne[2] != 1)) return false;
    // Preserve the backend's cuBLAS fallback for views with unsafe padding.
    if (a->view_src && ggml_backend_buffer_get_usage(a->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
        ggml_nbytes(a) != ggml_backend_buffer_get_alloc_size(a->buffer, a)) return false;
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    return GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_should_use_mmvq(a->type, cc, 1);
}
