# Extend only the generated, already SHA-checked MiniMax CUDA dispatch.
string(PREPEND mm27_quant_dispatch "#include \"minimax_tokenwise.hpp\"\n")
set(anchor "static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {")
set(body [=[
    if (mm27_tokenwise_dense(src0, src1, dst)) {
        for (int64_t i = 0; i < src1->ne[1]; ++i) {
            ggml_tensor x = *src1, y = *dst;
            x.ne[1] = y.ne[1] = 1;
            x.data = (char *) src1->data + i*src1->nb[1];
            y.data = (char *) dst->data + i*dst->nb[1];
            y.src[1] = &x;
            ggml_cuda_mul_mat(ctx, src0, &x, &y);
        }
        return;
    }
]=])
mm27_replace_once("${anchor}" "${anchor}\n${body}")
set(anchor "static void ggml_cuda_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {")
set(body [=[
    if (mm27_tokenwise_routed(dst)) {
        for (int64_t i = 0; i < dst->ne[2]; ++i) {
            ggml_tensor x = *dst->src[1], ids = *dst->src[2], y = *dst;
            x.ne[2] = y.ne[2] = ids.ne[1] = 1;
            x.data = (char *) dst->src[1]->data + i*dst->src[1]->nb[2];
            ids.data = (char *) dst->src[2]->data + i*dst->src[2]->nb[1];
            y.data = (char *) dst->data + i*dst->nb[2];
            y.src[1] = &x; y.src[2] = &ids;
            ggml_cuda_mul_mat_id(ctx, &y);
        }
        return;
    }
]=])
mm27_replace_once("${anchor}" "${anchor}\n${body}")
string(APPEND mm27_patch_set ",mm27-optional-tokenwise-verification")
