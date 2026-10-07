# Apply to the MiMo-local dispatch AFTER its routed stride correction.
# Never change archive sources or import a GLM generated translation unit.
string(PREPEND mimo_dispatch "#include \"tokenwise_matmul.hpp\"\n")
function(mimo_dispatch_insert anchor body)
  string(FIND "${mimo_dispatch}" "${anchor}" found)
  if(found EQUAL -1)
    message(FATAL_ERROR "Missing MiMo tokenwise dispatch anchor: ${anchor}")
  endif()
  string(REPLACE "${anchor}" "${anchor}\n${body}" changed "${mimo_dispatch}")
  set(mimo_dispatch "${changed}" PARENT_SCOPE)
endfunction()
mimo_dispatch_insert("static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {" [=[
    if (mimo2_tokenwise_dense(src0, src1, dst)) {
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
mimo_dispatch_insert("static void ggml_cuda_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {" [=[
    if (mimo2_tokenwise_routed(dst)) {
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
mimo_dispatch_insert("static bool ggml_cuda_mul_mat_id_needs_sync(const ggml_tensor * dst, const int cc) {" [=[
    if (mimo2_tokenwise_routed(dst)) {
        ggml_tensor x = *dst->src[1], ids = *dst->src[2], y = *dst;
        x.ne[2] = y.ne[2] = ids.ne[1] = 1;
        y.src[1] = &x; y.src[2] = &ids;
        return ggml_cuda_mul_mat_id_needs_sync(&y, cc);
    }
]=])
