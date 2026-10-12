# The early pass uses the same quantization, dot products, launch geometry and
# reductions. The normal unfused pass copies only already computed output rows.
# Device-side masks remain dynamic under CUDA graph replay.
function(strata_early_mmvq_replace old new)
  string(FIND "${ds4_mmvq}" "${old}" pos)
  if(pos EQUAL -1)
    message(FATAL_ERROR "Pinned early-hit MMVQ anchor missing: ${old}")
  endif()
  string(REPLACE "${old}" "${new}" patched "${ds4_mmvq}")
  set(ds4_mmvq "${patched}" PARENT_SCOPE)
endfunction()
set(early_generic [=[
    if constexpr (!has_fusion) {
        if (ids && fusion.early_ptrs) {
            const bool hit = fusion.early_ptrs[channel_x] != nullptr;
            if (fusion.early_phase && !hit) return;
            if (!fusion.early_phase && hit) {
                if (threadIdx.y == 0 && threadIdx.x < rows_per_cuda_block &&
                    uint32_t(row0 + threadIdx.x) < stride_col_dst) {
                    for (int j = 0; j < ncols_dst; ++j) {
                        const size_t offset = size_t(sample_dst)*stride_sample_dst +
                            size_t(channel_dst)*stride_channel_dst + size_t(j)*stride_col_dst + row0 + threadIdx.x;
                        dst[offset] = fusion.early_output[offset];
                    }
                }
                return;
            }
        }
    }
]=])
set(early_moe [=[
    if constexpr (!has_fusion) {
        if (fusion.early_ptrs) {
            const bool hit = fusion.early_ptrs[channel_x] != nullptr;
            if (fusion.early_phase && !hit) { ggml_cuda_pdl_lc(); return; }
            if (!fusion.early_phase && hit) {
                if (threadIdx.x < c_rows_per_block && uint32_t(row0 + threadIdx.x) < nrows_x) {
                    const size_t offset = size_t(channel_dst)*stride_channel_dst +
                        size_t(token_idx)*stride_col_dst + row0 + threadIdx.x;
                    dst[offset] = fusion.early_output[offset];
                }
                ggml_cuda_pdl_lc(); return;
            }
        }
    }
]=])
set(generic_y "    const block_q8_1 * y = ((const block_q8_1 *) vy) + sample_y*stride_sample_y + channel_y*stride_channel_y;")
set(moe_y "    const block_q8_1 * y = ((const block_q8_1 *) vy) + channel_y*stride_channel_y + token_idx*stride_col_y;")
strata_early_mmvq_replace("${generic_y}" "${early_generic}\n${generic_y}")
strata_early_mmvq_replace("${moe_y}" "${early_moe}\n${moe_y}")
set(matvec_head "void ggml_cuda_mul_mat_vec_q(\n        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,\n        const ggml_cuda_mm_fusion_args_host * fusion) {")
set(impl_head "static void strata_early_matvec_impl(\n        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,\n        const ggml_cuda_mm_fusion_args_host * fusion, const void *const *early_ptrs, bool early_phase, cudaStream_t early_stream, void *early_input) {")
strata_early_mmvq_replace("${matvec_head}" "${impl_head}")
strata_early_mmvq_replace("    // If src0 is a temporary compute buffer, clear any potential padding.\n    if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {"
  "    // Early work reads only cache allocations; never write scheduler scratch from its private stream.\n    if (!early_phase && ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {")
# Change only the regular matvec helper's stream and temporary input. Other
# MMVQ wrappers keep their existing pools and streams.
strata_early_mmvq_replace("    cudaStream_t stream = ctx.stream();\n\n    const size_t ts_src0"
  "    cudaStream_t stream = early_phase ? early_stream : ctx.stream();\n\n    const size_t ts_src0")
set(input_allocation "    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1);")
strata_early_mmvq_replace("${input_allocation}"
  "    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool());\n    char *strata_quantized = (char *) early_input;\n    if (!strata_quantized) strata_quantized = src1_q8_1.alloc(ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1);")
# These exact expressions belong only to this wrapper.
strata_early_mmvq_replace("quantize_row_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(),"
  "quantize_row_q8_1_cuda(src1_d, nullptr, strata_quantized,")
strata_early_mmvq_replace("src0->data, src0->type, src1_q8_1.get(), ids_d,"
  "src0->data, src0->type, strata_quantized, ids_d,")
set(direct_hooks [=[
    if (ids) {
        fusion_local.expert_ptrs = strata_expert_direct_table(ctx, src0);
        if (fusion && fusion->gate) fusion_local.gate_ptrs = strata_expert_direct_table(ctx, fusion->gate);
    }
]=])
set(early_hooks [=[
    if (ids && early_phase) {
        fusion_local.expert_ptrs = early_ptrs;
        fusion_local.early_ptrs = early_ptrs;
        fusion_local.early_phase = true;
    } else if (ids) {
        fusion_local.expert_ptrs = strata_expert_direct_table(ctx, src0);
        if (fusion && fusion->gate) fusion_local.gate_ptrs = strata_expert_direct_table(ctx, fusion->gate);
        strata_expert_early_result(ctx, src0, &fusion_local.early_ptrs, &fusion_local.early_output);
    }
]=])
strata_early_mmvq_replace("${direct_hooks}" "${early_hooks}")
set(early_wrappers [=[
void ggml_cuda_mul_mat_vec_q(
        ggml_backend_cuda_context &ctx,const ggml_tensor *w,const ggml_tensor *x,const ggml_tensor *ids,
        ggml_tensor *dst,const ggml_cuda_mm_fusion_args_host *fusion) {
    strata_early_matvec_impl(ctx,w,x,ids,dst,fusion,nullptr,false,nullptr,nullptr);
}
void strata_expert_early_matvec(
        ggml_backend_cuda_context &ctx,const ggml_tensor *w,const ggml_tensor *x,const ggml_tensor *ids,
        ggml_tensor *dst,const void *const *pointers,cudaStream_t stream,void *quantized_input) {
    strata_early_matvec_impl(ctx,w,x,ids,dst,nullptr,pointers,true,stream,quantized_input);
}
]=])
strata_early_mmvq_replace("void ggml_cuda_op_mul_mat_vec_q(" "${early_wrappers}\nvoid ggml_cuda_op_mul_mat_vec_q(")
