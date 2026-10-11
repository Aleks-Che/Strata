# ShortBatchPatches validates the complete pinned upstream source hash first.
# Keep arithmetic, launch geometry, bias indexing and fusion intact. Only the
# base address / expert offset differ when a GPU pointer table is supplied.
string(REPLACE "#include \"mmvq.cuh\"" "#include \"mmvq.cuh\"\n#include \"expert_direct.cuh\"" ds4_mmvq "${ds4_mmvq}")
string(REPLACE "ggml_cuda_mm_fusion_args_device" "StrataDirectFusion" ds4_mmvq "${ds4_mmvq}")
set(direct_select "    const bool direct_x = ids && fusion.expert_ptrs;\n    const bool direct_gate = ids && fusion.gate_ptrs;\n    if (direct_x) vx = fusion.expert_ptrs[channel_x];\n    if (direct_gate) vgate = fusion.gate_ptrs[channel_x];\n")
string(REPLACE
  "    const int kbx_offset = sample_x*stride_sample_x + channel_x*stride_channel_x + row0*stride_row_x;"
  "${direct_select}    const int kbx_offset = sample_x*stride_sample_x + (direct_x ? 0 : channel_x*stride_channel_x) + row0*stride_row_x;\n    const int kbg_offset = sample_x*stride_sample_x + (direct_gate ? 0 : channel_x*stride_channel_x) + row0*stride_row_x;"
  ds4_mmvq "${ds4_mmvq}")
string(REPLACE
  "    const int kbx_offset  = channel_x*stride_channel_x + row0*stride_row_x;"
  "${direct_select}    const int kbx_offset = (direct_x ? 0 : channel_x*stride_channel_x) + row0*stride_row_x;\n    const int kbg_offset = (direct_gate ? 0 : channel_x*stride_channel_x) + row0*stride_row_x;"
  ds4_mmvq "${ds4_mmvq}")
string(REPLACE "vgate, &y[j*stride_col_y + kby], kbx_offset" "vgate, &y[j*stride_col_y + kby], kbg_offset" ds4_mmvq "${ds4_mmvq}")
string(REPLACE "vgate, &y[kby], kbx_offset" "vgate, &y[kby], kbg_offset" ds4_mmvq "${ds4_mmvq}")
string(REPLACE "mmvq_prefetch_l2((const char *) vgate + off);"
  "mmvq_prefetch_l2((const char *) vgate + (size_t)(kbg_offset + i*stride_row_x + kbx_pf) * ggml_cuda_type_traits<type>::bs);"
  ds4_mmvq "${ds4_mmvq}")
string(REPLACE "    // If src0 is a temporary compute buffer, clear any potential padding."
  "    if (ids) {\n        fusion_local.expert_ptrs = strata_expert_direct_table(ctx, src0);\n        if (fusion && fusion->gate) fusion_local.gate_ptrs = strata_expert_direct_table(ctx, fusion->gate);\n    }\n\n    // If src0 is a temporary compute buffer, clear any potential padding."
  ds4_mmvq "${ds4_mmvq}")
