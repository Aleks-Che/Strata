# Bounded pinned host sources replace only the permutation-table H2D fence.
# The graph-capture planner, device pool and arithmetic remain unchanged.
if(STRATA_MM27_RUNTIME)
  string(PREPEND mm27_quant_dispatch "#include \"sort_table.hpp\"\n")
  set(anchor "    CUDA_CHECK(cudaMemcpyAsync(ids_buf_dev.ptr, ids_to_sorted_host.data(), 2*ne_get_rows*sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));")
  mm27_replace_once("${anchor}" "    if (!strata_mm27_quant_f32(src0->type) || !strata_mm27_sort_table_copy(ids_buf_dev.ptr, ids_to_sorted_host.data(), 2*ne_get_rows*sizeof(int32_t), stream)) {
${anchor}
    }")
  string(APPEND mm27_patch_set ",mm27-opt-in-pinned-sort-table-ring")
endif()
