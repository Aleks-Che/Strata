# Applied only to the private, SHA-checked runtime CUDA dispatch.
if(STRATA_MM27_RUNTIME)
  string(PREPEND mm27_quant_dispatch "#include \"router_ids.hpp\"\n")
  set(anchor "    CUDA_CHECK(cudaMemcpyAsync(ids_host.data(), ids->data, ggml_nbytes(ids), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));")
  mm27_replace_once("${anchor}" "    if (!strata_mm27_quant_f32(src0->type) || !strata_mm27_router_ids_take(dst, ids_host.data(), ids_host.size())) {
${anchor}
    }")
  string(APPEND mm27_patch_set ",mm27-opt-in-split-host-router-ids")
endif()
