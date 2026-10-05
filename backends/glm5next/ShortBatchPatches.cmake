# Batch verification tokens in the existing single-token quantized matvec.
# This changes launch/addressing only; vec_dot and reduction order are retained.
set(glm_mmvq_original "${glm_source}/ggml/src/ggml-cuda/mmvq.cu")
file(SHA256 "${glm_mmvq_original}" glm_mmvq_hash)
if(NOT glm_mmvq_hash STREQUAL "d4c70f68ab79d463fd6fd52cac66a62e915858397ba4b2b7f658a7fbf13121f3")
  message(FATAL_ERROR "Review GLM token-batch MMVQ patch for changed source")
endif()
file(READ "${glm_mmvq_original}" glm_mmvq)
string(REPLACE "#include \"mmvq.cuh\"" "#include \"mmvq.cuh\"\n#include \"mmvq_token_batch.hpp\"" glm_mmvq "${glm_mmvq}")
string(REPLACE
  "template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false, bool halve_iters = false>"
  "template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false, bool halve_iters = false, bool strata_token_batch = false>"
  glm_mmvq "${glm_mmvq}")
string(REPLACE "ids[channel_dst]" "ids[channel_dst + (strata_token_batch ? blockIdx.z*ids_stride : 0)]" glm_mmvq "${glm_mmvq}")
string(REPLACE
  "template<ggml_type type, int c_ncols_dst, bool small_k = false, bool halve_iters = false>"
  "template<ggml_type type, int c_ncols_dst, bool small_k = false, bool halve_iters = false, bool strata_token_batch = false>"
  glm_mmvq "${glm_mmvq}")
foreach(fused true false)
  string(REPLACE "mul_mat_vec_q<type, c_ncols_dst, ${fused}, small_k, halve_iters>"
    "mul_mat_vec_q<type, c_ncols_dst, ${fused}, small_k, halve_iters, strata_token_batch>" glm_mmvq "${glm_mmvq}")
endforeach()
string(REPLACE "template <ggml_type type>\nstatic void mul_mat_vec_q_switch_ncols_dst("
  "template <ggml_type type, bool strata_token_batch = false>\nstatic void mul_mat_vec_q_switch_ncols_dst("
  glm_mmvq "${glm_mmvq}")
string(REPLACE "    GGML_ASSERT(ncols_dst <= MMVQ_MAX_BATCH_SIZE);"
  "    GGML_ASSERT(ncols_dst <= MMVQ_MAX_BATCH_SIZE);\n#include \"mmvq_token_batch.inc\""
  glm_mmvq "${glm_mmvq}")
string(REPLACE "mul_mat_vec_q_switch_fusion<type, c_ncols_dst, c_small_k, c_halve_iters>"
  "mul_mat_vec_q_switch_fusion<type, c_ncols_dst, c_small_k, c_halve_iters, strata_token_batch>"
  glm_mmvq "${glm_mmvq}")
set(glm_mmvq_generated "${CMAKE_BINARY_DIR}/strata-glm-mmvq.cu")
set(glm_mmvq_old "")
if(EXISTS "${glm_mmvq_generated}")
  file(READ "${glm_mmvq_generated}" glm_mmvq_old)
endif()
if(NOT glm_mmvq_old STREQUAL glm_mmvq)
  file(WRITE "${glm_mmvq_generated}" "${glm_mmvq}")
endif()
get_target_property(glm_cuda_sources ggml-cuda SOURCES)
set(glm_mmvq_matches "${glm_cuda_sources}")
list(FILTER glm_mmvq_matches INCLUDE REGEX "(^|/)mmvq\\.cu$")
list(LENGTH glm_mmvq_matches glm_mmvq_count)
if(NOT glm_mmvq_count EQUAL 1)
  message(FATAL_ERROR "Expected one mmvq.cu in ggml-cuda")
endif()
list(REMOVE_ITEM glm_cuda_sources ${glm_mmvq_matches})
set_property(TARGET ggml-cuda PROPERTY SOURCES "${glm_cuda_sources};${glm_mmvq_generated}")
set_source_files_properties("${glm_mmvq_generated}" TARGET_DIRECTORY ggml-cuda PROPERTIES
  INCLUDE_DIRECTORIES "${glm_source}/ggml/src/ggml-cuda;${CMAKE_CURRENT_SOURCE_DIR}"
  OBJECT_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/mmvq_token_batch.hpp;${CMAKE_CURRENT_SOURCE_DIR}/mmvq_token_batch.inc")

# Route eligible 2..4-token graphs around the per-token dispatch loop.
set(glm_dispatch "${glm_cuda_dispatch}")
string(REPLACE "static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx,"
  "#include \"mmvq_token_batch.hpp\"\n\nstatic void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx," glm_dispatch "${glm_dispatch}")
string(REPLACE
  "    if (getenv(\"STRATA_GLM_TOKENWISE_MATMUL\") && src1->ne[1] > 1 && src1->ne[1] <= 4) {"
  "    if (strata_mmvq_token_batch_supported(src0, src1, dst, false)) {\n        ggml_cuda_mul_mat_vec_q(ctx, src0, src1, nullptr, dst); return;\n    }\n    if (getenv(\"STRATA_GLM_TOKENWISE_MATMUL\") && src1->ne[1] > 1 && src1->ne[1] <= 4) {"
  glm_dispatch "${glm_dispatch}")
set(glm_id_entry "static void ggml_cuda_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {")
string(REPLACE "${glm_id_entry}"
  "${glm_id_entry}\n    if (strata_mmvq_token_batch_supported(dst->src[0], dst->src[1], dst, true)) {\n        ggml_cuda_mul_mat_vec_q(ctx, dst->src[0], dst->src[1], dst->src[2], dst); return;\n    }"
  glm_dispatch "${glm_dispatch}")
set(glm_cuda_dispatch "${glm_dispatch}")
