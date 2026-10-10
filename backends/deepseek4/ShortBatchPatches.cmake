# Batch verification tokens in the existing single-token quantized matvec.
# This changes launch/addressing only; vec_dot and reduction order are retained.
set(ds4_mmvq_original "${STRATA_LLAMA_DIR}/ggml/src/ggml-cuda/mmvq.cu")
file(SHA256 "${ds4_mmvq_original}" ds4_mmvq_hash)
if(NOT ds4_mmvq_hash STREQUAL "d4c70f68ab79d463fd6fd52cac66a62e915858397ba4b2b7f658a7fbf13121f3")
  message(FATAL_ERROR "Review DeepSeek token-batch MMVQ patch for changed source")
endif()
file(READ "${ds4_mmvq_original}" ds4_mmvq)
string(REPLACE "#include \"mmvq.cuh\"" "#include \"mmvq.cuh\"\n#include \"mmvq_token_batch.hpp\"" ds4_mmvq "${ds4_mmvq}")
string(REPLACE
  "template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false, bool halve_iters = false>"
  "template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false, bool halve_iters = false, bool strata_token_batch = false>"
  ds4_mmvq "${ds4_mmvq}")
string(REPLACE "ids[channel_dst]" "ids[channel_dst + (strata_token_batch ? blockIdx.z*ids_stride : 0)]" ds4_mmvq "${ds4_mmvq}")
string(REPLACE
  "template<ggml_type type, int c_ncols_dst, bool small_k = false, bool halve_iters = false>"
  "template<ggml_type type, int c_ncols_dst, bool small_k = false, bool halve_iters = false, bool strata_token_batch = false>"
  ds4_mmvq "${ds4_mmvq}")
foreach(fused true false)
  string(REPLACE "mul_mat_vec_q<type, c_ncols_dst, ${fused}, small_k, halve_iters>"
    "mul_mat_vec_q<type, c_ncols_dst, ${fused}, small_k, halve_iters, strata_token_batch>" ds4_mmvq "${ds4_mmvq}")
endforeach()
string(REPLACE "template <ggml_type type>\nstatic void mul_mat_vec_q_switch_ncols_dst("
  "template <ggml_type type, bool strata_token_batch = false>\nstatic void mul_mat_vec_q_switch_ncols_dst("
  ds4_mmvq "${ds4_mmvq}")
string(REPLACE "    GGML_ASSERT(ncols_dst <= MMVQ_MAX_BATCH_SIZE);"
  "    GGML_ASSERT(ncols_dst <= MMVQ_MAX_BATCH_SIZE);\n#include \"mmvq_token_batch.inc\""
  ds4_mmvq "${ds4_mmvq}")
string(REPLACE "mul_mat_vec_q_switch_fusion<type, c_ncols_dst, c_small_k, c_halve_iters>"
  "mul_mat_vec_q_switch_fusion<type, c_ncols_dst, c_small_k, c_halve_iters, strata_token_batch>"
  ds4_mmvq "${ds4_mmvq}")
set(ds4_mmvq_generated "${CMAKE_BINARY_DIR}/strata-ds4-mmvq.cu")
set(ds4_mmvq_old "")
if(EXISTS "${ds4_mmvq_generated}")
  file(READ "${ds4_mmvq_generated}" ds4_mmvq_old)
endif()
if(NOT ds4_mmvq_old STREQUAL ds4_mmvq)
  file(WRITE "${ds4_mmvq_generated}" "${ds4_mmvq}")
endif()
get_target_property(ds4_cuda_sources ggml-cuda SOURCES)
set(ds4_mmvq_matches "${ds4_cuda_sources}")
list(FILTER ds4_mmvq_matches INCLUDE REGEX "(^|/)mmvq\\.cu$")
list(LENGTH ds4_mmvq_matches ds4_mmvq_count)
if(NOT ds4_mmvq_count EQUAL 1)
  message(FATAL_ERROR "Expected one mmvq.cu in ggml-cuda")
endif()
list(REMOVE_ITEM ds4_cuda_sources ${ds4_mmvq_matches})
set_property(TARGET ggml-cuda PROPERTY SOURCES "${ds4_cuda_sources};${ds4_mmvq_generated}")
set_source_files_properties("${ds4_mmvq_generated}" TARGET_DIRECTORY ggml-cuda PROPERTIES
  INCLUDE_DIRECTORIES "${STRATA_LLAMA_DIR}/ggml/src/ggml-cuda;${CMAKE_CURRENT_SOURCE_DIR}"
  OBJECT_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/mmvq_token_batch.hpp;${CMAKE_CURRENT_SOURCE_DIR}/mmvq_token_batch.inc")

