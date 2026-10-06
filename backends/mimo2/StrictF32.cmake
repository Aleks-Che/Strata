# Optional, MiMo-local correctness patch. Never edit the archive/extracted source
# or link a generated GLM translation unit into this build.
set(mimo_mmf_original "${mimo_source}/ggml/src/ggml-cuda/mmf.cu")
file(SHA256 "${mimo_mmf_original}" mimo_mmf_original_sha256)
if(NOT mimo_mmf_original_sha256 STREQUAL "08bfea71b2f25d273b82f53bbb2181c60b3e9bb22641d3c4ed93067ce4bb3de1")
  message(FATAL_ERROR "Review MiMo strict-F32 patch for the changed mmf.cu")
endif()
file(READ "${mimo_mmf_original}" mimo_mmf)
set(mimo_mmf_anchor "        const size_t * src0_nb, const int src1_ncols, bool mul_mat_id) {")
string(REGEX MATCHALL "const int src1_ncols, bool mul_mat_id\\) \\{" mimo_anchors "${mimo_mmf}")
list(LENGTH mimo_anchors mimo_anchor_count)
string(FIND "${mimo_mmf}" "${mimo_mmf_anchor}" mimo_anchor_position)
if(NOT mimo_anchor_count EQUAL 1 OR mimo_anchor_position EQUAL -1)
  message(FATAL_ERROR "Expected exactly one MiMo MMF dispatch boundary")
endif()
string(REPLACE "${mimo_mmf_anchor}" "${mimo_mmf_anchor}
    // NVIDIA_TF32_OVERRIDE affects cuBLAS, not explicit TF32 MMA instructions.
    // Respect correctness mode in both execution and graph synchronization planning.
    const char * tf32_override = std::getenv(\"NVIDIA_TF32_OVERRIDE\");
    if (type == GGML_TYPE_F32 && GGML_CUDA_CC_IS_NVIDIA(cc) &&
        tf32_override && std::strcmp(tf32_override, \"0\") == 0) {
        return false;
    }
    // For the MiMo BF16-weight baseline, keep FP32 activations consistently
    // across prefill and decode. MMVF already consumes FP32 activations;
    // custom BF16 MMF rounds them, unlike the requested FP32 cuBLAS path.
    const char * compute_override = std::getenv(\"GGML_CUDA_CUBLAS_COMPUTE_TYPE\");
    if (type == GGML_TYPE_BF16 && GGML_CUDA_CC_IS_NVIDIA(cc) &&
        compute_override && std::strcmp(compute_override, \"f32\") == 0) {
        return false;
    }
" mimo_mmf "${mimo_mmf}")
string(PREPEND mimo_mmf "#include <cstdlib>\n#include <cstring>\n")
set(mimo_mmf_generated "${CMAKE_BINARY_DIR}/strata-mimo2-mmf.cu")
set(mimo_mmf_previous "")
if(EXISTS "${mimo_mmf_generated}")
  file(READ "${mimo_mmf_generated}" mimo_mmf_previous)
endif()
if(NOT mimo_mmf_previous STREQUAL mimo_mmf)
  file(WRITE "${mimo_mmf_generated}" "${mimo_mmf}")
endif()
file(SHA256 "${mimo_mmf_generated}" mimo_mmf_generated_sha256)
get_target_property(mimo_cuda_sources ggml-cuda SOURCES)
set(mimo_mmf_sources "${mimo_cuda_sources}")
list(FILTER mimo_mmf_sources INCLUDE REGEX "(^|/)mmf\\.cu$")
list(LENGTH mimo_mmf_sources mimo_mmf_count)
if(NOT mimo_mmf_count EQUAL 1)
  message(FATAL_ERROR "Expected exactly one CUDA mmf.cu translation unit")
endif()
list(REMOVE_ITEM mimo_cuda_sources ${mimo_mmf_sources})
set_property(TARGET ggml-cuda PROPERTY SOURCES "${mimo_cuda_sources};${mimo_mmf_generated}")
set_source_files_properties("${mimo_mmf_generated}" TARGET_DIRECTORY ggml-cuda PROPERTIES
  INCLUDE_DIRECTORIES "${mimo_source}/ggml/src/ggml-cuda")
if(mimo_patch_set STREQUAL "none")
  set(mimo_patch_set "cuda-mmf-respect-f32-overrides")
else()
  string(APPEND mimo_patch_set ",cuda-mmf-respect-f32-overrides")
endif()
