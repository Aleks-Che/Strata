# Optional, Step-local correctness patch. Never edit the archive/extracted source
# or link a generated GLM translation unit into this build.
set(step_mmf_original "${step_source}/ggml/src/ggml-cuda/mmf.cu")
file(SHA256 "${step_mmf_original}" step_mmf_original_sha256)
if(NOT step_mmf_original_sha256 STREQUAL "08bfea71b2f25d273b82f53bbb2181c60b3e9bb22641d3c4ed93067ce4bb3de1")
  message(FATAL_ERROR "Review Step strict-F32 patch for the changed mmf.cu")
endif()
file(READ "${step_mmf_original}" step_mmf)
set(step_mmf_anchor "        const size_t * src0_nb, const int src1_ncols, bool mul_mat_id) {")
string(REGEX MATCHALL "const int src1_ncols, bool mul_mat_id\\) \\{" step_anchors "${step_mmf}")
list(LENGTH step_anchors step_anchor_count)
string(FIND "${step_mmf}" "${step_mmf_anchor}" step_anchor_position)
if(NOT step_anchor_count EQUAL 1 OR step_anchor_position EQUAL -1)
  message(FATAL_ERROR "Expected exactly one Step MMF dispatch boundary")
endif()
string(REPLACE "${step_mmf_anchor}" "${step_mmf_anchor}
    // NVIDIA_TF32_OVERRIDE affects cuBLAS, not explicit TF32 MMA instructions.
    // Respect correctness mode in both execution and graph synchronization planning.
    const char * tf32_override = std::getenv(\"NVIDIA_TF32_OVERRIDE\");
    if (type == GGML_TYPE_F32 && GGML_CUDA_CC_IS_NVIDIA(cc) &&
        tf32_override && std::strcmp(tf32_override, \"0\") == 0) {
        return false;
    }
" step_mmf "${step_mmf}")
string(PREPEND step_mmf "#include <cstdlib>\n#include <cstring>\n")
set(step_mmf_generated "${CMAKE_BINARY_DIR}/strata-step35-mmf.cu")
set(step_mmf_previous "")
if(EXISTS "${step_mmf_generated}")
  file(READ "${step_mmf_generated}" step_mmf_previous)
endif()
if(NOT step_mmf_previous STREQUAL step_mmf)
  file(WRITE "${step_mmf_generated}" "${step_mmf}")
endif()
file(SHA256 "${step_mmf_generated}" step_mmf_generated_sha256)
get_target_property(step_cuda_sources ggml-cuda SOURCES)
set(step_mmf_sources "${step_cuda_sources}")
list(FILTER step_mmf_sources INCLUDE REGEX "(^|/)mmf\\.cu$")
list(LENGTH step_mmf_sources step_mmf_count)
if(NOT step_mmf_count EQUAL 1)
  message(FATAL_ERROR "Expected exactly one CUDA mmf.cu translation unit")
endif()
list(REMOVE_ITEM step_cuda_sources ${step_mmf_sources})
set_property(TARGET ggml-cuda PROPERTY SOURCES "${step_cuda_sources};${step_mmf_generated}")
set_source_files_properties("${step_mmf_generated}" TARGET_DIRECTORY ggml-cuda PROPERTIES
  INCLUDE_DIRECTORIES "${step_source}/ggml/src/ggml-cuda")
set(step_patch_set "cuda-f32-mmf-respect-tf32-override")
