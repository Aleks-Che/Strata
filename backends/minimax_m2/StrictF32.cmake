# Optional, MiniMax-local correctness patch. Never edit the archive/extracted source
# or link a generated GLM translation unit into this build.
set(mm27_mmf_original "${mm27_source}/ggml/src/ggml-cuda/mmf.cu")
file(SHA256 "${mm27_mmf_original}" mm27_mmf_original_sha256)
if(NOT mm27_mmf_original_sha256 STREQUAL "08bfea71b2f25d273b82f53bbb2181c60b3e9bb22641d3c4ed93067ce4bb3de1")
  message(FATAL_ERROR "Review MiniMax strict-F32 patch for the changed mmf.cu")
endif()
file(READ "${mm27_mmf_original}" mm27_mmf)
set(mm27_mmf_anchor "        const size_t * src0_nb, const int src1_ncols, bool mul_mat_id) {")
string(REGEX MATCHALL "const int src1_ncols, bool mul_mat_id\\) \\{" mm27_anchors "${mm27_mmf}")
list(LENGTH mm27_anchors mm27_anchor_count)
string(FIND "${mm27_mmf}" "${mm27_mmf_anchor}" mm27_anchor_position)
if(NOT mm27_anchor_count EQUAL 1 OR mm27_anchor_position EQUAL -1)
  message(FATAL_ERROR "Expected exactly one MiniMax MMF dispatch boundary")
endif()
string(REPLACE "${mm27_mmf_anchor}" "${mm27_mmf_anchor}
    // NVIDIA_TF32_OVERRIDE affects cuBLAS, not explicit TF32 MMA instructions.
    // Respect correctness mode in both execution and graph synchronization planning.
    const char * tf32_override = std::getenv(\"NVIDIA_TF32_OVERRIDE\");
    if (type == GGML_TYPE_F32 && GGML_CUDA_CC_IS_NVIDIA(cc) &&
        tf32_override && std::strcmp(tf32_override, \"0\") == 0) {
        return false;
    }
" mm27_mmf "${mm27_mmf}")
string(PREPEND mm27_mmf "#include <cstdlib>\n#include <cstring>\n")
set(mm27_mmf_generated "${CMAKE_BINARY_DIR}/strata-mm27-mmf.cu")
set(mm27_mmf_previous "")
if(EXISTS "${mm27_mmf_generated}")
  file(READ "${mm27_mmf_generated}" mm27_mmf_previous)
endif()
if(NOT mm27_mmf_previous STREQUAL mm27_mmf)
  file(WRITE "${mm27_mmf_generated}" "${mm27_mmf}")
endif()
file(SHA256 "${mm27_mmf_generated}" mm27_mmf_generated_sha256)
get_target_property(mm27_cuda_sources ggml-cuda SOURCES)
set(mm27_mmf_sources "${mm27_cuda_sources}")
list(FILTER mm27_mmf_sources INCLUDE REGEX "(^|/)mmf\\.cu$")
list(LENGTH mm27_mmf_sources mm27_mmf_count)
if(NOT mm27_mmf_count EQUAL 1)
  message(FATAL_ERROR "Expected exactly one CUDA mmf.cu translation unit")
endif()
list(REMOVE_ITEM mm27_cuda_sources ${mm27_mmf_sources})
set_property(TARGET ggml-cuda PROPERTY SOURCES "${mm27_cuda_sources};${mm27_mmf_generated}")
set_source_files_properties("${mm27_mmf_generated}" TARGET_DIRECTORY ggml-cuda PROPERTIES
  INCLUDE_DIRECTORIES "${mm27_source}/ggml/src/ggml-cuda")
if(mm27_patch_set STREQUAL "none")
  set(mm27_patch_set "cuda-f32-mmf-respect-tf32-override")
else()
  string(APPEND mm27_patch_set ",cuda-f32-mmf-respect-tf32-override")
endif()
