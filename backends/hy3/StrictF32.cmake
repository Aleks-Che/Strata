# Optional, Hy3-local correctness patch. Never edit the archive/extracted source
# or link a generated GLM translation unit into this build.
set(hy3_mmf_original "${hy3_source}/ggml/src/ggml-cuda/mmf.cu")
file(SHA256 "${hy3_mmf_original}" hy3_mmf_original_sha256)
if(NOT hy3_mmf_original_sha256 STREQUAL "08bfea71b2f25d273b82f53bbb2181c60b3e9bb22641d3c4ed93067ce4bb3de1")
  message(FATAL_ERROR "Review Hy3 strict-F32 patch for the changed mmf.cu")
endif()
file(READ "${hy3_mmf_original}" hy3_mmf)
set(hy3_mmf_anchor "        const size_t * src0_nb, const int src1_ncols, bool mul_mat_id) {")
string(REGEX MATCHALL "const int src1_ncols, bool mul_mat_id\\) \\{" hy3_anchors "${hy3_mmf}")
list(LENGTH hy3_anchors hy3_anchor_count)
string(FIND "${hy3_mmf}" "${hy3_mmf_anchor}" hy3_anchor_position)
if(NOT hy3_anchor_count EQUAL 1 OR hy3_anchor_position EQUAL -1)
  message(FATAL_ERROR "Expected exactly one Hy3 MMF dispatch boundary")
endif()
string(REPLACE "${hy3_mmf_anchor}" "${hy3_mmf_anchor}
    // NVIDIA_TF32_OVERRIDE affects cuBLAS, not explicit TF32 MMA instructions.
    // Respect correctness mode in both execution and graph synchronization planning.
    const char * tf32_override = std::getenv(\"NVIDIA_TF32_OVERRIDE\");
    if (type == GGML_TYPE_F32 && GGML_CUDA_CC_IS_NVIDIA(cc) &&
        tf32_override && std::strcmp(tf32_override, \"0\") == 0) {
        return false;
    }
" hy3_mmf "${hy3_mmf}")
string(PREPEND hy3_mmf "#include <cstdlib>\n#include <cstring>\n")
set(hy3_mmf_generated "${CMAKE_BINARY_DIR}/strata-hy3-mmf.cu")
set(hy3_mmf_previous "")
if(EXISTS "${hy3_mmf_generated}")
  file(READ "${hy3_mmf_generated}" hy3_mmf_previous)
endif()
if(NOT hy3_mmf_previous STREQUAL hy3_mmf)
  file(WRITE "${hy3_mmf_generated}" "${hy3_mmf}")
endif()
file(SHA256 "${hy3_mmf_generated}" hy3_mmf_generated_sha256)
get_target_property(hy3_cuda_sources ggml-cuda SOURCES)
set(hy3_mmf_sources "${hy3_cuda_sources}")
list(FILTER hy3_mmf_sources INCLUDE REGEX "(^|/)mmf\\.cu$")
list(LENGTH hy3_mmf_sources hy3_mmf_count)
if(NOT hy3_mmf_count EQUAL 1)
  message(FATAL_ERROR "Expected exactly one CUDA mmf.cu translation unit")
endif()
list(REMOVE_ITEM hy3_cuda_sources ${hy3_mmf_sources})
set_property(TARGET ggml-cuda PROPERTY SOURCES "${hy3_cuda_sources};${hy3_mmf_generated}")
set_source_files_properties("${hy3_mmf_generated}" TARGET_DIRECTORY ggml-cuda PROPERTIES
  INCLUDE_DIRECTORIES "${hy3_source}/ggml/src/ggml-cuda")
if(hy3_patch_set STREQUAL "none")
  set(hy3_patch_set "cuda-f32-mmf-respect-tf32-override")
else()
  string(APPEND hy3_patch_set ",cuda-f32-mmf-respect-tf32-override")
endif()
