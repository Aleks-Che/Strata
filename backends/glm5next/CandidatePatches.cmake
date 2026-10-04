# Keep the audited archive untouched. Compile a generated translation unit only
# in this isolated GLM build. Refuse a different candidate or changed patch site.
set(glm_mmf_original "${glm_source}/ggml/src/ggml-cuda/mmf.cu")
file(SHA256 "${glm_mmf_original}" glm_mmf_original_sha256)
if(NOT glm_mmf_original_sha256 STREQUAL "08bfea71b2f25d273b82f53bbb2181c60b3e9bb22641d3c4ed93067ce4bb3de1")
  message(FATAL_ERROR "Unsupported GLM candidate mmf.cu: review the TF32 override patch for this source")
endif()
file(READ "${glm_mmf_original}" glm_mmf)
set(glm_mmf_anchor "        const size_t * src0_nb, const int src1_ncols, bool mul_mat_id) {")
string(FIND "${glm_mmf}" "${glm_mmf_anchor}" glm_mmf_found)
if(glm_mmf_found EQUAL -1)
  message(FATAL_ERROR "GLM candidate MMF selection boundary not found")
endif()
string(REPLACE "${glm_mmf_anchor}" "${glm_mmf_anchor}
    // Strata GLM correctness mode: F32 MMF emits explicit tf32 MMA instructions.
    // NVIDIA_TF32_OVERRIDE controls cuBLAS but cannot disable those instructions.
    // Both execution and CUDA-graph sync planning use this selection predicate.
    const char * tf32_override = std::getenv(\"NVIDIA_TF32_OVERRIDE\");
    if (type == GGML_TYPE_F32 && GGML_CUDA_CC_IS_NVIDIA(cc) &&
        tf32_override && std::strcmp(tf32_override, \"0\") == 0) {
        return false;
    }
" glm_mmf "${glm_mmf}")
string(PREPEND glm_mmf "#include <cstdlib>\n#include <cstring>\n")
set(glm_mmf_generated "${CMAKE_BINARY_DIR}/strata-glm-mmf.cu")
set(glm_mmf_old "")
if(EXISTS "${glm_mmf_generated}")
  file(READ "${glm_mmf_generated}" glm_mmf_old)
endif()
if(NOT glm_mmf_old STREQUAL glm_mmf)
  file(WRITE "${glm_mmf_generated}" "${glm_mmf}")
endif()
file(SHA256 "${glm_mmf_generated}" glm_mmf_generated_sha256)
get_target_property(glm_cuda_sources ggml-cuda SOURCES)
set(glm_mmf_sources "${glm_cuda_sources}")
list(FILTER glm_mmf_sources INCLUDE REGEX "(^|/)mmf\\.cu$")
list(LENGTH glm_mmf_sources glm_mmf_count)
if(NOT glm_mmf_count EQUAL 1)
  message(FATAL_ERROR "Expected one GLM CUDA mmf.cu translation unit")
endif()
list(FILTER glm_cuda_sources EXCLUDE REGEX "(^|/)mmf\\.cu$")
set_property(TARGET ggml-cuda PROPERTY SOURCES "${glm_cuda_sources};${glm_mmf_generated}")
set_source_files_properties("${glm_mmf_generated}" TARGET_DIRECTORY ggml-cuda PROPERTIES
  INCLUDE_DIRECTORIES "${glm_source}/ggml/src/ggml-cuda")
set(glm_candidate_patches "cuda-f32-mmf-respect-tf32-override")
