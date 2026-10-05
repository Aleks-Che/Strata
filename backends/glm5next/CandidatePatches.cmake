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

# Small verification windows must use the same per-token matvec arithmetic as
# ordinary decode. Batched GEMM/vector paths round activations differently;
# at IQ3 this can change routing and target logits even before any rollback.
set(glm_cuda_original "${glm_source}/ggml/src/ggml-cuda/ggml-cuda.cu")
file(SHA256 "${glm_cuda_original}" glm_cuda_hash)
if(NOT glm_cuda_hash STREQUAL "eee8c85f32f45c47d0f84210d82109598c0dd32c73b3d4605972a62569af7034")
  message(FATAL_ERROR "Review GLM small-batch CUDA dispatch patch for changed source")
endif()
file(READ "${glm_cuda_original}" glm_cuda_dispatch)
set(glm_dense_anchor "static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {")
string(REPLACE "${glm_dense_anchor}" "${glm_dense_anchor}
    if (getenv(\"STRATA_GLM_TOKENWISE_MATMUL\") && src1->ne[1] > 1 && src1->ne[1] <= 4) {
        for (int64_t i = 0; i < src1->ne[1]; ++i) {
            ggml_tensor x = *src1, y = *dst;
            x.ne[1] = y.ne[1] = 1;
            x.data = (char *) src1->data + i*src1->nb[1];
            y.data = (char *) dst->data + i*dst->nb[1];
            y.src[1] = &x;
            ggml_cuda_mul_mat(ctx, src0, &x, &y);
        }
        return;
    }
" glm_cuda_dispatch "${glm_cuda_dispatch}")
set(glm_id_anchor "static void ggml_cuda_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {")
string(REPLACE "${glm_id_anchor}" "${glm_id_anchor}
    if (getenv(\"STRATA_GLM_TOKENWISE_MATMUL\") && ggml_is_quantized(dst->src[0]->type) && dst->ne[2] > 1 && dst->ne[2] <= 4) {
        for (int64_t i = 0; i < dst->ne[2]; ++i) {
            ggml_tensor x = *dst->src[1], ids = *dst->src[2], y = *dst;
            x.ne[2] = y.ne[2] = ids.ne[1] = 1;
            x.data = (char *) dst->src[1]->data + i*dst->src[1]->nb[2];
            ids.data = (char *) dst->src[2]->data + i*dst->src[2]->nb[1];
            y.data = (char *) dst->data + i*dst->nb[2];
            y.src[1] = &x; y.src[2] = &ids;
            ggml_cuda_mul_mat_id(ctx, &y);
        }
        return;
    }
" glm_cuda_dispatch "${glm_cuda_dispatch}")
set(glm_sync_anchor "static bool ggml_cuda_mul_mat_id_needs_sync(const ggml_tensor * dst, const int cc) {")
string(REPLACE "${glm_sync_anchor}" "${glm_sync_anchor}
    if (getenv(\"STRATA_GLM_TOKENWISE_MATMUL\") && ggml_is_quantized(dst->src[0]->type) && dst->ne[2] > 1 && dst->ne[2] <= 4) {
        ggml_tensor x = *dst->src[1], y = *dst;
        x.ne[2] = y.ne[2] = 1; y.src[1] = &x;
        return ggml_cuda_mul_mat_id_needs_sync(&y, cc);
    }
" glm_cuda_dispatch "${glm_cuda_dispatch}")
set(glm_cuda_generated "${CMAKE_BINARY_DIR}/strata-glm-cuda.cu")
include(ShortBatchPatches.cmake)
set(glm_cuda_old "")
if(EXISTS "${glm_cuda_generated}")
  file(READ "${glm_cuda_generated}" glm_cuda_old)
endif()
if(NOT glm_cuda_old STREQUAL glm_cuda_dispatch)
  file(WRITE "${glm_cuda_generated}" "${glm_cuda_dispatch}")
endif()
get_target_property(glm_cuda_sources ggml-cuda SOURCES)
set(glm_dispatch_sources "${glm_cuda_sources}")
list(FILTER glm_dispatch_sources INCLUDE REGEX "(^|/)ggml-cuda\\.cu$")
list(LENGTH glm_dispatch_sources glm_dispatch_count)
if(NOT glm_dispatch_count EQUAL 1)
  message(FATAL_ERROR "Expected one ggml-cuda.cu source")
endif()
list(REMOVE_ITEM glm_cuda_sources ${glm_dispatch_sources})
set_property(TARGET ggml-cuda PROPERTY SOURCES "${glm_cuda_sources};${glm_cuda_generated}")
set_source_files_properties("${glm_cuda_generated}" TARGET_DIRECTORY ggml-cuda PROPERTIES
  INCLUDE_DIRECTORIES "${glm_source}/ggml/src/ggml-cuda;${CMAKE_CURRENT_SOURCE_DIR}"
  OBJECT_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/mmvq_token_batch.hpp")
string(APPEND glm_candidate_patches ",optional-tokenwise-small-batch-matmul,optional-token-batch-mmvq")
