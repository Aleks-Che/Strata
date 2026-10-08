# Opt-in numerical reference: retain F32 activations for Q4_K/Q6_K matmuls.
# Normal MMVQ/MMQ additionally quantize activations to Q8. Apply consistently
# to fusion, dispatch and the CUDA-graph synchronization planner.
# The reviewed original dispatch SHA is checked by RoutedStrides.cmake first.
if(NOT STRATA_MM27_ROUTED_STRIDES)
  message(FATAL_ERROR "MiniMax QuantF32 requires the reviewed routed-strides generator")
endif()
file(READ "${mm27_dispatch_generated}" mm27_quant_dispatch)
function(mm27_replace_once anchor replacement)
  string(FIND "${mm27_quant_dispatch}" "${anchor}" found)
  if(found EQUAL -1)
    message(FATAL_ERROR "MiniMax QuantF32 anchor missing: ${anchor}")
  endif()
  string(REPLACE "${anchor}" "" remainder "${mm27_quant_dispatch}")
  string(LENGTH "${mm27_quant_dispatch}" before)
  string(LENGTH "${remainder}" after)
  string(LENGTH "${anchor}" length)
  math(EXPR removed "${before}-${after}")
  if(NOT removed EQUAL length)
    message(FATAL_ERROR "MiniMax QuantF32 anchor is ambiguous")
  endif()
  string(REPLACE "${anchor}" "${replacement}" changed "${mm27_quant_dispatch}")
  set(mm27_quant_dispatch "${changed}" PARENT_SCOPE)
endfunction()
set(anchor "static bool ggml_cuda_should_fuse_mul_mat_vec_q(const ggml_tensor * tensor) {")
mm27_replace_once("${anchor}" "static bool strata_mm27_quant_f32(const ggml_type type) {
    static const bool enabled = []() {
        const char * value = std::getenv(\"STRATA_MM27_QUANT_F32\");
        return value && std::strcmp(value, \"1\") == 0;
    }();
    return enabled && (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q6_K);
}

${anchor}
    if (strata_mm27_quant_f32(tensor->src[0]->type)) return false;")
set(anchor "static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {")
mm27_replace_once("${anchor}" "${anchor}
    if (strata_mm27_quant_f32(src0->type)) {
        ggml_cuda_mul_mat_cublas(ctx, src0, src1, dst);
        return;
    }")
set(anchor "static bool ggml_cuda_mul_mat_id_needs_sync(const ggml_tensor * dst, const int cc) {")
mm27_replace_once("${anchor}" "${anchor}
    if (strata_mm27_quant_f32(dst->src[0]->type)) return true;")
set(anchor "    // [TAG_MUL_MAT_ID_CUDA_GRAPHS]
    if (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {")
mm27_replace_once("${anchor}" "    // [TAG_MUL_MAT_ID_CUDA_GRAPHS]
    if (!strata_mm27_quant_f32(src0->type) && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {")
string(PREPEND mm27_quant_dispatch "#include <cstdlib>\n#include <cstring>\n")
# Separate output avoids recompiling on each configure after RoutedStrides
# regenerates its own immutable intermediate.
set(mm27_quant_generated "${CMAKE_BINARY_DIR}/strata-mm27-quant-f32-cuda.cu")
set(previous "")
if(EXISTS "${mm27_quant_generated}")
  file(READ "${mm27_quant_generated}" previous)
endif()
if(NOT previous STREQUAL mm27_quant_dispatch)
  file(WRITE "${mm27_quant_generated}" "${mm27_quant_dispatch}")
endif()
file(SHA256 "${mm27_quant_generated}" mm27_quant_dispatch_sha256)
get_target_property(sources ggml-cuda SOURCES)
list(REMOVE_ITEM sources "${mm27_dispatch_generated}")
set_property(TARGET ggml-cuda PROPERTY SOURCES "${sources};${mm27_quant_generated}")
set_source_files_properties("${mm27_quant_generated}" TARGET_DIRECTORY ggml-cuda PROPERTIES
  INCLUDE_DIRECTORIES "${mm27_source}/ggml/src;${mm27_source}/ggml/src/ggml-cuda")
string(APPEND mm27_patch_set ",cuda-minimax-quant-f32-opt-in")
