# MiMo baseline keeps FP32 activations instead of native Blackwell FP4 MMQ.
# In 96-case admission, native MXFP4 MMQ reached NMSE 0.01025 against the
# same dequantized weights with FP32 inputs (budget 0.0001). MMVQ is retained.
# This is a precision policy, not a claim that native FP4 MMA is malfunctioning.
set(mimo_mmq_original "${mimo_source}/ggml/src/ggml-cuda/mmq.cu")
file(SHA256 "${mimo_mmq_original}" mimo_mmq_original_sha256)
if(NOT mimo_mmq_original_sha256 STREQUAL "8acdffe7ccf49698083dfc3e32ae3ef015fb99d6a790adc860344aed401fee08")
  message(FATAL_ERROR "Review MiMo MXFP4 precision policy for the changed mmq.cu")
endif()
file(READ "${mimo_mmq_original}" mimo_mmq)
set(mimo_mmq_anchor "bool ggml_cuda_should_use_mmq(enum ggml_type type, int cc, int64_t ne11, int64_t n_experts) {")
string(FIND "${mimo_mmq}" "${mimo_mmq_anchor}" mimo_mmq_found)
if(mimo_mmq_found EQUAL -1)
  message(FATAL_ERROR "MiMo MXFP4 dispatch boundary not found")
endif()
string(REPLACE "${mimo_mmq_anchor}" "${mimo_mmq_anchor}
    // Native Blackwell MXFP4 MMQ quantizes activations to FP4. The MiMo
    // correctness baseline instead uses GPU dequantization + cuBLAS here.
    // MMVQ remains selected upstream for small batches; other types unchanged.
    if (type == GGML_TYPE_MXFP4 && blackwell_mma_available(cc)) {
        return false;
    }
" mimo_mmq "${mimo_mmq}")
set(mimo_mmq_generated "${CMAKE_BINARY_DIR}/strata-mimo2-mmq.cu")
set(mimo_mmq_previous "")
if(EXISTS "${mimo_mmq_generated}")
  file(READ "${mimo_mmq_generated}" mimo_mmq_previous)
endif()
if(NOT mimo_mmq_previous STREQUAL mimo_mmq)
  file(WRITE "${mimo_mmq_generated}" "${mimo_mmq}")
endif()
file(SHA256 "${mimo_mmq_generated}" mimo_mmq_generated_sha256)
get_target_property(mimo_cuda_sources ggml-cuda SOURCES)
set(mimo_mmq_sources "${mimo_cuda_sources}")
list(FILTER mimo_mmq_sources INCLUDE REGEX "(^|/)mmq\\.cu$")
list(LENGTH mimo_mmq_sources mimo_mmq_count)
if(NOT mimo_mmq_count EQUAL 1)
  message(FATAL_ERROR "Expected exactly one CUDA mmq.cu translation unit")
endif()
list(REMOVE_ITEM mimo_cuda_sources ${mimo_mmq_sources})
set_property(TARGET ggml-cuda PROPERTY SOURCES "${mimo_cuda_sources};${mimo_mmq_generated}")
set_source_files_properties("${mimo_mmq_generated}" TARGET_DIRECTORY ggml-cuda PROPERTIES
  INCLUDE_DIRECTORIES "${mimo_source}/ggml/src/ggml-cuda")
if(mimo_patch_set STREQUAL "none")
  set(mimo_patch_set "cuda-mxfp4-blackwell-precision")
else()
  string(APPEND mimo_patch_set ",cuda-mxfp4-blackwell-precision")
endif()
