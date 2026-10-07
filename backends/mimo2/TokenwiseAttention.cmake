set(mimo_fattn_original "${mimo_source}/ggml/src/ggml-cuda/fattn.cu")
file(SHA256 "${mimo_fattn_original}" mimo_fattn_original_hash)
if(NOT mimo_fattn_original_hash STREQUAL "fbf690a15864ad11edac131823c174b05b3432896c14bc87d2c698929860e1d7")
  message(FATAL_ERROR "Review MiMo FA query-tile diagnostic for changed source")
endif()
file(READ "${mimo_fattn_original}" mimo_fattn)
set(anchor "static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {\n    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;\n    const ggml_tensor * Q = dst->src[0];")
string(FIND "${mimo_fattn}" "${anchor}" found)
if(found EQUAL -1)
  message(FATAL_ERROR "MiMo FA tile anchor not found")
endif()
set(body [=[
    // Same existing specialization as one-token MiMo decode. The launch still
    // covers every query via its grid; no mask views, slicing or host copies.
    if constexpr (DKQ == 192 && DV == 128 && (ncols2 == 8 || ncols2 == 16)) {
        if ((mimo2_tokenwise_mode() & 4) && GGML_CUDA_CC_IS_NVIDIA(cc) &&
            blackwell_mma_available(cc) && Q->ne[1] > 1 && Q->ne[1] <= 8) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 1, ncols2>(ctx, dst);
            return;
        }
    }
]=])
string(REPLACE "${anchor}" "${anchor}\n${body}" mimo_fattn "${mimo_fattn}")
string(PREPEND mimo_fattn "#include \"tokenwise_matmul.hpp\"\n")
set(mimo_fattn_generated "${CMAKE_BINARY_DIR}/strata-mimo2-fattn.cu")
set(previous "")
if(EXISTS "${mimo_fattn_generated}")
  file(READ "${mimo_fattn_generated}" previous)
endif()
if(NOT previous STREQUAL mimo_fattn)
  file(WRITE "${mimo_fattn_generated}" "${mimo_fattn}")
endif()
file(SHA256 "${mimo_fattn_generated}" mimo_fattn_generated_sha256)
get_target_property(sources ggml-cuda SOURCES)
set(matches "${sources}")
list(FILTER matches INCLUDE REGEX "(^|/)fattn\\.cu$")
list(LENGTH matches count)
if(NOT count EQUAL 1)
  message(FATAL_ERROR "Expected one MiMo fattn source")
endif()
list(REMOVE_ITEM sources ${matches})
set_property(TARGET ggml-cuda PROPERTY SOURCES "${sources};${mimo_fattn_generated}")
set_source_files_properties("${mimo_fattn_generated}" TARGET_DIRECTORY ggml-cuda PROPERTIES
  INCLUDE_DIRECTORIES "${mimo_source}/ggml/src/ggml-cuda;${CMAKE_CURRENT_SOURCE_DIR}"
  OBJECT_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tokenwise_matmul.hpp")
string(APPEND mimo_patch_set ",optional-mimo-single-query-fa-tile")
