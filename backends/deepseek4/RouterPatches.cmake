# Keep short BF16 router batches on MMVF, which retains F32 activations.
# The ordinary path is unchanged unless STRATA_DS4_ROUTER_MMVF=1 at startup.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/router_mmvf.inc")
file(SHA256 "${STRATA_LLAMA_DIR}/ggml/src/ggml-cuda/ggml-cuda.cu" ds4_router_hash)
if(NOT ds4_router_hash STREQUAL "af947e45946d0c47cfae9e056f589a25b61892e4273decf0a893e2dcad48d065")
  message(FATAL_ERROR "Review DeepSeek router dispatch for changed CUDA source")
endif()
set(ds4_router_anchor "    if (ggml_cuda_should_use_mmvf(src0->type, cc, src0->ne, src0->nb, ne11)) {")
string(FIND "${cuda_source}" "${ds4_router_anchor}" ds4_router_found)
if(ds4_router_found EQUAL -1)
  message(FATAL_ERROR "Missing DeepSeek router dispatch anchor")
endif()
file(READ "${CMAKE_CURRENT_LIST_DIR}/router_mmvf.inc" ds4_router_code)
string(REPLACE "${ds4_router_anchor}" "${ds4_router_code}\n${ds4_router_anchor}" cuda_source "${cuda_source}")
