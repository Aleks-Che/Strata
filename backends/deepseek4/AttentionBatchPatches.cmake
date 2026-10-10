# Research build only. The normal backend does not compile this patch.
set(ds4_fa_original "${STRATA_LLAMA_DIR}/ggml/src/ggml-cuda/fattn.cu")
file(SHA256 "${ds4_fa_original}" ds4_fa_hash)
if(NOT ds4_fa_hash STREQUAL "ba15c00dfe95086bee5d021e2b279d3b44089b91934b151023d97ce9bb542ced")
  message(FATAL_ERROR "Review DeepSeek compact attention experiment for changed source")
endif()
file(READ "${ds4_fa_original}" ds4_fa)
set(ds4_compact_guard [=[
static bool strata_ds4_compact_attention(const ggml_tensor * dst) {
    const char * mode = std::getenv("STRATA_DS4_FA_COMPACT");
    const auto * q=dst->src[0], *k=dst->src[1], *v=dst->src[2], *mask=dst->src[3];
    float max_bias, softcap;
    std::memcpy(&max_bias, (const float *)dst->op_params+1, sizeof(float));
    std::memcpy(&softcap, (const float *)dst->op_params+2, sizeof(float));
    return mode && (std::strcmp(mode,"1")==0 || std::strcmp(mode,"2")==0) &&
        max_bias==0.0f && softcap==0.0f &&
        q->type==GGML_TYPE_F32 && k->type==GGML_TYPE_F16 && v->type==GGML_TYPE_F16 &&
        q->ne[0]==512 && v->ne[0]==512 && q->ne[1]>=1 && q->ne[1]<=4 &&
        q->ne[2]==64 && k->ne[2]==1 && v->ne[2]==1 && q->ne[3]==1 && k->ne[3]==1 && v->ne[3]==1 &&
        mask && mask->type==GGML_TYPE_F16 && mask->ne[0]==k->ne[1] && mask->ne[1]>=q->ne[1] &&
        mask->ne[2]==1 && mask->ne[3]==1 && ggml_get_op_params_i32(dst,4)>0 &&
        ggml_is_contiguous(dst);
}
]=])
string(PREPEND ds4_fa "#include <cstdlib>\n#include <cstring>\n")
string(REPLACE "bool ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(" "${ds4_compact_guard}\nbool ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(" ds4_fa "${ds4_fa}")
string(REPLACE "K->ne[1] >= std::max<int64_t>(4096, 2*n_gather)"
  "(strata_ds4_compact_attention(dst) || K->ne[1] >= std::max<int64_t>(4096, 2*n_gather))" ds4_fa "${ds4_fa}")
string(REPLACE "void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {" [=[
void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);
    const char * mode=std::getenv("STRATA_DS4_FA_COMPACT");
    if (strata_ds4_compact_attention(dst) && mode && std::strcmp(mode,"2")==0 &&
        GGML_CUDA_CC_IS_NVIDIA(ggml_cuda_info().devices[ctx.device].cc) &&
        turing_mma_available(ggml_cuda_info().devices[ctx.device].cc) && dst->src[0]->ne[1]>1) {
        const auto * q=dst->src[0], *mask=dst->src[3];
        for (int64_t i=0; i<q->ne[1]; ++i) {
            ggml_tensor one_q=*q, one_mask=*mask, one_dst=*dst;
            one_q.ne[1]=1; one_q.data=(char *)q->data+i*q->nb[1];
            one_mask.ne[1]=1; one_mask.data=(char *)mask->data+i*mask->nb[1];
            one_dst.ne[2]=1; one_dst.data=(char *)dst->data+i*dst->nb[2];
            one_dst.src[0]=&one_q; one_dst.src[3]=&one_mask;
            ggml_cuda_flash_attn_ext_mma_f16(ctx,&one_dst);
        }
        return;
    }
]=] ds4_fa "${ds4_fa}")
set(ds4_fa_generated "${CMAKE_BINARY_DIR}/strata-ds4-fattn.cu")
file(WRITE "${ds4_fa_generated}" "${ds4_fa}")
get_target_property(ds4_cuda_sources ggml-cuda SOURCES)
set(ds4_fa_matches "${ds4_cuda_sources}")
list(FILTER ds4_fa_matches INCLUDE REGEX "(^|/)fattn\\.cu$")
list(LENGTH ds4_fa_matches ds4_fa_count)
if(NOT ds4_fa_count EQUAL 1)
  message(FATAL_ERROR "Expected one fattn.cu in ggml-cuda")
endif()
list(REMOVE_ITEM ds4_cuda_sources ${ds4_fa_matches})
set_property(TARGET ggml-cuda PROPERTY SOURCES "${ds4_cuda_sources};${ds4_fa_generated}")
set_source_files_properties("${ds4_fa_generated}" TARGET_DIRECTORY ggml-cuda PROPERTIES
  INCLUDE_DIRECTORIES "${STRATA_LLAMA_DIR}/ggml/src/ggml-cuda")

# Raw SWA has a proven bound. HCA additionally includes at most every slot
# of its compressed input buffer; this conservative bound never prunes keys.
set(ds4_model_original "${STRATA_LLAMA_DIR}/src/models/deepseek4.cpp")
file(SHA256 "${ds4_model_original}" ds4_model_hash)
if(NOT ds4_model_hash STREQUAL "6bad275618f2b3f6f671fe4e6506c7fcfa06083d4606456222b406792539e943")
  message(FATAL_ERROR "Review DeepSeek raw attention bound for changed source")
endif()
file(READ "${ds4_model_original}" ds4_model)
set(ds4_raw_call "build_attn_mha(q, k, k, nullptr, kq_mask, sinks, nullptr, 0, kq_scale, il)")
string(FIND "${ds4_model}" "${ds4_raw_call}" ds4_raw_found)
if(ds4_raw_found EQUAL -1)
  message(FATAL_ERROR "Missing DeepSeek raw SWA attention call")
endif()
string(REPLACE "${ds4_raw_call}" "build_attn_mha(q, k, k, nullptr, kq_mask, sinks, nullptr, (q->ne[2]<=4 && std::getenv(\"STRATA_DS4_FA_COMPACT\") && (std::strcmp(std::getenv(\"STRATA_DS4_FA_COMPACT\"), \"1\")==0 || std::strcmp(std::getenv(\"STRATA_DS4_FA_COMPACT\"), \"2\")==0)) ? hparams.n_swa : 0, kq_scale, il)" ds4_model "${ds4_model}")
set(ds4_hca_call "build_attn_mha(q, k_all, k_all, nullptr, kq_mask, sinks, nullptr, 0, kq_scale, il)")
string(FIND "${ds4_model}" "${ds4_hca_call}" ds4_hca_found)
if(ds4_hca_found EQUAL -1)
  message(FATAL_ERROR "Missing DeepSeek HCA attention call")
endif()
set(ds4_hca_bound [=[
build_attn_mha(q, k_all, k_all, nullptr, kq_mask, sinks, nullptr,
            (q->ne[2]<=4 && std::getenv("STRATA_DS4_HCA_COMPACT") &&
             std::strcmp(std::getenv("STRATA_DS4_HCA_COMPACT"), "1")==0 &&
             std::getenv("STRATA_DS4_FA_COMPACT") &&
             (std::strcmp(std::getenv("STRATA_DS4_FA_COMPACT"), "1")==0 ||
              std::strcmp(std::getenv("STRATA_DS4_FA_COMPACT"), "2")==0)) ?
                int64_t(hparams.n_swa) + n_hca : 0, kq_scale, il)
]=])
string(REPLACE "${ds4_hca_call}" "${ds4_hca_bound}" ds4_model "${ds4_model}")
string(PREPEND ds4_model "#include <cstdlib>\n#include <cstring>\n")
file(WRITE "${CMAKE_BINARY_DIR}/strata-ds4-attention-model.cpp" "${ds4_model}")
get_target_property(ds4_llama_sources llama SOURCES)
list(FILTER ds4_llama_sources EXCLUDE REGEX "(^|/)models/deepseek4\\.cpp$")
set_property(TARGET llama PROPERTY SOURCES "${ds4_llama_sources};${CMAKE_BINARY_DIR}/strata-ds4-attention-model.cpp")
set_source_files_properties("${CMAKE_BINARY_DIR}/strata-ds4-attention-model.cpp" TARGET_DIRECTORY llama PROPERTIES
  INCLUDE_DIRECTORIES "${STRATA_LLAMA_DIR}/src/models;${STRATA_LLAMA_DIR}/src")
