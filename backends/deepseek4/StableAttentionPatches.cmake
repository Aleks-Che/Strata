# Private launch helper and one explicit MMA instantiation. Renaming the host
# helpers avoids ODR collisions with launch_fattn instantiations used by other
# attention kernels. Optional variants are private; the pinned dependency stays unchanged.
set(ds4_cuda_dir "${STRATA_LLAMA_DIR}/ggml/src/ggml-cuda")
file(SHA256 "${ds4_cuda_dir}/fattn-common.cuh" ds4_common_hash)
file(SHA256 "${ds4_cuda_dir}/fattn-mma-f16.cuh" ds4_mma_hash)
if(NOT ds4_common_hash STREQUAL "c3620ddda02049b1dab2f9e12dd9824d2e10a81d356a8354a90f199a83e9d9e2" OR
   NOT ds4_mma_hash STREQUAL "39f766ffc260595f1483adbf4c5b3741c43a84691edba703d2cac6fa80a97a32")
  message(FATAL_ERROR "Review DeepSeek stable stream-K for changed CUDA headers")
endif()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/stable_stream_k.inc")
file(READ "${ds4_cuda_dir}/fattn-common.cuh" ds4_common)
file(READ "${CMAKE_CURRENT_LIST_DIR}/stable_stream_k.inc" ds4_stable_code)
set(ds4_stable_anchor "        if (ntiles_dst % blocks_num.x != 0) { // Fixup is only needed if the SMs work on fractional tiles.")
string(FIND "${ds4_common}" "${ds4_stable_anchor}" ds4_stable_found)
if(ds4_stable_found EQUAL -1)
  message(FATAL_ERROR "Missing DeepSeek stream-K launch anchor")
endif()
string(REPLACE "${ds4_stable_anchor}" "${ds4_stable_code}\n${ds4_stable_anchor}" ds4_common "${ds4_common}")
string(REPLACE "void launch_fattn(" "void strata_ds4_launch_fattn(" ds4_common "${ds4_common}")
if(STRATA_DEEPSEEK_COST_EXPERIMENT)
  include(AttentionStageProfilePatches.cmake)
endif()
string(PREPEND ds4_common "#include \"stable_stream_k.hpp\"\n#include <cstdlib>\n#include <cstring>\n#include <cstdio>\n#include <mutex>\n#include <set>\n#include <tuple>\n")
set(ds4_stable_dir "${CMAKE_BINARY_DIR}/ds4-stable-attention")
file(MAKE_DIRECTORY "${ds4_stable_dir}")
file(WRITE "${ds4_stable_dir}/strata-fattn-common.cuh" "${ds4_common}")
file(READ "${ds4_cuda_dir}/fattn-mma-f16.cuh" ds4_mma)
include(AttentionEmptyTilePatches.cmake)
string(REPLACE "#include \"fattn-common.cuh\"" "#include \"strata-fattn-common.cuh\"" ds4_mma "${ds4_mma}")
string(REPLACE "launch_fattn<DV, ncols1, ncols2>" "strata_ds4_launch_fattn<DV, ncols1, ncols2>" ds4_mma "${ds4_mma}")
string(REPLACE "ggml_cuda_flash_attn_ext_mma_f16_case" "strata_ds4_mma_case" ds4_mma "${ds4_mma}")
file(WRITE "${ds4_stable_dir}/strata-fattn-mma.cuh" "${ds4_mma}")
file(WRITE "${ds4_stable_dir}/strata-fattn-stable.cu" [=[
#include "strata-fattn-mma.cuh"
void strata_ds4_stable_attention(ggml_backend_cuda_context &ctx,ggml_tensor *dst) {
    // Gather and stream-K fixup storage otherwise grows with the entire
    // prefill. Bound temporary memory before entering the upstream wrapper.
    // Every query keeps its own mask, inputs, and serial K partition.
    const auto *q=dst->src[0],*mask=dst->src[3];
    const char *mode=std::getenv("STRATA_DS4_FA_COMPACT");
    constexpr int chunk=128;
    if(mode && std::strcmp(mode,"4")==0 && q->ne[1]>chunk) {
        for(int64_t begin=0;begin<q->ne[1];begin+=chunk) {
            const int64_t count=std::min<int64_t>(chunk,q->ne[1]-begin);
            ggml_tensor part_q=*q,part_mask=*mask,part_dst=*dst;
            part_q.ne[1]=count;part_q.data=(char *)q->data+begin*q->nb[1];
            part_mask.ne[1]=count;part_mask.data=(char *)mask->data+begin*mask->nb[1];
            part_dst.ne[2]=count;part_dst.data=(char *)dst->data+begin*dst->nb[2];
            part_dst.src[0]=&part_q;part_dst.src[3]=&part_mask;
            strata_ds4_mma_case<512,512,1,8>(ctx,&part_dst);
        }
        return;
    }
    strata_ds4_mma_case<512,512,1,8>(ctx,dst);
}
DECL_FATTN_MMA_F16_CASE(512,512,1,8);
]=])
target_sources(ggml-cuda PRIVATE "${ds4_stable_dir}/strata-fattn-stable.cu")
set_source_files_properties("${ds4_stable_dir}/strata-fattn-stable.cu" TARGET_DIRECTORY ggml-cuda PROPERTIES
  INCLUDE_DIRECTORIES "${ds4_cuda_dir};${CMAKE_CURRENT_LIST_DIR}")
