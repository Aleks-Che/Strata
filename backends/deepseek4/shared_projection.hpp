#pragma once
// Research-only fixed MMQ for the measured DeepSeek shared-expert shapes.
// One predicate controls both dense dispatch and shortcuts through MMVQ fusion.
#include <cstdlib>
#include <cstring>
static bool strata_shared_suffix(const char * name, const char * suffix) {
    const size_t n=std::strlen(name),s=std::strlen(suffix);
    return n>=s && std::memcmp(name+n-s,suffix,s)==0;
}
static bool strata_shared_mmq_eligible(const ggml_tensor * w,const ggml_tensor * x,
        const ggml_tensor * dst,int cc) {
    const char * mode=std::getenv("STRATA_DS4_SHARED_MMQ");
    const bool all=mode && std::strcmp(mode,"2")==0;
    if (!mode || (!all && std::strcmp(mode,"1")!=0)) return false;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || cc<GGML_CUDA_CC_AMPERE ||
        dst->op!=GGML_OP_MUL_MAT || dst->op_params[0]!=GGML_PREC_DEFAULT ||
        w->type!=GGML_TYPE_Q8_0 || x->type!=GGML_TYPE_F32 || dst->type!=GGML_TYPE_F32 ||
        w->view_src || w->ne[2]!=1 || w->ne[3]!=1 || x->ne[2]!=1 || x->ne[3]!=1 ||
        x->ne[1]<1 || x->ne[1]>4096 || !ggml_is_contiguous(w) ||
        !ggml_is_contiguous(x) || !ggml_is_contiguous(dst)) return false;
    const bool gate_up=w->ne[0]==4096 && w->ne[1]==2048 &&
        (strata_shared_suffix(w->name,".ffn_gate_shexp.weight") ||
         strata_shared_suffix(w->name,".ffn_up_shexp.weight"));
    const bool down=all && w->ne[0]==2048 && w->ne[1]==4096 &&
        strata_shared_suffix(w->name,".ffn_down_shexp.weight");
    return (gate_up || down) && ggml_cuda_should_use_mmq(w->type,cc,x->ne[1],0);
}
static bool strata_shared_mmq_tensor(const ggml_tensor * tensor) {
    if (!tensor || tensor->op!=GGML_OP_MUL_MAT) return false;
    return strata_shared_mmq_eligible(tensor->src[0],tensor->src[1],tensor,
        ggml_cuda_info().devices[ggml_cuda_get_device()].cc);
}
