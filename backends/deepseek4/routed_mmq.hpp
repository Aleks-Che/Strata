#pragma once
// Research-only: keep short expert batches on the same upstream MMQ family
// as prefill. MXFP4 on Blackwell also changes activation quantization to FP4.
#include <cstdlib>
#include <cstring>
#include <climits>
#include <string>
static bool strata_routed_mmq_tensor(const ggml_tensor *t) {
    const char *mode=std::getenv("STRATA_DS4_ROUTED_MMQ");
    const bool all=mode && !std::strcmp(mode,"2");
    if(!mode || (!all && std::strcmp(mode,"1")) || !t || t->op!=GGML_OP_MUL_MAT_ID)return false;
    const auto *w=t->src[0],*x=t->src[1],*ids=t->src[2];
    const int cc=ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if(!GGML_CUDA_CC_IS_NVIDIA(cc) || cc<GGML_CUDA_CC_AMPERE ||
       (w->type!=GGML_TYPE_IQ3_XXS && !(all && w->type==GGML_TYPE_MXFP4)) ||
       x->type!=GGML_TYPE_F32 || t->type!=GGML_TYPE_F32 || ids->type!=GGML_TYPE_I32 ||
       w->view_src || w->ne[2]<6 || w->ne[2]>256 || w->ne[3]!=1 ||
       x->ne[2]<1 || x->ne[2]>8 || x->ne[3]!=1 || ids->ne[0]!=6 ||
       ids->ne[1]!=x->ne[2] || ids->ne[2]!=1 || ids->ne[3]!=1 ||
       ids->nb[0]!=sizeof(int32_t) || ids->nb[1]<6*sizeof(int32_t) ||
       ids->nb[1]%sizeof(int32_t) || ids->nb[1]/sizeof(int32_t)>INT_MAX ||
       !ggml_is_contiguous(w) || !ggml_is_contiguous(x) || !ggml_is_contiguous(t))return false;
    // Top-k from layer 3 onward is a [6,N] view of a [256,N] argsort.
    // The upstream MMQ ID helper consumes nb[1] explicitly. Requiring full
    // contiguity would accept N=1 but reject the same layout at N>1, mixing
    // MMQ/MMVQ arithmetic (and FP4/Q8 activations for MXFP4) in verification.
    std::string name=w->name;
    // Scheduler copies append a numeric copy index after the original name.
    const auto hash=name.rfind('#');
    if(hash!=std::string::npos && hash+1<name.size()) {
        bool numeric=true;for(size_t i=hash+1;i<name.size();++i)numeric &= name[i]>='0' && name[i]<='9';
        if(numeric)name.resize(hash);
    }
    const bool gate_up=w->ne[0]==4096 && w->ne[1]==2048 && x->ne[1]==1 &&
        (strata_shared_suffix(name.c_str(),".ffn_gate_exps.weight") || strata_shared_suffix(name.c_str(),".ffn_up_exps.weight"));
    const bool down=w->ne[0]==2048 && w->ne[1]==4096 && x->ne[1]==6 &&
        strata_shared_suffix(name.c_str(),".ffn_down_exps.weight");
    return (gate_up || down) && ggml_cuda_should_use_mmq(w->type,cc,x->ne[2],w->ne[2]);
}
