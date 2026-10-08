#pragma once
#include "ggml.h"
#include "ggml-backend.h"
#include <cstdlib>
// Offline verification only: 1=ordinary matmuls (weights and attention), 2=experts.
// The host restores zero before prefill/draft. CUDA graphs remain disabled.
inline int mm27_tokenwise_mode() {
    const char *v=std::getenv("STRATA_MM27_TOKENWISE");
    return v && v[0]>='1' && v[0]<='3' && v[1]=='\0'?v[0]-'0':0;
}
inline bool mm27_tokenwise_dense(const ggml_tensor *a,const ggml_tensor *b,const ggml_tensor *y) {
    const int mode=mm27_tokenwise_mode();if(!mode)return false;
    return (mode&1) && (a->type==GGML_TYPE_F32 || a->type==GGML_TYPE_Q4_K || a->type==GGML_TYPE_Q6_K) &&
        b->type==GGML_TYPE_F32 && y->type==GGML_TYPE_F32 && b->ne[1]>1 && b->ne[1]<=8 && b->ne[1]==y->ne[1];
}
inline bool mm27_tokenwise_routed(const ggml_tensor *y) {
    const auto *a=y->src[0],*b=y->src[1],*ids=y->src[2];
    return (mm27_tokenwise_mode()&2) && (a->type==GGML_TYPE_Q4_K || a->type==GGML_TYPE_Q6_K) &&
        b->type==GGML_TYPE_F32 && y->type==GGML_TYPE_F32 && ids->type==GGML_TYPE_I32 &&
        y->ne[2]>1 && y->ne[2]<=8 && b->ne[2]==y->ne[2] && ids->ne[1]==y->ne[2] &&
        a->ne[3]==1 && b->ne[3]==1 && y->ne[3]==1 && ids->ne[2]==1 && ids->ne[3]==1;
}
