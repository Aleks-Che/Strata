#pragma once
#include "ggml.h"
#include "ggml-backend.h"
#include <cstdlib>
// Offline diagnostic bits: 1=dense weights, 2=routed quantized, 4=FA query tile.
// The probe enables this only during verification, not prompt prefill.
inline int mimo2_tokenwise_mode() {
    const char *v=std::getenv("STRATA_MIMO_TOKENWISE_MATMUL");
    return v && v[0]>='1' && v[0]<='7' && v[1]=='\0' ? v[0]-'0' : 0;
}
inline bool mimo2_tokenwise_dense(const ggml_tensor *a,const ggml_tensor *b,const ggml_tensor *y) {
    return (mimo2_tokenwise_mode()&1) && a->buffer &&
        ggml_backend_buffer_get_usage(a->buffer)==GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        b->type==GGML_TYPE_F32 && y->type==GGML_TYPE_F32 &&
        b->ne[1]>1 && b->ne[1]<=8 && b->ne[1]==y->ne[1] &&
        a->ne[2]==1 && a->ne[3]==1 && b->ne[2]==1 && b->ne[3]==1 && y->ne[2]==1 && y->ne[3]==1;
}
inline bool mimo2_tokenwise_routed(const ggml_tensor *y) {
    const auto *a=y->src[0],*b=y->src[1],*ids=y->src[2];
    return (mimo2_tokenwise_mode()&2) && ggml_is_quantized(a->type) &&
        b->type==GGML_TYPE_F32 && y->type==GGML_TYPE_F32 && ids->type==GGML_TYPE_I32 &&
        y->ne[2]>1 && y->ne[2]<=8 && b->ne[2]==y->ne[2] && ids->ne[1]==y->ne[2] &&
        a->ne[3]==1 && b->ne[3]==1 && y->ne[3]==1 && ids->ne[2]==1 && ids->ne[3]==1;
}
