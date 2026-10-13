#pragma once
#include "swiglu_compat.hpp"

// Match the CPU lane's supported routed short-window domain. Dense/shared
// experts, prefill, F16 and other activations retain their original dispatch.
static bool strata_swiglu_routed(const ggml_tensor *g,const ggml_tensor *u) {
    return g && u && g->op==GGML_OP_MUL_MAT_ID && u->op==GGML_OP_MUL_MAT_ID &&
        g->type==GGML_TYPE_F32 && u->type==GGML_TYPE_F32 &&
        g->ne[2]>=1 && g->ne[2]<=4 && g->ne[3]==1 &&
        ggml_are_same_shape(g,u) && strata_ds4::swiglu_compat_enabled();
}
static __global__ void strata_swiglu_kernel(const float *g,const float *u,float *h,
        int64_t count,int64_t columns,int64_t gs,int64_t us,float limit) {
    ggml_cuda_pdl_lc();
    const int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=count)return;
    ggml_cuda_pdl_sync();
    h[i]=strata_ds4::swiglu_compat(g[(i/columns)*gs+i%columns],u[(i/columns)*us+i%columns],limit);
}
static bool strata_swiglu_dispatch(ggml_backend_cuda_context &ctx,ggml_tensor *dst,float limit) {
    auto *g=dst->src[0],*u=dst->src[1];
    if(!strata_swiglu_routed(g,u))return false;
    GGML_ASSERT(dst->type==GGML_TYPE_F32 && ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous_1(g) && ggml_is_contiguous_1(u));
    const int64_t count=ggml_nelements(dst),columns=dst->ne[0];
    const ggml_cuda_kernel_launch_params launch(dim3((count+255)/256),256,0,ctx.stream());
    ggml_cuda_kernel_launch(strata_swiglu_kernel,launch,(const float *)g->data,(const float *)u->data,
        (float *)dst->data,count,columns,int64_t(g->nb[1]/4),int64_t(u->nb[1]/4),limit);
    return true;
}
