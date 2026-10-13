#include "cpu_cost_gpu_probe.hpp"
#include "swiglu_compat.cuh"
void strata_cpu_cost_swiglu(const float *gate,const float *up,float *hidden,int columns,int count,float clamp,cudaStream_t stream) {
    const ggml_cuda_kernel_launch_params launch(dim3((count+255)/256),256,0,stream);
    ggml_cuda_kernel_launch(strata_swiglu_kernel,launch,gate,up,hidden,int64_t(count),int64_t(columns),int64_t(columns),int64_t(columns),clamp);
}
void strata_cpu_cost_matvec(ggml_type type,const void *weights,const float *input,void *quantized,float *output,
        const int32_t *ids,const int *slots,const uint8_t *mask,int columns,int rows,int experts,int topk,int tokens,
        int channels,cudaStream_t stream) {
    GGML_ASSERT(columns%256==0 && tokens>=1 && tokens<=4 && topk>=1 && topk<=8);
    GGML_ASSERT(type==GGML_TYPE_IQ3_XXS || type==GGML_TYPE_MXFP4 || type==GGML_TYPE_Q8_0);
    const int padded=GGML_PAD(columns,MATRIX_ROW_PADDING),qstride=padded/QK8_1;
    strata_quantize_row_q8_1_compat_cuda(input,nullptr,quantized,type,columns,columns,columns*channels,
        columns*channels*tokens,padded,channels,tokens,1,stream);
    StrataDirectFusion fusion{};fusion.swiglu_compat=true;fusion.cpu_slots=slots;fusion.cpu_mask=mask;
    const int wstride=columns/ggml_blck_size(type);
    mul_mat_vec_q_switch_type(weights,type,quantized,ids,fusion,output,columns,rows,tokens,wstride,
        qstride*channels,rows*topk,experts,channels,topk,wstride*rows,qstride,rows,
        1,1,wstride*rows*experts,qstride*channels*tokens,rows*topk*tokens,topk,stream);
}
