// Use the real pinned CUDA quantizer; do not validate a host mirror against a
// second reimplementation. Only portable types cross the translation units.
#include "ggml-cuda/quantize.cuh"
#include "cpu_mmvq_reference.hpp"
#include "swiglu_compat.hpp"
#include "q8_1_compat.cuh"
#include <stdexcept>
static void check(cudaError_t rc) {if(rc!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(rc));}
static __global__ void test_swiglu(const float *g,const float *u,float *h,int count,float limit) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count)h[i]=strata_ds4::swiglu_compat(g[i],u[i],limit);
}
void strata_test_cuda_swiglu(const float *g,const float *u,float *h,int count,float limit) {
    float *buffer=nullptr;check(cudaMalloc(&buffer,size_t(count)*3*sizeof(float)));
    try {
        check(cudaMemcpy(buffer,g,count*sizeof(float),cudaMemcpyHostToDevice));
        check(cudaMemcpy(buffer+count,u,count*sizeof(float),cudaMemcpyHostToDevice));
        test_swiglu<<<(count+255)/256,256>>>(buffer,buffer+count,buffer+2*count,count,limit);
        check(cudaGetLastError());check(cudaMemcpy(h,buffer+2*count,count*sizeof(float),cudaMemcpyDeviceToHost));
    }catch(...){cudaFree(buffer);throw;}
    check(cudaFree(buffer));
}
bool strata_test_generic_mmvq_device() {
    cudaDeviceProp p{};check(cudaGetDeviceProperties(&p,0));return p.major>=8 && !(p.major==12 && p.minor==1);
}
void strata_test_cuda_quantize(const float *input,int columns,strata_ds4::CpuQ8_1 *output) {
    static_assert(sizeof(block_q8_1)==sizeof(strata_ds4::CpuQ8_1),"Q8_1 ABI");
    float *x=nullptr;void *q=nullptr;
    try {
        check(cudaMalloc(&x,columns*sizeof(float)));
        check(cudaMalloc(&q,columns/32*sizeof(block_q8_1)));
        check(cudaMemcpy(x,input,columns*sizeof(float),cudaMemcpyHostToDevice));
        (strata_ds4::swiglu_compat_enabled()?strata_quantize_row_q8_1_compat_cuda:quantize_row_q8_1_cuda)
            (x,nullptr,q,GGML_TYPE_IQ3_XXS,columns,columns,columns,columns,columns,1,1,1,nullptr);
        check(cudaGetLastError());
        check(cudaMemcpy(output,q,columns/32*sizeof(block_q8_1),cudaMemcpyDeviceToHost));
    } catch(...) {cudaFree(q);cudaFree(x);throw;}
    check(cudaFree(q));check(cudaFree(x));
}
