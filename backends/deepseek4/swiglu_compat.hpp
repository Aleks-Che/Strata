#pragma once
// Opt-in numerical contract for routed F32 experts with 1..4 tokens.
// Never depend on CUDA fast-math exp/reciprocal or the host libm expf.
// This scalar reference favors reproducibility over throughput. Finite inputs,
// IEEE round-to-nearest/even; flush F32 subnormals to signed zero on both sides.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>

#ifdef __CUDACC__
#define STRATA_SW_HD __host__ __device__ __forceinline__
#else
#define STRATA_SW_HD inline
#endif
namespace strata_ds4 {
inline bool swiglu_compat_enabled() {
    const char *p=std::getenv("STRATA_DS4_SWIGLU_COMPAT");
    if(!p)p=std::getenv("STRATA_DS4_CPU_MMVQ_REFERENCE");
    return p && p[0]=='1' && p[1]==0;
}
STRATA_SW_HD float sw_flush(float x) {
#ifdef __CUDA_ARCH__
    uint32_t bits=__float_as_uint(x);
#else
    uint32_t bits;std::memcpy(&bits,&x,4);
#endif
    if((bits&0x7f800000u)==0)bits&=0x80000000u;
#ifdef __CUDA_ARCH__
    return __uint_as_float(bits);
#else
    std::memcpy(&x,&bits,4);return x;
#endif
}
STRATA_SW_HD double sw_fma(double a,double b,double c) {
#ifdef __CUDA_ARCH__
    return __fma_rn(a,b,c);
#else
    return std::fma(a,b,c);
#endif
}
STRATA_SW_HD float sw_to_float(double x) {
#ifdef __CUDA_ARCH__
    return sw_flush(__double2float_rn(x));
#else
    volatile float v=static_cast<float>(x);return sw_flush(v);
#endif
}
STRATA_SW_HD float sw_exp(float x) {
    // Only exp(-gate) is needed. Outside this interval the F32 result is
    // zero/infinity, including the deliberate subnormal flush.
    if(x < -88.0f)return 0.0f;
    if(x > 89.0f)return sw_to_float(1.0e300);
    const int n=int(sw_fma(double(x),1.4426950408889634074,x>=0?0.5:-0.5));
    const double r=sw_fma(-double(n),0.69314718055994530942,double(x));
    // Degree 13 on [-ln(2)/2,ln(2)/2]. Explicit FMA order is the contract;
    // this is an approximation, not a claim of correctly rounded expf.
    double p=1.0/6227020800.0;
    p=sw_fma(p,r,1.0/479001600.0);p=sw_fma(p,r,1.0/39916800.0);
    p=sw_fma(p,r,1.0/3628800.0);p=sw_fma(p,r,1.0/362880.0);
    p=sw_fma(p,r,1.0/40320.0);p=sw_fma(p,r,1.0/5040.0);
    p=sw_fma(p,r,1.0/720.0);p=sw_fma(p,r,1.0/120.0);
    p=sw_fma(p,r,1.0/24.0);p=sw_fma(p,r,1.0/6.0);
    p=sw_fma(p,r,0.5);p=sw_fma(p,r,1.0);p=sw_fma(p,r,1.0);
    const uint64_t bits=uint64_t(n+1023)<<52;
#ifdef __CUDA_ARCH__
    const double scale=__longlong_as_double(static_cast<long long>(bits));
#else
    double scale;std::memcpy(&scale,&bits,8);
#endif
    return sw_to_float(sw_fma(p,scale,0.0));
}
STRATA_SW_HD float sw_add(float a,float b) {
#ifdef __CUDA_ARCH__
    return sw_flush(__fadd_rn(a,b));
#else
    volatile float v=a+b;return sw_flush(v);
#endif
}
STRATA_SW_HD float sw_div(float a,float b) {
#ifdef __CUDA_ARCH__
    return sw_flush(__fdiv_rn(a,b));
#else
    volatile float v=a/b;return sw_flush(v);
#endif
}
STRATA_SW_HD float sw_mul(float a,float b) {
#ifdef __CUDA_ARCH__
    return sw_flush(__fmul_rn(a,b));
#else
    volatile float v=a*b;return sw_flush(v);
#endif
}
STRATA_SW_HD float swiglu_compat(float gate,float up,float limit=0) {
    gate=sw_flush(gate);up=sw_flush(up);
    if(limit>0) {gate=gate>limit?limit:gate;up=up>limit?limit:up;up=up<-limit?-limit:up;}
    return sw_mul(sw_div(gate,sw_add(1.0f,sw_exp(-gate))),up);
}
} // namespace strata_ds4
#undef STRATA_SW_HD
