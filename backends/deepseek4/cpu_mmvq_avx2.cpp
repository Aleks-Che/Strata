// SIMD implementation of the arithmetic documented in cpu_mmvq_reference.cpp.
// Original GGUF bytes and ggml lookup tables; no persistent unpacked weights.
#include "cpu_mmvq_fast.hpp"
#include "ggml-quants.h"
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"
#include <immintrin.h>
#include <array>
#include <cstring>

namespace strata_ds4 {
namespace {
// Sign a four-byte IQ3 grid with one nibble. 16 KiB, shared by all workers.
const auto &grid_table() {
static const auto table=[] {
    std::array<std::array<uint32_t,256>,16> table{};
    for(int s=0;s<16;++s)for(int k=0;k<256;++k) {
        uint32_t v=0;
        for(int j=0;j<4;++j) {
            int x=(iq3xxs_grid[k]>>(8*j))&255;if(s&(1<<j))x=-x;
            v|=uint32_t(uint8_t(x))<<(8*j);
        }
        table[s][k]=v;
    }
    return table;
}();
return table;
}
inline int sum4(__m128i x) {
    x=_mm_add_epi32(x,_mm_shuffle_epi32(x,_MM_SHUFFLE(1,0,3,2)));
    x=_mm_add_epi32(x,_mm_shuffle_epi32(x,_MM_SHUFFLE(2,3,0,1)));
    return _mm_cvtsi128_si32(x);
}
struct Weight32 {
    __m256i lo,hi;
    explicit Weight32(__m256i x):lo(_mm256_cvtepi8_epi16(_mm256_castsi256_si128(x))),
                                hi(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(x,1))){}
};
// Signed 16-bit products avoid maddubs saturation (including Q8 weight -128).
inline void dots(const Weight32 &w,const int8_t *q,int *d) {
    const auto x=_mm256_loadu_si256(reinterpret_cast<const __m256i *>(q));
    auto a=_mm256_madd_epi16(w.lo,_mm256_cvtepi8_epi16(_mm256_castsi256_si128(x)));
    auto b=_mm256_madd_epi16(w.hi,_mm256_cvtepi8_epi16(_mm256_extracti128_si256(x,1)));
    d[0]=sum4(_mm256_castsi256_si128(a));d[1]=sum4(_mm256_extracti128_si256(a,1));
    d[2]=sum4(_mm256_castsi256_si128(b));d[3]=sum4(_mm256_extracti128_si256(b,1));
}
inline float half(uint16_t x){return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(x)));}
inline float e8m0(uint8_t e){uint32_t bits=e?uint32_t(e)<<23:0x00400000;float x;std::memcpy(&x,&bits,4);return x;}
template<int N,ggml_type Type>
void rows(int columns,int count,const void *weights,const CpuQ8_1 *const *q,float *const *out) {
    constexpr int parts=Type==GGML_TYPE_IQ3_XXS?8:Type==GGML_TYPE_MXFP4?2:4;
    constexpr int qk=Type==GGML_TYPE_IQ3_XXS?256:32;
    const size_t stride=ggml_row_size(Type,columns);
    // Initialize only after the baseline dispatcher checked AVX2/OS support.
    const auto &signed_grid=grid_table();
    for(int r=0;r<count;++r) {
        __m128 acc[128];for(auto &v:acc)v=_mm_setzero_ps();
        const auto *row=static_cast<const uint8_t *>(weights)+r*stride;
        for(int b=0;b<columns/qk;++b) {
            const int lane=(b%(128/parts))*parts;
            if constexpr(Type==GGML_TYPE_IQ3_XXS) {
                const auto &w=reinterpret_cast<const block_iq3_xxs *>(row)[b];
                const float wd=half(w.d);
                for(int p=0;p<8;++p) {
                    uint32_t aux;std::memcpy(&aux,w.qs+64+p*4,4);
                    alignas(32) uint32_t values[8];
                    for(int j=0;j<4;++j) {
                        const int signs=ksigns_iq2xs[(aux>>(7*j))&127];
                        values[j*2]=signed_grid[signs&15][w.qs[p*8+j*2]];
                        values[j*2+1]=signed_grid[signs>>4][w.qs[p*8+j*2+1]];
                    }
                    Weight32 v(_mm256_load_si256(reinterpret_cast<const __m256i *>(values)));
                    alignas(16) float scales[4]={},sums[4]={};
                    for(int t=0;t<N;++t) {
                        const auto &x=q[t][b*8+p];int d[4];dots(v,x.qs,d);
                        int dot=d[0]+d[1]+d[2]+d[3];dot=((aux>>28)*int64_t(dot)+dot/2)/2;
                        sums[t]=float(dot);scales[t]=wd*half(x.d);
                    }
                    acc[lane+p]=_mm_fmadd_ps(_mm_load_ps(scales),_mm_load_ps(sums),acc[lane+p]);
                }
            } else {
                __m256i packed;float wd;
                if constexpr(Type==GGML_TYPE_MXFP4) {
                    const auto &w=reinterpret_cast<const block_mxfp4 *>(row)[b];
                    const auto x=_mm_loadu_si128(reinterpret_cast<const __m128i *>(w.qs));
                    const auto lut=_mm_loadu_si128(reinterpret_cast<const __m128i *>(kvalues_mxfp4));
                    const auto mask=_mm_set1_epi8(15);
                    packed=_mm256_set_m128i(_mm_shuffle_epi8(lut,_mm_and_si128(_mm_srli_epi16(x,4),mask)),
                                           _mm_shuffle_epi8(lut,_mm_and_si128(x,mask)));
                    wd=e8m0(w.e)*0.5f;
                } else {
                    const auto &w=reinterpret_cast<const block_q8_0 *>(row)[b];
                    packed=_mm256_loadu_si256(reinterpret_cast<const __m256i *>(w.qs));wd=half(w.d);
                }
                Weight32 v(packed);alignas(16) float scales[4]={},sums[4][4]={};
                for(int t=0;t<N;++t) {
                    const auto &x=q[t][b];int d[4];dots(v,x.qs,d);scales[t]=wd*half(x.d);
                    if constexpr(Type==GGML_TYPE_MXFP4) {sums[0][t]=float(d[0]+d[2]);sums[1][t]=float(d[1]+d[3]);}
                    else for(int p=0;p<4;++p)sums[p][t]=float(d[p]);
                }
                for(int p=0;p<parts;++p)acc[lane+p]=_mm_fmadd_ps(_mm_load_ps(scales),_mm_load_ps(sums[p]),acc[lane+p]);
            }
        }
        // Explicit non-reassociated CUDA warp accumulation and XOR tree.
        for(int warp=1;warp<4;++warp)for(int i=0;i<32;++i)acc[i]=_mm_add_ps(acc[i],acc[warp*32+i]);
        for(int offset=16;offset;offset/=2)for(int i=0;i<offset;++i)acc[i]=_mm_add_ps(acc[i],acc[i+offset]);
        alignas(16) float result[4];_mm_store_ps(result,acc[0]);for(int t=0;t<N;++t)out[t][r]=result[t];
    }
}
template<int N> void dispatch(ggml_type type,int columns,int count,const void *w,const CpuQ8_1 *const *q,float *const *out) {
    switch(type) {
    case GGML_TYPE_IQ3_XXS:rows<N,GGML_TYPE_IQ3_XXS>(columns,count,w,q,out);break;
    case GGML_TYPE_MXFP4:rows<N,GGML_TYPE_MXFP4>(columns,count,w,q,out);break;
    case GGML_TYPE_Q8_0:rows<N,GGML_TYPE_Q8_0>(columns,count,w,q,out);break;
    default:break; // Validated by the portable entry point.
    }
}
}
void cpu_mmvq_avx2(ggml_type type,int columns,int rows,const void *w,const CpuQ8_1 *const *q,float *const *out,int n) {
    switch(n) {
    case 1:dispatch<1>(type,columns,rows,w,q,out);break;
    case 2:dispatch<2>(type,columns,rows,w,q,out);break;
    case 3:dispatch<3>(type,columns,rows,w,q,out);break;
    case 4:dispatch<4>(type,columns,rows,w,q,out);break;
    }
}
}
