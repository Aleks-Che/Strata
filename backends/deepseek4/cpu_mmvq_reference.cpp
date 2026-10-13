// Host diagnostic adaptation of llama.cpp ggml-cuda/{quantize.cu,vecdotq.cuh,mmvq.cu}.
// MIT License
// Copyright (c) 2023-2026 The ggml authors
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include "cpu_mmvq_reference.hpp"
#include "swiglu_compat.hpp"
#include "cpu_lane.hpp"
#include "ggml-quants.h"
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <chrono>

namespace strata_ds4 {
static float butterfly(float *v,bool compat=false) {
    for(int offset=16;offset;offset/=2) {
        float next[32];for(int i=0;i<32;++i)next[i]=compat?sw_add(v[i],v[i^offset]):v[i]+v[i^offset];
        std::copy(next,next+32,v);
    }
    return v[0];
}
std::vector<CpuQ8_1> cpu_mmvq_quantize(const float *x,int columns) {
    if(columns<=0 || columns%32)throw std::runtime_error("Q8_1 requires complete 32-element blocks");
    std::vector<CpuQ8_1> out(columns/32);
    const bool compat=swiglu_compat_enabled();
    for(int b=0;b<columns/32;++b) {
        float amax=0,sum[32];
        for(int j=0;j<32;++j) {
            const float v=compat?sw_flush(x[b*32+j]):x[b*32+j];if(!std::isfinite(v))throw std::runtime_error("nonfinite activation");
            amax=std::max(amax,std::abs(v));sum[j]=v;
        }
        const float d=compat?sw_div(amax,127.0f):amax/127.0f;
        if(!compat && amax!=0 && d==0)throw std::runtime_error("activation scale underflows the diagnostic Q8_1 quantizer");
        out[b].d=ggml_fp32_to_fp16(d);out[b].s=ggml_fp32_to_fp16(butterfly(sum,compat));
        for(int j=0;j<32;++j)out[b].qs[j]=d==0?0:int8_t(std::round(compat?sw_div(sw_flush(x[b*32+j]),d):x[b*32+j]/d));
    }
    return out;
}
static float e8m0(uint8_t e) {
    uint32_t bits=e?uint32_t(e)<<23:0x00400000;float result;std::memcpy(&result,&bits,4);return result;
}
static float partial(ggml_type type,const void *w,const CpuQ8_1 *q,int block,int iqs,float acc) {
    int dot=0;float scale=0;
    if(type==GGML_TYPE_IQ3_XXS) {
        const auto &a=static_cast<const block_iq3_xxs *>(w)[block];
        const auto &b=q[block*8+iqs/2];
        uint32_t aux;std::memcpy(&aux,a.qs+64+4*(iqs/2),4);
        for(int group=0;group<4;++group) {
            const auto signs=ksigns_iq2xs[(aux>>(7*group))&127];
            for(int j=0;j<8;++j) {
                const auto grid=iq3xxs_grid[a.qs[4*iqs+group*2+j/4]];
                const int value=int((grid>>(8*(j%4)))&255);
                dot+=(signs&(1<<j)?-value:value)*b.qs[group*8+j];
            }
        }
        const int ls=aux>>28;dot=(ls*dot+dot/2)/2;
        scale=ggml_fp16_to_fp32(a.d)*ggml_fp16_to_fp32(b.d);
    } else if(type==GGML_TYPE_MXFP4) {
        const auto &a=static_cast<const block_mxfp4 *>(w)[block];const auto &b=q[block];
        for(int j=4*iqs;j<4*iqs+8;++j) {
            dot+=kvalues_mxfp4[a.qs[j]&15]*b.qs[j];
            dot+=kvalues_mxfp4[a.qs[j]>>4]*b.qs[j+16];
        }
        scale=(e8m0(a.e)*0.5f)*ggml_fp16_to_fp32(b.d);
    } else {
        const auto &a=static_cast<const block_q8_0 *>(w)[block];const auto &b=q[block];
        for(int j=4*iqs;j<4*iqs+8;++j)dot+=a.qs[j]*b.qs[j];
        scale=ggml_fp16_to_fp32(a.d)*ggml_fp16_to_fp32(b.d);
    }
    // NVCC contracts the final multiply into the per-lane accumulator.
    return std::fma(scale,float(dot),acc);
}
std::vector<float> cpu_mmvq_reference(ggml_type type,int columns,int rows,const void *w,const CpuQ8_1 *q,int warps) {
    if((type!=GGML_TYPE_IQ3_XXS && type!=GGML_TYPE_MXFP4 && type!=GGML_TYPE_Q8_0) ||
       columns<=0 || columns%256 || rows<=0 || (warps!=1 && warps!=2 && warps!=4))
        throw std::runtime_error("unsupported MMVQ reference shape");
    const int qk=type==GGML_TYPE_IQ3_XXS?256:32;
    const int qi=type==GGML_TYPE_IQ3_XXS?16:type==GGML_TYPE_MXFP4?4:8;
    const int lanes=warps*32,per_block=qi/2,step=lanes/per_block;
    const size_t stride=ggml_row_size(type,columns);std::vector<float> out(rows);
    for(int row=0;row<rows;++row) {
        float acc[128]={};const auto *weight=static_cast<const uint8_t *>(w)+row*stride;
        for(int tid=0;tid<lanes;++tid)
            for(int b=tid/per_block;b<columns/qk;b+=step)
                acc[tid]=partial(type,weight,q,b,2*(tid%per_block),acc[tid]);
        for(int warp=1;warp<warps;++warp)for(int lane=0;lane<32;++lane)acc[lane]+=acc[warp*32+lane];
        out[row]=butterfly(acc);
    }
    return out;
}
void cpu_mmvq_reference_compute(CpuExpertWork &w,CpuLanePool &pool,CpuExpertTiming *timing,const CpuWeightRetire &retire) {
    if(w.tokens<1 || w.tokens>4)throw std::runtime_error("MMVQ reference requires 1..4 tokens");
    using Clock=std::chrono::steady_clock;
    auto tick=Clock::now();auto elapsed=[&]{auto end=Clock::now();auto us=std::chrono::duration_cast<std::chrono::microseconds>(end-tick).count();tick=end;return uint64_t(us);};
    std::vector<std::vector<CpuQ8_1>> input(w.tokens),hidden(w.experts.size());
    for(int t=0;t<w.tokens;++t)input[t]=cpu_mmvq_quantize(w.input+size_t(t)*w.columns,w.columns);
    w.hidden_f.resize(w.experts.size()*w.hidden);
    const bool compat=swiglu_compat_enabled();
    const int parts=std::max(1,2*pool.size()/int(w.experts.size()));
    if(timing)timing->prepare_us=elapsed();
    pool.run(int(w.experts.size())*parts,[&](int job) {
        const int i=job/parts,part=job%parts,first=w.hidden*part/parts,last=w.hidden*(part+1)/parts;
        if(first==last)return;const auto &e=w.experts[i];const auto *q=input[e.token].data();
        auto g=cpu_mmvq_reference(w.gate_type,w.columns,last-first,static_cast<const uint8_t *>(e.gate)+first*ggml_row_size(w.gate_type,w.columns),q);
        auto u=cpu_mmvq_reference(w.up_type,w.columns,last-first,static_cast<const uint8_t *>(e.up)+first*ggml_row_size(w.up_type,w.columns),q);
        for(int r=first;r<last;++r) {
            if(compat) {
                w.hidden_f[size_t(i)*w.hidden+r]=swiglu_compat(g[r-first],u[r-first],w.clamp);
                continue;
            }
            float a=g[r-first],b=u[r-first];if(w.clamp>0){a=std::min(a,w.clamp);b=std::clamp(b,-w.clamp,w.clamp);}
            w.hidden_f[size_t(i)*w.hidden+r]=(a/(1.0f+std::exp(-a)))*b;
        }
    },timing?&timing->gate_up_pool:nullptr);
    if(timing)timing->gate_up_us=elapsed();if(retire)retire(CpuWeightPhase::gate_up);tick=Clock::now();
    for(size_t i=0;i<w.experts.size();++i)hidden[i]=cpu_mmvq_quantize(w.hidden_f.data()+i*w.hidden,w.hidden);
    if(timing)timing->hidden_quant_us=elapsed();
    pool.run(int(w.experts.size())*parts,[&](int job) {
        const int i=job/parts,part=job%parts,first=w.columns*part/parts,last=w.columns*(part+1)/parts;
        if(first==last)return;
        auto out=cpu_mmvq_reference(w.down_type,w.hidden,last-first,static_cast<const uint8_t *>(w.experts[i].down)+first*ggml_row_size(w.down_type,w.hidden),hidden[i].data());
        std::copy(out.begin(),out.end(),w.output+size_t(i)*w.columns+first);
    },timing?&timing->down_pool:nullptr);
    if(timing)timing->down_us=elapsed();if(retire)retire(CpuWeightPhase::down);
}
}
