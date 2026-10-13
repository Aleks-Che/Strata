#include "cpu_mmvq_fast.hpp"
#include "swiglu_compat.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <stdexcept>
#if defined(_MSC_VER) && defined(_M_X64)
#include <intrin.h>
#endif

namespace strata_ds4 {
bool cpu_mmvq_fast_enabled() {
    const char *p=std::getenv("STRATA_DS4_CPU_MMVQ_FAST");return p && !std::strcmp(p,"1");
}
bool cpu_mmvq_avx2_available() {
    static const bool available=[] {
#if defined(STRATA_MMVQ_AVX2) && defined(_MSC_VER) && defined(_M_X64)
        int r[4];__cpuid(r,0);if(r[0]<7)return false;
        __cpuidex(r,1,0);
        // AVX + OSXSAVE + FMA + F16C, and OS preservation of XMM/YMM.
        const int mask=(1<<28)|(1<<27)|(1<<12)|(1<<29);
        if((r[2]&mask)!=mask || (_xgetbv(0)&6)!=6)return false;
        __cpuidex(r,7,0);return (r[1]&(1<<5))!=0;
#elif defined(STRATA_MMVQ_AVX2) && (defined(__x86_64__) || defined(__i386__))
        __builtin_cpu_init();
        return bool(__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c"));
#else
        return false;
#endif
    }();
    // Diagnostic fallback, also exercises portable builds on an AVX2 machine.
    const char *p=std::getenv("STRATA_DS4_CPU_MMVQ_NO_AVX2");
    return available && !(p && !std::strcmp(p,"1"));
}
#ifdef STRATA_MMVQ_AVX2
void cpu_mmvq_avx2(ggml_type,int,int,const void *,const CpuQ8_1 *const *,float *const *,int);
#endif
void cpu_mmvq_fast(ggml_type type,int columns,int rows,const void *weights,
                   const CpuQ8_1 *const *q,float *const *out,int tokens) {
    if(tokens<1 || tokens>4 || columns<=0 || columns%256 || rows<1 ||
       (type!=GGML_TYPE_IQ3_XXS && type!=GGML_TYPE_MXFP4 && type!=GGML_TYPE_Q8_0))
        throw std::runtime_error("unsupported fast MMVQ shape");
#ifdef STRATA_MMVQ_AVX2
    if(cpu_mmvq_avx2_available()) {
        const char *single=std::getenv("STRATA_DS4_CPU_MMVQ_SINGLE");
        if(single && !std::strcmp(single,"1")) {
            // Benchmark control: same grouped jobs, but each token rereads and
            // unpacks the weights. It preserves exactly the same arithmetic.
            for(int t=0;t<tokens;++t)cpu_mmvq_avx2(type,columns,rows,weights,q+t,out+t,1);
        } else cpu_mmvq_avx2(type,columns,rows,weights,q,out,tokens);
        return;
    }
#endif
    for(int t=0;t<tokens;++t) {
        const auto y=cpu_mmvq_reference(type,columns,rows,weights,q[t]);
        std::copy(y.begin(),y.end(),out[t]);
    }
}
void cpu_mmvq_fast_compute(CpuExpertWork &w,CpuLanePool &pool,CpuExpertTiming *timing,const CpuWeightRetire &retire) {
    if(w.experts.empty())return;
    if(w.tokens<1 || w.tokens>4 || !swiglu_compat_enabled())
        throw std::runtime_error("fast MMVQ requires compatible arithmetic and 1..4 tokens");
    using Clock=std::chrono::steady_clock;
    auto tick=Clock::now();auto elapsed=[&]{auto end=Clock::now();auto us=std::chrono::duration_cast<std::chrono::microseconds>(end-tick).count();tick=end;return uint64_t(us);};
    struct Group {std::array<size_t,4> rows{};int n=0;};
    std::vector<Group> groups;
    for(size_t i=0;i<w.experts.size();++i) {
        const auto &e=w.experts[i];if(e.token<0 || e.token>=w.tokens)throw std::runtime_error("invalid fast MMVQ token");
        auto g=std::find_if(groups.begin(),groups.end(),[&](const Group &g) {
            const auto &a=w.experts[g.rows[0]];
            return g.n<4 && a.gate==e.gate && a.up==e.up && a.down==e.down;
        });
        if(g==groups.end()){groups.push_back({});g=groups.end()-1;}
        g->rows[g->n++]=i;
    }
    std::vector<std::vector<CpuQ8_1>> input(w.tokens),hidden(w.experts.size());
    for(int t=0;t<w.tokens;++t)input[t]=cpu_mmvq_quantize(w.input+size_t(t)*w.columns,w.columns);
    w.hidden_f.resize(w.experts.size()*w.hidden);
    const int parts=std::max(1,pool.options().partitions*pool.size()/int(groups.size()));
    if(timing)timing->prepare_us=elapsed();
    pool.run(int(groups.size())*parts,[&](int job) {
        const auto &g=groups[job/parts];const auto &e=w.experts[g.rows[0]];
        const int first=w.hidden*(job%parts)/parts,last=w.hidden*(job%parts+1)/parts,count=last-first;
        if(!count)return;
        std::vector<float> gates(size_t(count)*g.n),ups(gates.size());
        const CpuQ8_1 *q[4];float *a[4],*b[4];
        for(int t=0;t<g.n;++t){q[t]=input[w.experts[g.rows[t]].token].data();a[t]=gates.data()+t*count;b[t]=ups.data()+t*count;}
        cpu_mmvq_fast(w.gate_type,w.columns,count,static_cast<const uint8_t *>(e.gate)+first*ggml_row_size(w.gate_type,w.columns),q,a,g.n);
        cpu_mmvq_fast(w.up_type,w.columns,count,static_cast<const uint8_t *>(e.up)+first*ggml_row_size(w.up_type,w.columns),q,b,g.n);
        for(int t=0;t<g.n;++t)for(int r=0;r<count;++r)
            w.hidden_f[g.rows[t]*w.hidden+first+r]=swiglu_compat(a[t][r],b[t][r],w.clamp);
    },timing?&timing->gate_up_pool:nullptr);
    if(timing)timing->gate_up_us=elapsed();if(retire)retire(CpuWeightPhase::gate_up);tick=Clock::now();
    for(size_t i=0;i<w.experts.size();++i)hidden[i]=cpu_mmvq_quantize(w.hidden_f.data()+i*w.hidden,w.hidden);
    if(timing)timing->hidden_quant_us=elapsed();
    pool.run(int(groups.size())*parts,[&](int job) {
        const auto &g=groups[job/parts];const auto &e=w.experts[g.rows[0]];
        const int first=w.columns*(job%parts)/parts,last=w.columns*(job%parts+1)/parts;
        if(first==last)return;
        const CpuQ8_1 *q[4];float *out[4];
        for(int t=0;t<g.n;++t){q[t]=hidden[g.rows[t]].data();out[t]=w.output+g.rows[t]*w.columns+first;}
        cpu_mmvq_fast(w.down_type,w.hidden,last-first,static_cast<const uint8_t *>(e.down)+first*ggml_row_size(w.down_type,w.hidden),q,out,g.n);
    },timing?&timing->down_pool:nullptr);
    if(timing)timing->down_us=elapsed();if(retire)retire(CpuWeightPhase::down);
}
}
