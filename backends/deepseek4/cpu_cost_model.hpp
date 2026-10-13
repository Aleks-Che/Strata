#pragma once
#include <cstdint>
#ifdef __CUDACC__
#define STRATA_COST_HD __host__ __device__
#else
#define STRATA_COST_HD
#endif
namespace strata_ds4 {
constexpr int cpu_cost_patterns=405, cpu_cost_tokens=4, cpu_cost_max_rows=8;
struct CpuCostModel {
    bool enabled=false;
    uint64_t generation=0; // Fixed for an entire request, including DSpark.
    // Exact measured histogram: c1 + 9*c2 + 45*c3 + 135*c4, <=8 rows.
    float compute[cpu_cost_tokens][cpu_cost_patterns]{};
    float loaded_compute[cpu_cost_tokens][cpu_cost_patterns]{};
    float boundary[cpu_cost_tokens][cpu_cost_max_rows+1]{};
    float gpu_compute[cpu_cost_tokens][cpu_cost_max_rows+1]{};
    float copy_us_per_byte=0;
    float loaded_copy_us_per_byte=0;
    bool contention=false,gpu_enabled=false,burst=false;
    int gpu_topk=8;
    float gpu_overlap=0;
    float budget_us_per_token=2000, margin=0.03f;
    int revision=1;
};
STRATA_COST_HD inline int cpu_cost_key(const int *counts,int n) {
    int h[5]={},rows=0;
    for(int i=0;i<n;++i) {if(counts[i]<0 || counts[i]>4)return -1;if(counts[i])++h[counts[i]];rows+=counts[i];}
    return rows<=8?h[1]+9*h[2]+45*h[3]+135*h[4]:-1;
}
STRATA_COST_HD inline float cpu_cost_compute(const CpuCostModel &m,int tokens,const int *counts,int n,bool copying=false) {
    const int key=cpu_cost_key(counts,n);if(key==0)return 0;
    if(tokens<1 || tokens>4 || key<0)return -1;
    float value=m.compute[tokens-1][key];if(value<=0)return -1;
    if(m.contention && copying) {
        const float loaded=m.loaded_compute[tokens-1][key];if(loaded<=0)return -1;
        if(loaded>value)value=loaded;
    }
    return value;
}
STRATA_COST_HD inline float cpu_cost_transfer(const CpuCostModel &m,uint64_t bytes,int cpu_rows) {
    float rate=m.copy_us_per_byte;
    if(m.contention && cpu_rows>0 && m.loaded_copy_us_per_byte>rate)rate=m.loaded_copy_us_per_byte;
    return float(bytes)*rate;
}
STRATA_COST_HD inline float cpu_cost_gpu(const CpuCostModel &m,int tokens,int cpu_rows,int topk) {
    if(!m.gpu_enabled)return 0;
    if(tokens<1 || tokens>4 || cpu_rows<0 || cpu_rows>8 || cpu_rows>tokens*topk || topk!=m.gpu_topk)return -1;
    const float value=m.gpu_compute[tokens-1][cpu_rows];return value>0?value:-1;
}
STRATA_COST_HD inline float cpu_cost_gpu_branch(const CpuCostModel &m,float transfer,float compute) {
    const float overlap=transfer<compute?transfer:compute;
    return transfer+compute-overlap*m.gpu_overlap;
}
struct CpuCostDecision {unsigned mask=0;float cpu_us=0,gpu_us=0,total_us=0;uint64_t saved_bytes=0;};
// Prefix-only: counts/gpu_required describe already committed tokens. No later
// route IDs or total window length are consulted. Exhaustively compare the
// <=8 ready candidates, including the empty assignment, within the row budget.
STRATA_COST_HD inline CpuCostDecision cpu_cost_choose(const CpuCostModel &m,int tokens,
        const int *counts,const bool *gpu_required,int experts,const int *ids,const int *occurrences,
        const uint64_t *missing,int candidates,int cap,uint64_t gpu_bytes,int prefix_rows,int topk=8) {
    CpuCostDecision best;
    if(tokens<1 || tokens>4 || experts<1 || experts>256 || candidates<0 || candidates>8)return best;
    const float prior=cpu_cost_compute(m,tokens,counts,experts,gpu_bytes>0);
    const float prior_gpu=cpu_cost_gpu(m,tokens,prefix_rows,topk);if(prior_gpu<0)return best;
    best.cpu_us=prior>0?prior:0;best.gpu_us=cpu_cost_transfer(m,gpu_bytes,prefix_rows);
    const float prior_branch=cpu_cost_gpu_branch(m,best.gpu_us,prior_gpu);
    const float boundary=prefix_rows>0 && prefix_rows<=8?m.boundary[tokens-1][prefix_rows]:0;
    best.total_us=(best.cpu_us>prior_branch?best.cpu_us:prior_branch)+boundary;
    if(prior<0 || tokens<1 || tokens>4)return best;
    for(unsigned mask=1;mask<(1u<<candidates);++mask) {
        int take=0,rows=prefix_rows;
        for(int i=0;i<candidates;++i)if(mask&(1u<<i)){++take;rows+=occurrences[i];}
        if(take>cap || rows>8)continue;
        int c[256];for(int i=0;i<experts;++i)c[i]=counts[i];
        uint64_t saved=0;
        for(int i=0;i<candidates;++i)if(mask&(1u<<i)) {
            c[ids[i]]+=occurrences[i];
            if(!gpu_required[ids[i]])saved+=missing[i];
        }
        if(saved>gpu_bytes)continue;
        const float cpu=cpu_cost_compute(m,tokens,c,experts,gpu_bytes>saved);
        if(cpu<0 || cpu+m.boundary[tokens-1][rows]>m.budget_us_per_token*tokens)continue;
        const float kernel=cpu_cost_gpu(m,tokens,rows,topk);if(kernel<0)continue;
        const float gpu=cpu_cost_transfer(m,gpu_bytes-saved,rows);
        const float branch=cpu_cost_gpu_branch(m,gpu,kernel);
        const float cost=(cpu>branch?cpu:branch)+m.boundary[tokens-1][rows];
        if(cost<best.total_us*(1-m.margin))best={mask,cpu,gpu,cost,saved};
    }
    return best;
}
}
#undef STRATA_COST_HD
