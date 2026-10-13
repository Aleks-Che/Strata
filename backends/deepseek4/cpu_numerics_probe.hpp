#pragma once
// Intrusive, opt-in diagnostic. All passes restore the SAME pre-decode state
// and evaluate the SAME token window. Caches may change placement between passes;
// timings from this request must not be used as performance measurements.
#include "prefill_probe.hpp"
#include "expert_transfer.h"
#include <fstream>
#include <chrono>

inline int64_t cpu_numerics_clock_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

inline void run_cpu_numerics_probe(llama_context *ctx,const llama_vocab *vocab,const llama_batch &batch,
        const std::vector<uint8_t> &state,StrataCpuExpertPhase phase,const std::string &path,
        const std::string &session,int position,int64_t first_begin) {
    if(!phase || state.empty())throw std::runtime_error("CPU numerical probe needs hybrid mode and a snapshot");
    const size_t nv=llama_vocab_n_tokens(vocab);
    auto read=[&] {
        std::vector<float> values(size_t(batch.n_tokens)*nv);
        for(int r=0;r<batch.n_tokens;++r)std::copy_n(llama_get_logits_ith(ctx,r),nv,values.data()+size_t(r)*nv);
        for(float v:values)if(!std::isfinite(v))throw std::runtime_error("nonfinite CPU numerical probe logits");
        return values;
    };
    struct RestorePhase {StrataCpuExpertPhase fn;~RestorePhase(){fn(true);}} restore_phase{phase};
    std::vector<std::vector<float>> results{read()};
    const char *names[]={"hybrid_initial","gpu_a","gpu_b","hybrid_a","hybrid_b"};
    ActivationProbe::Json spans=ActivationProbe::Json::array({{{"pass",names[0]},{"begin_us",first_begin},{"end_us",cpu_numerics_clock_us()}}});
    for(int i=1;i<5;++i) {
        if(llama_state_set_data(ctx,state.data(),state.size())!=state.size())throw std::runtime_error("CPU numerical probe restore failed");
        phase(i>=3);std::cerr<<"STRATA_CPU_NUMERICS_PASS name="<<names[i]<<"\n";
        const auto begin=cpu_numerics_clock_us();
        if(llama_decode(ctx,batch))throw std::runtime_error("CPU numerical probe decode failed");
        results.push_back(read());
        spans.push_back({{"pass",names[i]},{"begin_us",begin},{"end_us",cpu_numerics_clock_us()}});
    }
    ActivationProbe::Json report={{"session",session},{"position",position},
        {"input_ids",std::vector<llama_token>(batch.token,batch.token+batch.n_tokens)},
        {"values_per_pass",results[0].size()},{"trace_spans",spans},{"comparisons",ActivationProbe::Json::array()}};
    for(auto pair:{std::make_pair(0,1),std::make_pair(1,2),std::make_pair(3,1),std::make_pair(3,4),std::make_pair(0,4)}) {
        ActivationProbe::Json rows=ActivationProbe::Json::array();
        for(int r=0;r<batch.n_tokens;++r) {
            auto first=[&](int pass){return results[pass].begin()+size_t(r)*nv;};
            rows.push_back(prefill_logit_diff({first(pair.first),first(pair.first)+nv},{first(pair.second),first(pair.second)+nv}));
        }
        report["comparisons"].push_back({{"a",names[pair.first]},{"b",names[pair.second]},
            {"bit_exact",!std::memcmp(results[pair.first].data(),results[pair.second].data(),results[0].size()*4)},
            {"difference",prefill_logit_diff(results[pair.first],results[pair.second])},{"rows",rows}});
    }
    std::ofstream out(path,std::ios::app);if(!out)throw std::runtime_error("cannot open CPU numerical probe report");
    out<<report.dump()<<'\n';out.flush();if(!out)throw std::runtime_error("cannot write CPU numerical probe report");
    // GPU repetition is the state/control guard; hybrid differences are the
    // measured outcome. This does not relax the existing speculative validator.
    if(std::memcmp(results[1].data(),results[2].data(),results[0].size()*4))
        throw std::runtime_error("CPU numerical probe GPU control is not reproducible");
}
