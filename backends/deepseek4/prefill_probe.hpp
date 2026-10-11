#pragma once
// Diagnostic only: compare fixed token histories, without sampling or DSpark.
// Every pass ends with the same one-token graph. Measure snapshot restoration
// separately from observation, then clear all probe state before normal GEN.
#include "activation_probe.hpp"
#include "speculative.hpp"
#include <atomic>
#include <functional>
#include <iostream>

inline constexpr int PREFILL_PROBE_MAX_TOKENS=4096;
inline std::vector<int> prefill_probe_splits(const std::string &text,int prefix,int batch) {
    if(prefix<1 || prefix>=PREFILL_PROBE_MAX_TOKENS || batch<1)
        throw std::runtime_error("prefill probe requires 2..4096 tokens");
    std::vector<int> result;std::stringstream in(text);std::string item;int previous=0;
    while(std::getline(in,item,',')) {
        size_t end=0;int value=std::stoi(item,&end);
        if(end!=item.size() || value<=previous || value>prefix || value-previous>batch)
            throw std::runtime_error("prefill probe splits must increase within prefix and batch limits");
        result.push_back(value);previous=value;
    }
    if(previous<prefix) {
        if(prefix-previous>batch)throw std::runtime_error("prefill probe final split exceeds batch limit");
        result.push_back(prefix);
    }
    return result;
}

inline void prefill_probe_window(int row,int rows,int prefix,const std::vector<int> &ends) {
    if(rows<1 || rows>ActivationProbe::MAX_HISTORY_ROWS || row < -1 ||
       (row<0?rows!=1:row>prefix-rows))throw std::runtime_error("prefill history window is outside prefix");
    if(row<0)return;
    int begin=0;
    for(int end:ends) {
        if(begin<=row && row<end) {
            if(rows>end-row)throw std::runtime_error("prefill history window crosses a split");
            return;
        }
        begin=end;
    }
    throw std::runtime_error("prefill history window has no segment");
}

inline ActivationProbe::Json prefill_logit_diff(const std::vector<float> &a,const std::vector<float> &b) {
    if(a.empty() || a.size()!=b.size())throw std::runtime_error("prefill probe logits shape mismatch");
    double max=0,sum=0;size_t unequal=0;
    for(size_t i=0;i<a.size();++i) {
        if(!std::isfinite(a[i]) || !std::isfinite(b[i]))throw std::runtime_error("nonfinite prefill probe logits");
        const double d=double(a[i])-b[i];max=std::max(max,std::abs(d));sum+=d*d;unequal+=a[i]!=b[i];
    }
    return {{"max_abs",max},{"rms",std::sqrt(sum/a.size())},{"unequal",unequal},
        {"argmax_a",std::max_element(a.begin(),a.end())-a.begin()},
        {"argmax_b",std::max_element(b.begin(),b.end())-b.begin()}};
}

inline void run_prefill_probe(llama_context *ctx,const llama_vocab *vocab,ActivationProbe &probe,
        const std::vector<llama_token> &tokens,int batch,const std::string &path,
        const std::string &splits,const std::function<void()> &budget,const std::atomic<bool> &stop,int history_row=-1,int history_rows=1) {
    if(!probe.enabled())throw std::runtime_error("prefill probe requires STRATA_SPEC_TRACE_NODES");
    if(tokens.size()>llama_n_ctx(ctx))throw std::runtime_error("prefill probe exceeds context capacity");
    const int prefix=int(tokens.size())-1;
    auto segmented=prefill_probe_splits(splits,prefix,batch);
    auto full=prefill_probe_splits("",prefix,batch);
    prefill_probe_window(history_row,history_rows,prefix,segmented);
    struct Cleanup {llama_context *c;ActivationProbe &p;~Cleanup(){p.idle();llama_memory_clear(llama_get_memory(c),true);}} cleanup{ctx,probe};
    auto evaluate=[&](int begin,int end) {
        if(stop.load())throw std::runtime_error("prefill probe cancelled");
        Batch b(end-begin);b.positions(end-begin,begin,false);b.value.logits[end-begin-1]=true;
        std::copy(tokens.begin()+begin,tokens.begin()+end,b.value.token);
        budget();if(llama_decode(ctx,b.value))throw std::runtime_error("prefill probe decode failed");
    };
    auto logits=[&](){const auto *p=llama_get_logits_ith(ctx,-1);return std::vector<float>(p,p+llama_vocab_n_tokens(vocab));};
    ActivationProbe::Json report={{"tokens",tokens},{"full_splits",full},{"segmented_splits",segmented},{"history_row",history_row},{"history_rows",history_rows}};
    std::vector<std::vector<float>> controls,restored;
    float observer[2]={};
    for(int pass=0;pass<2;++pass) {
        probe.idle();llama_memory_clear(llama_get_memory(ctx),true);
        const auto &ends=pass?segmented:full;int begin=0;
        for(int end:ends){evaluate(begin,end);begin=end;}
        std::vector<uint8_t> state(llama_state_get_size(ctx));
        if(llama_state_get_data(ctx,state.data(),state.size())!=state.size())throw std::runtime_error("prefill probe snapshot failed");
        auto restore=[&](){probe.idle();if(llama_state_set_data(ctx,state.data(),state.size())!=state.size())throw std::runtime_error("prefill probe restore failed");};
        evaluate(prefix,prefix+1);controls.push_back(logits());
        restore();evaluate(prefix,prefix+1);restored.push_back(logits());
        if(history_row<0) {
            restore();if(pass)probe.serial(0);else probe.batch(prefix,1);
        } else {
            probe.idle();llama_memory_clear(llama_get_memory(ctx),true);begin=0;
            for(int end:ends) {
                if(!pass)probe.batch(0,prefix,history_row,history_rows);
                else if(begin<=history_row && history_row<end)probe.serial(history_row,end-begin,history_row-begin);
                else probe.idle();
                evaluate(begin,end);begin=end;
            }
            probe.idle();
        }
        evaluate(prefix,prefix+1);auto observed=logits();probe.idle();
        auto observation=prefill_logit_diff(history_row<0?restored.back():controls.back(),observed);observer[pass]=observation["max_abs"].get<float>();
        restore();evaluate(prefix,prefix+1);
        report[pass?"segmented":"full"]={{"restore",prefill_logit_diff(controls.back(),restored.back())},
            {"observer",observation},{"replay",prefill_logit_diff(restored.back(),logits())}};
    }
    report["control_difference"]=prefill_logit_diff(controls[0],controls[1]);
    report["restored_difference"]=prefill_logit_diff(restored[0],restored[1]);
    probe.finish(observer[0],observer[1]);
    std::ofstream out(path);if(!out)throw std::runtime_error("cannot open prefill probe report");
    out<<report.dump(2)<<'\n';out.flush();if(!out)throw std::runtime_error("cannot write prefill probe report");
    std::cerr<<"STRATA_PREFILL_PROBE tokens="<<tokens.size()<<" max_logit_diff="<<report["control_difference"]["max_abs"]<<"\n";
}
