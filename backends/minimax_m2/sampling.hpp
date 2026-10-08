#pragma once
// Request-local CPU sampling. CUDA logits and the greedy baseline stay intact.
#include "llama.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace minimax_m2 {
struct SamplingConfig {
    float temperature=0.f,top_p=.95f;
    int32_t top_k=40;
    uint32_t seed=42;
    static SamplingConfig parse(const nlohmann::ordered_json &value,int vocabulary) {
        auto check=[](bool ok,const char *message) {if(!ok)throw std::runtime_error(message);};
        check(vocabulary>0,"invalid sampling vocabulary");
        check(value.is_object(),"sampling must be an object");
        SamplingConfig c;c.top_k=std::min(c.top_k,vocabulary);
        for(const auto &entry:value.items())
            check(entry.key()=="temperature" || entry.key()=="top_p" || entry.key()=="top_k" || entry.key()=="seed",
                  "unsupported sampling key");
        if(value.contains("temperature")) {
            const auto &v=value.at("temperature");check(v.is_number(),"temperature must be numeric");
            const double t=v.get<double>();
            check(std::isfinite(t) && (t==0 || (t>=double(.01f) && t<=2)),"temperature must be 0 or 0.01..2");
            c.temperature=float(t);
        }
        if(value.contains("top_p")) {
            const auto &v=value.at("top_p");check(v.is_number(),"top_p must be numeric");
            const double p=v.get<double>();
            check(std::isfinite(p) && p>0 && p<=1 && float(p)>0,"top_p must be in (0, 1]");c.top_p=float(p);
        }
        if(value.contains("top_k")) {
            const auto &v=value.at("top_k");
            check(v.is_number_integer() && v>=0 && v<=vocabulary,"top_k must be an integer in 0..vocabulary");
            c.top_k=v.get<int32_t>();
        }
        if(value.contains("seed")) {
            const auto &v=value.at("seed");
            // UINT32_MAX is llama.cpp's random-seed sentinel. Do not silently
            // turn a supposedly reproducible request into a random one.
            check(v.is_number_integer() && v>=0 && v<uint64_t(UINT32_MAX),"seed must be an integer in 0..4294967294");
            c.seed=v.get<uint32_t>();
        }
        return c;
    }
    nlohmann::ordered_json json() const {
        return {{"temperature",temperature},{"top_p",top_p},{"top_k",top_k},{"seed",seed}};
    }
    const char *algorithm() const {return temperature==0?"greedy":"temperature->top_k->top_p->dist";}
};

class RequestSampler {
    SamplingConfig config_;
    std::unique_ptr<llama_sampler,decltype(&llama_sampler_free)> chain_{nullptr,llama_sampler_free};
    std::vector<llama_token_data> candidates_;
public:
    explicit RequestSampler(SamplingConfig config):config_(config) {
        if(config_.temperature==0)return;
        auto params=llama_sampler_chain_default_params();params.no_perf=true;
        chain_.reset(llama_sampler_chain_init(params));
        if(!chain_)throw std::runtime_error("cannot allocate sampler chain");
        auto add=[&](llama_sampler *s) {
            if(!s)throw std::runtime_error("cannot allocate sampler");
            llama_sampler_chain_add(chain_.get(),s);
        };
        // Apply temperature before nucleus truncation: for T != 1 its order
        // affects the retained probability mass and must not be implicit.
        add(llama_sampler_init_temp(config_.temperature));
        add(llama_sampler_init_top_k(config_.top_k));
        add(llama_sampler_init_top_p(config_.top_p,1));
        add(llama_sampler_init_dist(config_.seed));
    }
    llama_token sample(const float *logits,int count,std::vector<llama_token_data> *trace=nullptr) {
        if(!logits || count<=0)throw std::runtime_error("empty sampling logits");
        if(!std::all_of(logits,logits+count,[](float x){return std::isfinite(x);}))
            throw std::runtime_error("non-finite sampling logits");
        if(!chain_) {
            const auto id=llama_token(std::max_element(logits,logits+count)-logits);
            if(trace)*trace={{id,logits[id],1.f}};
            return id;
        }
        candidates_.resize(count);
        for(int i=0;i<count;++i) {
            if(!std::isfinite(logits[i]/config_.temperature))throw std::runtime_error("temperature scales logits outside F32");
            candidates_[i]={llama_token(i),logits[i],0.f};
        }
        llama_token_data_array distribution={candidates_.data(),candidates_.size(),-1,false};
        llama_sampler_apply(chain_.get(),&distribution);
        if(distribution.selected<0 || size_t(distribution.selected)>=distribution.size)
            throw std::runtime_error("sampler did not select a candidate");
        const auto selected=distribution.data[distribution.selected];
        if(selected.id<0 || selected.id>=count || !std::isfinite(selected.p) || selected.p<=0)
            throw std::runtime_error("invalid sampled token/probability");
        if(trace)trace->assign(distribution.data,distribution.data+distribution.size);
        llama_sampler_accept(chain_.get(),selected.id);
        return selected.id;
    }
};
} // namespace minimax_m2
