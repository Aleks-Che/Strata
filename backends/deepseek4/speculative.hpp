#pragma once
#include "llama.h"
#include "llama-ext.h"
#include "llama-context.h"
#include "shared_scratch.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

// A rejected position also rejects its entire suffix. Invalid head outputs
// must not turn into unverified proposals; zero disables the filter entirely.
inline int confidence_prefix(const std::vector<float>& confidence, float minimum) {
    if(minimum<=0)return int(confidence.size());
    int n=0;
    for(float p:confidence) {
        if(!std::isfinite(p) || p<minimum || p>1)break;
        ++n;
    }
    return n;
}

// One sequence, explicit positions, RAII. Feature injection must see exactly
// one target ubatch; the layer outputs are overwritten by the next decode.
struct Batch {
    llama_batch value;
    Batch(int capacity, int embedding=0):value(llama_batch_init(capacity,embedding,1)) {}
    ~Batch() { llama_batch_free(value); }
    Batch(const Batch&)=delete;
    void positions(int count, int start, bool logits) {
        value.n_tokens=count;
        for(int i=0;i<count;++i) {
            value.pos[i]=start+i;value.n_seq_id[i]=1;value.seq_id[i][0]=0;
            value.logits[i]=logits;
        }
    }
};

// The pinned llama.cpp DSpark graph implements feature fusion, Markov head,
// attention and MoE on GPU. This adapter only packs batches and selects tokens.
// Equivalent to common/speculative.cpp's draft-dspark, limited to DSV4 0731.
class DSpark {
    llama_context *target, *draft;
    int width, limit;
    float minimum;
    std::vector<float> confidence;
    int proposed=0;
    size_t shared_bytes=0;
    llama_token mask;
    std::vector<int32_t> layers;
    Batch features, noise;
    std::unique_ptr<llama_sampler,decltype(&llama_sampler_free)> greedy;
public:
    DSpark(llama_context *t, llama_context *d, int n, float p_min=0,bool share_scratch=false):target(t),draft(d),
        width(llama_model_n_embd(llama_get_model(t))),limit(n),minimum(p_min),
        mask(llama_vocab_mask(llama_model_get_vocab(llama_get_model(d)))),
        features(llama_n_ubatch(d),width*llama_model_target_layer_ids_n(llama_get_model(d))),
        noise(n),greedy(llama_sampler_init_greedy(),llama_sampler_free) {
        auto *model=llama_get_model(d);
        auto *ids=llama_model_target_layer_ids(model);
        layers.assign(ids,ids+llama_model_target_layer_ids_n(model));
        // GGUF stores inputs of the following layers (41,42,43), equivalent
        // to HF's outputs of layers (40,41,42). 43 is the final layer output.
        if(layers!=std::vector<int32_t>{41,42,43} || width!=4096 || mask<0)
            throw std::runtime_error("DSpark requires the matching DeepSeek V4 Flash 0731 sidecar");
        for(int layer:layers)llama_set_embeddings_layer_inp(target,layer,true);
        llama_set_embeddings_nextn(draft,true,true);
        llama_set_causal_attn(draft,false);
        if(share_scratch) {
            // Final feature/attention flags are set before reserving. Neither
            // context has executed; KV/state and expert caches stay separate.
            target->sched_reserve();draft->sched_reserve();
            shared_bytes=strata_ds4_sched_share_scratch(target->get_sched(),draft->get_sched());
            if(!shared_bytes)throw std::runtime_error("DSpark scratch sharing requires compatible fresh CUDA schedulers");
        }
    }
    size_t scratch_saved() const {return shared_bytes;}
    bool scratch_shared() const {return shared_bytes && strata_ds4_sched_scratch_is_shared(target->get_sched(),draft->get_sched());}
    const std::vector<float>& last_confidence() const {return confidence;}
    int last_proposed() const {return proposed;}
    void inject(const llama_batch &batch) {
        std::vector<const float *> inputs;
        for(int layer:layers) {
            inputs.push_back(llama_get_embeddings_layer_inp(target,layer));
            if(!inputs.back())throw std::runtime_error("DSpark target features are missing");
        }
        for(int offset=0;offset<batch.n_tokens;) {
            int count=std::min(int(llama_n_ubatch(draft)),batch.n_tokens-offset);
            features.positions(count,batch.pos[offset],false);
            for(size_t k=0;k<layers.size();++k) for(int i=0;i<count;++i)
                std::memcpy(features.value.embd+(size_t(i)*layers.size()+k)*width,inputs[k]+size_t(offset+i)*width,width*sizeof(float));
            if(llama_decode(draft,features.value))throw std::runtime_error("DSpark feature injection failed");
            offset+=count;
        }
        // Feature injection has no logits to force a host synchronization.
        // Drain it before the next target graph can reuse the shared scratch.
        if(shared_bytes)llama_synchronize(draft);
    }
    std::vector<llama_token> propose(llama_token anchor,int position,int remaining) {
        const int n=std::min(limit,remaining);
        proposed=n;confidence.clear();
        if(n<=0)return {};
        noise.positions(n,position,true);
        for(int i=0;i<n;++i)noise.value.token[i]=i?mask:anchor;
        if(llama_decode(draft,noise.value))throw std::runtime_error("DSpark draft decode failed");
        int kept=n;
        if(minimum>0 || std::getenv("STRATA_SPEC_CONFIDENCE_TRACE")) {
            const float *conf=llama_get_embeddings_nextn(draft);
            if(!conf && minimum>0)throw std::runtime_error("DSpark confidence head output is missing");
            if(conf) {
                for(int i=0;i<n;++i)confidence.push_back(conf[size_t(i)*width]);
                kept=confidence_prefix(confidence,minimum);
            }
        }
        std::vector<llama_token> result;
        for(int i=0;i<kept;++i)result.push_back(llama_sampler_sample(greedy.get(),draft,i));
        // Noise KV must never be used as target history. Injection replaces it
        // with real features after verification. Retain the previous ring rows.
        if(!llama_memory_seq_rm(llama_get_memory(draft),0,position,-1))
            throw std::runtime_error("DSpark noise rollback failed");
        return result;
    }
};

struct VerifiedBlock {
    std::vector<llama_token> tokens;
    int accepted=0;
    bool eog=false;
};

// Target-sampled matching preserves the target sampler distribution. Sampling
// already accepts each selected token in this llama.cpp version; never accept
// it a second time (penalties/history would then be wrong).
template<class Sample, class IsEog>
VerifiedBlock verify_draft(const std::vector<llama_token>&draft,int budget,Sample sample,IsEog is_eog) {
    VerifiedBlock out;
    for(int i=0;i<=int(draft.size()) && int(out.tokens.size())<budget;++i) {
        auto token=sample(i);
        out.tokens.push_back(token);
        bool match=i<int(draft.size()) && token==draft[i];
        if(match)++out.accepted;
        if(is_eog(token)) {out.eog=true;break;}
        if(!match)break;
    }
    return out;
}
