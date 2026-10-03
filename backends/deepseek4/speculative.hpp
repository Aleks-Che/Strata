#pragma once
#include "llama.h"
#include "llama-ext.h"
#include <algorithm>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

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
    llama_token mask;
    std::vector<int32_t> layers;
    Batch features, noise;
    std::unique_ptr<llama_sampler,decltype(&llama_sampler_free)> greedy;
public:
    DSpark(llama_context *t, llama_context *d, int n):target(t),draft(d),
        width(llama_model_n_embd(llama_get_model(t))),limit(n),
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
    }
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
    }
    std::vector<llama_token> propose(llama_token anchor,int position,int remaining) {
        const int n=std::min(limit,remaining);
        if(n<=0)return {};
        noise.positions(n,position,true);
        for(int i=0;i<n;++i)noise.value.token[i]=i?mask:anchor;
        if(llama_decode(draft,noise.value))throw std::runtime_error("DSpark draft decode failed");
        std::vector<llama_token> result;
        for(int i=0;i<n;++i)result.push_back(llama_sampler_sample(greedy.get(),draft,i));
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
