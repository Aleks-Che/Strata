#pragma once
#include "runtime.hpp"
#include "llama-ext.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>

namespace strata_glm {
// Native GLM NextN, one active sequence. The target sampler is called exactly
// once per emitted token: sample from p, accept a greedy draft only on equality.
// This sample-and-match scheme retains the target distribution without a p/q
// correction or extra RNG draws. Its acceptance can be lower than p/q sampling.
class Mtp {
    using Clock=std::chrono::steady_clock;
    llama_context * target;
    Context draft{nullptr,llama_free};
    int width,vocab,depth;
    bool cache_only_catch_up=true;
    std::vector<float> pending;
    static double ms(Clock::time_point t) {return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
    std::vector<float> features(llama_context * ctx,int count) {
        std::vector<float> rows(size_t(count)*width);
        for (int i=0;i<count;++i) {
            const auto * h=llama_get_embeddings_nextn_ith(ctx,i);
            require(h!=nullptr,"missing native MTP feature row");
            for (int j=0;j<width;++j) require(std::isfinite(h[j]),"non-finite MTP feature");
            std::copy(h,h+width,rows.begin()+size_t(i)*width);
        }
        return rows;
    }
    void draft_decode(const std::vector<llama_token> & ids,int pos,const std::vector<float> & hidden,bool outputs=true) {
        require(hidden.size()==ids.size()*width,"MTP hidden/token count mismatch");
        auto b=llama_batch_init(int(ids.size()),width,1);
        b.token=static_cast<llama_token *>(std::malloc(ids.size()*sizeof(llama_token)));
        if (!b.token) {llama_batch_free(b);throw std::bad_alloc();}
        b.n_tokens=int(ids.size());
        for (int i=0;i<b.n_tokens;++i) {
            b.token[i]=ids[i];b.pos[i]=pos+i;b.n_seq_id[i]=1;b.seq_id[i][0]=0;b.logits[i]=outputs && i+1==b.n_tokens;
        }
        std::copy(hidden.begin(),hidden.end(),b.embd);
        const int status=llama_decode(draft.get(),b);llama_batch_free(b);
        require(status==0,"MTP decode failed: "+std::to_string(status));llama_synchronize(draft.get());
    }
    void catch_up(const std::vector<llama_token> & ids,int pos,const std::vector<float> & target_rows) {
        // Pair x[p] with the target's h[p-1], never retain self-generated draft
        // features as accepted history. Row zero carries across microbatches.
        std::vector<float> shifted(ids.size()*width);
        std::copy(pending.begin(),pending.end(),shifted.begin());
        if (ids.size()>1) std::copy_n(target_rows.begin(),(ids.size()-1)*width,shifted.begin()+width);
        // The next draft uses target features, not the discarded catch-up logits
        // or hidden rows. With masked extraction, zero outputs means cache only.
        draft_decode(ids,pos,shifted,!cache_only_catch_up);
        pending.assign(target_rows.end()-width,target_rows.end());
    }
public:
    struct Counters {uint64_t proposed=0,accepted=0,rounds=0,reject_first=0,reject_middle=0,accept_all=0;double draft_ms=0,verify_ms=0,repair_ms=0;};
    Counters counters;
    struct Round {std::vector<llama_token> tokens;int next_position=0,accepted=0,proposed=0;bool cancelled=false;};
    Mtp(llama_model * model,llama_context * ctx,int size,int batch,int threads,int count)
        :target(ctx),width(llama_model_n_embd_out(model)),vocab(llama_vocab_n_tokens(llama_model_get_vocab(model))),depth(count),pending(width,0) {
        require(count>=1 && count<=3 && batch>=count+1 && llama_n_rs_seq(ctx)>=uint32_t(count),"MTP requires depth 1..3, batch >= depth+1 and recurrent rollback slots");
        require(llama_model_n_layer_nextn(model)==1,"GLM MTP requires one native NextN block");
        if (const char * value=std::getenv("STRATA_GLM_MTP_CACHE_ONLY")) {
            require(std::string(value)=="0" || std::string(value)=="1","STRATA_GLM_MTP_CACHE_ONLY must be 0 or 1");
            cache_only_catch_up=std::string(value)=="1";
        }
        draft=context(model,size,batch,threads,0,true);
        llama_set_embeddings_nextn(target,true,false);
        llama_set_embeddings_nextn(draft.get(),true,true);
    }
    void set_depth(int n) {require(n>=1 && n<=3 && uint32_t(n)<=llama_n_rs_seq(target),"invalid MTP depth");depth=n;}
    bool uses_cache_only_catch_up() const {return cache_only_catch_up;}
    void reset() {clear(draft.get());std::fill(pending.begin(),pending.end(),0);counters={};}
    void prefill(const std::vector<llama_token> & ids,int start,int count,int position) {
        auto h=features(target,count);
        catch_up({ids.begin()+start,ids.begin()+start+count},position,h);
    }
    // On cancellation the caller discards both contexts, as on a cancelled
    // normal decode. No partly accepted sequence is exposed for reuse.
    Round advance(llama_token carry,int position,int remaining,
                  const std::function<llama_token(int)> & sample,
                  const std::function<bool(llama_token)> & stop,
                  const std::function<bool()> & cancel,
                  const std::vector<llama_token> * forced=nullptr) {
        require(remaining>0,"MTP round requires output space");
        Round out;out.next_position=position;
        const int count=std::min(depth,remaining-1);
        std::vector<llama_token> input{carry};auto h=pending;
        auto start=Clock::now();
        for (int i=0;i<count;++i) {
            if (cancel()) {out.cancelled=true;return out;}
            draft_decode({input.back()},position+i,h);
            const auto * logits=llama_get_logits_ith(draft.get(),-1);
            require(logits!=nullptr,"MTP logits missing");
            for (int j=0;j<vocab;++j) require(std::isfinite(logits[j]),"non-finite MTP logits");
            llama_token token=llama_token(std::max_element(logits,logits+vocab)-logits);
            if (forced) {require(i<int(forced->size()),"short forced draft");token=forced->at(i);}
            require(token>=0 && token<vocab,"invalid draft token");
            input.push_back(token);++out.proposed;
            if (stop(token)) break;
            h=features(draft.get(),1);
        }
        counters.draft_ms+=ms(start);counters.proposed+=out.proposed;
        if (cancel()) {out.cancelled=true;return out;}
        start=Clock::now();decode(target,input,0,int(input.size()),position,true);
        counters.verify_ms+=ms(start);auto verified_h=features(target,int(input.size()));
        ++counters.rounds;
        for (int i=0;i<int(input.size());++i) {
            if (cancel()) {out.cancelled=true;return out;}
            const auto token=sample(i);out.tokens.push_back(token);
            const bool accepted=i<out.proposed && token==input[i+1];
            if (accepted) ++out.accepted;
            if (stop(token) || !accepted || int(out.tokens.size())==remaining) break;
        }
        counters.accepted+=out.accepted;
        if (out.proposed) {
            if (out.accepted==out.proposed) ++counters.accept_all;
            else if (out.accepted==0) ++counters.reject_first;
            else ++counters.reject_middle;
        }
        start=Clock::now();
        const int keep=1+out.accepted;out.next_position=position+keep;
        if (keep<int(input.size())) require(llama_memory_seq_rm(llama_get_memory(target),0,out.next_position,-1),"target hybrid rollback rejected");
        input.resize(keep);verified_h.resize(size_t(keep)*width);
        if (out.proposed) {
            // The first draft step already used the correct target feature for
            // carry. Preserve it; only later steps used speculative features.
            require(llama_memory_seq_rm(llama_get_memory(draft.get()),0,position+1,-1),"MTP indexer rollback rejected");
            pending.assign(verified_h.begin(),verified_h.begin()+width);
            if (keep>1) catch_up({input.begin()+1,input.end()},position+1,{verified_h.begin()+width,verified_h.end()});
        } else catch_up(input,position,verified_h);
        counters.repair_ms+=ms(start);
        return out;
    }
};
}
