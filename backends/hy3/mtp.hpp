#pragma once
#include "runtime.hpp"
#include "llama-ext.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>

namespace hy3 {
// Native Hy3 NextN with resident block weights and separate full-attention KV.
// Drafts are greedy; only the target sampler consumes/accepts emitted tokens.
// Main embeddings/output have one model owner shared by both contexts.
class Mtp {
    using Clock=std::chrono::steady_clock;
    llama_context * target;
    Context draft{nullptr,llama_free};
    int width,vocab,depth;
    std::vector<float> pending;
    static double ms(Clock::time_point t) {return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
    std::vector<float> features(llama_context * ctx,int count) {
        std::vector<float> rows(size_t(count)*width);
        for (int i=0;i<count;++i) {
            const auto * h=llama_get_embeddings_nextn_ith(ctx,i);
            require(h!=nullptr,"missing native MTP feature row");
            for (int j=0;j<width;++j) if(!std::isfinite(h[j])) throw std::runtime_error("non-finite MTP feature");
            std::copy(h,h+width,rows.begin()+size_t(i)*width);
        }
        return rows;
    }
    void draft_decode(const std::vector<llama_token> & ids,int pos,const std::vector<float> & hidden) {
        require(hidden.size()==ids.size()*width,"MTP hidden/token count mismatch");
        require(!ids.empty() && ids.size()<=llama_n_batch(draft.get()) && pos>=0 &&
            size_t(pos)+ids.size()<=llama_n_ctx(draft.get()),"MTP batch outside context");
        strata_hy3_memory_check();
        auto b=llama_batch_init(int(ids.size()),width,1);
        b.token=static_cast<llama_token *>(std::malloc(ids.size()*sizeof(llama_token)));
        if (!b.token) {llama_batch_free(b);throw std::bad_alloc();}
        b.n_tokens=int(ids.size());
        for (int i=0;i<b.n_tokens;++i) {
            b.token[i]=ids[i];b.pos[i]=pos+i;b.n_seq_id[i]=1;b.seq_id[i][0]=0;b.logits[i]=i+1==b.n_tokens;
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
        draft_decode(ids,pos,shifted);
        pending.assign(target_rows.end()-width,target_rows.end());
    }
public:
    struct Counters {uint64_t proposed=0,accepted=0,rounds=0,reject_first=0,reject_middle=0,accept_all=0;double draft_ms=0,verify_ms=0,repair_ms=0,prefill_ms=0;};
    Counters counters;
    struct Round {std::vector<llama_token> tokens;int next_position=0,accepted=0,proposed=0;bool cancelled=false;};
    Mtp(llama_model * model,llama_context * ctx,int size,int batch,int count)
        :target(ctx),width(llama_model_n_embd_out(model)),vocab(llama_vocab_n_tokens(llama_model_get_vocab(model))),depth(count),pending(width,0) {
        require(count>=1 && count<=3 && batch>=count+1,"MTP requires depth 1..3 and batch >= depth+1");
        require(llama_model_n_layer_nextn(model)==1,"Hy3 MTP requires one native NextN block");
        draft=context(model,size,batch,GGML_TYPE_F32,nullptr,nullptr,true);
        llama_set_embeddings_nextn(target,true,false);
        llama_set_embeddings_nextn(draft.get(),true,false);
    }
    void set_depth(int n) {require(n>=1 && n<=3 && uint32_t(n+1)<=llama_n_batch(target),"invalid MTP depth");depth=n;}
    void reset() {clear(draft.get());std::fill(pending.begin(),pending.end(),0);counters={};}
    void prefill(const std::vector<llama_token> & ids,int offset,int count,int position) {
        const auto start=Clock::now();
        auto h=features(target,count);
        catch_up({ids.begin()+offset,ids.begin()+offset+count},position,h);
        counters.prefill_ms+=ms(start);
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
            for (int j=0;j<vocab;++j) if(!std::isfinite(logits[j])) throw std::runtime_error("non-finite MTP logits");
            llama_token token=llama_token(std::max_element(logits,logits+vocab)-logits);
            if (forced) {require(i<int(forced->size()),"short forced draft");token=forced->at(i);}
            require(token>=0 && token<vocab,"invalid draft token");
            input.push_back(token);++out.proposed;
            if (stop(token)) break;
            h=features(draft.get(),1);
        }
        counters.draft_ms+=ms(start);counters.proposed+=out.proposed;
        if (cancel()) {out.cancelled=true;return out;}
        // A proposed EOG is checked by its predecessor's logits. Computing the
        // EOG input itself would only produce an unused token after the stop.
        const int verify_count=int(input.size())-(out.proposed && stop(input.back()) ? 1 : 0);
        start=Clock::now();decode(target,input,0,verify_count,position,true);
        counters.verify_ms+=ms(start);auto verified_h=features(target,verify_count);
        ++counters.rounds;
        for (int i=0;i<verify_count;++i) {
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
        const int keep=std::min(1+out.accepted,verify_count);out.next_position=position+keep;
        if (keep<verify_count) require(llama_memory_seq_rm(llama_get_memory(target),0,out.next_position,-1),"Hy3 target KV rollback rejected");
        input.resize(keep);verified_h.resize(size_t(keep)*width);
        if (out.proposed) {
            // The first draft step already used the correct target feature for
            // carry. Preserve it; only later steps used speculative features.
            require(llama_memory_seq_rm(llama_get_memory(draft.get()),0,position+1,-1),"Hy3 draft KV rollback rejected");
            pending.assign(verified_h.begin(),verified_h.begin()+width);
            if (keep>1) catch_up({input.begin()+1,input.end()},position+1,{verified_h.begin()+width,verified_h.end()});
        } else catch_up(input,position,verified_h);
        counters.repair_ms+=ms(start);
        return out;
    }
};
}
