#pragma once
#include "runtime.hpp"
#include "dflash_contract.hpp"
#include "llama-ext.h"
namespace minimax_m2 {
struct DraftBatch {
    llama_batch b;
    DraftBatch(int n,int pos,int width=0):b(llama_batch_init(n,width,1)) {
        b.n_tokens=n;for(int i=0;i<n;++i) {b.pos[i]=pos+i;b.n_seq_id[i]=1;b.seq_id[i][0]=0;b.logits[i]=width?0:1;}
    }
    ~DraftBatch() {llama_batch_free(b);}
    DraftBatch(const DraftBatch &)=delete;
};
inline llama_token greedy(llama_context *ctx,int row,int vocab) {
    const auto *p=llama_get_logits_ith(ctx,row);require(p,"missing logits");
    require(std::all_of(p,p+vocab,[](float v){return std::isfinite(v);}),"non-finite logits");
    return llama_token(std::max_element(p,p+vocab)-p);
}
struct VerifiedPrefix {std::vector<llama_token> tokens;int accepted=0,keep=0;};
template<class Sample>
VerifiedPrefix verify_greedy(const std::vector<llama_token> &proposals,int remaining,Sample sample) {
    require(remaining>0,"no output budget");VerifiedPrefix out;
    for(int i=0;i<=int(proposals.size()) && int(out.tokens.size())<remaining;++i) {
        const auto id=sample(i);out.tokens.push_back(id);
        const bool match=i<int(proposals.size()) && proposals[i]==id;out.accepted+=match;
        if(!match || is_stop(id))break;
    }
    out.keep=std::min(int(proposals.size())+1,out.accepted+1);return out;
}
class DFlashDecode {
    llama_context *target;
    Model model{nullptr,llama_model_free};
    Context ctx{nullptr,llama_free};
    void evaluate(DraftBatch &batch) {
        strata_mm27_memory();const int rc=llama_decode(ctx.get(),batch.b);llama_synchronize(ctx.get());
        require(rc==0,"DFlash decode failed: "+std::to_string(rc)+" "+strata_mm27_last_error());strata_mm27_memory();
    }
public:
    DFlashDecode(llama_context *t,const std::string &target_path,const std::string &draft_path):target(t) {
        const auto bytes=inspect_dflash(target_path,draft_path);
        strata_mm27_cache_clear();strata_mm27_memory(bytes+(2ull<<30),256ull<<20);
        auto *gpu=ggml_backend_dev_by_name("CUDA0");require(gpu,"CUDA0 missing");ggml_backend_dev_t devices[]={gpu,nullptr};
        auto mp=llama_model_default_params();mp.devices=devices;mp.n_gpu_layers=-1;mp.split_mode=LLAMA_SPLIT_MODE_NONE;
        mp.load_mtp=false;mp.use_extra_bufts=false;mp.no_host=true;mp.load_mode=LLAMA_LOAD_MODE_MMAP;
        model.reset(llama_model_load_from_file(draft_path.c_str(),mp));require(bool(model),"draft load failed");registered_dflash(*model);
        for(const auto &v:model->tensors_by_name)require(v.second->data && v.second->buffer && !ggml_backend_buffer_is_host(v.second->buffer),"draft weight on CPU: "+v.first);
        auto cp=llama_context_default_params();cp.n_ctx=llama_n_ctx(target);cp.n_batch=cp.n_ubatch=8;cp.n_seq_max=1;
        cp.n_threads=cp.n_threads_batch=4;cp.type_k=cp.type_v=GGML_TYPE_F32;cp.offload_kqv=cp.op_offload=true;
        cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;cp.ctx_other=target;
        ctx.reset(llama_init_from_model(model.get(),cp));require(bool(ctx),"draft context failed");
        clear(ctx.get());llama_set_causal_attn(ctx.get(),false);
        for(int layer:dflash_layers)llama_set_embeddings_layer_inp(target,layer,true);
        strata_mm27_memory();
    }
    void reset() {clear(ctx.get());}
    // Call while target layer inputs still belong to the last decode. Inject
    // only anchor + accepted inputs; rejected suffix features never enter KV.
    void process(int count,int pos) {
        require(count>=1 && count<=8,"invalid feature batch");DraftBatch b(count,pos,15360);
        for(size_t k=0;k<dflash_layers.size();++k) {
            const float *h=llama_get_embeddings_layer_inp(target,dflash_layers[k]);require(h,"missing target features");
            for(int i=0;i<count;++i)std::copy_n(h+i*3072,3072,b.b.embd+(i*5+k)*3072);
        }
        evaluate(b);
    }
    std::vector<llama_token> propose(llama_token anchor,int pos,int depth) {
        require(depth>=0 && depth<=7,"invalid DFlash depth");
        if(!depth || anchor<0 || anchor>=dflash_vocab)return {};
        DraftBatch b(depth+1,pos);b.b.token[0]=anchor;for(int i=1;i<=depth;++i)b.b.token[i]=dflash_mask;
        evaluate(b);std::vector<llama_token> out;
        for(int i=1;i<=depth;++i) {auto id=greedy(ctx.get(),i,dflash_vocab);out.push_back(id);if(is_stop(id))break;}
        require(llama_memory_seq_rm(llama_get_memory(ctx.get()),0,pos,-1),"DFlash noise rollback failed");return out;
    }
    int position() const {return llama_memory_seq_pos_max(llama_get_memory(ctx.get()),0);}
};
}
