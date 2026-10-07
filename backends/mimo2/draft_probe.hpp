#pragma once
// Offline, greedy comparison. MTP uses the first trained head only: the generic
// multi-head chaining rule is not assumed valid for MiMo's independently offset heads.
#include "runtime.hpp"
#include "llama-ext.h"
#include "ggml-alloc.h"
#include <cstring>
#include <fstream>
namespace mimo2 {
struct DraftBatch {
    llama_batch b;
    DraftBatch(int n,int width=0):b(llama_batch_init(n,width,1)) {}
    ~DraftBatch() {llama_batch_free(b);}
    DraftBatch(const DraftBatch &)=delete;
    void positions(int n,int pos,bool logits) {
        b.n_tokens=n;
        for(int i=0;i<n;++i) {b.pos[i]=pos+i;b.n_seq_id[i]=1;b.seq_id[i][0]=0;b.logits[i]=logits;}
    }
};
inline llama_token draft_greedy(llama_context *ctx,int row) {
    const auto *l=llama_get_logits_ith(ctx,row);require(l,"missing logits");
    int n=llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    for(int i=0;i<n;++i)require(std::isfinite(l[i]),"non-finite logits");
    return int(std::max_element(l,l+n)-l);
}
inline Context probe_context(llama_model *model,bool mtp=false,llama_context *other=nullptr,
                             ggml_backend_sched_eval_callback callback=nullptr,void *callback_data=nullptr) {
    auto cp=llama_context_default_params();cp.n_ctx=512;cp.n_batch=cp.n_ubatch=8;cp.n_seq_max=1;
    cp.type_k=cp.type_v=GGML_TYPE_F32;cp.swa_full=true; // Preserve all tested positions for exact rollback.
    cp.n_threads=cp.n_threads_batch=4;cp.offload_kqv=cp.op_offload=true;
    cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.ctx_type=mtp?LLAMA_CONTEXT_TYPE_MTP:LLAMA_CONTEXT_TYPE_DEFAULT;
    cp.ctx_other=other;
    cp.cb_eval=callback;cp.cb_eval_user_data=callback_data;
    Context ctx(llama_init_from_model(model,cp),llama_free);require(bool(ctx),"probe context failed");
    clear(ctx.get());strata_mimo_memory();return ctx;
}
struct DraftMask {
    std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx{nullptr,ggml_free};
    std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> buffer{nullptr,ggml_backend_buffer_free};
    ggml_tensor *tensor=nullptr;
    explicit DraftMask(const std::string &path) {
        ggml_context *raw=nullptr;
        std::unique_ptr<gguf_context,decltype(&gguf_free)> file(gguf_init_from_file(path.c_str(),{true,&raw}),gguf_free);
        std::unique_ptr<ggml_context,decltype(&ggml_free)> header(raw,ggml_free);
        require(file && header,"cannot read draft MASK header");
        const int64_t id=gguf_find_tensor(file.get(),"token_embd.weight");require(id>=0,"missing draft embedding");
        auto *weight=ggml_get_tensor(header.get(),"token_embd.weight");
        require(weight && weight->ne[0]==4096 && weight->ne[1]==152576 &&
            (weight->type==GGML_TYPE_BF16 || weight->type==GGML_TYPE_Q8_0),"wrong DFlash MASK geometry");
        const size_t row=ggml_row_size(weight->type,4096);
        std::vector<char> bytes(row);std::vector<float> values(4096);
        std::ifstream stream(path,std::ios::binary);stream.seekg(gguf_get_data_offset(file.get())+gguf_get_tensor_offset(file.get(),id)+151675ull*row);
        stream.read(bytes.data(),row);require(bool(stream),"cannot read learned MASK row");
        // Decode one stored constant at load time; all model matrix operations remain on CUDA.
        ggml_get_type_traits(weight->type)->to_float(bytes.data(),values.data(),4096);
        double norm=0;for(float value:values) {require(std::isfinite(value),"invalid MASK constant");norm+=double(value)*value;}
        require(norm>.01 && norm<4,"learned MASK missing or invalid");
        ctx.reset(ggml_init({ggml_tensor_overhead()+1024,nullptr,true}));require(bool(ctx),"MASK context allocation failed");
        tensor=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,4096,1);ggml_set_name(tensor,"mimo_dflash_learned_mask");
        buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(),ggml_backend_dev_buffer_type(ggml_backend_dev_by_name("CUDA0"))));
        require(bool(buffer),"MASK GPU allocation failed");ggml_backend_buffer_set_usage(buffer.get(),GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_tensor_set(tensor,values.data(),0,values.size()*sizeof(float));
    }
};
inline Model probe_load_draft(const std::string &path,bool mtp) {
    // The runner validates every tensor/range/tokenizer before launching this offline probe.
    strata_mimo_memory(std::filesystem::file_size(path)+(2ull<<30),256ull<<20);
    auto *gpu=ggml_backend_dev_by_name("CUDA0");require(gpu,"CUDA0 required");
    ggml_backend_dev_t devices[]={gpu,nullptr};
    llama_model_tensor_buft_override overrides[]={{"^token_embd\\.weight$",ggml_backend_dev_buffer_type(gpu)},{nullptr,nullptr}};
    auto mp=llama_model_default_params();mp.devices=devices;mp.n_gpu_layers=-1;mp.split_mode=LLAMA_SPLIT_MODE_NONE;
    mp.load_mtp=mtp;mp.no_host=true;mp.use_extra_bufts=false;mp.load_mode=LLAMA_LOAD_MODE_MMAP;mp.tensor_buft_overrides=overrides;
    Model m(llama_model_load_from_file(path.c_str(),mp),llama_model_free);require(bool(m),"draft load failed");
    require(m->arch==(mtp?LLM_ARCH_MIMO2:LLM_ARCH_DFLASH),"wrong draft architecture");
    if(mtp)require(m->hparams.n_layer()==48 && m->hparams.n_layer_nextn==3 && !m->layers[0].attn_norm,"expected MiMo-only NextN sidecar");
    else require(m->hparams.n_layer()==5 && m->target_layer_ids==std::vector<int32_t>{1,12,24,36,48} &&
        std::abs(m->hparams.f_attn_value_scale-.612f)<1e-7f,"wrong MiMo DFlash geometry/value scale");
    for(const auto &item:m->tensors_by_name)require(item.second->data && item.second->buffer &&
        !ggml_backend_buffer_is_host(item.second->buffer),"draft weight on CPU: "+item.first);
    strata_mimo_memory();return m;
}
class DraftProbe {
    llama_context *target,*draft;bool mtp;int width=4096;
    std::vector<float> pending=std::vector<float>(4096,0);
    const std::vector<int> layers{1,12,24,36,48};
    void evaluate(DraftBatch &b) {
        strata_mimo_memory();const int rc=llama_decode(draft,b.b);llama_synchronize(draft);
        require(rc==0,"draft decode failed: "+std::to_string(rc));strata_mimo_memory();
    }
    std::vector<float> target_hidden(int count) {
        std::vector<float> h(size_t(count)*width);
        const float *src=llama_get_embeddings_nextn(target);require(src,"missing target NextN features");
        std::copy_n(src,h.size(),h.begin());return h;
    }
    void mtp_input(const std::vector<llama_token> &ids,int pos,const std::vector<float> &h,bool logits) {
        DraftBatch b(int(ids.size()),width);b.b.token=static_cast<llama_token *>(std::malloc(ids.size()*sizeof(llama_token)));
        require(b.b.token,"MTP token allocation failed");b.positions(int(ids.size()),pos,logits);
        std::copy(ids.begin(),ids.end(),b.b.token);std::copy(h.begin(),h.end(),b.b.embd);evaluate(b);
    }
public:
    DraftProbe(llama_context *t,llama_context *d,bool m):target(t),draft(d),mtp(m) {
        if(mtp) {llama_set_embeddings_nextn(target,true,false);llama_set_nextn_layer_offset(draft,0);}
        else {
            for(int layer:layers)llama_set_embeddings_layer_inp(target,layer,true);
            llama_set_causal_attn(draft,false);
        }
    }
    void reset() {clear(draft);std::fill(pending.begin(),pending.end(),0);}
    // Target features still correspond to its last decode; accept only the
    // prefix being committed. No speculative suffix is injected into history.
    void process(const std::vector<llama_token> &ids,int pos,bool proposed_carry=false) {
        if(mtp) {
            auto h=target_hidden(int(ids.size()));
            if(proposed_carry) {
                pending.assign(h.begin(),h.begin()+width);
                if(ids.size()>1) {
                    std::vector<float> shifted(h.begin(),h.end()-width);
                    mtp_input({ids.begin()+1,ids.end()},pos+1,shifted,false);
                }
            } else {
                auto shifted=pending;shifted.insert(shifted.end(),h.begin(),h.end()-width);
                mtp_input(ids,pos,shifted,false);
            }
            pending.assign(h.end()-width,h.end());
        } else {
            DraftBatch b(int(ids.size()),width*int(layers.size()));b.positions(int(ids.size()),pos,false);
            for(size_t k=0;k<layers.size();++k) {
                const auto *h=llama_get_embeddings_layer_inp(target,layers[k]);require(h,"missing target DFlash features");
                for(size_t i=0;i<ids.size();++i)std::copy_n(h+i*width,width,b.b.embd+(i*layers.size()+k)*width);
            }
            evaluate(b);
        }
    }
    std::vector<llama_token> propose(llama_token anchor,int pos,int depth,float pmin) {
        if(!depth)return {};
        if(mtp) {
            require(depth==1,"MiMo MTP probe uses first trained head only");
            mtp_input({anchor},pos,pending,true);
        } else {
            require(depth<=7,"DFlash maximum is seven mask predictions, plus anchor");
            DraftBatch b(depth+1);b.positions(depth+1,pos,true);b.b.token[0]=anchor;
            for(int i=1;i<=depth;++i)b.b.token[i]=151675;evaluate(b);
        }
        std::vector<llama_token> out;
        const int vocab=llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(draft)));
        for(int i=0;i<depth;++i) {
            const int row=mtp?0:i+1;auto id=draft_greedy(draft,row);const auto *l=llama_get_logits_ith(draft,row);
            if(pmin>0) {
                double sum=0;for(int j=0;j<vocab;++j)sum+=std::exp(double(l[j])-l[id]);
                if(1.0/sum<pmin)break;
            }
            out.push_back(id);if(is_stop(id))break;
        }
        if(!mtp)require(llama_memory_seq_rm(llama_get_memory(draft),0,pos,-1),"DFlash noise rollback failed");
        return out;
    }
};
}
