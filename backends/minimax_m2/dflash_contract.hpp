#pragma once
// Admission for the three reviewed MiniMax DFlash exports. No payload allocation.
#include "contract.hpp"
#include "llama-model.h"
#include <cstring>
namespace minimax_m2 {
inline const std::vector<int32_t> dflash_layers{2,17,31,45,60};
constexpr int dflash_vocab=200055, dflash_mask=200054, target_vocab=200064;
struct DraftHeader {
    std::unique_ptr<gguf_context,decltype(&gguf_free)> file{nullptr,gguf_free};
    std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx{nullptr,ggml_free};
    explicit DraftHeader(const std::string &path) {
        ggml_context *raw=nullptr;file.reset(gguf_init_from_file(path.c_str(),{true,&raw}));ctx.reset(raw);
        require(file && ctx,"cannot read draft/target header: "+path);
    }
    int64_t key(const std::string &name,gguf_type type) const {
        auto k=gguf_find_key(file.get(),name.c_str());
        require(k>=0 && gguf_get_kv_type(file.get(),k)==type,"missing/wrong draft metadata: "+name);return k;
    }
    uint32_t u(const std::string &n) const {return gguf_get_val_u32(file.get(),key(n,GGUF_TYPE_UINT32));}
    float f(const std::string &n) const {return gguf_get_val_f32(file.get(),key(n,GGUF_TYPE_FLOAT32));}
    std::string s(const std::string &n) const {return gguf_get_val_str(file.get(),key(n,GGUF_TYPE_STRING));}
};
inline uint64_t inspect_dflash(const std::string &target_path,const std::string &draft_path) {
    DraftHeader target(target_path),draft(draft_path);auto *f=draft.file.get();
    require(target.s("general.architecture")=="minimax-m2" && draft.s("general.architecture")=="dflash",
        "expected MiniMax target and DFlash sidecar");
    require(gguf_get_version(f)==3 && gguf_get_alignment(f)==32,"unsupported draft GGUF layout");
    const std::map<std::string,uint32_t> integers={{"block_count",5},{"context_length",196608},
        {"embedding_length",3072},{"feed_forward_length",22016},{"attention.head_count",32},
        {"attention.head_count_kv",32},{"attention.key_length",128},{"attention.value_length",128},
        {"block_size",8},{"rope.scaling.original_context_length",4096}};
    const std::map<std::string,float> reals={{"rope.freq_base",5000000.f},{"rope.scaling.factor",48.f},
        {"rope.scaling.yarn_beta_fast",1.f},{"rope.scaling.yarn_beta_slow",1.f},{"attention.layer_norm_rms_epsilon",1e-6f}};
    std::set<std::string> reviewed={"target_layers","rope.scaling.type"};
    for(const auto &v:integers) {require(draft.u("dflash."+v.first)==v.second,"unsupported draft geometry: "+v.first);reviewed.insert(v.first);}
    for(const auto &v:reals) {require(draft.f("dflash."+v.first)==v.second,"unsupported draft scaling: "+v.first);reviewed.insert(v.first);}
    require(draft.s("dflash.rope.scaling.type")=="yarn","expected draft YaRN");
    for(int64_t i=0;i<gguf_get_n_kv(f);++i) {
        const std::string n=gguf_get_key(f,i);require(n.rfind("split.",0)!=0,"split draft not admitted");
        if(n.rfind("dflash.",0)==0)require(reviewed.count(n.substr(7)),"unreviewed draft metadata: "+n);
    }
    const auto layers=draft.key("dflash.target_layers",GGUF_TYPE_ARRAY);
    require(gguf_get_arr_type(f,layers)==GGUF_TYPE_INT32 && gguf_get_arr_n(f,layers)==5 &&
        std::memcmp(gguf_get_arr_data(f,layers),dflash_layers.data(),5*sizeof(int32_t))==0,"wrong target feature layers");
    for(const char *name:{"model","pre"})require(draft.s(std::string("tokenizer.ggml.")+name)==target.s(std::string("tokenizer.ggml.")+name),"tokenizer family mismatch");
    for(const auto &v:std::map<std::string,uint32_t>{{"bos",200034},{"eos",200020},{"unknown",200021}}) {
        auto n="tokenizer.ggml."+v.first+"_token_id";
        require(draft.u(n)==v.second && target.u(n)==v.second,"special token mismatch");
    }
    require(draft.u("tokenizer.ggml.mask_token_id")==dflash_mask,"wrong MASK token");
    for(const char *suffix:{"tokens","merges","token_type"}) {
        const std::string n=std::string("tokenizer.ggml.")+suffix;
        const auto dk=draft.key(n,GGUF_TYPE_ARRAY),tk=target.key(n,GGUF_TYPE_ARRAY);
        const bool types=std::string(suffix)=="token_type",merges=std::string(suffix)=="merges";
        const auto type=types?GGUF_TYPE_INT32:GGUF_TYPE_STRING;
        const size_t nd=merges?199744:dflash_vocab,nt=merges?199744:target_vocab;
        require(gguf_get_arr_type(f,dk)==type && gguf_get_arr_type(target.file.get(),tk)==type &&
            gguf_get_arr_n(f,dk)==nd && gguf_get_arr_n(target.file.get(),tk)==nt,"wrong tokenizer arrays");
        if(types)require(std::memcmp(gguf_get_arr_data(f,dk),gguf_get_arr_data(target.file.get(),tk),nd*sizeof(int32_t))==0,"token types differ");
        else for(size_t i=0;i<nd;++i)require(std::strcmp(gguf_get_arr_str(f,dk,i),gguf_get_arr_str(target.file.get(),tk,i))==0,"tokens/merges differ");
        if(std::string(suffix)=="tokens") {
            require(std::string(gguf_get_arr_str(f,dk,dflash_mask))=="[PAD200054]","wrong MASK string");
            for(int i=dflash_vocab;i<target_vocab;++i)require(std::string(gguf_get_arr_str(target.file.get(),tk,i))=="[PAD"+std::to_string(i)+"]","target suffix contains non-padding tokens");
        }
    }
    std::map<std::string,std::vector<int64_t>> expected={{"fc.weight",{15360,3072}},
        {"enc.output_norm.weight",{3072}},{"output_norm.weight",{3072}}};
    for(int i=0;i<5;++i) {
        const auto p="blk."+std::to_string(i)+".";
        for(const char *n:{"attn_norm","ffn_norm"})expected[p+n+".weight"]={3072};
        for(const char *n:{"attn_q_norm","attn_k_norm"})expected[p+n+".weight"]={128};
        for(const char *n:{"attn_q","attn_k","attn_v"})expected[p+n+".weight"]={3072,4096};
        expected[p+"attn_output.weight"]={4096,3072};
        for(const char *n:{"ffn_gate","ffn_up"})expected[p+n+".weight"]={3072,22016};
        expected[p+"ffn_down.weight"]={22016,3072};
    }
    require(size_t(gguf_get_n_tensors(f))==expected.size(),"unexpected draft tensors");
    const uint64_t size=std::filesystem::file_size(draft_path),base=gguf_get_data_offset(f);
    require(base<=size,"truncated draft header");uint64_t total=0;
    std::vector<std::pair<uint64_t,uint64_t>> ranges;std::set<std::string> seen;
    for(int64_t i=0;i<gguf_get_n_tensors(f);++i) {
        const std::string n=gguf_get_tensor_name(f,i);auto e=expected.find(n);
        require(e!=expected.end() && seen.insert(n).second,"unexpected/duplicate draft tensor: "+n);
        auto *t=ggml_get_tensor(draft.ctx.get(),n.c_str());require(t,"missing draft tensor");
        for(size_t j=0;j<4;++j)require(t->ne[j]==(j<e->second.size()?e->second[j]:1),"draft shape mismatch: "+n);
        const bool norm=e->second.size()==1;
        require(norm?t->type==GGML_TYPE_F32:(t->type==GGML_TYPE_Q3_K || t->type==GGML_TYPE_Q4_K ||
            t->type==GGML_TYPE_Q5_K || t->type==GGML_TYPE_Q6_K),"unsupported draft tensor type: "+n);
        const uint64_t offset=gguf_get_tensor_offset(f,i),bytes=gguf_get_tensor_size(f,i);
        require(offset%32==0 && offset<=size-base && bytes<=size-base-offset && bytes==ggml_nbytes(t),"invalid draft range: "+n);
        ranges.emplace_back(base+offset,base+offset+bytes);total+=bytes;
    }
    std::sort(ranges.begin(),ranges.end());uint64_t end=base;
    for(const auto &r:ranges) {require(r.first>=end,"overlapping draft tensors");end=r.second;}
    return total;
}
inline void registered_dflash(const llama_model &m) {
    const auto &h=m.hparams;
    require(m.arch==LLM_ARCH_DFLASH && h.n_layer()==5 && h.n_embd==3072 && h.n_embd_inp_enc()==15360 &&
        m.target_layer_ids==dflash_layers && h.dflash_block_size==8 && m.vocab.n_tokens()==dflash_vocab,
        "registered draft geometry mismatch");
    require(!m.tok_embd && !m.output && !m.d2t && !m.dspark_markov_w1 && !m.dflash_selector_hidden,
        "expected borrowed embedding/head without Markov/selector");
    for(int i=0;i<5;++i)require(h.n_head(i)==32 && h.n_head_kv(i)==32 && h.n_rot(i)==128 &&
        h.n_embd_head_k(i)==128 && h.n_embd_head_v(i)==128 && !h.is_swa(i),"registered draft attention mismatch");
}
}
