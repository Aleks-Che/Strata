#pragma once
// Header-only admission before weight allocation. Fixture opt-in is unavailable in the CLI.
#include "ggml.h"
#include "gguf.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
namespace minimax_m2 {
inline void require(bool ok,const std::string &message) {if(!ok)throw std::runtime_error(message);}
struct Contract {uint32_t blocks,width,vocab;uint64_t fixed_bytes=0,routed_bytes=0;};
inline Contract inspect(const std::string &path,bool fixture=false) {
    ggml_context *raw=nullptr;
    std::unique_ptr<gguf_context,decltype(&gguf_free)> file(gguf_init_from_file(path.c_str(),{true,&raw}),gguf_free);
    std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(raw,ggml_free);
    require(file && ctx,"cannot read MiniMax GGUF header");auto *f=file.get();
    require(gguf_get_version(f)==3,"only GGUF v3 is admitted");
    auto key=[&](const std::string &name,gguf_type type) {
        auto id=gguf_find_key(f,name.c_str());
        require(id>=0 && gguf_get_kv_type(f,id)==type,"missing/wrong metadata: "+name);return id;
    };
    auto str=[&](const std::string &n) {return std::string(gguf_get_val_str(f,key(n,GGUF_TYPE_STRING)));};
    auto u=[&](const std::string &n) {return gguf_get_val_u32(f,key("minimax-m2."+n,GGUF_TYPE_UINT32));};
    auto real=[&](const std::string &n) {return gguf_get_val_f32(f,key("minimax-m2."+n,GGUF_TYPE_FLOAT32));};
    require(str("general.architecture")=="minimax-m2","expected minimax-m2 architecture");
    const std::set<std::string> allowed={"block_count","context_length","embedding_length","feed_forward_length",
        "expert_feed_forward_length","expert_count","expert_used_count","expert_gating_func","attention.head_count",
        "attention.head_count_kv","attention.key_length","attention.value_length","attention.layer_norm_rms_epsilon",
        "rope.dimension_count","rope.freq_base"};
    for(int64_t i=0;i<gguf_get_n_kv(f);++i) {
        std::string n=gguf_get_key(f,i);require(n.rfind("split.",0)!=0,"split GGUF is not admitted");
        if(n.rfind("minimax-m2.",0)==0)require(allowed.count(n.substr(11)) ||
            (fixture && n=="minimax-m2.vocab_size"),"unreviewed metadata: "+n);
    }
    const auto blocks=u("block_count"),d=u("embedding_length"),ff=u("feed_forward_length"),e=u("expert_count");
    require(u("expert_feed_forward_length")==ff && u("expert_used_count")==8 && u("expert_gating_func")==2,"unsupported routing");
    require(u("attention.head_count")==48 && u("attention.head_count_kv")==8 && u("attention.key_length")==128 &&
        u("attention.value_length")==128 && u("rope.dimension_count")==64,"unsupported attention geometry");
    require(real("rope.freq_base")==5000000.f && std::abs(real("attention.layer_norm_rms_epsilon")-1e-6f)<1e-12f,
        "unsupported RoPE/normalization");
    uint32_t vocab=0;
    if(fixture) {
        vocab=u("vocab_size");
        require(str("tokenizer.ggml.model")=="none" && blocks==3 && d==256 && ff==512 && e==16 && vocab==64 &&
            u("context_length")==512,"unreviewed fixture geometry");
    } else {
        require(blocks==62 && d==3072 && ff==1536 && e==256 && u("context_length")==204800,"unreviewed production geometry");
        require(str("tokenizer.ggml.model")=="gpt2" && str("tokenizer.ggml.pre")=="minimax-m2","unsupported tokenizer");
        auto tokens=key("tokenizer.ggml.tokens",GGUF_TYPE_ARRAY),merges=key("tokenizer.ggml.merges",GGUF_TYPE_ARRAY);
        require(gguf_get_arr_type(f,tokens)==GGUF_TYPE_STRING && gguf_get_arr_n(f,tokens)==200064 &&
            gguf_get_arr_type(f,merges)==GGUF_TYPE_STRING && gguf_get_arr_n(f,merges)==199744,"unsupported vocabulary");
        vocab=200064;
        for(const auto &s:std::vector<std::pair<std::string,uint32_t>>{{"bos",200034},{"eos",200020},{"padding",200020},{"unknown",200021}})
            require(gguf_get_val_u32(f,key("tokenizer.ggml."+s.first+"_token_id",GGUF_TYPE_UINT32))==s.second,"unsupported special token ID");
        for(const auto &s:std::vector<std::pair<uint32_t,std::string>>{{200034,"]~!b["},{200020,"[e~["},{200021,"]!d~["}})
            require(gguf_get_arr_str(f,tokens,s.first)==s.second,"unsupported special token literal");
        require(!str("tokenizer.chat_template").empty(),"missing chat template");
    }
    struct Expected {std::vector<int64_t> shape;int type;bool routed;};
    std::map<std::string,Expected> expected;
    auto add=[&](std::string n,std::vector<int64_t> shape,int type=0,bool routed=false) {expected.emplace(n,Expected{shape,type,routed});};
    // 0 Q4_K, 1 F32, 2 Q6_K, 3 Q4_K or Q6_K.
    add("token_embd.weight",{d,vocab});add("output.weight",{d,vocab},2);add("output_norm.weight",{d},1);
    for(uint32_t i=0;i<blocks;++i) {
        const auto p="blk."+std::to_string(i)+".";
        add(p+"attn_norm.weight",{d},1);add(p+"ffn_norm.weight",{d},1);
        add(p+"attn_q.weight",{d,6144});add(p+"attn_k.weight",{d,1024});add(p+"attn_v.weight",{d,1024},3);
        add(p+"attn_output.weight",{6144,d});add(p+"attn_q_norm.weight",{6144},1);add(p+"attn_k_norm.weight",{1024},1);
        add(p+"ffn_gate_inp.weight",{d,e},1);add(p+"exp_probs_b.bias",{e},1);
        add(p+"ffn_gate_exps.weight",{d,ff,e},0,true);add(p+"ffn_up_exps.weight",{d,ff,e},0,true);add(p+"ffn_down_exps.weight",{ff,d,e},3,true);
    }
    require(size_t(gguf_get_n_tensors(f))==expected.size(),"unexpected MiniMax tensor count");
    const uint64_t size=std::filesystem::file_size(path),base=gguf_get_data_offset(f),alignment=gguf_get_alignment(f);
    require(alignment==32 && base<=size,"invalid alignment/data offset");
    Contract result{blocks,d,vocab};std::vector<std::pair<uint64_t,uint64_t>> ranges;std::set<std::string> names;
    for(int64_t i=0;i<gguf_get_n_tensors(f);++i) {
        const std::string n=gguf_get_tensor_name(f,i);auto found=expected.find(n);
        require(found!=expected.end() && names.insert(n).second,"unexpected/duplicate tensor: "+n);
        const auto &exp=found->second;auto *t=ggml_get_tensor(ctx.get(),n.c_str());require(t,"missing tensor");
        for(size_t j=0;j<4;++j)require(t->ne[j]==(j<exp.shape.size()?exp.shape[j]:1),"shape mismatch: "+n);
        const bool type_ok=exp.type==0?t->type==GGML_TYPE_Q4_K:exp.type==1?t->type==GGML_TYPE_F32:
            exp.type==2?t->type==GGML_TYPE_Q6_K:t->type==GGML_TYPE_Q4_K || t->type==GGML_TYPE_Q6_K;
        require(type_ok || (fixture && t->type==GGML_TYPE_F32),"unsupported tensor encoding: "+n);
        const uint64_t offset=gguf_get_tensor_offset(f,i),bytes=gguf_get_tensor_size(f,i);
        require(offset%alignment==0 && offset<=size-base && bytes<=size-base-offset && bytes==ggml_nbytes(t),"invalid tensor range: "+n);
        ranges.emplace_back(base+offset,base+offset+bytes);(exp.routed?result.routed_bytes:result.fixed_bytes)+=bytes;
    }
    std::sort(ranges.begin(),ranges.end());uint64_t end=base;
    for(const auto &r:ranges) {require(r.first>=end,"overlapping tensor ranges");end=r.second;}
    return result;
}
}
