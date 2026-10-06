#pragma once
// Native pre-allocation counterpart of tools/hy3_loader_contract.py.
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
namespace hy3 {
inline void require(bool ok,const std::string & message) {if(!ok) throw std::runtime_error(message);}
struct Contract {uint32_t blocks,width,vocab;uint64_t fixed_bytes=0,routed_bytes=0,mtp_bytes=0;};
inline Contract inspect(const std::string & path,bool fixture=false) {
    ggml_context * raw=nullptr;
    std::unique_ptr<gguf_context,decltype(&gguf_free)> file(gguf_init_from_file(path.c_str(),{true,&raw}),gguf_free);
    std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(raw,ggml_free);
    require(file && ctx,"cannot read Hy3 GGUF header"); auto * f=file.get();
    require(gguf_get_version(f)==3,"only GGUF v3 is admitted");
    const auto key=[&](const std::string & name,gguf_type type) {
        auto id=gguf_find_key(f,name.c_str());
        require(id>=0 && gguf_get_kv_type(f,id)==type,"missing/wrong metadata: "+name); return id;
    };
    const auto str=[&](const std::string & name) {return std::string(gguf_get_val_str(f,key(name,GGUF_TYPE_STRING)));};
    const auto u=[&](const std::string & name) {return gguf_get_val_u32(f,key("hy_v3."+name,GGUF_TYPE_UINT32));};
    const auto real=[&](const std::string & name) {return gguf_get_val_f32(f,key("hy_v3."+name,GGUF_TYPE_FLOAT32));};
    require(str("general.architecture")=="hy_v3","expected hy_v3 architecture");
    const std::set<std::string> allowed={"block_count","nextn_predict_layers","context_length","embedding_length",
        "feed_forward_length","expert_feed_forward_length","expert_shared_feed_forward_length","expert_count",
        "expert_used_count","attention.head_count","attention.head_count_kv","attention.key_length",
        "attention.value_length","expert_gating_func","expert_weights_norm","rope.freq_base",
        "attention.layer_norm_rms_epsilon","expert_weights_scale"};
    for(int64_t i=0;i<gguf_get_n_kv(f);++i) {
        std::string name=gguf_get_key(f,i);
        require(name.rfind("split.",0)!=0,"split Hy3 GGUF not admitted");
        if(name.rfind("hy_v3.",0)==0) require(allowed.count(name.substr(6)) ||
            (fixture && (name=="hy_v3.vocab_size" || name=="hy_v3.rope.dimension_count")),"unreviewed metadata: "+name);
    }
    const uint32_t blocks=u("block_count"),d=u("embedding_length"),ff=u("feed_forward_length"),
        ef=u("expert_feed_forward_length"),sf=u("expert_shared_feed_forward_length"),e=u("expert_count"),
        heads=u("attention.head_count"),kv=u("attention.head_count_kv"),hd=u("attention.key_length");
    require(blocks>=3 && blocks<=512 && u("nextn_predict_layers")==1 && d>0 && d<=16384 &&
        ff>0 && ef>0 && sf>0 && ff<=131072 && ef<=65536 && sf<=65536 && e>=8 && e<=1024 &&
        heads>0 && heads<=256 && kv>0 && heads%kv==0 && hd==128 && u("attention.value_length")==hd &&
        u("expert_used_count")==8 && u("expert_gating_func")==2,"unsupported Hy3 dimensions/routing");
    require(gguf_get_val_bool(f,key("hy_v3.expert_weights_norm",GGUF_TYPE_BOOL)) &&
        std::abs(real("expert_weights_scale")-2.826f)<1e-6f && real("rope.freq_base")==11158840.f &&
        std::abs(real("attention.layer_norm_rms_epsilon")-1e-5f)<1e-10f,"unsupported Hy3 normalization/RoPE");
    uint32_t vocab=0;
    if(fixture) {
        require(str("tokenizer.ggml.model")=="none","fixture flag is only for tokenizer-free synthetic models");
        vocab=u("vocab_size");require(blocks==3 && d==256 && vocab==64,"unsupported synthetic fixture");
    } else {
        require(str("tokenizer.ggml.model")=="gpt2" && str("tokenizer.ggml.pre")=="hunyuan-dense","unsupported tokenizer");
        auto id=key("tokenizer.ggml.tokens",GGUF_TYPE_ARRAY);
        require(gguf_get_arr_type(f,id)==GGUF_TYPE_STRING,"invalid vocabulary");
        vocab=uint32_t(gguf_get_arr_n(f,id));
        require(blocks==81 && d==4096 && ff==13312 && ef==1536 && sf==1536 && e==192 &&
            heads==64 && kv==8 && vocab==120832 && u("context_length")==262144,"unreviewed full Hy3 dimensions");
        for(const auto & p:std::map<std::string,uint32_t>{{"bos_token_id",120000},{"eos_token_id",120025},
                {"padding_token_id",120002},{"seperator_token_id",120007}})
            require(gguf_get_val_u32(f,key("tokenizer.ggml."+p.first,GGUF_TYPE_UINT32))==p.second,"unexpected special token");
        require(!str("tokenizer.chat_template").empty(),"missing chat template");
    }
    struct Expected {std::vector<int64_t> shape;bool f32;int layer;bool routed;};
    std::map<std::string,Expected> expected;
    auto add=[&](std::string name,std::vector<int64_t> shape,bool f32=false,int layer=-1,bool routed=false) {
        expected.emplace(name,Expected{shape,f32,layer,routed});
    };
    add("token_embd.weight",{d,vocab});add("output.weight",{d,vocab});add("output_norm.weight",{d},true);
    for(uint32_t il=0;il<blocks;++il) {
        const std::string p="blk."+std::to_string(il)+".";
        for(const std::string n:{"attn_norm","ffn_norm"}) add(p+n+".weight",{d},true,il);
        for(const std::string n:{"attn_q_norm","attn_k_norm"}) add(p+n+".weight",{hd},true,il);
        add(p+"attn_q.weight",{d,hd*heads},false,il);add(p+"attn_k.weight",{d,hd*kv},false,il);
        add(p+"attn_v.weight",{d,hd*kv},false,il);add(p+"attn_output.weight",{hd*heads,d},false,il);
        if(il==0) {
            add(p+"ffn_gate.weight",{d,ff},false,il);add(p+"ffn_up.weight",{d,ff},false,il);
            add(p+"ffn_down.weight",{ff,d},false,il);
        } else {
            add(p+"ffn_gate_inp.weight",{d,e},true,il);add(p+"exp_probs_b",{e},true,il);
            for(const std::string kind:{"gate","up","down"}) {
                add(p+"ffn_"+kind+"_exps.weight",kind=="down" ? std::vector<int64_t>{ef,d,e}:std::vector<int64_t>{d,ef,e},false,il,true);
                add(p+"ffn_"+kind+"_shexp.weight",kind=="down" ? std::vector<int64_t>{sf,d}:std::vector<int64_t>{d,sf},false,il);
            }
        }
        if(il==blocks-1) {
            add(p+"nextn.eh_proj.weight",{2*d,d},false,il);
            for(const std::string n:{"enorm","hnorm","shared_head_norm"}) add(p+"nextn."+n+".weight",{d},true,il);
        }
    }
    require(size_t(gguf_get_n_tensors(f))==expected.size(),"unexpected Hy3 tensor count");
    const uint64_t size=std::filesystem::file_size(path),base=gguf_get_data_offset(f),alignment=gguf_get_alignment(f);
    require(alignment==32 && base<=size,"invalid Hy3 alignment/data offset");
    Contract result{blocks,d,vocab}; std::vector<std::pair<uint64_t,uint64_t>> ranges;
    std::set<std::string> names;
    for(int64_t i=0;i<gguf_get_n_tensors(f);++i) {
        const std::string name=gguf_get_tensor_name(f,i);
        auto found=expected.find(name);require(found!=expected.end() && names.insert(name).second,"unexpected/duplicate tensor: "+name);
        const auto & exp=found->second;auto * t=ggml_get_tensor(ctx.get(),name.c_str());
        require(t,"missing tensor header");
        for(size_t j=0;j<4;++j) require(t->ne[j]==(j<exp.shape.size()?exp.shape[j]:1),"shape mismatch: "+name);
        const std::set<ggml_type> types={GGML_TYPE_F32,GGML_TYPE_Q8_0,GGML_TYPE_Q6_K,GGML_TYPE_Q5_K,
            GGML_TYPE_Q4_K,GGML_TYPE_Q3_K,GGML_TYPE_IQ3_XXS,GGML_TYPE_IQ4_XS};
        require(types.count(t->type) && (!exp.f32 || t->type==GGML_TYPE_F32),"unreviewed tensor type: "+name);
        const uint64_t offset=gguf_get_tensor_offset(f,i),bytes=gguf_get_tensor_size(f,i);
        require(offset%alignment==0 && offset<=size-base && bytes<=size-base-offset && bytes==ggml_nbytes(t),"invalid tensor range: "+name);
        ranges.emplace_back(base+offset,base+offset+bytes);
        if(exp.layer==int(blocks-1)) result.mtp_bytes+=bytes;
        else if(exp.routed) result.routed_bytes+=bytes;
        else result.fixed_bytes+=bytes;
    }
    std::sort(ranges.begin(),ranges.end());uint64_t end=base;
    for(const auto & r:ranges) {require(r.first>=end,"overlapping tensor ranges");end=r.second;}
    return result;
}
} // namespace hy3
