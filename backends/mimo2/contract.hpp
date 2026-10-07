#pragma once
// Pre-allocation layout/range checks, corresponding to mimo2_loader_contract.py.
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
namespace mimo2 {
inline void require(bool ok,const char *message) {if(!ok)throw std::runtime_error(message);}
inline void require(bool ok,const std::string &message) {if(!ok)throw std::runtime_error(message);}
struct Contract {uint32_t blocks,width,vocab;uint64_t fixed_bytes=0,routed_bytes=0;};
inline Contract inspect(const std::string &path,bool fixture=false) {
    ggml_context *raw=nullptr;
    std::unique_ptr<gguf_context,decltype(&gguf_free)> file(gguf_init_from_file(path.c_str(),{true,&raw}),gguf_free);
    std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(raw,ggml_free);
    require(file && ctx,"cannot read MiMo GGUF header");auto *f=file.get();
    require(gguf_get_version(f)==3,"only GGUF v3 is admitted");
    const auto key=[&](const std::string &name,gguf_type type) {
        auto id=gguf_find_key(f,name.c_str());
        require(id>=0 && gguf_get_kv_type(f,id)==type,"missing/wrong metadata: "+name);return id;
    };
    const auto str=[&](const std::string &name) {return std::string(gguf_get_val_str(f,key(name,GGUF_TYPE_STRING)));};
    const auto u=[&](const std::string &name) {return gguf_get_val_u32(f,key("mimo2."+name,GGUF_TYPE_UINT32));};
    const auto real=[&](const std::string &name) {return gguf_get_val_f32(f,key("mimo2."+name,GGUF_TYPE_FLOAT32));};
    require(str("general.architecture")=="mimo2","expected mimo2 architecture");
    const std::set<std::string> allowed={"block_count","context_length","embedding_length","feed_forward_length",
        "expert_feed_forward_length","expert_count","expert_used_count","expert_gating_func","expert_group_count",
        "expert_group_used_count","nextn_predict_layers","attention.head_count","attention.head_count_kv",
        "attention.key_length","attention.value_length","attention.sliding_window","attention.sliding_window_pattern",
        "attention.layer_norm_rms_epsilon","attention.value_scale","rope.dimension_count","rope.freq_base",
        "rope.freq_base_swa","export_scope","source_revision","source_quantization","converter_revision","reference_expert_storage"};
    for(int64_t i=0;i<gguf_get_n_kv(f);++i) {
        std::string name=gguf_get_key(f,i);
        require(name.rfind("split.",0)!=0,"split MiMo GGUF not admitted");
        if(name.rfind("mimo2.",0)==0)require(allowed.count(name.substr(6)) ||
            (fixture && name=="mimo2.vocab_size"),"unreviewed metadata: "+name);
    }
    const uint32_t blocks=u("block_count"),d=u("embedding_length"),ff=u("feed_forward_length"),
        ef=u("expert_feed_forward_length"),e=u("expert_count");
    require(u("nextn_predict_layers")==0 && u("expert_used_count")==8 && u("expert_gating_func")==2 &&
        u("expert_group_count")==1 && u("expert_group_used_count")==1,"unsupported MiMo routing/MTP");
    require(u("attention.head_count")==64 && u("attention.key_length")==192 && u("attention.value_length")==128 &&
        u("attention.sliding_window")==128 && u("rope.dimension_count")==64,"unsupported attention geometry");
    require(real("rope.freq_base")==1e7f && real("rope.freq_base_swa")==1e4f &&
        std::abs(real("attention.layer_norm_rms_epsilon")-1e-6f)<1e-12f &&
        std::abs(real("attention.value_scale")-.707f)<1e-7f,"unsupported RoPE/normalization/value scale");
    uint32_t vocab=0;
    if(fixture) {
        require(str("tokenizer.ggml.model")=="none","fixture requires tokenizer-free synthetic model");
        vocab=u("vocab_size");
        require(blocks==3 && d==256 && ff==512 && ef==256 && e==16 && vocab==64 && u("context_length")==1024,"unreviewed fixture geometry");
    } else {
        require(blocks==48 && d==4096 && ff==16384 && ef==2048 && e==256 && u("context_length")==1048576,"unreviewed production geometry");
        require(str("mimo2.export_scope")=="text_trunk_without_mtp_or_modality_companions" &&
            str("mimo2.source_revision")=="3b38d063180c3e4aed9691fdc735f3d10b266ee4" &&
            str("mimo2.converter_revision")=="58367713a6935c0810103378144008df32e3d5db","unreviewed export provenance");
        require(!str("mimo2.source_quantization").empty() && !str("mimo2.reference_expert_storage").empty(),"missing export provenance");
        require(str("tokenizer.ggml.model")=="gpt2" && str("tokenizer.ggml.pre")=="qwen2" &&
            !gguf_get_val_bool(f,key("tokenizer.ggml.add_bos_token",GGUF_TYPE_BOOL)),"unsupported tokenizer/BOS policy");
        auto tokens=key("tokenizer.ggml.tokens",GGUF_TYPE_ARRAY),merges=key("tokenizer.ggml.merges",GGUF_TYPE_ARRAY);
        require(gguf_get_arr_type(f,tokens)==GGUF_TYPE_STRING && gguf_get_arr_n(f,tokens)==152576 &&
            gguf_get_arr_type(f,merges)==GGUF_TYPE_STRING && gguf_get_arr_n(f,merges)==151387,"unsupported vocabulary");
        vocab=152576;
        require(gguf_get_val_u32(f,key("tokenizer.ggml.eos_token_id",GGUF_TYPE_UINT32))==151645 &&
            gguf_get_val_u32(f,key("tokenizer.ggml.padding_token_id",GGUF_TYPE_UINT32))==151643 &&
            std::string(gguf_get_arr_str(f,tokens,151645))=="<|im_end|>" &&
            std::string(gguf_get_arr_str(f,tokens,151643))=="<|endoftext|>" &&
            !str("tokenizer.chat_template").empty(),"unsupported special tokens/template");
    }
    auto array=[&](const char *name) {
        auto id=key(std::string("mimo2.")+name,GGUF_TYPE_ARRAY);
        // The reviewed export writes signed i32 arrays; our generated fixtures use u32.
        require(gguf_get_arr_type(f,id)==(fixture?GGUF_TYPE_UINT32:GGUF_TYPE_INT32) && gguf_get_arr_n(f,id)==blocks,"invalid attention array");
        auto *data=static_cast<const uint32_t *>(gguf_get_arr_data(f,id));return std::vector<uint32_t>(data,data+blocks);
    };
    auto kv=array("attention.head_count_kv"),swa=array("attention.sliding_window_pattern");
    const std::set<uint32_t> full=fixture?std::set<uint32_t>{0,2}:std::set<uint32_t>{0,5,11,17,23,29,35,41,47};
    for(uint32_t i=0;i<blocks;++i)require(swa[i]==(full.count(i)?0:1) && kv[i]==(swa[i]?8:4),"unsupported full/SWA pattern");
    struct Expected {std::vector<int64_t> shape;int family;}; // 0 BF16, 1 F32, 2 routed gate/up, 3 routed down
    std::map<std::string,Expected> expected;
    auto add=[&](std::string name,std::vector<int64_t> shape,int family=0) {expected.emplace(name,Expected{shape,family});};
    add("token_embd.weight",{d,vocab});add("output.weight",{d,vocab});add("output_norm.weight",{d},1);
    for(uint32_t i=0;i<blocks;++i) {
        const auto p="blk."+std::to_string(i)+".";
        add(p+"attn_norm.weight",{d},1);add(p+"ffn_norm.weight",{d},1);
        add(p+"attn_qkv.weight",{d,64*192+kv[i]*(192+128)});add(p+"attn_output.weight",{8192,d});
        if(swa[i])add(p+"attn_sinks.weight",{64},1);
        if(i==0) {
            add(p+"ffn_gate.weight",{d,ff});add(p+"ffn_up.weight",{d,ff});add(p+"ffn_down.weight",{ff,d});
        } else {
            add(p+"ffn_gate_inp.weight",{d,e},1);add(p+"exp_probs_b.bias",{e},1);
            add(p+"ffn_gate_exps.weight",{d,ef,e},2);add(p+"ffn_up_exps.weight",{d,ef,e},2);add(p+"ffn_down_exps.weight",{ef,d,e},3);
        }
    }
    require(size_t(gguf_get_n_tensors(f))==expected.size(),"unexpected MiMo tensor count");
    const uint64_t size=std::filesystem::file_size(path),base=gguf_get_data_offset(f),alignment=gguf_get_alignment(f);
    require(alignment==32 && base<=size,"invalid alignment/data offset");
    Contract result{blocks,d,vocab};std::vector<std::pair<uint64_t,uint64_t>> ranges;std::set<std::string> names;
    for(int64_t i=0;i<gguf_get_n_tensors(f);++i) {
        const std::string name=gguf_get_tensor_name(f,i);auto found=expected.find(name);
        require(found!=expected.end() && names.insert(name).second,"unexpected/duplicate tensor: "+name);
        const auto &exp=found->second;auto *t=ggml_get_tensor(ctx.get(),name.c_str());require(t,"missing tensor");
        for(size_t j=0;j<4;++j)require(t->ne[j]==(j<exp.shape.size()?exp.shape[j]:1),"shape mismatch: "+name);
        const bool type_ok=exp.family==0?t->type==GGML_TYPE_BF16:exp.family==1?t->type==GGML_TYPE_F32:
            t->type==GGML_TYPE_Q2_K || t->type==GGML_TYPE_Q3_K || (exp.family==3 && t->type==GGML_TYPE_MXFP4);
        require(type_ok || (fixture && t->type==GGML_TYPE_F32),"unsupported tensor encoding: "+name);
        const uint64_t offset=gguf_get_tensor_offset(f,i),bytes=gguf_get_tensor_size(f,i);
        require(offset%alignment==0 && offset<=size-base && bytes<=size-base-offset && bytes==ggml_nbytes(t),"invalid tensor range: "+name);
        ranges.emplace_back(base+offset,base+offset+bytes);
        (exp.family>=2?result.routed_bytes:result.fixed_bytes)+=bytes;
    }
    std::sort(ranges.begin(),ranges.end());uint64_t end=base;
    for(const auto &r:ranges) {require(r.first>=end,"overlapping tensor ranges");end=r.second;}
    return result;
}
} // namespace mimo2
