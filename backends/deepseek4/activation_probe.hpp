#pragma once
// Opt-in, bounded GPU activation readback for the serial verification probe.
// cb_eval introduces synchronization and can prevent fusion: its effect on final
// logits is measured separately. This is never used for performance timings.
#include "ggml-backend.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

class ActivationProbe {
public:
    using Json=nlohmann::ordered_json;
    struct Value {
        std::string name,op,type;
        std::array<int64_t,4> shape;
        std::vector<float> data;
        std::vector<std::string> sources;
        Json attention=Json::array();
    };
    static Json compare(const Value &batch,const Value &one,int tokens,int row) {
        Json r={{"name",batch.name},{"op",batch.op},{"type",batch.type},
            {"batch_shape",batch.shape},{"serial_shape",one.shape},{"sources",batch.sources},{"row",row}};
        if(!batch.attention.empty()) {r["batch_attention"]=batch.attention;r["serial_attention"]=one.attention;}
        int axis=-1;
        for(int i=0;i<4;++i)if(batch.shape[i]!=one.shape[i]) {
            if(axis!=-1 || batch.shape[i]!=tokens || one.shape[i]!=1) {
                r["skipped"]="shape is not one token axis";return r;
            }
            axis=i;
        }
        if(axis<0) {r["skipped"]="no token axis";return r;}
        if(row<0 || row>=tokens || batch.data.size()!=one.data.size()*tokens)
            throw std::runtime_error("activation probe invalid token slice");
        size_t inner=1;for(int i=0;i<axis;++i)inner*=one.shape[i];
        double sum=0,ref=0,max_abs=0;size_t at=0,unequal=0,nonfinite=0;
        for(size_t i=0;i<one.data.size();++i) {
            const size_t b=(i/inner*tokens+row)*inner+i%inner;
            const float x=batch.data[b],y=one.data[i];
            if(!std::isfinite(x) || !std::isfinite(y)){++nonfinite;continue;}
            double d=double(x)-y;sum+=d*d;ref+=double(x)*x;
            unequal+=x!=y;
            if(std::abs(d)>max_abs){max_abs=std::abs(d);at=i;}
        }
        const size_t b=(at/inner*tokens+row)*inner+at%inner;
        r["axis"]=axis;r["values"]=one.data.size();r["unequal"]=unequal;r["nonfinite"]=nonfinite;
        r["max_abs"]=max_abs;r["rms"]=std::sqrt(sum/one.data.size());
        r["reference_rms"]=std::sqrt(ref/one.data.size());
        r["max_index"]=at;r["batch_value"]=batch.data[b];r["serial_value"]=one.data[at];
        return r;
    }
private:
    std::string path;
    std::set<std::string> names;
    int layer=0,phase=0,position=0,tokens=0,row=0;
    size_t bytes=0;
    std::vector<Value> reference;
    std::map<std::string,size_t> occurrences,lookup;
    Json comparisons=Json::array();
    std::string error;
    bool attention_seen=false;
    std::vector<std::vector<uint8_t>> visible_reference;
    void attention_inputs(const ggml_tensor *t) {
        const auto *k=t->src[1],*v=t->src[2],*mask=t->src[3];
        if(!mask || mask->type!=GGML_TYPE_F16 || k->type!=GGML_TYPE_F16 || v->type!=GGML_TYPE_F16 ||
            k->ne[2]!=1 || k->ne[3]!=1 || v->ne[2]!=1 || v->ne[3]!=1 || mask->ne[2]!=1 || mask->ne[3]!=1)
            throw std::runtime_error("visible attention probe requires one F16 KV head/sequence and F16 mask");
        if(ggml_nbytes(k)+ggml_nbytes(v)+ggml_nbytes(mask)>(64ULL<<20))throw std::runtime_error("visible attention readback exceeds limit");
        std::vector<uint8_t> kr(ggml_nbytes(k)),vr(ggml_nbytes(v)),mr(ggml_nbytes(mask));
        ggml_backend_tensor_get(k,kr.data(),0,kr.size());ggml_backend_tensor_get(v,vr.data(),0,vr.size());
        ggml_backend_tensor_get(mask,mr.data(),0,mr.size());
        const int count=phase==1?tokens:1;
        for(int query=0;query<count;++query) {
            std::vector<uint8_t> visible;size_t keys=0;
            auto append=[&](const std::vector<uint8_t>&data,size_t offset,size_t size){
                if(offset+size>data.size())throw std::runtime_error("visible attention view exceeds backing tensor");
                visible.insert(visible.end(),data.begin()+offset,data.begin()+offset+size);
            };
            for(int64_t i=0;i<k->ne[1];++i) {
                ggml_fp16_t m;const size_t offset=query*mask->nb[1]+i*mask->nb[0];
                if(offset+sizeof(m)>mr.size())throw std::runtime_error("visible attention mask shape mismatch");
                std::memcpy(&m,mr.data()+offset,sizeof(m));
                if(!std::isfinite(ggml_fp16_to_fp32(m)))continue;
                ++keys;append(mr,offset,sizeof(m));
                for(const auto &pair:{std::make_pair(k,&kr),std::make_pair(v,&vr)})
                    for(int64_t j=0;j<pair.first->ne[0];++j)append(*pair.second,i*pair.first->nb[1]+j*pair.first->nb[0],2);
            }
            if(bytes+visible.size()>(256ULL<<20))throw std::runtime_error("visible attention reference exceeds limit");
            bytes+=visible.size();
            if(phase==1)visible_reference.push_back(std::move(visible));
            else {
                if(size_t(row)>=visible_reference.size())throw std::runtime_error("missing visible attention reference");
                comparisons.push_back({{"name","strata_fattn0_visible#0"},{"row",row},{"visible_keys",keys},
                    {"bytes",visible.size()},{"batch_bytes",visible_reference[row].size()},
                    {"visible_inputs_equal",visible==visible_reference[row]}});
            }
        }
    }
    bool selected(const ggml_tensor *t) const {
        if(!phase || t->op==GGML_OP_NONE)return false;
        if(t->op==GGML_OP_FLASH_ATTN_EXT && names.count("@attention0"))return !attention_seen;
        if(t->type!=GGML_TYPE_F32 && t->type!=GGML_TYPE_F16 && t->type!=GGML_TYPE_BF16 && t->type!=GGML_TYPE_I32)return false;
        const std::string name=t->name;
        if(!names.empty())return names.count(name)!=0;
        if(name=="hc_init" || name=="hc_head" || name=="result_norm" || name=="result_output")return true;
        auto dash=name.rfind('-');
        if(dash==std::string::npos || dash+1==name.size())return false;
        for(size_t i=dash+1;i<name.size();++i)if(name[i]<'0'||name[i]>'9')return false;
        return layer<0 || std::stoi(name.substr(dash+1))==layer;
    }
    void capture(ggml_tensor *t,const char *label=nullptr) {
        if(ggml_nbytes(t)>(32ULL<<20) || bytes+ggml_nelements(t)*sizeof(float)>(256ULL<<20))
            throw std::runtime_error("activation probe exceeds bounded readback size");
        const std::string name=label?label:t->name;
        const std::string key=name+"#"+std::to_string(occurrences[name]++);
        Value v;v.name=key;v.op=ggml_op_name(t->op);v.type=ggml_type_name(t->type);
        std::copy_n(t->ne,4,v.shape.begin());
        for(auto *src:t->src)if(src)v.sources.push_back(src->name);
        if(t->op==GGML_OP_FLASH_ATTN_EXT || std::string(t->name).find("attn_raw-")==0) {
            const ggml_tensor *fa=t;
            for(int depth=0;fa && depth<12;++depth,fa=fa->src[fa->op==GGML_OP_MUL_MAT?1:0])if(fa->op==GGML_OP_FLASH_ATTN_EXT) {
                for(int i=0;i<5;++i)if(auto *src=fa->src[i])
                    v.attention.push_back({{"slot",i},{"name",src->name},{"type",ggml_type_name(src->type)},
                        {"shape",std::vector<int64_t>(src->ne,src->ne+4)},
                        {"strides",std::vector<size_t>(src->nb,src->nb+4)}});
                break;
            }
        }
        std::vector<uint8_t> raw(ggml_nbytes(t));ggml_backend_tensor_get(t,raw.data(),0,raw.size());
        v.data.resize(ggml_nelements(t));bytes+=v.data.size()*sizeof(float);
        for(size_t j=0;j<v.data.size();++j) {
            size_t q=j,offset=0;
            for(int i=0;i<4;++i){offset+=(q%t->ne[i])*t->nb[i];q/=t->ne[i];}
            if(t->type==GGML_TYPE_F32)std::memcpy(&v.data[j],raw.data()+offset,4);
            else if(t->type==GGML_TYPE_I32){int32_t x;std::memcpy(&x,raw.data()+offset,4);v.data[j]=float(x);}
            else if(t->type==GGML_TYPE_F16){ggml_fp16_t x;std::memcpy(&x,raw.data()+offset,2);v.data[j]=ggml_fp16_to_fp32(x);}
            else {ggml_bf16_t x;std::memcpy(&x,raw.data()+offset,2);v.data[j]=ggml_bf16_to_fp32(x);}
        }
        if(phase==1){lookup[key]=reference.size();reference.push_back(std::move(v));}
        else {
            auto it=lookup.find(key);
            if(it==lookup.end())comparisons.push_back({{"name",key},{"row",row},{"skipped","missing in batch"}});
            else comparisons.push_back(compare(reference[it->second],v,tokens,row));
        }
    }
public:
    ActivationProbe() {
        if(const char *p=std::getenv("STRATA_SPEC_TRACE_NODES"))path=p;
        if(path.empty())return;
        if(const char *p=std::getenv("STRATA_SPEC_TRACE_LAYER"))layer=std::stoi(p);
        if(const char *p=std::getenv("STRATA_SPEC_TRACE_NODE")) {
            std::stringstream in(p);std::string name;
            while(std::getline(in,name,','))if(!name.empty())names.insert(name);
        }
        if(layer < -1 || layer>42)throw std::runtime_error("trace layer must be -1 or 0..42");
    }
    bool enabled() const {return !path.empty();}
    void idle(){phase=0;}
    void batch(int pos,int count) {
        position=pos;tokens=count;phase=1;bytes=0;attention_seen=false;reference.clear();lookup.clear();occurrences.clear();
        comparisons=Json::array();visible_reference.clear();error.clear();
    }
    void serial(int token_row){phase=2;row=token_row;bytes=0;attention_seen=false;occurrences.clear();}
    static bool callback(ggml_tensor *t,bool ask,void *user) {
        auto &self=*static_cast<ActivationProbe*>(user);
        // Do not propagate C++ exceptions through the C backend callback.
        try{
            if(ask)return self.selected(t);
            if(t->op==GGML_OP_FLASH_ATTN_EXT && self.names.count("@attention0")) {
                self.attention_seen=true;
                self.capture(t,"strata_fattn0");
                self.capture(t->src[0],"strata_fattn0_q");
                self.attention_inputs(t);
            } else self.capture(t);
        }
        catch(const std::exception &e){self.error=e.what();self.idle();return !ask;}
        return true;
    }
    void finish(float instrumentation_diff,float serial_instrumentation_diff) {
        idle();
        if(!error.empty())throw std::runtime_error(error);
        if(reference.empty() || comparisons.empty())throw std::runtime_error("activation probe captured no nodes");
        std::ofstream out(path,std::ios::app);
        if(!out)throw std::runtime_error("cannot open activation probe output");
        out<<Json{{"position",position},{"tokens",tokens},{"layer",layer},
            {"instrumentation_max_logit_diff",instrumentation_diff},
            {"serial_instrumentation_max_logit_diff",serial_instrumentation_diff},
            {"comparisons",comparisons}}.dump()<<'\n';
        out.flush();
        if(!out)throw std::runtime_error("cannot write activation probe output");
        reference.clear();visible_reference.clear();
    }
};
