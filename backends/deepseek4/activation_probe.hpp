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
        Json matrix=Json::object();
        Json slice=Json::object();
    };
    struct ReadSpan {size_t source,destination,size;};
    struct TokenSlice {
        int axis=-2;
        std::array<int64_t,4> shape{};
        size_t bytes=0;
        std::vector<ReadSpan> spans;
    };
    static constexpr int MAX_HISTORY_ROWS=128;
    // Plan a bounded, compact read of a token window, including permuted views.
    // -2 means absent/ambiguous token axis; -1 means an already single token.
    static TokenSlice token_slice(const ggml_tensor *t,int count,int row,int rows=1) {
        if(count<1 || row<0 || rows<1 || rows>MAX_HISTORY_ROWS || row>count-rows)
            throw std::runtime_error("invalid readback token window");
        TokenSlice s;std::copy_n(t->ne,4,s.shape.begin());
        if(count==1)s.axis=-1;
        else {
            for(int i=0;i<4;++i)if(t->ne[i]==count) {
                if(s.axis>=0){s.axis=-2;return s;}
                s.axis=i;
            }
            if(s.axis==-2)return s;
            s.shape[s.axis]=rows;
        }
        if(t->type!=GGML_TYPE_F32 && t->type!=GGML_TYPE_F16 && t->type!=GGML_TYPE_BF16 && t->type!=GGML_TYPE_I32)
            throw std::runtime_error("unsupported activation slice type");
        const size_t element=ggml_type_size(t->type),limit=32ULL<<20;
        size_t elements=1,inner=1;
        for(int i=0;i<4;++i) {
            if(s.shape[i]<1 || size_t(s.shape[i])>limit/element/elements)
                throw std::runtime_error("activation slice exceeds readback limit");
            elements*=s.shape[i];
        }
        for(int i=0;i<4;++i) {
            if(s.shape[i]!=1 && t->nb[i]!=inner*element)break;
            inner*=s.shape[i];
        }
        if(elements/inner>65536)throw std::runtime_error("activation slice has too many spans");
        s.bytes=elements*element;const size_t backing=ggml_nbytes(t);
        for(size_t j=0;j<elements;j+=inner) {
            size_t q=j,offset=s.axis>=0?size_t(row)*t->nb[s.axis]:0;
            for(int i=0;i<4;++i){offset+=(q%s.shape[i])*t->nb[i];q/=s.shape[i];}
            const size_t size=inner*element;
            if(offset>backing || size>backing-offset)throw std::runtime_error("activation slice exceeds tensor storage");
            s.spans.push_back({offset,j*element,size});
        }
        return s;
    }
    // Keep the axis from the original read plan: the compact window length
    // may equal a feature dimension, so re-inferring it would be ambiguous.
    static Value compact_row(const Value &v,int axis,int row,int rows) {
        if(rows<1 || rows>MAX_HISTORY_ROWS || row<0 || row>=rows || axis<0 || axis>3 || v.shape[axis]!=rows)
            throw std::runtime_error("invalid compact activation row");
        size_t size=1,inner=1;
        for(int i=0;i<4;++i) {
            if(v.shape[i]<1 || size>v.data.size()/size_t(v.shape[i]))
                throw std::runtime_error("compact activation storage mismatch");
            size*=v.shape[i];if(i<axis)inner*=v.shape[i];
        }
        if(size!=v.data.size())throw std::runtime_error("compact activation storage mismatch");
        Value one;one.name=v.name;one.op=v.op;one.type=v.type;one.shape=v.shape;one.shape[axis]=1;
        one.sources=v.sources;one.attention=v.attention;one.matrix=v.matrix;one.slice=v.slice;
        one.slice["window_offset"]=row;one.data.resize(size/rows);
        for(size_t i=0;i<one.data.size();++i)one.data[i]=v.data[(i/inner*rows+row)*inner+i%inner];
        return one;
    }
    // Select by the actual mask input, not by scheduler callback order.
    static bool attention_mask_matches(const ggml_tensor *t,const std::string &name) {
        return t->op==GGML_OP_FLASH_ATTN_EXT && t->src[3] && name==t->src[3]->name;
    }
    static bool attention_sinks_matches(const ggml_tensor *t,const std::string &name) {
        return t->op==GGML_OP_FLASH_ATTN_EXT && t->src[4] && name==t->src[4]->name;
    }
    static std::string matmul_selector(const ggml_tensor *t) {
        if(!t->src[0])return "";
        if(t->op==GGML_OP_MUL_MAT)return "@weight:"+std::string(t->src[0]->name);
        if(t->op==GGML_OP_MUL_MAT_ID) {
            // The scheduler names an offloaded copy CUDA0#<weight>#0.
            // Strip only that numeric copy suffix and a CUDA device prefix.
            std::string name=t->src[0]->name;
            const auto first=name.find('#'),last=name.rfind('#');
            if(name.rfind("CUDA",0)==0 && first>4 && first!=std::string::npos && last>first && last+1<name.size() &&
               std::all_of(name.begin()+4,name.begin()+first,[](char c){return c>='0' && c<='9';}) &&
               std::all_of(name.begin()+last+1,name.end(),[](char c){return c>='0' && c<='9';}))
                name=name.substr(first+1,last-first-1);
            return "@expert:"+name;
        }
        return "";
    }
    struct Visible {
        std::vector<uint8_t> data;
        std::vector<int64_t> indices;
    };
    static Visible visible_inputs(const ggml_tensor *k,const ggml_tensor *v,const ggml_tensor *mask,
            const std::vector<uint8_t> &kr,const std::vector<uint8_t> &vr,const std::vector<uint8_t> &mr,int query) {
        if(query<0 || query>=mask->ne[1] || mask->ne[0]!=k->ne[1] || v->ne[1]!=k->ne[1])
            throw std::runtime_error("visible attention mask shape mismatch");
        Visible visible;
        auto append=[&](const std::vector<uint8_t>&data,size_t offset,size_t size){
            if(offset>data.size() || size>data.size()-offset)throw std::runtime_error("visible attention view exceeds backing tensor");
            visible.data.insert(visible.data.end(),data.begin()+offset,data.begin()+offset+size);
        };
        for(int64_t i=0;i<k->ne[1];++i) {
            ggml_fp16_t m;const size_t offset=query*mask->nb[1]+i*mask->nb[0];
            if(offset>mr.size() || sizeof(m)>mr.size()-offset)throw std::runtime_error("visible attention mask exceeds backing tensor");
            std::memcpy(&m,mr.data()+offset,sizeof(m));
            const float value=ggml_fp16_to_fp32(m);
            if(std::isinf(value) && value<0)continue;
            if(!std::isfinite(value))throw std::runtime_error("invalid visible attention mask value");
            visible.indices.push_back(i);append(mr,offset,sizeof(m));
            for(const auto &pair:{std::make_pair(k,&kr),std::make_pair(v,&vr)})
                for(int64_t j=0;j<pair.first->ne[0];++j)append(*pair.second,i*pair.first->nb[1]+j*pair.first->nb[0],2);
        }
        return visible;
    }
    static Json compare_visible(const Visible &a,const Visible &b,int dk,int dv) {
        if(dk<1 || dv<1 || a.indices.size()!=b.indices.size())return {{"skipped","different visible dimensions"}};
        const size_t stride=2*(size_t(1)+dk+dv),body=stride*a.indices.size();
        if(a.data.size()<body || b.data.size()<body)throw std::runtime_error("truncated visible comparison");
        bool masks_equal=true;size_t unequal[2]={};double maximum[2]={};Json changed=Json::array();
        for(size_t i=0;i<a.indices.size();++i) {
            masks_equal &= std::memcmp(a.data.data()+i*stride,b.data.data()+i*stride,2)==0;
            bool different=false;
            for(int part=0;part<2;++part)for(int j=0;j<(part?dv:dk);++j) {
                const size_t offset=i*stride+2*(1+(part?dk:0)+j);
                ggml_fp16_t x,y;std::memcpy(&x,a.data.data()+offset,2);std::memcpy(&y,b.data.data()+offset,2);
                if(x!=y){++unequal[part];different=true;}
                const float xf=ggml_fp16_to_fp32(x),yf=ggml_fp16_to_fp32(y);
                if(!std::isfinite(xf) || !std::isfinite(yf))throw std::runtime_error("nonfinite visible KV comparison");
                maximum[part]=std::max(maximum[part],std::abs(double(xf)-yf));
            }
            if(different)changed.push_back({{"ordinal",i},{"batch_index",a.indices[i]},{"serial_index",b.indices[i]}});
        }
        const bool sinks_equal=a.data.size()==b.data.size() && std::equal(a.data.begin()+body,a.data.end(),b.data.begin()+body);
        return {{"masks_equal",masks_equal},{"sinks_equal",sinks_equal},{"changed_keys",changed},
            {"k_unequal",unequal[0]},{"v_unequal",unequal[1]},{"k_max_abs",maximum[0]},{"v_max_abs",maximum[1]}};
    }
    static Json compare(const Value &batch,const Value &one,int tokens,int row) {
        Json r={{"name",batch.name},{"op",batch.op},{"type",batch.type},
            {"batch_shape",batch.shape},{"serial_shape",one.shape},{"sources",batch.sources},{"row",row}};
        if(!batch.attention.empty()) {r["batch_attention"]=batch.attention;r["serial_attention"]=one.attention;}
        if(!batch.matrix.empty()) {r["batch_matrix"]=batch.matrix;r["serial_matrix"]=one.matrix;}
        if(!batch.slice.empty()) {r["batch_slice"]=batch.slice;r["serial_slice"]=one.slice;}
        int axis=-1;
        for(int i=0;i<4;++i)if(batch.shape[i]!=one.shape[i]) {
            if(axis!=-1 || batch.shape[i]!=tokens || one.shape[i]!=1) {
                r["skipped"]="shape is not one token axis";return r;
            }
            axis=i;
        }
        // Two one-token evaluations can have different context histories but
        // identical tensor shapes. No slicing is necessary in that case.
        if(axis<0 && tokens==1 && batch.shape==one.shape)axis=0;
        if(axis<0) {r["skipped"]="no token axis";return r;}
        if(one.data.empty()) {r["skipped"]="empty tensor";return r;}
        if(row<0 || row>=tokens || batch.data.size()!=one.data.size()*tokens)
            throw std::runtime_error("activation probe invalid token slice");
        size_t inner=1;for(int i=0;i<axis;++i)inner*=one.shape[i];
        double sum=0,ref=0,max_abs=0;size_t at=0,unequal=0,nonfinite=0;
        size_t f16_unequal=0,f16_compared=0;Json f16_examples=Json::array();
        for(size_t i=0;i<one.data.size();++i) {
            const size_t b=(i/inner*tokens+row)*inner+i%inner;
            const float x=batch.data[b],y=one.data[i];
            if(!std::isfinite(x) || !std::isfinite(y)){++nonfinite;continue;}
            double d=double(x)-y;sum+=d*d;ref+=double(x)*x;
            unequal+=x!=y;
            if(std::abs(x)<=65504 && std::abs(y)<=65504) {
                ++f16_compared;
                const auto hx=ggml_fp32_to_fp16(x),hy=ggml_fp32_to_fp16(y);
                if(hx!=hy) {
                    ++f16_unequal;
                    if(f16_examples.size()<16)f16_examples.push_back({{"index",i},{"a",x},{"b",y},
                        {"a_f16",ggml_fp16_to_fp32(hx)},{"b_f16",ggml_fp16_to_fp32(hy)}});
                }
            }
            if(std::abs(d)>max_abs){max_abs=std::abs(d);at=i;}
        }
        const size_t b=(at/inner*tokens+row)*inner+at%inner;
        r["axis"]=axis;r["values"]=one.data.size();r["unequal"]=unequal;r["nonfinite"]=nonfinite;
        r["max_abs"]=max_abs;r["rms"]=std::sqrt(sum/one.data.size());
        r["reference_rms"]=std::sqrt(ref/one.data.size());
        r["max_index"]=at;r["batch_value"]=batch.data[b];r["serial_value"]=one.data[at];
        r["rounded_f16_unequal"]=f16_unequal;r["rounded_f16_examples"]=f16_examples;
        r["rounded_f16_compared"]=f16_compared;
        // Expert IDs are small exact integers; retain the selected routes as
        // well as their difference. Never dump an unbounded activation array.
        if(batch.type=="i32" && one.type=="i32" && one.data.size()<=64) {
            r["batch_ids"]=Json::array();r["serial_ids"]=one.data;
            for(size_t i=0;i<one.data.size();++i)r["batch_ids"].push_back(batch.data[(i/inner*tokens+row)*inner+i%inner]);
        }
        return r;
    }
    // Compare one common token in two differently sized batches. Infer the
    // token axis only when unambiguous; do not guess for last-logit tensors.
    static Json compare_batches(const Value &a,const Value &b,int na,int ra,int nb,int rb) {
        if(na<1 || nb<1 || ra<0 || ra>=na || rb<0 || rb>=nb)
            throw std::runtime_error("invalid batch comparison rows");
        if(nb==1)return compare(a,b,na,ra);
        int axis=-1;
        for(int i=0;i<4;++i) {
            if(a.shape[i]==na && b.shape[i]==nb) {
                if(axis>=0)return {{"name",a.name},{"skipped","ambiguous batch token axis"}};
                axis=i;
            } else if(a.shape[i]!=b.shape[i])return {{"name",a.name},{"skipped","different non-token dimensions"}};
        }
        if(axis<0)return {{"name",a.name},{"skipped","no batch token axis"}};
        size_t size=1,inner=1;
        for(int i=0;i<4;++i){size*=b.shape[i];if(i<axis)inner*=b.shape[i];}
        if(size!=b.data.size() || a.data.size()/na!=size/nb || a.data.size()%na || !inner)
            throw std::runtime_error("batch comparison storage mismatch");
        Value one=b;one.shape[axis]=1;one.data.resize(size/nb);
        for(size_t i=0;i<one.data.size();++i)one.data[i]=b.data[(i/inner*nb+rb)*inner+i%inner];
        auto result=compare(a,one,na,ra);
        result["compared_batch_shape"]=b.shape;result["compared_tokens"]=nb;result["compared_row"]=rb;
        return result;
    }
private:
    std::string path,attention_mask,attention_sinks;
    std::set<std::string> names;
    int layer=0,phase=0,position=0,tokens=0,row=0,compared_tokens=1,compared_row=0,reference_row=-1;
    int history_rows=1;
    size_t bytes=0;
    std::vector<Value> reference;
    std::map<std::string,size_t> occurrences,lookup;
    Json comparisons=Json::array();
    std::string error;
    bool attention_seen=false;
    std::vector<Visible> visible_reference;
    bool attention_selected(const ggml_tensor *t) const {
        return t->op==GGML_OP_FLASH_ATTN_EXT && !attention_seen &&
            (names.count("@attention0") || (!attention_mask.empty() && attention_mask_matches(t,attention_mask)) ||
                (!attention_sinks.empty() && attention_sinks_matches(t,attention_sinks)));
    }
    void attention_inputs(const ggml_tensor *t,const std::string &label) {
        if(phase==2 && compared_tokens!=1 && reference_row<0)
            throw std::runtime_error("multi-token attention trace requires a selected historical row");
        const auto *k=t->src[1],*v=t->src[2],*mask=t->src[3];
        if(!mask || mask->type!=GGML_TYPE_F16 || k->type!=GGML_TYPE_F16 || v->type!=GGML_TYPE_F16 ||
            k->ne[2]!=1 || k->ne[3]!=1 || v->ne[2]!=1 || v->ne[3]!=1 || mask->ne[2]!=1 || mask->ne[3]!=1)
            throw std::runtime_error("visible attention probe requires one F16 KV head/sequence and F16 mask");
        if(ggml_nbytes(k)+ggml_nbytes(v)+ggml_nbytes(mask)>(64ULL<<20))throw std::runtime_error("visible attention readback exceeds limit");
        std::vector<uint8_t> kr(ggml_nbytes(k)),vr(ggml_nbytes(v)),mr(ggml_nbytes(mask));
        ggml_backend_tensor_get(k,kr.data(),0,kr.size());ggml_backend_tensor_get(v,vr.data(),0,vr.size());
        ggml_backend_tensor_get(mask,mr.data(),0,mr.size());
        const int count=reference_row>=0?history_rows:(phase==1?tokens:1);
        const int start=reference_row>=0?(phase==1?reference_row:compared_row):0;
        for(int query=start;query<start+count;++query) {
            auto visible=visible_inputs(k,v,mask,kr,vr,mr,query);
            // Sinks participate in the denominator even when no key is visible.
            if(const auto *s=t->src[4]) {
                if(s->type!=GGML_TYPE_F32 || ggml_nbytes(s)>(64ULL<<10))throw std::runtime_error("unsupported attention sinks");
                const size_t start=visible.data.size();visible.data.resize(start+ggml_nbytes(s));
                ggml_backend_tensor_get(s,visible.data.data()+start,0,ggml_nbytes(s));
            }
            if(bytes+visible.data.size()>(256ULL<<20))throw std::runtime_error("visible attention reference exceeds limit");
            bytes+=visible.data.size();
            if(phase==1)visible_reference.push_back(std::move(visible));
            else {
                const size_t at=reference_row>=0?size_t(query-start):size_t(row);
                if(at>=visible_reference.size())throw std::runtime_error("missing visible attention reference");
                comparisons.push_back({{"name",label+"_visible#0"},{"row",reference_row>=0?reference_row+query-start:row},{"visible_keys",visible.indices.size()},
                    {"bytes",visible.data.size()},{"batch_bytes",visible_reference[at].data.size()},
                    {"batch_indices",visible_reference[at].indices},{"serial_indices",visible.indices},
                    {"visible_inputs_equal",visible.data==visible_reference[at].data},
                    {"components",compare_visible(visible_reference[at],visible,int(k->ne[0]),int(v->ne[0]))}});
            }
        }
    }
    bool selected(const ggml_tensor *t) const {
        if(!phase || t->op==GGML_OP_NONE)return false;
        if(attention_selected(t))return true;
        if(matmul_selected(t))return true;
        if(t->type!=GGML_TYPE_F32 && t->type!=GGML_TYPE_F16 && t->type!=GGML_TYPE_BF16 && t->type!=GGML_TYPE_I32)return false;
        const std::string name=t->name;
        if(!names.empty())return names.count(name)!=0;
        if(name=="hc_init" || name=="hc_head" || name=="result_norm" || name=="result_output")return true;
        auto dash=name.rfind('-');
        if(dash==std::string::npos || dash+1==name.size())return false;
        for(size_t i=dash+1;i<name.size();++i)if(name[i]<'0'||name[i]>'9')return false;
        return layer<0 || std::stoi(name.substr(dash+1))==layer;
    }
    bool matmul_selected(const ggml_tensor *t) const {
        return names.count(matmul_selector(t))!=0;
    }
    void capture(ggml_tensor *t,const char *label=nullptr) {
        TokenSlice sliced;
        if(reference_row>=0)sliced=token_slice(t,phase==1?tokens:compared_tokens,phase==1?reference_row:compared_row,history_rows);
        const std::string name=label?label:t->name;
        const std::string key=name+"#"+std::to_string(occurrences[name]++);
        if(reference_row>=0 && sliced.axis==-2) {
            comparisons.push_back({{"name",key},{"phase",phase},{"skipped","no unique token axis for historical readback"}});return;
        }
        const size_t read_bytes=reference_row>=0?sliced.bytes:ggml_nbytes(t);
        const size_t elements=reference_row>=0?sliced.bytes/ggml_type_size(t->type):ggml_nelements(t);
        if(read_bytes>(32ULL<<20) || bytes+elements*sizeof(float)>(256ULL<<20))
            throw std::runtime_error("activation probe exceeds bounded readback size");
        Value v;v.name=key;v.op=ggml_op_name(t->op);v.type=ggml_type_name(t->type);
        std::copy_n(t->ne,4,v.shape.begin());
        if(reference_row>=0) {
            v.slice={{"shape",v.shape},{"axis",sliced.axis},{"row",phase==1?reference_row:compared_row},
                {"rows",history_rows},
                {"strides",std::vector<size_t>(t->nb,t->nb+4)},
                {"read_bytes",read_bytes},{"spans",sliced.spans.size()}};v.shape=sliced.shape;
        }
        for(auto *src:t->src)if(src)v.sources.push_back(src->name);
        if((t->op==GGML_OP_MUL_MAT || t->op==GGML_OP_MUL_MAT_ID) && t->src[0]) {
            auto *w=t->src[0];v.matrix={{"weight",w->name},{"type",ggml_type_name(w->type)},
                {"shape",std::vector<int64_t>(w->ne,w->ne+4)},{"precision",t->op_params[0]}};
            if(t->op==GGML_OP_MUL_MAT_ID) {
                v.matrix["layouts"]=Json::array();
                for(const auto *s:{w,t->src[1],t->src[2],t})if(s)
                    v.matrix["layouts"].push_back({{"name",s->name},{"type",ggml_type_name(s->type)},
                        {"shape",std::vector<int64_t>(s->ne,s->ne+4)},
                        {"strides",std::vector<size_t>(s->nb,s->nb+4)}});
            }
        }
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
        std::vector<uint8_t> raw(read_bytes);
        if(reference_row>=0)for(const auto &s:sliced.spans)ggml_backend_tensor_get(t,raw.data()+s.destination,s.source,s.size);
        else ggml_backend_tensor_get(t,raw.data(),0,raw.size());
        v.data.resize(elements);bytes+=v.data.size()*sizeof(float);
        for(size_t j=0;j<v.data.size();++j) {
            size_t q=j,offset=0;
            for(int i=0;i<4;++i){offset+=(q%t->ne[i])*t->nb[i];q/=t->ne[i];}
            if(reference_row>=0)offset=j*ggml_type_size(t->type);
            if(t->type==GGML_TYPE_F32)std::memcpy(&v.data[j],raw.data()+offset,4);
            else if(t->type==GGML_TYPE_I32){int32_t x;std::memcpy(&x,raw.data()+offset,4);v.data[j]=float(x);}
            else if(t->type==GGML_TYPE_F16){ggml_fp16_t x;std::memcpy(&x,raw.data()+offset,2);v.data[j]=ggml_fp16_to_fp32(x);}
            else {ggml_bf16_t x;std::memcpy(&x,raw.data()+offset,2);v.data[j]=ggml_bf16_to_fp32(x);}
        }
        if(phase==1){lookup[key]=reference.size();reference.push_back(std::move(v));}
        else {
            auto it=lookup.find(key);
            if(it==lookup.end())comparisons.push_back({{"name",key},{"row",row},{"skipped","missing in batch"}});
            else if(reference_row>=0) {
                const auto &ref=reference[it->second];
                for(int r=0;r<history_rows;++r) {
                    auto result=history_rows==1?compare(ref,v,1,0):
                        compare(compact_row(ref,ref.slice["axis"].get<int>(),r,history_rows),
                                compact_row(v,sliced.axis,r,history_rows),1,0);
                    result["row"]=reference_row+r;comparisons.push_back(std::move(result));
                }
            } else comparisons.push_back(compare_batches(reference[it->second],v,tokens,row,compared_tokens,compared_row));
        }
    }
public:
    ActivationProbe() {
        if(const char *p=std::getenv("STRATA_SPEC_TRACE_NODES"))path=p;
        if(path.empty())return;
        if(const char *p=std::getenv("STRATA_SPEC_TRACE_LAYER"))layer=std::stoi(p);
        if(const char *p=std::getenv("STRATA_SPEC_TRACE_NODE")) {
            std::stringstream in(p);std::string name;
            while(std::getline(in,name,','))if(!name.empty()) {
                if(name.rfind("@attention:",0)==0) {
                    if(!attention_mask.empty() || name.size()==11)throw std::runtime_error("specify one attention mask name");
                    attention_mask=name.substr(11);
                }
                if(name.rfind("@attention-sinks:",0)==0) {
                    if(!attention_sinks.empty() || name.size()==17)throw std::runtime_error("specify one attention sinks name");
                    attention_sinks=name.substr(17);
                }
                names.insert(name);
            }
            if(int(!attention_mask.empty())+int(!attention_sinks.empty())+int(names.count("@attention0"))>1)
                throw std::runtime_error("specify only one attention selector");
        }
        if(layer < -1 || layer>42)throw std::runtime_error("trace layer must be -1 or 0..42");
    }
    bool enabled() const {return !path.empty();}
    void idle(){phase=0;}
    void batch(int pos,int count,int trace_row=-1,int trace_rows=1) {
        if(trace_rows<1 || trace_rows>MAX_HISTORY_ROWS || trace_row < -1 ||
           (trace_row<0?trace_rows!=1:trace_row>count-trace_rows))throw std::runtime_error("invalid historical readback window");
        reference_row=trace_row;history_rows=trace_rows;
        position=pos;tokens=count;phase=1;bytes=0;attention_seen=false;reference.clear();lookup.clear();occurrences.clear();
        comparisons=Json::array();visible_reference.clear();error.clear();
    }
    void serial(int token_row,int count=1,int local_row=0) {
        if(reference_row>=0 && (local_row<0 || local_row>count-history_rows))
            throw std::runtime_error("historical window crosses compared batch");
        phase=2;row=token_row;compared_tokens=count;compared_row=local_row;bytes=0;attention_seen=false;occurrences.clear();
    }
    static bool callback(ggml_tensor *t,bool ask,void *user) {
        auto &self=*static_cast<ActivationProbe*>(user);
        // Do not propagate C++ exceptions through the C backend callback.
        try{
            if(ask)return self.selected(t);
            if(self.attention_selected(t)) {
                self.attention_seen=true;
                const std::string label=!self.attention_sinks.empty()?"strata_fattn_sinks_"+self.attention_sinks:
                    self.attention_mask.empty()?"strata_fattn0":"strata_fattn_"+self.attention_mask;
                self.capture(t,label.c_str());
                self.capture(t->src[0],(label+"_q").c_str());
                self.attention_inputs(t,label);
            } else if(self.matmul_selected(t)) {
                // Capture at the matmul callback, while its output and input
                // are live. Walking ancestors after RMSNorm can read reused
                // scratch rather than the original projection output.
                const bool routed=t->op==GGML_OP_MUL_MAT_ID;
                const std::string label=routed?"strata_expert_"+matmul_selector(t).substr(8):
                    "strata_matmul_"+std::string(t->src[0]->name);
                self.capture(t,label.c_str());self.capture(t->src[1],(label+"_input").c_str());
                if(routed)self.capture(t->src[2],(label+"_ids").c_str());
            } else self.capture(t);
        }
        catch(const std::exception &e){self.error=e.what();self.idle();return !ask;}
        return true;
    }
    void finish(float instrumentation_diff,float serial_instrumentation_diff) {
        idle();
        if(!error.empty())throw std::runtime_error(error);
        if((!attention_mask.empty() || !attention_sinks.empty() || names.count("@attention0")) && visible_reference.empty())
            throw std::runtime_error("activation probe did not find selected attention");
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
