#include "activation_probe.hpp"
#include <iostream>
#include <limits>
static void require(bool value){if(!value)throw std::runtime_error("activation comparison failed");}
int main(){
    try {
        // Reconstruct compact rows independently from strided/padded backing
        // storage. Check every token axis and a permutation with outer gaps.
        for(int axis=0;axis<4;++axis)for(bool permuted:{false,true}) {
            ggml_tensor t{};t.type=GGML_TYPE_F32;
            for(int i=0;i<4;++i)t.ne[i]=i==axis?3:2;
            t.nb[0]=4;
            for(int i=1;i<4;++i)t.nb[i]=t.nb[i-1]*t.ne[i-1]+(i==2?16:0);
            int actual_axis=axis;
            if(permuted){std::swap(t.ne[0],t.ne[3]);std::swap(t.nb[0],t.nb[3]);actual_axis=axis==0?3:axis==3?0:axis;}
            std::vector<uint8_t> backing(ggml_nbytes(&t));
            for(size_t i=0;i<backing.size();++i)backing[i]=uint8_t(i*37+(i>>7));
            for(int row:{0,2}) {
                auto s=ActivationProbe::token_slice(&t,3,row);require(s.axis==actual_axis && s.shape[actual_axis]==1);
                std::vector<uint8_t> compact(s.bytes),expected;
                for(auto span:s.spans)std::memcpy(compact.data()+span.destination,backing.data()+span.source,span.size);
                for(int64_t n=0;n<ggml_nelements(&t);++n) {
                    int64_t q=n;size_t offset=0;bool take=false;
                    for(int i=0;i<4;++i){int64_t index=q%t.ne[i];q/=t.ne[i];offset+=index*t.nb[i];if(i==actual_axis)take=index==row;}
                    if(take)expected.insert(expected.end(),backing.begin()+offset,backing.begin()+offset+4);
                }
                require(compact==expected);
            }
        }
        ggml_tensor uncertain{};uncertain.type=GGML_TYPE_F32;
        uncertain.ne[0]=uncertain.ne[1]=3;uncertain.ne[2]=uncertain.ne[3]=1;
        require(ActivationProbe::token_slice(&uncertain,3,0).axis==-2);
        require(ActivationProbe::token_slice(&uncertain,5,0).axis==-2);
        bool invalid_row=false;try{ActivationProbe::token_slice(&uncertain,3,3);}catch(const std::exception&){invalid_row=true;}
        require(invalid_row);
        uncertain.ne[0]=uncertain.ne[1]=4096;uncertain.ne[2]=3;
        bool too_large=false;try{ActivationProbe::token_slice(&uncertain,3,0);}catch(const std::exception&){too_large=true;}
        require(too_large);
        // HCA moves the compressed keys when raw KV padding changes. Compare
        // visible bytes in order while retaining physical indices for diagnosis.
        ggml_tensor k{},v{},mask{},fa{};
        ggml_tensor weight{},mm{};std::strcpy(weight.name,"blk.0.attn_kv.weight");
        mm.op=GGML_OP_MUL_MAT;mm.src[0]=&weight;
        require(ActivationProbe::matmul_selector(&mm)=="@weight:blk.0.attn_kv.weight");
        mm.src[0]=nullptr;require(ActivationProbe::matmul_selector(&mm).empty());mm.src[0]=&weight;
        mm.op=GGML_OP_RMS_NORM;require(ActivationProbe::matmul_selector(&mm).empty());
        mm.op=GGML_OP_MUL_MAT_ID;std::strcpy(weight.name,"blk.0.ffn_gate_exps.weight");
        require(ActivationProbe::matmul_selector(&mm)=="@expert:blk.0.ffn_gate_exps.weight");
        std::strcpy(weight.name,"CUDA0#blk.0.ffn_gate_exps.weight#0");
        require(ActivationProbe::matmul_selector(&mm)=="@expert:blk.0.ffn_gate_exps.weight");
        for(const char *bad:{"CUDA#blk.0.ffn_gate_exps.weight#0","CUDAx#blk.0.ffn_gate_exps.weight#0","CUDA0#blk.0.ffn_gate_exps.weight#x"}) {
            std::strcpy(weight.name,bad);require(ActivationProbe::matmul_selector(&mm)==std::string("@expert:")+bad);
        }
        mm.src[0]=nullptr;require(ActivationProbe::matmul_selector(&mm).empty());
        // Six route IDs with padded token stride: selected token must not read
        // the neighboring token or include padding in the comparison.
        ggml_tensor ids{};ids.type=GGML_TYPE_I32;
        ids.ne[0]=6;ids.ne[1]=3;ids.ne[2]=ids.ne[3]=1;
        ids.nb[0]=4;ids.nb[1]=32;ids.nb[2]=ids.nb[3]=96;
        const auto id_slice=ActivationProbe::token_slice(&ids,3,1);
        require(id_slice.axis==1 && id_slice.bytes==24 && id_slice.spans.size()==1);
        require(id_slice.spans[0].source==32 && id_slice.spans[0].size==24);
        ActivationProbe::Value id_batch,id_one;id_batch.type=id_one.type="i32";
        id_batch.shape={6,3,1,1};id_one.shape={6,1,1,1};
        id_batch.data={0,1,2,3,4,5,250,254,10,11,12,255,20,21,22,23,24,25};
        id_one.data={250,254,10,11,12,255};
        auto id_compare=ActivationProbe::compare(id_batch,id_one,3,1);
        require(id_compare["max_abs"]==0 && id_compare["batch_ids"]==id_one.data);
        id_one.data[2]=9;id_compare=ActivationProbe::compare(id_batch,id_one,3,1);
        require(id_compare["unequal"]==1 && id_compare["serial_ids"][2]==9);
        k.ne[0]=v.ne[0]=2;k.ne[1]=v.ne[1]=4;
        k.nb[0]=v.nb[0]=2;k.nb[1]=v.nb[1]=8;
        mask.ne[0]=4;mask.ne[1]=1;mask.nb[0]=2;mask.nb[1]=8;
        std::strcpy(mask.name,"hca_kq_mask-3");fa.op=GGML_OP_FLASH_ATTN_EXT;fa.src[3]=&mask;
        require(ActivationProbe::attention_mask_matches(&fa,"hca_kq_mask-3"));
        require(!ActivationProbe::attention_mask_matches(&fa,"hca_kq_mask-4"));
        fa.src[3]=nullptr;require(!ActivationProbe::attention_mask_matches(&fa,"hca_kq_mask-3"));
        ggml_tensor sinks{};std::strcpy(sinks.name,"blk.1.attn_sinks.weight");fa.src[4]=&sinks;
        require(ActivationProbe::attention_sinks_matches(&fa,"blk.1.attn_sinks.weight"));
        require(!ActivationProbe::attention_sinks_matches(&fa,"blk.0.attn_sinks.weight"));
        fa.src[4]=nullptr;require(!ActivationProbe::attention_sinks_matches(&fa,"blk.1.attn_sinks.weight"));
        std::vector<uint8_t> kr(32),vr(32),mr(8);
        auto put=[](std::vector<uint8_t> &data,size_t offset,float value) {
            auto h=ggml_fp32_to_fp16(value);std::memcpy(data.data()+offset,&h,2);
        };
        for(int i=0;i<4;++i) {
            put(mr,i*2,i==1 || i==3?0:-std::numeric_limits<float>::infinity());
            for(int j=0;j<2;++j){put(kr,i*8+j*2,float(i+j));put(vr,i*8+j*2,float(i-j));}
        }
        const auto visible=ActivationProbe::visible_inputs(&k,&v,&mask,kr,vr,mr,0);
        require(visible.indices==std::vector<int64_t>({1,3}) && visible.data.size()==20);
        require(ActivationProbe::compare_visible(visible,visible,2,2)["k_unequal"]==0);
        auto altered=visible;put(altered.data,2,3.0f);
        auto components=ActivationProbe::compare_visible(visible,altered,2,2);
        require(components["k_unequal"]==1 && components["v_unequal"]==0 && components["k_max_abs"]==2.0);
        require(components["masks_equal"]==true && components["changed_keys"].size()==1);
        auto compact_k=k,compact_v=v,compact_mask=mask;
        compact_k.ne[1]=compact_v.ne[1]=compact_mask.ne[0]=2;
        std::vector<uint8_t> kc(16),vc(16),mc(4);
        for(int i=0;i<2;++i) {
            std::memcpy(kc.data()+i*8,kr.data()+(1+2*i)*8,4);
            std::memcpy(vc.data()+i*8,vr.data()+(1+2*i)*8,4);
        }
        require(visible.data==ActivationProbe::visible_inputs(&compact_k,&compact_v,&compact_mask,kc,vc,mc,0).data);
        put(mc,2,-0.25f);require(visible.data!=ActivationProbe::visible_inputs(&compact_k,&compact_v,&compact_mask,kc,vc,mc,0).data);
        for(float bad:{std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}) {
            put(mc,2,bad);bool threw=false;
            try{ActivationProbe::visible_inputs(&compact_k,&compact_v,&compact_mask,kc,vc,mc,0);}catch(const std::runtime_error&){threw=true;}
            require(threw);
        }
        bool truncated=false;
        try{ActivationProbe::visible_inputs(&k,&v,&mask,{},vr,mr,0);}catch(const std::runtime_error&){truncated=true;}
        require(truncated);
        ActivationProbe::Value large,small;
        large.name=small.name="large-prefix";large.shape={2,5,1,1};small.shape={2,3,1,1};
        large.data={0,1,2,3,4,5,6,7,8,9};small.data={0,1,2,3,4,5};
        auto pair=ActivationProbe::compare_batches(large,small,5,2,3,2);
        require(pair["max_abs"]==0 && pair["compared_row"]==2);
        small.data[4]=4.01f;pair=ActivationProbe::compare_batches(large,small,5,2,3,2);
        require(pair["unequal"]==1 && pair["rounded_f16_unequal"]==1);
        require(pair["rounded_f16_examples"][0]["index"]==0);
        small.data.pop_back();bool short_batch=false;
        try{ActivationProbe::compare_batches(large,small,5,2,3,2);}catch(const std::exception&){short_batch=true;}
        require(short_batch);
        large.shape={1,1,1,1};small.shape=large.shape;
        const float midpoint=1.0f+1.0f/2048;
        large.data={midpoint};small.data={std::nextafter(midpoint,2.0f)};
        pair=ActivationProbe::compare_batches(large,small,1,0,1,0);
        require(pair["rounded_f16_unequal"]==1 && pair["max_abs"].get<double>()<2e-7);
        require(pair["rounded_f16_compared"]==1);
        large.data={70000};small.data={71000};
        require(ActivationProbe::compare_batches(large,small,1,0,1,0)["rounded_f16_compared"]==0);
        for(int axis=0;axis<4;++axis) {
            ActivationProbe::Value batch,one;
            batch.shape={2,4,2,2};one.shape=batch.shape;
            batch.shape[axis]=3;one.shape[axis]=1;
            size_t size=1,inner=1;
            for(int d=0;d<4;++d){size*=one.shape[d];if(d<axis)inner*=one.shape[d];}
            batch.data.resize(size*3);one.data.resize(size);
            for(size_t i=0;i<batch.data.size();++i)batch.data[i]=float(i)*0.5f;
            for(int row=0;row<3;++row) {
                for(size_t i=0;i<size;++i)one.data[i]=batch.data[(i/inner*3+row)*inner+i%inner];
                auto exact=ActivationProbe::compare(batch,one,3,row);
                require(exact["axis"]==axis && exact["unequal"]==0 && exact["max_abs"]==0);
                if(row<2) {
                    auto two=batch;two.shape[axis]=2;two.data.resize(size*2);
                    for(size_t j=0;j<two.data.size();++j)two.data[j]=batch.data[(j/(inner*2)*3+j/inner%2)*inner+j%inner];
                    require(ActivationProbe::compare_batches(batch,two,3,row,2,row)["unequal"]==0);
                }
                one.data.back()+=0.25f;
                auto diff=ActivationProbe::compare(batch,one,3,row);
                require(diff["unequal"]==1 && diff["max_abs"]==0.25 && diff["max_index"]==size-1);
                one.data.back()=std::numeric_limits<float>::quiet_NaN();
                require(ActivationProbe::compare(batch,one,3,row)["nonfinite"]==1);
            }
            auto wrong=one;wrong.shape[(axis+1)%4]+=1;
            require(ActivationProbe::compare(batch,wrong,3,0).contains("skipped"));
            require(ActivationProbe::compare(one,one,3,0)["skipped"]=="no token axis");
            auto same=one;for(auto &x:same.data)x=1.0f;
            require(ActivationProbe::compare(same,same,1,0)["max_abs"]==0);
            auto changed=same;changed.data.back()=2.0f;
            require(ActivationProbe::compare(same,changed,1,0)["unequal"]==1);
            auto empty=same;empty.data.clear();empty.shape[0]=0;
            require(ActivationProbe::compare(empty,empty,1,0)["skipped"]=="empty tensor");
            bool threw=false;
            try{ActivationProbe::compare(batch,one,3,3);}catch(const std::runtime_error&){threw=true;}
            require(threw);
        }
        std::cout<<"PASS visible KV/masks, attention selection, token axes, nonfinite values and bounds\n";
    }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
