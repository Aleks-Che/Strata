#include "activation_probe.hpp"
#include <iostream>
#include <limits>
static void require(bool value){if(!value)throw std::runtime_error("activation comparison failed");}
int main(){
    try {
        // HCA moves the compressed keys when raw KV padding changes. Compare
        // visible bytes in order while retaining physical indices for diagnosis.
        ggml_tensor k{},v{},mask{},fa{};
        k.ne[0]=v.ne[0]=2;k.ne[1]=v.ne[1]=4;
        k.nb[0]=v.nb[0]=2;k.nb[1]=v.nb[1]=8;
        mask.ne[0]=4;mask.ne[1]=1;mask.nb[0]=2;mask.nb[1]=8;
        std::strcpy(mask.name,"hca_kq_mask-3");fa.op=GGML_OP_FLASH_ATTN_EXT;fa.src[3]=&mask;
        require(ActivationProbe::attention_mask_matches(&fa,"hca_kq_mask-3"));
        require(!ActivationProbe::attention_mask_matches(&fa,"hca_kq_mask-4"));
        fa.src[3]=nullptr;require(!ActivationProbe::attention_mask_matches(&fa,"hca_kq_mask-3"));
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
                one.data.back()+=0.25f;
                auto diff=ActivationProbe::compare(batch,one,3,row);
                require(diff["unequal"]==1 && diff["max_abs"]==0.25 && diff["max_index"]==size-1);
                one.data.back()=std::numeric_limits<float>::quiet_NaN();
                require(ActivationProbe::compare(batch,one,3,row)["nonfinite"]==1);
            }
            auto wrong=one;wrong.shape[(axis+1)%4]+=1;
            require(ActivationProbe::compare(batch,wrong,3,0).contains("skipped"));
            require(ActivationProbe::compare(one,one,3,0)["skipped"]=="no token axis");
            bool threw=false;
            try{ActivationProbe::compare(batch,one,3,3);}catch(const std::runtime_error&){threw=true;}
            require(threw);
        }
        std::cout<<"PASS visible KV/masks, attention selection, token axes, nonfinite values and bounds\n";
    }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
