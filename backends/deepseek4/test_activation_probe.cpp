#include "activation_probe.hpp"
#include <iostream>
#include <limits>
static void require(bool value){if(!value)throw std::runtime_error("activation comparison failed");}
int main(){
    try {
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
        std::cout<<"PASS token axes, error location, nonfinite values, unmatched shapes and row bounds\n";
    }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
