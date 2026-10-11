#include "prefill_probe.hpp"
#include <iostream>
#include <limits>
static void require(bool ok){if(!ok)throw std::runtime_error("prefill probe contract failed");}
int main(){try{
    require(prefill_probe_splits("",9,16)==std::vector<int>({9}));
    require(prefill_probe_splits("3,4,7",9,4)==std::vector<int>({3,4,7,9}));
    require(prefill_probe_splits("3,6,9",9,4)==std::vector<int>({3,6,9}));
    for(const auto &s:{"0","-1","3,3","4,2","10","1x","1,,2","5"}) {
        bool failed=false;try{prefill_probe_splits(s,9,4);}catch(const std::exception&){failed=true;}require(failed);
    }
    for(int n:{0,1024}){bool failed=false;try{prefill_probe_splits("",n,2048);}catch(const std::exception&){failed=true;}require(failed);}
    auto d=prefill_logit_diff({1,2,3},{1,4,3});require(d["max_abs"]==2 && d["unequal"]==1 && d["argmax_a"]==2 && d["argmax_b"]==1);
    require(prefill_logit_diff({1,2,3},{1,2,3})["max_abs"]==0);
    for(const std::vector<float> &v:std::vector<std::vector<float>>{{},{1},{1,2,std::numeric_limits<float>::quiet_NaN()}}) {
        bool failed=false;try{prefill_logit_diff({1,2,3},v);}catch(const std::exception&){failed=true;}require(failed);
    }
    std::cout<<"PASS fixed prefix partitions, limits, logits comparison and nonfinite rejection\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
