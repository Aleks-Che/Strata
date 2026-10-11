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
    require(prefill_probe_splits("",4095,4096)==std::vector<int>({4095}));
    require(prefill_probe_splits("2047,2048,2049,3071",4095,2048)==std::vector<int>({2047,2048,2049,3071,4095}));
    for(int n:{0,4096}){bool failed=false;try{prefill_probe_splits("",n,8192);}catch(const std::exception&){failed=true;}require(failed);}
    {bool failed=false;try{prefill_probe_splits("",4095,2048);}catch(const std::exception&){failed=true;}require(failed);}
    const std::vector<int> ends={2047,2048,2049,2303,2561,2815,3071};
    prefill_probe_window(2304,128,3071,ends);
    prefill_probe_window(2433,128,3071,ends);
    prefill_probe_window(-1,1,3071,ends);
    for(auto window:{std::make_pair(2304,129),std::make_pair(2304,0),std::make_pair(2434,128),
                     std::make_pair(3071,1),std::make_pair(-1,128),std::make_pair(-2,1)}) {
        bool failed=false;try{prefill_probe_window(window.first,window.second,3071,ends);}catch(const std::exception&){failed=true;}
        require(failed);
    }
    auto d=prefill_logit_diff({1,2,3},{1,4,3});require(d["max_abs"]==2 && d["unequal"]==1 && d["argmax_a"]==2 && d["argmax_b"]==1);
    require(prefill_logit_diff({1,2,3},{1,2,3})["max_abs"]==0);
    for(const std::vector<float> &v:std::vector<std::vector<float>>{{},{1},{1,2,std::numeric_limits<float>::quiet_NaN()}}) {
        bool failed=false;try{prefill_logit_diff({1,2,3},v);}catch(const std::exception&){failed=true;}require(failed);
    }
    std::cout<<"PASS fixed prefix partitions, limits, logits comparison and nonfinite rejection\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
