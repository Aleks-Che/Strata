// Small real-CUDA cache fixture plus deterministic admission-history boundaries.
#include "../step35/expert_cache.hpp"
#include "sync_runtime.h"
#include "nlohmann/json.hpp"
#include <iostream>
using json=nlohmann::ordered_json;
int main() {
    json cases=json::array();
    auto check=[&](const std::string &name,bool pass) {
        cases.push_back({{"name",name},{"pass",pass}});
        if(!pass)throw std::runtime_error(name);
    };
    try {
        for(uint64_t period:{65536ull,131072ull,262144ull}) {
            StrataExpertFrequencyHistory<int> history(period);history.seed(1,8);
            for(uint64_t i=1;i<period;++i)history.record(2);
            check("before_boundary_"+std::to_string(period),history.score(1)==8);
            history.record(2);
            check("half_at_boundary_"+std::to_string(period),history.score(1)==4);
            for(uint64_t i=0;i<period;++i)history.record(2);
            check("quarter_at_second_boundary_"+std::to_string(period),history.score(1)==2);
            strata_mm27_cache_configure(0,false,64,0,false,period);
        }
        for(uint64_t period:{0ull,4294967296ull}) {
            bool refused=false;
            try {strata_mm27_cache_configure(0,false,64,0,false,period);}
            catch(const std::runtime_error &) {refused=true;}
            check("runtime_rejects_"+std::to_string(period),refused);
        }
        const size_t GiB=size_t(1)<<30;
        const step35::MemorySample sample{16*GiB,32*GiB,64*GiB,128*GiB};
        const step35::MatrixKey a{1,0,0},b{1,0,1},filler{1,0,2};
        for(uint64_t period:{8ull,65536ull}) {
            step35::ExpertCache cache(65536,[&]{return sample;},
                [](void **p,size_t n){return cudaMalloc(p,n);},[](void *p){return cudaFree(p);},period);
            cache.refresh();cache.set_reuse_allocations(true);cache.set_match_size(true);cache.set_fast_scan(true);
            for(int i=0;i<8;++i)cache.get(a,1024);
            auto *original=cache.admit(a,1024);check("initial_admission",original!=nullptr);
            for(int i=0;i<1000;++i)cache.get(filler,1024,false);
            cache.reset_counters();cache.get(b,1024);
            check("untrained_lookups_and_counter_reset_preserve_history",cache.admit(b,1024)==nullptr);
            for(int i=0;i<16;++i)cache.get(filler,1024);
            cache.get(b,1024);
            auto *replacement=cache.admit(b,1024);
            check("period_controls_topic_replacement_"+std::to_string(period),
                period==8 ? replacement==original && !cache.contains(a,1024) && cache.counters().reuses==1
                          : !replacement && cache.contains(a,1024));
            cache.trim(0);check("trim_releases_residency",cache.resident_bytes()==0);
        }
        strata_mm27_release();
        std::cout<<json({{"pass",true},{"cases",cases}}).dump(2)<<'\n';return 0;
    } catch(const std::exception &e) {
        std::cerr<<json({{"pass",false},{"error",e.what()},{"cases",cases}}).dump(2)<<'\n';return 1;
    }
}
