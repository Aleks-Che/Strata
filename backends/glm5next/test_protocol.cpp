#include "protocol.hpp"
#include <iostream>
int main() {
    try {
        auto r=strata_glm::request("GEN 8 temperature=0.7 top_p=0.9 top_k=8 min_p=0.02 seed=42 penalty_last_n=16 penalty_repeat=1.1 penalty_freq=0.5 penalty_present=-0.5 1,2,3",32,64);
        if (r.count!=8 || r.tokens!=std::vector<int32_t>({1,2,3}) || r.value("seed",0)!=42) return 1;
        strata_glm::request("GEN 1 session="+std::string(64,'a')+" 1",32,64);
        for (const std::string & text : {"", "GEN", "GENI 1 2", "GEN 0 1", "GEN -1 1", "GEN 32 1",
             "GEN 1", "GEN 1 64", "GEN 1 -1", "GEN 1 1,", "GEN 1 1,,2", "GEN 1 1 2", "GEN 1 1x",
             "GEN 1 temperature=nan 1", "GEN 1 temperature=inf 1", "GEN 1 temperature=-1 1",
             "GEN 1 top_p=0 1", "GEN 1 min_p=2 1", "GEN 1 seed=1.5 1", "GEN 1 seed=4294967296 1",
             "GEN 1 top_k=65 1", "GEN 1 penalty_repeat=0 1", "GEN 1 penalty_last_n=-1 1",
             "GEN 1 penalty_freq=1e100 1", "GEN 1 temperature=1 temperature=2 1", "GEN 1 cvec=1 1",
             "GEN 1 session=bad 1", "GEN 1 1 temperature=0"}) {
            bool rejected=false;
            try { strata_glm::request(text,32,64); } catch(const std::exception &) { rejected=true; }
            if (!rejected) { std::cerr<<"accepted invalid request: "<<text<<"\n"; return 1; }
        }
        std::cout<<"GLM protocol: 2 valid and 28 invalid requests passed\n"; return 0;
    } catch(const std::exception & e) { std::cerr<<e.what()<<"\n"; return 1; }
}
