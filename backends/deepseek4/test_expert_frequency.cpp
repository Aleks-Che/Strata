#include "expert_frequency.hpp"
#include <cstdio>
#include <stdexcept>

static void require(bool value) {if(!value)throw std::runtime_error("frequency history contract failed");}
int main() {
    try {
        int hot,cold,other;
        StrataExpertFrequency history(16);
        for(int i=0;i<8;++i)history.record(&hot);
        history.record(&cold);
        require(history.score(&hot)==8 && history.score(&cold)==1);
        // A new workload overtakes the old hot set, including unseen accesses
        // which bypassed residency. Lazy decay applies to untouched keys too.
        for(int i=0;i<39;++i)history.record(&cold);
        require(history.score(&hot)==1 && history.score(&cold)>history.score(&hot));
        for(int i=0;i<96;++i)history.record(&other);
        require(history.score(&hot)==0);
        StrataExpertFrequency saturated(100000);
        for(int i=0;i<1000;++i)saturated.record(&hot);
        require(saturated.score(&hot)==255);
        // The bound is independent of model size and never retains stale
        // addresses indefinitely if a caller presents a changing weight set.
        StrataExpertFrequency bounded(16,2);
        bounded.record(&hot);bounded.record(&cold);bounded.record(&other);
        require(bounded.size()==1 && bounded.score(&hot)==0 && bounded.score(&other)==1);
        std::puts("Frequency admission history, decay, saturation and bound passed");
        return 0;
    } catch(const std::exception &e) {std::fprintf(stderr,"%s\n",e.what());return 1;}
}
