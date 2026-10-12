#include "cpu_expert_fixture.hpp"
#include <algorithm>
#include <chrono>

int main(int argc,char **argv) {try {
    if(argc<2 || argc>3)throw std::runtime_error("usage: deepseek4_cpu_replay fixture.bin [threads]");
    const int threads=argc==3?std::atoi(argv[2]):8;
    strata_ds4::CpuLanePool pool(threads);
    strata_ds4::CpuExpertFixture fixture(argv[1]);
    for(int pass=0;pass<5;++pass) {
        strata_ds4::CpuExpertTiming timing;
        fixture.work.compute(pool,&timing);
        if(std::memcmp(fixture.expected.data(),fixture.output.data(),fixture.output.size()*4))
            throw std::runtime_error("CPU fixture replay differs from captured output");
        std::printf("{\"pass\":%d,\"threads\":%d,\"tokens\":%d,\"rows\":%zu,\"prepare_us\":%llu,\"gate_up_us\":%llu,\"hidden_quant_us\":%llu,\"down_us\":%llu,\"exact\":true}\n",
            pass,threads,fixture.work.tokens,fixture.work.experts.size(),(unsigned long long)timing.prepare_us,
            (unsigned long long)timing.gate_up_us,(unsigned long long)timing.hidden_quant_us,(unsigned long long)timing.down_us);
    }
    return 0;
}catch(const std::exception &e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
