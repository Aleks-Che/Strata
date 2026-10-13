#include "cpu_mmvq_fast.hpp"
#include "cpu_expert_fixture.hpp"
#include "swiglu_compat.hpp"
#include "ggml-quants.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <string>
using namespace strata_ds4;
using json=nlohmann::ordered_json;
static void require(bool b,const char *s){if(!b)throw std::runtime_error(s);}
static void env(const char *name,const char *value) {
#ifdef _WIN32
    _putenv_s(name,value);
#else
    setenv(name,value,1);
#endif
}
static uint32_t random_word(uint32_t &s){s^=s<<13;s^=s>>17;s^=s<<5;return s;}
static std::vector<uint8_t> weights(ggml_type type,int k,int rows,uint32_t seed) {
    std::vector<uint8_t> w(ggml_row_size(type,k)*rows);
    for(auto &v:w)v=uint8_t(random_word(seed));
    const size_t block=ggml_type_size(type);
    for(size_t i=0;i<w.size();i+=block) {
        if(type==GGML_TYPE_MXFP4)w[i]=uint8_t(115+random_word(seed)%12);
        else {auto d=ggml_fp32_to_fp16(std::ldexp(float(1+random_word(seed)%16),-10));std::memcpy(w.data()+i,&d,2);}
    }
    return w;
}
static void matrices() {
    int comparisons=0;
    for(auto type:{GGML_TYPE_IQ3_XXS,GGML_TYPE_MXFP4,GGML_TYPE_Q8_0})for(int k:{256,512,2048,4096,8192}) {
        constexpr int rows=17;auto w=weights(type,k,rows,39+k);
        std::vector<CpuQ8_1> q[4];std::vector<float> expected[4],actual[4];
        const CpuQ8_1 *inputs[4];float *outputs[4];uint32_t seed=139;
        for(int t=0;t<4;++t) {
            q[t].resize(k/32);actual[t].resize(rows);
            for(auto &b:q[t]) {
                b.d=ggml_fp32_to_fp16(std::ldexp(float(1+random_word(seed)%8),-9));b.s=0;
                for(auto &v:b.qs)v=int8_t(random_word(seed)); // includes -128, adversarial full range
            }
            inputs[t]=q[t].data();outputs[t]=actual[t].data();
            expected[t]=cpu_mmvq_reference(type,k,rows,w.data(),inputs[t]);
        }
        for(int n=1;n<=4;++n) {
            cpu_mmvq_fast(type,k,rows,w.data(),inputs,outputs,n);
            for(int t=0;t<n;++t){require(!std::memcmp(actual[t].data(),expected[t].data(),rows*4),"SIMD matrix differs from reference");++comparisons;}
        }
        env("STRATA_DS4_CPU_MMVQ_NO_AVX2","1");cpu_mmvq_fast(type,k,rows,w.data(),inputs,outputs,4);
        for(int t=0;t<4;++t)require(!std::memcmp(actual[t].data(),expected[t].data(),rows*4),"portable fallback mismatch");
        env("STRATA_DS4_CPU_MMVQ_NO_AVX2","0");
    }
    std::cout<<json{{"matrix_comparisons",comparisons},{"fallback_comparisons",60},{"exact",true}}.dump()<<std::endl;
}
static void expert_check(CpuExpertWork w) {
    std::vector<float> expected(w.experts.size()*w.columns),actual(expected.size());
    CpuLanePool oracle(1);w.output=expected.data();cpu_mmvq_reference_compute(w,oracle,nullptr,{});
    for(int threads:{1,8,12,16})for(auto tuning:{CpuPoolOptions{0,2},CpuPoolOptions{0,1},CpuPoolOptions{0,4},CpuPoolOptions{200,2},CpuPoolOptions{200,4}}) {
        CpuLanePool pool(threads,tuning);w.output=actual.data();int phases=0;
        w.compute(pool,nullptr,[&](CpuWeightPhase phase) {
            require(int(phase)==phases,"wrong weight retirement order");++phases;
        });
        require(phases==2 && !std::memcmp(actual.data(),expected.data(),actual.size()*4),"fast expert differs from compatible reference");
    }
}
static void experts() {
    CpuExpertWork w{};w.columns=512;w.hidden=256;w.topk=3;
    uint32_t seed=351;std::vector<float> x(4*w.columns);for(auto &v:x)v=(int(random_word(seed)%65536)-32768)/65536.f;w.input=x.data();
    for(int mixed=0;mixed<3;++mixed) {
        w.gate_type=mixed==2?GGML_TYPE_Q8_0:GGML_TYPE_IQ3_XXS;
        w.up_type=mixed==1?GGML_TYPE_MXFP4:w.gate_type;w.down_type=mixed==0?GGML_TYPE_IQ3_XXS:GGML_TYPE_MXFP4;
        auto g=weights(w.gate_type,w.columns,w.hidden,33),u=weights(w.up_type,w.columns,w.hidden,35),u2=weights(w.up_type,w.columns,w.hidden,89);
        auto d=weights(w.down_type,w.hidden,w.columns,71),d2=weights(w.down_type,w.hidden,w.columns,97);
        for(int n=1;n<=4;++n)for(float clamp:{0.f,0.1f}) {
            w.tokens=n;w.clamp=clamp;w.experts.clear();
            // Shared gate with differing up/down must NOT merge groups. More
            // than four rows in one group exercises safe microbatch splitting.
            for(int t=n-1;t>=0;--t)for(int j=0;j<3;++j)
                w.experts.push_back({g.data(),j==1?u2.data():u.data(),j==1?d2.data():d.data(),t,t*3+j});
            expert_check(w);
        }
    }
    std::cout<<json{{"expert_cases",24},{"threads",{1,8,12,16}},{"exact",true}}.dump()<<std::endl;
}
static void benchmark(bool routes=false) {
    // Eight separate RAM banks exceed this PC's 64 MiB aggregate L3 even for
    // IQ3. No file I/O, loading or pool creation inside the timed interval.
    using Clock=std::chrono::steady_clock;
    for(auto type:{GGML_TYPE_IQ3_XXS,GGML_TYPE_MXFP4,GGML_TYPE_Q8_0}) {
        constexpr int k=4096,h=2048,banks=8;
        std::array<std::array<std::vector<uint8_t>,3>,banks> storage;
        for(auto &bank:storage){bank[0]=weights(type,k,h,33);bank[1]=weights(type,k,h,35);bank[2]=weights(type,h,k,71);}
        const size_t bytes=banks*(storage[0][0].size()+storage[0][1].size()+storage[0][2].size());
        std::vector<float> x(4*k);for(size_t i=0;i<x.size();++i)x[i]=0.3f*std::sin(float(i)*0.017f);
        for(int pattern:routes?std::vector<int>{1,2}:std::vector<int>{0})
        for(int threads:{8,12,16})for(int n:routes?std::vector<int>{3}:std::vector<int>{1,2,3,4}) {
            CpuLanePool pool(threads);std::vector<float> expected(n*k),actual(n*k);
            CpuExpertWork w{};w.gate_type=w.up_type=w.down_type=type;
            w.columns=k;w.hidden=h;w.tokens=n;w.topk=1;w.clamp=0.1f;w.input=x.data();w.output=actual.data();
            auto select=[&](int b) {w.experts.clear();for(int t=0;t<n;++t) {
                const int bank=(b+(pattern==1?t%2:pattern==2?t:0))%banks;
                w.experts.push_back({storage[bank][0].data(),storage[bank][1].data(),storage[bank][2].data(),t,t});
            }};
            select(0);w.output=expected.data();cpu_mmvq_reference_compute(w,pool,nullptr,{});w.output=actual.data();
            std::vector<double> samples[3];CpuExpertTiming totals[3]{};
            // Alternating order across rounds reduces systematic clock/heat bias.
            for(int pass=0;pass<4;++pass)for(int m=0;m<3;++m) {
                int mode=pass%2?2-m:m;env("STRATA_DS4_CPU_MMVQ_SINGLE",mode==1?"1":"0");
                double duration=0;
                for(int b=0;b<banks;++b) {
                    select(b);CpuExpertTiming timing;const auto start=Clock::now();
                    if(mode==0)cpu_mmvq_reference_compute(w,pool,&timing,{});else w.compute(pool,&timing);
                    duration+=std::chrono::duration<double,std::micro>(Clock::now()-start).count();
                    require(!std::memcmp(expected.data(),actual.data(),actual.size()*4),"benchmark changed arithmetic");
                    totals[mode].prepare_us+=timing.prepare_us;totals[mode].gate_up_us+=timing.gate_up_us;
                    totals[mode].hidden_quant_us+=timing.hidden_quant_us;totals[mode].down_us+=timing.down_us;
                }
                samples[mode].push_back(duration/banks);
            }
            json result={{"format",ggml_type_name(type)},{"threads",threads},{"tokens",n},
                {"pattern",pattern==0?"shared":pattern==1?"partial":"unique"},{"weight_mib",bytes/double(1<<20)},{"exact",true}};
            for(int mode=0;mode<3;++mode) {
                auto s=samples[mode];std::sort(s.begin(),s.end());const auto &t=totals[mode];
                result[mode==0?"reference":mode==1?"simd_single":"simd_multi"]={{"median_us",(s[1]+s[2])/2},
                    {"rounds_us",samples[mode]},{"gate_up_us",t.gate_up_us/32.},{"hidden_quant_us",t.hidden_quant_us/32.},{"down_us",t.down_us/32.}};
            }
            std::cout<<result.dump()<<std::endl;
        }
    }
    env("STRATA_DS4_CPU_MMVQ_SINGLE","0");
}
int main(int argc,char **argv){try {
    CpuLanePool init(1);env("STRATA_DS4_CPU_MMVQ_FAST","1");env("STRATA_DS4_SWIGLU_COMPAT","0");
    require(swiglu_compat_enabled(),"fast contract overridden");
    std::cout<<json{{"avx2",cpu_mmvq_avx2_available()}}.dump()<<std::endl;
    if(argc==2 && std::string(argv[1])=="--bench"){benchmark();return 0;}
    if(argc==2 && std::string(argv[1])=="--bench-routes"){benchmark(true);return 0;}
    matrices();experts();
    for(int i=1;i<argc;++i){CpuExpertFixture f(argv[i]);expert_check(f.work);std::cout<<json{{"fixture",argv[i]},{"rows",f.work.experts.size()},{"exact",true}}.dump()<<std::endl;}
    return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<std::endl;return 1;}}
