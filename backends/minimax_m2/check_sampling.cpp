#include "sampling.hpp"
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
using namespace minimax_m2;
using json=nlohmann::ordered_json;
static void require(bool ok,const std::string &message) {if(!ok)throw std::runtime_error(message);}
static json read(const char *path) {std::ifstream f(path);require(bool(f),"cannot read JSON");json j;f>>j;return j;}
int main(int argc,char **argv) {
    try {
        const bool replay=argc==5 && std::string(argv[1])=="--replay";
        require(argc==2 || replay,"usage: sampling-check NEW_DIR | --replay GENERATION.json LOGITS.f32 NEW_DIR");
        const std::filesystem::path dir=argv[replay?4:1];
        require(!std::filesystem::exists(dir),"output directory exists");std::filesystem::create_directories(dir);
        json tests=json::array();
        auto test=[&](const std::string &name,bool pass) {tests.push_back({{"name",name},{"pass",pass}});};
        if(replay) {
            const auto report=read(argv[2]);const auto results=report.contains("results")?report.at("results"):json::array({report.at("result")});
            std::ifstream input(argv[3],std::ios::binary);require(bool(input),"cannot read logits");
            std::vector<float> logits(200064);uint64_t tokens=0,mismatch=0;json runs=json::array();
            for(const auto &result:results) {
                RequestSampler sampler(SamplingConfig::parse(result.at("sampling"),int(logits.size())));
                uint64_t wrong=0;for(const auto &id:result.at("token_ids")) {
                    input.read(reinterpret_cast<char *>(logits.data()),logits.size()*sizeof(float));require(bool(input),"short logits file");
                    wrong+=sampler.sample(logits.data(),int(logits.size()))!=id.get<llama_token>();++tokens;
                }
                mismatch+=wrong;runs.push_back({{"tokens",result.at("generated_tokens")},{"mismatches",wrong},{"pass",wrong==0}});
            }
            require(input.peek()==std::char_traits<char>::eof(),"extra logit rows");
            json result={{"scope","CPU replay of request-local sampler against raw generated logits; not an independent model oracle"},
                {"tokens",tokens},{"mismatches",mismatch},{"runs",runs},{"pass",mismatch==0}};
            std::ofstream(dir/"sampling-report.json")<<result.dump(2)<<'\n';std::cout<<result.dump()<<'\n';return mismatch?1:0;
        }
        const std::vector<json> invalid={nullptr,true,json::array(),{{"unknown",1}},
            {{"temperature",true}},{{"temperature",nullptr}},{{"temperature","1"}},{{"temperature",-.1}},
            {{"temperature",.0001}},{{"temperature",2.1}},{{"temperature",std::numeric_limits<double>::infinity()}},
            {{"temperature",std::numeric_limits<double>::quiet_NaN()}},
            {{"top_p",false}},{{"top_p",nullptr}},{{"top_p",".95"}},{{"top_p",0}},{{"top_p",-.1}},
            {{"top_p",1.01}},{{"top_p",1e-100}},{{"top_p",std::numeric_limits<double>::infinity()}},
            {{"top_k",true}},{{"top_k",-1}},{{"top_k",40.5}},{{"top_k",200065}},{{"top_k",UINT64_MAX}},
            {{"seed",false}},{{"seed",-1}},{{"seed",1.5}},{{"seed",uint64_t(UINT32_MAX)}},{{"seed",UINT64_MAX}}};
        for(size_t i=0;i<invalid.size();++i) {
            bool rejected=false;try {SamplingConfig::parse(invalid[i],200064);}catch(const std::exception &){rejected=true;}
            test("invalid_config_"+std::to_string(i),rejected);
        }
        auto defaults=SamplingConfig::parse(json::object(),200064);
        test("default_remains_greedy",defaults.temperature==0 && defaults.top_k==40 && defaults.top_p==.95f && defaults.seed==42);
        for(const auto &value:std::vector<json>{{{"temperature",0},{"top_k",0},{"top_p",1},{"seed",0}},
                {{"temperature",.01},{"top_k",200064},{"top_p",.001},{"seed",uint64_t(UINT32_MAX)-1}},{{"temperature",2}}}) {
            const auto c=SamplingConfig::parse(value,200064);test("valid_config_round_trip",SamplingConfig::parse(c.json(),200064).json()==c.json());
        }
        std::vector<float> logits={-5.f,2.f,2.f,-1.f};const auto original=logits;
        RequestSampler greedy(defaults);test("greedy_first_max_tie",greedy.sample(logits.data(),4)==1);
        for(float bad:{std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}) {
            auto broken=logits;broken[2]=bad;bool rejected=false;
            try {greedy.sample(broken.data(),4);}catch(const std::exception &){rejected=true;}
            test("nonfinite_rejected",rejected);
        }
        auto config=SamplingConfig::parse({{"temperature",.5},{"top_k",0},{"top_p",.95}},200064);
        RequestSampler golden(config);logits={2.f,1.f,0.f,-1.f};const auto saved=logits;
        std::vector<llama_token_data> distribution;golden.sample(logits.data(),4,&distribution);
        const double p0=1./(1.+std::exp(-2.));
        test("temperature_before_top_p_keeps_two",distribution.size()==2 && distribution[0].id==0 && distribution[1].id==1);
        test("independent_two_token_softmax",distribution.size()==2 && std::abs(distribution[0].p-p0)<1e-6 && std::abs(distribution[1].p-(1-p0))<1e-6);
        test("raw_logits_unchanged",std::memcmp(saved.data(),logits.data(),4*sizeof(float))==0);
        RequestSampler one(SamplingConfig::parse({{"temperature",2},{"top_k",1}},200064));bool only_best=true;
        for(int i=0;i<64;++i)only_best&=one.sample(logits.data(),4)==0;
        test("top_k_one_always_argmax",only_best);
        RequestSampler all(SamplingConfig::parse({{"temperature",1},{"top_k",0},{"top_p",1}},200064));
        all.sample(logits.data(),4,&distribution);double total=0;for(auto p:distribution)total+=p.p;
        test("disabled_filters_keep_vocabulary",distribution.size()==4 && std::abs(total-1)<1e-6);
        RequestSampler nucleus(SamplingConfig::parse({{"temperature",1},{"top_k",0},{"top_p",.1}},200064));
        nucleus.sample(logits.data(),4,&distribution);test("nucleus_min_keep_one",distribution.size()==1 && distribution[0].id==0 && distribution[0].p==1);
        auto equal=[](int seed) {return SamplingConfig::parse({{"temperature",1},{"top_k",0},{"top_p",1},{"seed",seed}},200064);};
        RequestSampler a(equal(42)),b(equal(42)),other(equal(7));logits.assign(64,0.f);
        std::vector<llama_token> sequence;bool same=true,different=false;
        for(int i=0;i<128;++i) {auto x=a.sample(logits.data(),64);sequence.push_back(x);same&=x==b.sample(logits.data(),64);different|=x!=other.sample(logits.data(),64);}
        test("fixed_seed_reproduces",same);test("different_seed_changes_draws",different);
        RequestSampler reset(equal(42));bool reset_ok=true;
        for(auto x:sequence)reset_ok&=reset.sample(logits.data(),64)==x;
        test("new_request_restarts_rng",reset_ok);
        logits={std::numeric_limits<float>::max(),0};bool overflow=false;
        try {golden.sample(logits.data(),2);}catch(const std::exception &){overflow=true;}
        test("temperature_overflow_rejected",overflow);
        const auto failures=std::count_if(tests.begin(),tests.end(),[](const auto &t){return !t.at("pass").template get<bool>();});
        json result={{"scope","CPU config/distribution/RNG tests; independent closed-form nucleus fixture"},
            {"cases",tests.size()},{"failures",failures},{"tests",tests},{"pass",failures==0}};
        std::ofstream(dir/"sampling-report.json")<<result.dump(2)<<'\n';std::cout<<json({{"cases",tests.size()},{"failures",failures},{"pass",failures==0}}).dump()<<'\n';
        return failures?1:0;
    }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 2;}
}
