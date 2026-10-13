#pragma once
// Bounded target-only benchmark, independent of sampling and draft acceptance.
// Each pass clears KV but preserves expert caches. No observer callbacks.
#include "speculative.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include "nlohmann/json.hpp"

inline void fixed_workload_phase(const std::string &phase) {
#ifdef _WIN32
    _putenv_s("STRATA_DS4_COST_PHASE",phase.c_str());
#else
    setenv("STRATA_DS4_COST_PHASE",phase.c_str(),1);
#endif
}
inline void run_fixed_workload(llama_context *ctx,const llama_vocab *vocab,
        const std::vector<llama_token> &tokens,int batch,const char *path,
        const std::function<void()> &budget,const std::atomic<bool> &stop,
        const std::function<void(bool)> &cpu_phase={}) {
    using Json=nlohmann::ordered_json;using Timer=std::chrono::steady_clock;
    int prefix=36,repeats=4;
    if(const char *v=std::getenv("STRATA_FIXED_PREFIX_TOKENS"))prefix=std::atoi(v);
    if(const char *v=std::getenv("STRATA_FIXED_REPEATS"))repeats=std::atoi(v);
    if(prefix<1 || prefix>int(tokens.size())-32 || tokens.size()>llama_n_ctx(ctx) || batch<1 || repeats<1 || repeats>4)
        throw std::runtime_error("fixed workload needs prefix + 32 tokens within context, repeats 1..4 and batch >=1");
    struct Cleanup {llama_context *c;~Cleanup(){llama_memory_clear(llama_get_memory(c),true);fixed_workload_phase("idle");}} cleanup{ctx};
    Json report={{"tokens",tokens},{"prefix_tokens",prefix},{"repeats",repeats},
        {"scope","target only; fixed input IDs; no sampling; KV cleared each pass; expert caches retained"},{"runs",Json::array()}};
    auto evaluate=[&](int begin,int end) {
        if(stop.load())throw std::runtime_error("fixed workload cancelled");
        Batch b(end-begin);b.positions(end-begin,begin,false);b.value.logits[end-begin-1]=true;
        std::copy(tokens.begin()+begin,tokens.begin()+end,b.value.token);budget();
        if(llama_decode(ctx,b.value))throw std::runtime_error("fixed workload decode failed");
        llama_synchronize(ctx);
    };
    auto digest=[&]() {
        const float *p=llama_get_logits_ith(ctx,-1);uint64_t hash=14695981039346656037ULL;
        const size_t count=llama_vocab_n_tokens(vocab);const auto *bytes=reinterpret_cast<const unsigned char *>(p);
        for(size_t i=0;i<count*sizeof(float);++i){hash^=bytes[i];hash*=1099511628211ULL;}
        for(size_t i=0;i<count;++i)if(!std::isfinite(p[i]))throw std::runtime_error("nonfinite fixed workload logits");
        return Json{{"fnv1a64",hash},{"argmax",std::max_element(p,p+count)-p}};
    };
    for(int repeat=0;repeat<repeats;++repeat)for(int chunk:{0,1,3}) {
        llama_memory_clear(llama_get_memory(ctx),true);
        std::string phase="r"+std::to_string(repeat)+(chunk?"_decode"+std::to_string(chunk):"_prefill");
        fixed_workload_phase(phase+(chunk?"_prefix":""));auto start=Timer::now();
        if(cpu_phase)cpu_phase(false);
        const int prefix_end=chunk?prefix:int(tokens.size());
        for(int begin=0;begin<prefix_end;begin+=batch)evaluate(begin,std::min(begin+batch,prefix_end));
        double prefix_ms=std::chrono::duration<double,std::milli>(Timer::now()-start).count();
        Json row={{"phase",phase},{"repeat",repeat},{"chunk",chunk},{"prefix_ms",prefix_ms},{"prefix_logits",digest()},{"steps",Json::array()}};
        if(chunk) {
            if(cpu_phase)cpu_phase(true);
            fixed_workload_phase(phase);double decode_ms=0;
            for(int begin=prefix;begin<prefix+32;begin+=chunk) {
                int end=std::min(begin+chunk,prefix+32);start=Timer::now();evaluate(begin,end);
                decode_ms+=std::chrono::duration<double,std::milli>(Timer::now()-start).count();
                row["steps"].push_back({{"end",end},{"logits",digest()}});
            }
            row["decode_ms"]=decode_ms;
        }
        report["runs"].push_back(row);
        std::cerr<<"STRATA_FIXED_WORKLOAD "<<phase<<" prefix_ms="<<prefix_ms<<"\n";
    }
    std::ofstream out(path);if(!out)throw std::runtime_error("cannot open fixed workload report");
    out<<report.dump(2)<<'\n';out.flush();if(!out)throw std::runtime_error("cannot write fixed workload report");
}
