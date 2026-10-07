// Cache metadata parity and optional CPU benchmark; no model arithmetic.
#include "expert_cache.hpp"
#include <chrono>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
using namespace mimo2;
struct LegacyHash {
    size_t operator()(const MatrixKey &key) const {
        return std::hash<uint64_t>{}(key.generation)^(size_t(key.tensor)<<16)^key.expert;
    }
};
static void require(bool ok,const char *message) {if(!ok)throw std::runtime_error(message);}
static std::vector<MatrixKey> keys() {
    std::vector<MatrixKey> result;
    for(uint32_t tensor=0;tensor<141;++tensor)
        for(uint32_t expert=0;expert<256;++expert)result.push_back({1,tensor,expert});
    return result;
}
template<class Hash> static void distribution(const char *name,const std::vector<MatrixKey> &all) {
    std::unordered_map<MatrixKey,unsigned,Hash> table;
    for(auto key:all)table.emplace(key,1);
    size_t used=0,largest=0;
    for(size_t i=0;i<table.bucket_count();++i) {
        const auto n=table.bucket_size(i);used+=n!=0;largest=std::max(largest,n);
    }
    std::cout<<"distribution "<<name<<" keys="<<table.size()<<" buckets="<<table.bucket_count()
             <<" occupied="<<used<<" largest="<<largest<<'\n';
}
template<class Hash> static uint64_t benchmark(const char *name,const std::vector<MatrixKey> &all,
                                             const std::vector<uint32_t> &trace) {
    StrataExpertFrequencyHistory<MatrixKey,Hash> history(65536,65536);
    for(auto key:all)history.record(key);
    uint64_t digest=0;
    const auto start=std::chrono::steady_clock::now();
    for(size_t i=0;i<trace.size();++i) {
        const auto &key=all[trace[i]];
        if(i%8==0)history.record(key);
        digest+=history.score(key);
    }
    const auto ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    std::cout<<"benchmark "<<name<<" probes="<<trace.size()<<" ms="<<ms<<" digest="<<digest<<'\n';
    return digest;
}
int main(int argc,char **argv) {
    try {
        const auto all=keys();size_t checks=0;
        for(const uint64_t period:{8,16384,65536,131072}) {
            // A small bound exercises history clearing; the full bound covers
            // actual MiMo identities and several model generations.
            for(const size_t limit:{97,65536}) {
                StrataExpertFrequencyHistory<MatrixKey,LegacyHash> old(period,limit);
                StrataExpertFrequencyHistory<MatrixKey,MatrixHash> fresh(period,limit);
                std::mt19937 rng(3711);
                for(size_t i=0;i<150000;++i) {
                    auto key=all[rng()%all.size()];
                    if(i%7==0)key.generation=2;
                    if(i%31==0)key.tensor=UINT32_MAX;
                    if(i%101==0)key.expert=UINT32_MAX;
                    old.record(key);fresh.record(key);
                    if(i%127==0) {old.seed(key,255);fresh.seed(key,255);}
                    require(old.score(key)==fresh.score(key),"selected score changed");
                    const auto probe=all[rng()%all.size()];
                    require(old.score(probe)==fresh.score(probe),"candidate score changed");
                    require(old.size()==fresh.size(),"history bound changed");++checks;
                }
                for(auto key:all) {require(old.score(key)==fresh.score(key),"final history changed");++checks;}
                old.erase_if([](const MatrixKey &key){return key.generation==1;});
                fresh.erase_if([](const MatrixKey &key){return key.generation==1;});
                require(old.size()==fresh.size(),"generation removal changed");++checks;
            }
        }
        distribution<LegacyHash>("legacy",all);distribution<MatrixHash>("mixed",all);
        std::cout<<"PASS "<<checks<<" history comparisons\n";
        if(argc==2 && std::string(argv[1])=="--benchmark") {
            std::mt19937 rng(825);std::vector<uint32_t> trace(4000000);
            for(auto &index:trace)index=rng()%all.size();
            const auto reference=benchmark<LegacyHash>("legacy",all,trace);
            require(benchmark<MatrixHash>("mixed",all,trace)==reference,"benchmark history changed");
            require(benchmark<MatrixHash>("mixed",all,trace)==reference,"repeat history changed");
            require(benchmark<LegacyHash>("legacy",all,trace)==reference,"control history changed");
        } else require(argc==1,"unexpected arguments");
        return 0;
    } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
