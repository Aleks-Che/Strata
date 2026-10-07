// Deterministic replay of cache policy plus optional CPU metadata benchmark.
// Golden digests are captured from the ordered-map implementation before the
// index change; allocations/addresses and wall time never enter the digest.
#include "expert_cache.hpp"
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
using namespace mimo2;
static void require(bool value,const char *message) {if(!value)throw std::runtime_error(message);}
static MatrixKey key(uint32_t n) {return {1+n%3,n/256,n%256};}
static size_t size(uint32_t n) {return 256*(1+n%3);}
struct Digest {
    uint64_t value=14695981039346656037ULL;
    void add(uint64_t n) {for(unsigned i=0;i<8;++i) {value^=(n>>(8*i))&255;value*=1099511628211ULL;}}
};
static void state(ExpertCache &cache,Digest &digest) {
    const auto s=cache.stats();
    for(auto value:{s.hits,s.misses,s.evictions,s.allocations,s.reuses,s.bypasses,s.oom,
                   s.frequency_updates,s.frequency_rejected,s.frequency_candidates})digest.add(value);
    for(auto value:{cache.bytes(),cache.slot_bytes(),cache.payload_bytes(),cache.budget(),cache.pending(),cache.history_keys()})digest.add(value);
}
static uint64_t replay(uint64_t decay,unsigned routes,bool audit) {
    size_t live=0;Digest digest;
    {
        ExpertCache cache(4096*256,[&](size_t n) {auto p=std::malloc(n);require(p,"allocation");++live;return p;},
            [&](void *p){require(live>0,"allocation accounting");--live;std::free(p);},256,{}, {},decay);
        cache.constrain(size_t(8)<<30,size_t(32)<<30);
        // Pins survive table growth and overlapping plans, including duplicates.
        const MatrixKey anchor{UINT64_MAX,UINT32_MAX,UINT32_MAX};
        cache.store(anchor,100,[](void *p){std::memset(p,231,100);});
        auto outer=cache.protect({anchor,anchor});auto inner=cache.protect({anchor});
        for(uint32_t n=0;n<4096;++n)cache.store(key(n),size(n),[&](void *p){std::memset(p,n,size(n));});
        outer.reset();cache.trim(0);
        require(cache.contains(anchor,100) && cache.payload_bytes()==100,"pin invalidated by table growth or nested release");
        require(static_cast<unsigned char *>(cache.get(anchor,100))[99]==231,"pinned payload overwritten");
        state(cache,digest);inner.reset();cache.trim(0);require(live==0,"unpin did not release entries");
        std::mt19937 random(3712);
        for(unsigned route=0;route<routes;++route) {
            std::vector<uint32_t> selected;std::vector<MatrixKey> plan;
            for(unsigned i=0;i<24;++i) {
                // Sequence PRNG draws explicitly so the frozen trace does not
                // depend on the compiler's operand evaluation order.
                // The recorded MSVC reference evaluated the divisor first.
                const auto selector=random();const auto sample=random();
                const auto n=sample%(selector%5?1536:4096);
                selected.push_back(n);plan.push_back(key(n));
            }
            auto pins=cache.protect(plan);auto nested=cache.protect({plan[0],plan[0]});
            if(route%37==0)cache.trim(256*512);
            for(auto n:selected) {
                const auto k=key(n);const auto bytes=size(n);cache.record(k);
                const bool deferred=n%2==0;
                if(auto p=cache.get(k,bytes,!deferred)) {
                    require(static_cast<unsigned char *>(p)[0]==(n&255) && static_cast<unsigned char *>(p)[bytes-1]==(n&255),"cached bytes differ");
                    digest.add(n+1);if(deferred)cache.touch(k);
                } else {cache.store(k,bytes,[&](void *p){std::memset(p,n,bytes);});digest.add(0);}
                require(cache.bytes()<=cache.budget() && cache.payload_bytes()==cache.slot_bytes(),"cache budget violated");
            }
            pins.reset();if(route%53==0)cache.trim(256*256);nested.reset();
            if(route%113==0) {cache.constrain(0,size_t(32)<<30);cache.constrain(size_t(8)<<30,size_t(32)<<30);}
            if(audit) {
                state(cache,digest);
                for(uint32_t n=0;n<4096;++n)digest.add(cache.contains(key(n),size(n)));
            }
        }
        state(cache,digest);cache.trim(0);require(!live && !cache.bytes(),"final trim leaked");
    }
    require(!live,"destructor leaked");return digest.value;
}
int main(int argc,char **argv) {
    try {
        for(uint64_t decay:{uint64_t(0),uint64_t(65536)}) {
            const auto digest=replay(decay,400,true);
            std::cout<<"policy decay="<<decay<<" routes=400 operations=9600 digest="<<digest<<'\n';
            require(digest==(decay?1466279367919367327ULL:12436742711461247765ULL),"cache policy differs from ordered-map reference");
        }
        if(argc==2 && std::string(argv[1])=="--benchmark") {
            const auto begin=std::chrono::steady_clock::now();
            const auto digest=replay(65536,6000,false);
            const auto ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
            std::cout<<"benchmark routes=6000 operations=144000 ms="<<ms<<" digest="<<digest<<'\n';
            require(digest==16589823275981314459ULL,"benchmark cache policy differs from reference");
        } else require(argc==1,"unexpected arguments");
        return 0;
    }catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
