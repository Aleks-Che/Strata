// CPU ownership/admission tests; no CUDA or model required.
#include "expert_cache.hpp"
#include "expert_slab.hpp"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <random>
using namespace mimo2;
static void require(bool ok,const char *message) {if(!ok)throw std::runtime_error(message);}
int main() {
    try {
        size_t checks=0,physical=0;bool oom=false;std::map<void *,size_t> live;
        auto check=[&](bool ok,const char *name){require(ok,name);++checks;};
        auto alloc=[&](size_t n)->void * {if(oom)return nullptr;auto p=std::malloc(n);require(p,"allocation");live[p]=n;physical+=n;return p;};
        auto release=[&](void *p){auto it=live.find(p);require(it!=live.end(),"foreign free");physical-=it->second;live.erase(it);std::free(p);};
        const MatrixKey a{1,1,1},b{1,1,2},c{1,1,3},d{2,1,1},e{3,1,1};
        auto fill=[](void *p){std::memset(p,42,100);};
        auto no_fill=[](void *){throw std::runtime_error("rejected entry filled");};
        {
            ExpertCache cache(512,alloc,release,256,{}, {},1024);
            cache.constrain(size_t(8)<<30,size_t(32)<<30);
            for(int i=0;i<10;++i)cache.record(a);cache.store(a,100,fill);
            for(int i=0;i<2;++i)cache.record(b);cache.store(b,100,fill);
            cache.record(c);cache.store(c,100,no_fill);
            check(cache.contains(a,100) && cache.contains(b,100) && !cache.contains(c,100),"colder newcomer rejected before eviction");
            check(cache.stats().frequency_rejected==1 && cache.stats().evictions==0,"rejection accounting");
            cache.record(c);cache.record(c);cache.store(c,100,fill);
            check(cache.contains(a,100) && cache.contains(c,100) && !cache.contains(b,100),"miss history admits recurring newcomer and protects older hot entry");
            check(cache.stats().reuses==1 && physical==512,"same-size reuse retains physical limit");
            auto pin=cache.protect({a});
            for(int i=0;i<10;++i)cache.record(c);
            for(int i=0;i<5;++i)cache.record(d);
            cache.store(d,100,no_fill);
            check(cache.contains(a,100) && cache.contains(c,100),"pins excluded from frequency victim selection");
            pin.reset();for(int i=0;i<15;++i)cache.record(d);cache.store(d,100,fill);
            check(!cache.contains(a,100) && cache.contains(d,100) && cache.contains(c,100),"hottest newcomer replaces lower-frequency eligible entry");
            cache.record(e);cache.store(e,100,no_fill);
            check(!cache.contains(e,100),"model generations have independent history");
            const auto keys=cache.history_keys();cache.reset_stats();cache.store(e,100,no_fill);
            check(cache.history_keys()==keys && cache.stats().frequency_rejected==1,"request reset retains admission history");
            cache.get(c,100,false);cache.touch(c);cache.contains(c,100);
            check(cache.stats().frequency_updates==0,"lookup guard and deferred touch never train implicitly");
            cache.constrain(0,size_t(32)<<30);
            check(!cache.bytes() && !physical,"pressure can evict hot entries irrespective of frequency");
            cache.store(e,100,no_fill);check(cache.stats().bypasses==1,"zero budget still bypasses");
            cache.constrain(size_t(8)<<30,size_t(32)<<30);oom=true;cache.store(e,100,no_fill);oom=false;
            check(cache.stats().oom==1 && !physical,"frequency mode OOM leaves no allocation");
            bool failed=false;try {cache.store(e,100,[](void *){throw std::runtime_error("fill fault");});}catch(...) {failed=true;}
            check(failed && !cache.contains(e,100) && !physical,"failed fill never published");
        }
        check(live.empty(),"frequency cache teardown");
        {
            ExpertCache cache(256,alloc,release,256,{}, {},8);
            cache.constrain(size_t(8)<<30,size_t(32)<<30);
            for(int i=0;i<7;++i)cache.record(a);cache.store(a,100,fill);
            cache.record(b);cache.store(b,100,no_fill);
            for(int i=0;i<8;++i)cache.record(b);cache.store(b,100,fill);
            check(cache.contains(b,100) && !cache.contains(a,100),"decay lets changed workload replace old hot expert");
        }
        {
            StrataExpertFrequencyHistory<MatrixKey,MatrixHash> history(1024,2);
            for(int i=0;i<300;++i)history.record(a);
            check(history.score(a)==255,"bounded frequency saturation");
            history.record(b);history.record(c);
            check(history.size()==1 && history.score(a)==0 && history.score(c)==1,"bounded history reset");
        }
        {
            ExpertSlab slab(4096,1024,alloc,release);ExpertCache *owner=nullptr;
            ExpertCache cache(8192,[&](size_t n){return slab.get(n,owner->budget());},[&](void *p){slab.put(p);},256,
                [&]{return slab.reserved();},[&](size_t n){return slab.growth(n);},64);
            owner=&cache;cache.constrain(size_t(8)<<30,size_t(32)<<30);
            std::mt19937 random(709);
            for(uint32_t i=0;i<5000;++i) {
                const uint32_t k=random()%5?random()%12:random()%96;
                const size_t bytes=256*(1+k%3);const MatrixKey key{1,k%3,k};
                cache.record(key);
                if(auto p=cache.get(key,bytes))require(static_cast<unsigned char *>(p)[0]==k && static_cast<unsigned char *>(p)[bytes-1]==k,"cached payload changed");
                else cache.store(key,bytes,[&](void *p){std::memset(p,k,bytes);});
                require(cache.bytes()==physical && cache.bytes()<=cache.budget() && cache.payload_bytes()<=cache.slot_bytes() && cache.slot_bytes()<=physical,"packed frequency physical accounting");
                if(i%71==0)cache.trim(4096);
            }
            check(cache.stats().frequency_rejected>0 && cache.stats().reuses>0,"mixed packed cache exercises rejection and reuse");
            check(cache.history_keys()<=96 && cache.stats().frequency_updates==5000,"bounded actual-matrix training");
            cache.trim(0);check(!physical && !cache.bytes(),"packed frequency pressure releases whole blocks");
        }
        check(live.empty() && !physical,"no leaks");
        std::cout<<"PASS "<<checks<<" frequency cases and 5000 mixed-cache operations\n";return 0;
    }catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
