// CPU ownership tests with deferred writes: freeing in-flight destinations fails.
#include "expert_cache.hpp"
#include "expert_slab.hpp"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <random>
using namespace mimo2;
static void require(bool value,const char *name) {if(!value)throw std::runtime_error(name);}
int main() {
    try {
        size_t checks=0,fences=0;bool fail_fence=false;
        std::map<void *,size_t> live;std::vector<std::pair<void *,unsigned char>> writes;
        auto check=[&](bool value,const char *name){require(value,name);++checks;};
        auto alloc=[&](size_t n)->void * {auto p=std::malloc(n);require(p,"allocation");live[p]=n;return p;};
        auto release=[&](void *p) {
            for(const auto &w:writes)require(w.first!=p,"freed in-flight destination");
            require(live.erase(p)==1,"foreign free");std::free(p);
        };
        auto fence=[&] {
            ++fences;for(const auto &w:writes) {require(live.count(w.first),"write to dead allocation");std::memset(w.first,w.second,100);}
            writes.clear();if(fail_fence)throw std::runtime_error("completed fence reports error");
        };
        auto enqueue=[&](unsigned char value) {return [&,value](void *p){writes.push_back({p,value});};};
        const MatrixKey a{1,1,1},b{1,1,2},c{1,1,3};
        {
            ExpertCache cache(512,alloc,release,256);cache.constrain(size_t(8)<<30,size_t(32)<<30);
            {
                ExpertCache::FillBatch batch(cache,fence);batch.store(a,100,enqueue(1));batch.store(b,100,enqueue(2));
                check(cache.pending()==2 && !cache.get(a,100) && !cache.contains(b,100),"reservations invisible before completion");
                check(writes.empty() && !fences,"admission submits no copies");
                bool nested=false;try {ExpertCache::FillBatch bad(cache,fence);}catch(...) {nested=true;}
                check(nested,"nested batch rejected");
                bool sync=false;try {cache.store(c,100,[](void *){});}catch(...) {sync=true;}
                check(sync,"mixed synchronous admission rejected");
                batch.finish();check(fences==1 && cache.pending()==0,"one fence publishes both fills");
                check(static_cast<unsigned char *>(cache.get(a,100))[99]==1 && static_cast<unsigned char *>(cache.get(b,100))[99]==2,"completed payloads");
            }
            cache.trim(0);fences=0;
            {
                ExpertCache::FillBatch batch(cache,fence);
                batch.store(a,100,[](void *){throw std::runtime_error("obsolete fill executed");});
                batch.store(b,100,enqueue(3));batch.store(c,100,enqueue(4));
                check(cache.pending()==2 && cache.stats().reuses>0,"reservation reuse retains budget");
                batch.finish();check(fences==1 && !cache.contains(a,100) && cache.contains(c,100),"evicted pending fill never submitted");
            }
            cache.trim(0);fences=0;
            {
                ExpertCache::FillBatch batch(cache,fence);
                batch.store(a,100,[](void *){throw std::runtime_error("old ticket executed");});
                batch.store(b,100,enqueue(2));batch.store(c,100,enqueue(3));batch.store(a,100,enqueue(9));
                batch.finish();check(fences==1 && static_cast<unsigned char *>(cache.get(a,100))[0]==9,"same key and reused address have distinct tickets");
            }
            cache.trim(0);fences=0;
            {ExpertCache::FillBatch batch(cache,fence);batch.store(a,100,enqueue(1));}
            check(!fences && !cache.pending() && !cache.contains(a,100) && live.empty(),"abandoned unsubmitted batch discarded");
            {
                ExpertCache::FillBatch batch(cache,fence);batch.store(a,100,enqueue(1));
                batch.store(b,100,[&](void *p){writes.push_back({p,2});throw std::runtime_error("cancel after enqueue");});
                bool failed=false;try {batch.finish();}catch(...) {failed=true;}
                check(failed && fences==1 && writes.empty() && live.empty() && !cache.pending(),"enqueue failure drains before discard");
            }
            fences=0;fail_fence=true;
            {
                ExpertCache::FillBatch batch(cache,fence);batch.store(a,100,enqueue(1));
                bool failed=false;try {batch.finish();}catch(...) {failed=true;}
                check(failed && fences==1 && writes.empty() && live.empty() && !cache.pending(),"failed fence never publishes");
            }
            fail_fence=false;fences=0;
            cache.store(a,100,[](void *p){std::memset(p,7,100);});auto pins=cache.protect({a});
            {ExpertCache::FillBatch batch(cache,fence);batch.store(b,100,enqueue(2));batch.store(c,100,enqueue(3));batch.finish();}
            check(cache.contains(a,100) && !cache.contains(b,100) && cache.contains(c,100),"completed route pins survive pending evictions");
            pins.reset();cache.trim(0);fences=0;
            {ExpertCache::FillBatch batch(cache,fence);batch.finish();}
            check(!fences,"empty batch needs no fence");
        }
        check(live.empty() && writes.empty(),"batch teardown");
        // Compare final residency and data to synchronous admission after each
        // tensor, including repeated keys, decay, mixed slots and pressure.
        for(uint64_t decay:{uint64_t(0),uint64_t(64)}) {
            auto raw=[](size_t n){return std::malloc(n);};auto free=[](void *p){std::free(p);};
            ExpertSlab slab_a(4096,1024,raw,free),slab_b(4096,1024,raw,free);ExpertCache *pa=nullptr,*pb=nullptr;
            ExpertCache sync(8192,[&](size_t n){return slab_a.get(n,pa->budget());},[&](void *p){slab_a.put(p);},256,[&]{return slab_a.reserved();},[&](size_t n){return slab_a.growth(n);},decay);
            ExpertCache batch(8192,[&](size_t n){return slab_b.get(n,pb->budget());},[&](void *p){slab_b.put(p);},256,[&]{return slab_b.reserved();},[&](size_t n){return slab_b.growth(n);},decay);
            pa=&sync;pb=&batch;sync.constrain(size_t(8)<<30,size_t(32)<<30);batch.constrain(size_t(8)<<30,size_t(32)<<30);
            std::mt19937 random(710);
            for(unsigned tensor=0;tensor<1000;++tensor) {
                bool seen[96]{};
                std::vector<std::function<void()>> deferred;
                ExpertCache::FillBatch fills(batch,[&]{for(auto &fn:deferred)fn();deferred.clear();});
                for(unsigned i=0;i<8;++i) {
                    unsigned k;do {k=random()%96;}while(seen[k]);seen[k]=true;
                    const MatrixKey key{1,k%3,k};const size_t bytes=256*(1+k%3);
                    sync.record(key);batch.record(key);
                    if(sync.contains(key,bytes)) {sync.touch(key);batch.touch(key);}
                    else {
                        sync.store(key,bytes,[&](void *p){std::memset(p,k,bytes);});
                        fills.store(key,bytes,[&,k,bytes](void *p){deferred.push_back([=]{std::memset(p,k,bytes);});});
                    }
                }
                fills.finish();require(!batch.pending(),"random drain");
                require(sync.bytes()==batch.bytes() && sync.payload_bytes()==batch.payload_bytes(),"random physical/residency parity");
                for(unsigned k=0;k<96;++k) {
                    const MatrixKey key{1,k%3,k};const size_t bytes=256*(1+k%3);
                    require(sync.contains(key,bytes)==batch.contains(key,bytes),"random selected victim parity");
                    if(batch.contains(key,bytes))require(std::memcmp(sync.get(key,bytes,false),batch.get(key,bytes,false),bytes)==0,"random payload parity");
                }
                if(tensor%31==0) {sync.trim(4096);batch.trim(4096);}
            }
            check(true,"8000 mixed admission operations preserve cache policy");
        }
        std::cout<<"PASS "<<checks<<" fill batch cases and 16000 mixed admission operations\n";return 0;
    }catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
