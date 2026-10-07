// CPU-only fault injection for cache ownership/budgets. No model or CUDA allocation.
#include "expert_cache.hpp"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <set>
#include <string>
using namespace mimo2;
static void require(bool ok,const char *message) {if(!ok)throw std::runtime_error(message);}
int main() {
    try {
        std::set<void *> live;bool oom=false;
        auto alloc=[&](size_t n)->void * {if(oom)return nullptr;auto *p=std::malloc(n);require(p,"test allocation failed");live.insert(p);return p;};
        auto free=[&](void *p) {require(live.erase(p)==1,"double/foreign free");std::free(p);};
        const size_t page=65536;const MatrixKey a{1,1,1},b{1,1,2},c{1,2,1},d{2,1,1};
        size_t cases=0;auto check=[&](bool ok,const char *name) {require(ok,name);++cases;};
        {
            ExpertCache cache(2*page,alloc,free);cache.constrain(size_t(8)<<30,size_t(32)<<30);
            cache.store(a,100,[&](void *p) {std::memset(p,3,100);});
            cache.store(b,100,[&](void *p) {std::memset(p,7,100);});
            check(cache.bytes()==2*page && live.size()==2,"charged allocations");
            check(static_cast<char *>(cache.get(a,100))[99]==3,"cache payload");
            cache.store(c,100,[&](void *p) {std::memset(p,9,100);});
            check(cache.get(b,100)==nullptr && cache.get(a,100) && cache.get(c,100),"LRU eviction");
            check(cache.stats().reuses==1 && cache.stats().allocations==2,"same-size allocation reuse");
            check(cache.get(d,100)==nullptr,"model generation isolation");
            bool invalid=false;try {cache.get(a,101);} catch(...) {invalid=true;}
            check(invalid,"size mismatch rejected");
            invalid=false;try {cache.store(d,100,[](void *p) {std::memset(p,1,50);throw std::runtime_error("injected fill failure");});} catch(...) {invalid=true;}
            check(invalid && cache.get(d,100)==nullptr && live.size()==1 && cache.bytes()==page,"failed fill not published and allocation released");
            cache.constrain(0,size_t(32)<<30);
            check(cache.bytes()==0 && cache.budget()==0 && live.empty(),"external pressure evicts cache");
            cache.store(a,100,[](void *) {throw std::runtime_error("should not fill");});
            check(cache.stats().bypasses==1 && live.empty(),"zero budget bypass");
            cache.constrain(size_t(8)<<30,size_t(32)<<30);oom=true;
            cache.store(a,100,[](void *) {throw std::runtime_error("should not fill");});oom=false;
            check(cache.stats().oom==1 && cache.bytes()==0,"allocation OOM falls back without publishing");
            cache.store(a,100,[](void *p) {std::memset(p,5,100);});
            cache.reset_stats();check(cache.stats().hits==0 && cache.bytes()==page,"request stats reset preserves weights");
            invalid=false;try {cache.store(a,100,[](void *) {});} catch(...) {invalid=true;}
            check(invalid && cache.bytes()==page,"duplicate admission rejected");
        }
        check(live.empty(),"destructor releases all allocations");
        {
            ExpertCache cache(8u<<20,alloc,free,2u<<20);cache.constrain(size_t(8)<<30,size_t(32)<<30);
            cache.store(a,2752512,[](void *) {});cache.store(b,3604480,[](void *) {});
            check(cache.bytes()==(8u<<20) && cache.payload_bytes()==2752512+3604480,"rounded physical budget separated from quantized payload");
            cache.trim(4u<<20);check(cache.bytes()==(4u<<20) && cache.payload_bytes()==3604480,"eviction updates physical and payload budgets");
        }
        check(live.empty(),"rounded allocation cleanup");
        {
            ExpertCache cache(2*page,alloc,free);cache.constrain(size_t(8)<<30,size_t(32)<<30);
            cache.store(a,100,[](void *) {});cache.store(b,100,[](void *) {});
            auto pins=cache.protect({a,b,c});
            cache.store(c,100,[](void *) {throw std::runtime_error("pinned entries must not be overwritten");});
            check(cache.contains(a,100) && cache.contains(b,100) && !cache.contains(c,100),"planned hits survive allocation reuse");
            check(cache.stats().bypasses==1 && cache.bytes()==2*page,"pinned budget bypass");
            auto nested=cache.protect({a});pins.reset();cache.trim(0);
            check(cache.contains(a,100) && !cache.contains(b,100),"nested plan retains ownership");
            nested.reset();cache.trim(0);check(cache.bytes()==0,"unpin permits pressure trim");
            auto absent=cache.protect({d});cache.store(d,100,[](void *) {});cache.trim(0);
            check(!cache.contains(d,100),"future synchronous fills are not pinned");
        }
        check(live.empty(),"plan ownership cleanup");
        {
            ExpertCache cache(2*page,alloc,free);cache.constrain(size_t(8)<<30,size_t(32)<<30);
            cache.store(a,100,[](void *) {});cache.store(b,100,[](void *) {});
            cache.get(a,100,false);const auto hits=cache.stats().hits;cache.touch(a);
            check(cache.stats().hits==hits,"deferred LRU touch does not double-count hits");
            cache.store(c,100,[](void *) {});
            check(cache.contains(a,100) && !cache.contains(b,100),"deferred touch preserves LRU victim order");
            bool missing=false;try {cache.touch(d);} catch(...) {missing=true;}
            check(missing,"absent deferred touch rejected");
        }
        check(live.empty(),"deferred touch cleanup");
        bool invalid=false;try {ExpertCache::charge(SIZE_MAX);} catch(...) {invalid=true;}
        check(invalid,"size overflow rejected");
        std::cout<<"PASS "<<cases<<" cache ownership/budget cases\n";return 0;
    } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
