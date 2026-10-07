// Fault-injected allocator/cache ownership checks; no model arithmetic.
#include "expert_slab.hpp"
#include "expert_cache.hpp"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <set>
using namespace mimo2;
static void require(bool ok,const char *message) {if(!ok)throw std::runtime_error(message);}
int main() {
    try {
        std::set<void *> live;bool oom=false;size_t checks=0,physical=0;
        auto check=[&](bool ok,const char *name) {require(ok,name);++checks;};
        std::map<void *,size_t> sizes;
        auto alloc=[&](size_t n)->void * {if(oom)return nullptr;auto *p=std::malloc(n);require(p,"host alloc");live.insert(p);sizes[p]=n;physical+=n;return p;};
        auto free=[&](void *p) {require(live.erase(p)==1,"foreign/double physical free");physical-=sizes.at(p);sizes.erase(p);std::free(p);};
        for(size_t matrix:{size_t(2752512),size_t(3604480),size_t(4456448)}) {
            ExpertSlab slab(16u<<20,2u<<20,alloc,free);std::vector<void *> pointers;
            const size_t count=(16u<<20)/matrix,capacity=((count*matrix+(2u<<20)-1)/(2u<<20))*(2u<<20);
            for(size_t i=0;i<count;++i) {auto *p=slab.get(matrix,capacity);require(p,"slot allocation");std::memset(p,int(i+1),matrix);pointers.push_back(p);}
            check(slab.reserved()==capacity && physical==capacity && slab.stats().allocations==1,"physical block rounding");
            check(slab.stats().slots==matrix*count,"slot and padding accounting");
            check(slab.get(matrix,capacity)==nullptr,"full slab cannot exceed limit");
            auto old=pointers.back();pointers.pop_back();slab.put(old);
            check(slab.reserved()==capacity && slab.growth(matrix)==0,"holes stay physically charged");
            auto replacement=slab.get(matrix,capacity);check(replacement==old && slab.stats().reuses==count,"compatible slot reuse");
            slab.put(replacement);
            for(size_t i=0;i<pointers.size();++i) {
                check(static_cast<uint8_t *>(pointers[i])[0]==i+1 && static_cast<uint8_t *>(pointers[i])[matrix-1]==i+1,"neighbor payload intact");
                slab.put(pointers[i]);
            }
            check(!slab.reserved() && live.empty(),"empty block returned");
            auto *tail=slab.get(matrix,6u<<20);require(tail,"partial final block");
            check(slab.reserved()<=(6u<<20) && slab.stats().slots==matrix,"last block obeys remaining cap");
            bool rejected=false;try {slab.put(static_cast<uint8_t *>(tail)+1);}catch(...) {rejected=true;}
            check(rejected,"interior pointer rejected");slab.put(tail);
            rejected=false;try {slab.put(tail);}catch(...) {rejected=true;}
            check(rejected,"double release rejected");
            oom=true;auto *p=slab.get(matrix,16u<<20);oom=false;
            check(!p && !slab.reserved() && slab.stats().oom==1,"OOM does not publish a block");
        }
        check(live.empty() && physical==0,"all production classes cleaned up");
        {
            ExpertSlab slab(4096,1024,alloc,free);std::vector<void *> pointers;std::set<void *> used;
            for(int i=0;i<24;++i) {auto *p=slab.get(512,12288);require(p,"three-block allocation");pointers.push_back(p);used.insert(p);}
            check(slab.stats().allocations==3 && slab.reserved()==12288,"multiple full blocks");
            for(int i:{0,8,9,16}) {slab.put(pointers[i]);used.erase(pointers[i]);}
            auto *expected=reinterpret_cast<uintptr_t>(pointers[0])<reinterpret_cast<uintptr_t>(pointers[16])?pointers[0]:pointers[16];
            auto *p=slab.get(512,12288);used.insert(p);
            check(p==expected,"partial index preserves fullest-block and address tie order");
            for(auto *entry:used)slab.put(entry);
            check(!slab.reserved() && !slab.stats().slots,"empty blocks unlink from partial index");
            p=slab.get(512,4096);check(p && slab.reserved()==4096,"empty size class can be reused");slab.put(p);
        }
        {
            ExpertSlab slab(4096,1024,alloc,free);
            ExpertCache *owner=nullptr;
            ExpertCache cache(4096,[&](size_t n){return slab.get(n,owner->budget());},[&](void *p){slab.put(p);},256,
                [&]{return slab.reserved();},[&](size_t n){return slab.growth(n);});
            owner=&cache;
            cache.constrain(size_t(8)<<30,size_t(32)<<30);
            for(uint32_t i=0;i<16;++i)cache.store({1,1,i},256,[&](void *p){std::memset(p,int(i),256);});
            check(cache.bytes()==4096 && cache.payload_bytes()==4096,"cache uses physical limit");
            auto pin=cache.protect({{1,1,0}});
            cache.store({1,1,16},256,[](void *p){std::memset(p,16,256);});
            check(cache.contains({1,1,0},256) && !cache.contains({1,1,1},256) && cache.stats().reuses==1,"reuse respects plan pins");
            bool failed=false;try {cache.store({1,1,17},256,[](void *p){std::memset(p,17,128);throw std::runtime_error("fill failure");});}catch(...) {failed=true;}
            check(failed && !cache.contains({1,1,17},256) && cache.bytes()==4096 && cache.payload_bytes()==15*256,"failed fill returns slot without hiding physical block");
            cache.store({1,1,18},256,[](void *p){std::memset(p,18,256);});
            check(cache.contains({1,1,18},256) && slab.stats().allocations==1,"hole reused at physical ceiling");
            cache.constrain(0,size_t(32)<<30);
            check(cache.budget()==0 && cache.bytes()==4096 && cache.payload_bytes()==256,"pressure keeps pinned block charged");
            cache.store({1,1,19},256,[](void *){throw std::runtime_error("no admission under pressure");});
            check(!cache.contains({1,1,19},256),"pressure blocks admission");
            pin.reset();cache.trim(0);check(!physical && !cache.bytes(),"unpin frees pressure block");
        }
        {
            ExpertSlab slab(4096,1024,alloc,free);
            ExpertCache *owner=nullptr;
            ExpertCache cache(8192,[&](size_t n){return slab.get(n,owner->budget());},[&](void *p){slab.put(p);},256,
                [&]{return slab.reserved();},[&](size_t n){return slab.growth(n);});
            owner=&cache;
            cache.constrain(size_t(8)<<30,size_t(32)<<30);
            std::mt19937 random(703);
            for(uint32_t i=0;i<5000;++i) {
                const uint32_t k=random()%96;const size_t n=256*(1+k%3);const MatrixKey key{1,k%3,k};
                if(auto *p=cache.get(key,n)) {
                    require(static_cast<uint8_t *>(p)[0]==k && static_cast<uint8_t *>(p)[n-1]==k,"mixed reuse corrupted payload");
                }else cache.store(key,n,[&](void *p){std::memset(p,k,n);});
                require(cache.bytes()==physical && cache.bytes()<=cache.budget() && cache.payload_bytes()<=cache.slot_bytes() && cache.slot_bytes()<=physical,"mixed physical accounting");
                if(i%71==0)cache.trim(4096);
            }
            check(cache.stats().evictions>0 && cache.stats().reuses>0,"mixed class pressure and reuse");
            cache.trim(0);check(!physical && !cache.payload_bytes(),"mixed trim releases blocks");
        }
        check(live.empty() && !physical,"no leaked blocks");
        bool rejected=false;try {ExpertSlab bad(4096,768,alloc,free);}catch(...) {rejected=true;}
        check(rejected,"unaligned block geometry rejected");
        std::cout<<"PASS "<<checks<<" slab ownership cases and 5000 mixed-cache operations\n";return 0;
    }catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
