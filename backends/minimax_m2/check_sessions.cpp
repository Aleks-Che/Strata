#include "sessions.hpp"
#include <iostream>
#include <stdexcept>
using namespace minimax_m2;
int main() {
    try {
        size_t checks=0;
        auto check=[&](bool b){++checks;if(!b)throw std::runtime_error("check "+std::to_string(checks));};
        ResidentPrefix p;p.commit(std::string(64,'a'),std::vector<int32_t>(52,42),16,59);
        const auto charge=SessionArchive::charge_for(p,100);
        auto plenty=[](){return size_t(1)<<30;};
        auto write=[](uint8_t *data,size_t n){std::fill(data,data+n,17);return true;};
        for(size_t slots:{1,2,4})for(size_t capacity:{charge-1,charge,2*charge,4*charge}) {
            SessionArchive a(capacity,slots);
            for(int i=0;i<12;++i) {
                p.key=std::string(64,char('a'+i));
                const bool room=a.room(charge,256,{},plenty);
                check(room==(capacity>=charge));
                if(room){check(a.save(p,100,write));const auto *e=a.find(p.key);check(e && e->data[99]==17 && e->prefix.computed==59);}
                check(a.used<=capacity && a.count()<=slots);
            }
        }
        p.key=std::string(64,'a');SessionArchive a(charge*2,2);
        check(a.room(charge,0,{},plenty));check(a.save(p,100,write));
        p.key=std::string(64,'b');check(a.room(charge,0,{},plenty));check(a.save(p,100,write));
        const auto key_a=std::string(64,'a'),key_b=std::string(64,'b');
        check(a.room(charge,0,key_a,plenty));check(a.find(key_a) && !a.find(key_b)); // protect requested oldest
        p.key=key_b;check(a.save(p,100,write));
        check(!a.room(charge*2,0,key_a,plenty));check(a.count()==1 && a.find(key_a));
        // Simulated global pressure frees nothing until all virtual blobs go:
        // re-query available rather than adding the evicted charge to it.
        size_t reads=0;a.trim(100,[&]{++reads;return size_t(0);});check(a.count()==0 && a.used==0 && reads==2);
        check(!a.room(charge,100,{},[](){return size_t(99);}));
        check(!a.room(charge,SIZE_MAX,{},plenty));
        check(a.room(charge,0,{},plenty));check(!a.save(p,100,[](uint8_t*,size_t){return false;}));check(a.used==0 && a.count()==0);
        check(!a.save(p,100,[](uint8_t*,size_t)->bool{throw std::bad_alloc();}));check(a.used==0);
        check(a.save(p,100,write));
        const auto *entry=a.find(key_b);check(entry->prefix.reusable(key_b,std::vector<int32_t>(52,42),16)==48);
        check(entry->prefix.reusable(key_b,std::vector<int32_t>(52,43),16)==0);
        check(entry->prefix.reusable(key_b,std::vector<int32_t>(52,42),8)==0);
        a.erase(key_b);check(a.used==0 && a.count()==0);a.erase(key_b);
        SessionArchive disabled(0,4);check(!disabled.room(charge,0,{},plenty));
        SessionArchive single(charge,1);p.key=key_a;check(single.save(p,100,write));
        check(!single.room(charge,0,key_a,plenty));check(single.find(key_a) && single.used==charge);
        check(SessionArchive::charge_for(p,SIZE_MAX)==SIZE_MAX);
        std::cout<<"PASS "<<checks<<" session archive checks\n";
        return 0;
    } catch(const std::exception &e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
}
