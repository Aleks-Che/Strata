#include "vram_policy.hpp"
#include <cstdio>

int main() {
    constexpr uint64_t mib=1ULL<<20;
    StrataVramPolicy p;
    p.mode=2;p.target_mib=12000;p.reserve_mib=1024;
    int failed=0;
    auto check=[&](bool ok,const char *name){if(!ok){std::fprintf(stderr,"FAIL: %s\n",name);++failed;}};
    check(p.valid(),"valid policy");
    check(p.byte_limit(4000*mib,16000*mib,6000*mib)==6000*mib,"steady state");
    check(p.byte_limit(2000*mib,16000*mib,6000*mib)==4000*mib,"other app consumes 2 GiB");
    check(p.byte_limit(7000*mib,16000*mib,4000*mib)==7000*mib,"other app releases memory");
    check(p.byte_limit(1000*mib,16000*mib,1000*mib)==0,"fixed memory above target");
    p.mode=1;p.matrices=0;
    check(p.valid(),"zero cache count");
    check(p.byte_limit(512*mib,16000*mib,6000*mib)==5488*mib,"manual count still keeps reserve");
    p.reserve_mib=20000;
    check(p.byte_limit(16000*mib,16000*mib,0)==0,"reserve exceeds device");
    p.reserve_mib=0;check(!p.valid(),"reserve floor");
    p.reserve_mib=1024;p.mode=2;p.target_mib=0;check(!p.valid(),"zero target invalid");
    p.mode=1;p.matrices=1000001;check(!p.valid(),"bounded count");
    return failed?1:0;
}
