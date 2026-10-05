#include "../common/host_staging_copy.hpp"
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif
static void check(bool ok,const char *message) {if(!ok)throw std::runtime_error(message);}
int main() {
    try {
        constexpr size_t limit=131073,guard=128;
        std::vector<uint8_t> source(limit+256),actual(limit+256),expected(limit+256);
        for(size_t i=0;i<source.size();++i)source[i]=uint8_t(i*17+i/251);
        for(size_t n:{size_t(0),size_t(1),size_t(63),size_t(64),size_t(65),size_t(65535),size_t(65536),size_t(65537),limit})
            for(size_t so=0;so<64;++so)for(size_t dest=0;dest<64;++dest) {
                std::fill(actual.begin(),actual.end(),0xA5);expected=actual;
                std::memcpy(expected.data()+guard+dest,source.data()+so,n);
                const bool streamed=strata_host_copy::copy(actual.data()+guard+dest,source.data()+so,n,true);
                check(streamed==(strata_host_copy::supported && n>=strata_host_copy::threshold),"streaming threshold differs");
                check(actual==expected,"unaligned copy or guard differs");
            }
#ifdef _WIN32
        // A vectorized read past the last source byte must fault this fixture.
        SYSTEM_INFO sys{};GetSystemInfo(&sys);const size_t page=sys.dwPageSize,usable=page*33;
        auto *region=static_cast<uint8_t *>(VirtualAlloc(nullptr,usable+page,MEM_RESERVE|MEM_COMMIT,PAGE_READWRITE));
        check(region!=nullptr,"guarded allocation failed");DWORD old=0;
        check(VirtualProtect(region+usable,page,PAGE_NOACCESS,&old)!=0,"source guard failed");
        std::fill_n(region,usable,0x6B);
        for(size_t n:{size_t(65536),size_t(65537),size_t(131071)}) {
            strata_host_copy::copy(actual.data()+guard+1,region+usable-n,n,true);
            check(std::all_of(actual.begin()+guard+1,actual.begin()+guard+1+n,[](uint8_t v){return v==0x6B;}),"guarded source differs");
        }
        VirtualFree(region,0,MEM_RELEASE);
#endif
        std::cout<<"PASS all source/destination offsets, threshold edges, guards and last source page\n";
        return 0;
    }catch(const std::exception &e) {std::cerr<<e.what()<<"\n";return 1;}
}
