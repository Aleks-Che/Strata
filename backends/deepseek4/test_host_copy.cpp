#include "host_copy.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif
struct Guarded {
    uint8_t *base=nullptr;size_t page=4096,capacity=8<<20,total=0;
    Guarded() {
#ifdef _WIN32
        SYSTEM_INFO info;GetSystemInfo(&info);page=info.dwPageSize;
        total=capacity+2*page;
        base=static_cast<uint8_t *>(VirtualAlloc(nullptr,total,MEM_COMMIT|MEM_RESERVE,PAGE_READWRITE));
        DWORD old=0;
        if(!base || !VirtualProtect(base,page,PAGE_NOACCESS,&old) ||
           !VirtualProtect(base+page+capacity,page,PAGE_NOACCESS,&old))throw std::runtime_error("guard allocation failed");
#else
        page=size_t(sysconf(_SC_PAGESIZE));total=capacity+2*page;
        base=static_cast<uint8_t *>(mmap(nullptr,total,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0));
        if(base==MAP_FAILED || mprotect(base,page,PROT_NONE) || mprotect(base+page+capacity,page,PROT_NONE))
            throw std::runtime_error("guard allocation failed");
#endif
    }
    ~Guarded() {
#ifdef _WIN32
        if(base)VirtualFree(base,0,MEM_RELEASE);
#else
        if(base && base!=MAP_FAILED)munmap(base,total);
#endif
    }
    uint8_t *data(){return base+page;}
    uint8_t *end(){return data()+capacity;}
};
int main() {
    if(!strata_ds4_host_copy::available()){std::puts("SKIP: AVX2 unavailable");return 77;}
    try {
        Guarded source,dest;size_t cases=0;
        for(size_t i=0;i<source.capacity;++i)source.data()[i]=uint8_t((i*131+i/4093)%251);
        // Include DeepSeek IQ3_XXS/MXFP4 payloads and the final-expert padding.
        for(size_t bytes:{size_t(0),size_t(1),size_t(31),size_t(63),size_t(64),size_t(65),size_t(127),
            size_t(128),size_t(129),size_t(511),size_t(512),size_t(513),size_t(4095),size_t(4096),
            size_t(4097),size_t(3211264),size_t(3211776),size_t(4456448),size_t(4456960)})
            for(size_t a:{0,1,15,31,32,63})for(size_t b:{0,1,15,31,32,63}) {
                auto *src=source.end()-bytes-a;auto *dst=dest.end()-bytes-b;
                std::memset(dst-64,0xD7,bytes+b+64);
                strata_ds4_host_copy::copy(dst,src,bytes);
                if(std::memcmp(src,dst,bytes) || !std::all_of(dst-64,dst,[](uint8_t x){return x==0xD7;}) ||
                   !std::all_of(dst+bytes,dest.end(),[](uint8_t x){return x==0xD7;}))
                    throw std::runtime_error("AVX2 bytes/guards differ");
                ++cases;
            }
        std::printf("AVX2 byte/guard cases passed: %zu\n",cases);return 0;
    }catch(const std::exception &e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
