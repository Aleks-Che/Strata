#include "host_copy.hpp"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#ifndef _WIN32
#include <sys/mman.h>
#include <unistd.h>
#endif
struct Guarded {
    uint8_t * base=nullptr;size_t page=4096,capacity=4<<20,total=0;
    Guarded(){
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
    ~Guarded(){
#ifdef _WIN32
        if(base)VirtualFree(base,0,MEM_RELEASE);
#else
        if(base && base!=MAP_FAILED)munmap(base,total);
#endif
    }
    uint8_t * data(){return base+page;}
    uint8_t * end(){return data()+capacity;}
};
int main(int argc,char **argv){
    using json=nlohmann::ordered_json;
    json report={{"status","error"},{"scope","AVX2 cached copy, unaligned ranges and inaccessible end pages"}};
    try{
        if(!step35::host_copy_avx2_available())throw std::runtime_error("AVX2 unavailable; optimized checks not run");
        Guarded source,destination;
        for(size_t i=0;i<source.capacity;++i)source.data()[i]=uint8_t((i*131+i/4093)%251);
        size_t cases=0;
        for(size_t bytes:{size_t(0),size_t(1),size_t(7),size_t(15),size_t(16),size_t(31),size_t(32),size_t(63),
                size_t(64),size_t(65),size_t(127),size_t(128),size_t(129),size_t(255),size_t(256),size_t(511),
                size_t(512),size_t(513),size_t(4095),size_t(4096),size_t(4097),size_t(2949120),size_t(2949632)})
            for(size_t a:{0,1,15,31,32,63})for(size_t b:{0,1,15,31,32,63}) {
                auto *src=source.end()-bytes-a;auto *dst=destination.end()-bytes-b;
                std::memset(dst-64,0xD7,bytes+b+64);
                step35::host_copy_avx2_cached(dst,src,bytes);
                if(std::memcmp(src,dst,bytes) || !std::all_of(dst-64,dst,[](uint8_t v){return v==0xD7;}) ||
                   !std::all_of(dst+bytes,destination.end(),[](uint8_t v){return v==0xD7;}))
                    throw std::runtime_error("host copy bytes/guards differ");
                ++cases;
            }
        step35::HostCopyState state;state.configure(0,false);state(destination.data(),source.data(),513);
        if(state.snapshot().copies)throw std::runtime_error("disabled host profiling recorded samples");
        state.configure(1,true);state(destination.data()+1,source.data()+3,2949632);
        auto p=state.snapshot();
        if(p.copies!=1 || p.bytes!=2949632 || !p.wall_ns || std::memcmp(destination.data()+1,source.data()+3,2949632))
            throw std::runtime_error("host profiling accounting failed");
        bool rejected=false;try{state.configure(2,false);}catch(const std::runtime_error&){rejected=true;}
        if(!rejected)throw std::runtime_error("invalid copy mode accepted");
        report["status"]="pass";report["byte_guard_cases"]=cases;report["profile_switch_invalid_cases"]=3;
    }catch(const std::exception&e){report["error"]=e.what();}
    if(argc==2){std::ofstream out(argv[1]);out<<report.dump(2)<<'\n';}
    std::cout<<report.dump(2)<<'\n';return report["status"]=="pass"?0:1;
}
