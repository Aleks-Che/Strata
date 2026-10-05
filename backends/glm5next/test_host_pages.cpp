#include "host_pages.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
using namespace strata_glm;
static void check(bool value,const char *why) {if(!value)throw std::runtime_error(why);}
int main(int argc,char **argv) {
    try {
        constexpr uint64_t GiB=1ULL<<30;
        check(host_working_set_limit(100*GiB,20*GiB,70*GiB,90)==80*GiB,"other processes must be charged to the host target");
        check(host_working_set_limit(100*GiB,20*GiB,90*GiB,90)==90*GiB,"changing memory samples must not underflow");
        check(host_working_set_limit(100*GiB,0,0,90)==(512ULL<<20),"minimum working set allowance differs");
        check(host_working_set_limit(100*GiB,20*GiB,70*GiB,0)==0,"disabled budget must stay disabled");
        for(int percent:{-1,9,96,100}) {
            bool rejected=false;try {host_working_set_limit(100*GiB,20*GiB,70*GiB,percent);}catch(const std::invalid_argument &) {rejected=true;}
            check(rejected,"invalid RAM percentage accepted");
        }
        HostResidency cpu,gpu;cpu.bytes=85*GiB;cpu.estimated_resident_bytes=74*GiB;gpu.estimated_resident_bytes=17*GiB;
        const auto needed=required_host_working_set(99*GiB,cpu,gpu);
        check(needed==93*GiB,"non-expert working set accounting differs");
        check(host_scan_fits(needed,99*GiB) && !host_scan_fits(needed,92*GiB),"headroom guard differs");
        check(host_scan_fits(needed,needed+(256ULL<<20)) && !host_scan_fits(needed,needed+(256ULL<<20)-1),"sampling margin differs");
        check(required_host_working_set(1,cpu,gpu)==cpu.bytes,"overlapping sampled residency must not underflow");
        cpu.bytes=UINT64_MAX;cpu.estimated_resident_bytes=0;gpu.estimated_resident_bytes=0;
        bool invalid=false;try {required_host_working_set(1,cpu,gpu);}catch(const std::invalid_argument &) {invalid=true;}
        check(invalid,"budget overflow accepted");
        invalid=false;try {sample_host_residency(reinterpret_cast<void *>(UINTPTR_MAX-1),2);}catch(const std::invalid_argument &) {invalid=true;}
        check(invalid,"sample overflow accepted");
#ifdef _WIN32
        SIZE_T original_min=0,original_max=0;DWORD original_flags=0;
        check(GetProcessWorkingSetSizeEx(GetCurrentProcess(),&original_min,&original_max,&original_flags)!=0,"read initial working set limits failed");
        {
            HostWorkingSetBudget outer;outer.apply(90,2*GiB,2*GiB,0);
            SIZE_T outer_min=0,outer_max=0;DWORD outer_flags=0;
            check(GetProcessWorkingSetSizeEx(GetCurrentProcess(),&outer_min,&outer_max,&outer_flags)!=0,"read outer limit failed");
            check(outer_min==original_min && (outer_flags&QUOTA_LIMITS_HARDWS_MAX_ENABLE),"budget must retain minimum and enforce maximum");
            {
                HostWorkingSetBudget inner;inner.apply(90,GiB,GiB,0);
                // Applying twice must still restore the limits preceding this scope.
                inner.apply(80,GiB,GiB,0);
            }
            SIZE_T after_min=0,after_max=0;DWORD after_flags=0;
            check(GetProcessWorkingSetSizeEx(GetCurrentProcess(),&after_min,&after_max,&after_flags)!=0,"read restored outer limit failed");
            check(after_min==outer_min && after_max==outer_max && after_flags==outer_flags,"nested budget did not restore outer limits");
        }
        SIZE_T restored_min=0,restored_max=0;DWORD restored_flags=0;
        check(GetProcessWorkingSetSizeEx(GetCurrentProcess(),&restored_min,&restored_max,&restored_flags)!=0,"read restored limits failed");
        check(restored_min==original_min && restored_max==original_max && restored_flags==original_flags,"budget did not restore original limits");
        check(argc==2,"fixture path required");auto path=std::filesystem::u8path(argv[1]);
        SYSTEM_INFO system{};GetSystemInfo(&system);const auto page=size_t(system.dwPageSize);
        std::vector<uint8_t> reference((2<<20)+page);for(size_t i=0;i<reference.size();++i)reference[i]=uint8_t(i*17+i/251);
        {std::ofstream out(path,std::ios::binary|std::ios::trunc);out.write(reinterpret_cast<const char *>(reference.data()),reference.size());check(bool(out),"write fixture failed");}
        const auto file=CreateFileW(path.c_str(),GENERIC_READ,FILE_SHARE_READ,nullptr,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,nullptr);
        check(file!=INVALID_HANDLE_VALUE,"open fixture failed");
        const auto mapping=CreateFileMappingW(file,nullptr,PAGE_READONLY,0,0,nullptr);check(mapping!=nullptr,"create mapping failed");
        const auto data=static_cast<const uint8_t *>(MapViewOfFile(mapping,FILE_MAP_READ,0,0,reference.size()));check(data!=nullptr,"map fixture failed");
        auto cold=sample_host_residency(data,reference.size());
        check(cold.valid && cold.samples==3 && cold.valid_samples==0,"residency query must not fault mapped payload in");
        check(std::equal(reference.begin(),reference.end(),data),"mapped payload differs");
        auto warm=sample_host_residency(data,reference.size());
        check(warm.valid && warm.samples==3 && warm.valid_samples==3 && warm.estimated_resident_bytes==reference.size(),"weighted resident sample differs");
        check(std::equal(reference.begin(),reference.end(),data),"sampling changed payload");
        UnmapViewOfFile(data);CloseHandle(mapping);CloseHandle(file);std::filesystem::remove(path);
#else
        (void)argc;(void)argv;
#endif
        std::cout<<"PASS host budget guards/overflow; non-faulting residency probes and unchanged mapped bytes\n";
        return 0;
    }catch(const std::exception &e) {std::cerr<<e.what()<<"\n";return 1;}
}
