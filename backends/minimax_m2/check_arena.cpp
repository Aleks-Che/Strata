#include "gpu_arena.hpp"
#include "../step35/expert_cache.hpp"
#include "nlohmann/json.hpp"
#include <fstream>
#include <iostream>
#include <thread>
#include <set>
using json=nlohmann::ordered_json;
using step35::cuda_check;
static minimax_m2::GpuArena::Memory sample() {
    const auto m=step35::memory_sample();MEMORYSTATUSEX ram{};ram.dwLength=sizeof(ram);
    if(!GlobalMemoryStatusEx(&ram))throw std::runtime_error("commit sample unavailable");
    return {m.gpu_free,m.gpu_total,size_t(ram.ullAvailPageFile)};
}
static size_t used() {const auto m=sample();return m.total-m.free;}
static void settle() {cuda_check(cudaDeviceSynchronize());std::this_thread::sleep_for(std::chrono::milliseconds(200));}
int main(int argc,char **argv) {
    json tests=json::array(),measurements=json::array();
    auto check=[&](const char *name,bool pass) {
        tests.push_back({{"name",name},{"pass",pass}});
        if(!pass)throw std::runtime_error(name);
    };
    json report;bool writable=false;
    try {
        if(argc!=2)throw std::runtime_error("usage: arena-check NEW_REPORT.json");
        std::ifstream previous(argv[1]);if(previous.good())throw std::runtime_error("report already exists");
        writable=true;
        cuda_check(cudaSetDevice(0));cuda_check(cudaFree(nullptr));
        constexpr size_t q4=41*65536,q6=60*65536;
        {
            minimax_m2::GpuArena arena(128u<<20,sample);std::vector<void *> p(24,nullptr);
            for(size_t i=0;i<p.size();++i) {cuda_check(arena.allocate(&p[i],q4));cuda_check(cudaMemset(p[i],int(i+1),q4));}
            check("24_Q4_slots_one_64MiB_slab",arena.snapshot().reserved==(64u<<20) && arena.snapshot().live==24*q4);
            check("unique_64KiB_aligned_slots",std::set<void *>(p.begin(),p.end()).size()==24 &&
                std::all_of(p.begin(),p.end(),[](void *p){return uintptr_t(p)%65536==0;}));
            check("misaligned_release_refused",arena.release(static_cast<char *>(p[0])+1)==cudaErrorInvalidValue);
            for(size_t i=0;i<p.size();i+=2)cuda_check(arena.release(p[i]));
            check("holes_count_as_backing",arena.snapshot().reserved==(64u<<20) && arena.snapshot().live==12*q4);
            check("double_release_refused",arena.release(p[0])==cudaErrorInvalidValue);
            for(size_t i=0;i<p.size();i+=2){cuda_check(arena.allocate(&p[i],q4));cuda_check(cudaMemset(p[i],int(i+1),q4));}
            bool exact=true;
            for(size_t i=0;i<p.size();++i) {
                std::vector<unsigned char> bytes(q4);cuda_check(cudaMemcpy(bytes.data(),p[i],q4,cudaMemcpyDeviceToHost));
                exact&=std::all_of(bytes.begin(),bytes.end(),[i](unsigned char b){return b==i+1;});
            }
            check("reuse_all_bytes_exact",exact && arena.snapshot().allocations==1);
            void *other=nullptr,*third=nullptr;cuda_check(arena.allocate(&other,q6));
            check("two_size_classes_accounted",arena.snapshot().reserved==(128u<<20));
            check("hard_backing_cap_includes_holes",arena.allocate(&third,65536)==cudaErrorMemoryAllocation && !third && arena.snapshot().reserved==(128u<<20));
            cuda_check(arena.release(other));check("last_slot_releases_its_slab",arena.snapshot().reserved==(64u<<20));
            for(auto ptr:p)cuda_check(arena.release(ptr));
            check("all_backing_released",!arena.snapshot().reserved && !arena.snapshot().live);
        }
        for(bool commit:{false,true}) {
            auto m=sample();if(commit)m.commit_free=(1u<<30)+(32u<<20);else m.free=m.total/20+(256u<<20);
            minimax_m2::GpuArena arena(128u<<20,[&]{return m;});void *p=nullptr;
            check(commit?"whole_slab_commit_reserve":"VRAM_reserve_refuses_slab",
                arena.allocate(&p,q4)==cudaErrorMemoryAllocation && !p && !arena.snapshot().reserved);
        }
        // Identical touched matrices, sequential allocators, same GPU. Global
        // NVML deltas include external activity; do not equate them to payload.
        for(bool arena:{false,true}) {
            minimax_m2::GpuArena pool(size_t(2)<<30,sample);std::vector<void *> p;size_t live=0;
            settle();const auto before=used();
            for(int i=0;i<512;++i) {
                const size_t n=i<384?q4:q6;void *ptr=nullptr;
                cuda_check(arena?pool.allocate(&ptr,n):cudaMalloc(&ptr,n));p.push_back(ptr);live+=n;
                cuda_check(cudaMemset(ptr,17,n));
            }
            settle();const auto during=used();
            measurements.push_back({{"allocator",arena?"arena":"cuda"},{"matrices",p.size()},{"requested_bytes",live},
                {"arena_reserved",pool.snapshot().reserved},{"NVML_before",before},{"NVML_during",during},
                {"NVML_delta",int64_t(during)-int64_t(before)}});
            for(auto ptr:p)cuda_check(arena?pool.release(ptr):cudaFree(ptr));settle();
            check(arena?"measured_arena_released":"measured_cuda_released",!pool.snapshot().reserved);
        }
        report["pass"]=true;
    }catch(const std::exception &e){report["pass"]=false;report["error"]=e.what();std::cerr<<e.what()<<'\n';}
    report["tests"]=tests;report["cases"]=tests.size();report["measurements"]=measurements;
    if(writable){std::ofstream out(argv[1]);out<<report.dump(2)<<'\n';}
    std::cout<<report.dump(2)<<'\n';return report.value("pass",false)?0:1;
}
