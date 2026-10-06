#include "gpu_arena.hpp"
#include "../step35/expert_cache.hpp"
#include "nlohmann/json.hpp"
#include <fstream>
#include <iostream>
#include <set>
using json=nlohmann::ordered_json;
int main(int argc,char **argv) {
    json report={{"status","error"},{"cases",json::array()}};
    auto test=[&](const char *name,bool pass) {
        report["cases"].push_back({{"name",name},{"pass",pass}});
        if(!pass)throw std::runtime_error(name);
    };
    using step35::cuda_check;
    try {
        if(argc!=2)throw std::runtime_error("usage: arena-check NEW_REPORT.json");
        cuda_check(cudaSetDevice(0));
        auto probe=[] {const auto m=step35::memory_sample();return hy3::GpuArena::Memory{m.gpu_free,m.gpu_total};};
        hy3::GpuArena arena(64<<20,probe);
        std::vector<void*> ptrs(27,nullptr);
        for(size_t i=0;i<ptrs.size();++i) {
            cuda_check(arena.allocate(&ptrs[i],2424832));
            cuda_check(cudaMemset(ptrs[i],int(i+1),2424832));
        }
        test("27_matrices_one_64MiB_block",arena.snapshot().reserved==64<<20 && arena.snapshot().blocks==1 && arena.snapshot().allocations==1);
        test("unique_aligned_slots",std::set<void*>(ptrs.begin(),ptrs.end()).size()==27 &&
            std::all_of(ptrs.begin(),ptrs.end(),[](void *p){return reinterpret_cast<uintptr_t>(p)%65536==0;}));
        test("misaligned_release_refused",arena.release(static_cast<uint8_t*>(ptrs[0])+1)==cudaErrorInvalidValue);
        for(size_t i=0;i<ptrs.size();i+=2)cuda_check(arena.release(ptrs[i]));
        test("partial_release_keeps_live_block",arena.snapshot().reserved==64<<20 && arena.snapshot().live==13*2424832ULL);
        test("double_release_refused",arena.release(ptrs[0])==cudaErrorInvalidValue);
        for(size_t i=0;i<ptrs.size();i+=2) {
            cuda_check(arena.allocate(&ptrs[i],2424832));cuda_check(cudaMemset(ptrs[i],int(i+1),2424832));
        }
        bool exact=true;
        for(size_t i=0;i<ptrs.size();++i) {
            uint8_t first=0,last=0;
            cuda_check(cudaMemcpy(&first,ptrs[i],1,cudaMemcpyDeviceToHost));
            cuda_check(cudaMemcpy(&last,static_cast<uint8_t*>(ptrs[i])+2424831,1,cudaMemcpyDeviceToHost));
            exact=exact && first==i+1 && last==i+1;
        }
        test("reuse_preserves_other_slots",exact && arena.snapshot().allocations==1);
        for(auto p:ptrs)cuda_check(arena.release(p));
        test("empty_block_returned",!arena.snapshot().reserved && !arena.snapshot().live && arena.snapshot().frees==1);
        void *a=nullptr,*b=nullptr,*c=nullptr;
        cuda_check(arena.allocate(&a,2424832));cuda_check(arena.allocate(&b,3407872));cuda_check(arena.allocate(&c,5177344));
        test("mixed_classes_accounted",arena.snapshot().blocks==3 && arena.snapshot().reserved==192<<20);
        cuda_check(arena.release(b));test("empty_class_released_independently",arena.snapshot().blocks==2 && arena.snapshot().reserved==128<<20);
        cuda_check(arena.release(a));cuda_check(arena.release(c));
        hy3::GpuArena refused(64<<20,[]{return hy3::GpuArena::Memory{0,size_t(32)<<30};});
        test("pressure_refuses_new_block",refused.allocate(&a,2424832)==cudaErrorMemoryAllocation && !a && !refused.snapshot().reserved);
        cudaGetLastError();
        {
            const size_t GiB=size_t(1)<<30;
            auto shared=std::make_shared<hy3::GpuArena>(16<<20,probe);
            step35::MemorySample m{16*GiB,32*GiB,64*GiB,128*GiB};
            step35::ExpertCache cache(16<<20,[&]{return m;},[shared](void **p,size_t n){return shared->allocate(p,n);},
                [shared](void *p){return shared->release(p);});
            cache.set_backing_bytes([shared]{return shared->snapshot().reserved;});
            cache.refresh();cache.set_reuse_allocations(true);
            step35::MatrixKey k{1,0,0};cache.get(k,2424832);auto *p=cache.admit(k,2424832);
            auto pins=cache.protect({k});cache.trim(0);
            test("plan_pin_preserves_arena_slot",p && cache.get(k,2424832)==p && shared->snapshot().reserved==16<<20);
            pins.reset();cache.trim(0);
            test("unpin_trim_returns_physical_block",!shared->snapshot().reserved && !cache.resident_bytes());
            cache.refresh();cache.get(k,2424832);cache.admit(k,2424832);
            m.gpu_free=((m.gpu_total/20+(1<<20)-1)/(1<<20)+256)*(1<<20);cache.refresh();
            step35::MatrixKey k2{1,0,1};cache.get(k2,2424832);auto *p2=cache.admit(k2,2424832);
            test("arena_holes_remain_available_at_reserve",p2 && cache.size()==2 &&
                cache.resident_bytes()==2*2424832ULL && shared->snapshot().reserved==16<<20 && cache.budget()==16<<20);
            m.gpu_free=m.gpu_total/20;cache.refresh();
            test("cache_pressure_frees_backing",!shared->snapshot().reserved && !cache.resident_bytes());
        }
        {
            const size_t GiB=size_t(1)<<30;
            step35::MemorySample m{16*GiB,32*GiB,64*GiB,128*GiB};
            bool deny=false;
            step35::ExpertCache cache(4*65536,[&]{return m;},[&](void **p,size_t n) {
                if(deny) {*p=nullptr;return cudaErrorMemoryAllocation;}
                return cudaMalloc(p,n);
            });
            cache.set_reuse_allocations(true);cache.set_match_size(true);cache.refresh();
            step35::MatrixKey a{2,0,0},b{2,0,1},c{2,0,2},d{2,0,3};
            for(int i=0;i<8;++i)cache.get(a,1024);
            void *pa=cache.admit(a,1024);cache.get(b,1024);void *pb=cache.admit(b,1024);
            deny=true;auto pins=cache.protect({a,b});cache.get(c,1024);
            test("allocation_fallback_respects_pins",!cache.admit(c,1024) && cache.get(a,1024)==pa && cache.get(b,1024)==pb);
            pins.reset();cache.get(c,1024);
            test("allocation_fallback_reuses_colder_slot",cache.admit(c,1024)==pb && cache.resident_bytes()==2*65536 && cache.counters().reuses==1);
            cache.get(d,1024);
            test("allocation_fallback_preserves_frequency",!cache.admit(d,1024) && cache.contains(a,1024) && cache.contains(c,1024));
            cache.trim(0);test("fallback_release_complete",!cache.resident_bytes());
        }
        report["status"]="pass";
    } catch(const std::exception &e) {report["error"]=e.what();}
    report["case_count"]=report["cases"].size();
    if(argc==2){std::ofstream out(argv[1]);out<<report.dump(2)<<'\n';}
    std::cout<<report["status"]<<' '<<report.value("error","")<<'\n';
    return report["status"]=="pass"?0:1;
}
