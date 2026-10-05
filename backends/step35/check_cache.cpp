#include "expert_cache.hpp"
#include "nlohmann/json.hpp"
#include <fstream>
#include <iostream>
using namespace step35;
using json=nlohmann::ordered_json;
int main(int argc,char ** argv) {
    json report={{"status","error"},{"scope","real CUDA cache allocations with deterministic memory/allocator probes"}};
    json cases=json::array();
    auto test=[&](const char * name,bool pass) {cases.push_back({{"name",name},{"pass",pass}});if(!pass)throw std::runtime_error(name);};
    try {
        cuda_check(cudaSetDevice(0));
        const size_t GiB=size_t(1)<<30;
        MemorySample sample{16*GiB,32*GiB,64*GiB,128*GiB};
        ExpertCache cache(65536,[&]{return sample;});cache.refresh();
        MatrixKey a{1,0,0},b{1,0,1};
        test("cold_miss",cache.get(a,1024)==nullptr);
        auto * p=cache.admit(a,1024);
        test("allocated_and_hit",p && cache.get(a,1024)==p && cache.resident_bytes()==65536);
        cache.get(a,1024);cache.get(b,1024);
        test("frequency_rejects_cold",cache.admit(b,1024)==nullptr && cache.counters().rejected==1);
        for(int i=0;i<4;++i)cache.get(b,1024);
        test("eviction_replaces_colder",cache.admit(b,1024)!=nullptr && cache.counters().evictions==1 && cache.get(a,1024)==nullptr);
        cache.trim(0);test("trim_releases_all",cache.resident_bytes()==0 && cache.budget()==0);
        cache.refresh();cache.get(a,1024);cache.admit(a,1024);
        sample.gpu_free=sample.gpu_total/20;
        bool refused=false;try{cache.refresh();}catch(const std::runtime_error &){refused=true;}
        test("external_pressure_trims",cache.resident_bytes()==0 && cache.budget()==0 && !refused);
        sample.gpu_free=16*GiB;sample.ram_free=GiB;
        refused=false;try{cache.refresh();}catch(const std::runtime_error &){refused=true;}
        test("ram_pressure_refused",refused && cache.budget()==0);
        sample.ram_free=64*GiB;sample.gpu_total=0;
        refused=false;try{cache.refresh();}catch(const std::runtime_error &){refused=true;}
        test("invalid_sample_closed",refused && cache.budget()==0);
        sample.gpu_total=32*GiB;
        cache.refresh();cache.get(a,1024);cache.admit(a,1024);
        sample.gpu_free=sample.gpu_total;
        refused=false;try{cache.refresh();}catch(const std::runtime_error &){refused=true;}
        test("unaccounted_residency_refused",refused && cache.resident_bytes()==0 && cache.budget()==0);
        sample.gpu_free=16*GiB;
        ExpertCache oom(65536,[&]{return sample;},[](void **,size_t){return cudaErrorMemoryAllocation;});oom.refresh();
        test("allocation_oom_bypass",!oom.admit(a,1024) && oom.resident_bytes()==0 && oom.counters().oom==1);
        test("generation_identity",!(a==MatrixKey{2,0,0}) && MatrixHash{}(a)!=MatrixHash{}(MatrixKey{2,0,0}));
        cache.refresh();cache.get(a,1024);cache.admit(a,1024);
        auto pins=cache.protect({a});
        for(int i=0;i<8;++i)cache.get(b,1024);
        test("planned_hit_blocks_eviction",!cache.admit(b,1024) && cache.contains(a,1024));
        cache.trim(0);
        test("pinned_bytes_remain_charged",cache.resident_bytes()==65536 && cache.budget()==0);
        pins.reset();cache.trim(0);
        test("unpin_allows_trim",cache.resident_bytes()==0);
        report["status"]="pass";
    } catch(const std::exception & e) {report["error"]=e.what();}
    report["case_count"]=cases.size();report["cases"]=cases;
    if(argc==2){std::ofstream out(argv[1]);out<<report.dump(2)<<'\n';}
    std::cout<<report.dump(2)<<'\n';return report["status"]=="pass"?0:1;
}
