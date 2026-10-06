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
        size_t allocation_calls=0;
        ExpertCache reuse(65536,[&]{return sample;},[&](void ** ptr,size_t bytes){
            ++allocation_calls;
            return allocation_calls==1 ? cudaMalloc(ptr,bytes) : cudaErrorMemoryAllocation;
        });
        reuse.set_reuse_allocations(true);reuse.refresh();
        reuse.get(a,1024);auto * original=reuse.admit(a,1024);
        cuda_check(cudaMemset(original,0x31,1024));cuda_check(cudaDeviceSynchronize());
        reuse.get(b,1536);auto * replacement=reuse.admit(b,1536);
        test("reuse_equal_allocation_without_allocator",replacement==original && allocation_calls==1 &&
             reuse.counters().reuses==1 && reuse.counters().allocations==1 && reuse.resident_bytes()==65536);
        test("reuse_removes_old_identity",!reuse.contains(a,1024) && reuse.contains(b,1536));
        cuda_check(cudaMemset(replacement,0x7a,1536));
        std::vector<uint8_t> bytes(1536);
        cuda_check(cudaMemcpy(bytes.data(),replacement,bytes.size(),cudaMemcpyDeviceToHost));
        test("reuse_filled_bytes",std::all_of(bytes.begin(),bytes.end(),[](uint8_t v){return v==0x7a;}));
        auto reuse_pins=reuse.protect({b});
        for(int i=0;i<4;++i)reuse.get(a,1024);
        test("reuse_respects_plan_pins",!reuse.admit(a,1024) && reuse.get(b,1536)==replacement);
        reuse_pins.reset();
        test("reuse_after_unpin",reuse.admit(a,1024)==replacement && reuse.counters().reuses==2);
        refused=false;try{reuse.admit(a,1024);}catch(const std::runtime_error &){refused=true;}
        test("duplicate_admission_preserves_resident",refused && reuse.get(a,1024)==replacement && reuse.resident_bytes()==65536);
        sample.gpu_free=sample.gpu_total/20;reuse.refresh();
        test("reuse_pressure_releases_all",reuse.resident_bytes()==0 && reuse.budget()==0);
        sample.gpu_free=16*GiB;
        ExpertCache mixed(131072,[&]{return sample;});mixed.set_reuse_allocations(true);mixed.refresh();
        mixed.get(a,1024);mixed.admit(a,1024);mixed.get(b,1024);mixed.admit(b,1024);
        MatrixKey large{1,1,0};mixed.get(large,70000);
        test("reuse_different_size_allocates",mixed.admit(large,70000) && mixed.counters().reuses==0 &&
             mixed.counters().allocations==3 && mixed.counters().evictions==2 && mixed.resident_bytes()==131072);
        mixed.trim(0);test("reuse_trim_has_no_hidden_pool",mixed.resident_bytes()==0 && mixed.size()==0);
        {
            ExpertCache baseline(32*65536,[&]{return sample;}), direct(32*65536,[&]{return sample;});
            baseline.set_reuse_allocations(true);direct.set_reuse_allocations(true);
            direct.set_fast_scan(true);baseline.refresh();direct.refresh();
            uint32_t random=0x5a17beef;
            auto key=[](unsigned i){return MatrixKey{7,i/16,i%16};};
            auto size=[](unsigned i)->size_t{return i%5==0?70000:1024;};
            // Cross two decay boundaries, mix allocation sizes, revisit pinned
            // entries, trim, and switch modes on the same live cache. Compare
            // each decision and full membership, not only aggregate hit rate.
            bool parity=true;
            for (unsigned block=0;block<2050;++block) {
                std::vector<MatrixKey> held;
                for(unsigned i=0;i<8;++i)held.push_back(key((block+i)%64));
                auto pa=baseline.protect(held),pb=direct.protect(held);
                for(unsigned j=0;j<64;++j) {
                    random^=random<<13;random^=random>>17;random^=random<<5;
                    const unsigned i=(block%4==0)?block%64:random%64;
                    const bool a_hit=bool(baseline.get(key(i),size(i))), b_hit=bool(direct.get(key(i),size(i)));
                    parity &= a_hit==b_hit;
                    if(!a_hit && !b_hit) {
                        parity &= bool(baseline.admit(key(i),size(i)))==bool(direct.admit(key(i),size(i)));
                        for(unsigned k=0;k<64;++k)parity &= baseline.contains(key(k),size(k))==direct.contains(key(k),size(k));
                    }
                    if(!parity)throw std::runtime_error("direct victim policy differs");
                }
                if(block%127==0) {baseline.trim(0);direct.trim(0);}
                pa.reset();pb.reset();
                if(block%127==0) {baseline.refresh();direct.refresh();}
                direct.set_fast_scan(block%5!=0);
            }
            auto a=baseline.counters(),b=direct.counters();
            test("direct_scan_131200_decisions_membership_parity",parity && a.hits==b.hits && a.misses==b.misses &&
                a.evictions==b.evictions && a.rejected==b.rejected && a.bypasses==b.bypasses && a.samples==b.samples &&
                a.allocations==b.allocations && a.reuses==b.reuses && baseline.resident_bytes()==direct.resident_bytes());
        }
        {
            ExpertCache zeros(64*65536,[&]{return sample;});zeros.set_reuse_allocations(true);
            zeros.set_fast_scan(true);zeros.set_profile(true);zeros.refresh();
            for(unsigned i=0;i<64;++i) {MatrixKey k{9,0,i};zeros.get(k,1024);zeros.admit(k,1024);}
            for(unsigned i=0;i<65536;++i)zeros.get({9,0,63},1024);
            zeros.reset_counters();zeros.get({9,1,0},1024);zeros.admit({9,1,0},1024);
            test("direct_scan_first_zero_is_oldest_minimum",zeros.counters().victim_candidates==1 &&
                !zeros.contains({9,0,0},1024) && zeros.contains({9,0,1},1024) && zeros.contains({9,1,0},1024));
        }
        report["status"]="pass";
    } catch(const std::exception & e) {report["error"]=e.what();}
    report["case_count"]=cases.size();report["cases"]=cases;
    if(argc==2){std::ofstream out(argv[1]);out<<report.dump(2)<<'\n';}
    std::cout<<report.dump(2)<<'\n';return report["status"]=="pass"?0:1;
}
