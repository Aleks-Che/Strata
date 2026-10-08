#include "gpu_arena.hpp"
#include "../step35/expert_cache.hpp"
#include "nlohmann/json.hpp"
#include <fstream>
#include <iostream>
using json=nlohmann::ordered_json;
using step35::cuda_check;
static constexpr size_t MiB=1u<<20,slot=4*MiB,cap=256*MiB;
static step35::MatrixKey key(unsigned i) {return {1,0,i};}
struct Fixture {
    size_t budget=cap;
    size_t block=64*MiB;
    size_t growth_reserve=0;
    minimax_m2::GpuArena arena{cap,[this] {
        const auto m=step35::memory_sample();MEMORYSTATUSEX ram{};ram.dwLength=sizeof(ram);
        if(!GlobalMemoryStatusEx(&ram))throw std::runtime_error("commit sample unavailable");
        const size_t room=budget-std::min(budget,arena.snapshot().reserved);
        return minimax_m2::GpuArena::Memory{std::min(m.gpu_free,m.gpu_total/20+256*MiB+room),m.gpu_total,size_t(ram.ullAvailPageFile)};
    },block,growth_reserve};
    step35::ExpertCache cache{cap,[this] {
        const size_t total=size_t(32)<<30;
        const size_t reserve=((total/20+MiB-1)/MiB+256)*MiB;
        return step35::MemorySample{reserve+budget-arena.snapshot().reserved,total,size_t(64)<<30,size_t(128)<<30};
    },[this](void **p,size_t n){return arena.allocate(p,n);},[this](void *p){return arena.release(p);}};
    explicit Fixture(bool grouped,unsigned count=64,size_t block_bytes=64*MiB,size_t reserve=0):block(block_bytes),growth_reserve(reserve) {
        cache.set_backing_bytes([this]{return arena.snapshot().reserved;});
        cache.set_reuse_allocations(true);cache.set_match_size(true);
        if(grouped)cache.set_backing_groups([this](const void *p){return arena.allocation_group(p);});
        cache.refresh();
        for(unsigned i=0;i<count;++i) {
            void *p=cache.admit(key(i),slot);
            if(!p)throw std::runtime_error("fixture allocation refused");
            cuda_check(cudaMemset(p,int(i+1),slot));
        }
        cuda_check(cudaDeviceSynchronize());
        // Interleave age across four physical slabs: entry-wise LRU creates
        // holes everywhere before the first allocation can return to CUDA.
        for(unsigned i=0;i<16;++i)for(unsigned s=0;s<4;++s)
            if(s*16+i<count)cache.get(key(s*16+i),slot);
        cache.reset_counters();
    }
    bool exact() {
        std::vector<unsigned char> bytes(slot);
        for(unsigned i=0;i<64;++i)if(void *p=cache.get(key(i),slot,false)) {
            cuda_check(cudaMemcpy(bytes.data(),p,slot,cudaMemcpyDeviceToHost));
            if(!std::all_of(bytes.begin(),bytes.end(),[i](unsigned char b){return b==i+1;}))return false;
        }
        return true;
    }
    // MM27-12 pressure algorithm retained verbatim in this test for comparison.
    void legacy_refresh() {
        cache.refresh();const auto limit=cache.budget();
        while(arena.snapshot().reserved>limit) {
            const auto s=arena.snapshot();const auto excess=s.reserved-limit;
            const size_t remove=std::max<size_t>(excess,64u<<20);
            cache.trim(s.live>remove?s.live-remove:0);
            if(arena.snapshot().live==s.live && arena.snapshot().reserved>limit)
                throw std::runtime_error("legacy pinned cache cannot shrink");
            if(arena.snapshot().reserved>limit && !cache.resident_bytes())
                throw std::runtime_error("legacy retained empty backing");
        }
    }
};
int main(int argc,char **argv) {
    json report={{"pass",false},{"tests",json::array()},{"measurements",json::array()}};bool writable=false;
    auto check=[&](const char *name,bool pass) {
        report["tests"].push_back({{"name",name},{"pass",pass}});
        if(!pass)throw std::runtime_error(name);
    };
    try {
        if(argc!=2)throw std::runtime_error("usage: reclaim-check NEW_REPORT.json");
        std::ifstream previous(argv[1]);if(previous.good())throw std::runtime_error("report already exists");
        writable=true;cuda_check(cudaSetDevice(0));
        for(bool grouped:{false,true}) {
            Fixture f(grouped);f.budget=255*MiB;
            if(grouped)f.cache.refresh();else f.legacy_refresh();
            const auto a=f.arena.snapshot();const auto c=f.cache.counters();
            report["measurements"].push_back({{"policy",grouped?"whole_blocks":"legacy_entries"},
                {"budget",f.budget},{"before_live",cap},{"after_live",a.live},{"after_reserved",a.reserved},
                {"evictions",c.evictions},{"pressure_groups",c.pressure_groups},
                {"evicted_bytes",cap-a.live},{"released_bytes",cap-a.reserved}});
            check(grouped?"grouped_drops_one_block":"legacy_scattered_eviction_reproduced",
                a.reserved==(grouped?192:0)*MiB && a.live==(grouped?192:0)*MiB && c.evictions==(grouped?16:64));
            check("remaining_full_payload_exact",f.exact());
            if(grouped) {
                check("sampled_budget_not_lowered_by_reclaim",f.cache.budget()==255*MiB);
                check("pressure_accounting",c.pressure_trims==1 && c.pressure_groups==1 &&
                    c.pressure_evicted_bytes==64*MiB && c.pressure_released_bytes==64*MiB);
                f.budget=cap;f.cache.refresh();
                for(unsigned i=0;i<64;++i)if(!f.cache.contains(key(i),slot)) {
                    void *p=f.cache.admit(key(i),slot);if(!p)throw std::runtime_error("recovery admission failed");
                    cuda_check(cudaMemset(p,int(i+1),slot));
                }
                check("pressure_recovery_refills_exact",f.cache.resident_bytes()==cap && f.exact());
            }
        }
        {
            Fixture f(true);auto pins=f.cache.protect({key(0)});f.budget=255*MiB;f.cache.refresh();
            bool intact=true;for(unsigned i=0;i<16;++i)intact&=f.cache.contains(key(i),slot);
            check("one_pin_preserves_whole_block",intact && f.arena.snapshot().live==192*MiB && f.exact());
        }
        {
            Fixture f(true);auto pins=f.cache.protect({key(0),key(16),key(32),key(48)});f.budget=255*MiB;
            bool failed=false;try{f.cache.refresh();}catch(const std::runtime_error &e){failed=std::string(e.what()).find("pinned cache")!=std::string::npos;}
            check("all_pinned_fail_without_partial_eviction",failed && f.cache.resident_bytes()==cap && !f.cache.counters().evictions && f.exact());
            pins.reset();f.cache.refresh();check("unpin_retries_pressure",f.arena.snapshot().reserved==192*MiB);
        }
        {
            Fixture f(true,56);f.budget=255*MiB;f.cache.refresh();
            bool dense=true;for(unsigned i=0;i<48;++i)dense&=f.cache.contains(key(i),slot);
            check("least_live_block_reclaimed_first",dense && f.cache.resident_bytes()==192*MiB &&
                f.cache.counters().pressure_evicted_bytes==32*MiB && f.exact());
        }
        {
            // After 64 admissions growth=64 MiB: this refresh is triggered
            // inside admit(), not by the public MiniMax memory guard.
            Fixture f(true);f.budget=255*MiB;
            f.cache.get(key(64),slot);f.cache.get(key(64),slot);
            void *p=f.cache.admit(key(64),slot);
            check("admission_refresh_enforces_physical_budget",p && f.cache.counters().pressure_groups==1 &&
                f.arena.snapshot().reserved==192*MiB && f.cache.resident_bytes()==192*MiB && f.exact());
        }
        {
            Fixture f(true);f.budget=0;f.cache.refresh();
            check("zero_budget_returns_all_backing",!f.arena.snapshot().reserved && !f.cache.resident_bytes());
        }
        for(size_t block:{8*MiB,16*MiB,32*MiB,64*MiB}) {
            Fixture f(true,64,block);f.budget=cap-MiB;f.cache.refresh();
            check("configurable_block_pressure_exact",f.arena.snapshot().live==cap-block &&
                f.arena.snapshot().reserved==cap-block && f.cache.counters().pressure_released_bytes==block && f.exact());
            report["measurements"].push_back({{"block_mib",block/MiB},{"pressure_mib",1},
                {"evicted_mib",(cap-f.cache.resident_bytes())/MiB},{"remaining_mib",f.cache.resident_bytes()/MiB}});
        }
        for(size_t reserve:{size_t(0),32*MiB}) {
            const unsigned count=unsigned((cap-reserve)/slot);
            Fixture f(true,count,8*MiB,reserve);f.arena.reset_counters();
            bool exact=true;
            for(unsigned cycle=0;cycle<20;++cycle) {
                f.budget=cap-MiB;f.cache.refresh();exact&=f.exact();
                f.budget=cap;f.cache.refresh();
                for(unsigned i=0;i<count;++i)if(!f.cache.contains(key(i),slot)) {
                    void *p=f.cache.admit(key(i),slot);
                    if(!p)throw std::runtime_error("oscillation refill refused");
                    cuda_check(cudaMemset(p,int(i+1),slot));
                }
                exact&=f.exact();
            }
            const auto a=f.arena.snapshot();const auto c=f.cache.counters();
            check(reserve?"growth_band_avoids_oscillation":"zero_band_reproduces_oscillation",
                a.allocations==(reserve?0:20) && a.frees==(reserve?0:20) &&
                c.pressure_groups==(reserve?0:20) && a.reserved==cap-reserve && exact);
            check("growth_band_does_not_lower_shrink_budget",f.cache.budget()==cap);
            report["measurements"].push_back({{"growth_reserve_mib",reserve/MiB},{"block_mib",8},
                {"budget_low_mib",255},{"budget_high_mib",256},{"cycles",20},
                {"allocations",a.allocations},{"frees",a.frees},{"live_mib",a.live/MiB},{"exact",exact}});
            // A larger global budget allows growth all the way to the cache
            // cap: the band is not subtracted from that independent hard cap.
            f.budget=cap+reserve;f.cache.refresh();
            for(unsigned i=count;i<64;++i) {
                void *p=f.cache.admit(key(i),slot);if(!p)throw std::runtime_error("band recovery refused");
                cuda_check(cudaMemset(p,int(i+1),slot));
            }
            check("growth_recovers_to_full_cap",f.arena.snapshot().reserved==cap && f.exact());
            f.budget=128*MiB;f.cache.refresh();
            check("growth_band_still_shrinks_under_pressure",f.arena.snapshot().reserved==128*MiB && f.exact());
        }
        report["pass"]=true;
    }catch(const std::exception &e){report["error"]=e.what();std::cerr<<e.what()<<'\n';}
    report["cases"]=report["tests"].size();
    if(writable){std::ofstream out(argv[1]);out<<report.dump(2)<<'\n';}
    std::cout<<report.dump(2)<<'\n';return report["pass"].get<bool>()?0:1;
}
