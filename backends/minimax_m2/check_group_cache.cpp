#include "group_cache.hpp"
#include "gpu_arena.hpp"
#include "nlohmann/json.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
using json=nlohmann::ordered_json;
using step35::cuda_check;
using Key=step35::MatrixKey;
static constexpr size_t MiB=1u<<20,cap=128*MiB;
struct Fixture {
    size_t budget=cap;
    std::array<size_t,3> strides;
    minimax_m2::GpuArena arena{cap,[] {
        const auto m=step35::memory_sample();MEMORYSTATUSEX ram{};ram.dwLength=sizeof(ram);
        if(!GlobalMemoryStatusEx(&ram))throw std::runtime_error("commit sample unavailable");
        return minimax_m2::GpuArena::Memory{m.gpu_free,m.gpu_total,size_t(ram.ullAvailPageFile)};
    }};
    minimax_m2::ExpertCache cache{cap,[this] {
        auto m=step35::memory_sample();const size_t reserve=((m.gpu_total/20+MiB-1)/MiB+256)*MiB;
        m.gpu_free=std::min(m.gpu_free,reserve+budget-arena.snapshot().reserved);return m;
    },[this](void **p,size_t n){return arena.allocate(p,n);},[this](void *p){return arena.release(p);}};
    explicit Fixture(bool mixed):strides{2654208,2654208,mixed?size_t(3870720):size_t(2654208)} {
        cache.set_backing_bytes([this]{return arena.snapshot().reserved;});
        cache.set_backing_groups([this](const void *p){return arena.allocation_group(p);});
        cache.set_reuse_allocations(true);cache.set_match_size(true);cache.set_fast_scan(true);
        cache.set_groups(3,{{{0,1,2},strides,256}});cache.refresh();
    }
    size_t bytes(unsigned e,unsigned role) const {return strides[role]+(e==255?0:512);}
    static Key key(unsigned e,unsigned role,uint64_t generation=1) {return {generation,role,e};}
    static unsigned char pattern(unsigned e,unsigned role) {return uint8_t((e*3+role)%251+1);}
    void *fill(unsigned e,unsigned role,uint64_t generation=1) {
        const auto k=key(e,role,generation);const auto n=bytes(e,role);
        if(cache.get(k,n,false))throw std::runtime_error("fixture expected a missing matrix");
        void *p=cache.admit(k,n);if(!p)throw std::runtime_error("group fixture admission refused");
        if(cache.contains(k,n))throw std::runtime_error("uncommitted bytes became visible");
        auto pin=cache.protect({k});cuda_check(cudaMemset(p,pattern(e,role),n));cuda_check(cudaDeviceSynchronize());
        cache.commit(k,n);return p;
    }
    void fill_group(unsigned e,uint64_t generation=1) {for(unsigned r=0;r<3;++r)fill(e,r,generation);}
    bool exact(unsigned count=256,uint64_t generation=1) {
        std::vector<unsigned char> values;
        for(unsigned e=0;e<count;++e)for(unsigned r=0;r<3;++r)if(void *p=cache.get(key(e,r,generation),bytes(e,r),false)) {
            values.resize(bytes(e,r));cuda_check(cudaMemcpy(values.data(),p,values.size(),cudaMemcpyDeviceToHost));
            if(!std::all_of(values.begin(),values.end(),[e,r](unsigned char v){return v==pattern(e,r);}))return false;
        }
        return true;
    }
};
int main(int argc,char **argv) {
    json report={{"pass",false},{"tests",json::array()},{"measurements",json::array()}};bool writable=false;
    auto check=[&](const std::string &name,bool ok) {
        report["tests"].push_back({{"name",name},{"pass",ok}});if(!ok)throw std::runtime_error(name);
    };
    try {
        if(argc!=2 || std::filesystem::exists(argv[1]))throw std::runtime_error("usage: group-cache-check NEW_REPORT.json");
        writable=true;cuda_check(cudaSetDevice(0));
        for(bool mixed:{false,true}) {
            Fixture f(mixed);const unsigned count=mixed?14:16;
            for(unsigned e=0;e<count;++e)f.fill_group(e);
            const auto g=f.cache.group_stats();const auto a=f.arena.snapshot();
            check("complete_groups_and_ready_bytes",g.groups==count && !g.partial && !g.pending_matrices && g.ready_matrices==3*count && f.exact(count));
            check("two_dense_blocks",a.reserved==cap && a.live==f.cache.resident_bytes() && g.ready_bytes<=a.live);
            report["measurements"].push_back({{"mixed",mixed},{"groups",g.groups},{"physical_bytes",a.reserved},{"charged_bytes",a.live},{"ready_bytes",g.ready_bytes}});
            auto pin=f.cache.protect({Fixture::key(0,2)});
            auto *victim=f.cache.get(Fixture::key(1,0),f.bytes(1,0),false);
            void *next=f.fill(count,0);
            check("group_reuses_one_allocation",next==victim && f.cache.counters().reuses>0);
            bool gone=true,protected_group=true;
            for(unsigned r=0;r<3;++r) {gone&=!f.cache.contains(Fixture::key(1,r),f.bytes(1,r));protected_group&=f.cache.contains(Fixture::key(0,r),f.bytes(0,r));}
            check("replacement_removes_all_three_matrices",gone);
            check("one_matrix_pin_protects_all_three",protected_group);
            check("reused_group_has_no_stale_ready_bits",!f.cache.contains(Fixture::key(count,1),f.bytes(count,1)) && f.cache.group_stats().partial==1);
            f.fill(count,1);f.fill(count,2);check("complete_reused_group_exact",!f.cache.group_stats().partial && f.exact(count+1));
            f.budget=cap-MiB;f.cache.refresh();
            check("pressure_releases_whole_unpinned_block",f.arena.snapshot().reserved==64*MiB && f.cache.group_stats().groups==count/2 &&
                f.cache.contains(Fixture::key(0,2),f.bytes(0,2)) && f.exact(count+1));
            f.budget=0;bool refused=false;try{f.cache.refresh();}catch(const std::exception &e){refused=std::string(e.what()).find("pinned cache")!=std::string::npos;}
            check("pinned_pressure_refused",refused && f.cache.contains(Fixture::key(0,0),f.bytes(0,0)));
            pin.reset();f.cache.refresh();check("drained_zero_budget_frees_all_groups",!f.arena.snapshot().reserved && !f.cache.group_stats().groups);
            f.budget=cap;f.cache.refresh();f.fill_group(255,2);
            check("new_generation_and_last_expert_guard",f.exact(256,2) && !f.cache.contains(Fixture::key(255,0,1),f.bytes(255,0)));
            bool wrong=false;try{f.cache.get(Fixture::key(255,1,2),f.bytes(255,1)+512);}catch(const std::exception &){wrong=true;}
            check("wrong_guard_size_rejected",wrong);
        }
        {
            Fixture f(true);const auto key=Fixture::key(3,0);const auto bytes=f.bytes(3,0);
            void *p=f.cache.admit(key,bytes);check("pending_allocation",p!=nullptr);
            auto pin=f.cache.protect({Fixture::key(3,2)}); // Down is not ready yet.
            cudaStream_t stream;cuda_check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
            cuda_check(cudaMemsetAsync(p,77,bytes,stream));f.budget=0;
            bool failed=false;try{f.cache.refresh();}catch(const std::exception &){failed=true;}
            const auto s=f.cache.group_stats();check("pending_group_pin_survives_pressure",failed && s.pending_matrices==1 && !s.ready_matrices && !f.cache.get(key,bytes,false));
            cuda_check(cudaStreamSynchronize(stream));cuda_check(cudaStreamDestroy(stream));pin.reset();f.cache.trim(0);
            check("aborted_unpublished_fill_discards_state",!f.cache.group_stats().pending_matrices && !f.cache.size() && !f.arena.snapshot().reserved);
            f.budget=cap;f.cache.refresh();f.fill_group(3);check("failed_fill_recovery_exact",f.exact(4));
        }
        {
            Fixture f(false);std::vector<Key> keys;
            for(unsigned e=0;e<16;++e){f.fill_group(e);keys.push_back(Fixture::key(e,0));}
            auto pins=f.cache.protect(keys);f.cache.begin_plan();
            f.cache.get(Fixture::key(20,0),f.bytes(20,0));
            check("first_matrix_admission_refused_while_pinned",!f.cache.admit(Fixture::key(20,0),f.bytes(20,0)));
            pins.reset();f.cache.get(Fixture::key(20,1),f.bytes(20,1));
            check("declined_group_cannot_enter_at_later_matrix",!f.cache.admit(Fixture::key(20,1),f.bytes(20,1)) && !f.cache.group_stats().partial);
            f.cache.end_plan();f.cache.begin_plan();f.cache.get(Fixture::key(20,0),f.bytes(20,0));f.fill_group(20);f.cache.end_plan();
            check("next_route_can_retry_whole_group",f.cache.contains(Fixture::key(20,2),f.bytes(20,2)) && !f.cache.group_stats().partial && f.exact(21));
        }
        report["pass"]=true;report["cases"]=report["tests"].size();
        std::ofstream(argv[1])<<report.dump(2)<<'\n';std::cout<<report.dump(2)<<'\n';return 0;
    }catch(const std::exception &e) {
        report["error"]=e.what();if(writable)std::ofstream(argv[1])<<report.dump(2)<<'\n';std::cerr<<e.what()<<'\n';return 1;
    }
}
