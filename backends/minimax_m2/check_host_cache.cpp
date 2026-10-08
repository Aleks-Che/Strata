#include "host_cache.hpp"
#include "nlohmann/json.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <fcntl.h>
#include <thread>
using json=nlohmann::ordered_json;
using Cache=minimax_m2::HostCache;
static constexpr size_t MiB=1u<<20;
int main(int argc,char **argv) {
    json tests=json::array(),report;std::filesystem::path dir;bool created=false;
    try {
        if(argc!=2 || std::filesystem::exists(argv[1]))throw std::runtime_error("supply a new output directory");
        dir=argv[1];std::filesystem::create_directories(dir);created=true;
        auto check=[&](const char *name,bool ok){tests.push_back({{"name",name},{"pass",ok}});if(!ok)throw std::runtime_error(name);};
        std::vector<uint8_t> expected(8*MiB);
        for(size_t i=0;i<expected.size();++i)expected[i]=uint8_t((i*7+i/4096)%251);
        const auto path=dir/"source.bin";
        {std::ofstream out(path,std::ios::binary);out.write(reinterpret_cast<char *>(expected.data()),expected.size());}
        const int fd=_wopen(path.wstring().c_str(),_O_RDONLY|_O_BINARY);
        if(fd<0)throw std::runtime_error("cannot open test file");
        // read_at uses file offsets, so no original view needs to be faulted in.
        auto source=std::make_shared<strata_expert_file::Source>(reinterpret_cast<void *>(0x100000),expected.size(),fd);
        auto other=std::make_shared<strata_expert_file::Source>(reinterpret_cast<void *>(0x2000000),expected.size(),fd);_close(fd);
        Cache::Memory memory{1ull<<30,1ull<<30,1ull<<30};Cache cache(4*MiB,[&]{return memory;});
        std::atomic<bool> cancel{false};strata_expert_file::Request request;
        auto read=[&](uint64_t offset,size_t bytes) {
            std::vector<uint8_t> data(bytes);
            return cache.read(source,offset,data.data(),bytes,cancel,request) &&
                std::equal(data.begin(),data.end(),expected.begin()+offset);
        };
        check("prefill_bytes_exact",read(65530,MiB+512));
        check("prefill_does_not_admit",!cache.snapshot().bytes && cache.snapshot().prefill_bypasses==1);
        cache.set_prefill(false);
        check("unaligned_map_and_guard_exact",read(65530,MiB+512));
        const auto cold=cache.snapshot();
        check("page_and_granularity_charge",cold.bytes==1118208 && cold.admissions==1 && !cold.readers);
        check("warm_bytes_exact",read(65530,MiB+512));
        check("warm_view_reuse",cache.snapshot().hits==1 && cache.snapshot().admissions==1);
        cache.drop_gpu(source,65530,MiB+512);
        check("gpu_commit_removes_view",!cache.snapshot().bytes && cache.snapshot().gpu_drops==1 && cache.snapshot().gpu_drop_bytes==cold.bytes);
        check("gpu_eviction_can_refill",read(65530,MiB+512));
        check("last_file_bytes_exact",read(expected.size()-65535,65535));
        bool refused=false;try{read(expected.size()-1,2);}catch(const std::exception &){refused=true;}
        check("out_of_file_rejected",refused);
        const auto before=cache.snapshot();cancel.store(true);
        check("cancelled_read_does_not_publish",!read(3*MiB,MiB) && cache.snapshot().bytes==before.bytes);cancel.store(false);
        cache.clear();check("clear_releases_all_views",!cache.snapshot().bytes && !cache.snapshot().entries);
        for(size_t i=0;i<6;++i)check("lru_bytes_and_cap",read(i*MiB,MiB) && cache.snapshot().bytes<=4*MiB);
        const auto evicted=cache.snapshot();check("lru_replacement",evicted.entries==4 && evicted.evictions>=2);
        const auto hits=evicted.hits;check("lru_recent_hit",read(5*MiB,MiB) && cache.snapshot().hits==hits+1);
        std::vector<uint8_t> data(MiB);
        check("separate_source_identity",cache.read(other,5*MiB,data.data(),data.size(),cancel,request) &&
            cache.snapshot().hits==hits+1 && std::equal(data.begin(),data.end(),expected.begin()+5*MiB));
        memory.available=0;cache.refresh();check("physical_pressure_drops_views",!cache.snapshot().bytes && !cache.snapshot().budget);
        check("pressure_bypass_exact",read(0,MiB) && !cache.snapshot().bytes);
        memory.available=memory.total;memory.commit_available=0;cache.refresh();
        check("commit_pressure_disables_admission",read(0,MiB) && !cache.snapshot().bytes);
        memory.commit_available=memory.total;cache.refresh();check("pressure_recovery",read(0,MiB) && cache.snapshot().bytes==MiB);
        std::atomic<bool> exact{true};
        auto reader=[&](int salt) {
            try {
                strata_expert_file::Request io;std::vector<uint8_t> values(128*1024+17);
                for(int i=0;i<100;++i) {
                    const size_t at=size_t((i+salt)%40)*65536+31;
                    if(!cache.read(source,at,values.data(),values.size(),cancel,io) ||
                       !std::equal(values.begin(),values.end(),expected.begin()+at))exact.store(false);
                }
            }catch(...) {exact.store(false);}
        };
        std::thread a(reader,1),b(reader,7);
        for(int i=0;i<100;++i) {
            cache.drop_gpu(source,size_t(i%40)*65536,256*1024);
            if(i%7==0)cache.clear();
        }
        a.join();b.join();
        check("concurrent_read_drop_clear_exact",exact.load());
        check("concurrent_views_drained_and_bounded",!cache.snapshot().readers && cache.snapshot().bytes<=cache.snapshot().budget);
        cache.clear();cache.reset_counters();
        check("reset_empty_accounting",!cache.snapshot().bytes && !cache.snapshot().admissions && !cache.snapshot().gpu_drops);
        check("final_refill_exact",read(1,65537));
        cache.clear();cache.reset_counters();cache.set_gpu_partition(true);
        check("GPU_gets_first_admission_chance",read(MiB-17,MiB) && !cache.snapshot().bytes && cache.snapshot().gpu_waits==1);
        cache.gpu_bypass(source,MiB-17,2*MiB+512);
        check("prior_GPU_bypass_allows_RAM_map",read(MiB-17,MiB) && cache.snapshot().bytes>0);
        check("split_matrix_chunk_eligible",read(2*MiB-17,MiB+512) && cache.snapshot().entries==2);
        cache.drop_gpu(source,MiB-17,2*MiB+512);
        check("GPU_commit_clears_views_and_history",!cache.snapshot().bytes && !cache.snapshot().history_entries);
        check("GPU_commit_resets_RAM_eligibility",read(MiB-17,MiB) && !cache.snapshot().bytes);
        cache.set_gpu_partition(false);
        check("RAM_only_mode_can_admit",read(MiB-17,MiB) && cache.snapshot().bytes>0);
        const auto s=cache.snapshot();report["final"]={{"bytes",s.bytes},{"mapped_bytes",s.mapped_bytes},{"file_bytes",s.file_bytes}};
        report["pass"]=true;
    }catch(const std::exception &e) {report["pass"]=false;report["error"]=e.what();std::cerr<<e.what()<<'\n';}
    report["tests"]=tests;report["cases"]=tests.size();
    if(created)std::ofstream(dir/"host-cache-report.json")<<report.dump(2)<<'\n';
    std::cout<<report.dump(2)<<'\n';return report.value("pass",false)?0:1;
}
