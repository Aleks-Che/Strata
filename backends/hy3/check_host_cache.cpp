// CPU-only native I/O, concurrency and memory-pressure checks.
#include "host_cache.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>
#include <fcntl.h>
#include <nlohmann/json.hpp>
using json=nlohmann::ordered_json;
using Cache=hy3::HostCache;
static int checks=0;
static void check(bool v,const char *what) {++checks;if(!v)throw std::runtime_error(what);}
int main(int argc,char **argv) {
    json report={{"status","error"}};
    std::filesystem::path output=argc==2?argv[1]:"host-cache-report.json";
    try {
        auto directory=output.parent_path();std::filesystem::create_directories(directory);
        auto path=directory/"source.bin";
        std::vector<uint8_t> original(2<<20);
        for(size_t i=0;i<original.size();++i)original[i]=uint8_t((i*17)^(i>>9));
        {std::ofstream file(path,std::ios::binary);file.write((char *)original.data(),original.size());}
        int fd=_wopen(path.c_str(),_O_RDONLY|_O_BINARY);
        check(fd>=0,"open source");
        auto a=std::make_shared<strata_expert_file::Source>((void *)0x10000,original.size(),fd);
        auto b=std::make_shared<strata_expert_file::Source>((void *)0x10000,original.size(),fd);
        _close(fd);
        std::atomic<bool> cancel{false};
        strata_expert_file::Request request;
        std::vector<uint8_t> dest(4096);
        std::atomic<bool> pressure{false};
        auto probe=[&] {return Cache::Memory{8ULL<<30,pressure.load()?0:6ULL<<30,16ULL<<30};};
        Cache cache(8192,probe,Cache::Policy::Lru);
        auto read=[&](auto source,size_t off) {
            bool ok=cache.read(source,off,dest.data(),dest.size(),cancel,request);
            check(ok && std::equal(dest.begin(),dest.end(),original.begin()+off),"bit-exact native/cached bytes");
        };
        read(a,0);read(a,0);
        check(cache.snapshot().hits==1 && cache.snapshot().file_bytes==4096,"warm read bypasses file");
        read(a,4096);read(a,0);read(a,8192);
        auto s=cache.snapshot();check(s.bytes==8192 && s.entries==2 && s.evictions==1,"bounded LRU eviction");
        check(s.allocations==2 && s.reuses==1,"eviction reuses committed allocation");
        read(a,0);check(cache.snapshot().hits==3,"recent entry survives");
        read(a,4096);check(cache.snapshot().misses==4,"old entry evicted");
        read(b,4096);check(cache.snapshot().misses==5,"new source identity cannot hit old mapping");
        pressure=true;cache.refresh();s=cache.snapshot();
        check(!s.bytes && !s.budget && !s.entries,"physical pressure reclaims cache");
        read(a,0);check(cache.snapshot().rejected==1 && !cache.snapshot().bytes,"pressure falls back to file");
        pressure=false;
        auto interrupted=[&](void *p,size_t n){std::memcpy(p,original.data(),n/2);return false;};
        Cache interrupted_cache(4096,probe,Cache::Policy::Lru);
        check(interrupted_cache.read(a,0,dest.data(),dest.size(),cancel,request),"prime reused cancel buffer");
        check(!interrupted_cache.read(a,4096,dest.data(),dest.size(),cancel,interrupted),"cancel recycled buffer fill");
        check(!interrupted_cache.snapshot().bytes && !interrupted_cache.snapshot().pending,"cancelled recycled entry unpublished");
        check(interrupted_cache.read(a,0,dest.data(),dest.size(),cancel,request) && dest[513]==original[513],"recycled cancel recovery exact");
        check(!cache.read(a,0,dest.data(),dest.size(),cancel,interrupted),"partial read cancelled");
        check(!cache.snapshot().bytes && !cache.snapshot().pending,"partial entry not published or reserved");
        bool caught=false;
        try {cache.read(a,0,dest.data(),dest.size(),cancel,[](void *,size_t)->bool {throw std::runtime_error("injected I/O");});}
        catch(const std::runtime_error &) {caught=true;}
        check(caught && !cache.snapshot().pending && !cache.snapshot().bytes,"I/O error drains reservation");
        read(a,0);cancel=true;
        check(!cache.read(a,0,dest.data(),dest.size(),cancel,request),"cancel before hit");cancel=false;
        cache.reset_counters();s=cache.snapshot();check(s.bytes==4096 && !s.hits && !s.misses,"reset preserves payload");
        caught=false;try {cache.read(a,original.size()-1,dest.data(),dest.size(),cancel,request);}
        catch(const std::runtime_error &) {caught=true;}check(caught,"range rejected");
        Cache concurrent(128<<10,probe);
        std::atomic<int> errors{0},completed{0};
        auto worker=[&](int id) {
            try {
                strata_expert_file::Request io;std::vector<uint8_t> bytes(4096);
                for(int i=0;i<600;++i) {
                    size_t off=((i*7+id*11)%128)*4096;
                    if(!concurrent.read(a,off,bytes.data(),bytes.size(),cancel,io) ||
                       !std::equal(bytes.begin(),bytes.end(),original.begin()+off))++errors;
                    ++completed;
                }
            } catch(...) {++errors;}
        };
        std::thread first(worker,0),second(worker,1);
        first.join();second.join();s=concurrent.snapshot();
        check(!errors && completed==1200,"two native readers retain exact data");
        check(s.bytes<=(128<<10) && !s.pending,"concurrent admission bounded and drained");
        report["concurrent_reads"]=completed.load();
        constexpr uint64_t GiB=1ULL<<30;
        check(Cache::allowance({100*GiB,20*GiB,40*GiB},30*GiB,100*GiB)==(42*GiB+(GiB>>1)),"physical budget excludes reserve");
        check(Cache::allowance({100*GiB,2*GiB,40*GiB},30*GiB,100*GiB)==(24*GiB+(GiB>>1)),"pressure budget shrinks owned bytes");
        check(Cache::allowance({100*GiB,20*GiB,GiB},30*GiB,100*GiB)==(30*GiB+(GiB>>1)),"commit budget limits growth");
        check(Cache::allowance({100*GiB,20*GiB,0},30*GiB,100*GiB)==(29*GiB+(GiB>>1)),"commit pressure reclaims payload");
        uint64_t available=20*GiB;
        Cache startup(SIZE_MAX,[&]{return Cache::Memory{100*GiB,available,100*GiB};});
        auto initial=startup.snapshot().budget;available=80*GiB;startup.refresh();
        check(startup.snapshot().budget==initial,"paging cannot inflate startup cap");
        Cache frequent(8192,probe,Cache::Policy::Frequency);
        auto freq_read=[&](size_t off) {
            check(frequent.read(a,off,dest.data(),dest.size(),cancel,request) &&
                std::equal(dest.begin(),dest.end(),original.begin()+off),"frequency cache exact bytes");
        };
        freq_read(0);s=frequent.snapshot();
        check(!s.bytes && s.frequency_bypasses==1,"one use does not retain payload");
        freq_read(0);for(int i=0;i<6;++i)freq_read(0);
        freq_read(4096);freq_read(4096);
        s=frequent.snapshot();check(s.entries==2 && s.allocations==2,"repeated demand admitted");
        check(s.reused_payload_bytes==4096,"payload reuse excludes admitted but unread entry");
        frequent.set_prefill(true);
        for(int i=0;i<12;++i)freq_read(size_t(2+i%6)*4096);
        freq_read(0);s=frequent.snapshot();
        check(s.history_entries==2 && s.prefill_bypasses==12 && s.evictions==0 && s.allocations==2,
            "prefill serves hits without polluting decode history or payload");
        frequent.set_prefill(false);
        for(int i=20;i<50;++i)freq_read(size_t(i)*4096);
        check(frequent.snapshot().evictions==0,"one-use scan cannot evict hot entries");
        freq_read(8192);freq_read(8192);
        check(frequent.snapshot().evictions==0,"tied newcomer does not churn residents");
        freq_read(8192);s=frequent.snapshot();
        check(s.evictions==1 && s.reuses==1 && s.allocations==2,"hotter newcomer replaces colder buffer");
        auto hits=s.hits;freq_read(0);freq_read(8192);
        check(frequent.snapshot().hits==hits+2,"hot residents remain usable");
        check(frequent.snapshot().reused_payload_bytes==8192,"payload reuse includes actual cache reads");
        frequent.reset_counters();s=frequent.snapshot();
        check(s.history_entries>0 && s.bytes==8192 && !s.frequency_bypasses,"counter reset retains learned demand");
        check(s.reused_payload_bytes==8192,"counter reset retains payload reuse gauge");
        pressure=true;frequent.refresh();s=frequent.snapshot();
        check(!s.bytes && !s.entries && !s.reused_payload_bytes,"frequency cache releases payload under pressure");pressure=false;
        Cache changing(4096,probe,Cache::Policy::Frequency,16);
        for(int i=0;i<30;++i)check(changing.read(a,0,dest.data(),dest.size(),cancel,request),"prime old hot workload");
        for(int i=0;i<20;++i)check(changing.read(a,4096,dest.data(),dest.size(),cancel,request),"changed workload exact I/O");
        s=changing.snapshot();hits=s.hits;
        check(changing.read(a,4096,dest.data(),dest.size(),cancel,request) && changing.snapshot().hits==hits+1 &&
            s.evictions>0,"decay allows fewer new uses to replace old popularity");
        Cache metadata(8192,probe,Cache::Policy::Frequency,131072,8);
        for(int i=0;i<24;++i)check(metadata.read(a,size_t(i)*4096,dest.data(),dest.size(),cancel,request),"bounded history reads");
        s=metadata.snapshot();check(s.history_entries<=8 && !s.bytes,"history bound independent of payload capacity");
        Cache failed(8192,probe);
        check(failed.read(a,0,dest.data(),dest.size(),cancel,request),"first bypass before failed promotion");
        check(!failed.read(a,0,dest.data(),dest.size(),cancel,interrupted) &&
            !failed.snapshot().bytes && !failed.snapshot().pending,"failed promotion never publishes partial bytes");
        check(failed.read(a,0,dest.data(),dest.size(),cancel,request) &&
            std::equal(dest.begin(),dest.end(),original.begin()),"promotion recovers after partial cancellation");
        Cache multi(8192,probe);
        for(size_t off:{0,4096})for(int i=0;i<2;++i)
            check(multi.read(a,off,dest.data(),dest.size(),cancel,request),"prime multi-victim cache");
        std::vector<uint8_t> large(8192);
        for(int i=0;i<2;++i)check(multi.read(a,8192,large.data(),large.size(),cancel,request),"large newcomer reads");
        check(multi.snapshot().bytes==8192 && !multi.snapshot().evictions,"rejected large candidate evicts nothing");
        check(multi.read(a,8192,large.data(),large.size(),cancel,request) &&
            std::equal(large.begin(),large.end(),original.begin()+8192) && multi.snapshot().evictions==2 &&
            multi.snapshot().entries==1,"hot large candidate reserves complete victim set");
        report["status"]="pass";
    } catch(const std::exception &e) {report["error"]=e.what();}
    report["checks"]=checks;std::ofstream(output)<<report.dump(2)<<'\n';
    std::cout<<report.dump(2)<<'\n';return report["status"]=="pass"?0:1;
}
