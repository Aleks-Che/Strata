// CPU-only native I/O, concurrency and memory-pressure checks.
#include "host_cache.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>
#include <fcntl.h>
#include <nlohmann/json.hpp>
using json=nlohmann::ordered_json;
using Cache=strata_ds4::HostCache;
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
        Cache per_job(4096,probe);
        auto loader=[&](void *p,size_t n){return request.read_at(*a,0,p,n,cancel);};
        for(int i=0;i<3;++i)check(per_job.read(a,0,dest.data(),dest.size(),cancel,loader,true),"prefill job read");
        check(!per_job.snapshot().bytes && !per_job.snapshot().history_entries,"per-job prefill does not train");
        for(int i=0;i<2;++i)check(per_job.read(a,0,dest.data(),dest.size(),cancel,loader,false),"decode job read");
        auto per_hits=per_job.snapshot().hits;
        check(per_job.read(a,0,dest.data(),dest.size(),cancel,loader,true) &&
            per_job.snapshot().hits==per_hits+1,"prefill job serves hot decode entry");
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
        // Ready leases share the pipeline payload, survive pressure/eviction,
        // and accept its padding without exposing an undersized matrix.
        int allocated=0,released=0;
        Cache::Storage storage;
        storage.allocate=[&](size_t n){++allocated;return new uint8_t[n];};
        storage.release=[&](uint8_t *p){++released;delete[] p;};
        Cache::Lease held;
        {
            Cache leased(8192,probe,Cache::Policy::Lru,131072,131072,storage);
            check(!leased.acquire(a,0,4096),"lease miss performs no I/O");
            check(leased.read(a,0,dest.data(),4096,cancel,request),"publish lease payload");
            held=leased.acquire(a,0,3584);
            check(bool(held) && !std::memcmp(held.data(),original.data(),3584),"padded block lease exact");
            auto same=leased.acquire(a,0,4096);
            check(same.data()==held.data(),"lease shares allocation without a copy");
            check(!leased.acquire(a,0,4097) && !leased.acquire(b,0,4096),"lease rejects partial block and wrong source");
            check(leased.read(a,4096,dest.data(),4096,cancel,request),"second lease block");
            auto second=leased.acquire(a,4096,4096);
            auto evictions=leased.snapshot().evictions;
            check(leased.read(a,8192,dest.data(),4096,cancel,request),"full leased cache falls back to file");
            check(leased.snapshot().evictions==evictions && allocated==2,"held allocations neither reused nor expanded past cap");
            pressure=true;leased.refresh();
            check(leased.snapshot().bytes==8192 && !released,"pressure respects live readers");
            second={};leased.refresh();
            check(leased.snapshot().bytes==4096 && released==1,"pressure frees only released readers");
            check(!std::memcmp(held.data(),original.data(),4096),"held matrix immutable during competing reads");
            pressure=false;
        }
        check(released==1 && !std::memcmp(held.data(),original.data(),4096),"lease outlives cache owner");
        held={};check(allocated==released,"custom storage released exactly once");
        Cache unpublished(4096,probe,Cache::Policy::Lru);
        check(!unpublished.read(a,0,dest.data(),4096,cancel,interrupted) && !unpublished.acquire(a,0,4096),
            "cancelled fill cannot be leased");
        // Promotion discards complete chunks only, retaining live leases in
        // the budget until release; it never mutates DMA/CPU source bytes.
        {
            Cache promoted(16384,probe,Cache::Policy::Lru);
            for(size_t off:{0,4096,8192})check(promoted.read(a,off,dest.data(),4096,cancel,request),"prime promotion chunks");
            check(promoted.read(b,0,dest.data(),4096,cancel,request),"prime separate file identity");
            auto lease=promoted.acquire(a,4096,4096);auto dma=lease.shared_data();lease={};
            promoted.discard(a,0,8192);
            s=promoted.snapshot();
            check(s.bytes==12288 && s.retired_bytes==4096 && s.gpu_drop_bytes==4096,"promotion retains charged DMA lease");
            check(!promoted.acquire(a,0,4096) && !promoted.acquire(a,4096,4096),"retired payload unavailable to new readers");
            check(bool(promoted.acquire(a,8192,4096)) && bool(promoted.acquire(b,0,4096)),"unpromoted chunks and other identities survive");
            promoted.discard(a,8192,4095);
            check(bool(promoted.acquire(a,8192,4096)),"partial promotion does not discard a full chunk");
            auto fills=promoted.snapshot().fill_bytes;
            check(promoted.read(a,4096,dest.data(),4096,cancel,request),"retired miss uses ordinary delivery");
            check(promoted.snapshot().fill_bytes==fills && !std::memcmp(dma.get(),original.data()+4096,4096),"retired read neither duplicates nor mutates held allocation");
            promoted.reset_counters();check(promoted.snapshot().retired_bytes==4096,"reset retains retired budget gauge");
            pressure=true;promoted.refresh();s=promoted.snapshot();
            check(s.bytes==4096 && s.retired_bytes==4096 && !s.budget,"pressure preserves only live DMA payload");
            dma.reset();promoted.refresh();s=promoted.snapshot();
            check(!s.bytes && !s.retired_bytes && !s.entries && s.gpu_drop_bytes==4096,"last DMA release permits reclamation");
            pressure=false;
            check(promoted.read(a,4096,dest.data(),4096,cancel,request) && bool(promoted.acquire(a,4096,4096)),"one-shot promotion allows later demand refill");
        }
        // Delivery observes demand but defers payload allocation until the
        // consumer knows the matrix was rejected by the GPU cache.
        {
            Cache late(8192,probe);size_t loads=0;
            auto delivered=[&](uint64_t off,bool phase=false) {
                return late.read(a,off,dest.data(),4096,cancel,[&](void *p,size_t n){
                    ++loads;std::memcpy(p,original.data()+off,n);return true;
                },phase,true);
            };
            check(delivered(0),"deferred first delivery");late.retain(a,0,dest.data(),4096,cancel);
            check(!late.snapshot().bytes && !late.snapshot().allocations,"retain does not train a first-use miss twice");
            check(delivered(0),"deferred repeated delivery");
            check(!late.snapshot().bytes && !late.snapshot().pending,"producer does not allocate before placement");
            late.retain(a,0,dest.data(),4096,cancel);s=late.snapshot();
            check(s.bytes==4096 && s.fill_bytes==4096 && loads==2 && !s.pending,"retain publishes delivered bytes without file I/O");
            check(delivered(0) && loads==2 && !std::memcmp(dest.data(),original.data(),4096),"retained payload supplies exact later hit");
            late.retain(a,0,dest.data(),4096,cancel);
            check(late.snapshot().retain_hits==1 && late.snapshot().allocations==1,"retaining a hit does not copy or allocate");
            for(int i=0;i<3;++i)check(delivered(4096),"GPU-admitted delivery without RAM retention");
            check(!late.acquire(a,4096,4096) && late.snapshot().bytes==4096,"GPU admissions leave no RAM duplicate");
            check(delivered(4096),"staging ready before cancellation");cancel=true;late.retain(a,4096,dest.data(),4096,cancel);cancel=false;
            check(!late.acquire(a,4096,4096),"cancelled retention publishes nothing");
            late.set_prefill(true);late.retain(a,4096,dest.data(),4096,cancel);late.set_prefill(false);
            check(!late.acquire(a,4096,4096),"prefill phase suppresses late admission");
            late.retain(a,4096,dest.data(),4096,cancel,true);
            check(!late.acquire(a,4096,4096),"per-call prefill suppresses late admission");
            pressure=true;late.retain(a,4096,dest.data(),4096,cancel);s=late.snapshot();
            check(!s.bytes && !s.budget && !s.pending,"pressure bounds late retention");pressure=false;
            check(delivered(4096),"delivery after pressure");late.retain(a,4096,dest.data(),4096,cancel);
            check(bool(late.acquire(a,4096,4096)),"late retention resumes with memory headroom");
        }
        {
            Cache::Storage fail_storage;
            fail_storage.allocate=[](size_t)->uint8_t *{throw std::runtime_error("injected allocator failure");};
            fail_storage.release=[](uint8_t *){};
            Cache fail(4096,probe,Cache::Policy::Lru,131072,131072,fail_storage);
            bool threw=false;try{fail.retain(a,0,original.data(),4096,cancel);}catch(const std::runtime_error &){threw=true;}
            check(threw && !fail.snapshot().pending && !fail.snapshot().bytes,"retention allocator exception drains reservation");
        }
        // Free allocations from promotion remain within the original budget.
        // Repeated drops reuse storage, while held DMA leases stay immutable.
        {
            int allocations=0,releases=0;
            Cache::Storage allocator;
            allocator.allocate=[&](size_t n){++allocations;return new uint8_t[n];};
            allocator.release=[&](uint8_t *p){++releases;delete[] p;};
            {
                Cache pooled(8192,probe,Cache::Policy::Lru,131072,131072,allocator,8192);
                for(int i=0;i<12;++i) {
                    pooled.retain(a,0,original.data(),4096,cancel);
                    auto lease=pooled.acquire(a,0,4096);
                    check(bool(lease) && !std::memcmp(lease.data(),original.data(),4096),"pooled bytes exact");
                    pooled.discard(a,0,4096);s=pooled.snapshot();
                    check(s.bytes==4096 && s.retired_bytes==4096 && !s.pool_bytes,"leased block not pooled");
                    lease={};s=pooled.snapshot();
                    check(s.bytes==4096 && s.pool_bytes==4096 && !s.entries,"free pool remains charged");
                }
                s=pooled.snapshot();check(allocations==1 && !releases && s.pool_reuses==11,"promotion churn reuses one allocation");
                pooled.reset_counters();s=pooled.snapshot();
                check(s.bytes==4096 && s.pool_bytes==4096 && s.pool_peak==4096 && !s.pool_puts,"reset preserves pool gauge");
                pooled.retain(b,4096,original.data()+4096,4096,cancel);
                auto lease=pooled.acquire(b,4096,4096);
                check(bool(lease) && !std::memcmp(lease.data(),original.data()+4096,4096) && !pooled.acquire(a,0,4096),"pooled allocation changes file identity safely");
                pooled.discard(b,4096,4096);lease={};pooled.snapshot();
                // A different size releases free storage before evicting payload
                // or admitting memory beyond the cap.
                pooled.retain(a,8192,original.data()+8192,8192,cancel);s=pooled.snapshot();
                check(s.bytes==8192 && !s.pool_bytes && allocations==2 && releases==1,"size mismatch frees pool within total cap");
                pooled.discard(a,8192,8192);s=pooled.snapshot();
                check(s.pool_bytes==8192 && s.bytes==8192,"large spare included in cap");
                pressure=true;pooled.refresh();s=pooled.snapshot();
                check(!s.bytes && !s.pool_bytes && !s.budget && releases==2,"memory pressure reclaims free pool");pressure=false;
                pooled.retain(a,0,original.data(),4096,cancel);pooled.discard(a,0,4096);
            }
            check(allocations==releases,"pool destructor releases each allocation once");
            Cache capped(16384,probe,Cache::Policy::Lru,131072,131072,{},4096);
            capped.retain(a,0,original.data(),4096,cancel);capped.retain(a,4096,original.data()+4096,4096,cancel);
            capped.discard(a,0,8192);s=capped.snapshot();
            check(s.bytes==4096 && s.pool_bytes==4096 && !s.entries,"separate pool cap enforced");
        }
        // Publication tickets prevent duplicate fills and late admission after
        // promotion, even when an asynchronous producer already owns its bytes.
        {
            Cache tickets(8192,probe,Cache::Policy::Lru,131072,131072,{},8192);
            check(tickets.read(a,0,dest.data(),4096,cancel,[&](void *p,size_t n){
                check(tickets.residency(a,0,n).ram==Cache::State::Filling,"in-flight fill visible only as state");
                check(!tickets.acquire(a,0,n),"pending bytes cannot be acquired");
                std::vector<uint8_t> other(n);
                check(tickets.read(a,0,other.data(),n,cancel,request),"concurrent duplicate still delivers bytes");
                check(tickets.snapshot().pending==n && tickets.snapshot().duplicate_fills==1,"same key reserves once");
                tickets.discard(a,0,n);std::memcpy(p,original.data(),n);return true;
            }),"invalidated fill still completes caller delivery");
            check(!tickets.acquire(a,0,4096) && !tickets.snapshot().pending && tickets.snapshot().stale_fills==1,
                "discard prevents late producer publication");
            tickets.retain(a,0,original.data(),4096,cancel);
            auto lease=tickets.acquire(a,0,4096);const auto generation=lease.generation();const auto pointer=lease.data();
            check(generation && tickets.residency(a,0,4096).ram==Cache::State::Held,"ready publication has held generation");
            tickets.discard(a,0,4096);
            check(tickets.residency(a,0,4096).ram==Cache::State::Retired && lease.generation()==generation,
                "retirement never changes an existing lease generation");
            lease={};tickets.snapshot();tickets.retain(b,4096,original.data()+4096,4096,cancel);
            lease=tickets.acquire(b,4096,4096);
            check(lease.data()==pointer && lease.generation()>generation,"same pooled address gets a fresh publication generation");
        }
        {
            Cache tiers(16384,probe,Cache::Policy::Lru);
            tiers.retain(a,0,original.data(),4096,cancel);auto held=tiers.acquire(a,0,4096);
            auto target=tiers.begin_gpu(a,0,4096,true),draft=tiers.begin_gpu(a,0,4096,true);
            check(tiers.residency(a,0,4096).gpu==Cache::State::Transferring && bool(tiers.acquire(a,0,4096)),
                "upload enqueue does not retire ready RAM before event completion");
            check(tiers.ready_gpu(target),"publish completed GPU generation");
            check(!tiers.acquire(a,0,4096) && tiers.snapshot().retired_bytes==4096 &&
                !std::memcmp(held.data(),original.data(),4096),"promotion retires but retains live CPU/DMA data");
            held={};tiers.snapshot();tiers.end_gpu(target);
            tiers.retain(a,0,original.data(),4096,cancel);
            check(!tiers.acquire(a,0,4096) && tiers.snapshot().gpu_bypasses==1,"remaining backend suppresses RAM duplicate");
            check(!tiers.ready_gpu(target) && tiers.snapshot().gpu_entries==1,"stale GPU completion cannot revive an evicted generation");
            tiers.end_gpu(draft);tiers.retain(a,0,original.data(),4096,cancel);
            check(bool(tiers.acquire(a,0,4096)),"last GPU eviction allows demand refill");
            auto padded=tiers.begin_gpu(a,8192,8192,true);
            tiers.retain(a,12288,original.data()+12288,4096,cancel);
            tiers.retain(b,12288,original.data()+12288,4096,cancel);
            check(!tiers.acquire(a,12288,4096) && bool(tiers.acquire(b,12288,4096)),"coverage includes chunks but keeps file identities separate");
            tiers.end_gpu(padded);
            check(tiers.read(a,16384,dest.data(),4096,cancel,[&](void *p,size_t n){
                auto upload=tiers.begin_gpu(a,16384,n,true);tiers.ready_gpu(upload);tiers.end_gpu(upload);
                std::memcpy(p,original.data()+16384,n);return true;
            }) && !tiers.acquire(a,16384,4096),"late fill stays invalid after promote followed by immediate eviction");
        }
        {
            auto tiers=std::make_shared<Cache>(12288,probe,Cache::Policy::Lru,131072,131072,Cache::Storage{},0,8192);
            auto x=tiers->begin_gpu(a,0,4096,true),y=tiers->begin_gpu(a,4096,4096,true),z=tiers->begin_gpu(a,8192,4096,true);
            check(!tiers->begin_demotion(x),"unfinished upload cannot be demoted");
            tiers->ready_gpu(x);tiers->ready_gpu(y);tiers->ready_gpu(z);
            check(!tiers->begin_demotion(x,true),"prefill cannot evict hot RAM through demotion");
            auto dx=tiers->begin_demotion(x),dy=tiers->begin_demotion(y);
            check(dx && dy && !tiers->begin_demotion(z) && tiers->snapshot().pending==8192,
                "target and draft share two demotions and one byte budget");
            check(tiers->residency(a,0,4096).ram==Cache::State::Transferring && !tiers->acquire(a,0,4096),"partial D2H is unavailable");
            std::memcpy(dx->data(),original.data(),4096);tiers->end_gpu(x);dx->complete();
            check(bool(tiers->acquire(a,0,4096)) && tiers->snapshot().demotion_bytes==4096,"completed D2H publishes exact bytes");
            tiers->end_gpu(y);dy.reset();check(!tiers->acquire(a,4096,4096) && !tiers->snapshot().pending,"cancelled demotion releases its reservation");
            auto dz=tiers->begin_demotion(z);check(bool(dz),"demotion slots return after completion and cancellation");
            std::memcpy(dz->data(),original.data()+8192,4096);tiers->end_gpu(z);
            auto newer=tiers->begin_gpu(a,8192,4096,true);tiers->ready_gpu(newer);dz->complete();
            check(!tiers->acquire(a,8192,4096) && tiers->snapshot().stale_fills==1,"new GPU promotion cancels an old in-flight demotion");
            tiers->end_gpu(newer);
            auto other=tiers->begin_gpu(a,4096,4096,true);tiers->ready_gpu(other);
            auto old=tiers->begin_gpu(a,4096,4096,true);tiers->ready_gpu(old);
            check(!tiers->begin_demotion(old),"another resident GPU copy suppresses demotion duplicate");
            tiers->end_gpu(other);tiers->end_gpu(old);
            s=tiers->snapshot();check(!s.pending && !s.gpu_entries && !s.gpu_pending && !s.demotion_jobs,
                "all reservations and GPU generations drain");
        }
        {
            Cache held(8192,probe,Cache::Policy::Lru);
            held.retain(a,0,original.data(),4096,cancel);held.retain(a,4096,original.data()+4096,4096,cancel);
            const auto before=held.snapshot();auto snapshot=held.acquire(a,0,4096,false,false);
            const auto after=held.snapshot();
            check(snapshot && after.lease_hits==before.lease_hits && after.reused_payload_bytes==before.reused_payload_bytes,
                "unselected routing snapshot does not train or count a RAM hit");
            held.discard(a,0,4096);
            check(!held.acquire(a,0,4096,false,false) && !std::memcmp(snapshot.data(),original.data(),4096),
                "snapshot survives retirement but cannot be newly acquired");
            snapshot={};held.snapshot();check(held.snapshot().retired_bytes==0,"retired snapshot releases its payload");
        }
        report["status"]="pass";
    } catch(const std::exception &e) {report["error"]=e.what();}
    report["checks"]=checks;std::ofstream(output)<<report.dump(2)<<'\n';
    std::cout<<report.dump(2)<<'\n';return report["status"]=="pass"?0:1;
}
