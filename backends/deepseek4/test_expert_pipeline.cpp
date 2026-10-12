#include "expert_pipeline.hpp"
#include "expert_reuse.hpp"
#include "host_copy.hpp"
#include "host_cache.hpp"
#include <atomic>
#include <cstdio>

static void check(bool value) {if(!value)throw std::runtime_error("pipeline contract failed");}
static void cuda_ok(cudaError_t code) {if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));}
struct Gate {
    cudaStream_t stream;
    std::atomic<bool> release{false},expired{false};
    explicit Gate(cudaStream_t s):stream(s) {}
    ~Gate() {release.store(true);if(stream)cudaStreamSynchronize(stream);}
};
static void CUDART_CB blocked(void *ptr) {
    auto &gate=*(Gate *)ptr;
    auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(3);
    while(!gate.release.load()) {
        if(std::chrono::steady_clock::now()>deadline) {gate.expired.store(true);break;}
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

static void test_reuse(cudaStream_t compute,uint8_t *slot,size_t size) {
    cudaStream_t next;cuda_ok(cudaStreamCreateWithFlags(&next,cudaStreamNonBlocking));
    uint8_t *old=nullptr;cuda_ok(cudaMalloc((void **)&old,size));
    std::vector<uint8_t> before(size),after(size);
    StrataExpertReuseFence fence;
    for(bool cross:{false,true}) {
        Gate gate(compute);
        cuda_ok(cudaLaunchHostFunc(compute,blocked,&gate));
        cuda_ok(cudaMemsetAsync(slot,0x19,size,compute));
        cuda_ok(cudaMemcpyAsync(old,slot,size,cudaMemcpyDeviceToDevice,compute));
        auto consumer=cross?next:compute;
        check(fence.order(compute,consumer)==cross);
        // Host must not wait for the deliberately blocked source stream.
        bool returned_early=!gate.expired.load();gate.release.store(true);
        check(returned_early);
        cuda_ok(cudaMemsetAsync(slot,0xA3,size,consumer));
        cuda_ok(cudaStreamSynchronize(consumer));cuda_ok(cudaStreamSynchronize(compute));
        cuda_ok(cudaMemcpy(before.data(),old,size,cudaMemcpyDeviceToHost));
        cuda_ok(cudaMemcpy(after.data(),slot,size,cudaMemcpyDeviceToHost));
        check(std::all_of(before.begin(),before.end(),[](uint8_t x){return x==0x19;}));
        check(std::all_of(after.begin(),after.end(),[](uint8_t x){return x==0xA3;}));
    }
    cuda_ok(cudaFree(old));cuda_ok(cudaStreamDestroy(next));
}

#ifdef _WIN32
struct MappedFixture {
    FILE *file=nullptr;HANDLE mapping=nullptr;void *ptr=nullptr;size_t size;
    explicit MappedFixture(const std::vector<uint8_t> &bytes):size(bytes.size()) {
        file=std::tmpfile();check(file!=nullptr);
        check(std::fwrite(bytes.data(),1,size,file)==size && std::fflush(file)==0);
        mapping=CreateFileMappingW((HANDLE)_get_osfhandle(_fileno(file)),nullptr,PAGE_READONLY,0,0,nullptr);
        check(mapping!=nullptr);ptr=MapViewOfFile(mapping,FILE_MAP_READ,0,0,0);check(ptr!=nullptr);
        strata_expert_file::add(ptr,size,_fileno(file));
    }
    ~MappedFixture() {
        if(ptr) {strata_expert_file::remove(ptr);UnmapViewOfFile(ptr);}
        if(mapping)CloseHandle(mapping);
        if(file)std::fclose(file);
    }
};
static void test_borrow(cudaStream_t compute,uint8_t *dest,const std::vector<uint8_t> &expected) {
    MappedFixture mapped(expected);auto *source=(const uint8_t *)mapped.ptr+13;
    const size_t bytes=expected.size()-13,chunk=256<<10;
    auto file=strata_expert_file::find(source,bytes);check(bool(file));
    using Cache=strata_ds4::HostCache;
    for(bool early:{false,true}) {
        Cache::Storage storage;
        storage.allocate=[](size_t n){void *p=nullptr;cuda_ok(cudaMallocHost(&p,n));return (uint8_t *)p;};
        storage.release=[](uint8_t *p){cuda_ok(cudaFreeHost(p));};
        Cache cache(4<<20,[]{return Cache::Memory{8ULL<<30,6ULL<<30,16ULL<<30};},Cache::Policy::Lru,131072,131072,storage);
        StrataExpertPipeline::SourceCopy copy=[&](const auto &f,const void *src,void *dst,size_t n,
                const std::atomic<bool> &cancel,strata_expert_file::Request &request,bool,bool) {
            return cache.read(f,f->offset_of(src,n),dst,n,cancel,request);
        };
        StrataExpertPipeline::SourceBorrow borrow=[&](const auto &f,const void *src,size_t n,bool cacheable) {
            return cache.acquire(f,f->offset_of(src,n),n,!cacheable).shared_data();
        };
        {
            StrataExpertPipeline pipeline(0,chunk,false,2,1,{},1,early,{}, {},copy,borrow);
            std::vector<uint8_t> actual(bytes);
            for(int repeat=0;repeat<4;++repeat) {
                pipeline.start({{source,bytes,true}});
                check(pipeline.transfer(dest,source,bytes,compute));pipeline.finish();cuda_ok(cudaStreamSynchronize(compute));
                cuda_ok(cudaMemcpy(actual.data(),dest,bytes,cudaMemcpyDeviceToHost));
                check(std::equal(actual.begin(),actual.end(),expected.begin()+13));
            }
            auto counts=pipeline.counters();
            check(counts.borrowed_bytes==3*bytes && counts.file_bytes==bytes && !counts.mmap_bytes);
            // Abandoned slots own leases; promotion may not free their bytes.
            pipeline.start({{source,bytes,true}});
            auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(2);
            while(pipeline.ready_chunks()!=4 && std::chrono::steady_clock::now()<deadline)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            check(pipeline.ready_chunks()==4);cache.discard(file,13,bytes);
            check(cache.snapshot().retired_bytes>=4*chunk);
            pipeline.finish();cache.refresh();
            check(!cache.snapshot().retired_bytes && !cache.snapshot().bytes);
            // Restart mixes new fills with direct hits and wraps the ring.
            for(int repeat=0;repeat<2;++repeat) {
                pipeline.start({{source,bytes,true},{source,bytes,true}});
                check(pipeline.transfer(dest,source,bytes,compute));pipeline.finish();cuda_ok(cudaStreamSynchronize(compute));
                cuda_ok(cudaMemcpy(actual.data(),dest,bytes,cudaMemcpyDeviceToHost));
                check(std::equal(actual.begin(),actual.end(),expected.begin()+13));
            }
        }
        cache.discard(file,13,bytes);check(!cache.snapshot().bytes);
    }
    // An observer error after H2D enqueue must not release its host source.
    void *p=nullptr;cuda_ok(cudaMallocHost(&p,chunk));std::memcpy(p,source,chunk);
    std::shared_ptr<const void> owner(p,[](const void *q){cuda_ok(cudaFreeHost(const_cast<void *>(q)));});
    std::weak_ptr<const void> weak=owner;
    std::unique_ptr<Gate> gate;
    StrataExpertPipeline::CopyObserver observer=[&](cudaStream_t stream,bool start,size_t) {
        if(start){gate=std::make_unique<Gate>(stream);cuda_ok(cudaLaunchHostFunc(stream,blocked,gate.get()));}
        else throw std::runtime_error("injected after H2D enqueue");
    };
    auto pipeline=std::make_unique<StrataExpertPipeline>(0,chunk,false,1,0,observer,1,false,
        StrataExpertPipeline::HostCopy{},StrataExpertPipeline::SourceRead{},StrataExpertPipeline::SourceCopy{},
        [&](const auto &,const void *,size_t,bool){return owner;});
    pipeline->start({{source,chunk,true}});
    bool failed=false;try{pipeline->transfer(dest,source,chunk,compute);}catch(const std::runtime_error &){failed=true;}
    check(failed && gate && !gate->expired.load());owner.reset();pipeline->finish(false);
    check(!weak.expired());gate->release.store(true);pipeline.reset();
    // Gate owns no stream; the pipeline destroyed it after draining.
    gate->stream=nullptr;check(weak.expired());
}
static void test_late_retention(cudaStream_t compute,uint8_t *dest,const std::vector<uint8_t> &expected) {
    MappedFixture mapped(expected);auto *source=(const uint8_t *)mapped.ptr+13;
    const size_t bytes=expected.size()-13,chunk=256<<10;
    auto file=strata_expert_file::find(source,bytes);using Cache=strata_ds4::HostCache;
    for(bool direct:{false,true}) {
        Cache::Storage storage;
        storage.allocate=[](size_t n){void *p=nullptr;cuda_ok(cudaMallocHost(&p,n));return (uint8_t *)p;};
        storage.release=[](uint8_t *p){cuda_ok(cudaFreeHost(p));};
        Cache cache(4<<20,[]{return Cache::Memory{8ULL<<30,6ULL<<30,16ULL<<30};},Cache::Policy::Frequency,131072,131072,storage);
        StrataExpertPipeline::SourceCopy copy=[&](const auto &f,const void *src,void *dst,size_t n,
                const std::atomic<bool> &cancel,strata_expert_file::Request &request,bool cacheable,bool) {
            return cache.read(f,f->offset_of(src,n),dst,n,cancel,[&](void *p,size_t count){
                return request.read(*f,src,p,count,cancel);
            },!cacheable,true);
        };
        StrataExpertPipeline::SourceBorrow borrow;
        if(direct)borrow=[&](const auto &f,const void *src,size_t n,bool cacheable){
            return cache.acquire(f,f->offset_of(src,n),n,!cacheable).shared_data();
        };
        size_t calls=0;
        StrataExpertPipeline::SourceRetain retain=[&](const auto &f,const void *src,const void *payload,size_t n,const auto &cancel){
            ++calls;cache.retain(f,f->offset_of(src,n),payload,n,cancel);
        };
        {
            StrataExpertPipeline pipeline(0,chunk,false,2,1,{},1,true,{}, {},copy,borrow,retain);
            std::vector<uint8_t> actual(bytes);
            auto deliver=[&](bool keep,bool cacheable=true) {
                pipeline.start({{source,bytes,cacheable}});
                check(pipeline.transfer(dest,source,bytes,compute,{},keep));pipeline.finish();cuda_ok(cudaStreamSynchronize(compute));
                cuda_ok(cudaMemcpy(actual.data(),dest,bytes,cudaMemcpyDeviceToHost));
                check(std::equal(actual.begin(),actual.end(),expected.begin()+13));
            };
            deliver(false);deliver(false);
            check(!calls && !cache.snapshot().bytes && !cache.snapshot().allocations);
            deliver(true);check(calls==5 && cache.snapshot().bytes==bytes && cache.snapshot().fill_bytes==bytes);
            deliver(true);check(calls==(direct?5:10) && cache.snapshot().fill_bytes==bytes);
            const auto counts=pipeline.counters();
            check(counts.retain_chunks==calls && counts.retain_bytes==(direct?bytes:2*bytes));
            check(counts.borrowed_bytes==(direct?bytes:0));
            cache.discard(file,13,bytes);
            pipeline.start({{source,bytes,true}});
            auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(2);
            while(pipeline.ready_chunks()!=4 && std::chrono::steady_clock::now()<deadline)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            check(pipeline.ready_chunks()==4);auto before=calls;pipeline.finish();cache.refresh();
            check(calls==before && !cache.snapshot().bytes && !cache.snapshot().pending);
            deliver(true,false);check(calls==before && !cache.snapshot().bytes); // prefill never retained
        }
    }
    // Retention may fail after D2D is queued. Consumer fences must remain live.
    Gate gate(compute);cuda_ok(cudaLaunchHostFunc(compute,blocked,&gate));
    {
        StrataExpertPipeline pipeline(0,chunk,false,1,0,{},1,false,{}, {},{}, {},
            [](const auto &,const void *,const void *,size_t,const auto &){throw std::runtime_error("injected retain failure");});
        pipeline.start({{source,chunk,true}});bool failed=false;
        try{pipeline.transfer(dest,source,chunk,compute,{},true);}catch(const std::runtime_error &){failed=true;}
        check(failed && !gate.expired.load());gate.release.store(true);pipeline.finish(false);
        failed=false;try{pipeline.start({{source,chunk,true}});}catch(const std::runtime_error &){failed=true;}
        check(failed);cuda_ok(cudaStreamSynchronize(compute));
        std::vector<uint8_t> actual(chunk);cuda_ok(cudaMemcpy(actual.data(),dest,chunk,cudaMemcpyDeviceToHost));
        check(std::equal(actual.begin(),actual.end(),expected.begin()+13));
    }
}
static void test_async_retention(cudaStream_t compute,uint8_t *dest,const std::vector<uint8_t> &expected) {
    MappedFixture mapped(expected);auto *source=(const uint8_t *)mapped.ptr+13;
    const size_t chunk=256<<10,bytes=expected.size()-13;
    auto file=strata_expert_file::find(source,bytes);using Cache=strata_ds4::HostCache;
    Cache cache(4<<20,[]{return Cache::Memory{8ULL<<30,6ULL<<30,16ULL<<30};},Cache::Policy::Lru,131072,131072,{},1<<20);
    std::atomic<bool> release{false},entered{false},saw_cancel{false},inject{false};
    const auto caller=std::this_thread::get_id();std::atomic<size_t> calls{0};
    StrataExpertPipeline::SourceRetain retain=[&](const auto &f,const void *src,const void *payload,size_t n,const auto &cancel) {
        check(std::this_thread::get_id()!=caller);entered.store(true);
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(2);
        while(!release.load() && !cancel.load() && std::chrono::steady_clock::now()<deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        check(release.load() || cancel.load());
        if(inject.load())throw std::runtime_error("injected background retain failure");
        const auto off=f->offset_of(src,n);check(!std::memcmp(payload,expected.data()+off,n));
        if(cancel.load())saw_cancel.store(true);
        cache.retain(f,off,payload,n,cancel);++calls;
    };
    auto wait_entered=[&] {
        auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(1);
        while(!entered.load() && std::chrono::steady_clock::now()<deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        check(entered.load());
    };
    struct Release {std::atomic<bool> &flag;~Release(){flag.store(true);}};
    {
        StrataExpertPipeline pipeline(0,chunk,false,2,1,{},1,true,{}, {},{}, {},retain,2);
        Release guard{release};
        // Two retained slots include the active callback; third is dropped.
        // GPU delivery must finish on the caller while that callback is held.
        pipeline.start({{source,3*chunk,true}});
        check(pipeline.transfer(dest,source,3*chunk,compute,{},true));wait_entered();
        auto counts=pipeline.counters();
        check(counts.retain_queued==2 && counts.retain_dropped==1 && counts.retain_pending_peak==2 && !calls.load());
        release.store(true);pipeline.finish();cuda_ok(cudaStreamSynchronize(compute));
        std::vector<uint8_t> actual(bytes);cuda_ok(cudaMemcpy(actual.data(),dest,3*chunk,cudaMemcpyDeviceToHost));
        check(!std::memcmp(actual.data(),source,3*chunk));
        check(calls.load()==2 && cache.snapshot().fill_bytes==2*chunk);
        // Repeat full ring wrap-around. Retention may be dropped, delivery may not.
        for(int repeat=0;repeat<8;++repeat) {
            pipeline.start({{source,bytes,true}});check(pipeline.transfer(dest,source,bytes,compute,{},true));
            pipeline.finish();cuda_ok(cudaStreamSynchronize(compute));
            cuda_ok(cudaMemcpy(actual.data(),dest,bytes,cudaMemcpyDeviceToHost));
            check(!std::memcmp(actual.data(),source,bytes));
        }
        auto before=calls.load();pipeline.start({{source,chunk,false}});
        check(pipeline.transfer(dest,source,chunk,compute,{},true));pipeline.finish();check(calls.load()==before);
        cache.discard(file,13,bytes);auto fills=cache.snapshot().fill_bytes;
        // An abandoned suffix cancels an active fill and releases the ring.
        release.store(false);entered.store(false);
        pipeline.start({{source,bytes,true}});check(pipeline.transfer(dest,source,chunk,compute,{},true));wait_entered();
        pipeline.finish();check(saw_cancel.load() && cache.snapshot().fill_bytes==fills);
        release.store(true);pipeline.start({{source,bytes,true}});
        check(pipeline.transfer(dest,source,bytes,compute,{},true));pipeline.finish();
        counts=pipeline.counters();check(counts.retain_pending_peak==2 && !cache.snapshot().pending);
    }
    // Worker exceptions propagate at finish and stay sticky. Queued D2D still
    // owns the device slot until the consumer fence completes.
    release.store(false);entered.store(false);inject.store(true);
    Gate gate(compute);cuda_ok(cudaLaunchHostFunc(compute,blocked,&gate));
    {
        StrataExpertPipeline pipeline(0,chunk,false,1,0,{},1,false,{}, {},{}, {},retain,2);
        Release guard{release};pipeline.start({{source,chunk,true}});
        check(pipeline.transfer(dest,source,chunk,compute,{},true));wait_entered();release.store(true);
        bool failed=false;try{pipeline.finish();}catch(const std::runtime_error &){failed=true;}
        check(failed && !gate.expired.load());gate.release.store(true);
        failed=false;try{pipeline.start({{source,chunk,true}});}catch(const std::runtime_error &){failed=true;}
        check(failed);cuda_ok(cudaStreamSynchronize(compute));
        std::vector<uint8_t> actual(chunk);cuda_ok(cudaMemcpy(actual.data(),dest,chunk,cudaMemcpyDeviceToHost));
        check(!std::memcmp(actual.data(),source,chunk));
    }
}
static void test_file(cudaStream_t compute,uint8_t *dest,const std::vector<uint8_t> &expected) {
    MappedFixture mapped(expected);
    auto *source=(const uint8_t *)mapped.ptr+13;
    size_t bytes=expected.size()-13;
    check(!strata_expert_file::find(source,bytes+1));
    std::vector<uint8_t> actual(bytes);
    {
        strata_ds4::HostCache cache(4<<20,[]{return strata_ds4::HostCache::Memory{8ULL<<30,6ULL<<30,16ULL<<30};});
        std::atomic<int> decode_jobs{0},prefill_jobs{0};
        StrataExpertPipeline::SourceCopy copy=[&](const auto &file,const void *src,void *dst,size_t n,
                const std::atomic<bool> &cancel,strata_expert_file::Request &,bool cacheable,bool native) {
            check(!native); // adding a cache must preserve mode=0 mmap delivery
            ++(cacheable?decode_jobs:prefill_jobs);
            return cache.read(file,file->offset_of(src,n),dst,n,cancel,[&](void *p,size_t count) {
                std::memcpy(p,src,count);return !cancel.load();
            },!cacheable);
        };
        StrataExpertPipeline pipeline(0,256<<10,false,2,0,{},1,false,{}, {},copy);
        for(bool cacheable:{true,true,true,false,true}) {
            pipeline.start({{source,bytes,cacheable}});
            check(pipeline.transfer(dest,source,bytes,compute));pipeline.finish();cuda_ok(cudaStreamSynchronize(compute));
            cuda_ok(cudaMemcpy(actual.data(),dest,bytes,cudaMemcpyDeviceToHost));
            check(std::equal(actual.begin(),actual.end(),expected.begin()+13));
        }
        const auto h=cache.snapshot();
        check(decode_jobs>0 && prefill_jobs>0 && h.hit_bytes>=3*bytes && h.bytes==bytes && !h.pending);
    }
    {
        StrataExpertPipeline pipeline(0,256<<10,false,4,1);
        DWORD protection;
        check(VirtualProtect(mapped.ptr,mapped.size,PAGE_NOACCESS,&protection)!=0);
        for(int repeat=0;repeat<12;++repeat) {
            pipeline.start({{source,bytes,false},{source,bytes,false}});
            // Requests retain their file identity after the registry removes
            // this mapping. Native reads must not dereference mmap pages.
            strata_expert_file::remove(mapped.ptr);
            check(pipeline.transfer(dest,source,bytes,compute));
            pipeline.finish();cuda_ok(cudaStreamSynchronize(compute));
            cuda_ok(cudaMemcpy(actual.data(),dest,bytes,cudaMemcpyDeviceToHost));
            check(std::equal(actual.begin(),actual.end(),expected.begin()+13));
            strata_expert_file::add(mapped.ptr,mapped.size,_fileno(mapped.file));
        }
        auto counts=pipeline.counters();
        check(counts.file_bytes>=12*bytes && counts.mmap_bytes==0 && counts.read_peak>0 && counts.read_peak<=4);
        check(VirtualProtect(mapped.ptr,mapped.size,PAGE_READONLY,&protection)!=0);
    }
    {
        StrataExpertPipeline pipeline(0,256<<10,false,2,2);
        pipeline.start({{source,bytes,false}});
        check(pipeline.transfer(dest,source,bytes,compute));pipeline.finish();cuda_ok(cudaStreamSynchronize(compute));
        cuda_ok(cudaMemcpy(actual.data(),dest,bytes,cudaMemcpyDeviceToHost));
        check(std::equal(actual.begin(),actual.end(),expected.begin()+13));
        pipeline.start({{source,bytes,true}});
        check(pipeline.transfer(dest,source,bytes,compute));pipeline.finish();cuda_ok(cudaStreamSynchronize(compute));
        cuda_ok(cudaMemcpy(actual.data(),dest,bytes,cudaMemcpyDeviceToHost));
        check(std::equal(actual.begin(),actual.end(),expected.begin()+13));
        check(pipeline.counters().mmap_bytes>=bytes);
    }
}
#endif
int main() {
    try {
        int count=0;
        if(cudaGetDeviceCount(&count)!=cudaSuccess || !count)return 77;
        constexpr size_t size=(1<<20)+17,chunk=256<<10;
        std::vector<uint8_t> source(size),actual(size);
        for(size_t i=0;i<size;++i)source[i]=uint8_t(i*17+(i>>7));
        uint8_t *dest=nullptr;cuda_ok(cudaMalloc((void **)&dest,size));
        cudaStream_t compute;cuda_ok(cudaStreamCreateWithFlags(&compute,cudaStreamNonBlocking));
        for(bool avx2:{false,true})for(bool early:{false,true}) {
            if(avx2 && !strata_ds4_host_copy::available())continue;
            StrataExpertPipeline pipeline(0,chunk,false,2,0,{},1,early,
                avx2?StrataExpertPipeline::HostCopy(strata_ds4_host_copy::copy):StrataExpertPipeline::HostCopy{});
            Gate gate(compute);
            cuda_ok(cudaLaunchHostFunc(compute,blocked,&gate));
            pipeline.start({{source.data(),size,false}});
            auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(2);
            while(pipeline.ready_chunks()<StrataExpertPipeline::slots && std::chrono::steady_clock::now()<deadline)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            bool overlapped=pipeline.ready_chunks()==StrataExpertPipeline::slots && !gate.expired.load();
            bool blocked_compute=cudaStreamQuery(compute)==cudaErrorNotReady;
            const auto full=pipeline.counters();
            check(full.pinned_bytes==4*chunk && full.device_ring_bytes==4*chunk &&
                  full.queued_bytes==4*chunk && full.queued_peak==4*chunk && full.reader_owned_bytes==0);
            gate.release.store(true);
            check(overlapped && blocked_compute);
            check(pipeline.transfer(dest,source.data(),size,compute));
            pipeline.finish();cuda_ok(cudaStreamSynchronize(compute));
            cuda_ok(cudaMemcpy(actual.data(),dest,size,cudaMemcpyDeviceToHost));check(actual==source);
            // A full ring with an unconsumed suffix must cancel without needing
            // consumer events that were never recorded. Then restart safely.
            for(int repeat=0;repeat<20;++repeat) {
                pipeline.start({{source.data(),size,false},{source.data(),size,false}});
                check(pipeline.transfer(dest,source.data(),size,compute));
                pipeline.finish();
                pipeline.start({{source.data(),size,false}});
                check(pipeline.transfer(dest,source.data(),size,compute));
                pipeline.finish();cuda_ok(cudaStreamSynchronize(compute));
                cuda_ok(cudaMemcpy(actual.data(),dest,size,cudaMemcpyDeviceToHost));check(actual==source);
            }
            const auto stats=pipeline.counters();
            check(stats.queued_bytes==0 && stats.reader_owned_bytes==0 && stats.queued_peak==4*chunk &&
                  stats.wait_us==stats.slot_wait_us+stats.consumer_wait_us && stats.unused_bytes<=stats.h2d_bytes);
        }
        test_reuse(compute,dest,size);
#ifdef _WIN32
        test_file(compute,dest,source);
        test_borrow(compute,dest,source);
        test_late_retention(compute,dest,source);
        test_async_retention(compute,dest,source);
#endif
        cuda_ok(cudaStreamDestroy(compute));cuda_ok(cudaFree(dest));
        std::puts("Pipeline overlap, file queue, byte parity, ordered reuse and cancellation passed");
        return 0;
    } catch(const std::exception &e) {std::fprintf(stderr,"%s\n",e.what());return 1;}
}
