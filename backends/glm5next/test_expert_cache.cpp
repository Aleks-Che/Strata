#include "expert_cache.hpp"
#include "expert_memory.hpp"
#include "../common/expert_pipeline.hpp"
#include <cstdio>
#include <functional>
#include <optional>

using namespace strata_glm;
static void require(bool ok,const char *message) {if(!ok)throw std::runtime_error(message);}
static void cuda_ok(cudaError_t e) {if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Stream {
    cudaStream_t value=nullptr;
    Stream() {cuda_ok(cudaStreamCreateWithFlags(&value,cudaStreamNonBlocking));}
    ~Stream() {cudaStreamSynchronize(value);cudaStreamDestroy(value);}
    void sync() {cuda_ok(cudaStreamSynchronize(value));}
};
static ExpertKey key() {return {"model-A",1,Branch::main,3,7,Projection::gate,"IQ2_S",4096,2048,"part.gguf",1024,64};}
static ExpertCache::Lease load(ExpertCache &cache,const ExpertKey &k,uint8_t value,cudaStream_t stream,int *uploads=nullptr) {
    auto source=std::make_shared<std::vector<uint8_t>>(size_t(k.bytes),value);
    return cache.get(k,source,stream,[source,uploads](void *dest,size_t bytes,cudaStream_t s) {
        if(uploads)++*uploads;
        cuda_ok(cudaMemcpyAsync(dest,source->data(),bytes,cudaMemcpyHostToDevice,s));
    });
}
static void bytes_equal(ExpertCache::Lease &lease,size_t size,uint8_t expected,Stream &stream) {
    require(bool(lease),"unexpected cache bypass");stream.sync();
    std::vector<uint8_t> actual(size);
    cuda_ok(cudaMemcpy(actual.data(),lease.data(),size,cudaMemcpyDeviceToHost));
    require(std::all_of(actual.begin(),actual.end(),[&](uint8_t x){return x==expected;}),"cached bytes differ");
}

static void test_keys() {
    const auto base=key();
    std::vector<ExpertKey> keys{base};
    auto change=[&](const std::function<void(ExpertKey&)> &f) {auto k=base;f(k);keys.push_back(k);};
    change([](auto &k){k.model="model-B";});
    change([](auto &k){++k.generation;});
    change([](auto &k){k.branch=Branch::mtp;});
    change([](auto &k){++k.layer;});
    change([](auto &k){++k.expert;});
    change([](auto &k){k.projection=Projection::down;});
    change([](auto &k){k.quant="Q3_K";});
    change([](auto &k){++k.columns;});
    change([](auto &k){++k.rows;});
    change([](auto &k){k.shard="another.gguf";});
    change([](auto &k){k.offset+=(uint64_t(1)<<35);});
    change([](auto &k){++k.bytes;});
    ExpertCache cache(4096);Stream stream;int uploads=0;
    for(size_t i=0;i<keys.size();++i) {
        auto lease=load(cache,keys[i],uint8_t(i+19),stream.value,&uploads);
        bytes_equal(lease,size_t(keys[i].bytes),uint8_t(i+19),stream);
    }
    for(size_t i=0;i<keys.size();++i) {
        auto lease=load(cache,keys[i],0,stream.value,&uploads);
        bytes_equal(lease,size_t(keys[i].bytes),uint8_t(i+19),stream);
    }
    require(uploads==int(keys.size()) && cache.counters().hits==keys.size(),"cache key collision or repeated upload");
    require(cache.invalidate("model-A",1)==keys.size()-2,"invalidation must match model and generation");
    require(cache.size()==2,"invalidation removed another model/generation");
    for(auto mutate:std::vector<std::function<void(ExpertKey&)>>{
        [](auto &k){k.model.clear();},[](auto &k){k.shard.clear();},[](auto &k){k.quant.clear();},
        [](auto &k){k.layer=-1;},[](auto &k){k.expert=-1;},[](auto &k){k.bytes=0;},
        [](auto &k){k.offset=UINT64_MAX;},[](auto &k){k.columns=0;},[](auto &k){k.rows=0;},
        [](auto &k){k.branch=Branch(8);},[](auto &k){k.projection=Projection(8);}}) {
        auto invalid=base;mutate(invalid);bool rejected=false;
        try {invalid.validate();}catch(const std::invalid_argument &) {rejected=true;}
        require(rejected,"invalid cache key accepted");
    }
    std::puts("PASS: all 12 key fields, hit byte parity, model/generation invalidation and malformed keys");
}

static void test_lru_budget() {
    Stream stream;ExpertCache cache(128);auto a=key(),b=a,c=a;b.expert++;c.expert+=2;
    {auto lease=load(cache,a,1,stream.value);}stream.sync();
    {auto lease=load(cache,b,2,stream.value);}stream.sync();
    {auto lease=load(cache,a,9,stream.value);bytes_equal(lease,64,1,stream);}stream.sync();
    {auto lease=load(cache,c,3,stream.value);}stream.sync(); // B is LRU.
    int uploads=0;
    {auto lease=load(cache,a,9,stream.value,&uploads);bytes_equal(lease,64,1,stream);}stream.sync();
    require(uploads==0,"recent entry was evicted");
    {auto lease=load(cache,b,2,stream.value,&uploads);}stream.sync();
    require(uploads==1 && cache.counters().evictions==2,"LRU eviction order");
    auto held=load(cache,b,0,stream.value);
    require(!cache.set_budget(0) && cache.resident_bytes()==64,"leased entry was freed during trim");
    auto bypass=load(cache,a,9,stream.value);
    require(!bypass && cache.resident_bytes()==64,"over-budget cache allocated");
    held.release();stream.sync();
    require(cache.set_budget(0) && cache.resident_bytes()==0,"deferred trim failed");
    require(cache.counters().bypasses==1,"bypass counter");
    std::puts("PASS: LRU, budget shrink, plan leases, bypass and deferred trim");
}

struct Gate {
    std::atomic<bool> release{false},expired{false};
    cudaStream_t stream;
    explicit Gate(cudaStream_t s):stream(s) {}
    ~Gate() {release.store(true);cudaStreamSynchronize(stream);}
};
static void CUDART_CB block(void *ptr) {
    auto &gate=*static_cast<Gate *>(ptr);
    auto end=std::chrono::steady_clock::now()+std::chrono::seconds(3);
    while(!gate.release.load()) {
        if(std::chrono::steady_clock::now()>end) {gate.expired.store(true);break;}
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}
static void test_pending_events(bool frequency=false) {
    Stream upload,consumer;ExpertCache cache(64,{frequency,100000,32});auto a=key(),b=a;++b.expert;
    void *result=nullptr;cuda_ok(cudaMalloc(&result,64));
    Gate gate(upload.value);
    auto owner=std::make_shared<int>(42);std::weak_ptr<int> weak=owner;
    auto first=cache.get(a,owner,upload.value,[&](void *dest,size_t bytes,cudaStream_t s) {
        cuda_ok(cudaLaunchHostFunc(s,block,&gate));
        cuda_ok(cudaMemsetAsync(dest,0x63,bytes,s));
    });
    owner.reset();require(!weak.expired(),"source released while upload pending");
    bool called=false;
    auto hit=cache.get(a,nullptr,consumer.value,[&](void *,size_t,cudaStream_t){called=true;});
    cuda_ok(cudaMemcpyAsync(result,hit.data(),64,cudaMemcpyDeviceToDevice,consumer.value));
    first.release();hit.release(); // Both are released; GPU operations are still pending.
    auto bypass=load(cache,b,0x29,consumer.value);
    require(!called && !bypass && cache.resident_bytes()==64,"pending entry evicted or uploaded twice");
    require(!gate.expired.load(),"cache operation blocked on pending upload/consumer");
    gate.release.store(true);upload.sync();consumer.sync();
    std::vector<uint8_t> actual(64);cuda_ok(cudaMemcpy(actual.data(),result,64,cudaMemcpyDeviceToHost));
    cuda_ok(cudaFree(result));
    require(std::all_of(actual.begin(),actual.end(),[](uint8_t x){return x==0x63;}),"hit ignored ready event");
    {auto replacement=load(cache,b,0x29,consumer.value);bytes_equal(replacement,64,0x29,consumer);}
    require(weak.expired(),"evicted entry retains source");

    // Ready upload, but a different stream still consumes it: protect by its
    // use event, rather than only the upload event or active host leases.
    consumer.sync();Gate reading(consumer.value);
    auto reading_lease=load(cache,b,0,consumer.value);
    cuda_ok(cudaLaunchHostFunc(consumer.value,block,&reading));
    reading_lease.release();
    auto again=load(cache,a,1,upload.value);
    require(!again && !reading.expired.load(),"pending consumer was not protected");
    reading.release.store(true);consumer.sync();
    std::printf("PASS: ready wait across streams, source ownership, pending consumer events and nonblocking bypass (%s)\n",
                frequency?"frequency":"lru");
}

static void test_reload_and_failure() {
    Stream stream;auto a=key(),next=a;++next.generation;
    auto cache=std::make_unique<ExpertCache>(64);
    auto old=load(*cache,a,0x37,stream.value);
    require(cache->invalidate(a.model,a.generation)==1 && cache->size()==0 && cache->resident_bytes()==64,
            "invalidated live lease lost accounting");
    auto blocked=load(*cache,next,0x91,stream.value);
    require(!blocked,"reload overcommitted live old generation");
    bytes_equal(old,64,0x37,stream);old.release();stream.sync();
    require(cache->resident_bytes()==0,"old generation allocation leaked");
    auto fresh=load(*cache,next,0x91,stream.value);
    bytes_equal(fresh,64,0x91,stream);
    cache.reset(); // A plan lease keeps allocation/source/accounting alive.
    bytes_equal(fresh,64,0x91,stream);fresh.release();stream.sync();

    ExpertCache retry(64);auto source=std::make_shared<int>(0);std::weak_ptr<int> weak=source;
    bool caught=false;
    try {
        retry.get(a,source,stream.value,[](void *dest,size_t n,cudaStream_t s) {
            cuda_ok(cudaMemsetAsync(dest,0xE1,n,s));throw std::runtime_error("injected upload failure");
        });
    }catch(const std::runtime_error &) {caught=true;}
    source.reset();
    require(caught && retry.size()==0 && retry.resident_bytes()==0 && weak.expired(),"failed upload published or leaked");
    auto good=load(retry,a,0x24,stream.value);bytes_equal(good,64,0x24,stream);
    std::puts("PASS: reload generation, invalidated lease accounting, cache teardown and failed-upload recovery");
}

static void test_allocation_failure(bool frequency) {
    Stream stream;cudaError_t failure=cudaErrorMemoryAllocation;int allocations=0,uploads=0;
    ExpertCache cache(128,{frequency,100000,32},[&](void **ptr,size_t n) {
        ++allocations;return failure==cudaSuccess?cudaMalloc(ptr,n):failure;
    });
    auto a=key(),b=a,c=a;++b.expert;c.expert+=2;
    auto source=std::make_shared<int>(42);std::weak_ptr<int> weak=source;
    auto bypass=cache.get(a,source,stream.value,[&](void *,size_t,cudaStream_t){++uploads;});
    source.reset();
    require(!bypass && weak.expired() && uploads==0 && cache.size()==0 && cache.resident_bytes()==0,
            "failed allocation retained source, uploaded or admitted");
    require(cache.counters().allocation_bypasses==1 && cache.counters().bypasses==1 &&
            cache.counters().admissions==0,"allocation bypass counters");
    failure=cudaSuccess;
    {auto lease=load(cache,a,0x31,stream.value,&uploads);bytes_equal(lease,64,0x31,stream);}stream.sync();
    {auto lease=load(cache,b,0x52,stream.value,&uploads);}stream.sync();
    {
        auto pins=cache.protect_plan({a});failure=cudaErrorMemoryAllocation;
        auto missing=load(cache,c,0x73,stream.value,&uploads);
        require(!missing && cache.resident(a) && !cache.resident(b) && !cache.resident(c) &&
                cache.resident_bytes()==64 && cache.counters().evictions==1,
                "allocation failure corrupted pins, residency or eviction accounting");
        const auto attempts=allocations;
        {auto hit=load(cache,a,0xFF,stream.value,&uploads);bytes_equal(hit,64,0x31,stream);}stream.sync();
        require(allocations==attempts && uploads==2,"hit allocated or uploaded during OOM");
        failure=cudaSuccess;
        {auto recovered=load(cache,c,0x73,stream.value,&uploads);bytes_equal(recovered,64,0x73,stream);}stream.sync();
    }
    require(cache.counters().allocation_bypasses==2 && cache.counters().bypasses==2 && uploads==3,
            "OOM recovery counters");
    require(cache.set_budget(0),"allocation fixture could not trim");cache.set_budget(128);
    failure=cudaErrorInvalidValue;bool caught=false;
    try {auto lease=load(cache,b,0,stream.value,&uploads);}
    catch(const std::runtime_error &e) {caught=std::string(e.what())==cudaGetErrorString(failure);}
    require(caught && uploads==3 && cache.size()==0 && cache.resident_bytes()==0 &&
            cache.counters().allocation_bypasses==2,"non-OOM allocation error swallowed");
    failure=cudaSuccess;caught=false;
    try {
        cache.get(b,std::make_shared<int>(1),stream.value,[](void *dest,size_t n,cudaStream_t s) {
            cuda_ok(cudaMemsetAsync(dest,0x99,n,s));
            throw std::runtime_error(cudaGetErrorString(cudaErrorMemoryAllocation));
        });
    }catch(const std::runtime_error &) {caught=true;}
    require(caught && cache.size()==0 && cache.resident_bytes()==0 &&
            cache.counters().allocation_bypasses==2,"upload error misclassified as allocation bypass");
    {auto recovered=load(cache,b,0x94,stream.value);bytes_equal(recovered,64,0x94,stream);}
    std::printf("PASS: injected cache allocation OOM, pins/eviction, no upload/leak, recovery and fatal error propagation (%s)\n",
                frequency?"frequency":"lru");
}

static void test_branch_budgets(bool frequency) {
    Stream stream;ExpertCache cache(256,{frequency,100000,32,true});
    require(cache.set_branch_budgets(128,64),"initial branch limits");
    auto main=key(),mtp=main,next=mtp,large=mtp;
    mtp.branch=next.branch=large.branch=Branch::mtp;next.expert++;large.expert+=2;large.bytes=128;
    {auto lease=load(cache,main,0x12,stream.value);}stream.sync();
    {auto lease=load(cache,mtp,0x34,stream.value);}stream.sync();
    // Global budget has room: only the MTP ceiling requires eviction.
    {auto lease=load(cache,next,0x56,stream.value);bytes_equal(lease,64,0x56,stream);}stream.sync();
    require(cache.resident(main) && !cache.resident(mtp) && cache.resident(next) &&
            cache.resident_bytes(Branch::main)==64 && cache.resident_bytes(Branch::mtp)==64 &&
            cache.resident_bytes()==128 && cache.counters().evictions==1,"MTP quota evicted main or exceeded cap");
    auto pins=cache.protect_plan({next});
    require(!cache.set_branch_budgets(128,0) && cache.resident_bytes(Branch::mtp)==64,"quota trim freed pinned MTP");
    int uploads=0;
    auto bypass=load(cache,mtp,0,stream.value,&uploads);
    require(!bypass && uploads==0 && cache.resident(main),"disabled MTP cache uploaded/evicted main");
    {auto hit=load(cache,next,0,stream.value);bytes_equal(hit,64,0x56,stream);}stream.sync();
    pins.release();
    require(cache.set_branch_budgets(128,0) && cache.resident_bytes(Branch::mtp)==0 &&
            cache.resident_bytes()==64,"released MTP pin did not trim");
    cache.set_branch_budgets(128,64);
    auto oversized=load(cache,large,0,stream.value,&uploads);
    require(!oversized && uploads==0 && cache.resident(main),"oversized branch candidate changed cache");
    auto held=load(cache,next,0x78,stream.value);bytes_equal(held,64,0x78,stream);
    require(cache.invalidate(main.model,main.generation)==2 && cache.resident_bytes()==64 &&
            cache.resident_bytes(Branch::mtp)==64 && cache.resident_bytes(Branch::main)==0,
            "retired MTP lease lost branch/global accounting");
    auto reloaded=next;++reloaded.generation;
    auto blocked=load(cache,reloaded,0,stream.value);
    require(!blocked,"reload exceeded MTP cap with a retired lease");
    held.release();stream.sync();
    require(cache.resident_bytes()==0 && cache.resident_bytes(Branch::mtp)==0,"retired MTP bytes leaked");
    {auto lease=load(cache,reloaded,0x9A,stream.value);bytes_equal(lease,64,0x9A,stream);}stream.sync();
    require(cache.set_budget(32) && cache.resident_bytes()==0 && cache.byte_budget(Branch::mtp)==64,
            "global reserve trim ignored branch cache");
    require(!load(cache,main,0,stream.value),"branch cap bypassed smaller global budget");
    bool rejected=false;
    try {cache.resident_bytes(Branch(7));}catch(const std::invalid_argument &) {rejected=true;}
    require(rejected,"invalid branch accepted");
    std::printf("PASS: main/MTP caps, branch-local eviction, pins, disabled/oversized bypass, retired reload and global reserve (%s)\n",
                frequency?"frequency":"lru");
}

static void test_branch_frequency() {
    Stream stream;ExpertCache cache(128,{true,8,32,true});cache.set_branch_budgets(64,0);
    auto hot=key(),cold=hot,mtp=hot;++cold.expert;mtp.branch=Branch::mtp;
    for(int i=0;i<4;++i) {auto lease=load(cache,hot,0x25,stream.value);}stream.sync();
    for(int i=0;i<32;++i)require(!load(cache,mtp,0,stream.value),"disabled MTP cache admitted");
    require(!load(cache,cold,0,stream.value) && cache.resident(hot),"MTP accesses aged main frequency history");
    require(cache.counters().admission_rejects==1 && cache.history_size()==3,"separate branch history counters");
    cache.set_branch_budgets(64,64);cache.set_budget(64);
    // Under global pressure, independently aged histories use cross-branch LRU.
    {auto lease=load(cache,mtp,0x67,stream.value);bytes_equal(lease,64,0x67,stream);}stream.sync();
    require(!cache.resident(hot) && cache.resident(mtp) && cache.resident_bytes()==64,"cross-branch global LRU failed");
    require(cache.invalidate(hot.model,hot.generation)==1 && cache.history_size()==0,"branch history invalidation leaked");

    ExpertCache atomic(192,{true,100000,32,true});atomic.set_branch_budgets(128,64);
    auto a=key(),b=a,large=a;large.bytes=128;b.expert++;large.expert+=2;
    {auto lease=load(atomic,a,1,stream.value);}stream.sync();
    for(int i=0;i<4;++i) {auto lease=load(atomic,b,2,stream.value);}stream.sync();
    {auto lease=load(atomic,mtp,3,stream.value);}stream.sync();
    require(!load(atomic,large,4,stream.value) && atomic.size()==3 && atomic.counters().evictions==0,
            "quota frequency rejection partially evicted selected victims");
    std::puts("PASS: independent main/MTP decay, combined global LRU, history invalidation and atomic quota admission");
}

static void test_branch_pending() {
    Stream stream;ExpertCache cache(128);cache.set_branch_budgets(128,64);
    auto mtp=key();mtp.branch=Branch::mtp;Gate gate(stream.value);
    auto lease=cache.get(mtp,std::make_shared<int>(1),stream.value,[&](void *dest,size_t n,cudaStream_t s) {
        cuda_ok(cudaLaunchHostFunc(s,block,&gate));cuda_ok(cudaMemsetAsync(dest,0x64,n,s));
    });
    lease.release(); // No host lease/pin remains, only pending CUDA work.
    require(!cache.set_branch_budgets(128,0) && cache.resident_bytes(Branch::mtp)==64 && !gate.expired.load(),
            "branch trim freed or blocked on pending CUDA entry");
    gate.release.store(true);stream.sync();
    require(cache.set_branch_budgets(128,0) && cache.resident_bytes()==0,"completed branch entry did not trim");
    {auto main=load(cache,key(),0x42,stream.value);bytes_equal(main,64,0x42,stream);}
    std::puts("PASS: branch trim defers pending CUDA work, then releases bytes without disabling main");
}

static void test_pipeline_upload() {
    Stream stream;constexpr size_t n=(1<<20)+17;
    ExpertCache cache(n);auto k=key();k.bytes=n;
    auto source=std::make_shared<std::vector<uint8_t>>(n);
    for(size_t i=0;i<n;++i)(*source)[i]=uint8_t(i*13+(i>>8));
    StrataExpertPipeline pipeline(0,65553,false);
    pipeline.start({{source->data(),n,false}});
    auto lease=cache.get(k,source,stream.value,[&](void *dest,size_t bytes,cudaStream_t s) {
        require(pipeline.transfer(dest,source->data(),bytes,s),"pipeline fallback");
    });
    pipeline.finish();lease.release();stream.sync();
    auto hit=cache.get(k,nullptr,stream.value,[](void *,size_t,cudaStream_t){throw std::runtime_error("cache hit reloaded");});
    stream.sync();std::vector<uint8_t> actual(n);
    cuda_ok(cudaMemcpy(actual.data(),hit.data(),n,cudaMemcpyDeviceToHost));
    require(actual==*source && pipeline.counters().h2d_bytes==n && cache.counters().hits==1,"pipeline/cache byte parity");
    std::puts("PASS: common pipeline upload, partial chunks, cache hit parity and no second H2D");
}

static void test_frequency_history() {
    StrataExpertFrequencyHistory<ExpertKey,ExpertKeyHash> history(100000,32);
    auto base=key();
    history.record(base);history.record(base);
    for(auto mutate:std::vector<std::function<void(ExpertKey&)>>{
        [](auto &k){k.model="other";},[](auto &k){++k.generation;},
        [](auto &k){k.branch=Branch::mtp;},[](auto &k){++k.layer;},
        [](auto &k){++k.expert;},[](auto &k){k.projection=Projection::up;},
        [](auto &k){k.quant="Q3_K";},[](auto &k){++k.columns;},
        [](auto &k){++k.rows;},[](auto &k){k.shard="other";},
        [](auto &k){k.offset+=uint64_t(1)<<35;},[](auto &k){++k.bytes;}}) {
        auto changed=base;mutate(changed);
        require(history.score(changed)==0,"frequency history aliases a key field");
        history.record(changed);
        require(history.score(changed)==1 && history.score(base)==2,"frequency counts mixed");
    }
    require(history.size()==13,"full-key history size");
    history.erase_if([&](const ExpertKey &k){return k.model==base.model && k.generation==base.generation;});
    auto other=base;other.model="other";
    auto next=base;++next.generation;
    require(history.size()==2 && history.score(base)==0 && history.score(other)==1 && history.score(next)==1,
            "history invalidation mixed identities");
    std::puts("PASS: frequency history separates all 12 key fields and invalidates only one model/generation");
}

static void test_frequency_admission() {
    Stream stream;auto hot=key(),cold=hot;++cold.expert;
    ExpertCache cache(64,{true,100000,32});int uploads=0;
    for(int i=0;i<4;++i) {
        auto lease=load(cache,hot,0x31,stream.value,&uploads);
        bytes_equal(lease,64,0x31,stream);
    }
    stream.sync();
    for(int i=0;i<3;++i) {
        auto rejected=load(cache,cold,0x72,stream.value,&uploads);
        require(!rejected && uploads==1 && cache.resident_bytes()==64,"cold miss displaced hot entry");
    }
    // Rejected accesses still count. Ties admit and the old entry is LRU.
    {auto admitted=load(cache,cold,0x72,stream.value,&uploads);bytes_equal(admitted,64,0x72,stream);}
    stream.sync();
    auto c=cache.counters();
    require(uploads==2 && c.admissions==2 && c.admission_rejects==3 && c.bypasses==3 &&
            c.hits==3 && c.misses==5 && c.evictions==1,"frequency admission counters");
    require(cache.set_budget(0) && cache.resident_bytes()==0,"frequency blocked explicit budget trim");
    require(cache.history_size()==2 && cache.invalidate(hot.model,hot.generation)==0 && cache.history_size()==0,
            "invalidation failed to erase nonresident history");

    ExpertCache decay(64,{true,8,32});
    for(int i=0;i<7;++i) {auto lease=load(decay,hot,1,stream.value);stream.sync();}
    stream.sync();
    {auto rejected=load(decay,cold,2,stream.value);require(!rejected,"decay admitted too early");}
    {auto rejected=load(decay,cold,2,stream.value);require(!rejected,"decay admitted too early");}
    {auto admitted=load(decay,cold,2,stream.value);bytes_equal(admitted,64,2,stream);}
    // At access 8 the untouched hot score halves from 7 to 3; cold needs only 3.
    require(decay.counters().admission_rejects==2,"lazy decay did not age untouched resident");

    ExpertCache bounded(0,{true,100000,2});
    for(int i=0;i<5;++i) {auto k=hot;k.expert+=i;auto lease=load(bounded,k,0,stream.value);require(!lease,"zero budget");}
    require(bounded.history_size()==1 && bounded.counters().bypasses==5,"history key bound");
    std::puts("PASS: hot protection, repeat-miss admission, ties, counters, decay, history bound and reload cleanup");
}

static void test_frequency_mixed_sizes() {
    Stream stream;auto cold=key(),hot=cold,large=cold;++hot.expert;large.expert+=2;large.bytes=128;
    ExpertCache cache(128,{true,100000,32});
    {auto lease=load(cache,cold,0x21,stream.value);}stream.sync();
    for(int i=0;i<4;++i) {auto lease=load(cache,hot,0x63,stream.value);}stream.sync();
    int uploads=0;
    {auto rejected=load(cache,large,0x45,stream.value,&uploads);require(!rejected,"large cold candidate admitted");}
    require(uploads==0 && cache.size()==2 && cache.resident_bytes()==128 && cache.counters().evictions==0,
            "rejected mixed-size candidate partially evicted cache");
    auto held=load(cache,hot,0,stream.value);
    for(int i=0;i<5;++i) {auto rejected=load(cache,large,0x45,stream.value);require(!rejected,"leased victim evicted");}
    require(cache.size()==2 && cache.counters().evictions==0,"insufficient idle bytes partially evicted cache");
    bytes_equal(held,64,0x63,stream);held.release();stream.sync();
    {auto admitted=load(cache,large,0x45,stream.value,&uploads);bytes_equal(admitted,128,0x45,stream);}
    stream.sync();
    require(uploads==1 && cache.size()==1 && cache.counters().evictions==2,"multi-victim admission failed");
    std::puts("PASS: mixed-size admission is atomic, protects leases and replaces multiple idle victims");
}

static void test_frequency_pipeline_bypass() {
    Stream stream;constexpr size_t n=(1<<20)+17;
    auto hot=key(),cold=hot;hot.bytes=cold.bytes=n;++cold.expert;
    ExpertCache cache(n,{true,100000,32});
    for(int i=0;i<3;++i) {auto lease=load(cache,hot,0x18,stream.value);}stream.sync();
    auto source=std::make_shared<std::vector<uint8_t>>(n,0xA7);
    auto rejected=cache.get(cold,source,stream.value,[](void *,size_t,cudaStream_t) {
        throw std::runtime_error("rejected candidate must use uncached fallback");
    });
    require(!rejected,"cold matrix unexpectedly cached");
    void *dest=nullptr;cuda_ok(cudaMalloc(&dest,n));
    StrataExpertPipeline pipeline(0,65553,false);
    pipeline.start({{source->data(),n,false}});
    require(pipeline.transfer(dest,source->data(),n,stream.value),"uncached pipeline fallback missing");
    pipeline.finish();stream.sync();
    std::vector<uint8_t> actual(n);cuda_ok(cudaMemcpy(actual.data(),dest,n,cudaMemcpyDeviceToHost));
    cuda_ok(cudaFree(dest));
    require(actual==*source && pipeline.counters().h2d_bytes==n,"bypass lost selected matrix bytes");
    auto hit=cache.get(hot,nullptr,stream.value,[](void *,size_t,cudaStream_t) {
        throw std::runtime_error("hot matrix reuploaded after cold bypass");
    });
    bytes_equal(hit,n,0x18,stream);
    std::puts("PASS: admission bypass still delivers full matrix via pipeline; hot hit keeps original bytes");
}
static void test_plan_pins(bool frequency) {
    Stream stream;ExpertCache cache(128,{frequency,100000,32});
    auto a=key(),b=a,c=a,d=a;++b.expert;c.expert+=2;d.expert+=3;
    {auto lease=load(cache,a,0x11,stream.value);}stream.sync();
    {auto lease=load(cache,b,0x22,stream.value);}stream.sync();
    const auto before=cache.counters();const auto history=cache.history_size();
    auto pins=cache.protect_plan({c,a,a}); // The first route misses; the later A hit is LRU.
    require(pins.size()==1 && cache.counters().hits==before.hits && cache.counters().misses==before.misses &&
            cache.history_size()==history,"plan protection changed accesses or pinned duplicate/absent keys");
    int uploads=0;
    auto fresh=load(cache,c,0x33,stream.value,&uploads); // Must evict B, not the later planned A.
    {auto later=load(cache,a,0xFF,stream.value,&uploads);bytes_equal(later,64,0x11,stream);}stream.sync();
    require(uploads==1 && cache.counters().evictions==1,"earlier miss displaced a later planned hit");
    auto bypass=load(cache,d,0x44,stream.value,&uploads);
    require(!bypass && uploads==1,"plan pin/newly loaded lease did not protect current plan");
    require(!cache.set_budget(0) && cache.resident_bytes()==128,"budget shrink freed protected plan entries");
    fresh.release();stream.sync();pins.release();pins.release();
    require(cache.set_budget(0) && cache.resident_bytes()==0,"plan cancellation did not release protection");
    std::printf("PASS: plan-wide pins protect later hits, deduplicate, preserve counters and cooperate with miss leases/trim (%s)\n",
                frequency?"frequency":"lru");
}

static void test_pin_reload_and_teardown() {
    Stream stream;auto a=key(),next=a;++next.generation;
    auto owner=std::make_shared<std::vector<uint8_t>>(64,0x61);std::weak_ptr<std::vector<uint8_t>> weak=owner;
    auto cache=std::make_unique<ExpertCache>(64);
    {auto lease=cache->get(a,owner,stream.value,[&](void *dest,size_t n,cudaStream_t s) {
        cuda_ok(cudaMemcpyAsync(dest,owner->data(),n,cudaMemcpyHostToDevice,s));
    });}stream.sync();owner.reset();
    auto pins=cache->protect_plan({a});auto second=cache->protect_plan({a});
    require(cache->invalidate(a.model,a.generation)==1 && cache->size()==0 && cache->resident_bytes()==64,
            "invalidation lost pinned allocation accounting");
    auto blocked=load(*cache,next,0x72,stream.value);
    require(!blocked && !weak.expired(),"generation reload overcommitted pinned old weights");
    pins.release();require(cache->resident_bytes()==64 && !weak.expired(),"overlapping plan pins lost source");
    second.release();require(cache->resident_bytes()==0 && weak.expired(),"invalidated source leaked after last pin");
    {auto lease=load(*cache,next,0x72,stream.value);bytes_equal(lease,64,0x72,stream);}stream.sync();
    auto survives=cache->protect_plan({next});
    auto consumer=load(*cache,next,0,stream.value);cache.reset();
    bytes_equal(consumer,64,0x72,stream);consumer.release();stream.sync();survives.release();

    ExpertCache retry(64);
    {auto lease=load(retry,a,1,stream.value);}stream.sync();
    try {auto guard=retry.protect_plan({a});throw std::runtime_error("fixture cancellation");}
    catch(const std::runtime_error &) {}
    require(retry.set_budget(0) && retry.resident_bytes()==0,"exception unwinding retained pin");
    std::puts("PASS: overlapping pins, invalidation accounting, generation reload, cache teardown and exception cancellation");
}

static void test_pin_validation_and_lru() {
    Stream stream;auto a=key(),b=a,c=a;++b.expert;c.expert+=2;
    ExpertCache cache(128);
    {auto lease=load(cache,a,1,stream.value);}stream.sync();
    {auto lease=load(cache,b,2,stream.value);}stream.sync();
    auto invalid=a;invalid.bytes=0;bool rejected=false;
    try {auto bad=cache.protect_plan({a,invalid});}catch(const std::invalid_argument &) {rejected=true;}
    require(rejected,"invalid plan key accepted");
    rejected=false;
    try {auto big=cache.protect_plan(std::vector<ExpertKey>(12289,a));}catch(const std::invalid_argument &) {rejected=true;}
    require(rejected,"plan pin metadata limit ignored");
    {auto empty=cache.protect_plan({});require(empty.size()==0,"empty pin plan");}
    {auto guard=cache.protect_plan({a});} // Protection alone must not touch LRU.
    {auto lease=load(cache,c,3,stream.value);}stream.sync();
    auto remaining=cache.protect_plan({a,b,c});
    require(remaining.size()==2 && cache.counters().evictions==1,"validation leaked a pin");
    int uploads=0;
    {auto b_hit=load(cache,b,0,stream.value,&uploads);bytes_equal(b_hit,64,2,stream);}
    require(uploads==0,"plan probe changed LRU order");
    std::puts("PASS: plan validation/limit/empty plan and protection without LRU changes");
}

static void test_pins_do_not_wait_for_upload() {
    Stream stream;ExpertCache cache(64);auto a=key();Gate gate(stream.value);
    auto owner=std::make_shared<int>(1);
    auto lease=cache.get(a,owner,stream.value,[&](void *dest,size_t n,cudaStream_t s) {
        cuda_ok(cudaLaunchHostFunc(s,block,&gate));cuda_ok(cudaMemsetAsync(dest,0x39,n,s));
    });
    lease.release();
    auto pins=cache.protect_plan({a});
    require(pins.size()==1 && !gate.expired.load(),"plan protection waited for pending upload");
    gate.release.store(true);stream.sync();
    auto hit=load(cache,a,0,stream.value);bytes_equal(hit,64,0x39,stream);
    std::puts("PASS: plan pins do not wait for upload; consumer lease preserves ready-event byte parity");
}

static void test_memory_controller(bool frequency) {
    constexpr size_t mib=1ULL<<20,reserve=128*mib;
    Stream stream;ExpertCache cache(128,{frequency,100000,32});
    auto a=key(),b=a,c=a;b.branch=Branch::mtp;b.layer=45;c.expert++;
    StrataVramPolicy policy;policy.reserve_mib=128;
    size_t free=reserve+1024,total=1024*mib;bool available=true,throws=false;
    ExpertMemoryController control(cache,128,policy,[&](int device,size_t &f,size_t &t) {
        require(device==cache.device(),"probe got another device");
        if(throws)throw std::runtime_error("fixture sample failure");
        f=free;t=total;return available;
    });
    require(!load(cache,a,1,stream.value),"unsampled controller admitted weights");
    auto state=control.refresh();
    require(state.sample_valid && state.target==128 && state.trim_complete,"configured cap ignored");
    {auto lease=load(cache,a,0x31,stream.value);}stream.sync();
    auto held=load(cache,b,0x42,stream.value);stream.sync();auto pin=cache.protect_plan({a});
    free=reserve-64;state=control.refresh();
    require(state.target==64 && state.resident==128 && state.deferred==64 && !state.trim_complete,
            "main/MTP pressure lost protected bytes or deferred trim accounting");
    require(!load(cache,c,0,stream.value),"pressure admitted a new matrix");
    bytes_equal(held,64,0x42,stream);
    require(cache.invalidate(a.model,a.generation)==2,"combined cache generation invalidation");
    state=control.refresh();
    require(state.resident==128 && state.deferred==64,"retired pinned/leased allocations omitted from budget");
    pin.release();free=reserve;state=control.refresh();
    require(state.resident==64 && state.target==64 && state.trim_complete,"deferred pin trim did not recover");
    held.release();stream.sync();
    require(cache.resident_bytes()==0,"retired lease did not release bytes");
    free=reserve+64;control.refresh();
    {auto lease=load(cache,a,0x53,stream.value);}stream.sync();
    const auto before=cache.counters();available=false;state=control.refresh();
    require(!state.sample_valid && state.failed_samples==1 && state.resident==64,"unavailable sample status");
    {auto hit=load(cache,a,0,stream.value);bytes_equal(hit,64,0x53,stream);}stream.sync();
    require(!load(cache,c,0,stream.value) && cache.counters().paused_bypasses==before.paused_bypasses+1 &&
            cache.counters().evictions==before.evictions,"sample failure evicted hit or admitted miss");
    available=true;
    for(int invalid=0;invalid<3;++invalid) {
        total=invalid==0?0:1024*mib;free=invalid==1?total+1:total;
        require(!control.refresh().sample_valid,"invalid/inconsistent global sample accepted");
    }
    throws=true;bool caught=false;
    try {control.refresh();}catch(const std::runtime_error &) {caught=true;}
    require(caught && !control.status().sample_valid && !load(cache,c,0,stream.value),"probe exception reopened admissions");
    throws=false;total=1024*mib;free=reserve;
    require(control.refresh().sample_valid,"fresh sample did not recover controller");
    free=reserve-64;state=control.refresh();
    require(state.target==0 && state.trim_complete && state.resident==0,"reserve pressure did not trim idle weights");
    free=reserve+128;state=control.refresh();
    {auto lease=load(cache,c,0x64,stream.value);bytes_equal(lease,64,0x64,stream);}stream.sync();
    require(state.target==128 && cache.counters().paused_bypasses==3,"recovery or pause accounting");
    std::printf("PASS: reserve cap, main/MTP, pins/retired leases, unavailable/malformed/throwing samples, trim and recovery (%s)\n",
                frequency?"frequency":"lru");
}

static void test_memory_policy_and_pending() {
    constexpr size_t mib=1ULL<<20;
    Stream stream;ExpertCache cache(64);auto a=key();
    StrataVramPolicy policy;policy.reserve_mib=128;policy.mode=2;policy.target_mib=256;
    size_t free=768*mib+64;
    ExpertMemoryController control(cache,128,policy,[&](int,size_t &f,size_t &t){f=free;t=1024*mib;return true;});
    require(control.refresh().target==64,"total-device usage target ignored");
    Gate gate(stream.value);auto owner=std::make_shared<int>(1);
    auto lease=cache.get(a,owner,stream.value,[&](void *dest,size_t n,cudaStream_t s) {
        cuda_ok(cudaLaunchHostFunc(s,block,&gate));cuda_ok(cudaMemsetAsync(dest,0x75,n,s));
    });
    lease.release();free=768*mib-64;
    auto state=control.refresh();
    require(state.target==0 && state.deferred==64 && !state.trim_complete && !gate.expired.load(),
            "memory refresh waited for/freed pending GPU work");
    gate.release.store(true);stream.sync();
    require(control.refresh().resident==0,"pending allocation not trimmed after completion");
    for(auto bad:std::vector<StrataVramPolicy>{{1,1,0,128},{0,0,0,0},{2,0,0,128}}) {
        bool rejected=false;
        try {ExpertMemoryController invalid(cache,128,bad,[](int,size_t &,size_t &){return true;});}
        catch(const std::invalid_argument &) {rejected=true;}
        require(rejected,"invalid memory controller policy accepted");
    }
    bool rejected=false;
    try {ExpertMemoryController invalid(cache,128,policy,{});}catch(const std::invalid_argument &) {rejected=true;}
    require(rejected,"missing global probe accepted");
    std::puts("PASS: total-device usage target, pending events, invalid policies and missing probe");
}

int main() {
    try {
        cudaDeviceProp p{};cuda_ok(cudaGetDeviceProperties(&p,0));
        std::printf("GPU=%s; synthetic cache fixtures, no GLM inference\n",p.name);
        test_keys();test_lru_budget();test_pending_events();test_reload_and_failure();test_pipeline_upload();
        test_allocation_failure(false);test_allocation_failure(true);
        test_branch_budgets(false);test_branch_budgets(true);test_branch_frequency();test_branch_pending();
        test_frequency_history();test_frequency_admission();test_frequency_mixed_sizes();test_frequency_pipeline_bypass();
        test_pending_events(true);
        test_plan_pins(false);test_plan_pins(true);test_pin_reload_and_teardown();
        test_pin_validation_and_lru();test_pins_do_not_wait_for_upload();
        test_memory_controller(false);test_memory_controller(true);test_memory_policy_and_pending();
        return 0;
    }catch(const std::exception &e) {std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
}
