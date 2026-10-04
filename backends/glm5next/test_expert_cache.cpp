#include "expert_cache.hpp"
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
static void test_pending_events() {
    Stream upload,consumer;ExpertCache cache(64);auto a=key(),b=a;++b.expert;
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
    std::puts("PASS: ready wait across streams, source ownership, pending consumer events and nonblocking bypass");
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
int main() {
    try {
        cudaDeviceProp p{};cuda_ok(cudaGetDeviceProperties(&p,0));
        std::printf("GPU=%s; synthetic cache fixtures, no GLM inference\n",p.name);
        test_keys();test_lru_budget();test_pending_events();test_reload_and_failure();test_pipeline_upload();
        return 0;
    }catch(const std::exception &e) {std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
}
