#include "expert_transport.hpp"
#include "expert_plan.hpp"
#include "expert_dispatch.hpp"
#include "expert_memory.hpp"
#include <cstdio>
#include <functional>

using namespace strata_glm;
static void require(bool ok,const char *message) {if(!ok)throw std::runtime_error(message);}
static void cuda_ok(cudaError_t e) {if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Device {
    uint8_t *data=nullptr;cudaStream_t stream=nullptr;size_t bytes;
    explicit Device(size_t n):bytes(n) {
        cuda_ok(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        auto e=cudaMalloc((void **)&data,n);
        if(e!=cudaSuccess) {cudaStreamDestroy(stream);cuda_ok(e);}
        cuda_ok(cudaMemset(data,0xA5,n));
    }
    ~Device() {cudaStreamSynchronize(stream);cudaFree(data);cudaStreamDestroy(stream);}
};
struct Source {
    std::vector<uint8_t> bytes;
    ExpertLayerLayout layout{3,45,46,3,{}};
    FILE *file=nullptr;HANDLE mapping=nullptr;uint8_t *view=nullptr;
    bool registered=false;
    Source() {
        constexpr uint64_t block[]={82,110,136};
        constexpr const char *quant[]={"IQ2_S","IQ3_S","IQ4_XS"};
        size_t end=96;
        for(size_t p=0;p<3;++p) {
            const uint64_t n=256*block[p]*10;
            layout.tensors[p]={Projection(p),quant[p],256,256,10,0,end,n};end+=size_t(n);
        }
        bytes.resize(end);
        for(size_t i=0;i<end;++i)bytes[i]=uint8_t(i*17+(i>>9));
        try {
            file=std::tmpfile();require(file!=nullptr,"fixture file");
            require(std::fwrite(bytes.data(),1,end,file)==end && std::fflush(file)==0,"fixture write");
            mapping=CreateFileMappingW((HANDLE)_get_osfhandle(_fileno(file)),nullptr,PAGE_READONLY,0,0,nullptr);
            require(mapping!=nullptr,"fixture mapping");
            view=(uint8_t *)MapViewOfFile(mapping,FILE_MAP_READ,0,0,0);require(view!=nullptr,"fixture view");
            strata_expert_file::add(view,bytes.size(),_fileno(file));registered=true;
        }catch(...) {close();throw;}
    }
    void close() {
        if(registered)strata_expert_file::remove(view);
        if(view)UnmapViewOfFile(view);
        if(mapping)CloseHandle(mapping);
        if(file)std::fclose(file);
    }
    ~Source() {close();}
    std::vector<ExpertKey> plan(uint64_t generation=1) {
        return plan_experts("model",generation,layout,{{"fixture.gguf",96,bytes.size()}},
                            {9,0,9,8,2,7,3,6,4,0});
    }
};
static std::vector<ExpertSourceView> views(const std::shared_ptr<Source> &s,uint64_t generation=1) {
    return {{"model",generation,"fixture.gguf",s->view,s->bytes.size(),s}};
}
static void wait_prefetch(ExpertTransport &transport) {
    auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(2);
    while(transport.counters().h2d_bytes<4*4093 && std::chrono::steady_clock::now()<deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    require(transport.counters().h2d_bytes>=4*4093,"prefetch did not fill ring before cancellation");
}
static void test_cancel_restart(int mode,bool decode,int readers=1,bool write_combined=false,bool early_host_refill=false) {
    ExpertTransport transport(0,4093,write_combined,4,mode,{},readers,early_host_refill);
    size_t comparisons=0;
    for(size_t prefix:{size_t(0),size_t(1),size_t(7),size_t(24)}) {
        auto source=std::make_shared<Source>();std::weak_ptr<Source> weak=source;
        auto plan=source->plan(prefix+1);auto sources=views(source,prefix+1);
        constexpr size_t guard=37;
        std::vector<size_t> offsets;size_t total=guard;
        for(const auto &key:plan) {offsets.push_back(total);total+=size_t(key.bytes)+guard;}
        Device dest(total);std::vector<uint8_t> expected(total,0xA5),actual(total);
        for(size_t i=0;i<prefix;++i)
            std::copy_n(source->bytes.data()+plan[i].offset,size_t(plan[i].bytes),expected.data()+offsets[i]);
        if(mode==1) {
            DWORD old=0;require(VirtualProtect(source->view,source->bytes.size(),PAGE_NOACCESS,&old)!=0,"protect mmap");
        }
        transport.begin(plan,sources,decode);
        sources.clear();source.reset(); // The adapter must own all remaining source lifetime.
        require(!weak.expired(),"source released with queued reads");
        if(prefix==0)wait_prefetch(transport);
        for(size_t i=0;i<prefix;++i)transport.transfer(i,dest.data+offsets[i],dest.stream);
        if(prefix==plan.size())transport.finish();else transport.cancel();
        require(!transport.in_progress() && transport.remaining()==0 && weak.expired(),"source not retired after drain");
        transport.cancel(); // Idempotent; next loop starts a new generation/source.
        cuda_ok(cudaStreamSynchronize(dest.stream));
        cuda_ok(cudaMemcpy(actual.data(),dest.data,total,cudaMemcpyDeviceToHost));
        require(actual==expected,"cancel/restart corrupted delivered prefix or untouched suffix guards");
        comparisons+=prefix;
    }
    const auto c=transport.counters();
    require(c.groups==4 && c.unused>0,"abandoned prefetch not accounted");
    require(c.read_peak>=1 && c.read_peak<=uint64_t(decode && mode!=1?readers:4),"decode reader bound ignored");
    if(mode==1 || (mode==2 && !decode))require(c.file_bytes>0 && c.mmap_bytes==0,"native path not used");
    else require(c.mmap_bytes>0 && c.file_bytes==0,"mmap path not used");
    std::printf("PASS mode=%d decode=%d readers=%d write_combined=%d early_host_refill=%d: 4 plans, %zu matrix comparisons, source release and restart\n",mode,int(decode),readers,int(write_combined),int(early_host_refill),comparisons);
}
static void test_destructor() {
    auto source=std::make_shared<Source>();std::weak_ptr<Source> weak=source;
    auto plan=source->plan();auto sources=views(source);
    try {
        ExpertTransport transport(0,4093,false,4,1);
        transport.begin(plan,sources,false);sources.clear();source.reset();
        wait_prefetch(transport);require(!weak.expired(),"source died during prefetch");
        throw std::runtime_error("graph fixture exception");
    }catch(const std::runtime_error &e) {
        require(std::string(e.what())=="graph fixture exception","unexpected destructor fixture failure");
    }
    require(weak.expired(),"exception teardown leaked source");
    std::puts("PASS: exception teardown joins workers and releases sole mapping owner");
}
static void test_validation_and_finish() {
    auto source=std::make_shared<Source>();auto plan=source->plan();auto good=views(source);
    ExpertTransport transport(0,4093,false,4,0);
    for(auto change:std::vector<std::function<void(std::vector<ExpertKey>&,std::vector<ExpertSourceView>&)>>{
        [](auto &p,auto &){p[0].generation++;},[](auto &p,auto &){p[0].model="other";},
        [](auto &p,auto &){p[0].shard="other.gguf";},[](auto &p,auto &){p.push_back(p[0]);},
        [](auto &p,auto &){p[0].offset=UINT64_MAX;},[](auto &p,auto &s){p[0].offset=s[0].bytes;},
        [](auto &,auto &s){s[0].owner.reset();},[](auto &,auto &s){s[0].data=nullptr;},
        [](auto &,auto &s){s.push_back(s[0]);},[](auto &,auto &s){s[0].bytes=SIZE_MAX;}}) {
        auto p=plan;auto s=good;change(p,s);bool rejected=false;
        try {transport.begin(p,s,false);}catch(const std::invalid_argument &) {rejected=true;}
        require(rejected && !transport.in_progress() && transport.counters().groups==0,"invalid binding submitted work");
    }
    bool limited=false;
    try {transport.begin(std::vector<ExpertKey>(12289,plan[0]),good,false);}
    catch(const std::invalid_argument &) {limited=true;}
    require(limited && transport.counters().groups==0,"matrix count bound ignored");
    {
        ExpertTransport small_chunks(0,1,false,1,0);
        auto big=plan;auto next=source->plan(2);big.insert(big.end(),next.begin(),next.end());
        auto two=good;two.push_back(views(source,2)[0]);limited=false;
        try {small_chunks.begin(big,two,false);}catch(const std::invalid_argument &) {limited=true;}
        require(limited && !small_chunks.in_progress() && small_chunks.counters().groups==0,"chunk metadata bound ignored");
    }
    transport.begin(plan,good,false);Device dest(size_t(plan[0].bytes));
    bool rejected=false;
    try {transport.begin({},good,false);}catch(const std::logic_error &) {rejected=true;}
    require(rejected,"nested begin accepted");
    rejected=false;
    try {transport.transfer(1,dest.data,dest.stream);}catch(const std::logic_error &) {rejected=true;}
    require(rejected && transport.remaining()==plan.size(),"out-of-order transfer consumed jobs");
    rejected=false;
    try {transport.transfer(0,nullptr,dest.stream);}catch(const std::logic_error &) {rejected=true;}
    require(rejected,"null destination accepted");
    transport.transfer(0,dest.data,dest.stream);
    rejected=false;
    try {transport.finish();}catch(const std::logic_error &) {rejected=true;}
    require(rejected && !transport.in_progress(),"incomplete finish silently succeeded or did not drain");
    transport.begin({},good,false);transport.finish();
    require(!transport.in_progress(),"empty plan could not finish");
    std::puts("PASS: binding validation before submission, matrix/chunk limits, model/generation isolation, ordered transfer, incomplete finish and empty restart");
}
static void test_dispatch(int mode,bool frequency,bool decode) {
    auto source=std::make_shared<Source>();auto all=source->plan();
    const auto a=all[0],b=all[1],c=all[2],d=all[3];
    cudaError_t next_allocation=cudaSuccess;int allocations=0;
    ExpertCache cache(size_t(a.bytes+c.bytes),{frequency,100000,32},[&](void **ptr,size_t bytes) {
        ++allocations;auto result=next_allocation;next_allocation=cudaSuccess;
        return result==cudaSuccess?cudaMalloc(ptr,bytes):result;
    });
    ExpertTransport transport(0,4093,false,4,mode);
    if(mode==1) {
        DWORD old=0;require(VirtualProtect(source->view,source->bytes.size(),PAGE_NOACCESS,&old)!=0,"dispatch native protection");
    }
    auto run=[&](const std::vector<ExpertKey> &plan,bool supply_sources) {
        constexpr size_t guard=37;size_t total=guard;
        std::vector<size_t> offsets;
        for(const auto &key:plan) {offsets.push_back(total);total+=size_t(key.bytes)+guard;}
        Device dest(total);std::vector<uint8_t> expected(total,0xA5),actual(total);
        for(size_t i=0;i<plan.size();++i)
            std::copy_n(source->bytes.data()+plan[i].offset,size_t(plan[i].bytes),expected.data()+offsets[i]);
        ExpertDispatch dispatch(cache,transport,plan,supply_sources?views(source):std::vector<ExpertSourceView>{},decode);
        for(size_t i=0;i<plan.size();++i)dispatch.copy(i,dest.data+offsets[i],dest.stream);
        dispatch.finish();require(dispatch.remaining()==0 && !transport.in_progress(),"dispatch did not finish miss plan");
        cuda_ok(cudaStreamSynchronize(dest.stream));
        cuda_ok(cudaMemcpy(actual.data(),dest.data,total,cudaMemcpyDeviceToHost));
        require(actual==expected,"dispatch hit/miss/bypass bytes or guards differ");
    };
    run({a,b},true);const auto before=transport.counters();const auto cached=cache.counters();
    run({c,a,d},true); // C evicts unplanned B; pinned A hits; D must bypass held C/A.
    const auto after=transport.counters();const auto counts=cache.counters();
    require(after.h2d_bytes-before.h2d_bytes==c.bytes+d.bytes &&
            after.file_bytes-before.file_bytes+after.mmap_bytes-before.mmap_bytes==c.bytes+d.bytes,
            "dispatch prefetched a cache hit or lost a bypass");
    require(counts.hits-cached.hits==1 && counts.misses-cached.misses==2 &&
            counts.admissions-cached.admissions==1 && counts.evictions-cached.evictions==1 &&
            counts.bypasses-cached.bypasses==1,"dispatch cache decisions differ from plan");
    run({c,a},false); // All hits require neither source views nor transport work.
    const auto hot=transport.counters();
    require(hot.h2d_bytes==after.h2d_bytes && hot.file_bytes==after.file_bytes && hot.mmap_bytes==after.mmap_bytes &&
            hot.groups==after.groups,"all-hit dispatch performed source/H2D work");
    if(frequency) {
        run({c,a},false); // Make both unpinned victims hotter than the bypassed D.
        const auto before_reject=cache.counters();const auto before_io=transport.counters();
        run({d},true);
        require(cache.counters().admission_rejects==before_reject.admission_rejects+1 &&
                transport.counters().h2d_bytes==before_io.h2d_bytes+d.bytes,
                "frequency-rejected matrix was not delivered via bypass");
    }
    {
        StrataVramPolicy policy;
        ExpertMemoryController controller(cache,size_t(a.bytes+c.bytes),policy,
                                         [](int,size_t &,size_t &){return false;});
        require(!controller.refresh().sample_valid,"fixture probe unexpectedly available");
        const auto before_pause=cache.counters();
        const auto io=transport.counters();
        run({d,a},true);
        require(cache.counters().paused_bypasses==before_pause.paused_bypasses+1 &&
                cache.counters().hits==before_pause.hits+1 && cache.resident(a) && cache.resident(c) &&
                transport.counters().h2d_bytes==io.h2d_bytes+d.bytes,"paused admission lost bypass payload or cache hit");
    }
    cache.set_admission_enabled(true); // Explicit return to manual fixture control.
    Device dest(size_t(c.bytes));
    {
        ExpertDispatch cancelled(cache,transport,{c,d,a},views(source),false);
        cancelled.copy(0,dest.data,dest.stream);
        cancelled.cancel();cancelled.cancel();
    }
    cuda_ok(cudaStreamSynchronize(dest.stream));
    require(cache.set_budget(0) && cache.resident_bytes()==0,"dispatch cancel retained pins or leases");
    run({a,b},true); // Zero budget: every selected matrix still arrives via bypass.
    require(cache.resident_bytes()==0,"zero-budget dispatch allocated cache bytes");
    cache.invalidate(a.model,a.generation); // Reset frequency history for OOM fixture.
    cache.set_budget(size_t(a.bytes+c.bytes));run({a},true);
    const auto before_oom=cache.counters();const auto io=transport.counters();const auto attempts=allocations;
    next_allocation=cudaErrorMemoryAllocation;
    run({b,a,c},true); // Failed miss must consume exactly one range, preserving later hit/miss order.
    const auto after_oom=cache.counters();const auto transferred=transport.counters();
    require(after_oom.allocation_bypasses==before_oom.allocation_bypasses+1 &&
            after_oom.bypasses==before_oom.bypasses+1 && after_oom.hits==before_oom.hits+1 &&
            after_oom.admissions==before_oom.admissions+1 && allocations==attempts+2 &&
            cache.resident(a) && cache.resident(c) && !cache.resident(b),"OOM dispatch cache decisions");
    require(transferred.h2d_bytes-io.h2d_bytes==b.bytes+c.bytes &&
            transferred.file_bytes-io.file_bytes+transferred.mmap_bytes-io.mmap_bytes==b.bytes+c.bytes,
            "OOM dispatch omitted/duplicated source or H2D bytes");
    run({a,c},false);
    require(cache.set_budget(0),"OOM dispatch retained leases/pins");cache.set_budget(size_t(a.bytes+c.bytes));
    next_allocation=cudaErrorInvalidValue;bool caught=false;
    try {run({b,a},true);}catch(const std::runtime_error &e) {
        caught=std::string(e.what())==cudaGetErrorString(cudaErrorInvalidValue);
    }
    require(caught && !transport.in_progress() && cache.resident_bytes()==0 &&
            cache.counters().allocation_bypasses==after_oom.allocation_bypasses,
            "fatal allocation did not propagate/cancel dispatch");
    run({b,a},true); // Restart the cancelled pipeline after a non-OOM error.
    std::printf("PASS dispatch mode=%d frequency=%d decode=%d: mixed miss/hit/bypass, all-hit no I/O, paused admission, cancellation, zero budget, injected OOM bytes/guards and fatal-error restart\n",mode,int(frequency),int(decode));
}

static void test_dispatch_branch_budgets(int mode,bool frequency,bool decode) {
    auto source=std::make_shared<Source>();auto plan=source->plan();
    auto main=plan[0],mtp=plan[1],other=plan[2];mtp.branch=other.branch=Branch::mtp;
    ExpertCache cache(size_t(main.bytes+mtp.bytes),{frequency,100000,32,true});
    cache.set_branch_budgets(size_t(main.bytes),0);
    ExpertTransport transport(0,4093,false,4,mode);
    if(mode==1) {
        DWORD old=0;require(VirtualProtect(source->view,source->bytes.size(),PAGE_NOACCESS,&old)!=0,"branch native protection");
    }
    auto run=[&](const std::vector<ExpertKey> &keys) {
        constexpr size_t guard=37;size_t total=guard;std::vector<size_t> offsets;
        for(const auto &key:keys) {offsets.push_back(total);total+=size_t(key.bytes)+guard;}
        Device dest(total);std::vector<uint8_t> expected(total,0xA5),actual(total);
        for(size_t i=0;i<keys.size();++i)
            std::copy_n(source->bytes.data()+keys[i].offset,size_t(keys[i].bytes),expected.data()+offsets[i]);
        ExpertDispatch dispatch(cache,transport,keys,views(source),decode);
        for(size_t i=0;i<keys.size();++i)dispatch.copy(i,dest.data+offsets[i],dest.stream);
        dispatch.finish();cuda_ok(cudaStreamSynchronize(dest.stream));
        cuda_ok(cudaMemcpy(actual.data(),dest.data,total,cudaMemcpyDeviceToHost));
        require(actual==expected,"branch dispatch payload/guards differ");
    };
    run({main});const auto before=transport.counters();const auto counts=cache.counters();
    run({mtp,main,other});
    const auto after=transport.counters();
    require(cache.resident(main) && !cache.resident(mtp) && !cache.resident(other) &&
            cache.resident_bytes(Branch::main)==main.bytes && cache.resident_bytes(Branch::mtp)==0 &&
            cache.counters().hits==counts.hits+1 && cache.counters().bypasses==counts.bypasses+2 &&
            after.h2d_bytes-before.h2d_bytes==mtp.bytes+other.bytes &&
            after.file_bytes-before.file_bytes+after.mmap_bytes-before.mmap_bytes==mtp.bytes+other.bytes,
            "disabled MTP quota lost bytes, hit or accounting");
    cache.set_branch_budgets(size_t(main.bytes),size_t(mtp.bytes));run({mtp});
    require(cache.resident_bytes()==main.bytes+mtp.bytes && cache.resident_bytes(Branch::mtp)==mtp.bytes,
            "MTP quota recovery/global cap mismatch");
    const auto loaded=transport.counters();run({mtp,main});
    require(transport.counters().h2d_bytes==loaded.h2d_bytes && cache.set_budget(0),"branch hits uploaded or retained pins");
    std::printf("PASS branch dispatch mode=%d frequency=%d decode=%d: disabled MTP full bypass/guards, main hit, quota recovery and shared budget\n",
                mode,int(frequency),int(decode));
}

static void test_dispatch_errors() {
    auto source=std::make_shared<Source>();auto plan=source->plan();
    ExpertCache cache(1<<20);ExpertTransport transport(0,4093,false,4,0);
    auto bad=views(source);++bad[0].generation;bool rejected=false;
    try {ExpertDispatch invalid(cache,transport,{plan[0]},bad,false);}
    catch(const std::invalid_argument &) {rejected=true;}
    require(rejected && !transport.in_progress() && cache.counters().misses==0,"bad dispatch binding changed cache or transport");
    rejected=false;
    try {ExpertDispatch duplicate(cache,transport,{plan[0],plan[0]},views(source),false);}
    catch(const std::invalid_argument &) {rejected=true;}
    require(rejected,"duplicate dispatch key accepted");
    Device dest(size_t(plan[0].bytes));
    {
        ExpertDispatch dispatch(cache,transport,{plan[0],plan[1]},views(source),false);
        rejected=false;
        try {ExpertDispatch nested(cache,transport,{}, {},false);}catch(const std::logic_error &) {rejected=true;}
        require(rejected,"nested dispatch accepted");
        rejected=false;
        try {dispatch.copy(1,dest.data,dest.stream);}catch(const std::logic_error &) {rejected=true;}
        require(rejected && dispatch.remaining()==2,"invalid dispatch order consumed work");
        dispatch.copy(0,dest.data,dest.stream);
        rejected=false;
        try {dispatch.finish();}catch(const std::logic_error &) {rejected=true;}
        require(rejected && !transport.in_progress(),"incomplete dispatch did not drain");
    }
    cuda_ok(cudaStreamSynchronize(dest.stream));
    // A loader must not invalidate mid-plan. Detect the loss of a planned hit
    // and cancel instead of using an absent/misordered pipeline range.
    {
        ExpertDispatch dispatch(cache,transport,{plan[0]}, {},false);
        cache.invalidate(plan[0].model,plan[0].generation);rejected=false;
        try {dispatch.copy(0,dest.data,dest.stream);}catch(const std::logic_error &) {rejected=true;}
        require(rejected && !transport.in_progress() && cache.resident_bytes()==0,"stale residency was not cancelled safely");
    }
    try {
        ExpertDispatch dispatch(cache,transport,{plan[0],plan[1]},views(source),false);
        dispatch.copy(0,dest.data,dest.stream);throw std::runtime_error("dispatch fixture exception");
    }catch(const std::runtime_error &e) {require(std::string(e.what())=="dispatch fixture exception","unexpected dispatch exception");}
    cuda_ok(cudaStreamSynchronize(dest.stream));
    require(!transport.in_progress() && cache.set_budget(0),"dispatch destructor retained active plan protection");
    {ExpertDispatch empty(cache,transport,{}, {},false);empty.finish();}
    std::puts("PASS: dispatch validation, duplicate/nested/order rejection, incomplete finish, invalidation detection, exception cleanup and empty plan");
}

int main() {
    try {
        cudaDeviceProp p{};cuda_ok(cudaGetDeviceProperties(&p,0));
        std::printf("GPU=%s; synthetic native plans; no GLM inference\n",p.name);
        for(int mode:{0,1,2})for(bool decode:{false,true})test_cancel_restart(mode,decode);
        for(int mode:{0,1,2})for(bool decode:{false,true})for(int readers:{2,4})
            test_cancel_restart(mode,decode,readers,true);
        for(int mode:{0,1,2})for(bool decode:{false,true})for(int readers:{1,4})
            test_cancel_restart(mode,decode,readers,false,true);
        for(int readers:{0,3}) {
            bool rejected=false;
            try {ExpertTransport invalid(0,4093,false,2,0,{},readers);}
            catch(const std::runtime_error &) {rejected=true;}
            require(rejected,"invalid decode reader count accepted");
        }
        test_destructor();test_validation_and_finish();
        for(int mode:{0,1,2})for(bool frequency:{false,true})for(bool decode:{false,true})test_dispatch(mode,frequency,decode);
        for(int mode:{0,1,2})for(bool frequency:{false,true})for(bool decode:{false,true})test_dispatch_branch_budgets(mode,frequency,decode);
        test_dispatch_errors();return 0;
    }catch(const std::exception &e) {std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
}
