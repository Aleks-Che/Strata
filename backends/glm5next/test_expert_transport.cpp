#include "expert_transport.hpp"
#include "expert_plan.hpp"
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
static void test_cancel_restart(int mode,bool decode) {
    ExpertTransport transport(0,4093,false,4,mode);
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
    if(mode==1 || (mode==2 && !decode))require(c.file_bytes>0 && c.mmap_bytes==0,"native path not used");
    else require(c.mmap_bytes>0 && c.file_bytes==0,"mmap path not used");
    std::printf("PASS mode=%d decode=%d: 4 plans, %zu matrix comparisons, source release and restart\n",mode,int(decode),comparisons);
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
int main() {
    try {
        cudaDeviceProp p{};cuda_ok(cudaGetDeviceProperties(&p,0));
        std::printf("GPU=%s; synthetic native plans; no GLM inference\n",p.name);
        for(int mode:{0,1,2})for(bool decode:{false,true})test_cancel_restart(mode,decode);
        test_destructor();test_validation_and_finish();return 0;
    }catch(const std::exception &e) {std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
}
