// Packed-byte delivery only: no GLM forward pass or quantization arithmetic.
#include "../common/expert_pipeline.hpp"
#include "expert_plan.hpp"
#include <cstdio>

static void check(bool value,const char *message) {
    if(!value)throw std::runtime_error(message);
}
static void cuda_ok(cudaError_t code) {
    if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));
}

struct DeviceBuffer {
    uint8_t *data=nullptr;
    explicit DeviceBuffer(size_t bytes) {cuda_ok(cudaMalloc((void **)&data,bytes));}
    ~DeviceBuffer() {if(data)cudaFree(data);}
};
struct Stream {
    cudaStream_t value=nullptr;
    Stream() {cuda_ok(cudaStreamCreateWithFlags(&value,cudaStreamNonBlocking));}
    ~Stream() {if(value) {cudaStreamSynchronize(value);cudaStreamDestroy(value);}}
};

struct Fixture {
    std::vector<uint8_t> bytes;
    std::array<size_t,3> starts{},matrix_bytes{};
#ifdef _WIN32
    FILE *file=nullptr;
    HANDLE mapping=nullptr;
    uint8_t *view=nullptr;
    bool registered=false;
#endif
    explicit Fixture(size_t group) {
        // Independent packed geometry for the eight routed types in both GGUFs.
        // gate/up [4096,2048,10], down [2048,4096,10]; 32768 blocks/expert.
        constexpr size_t block_bytes[]={82,110,136,84,110,98,210,144};
        size_t end=96;
        for(size_t p=0;p<3;++p) {
            starts[p]=(end+31)/32*32;
            matrix_bytes[p]=32768*block_bytes[(group*3+p)%8];
            end=starts[p]+10*matrix_bytes[p]+47;
        }
        // No trailing padding after the final tensor: expert 9 ends at EOF.
        end-=47;
        bytes.resize(end);
        for(size_t i=0;i<end;++i)bytes[i]=uint8_t(i*29+(i>>8)*17+(i>>16)*7+(i>>24)+group*53);
#ifdef _WIN32
        try {
            file=std::tmpfile();check(file!=nullptr,"fixture file");
            check(std::fwrite(bytes.data(),1,end,file)==end && std::fflush(file)==0,"fixture write");
            mapping=CreateFileMappingW((HANDLE)_get_osfhandle(_fileno(file)),nullptr,PAGE_READONLY,0,0,nullptr);
            check(mapping!=nullptr,"fixture mapping");
            view=(uint8_t *)MapViewOfFile(mapping,FILE_MAP_READ,0,0,0);
            check(view!=nullptr,"fixture view");
            register_source();
        } catch(...) {close();throw;}
#endif
    }
    const uint8_t *data() const {
#ifdef _WIN32
        return view;
#else
        return bytes.data();
#endif
    }
#ifdef _WIN32
    void register_source() {
        strata_expert_file::add(view,bytes.size(),_fileno(file));registered=true;
    }
    void unregister_source() {
        if(registered)strata_expert_file::remove(view);
        registered=false;
    }
    void protect(DWORD access) {
        DWORD old=0;
        check(VirtualProtect(view,bytes.size(),access,&old)!=0,"fixture protection");
    }
    void close() {
        unregister_source();
        if(view)UnmapViewOfFile(view);
        view=nullptr;
        if(mapping)CloseHandle(mapping);
        mapping=nullptr;
        if(file)std::fclose(file);
        file=nullptr;
    }
#endif
    ~Fixture() {
#ifdef _WIN32
        close();
#endif
    }
};

static void test_group(Fixture &fixture,size_t group,int mode,bool decode,bool early_host_refill) {
    constexpr size_t chunk=(256<<10)+17; // split quant blocks and leave a short tail
    constexpr size_t guard=37;
    std::vector<StrataExpertSlice> slices;
    std::vector<size_t> destination_offsets;
    size_t dest_bytes=guard,source_bytes=0,chunks=0;
    // Route through the native planner, with duplicate IDs and both boundaries.
    using namespace strata_glm;
    constexpr const char *quants[]={"IQ2_S","IQ3_S","IQ4_XS","Q2_K","Q3_K","IQ3_XXS","Q6_K","Q4_K"};
    ExpertLayerLayout layout{group==2?45:group==1?11:3,45,46,3,{}};
    for(size_t p=0;p<3;++p)
        layout.tensors[p]={Projection(p),quants[(group*3+p)%8],p==2?2048u:4096u,p==2?4096u:2048u,
                           10,0,fixture.starts[p],fixture.matrix_bytes[p]*10};
    auto plan=plan_experts("synthetic-model-"+std::to_string(group),1,layout,
                          {{"fixture.gguf",96,fixture.bytes.size()}},{9,0,9,8,2,7,3,6,4,0});
    check(plan.size()==24,"native planner lost a routed triple");
    constexpr size_t experts[]={9,0,8,2,7,3,6,4};
    for(size_t i=0;i<plan.size();++i) {
        const auto &key=plan[i];const size_t p=i%3,expert=experts[i/3];
        const size_t bytes=fixture.matrix_bytes[p];
        // Independent fixture offsets/sizes remain the oracle for the plan.
        check(key.expert==int(expert) && key.projection==Projection(p) && key.bytes==bytes &&
              key.offset==fixture.starts[p]+expert*bytes,"native planner disagrees with fixture ranges");
        slices.push_back({fixture.data()+key.offset,size_t(key.bytes),decode});
        destination_offsets.push_back(dest_bytes);
        dest_bytes+=bytes+guard;
        source_bytes+=bytes;
        chunks+=(bytes+chunk-1)/chunk;
    }
    std::vector<uint8_t> expected(dest_bytes,0xA5),actual(dest_bytes);
    for(size_t i=0;i<slices.size();++i) {
        const size_t offset=uintptr_t(slices[i].data)-uintptr_t(fixture.data());
        std::copy_n(fixture.bytes.data()+offset,slices[i].bytes,expected.data()+destination_offsets[i]);
    }
    DeviceBuffer dest(dest_bytes);
    Stream even,odd;
    cuda_ok(cudaMemset(dest.data,0xA5,dest_bytes));
    StrataExpertPipeline pipeline(0,chunk,false,4,mode,{},1,early_host_refill);
#ifdef _WIN32
    if(mode==1)fixture.protect(PAGE_NOACCESS); // Native reads must not touch mmap.
#endif
    pipeline.start(slices);
#ifdef _WIN32
    if(mode==1)fixture.unregister_source(); // Jobs must retain source handles.
#endif
    for(size_t i=0;i<slices.size();++i) {
        check(pipeline.transfer(dest.data+destination_offsets[i],slices[i].data,slices[i].bytes,
                                i%2?odd.value:even.value),"planned matrix fell back");
    }
    pipeline.finish();
    cuda_ok(cudaStreamSynchronize(even.value));
    cuda_ok(cudaStreamSynchronize(odd.value));
    cuda_ok(cudaMemcpy(actual.data(),dest.data,dest_bytes,cudaMemcpyDeviceToHost));
    check(actual==expected,"GPU matrix/guard bytes differ");
    const auto counters=pipeline.counters();
    check(counters.h2d_bytes==source_bytes && counters.d2d_bytes==source_bytes,"transfer byte counters");
    check(counters.file_bytes+counters.mmap_bytes==source_bytes,"source byte counters");
    check(counters.chunks==chunks && counters.unused==0 && counters.groups==1,"chunk counters");
    check(counters.read_peak>=1 && counters.read_peak<=4,"reader bound");
    check(counters.pinned_bytes==4*chunk && counters.device_ring_bytes==4*chunk,"allocated staging counters");
    check(counters.queued_bytes==0 && counters.reader_owned_bytes==0 && counters.unused_bytes==0,"completed plan telemetry not drained");
    check(counters.queued_peak>0 && counters.queued_peak<=4*chunk &&
          counters.reader_owned_peak>0 && counters.reader_owned_peak<=4*chunk,"payload high-water bounds");
    check(counters.wait_us==counters.slot_wait_us+counters.consumer_wait_us,"wait breakdown changed legacy total");
    if(decode && mode!=1)check(counters.read_peak==1,"decode must use one reader");
    if(mode==0 || (mode==2 && decode))check(counters.mmap_bytes==source_bytes,"mmap mode not exercised");
#ifdef _WIN32
    if(mode==1) {
        check(counters.file_bytes==source_bytes && counters.mmap_bytes==0,"native mode not exercised");
        fixture.protect(PAGE_READONLY);
        fixture.register_source();
    }
#endif
    std::printf("PASS group=%zu mode=%d decode=%d early_host_refill=%d matrices=24 bytes=%zu chunks=%zu file_bytes=%llu mmap_bytes=%llu\n",
                group,mode,int(decode),int(early_host_refill),source_bytes,chunks,
                (unsigned long long)counters.file_bytes,(unsigned long long)counters.mmap_bytes);
}

// A stalled GPU consumer must prevent device reuse, but must not prevent
// filling the now-idle host slot. Gate lifetime also releases failed tests.
struct RefillGate {
    cudaStream_t stream;
    std::atomic<bool> release{false},expired{false};
    ~RefillGate() {release.store(true);cudaStreamSynchronize(stream);}
};
static void CUDART_CB block_refill_consumer(void *ptr) {
    auto &gate=*static_cast<RefillGate *>(ptr);
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
    while(!gate.release.load()) {
        if(std::chrono::steady_clock::now()>deadline) {gate.expired.store(true);break;}
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}
static void test_early_refill() {
    constexpr size_t chunk=65553,n=8*chunk+17;
    std::vector<uint8_t> source(n),actual(n);
    for(size_t i=0;i<n;++i)source[i]=uint8_t(i*17+i/251);
    DeviceBuffer dest(n);Stream stream;
    StrataExpertPipeline pipeline(0,chunk,false,1,0,{},1,true);
    pipeline.start({{source.data(),n,true}});
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(2);
    while(pipeline.ready_chunks()!=4 && std::chrono::steady_clock::now()<deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    check(pipeline.ready_chunks()==4,"refill fixture did not fill ring");
    RefillGate gate{stream.value};
    cuda_ok(cudaLaunchHostFunc(stream.value,block_refill_consumer,&gate));
    check(pipeline.transfer(dest.data,source.data(),chunk,stream.value),"first refill transfer");
    const auto refill_deadline=std::chrono::steady_clock::now()+std::chrono::seconds(2);
    while(pipeline.counters().mmap_bytes<5*chunk && std::chrono::steady_clock::now()<refill_deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    const auto c=pipeline.counters();
    check(c.mmap_bytes==5*chunk && c.h2d_bytes==4*chunk &&
          cudaStreamQuery(stream.value)==cudaErrorNotReady && !gate.expired.load(),
          "host refill did not overlap blocked consumer, or overwrote its GPU slot");
    gate.release.store(true);
    check(pipeline.transfer(dest.data+chunk,source.data()+chunk,n-chunk,stream.value),"remaining refill transfer");
    pipeline.finish();cuda_ok(cudaStreamSynchronize(stream.value));
    cuda_ok(cudaMemcpy(actual.data(),dest.data,n,cudaMemcpyDeviceToHost));
    check(actual==source,"early refill corrupted GPU bytes");
    const auto done=pipeline.counters();
    check(done.wait_us==done.slot_wait_us+done.consumer_wait_us && !done.reader_owned_bytes && !done.queued_bytes,
          "refill wait/lifetime accounting");
    std::puts("PASS: early host refill overlaps blocked consumer; GPU slot protected; all bytes exact");
}
static void test_telemetry() {
    constexpr size_t chunk=65553,n=chunk+17;
    std::vector<uint8_t> source(n,0x31),actual(n);
    DeviceBuffer dest(n);Stream stream;
    StrataExpertPipeline pipeline(0,chunk,false,1);
    const auto empty=pipeline.counters();
    check(empty.pinned_bytes==4*chunk && empty.device_ring_bytes==4*chunk &&
          !empty.queued_peak && !empty.reader_owned_peak && !empty.wait_us,"initial telemetry");
    auto fill=[&](size_t chunks) {
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(2);
        while(pipeline.ready_chunks()<chunks && std::chrono::steady_clock::now()<deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        check(pipeline.ready_chunks()==chunks,"telemetry fixture did not fill queue");
    };
    pipeline.start({{source.data(),n,false}});fill(2);
    auto queued=pipeline.counters();
    check(queued.queued_bytes==n && queued.queued_peak==n && queued.reader_owned_bytes==0 &&
          queued.reader_owned_peak==chunk,"partial-tail queue/reader byte peak");
    pipeline.finish();auto cancelled=pipeline.counters();
    check(cancelled.queued_bytes==0 && cancelled.reader_owned_bytes==0 &&
          cancelled.unused==2 && cancelled.unused_bytes==n,"cancellation payload accounting");
    pipeline.start({{source.data(),n,true}});
    check(pipeline.transfer(dest.data,source.data(),n,stream.value),"telemetry restart transfer");
    pipeline.finish();cuda_ok(cudaStreamSynchronize(stream.value));
    cuda_ok(cudaMemcpy(actual.data(),dest.data,n,cudaMemcpyDeviceToHost));check(actual==source,"telemetry changed bytes");
    auto done=pipeline.counters();
    check(done.unused_bytes==n && done.h2d_bytes==2*n && done.d2d_bytes==n && done.queued_peak==n &&
          done.reader_owned_bytes==0 && done.queued_bytes==0 &&
          done.wait_us==done.slot_wait_us+done.consumer_wait_us,"restart/legacy counters changed");
    pipeline.start({});pipeline.finish();
    check(pipeline.counters().queued_peak==n,"empty plan reset cumulative high-water mark");
    bool rejected=false;
    try {StrataExpertPipeline overflow(0,std::numeric_limits<size_t>::max(),false);}
    catch(const std::runtime_error &) {rejected=true;}
    check(rejected,"staging byte multiplication overflow accepted");
    std::printf("PASS telemetry: pinned=%llu ring=%llu reader_peak=%llu queued_peak=%llu unused_bytes=%llu; CPU waits slot=%llu consumer=%llu total=%llu\n",
                (unsigned long long)done.pinned_bytes,(unsigned long long)done.device_ring_bytes,
                (unsigned long long)done.reader_owned_peak,(unsigned long long)done.queued_peak,
                (unsigned long long)done.unused_bytes,(unsigned long long)done.slot_wait_us,
                (unsigned long long)done.consumer_wait_us,(unsigned long long)done.wait_us);
}

int main() {
    try {
        int devices=0;
        cuda_ok(cudaGetDeviceCount(&devices));
        check(devices>0,"CUDA transport tests require a CUDA device");
        cudaDeviceProp props{};
        cuda_ok(cudaGetDeviceProperties(&props,0));
        int runtime=0,driver=0;
        cuda_ok(cudaRuntimeGetVersion(&runtime));cuda_ok(cudaDriverGetVersion(&driver));
        std::printf("GPU=%s CUDA_runtime=%d CUDA_driver=%d\n",props.name,runtime,driver);
        test_telemetry();test_early_refill();
    std::puts("Native router plan -> synthetic packed bytes: IQ2_S/IQ3_S/IQ4_XS/Q2_K/Q3_K/IQ3_XXS/Q6_K/Q4_K; no numerical GLM inference");
        size_t cases=0;
        for(size_t group=0;group<3;++group) {
            Fixture fixture(group);
#ifdef _WIN32
            // Auto prefill runs before touching mmap; its residency policy can
            // choose native reads. The counters report what actually happened.
            for(int mode:{2,1,0})for(bool decode:{false,true}) {
#else
            // The shared source registry has no POSIX native implementation.
            for(int mode:{0})for(bool decode:{false,true}) {
#endif
                for(bool refill:{false,true}) {test_group(fixture,group,mode,decode,refill);++cases;}
            }
        }
        std::printf("PASS: %zu cases, %zu GPU matrix comparisons, guarded destinations, partial chunks, two consumer streams\n",cases,cases*24);
        return 0;
    } catch(const std::exception &e) {std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
}
