// Packed-byte delivery only: no GLM forward pass or quantization arithmetic.
#include "../common/expert_pipeline.hpp"
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

static void test_group(Fixture &fixture,size_t group,int mode,bool decode) {
    constexpr size_t chunk=(256<<10)+17; // split quant blocks and leave a short tail
    constexpr size_t guard=37;
    std::vector<StrataExpertSlice> slices;
    std::vector<size_t> destination_offsets;
    size_t dest_bytes=guard,source_bytes=0,chunks=0;
    // Eight distinct routes, including both boundaries; retain every triple.
    for(size_t expert:{9,0,8,2,7,3,6,4})for(size_t projection=0;projection<3;++projection) {
        const size_t bytes=fixture.matrix_bytes[projection];
        slices.push_back({fixture.data()+fixture.starts[projection]+expert*bytes,bytes,decode});
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
    StrataExpertPipeline pipeline(0,chunk,false,4,mode);
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
    if(decode && mode!=1)check(counters.read_peak==1,"decode must use one reader");
    if(mode==0 || (mode==2 && decode))check(counters.mmap_bytes==source_bytes,"mmap mode not exercised");
#ifdef _WIN32
    if(mode==1) {
        check(counters.file_bytes==source_bytes && counters.mmap_bytes==0,"native mode not exercised");
        fixture.protect(PAGE_READONLY);
        fixture.register_source();
    }
#endif
    std::printf("PASS group=%zu mode=%d decode=%d matrices=24 bytes=%zu chunks=%zu file_bytes=%llu mmap_bytes=%llu\n",
                group,mode,int(decode),source_bytes,chunks,
                (unsigned long long)counters.file_bytes,(unsigned long long)counters.mmap_bytes);
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
        std::puts("Synthetic packed bytes: IQ2_S/IQ3_S/IQ4_XS/Q2_K/Q3_K/IQ3_XXS/Q6_K/Q4_K; no numerical GLM inference");
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
                test_group(fixture,group,mode,decode);++cases;
            }
        }
        std::printf("PASS: %zu cases, %zu GPU matrix comparisons, guarded destinations, partial chunks, two consumer streams\n",cases,cases*24);
        return 0;
    } catch(const std::exception &e) {std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
}
