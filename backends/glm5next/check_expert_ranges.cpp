// Read-only Windows GGUF range -> shared pipeline -> GPU -> byte comparison.
// The Python inspector supplies ranges; this executable validates file bounds.
#include "../common/expert_pipeline.hpp"
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>

static void require(bool ok,const char *message) {if(!ok)throw std::runtime_error(message);}
static void cuda_ok(cudaError_t e) {if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
static uint64_t number(const std::string &s) {
    require(!s.empty() && s.find_first_not_of("0123456789")==std::string::npos,"expected unsigned decimal");
    return std::stoull(s);
}
static std::string unhex(const std::string &s) {
    require(!s.empty() && s.size()%2==0 && s.size()<=131072,"invalid hex path size");
    std::string out;
    for(size_t i=0;i<s.size();i+=2) {
        auto digit=[](char c)->unsigned {
            if(c>='0' && c<='9')return c-'0';
            if(c>='a' && c<='f')return c-'a'+10;
            throw std::runtime_error("invalid hex path");
        };
        char c=char(digit(s[i])*16+digit(s[i+1]));
        require(c!=0,"NUL in path");out+=c;
    }
    return out;
}
static std::string hex(const std::string &s) {
    constexpr char digits[]="0123456789abcdef";
    std::string out;
    for(unsigned char c:s) {out+=digits[c>>4];out+=digits[c&15];}
    return out;
}
struct Range {uint64_t offset;size_t bytes;std::string path;};
struct Plan {size_t chunk;int mode;std::vector<Range> ranges;};
static Plan parse(std::istream &input) {
    std::string version,chunk,mode,count;
    require(bool(input>>version>>chunk>>mode>>count) && version=="GLM_RANGES_V1","invalid range header");
    const auto c=number(chunk),m=number(mode),n=number(count);
    require(c>=1 && c<=16*1024*1024 && m<=2 && n>=1 && n<=4096,"invalid plan limits");
    Plan plan{size_t(c),int(m),{}};
    uint64_t chunks=0;
    for(size_t i=0;i<n;++i) {
        std::string offset,bytes,path;
        require(bool(input>>offset>>bytes>>path),"missing range");
        auto size=number(bytes);
        require(size>=1 && size<=256*1024*1024,"matrix exceeds 256 MiB limit");
        chunks+=(size+c-1)/c;
        require(chunks<=1048576,"plan exceeds chunk metadata limit");
        auto decoded=unhex(path);
        require(std::filesystem::u8path(decoded).is_absolute(),"source path must be absolute");
        plan.ranges.push_back({number(offset),size_t(size),decoded});
    }
    std::string extra;
    require(!(input>>extra),"extra manifest fields");
    return plan;
}
static void parser_tests() {
    const auto path=hex("C:/fixture with spaces.gguf");
    auto valid="GLM_RANGES_V1 262161 1 1\n4294967419 1031 "+path+"\n";
    std::istringstream input(valid);
    auto p=parse(input);
    require(p.ranges[0].offset==4294967419 && p.ranges[0].bytes==1031,"64-bit parse");
    for(const auto &bad:std::vector<std::string>{
        "", "GLM_RANGES_V0 1 0 1", "GLM_RANGES_V1 0 0 1", "GLM_RANGES_V1 1 3 1",
        "GLM_RANGES_V1 1 0 0", "GLM_RANGES_V1 1 0 4097", valid+"extra",
        "GLM_RANGES_V1 1 0 1\n-1 1 "+path,
        "GLM_RANGES_V1 1 0 1\n18446744073709551616 1 "+path,
        "GLM_RANGES_V1 1 0 1\n0 268435457 "+path,
        "GLM_RANGES_V1 1 0 1\n0 1048577 "+path,
        "GLM_RANGES_V1 1 0 1\n0 0 "+path,
        "GLM_RANGES_V1 1 0 1\n0 1 00", "GLM_RANGES_V1 1 0 1\n0 1 gg",
        "GLM_RANGES_V1 1 0 1\n0 1 7", "GLM_RANGES_V1 1 0 1\n0 1 "+hex("relative.gguf")}) {
        bool rejected=false;
        try {std::istringstream broken(bad);parse(broken);}catch(const std::exception &) {rejected=true;}
        require(rejected,"invalid manifest accepted");
    }
    std::puts("Range parser limits, paths, malformed records and 64-bit offsets passed");
}

struct MappedSource {
    FILE *file=nullptr;HANDLE mapping=nullptr;uint8_t *view=nullptr;
    size_t bytes=0;bool registered=false;
    explicit MappedSource(const std::string &path) {
        try {
            file=_wfopen(std::filesystem::u8path(path).c_str(),L"rb");
            require(file!=nullptr,"cannot open source read-only");
            LARGE_INTEGER size{};
            auto handle=(HANDLE)_get_osfhandle(_fileno(file));
            require(GetFileSizeEx(handle,&size) && size.QuadPart>0,"invalid source file size");
            bytes=size_t(size.QuadPart);
            mapping=CreateFileMappingW(handle,nullptr,PAGE_READONLY,0,0,nullptr);
            require(mapping!=nullptr,"source mapping failed");
            view=(uint8_t *)MapViewOfFile(mapping,FILE_MAP_READ,0,0,0);
            require(view!=nullptr,"source map view failed");
            strata_expert_file::add(view,bytes,_fileno(file));registered=true;
        }catch(...) {close();throw;}
    }
    void close() {
        if(registered)strata_expert_file::remove(view);
        registered=false;
        if(view)UnmapViewOfFile(view);view=nullptr;
        if(mapping)CloseHandle(mapping);mapping=nullptr;
        if(file)std::fclose(file);file=nullptr;
    }
    ~MappedSource() {close();}
    void baseline(const Range &r,uint8_t *dest) {
        require(_fseeki64(file,r.offset,SEEK_SET)==0,"baseline seek failed");
        require(std::fread(dest,1,r.bytes,file)==r.bytes,"short baseline read");
    }
};
struct Device {
    uint8_t *data=nullptr;cudaStream_t stream=nullptr;
    explicit Device(size_t bytes) {
        cuda_ok(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        auto e=cudaMalloc((void **)&data,bytes);
        if(e!=cudaSuccess) {cudaStreamDestroy(stream);cuda_ok(e);}
    }
    ~Device() {cudaStreamSynchronize(stream);cudaFree(data);cudaStreamDestroy(stream);}
};
static void run(const Plan &plan) {
    static_assert(sizeof(size_t)==8,"64-bit process required");
    std::map<std::string,std::unique_ptr<MappedSource>> sources;
    std::vector<StrataExpertSlice> slices;
    size_t largest=0;uint64_t total=0,chunks=0;
    for(const auto &r:plan.ranges) {
        auto &source=sources[r.path];
        if(!source)source=std::make_unique<MappedSource>(r.path);
        require(r.offset<=source->bytes && r.bytes<=source->bytes-r.offset,"range outside source file");
        slices.push_back({source->view+r.offset,r.bytes,false});
        largest=std::max(largest,r.bytes);total+=r.bytes;chunks+=(r.bytes+plan.chunk-1)/plan.chunk;
    }
    cudaDeviceProp props{};cuda_ok(cudaGetDeviceProperties(&props,0));
    int runtime=0,driver=0;cuda_ok(cudaRuntimeGetVersion(&runtime));cuda_ok(cudaDriverGetVersion(&driver));
    std::printf("GPU %s %d %d\n",hex(props.name).c_str(),runtime,driver);
    constexpr size_t guard=37;
    Device device(largest+2*guard);
    StrataExpertPipeline pipeline(0,plan.chunk,false,4,plan.mode);
    pipeline.start(slices);
    for(size_t i=0;i<slices.size();++i) {
        const auto &r=plan.ranges[i];
        std::vector<uint8_t> expected(r.bytes+2*guard,0xA5),actual(expected.size());
        sources.at(r.path)->baseline(r,expected.data()+guard); // Independent stdio path.
        cuda_ok(cudaMemsetAsync(device.data,0xA5,expected.size(),device.stream));
        require(pipeline.transfer(device.data+guard,slices[i].data,r.bytes,device.stream),"unplanned transfer");
        cuda_ok(cudaStreamSynchronize(device.stream));
        cuda_ok(cudaMemcpy(actual.data(),device.data,actual.size(),cudaMemcpyDeviceToHost));
        require(actual==expected,"GPU payload or guard mismatch");
        std::printf("OK %zu %zu\n",i,r.bytes);
    }
    pipeline.finish();
    auto c=pipeline.counters();
    require(c.h2d_bytes==total && c.d2d_bytes==total && c.file_bytes+c.mmap_bytes==total &&
            c.chunks==chunks && c.unused==0,"transport counters differ from plan");
    if(plan.mode==0)require(c.mmap_bytes==total,"mmap path not exercised");
    if(plan.mode==1)require(c.file_bytes==total,"native path not exercised");
    std::printf("TOTAL %llu %llu %llu %llu %llu\n",(unsigned long long)c.h2d_bytes,
                (unsigned long long)c.d2d_bytes,(unsigned long long)c.file_bytes,
                (unsigned long long)c.mmap_bytes,(unsigned long long)c.chunks);
}
int main(int argc,char **argv) {
    try {
        if(argc==2 && std::string(argv[1])=="--test-parser") {parser_tests();return 0;}
        require(argc==1,"ranges must be provided on stdin; no arguments accepted");
        run(parse(std::cin));return 0;
    }catch(const std::exception &e) {std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
}
