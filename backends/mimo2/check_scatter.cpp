#include "scatter_copy.hpp"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
using json=nlohmann::ordered_json;
static void check(cudaError_t rc) {if(rc!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(rc));}
static void require(bool ok,const char *message) {if(!ok)throw std::runtime_error(message);}
struct Buffer {
    void *p=nullptr;
    explicit Buffer(size_t n) {check(cudaMalloc(&p,n));}
    ~Buffer() {if(p)cudaFree(p);}
};
struct Stream {
    cudaStream_t p=nullptr;
    Stream() {check(cudaStreamCreateWithFlags(&p,cudaStreamNonBlocking));}
    ~Stream() {if(p) {cudaStreamSynchronize(p);cudaStreamDestroy(p);}}
};
static unsigned char pattern(size_t i) {uint32_t x=uint32_t(i)+0x9e3779b9u;x^=x>>16;x*=0x85ebca6bu;x^=x>>13;x*=0xc2b2ae35u;return static_cast<unsigned char>(x^(x>>16));}
static size_t align(size_t n) {return (n+63)&~size_t(63);}
int main(int argc,char **argv) {
    try {
        require(argc>=2 && argc<=3,"usage: scatter-check fresh.json [--benchmark]");
        require(!std::ifstream(argv[1]).good(),"report already exists");
        const bool benchmark=argc==3;require(!benchmark || std::string(argv[2])=="--benchmark","invalid option");
        Stream stream;json cases=json::array(),timings=json::array();
        const size_t sizes[]={0,1,15,16,17,511,512,513,16383,16384,16385,65535,65536,65537};
        for(int tile:{16,64,256})for(size_t count:{0,1,31,32,33,65,129})for(int misalign:{0,1})for(int zeros:{0,1}) {
            std::vector<size_t> offsets,bytes;size_t total=64;
            for(size_t i=0;i<count;++i) {offsets.push_back(total);bytes.push_back(sizes[zeros?i%14:1+i%13]);total=align(total+bytes.back()+64);}
            // Trailing empty descriptors exercise the final partial packet.
            if(zeros && !bytes.empty())bytes.back()=0;
            std::vector<unsigned char> source(total),expected(total,0xA5),actual(total);
            for(size_t i=0;i<total;++i)source[i]=pattern(i);
            Buffer src(total),dst(total);
            std::vector<void *> to;std::vector<const void *> from;
            for(size_t i=0;i<count;++i) {
                to.push_back(static_cast<unsigned char *>(dst.p)+offsets[i]+3*misalign);
                from.push_back(static_cast<unsigned char *>(src.p)+offsets[i]+misalign);
                std::memcpy(expected.data()+offsets[i]+3*misalign,source.data()+offsets[i]+misalign,bytes[i]);
            }
            check(cudaMemcpyAsync(src.p,source.data(),total,cudaMemcpyHostToDevice,stream.p));
            check(cudaMemsetAsync(dst.p,0xA5,total,stream.p));size_t launches=0;
            check(mimo_scatter_copy(to.data(),from.data(),bytes.data(),count,stream.p,&launches,tile));
            check(cudaMemcpyAsync(actual.data(),dst.p,total,cudaMemcpyDeviceToHost,stream.p));check(cudaStreamSynchronize(stream.p));
            require(actual==expected,"scatter bytes or destination guards differ");
            check(cudaMemcpyAsync(actual.data(),src.p,total,cudaMemcpyDeviceToHost,stream.p));check(cudaStreamSynchronize(stream.p));
            require(actual==source,"scatter modified its source");
            const size_t nonempty=std::count_if(bytes.begin(),bytes.end(),[](size_t n){return n>0;});
            require(launches==(nonempty+31)/32,"scatter launch count differs");
            cases.push_back({{"count",count},{"misaligned",misalign},{"zeros",zeros},{"tile_kib",tile},{"bytes",total},{"launches",launches},{"pass",true}});
        }
        {
            size_t n=1,launches=99;void *dst=nullptr;const void *src=nullptr;
            require(mimo_scatter_copy(&dst,&src,&n,1,stream.p,&launches)==cudaErrorInvalidValue && !launches,"invalid pointers accepted");
            require(mimo_scatter_copy(nullptr,nullptr,nullptr,8193,stream.p)==cudaErrorInvalidValue,"unbounded descriptors accepted");
            require(mimo_scatter_copy(nullptr,nullptr,nullptr,0,stream.p,nullptr,32)==cudaErrorInvalidValue,"invalid tile accepted");
            cases.push_back({{"name","invalid_arguments"},{"pass",true}});
        }
        // Full production strides and literal 512-byte guards, four disjoint
        // banks to avoid measuring just a repeatedly copied L2-sized fixture.
        const size_t matrix_sizes[]={2752512,3604480,4456448};
        std::vector<size_t> offsets,bytes;size_t bank_bytes=64;
        for(int i=0;i<8;++i)for(size_t n:{matrix_sizes[i%3],size_t(512)}) {
            offsets.push_back(bank_bytes);bytes.push_back(n);bank_bytes=align(bank_bytes+n+64);
        }
        const size_t total=bank_bytes*4;
        std::vector<unsigned char> source(total),expected(total,0xA5),actual(total);
        for(size_t i=0;i<total;++i)source[i]=pattern(i);
        Buffer src(total),dst(total);std::vector<void *> to[4];std::vector<const void *> from[4];
        for(int bank=0;bank<4;++bank)for(size_t i=0;i<bytes.size();++i) {
            const size_t off=bank*bank_bytes+offsets[i];
            to[bank].push_back(static_cast<unsigned char *>(dst.p)+off);
            from[bank].push_back(static_cast<unsigned char *>(src.p)+off);
            std::memcpy(expected.data()+off,source.data()+off,bytes[i]);
        }
        check(cudaMemcpyAsync(src.p,source.data(),total,cudaMemcpyHostToDevice,stream.p));
        for(int tile:{16,64,256}) {
            check(cudaMemsetAsync(dst.p,0xA5,total,stream.p));
            for(int bank=0;bank<4;++bank)check(mimo_scatter_copy(to[bank].data(),from[bank].data(),bytes.data(),bytes.size(),stream.p,nullptr,tile));
            check(cudaMemcpyAsync(actual.data(),dst.p,total,cudaMemcpyDeviceToHost,stream.p));check(cudaStreamSynchronize(stream.p));
            require(actual==expected,"production stride bytes/guards differ");
            cases.push_back({{"name","production_strides_four_banks"},{"tile_kib",tile},{"pass",true}});
        }
        if(benchmark)for(int mode:{0,16,64,256,256,64,16,0}) {
            constexpr int iterations=256;
            const auto begin=std::chrono::steady_clock::now();
            for(int i=0;i<iterations;++i) {
                const int bank=i%4;
                if(mode)check(mimo_scatter_copy(to[bank].data(),from[bank].data(),bytes.data(),bytes.size(),stream.p,nullptr,mode));
                else for(size_t j=0;j<bytes.size();++j)check(cudaMemcpyAsync(to[bank][j],from[bank][j],bytes[j],cudaMemcpyDeviceToDevice,stream.p));
                check(cudaStreamSynchronize(stream.p));
            }
            const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
            timings.push_back({{"tile_kib_or_memcpy0",mode},{"iterations",iterations},{"wall_ms",ms},{"us_per_tensor",ms*1000/iterations}});
        }
        json result={{"status","pass"},{"scope","byte copies only; production strides, holes, tails and packet boundaries; not inference speed"},
            {"case_count",cases.size()},{"cases",cases},{"device_allocation_bytes",2*total},{"timings",timings}};
        std::ofstream file(argv[1]);file<<result.dump(2)<<'\n';require(bool(file),"report write failed");
        std::cout<<"PASS "<<cases.size()<<" cases\n"<<timings.dump(2)<<'\n';return 0;
    } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
