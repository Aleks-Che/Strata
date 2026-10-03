#include "expert_pipeline.hpp"
#include "expert_reuse.hpp"
#include <atomic>
#include <cstdio>

static void check(bool value) {if(!value)throw std::runtime_error("pipeline contract failed");}
static void cuda_ok(cudaError_t code) {if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));}
struct Gate {
    cudaStream_t stream;
    std::atomic<bool> release{false},expired{false};
    explicit Gate(cudaStream_t s):stream(s) {}
    ~Gate() {release.store(true);cudaStreamSynchronize(stream);}
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
static void test_file(cudaStream_t compute,uint8_t *dest,const std::vector<uint8_t> &expected) {
    MappedFixture mapped(expected);
    auto *source=(const uint8_t *)mapped.ptr+13;
    size_t bytes=expected.size()-13;
    check(!strata_expert_file::find(source,bytes+1));
    std::vector<uint8_t> actual(bytes);
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
        {
            StrataExpertPipeline pipeline(0,chunk,false);
            Gate gate(compute);
            cuda_ok(cudaLaunchHostFunc(compute,blocked,&gate));
            pipeline.start({{source.data(),size,false}});
            auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(2);
            while(pipeline.ready_chunks()<StrataExpertPipeline::slots && std::chrono::steady_clock::now()<deadline)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            bool overlapped=pipeline.ready_chunks()==StrataExpertPipeline::slots && !gate.expired.load();
            bool blocked_compute=cudaStreamQuery(compute)==cudaErrorNotReady;
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
        }
        test_reuse(compute,dest,size);
#ifdef _WIN32
        test_file(compute,dest,source);
#endif
        cuda_ok(cudaStreamDestroy(compute));cuda_ok(cudaFree(dest));
        std::puts("Pipeline overlap, file queue, byte parity, ordered reuse and cancellation passed");
        return 0;
    } catch(const std::exception &e) {std::fprintf(stderr,"%s\n",e.what());return 1;}
}
