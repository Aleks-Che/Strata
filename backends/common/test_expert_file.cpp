// Native file -> RAM transport fixtures. No GGUF loader, CUDA or dequantization.
#ifdef STRATA_TEST_DEEPSEEK_READER
#include "../deepseek4/expert_file.hpp"
#else
#include "expert_file.hpp"
#endif
#include <algorithm>
#include <array>
#include <cstdio>
#include <exception>
#include <functional>
#include <thread>
#include <winioctl.h>

namespace file = strata_expert_file;
static void check(bool value,const char *message) {
    if(!value)throw std::runtime_error(message);
}
static void rejects(const std::function<void()> &call) {
    try {call();} catch(const std::exception &) {return;}
    throw std::runtime_error("invalid file operation was accepted");
}

struct TempFile {
    FILE *handle=std::tmpfile();
    TempFile() {check(handle!=nullptr,"tmpfile");}
    ~TempFile() {close();}
    void close() {if(handle)std::fclose(handle);handle=nullptr;}
    int fd() const {return _fileno(handle);}
    void write(const std::vector<uint8_t> &bytes) {
        check(std::fwrite(bytes.data(),1,bytes.size(),handle)==bytes.size(),"fixture write");
        check(std::fflush(handle)==0,"fixture flush");
    }
};

struct MappedFile {
    TempFile file;
    HANDLE mapping=nullptr;
    uint8_t *data=nullptr;
    size_t size;
    bool registered=false;
    explicit MappedFile(const std::vector<uint8_t> &bytes):size(bytes.size()) {
        try {
            file.write(bytes);
            mapping=CreateFileMappingW((HANDLE)_get_osfhandle(file.fd()),nullptr,PAGE_READONLY,0,0,nullptr);
            check(mapping!=nullptr,"fixture mapping");
            data=(uint8_t *)MapViewOfFile(mapping,FILE_MAP_READ,0,0,0);
            check(data!=nullptr,"fixture view");
            file::add(data,size,file.fd());registered=true;
        } catch(...) {close();throw;}
    }
    void close() {
        if(registered)file::remove(data);
        registered=false;
        if(data)UnmapViewOfFile(data);
        data=nullptr;
        if(mapping)CloseHandle(mapping);
        mapping=nullptr;
        file.close();
    }
    ~MappedFile() {close();}
};

static void test_quant_ranges() {
    // Independent GGUF block sizes for both GLM profiles; no decoder is tested.
    struct Quant {const char *name;size_t block_bytes;};
    const std::array<Quant,8> quants{{{"IQ2_S",82},{"IQ3_S",110},{"IQ4_XS",136},
        {"Q2_K",84},{"Q3_K",110},{"IQ3_XXS",98},{"Q6_K",210},{"Q4_K",144}}};
    size_t checked=0;
    for(size_t profile=0;profile<quants.size();++profile) {
        // Mixed gate/up/down. Shapes [256,512,10], [256,512,10], [512,256,10].
        // All have 512 blocks/expert. Padding is between tensors, never experts.
        std::array<size_t,3> starts{},sizes{};
        size_t total=96;
        for(size_t projection=0;projection<3;++projection) {
            total=(total+31)/32*32;
            starts[projection]=total;
            sizes[projection]=512*quants[(profile+projection)%quants.size()].block_bytes;
            total+=10*sizes[projection]+17;
        }
        std::vector<uint8_t> expected(total);
        for(size_t i=0;i<total;++i)expected[i]=uint8_t(i*29+(i>>8)*17+(i>>16)+profile*53);
        MappedFile mapped(expected);
        auto source=file::find(mapped.data,mapped.size);
        check(bool(source),"registered source missing");
        std::atomic<bool> cancel{false};
        file::Request request;
        for(size_t expert:{9,0,8,2,7,3,6,4}) {
            for(size_t p=0;p<3;++p) {
                const size_t start=starts[p]+expert*sizes[p],count=sizes[p];
                for(size_t limit:{size_t(997),size_t(65536)}) {
                    std::vector<uint8_t> actual(count+32,0xA5);
                    for(size_t off=0;off<count;off+=limit) {
                        const size_t chunk=std::min(limit,count-off);
                        check(request.read(*source,mapped.data+start+off,actual.data()+16+off,chunk,cancel),"matrix read cancelled");
                    }
                    check(std::equal(actual.begin()+16,actual.end()-16,expected.begin()+start),"matrix bytes differ");
                    check(std::all_of(actual.begin(),actual.begin()+16,[](uint8_t x){return x==0xA5;}) &&
                          std::all_of(actual.end()-16,actual.end(),[](uint8_t x){return x==0xA5;}),"destination guards changed");
                    ++checked;
                }
            }
        }
    }
    check(checked==384,"matrix fixture count");
    std::puts("PASS: 384 matrix reads, 8 quant layouts, 8 experts, gate/up/down, partial chunks and guards");
}

static void test_bounds_and_cancel() {
    MappedFile mapped(std::vector<uint8_t>(4096,0x73));
    auto source=file::find(mapped.data,mapped.size);
    check(!file::find(mapped.data+1,mapped.size),"registry end boundary");
    check(!file::find((void *)(uintptr_t(mapped.data)-1),1),"registry start boundary");
    check(!file::find(mapped.data,std::numeric_limits<size_t>::max()),"registry overflow");
    rejects([&]{file::add(mapped.data,mapped.size,mapped.file.fd());});
    rejects([&]{file::add(mapped.data+1,16,mapped.file.fd());});
    rejects([&]{file::add((void *)(uintptr_t(mapped.data)-1),2,mapped.file.fd());});
    check(file::find(mapped.data,1)==source,"duplicate registration replaced source");
    rejects([&]{file::Source bad((void *)(std::numeric_limits<uintptr_t>::max()-1),8,mapped.file.fd());});
    file::Request request;
    std::atomic<bool> cancel{false};
    std::vector<uint8_t> dest(33,0xA5);
    for(uint64_t offset:{uint64_t(4096),uint64_t(4097),std::numeric_limits<uint64_t>::max()})
        rejects([&]{request.read_at(*source,offset,dest.data(),1,cancel);});
    rejects([&]{request.read(*source,(void *)(source->base-1),dest.data(),1,cancel);});
    rejects([&]{request.read_at(*source,1,dest.data(),std::numeric_limits<size_t>::max(),cancel);});
    rejects([&]{request.read_at(*source,0,nullptr,1,cancel);});
    check(request.read_at(*source,4096,nullptr,0,cancel),"zero read at EOF");
    cancel.store(true);
    check(!request.read_at(*source,0,dest.data(),dest.size(),cancel),"pre-cancel ignored");
    check(std::all_of(dest.begin(),dest.end(),[](uint8_t x){return x==0xA5;}),"cancel or invalid read changed buffer");
    cancel.store(false);
    check(request.read_at(*source,4063,dest.data(),dest.size(),cancel),"reuse after cancel");
    check(std::all_of(dest.begin(),dest.end(),[](uint8_t x){return x==0x73;}),"last file bytes differ");
    check(file::resident(nullptr,0),"empty residency query");
    check(!file::resident((void *)(std::numeric_limits<uintptr_t>::max()-1),8),"residency overflow");
    std::puts("PASS: bounds, overflow, overlapping registration, pre-cancel and request reuse");
}

static void test_lifetime_and_concurrent_reads() {
    std::vector<uint8_t> expected(65537);
    for(size_t i=0;i<expected.size();++i)expected[i]=uint8_t(i*17+(i>>8));
    MappedFile mapped(expected);
    auto source=file::find(mapped.data,mapped.size);
    const auto old_base=mapped.data;
    mapped.close(); // Remove registry, unmap and close original descriptor.
    check(!file::find(old_base,1),"removed mapping still registered");
    std::array<std::exception_ptr,4> errors{};
    std::vector<std::thread> readers;
    for(size_t worker=0;worker<errors.size();++worker)readers.emplace_back([&,worker,source] {
        try {
            file::Request request;
            std::atomic<bool> cancel{false};
            std::vector<uint8_t> actual(997);
            for(size_t i=0;i<40;++i) {
                size_t offset=(i*1543+worker*8731)%(expected.size()-actual.size());
                check(request.read_at(*source,offset,actual.data(),actual.size(),cancel),"concurrent read cancelled");
                check(std::equal(actual.begin(),actual.end(),expected.begin()+offset),"concurrent retained-source bytes differ");
            }
        } catch(...) {errors[worker]=std::current_exception();}
    });
    for(auto &reader:readers)reader.join();
    for(auto &error:errors)if(error)std::rethrow_exception(error);
    // Re-register the same address for another file: old queued work retains
    // its original file identity, independent of the current registry entry.
    TempFile next;
    next.write(std::vector<uint8_t>(expected.size(),0xD2));
    file::add(old_base,expected.size(),next.fd());
    auto replacement=file::find(old_base,1);
    file::remove(old_base);
    check(replacement!=source,"reloaded source aliases previous generation");
    file::Request request;
    std::atomic<bool> cancel{false};
    uint8_t byte=0;
    check(request.read_at(*replacement,29,&byte,1,cancel) && byte==0xD2,"new file identity");
    check(request.read_at(*source,29,&byte,1,cancel) && byte==expected[29],"old file identity");
    const HANDLE retained=source->file;
    source.reset();
    DWORD flags=0;
    check(!GetHandleInformation(retained,&flags) && GetLastError()==ERROR_INVALID_HANDLE,"retained handle leaked");
    std::puts("PASS: 160 concurrent reads after unmap/close, reload identity and handle release");
}

static void test_large_offset_and_short_read() {
    TempFile file;
    HANDLE handle=(HANDLE)_get_osfhandle(file.fd());
    DWORD done=0;
    check(DeviceIoControl(handle,FSCTL_SET_SPARSE,nullptr,0,nullptr,0,&done,nullptr)!=0,"sparse fixture requires sparse-file support");
    constexpr uint64_t offset=(uint64_t(1)<<32)+123;
    check(_fseeki64(file.handle,offset,SEEK_SET)==0,"sparse fixture seek");
    std::vector<uint8_t> expected(1031);
    for(size_t i=0;i<expected.size();++i)expected[i]=uint8_t(i*31+(i>>7));
    file.write(expected);
    file::Source source(nullptr,size_t(offset+expected.size()),file.fd());
    file::Request request;
    std::atomic<bool> cancel{false};
    std::vector<uint8_t> actual(expected.size());
    check(request.read_at(source,offset,actual.data(),actual.size(),cancel),"64-bit offset read");
    check(actual==expected,"64-bit offset bytes differ");
    check(_ftelli64(file.handle)==__int64(offset+expected.size()),"native read moved original file cursor");
    rejects([&]{request.read_at(source,0,actual.data(),size_t(MAXDWORD)+1,cancel);});
    // Deliberately stale size after truncation: fail on short I/O, never expose
    // a partial chunk as a complete matrix. Then verify the request is reusable.
    check(_chsize_s(file.fd(),offset+expected.size()-1)==0,"truncate fixture");
    rejects([&]{request.read_at(source,offset,actual.data(),actual.size(),cancel);});
    check(request.read_at(source,offset,actual.data(),actual.size()-1,cancel),"reuse after short read");
    check(std::equal(actual.begin(),actual.end()-1,expected.begin()),"short-read recovery bytes differ");
    std::puts("PASS: offsets above 4 GiB, chunk-size limit, short-read rejection and recovery");
}

int main() {
    try {
        test_quant_ranges();
        test_bounds_and_cancel();
        test_lifetime_and_concurrent_reads();
        test_large_offset_and_short_read();
        return 0;
    } catch(const std::exception &e) {std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
}
