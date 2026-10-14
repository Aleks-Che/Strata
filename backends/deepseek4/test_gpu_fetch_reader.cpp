#include "gpu_fetch_reader.hpp"
#include <cstdio>
#include <share.h>
#include <string>

static void check(bool ok,const char *message) {if(!ok)throw std::runtime_error(message);}
int main(int argc,char **argv) {try {
    check(argc==2,"fixture path required");
    constexpr size_t bytes=(3<<20)+1337;
    std::vector<uint8_t> expected(bytes),output(bytes+2,171);
    for(size_t i=0;i<bytes;++i)expected[i]=uint8_t((i*37+i/251)%253);
    FILE *file=_fsopen(argv[1],"w+b",_SH_DENYNO);check(file,"fixture open");
    check(fwrite(expected.data(),1,bytes,file)==bytes && !fflush(file),"fixture write");
    HANDLE mapping=CreateFileMappingW((HANDLE)_get_osfhandle(_fileno(file)),nullptr,PAGE_READONLY,0,0,nullptr);
    check(mapping,"create mapping");
    auto *view=(const uint8_t *)MapViewOfFile(mapping,FILE_MAP_READ,0,0,bytes);check(view,"map view");
    {
        strata_expert_file::Source source(view,bytes,_fileno(file));
        // Reserved pages are definitely nonresident, so this exercises the
        // residency fallback without evicting unrelated process/OS caches.
        void *absent=VirtualAlloc(nullptr,bytes,MEM_RESERVE,PAGE_NOACCESS);check(absent,"reserve absent view");
        strata_expert_file::Source cold(absent,bytes,_fileno(file));
        fclose(file); // the reader's separate handle survives its original fd
        strata_ds4::GpuFetchReader reader;strata_expert_file::Request local;
        std::atomic<bool> cancel{false};
        for(auto mode:{strata_ds4::GpuFetchReader::native,strata_ds4::GpuFetchReader::reuse,
                       strata_ds4::GpuFetchReader::resident,strata_ds4::GpuFetchReader::mapped}) {
            reader.mode=mode;
            for(int round=0;round<3;++round) {
                std::fill(output.begin(),output.end(),171);
                // Unaligned offsets, ring-sized chunks, a short final chunk.
                for(size_t off=1;off<bytes;) {
                    size_t n=std::min<size_t>((1<<20)+7,bytes-off);
                    check(reader.read(local,source,off,view+off,output.data()+off,n,cancel),"read failed");off+=n;
                }
                check(!memcmp(output.data()+1,expected.data()+1,bytes-1),"payload mismatch");
                check(output[0]==171 && output[bytes]==171 && output[bytes+1]==171,"read outside destination");
            }
            auto files=reader.file_bytes;
            check(reader.read(local,source,bytes-17,nullptr,output.data(),17,cancel),"missing view fallback");
            check(reader.file_bytes==files+17 && !memcmp(output.data(),expected.data()+bytes-17,17),"fallback payload/counter");
            cancel.store(true);output[0]=171;
            check(!reader.read(local,source,0,view,output.data(),1,cancel) && output[0]==171,"cancel wrote destination");
            cancel.store(false);
            bool rejected=false;
            try {reader.read(local,source,bytes-1,view+bytes-1,output.data(),2,cancel);}catch(const std::runtime_error&){rejected=true;}
            check(rejected,"out-of-bounds accepted");
            check(reader.read(local,source,bytes,nullptr,nullptr,0,cancel),"empty read");
        }
        reader.mode=strata_ds4::GpuFetchReader::mapped;bool rejected=false;
        try {reader.read(local,source,1,view+2,output.data(),17,cancel);}catch(const std::runtime_error&){rejected=true;}
        check(rejected,"wrong view offset accepted");
        check(reader.mmap_bytes && reader.file_bytes && reader.resident_checks,"paths not exercised");
        reader.mode=strata_ds4::GpuFetchReader::resident;
        auto misses=reader.resident_misses;
        check(reader.read(local,cold,0,absent,output.data(),17,cancel),"nonresident fallback failed");
        check(reader.resident_misses==misses+1 && !memcmp(output.data(),expected.data(),17),"nonresident fallback payload/counter");
        VirtualFree(absent,0,MEM_RELEASE);
        // No borrowed view: native fallback remains valid after unmapping too.
        UnmapViewOfFile(view);CloseHandle(mapping);
        check(reader.read(local,source,bytes-17,nullptr,output.data(),17,cancel),"fallback after unmap");
        check(!memcmp(output.data(),expected.data()+bytes-17,17),"fallback after unmap payload");
    }
    std::puts("GPU fetch reader: byte equality, bounds, cancellation, fallback and lifetime passed");return 0;
}catch(const std::exception &e){std::fprintf(stderr,"ERROR: %s\n",e.what());return 1;}}
