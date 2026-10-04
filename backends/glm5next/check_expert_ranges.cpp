// Read-only Windows GGUF range -> shared pipeline -> GPU -> byte comparison.
// The Python inspector supplies ranges; this executable validates file bounds.
#include "../common/expert_pipeline.hpp"
#include "expert_cache.hpp"
#include "expert_dispatch.hpp"
#include <cstdio>
#include <climits>
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
struct Range {uint64_t offset;size_t bytes;std::string path;strata_glm::ExpertKey key{};};
struct Plan {size_t chunk;int mode;std::vector<Range> ranges;bool cached=false,dispatch=false;};
static Plan parse(std::istream &input) {
    std::string version,chunk,mode,count;
    require(bool(input>>version>>chunk>>mode>>count) &&
            (version=="GLM_RANGES_V1" || version=="GLM_CACHE_RANGES_V1" ||
             version=="GLM_DISPATCH_RANGES_V1"),"invalid range header");
    const auto c=number(chunk),m=number(mode),n=number(count);
    require(c>=1 && c<=16*1024*1024 && m<=2 && n>=1 && n<=4096,"invalid plan limits");
    Plan plan{size_t(c),int(m),{}};
    plan.dispatch=version=="GLM_DISPATCH_RANGES_V1";
    plan.cached=version=="GLM_CACHE_RANGES_V1" || plan.dispatch;
    require(!plan.dispatch || n%3==0,"dispatch requires complete triples");
    std::string identity;
    if(plan.cached) {
        require(bool(input>>identity) && identity.size()==64 &&
                identity.find_first_not_of("0123456789abcdef")==std::string::npos,"invalid model identity");
    }
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
        if(plan.cached) {
            std::string branch,layer,expert,projection,quant,columns,rows,shard;
            require(bool(input>>branch>>layer>>expert>>projection>>quant>>columns>>rows>>shard),"missing cache key fields");
            require(branch=="main" || branch=="mtp","invalid cache branch");
            require(projection=="gate" || projection=="up" || projection=="down","invalid cache projection");
            require(!quant.empty() && quant.size()<=64 &&
                    quant.find_first_not_of("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")==std::string::npos,"invalid quant label");
            const auto l=number(layer),e=number(expert);
            require(l<=INT_MAX && e<=INT_MAX,"cache index exceeds int range");
            using namespace strata_glm;
            auto &r=plan.ranges.back();
            r.key={identity,1,branch=="main"?Branch::main:Branch::mtp,int(l),int(e),
                   projection=="gate"?Projection::gate:projection=="up"?Projection::up:Projection::down,
                   quant,number(columns),number(rows),unhex(shard),r.offset,uint64_t(r.bytes)};
            require(r.key.shard==std::filesystem::u8path(decoded).filename().u8string(),"cache shard differs from source path");
            r.key.validate();
        }
    }
    std::string extra;
    require(!(input>>extra),"extra manifest fields");
    if(plan.dispatch) {
        std::set<strata_glm::ExpertKey> unique;
        for(const auto &r:plan.ranges)
            require(unique.insert(r.key).second,"duplicate dispatch key");
    }
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
    const std::string id(64,'a');
    const auto cache_header="GLM_CACHE_RANGES_V1 262161 1 1 "+id+"\n";
    const auto range="4294967419 1031 "+path+" ";
    const auto fields="mtp 45 287 down Q3_K 2048 4096 "+hex("fixture with spaces.gguf");
    std::istringstream cached(cache_header+range+fields);
    auto cp=parse(cached);
    require(cp.cached && cp.ranges[0].key.model==id && cp.ranges[0].key.generation==1 &&
            cp.ranges[0].key.branch==strata_glm::Branch::mtp && cp.ranges[0].key.layer==45 &&
            cp.ranges[0].key.expert==287 && cp.ranges[0].key.projection==strata_glm::Projection::down &&
            cp.ranges[0].key.quant=="Q3_K" && cp.ranges[0].key.columns==2048 && cp.ranges[0].key.rows==4096 &&
            cp.ranges[0].key.offset==4294967419 && cp.ranges[0].key.bytes==1031,"cache key parse mismatch");
    for(const auto &bad:std::vector<std::string>{
        "GLM_CACHE_RANGES_V1 1 0 1 bad\n"+range+fields,
        cache_header+range,cache_header+range+fields+" extra",
        cache_header+range+"bad 45 287 down Q3_K 2048 4096 "+hex("fixture with spaces.gguf"),
        cache_header+range+"mtp 45 287 bad Q3_K 2048 4096 "+hex("fixture with spaces.gguf"),
        cache_header+range+"mtp 2147483648 287 down Q3_K 2048 4096 "+hex("fixture with spaces.gguf"),
        cache_header+range+"mtp 45 -1 down Q3_K 2048 4096 "+hex("fixture with spaces.gguf"),
        cache_header+range+"mtp 45 287 down Q3_K 0 4096 "+hex("fixture with spaces.gguf"),
        cache_header+range+"mtp 45 287 down Q3_K 2048 4096 "+hex("another.gguf"),
        cache_header+"18446744073709551615 1031 "+path+" "+fields}) {
        bool rejected=false;
        try {std::istringstream broken(bad);parse(broken);}catch(const std::exception &) {rejected=true;}
        require(rejected,"invalid cached manifest accepted");
    }
    std::puts("Cache manifest full keys, identity, branch/projection, shard, overflow and missing fields passed");
    const auto dispatch_header="GLM_DISPATCH_RANGES_V1 262161 2 3 "+id+"\n";
    std::string records;
    for(const auto &projection:{"gate","up","down"})
        records+=range+"mtp 45 287 "+projection+" Q3_K 2048 4096 "+hex("fixture with spaces.gguf")+"\n";
    std::istringstream dispatch(dispatch_header+records);
    require(parse(dispatch).dispatch,"dispatch manifest not selected");
    for(const auto &bad:std::vector<std::string>{
        "GLM_DISPATCH_RANGES_V1 1 0 1 "+id+"\n"+range+fields,
        dispatch_header+range+fields+"\n"+range+fields+"\n"+range+fields,
        dispatch_header+records+"extra"}) {
        bool rejected=false;
        try {std::istringstream broken(bad);parse(broken);}catch(const std::exception &) {rejected=true;}
        require(rejected,"invalid dispatch manifest accepted");
    }
    std::puts("Dispatch manifest selection, complete triples, duplicate keys and trailing fields passed");
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
// Each consecutive triple supplies A, C, D. B uses C's actual bytes with a
// distinct generation, giving a same-sized eviction victim even for mixed quants.
// Serialized byte checks intentionally do not measure overlap or throughput.
static void run_dispatch(const Plan &plan,
                         const std::map<std::string,std::shared_ptr<MappedSource>> &sources,size_t largest) {
    using namespace strata_glm;
    constexpr size_t guard=37;
    Device first(largest+2*guard),second(largest+2*guard);
    std::vector<ExpertSourceView> views;
    for(const auto &s:sources)for(uint64_t generation:{1,2,3})
        views.push_back({plan.ranges[0].key.model,generation,
                         std::filesystem::u8path(s.first).filename().u8string(),
                         s.second->view,s.second->bytes,s.second});
    for(size_t base=0;base<plan.ranges.size();base+=3) {
        std::vector<std::vector<uint8_t>> expected;
        for(size_t j=0;j<3;++j) {
            const auto &r=plan.ranges[base+j];
            expected.emplace_back(r.bytes+2*guard,0xA5);
            sources.at(r.path)->baseline(r,expected.back().data()+guard);
        }
        for(bool frequency:{false,true})for(bool decode:{false,true}) {
            const auto a=plan.ranges[base].key,c=plan.ranges[base+1].key,d=plan.ranges[base+2].key;
            auto b=c;b.generation=2;
            ExpertCache cache(size_t(a.bytes+c.bytes),{frequency,4096,131072});
            ExpertTransport transport(0,plan.chunk,false,4,plan.mode);
            size_t comparisons=0;
            auto compare=[&](ExpertDispatch &dispatch,size_t index,size_t which) {
                Device &device=comparisons%2?second:first;
                const auto &want=expected[which];
                cuda_ok(cudaMemsetAsync(device.data,0xA5,want.size(),device.stream));
                dispatch.copy(index,device.data+guard,device.stream);
                cuda_ok(cudaStreamSynchronize(device.stream));
                std::vector<uint8_t> actual(want.size());
                cuda_ok(cudaMemcpy(actual.data(),device.data,actual.size(),cudaMemcpyDeviceToHost));
                require(actual==want,"dispatch GPU payload or guard mismatch");++comparisons;
            };
            auto sync=[&] {
                // finish() releases leases and records events after the byte
                // check; wait for those events before deterministic eviction.
                cuda_ok(cudaStreamSynchronize(first.stream));cuda_ok(cudaStreamSynchronize(second.stream));
            };
            auto exercise=[&](std::vector<ExpertKey> keys,std::vector<size_t> indices,
                              std::vector<ExpertKey> misses,bool all_hit=false) {
                const auto before=transport.counters();
                ExpertDispatch dispatch(cache,transport,std::move(keys),all_hit?std::vector<ExpertSourceView>{}:views,decode);
                for(size_t i=0;i<indices.size();++i)compare(dispatch,i,indices[i]);
                dispatch.finish();sync();
                uint64_t bytes=0,chunks=0;
                for(const auto &key:misses) {bytes+=key.bytes;chunks+=(key.bytes+plan.chunk-1)/plan.chunk;}
                const auto after=transport.counters();
                require(after.h2d_bytes-before.h2d_bytes==bytes && after.d2d_bytes-before.d2d_bytes==bytes &&
                        after.file_bytes-before.file_bytes+after.mmap_bytes-before.mmap_bytes==bytes &&
                        after.chunks-before.chunks==chunks && after.unused==before.unused,
                        "dispatch phase transport counters mismatch");
                if(all_hit)require(after.groups==before.groups,"all-hit dispatch submitted source jobs");
            };
            exercise({a,b},{0,1},{a,b});
            exercise({c,a,d},{1,0,2},{c,d}); // C evicts B, A stays pinned, D bypasses.
            auto counts=cache.counters();
            require(counts.hits==1 && counts.misses==4 && counts.admissions==3 &&
                    counts.evictions==1 && counts.bypasses==1 && cache.resident(a) && cache.resident(c) &&
                    !cache.resident(b) && !cache.resident(d),"mixed dispatch cache counters/residency mismatch");
            exercise({c,a},{1,0},{},true); // No source views, reads, H2D or groups.
            {
                ExpertDispatch dispatch(cache,transport,{a,d},views,decode);
                compare(dispatch,0,0);dispatch.cancel();sync();
                require(!transport.in_progress(),"cancel left active dispatch transport");
            }
            require(cache.invalidate(a.model,1)==2 && cache.resident_bytes()==0,
                    "cancel/invalidation retained cache pins or leases");
            auto aa=a,cc=c,dd=d;aa.generation=cc.generation=dd.generation=3;
            exercise({aa,cc,dd},{0,1,2},{aa,cc,dd});
            require(cache.set_budget(0) && cache.resident_bytes()==0,"dispatch left protected allocations");
            exercise({a,c,d},{0,1,2},{a,c,d}); // All bypass; full payload still delivered.
            counts=cache.counters();
            require(comparisons==14 && counts.hits==4 && counts.misses==10 && counts.admissions==5 &&
                    counts.evictions==3 && counts.bypasses==5 && counts.invalidations==2 &&
                    counts.admission_rejects==0 && cache.resident_bytes()==0,"dispatch lifecycle mismatch");
            const auto t=transport.counters();
            const uint64_t delivered=3*a.bytes+4*c.bytes+3*d.bytes,source=t.file_bytes+t.mmap_bytes;
            require(t.d2d_bytes==delivered && t.h2d_bytes>=delivered && source>=t.h2d_bytes &&
                    source<=delivered+d.bytes,"cancel read-ahead counters out of bounds");
            // Auto prefill uses mmap for resident pages and native reads for
            // cold ones; earlier scenarios can warm the mapping. Decode uses mmap.
            if(plan.mode==1)require(t.file_bytes==source,"native dispatch reader policy mismatch");
            if(plan.mode==0 || (plan.mode==2 && decode))
                require(t.mmap_bytes==source,"mmap dispatch reader policy mismatch");
            std::printf("DISPATCH_OK %zu %d %d %zu %llu %llu %llu %llu %llu\n",base/3,int(frequency),int(decode),comparisons,
                        (unsigned long long)t.h2d_bytes,(unsigned long long)t.d2d_bytes,
                        (unsigned long long)t.file_bytes,(unsigned long long)t.mmap_bytes,(unsigned long long)t.unused);
        }
    }
}
static void run(const Plan &plan) {
    static_assert(sizeof(size_t)==8,"64-bit process required");
    std::map<std::string,std::shared_ptr<MappedSource>> sources;
    std::vector<StrataExpertSlice> slices;
    size_t largest=0;uint64_t total=0,chunks=0;
    for(const auto &r:plan.ranges) {
        auto &source=sources[r.path];
        if(!source)source=std::make_shared<MappedSource>(r.path);
        require(r.offset<=source->bytes && r.bytes<=source->bytes-r.offset,"range outside source file");
        slices.push_back({source->view+r.offset,r.bytes,false});
        largest=std::max(largest,r.bytes);total+=r.bytes;chunks+=(r.bytes+plan.chunk-1)/plan.chunk;
    }
    cudaDeviceProp props{};cuda_ok(cudaGetDeviceProperties(&props,0));
    int runtime=0,driver=0;cuda_ok(cudaRuntimeGetVersion(&runtime));cuda_ok(cudaDriverGetVersion(&driver));
    std::printf("GPU %s %d %d\n",hex(props.name).c_str(),runtime,driver);
    if(plan.dispatch) {run_dispatch(plan,sources,largest);return;}
    constexpr size_t guard=37;
    Device device(largest+2*guard);
    std::unique_ptr<Device> hit_device;
    if(plan.cached)hit_device=std::make_unique<Device>(largest+2*guard);
    StrataExpertPipeline pipeline(0,plan.chunk,false,4,plan.mode);
    if(!plan.cached)pipeline.start(slices);
    for(size_t i=0;i<slices.size();++i) {
        const auto &r=plan.ranges[i];
        std::vector<uint8_t> expected(r.bytes+2*guard,0xA5),actual(expected.size());
        sources.at(r.path)->baseline(r,expected.data()+guard); // Independent stdio path.
        if(plan.cached) {
            // Deliberately one matrix of capacity: every range exercises eviction
            // independent of its quant/size. This is a serialized correctness test.
            strata_glm::ExpertCache cache(r.bytes);
            auto exercise=[&](const strata_glm::ExpertKey &key,Device &consumer,bool miss) {
                const auto before=pipeline.counters();bool uploaded=false;
                auto lease=cache.get(key,sources.at(r.path),consumer.stream,[&](void *dest,size_t bytes,cudaStream_t stream) {
                    require(miss,"cache hit unexpectedly uploaded");uploaded=true;
                    pipeline.start({slices[i]});
                    require(pipeline.transfer(dest,slices[i].data,bytes,stream),"unplanned cache transfer");
                });
                pipeline.finish();
                require(bool(lease) && uploaded==miss,"missing cache lease or expected upload");
                cuda_ok(cudaMemsetAsync(consumer.data,0xA5,expected.size(),consumer.stream));
                cuda_ok(cudaMemcpyAsync(consumer.data+guard,lease.data(),r.bytes,cudaMemcpyDeviceToDevice,consumer.stream));
                lease.release();
                cuda_ok(cudaStreamSynchronize(consumer.stream));
                cuda_ok(cudaMemcpy(actual.data(),consumer.data,actual.size(),cudaMemcpyDeviceToHost));
                require(actual==expected,"cached GPU payload or guard mismatch");
                const auto after=pipeline.counters();
                const uint64_t bytes=miss?r.bytes:0,nchunks=miss?(r.bytes+plan.chunk-1)/plan.chunk:0;
                require(after.h2d_bytes-before.h2d_bytes==bytes && after.d2d_bytes-before.d2d_bytes==bytes &&
                        after.file_bytes-before.file_bytes+after.mmap_bytes-before.mmap_bytes==bytes &&
                        after.chunks-before.chunks==nchunks,"cache stage transport counters mismatch");
            };
            exercise(r.key,device,true);           // Cold insertion, generation 1.
            exercise(r.key,*hit_device,false);     // Hit on another stream; no read or H2D.
            auto pressure=r.key;pressure.generation=2;
            exercise(pressure,device,true);        // Full cache forces eviction of generation 1.
            exercise(r.key,device,true);           // Original must reload, evicting generation 2.
            require(cache.invalidate(r.key.model,1)==1 && cache.size()==0 && cache.resident_bytes()==0,
                    "cache invalidation failed to retire original generation");
            auto reload=r.key;reload.generation=3;
            exercise(reload,device,true);          // Fresh loader generation after invalidation.
            auto stats=cache.counters();
            require(stats.hits==1 && stats.misses==4 && stats.admissions==4 && stats.evictions==2 &&
                    stats.invalidations==1 && stats.bypasses==0 && cache.resident_bytes()==r.bytes,
                    "cache lifecycle counters mismatch");
            std::printf("CACHE_OK %zu %zu 5 1 4 2 1\n",i,r.bytes);
            continue;
        }
        cuda_ok(cudaMemsetAsync(device.data,0xA5,expected.size(),device.stream));
        require(pipeline.transfer(device.data+guard,slices[i].data,r.bytes,device.stream),"unplanned transfer");
        cuda_ok(cudaStreamSynchronize(device.stream));
        cuda_ok(cudaMemcpy(actual.data(),device.data,actual.size(),cudaMemcpyDeviceToHost));
        require(actual==expected,"GPU payload or guard mismatch");
        std::printf("OK %zu %zu\n",i,r.bytes);
    }
    pipeline.finish();
    if(plan.cached) {total*=4;chunks*=4;}
    auto c=pipeline.counters();
    require(c.h2d_bytes==total && c.d2d_bytes==total && c.file_bytes+c.mmap_bytes==total &&
            c.chunks==chunks && c.unused==0,"transport counters differ from plan");
    if(plan.mode==0)require(c.mmap_bytes==total,"mmap path not exercised");
    if(plan.mode==1)require(c.file_bytes==total,"native path not exercised");
    std::printf("TOTAL %llu %llu %llu %llu %llu\n",(unsigned long long)c.h2d_bytes,
                (unsigned long long)c.d2d_bytes,(unsigned long long)c.file_bytes,
                (unsigned long long)c.mmap_bytes,(unsigned long long)c.chunks);
    if(plan.cached) {
        const auto n=(unsigned long long)plan.ranges.size();
        std::printf("CACHE_TOTAL %llu %llu %llu %llu %llu\n",n,4*n,2*n,n,5*n);
    }
}
int main(int argc,char **argv) {
    try {
        if(argc==2 && std::string(argv[1])=="--test-parser") {parser_tests();return 0;}
        require(argc==1,"ranges must be provided on stdin; no arguments accepted");
        run(parse(std::cin));return 0;
    }catch(const std::exception &e) {std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
}
