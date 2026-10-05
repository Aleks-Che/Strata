#include "runtime.hpp"
#include "expert_cache.hpp"
#include "../common/expert_pipeline.hpp"
#include "llama-mmap.h"
#include "nlohmann/json.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>

using namespace step35;
using json=nlohmann::ordered_json;
struct Mapping {
    llama_file file;
    llama_mmap view;
    std::unique_ptr<gguf_context,decltype(&gguf_free)> header;
    std::unique_ptr<ggml_context,decltype(&ggml_free)> tensors{nullptr,ggml_free};
    explicit Mapping(const std::string & path):file(path.c_str(),"rb"),view(&file,0),header(nullptr,gguf_free) {
        ggml_context * ctx=nullptr;
        header.reset(gguf_init_from_file(path.c_str(),{true,&ctx}));tensors.reset(ctx);
        require(bool(header) && bool(tensors),"cannot read shard metadata");
    }
};
struct Device {
    uint8_t * data=nullptr;
    explicit Device(size_t bytes){cuda_check(cudaMalloc(reinterpret_cast<void **>(&data),bytes));}
    ~Device(){cudaFree(data);}
};
struct Stream {
    cudaStream_t value=nullptr;
    Stream(){cuda_check(cudaStreamCreateWithFlags(&value,cudaStreamNonBlocking));}
    ~Stream(){cudaStreamSynchronize(value);cudaStreamDestroy(value);}
};
struct Slice {const uint8_t *source;size_t bytes;MatrixKey key;std::string path,name;size_t offset;};
int main(int argc,char ** argv) {
    json report={{"status","error"},{"scope","real GGUF mmap-to-GPU byte parity, guards, cache and reload"}};
    json cases=json::array();
    bool output_safe=false;
    try {
        require(argc==3,"usage: strata-step35-shards-check MANIFEST.json REPORT.json");
        std::ifstream input(argv[1]);json plan;input>>plan;
        const auto & samples=plan.at("samples");
        require(samples.is_array() && !samples.empty() && samples.size()<=1024,"invalid sample count");
        require(std::filesystem::path(argv[2]).extension()==".json","report must be JSON");
        if(std::filesystem::exists(argv[2])) {
            require(!std::filesystem::equivalent(argv[2],argv[1]),"report aliases manifest");
            for(const auto & s:samples)require(!std::filesystem::equivalent(argv[2],s.at("path").get<std::string>()),"report aliases shard");
        }
        output_safe=true;
        std::map<std::string,std::unique_ptr<Mapping>> mappings;
        std::vector<Slice> slices;
        size_t total=0,max_bytes=0;
        std::set<std::tuple<std::string,std::string,int>> identities;
        for(const auto & s:samples) {
            const std::string path=s.at("path"),name=s.at("tensor");const int id=s.at("expert");
            require(expert(name) && identities.emplace(path,name,id).second,"invalid/duplicate sample identity");
            if(!mappings.count(path))mappings[path]=std::make_unique<Mapping>(path);
            auto & m=*mappings.at(path);auto * t=ggml_get_tensor(m.tensors.get(),name.c_str());
            require(t && t->type==GGML_TYPE_Q4_K && ggml_is_contiguous(t) && t->ne[3]==1 && id>=0 && id<t->ne[2],"unsupported sample tensor");
            const auto index=gguf_find_tensor(m.header.get(),name.c_str());
            require(index>=0,"missing sample tensor");
            const size_t offset=gguf_get_data_offset(m.header.get())+gguf_get_tensor_offset(m.header.get(),index)+id*t->nb[2];
            const size_t bytes=t->nb[2]+(id+1<t->ne[2]?std::min<size_t>(512,t->nb[2]):0);
            require(offset==s.at("file_offset").get<size_t>() && bytes==s.at("bytes").get<size_t>(),"independent Python/native range mismatch");
            require(bytes && bytes<=128*1024*1024 && offset<=m.view.size() && bytes<=m.view.size()-offset,"sample outside mapping");
            total+=bytes;require(total<=size_t(1)<<30,"sample set too large");max_bytes=std::max(max_bytes,bytes);
            slices.push_back({static_cast<const uint8_t *>(m.view.addr())+offset,bytes,{1,uint32_t(slices.size()),uint32_t(id)},path,name,offset});
        }
        cuda_check(cudaSetDevice(0));const auto memory=memory_sample();
        require(memory.gpu_free>memory.gpu_total/20+(512<<20) && memory.ram_free>memory.ram_total/20+(512<<20),"insufficient test headroom");
        const size_t guard=256;
        Device output(max_bytes+2*guard);Stream streams[2];
        std::vector<uint8_t> actual(max_bytes+2*guard),expected(max_bytes);
        auto verify=[&](const Slice & s,cudaStream_t stream){
            cuda_check(cudaStreamSynchronize(stream));
            cuda_check(cudaMemcpy(actual.data(),output.data,s.bytes+2*guard,cudaMemcpyDeviceToHost));
            std::ifstream oracle(s.path,std::ios::binary);oracle.seekg(s.offset);oracle.read(reinterpret_cast<char *>(expected.data()),s.bytes);
            require(size_t(oracle.gcount())==s.bytes,"short independent file read");
            require(std::memcmp(expected.data(),s.source,s.bytes)==0,"mmap differs from independent file read");
            return std::memcmp(expected.data(),actual.data()+guard,s.bytes)==0 &&
                std::all_of(actual.begin(),actual.begin()+guard,[](uint8_t v){return v==0xCD;}) &&
                std::all_of(actual.begin()+guard+s.bytes,actual.begin()+2*guard+s.bytes,[](uint8_t v){return v==0xCD;});
        };
        for(int chunk:{4,8,16})for(int readers:{1,2}) {
            // Batches of nine matrices fit in this small cache. Real VRAM
            // budgets still come from the production controller.
            ExpertCache cache(64<<20);cache.refresh();
            StrataExpertPipeline pipe(0,size_t(chunk)<<20,false,readers,0,{},readers,false);
            for(size_t first=0;first<slices.size();first+=9) {
                const auto end=std::min(first+9,slices.size());
                std::vector<StrataExpertSlice> inputs;std::vector<MatrixKey> keys;
                for(size_t i=first;i<end;++i){inputs.push_back({slices[i].source,slices[i].bytes,true});keys.push_back(slices[i].key);}
                pipe.start(inputs);
                for(size_t i=first;i<end;++i) {
                    const auto & s=slices[i];auto stream=streams[i%2].value;
                    require(!cache.get(s.key,s.bytes),"unexpected cold hit");
                    cuda_check(cudaMemsetAsync(output.data,0xCD,s.bytes+2*guard,stream));
                    require(pipe.transfer(output.data+guard,s.source,s.bytes,stream),"planned transfer refused");
                    const bool ok=verify(s,stream);
                    cases.push_back({{"sample",i},{"readers",readers},{"chunk_mib",chunk},{"phase","cold"},{"pass",ok}});
                    require(ok,"cold bytes/guards differ");
                    auto * cached=cache.admit(s.key,s.bytes);require(cached!=nullptr,"sample cache admission refused");
                    cuda_check(cudaMemcpyAsync(cached,output.data+guard,s.bytes,cudaMemcpyDeviceToDevice,stream));
                    cuda_check(cudaStreamSynchronize(stream));
                }
                pipe.finish();auto pins=cache.protect(keys);const auto resident=cache.resident_bytes();
                cache.trim(0);require(cache.resident_bytes()==resident,"future hits evicted while pinned");
                for(size_t i=first;i<end;++i) {
                    const auto & s=slices[i];auto stream=streams[i%2].value;auto * hit=cache.get(s.key,s.bytes);
                    require(hit!=nullptr,"missing pinned hit");
                    cuda_check(cudaMemsetAsync(output.data,0xCD,s.bytes+2*guard,stream));
                    cuda_check(cudaMemcpyAsync(output.data+guard,hit,s.bytes,cudaMemcpyDeviceToDevice,stream));
                    const bool ok=verify(s,stream);
                    cases.push_back({{"sample",i},{"readers",readers},{"chunk_mib",chunk},{"phase","warm_pinned"},{"pass",ok}});
                    require(ok,"cache bytes/guards differ");
                }
                pins.reset();cache.trim(0);require(cache.resident_bytes()==0,"trim retained allocations");cache.refresh();
            }
            const auto c=pipe.counters();require(c.h2d_bytes==total && c.d2d_bytes==total && !c.unused_bytes,"transport accounting differs");
            // Abort an uploaded suffix, destroy all workers, then reload maps
            // below. Sources remain alive throughout finish and destruction.
            pipe.start({{slices.front().source,slices.front().bytes,true}});pipe.finish();
            require(!pipe.counters().queued_bytes && !pipe.counters().reader_owned_bytes,"source views not drained");
        }
        const auto sample=samples.front();slices.clear();mappings.clear();
        Mapping reloaded(sample.at("path").get<std::string>());
        const auto bytes=sample.at("bytes").get<size_t>(),offset=sample.at("file_offset").get<size_t>();
        Slice fresh{static_cast<const uint8_t *>(reloaded.view.addr())+offset,bytes,{2,0,0},sample.at("path"),sample.at("tensor"),offset};
        StrataExpertPipeline restarted(0,4<<20,false,1,0);restarted.start({{fresh.source,bytes,true}});
        cuda_check(cudaMemsetAsync(output.data,0xCD,bytes+2*guard,streams[0].value));
        require(restarted.transfer(output.data+guard,fresh.source,bytes,streams[0].value),"reload transfer refused");
        const bool ok=verify(fresh,streams[0].value);restarted.finish();
        cases.push_back({{"phase","reload"},{"pass",ok}});require(ok,"reload bytes differ");
        report.update({{"status","pass"},{"sample_count",samples.size()},{"sample_bytes",total},{"mapped_shards",plan.at("payload_shards")}});
    } catch(const std::exception & e){report["error"]=e.what();}
    report["case_count"]=cases.size();report["cases"]=cases;
    if(output_safe){std::ofstream out(argv[2]);out<<report.dump(2)<<'\n';}
    std::cout<<report["status"]<<": "<<cases.size()<<" real-shard byte checks\n";
    return report["status"]=="pass"?0:1;
}
