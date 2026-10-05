#include "../common/expert_pipeline.hpp"
#include "nlohmann/json.hpp"
#include <fstream>
#include <iostream>

static void check(cudaError_t code) {if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));}
static void require(bool ok,const char * why) {if(!ok)throw std::runtime_error(why);}
struct Device {
    uint8_t *data=nullptr;
    explicit Device(size_t bytes) {check(cudaMalloc(reinterpret_cast<void **>(&data),bytes));}
    ~Device(){cudaFree(data);}
};
struct Stream {
    cudaStream_t value=nullptr;
    Stream(){check(cudaStreamCreateWithFlags(&value,cudaStreamNonBlocking));}
    ~Stream(){cudaStreamSynchronize(value);cudaStreamDestroy(value);}
};
int main(int argc,char ** argv) {
    using json=nlohmann::ordered_json;
    json report={{"status","error"},{"scope","real CUDA ring wrap, tail guards, plan mismatch and cancellation/restart"}};
    json cases=json::array();
    try {
        check(cudaSetDevice(0));
        const size_t bytes=18*1024*1024+513,guard=64;
        std::vector<uint8_t> source(bytes),actual(bytes+2*guard);
        for(size_t i=0;i<bytes;++i)source[i]=uint8_t((i*131+i/4093)%251);
        Device target(actual.size());Stream consumer;
        for(int chunk:{4,8,16})for(int readers:{1,2}) {
            StrataExpertPipeline pipe(0,size_t(chunk)<<20,false,readers,0,{},readers,false);
            const std::string name="readers="+std::to_string(readers)+"/chunk="+std::to_string(chunk);
            auto test=[&](const char * suffix,bool ok){cases.push_back({{"name",name+suffix},{"pass",ok}});require(ok,suffix);};
            auto transfer=[&]{
                check(cudaMemsetAsync(target.data,0xFE,actual.size(),consumer.value));
                require(pipe.transfer(target.data+guard,source.data(),bytes,consumer.value),"valid transfer rejected");
                // No per-chunk host synchronization: ready/used events must
                // protect every ring wrap and both pinned/device slot reuse.
                check(cudaStreamSynchronize(consumer.value));pipe.finish();
                check(cudaMemcpy(actual.data(),target.data,actual.size(),cudaMemcpyDeviceToHost));
                return std::equal(source.begin(),source.end(),actual.begin()+guard) &&
                    std::all_of(actual.begin(),actual.begin()+guard,[](uint8_t b){return b==0xFE;}) &&
                    std::all_of(actual.end()-guard,actual.end(),[](uint8_t b){return b==0xFE;});
            };
            pipe.start({{source.data(),bytes,true}});
            test("/tail_guard_parity",transfer());
            auto c=pipe.counters();
            require(c.h2d_bytes==bytes && c.d2d_bytes==bytes && !c.unused_bytes &&
                    c.chunks==(bytes+(size_t(chunk)<<20)-1)/(size_t(chunk)<<20),"chunk accounting failed");
            pipe.start({{source.data(),bytes,true}});
            test("/reject_mismatched_source",!pipe.transfer(target.data+guard,source.data()+1,bytes-1,consumer.value));
            pipe.finish();
            pipe.start({{source.data(),bytes,true},{source.data(),bytes,true}});
            const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
            while(!pipe.ready_chunks() && std::chrono::steady_clock::now()<deadline)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            require(pipe.ready_chunks()>0,"prefetch did not complete before cancellation");
            pipe.finish();
            c=pipe.counters();require(c.unused_bytes>0 && !c.queued_bytes && !c.reader_owned_bytes,"cancel did not drain source views");
            for(auto & value:source)value^=0xA5;
            pipe.start({{source.data(),bytes,false}});
            test("/cancel_restart_fresh_source",transfer());
        }
        require(cases.size()==18,"wrong pipeline case count");report["status"]="pass";
    } catch(const std::exception & error){report["error"]=error.what();}
    report["case_count"]=cases.size();report["cases"]=cases;
    if(argc==2){std::ofstream out(argv[1]);out<<report.dump(2)<<'\n';}
    std::cout<<report["status"]<<": "<<cases.size()<<" pipeline checks\n";
    return report["status"]=="pass"?0:1;
}
