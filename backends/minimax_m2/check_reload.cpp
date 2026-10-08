// Diagnose history-dependent outputs across cached/native model lifetimes.
// A fresh native run is the exact oracle; resident weights are a second oracle.
#include "runtime.hpp"
#include "synthetic_minimax_m2.hpp"
#include "../common/expert_file.hpp"
#include "nlohmann/json.hpp"
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
using namespace minimax_m2;
using json=nlohmann::ordered_json;
using Floats=std::vector<float>;
struct Run {Floats logits,continuation;std::vector<llama_token> ids;};
static Run evaluate(llama_context *ctx,int count) {
    clear(ctx);strata_mm27_reset();strata_mm27_cache_decode(false);
    std::vector<llama_token> prompt(count,11);Run r;
    if(count>64)for(int i=count-33;i<count;++i)prompt[i]=(11+(i*7)%32)%64;
    const auto capture=[&](Floats &out) {
        const auto *l=llama_get_logits_ith(ctx,-1);require(l,"missing logits");
        require(std::all_of(l,l+64,[](float x){return std::isfinite(x);}),"nonfinite logits");
        out.insert(out.end(),l,l+64);
    };
    for(int i=0;i<count;i+=16) {
        decode(ctx,prompt,i,std::min(16,count-i),i);capture(r.logits);
    }
    for(int i=0;i<8;++i) {
        capture(r.continuation);
        const auto *l=llama_get_logits_ith(ctx,-1);
        const auto next=llama_token(std::max_element(l,l+64)-l);r.ids.push_back(next);
        if(i+1<8) {
            decode(ctx,{next},0,1,count+i);capture(r.logits);strata_mm27_cache_decode(true);
        }
    }
    return r;
}
static json compare(const std::string &name,const Run &a,const Run &b,bool exact=true) {
    require(a.logits.size()==b.logits.size() && !a.logits.empty(),"comparison size");
    double error=0,energy=0,delta=0;size_t first=a.logits.size();
    for(size_t i=0;i<a.logits.size();++i) {
        const double d=double(a.logits[i])-b.logits[i];error+=d*d;energy+=double(b.logits[i])*b.logits[i];
        delta=std::max(delta,std::abs(d));if(a.logits[i]!=b.logits[i])first=std::min(first,i);
    }
    const bool bits=std::memcmp(a.logits.data(),b.logits.data(),a.logits.size()*sizeof(float))==0;
    const double nmse=error/std::max(energy,1e-30);
    return {{"name",name},{"pass",delta<=5e-4 && nmse<=1e-7 && (!exact || bits) && a.ids==b.ids},
        {"bit_exact",bits},{"exact_required",exact},{"max_abs",delta},{"nmse",nmse},
        {"elements",a.logits.size()},{"first_different",first},{"greedy_ids_match",a.ids==b.ids}};
}
static void save(const std::filesystem::path &p,const Floats &values) {
    std::ofstream out(p,std::ios::binary);out.write(reinterpret_cast<const char *>(values.data()),values.size()*sizeof(float));
    require(bool(out),"cannot write logits");
}
struct Poison {
    std::vector<void *> allocations;
    ~Poison() {for(void *p:allocations)cudaFree(p);}
    explicit Poison(int round) {
        // Distinct allocation sizes and contents expose reuse/padding dependence.
        for(size_t mib:{13,17,64,96}) {
            strata_mm27_memory((mib+256)<<20,256u<<20);void *p=nullptr;
            require(cudaMalloc(&p,mib<<20)==cudaSuccess,"poison allocation failed");allocations.push_back(p);
            require(cudaMemset(p,round%2?0xff:0x5a,mib<<20)==cudaSuccess,"poison fill failed");
        }
        require(cudaDeviceSynchronize()==cudaSuccess,"poison synchronization failed");
        // Keep one block live in odd rounds to change later allocation addresses.
        for(size_t i=round%2?1:0;i<allocations.size();++i)require(cudaFree(allocations[i])==cudaSuccess,"poison free failed");
        allocations.resize(round%2?1:0);
    }
};
int main(int argc,char **argv) {
    json report={{"source_revision",STRATA_MM27_SOURCE_SHA},{"patches",STRATA_MM27_PATCH_SET},
        {"scope","synthetic long-context reload stress; every prefill batch and decode step"},{"tests",json::array()}};
    json tests=json::array();std::filesystem::path dir;bool created=false;
    try {
        bool fingerprints=false;int reader=0,pipeline_readers=0,pipeline_chunk=8;bool pipeline_lookahead=false,pipeline_d2d_batch=false;
        require(argc>=2,"usage: reload-check NEW_DIRECTORY [--expert-reader file|mmap|mmap-direct|mmap-decode] [--precision-fingerprints]");
        for(int i=2;i<argc;++i) {
            const std::string arg=argv[i];
            if(arg=="--precision-fingerprints")fingerprints=true;
            else if(arg=="--expert-reader" && i+1<argc) {
                const std::string value=argv[++i];require(value=="file" || value=="mmap" || (value=="mmap-direct" || value=="mmap-decode"),"invalid reader");
                reader=value=="file"?0:value=="mmap"?1:value=="mmap-direct"?2:3;
            } else if(arg=="--pipeline-readers" && i+1<argc) {const std::string v=argv[++i];require(v=="0" || v=="1" || v=="2","invalid pipeline readers");pipeline_readers=std::stoi(v);}
            else if(arg=="--pipeline-chunk-mib" && i+1<argc) {const std::string v=argv[++i];require(v=="4" || v=="8" || v=="16","invalid pipeline chunk");pipeline_chunk=std::stoi(v);}
            else if(arg=="--pipeline-lookahead" && i+1<argc) {const std::string v=argv[++i];require(v=="0" || v=="1","invalid pipeline chunk");pipeline_lookahead=v=="1";}
            else if(arg=="--pipeline-d2d-batch" && i+1<argc) {const std::string v=argv[++i];require(v=="0" || v=="1","invalid pipeline chunk");pipeline_d2d_batch=v=="1";}
            else throw std::runtime_error("unknown reload argument: "+arg);
        }
        report["reader"]=reader;dir=argv[1];require(!std::filesystem::exists(dir),"directory exists");
        report["pipeline_readers"]=pipeline_readers;report["pipeline_chunk_mib"]=pipeline_chunk;report["pipeline_lookahead"]=pipeline_lookahead;report["pipeline_d2d_batch"]=pipeline_d2d_batch;
        std::filesystem::create_directories(dir);created=true;environment();ggml_backend_load_all();
        struct Release {~Release(){strata_mm27_release();}} release;
        const auto path=(dir/"fixture.gguf").string();write_synthetic_minimax_m2(path,true);
        Run baseline;
        {
            strata_mm27_mode(1);auto model=load(path,false,true);auto ctx=context(model.get(),4096,16);
            baseline=evaluate(ctx.get(),4079);save(dir/"fresh-native.f32",baseline.continuation);
        }
        strata_mm27_release();
        if(fingerprints) {
            // Diagnostic only: deliberately change numerical modes after the
            // strict baseline. Never used by the benchmark or serving runtime.
            struct Restore {
                ~Restore(){_putenv_s("GGML_CUDA_CUBLAS_COMPUTE_TYPE","f32");_putenv_s("NVIDIA_TF32_OVERRIDE","0");}
            } restore;
            report["precision_fingerprints"]=json::array();
            for(const std::string mode:{"f16","bf16","explicit_tf32"}) {
                _putenv_s("GGML_CUDA_CUBLAS_COMPUTE_TYPE",mode=="explicit_tf32"?"f32":mode.c_str());
                _putenv_s("NVIDIA_TF32_OVERRIDE",mode=="explicit_tf32"?"1":"0");
                strata_mm27_mode(1);
                {
                    auto model=load(path,false,true);auto ctx=context(model.get(),4096,16);
                    const auto value=evaluate(ctx.get(),4079);save(dir/("fingerprint-"+mode+".f32"),value.continuation);
                    report["precision_fingerprints"].push_back(compare(mode,value,baseline,false));
                }
                strata_mm27_release();
            }
            report["status"]="DIAGNOSTIC";report["scope"]="diagnostic numerical fingerprints only; not a correctness gate";
            std::ofstream(dir/"reload-report.json")<<report.dump(2)<<'\n';
            return 0;
        }
        {
            strata_mm27_mode(1);auto model=load(path,true,true);auto ctx=context(model.get(),4096,16);
            const auto resident=evaluate(ctx.get(),4079);tests.push_back(compare("resident_vs_fresh_native",resident,baseline,false));
        }
        strata_mm27_release();
        for(int round=0;round<9;++round) {
            const int variant=round%3;const std::string label=std::to_string(round)+"/"+(variant==0?"off":variant==1?"cuda":"arena");
            std::cerr<<"Reload round "<<label<<'\n';
            {
                strata_mm27_mode(2);strata_mm27_cache_configure(variant?256u<<20:0,variant==2);
                strata_mm27_reader(reader);
                strata_mm27_pipeline(pipeline_readers,pipeline_chunk,pipeline_lookahead,pipeline_d2d_batch);
                auto model=load(path,false,true);auto ctx=context(model.get(),4096,16);
                tests.push_back(compare(label+"/file",evaluate(ctx.get(),4079),baseline));
                // Change both KV and the cached expert working set before unload.
                evaluate(ctx.get(),97);evaluate(ctx.get(),31);
                const auto s=strata_mm27_snapshot();
                tests.push_back({{"name",label+"/cache_audit"},{"pass",!s.rejected_cpu_nodes && !s.rejected_full_copies &&
                    !s.pressure_rejections && (!variant || (s.cache_hit_bytes && s.cache_resident && s.arena_reserved<=(256u<<20)))}});
            }
            strata_mm27_release();
            {
                auto &r=strata_expert_file::registry();std::lock_guard<std::mutex> lock(r.mutex);
                tests.push_back({{"name",label+"/released"},{"pass",r.files.empty() && !strata_mm27_snapshot().arena_reserved && !strata_mm27_snapshot().host_working_set_limit && !strata_mm27_snapshot().pipeline_device_bytes && !strata_mm27_snapshot().pipeline_queued_bytes}});
            }
            Poison poison(round);
            {
                strata_mm27_mode(1);auto model=load(path,false,true);auto ctx=context(model.get(),4096,16);
                const auto native=evaluate(ctx.get(),4079);save(dir/("native-"+std::to_string(round)+".f32"),native.continuation);
                tests.push_back(compare(label+"/native_after_cache",native,baseline));
                strata_mm27_mode(2);const auto replay=evaluate(ctx.get(),4079);
                tests.push_back(compare(label+"/same_context_file",replay,native));
                tests.push_back(compare(label+"/replay_vs_initial",replay,baseline));
            }
            strata_mm27_release();
        }
        require(tests.size()==55,"incomplete reload coverage");
        size_t failed=0;for(const auto &t:tests)if(!t.at("pass").get<bool>())++failed;
        report["cases"]=tests.size();report["failures"]=failed;report["pass"]=failed==0;
    } catch(const std::exception &e) {report["pass"]=false;report["error"]=e.what();std::cerr<<e.what()<<'\n';}
    report["tests"]=tests;
    if(created)std::ofstream(dir/"reload-report.json")<<report.dump(2)<<'\n';
    std::cout<<json({{"pass",report.value("pass",false)},{"cases",tests.size()}}).dump()<<'\n';
    return report.value("pass",false)?0:1;
}
