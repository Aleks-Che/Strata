#include "runtime.hpp"
#include "synthetic_mimo2.hpp"
#include "nlohmann/json.hpp"
#include <cstring>
#include <fstream>
#include <iostream>
using namespace mimo2;
using json=nlohmann::ordered_json;
using Floats=std::vector<float>;
static json compare(const std::string &name,const Floats &a,const Floats &b,bool exact=false) {
    require(a.size()==b.size() && !a.empty(),"comparison size");
    double err=0,energy=0,max_abs=0;bool finite=true;size_t worst=0,first=a.size(),different=0;
    for(size_t i=0;i<a.size();++i) {
        finite&=std::isfinite(a[i]) && std::isfinite(b[i]);double diff=double(a[i])-b[i];
        err+=diff*diff;energy+=double(b[i])*b[i];if(std::abs(diff)>max_abs) {max_abs=std::abs(diff);worst=i;}
        if(a[i]!=b[i]) {first=std::min(first,i);++different;}
    }
    const bool bits=std::memcmp(a.data(),b.data(),a.size()*sizeof(float))==0;
    const double nmse=err/std::max(energy,1e-30);
    return {{"name",name},{"pass",finite && max_abs<=5e-4 && nmse<=1e-7 && (!exact || bits)},
        {"elements",a.size()},{"max_abs",max_abs},{"nmse",nmse},{"bit_exact",bits},{"exact_required",exact},
        {"first_different",first},{"different_values",different},{"worst_index",worst},{"actual_worst",a[worst]},{"expected_worst",b[worst]}};
}
struct Observer {
    size_t calls=0,bytes=0;std::atomic<bool> *cancel=nullptr;
    static void check(ggml_backend_t,const ggml_tensor *dst,const ggml_tensor *src,size_t offset,size_t count,void *owner) {
        auto &o=*static_cast<Observer *>(owner);std::vector<uint8_t> data(count);
        ggml_backend_tensor_get(dst,data.data(),offset,count);
        require(std::memcmp(data.data(),static_cast<const char *>(src->data)+offset,count)==0,"selected copy byte mismatch");
        ++o.calls;o.bytes+=count;if(o.cancel)o.cancel->store(true);
    }
};
static Floats run(llama_context *ctx,int begin,int count,int chunk,int salt=0) {
    Floats out;std::vector<llama_token> tokens;
    for(int p=begin;p<begin+count;++p)tokens.push_back((p*7+11+salt)%64);
    for(int i=0;i<count;i+=chunk) {
        const int n=std::min(chunk,count-i);decode(ctx,tokens,i,n,begin+i,true);
        for(int j=0;j<n;++j) {auto *l=llama_get_logits_ith(ctx,j);require(l,"missing fixture logits");out.insert(out.end(),l,l+64);}
    }
    return out;
}
int main(int argc,char **argv) {
    if(argc==2 && std::string(argv[1])=="--version") {std::cout<<"MiMo synchronous runtime fixture\n";return 0;}
    try {
        require(argc==2,"usage: runtime-check <fresh-output-dir>");
        const std::filesystem::path dir=argv[1];require(!std::filesystem::exists(dir),"output directory already exists");
        std::filesystem::create_directories(dir);environment();ggml_backend_load_all();json tests=json::array(),runs=json::array();
        bool stops=is_stop(151645);
        for(int token:{11,128247,151643,151662,151663,151664})stops&=!is_stop(token);
        tests.push_back({{"name","exported_EOS_only_no_BOS_PAD_FIM_fallback_stops"},{"pass",stops}});
        for(bool gpu:{false,true}) {
            bool rejected=false;
            try {strata_mimo_memory(gpu?UINT64_MAX:0,gpu?0:UINT64_MAX);} catch(const std::exception &) {rejected=true;}
            tests.push_back({{"name",gpu?"VRAM_reservation_over_budget_rejected":"RAM_reservation_over_budget_rejected"},{"pass",rejected}});
        }
        for(bool mixed:{false,true}) {
            const auto name=mixed?"mixed":"f32";const auto path=(dir/(std::string(name)+".gguf")).string();
            write_synthetic_mimo2(path,mixed);Floats resident,native;
            const auto truncated=dir/(std::string(name)+"-truncated.gguf");
            std::filesystem::copy_file(path,truncated);
            std::filesystem::resize_file(truncated,std::filesystem::file_size(truncated)-32);
            bool bad_range=false;try {inspect(truncated.string(),true);} catch(const std::exception &) {bad_range=true;}
            tests.push_back({{"name",std::string(name)+"/truncated_payload_rejected_before_load"},{"pass",bad_range}});
            bool rejected=false;try {inspect(path);} catch(const std::exception &) {rejected=true;}
            tests.push_back({{"name",std::string(name)+"/fixture_not_production"},{"pass",rejected}});
            for(int mode:{0,1,2}) {
                strata_mimo_mode(mode?mode:1);auto model=load(path,mode==0,true);auto ctx=context(model.get(),512,8);
                clear(ctx.get());strata_mimo_reset();Observer observer;
                if(mode==2)strata_mimo_observe(Observer::check,&observer);
                const auto values=run(ctx.get(),0,273,8);
                if(!mode)resident=values;
                else tests.push_back(compare(std::string(name)+"/resident_vs_"+std::to_string(mode),values,resident));
                if(mode==1)native=values;
                if(mode==2)tests.push_back(compare(std::string(name)+"/native_vs_pinned",values,native,true));
                const auto s=strata_mimo_snapshot();
                tests.push_back({{"name",std::string(name)+"/gpu_audit/"+std::to_string(mode)},
                    {"pass",s.gpu_nodes>0 && s.expert_nodes>0 && !s.rejected_cpu_nodes && !s.rejected_full_copies &&
                        (mode==0?s.h2d_bytes==0:s.h2d_bytes>0)}});
                runs.push_back({{"fixture",name},{"mode",mode},{"gpu_nodes",s.gpu_nodes},{"expert_nodes",s.expert_nodes},
                    {"h2d_bytes",s.h2d_bytes},{"source_bytes",s.source_bytes},{"staging_bytes",s.staging_bytes},
                    {"byte_checked",observer.bytes},{"chunks",s.chunks}});
                if(mode==2) {
                    tests.push_back({{"name",std::string(name)+"/bounded_pinned_bytes"},
                        {"pass",observer.bytes==s.h2d_bytes && s.source_bytes==s.h2d_bytes && s.staging_bytes==(16u<<20)}});
                    clear(ctx.get());const auto fresh=run(ctx.get(),0,9,8,3);
                    clear(ctx.get());std::atomic<bool> cancel{false};observer.cancel=&cancel;strata_mimo_cancel(&cancel);
                    bool aborted=false;try {run(ctx.get(),0,9,8,3);} catch(const std::exception &) {aborted=true;}
                    tests.push_back({{"name",std::string(name)+"/cancel_during_copy"},{"pass",aborted && cancel.load()}});
                    strata_mimo_cancel(nullptr);observer.cancel=nullptr;clear(ctx.get());
                    tests.push_back(compare(std::string(name)+"/cancel_then_fresh",run(ctx.get(),0,9,8,3),fresh,true));
                    strata_mimo_observe(nullptr,nullptr);
                }
                clear(ctx.get());strata_mimo_release();
            }
            // A new mapping after all contexts/models are gone must be registered independently.
            strata_mimo_mode(2);auto model=load(path,false,true);auto ctx=context(model.get());
            tests.push_back(compare(std::string(name)+"/unload_reload",run(ctx.get(),0,8,8),Floats(native.begin(),native.begin()+8*64),true));
            strata_mimo_release();
            // Reload with independent cache identities; compare exactly to the uncached graph.
            for(bool mmap:{false,true}) for(size_t cap:{size_t(8)<<20,size_t(256)<<20}) {
                auto cached_model=load(path,false,true);auto cached_ctx=context(cached_model.get(),512,8);
                strata_mimo_mode(2);strata_mimo_cache(cap);strata_mimo_cache_prefill(true);strata_mimo_phase(false);
                strata_mimo_reader(mmap);
                Observer audit;strata_mimo_observe(Observer::check,&audit);strata_mimo_reset();
                const auto values=run(cached_ctx.get(),0,273,8);const auto stats=strata_mimo_snapshot();
                const std::string label=std::string(name)+(mmap?"/mmap_cache_":"/file_cache_")+std::to_string(cap>>20);
                tests.push_back(compare(label+"/logits",values,native,true));
                tests.push_back({{"name",label+"/accounting"},{"pass",stats.cache_bytes<=cap && stats.cache_bytes<=stats.cache_limit &&
                    stats.cache_hit_bytes+stats.h2d_bytes==stats.requested_bytes && (mmap?stats.source_bytes==0:stats.source_bytes==stats.h2d_bytes) &&
                    audit.bytes==stats.requested_bytes && !stats.rejected_cpu_nodes && !stats.rejected_full_copies}});
                tests.push_back({{"name",label+"/hits_or_eviction"},{"pass",cap==(256u<<20)?stats.cache_hits>0 && stats.cache_hit_bytes>0:stats.cache_evictions>0 && stats.cache_reuses>0}});
                runs.push_back({{"fixture",name},{"cache_cap",cap},{"cache_bytes",stats.cache_bytes},{"cache_hits",stats.cache_hits},
                    {"cache_evictions",stats.cache_evictions},{"cache_reuses",stats.cache_reuses},{"requested_bytes",stats.requested_bytes},
                    {"h2d_bytes",stats.h2d_bytes},{"cache_hit_bytes",stats.cache_hit_bytes},{"byte_checked",audit.bytes}});
                clear(cached_ctx.get());const auto fresh=run(cached_ctx.get(),0,9,8,3);
                clear(cached_ctx.get());std::atomic<bool> cancelled{false};audit.cancel=&cancelled;strata_mimo_cancel(&cancelled);
                bool stopped=false;try {run(cached_ctx.get(),0,9,8,3);} catch(const std::exception &) {stopped=true;}
                tests.push_back({{"name",label+"/cancel"},{"pass",stopped && cancelled.load()}});
                strata_mimo_cancel(nullptr);audit.cancel=nullptr;clear(cached_ctx.get());
                tests.push_back(compare(label+"/cancel_recovery",run(cached_ctx.get(),0,9,8,3),fresh,true));
                strata_mimo_cache(cap);strata_mimo_cache_prefill(false);strata_mimo_phase(true);strata_mimo_reset();clear(cached_ctx.get());
                const auto no_admission=run(cached_ctx.get(),0,9,8,3);
                tests.push_back(compare(label+"/prefill_no_admission_logits",no_admission,fresh,true));
                tests.push_back({{"name",label+"/prefill_no_admission_bytes"},{"pass",strata_mimo_snapshot().cache_bytes==0 && strata_mimo_snapshot().cache_fill_bytes==0}});
                strata_mimo_phase(false);strata_mimo_cache_prefill(true);strata_mimo_release();
            }
            for(bool mmap:{false,true})for(int readers:{1,2})for(size_t cap:{size_t(8)<<20,size_t(256)<<20}) {
                auto model=load(path,false,true);auto ctx=context(model.get(),512,8);
                strata_mimo_cache(cap);strata_mimo_reader(mmap);strata_mimo_cache_prefill(true);strata_mimo_phase(false);
                const int chunk=readers==1?4:16;
                strata_mimo_pipeline_config(readers,chunk,2);strata_mimo_reset();
                Observer audit;strata_mimo_observe(Observer::check,&audit);
                const auto values=run(ctx.get(),0,273,8);const auto s=strata_mimo_snapshot();
                const std::string label=std::string(name)+(mmap?"/mmap_pipe_":"/file_pipe_")+std::to_string(readers)+"_"+std::to_string(cap>>20);
                strata_mimo_trace_write((dir/(label.substr(0,label.find('/'))+"-"+label.substr(label.find('/')+1)+".json")).string().c_str());
                tests.push_back(compare(label+"/logits",values,native,true));
                tests.push_back({{"name",label+"/bytes_budget_drain"},{"pass",s.requested_bytes==s.h2d_bytes+s.cache_hit_bytes &&
                    audit.bytes==s.requested_bytes && s.pipeline_delivered_bytes>0 && s.pipeline_groups>0 && s.pipeline_chunks>0 &&
                    s.pipeline_device_bytes==size_t(4*chunk)*(1u<<20) && !s.pipeline_reader_owned && !s.pipeline_queued &&
                    s.cache_bytes<=s.cache_limit && s.cache_limit<=cap && !s.rejected_cpu_nodes && !s.rejected_full_copies}});
                tests.push_back({{"name",label+"/hits_evictions"},{"pass",cap==(256u<<20)?s.cache_hits>0:s.cache_evictions>0}});
                clear(ctx.get());const auto fresh=run(ctx.get(),0,9,8,3);
                clear(ctx.get());std::atomic<bool> cancel{false};audit.cancel=&cancel;strata_mimo_cancel(&cancel);
                bool aborted=false;try {run(ctx.get(),0,9,8,3);}catch(const std::exception &) {aborted=true;}
                const auto cancelled=strata_mimo_snapshot();
                tests.push_back({{"name",label+"/cancel_drains"},{"pass",aborted && cancel.load() && !cancelled.pipeline_reader_owned && !cancelled.pipeline_queued}});
                strata_mimo_cancel(nullptr);audit.cancel=nullptr;clear(ctx.get());
                tests.push_back(compare(label+"/cancel_recovery",run(ctx.get(),0,9,8,3),fresh,true));
                strata_mimo_pipeline_config(0);strata_mimo_cache(cap);strata_mimo_pipeline_config(readers,chunk);
                strata_mimo_test_pipeline_failure(1);clear(ctx.get());
                bool failed=false;try {run(ctx.get(),0,9,8,3);}catch(const std::exception &) {failed=true;}
                const auto failure=strata_mimo_snapshot();
                tests.push_back({{"name",label+"/reader_error_drains"},{"pass",failed && !failure.pipeline_reader_owned && !failure.pipeline_queued}});
                clear(ctx.get());tests.push_back(compare(label+"/reader_error_recovery",run(ctx.get(),0,9,8,3),fresh,true));
                strata_mimo_pipeline_config(0);strata_mimo_cache(cap);strata_mimo_pipeline_config(readers,chunk);
                strata_mimo_cache_prefill(false);strata_mimo_phase(true);strata_mimo_reset();clear(ctx.get());
                auto no_admission=compare(label+"/prefill_no_admission",run(ctx.get(),0,9,8,3),fresh,true);
                no_admission["pass"]=no_admission["pass"].get<bool>() && !strata_mimo_snapshot().cache_bytes && !strata_mimo_snapshot().cache_fill_bytes;
                tests.push_back(no_admission);strata_mimo_release();
            }
        }
        {
            auto *cpu=ggml_backend_dev_by_name("CPU");ggml_backend_dev_t devices[]={cpu,nullptr};
            auto mp=llama_model_default_params();mp.devices=devices;mp.n_gpu_layers=0;
            mp.load_mtp=false;mp.use_extra_bufts=false;mp.load_mode=LLAMA_LOAD_MODE_NONE;
            Model model(llama_model_load_from_file((dir/"mixed.gguf").string().c_str(),mp),llama_model_free);
            require(bool(model),"CPU rejection fixture load failed");
            auto ctx=context(model.get());strata_mimo_reset();strata_mimo_mode(1);
            bool rejected=false;try {run(ctx.get(),0,1,1);} catch(const std::exception &) {rejected=true;}
            tests.push_back({{"name","CPU_compute_rejected_before_execution"},
                {"pass",rejected && strata_mimo_snapshot().rejected_cpu_nodes>0}});
            strata_mimo_release();
        }
        size_t passed=0;for(const auto &t:tests)passed+=t["pass"].get<bool>();
        require(tests.size()==212,"incomplete runtime coverage");
        json report={{"status",passed==tests.size()?"pass":"fail"},{"scope","synthetic native full/SWA GPU graph; no full-model inference"},
            {"requested_revision",STRATA_MIMO_SOURCE_SHA},{"archive_sha256",STRATA_MIMO_ARCHIVE_SHA256},{"patch_set",STRATA_MIMO_PATCH_SET},
            {"passed",passed},{"case_count",tests.size()},{"positions",273},{"runs",runs},{"results",tests}};
        std::ofstream file(dir/"runtime-report.json");file<<report.dump(2)<<'\n';require(bool(file),"report write failed");
        std::cout<<report.dump()<<'\n';llama_backend_free();return passed==tests.size()?0:2;
    } catch(const std::exception &e) {strata_mimo_release();std::cerr<<e.what()<<'\n';return 1;}
}
