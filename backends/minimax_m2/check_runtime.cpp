#include "runtime.hpp"
#include "synthetic_minimax_m2.hpp"
#include "../common/expert_file.hpp"
#include "nlohmann/json.hpp"
#include <cstring>
#include <fstream>
#include <iostream>
using namespace minimax_m2;
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
    if(argc==2 && std::string(argv[1])=="--version") {std::cout<<"MiniMax synchronous runtime fixture\n";return 0;}
    try {
        require(argc==2,"usage: runtime-check <fresh-output-dir>");
        const std::filesystem::path dir=argv[1];require(!std::filesystem::exists(dir),"output directory already exists");
        std::filesystem::create_directories(dir);environment();ggml_backend_load_all();json tests=json::array(),runs=json::array();
        bool stops=is_stop(200020);
        for(int token:{11,200004,200005,200034,200021})stops&=!is_stop(token);
        tests.push_back({{"name","exported_EOS_PAD_only_no_BOS_FIM_fallback_stops"},{"pass",stops}});
        for(bool gpu:{false,true}) {
            bool rejected=false;
            try {strata_mm27_memory(gpu?UINT64_MAX:0,gpu?0:UINT64_MAX);} catch(const std::exception &) {rejected=true;}
            tests.push_back({{"name",gpu?"VRAM_reservation_over_budget_rejected":"RAM_reservation_over_budget_rejected"},{"pass",rejected}});
        }
        for(bool mixed:{false,true}) {
            const auto name=mixed?"mixed":"f32";const auto path=(dir/(std::string(name)+".gguf")).string();
            write_synthetic_minimax_m2(path,mixed);Floats resident,native;
            const auto truncated=dir/(std::string(name)+"-truncated.gguf");
            std::filesystem::copy_file(path,truncated);
            std::filesystem::resize_file(truncated,std::filesystem::file_size(truncated)-32);
            bool bad_range=false;try {inspect(truncated.string(),true);} catch(const std::exception &) {bad_range=true;}
            tests.push_back({{"name",std::string(name)+"/truncated_payload_rejected_before_load"},{"pass",bad_range}});
            bool rejected=false;try {inspect(path);} catch(const std::exception &) {rejected=true;}
            tests.push_back({{"name",std::string(name)+"/fixture_not_production"},{"pass",rejected}});
            for(int mode:{0,1,2}) {
                strata_mm27_mode(mode?mode:1);auto model=load(path,mode==0,true);auto ctx=context(model.get(),512,8);
                clear(ctx.get());strata_mm27_reset();Observer observer;
                if(mode==2)strata_mm27_observe(Observer::check,&observer);
                const auto values=run(ctx.get(),0,65,8);
                if(!mode)resident=values;
                else tests.push_back(compare(std::string(name)+"/resident_vs_"+std::to_string(mode),values,resident));
                if(mode==1)native=values;
                if(mode==2)tests.push_back(compare(std::string(name)+"/native_vs_pinned",values,native,true));
                const auto s=strata_mm27_snapshot();
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
                    clear(ctx.get());std::atomic<bool> cancel{false};observer.cancel=&cancel;strata_mm27_cancel(&cancel);
                    bool aborted=false;try {run(ctx.get(),0,9,8,3);} catch(const std::exception &) {aborted=true;}
                    tests.push_back({{"name",std::string(name)+"/cancel_during_copy"},{"pass",aborted && cancel.load()}});
                    strata_mm27_cancel(nullptr);observer.cancel=nullptr;clear(ctx.get());
                    tests.push_back(compare(std::string(name)+"/cancel_then_fresh",run(ctx.get(),0,9,8,3),fresh,true));
                    strata_mm27_observe(nullptr,nullptr);
                }
                // Both sides of the ubatch boundary and a serial decode must use
                // the same selected experts after clear, independent of old scratch.
                for(int batch:{1,16}) {
                    clear(ctx.get());auto alternate=context(model.get(),512,batch);
                    const auto short_values=run(alternate.get(),0,17,batch);
                    tests.push_back(compare(std::string(name)+"/batch_"+std::to_string(batch)+"/mode_"+std::to_string(mode),
                        short_values,Floats(resident.begin(),resident.begin()+17*64)));
                }
                clear(ctx.get());ctx.reset();model.reset();strata_mm27_release();
                {
                    auto &registry=strata_expert_file::registry();std::lock_guard<std::mutex> lock(registry.mutex);
                    tests.push_back({{"name",std::string(name)+"/mapping_release/"+std::to_string(mode)},{"pass",registry.files.empty()}});
                }
            }
            // A new mapping after all contexts/models are gone must be registered independently.
            strata_mm27_mode(2);auto model=load(path,false,true);auto ctx=context(model.get());
            tests.push_back(compare(std::string(name)+"/unload_reload",run(ctx.get(),0,8,8),Floats(native.begin(),native.begin()+8*64),true));
            // Lose the registered file source after a successful request: fail
            // closed without falling back to touching arbitrary mmap pointers.
            const auto found=std::find_if(model->tensors_by_name.begin(),model->tensors_by_name.end(),
                [](const auto &entry){return entry.first=="blk.0.ffn_gate_exps.weight";});
            require(found!=model->tensors_by_name.end(),"missing expert tensor");const auto *weight=found->second;
            const auto source=strata_expert_file::find(weight->data,ggml_nbytes(weight));
            require(bool(source),"missing source before fault injection");
            strata_expert_file::remove(reinterpret_cast<const void *>(source->base));
            clear(ctx.get());bool failed=false;
            try {run(ctx.get(),0,8,8);} catch(const std::exception &) {failed=true;}
            tests.push_back({{"name",std::string(name)+"/missing_file_source_rejected"},{"pass",failed}});
            ctx.reset();model.reset();strata_mm27_release();
            strata_mm27_mode(2);model=load(path,false,true);ctx=context(model.get());
            tests.push_back(compare(std::string(name)+"/read_error_reload_recovery",run(ctx.get(),0,8,8),Floats(native.begin(),native.begin()+8*64),true));
            ctx.reset();model.reset();strata_mm27_release();
        }
        int failures=0;for(const auto &t:tests)if(!t.at("pass").get<bool>())++failures;
        json report={{"architecture","minimax-m2"},{"source_revision",STRATA_MM27_SOURCE_SHA},
            {"patches",STRATA_MM27_PATCH_SET},{"strict_f32",true},{"flash_attention",false},{"staging_limit",16u<<20},
            {"tests",tests},{"runs",runs},{"cases",tests.size()},{"failures",failures},{"pass",failures==0}};
        std::ofstream(dir/"runtime-report.json")<<report.dump(2)<<'\n';
        std::cout<<json({{"cases",tests.size()},{"failures",failures},{"pass",failures==0}}).dump()<<'\n';
        return failures?1:0;
    } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';strata_mm27_release();return 2;}
}
