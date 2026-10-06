#include "runtime.hpp"
#include "synthetic_hy3.hpp"
#include "sync_test.h"
#include "../common/expert_file.hpp"
#include "llama-context.h"
#include "nlohmann/json.hpp"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <chrono>
using json = nlohmann::ordered_json;
using namespace hy3;
using Floats = std::vector<float>;
static json stats() {
    const auto s = strata_hy3_sync_snapshot();
    return {{"source_bytes",s.source_bytes},{"h2d_bytes",s.h2d_bytes},{"ranges",s.ranges},
        {"chunks",s.chunks},{"staging_bytes",s.staging_bytes},{"compute_calls",s.compute_calls},
        {"gpu_nodes",s.gpu_nodes},{"expert_nodes",s.expert_nodes},{"rejected_cpu_nodes",s.rejected_cpu_nodes},
        {"rejected_full_copies",s.rejected_full_copies},{"source_ms",s.source_ms},{"h2d_ms",s.h2d_ms}};
}
struct Observer {
    uint64_t bytes = 0, ranges = 0;
    static void copy(void * owner, ggml_backend *, ggml_tensor * dst, const ggml_tensor * src, size_t offset, size_t size) {
        auto & out = *static_cast<Observer *>(owner);
        std::vector<uint8_t> bytes(size);
        ggml_backend_tensor_get(dst, bytes.data(), offset, size);
        require(std::memcmp(bytes.data(), static_cast<const uint8_t *>(src->data)+offset, size) == 0, "expert GPU bytes differ from mmap");
        ++out.ranges; out.bytes += size;
    }
};
static Floats run(llama_context * ctx, int prompt, int batch, strata_hy3_sync_stats * prefill=nullptr) {
    clear(ctx);
    std::vector<llama_token> tokens(prompt);
    for (int i = 0; i < prompt; ++i) tokens[i] = (i*7+11)%64;
    Floats logits;
    for (int i = 0; i < prompt; i += batch) {
        const int count = std::min(batch,prompt-i);
        decode(ctx, tokens, i, count, i, true);
        for (int row = 0; row < count; ++row) {
            auto * p = llama_get_logits_ith(ctx, row); logits.insert(logits.end(),p,p+64);
        }

    }
    if (prefill) *prefill=strata_hy3_sync_snapshot();
    for (int i = 0; i < 8; ++i) {
        decode(ctx, std::vector<llama_token>{(prompt+i)%64}, 0, 1, prompt+i);
        auto * p = llama_get_logits_ith(ctx, -1); logits.insert(logits.end(),p,p+64);
    }
    for (float value : logits) require(std::isfinite(value), "non-finite logits");
    return logits;
}
static json compare(const std::string & name, const Floats & a, const Floats & b) {
    require(a.size() == b.size() && !a.empty(), "logit size mismatch");
    double max_abs = 0;
    for (size_t i = 0; i < a.size(); ++i) max_abs = std::max(max_abs,std::abs(double(a[i])-b[i]));
    const bool exact = std::memcmp(a.data(),b.data(),a.size()*sizeof(float)) == 0;
    return {{"name",name},{"pass",exact},{"elements",a.size()},{"max_abs",max_abs},{"bit_exact_required",true}};
}
struct InterruptedCopy {
    std::shared_ptr<strata_expert_file::Source> source;
    HANDLE original=INVALID_HANDLE_VALUE;
    std::atomic<bool> cancel{false};
    bool read_error;
    size_t uploaded=0;
    explicit InterruptedCopy(bool fail_read):read_error(fail_read) {}
    void restore() {
        if(source) {source->file=original;source.reset();}
        strata_hy3_sync_cancel(nullptr);strata_hy3_sync_observer(nullptr,nullptr);
    }
    ~InterruptedCopy() {restore();}
    static void copy(void * owner,ggml_backend *,ggml_tensor *,const ggml_tensor * src,size_t,size_t bytes) {
        auto & test=*static_cast<InterruptedCopy *>(owner);
        test.uploaded+=bytes;
        if(test.read_error && !test.source) {
            // Trigger a real Win32 ReadFile failure on the next expert range.
            // Keep the original handle alive and restore it before recovery.
            test.source=strata_expert_file::find(src->data,ggml_nbytes(src));
            require(bool(test.source),"missing source for read failure fixture");
            test.original=test.source->file;test.source->file=INVALID_HANDLE_VALUE;
        } else if(!test.read_error) test.cancel.store(true);
    }
};
static void failures(json & cases,const std::string & path) {
    std::cerr<<"HY3_FAILURE_CHECK reference\n";
    Floats reference;
    {
        auto model=load(path,false,false,true);auto ctx=context(model.get());
        strata_hy3_sync_mode(2);reference=run(ctx.get(),19,17);
    }
    strata_hy3_sync_release(); // Next request must allocate a new staging buffer.
    std::cerr<<"HY3_FAILURE_CHECK pinned allocation\n";
    {
        auto model=load(path,false,false,true);auto ctx=context(model.get());
        strata_hy3_sync_mode(2);strata_hy3_sync_reset();
        strata_hy3_test_limits limits;limits.fail_next_pinned_allocation=true;
        strata_hy3_test_set_limits(limits);
        bool rejected=false;try {run(ctx.get(),19,17);}catch(const std::runtime_error &) {rejected=true;}
        const auto s=strata_hy3_sync_snapshot();
        cases.push_back({{"name","injected_pinned_oom_before_upload"},
            {"pass",rejected && s.h2d_bytes==0 && s.staging_bytes==0}});
        strata_hy3_test_set_limits({});
        cases.push_back(compare("recovery_after_pinned_oom",run(ctx.get(),19,17),reference));
        for(bool fail_read:{true,false}) {
            std::cerr<<"HY3_FAILURE_CHECK partial copy "<<fail_read<<'\n';
            strata_hy3_sync_reset();InterruptedCopy test(fail_read);
            strata_hy3_sync_observer(InterruptedCopy::copy,&test);
            if(!fail_read) strata_hy3_sync_cancel(&test.cancel);
            const auto start=std::chrono::steady_clock::now();
            rejected=false;try {run(ctx.get(),19,17);}catch(const std::runtime_error &) {rejected=true;}
            const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            const auto partial=strata_hy3_sync_snapshot();
            test.restore();
            const std::string name=fail_read ? "read_error" : "cancel";
            cases.push_back({{"name",name+"_after_partial_upload"},
                {"pass",rejected && test.uploaded>0 && partial.h2d_bytes==test.uploaded && elapsed<5},
                {"completed_upload_bytes",test.uploaded},{"seconds_until_failure",elapsed}});
            cases.push_back(compare("recovery_after_partial_"+name,run(ctx.get(),19,17),reference));
        }
        for(bool ram:{true,false}) {
            std::cerr<<"HY3_FAILURE_CHECK pressure "<<ram<<'\n';
            strata_hy3_sync_reset();strata_hy3_test_limits low;
            if(ram) low.ram_available=0;else low.gpu_available=0;
            strata_hy3_test_set_limits(low);
            rejected=false;std::string error;
            try {run(ctx.get(),19,17);}catch(const std::runtime_error & e) {rejected=true;error=e.what();}
            strata_hy3_test_set_limits({});
            const auto s=strata_hy3_sync_snapshot();
            const std::string name=ram ? "ram" : "vram";
            cases.push_back({{"name","simulated_"+name+"_pressure_before_compute"},
                {"pass",rejected && s.compute_calls==0 && s.h2d_bytes==0 && error.find("95% limit")!=std::string::npos}});
            cases.push_back(compare("recovery_after_"+name+"_pressure",run(ctx.get(),19,17),reference));
        }
    }
    bool rejected=false;
    std::cerr<<"HY3_FAILURE_CHECK unload\n";
    try {strata_hy3_memory_check(std::numeric_limits<size_t>::max());}
    catch(const std::runtime_error &) {rejected=true;}
    cases.push_back({{"name","reject_impossible_gpu_reserve_without_allocation"},{"pass",rejected}});
    strata_hy3_sync_release();
    {
        auto & r=strata_expert_file::registry();std::lock_guard lock(r.mutex);
        cases.push_back({{"name","unload_releases_registered_sources"},{"pass",r.files.empty()}});
    }
    {
        auto model=load(path,false,false,true);auto ctx=context(model.get());
        strata_hy3_sync_mode(2);
        cases.push_back(compare("reload_after_failures",run(ctx.get(),19,17),reference));
    }
}
int main(int argc,char ** argv) {
    json report={{"status","error"},{"scope","synthetic resident/native-selected/pinned-file byte and logit parity"},
        {"source_sha",STRATA_HY3_SOURCE_SHA},{"patch_set",STRATA_HY3_PATCH_SET},{"cuda_fusion",false},{"cases",json::array()}};
    std::filesystem::path directory;bool created=false;
    try {
        require(argc==2,"usage: strata-hy3-runtime-check NEW_DIRECTORY");directory=argv[1];
        require(!std::filesystem::exists(directory),"directory must be new");
        require(std::filesystem::create_directories(directory),"cannot create output directory");created=true;
        environment();ggml_backend_load_all();
        for(int variant:{0,1,2}) {
            const bool quant=variant==1,wide=variant==2;
            const auto path=(directory/(wide ? "hy3-wide.gguf" : quant ? "hy3-mixed.gguf" : "hy3-f32.gguf")).string();
            write_synthetic_hy3(path,quant,wide);
            auto resident=load(path,true,false,true),mapped=load(path,false,false,true);
            for(int batch:{1,4,17}) {
                if(wide && batch!=1) continue;
                auto a=context(resident.get(),2048,batch),b=context(mapped.get(),2048,batch);
                strata_hy3_sync_mode(1);strata_hy3_sync_reset();
                const auto reference=run(a.get(),19,batch),candidate=run(b.get(),19,batch);
                Observer observer;strata_hy3_sync_observer(Observer::copy,&observer);
                strata_hy3_sync_mode(2);strata_hy3_sync_reset();
                const auto actual=run(b.get(),19,batch);
                const std::string name=std::string(wide ? "wide" : quant ? "mixed" : "f32")+"/batch="+std::to_string(batch);
                report["cases"].push_back(compare(name+"/native_vs_resident",candidate,reference));
                auto c=compare(name+"/pinned_file_vs_native",actual,candidate);c["transport"]=stats();c["verified_gpu_bytes"]=observer.bytes;
                const auto s=strata_hy3_sync_snapshot();
                require(observer.bytes==s.h2d_bytes && observer.ranges==s.ranges && observer.bytes>0,"observer byte coverage mismatch");
                require(s.source_bytes==s.h2d_bytes && s.staging_bytes==16*1024*1024 && !s.rejected_cpu_nodes && !s.rejected_full_copies,"staging budget/audit failed");
                if(wide) require(s.chunks>s.ranges,"fixture must cross staging boundary");
                report["cases"].push_back(c);strata_hy3_sync_observer(nullptr,nullptr);
            }
        }
        const auto path=(directory/"hy3-mixed.gguf").string();
        {
            auto model=load(path,false,false,true);auto ctx=context(model.get());
            strata_hy3_sync_mode(2);auto reference=run(ctx.get(),19,17);
            std::atomic<bool> cancel{true};strata_hy3_sync_cancel(&cancel);
            bool rejected=false;try {run(ctx.get(),19,17);}catch(const std::runtime_error &){rejected=true;}
            strata_hy3_sync_cancel(nullptr);
            report["cases"].push_back({{"name","cancel_before_compute"},{"pass",rejected}});
            report["cases"].push_back(compare("recovery_after_cancel",run(ctx.get(),19,17),reference));
        }
        {
            auto model=load(path,false,true,true);auto ctx=context(model.get());
            strata_hy3_sync_mode(2);strata_hy3_sync_reset();bool rejected=false;
            try {run(ctx.get(),19,17);}catch(const std::runtime_error &){rejected=true;}
            const auto s=strata_hy3_sync_snapshot();
            report["cases"].push_back({{"name","reject_cpu_tensor_math_before_copy"},
                {"pass",rejected && s.rejected_cpu_nodes>0 && s.h2d_bytes==0}});
        }
        bool rejected=false;try {inspect(path,false);}catch(const std::runtime_error &){rejected=true;}
        report["cases"].push_back({{"name","fixture_requires_explicit_flag"},{"pass",rejected}});
        const auto truncated=directory/"truncated.gguf";
        std::filesystem::copy_file(path,truncated);std::filesystem::resize_file(truncated,std::filesystem::file_size(truncated)-32);
        rejected=false;try {inspect(truncated.string(),true);}catch(const std::runtime_error &){rejected=true;}
        report["cases"].push_back({{"name","reject_truncated_before_weight_allocation"},{"pass",rejected}});
        failures(report["cases"],path);
        bool pass=true;for(const auto & c:report["cases"]) pass&=c["pass"].get<bool>();
        report["status"]=pass ? "pass" : "fail";
    } catch(const std::exception & e) {report["error"]=e.what();}
    strata_hy3_sync_release();ggml_quantize_free();llama_backend_free();
    report["case_count"]=report["cases"].size();
    if(created) {std::ofstream out(directory/"runtime-report.json");out<<report.dump(2)<<'\n';if(!out) return 1;}
    std::cout<<report.dump(2)<<'\n';return report["status"]=="pass" ? 0 : 1;
}
