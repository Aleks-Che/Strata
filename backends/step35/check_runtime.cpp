#include "runtime.hpp"
#include "cache_registry.hpp"
#include "synthetic_step.hpp"
#include "llama-context.h"
#include "nlohmann/json.hpp"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
using json = nlohmann::ordered_json;
using namespace step35;
using Floats = std::vector<float>;
static json stats() {
    const auto s = strata_step_sync_snapshot();
    return {{"source_bytes",s.source_bytes},{"h2d_bytes",s.h2d_bytes},{"ranges",s.ranges},
        {"chunks",s.chunks},{"staging_bytes",s.staging_bytes},{"compute_calls",s.compute_calls},
        {"gpu_nodes",s.gpu_nodes},{"expert_nodes",s.expert_nodes},{"rejected_cpu_nodes",s.rejected_cpu_nodes},
        {"rejected_full_copies",s.rejected_full_copies},{"source_ms",s.source_ms},{"h2d_ms",s.h2d_ms}};
}
static json cache_stats() {
    const auto s=strata_step_sync_snapshot();
    return {{"hits",s.cache_hits},{"misses",s.cache_misses},{"evictions",s.cache_evictions},
        {"h2d_bytes",s.h2d_bytes},{"d2d_bytes",s.d2d_bytes},{"fill_bytes",s.cache_fill_bytes},
        {"resident_bytes",s.cache_bytes},{"limit",s.cache_limit},{"oom",s.cache_oom}};
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
static Floats run(llama_context * ctx, int prompt, int batch, strata_step_sync_stats * prefill=nullptr) {
    clear(ctx);
    RequestPhase phase(1);
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
    if (prefill) *prefill=strata_step_sync_snapshot();
    phase.decode();
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
int main(int argc, char ** argv) {
    json report = {{"status","error"},{"scope","synthetic Step resident/native-selected/pinned-selected byte and logit parity"},
        {"source_sha",STRATA_STEP_SOURCE_SHA},{"patch_set",STRATA_STEP_PATCH_SET}};
    json cases = json::array();
    std::filesystem::path directory;
    try {
        require(argc == 2, "usage: strata-step35-runtime-check WORK_DIRECTORY");
        directory = argv[1]; std::filesystem::create_directories(directory);
        environment(); ggml_backend_load_all();
        for (int variant : {0, 1, 2}) {
            const bool quant = variant == 1, wide = variant == 2;
            const auto path = (directory / (wide ? "step-wide.gguf" : quant ? "step-mixed.gguf" : "step-f32.gguf")).string();
            write_synthetic_step(path,quant,wide);
            auto resident = load(path,true), mapped = load(path);
            for (auto kv : {GGML_TYPE_F32, GGML_TYPE_F16}) for (int batch : {1,4,17}) {
                if (wide && (kv != GGML_TYPE_F32 || batch != 1)) continue;
                const int prompt = wide ? 1 : batch == 17 ? 513 : 19;
                auto a = context(resident.get(),2048,batch,kv), b = context(mapped.get(),2048,batch,kv);
                strata_step_sync_mode(1); strata_step_sync_reset();
                const auto reference = run(a.get(),prompt,batch);
                const auto candidate = run(b.get(),prompt,batch);
                Observer observer; strata_step_sync_observer(Observer::copy,&observer);
                strata_step_sync_mode(2); strata_step_sync_reset();
                const auto actual = run(b.get(),prompt,batch);
                const std::string name = std::string(wide?"wide_f32":quant?"mixed":"f32")+"/kv="+ggml_type_name(kv)+"/batch="+std::to_string(batch);
                auto c = compare(name+"/native_vs_resident",candidate,reference); cases.push_back(c);
                c = compare(name+"/pinned_vs_native",actual,candidate);
                c["transport"] = stats(); c["verified_gpu_bytes"] = observer.bytes;
                const auto s = strata_step_sync_snapshot();
                require(observer.bytes == s.h2d_bytes && observer.ranges == s.ranges && observer.bytes > 0, "copy observer did not cover all bytes");
                require(s.staging_bytes <= 16*1024*1024+4096 && !s.rejected_cpu_nodes && !s.rejected_full_copies, "transport budget/audit failed");
                if (wide) require(s.chunks > s.ranges, "wide fixture did not cross the pinned chunk boundary");
                cases.push_back(c); strata_step_sync_observer(nullptr,nullptr);
            }
        }
        for (bool quant:{false,true}) for (auto kv:{GGML_TYPE_F32,GGML_TYPE_F16}) {
            const auto path=(directory/(quant?"step-mixed.gguf":"step-f32.gguf")).string();
            auto model=load(path); auto ctx=context(model.get(),2048,17,kv);
            strata_step_sync_mode(2); strata_step_cache_begin(path.c_str(),0);
            const auto reference=run(ctx.get(),513,17);
            const std::string name=std::string(quant?"mixed":"f32")+"/kv="+ggml_type_name(kv);
            register_cache(model.get(),path,64*1024*1024);
            Observer observer;strata_step_sync_observer(Observer::copy,&observer);strata_step_sync_reset();
            auto actual=run(ctx.get(),513,17);
            auto c=compare(name+"/cache_cold",actual,reference);c["cache"]=cache_stats();cases.push_back(c);
            auto s=strata_step_sync_snapshot();
            require(s.cache_hits && s.cache_misses && s.cache_bytes<=s.cache_limit && observer.bytes==s.h2d_bytes+s.d2d_bytes,"cache cold coverage/accounting failed");
            strata_step_sync_reset();observer={};actual=run(ctx.get(),513,17);
            c=compare(name+"/cache_warm",actual,reference);c["cache"]=cache_stats();cases.push_back(c);
            s=strata_step_sync_snapshot();
            require(s.cache_hits && !s.h2d_bytes && !s.cache_misses && observer.bytes==s.d2d_bytes,"warm fixture unexpectedly read host weights");
            // Re-register the same live addresses as a new generation: no stale hits.
            register_cache(model.get(),path,1024*1024);strata_step_sync_reset();observer={};
            actual=run(ctx.get(),513,17);
            c=compare(name+"/cache_small_eviction",actual,reference);c["cache"]=cache_stats();cases.push_back(c);
            s=strata_step_sync_snapshot();
            require(s.cache_misses && s.cache_evictions && s.cache_bytes<=1024*1024 && observer.bytes==s.h2d_bytes+s.d2d_bytes,"eviction coverage failed");
            strata_step_sync_observer(nullptr,nullptr);strata_step_sync_release();
        }
        {
            const auto path=(directory/"step-mixed.gguf").string();
            auto model=load(path);auto ctx=context(model.get(),2048,17);
            strata_step_sync_mode(2);strata_step_cache_begin(path.c_str(),0);
            const auto reference=run(ctx.get(),19,17);
            strata_step_cache_begin(path.c_str(),64*1024*1024);
            bool rejected=false;
            try {run(ctx.get(),19,17);} catch(const std::runtime_error &){rejected=true;}
            cases.push_back({{"name","reject_unregistered_expert"},{"pass",rejected && !strata_step_sync_snapshot().cache_bytes}});
            register_cache(model.get(),path,64*1024*1024);
            cases.push_back(compare("recover_after_registry_error",run(ctx.get(),19,17),reference));
            strata_step_sync_release();
        }
        for (bool quant:{false,true}) {
            const auto path=(directory/(quant?"step-mixed.gguf":"step-f32.gguf")).string();
            auto model=load(path);auto ctx=context(model.get(),2048,17);
            for(int chunk:{4,8,16}) for(int readers:{1,2}) for(int cap:{0,1,64}) {
                const int prompt=chunk==8 && cap==1?513:19;
                strata_step_sync_mode(2);strata_step_cache_begin(path.c_str(),0);
                const auto reference=run(ctx.get(),prompt,17);
                register_cache(model.get(),path,size_t(cap)<<20);
                strata_step_pipeline_config(readers,chunk,0);
                Observer observer;strata_step_sync_observer(Observer::copy,&observer);strata_step_sync_reset();
                const auto actual=run(ctx.get(),prompt,17);
                const std::string name=std::string(quant?"mixed":"f32")+"/pipeline/readers="+std::to_string(readers)+"/chunk="+std::to_string(chunk)+"/cap="+std::to_string(cap);
                auto c=compare(name,actual,reference);c["cache"]=cache_stats();
                const auto s=strata_step_sync_snapshot();
                require(s.pipeline_groups && s.pipeline_chunks && s.pipeline_unused_bytes==0 &&
                    s.staging_bytes==uint64_t(chunk)*4*1024*1024 && s.pipeline_device_bytes==s.staging_bytes &&
                    observer.bytes==s.d2d_bytes && s.cache_bytes<=s.cache_limit,"pipeline accounting/bytes failed");
                c["groups"]=s.pipeline_groups;c["chunks"]=s.pipeline_chunks;c["verified_bytes"]=observer.bytes;
                cases.push_back(c);
                if(cap==64) {
                    strata_step_sync_reset();observer={};
                    c=compare(name+"/warm",run(ctx.get(),prompt,17),reference);c["cache"]=cache_stats();
                    const auto warm=strata_step_sync_snapshot();
                    require(warm.cache_hits && !warm.h2d_bytes && !warm.source_bytes && !warm.pipeline_groups &&
                        observer.bytes==warm.d2d_bytes,"all-hit pipeline should not read/upload");
                    cases.push_back(c);
                }
                strata_step_sync_observer(nullptr,nullptr);strata_step_sync_release();
            }
        }
        {
            const auto path=(directory/"step-mixed.gguf").string();
            auto model=load(path);auto ctx=context(model.get(),2048,17,GGML_TYPE_F16);
            strata_step_sync_mode(2);strata_step_cache_begin(path.c_str(),0);
            const auto reference=run(ctx.get(),513,17);
            register_cache(model.get(),path,1024*1024);strata_step_pipeline_config(2,4,0);
            Observer observer;strata_step_sync_observer(Observer::copy,&observer);strata_step_sync_reset();
            auto c=compare("mixed/kv=f16/pipeline/SWA_boundary",run(ctx.get(),513,17),reference);
            const auto s=strata_step_sync_snapshot();
            require(s.pipeline_chunks && s.cache_evictions && observer.bytes==s.d2d_bytes,"F16 pipeline coverage failed");
            c["verified_bytes"]=observer.bytes;cases.push_back(c);
            strata_step_sync_observer(nullptr,nullptr);strata_step_sync_release();
        }
        {
            const auto path=(directory/"step-mixed.gguf").string();
            auto model=load(path);auto ctx=context(model.get(),2048,17);
            strata_step_sync_mode(2);strata_step_cache_begin(path.c_str(),0);
            const auto reference=run(ctx.get(),19,17);
            strata_step_pipeline_config(2,4,0);
            bool rejected=false;
            try {run(ctx.get(),19,17);}catch(const std::runtime_error &){rejected=true;}
            cases.push_back({{"name","pipeline/reject_unregistered_router_plan"},{"pass",rejected}});
            register_cache(model.get(),path,1024*1024);strata_step_pipeline_config(2,4,0);
            cases.push_back(compare("pipeline/recover_after_registry_error",run(ctx.get(),19,17),reference));
            strata_step_sync_release();
        }
        {
            const auto path=(directory/"step-mixed.gguf").string();
            auto model=load(path);auto ctx=context(model.get(),2048,17);
            // Explicit request phase covers a one-token prompt, a one-token
            // tail after a full batch, and a prompt past the SWA boundary.
            for (int readers:{0,1,2}) for (int prompt:{1,18,513}) {
                strata_step_sync_mode(2);strata_step_cache_begin(path.c_str(),0);
                const auto reference=run(ctx.get(),prompt,17);
                register_cache(model.get(),path,64*1024*1024);
                strata_step_pipeline_config(readers,8,0);strata_step_cache_prefill(false);
                Observer observer;strata_step_sync_observer(Observer::copy,&observer);
                for (bool warm:{false,true}) {
                    strata_step_sync_reset();observer={};strata_step_sync_stats prefill{};
                    auto c=compare("prefill_off/readers="+std::to_string(readers)+"/prompt="+std::to_string(prompt)+
                        (warm?"/warm":"/cold"),run(ctx.get(),prompt,17,&prefill),reference);
                    const auto s=strata_step_sync_snapshot();
                    require(!prefill.cache_fill_bytes && !prefill.cache_evictions &&
                        prefill.prefill_admission_skips==prefill.cache_misses &&
                        s.prefill_admission_skips==prefill.prefill_admission_skips &&
                        observer.bytes==s.d2d_bytes+(readers?0:s.h2d_bytes) &&
                        strata_step_request_phase(0)==0,"prefill-off phase/byte accounting failed");
                    if (!warm) require(!prefill.cache_bytes && prefill.prefill_admission_skips &&
                        s.cache_fill_bytes && s.cache_bytes,"cold prefill filled cache or decode did not fill");
                    else require(prefill.cache_hits,"prefill-off ignored existing cache hits");
                    c["prefill_skips"]=prefill.prefill_admission_skips;c["prefill_hits"]=prefill.cache_hits;
                    c["prefill_fill_bytes"]=prefill.cache_fill_bytes;c["decode_fill_bytes"]=s.cache_fill_bytes;
                    c["verified_gpu_bytes"]=observer.bytes;cases.push_back(c);
                }
                strata_step_sync_observer(nullptr,nullptr);strata_step_sync_release();
            }
            bool caught=false;
            try {RequestPhase phase(1);throw std::runtime_error("fixture phase unwind");}
            catch(const std::runtime_error &){caught=true;}
            cases.push_back({{"name","request_phase_exception_unwind"},{"pass",caught && strata_step_request_phase(0)==0}});
        }
        // Deliberately force a model operation onto CPU. Admission must fail
        // before weights are copied or any compute split is submitted.
        auto model = load((directory/"step-f32.gguf").string(),false,true);
        auto ctx = context(model.get(),2048,1);
        strata_step_sync_mode(2); strata_step_sync_reset();
        bool rejected = false;
        try { decode(ctx.get(),std::vector<llama_token>{1},0,1,0); }
        catch (const std::runtime_error &) { rejected = true; }
        const auto s = strata_step_sync_snapshot();
        cases.push_back({{"name","reject_cpu_embedding_before_compute"},
            {"pass",rejected && s.rejected_cpu_nodes > 0 && s.compute_calls == 0 && s.h2d_bytes == 0}});
        require(cases.size() == 111,"wrong runtime case count");
        bool pass = true; for (const auto & c : cases) pass &= c["pass"].get<bool>();
        report["status"] = pass ? "pass" : "fail";
    } catch (const std::exception & error) { report["error"] = error.what(); }
    strata_step_sync_release();
    report["case_count"] = cases.size(); report["cases"] = std::move(cases);
    if (!directory.empty()) { std::ofstream out(directory/"runtime-report.json"); out << report.dump(2) << '\n'; }
    std::cout << report["status"] << ": " << report["case_count"] << " cases\n";
    return report["status"] == "pass" ? 0 : 1;
}
