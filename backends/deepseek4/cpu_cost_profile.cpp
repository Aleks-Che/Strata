#include "cpu_cost_profile.hpp"
#include "cpu_mmvq_fast.hpp"
#include "cpu_cost_build.hpp"
#include "nlohmann/json.hpp"
#include <fstream>
#include <map>
#include <mutex>
#include <thread>
#include <cmath>
#if defined(_MSC_VER) && defined(_M_X64)
#include <intrin.h>
#endif
namespace strata_ds4 {
std::string cpu_cost_build_id(){return STRATA_CPU_COST_BUILD_ID;}
std::string cpu_cost_mode() {
    const char *single=std::getenv("STRATA_DS4_CPU_MMVQ_SINGLE");
    if(!cpu_mmvq_fast_enabled() || !cpu_mmvq_avx2_available() || (single && std::string(single)=="1"))return "unsupported";
    const auto options=cpu_pool_options();
    return "avx2-multi-v1/spin="+std::to_string(options.spin_us)+"/parts="+std::to_string(options.partitions);
}
std::string cpu_cost_machine() {
    std::string brand;
#if defined(_MSC_VER) && defined(_M_X64)
    int regs[4];__cpuid(regs,0x80000000);
    if(unsigned(regs[0])>=0x80000004u)for(unsigned i=0x80000002u;i<=0x80000004u;++i){__cpuid(regs,int(i));brand.append(reinterpret_cast<char *>(regs),16);}
    brand=brand.c_str();
#else
    if(const char *s=std::getenv("PROCESSOR_IDENTIFIER"))brand=s;
    // Portable builds must provide an explicit machine identity to use a profile.
    if(const char *s=std::getenv("STRATA_DS4_CPU_MACHINE"))brand=s;
#endif
    return brand+"/threads="+std::to_string(std::thread::hardware_concurrency());
}
CpuCostModel cpu_cost_load(const std::string &path,const CpuExpertWork &w,int threads,const std::string &gpu,std::string &reason) {
    using json=nlohmann::json;static std::mutex mutex;static std::map<std::string,json> snapshots;
    CpuCostModel model;
    try {
        if(cpu_cost_machine().rfind("/threads=",0)==0)throw std::runtime_error("CPU identity unavailable; set STRATA_DS4_CPU_MACHINE");
        std::lock_guard<std::mutex> lock(mutex);
        auto it=snapshots.find(path);
        if(it==snapshots.end()) {
            std::ifstream f(path,std::ios::binary|std::ios::ate);
            if(!f || f.tellg()<1 || f.tellg()>(4<<20))throw std::runtime_error("profile missing or exceeds 4 MiB");
            f.seekg(0);json j;f>>j;it=snapshots.emplace(path,std::move(j)).first;
        }
        const auto &j=it->second;
        const int version=j.at("version");
        if(cpu_cost_mode()=="unsupported" || version<1 || version>3 || j.at("build")!=cpu_cost_build_id() || j.at("machine")!=cpu_cost_machine() ||
           j.at("gpu")!=gpu || j.at("mode")!=cpu_cost_mode() || j.at("threads")!=threads)
            throw std::runtime_error("profile fingerprint mismatch");
        for(const auto &p:j.at("profiles")) {
            if(p.at("gate")!=int(w.gate_type) || p.at("up")!=int(w.up_type) || p.at("down")!=int(w.down_type) ||
               p.at("columns")!=w.columns || p.at("hidden")!=w.hidden || p.at("clamp").get<float>()!=w.clamp)continue;
            if(p.at("resident_weight_bytes").get<uint64_t>()<(65ULL<<20))throw std::runtime_error("calibration working set too small");
            auto positive=[](float v){return std::isfinite(v) && v>0;};
            model.copy_us_per_byte=p.at("copy_us_per_byte").get<float>();
            if(!positive(model.copy_us_per_byte))throw std::runtime_error("invalid transfer cost");
            const char *burst=std::getenv("STRATA_DS4_CPU_COST_BURST");
            model.burst=version>=3 && burst && std::string(burst)=="1";
            if(version>=3 && p.at("burst_gap_requested_us")!=20000)throw std::runtime_error("unsupported burst calibration gap");
            for(const auto &row:p.at("cases")) {
                const int t=row.at("tokens").get<int>();auto counts=row.at("counts").get<std::vector<int>>();
                if(t<1 || t>4 || counts.size()>8)throw std::runtime_error("invalid calibration case");
                for(int c:counts)if(c<1 || c>t)throw std::runtime_error("invalid repeat count");
                const int key=cpu_cost_key(counts.data(),int(counts.size()));const float us=row.at("compute_us");
                if(key<=0 || !positive(us) || model.compute[t-1][key])throw std::runtime_error("invalid or duplicate compute cost");
                model.compute[t-1][key]=us;
                if(version>=2) {
                    const float loaded=row.at("loaded_compute_us");if(!positive(loaded))throw std::runtime_error("invalid loaded CPU cost");
                    model.loaded_compute[t-1][key]=loaded;
                }
                if(version>=3) {
                    const float burst_us=row.at("burst_compute_us"),burst_loaded=row.at("burst_loaded_compute_us");
                    if(!positive(burst_us) || !positive(burst_loaded))throw std::runtime_error("invalid burst CPU cost");
                    for(const char *field:{"burst_samples_us","burst_loaded_samples_us","burst_gaps_us","burst_loaded_gaps_us"}) {
                        const auto &values=row.at(field);if(!values.is_array() || values.size()!=8)throw std::runtime_error("incomplete burst measurements");
                        for(const auto &value:values)if(!positive(value.get<float>()))throw std::runtime_error("invalid burst measurement");
                    }
                    if(model.burst) {
                        model.compute[t-1][key]=std::max(us,burst_us);
                        model.loaded_compute[t-1][key]=std::max(model.loaded_compute[t-1][key],burst_loaded);
                    }
                }
            }
            for(const auto &row:p.at("boundary")) {
                const int t=row.at("tokens"),r=row.at("rows");const float us=row.at("us");
                if(t<1 || t>4 || r<1 || r>8 || !positive(us))throw std::runtime_error("invalid queue cost");
                if(model.boundary[t-1][r])throw std::runtime_error("duplicate queue cost");
                model.boundary[t-1][r]=us;
            }
            for(int t=0;t<4;++t)for(int r=1;r<=8;++r)if(!positive(model.boundary[t][r]))throw std::runtime_error("missing queue cost");
            if(version>=2) {
                model.loaded_copy_us_per_byte=p.at("loaded_copy_us_per_byte");
                if(!positive(model.loaded_copy_us_per_byte))throw std::runtime_error("invalid loaded transfer cost");
                model.gpu_topk=p.at("gpu").at("topk");
                if(model.gpu_topk<1 || model.gpu_topk>8 || (w.topk>0 && model.gpu_topk!=w.topk))throw std::runtime_error("unsupported calibrated top-k");
                int samples[4][9]={};
                for(const auto &row:p.at("gpu").at("cases")) {
                    const int t=row.at("tokens"),r=row.at("cpu_rows");const float us=row.at("us");
                    if(t<1 || t>4 || r<0 || r>std::min(4,t*model.gpu_topk) || !positive(us))throw std::runtime_error("invalid GPU compute cost");
                    // Conservative envelope of repeated/distinct routes and packed/spread masks.
                    const std::string layout=row.at("layout"),mask=row.at("mask");
                    if((layout!="repeated" && layout!="distinct") || (mask!="packed" && mask!="spread"))throw std::runtime_error("unknown GPU calibration layout");
                    const int bit=1<<((layout=="distinct"?2:0)+(mask=="spread"?1:0));
                    if(samples[t-1][r]&bit)throw std::runtime_error("duplicate GPU cost layout");
                    model.gpu_compute[t-1][r]=std::max(model.gpu_compute[t-1][r],us);samples[t-1][r]|=bit;
                }
                for(int t=0;t<4;++t)for(int r=0;r<=std::min(4,(t+1)*model.gpu_topk);++r)if(samples[t][r]!=15)throw std::runtime_error("incomplete GPU cost coverage");
                const auto &overlap=p.at("gpu").at("overlap");
                const float serial=overlap.at("serial_us"),parallel=overlap.at("parallel_us");
                if(!positive(serial) || !positive(parallel) || overlap.at("tokens")!=3 || overlap.at("cpu_rows")!=3)
                    throw std::runtime_error("invalid overlap measurement");
                // No credit for overlap unless the alternating measurement
                // actually reduced elapsed time. Never subtract more than the
                // shorter component when applying this fraction to a prefix.
                model.gpu_overlap=std::clamp((serial-parallel)/model.gpu_compute[2][3],0.0f,1.0f);
                const char *load=std::getenv("STRATA_DS4_CPU_COST_CONTENTION"),*gpu_cost=std::getenv("STRATA_DS4_CPU_COST_GPU");
                model.contention=!(load && std::string(load)=="0");model.gpu_enabled=!(gpu_cost && std::string(gpu_cost)=="0");model.revision=version;
            }
            model.enabled=true;reason="loaded";return model;
        }
        reason="unmeasured format/geometry/clamp";
    }catch(const std::exception &e){reason=e.what();}
    return {};
}
}
