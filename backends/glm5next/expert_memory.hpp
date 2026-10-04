#pragma once
#include "expert_cache.hpp"
#include "../common/vram_policy.hpp"
#include <functional>

namespace strata_glm {
// One combined main/MTP cache per device. Invoke at graph/dispatch boundaries,
// after fixed weights, state, ring and workspace have been allocated. The probe
// must report GLOBAL free/total for this CUDA device (PCI matched NVML on WDDM),
// or return false. Do not substitute the WDDM per-process CUDA memory view.
// This computes a target, not an atomic reservation against other processes.
class ExpertMemoryController {
public:
    using Probe=std::function<bool(int,size_t &,size_t &)>;
    struct Status {
        bool sample_valid=false,trim_complete=false;
        size_t free=0,total=0,target=0,resident=0,deferred=0;
        uint64_t samples=0,failed_samples=0;
    };
private:
    ExpertCache &cache;
    size_t cap;
    StrataVramPolicy policy;
    Probe probe;
    Status state;
    void device_check() const {
        int device=-1;
        auto error=cudaGetDevice(&device);
        if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));
        if(device!=cache.device())throw std::runtime_error("GLM memory controller used on another CUDA device");
    }
    void unavailable() {
        ++state.failed_samples;
        state.sample_valid=false;state.trim_complete=false;
        state.free=state.total=0;state.target=cache.byte_budget();
        state.resident=cache.resident_bytes();
        state.deferred=state.resident>state.target?state.resident-state.target:0;
    }
public:
    ExpertMemoryController(ExpertCache &c,size_t configured_cap,StrataVramPolicy p,Probe reader)
        :cache(c),cap(configured_cap),policy(p),probe(std::move(reader)) {
        // A matrix count is not a byte budget for mixed quantized matrices.
        if(!policy.valid() || policy.mode==1 || !probe)
            throw std::invalid_argument("GLM memory controller requires a byte policy and a global probe");
        device_check();cache.set_admission_enabled(false);
        state.target=cache.byte_budget();state.resident=cache.resident_bytes();
    }
    ExpertMemoryController(const ExpertMemoryController&)=delete;
    ExpertMemoryController &operator=(const ExpertMemoryController&)=delete;
    Status status() const {return state;} // Snapshot from last refresh, not live usage.
    Status refresh() {
        device_check();++state.samples;
        cache.set_admission_enabled(false); // Also stays closed on probe/trim exceptions.
        size_t free=0,total=0;
        bool valid=false;
        try {valid=probe(cache.device(),free,total);}
        catch(...) {unavailable();throw;}
        const auto resident=cache.resident_bytes();
        if(!valid || !total || free>total || resident>total-free) {unavailable();return state;}
        state.sample_valid=true;state.free=free;state.total=total;
        state.target=size_t(std::min<uint64_t>(cap,policy.byte_limit(free,total,resident)));
        state.trim_complete=false;
        state.trim_complete=cache.set_budget(state.target);
        state.resident=cache.resident_bytes();
        state.deferred=state.resident>state.target?state.resident-state.target:0;
        cache.set_admission_enabled(true);
        return state;
    }
    // Destruction deliberately does not enable admissions after an invalid sample.
    // cache outlives the controller; caller owns the explicit return to manual mode.
};
}
