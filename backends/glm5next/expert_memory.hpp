#pragma once
#include "expert_cache.hpp"
#include "global_memory.hpp"
#include "../common/vram_policy.hpp"
#include <chrono>
#include <functional>
#include <optional>

namespace strata_glm {
// One combined main/MTP cache per device. Invoke at graph/dispatch boundaries,
// after fixed weights, state, ring and workspace have been allocated. The probe
// must report GLOBAL free/total for this CUDA device (PCI matched NVML on WDDM),
// or return false. Do not substitute the WDDM per-process CUDA memory view.
// This computes a target, not an atomic reservation against other processes.
class ExpertMemoryController {
public:
    using Clock=std::chrono::steady_clock;
    using Probe=std::function<bool(int,size_t &,size_t &)>;
    struct BranchStatus {
        size_t limit=0,resident=0,deferred=0;
    };
    struct Status {
        bool sample_valid=false,trim_complete=false;
        size_t free=0,total=0,target=0,resident=0,deferred=0;
        size_t global_deferred=0;
        BranchStatus main,mtp;
        uint64_t samples=0,failed_samples=0;
        uint64_t skipped_refreshes=0;
    };
private:
    ExpertCache &cache;
    size_t cap;
    StrataVramPolicy policy;
    Probe probe;
    Status state;
    std::optional<Clock::time_point> last_attempt;
    void device_check() const {
        int device=-1;
        auto error=cudaGetDevice(&device);
        if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));
        if(device!=cache.device())throw std::runtime_error("GLM memory controller used on another CUDA device");
    }
    void snapshot_cache() {
        state.target=cache.byte_budget();state.resident=cache.resident_bytes();
        auto branch=[&](Branch b) {
            const auto limit=cache.byte_budget(b),resident=cache.resident_bytes(b);
            return BranchStatus{limit,resident,resident>limit?resident-limit:0};
        };
        state.main=branch(Branch::main);state.mtp=branch(Branch::mtp);
        state.global_deferred=state.resident>state.target?state.resident-state.target:0;
        // Branch excesses are disjoint, but releasing them also reduces global
        // excess. Do not count those same bytes twice. Whole-matrix eviction can
        // release more than this byte lower bound when pins/events permit it.
        state.deferred=std::max(state.global_deferred,state.main.deferred+state.mtp.deferred);
    }
    void unavailable() {
        ++state.failed_samples;
        state.sample_valid=false;state.trim_complete=false;
        state.free=state.total=0;snapshot_cache();
    }
public:
    ExpertMemoryController(ExpertCache &c,size_t configured_cap,StrataVramPolicy p)
        :ExpertMemoryController(c,configured_cap,p,make_global_memory_probe()) {}
    ExpertMemoryController(ExpertCache &c,size_t configured_cap,StrataVramPolicy p,Probe reader)
        :cache(c),cap(configured_cap),policy(p),probe(std::move(reader)) {
        // A matrix count is not a byte budget for mixed quantized matrices.
        if(!policy.valid() || policy.mode==1 || !probe)
            throw std::invalid_argument("GLM memory controller requires a byte policy and a global probe");
        device_check();cache.set_admission_enabled(false);
        snapshot_cache();
    }
    ExpertMemoryController(const ExpertMemoryController&)=delete;
    ExpertMemoryController &operator=(const ExpertMemoryController&)=delete;
    Status status() const {return state;} // Snapshot from last refresh, not live usage.
    // Force a new sample after fixed/state/workspace allocations or other known
    // pressure changes. Explicit timestamps allow deterministic scheduler tests.
    Status refresh(Clock::time_point now=Clock::now()) {
        device_check();++state.samples;
        last_attempt=now; // Rate-limit failures too; do not retry on every layer.
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
        try {state.trim_complete=cache.set_budget(state.target);}
        catch(...) {snapshot_cache();throw;}
        snapshot_cache();
        cache.set_admission_enabled(true);
        return state;
    }
    // Call only at safe dispatch boundaries, as with refresh(). No background
    // thread: a skipped call returns the previous snapshot and does not trim or
    // change admission. The caller chooses the interval; zero always samples.
    Status refresh_if_due(Clock::duration interval,Clock::time_point now=Clock::now()) {
        if(interval<Clock::duration::zero())
            throw std::invalid_argument("GLM memory refresh interval must be nonnegative");
        if(!last_attempt || now<*last_attempt || now-*last_attempt>=interval)
            return refresh(now);
        device_check();
        ++state.skipped_refreshes;
        return state;
    }
    // Destruction deliberately does not enable admissions after an invalid sample.
    // cache outlives the controller; caller owns the explicit return to manual mode.
};
}
