#pragma once
#include "cpu_cost_profile.hpp"
#include <array>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>

namespace strata_ds4 {
// Only request boundaries publish a successor. Service threads append bounded
// measurements; they never change a table used by an active CUDA graph.
class CpuCostRefresh {
public:
    struct Result { uint64_t generation=0, samples=0; int changed=0; std::string error; };
    using Writer=std::function<void(const CpuCostModel &)>;
    explicit CpuCostRefresh(const CpuCostModel &base):base_(base),current_(base),budget_(base) {}
    CpuCostModel snapshot() const;
    bool budget_possible(int tokens) const;
    void begin(bool eligible);
    void observe(uint64_t generation,int tokens,int pattern,bool copying,double us);
    Result finish(bool success,const Writer &write={});
private:
    using Key=std::tuple<int,int,bool>;
    struct Samples { std::array<float,63> values{}; uint64_t count=0; int direction=0; };
    mutable std::mutex mutex_;
    CpuCostModel base_,current_;
    CpuCostBudgetProof budget_;
    std::map<Key,Samples> cells_;
    bool active_=false;
};
bool cpu_cost_auto_enabled();
std::shared_ptr<CpuCostRefresh> cpu_cost_refresh_for(const std::string &path,
    const CpuExpertWork &,int threads,const std::string &gpu,const CpuCostModel &);
// Called by the CUDA extension only after draining ALL target/draft contexts.
// operation: 0 begin eligible, 1 begin excluded, 2 commit, 3 discard.
void cpu_cost_request_boundary(int operation);
}
