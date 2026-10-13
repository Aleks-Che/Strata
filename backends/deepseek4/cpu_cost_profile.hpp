#pragma once
#include "cpu_cost_model.hpp"
#include "cpu_lane.hpp"
#include <string>
namespace strata_ds4 {
std::string cpu_cost_build_id();
std::string cpu_cost_machine();
std::string cpu_cost_mode();
// Immutable per-process JSON snapshot, validated before any decision reaches GPU.
CpuCostModel cpu_cost_load(const std::string &path,const CpuExpertWork &,int threads,
                          const std::string &gpu,std::string &reason);
}
