#pragma once
#include "ggml.h"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

// Sleeping persistent workers; only the optional routed-matrix lane uses them.
// The graph scheduler still rejects unrelated CPU operations.
namespace strata_ds4 {
struct CpuPoolTiming {
    uint64_t dispatch_us=0, total_us=0, first_worker_us=0, last_worker_us=0;
    int workers=0;
};
struct CpuExpertTiming {
    uint64_t prepare_us=0, gate_up_us=0, hidden_quant_us=0, down_us=0;
    CpuPoolTiming gate_up_pool, down_pool;
};
class CpuLanePool {
    struct Impl;
    std::unique_ptr<Impl> p;
public:
    explicit CpuLanePool(int threads);
    ~CpuLanePool();
    void submit(int jobs, std::function<void(int)> job);
    void wait();
    void run(int jobs, const std::function<void(int)> &job, CpuPoolTiming *timing=nullptr);
    int size() const;
};
bool cpu_lane_format(ggml_type type, int columns);
// Slow diagnostic only; never enabled by the automatic placement policy.
bool cpu_mmvq_reference_enabled();
size_t cpu_lane_activation_bytes(ggml_type type, int columns);
void cpu_lane_quantize(ggml_type type, int columns, const float *x, void *q);
void cpu_lane_rows(ggml_type type, int columns, const void *weight,
                   const void *q, float *output, int first, int last);
struct CpuExpert {
    const void *gate, *up, *down;
    int token, slot;
};
enum class CpuWeightPhase { gate_up, down };
using CpuWeightRetire = std::function<void(CpuWeightPhase)>;
struct CpuExpertWork {
    ggml_type gate_type, up_type, down_type;
    int columns=0, hidden=0, tokens=0, topk=0;
    float clamp=0;
    const float *input=nullptr;
    float *output=nullptr;
    std::vector<CpuExpert> experts;
    std::vector<uint8_t> gate_q, up_q, hidden_q;
    std::vector<float> hidden_f;
    // Called after the pool barrier: the completed phase will never read its
    // weights again. Output/activations and the work descriptor remain live.
    void compute(CpuLanePool &pool, CpuExpertTiming *timing=nullptr, const CpuWeightRetire &retire={});
};
void cpu_mmvq_reference_compute(CpuExpertWork &,CpuLanePool &,CpuExpertTiming *,const CpuWeightRetire &);
}
