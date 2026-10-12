#pragma once
#include "ggml.h"
#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

// Sleeping persistent workers; only the optional routed-matrix lane uses them.
// The graph scheduler still rejects unrelated CPU operations.
namespace strata_ds4 {
class CpuLanePool {
    struct Impl;
    std::unique_ptr<Impl> p;
public:
    explicit CpuLanePool(int threads);
    ~CpuLanePool();
    void submit(int jobs, std::function<void(int)> job);
    void wait();
    void run(int jobs, const std::function<void(int)> &job);
    int size() const;
};
bool cpu_lane_format(ggml_type type, int columns);
size_t cpu_lane_activation_bytes(ggml_type type, int columns);
void cpu_lane_quantize(ggml_type type, int columns, const float *x, void *q);
void cpu_lane_rows(ggml_type type, int columns, const void *weight,
                   const void *q, float *output, int first, int last);
struct CpuExpert {
    const void *gate, *up, *down;
    int token, slot;
};
struct CpuExpertWork {
    ggml_type gate_type, up_type, down_type;
    int columns=0, hidden=0, tokens=0, topk=0;
    float clamp=0;
    const float *input=nullptr;
    float *output=nullptr;
    std::vector<CpuExpert> experts;
    std::vector<uint8_t> gate_q, up_q, hidden_q;
    std::vector<float> hidden_f;
    void compute(CpuLanePool &pool);
};
}
