#pragma once
#include <cstddef>
#include <cstdint>

// P1 correctness baseline, one owner thread. No cache or speculative prefetch.
struct strata_glm_sync_stats {
    uint64_t source_bytes, h2d_bytes, ranges, chunks, staging_bytes;
    uint64_t compute_calls, gpu_nodes, expert_nodes, rejected_cpu_nodes;
    double source_ms, h2d_ms;
};
struct ggml_backend;
struct ggml_tensor;
using strata_glm_copy_hook = void (*)(void *, ggml_backend *, ggml_tensor *, const ggml_tensor *, size_t, size_t, strata_glm_sync_stats *);
extern "C" {
void strata_glm_sync_copy_hook(strata_glm_copy_hook hook, void * owner);
void strata_glm_sync_enable(bool enabled);
// Validation reference: use the candidate's original selected-range transfer.
// GPU audit remains enabled; no pinned/source timing claim is made for it.
void strata_glm_sync_candidate_copy(bool enabled);
void strata_glm_sync_reset();
strata_glm_sync_stats strata_glm_sync_snapshot();
void strata_glm_sync_release();
}
