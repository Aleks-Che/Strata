#pragma once
#include <cstddef>
#include <cstdint>

// One owner thread; optional cache/pipeline hooks retain the synchronous reference.
struct strata_glm_sync_stats {
    uint64_t source_bytes, h2d_bytes, ranges, chunks, staging_bytes;
    uint64_t compute_calls, gpu_nodes, expert_nodes, rejected_cpu_nodes;
    double source_ms, h2d_ms;
};
struct ggml_backend;
struct ggml_tensor;
struct strata_glm_expert { const ggml_tensor * tensor; int32_t expert; };
using strata_glm_copy_hook = void (*)(void *, ggml_backend *, ggml_tensor *, const ggml_tensor *, size_t, size_t, size_t, strata_glm_sync_stats *);
using strata_glm_plan_hook = void (*)(void *, const strata_glm_expert *, size_t, bool, strata_glm_sync_stats *);
using strata_glm_finish_hook = void (*)(void *, bool, strata_glm_sync_stats *);
using strata_glm_compute_hook = void (*)(void *, ggml_backend *, bool);
extern "C" {
void strata_glm_sync_copy_hook(strata_glm_copy_hook hook, void * owner);
void strata_glm_sync_plan_hooks(strata_glm_plan_hook plan, strata_glm_finish_hook finish);
void strata_glm_sync_compute_hook(strata_glm_compute_hook compute);
// The copy hook itself fences both scratch reuse and compute readiness on GPU.
void strata_glm_sync_event_copy(bool enabled);
void strata_glm_sync_enable(bool enabled);
// Validation reference: use the candidate's original selected-range transfer.
// GPU audit remains enabled; no pinned/source timing claim is made for it.
void strata_glm_sync_candidate_copy(bool enabled);
void strata_glm_sync_reset();
strata_glm_sync_stats strata_glm_sync_snapshot();
void strata_glm_sync_release();
}
