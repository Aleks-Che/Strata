#pragma once
#include "ggml-backend.h"
#include <cstdint>
#include "vram_policy.hpp"
#include "../common/expert_slice.hpp"

struct StrataExpertCounters {
    uint64_t hits=0, misses=0, bypass=0, evictions=0;
    uint64_t h2d_bytes=0, d2d_bytes=0, stage_bytes=0;
    uint64_t stage_us=0, wait_us=0, submit_us=0;
    uint64_t pipeline_groups=0, pipeline_chunks=0, pipeline_unused=0, pipeline_fallbacks=0;
    uint64_t file_read_bytes=0, mmap_read_bytes=0, read_peak=0;
    uint64_t ordered_reuses=0, reuse_events=0, eviction_syncs=0;
    uint64_t admission_rejects=0;
    uint64_t prefill_hits=0;
    uint64_t frequency_decay=0;
    uint64_t direct_hits=0,direct_bytes=0,direct_materialized_bytes=0,direct_tables=0;
    uint64_t direct_admissions=0;
    uint64_t hash_reads=0, early_plans=0, early_reuses=0, early_mismatches=0;
    uint64_t early_hit_ops=0, early_hit_experts=0, early_miss_experts=0;
    uint64_t cpu_lane_ops=0,cpu_lane_experts=0,cpu_lane_saved_bytes=0;
    uint64_t cpu_queue_jobs=0,cpu_queue_rows=0;
    uint64_t gpu_fetch_branches=0,gpu_fetch_matrices=0;
};
using StrataExpertPlan = void (*)(ggml_backend_t, const StrataExpertSlice *, size_t);
// Exact GPU table lookup for at most three hash-routed layers. These are input
// token IDs, not predicted expert IDs. The model graph is not rewritten.
struct StrataHashRoutes {
    const ggml_tensor *tables[3]={};
    int32_t tokens[8]={};
    int count=0, n_tokens=0;
    int32_t ids[3][64]={};
};
using StrataExpertHashRead = bool (*)(ggml_backend_t, StrataHashRoutes *);
using StrataExpertEarlyPlan = void (*)(ggml_backend_t, const StrataExpertSlice *, size_t);
// Discard unconsumed lookahead and join the producer at every graph exit,
// including errors and cancellation, before model mappings can be released.
using StrataExpertFinish = void (*)(ggml_backend_t);
// Materialize pending cached matrices before the scheduler launches consumers.
using StrataExpertFlush = void (*)(ggml_backend_t);
// Called once per selected-expert input, before copy(). False retains ordinary
// materialization. Every graph execution (including replay) needs a new prepare.
using StrataExpertPrepare = bool (*)(ggml_backend_t,const ggml_tensor *consumer);
using StrataExpertCompact = int (*)(ggml_backend_t,const ggml_tensor *consumer);
using StrataExpertCpuSelect = void (*)(ggml_backend_t,const ggml_tensor *source,
    const ggml_tensor *consumer,const ggml_tensor *route,const int32_t *ids,size_t ids_bytes,uint8_t *selected);
using StrataExpertCpuBegin = bool (*)(ggml_backend_t,const ggml_tensor *source,
    ggml_tensor *consumer,const ggml_tensor *route,const int32_t *ids,size_t ids_bytes);
// A complete, unfused MoE branch. Nodes and original host weights are ordered
// up, gate, down. Only this exact branch may share a CPU assignment.
struct StrataCpuExpertBranch {
    ggml_tensor *nodes[3]={};
    const ggml_tensor *weights[3]={};
    float clamp=0;
};
using StrataCpuExpertPlan = bool (*)(ggml_backend_t,const StrataCpuExpertBranch *,
    const ggml_tensor *route,const int32_t *ids,size_t ids_bytes,uint8_t *selected);
// With CPU_GPU_QUEUE opt-in, null IDs publish a retained table and launch GPU
// routing before the scheduler reads IDs. A later non-null call adopts that
// decision only for the legacy transfer planner; it does not choose CPU work.
using StrataCpuExpertPhase = void (*)(bool decode);
// Optional GPU-requested weight transfer; returns true only for a prepared
// branch. The scheduler must skip its host route read and selected copies.
using StrataExpertGpuFetch = bool (*)(ggml_backend_t,const ggml_tensor *,ggml_tensor *,const ggml_tensor *);
// Whether selected-weight writes can rely on CUDA stream order instead of a
// host wait. The caller still reads and synchronizes every new routing tensor.
using StrataExpertOrdered = bool (*)(ggml_backend_t,const ggml_tensor *consumer);
// Compute cache hits into private scratch before waiting for missing weights.
// The input activation and the supplied routing tensor must already be queued
// on this backend. A false return leaves the normal matvec unchanged.
using StrataExpertEarlyHits = bool (*)(ggml_backend_t,const ggml_tensor *source,
    ggml_tensor *consumer,const ggml_tensor *route,const int32_t *selected,size_t count);
using StrataExpertCopy = void (*)(ggml_backend_t, const ggml_tensor *, ggml_tensor *, int, int, int);
using StrataExpertStats = void (*)(ggml_backend_t, StrataExpertCounters *);
// Set the startup budget on the calling decode thread. Each CUDA backend keeps
// this budget for config mode; live policies can override its residency.
using StrataExpertBudget = void (*)(int cache_mib);
// Selected before the backend's first graph; existing histories retain their
// period. Like the budget, this belongs to the calling execution thread.
using StrataExpertDecay = void (*)(int accesses);

struct StrataVramStatus {
    uint64_t cache_bytes=0, matrices=0, limit_bytes=0, free_bytes=0, total_bytes=0;
    uint64_t allocation_failures=0;
    // CUDA-requested reservation, including unused slots, not driver rounding.
    uint64_t cache_reserved_bytes=0, slab_blocks=0, slab_allocations=0;
    bool enabled=false, telemetry_ok=false, target_unreachable=false;
};
// Called on the graph execution thread, between graph evaluations (also idle).
// A null policy samples/trims without changing the current policy.
using StrataExpertControl = bool (*)(int device, const StrataVramPolicy *, StrataVramStatus *);
