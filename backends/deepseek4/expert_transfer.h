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
};
using StrataExpertPlan = void (*)(ggml_backend_t, const StrataExpertSlice *, size_t);
// Discard unconsumed lookahead and join the producer at every graph exit,
// including errors and cancellation, before model mappings can be released.
using StrataExpertFinish = void (*)(ggml_backend_t);
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
