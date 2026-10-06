#pragma once
#include <cstddef>
#include <cstdint>
struct ggml_backend;
struct ggml_tensor;
struct strata_step_sync_stats {
    uint64_t source_bytes, h2d_bytes, ranges, chunks, staging_bytes;
    uint64_t compute_calls, gpu_nodes, expert_nodes, rejected_cpu_nodes, rejected_full_copies;
    double source_ms, h2d_ms;
    uint64_t cache_hits, cache_misses, cache_evictions, cache_bypasses, cache_oom, cache_rejected;
    uint64_t cache_bytes, cache_limit, d2d_bytes, cache_fill_bytes, requested_bytes;
    uint64_t prefill_admission_skips, prefill_admission_skip_bytes;
    uint64_t gpu_free, gpu_total, ram_free, ram_total, memory_samples;
    double d2d_ms;
    uint64_t pipeline_groups, pipeline_chunks, pipeline_unused_bytes, pipeline_device_bytes;
    uint64_t pipeline_wait_us, pipeline_slot_wait_us, pipeline_submit_us, pipeline_read_peak;
    uint64_t cache_allocations, cache_reuses;
    uint64_t pipeline_copy_batches, pipeline_copy_fences;
    double pipeline_batch_ms;
    // Optional inclusive host wall times; nested fields cannot be summed.
    double cache_get_ms, cache_admit_ms, cache_refresh_ms, memory_probe_ms, cache_protect_ms, cache_trim_ms;
    double cache_victim_ms, cache_allocate_ms, cache_free_ms, plan_build_ms, plan_end_ms;
    uint64_t cache_victim_candidates;
    uint64_t host_copy_calls,host_copy_bytes,host_copy_wall_ns,host_copy_cycles,host_copy_slow;
};
// mode0: unmodified scheduler; mode1: native selected-copy oracle + GPU audit;
// mode2: bounded synchronous pinned staging + GPU audit. One owning thread.
using strata_step_copy_observer = void (*)(void *, ggml_backend *, ggml_tensor *, const ggml_tensor *, size_t, size_t);
extern "C" {
void strata_step_sync_mode(int mode);
void strata_step_sync_reset();
strata_step_sync_stats strata_step_sync_snapshot();
void strata_step_sync_observer(strata_step_copy_observer observer, void * owner);
void strata_step_sync_release();
void strata_step_cache_begin(const char * model_identity, size_t cap_bytes);
void strata_step_cache_register(const ggml_tensor *, const char * shard, uint64_t file_offset);
void strata_step_cache_refresh();
void strata_step_cache_trim(size_t bytes);
void strata_step_cache_prefill(bool enabled);
// Opt-in experiment; owning thread, between requests, no outstanding plan.
void strata_step_cache_reuse(bool enabled);
void strata_step_cache_profile(bool enabled) noexcept(false);
// Same victim policy, direct order-node lookup; experimental, default off.
void strata_step_cache_fast_scan(bool enabled) noexcept(false);
// Owning thread only. 0=idle, 1=prefill (including its tail), 2=decode.
// Returns the previous phase for exception-safe scopes.
int strata_step_request_phase(int phase);
// Explicit throwing contract: MSVC /EHsc otherwise assumes C-linkage calls
// cannot throw and can remove a caller's guard/recovery handler.
void strata_step_pipeline_config(int readers, int chunk_mib, int trace_graphs) noexcept(false);
// Experimental tensor batching: pins cache sources/fills through its final fence.
// Must be called by the owning thread between requests.
void strata_step_pipeline_config_ex(int readers, int chunk_mib, int trace_graphs, bool early_refill, bool batch_copy) noexcept(false);
// Explicit pinned-host allocation policy; older entry points retain cacheable RAM.
void strata_step_pipeline_config_staging(int readers, int chunk_mib, int trace_graphs, bool early_refill, bool batch_copy, bool write_combined) noexcept(false);
// 0=CRT, 1=AVX2 temporal stores (runtime CPU/OS guard). Between drained requests.
void strata_step_pipeline_config_copy(int readers, int chunk_mib, int trace_graphs, bool early_refill, bool batch_copy, bool write_combined, int host_copy, bool profile) noexcept(false);
void strata_step_trace_write(const char * path) noexcept(false);
}
