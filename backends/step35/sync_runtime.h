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
// Owning thread only. 0=idle, 1=prefill (including its tail), 2=decode.
// Returns the previous phase for exception-safe scopes.
int strata_step_request_phase(int phase);
void strata_step_pipeline_config(int readers, int chunk_mib, int trace_graphs);
void strata_step_trace_write(const char * path);
}
