#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
struct ggml_backend;
struct ggml_tensor;
struct strata_hy3_sync_stats {
    uint64_t source_bytes, h2d_bytes, ranges, chunks, staging_bytes;
    uint64_t compute_calls, gpu_nodes, expert_nodes, rejected_cpu_nodes, rejected_full_copies;
    uint64_t gpu_free, gpu_total, ram_free, ram_total, memory_samples;
    uint64_t working_set_limit;
    uint64_t cache_generation, cache_bytes, cache_budget, cache_entries;
    uint64_t cache_hits, cache_misses, cache_evictions, cache_oom, cache_rejected, cache_reuses;
    uint64_t cache_fill_bytes, d2d_bytes;
    uint64_t pipeline_groups, pipeline_chunks, pipeline_unused_bytes, pipeline_device_bytes;
    uint64_t pipeline_wait_us, pipeline_slot_wait_us, pipeline_submit_us, pipeline_read_peak;
    uint64_t pipeline_reader_owned, pipeline_queued, pipeline_copy_fences;
    uint64_t pipeline_copy_batches, pipeline_pending_fills_peak;
    double source_ms, h2d_ms, pipeline_batch_ms;
};
using strata_hy3_copy_observer = void (*)(void *, ggml_backend *, ggml_tensor *, const ggml_tensor *, size_t, size_t);
// C++ linkage is required: mode/memory admission can throw. MSVC /EHsc
// assumes extern-C calls cannot throw and can omit required unwind state.
// 0: unchanged scheduler, 1: native selected-copy oracle + GPU audit,
// 2: synchronous bounded native-file -> pinned -> GPU + audit. One owner thread.
void strata_hy3_sync_mode(int mode);
void strata_hy3_sync_reset();
strata_hy3_sync_stats strata_hy3_sync_snapshot();
void strata_hy3_sync_observer(strata_hy3_copy_observer observer, void * owner);
void strata_hy3_sync_cancel(const std::atomic<bool> * cancel);
void strata_hy3_memory_check(size_t gpu_reserve=0);
void strata_hy3_sync_release();
// One live immutable model per owner thread. begin invalidates every prior key.
void strata_hy3_cache_begin(size_t cap);
void strata_hy3_cache_register(const ggml_tensor * tensor);
// Four shared transport slots; explicit opt-in, one model/scheduler owner.
void strata_hy3_pipeline_config(int readers, int chunk_mib=4, int trace_graphs=0, bool batch_copy=false);
void strata_hy3_trace_write(const char * path);
