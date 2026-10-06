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
    uint64_t ram_cache_bytes, ram_cache_budget, ram_cache_entries, ram_cache_peak_bytes, ram_cache_pending;
    uint64_t ram_cache_hits, ram_cache_misses, ram_cache_hit_bytes, ram_cache_file_bytes, ram_cache_fill_bytes;
    uint64_t ram_cache_evictions, ram_cache_rejected, ram_cache_oom, ram_cache_allocations, ram_cache_reuses;
    uint64_t ram_cache_prefill_bypasses, ram_cache_frequency_bypasses, ram_cache_victim_candidates, ram_cache_history_entries;
    uint64_t ram_cache_reused_payload_bytes;
    uint64_t mtp_cache_bytes;
    uint64_t cache_generation, cache_bytes, cache_budget, cache_entries;
    uint64_t cache_backing_bytes, cache_backing_slack_bytes, cache_arena_blocks;
    uint64_t cache_arena_allocations, cache_arena_frees;
    uint64_t cache_arena_budget_rejects;
    uint64_t cache_hits, cache_misses, cache_evictions, cache_oom, cache_rejected, cache_reuses;
    uint64_t cache_fill_bytes, d2d_bytes;
    uint64_t cache_prefill_bypasses;
    uint64_t pipeline_groups, pipeline_chunks, pipeline_unused_bytes, pipeline_device_bytes;
    uint64_t pipeline_wait_us, pipeline_slot_wait_us, pipeline_submit_us, pipeline_read_peak;
    uint64_t pipeline_reader_owned, pipeline_queued, pipeline_copy_fences;
    uint64_t pipeline_copy_batches, pipeline_pending_fills_peak;
    uint64_t cache_allocations, cache_victim_candidates;
    double source_ms, h2d_ms, pipeline_batch_ms;
    double cache_get_ms, cache_admit_ms, cache_victim_ms, cache_allocate_ms, cache_free_ms;
    double cache_refresh_ms, cache_probe_ms, cache_protect_ms, cache_trim_ms;
    double pipeline_scratch_wait_ms, pipeline_delivery_wait_ms;
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
void strata_hy3_cache_register(const ggml_tensor * tensor, bool mtp=false);
void strata_hy3_gpu_cache_policy(bool admit_prefill);
// Applies at the next cache_begin; existing cache ownership is unchanged.
void strata_hy3_gpu_cache_allocator(bool arena);
// 0 disables, SIZE_MAX selects the dynamic host budget. Configure before readers.
void strata_hy3_ram_cache_config(size_t cap, bool frequency=true);
// Explicit request phase, including batched MTP verification as decode.
void strata_hy3_cache_prefill(bool enabled);
// Opt-in inclusive CPU wall times; nested cache scopes must not be summed.
void strata_hy3_delivery_profile(bool enabled);
// Four shared transport slots; explicit opt-in, one model/scheduler owner.
void strata_hy3_pipeline_config(int readers, int chunk_mib=4, int trace_graphs=0, bool batch_copy=false);
void strata_hy3_trace_write(const char * path);
