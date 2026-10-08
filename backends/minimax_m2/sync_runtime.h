#pragma once
#include "ggml-backend.h"
#include <atomic>
#include <cstdint>
struct strata_mm27_stats {
    uint64_t compute_calls=0,gpu_nodes=0,expert_nodes=0,rejected_cpu_nodes=0,rejected_full_copies=0;
    uint64_t ranges=0,chunks=0,source_bytes=0,h2d_bytes=0,staging_bytes=0;
    uint64_t file_bytes=0,mmap_bytes=0,host_working_set_limit=0,host_budget_updates=0;
    uint64_t memory_checks=0,pressure_rejections=0;
    uint64_t sampled_ram_used_peak=0,sampled_vram_used_peak=0,sampled_private_peak=0,sampled_working_set_peak=0;
    uint64_t cache_hits=0,cache_misses=0,cache_hit_bytes=0,cache_fill_bytes=0,cache_guard_bytes=0;
    uint64_t cache_evictions=0,cache_reuses=0,cache_oom=0,cache_bypasses=0,cache_resident=0,cache_limit=0;
    uint64_t selected_bytes=0;
    uint64_t arena_reserved=0,arena_live=0,arena_blocks=0,arena_allocations=0,arena_frees=0,arena_rejects=0;
    uint64_t pipeline_groups=0,pipeline_chunks=0,pipeline_h2d_bytes=0,pipeline_d2d_bytes=0,pipeline_unused_bytes=0;
    uint64_t pipeline_device_bytes=0,pipeline_queued_bytes=0,pipeline_reader_owned_bytes=0,pipeline_read_peak=0;
    uint64_t pipeline_wait_us=0,pipeline_slot_wait_us=0,pipeline_read_us=0,pipeline_submit_us=0;
    uint64_t pipeline_plans=0,pipeline_matrices=0,pipeline_plan_peak=0,pipeline_lookahead_plans=0;
    uint64_t pipeline_copy_batches=0,pipeline_copy_fences=0,pipeline_scratch_fences=0;
    uint64_t pipeline_copy_submissions=0,pipeline_pending_fills_peak=0,pipeline_abort_fences=0;
    double pipeline_delivery_ms=0;
    double source_ms=0,h2d_ms=0,compute_ms=0;
};
struct strata_mm27_memory_info {
    uint64_t ram_total,ram_available,vram_total,vram_available,process_private,process_working_set;
};
using strata_mm27_observer=void (*)(ggml_backend_t,const ggml_tensor *,const ggml_tensor *,size_t,size_t,void *);
// Thread-confined runtime; mode 1 is the native selected-copy reference, 2 bounded file reads.
void strata_mm27_mode(int mode);
// Mode 2 source: 0 ReadFile->pinned, 1 mmap->pinned, 2 pageable mmap->CUDA,
// 3 ReadFile for prefill/workspace warmup, then mmap->pinned for serial decode.
// Mapped readers constrain this process's working set; release restores it.
void strata_mm27_reader(int reader);
// 0 disables; 1/2 native-file producers, four slots of 4/8/16 MiB each.
// Configuration is lazy: allocation occurs after model/context admission.
// Lookahead plans at most three matrices using already-observed router IDs.
void strata_mm27_pipeline(int readers,int chunk_mib=8,bool lookahead=false,bool d2d_batch=false);
void strata_mm27_pipeline_trace(bool enabled);
void strata_mm27_pipeline_trace_write(const char *path);
void strata_mm27_cancel(const std::atomic<bool> *flag);
void strata_mm27_observe(strata_mm27_observer fn,void *owner);
void strata_mm27_reset();
void strata_mm27_release(); // Only after synchronization; clears hooks and pinned storage.
strata_mm27_stats strata_mm27_snapshot();
const char *strata_mm27_last_error();
// One live model per thread. Bind validates immutable tensor/mapping identities;
// unbind invalidates all entries before the corresponding model is freed.
void strata_mm27_cache_configure(uint64_t bytes,bool arena=false);
void strata_mm27_cache_bind(const void *model,const ggml_tensor *const *tensors,size_t count);
void strata_mm27_cache_unbind(const void *model);
void strata_mm27_cache_clear();
void strata_mm27_cache_decode(bool admit); // false during prefill/workspace warmup
strata_mm27_memory_info strata_mm27_memory(uint64_t gpu_reserve=0,uint64_t ram_reserve=0);
