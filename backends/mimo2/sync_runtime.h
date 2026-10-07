#pragma once
#include "ggml-backend.h"
#include <atomic>
#include <cstdint>
// One owning compute thread. The stdin reader may only set the cancel flag.
struct strata_mimo_stats {
    uint64_t ranges=0,chunks=0,source_bytes=0,h2d_bytes=0,staging_bytes=0;
    uint64_t compute_calls=0,gpu_nodes=0,expert_nodes=0,rejected_cpu_nodes=0,rejected_full_copies=0;
    uint64_t gpu_free=0,gpu_total=0,ram_free=0,ram_total=0;
    double source_ms=0,h2d_ms=0;
    uint64_t requested_bytes=0,cache_hits=0,cache_misses=0,cache_evictions=0,cache_allocations=0,cache_reuses=0;
    uint64_t cache_bypasses=0,cache_oom=0,cache_bytes=0,cache_payload_bytes=0,cache_limit=0,cache_hit_bytes=0,cache_fill_bytes=0;
    uint64_t cache_slot_bytes=0,cache_slab_mib=0,cache_slab_blocks=0,cache_slab_allocations=0,cache_slab_reuses=0,cache_slab_denied=0,cache_slab_oom=0;
    uint64_t cache_decay=0,cache_history_keys=0,cache_frequency_updates=0,cache_frequency_rejected=0,cache_frequency_candidates=0;
    double d2d_ms=0;
    uint64_t host_working_set_limit=0;
    uint64_t pipeline_groups=0,pipeline_chunks=0,pipeline_device_bytes=0,pipeline_unused_bytes=0;
    uint64_t pipeline_read_peak=0,pipeline_reader_owned=0,pipeline_queued=0,pipeline_delivered_bytes=0;
    uint64_t pipeline_consumer_wait_us=0,pipeline_slot_wait_us=0,pipeline_submit_us=0;
    uint64_t pipeline_file_bytes=0,pipeline_mmap_bytes=0,pipeline_d2d_bytes=0;
    uint64_t pipeline_batch=0,pipeline_copy_batches=0,pipeline_copy_fences=0,pipeline_scratch_fences=0;
    uint64_t cache_fill_batch=0,cache_pending=0,pipeline_fill_batches=0,pipeline_fill_submissions=0;
    double pipeline_batch_ms=0;
};
using strata_mimo_observer=void (*)(ggml_backend_t,const ggml_tensor *,const ggml_tensor *,size_t,size_t,void *);
void strata_mimo_mode(int mode); // 0: test/reference only; 1: native mmap copy; 2: managed file/mmap copy and cache
void strata_mimo_cancel(const std::atomic<bool> * flag);
void strata_mimo_observe(strata_mimo_observer observer,void * owner);
void strata_mimo_reset();
strata_mimo_stats strata_mimo_snapshot();
void strata_mimo_memory(uint64_t extra_gpu=0,uint64_t extra_ram=0);
void strata_mimo_release();
void strata_mimo_model_begin();
void strata_mimo_register(const ggml_tensor *tensor);
void strata_mimo_cache(size_t bytes);
int strata_mimo_cache_slab_mib(); // 0: individual allocations; 16/32: physical block MiB.
int strata_mimo_cache_decay(); // 0: LRU; positive values count selected matrices per half-life.
void strata_mimo_cache_prefill(bool admit); // Hits remain usable when admission is disabled.
void strata_mimo_phase(bool prefill);
void strata_mimo_reader(bool mmap); // file: bounded native read buffer; mmap: driver copy from mapped pages
void strata_mimo_workspace_ready(); // First successful decode at configured batch capacity warmed CUDA pools.
void strata_mimo_pipeline_config(int readers,int chunk_mib=8,int trace_graphs=0);
int strata_mimo_pipeline_batch_mode(); // STRATA_MIMO_PIPELINE_BATCH=0/1, fixed at config.
int strata_mimo_cache_fill_batch_mode(); // Only tensor pipeline supports grouped cache fills.
void strata_mimo_trace_write(const char *path);
void strata_mimo_test_pipeline_failure(int submissions); // One-shot injected reader error, fixtures only.
void strata_mimo_test_fill_failure(int submissions); // Throws after enqueueing the selected fill, fixtures only.
