// Included at file scope in a generated copy of ggml-backend.cpp only.
#include "sync_runtime.h"
#include <chrono>

static thread_local bool strata_glm_sync_active = false;
static thread_local bool strata_glm_candidate_copy = false;
static thread_local strata_glm_sync_stats strata_glm_sync_counters = {};
static thread_local ggml_backend_buffer_t strata_glm_staging = nullptr;
static thread_local strata_glm_copy_hook strata_glm_copy_callback = nullptr;
static thread_local void * strata_glm_copy_owner = nullptr;
static thread_local bool strata_glm_event_copy = false;
extern "C" void strata_glm_sync_event_copy(bool enabled) {strata_glm_event_copy=enabled;}
static thread_local strata_glm_plan_hook strata_glm_plan_callback = nullptr;
static thread_local strata_glm_finish_hook strata_glm_finish_callback = nullptr;
static thread_local strata_glm_compute_hook strata_glm_compute_callback = nullptr;
extern "C" void strata_glm_sync_compute_hook(strata_glm_compute_hook compute) {strata_glm_compute_callback=compute;}
static ggml_status strata_glm_graph_compute(ggml_backend_t backend,ggml_cgraph * graph) {
    // GPU-only audit permits host metadata/views; these are not GPU intervals.
    const bool measure=strata_glm_compute_callback && ggml_backend_get_device(backend) &&
        ggml_backend_dev_type(ggml_backend_get_device(backend))==GGML_BACKEND_DEVICE_TYPE_GPU;
    if (measure) strata_glm_compute_callback(strata_glm_copy_owner,backend,true);
    auto result=ggml_backend_graph_compute_async(backend,graph);
    if (measure) strata_glm_compute_callback(strata_glm_copy_owner,backend,false);
    return result;
}
extern "C" void strata_glm_sync_plan_hooks(strata_glm_plan_hook plan, strata_glm_finish_hook finish) {
    strata_glm_plan_callback=plan; strata_glm_finish_callback=finish;
}
struct strata_glm_plan_scope {
    bool complete=false;
    void finish() {
        if (strata_glm_finish_callback) strata_glm_finish_callback(strata_glm_copy_owner,false,&strata_glm_sync_counters);
        complete=true;
    }
    ~strata_glm_plan_scope() {
        if (!complete && strata_glm_finish_callback) {
            try {strata_glm_finish_callback(strata_glm_copy_owner,true,&strata_glm_sync_counters);} catch (...) {}
        }
    }
};
extern "C" void strata_glm_sync_copy_hook(strata_glm_copy_hook hook, void * owner) {
    strata_glm_copy_callback = hook; strata_glm_copy_owner = owner;
}
static constexpr size_t strata_glm_staging_limit = 16 * 1024 * 1024;
using strata_glm_clock = std::chrono::steady_clock;
static double strata_glm_elapsed(strata_glm_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(strata_glm_clock::now()-start).count();
}
extern "C" void strata_glm_sync_enable(bool enabled) { strata_glm_sync_active = enabled; }
extern "C" void strata_glm_sync_candidate_copy(bool enabled) { strata_glm_candidate_copy = enabled; }
extern "C" void strata_glm_sync_reset() {
    strata_glm_sync_counters = {};
    if (strata_glm_staging) strata_glm_sync_counters.staging_bytes = ggml_backend_buffer_get_size(strata_glm_staging);
}
extern "C" strata_glm_sync_stats strata_glm_sync_snapshot() { return strata_glm_sync_counters; }
extern "C" void strata_glm_sync_release() {
    if (strata_glm_staging) ggml_backend_buffer_free(strata_glm_staging);
    strata_glm_staging = nullptr;
    strata_glm_sync_active = false;
}
static void strata_glm_copy_range(ggml_backend_t backend, ggml_tensor * dst,
                                  const ggml_tensor * src, size_t offset, size_t bytes, size_t count) {
    GGML_ASSERT(offset <= ggml_nbytes(src) && bytes <= ggml_nbytes(src)-offset);
    GGML_ASSERT(offset <= ggml_nbytes(dst) && bytes <= ggml_nbytes(dst)-offset);
    if (strata_glm_copy_callback) {
        strata_glm_copy_callback(strata_glm_copy_owner, backend, dst, src, offset, bytes, count, &strata_glm_sync_counters);
        return;
    }
    if (!strata_glm_staging) {
        auto * type = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(backend));
        GGML_ASSERT(type && "GLM synchronous baseline requires pinned host staging");
        strata_glm_staging = ggml_backend_buft_alloc_buffer(type, strata_glm_staging_limit);
        // CUDA host allocation can silently fall back to pageable CPU storage.
        // Reject that fallback instead of reporting it as pinned staging.
        GGML_ASSERT(strata_glm_staging && ggml_backend_buffer_is_host(strata_glm_staging) &&
                    ggml_backend_buffer_get_type(strata_glm_staging) == type);
        strata_glm_sync_counters.staging_bytes = ggml_backend_buffer_get_size(strata_glm_staging);
    }
    // The previous split may still use this scratch tensor. Complete it before
    // overwriting the tensor, and each transfer before reusing the pinned bytes.
    ggml_backend_synchronize(backend);
    auto * staging = static_cast<uint8_t *>(ggml_backend_buffer_get_base(strata_glm_staging));
    ++strata_glm_sync_counters.ranges;
    for (size_t done = 0; done < bytes;) {
        const size_t chunk = std::min(bytes-done, strata_glm_staging_limit);
        auto start = strata_glm_clock::now();
        memcpy(staging, static_cast<const uint8_t *>(src->data)+offset+done, chunk);
        strata_glm_sync_counters.source_ms += strata_glm_elapsed(start);
        start = strata_glm_clock::now();
        ggml_backend_tensor_set_async(backend, dst, staging, offset+done, chunk);
        ggml_backend_synchronize(backend);
        strata_glm_sync_counters.h2d_ms += strata_glm_elapsed(start);
        strata_glm_sync_counters.source_bytes += chunk;
        strata_glm_sync_counters.h2d_bytes += chunk;
        ++strata_glm_sync_counters.chunks;
        done += chunk;
    }
}
