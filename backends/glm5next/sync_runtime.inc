// Included at file scope in a generated copy of ggml-backend.cpp only.
#include "sync_runtime.h"
#include <chrono>

static thread_local bool strata_glm_sync_active = false;
static thread_local bool strata_glm_candidate_copy = false;
static thread_local strata_glm_sync_stats strata_glm_sync_counters = {};
static thread_local ggml_backend_buffer_t strata_glm_staging = nullptr;
static thread_local strata_glm_copy_hook strata_glm_copy_callback = nullptr;
static thread_local void * strata_glm_copy_owner = nullptr;
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
                                  const ggml_tensor * src, size_t offset, size_t bytes) {
    GGML_ASSERT(offset <= ggml_nbytes(src) && bytes <= ggml_nbytes(src)-offset);
    GGML_ASSERT(offset <= ggml_nbytes(dst) && bytes <= ggml_nbytes(dst)-offset);
    if (strata_glm_copy_callback) {
        strata_glm_copy_callback(strata_glm_copy_owner, backend, dst, src, offset, bytes, &strata_glm_sync_counters);
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
