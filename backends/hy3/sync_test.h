#pragma once
#include <cstddef>
#include <limits>
#include <atomic>

// Test-process-only controls: no CLI/environment entry point. Caps can only
// reduce observed availability, never bypass the real global memory guard.
struct strata_hy3_test_limits {
    size_t ram_available = std::numeric_limits<size_t>::max();
    size_t gpu_available = std::numeric_limits<size_t>::max();
    bool fail_next_pinned_allocation = false;
    bool fail_next_cache_allocation = false;
    bool fail_next_cache_fill = false;
    int fail_pipeline_submission_after = -1;
    int fail_batch_enqueue_after = -1;
    int cancel_batch_enqueue_after = -1;
    std::atomic<bool> * batch_cancel = nullptr;
};
extern "C" void strata_hy3_test_set_limits(strata_hy3_test_limits limits);
