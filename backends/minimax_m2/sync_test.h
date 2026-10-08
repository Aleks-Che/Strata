#pragma once
#include <atomic>
#include <cstdint>
// Fault-injection seam for native tests only. Effective availability is clamped
// to the real reading: this can make admission stricter, never bypass the cap.
struct strata_mm27_test_limits {
    uint64_t ram_available=UINT64_MAX,vram_available=UINT64_MAX;
    bool cache_oom=false,cache_fill_failure=false;
    bool pipeline_read_failure=false,pipeline_copy_failure=false;
    bool pipeline_compute_failure=false;
    int batch_fail_after=-1,batch_cancel_after=-1,batch_pressure_after=-1;
    std::atomic<bool> *batch_cancel=nullptr;
};
void strata_mm27_test_memory_limits(strata_mm27_test_limits limits);
