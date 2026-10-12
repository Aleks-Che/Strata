#pragma once
#include <chrono>
#include <cstdint>
#include <cstdlib>

// Scheduler and CUDA backend are statically linked. One host thread acquires
// a route and plans its CPU branch; never transfer this snapshot to a callback.
namespace strata_ds4 {
struct CpuRouteProfile {
    const void *tensor=nullptr;
    uint64_t get_us=0,wait_us=0;
    bool reused=false;
    static bool enabled() {
        static const bool value=[] {const char *v=std::getenv("STRATA_DS4_CPU_TRACE");return v && *v;}();
        return value;
    }
    static CpuRouteProfile &current(){static thread_local CpuRouteProfile p;return p;}
    static uint64_t now() {
        return std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    static void begin(const void *tensor) {
        if(enabled()){current()={};current().tensor=tensor;}
    }
    struct Timer {
        bool wait;uint64_t start;
        explicit Timer(bool waiting):wait(waiting),start(enabled()?now():0) {}
        ~Timer(){if(enabled())(wait?current().wait_us:current().get_us)+=now()-start;}
    };
};
}
