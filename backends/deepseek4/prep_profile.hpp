#pragma once
#include <chrono>
#include <cstdint>
#include <cstdlib>

namespace strata_ds4 {
// Host wall time, opt-in. Nested intervals and worker intervals overlap;
// they must not be added to estimate request latency. No CUDA events/fences.
inline bool prep_profile_enabled() {
    static const bool enabled=[] {const char *s=std::getenv("STRATA_DS4_PREP_PROFILE");return s && std::atoi(s)!=0;}();
    return enabled;
}
inline uint64_t prep_now_ns() {
    return prep_profile_enabled()?std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count():0;
}
struct PrepTimer {
    using Clock=std::chrono::steady_clock;
    uint64_t *destination;
    const bool enabled=prep_profile_enabled();
    Clock::time_point start=enabled?Clock::now():Clock::time_point{};
    explicit PrepTimer(uint64_t *value):destination(value) {}
    ~PrepTimer() {if(enabled && destination)*destination+=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-start).count();}
};
}
