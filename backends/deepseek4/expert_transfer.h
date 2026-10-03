#pragma once
#include "ggml-backend.h"
#include <cstdint>

struct StrataExpertCounters {
    uint64_t hits=0, misses=0, bypass=0, evictions=0;
    uint64_t h2d_bytes=0, d2d_bytes=0, stage_bytes=0;
    uint64_t stage_us=0, wait_us=0, submit_us=0;
};
using StrataExpertCopy = void (*)(ggml_backend_t, const ggml_tensor *, ggml_tensor *, int, int, int);
using StrataExpertStats = void (*)(ggml_backend_t, StrataExpertCounters *);
