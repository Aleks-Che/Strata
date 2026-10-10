#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace strata_ds4 {
constexpr int max_frequency_decay=1000000000;
// Zero preserves the original cache-size-dependent default. Units are recorded
// matrix accesses, not tokens, bytes or milliseconds. Prefill does not record.
inline uint64_t frequency_decay(size_t cache_bytes,int configured) {
    return configured?uint64_t(configured):std::max<uint64_t>(4096,cache_bytes>>17);
}
}
