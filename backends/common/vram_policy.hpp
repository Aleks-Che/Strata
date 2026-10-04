#pragma once
#include <algorithm>
#include <cstdint>

// Pure arithmetic, shared by device controllers and GPU-free tests.
// The target is total device usage, including other processes and fixed weights.
struct StrataVramPolicy {
    int mode = 0; // 0: configured budgets, 1: matrix count, 2: device usage
    uint64_t matrices = 0, target_mib = 0, reserve_mib = 1024;
    bool valid() const {
        return mode >= 0 && mode <= 2 && matrices <= 1000000 && target_mib <= 1048576 &&
            reserve_mib >= 128 && reserve_mib <= 1048576 && (mode != 2 || target_mib > 0);
    }
    uint64_t byte_limit(uint64_t free, uint64_t total, uint64_t cached) const {
        free = std::min(free, total);
        const auto used = total - free;
        const auto fixed = used - std::min(used, cached);
        const auto reserve = reserve_mib << 20;
        auto ceiling = total - std::min(total, reserve);
        if (mode == 2) ceiling = std::min(ceiling, target_mib << 20);
        return ceiling - std::min(ceiling, fixed);
    }
};
