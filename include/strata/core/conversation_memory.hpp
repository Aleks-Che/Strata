#pragma once

#include <cstdint>
#include <algorithm>
#include <istream>
#include <optional>
#include <limits>

namespace strata::core {

// Host physical headroom. Windows commit is checked separately: the configured
// physical-RAM floor must not also be reserved against the commit limit.
// This is not a reservation or enforcement of container/job memory limits.
// Unknown telemetry is deliberately distinct from a measured zero.
std::optional<uint64_t> conversation_available_memory();
std::optional<uint64_t> conversation_mem_available(std::istream& meminfo);

struct ConversationMemory {
    std::optional<uint64_t> physical;
    std::optional<uint64_t> commit; // Windows; absent on platforms without this limit
};
ConversationMemory conversation_memory_status();
// Recheck after a bounded Windows allocation probe. This lets an OS-configured
// growable page file expand before treating current commit headroom as a hard cap.
// No OS settings are changed and no snapshot data is written to disk.
ConversationMemory conversation_memory_prepare(uint64_t allocation, uint64_t floor);

inline bool conversation_memory_admit(std::optional<uint64_t> available,
                                      uint64_t allocation, uint64_t floor) {
    return available && *available >= floor && allocation <= *available - floor;
}

inline const char* conversation_memory_blocker(const ConversationMemory& memory, uint64_t allocation, uint64_t floor) {
    // Keep 256 MiB of commit headroom (or the explicitly smaller configured
    // floor). Physical RAM retains the full user-configured reserve.
    const uint64_t commit_floor = std::min<uint64_t>(floor, 256ULL * 1024 * 1024);
    if (!memory.physical) return "memory_unknown";
    if (!conversation_memory_admit(memory.physical, allocation, floor)) return "physical_ram";
    if (memory.commit && !conversation_memory_admit(memory.commit, allocation, commit_floor)) return "commit_limit";
    return nullptr;
}

inline bool conversation_snapshot_admit(const ConversationMemory& memory, uint64_t allocation, uint64_t floor) {
    return conversation_memory_blocker(memory, allocation, floor) == nullptr;
}

inline uint64_t conversation_commit_probe_bytes(const ConversationMemory& memory, uint64_t allocation, uint64_t floor) {
    const uint64_t reserve = std::min<uint64_t>(floor, 256ULL * 1024 * 1024);
    if (!conversation_memory_admit(memory.physical, allocation, floor) || !memory.commit ||
        conversation_memory_admit(memory.commit, allocation, reserve) ||
        allocation > std::numeric_limits<uint64_t>::max() - reserve) return 0;
    return allocation + reserve;
}

} // namespace strata::core
