#include "strata/core/conversation_memory.hpp"

#include <charconv>
#include <algorithm>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace strata::core {

std::optional<uint64_t> conversation_mem_available(std::istream& meminfo) {
    std::optional<uint64_t> result;
    std::string line;
    while (std::getline(meminfo, line)) {
        std::istringstream fields(line);
        std::string key, value, unit, extra;
        if (!(fields >> key) || key != "MemAvailable:") continue;
        if (result || !(fields >> value >> unit) || unit != "kB" || (fields >> extra)) return {};
        uint64_t kb = 0;
        const auto parsed = std::from_chars(value.data(), value.data() + value.size(), kb);
        if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() ||
            kb > std::numeric_limits<uint64_t>::max() / 1024) return {};
        result = kb * 1024;
    }
    if (meminfo.bad() || (meminfo.fail() && !meminfo.eof())) return {};
    return result;
}

ConversationMemory conversation_memory_status() {
#if defined(_WIN32)
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof status;
    if (GlobalMemoryStatusEx(&status)) return {status.ullAvailPhys, status.ullAvailPageFile};
    return {};
#elif defined(__linux__)
    std::ifstream meminfo("/proc/meminfo");
    if (!meminfo) return {};
    return {conversation_mem_available(meminfo), {}};
#else
    return {};
#endif
}

std::optional<uint64_t> conversation_available_memory() {
    return conversation_memory_status().physical;
}

ConversationMemory conversation_memory_prepare(uint64_t allocation, uint64_t floor) {
    auto memory = conversation_memory_status();
#if defined(_WIN32)
    const uint64_t required = conversation_commit_probe_bytes(memory, allocation, floor);
    if (required && required <= std::numeric_limits<SIZE_T>::max()) {
        // GlobalMemoryStatusEx reports the current limit, not a configured future
        // page-file maximum. Ask Windows whether this allocation can be committed
        // before evicting useful snapshots. Untouched pages consume no physical
        // RAM; release the probe before allocating the actual snapshot vectors.
        void* probe = VirtualAlloc(nullptr, (SIZE_T) required, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        if (probe) VirtualFree(probe, 0, MEM_RELEASE);
        memory = conversation_memory_status();
    }
#endif
    return memory;
}

} // namespace strata::core
