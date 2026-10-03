#include "strata/core/conversation_memory.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sstream>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

using namespace strata::core;
namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
}

int main() {
    for (const char* text : {"", "MemFree: 100 kB\n", "MemAvailable: -1 kB\n",
                            "MemAvailable: +1 kB\n", "MemAvailable: 1 MB\n",
                            "MemAvailable: 1\n", "MemAvailable: 1x kB\n",
                            "MemAvailable: 18446744073709551615 kB\n",
                            "MemAvailable: 18446744073709551616 kB\n",
                            "MemAvailable: 1 kB trailing\n",
                            "MemAvailable: 1 kB\nMemAvailable: 2 kB\n"}) {
        std::istringstream input(text);
        check(!conversation_mem_available(input), "missing/malformed telemetry is unknown");
    }
    std::istringstream normal("MemTotal: 999999 kB\nMemAvailable:    12345 kB\nSwapFree: 777 kB\n");
    check(conversation_mem_available(normal) == 12345ULL * 1024, "only MemAvailable is counted");
    std::istringstream zero("MemAvailable: 0 kB");
    check(conversation_mem_available(zero) == 0, "zero available memory is known");
    std::istringstream broken("MemAvailable: 123 kB\n");
    broken.setstate(std::ios::badbit);
    check(!conversation_mem_available(broken), "I/O failure declines admission");
    check(!conversation_memory_admit({}, 0, 0), "unknown fails closed even with zero floor");
    check(conversation_memory_admit(100, 40, 60), "exact allocation plus floor fits");
    check(!conversation_memory_admit(99, 40, 60), "one byte below required memory rejected");
    check(!conversation_memory_admit(59, 0, 60), "floor subtraction cannot underflow");
    check(conversation_memory_admit(60, 0, 60), "post-capture floor check");
    check(!conversation_memory_admit(100, std::numeric_limits<uint64_t>::max(), 1), "allocation arithmetic cannot overflow");
    check(conversation_memory_admit(std::numeric_limits<uint64_t>::max(),
                                   std::numeric_limits<uint64_t>::max(), 0), "maximal exact bound");
    check(!conversation_memory_admit(std::numeric_limits<uint64_t>::max(),
                                    std::numeric_limits<uint64_t>::max(), 1), "maximal sum overflow rejected");
    // Test the real provider without assuming any particular amount of free RAM.
    constexpr uint64_t MiB = 1024 * 1024;
    check(conversation_commit_probe_bytes({40000*MiB, 229*MiB}, 0, 4096*MiB) == 256*MiB,
          "post-capture reserve can request commit growth without another snapshot");
    check(conversation_commit_probe_bytes({40000*MiB, 1900*MiB}, 4137*MiB, 4096*MiB) == (4137+256)*MiB,
          "pre-capture probe covers snapshot and reserve");
    check(conversation_commit_probe_bytes({40000*MiB, 5000*MiB}, 4137*MiB, 4096*MiB) == 0,
          "no allocation probe when memory already fits");
    check(conversation_commit_probe_bytes({5000*MiB, 1900*MiB}, 4137*MiB, 4096*MiB) == 0,
          "commit probing cannot override physical memory floor");
    check(conversation_commit_probe_bytes({40000*MiB, {}}, 4137*MiB, 4096*MiB) == 0,
          "platform without a commit limit does not probe");
    check(std::strcmp(conversation_memory_blocker({40*1024*MiB, 1900*MiB}, 4137*MiB, 4096*MiB), "commit_limit") == 0,
          "real long-session failure identifies commit, not physical RAM");
    check(std::strcmp(conversation_memory_blocker({5000*MiB, 40000*MiB}, 4137*MiB, 4096*MiB), "physical_ram") == 0,
          "physical memory failure is distinct");
    check(std::strcmp(conversation_memory_blocker({{}, {}}, 0, 0), "memory_unknown") == 0,
          "unknown telemetry has its own reason");
    check(conversation_memory_blocker({40000*MiB, 6000*MiB}, 4137*MiB, 4096*MiB) == nullptr,
          "long snapshot accepted when both limits allow it");
    check(conversation_snapshot_admit({40*1024*MiB, 3*1024*MiB}, 1536*MiB, 4096*MiB),
          "ample physical RAM and sufficient commit admit a snapshot below the physical reserve");
    check(!conversation_snapshot_admit({40*1024*MiB, 1792*MiB-1}, 1536*MiB, 4096*MiB),
          "commit still needs allocation plus 256 MiB reserve");
    check(conversation_snapshot_admit({40*1024*MiB, 1792*MiB}, 1536*MiB, 4096*MiB),
          "exact commit boundary fits");
    check(!conversation_snapshot_admit({5*1024*MiB, 40*1024*MiB}, 1536*MiB, 4096*MiB),
          "full physical reserve remains enforced");
    check(!conversation_snapshot_admit({{}, 40*1024*MiB}, 0, 0), "unknown physical telemetry fails closed");
    check(conversation_snapshot_admit({8*1024*MiB, {}}, 1536*MiB, 4096*MiB), "platform without commit uses physical headroom");
    check(!conversation_snapshot_admit({8*1024*MiB, 0}, 1, 0), "zero commit differs from unavailable commit");
    check(conversation_snapshot_admit({8*1024*MiB, 128*MiB}, 0, 128*MiB), "smaller explicit reserve honored");
    check(conversation_snapshot_admit({4096*MiB, 256*MiB}, 0, 4096*MiB), "post-capture memory floors checked independently");
    const auto available = conversation_available_memory();
    check(!available || conversation_memory_admit(available, 0, 0), "provider returns bytes or unknown");
#if defined(_WIN32)
    // Unknown must fail closed in production, but must not let a broken Windows provider pass this test.
    // Compare with total physical memory, not a second available reading: other processes can allocate
    // between calls. No pressure allocation or fixed free-memory assumption is needed.
    MEMORYSTATUSEX physical{};
    physical.dwLength = sizeof physical;
    check(GlobalMemoryStatusEx(&physical) != 0, "Windows physical-memory API is available");
    check(available.has_value() && *available <= physical.ullTotalPhys,
          "Windows provider returns a known, physically bounded sample");
#endif
    std::printf("conversation_memory_test: %d checks passed\n", checks);
}
