#pragma once
#include "runtime.hpp"
#include "nlohmann/json.hpp"

namespace strata_glm {
// One model/owner thread; selected expert bytes still finish before compute.
// Percentages refer to GLOBAL physical usage, including other processes.
class RuntimeMemory {
    struct Impl;
    std::unique_ptr<Impl> impl;
public:
    RuntimeMemory(Model model, const std::string & path, int ram_percent, int vram_percent, bool pipeline=false, int chunk_mib=4, size_t mtp_cache_mib=512);
    ~RuntimeMemory();
    RuntimeMemory(const RuntimeMemory &)=delete;
    RuntimeMemory & operator=(const RuntimeMemory &)=delete;
    void warm();
    void refresh();
    // Optional expert-placement metadata only, after a completed request.
    void checkpoint();
    nlohmann::json snapshot() const;
};
}
