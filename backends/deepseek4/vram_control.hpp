#pragma once
#include "expert_transfer.h"
#include <chrono>
#include <deque>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>

// Reader only enqueues. All CUDA work and stdout writes stay on the inference
// thread at graph boundaries, so control replies cannot corrupt token lines.
class VramControl {
    StrataExpertControl control;
    int device;
    std::mutex mutex;
    std::deque<std::string> pending;
    StrataVramPolicy policy;
    std::string applied;
    std::chrono::steady_clock::time_point last{};
public:
    VramControl(StrataExpertControl fn,int gpu):control(fn),device(gpu) {}
    bool configured() const { return !applied.empty(); }
    void enqueue(const std::string &line) {
        std::lock_guard<std::mutex> lock(mutex);
        // The server permits one settings update in flight. Bound untrusted
        // pipe input as well, retaining the most recent desired policy.
        if(pending.size()>=16)pending.pop_front();
        pending.push_back(line);
    }
    void poll() {
        if(!control)return;
        std::deque<std::string> commands;
        {std::lock_guard<std::mutex> lock(mutex);commands.swap(pending);}
        auto now=std::chrono::steady_clock::now();
        if(commands.empty() && now-last<std::chrono::seconds(1))return;
        last=now;
        StrataVramStatus status;
        for(auto &line:commands) {
            std::istringstream in(line);std::string verb,id,extra;
            StrataVramPolicy next;
            if(!(in>>verb>>id>>next.mode>>next.matrices>>next.target_mib>>next.reserve_mib) || in>>extra ||
               id.empty() || id.size()>64 || id.find_first_not_of("0123456789abcdef")!=std::string::npos ||
               !next.valid())continue;
            if(control(device,&next,&status)) {policy=next;applied=id;}
        }
        control(device,nullptr,&status);
        std::cout<<"VRAM_STATUS {\"applied\":\""<<applied<<"\",\"mode\":"<<policy.mode
            <<",\"matrices\":"<<policy.matrices<<",\"target_mib\":"<<policy.target_mib<<",\"reserve_mib\":"<<policy.reserve_mib
            <<",\"cache_bytes\":"<<status.cache_bytes<<",\"cached_matrices\":"<<status.matrices
            <<",\"limit_bytes\":"<<status.limit_bytes<<",\"free_bytes\":"<<status.free_bytes<<",\"total_bytes\":"<<status.total_bytes
            <<",\"allocation_failures\":"<<status.allocation_failures
            <<",\"cache_reserved_bytes\":"<<status.cache_reserved_bytes
            <<",\"slab_blocks\":"<<status.slab_blocks<<",\"slab_allocations\":"<<status.slab_allocations
            <<",\"enabled\":"<<(status.enabled?"true":"false")
            <<",\"telemetry_ok\":"<<(status.telemetry_ok?"true":"false")
            <<",\"target_unreachable\":"<<(status.target_unreachable?"true":"false")<<"}\n"<<std::flush;
    }
};
