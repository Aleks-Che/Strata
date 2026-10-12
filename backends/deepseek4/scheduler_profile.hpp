#pragma once
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>

// Host wall time only. No CUDA events, extra barriers or changes to graph use.
// In particular, route_wait includes previously queued GPU computation; it is
// not a measurement of synchronization overhead or of D2H bandwidth.
struct StrataSchedulerProfile {
    using Clock = std::chrono::steady_clock;
    enum Kind { route_get, route_wait, plan, copy, graph_submit, destination_wait, ordered_experts, hash_reuses, gpu_fetch, count };
    struct Sample { uint64_t calls=0, bytes=0; double ms=0; };
    struct Output {
        std::mutex mutex;
        FILE *file=nullptr;
        Output() {
            const char *path=std::getenv("STRATA_DS4_SCHED_PROFILE");
            if(path && *path)file=std::fopen(path,"w");
            if(file)std::fputs("phase\tgraph\ttokens\tkind\tcalls\tbytes\tms\n",file);
        }
        ~Output() { if(file)std::fclose(file); }
        uint64_t graph=0;
    };
    static Output &output() { static Output out;return out; }
    FILE *file=output().file;
    Clock::time_point start=file?Clock::now():Clock::time_point{};
    std::array<Sample,count> samples{};
    int64_t tokens=0;
    struct Timer {
        StrataSchedulerProfile &owner;
        Kind kind;
        uint64_t bytes;
        Clock::time_point start;
        Timer(StrataSchedulerProfile &p,Kind k,uint64_t n=0):owner(p),kind(k),bytes(n),
            start(p.file?Clock::now():Clock::time_point{}) {}
        ~Timer() {
            if(!owner.file)return;
            auto &s=owner.samples[kind];++s.calls;s.bytes+=bytes;
            s.ms+=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
        }
    };
    ~StrataSchedulerProfile() {
        if(!file)return;
        const double total=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
        const char *phase=std::getenv("STRATA_DS4_COST_PHASE");
        if(!phase)phase="init";
        std::lock_guard<std::mutex> lock(output().mutex);
        const auto graph=++output().graph;
        const char *names[]={"route_get","route_wait","plan","copy","graph_submit","destination_wait","ordered_experts","hash_reuses","gpu_fetch"};
        for(size_t i=0;i<samples.size();++i) {
            const auto &s=samples[i];
            std::fprintf(file,"%s\t%llu\t%lld\t%s\t%llu\t%llu\t%.6f\n",phase,
                (unsigned long long)graph,(long long)tokens,names[i],
                (unsigned long long)s.calls,(unsigned long long)s.bytes,s.ms);
        }
        std::fprintf(file,"%s\t%llu\t%lld\ttotal\t1\t0\t%.6f\n",phase,
            (unsigned long long)graph,(long long)tokens,total);
        std::fflush(file);
    }
};
