#pragma once
// Diagnostic build only. Event spans are stream time, including launch gaps;
// CPU wait/read sums can overlap one another and GPU work.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

class StrataExpertTransportProfile {
    struct Sink {
        FILE *file=nullptr; std::mutex mutex;
        Sink() {
            const char *path=std::getenv("STRATA_DS4_TRANSPORT_PROFILE");if(!path || !*path)return;
            const char *disabled=std::getenv("GGML_CUDA_DISABLE_GRAPHS");
            if(!disabled || std::strcmp(disabled,"1"))GGML_ABORT("transport events require disabled CUDA graphs");
            file=std::fopen(path,"w");if(!file)GGML_ABORT("cannot open transport profile");
            std::fprintf(file,"phase\towner\tkind\tbytes\tms\n");
        }
        ~Sink(){if(file)std::fclose(file);}
    };
    static Sink &sink(){static Sink s;return s;}
    struct Span {
        cudaEvent_t begin{},end{};std::string phase;const char *kind=nullptr;
        size_t bytes=0;bool active=false,closed=false;
    };
    std::mutex mutex;
    std::vector<Span> pool;
    size_t pending_h2d=SIZE_MAX,pending_delivery=SIZE_MAX;
    static std::string phase(){const char *p=std::getenv("STRATA_DS4_COST_PHASE");return p?p:"init";}
public:
    bool enabled() const {return sink().file!=nullptr;}
    ~StrataExpertTransportProfile(){for(auto &s:pool){cudaEventDestroy(s.begin);cudaEventDestroy(s.end);}}
    size_t begin(cudaStream_t stream,const char *kind,size_t bytes) {
        if(!enabled())return SIZE_MAX;
        std::lock_guard<std::mutex> lock(mutex);size_t id=0;
        while(id<pool.size() && pool[id].active)++id;
        if(id==pool.size()) {
            if(id>=8192)GGML_ABORT("transport event pool limit exceeded");
            Span s;CUDA_CHECK(cudaEventCreate(&s.begin));CUDA_CHECK(cudaEventCreate(&s.end));pool.push_back(s);
        }
        auto &s=pool[id];s.phase=phase();s.kind=kind;s.bytes=bytes;s.active=true;s.closed=false;
        CUDA_CHECK(cudaEventRecord(s.begin,stream));return id;
    }
    void end(size_t id,cudaStream_t stream) {
        if(id==SIZE_MAX)return;
        std::lock_guard<std::mutex> lock(mutex);auto &s=pool[id];
        CUDA_CHECK(cudaEventRecord(s.end,stream));s.closed=true;
    }
    // Pipeline serializes H2D pairs on its copy mutex; delivery is single-consumer.
    void observe(cudaStream_t stream,bool start,size_t bytes,bool h2d) {
        auto &pending=h2d?pending_h2d:pending_delivery;
        if(start)pending=begin(stream,h2d?"pipeline_h2d":"ring_d2d",bytes);
        else {end(pending,stream);pending=SIZE_MAX;}
    }
    void drain() {
        if(!enabled())return;
        std::lock_guard<std::mutex> lock(mutex);auto &writer=sink();std::lock_guard<std::mutex> output(writer.mutex);
        for(auto &s:pool)if(s.active && s.closed) {
            CUDA_CHECK(cudaEventSynchronize(s.end));float ms=0;CUDA_CHECK(cudaEventElapsedTime(&ms,s.begin,s.end));
            std::fprintf(writer.file,"%s\t%p\t%s\t%zu\t%.6f\n",s.phase.c_str(),(void *)this,s.kind,s.bytes,ms);s.active=false;
        }
        if(std::fflush(writer.file))GGML_ABORT("cannot write transport events");
    }
    void host_delta(const char *kind,uint64_t us) {
        if(!enabled() || !us)return;
        auto &writer=sink();std::lock_guard<std::mutex> output(writer.mutex);auto p=phase();
        std::fprintf(writer.file,"%s\t%p\t%s\t0\t%.6f\n",p.c_str(),(void *)this,kind,double(us)/1000);
    }
};
