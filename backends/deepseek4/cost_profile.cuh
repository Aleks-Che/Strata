#pragma once
// Diagnostic build only. Event spans include stream stalls and host launch gaps;
// they are not kernel-only durations. Drain once per scheduler subgraph, retaining
// fusion and stream assignment, but deliberately disallow CUDA graph capture.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>
#include "attention_stage_profile.cuh"

class StrataCostProfile {
    struct Entry { cudaEvent_t begin{},end{}; const ggml_tensor *node=nullptr; int fused=1; };
    std::vector<Entry> pool;
    size_t used=0;
    struct Sink {
        FILE *out=nullptr;
        std::mutex mutex;
        uint64_t graph=0;
        Sink() {
            const char *path=std::getenv("STRATA_DS4_COST_PROFILE");
            if(!path || !*path)return;
            const char *disable=std::getenv("GGML_CUDA_DISABLE_GRAPHS");
            if(!disable || std::strcmp(disable,"1"))GGML_ABORT("cost profiling requires GGML_CUDA_DISABLE_GRAPHS=1");
            out=std::fopen(path,"w");
            if(!out)GGML_ABORT("cannot open cost profile");
            std::fprintf(out,"graph\tphase\top\tnode\tweight\ttype\tk\tm\tn\tfused\tstream_ms\n");
        }
        ~Sink(){if(out)std::fclose(out);}
    };
    // Reserve and execution may use different host threads. Open once per
    // process; each thread owns its events but shares a serialized output sink.
    static Sink &sink(){static Sink s;return s;}
    FILE *out=nullptr;
public:
    StrataCostProfile():out(sink().out){}
    ~StrataCostProfile() {
        for(auto &e:pool){cudaEventDestroy(e.begin);cudaEventDestroy(e.end);}
    }
    size_t begin(const ggml_tensor *node,cudaStream_t stream,bool capture) {
        if(!out)return SIZE_MAX;
        if(capture)GGML_ABORT("cost profiling cannot capture CUDA graphs");
        if(used>=32768)GGML_ABORT("cost profile node bound exceeded");
        if(used==pool.size()) {
            Entry e;CUDA_CHECK(cudaEventCreate(&e.begin));CUDA_CHECK(cudaEventCreate(&e.end));pool.push_back(e);
        }
        auto &e=pool[used];e.node=node;e.fused=1;
        CUDA_CHECK(cudaEventRecord(e.begin,stream));return used++;
    }
    void end(size_t index,cudaStream_t stream,int fused) {
        if(index==SIZE_MAX)return;
        pool[index].fused=fused;CUDA_CHECK(cudaEventRecord(pool[index].end,stream));
    }
    void drain() {
        strata_attention_stage_profile().drain();
        if(!used)return;
        auto &writer=sink();std::lock_guard<std::mutex> lock(writer.mutex);
        // Streams may overlap: sum is aggregate event time, not elapsed time.
        const char *phase=std::getenv("STRATA_DS4_COST_PHASE");
        for(size_t i=0;i<used;++i) {
            auto &e=pool[i];CUDA_CHECK(cudaEventSynchronize(e.end));float elapsed=0;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed,e.begin,e.end));
            const auto *s=e.node->src[0];
            std::fprintf(out,"%llu\t%s\t%s\t%s\t%s\t%s\t%lld\t%lld\t%lld\t%d\t%.6f\n",
                (unsigned long long)writer.graph,phase?phase:"init",ggml_op_name(e.node->op),e.node->name,
                s?s->name:"-",s?ggml_type_name(s->type):"-",(long long)(s?s->ne[0]:0),
                (long long)e.node->ne[0],(long long)e.node->ne[1],e.fused,elapsed);
        }
        if(std::fflush(out))GGML_ABORT("cannot write cost profile");
        used=0;++writer.graph;
    }
};
static thread_local StrataCostProfile strata_cost_profile;
