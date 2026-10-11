#pragma once
// Included only in a COST_EXPERIMENT build. Record on the launch stream and
// drain at the scheduler boundary, without synchronizing between kernels.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

class StrataAttentionStageProfile {
    struct Entry {
        cudaEvent_t begin{}, end{};
        std::string node;
        const char *stage = nullptr;
        int queries = 0, kv = 0, bound = 0, blocks = 0;
    };
    struct Sink {
        FILE *out = nullptr;
        std::mutex mutex;
        Sink() {
            const char *path = std::getenv("STRATA_DS4_ATTENTION_STAGES");
            if (!path || !*path) return;
            const char *disable = std::getenv("GGML_CUDA_DISABLE_GRAPHS");
            if (!disable || std::strcmp(disable, "1")) GGML_ABORT("attention profiling requires disabled CUDA graphs");
            out = std::fopen(path, "w");
            if (!out) GGML_ABORT("cannot open attention stage profile");
            std::fprintf(out, "phase\tnode\tqueries\tkv\tbound\tblocks\tstage\tstream_ms\n");
        }
        ~Sink() { if (out) std::fclose(out); }
    };
    static Sink &sink() { static Sink value; return value; }
    std::vector<Entry> pool;
    size_t used = 0;
public:
    ~StrataAttentionStageProfile() {
        for (auto &e : pool) { cudaEventDestroy(e.begin); cudaEventDestroy(e.end); }
    }
    size_t begin(const ggml_tensor *dst, cudaStream_t stream, const char *stage, int blocks) {
        if (!sink().out) return SIZE_MAX;
        if (used >= 16384) GGML_ABORT("attention stage profile bound exceeded");
        if (used == pool.size()) {
            Entry e; CUDA_CHECK(cudaEventCreate(&e.begin)); CUDA_CHECK(cudaEventCreate(&e.end));
            pool.push_back(e);
        }
        auto &e = pool[used];
        // dst may be a stack-local chunk view. Copy metadata before it expires.
        e.node = dst->name; e.stage = stage; e.queries = int(dst->src[0]->ne[1]);
        e.kv = int(dst->src[1]->ne[1]); e.bound = ggml_get_op_params_i32(dst, 4); e.blocks = blocks;
        CUDA_CHECK(cudaEventRecord(e.begin, stream));
        return used++;
    }
    void end(size_t id, cudaStream_t stream) {
        if (id != SIZE_MAX) CUDA_CHECK(cudaEventRecord(pool[id].end, stream));
    }
    void drain() {
        if (!used) return;
        auto &writer = sink(); std::lock_guard<std::mutex> lock(writer.mutex);
        const char *phase = std::getenv("STRATA_DS4_COST_PHASE");
        for (size_t i = 0; i < used; ++i) {
            const auto &e = pool[i]; float elapsed = 0;
            CUDA_CHECK(cudaEventSynchronize(e.end));
            CUDA_CHECK(cudaEventElapsedTime(&elapsed, e.begin, e.end));
            std::fprintf(writer.out, "%s\t%s\t%d\t%d\t%d\t%d\t%s\t%.6f\n",
                phase ? phase : "init", e.node.c_str(), e.queries, e.kv, e.bound, e.blocks, e.stage, elapsed);
        }
        if (std::fflush(writer.out)) GGML_ABORT("cannot write attention stage profile");
        used = 0;
    }
};
// One TLS pool shared between the private attention TU and scheduler TU.
inline StrataAttentionStageProfile &strata_attention_stage_profile() {
    static thread_local StrataAttentionStageProfile value;
    return value;
}
