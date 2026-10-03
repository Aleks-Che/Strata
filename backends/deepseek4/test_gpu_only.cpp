#include "ggml.h"
#include "ggml-backend.h"
#include <cmath>
#include <cstdio>

// A CPU graph must fail before writing its output. The same graph must compute
// correctly on a GPU. This tests the patched scheduler, not just CLI flags.
static bool check(ggml_backend_t cpu, ggml_backend_t gpu, bool force_cpu) {
    auto *ctx = ggml_init({ggml_tensor_overhead()*16 + ggml_graph_overhead_custom(16, false), nullptr, true});
    auto *input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_set_input(input);
    auto *output = ggml_scale(ctx, input, 2.0f);
    ggml_set_name(output, "gpu_only_contract");
    ggml_set_output(output);
    auto *graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph, output);
    ggml_backend_t backends[] = {gpu ? gpu : cpu, cpu};
    auto *sched = ggml_backend_sched_new(backends, nullptr, gpu ? 2 : 1, 16, false, true);
    ggml_backend_sched_set_tensor_backend(sched, output, force_cpu ? cpu : gpu);
    bool ok = ggml_backend_sched_alloc_graph(sched, graph);
    if (ok) {
        float values[] = {1, 2, 3, 4}, sentinel[] = {-99, -99, -99, -99}, result[4];
        ggml_backend_tensor_set(input, values, 0, sizeof(values));
        ggml_backend_tensor_set(output, sentinel, 0, sizeof(sentinel));
        auto status = ggml_backend_sched_graph_compute(sched, graph);
        ggml_backend_tensor_get(output, result, 0, sizeof(result));
        ok = status == (force_cpu ? GGML_STATUS_FAILED : GGML_STATUS_SUCCESS);
        for (int i=0; i<4; ++i) ok &= std::fabs(result[i] - (force_cpu ? -99.0f : 2*values[i])) < 0.001f;
    }
    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    return ok;
}
int main() {
    ggml_backend_load_all();
    auto *cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    auto *gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!cpu) return 1;
    bool ok = check(cpu, gpu, true);
    if (gpu) ok &= check(cpu, gpu, false);
    else std::puts("GPU unavailable: positive GPU case skipped");
    if (gpu) ggml_backend_free(gpu);
    ggml_backend_free(cpu);
    std::puts(ok ? "GPU-only scheduler contract passed" : "GPU-only scheduler contract FAILED");
    return ok ? 0 : 1;
}
