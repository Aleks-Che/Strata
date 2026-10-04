// Deterministic numerical candidate fixtures. CPU is an explicit test oracle;
// the CUDA side executes directly on its backend, without a fallback scheduler.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

using json = nlohmann::ordered_json;
using Floats = std::vector<float>;
static void require(bool ok, const char * what) { if (!ok) throw std::runtime_error(what); }

struct Random {
    uint32_t state = 0x53f1a5u;
    float next(float lo, float hi) {
        state ^= state << 13; state ^= state >> 17; state ^= state << 5;
        return lo + (hi - lo) * float(state & 0xffffu) / 65535.0f;
    }
    Floats values(size_t n, float lo, float hi) {
        Floats v(n); for (auto & x : v) x = next(lo, hi); return v;
    }
};

struct Graph {
    ggml_context_ptr ctx{ggml_init({ggml_tensor_overhead()*64 + ggml_graph_overhead(), nullptr, true})};
    ggml_backend_buffer_ptr buffer;
    ggml_cgraph * gf = nullptr;
    ggml_backend_t backend;
    explicit Graph(ggml_backend_t b) : backend(b) {
        require(bool(ctx), "metadata allocation failed");
        gf = ggml_new_graph(ctx.get());
    }
    ggml_tensor * f32(int64_t x, int64_t y = 1, int64_t z = 1) {
        return ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, x, y, z);
    }
    void allocate(ggml_tensor * out) {
        ggml_build_forward_expand(gf, out);
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            auto * node = ggml_graph_node(gf, i);
            if (!ggml_backend_supports_op(backend, node))
                throw std::runtime_error(std::string(ggml_backend_name(backend)) + " unsupported " + ggml_op_desc(node));
        }
        buffer.reset(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
        require(bool(buffer), "tensor allocation failed");
        ggml_backend_buffer_clear(buffer.get(), 0);
    }
    template<class T> void set(ggml_tensor * t, const std::vector<T> & data) {
        require(ggml_nbytes(t) == data.size()*sizeof(T), "fixture input size mismatch");
        ggml_backend_tensor_set(t, data.data(), 0, ggml_nbytes(t));
    }
    template<class T = float> std::vector<T> execute(ggml_tensor * out) {
        require(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS, "graph execution failed");
        std::vector<T> values(ggml_nelements(out));
        require(ggml_nbytes(out) == values.size()*sizeof(T), "fixture output size mismatch");
        ggml_backend_tensor_get(out, values.data(), 0, ggml_nbytes(out));
        return values;
    }
};

static json compare(const std::string & name, const Floats & gpu, const Floats & ref) {
    require(gpu.size() == ref.size() && !gpu.empty(), "comparison size mismatch");
    double sqerr = 0, energy = 0, max_abs = 0;
    bool finite = true;
    for (size_t i = 0; i < gpu.size(); ++i) {
        finite = finite && std::isfinite(gpu[i]) && std::isfinite(ref[i]);
        double delta = double(gpu[i]) - ref[i];
        sqerr += delta*delta; energy += double(ref[i])*ref[i];
        max_abs = std::max(max_abs, std::abs(delta));
    }
    const double nmse = sqerr / std::max(energy, 1e-30);
    return {{"name", name}, {"status", finite && nmse <= 1e-7 && max_abs <= 5e-4 ? "pass" : "fail"},
            {"elements", gpu.size()}, {"finite", finite}, {"nmse", nmse}, {"max_abs", max_abs},
            {"nmse_limit", 1e-7}, {"max_abs_limit", 5e-4}};
}

struct Kda {
    static constexpr int d = 128;
    int heads, tokens;
    Floats q, k, v, g, beta, state;
    Kda(int h, int n) : heads(h), tokens(n) {
        Random rng;
        const size_t size = size_t(d)*h*n;
        q = rng.values(size, -1, 1); k = rng.values(size, -1, 1);
        v = rng.values(size, -0.3f, 1.3f); g = rng.values(size, -5, -1e-4f);
        beta = rng.values(size_t(h)*n, 0, 1);
        state = rng.values(size_t(d)*d*h, -0.02f, 0.02f);
        for (auto * values : {&q, &k}) {
            for (size_t row = 0; row < size; row += d) {
                double norm = 0; for (int i = 0; i < d; ++i) norm += double((*values)[row+i])*(*values)[row+i];
                for (int i = 0; i < d; ++i) (*values)[row+i] /= float(std::sqrt(norm));
            }
        }
    }
    size_t stride() const { return size_t(d)*heads; }
    size_t state_size() const { return size_t(d)*d*heads; }
};

static Floats run_kda(ggml_backend_t backend, const Kda & in, int start, int count,
                      const Floats & state, int snapshots = 1) {
    require(start >= 0 && count > 0 && start+count <= in.tokens && snapshots <= count, "bad KDA slice");
    Graph graph(backend);
    auto * q = graph.f32(Kda::d, in.heads, count);
    auto * k = graph.f32(Kda::d, in.heads, count);
    auto * v = graph.f32(Kda::d, in.heads, count);
    auto * g = graph.f32(Kda::d, in.heads, count);
    auto * beta = graph.f32(1, in.heads, count);
    auto * s = graph.f32(Kda::d, Kda::d, in.heads);
    auto * out = ggml_gated_delta_net(graph.ctx.get(), q, k, v, g, beta, s, snapshots);
    graph.allocate(out);
    auto slice = [start, count](const Floats & src, size_t stride) {
        return Floats(src.begin()+start*stride, src.begin()+(start+count)*stride);
    };
    graph.set(q, slice(in.q, in.stride())); graph.set(k, slice(in.k, in.stride()));
    graph.set(v, slice(in.v, in.stride())); graph.set(g, slice(in.g, in.stride()));
    graph.set(beta, slice(in.beta, in.heads)); graph.set(s, state);
    return graph.execute(out);
}

// Double-precision scalar recurrence, independent of both backend kernels.
// M[value,key] holds state; decay is along keys, then delta update and q readout.
static Floats scalar_kda(const Kda & in, int snapshots) {
    std::vector<double> state(in.state.begin(), in.state.end());
    Floats out(in.stride()*in.tokens + in.state_size()*snapshots);
    for (int t = 0; t < in.tokens; ++t) {
        for (int h = 0; h < in.heads; ++h) {
            const size_t off = (size_t(t)*in.heads+h)*Kda::d;
            for (int j = 0; j < Kda::d; ++j) {
                const size_t row = (size_t(h)*Kda::d+j)*Kda::d;
                double kv = 0;
                for (int i = 0; i < Kda::d; ++i) {
                    state[row+i] *= std::exp(double(in.g[off+i]));
                    kv += state[row+i]*in.k[off+i];
                }
                const double delta = (in.v[off+j]-kv)*in.beta[size_t(t)*in.heads+h];
                double value = 0;
                for (int i = 0; i < Kda::d; ++i) {
                    state[row+i] += delta*in.k[off+i];
                    value += state[row+i]*in.q[off+i];
                }
                out[off+j] = float(value/std::sqrt(double(Kda::d)));
            }
        }
        const int slot = in.tokens-1-t;
        if (slot < snapshots)
            std::copy(state.begin(), state.end(), out.begin()+in.stride()*in.tokens+slot*in.state_size());
    }
    return out;
}

static Floats final_state(const Floats & values, const Kda & in, int count, int slot = 0) {
    const auto first = values.begin()+in.stride()*count+slot*in.state_size();
    return Floats(first, first+in.state_size());
}

static void kda_checks(ggml_backend_t gpu, ggml_backend_t cpu, json & results) {
    // Actual head width/count, including both sides of the CUDA 16-token chunk.
    for (int n : {1, 4, 15, 16, 17, 33, 64}) {
        Kda in(64, n);
        const int snapshots = std::min(n, 4);
        results.push_back(compare("kda/cpu/tokens="+std::to_string(n)+"/snapshots="+std::to_string(snapshots),
            run_kda(gpu, in, 0, n, in.state, snapshots), run_kda(cpu, in, 0, n, in.state, snapshots)));
    }
    Kda small(2, 17);
    const auto scalar = scalar_kda(small, 4);
    results.push_back(compare("kda/cuda-vs-scalar", run_kda(gpu, small, 0, 17, small.state, 4), scalar));
    results.push_back(compare("kda/cpu-vs-scalar", run_kda(cpu, small, 0, 17, small.state, 4), scalar));

    Kda in(64, 33);
    const auto full = run_kda(gpu, in, 0, 33, in.state);
    for (int chunk : {1, 4, 16}) {
        Floats stitched, state = in.state;
        for (int start = 0; start < 33; start += chunk) {
            const int n = std::min(chunk, 33-start);
            auto part = run_kda(gpu, in, start, n, state);
            stitched.insert(stitched.end(), part.begin(), part.begin()+n*in.stride());
            state = final_state(part, in, n);
        }
        stitched.insert(stitched.end(), state.begin(), state.end());
        results.push_back(compare("kda/microbatch="+std::to_string(chunk), stitched, full));
    }
    // Save state after 8-prefix + 4 proposed tokens. Reject 0..4 proposals,
    // replace the rejected suffix, and compare with a clean accepted prefix.
    Kda round(8, 16);
    const auto prefix = run_kda(gpu, round, 0, 8, round.state);
    const auto prefix_state = final_state(prefix, round, 8);
    const auto draft = run_kda(gpu, round, 8, 4, prefix_state, 4);
    for (int accepted = 0; accepted <= 4; ++accepted) {
        const auto restored = accepted == 0 ? prefix_state : final_state(draft, round, 4, 4-accepted);
        Kda changed = round;
        for (size_t i = (8+accepted)*round.stride(); i < changed.v.size(); ++i) changed.v[i] = -changed.v[i];
        const auto continuation = run_kda(gpu, changed, 8+accepted, 16-8-accepted, restored);
        const auto clean = run_kda(gpu, changed, 0, 16, changed.state);
        Floats expected(clean.begin()+(8+accepted)*round.stride(), clean.end());
        results.push_back(compare("kda/rollback-accepted="+std::to_string(accepted), continuation, expected));
    }
}

static Floats mhc(ggml_backend_t backend, int tokens) {
    constexpr int width = 4096, hc = 4;
    Random rng;
    Graph graph(backend);
    auto * x = graph.f32(width, hc, tokens);
    auto * pre = graph.f32(hc, tokens);
    auto * post = graph.f32(hc, tokens);
    auto * mixes = graph.f32(24, tokens);
    auto * scale = graph.f32(3);
    auto * base = graph.f32(24);
    auto * comb = ggml_dsv4_hc_comb(graph.ctx.get(), mixes, scale, base, 1e-6f, 20);
    auto * reduced = ggml_dsv4_hc_pre(graph.ctx.get(), x, pre);
    auto * out = ggml_dsv4_hc_post(graph.ctx.get(), reduced, x, post, comb);
    graph.allocate(out);
    graph.set(x, rng.values(size_t(width)*hc*tokens, -1, 1));
    graph.set(pre, rng.values(hc*tokens, 0.01f, 1));
    graph.set(post, rng.values(hc*tokens, 0.01f, 2));
    graph.set(mixes, rng.values(24*tokens, -2, 2));
    graph.set(scale, Floats{0.7f, 0.8f, 0.6f}); graph.set(base, rng.values(24, -0.5f, 0.5f));
    auto values = graph.execute(out);
    // Include intermediate Sinkhorn and reduction values, not only final output.
    for (auto * t : {comb, reduced}) {
        const auto offset = values.size(); values.resize(offset+ggml_nelements(t));
        ggml_backend_tensor_get(t, values.data()+offset, 0, ggml_nbytes(t));
    }
    return values;
}

static json integer_gather(ggml_backend_t backend) {
    Graph graph(backend);
    auto * cells = ggml_new_tensor_2d(graph.ctx.get(), GGML_TYPE_I32, 4, 1024);
    auto * ids = ggml_new_tensor_1d(graph.ctx.get(), GGML_TYPE_I32, 512);
    auto * out = ggml_get_rows(graph.ctx.get(), cells, ids);
    graph.allocate(out);
    std::vector<int32_t> values(4096), selection(512), expected;
    // Odd values above 2^24 detect accidental I32 -> F32 -> I32 conversion.
    for (int i = 0; i < 4096; ++i) values[i] = 16777217 + 2*i;
    values[0] = -1; values[4095] = 2147483647;
    for (int i = 0; i < 512; ++i) {
        selection[i] = (i*137) % 1024;
        if (i == 511) selection[i] = 1023;
        for (int j = 0; j < 4; ++j) expected.push_back(values[4*selection[i]+j]);
    }
    graph.set(cells, values); graph.set(ids, selection);
    const auto actual = graph.execute<int32_t>(out);
    return {{"name", "indexer/integer-pool-cell-gather"}, {"status", actual == expected ? "pass" : "fail"},
            {"elements", actual.size()}, {"comparison", "exact I32; includes -1, odd values >2^24, INT32_MAX"}};
}

int main() {
    json report = {{"schema_version", 1}, {"status", "error"},
        {"scope", "synthetic KDA/mHC/integer-gather kernels; not full GLM inference or hybrid-state rollback"},
        {"requested_revision", STRATA_GLM_SOURCE_SHA}, {"archive_sha256", STRATA_GLM_ARCHIVE_SHA256},
        {"seed", 0x53f1a5u}, {"results", json::array()}};
    try {
        const auto * tf32 = std::getenv("NVIDIA_TF32_OVERRIDE");
        require(tf32 && std::string(tf32) == "0", "set NVIDIA_TF32_OVERRIDE=0");
        ggml_backend_load_all();
        auto * device = ggml_backend_dev_by_name("CUDA0");
        require(device && ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_GPU, "CUDA0 required");
        ggml_backend_ptr gpu(ggml_backend_dev_init(device, nullptr));
        ggml_backend_ptr cpu(ggml_backend_cpu_init());
        require(bool(gpu) && bool(cpu), "backend initialization failed");
        ggml_backend_cpu_set_n_threads(cpu.get(), 4);
        report["device"] = ggml_backend_dev_description(device);
        report["NVIDIA_TF32_OVERRIDE"] = tf32;
        kda_checks(gpu.get(), cpu.get(), report["results"]);
        for (int n : {1, 4, 17, 256})
            report["results"].push_back(compare("mhc/width=4096/streams=4/sinkhorn=20/tokens="+std::to_string(n),
                mhc(gpu.get(), n), mhc(cpu.get(), n)));
        report["results"].push_back(integer_gather(gpu.get()));
        bool pass = true;
        for (const auto & r : report["results"]) pass = pass && r["status"] == "pass";
        report["status"] = pass ? "pass" : "fail";
    } catch (const std::exception & e) { report["error"] = e.what(); }
    std::cout << report.dump(2) << '\n';
    return report["status"] == "pass" ? 0 : 1;
}
