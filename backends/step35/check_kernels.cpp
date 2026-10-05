// Step P0.4: direct CUDA execution with CPU and double-accumulating oracles.
// Synthetic packed weights only. No Strata scheduler or CPU fallback is used.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

using json = nlohmann::ordered_json;
using Floats = std::vector<float>;
static void require(bool ok, const char * message) { if (!ok) throw std::runtime_error(message); }
static constexpr int K = 512, M = 64, EXPERTS = 288, USED = 8;

struct Random {
    uint32_t state;
    float next(float scale) {
        state ^= state << 13; state ^= state >> 17; state ^= state << 5;
        return (float(state & 0xffffu) / 32767.5f - 1) * scale;
    }
};

struct Weights {
    ggml_type type;
    std::vector<uint8_t> packed;
    Floats dequant;
    size_t row_bytes;
    explicit Weights(ggml_type t) : type(t), row_bytes(ggml_row_size(t, K)) {
        Random rng{0x37a001u};
        Floats source(K * M * EXPERTS), importance(K, 1.0f);
        for (auto & x : source) x = rng.next(0.06f);
        packed.resize(row_bytes * M * EXPERTS);
        require(ggml_quantize_chunk(type, source.data(), packed.data(), 0, M * EXPERTS, K, importance.data()) == packed.size(),
                "packed weight size mismatch");
        dequant.resize(source.size());
        if (type == GGML_TYPE_F32) std::memcpy(dequant.data(), packed.data(), packed.size());
        else {
            const auto * traits = ggml_get_type_traits(type);
            require(traits->to_float != nullptr, "missing scalar dequantization reference");
            for (int row = 0; row < M * EXPERTS; ++row)
                traits->to_float(packed.data() + row * row_bytes, dequant.data() + row * K, K);
        }
    }
};

struct Graph {
    ggml_context_ptr ctx{ggml_init({ggml_tensor_overhead() * 64 + ggml_graph_overhead(), nullptr, true})};
    ggml_backend_buffer_ptr buffer;
    ggml_cgraph * graph;
    ggml_backend_t backend;
    explicit Graph(ggml_backend_t b) : backend(b) {
        require(bool(ctx), "context allocation failed");
        graph = ggml_new_graph(ctx.get());
    }
    void allocate(ggml_tensor * output) {
        ggml_build_forward_expand(graph, output);
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            auto * node = ggml_graph_node(graph, i);
            if (!ggml_backend_supports_op(backend, node))
                throw std::runtime_error(std::string(ggml_backend_name(backend)) + " unsupported " + ggml_op_desc(node));
        }
        buffer.reset(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
        require(bool(buffer), "backend allocation failed");
        ggml_backend_buffer_clear(buffer.get(), 0);
    }
    void execute() {
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "backend compute failed");
    }
    template<class T> std::vector<T> read(ggml_tensor * tensor) {
        std::vector<T> out(ggml_nelements(tensor));
        require(ggml_is_contiguous(tensor) && ggml_nbytes(tensor) == out.size() * sizeof(T), "expected contiguous output");
        ggml_backend_tensor_get(tensor, out.data(), 0, ggml_nbytes(tensor));
        return out;
    }
};

static Floats inputs(int tokens, int lanes) {
    Random rng{0x37b001u};
    Floats out(K * tokens * lanes);
    for (auto & x : out) x = rng.next(0.5f);
    return out;
}
static std::vector<int32_t> routes(int tokens) {
    std::vector<int32_t> ids(tokens * USED);
    // Include both boundary experts and repeated experts across tokens, but
    // never duplicate a lane within one token's top-8 selection.
    const int base[USED] = {0, 287, 31, 32, 143, 144, 255, 256};
    for (int t = 0; t < tokens; ++t) for (int lane = 0; lane < USED; ++lane)
        ids[t * USED + lane] = (base[lane] + (t % 3) * 7) % EXPERTS;
    return ids;
}

struct MatResult { Floats values; bool inputs_unchanged; };
static MatResult matmul(ggml_backend_t backend, const Weights & w, int tokens, int mode, bool padded,
                        const Floats & x, const std::vector<int32_t> & ids) {
    Graph g(backend);
    auto * ctx = g.ctx.get();
    const int experts = mode == 0 ? 1 : EXPERTS, lanes = mode == 2 ? USED : 1;
    const int weight_rows = M + (padded ? 4 : 0), input_rows = lanes + (padded ? 2 : 0);
    const int id_rows = USED + (padded ? 3 : 0);
    const size_t weight_offset = padded ? w.row_bytes : 0;
    const size_t input_offset = padded ? K * sizeof(float) : 0;
    const size_t id_offset = padded ? sizeof(int32_t) : 0;
    auto * wa = ggml_new_tensor_3d(ctx, w.type, K, weight_rows, experts);
    auto * xb = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, input_rows, tokens);
    auto * ri = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, id_rows, tokens);
    auto * a = ggml_view_3d(ctx, wa, K, M, experts, wa->nb[1], wa->nb[2], weight_offset);
    auto * b = mode == 0 ? ggml_view_2d(ctx, xb, K, tokens, xb->nb[2], input_offset)
                        : ggml_view_3d(ctx, xb, K, lanes, tokens, xb->nb[1], xb->nb[2], input_offset);
    auto * selected = ggml_view_2d(ctx, ri, USED, tokens, ri->nb[1], id_offset);
    auto * out = mode == 0 ? ggml_mul_mat(ctx, a, b) : ggml_mul_mat_id(ctx, a, b, selected);
    g.allocate(out);
    std::vector<uint8_t> weights(ggml_nbytes(wa), 0xbd);
    Floats input(ggml_nelements(xb), 123.0f);
    std::vector<int32_t> routing(ggml_nelements(ri), EXPERTS - 1);
    for (int expert = 0; expert < experts; ++expert)
        std::memcpy(weights.data() + expert * wa->nb[2] + weight_offset,
                    w.packed.data() + expert * M * w.row_bytes, M * w.row_bytes);
    for (int t = 0; t < tokens; ++t) {
        std::memcpy(input.data() + t * K * input_rows + input_offset / sizeof(float), x.data() + t * K * lanes,
                    K * lanes * sizeof(float));
        std::memcpy(routing.data() + t * id_rows + id_offset / sizeof(int32_t), ids.data() + t * USED, USED * sizeof(int32_t));
    }
    ggml_backend_tensor_set(wa, weights.data(), 0, weights.size());
    ggml_backend_tensor_set(xb, input.data(), 0, ggml_nbytes(xb));
    ggml_backend_tensor_set(ri, routing.data(), 0, ggml_nbytes(ri));
    g.execute();
    std::vector<uint8_t> after(weights.size());
    ggml_backend_tensor_get(wa, after.data(), 0, after.size());
    const auto input_after = g.read<float>(xb);
    return {g.read<float>(out), after == weights && input_after == input && g.read<int32_t>(ri) == routing};
}

static Floats scalar(const Weights & w, int tokens, int mode, const Floats & x, const std::vector<int32_t> & ids) {
    const int selected = mode ? USED : 1, lanes = mode == 2 ? USED : 1;
    Floats out(M * selected * tokens);
    for (int t = 0; t < tokens; ++t) for (int lane = 0; lane < selected; ++lane) for (int row = 0; row < M; ++row) {
        const int expert = mode ? ids[t * USED + lane] : 0;
        const auto * weight = w.dequant.data() + (expert * M + row) * K;
        const auto * input = x.data() + (t * lanes + (lanes == 1 ? 0 : lane)) * K;
        double sum = 0;
        for (int i = 0; i < K; ++i) sum += double(weight[i]) * input[i];
        out[(t * selected + lane) * M + row] = float(sum);
    }
    return out;
}

static json compare(const Floats & actual, const Floats & ref, double nmse_limit, double absolute_limit) {
    require(actual.size() == ref.size() && !actual.empty(), "comparison shape mismatch");
    double sq = 0, energy = 0, max_abs = 0;
    bool finite = true;
    for (size_t i = 0; i < ref.size(); ++i) {
        finite &= std::isfinite(actual[i]) && std::isfinite(ref[i]);
        const double delta = double(actual[i]) - ref[i];
        sq += delta * delta; energy += double(ref[i]) * ref[i]; max_abs = std::max(max_abs, std::abs(delta));
    }
    const auto nmse = sq / std::max(energy, 1e-30);
    return {{"pass", finite && nmse <= nmse_limit && max_abs <= absolute_limit}, {"finite", finite},
            {"nmse", nmse}, {"max_abs", max_abs}, {"nmse_limit", nmse_limit}, {"absolute_limit", absolute_limit}};
}

static json router(ggml_backend_t backend, int tokens, bool padded) {
    Graph g(backend);
    auto * ctx = g.ctx.get();
    const int rows = EXPERTS + (padded ? 16 : 0);
    auto * storage = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rows, tokens);
    auto * logits = ggml_view_2d(ctx, storage, EXPERTS, tokens, storage->nb[1], 0);
    // CUDA SIGMOID requires contiguous rows in this candidate. Pack the view
    // explicitly on the same GPU; never route the unary op to a CPU fallback.
    if (padded) logits = ggml_cont(ctx, logits);
    auto * bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, EXPERTS);
    auto * probs = ggml_sigmoid(ctx, logits);
    auto * selection = ggml_add(ctx, probs, bias);
    // Same sorted top-k op as the pinned Step graph, not the unordered top_k.
    auto * selected = ggml_argsort_top_k(ctx, selection, USED);
    auto * picked = ggml_get_rows(ctx, ggml_reshape_3d(ctx, probs, 1, EXPERTS, tokens), selected);
    auto * weights = ggml_reshape_2d(ctx, picked, USED, tokens);
    weights = ggml_scale(ctx, ggml_div(ctx, weights, ggml_sum_rows(ctx, weights)), 3.0f);
    auto * ids_out = ggml_cont(ctx, selected);
    ggml_build_forward_expand(g.graph, ids_out);
    g.allocate(weights);
    Random rng{0x37c001u};
    Floats x(rows * tokens, 123.0f), correction(EXPERTS);
    for (int e = 0; e < EXPERTS; ++e) correction[e] = rng.next(0.2f);
    for (int t = 0; t < tokens; ++t) for (int e = 0; e < EXPERTS; ++e) x[t * rows + e] = rng.next(3.0f);
    ggml_backend_tensor_set(storage, x.data(), 0, ggml_nbytes(storage));
    ggml_backend_tensor_set(bias, correction.data(), 0, ggml_nbytes(bias));
    g.execute();
    auto actual_ids = g.read<int32_t>(ids_out);
    auto actual_weights = g.read<float>(weights);
    std::vector<int32_t> reference_ids;
    Floats reference_weights;
    double margin = 1;
    for (int t = 0; t < tokens; ++t) {
        std::vector<int32_t> order(EXPERTS); std::iota(order.begin(), order.end(), 0);
        std::vector<double> probabilities(EXPERTS), scores(EXPERTS);
        for (int e = 0; e < EXPERTS; ++e) {
            probabilities[e] = 1.0 / (1.0 + std::exp(-double(x[t * rows + e])));
            scores[e] = probabilities[e] + correction[e];
        }
        std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return scores[a] > scores[b]; });
        for (int i = 0; i < USED; ++i) margin = std::min(margin, scores[order[i]] - scores[order[i + 1]]);
        double total = 0;
        for (int i = 0; i < USED; ++i) total += probabilities[order[i]];
        for (int i = 0; i < USED; ++i) {
            reference_ids.push_back(order[i]); reference_weights.push_back(float(3 * probabilities[order[i]] / total));
        }
    }
    require(margin > 1e-6, "router fixture has numerically ambiguous top-8 ties");
    auto err = compare(actual_weights, reference_weights, 1e-10, 2e-6);
    const bool guards = g.read<float>(storage) == x && g.read<float>(bias) == correction;
    const bool ok = actual_ids == reference_ids && err["pass"].get<bool>() && guards;
    return {{"kind", "sigmoid+bias+sorted_top8+unbiased_norm*3"}, {"tokens", tokens}, {"padded", padded},
            {"pass", ok}, {"ids_exact", actual_ids == reference_ids}, {"inputs_unchanged", guards},
            {"selection_margin_min", margin}, {"weights_vs_scalar", err}};
}

int main(int argc, char ** argv) {
    json report = {{"schema_version", 1}, {"status", "error"},
        {"scope", "synthetic direct CUDA matrix/router operations; no full Step graph, model weights, pipeline or inference"},
        {"requested_revision", STRATA_STEP_SOURCE_SHA}, {"archive_sha256", STRATA_STEP_ARCHIVE_SHA256},
        {"patch_set", STRATA_STEP_PATCH_SET}, {"NVIDIA_TF32_OVERRIDE", "0"},
        {"shape", {{"input", K}, {"output", M}, {"experts", EXPERTS}, {"selected", USED}}},
        {"weight_seed", 0x37a001u}, {"input_seed", 0x37b001u}, {"router_seed", 0x37c001u},
        {"tolerance_policy", "declared before execution: F32 NMSE<=1e-10 and abs<=5e-5; quant NMSE<=1e-4 and abs<=0.02; router exact IDs, NMSE<=1e-10 abs<=2e-6"},
        {"results", json::array()}};
    std::string output;
    try {
        require(argc == 1 || (argc == 3 && std::string(argv[1]) == "--output"), "usage: strata-step35-kernels-check [--output report.json]");
        if (argc == 3) {
            output = argv[2];
            require(output.size() >= 5 && output.substr(output.size() - 5) == ".json", "report path must end in .json");
        }
        const char * tf32 = std::getenv("NVIDIA_TF32_OVERRIDE");
        require(tf32 && std::string(tf32) == "0", "set NVIDIA_TF32_OVERRIDE=0 before starting this check");
        ggml_backend_load_all();
        auto * device = ggml_backend_dev_by_name("CUDA0");
        require(device && ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_GPU, "CUDA0 required");
        ggml_backend_ptr gpu(ggml_backend_dev_init(device, nullptr)), cpu(ggml_backend_cpu_init());
        require(bool(gpu) && bool(cpu), "backend initialization failed");
        ggml_backend_cpu_set_n_threads(cpu.get(), 4);
        report["device"] = ggml_backend_dev_description(device);
        report["cuda_backend"] = ggml_backend_name(gpu.get());
        report["cpu_role"] = "explicit test oracle only; CUDA graph executed directly without a fallback scheduler";
        bool pass = true;
        for (auto type : {GGML_TYPE_F32, GGML_TYPE_Q8_0, GGML_TYPE_Q6_K, GGML_TYPE_Q4_K}) {
            Weights w(type);
            for (int n : {1, 4, 17}) for (int mode : {0, 1, 2}) {
                auto x = inputs(n, mode == 2 ? USED : 1);
                auto ids = routes(n);
                auto reference = scalar(w, n, mode, x, ids);
                Floats compact;
                for (bool padded : {false, true}) {
                    auto actual = matmul(gpu.get(), w, n, mode, padded, x, ids);
                    auto control = matmul(cpu.get(), w, n, mode, padded, x, ids);
                    const double nmse = type == GGML_TYPE_F32 ? 1e-10 : 1e-4;
                    const double absolute = type == GGML_TYPE_F32 ? 5e-5 : 0.02;
                    auto gpu_error = compare(actual.values, reference, nmse, absolute);
                    auto cpu_error = compare(control.values, reference, nmse, absolute);
                    auto cross_error = compare(actual.values, control.values, nmse, absolute);
                    if (!padded) compact = actual.values;
                    const bool exact = compact.size() == actual.values.size() &&
                        std::memcmp(compact.data(), actual.values.data(), compact.size() * sizeof(float)) == 0;
                    const bool ok = gpu_error["pass"].get<bool>() && cpu_error["pass"].get<bool>() && cross_error["pass"].get<bool>() &&
                                    actual.inputs_unchanged && control.inputs_unchanged && exact;
                    pass &= ok;
                    report["results"].push_back({{"kind", mode == 0 ? "MUL_MAT" : mode == 1 ? "MUL_MAT_ID broadcast" : "MUL_MAT_ID per-route"},
                        {"type", ggml_type_name(type)}, {"tokens", n}, {"padded", padded}, {"pass", ok},
                        {"inputs_unchanged", actual.inputs_unchanged && control.inputs_unchanged}, {"padded_vs_compact_exact", exact},
                        {"cuda_vs_scalar", gpu_error}, {"cpu_vs_scalar", cpu_error}, {"cuda_vs_cpu", cross_error}});
                }
            }
        }
        for (int n : {1, 4, 17}) for (bool padded : {false, true}) {
            auto result = router(gpu.get(), n, padded);
            pass &= result["pass"].get<bool>();
            report["results"].push_back(result);
        }
        report["case_count"] = report["results"].size();
        report["status"] = pass ? "pass" : "fail";
    } catch (const std::exception & e) { report["error"] = e.what(); }
    ggml_quantize_free();
    if (!output.empty()) {
        std::ofstream file(output);
        file << report.dump(2) << '\n';
        if (!file) { std::cerr << "cannot write kernel report\n"; return 1; }
        std::cout << report["status"] << ": " << report["results"].size() << " cases; " << output << '\n';
    } else std::cout << report.dump(2) << '\n';
    return report["status"] == "pass" ? 0 : 1;
}
