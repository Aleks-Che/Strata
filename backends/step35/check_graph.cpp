// Native Step graph and full/SWA memory validation. All weights are synthetic.
#include "synthetic_step.hpp"
#include "session_snapshot.hpp"
#include "llama.h"
#include "llama-context.h"
#include "llama-model.h"
#include "llama-kv-cache-iswa.h"
#include "ggml-backend.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>

using json = nlohmann::ordered_json;
using Floats = std::vector<float>;
using Model = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
using Context = std::unique_ptr<llama_context, decltype(&llama_free)>;
static void require(bool ok, const std::string & text) { if (!ok) throw std::runtime_error(text); }

static Floats read_f32(ggml_tensor * t) {
    require(t->type == GGML_TYPE_F32, "expected F32 capture");
    std::vector<uint8_t> bytes(ggml_nbytes(t));
    ggml_backend_tensor_get(t, bytes.data(), 0, bytes.size());
    Floats values;
    for (int64_t d = 0; d < t->ne[3]; ++d) for (int64_t c = 0; c < t->ne[2]; ++c)
        for (int64_t b = 0; b < t->ne[1]; ++b) for (int64_t a = 0; a < t->ne[0]; ++a) {
            float value;
            std::memcpy(&value, bytes.data() + a*t->nb[0] + b*t->nb[1] + c*t->nb[2] + d*t->nb[3], sizeof(value));
            values.push_back(value);
        }
    return values;
}

static json compare(const std::string & name, const Floats & actual, const Floats & expected,
                    double abs_limit = 5e-4, double nmse_limit = 1e-7, bool exact = false) {
    require(actual.size() == expected.size() && !actual.empty(), "comparison shape: " + name);
    double error = 0, energy = 0, max_abs = 0;
    bool finite = true;
    for (size_t i = 0; i < actual.size(); ++i) {
        finite &= std::isfinite(actual[i]) && std::isfinite(expected[i]);
        const double delta = double(actual[i]) - expected[i];
        error += delta * delta; energy += double(expected[i]) * expected[i];
        max_abs = std::max(max_abs, std::abs(delta));
    }
    const double nmse = error / std::max(energy, 1e-30);
    const bool bits = std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)) == 0;
    return {{"name", name}, {"pass", finite && max_abs <= abs_limit && nmse <= nmse_limit && (!exact || bits)},
            {"elements", actual.size()}, {"finite", finite}, {"max_abs", max_abs}, {"nmse", nmse},
            {"abs_limit", abs_limit}, {"nmse_limit", nmse_limit}, {"exact_required", exact}, {"bit_exact", bits}};
}

struct Audit {
    llama_context * ctx = nullptr;
    bool gpu = false, capture = false;
    std::set<std::string> cpu_nodes;
    std::map<std::string, size_t> ops;
    std::map<std::string, Floats> values;
    static bool wanted(const std::string & name) {
        for (const std::string prefix : {"Qcur", "Kcur", "attn_gate", "attn_out", "attn_gated",
                                        "ffn_up", "ffn_gate", "ffn_silu", "ffn_swiglu", "ffn_moe_"})
            if (name.find(prefix) == 0) return true;
        return false;
    }
    static bool callback(ggml_tensor * t, bool ask, void * data) {
        auto & a = *static_cast<Audit *>(data);
        if (!ask) {
            a.values[t->name] = read_f32(t);
            return true;
        }
        if (t->op != GGML_OP_NONE && t->op != GGML_OP_VIEW && t->op != GGML_OP_RESHAPE &&
            t->op != GGML_OP_TRANSPOSE && t->op != GGML_OP_PERMUTE) {
            ++a.ops[ggml_op_desc(t)];
            auto * backend = ggml_backend_sched_get_tensor_backend(a.ctx->get_sched(), t);
            auto * device = backend ? ggml_backend_get_device(backend) : nullptr;
            if (a.gpu && (!device || ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_GPU))
                a.cpu_nodes.insert(std::string(t->name) + " / " + ggml_op_desc(t));
        }
        return a.capture && t->type == GGML_TYPE_F32 && wanted(t->name);
    }
};

static Model load(const std::string & path, bool gpu) {
    auto * dev = ggml_backend_dev_by_name(gpu ? "CUDA0" : "CPU");
    require(dev != nullptr, "missing backend device");
    ggml_backend_dev_t devices[] = {dev, nullptr};
    llama_model_tensor_buft_override overrides[] = {{"token_embd\\.weight", ggml_backend_dev_buffer_type(dev)}, {nullptr, nullptr}};
    auto mp = llama_model_default_params();
    mp.devices = devices; mp.tensor_buft_overrides = overrides;
    mp.n_gpu_layers = gpu ? -1 : 0; mp.load_mtp = false;
    mp.use_extra_bufts = false; mp.load_mode = LLAMA_LOAD_MODE_NONE;
    Model model(llama_model_load_from_file(path.c_str(), mp), llama_model_free);
    require(bool(model), "Step fixture load failed");
    return model;
}

struct Run {
    Audit audit;
    Context ctx{nullptr, llama_free};
    Run(llama_model * model, bool gpu, int ubatch, bool full_swa = false) {
        audit.gpu = gpu;
        auto cp = llama_context_default_params();
        cp.n_ctx = 2048; cp.n_batch = 1024; cp.n_ubatch = ubatch; cp.n_seq_max = 1;
        cp.n_threads = cp.n_threads_batch = 4; cp.type_k = cp.type_v = GGML_TYPE_F32;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.offload_kqv = gpu; cp.op_offload = gpu; cp.swa_full = full_swa;
        cp.cb_eval = Audit::callback; cp.cb_eval_user_data = &audit;
        ctx.reset(llama_init_from_model(model, cp));
        require(bool(ctx), "Step context creation failed");
        audit.ctx = ctx.get();
        require(memory() != nullptr, "Step must use separate full/SWA KV caches");
    }
    llama_kv_cache_iswa * memory() { return dynamic_cast<llama_kv_cache_iswa *>(ctx->get_memory()); }
    void clear() { llama_memory_clear(ctx->get_memory(), true); }
    Floats decode(int first, int count, int chunk, int salt = 0) {
        Floats out;
        for (int pos = first; pos < first + count; pos += chunk) {
            const int n = std::min(chunk, first + count - pos);
            auto batch = llama_batch_init(n, 0, 1);
            batch.n_tokens = n;
            for (int i = 0; i < n; ++i) {
                batch.token[i] = (7 * (pos + i) + 11 + salt) % 64; batch.pos[i] = pos + i;
                batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0; batch.logits[i] = 1;
            }
            const int status = llama_decode(ctx.get(), batch);
            llama_batch_free(batch);
            require(status == 0, "Step decode failed at " + std::to_string(pos) + ": " + std::to_string(status));
            for (int i = 0; i < n; ++i) {
                const float * logits = llama_get_logits_ith(ctx.get(), i);
                require(logits != nullptr, "missing logits"); out.insert(out.end(), logits, logits + 64);
            }
            require(audit.cpu_nodes.empty(), "CPU tensor math in GPU graph: " + json(audit.cpu_nodes).dump());
        }
        return out;
    }
    step35::SessionSnapshot save() {
        return step35::SessionSnapshot::capture(ctx.get());
    }
    void restore(const step35::SessionSnapshot & saved) {
        saved.restore(ctx.get());
    }
    json positions() {
        return {{"base_min", memory()->get_base()->seq_pos_min(0)}, {"base_max", memory()->get_base()->seq_pos_max(0)},
                {"swa_min", memory()->get_swa()->seq_pos_min(0)}, {"swa_max", memory()->get_swa()->seq_pos_max(0)},
                {"base_cells", memory()->get_base()->get_size()}, {"swa_cells", memory()->get_swa()->get_size()}};
    }
};

static void components(Run & run, llama_model & model, json & results, int position) {
    const size_t first_result = results.size();
    run.audit.values.clear(); run.audit.capture = true;
    run.decode(position, 1, 1);
    run.audit.capture = false;
    const auto & values = run.audit.values;
    auto get = [&](const std::string & name) -> const Floats & {
        auto it = values.find(name);
        require(it != values.end(), "missing captured graph node: " + name);
        return it->second;
    };
    for (int layer = 0; layer < 3; ++layer) {
        const auto suffix = "-" + std::to_string(layer);
        const int heads = layer == 1 ? 3 : 2, rotary = layer == 1 ? 128 : 64;
        for (const std::string kind : {"Q", "K"}) {
            const auto & raw = get(kind + "cur" + suffix);
            const auto & normed = get(kind + "cur_normed" + suffix);
            const auto norm = read_f32(kind == "Q" ? model.layers[layer].attn_q_norm : model.layers[layer].attn_k_norm);
            Floats expected(raw.size());
            for (size_t offset = 0; offset < raw.size(); offset += 128) {
                double sq = 0; for (int i = 0; i < 128; ++i) sq += double(raw[offset+i]) * raw[offset+i];
                const double factor = 1.0 / std::sqrt(sq / 128 + 1e-5);
                for (int i = 0; i < 128; ++i) expected[offset+i] = float(raw[offset+i] * factor * norm[i]);
            }
            results.push_back(compare("scalar_" + kind + "_norm" + suffix, normed, expected, 2e-5, 1e-10));
            expected = normed;
            const double base = layer == 1 ? 10000 : 5000000;
            for (size_t offset = 0; offset < normed.size(); offset += 128) for (int pair = 0; pair < rotary / 2; ++pair) {
                // NEOX pairing; full attention alone consumes the stored factors.
                const double factor = layer == 1 ? 1.0 : double(1 + float(pair) * 0.007f);
                const double angle = position * std::pow(base, -2.0 * pair / rotary) / factor;
                const double a = normed[offset+pair], b = normed[offset+pair+rotary/2];
                expected[offset+pair] = float(a*std::cos(angle) - b*std::sin(angle));
                expected[offset+pair+rotary/2] = float(a*std::sin(angle) + b*std::cos(angle));
            }
            results.push_back(compare("scalar_" + kind + "_rope_pos" + std::to_string(position) + suffix,
                                      get(kind + "cur_pos" + suffix), expected, 5e-4, 1e-7));
        }
        const auto & raw_gate = get("attn_gate" + suffix);
        const auto & sigmoid = get("attn_gate_sigmoid" + suffix);
        Floats expected(raw_gate.size());
        for (size_t i = 0; i < expected.size(); ++i) expected[i] = float(1.0 / (1.0 + std::exp(-double(raw_gate[i]))));
        results.push_back(compare("scalar_head_sigmoid" + suffix, sigmoid, expected, 2e-6, 1e-10));
        const auto & attention = get("attn_out" + suffix);
        require(attention.size() == size_t(heads * 128), "wrong per-head attention layout");
        expected = attention;
        for (size_t i = 0; i < expected.size(); ++i) expected[i] *= sigmoid[i / 128];
        results.push_back(compare("scalar_head_gate" + suffix, get("attn_gated" + suffix), expected, 2e-6, 1e-10));
    }
    for (const std::string prefix : {"ffn_", "ffn_moe_"}) {
        const float limit = prefix == "ffn_" ? 16.0f : 7.0f;
        const auto & silu = get(prefix + "silu-2");
        const auto & clipped = get(prefix + "silu_clamped-2");
        const auto & up = get(prefix + "up_clamped-2");
        Floats expected(silu.size());
        size_t gate_hits = 0, up_hits = 0;
        for (size_t i = 0; i < silu.size(); ++i) {
            expected[i] = std::min(silu[i], limit);
            gate_hits += silu[i] > limit;
            up_hits += std::abs(up[i]) == limit;
            require(std::isfinite(up[i]) && std::abs(up[i]) <= limit, "up clamp escaped limit");
        }
        require(gate_hits > 0 && up_hits > 0, "fixture failed to exercise clamps");
        auto check = compare("scalar_" + prefix + "post_silu_clamp", clipped, expected, 0, 0, true);
        check["gate_clipped_elements"] = gate_hits; check["up_at_limit_elements"] = up_hits;
        results.push_back(check);
        for (size_t i = 0; i < expected.size(); ++i) expected[i] *= up[i];
        results.push_back(compare("scalar_" + prefix + "limited_swiglu", get(prefix + "swiglu_limited-2"), expected, 2e-5, 1e-10));
    }
    for (size_t i = first_result; i < results.size(); ++i)
        results[i]["name"] = results[i]["name"].get<std::string>() + "/position=" + std::to_string(position);
}

int main(int argc, char ** argv) {
    json report = {{"schema_version", 1}, {"status", "error"}, {"requested_revision", STRATA_STEP_SOURCE_SHA},
        {"archive_sha256", STRATA_STEP_ARCHIVE_SHA256}, {"patch_set", STRATA_STEP_PATCH_SET},
        {"scope", "synthetic native Step decoder graph and full/SWA state; not the 106 GiB model, pipeline or throughput"},
        {"fixture", {{"seed", 0x37d001u}, {"layers", 3}, {"width", 256}, {"heads", {2,3,2}}, {"kv_heads", 1},
                     {"head_dimension", 128}, {"full_rope", 64}, {"swa_rope", 128}, {"swa_window", 512},
                     {"experts", 16}, {"top_k", 8}, {"vocab", 64}, {"weight_type", "F32"}}},
        {"configuration", {{"context", 2048}, {"batch", 1024}, {"main_ubatch", 256}, {"split_ubatch", 32},
                           {"kv_type", "F32"}, {"flash_attention", "off"}, {"NVIDIA_TF32_OVERRIDE", "0"}}},
        {"tolerances", {{"logits_max_abs", 5e-4}, {"logits_nmse", 1e-7}, {"restored_same_decode", "bit-exact"}}},
        {"results", json::array()}};
    std::filesystem::path base;
    // ordered_json stores object members in a vector. A reference to
    // report["results"] would be invalidated when adding later root fields.
    json results = json::array();
    try {
        require(argc == 2, "usage: strata-step35-graph-check WORK_DIRECTORY");
        base = argv[1]; std::filesystem::create_directories(base);
        const char * tf32 = std::getenv("NVIDIA_TF32_OVERRIDE");
        require(tf32 && std::string(tf32) == "0", "set NVIDIA_TF32_OVERRIDE=0");
        const auto fixture = (base / "synthetic-step.gguf").string();
        write_synthetic_step(fixture);
        ggml_backend_load_all();
        auto * device = ggml_backend_dev_by_name("CUDA0");
        require(device && ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_GPU, "CUDA0 required");
        report["device"] = ggml_backend_dev_description(device);
        auto gpu_model = load(fixture, true), cpu_model = load(fixture, false);
        for (int layer = 0; layer < 3; ++layer) {
            const auto & hp = gpu_model->hparams;
            require(hp.n_rot(layer) == (layer == 1 ? 128u : 64u) && hp.is_swa(layer) == (layer == 1), "loaded RoPE/SWA metadata mismatch");
        }
        Run reference(gpu_model.get(), true, 256, true), ring(gpu_model.get(), true, 32),
            cpu(cpu_model.get(), false, 64, true), serial(gpu_model.get(), true, 32);
        const auto full = reference.decode(0, 1024, 1024);
        const auto cpu_logits = cpu.decode(0, 1024, 128);
        results.push_back(compare("gpu_full_prefill_vs_cpu", full, cpu_logits));
        const auto split = ring.decode(0, 1024, 127);
        results.push_back(compare("swa_ring_split_prefill_vs_full_cache", split, full));
        results.push_back(compare("serial_decode_vs_full_prefill", serial.decode(0, 1024, 1), full));
        for (int pos : {510, 511, 512, 1023})
            results.push_back(compare("boundary_last_position_" + std::to_string(pos),
                Floats(split.begin()+pos*64, split.begin()+(pos+1)*64), Floats(full.begin()+pos*64, full.begin()+(pos+1)*64)));
        report["ring_after_1024"] = ring.positions();
        require(ring.memory()->get_swa()->get_size() < ring.memory()->get_base()->get_size(), "SWA fixture did not use bounded cache");
        require(ring.memory()->get_swa()->seq_pos_min(0) > 0, "SWA fixture did not evict old positions");

        const auto saved_a = ring.save();
        report["snapshot_bytes_after_1024"] = saved_a.size();
        const auto continuation = ring.decode(1024, 24, 1);
        results.push_back(compare("append_after_ring_wrap", continuation, reference.decode(1024, 24, 1)));
        ring.clear(); ring.decode(0, 160, 32, 19); // unrelated conversation B
        ring.clear(); ring.restore(saved_a);
        results.push_back(compare("A_B_A_restore", ring.decode(1024, 24, 1), continuation, 0, 0, true));

        // Rollback stays within retained SWA history. Arbitrary old rollback
        // after eviction requires a saved snapshot, tested separately above.
        require(llama_memory_seq_rm(ring.ctx->get_memory(), 0, 1040, -1), "short rollback rejected");
        results.push_back(compare("short_rollback_replay", ring.decode(1040, 8, 1),
                                  Floats(continuation.end()-8*64, continuation.end()), 0, 0, true));
        ring.clear();
        results.push_back(compare("clear_and_replay_prefix", ring.decode(0, 128, 32), Floats(full.begin(), full.begin()+128*64)));

        for (int prefix : {0, 511, 512, 513, 767, 768, 769, 1530, 1536}) {
            ring.clear();
            if (prefix) ring.decode(0, prefix, 127);
            const auto snapshot = ring.save();
            const auto expected = ring.decode(prefix, 24, 1);
            // Restore must replace existing state without a caller-side clear.
            ring.restore(snapshot);
            results.push_back(compare("restore_prefix_" + std::to_string(prefix),
                ring.decode(prefix, 24, 1), expected, 0, 0, true));
            // An exact rollback uses a checkpoint taken before the tentative
            // suffix. Native tail removal can change allocation after ring wrap.
            ring.restore(snapshot);
            ring.decode(prefix, 4, 1);
            const auto accepted = ring.save();
            ring.decode(prefix + 4, 20, 1, 23);
            ring.restore(accepted);
            results.push_back(compare("checkpoint_rollback_prefix_" + std::to_string(prefix),
                ring.decode(prefix + 4, 20, 1), Floats(expected.begin()+4*64, expected.end()), 0, 0, true));
        }
        // Restoring the older checkpoint also recovers history after eviction.
        ring.restore(saved_a);
        results.push_back(compare("checkpoint_rollback_across_eviction", ring.decode(1024, 24, 1), continuation, 0, 0, true));
        bool rejected = false;
        try { ring.save().restore(serial.ctx.get()); }
        catch (const std::runtime_error &) { rejected = true; }
        require(rejected, "cross-context snapshot accepted");
        results.push_back({{"name", "reject_foreign_context_snapshot"}, {"pass", true}});

        Run captured(gpu_model.get(), true, 32);
        captured.decode(0, 511, 127);
        components(captured, *gpu_model, results, 511);
        components(captured, *gpu_model, results, 512);
        report["captured_node_names"] = json::array();
        for (const auto & item : captured.audit.values) report["captured_node_names"].push_back(item.first);
        report["gpu_compute_ops"] = reference.audit.ops;
        report["gpu_cpu_nodes"] = reference.audit.cpu_nodes;
        report["ring_final"] = ring.positions();
        require(results.size() == 75, "unexpected graph case count");
        bool pass = true; for (const auto & result : results) pass &= result["pass"].get<bool>();
        report["status"] = pass ? "pass" : "fail";
    } catch (const std::exception & error) { report["error"] = error.what(); }
    report["results"] = std::move(results);
    report["case_count"] = report["results"].size();
    if (!base.empty()) {
        std::ofstream file(base / "step-graph-report.json");
        file << report.dump(2) << '\n';
        if (!file) { std::cerr << "cannot write graph report\n"; return 1; }
        std::cout << report["status"] << ": " << report["results"].size() << " cases; " << (base / "step-graph-report.json").string() << '\n';
    } else std::cout << report.dump(2) << '\n';
    return report["status"] == "pass" ? 0 : 1;
}
