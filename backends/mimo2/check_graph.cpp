// Native MiMo graph and full/SWA memory validation. All weights are synthetic.
#include "synthetic_mimo2.hpp"
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
        for (const std::string prefix : {"wqkv", "Qcur", "Kcur", "Vcur", "attn_out", "kq",
                                        "ffn_moe_weights"})
            if (name.find(prefix) == 0) return true;
        return false;
    }
    static bool callback(ggml_tensor * t, bool ask, void * data) {
        auto & a = *static_cast<Audit *>(data);
        if (!ask) {
            a.values[t->name] = read_f32(t);
            if (t->op == GGML_OP_SOFT_MAX && std::string(t->name).find("kq_soft_max") == 0) {
                a.values[std::string(t->name)+"/mask"] = read_f32(t->src[1]);
            }
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
    require(bool(model), "MiMo fixture load failed");
    return model;
}

struct Run {
    Audit audit;
    Context ctx{nullptr, llama_free};
    Run(llama_model * model, bool gpu, int ubatch, bool full_swa = false, bool fa = false) {
        audit.gpu = gpu;
        auto cp = llama_context_default_params();
        cp.n_ctx = 1024; cp.n_batch = 512; cp.n_ubatch = ubatch; cp.n_seq_max = 1;
        cp.n_threads = cp.n_threads_batch = 4; cp.type_k = cp.type_v = GGML_TYPE_F32;
        cp.flash_attn_type = fa ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.offload_kqv = gpu; cp.op_offload = gpu; cp.swa_full = full_swa;
        cp.cb_eval = Audit::callback; cp.cb_eval_user_data = &audit;
        ctx.reset(llama_init_from_model(model, cp));
        require(bool(ctx), "MiMo context creation failed");
        audit.ctx = ctx.get();
        require(ctx->get_cparams().flash_attn == fa, "requested FA mode was not honored");
        require(memory() != nullptr, "MiMo must use separate full/SWA KV caches");
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
            require(status == 0, "MiMo decode failed at " + std::to_string(pos) + ": " + std::to_string(status));
            for (int i = 0; i < n; ++i) {
                const float * logits = llama_get_logits_ith(ctx.get(), i);
                require(logits != nullptr, "missing logits"); out.insert(out.end(), logits, logits + 64);
            }
            require(audit.cpu_nodes.empty(), "CPU tensor math in GPU graph: " + json(audit.cpu_nodes).dump());
        }
        return out;
    }
    std::vector<uint8_t> save() {
        std::vector<uint8_t> data(llama_state_get_size(ctx.get()));
        data.resize(llama_state_get_data(ctx.get(), data.data(), data.size()));
        require(!data.empty(), "empty state snapshot"); return data;
    }
    void restore(const std::vector<uint8_t> & data) {
        require(llama_state_set_data(ctx.get(), data.data(), data.size()) == data.size(), "state restore failed");
    }
    json positions() {
        return {{"base_min", memory()->get_base()->seq_pos_min(0)}, {"base_max", memory()->get_base()->seq_pos_max(0)},
                {"swa_min", memory()->get_swa()->seq_pos_min(0)}, {"swa_max", memory()->get_swa()->seq_pos_max(0)},
                {"base_cells", memory()->get_base()->get_size()}, {"swa_cells", memory()->get_swa()->get_size()}};
    }
};

static void components(Run & run, llama_model & model, json & results, int pos) {
    run.audit.values.clear(); run.audit.capture = true;
    run.decode(pos, 1, 1); run.audit.capture = false;
    const auto get = [&](const std::string & name) -> const Floats & {
        auto i = run.audit.values.find(name);
        require(i != run.audit.values.end(), "missing captured MiMo node: "+name);
        return i->second;
    };
    for (int layer = 0; layer < 3; ++layer) {
        const std::string suffix = "-"+std::to_string(layer);
        const std::string at = "/layer="+std::to_string(layer)+"/pos="+std::to_string(pos);
        const int kv = layer == 1 ? 8 : 4;
        const auto & qkv = get("wqkv"+suffix);
        require(qkv.size() == size_t(64*192+kv*(192+128)), "fused QKV width mismatch");
        for (bool key : {false, true}) {
            const int offset = key ? 64*192 : 0, heads = key ? kv : 64;
            Floats expected(qkv.begin()+offset, qkv.begin()+offset+heads*192);
            for (int h = 0; h < heads; ++h) for (int i = 0; i < 32; ++i) {
                const double angle = pos*std::pow(layer == 1 ? 1e4 : 1e7, -2.0*i/64);
                const double a = expected[h*192+i], b = expected[h*192+i+32];
                expected[h*192+i] = float(a*std::cos(angle)-b*std::sin(angle));
                expected[h*192+i+32] = float(a*std::sin(angle)+b*std::cos(angle));
            }
            results.push_back(compare(std::string("fused_")+(key ? "K" : "Q")+"_partial_neox"+at,
                                      get(std::string(key ? "Kcur" : "Qcur")+suffix), expected, 2e-5, 1e-9));
        }
        results.push_back(compare("fused_V_offsets"+at, get("Vcur"+suffix),
                                  Floats(qkv.begin()+(64+kv)*192, qkv.end()), 0, 0, true));
        // Capture KQ before softmax: the allocator may reuse its storage later.
        const auto & raw = get("kq"+suffix);
        const auto & mask = get("kq_soft_max"+suffix+"/mask");
        const size_t cells = raw.size()/64;
        require(cells*64 == raw.size() && mask.size() >= cells, "softmax shape mismatch");
        size_t unmasked = 0;
        for (size_t c = 0; c < cells; ++c) unmasked += std::isfinite(mask[c]);
        require(unmasked == size_t(layer == 1 ? std::min(pos+1, 128) : pos+1), "wrong full/SWA causal mask extent");
        const auto sinks = layer == 1 ? read_f32(model.layers[layer].attn_sinks) : Floats{};
        Floats expected(raw.size());
        for (int h = 0; h < 64; ++h) {
            double maximum = sinks.empty() ? -INFINITY : sinks[h];
            for (size_t c = 0; c < cells; ++c) maximum = std::max(maximum, raw[h*cells+c]/std::sqrt(192.0)+mask[c]);
            double denominator = sinks.empty() ? 0 : std::exp(sinks[h]-maximum);
            for (size_t c = 0; c < cells; ++c) denominator += std::exp(raw[h*cells+c]/std::sqrt(192.0)+mask[c]-maximum);
            for (size_t c = 0; c < cells; ++c) expected[h*cells+c] = float(std::exp(raw[h*cells+c]/std::sqrt(192.0)+mask[c]-maximum)/denominator);
        }
        auto softmax = compare("scalar_scaled_softmax_mask_sinks"+at, get("kq_soft_max"+suffix), expected, 2e-6, 1e-10);
        softmax["unmasked_keys"] = unmasked; softmax["sinks"] = !sinks.empty(); results.push_back(softmax);
        expected = get("attn_out"+suffix);
        for (auto & v : expected) v *= .707f;
        results.push_back(compare("value_scale_after_projection"+at, get("attn_out_scaled"+suffix), expected, 2e-6, 1e-10));
        if (layer > 0) {
            const auto & w = get("ffn_moe_weights_norm"+suffix);
            double sum = 0; for (float v : w) sum += v;
            results.push_back(compare("unscaled_router_sum"+at, {float(sum)}, {1.f}, 2e-6, 1e-10));
        }
    }
}

int main(int argc, char ** argv) {
    json report = {{"schema_version", 1}, {"status", "error"}, {"requested_revision", STRATA_MIMO_SOURCE_SHA},
        {"archive_sha256", STRATA_MIMO_ARCHIVE_SHA256}, {"patch_set", STRATA_MIMO_PATCH_SET},
        {"scope", "synthetic native MiMo graph, full/SWA128 masks and state; not full GGUF or inference throughput"},
        {"fixture", {{"seed", 0x26d001u}, {"layers", 3}, {"width", 256}, {"heads", 64}, {"kv_heads", {4,8,4}},
            {"key_dim", 192}, {"value_dim", 128}, {"rope_dim", 64}, {"swa_window", 128},
            {"experts", 16}, {"top_k", 8}, {"vocab", 64}, {"weights", {"F32", "BF16/Q2_K/Q3_K/MXFP4/F32"}}}},
        {"configuration", {{"context", 1024}, {"batch", 512}, {"reference_ubatch", 128}, {"ring_ubatch", 32},
            {"kv_type", "F32"}, {"flash_attention", "off/on (on casts KV to F16)"},
            {"NVIDIA_TF32_OVERRIDE", "0"}, {"GGML_CUDA_CUBLAS_COMPUTE_TYPE", "f32"}, {"cuda_graphs", false}, {"graph_reuse", false}}},
        {"tolerances", {{"F32_logit_abs", 5e-4}, {"F32_logit_nmse", 1e-7},
                        {"mixed_logit_abs", .002}, {"mixed_logit_nmse", 1e-5},
                        {"FA_logit_abs", .001}, {"FA_logit_nmse", 1e-6}}}};
    json results = json::array();
    std::filesystem::path directory;
    bool created_directory = false;
    try {
        require(argc == 2, "usage: strata-mimo2-graph-check NEW_WORK_DIRECTORY");
        directory = argv[1];
        require(!std::filesystem::exists(directory), "graph work directory must be new");
        std::filesystem::create_directories(directory);
        created_directory = true;
        for (const auto & pair : std::vector<std::pair<const char *, const char *>>{
                {"NVIDIA_TF32_OVERRIDE", "0"}, {"GGML_CUDA_CUBLAS_COMPUTE_TYPE", "f32"},
                {"GGML_CUDA_DISABLE_GRAPHS", "1"}, {"LLAMA_GRAPH_REUSE_DISABLE", "1"}}) {
            const char * value = std::getenv(pair.first);
            require(value && std::string(value) == pair.second, std::string("set ")+pair.first+"="+pair.second);
        }
        ggml_backend_load_all();
        auto * device = ggml_backend_dev_by_name("CUDA0");
        require(device && ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_GPU, "CUDA0 required");
        report["device"] = ggml_backend_dev_description(device);
        report["runs"] = json::array();
        for (bool mixed : {false, true}) {
            const std::string kind = mixed ? "mixed" : "f32";
            const auto fixture = (directory/(kind+".gguf")).string();
            const auto scalar_fixture = (directory/(kind+"-dequant.gguf")).string();
            write_synthetic_mimo2(fixture, mixed);
            write_synthetic_mimo2(scalar_fixture, mixed, true);
            auto gpu_model = load(fixture, true), cpu_model = load(scalar_fixture, false);
            const auto & h = gpu_model->hparams;
            require(h.n_layer() == 3 && h.n_swa == 128 && h.expert_weights_scale == 0.f && h.n_layer_nextn == 0, "wrong fixture hparams");
            for (int l = 0; l < 3; ++l) require(h.n_head(l) == 64 && h.n_head_kv(l) == (l == 1 ? 8u : 4u) &&
                h.n_rot(l) == 64 && h.n_embd_head_k(l) == 192 && h.n_embd_head_v(l) == 128 && h.is_swa(l) == (l == 1), "wrong attention geometry");
            Run reference(gpu_model.get(), true, 128, true), ring(gpu_model.get(), true, 32),
                cpu(cpu_model.get(), false, 64, true), serial(gpu_model.get(), true, 32), fa(gpu_model.get(), true, 32, false, true);
            const auto full = reference.decode(0, 384, 384);
            const auto cpu_logits = cpu.decode(0, 384, 64);
            const double abs = mixed ? .002 : 5e-4, nmse = mixed ? 1e-5 : 1e-7;
            results.push_back(compare(kind+"/gpu_vs_dequantized_cpu_f32", full, cpu_logits, abs, nmse));
            const auto split = ring.decode(0, 384, 63);
            results.push_back(compare(kind+"/bounded_swa_prefill_vs_full_cache", split, full, abs, nmse));
            results.push_back(compare(kind+"/serial_vs_prefill", serial.decode(0, 384, 1), full, abs, nmse));
            results.push_back(compare(kind+"/FA_on_vs_off", fa.decode(0, 384, 63), split, .001, 1e-6));
            require(fa.audit.ops["FLASH_ATTN_EXT"] > 0, "FA was not executed");
            for (int pos : {127,128,129,255,256,257,383}) results.push_back(compare(kind+"/boundary="+std::to_string(pos),
                Floats(split.begin()+pos*64, split.begin()+(pos+1)*64), Floats(full.begin()+pos*64, full.begin()+(pos+1)*64), abs, nmse));
            require(ring.memory()->get_swa()->get_size() < ring.memory()->get_base()->get_size() &&
                ring.memory()->get_swa()->seq_pos_min(0) > 0, "SWA cache did not evict history");
            const auto before = ring.positions();
            const auto saved = ring.save();
            const auto continuation = ring.decode(384, 24, 1);
            results.push_back(compare(kind+"/append_after_eviction", continuation, reference.decode(384, 24, 1), abs, nmse));
            require(llama_memory_seq_rm(ring.ctx->get_memory(), 0, 400, -1), "short rollback rejected");
            results.push_back(compare(kind+"/tail_rollback", ring.decode(400, 8, 1), Floats(continuation.end()-8*64, continuation.end()), abs, nmse));
            ring.clear(); ring.decode(0, 40, 20, 19); ring.clear(); ring.restore(saved);
            results.push_back(compare(kind+"/state_restore_after_other_conversation", ring.decode(384, 24, 1), continuation, abs, nmse));
            ring.clear();
            results.push_back(compare(kind+"/clear_replay", ring.decode(0, 130, 31), Floats(full.begin(), full.begin()+130*64), abs, nmse));
            if (!mixed) {
                Run captured(gpu_model.get(), true, 32);
                captured.decode(0, 127, 63);
                for (int pos : {127,128,129}) components(captured, *gpu_model, results, pos);
                captured.decode(130, 125, 63);
                for (int pos : {255,256,257}) components(captured, *gpu_model, results, pos);
            }
            report["runs"].push_back({{"weights", kind}, {"ring_after_384", before}, {"snapshot_bytes", saved.size()},
                {"gpu_compute_ops", reference.audit.ops}, {"fa_compute_ops", fa.audit.ops},
                {"gpu_cpu_nodes", reference.audit.cpu_nodes}, {"fa_cpu_nodes", fa.audit.cpu_nodes}});
        }
        require(results.size() == 132, "incomplete MiMo graph coverage");
        bool pass = true; for (const auto & r : results) pass &= r["pass"].get<bool>();
        report["status"] = pass ? "pass" : "fail";
    } catch (const std::exception & e) { report["error"] = e.what(); }
    report["results"] = results; report["case_count"] = results.size();
    if (created_directory) {
        std::ofstream file(directory/"graph-report.json"); file << report.dump(2) << '\n';
        if (!file) { std::cerr << "cannot write graph report\n"; return 1; }
        std::cout << report["status"] << ": " << results.size() << " cases; " << directory.string() << '\n';
    } else std::cout << report.dump(2) << '\n';
    ggml_quantize_free();
    return report["status"] == "pass" ? 0 : 1;
}

