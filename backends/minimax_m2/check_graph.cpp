// Native MiniMax graph validation against dequantized CPU and scalar component oracles.
#include "synthetic_minimax_m2.hpp"
#include "llama.h"
#include "llama-context.h"
#include "llama-model.h"
#include "llama-kv-cache.h"
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
#include <numeric>
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
    std::map<const ggml_tensor *, Floats> produced;
    static bool wanted(const std::string & name) {
        for (const std::string prefix : {"Qcur", "Kcur", "Vcur", "attn_out", "kq", "ffn_moe_", "ffn_inp", "l_out"})
            if (name.find(prefix) == 0) return true;
        return false;
    }
    static bool callback(ggml_tensor * t, bool ask, void * data) {
        auto & a = *static_cast<Audit *>(data);
        if (!ask) {
            std::string name = t->name;
            if (t->type == GGML_TYPE_I32) {
                std::vector<int32_t> ids(ggml_nelements(t));
                require(ggml_is_contiguous(t), "expected contiguous routing IDs");
                ggml_backend_tensor_get(t, ids.data(), 0, ids.size()*sizeof(int32_t));
                a.values[name] = Floats(ids.begin(), ids.end());
            } else {
                if ((name.find("Qcur-") == 0 || name.find("Kcur-") == 0) && t->op == GGML_OP_MUL_MAT)
                    name += "/pre_norm";
                auto value = read_f32(t);
                if (t->op == GGML_OP_ADD && (name.find("ffn_inp-") == 0 || name.find("l_out-") == 0)) {
                    // The allocator may reuse an operand for this ADD output.
                    // Compare with the values captured when operands were produced,
                    // never with their already overwritten backing allocation.
                    require(a.produced.count(t->src[0]) && a.produced.count(t->src[1]), "missing residual producer");
                    a.values[name+"/left"] = a.produced.at(t->src[0]);
                    a.values[name+"/right"] = a.produced.at(t->src[1]);
                }
                a.produced[t] = value;
                a.values[name] = std::move(value);
            }
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
        return a.capture && ((t->type == GGML_TYPE_F32) ||
            (t->type == GGML_TYPE_I32 && std::string(t->name).find("ffn_moe_argsort") == 0));
    }
};

// Full projection normalization, independently accumulated in double precision.
static Floats norm_reference(const Floats & raw, const Floats & weight, int tokens, int group) {
    const int width = int(weight.size());
    require(raw.size() == size_t(width*tokens) && width%group == 0, "norm shape mismatch");
    Floats out(raw.size());
    for (int t = 0; t < tokens; ++t) for (int base = 0; base < width; base += group) {
        double square = 0;
        for (int i = 0; i < group; ++i) square += double(raw[t*width+base+i])*raw[t*width+base+i];
        const double scale = 1/std::sqrt(square/group + 1e-6);
        for (int i = 0; i < group; ++i) out[t*width+base+i] = float(raw[t*width+base+i]*scale*weight[base+i]);
    }
    return out;
}

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
    require(bool(model), "MiniMax fixture load failed");
    return model;
}

struct Run {
    Audit audit;
    Context ctx{nullptr, llama_free};
    Run(llama_model * model, bool gpu, int ubatch, bool fa = false) {
        audit.gpu = gpu;
        auto cp = llama_context_default_params();
        cp.n_ctx = 512; cp.n_batch = 256; cp.n_ubatch = ubatch; cp.n_seq_max = 1;
        cp.n_threads = cp.n_threads_batch = 4; cp.type_k = cp.type_v = GGML_TYPE_F32;
        cp.flash_attn_type = fa ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.offload_kqv = gpu; cp.op_offload = gpu;
        cp.cb_eval = Audit::callback; cp.cb_eval_user_data = &audit;
        ctx.reset(llama_init_from_model(model, cp));
        require(bool(ctx), "MiniMax context creation failed");
        audit.ctx = ctx.get();
        require(ctx->get_cparams().flash_attn == fa, "requested FA mode was not honored");
        require(dynamic_cast<llama_kv_cache *>(ctx->get_memory()) != nullptr, "MiniMax must use full-attention KV cache");
    }
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
            require(status == 0, "MiniMax decode failed at " + std::to_string(pos) + ": " + std::to_string(status));
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

};


static void components(Run & run, llama_model & model, json & results, int pos, int tokens) {
    run.audit.values.clear(); run.audit.produced.clear(); run.audit.capture = true;
    run.decode(pos, tokens, tokens); run.audit.capture = false;
    const auto get = [&](const std::string & name) -> const Floats & {
        auto it = run.audit.values.find(name);
        require(it != run.audit.values.end(), "missing captured MiniMax node: "+name);
        return it->second;
    };
    for (int layer = 0; layer < 3; ++layer) {
        const std::string suffix = "-"+std::to_string(layer);
        const std::string at = "/layer="+std::to_string(layer)+"/pos="+std::to_string(pos)+"/tokens="+std::to_string(tokens);
        for (bool key : {false, true}) {
            const std::string name = key ? "Kcur" : "Qcur";
            const int heads = key ? 8 : 48, width = heads*128;
            const auto & raw = get(name+suffix+"/pre_norm");
            const auto weights = read_f32(key ? model.layers[layer].attn_k_norm : model.layers[layer].attn_q_norm);
            auto expected = norm_reference(raw, weights, tokens, width);
            results.push_back(compare(name+"/whole_projection_RMSNorm"+at, get(name+"_normed"+suffix), expected, 3e-5, 1e-10));
            const auto wrong = norm_reference(raw, weights, tokens, 128);
            double contrast = 0;
            for (size_t i = 0; i < wrong.size(); ++i) contrast = std::max(contrast, std::abs(double(wrong[i])-expected[i]));
            results.push_back({{"name", name+"/per_head_norm_negative_control"+at}, {"pass", contrast > .01},
                               {"wrong_per_head_max_abs", contrast}, {"required_difference", .01}});
            for (int t = 0; t < tokens; ++t) for (int h = 0; h < heads; ++h) for (int i = 0; i < 32; ++i) {
                const size_t base = (t*heads+h)*128;
                // RoPE stores theta_scale as F32 before raising it to i. Match
                // that representable frequency while accumulating rotation in double.
                const float theta_scale = std::pow(5000000.f, -2.f/64);
                const float angle_f32 = (pos+t)*std::pow(theta_scale, float(i));
                const double angle = angle_f32;
                const double a = expected[base+i], b = expected[base+i+32];
                expected[base+i] = float(a*std::cos(angle)-b*std::sin(angle));
                expected[base+i+32] = float(a*std::sin(angle)+b*std::cos(angle));
            }
            // The comparison includes the untouched upper 64 head dimensions.
            // Unit-RMS Q/K are larger than the unnormalized projections used
            // by other fixtures. Allow 1e-4 absolute for F32 powf/sincos at
            // position 127; keep the independent normalized error bound.
            results.push_back(compare(name+"/partial_NeoX_RoPE64"+at, get(name+suffix), expected, 1e-4, 1e-10));
        }
        const auto & raw = get("kq"+suffix);
        const auto & mask = get("kq_soft_max"+suffix+"/mask");
        const size_t cells = raw.size()/(48*tokens);
        require(cells*48*tokens == raw.size() && mask.size() >= cells*tokens, "softmax shape mismatch");
        Floats softmax(raw.size());
        bool causal = true;
        for (int t = 0; t < tokens; ++t) {
            size_t unmasked = 0;
            for (size_t c = 0; c < cells; ++c) unmasked += std::isfinite(mask[t*cells+c]);
            causal &= unmasked == size_t(pos+t+1);
            for (int h = 0; h < 48; ++h) {
                const size_t base = (h*tokens+t)*cells;
                double maximum = -INFINITY;
                for (size_t c = 0; c < cells; ++c) maximum = std::max(maximum, raw[base+c]/std::sqrt(128.0)+mask[t*cells+c]);
                double denominator = 0;
                for (size_t c = 0; c < cells; ++c) denominator += std::exp(raw[base+c]/std::sqrt(128.0)+mask[t*cells+c]-maximum);
                for (size_t c = 0; c < cells; ++c) softmax[base+c] = float(std::exp(raw[base+c]/std::sqrt(128.0)+mask[t*cells+c]-maximum)/denominator);
            }
        }
        results.push_back({{"name", "full_causal_mask"+at}, {"pass", causal}});
        results.push_back(compare("scaled_softmax"+at, get("kq_soft_max"+suffix), softmax, 3e-6, 1e-10));
        const auto & logits = get("ffn_moe_logits"+suffix);
        const auto & sorted = get("ffn_moe_argsort"+suffix);
        const auto bias = read_f32(model.layers[layer].ffn_exp_probs_b);
        require(logits.size() == size_t(16*tokens) && sorted.size() == size_t(16*tokens), "router capture size");
        Floats probabilities(logits.size()), expected_weights(8*tokens);
        bool ids_exact = true;
        double selection_margin = 1;
        for (int t = 0; t < tokens; ++t) {
            std::vector<int> order(16); std::iota(order.begin(), order.end(), 0);
            std::vector<double> probs(16), scores(16);
            for (int e = 0; e < 16; ++e) {
                probs[e] = 1/(1+std::exp(-double(logits[t*16+e])));
                probabilities[t*16+e] = float(probs[e]); scores[e] = probs[e]+bias[e];
            }
            std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return scores[a] > scores[b]; });
            double sum = 0;
            for (int i = 0; i < 8; ++i) {
                ids_exact &= sorted[t*16+i] == order[i]; sum += probs[order[i]];
                selection_margin = std::min(selection_margin, scores[order[i]]-scores[order[i+1]]);
            }
            for (int i = 0; i < 8; ++i) expected_weights[t*8+i] = float(probs[order[i]]/std::max(sum, 6.103515625e-5));
        }
        results.push_back(compare("sigmoid_router"+at, get("ffn_moe_probs"+suffix), probabilities, 2e-6, 1e-10));
        results.push_back({{"name", "biased_top8_exact"+at}, {"pass", ids_exact && selection_margin > 1e-6},
                           {"selection_margin_min", selection_margin}});
        results.push_back(compare("unbiased_normalized_weights"+at, get("ffn_moe_weights_norm"+suffix), expected_weights, 2e-6, 1e-10));
        for (const std::string name : {"ffn_inp", "l_out"}) {
            const auto & left = get(name+suffix+"/left");
            const auto & right = get(name+suffix+"/right");
            require(left.size() == right.size(), "residual shape mismatch");
            Floats sum(left.size());
            for (size_t i = 0; i < sum.size(); ++i) sum[i] = left[i]+right[i];
            results.push_back(compare(name+"/residual"+at, get(name+suffix), sum, 0, 0, true));
        }
    }
}

static std::pair<std::vector<int>, Floats> generate(Run & run) {
    run.clear();
    auto logits = run.decode(0, 17, 17);
    Floats all(logits.end()-64, logits.end());
    std::vector<int> ids;
    for (int pos = 17; pos < 29; ++pos) {
        const auto begin = all.end()-64;
        const int token = int(std::max_element(begin, all.end())-begin);
        ids.push_back(token);
        auto batch = llama_batch_init(1, 0, 1);
        batch.n_tokens = 1; batch.token[0] = token; batch.pos[0] = pos;
        batch.n_seq_id[0] = 1; batch.seq_id[0][0] = 0; batch.logits[0] = 1;
        const int rc = llama_decode(run.ctx.get(), batch);
        llama_batch_free(batch); require(rc == 0, "generated decode failed");
        const float * next = llama_get_logits_ith(run.ctx.get(), 0);
        require(next != nullptr, "missing generated logits");
        all.insert(all.end(), next, next+64);
    }
    require(run.audit.cpu_nodes.empty(), "CPU tensor math in GPU generation");
    return {ids, all};
}

int main(int argc, char ** argv) {
    json report = {{"schema_version", 1}, {"status", "error"}, {"requested_revision", STRATA_MM27_SOURCE_SHA},
        {"archive_sha256", STRATA_MM27_ARCHIVE_SHA256}, {"patch_set", STRATA_MM27_PATCH_SET},
        {"scope", "synthetic native MiniMax full-attention graph; not original GGUF or inference throughput"},
        {"fixture", {{"seed", 0x26d001u}, {"layers", 3}, {"width", 256}, {"ffn", 512},
            {"heads", 48}, {"kv_heads", 8}, {"key_dim", 128}, {"value_dim", 128}, {"rope_dim", 64},
            {"experts", 16}, {"top_k", 8}, {"vocab", 64}, {"weights", {"F32", "Q4_K/Q6_K/F32"}}}},
        {"configuration", {{"context", 512}, {"batch", 256}, {"ubatches", {8,32,64}}, {"kv_type", "F32"},
            {"flash_attention", "off/on (on casts KV to F16)"}, {"NVIDIA_TF32_OVERRIDE", "0"},
            {"GGML_CUDA_CUBLAS_COMPUTE_TYPE", "f32"},
            {"STRATA_MM27_QUANT_F32", std::getenv("STRATA_MM27_QUANT_F32") ? std::getenv("STRATA_MM27_QUANT_F32") : "0"},
            {"cuda_graphs", false}, {"graph_reuse", false}}},
        {"tolerances", {{"F32_logit_abs", 5e-4}, {"F32_logit_nmse", 1e-7},
            {"mixed_logit_abs", .002}, {"mixed_logit_nmse", 1e-5}, {"FA_logit_abs", .001}, {"FA_logit_nmse", 1e-6},
            {"partial_RoPE_abs", 1e-4}, {"partial_RoPE_nmse", 1e-10}}}};
    json results = json::array();
    std::filesystem::path directory;
    bool created = false;
    try {
        require(argc == 2, "usage: strata-minimax-m2-graph-check NEW_WORK_DIRECTORY");
        directory = argv[1]; require(!std::filesystem::exists(directory), "graph work directory must be new");
        std::filesystem::create_directories(directory); created = true;
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
            write_synthetic_minimax_m2(fixture, mixed);
            write_synthetic_minimax_m2(scalar_fixture, mixed, true);
            auto gpu_model = load(fixture, true), cpu_model = load(scalar_fixture, false);
            const auto & h = gpu_model->hparams;
            require(h.n_layer() == 3 && h.n_swa == 0 && h.expert_weights_scale == 0.f && h.n_layer_nextn == 0, "wrong fixture hparams");
            for (int l = 0; l < 3; ++l) require(h.n_head(l) == 48 && h.n_head_kv(l) == 8 && h.n_rot(l) == 64 &&
                h.n_embd_head_k(l) == 128 && h.n_embd_head_v(l) == 128 && !h.is_swa(l), "wrong GQA/partial RoPE geometry");
            Run reference(gpu_model.get(), true, 64), split(gpu_model.get(), true, 8), cpu(cpu_model.get(), false, 32),
                serial(gpu_model.get(), true, 8), fa(gpu_model.get(), true, 32, true);
            const auto full = reference.decode(0, 128, 128);
            const auto cpu_logits = cpu.decode(0, 128, 32);
            const double abs = mixed ? .002 : 5e-4, nmse = mixed ? 1e-5 : 1e-7;
            results.push_back(compare(kind+"/gpu_vs_dequantized_cpu_f32", full, cpu_logits, abs, nmse));
            results.push_back(compare(kind+"/first_output_logit", Floats(full.begin(), full.begin()+64), Floats(cpu_logits.begin(), cpu_logits.begin()+64), abs, nmse));
            const auto split_logits = split.decode(0, 128, 17);
            results.push_back(compare(kind+"/ubatch8_split_vs_64", split_logits, full, abs, nmse));
            results.push_back(compare(kind+"/serial_vs_prefill", serial.decode(0, 128, 1), full, abs, nmse));
            results.push_back(compare(kind+"/FA_on_vs_off", fa.decode(0, 128, 17), split_logits, .001, 1e-6));
            require(fa.audit.ops["FLASH_ATTN_EXT"] > 0, "FA was not executed");
            const auto saved = split.save();
            const auto continuation = split.decode(128, 8, 1);
            results.push_back(compare(kind+"/append", continuation, reference.decode(128, 8, 1), abs, nmse));
            split.clear(); split.decode(0, 16, 8, 19); split.clear(); split.restore(saved);
            results.push_back(compare(kind+"/state_restore", split.decode(128, 8, 1), continuation, abs, nmse));
            split.clear();
            results.push_back(compare(kind+"/clear_replay", split.decode(0, 32, 2), Floats(full.begin(), full.begin()+32*64), abs, nmse));
            const auto gpu_generated = generate(serial), cpu_generated = generate(cpu), fa_generated = generate(fa);
            results.push_back({{"name", kind+"/greedy_from_first_generated_token"},
                {"pass", gpu_generated.first == cpu_generated.first && gpu_generated.first == fa_generated.first},
                {"gpu_ids", gpu_generated.first}, {"cpu_ids", cpu_generated.first}, {"fa_ids", fa_generated.first}});
            results.push_back(compare(kind+"/generated_logits_vs_cpu", gpu_generated.second, cpu_generated.second, abs, nmse));
            results.push_back(compare(kind+"/generated_logits_FA", fa_generated.second, gpu_generated.second, .001, 1e-6));
            if (!mixed) {
                Run captured(gpu_model.get(), true, 32);
                int next = 0;
                for (const auto & point : std::vector<std::pair<int,int>>{{0,1},{1,1},{2,8},{31,2},{127,1}}) {
                    if (point.first > next) captured.decode(next, point.first-next, 17);
                    components(captured, *gpu_model, results, point.first, point.second);
                    next = point.first+point.second;
                }
            }
            report["runs"].push_back({{"weights", kind}, {"snapshot_bytes", saved.size()},
                {"gpu_compute_ops", reference.audit.ops}, {"fa_compute_ops", fa.audit.ops},
                {"gpu_cpu_nodes", reference.audit.cpu_nodes}, {"fa_cpu_nodes", fa.audit.cpu_nodes}});
        }
        require(results.size() == 217, "incomplete MiniMax graph coverage");
        bool pass = true; for (const auto & r : results) pass &= r["pass"].get<bool>();
        report["status"] = pass ? "pass" : "fail";
    } catch (const std::exception & e) { report["error"] = e.what(); }
    report["results"] = results; report["case_count"] = results.size();
    if (created) {
        std::ofstream file(directory/"graph-report.json"); file << report.dump(2) << '\n';
        if (!file) { std::cerr << "cannot write graph report\n"; return 1; }
        std::cout << report["status"] << ": " << results.size() << " cases; " << directory.string() << '\n';
    } else std::cout << report.dump(2) << '\n';
    ggml_quantize_free();
    return report["status"] == "pass" ? 0 : 1;
}
