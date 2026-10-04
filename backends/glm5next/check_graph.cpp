// Candidate-specific graph construction audit. No weight payloads, forward pass,
// or real weight/state/workspace allocations: never remove no_alloc below.
#include "llama.h"
#include "llama-context.h"
#include "llama-model.h"
#include "ggml-backend.h"
#include "nlohmann/json.hpp"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

using json = nlohmann::ordered_json;

static void require(bool ok, const std::string & message) {
    if (!ok) throw std::runtime_error(message);
}

static int weight_layer(const char * name) {
    const std::string s(name);
    if (s.rfind("blk.", 0) != 0) return -1;
    const auto end = s.find('.', 4);
    if (end == std::string::npos || end == 4) return -1;
    const auto number = s.substr(4, end - 4);
    if (number.find_first_not_of("0123456789") != std::string::npos) return -1;
    return std::stoi(number);
}

static json audit(llama_context & ctx, const llama_model & model, uint32_t tokens) {
    auto mctx = ctx.get_memory()->init_full();
    // split_only=true is essential: llama_graph_reserve's public C wrapper
    // otherwise allocates a real workspace even for a no_alloc model.
    auto * gf = ctx.graph_reserve(tokens, 1, tokens, mctx.get(), true);
    require(gf != nullptr, "graph construction failed");
    const auto & hp = model.hparams;
    std::set<int> layers, finished;
    std::set<std::string> weights;
    std::unordered_set<const ggml_tensor *> seen;
    std::vector<const ggml_tensor *> pending;
    json nodes = json::array(), cpu = json::array(), ops = json::object();
    int gpu_compute = 0, expert_matmuls = 0, indexer_pools = 0;
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        auto * t = ggml_graph_node(gf, i);
        pending.push_back(t);
        const std::string name(t->name), op(ggml_op_desc(t));
        if (!ops.contains(op)) ops[op] = 0;
        ops[op] = ops[op].get<int>() + 1;
        for (uint32_t layer = 0; layer < hp.n_layer(); ++layer) {
            if (name == "l_last-" + std::to_string(layer)) finished.insert(int(layer));
            if (name == "indexer_top_k_pools-" + std::to_string(layer)) ++indexer_pools;
        }
        if (t->op == GGML_OP_MUL_MAT_ID) ++expert_matmuls;
        auto * backend = ggml_backend_sched_get_tensor_backend(ctx.get_sched(), t);
        const auto * backend_name = backend ? ggml_backend_name(backend) : "unassigned";
        auto * device = backend ? ggml_backend_get_device(backend) : nullptr;
        const bool compute = t->op != GGML_OP_NONE && t->op != GGML_OP_VIEW &&
            t->op != GGML_OP_RESHAPE && t->op != GGML_OP_PERMUTE && t->op != GGML_OP_TRANSPOSE;
        const bool gpu = device && ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_GPU;
        json node = {{"name", name}, {"op", op}, {"backend", backend_name},
                     {"type", ggml_type_name(t->type)}, {"shape", {t->ne[0], t->ne[1], t->ne[2], t->ne[3]}}};
        if (compute) {
            if (gpu) ++gpu_compute;
            else cpu.push_back(node);
        }
        // Keep layer boundaries and indexer selection in the durable trace.
        // Every executable node above is still audited and counted.
        if (name.rfind("l_last-", 0) == 0 || name.rfind("indexer_top_k_pools-", 0) == 0 ||
            t->op == GGML_OP_GATED_DELTA_NET || t->op == GGML_OP_LIGHTNING_INDEXER)
            nodes.push_back(std::move(node));
    }
    while (!pending.empty()) {
        const auto * t = pending.back();
        pending.pop_back();
        if (!seen.insert(t).second) continue;
        const int layer = weight_layer(t->name);
        if (layer >= 0) {
            layers.insert(layer);
            weights.insert(t->name);
        }
        for (const auto * src : t->src) if (src) pending.push_back(src);
        if (t->view_src) pending.push_back(t->view_src);
    }
    const bool layer_bounds = !layers.empty() && *layers.begin() == 0 &&
        *layers.rbegin() == int(hp.n_layer()) - 1 && layers.size() == hp.n_layer();
    const bool complete = finished.size() == hp.n_layer();
    const bool sparse = ctx.get_cparams().n_ctx > hp.indexer_top_k + hp.indexer_kpool - 1;
    int attention_layers = 0;
    for (uint32_t i = 0; i < hp.n_layer(); ++i) attention_layers += !hp.is_recr(i);
    const bool indexer_ok = indexer_pools == (sparse ? attention_layers : 0);
    const auto count = [&ops](const char * op) { return ops.contains(op) ? ops[op].get<int>() : 0; };
    const bool families_ok = expert_matmuls == 3*int(hp.n_layer()-hp.n_layer_dense_lead) &&
        count("GATED_DELTA_NET") == int(hp.n_layer())-attention_layers &&
        count("DSV4_HC_PRE") == 2*int(hp.n_layer()) && count("DSV4_HC_POST") == 2*int(hp.n_layer()) &&
        count("DSV4_HC_COMB") == 2*int(hp.n_layer()) && count("LIGHTNING_INDEXER") == (sparse ? attention_layers : 0);
    return {{"status", layer_bounds && complete && cpu.empty() && indexer_ok && families_ok ? "pass" : "fail"},
            {"n_ctx", ctx.n_ctx()}, {"n_tokens", tokens}, {"n_outputs", tokens},
            {"no_alloc", true}, {"load_mtp", false}, {"flash_attention", "off"},
            {"n_rs_seq", ctx.get_cparams().n_rs_seq}, {"indexer_scoring", sparse},
            {"main_layer_boundary_ok", layer_bounds}, {"all_layers_finished", complete},
            {"indexer_branch_ok", indexer_ok}, {"op_families_ok", families_ok}, {"weight_layers", layers},
            {"finished_layers", finished}, {"referenced_layer_weight_count", weights.size()},
            {"node_count", ggml_graph_n_nodes(gf)}, {"gpu_compute_nodes", gpu_compute},
            {"cpu_compute_nodes", cpu}, {"expert_matmuls", expert_matmuls},
            {"indexer_pool_selection_nodes", indexer_pools}, {"op_counts", ops}, {"layer_trace", nodes}};
}

int main(int argc, char ** argv) {
    json result = {{"schema_version", 1}, {"status", "error"},
                   {"scope", "no_alloc graph construction and scheduler placement; no numerical inference"},
                   {"requested_revision", STRATA_GLM_SOURCE_SHA},
                   {"archive_sha256", STRATA_GLM_ARCHIVE_SHA256}};
    try {
        require((argc == 3 || (argc == 4 && std::string(argv[3]) == "--cpu-embedding")) &&
            std::string(argv[1]) == "--gguf",
            "usage: strata-glm5next-graph-check --gguf FIRST.gguf [--cpu-embedding]");
        result["gguf"] = argv[2];
        const char * tf32 = std::getenv("NVIDIA_TF32_OVERRIDE");
        require(tf32 && std::string(tf32) == "0", "set NVIDIA_TF32_OVERRIDE=0 before starting the checker");
        result["NVIDIA_TF32_OVERRIDE"] = tf32;
        ggml_backend_load_all();
        auto * dev = ggml_backend_dev_by_name("CUDA0");
        require(dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU, "CUDA0 GPU required");
        result["device"] = ggml_backend_dev_description(dev);
        ggml_backend_dev_t devices[] = {dev, nullptr};
        // llama's input embedding stays on CPU even with all repeating layers
        // offloaded. Strata's GPU-only profile also needs it on CUDA.
        llama_model_tensor_buft_override overrides[] = {
            {"token_embd\\.weight", ggml_backend_dev_buffer_type(dev)}, {nullptr, nullptr}};
        auto mp = llama_model_default_params();
        mp.devices = devices;
        mp.tensor_buft_overrides = argc == 4 ? nullptr : overrides;
        result["embedding_placement"] = argc == 4 ? "upstream default (negative audit)" : "CUDA0";
        mp.no_alloc = true;
        // AUTO may select mmap, whose host-buffer branch asserts !no_alloc.
        mp.load_mode = LLAMA_LOAD_MODE_NONE;
        mp.load_mtp = false;
        mp.n_gpu_layers = -1;
        mp.split_mode = LLAMA_SPLIT_MODE_NONE;
        mp.use_extra_bufts = false;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(argv[2], mp), llama_model_free);
        require(model != nullptr, "candidate loader failed");
        require(model->arch == LLM_ARCH_GLM5NEXT, "expected glm5next architecture");
        const auto & hp = model->hparams;
        require(hp.no_alloc, "model lost no_alloc flag");
        require(hp.n_layer_nextn == 1 && hp.n_layer() == 45, "expected 45 main + 1 MTP GLM Flash profile");
        require(!model->tensors_by_name.empty(), "no model tensor descriptors loaded");
        int recurrent = 0;
        for (uint32_t i = 0; i < hp.n_layer(); ++i) recurrent += hp.is_recr(i);
        result["model"] = {{"layers", hp.n_layer_all}, {"main_layers", hp.n_layer()},
                           {"mtp_layers", hp.n_layer_nextn}, {"kda_layers", recurrent},
                           {"dsa_layers", hp.n_layer() - recurrent}};
        size_t weight_buffer_bytes = 0;
        std::set<ggml_backend_buffer_t> buffers;
        for (const auto & item : model->tensors_by_name) {
            require(weight_layer(item.first.c_str()) < int(hp.n_layer()), "MTP weight was loaded with load_mtp=false: " + item.first);
            require(item.second->data == nullptr, "unexpected weight payload allocation");
            if (item.second->buffer) buffers.insert(item.second->buffer);
        }
        for (auto * buffer : buffers) weight_buffer_bytes += ggml_backend_buffer_get_size(buffer);
        require(weight_buffer_bytes == 0, "unexpected nonzero weight buffer");
        result["weight_buffer_bytes"] = weight_buffer_bytes;
        result["loaded_tensor_descriptors"] = model->tensors_by_name.size();
        result["graphs"] = json::array();
        bool pass = true;
        for (uint32_t n_ctx : {2048u, 4096u}) {
            auto cp = llama_context_default_params();
            cp.n_ctx = n_ctx;
            cp.n_batch = cp.n_ubatch = 256;
            cp.n_seq_max = 1;
            cp.n_rs_seq = 4;
            cp.n_threads = cp.n_threads_batch = 4;
            cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
            cp.type_k = cp.type_v = GGML_TYPE_F16;
            std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
                llama_init_from_model(model.get(), cp), llama_free);
            require(ctx != nullptr, "context graph reservation failed");
            for (uint32_t tokens : {1u, 4u, 16u, 256u}) {
                auto graph = audit(*ctx, *model, tokens);
                pass = pass && graph["status"] == "pass";
                result["graphs"].push_back(std::move(graph));
            }
        }
        result["status"] = pass ? "pass" : "fail";
    } catch (const std::exception & e) {
        result["error"] = e.what();
    }
    std::cout << result.dump(2) << '\n';
    return result["status"] == "pass" ? 0 : 1;
}
