// Actual llama tensor registration: no allocations, mmap, lazy reads or inference.
#include "llama.h"
#include "llama-model.h"
#include "ggml-backend.h"
#include "nlohmann/json.hpp"
#include <iostream>
#include <memory>
#include <stdexcept>

using json = nlohmann::ordered_json;
static json version() {
    return {{"architecture", "minimax-m2"}, {"requested_revision", STRATA_MM27_SOURCE_SHA},
            {"archive_sha256", STRATA_MM27_ARCHIVE_SHA256}, {"patches", STRATA_MM27_PATCH_SET}};
}
int main(int argc, char ** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--version") {
            std::cout << version().dump() << '\n'; return 0;
        }
        if (argc != 3 || std::string(argv[1]) != "--gguf")
            throw std::invalid_argument("usage: strata-minimax-m2-loader --gguf MODEL.gguf | --version");
        llama_backend_init();
        json runs = json::array();
        for (bool mtp : {false, true}) {
            auto params = llama_model_default_params();
            params.no_alloc = true;
            params.n_gpu_layers = 0;
            params.load_mode = LLAMA_LOAD_MODE_NONE;
            params.lazy_mode = LLAMA_LAZY_MODE_OFF;
            params.use_extra_bufts = false;
            params.load_mtp = mtp;
            std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
                llama_model_load_from_file(argv[2], params), llama_model_free);
            if (!model) throw std::runtime_error("no-allocation registration failed");
            char arch[64] = {};
            if (llama_model_meta_val_str(model.get(), "general.architecture", arch, sizeof(arch)) < 0 || std::string(arch) != "minimax-m2")
                throw std::runtime_error("expected minimax-m2 model");
            if (model->hparams.n_layer_nextn != 0)
                throw std::runtime_error("this oracle admits text trunk without MTP only");
            uint64_t logical_bytes = 0;
            json tensors = json::array(), layers = json::array();
            for (const auto & [name, t] : model->tensors_by_name) {
                if (t->data || (t->buffer && ggml_backend_buffer_get_size(t->buffer) != 0))
                    throw std::runtime_error("no_alloc unexpectedly allocated tensor payload");
                logical_bytes += ggml_nbytes(t);
                tensors.push_back({{"name", name}, {"shape", {t->ne[0], t->ne[1], t->ne[2], t->ne[3]}},
                                   {"type_id", int(t->type)}, {"bytes", ggml_nbytes(t)}});
            }
            for (uint32_t i = 0; i < model->hparams.n_layer(); ++i) {
                const auto & h = model->hparams;
                layers.push_back({{"layer", i}, {"swa", h.is_swa(i)}, {"heads", h.n_head(i)},
                                  {"kv_heads", h.n_head_kv(i)}, {"key_dim", h.n_embd_head_k(i)},
                                  {"value_dim", h.n_embd_head_v(i)}, {"rope_dim", h.n_rot(i)}});
            }
            runs.push_back({{"load_mtp", mtp}, {"main_blocks", model->hparams.n_layer()},
                            {"all_blocks", model->hparams.n_layer_all}, {"nextn_blocks", model->hparams.n_layer_nextn},
                            {"logical_bytes", logical_bytes}, {"allocated_weight_bytes", 0},
                            {"swa_window", model->hparams.n_swa}, {"layers", layers}, {"tensors", tensors}});
        }
        llama_backend_free();
        auto report = version();
        report["status"] = "pass";
        report["scope"] = "compiled registration only; no weight payload allocation/reads or inference";
        report["runs"] = runs;
        std::cout << report.dump(2) << '\n';
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "MiniMax loader oracle: " << error.what() << '\n'; return 1;
    }
}
