// Exercise actual tensor registration without allocating or reading weight data.
#include "llama.h"
#include "llama-model.h"
#include "ggml-backend.h"
#include "nlohmann/json.hpp"
#include <iostream>
#include <memory>
#include <stdexcept>

using json = nlohmann::ordered_json;
int main(int argc, char ** argv) {
    try {
        if (argc != 3 || std::string(argv[1]) != "--gguf")
            throw std::invalid_argument("usage: strata-hy3-loader --gguf MODEL.gguf (no allocations)");
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
            if (!model) throw std::runtime_error("no-allocation model loading failed");
            char arch[64] = {};
            if (llama_model_meta_val_str(model.get(), "general.architecture", arch, sizeof(arch)) < 0 || std::string(arch) != "hy_v3")
                throw std::runtime_error("expected hy_v3 model");
            uint64_t main_bytes = 0, mtp_bytes = 0;
            size_t main_count = 0;
            json mtp_tensors = json::array();
            const std::string prefix = "blk." + std::to_string(model->hparams.n_layer()) + ".";
            for (const auto & [name, tensor] : model->tensors_by_name) {
                if (tensor->data || (tensor->buffer && ggml_backend_buffer_get_size(tensor->buffer) != 0))
                    throw std::runtime_error("no_alloc unexpectedly allocated tensor payload");
                const auto bytes = ggml_nbytes(tensor);
                if (name.rfind(prefix, 0) == 0) {
                    mtp_bytes += bytes;
                    mtp_tensors.push_back({{"name", name}, {"bytes", bytes}});
                } else {
                    main_bytes += bytes;
                    ++main_count;
                }
            }
            runs.push_back({{"load_mtp", mtp}, {"main_blocks", model->hparams.n_layer()},
                            {"all_blocks", model->hparams.n_layer_all}, {"main_tensors", main_count},
                            {"main_logical_bytes", main_bytes}, {"mtp_logical_bytes", mtp_bytes},
                            {"mtp_tensors", mtp_tensors}, {"allocated_weight_bytes", 0}});
        }
        llama_backend_free();
        std::cout << json({{"status", "pass"}, {"scope", "compiled tensor registration, no_alloc, no payload reads or inference"},
                           {"source_revision", STRATA_HY3_SOURCE_SHA}, {"archive_sha256", STRATA_HY3_ARCHIVE_SHA256},
                           {"patches", STRATA_HY3_PATCH_SET}, {"runs", runs}}).dump(2) << '\n';
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "Hy3 loader oracle: " << error.what() << '\n';
        return 1;
    }
}
