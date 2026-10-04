#include "llama.h"
#include "tokenizer_protocol.hpp"
#include <iostream>
#include <limits>
#include <memory>

int main(int argc, char ** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--version") {
            std::cout << "requested_revision=" << STRATA_GLM_SOURCE_SHA
                      << "\narchive_sha256=" << STRATA_GLM_ARCHIVE_SHA256 << '\n';
            return 0;
        }
        if (argc != 3 || std::string(argv[1]) != "--gguf")
            throw std::invalid_argument("usage: strata-glm5next-tokenizer --gguf FIRST.gguf | --version");
        auto params = llama_model_default_params();
        params.vocab_only = true;
        params.n_gpu_layers = 0;
        params.load_mtp = false;
        params.use_extra_bufts = false;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(argv[2], params), llama_model_free);
        if (!model) throw std::runtime_error("failed to load vocabulary");
        for (const auto & field : std::vector<std::pair<std::string, std::string>>{
                 {"general.architecture", "glm5next"}, {"tokenizer.ggml.pre", "glm4"}}) {
            char value[64] = {};
            const int length = llama_model_meta_val_str(model.get(), field.first.c_str(), value, sizeof(value));
            if (length < 0 || length >= int(sizeof(value)) || value != field.second)
                throw std::runtime_error("unexpected " + field.first);
        }
        const auto * vocab = llama_model_get_vocab(model.get());
        strata_glm::tokenizer_loop(std::cin, std::cout, [vocab](const strata_glm::EncodeRequest & request) {
            // add_special=false: the embedded chat template owns BOS/prefixes.
            const auto n = llama_tokenize(vocab, request.text.data(), int32_t(request.text.size()),
                                          nullptr, 0, false, request.parse_special);
            if (n > 0 || n == std::numeric_limits<int32_t>::min())
                throw std::runtime_error("invalid token count");
            std::vector<llama_token> ids(static_cast<size_t>(-int64_t(n)));
            const auto count = llama_tokenize(vocab, request.text.data(), int32_t(request.text.size()),
                                              ids.data(), int32_t(ids.size()), false, request.parse_special);
            if (count < 0 || size_t(count) > ids.size()) throw std::runtime_error("tokenization failed");
            ids.resize(size_t(count));
            return ids;
        });
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "GLM tokenizer: " << e.what() << '\n';
        return 1;
    }
}
