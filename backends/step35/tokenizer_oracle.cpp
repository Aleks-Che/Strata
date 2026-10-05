#include "llama.h"
#include "nlohmann/json.hpp"
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

static std::string hex(const std::string & bytes) {
    const char * digits = "0123456789abcdef";
    std::string out;
    out.reserve(bytes.size() * 2);
    for (unsigned char byte : bytes) {
        out += digits[byte >> 4];
        out += digits[byte & 15];
    }
    return out;
}

int main(int argc, char ** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--version") {
            std::cout << json({{"architecture", "step35"}, {"requested_revision", STRATA_STEP_SOURCE_SHA},
                               {"archive_sha256", STRATA_STEP_ARCHIVE_SHA256}, {"patches", STRATA_STEP_PATCH_SET}}).dump() << '\n';
            return 0;
        }
        if (argc != 3 || std::string(argv[1]) != "--gguf")
            throw std::invalid_argument("usage: strata-step35-tokenizer --gguf FIRST.gguf | --version");
        auto params = llama_model_default_params();
        params.vocab_only = true;
        params.n_gpu_layers = 0;
        params.load_mtp = false;
        params.use_extra_bufts = false;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(argv[2], params), llama_model_free);
        if (!model) throw std::runtime_error("failed to load vocabulary");
        for (const auto & field : std::vector<std::pair<std::string, std::string>>{
                 {"general.architecture", "step35"}, {"tokenizer.ggml.pre", "deepseek-v3"}}) {
            char value[64] = {};
            const auto length = llama_model_meta_val_str(model.get(), field.first.c_str(), value, sizeof(value));
            if (length < 0 || length >= int(sizeof(value)) || value != field.second)
                throw std::runtime_error("unexpected " + field.first);
        }
        const auto * vocab = llama_model_get_vocab(model.get());
        std::vector<llama_token> eog;
        for (llama_token id = 0; id < llama_vocab_n_tokens(vocab); ++id)
            if (llama_vocab_is_eog(vocab, id)) eog.push_back(id);
        std::cout << json({{"ready", true}, {"vocab_size", llama_vocab_n_tokens(vocab)},
                           {"bos", llama_vocab_bos(vocab)}, {"eos", llama_vocab_eos(vocab)},
                           {"eog_ids", eog}, {"add_special", false}}).dump() << '\n' << std::flush;
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line == "QUIT" || line == "QUIT\r") break;
            try {
                if (line.size() > 8 * 1024 * 1024) throw std::invalid_argument("request exceeds limit");
                auto request = json::parse(line);
                const auto text = request.at("text").get<std::string>();
                const auto parse_special = request.at("parse_special").get<bool>();
                if (text.size() > 1024 * 1024) throw std::invalid_argument("text exceeds 1 MiB");
                // The embedded template owns BOS: never insert it a second time.
                const auto n = llama_tokenize(vocab, text.data(), int32_t(text.size()), nullptr, 0, false, parse_special);
                if (n > 0 || n == std::numeric_limits<int32_t>::min()) throw std::runtime_error("invalid token count");
                std::vector<llama_token> ids(static_cast<size_t>(-int64_t(n)));
                const auto count = llama_tokenize(vocab, text.data(), int32_t(text.size()), ids.data(), int32_t(ids.size()), false, parse_special);
                if (count < 0 || size_t(count) > ids.size()) throw std::runtime_error("tokenization failed");
                ids.resize(count);
                const auto bytes = llama_detokenize(vocab, ids.data(), count, nullptr, 0, false, true);
                if (bytes > 0 || bytes == std::numeric_limits<int32_t>::min()) throw std::runtime_error("invalid decoded length");
                std::string decoded(size_t(-int64_t(bytes)), '\0');
                const auto written = llama_detokenize(vocab, ids.data(), count, decoded.data(), int32_t(decoded.size()), false, true);
                if (written < 0 || size_t(written) > decoded.size()) throw std::runtime_error("detokenization failed");
                decoded.resize(written);
                std::cout << json({{"ids", ids}, {"decoded_hex", hex(decoded)}}).dump() << '\n' << std::flush;
            } catch (const std::exception &) {
                std::cout << "{\"error\":\"invalid request or tokenizer failure\"}\n" << std::flush;
            }
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "Step tokenizer: " << error.what() << '\n';
        return 1;
    }
}
