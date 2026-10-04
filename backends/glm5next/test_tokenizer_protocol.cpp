#include "tokenizer_protocol.hpp"
#include <iostream>

static void require(bool ok, const char * what) {
    if (!ok) throw std::runtime_error(what); // active in Release, unlike assert
}
int main() {
    try {
        require(strata_glm::parse_encode("ENC 0").text.empty(), "empty text");
        const auto raw = strata_glm::parse_encode("ENC 1 D09FF09F9880000A");
        require(raw.parse_special, "special flag");
        require(raw.text == std::string("\xd0\x9f\xf0\x9f\x98\x80\0\n", 8), "raw UTF-8/NUL/newline bytes");
        for (const auto & bad : {"", "ENC", "ENC 2 41", "ENC -1 41", "ENC 01 41", "ENC 0 4",
                                 "ENC 0 4g", "ENC 0 41 more", "GEN 0 41"}) {
            bool rejected = false;
            try { strata_glm::parse_encode(bad); } catch (const std::invalid_argument &) { rejected = true; }
            require(rejected, "malformed request accepted");
        }
        bool rejected = false;
        try { strata_glm::parse_encode("ENC 0 " + std::string(2 * 1024 * 1024 + 2, '0')); }
        catch (const std::invalid_argument &) { rejected = true; }
        require(rejected, "oversized request accepted");
        require(strata_glm::format_ids({}) == "IDS", "empty IDs");
        require(strata_glm::format_ids({0, 154829, 2147483647}) == "IDS 0 154829 2147483647", "ID formatting");
        std::istringstream input("ENC 0 41\r\nENC 1 42\nENC 0 00\nENC 0 ff\nENC 0 zz\nENC 0\nQUIT\nENC 0 43\n");
        std::ostringstream output;
        int calls = 0;
        strata_glm::tokenizer_loop(input, output, [&calls](const strata_glm::EncodeRequest & r) {
            ++calls;
            if (r.text == std::string(1, '\0')) throw std::runtime_error("mock error\nsecond line");
            if (r.text == std::string(1, char(255))) return std::vector<int32_t>{-1};
            if (r.text.empty()) return std::vector<int32_t>{};
            return std::vector<int32_t>{r.parse_special ? 11 : 10};
        });
        require(calls == 5, "callback count / malformed request / QUIT");
        require(output.str() == "READY_TOKENIZER\nIDS 10\nIDS 11\n"
                "ERR invalid ENC request or tokenization failure\n"
                "ERR invalid ENC request or tokenization failure\n"
                "ERR invalid ENC request or tokenization failure\nIDS\n", "wire output / error recovery");
        std::cout << "GLM tokenizer protocol checks passed (mock encoder, no llama.cpp)\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
