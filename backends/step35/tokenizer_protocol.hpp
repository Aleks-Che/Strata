#pragma once
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace step35 {
struct EncodeRequest {
    bool parse_special;
    std::string text;
};
inline int hex_digit(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    throw std::invalid_argument("invalid hex byte");
}
inline EncodeRequest parse_encode(const std::string & line) {
    std::istringstream in(line);
    std::string command, flag, hex, extra;
    if (!(in >> command >> flag) || command != "ENC" || (flag != "0" && flag != "1"))
        throw std::invalid_argument("expected ENC 0|1 [UTF-8-hex]");
    in >> hex; // no hex encodes the empty string
    if (in >> extra) throw std::invalid_argument("extra ENC fields");
    if (hex.size() % 2) throw std::invalid_argument("odd hex length");
    if (hex.size() > 2 * 1024 * 1024) throw std::invalid_argument("text exceeds 1 MiB");
    EncodeRequest request{flag == "1", {}};
    request.text.reserve(hex.size() / 2);
    for (size_t i = 0; i < hex.size(); i += 2)
        request.text.push_back(static_cast<char>(16 * hex_digit(hex[i]) + hex_digit(hex[i + 1])));
    return request;
}
inline std::string format_ids(const std::vector<int32_t> & ids) {
    std::string line = "IDS";
    for (auto id : ids) {
        if (id < 0) throw std::invalid_argument("negative token ID");
        line += " " + std::to_string(id);
    }
    return line;
}
template<class Encode>
void tokenizer_loop(std::istream & input, std::ostream & output, Encode encode) {
    output << "READY_TOKENIZER\n" << std::flush;
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line == "QUIT") return;
        try {
            const auto request = parse_encode(line);
            output << format_ids(encode(request)) << '\n' << std::flush;
        } catch (const std::exception &) {
            // Keep one output line per request even if a callback error contains
            // newlines or model paths. Detail belongs on stderr, not the wire.
            output << "ERR invalid ENC request or tokenization failure\n" << std::flush;
        }
    }
}
} // namespace step35
