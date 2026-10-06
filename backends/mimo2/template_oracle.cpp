// Raw embedded-template semantics in the dependency's independent Jinja engine.
// API argument normalization and media admission happen before rendering.
#include "jinja/lexer.h"
#include "jinja/parser.h"
#include "jinja/runtime.h"
#include "jinja/caps.h"
#include "json.h"
#include <iostream>
using json = common_json;
int main(int argc, char ** argv) {
    if (argc == 2 && std::string(argv[1]) == "--version") {
        std::cout << json({{"architecture", "mimo2"}, {"requested_revision", STRATA_MIMO_SOURCE_SHA},
            {"archive_sha256", STRATA_MIMO_ARCHIVE_SHA256}, {"patches", STRATA_MIMO_PATCH_SET},
            {"renderer", "native-jinja-raw"}}).dump() << '\n'; return 0;
    }
    if (argc != 1) { std::cerr << "usage: strata-mimo2-template [--version]\n"; return 2; }
    for (std::string line; std::getline(std::cin, line);) {
        if (line == "QUIT" || line == "QUIT\r") break;
        json response;
        try {
            if (line.size() > 8*1024*1024) throw std::invalid_argument("request exceeds limit");
            const auto request = json::parse(line);
            const auto source = request.at("template").get<std::string>();
            const auto inputs = request.at("context");
            jinja::lexer lexer;
            auto ast = jinja::parse_from_tokens(lexer.tokenize(source));
            jinja::caps_get(ast);
            jinja::context context(source);
            jinja::global_from_json(context, inputs, true);
            jinja::runtime runtime(context);
            const auto output = jinja::runtime::gather_string_parts(runtime.execute(ast));
            response = {{"rendered", output->as_string().str()}};
        } catch (const std::exception & error) { response = {{"error", error.what()}}; }
        std::cout << response.dump() << '\n' << std::flush;
    }
}
