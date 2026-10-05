// Direct Jinja renderer from the pinned dependency. No model weights or GPU.
#include "jinja/lexer.h"
#include "jinja/parser.h"
#include "jinja/runtime.h"
#include "jinja/caps.h"
#include "json.h"
#include <iostream>

using json = common_json;
static json version() {
    return {{"architecture", "step35"}, {"requested_revision", STRATA_STEP_SOURCE_SHA},
            {"archive_sha256", STRATA_STEP_ARCHIVE_SHA256},
            {"renderer", "native-jinja-with-tool-json-normalization"}};
}
static void normalize_tool_arguments(json & context) {
    // The pinned Jinja engine lacks the template's fromjson filter. Decode
    // OpenAI string arguments using the native JSON parser before rendering;
    // the unchanged template then uses its mapping branch. Python's independent
    // oracle still exercises the original fromjson branch on the original input.
    for (auto & message : context.at("messages")) {
        if (!message.contains("tool_calls") || message.at("tool_calls").is_null()) continue;
        for (auto & call : message["tool_calls"]) {
            auto & function = call.contains("function") ? call["function"] : call;
            if (function.contains("arguments") && function.at("arguments").is_string()) {
                auto parsed = json::parse(function.at("arguments").get<std::string>());
                if (!parsed.is_object()) throw std::runtime_error("tool arguments must be a JSON object");
                function["arguments"] = std::move(parsed);
            }
        }
    }
}
int main(int argc, char ** argv) {
    if (argc == 2 && std::string(argv[1]) == "--version") { std::cout << version().dump() << '\n'; return 0; }
    if (argc != 1) { std::cerr << "usage: strata-step35-template [--version]\n"; return 2; }
    for (std::string line; std::getline(std::cin, line);) {
        if (line == "QUIT") break;
        json response;
        try {
            const auto request = json::parse(line);
            const auto source = request.at("template").get<std::string>();
            auto inputs = request.at("context");
            normalize_tool_arguments(inputs);
            jinja::lexer lexer;
            auto ast = jinja::parse_from_tokens(lexer.tokenize(source));
            jinja::caps_get(ast);
            jinja::context context(source);
            jinja::global_from_json(context, inputs, true);
            jinja::runtime runtime(context);
            const auto output = jinja::runtime::gather_string_parts(runtime.execute(ast));
            response = {{"rendered", output->as_string().str()}};
        } catch (const std::exception & error) { response = {{"error", error.what()}}; }
        std::cout << response.dump() << '\n';
    }
}
