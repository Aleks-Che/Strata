"""Compare the embedded Step template with the pinned native Jinja renderer."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment

if __package__:
    from .check_step35_tokenizer import ARCHIVE_SHA256, validate_provenance
    from .gguf_reader import GGUFFile
    from .setup_step35 import first_shard
    from .step35_loader_contract import LOADER_SHA
    from .strata_tokenizer import Tokenizer
else:
    from check_step35_tokenizer import ARCHIVE_SHA256, validate_provenance
    from gguf_reader import GGUFFile
    from setup_step35 import first_shard
    from step35_loader_contract import LOADER_SHA
    from strata_tokenizer import Tokenizer

TEMPLATE_SHA = "f428623fc81c940c35be3509fbffc086b4b4360d8800e46103e6f34d02891633"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def renderer(source):
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                       extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = lambda value, **kw: json.dumps(value, ensure_ascii=kw.pop("ensure_ascii", False), **kw)
    env.filters["fromjson"] = json.loads
    return env.from_string(source)


def corpus():
    user = {"role": "user", "content": "Привет, 你好!"}
    tool = {"type": "function", "function": {"name": "weather", "description": "Погода",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
    args = {"city": "Екатеринбург", "multiline": "one\ntwo", "count": 2, "ok": True,
            "absent": None, "fraction": 1.25, "options": {"unit": "°C", "x": [1, "中", False]}}
    cases = []
    def add(name, messages, **extra):
        cases.append({"name": name, "context": {"messages": messages, "tools": None,
                      "add_generation_prompt": True, **extra}})
    add("user", [user])
    add("system", [{"role": "system", "content": "Будь краток."}, user])
    add("history", [user, {"role": "assistant", "content": "Ответ."}, {"role": "user", "content": "Далее"}])
    add("reasoning_field", [user, {"role": "assistant", "content": "Ответ.", "reasoning_content": "Думаю."}])
    add("reasoning_inline", [user, {"role": "assistant", "content": "<think>\nДумаю.\n</think>\nОтвет."}])
    add("old_reasoning", [user, {"role": "assistant", "content": "<think>hidden</think>shown"}, user])
    add("reasoning_empty", [user, {"role": "assistant", "content": "Ответ", "reasoning_content": ""}])
    add("null_assistant", [user, {"role": "assistant", "content": None}])
    add("mapping_text", [{"role": "user", "content": {"text": "one"}}])
    add("mapping_value", [{"role": "user", "content": {"value": "one", "text": "unused"}}])
    add("text_parts", [{"role": "user", "content": [{"type": "text", "text": "one"}, {"type": "text", "value": "two"}]}])
    add("empty_text", [{"role": "user", "content": ""}])
    add("unicode_whitespace", [{"role": "user", "content": "\r\n\t e\u0301 🧑🏽‍💻 \x00\n"}])
    add("observation", [user, {"role": "system", "name": "observation", "content": "result"}])
    add("later_system", [user, {"role": "system", "content": "additional rules"}])
    add("tools_available", [user], tools=[tool])
    add("tools_system", [{"role": "system", "content": "Правило"}, user], tools=[tool])
    for representation, arguments in [("object", args), ("json", json.dumps(args, ensure_ascii=False))]:
        call = {"id": "call_1", "type": "function", "function": {"name": "weather", "arguments": arguments}}
        assistant = {"role": "assistant", "content": None, "reasoning_content": "Check", "tool_calls": [call]}
        add("call_" + representation, [user, assistant], tools=[tool])
        add("tool_response_" + representation, [user, assistant, {"role": "tool", "tool_call_id": "call_1", "content": "5°C"}], tools=[tool])
        add("tool_responses_" + representation, [user, assistant, {"role": "tool", "content": "one"},
            {"role": "tool", "content": "two"}, {"role": "assistant", "content": "done"}], tools=[tool])
    add("flat_call", [user, {"role": "assistant", "content": "", "tool_calls": [{"name": "weather", "arguments": {}}]}], tools=[tool])
    add("multiple_calls", [user, {"role": "assistant", "content": "", "tool_calls": [
        {"name": "a", "arguments": {"x": 1}}, {"function": {"name": "b", "arguments": {"s": "text"}}}]}], tools=[tool])
    add("user_tool_response", [user, {"role": "assistant", "content": "<think>x</think>y"},
        {"role": "user", "content": "<tool_response>done</tool_response>"}])
    # Every history is checked with and without an assistant generation prefix,
    # and with all documented reasoning levels or no explicit level.
    expanded = []
    for case in cases:
        for generation in (False, True):
            for effort in (None, "low", "medium", "high"):
                ctx = {**case["context"], "add_generation_prompt": generation}
                if effort is not None:
                    ctx["reasoning_effort"] = effort
                expanded.append({"name": f'{case["name"]}/gen={generation}/effort={effort}', "context": ctx})
    return expanded


def compare(gguf, oracle, tokenizer_oracle, runtime_adapter=False):
    provenance = json.loads(subprocess.run([str(oracle), "--version"], check=True, capture_output=True, timeout=15).stdout)
    expected = {"architecture": "step35", "requested_revision": LOADER_SHA,
                "archive_sha256": ARCHIVE_SHA256, "renderer": "native-jinja-with-tool-json-normalization"}
    if provenance != expected:
        raise ValueError("Wrong Step template oracle provenance")
    tokenizer_version = json.loads(subprocess.run([str(tokenizer_oracle), "--version"], check=True, capture_output=True, timeout=15).stdout)
    validate_provenance(tokenizer_version)
    meta = GGUFFile(gguf).metadata
    source = meta["tokenizer.chat_template"]
    if hashlib.sha256(source.encode()).hexdigest() != TEMPLATE_SHA:
        raise ValueError("Unreviewed Step template")
    tokens = meta["tokenizer.ggml.tokens"]
    bos_id = meta["tokenizer.ggml.bos_token_id"]
    bos, eos = tokens[bos_id], tokens[meta["tokenizer.ggml.eos_token_id"]]
    cases = corpus()
    for case in cases:
        case["context"].update(bos_token=bos, eos_token=eos)
    requests = [json.dumps({"template": source, "context": c["context"]}, ensure_ascii=False) for c in cases]
    process = subprocess.run([str(oracle)], input=("\n".join(requests) + "\nQUIT\n").encode(),
                             capture_output=True, timeout=120, check=True)
    responses = [json.loads(line) for line in process.stdout.decode().splitlines()]
    if len(responses) != len(cases):
        raise ValueError("Wrong template oracle response count")
    template, tokenizer = renderer(source), Tokenizer.from_gguf(gguf)
    render = template.render
    if runtime_adapter:
        # Also work when invoked as `python tools/check_step35_template.py`.
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from serve.step35 import StepTemplate
        with tempfile.TemporaryDirectory(prefix="strata-step-template-") as directory:
            path = Path(directory) / "chat_template.jinja"
            path.write_bytes(source.encode("utf8"))
            adapter = StepTemplate(path, bos_token=bos, eos_token=eos)
        def render(**context):
            return adapter.render(**{k:v for k,v in context.items() if k not in ("bos_token", "eos_token")})
    rendered = [render(**c["context"]) for c in cases]
    # Tokenize native renderings independently; compare against Python rendering
    # and Python tokenizer. No add_special: the template already emits BOS.
    token_requests = [json.dumps({"text": r.get("rendered", ""), "parse_special": True}, ensure_ascii=False) for r in responses]
    native = subprocess.run([str(tokenizer_oracle), "--gguf", str(gguf)],
                            input=("\n".join(token_requests) + "\nQUIT\n").encode(),
                            capture_output=True, timeout=120, check=True)
    token_rows = [json.loads(line) for line in native.stdout.decode().splitlines()]
    if len(token_rows) != len(cases) + 1:
        raise ValueError("Wrong tokenizer oracle response count")
    results = []
    for case, py, response, row in zip(cases, rendered, responses, token_rows[1:]):
        ids = tokenizer.encode(py, parse_special=True)
        suffix_ok = not case["context"]["add_generation_prompt"] or py.endswith("<|im_start|>assistant\n<think>\n")
        match = (response.get("rendered") == py and row.get("ids") == ids and
                 row.get("decoded_hex") == py.encode().hex() and ids.count(bos_id) == 1 and ids[0] == bos_id and suffix_ok)
        result = {"name": case["name"], "pass": match, "bytes": len(py.encode()),
                  "render_sha256": hashlib.sha256(py.encode()).hexdigest(), "token_count": len(ids)}
        if not match:
            result.update(python_rendered=py, native=response, python_ids=ids, native_ids=row.get("ids"))
        results.append(result)
    # A hand-written fixture prevents both renderers sharing the same BOS/prefix mistake.
    manual = bos + "<|im_start|>user\nПривет, 你好!<|im_end|>\n<|im_start|>assistant\n<think>\n"
    base = {"messages": [{"role": "user", "content": "Привет, 你好!"}], "tools": None,
            "add_generation_prompt": True, "bos_token": bos, "eos_token": eos}
    results.append({"name": "manual_single_user_prompt", "pass": render(**base) == manual})
    invalid = []
    for value in ("{", "{\"x\":}", "[]", "null", "123", '"text"'):
        ctx = {**base, "messages": [base["messages"][0], {"role": "assistant", "content": "",
               "tool_calls": [{"name": "weather", "arguments": value}]}]}
        invalid.append(ctx)
    invalid_requests = [json.dumps({"template": source, "context": c}, ensure_ascii=False) for c in invalid]
    rejected = subprocess.run([str(oracle)], input=("\n".join(invalid_requests) + "\nQUIT\n").encode(),
                              check=True, capture_output=True, timeout=30)
    errors = [json.loads(line) for line in rejected.stdout.decode().splitlines()]
    if len(errors) != len(invalid):
        raise ValueError("Wrong invalid-request response count")
    for index, (ctx, error) in enumerate(zip(invalid, errors)):
        python_error = False
        try:
            render(**ctx)
        except (ValueError, TypeError, jinja2.TemplateError):
            python_error = True
        results.append({"name": f"reject_invalid_tool_json_{index}", "pass": python_error and "error" in error})
    return {"schema_version": 1, "status": "pass" if all(c["pass"] for c in results) else "fail",
            "scope": "embedded-template byte parity, native token IDs, BOS/generation-prefix; no API or generation",
            "runtime_adapter": runtime_adapter,
            "case_count": len(results), "mismatch_count": sum(not c["pass"] for c in results),
            "gguf": str(gguf), "template_sha256": TEMPLATE_SHA, "jinja2_version": jinja2.__version__,
            "oracle_provenance": provenance, "oracle_sha256": sha(oracle),
            "tokenizer_provenance": tokenizer_version, "tokenizer_oracle_sha256": sha(tokenizer_oracle),
            "cases": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--tokenizer-oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-adapter", action="store_true", help="compare serve.step35.StepTemplate against native Jinja")
    args = parser.parse_args()
    try:
        gguf, oracle, tok, output = first_shard(args.gguf), args.oracle.resolve(), args.tokenizer_oracle.resolve(), args.output.resolve()
        for source in [oracle, tok, *gguf.parent.glob("*.gguf")]:
            if source.resolve() == output or (output.exists() and output.samefile(source)):
                raise ValueError("Output must differ from model and oracle inputs")
        report = compare(gguf, oracle, tok, args.runtime_adapter)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f'{report["status"]}: {report["case_count"]} cases, {report["mismatch_count"]} mismatches; {output}')
        return int(report["status"] != "pass")
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"Step template check failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
