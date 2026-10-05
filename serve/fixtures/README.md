# GLM prompt fixture

`glm53_chat_template.jinja` is the exact UTF-8 value of `tokenizer.chat_template`
extracted on 2026-10-04 with `tools.gguf_reader.GGUFFile` from:

`H:/GLM-5.3-Flash-GGUF/UD-Q3_K_XL/GLM-5.3-Flash-UD-Q3_K_XL-00001-of-00004.gguf`

- Architecture: `glm5next`.
- Length: 10648 bytes.
- SHA-256: `a4fddbbf0b432101a296c17094f8bc5a2b0d30713b5b5cd92f86be78511aa724`.
- Preserve bytes and line endings; do not reformat this fixture.

Extraction from the repository root (Python):

```python
from pathlib import Path
from tools.gguf_reader import GGUFFile

gguf = GGUFFile(Path("H:/GLM-5.3-Flash-GGUF/UD-Q3_K_XL/GLM-5.3-Flash-UD-Q3_K_XL-00001-of-00004.gguf"))
Path("serve/fixtures/glm53_chat_template.jinja").write_bytes(
    gguf.metadata["tokenizer.chat_template"].encode("utf-8"))
```

`serve.test_glm5next` checks handwritten expected prompts against this template,
including branches using Jinja `break`. These are template tests, not tokenizer
ID parity or generated-output comparisons with an inference oracle.

`serve.glm5next.GLMTemplate` accepts the path to a model's extracted template,
messages retaining tool IDs, wrapped or bare tool schemas, and GLM settings
`reasoning_effort=low/high/max`, `clear_thinking=True/False`. It converts tool
argument JSON objects without changing caller-owned history. It rejects Qwen
template settings instead of silently mapping them to GLM's default max effort.

Service uses `GLMTemplate.normalize_openai` / `normalize_anthropic` when explicitly
constructed with this template. Both generation handlers and the Anthropic count
handler dispatch through Service. Runtime selection of the GLM backend/template
and tokenizer support remain separate work. The shared Qwen normalizer drops tool
IDs and merges high/max into xhigh; it must not precede the GLM adapter. Setup must
extract the selected model's own template; this fixture is not a runtime fallback.

The GLM request adapters accept OpenAI `reasoning_effort` (or `reasoning.effort`)
and Anthropic `output_config.effort`, preserving low/high/max exactly. Defaults
are max and `clear_thinking=false`. `chat_template_kwargs` may override either
`reasoning_effort` or `clear_thinking` for both APIs; a top-level `clear_thinking`
is also accepted. OpenAI's top-level effort overrides `reasoning.effort`.
Other template options, unsupported efforts, disabled thinking and budget-to-effort
conversion are errors. Anthropic enabled/adaptive thinking is accepted without a
budget; GLM always reasons, including when the shared opt-in default is false.

OpenAI tool-call/result IDs and Anthropic tool-use/result IDs survive normalization.
Anthropic mixed text/result blocks retain order; `is_error` results receive an
`Error: ` prefix. Late OpenAI system/developer messages remain system messages.
Request data is not mutated. This adapter accepts text only and rejects unknown
or media blocks rather than silently dropping them. Native tool references
are not yet integrated.
`serve.test_glm5next_requests` verifies normalization, rendering and Service/count
handler dispatch without a listening HTTP server or a real GLM tokenizer.

`GLMTemplate.output_parser` selects `GLMOutputParser`. The parser accepts partial
text chunks and returns the existing `reasoning`, `content`, and `tool_call`
events. It starts inside reasoning by default because the prompt ends in
`<think>`. A leading `<think>` is also accepted when parsing complete text.
Tools may have bare or OpenAI-wrapped schemas. Declared string arguments retain
their exact text; other arguments are JSON-decoded, with a raw-string fallback
only when no type is declared. This is not full JSON Schema validation. Supply
schemas to disambiguate string arguments such as `true`, `123`, or `null`.

Calls are buffered and validated in full, even with `stream_tools=True`; early
`tool_start`/`tool_args` events are not implemented. Malformed or truncated calls
are returned as content, with no executable call event. A literal closing tag in
a string is supported unless it forms a structural delimiter sequence:
`</arg_value>` followed by whitespace and `<arg_key>` or `</tool_call>`. The
embedded format emits strings without escaping, so that sequence is ambiguous.
`finish()` flushes incomplete data and can be called repeatedly without duplicates.

The parser tests check every two-chunk split and single-character streaming,
typed arguments, malformed/truncated calls, literal tags in values, multiple
calls, and a template/call/result/continuation round trip. API dispatch and
full-model output validation are separate integration work.

`GLMTemplate.stop_token_keys` declares all three GGUF stop metadata keys. Service
resolves them to distinct nonnegative integer IDs before generation, with no Qwen
marker encoding. In the local GGUF these are EOS 154820 (`<|endoftext|>`), EOT
154827 (`<|user|>`) and EOM 154829 (`<|observation|>`). These IDs stop generation
before detokenization and count as generated tokens. Both API serializers use
validated call events to choose `tool_calls`/`tool_use`; EOM alone does not create
a call. A token limit remains `length`/`max_tokens` even after a complete call.

`serve.test_glm5next_service` checks this behavior through Service and both stream
serializers/collectors with a byte-token mock, not through HTTP or real glm4 BPE.
The future backend must deliver the terminal token ID to Service (or introduce
an explicit completion-reason protocol); silently ending its stream would leave
Service's default `length` reason. Full-model verification remains required.

GLM declares reasoning capabilities (low/high/max, default max, clear_thinking).
Service exposes them via `/health` and `/settings`, validates shared settings
against them, and uses the same validation when restoring saved settings. Explicit
request values, including `clear_thinking=false`, override shared defaults.
The web drawer hides Off/Medium for GLM, shows Max and Clear earlier thinking,
and sends these controls in chat requests and shared settings. With no saved
settings it defaults to max; an unsupported saved effort migrates to the model's
default. Qwen and older servers retain the previous effort controls.

Checks: `python -m unittest serve.test_glm5next_settings` and
`node serve/test_glm5next_settings_ui.cjs`. The Node check executes the app's
settings functions with a small DOM stand-in; it is not a browser visual test.

The built-in MCP loop preserves each parsed call ID in both assistant tool_calls
and tool results before rendering the continuation. `serve.test_glm5next_mcp`
checks multiple calls/rounds with GLM's embedded template, result reordering by
IDs, cancellation, round/token limits, client-owned tools and Qwen compatibility.
The hub is an in-memory stand-in, and the engine replays token scripts; no real
MCP service or full model is exercised. `node serve/test_glm5next_mcp_ui.cjs`
replays stream events through the existing web history functions, including saved
chat restoration, skipped calls and unfinished calls. The web history already
preserved IDs; the missing IDs were in the server's continuation history.

GLM also declares `replay_reasoning=true`. For that capability, web history sends
`reasoning_content` for ordinary answers and for each MCP round, using the saved
reasoning offsets (`rat`). The final continuation can contain reasoning alone.
Older MCP records with missing/invalid/decreasing offsets keep text and calls but
omit reasoning rather than guessing its round. Models without this capability
retain their previous history behavior. The model's template applies
`clear_thinking`; the web app does not erase stored thoughts when the toggle changes.

The Node history test exports its actual replay with `--history`. Python tests
feed that output through GLM normalization and the embedded template, verifying
that clear_thinking retains the current tool turn and clears only prior turns
after a new user message. These two cross-language tests require Node and are
skipped when Node is unavailable; no real tokenizer or model is exercised.

Service's `engine_facts()` forwards reported INFO fields to metrics and adds the
frontend identity/reasoning capabilities. Architecture comes from INFO when
present, otherwise from the selected template. The template alone never implies
GPU computation, expert caching, pipelining or MTP execution. Model alias, active
context size and image availability come from Service, not stale INFO fields.

About/Monitor format `expert_compute`, `expert_storage`, `gpu_expert_layers`,
`expert_pipeline`, pipeline slots/readers/read mode and cache policy whenever
reported, independently of architecture. Missing values stay unknown; zero is
displayed. Matrix counts take precedence over legacy expert-slot counts.
`expert_cache_live_bytes` is used memory; `expert_cache_mib` is a budget.
Speculation uses explicit `speculative=none/dspark/mtp`; native MTP may report
`draft_tokens` for the actual draft count. Legacy Qwen `mtp_max` retains its
verify-window interpretation (window size minus the main token). A bare positive
`spec` without a strategy or legacy `mtp_max` does not establish MTP support.

Checks: `python -m unittest serve.test_glm5next_info` and
`node serve/test_glm5next_info_ui.cjs`. These use reported-field fixtures; no
new GLM engine INFO emission or real GPU behavior has been implemented here.

`python -m unittest serve.test_glm5next_handlers` exercises actual `do_POST`
dispatch, JSON/SSE writers, option errors and client tool-result continuation
for both APIs. A local mock hub checks that client tools win name collisions
with MCP, including OpenAI schemas wrapped in `function`; noncolliding MCP
tools still execute. Streams and response headers are captured in memory.
The engine and tokenizer are stand-ins; no HTTP listener, disconnect watcher,
external MCP server, real tokenizer or GPU is exercised.

Handler checks also cover cancellation before generation and while waiting for
the engine lock. Service emits a zero-output `done/cancel` after releasing the
lock, so both API formats terminate and an MCP continuation cannot reuse the
previous round's token count. Such a request never starts the engine or adds
work to usage statistics. Scripted BrokenPipeError during prefill, reasoning and
an incomplete tool call verifies cleanup with monitoring on/off and a subsequent
successful request. The engine generator closes while the engine lock is held;
incomplete tools are not executed. These are in-memory disconnect simulations,
not socket watcher or native engine STOP/drain validation.

## Step-3.7-Flash fixture and adapter

`step37_chat_template.jinja` is the exact 5723-byte UTF-8 GGUF template extracted
on 2026-10-05 from
`H:/models/Step-3.7-Flash/UD-Q4_K_S/Step-3.7-Flash-UD-Q4_K_S-00001-of-00004.gguf`.
Its SHA-256 is `f428623fc81c940c35be3509fbffc086b4b4360d8800e46103e6f34d02891633`.
Do not reformat it. It is a test fixture, not a runtime fallback.

`serve.step35.StepTemplate` takes the extracted template path and the model's
`bos_token`/`eos_token` explicitly. It checks the template hash, uses Step's
`fromjson` filter, converts tool argument JSON objects without mutating history,
and wraps bare function schemas. Text blocks retain the embedded template's
space separator. Unsupported media blocks are rejected. The shared Qwen and
GLM renderers are unchanged.

OpenAI/Anthropic normalizers preserve call/result IDs and reasoning. Low, medium
and high effort are passed literally; no default effort instruction is inserted
when absent. The prompt still ends in `<think>`. Disabled thinking, hard budgets,
`clear_thinking` and unrelated template overrides are rejected. Step's template
automatically removes reasoning before the last real user query and replays the
current tool turn. It keeps tool results in arrival order; IDs do not cause the
GLM-style reordering. Anthropic text after a tool-use block starts a new assistant
message so that block order survives normalization.

`StepOutputParser` streams reasoning/content and buffers each tool block until
its closing `</tool_call>`. It emits only complete validated `tool_call` events,
including with `stream_tools=True`; early tool-name/argument events are not
implemented. Malformed/truncated blocks remain exact text. XML values lose at
most one framing newline on either side. Declared strings retain their content,
including JSON-looking strings. Other declared base types must match decoded
values. Step's native scalar spellings `True`, `False`, `None` and their JSON
equivalents are supported; nested arrays/objects use JSON. Duplicate XML arguments are rejected. Request argument objects and declared
JSON values reject duplicate JSON members. This is base-type conversion, not full
JSON Schema validation of required fields, ranges, `$ref`, etc. Without a declared
type, non-JSON values remain strings; a union containing `string` also preserves
the literal because the raw format cannot distinguish all nullable strings.

Raw strings are not XML-escaped by the embedded template. Literal tags are kept
unless they form its structural delimiter: `</parameter>` followed by whitespace
and `<parameter=` or `</function>`. That sequence is inherently ambiguous.
Tool markup inside reasoning never becomes a call. `finish()` flushes held text
and is idempotent.

Checks: `python -m unittest serve.test_step35`; native template/token-ID comparison
via `tools/check_step35_template.py --runtime-adapter`. STEP-09 tested 215 native
comparisons, 3415 parser fragmentation sequences, and 148 total tests including
Qwen/GLM/DeepSeek regressions. A real model smoke produced one typed weather call
and a final answer after a local result stub; no external tool was invoked.
All three actual prompts matched the native template/tokenizer oracles, and the
P1 control retained exact F32 logits. Final adapter sources also replayed those
saved prompts/outputs after input-validation hardening.

STEP-10 registers this module only for explicit `architecture: step35` profiles.
`tools/prepare_step35_profile.py` exports a new isolated profile; setup's default
model choice is unchanged. EOG resolves directly from reviewed control-token
spellings and metadata (native 1/128007, PAD2 excluded). Missing or altered
control tokens/templates are rejected before engine startup.

`serve.test_step35_http` covers both APIs over loopback JSON/SSE, Unicode,
EOG/PAD, stop strings, length/tool finish reasons, a local MCP stub, settings and
disconnects with a clean subsequent request. Step's stop filter matches raw
generated text across chunks, before tool parsing. Anthropic returns the matched
`stop_sequence`; OpenAI reports `stop`. Hard thinking budgets are unsupported;
low/medium/high effort changes the prompt, not a guaranteed thinking-token cap.
Browser interaction and external MCP cancellation remain separate gates.
