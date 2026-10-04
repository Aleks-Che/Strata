"""GLM-5.3-Flash prompt rendering and streaming reasoning/tool-call parsing.

Rendering uses the GGUF's extracted chat_template.jinja. API adapters retain
tool-call IDs and GLM effort values. This is not a backend registration, and no
fallback prompt is substituted.
"""
from __future__ import annotations

from copy import deepcopy
import json
import re

from serve.frontend import ChatTemplate, Event, OutputParser, TemplateRequestError, ToolCall, _object_list


def _text_content(content):
    """Text-only API content; reject unsupported blocks instead of dropping them."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = _object_list(content, "content")
    text = []
    for part in parts:
        if part.get("type") not in ("text", "input_text") or not isinstance(part.get("text"), str):
            raise TemplateRequestError("GLM currently accepts text content only")
        text.append(part["text"])
    return "".join(text)


def _required_string(value, name):
    if not isinstance(value, str) or not value.strip():
        raise TemplateRequestError(f"GLM {name} must be a nonempty string")
    return value


def _arguments(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError as exc:
            raise TemplateRequestError("GLM tool arguments must be a JSON object") from exc
    if not isinstance(value, dict):
        raise TemplateRequestError("GLM tool arguments must be a JSON object")
    return deepcopy(value)


def _request_options(req, api):
    options = {"reasoning_effort": "max", "clear_thinking": False}
    if "enable_thinking" in req:
        raise TemplateRequestError("GLM does not support enable_thinking; use reasoning_effort")
    if req.get("reasoning_budget_tokens"):
        raise TemplateRequestError("GLM budget control is not supported; use reasoning_effort")
    if api == "openai":
        reasoning = req.get("reasoning") or {}
        if not isinstance(reasoning, dict):
            raise TemplateRequestError("reasoning must be an object")
        if "effort" in reasoning:
            options["reasoning_effort"] = reasoning["effort"]
        if "reasoning_effort" in req:
            options["reasoning_effort"] = req["reasoning_effort"]
    else:
        thinking = req.get("thinking")
        if thinking is not None:
            if not isinstance(thinking, dict) or thinking.get("type") not in ("enabled", "adaptive"):
                raise TemplateRequestError("GLM always uses reasoning; thinking.type must be enabled or adaptive")
            if "budget_tokens" in thinking:
                raise TemplateRequestError("GLM does not map budget_tokens to effort; use output_config.effort")
        config = req.get("output_config") or {}
        if not isinstance(config, dict):
            raise TemplateRequestError("output_config must be an object")
        if "effort" in config:
            options["reasoning_effort"] = config["effort"]
    if "clear_thinking" in req:
        options["clear_thinking"] = req["clear_thinking"]
    overrides = req.get("chat_template_kwargs")
    if overrides is not None:
        if not isinstance(overrides, dict) or set(overrides) - options.keys():
            raise TemplateRequestError("GLM chat_template_kwargs accepts only reasoning_effort and clear_thinking")
        options.update(overrides)
    if options["reasoning_effort"] not in ("low", "high", "max"):
        raise TemplateRequestError("GLM reasoning_effort must be low, high or max")
    if type(options["clear_thinking"]) is not bool:
        raise TemplateRequestError("GLM clear_thinking must be a boolean")
    return options


def openai_to_glm_messages(req):
    """Normalize GLM requests before Qwen's normalizer can lose IDs/effort."""
    options = _request_options(req, "openai")
    messages = []
    for message in _object_list(req.get("messages"), "messages"):
        role = message.get("role")
        role = "system" if role == "developer" else role
        if role not in ("system", "user", "assistant", "tool"):
            raise TemplateRequestError("Unsupported GLM message role")
        out = {"role": role, "content": _text_content(message.get("content"))}
        if "reasoning_content" in message:
            if not isinstance(message["reasoning_content"], str):
                raise TemplateRequestError("reasoning_content must be a string")
            out["reasoning_content"] = message["reasoning_content"]
        if role == "tool":
            out["tool_call_id"] = _required_string(message.get("tool_call_id"), "tool_call_id")
        if message.get("tool_calls"):
            if role != "assistant":
                raise TemplateRequestError("GLM tool_calls belong to assistant messages")
            calls = []
            for call in _object_list(message["tool_calls"], "tool_calls"):
                function = call.get("function", call)
                if not isinstance(function, dict) or call.get("type", "function") != "function":
                    raise TemplateRequestError("GLM tool_calls require function objects")
                calls.append({"id": _required_string(call.get("id"), "tool call id"), "type": "function",
                              "function": {"name": _required_string(function.get("name"), "function name"),
                                           "arguments": _arguments(function.get("arguments", {}))}})
            out["tool_calls"] = calls
        messages.append(out)
    tools = deepcopy(_object_list(req.get("tools"), "tools"))
    for tool in tools:
        function = tool.get("function", tool)
        if not isinstance(function, dict) or tool.get("type", "function") != "function":
            raise TemplateRequestError("GLM tools require function objects")
        _required_string(function.get("name"), "function name")
    # GLM permits system messages anywhere; do not apply Qwen's late-system rewrite.
    return messages, tools or None, options


def anthropic_to_glm_messages(req, think_unasked=True):
    """Anthropic blocks -> GLM history. GLM always reasons, regardless of opt-in defaults."""
    options = _request_options(req, "anthropic")
    messages = []
    if req.get("system"):
        messages.append({"role": "system", "content": _text_content(req["system"])})
    for message in _object_list(req.get("messages"), "messages"):
        role = message.get("role")
        if role not in ("user", "assistant"):
            raise TemplateRequestError("Anthropic GLM messages must have user or assistant role")
        content = message.get("content")
        if isinstance(content, str) or content is None:
            messages.append({"role": role, "content": _text_content(content)})
            continue
        out = {"role": role, "content": ""}
        for block in _object_list(content, "content"):
            kind = block.get("type")
            if kind == "text":
                out["content"] += _text_content([block])
            elif kind == "thinking" and role == "assistant":
                value = block.get("thinking")
                if not isinstance(value, str):
                    raise TemplateRequestError("thinking content must be a string")
                out["reasoning_content"] = out.get("reasoning_content", "") + value
            elif kind == "tool_use" and role == "assistant":
                out.setdefault("tool_calls", []).append({
                    "id": _required_string(block.get("id"), "tool use id"),
                    "function": {"name": _required_string(block.get("name"), "function name"),
                                 "arguments": _arguments(block.get("input", {}))}})
            elif kind == "tool_result" and role == "user":
                # Preserve mixed text/result order, and consecutive result blocks
                # so the embedded template can sort those by the preceding call IDs.
                if out["content"]:
                    messages.append(out)
                    out = {"role": role, "content": ""}
                result = _text_content(block.get("content"))
                if block.get("is_error"):
                    result = "Error: " + result
                messages.append({"role": "tool", "tool_call_id": _required_string(block.get("tool_use_id"), "tool_use_id"),
                                 "content": result})
            else:
                raise TemplateRequestError(f"Unsupported GLM {role} content block: {kind}")
        if out["content"] or "reasoning_content" in out or "tool_calls" in out:
            messages.append(out)
    tools = []
    for tool in _object_list(req.get("tools"), "tools"):
        tools.append({"name": _required_string(tool.get("name"), "function name"),
                      "description": tool.get("description", ""), "parameters": deepcopy(tool.get("input_schema", {}))})
    return messages, tools or None, options


class GLMTemplate(ChatTemplate):
    architecture = "glm5next"
    reasoning_capabilities = {"efforts": ["low", "high", "max"], "default": "max",
                              "clear_thinking": True, "replay_reasoning": True}
    normalize_openai = staticmethod(openai_to_glm_messages)
    normalize_anthropic = staticmethod(anthropic_to_glm_messages)
    # EOS ends text; EOT starts the user's turn; EOM hands off to a tool.
    # Resolve from this model's metadata, never by encoding Qwen delimiters.
    stop_token_keys = ("tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id",
                       "tokenizer.ggml.eom_token_id")

    def render(self, messages, tools=None, add_generation_prompt=True, *,
               reasoning_effort="max", clear_thinking=False, **kwargs):
        # The embedded template silently maps unsupported effort values to max.
        # Reject them here so an API adapter cannot accidentally change a request.
        if reasoning_effort not in ("low", "high", "max"):
            raise TemplateRequestError("GLM reasoning_effort must be low, high or max")
        if not isinstance(clear_thinking, bool):
            raise TemplateRequestError("GLM clear_thinking must be a boolean")
        if kwargs:
            raise TemplateRequestError(
                "Unsupported GLM template options: " + ", ".join(sorted(kwargs))
                + "; use reasoning_effort and clear_thinking")

        # GLM's template needs mapping arguments and uses IDs to reorder tool
        # results. Preserve the caller's history, wrappers, IDs and result order.
        messages = deepcopy(messages)
        for message in messages:
            if message.get("content") is None:
                message["content"] = ""
            for call in message.get("tool_calls") or []:
                function = call.get("function", call)
                function["arguments"] = _arguments(function.get("arguments", {}))

        return super().render(messages, tools, add_generation_prompt,
                              reasoning_effort=reasoning_effort, clear_thinking=clear_thinking)


THINK_START, THINK_END = "<think>", "</think>"
CALL_START, CALL_END = "<tool_call>", "</tool_call>"
KEY_START = "<arg_key>"
VALUE_START, VALUE_END = "<arg_value>", "</arg_value>"
ARGUMENT = re.compile(
    r"<arg_key>([^<>]+)</arg_key>\s*<arg_value>(.*?)</arg_value>(?=\s*(?:<arg_key>|</tool_call>))",
    re.S)


def _value_end(text, start):
    # GLM emits string values verbatim. A literal closing tag in a value is not
    # structural unless followed by the next argument or the call's end.
    at = text.find(VALUE_END, start)
    while at >= 0:
        after = text[at + len(VALUE_END):].lstrip()
        if after.startswith((KEY_START, CALL_END)):
            return at
        at = text.find(VALUE_END, at + len(VALUE_END))
    return -1


def _call_end(text):
    """Find the closing call tag outside raw argument values; -1 if incomplete."""
    pos = 0
    while True:
        end = text.find(CALL_END, pos)
        value = text.find(VALUE_START, pos)
        if value < 0 or (end >= 0 and end < value):
            return end
        close = _value_end(text, value + len(VALUE_START))
        if close < 0:
            return -1
        pos = close + len(VALUE_END)


def _reject_constant(value):
    raise ValueError("non-JSON constant: " + value)


def _parse_call(body, schemas):
    """Parse an entire GLM call or fail without emitting executable fragments."""
    text = body.strip() + CALL_END
    name_end = text.find("<")
    name = text[:name_end].strip()
    if not name or any(c.isspace() or c in "<>" for c in name):
        raise ValueError("invalid GLM tool name")
    props = ((schemas.get(name) or {}).get("parameters") or {}).get("properties") or {}
    args, pos = {}, name_end
    while not text[pos:].startswith(CALL_END):
        match = ARGUMENT.match(text, pos)
        if match is None:
            raise ValueError("invalid GLM argument framing")
        key, value = match.groups()
        if key in args:
            raise ValueError("duplicate GLM argument")
        declared = (props.get(key) or {}).get("type")
        if declared == "string":
            args[key] = value
        else:
            try:
                args[key] = json.loads(value, parse_constant=_reject_constant)
                # Also rejects numeric overflow such as 1e400, including nested values.
                json.dumps(args[key], allow_nan=False)
            except ValueError:
                if declared is not None:
                    raise
                args[key] = value
        pos = match.end()
        while text[pos:pos + 1].isspace():
            pos += 1
    if text[pos:] != CALL_END:
        raise ValueError("unexpected text after GLM arguments")
    return ToolCall(name, args)


class GLMOutputParser:
    """Streaming reasoning/text and complete GLM calls using frontend events.

    The prompt already ends in <think>, so thinking=True starts inside reasoning.
    A repeated opening tag at the start is accepted for full-text fixtures too.
    Calls are buffered until validated, also with stream_tools=True; no partial
    argument events are emitted. Truncated/malformed calls are returned as text.
    Token-level EOS/EOT/EOM handling belongs to the backend, not this parser.
    """

    def __init__(self, thinking=True, tools=None, stream_tools=False):
        self.state = "reasoning" if thinking else "content"
        self.buf = ""
        self.initial = True
        self.schemas = {}
        for tool in tools or []:
            function = tool.get("function", tool)
            self.schemas[function.get("name")] = function

    def feed(self, delta):
        self.buf += delta
        out = []
        if self.initial:
            if THINK_START.startswith(self.buf):
                if self.buf != THINK_START:
                    return out
            if self.buf.startswith(THINK_START):
                self.buf = self.buf[len(THINK_START):]
                self.state = "reasoning"
            self.initial = False
        while True:
            if self.state == "call":
                at = _call_end(self.buf)
                if at < 0:
                    return out
                body, self.buf = self.buf[:at], self.buf[at + len(CALL_END):]
                try:
                    out.append(Event("tool_call", call=_parse_call(body, self.schemas)))
                except (ValueError, TypeError):
                    out.append(Event("content", CALL_START + body + CALL_END))
                self.state = "content"
                continue
            markers = (THINK_END,) if self.state == "reasoning" else (THINK_START, CALL_START)
            positions = [(self.buf.find(tag), tag) for tag in markers if tag in self.buf]
            if positions:
                at, tag = min(positions)
                if at:
                    out.append(Event(self.state, self.buf[:at]))
                self.buf = self.buf[at + len(tag):]
                self.state = {THINK_END: "content", THINK_START: "reasoning", CALL_START: "call"}[tag]
                continue
            end = len(self.buf) - OutputParser._hold(self, self.buf, markers)
            if end:
                out.append(Event(self.state, self.buf[:end]))
                self.buf = self.buf[end:]
            return out

    def finish(self):
        text = (CALL_START if self.state == "call" else "") + self.buf
        kind = "reasoning" if self.state == "reasoning" else "content"
        self.buf, self.state, self.initial = "", "content", False
        return [Event(kind, text)] if text else []


GLMTemplate.output_parser = GLMOutputParser
