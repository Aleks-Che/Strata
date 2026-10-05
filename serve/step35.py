"""Step-3.7-Flash template/request adapter and incremental output parser.

Uses the reviewed GGUF template and its native EOG vocabulary. Registration is
explicit through a step35 profile; other model families keep their adapters.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

from serve.frontend import (ChatTemplate, Event, OutputParser, TemplateRequestError,
                            ToolCall, _object_list, call_end, param_end)

TEMPLATE_SHA256 = "f428623fc81c940c35be3509fbffc086b4b4360d8800e46103e6f34d02891633"
EFFORTS = ("low", "medium", "high")
THINK_START, THINK_END = "<think>", "</think>"
CALL_START, CALL_END = "<tool_call>", "</tool_call>"
FUNCTION_END, PARAMETER_END = "</function>", "</parameter>"
NAME = re.compile(r"[^\s<>=\x00-\x1f]+\Z")


def _name(value, field):
    if not isinstance(value, str) or not NAME.fullmatch(value):
        raise TemplateRequestError(f"Step {field} must be a nonempty name without whitespace or XML delimiters")
    return value


def _string(value, field):
    if not isinstance(value, str) or not value:
        raise TemplateRequestError(f"Step {field} must be a nonempty string")
    return value


def _constant(value):
    raise ValueError("non-JSON constant: " + value)


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON member: " + key)
        result[key] = value
    return result


def _json(value):
    value = json.loads(value, parse_constant=_constant, object_pairs_hook=_pairs)
    json.dumps(value, allow_nan=False)  # Also reject nested numeric overflow.
    return value


def _arguments(value):
    try:
        value = _json(value) if isinstance(value, str) else deepcopy(value)
        if not isinstance(value, dict):
            raise ValueError("object required")
        for key in value:
            _name(key, "parameter name")
        json.dumps(value, allow_nan=False)
        return value
    except (ValueError, TypeError, OverflowError) as exc:
        raise TemplateRequestError("Step tool arguments must be a finite JSON object with valid parameter names") from exc


def _content(value):
    """Preserve the embedded template's space-separated text blocks."""
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, dict):
        key = "value" if "value" in value else "text"
        if not isinstance(value.get(key), str):
            raise TemplateRequestError("Step text content must contain a string")
        return deepcopy(value)
    result = []
    for part in _object_list(value, "content"):
        key = "value" if "value" in part else "text"
        if part.get("type") not in ("text", "input_text") or not isinstance(part.get(key), str):
            raise TemplateRequestError("Step currently accepts text content only")
        result.append({**deepcopy(part), "type": "text"})
    return result


def _tools(tools):
    result = []
    for tool in _object_list(tools, "tools"):
        function = tool.get("function", tool)
        if tool.get("type", "function") != "function" or not isinstance(function, dict):
            raise TemplateRequestError("Step tools require function objects")
        _name(function.get("name"), "function name")
        if "parameters" in function and not isinstance(function["parameters"], dict):
            raise TemplateRequestError("Step function parameters must be a schema object")
        properties = function.get("parameters", {}).get("properties", {})
        if not isinstance(properties, dict):
            raise TemplateRequestError("Step schema properties must be an object")
        result.append(deepcopy(tool) if "function" in tool else {"type": "function", "function": deepcopy(function)})
    return result or None


def _messages(messages, api=False):
    result = deepcopy(_object_list(messages, "messages"))
    for message in result:
        if api and message.get("role") == "developer":
            message["role"] = "system"
        role = message.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            raise TemplateRequestError("Unsupported Step message role")
        message["content"] = _content(message.get("content"))
        if "reasoning_content" in message and not isinstance(message["reasoning_content"], str):
            raise TemplateRequestError("Step reasoning_content must be a string")
        if api and role == "tool":
            _string(message.get("tool_call_id"), "tool_call_id")
        if message.get("tool_calls"):
            if role != "assistant":
                raise TemplateRequestError("Step tool_calls belong to assistant messages")
            for call in _object_list(message["tool_calls"], "tool_calls"):
                function = call.get("function", call)
                if call.get("type", "function") != "function" or not isinstance(function, dict):
                    raise TemplateRequestError("Step tool_calls require function objects")
                if api:
                    _string(call.get("id"), "tool call id")
                _name(function.get("name"), "function name")
                function["arguments"] = _arguments(function.get("arguments", {}))
    return result


def stop_sequences(req):
    values = req.get("stop", req.get("stop_sequences"))
    if "stop" in req and "stop_sequences" in req:
        raise TemplateRequestError("Use only one of stop or stop_sequences")
    if values is None:
        return []
    values = [values] if isinstance(values, str) else values
    if not isinstance(values, list) or any(not isinstance(s, str) or not s for s in values):
        raise TemplateRequestError("Step stop sequences must be nonempty strings")
    return values


def _options(req, api):
    if not isinstance(req, dict):
        raise TemplateRequestError("Step request must be an object")
    stop_sequences(req)
    # No default effort instruction is invented. Omission keeps the GGUF's
    # default prompt; it still always opens a reasoning block.
    result = {}
    if "enable_thinking" in req or "clear_thinking" in req:
        raise TemplateRequestError("Step uses its embedded thinking/replay policy; use reasoning_effort")
    if req.get("reasoning_budget_tokens") not in (None, 0):
        raise TemplateRequestError("Step hard reasoning budget is not integrated; use reasoning_effort")
    if api == "openai":
        reasoning = req.get("reasoning", {})
        if not isinstance(reasoning, dict):
            raise TemplateRequestError("reasoning must be an object")
        if "effort" in reasoning:
            result["reasoning_effort"] = reasoning["effort"]
        if "reasoning_effort" in req:
            result["reasoning_effort"] = req["reasoning_effort"]
    else:
        thinking = req.get("thinking")
        if thinking is not None:
            if not isinstance(thinking, dict) or thinking.get("type") not in ("enabled", "adaptive"):
                raise TemplateRequestError("Step always opens reasoning; thinking.type must be enabled or adaptive")
            if "budget_tokens" in thinking:
                raise TemplateRequestError("Step does not convert budget_tokens to effort")
        config = req.get("output_config", {})
        if not isinstance(config, dict):
            raise TemplateRequestError("output_config must be an object")
        if "effort" in config:
            result["reasoning_effort"] = config["effort"]
    overrides = req.get("chat_template_kwargs", {})
    if not isinstance(overrides, dict) or set(overrides) - {"reasoning_effort"}:
        raise TemplateRequestError("Step chat_template_kwargs accepts only reasoning_effort")
    result.update(overrides)
    if "reasoning_effort" in result and result["reasoning_effort"] not in EFFORTS:
        raise TemplateRequestError("Step reasoning_effort must be low, medium or high")
    return result


def openai_to_step_messages(req):
    options = _options(req, "openai")
    return _messages(req.get("messages"), api=True), _tools(req.get("tools")), options


def anthropic_to_step_messages(req, think_unasked=True):
    options = _options(req, "anthropic")
    messages = []
    if req.get("system"):
        messages.append({"role": "system", "content": _content(req["system"])})
    for message in _object_list(req.get("messages"), "messages"):
        role, content = message.get("role"), message.get("content")
        if role not in ("user", "assistant"):
            raise TemplateRequestError("Anthropic Step messages must have user or assistant role")
        if content is None or isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        out = {"role": role, "content": []}

        def flush():
            nonlocal out
            if out["content"] or "reasoning_content" in out or "tool_calls" in out:
                messages.append(out)
            out = {"role": role, "content": []}

        for block in _object_list(content, "content"):
            kind = block.get("type")
            if kind == "text":
                if out.get("tool_calls"):
                    flush()  # Keep text following a call after that call.
                out["content"].extend(_content([block]))
            elif kind == "thinking" and role == "assistant":
                if not isinstance(block.get("thinking"), str):
                    raise TemplateRequestError("Step thinking content must be a string")
                if out.get("tool_calls"):
                    flush()
                out["reasoning_content"] = out.get("reasoning_content", "") + block["thinking"]
            elif kind == "tool_use" and role == "assistant":
                out.setdefault("tool_calls", []).append({"id": _string(block.get("id"), "tool use id"),
                    "type": "function", "function": {"name": _name(block.get("name"), "function name"),
                                                     "arguments": _arguments(block.get("input", {}))}})
            elif kind == "tool_result" and role == "user":
                flush()
                result = _content(block.get("content"))
                if block.get("is_error"):
                    if isinstance(result, dict):
                        result = result.get("value", result.get("text"))
                    result = [{"type": "text", "text": "Error:"}, *(
                        result if isinstance(result, list) else [{"type": "text", "text": result or ""}])]
                messages.append({"role": "tool", "tool_call_id": _string(block.get("tool_use_id"), "tool_use_id"),
                                 "content": result})
            else:
                raise TemplateRequestError(f"Unsupported Step {role} content block: {kind}")
        flush()
    tools = [{"type": "function", "function": {"name": _name(t.get("name"), "function name"),
              "description": t.get("description", ""), "parameters": deepcopy(t.get("input_schema", {}))}}
             for t in _object_list(req.get("tools"), "tools")]
    return messages, _tools(tools), options


class StepTemplate(ChatTemplate):
    architecture = "step35"
    reasoning_capabilities = {"efforts": list(EFFORTS), "default": "high",
                              "clear_thinking": False, "replay_reasoning": True}
    supports_reasoning_budget = False
    normalize_openai = staticmethod(openai_to_step_messages)
    normalize_anthropic = staticmethod(anthropic_to_step_messages)

    @staticmethod
    def create_output_parser(thinking, tools, sampling):
        return StepStopParser(thinking=thinking, tools=tools, stops=stop_sequences(sampling or {}))

    @staticmethod
    def resolve_stop_ids(tokenizer):
        """Match the reviewed native vocabulary: EOS + end-of-sentence, never PAD.

        Look up control tokens directly; encoding an absent spelling may produce
        ordinary byte tokens and accidentally terminate on common text.
        """
        if tokenizer.pre != "deepseek-v3":
            raise ValueError("Step requires its exported deepseek-v3 tokenizer")
        ids, tokens, types = tokenizer.ids, tokenizer.tokens, tokenizer.token_types
        spellings = ("<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>", "<|im_end|>", "<｜▁pad▁｜>")
        resolved = []
        for spelling in spellings:
            token_id = ids.get(spelling)
            if (type(token_id) is not int or not 0 <= token_id < len(tokens)
                    or tokens[token_id] != spelling or types is None
                    or token_id >= len(types) or types[token_id] != 3):
                raise ValueError("Step tokenizer missing control token: " + spelling)
            resolved.append(token_id)
        bos, end, eos, pad = resolved
        if len(set(resolved)) != 4:
            raise ValueError("Step control token IDs must be distinct")
        for key, expected in (("bos", bos), ("eos", eos), ("padding", pad)):
            actual = tokenizer.special_ids.get("tokenizer.ggml." + key + "_token_id")
            if type(actual) is not int or actual != expected:
                raise ValueError("Step tokenizer metadata disagrees with " + key + " token")
        for key in ("eot", "eom"):
            actual = tokenizer.special_ids.get("tokenizer.ggml." + key + "_token_id")
            if actual is not None and (type(actual) is not int or actual not in (end, eos)):
                raise ValueError("Unreviewed Step " + key + " terminator")
        return {end, eos}

    def __init__(self, path, *, bos_token, eos_token):
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != TEMPLATE_SHA256:
            raise TemplateRequestError("Unreviewed Step chat template")
        self.bos_token = _string(bos_token, "BOS token")
        self.eos_token = _string(eos_token, "EOS token")
        super().__init__(path)
        # Only this template receives Step's spelling; shared templates stay as-is.
        self.template.environment.filters["fromjson"] = _arguments

    def render(self, messages, tools=None, add_generation_prompt=True, *, reasoning_effort=None, **kwargs):
        if kwargs:
            raise TemplateRequestError("Unsupported Step template options: " + ", ".join(sorted(kwargs)))
        if reasoning_effort is not None and reasoning_effort not in EFFORTS:
            raise TemplateRequestError("Step reasoning_effort must be low, medium or high")
        messages, tools = _messages(messages), _tools(tools)
        if not messages:
            raise TemplateRequestError("Step requires at least one message")
        options = {} if reasoning_effort is None else {"reasoning_effort": reasoning_effort}
        return super().render(messages, tools, add_generation_prompt, bos_token=self.bos_token,
                              eos_token=self.eos_token, **options)


def _value(text, schema):
    # The GGUF emits a framing newline before and after each value, in addition
    # to any newlines belonging to the value itself. Remove at most one each.
    if text.startswith("\n"):
        text = text[1:]
    if text.endswith("\n"):
        text = text[:-1]
    declared = schema.get("type") if isinstance(schema, dict) else None
    types = declared if isinstance(declared, list) else [declared] if declared is not None else []
    if "string" in types:
        return text  # Includes ambiguous nullable strings: preserve the literal.
    try:
        # Step's embedded template uses Jinja string conversion for scalars.
        value = {"True": True, "False": False, "None": None}[text.strip()] if text.strip() in ("True", "False", "None") else _json(text)
    except ValueError:
        if types:
            raise
        return text
    matches = {"null": value is None, "boolean": type(value) is bool,
               "integer": type(value) is int or (type(value) is float and value.is_integer()),
               "number": type(value) in (int, float), "object": isinstance(value, dict), "array": isinstance(value, list)}
    if types and not any(matches.get(t, False) for t in types):
        raise ValueError("Step argument type differs from schema")
    return value


def _parse_call(body, schemas):
    text = body.strip()
    match = re.match(r"<function=([^<>]+)>", text)
    if not match:
        raise ValueError("invalid Step function framing")
    name = _name(match[1], "function name")
    props = ((schemas.get(name) or {}).get("parameters") or {}).get("properties") or {}
    args, pos = {}, match.end()
    while True:
        while text[pos:pos+1].isspace():
            pos += 1
        if text[pos:] == FUNCTION_END:
            return ToolCall(name, args)
        match = re.match(r"<parameter=([^<>]+)>", text[pos:])
        if not match:
            raise ValueError("invalid Step parameter framing or trailing content")
        key = _name(match[1], "parameter name")
        if key in args:
            raise ValueError("duplicate Step parameter")
        start = pos + match.end()
        end = param_end(text[start:])
        if end < 0:
            raise ValueError("missing Step parameter delimiter")
        args[key] = _value(text[start:start+end], props.get(key))
        pos = start + end + len(PARAMETER_END)


class StepOutputParser:
    """Stream text/reasoning; emit a tool only after validating its entire block.

    stream_tools is accepted for Service compatibility. No speculative tool_start
    or tool_args events are emitted. Truncated/malformed blocks stay literal text.
    Service handles EOG token IDs before this parser.
    """
    def __init__(self, thinking=True, tools=None, stream_tools=False):
        self.state = "reasoning" if thinking else "content"
        self.buf, self.initial = "", True
        self.schemas = {t.get("function", t)["name"]: t.get("function", t) for t in _tools(tools) or []}

    def feed(self, delta):
        self.buf += delta
        out = []
        if self.initial:
            if THINK_START.startswith(self.buf) and self.buf != THINK_START:
                return out
            if self.buf.startswith(THINK_START):
                self.buf = self.buf[len(THINK_START):]
                self.state = "reasoning"
            self.initial = False
        while True:
            if self.state == "call":
                at = call_end(self.buf)
                if at < 0:
                    return out
                body, self.buf = self.buf[:at], self.buf[at+len(CALL_END):]
                try:
                    out.append(Event("tool_call", call=_parse_call(body, self.schemas)))
                except (ValueError, TypeError, OverflowError):
                    out.append(Event("content", CALL_START+body+CALL_END))
                self.state = "content"
                continue
            markers = (THINK_END,) if self.state == "reasoning" else (THINK_START, CALL_START)
            positions = [(self.buf.find(tag), tag) for tag in markers if tag in self.buf]
            if positions:
                at, tag = min(positions)
                if at:
                    out.append(Event(self.state, self.buf[:at]))
                self.buf = self.buf[at+len(tag):]
                self.state = {THINK_END: "content", THINK_START: "reasoning", CALL_START: "call"}[tag]
                continue
            end = len(self.buf)-OutputParser._hold(self, self.buf, markers)
            if end:
                out.append(Event(self.state, self.buf[:end]))
                self.buf = self.buf[end:]
            return out

    def finish(self):
        text = (CALL_START if self.state == "call" else "") + self.buf
        kind = "reasoning" if self.state == "reasoning" else "content"
        self.buf, self.state, self.initial = "", "content", False
        return [Event(kind, text)] if text else []


StepTemplate.output_parser = StepOutputParser


class StepStopParser(StepOutputParser):
    """Match raw generated text across token boundaries before interpreting tools."""
    def __init__(self, *, stops, **kwargs):
        super().__init__(**kwargs)
        self.stops, self.pending, self.stop_sequence = stops, "", None

    def feed(self, delta):
        if self.stop_sequence is not None:
            return []
        self.pending += delta
        # Stop at the first completed sequence. Earliest start alone would make
        # ['abcd', 'bc'] choose differently for one big chunk versus characters.
        matches = [(self.pending.find(s)+len(s), i, self.pending.find(s), s)
                   for i, s in enumerate(self.stops) if s in self.pending]
        if matches:
            _, _, at, self.stop_sequence = min(matches)
            text, self.pending = self.pending[:at], ""
        else:
            hold = OutputParser._hold(self, self.pending, self.stops)
            at = len(self.pending) - hold
            text, self.pending = self.pending[:at], self.pending[at:]
        return super().feed(text)

    def finish(self):
        events = super().feed(self.pending)
        self.pending = ""
        return events + super().finish()
