"""DeepSeek V4 chat and DSML output, selected only by architecture=deepseek4.

The GGUF's own template remains authoritative; do not feed it Qwen chat markers.
Tool calls are emitted after a complete DSML block, including typed JSON arguments.
"""
from __future__ import annotations

import json
import re

from serve.frontend import ChatTemplate, Event, OutputParser, ToolCall, images_of


class DeepSeekTemplate(ChatTemplate):
    architecture = "deepseek4"
    stop_token_keys = ("tokenizer.ggml.eos_token_id",)

    def render(self, messages, tools=None, add_generation_prompt=True, **kwargs):
        if images_of(messages):
            raise ValueError("This DeepSeek backend supports text only; no vision encoder is configured")
        kwargs = dict(kwargs)
        kwargs.setdefault("enable_thinking", True)
        # Strata's shared effort normalization uses xhigh for high/max.
        if kwargs.get("reasoning_effort") == "xhigh":
            kwargs["reasoning_effort"] = "high"
        kwargs["bos_token"] = "<｜begin▁of▁sentence｜>"
        # The shared request normalizer unwraps OpenAI functions for Qwen. The
        # DeepSeek GGUF template emits schemas only for type=function wrappers.
        if tools:
            tools = [{"type": "function", "function": t.get("function", t)} for t in tools]
        return super().render(messages, tools, add_generation_prompt, **kwargs)


START = "<｜DSML｜tool_calls>"
END = "</｜DSML｜tool_calls>"
INVOKE = re.compile(r'<｜DSML｜invoke name="([^"<>]+)">(.*?)</｜DSML｜invoke>', re.S)
PARAM = re.compile(r'<｜DSML｜parameter name="([^"<>]+)" string="(true|false)">(.*?)</｜DSML｜parameter>(?=\s*(?:<｜DSML｜parameter|$))', re.S)


def calls_of(body):
    """Malformed or truncated DSML is text, never an executable partial call."""
    calls, pos = [], 0
    for invoke in INVOKE.finditer(body):
        if body[pos:invoke.start()].strip():
            raise ValueError("unexpected text in tool block")
        name, params = invoke.groups()
        args, p = {}, 0
        for param in PARAM.finditer(params):
            if params[p:param.start()].strip():
                raise ValueError("invalid DSML parameter")
            key, string, value = param.groups()
            if key in args:
                raise ValueError("duplicate DSML parameter")
            args[key] = value if string == "true" else json.loads(value)
            p = param.end()
        if params[p:].strip():
            raise ValueError("incomplete DSML parameters")
        calls.append(ToolCall(name, args))
        pos = invoke.end()
    if body[pos:].strip() or not calls:
        raise ValueError("incomplete DSML invokes")
    return calls


class DeepSeekOutputParser:
    def __init__(self, thinking=True, tools=None, stream_tools=False):
        self.state = "reasoning" if thinking else "content"
        self.buf = ""
        self.lead = False

    def feed(self, delta):
        self.buf += delta
        out = []
        while True:
            if self.state == "call":
                at = self.buf.find(END)
                if at < 0:
                    return out
                body, self.buf = self.buf[:at], self.buf[at + len(END):]
                try:
                    out.extend(Event("tool_call", call=c) for c in calls_of(body))
                except (ValueError, TypeError):
                    out.append(Event("content", START + body + END))
                self.state, self.lead = "content", True
                continue
            if self.lead:
                self.buf = self.buf.lstrip("\n")
                if not self.buf:
                    return out
                self.lead = False
            marker = "</think>" if self.state == "reasoning" else START
            at = self.buf.find(marker)
            if at >= 0:
                if at:
                    out.append(Event(self.state, self.buf[:at]))
                self.buf = self.buf[at + len(marker):]
                self.state = "content" if self.state == "reasoning" else "call"
                self.lead = self.state == "content"
                continue
            hold = OutputParser._hold(self, self.buf, (marker,))
            end = len(self.buf) - hold
            if end:
                out.append(Event(self.state, self.buf[:end]))
                self.buf = self.buf[end:]
            return out

    def finish(self):
        if not self.buf and self.state != "call":
            return []
        text, self.buf = self.buf, ""
        return [Event("content" if self.state == "call" else self.state,
                      START + text if self.state == "call" else text)]


DeepSeekTemplate.output_parser = DeepSeekOutputParser
