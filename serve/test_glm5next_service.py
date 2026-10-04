"""GLM token boundaries through Service and both API serializers, without HTTP/GPU.

The byte-token stand-in is not a glm4 tokenizer oracle. Stop IDs and spellings
come from the local model's GGUF inventory.
"""
import contextlib
import io
import json
from pathlib import Path
import threading
import unittest
from unittest import mock

from serve.frontend import ChatTemplate
from serve.glm5next import GLMTemplate
from serve.server import (ByteTokenizer, MockEngine, Service, anthropic_collect,
                          anthropic_events, openai_chunks, openai_collect)


FIXTURE = Path(__file__).parent / "fixtures" / "glm53_chat_template.jinja"
STOP_IDS = {"tokenizer.ggml.eos_token_id": 154820, "tokenizer.ggml.eot_token_id": 154827,
            "tokenizer.ggml.eom_token_id": 154829}


class Tokenizer(ByteTokenizer):
    def __init__(self):
        self.special_ids = dict(STOP_IDS)

    def token_bytes(self, token_id):
        if token_id in self.special_ids.values():
            raise AssertionError("GLM stop token reached detokenization")
        return bytes([token_id])


class RecordingEngine(MockEngine):
    def generate(self, *args, **kwargs):
        self.consumed, self.closed = [], False
        try:
            for token_id in super().generate(*args, **kwargs):
                self.consumed.append(token_id)
                yield token_id
        finally:
            self.closed = True


class GLMServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template = GLMTemplate(FIXTURE)

    def service(self, text="", stop=154820):
        tok = Tokenizer()
        engine = RecordingEngine(tok, "")
        engine.script = tok.encode(text) + [stop] + tok.encode("MUST NOT APPEAR")
        svc = Service(engine, tok, self.template, model_name="glm-5.3-flash")
        return svc, engine

    def response(self, api, text, stop, max_new=2048):
        svc, engine = self.service(text, stop)
        args = (svc, {}, [1], True, None, max_new, threading.Event())
        with contextlib.redirect_stdout(io.StringIO()):
            if api == "openai":
                streamed = list(openai_chunks(*args))
                result = openai_collect(streamed)
            else:
                streamed = list(anthropic_events(*args))
                result = anthropic_collect(streamed)
        self.assertTrue(engine.closed)
        return result, streamed, engine

    def test_stop_ids_resolved_without_encoding_qwen_markers(self):
        svc, engine = self.service()
        self.assertEqual(svc.stop_ids, set(STOP_IDS.values()))
        with mock.patch.object(svc.tok, "encode", side_effect=AssertionError("stop markers must use metadata")):
            self.assertEqual(Service(engine, svc.tok, self.template).stop_ids, set(STOP_IDS.values()))

    def test_stop_ids_are_not_hardcoded(self):
        svc, engine = self.service()
        svc.tok.special_ids = {key: value + 10 for key, value in STOP_IDS.items()}
        self.assertEqual(Service(engine, svc.tok, self.template).stop_ids, {154830, 154837, 154839})

    def test_invalid_metadata_fails_at_service_construction(self):
        for key in STOP_IDS:
            for bad in (None, -1, True, "154820", 1.5):
                with self.subTest(key=key, bad=bad):
                    svc, engine = self.service()
                    svc.tok.special_ids[key] = bad
                    with self.assertRaisesRegex(ValueError, key):
                        Service(engine, svc.tok, self.template)
            svc, engine = self.service()
            del svc.tok.special_ids[key]
            with self.assertRaisesRegex(ValueError, key):
                Service(engine, svc.tok, self.template)
        svc, engine = self.service()
        svc.tok.special_ids["tokenizer.ggml.eom_token_id"] = 154820
        with self.assertRaisesRegex(ValueError, "distinct"):
            Service(engine, svc.tok, self.template)

    def test_each_boundary_stops_text_in_both_api_streams(self):
        text = 'reason</think>Привет 🌍'
        for api in ("openai", "anthropic"):
            for stop in STOP_IDS.values():
                with self.subTest(api=api, stop=stop):
                    result, stream, engine = self.response(api, text, stop)
                    self.assertEqual(engine.consumed, list(text.encode()) + [stop])
                    self.assertNotIn("MUST NOT APPEAR", json.dumps(stream))
                    if api == "openai":
                        choice = result["choices"][0]
                        self.assertEqual(choice["finish_reason"], "stop")
                        self.assertEqual(choice["message"]["content"], 'Привет 🌍')
                        self.assertEqual(choice["message"]["reasoning_content"], 'reason')
                        self.assertEqual(stream[-1]["choices"][0]["finish_reason"], "stop")
                        generated = result["usage"]["completion_tokens"]
                    else:
                        self.assertEqual(result["stop_reason"], "end_turn")
                        self.assertEqual([(b["type"], b.get("thinking", b.get("text"))) for b in result["content"]],
                                         [("thinking", "reason"), ("text", 'Привет 🌍')])
                        self.assertEqual(stream[-2][1]["delta"]["stop_reason"], "end_turn")
                        generated = result["usage"]["output_tokens"]
                    self.assertEqual(generated, len(text.encode()) + 1)

    def test_complete_calls_report_handoff_in_both_apis(self):
        text = '</think><tool_call>clock</tool_call><tool_call>search<arg_key>n</arg_key><arg_value>2</arg_value></tool_call>'
        for api in ("openai", "anthropic"):
            for stop in STOP_IDS.values():
                with self.subTest(api=api, stop=stop):
                    result, stream, engine = self.response(api, text, stop)
                    if api == "openai":
                        choice = result["choices"][0]
                        self.assertEqual(choice["finish_reason"], "tool_calls")
                        calls = choice["message"]["tool_calls"]
                        self.assertEqual([(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in calls],
                                         [("clock", {}), ("search", {"n": 2})])
                        self.assertEqual(stream[-1]["choices"][0]["finish_reason"], "tool_calls")
                    else:
                        self.assertEqual(result["stop_reason"], "tool_use")
                        calls = result["content"]
                        self.assertEqual([(c["name"], c["input"]) for c in calls], [("clock", {}), ("search", {"n": 2})])
                        self.assertEqual(stream[-2][1]["delta"]["stop_reason"], "tool_use")
                    self.assertNotEqual(calls[0]["id"], calls[1]["id"])
                    self.assertEqual(engine.consumed[-1], stop)

    def test_eom_does_not_invent_call_for_truncated_or_malformed_output(self):
        for body in ('<tool_call>clock', '<tool_call></tool_call>'):
            for api in ("openai", "anthropic"):
                with self.subTest(api=api, body=body):
                    result, _, _ = self.response(api, '</think>' + body, 154829)
                    if api == "openai":
                        choice = result["choices"][0]
                        self.assertEqual(choice["finish_reason"], "stop")
                        self.assertNotIn("tool_calls", choice["message"])
                        self.assertEqual(choice["message"]["content"], body)
                    else:
                        self.assertEqual(result["stop_reason"], "end_turn")
                        self.assertEqual(result["content"], [{"type": "text", "text": body}])

    def test_length_takes_precedence_even_after_complete_call(self):
        text = '</think><tool_call>clock</tool_call>'
        for api in ("openai", "anthropic"):
            result, _, engine = self.response(api, text, 154829, max_new=len(text))
            self.assertNotIn(154829, engine.consumed)
            if api == "openai":
                self.assertEqual(result["choices"][0]["finish_reason"], "length")
            else:
                self.assertEqual(result["stop_reason"], "max_tokens")

    def test_literal_marker_text_is_not_a_token_boundary(self):
        # Ordinary byte tokens spelling a marker must not be stripped from text.
        text = '<|user|> <|observation|> <|endoftext|>'
        result, _, _ = self.response("openai", '</think>' + text, 154820)
        self.assertEqual(result["choices"][0]["message"]["content"], text)

    def test_qwen_fallback_unchanged(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, ""), tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.assertEqual(svc.stop_ids, {257, 258})


if __name__ == "__main__":
    unittest.main()
