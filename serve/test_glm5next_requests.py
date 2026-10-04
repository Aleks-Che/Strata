"""GLM API normalization with the real embedded prompt and a mock Service."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest import mock

from serve.frontend import ChatTemplate, anthropic_to_messages, openai_to_messages
from serve.glm5next import GLMTemplate, anthropic_to_glm_messages, openai_to_glm_messages
from serve.server import Service, MockEngine, ByteTokenizer, make_handler
from serve.test_glm5next_service import Tokenizer


FIXTURE = Path(__file__).parent / "fixtures" / "glm53_chat_template.jinja"


class GLMRequestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template = GLMTemplate(FIXTURE)

    def test_openai_ids_arguments_and_late_system_preserved(self):
        req = {"messages": [{"role": "user", "content": [{"type": "text", "text": "Find"}]},
                            {"role": "developer", "content": "Late instruction"},
                            {"role": "assistant", "content": None, "reasoning_content": "r", "tool_calls": [
                                {"id": "a", "type": "function", "function": {"name": "f", "arguments": '{"x":1}'}},
                                {"id": "b", "function": {"name": "g", "arguments": {}}}]},
                            {"role": "tool", "tool_call_id": "b", "content": "B"},
                            {"role": "tool", "tool_call_id": "a", "content": "A"}],
               "tools": [{"type": "function", "function": {"name": "f", "parameters": {}, "strict": True}}]}
        original = deepcopy(req)
        messages, tools, options = openai_to_glm_messages(req)
        self.assertEqual(req, original)
        self.assertEqual(messages[1], {"role": "system", "content": "Late instruction"})
        self.assertEqual(messages[2]["tool_calls"][0]["function"]["arguments"], {"x": 1})
        self.assertEqual(messages[2]["tool_calls"][0]["id"], "a")
        self.assertEqual(messages[3]["tool_call_id"], "b")
        self.assertEqual(tools, req["tools"])
        prompt = self.template.render(messages, tools=tools, **options)
        self.assertIn('<|system|>Late instruction<|assistant|><think>r</think>', prompt)
        self.assertTrue(prompt.endswith('<|observation|><tool_response>A</tool_response>'
                                       '<tool_response>B</tool_response><|assistant|><think>'))
        messages[2]["tool_calls"][0]["function"]["arguments"]["x"] = 5
        tools[0]["function"]["name"] = "changed"
        self.assertEqual(req, original)

    def test_effort_sources_and_override_precedence(self):
        for effort in ("low", "high", "max"):
            requests = [(openai_to_glm_messages, {"reasoning_effort": effort}),
                        (openai_to_glm_messages, {"reasoning": {"effort": effort}}),
                        (anthropic_to_glm_messages, {"output_config": {"effort": effort}})]
            for normalize, req in requests:
                with self.subTest(req=req):
                    messages, tools, options = normalize(req)
                    self.assertEqual(options["reasoning_effort"], effort)
                    self.assertIn("Reasoning Effort: " + effort.capitalize(), self.template.render(messages, tools, **options))
        _, _, options = openai_to_glm_messages({"reasoning": {"effort": "low"}, "reasoning_effort": "high"})
        self.assertEqual(options["reasoning_effort"], "high")
        _, _, options = openai_to_glm_messages({"reasoning_effort": "high", "clear_thinking": False,
                                              "chat_template_kwargs": {"reasoning_effort": "max", "clear_thinking": True}})
        self.assertEqual(options, {"reasoning_effort": "max", "clear_thinking": True})

    def test_unsupported_options_fail_for_both_apis(self):
        for normalize in (openai_to_glm_messages, anthropic_to_glm_messages):
            for req in ({"enable_thinking": False}, {"chat_template_kwargs": {"enable_thinking": True}},
                        {"chat_template_kwargs": {"unknown": 1}}, {"chat_template_kwargs": "bad"},
                        {"clear_thinking": "false"}, {"reasoning_budget_tokens": 50}):
                with self.subTest(normalize=normalize.__name__, req=req), self.assertRaises(ValueError):
                    normalize(req)
            for effort in ("medium", "xhigh", "none", None, False):
                with self.subTest(effort=effort), self.assertRaises(ValueError):
                    normalize({"chat_template_kwargs": {"reasoning_effort": effort}})
        for thinking in ({"type": "disabled"}, False, {"type": "enabled", "budget_tokens": 2000}):
            with self.subTest(thinking=thinking), self.assertRaises(ValueError):
                anthropic_to_glm_messages({"thinking": thinking})

    def test_anthropic_thinking_is_always_on_without_invented_budget_mapping(self):
        for thinking in (None, {"type": "enabled"}, {"type": "adaptive"}):
            _, _, options = anthropic_to_glm_messages({"thinking": thinking}, think_unasked=False)
            self.assertEqual(options, {"reasoning_effort": "max", "clear_thinking": False})

    def test_anthropic_tool_results_and_reasoning_render(self):
        req = {"system": [{"type": "text", "text": "System"}], "messages": [
            {"role": "user", "content": "Find"},
            {"role": "assistant", "content": [{"type": "thinking", "thinking": "reason", "signature": "opaque"},
             {"type": "tool_use", "id": "a", "name": "f", "input": {"x": 1}},
             {"type": "tool_use", "id": "b", "name": "g", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "b", "content": "B"},
             {"type": "tool_result", "tool_use_id": "a", "content": [{"type": "text", "text": "A"}]},
             {"type": "text", "text": "Continue"}]}],
             "tools": [{"name": "f", "input_schema": {"type": "object"}}]}
        original = deepcopy(req)
        messages, tools, options = anthropic_to_glm_messages(req)
        self.assertEqual(req, original)
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "tool", "tool", "user"])
        self.assertEqual(messages[2]["reasoning_content"], "reason")
        self.assertEqual(tools[0]["parameters"], {"type": "object"})
        prompt = self.template.render(messages, tools=tools, **options)
        self.assertTrue(prompt.endswith('<|observation|><tool_response>A</tool_response><tool_response>B</tool_response>'
                                       '<|user|>Continue<|assistant|><think>'))

    def test_mixed_anthropic_text_results_keep_order(self):
        req = {"messages": [{"role": "user", "content": [
            {"type": "text", "text": "Before"}, {"type": "tool_result", "tool_use_id": "a", "content": "Result"},
            {"type": "text", "text": "After"}]}]}
        messages, _, _ = anthropic_to_glm_messages(req)
        self.assertEqual([(m["role"], m["content"]) for m in messages],
                         [("user", "Before"), ("tool", "Result"), ("user", "After")])

    def test_anthropic_tool_error_is_visible_to_model(self):
        messages, tools, options = anthropic_to_glm_messages({"messages": [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "a", "is_error": True, "content": "failed"}]}]})
        self.assertIn('<tool_response>Error: failed</tool_response>', self.template.render(messages, tools, **options))

    def test_clear_thinking_reaches_template(self):
        req = {"messages": [{"role": "assistant", "content": "old answer", "reasoning_content": "old reason"},
                            {"role": "user", "content": "New"}], "chat_template_kwargs": {"clear_thinking": True}}
        messages, tools, options = openai_to_glm_messages(req)
        self.assertNotIn("old reason", self.template.render(messages, tools=tools, **options))
        self.assertEqual(req["messages"][0]["reasoning_content"], "old reason")

    def test_bad_calls_and_content_are_request_errors(self):
        for bad in ([], None, "[]", "invalid JSON"):
            req = {"messages": [{"role": "assistant", "tool_calls": [
                {"id": "a", "function": {"name": "f", "arguments": bad}}]}]}
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                openai_to_glm_messages(req)
        for message in ({"role": "tool", "content": "no id"},
                        {"role": "assistant", "tool_calls": [{"function": {"name": "f"}}]},
                        {"role": "user", "tool_calls": [{"id": "a", "function": {"name": "f"}}]},
                        {"role": "user", "content": [{"type": "image_url", "image_url": "x"}]}):
            with self.subTest(message=message), self.assertRaises(ValueError):
                openai_to_glm_messages({"messages": [message]})
        for block in ({"type": "tool_result", "content": "no id"}, {"type": "image"}, {"type": "unknown"}):
            with self.subTest(block=block), self.assertRaises(ValueError):
                anthropic_to_glm_messages({"messages": [{"role": "user", "content": [block]}]})

    def test_double_encoded_lists_supported(self):
        messages = [{"role": "assistant", "tool_calls": json.dumps([{"id": "a", "function": {"name": "f", "arguments": "{}"}}])}]
        result, _, _ = openai_to_glm_messages({"messages": json.dumps(messages)})
        self.assertEqual(result[0]["tool_calls"][0]["id"], "a")

    def test_service_selects_glm_adapter_and_prepares_prompt(self):
        tok = Tokenizer()
        svc = Service(MockEngine(tok, ""), tok, self.template)
        for api, req in (("openai", {"reasoning_effort": "high"}), ("anthropic", {"output_config": {"effort": "max"}})):
            with self.subTest(api=api):
                normalized = svc.normalize_request(req, api)
                ids, thinking, maximum = svc.prepare(*normalized, max_new=50)
                self.assertTrue(thinking)
                self.assertEqual(maximum, 50)
                self.assertIn("Reasoning Effort: " + normalized[2]["reasoning_effort"].capitalize(), tok.decode(ids))
        with self.assertRaises(ValueError):
            svc.normalize_request({}, "unknown")

    def test_default_dispatch_keeps_existing_normalizers(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, ""), tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        req = {"messages": [{"role": "user", "content": "Hi"}], "reasoning_effort": "high"}
        self.assertEqual(svc.normalize_request(req, "openai"), openai_to_messages(req))
        self.assertEqual(svc.normalize_request(req, "anthropic"), anthropic_to_messages(req))

    def test_count_tokens_handler_uses_glm_adapter_without_http(self):
        tok = Tokenizer()
        svc = Service(MockEngine(tok, ""), tok, self.template)
        handler = object.__new__(make_handler(svc))
        handler._json = mock.Mock()
        handler._count_tokens({"messages": [{"role": "user", "content": "Hi"}], "output_config": {"effort": "high"}})
        expected = '[gMASK]<sop><|system|>Reasoning Effort: High<|user|>Hi<|assistant|><think>'
        handler._json.assert_called_once_with(200, {"input_tokens": len(expected.encode())})


if __name__ == "__main__":
    unittest.main()
