"""Capability-driven settings validation, persistence and API defaults (no GPU)."""
from copy import deepcopy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from serve.frontend import ChatTemplate
from serve.glm5next import GLMTemplate
from serve.server import Service, MockEngine, ByteTokenizer, clean_shared_defaults, make_handler
from serve.test_glm5next_service import Tokenizer


class GLMSettingsTests(unittest.TestCase):
    def setUp(self):
        tok = Tokenizer()
        template = GLMTemplate(Path(__file__).parent / "fixtures/glm53_chat_template.jinja")
        self.svc = Service(MockEngine(tok, ""), tok, template)

    def test_capabilities_and_validation(self):
        caps = self.svc.reasoning_capabilities()
        self.assertEqual(caps, {"efforts": ["low", "high", "max"], "default": "max",
                               "clear_thinking": True, "replay_reasoning": True})
        caps["efforts"].append("none")
        for effort in ("low", "high", "max"):
            self.assertEqual(self.svc.set_shared({"reasoning_effort": effort, "clear_thinking": False}),
                             {"reasoning_effort": effort, "clear_thinking": False})
        for bad in ({"reasoning_effort": "none"}, {"reasoning_effort": "medium"},
                    {"reasoning_effort": "xhigh"}, {"clear_thinking": "false"}, {"clear_thinking": 1}):
            before = dict(self.svc.shared)
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.svc.set_shared(bad)
            self.assertEqual(self.svc.shared, before)

    def test_defaults_reach_both_normalizers(self):
        self.svc.set_shared({"reasoning_effort": "low", "clear_thinking": True})
        for api, req in (("openai", {}), ("anthropic", {}), ("anthropic", {"thinking": {"type": "adaptive"}})):
            with self.subTest(api=api, req=req):
                merged = self.svc.with_shared(req, api)
                self.assertEqual(self.svc.normalize_request(merged, api)[2],
                                 {"reasoning_effort": "low", "clear_thinking": True})

    def test_explicit_request_values_win_without_mutation(self):
        self.svc.set_shared({"reasoning_effort": "max", "clear_thinking": True})
        for api, req in (("openai", {"reasoning_effort": "high", "clear_thinking": False}),
                         ("openai", {"reasoning": {"effort": "high"}, "clear_thinking": False}),
                         ("anthropic", {"output_config": {"effort": "high"}, "clear_thinking": False}),
                         ("anthropic", {"chat_template_kwargs": {"reasoning_effort": "high", "clear_thinking": False}})):
            original = deepcopy(req)
            with self.subTest(api=api, req=req):
                merged = self.svc.with_shared(req, api)
                self.assertEqual(self.svc.normalize_request(merged, api)[2],
                                 {"reasoning_effort": "high", "clear_thinking": False})
                self.assertEqual(req, original)
        # Explicit invalid effort must not become a valid shared default.
        merged = self.svc.with_shared({"reasoning_effort": None}, "openai")
        with self.assertRaises(ValueError):
            self.svc.normalize_request(merged, "openai")

    def test_persistence_and_reload_use_same_capabilities(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shared.json"
            self.svc.shared_path = str(path)
            expected = {"reasoning_effort": "max", "clear_thinking": True, "temperature": 0.4}
            self.svc.set_shared(expected)
            restored = clean_shared_defaults(json.loads(path.read_text()), self.svc.reasoning_capabilities())
            self.assertEqual(restored, expected)
            self.svc.set_shared(None)
            self.assertFalse(path.exists())

    def test_qwen_settings_still_reject_glm_only_controls(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, ""), tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        for value in ("none", "low", "medium", "high"):
            self.assertEqual(svc.set_shared({"reasoning_effort": value}), {"reasoning_effort": value})
        for bad in ({"reasoning_effort": "max"}, {"clear_thinking": False}):
            with self.assertRaises(ValueError):
                svc.set_shared(bad)
        self.assertFalse(svc.reasoning_capabilities()["clear_thinking"])

    def test_settings_post_returns_capabilities_and_validation_error(self):
        handler = object.__new__(make_handler(self.svc))
        handler._own_page = lambda _: True
        handler._json = mock.Mock()
        for defaults, status in (({"reasoning_effort": "max", "clear_thinking": True}, 200),
                                 ({"reasoning_effort": "none"}, 400)):
            body = json.dumps({"defaults": defaults}).encode()
            handler.rfile = io.BytesIO(body)
            handler.headers = {"Content-Length": str(len(body))}
            with contextlib.redirect_stdout(io.StringIO()):
                handler._settings()
            self.assertEqual(handler._json.call_args.args[0], status)
            if status == 200:
                snapshot = handler._json.call_args.args[1]
                self.assertEqual(snapshot["reasoning"]["efforts"], ["low", "high", "max"])
                self.assertEqual(snapshot["defaults"], defaults)


if __name__ == "__main__":
    unittest.main()
