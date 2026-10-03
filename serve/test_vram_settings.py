"""Live VRAM policy validation, persistence, HTTP access and pipe isolation."""
import io
import json
import queue
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, StrataEngine, serve
from serve.vram_settings import VramSettings

VALID = {"mode": "count", "matrices": 200, "target_mib": 24576, "reserve_mib": 1024}


class SettingsTests(unittest.TestCase):
    def test_validate_bounds_and_reject_pipe_injection(self):
        for key, values in {"mode": [None, "vram\nQUIT", 0], "matrices": [-1, True, 1.5, 1000001, "1\nQUIT"],
                            "target_mib": [-1, 1048577, False, float("nan")],
                            "reserve_mib": [0, 127, float("inf"), 1048577]}.items():
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    VramSettings.validate({**VALID, key: value})
        with self.assertRaises(ValueError):
            VramSettings.validate({**VALID, "mode": "vram", "target_mib": 0})
        self.assertEqual(VramSettings.validate({**VALID, "matrices": 0})["matrices"], 0)

    def test_atomic_persistence_and_failed_save_retains_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            settings = VramSettings(path)
            settings.update(VALID)
            restored = VramSettings(path)
            self.assertEqual(restored.value, VALID)
            self.assertNotEqual(restored.revision, settings.revision)
            revision = settings.revision
            with patch("serve.vram_settings.os.replace", side_effect=OSError("disk full")), self.assertRaises(OSError):
                settings.update({**VALID, "matrices": 0})
            self.assertEqual(settings.value, VALID)
            self.assertEqual(settings.revision, revision)
            self.assertEqual(json.loads(path.read_text()), VALID)
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_bad_saved_settings_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            path.write_text("[]")
            settings = VramSettings(path)
            self.assertIsNotNone(settings.storage_error)
            self.assertEqual(settings.wire(), "")

    def test_control_status_never_enters_token_queue(self):
        engine = StrataEngine.__new__(StrataEngine)
        engine.lines, engine.cache_replies = queue.Queue(), queue.Queue()
        engine.proc = SimpleNamespace(stdout=io.StringIO(
            'VRAM_STATUS {"applied":"a1","cached_matrices":12}\nT 9\nDONE 1 2 3 4 stop\n'))
        engine._pump()
        self.assertEqual(engine.vram_status["cached_matrices"], 12)
        self.assertEqual(list(engine.lines.queue), ['T 9\n', 'DONE 1 2 3 4 stop\n', None])

    def test_policy_updates_current_pipe_and_restart_environment(self):
        engine = StrataEngine.__new__(StrataEngine)
        engine.stdin_lock = threading.RLock()
        engine.spawn = ("engine", [], None, None, {"PATH": "original"})
        engine.proc = SimpleNamespace(stdin=io.StringIO(), poll=lambda: None)
        engine.ended = False
        engine.set_vram_policy("abc 1 200 0 1024")
        self.assertEqual(engine.proc.stdin.getvalue(), "VRAM_SET abc 1 200 0 1024\n")
        self.assertEqual(engine.spawn[-1], {"PATH": "original", "STRATA_VRAM_POLICY": "abc 1 200 0 1024"})
        engine.ended = True
        engine.set_vram_policy("def 1 0 0 1024")
        self.assertEqual(engine.spawn[-1]["STRATA_VRAM_POLICY"], "def 1 0 0 1024")
        self.assertNotIn("def", engine.proc.stdin.getvalue())


class VramHTTP(unittest.TestCase):
    def setUp(self):
        tok = ByteTokenizer()
        self.engine = MockEngine(tok, "ok")
        self.engine.can_vram_control = True
        self.engine.alive = lambda: True
        self.sent = []
        self.engine.set_vram_policy = self.sent.append
        self.svc = Service(self.engine, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/vram/settings"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def request(self, body=None, headers=None):
        req = urllib.request.Request(self.url, None if body is None else json.dumps(body).encode(),
                                     {"Content-Type": "application/json", **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=3) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as error:
            return error.code, json.load(error)

    def test_save_while_generating_then_acknowledge(self):
        self.svc.status["busy"] = True
        with self.svc.fifo:
            code, data = self.request(VALID)
        self.assertEqual(code, 200)
        self.assertTrue(data["pending"])
        self.assertEqual(len(self.sent), 1)
        self.engine.vram_status = {"applied": self.svc.vram_settings.revision, "cached_matrices": 42}
        self.assertFalse(self.request()[1]["pending"])
        self.assertEqual(self.svc.metrics()["vram"]["live"]["cached_matrices"], 42)

    def test_access_and_unsupported(self):
        self.assertEqual(self.request(VALID, {"Origin": "https://foreign.example"})[0], 403)
        self.assertEqual(self.request(VALID, {"Content-Type": "text/plain"})[0], 415)
        self.svc.api_key = "secret"
        self.assertEqual(self.request()[0], 401)
        self.assertEqual(self.request(VALID)[0], 401)
        self.assertEqual(self.request(VALID, {"Authorization": "Bearer secret"})[0], 200)
        self.svc.api_key = ""
        self.engine.can_vram_control = False
        self.assertFalse(self.request()[1]["supported"])
        self.assertEqual(self.request(VALID)[0], 501)

    def test_invalid_request_does_not_change_settings(self):
        for value in ([], {}, {**VALID, "reserve_mib": 1}, {**VALID, "matrices": True}):
            self.assertEqual(self.request(value)[0], 400)
        self.assertFalse(self.sent)

    def test_unloaded_model_records_pending_settings(self):
        self.engine.alive = lambda: False
        code, data = self.request(VALID)
        self.assertEqual(code, 200)
        self.assertFalse(data["alive"])
        self.assertTrue(data["pending"])


if __name__ == "__main__":
    unittest.main()
