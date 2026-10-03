"""HTTP session identity and cache protocol tests; no GPU or model required."""
import hashlib
import io
import json
import queue
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, StrataEngine, serve


class SessionHeaders(unittest.TestCase):
    def setUp(self):
        tok = ByteTokenizer()
        self.engine = MockEngine(tok, "ok", max_context=4096)
        self.svc = Service(self.engine, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_port}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def post(self, api, identity=None, stream=False):
        body = {"model": "test", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4, "stream": stream}
        headers = {"Content-Type": "application/json"}
        if identity is not None:
            headers["X-Strata-Session-Id"] = identity
        request = urllib.request.Request(self.base + api, json.dumps(body).encode(), headers)
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.read()

    def test_both_apis_keep_identity_out_of_prompt(self):
        for api in ("/v1/chat/completions", "/v1/messages"):
            for stream in (False, True):
                with self.subTest(api=api, stream=stream):
                    status, _ = self.post(api, "project-a/chat-1", stream)
                    self.assertEqual(status, 200)
                    self.assertEqual(self.engine.last_session_id, "project-a/chat-1")
                    named_prompt = self.engine.last_prompt[:]
                    self.assertEqual(self.post(api, None, stream)[0], 200)
                    self.assertIsNone(self.engine.last_session_id)
                    self.assertEqual(named_prompt, self.engine.last_prompt)

    def test_invalid_header_rejected_before_stream(self):
        for api in ("/v1/chat/completions", "/v1/messages"):
            for identity in (" ", "x" * 257):
                self.assertEqual(self.post(api, identity, True)[0], 400)

    def test_old_engine_explicit_id_fails_without_breaking_automatic_clients(self):
        self.engine.can_session_id = False
        self.assertEqual(self.post("/v1/chat/completions", "named", True)[0], 400)
        self.assertEqual(self.post("/v1/chat/completions")[0], 200)

    def test_metrics_include_archive_and_transfer_phase(self):
        self.engine.cache = {"bytes": 123, "sessions": 2, "phase": "restoring"}
        self.svc.status.update(busy=True, first_token=None, started=0)
        metrics = self.svc.metrics()
        self.assertEqual(metrics["conversation_cache"]["bytes"], 123)
        self.assertEqual(metrics["live"]["phase"], "restoring session")
        with urllib.request.urlopen(self.base + "/status", timeout=5) as response:
            self.assertEqual(json.load(response)["phase"], "restoring session")

    def test_request_finalizes_metrics_before_releasing_queue(self):
        # At handoff, the next session may immediately overwrite engine.last.
        # It must see the previous request's history already committed and idle.
        observed = []
        svc = self.svc
        class Handoff:
            def __enter__(self):
                pass
            def __exit__(self, *exc):
                observed.append((svc.status['busy'], len(svc.history)))
        svc.fifo = Handoff()
        list(svc.run([1, 2, 3], False, [], 4, {}, threading.Event(), session_id='a'))
        self.assertEqual(observed, [(False, 1)])
        cancelled = threading.Event()
        cancelled.set()
        list(svc.run([4, 5, 6], False, [], 4, {}, cancelled, session_id='b'))
        self.assertEqual(observed[-1], (False, 1))
        self.assertEqual(self.engine.last_session_id, 'a')


class CacheProtocol(unittest.TestCase):
    def engine(self):
        engine = StrataEngine.__new__(StrataEngine)
        engine.stdin_lock = threading.RLock()
        engine.proc = SimpleNamespace(stdin=io.StringIO())
        engine.can_session_id = True
        engine.can_stop = True
        engine.cache = {}
        engine.last = {"reused": 999, "cache_source": "old"}
        engine.lines = queue.Queue()
        for line in ("CACHE phase=restoring source=cold bytes=10 sessions=1 hits=1 misses=2 save_ms=4 restore_ms=0\n",
                     "RESUME 2\n", "CACHE phase=done source=ram bytes=10 sessions=1 hits=1 misses=2 save_ms=4 restore_ms=5\n",
                     "T 7\n", "DONE 1 3 6.0 1.0 length 0 0 2\n"):
            engine.lines.put(line)
        return engine

    def test_text_and_image_protocol_hash_ids_before_payload(self):
        for embeddings in (None, "image.sve"):
            engine = self.engine()
            result = list(engine.generate([1, 2, 3], 1, {}, threading.Event(), embeddings, "project-a"))
            self.assertEqual([x for x in result if x is not None], [7])
            line = engine.proc.stdin.getvalue()
            key = hashlib.sha256(b"project-a").hexdigest()
            prefix = "GENI" if embeddings else "GEN"
            self.assertTrue(line.startswith(f"{prefix} 1 session={key} "))
            self.assertNotIn("project-a", line)
            if embeddings:
                self.assertIn("image.sve 1,2,3", line)
            self.assertEqual(engine.progress, (2, 3))
            self.assertEqual(engine.last["cache_source"], "ram")
            self.assertEqual(engine.last["cache_restore_ms"], 5)
            self.assertEqual(engine.last["reused"], 2)

    def test_cancel_drain_keeps_cache_metrics_with_matching_done(self):
        engine = self.engine()
        gen = engine.generate([1, 2, 3], 1, {}, threading.Event())
        next(gen)
        gen.close()
        self.assertIn("STOP\n", engine.proc.stdin.getvalue())
        self.assertEqual(engine.last["cache_source"], "ram")
        self.assertEqual(engine.last["generated"], 1)

    def test_idle_expiration_updates_metrics_without_polluting_request_or_queue(self):
        engine = self.engine()
        engine.lines = queue.Queue()
        before = dict(engine.last)
        engine.proc.stdout = io.StringIO('CACHE phase=idle bytes=0 sessions=0 expired=2 pressure_evictions=1 deduplicated=3\n')
        engine._pump()
        self.assertEqual(engine.cache['expired'], 2)
        self.assertEqual(engine.cache['bytes'], 0)
        self.assertEqual(engine.last, before)
        self.assertIsNone(engine.lines.get_nowait())
        self.assertTrue(engine.lines.empty())


if __name__ == "__main__":
    unittest.main()
