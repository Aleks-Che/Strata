"""Archive control isolation, access checks and stale-action protection (no GPU)."""
import io
import json
import queue
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, StrataEngine, serve


class ArchiveHTTP(unittest.TestCase):
    def setUp(self):
        tok = ByteTokenizer()
        self.engine = MockEngine(tok, "ok")
        self.engine.can_cache_admin = True
        self.engine.cache_generation = "generation-one"
        self.engine.alive = lambda: True
        self.engine.cache_inventory = {"updated_at_ms": 123, "entries": [
            {"id": "active", "kind": "active", "deletable": False},
            {"id": "session-1", "kind": "session", "deletable": True},
            {"id": "session-1:checkpoint-42", "kind": "checkpoint", "deletable": True}]}
        self.removed = []
        def drop(entry_id):
            self.removed.append(entry_id)
            self.engine.cache_inventory = {"entries": [e for e in self.engine.cache_inventory["entries"] if e["id"] != entry_id]}
            return True
        self.engine.drop_cache_entry = drop
        self.svc = Service(self.engine, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_port}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def request(self, path="/cache/release", body=None, headers=None):
        if body is None and path != "/cache/entries":
            body = {"id": "session-1", "generation": "generation-one"}
        request = urllib.request.Request(self.base + path, None if body is None else json.dumps(body).encode(),
                                         {"Content-Type": "application/json", **(headers or {})})
        try:
            with urllib.request.urlopen(request, timeout=3) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as error:
            return error.code, json.load(error)

    def test_inventory_and_release(self):
        code, data = self.request("/cache/entries")
        self.assertEqual(code, 200)
        self.assertEqual(len(data["entries"]), 2)
        self.assertEqual(data["saved_sessions"], 1)
        self.assertEqual(data["active_sessions"], 1)
        code, data = self.request()
        self.assertEqual(code, 200)
        self.assertEqual(self.removed, ["session-1"])
        self.assertEqual([e["id"] for e in data["entries"]], ["active"])
        self.assertEqual(self.request()[0], 404)

    def test_save_failure_is_visible_separately_from_active_session(self):
        self.engine.cache = {"sessions": 0, "bytes": 0, "last_save_reason": "commit_limit",
                             "last_save_bytes": 4200000000, "skipped_saves": 2}
        self.engine.cache_inventory = {"entries": [{"id": "active", "kind": "active", "deletable": False}]}
        code, data = self.request("/cache/entries")
        self.assertEqual(code, 200)
        self.assertEqual(data["saved_sessions"], 0)
        self.assertEqual(data["active_sessions"], 1)
        self.assertEqual(data["diagnostics"]["last_save_reason"], "commit_limit")
        self.assertEqual(self.svc.metrics()["conversation_cache"]["active_sessions"], 1)

    def test_busy_counters_and_inventory_use_the_same_completed_snapshot(self):
        self.engine.cache = {"sessions": 0, "bytes": 0}
        self.engine.cache_inventory = {"entries": [
            {"id": "active", "kind": "active", "bytes": None},
            {"id": "session-1", "kind": "session", "bytes": 3000000000}]}
        self.svc.status['queued'] = 1
        data = self.request('/cache/entries')[1]
        self.assertTrue(data['busy'])
        self.assertEqual(data['archive_bytes'], 3000000000)
        cache = self.svc.metrics()['conversation_cache']
        self.assertEqual((cache['sessions'], cache['bytes']), (1, 3000000000))

    def test_auth_on_both_endpoints(self):
        self.svc.api_key = "secret"
        self.assertEqual(self.request("/cache/entries")[0], 401)
        self.assertEqual(self.request()[0], 401)
        self.assertEqual(self.request(headers={"Authorization": "Bearer secret"})[0], 200)

    def test_foreign_origin_and_form_cannot_evict(self):
        self.assertEqual(self.request(headers={"Origin": "https://foreign.example"})[0], 403)
        self.assertEqual(self.request(headers={"Content-Type": "text/plain"})[0], 415)
        self.assertFalse(self.removed)

    def test_busy_and_queued_do_not_write_control_pipe(self):
        self.svc.fifo.acquire()
        try:
            self.assertEqual(self.request()[0], 409)
        finally:
            self.svc.fifo.release()
        self.svc.status["queued"] = 1
        self.assertEqual(self.request()[0], 409)
        self.assertFalse(self.removed)

    def test_stale_restart_id_and_reserved_state(self):
        self.assertEqual(self.request(body={"id": "session-1", "generation": "old-process"})[0], 409)
        self.assertEqual(self.request(body={"id": "active", "generation": "generation-one"})[0], 409)
        self.assertEqual(self.request(body={"id": "session-1\nQUIT", "generation": "generation-one"})[0], 404)
        self.assertEqual(self.request(body=[])[0], 400)
        self.assertFalse(self.removed)

    def test_checkpoint_release_is_rejected_even_for_old_engine_inventory(self):
        code, data = self.request("/cache/entries")
        self.assertEqual(code, 200)
        self.assertTrue(all(e["kind"] in ("active", "session") for e in data["entries"]))
        self.assertEqual(self.request(body={"id": "session-1:checkpoint-42", "generation": "generation-one"})[0], 409)
        self.assertFalse(self.removed)

    def test_old_engine_reports_unsupported(self):
        self.engine.can_cache_admin = False
        self.assertFalse(self.request("/cache/entries")[1]["supported"])
        self.assertEqual(self.request()[0], 501)

    def test_expired_entry_and_error_release_lock(self):
        self.engine.drop_cache_entry = lambda entry_id: False
        self.assertEqual(self.request()[0], 404)
        self.assertTrue(self.svc.fifo.acquire(blocking=False))
        self.svc.fifo.release()

    def test_auto_release_defaults_validation_and_auth(self):
        self.assertEqual(self.request("/cache/entries")[1]["auto_release"]["hours"], 2)
        self.assertFalse(self.request("/cache/entries")[1]["auto_release"]["enabled"])
        valid = {"enabled": True, "hours": 2.5}
        for body in ([], {}, {"enabled": 1, "hours": 2}, {"enabled": False, "hours": True},
                     *({"enabled": True, "hours": hours} for hours in (0, -1, 9000, 10**1000, "2", float("nan"), float("inf")))):
            with self.subTest(body=body):
                self.assertEqual(self.request("/cache/settings", body=body)[0], 400)
        for headers, status in (({"Origin": "https://foreign.example"}, 403), ({"Content-Type": "text/plain"}, 415)):
            self.assertEqual(self.request("/cache/settings", valid, headers)[0], status)
        self.svc.api_key = "secret"
        self.assertEqual(self.request("/cache/settings", valid)[0], 401)
        code, data = self.request("/cache/settings", valid, {"Authorization": "Bearer secret"})
        self.assertEqual(code, 200)
        self.assertTrue(data["auto_release"]["enabled"])
        self.assertEqual(data["auto_release"]["hours"], 2.5)

    def test_policy_can_change_while_busy_and_expiry_runs_at_queued_handoff(self):
        self.svc.status["queued"] = 1
        self.engine.cache_inventory["entries"] = [
            {"id": "active", "kind": "active", "last_used_ms": 1, "deletable": False},
            {"id": "session-old", "kind": "session", "last_used_ms": (time.time() - 3 * 3600) * 1000, "deletable": True}]
        with self.svc.fifo:
            self.assertEqual(self.request("/cache/settings", {"enabled": True, "hours": 2})[0], 200)
            self.assertFalse(self.removed)
        generate = self.engine.generate
        def checked_generate(*args, **kwargs):
            self.assertEqual(self.removed, ["session-old"])
            yield from generate(*args, **kwargs)
        self.engine.generate = checked_generate
        list(self.svc.run([1, 2], False, [], 8, {}, threading.Event()))
        self.assertEqual(self.removed, ["session-old"])
        self.assertEqual(self.engine.cache_inventory["entries"][0]["id"], "active")


class ControlPump(unittest.TestCase):
    def test_save_diagnostics_survive_a_later_active_cache_hit(self):
        engine = StrataEngine.__new__(StrataEngine)
        engine.cache, engine.last = {}, {}
        engine._parse_cache('CACHE reason=commit_limit last_save_reason=commit_limit last_save_bytes=4300000000 '
                            'last_save_commit=1900000000 last_save_physical=41000000000 last_save_at_ms=1790838123000 '
                            'save_attempts=2 skipped_saves=1')
        engine._parse_cache('CACHE phase=done source=active reason=none bytes=0 sessions=0')
        self.assertEqual(engine.cache['last_save_reason'], 'commit_limit')
        self.assertEqual(engine.cache['last_save_bytes'], 4300000000)
        self.assertEqual(engine.cache['skipped_saves'], 1)
        self.assertEqual(engine.last['cache_reason'], 'none')

    def test_inventory_and_control_replies_never_reach_generation(self):
        engine = StrataEngine.__new__(StrataEngine)
        engine.lines, engine.cache_replies = queue.Queue(), queue.Queue()
        engine.cache, engine.last = {}, {"reused": 42}
        engine.proc = SimpleNamespace(stdout=io.StringIO(
            'CACHE_ENTRIES {"entries":[{"id":"session-7"}]}\n'
            'CACHE phase=idle bytes=64 sessions=1\n'
            'CACHE_DROPPED request1 removed\nT 9\nDONE 1 2 3 4 stop\n'))
        engine._pump()
        self.assertEqual(engine.cache_inventory["entries"][0]["id"], "session-7")
        self.assertEqual(engine.cache_replies.get_nowait(), ["request1", "removed"])
        self.assertEqual(list(engine.lines.queue), ['T 9\n', 'DONE 1 2 3 4 stop\n', None])
        self.assertEqual(engine.last, {"reused": 42})


if __name__ == "__main__":
    unittest.main()
