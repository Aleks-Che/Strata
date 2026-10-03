"""Accounting, durable restart, concurrent writes and both HTTP APIs; no GPU needed."""
import concurrent.futures
import json
import sqlite3
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, serve
from serve.usage_stats import UsageStatistics

ROOT = Path(__file__).resolve().parents[1]


class DurableStatistics(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "usage.sqlite"
        self.stats = UsageStatistics(self.path)

    def tearDown(self):
        self.stats.close()
        self.temp.cleanup()

    def test_exact_counts_survive_restart_and_idle_intervals_are_zero(self):
        now = 1800000000
        self.stats.record(input_tokens=7, cached_tokens=60097, output_tokens=18, finished_at=now - 7200)
        self.stats.record(input_tokens=9, cached_tokens=90001, output_tokens=22, finished_at=now)
        before = self.stats.snapshot(now=now)
        self.assertEqual(before["totals"], dict(requests=2, input_tokens=16, cached_tokens=150098, output_tokens=40))
        self.assertEqual(len(before["points"]), 24)
        self.assertEqual(sum(p["requests"] == 0 for p in before["points"]), 22)
        self.stats.close()
        self.stats = UsageStatistics(self.path)
        after = self.stats.snapshot(now=now)
        self.assertEqual(before, after)
        self.assertEqual(self.stats.db.execute("PRAGMA integrity_check").fetchone()[0], "ok")

    def test_range_boundaries_and_all_time_totals(self):
        now = 1800000000 // 86400 * 86400
        for age in (0, 3600, 2 * 86400, 15 * 86400, 40 * 86400):
            self.stats.record(input_tokens=100, cached_tokens=200, output_tokens=3, finished_at=now - age)
        for period, expected in (("24h", 2), ("7d", 3), ("30d", 4), ("all", 5)):
            with self.subTest(period=period):
                data = self.stats.snapshot(period, now=now)
                self.assertEqual(data["period_totals"]["requests"], expected)
                self.assertEqual(data["totals"]["requests"], 5)
                self.assertEqual(sum(p["input_tokens"] for p in data["points"]), 100 * expected)
        with self.assertRaises(ValueError):
            self.stats.snapshot("invalid")

    def test_concurrent_records_are_counted_once(self):
        def record(_):
            self.stats.record(input_tokens=5, cached_tokens=7, output_tokens=11)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(record, range(60)))
        self.assertEqual(self.stats.snapshot()["totals"], dict(requests=60, input_tokens=300, cached_tokens=420, output_tokens=660))

    def test_locked_file_retries_without_loss_or_double_counting(self):
        self.stats.db.execute("PRAGMA busy_timeout=1")
        blocker = sqlite3.connect(self.path)
        try:
            blocker.execute("BEGIN IMMEDIATE")
            self.stats.record(input_tokens=10, cached_tokens=40, output_tokens=5)
            unsaved = self.stats.snapshot()
            self.assertTrue(unsaved["storage_error"])
            self.assertEqual(unsaved["pending_requests"], 1)
            self.assertEqual(unsaved["totals"]["input_tokens"], 10)
            self.assertEqual(unsaved["period_totals"]["requests"], 1)
            self.assertEqual(self.stats.snapshot("all")["period_totals"]["requests"], 1)
        finally:
            blocker.rollback()
            blocker.close()
        saved = self.stats.snapshot()
        self.assertIsNone(saved["storage_error"])
        self.assertEqual(saved["pending_requests"], 0)
        self.assertEqual(saved["totals"], unsaved["totals"])
        self.assertEqual(self.stats.snapshot()["totals"]["requests"], 1)

    def test_empty_database_is_zero_and_large_range_is_bounded(self):
        self.assertEqual(self.stats.snapshot("all")["totals"]["requests"], 0)
        self.stats.record(input_tokens=1, cached_tokens=0, output_tokens=1, finished_at=1000000000)
        data = self.stats.snapshot("all", now=1800000000)
        self.assertLessEqual(len(data["points"]), 182)
        self.assertEqual(data["period_totals"]["requests"], 1)


class CachedEngine(MockEngine):
    def generate(self, ids, max_new, sampling, cancel, **kwargs):
        self.last = {"reused": min(12, len(ids))}
        yield from super().generate(ids, max_new, sampling, cancel, **kwargs)


class StatisticsHTTP(unittest.TestCase):
    def setUp(self):
        self.tok = ByteTokenizer()
        self.engine = CachedEngine(self.tok, "ok")
        self.svc = Service(self.engine, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_port}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.svc.statistics.close()

    def call(self, path, body=None, headers=None):
        req = urllib.request.Request(self.base + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=5) as response:
                return response.status, response.read().decode()
        except urllib.error.HTTPError as error:
            return error.code, error.read().decode()

    def test_streaming_and_nonstreaming_both_apis_count_once(self):
        for path in ("/v1/chat/completions", "/v1/messages"):
            for stream in (False, True):
                code, body = self.call(path, {"model": "m", "max_tokens": 8, "stream": stream,
                                            "messages": [{"role": "user", "content": "hello"}]})
                self.assertEqual(code, 200, body)
        data = json.loads(self.call("/statistics")[1])
        recent = list(self.svc.history)
        self.assertEqual(data["totals"]["requests"], 4)
        self.assertEqual(data["totals"]["cached_tokens"], 48)
        self.assertEqual(data["totals"]["input_tokens"], sum(r["prompt_tokens"] - 12 for r in recent))
        self.assertEqual(data["totals"]["output_tokens"], sum(r["output_tokens"] for r in recent))

    def test_auth_range_validation_and_rejected_requests(self):
        self.svc.api_key = "test-secret"
        self.assertEqual(self.call("/statistics")[0], 401)
        auth = {"Authorization": "Bearer test-secret"}
        self.assertEqual(self.call("/statistics?range=wrong", headers=auth)[0], 400)
        self.assertEqual(self.call("/v1/chat/completions", {"max_tokens": 900000,
                             "messages": [{"role": "user", "content": "hello"}]}, auth)[0], 400)
        data = json.loads(self.call("/statistics?range=7d", headers=auth)[1])
        self.assertEqual(data["totals"]["requests"], 0)

    def test_cancelled_prefill_counts_only_known_processed_tokens(self):
        cancel = threading.Event()
        def partial(ids, max_new, sampling, cancel):
            self.engine.last = {"reused": 20, "generated": 0}
            self.engine.progress = (35, len(ids))
            cancel.set()
            yield None
        self.engine.generate = partial
        list(self.svc.run([1] * 100, False, None, 10, {}, cancel))
        self.assertEqual(self.svc.statistics.snapshot()["totals"],
                         dict(requests=1, input_tokens=15, cached_tokens=20, output_tokens=0))

    def test_disconnected_stream_counts_drained_engine_output(self):
        def generating(ids, max_new, sampling, cancel):
            self.engine.last = {"reused": 20}
            try:
                yield ord("x")
                yield ord("y")
            finally:
                self.engine.last["generated"] = 5  # extra engine output consumed during STOP/drain
        self.engine.generate = generating
        gen = self.svc.run([1] * 100, False, None, 10, {}, threading.Event())
        next(gen)
        gen.close()
        self.assertEqual(self.svc.statistics.snapshot()["totals"],
                         dict(requests=1, input_tokens=80, cached_tokens=20, output_tokens=5))

    def test_cancelled_queue_entry_is_not_counted(self):
        cancel = threading.Event()
        cancel.set()
        list(self.svc.run([1] * 100, False, None, 10, {}, cancel))
        self.assertEqual(self.svc.statistics.snapshot()["totals"]["requests"], 0)

    def test_reasoning_budget_counts_both_engine_passes_once_without_injected_output(self):
        calls = []
        def generate(ids, max_new, sampling, cancel, **kwargs):
            calls.append((len(ids), kwargs.get("session_id")))
            script = "abcdefgh" if len(calls) == 1 else "OK"
            count = 0
            try:
                for token in self.tok.encode(script + "<|im_end|>", parse_special=True):
                    count += 1
                    yield token
            finally:
                self.engine.last = {"generated": count, "reused": 5}
        self.engine.generate = generate
        self.svc.request_trace.session_id = "budget-session"
        list(self.svc.run([1] * 20, True, [], 200, {"reasoning_budget_tokens": 2}, threading.Event()))
        self.assertEqual(len(calls), 2)
        self.assertEqual([session for _, session in calls], ["budget-session"] * 2)
        self.assertEqual(self.svc.statistics.snapshot()["totals"],
                         dict(requests=1, input_tokens=sum(n-5 for n, _ in calls), cached_tokens=10, output_tokens=5))

    def test_failed_request_cannot_reuse_previous_done_statistics(self):
        self.engine.last = {"generated": 100, "reused": 90}
        def broken(*args, **kwargs):
            raise ValueError("engine rejected the request")
            yield
        self.engine.generate = broken
        with self.assertRaises(ValueError):
            list(self.svc.run([1] * 100, False, [], 10, {}, threading.Event()))
        self.assertEqual(self.svc.statistics.snapshot()["totals"],
                         dict(requests=1, input_tokens=0, cached_tokens=0, output_tokens=0))


if __name__ == "__main__":
    unittest.main()
