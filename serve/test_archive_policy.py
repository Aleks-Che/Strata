"""Expiry, request isolation and durable archive preferences without a GPU."""
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from serve.archive_policy import ArchivePolicy


class ArchivePolicyTests(unittest.TestCase):
    def setUp(self):
        self.now = time.time() * 1000
        self.removed = []
        self.engine = SimpleNamespace(can_cache_admin=True, alive=lambda: True, cache_inventory={"entries": []})
        def drop(entry_id):
            self.removed.append(entry_id)
            self.engine.cache_inventory["entries"] = [e for e in self.engine.cache_inventory["entries"] if e["id"] != entry_id]
            return True
        self.engine.drop_cache_entry = drop
        self.policy = ArchivePolicy()

    def entry(self, name, hours, kind="session", deletable=True):
        return {"id": name, "kind": kind, "deletable": deletable, "last_used_ms": self.now - hours * 3600000}

    def test_default_off_then_boundary_and_only_whole_saved_sessions(self):
        self.engine.cache_inventory["entries"] = [self.entry("old", 3), self.entry("boundary", 2),
            self.entry("recent", 1.999), self.entry("active", 9, "active"),
            self.entry("checkpoint", 9, "checkpoint"), self.entry("protected", 9, deletable=False)]
        self.assertEqual((self.policy.enabled, self.policy.hours), (False, 2))
        self.assertEqual(self.policy.release_expired_locked(self.engine, self.now), 0)
        self.policy.update({"enabled": True, "hours": 2})
        self.assertEqual(self.policy.release_expired_locked(self.engine, self.now), 2)
        self.assertEqual(self.removed, ["old", "boundary"])
        self.assertEqual(self.policy.snapshot()["released_sessions"], 2)

    def test_last_use_renewal_disable_and_unknown_timestamps(self):
        entry = self.entry("reused", 3)
        self.engine.cache_inventory["entries"] = [entry]
        self.policy.update({"enabled": True, "hours": 2})
        entry["last_used_ms"] = self.now
        self.assertEqual(self.policy.release_expired_locked(self.engine, self.now), 0)
        for value in (None, 0, True, float("nan"), float("inf"), "123"):
            entry["last_used_ms"] = value
            self.assertEqual(self.policy.release_expired_locked(self.engine, self.now), 0)
        entry["last_used_ms"] = self.now - 4 * 3600000
        self.policy.update({"enabled": False, "hours": .5})
        self.assertEqual(self.policy.release_expired_locked(self.engine, self.now), 0)
        self.policy.update({"enabled": True, "hours": .5})
        self.assertEqual(self.policy.release_expired_locked(self.engine, self.now), 1)

    def test_settings_survive_restart_and_failed_write_keeps_previous_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "archive.json"
            policy = ArchivePolicy(path)
            policy.update({"enabled": True, "hours": 3.5})
            restored = ArchivePolicy(path)
            self.assertEqual((restored.enabled, restored.hours), (True, 3.5))
            with patch("serve.archive_policy.os.replace", side_effect=OSError("disk unavailable")):
                with self.assertRaises(OSError):
                    restored.update({"enabled": False, "hours": 8})
            self.assertEqual((restored.enabled, restored.hours), (True, 3.5))
            self.assertEqual(json.loads(path.read_text()), {"enabled": True, "hours": 3.5})
            self.assertEqual(len(list(Path(directory).iterdir())), 1)
            restored.update({"enabled": False, "hours": 8})
            self.assertIsNone(restored.storage_error)
            self.assertFalse(ArchivePolicy(path).enabled)

    def test_invalid_saved_settings_fail_closed_without_overwriting_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "archive.json"
            path.write_text("broken")
            policy = ArchivePolicy(path)
            self.assertFalse(policy.enabled)
            self.assertIsNotNone(policy.storage_error)
            self.assertEqual(path.read_text(), "broken")

    def test_busy_queue_and_fifo_protect_engine(self):
        self.engine.cache_inventory["entries"] = [self.entry("old", 3)]
        self.policy.update({"enabled": True, "hours": 2})
        service = SimpleNamespace(engine=self.engine, fifo=threading.Lock(), status_lock=threading.Lock(), status={})
        with service.fifo:
            self.assertEqual(self.policy.sweep(service), 0)
        for status in ({"busy": True}, {"queued": 1}):
            service.status = status
            self.assertEqual(self.policy.sweep(service), 0)
        self.assertFalse(self.removed)
        service.status = {}
        self.assertEqual(self.policy.sweep(service), 1)
        self.assertTrue(service.fifo.acquire(blocking=False))
        service.fifo.release()

    def test_idle_worker_runs_without_browser_and_stops(self):
        self.engine.cache_inventory["entries"] = [self.entry("old", 3)]
        service = SimpleNamespace(engine=self.engine, fifo=threading.Lock(), status_lock=threading.Lock(), status={})
        removed = threading.Event()
        drop = self.engine.drop_cache_entry
        def observed(entry_id):
            result = drop(entry_id)
            removed.set()
            return result
        self.engine.drop_cache_entry = observed
        self.policy.start(service)
        try:
            self.policy.update({"enabled": True, "hours": 2})
            self.assertTrue(removed.wait(2))
        finally:
            self.policy.stop()
        self.assertFalse(self.policy.worker.is_alive())

    def test_failed_release_is_reported_and_does_not_escape_into_inference(self):
        self.engine.cache_inventory["entries"] = [self.entry("old", 3)]
        self.policy.update({"enabled": True, "hours": 2})
        with patch.object(self.engine, "drop_cache_entry", side_effect=RuntimeError("engine stopped")):
            self.assertEqual(self.policy.release_expired_locked(self.engine, self.now), 0)
        self.assertIn("engine stopped", self.policy.snapshot()["last_error"])
        self.assertEqual(self.policy.release_expired_locked(self.engine, self.now), 1)
        self.assertIsNone(self.policy.snapshot()["last_error"])


if __name__ == "__main__":
    unittest.main()
