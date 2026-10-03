"""Persistent, opt-in expiry of whole archived sessions. The active state is never released."""
from __future__ import annotations

import json
import math
import os
import tempfile
import threading
import time
from pathlib import Path


class ArchivePolicy:
    def __init__(self, path=None):
        self.path = Path(path) if path is not None else None
        self.lock = threading.RLock()
        self.enabled, self.hours = False, 2
        self.storage_error = self.last_error = None
        self.released_sessions = 0
        self.wake = threading.Event()
        self.stopping = threading.Event()
        self.worker = None
        if self.path:
            try:
                self.enabled, self.hours = self.validate(json.loads(self.path.read_text(encoding="utf-8")))
            except FileNotFoundError:
                pass
            except (OSError, ValueError) as error:
                self.storage_error = f"Could not load auto-release settings: {error}"

    @staticmethod
    def validate(value):
        if not isinstance(value, dict) or type(value.get("enabled")) is not bool:
            raise ValueError("enabled must be true or false")
        hours = value.get("hours")
        if type(hours) not in (int, float) or not .1 <= hours <= 8760 or not math.isfinite(hours):
            raise ValueError("hours must be a number between 0.1 and 8760")
        return value["enabled"], hours

    def snapshot(self):
        with self.lock:
            return {"enabled": self.enabled, "hours": self.hours, "persistent": self.path is not None,
                    "storage_error": self.storage_error, "last_error": self.last_error,
                    "released_sessions": self.released_sessions}

    def update(self, value):
        enabled, hours = self.validate(value)
        with self.lock:
            if self.path:
                temporary = None
                try:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent,
                                                     prefix=self.path.name + ".", suffix=".tmp", delete=False) as f:
                        temporary = Path(f.name)
                        json.dump({"enabled": enabled, "hours": hours}, f)
                        f.write("\n")
                        f.flush()
                        os.fsync(f.fileno())
                    os.replace(temporary, self.path)
                except OSError as error:
                    self.storage_error = "Could not save auto-release settings; previous settings are still active"
                    raise OSError(self.storage_error) from error
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
            self.enabled, self.hours = enabled, hours
            self.storage_error = self.last_error = None
            self.wake.set()
            return self.snapshot()

    def release_expired_locked(self, engine, now_ms=None):
        """Caller owns the service FIFO, including at request boundaries under continuous traffic."""
        with self.lock:
            if not self.enabled or not getattr(engine, "can_cache_admin", False) or not engine.alive():
                return 0
            cutoff = (time.time() * 1000 if now_ms is None else now_ms) - self.hours * 3600000
            entries = list(engine.cache_inventory.get("entries", []))
            released = 0
            try:
                for entry in entries:
                    used = entry.get("last_used_ms")
                    if (entry.get("kind") != "session" or not entry.get("deletable") or
                            type(used) not in (int, float) or not math.isfinite(used) or used <= 0 or used > cutoff):
                        continue
                    if engine.drop_cache_entry(entry["id"]):
                        released += 1
                        self.released_sessions += 1
                self.last_error = None
            except (OSError, RuntimeError) as error:
                self.last_error = f"Auto-release could not finish: {error}"
            return released

    def sweep(self, service):
        if not service.fifo.acquire(blocking=False):
            return 0
        try:
            with service.status_lock:
                if service.status.get("busy") or service.status.get("queued", 0):
                    return 0
            return self.release_expired_locked(service.engine)
        finally:
            service.fifo.release()

    def start(self, service):
        def run():
            while not self.stopping.is_set():
                self.wake.clear()
                self.sweep(service)
                self.wake.wait(5)
        self.worker = threading.Thread(target=run, name="archive-auto-release", daemon=True)
        self.worker.start()

    def stop(self):
        self.stopping.set()
        self.wake.set()
        if self.worker:
            self.worker.join(timeout=16)
