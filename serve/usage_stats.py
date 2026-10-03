"""Durable, anonymous token accounting. One aggregate per UTC minute, no prompt text."""
from __future__ import annotations

import math
import sqlite3
import threading
import time
from pathlib import Path

FIELDS = ("requests", "input_tokens", "cached_tokens", "output_tokens")
RANGES = {"24h": (86400, 3600), "7d": (7 * 86400, 6 * 3600), "30d": (30 * 86400, 86400)}


class UsageStatistics:
    def __init__(self, path=":memory:"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.pending = {}
        self.storage_error = None
        self.db = sqlite3.connect(self.path, timeout=2, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        # FULL + the rollback journal commit both tables together, including on abrupt exit.
        self.db.execute("PRAGMA synchronous=FULL")
        with self.db:
            self.db.execute("""CREATE TABLE IF NOT EXISTS usage_minutes (
                time INTEGER PRIMARY KEY, requests INTEGER NOT NULL, input_tokens INTEGER NOT NULL,
                cached_tokens INTEGER NOT NULL, output_tokens INTEGER NOT NULL)""")
            self.db.execute("""CREATE TABLE IF NOT EXISTS usage_totals (
                id INTEGER PRIMARY KEY CHECK (id=1), started_at REAL NOT NULL, updated_at REAL,
                requests INTEGER NOT NULL, input_tokens INTEGER NOT NULL,
                cached_tokens INTEGER NOT NULL, output_tokens INTEGER NOT NULL)""")
            self.db.execute("INSERT OR IGNORE INTO usage_totals VALUES (1, ?, NULL, 0, 0, 0, 0)", (time.time(),))

    def record(self, *, input_tokens, cached_tokens, output_tokens, finished_at=None):
        """Record once after the engine has finished/drained a request, independent of HTTP streaming."""
        stamp = time.time() if finished_at is None else finished_at
        values = [1, max(0, int(input_tokens)), max(0, int(cached_tokens)), max(0, int(output_tokens))]
        minute = int(stamp // 60) * 60
        with self.lock:
            row = self.pending.setdefault(minute, [0, 0, 0, 0, stamp])
            for i, value in enumerate(values):
                row[i] += value
            row[4] = max(row[4], stamp)
            self._flush()

    def _flush(self):
        if not self.pending:
            return
        try:
            with self.db:
                for minute, row in self.pending.items():
                    self.db.execute("""INSERT INTO usage_minutes VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(time) DO UPDATE SET requests=requests+excluded.requests,
                        input_tokens=input_tokens+excluded.input_tokens,
                        cached_tokens=cached_tokens+excluded.cached_tokens,
                        output_tokens=output_tokens+excluded.output_tokens""", (minute, *row[:4]))
                totals = [sum(row[i] for row in self.pending.values()) for i in range(4)]
                self.db.execute("""UPDATE usage_totals SET requests=requests+?, input_tokens=input_tokens+?,
                    cached_tokens=cached_tokens+?, output_tokens=output_tokens+?,
                    updated_at=MAX(COALESCE(updated_at, 0), ?) WHERE id=1""",
                    (*totals, max(row[4] for row in self.pending.values())))
            self.pending.clear()
            self.storage_error = None
        except sqlite3.Error as error:
            # Keep unsaved increments for retry. Never reset or replace a damaged statistics file.
            message = str(error)
            if message != self.storage_error:
                print(f"[strata] statistics not saved; will retry: {message}", flush=True)
            self.storage_error = message

    def snapshot(self, period="24h", now=None):
        if period not in (*RANGES, "all"):
            raise ValueError("range must be 24h, 7d, 30d or all")
        now = time.time() if now is None else now
        with self.lock:
            self._flush()
            total = dict(self.db.execute("SELECT * FROM usage_totals WHERE id=1").fetchone())
            first = self.db.execute("SELECT MIN(time) FROM usage_minutes").fetchone()[0]
            if self.pending:
                first = min([*self.pending, *([] if first is None else [first])])
            if period == "all":
                beginning = min(first if first is not None else now, now)
                step = max(86400, math.ceil((now - beginning) / (180 * 86400)) * 86400)
                start = int(beginning // step) * step
            else:
                span, step = RANGES[period]
                # Include the current partial interval and the preceding intervals.
                start = int(now // step) * step - span + step
            end = int(now // step) * step + step
            sums = ", ".join(f"SUM({name}) AS {name}" for name in FIELDS)
            rows = self.db.execute(f"SELECT (time / ?) * ? AS time, {sums} FROM usage_minutes "
                                   "WHERE time >= ? AND time < ? GROUP BY 1", (step, step, start, end)).fetchall()
            buckets = {row["time"]: {key: row[key] for key in FIELDS} for row in rows}
            for minute, row in self.pending.items():
                for i, key in enumerate(FIELDS):
                    total[key] += row[i]
                if start <= minute < end:
                    bucket = buckets.setdefault(minute // step * step, dict.fromkeys(FIELDS, 0))
                    for i, key in enumerate(FIELDS):
                        bucket[key] += row[i]
            points = [{"time": stamp, **buckets.get(stamp, dict.fromkeys(FIELDS, 0))}
                      for stamp in range(start, end, step)]
            return {"range": period, "interval_s": step, "start": start, "end": end, "time": now,
                    "started_at": total["started_at"], "updated_at": total["updated_at"],
                    "totals": {key: total[key] for key in FIELDS},
                    "period_totals": {key: sum(p[key] for p in points) for key in FIELDS}, "points": points,
                    "persistent": self.path != ":memory:", "storage_file": self.path,
                    "storage_error": self.storage_error,
                    "pending_requests": sum(row[0] for row in self.pending.values())}

    def close(self):
        with self.lock:
            self._flush()
            self.db.close()
