"""Persist the requested GPU cache policy; the engine owns sampling and eviction."""
import json
import os
import tempfile
import threading
import uuid
from pathlib import Path


class VramSettings:
    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.lock = threading.RLock()
        self.value = {"mode": "config", "matrices": 0, "target_mib": 0, "reserve_mib": 1024}
        self.revision = ""
        self.storage_error = None
        if self.path:
            try:
                self.value = self.validate(json.loads(self.path.read_text(encoding="utf-8")))
                self.revision = uuid.uuid4().hex
            except FileNotFoundError:
                pass
            except (OSError, ValueError) as error:
                self.storage_error = f"Could not load VRAM settings: {error}"

    @staticmethod
    def validate(value):
        if not isinstance(value, dict) or value.get("mode") not in ("config", "count", "vram"):
            raise ValueError("mode must be config, count or vram")
        clean = {"mode": value["mode"]}
        for name, low, high in (("matrices", 0, 1000000), ("target_mib", 0, 1048576), ("reserve_mib", 128, 1048576)):
            n = value.get(name)
            if type(n) is not int or not low <= n <= high:
                raise ValueError(f"{name} must be a whole number between {low} and {high}")
            clean[name] = n
        if clean["mode"] == "vram" and not clean["target_mib"]:
            raise ValueError("target_mib must be positive in VRAM mode")
        return clean

    def wire(self):
        with self.lock:
            p = self.value
            mode = {"config": 0, "count": 1, "vram": 2}[p["mode"]]
            return f'{self.revision} {mode} {p["matrices"]} {p["target_mib"]} {p["reserve_mib"]}' if self.revision else ""

    def snapshot(self, engine):
        with self.lock:
            live = dict(getattr(engine, "vram_status", {}) or {})
            supported = bool(getattr(engine, "can_vram_control", False))
            alive = bool(supported and engine.alive())
            return {"settings": dict(self.value), "supported": supported, "alive": alive,
                    "pending": bool(self.revision and (not alive or live.get("applied") != self.revision)),
                    "persistent": self.path is not None, "storage_error": self.storage_error, "live": live}

    def update(self, value):
        clean = self.validate(value)
        with self.lock:
            temporary = None
            try:
                if self.path:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent,
                                                     prefix=self.path.name + ".", suffix=".tmp", delete=False) as f:
                        temporary = Path(f.name)
                        json.dump(clean, f)
                        f.write("\n")
                        f.flush()
                        os.fsync(f.fileno())
                    os.replace(temporary, self.path)
            except OSError as error:
                self.storage_error = "Could not save VRAM settings; previous settings remain active"
                raise OSError(self.storage_error) from error
            finally:
                if temporary:
                    temporary.unlink(missing_ok=True)
            self.value, self.revision = clean, uuid.uuid4().hex
            self.storage_error = None
