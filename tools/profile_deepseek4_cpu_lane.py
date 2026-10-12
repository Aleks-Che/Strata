"""Run the existing fixed-history probe with CPU trace A/B controls.

Uses a caller-supplied engine configuration and exact token IDs, does not modify
the production profile. Every result directory must be new. Capture, if requested,
is a separate run and is excluded from timing comparisons.
"""
import argparse
import hashlib
import json
import sys
import threading
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve.server import StrataEngine, child_env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True, help="JSON array of token IDs")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-exe", type=Path)
    parser.add_argument("--sequence", default="off,on,on,off")
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--capture-tokens", type=int, choices=range(1,9), help="capture only this CPU window size")
    parser.add_argument("--gpu-events", action="store_true", help="disable graphs in ALL controls and measure CUDA spans in trace runs")
    parser.add_argument("--disk", help="psutil physical-disk name, for all test phases together")
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    ids = json.loads(args.tokens.read_text(encoding="utf-8-sig"))
    if not isinstance(ids, list) or len(ids) != 68 or any(type(x) is not int for x in ids):
        parser.error("the current engine fixed-history probe requires exactly 68 integer IDs")
    sequence = args.sequence.split(",")
    if any(x not in {"baseline", "off", "on", "gpu"} for x in sequence):
        parser.error("sequence modes: baseline,off,on,gpu")
    if "baseline" in sequence and not args.baseline_exe:
        parser.error("baseline mode requires --baseline-exe")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out/"tokens.json").write_text(json.dumps(ids)+"\n", encoding="utf-8")
    binaries = {"experimental": config["exe"]}
    if args.baseline_exe:
        binaries["baseline"] = str(args.baseline_exe.resolve())
    manifest = {"config": config, "sequence": sequence, "disk": args.disk,
                "executables": {name: {"path": path, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
                                for name, path in binaries.items()}}
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    results = []
    for index, mode in enumerate(sequence + (["capture"] if args.capture else [])):
        label = f"{index:02d}-{mode}"
        c = json.loads(json.dumps(config))
        c.setdefault("env", {})
        for key in ["STRATA_DS4_CPU_TRACE", "STRATA_DS4_CPU_CAPTURE_DIR", "STRATA_DS4_CPU_CAPTURE_JOBS", "STRATA_DS4_CPU_CAPTURE_TOKENS",
                    "STRATA_DS4_CPU_TRACE_GPU", "STRATA_DS4_SCHED_PROFILE", "STRATA_DS4_TRANSPORT_PROFILE"]:
            c["env"][key] = ""
        if args.gpu_events:
            c["env"]["GGML_CUDA_DISABLE_GRAPHS"] = "1"
        for option in ["--draft-model", "--draft-max", "--draft-expert-cache-mib", "--draft-gpu-expert-layers",
                       "--draft-shared-scratch", "--draft-expert-frequency-decay"]:
            if option in c["args"]:
                i = c["args"].index(option)
                del c["args"][i:i+2]
        if mode == "baseline":
            c["exe"] = binaries["baseline"]
        if mode == "gpu":
            for key in list(c["env"]):
                if key.startswith(("STRATA_DS4_CPU_", "STRATA_DS4_RAM_")):
                    c["env"][key] = "0"
            if "--expert-cpu-threads" in c["args"]:
                c["args"][c["args"].index("--expert-cpu-threads")+1] = "0"
            for key in ["STRATA_DS4_CPU_TRACE", "STRATA_DS4_CPU_CAPTURE_DIR"]:
                c["env"][key] = ""
        probe = out/f"{label}.json"
        c["env"]["STRATA_FIXED_WORKLOAD"] = str(probe)
        if mode in {"on", "capture"}:
            c["env"]["STRATA_DS4_CPU_TRACE"] = str(out/f"{label}.tsv")
            c["env"]["STRATA_DS4_CPU_TRACE_GPU"] = "1" if args.gpu_events else "0"
        if mode == "capture":
            c["env"]["STRATA_DS4_CPU_CAPTURE_DIR"] = str(out/"captures")
            c["env"]["STRATA_DS4_CPU_CAPTURE_JOBS"] = "2"
            c["env"]["STRATA_DS4_CPU_CAPTURE_TOKENS"] = str(args.capture_tokens or 0)
            c["env"]["STRATA_DS4_SCHED_PROFILE"] = str(out/f"{label}-scheduler.tsv")
        (out/f"{label}-config.json").write_text(json.dumps(c, indent=2)+"\n", encoding="utf-8")
        engine = None
        expired = threading.Event()
        def timeout():
            expired.set()
            if engine is not None:
                engine.proc.kill()
        timer = threading.Timer(args.timeout, timeout)
        timer.start()
        def disk_bytes():
            return psutil.disk_io_counters(perdisk=True)[args.disk].read_bytes if args.disk else None
        try:
            print("start", label, flush=True)
            engine = StrataEngine(c["exe"], c["args"], cwd=c["cwd"], log=str(out/f"{label}.log"), env=child_env(c))
            if expired.is_set():
                engine.proc.kill()
                raise TimeoutError(label)
            before = disk_bytes()
            list(engine.generate(ids, 1, {"temperature": 0}, threading.Event(), session_id="fixed"))
            after = disk_bytes()
            if expired.is_set():
                raise TimeoutError(label)
            data = json.loads(probe.read_text(encoding="utf-8"))
            if data["tokens"] != ids:
                raise RuntimeError("fixed input differs")
            result = {"label": label, "mode": mode, "runs": data["runs"],
                      "ssd_bytes_all_passes": after-before if before is not None else None}
            results.append(result)
            (out/"results.json").write_text(json.dumps(results, indent=2)+"\n", encoding="utf-8")
            print(json.dumps({"label": label, "ssd": result["ssd_bytes_all_passes"],
                              "warm": [{k: v for k, v in r.items() if k in {"phase", "chunk", "decode_ms"}}
                                       for r in data["runs"] if r["repeat"] >= 2]}), flush=True)
        finally:
            timer.cancel()
            if engine is not None:
                engine.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
