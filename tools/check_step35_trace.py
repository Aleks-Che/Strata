"""Check real Step CUDA stream overlap separately from throughput measurements."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from check_step35_engine import Engine
from check_step35_model import Monitor
from setup_step35 import first_shard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--readers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--chunk-mib", type=int, choices=(4, 8, 16), default=8)
    args = parser.parse_args()
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    model = first_shard(args.gguf)
    reference = json.loads(args.reference.read_text(encoding="utf8"))
    if reference["status"] != "pass" or Path(reference["model"]).resolve() != model.resolve():
        parser.error("a passing baseline for this model is required")
    report = {"status": "error", "scope": "diagnostic CUDA-event overlap, not a speed benchmark",
              "model": str(model), "engine_sha256": hashlib.sha256(args.engine.read_bytes()).hexdigest(),
              "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
              "pipeline_readers": args.readers, "pipeline_chunk_mib": args.chunk_mib}
    engine, monitor = None, Monitor()
    try:
        trace = directory / "cuda-trace.json"
        engine = Engine(args.engine.resolve(), model, directory, logits=True, on_start=monitor.start,
                        extra_args=["--expert-cache-mib", "auto", "--expert-pipeline-readers", str(args.readers),
                                    "--expert-pipeline-chunk-mib", str(args.chunk_mib),
                                    "--trace-file", str(trace), "--trace-graphs", "8"])
        report["info"] = engine.info
        prompt = reference["prompts"][0]["ids"]
        result = engine.generate(prompt, reference["predict"], sampling="temperature=0")
        expected = reference["runs"][0]["requests"][0]
        raw = (directory / "logits.f32").read_bytes()
        report["exact_ids"] = result["ids"] == expected["ids"]
        report["exact_logits"] = len(raw) == expected["logits_bytes"] and hashlib.sha256(raw).hexdigest() == expected["logits_sha256"]
        if not report["exact_ids"] or not report["exact_logits"]:
            raise ValueError("traced request differs from P1 reference")
        engine.close()
        report["exit_code"] = engine.process.returncode
        if engine.process.returncode:
            raise ValueError("traced engine did not exit cleanly")
        data = json.loads(trace.read_text(encoding="utf8"))
        graphs = data["graphs"]
        if len(graphs) != 8:
            raise ValueError("incomplete bounded CUDA trace")
        phases = {"prefill": [], "decode": []}
        for index, graph in enumerate(graphs):
            if graph["cancelled"]:
                raise ValueError("unexpected cancelled graph")
            for key in ("h2d_ms", "compute_ms", "overlap_ms"):
                if not math.isfinite(graph[key]) or graph[key] < 0:
                    raise ValueError("invalid event timing")
            if graph["overlap_ms"] > min(graph["h2d_ms"], graph["compute_ms"]) + 1e-6:
                raise ValueError("overlap exceeds measured duration")
            # The fixed batch=17 engine submits each prompt microbatch, then
            # one graph per decode step. No synthetic GPU work is inserted.
            phases["prefill" if index < math.ceil(len(prompt)/17) else "decode"].append(graph)
        report["phases"] = {phase: {"graphs": len(values),
            **{key: sum(g[key] for g in values) for key in ("h2d_ms", "compute_ms", "overlap_ms")}}
            for phase, values in phases.items()}
        if any(values["overlap_ms"] <= 0 for values in report["phases"].values()):
            raise ValueError("no measured H2D/compute overlap in prefill or decode")
        report["trace_sha256"] = hashlib.sha256(trace.read_bytes()).hexdigest()
        report["status"] = "pass"
    except Exception as error:
        report["error"] = str(error)
    finally:
        if engine and not engine.stderr.closed:
            engine.close()
        monitor.close()
        report["memory_samples"] = monitor.samples
        report["monitor_error"] = monitor.error
        if monitor.error:
            report.update(status="error", error=monitor.error)
    (directory / "trace-report.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf8")
    print(report["status"], report.get("error", ""), report.get("phases", {}), flush=True)
    return int(report["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
