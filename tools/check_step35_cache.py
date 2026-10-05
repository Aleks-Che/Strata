"""Full Step cache on/off comparison against saved exact baseline logits.

ABBA mode order limits ordering bias; file cache is not forcibly purged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

from check_step35_engine import Engine
from check_step35_model import Monitor
from setup_step35 import first_shard


def check(binary, model, reference_path, output, order, rounds, pipeline_order=None):
    reference = json.loads(reference_path.read_text(encoding="utf8"))
    if reference["status"] != "pass" or Path(reference["model"]).resolve() != model.resolve():
        raise ValueError("a passing baseline for this model is required")
    report = {"status": "error", "scope": "full-model cache exact logits and paired sequential on/off timings",
              "reference": str(reference_path), "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
              "model": str(model), "engine_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
              "order": order, "rounds": rounds, "context": 2048, "batch": 17, "kv": "f32",
              "mtp": False, "pipeline": bool(pipeline_order), "pipeline_order": pipeline_order,
              "file_cache_purged": False, "runs": []}
    if pipeline_order:
        if len(pipeline_order) != len(order):
            raise ValueError("one pipeline configuration per cache run is required")
        report["scope"] = "full-model pipeline exact logits and sequential configuration timings"
    report_name = "pipeline-model-report.json" if pipeline_order else "cache-model-report.json"
    def save():
        (output / report_name).write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf8")
    try:
        for run_index, cap in enumerate(order):
            directory = output / f"{run_index}-cache-{cap}"
            directory.mkdir(parents=True, exist_ok=True)
            record = {"cache_mib": cap, "requests": []}
            extra_args = ["--expert-cache-mib", cap]
            if pipeline_order:
                readers, chunk = pipeline_order[run_index]
                record.update(pipeline_readers=readers, pipeline_chunk_mib=chunk)
                extra_args += ["--expert-pipeline-readers", str(readers), "--expert-pipeline-chunk-mib", str(chunk)]
            report["runs"].append(record)
            engine, monitor = None, Monitor()
            try:
                print(f"Loading cache={cap} run={run_index} args={extra_args}", flush=True)
                start = time.perf_counter()
                engine = Engine(binary, model, directory, logits=True, on_start=monitor.start,
                                extra_args=extra_args)
                record["ready_seconds"] = time.perf_counter()-start
                record["info"] = engine.info
                for repeat in range(rounds):
                    for index, prompt in enumerate(reference["prompts"]):
                        expected = reference["runs"][0]["requests"][index]
                        result = engine.generate(prompt["ids"], reference["predict"], sampling="temperature=0")
                        raw = (directory / "logits.f32").read_bytes()
                        result.update(prompt_index=index, repeat=repeat,
                                      logits_sha256=hashlib.sha256(raw).hexdigest(), logits_bytes=len(raw))
                        result["exact_ids"] = result["ids"] == expected["ids"]
                        result["exact_logits"] = result["logits_sha256"] == expected["logits_sha256"] and len(raw) == expected["logits_bytes"]
                        record["requests"].append(result)
                        if not result["exact_ids"] or not result["exact_logits"]:
                            raise ValueError(f"cache={cap} prompt={index} logits/IDs differ from P1")
                        print(f"cache={cap} repeat={repeat} prompt={index}: exact, {len(result['ids'])} tokens, prefill={result['prefill_ms']/1000:.2f}s generation={result['generation_wall_ms']/1000:.2f}s", flush=True)
                        save()
                # Cached cancellation must leave a fresh request numerically unchanged.
                cancelled = engine.generate(reference["prompts"][0]["ids"], 512, cancel=True)
                record["cancelled"] = cancelled
                if cancelled["finish"] != "cancel":
                    raise ValueError("STOP did not cancel cache run")
                after = engine.generate(reference["prompts"][0]["ids"], 16, sampling="temperature=0")
                record["after_cancel_exact"] = after["ids"] == reference["runs"][0]["requests"][0]["ids"][:16]
                if not record["after_cancel_exact"]:
                    raise ValueError("fresh request after cancel differs")
                engine.close()
                record["exit_code"] = engine.process.returncode
                if record["exit_code"]:
                    raise ValueError("engine did not exit cleanly")
            finally:
                if engine and not engine.stderr.closed:
                    engine.close()
                monitor.close()
                record["monitor_error"] = monitor.error
                record["memory_samples"] = monitor.samples
                record["metrics"] = [json.loads(line.split(" ", 1)[1]) for line in
                    (directory / "stderr.log").read_text(encoding="utf8", errors="replace").splitlines()
                    if line.startswith("STRATA_STEP_REQUEST ")]
                save()
            if monitor.error:
                raise ValueError(monitor.error)
        report["status"] = "pass"
    except Exception as error:
        report["error"] = str(error)
    save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--order", default="0,auto,auto,0")
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()
    order = args.order.split(",")
    if not 1 <= args.rounds <= 4 or len(order) > 8 or any(x != "auto" and (not x.isdigit() or int(x)>1048576) for x in order):
        parser.error("invalid rounds/cache order")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = check(args.engine.resolve(), first_shard(args.gguf), args.reference, args.output_dir.resolve(), order, args.rounds)
    print(report["status"], report.get("error", ""), flush=True)
    return int(report["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
