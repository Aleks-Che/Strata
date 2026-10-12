"""Summarize per-job CPU lane diagnostics without mixing CPU and CUDA clocks."""
import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def summarize(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    phases = defaultdict(list)
    jobs = set()
    for row in rows:
        job = int(row["job"])
        if job in jobs:
            raise ValueError(f"duplicate job {job}")
        jobs.add(job)
        for key, value in row.items():
            if key not in {"phase", "owner", "layer"}:
                row[key] = float(value) if key.endswith("_ms") else int(value)
                if not math.isfinite(row[key]) or row[key] < 0:
                    raise ValueError(f"negative {key} in job {job}")
        if row["rows"] and row["completed"]:
            if not 0 < row["queued_us"] <= row["callback_us"] <= row["callback_end_us"]:
                raise ValueError(f"invalid callback chronology in job {job}")
            if not 0 < row["unique_experts"] <= row["rows"]:
                raise ValueError(f"invalid expert count in job {job}")
        phases[row["phase"]].append(row)
    result = {}
    for phase, entries in phases.items():
        executed = [r for r in entries if r["rows"] and r["completed"]]
        item = {"plans": len(entries), "executed": len(executed),
                "declined": sum(r["rows"] == 0 for r in entries),
                "incomplete": sum(r["rows"] > 0 and not r["completed"] for r in entries)}
        sums = ["select_us", "lease_us", "lease_calls", "lease_failures", "retire_us", "route_get_us", "route_wait_us", "route_reused", "route_measured"]
        item.update({key: sum(r[key] for r in entries) for key in sums})
        fields = ["pack_us", "prepare_us", "gate_up_us", "hidden_quant_us", "down_us",
                  "gu_dispatch_us", "gu_first_worker_us", "gu_last_worker_us",
                  "down_dispatch_us", "down_first_worker_us", "down_last_worker_us",
                  "input_ms", "callback_stream_ms", "output_ms", "join_ms"]
        item["executed_spans"] = {}
        for key in fields:
            values = sorted(r[key] for r in executed if not key.endswith("_ms") or r["gpu_events"])
            item["executed_spans"][key] = {
                "samples": len(values), "sum": sum(values) if values else None, "median": statistics.median(values) if values else None,
                "p95": values[min(len(values)-1, int((len(values)-1)*0.95))] if values else None,
            }
        item["callback_queue_us"] = sum(r["callback_us"]-r["queued_us"] for r in executed)
        item["rows"] = sum(r["rows"] for r in executed)
        item["unique_experts_per_job_sum"] = sum(r["unique_experts"] for r in executed)
        result[phase] = item
    return {"jobs": len(rows), "phases": result, "interpretation": {
        "select_us": "Includes lease_us; do not add these sums together.",
        "route_wait_us": "Existing scheduler wait includes preceding queued GPU computation; not a measure of pure route-transfer overhead.",
        "route_get_us": "The D2H API itself can block on queued GPU work and pageable host delivery; not all of this span is removable overhead.",
        "callback_queue_us": "Host enqueue to callback start, including GPU dependencies and D2H; not just dispatcher overhead.",
        "pool": "Dispatch/worker-start metrics are included in gate_up_us/down_us; workers counts only participants that claimed a job.",
        "cuda": "Stream spans include launch gaps. join_ms surrounds the CPU-result wait; it is not pure CPU execution time.",
        "clocks": "Host steady_clock and CUDA event timestamps are never subtracted from one another.",
        "retire_us": "Existing retirement barrier; may overlap spans above. Trace/capture output I/O is outside it.",
    }}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(args.trace)
    text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
