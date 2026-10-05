"""Compare Step expert pipeline configurations against saved exact full-model logits."""
import argparse
from pathlib import Path

from check_step35_cache import check
from setup_step35 import first_shard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--order", default="0:8,1:8,2:8,0:8", help="readers:chunk_mib configurations, in run order")
    parser.add_argument("--cache-mib", default="auto")
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()
    try:
        configs = [tuple(map(int, value.split(":"))) for value in args.order.split(",")]
        if not 1 <= len(configs) <= 8 or any(len(c) != 2 or c[0] not in (0, 1, 2) or c[1] not in (4, 8, 16) for c in configs):
            raise ValueError()
    except ValueError:
        parser.error("invalid pipeline order")
    if not 1 <= args.rounds <= 4 or (args.cache_mib != "auto" and
            (not args.cache_mib.isdigit() or int(args.cache_mib) > 1048576)):
        parser.error("invalid rounds/cache budget")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = check(args.engine.resolve(), first_shard(args.gguf), args.reference, args.output_dir.resolve(),
                   [args.cache_mib] * len(configs), args.rounds, pipeline_order=configs)
    print(report["status"], report.get("error", ""), flush=True)
    return int(report["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
