"""Prepare a separate local DeepSeek V4 profile. No downloads or changes to Qwen.

python tools/setup_deepseek4.py --model-dir H:/deepseek-v4-flash-0731/UD-Q8_K_XL
Build the backend first; see docs/deepseek-v4-flash-0731/DEEPSEEK4.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from gguf_reader import GGUFFile
from strata_tokenizer import extract


def inspect_model(first):
    first = Path(first).resolve()
    meta = GGUFFile(first).metadata
    if meta.get("general.architecture") != "deepseek4":
        raise ValueError("Expected deepseek4. Keep using the existing Qwen profile for qwen4exp.")
    count = int(meta.get("split.count", 1))
    match = re.fullmatch(r"(.+)-00001-of-(\d{5})\.gguf", first.name)
    if count > 1 and (not match or int(match[2]) != count):
        raise ValueError("Select the first GGUF shard (00001-of-...).")
    paths = [first] if count == 1 else [first.with_name(f"{match[1]}-{i:05d}-of-{count:05d}.gguf") for i in range(1, count + 1)]
    total = experts = tensors = 0
    names = set()
    for index, path in enumerate(paths):
        gguf = GGUFFile(path)  # copying/locked, missing and truncated headers fail here
        if count > 1 and (gguf.metadata.get("split.no") != index or gguf.metadata.get("split.count") != count):
            raise ValueError(f"Wrong shard metadata: {path.name}")
        end = gguf.data_start
        for tensor in gguf.tensors:
            if tensor.name in names:
                raise ValueError(f"Duplicate tensor: {tensor.name}")
            names.add(tensor.name)
            size = tensor.expected_bytes()
            if size is None:
                raise ValueError(f"Unknown tensor encoding: {tensor.name}: {tensor.type_name}")
            end = max(end, gguf.data_start + tensor.offset + size)
            total += size
            if "_exps." in tensor.name:
                experts += size
        if path.stat().st_size < end:
            raise ValueError(f"Incomplete shard: {path.name} ({path.stat().st_size} < {end} bytes)")
        tensors += len(gguf.tensors)
    if meta.get("split.tensors.count", tensors) != tensors:
        raise ValueError("The number of tensors does not match the split metadata.")
    return {"architecture": "deepseek4", "shards": len(paths), "tensors": tensors,
            "weight_bytes": total, "expert_bytes": experts, "dense_bytes": total - experts,
            "first_shard": str(first)}


def inspect_draft(path):
    """Reject wrong architecture, incompatible 0731 sidecars and partial copies."""
    path = Path(path).resolve()
    gguf = GGUFFile(path)
    meta = gguf.metadata
    required = {"general.architecture": "dflash", "dflash.block_count": 3,
                "dflash.embedding_length": 4096, "dflash.hyper_connection.count": 4,
                "dflash.block_size": 5, "dflash.target_layers": [41, 42, 43]}
    for key, value in required.items():
        if meta.get(key) != value:
            raise ValueError(f"Incompatible DeepSeek 0731 DSpark: {key}")
    if meta.get("dflash.sample_from_anchor", True) is not True or meta.get("dflash.attention.causal", False) is not False:
        raise ValueError("DSpark requires anchor-first non-causal decoding")
    names = {tensor.name for tensor in gguf.tensors}
    if not {"markov_w1.weight", "markov_w2.weight"} <= names:
        raise ValueError("Missing DSpark Markov head")
    total = experts = 0
    for tensor in gguf.tensors:
        size = tensor.expected_bytes()
        if size is None or path.stat().st_size < gguf.data_start + tensor.offset + size:
            raise ValueError(f"Incomplete or unsupported DSpark tensor: {tensor.name}")
        total += size
        if "_exps." in tensor.name:
            experts += size
    return {"path": str(path), "tensors": len(gguf.tensors), "weight_bytes": total,
            "expert_bytes": experts, "dense_bytes": total - experts,
            "confidence_head": "conf_proj.weight" in names}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--profile-tag", default="", help="Suffix for a separate profile and launcher, e.g. ud-q3-k-xl")
    ap.add_argument("--exe", type=Path, default=ROOT / "build-deepseek4/bin" / ("strata-deepseek4.exe" if sys.platform == "win32" else "strata-deepseek4"))
    ap.add_argument("--cuda-dir", type=Path)
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--gpu-expert-layers", type=int, default=0, help="Keep the last N layers' routed experts in VRAM; size depends on quantization")
    ap.add_argument("--expert-cache-mib", type=int, default=0, help="GPU LRU budget for individual expert matrices")
    ap.add_argument("--expert-cache-policy", choices=("lru", "frequency"), default="lru", help="Frequency admission keeps repeatedly used matrices over one-use weights; access counts decay")
    ap.add_argument("--expert-frequency-decay", type=int, default=0, help="Target matrix accesses per decay period; 0 keeps the cache-size default")
    ap.add_argument("--draft-expert-frequency-decay", type=int, default=0, help="DSpark matrix accesses per decay period; 0 keeps the cache-size default")
    ap.add_argument("--expert-cache-match-size", type=int, choices=(0, 1), default=0, help="Prefer same-size victims near the LRU tail of the fixed GPU cache; reduces arena fragmentation")
    ap.add_argument("--expert-stage-mib", type=int, default=0, help="Size of each of two pinned upload buffers")
    ap.add_argument("--expert-slab-mib", type=int, default=0, help="Dynamic GPU cache block size in MiB; 0 disables, fixed arena unaffected")
    ap.add_argument("--mmvq-token-batch", type=int, choices=(0, 1, 2), default=0, help="Experimental short MMVQ: 0 upstream, 1 routed, 2 routed+dense")
    ap.add_argument("--expert-pipeline", type=int, choices=(0, 1), default=0, help="Background mmap reads, four staging slots and a separate H2D stream; needs expert-stage-mib > 0")
    ap.add_argument("--expert-readers", type=int, choices=range(1, 5), default=2, help="Bounded reader concurrency; uses the existing four staging slots")
    ap.add_argument("--expert-read-mode", choices=("mmap", "file", "auto"), default="mmap", help="Windows file uses overlapped reads; auto queues nonresident prefill slices and uses mmap for decode")
    ap.add_argument("--expert-host-copy", choices=("crt", "avx2"), default="crt", help="Mmap to pinned copy; AVX2 requires pipeline and a compatible CPU/OS")
    ap.add_argument("--expert-early-refill", type=int, choices=(0, 1), default=0, help="Refill pinned slots after H2D while retaining device-slot protection")
    ap.add_argument("--expert-prefill-cache-hits", type=int, choices=(0, 1), default=0, help="Read existing GPU cache entries in prefill without admitting new weights")
    ap.add_argument("--draft-model", type=Path, help="Optional matching 0731 DSpark GGUF")
    ap.add_argument("--draft-shared-scratch", type=int, choices=(0, 1), default=0, help="Use one GPU compute buffer for sequential target/draft execution; KV and expert caches remain separate")
    ap.add_argument("--draft-max", type=int, default=3, choices=range(1, 6))
    ap.add_argument("--draft-min-confidence", type=float, default=0, help="0 disables filtering; otherwise keep the draft prefix with predicted acceptance >= this threshold")
    ap.add_argument("--draft-expert-cache-mib", type=int, default=1024)
    ap.add_argument("--draft-gpu-expert-layers", type=int, default=0, choices=range(4))
    ap.add_argument("--working-set-mib", type=int, default=0, help="Windows-only process working-set cap; 0 uses OS default")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    if not 0 <= args.expert_slab_mib <= 256:
        ap.error("expert-slab-mib must be in [0, 256]")
    if not all(0 <= value <= 1000000000 for value in (args.expert_frequency_decay, args.draft_expert_frequency_decay)):
        ap.error("frequency decay periods must be in [0, 1000000000]")
    if args.draft_shared_scratch and not args.draft_model:
        ap.error("draft-shared-scratch requires --draft-model")
    if args.expert_pipeline and args.expert_stage_mib <= 0:
        ap.error("expert-pipeline requires expert-stage-mib > 0")
    if (args.expert_host_copy != "crt" or args.expert_early_refill) and not args.expert_pipeline:
        ap.error("AVX2 host-copy and early-refill require expert-pipeline")
    if args.expert_pipeline and args.expert_read_mode == "file" and sys.platform != "win32":
        ap.error("native expert file reads require Windows")
    if args.profile_tag and not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", args.profile_tag):
        ap.error("--profile-tag must contain lowercase letters, numbers, hyphens or underscores")
    first = sorted(args.model_dir.glob("*-00001-of-*.gguf"))
    if not first:
        first = sorted(args.model_dir.glob("*.gguf"))
    if len(first) != 1:
        ap.error("The directory must contain exactly one model's first shard.")
    report = inspect_model(first[0])
    if args.draft_model:
        report["draft"] = inspect_draft(args.draft_model)
        if args.draft_min_confidence > 0 and not report["draft"]["confidence_head"]:
            ap.error("DSpark sidecar has no confidence head")
        if args.batch_size < args.draft_max + 1 or not 0 <= args.draft_expert_cache_mib <= 65536:
            ap.error("DSpark needs batch >= draft-max + 1 and a valid draft cache budget")
    if not 0 <= args.draft_min_confidence <= 1 or (args.draft_min_confidence > 0 and not args.draft_model):
        ap.error("draft-min-confidence must be in [0, 1] and needs a draft model when enabled")
    print(json.dumps(report, indent=2))
    if args.check_only:
        return
    if not args.exe.is_file():
        ap.error(f"Build the DeepSeek backend first: {args.exe}")
    if not (32 <= args.context <= 1048576 and 1 <= args.threads <= 256 and 1 <= args.batch_size <= 4096 and 1 <= args.port <= 65535 and 0 <= args.gpu_expert_layers <= 43 and args.working_set_mib >= 0 and 0 <= args.expert_cache_mib <= 65536 and 0 <= args.expert_stage_mib <= 256):
        ap.error("Invalid context, threads, batch size or port")
    quant_tag = re.sub(r"[^a-z0-9_-]+", "-", args.model_dir.name.lower()).strip("-") or "model"
    pack = ROOT / ("packs/deepseek4-" + (args.profile_tag or quant_tag))
    profile_name = "deepseek4" + ("-" + args.profile_tag if args.profile_tag else "")
    extract(first[0], pack)
    cfg = {"architecture": "deepseek4", "exe": str(args.exe.resolve()), "cwd": str(ROOT),
           "args": ["--native", report["first_shard"], "--max-context", str(args.context),
                    "--threads", str(args.threads), "--batch-size", str(args.batch_size),
                    "--gpu-layers", "99", "--gpu-expert-layers", str(args.gpu_expert_layers), "--conversation-cache-mib", "2048",
                    "--expert-cache-mib", str(args.expert_cache_mib), "--expert-stage-mib", str(args.expert_stage_mib),
                    "--expert-slab-mib", str(args.expert_slab_mib),
                    "--mmvq-token-batch", str(args.mmvq_token_batch),
                    "--expert-cache-policy", args.expert_cache_policy,
                    "--expert-frequency-decay", str(args.expert_frequency_decay),
                    "--draft-expert-frequency-decay", str(args.draft_expert_frequency_decay),
                    "--expert-cache-match-size", str(args.expert_cache_match_size),
                    "--expert-pipeline", str(args.expert_pipeline),
                    "--expert-readers", str(args.expert_readers), "--expert-read-mode", args.expert_read_mode,
                    "--expert-host-copy", args.expert_host_copy,
                    "--expert-early-refill", str(args.expert_early_refill),
                    "--expert-prefill-cache-hits", str(args.expert_prefill_cache_hits),
                    "--conversation-cache-slots", "4", "--conversation-cache-min-free-mib", "8192"],
           "tokenizer": str(pack / "tokenizer"), "model_name": "deepseek-v4-flash-0731" + ("-" + args.profile_tag if args.profile_tag else ""),
           "log": str(ROOT / f"strata-{profile_name}.log"), "host": "127.0.0.1", "port": args.port,
           "gpu": 0, "statistics_file": f"data/{profile_name}-usage-statistics.sqlite"}
    if args.cuda_dir:
        cfg["lib_dirs"] = [str((args.cuda_dir / sub).resolve()) for sub in ("bin", "bin/x64") if (args.cuda_dir / sub).is_dir()]
    if args.draft_model:
        cfg["args"] += ["--draft-model", report["draft"]["path"], "--draft-max", str(args.draft_max),
                        "--draft-shared-scratch", str(args.draft_shared_scratch),
                        "--draft-min-confidence", str(args.draft_min_confidence),
                        "--draft-expert-cache-mib", str(args.draft_expert_cache_mib),
                        "--draft-gpu-expert-layers", str(args.draft_gpu_expert_layers)]
    if args.working_set_mib:
        if sys.platform != "win32":
            ap.error("--working-set-mib is Windows-only")
        cfg["args"] += ["--working-set-mib", str(args.working_set_mib)]
    path = ROOT / f"strata-{profile_name}.json"
    path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    command = [sys.executable, str(ROOT / "serve/server.py"), "--engine", "strata", "--config", str(path), "--port", str(args.port), "--open"]
    if sys.platform == "win32":
        import subprocess
        (ROOT / f"run-{profile_name}.bat").write_text('@echo off\ncd /d "' + str(ROOT) + '"\n' + subprocess.list2cmdline(command) + '\nif errorlevel 1 pause\n', encoding="utf-8")
    else:
        launcher = ROOT / f"run-{profile_name}.sh"
        launcher.write_text("#!/bin/sh\ncd " + shlex.quote(str(ROOT)) + "\nexec " + shlex.join(command) + "\n")
        launcher.chmod(0o755)
    print(f"Prepared {path}; Qwen configuration unchanged.")


if __name__ == "__main__":
    main()
