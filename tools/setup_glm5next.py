"""Inspect local Unsloth glm5next GGUF v3 shards without reading weight payloads.

python tools/setup_glm5next.py --model-dir H:/GLM-5.3-Flash-GGUF/UD-Q3_K_XL --check-only

Without --check-only, export the tokenizer and a separate synchronous GLM profile
for the already-built backend. No downloads or changes to other model profiles.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import struct
import os
import subprocess
import sys

if __package__:
    from .gguf_reader import BLOCK_GEOMETRY, GGUFFile
else:
    from gguf_reader import BLOCK_GEOMETRY, GGUFFile


def _integer(meta, key, minimum=1):
    value = meta.get(key)
    if type(value) is not int or value < minimum:
        raise ValueError(f"Missing or invalid metadata: {key} (expected integer >= {minimum})")
    return value


def _model_metadata(meta):
    for key, expected in (("general.architecture", "glm5next"),
                          ("tokenizer.ggml.model", "gpt2"), ("tokenizer.ggml.pre", "glm4")):
        if meta.get(key) != expected:
            raise ValueError(f"Expected {key}={expected!r}, got {meta.get(key)!r}")
    keys = ("block_count", "nextn_predict_layers", "embedding_length", "feed_forward_length",
            "expert_feed_forward_length", "expert_count", "expert_used_count", "vocab_size",
            "context_length")
    values = {key: _integer(meta, "glm5next." + key) for key in keys}
    blocks = values["block_count"]
    mtp = values["nextn_predict_layers"]
    if mtp >= blocks:
        raise ValueError("NextN layer count must be smaller than block_count")
    main = blocks - mtp
    dense = _integer(meta, "glm5next.leading_dense_block_count", 0)
    if dense > main or values["expert_used_count"] > values["expert_count"]:
        raise ValueError("Invalid dense block count or expert_used_count")
    heads = meta.get("glm5next.attention.head_count_kv")
    if not isinstance(heads, list) or len(heads) != blocks or any(type(h) is not int or h < 0 for h in heads):
        raise ValueError("glm5next.attention.head_count_kv must describe every block including NextN")
    tokens = meta.get("tokenizer.ggml.tokens")
    if not isinstance(tokens, list) or len(tokens) != values["vocab_size"] or any(not isinstance(t, str) for t in tokens):
        raise ValueError("tokenizer.ggml.tokens does not match vocab_size")
    merges = meta.get("tokenizer.ggml.merges")
    if not isinstance(merges, list) or any(not isinstance(t, str) for t in merges):
        raise ValueError("Missing or invalid tokenizer.ggml.merges")
    if not isinstance(meta.get("tokenizer.chat_template"), str) or not meta["tokenizer.chat_template"]:
        raise ValueError("Missing tokenizer.chat_template")
    for suffix in ("eos_token_id", "eot_token_id", "eom_token_id"):
        if _integer(meta, "tokenizer.ggml." + suffix, 0) >= len(tokens):
            raise ValueError(f"tokenizer.ggml.{suffix} is outside the vocabulary")
    return values, main, dense


def _read(path):
    try:
        gguf = GGUFFile(path)
    except (ValueError, KeyError, struct.error, OverflowError) as exc:
        raise ValueError(f"Invalid GGUF header: {path.name}: {exc}") from exc
    alignment = gguf.metadata.get("general.alignment", 32)
    if type(alignment) is not int or alignment <= 0 or alignment & (alignment - 1):
        raise ValueError(f"Invalid general.alignment: {path.name}")
    return gguf


def inspect_model(first, *, tensor_details=False, loader_contract=False):
    """Validate structure and core GLM shapes; full loader tensor mapping is P0.2."""
    first = Path(first).resolve()
    initial = _read(first)
    meta = initial.metadata
    values, main_blocks, dense_blocks = _model_metadata(meta)
    count = _integer(meta, "split.count") if "split.count" in meta else 1
    match = re.fullmatch(r"(.+)-00001-of-(\d{5})\.gguf", first.name)
    if count > 1 and (not match or int(match[2]) != count):
        raise ValueError("Select the first GGUF shard (00001-of-...).")
    paths = [first] if count == 1 else [
        first.with_name(f"{match[1]}-{i:05d}-of-{count:05d}.gguf") for i in range(1, count + 1)]
    missing = [p.name for p in paths if not p.is_file()]
    if missing:
        raise ValueError("Missing GGUF shard(s): " + ", ".join(missing))
    expected_count = _integer(meta, "split.tensors.count") if count > 1 or "split.tensors.count" in meta else None
    tensors = {}
    shards = []
    details = []
    types = Counter()
    groups = {name: {"tensors": 0, "weight_bytes": 0, "expert_bytes": 0}
              for name in ("main", "mtp")}
    seen_blocks = set()
    for index, path in enumerate(paths):
        gguf = initial if index == 0 else _read(path)
        if count > 1 or "split.no" in gguf.metadata:
            for key, expected in (("split.no", index), ("split.count", count),
                                  ("split.tensors.count", expected_count)):
                actual = gguf.metadata.get(key)
                if type(actual) is not int or actual != expected:
                    raise ValueError(f"Wrong shard metadata: {path.name}: {key}")
        # Later shards normally contain only split metadata. Reject conflicting
        # model metadata if present, rather than silently mixing model families.
        for key, value in gguf.metadata.items():
            if key in meta and (key.startswith("glm5next.") or key.startswith("tokenizer.") or key == "general.architecture") and value != meta[key]:
                raise ValueError(f"Conflicting metadata: {path.name}: {key}")
        file_bytes = path.stat().st_size
        required_end = gguf.header_end
        ranges = []
        for tensor in gguf.tensors:
            name = tensor.name
            if not name or name in tensors:
                raise ValueError(f"Empty or duplicate tensor: {name}")
            if not 1 <= len(tensor.shape) <= 4 or any(d <= 0 for d in tensor.shape):
                raise ValueError(f"Invalid tensor shape: {name}: {tensor.shape}")
            geometry = BLOCK_GEOMETRY.get(tensor.type_name)
            if geometry is None:
                raise ValueError(f"Unknown tensor encoding: {name}: {tensor.type_name}")
            if tensor.shape[0] % geometry[0]:
                raise ValueError(f"Invalid quantized row shape: {name}: {tensor.shape}")
            size = tensor.expected_bytes()
            start = gguf.data_start + tensor.offset
            end = start + size
            if tensor.offset % gguf.alignment:
                raise ValueError(f"Unaligned tensor offset: {name}")
            if end > file_bytes:
                raise ValueError(f"Incomplete shard: {path.name}: {name} ({file_bytes} < {end} bytes)")
            ranges.append((start, end, name))
            required_end = max(required_end, end)
            block = re.match(r"^blk\.(\d+)\.", name)
            layer = int(block[1]) if block else None
            if name.startswith("blk.") and block is None:
                raise ValueError(f"Invalid block tensor name: {name}")
            if layer is not None:
                if layer >= values["block_count"]:
                    raise ValueError(f"Tensor outside block_count: {name}")
                seen_blocks.add(layer)
            is_mtp = layer is not None and layer >= main_blocks
            if ".nextn." in name and not is_mtp:
                raise ValueError(f"NextN tensor outside MTP blocks: {name}")
            group = groups["mtp" if is_mtp else "main"]
            group["tensors"] += 1
            group["weight_bytes"] += size
            if "_exps." in name:
                group["expert_bytes"] += size
            types[tensor.type_name] += 1
            tensors[name] = tensor
            if tensor_details:
                details.append({"name": name, "shape": tensor.shape, "type": tensor.type_name,
                                "shard": index, "file_offset": start, "bytes": size})
        ranges.sort()
        for previous, current in zip(ranges, ranges[1:]):
            if current[0] < previous[1]:
                raise ValueError(f"Overlapping tensors: {path.name}: {previous[2]}, {current[2]}")
        if file_bytes < required_end:
            raise ValueError(f"Incomplete header: {path.name}")
        shards.append({"name": path.name, "split_no": index, "tensors": len(gguf.tensors),
                       "file_bytes": file_bytes, "header_end": gguf.header_end,
                       "data_start": gguf.data_start, "required_end": required_end,
                       "metadata_only": not gguf.tensors})
    if expected_count is not None and len(tensors) != expected_count:
        raise ValueError("The number of tensors does not match split.tensors.count")
    if seen_blocks != set(range(values["block_count"])):
        raise ValueError("Missing block tensors: " + str(sorted(set(range(values["block_count"])) - seen_blocks)))

    def require(name, shape):
        tensor = tensors.get(name)
        if tensor is None or tensor.shape != shape:
            raise ValueError(f"Missing tensor or wrong shape: {name}: expected {shape}")

    width, vocab = values["embedding_length"], values["vocab_size"]
    require("token_embd.weight", [width, vocab])
    require("output.weight", [width, vocab])
    require("output_norm.weight", [width])
    for layer in range(values["block_count"]):
        prefix = f"blk.{layer}."
        require(prefix + "attn_norm.weight", [width])
        require(prefix + "ffn_norm.weight", [width])
        if layer < dense_blocks:
            ff = values["feed_forward_length"]
            for matrix, shape in (("gate", [width, ff]), ("up", [width, ff]), ("down", [ff, width])):
                require(prefix + f"ffn_{matrix}.weight", shape)
        else:
            ff, experts = values["expert_feed_forward_length"], values["expert_count"]
            require(prefix + "ffn_gate_inp.weight", [width, experts])
            for matrix, shape in (("gate", [width, ff, experts]), ("up", [width, ff, experts]), ("down", [ff, width, experts])):
                require(prefix + f"ffn_{matrix}_exps.weight", shape)
        if layer >= main_blocks:
            require(prefix + "nextn.eh_proj.weight", [2 * width, width])
            for norm in ("enorm", "hnorm", "shared_head_norm"):
                require(prefix + f"nextn.{norm}.weight", [width])
    for group in groups.values():
        group["dense_bytes"] = group["weight_bytes"] - group["expert_bytes"]
    report = {"schema_version": 1, "architecture": "glm5next", "first_shard": str(first),
              "validation_scope": "headers, split metadata, tensor geometry/ranges and core GLM/NextN shapes; no payload hashes or loader/GPU validation",
              "shards": count, "tensors": len(tensors), "block_count": values["block_count"],
              "main_blocks": main_blocks, "mtp_blocks": values["nextn_predict_layers"],
              "file_bytes": sum(s["file_bytes"] for s in shards),
              "weight_bytes": sum(g["weight_bytes"] for g in groups.values()),
              "groups": groups, "tensor_type_counts": dict(sorted(types.items())), "shard_details": shards}
    if tensor_details:
        report["tensor_details"] = details
    if loader_contract:
        if __package__:
            from .glm5next_loader_contract import validate_loader_contract
        else:
            from glm5next_loader_contract import validate_loader_contract
        report["loader_contract"] = validate_loader_contract(meta, tensors)
    return report


ROOT = Path(__file__).resolve().parents[1]


def prepare_profile(report, exe, profile, *, context=2048, batch=16, threads=4, port=8081, cuda_dir=None,
                    ram_target_percent=0, vram_target_percent=0):
    """Write a new profile only after validating backend identity and settings."""
    if not (32 <= context <= 1048576 and 1 <= batch <= min(context, 4096) and
            1 <= threads <= 256 and 1 <= port <= 65535):
        raise ValueError("Invalid context, batch, threads or port")
    if any(p != 0 and not 10 <= p <= 95 for p in (ram_target_percent, vram_target_percent)):
        raise ValueError("Memory targets must be 0 (off) or 10..95 percent")
    exe, profile = Path(exe).resolve(), Path(profile).resolve()
    if not exe.is_file():
        raise ValueError(f"Build strata-glm5next first: {exe}")
    pack = profile.parent / "packs" / profile.stem
    if profile.exists() or pack.exists():
        raise ValueError("Profile/tokenizer destination already exists; choose a new --profile path")
    env = dict(os.environ)
    dirs = [str((Path(cuda_dir) / sub).resolve()) for sub in ("bin", "bin/x64")
            if (Path(cuda_dir) / sub).is_dir()] if cuda_dir else []
    if dirs:
        key = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
        env[key] = os.pathsep.join(dirs + [env.get(key, "")])
    result = subprocess.run([str(exe), "--version"], env=env, capture_output=True, text=True, timeout=30, check=True)
    identity = json.loads(result.stdout)
    if __package__:
        from .glm5next_loader_contract import LOADER_SHA
        from .strata_tokenizer import extract
    else:
        from glm5next_loader_contract import LOADER_SHA
        from strata_tokenizer import extract
    if (not isinstance(identity, dict) or identity.get("architecture") != "glm5next" or identity.get("protocol_version") != 1 or
            identity.get("source_sha") != LOADER_SHA):
        raise ValueError("Expected the audited GLM backend with pipe protocol v1")
    profile.parent.mkdir(parents=True, exist_ok=True)
    extract(report["first_shard"], pack)
    cfg = {"architecture": "glm5next", "exe": str(exe), "cwd": str(ROOT),
           "args": ["--native", report["first_shard"], "--max-context", str(context),
                    "--batch-size", str(batch), "--threads", str(threads)],
           "tokenizer": str(pack / "tokenizer"), "model_name": "glm-5.3-flash",
           "log": str(profile.with_suffix(".log")), "host": "127.0.0.1", "port": port,
           "gpu": 0, "sampling": {"temperature": 0},
           "statistics_file": str(profile.with_suffix(".usage.sqlite")),
           "backend_identity": identity}
    if dirs:
        cfg["lib_dirs"] = dirs
    for flag, percent in (("--ram-target-percent", ram_target_percent), ("--vram-target-percent", vram_target_percent)):
        if percent:
            cfg["args"] += [flag, str(percent)]
    with profile.open("x", encoding="utf-8") as out:
        out.write(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    model = parser.add_mutually_exclusive_group(required=True)
    model.add_argument("--model-dir", type=Path)
    model.add_argument("--model", type=Path, help="Exact GGUF or first shard")
    parser.add_argument("--check-only", action="store_true", help="Inspect without creating a profile")
    parser.add_argument("--exe", type=Path, default=ROOT / "build-local/glm5next-candidate-cuda/bin" / ("strata-glm5next.exe" if os.name == "nt" else "strata-glm5next"))
    parser.add_argument("--profile", type=Path, default=ROOT / "strata-glm5next.json")
    parser.add_argument("--cuda-dir", type=Path)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--ram-target-percent", type=int, default=0, help="Global physical RAM target; 0 disables warmup")
    parser.add_argument("--vram-target-percent", type=int, default=0, help="Global VRAM target; 0 disables expert cache")
    parser.add_argument("--output", type=Path, help="Also save the JSON report to this path")
    parser.add_argument("--tensor-details", action="store_true", help="Include every tensor's shape, type and byte range")
    parser.add_argument("--loader-contract", action="store_true",
                        help="Also check full-model names/shapes against the audited Unsloth loader revision")
    args = parser.parse_args()
    try:
        first = [args.model] if args.model else sorted(args.model_dir.glob("*-00001-of-*.gguf"))
        if not first and args.model_dir:
            first = sorted(args.model_dir.glob("*.gguf"))
        if len(first) != 1:
            raise ValueError("The directory must contain exactly one model's first shard")
        report = inspect_model(first[0], tensor_details=args.tensor_details, loader_contract=args.loader_contract or not args.check_only)
        if not args.check_only:
            prepare_profile(report, args.exe, args.profile, context=args.context, batch=args.batch_size,
                            threads=args.threads, port=args.port, cuda_dir=args.cuda_dir,
                            ram_target_percent=args.ram_target_percent, vram_target_percent=args.vram_target_percent)
            report["profile"] = str(args.profile.resolve())
        output = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        if args.output:
            args.output.write_text(output, encoding="utf-8")
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"GLM setup failed: {exc}\n")
    print(output, end="")


if __name__ == "__main__":
    main()
