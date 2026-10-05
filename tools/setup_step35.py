"""Read-only admission and inventory for the Step-3.7-Flash text GGUF.

python tools/setup_step35.py --model-dir H:\\models\\Step-3.7-Flash\\UD-Q4_K_S --inspect --output inventory.json
No profile is installed and no weight payload is loaded or hashed.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import struct
import sys

if __package__:
    from .gguf_reader import BLOCK_GEOMETRY, GGUFFile
    from .step35_loader_contract import model_metadata, validate_loader_contract
else:
    from gguf_reader import BLOCK_GEOMETRY, GGUFFile
    from step35_loader_contract import model_metadata, validate_loader_contract


def header_hash(path, count):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while count:
            chunk = stream.read(min(count, 1024 * 1024))
            if not chunk:
                raise ValueError(f"Truncated header while hashing: {path}")
            digest.update(chunk)
            count -= len(chunk)
    return digest.hexdigest()


def _read(path):
    try:
        parsed = GGUFFile(path)
    except (ValueError, KeyError, struct.error, OverflowError) as exc:
        raise ValueError(f"Invalid GGUF header: {path.name}: {exc}") from exc
    align = parsed.metadata.get("general.alignment", 32)
    if type(align) is not int or align <= 0 or align & (align - 1) or align > 2**20:
        raise ValueError(f"Invalid GGUF alignment: {path.name}: {align}")
    return parsed


def first_shard(path):
    path = Path(path).resolve()
    if not path.is_dir():
        return path
    files = sorted(path.glob("*.gguf"))
    first = [p for p in files if re.fullmatch(r".+-00001-of-\d{5}\.gguf", p.name)
             or not re.search(r"-\d{5}-of-\d{5}\.gguf$", p.name)]
    if len(first) != 1:
        raise ValueError("Model directory must contain exactly one first shard or one unsplit GGUF")
    return first[0]


def inspect_model(path):
    first = first_shard(path)
    initial = _read(first)
    meta = initial.metadata
    h = model_metadata(meta)
    split_keys = ("split.no", "split.count", "split.tensors.count")
    split = any(k in meta for k in split_keys)
    count = meta.get("split.count", 1)
    if type(count) is not int or not 1 <= count <= 65535:
        raise ValueError("Invalid split.count")
    match = re.fullmatch(r"(.+)-00001-of-(\d{5})\.gguf", first.name)
    advertised = re.fullmatch(r".+-(\d{5})-of-(\d{5})\.gguf", first.name)
    if advertised and (int(advertised[1]) != 1 or int(advertised[2]) != count):
        raise ValueError("Shard filename and split metadata disagree; select the first shard")
    if count > 1 and (not match or int(match[2]) != count):
        raise ValueError("Select the first GGUF shard (00001-of-...)")
    if match and int(match[2]) != count:
        raise ValueError("Shard filename and split.count disagree")
    total = meta.get("split.tensors.count", len(initial.tensors))
    if type(total) is not int or total <= 0:
        raise ValueError("Invalid split.tensors.count")
    paths = [first] if count == 1 else [first.with_name(f"{match[1]}-{i:05d}-of-{count:05d}.gguf")
                                            for i in range(1, count + 1)]
    missing = [p.name for p in paths if not p.is_file()]
    if missing:
        raise ValueError("Missing GGUF shard(s): " + ", ".join(missing))
    tensors, shards, details = {}, [], []
    types = Counter()
    weight_bytes = routed_bytes = active_bytes = 0
    for index, path in enumerate(paths):
        parsed = initial if index == 0 else _read(path)
        if split:
            for key, expected in zip(split_keys, (index, count, total)):
                if type(parsed.metadata.get(key)) is not int or parsed.metadata[key] != expected:
                    raise ValueError(f"Wrong shard metadata: {path.name}: {key}")
        for key, value in parsed.metadata.items():
            if key.startswith(("step35.", "tokenizer.")) or key in ("general.architecture", "general.name"):
                if key not in meta or value != meta[key]:
                    raise ValueError(f"Conflicting metadata: {path.name}: {key}")
        size = path.stat().st_size
        ranges = []
        for tensor in parsed.tensors:
            name, shape = tensor.name, tensor.shape
            if not name or name in tensors:
                raise ValueError(f"Empty or duplicate tensor: {name}")
            if any(d <= 0 for d in shape):
                raise ValueError(f"Invalid tensor shape: {name}: {shape}")
            geometry = BLOCK_GEOMETRY.get(tensor.type_name)
            if geometry is None:
                raise ValueError(f"Unknown tensor encoding: {name}: {tensor.type_name}")
            if shape[0] % geometry[0]:
                raise ValueError(f"Invalid quantized row shape: {name}: {shape}")
            nbytes = tensor.expected_bytes()
            start = parsed.data_start + tensor.offset
            end = start + nbytes
            if end > 2**63 - 1:
                raise ValueError(f"Tensor range exceeds signed 64-bit offsets: {name}")
            if tensor.offset % parsed.alignment:
                raise ValueError(f"Unaligned tensor offset: {name}")
            if end > size:
                raise ValueError(f"Incomplete shard: {path.name}: {name}: {end} > {size}")
            ranges.append((start, end, name))
            tensors[name] = tensor
            types[tensor.type_name] += 1
            weight_bytes += nbytes
            detail = {"name": name, "shape": shape, "type": tensor.type_name, "shard": index,
                      "file_offset": start, "bytes": nbytes, "row_bytes": shape[0] // geometry[0] * geometry[1]}
            if re.fullmatch(r"blk\.\d+\.ffn_(gate|up|down)_exps\.weight", name):
                if len(shape) != 3 or shape[2] != h["expert_count"]:
                    raise ValueError(f"Invalid expert tensor shape: {name}")
                expert_bytes = nbytes // shape[2]
                detail["expert_bytes"] = expert_bytes
                routed_bytes += nbytes
                active_bytes += expert_bytes * h["expert_used_count"]
            details.append(detail)
        ranges.sort()
        for a, b in zip(ranges, ranges[1:]):
            if b[0] < a[1]:
                raise ValueError(f"Overlapping tensors: {path.name}: {a[2]}, {b[2]}")
        shards.append({"name": path.name, "split_no": index, "tensors": len(parsed.tensors),
                       "file_bytes": size, "header_end": parsed.header_end, "data_start": parsed.data_start,
                       "alignment": parsed.alignment, "metadata_only": not parsed.tensors,
                       "required_end": max([parsed.header_end] + [r[1] for r in ranges]),
                       "header_sha256": header_hash(path, parsed.header_end)})
    if len(tensors) != total:
        raise ValueError("The number of tensors does not match split.tensors.count")
    contract = validate_loader_contract(meta, tensors)
    fingerprint_input = [{k: s[k] for k in ("split_no", "file_bytes", "header_sha256")} for s in shards]
    structural_hash = hashlib.sha256(json.dumps(fingerprint_input, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    token_keys = ("model", "pre", "bos_token_id", "eos_token_id", "padding_token_id", "add_bos_token", "add_sep_token")
    return {"schema_version": 1, "architecture": "step35", "model_name": meta.get("general.name"),
            "first_shard": str(first), "scope": "headers, ranges and static trunk contract; no weight payload hashes, compiled loader, GPU or inference checks",
            "structural_fingerprint_sha256": structural_hash,
            "fingerprint_scope": "canonical JSON of ordered split_no/file_bytes/header_sha256; changes to weight payload bytes are not detected",
            "shards": count, "tensors": len(tensors), "file_bytes": sum(s["file_bytes"] for s in shards),
            "weight_bytes": weight_bytes, "routed_bytes": routed_bytes, "non_routed_bytes": weight_bytes - routed_bytes,
            "estimated_no_hit_decode_expert_bytes": active_bytes,
            "estimate_scope": "each active expert triple once per main MoE layer; not measured traffic or speed",
            "tensor_type_counts": dict(sorted(types.items())), "shard_details": shards,
            "gguf_architecture_metadata": {k: v for k, v in meta.items() if k.startswith("step35.")},
            "tokenizer": {**{k: meta.get("tokenizer.ggml." + k) for k in token_keys},
                          "vocab_size": h["vocab_size"], "merges": len(meta["tokenizer.ggml.merges"]),
                          "template_sha256": hashlib.sha256(meta["tokenizer.chat_template"].encode()).hexdigest()},
            "loader_contract": contract, "tensor_details": details}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model-dir", type=Path)
    source.add_argument("--gguf", type=Path)
    parser.add_argument("--inspect", action="store_true", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        report = inspect_model(args.gguf or args.model_dir)
        if args.output:
            # Protect every shard, including hardlink/symlink aliases.
            for shard in report["shard_details"]:
                path = Path(report["first_shard"]).with_name(shard["name"])
                if args.output.resolve() == path.resolve() or (args.output.exists() and args.output.samefile(path)):
                    raise ValueError("--output must not overwrite a GGUF input")
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f'PASS: {report["shards"]} shards, {report["tensors"]} tensors, {report["file_bytes"] / 2**30:.6f} GiB')
        print(f'Static trunk contract passed; GPU/inference not tested. Fingerprint: {report["structural_fingerprint_sha256"]}')
        return 0
    except (ValueError, OSError) as exc:
        print(f"Step inspection failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
