"""GLM GGUF admission fixtures; stdlib only, no model downloads or GPU.

python tools/test_setup_glm5next.py
"""
from copy import deepcopy
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

if __package__:
    from .gguf_reader import GGUFFile
    from .setup_glm5next import inspect_model
else:
    from gguf_reader import GGUFFile
    from setup_glm5next import inspect_model


def string(value):
    data = value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def value(data):
    if isinstance(data, str):
        return struct.pack("<I", 8) + string(data)
    if isinstance(data, list):
        element = 8 if data and isinstance(data[0], str) else 4
        return struct.pack("<IIQ", 9, element, len(data)) + b"".join(value(v)[4:] for v in data)
    return struct.pack("<II", 4, data)


def write_shard(path, meta, tensors):
    """Independent fixture encoder with optional invalid offsets/types/shapes."""
    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(meta))
    for key, item in meta.items():
        header += string(key) + value(item)
    payload = b""
    for tensor in tensors:
        name, shape = tensor["name"], tensor["shape"]
        type_id = tensor.get("type", 0)
        offset = tensor.get("offset", len(payload))
        header += string(name) + struct.pack("<I", len(shape))
        header += struct.pack(f"<{len(shape)}Q", *shape) + struct.pack("<IQ", type_id, offset)
        # Independent GGML geometry for the five expert formats in UD-Q3_K_XL.
        block, size = {0: (1, 4), 18: (256, 98), 23: (256, 136),
                       14: (256, 210), 11: (256, 110), 12: (256, 144)}.get(type_id, (1, 4))
        nbytes = math.prod(shape) // block * size
        payload += bytes(nbytes)
        payload += bytes(-len(payload) % 32)
    if tensors:
        header += bytes(-len(header) % 32)
    path.write_bytes(header + payload)  # Empty shards deliberately have no padding.


def model_fixture():
    width, ff, vocab, experts = 256, 256, 8, 2
    meta = {"general.architecture": "glm5next", "glm5next.block_count": 3,
            "glm5next.nextn_predict_layers": 1, "glm5next.embedding_length": width,
            "glm5next.feed_forward_length": ff, "glm5next.expert_feed_forward_length": ff,
            "glm5next.leading_dense_block_count": 1, "glm5next.expert_count": experts,
            "glm5next.expert_used_count": 1, "glm5next.context_length": 128,
            "glm5next.vocab_size": vocab, "glm5next.attention.head_count_kv": [0, 1, 1],
            "tokenizer.ggml.model": "gpt2", "tokenizer.ggml.pre": "glm4",
            "tokenizer.ggml.tokens": [f"token{i}" for i in range(vocab)],
            "tokenizer.ggml.merges": ["a b"], "tokenizer.chat_template": "{{ messages }}",
            "tokenizer.ggml.eos_token_id": 0, "tokenizer.ggml.eot_token_id": 1,
            "tokenizer.ggml.eom_token_id": 2}
    tensors = []

    def add(name, shape, type_id=0):
        tensors.append({"name": name, "shape": shape, "type": type_id})

    add("token_embd.weight", [width, vocab])
    add("output.weight", [width, vocab])
    add("output_norm.weight", [width])
    for layer in range(3):
        prefix = f"blk.{layer}."
        for norm in ("attn_norm", "ffn_norm"):
            add(prefix + norm + ".weight", [width])
        if layer == 0:
            for matrix in ("gate", "up", "down"):
                add(prefix + f"ffn_{matrix}.weight", [width, ff])
        else:
            add(prefix + "ffn_gate_inp.weight", [width, experts])
            for matrix, quant in zip(("gate", "up", "down"), (18, 23, 14) if layer == 1 else (11, 11, 12)):
                add(prefix + f"ffn_{matrix}_exps.weight", [width, ff, experts], quant)
        if layer == 2:
            add(prefix + "nextn.eh_proj.weight", [2 * width, width])
            for norm in ("enorm", "hnorm", "shared_head_norm"):
                add(prefix + f"nextn.{norm}.weight", [width])
    return meta, tensors


class GLMAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.paths = [self.directory / f"glm-{i:05d}-of-00004.gguf" for i in range(1, 5)]
        self.meta, tensors = model_fixture()
        self.parts = [[], tensors[:8], tensors[8:16], tensors[16:]]
        self.write()

    def write(self):
        total = sum(map(len, self.parts))
        for i, path in enumerate(self.paths):
            meta = dict(self.meta) if i == 0 else {}
            meta.update({"split.no": i, "split.count": 4, "split.tensors.count": total})
            write_shard(path, meta, self.parts[i])

    def test_metadata_only_and_all_expert_quants(self):
        report = inspect_model(self.paths[0], tensor_details=True)
        self.assertEqual((report["shards"], report["tensors"], report["main_blocks"], report["mtp_blocks"]), (4, 24, 2, 1))
        first = report["shard_details"][0]
        self.assertEqual(first["file_bytes"], first["header_end"])
        self.assertLess(first["file_bytes"], first["data_start"])
        self.assertEqual(report["tensor_type_counts"], {"F32": 18, "IQ3_XXS": 1, "IQ4_XS": 1, "Q6_K": 1, "Q3_K": 2, "Q4_K": 1})
        self.assertEqual(report["groups"]["main"]["expert_bytes"], 2 * 256 * (98 + 136 + 210))
        self.assertEqual(report["groups"]["mtp"]["expert_bytes"], 2 * 256 * (110 + 110 + 144))
        self.assertEqual(len(report["tensor_details"]), 24)
        self.assertEqual(sum(t["bytes"] for t in report["tensor_details"]), report["weight_bytes"])

    def test_missing_and_truncated_payload(self):
        self.paths[2].unlink()
        with self.assertRaisesRegex(ValueError, "Missing GGUF shard.*00003"):
            inspect_model(self.paths[0])
        self.write()
        last = self.paths[3]
        last.write_bytes(last.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, "Incomplete shard"):
            inspect_model(self.paths[0])

    def test_duplicate_tensor_across_shards(self):
        self.parts[3].append(deepcopy(self.parts[1][0]))
        self.write()
        with self.assertRaisesRegex(ValueError, "duplicate tensor"):
            inspect_model(self.paths[0])

    def test_split_metadata(self):
        for key, bad in (("split.no", 0), ("split.count", 3), ("split.tensors.count", 999)):
            with self.subTest(key=key):
                meta = {"split.no": 1, "split.count": 4, "split.tensors.count": 24, key: bad}
                write_shard(self.paths[1], meta, self.parts[1])
                with self.assertRaisesRegex(ValueError, "Wrong shard metadata"):
                    inspect_model(self.paths[0])

    def test_total_tensor_count(self):
        for i, path in enumerate(self.paths):
            meta = dict(self.meta) if i == 0 else {}
            meta.update({"split.no": i, "split.count": 4, "split.tensors.count": 25})
            write_shard(path, meta, self.parts[i])
        with self.assertRaisesRegex(ValueError, "number of tensors"):
            inspect_model(self.paths[0])

    def test_required_metadata(self):
        for key, bad in (("general.architecture", "glm5-next"), ("glm5next.embedding_length", 0),
                         ("glm5next.nextn_predict_layers", 3), ("glm5next.expert_used_count", 3),
                         ("glm5next.attention.head_count_kv", [1]), ("tokenizer.ggml.tokens", ["only"]),
                         ("tokenizer.ggml.pre", "qwen35"), ("tokenizer.ggml.eot_token_id", 8),
                         ("tokenizer.chat_template", ""), ("general.alignment", 3)):
            with self.subTest(key=key):
                saved = dict(self.meta)
                self.meta[key] = bad
                self.write()
                with self.assertRaises(ValueError):
                    inspect_model(self.paths[0])
                self.meta = saved
        del self.meta["glm5next.nextn_predict_layers"]
        self.write()
        with self.assertRaisesRegex(ValueError, "nextn_predict_layers"):
            inspect_model(self.paths[0])

    def test_core_shapes_nextn_and_block_coverage(self):
        original = deepcopy(self.parts)
        for change in ("shape", "nextn_missing", "block_missing", "outside", "nextn_in_main"):
            with self.subTest(change=change):
                self.parts = deepcopy(original)
                if change == "shape":
                    self.parts[1][0]["shape"] = [256, 7]
                elif change == "nextn_missing":
                    self.parts[3] = [t for t in self.parts[3] if "nextn.hnorm" not in t["name"]]
                elif change == "block_missing":
                    self.parts = [[t for t in part if not t["name"].startswith("blk.1.")] for part in self.parts]
                elif change == "outside":
                    self.parts[3][-1]["name"] = "blk.3.nextn.shared_head_norm.weight"
                else:
                    self.parts[3][-1]["name"] = "blk.0.nextn.shared_head_norm.weight"
                self.write()
                with self.assertRaises(ValueError):
                    inspect_model(self.paths[0])

    def test_encoding_geometry_and_ranges(self):
        original = deepcopy(self.parts)
        for patch, error in (({"type": 999}, "Unknown tensor encoding"),
                             ({"shape": [0]}, "Invalid tensor shape"),
                             ({"shape": [1, 1, 1, 1, 1]}, "invalid tensor rank"),
                             ({"shape": [128, 512], "type": 18}, "quantized row shape"),
                             ({"offset": 1}, "Unaligned tensor offset"),
                             ({"offset": 0}, "Overlapping tensors"),
                             ({"offset": 2**40}, "Incomplete shard")):
            with self.subTest(patch=patch):
                self.parts = deepcopy(original)
                self.parts[1][1].update(patch)
                self.write()
                with self.assertRaisesRegex(ValueError, error):
                    inspect_model(self.paths[0])

    def test_conflicting_architecture_in_later_shard(self):
        meta = {"split.no": 1, "split.count": 4, "split.tensors.count": 24,
                "general.architecture": "deepseek4"}
        write_shard(self.paths[1], meta, self.parts[1])
        with self.assertRaisesRegex(ValueError, "Conflicting metadata"):
            inspect_model(self.paths[0])

    def test_unsplit_model(self):
        path = self.directory / "single.gguf"
        write_shard(path, self.meta, sum(self.parts, []))
        self.assertEqual(inspect_model(path)["shards"], 1)

    def test_cli_json_and_failure(self):
        output = self.directory / "report.json"
        command = [sys.executable, str(Path(__file__).with_name("setup_glm5next.py")),
                   "--model-dir", str(self.directory), "--check-only", "--output", str(output)]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), json.loads(output.read_text(encoding="utf-8")))
        self.paths[2].unlink()
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("Missing GGUF shard", result.stderr)
        self.assertNotIn("Traceback", result.stderr)


class HeaderReaderTests(unittest.TestCase):
    def test_truncated_header_and_duplicate_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.gguf"
            header = b"GGUF" + struct.pack("<IQQ", 3, 0, 1)
            for payload in (header + string("key") + struct.pack("<IQ", 8, 5) + b"abc",
                            header + struct.pack("<Q", 2**63)):
                path.write_bytes(payload)
                with self.assertRaisesRegex(ValueError, "truncated GGUF string"):
                    GGUFFile(path)
            item = string("key") + value(1)
            path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 2) + item + item)
            with self.assertRaisesRegex(ValueError, "duplicate metadata key"):
                GGUFFile(path)


if __name__ == "__main__":
    unittest.main()
