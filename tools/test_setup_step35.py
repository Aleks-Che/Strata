"""Small independent Step GGUF fixtures; no GPU, network or real model required."""
from copy import deepcopy
import math
from pathlib import Path
import struct
import tempfile
import unittest

if __package__:
    from .gguf_reader import GGUFFile
    from .setup_step35 import inspect_model, main
else:
    from gguf_reader import GGUFFile
    from setup_step35 import inspect_model, main


def string(value):
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def value(item):
    if isinstance(item, str):
        return struct.pack("<I", 8) + string(item)
    if type(item) is bool:
        return struct.pack("<I?", 7, item)
    if type(item) is float:
        return struct.pack("<If", 6, item)
    if isinstance(item, list):
        encoded = [value(v) for v in item]
        kind = encoded[0][:4] if encoded else struct.pack("<I", 4)
        assert all(v[:4] == kind for v in encoded)
        return struct.pack("<I", 9) + kind + struct.pack("<Q", len(item)) + b"".join(v[4:] for v in encoded)
    return struct.pack("<IQ", 10, item)


def write_shard(path, meta, tensors, pad_empty=False):
    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(meta))
    for key, item in meta.items():
        header += string(key) + value(item)
    payload = b""
    for tensor in tensors:
        shape = tensor["shape"]
        kind = tensor.get("type", 0)
        header += string(tensor["name"]) + struct.pack("<I", len(shape))
        header += struct.pack(f"<{len(shape)}Q", *shape) + struct.pack("<IQ", kind, tensor.get("offset", len(payload)))
        # Independent geometry; intentionally not imported from the inspector.
        block, size = {0: (1, 4), 8: (32, 34), 12: (256, 144), 14: (256, 210)}.get(kind, (1, 4))
        payload += bytes(math.prod(shape) // block * size)
        payload += bytes(-len(payload) % 32)
    if tensors or pad_empty:
        header += bytes(-len(header) % 32)
    path.write_bytes(header + payload)


def fixture():
    meta = {"general.architecture": "step35", "general.name": "Step fixture",
            "step35.block_count": 3, "step35.context_length": 2048, "step35.embedding_length": 256,
            "step35.feed_forward_length": 512, "step35.leading_dense_block_count": 1,
            "step35.expert_feed_forward_length": 256, "step35.expert_shared_feed_forward_length": 256,
            "step35.expert_count": 4, "step35.expert_used_count": 2, "step35.expert_gating_func": 2,
            "step35.expert_weights_scale": 3.0, "step35.expert_weights_norm": True,
            "step35.attention.head_count": [2, 3, 2], "step35.attention.head_count_kv": [1, 1, 1],
            "step35.attention.key_length": 8, "step35.attention.value_length": 8,
            "step35.attention.layer_norm_rms_epsilon": 1e-5,
            "step35.attention.sliding_window": 512, "step35.attention.sliding_window_pattern": [False, True, False],
            "step35.rope.freq_base": 5000000.0, "step35.rope.freq_base_swa": 10000.0,
            "step35.swiglu_clamp_exp": [0.0, 0.0, 7.0], "step35.swiglu_clamp_shexp": [0.0, 0.0, 16.0],
            "tokenizer.ggml.model": "gpt2", "tokenizer.ggml.pre": "deepseek-v3",
            "tokenizer.ggml.tokens": [f"token{i}" for i in range(16)], "tokenizer.ggml.token_type": [1] * 16,
            "tokenizer.ggml.merges": ["token0 token1"], "tokenizer.chat_template": "{{ messages }}",
            "tokenizer.ggml.bos_token_id": 0, "tokenizer.ggml.eos_token_id": 1, "tokenizer.ggml.padding_token_id": 2}
    tensors = []

    def add(name, shape, quant=0):
        tensors.append({"name": name, "shape": shape, "type": quant})

    add("token_embd.weight", [256, 16], 8)
    add("output.weight", [256, 16], 14)
    add("output_norm.weight", [256])
    add("rope_freqs.weight", [4])
    for layer, heads in enumerate([2, 3, 2]):
        prefix = f"blk.{layer}."
        for suffix in ("attn_norm", "ffn_norm"):
            add(prefix + suffix + ".weight", [256])
        for suffix in ("attn_q_norm", "attn_k_norm"):
            add(prefix + suffix + ".weight", [8])
        for suffix, shape in (("attn_q", [256, heads * 8]), ("attn_k", [256, 8]),
                              ("attn_v", [256, 8]), ("attn_output", [heads * 8, 256]),
                              ("attn_gate", [256, heads])):
            # F32 output permits the deliberately small head dimension.
            add(prefix + suffix + ".weight", shape, 0 if suffix == "attn_output" else 8)
        if layer == 0:
            for suffix, shape in (("gate", [256, 512]), ("up", [256, 512]), ("down", [512, 256])):
                add(prefix + "ffn_" + suffix + ".weight", shape, 8)
        else:
            add(prefix + "ffn_gate_inp.weight", [256, 4])
            add(prefix + "exp_probs_b.bias", [4])
            for suffix in ("gate", "up", "down"):
                add(prefix + "ffn_" + suffix + "_exps.weight", [256, 256, 4], 12)
                add(prefix + "ffn_" + suffix + "_shexp.weight", [256, 256], 8)
    return meta, tensors


class StepAdmissionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.directory = Path(temp.name)
        self.paths = [self.directory / f"step-{i:05d}-of-00004.gguf" for i in range(1, 5)]
        self.meta, tensors = fixture()
        self.parts = [[], tensors[:10], tensors[10:27], tensors[27:]]
        self.write()

    def write(self, overrides=None, pad_empty=False):
        for index, path in enumerate(self.paths):
            meta = dict(self.meta) if index == 0 else {}
            meta.update({"split.no": index, "split.count": 4, "split.tensors.count": sum(map(len, self.parts))})
            meta.update((overrides or {}).get(index, {}))
            write_shard(path, meta, self.parts[index], pad_empty)

    def inspect(self):
        return inspect_model(self.paths[0])

    def test_valid_metadata_only_first(self):
        for pad in (False, True):
            with self.subTest(padded=pad):
                self.write(pad_empty=pad)
                report = self.inspect()
                self.assertEqual(report["tensors"], 50)
                self.assertEqual(report["loader_contract"]["metadata"]["rope_dimensions"], [4, 8, 4])
                self.assertEqual(report["loader_contract"]["mtp_blocks"], 0)
                self.assertEqual(len(report["loader_contract"]["tensor_mapping"]), 50)
                self.assertTrue(report["shard_details"][0]["metadata_only"])
                self.assertEqual(report["estimated_no_hit_decode_expert_bytes"], 2 * 2 * 3 * 256 * 256 // 256 * 144)
                self.assertEqual(report, inspect_model(self.directory))

    def test_single_file(self):
        path = self.directory / "single.gguf"
        write_shard(path, self.meta, sum(self.parts, []))
        self.assertEqual(inspect_model(path)["shards"], 1)

    def test_single_file_cannot_claim_to_be_a_later_shard(self):
        write_shard(self.paths[1], self.meta, sum(self.parts, []))
        with self.assertRaisesRegex(ValueError, "first shard"):
            inspect_model(self.paths[1])

    def test_trailing_singleton_dimensions(self):
        self.parts[1][2]["shape"] = [256, 1, 1, 1]
        self.write()
        self.assertEqual(self.inspect()["tensors"], 50)

    def test_scalar_clamps_are_broadcast(self):
        self.write({0: {"step35.swiglu_clamp_exp": 0, "step35.swiglu_clamp_shexp": 7.0}})
        metadata = self.inspect()["loader_contract"]["metadata"]
        self.assertEqual(metadata["swiglu_clamp_exp"], [0, 0, 0])
        self.assertEqual(metadata["swiglu_clamp_shexp"], [7.0, 7.0, 7.0])

    def test_ambiguous_directory(self):
        (self.directory / "other.gguf").write_bytes(b"GGUF")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            inspect_model(self.directory)

    def test_missing_shard(self):
        self.paths[2].unlink()
        with self.assertRaisesRegex(ValueError, "Missing GGUF"):
            self.inspect()

    def test_reordered_shards(self):
        a, b = self.paths[1].read_bytes(), self.paths[2].read_bytes()
        self.paths[1].write_bytes(b)
        self.paths[2].write_bytes(a)
        with self.assertRaisesRegex(ValueError, "Wrong shard"):
            self.inspect()

    def test_wrong_split_or_conflicting_metadata(self):
        for index, patch in ((1, {"split.no": 0}), (2, {"split.count": 3}), (3, {"split.tensors.count": 999}),
                             (0, {"split.count": 3}), (1, {"step35.block_count": 4}),
                             (1, {"step35.nextn_predict_layers": 1}), (1, {"general.architecture": "glm5next"})):
            with self.subTest(index=index, patch=patch), self.assertRaises(ValueError):
                self.write({index: patch})
                self.inspect()

    def test_invalid_metadata(self):
        for key, bad in (
            ("general.architecture", "glm5next"), ("tokenizer.ggml.pre", "qwen35"),
            ("step35.block_count", 0), ("step35.block_count", 513),
            ("step35.nextn_predict_layers", 3), ("step35.expert_used_count", 5),
            ("step35.expert_count", True), ("step35.attention.head_count", [2, 3]),
            ("step35.attention.head_count_kv", [1, 2, 1]),
            ("step35.attention.sliding_window_pattern", [0, 1, 0]),
            ("step35.swiglu_clamp_exp", [0.0, float("nan"), 7.0]),
            ("step35.swiglu_clamp_shexp", [0.0, -1.0, 16.0]),
            ("step35.rope.dimension_count", 6), ("step35.rope.scaling.type", "longrope"),
            ("step35.expert_weights_scale", float("inf")), ("step35.expert_weights_norm", 1),
            ("tokenizer.ggml.tokens", ["same"] * 16), ("tokenizer.ggml.token_type", [1]),
            ("tokenizer.ggml.eos_token_id", 16), ("tokenizer.chat_template", "")):
            with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                self.write({0: {key: bad}})
                self.inspect()

    def test_tensor_mutations(self):
        original = deepcopy(self.parts)
        for patch, error in (({"name": "unknown.weight"}, "Loader tensor names"),
                             ({"name": "token_embd.weight"}, "duplicate tensor"),
                             ({"shape": [0]}, "Invalid tensor shape"),
                             ({"shape": [257, 16], "type": 12}, "row shape"),
                             ({"type": 999}, "Unknown tensor encoding"),
                             ({"offset": 1}, "Unaligned"), ({"offset": 0}, "Overlapping"),
                             ({"offset": 2**63}, "64-bit offsets"),
                             ({"shape": [256, 15]}, "Loader tensor shape")):
            with self.subTest(patch=patch):
                self.parts = deepcopy(original)
                self.parts[1][1].update(patch)
                self.write()
                with self.assertRaisesRegex(ValueError, error):
                    self.inspect()

    def test_missing_required_profile_tensor(self):
        self.parts[1] = [t for t in self.parts[1] if t["name"] != "rope_freqs.weight"]
        self.write()
        with self.assertRaisesRegex(ValueError, "Loader tensor names"):
            self.inspect()

    def test_truncated_weight(self):
        with self.paths[3].open("r+b") as stream:
            stream.truncate(2000)
        with self.assertRaisesRegex(ValueError, "Incomplete shard"):
            self.inspect()

    def test_malformed_headers(self):
        for content in (b"", b"BAD!", b"GGUF" + struct.pack("<IQQ", 2, 0, 0),
                        b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + string("array") + struct.pack("<IIQ", 9, 10, 2**63),
                        b"GGUF" + struct.pack("<IQQ", 3, 0, 2) + string("a") + value(1) + string("a") + value(2)):
            with self.subTest(content=content):
                self.paths[0].write_bytes(content)
                with self.assertRaisesRegex(ValueError, "Invalid GGUF header"):
                    self.inspect()

    def test_invalid_alignment(self):
        for bad in (0, 3, True, 2**22):
            self.write({1: {"general.alignment": bad}})
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "alignment"):
                self.inspect()

    def test_fingerprint_scope(self):
        a = self.inspect()["structural_fingerprint_sha256"]
        offset = GGUFFile(self.paths[1]).data_start
        with self.paths[1].open("r+b") as stream:
            stream.seek(offset)
            stream.write(b"\x01")
        self.assertEqual(a, self.inspect()["structural_fingerprint_sha256"])
        with self.paths[1].open("ab") as stream:
            stream.write(b"\x00")
        self.assertNotEqual(a, self.inspect()["structural_fingerprint_sha256"])

    def test_output_cannot_overwrite_any_shard(self):
        for path in self.paths:
            before = path.read_bytes()
            self.assertEqual(main(["--gguf", str(self.paths[0]), "--inspect", "--output", str(path)]), 1)
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
