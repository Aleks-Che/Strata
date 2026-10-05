"""Static contract for the Step-3.7 text trunk; not proof of GPU execution.

This deliberately admits the separate-QKV, gated, Q/K-normalized trunk layout
present in UD-Q4_K_S. Other Step layouts and MTP sidecars need separate audits.
"""
from collections import Counter
import math

LOADER_SHA = "86ebfef2c6a0f3359a2a07d2c215d61b0fa885c9"
LOADER_FILE_SHA256 = "4c42074b6f859b5734572eb0b1f1f07c8b783e20bfeb3a357871c77a8255c8b3"
SOURCE_BASE = f"https://github.com/unslothai/llama.cpp/blob/{LOADER_SHA}/src/"


def model_metadata(meta):
    if meta.get("general.architecture") != "step35":
        raise ValueError("Expected general.architecture=step35")
    if (meta.get("tokenizer.ggml.model"), meta.get("tokenizer.ggml.pre")) != ("gpt2", "deepseek-v3"):
        raise ValueError("Expected gpt2/deepseek-v3 tokenizer")

    def integer(key, minimum=1, default=None):
        value = meta.get("step35." + key, default)
        if type(value) is not int or not minimum <= value <= 2**31 - 1:
            raise ValueError(f"Invalid step35.{key}: expected integer >= {minimum}")
        return value

    blocks = integer("block_count")
    if blocks > 512:
        raise ValueError("step35.block_count exceeds the audited loader limit (512)")
    # Do not infer NextN count from config.json or subtract it from a trunk GGUF.
    if integer("nextn_predict_layers", 0, 0):
        raise ValueError("NextN metadata needs a separate MTP audit; only the main text trunk is admitted")
    result = {key: integer(key) for key in (
        "embedding_length", "context_length", "feed_forward_length",
        "expert_feed_forward_length", "expert_shared_feed_forward_length",
        "expert_count", "expert_used_count", "attention.sliding_window",
        "attention.key_length", "attention.value_length")}
    result["block_count"] = blocks
    dense = integer("leading_dense_block_count", 0)
    if dense >= blocks or result["expert_used_count"] > result["expert_count"]:
        raise ValueError("Invalid dense block count or expert top-k")
    result["leading_dense_block_count"] = dense
    if integer("moe_every_n_layers", 1, 1) != 1:
        raise ValueError("Only consecutive MoE layers after the leading dense blocks are audited")
    if integer("expert_gating_func", 0, 2) not in (0, 2):
        raise ValueError("Step trunk requires sigmoid expert gating (2, or default 0)")
    if meta.get("step35.expert_weights_norm") is not True:
        raise ValueError("Step trunk requires expert_weights_norm=true")
    result["expert_gating_func"] = "sigmoid"
    result["expert_weights_norm"] = True

    def numeric(key, positive=True):
        value = meta.get("step35." + key)
        if type(value) not in (int, float) or not math.isfinite(value) or (positive and value <= 0):
            raise ValueError(f"Invalid step35.{key}: expected finite positive number")
        result[key] = value

    for key in ("attention.layer_norm_rms_epsilon", "rope.freq_base", "rope.freq_base_swa", "expert_weights_scale"):
        numeric(key)

    def array(key, kind, minimum=0, scalar=True):
        values = meta.get("step35." + key)
        if scalar and (type(values) is kind or (kind is float and type(values) is int)):
            values = [values] * blocks
        if not isinstance(values, list) or len(values) != blocks:
            raise ValueError(f"step35.{key}: expected {blocks} per-layer values")
        for item in values:
            valid_type = type(item) in (int, float) if kind is float else type(item) is kind
            if not valid_type or not math.isfinite(item) or item < minimum:
                raise ValueError(f"step35.{key}: invalid per-layer value")
        result[key] = values
        return values

    heads = array("attention.head_count", int, 1)
    kv = array("attention.head_count_kv", int, 1)
    swa = array("attention.sliding_window_pattern", bool, scalar=False)
    for n, k in zip(heads, kv):
        if n % k:
            raise ValueError("Attention head_count must be divisible by head_count_kv")
    for key in ("swiglu_clamp_exp", "swiglu_clamp_shexp"):
        array(key, float)
    key_dim, value_dim = result["attention.key_length"], result["attention.value_length"]
    if key_dim != value_dim:
        raise ValueError("Different K/V head dimensions are outside this Step contract")
    rot = integer("rope.dimension_count", 1, key_dim)
    rot_swa = integer("rope.dimension_count_swa", 1, rot)
    if rot % 4 or rot_swa % 2 or rot > key_dim or rot_swa > key_dim:
        raise ValueError("Invalid partial rotary dimensions")
    if meta.get("step35.rope.scaling.type", "none") == "longrope":
        raise ValueError("LongRoPE tensors are outside this Step contract")
    result["rope_dimensions"] = [rot_swa if s else rot // 2 for s in swa]
    result["full_attention_blocks"] = [i for i, s in enumerate(swa) if not s]
    result["swa_blocks"] = [i for i, s in enumerate(swa) if s]
    tokens = meta.get("tokenizer.ggml.tokens")
    types = meta.get("tokenizer.ggml.token_type")
    merges = meta.get("tokenizer.ggml.merges")
    if not isinstance(tokens, list) or not tokens or any(not isinstance(t, str) for t in tokens):
        raise ValueError("Invalid tokenizer vocabulary")
    if len(set(tokens)) != len(tokens):
        raise ValueError("Duplicate vocabulary entries")
    if not isinstance(types, list) or len(types) != len(tokens) or any(type(t) is not int or t not in range(1, 7) for t in types):
        raise ValueError("Invalid tokenizer token_type array")
    if not isinstance(merges, list) or any(not isinstance(m, str) or len(m.split(" ")) != 2 for m in merges):
        raise ValueError("Invalid tokenizer merges")
    if not isinstance(meta.get("tokenizer.chat_template"), str) or not meta["tokenizer.chat_template"]:
        raise ValueError("Missing embedded chat template")
    result["vocab_size"] = len(tokens)
    if meta.get("step35.vocab_size", len(tokens)) != len(tokens):
        raise ValueError("step35.vocab_size differs from the tokenizer")
    for key in ("bos_token_id", "eos_token_id", "padding_token_id"):
        value = meta.get("tokenizer.ggml." + key)
        if type(value) is not int or not 0 <= value < len(tokens):
            raise ValueError(f"Invalid tokenizer.ggml.{key}")
    return result


def validate_loader_contract(meta, tensors):
    h = model_metadata(meta)
    width, vocab = h["embedding_length"], h["vocab_size"]
    ff, expert_ff, shared_ff = (h[k] for k in ("feed_forward_length", "expert_feed_forward_length", "expert_shared_feed_forward_length"))
    experts = h["expert_count"]
    expected = {}

    def add(name, shape, family, layer=None):
        expected[name] = (shape, family, layer)

    add("token_embd.weight", [width, vocab], "embedding")
    add("output.weight", [width, vocab], "output")
    add("output_norm.weight", [width], "norm")
    add("rope_freqs.weight", [max(h["rope_dimensions"]) // 2], "rope")
    for layer in range(h["block_count"]):
        prefix = f"blk.{layer}."

        def weight(name, shape, family, suffix="weight"):
            add(prefix + name + "." + suffix, shape, family, layer)

        heads, kv = h["attention.head_count"][layer], h["attention.head_count_kv"][layer]
        kdim, vdim = h["attention.key_length"], h["attention.value_length"]
        for name in ("attn_norm", "ffn_norm"):
            weight(name, [width], "norm")
        for name in ("attn_q_norm", "attn_k_norm"):
            weight(name, [kdim], "qk_norm")
        for name, shape in (("attn_q", [width, kdim * heads]), ("attn_k", [width, kdim * kv]),
                            ("attn_v", [width, vdim * kv]), ("attn_output", [vdim * heads, width])):
            weight(name, shape, "attention")
        weight("attn_gate", [width, heads], "head_gate")
        if layer < h["leading_dense_block_count"]:
            for name, shape in (("gate", [width, ff]), ("up", [width, ff]), ("down", [ff, width])):
                weight("ffn_" + name, shape, "dense_ffn")
        else:
            weight("ffn_gate_inp", [width, experts], "router")
            weight("exp_probs_b", [experts], "router", "bias")
            for name, a, b in (("gate", width, expert_ff), ("up", width, expert_ff), ("down", expert_ff, width)):
                weight(f"ffn_{name}_exps", [a, b, experts], "routed_experts")
            for name, a, b in (("gate", width, shared_ff), ("up", width, shared_ff), ("down", shared_ff, width)):
                weight(f"ffn_{name}_shexp", [a, b], "shared_experts")
    missing, extra = sorted(expected.keys() - tensors.keys()), sorted(tensors.keys() - expected.keys())
    if missing or extra:
        raise ValueError(f"Loader tensor names: missing={missing}; unexpected={extra}")
    mapping, families = [], {}
    for name, tensor in sorted(tensors.items()):
        shape, family, layer = expected[name]
        if tensor.shape + [1] * (4 - len(tensor.shape)) != shape + [1] * (4 - len(shape)):
            raise ValueError(f"Loader tensor shape: {name}: expected {shape}, got {tensor.shape}")
        counts = families.setdefault(family, {"tensors": 0, "types": Counter()})
        counts["tensors"] += 1
        counts["types"][tensor.type_name] += 1
        mapping.append({"name": name, "expected_shape": shape, "family": family, "layer": layer})
    return {"source_revision": LOADER_SHA, "step35_cpp_sha256": LOADER_FILE_SHA256,
            "sources": [SOURCE_BASE + p for p in ("models/step35.cpp", "llama-model.cpp", "llama-graph.cpp", "llama-arch.cpp")],
            "scope": "static main-text separate-QKV contract, including tensors optional in upstream but required by this profile; not compiled loading, payload correctness or CUDA support",
            "validated_tensors": len(tensors), "mtp_blocks": 0, "metadata": h,
            "families": families, "tensor_mapping": mapping}
