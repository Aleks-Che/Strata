"""Static tensor contract for the complete Unsloth GLM-5.3-Flash artifact.

This is a header preflight, not a compiled loader or numerical compatibility test.
The source revision is an audit candidate, not the production backend pin.
See docs/GLM53/GLM53_FLASH_LOADER_COMPATIBILITY.md for scope and evidence.
"""
from collections import Counter
import math

LOADER_SHA = "86ebfef2c6a0f3359a2a07d2c215d61b0fa885c9"
SOURCE_BASE = f"https://github.com/unslothai/llama.cpp/blob/{LOADER_SHA}/src/"


def validate_loader_contract(metadata, tensors):
    """Check full-model, separate-Q/K/V layout; fail closed on unaccounted weights.

`tensors` maps names to TensorInfo objects already admitted by the GGUF inspector.
Partial trunk/draft artifacts and alternate fused layouts are outside this profile.
"""
    used = {}

    def get(key, kind, default=None, minimum=None):
        name = "glm5next." + key
        value = metadata.get(name, default)
        valid = type(value) is kind
        if kind is float:
            valid = type(value) in (int, float) and math.isfinite(value)
        if not valid or (minimum is not None and value < minimum):
            raise ValueError(f"Loader metadata: {name}: invalid value {value!r}")
        used[name] = value
        return value

    if metadata.get("general.architecture") != "glm5next":
        raise ValueError("Loader contract requires general.architecture=glm5next")
    blocks = get("block_count", int, minimum=2)
    mtp = get("nextn_predict_layers", int, minimum=1)
    main = blocks - mtp
    if main <= 0:
        raise ValueError("Loader metadata: NextN must leave at least one main block")
    width = get("embedding_length", int, minimum=1)
    vocab = get("vocab_size", int, minimum=1)
    get("context_length", int, minimum=1)
    heads = get("attention.head_count", int, minimum=1)
    kv_heads = get("attention.head_count_kv", list)
    if len(kv_heads) != blocks or any(type(h) is not int or h not in (0, 1) for h in kv_heads):
        raise ValueError("Loader metadata: head_count_kv needs one 0/1 entry per block")
    recurrent = [i for i in range(main) if kv_heads[i] == 0]
    if not 0 < len(recurrent) < main:
        raise ValueError("Loader metadata: main graph needs both KDA and DSA blocks")
    if any(h != 1 for h in kv_heads[main:]):
        raise ValueError("Loader metadata: this GLM Flash profile requires DSA NextN blocks")
    dense = get("leading_dense_block_count", int, minimum=0)
    if dense > main:
        raise ValueError("Loader metadata: dense block count exceeds main graph")
    ff = get("feed_forward_length", int, minimum=1)
    expert_ff = get("expert_feed_forward_length", int, minimum=1)
    experts = get("expert_count", int, minimum=1)
    selected = get("expert_used_count", int, minimum=1)
    if selected > experts:
        raise ValueError("Loader metadata: more selected experts than available")
    shared_count = get("expert_shared_count", int, minimum=0)
    shared_ff = get("expert_shared_feed_forward_length", int, default=0, minimum=0)
    shared_ff = shared_ff or expert_ff * max(1, shared_count)
    get("expert_weights_scale", float, minimum=0)
    get("expert_weights_norm", bool)
    get("expert_gating_func", int, minimum=0)
    get("attention.layer_norm_rms_epsilon", float, minimum=0)
    get("attention.layer_norm_epsilon", float, minimum=0)
    q_rank = get("attention.q_lora_rank", int, minimum=1)
    kv_rank = get("attention.kv_lora_rank", int, minimum=1)
    key_dim = get("attention.key_length_mla", int, minimum=1)
    value_dim = get("attention.value_length_mla", int, minimum=1)
    if get("rope.dimension_count", int, minimum=0) != 0:
        raise ValueError("Loader metadata: GLM MLA requires rope.dimension_count=0")
    head_dim = get("kda.head_dim", int, minimum=1)
    conv = get("ssm.conv_kernel", int, minimum=2)
    if get("kda.gate_lower_bound", float) >= 0:
        raise ValueError("Loader metadata: kda.gate_lower_bound must be negative")
    index_heads = get("attention.indexer.head_count", int, minimum=1)
    index_dim = get("attention.indexer.key_length", int, minimum=1)
    top_k = get("attention.indexer.top_k", int, minimum=1)
    pool = get("attention.indexer.kpool", int, minimum=1)
    if pool <= 1 or top_k < pool or top_k % pool:
        raise ValueError("Loader metadata: indexer top_k must be divisible by kpool > 1")
    hc = get("hyper_connection.count", int, minimum=1)
    get("hyper_connection.sinkhorn_iterations", int, minimum=1)
    get("hyper_connection.epsilon", float, minimum=0)
    for key in ("swiglu_clamp_exp", "swiglu_clamp_shexp"):
        name = "glm5next." + key
        value = metadata.get(name, 0.0)
        numbers = value if isinstance(value, list) else [value] * blocks
        if len(numbers) != blocks or any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in numbers):
            raise ValueError(f"Loader metadata: {name}: expected scalar or block_count finite values")
        used[name] = value

    expected = {}
    optional = set()

    def add(name, shape, family, layer=None, required=True):
        expected[name] = (shape, family, layer)
        if not required:
            optional.add(name)

    add("token_embd.weight", [width, vocab], "global")
    add("output_norm.weight", [width], "global")
    add("output.weight", [width, vocab], "global", required=False)
    inner = heads * head_dim
    for layer in range(blocks):
        def weight(name, shape, family, suffix="weight", required=True):
            add(f"blk.{layer}.{name}" + ("." + suffix if suffix else ""), shape, family, layer, required)

        for name in ("attn_norm", "ffn_norm"):
            weight(name, [width], "norm")
        if layer < main:
            for branch in ("attn", "ffn"):
                for name, shape in (("fn", [hc * width, hc * (hc + 2)]),
                                    ("base", [hc * (hc + 2)]), ("scale", [3])):
                    weight(f"hc_{branch}_{name}", shape, "mhc")
        if kv_heads[layer] == 0:
            for axis in ("q", "k", "v"):
                weight("attn_" + axis, [width, inner], "kda")
                weight("ssm_conv1d_" + axis, [conv, 1, inner, 1], "kda")
            for gate in ("f", "g"):
                weight(f"ssm_{gate}_a", [width, head_dim], "kda")
                weight(f"ssm_{gate}_b", [head_dim, inner], "kda")
            for name, shape, suffix in (("ssm_beta", [width, heads], "weight"),
                                        ("ssm_a", [heads], ""), ("ssm_dt", [inner], "bias"),
                                        ("ssm_norm", [head_dim], "weight"),
                                        ("attn_output", [inner, width], "weight")):
                weight(name, shape, "kda", suffix)
        else:
            for name, shape in {
                "attn_q_a": [width, q_rank], "attn_q_a_norm": [q_rank],
                "attn_q_b": [q_rank, heads * key_dim], "attn_kv_a_mqa": [width, kv_rank],
                "attn_kv_a_norm": [kv_rank], "attn_k_b": [key_dim, kv_rank, heads],
                "attn_v_b": [kv_rank, value_dim, heads], "attn_output": [heads * value_dim, width],
            }.items():
                weight(name, shape, "dsa_mla")
            for name, shape in {
                "indexer.k_norm": [index_dim], "indexer.proj": [width, index_heads],
                "indexer.attn_k": [width, index_dim], "indexer.attn_q_b": [q_rank, index_heads * index_dim],
                "indexer_compressor_gate": [width, index_dim], "indexer_compressor_ape": [index_dim, pool],
            }.items():
                weight(name, shape, "indexer")
            weight("indexer.k_norm", [index_dim], "indexer", "bias")
        if layer < dense:
            for name, shape in (("gate", [width, ff]), ("up", [width, ff]), ("down", [ff, width])):
                weight("ffn_" + name, shape, "dense_ffn")
        else:
            weight("ffn_gate_inp", [width, experts], "router")
            weight("exp_probs_b", [experts], "router", "bias")
            for name, in_dim, out_dim in (("gate", width, expert_ff), ("up", width, expert_ff), ("down", expert_ff, width)):
                weight(f"ffn_{name}_exps", [in_dim, out_dim, experts], "routed_experts")
            for name, shape in (("gate", [width, shared_ff]), ("up", [width, shared_ff]), ("down", [shared_ff, width])):
                weight(f"ffn_{name}_shexp", shape, "shared_experts")
        if layer >= main:
            weight("nextn.eh_proj", [2 * width, width], "nextn")
            for norm in ("enorm", "hnorm", "shared_head_norm"):
                weight("nextn." + norm, [width], "nextn")
            for name in ("embed_tokens", "shared_head_head"):
                weight("nextn." + name, [width, vocab], "nextn", required=False)

    missing = sorted(expected.keys() - tensors.keys() - optional)
    unexpected = sorted(tensors.keys() - expected.keys())
    if missing or unexpected:
        raise ValueError(f"Loader tensor names: missing={missing}; unexpected={unexpected}")
    families = {}
    main_tensors = mtp_tensors = 0
    for name, tensor in sorted(tensors.items()):
        shape, family, layer = expected[name]
        actual = tensor.shape
        # ggml fills omitted trailing dimensions with ones. Do not squeeze inner
        # singleton dimensions: that would accept a different convolution layout.
        if not 1 <= len(actual) <= 4 or actual + [1] * (4 - len(actual)) != shape + [1] * (4 - len(shape)):
            raise ValueError(f"Loader tensor shape: {name}: expected {shape}, got {actual}")
        branch = "mtp" if layer is not None and layer >= main else "main"
        if branch == "mtp":
            mtp_tensors += 1
        else:
            main_tensors += 1
        summary = families.setdefault(branch + "." + family, {"tensors": 0, "types": Counter()})
        summary["tensors"] += 1
        summary["types"][tensor.type_name] += 1
    return {
        "source_revision": LOADER_SHA,
        "sources": [SOURCE_BASE + path for path in ("models/glm5next.cpp", "llama-arch.cpp", "llama-model-loader.cpp")],
        "scope": "static full-model separate-QKV tensor names/shapes and metadata; not compiled loader, GPU types or inference",
        "validated_tensors": len(tensors), "main_tensors": main_tensors, "mtp_tensors": mtp_tensors,
        "main_graph_blocks": list(range(main)), "mtp_weight_blocks": list(range(main, blocks)),
        "kda_main_blocks": recurrent,
        "dsa_main_blocks": [i for i in range(main) if kv_heads[i] != 0],
        "optional_tensors_absent": sorted(optional - tensors.keys()),
        "metadata": used, "families": dict(sorted(families.items())),
    }
