"""Static Hy3 separate-QKV, dense0 + MoE trunk + one embedded MTP contract.

Tensor metadata admission is not proof of payload correctness or CUDA support.
Small shapes are permitted for independent fixtures; alternative layouts fail closed.
"""
from collections import Counter
import math

LOADER_SHA = '86ebfef2c6a0f3359a2a07d2c215d61b0fa885c9'
ARCHIVE_SHA256 = 'f8e524b635b726bae74fd8f84bb9249e5b09384c63207f6707c5a3f921acad99'
LOADER_FILE_SHA256 = '224921b8ce6f9be02dc1252ef6847a93f81386021b24d7d08eb25444e787d62e'
MATRIX_TYPES = {'F32', 'Q8_0', 'Q6_K', 'Q5_K', 'Q4_K', 'Q3_K', 'IQ3_XXS', 'IQ4_XS'}


def model_metadata(meta):
    if meta.get('general.architecture') != 'hy_v3':
        raise ValueError('Expected general.architecture=hy_v3')
    if any(k.startswith('split.') for k in meta):
        raise ValueError('This Hy3 contract admits one unsplit GGUF')
    if (meta.get('tokenizer.ggml.model'), meta.get('tokenizer.ggml.pre')) != ('gpt2', 'hunyuan-dense'):
        raise ValueError('Expected gpt2/hunyuan-dense tokenizer')
    h = {}
    for key in ('block_count', 'nextn_predict_layers', 'context_length', 'embedding_length',
                'feed_forward_length', 'expert_feed_forward_length', 'expert_shared_feed_forward_length',
                'expert_count', 'expert_used_count', 'attention.head_count', 'attention.head_count_kv',
                'attention.key_length', 'attention.value_length', 'expert_gating_func'):
        v = meta.get('hy_v3.' + key)
        if type(v) is not int or not 1 <= v <= 2**31 - 1:
            raise ValueError(f'Invalid hy_v3.{key}: expected positive integer')
        h[key] = v
    if not 3 <= h['block_count'] <= 512 or h['nextn_predict_layers'] != 1:
        raise ValueError('Expected dense0, at least one trunk MoE and exactly one embedded MTP block')
    if h['expert_used_count'] > h['expert_count']:
        raise ValueError('Expert top-k exceeds expert count')
    if h['attention.head_count'] % h['attention.head_count_kv']:
        raise ValueError('Query heads must be divisible by KV heads')
    if h['attention.key_length'] != h['attention.value_length'] or h['attention.key_length'] % 2:
        raise ValueError('Expected equal K/V dimensions and even full NeoX rotary dimension')
    if h['expert_gating_func'] != 2 or meta.get('hy_v3.expert_weights_norm') is not True:
        raise ValueError('Expected sigmoid routing and normalized expert weights')
    h['expert_weights_norm'] = True
    for key in ('rope.freq_base', 'attention.layer_norm_rms_epsilon', 'expert_weights_scale'):
        v = meta.get('hy_v3.' + key)
        if type(v) not in (int, float) or not math.isfinite(v) or v <= 0:
            raise ValueError(f'Invalid hy_v3.{key}: expected finite positive number')
        h[key] = v
    extra = {k for k in meta if k.startswith('hy_v3.')} - {'hy_v3.' + k for k in h}
    if extra:
        raise ValueError(f'Unreviewed Hy3 metadata: {sorted(extra)}')
    tokens, kinds, merges = (meta.get('tokenizer.ggml.' + k) for k in ('tokens', 'token_type', 'merges'))
    if not isinstance(tokens, list) or not tokens or any(not isinstance(t, str) for t in tokens):
        raise ValueError('Invalid vocabulary')
    if len(set(tokens)) != len(tokens):
        raise ValueError('Duplicate vocabulary entry')
    if not isinstance(kinds, list) or len(kinds) != len(tokens) or any(type(t) is not int or t not in range(1, 7) for t in kinds):
        raise ValueError('Invalid token_type array')
    if not isinstance(merges, list) or any(not isinstance(m, str) or len(m.split(' ')) != 2 for m in merges):
        raise ValueError('Invalid merge list')
    if len(set(merges)) != len(merges):
        raise ValueError('Duplicate BPE merge')
    if not isinstance(meta.get('tokenizer.chat_template'), str) or not meta['tokenizer.chat_template']:
        raise ValueError('Missing embedded chat template')
    for key in ('bos_token_id', 'eos_token_id', 'padding_token_id', 'seperator_token_id'):
        v = meta.get('tokenizer.ggml.' + key)
        if type(v) is not int or not 0 <= v < len(tokens):
            raise ValueError(f'Invalid tokenizer.ggml.{key}')
    h.update(main_blocks=h['block_count'] - 1, mtp_blocks=1, dense_blocks=[0], vocab_size=len(tokens))
    return h


def validate_loader_contract(meta, tensors):
    h = model_metadata(meta)
    d, v, ff, ef, sf, e = (h[k] for k in ('embedding_length', 'vocab_size', 'feed_forward_length',
                                          'expert_feed_forward_length', 'expert_shared_feed_forward_length', 'expert_count'))
    hd, nh, nk = (h[k] for k in ('attention.key_length', 'attention.head_count', 'attention.head_count_kv'))
    expected = {}

    def add(name, shape, family, layer=None):
        expected[name] = (shape, family, layer)

    add('token_embd.weight', [d, v], 'embedding')
    add('output.weight', [d, v], 'output')
    add('output_norm.weight', [d], 'norm')
    for layer in range(h['block_count']):
        p = f'blk.{layer}.'
        for name in ('attn_norm', 'ffn_norm'):
            add(p + name + '.weight', [d], 'norm', layer)
        for name in ('attn_q_norm', 'attn_k_norm'):
            add(p + name + '.weight', [hd], 'norm', layer)
        for name, shape in (('attn_q', [d, hd * nh]), ('attn_k', [d, hd * nk]),
                            ('attn_v', [d, hd * nk]), ('attn_output', [hd * nh, d])):
            add(p + name + '.weight', shape, 'attention', layer)
        if layer == 0:
            for name, shape in (('gate', [d, ff]), ('up', [d, ff]), ('down', [ff, d])):
                add(p + 'ffn_' + name + '.weight', shape, 'dense', layer)
        else:
            add(p + 'ffn_gate_inp.weight', [d, e], 'router', layer)
            add(p + 'exp_probs_b', [e], 'router', layer)  # Deliberately no .bias suffix.
            for name, a, b in (('gate', d, ef), ('up', d, ef), ('down', ef, d)):
                add(p + f'ffn_{name}_exps.weight', [a, b, e], 'routed', layer)
            for name, a, b in (('gate', d, sf), ('up', d, sf), ('down', sf, d)):
                add(p + f'ffn_{name}_shexp.weight', [a, b], 'shared', layer)
        if layer == h['main_blocks']:
            add(p + 'nextn.eh_proj.weight', [2 * d, d], 'mtp_projection', layer)
            for name in ('enorm', 'hnorm', 'shared_head_norm'):
                add(p + 'nextn.' + name + '.weight', [d], 'norm', layer)
    items = list(tensors)
    by_name = {t.name: t for t in items}
    if len(by_name) != len(items):
        raise ValueError('Duplicate tensor name')
    missing, extra = sorted(expected.keys() - by_name.keys()), sorted(by_name.keys() - expected.keys())
    if missing or extra:
        raise ValueError(f'Loader tensor names: missing={missing}; unexpected={extra}')
    mapping, families = [], {}
    for name, t in sorted(by_name.items()):
        shape, family, layer = expected[name]
        if t.shape != shape:
            raise ValueError(f'Loader tensor shape: {name}: expected {shape}, got {t.shape}')
        allowed = {'F32'} if family in ('router', 'norm') else MATRIX_TYPES
        if t.type_name not in allowed:
            raise ValueError(f'Unreviewed tensor encoding: {name}: {t.type_name}')
        counts = families.setdefault(family, {'tensors': 0, 'types': Counter()})
        counts['tensors'] += 1
        counts['types'][t.type_name] += 1
        mapping.append({'name': name, 'expected_shape': shape, 'family': family, 'layer': layer,
                        'scope': 'mtp' if layer == h['main_blocks'] else 'main'})
    return {'source_revision': LOADER_SHA, 'loader_sha256': LOADER_FILE_SHA256,
            'scope': 'static dense0/separate-QKV/MoE/one embedded MTP layout; not payload or execution validation',
            'validated_tensors': len(items), 'metadata': h, 'families': families, 'tensor_mapping': mapping}
