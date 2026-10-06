"""Admission for the reviewed MiMo-V2.6-Flash-RL text-only GGUF layout.

No payload reads. Small fixtures require an explicit opt-in unavailable in the CLI.
Changing weights, topology or export scope requires a separate admission review.
"""
from collections import Counter
import math

from .gguf_reader import BLOCK_GEOMETRY, GGML_TYPES

LOADER_SHA = '86ebfef2c6a0f3359a2a07d2c215d61b0fa885c9'
ARCHIVE_SHA256 = 'f8e524b635b726bae74fd8f84bb9249e5b09384c63207f6707c5a3f921acad99'
LOADER_FILE_SHA256 = '2ec5caa11fb9ab7604b798a17cd80244427596581c06cdb033762b6d56f50823'
REFERENCE_SHA = '58367713a6935c0810103378144008df32e3d5db'
SOURCE_SHA = '3b38d063180c3e4aed9691fdc735f3d10b266ee4'
EXPORT_SCOPE = 'text_trunk_without_mtp_or_modality_companions'
TEMPLATE_SHA256 = '11ea52e156de38a458e6b7720ad45915d65b97d4ec979a09f55e3c9bd1b4d059'
FULL_LAYERS = [0, 5, 11, 17, 23, 29, 35, 41, 47]


def model_metadata(meta, *, allow_fixture=False):
    if meta.get('general.architecture') != 'mimo2':
        raise ValueError('Expected general.architecture=mimo2')
    if any(k.startswith('split.') for k in meta):
        raise ValueError('MiMo admission requires one unsplit GGUF')
    h = {}
    positive = ('block_count', 'context_length', 'embedding_length', 'feed_forward_length',
                'expert_feed_forward_length', 'expert_count', 'expert_used_count',
                'attention.head_count', 'attention.key_length', 'attention.value_length',
                'attention.sliding_window', 'rope.dimension_count', 'expert_gating_func',
                'expert_group_count', 'expert_group_used_count')
    for key in positive:
        v = meta.get('mimo2.' + key)
        if type(v) is not int or not 1 <= v <= 2**31-1:
            raise ValueError(f'Invalid mimo2.{key}: expected positive integer')
        h[key] = v
    if not 2 <= h['block_count'] <= 512 or h['expert_used_count'] > h['expert_count']:
        raise ValueError('Invalid block count or top-k')
    if (h['expert_gating_func'], h['expert_group_count'], h['expert_group_used_count']) != (2, 1, 1):
        raise ValueError('Expected sigmoid routing with one expert group')
    if type(meta.get('mimo2.nextn_predict_layers')) is not int or meta['mimo2.nextn_predict_layers'] != 0:
        raise ValueError('This MiMo contract excludes MTP weights')
    h['nextn_predict_layers'] = 0
    if meta.get('mimo2.export_scope') != EXPORT_SCOPE:
        raise ValueError('Expected text-only export scope without MTP or modality companions')
    h['export_scope'] = EXPORT_SCOPE
    for key in ('attention.head_count_kv', 'attention.sliding_window_pattern'):
        v = meta.get('mimo2.' + key)
        if not isinstance(v, list) or len(v) != h['block_count'] or any(type(x) is not int for x in v):
            raise ValueError(f'Invalid per-layer {key}')
        h[key] = v
    pattern, kv = h['attention.sliding_window_pattern'], h['attention.head_count_kv']
    if set(pattern) != {0, 1} or pattern[0] != 0:
        raise ValueError('Expected full and SWA layers, with a full first layer')
    if any(x <= 0 or h['attention.head_count'] % x for x in kv):
        raise ValueError('Invalid KV head count')
    if h['rope.dimension_count'] % 2 or h['rope.dimension_count'] > h['attention.key_length']:
        raise ValueError('Invalid partial rotary dimension')
    for key in ('rope.freq_base', 'rope.freq_base_swa', 'attention.layer_norm_rms_epsilon', 'attention.value_scale'):
        v = meta.get('mimo2.' + key)
        if type(v) not in (int, float) or not math.isfinite(v) or v <= 0:
            raise ValueError(f'Invalid mimo2.{key}')
        h[key] = v
    provenance = ('source_revision', 'source_quantization', 'converter_revision', 'reference_expert_storage')
    for key in provenance:
        v = meta.get('mimo2.' + key)
        if not isinstance(v, str) or not v:
            raise ValueError(f'Missing mimo2.{key}')
        h[key] = v
    if h['source_revision'] != SOURCE_SHA or h['converter_revision'] != REFERENCE_SHA:
        raise ValueError('Unreviewed source or converter revision')
    extra = {k for k in meta if k.startswith('mimo2.')} - {'mimo2.' + k for k in h}
    if extra:
        raise ValueError(f'Unreviewed MiMo metadata: {sorted(extra)}')

    if (meta.get('tokenizer.ggml.model'), meta.get('tokenizer.ggml.pre')) != ('gpt2', 'qwen2'):
        raise ValueError('Expected gpt2/qwen2 tokenizer')
    if meta.get('tokenizer.ggml.add_bos_token') is not False:
        raise ValueError('Expected add_bos_token=false')
    tokens, kinds, merges = (meta.get('tokenizer.ggml.' + k) for k in ('tokens', 'token_type', 'merges'))
    if not isinstance(tokens, list) or not tokens or any(not isinstance(t, str) or not t for t in tokens):
        raise ValueError('Invalid vocabulary')
    if len(set(tokens)) != len(tokens):
        raise ValueError('Duplicate vocabulary entry')
    if not isinstance(kinds, list) or len(kinds) != len(tokens) or any(type(k) is not int or k not in range(1, 7) for k in kinds):
        raise ValueError('Invalid token_type array')
    if not isinstance(merges, list) or any(not isinstance(m, str) or len(m.split(' ')) != 2 for m in merges):
        raise ValueError('Invalid merge list')
    if len(set(merges)) != len(merges):
        raise ValueError('Duplicate BPE merge')
    token_set = set(tokens)
    if any(a not in token_set or b not in token_set or a+b not in token_set for a, b in (m.split(' ') for m in merges)):
        raise ValueError('BPE merge outside vocabulary')
    for key in ('eos_token_id', 'padding_token_id'):
        v = meta.get('tokenizer.ggml.' + key)
        if type(v) is not int or not 0 <= v < len(tokens):
            raise ValueError(f'Invalid tokenizer.ggml.{key}')
    for key, v in meta.items():
        if key.startswith('tokenizer.ggml.') and key.endswith('_token_id'):
            if type(v) is not int or not 0 <= v < len(tokens):
                raise ValueError(f'Invalid {key}')
    template = meta.get('tokenizer.chat_template')
    if not isinstance(template, str) or not template:
        raise ValueError('Missing embedded chat template')
    if not allow_fixture:
        reviewed = {'block_count': 48, 'context_length': 1048576, 'embedding_length': 4096,
                    'feed_forward_length': 16384, 'expert_feed_forward_length': 2048,
                    'expert_count': 256, 'expert_used_count': 8, 'attention.head_count': 64,
                    'attention.key_length': 192, 'attention.value_length': 128,
                    'attention.sliding_window': 128, 'rope.dimension_count': 64}
        if any(h[k] != v for k, v in reviewed.items()):
            raise ValueError('Unreviewed MiMo production geometry')
        if [i for i, s in enumerate(pattern) if not s] != FULL_LAYERS or kv != [8 if s else 4 for s in pattern]:
            raise ValueError('Unreviewed MiMo full/SWA pattern or KV heads')
        for key, expected in [('rope.freq_base', 1e7), ('rope.freq_base_swa', 1e4),
                              ('attention.layer_norm_rms_epsilon', 1e-6), ('attention.value_scale', .707)]:
            if not math.isclose(h[key], expected, rel_tol=1e-6):
                raise ValueError(f'Unreviewed MiMo {key}')
        if len(tokens) != 152576 or len(merges) != 151387:
            raise ValueError('Unreviewed MiMo vocabulary size')
        if meta['tokenizer.ggml.eos_token_id'] != 151645 or meta['tokenizer.ggml.padding_token_id'] != 151643:
            raise ValueError('Unreviewed MiMo EOS/PAD')
        if (tokens[151645], tokens[151643]) != ('<|im_end|>', '<|endoftext|>'):
            raise ValueError('MiMo EOS/PAD token mismatch')
    h.update(vocab_size=len(tokens), main_blocks=h['block_count'], mtp_blocks=0,
             full_layers=[i for i, s in enumerate(pattern) if not s],
             swa_layers=[i for i, s in enumerate(pattern) if s], fixture=allow_fixture)
    return h


def validate_loader_contract(meta, tensors, *, allow_fixture=False):
    h = model_metadata(meta, allow_fixture=allow_fixture)
    d, ff, ef, e, v = (h[k] for k in ('embedding_length', 'feed_forward_length', 'expert_feed_forward_length', 'expert_count', 'vocab_size'))
    nh, dk, dv = (h[k] for k in ('attention.head_count', 'attention.key_length', 'attention.value_length'))
    expected = {}

    def add(name, shape, family, layer=None):
        expected[name] = (shape, family, layer)

    add('token_embd.weight', [d, v], 'embedding')
    add('output.weight', [d, v], 'output')
    add('output_norm.weight', [d], 'norm')
    for i, nk in enumerate(h['attention.head_count_kv']):
        p = f'blk.{i}.'
        add(p+'attn_norm.weight', [d], 'norm', i)
        add(p+'ffn_norm.weight', [d], 'norm', i)
        add(p+'attn_qkv.weight', [d, nh*dk+nk*(dk+dv)], 'attention', i)
        add(p+'attn_output.weight', [nh*dv, d], 'attention', i)
        if h['attention.sliding_window_pattern'][i]:
            add(p+'attn_sinks.weight', [nh], 'sink', i)
        if i == 0:
            for name, shape in [('gate', [d, ff]), ('up', [d, ff]), ('down', [ff, d])]:
                add(p+f'ffn_{name}.weight', shape, 'dense', i)
        else:
            add(p+'ffn_gate_inp.weight', [d, e], 'router', i)
            add(p+'exp_probs_b.bias', [e], 'router', i)
            for name, shape in [('gate', [d, ef, e]), ('up', [d, ef, e]), ('down', [ef, d, e])]:
                add(p+f'ffn_{name}_exps.weight', shape, 'routed', i)
    items = list(tensors)
    by_name = {t.name: t for t in items}
    if len(by_name) != len(items):
        raise ValueError('Duplicate tensor name')
    missing, extra = sorted(expected.keys()-by_name.keys()), sorted(by_name.keys()-expected.keys())
    if missing or extra:
        raise ValueError(f'Loader tensor names: missing={missing}; unexpected={extra}')
    mapping, families = [], {}
    for name, t in sorted(by_name.items()):
        shape, family, layer = expected[name]
        if t.shape != shape or any(type(n) is not int for n in t.shape):
            raise ValueError(f'Loader tensor shape: {name}: expected {shape}, got {t.shape}')
        allowed = {'F32'} if family in ('router', 'norm', 'sink') else {'BF16'}
        if family == 'routed':
            allowed = {'Q2_K', 'Q3_K', 'MXFP4'} if '.ffn_down_' in name else {'Q2_K', 'Q3_K'}
        if t.type_name not in allowed or type(t.type_id) is not int or GGML_TYPES.get(t.type_id) != t.type_name:
            raise ValueError(f'Unreviewed tensor encoding: {name}: {t.type_name}')
        if t.shape[0] % BLOCK_GEOMETRY[t.type_name][0]:
            raise ValueError(f'Invalid quantized row: {name}')
        counts = families.setdefault(family, {'tensors': 0, 'types': Counter()})
        counts['tensors'] += 1
        counts['types'][t.type_name] += 1
        mapping.append({'name': name, 'expected_shape': shape, 'family': family, 'layer': layer, 'scope': 'main'})
    return {'source_revision': LOADER_SHA, 'loader_sha256': LOADER_FILE_SHA256,
            'scope': 'static fused-QKV/full-SWA/dense0/MoE text trunk; not payload or execution validation',
            'validated_tensors': len(items), 'metadata': h, 'families': families, 'tensor_mapping': mapping}
