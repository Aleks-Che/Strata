"""Strict admission of the reviewed MiniMax-M2.7 Q4_K_M text trunk.

This checks metadata and tensor headers, not the payload or numerical graph.
Synthetic geometries require an explicit opt-in unavailable from the inspector CLI.
"""
from collections import Counter
import math

from .gguf_reader import BLOCK_GEOMETRY, GGML_TYPES

LOADER_SHA = '86ebfef2c6a0f3359a2a07d2c215d61b0fa885c9'
ARCHIVE_SHA256 = 'f8e524b635b726bae74fd8f84bb9249e5b09384c63207f6707c5a3f921acad99'
LOADER_FILE_SHA256 = '6574a8618655d8164588627c1a4d2fc4fa8953b65fd783b8c1849f1c5c93638e'
TEMPLATE_SHA256 = '893d908f7b5cc65fdde270dcae5ea1a99647c6a7ce572ae874a57b7160069566'
PRODUCTION = {'block_count': 62, 'context_length': 204800, 'embedding_length': 3072,
              'feed_forward_length': 1536, 'expert_feed_forward_length': 1536,
              'expert_count': 256, 'expert_used_count': 8, 'expert_gating_func': 2,
              'attention.head_count': 48, 'attention.head_count_kv': 8,
              'attention.key_length': 128, 'attention.value_length': 128,
              'rope.dimension_count': 64}


def model_metadata(meta, *, allow_fixture=False):
    if meta.get('general.architecture') != 'minimax-m2':
        raise ValueError('Expected general.architecture=minimax-m2')
    if any(k.startswith('split.') for k in meta):
        raise ValueError('MiniMax admission requires one unsplit GGUF')
    h = {}
    for key in PRODUCTION:
        value = meta.get('minimax-m2.' + key)
        if type(value) is not int or not 1 <= value <= 2**31-1:
            raise ValueError(f'Invalid minimax-m2.{key}: expected positive integer')
        h[key] = value
    if not 1 <= h['block_count'] <= 512 or h['expert_used_count'] > h['expert_count']:
        raise ValueError('Invalid block count or top-k')
    if h['attention.head_count'] % h['attention.head_count_kv']:
        raise ValueError('Invalid GQA head counts')
    if h['attention.key_length'] != h['attention.value_length']:
        raise ValueError('MiniMax requires equal K/V head dimensions')
    if h['rope.dimension_count'] % 2 or h['rope.dimension_count'] > h['attention.key_length']:
        raise ValueError('Invalid partial rotary dimension')
    if h['feed_forward_length'] != h['expert_feed_forward_length'] or h['expert_gating_func'] != 2:
        raise ValueError('Expected matching expert FF widths and sigmoid routing')
    for key in ('rope.freq_base', 'attention.layer_norm_rms_epsilon'):
        value = meta.get('minimax-m2.' + key)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'Invalid minimax-m2.{key}')
        h[key] = value
    # Absence is intentional: the reviewed loader defaults to no NextN and no
    # additional expert scale. Its graph explicitly normalizes selected weights.
    extra = {k for k in meta if k.startswith('minimax-m2.')} - {'minimax-m2.'+k for k in h}
    if extra:
        raise ValueError(f'Unreviewed MiniMax metadata: {sorted(extra)}')
    if (meta.get('tokenizer.ggml.model'), meta.get('tokenizer.ggml.pre')) != ('gpt2', 'minimax-m2'):
        raise ValueError('Expected gpt2/minimax-m2 tokenizer')
    tokens, kinds, merges = (meta.get('tokenizer.ggml.'+k) for k in ('tokens', 'token_type', 'merges'))
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
    required_ids = ('bos_token_id', 'eos_token_id', 'padding_token_id', 'unknown_token_id')
    for key in required_ids:
        v = meta.get('tokenizer.ggml.'+key)
        if type(v) is not int or not 0 <= v < len(tokens):
            raise ValueError(f'Invalid tokenizer.ggml.{key}')
    allowed_tokenizer = {'model', 'pre', 'tokens', 'token_type', 'merges', *required_ids}
    unexpected = {k for k in meta if k.startswith('tokenizer.')} - {
        'tokenizer.ggml.'+k for k in allowed_tokenizer} - {'tokenizer.chat_template'}
    if unexpected:
        raise ValueError(f'Unreviewed tokenizer policy: {sorted(unexpected)}')
    if not isinstance(meta.get('tokenizer.chat_template'), str) or not meta['tokenizer.chat_template']:
        raise ValueError('Missing embedded chat template')
    if not allow_fixture:
        if any(h[k] != v for k, v in PRODUCTION.items()) or len(tokens) != 200064 or len(merges) != 199744:
            raise ValueError('Unreviewed MiniMax production geometry/vocabulary')
        for key, expected in [('rope.freq_base', 5000000), ('attention.layer_norm_rms_epsilon', 1e-6)]:
            if not math.isclose(h[key], expected, rel_tol=1e-7):
                raise ValueError(f'Unreviewed production {key}')
        for key, expected, literal in [('bos_token_id', 200034, ']~!b['), ('eos_token_id', 200020, '[e~['),
                                       ('padding_token_id', 200020, '[e~['), ('unknown_token_id', 200021, ']!d~[')]:
            if meta['tokenizer.ggml.'+key] != expected or tokens[expected] != literal:
                raise ValueError(f'Unreviewed special token: {key}')
    h.update(vocab_size=len(tokens), main_blocks=h['block_count'], mtp_blocks=0,
             nextn_metadata_present=False, qk_norm_scope='whole_projection_before_head_reshape')
    return h


def validate_loader_contract(meta, tensors, *, allow_fixture=False):
    h = model_metadata(meta, allow_fixture=allow_fixture)
    d, ff, e, v = (h[k] for k in ('embedding_length', 'expert_feed_forward_length', 'expert_count', 'vocab_size'))
    q, kv = h['attention.head_count']*h['attention.key_length'], h['attention.head_count_kv']*h['attention.key_length']
    expected = {}

    def add(name, shape, family, layer=None):
        expected[name] = (shape, family, layer)

    add('token_embd.weight', [d, v], 'embedding')
    add('output.weight', [d, v], 'output')
    add('output_norm.weight', [d], 'norm')
    for i in range(h['block_count']):
        p = f'blk.{i}.'
        for name, shape in [('attn_norm', [d]), ('ffn_norm', [d]), ('attn_q_norm', [q]), ('attn_k_norm', [kv])]:
            add(p+name+'.weight', shape, 'norm', i)
        for name, shape in [('attn_q', [d, q]), ('attn_k', [d, kv]), ('attn_v', [d, kv]), ('attn_output', [q, d])]:
            add(p+name+'.weight', shape, 'attention', i)
        add(p+'ffn_gate_inp.weight', [d, e], 'router', i)
        add(p+'exp_probs_b.bias', [e], 'router', i)
        for name in ('gate', 'up', 'down'):
            add(p+f'ffn_{name}_exps.weight', [ff, d, e] if name == 'down' else [d, ff, e], 'routed', i)
    items = list(tensors)
    by_name = {t.name: t for t in items}
    if len(items) != len(by_name):
        raise ValueError('Duplicate tensor name')
    missing, extra = sorted(expected.keys()-by_name.keys()), sorted(by_name.keys()-expected.keys())
    if missing or extra:
        raise ValueError(f'Loader tensor names: missing={missing}; unexpected={extra}')
    mapping, families = [], {}
    for name, t in sorted(by_name.items()):
        shape, family, layer = expected[name]
        if t.shape != shape or any(type(n) is not int for n in t.shape):
            raise ValueError(f'Loader tensor shape: {name}: expected {shape}, got {t.shape}')
        allowed = {'F32'} if family in ('norm', 'router') else {'Q4_K'}
        if family == 'output':
            allowed = {'Q6_K'}
        if name.endswith(('attn_v.weight', 'ffn_down_exps.weight')):
            allowed = {'Q4_K', 'Q6_K'}
        if allow_fixture:
            allowed = allowed | {'F32'}
        if t.type_name not in allowed or type(t.type_id) is not int or GGML_TYPES.get(t.type_id) != t.type_name:
            raise ValueError(f'Unreviewed tensor encoding: {name}: {t.type_name}')
        if t.shape[0] % BLOCK_GEOMETRY[t.type_name][0]:
            raise ValueError(f'Invalid quantized row: {name}')
        counts = families.setdefault(family, {'tensors': 0, 'types': Counter()})
        counts['tensors'] += 1
        counts['types'][t.type_name] += 1
        mapping.append({'name': name, 'expected_shape': shape, 'family': family, 'layer': layer, 'scope': 'main'})
    return {'source_revision': LOADER_SHA, 'loader_sha256': LOADER_FILE_SHA256,
            'scope': 'static all-MoE/separate-QKV/flattened-QK-norm contract; no payload or execution validation',
            'validated_tensors': len(items), 'metadata': h, 'families': families, 'tensor_mapping': mapping}
