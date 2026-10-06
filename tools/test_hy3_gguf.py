"""Independent small GGUF fixtures: no real model, network or GPU required."""
from dataclasses import replace
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest

if __package__:
    from .gguf_reader import TensorInfo
    from .hy3_loader_contract import validate_loader_contract
    from .check_hy3_loader import validate_registration, ARCHIVE_SHA256, LOADER_SHA
    from .inspect_hy3_gguf import inspect_model, main, protect_output, validate_ranges
else:
    from gguf_reader import TensorInfo
    from hy3_loader_contract import validate_loader_contract
    from check_hy3_loader import validate_registration, ARCHIVE_SHA256, LOADER_SHA
    from inspect_hy3_gguf import inspect_model, main, protect_output, validate_ranges


def fixture():
    m = {'general.architecture': 'hy_v3', 'hy_v3.block_count': 3, 'hy_v3.nextn_predict_layers': 1,
         'hy_v3.context_length': 2048, 'hy_v3.embedding_length': 256, 'hy_v3.feed_forward_length': 512,
         'hy_v3.expert_feed_forward_length': 256, 'hy_v3.expert_shared_feed_forward_length': 256,
         'hy_v3.expert_count': 4, 'hy_v3.expert_used_count': 2, 'hy_v3.attention.head_count': 4,
         'hy_v3.attention.head_count_kv': 1, 'hy_v3.attention.key_length': 64, 'hy_v3.attention.value_length': 64,
         'hy_v3.expert_gating_func': 2, 'hy_v3.expert_weights_norm': True,
         'hy_v3.expert_weights_scale': 2.826, 'hy_v3.rope.freq_base': 11158840.0,
         'hy_v3.attention.layer_norm_rms_epsilon': 1e-5,
         'tokenizer.ggml.model': 'gpt2', 'tokenizer.ggml.pre': 'hunyuan-dense',
         'tokenizer.ggml.tokens': [f'token{i}' for i in range(16)], 'tokenizer.ggml.token_type': [1]*16,
         'tokenizer.ggml.merges': [], 'tokenizer.chat_template': '{{ messages }}',
         'tokenizer.ggml.bos_token_id': 0, 'tokenizer.ggml.eos_token_id': 1,
         'tokenizer.ggml.padding_token_id': 2, 'tokenizer.ggml.seperator_token_id': 3}
    ts, end = [], 0
    # Fixture geometry is intentionally independent of the production table.
    geom = {'F32': (0, 1, 4), 'Q8_0': (8, 32, 34), 'Q6_K': (14, 256, 210),
            'IQ3_XXS': (18, 256, 98), 'IQ4_XS': (23, 256, 136),
            'Q3_K': (11, 256, 110), 'Q4_K': (12, 256, 144)}

    def add(name, shape, kind='F32'):
        nonlocal end
        type_id, block, byte = geom[kind]
        t = TensorInfo(name, shape, type_id, kind, end)
        ts.append(t)
        elements = 1
        for d in shape:
            elements *= d
        end += elements // block * byte
        end += -end % 32

    add('output_norm.weight', [256])
    add('token_embd.weight', [256, 16], 'Q6_K')
    add('output.weight', [256, 16], 'Q6_K')
    for i in range(3):
        p = f'blk.{i}.'
        for name, shape in [('attn_norm', [256]), ('ffn_norm', [256]), ('attn_q_norm', [64]), ('attn_k_norm', [64])]:
            add(p + name + '.weight', shape)
        for name, shape in [('attn_q', [256, 256]), ('attn_k', [256, 64]), ('attn_v', [256, 64]), ('attn_output', [256, 256])]:
            add(p + name + '.weight', shape, 'Q8_0')
        if i == 0:
            for name, shape in [('gate', [256, 512]), ('up', [256, 512]), ('down', [512, 256])]:
                add(p + 'ffn_' + name + '.weight', shape, 'Q8_0')
        else:
            add(p + 'ffn_gate_inp.weight', [256, 4])
            add(p + 'exp_probs_b', [4])
            for name in ('gate', 'up', 'down'):
                kind = ('IQ4_XS' if name == 'down' else 'IQ3_XXS') if i == 1 else ('Q4_K' if name == 'down' else 'Q3_K')
                add(p + f'ffn_{name}_exps.weight', [256, 256, 4], kind)
                add(p + f'ffn_{name}_shexp.weight', [256, 256], 'Q8_0')
        if i == 2:
            add(p + 'nextn.eh_proj.weight', [512, 256], 'Q8_0')
            for name in ('enorm', 'hnorm', 'shared_head_norm'):
                add(p + 'nextn.' + name + '.weight', [256])
    return m, ts


def text(s):
    b = s.encode('utf-8')
    return struct.pack('<Q', len(b)) + b


def value(v):
    if isinstance(v, str):
        return struct.pack('<I', 8) + text(v)
    if type(v) is bool:
        return struct.pack('<I?', 7, v)
    if type(v) is float:
        return struct.pack('<If', 6, v)
    if isinstance(v, list):
        encoded = [value(x) for x in v]
        kind = encoded[0][:4] if encoded else struct.pack('<I', 8)
        return struct.pack('<I', 9) + kind + struct.pack('<Q', len(v)) + b''.join(x[4:] for x in encoded)
    return struct.pack('<IQ', 10, v)


def write_fixture(path, meta, tensors):
    header = b'GGUF' + struct.pack('<IQQ', 3, len(tensors), len(meta))
    for k, v in meta.items():
        header += text(k) + value(v)
    for t in tensors:
        header += text(t.name) + struct.pack('<I', len(t.shape)) + struct.pack(f'<{len(t.shape)}Q', *t.shape)
        header += struct.pack('<IQ', t.type_id, t.offset)
    header += bytes(-len(header) % 32)
    path.write_bytes(header + bytes(max(t.offset + t.expected_bytes() for t in tensors)))


class Hy3Contract(unittest.TestCase):
    def setUp(self):
        self.meta, self.tensors = fixture()

    def test_layout_and_mtp_boundary(self):
        result = validate_loader_contract(self.meta, self.tensors)
        self.assertEqual(result['validated_tensors'], 50)
        self.assertEqual(sum(r['scope'] == 'mtp' for r in result['tensor_mapping']), 20)
        self.assertEqual(result['metadata']['main_blocks'], 2)

    def test_bad_metadata(self):
        cases = [('general.architecture', 'hunyuan-moe'), ('hy_v3.block_count', True),
                 ('hy_v3.nextn_predict_layers', 0), ('hy_v3.nextn_predict_layers', 2),
                 ('hy_v3.expert_gating_func', 1), ('hy_v3.expert_weights_norm', False),
                 ('hy_v3.expert_used_count', 5), ('hy_v3.attention.head_count_kv', 3),
                 ('hy_v3.attention.value_length', 32), ('hy_v3.rope.freq_base', float('nan')),
                 ('hy_v3.expert_weights_scale', float('inf')), ('hy_v3.attention.sliding_window', 512),
                 ('split.count', 2), ('tokenizer.ggml.pre', 'deepseek-v3'),
                 ('tokenizer.ggml.eos_token_id', 16), ('tokenizer.chat_template', '')]
        for k, v in cases:
            with self.subTest(key=k, value=v), self.assertRaises(ValueError):
                validate_loader_contract({**self.meta, k: v}, self.tensors)

    def test_missing_mandatory_tensors(self):
        for name in ('blk.0.ffn_gate.weight', 'blk.1.exp_probs_b', 'blk.2.ffn_gate_exps.weight',
                     'blk.2.nextn.shared_head_norm.weight', 'output.weight'):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'missing'):
                validate_loader_contract(self.meta, [t for t in self.tensors if t.name != name])

    def test_extra_or_duplicate_tensor(self):
        for added in (self.tensors[0], replace(self.tensors[0], name='blk.2.nextn.embed_tokens.weight')):
            with self.subTest(name=added.name), self.assertRaises(ValueError):
                validate_loader_contract(self.meta, self.tensors + [added])

    def test_shape_type_and_bias_suffix(self):
        for old, change in [('blk.1.exp_probs_b', {'name': 'blk.1.exp_probs_b.bias'}),
                            ('blk.2.nextn.eh_proj.weight', {'shape': [256, 512]}),
                            ('blk.1.ffn_gate_exps.weight', {'type_id': 42, 'type_name': 'Q2_0'}),
                            ('output_norm.weight', {'type_id': 1, 'type_name': 'F16'})]:
            ts = [replace(t, **change) if t.name == old else t for t in self.tensors]
            with self.subTest(old=old), self.assertRaises(ValueError):
                validate_loader_contract(self.meta, ts)

    def test_bad_tokenizer_arrays(self):
        for k, v in [('tokens', ['same', 'same']), ('token_type', [1]), ('merges', ['a b', 'a b']), ('merges', ['abc'])]:
            with self.subTest(k=k), self.assertRaises(ValueError):
                validate_loader_contract({**self.meta, 'tokenizer.ggml.' + k: v}, self.tensors)

    def test_ranges_above_4gib(self):
        t = TensorInfo('matrix', [256, 2], 12, 'Q4_K', 2**32 + 32)
        rows = validate_ranges([t], 64, 2**32 + 384, 32)
        self.assertEqual(rows[0]['file_offset'], 2**32 + 96)
        self.assertEqual(rows[0]['bytes'], 288)

    def test_q5_q6_expert_layers(self):
        for type_id, kind in ((13, 'Q5_K'), (14, 'Q6_K')):
            ts = [replace(t, type_id=type_id, type_name=kind) if t.name == 'blk.1.ffn_down_exps.weight' else t for t in self.tensors]
            with self.subTest(kind=kind):
                self.assertEqual(validate_loader_contract(self.meta, ts)['validated_tensors'], 50)

    def test_bad_ranges(self):
        t = TensorInfo('x', [256, 2], 12, 'Q4_K', 0)
        cases = [([t, t], 64, 1024, 32), ([t, replace(t, name='y', offset=32)], 64, 1024, 32),
                 ([t], 64, 351, 32), ([replace(t, offset=1)], 64, 1024, 32),
                 ([replace(t, shape=[128, 4])], 64, 1024, 32),
                 ([replace(t, shape=[0, 2])], 64, 1024, 32),
                 ([replace(t, type_id=999)], 64, 1024, 32),
                 ([replace(t, offset=2**63 - 32)], 64, 2**63 - 1, 32),
                 ([t], 64, 1024, 3), ([t], 64, 1024, True)]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                validate_ranges(*args)

    def test_compiled_registration_guard(self):
        from copy import deepcopy
        inventory = {'tensors': [{'name': 'main', 'scope': 'main', 'bytes': 4},
                                 {'name': 'mtp', 'scope': 'mtp', 'bytes': 8}],
                     'loader_contract': {'metadata': {'main_blocks': 2, 'block_count': 3}}}
        base = {'source_revision': LOADER_SHA, 'archive_sha256': ARCHIVE_SHA256, 'patches': 'hy3-mtp-load-flags',
                'runs': [{'load_mtp': flag, 'main_blocks': 2, 'all_blocks': 3, 'main_tensors': 1,
                          'main_logical_bytes': 4, 'allocated_weight_bytes': 0,
                          'mtp_tensors': [{'name': 'mtp', 'bytes': 8}] if flag else [],
                          'mtp_logical_bytes': 8 if flag else 0} for flag in (False, True)]}
        validate_registration(base, inventory)
        for changes in ({'mtp_tensors': [{'name': 'mtp', 'bytes': 8}], 'mtp_logical_bytes': 8},
                        {'allocated_weight_bytes': 8}, {'main_logical_bytes': 0}):
            bad = deepcopy(base)
            bad['runs'][0].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_registration(bad, inventory)

    def test_file_inspection_and_output_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'small.gguf'
            write_fixture(p, self.meta, self.tensors)
            report = inspect_model(p)
            self.assertEqual(report['status'], 'pass')
            self.assertEqual(report['tensor_count'], 50)
            self.assertEqual(sum(report['group_bytes'].values()), report['payload_bytes'])
            before = p.read_bytes()
            self.assertEqual(main(['--gguf', str(p), '--output', str(p)]), 1)
            self.assertEqual(p.read_bytes(), before)
            out = Path(directory) / 'report.json'
            self.assertEqual(main(['--gguf', str(p), '--output', str(out)]), 0)
            self.assertEqual(json.loads(out.read_text(encoding='utf-8'))['header_sha256'], report['header_sha256'])

    def test_hardlink_output_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            p, out = Path(directory) / 'source', Path(directory) / 'alias'
            p.write_bytes(b'protected')
            os.link(p, out)
            with self.assertRaises(ValueError):
                protect_output(out, p)

    def test_truncated_header_and_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            p, out = Path(directory) / 'bad.gguf', Path(directory) / 'report.json'
            write_fixture(p, self.meta, self.tensors)
            original = p.read_bytes()
            for raw in (b'GGUF', original[:50], original[:-1]):
                p.write_bytes(raw)
                self.assertEqual(main(['--gguf', str(p), '--output', str(out)]), 1)
                self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()
