"""Independent MiniMax tensor geometry and negative admission fixtures; no GPU."""
from copy import deepcopy
from dataclasses import replace
import os
from pathlib import Path
import struct
import tempfile
import unittest

from tools.gguf_reader import TensorInfo
from tools.minimax_m2_loader_contract import validate_loader_contract, LOADER_SHA, ARCHIVE_SHA256
from tools.inspect_minimax_m2_gguf import inspect_model, main, protect_output, validate_ranges
from tools.check_minimax_m2_oracles import validate_registration
from tools.test_hy3_gguf import write_fixture  # Generic small GGUF binary writer.


def fixture():
    meta = {'general.architecture': 'minimax-m2', 'minimax-m2.block_count': 2,
            'minimax-m2.context_length': 1024, 'minimax-m2.embedding_length': 256,
            'minimax-m2.feed_forward_length': 512, 'minimax-m2.expert_feed_forward_length': 512,
            'minimax-m2.expert_count': 4, 'minimax-m2.expert_used_count': 2,
            'minimax-m2.expert_gating_func': 2, 'minimax-m2.attention.head_count': 8,
            'minimax-m2.attention.head_count_kv': 2, 'minimax-m2.attention.key_length': 64,
            'minimax-m2.attention.value_length': 64, 'minimax-m2.rope.dimension_count': 32,
            'minimax-m2.rope.freq_base': 5000000.0, 'minimax-m2.attention.layer_norm_rms_epsilon': 1e-6,
            'tokenizer.ggml.model': 'gpt2', 'tokenizer.ggml.pre': 'minimax-m2',
            'tokenizer.ggml.tokens': [f'token{i}' for i in range(16)], 'tokenizer.ggml.token_type': [1]*16,
            'tokenizer.ggml.merges': [], 'tokenizer.ggml.bos_token_id': 0, 'tokenizer.ggml.eos_token_id': 1,
            'tokenizer.ggml.padding_token_id': 1, 'tokenizer.ggml.unknown_token_id': 2,
            'tokenizer.chat_template': '{{ messages }}'}
    tensors, end = [], 0
    geometry = {'F32': (0, 1, 4), 'Q4_K': (12, 256, 144), 'Q6_K': (14, 256, 210)}

    def add(name, shape, kind='F32'):
        nonlocal end
        type_id, block, size = geometry[kind]
        t = TensorInfo(name, shape, type_id, kind, end)
        tensors.append(t)
        end += t.elements//block*size
        end += -end % 32

    add('token_embd.weight', [256, 16], 'Q4_K')
    add('output.weight', [256, 16], 'Q6_K')
    add('output_norm.weight', [256])
    for i in range(2):
        p = f'blk.{i}.'
        add(p+'attn_norm.weight', [256])
        add(p+'ffn_norm.weight', [256])
        # Explicit constants, independently derived from 8/2 heads * 64.
        add(p+'attn_q_norm.weight', [512])
        add(p+'attn_k_norm.weight', [128])
        add(p+'attn_q.weight', [256, 512], 'Q4_K')
        add(p+'attn_k.weight', [256, 128], 'Q4_K')
        add(p+'attn_v.weight', [256, 128], 'Q6_K' if i else 'Q4_K')
        add(p+'attn_output.weight', [512, 256], 'Q4_K')
        add(p+'ffn_gate_inp.weight', [256, 4])
        add(p+'exp_probs_b.bias', [4])
        add(p+'ffn_gate_exps.weight', [256, 512, 4], 'Q4_K')
        add(p+'ffn_up_exps.weight', [256, 512, 4], 'Q4_K')
        add(p+'ffn_down_exps.weight', [512, 256, 4], 'Q6_K' if i else 'Q4_K')
    return meta, tensors


class MiniMaxContract(unittest.TestCase):
    def setUp(self):
        self.meta, self.tensors = fixture()

    def validate(self, meta=None, tensors=None):
        return validate_loader_contract(self.meta if meta is None else meta,
                                        self.tensors if tensors is None else tensors, allow_fixture=True)

    def test_all_moe_flattened_norms(self):
        result = self.validate()
        self.assertEqual(result['validated_tensors'], 29)
        self.assertEqual(result['families']['routed']['tensors'], 6)
        self.assertEqual(result['families']['router']['tensors'], 4)
        self.assertFalse(result['metadata']['nextn_metadata_present'])
        self.assertEqual(result['metadata']['mtp_blocks'], 0)
        norms = {t['name']: t['expected_shape'] for t in result['tensor_mapping'] if t['family'] == 'norm'}
        self.assertEqual(norms['blk.0.attn_q_norm.weight'], [512])
        self.assertEqual(norms['blk.0.attn_k_norm.weight'], [128])

    def test_fixture_not_production(self):
        with self.assertRaisesRegex(ValueError, 'production geometry'):
            validate_loader_contract(self.meta, self.tensors)

    def test_bad_metadata(self):
        cases = [('general.architecture', 'hy_v3'), ('split.count', 2), ('minimax-m2.block_count', True),
                 ('minimax-m2.expert_used_count', 5), ('minimax-m2.expert_gating_func', 1),
                 ('minimax-m2.feed_forward_length', 256), ('minimax-m2.attention.head_count_kv', 3),
                 ('minimax-m2.attention.value_length', 32), ('minimax-m2.rope.dimension_count', 65),
                 ('minimax-m2.rope.dimension_count', 128), ('minimax-m2.rope.freq_base', float('nan')),
                 ('minimax-m2.attention.layer_norm_rms_epsilon', float('inf')),
                 ('minimax-m2.nextn_predict_layers', 1), ('minimax-m2.nextn_predict_layers', 0),
                 ('minimax-m2.expert_weights_scale', 2.0), ('minimax-m2.expert_weights_norm', False),
                 ('minimax-m2.attention.sliding_window', 512), ('tokenizer.ggml.pre', 'qwen2'),
                 ('tokenizer.ggml.add_bos_token', True), ('tokenizer.ggml.eos_token_id', 16),
                 ('tokenizer.ggml.eos_token_id', True), ('tokenizer.chat_template', '')]
        for key, value in cases:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.validate({**self.meta, key: value})

    def test_missing_weights(self):
        for name in ('output.weight', 'blk.0.ffn_gate_exps.weight', 'blk.1.attn_k_norm.weight', 'blk.1.exp_probs_b.bias'):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'missing'):
                self.validate(tensors=[t for t in self.tensors if t.name != name])

    def test_extra_and_duplicate_weights(self):
        for name in ('blk.0.attn_qkv.weight', 'blk.0.ffn_up_shexp.weight', 'blk.0.ffn_gate.weight',
                     'blk.2.nextn.eh_proj.weight', 'output.weight'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.validate(tensors=self.tensors+[replace(self.tensors[0], name=name)])

    def test_shapes_types_and_bias_suffix(self):
        cases = [('blk.0.attn_q_norm.weight', {'shape': [64]}),
                 ('blk.0.attn_k_norm.weight', {'shape': [64]}),
                 ('blk.0.ffn_down_exps.weight', {'shape': [256, 512, 4]}),
                 ('blk.0.ffn_up_exps.weight', {'shape': [256, 4, 512]}),
                 ('blk.1.exp_probs_b.bias', {'name': 'blk.1.exp_probs_b'}),
                 ('blk.0.ffn_gate_inp.weight', {'type_id': 12, 'type_name': 'Q4_K'}),
                 ('blk.0.ffn_gate_exps.weight', {'type_id': 30, 'type_name': 'BF16'}),
                 ('output.weight', {'type_id': 12}), ('output.weight', {'type_id': True})]
        for name, change in cases:
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.validate(tensors=[replace(t, **change) if t.name == name else t for t in self.tensors])

    def test_bad_vocabulary(self):
        for key, value in [('tokens', ['same', 'same']), ('token_type', [1]), ('merges', ['bad']),
                           ('merges', ['a b', 'a b']), ('merges', ['token0 token1'])]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate({**self.meta, 'tokenizer.ggml.'+key: value})

    def test_ranges_above_4gib(self):
        t = TensorInfo('x', [256, 2], 14, 'Q6_K', 2**32+32)
        row = validate_ranges([t], 64, 2**32+1024, 32)[0]
        self.assertEqual((row['file_offset'], row['row_bytes'], row['bytes']), (2**32+96, 210, 420))

    def test_bad_ranges_and_quantized_rows(self):
        t = TensorInfo('x', [256, 2], 12, 'Q4_K', 0)
        cases = [([t, t], 64, 1024, 32), ([t, replace(t, name='y', offset=32)], 64, 1024, 32),
                 ([t], 64, 300, 32), ([replace(t, offset=1)], 64, 1024, 32),
                 ([replace(t, shape=[128, 4])], 64, 1024, 32),
                 ([replace(t, offset=2**63-32)], 64, 2**63-1, 32), ([t], 64, 1024, 3)]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                validate_ranges(*args)

    def test_header_and_output_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            p, out = Path(directory)/'fixture.gguf', Path(directory)/'report.json'
            write_fixture(p, self.meta, self.tensors)
            report = inspect_model(p, allow_fixture=True)
            self.assertEqual(report['tensor_count'], 29)
            self.assertEqual(report['mtp_tensor_count'], 0)
            raw = p.read_bytes()
            self.assertEqual(main(['--gguf', str(p), '--output', str(p)]), 1)
            self.assertEqual(p.read_bytes(), raw)
            self.assertEqual(main(['--gguf', str(p), '--output', str(out)]), 1)
            self.assertFalse(out.exists())
            alias = Path(directory)/'alias'
            os.link(p, alias)
            with self.assertRaises(ValueError):
                protect_output(alias, p)
            for broken in (b'GGUF', raw[:40], raw[:-1]):
                p.write_bytes(broken)
                with self.assertRaises((ValueError, OSError, struct.error)):
                    inspect_model(p, allow_fixture=True)

    def test_registration_checks_every_tensor_and_layer(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'small.gguf'
            write_fixture(p, self.meta, self.tensors)
            inv = inspect_model(p, allow_fixture=True)
        tensors = [{'name': t['name'], 'shape': t['shape']+[1]*(4-len(t['shape'])),
                    'type_id': t['type_id'], 'bytes': t['bytes']} for t in inv['tensors']]
        layers = [{'layer': i, 'swa': False, 'heads': 8, 'kv_heads': 2,
                   'key_dim': 64, 'value_dim': 64, 'rope_dim': 32} for i in range(2)]
        report = {'status': 'pass', 'architecture': 'minimax-m2', 'requested_revision': LOADER_SHA,
                  'archive_sha256': ARCHIVE_SHA256, 'patches': 'none',
                  'runs': [{'load_mtp': flag, 'main_blocks': 2, 'all_blocks': 2, 'nextn_blocks': 0,
                            'allocated_weight_bytes': 0, 'logical_bytes': inv['payload_bytes'],
                            'swa_window': 0, 'tensors': tensors, 'layers': layers} for flag in (False, True)]}
        validate_registration(report, inv)
        for error in ('allocate', 'duplicate', 'type', 'shape', 'kv', 'missing', 'mtp', 'revision'):
            bad = deepcopy(report)
            r = bad['runs'][0]
            if error == 'allocate': r['allocated_weight_bytes'] = 1
            if error == 'duplicate': r['tensors'].append(r['tensors'][0])
            if error == 'type': r['tensors'][0]['type_id'] = 8
            if error == 'shape': r['tensors'][0]['shape'][0] += 1
            if error == 'kv': r['layers'][1]['kv_heads'] = 1
            if error == 'missing': r['tensors'].pop()
            if error == 'mtp': r['nextn_blocks'] = 1
            if error == 'revision': bad['requested_revision'] = 'unreviewed'
            with self.subTest(error=error), self.assertRaises(ValueError):
                validate_registration(bad, inv)


if __name__ == '__main__':
    unittest.main()
