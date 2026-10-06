"""Independent tiny text-trunk fixtures and negative admission tests; no GPU/model."""
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest

from tools.gguf_reader import TensorInfo
from tools.mimo2_loader_contract import validate_loader_contract, SOURCE_SHA, REFERENCE_SHA, EXPORT_SCOPE
from tools.inspect_mimo2_gguf import inspect_model, main, protect_output, validate_ranges
from tools.check_mimo2_oracles import validate_registration, LOADER_SHA, ARCHIVE_SHA256
from tools.test_hy3_gguf import write_fixture  # Generic tiny GGUF binary writer.


def fixture():
    meta = {'general.architecture': 'mimo2', 'mimo2.block_count': 3, 'mimo2.nextn_predict_layers': 0,
            'mimo2.context_length': 1024, 'mimo2.embedding_length': 256, 'mimo2.feed_forward_length': 512,
            'mimo2.expert_feed_forward_length': 256, 'mimo2.expert_count': 4, 'mimo2.expert_used_count': 2,
            'mimo2.expert_group_count': 1, 'mimo2.expert_group_used_count': 1, 'mimo2.expert_gating_func': 2,
            'mimo2.attention.head_count': 4, 'mimo2.attention.head_count_kv': [1, 2, 1],
            'mimo2.attention.key_length': 192, 'mimo2.attention.value_length': 128,
            'mimo2.attention.sliding_window': 128, 'mimo2.attention.sliding_window_pattern': [0, 1, 0],
            'mimo2.rope.dimension_count': 64, 'mimo2.rope.freq_base': 1e7, 'mimo2.rope.freq_base_swa': 1e4,
            'mimo2.attention.layer_norm_rms_epsilon': 1e-6, 'mimo2.attention.value_scale': .707,
            'mimo2.source_revision': SOURCE_SHA, 'mimo2.converter_revision': REFERENCE_SHA,
            'mimo2.source_quantization': 'fixture', 'mimo2.reference_expert_storage': 'fixture',
            'mimo2.export_scope': EXPORT_SCOPE,
            'tokenizer.ggml.model': 'gpt2', 'tokenizer.ggml.pre': 'qwen2',
            'tokenizer.ggml.tokens': [f'token{i}' for i in range(16)], 'tokenizer.ggml.token_type': [1]*16,
            'tokenizer.ggml.merges': [], 'tokenizer.ggml.eos_token_id': 1,
            'tokenizer.ggml.padding_token_id': 2, 'tokenizer.ggml.add_bos_token': False,
            'tokenizer.chat_template': '{{ messages }}'}
    tensors, end = [], 0
    geometry = {'BF16': (30, 1, 2), 'F32': (0, 1, 4), 'Q2_K': (10, 256, 84),
                'Q3_K': (11, 256, 110), 'MXFP4': (39, 32, 17)}

    def add(name, shape, kind='F32'):
        nonlocal end
        type_id, block, size = geometry[kind]
        tensors.append(TensorInfo(name, shape, type_id, kind, end))
        elements = 1
        for n in shape:
            elements *= n
        end += elements//block*size
        end += -end % 32

    add('token_embd.weight', [256, 16], 'BF16')
    add('output.weight', [256, 16], 'BF16')
    add('output_norm.weight', [256])
    # Expected widths are independent constants for Q4*192, KV1/2*(192+128).
    for i, qkv in [(0, 1088), (1, 1408), (2, 1088)]:
        p = f'blk.{i}.'
        add(p+'attn_norm.weight', [256])
        add(p+'ffn_norm.weight', [256])
        add(p+'attn_qkv.weight', [256, qkv], 'BF16')
        add(p+'attn_output.weight', [512, 256], 'BF16')
        if i == 1:
            add(p+'attn_sinks.weight', [4])
        if i == 0:
            add(p+'ffn_gate.weight', [256, 512], 'BF16')
            add(p+'ffn_up.weight', [256, 512], 'BF16')
            add(p+'ffn_down.weight', [512, 256], 'BF16')
        else:
            add(p+'ffn_gate_inp.weight', [256, 4])
            add(p+'exp_probs_b.bias', [4])
            add(p+'ffn_gate_exps.weight', [256, 256, 4], 'Q2_K')
            add(p+'ffn_up_exps.weight', [256, 256, 4], 'Q3_K')
            add(p+'ffn_down_exps.weight', [256, 256, 4], 'MXFP4')
    return meta, tensors


class MiMoContract(unittest.TestCase):
    def setUp(self):
        self.meta, self.tensors = fixture()

    def validate(self, meta=None, tensors=None):
        return validate_loader_contract(self.meta if meta is None else meta, self.tensors if tensors is None else tensors, allow_fixture=True)

    def test_independent_fused_and_swa_layout(self):
        r = self.validate()
        self.assertEqual(r['validated_tensors'], 29)
        self.assertEqual(r['metadata']['full_layers'], [0, 2])
        self.assertEqual(r['families']['routed']['tensors'], 6)
        self.assertEqual(r['families']['sink']['tensors'], 1)
        self.assertTrue(all(t['scope'] == 'main' for t in r['tensor_mapping']))

    def test_fixture_not_production(self):
        with self.assertRaisesRegex(ValueError, 'production geometry'):
            validate_loader_contract(self.meta, self.tensors)

    def test_bad_metadata(self):
        cases = [('general.architecture', 'step35'), ('mimo2.block_count', True),
                 ('mimo2.nextn_predict_layers', 1), ('mimo2.nextn_predict_layers', False),
                 ('mimo2.expert_gating_func', 1), ('mimo2.expert_group_count', 2),
                 ('mimo2.expert_used_count', 5), ('mimo2.attention.head_count_kv', [1, 3, 1]),
                 ('mimo2.attention.head_count_kv', [1, 2]), ('mimo2.attention.sliding_window_pattern', [0, 2, 0]),
                 ('mimo2.attention.sliding_window_pattern', [False, True, False]),
                 ('mimo2.rope.dimension_count', 193), ('mimo2.rope.dimension_count', 256),
                 ('mimo2.rope.freq_base', float('nan')), ('mimo2.attention.value_scale', float('inf')),
                 ('mimo2.export_scope', 'multimodal'), ('mimo2.source_revision', 'unknown'),
                 ('mimo2.attention.qk_norm', True), ('split.count', 2),
                 ('tokenizer.ggml.pre', 'qwen35'), ('tokenizer.ggml.add_bos_token', True),
                 ('tokenizer.ggml.eos_token_id', 16), ('tokenizer.chat_template', '')]
        for key, value in cases:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.validate({**self.meta, key: value})

    def test_missing_weights(self):
        for name in ('output.weight', 'blk.0.ffn_gate.weight', 'blk.1.attn_sinks.weight',
                     'blk.1.exp_probs_b.bias', 'blk.2.ffn_down_exps.weight'):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'missing'):
                self.validate(tensors=[t for t in self.tensors if t.name != name])

    def test_extra_weights_and_duplicate(self):
        for name in ('blk.0.attn_sinks.weight', 'blk.1.ffn_up_shexp.weight',
                     'blk.3.nextn.eh_proj.weight', 'output.weight'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.validate(tensors=self.tensors+[replace(self.tensors[0], name=name)])

    def test_shapes_types_and_bias_suffix(self):
        cases = [('blk.1.attn_qkv.weight', {'shape': [256, 1088]}),
                 ('blk.0.attn_output.weight', {'shape': [768, 256]}),
                 ('blk.1.exp_probs_b.bias', {'name': 'blk.1.exp_probs_b'}),
                 ('blk.1.ffn_gate_exps.weight', {'type_name': 'MXFP4', 'type_id': 39}),
                 ('blk.1.ffn_down_exps.weight', {'type_name': 'Q2_0', 'type_id': 42}),
                 ('blk.1.ffn_down_exps.weight', {'type_id': 11}),
                 ('output.weight', {'type_name': 'Q8_0', 'type_id': 8})]
        for name, change in cases:
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.validate(tensors=[replace(t, **change) if t.name == name else t for t in self.tensors])

    def test_vocab_damage(self):
        for k, v in [('tokens', ['same', 'same']), ('token_type', [1]), ('merges', ['abc']),
                     ('merges', ['token0 token1']), ('merges', ['a b', 'a b'])]:
            with self.subTest(k=k), self.assertRaises(ValueError):
                self.validate({**self.meta, 'tokenizer.ggml.'+k: v})

    def test_ranges_above_4gib_and_mxfp4_geometry(self):
        t = TensorInfo('x', [256, 2], 39, 'MXFP4', 2**32+32)
        r = validate_ranges([t], 64, 2**32+512, 32)[0]
        self.assertEqual((r['file_offset'], r['row_bytes'], r['bytes']), (2**32+96, 136, 272))

    def test_bad_ranges(self):
        t = TensorInfo('x', [256, 2], 10, 'Q2_K', 0)
        cases = [([t, t], 64, 1024, 32), ([t, replace(t, name='y', offset=32)], 64, 1024, 32),
                 ([t], 64, 231, 32), ([replace(t, offset=1)], 64, 1024, 32),
                 ([replace(t, shape=[128, 4])], 64, 1024, 32),
                 ([replace(t, offset=2**63-32)], 64, 2**63-1, 32), ([t], 64, 1024, 3)]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                validate_ranges(*args)

    def test_header_end_to_end_and_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            p, out = Path(directory)/'fixture.gguf', Path(directory)/'report.json'
            write_fixture(p, self.meta, self.tensors)
            report = inspect_model(p, allow_fixture=True)
            self.assertEqual(report['tensor_count'], 29)
            self.assertTrue(report['fixture'])
            raw = p.read_bytes()
            self.assertEqual(main(['--gguf', str(p), '--output', str(p)]), 1)
            self.assertEqual(p.read_bytes(), raw)
            self.assertEqual(main(['--gguf', str(p), '--output', str(out)]), 1)  # CLI rejects tiny fixtures.
            self.assertFalse(out.exists())
            alias = Path(directory)/'alias'
            os.link(p, alias)
            with self.assertRaises(ValueError):
                protect_output(alias, p)
            for broken in (b'GGUF', raw[:40], raw[:-1]):
                p.write_bytes(broken)
                self.assertEqual(main(['--gguf', str(p), '--output', str(out)]), 1)
                self.assertFalse(out.exists())

    def test_registration_checks_every_tensor_and_layer(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'small.gguf'
            write_fixture(p, self.meta, self.tensors)
            inv = inspect_model(p, allow_fixture=True)
        tensors = [{'name': t['name'], 'shape': t['shape']+[1]*(4-len(t['shape'])),
                    'type_id': t['type_id'], 'bytes': t['bytes']} for t in inv['tensors']]
        layers = [{'layer': i, 'swa': i == 1, 'heads': 4, 'kv_heads': 2 if i == 1 else 1,
                   'key_dim': 192, 'value_dim': 128, 'rope_dim': 64} for i in range(3)]
        report = {'status': 'pass', 'architecture': 'mimo2', 'requested_revision': LOADER_SHA,
                  'archive_sha256': ARCHIVE_SHA256, 'patches': 'none',
                  'runs': [{'load_mtp': f, 'main_blocks': 3, 'all_blocks': 3, 'nextn_blocks': 0,
                            'allocated_weight_bytes': 0, 'logical_bytes': inv['payload_bytes'],
                            'swa_window': 128, 'tensors': tensors, 'layers': layers} for f in (False, True)]}
        validate_registration(report, inv)
        for error in ('allocate', 'duplicate', 'type', 'shape', 'kv', 'missing', 'mtp'):
            bad = deepcopy(report)
            r = bad['runs'][0]
            if error == 'allocate': r['allocated_weight_bytes'] = 1
            if error == 'duplicate': r['tensors'].append(r['tensors'][0])
            if error == 'type': r['tensors'][0]['type_id'] = 8
            if error == 'shape': r['tensors'][0]['shape'][0] += 1
            if error == 'kv': r['layers'][1]['kv_heads'] = 1
            if error == 'missing': r['tensors'].pop()
            if error == 'mtp': r['nextn_blocks'] = 1
            with self.subTest(error=error), self.assertRaises(ValueError):
                validate_registration(bad, inv)


if __name__ == '__main__':
    unittest.main()
