"""DeepSeek model admission and tokenizer boundaries; no model/GPU required."""
from pathlib import Path
import struct
import tempfile
import unittest

from tools.setup_deepseek4 import inspect_model, inspect_draft
from tools.strata_tokenizer import Tokenizer, bytes_to_unicode


def string(value):
    b = value.encode()
    return struct.pack('<Q', len(b)) + b


def shard(path, number=0, architecture='deepseek4', count=2, tensor=True, truncated=False):
    metadata = {'split.count': count, 'split.no': number, 'split.tensors.count': 1}
    if number == 0:
        metadata['general.architecture'] = architecture
    payload = b'GGUF' + struct.pack('<IQQ', 3, int(tensor), len(metadata))
    for key, value in metadata.items():
        payload += string(key)
        payload += struct.pack('<I', 8) + string(value) if isinstance(value, str) else struct.pack('<II', 4, value)
    if tensor:
        payload += string('blk.0.ffn_gate_tid2eid.weight') + struct.pack('<IQQIQ', 2, 2, 2, 26, 0)
    payload += bytes((-len(payload)) % 32)
    if tensor and not truncated:
        payload += bytes(16)
    path.write_bytes(payload)


class ModelAdmissionTests(unittest.TestCase):
    def test_metadata_only_first_and_integer_hash_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            a = Path(directory) / 'test-00001-of-00002.gguf'
            b = a.with_name('test-00002-of-00002.gguf')
            shard(a, tensor=False)
            shard(b, number=1)
            report = inspect_model(a)
            self.assertEqual((report['shards'], report['tensors'], report['dense_bytes']), (2, 1, 16))

    def test_missing_or_incomplete_shards_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            a = Path(directory) / 'test-00001-of-00002.gguf'
            b = a.with_name('test-00002-of-00002.gguf')
            shard(a, tensor=False)
            with self.assertRaises(FileNotFoundError):
                inspect_model(a)
            shard(b, number=1, truncated=True)
            with self.assertRaisesRegex(ValueError, 'Incomplete shard'):
                inspect_model(a)
            shard(b, number=0)
            with self.assertRaisesRegex(ValueError, 'Wrong shard'):
                inspect_model(a)

    def test_qwen_is_not_admitted_to_deepseek_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'qwen.gguf'
            shard(path, architecture='qwen4exp', count=1)
            with self.assertRaisesRegex(ValueError, 'existing Qwen profile'):
                inspect_model(path)


class DraftAdmissionTests(unittest.TestCase):
    def write_draft(self, path, overrides=None, truncated=False, missing_head=False):
        meta = {'general.architecture': 'dflash', 'dflash.block_count': 3,
                'dflash.embedding_length': 4096, 'dflash.hyper_connection.count': 4,
                'dflash.block_size': 5, 'dflash.target_layers': [41, 42, 43]}
        meta.update(overrides or {})
        names = ['markov_w1.weight', 'markov_w2.weight', 'blk.0.ffn_up_exps.weight']
        if missing_head:
            names.remove('markov_w2.weight')
        data = b'GGUF' + struct.pack('<IQQ', 3, len(names), len(meta))
        for key, value in meta.items():
            data += string(key)
            if isinstance(value, str):
                data += struct.pack('<I', 8) + string(value)
            elif isinstance(value, list):
                data += struct.pack('<IIQ', 9, 4, len(value)) + struct.pack(f'<{len(value)}I', *value)
            elif isinstance(value, bool):
                data += struct.pack('<I?', 7, value)
            else:
                data += struct.pack('<II', 4, value)
        for i, name in enumerate(names):
            data += string(name) + struct.pack('<IQIQ', 1, 8, 0, i * 32)
        data += bytes((-len(data)) % 32) + bytes(32 * len(names) - int(truncated))
        path.write_bytes(data)

    def test_admits_matching_complete_sidecar_and_counts_experts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'draft.gguf'
            self.write_draft(path)
            report = inspect_draft(path)
            self.assertEqual((report['tensors'], report['weight_bytes'], report['expert_bytes']), (3, 96, 32))

    def test_rejects_wrong_target_layers_architecture_and_attention(self):
        for override in ({'dflash.target_layers': [40, 41, 42]}, {'general.architecture': 'qwen4exp'},
                         {'dflash.attention.causal': True}, {'dflash.sample_from_anchor': False}):
            with self.subTest(override=override), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'draft.gguf'
                self.write_draft(path, override)
                with self.assertRaises(ValueError):
                    inspect_draft(path)

    def test_rejects_incomplete_copy_and_missing_markov(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'draft.gguf'
            self.write_draft(path, truncated=True)
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                inspect_draft(path)
            self.write_draft(path, missing_head=True)
            with self.assertRaisesRegex(ValueError, 'Markov'):
                inspect_draft(path)


class TokenizerTests(unittest.TestCase):
    def make_tokenizer(self, pre):
        tokens = list(bytes_to_unicode().values()) + ['<｜end▁of▁sentence｜>']
        return Tokenizer(tokens, [], [1] * 256 + [3], pre=pre,
                         special_ids={'tokenizer.ggml.eos_token_id': 256})

    def test_unicode_special_tokens_decode_as_literals(self):
        tok = self.make_tokenizer('joyai-llm')
        text = 'Привет 世界 1234567890\n\t😀<｜end▁of▁sentence｜>'
        self.assertEqual(tok.decode(tok.encode(text, True)), text)
        self.assertEqual(tok.encode('<｜end▁of▁sentence｜>', True), [256])
        self.assertNotEqual(tok.encode('<｜end▁of▁sentence｜>'), [256])

    def test_qwen_roundtrip_unchanged(self):
        tok = self.make_tokenizer('qwen35')
        text = "café e\u0301 a\u0308 Привет 😀 123\n\n"
        self.assertEqual(tok.decode(tok.encode(text)), text)

    def test_unknown_pre_tokenizer_is_not_silently_qwen(self):
        with self.assertRaisesRegex(ValueError, 'unsupported pre-tokenizer'):
            self.make_tokenizer('unknown')


if __name__ == '__main__':
    unittest.main()
