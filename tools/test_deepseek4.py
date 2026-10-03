"""DeepSeek model admission and tokenizer boundaries; no model/GPU required."""
from pathlib import Path
import struct
import tempfile
import unittest

from tools.setup_deepseek4 import inspect_model
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
