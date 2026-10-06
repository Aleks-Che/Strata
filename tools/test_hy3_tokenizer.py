"""Hy3 tokenizer boundaries and the GGUF's missing CR byte; no model or GPU."""
import unittest

from tools.strata_tokenizer import BYTE_TO_UNICODE, Tokenizer
from tools.check_hy3_tokenizer import validate_provenance
from tools.hy3_loader_contract import ARCHIVE_SHA256, LOADER_SHA


def fixture(pre='hunyuan-dense', cr=False):
    tokens = [s for b, s in BYTE_TO_UNICODE.items() if cr or b != 13]
    merges = []
    for word in ('123456', '456', '中文abc', 'abc中文', 'a\u0301', '!ABC'):
        mapped = ''.join(BYTE_TO_UNICODE[b] for b in word.encode())
        for n in range(2, len(mapped) + 1):
            if mapped[:n] not in tokens:
                tokens.append(mapped[:n])
                merges.append(mapped[:n - 1] + ' ' + mapped[n - 1])
    kinds = [1] * len(tokens)
    tokens += ['<｜hy_eos:opensource｜>', '<think:opensource>']
    kinds += [3, 4]
    return Tokenizer(tokens, merges, kinds, pre=pre)


class Hy3Tokenizer(unittest.TestCase):
    def test_boundaries(self):
        tk = fixture()
        pieces = lambda s: [tk.token_bytes(i).decode() for i in tk.encode(s)]
        self.assertEqual(pieces('123456'), ['123', '456'])
        self.assertEqual(pieces('中文abc'), ['中文', 'abc'])
        self.assertEqual(pieces('a\u0301'), ['a\u0301'])
        self.assertEqual(pieces('!ABC'), ['!ABC'])

    def test_missing_cr_and_other_profiles(self):
        tk = fixture()
        self.assertEqual(tk.encode('\r'), [])
        self.assertEqual(tk.decode(tk.encode('a\r\nb\rc')), 'a\nbc')
        present = fixture(cr=True)
        self.assertEqual(present.decode(present.encode('\r')), '\r')
        for pre in ('qwen35', 'deepseek-v3', 'joyai-llm', 'glm4'):
            with self.subTest(pre=pre), self.assertRaises(KeyError):
                fixture(pre).encode('\r')

    def test_controls_user_tokens_and_no_duplicate_bos(self):
        tk = fixture()
        eos, think = '<｜hy_eos:opensource｜>', '<think:opensource>'
        self.assertEqual(tk.encode(eos, True), [tk.ids[eos]])
        self.assertNotEqual(tk.encode(eos, False), [tk.ids[eos]])
        self.assertEqual(tk.encode(think, False), [tk.ids[think]])
        self.assertEqual(tk.encode(''), [])

    def test_qwen_digits_unchanged(self):
        tk = fixture('qwen35', cr=True)
        self.assertEqual([tk.token_bytes(i) for i in tk.encode('123456')], list(b'123456'[i:i+1] for i in range(6)))

    def test_does_not_enable_glm_ignore_merges(self):
        tokens = list(BYTE_TO_UNICODE.values()) + ['ab', 'bc', 'abc']
        tk = Tokenizer(tokens, ['b c', 'a b', 'ab c'], pre='hunyuan-dense')
        self.assertEqual(tk.encode('abc'), [tk.ids['a'], tk.ids['bc']])

    def test_provenance(self):
        valid = {'architecture': 'hy_v3', 'requested_revision': LOADER_SHA,
                 'archive_sha256': ARCHIVE_SHA256, 'patches': 'none'}
        validate_provenance(valid)
        for bad in ({}, {**valid, 'architecture': 'step35'}, {**valid, 'patches': 'unknown'},
                    {**valid, 'requested_revision': 'main'}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_provenance(bad)


if __name__ == '__main__':
    unittest.main()
