"""GLM4 split/BPE fixtures; no weights, GPU or tokenizer oracle required."""
import json
from pathlib import Path
import tempfile
import unittest

from tools.strata_tokenizer import BYTE_TO_UNICODE, GLM4_PATTERN, Tokenizer, extract
from tools.test_setup_glm5next import write_shard
from tools.glm5next_tokenizer_corpus import CORPUS


def fixture(pre='glm4'):
    tokens = list(BYTE_TO_UNICODE.values())
    merges = []
    # Include cross-boundary merges: wrong pre-tokenization changes IDs while
    # still giving a perfect round trip. Expectations below name the pieces.
    for word in ('123456', '456', '789', 'a\u0301', '_b', '\u0301_'):
        mapped = ''.join(BYTE_TO_UNICODE[b] for b in word.encode())
        for n in range(2, len(mapped) + 1):
            token = mapped[:n]
            if token not in tokens:
                tokens.append(token)
                merges.append(mapped[:n-1] + ' ' + mapped[n-1])
    types = [1] * len(tokens)
    for token, ty in (('[gMASK]', 3), ('<sop>', 3), ('<think>', 4), ('<|user|>', 3)):
        tokens.append(token)
        types.append(ty)
    return Tokenizer(tokens, merges, types, pre=pre,
                     special_ids={'tokenizer.ggml.eot_token_id': tokens.index('<|user|>')})


class GLMTokenizerTests(unittest.TestCase):
    def pieces(self, tok, text):
        return [tok.token_bytes(i).decode() for i in tok.encode(text)]

    def test_digits_do_not_merge_across_three_digit_boundary(self):
        tok = fixture()
        self.assertEqual(self.pieces(tok, '1234567890'), ['123', '456', '789', '0'])
        self.assertEqual(tok._re.findall('١٢٣٤٥٦٧ １２３４５６７'),
                         ['١٢٣', '٤٥٦', '٧', ' ', '１２３', '４５６', '７'])

    def test_combining_marks_are_not_qwen_letters(self):
        tok = fixture()
        self.assertEqual(self.pieces(tok, 'a\u0301_b'), ['a', '\u0301_', 'b'])
        self.assertEqual(tok._re.findall('e\u0301clair'), ['e', '\u0301clair'])

    def test_contractions_whitespace_and_multilingual_boundaries(self):
        tok = fixture()
        for text, expected in [
            (" I'M we'RE can't", [' I', "'M", ' we', "'RE", ' can', "'t"]),
            ('x\r\n  y  ', ['x', '\r\n', ' ', ' y', '  ']),
            ('Привет, мир!你好', ['Привет', ',', ' мир', '!你好']),
        ]:
            with self.subTest(text=text):
                self.assertEqual(tok._re.findall(text), expected)

    def test_special_token_types_and_no_automatic_bos(self):
        tok = fixture()
        self.assertEqual(tok.encode('', parse_special=True), [])
        self.assertEqual(tok.encode('[gMASK]<sop><think>', parse_special=True),
                         [tok.ids[t] for t in ('[gMASK]', '<sop>', '<think>')])
        ids = tok.encode('[gMASK]<sop><think>', parse_special=False)
        self.assertNotIn(tok.ids['[gMASK]'], ids)
        self.assertNotIn(tok.ids['<sop>'], ids)
        self.assertIn(tok.ids['<think>'], ids)

    def test_fixture_corpus_round_trip(self):
        tok = fixture()
        for text in CORPUS:
            for special in (False, True):
                with self.subTest(text=text[:50], special=special):
                    self.assertEqual(tok.decode(tok.encode(text, parse_special=special)), text)

    def test_qwen_boundaries_remain_distinct_and_unknown_pre_rejected(self):
        qwen = fixture('qwen35')
        self.assertEqual(self.pieces(qwen, '123456'), list('123456'))
        self.assertEqual(self.pieces(qwen, 'a\u0301_b'), ['a\u0301', '_b'])
        with self.assertRaisesRegex(ValueError, 'unsupported pre-tokenizer'):
            fixture('unknown-glm')

    def test_metadata_only_gguf_load_and_pack_export(self):
        original = fixture()
        metadata = {'general.architecture': 'glm5next', 'tokenizer.ggml.model': 'gpt2',
                    'tokenizer.ggml.pre': 'glm4', 'tokenizer.ggml.tokens': original.tokens,
                    'tokenizer.ggml.merges': [' '.join(pair) for pair in original.ranks],
                    'tokenizer.ggml.token_type': original.token_types,
                    'tokenizer.chat_template': '[gMASK]<sop>{{ messages }}', **original.special_ids}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gguf = root / 'metadata.gguf'
            write_shard(gguf, metadata, [])
            loaded = Tokenizer.from_gguf(gguf)
            self.assertEqual(loaded.pre, 'glm4')
            self.assertEqual(loaded.special_ids, original.special_ids)
            for text in CORPUS:
                self.assertEqual(loaded.encode(text, True), original.encode(text, True))
            cfg = extract(gguf, root / 'pack')
            saved = json.loads((root / 'pack/tokenizer/tokenizer.json').read_text(encoding='utf-8'))
            self.assertEqual(saved, cfg)
            self.assertEqual(saved['pre_pattern'], GLM4_PATTERN)
            self.assertFalse(saved['add_bos_token'])
            self.assertEqual((root / 'pack/tokenizer/chat_template.jinja').read_text(encoding='utf-8'),
                             metadata['tokenizer.chat_template'])


if __name__ == '__main__':
    unittest.main()
