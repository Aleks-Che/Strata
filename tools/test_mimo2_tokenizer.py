"""Qwen2 split/merge boundaries and isolated text admission regressions."""
import unittest
from tools.strata_tokenizer import Tokenizer, BYTE_TO_UNICODE
from tools.mimo2_template import text_context


class Qwen2Tokenizer(unittest.TestCase):
    def make(self, pre='qwen2'):
        tokens = list(BYTE_TO_UNICODE.values())+['<think>', '<|im_end|>']
        return Tokenizer(tokens, [], [1]*256+[4, 3], pre=pre)

    def test_digits_and_combining_marks(self):
        tok = self.make()
        self.assertEqual(tok._re.findall('12345'), list('12345'))
        self.assertEqual(tok._re.findall('a\u0301_b'), ['a', '\u0301_', 'b'])
        self.assertEqual(self.make('qwen35')._re.findall('a\u0301_b'), ['a\u0301', '_b'])

    def test_lossless_and_special_literals(self):
        tok = self.make()
        for s in ['', '\r\n\r\x00', 'Привет 中文 🧑🏽‍💻 a\u0301 १२३', '<think><|im_end|>']:
            for flag in (False, True):
                self.assertEqual(tok.decode(tok.encode(s, flag)), s)
        self.assertEqual(tok.encode('<think>', False), [256])
        self.assertEqual(tok.encode('<|im_end|>', True), [257])
        self.assertNotEqual(tok.encode('<|im_end|>', False), [257])

    def test_no_glm_ignore_merges(self):
        tokens = list(BYTE_TO_UNICODE.values())+['ab', 'bc', 'abc']
        tok = Tokenizer(tokens, ['b c', 'a b', 'ab c'], pre='qwen2')
        self.assertEqual(tok.encode('abc'), [tok.ids['a'], tok.ids['bc']])

    def test_generic_qwen2_does_not_promote_mimo_eog(self):
        tokens = list(BYTE_TO_UNICODE.values())+['</s>']
        tok = Tokenizer(tokens, [], [1]*len(tokens), pre='qwen2')
        self.assertNotEqual(tok.encode('</s>', True), [256])
        self.assertEqual(tok.encode('</s>', True), tok.encode('</s>', False))

    def test_media_rejected_before_template(self):
        for part in [{'type': 'image', 'image_url': 'x'}, {'type': 'input_audio', 'input_audio': {}},
                     {'type': 'video', 'video': 'x'}, {'type': 'text', 'text': 'x', 'image_url': 'x'}]:
            with self.subTest(part=part), self.assertRaisesRegex(ValueError, 'text parts only'):
                text_context({'messages': [{'role': 'user', 'content': [part]}]})

    def test_context_types_and_copy(self):
        original = {'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'Привет'}]}]}
        result = text_context(original)
        result['messages'][0]['content'][0]['text'] = 'changed'
        self.assertEqual(original['messages'][0]['content'][0]['text'], 'Привет')
        for context in [{'messages': {}}, {'messages': [], 'enable_thinking': 'false'},
                        {'messages': [{'role': 'unknown', 'content': ''}]},
                        {'messages': [{'role': 'user', 'content': {'text': 'wrong'}}]}]:
            with self.assertRaises(ValueError):
                text_context(context)


if __name__ == '__main__':
    unittest.main()
