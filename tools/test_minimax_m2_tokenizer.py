"""MiniMax split boundaries and text/function input regressions; no real weights."""
from copy import deepcopy
import unittest
from tools.strata_tokenizer import Tokenizer, BYTE_TO_UNICODE
from tools.minimax_m2_template import text_context


class MiniMaxTokenizer(unittest.TestCase):
    def make(self, pre='minimax-m2'):
        return Tokenizer(list(BYTE_TO_UNICODE.values())+['<think>', '[e~['], [], [1]*256+[4, 3], pre=pre)

    def test_split_boundaries(self):
        tok = self.make()
        self.assertEqual(tok._re.findall('12345'), ['123', '45'])
        self.assertEqual(tok._re.findall("I'm WE'RE he'll"), ["I'm", " WE'RE", " he'll"])
        self.assertEqual(tok._re.findall('fooBar ABCdef'), ['foo', 'Bar', ' ABCdef'])
        self.assertEqual(tok._re.findall('a\u0301_b'), ['a', '\u0301_', 'b'])
        self.assertEqual(self.make('qwen2')._re.findall('12345'), list('12345'))
        self.assertEqual(self.make('qwen35')._re.findall('a\u0301_b'), ['a\u0301', '_b'])

    def test_roundtrip_and_literals(self):
        tok = self.make()
        for s in ['', '\r\n\r\x00', 'Привет 中文 🧑🏽‍💻 a\u0301 १२३', '<think>[e~[']:
            for flag in (False, True):
                self.assertEqual(tok.decode(tok.encode(s, flag)), s)
        self.assertEqual(tok.encode('<think>', False), [256])
        self.assertEqual(tok.encode('[e~[', True), [257])
        self.assertNotEqual(tok.encode('[e~[', False), [257])

    def test_merge_order_not_glm_ignore_merges(self):
        tokens = list(BYTE_TO_UNICODE.values())+['ab', 'bc', 'abc']
        tok = Tokenizer(tokens, ['b c', 'a b', 'ab c'], pre='minimax-m2')
        self.assertEqual(tok.encode('abc'), [tok.ids['a'], tok.ids['bc']])

    def test_normalize_nested_string_arguments_without_mutation(self):
        context = {'messages': [{'role': 'assistant', 'content': None, 'tool_calls': [
            {'type': 'function', 'id': 'c1', 'function': {'name': 'weather', 'arguments': '{"city":"Уфа"}'}}]},
            {'role': 'tool', 'content': [{'type': 'text', 'text': 'Ясно'}, '!']}]}
        saved = deepcopy(context)
        result = text_context(context)
        self.assertEqual(context, saved)
        self.assertEqual(result['messages'][0]['tool_calls'], [{'name': 'weather', 'arguments': {'city': 'Уфа'}}])
        self.assertEqual(result['messages'][0]['content'], '')
        self.assertEqual(result['messages'][1]['content'], 'Ясно!')

    def test_bad_arguments(self):
        for args in ('not JSON', '[]', 'null', None, [1], {1: 'value'}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                text_context({'messages': [{'role': 'assistant', 'tool_calls': [{'name': 'x', 'arguments': args}]}]})

    def test_media_and_invalid_history_rejected(self):
        for context in [
            {'messages': {}}, {'messages': [], 'enable_thinking': 'false'},
            {'messages': [{'role': 'user', 'content': [{'type': 'image', 'image_url': 'x'}]}]},
            {'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'x', 'image_url': 'x'}]}]},
            {'messages': [{'role': 'tool', 'content': 'orphan'}]},
            {'messages': [{'role': 'user', 'content': 'x'}, {'role': 'system', 'content': 'lost'}]},
            {'messages': [{'role': 'assistant', 'tool_calls': [{'type': 'custom', 'name': 'x', 'arguments': {}}]}]},
        ]:
            with self.subTest(context=context), self.assertRaises(ValueError):
                text_context(context)


if __name__ == '__main__':
    unittest.main()
