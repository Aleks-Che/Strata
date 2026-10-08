"""Unregistered MiniMax reasoning parser: chunk/UTF-8 boundaries and truncation."""
import codecs
import unittest

from serve.minimax_m2 import MiniMaxReasoningParser


def signature(events):
    result = []
    for event in events:
        if result and result[-1][0] == event.kind:
            result[-1] = (event.kind, result[-1][1]+event.text)
        else:
            result.append((event.kind, event.text))
    return result


class MiniMaxReasoningTests(unittest.TestCase):
    def test_all_character_boundaries(self):
        text = 'Думаю 中文 🧑🏽‍💻\n</think>\n\nОтвет 42.\n'
        expected = [('reasoning', 'Думаю 中文 🧑🏽‍💻\n'), ('content', '\n\nОтвет 42.\n')]
        for split in range(len(text)+1):
            with self.subTest(split=split):
                parser = MiniMaxReasoningParser()
                events = parser.feed(text[:split])+parser.feed(text[split:])+parser.finish()
                self.assertEqual(signature(events), expected)
                self.assertTrue(parser.reasoning_complete)

    def test_all_utf8_byte_boundaries(self):
        raw = 'Разбор e\u0301 中文 🧑🏽‍💻</think>\nГотово ✅'.encode('utf-8')
        expected = [('reasoning', 'Разбор e\u0301 中文 🧑🏽‍💻'), ('content', '\nГотово ✅')]
        chunks = [[raw[:i], raw[i:]] for i in range(len(raw)+1)]
        chunks.append([raw[i:i+1] for i in range(len(raw))])
        for parts in chunks:
            decoder = codecs.getincrementaldecoder('utf-8')('strict')
            parser = MiniMaxReasoningParser()
            events = []
            for part in parts:
                events.extend(parser.feed(decoder.decode(part)))
            events.extend(parser.feed(decoder.decode(b'', final=True)))
            events.extend(parser.finish())
            self.assertEqual(signature(events), expected)

    def test_each_truncated_marker_preserved_as_reasoning(self):
        for length in range(len('</think>')):
            text = 'unfinished\n'+'</think>'[:length]
            parser = MiniMaxReasoningParser()
            events = []
            for char in text:
                events.extend(parser.feed(char))
            events.extend(parser.finish())
            self.assertEqual(signature(events), [('reasoning', text)])
            self.assertFalse(parser.reasoning_complete)

    def test_only_first_closing_marker_is_control(self):
        parser = MiniMaxReasoningParser()
        literal = 'Example: </think> [e~[ <fim_pad> <reponame> <minimax:tool_call><invoke name="x">'
        events = parser.feed('</think>'+literal)+parser.finish()
        self.assertEqual(signature(events), [('content', literal)])
        self.assertTrue(parser.reasoning_complete)

    def test_false_prefixes_and_bounded_buffer(self):
        parser = MiniMaxReasoningParser()
        text = 'a'*10000+'</thinX> < </thinkX> </thi'
        events = []
        for i in range(0, len(text), 17):
            events.extend(parser.feed(text[i:i+17]))
            self.assertLess(len(parser._pending), len(parser.END))
        events.extend(parser.finish())
        self.assertEqual(signature(events), [('reasoning', text)])

    def test_empty_closed_reasoning_and_lifecycle(self):
        parser = MiniMaxReasoningParser()
        self.assertEqual(parser.feed(''), [])
        self.assertEqual(parser.feed('</think>'), [])
        self.assertTrue(parser.reasoning_complete)
        self.assertEqual(parser.finish(), [])
        self.assertEqual(parser.finish(), [])
        with self.assertRaises(ValueError):
            parser.feed('late')
        with self.assertRaises(TypeError):
            MiniMaxReasoningParser().feed(b'\xd0')

    def test_incomplete_utf8_is_not_silently_replaced(self):
        decoder = codecs.getincrementaldecoder('utf-8')('strict')
        self.assertEqual(decoder.decode(b'\xd0'), '')
        with self.assertRaises(UnicodeDecodeError):
            decoder.decode(b'', final=True)


if __name__ == '__main__':
    unittest.main()
