"""Generated MiniMax groups: atomicity, schema/bytes, truncation and stop ordering."""
import builtins
import codecs
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from serve.frontend import TemplateRequestError
from serve.minimax_m2_history import MiniMaxChatTemplate, prepare_minimax_context
from serve.minimax_m2_tools import GROUP_END, MiniMaxOutputParser
from tools.minimax_m2_tool_cases import BODY, EXPECTED, TOOLS, group, invalid_groups, text_group


def signature(events):
    result = []
    for event in events:
        if event.kind == 'tool_call':
            result.append(('tool_call', event.call.name, event.call.arguments))
        elif result and result[-1][0] == event.kind:
            result[-1] = (event.kind, result[-1][1]+event.text)
        elif event.text:
            result.append((event.kind, event.text))
    return result


def parse(parts, **kwargs):
    parser = MiniMaxOutputParser(tools=kwargs.pop('tools', TOOLS), **kwargs)
    events = []
    for part in parts:
        events.extend(parser.feed(part))
    events.extend(parser.finish())
    return parser, events


def definition(properties, **kwargs):
    return [{'type': 'function', 'function': {'name': 'f', 'parameters': {
        'type': 'object', 'properties': properties, **kwargs}}}]


class MiniMaxToolsTests(unittest.TestCase):
    def test_all_character_splits_and_atomic_group(self):
        wire = 'Разбор</think>\n'+group(BODY)+'\nDone'
        expected = [('reasoning', 'Разбор'), ('content', '\n')]
        expected += [('tool_call', name, args) for name, args in EXPECTED]+[('content', '\nDone')]
        for split in range(len(wire)+1):
            with self.subTest(split=split):
                parser, events = parse([wire[:split], wire[split:]])
                self.assertEqual(signature(events), expected)
                self.assertTrue(parser.reasoning_complete)
                self.assertEqual(parser.buffered_chars, 0)
        parser = MiniMaxOutputParser(tools=TOOLS)
        self.assertFalse(any(e.kind == 'tool_call' for e in parser.feed('</think>'+group(BODY)[:-1])))
        calls = parser.feed('>')
        self.assertEqual([e.call.name for e in calls], ['typed', 'ping'])
        self.assertEqual(len({e.call.id for e in calls}), 2)

    def test_every_utf8_split_and_bytewise(self):
        raw = ('想</think>'+text_group('Уфа e\u0301 中文 🧑🏽‍💻')).encode('utf-8')
        expected = [('reasoning', '想'), ('tool_call', 'text', {'s': 'Уфа e\u0301 中文 🧑🏽‍💻'})]
        chunks = [[raw[:i], raw[i:]] for i in range(len(raw)+1)]+[[raw[i:i+1] for i in range(len(raw))]]
        for parts in chunks:
            decoder = codecs.getincrementaldecoder('utf-8')('strict')
            decoded = [decoder.decode(part) for part in parts]+[decoder.decode(b'', final=True)]
            self.assertEqual(signature(parse(decoded)[1]), expected)
        decoder = codecs.getincrementaldecoder('utf-8')('strict')
        decoder.decode(b'\xf0\x9f')
        with self.assertRaises(UnicodeDecodeError):
            decoder.decode(b'', final=True)

    def test_invalid_groups_are_literal_and_disable_recovery(self):
        for name, wire in invalid_groups():
            for parts in [[wire], list(wire)]:
                with self.subTest(name=name, parts=len(parts)):
                    parser, events = parse(['</think>', *parts, group(BODY)])
                    self.assertEqual(signature(events), [('content', wire+group(BODY))])
                    self.assertIsNotNone(parser.tool_error)

    def test_all_group_truncations_preserve_text(self):
        wire = group(BODY)
        for end in range(len(wire)):
            _, events = parse(['R</think>', wire[:end]])
            expected = [('reasoning', 'R')]+([('content', wire[:end])] if end else [])
            self.assertEqual(signature(events), expected)

    def test_reasoning_tools_and_second_think_are_literal(self):
        _, events = parse([group(BODY)+'</think><think>literal</think>'])
        self.assertEqual(signature(events), [('reasoning', group(BODY)), ('content', '<think>literal</think>')])

    def test_responses_and_eos_spellings_are_literal(self):
        text = '<response>answer</response>[e~[<fim_pad><reponame></invoke>'
        self.assertEqual(signature(parse(['</think>'+text])[1]), [('content', text)])

    def test_no_tools_disables_controls(self):
        text = group(BODY)
        self.assertEqual(signature(parse(['</think>'+text], tools=[])[1]), [('content', text)])

    def test_raw_strings_never_unquote_unescape_or_trim(self):
        for value in ['', 'true', 'null', '42', 'NaN', '"abc"', '{"x":1}', '  x\r\n',
                      '&quot;&amp;&lt;/invoke&gt;', '<xml>hello</xml>', '\\n\\u003c/response>']:
            self.assertEqual(signature(parse(['</think>'+text_group(value)])[1]), [('tool_call', 'text', {'s': value})])

    def test_string_union_ambiguity_is_not_guessed(self):
        tools = definition({'s': {'type': ['string', 'null', 'integer']}})
        for value in ['null', '42']:
            wire = text_group(value).replace('name="text"', 'name="f"')
            self.assertEqual(signature(parse(['</think>'+wire], tools=tools)[1]), [('content', wire)])
        wire = text_group('abc').replace('name="text"', 'name="f"')
        self.assertEqual(signature(parse(['</think>'+wire], tools=tools)[1]), [('tool_call', 'f', {'s': 'abc'})])
        tools = definition({'s': {'type': ['string', 'number']}})
        for value in ['NaN', '1e400', '9223372036854775808']:
            wire = text_group(value).replace('name="text"', 'name="f"')
            self.assertEqual(signature(parse(['</think>'+wire], tools=tools)[1]), [('tool_call', 'f', {'s': value})])

    def test_full_schema_constraints_and_local_refs(self):
        tools = definition({'n': {'type': 'number', 'minimum': 1, 'maximum': 3},
                            's': {'type': 'string', 'enum': ['yes']},
                            'o': {'type': 'object', '$ref': '#/$defs/entry'}},
                           required=['n', 's', 'o'], additionalProperties=False,
                           **{'$defs': {'entry': {'type': 'object', 'properties': {'k': {'const': 7}}, 'required': ['k']}}})
        body = '<invoke name="f"><parameter name="n">2</parameter><parameter name="s">yes</parameter><parameter name="o">{"k":7}</parameter></invoke>'
        self.assertEqual(signature(parse(['</think>'+group(body)], tools=tools)[1]),
                         [('tool_call', 'f', {'n': 2, 's': 'yes', 'o': {'k': 7}})])
        for before, after in [('>2<', '>4<'), ('>yes<', '>no<'), ('{"k":7}', '{"k":8}')]:
            wire = group(body.replace(before, after))
            self.assertEqual(signature(parse(['</think>'+wire], tools=tools)[1]), [('content', wire)])

    def test_schema_errors_and_unsupported_wire_type(self):
        for tools in [{}, 'bad', definition({'s': {}}), definition({'s': True}),
                      definition({'s': {'type': 'unknown'}}), definition({'s': {'type': 'string', '$ref': 'https://example.invalid/schema'}}),
                      definition({}, **{'$schema': 'https://example.invalid/draft'})]:
            with self.subTest(tools=tools), self.assertRaises(TemplateRequestError):
                MiniMaxOutputParser(tools=tools)
        tools = definition({'s': {'type': 'string', '$ref': '#/$defs/missing'}})
        wire = text_group('x').replace('name="text"', 'name="f"')
        self.assertEqual(signature(parse(['</think>'+wire], tools=tools)[1]), [('content', wire)])

    def test_dependency_required_only_with_tools(self):
        original = builtins.__import__
        def no_schema(name, *args, **kwargs):
            if name == 'jsonschema':
                raise ImportError('fixture')
            return original(name, *args, **kwargs)
        with patch('builtins.__import__', no_schema):
            MiniMaxOutputParser()
            with self.assertRaises(TemplateRequestError):
                MiniMaxOutputParser(tools=TOOLS)

    def test_group_limit_is_chunk_invariant(self):
        wire = text_group('x'*10000)
        for size in [1, 7, 4096, len(wire)]:
            parser = MiniMaxOutputParser(tools=TOOLS, max_group_chars=256)
            events = parser.feed('</think>')
            for i in range(0, len(wire), size):
                events += parser.feed(wire[i:i+size])
                self.assertLessEqual(parser.buffered_chars, 256)
            events += parser.feed(group(BODY))+parser.finish()
            self.assertEqual(signature(events), [('content', wire+group(BODY))])
            self.assertIsNotNone(parser.tool_error)
        wire = group(BODY)
        for size in [1, len(wire)]:
            _, events = parse(['</think>']+[wire[i:i+size] for i in range(0, len(wire), size)], max_group_chars=len(wire))
            self.assertEqual(signature(events), [('tool_call', name, args) for name, args in EXPECTED])

    def test_stop_inside_tools_never_emits_partial_group(self):
        wire = group(BODY)
        for stop in ['</parameter>', '</invoke>', GROUP_END, '🧑🏽‍💻']:
            raw = 'R</think>'+wire+'ignored'
            expected = [('reasoning', 'R'), ('content', wire[:wire.index(stop)])]
            for split in range(len(raw)+1):
                parser, events = parse([raw[:split], raw[split:]], stops=[stop])
                self.assertEqual(signature(events), expected)
                self.assertEqual(parser.stop_sequence, stop)

    def test_overlapping_stops_earliest_completion(self):
        for stops, wire, expected, stop in [
            (['abcd', 'bc'], 'abcde', 'a', 'bc'), (['bc', 'abc'], 'abc', 'a', 'bc'),
            (['abc', 'bc'], 'abc', '', 'abc'), (['</think>'], 'R</think>x', 'R', '</think>'),
        ]:
            for split in range(len(wire)+1):
                parser, events = parse([wire[:split], wire[split:]], stops=stops)
                self.assertEqual(signature(events), [('reasoning', expected)] if expected else [])
                self.assertEqual(parser.stop_sequence, stop)
        parser, events = parse(['R</thi'], stops=['</think>'])
        self.assertEqual(signature(events), [('reasoning', 'R</thi')])
        self.assertFalse(parser.reasoning_complete)

    def test_tools_then_stop_and_separate_groups(self):
        raw = '</think>'+group(BODY)+' mid '+text_group('x')+'STOPsuffix'
        expected = [('tool_call', name, args) for name, args in EXPECTED]+[('content', ' mid '), ('tool_call', 'text', {'s': 'x'})]
        for parts in [[raw], list(raw)]:
            self.assertEqual(signature(parse(parts, stops=['STOP'])[1]), expected)

    def test_finish_and_argument_contract(self):
        parser = MiniMaxOutputParser(tools=TOOLS)
        with self.assertRaises(TypeError):
            parser.feed(b'bytes')
        self.assertEqual(parser.feed(''), [])
        self.assertEqual(parser.finish(), [])
        self.assertEqual(parser.finish(), [])
        with self.assertRaises(ValueError):
            parser.feed('late')
        for stops in [1, [''], [1], ()]:
            with self.assertRaises(ValueError):
                MiniMaxOutputParser(stops=stops)
        for limit in [0, 20, True, 1.5]:
            with self.assertRaises(ValueError):
                MiniMaxOutputParser(max_group_chars=limit)

    def test_call_ids_to_history_results_and_template(self):
        tools = deepcopy(TOOLS)
        parser = MiniMaxOutputParser(tools=tools)
        tools.clear()  # Request mutations must not change the parser's snapshot.
        calls = [e.call for e in parser.feed('</think>'+group(BODY)) if e.call]
        request = {'messages': [{'role': 'user', 'content': 'Q'}, {'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': c.id, 'function': {'name': c.name, 'arguments': json.dumps(c.arguments, ensure_ascii=False)}} for c in calls]},
            *[{'role': 'tool', 'tool_call_id': c.id, 'content': 'result-'+str(i)} for i, c in reversed(list(enumerate(calls)))]],
            'tools': TOOLS}
        context = prepare_minimax_context(request)
        self.assertEqual([m['content'] for m in context['messages'][-2:]], ['result-0', 'result-1'])
        source = (Path(__file__).parent/'fixtures/minimax_m27_chat_template.jinja').read_bytes().decode('utf-8')
        rendered = MiniMaxChatTemplate(source).render(request)
        self.assertIn('<response>result-0</response>\n<response>result-1</response>', rendered)
        self.assertTrue(rendered.endswith(']~b]ai\n<think>\n'))


if __name__ == '__main__':
    unittest.main()
