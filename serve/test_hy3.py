"""Hy3 template/parser admission; CPU only, no HTTP listener or tool execution."""
import codecs
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from serve.frontend import TemplateRequestError
from serve.hy3 import (Hy3Template, Hy3OutputParser, Hy3StopParser, TEMPLATE_SHA256,
    THINK_START, THINK_END, GROUP_START, GROUP_END, CALL_START, CALL_END,
    SEPARATOR, KEY_START, KEY_END, VALUE_START, VALUE_END, stop_sequences)
from tools.check_hy3_template import corpus

FIXTURE = Path(__file__).parent/'fixtures/hy3_chat_template.jinja'
TOOLS = [{'type': 'function', 'function': {'name': 'f', 'parameters': {'properties': {
    's': {'type': 'string'}, 'n': {'type': 'integer'}, 'b': {'type': 'boolean'},
    'nil': {'type': 'null'}, 'obj': {'type': 'object'}, 'arr': {'type': 'array'},
    'num': {'type': 'number'}, 'maybe': {'type': ['string', 'null']}}}}}]
FRAGMENTATION_SEQUENCES = 0


def call(args=None, name='f'):
    # Same visible framing as the GGUF; expectations below are typed values,
    # literal strings or saved native oracle hashes, not parser-generated data.
    text = CALL_START+name+SEPARATOR+'\n'
    for key, value in (args or {}).items():
        text += KEY_START+key+KEY_END+'\n'+VALUE_START+value+VALUE_END+'\n'
    return text+CALL_END+'\n'


def group(body):
    return GROUP_START+'\n'+body+GROUP_END


def signature(events):
    result = []
    for event in events:
        if event.kind == 'tool_call':
            result.append(('tool_call', event.call.name, event.call.arguments))
        elif event.text:
            if result and result[-1][0] == event.kind:
                result[-1] = (event.kind, result[-1][1]+event.text)
            else:
                result.append((event.kind, event.text))
    return result


class TemplateTests(unittest.TestCase):
    def test_fixture_hash_manual_prefix_and_efforts(self):
        self.assertEqual(hashlib.sha256(FIXTURE.read_bytes()).hexdigest(), TEMPLATE_SHA256)
        template = Hy3Template(FIXTURE)
        for effort in ('no_think', 'low', 'high'):
            expected = ('<｜hy_begin_of_sentence:opensource｜><｜reasoning_mode:opensource｜>reasoning_effort:'+effort+
                        '<｜hy_User:opensource｜>Привет, 你好!<｜hy_Assistant:opensource｜>'+THINK_START)
            if effort == 'no_think':
                expected += THINK_END
            self.assertEqual(template.render([{'role': 'user', 'content': 'Привет, 你好!'}], reasoning_effort=effort), expected)

    def test_saved_native_oracle_corpus(self):
        template = Hy3Template(FIXTURE)
        report = json.loads((FIXTURE.parents[2]/'docs/hy3/HY3_TEMPLATE_VALIDATION.json').read_text(encoding='utf8'))
        native = {c['name']: c for c in report['cases']}
        for case in corpus():
            with self.subTest(case=case['name']):
                before = deepcopy(case['context'])
                if case.get('error'):
                    with self.assertRaises(ValueError):
                        template.render(**case['context'])
                else:
                    rendered = template.render(**case['context'])
                    self.assertEqual(hashlib.sha256(rendered.encode()).hexdigest(), native[case['name']]['render_sha256'])
                self.assertEqual(case['context'], before)

    def test_changed_template_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'template.jinja'
            path.write_bytes(FIXTURE.read_bytes()+b' ')
            with self.assertRaises(TemplateRequestError):
                Hy3Template(path)

    def test_invalid_requests_and_ambiguous_raw_values(self):
        template = Hy3Template(FIXTURE)
        for arguments in ('[]', '{"a":1,"a":2}', '{"x":NaN}', {'x': float('inf')},
                          {'s': VALUE_END+'\n'+CALL_END}):
            with self.subTest(arguments=arguments), self.assertRaises(TemplateRequestError):
                template.render([{'role': 'assistant', 'tool_calls': [
                    {'function': {'name': 'f', 'arguments': arguments}}]}])
        for kwargs in ({'enable_thinking': True}, {'preserved_thinking': 'yes'}, {'reasoning_effort': 'medium'}):
            with self.assertRaises(TemplateRequestError):
                template.render([], **kwargs)
        with self.assertRaises(TemplateRequestError):
            template.render([{'role': 'user', 'content': [{'type': 'image', 'text': 'x'}]}])

    def test_tool_result_order_and_thinking_replay(self):
        template = Hy3Template(FIXTURE)
        messages = [{'role': 'user', 'content': 'go'}, {'role': 'assistant', 'content': '', 'reasoning_content': 'plan',
            'tool_calls': [{'id': 'c1', 'function': {'name': 'f', 'arguments': {'n': 2}}},
                           {'id': 'c2', 'function': {'name': 'f', 'arguments': {'n': 3}}}]},
            {'role': 'tool', 'tool_call_id': 'c2', 'content': 'second'},
            {'role': 'tool', 'tool_call_id': 'c1', 'content': 'first'}]
        before = deepcopy(messages)
        rendered = template.render(messages, TOOLS)
        self.assertIn(THINK_START+'plan'+THINK_END, rendered)
        self.assertLess(rendered.index('<tool_response:opensource>\nsecond'),
                        rendered.index('<tool_response:opensource>\nfirst'))
        self.assertTrue(rendered.endswith('<｜hy_Assistant:opensource｜>'+THINK_START+THINK_END))
        self.assertEqual(messages, before)


class ParserTests(unittest.TestCase):
    def parse(self, chunks, *, stops=None, **kwargs):
        parser = Hy3StopParser(stops=stops, tools=TOOLS, **kwargs) if stops is not None else Hy3OutputParser(tools=TOOLS, **kwargs)
        events = []
        for chunk in chunks:
            events.extend(parser.feed(chunk))
        events.extend(parser.finish())
        self.assertEqual(parser.finish(), [])
        return signature(events)

    def fragmented(self, text, expected, **kwargs):
        global FRAGMENTATION_SEQUENCES
        partitions = [[text], list(text)] + [[text[:i], text[i:]] for i in range(len(text)+1)]
        decoder = codecs.getincrementaldecoder('utf8')()
        partitions.append([decoder.decode(bytes([b])) for b in text.encode()]+[decoder.decode(b'', final=True)])
        for chunks in partitions:
            FRAGMENTATION_SEQUENCES += 1
            self.assertEqual(self.parse(chunks, **kwargs), expected, f'fragments={list(map(len,chunks))[:8]}')

    def test_reasoning_and_unicode(self):
        text = THINK_START+'考える🌍'+THINK_END+'Ответ.'
        self.fragmented(text, [('reasoning', '考える🌍'), ('content', 'Ответ.')])
        self.fragmented('мысль'+THINK_END+'4', [('reasoning', 'мысль'), ('content', '4')], thinking=True)

    def test_tools_inside_reasoning_are_text(self):
        markup = group(call({'n': '2'}))
        self.fragmented(markup+THINK_END+'done', [('reasoning', markup), ('content', 'done')], thinking=True)

    def test_typed_values_and_two_calls(self):
        text = 'before'+group(call({'s': '\n 123\n', 'n': '2', 'b': 'true', 'nil': 'null',
            'obj': '{"a":[1,"中",false]}', 'arr': '[1,null]', 'num': '2.5', 'maybe': 'null'})+call())+'after'
        self.fragmented(text, [('content', 'before'), ('tool_call', 'f', {'s': '\n 123\n', 'n': 2, 'b': True,
            'nil': None, 'obj': {'a': [1, '中', False]}, 'arr': [1, None], 'num': 2.5, 'maybe': 'null'}),
            ('tool_call', 'f', {}), ('content', 'after')], stream_tools=True)

    def test_literal_tags_escaping_and_json_quotes(self):
        raw = ' a &amp; <x> "\\path\n'+GROUP_END+' '+CALL_START+'f'+SEPARATOR+' '+VALUE_END+' literal'
        structured = {'quoted': VALUE_END+CALL_END+GROUP_END, 'slash': '\\"中'}
        text = group(call({'s': raw, 'obj': json.dumps(structured, ensure_ascii=False)}))
        self.fragmented(text, [('tool_call', 'f', {'s': raw, 'obj': structured})])
        # Raw typed strings are not assumed to contain valid JSON.
        self.fragmented(group(call({'s': '{"unfinished'})), [('tool_call', 'f', {'s': '{"unfinished'})])

    def test_invalid_typed_json_never_becomes_a_call(self):
        for key, value in [('n', 'true'), ('n', '1.2'), ('b', '1'), ('nil', 'false'), ('obj', '[]'),
                           ('arr', '{}'), ('num', 'NaN'), ('num', 'Infinity'), ('num', '1e400'),
                           ('obj', '{"x":NaN}'), ('obj', '{"x":1,"x":2}')]:
            text = group(call({key: value}))
            with self.subTest(key=key, value=value):
                self.fragmented(text, [('content', text)])

    def test_invalid_second_call_rejects_whole_group(self):
        valid = call({'n': '2'})
        for bad in (call(name='unknown'), call({'n': 'false'}), call({'n': '1'})+'garbage',
                    call({'n': '1'}).replace(CALL_END, KEY_START+'n'+KEY_END+VALUE_START+'2'+VALUE_END+CALL_END)):
            text = group(valid+bad)
            self.fragmented(text, [('content', text)])
        parser = Hy3OutputParser(tools=TOOLS, stream_tools=True)
        self.assertEqual(parser.feed(GROUP_START+valid), [])
        events = parser.feed(GROUP_END)
        self.assertEqual(signature(events), [('tool_call', 'f', {'n': 2})])

    def test_every_truncated_group_prefix_never_announces_a_tool(self):
        text = group(call({'s': '中🙂', 'n': '2'})+call())
        for end in range(1, len(text)):
            self.assertEqual(self.parse(list(text[:end]), stream_tools=True), [('content', text[:end])])
        for text in (GROUP_START+'garbage'+GROUP_END, GROUP_START+GROUP_END,
                     group(call({'s': 'x'})).replace(VALUE_END, '')):
            self.fragmented(text, [('content', text)])

    def test_undeclared_tools_and_no_group_are_literal(self):
        text = group(call({'n': '2'}))
        parser = Hy3OutputParser()
        self.assertEqual(signature(parser.feed(text)+parser.finish()), [('content', text)])
        self.fragmented(call({'n': '2'}), [('content', call({'n': '2'}))])

    def test_template_to_parser_roundtrip(self):
        args = {'s': '\n 中文 & "quoted"\\\n', 'n': 3, 'b': False, 'nil': None, 'obj': {'v': [1, True]}}
        history = [{'role': 'user', 'content': 'go'}, {'role': 'assistant', 'content': '',
            'tool_calls': [{'id': 'c', 'type': 'function', 'function': {'name': 'f', 'arguments': json.dumps(args)}}]}]
        rendered = Hy3Template(FIXTURE).render(history, TOOLS, add_generation_prompt=False)
        text = rendered[rendered.rindex(GROUP_START):].removesuffix('<｜hy_eos:opensource｜>')
        self.fragmented(text, [('tool_call', 'f', args)])

    def test_stop_inside_reasoning_and_tool_group(self):
        self.fragmented('thoughtSTOPtail'+THINK_END, [('reasoning', 'thought')], stops=['STOP'], thinking=True)
        prefix = GROUP_START+call({'s': 'beforeSTOPafter'})
        self.fragmented(prefix+GROUP_END, [('content', prefix.split('STOP')[0])], stops=['STOP'])
        self.fragmented('xabcdTAIL', [('content', 'xa')], stops=['abcd', 'bc'])
        self.fragmented('xabcTAIL', [('content', 'x')], stops=['abc', 'bc'])
        parser = Hy3StopParser(stops=['STOP'])
        self.assertEqual(signature(parser.feed('yesSTOPignored')), [('content', 'yes')])
        self.assertEqual(parser.stop_sequence, 'STOP')
        self.assertEqual(parser.feed(group(call())), [])
        self.assertEqual(parser.finish(), [])

    def test_large_delta_stop_and_group_buffer_limit(self):
        text = 'a'*10000+'STOPtail'
        self.assertEqual(self.parse([text], stops=['STOP']), [('content', 'a'*10000)])
        text = group(call({'s': 'b'*10000}))+group(call())
        self.assertEqual(self.parse([text], max_group_chars=256), [('content', text)])
        self.assertEqual(self.parse(list(text), max_group_chars=256), [('content', text)])

    def test_final_partial_marker_and_stop_validation(self):
        for suffix in ('<', '<tool_calls:', '<think:', '</thi'):
            self.fragmented('text'+suffix, [('content', 'text'+suffix)])
        for value in ([''], [1], {}, 1):
            with self.assertRaises(TemplateRequestError):
                stop_sequences({'stop': value})
        with self.assertRaises(TemplateRequestError):
            stop_sequences({'stop': 'a', 'stop_sequences': ['b']})


if __name__ == '__main__':
    unittest.main()
