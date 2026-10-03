import json
from pathlib import Path
import tempfile
import threading
import unittest

from serve.deepseek import DeepSeekOutputParser, DeepSeekTemplate, START, END
from serve.frontend import ChatTemplate, OutputParser
from serve.server import ByteTokenizer, MockEngine, Service, openai_chunks, openai_collect


class DeepSeekParserTests(unittest.TestCase):
    def parse(self, text, width, thinking=False):
        parser = DeepSeekOutputParser(thinking=thinking)
        events = []
        for i in range(0, len(text), width):
            events += parser.feed(text[i:i + width])
        return events + parser.finish()

    def test_dsml_fragmentation_and_typed_arguments(self):
        text = ('Reason.</think>\n\n' + START + '\n'
                '<｜DSML｜invoke name="search"><｜DSML｜parameter name="q" string="true">'
                'a & b <tag>\nПривет</｜DSML｜parameter>\n'
                '<｜DSML｜parameter name="limit" string="false">3</｜DSML｜parameter>\n'
                '<｜DSML｜parameter name="opts" string="false">{"ok":true}</｜DSML｜parameter>'
                '</｜DSML｜invoke>\n<｜DSML｜invoke name="clock"></｜DSML｜invoke>\n' + END)
        for width in range(1, len(text) + 1):
            events = self.parse(text, width, True)
            self.assertEqual(''.join(e.text for e in events if e.kind == 'reasoning'), 'Reason.')
            calls = [e.call for e in events if e.kind == 'tool_call']
            self.assertEqual([(c.name, c.arguments) for c in calls], [
                ('search', {'q': 'a & b <tag>\nПривет', 'limit': 3, 'opts': {'ok': True}}), ('clock', {})])
            self.assertFalse(''.join(e.text for e in events if e.kind == 'content').strip())

    def test_partial_or_bad_calls_never_execute(self):
        for text in [START, START + '<｜DSML｜invoke name="x">',
                     START + '<｜DSML｜invoke name="x"><｜DSML｜parameter name="q" string="false">oops</｜DSML｜parameter></｜DSML｜invoke>' + END,
                     START + 'unknown' + END]:
            events = self.parse(text, 1)
            self.assertFalse(any(e.kind == 'tool_call' for e in events))
            self.assertEqual(''.join(e.text for e in events), text)

    def test_plain_text_and_literal_qwen_tags(self):
        text = 'Обычный текст <tool_call> не DSML.'
        for width in (1, 5, 100):
            self.assertEqual(''.join(e.text for e in self.parse(text, width)), text)

    def test_template_uses_deepseek_tokens_and_thinking(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'chat.jinja'
            path.write_text('{{bos_token}}{% if enable_thinking %}<think>{% else %}</think>{% endif %}{{reasoning_effort}}', encoding='utf-8')
            tpl = DeepSeekTemplate(path)
            self.assertEqual(tpl.render([]), '<｜begin▁of▁sentence｜><think>')
            self.assertEqual(tpl.render([], enable_thinking=False), '<｜begin▁of▁sentence｜></think>')
            self.assertTrue(tpl.render([], reasoning_effort='xhigh').endswith('high'))
            self.assertFalse(hasattr(ChatTemplate(path), 'architecture'))

    def test_service_uses_only_deepseek_eos_not_qwen_fragments(self):
        class Tokenizer(ByteTokenizer):
            special_ids = {'tokenizer.ggml.eos_token_id': 9999}
        tok = Tokenizer()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'chat.jinja'
            path.write_text('prompt')
            engine = MockEngine(tok, 'Hello!')
            engine.script = tok.encode('Hello!') + [9999]
            svc = Service(engine, tok, DeepSeekTemplate(path))
            self.assertEqual(svc.stop_ids, {9999})
            answer = openai_collect(openai_chunks(svc, {}, [1], False, None, 32, threading.Event()))
            self.assertEqual(answer['choices'][0]['message']['content'], 'Hello!')

    def test_normalized_tools_are_wrapped_for_gguf_template(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'chat.jinja'
            path.write_text('{% for tool in tools %}{% if tool.type == "function" %}{{tool.function|tojson}}{% endif %}{% endfor %}')
            tpl = DeepSeekTemplate(path)
            fn = {'name': 'weather', 'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}}}
            self.assertEqual(json.loads(tpl.render([], [fn])), fn)
            self.assertEqual(json.loads(tpl.render([], [{'type': 'function', 'function': fn}])), fn)

    def test_service_returns_dsml_as_openai_tool_calls(self):
        tok = ByteTokenizer()
        tok.special_ids = {'tokenizer.ggml.eos_token_id': 9999}
        text = START + '<｜DSML｜invoke name="clock"></｜DSML｜invoke>' + END
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'chat.jinja'
            path.write_text('prompt')
            engine = MockEngine(tok, text)
            svc = Service(engine, tok, DeepSeekTemplate(path))
            answer = openai_collect(openai_chunks(svc, {}, [1], False, None, 1000, threading.Event()))
            fn = answer['choices'][0]['message']['tool_calls'][0]['function']
            self.assertEqual(fn['name'], 'clock')
            self.assertEqual(json.loads(fn['arguments']), {})


if __name__ == '__main__':
    unittest.main()
