"""Prompt fixtures only: no GPU, model weights, tokenizer or server required."""
from copy import deepcopy
import hashlib
from pathlib import Path
import unittest

from serve.frontend import TemplateRequestError
from serve.glm5next import GLMOutputParser, GLMTemplate


FIXTURE = Path(__file__).parent / "fixtures" / "glm53_chat_template.jinja"
PREFIX = "[gMASK]<sop><|system|>Reasoning Effort: Max"
GENERATION = "<|assistant|><think>"


def call(call_id, name, arguments):
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": arguments}}


class GLMTemplateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template = GLMTemplate(FIXTURE)

    def test_embedded_template_bytes(self):
        self.assertEqual(hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
                         "a4fddbbf0b432101a296c17094f8bc5a2b0d30713b5b5cd92f86be78511aa724")

    def test_multi_message_exact_prompt(self):
        messages = [
            {"role": "system", "content": "Be precise."},
            {"role": "user", "content": "Привет 🌍"},
            {"role": "assistant", "content": "  你好  ", "reasoning_content": "Reason"},
            {"role": "user", "content": [{"type": "text", "text": "Next"}, "!"]},
        ]
        expected = (PREFIX + "<|system|>Be precise.<|user|>Привет 🌍"
                    "<|assistant|><think>Reason</think>你好<|user|>Next!")
        self.assertEqual(self.template.render(messages), expected + GENERATION)
        self.assertEqual(self.template.render(messages, add_generation_prompt=False), expected)

    def test_effort_values_and_default(self):
        for effort in ("low", "high", "max"):
            with self.subTest(effort=effort):
                self.assertEqual(self.template.render([], reasoning_effort=effort),
                                 "[gMASK]<sop><|system|>Reasoning Effort: " + effort.capitalize() + GENERATION)
        self.assertEqual(self.template.render([]), PREFIX + GENERATION)

    def test_reject_incompatible_qwen_settings(self):
        for kwargs in ({"reasoning_effort": "xhigh"}, {"reasoning_effort": "medium"},
                       {"reasoning_effort": None}, {"enable_thinking": False},
                       {"enable_thinking": True}, {"clear_thinking": "false"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(TemplateRequestError):
                self.template.render([], **kwargs)

    def test_clear_thinking_retains_current_tool_turn(self):
        messages = [
            {"role": "user", "content": "Old"},
            {"role": "assistant", "content": "<think>old reason</think>old answer"},
            {"role": "user", "content": "New"},
            {"role": "assistant", "content": None, "reasoning_content": "current reason",
             "tool_calls": [call("a", "lookup", {})]},
            {"role": "tool", "tool_call_id": "a", "content": "result"},
        ]
        expected = (PREFIX + "<|user|>Old<|assistant|><think></think>old answer<|user|>New"
                    "<|assistant|><think>current reason</think><tool_call>lookup</tool_call>"
                    "<|observation|><tool_response>result</tool_response>" + GENERATION)
        self.assertEqual(self.template.render(messages, clear_thinking=True), expected)
        self.assertEqual(self.template.render(messages),
                         expected.replace("<think></think>old answer", "<think>old reason</think>old answer"))

    def test_typed_arguments_and_no_history_mutation(self):
        messages = [{"role": "assistant", "content": None, "tool_calls": [
            call("a", "search", '{"q":"你好","n":2,"ok":true,"ids":[1,2],"extra":null}'),
            {"id": "b", "name": "done", "arguments": {"obj": {"x": 1}}}]}]
        original = deepcopy(messages)
        expected = (PREFIX + "<|assistant|><think></think><tool_call>search"
                    "<arg_key>q</arg_key><arg_value>你好</arg_value>"
                    "<arg_key>n</arg_key><arg_value>2</arg_value>"
                    "<arg_key>ok</arg_key><arg_value>true</arg_value>"
                    "<arg_key>ids</arg_key><arg_value>[1, 2]</arg_value>"
                    "<arg_key>extra</arg_key><arg_value>null</arg_value></tool_call>"
                    '<tool_call>done<arg_key>obj</arg_key><arg_value>{"x": 1}</arg_value></tool_call>')
        self.assertEqual(self.template.render(messages, add_generation_prompt=False), expected)
        self.assertEqual(messages, original)

    def test_invalid_arguments_are_request_errors(self):
        for arguments in ("{", "[]", "null", "1", None, [], 1):
            with self.subTest(arguments=arguments), self.assertRaises(TemplateRequestError):
                self.template.render([{"role": "assistant", "tool_calls": [call("a", "f", arguments)]}])

    def test_tool_results_follow_call_ids(self):
        messages = [{"role": "assistant", "tool_calls": [call("a", "f", {}), call("b", "g", {})]},
                    {"role": "tool", "tool_call_id": "b", "content": "B"},
                    {"role": "tool", "tool_call_id": "a", "content": "A"},
                    {"role": "user", "content": "Continue"}]
        expected = (PREFIX + "<|assistant|><think></think><tool_call>f</tool_call><tool_call>g</tool_call>"
                    "<|observation|><tool_response>A</tool_response><tool_response>B</tool_response>"
                    "<|user|>Continue" + GENERATION)
        self.assertEqual(self.template.render(messages), expected)
        # The template also accepts a batch of outputs in one tool message.
        messages[1:3] = [{"role": "tool", "content": [{"id": "b", "output": "B"},
                                                      {"id": "a", "output": [{"type": "text", "text": "A"}]}]}]
        self.assertEqual(self.template.render(messages), expected)

    def test_ambiguous_result_ids_preserve_arrival_order(self):
        for ids in (("a", "a"), ("unknown", "a"), (None, "a")):
            with self.subTest(ids=ids):
                messages = [{"role": "assistant", "tool_calls": [call("a", "f", {}), call("b", "g", {})]},
                            {"role": "tool", "tool_call_id": ids[0], "content": "B"},
                            {"role": "tool", "tool_call_id": ids[1], "content": "A"}]
                self.assertTrue(self.template.render(messages).endswith(
                    "<|observation|><tool_response>B</tool_response><tool_response>A</tool_response>" + GENERATION))

    def test_wrapped_and_bare_tool_schemas_and_references(self):
        tools = [{"name": "f", "description": "查找", "parameters": {"type": "object"}, "strict": True},
                 {"name": "later", "defer_loading": True, "parameters": {"type": "object"}}]
        messages = [{"role": "user", "content": "Find"},
                    {"role": "tool", "content": [{"type": "tool_reference", "name": "later"}]}]
        prompt = self.template.render(messages, tools=tools)
        self.assertEqual(prompt, self.template.render(messages, tools=[{"type": "function", "function": t} for t in tools]))
        self.assertIn('{"name": "f", "description": "查找", "parameters": {"type": "object"}}', prompt)
        self.assertNotIn('"strict"', prompt)
        self.assertNotIn('"defer_loading"', prompt)
        self.assertEqual(prompt.count('"name": "later"'), 1)
        self.assertTrue(prompt.endswith('<|observation|><tool_response><tools>\n'
                                        '{"name": "later", "parameters": {"type": "object"}}\n'
                                        '</tools></tool_response>' + GENERATION))


def signature(events):
    """Compare semantic events independently of text chunking and generated IDs."""
    out = []
    for event in events:
        if event.kind == "tool_call":
            out.append((event.kind, event.call.name, event.call.arguments))
        elif out and out[-1][0] == event.kind:
            out[-1] = (event.kind, out[-1][1] + event.text)
        else:
            out.append((event.kind, event.text))
    return out


class GLMParserTests(unittest.TestCase):
    def parse(self, chunks, **kwargs):
        parser = GLMOutputParser(**kwargs)
        events = []
        for chunk in chunks:
            events.extend(parser.feed(chunk))
        events.extend(parser.finish())
        self.assertEqual(parser.finish(), [])
        return events

    def assert_fragmentation(self, text, expected, **kwargs):
        # Every two-chunk boundary, then uniform widths including single chars.
        for split in range(len(text) + 1):
            with self.subTest(split=split):
                self.assertEqual(signature(self.parse([text[:split], text[split:]], **kwargs)), expected)
        for width in (1, 2, 3, 7, 19):
            with self.subTest(width=width):
                self.assertEqual(signature(self.parse(
                    [text[i:i + width] for i in range(0, len(text), width)], **kwargs)), expected)

    def test_reasoning_and_multiple_calls_all_boundaries(self):
        text = ('Рассуждение 🌍</think>\nAnswer\n<tool_call>search'
                '<arg_key>q</arg_key><arg_value>你好</arg_value>'
                '<arg_key>n</arg_key><arg_value>2</arg_value>'
                '<arg_key>options</arg_key><arg_value>{"ok":true,"ids":[1,null]}</arg_value>'
                '</tool_call>\n<tool_call>clock</tool_call>Done')
        expected = [("reasoning", "Рассуждение 🌍"), ("content", "\nAnswer\n"),
                    ("tool_call", "search", {"q": "你好", "n": 2, "options": {"ok": True, "ids": [1, None]}}),
                    ("content", "\n"), ("tool_call", "clock", {}), ("content", "Done")]
        self.assert_fragmentation(text, expected)

    def test_optional_opening_think_and_empty_reasoning(self):
        for thinking in (True, False):
            self.assert_fragmentation('<think>reason</think>answer',
                                      [("reasoning", "reason"), ("content", "answer")], thinking=thinking)
            self.assert_fragmentation('<think></think>answer', [("content", "answer")], thinking=thinking)
        self.assert_fragmentation('</think>answer', [("content", "answer")])
        self.assert_fragmentation('intro<think>reason</think>end',
                                  [("content", "intro"), ("reasoning", "reason"), ("content", "end")], thinking=False)

    def test_calls_in_reasoning_are_not_executable(self):
        text = '<tool_call>clock</tool_call>'
        self.assert_fragmentation(text + '</think>done', [("reasoning", text), ("content", "done")])

    def test_schema_preserves_json_looking_strings(self):
        fn = {"name": "f", "parameters": {"properties": {"s": {"type": "string"}, "n": {"type": "integer"}}}}
        text = ('<tool_call>f<arg_key>s</arg_key><arg_value>  true\n</arg_value>'
                '<arg_key>n</arg_key><arg_value>3</arg_value></tool_call>')
        for tools in ([fn], [{"type": "function", "function": fn}]):
            self.assert_fragmentation(text, [("tool_call", "f", {"s": "  true\n", "n": 3})],
                                      thinking=False, tools=tools)

    def test_literal_tags_inside_string_value(self):
        value = 'keep </tool_call> and <tool_call>nested</tool_call>; </arg_value> is text <think>x</think>'
        text = '<tool_call>write<arg_key>content</arg_key><arg_value>' + value + '</arg_value></tool_call>'
        self.assert_fragmentation(text, [("tool_call", "write", {"content": value})], thinking=False)

    def test_malformed_calls_remain_exact_text(self):
        for text in ('<tool_call></tool_call>', '<tool_call>bad name</tool_call>',
                     '<tool_call><function=clock></function></tool_call>',
                     '<tool_call>f<arg_key></arg_key><arg_value>x</arg_value></tool_call>',
                     '<tool_call>f<arg_key>x</arg_key><arg_value>1</arg_value>'
                     '<arg_key>x</arg_key><arg_value>2</arg_value></tool_call>',
                     '<tool_call>f garbage<arg_key>x</arg_key><arg_value>1</arg_value></tool_call>',
                     '<tool_call>f<arg_key>x</arg_key>missing value</tool_call>'):
            with self.subTest(text=text):
                self.assert_fragmentation(text, [("content", text)], thinking=False, stream_tools=True)

    def test_invalid_declared_json_never_executes(self):
        tools = [{"name": "f", "parameters": {"properties": {"n": {"type": "number"}}}}]
        for value in ('oops', 'NaN', 'Infinity', '1e400', '{"x":NaN}'):
            text = '<tool_call>f<arg_key>n</arg_key><arg_value>' + value + '</arg_value></tool_call>'
            self.assert_fragmentation(text, [("content", text)], thinking=False, tools=tools)

    def test_every_truncated_prefix_of_call_is_text(self):
        text = '<tool_call>f<arg_key>x</arg_key><arg_value>{"a":[1,true]}</arg_value></tool_call>'
        for end in range(1, len(text)):
            with self.subTest(end=end):
                self.assertEqual(signature(self.parse(list(text[:end]), thinking=False, stream_tools=True)),
                                 [("content", text[:end])])

    def test_stream_tools_waits_for_complete_validation(self):
        parser = GLMOutputParser(thinking=False, stream_tools=True)
        text = '<tool_call>clock</tool_call>'
        for char in text[:-1]:
            self.assertEqual(parser.feed(char), [])
        events = parser.feed(text[-1])
        self.assertEqual(signature(events), [("tool_call", "clock", {})])
        self.assertTrue(events[0].call.id.startswith('call_'))
        self.assertEqual(parser.finish(), [])

    def test_valid_call_after_invalid_call(self):
        text = '<tool_call></tool_call><tool_call>clock</tool_call><tool_call>clock</tool_call>'
        events = self.parse(list(text), thinking=False)
        self.assertEqual(signature(events), [("content", '<tool_call></tool_call>'),
                                             ("tool_call", "clock", {}), ("tool_call", "clock", {})])
        calls = [e.call for e in events if e.kind == "tool_call"]
        self.assertNotEqual(calls[0].id, calls[1].id)

    def test_partial_markers_and_plain_text_preserved(self):
        for text in ('plain <tag>\n\n', '<thi', 'x<tool_', ' x <thinkish> ', ''):
            self.assert_fragmentation(text, [("content", text)] if text else [], thinking=False)
        self.assert_fragmentation('reason</thi', [("reasoning", 'reason</thi')])

    def test_template_call_result_continuation_round_trip(self):
        template = GLMTemplate(FIXTURE)
        self.assertIs(template.output_parser, GLMOutputParser)
        fn = {"name": "f", "parameters": {"properties": {"s": {"type": "string"}}}}
        original = {"s": "true", "array": [1, False], "obj": {"key": "值"}}
        rendered = template.render([{"role": "assistant", "content": "", "tool_calls": [call("a", "f", original)]}],
                                   add_generation_prompt=False)
        completion = rendered.split('<|assistant|>', 1)[1]
        events = self.parse(list(completion), tools=[fn])
        self.assertEqual(signature(events), [("tool_call", "f", original)])
        parsed = events[0].call
        messages = [{"role": "user", "content": "Run"},
                    {"role": "assistant", "tool_calls": [call(parsed.id, parsed.name, parsed.arguments)]},
                    {"role": "tool", "tool_call_id": parsed.id, "content": "Result"}]
        self.assertTrue(template.render(messages).endswith(
            '<|observation|><tool_response>Result</tool_response>' + GENERATION))
        self.assertEqual(signature(self.parse(list('checked</think>Done'))),
                         [("reasoning", "checked"), ("content", "Done")])


if __name__ == "__main__":
    unittest.main()
