"""GLM MCP continuation using a local in-memory hub and scripted engine."""
from copy import deepcopy
import contextlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import threading
import unittest

from serve.frontend import ChatTemplate
from serve.glm5next import GLMTemplate
from serve.mcp import McpCancelled
from serve.server import ByteTokenizer, MockEngine, Service, openai_chunks, openai_collect, run_with_mcp
from serve.test_glm5next_service import Tokenizer


TOOLS = [{"name": "mock__echo", "parameters": {"properties": {"text": {"type": "string"}}}}]


def call(text):
    return '<tool_call>mock__echo<arg_key>text</arg_key><arg_value>' + text + '</arg_value></tool_call>'


class Hub:
    def __init__(self, max_rounds=3, cancel=False):
        self.settings = {"max_rounds": max_rounds}
        self.calls = []
        self.cancel = cancel

    def routes(self):
        return {"mock__echo": (None, "echo")}

    def call(self, name, arguments, cancel):
        self.calls.append((name, dict(arguments)))
        if self.cancel:
            cancel.set()
            raise McpCancelled()
        text = 'result:' + arguments['text']
        return {"ok": True, "text": text, "chars": len(text), "truncated": False, "ms": 0}


class Engine(MockEngine):
    def __init__(self, tok, scripts, stop):
        super().__init__(tok, "")
        self.scripts = [tok.encode(s) + [stop] for s in scripts]
        self.script = self.scripts[0]


class RecordingService(Service):
    def __init__(self, *args):
        super().__init__(*args)
        self.prepared = []

    def prepare(self, messages, tools, kwargs, max_new=None):
        self.prepared.append(deepcopy(messages))
        return super().prepare(messages, tools, kwargs, max_new)


class GLMMcpTests(unittest.TestCase):
    def run_loop(self, scripts, hub=None, max_new=2048, qwen=False):
        hub = hub or Hub()
        tok = ByteTokenizer() if qwen else Tokenizer()
        template = ChatTemplate(Path(__file__).parent / 'chat_template.jinja') if qwen else GLMTemplate(
            Path(__file__).parent / 'fixtures/glm53_chat_template.jinja')
        svc = RecordingService(Engine(tok, scripts, 257 if qwen else 154829), tok, template)
        original = [{"role": "user", "content": "Use tools"}]
        ids, thinking, maximum = svc.prepare(original, TOOLS, {}, max_new)
        cancel = threading.Event()
        run = run_with_mcp(svc, hub, original, TOOLS, {}, ids, thinking, maximum, max_new, {}, cancel, {'mock__echo'})
        with contextlib.redirect_stdout(io.StringIO()):
            chunks = list(openai_chunks(svc, {}, ids, thinking, TOOLS, maximum, cancel, run=run))
        self.assertEqual(original, [{"role": "user", "content": "Use tools"}])
        return svc, hub, chunks, openai_collect(chunks)

    def test_multiple_calls_keep_ids_through_prompt_and_api(self):
        first = 'Need both</think>Checking.' + call('A') + call('B')
        svc, hub, chunks, answer = self.run_loop([first, 'Have results</think>Done'])
        self.assertEqual(hub.calls, [('mock__echo', {'text': 'A'}), ('mock__echo', {'text': 'B'})])
        self.assertEqual(len(svc.prepared), 2)
        history = svc.prepared[1]
        self.assertEqual(history[1]['reasoning_content'], 'Need both')
        self.assertEqual(history[1]['content'], 'Checking.')
        calls = history[1]['tool_calls']
        self.assertNotEqual(calls[0]['id'], calls[1]['id'])
        self.assertEqual([m['tool_call_id'] for m in history[2:]], [c['id'] for c in calls])
        self.assertEqual([m['content'] for m in history[2:]], ['result:A', 'result:B'])
        # Server history can now round-trip through the strict GLM API normalizer.
        normalized = svc.normalize_request({'messages': history}, 'openai')[0]
        self.assertEqual(normalized, history)
        reordered = history[:2] + list(reversed(history[2:]))
        prompt = svc.template.render(reordered, tools=TOOLS)
        self.assertIn('<tool_response>result:A</tool_response><tool_response>result:B</tool_response>', prompt)
        events = [c['strata_mcp'] for c in chunks if 'strata_mcp' in c]
        self.assertEqual([e['id'] for e in events if e['event'] == 'call'], [c['id'] for c in calls])
        self.assertEqual([e['id'] for e in events if e['event'] == 'result'], [c['id'] for c in calls])
        self.assertEqual(answer['choices'][0]['finish_reason'], 'stop')
        self.assertEqual(answer['choices'][0]['message']['content'], 'Checking.Done')
        self.assertNotIn('tool_calls', answer['choices'][0]['message'])

    def test_repeated_tool_across_rounds_has_distinct_ids(self):
        svc, hub, _, _ = self.run_loop(['</think>' + call('A'), '</think>' + call('B'), '</think>Done'])
        self.assertEqual(len(hub.calls), 2)
        history = svc.prepared[-1]
        self.assertEqual([m['role'] for m in history], ['user', 'assistant', 'tool', 'assistant', 'tool'])
        self.assertNotEqual(history[2]['tool_call_id'], history[4]['tool_call_id'])
        for assistant, result in ((history[1], history[2]), (history[3], history[4])):
            self.assertEqual(assistant['tool_calls'][0]['id'], result['tool_call_id'])

    def test_cancel_does_not_prepare_partial_results(self):
        svc, hub, _, answer = self.run_loop(['</think>' + call('A') + call('B')], Hub(cancel=True))
        self.assertEqual(len(svc.prepared), 1)
        self.assertEqual(len(hub.calls), 1)
        self.assertEqual(answer['choices'][0]['finish_reason'], 'stop')

    def test_round_limit_and_length_never_execute_calls(self):
        for hub, maximum in ((Hub(max_rounds=0), 2048), (Hub(), len('</think>' + call('A')))):
            with self.subTest(maximum=maximum):
                svc, hub, chunks, answer = self.run_loop(['</think>' + call('A')], hub, maximum)
                self.assertEqual(hub.calls, [])
                self.assertEqual(len(svc.prepared), 1)
                skipped = [c['strata_mcp'] for c in chunks if c.get('strata_mcp', {}).get('skipped')]
                self.assertEqual(len(skipped), 1)
                self.assertEqual(answer['choices'][0]['finish_reason'], 'stop' if maximum == 2048 else 'length')

    def test_client_tools_prevent_mcp_execution(self):
        svc, hub, _, answer = self.run_loop(['</think>' + call('A') + '<tool_call>client_tool</tool_call>'])
        self.assertEqual(hub.calls, [])
        self.assertEqual(len(svc.prepared), 1)
        self.assertEqual(answer['choices'][0]['finish_reason'], 'tool_calls')
        self.assertEqual(answer['choices'][0]['message']['tool_calls'][0]['function']['name'], 'client_tool')

    def test_qwen_loop_accepts_added_ids(self):
        first = '</think><tool_call><function=mock__echo><parameter=text>A</parameter></function></tool_call>'
        svc, hub, _, answer = self.run_loop([first, '</think>Done'], qwen=True)
        self.assertEqual(hub.calls, [('mock__echo', {'text': 'A'})])
        self.assertEqual(svc.prepared[1][1]['tool_calls'][0]['id'], svc.prepared[1][2]['tool_call_id'])
        self.assertEqual(answer['choices'][0]['message']['content'], 'Done')


@unittest.skipUnless(shutil.which('node'), 'Node is required for real web history replay')
class GLMWebHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        script = Path(__file__).parent / 'test_glm5next_mcp_ui.cjs'
        result = subprocess.run([shutil.which('node'), str(script), '--history'],
                                capture_output=True, text=True, encoding='utf-8', check=True, timeout=15)
        cls.history = json.loads(result.stdout)
        cls.template = GLMTemplate(Path(__file__).parent / 'fixtures/glm53_chat_template.jinja')

    def render(self, history, clear):
        messages, tools, options = self.template.normalize_openai(
            {'messages': history, 'chat_template_kwargs': {'clear_thinking': clear}})
        return self.template.render(messages, tools=tools, **options)

    def test_browser_history_retains_current_tool_turn_reasoning(self):
        for clear in (False, True):
            with self.subTest(clear=clear):
                prompt = self.render(self.history, clear)
                self.assertIn('<think>reason1</think>Before<tool_call>', prompt)
                self.assertIn('<think>reason2</think>After<tool_call>', prompt)
                self.assertIn('<think>reason3</think>Final', prompt)
                self.assertIn('<tool_response>result:A</tool_response><tool_response>error:B</tool_response>', prompt)
                self.assertEqual(prompt.count('<tool_call>'), 3)

    def test_clear_thinking_clears_only_prior_turn_after_new_user_message(self):
        history = self.history + [{'role': 'user', 'content': 'Next question'}]
        kept = self.render(history, False)
        cleared = self.render(history, True)
        for thought in ('reason1', 'reason2', 'reason3'):
            self.assertIn(thought, kept)
            self.assertNotIn(thought, cleared)
        for visible in ('Before', 'After', 'Final', 'result:A', 'error:B', 'result:C', 'Next question'):
            self.assertIn(visible, cleared)
        self.assertEqual(cleared.count('<tool_call>'), 3)


if __name__ == '__main__':
    unittest.main()
