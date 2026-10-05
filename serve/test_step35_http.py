"""Step EOG and both APIs through real loopback sockets; scripted byte engine."""
from copy import deepcopy
import json
import socket
from pathlib import Path
import tempfile
import threading
import unittest

from serve.server import ByteTokenizer, MockEngine, Service, configured_template
from serve.step35 import StepTemplate, StepStopParser
from serve.test_step35 import FIXTURE, template
from serve import test_glm5next_http as http_helpers


class Tokenizer(ByteTokenizer):
    # Relocated IDs prove lookup uses vocabulary/metadata, not fixed numbers.
    SPECIALS = ['<｜begin▁of▁sentence｜>', '<｜end▁of▁sentence｜>', '<|im_end|>', '<｜▁pad▁｜>', '<|im_start|>']
    pre = 'deepseek-v3'

    def __init__(self):
        self.tokens = [str(i) for i in range(256)] + self.SPECIALS
        self.ids = {t: i for i, t in enumerate(self.tokens)}
        self.token_types = [1] * 256 + [3] * len(self.SPECIALS)
        self.special_ids = {'tokenizer.ggml.bos_token_id': 256, 'tokenizer.ggml.eos_token_id': 258,
                            'tokenizer.ggml.padding_token_id': 259}

    def token_bytes(self, token_id):
        if token_id in (257, 258):
            raise AssertionError('Step EOG reached detokenization')
        return self.tokens[token_id].encode() if token_id >= 256 else bytes([token_id])


class Engine(MockEngine):
    def __init__(self, tok, scripts=None, prefix=None, stop=258):
        super().__init__(tok, '')
        self.scripts = [tok.encode(s, parse_special=True) + [stop] + tok.encode('LEAK')
                        for s in (scripts or ['</think>Next'])]
        self.script = self.scripts[0]
        self.prefix, self.calls = prefix, 0
        self.entered, self.closed = threading.Event(), threading.Event()
        self.cancellations, self.closes = [], []
        self.timed_out = False

    def generate(self, ids, max_new, sampling, cancel, **kwargs):
        self.calls += 1
        self.cancellations.append(cancel)
        try:
            if self.prefix is not None and self.calls == 1:
                self.last_prompt = list(ids)
                yield from self.tok.encode(self.prefix)
                self.entered.set()
                self.timed_out = not cancel.wait(5)
            else:
                yield from super().generate(ids, max_new, sampling, cancel, **kwargs)
        finally:
            self.closes.append(self.service.fifo.locked())
            self.closed.set()


class StepHTTPTests(unittest.TestCase):
    # Only transport/assertion helpers; no GLM grammar or test cases inherited.
    listener = http_helpers.GLMHTTPTests.listener
    path = staticmethod(http_helpers.GLMHTTPTests.path)
    post = http_helpers.GLMHTTPTests.post
    pending = http_helpers.GLMHTTPTests.pending
    assert_answer = http_helpers.GLMHTTPTests.assert_answer

    @staticmethod
    def request(**options):
        return {'model': 'step-3.7-flash', 'messages': [{'role': 'user', 'content': 'Hello'}],
                'max_tokens': 2048, **options}

    def service(self, scripts=None, prefix=None, stop=258):
        tok = Tokenizer()
        engine = Engine(tok, scripts, prefix, stop)
        svc = Service(engine, tok, template(), model_name='step-3.7-flash')
        svc.api_monitor = True
        engine.service = svc
        return svc

    def test_eog_json_sse_unicode_and_pad(self):
        for api in ('openai', 'anthropic'):
            for stream in (False, True):
                for stop in (257, 258):
                    with self.subTest(api=api, stream=stream, stop=stop):
                        svc = self.service(['Think</think>Привет 🌍<｜▁pad▁｜>after PAD'], stop=stop)
                        with self.listener(svc) as httpd:
                            answer = self.post(httpd, api, self.request(stream=stream))
                            self.assert_answer(api, answer, 'Привет 🌍<｜▁pad▁｜>after PAD', 'Think')
                            self.assertNotIn('LEAK', json.dumps(answer))
                            self.assertEqual(svc.engine.closes, [True])

    def test_stop_string_and_length(self):
        for api in ('openai', 'anthropic'):
            for stream in (False, True):
                for length in (False, True):
                    svc = self.service(['</think>A!STOP!LEAK'])
                    options = {'max_tokens': 9} if length else {('stop' if api == 'openai' else 'stop_sequences'): ['!STOP!']}
                    with self.subTest(api=api, stream=stream, length=length), self.listener(svc) as httpd:
                        answer = self.post(httpd, api, self.request(stream=stream, **options))
                        if api == 'openai':
                            self.assertEqual(answer['choices'][0]['finish_reason'], 'length' if length else 'stop')
                            self.assertEqual(answer['choices'][0]['message']['content'], 'A')
                        else:
                            self.assertEqual(answer['stop_reason'], 'max_tokens' if length else 'stop_sequence')
                            self.assertEqual(answer['stop_sequence'], None if length else '!STOP!')
                            self.assertEqual([b['text'] for b in answer['content'] if b['type'] == 'text'], ['A'])

    def test_tool_roundtrip(self):
        function = {'name': 'echo', 'parameters': {'type': 'object', 'properties': {'text': {'type': 'string'}}}}
        raw = 'Need tool</think><tool_call><function=echo><parameter=text>Привет</parameter></function></tool_call>'
        for api in ('openai', 'anthropic'):
            for stream in (False, True):
                svc = self.service([raw, 'Have result</think>Next'])
                tool = {'type': 'function', 'function': function} if api == 'openai' else {
                    'name': function['name'], 'input_schema': function['parameters']}
                req = self.request(stream=stream, tools=[tool])
                with self.subTest(api=api, stream=stream), self.listener(svc) as httpd:
                    answer = self.post(httpd, api, req)
                    if api == 'openai':
                        self.assertEqual(answer['choices'][0]['finish_reason'], 'tool_calls')
                        msg = answer['choices'][0]['message']; call = msg['tool_calls'][0]
                        self.assertEqual(json.loads(call['function']['arguments']), {'text': 'Привет'})
                        req['messages'] += [msg, {'role': 'tool', 'tool_call_id': call['id'], 'content': 'result:A'}]
                    else:
                        self.assertEqual(answer['stop_reason'], 'tool_use')
                        call = next(b for b in answer['content'] if b['type'] == 'tool_use')
                        self.assertEqual(call['input'], {'text': 'Привет'})
                        req['messages'] += [{'role': 'assistant', 'content': answer['content']}, {'role': 'user', 'content': [
                            {'type': 'tool_result', 'tool_use_id': call['id'], 'content': 'result:A'}]}]
                    self.assert_answer(api, self.post(httpd, api, req), 'Next', 'Have result')
                    self.assertIn('result:A', svc.tok.decode(svc.engine.last_prompt))
                    self.assertEqual(svc.engine.closes, [True, True])

    def test_rejected_options_do_not_start_engine(self):
        for api in ('openai', 'anthropic'):
            for options in ({'reasoning_budget_tokens': 8}, {'chat_template_kwargs': {'enable_thinking': False}},
                            {'chat_template_kwargs': {'reasoning_effort': 'none'}}, {'stop': ['']}, {'stop': 7}):
                svc = self.service()
                with self.subTest(api=api, options=options), self.listener(svc) as httpd:
                    answer = self.post(httpd, api, self.request(stream=True, **options), expected_status=400)
                    self.assertEqual(answer['error']['type'], 'invalid_request_error')
                    self.assertEqual(svc.engine.cancellations, [])
                    self.assert_answer(api, self.post(httpd, api, self.request()), 'Next')

    def test_mcp_local_stub_continuation(self):
        from serve.test_glm5next_handlers import HandlerHub
        for stream in (False, True):
            svc = self.service(['Need tool</think><tool_call><function=mock__echo><parameter=text>A</parameter></function></tool_call>',
                                'Have result</think>Next'])
            svc.mcp = HandlerHub()
            with self.subTest(stream=stream), self.listener(svc) as httpd:
                answer = self.post(httpd, 'openai', self.request(stream=stream, strata_mcp=True))
                self.assertEqual(svc.mcp.calls, [('mock__echo', {'text': 'A'})])
                self.assert_answer('openai', answer, 'Next')
                self.assertIn('result:A', svc.tok.decode(svc.engine.last_prompt))
                self.assertEqual(svc.engine.closes, [True, True])

    def test_stop_chunk_boundaries_and_incomplete_tool(self):
        text = '</think>Привет🌍!STOP!LEAK'
        for split in range(len(text)+1):
            parser = StepStopParser(thinking=True, stops=['!STOP!', '!STOP!long'])
            events = parser.feed(text[:split]) + parser.feed(text[split:]) + parser.finish()
            self.assertEqual(''.join(e.text for e in events), 'Привет🌍')
            self.assertEqual(parser.stop_sequence, '!STOP!')
        parser = StepStopParser(thinking=False, stops=['abcd'])
        events = parser.feed('xabc') + parser.finish()
        self.assertEqual(''.join(e.text for e in events), 'xabc')
        for split in range(5):
            parser = StepStopParser(thinking=False, stops=['abcd', 'bc'])
            events = parser.feed('abcd'[:split]) + parser.feed('abcd'[split:]) + parser.finish()
            self.assertEqual(''.join(e.text for e in events), 'a')
            self.assertEqual(parser.stop_sequence, 'bc')
        parser = StepStopParser(thinking=False, stops=['CUT'])
        events = parser.feed('<tool_call><function=echo><parameter=text>A')
        events += parser.feed('CUT</parameter></function></tool_call>') + parser.finish()
        self.assertFalse(any(e.kind == 'tool_call' for e in events))
        self.assertNotIn('CUT', ''.join(e.text for e in events))

    def test_disconnect_and_fresh_request(self):
        for api in ('openai', 'anthropic'):
            for stream in (False, True):
                for prefix in ('', 'Thinking now', '</think><tool_call><function=echo><parameter=text>A'):
                    svc = self.service(prefix=prefix)
                    with self.subTest(api=api, stream=stream, prefix=prefix), self.listener(svc) as httpd:
                        client = self.pending(httpd, api, self.request(stream=stream))
                        try:
                            self.assertTrue(svc.engine.entered.wait(5))
                            self.assertFalse(svc.engine.closed.wait(.3))
                            client.shutdown(socket.SHUT_RDWR)
                        finally:
                            client.close()
                        self.assertTrue(svc.engine.closed.wait(3))
                        self.assertFalse(svc.engine.timed_out)
                        self.assertTrue(svc.engine.cancellations[0].is_set())
                        http_helpers.wait_for(lambda: not svc.fifo.locked() and not svc.status['busy'] and
                                              svc.api_requests[-1]['state'] == 'disconnected')
                        self.assertEqual(svc.engine.closes, [True])
                        self.assert_answer(api, self.post(httpd, api, self.request()), 'Next')
                        self.assertEqual(svc.engine.closes, [True, True])

    test_queued_disconnect = http_helpers.GLMHTTPTests.test_queued_socket_disconnect_never_starts_engine

    def test_metadata_and_capabilities(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'chat_template.jinja'
            path.write_bytes(FIXTURE.read_bytes())
            tok = Tokenizer()
            tpl = configured_template({'architecture': 'step35'}, tok, path.parent)
            self.assertIsInstance(tpl, StepTemplate)
            self.assertEqual(tpl.resolve_stop_ids(tok), {257, 258})
            for key, value in (('tokenizer.ggml.eos_token_id', 259), ('tokenizer.ggml.bos_token_id', True),
                               ('tokenizer.ggml.eot_token_id', 259)):
                bad = deepcopy(tok); bad.special_ids[key] = value
                with self.assertRaises(SystemExit): configured_template({'architecture': 'step35'}, bad, path.parent)
            bad = deepcopy(tok); bad.token_types[257] = 1
            with self.assertRaises(SystemExit): configured_template({'architecture': 'step35'}, bad, path.parent)
            bad = deepcopy(tok); bad.pre = 'qwen35'
            with self.assertRaises(SystemExit): configured_template({'architecture': 'step35'}, bad, path.parent)
            path.write_text('unreviewed', encoding='utf8')
            with self.assertRaises(SystemExit): configured_template({'architecture': 'step35'}, tok, path.parent)
            path.unlink()
            with self.assertRaises(SystemExit): configured_template({'architecture': 'step35'}, tok, path.parent)

    def test_shared_settings_and_budget(self):
        svc = self.service()
        self.assertEqual(svc.engine_facts()['architecture'], 'step35')
        caps = svc.reasoning_capabilities()
        self.assertEqual(caps['efforts'], ['low', 'medium', 'high'])
        self.assertTrue(caps['replay_reasoning'])
        self.assertFalse(caps['clear_thinking'])
        for invalid in ({'reasoning_effort': 'none'}, {'clear_thinking': True}):
            with self.assertRaises(ValueError): svc.set_shared(invalid)
        svc.set_shared({'reasoning_effort': 'medium'})
        for api in ('openai', 'anthropic'):
            req = svc.with_shared(self.request(), api)
            self.assertEqual(svc.normalize_request(req, api)[2], {'reasoning_effort': 'medium'})
        svc.reasoning_budget_tokens = 32
        with self.assertRaises(ValueError): svc.reasoning_budget({})
        self.assertIsNone(svc.reasoning_budget({'reasoning_budget_tokens': 0}))


if __name__ == '__main__':
    unittest.main()
