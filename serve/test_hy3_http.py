"""Hy3 request adapters, EOG and real loopback JSON/SSE with a scripted engine."""
from copy import deepcopy
import json
from pathlib import Path
import socket
import tempfile
import unittest

from serve.server import ByteTokenizer, Service, configured_template
from serve.hy3 import (Hy3Template, THINK_START, THINK_END, openai_to_hy3_messages,
                      anthropic_to_hy3_messages)
from serve.test_hy3 import FIXTURE, call, group
from serve.test_step35_http import Engine
from serve import test_glm5next_http as transport


class Tokenizer(ByteTokenizer):
    pre = 'hunyuan-dense'
    CONTROL = ['<｜hy_begin_of_sentence:opensource｜>', '<｜hy_eos:opensource｜>',
               '<｜hy_pad:opensource｜>', '<｜hy_Assistant:opensource｜>', '<｜hy_User:opensource｜>',
               '<｜reasoning_mode:opensource｜>', '<｜hy_place▁holder▁no▁8｜>']
    USER = [THINK_START, THINK_END, '<tool_calls:opensource>', '</tool_calls:opensource>',
            '<tool_call:opensource>', '</tool_call:opensource>', '<tool_sep:opensource>',
            '<arg_key:opensource>', '</arg_key:opensource>', '<arg_value:opensource>', '</arg_value:opensource>',
            '<tool_responses:opensource>', '</tool_responses:opensource>',
            '<tool_response:opensource>', '</tool_response:opensource>']

    def __init__(self):
        self.tokens = [str(i) for i in range(256)]+self.CONTROL+self.USER
        self.ids = {t: i for i, t in enumerate(self.tokens)}
        self.token_types = [1]*256+[3]*len(self.CONTROL)+[4]*len(self.USER)
        self.special_ids = {'tokenizer.ggml.bos_token_id': 256, 'tokenizer.ggml.eos_token_id': 257,
                           'tokenizer.ggml.padding_token_id': 258, 'tokenizer.ggml.seperator_token_id': 259}

    def token_bytes(self, token_id):
        if token_id == 257:
            raise AssertionError('Hy3 EOS reached detokenization')
        return self.tokens[token_id].encode() if token_id >= 256 else bytes([token_id])


class AdapterTests(unittest.TestCase):
    def test_openai_history_ids_arguments_and_no_mutation(self):
        req = {'messages': [{'role': 'developer', 'content': [{'type': 'text', 'text': 'a'}, {'type': 'input_text', 'text': 'b'}]},
            {'role': 'assistant', 'content': None, 'reasoning_content': 'r', 'tool_calls': [
                {'id': 'c1', 'type': 'function', 'function': {'name': 'f', 'arguments': '{"s":"中","n":2}'}}]},
            {'role': 'tool', 'tool_call_id': 'c1', 'content': 'result'}],
            'reasoning': {'effort': 'high'}, 'reasoning_effort': 'low',
            'chat_template_kwargs': {'reasoning_effort': 'no_think', 'preserved_thinking': False}}
        before = deepcopy(req)
        messages, tools, options = openai_to_hy3_messages(req)
        self.assertEqual(messages[0], {'role': 'system', 'content': 'ab'})
        self.assertEqual(messages[1]['tool_calls'][0]['function']['arguments'], {'s': '中', 'n': 2})
        self.assertEqual(messages[-1]['tool_call_id'], 'c1')
        self.assertEqual(options, {'reasoning_effort': 'no_think', 'preserved_thinking': False})
        self.assertEqual(req, before)

    def test_anthropic_mixed_blocks_keep_order_ids_and_errors(self):
        req = {'system': [{'type': 'text', 'text': 'system'}], 'messages': [
            {'role': 'assistant', 'content': [{'type': 'thinking', 'thinking': 'r'}, {'type': 'text', 'text': 'before'},
                {'type': 'tool_use', 'id': 'c1', 'name': 'f', 'input': {'n': 2}}, {'type': 'text', 'text': 'after'}]},
            {'role': 'user', 'content': [{'type': 'text', 'text': 'first'},
                {'type': 'tool_result', 'tool_use_id': 'c1', 'content': [{'type': 'text', 'text': 'bad'}], 'is_error': True},
                {'type': 'text', 'text': 'last'}]}], 'thinking': {'type': 'enabled'}, 'output_config': {'effort': 'low'}}
        before = deepcopy(req)
        messages, _, options = anthropic_to_hy3_messages(req, False)
        self.assertEqual([m['role'] for m in messages], ['system', 'assistant', 'assistant', 'user', 'tool', 'user'])
        self.assertEqual(messages[1]['reasoning_content'], 'r')
        self.assertEqual(messages[1]['tool_calls'][0]['id'], 'c1')
        self.assertEqual(messages[2]['content'], 'after')
        self.assertEqual(messages[4], {'role': 'tool', 'tool_call_id': 'c1', 'content': 'Error: bad'})
        self.assertEqual(options, {'reasoning_effort': 'low'})
        self.assertEqual(req, before)

    def test_effort_defaults_prefix_and_shared_precedence(self):
        tok = Tokenizer()
        svc = Service(Engine(tok, ['Next'], stop=257), tok, Hy3Template(FIXTURE))
        for api in ('openai', 'anthropic'):
            for effort, thinking in [('no_think', False), ('low', True), ('high', True)]:
                req = {'messages': [{'role': 'user', 'content': 'hi'}], 'chat_template_kwargs': {'reasoning_effort': effort}}
                normalized = svc.normalize_request(req, api)
                self.assertEqual(svc.prepare(*normalized, max_new=32)[1], thinking)
            normal = svc.normalize_request({'messages': [{'role': 'user', 'content': 'hi'}]}, api)
            self.assertFalse(svc.prepare(*normal, max_new=32)[1])
        svc.set_shared({'reasoning_effort': 'high'})
        req = {'messages': [{'role': 'user', 'content': 'hi'}], 'thinking': {'type': 'disabled'}}
        self.assertEqual(svc.normalize_request(svc.with_shared(req, 'anthropic'), 'anthropic')[2], {'reasoning_effort': 'no_think'})
        req['thinking'] = {'type': 'enabled'}
        self.assertEqual(anthropic_to_hy3_messages(req)[2], {'reasoning_effort': 'high'})
        with self.assertRaises(ValueError):
            anthropic_to_hy3_messages({**req, 'output_config': {'effort': 'no_think'}})
        svc.reasoning_budget_tokens = 32
        with self.assertRaises(ValueError): svc.reasoning_budget({})

    def test_invalid_messages_ids_media_and_json(self):
        for messages in ([], 'x', [{'role': 'user', 'content': [{'type': 'image', 'text': 'x'}]}],
                         [{'role': 'tool', 'content': 'x'}],
                         [{'role': 'assistant', 'tool_calls': [{'id': 'c', 'function': {'name': 'f', 'arguments': '{"x":1,"x":2}'}}]}]):
            with self.subTest(messages=messages), self.assertRaises(ValueError):
                openai_to_hy3_messages({'messages': messages})

    def test_profile_admission_resolves_only_actual_eos(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'chat_template.jinja'; path.write_bytes(FIXTURE.read_bytes())
            tok = Tokenizer()
            self.assertEqual(configured_template({'architecture': 'hy_v3'}, tok, path.parent).resolve_stop_ids(tok), {257})
            for key, value in [('eos', 262), ('padding', True), ('eod', 262), ('seperator', 256)]:
                bad = deepcopy(tok); bad.special_ids['tokenizer.ggml.'+key+'_token_id'] = value
                with self.assertRaises(SystemExit): configured_template({'architecture': 'hy_v3'}, bad, path.parent)
            for spelling in (THINK_START, '<｜hy_eos:opensource｜>'):
                bad = deepcopy(tok); bad.token_types[bad.ids[spelling]] = 1
                with self.assertRaises(SystemExit): configured_template({'architecture': 'hy_v3'}, bad, path.parent)
            bad = deepcopy(tok); bad.pre = 'qwen35'
            with self.assertRaises(SystemExit): configured_template({'architecture': 'hy_v3'}, bad, path.parent)
            path.write_text('unreviewed', encoding='utf8')
            with self.assertRaises(SystemExit): configured_template({'architecture': 'hy_v3'}, tok, path.parent)


class Hy3HTTPTests(unittest.TestCase):
    listener = transport.GLMHTTPTests.listener
    path = staticmethod(transport.GLMHTTPTests.path)
    post = transport.GLMHTTPTests.post
    pending = transport.GLMHTTPTests.pending
    def assert_answer(self, api, answer, text, thought=None):
        if api == 'openai':
            message = answer['choices'][0]['message']
            self.assertEqual(message['content'], text)
            self.assertEqual(message.get('reasoning_content', ''), thought or '')
        else:
            self.assertEqual(''.join(b['text'] for b in answer['content'] if b['type'] == 'text'), text)
            self.assertEqual(''.join(b['thinking'] for b in answer['content'] if b['type'] == 'thinking'), thought or '')

    @staticmethod
    def request(**options):
        return {'model': 'hy3', 'messages': [{'role': 'user', 'content': 'Hello'}], 'max_tokens': 1024, **options}

    def service(self, scripts=None, prefix=None):
        tok = Tokenizer()
        engine = Engine(tok, scripts or ['Next'], prefix=prefix, stop=257)
        svc = Service(engine, tok, Hy3Template(FIXTURE), model_name='hy3')
        svc.api_monitor = True
        engine.service = svc
        return svc

    def test_json_sse_reasoning_unicode_usage_eog_pad_placeholder(self):
        for api in ('openai', 'anthropic'):
            for stream in (False, True):
                for effort in ('no_think', 'low', 'high'):
                    reasoning = 'Думаю🌍' if effort != 'no_think' else ''
                    svc = self.service([reasoning+(THINK_END if reasoning else '')+'Привет'])
                    svc.engine.scripts[0][-5:-5] = [258, 262]  # PAD and placeholder precede EOS.
                    with self.subTest(api=api, stream=stream, effort=effort), self.listener(svc) as httpd:
                        answer = self.post(httpd, api, self.request(stream=stream, chat_template_kwargs={'reasoning_effort': effort}))
                        self.assert_answer(api, answer, 'Привет'+svc.tok.tokens[258]+svc.tok.tokens[262], reasoning)
                        self.assertNotIn('LEAK', json.dumps(answer))
                        self.assertEqual(svc.engine.closes, [True])
                        self.assertGreater(answer['usage']['completion_tokens' if api == 'openai' else 'output_tokens'], 0)

    def test_tool_result_roundtrip_both_apis_and_streams(self):
        function = {'name': 'f', 'parameters': {'type': 'object', 'properties': {'s': {'type': 'string'}, 'n': {'type': 'integer'}}}}
        for api in ('openai', 'anthropic'):
            for stream in (False, True):
                svc = self.service([group(call({'s': 'Привет', 'n': '2'})), 'Next'])
                tool = {'type': 'function', 'function': function} if api == 'openai' else {'name': 'f', 'input_schema': function['parameters']}
                req = self.request(stream=stream, tools=[tool])
                with self.subTest(api=api, stream=stream), self.listener(svc) as httpd:
                    answer = self.post(httpd, api, req)
                    if api == 'openai':
                        self.assertEqual(answer['choices'][0]['finish_reason'], 'tool_calls')
                        msg = answer['choices'][0]['message']; c = msg['tool_calls'][0]
                        self.assertEqual(json.loads(c['function']['arguments']), {'s': 'Привет', 'n': 2})
                        req['messages'] += [msg, {'role': 'tool', 'tool_call_id': c['id'], 'content': 'result:A'}]
                    else:
                        self.assertEqual(answer['stop_reason'], 'tool_use')
                        c = next(b for b in answer['content'] if b['type'] == 'tool_use')
                        self.assertEqual(c['input'], {'s': 'Привет', 'n': 2})
                        req['messages'] += [{'role': 'assistant', 'content': answer['content']}, {'role': 'user', 'content': [
                            {'type': 'tool_result', 'tool_use_id': c['id'], 'content': 'result:A'}]}]
                    self.assert_answer(api, self.post(httpd, api, req), 'Next')
                    self.assertIn('<tool_response:opensource>\nresult:A', svc.tok.decode(svc.engine.last_prompt))

    def test_stop_length_and_invalid_options_recovery(self):
        for api in ('openai', 'anthropic'):
            for stream in (False, True):
                for length in (False, True):
                    svc = self.service(['A!STOP!LEAK'])
                    extra = {'max_tokens': 1} if length else {('stop' if api == 'openai' else 'stop_sequences'): ['!STOP!']}
                    with self.listener(svc) as httpd:
                        answer = self.post(httpd, api, self.request(stream=stream, **extra))
                        self.assert_answer(api, answer, 'A')
                        self.assertEqual(answer['choices'][0]['finish_reason'] if api == 'openai' else answer['stop_reason'],
                            ('length' if length else 'stop') if api == 'openai' else ('max_tokens' if length else 'stop_sequence'))
            for extra in ({'reasoning_budget_tokens': 8}, {'chat_template_kwargs': {'reasoning_effort': 'medium'}},
                          {'chat_template_kwargs': {'clear_thinking': True}}, {'stop': ['']}, {'preserved_thinking': 'yes'}):
                svc = self.service()
                with self.listener(svc) as httpd:
                    self.post(httpd, api, self.request(stream=True, **extra), expected_status=400)
                    self.assertEqual(svc.engine.calls, 0)
                    self.assert_answer(api, self.post(httpd, api, self.request()), 'Next')

    def test_disconnect_during_prefill_reasoning_partial_tool(self):
        for api in ('openai', 'anthropic'):
            for stream in (False, True):
                for prefix in ('', THINK_START+'Thinking', '<tool_calls:opensource>'+call({'s': 'partial'})):
                    svc = self.service(prefix=prefix)
                    with self.listener(svc) as httpd:
                        client = self.pending(httpd, api, self.request(stream=stream))
                        try:
                            self.assertTrue(svc.engine.entered.wait(5))
                            client.shutdown(socket.SHUT_RDWR)
                        finally:
                            client.close()
                        self.assertTrue(svc.engine.closed.wait(3))
                        self.assertFalse(svc.engine.timed_out)
                        self.assertTrue(svc.engine.cancellations[0].is_set())
                        transport.wait_for(lambda: not svc.fifo.locked() and not svc.status['busy'])
                        self.assert_answer(api, self.post(httpd, api, self.request()), 'Next')
                        self.assertEqual(svc.engine.closes, [True, True])

    def test_queued_disconnect(self):
        for api in ('openai', 'anthropic'):
            svc = self.service()
            with self.listener(svc) as httpd:
                svc.fifo.acquire()
                try:
                    client = self.pending(httpd, api, self.request(stream=True))
                    transport.wait_for(lambda: svc.status['queued'] == 1)
                    client.shutdown(socket.SHUT_RDWR); client.close()
                    transport.wait_for(lambda: svc.api_requests[-1].get('outcome') == 'disconnected')
                finally:
                    svc.fifo.release()
                transport.wait_for(lambda: svc.status['queued'] == 0)
                self.assertEqual(svc.engine.calls, 0)
                self.assert_answer(api, self.post(httpd, api, self.request()), 'Next')


if __name__ == '__main__':
    unittest.main()
