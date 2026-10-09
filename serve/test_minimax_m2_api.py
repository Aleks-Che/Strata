"""MiniMax request contracts and real loopback HTTP/SSE with a scripted engine."""
from copy import deepcopy
import http.client
import json
from pathlib import Path
import socket
import unittest
from unittest.mock import patch

from serve.frontend import TemplateRequestError
from serve.minimax_m2_api import (MiniMaxAPITemplate, StrictMiniMaxDetokenizer,
    openai_to_minimax_messages, anthropic_to_minimax_messages)
from serve.server import ByteTokenizer, Service
from serve.test_step35_http import Engine
from serve import test_glm5next_http as transport
from tools.minimax_m2_tool_cases import TOOLS, BODY, EXPECTED, group, text_group

FIXTURE = Path(__file__).parent/'fixtures/minimax_m27_chat_template.jinja'


class Tokenizer(ByteTokenizer):
    pre = 'minimax-m2'
    SPECIALS = [']~!b[', '[e~[', ']!d~[', ']~b]', '<fim_pad>', '<reponame>',
                '<think>', '</think>', '<minimax:tool_call>', '</minimax:tool_call>']

    def __init__(self):
        self.tokens = [str(i) for i in range(256)]+self.SPECIALS
        self.ids = {s: i for i, s in enumerate(self.tokens)}
        self.token_types = [1]*256+[3]*6+[4]*4
        self.special_ids = {'tokenizer.ggml.'+key+'_token_id': self.ids[value] for key, value in
                            [('bos', ']~!b['), ('eos', '[e~['), ('padding', '[e~['), ('unknown', ']!d~[')]}

    def token_bytes(self, token):
        if token == self.ids['[e~[']:
            raise AssertionError('EOS reached detokenization')
        return bytes([token]) if token < 256 else self.tokens[token].encode('utf-8')


def normalized(req, api):
    return (openai_to_minimax_messages(req) if api == 'openai' else anthropic_to_minimax_messages(req))


class MiniMaxAdapterTests(unittest.TestCase):
    def test_openai_instructions_metadata_history_no_mutation(self):
        req = {'messages': [{'role': 'system', 'content': 'A'}, {'role': 'developer', 'content': [{'type': 'input_text', 'text': 'B'}]},
            {'role': 'user', 'content': 'Q'}, {'role': 'assistant', 'content': '<think> R </think> A ', 'tool_calls': [
                {'id': 'c', 'function': {'name': 'text', 'arguments': '{"s":"Уфа"}'}}]},
            {'role': 'tool', 'tool_call_id': 'c', 'content': 'done'}], 'tools': TOOLS,
            'chat_template_kwargs': {'current_date': '2026-10-09'}}
        before = deepcopy(req)
        messages, tools, kwargs = openai_to_minimax_messages(req)
        self.assertEqual(messages[0], {'role': 'system', 'content': 'A\n\nB', 'current_date': '2026-10-09'})
        self.assertEqual(messages[2]['reasoning_content'], ' R ')
        self.assertEqual(messages[2]['content'], ' A ')
        self.assertEqual(messages[2]['tool_calls'][0]['arguments'], {'s': 'Уфа'})
        rendered = MiniMaxAPITemplate(FIXTURE).render(messages, tools=tools, **kwargs)
        self.assertIn('<response>done</response>', rendered)
        self.assertEqual(req, before)

    def test_anthropic_results_order_errors_and_trailing_user(self):
        req = {'system': [{'type': 'text', 'text': 'S'}], 'messages': [{'role': 'user', 'content': 'Q'},
            {'role': 'assistant', 'content': [{'type': 'thinking', 'thinking': 'R', 'signature': ''}, {'type': 'text', 'text': 'A'},
                {'type': 'tool_use', 'id': 'a', 'name': 'text', 'input': {'s': 'A'}},
                {'type': 'tool_use', 'id': 'b', 'name': 'text', 'input': {'s': 'B'}}]},
            {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'b', 'is_error': True, 'content': [{'type': 'text', 'text': 'bad'}]},
                {'type': 'tool_result', 'tool_use_id': 'a', 'content': 'good'}, {'type': 'text', 'text': 'Continue'}]}]}
        before = deepcopy(req)
        messages, _, _ = anthropic_to_minimax_messages(req)
        self.assertEqual([m['role'] for m in messages], ['system', 'user', 'assistant', 'tool', 'tool', 'user'])
        self.assertEqual([m['content'] for m in messages[-3:]], ['good', 'Error: bad', 'Continue'])
        self.assertEqual(messages[2]['reasoning_content'], '')  # a newer user turn follows
        self.assertEqual(req, before)

    def test_unrepresentable_anthropic_blocks_rejected(self):
        tool = {'type': 'tool_use', 'id': 'c', 'name': 'ping', 'input': {}}
        for blocks in [[tool, {'type': 'text', 'text': 'after'}], [{'type': 'text', 'text': 'A'}, {'type': 'thinking', 'thinking': 'R'}],
                       [{'type': 'redacted_thinking', 'data': 'x'}], [{'type': 'image', 'source': {}}]]:
            with self.subTest(blocks=blocks), self.assertRaises(TemplateRequestError):
                anthropic_to_minimax_messages({'messages': [{'role': 'assistant', 'content': blocks}]})
        with self.assertRaises(TemplateRequestError):
            anthropic_to_minimax_messages({'messages': [{'role': 'assistant', 'content': [tool]}, {'role': 'user', 'content': [
                {'type': 'text', 'text': 'before result'}, {'type': 'tool_result', 'tool_use_id': 'c', 'content': 'R'}]}]})

    def test_fixed_reasoning_and_shared_settings(self):
        tok = Tokenizer(); engine = Engine(tok, stop=tok.ids['[e~['])
        svc = Service(engine, tok, MiniMaxAPITemplate(FIXTURE))
        svc.set_shared({'reasoning_effort': 'high'})
        for api in ('openai', 'anthropic'):
            req = {'messages': [{'role': 'user', 'content': 'Q'}]}
            args = svc.normalize_request(svc.with_shared(req, api), api)
            self.assertTrue(svc.prepare(*args, max_new=32)[1])
            self.assertTrue(svc.normalize_request(req, api))
        for option in [{'thinking': {'type': 'disabled'}}, {'thinking': {'type': 'adaptive'}},
                       {'thinking': {'type': 'enabled', 'budget_tokens': 100}}, {'output_config': {'effort': 'low'}}]:
            with self.assertRaises(TemplateRequestError):
                anthropic_to_minimax_messages({**req, **option})
        with self.assertRaises(ValueError):
            svc.reasoning_budget({'reasoning_budget_tokens': 1})

    def test_schema_dependency_checked_before_generation(self):
        with patch('serve.minimax_m2_api._schemas', side_effect=TemplateRequestError('no schema package')):
            with self.assertRaises(TemplateRequestError):
                openai_to_minimax_messages({'messages': [{'role': 'user', 'content': 'Q'}], 'tools': TOOLS})
        with self.assertRaises(TemplateRequestError):
            openai_to_minimax_messages({'messages': [{'role': 'user', 'content': 'Q'}], 'tools': [
                {'type': 'function', 'function': {'name': 'f', 'parameters': {'type': 'object', 'properties': {'s': {}}}}}]})

    def test_stop_metadata_and_byte_decoder(self):
        tok = Tokenizer()
        self.assertEqual(MiniMaxAPITemplate.resolve_stop_ids(tok), {tok.ids['[e~[']})
        for field, value in [('pre', 'qwen35'), ('special_ids', {}), ('token_types', None)]:
            bad = deepcopy(tok); setattr(bad, field, value)
            with self.assertRaises(TemplateRequestError):
                MiniMaxAPITemplate.resolve_stop_ids(bad)
        bad = deepcopy(tok); bad.special_ids['tokenizer.ggml.eot_token_id'] = bad.ids['<fim_pad>']
        with self.assertRaises(TemplateRequestError):
            MiniMaxAPITemplate.resolve_stop_ids(bad)
        decoder = StrictMiniMaxDetokenizer(tok)
        self.assertEqual(''.join(decoder.push(t) for t in '中文🌍'.encode())+decoder.finish(), '中文🌍')
        decoder = StrictMiniMaxDetokenizer(tok); decoder.push(0xf0)
        with self.assertRaises(UnicodeDecodeError): decoder.finish()
        with self.assertRaises(UnicodeDecodeError): StrictMiniMaxDetokenizer(tok).push(0xff)


class MiniMaxHTTPTests(unittest.TestCase):
    listener = transport.GLMHTTPTests.listener
    path = staticmethod(transport.GLMHTTPTests.path)
    post = transport.GLMHTTPTests.post
    pending = transport.GLMHTTPTests.pending

    @staticmethod
    def request(**options):
        return {'model': 'minimax-m2.7', 'messages': [{'role': 'user', 'content': 'Q'}], 'max_tokens': 1024, **options}

    def service(self, scripts=None, prefix=None):
        tok = Tokenizer()
        engine = Engine(tok, scripts or ['R</think>Next'], prefix, stop=tok.ids['[e~['])
        svc = Service(engine, tok, MiniMaxAPITemplate(FIXTURE), model_name='minimax-m2.7')
        engine.service = svc
        svc.api_monitor = True
        return svc

    @staticmethod
    def api_tools(api):
        return deepcopy(TOOLS) if api == 'openai' else [{'name': t['function']['name'], 'input_schema': t['function']['parameters']} for t in TOOLS]

    def assert_answer(self, api, result, answer, reasoning='R'):
        if api == 'openai':
            message = result['choices'][0]['message']
            self.assertEqual(message['content'] or '', answer)
            self.assertEqual(message.get('reasoning_content', ''), reasoning)
        else:
            self.assertEqual(''.join(b['text'] for b in result['content'] if b['type'] == 'text'), answer)
            self.assertEqual(''.join(b['thinking'] for b in result['content'] if b['type'] == 'thinking'), reasoning)

    def test_json_sse_unicode_usage_eos_and_nonstop_aliases(self):
        wire = 'Думаю</think>Уфа 中文 🌍<fim_pad><reponame>'
        for api in ('openai', 'anthropic'):
            for stream in (False, True):
                svc = self.service([wire])
                with self.subTest(api=api, stream=stream), self.listener(svc) as httpd:
                    result = self.post(httpd, api, self.request(stream=stream))
                    self.assert_answer(api, result, 'Уфа 中文 🌍<fim_pad><reponame>', 'Думаю')
                    output = result['usage']['completion_tokens' if api == 'openai' else 'output_tokens']
                    self.assertEqual(output, len(svc.tok.encode(wire, parse_special=True))+1)
                    inputs = result['usage']['prompt_tokens' if api == 'openai' else 'input_tokens']
                    self.assertEqual(inputs, len(svc.engine.last_prompt))
                    self.assertNotIn('LEAK', json.dumps(result))
                    self.assertEqual(svc.engine.closes, [True])

    def test_multi_tool_result_cycle_ids_types_json_sse(self):
        for api in ('openai', 'anthropic'):
            for stream in (False, True):
                svc = self.service(['R</think>'+group(BODY), 'Done</think>Next'])
                req = self.request(stream=stream, tools=self.api_tools(api))
                with self.subTest(api=api, stream=stream), self.listener(svc) as httpd:
                    result = self.post(httpd, api, req)
                    if api == 'openai':
                        choice = result['choices'][0]
                        self.assertEqual(choice['finish_reason'], 'tool_calls')
                        calls = choice['message']['tool_calls']
                        actual = [(c['function']['name'], json.loads(c['function']['arguments'])) for c in calls]
                        req['messages'] += [choice['message'], *[{'role': 'tool', 'tool_call_id': c['id'], 'content': 'result-'+str(i)}
                            for i, c in reversed(list(enumerate(calls)))]]
                    else:
                        self.assertEqual(result['stop_reason'], 'tool_use')
                        calls = [b for b in result['content'] if b['type'] == 'tool_use']
                        actual = [(c['name'], c['input']) for c in calls]
                        req['messages'] += [{'role': 'assistant', 'content': result['content']}, {'role': 'user', 'content': [
                            {'type': 'tool_result', 'tool_use_id': c['id'], 'content': 'result-'+str(i)} for i, c in reversed(list(enumerate(calls)))]}]
                    self.assertEqual(actual, EXPECTED)
                    self.assertEqual(len({c['id'] for c in calls}), 2)
                    self.assert_answer(api, self.post(httpd, api, req), 'Next', 'Done')
                    prompt = svc.tok.decode(svc.engine.last_prompt)
                    self.assertIn('<response>result-0</response>\n<response>result-1</response>', prompt)
                    self.assertEqual(svc.engine.closes, [True, True])

    def test_truncated_groups_length_eos_and_stop(self):
        wire = text_group('partial')
        for api in ('openai', 'anthropic'):
            for stream in (False, True):
                for end in ('eos', 'length', 'stop'):
                    raw = 'R</think>'+wire[:-1] if end == 'eos' else 'R</think>'+wire+'LEAK'
                    svc = self.service([raw]); extra = {}
                    if end == 'length':
                        # GROUP_END is one special token: stop before that token,
                        # not inside its spelling with a differently tokenized cap.
                        extra['max_tokens'] = len(svc.tok.encode('R</think>'+wire[:-len('</minimax:tool_call>')], parse_special=True))
                    if end == 'stop':
                        extra['stop' if api == 'openai' else 'stop_sequences'] = ['</parameter>']
                    with self.subTest(api=api, stream=stream, end=end), self.listener(svc) as httpd:
                        result = self.post(httpd, api, self.request(stream=stream, tools=self.api_tools(api), **extra))
                        expected = wire[:wire.index('</parameter>')] if end == 'stop' else wire[:-1]
                        if end == 'length':
                            expected = wire[:-len('</minimax:tool_call>')]
                        self.assert_answer(api, result, expected)
                        if api == 'openai':
                            self.assertNotIn('tool_calls', result['choices'][0]['message'])
                            self.assertEqual(result['choices'][0]['finish_reason'], 'length' if end == 'length' else 'stop')
                        else:
                            self.assertEqual(result['stop_reason'], {'eos': 'end_turn', 'length': 'max_tokens', 'stop': 'stop_sequence'}[end])

    def test_tool_choice_none_preserves_literal_group(self):
        for api in ('openai', 'anthropic'):
            for stream in (False, True):
                wire = text_group('x'); svc = self.service(['R</think>'+wire])
                with self.listener(svc) as httpd:
                    result = self.post(httpd, api, self.request(stream=stream, tools=self.api_tools(api),
                        tool_choice='none' if api == 'openai' else {'type': 'none'}))
                    self.assert_answer(api, result, wire)
                    self.assertNotIn('# Tools', svc.tok.decode(svc.engine.last_prompt))

    def test_invalid_requests_400_before_generation_and_recovery(self):
        common = [{'temperature': True}, {'temperature': float('nan')}, {'top_p': 1e-100}, {'seed': -1},
                  {'top_k': 200065}, {'min_p': .1}, {'max_tokens': True}, {'stream': 'yes'},
                  {'reasoning_budget_tokens': 1}, {'chat_template_kwargs': {'enable_thinking': False}},
                  {'messages': [{'role': 'user', 'content': [{'type': 'image', 'source': {}}]}]},
                  {'tools': [{'bad': True}]}]
        for api in ('openai', 'anthropic'):
            extras = common+([{'tool_choice': 'required'}, {'parallel_tool_calls': False}, {'stop': ['']},
                             {'reasoning_effort': 'low'}, {'max_tokens': 32, 'max_completion_tokens': 64}]
                            if api == 'openai' else [{'tool_choice': {'type': 'any'}}, {'thinking': {'type': 'disabled'}}, {'stop_sequences': ['']}])
            svc = self.service()
            with self.listener(svc) as httpd:
                for extra in extras:
                    with self.subTest(api=api, extra=extra):
                        self.post(httpd, api, self.request(**{'stream': True, **extra}), expected_status=400)
                        self.assertEqual(svc.engine.calls, 0)
                self.assert_answer(api, self.post(httpd, api, self.request()), 'Next')

    def test_missing_result_does_not_start_next_generation(self):
        for api in ('openai', 'anthropic'):
            svc = self.service()
            if api == 'openai':
                history = [{'role': 'assistant', 'tool_calls': [{'id': 'a', 'name': 'ping', 'arguments': {}}]},
                           {'role': 'tool', 'tool_call_id': 'wrong', 'content': 'R'}]
            else:
                history = [{'role': 'assistant', 'content': [{'type': 'tool_use', 'id': 'a', 'name': 'ping', 'input': {}}]},
                           {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'wrong', 'content': 'R'}]}]
            with self.listener(svc) as httpd:
                self.post(httpd, api, self.request(messages=history, stream=True), expected_status=400)
                self.assertEqual(svc.engine.calls, 0)

    def test_disconnect_prefill_reasoning_tool_and_queued(self):
        for api in ('openai', 'anthropic'):
            for stream in (False, True):
                for prefix in ('', 'Thinking', 'R</think><minimax:tool_call><invoke name="text">'):
                    svc = self.service(prefix=prefix)
                    with self.listener(svc) as httpd:
                        client = self.pending(httpd, api, self.request(stream=stream, tools=self.api_tools(api)))
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

    def test_invalid_and_incomplete_utf8_error_and_recovery(self):
        for api in ('openai', 'anthropic'):
            for stream in (False, True):
                for broken in ([0xff], [0xf0, 0x9f]):
                    svc = self.service(['unused', 'R</think>Next'])
                    svc.engine.scripts[0] = svc.tok.encode('R</think>')+broken+[svc.tok.ids['[e~[']]
                    with self.listener(svc) as httpd:
                        client = http.client.HTTPConnection(*httpd.server_address, timeout=5)
                        try:
                            client.request('POST', self.path(api), json.dumps(self.request(stream=stream)), {'Content-Type': 'application/json'})
                            response = client.getresponse(); body = response.read().decode()
                            self.assertEqual(response.status, 200 if stream else 400, body)
                            self.assertIn('error', body)
                            self.assertNotIn('�', body)
                            if stream:
                                self.assertNotIn('"finish_reason": "stop"', body)
                                self.assertNotIn('event: message_stop', body)
                        finally:
                            client.close()
                        self.assertEqual(svc.history[-1]['finish'], 'error')
                        self.assertFalse(svc.fifo.locked())
                        self.assert_answer(api, self.post(httpd, api, self.request()), 'Next')

    def test_length_after_complete_call_keeps_limit_reason(self):
        wire = 'R</think>'+text_group('x')
        for api in ('openai', 'anthropic'):
            for stream in (False, True):
                svc = self.service([wire])
                with self.listener(svc) as httpd:
                    result = self.post(httpd, api, self.request(stream=stream, tools=self.api_tools(api),
                        max_tokens=len(svc.tok.encode(wire, parse_special=True))))
                    self.assertEqual(result['choices'][0]['finish_reason'] if api == 'openai' else result['stop_reason'],
                                     'length' if api == 'openai' else 'max_tokens')


if __name__ == '__main__':
    unittest.main()
