"""Check the live-test client's SSE reader and cross-API history construction."""
from copy import deepcopy
import json
import unittest

from serve.frontend import TemplateRequestError
from serve.minimax_m2_api import MiniMaxAPITemplate
from serve.test_minimax_m2_api import FIXTURE
from tools.check_minimax_m2_live_tools import collect_sse, request, followup


def openai_wire():
    parts = [({'role': 'assistant'}, None), ({'reasoning_content': 'Думаю'}, None),
             ({'tool_calls': [{'index': 0, 'id': 'a', 'function': {'name': 'get_code', 'arguments': '{"city":'}}]}, None),
             ({'tool_calls': [{'index': 0, 'function': {'arguments': '"Уфа","revision":3}'}}]}, None), ({}, 'tool_calls')]
    chunks = [{'choices': [{'delta': delta, 'finish_reason': finish}], 'usage': {'completion_tokens': 12}} for delta, finish in parts]
    return ''.join('data: '+json.dumps(c)+'\n\n' for c in chunks)+'data: [DONE]\n\n'


def anthropic_wire():
    events = [dict(type='message_start', message={'content': [], 'usage': {'input_tokens': 5}}),
              dict(type='content_block_start', index=0, content_block={'type': 'thinking', 'thinking': ''}),
              dict(type='content_block_delta', index=0, delta={'type': 'thinking_delta', 'thinking': 'Думаю'}),
              dict(type='content_block_stop', index=0),
              dict(type='content_block_start', index=1, content_block={'type': 'tool_use', 'id': 'a', 'name': 'get_code', 'input': {}}),
              dict(type='content_block_delta', index=1, delta={'type': 'input_json_delta', 'partial_json': '{"city":"Уфа",'}),
              dict(type='content_block_delta', index=1, delta={'type': 'input_json_delta', 'partial_json': '"revision":3}'}),
              dict(type='content_block_stop', index=1),
              dict(type='message_delta', delta={'stop_reason': 'tool_use'}, usage={'output_tokens': 12}),
              dict(type='message_stop')]
    return ''.join('event: '+e['type']+'\ndata: '+json.dumps(e)+'\n\n' for e in events)


class LiveToolClientTests(unittest.TestCase):
    def test_collect_openai_fragments_and_unicode(self):
        answer = collect_sse('openai', ': heartbeat\n\n'+openai_wire())
        message = answer['choices'][0]['message']
        self.assertEqual(message['reasoning_content'], 'Думаю')
        self.assertEqual(json.loads(message['tool_calls'][0]['function']['arguments']), {'city': 'Уфа', 'revision': 3})
        self.assertEqual(message['tool_calls'][0]['id'], 'a')

    def test_collect_anthropic_blocks_and_unicode(self):
        answer = collect_sse('anthropic', anthropic_wire().replace('\n', '\r\n'))
        self.assertEqual(answer['content'][0]['thinking'], 'Думаю')
        self.assertEqual(answer['content'][1]['input'], {'city': 'Уфа', 'revision': 3})
        self.assertEqual(answer['usage'], {'input_tokens': 5, 'output_tokens': 12})

    def test_reject_broken_sse(self):
        for api, wire in [('openai', openai_wire().replace('data: [DONE]\n\n', '')),
                          ('openai', openai_wire().replace('"index": 0', '"index": 3')),
                          ('anthropic', anthropic_wire().replace('"index": 1', '"index": 7')),
                          ('anthropic', anthropic_wire().replace('event: message_stop', 'event: error'))]:
            with self.subTest(api=api, wire=wire), self.assertRaises(AssertionError):
                collect_sse(api, wire)

    def test_cross_api_prompt_and_reversed_error_results(self):
        template = MiniMaxAPITemplate(FIXTURE)
        calls = [{'id': 'a', 'name': 'get_code', 'input': {'city': 'Oslo', 'revision': 2}},
                 {'id': 'b', 'name': 'get_code', 'input': {'city': 'Уфа', 'revision': 3}}]
        oa = {'choices': [{'message': {'role': 'assistant', 'content': '\n', 'reasoning_content': 'R',
              'tool_calls': [{'id': c['id'], 'type': 'function', 'function': {'name': c['name'], 'arguments': json.dumps(c['input'])}} for c in calls]}}]}
        an = {'content': [{'type': 'thinking', 'thinking': 'R'}, {'type': 'text', 'text': '\n'},
                         *[dict(c, type='tool_use') for c in calls]]}
        for error in (False, True):
            prompts, initial = [], []
            for api, answer in [('openai', oa), ('anthropic', an)]:
                req = request(api, error)
                before = deepcopy(req)
                normalize = template.normalize_openai if api == 'openai' else template.normalize_anthropic
                m, tools, kwargs = normalize(req)
                initial.append(template.render(m, tools=tools, **kwargs))
                next_req = followup(req, api, answer, calls, error)
                self.assertEqual(req, before)
                m, tools, kwargs = normalize(next_req)
                prompts.append(template.render(m, tools=tools, **kwargs))
            self.assertEqual(initial[0], initial[1])
            self.assertEqual(prompts[0], prompts[1])
            self.assertIn('<response>OSLO-4179</response>\n<response>'+('Error: CODE_UNAVAILABLE' if error else 'UFA-9264')+'</response>', prompts[0])

    def test_mismatched_result_id_rejected_before_generation(self):
        template = MiniMaxAPITemplate(FIXTURE)
        req = request('anthropic', True)
        req['messages'] += [{'role': 'assistant', 'content': [{'type': 'tool_use', 'id': 'a', 'name': 'get_code',
                             'input': {'city': 'Oslo', 'revision': 2}}]},
                            {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'wrong', 'content': 'X'}]}]
        with self.assertRaises(TemplateRequestError):
            template.normalize_anthropic(req)


if __name__ == '__main__':
    unittest.main()
