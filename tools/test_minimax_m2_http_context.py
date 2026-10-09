"""CPU HTTP boundary regressions; native model coverage is opt-in in the checker."""
import unittest

from serve.server import CTX_SLACK
from serve import test_minimax_m2_api as api_tests


class MiniMaxHTTPContextTests(unittest.TestCase):
    listener = api_tests.MiniMaxHTTPTests.listener
    service = api_tests.MiniMaxHTTPTests.service
    path = staticmethod(api_tests.MiniMaxHTTPTests.path)
    post = api_tests.MiniMaxHTTPTests.post

    @staticmethod
    def request_at(svc, count):
        req = {'model': 'minimax-m2.7', 'messages': [{'role': 'user', 'content': 'x'}], 'max_tokens': 8}
        def ids():
            messages, tools, kwargs = svc.normalize_request(req, 'openai')
            return svc.tok.encode(svc.template.render(messages, tools=tools, **kwargs), parse_special=True)
        req['messages'][0]['content'] = 'x'*(count-len(ids())+1)
        assert len(ids()) == count
        return req

    def test_boundary_rejection_does_not_generate_or_truncate(self):
        for api in ('openai', 'anthropic'):
            for stream in (False, True):
                svc = self.service(['Long reasoning that does not reach the answer in eight tokens.'])
                svc.engine.max_context = 4096
                req = self.request_at(svc, 4096-CTX_SLACK-8)
                with self.subTest(api=api, stream=stream), self.listener(svc) as server:
                    for invalid in ({**req, 'max_tokens': 9},
                                    {**self.request_at(svc, 4096-CTX_SLACK), 'max_tokens': 0}):
                        reply = self.post(server, api, {**invalid, 'stream': stream}, expected_status=400)
                        self.assertIn('context', reply['error']['message'])
                        self.assertEqual(svc.engine.calls, 0)
                    reply = self.post(server, api, {**req, 'stream': stream})
                    self.assertEqual(svc.engine.calls, 1)
                    self.assertEqual(len(svc.engine.last_prompt), 4080)
                    self.assertEqual(reply['usage']['prompt_tokens' if api == 'openai' else 'input_tokens'], 4080)
                    self.assertEqual(reply['usage']['completion_tokens' if api == 'openai' else 'output_tokens'], 8)
                    self.assertEqual(reply['choices'][0]['finish_reason'] if api == 'openai' else reply['stop_reason'],
                                     'length' if api == 'openai' else 'max_tokens')

    def test_implicit_output_budget_uses_exact_remaining_room(self):
        for api in ('openai', 'anthropic'):
            svc = self.service(['Continuing reasoning for at least eight tokens.'])
            svc.engine.max_context = 4096
            req = self.request_at(svc, 4096-CTX_SLACK-8)
            with self.subTest(api=api), self.listener(svc) as server:
                for maximum in (None, 0):
                    body = {**req, 'stream': True}
                    if maximum is None:
                        del body['max_tokens']
                    else:
                        body['max_tokens'] = maximum
                    reply = self.post(server, api, body)
                    self.assertEqual(len(svc.engine.last_prompt), 4080)
                    self.assertEqual(reply['usage']['completion_tokens' if api == 'openai' else 'output_tokens'], 8)


if __name__ == '__main__':
    unittest.main()
