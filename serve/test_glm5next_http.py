"""GLM frontend over real loopback HTTP sockets, with a byte tokenizer/mock engine.

Uses the production Server, handlers and disconnect watcher. No GPU, real glm4
oracle, external MCP calls, telemetry sampler or archive scheduler is involved.
"""
import contextlib
from copy import deepcopy
import http.client
import io
import json
import socket
import sys
import threading
import time
import unittest

from serve.glm5next import GLMTemplate
from serve.server import Server, Service, anthropic_collect, make_handler, openai_collect
from serve.test_glm5next_handlers import LifecycleEngine
from serve.test_glm5next_mcp import TOOLS, call
from serve.test_glm5next_service import FIXTURE, Tokenizer


def wait_for(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError('HTTP lifecycle did not settle before deadline')
        time.sleep(.01)


class QuietEngine(LifecycleEngine):
    """First request stops emitting at the selected phase until the real watcher cancels it."""
    def __init__(self, tok, prefix):
        super().__init__(tok, ['</think>Next'])
        self.prefix = prefix
        self.entered, self.closed = threading.Event(), threading.Event()
        self.calls, self.timed_out = 0, False

    def generate(self, ids, max_new, sampling, cancel, **kwargs):
        self.calls += 1
        if self.calls != 1:
            yield from super().generate(ids, max_new, sampling, cancel, **kwargs)
            return
        self.cancellations.append(cancel)
        self.last_prompt = list(ids)
        try:
            for token in self.tok.encode(self.prefix):
                yield token
            self.entered.set()
            # No pings or writes: cancellation must come from the socket watcher.
            self.timed_out = not cancel.wait(5)
        finally:
            self.closes.append(self.service.fifo.locked())
            self.closed.set()


class ObservedServer(Server):
    # Join request threads on close; the fixture's engine has a bounded wait.
    daemon_threads = False

    def handle_error(self, request, client_address):
        error = sys.exc_info()[1]
        if not isinstance(error, ConnectionError):
            self.errors.append(error)


class GLMHTTPTests(unittest.TestCase):
    def service(self, scripts=None, prefix=None):
        tok = Tokenizer()
        engine = QuietEngine(tok, prefix) if prefix is not None else LifecycleEngine(tok, scripts)
        svc = Service(engine, tok, GLMTemplate(FIXTURE), model_name='glm-5.3-flash')
        svc.api_monitor = True
        engine.service = svc
        return svc

    @contextlib.contextmanager
    def listener(self, svc):
        old_watchers = {t for t in threading.enumerate() if t.name == 'strata-client-watch'}
        httpd = ObservedServer(('127.0.0.1', 0), make_handler(svc))
        httpd.errors = []
        worker = threading.Thread(target=lambda: httpd.serve_forever(poll_interval=.05), daemon=True)
        with contextlib.redirect_stdout(io.StringIO()):
            worker.start()
            try:
                yield httpd
            finally:
                for cancel in svc.engine.cancellations:
                    cancel.set()
                httpd.shutdown()
                httpd.server_close()
                worker.join(5)
                self.assertFalse(worker.is_alive(), 'HTTP listener leaked')
                wait_for(lambda: not any(t not in old_watchers and t.name == 'strata-client-watch'
                                         for t in threading.enumerate()))
                self.assertEqual(httpd.errors, [], 'unexpected request-thread exception')

    @staticmethod
    def path(api):
        return '/v1/chat/completions' if api == 'openai' else '/v1/messages?beta=true'

    @staticmethod
    def request(**options):
        return {'model': 'glm-5.3-flash', 'messages': [{'role': 'user', 'content': 'Hello'}],
                'max_tokens': 2048, **options}

    def post(self, httpd, api, req, expected_status=200):
        client = http.client.HTTPConnection(*httpd.server_address, timeout=5)
        try:
            client.request('POST', self.path(api), json.dumps(req, ensure_ascii=False).encode('utf-8'),
                           {'Content-Type': 'application/json'})
            response = client.getresponse()
            body = response.read().decode('utf-8')
            self.assertEqual(response.status, expected_status, body)
            self.assertEqual(response.version, 10)
            if expected_status != 200 or not req.get('stream'):
                self.assertIn('application/json', response.getheader('Content-Type'))
                self.assertEqual(int(response.getheader('Content-Length')), len(body.encode('utf-8')))
                return json.loads(body)
            self.assertIn('text/event-stream', response.getheader('Content-Type'))
            self.assertEqual(response.getheader('Cache-Control'), 'no-cache')
            data = [line[6:] for line in body.splitlines() if line.startswith('data: ')]
            if api == 'openai':
                self.assertEqual(data.count('[DONE]'), 1)
                self.assertEqual(data.pop(), '[DONE]')
                return openai_collect([json.loads(line) for line in data])
            events = [line[7:] for line in body.splitlines() if line.startswith('event: ')]
            self.assertEqual(events.count('message_stop'), 1)
            self.assertEqual(events[-1], 'message_stop')
            self.assertEqual(len(events), len(data))
            return anthropic_collect(list(zip(events, map(json.loads, data))))
        finally:
            client.close()

    def pending(self, httpd, api, req):
        client = socket.create_connection(httpd.server_address, timeout=5)
        try:
            body = json.dumps(req).encode('utf-8')
            client.sendall((f'POST {self.path(api)} HTTP/1.1\r\nHost: 127.0.0.1\r\n'
                            f'Content-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n').encode() + body)
            return client
        except BaseException:
            client.close()
            raise

    def assert_answer(self, api, answer, text, thought=None):
        if api == 'openai':
            choice = answer['choices'][0]
            self.assertEqual(choice['finish_reason'], 'stop')
            self.assertEqual(choice['message']['content'], text)
            if thought is not None:
                self.assertEqual(choice['message']['reasoning_content'], thought)
        else:
            self.assertEqual(answer['stop_reason'], 'end_turn')
            self.assertEqual([b['text'] for b in answer['content'] if b['type'] == 'text'], [text])
            if thought is not None:
                self.assertEqual([b['thinking'] for b in answer['content'] if b['type'] == 'thinking'], [thought])

    def test_json_and_sse_unicode_reasoning_and_boundaries(self):
        for api in ('openai', 'anthropic'):
            for stream in (False, True):
                with self.subTest(api=api, stream=stream):
                    svc = self.service(['Think</think>Привет 🌍'])
                    with self.listener(svc) as httpd:
                        answer = self.post(httpd, api, self.request(stream=stream))
                        self.assert_answer(api, answer, 'Привет 🌍', 'Think')
                        self.assertNotIn('</think>', json.dumps(answer))
                        self.assertEqual(svc.engine.closes, [True])
                        self.assertEqual(svc.statistics.snapshot()['totals']['requests'], 1)

    def test_tool_result_round_trip_json_and_sse(self):
        for api in ('openai', 'anthropic'):
            for stream in (False, True):
                with self.subTest(api=api, stream=stream):
                    svc = self.service(['Need tool</think>' + call('A'), 'Have result</think>Next'])
                    tool = ({'type': 'function', 'function': deepcopy(TOOLS[0])} if api == 'openai' else
                            {'name': TOOLS[0]['name'], 'input_schema': deepcopy(TOOLS[0]['parameters'])})
                    req = self.request(stream=stream, tools=[tool])
                    with self.listener(svc) as httpd:
                        answer = self.post(httpd, api, req)
                        if api == 'openai':
                            self.assertEqual(answer['choices'][0]['finish_reason'], 'tool_calls')
                            message = answer['choices'][0]['message'];tool_call = message['tool_calls'][0]
                            self.assertEqual(json.loads(tool_call['function']['arguments']), {'text': 'A'})
                            req['messages'] += [message, {'role': 'tool', 'tool_call_id': tool_call['id'], 'content': 'result:A'}]
                        else:
                            self.assertEqual(answer['stop_reason'], 'tool_use')
                            tool_call = next(b for b in answer['content'] if b['type'] == 'tool_use')
                            self.assertEqual(tool_call['input'], {'text': 'A'})
                            req['messages'] += [{'role': 'assistant', 'content': answer['content']},
                                                {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': tool_call['id'],
                                                                            'content': 'result:A'}]}]
                        self.assert_answer(api, self.post(httpd, api, req), 'Next', 'Have result')
                        self.assertIn('<tool_response>result:A</tool_response>', svc.tok.decode(svc.engine.last_prompt))
                        self.assertEqual(svc.engine.closes, [True, True])

    def test_invalid_options_return_json_before_stream_and_engine(self):
        for api in ('openai', 'anthropic'):
            effort = {'reasoning_effort': 'medium'} if api == 'openai' else {'output_config': {'effort': 'medium'}}
            for options in (effort, {'chat_template_kwargs': {'enable_thinking': False}},
                            {'chat_template_kwargs': {'clear_thinking': 'false'}}):
                with self.subTest(api=api, options=options):
                    svc = self.service(['</think>Next'])
                    with self.listener(svc) as httpd:
                        answer = self.post(httpd, api, self.request(stream=True, **options), expected_status=400)
                        self.assertEqual(answer['error']['type'], 'invalid_request_error')
                        self.assertEqual(svc.engine.cancellations, [])
                        self.assertEqual(svc.statistics.snapshot()['totals']['requests'], 0)
                        self.assert_answer(api, self.post(httpd, api, self.request()), 'Next')

    def test_socket_disconnect_in_prefill_reasoning_and_partial_tool(self):
        for api in ('openai', 'anthropic'):
            for stream in (False, True):
                for phase, prefix in (('prefill', ''), ('reasoning', 'Thinking now'),
                                      ('partial_tool', '</think><tool_call>mock__echo<arg_key>text</arg_key><arg_value>A')):
                    with self.subTest(api=api, stream=stream, phase=phase):
                        svc = self.service(prefix=prefix)
                        with self.listener(svc) as httpd:
                            client = self.pending(httpd, api, self.request(stream=stream))
                            try:
                                self.assertTrue(svc.engine.entered.wait(5), 'engine never reached quiet phase')
                                # An open but quiet connection must survive a watcher poll.
                                self.assertFalse(svc.engine.closed.wait(.65), 'watcher cancelled a live client')
                                client.shutdown(socket.SHUT_RDWR)
                            finally:
                                client.close()
                            self.assertTrue(svc.engine.closed.wait(3), 'real socket watcher did not cancel quiet engine')
                            self.assertFalse(svc.engine.timed_out)
                            self.assertTrue(svc.engine.cancellations[0].is_set())
                            wait_for(lambda: not svc.fifo.locked() and not svc.status['busy'] and
                                     svc.api_requests[-1]['state'] == 'disconnected')
                            self.assertEqual(svc.engine.closes, [True], 'engine cleanup must hold FIFO')
                            self.assertEqual(svc.status['queued'], 0)
                            self.assertEqual(svc.statistics.snapshot()['totals']['requests'], 1)
                            self.assertEqual(len(svc.history), 1)
                            self.assertIn(svc.history[-1]['finish'], ('cancel', 'disconnect'))
                            self.assert_answer(api, self.post(httpd, api, self.request()), 'Next')
                            self.assertEqual(svc.engine.closes, [True, True])
                            self.assertEqual(svc.statistics.snapshot()['totals']['requests'], 2)

    def test_queued_socket_disconnect_never_starts_engine(self):
        for api in ('openai', 'anthropic'):
            for stream in (False, True):
                with self.subTest(api=api, stream=stream):
                    svc = self.service(['</think>Next'])
                    with self.listener(svc) as httpd:
                        svc.fifo.acquire()
                        client = None
                        try:
                            client = self.pending(httpd, api, self.request(stream=stream))
                            wait_for(lambda: svc.status['queued'] == 1)
                            client.shutdown(socket.SHUT_RDWR);client.close();client = None
                            wait_for(lambda: svc.api_requests[-1].get('outcome') == 'disconnected')
                            self.assertEqual(svc.engine.cancellations, [])
                        finally:
                            if client is not None:
                                client.close()
                            svc.fifo.release()
                        wait_for(lambda: svc.status['queued'] == 0 and svc.api_requests[-1]['state'] == 'disconnected')
                        self.assertEqual(svc.engine.cancellations, [])
                        self.assertEqual(svc.statistics.snapshot()['totals']['requests'], 0)
                        self.assertEqual(list(svc.history), [])
                        self.assert_answer(api, self.post(httpd, api, self.request()), 'Next')
                        self.assertEqual(svc.engine.closes, [True])


if __name__ == '__main__':
    unittest.main()
