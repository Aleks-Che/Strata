"""Real request handlers/JSON/SSE with in-memory IO, byte tokenizer and mock engine.

No HTTP listener, client-disconnect watcher, real MCP server or GPU is exercised.
"""
import contextlib
from copy import deepcopy
from email.message import Message
import io
import json
import threading
import unittest

from serve.glm5next import GLMTemplate
from serve.server import Service, anthropic_collect, make_handler, openai_collect
from serve.test_glm5next_mcp import Engine, Hub, TOOLS, call
from serve.test_glm5next_service import FIXTURE, Tokenizer


class HandlerHub(Hub):
    def wait(self, timeout):
        pass

    def template_tools(self, exclude=()):
        self.excluded = set(exclude)
        return [deepcopy(t) for t in TOOLS if t['name'] not in exclude]


class LifecycleEngine(Engine):
    def __init__(self, tok, scripts):
        super().__init__(tok, scripts, 154829)
        self.cancellations, self.closes = [], []

    def generate(self, ids, max_new, sampling, cancel, **kwargs):
        self.cancellations.append(cancel)
        try:
            yield from super().generate(ids, max_new, sampling, cancel, **kwargs)
        finally:
            self.closes.append(self.service.fifo.locked())


class DisconnectWriter(io.BytesIO):
    def __init__(self, marker):
        super().__init__()
        self.marker, self.disconnected = marker, False

    def write(self, data):
        if self.marker in data:
            self.disconnected = True
            raise BrokenPipeError('scripted client disconnect')
        return super().write(data)


class GLMHandlerTests(unittest.TestCase):
    def service(self, scripts):
        tok = Tokenizer()
        return Service(Engine(tok, scripts, 154829), tok, GLMTemplate(FIXTURE))

    def post(self, svc, path, req, writer=None, watch=None):
        handler = object.__new__(make_handler(svc))
        body = json.dumps(req).encode()
        handler.path = path
        handler.headers = Message()
        handler.headers['Content-Type'] = 'application/json'
        handler.headers['Content-Length'] = str(len(body))
        handler.headers['Host'] = 'localhost'
        handler.rfile = io.BytesIO(body)
        handler.wfile = io.BytesIO() if writer is None else writer
        statuses, headers = [], {}
        handler.send_response = statuses.append
        handler.send_header = lambda name, value: headers.update({name: value})
        handler.end_headers = lambda: None
        handler._watch_client = watch or (lambda cancel: None)
        with contextlib.redirect_stdout(io.StringIO()):
            handler.do_POST()
        self.assertEqual(len(statuses), 1)
        return statuses[0], headers, handler.wfile.getvalue().decode('utf-8')

    def request(self, api, **options):
        return {'messages': [{'role': 'user', 'content': 'Hello'}],
                'max_tokens': 2048, **options}

    def path(self, api):
        return '/v1/chat/completions' if api == 'openai' else '/v1/messages?beta=true'

    def lifecycle_service(self, scripts):
        tok = Tokenizer()
        engine = LifecycleEngine(tok, scripts)
        svc = Service(engine, tok, GLMTemplate(FIXTURE))
        engine.service = svc
        return svc

    def test_cancel_before_generation_finishes_both_api_responses(self):
        for api in ('openai', 'anthropic'):
            for stream in (False, True):
                for mcp in ((False, True) if api == 'openai' else (False,)):
                    with self.subTest(api=api, stream=stream, mcp=mcp):
                        svc = self.lifecycle_service(['</think>Must not run'])
                        svc.mcp = HandlerHub()
                        status, _, body = self.post(svc, self.path(api), self.request(
                            api, stream=stream, strata_mcp=mcp), watch=lambda cancel: cancel.set())
                        self.assertEqual(status, 200)
                        if stream:
                            data = [line[6:] for line in body.splitlines() if line.startswith('data: ')]
                            if api == 'openai':
                                self.assertEqual(data.pop(), '[DONE]')
                                result = openai_collect([json.loads(line) for line in data])
                            else:
                                events = [line[7:] for line in body.splitlines() if line.startswith('event: ')]
                                self.assertEqual(events[-1], 'message_stop')
                                result = anthropic_collect(list(zip(events, map(json.loads, data))))
                        else:
                            result = json.loads(body)
                        if api == 'openai':
                            self.assertEqual(result['choices'][0]['finish_reason'], 'stop')
                            self.assertEqual(result['usage']['completion_tokens'], 0)
                        else:
                            self.assertEqual(result['stop_reason'], 'end_turn')
                            self.assertEqual(result['usage']['output_tokens'], 0)
                        self.assertEqual(svc.engine.cancellations, [])
                        self.assertEqual(svc.mcp.calls, [])
                        self.assertEqual(svc.statistics.snapshot()['totals']['requests'], 0)
                        self.assertEqual(list(svc.history), [])
                        self.assertEqual(svc.status['queued'], 0)
                        self.assertFalse(svc.fifo.locked())

    def test_cancel_mcp_continuation_does_not_count_previous_round_twice(self):
        first = 'Need tool</think>' + call('A')
        svc = self.lifecycle_service([first, '</think>Must not run'])
        svc.mcp = HandlerHub()
        cancellations, preparations = [], []
        original = svc.prepare

        def prepare(*args, **kwargs):
            preparations.append(args[0])
            if len(preparations) == 2:
                cancellations[0].set()
            return original(*args, **kwargs)

        svc.prepare = prepare
        status, _, body = self.post(svc, self.path('openai'), self.request('openai', strata_mcp=True),
                                    watch=cancellations.append)
        self.assertEqual(status, 200)
        result = json.loads(body)
        self.assertEqual(result['usage']['completion_tokens'], len(svc.tok.encode(first)) + 1)
        self.assertEqual(len(svc.engine.cancellations), 1)
        self.assertEqual(svc.mcp.calls, [('mock__echo', {'text': 'A'})])
        self.assertFalse(svc.status['busy'])
        self.assertFalse(svc.fifo.locked())

    def test_request_cancelled_while_waiting_for_fifo_never_runs_engine(self):
        for api in ('openai', 'anthropic'):
            with self.subTest(api=api):
                svc = self.lifecycle_service(['</think>Next'])
                cancel, waiting = threading.Event(), threading.Event()
                gate = svc.fifo

                class ObservedLock:
                    def __enter__(self):
                        waiting.set()
                        gate.acquire()
                        return self

                    def __exit__(self, *args):
                        gate.release()

                    def __getattr__(self, name):
                        return getattr(gate, name)

                svc.fifo = ObservedLock()
                answers, errors = [], []

                def request():
                    try:
                        if api == 'openai':
                            from serve.server import openai_chunks
                            answers.append(openai_collect(openai_chunks(svc, {}, [1], True, None, 20, cancel)))
                        else:
                            from serve.server import anthropic_events
                            answers.append(anthropic_collect(anthropic_events(svc, {}, [1], True, None, 20, cancel)))
                    except Exception as exc:
                        errors.append(exc)

                gate.acquire()
                worker = threading.Thread(target=request, daemon=True)
                worker.start()
                try:
                    self.assertTrue(waiting.wait(5), 'request did not reach FIFO')
                    self.assertEqual(svc.status['queued'], 1)
                    cancel.set()
                    self.assertEqual(svc.engine.cancellations, [])
                finally:
                    gate.release()
                    worker.join(5)
                self.assertFalse(worker.is_alive())
                self.assertEqual(errors, [])
                result = answers[0]
                self.assertEqual(result['choices'][0]['finish_reason'] if api == 'openai' else result['stop_reason'],
                                 'stop' if api == 'openai' else 'end_turn')
                self.assertEqual(svc.engine.cancellations, [])
                self.assertEqual(svc.statistics.snapshot()['totals']['requests'], 0)
                self.assertEqual(svc.status['queued'], 0)
                status, _, body = self.post(svc, self.path(api), self.request(api))
                self.assertEqual(status, 200)
                self.assertIn('Next', body)
                self.assertEqual(svc.engine.closes, [True])

    def test_cancelled_done_event_is_emitted_after_releasing_fifo(self):
        svc = self.lifecycle_service(['</think>Must not run'])
        cancel = threading.Event()
        cancel.set()
        events = svc.run([1], True, None, 20, {}, cancel)
        try:
            kind, done = next(events)
            self.assertEqual((kind, done['finish'], done['completion_tokens']), ('done', 'cancel', 0))
            self.assertFalse(svc.fifo.locked(), 'slow delivery of terminal event must not hold engine slot')
            self.assertEqual(svc.engine.cancellations, [])
            self.assertEqual(list(events), [])
        finally:
            events.close()

    def test_disconnect_during_prefill_reasoning_or_incomplete_tool_releases_engine(self):
        for api in ('openai', 'anthropic'):
            for phase in ('prefill', 'reasoning', 'partial-tool'):
                for monitor in (False, True):
                    with self.subTest(api=api, phase=phase, monitor=monitor):
                        first = ('</think><tool_call>mock__echo<arg_key>text</arg_key><arg_value>A'
                                 if phase == 'partial-tool' else 'Long thought</think>Answer')
                        svc = self.lifecycle_service([first, '</think>Next'])
                        svc.api_monitor = monitor
                        svc.mcp = HandlerHub()
                        marker = b'reasoning_content' if api == 'openai' else b'thinking_delta'
                        if phase != 'reasoning':
                            svc.engine.scripts[0].insert(0 if phase == 'prefill' else len(svc.engine.scripts[0]) - 1, None)
                            marker = b': keep-alive'
                        writer = DisconnectWriter(marker)
                        self.post(svc, self.path(api), self.request(api, stream=True, strata_mcp=api == 'openai'),
                                  writer=writer)
                        self.assertTrue(writer.disconnected)
                        self.assertTrue(svc.engine.cancellations[0].is_set())
                        self.assertEqual(svc.engine.closes, [True], 'engine must drain while FIFO is held')
                        self.assertFalse(svc.fifo.locked())
                        self.assertFalse(svc.status['busy'])
                        self.assertEqual(svc.status['queued'], 0)
                        self.assertEqual(svc.mcp.calls, [])
                        self.assertEqual(svc.history[-1]['finish'], 'disconnect')
                        self.assertIsNone(svc.request_trace.session_id)
                        if monitor:
                            self.assertEqual(svc.api_requests[-1]['state'], 'disconnected')
                            self.assertIsNone(svc.request_trace.record)
                        status, _, body = self.post(svc, self.path(api), self.request(api))
                        self.assertEqual(status, 200)
                        answer = json.loads(body)
                        content = answer['choices'][0]['message']['content'] if api == 'openai' else answer['content'][0]['text']
                        self.assertEqual(content, 'Next')
                        self.assertEqual(svc.engine.closes, [True, True])

    def test_json_and_sse_unicode_reasoning_and_stop(self):
        for api in ('openai', 'anthropic'):
            for stream in (False, True):
                with self.subTest(api=api, stream=stream):
                    svc = self.service(['Think</think>Привет 🌍'])
                    status, headers, body = self.post(svc, self.path(api), self.request(api, stream=stream))
                    self.assertEqual(status, 200)
                    if stream:
                        self.assertIn('text/event-stream', headers['Content-Type'])
                        data = [line[6:] for line in body.splitlines() if line.startswith('data: ')]
                        if api == 'openai':
                            self.assertEqual(data.pop(), '[DONE]')
                            answer = openai_collect([json.loads(d) for d in data])
                        else:
                            events = [line[7:] for line in body.splitlines() if line.startswith('event: ')]
                            self.assertEqual(len(events), len(data))
                            self.assertEqual(events[-1], 'message_stop')
                            answer = anthropic_collect(list(zip(events, map(json.loads, data))))
                    else:
                        self.assertIn('application/json', headers['Content-Type'])
                        answer = json.loads(body)
                    if api == 'openai':
                        choice = answer['choices'][0]
                        self.assertEqual(choice['finish_reason'], 'stop')
                        self.assertEqual(choice['message']['content'], 'Привет 🌍')
                        self.assertEqual(choice['message']['reasoning_content'], 'Think')
                    else:
                        self.assertEqual(answer['stop_reason'], 'end_turn')
                        self.assertEqual([b['text'] for b in answer['content'] if b['type'] == 'text'], ['Привет 🌍'])
                        self.assertEqual([b['thinking'] for b in answer['content'] if b['type'] == 'thinking'], ['Think'])
                    self.assertNotIn('</think>', body)
                    self.assertIsNone(svc.request_trace.session_id)

    def test_invalid_options_return_json_before_stream_or_generation(self):
        for api in ('openai', 'anthropic'):
            invalid_effort = ({'reasoning_effort': 'medium'} if api == 'openai' else
                              {'output_config': {'effort': 'medium'}})
            for options in (invalid_effort,
                            {'chat_template_kwargs': {'enable_thinking': False}},
                            {'chat_template_kwargs': {'clear_thinking': 'false'}}):
                with self.subTest(api=api, options=options):
                    svc = self.service(['</think>Must not run'])
                    status, headers, body = self.post(svc, self.path(api), self.request(api, stream=True, **options))
                    self.assertEqual(status, 400)
                    self.assertIn('application/json', headers['Content-Type'])
                    self.assertEqual(json.loads(body)['error']['type'], 'invalid_request_error')
                    self.assertEqual(svc.engine.last_prompt, [])

    def test_client_tool_name_wins_over_mcp_wrapped_and_bare(self):
        for wrapped in (True, False):
            with self.subTest(wrapped=wrapped):
                svc = self.service(['</think>' + call('A'), '</think>Should not continue'])
                svc.mcp = HandlerHub()
                schema = deepcopy(TOOLS[0])
                tool = {'type': 'function', 'function': schema} if wrapped else schema
                status, _, body = self.post(svc, self.path('openai'), self.request(
                    'openai', tools=[tool], strata_mcp=True))
                self.assertEqual(status, 200)
                self.assertEqual(svc.mcp.excluded, {'mock__echo'})
                self.assertEqual(svc.mcp.calls, [])
                choice = json.loads(body)['choices'][0]
                self.assertEqual(choice['finish_reason'], 'tool_calls')
                self.assertEqual(choice['message']['tool_calls'][0]['function']['name'], 'mock__echo')
                self.assertEqual(svc.engine.turns, 1)

    def test_noncolliding_mcp_tool_still_executes(self):
        svc = self.service(['</think>' + call('A'), '</think>Done'])
        svc.mcp = HandlerHub()
        status, _, body = self.post(svc, self.path('openai'), self.request('openai', strata_mcp=True))
        self.assertEqual(status, 200)
        self.assertEqual(svc.mcp.calls, [('mock__echo', {'text': 'A'})])
        self.assertEqual(json.loads(body)['choices'][0]['message']['content'], 'Done')

    def test_openai_tool_result_round_trip(self):
        svc = self.service(['Need tool</think>' + call('A'), 'Have result</think>Done'])
        req = self.request('openai', tools=[{'type': 'function', 'function': TOOLS[0]}])
        status, _, body = self.post(svc, self.path('openai'), req)
        self.assertEqual(status, 200)
        answer = json.loads(body)['choices'][0]
        self.assertEqual(answer['finish_reason'], 'tool_calls')
        message = answer['message']
        tool = message['tool_calls'][0]
        self.assertEqual(json.loads(tool['function']['arguments']), {'text': 'A'})
        req['messages'] += [message, {'role': 'tool', 'tool_call_id': tool['id'], 'content': 'result:A'}]
        status, _, body = self.post(svc, self.path('openai'), req)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)['choices'][0]['message']['content'], 'Done')
        self.assertIn('<tool_response>result:A</tool_response>', svc.tok.decode(svc.engine.last_prompt))

    def test_anthropic_tool_result_round_trip(self):
        svc = self.service(['Need tool</think>' + call('A'), 'Have result</think>Done'])
        req = self.request('anthropic', tools=[{'name': 'mock__echo', 'input_schema': TOOLS[0]['parameters']}])
        status, _, body = self.post(svc, self.path('anthropic'), req)
        self.assertEqual(status, 200)
        answer = json.loads(body)
        self.assertEqual(answer['stop_reason'], 'tool_use')
        tool = next(b for b in answer['content'] if b['type'] == 'tool_use')
        self.assertEqual(tool['input'], {'text': 'A'})
        req['messages'] += [{'role': 'assistant', 'content': answer['content']},
                            {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': tool['id'],
                                                         'content': 'result:A'}]}]
        status, _, body = self.post(svc, self.path('anthropic'), req)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)['stop_reason'], 'end_turn')
        self.assertIn('<tool_response>result:A</tool_response>', svc.tok.decode(svc.engine.last_prompt))


if __name__ == '__main__':
    unittest.main()
