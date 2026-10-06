"""Full Hy3 GGUF through production Service and loopback JSON/SSE handlers.

Uses a fixed local temperature stub, records native IDs, and stops only its
engine if sampled global RAM/VRAM exceeds 95%. This is a correctness check.
"""
import argparse
from copy import deepcopy
import hashlib
import http.client
import json
from pathlib import Path
import socket
import sys
import threading
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import server
from tools.check_step35_model import Monitor
from tools.prepare_hy3_profile import load_tokenizer


class ObservedEngine(server.StrataEngine):
    def generate(self, *args, **kwargs):
        record = {'prompt_ids': list(args[0]), 'ids': []}
        self.records.append(record)
        generator = super().generate(*args, **kwargs)
        try:
            for token in generator:
                if token is not None:
                    record['ids'].append(token)
                yield token
        finally:
            generator.close()
            record['done'] = dict(self.last)


def wait_for(predicate, engine, timeout=240):
    end = time.monotonic()+timeout
    while not predicate():
        if not engine.alive():
            raise RuntimeError('Hy3 engine exited during HTTP check')
        if time.monotonic() > end:
            raise TimeoutError('Hy3 HTTP phase timed out')
        time.sleep(.05)


def run(profile, reference, output, allow_engine_change=False):
    cfg = json.loads(profile.read_text(encoding='utf8'))
    ref = json.loads(reference.read_text(encoding='utf8'))
    sha = hashlib.sha256(Path(cfg['exe']).read_bytes()).hexdigest()
    if cfg['architecture'] != 'hy_v3' or ref['status'] != 'pass':
        raise ValueError('Explicit Hy3 profile and passing dialogue reference required')
    if sha != cfg['engine_sha256'] or (sha != ref['engine_sha256'] and not allow_engine_change):
        raise ValueError('Profile/engine checksum mismatch or reference engine changed without explicit --allow-engine-change')
    if cfg['model_header_sha256'] != ref['header_sha256']:
        raise ValueError('Profile and reference must describe the same model header')
    tok = load_tokenizer(cfg['tokenizer'])
    template = server.configured_template(cfg, tok, Path(cfg['tokenizer']))
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='error', scope='production Service/StrataEngine; loopback JSON/SSE; local tool stub; no browser interaction',
        profile=str(profile), reference=str(reference), engine_sha256=sha,
        reference_engine_sha256=ref['engine_sha256'], allow_engine_change=allow_engine_change,
        engine_args=server.engine_args(cfg), requests=[], cancellations=[], endpoints={})
    monitor, engine, httpd, worker = Monitor(), None, None, None
    phase, finished = ['loading'], threading.Event()

    def progress():
        while not finished.wait(20):
            print('Hy3 HTTP: '+phase[0], flush=True)

    narrator = threading.Thread(target=progress, daemon=True)
    narrator.start()

    def save():
        (output/'http-model-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf8')

    def path(api):
        return '/v1/chat/completions' if api == 'openai' else '/v1/messages'

    try:
        original_contain = server.contain

        def contain(proc):
            original_contain(proc)
            monitor.start(proc)

        with patch.object(server, 'contain', contain):
            engine = ObservedEngine(cfg['exe'], server.engine_args(cfg), cwd=cfg['cwd'],
                log=str(output/'engine.log'), env=server.child_env(cfg))
        engine.records = []
        svc = server.Service(engine, tok, template, model_name=cfg['model_name'],
            sampling_defaults=cfg['sampling'], fit_max_tokens=cfg.get('fit_max_tokens', False))
        svc.api_monitor = True
        httpd = server.Server(('127.0.0.1', 0), server.make_handler(svc))
        worker = threading.Thread(target=lambda: httpd.serve_forever(poll_interval=.05), daemon=True)
        worker.start()
        report.update(info=svc.engine_facts(), eog_ids=sorted(svc.stop_ids), address=list(httpd.server_address))
        assert report['eog_ids'] == [120025]
        assert not engine.can_cache_admin and svc.vision is None and engine.info['conversation_cache'] == 0
        report['pipe_capabilities'] = dict(stop=engine.can_stop, session_id=engine.can_session_id,
            cache_admin=engine.can_cache_admin, conversation_cache=engine.info['conversation_cache'])
        for endpoint in ('/', '/health', '/v1/models', '/settings'):
            conn = http.client.HTTPConnection(*httpd.server_address, timeout=30)
            try:
                conn.request('GET', endpoint)
                response = conn.getresponse()
                raw = response.read()
                assert response.status == 200, raw
                report['endpoints'][endpoint] = json.loads(raw) if endpoint != '/' else {'bytes': len(raw), 'html': b'data-v="no_think"' in raw}
            finally:
                conn.close()
        assert report['endpoints']['/']['html']
        assert report['endpoints']['/health']['reasoning']['default'] == 'no_think'
        assert report['endpoints']['/v1/models']['data'][0]['id'] == 'hy3'
        save()
        print('Real Hy3 HTTP READY', flush=True)

        def request(api, context, stream=False):
            tools = deepcopy(context.get('tools') or [])
            if api == 'anthropic':
                tools = [dict(name=t['function']['name'], description=t['function'].get('description', ''),
                    input_schema=t['function']['parameters']) for t in tools]
            return dict(model=svc.model, messages=deepcopy(context['messages']), tools=tools,
                max_tokens=96, temperature=0, stream=stream,
                **({'reasoning_effort': 'no_think'} if api == 'openai' else {'thinking': {'type': 'disabled'}}))

        def post(name, api, req, expected=None):
            phase[0] = name
            index, start = len(engine.records), time.perf_counter()
            conn = http.client.HTTPConnection(*httpd.server_address, timeout=300)
            try:
                conn.request('POST', path(api), json.dumps(req, ensure_ascii=False).encode(), {'Content-Type': 'application/json'})
                response = conn.getresponse()
                raw = response.read().decode('utf8')
                assert response.status == 200, raw
            finally:
                conn.close()
            if req['stream']:
                data = [x[6:] for x in raw.splitlines() if x.startswith('data: ')]
                if api == 'openai':
                    assert data.pop() == '[DONE]'
                    answer = server.openai_collect([json.loads(x) for x in data])
                else:
                    events = [x[7:] for x in raw.splitlines() if x.startswith('event: ')]
                    assert events[-1] == 'message_stop'
                    answer = server.anthropic_collect(list(zip(events, map(json.loads, data))))
            else:
                answer = json.loads(raw)
            record = engine.records[index]
            record.update(name=name, api=api, stream=req['stream'], answer=answer,
                wall_seconds=time.perf_counter()-start, reference_ids_exact=False)
            report['requests'].append(record)
            save()
            assert record['ids'][-1] == 120025 and record['done']['finish'] == 'stop', record
            if expected:
                assert record['prompt_ids'] == expected['prompt_ids'], name+': prompt IDs changed'
                assert record['ids'] == expected['ids'], name+': output IDs changed'
                record['reference_ids_exact'] = True
            save()
            print(f'{name}: {len(record["ids"])} output tokens; reference={bool(expected)}', flush=True)
            return answer, record

        base = {'messages': [{'role': 'user', 'content': 'Сколько будет 2 + 2? Ответь одной цифрой.'}]}
        baseline = None
        for api in ('openai', 'anthropic'):
            for stream in (False, True):
                answer, record = post(f'math_{api}_{stream}', api, request(api, base, stream), baseline)
                content = answer['choices'][0]['message']['content'] if api == 'openai' else ''.join(
                    b['text'] for b in answer['content'] if b['type'] == 'text')
                assert content.strip() == '4', answer
                baseline = baseline or record
        first, second = ref['turns']
        expected = lambda turn: {'prompt_ids': turn['prompt_ids'], 'ids': turn['result']['ids']}
        answer, _ = post('tool_call_openai', 'openai', request('openai', first['context']), expected(first))
        msg = answer['choices'][0]['message']
        call = msg['tool_calls'][0]
        assert len(msg['tool_calls']) == 1 and call['function']['name'] == 'get_temperature'
        assert json.loads(call['function']['arguments']) == {'city': 'Paris'}
        continuation = request('anthropic', first['context'], True)
        continuation['messages'] += [dict(role='assistant', content=[
            dict(type='thinking', thinking=msg.get('reasoning_content') or ''),
            dict(type='text', text=msg.get('content') or ''),
            dict(type='tool_use', id=call['id'], name=call['function']['name'], input=json.loads(call['function']['arguments']))]),
            dict(role='user', content=[dict(type='tool_result', tool_use_id=call['id'], content=second['context']['messages'][-1]['content'])])]
        answer, _ = post('tool_result_anthropic', 'anthropic', continuation, expected(second))
        assert ''.join(b['text'] for b in answer['content'] if b['type'] == 'text').strip() == '17 C'
        report['local_stub_result'] = ref['local_stub_result']

        phase[0] = 'disconnect_prefill'
        req = request('openai', {'messages': [{'role': 'user', 'content': 'Read: '+'alpha beta gamma. '*190}]}, True)
        index = len(engine.records)
        client = socket.create_connection(httpd.server_address, timeout=300)
        try:
            body = json.dumps(req).encode()
            client.sendall((f'POST {path("openai")} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n').encode()+body)
            wait_for(lambda: len(engine.records) > index and engine.progress is not None, engine)
            assert not engine.records[index]['ids']
            start = time.perf_counter()
            client.shutdown(socket.SHUT_RDWR)
        finally:
            client.close()
        wait_for(lambda: 'done' in engine.records[index] and not svc.fifo.locked() and not svc.status['busy'], engine, 45)
        report['cancellations'].append(dict(phase='prefill', api='openai', settle_seconds=time.perf_counter()-start, **engine.records[index]))
        save()
        assert engine.records[index]['done']['finish'] == 'cancel'
        post('after_disconnect', 'openai', request('openai', base), baseline)
        proc = engine.proc
        engine.close()
        report['engine_exit_code'] = proc.poll()
        assert report['engine_exit_code'] == 0
        assert monitor.error is None, monitor.error
        report['status'] = 'pass'
    except BaseException as error:
        report.update(error=str(error), failed_phase=phase[0])
        raise
    finally:
        finished.set()
        narrator.join(2)
        if httpd:
            httpd.shutdown()
            httpd.server_close()
            worker.join(5)
        if engine:
            engine.close()
        monitor.close()
        report.update(memory_samples=monitor.samples, memory_guard_error=monitor.error)
        save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('profile', 'reference', 'output-dir'):
        parser.add_argument('--'+arg, type=Path, required=True)
    parser.add_argument('--allow-engine-change', action='store_true', help='compare a new checksummed engine to historical native prompt/output IDs')
    args = parser.parse_args()
    run(args.profile.resolve(), args.reference.resolve(), args.output_dir.resolve(), args.allow_engine_change)


if __name__ == '__main__':
    main()
