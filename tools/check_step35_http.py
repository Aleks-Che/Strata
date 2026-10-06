"""Real Step engine + production HTTP handlers on loopback, with a local tool stub.

Explicit experimental profile and passing STEP-09 reference required. Test-only
instrumentation records native token IDs and applies the existing 95% monitor.
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
from tools.prepare_step35_profile import load_tokenizer


class ObservedEngine(server.StrataEngine):
    def generate(self, *args, **kwargs):
        record = {'prompt_ids': list(args[0]), 'ids': []}
        self.records.append(record)
        gen = super().generate(*args, **kwargs)
        try:
            for token in gen:
                if token is not None:
                    record['ids'].append(token)
                yield token
        finally:
            gen.close()
            record['done'] = dict(self.last)


def wait_for(predicate, timeout=240, engine=None):
    end = time.monotonic() + timeout
    while not predicate():
        if engine is not None and not engine.alive():
            raise RuntimeError('Step engine exited while waiting for HTTP phase')
        if time.monotonic() > end:
            raise TimeoutError('Step HTTP gate timed out')
        time.sleep(.05)


def run(profile, reference, output, allow_engine_change=False, lifecycle=False):
    output.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(profile.read_text(encoding='utf8'))
    ref = json.loads(reference.read_text(encoding='utf8'))
    if cfg['architecture'] != 'step35' or ref['status'] != 'pass':
        raise ValueError('Step profile and passing reference required')
    engine_sha = hashlib.sha256(Path(cfg['exe']).read_bytes()).hexdigest()
    if engine_sha != cfg['engine_sha256']:
        raise ValueError('Engine differs from the explicit profile checksum')
    if engine_sha != ref['engine_sha256'] and not allow_engine_change:
        raise ValueError('Real HTTP test requires the STEP-09 reference engine')
    tok = load_tokenizer(cfg['tokenizer'])
    tpl = server.configured_template(cfg, tok, Path(cfg['tokenizer']))
    report = {'status': 'error', 'scope': 'production Service/StrataEngine and loopback HTTP; local tool stub; no external tools or browser UI',
              'profile': str(profile), 'engine_args': server.engine_args(cfg),
              'engine_sha256': engine_sha, 'reference_engine_sha256': ref['engine_sha256'],
              'allow_engine_change': allow_engine_change, 'requests': [], 'cancellations': []}
    monitor, engine, httpd, worker = Monitor(), None, None, None
    monitors = [monitor]
    monitor_patch = None
    progress_done = threading.Event()
    phase = ['starting engine']
    def progress():
        while not progress_done.wait(20):
            print('HTTP check: ' + phase[0], flush=True)
    narrator = threading.Thread(target=progress, daemon=True); narrator.start()
    def save():
        (output / 'http-model-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    save()
    try:
        contain = server.contain
        def monitored(proc):
            contain(proc)
            if monitors[-1].thread is not None:
                monitors.append(Monitor())
            monitors[-1].start(proc)
        monitor_patch = patch.object(server, 'contain', monitored)
        monitor_patch.start()
        engine = ObservedEngine(cfg['exe'], server.engine_args(cfg), cwd=cfg['cwd'],
                                log=str(output/'engine.log'), env=server.child_env(cfg))
        engine.records = []
        svc = server.Service(engine, tok, tpl, model_name=cfg['model_name'], sampling_defaults=cfg['sampling'])
        svc.api_monitor = True
        httpd = server.Server(('127.0.0.1', 0), server.make_handler(svc))
        worker = threading.Thread(target=lambda: httpd.serve_forever(poll_interval=.05), daemon=True); worker.start()
        report.update(info=svc.engine_facts(), eog_ids=sorted(svc.stop_ids), address=list(httpd.server_address))
        assert report['eog_ids'] == [1, 128007]
        report['pipe_capabilities'] = {'session_id': engine.can_session_id, 'cache_admin': engine.can_cache_admin,
                                       'stop': engine.can_stop, 'conversation_cache': engine.info['conversation_cache']}
        assert not engine.can_cache_admin and svc.vision is None and engine.info['conversation_cache'] == 0
        print('Real Step HTTP READY', flush=True)
        def request(api, item, stream=False):
            tools = deepcopy(item['tools'])
            if api == 'anthropic' and tools:
                tools = [{'name': t['function']['name'], 'description': t['function'].get('description', ''),
                          'input_schema': t['function']['parameters']} for t in tools]
            return {'model': svc.model, 'messages': deepcopy(item['messages']), 'tools': tools,
                    'max_tokens': 256, 'temperature': 0, 'stream': stream,
                    **({'reasoning_effort': 'low'} if api == 'openai' else {'output_config': {'effort': 'low'}})}
        def path(api):
            return '/v1/chat/completions' if api == 'openai' else '/v1/messages'
        def post(name, api, req, expected=None):
            phase[0] = name; start = time.perf_counter()
            index = len(engine.records)
            conn = http.client.HTTPConnection(*httpd.server_address, timeout=300)
            try:
                conn.request('POST', path(api), json.dumps(req, ensure_ascii=False).encode(), {'Content-Type': 'application/json'})
                response = conn.getresponse(); raw = response.read().decode('utf8')
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
            if expected is not None:
                assert record['prompt_ids'] == expected['prompt_ids'], name + ': prompt IDs changed'
                assert record['ids'] == expected['ids'], name + ': output IDs changed'
            record.update(name=name, api=api, stream=req['stream'], answer=answer,
                          reference_ids_exact=expected is not None, wall_seconds=time.perf_counter()-start)
            report['requests'].append(record); save()
            print(f'{name}: {len(record["ids"])} output tokens; HTTP complete, reference={expected is not None}', flush=True)
            return answer
        base = ref['requests'][0]
        for api in ('openai', 'anthropic'):
            for stream in (False, True):
                post(f'P1_{api}_{stream}', api, request(api, base, stream), base)
        tool = ref['requests'][1]
        answer = post('tool_call_openai', 'openai', request('openai', tool), tool)
        msg = answer['choices'][0]['message']; call = msg['tool_calls'][0]
        assert call['function']['name'] == 'lookup_weather'
        assert json.loads(call['function']['arguments']) == {'city': 'Yekaterinburg', 'days': 2, 'include_wind': True}
        continuation = request('anthropic', tool, True)
        continuation['messages'] += [{'role': 'assistant', 'content': [
            {'type': 'thinking', 'thinking': msg['reasoning_content']},
            {'type': 'text', 'text': msg.get('content') or ''},
            {'type': 'tool_use', 'id': call['id'], 'name': call['function']['name'], 'input': json.loads(call['function']['arguments'])}]},
            {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': call['id'],
                                        'content': ref['requests'][2]['messages'][-1]['content']}]}]
        post('tool_result_anthropic', 'anthropic', continuation, ref['requests'][2])
        for name, api in (('prefill', 'openai'), ('decode', 'anthropic'), ('partial_tool', 'openai')):
            phase[0] = 'disconnect_' + name
            req = request(api, tool if name == 'partial_tool' else base, True)
            if name == 'prefill':
                req['messages'] = [{'role': 'user', 'content': 'Read these words: ' + 'alpha beta gamma. ' * 190}]
            index = len(engine.records)
            client = socket.create_connection(httpd.server_address, timeout=300)
            try:
                body = json.dumps(req).encode()
                client.sendall((f'POST {path(api)} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n').encode()+body)
                wait_for(lambda: len(engine.records) > index, engine=engine)
                if name == 'prefill':
                    wait_for(lambda: engine.progress is not None, engine=engine)
                    assert not engine.records[index]['ids']
                elif name == 'decode':
                    wait_for(lambda: len(engine.records[index]['ids']) >= 4, engine=engine)
                else:
                    wait_for(lambda: '<parameter=' in tok.decode(engine.records[index]['ids']), engine=engine)
                start = time.perf_counter(); client.shutdown(socket.SHUT_RDWR)
            finally:
                client.close()
            wait_for(lambda: 'done' in engine.records[index] and not svc.fifo.locked() and not svc.status['busy'], timeout=45, engine=engine)
            item = {'phase': name, 'api': api, 'settle_seconds': time.perf_counter()-start, **engine.records[index]}
            assert item['done']['finish'] == 'cancel', item
            report['cancellations'].append(item); save()
            post('after_' + name, api, request(api, base), base)
        if lifecycle:
            phase[0] = 'unload_reload'
            previous = engine.proc
            unloaded = svc.unload()
            assert not svc.loaded() and previous.poll() == 0, unloaded
            loaded = svc.load()
            assert svc.loaded() and engine.proc.pid != previous.pid, loaded
            report['lifecycle'] = {'unload': unloaded, 'load': loaded,
                                   'previous_pid': previous.pid, 'new_pid': engine.proc.pid}
            post('after_reload', 'openai', request('openai', base), base)
        process = engine.proc
        engine.close()
        report['engine_exit_code'] = process.poll()
        assert report['engine_exit_code'] == 0
        assert all(m.error is None for m in monitors), [m.error for m in monitors]
        report['status'] = 'pass'
    except BaseException as error:
        report.update(error=str(error), failed_phase=phase[0])
        raise
    finally:
        progress_done.set(); narrator.join(2)
        if httpd:
            httpd.shutdown(); httpd.server_close(); worker.join(5)
        if engine:
            engine.close()
        for m in monitors:
            m.close()
        if monitor_patch:
            monitor_patch.stop()
        report.update(memory_samples=[dict(s, process_index=i) for i,m in enumerate(monitors) for s in m.samples],
                      memory_guard_error=next((m.error for m in monitors if m.error), None))
        save()
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('profile', 'reference', 'output-dir'):
        p.add_argument('--'+arg, type=Path, required=True)
    p.add_argument('--allow-engine-change', action='store_true', help='compare a new explicitly checksummed engine against the original reference IDs')
    p.add_argument('--lifecycle', action='store_true', help='also unload/reload the real model and recheck reference IDs')
    args = p.parse_args()
    run(args.profile.resolve(), args.reference.resolve(), args.output_dir.resolve(), args.allow_engine_change, args.lifecycle)
