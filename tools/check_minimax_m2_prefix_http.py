"""Live HTTP session isolation and native cancellation/restart after prefix reuse.

Requires a passing offline prefix audit and the admitted executable. GPU tests
are sequential. All HTTP requests stay on loopback; children belong to this run.
"""
import argparse
from copy import deepcopy
import http.client
import json
from pathlib import Path
import shutil
import sys
import threading
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve.minimax_m2_api import MiniMaxAPITemplate
from serve.minimax_m2_engine import MiniMaxEngine, NativeRequestError, EXE_SHA256
from serve.server import Service, Server, make_handler
from tools.check_minimax_m2_completion_pipe import result_checks
from tools.check_minimax_m2_prefix import save, sha
from tools.strata_tokenizer import Tokenizer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gguf', type=Path, required=True)
    p.add_argument('--offline', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--engine', type=Path, default=ROOT/'build-local/minimax-m2-cuda/bin/strata-minimax-m2-bench.exe')
    p.add_argument('--cuda-root', type=Path, default=ROOT/'build-local/cuda-13.0')
    args = p.parse_args()
    offline = json.loads((args.offline/'report.json').read_text(encoding='utf-8'))
    assert offline['pass'] and offline['engine_sha256'] == sha(args.engine) == EXE_SHA256
    args.out.mkdir(parents=True, exist_ok=False)
    report = {'pass': False, 'scope': 'HTTP token parity/session isolation; native cancellation/restart',
              'engine_sha256': EXE_SHA256, 'offline_report_sha256': sha(args.offline/'report.json'),
              'cases': [], 'sources': {}}
    for rel in ['serve/minimax_m2_engine.py', 'serve/minimax_m2_server.py', 'serve/minimax_m2_worker.py',
                'serve/server.py', 'serve/winjob.py', 'tools/check_minimax_m2_prefix_http.py',
                'serve/minimax_m2_api.py', 'serve/minimax_m2_history.py', 'serve/minimax_m2_tools.py']:
        dest = args.out/'sources'/rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/rel, dest)
        report['sources'][rel] = sha(dest)
    reference = json.loads((args.offline/'fresh.json').read_text(encoding='utf-8'))['results'][0]['token_ids']
    prompts = json.loads((args.offline/'fresh.requests.json').read_text(encoding='utf-8'))
    base = {'model': 'minimax-m2.7', 'messages': [{'role': 'user', 'content':
            'Explain in a few sentences why the sky appears blue during the day.'}], 'max_tokens': 8, 'temperature': 0}
    engine = server = thread = None
    try:
        engine = MiniMaxEngine(args.gguf, args.engine, args.cuda_root, args.out/'native.stderr.log', context=2048, batch=16,
                              gpu_cache_mib=18432, pipeline_readers=2, pipeline_chunk_mib=4, prefix_cache=True)
        report['header'], report['command'] = deepcopy(engine.header), engine.command
        original_pid = engine.header['native_pid']
        tok = Tokenizer.from_gguf(args.gguf)
        template = MiniMaxAPITemplate(ROOT/'serve/fixtures/minimax_m27_chat_template.jinja')
        svc = Service(engine, tok, template, model_name='minimax-m2.7')
        server = Server(('127.0.0.1', 0), make_handler(svc))
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        port = server.server_address[1]
        conn = http.client.HTTPConnection('127.0.0.1', port, timeout=30)
        conn.request('GET', '/health')
        response = conn.getresponse(); health = json.loads(response.read()); conn.close()
        assert response.status == 200 and health['session_id'] is True
        report['health'] = health

        def post(name, session, expected, api='openai', stream=False):
            print('START '+name, flush=True)
            req = {**base, 'stream': stream}
            if stream and api == 'openai':
                req['stream_options'] = {'include_usage': True}
            save(args.out/(name+'.request.json'), req)
            headers = {'Content-Type': 'application/json'}
            if session is not None:
                headers['X-Strata-Session-Id'] = session
            conn = http.client.HTTPConnection('127.0.0.1', port, timeout=900)
            try:
                conn.request('POST', '/v1/chat/completions' if api == 'openai' else '/v1/messages', json.dumps(req), headers)
                response = conn.getresponse(); body = response.read().decode('utf-8')
            finally:
                conn.close()
            (args.out/(name+'.response.txt')).write_text(body, encoding='utf-8')
            r = deepcopy(engine.last_result)
            check = result_checks(r, engine.header) if r else {}
            case = {'name': name, 'status': response.status, 'api': api, 'stream': stream, 'expected_reused': expected,
                    'native': r, 'checks': check, 'native_pid': engine.header['native_pid']}
            report['cases'].append(case)
            case['pass'] = (response.status == 200 and r is not None and all(check.values()) and
                            r['token_ids'] == reference and r['reused_tokens'] == expected and engine.last['reused'] == expected)
            save(args.out/'report.json', report)
            assert case['pass'], name
            if stream:
                events = [json.loads(s[6:]) for s in body.splitlines() if s.startswith('data: ') and s != 'data: [DONE]']
                assert events and not any('error' in e or e.get('type') == 'error' for e in events)
                usage = next(e['usage'] for e in reversed(events) if 'usage' in e)
            else:
                usage = json.loads(body)['usage']
            case['usage'] = usage
            if api == 'openai':
                assert usage['prompt_tokens'] == 52 and usage['completion_tokens'] == 8
                assert usage['prompt_tokens_details']['cached_tokens'] == expected
            else:
                assert usage['input_tokens'] == 52-expected and usage['output_tokens'] == 8
                assert usage['cache_read_input_tokens'] == expected
            case['usage_pass'] = True
            print('DONE '+name+' reused='+str(expected)+' prefill_ms='+str(round(r['prefill_ms'], 3)), flush=True)

        post('initial', 'a', 0)
        post('repeat-openai-json', 'a', 48)
        post('repeat-openai-sse', 'a', 48, stream=True)
        post('repeat-anthropic-json', 'a', 48, api='anthropic')
        post('repeat-anthropic-sse', 'a', 48, api='anthropic', stream=True)
        post('other-session', 'b', 0)
        post('return-evicted', 'a', 0)
        post('anonymous', None, 0)
        post('after-anonymous', 'a', 0)

        for name, ids, cancel_after_token in [('prefill-cancel', prompts[12]['tokens'], False),
                                              ('decode-cancel', prompts[0]['tokens'], True)]:
            post(name+'-prime', 'a', 48)
            cancel = threading.Event()
            gen = engine.generate(ids, 128, {}, cancel, session_id='a')
            received = []
            try:
                item = next(gen)
                if cancel_after_token:
                    while item is None:
                        item = next(gen)
                    received.append(item)
                else:
                    assert item is None, 'prefill completed before cancellation'
                start = time.monotonic(); cancel.set()
                received.extend(t for t in gen if t is not None)
            finally:
                gen.close()
            case = {'name': name, 'cancel_ms': (time.monotonic()-start)*1000, 'received': received,
                    'error': deepcopy(engine.last_error), 'native_pid': engine.header['native_pid']}
            report['cases'].append(case)
            case['pass'] = (engine.alive() and engine.last_error is not None and
                            'cancel' in engine.last_error['message'].lower() and
                            engine.header['native_pid'] == original_pid and len(received) == int(cancel_after_token))
            assert case['pass'], name
            post(name+'-recovery', 'a', 0)
        already_cancelled = threading.Event(); already_cancelled.set()
        assert list(engine.generate(prompts[0]['tokens'], 8, {}, already_cancelled, session_id='a')) == []
        assert engine.last_result is None and engine.last_error is None and engine.alive()
        report['cases'].append({'name': 'cancel-before-submit', 'pass': True})
        post('native-error-prime', 'a', 48)
        original_send = engine._send
        def invalid_key(item):
            item = deepcopy(item)
            if item.get('command') == 'generate':
                item['request']['session_key'] = 'invalid'
            original_send(item)
        try:
            with patch.object(engine, '_send', side_effect=invalid_key):
                list(engine.generate(prompts[0]['tokens'], 8, {}, threading.Event(), session_id='a'))
            raise AssertionError('native accepted malformed session key')
        except NativeRequestError:
            assert engine.alive() and 'SHA-256' in engine.last_error['message']
        report['cases'].append({'name': 'native-error', 'pass': True, 'error': deepcopy(engine.last_error)})
        post('native-error-recovery', 'a', 0)
        post('before-restart', 'a', 48)
        engine.restart()
        assert engine.header['native_pid'] != original_pid
        report['restarted_header'] = deepcopy(engine.header)
        post('after-restart', 'a', 0)
        report['pass'] = all(c['pass'] for c in report['cases'])
    except BaseException:
        report['error'] = traceback.format_exc()
        raise
    finally:
        if server:
            server.shutdown(); server.server_close(); thread.join(10)
        if engine:
            engine.close()
            report['closed'] = not engine.alive()
        report['artifacts'] = {f.name: sha(f) for f in args.out.iterdir() if f.is_file() and f.name != 'report.json'}
        save(args.out/'report.json', report)


if __name__ == '__main__':
    main()
