"""Opt-in live CUDA API smoke. Keeps all requests, responses and native records."""
import argparse
from copy import deepcopy
import hashlib
import http.client
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve.minimax_m2_api import MiniMaxAPITemplate
from serve.minimax_m2_engine import MiniMaxEngine, EXE_SHA256, HEADER_SHA256
from serve.server import Server, Service, make_handler
from tools.check_minimax_m2_completion_pipe import result_checks
from tools.strata_tokenizer import Tokenizer


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gguf', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--reference', type=Path, default=ROOT/'build-local/minimax-m2-completion-01/generation.json')
    p.add_argument('--engine', type=Path, default=ROOT/'build-local/minimax-m2-cuda/bin/strata-minimax-m2-bench.exe')
    p.add_argument('--cuda-root', type=Path, default=ROOT/'build-local/cuda-13.0')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    report = {'pass': False, 'scope': 'live resident CUDA via real loopback HTTP, token parity (not logits)',
              'engine_sha256': EXE_SHA256, 'header_sha256': HEADER_SHA256, 'model': str(args.gguf.resolve()),
              'reference': str(args.reference.resolve()), 'reference_sha256': hashlib.sha256(args.reference.read_bytes()).hexdigest(),
              'cases': [], 'sources': {}}
    for file in ['serve/minimax_m2_engine.py', 'serve/minimax_m2_worker.py', 'serve/minimax_m2_server.py',
                 'serve/minimax_m2_api.py', 'serve/minimax_m2_history.py', 'serve/minimax_m2_tools.py',
                 'serve/server.py', 'serve/winjob.py', 'backends/minimax_m2/main.cpp',
                 'tools/check_minimax_m2_live_api.py']:
        report['sources'][file] = hashlib.sha256((ROOT/file).read_bytes()).hexdigest()
        target = args.out/'sources'/file
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/file, target)
    shutil.copyfile(args.engine, args.out/'engine.exe')
    report['gpu'] = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
                                             encoding='utf-8', timeout=15).strip()
    engine = server = thread = None
    try:
        reference = json.loads(args.reference.read_text(encoding='utf-8'))['results'][0]
        tok = Tokenizer.from_gguf(args.gguf)
        template = MiniMaxAPITemplate(ROOT/'serve/fixtures/minimax_m27_chat_template.jinja')
        engine = MiniMaxEngine(args.gguf, args.engine, args.cuda_root, args.out/'native.stderr.log', context=2048, batch=16,
                              gpu_cache_mib=18432, pipeline_readers=2, pipeline_chunk_mib=4)
        report['header'], report['command'] = engine.header, engine.command
        report['environment'] = {k: v for k, v in engine.env.items() if k.startswith(('STRATA_', 'GGML_', 'LLAMA_', 'NVIDIA_'))}
        svc = Service(engine, tok, template, model_name='minimax-m2.7')
        server = Server(('127.0.0.1', 0), make_handler(svc))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        report['port'] = server.server_address[1]
        print('READY live MiniMax HTTP, pid='+str(engine.header['native_pid']), flush=True)
        base = {'model': 'minimax-m2.7', 'messages': [{'role': 'user',
                'content': 'Explain in a few sentences why the sky appears blue during the day.'}], 'temperature': 0, 'max_tokens': 8}

        def post(name, api, req):
            print('START '+name, flush=True)
            save(args.out/(name+'.request.json'), req)
            conn = http.client.HTTPConnection('127.0.0.1', report['port'], timeout=900)
            started = time.monotonic()
            try:
                conn.request('POST', '/v1/chat/completions' if api == 'openai' else '/v1/messages',
                             json.dumps(req).encode(), {'Content-Type': 'application/json'})
                response = conn.getresponse()
                body = response.read().decode('utf-8')
                (args.out/(name+'.response.txt')).write_text(body, encoding='utf-8')
                case = {'name': name, 'api': api, 'http_status': response.status, 'wall_ms': (time.monotonic()-started)*1000,
                        'native_result': deepcopy(engine.last_result), 'native_error': deepcopy(engine.last_error)}
                report['cases'].append(case)
                save(args.out/'report.json', report)
                assert response.status == 200, body
                if engine.last_result:
                    case['native_checks'] = result_checks(engine.last_result, engine.header)
                    assert all(case['native_checks'].values()), case['native_checks']
                    print('DONE '+name+': '+str(engine.last_result['generated_tokens'])+' tokens, '+
                          str(round(engine.last_result['decode_tokens_per_second'], 3))+' decode tok/s', flush=True)
                else:
                    print('DONE '+name+': native '+str(engine.last_error), flush=True)
                if req.get('stream'):
                    events = [json.loads(line[6:]) for line in body.splitlines() if line.startswith('data: ') and line != 'data: [DONE]']
                    assert not any(e.get('type') == 'error' or 'error' in e for e in events), events
                    return events, case
                return json.loads(body), case
            finally:
                conn.close()

        for api, stream in [('openai', False), ('anthropic', True)]:
            answer, case = post(api+'-prefix', api, {**base, 'stream': stream})
            assert case['native_result']['token_ids'] == reference['token_ids'][:8]
            if api == 'openai':
                assert answer['choices'][0]['finish_reason'] == 'length'
                assert answer['usage']['completion_tokens'] == 8
                text = answer['choices'][0]['message']['reasoning_content']
            else:
                delta = next(e['delta'] for e in answer if e.get('type') == 'message_delta')
                assert delta['stop_reason'] == 'max_tokens'
                assert next(e['usage']['output_tokens'] for e in answer if e.get('type') == 'message_delta') == 8
                text = ''.join(e['delta'].get('thinking', '') for e in answer if e.get('type') == 'content_block_delta')
            assert text == b''.join(tok.token_bytes(t) for t in reference['token_ids'][:8]).decode('utf-8')
            case['pass'] = True

        answer, case = post('openai-natural-eos', 'openai', {**base, 'max_tokens': 1536})
        assert case['native_result']['token_ids'] == reference['token_ids']
        assert answer['choices'][0]['finish_reason'] == 'stop' and answer['usage']['completion_tokens'] == len(reference['token_ids'])
        assert answer['choices'][0]['message']['content']
        case['pass'] = True

        stop = b''.join(tok.token_bytes(t) for t in reference['token_ids'][:8]).decode('utf-8')[-12:]
        answer, case = post('openai-stop', 'openai', {**base, 'max_tokens': 1536, 'stop': [stop]})
        assert answer['choices'][0]['finish_reason'] == 'stop'
        assert stop not in answer['choices'][0]['message'].get('reasoning_content', '')
        assert engine.alive() and engine.last_error and 'cancel' in engine.last_error['message'].lower()
        case['pass'] = True

        # Close the actual HTTP connection while the engine is prefilling.
        req = {**base, 'messages': [{'role': 'user', 'content': 'Read this list: '+', '.join(map(str, range(300)))}], 'max_tokens': 64}
        save(args.out/'disconnect.request.json', req)
        native_pid = engine.header['native_pid']
        for i in range(3):
            conn = http.client.HTTPConnection('127.0.0.1', report['port'], timeout=10)
            previous = engine._sequence
            started = time.monotonic()
            conn.request('POST', '/v1/chat/completions', json.dumps(req), {'Content-Type': 'application/json'})
            deadline = time.monotonic()+10
            while engine._sequence == previous and time.monotonic() < deadline:
                time.sleep(.02)
            assert engine._sequence > previous, 'HTTP request never reached engine'
            conn.close()
            deadline = time.monotonic()+40
            while engine._gate.locked() and time.monotonic() < deadline:
                time.sleep(.05)
            assert not engine._gate.locked() and engine.alive() and engine.last_error, 'disconnect did not drain and recover'
            assert engine.header['native_pid'] == native_pid, 'unexpected process replacement'
            report['cases'].append({'name': 'http-prefill-disconnect-'+str(i), 'pass': True,
                                    'wall_ms': (time.monotonic()-started)*1000, 'native_error': engine.last_error,
                                    'native_pid': native_pid})
        answer, case = post('post-cancel-recovery', 'openai', base)
        assert case['native_result']['token_ids'] == reference['token_ids'][:8]
        case['pass'] = True

        tools = [{'type': 'function', 'function': {'name': 'get_code', 'description': 'Look up the current city code.',
                  'parameters': {'type': 'object', 'properties': {'city': {'type': 'string', 'enum': ['Oslo']}},
                                 'required': ['city'], 'additionalProperties': False}}}]
        req = {'model': 'minimax-m2.7', 'messages': [{'role': 'user', 'content':
               'Call get_code for Oslo, then tell me the code returned by the tool. Do not guess the code.'}],
               'tools': tools, 'max_tokens': 768, 'temperature': 1, 'top_p': .95, 'top_k': 40, 'seed': 42}
        answer, case = post('openai-tool-call', 'openai', req)
        choice = answer['choices'][0]
        calls = choice['message'].get('tool_calls', [])
        assert choice['finish_reason'] == 'tool_calls' and len(calls) == 1, answer
        assert calls[0]['function']['name'] == 'get_code' and json.loads(calls[0]['function']['arguments']) == {'city': 'Oslo'}, calls
        case['pass'] = True
        req['messages'] += [choice['message'], {'role': 'tool', 'tool_call_id': calls[0]['id'], 'content': 'OSLO-4179'}]
        answer, case = post('openai-tool-answer', 'openai', req)
        assert answer['choices'][0]['finish_reason'] == 'stop' and 'OSLO-4179' in answer['choices'][0]['message']['content'], answer
        case['pass'] = True

        engine.unload()
        answer, case = post('http-reload', 'openai', base)
        assert case['native_result']['token_ids'] == reference['token_ids'][:8]
        case['pass'] = True
        report['pass'] = all(c.get('pass') for c in report['cases'])
    except Exception:
        report['error'] = traceback.format_exc()
        raise
    finally:
        if server:
            server.shutdown()
        if engine:
            engine.close()
        if server:
            server.server_close()
        if thread:
            thread.join(5)
        save(args.out/'report.json', report)
        print('REPORT '+str(args.out/'report.json'), flush=True)


if __name__ == '__main__':
    main()
