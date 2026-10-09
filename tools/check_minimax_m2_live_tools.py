"""Live MiniMax multi-call/error matrix over OpenAI/Anthropic JSON/SSE. No external tools."""
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

FUNCTION = {'name': 'get_code', 'description': 'Look up the current code for a city and revision.',
            'parameters': {'type': 'object', 'properties': {'city': {'type': 'string', 'enum': ['Oslo', 'Уфа']},
                'revision': {'type': 'integer', 'minimum': 1}}, 'required': ['city', 'revision'], 'additionalProperties': False}}
PROMPT = ('Use get_code twice: city Oslo with revision 2, and city Уфа with revision 3. '
          'Make both independent calls before answering. Then report the exact returned codes. '
          'If a lookup fails, report its exact error code; do not retry or invent a code.')


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def collect_sse(api, wire):
    """Independent small client: validate event/block order and assemble calls."""
    records = []
    for block in wire.replace('\r\n', '\n').split('\n\n'):
        data = [line[6:] for line in block.splitlines() if line.startswith('data: ')]
        event = [line[7:] for line in block.splitlines() if line.startswith('event: ')]
        if data:
            assert len(data) == 1 and len(event) <= 1
            records.append((event[0] if event else None, data[0]))
    assert records
    if api == 'openai':
        assert records[-1] == (None, '[DONE]') and sum(d == '[DONE]' for _, d in records) == 1
        text, reasoning, calls, finals = [], [], {}, []
        for _, raw in records[:-1]:
            event = json.loads(raw)
            assert 'error' not in event and len(event['choices']) == 1
            c = event['choices'][0]
            delta = c['delta']
            text.append(delta.get('content') or '')
            reasoning.append(delta.get('reasoning_content') or '')
            for part in delta.get('tool_calls', []):
                i = part['index']
                assert type(i) is int and i >= 0
                current = calls.setdefault(i, {'id': None, 'type': 'function', 'function': {'name': '', 'arguments': ''}})
                if part.get('id'):
                    assert current['id'] in (None, part['id'])
                    current['id'] = part['id']
                for key in ('name', 'arguments'):
                    current['function'][key] += part.get('function', {}).get(key, '')
            if c['finish_reason'] is not None:
                finals.append(event)
        assert len(finals) == 1 and json.loads(records[-2][1]) == finals[0]
        assert sorted(calls) == list(range(len(calls)))
        ordered = [calls[i] for i in sorted(calls)]
        assert len({c['id'] for c in ordered}) == len(ordered) and all(c['id'] for c in ordered)
        for call in ordered:
            assert isinstance(json.loads(call['function']['arguments']), dict)
        message = {'role': 'assistant', 'content': ''.join(text) or None, 'reasoning_content': ''.join(reasoning)}
        if ordered:
            message['tool_calls'] = ordered
        return {'choices': [{'message': message, 'finish_reason': finals[0]['choices'][0]['finish_reason']}],
                'usage': finals[0]['usage']}

    blocks, active, initial, final = [], None, None, None
    assert records[-1][0] == 'message_stop' and sum(e == 'message_stop' for e, _ in records) == 1
    for name, raw in records:
        event = json.loads(raw)
        assert name == event['type'] and name != 'error'
        if name == 'message_start':
            assert initial is None and not blocks
            initial = event['message']
        elif name == 'content_block_start':
            assert initial is not None and final is None and active is None and event['index'] == len(blocks)
            blocks.append(deepcopy(event['content_block']))
            active = event['index']
        elif name == 'content_block_delta':
            assert active is not None and event['index'] == active
            block, delta = blocks[active], event['delta']
            if delta['type'] in ('text_delta', 'thinking_delta'):
                key = 'text' if delta['type'] == 'text_delta' else 'thinking'
                assert block['type'] == ('text' if key == 'text' else 'thinking')
                block[key] += delta[key]
            else:
                assert block['type'] == 'tool_use' and delta['type'] == 'input_json_delta'
                block['_json'] = block.get('_json', '')+delta['partial_json']
        elif name == 'content_block_stop':
            assert active is not None and event['index'] == active
            if blocks[active]['type'] == 'tool_use':
                blocks[active]['input'] = json.loads(blocks[active].pop('_json', '{}'))
            active = None
        elif name == 'message_delta':
            assert active is None and final is None
            final = event
        elif name == 'message_stop':
            assert final is not None and active is None
        else:
            assert name == 'ping'
    calls = [b for b in blocks if b['type'] == 'tool_use']
    assert len({c['id'] for c in calls}) == len(calls)
    return {**initial, 'content': blocks, **final['delta'], 'usage': {**initial['usage'], **final['usage']}}


def request(api, stream):
    tool = {'type': 'function', 'function': deepcopy(FUNCTION)} if api == 'openai' else {
        'name': FUNCTION['name'], 'description': FUNCTION['description'], 'input_schema': deepcopy(FUNCTION['parameters'])}
    return {'model': 'minimax-m2.7', 'messages': [{'role': 'user', 'content': PROMPT}], 'tools': [tool],
            'max_tokens': 768, 'temperature': 1, 'top_p': .95, 'top_k': 40, 'seed': 42, 'stream': stream}


def tool_calls(api, answer):
    if api == 'openai':
        assert answer['choices'][0]['finish_reason'] == 'tool_calls', answer
        return [{'id': c['id'], 'name': c['function']['name'], 'input': json.loads(c['function']['arguments'])}
                for c in answer['choices'][0]['message'].get('tool_calls', [])]
    assert answer['stop_reason'] == 'tool_use', answer
    return [c for c in answer['content'] if c['type'] == 'tool_use']


def followup(req, api, answer, calls, error):
    result = deepcopy(req)
    results = []
    # Deliberately reverse transport order; the template must correlate IDs.
    for c in reversed(calls):
        failed = error and c['input']['city'] == 'Уфа'
        value = 'CODE_UNAVAILABLE' if failed else 'OSLO-4179' if c['input']['city'] == 'Oslo' else 'UFA-9264'
        results.append({'role': 'tool', 'tool_call_id': c['id'], 'content': ('Error: ' if failed else '')+value} if api == 'openai' else
                       {'type': 'tool_result', 'tool_use_id': c['id'], 'content': value, 'is_error': failed})
    result['messages'] += ([answer['choices'][0]['message'], *results] if api == 'openai' else
                           [{'role': 'assistant', 'content': answer['content']}, {'role': 'user', 'content': results}])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gguf', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    report = {'pass': False, 'scope': 'one live two-call task through both APIs, JSON normal/SSE error results; no external tools',
              'model': str(args.gguf.resolve()), 'engine_sha256': EXE_SHA256, 'header_sha256': HEADER_SHA256,
              'sources': {}, 'requests': [], 'cycles': []}
    for source in [*sorted((ROOT/'serve').glob('*.py')), Path(__file__).resolve(),
                   ROOT/'serve/fixtures/minimax_m27_chat_template.jinja', ROOT/'tools/strata_tokenizer.py',
                   *sorted((ROOT/'serve/web').glob('*'))]:
        if source.is_file():
            name = source.relative_to(ROOT)
            target = args.out/'sources'/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            report['sources'][str(name)] = hashlib.sha256(target.read_bytes()).hexdigest()
    binary = ROOT/'build-local/minimax-m2-cuda/bin/strata-minimax-m2-bench.exe'
    shutil.copyfile(binary, args.out/'engine.exe')
    report['gpu'] = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
                                             encoding='utf-8', timeout=15).strip()
    engine = server = thread = None
    try:
        tok = Tokenizer.from_gguf(args.gguf)
        engine = MiniMaxEngine(args.gguf, binary, ROOT/'build-local/cuda-13.0', args.out/'native.log',
                              context=2048, batch=16, gpu_cache_mib=18432, pipeline_readers=2, pipeline_chunk_mib=4)
        report['header'], report['command'] = engine.header, engine.command
        svc = Service(engine, tok, MiniMaxAPITemplate(ROOT/'serve/fixtures/minimax_m27_chat_template.jinja'), model_name='minimax-m2.7')
        server = Server(('127.0.0.1', 0), make_handler(svc))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        print('READY live tool matrix', flush=True)

        def post(name, api, req):
            print('START '+name, flush=True)
            save(args.out/(name+'.request.json'), req)
            normalized = svc.normalize_request(req, api)
            ids = svc.prepare(*normalized, max_new=req['max_tokens'])[0]
            save(args.out/(name+'.prompt.json'), {'ids': ids, 'text': tok.decode(ids)})
            client = http.client.HTTPConnection(*server.server_address, timeout=900)
            started = time.monotonic()
            try:
                client.request('POST', '/v1/chat/completions' if api == 'openai' else '/v1/messages',
                               json.dumps(req, ensure_ascii=False).encode('utf-8'), {'Content-Type': 'application/json'})
                response = client.getresponse()
                wire = response.read().decode('utf-8')
                (args.out/(name+'.response.txt')).write_text(wire, encoding='utf-8')
                row = {'name': name, 'http_status': response.status, 'wall_ms': 1000*(time.monotonic()-started),
                       'native_result': deepcopy(engine.last_result), 'native_error': deepcopy(engine.last_error),
                       'prompt_ids': ids}
                report['requests'].append(row)
                save(args.out/'report.json', report)
                assert response.status == 200, wire
                answer = collect_sse(api, wire) if req['stream'] else json.loads(wire)
                save(args.out/(name+'.collected.json'), answer)
                native = row['native_result']
                assert native and native['stop_reason'] == 'eos', row
                row['native_checks'] = result_checks(native, engine.header)
                assert all(row['native_checks'].values()), row['native_checks']
                assert answer['usage']['completion_tokens' if api == 'openai' else 'output_tokens'] == native['generated_tokens']
                assert answer['usage']['prompt_tokens' if api == 'openai' else 'input_tokens'] == native['prompt_tokens'] == len(ids)
                row['pass'] = True
                print('DONE '+name+': '+str(native['generated_tokens'])+' tokens, '+str(round(native['decode_tokens_per_second'],3))+' tok/s', flush=True)
                return answer, row
            finally:
                client.close()

        for api, stream in [('anthropic', True), ('openai', False), ('anthropic', False), ('openai', True)]:
            name = api+('-sse-error' if stream else '-json-normal')
            cycle = {'name': name, 'pass': False}
            report['cycles'].append(cycle)
            try:
                req = request(api, stream)
                answer, first = post(name+'-calls', api, req)
                calls = tool_calls(api, answer)
                assert len(calls) == 2 and len({c['id'] for c in calls}) == 2, calls
                assert all(c['name'] == 'get_code' and type(c['input']['revision']) is int for c in calls), calls
                assert sorted((c['input']['city'], c['input']['revision']) for c in calls) == [('Oslo', 2), ('Уфа', 3)], calls
                cycle['calls'] = calls
                next_req = followup(req, api, answer, calls, error=stream)
                final, second = post(name+'-answer', api, next_req)
                if api == 'openai':
                    assert final['choices'][0]['finish_reason'] == 'stop' and not final['choices'][0]['message'].get('tool_calls'), final
                    text = final['choices'][0]['message']['content'] or ''
                else:
                    assert final['stop_reason'] == 'end_turn' and not any(b['type'] == 'tool_use' for b in final['content']), final
                    text = ''.join(b['text'] for b in final['content'] if b['type'] == 'text')
                assert 'OSLO-4179' in text and ('CODE_UNAVAILABLE' if stream else 'UFA-9264') in text, text
                if stream:
                    assert 'UFA-9264' not in text
                cycle.update(pass_=True, final_text=text)
                cycle['pass'] = cycle.pop('pass_')
            except Exception:
                cycle['error'] = traceback.format_exc()
                print('FAIL '+name+': '+cycle['error'].splitlines()[-1], flush=True)
            save(args.out/'report.json', report)
        # Same canonical calls prompt/sampler must produce identical native IDs.
        calls = [r for r in report['requests'] if r['name'].endswith('-calls') and r.get('pass')]
        report['cross_api_call_parity'] = len(calls) == 4 and all(
            c['prompt_ids'] == calls[0]['prompt_ids'] and c['native_result']['token_ids'] == calls[0]['native_result']['token_ids'] for c in calls)
        report['pass'] = all(c['pass'] for c in report['cycles']) and report['cross_api_call_parity']
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
        print('REPORT '+str(args.out/'report.json')+' pass='+str(report['pass']), flush=True)
    return 0 if report['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
