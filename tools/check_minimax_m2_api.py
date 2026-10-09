"""MiniMax API adapters over real loopback HTTP, scripted native token IDs. No CUDA inference."""
import argparse
import contextlib
from copy import deepcopy
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve.minimax_m2_api import MiniMaxAPITemplate
from serve.minimax_m2_history import MiniMaxChatTemplate, prepare_minimax_context, TEMPLATE_SHA256
from serve.server import Service
from serve.test_minimax_m2_api import MiniMaxHTTPTests, FIXTURE
from serve.test_step35_http import Engine
from tools.check_minimax_m2_oracles import provenance, run_lines
from tools.inspect_minimax_m2_gguf import inspect_model
from tools.minimax_m2_tool_cases import EXPECTED, TOOLS
from tools.strata_tokenizer import Tokenizer


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, allow_nan=False, indent=2)+'\n', encoding='utf-8')


def canonical(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gguf', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--replay', type=Path, required=True)
    p.add_argument('--bin-dir', type=Path, default=ROOT/'build-local/minimax-m2-oracles/bin')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    report = {'pass': False, 'scope': 'P2.5/P2.6 adapters + real loopback HTTP JSON/SSE with scripted IDs; no live GPU/model tool generation',
              'native_engine_connected': False, 'startup_profile_registered': False, 'ready_for_use': False,
              'model': str(args.gguf.resolve()), 'source_sha256': {}, 'cycles': [], 'replays': []}
    try:
        report['python'] = sys.version
        report['packages'] = {k: importlib.metadata.version(k) for k in ('jinja2', 'jsonschema', 'referencing')}
        # Freeze local runtime modules, all new tests and their existing HTTP helpers.
        sources = [*sorted((ROOT/'serve').glob('*.py')), ROOT/'serve/fixtures/minimax_m27_chat_template.jinja',
                   ROOT/'serve/chat_template.jinja', *sorted((ROOT/'serve/fixtures').glob('*chat_template.jinja')),
                   *[ROOT/'tools'/name for name in ('check_minimax_m2_api.py', 'strata_tokenizer.py', 'gguf_reader.py',
                     'check_minimax_m2_oracles.py', 'inspect_minimax_m2_gguf.py', 'minimax_m2_loader_contract.py',
                     'minimax_m2_history_cases.py', 'minimax_m2_tool_cases.py', 'minimax_m2_template.py', 'test_minimax_m2_tokenizer.py')]]
        for source in sources:
            rel = source.relative_to(ROOT)
            dest = args.out/'sources'/rel; dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest); report['source_sha256'][str(rel)] = sha(dest)
        suffix = '.exe' if sys.platform == 'win32' else ''
        bins = {}
        for kind in ('template-json', 'tokenizer'):
            dest = args.out/('strata-minimax-m2-'+kind+suffix)
            shutil.copyfile(args.bin_dir/dest.name, dest); bins[kind] = dest.resolve()
        report['binary_sha256'] = {kind: sha(path) for kind, path in bins.items()}
        token_version = provenance(bins['tokenizer'])
        version = json.loads(subprocess.check_output([str(bins['template-json']), '--version'], timeout=15))
        assert version == dict(token_version, renderer='native-jinja-json-float-roundtrip', patches=token_version['patches']+';jinja-json-float-roundtrip')
        report['provenance'] = {'template-json': version, 'tokenizer': token_version}
        report['model_header_sha256'] = inspect_model(args.gguf)['header_sha256']
        source = FIXTURE.read_bytes().decode('utf-8')
        assert hashlib.sha256(source.encode()).hexdigest() == TEMPLATE_SHA256
        report['template_sha256'] = TEMPLATE_SHA256
        tokenizer = Tokenizer.from_gguf(args.gguf)
        assert MiniMaxAPITemplate.resolve_stop_ids(tokenizer) == {200020}
        modules = ['serve.test_minimax_m2_api', 'serve.test_minimax_m2_tools', 'serve.test_minimax_m2_history',
                   'serve.test_minimax_m2', 'tools.test_minimax_m2_tokenizer', 'serve.test_server',
                   'serve.test_glm5next_service', 'serve.test_glm5next_http', 'serve.test_step35_http', 'serve.test_hy3_http']
        suite = unittest.defaultTestLoader.loadTestsFromNames(modules)
        log, stdout = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stdout):
            unit = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
        (args.out/'unit.log').write_text(log.getvalue(), encoding='utf-8')
        (args.out/'unit.stdout.log').write_text(stdout.getvalue(), encoding='utf-8')
        report['unit'] = {'pass': unit.wasSuccessful(), 'modules': modules, 'methods': unit.testsRun,
                          'failures': len(unit.failures), 'errors': len(unit.errors), 'skipped': len(unit.skipped)}
        print('unit/regression:', report['unit']['methods'], 'PASS' if unit.wasSuccessful() else 'FAIL', flush=True)
        if not unit.wasSuccessful():
            raise RuntimeError('unit/regression failures; inspect unit.log')
        http = MiniMaxHTTPTests()
        render = MiniMaxChatTemplate(source)
        requests = []
        values = [.12345678901234566, -0.0, 5e-324, 1.7976931348623157e308]
        for number in values:
            calls = deepcopy(EXPECTED); calls[0][1]['n'] = number
            context = prepare_minimax_context({'messages': [{'role': 'user', 'content': 'Q'},
                {'role': 'assistant', 'content': 'Prefix', 'reasoning_content': 'R', 'tool_calls': [
                    {'id': 'fixture-'+str(i), 'name': n, 'arguments': a} for i, (n, a) in enumerate(calls)]}],
                'tools': TOOLS, 'add_generation_prompt': False})
            requests.append({'template': source, 'context': context})
        native = run_lines(bins['template-json'], requests)
        prompt_jobs, prompt_rows, evidence_cycles = [], [], []
        for index, (request, result) in enumerate(zip(requests, native)):
            prefix = render.render({**request['context'], 'messages': request['context']['messages'][:-1]})+']~b]ai\n<think>\n'
            assert result['rendered'].startswith(prefix) and result['rendered'].endswith('[e~[\n')
            wire = result['rendered'][len(prefix):-len('[e~[\n')]
            expected = [(c['name'], c['arguments']) for c in request['context']['messages'][-1]['tool_calls']]
            for api in ('openai', 'anthropic'):
                for stream in (False, True):
                    engine = Engine(tokenizer, [wire, 'Done</think>Answer'], stop=200020)
                    svc = Service(engine, tokenizer, MiniMaxAPITemplate(FIXTURE), model_name='minimax-m2.7')
                    engine.service = svc
                    req = http.request(stream=stream, tools=http.api_tools(api))
                    replies, records = [], []
                    with http.listener(svc) as listener:
                        first = http.post(listener, api, req); replies.append(first)
                        records.append({'request': deepcopy(req), 'ids': list(engine.last_prompt)})
                        if api == 'openai':
                            message = first['choices'][0]['message']; calls = message['tool_calls']
                            actual = [(c['function']['name'], json.loads(c['function']['arguments'])) for c in calls]
                            handoff = first['choices'][0]['finish_reason'] == 'tool_calls'
                            req['messages'] += [message, *[{'role': 'tool', 'tool_call_id': c['id'], 'content': 'result-'+str(i)}
                                for i, c in reversed(list(enumerate(calls)))]]
                        else:
                            calls = [b for b in first['content'] if b['type'] == 'tool_use']
                            actual = [(c['name'], c['input']) for c in calls]
                            handoff = first['stop_reason'] == 'tool_use'
                            req['messages'] += [{'role': 'assistant', 'content': first['content']}, {'role': 'user', 'content': [
                                {'type': 'tool_result', 'tool_use_id': c['id'], 'content': 'result-'+str(i)} for i, c in reversed(list(enumerate(calls)))]}]
                        second = http.post(listener, api, req); replies.append(second)
                        records.append({'request': deepcopy(req), 'ids': list(engine.last_prompt)})
                        http.assert_answer(api, second, 'Answer', 'Done')
                    checks = {'exact_json_values': canonical(actual) == canonical(expected), 'tool_handoff': handoff,
                              'unique_call_ids': len({c['id'] for c in calls}) == 2,
                              'result_order': '<response>result-0</response>\n<response>result-1</response>' in tokenizer.decode(engine.last_prompt),
                              'engine_closed_under_fifo': engine.closes == [True, True]}
                    row = {'fixture': index, 'api': api, 'stream': stream, 'checks': checks}
                    report['cycles'].append(row)
                    for record in records:
                        messages, tools, kwargs = svc.normalize_request(record['request'], api)
                        context = prepare_minimax_context({'messages': messages, 'tools': tools, **kwargs})
                        prompt_jobs.append({'template': source, 'context': context})
                        prompt_rows.append((row, record))
                    evidence_cycles.append({'wire': wire, 'replies': replies, 'records': records})
        native_prompts = run_lines(bins['template-json'], prompt_jobs)
        native_ids = run_lines(bins['tokenizer'], [{'text': p['rendered'], 'parse_special': True} for p in native_prompts], args.gguf.resolve())
        assert len(native_ids) == len(prompt_rows)+1 and native_ids[0]['ready']
        for i, ((row, record), native_prompt, tokens) in enumerate(zip(prompt_rows, native_prompts, native_ids[1:])):
            row['checks']['native_prompt_'+str(i%2)] = tokenizer.decode(record['ids']) == native_prompt['rendered']
            row['checks']['native_ids_'+str(i%2)] = record['ids'] == tokens['ids']
        saved = json.loads(args.replay.read_text(encoding='utf-8'))
        assert Path(saved['model']).resolve() == args.gguf.resolve()
        for reference in saved['source_reports']:
            assert sha(ROOT/reference['report']) == reference['sha256']
        shutil.copyfile(args.replay, args.out/'replay-input.json')
        report['replay_input_sha256'] = sha(args.replay)
        replay_replies = []
        for index, original in enumerate(saved['results']):
            ids = original['token_ids']; eos = original['stop_reason'] == 'eos'
            assert (ids[-1] == 200020 and ids.count(200020) == 1) if eos else 200020 not in ids
            assert b''.join(tokenizer.token_bytes(t) for t in ids).decode() == original['text']
            body = b''.join(tokenizer.token_bytes(t) for t in (ids[:-1] if eos else ids)).decode()
            parts = body.split('</think>', 1)
            for api in ('openai', 'anthropic'):
                for stream in (False, True):
                    engine = Engine(tokenizer, ['unused'], stop=200020)
                    engine.script = engine.scripts[0] = list(ids)
                    svc = Service(engine, tokenizer, MiniMaxAPITemplate(FIXTURE), model_name='minimax-m2.7'); engine.service = svc
                    with http.listener(svc) as listener:
                        reply = http.post(listener, api, http.request(stream=stream, max_tokens=len(ids)))
                        http.assert_answer(api, reply, parts[1] if len(parts) == 2 else '', parts[0])
                    count = reply['usage']['completion_tokens' if api == 'openai' else 'output_tokens']
                    finish = reply['choices'][0]['finish_reason'] if api == 'openai' else reply['stop_reason']
                    expected_finish = ('stop' if eos else 'length') if api == 'openai' else ('end_turn' if eos else 'max_tokens')
                    checks = {'exact_segments': True, 'usage': count == len(ids), 'finish': finish == expected_finish,
                              'closed_under_fifo': engine.closes == [True]}
                    report['replays'].append({'request': index, 'api': api, 'stream': stream, 'checks': checks, 'generated_tokens': count})
                    replay_replies.append(reply)
        save(args.out/'fixtures.json', {'native_requests': requests, 'native_rendered': native, 'cycles': evidence_cycles,
            'prompt_requests': prompt_jobs, 'native_prompts': native_prompts, 'native_ids': native_ids, 'replay_replies': replay_replies})
        for row in report['cycles']+report['replays']:
            row['pass'] = all(row['checks'].values())
        report['checks'] = sum(len(row['checks']) for row in report['cycles']+report['replays'])
        report['native_prompts'] = len(native_prompts)
        report['native_prompt_token_ids'] = sum(len(p['ids']) for p in native_ids[1:])
        report['pass'] = report['unit']['pass'] and all(row['pass'] for row in report['cycles']+report['replays'])
        report['artifact_sha256'] = {path.name: sha(path) for path in args.out.iterdir() if path.is_file()}
    except Exception as exc:
        report['error'] = str(exc)
        raise
    finally:
        save(args.out/'api-report.json', report)
    print(json.dumps({'pass': report['pass'], 'unit': report['unit'], 'cycles': len(report['cycles']),
                      'replays': len(report['replays']), 'checks': report['checks'], 'native_prompts': report['native_prompts']}))
    return 0 if report['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
