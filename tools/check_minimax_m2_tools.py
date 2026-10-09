"""MiniMax tool parser: native template/token bytes, history round trips, saved completions. No GPU/API."""
import argparse
import codecs
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
from serve.minimax_m2_history import MiniMaxChatTemplate, TEMPLATE_SHA256, prepare_minimax_context
from serve.minimax_m2_tools import MiniMaxOutputParser
from serve.test_minimax_m2_tools import signature
from tools.check_minimax_m2_oracles import provenance, run_lines
from tools.gguf_reader import GGUFFile
from tools.inspect_minimax_m2_gguf import inspect_model
from tools.minimax_m2_history_cases import numeric_cases
from tools.minimax_m2_tool_cases import TOOLS, EXPECTED
from tools.strata_tokenizer import Tokenizer


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def replay(chunks, tools):
    decoder = codecs.getincrementaldecoder('utf-8')('strict')
    parser = MiniMaxOutputParser(tools=tools)
    events = []
    for chunk in chunks:
        events.extend(parser.feed(decoder.decode(chunk)))
    events.extend(parser.feed(decoder.decode(b'', final=True)))
    events.extend(parser.finish())
    assert parser.buffered_chars == 0
    return parser, events


def corpus():
    cases = []
    def add(name, calls, tools):
        cases.append({'name': name, 'request': {'messages': [{'role': 'user', 'content': 'Q'},
            {'role': 'assistant', 'reasoning_content': 'R', 'content': 'Prefix',
             'tool_calls': [{'id': 'expected-'+str(i), 'name': n, 'arguments': a} for i, (n, a) in enumerate(calls)]}],
            'tools': tools, 'add_generation_prompt': False}})
    add('typed-multiple-calls', EXPECTED, TOOLS)
    for i, value in enumerate(['', 'null', 'true', '42', '"abc"', '{"x":1}', '  \t\r\n ',
                               'Уфа 中文 🧑🏽‍💻 e\u0301', '&lt;invoke&gt; &amp;', '<xml>literal</xml>',
                               'backslash \\n \\u003c', 'quote " and apostrophe \'' ]):
        add('raw-string-'+str(i), [('text', {'s': value})], TOOLS)
    for case in numeric_cases():
        request = case['request']
        arguments = request['messages'][-1]['tool_calls'][0]['arguments']
        tools = [{'type': 'function', 'function': {'name': 'f', 'parameters': {
            'type': 'object', 'properties': {'scalar': {'type': 'number'},
                'array': {'type': 'array', 'items': {'type': 'number'}},
                'object': {'type': 'object', 'properties': {'v': {'type': 'number'}}, 'required': ['v']}},
            'required': ['scalar', 'array', 'object'], 'additionalProperties': False}}}]
        add(case['name'], [('f', arguments)], tools)
    return deepcopy(cases)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--gguf', type=Path, required=True)
    cli.add_argument('--out', type=Path, required=True)
    cli.add_argument('--bin-dir', type=Path, default=ROOT/'build-local/minimax-m2-oracles/bin')
    cli.add_argument('--replay', type=Path, help='saved generation report containing results and model')
    args = cli.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    report = {'pass': False, 'scope': 'P2.4 standalone parser/native-template fixtures; no new inference, HTTP/API or speed measurement',
              'model': str(args.gguf.resolve()), 'ready_for_api': False, 'source_sha256': {}, 'cases': [], 'replay': []}
    try:
        report['packages'] = {name: importlib.metadata.version(name) for name in ['jsonschema', 'referencing', 'jinja2']}
        report['python'] = sys.version
        for rel in ['serve/minimax_m2_tools.py', 'serve/test_minimax_m2_tools.py', 'tools/minimax_m2_tool_cases.py',
                    'tools/check_minimax_m2_tools.py', 'serve/minimax_m2.py', 'serve/minimax_m2_history.py',
                    'serve/test_minimax_m2.py', 'serve/test_minimax_m2_history.py', 'tools/test_minimax_m2_tokenizer.py',
                    'tools/minimax_m2_history_cases.py', 'serve/frontend.py', 'tools/minimax_m2_template.py',
                    'tools/strata_tokenizer.py', 'tools/check_minimax_m2_oracles.py', 'tools/gguf_reader.py',
                    'tools/inspect_minimax_m2_gguf.py', 'tools/minimax_m2_loader_contract.py',
                    'serve/fixtures/minimax_m27_chat_template.jinja']:
            dest = args.out/'sources'/rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/rel, dest)
            report['source_sha256'][rel] = sha(dest)
        bins = {}
        suffix = '.exe' if sys.platform == 'win32' else ''
        for kind in ['template-json', 'tokenizer']:
            dest = args.out/('strata-minimax-m2-'+kind+suffix)
            shutil.copyfile(args.bin_dir/dest.name, dest)
            bins[kind] = dest.resolve()
        report['binary_sha256'] = {kind: sha(path) for kind, path in bins.items()}
        token_version = provenance(bins['tokenizer'])
        version = json.loads(subprocess.run([str(bins['template-json']), '--version'], capture_output=True, check=True, timeout=15).stdout)
        assert version == dict(token_version, renderer='native-jinja-json-float-roundtrip',
                               patches=token_version['patches']+';jinja-json-float-roundtrip')
        report['provenance'] = {'template-json': version, 'tokenizer': token_version}
        report['model_header_sha256'] = inspect_model(args.gguf)['header_sha256']
        source = GGUFFile(args.gguf).metadata['tokenizer.chat_template']
        assert source.encode() == (ROOT/'serve/fixtures/minimax_m27_chat_template.jinja').read_bytes()
        assert hashlib.sha256(source.encode()).hexdigest() == TEMPLATE_SHA256
        report['template_sha256'] = TEMPLATE_SHA256
        suite = unittest.defaultTestLoader.loadTestsFromNames(['serve.test_minimax_m2_tools', 'serve.test_minimax_m2',
                                                               'serve.test_minimax_m2_history', 'tools.test_minimax_m2_tokenizer'])
        log = io.StringIO()
        unit = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
        (args.out/'unit.log').write_text(log.getvalue(), encoding='utf-8')
        report['unit'] = {'pass': unit.wasSuccessful(), 'methods': unit.testsRun, 'failures': len(unit.failures), 'errors': len(unit.errors)}
        cases = corpus()
        contexts = [prepare_minimax_context(c['request']) for c in cases]
        native = run_lines(bins['template-json'], [{'template': source, 'context': c} for c in contexts])
        assert len(native) == len(cases)
        template, tokenizer = MiniMaxChatTemplate(source), Tokenizer.from_gguf(args.gguf)
        wires = []
        for case, actual in zip(cases, native):
            request = case['request']
            prefix = template.render({**request, 'messages': request['messages'][:-1]})+']~b]ai\n<think>\n'
            rendered = actual['rendered']
            assert rendered == template.render(request) and rendered.startswith(prefix) and rendered.endswith('[e~[\n')
            wires.append(rendered[len(prefix):-len('[e~[\n')])
        tokens = run_lines(bins['tokenizer'], [{'text': wire, 'parse_special': True} for wire in wires], args.gguf.resolve())
        assert len(tokens) == len(cases)+1 and tokens[0]['ready']
        histories = []
        for case, wire, encoded in zip(cases, wires, tokens[1:]):
            request, calls = case['request'], case['request']['messages'][-1]['tool_calls']
            expected = [('reasoning', 'R\n'), ('content', '\n\nPrefix\n')]+[('tool_call', c['name'], c['arguments']) for c in calls]
            ids = tokenizer.encode(wire, parse_special=True)
            pieces = [tokenizer.token_bytes(t) for t in ids]
            modes = {'whole': [wire.encode()], 'tokens': pieces, 'bytes': [bytes([b]) for b in wire.encode()]}
            checks = {'native_ids_equal': encoded['ids'] == ids,
                      'native_bytes_equal': encoded['decoded_hex'] == b''.join(pieces).hex() == wire.encode().hex()}
            parsed = None
            for mode, chunks in modes.items():
                parser, events = replay(chunks, request['tools'])
                checks[mode+'_exact_values'] = canonical(signature(events)) == canonical(expected)
                parsed = [e.call for e in events if e.call]
                checks[mode+'_ids'] = len({c.id for c in parsed}) == len(calls) and all(c.id.startswith('call_') for c in parsed)
                checks[mode+'_complete'] = parser.reasoning_complete and parser.tool_error is None
            assert parsed is not None
            messages = [request['messages'][0], {'role': 'assistant', 'content': 'Prefix', 'reasoning_content': 'R',
                'tool_calls': [{'id': c.id, 'function': {'name': c.name, 'arguments': json.dumps(c.arguments, ensure_ascii=False)}} for c in parsed]}]
            messages += [{'role': 'tool', 'tool_call_id': c.id, 'content': 'result-'+str(i)} for i, c in reversed(list(enumerate(parsed)))]
            history = prepare_minimax_context({'messages': messages, 'tools': request['tools']})
            histories.append(history)
            checks['history_result_order'] = [m['content'] for m in history['messages'] if m['role'] == 'tool'] == ['result-'+str(i) for i in range(len(calls))]
            report['cases'].append({'name': case['name'], 'checks': checks, 'wire_tokens': len(ids),
                                    'wire_sha256': hashlib.sha256(wire.encode()).hexdigest()})
        native_histories = run_lines(bins['template-json'], [{'template': source, 'context': h} for h in histories])
        assert len(native_histories) == len(histories)
        for row, history, actual in zip(report['cases'], histories, native_histories):
            rendered = template.render(history)
            row['checks']['history_native_equal'] = actual['rendered'] == rendered
            row['checks']['history_generation_prefix'] = rendered.endswith(']~b]ai\n<think>\n')
            row['pass'] = all(row['checks'].values())
        save(args.out/'fixtures.json', {'cases': cases, 'contexts': contexts, 'native': native, 'wires': wires, 'tokens': tokens,
                                       'histories': histories, 'native_histories': native_histories})
        if args.replay:
            saved = json.loads(args.replay.read_text(encoding='utf-8'))
            assert Path(saved['model']).resolve() == args.gguf.resolve()
            shutil.copyfile(args.replay, args.out/'replay-input.json')
            report['replay_input'] = {'path': str(args.replay), 'sha256': sha(args.replay)}
            for reference in saved.get('source_reports', []):
                assert sha(ROOT/reference['report']) == reference['sha256']
            for i, result in enumerate(saved['results']):
                ids = result['token_ids']
                eos = result['stop_reason'] == 'eos'
                assert (ids[-1] == 200020 and ids.count(200020) == 1) if eos else 200020 not in ids
                pieces = [tokenizer.token_bytes(t) for t in ids]
                assert b''.join(pieces).decode('utf-8') == result['text']
                if eos:
                    pieces = pieces[:-1]
                text = b''.join(pieces).decode('utf-8')
                expected = [('reasoning', text.split('</think>', 1)[0])]
                if '</think>' in text:
                    expected.append(('content', text.split('</think>', 1)[1]))
                checks = {}
                for mode, chunks in [('tokens', pieces), ('bytes', [bytes([b]) for p in pieces for b in p])]:
                    for enabled in [False, True]:
                        parser, events = replay(chunks, TOOLS if enabled else [])
                        checks[mode+'_tools='+str(enabled)] = canonical(signature(events)) == canonical(expected) and parser.tool_error is None
                report['replay'].append({'request': i, 'pass': all(checks.values()), 'checks': checks, 'generated_tokens': len(ids)})
        report['native_tool_groups'] = len(cases)
        report['native_history_prompts'] = len(histories)
        report['wire_tokens_compared'] = sum(c['wire_tokens'] for c in report['cases'])
        report['checks'] = sum(len(c['checks']) for c in report['cases']+report['replay'])
        report['pass'] = report['unit']['pass'] and all(c['pass'] for c in report['cases']+report['replay'])
        report['artifact_sha256'] = {p.name: sha(p) for p in args.out.glob('*') if p.is_file()}
    except Exception as exc:
        report['error'] = str(exc)
        raise
    finally:
        save(args.out/'tools-report.json', report)
    print(json.dumps({k: report[k] for k in ['pass', 'unit', 'native_tool_groups', 'native_history_prompts', 'wire_tokens_compared', 'checks']}))
    return 0 if report['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
