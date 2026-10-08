"""Compare MiniMax header registration, token IDs and template bytes without inference."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.gguf_reader import GGUFFile
from tools.inspect_minimax_m2_gguf import inspect_model, file_hash, protect_output
from tools.minimax_m2_loader_contract import LOADER_SHA, ARCHIVE_SHA256, TEMPLATE_SHA256
from tools.strata_tokenizer import Tokenizer


def provenance(oracle, *, template=False):
    actual = json.loads(subprocess.run([str(oracle), '--version'], capture_output=True, check=True, timeout=15).stdout)
    expected = {'architecture': 'minimax-m2', 'requested_revision': LOADER_SHA,
                'archive_sha256': ARCHIVE_SHA256, 'patches': 'none'}
    if template:
        expected['renderer'] = 'native-jinja-raw'
    if actual != expected:
        raise ValueError('Unexpected MiniMax oracle provenance')
    return actual


def run_lines(oracle, requests, gguf=None):
    command = [str(oracle)] + (['--gguf', str(gguf)] if gguf else [])
    data = ''.join(json.dumps(r, ensure_ascii=True)+'\n' for r in requests)+'QUIT\n'
    result = subprocess.run(command, input=data.encode('utf-8'), capture_output=True, check=True, timeout=120)
    return [json.loads(line) for line in result.stdout.splitlines()]


def validate_registration(report, inventory):
    if {k: report.get(k) for k in ('architecture', 'requested_revision', 'archive_sha256', 'patches')} != {
            'architecture': 'minimax-m2', 'requested_revision': LOADER_SHA, 'archive_sha256': ARCHIVE_SHA256, 'patches': 'none'}:
        raise ValueError('Wrong registration provenance')
    if report.get('status') != 'pass':
        raise ValueError('Native registration did not pass')
    h = inventory['loader_contract']['metadata']
    expected = {t['name']: {'shape': t['shape']+[1]*(4-len(t['shape'])), 'type_id': t['type_id'], 'bytes': t['bytes']}
                for t in inventory['tensors']}
    runs = report.get('runs', [])
    if len(runs) != 2 or [r.get('load_mtp') for r in runs] != [False, True]:
        raise ValueError('Expected both load_mtp flag settings')
    for r in runs:
        if (r['main_blocks'], r['all_blocks'], r['nextn_blocks'], r['allocated_weight_bytes'], r['logical_bytes'], r['swa_window']) != (
                h['block_count'], h['block_count'], 0, 0, inventory['payload_bytes'], 0):
            raise ValueError('Registration allocation/geometry differs')
        actual = {t['name']: {k: t[k] for k in ('shape', 'type_id', 'bytes')} for t in r['tensors']}
        if len(actual) != len(r['tensors']) or actual != expected:
            raise ValueError('Compiled tensor names/shapes/types/bytes differ')
        expected_layers = [{'layer': i, 'swa': False, 'heads': h['attention.head_count'],
                            'kv_heads': h['attention.head_count_kv'], 'key_dim': h['attention.key_length'],
                            'value_dim': h['attention.value_length'], 'rope_dim': h['rope.dimension_count']}
                           for i in range(h['block_count'])]
        if r['layers'] != expected_layers:
            raise ValueError('Compiled per-layer attention geometry differs')


def check_loader(gguf, oracle):
    provenance(oracle)
    inventory = inspect_model(gguf)
    result = subprocess.run([str(oracle), '--gguf', str(gguf)], capture_output=True, check=True, timeout=120)
    report = json.loads(result.stdout)
    validate_registration(report, inventory)
    report.update(gguf_header_sha256=inventory['header_sha256'],
                  binary_sha256=file_hash(oracle, oracle.stat().st_size), tensor_count=inventory['tensor_count'],
                  mtp_flag_does_not_create_weights=True)
    return report


def token_corpus(meta):
    texts = ['', 'Hello, world!', 'Привет! Как работает конвейер?', '你好。日本語 한국어',
             'مرحبا بالعالم Ελληνικά हिन्दी', '1234567890 ١٢٣٤٥٦٧ １２３４５６７ 3.14159',
             'a\u0301_б e\u0308 !ABC /test +foo 123中文456', ' leading  spaces\t\tend  ',
             '\r\n\n\r abc\n\n', '\x00\x01\x1b\x7f', '👨‍👩‍👧‍👦 🚀 🏳️‍🌈 🧑🏽‍💻',
             '\u2028\u2029\u00a0\u200b\ufeff', 'abc '*257,
             'def fib(n):\n    return n if n < 2 else fib(n-1)+fib(n-2)\n',
             ']~!b[]~b]user\nПривет[e~[]~b]ai\n<think></think>',
             '<minimax:tool_call><invoke name="weather"><parameter name="city">Екатеринбург</parameter></invoke></minimax:tool_call>',
             'normal text </s> after </s>', ']~b]ai\n', "I'm WE'RE he'll they've can't"]
    cases = [(f'text-{i}', t) for i, t in enumerate(texts)]
    for i, (token, kind) in enumerate(zip(meta['tokenizer.ggml.tokens'], meta['tokenizer.ggml.token_type'])):
        if kind in (3, 4):
            cases.append((f'special-{i}', 'A'+token+'Б'+token))
    rng = random.Random(2606)
    pieces = ['abc', '!ABC', '12345', '你好', 'Привет', 'a\u0301', ' ', '\r\n', '\t',
              '🧑🏽‍💻', '<think>', '[e~[', '\x00', 'ß', '१२३४५', '</s>']
    cases += [(f'mixed-{i}', ''.join(rng.choices(pieces, k=rng.randrange(1, 30)))) for i in range(256)]
    return [{'name': n, 'text': t, 'parse_special': flag, 'add_special': add_special} for n, t in cases for flag in (False, True) for add_special in (False, True)]


def check_tokenizer(gguf, oracle):
    prov = provenance(oracle)
    g = GGUFFile(gguf)
    tokenizer = Tokenizer.from_gguf(gguf)
    cases = token_corpus(g.metadata)
    rows = run_lines(oracle, cases, gguf)
    if len(rows) != len(cases)+1:
        raise ValueError('Wrong tokenizer response count')
    info = rows[0]
    if info.get('ready') is not True or info.get('add_bos') is not False or info.get('add_eos') is not False or info.get('vocab_size') != len(tokenizer.tokens) or info.get('eos') != 200020:
        raise ValueError('Unexpected vocabulary/ready response')
    # Record native FIM/EOS heuristics separately from the future serving policy.
    # This is not approval of the future engine's stop policy.
    if info.get('eog_ids') != [200004, 200005, 200020] or info.get('bos') != 200034:
        raise ValueError('Unreviewed native BOS/EOG heuristics')
    results = []
    for case, native in zip(cases, rows[1:]):
        ids = tokenizer.encode(case['text'], parse_special=case['parse_special'])
        decoded = b''.join(tokenizer.token_bytes(i) for i in ids)
        ok = native.get('ids') == ids and native.get('decoded_hex') == decoded.hex() == case['text'].encode('utf-8').hex()
        row = {'name': case['name'], 'parse_special': case['parse_special'], 'add_special': case['add_special'], 'pass': ok, 'tokens': len(ids),
               'text_sha256': hashlib.sha256(case['text'].encode()).hexdigest()}
        if not ok:
            row.update(text=case['text'], python_ids=ids, native=native, python_hex=decoded.hex())
        results.append(row)
    failures = sum(not r['pass'] for r in results)
    return {'status': 'pass' if not failures else 'fail', 'scope': 'token IDs/decoded bytes; no generation or runtime stop-policy validation',
            'oracle_provenance': prov, 'binary_sha256': file_hash(oracle, oracle.stat().st_size),
            'gguf_header_sha256': file_hash(gguf, g.header_end), 'native_info': info,
            'native_eog_tokens': {str(i): tokenizer.tokens[i] for i in info['eog_ids']},
            'runtime_stop_policy_validated': False, 'case_count': len(results), 'mismatch_count': failures, 'cases': results}


def template_corpus():
    user = {'role': 'user', 'content': 'Привет, 你好!'}
    assistant = {'role': 'assistant', 'content': 'Ответ', 'reasoning_content': 'Думаю'}
    tool = {'type': 'function', 'function': {'name': 'weather', 'description': 'Погода',
            'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}}}}
    args = {'city': 'Екатеринбург', 'n': 2, 'yes': True, 'none': None, 'nested': {'a': [1, '中']}, 's': 'a\nb<xml>'}
    flat = {'role': 'assistant', 'content': '', 'tool_calls': [{'name': 'weather', 'arguments': args}]}
    nested = {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'c1', 'type': 'function',
              'function': {'name': 'weather', 'arguments': args}}]}
    string_call = deepcopy(nested)
    string_call['tool_calls'][0]['function']['arguments'] = json.dumps(args, ensure_ascii=False)
    result = {'role': 'tool', 'tool_call_id': 'c1', 'content': '{"t":5}'}
    bases = [
        ('empty', {'messages': []}), ('user', {'messages': [user]}),
        ('system', {'messages': [{'role': 'system', 'content': 'Кратко', 'current_date': '2026-10-07',
                                 'current_location': 'Екатеринбург'}, user]}),
        ('identity', {'messages': [user], 'model_identity': 'Локальный помощник'}),
        ('history', {'messages': [user, assistant, user]}),
        ('reasoning', {'messages': [user, assistant]}),
        ('inline_reasoning', {'messages': [user, {'role': 'assistant', 'content': '<think>Думаю</think>Ответ'}]}),
        ('empty_assistant', {'messages': [user, {'role': 'assistant', 'content': ''}]}),
        ('text_parts', {'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'Первый'}, ' second']}]}),
        ('tools', {'messages': [user], 'tools': [tool]}),
        ('flat_call', {'messages': [user, flat], 'tools': [tool]}),
        ('nested_call', {'messages': [user, nested], 'tools': [tool]}),
        ('string_args', {'messages': [user, string_call, result], 'tools': [tool]}),
        ('results', {'messages': [user, nested, result], 'tools': [tool]}),
        ('after_result', {'messages': [user, nested, result, assistant], 'tools': [tool]}),
        ('multiple_calls', {'messages': [user, {**flat, 'tool_calls': flat['tool_calls']*2}, result, result], 'tools': [tool]}),
    ]
    cases = []
    for name, base in bases:
        for prompt in (False, True):
            for thinking in (None, False, True):
                context = deepcopy(base)
                context['add_generation_prompt'] = prompt
                if thinking is not None:
                    context['enable_thinking'] = thinking
                cases.append({'name': f'{name}/prompt={prompt}/thinking={thinking}', 'context': context})
    return cases


def check_template(gguf, oracle, token_oracle):
    from tools.minimax_m2_template import renderer, text_context
    prov = provenance(oracle, template=True)
    provenance(token_oracle)
    g = GGUFFile(gguf)
    source = g.metadata['tokenizer.chat_template']
    if hashlib.sha256(source.encode()).hexdigest() != TEMPLATE_SHA256:
        raise ValueError('Unreviewed MiniMax template')
    cases = template_corpus()
    requests = [{'template': source, 'context': text_context(c['context'])} for c in cases]
    native = run_lines(oracle, requests)
    if len(native) != len(cases):
        raise ValueError('Wrong template response count')
    py_template, tokenizer = renderer(source), Tokenizer.from_gguf(gguf)
    encoded = run_lines(token_oracle, [{'text': r.get('rendered', ''), 'parse_special': True} for r in native], gguf)
    if len(encoded) != len(cases)+1:
        raise ValueError('Wrong template tokenizer response count')
    results = []
    for c, req, actual, token in zip(cases, requests, native, encoded[1:]):
        py = py_template.render(**req['context'])
        ids = tokenizer.encode(py, parse_special=True)
        ok = actual.get('rendered') == py and token.get('ids') == ids and token.get('decoded_hex') == py.encode().hex()
        row = {'name': c['name'], 'pass': ok, 'render_sha256': hashlib.sha256(py.encode()).hexdigest(),
               'token_count': len(ids), 'prompt_ids': ids}
        if not ok:
            row.update(python=py, native=actual, python_ids=ids, native_tokens=token)
        results.append(row)
    # Independent expected bytes, so agreement of two renderers alone is insufficient.
    expected = (']~!b[]~b]system\nYou are a helpful assistant. Your name is MiniMax-M2.7 and is built by MiniMax.'
                '[e~[\n]~b]user\nПривет, 你好![e~[\n]~b]ai\n<think>\n')
    for flag in (False, True):
        actual = py_template.render(messages=[{'role': 'user', 'content': 'Привет, 你好!'}],
                                    add_generation_prompt=True, enable_thinking=flag)
        results.append({'name': f'manual-prefix/thinking={flag}', 'pass': actual == expected})
    rendered = {c['name']: r.get('rendered', '') for c, r in zip(cases, native)}
    flat = rendered['flat_call/prompt=False/thinking=None']
    nested = rendered['nested_call/prompt=False/thinking=None']
    multiple = rendered['multiple_calls/prompt=False/thinking=None']
    results += [
        {'name': 'flat-nested-equivalence', 'pass': flat == nested},
        {'name': 'manual-tool-xml', 'pass': '<invoke name="weather">\n<parameter name="city">Екатеринбург</parameter>' in flat
         and '<parameter name="nested">{"a": [1, "中"]}</parameter>' in flat},
        {'name': 'group-consecutive-tool-results', 'pass': multiple.count(']~b]tool') == 1
         and multiple.count('<response>{"t":5}</response>') == 2},
        {'name': 'discard-reasoning-before-last-user', 'pass': 'Думаю' not in rendered['history/prompt=False/thinking=None']},
        {'name': 'preserve-recent-reasoning', 'pass': '<think>\nДумаю\n</think>\n\nОтвет' in rendered['reasoning/prompt=False/thinking=None']},
    ]
    failures = sum(not r['pass'] for r in results)
    return {'status': 'pass' if not failures else 'fail',
            'scope': 'Python/native Jinja after text/function normalization, prompt IDs; no HTTP or generation',
            'oracle_provenance': prov, 'binary_sha256': file_hash(oracle, oracle.stat().st_size),
            'tokenizer_binary_sha256': file_hash(token_oracle, token_oracle.stat().st_size),
            'gguf_header_sha256': file_hash(gguf, g.header_end), 'template_sha256': TEMPLATE_SHA256,
            'enable_thinking_changes_template': False,
            'case_count': len(results), 'mismatch_count': failures, 'cases': results}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gguf', type=Path, required=True)
    parser.add_argument('--bin-dir', type=Path, required=True)
    parser.add_argument('--kind', choices=('loader', 'tokenizer', 'template'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    oracle = (args.bin_dir / ('strata-minimax-m2-'+args.kind+('.exe' if sys.platform == 'win32' else ''))).resolve()
    token_oracle = (args.bin_dir / ('strata-minimax-m2-tokenizer'+('.exe' if sys.platform == 'win32' else ''))).resolve()
    try:
        output = protect_output(args.output, args.gguf, oracle, token_oracle)
        inspect_model(args.gguf)  # Admission precedes every invocation, not just the loader check.
        if args.kind == 'loader':
            report = check_loader(args.gguf.resolve(), oracle)
        elif args.kind == 'tokenizer':
            report = check_tokenizer(args.gguf.resolve(), oracle)
        else:
            report = check_template(args.gguf.resolve(), oracle, token_oracle)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        print(report['status'], args.kind, 'mismatches', report.get('mismatch_count', 0), output)
        return int(report['status'] != 'pass')
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as exc:
        print('MiniMax oracle check failed:', exc, file=sys.stderr)
        if isinstance(exc, subprocess.CalledProcessError):
            print((exc.stderr or b'')[-3000:].decode(errors='replace'), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
