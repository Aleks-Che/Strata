"""Compare MiMo header registration, token IDs and template bytes without inference."""
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
from tools.inspect_mimo2_gguf import inspect_model, file_hash, protect_output
from tools.mimo2_loader_contract import LOADER_SHA, ARCHIVE_SHA256, TEMPLATE_SHA256
from tools.mimo2_tokenizer import from_gguf as mimo_tokenizer


def provenance(oracle, *, template=False):
    actual = json.loads(subprocess.run([str(oracle), '--version'], capture_output=True, check=True, timeout=15).stdout)
    expected = {'architecture': 'mimo2', 'requested_revision': LOADER_SHA,
                'archive_sha256': ARCHIVE_SHA256, 'patches': 'none'}
    if template:
        expected['renderer'] = 'native-jinja-raw'
    if actual != expected:
        raise ValueError('Unexpected MiMo oracle provenance')
    return actual


def run_lines(oracle, requests, gguf=None):
    command = [str(oracle)] + (['--gguf', str(gguf)] if gguf else [])
    data = ''.join(json.dumps(r, ensure_ascii=True)+'\n' for r in requests)+'QUIT\n'
    result = subprocess.run(command, input=data.encode('utf-8'), capture_output=True, check=True, timeout=120)
    return [json.loads(line) for line in result.stdout.splitlines()]


def validate_registration(report, inventory):
    if {k: report.get(k) for k in ('architecture', 'requested_revision', 'archive_sha256', 'patches')} != {
            'architecture': 'mimo2', 'requested_revision': LOADER_SHA, 'archive_sha256': ARCHIVE_SHA256, 'patches': 'none'}:
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
                h['block_count'], h['block_count'], 0, 0, inventory['payload_bytes'], h['attention.sliding_window']):
            raise ValueError('Registration allocation/geometry differs')
        actual = {t['name']: {k: t[k] for k in ('shape', 'type_id', 'bytes')} for t in r['tensors']}
        if len(actual) != len(r['tensors']) or actual != expected:
            raise ValueError('Compiled tensor names/shapes/types/bytes differ')
        expected_layers = [{'layer': i, 'swa': bool(s), 'heads': h['attention.head_count'],
                            'kv_heads': h['attention.head_count_kv'][i], 'key_dim': h['attention.key_length'],
                            'value_dim': h['attention.value_length'], 'rope_dim': h['rope.dimension_count']}
                           for i, s in enumerate(h['attention.sliding_window_pattern'])]
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
             '<|im_start|>user\nПривет<|im_end|><|im_start|>assistant\n<think></think>',
             '<tool_call><function=weather><parameter=city>Екатеринбург</parameter></function></tool_call>',
             'normal text </s> after </s>', '<|mimo_audio_eod|>', "I'm WE'RE he'll they've can't"]
    cases = [(f'text-{i}', t) for i, t in enumerate(texts)]
    for i, (token, kind) in enumerate(zip(meta['tokenizer.ggml.tokens'], meta['tokenizer.ggml.token_type'])):
        if kind in (3, 4):
            cases.append((f'special-{i}', 'A'+token+'Б'+token))
    rng = random.Random(2606)
    pieces = ['abc', '!ABC', '12345', '你好', 'Привет', 'a\u0301', ' ', '\r\n', '\t',
              '🧑🏽‍💻', '<think>', '<|im_end|>', '\x00', 'ß', '१२३४५', '</s>']
    cases += [(f'mixed-{i}', ''.join(rng.choices(pieces, k=rng.randrange(1, 30)))) for i in range(256)]
    return [{'name': n, 'text': t, 'parse_special': flag} for n, t in cases for flag in (False, True)]


def check_tokenizer(gguf, oracle):
    prov = provenance(oracle)
    g = GGUFFile(gguf)
    tokenizer = mimo_tokenizer(gguf)
    cases = token_corpus(g.metadata)
    rows = run_lines(oracle, cases, gguf)
    if len(rows) != len(cases)+1:
        raise ValueError('Wrong tokenizer response count')
    info = rows[0]
    if info.get('ready') is not True or info.get('add_special') is not False or info.get('vocab_size') != len(tokenizer.tokens) or info.get('eos') != 151645:
        raise ValueError('Unexpected vocabulary/ready response')
    # Record the dependency's heuristics, including normal-looking </s> and FIM.
    # This is not approval of the future engine's stop policy.
    if info.get('eog_ids') != [128247, 151643, 151645, 151662, 151663, 151664] or info.get('bos') != 11:
        raise ValueError('Unreviewed native BOS/EOG heuristics')
    results = []
    for case, native in zip(cases, rows[1:]):
        ids = tokenizer.encode(case['text'], parse_special=case['parse_special'])
        decoded = b''.join(tokenizer.token_bytes(i) for i in ids)
        ok = native.get('ids') == ids and native.get('decoded_hex') == decoded.hex() == case['text'].encode('utf-8').hex()
        row = {'name': case['name'], 'parse_special': case['parse_special'], 'pass': ok, 'tokens': len(ids),
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
    call = {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'c1', 'type': 'function',
            'function': {'name': 'weather', 'arguments': args}}]}
    string_call = deepcopy(call)
    string_call['tool_calls'][0]['function']['arguments'] = json.dumps(args, ensure_ascii=False)
    custom = {'role': 'assistant', 'content': '', 'tool_calls': [{'custom': {'name': 'shell', 'input': 'echo test'}}]}
    history = [user, call, {'role': 'tool', 'tool_call_id': 'c1', 'content': '{"t":5}'}, assistant]
    bases = [
        ('empty', {'messages': []}), ('user', {'messages': [user]}),
        ('systems', {'messages': [{'role': 'system', 'content': 'Кратко'}, {'role': 'system', 'content': '中文'}, user]}),
        ('history', {'messages': [user, assistant, user]}),
        ('empty_assistant', {'messages': [user, {'role': 'assistant', 'content': ''}]}),
        ('text_parts', {'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'Первый'}, ' second']}]}),
        ('tools', {'messages': [user], 'tools': [tool]}), ('call', {'messages': [user, call], 'tools': [tool]}),
        ('string_args', {'messages': [user, string_call], 'tools': [tool]}),
        ('custom_input', {'messages': [user, custom]}), ('results', {'messages': history[:-1], 'tools': [tool]}),
        ('after_result', {'messages': history, 'tools': [tool]}),
        ('multiple_calls', {'messages': [user, {**call, 'tool_calls': call['tool_calls']*2}], 'tools': [tool]}),
        ('message_tools', {'messages': [{**user, 'tools': [tool]}]}),
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
    from tools.mimo2_template import renderer, text_context
    prov = provenance(oracle, template=True)
    provenance(token_oracle)
    g = GGUFFile(gguf)
    source = g.metadata['tokenizer.chat_template']
    if hashlib.sha256(source.encode()).hexdigest() != TEMPLATE_SHA256:
        raise ValueError('Unreviewed MiMo template')
    cases = template_corpus()
    requests = [{'template': source, 'context': text_context(c['context'])} for c in cases]
    native = run_lines(oracle, requests)
    if len(native) != len(cases):
        raise ValueError('Wrong template response count')
    py_template, tokenizer = renderer(source), mimo_tokenizer(gguf)
    encoded = run_lines(token_oracle, [{'text': r.get('rendered', ''), 'parse_special': True} for r in native], gguf)
    if len(encoded) != len(cases)+1:
        raise ValueError('Wrong template tokenizer response count')
    results = []
    for c, req, actual, token in zip(cases, requests, native, encoded[1:]):
        py = py_template.render(**req['context'])
        ids = tokenizer.encode(py, parse_special=True)
        ok = actual.get('rendered') == py and token.get('ids') == ids and token.get('decoded_hex') == py.encode().hex()
        row = {'name': c['name'], 'pass': ok, 'render_sha256': hashlib.sha256(py.encode()).hexdigest(), 'token_count': len(ids)}
        if not ok:
            row.update(python=py, native=actual, python_ids=ids, native_tokens=token)
        results.append(row)
    expected = '<|im_start|>user\nПривет, 你好!<|im_end|><|im_start|>assistant\n<think></think>'
    actual = py_template.render(messages=[{'role': 'user', 'content': 'Привет, 你好!'}], add_generation_prompt=True, enable_thinking=False)
    results.append({'name': 'manual-no-think-prefix', 'pass': actual == expected})
    failures = sum(not r['pass'] for r in results)
    return {'status': 'pass' if not failures else 'fail', 'scope': 'Python/native Jinja and token IDs; text-only guard; no HTTP or generation',
            'oracle_provenance': prov, 'binary_sha256': file_hash(oracle, oracle.stat().st_size),
            'tokenizer_binary_sha256': file_hash(token_oracle, token_oracle.stat().st_size),
            'gguf_header_sha256': file_hash(gguf, g.header_end), 'template_sha256': TEMPLATE_SHA256,
            'case_count': len(results), 'mismatch_count': failures, 'cases': results}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gguf', type=Path, required=True)
    parser.add_argument('--bin-dir', type=Path, required=True)
    parser.add_argument('--kind', choices=('loader', 'tokenizer', 'template'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    oracle = (args.bin_dir / ('strata-mimo2-'+args.kind+('.exe' if sys.platform == 'win32' else ''))).resolve()
    token_oracle = (args.bin_dir / ('strata-mimo2-tokenizer'+('.exe' if sys.platform == 'win32' else ''))).resolve()
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
        print('MiMo oracle check failed:', exc, file=sys.stderr)
        if isinstance(exc, subprocess.CalledProcessError):
            print((exc.stderr or b'')[-3000:].decode(errors='replace'), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
