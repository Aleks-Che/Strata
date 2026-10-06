"""Compare the embedded Hy3 template against pinned native Jinja and BPE oracles."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from jinja2.sandbox import ImmutableSandboxedEnvironment

if __package__:
    from .check_hy3_tokenizer import validate_provenance
    from .gguf_reader import GGUFFile
    from .hy3_loader_contract import ARCHIVE_SHA256, LOADER_SHA
    from .inspect_hy3_gguf import file_hash, protect_output
    from .strata_tokenizer import Tokenizer
else:
    from check_hy3_tokenizer import validate_provenance
    from gguf_reader import GGUFFile
    from hy3_loader_contract import ARCHIVE_SHA256, LOADER_SHA
    from inspect_hy3_gguf import file_hash, protect_output
    from strata_tokenizer import Tokenizer

TEMPLATE_SHA = '7fc351fee674c13754656ba7f33a3ca426bfb7231039f48444360a3f3c5ecf3e'


def normalize_context(original):
    context = deepcopy(original)
    for message in context['messages']:
        for call in message.get('tool_calls') or []:
            function = call['function']
            args = function.get('arguments')
            if isinstance(args, str):
                args = json.loads(args)
                if not isinstance(args, dict):
                    raise ValueError('tool arguments must be a JSON object')
                function['arguments'] = args
    return context


def renderer(source):
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                       extensions=['jinja2.ext.loopcontrols'])
    env.filters['tojson'] = lambda value, **kw: json.dumps(value, ensure_ascii=kw.pop('ensure_ascii', False), **kw)

    def fail(message):
        raise ValueError(message)

    env.globals['raise_exception'] = fail
    return env.from_string(source)


def corpus():
    user = {'role': 'user', 'content': 'Привет, 你好!'}
    assistant = {'role': 'assistant', 'content': 'Ответ', 'reasoning_content': 'Думаю'}
    tool = {'type': 'function', 'function': {'name': 'weather', 'description': 'Погода',
            'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}}}}
    arguments = {'city': 'Екатеринбург', 'n': 2, 'yes': True, 'empty': None, 'nested': {'a': [1, '中']}, 'lines': 'a\nb'}
    call = {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'c1', 'type': 'function',
            'function': {'name': 'weather', 'arguments': arguments}}]}
    history = [user, call, {'role': 'tool', 'tool_call_id': 'c1', 'content': '{"temperature":5}'}, assistant]
    bases = [
        ('user', {'messages': [user]}),
        ('systems', {'messages': [{'role': 'system', 'content': 'Будь краток.'}, {'role': 'system', 'content': 'Use JSON.'}, user]}),
        ('multiturn', {'messages': [user, assistant, user]}),
        ('preserved', {'messages': [user, assistant, user], 'preserved_thinking': True}),
        ('training', {'messages': [user, assistant], 'is_training': True}),
        ('raw_assistant', {'messages': [user, assistant], 'raw_last_assistant': True}),
        ('text_parts', {'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'Один'}, ' two']}]}),
        ('tools', {'messages': [user], 'tools': [tool]}),
        ('call', {'messages': [user, call], 'tools': [tool]}),
        ('result', {'messages': history[:-1], 'tools': [tool]}),
        ('after_result', {'messages': history, 'tools': [tool]}),
        ('next_user_after_result', {'messages': history[:-1] + [user], 'tools': [tool]}),
        ('multiple_results', {'messages': history[:-1] + [{'role': 'tool', 'content': 'second'}], 'tools': [tool]}),
        ('null_content', {'messages': [{'role': 'assistant', 'content': None}]}),
        ('fallback', {'messages': [user], 'fallback_strategy': 'reasoning_toolcall_retry'}),
    ]
    string_call = deepcopy(call)
    string_call['tool_calls'][0]['function']['arguments'] = json.dumps(arguments, ensure_ascii=False)
    bases.append(('string_arguments', {'messages': [user, string_call], 'tools': [tool]}))
    double_call = deepcopy(call)
    double_call['tool_calls'].append({'id': 'c2', 'type': 'function', 'function': {'name': 'other', 'arguments': {'x': 0}}})
    bases.append(('two_calls', {'messages': [user, double_call], 'tools': [tool]}))
    cases = []
    for name, base in bases:
        for generation in (False, True):
            for effort in ('default', 'no_think', 'low', 'high'):
                ctx = {'tools': None, **deepcopy(base), 'add_generation_prompt': generation}
                if effort != 'default':
                    ctx['reasoning_effort'] = effort
                cases.append({'name': f'{name}/gen={generation}/effort={effort}', 'context': ctx})
    for effort in ('medium', '', None, 1):
        cases.append({'name': f'bad-effort-{effort}', 'context': {'messages': [user], 'reasoning_effort': effort}, 'error': True})
    for args in ('{bad', '[]', '42'):
        invalid_call = deepcopy(string_call)
        invalid_call['tool_calls'][0]['function']['arguments'] = args
        cases.append({'name': 'bad-args-' + args, 'context': {'messages': [user, invalid_call], 'tools': [tool]}, 'error': True})
    return cases


def compare(gguf, oracle, tokenizer_oracle):
    provenance = json.loads(subprocess.run([str(oracle), '--version'], capture_output=True, check=True, timeout=15).stdout)
    if provenance != {'architecture': 'hy_v3', 'requested_revision': LOADER_SHA, 'archive_sha256': ARCHIVE_SHA256,
                      'renderer': 'native-jinja-with-tool-json-normalization'}:
        raise ValueError('Wrong Hy3 template oracle provenance')
    token_version = json.loads(subprocess.run([str(tokenizer_oracle), '--version'], capture_output=True, check=True, timeout=15).stdout)
    validate_provenance(token_version)
    header = GGUFFile(gguf)
    source = header.metadata['tokenizer.chat_template']
    if hashlib.sha256(source.encode()).hexdigest() != TEMPLATE_SHA:
        raise ValueError('Unreviewed Hy3 chat template')
    cases = corpus()
    request = ''.join(json.dumps({'template': source, 'context': c['context']}, ensure_ascii=True) + '\n' for c in cases)
    native = subprocess.run([str(oracle)], input=(request + 'QUIT\n').encode(), capture_output=True, check=True, timeout=120)
    rows = [json.loads(line) for line in native.stdout.splitlines()]
    if len(rows) != len(cases):
        raise ValueError('Wrong template response count')
    template, tokenizer = renderer(source), Tokenizer.from_gguf(gguf)
    token_request = ''.join(json.dumps({'text': r.get('rendered', ''), 'parse_special': True}, ensure_ascii=True) + '\n' for r in rows)
    encoded = subprocess.run([str(tokenizer_oracle), '--gguf', str(gguf)], input=(token_request + 'QUIT\n').encode(),
                             capture_output=True, check=True, timeout=120)
    token_rows = [json.loads(line) for line in encoded.stdout.splitlines()]
    if len(token_rows) != len(cases) + 1:
        raise ValueError('Wrong tokenization response count')
    results = []
    for case, row, tokens in zip(cases, rows, token_rows[1:]):
        try:
            py = template.render(**normalize_context(case['context']))
            error = None
        except Exception as exc:
            py, error = None, str(exc)
        if case.get('error'):
            result = {'name': case['name'], 'pass': error is not None and 'error' in row,
                      'python_error': error, 'native_error': row.get('error')}
        elif error is not None:
            result = {'name': case['name'], 'pass': False, 'python_error': error, 'native': row}
        else:
            ids = tokenizer.encode(py, parse_special=True)
            result = {'name': case['name'], 'pass': row.get('rendered') == py and tokens.get('ids') == ids
                      and tokens.get('decoded_hex') == py.encode().hex(),
                      'render_sha256': hashlib.sha256(py.encode()).hexdigest(), 'token_count': len(ids)}
            if not result['pass']:
                result.update(python=py, native=row, native_tokens=tokens, python_ids=ids)
        results.append(result)
    manual = ('<｜hy_begin_of_sentence:opensource｜><｜reasoning_mode:opensource｜>reasoning_effort:no_think'
              '<｜hy_User:opensource｜>Привет, 你好!<｜hy_Assistant:opensource｜><think:opensource></think:opensource>')
    actual = template.render(messages=[{'role': 'user', 'content': 'Привет, 你好!'}], tools=None, add_generation_prompt=True)
    results.append({'name': 'manual-default-prefix', 'pass': actual == manual})
    failures = sum(not r['pass'] for r in results)
    return {'status': 'pass' if not failures else 'fail', 'scope': 'native/Python rendering and token IDs; no API or generation',
            'case_count': len(results), 'mismatch_count': failures, 'template_sha256': TEMPLATE_SHA,
            'gguf_header_sha256': file_hash(gguf, header.header_end), 'oracle_provenance': provenance,
            'template_binary_sha256': file_hash(oracle, oracle.stat().st_size),
            'tokenizer_binary_sha256': file_hash(tokenizer_oracle, tokenizer_oracle.stat().st_size), 'cases': results}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gguf', type=Path, required=True)
    parser.add_argument('--oracle', type=Path, required=True)
    parser.add_argument('--tokenizer-oracle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        output = protect_output(args.output, args.gguf, args.oracle, args.tokenizer_oracle)
        report = compare(args.gguf.resolve(), args.oracle.resolve(), args.tokenizer_oracle.resolve())
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(f'{report["status"]}: {report["case_count"]} cases, {report["mismatch_count"]} mismatches; {output}')
        return int(report['status'] != 'pass')
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f'Hy3 template check failed: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
