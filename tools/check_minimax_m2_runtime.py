"""Compare bounded MiniMax file delivery against the pinned native mmap path.

Runs real weights sequentially, one fresh process per case/mode. This is a
correctness corpus, not a cold/warm performance A/B. ReadFile byte counts include
Windows file-cache hits and must not be reported as physical SSD bytes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.gguf_reader import GGUFFile
from tools.minimax_m2_loader_contract import validate_loader_contract
from tools.minimax_m2_template import renderer, text_context
from tools.run_minimax_m2 import runtime_environment

CASES = {
    'english': 'What is 2 + 2?',
    'russian': 'Назови столицу Франции.',
    'chinese': '一加一等于几？',
    'code': 'Complete this Python function: def square(x):\n    return',
    'long': 'Read this list and give its last number: ' + ', '.join(str(i) for i in range(1, 49)),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_pipe(args, env, template):
    prompt = template.render(**text_context({'messages': [{'role': 'user', 'content': CASES['english']}],
                                             'add_generation_prompt': True}))
    valid = {'prompt': prompt, 'max_tokens': 2}
    requests = [valid, {'tokens': [200064], 'max_tokens': 2}, valid]
    payload = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in requests)
    (args.out / 'pipe-input.jsonl').write_text(payload, encoding='utf-8')
    stdout = args.out / 'pipe-output.jsonl'
    with stdout.open('w', encoding='utf-8') as out, (args.out / 'pipe-stderr.log').open('w', encoding='utf-8') as err:
        run = subprocess.run([str(args.engine.resolve()), '--gguf', str(args.model.resolve()), '--pipe'],
                             input=payload, encoding='utf-8', env=env, stdout=out, stderr=err, timeout=600)
    events = [json.loads(line) for line in stdout.read_text(encoding='utf-8').splitlines()]
    results = [e for e in events if e.get('event') == 'result']
    order = [e.get('event') for e in events]
    passed = run.returncode == 0 and order == ['ready', 'token', 'token', 'result', 'error', 'token', 'token', 'result']
    passed = passed and results[0]['token_ids'] == results[1]['token_ids']
    passed = passed and all(s['rejected_cpu_nodes'] == s['rejected_full_copies'] == 0 and s['h2d_bytes'] == s['source_bytes']
                            for r in results for s in (r['prefill'], r['decode']))
    return {'pass': passed, 'exit_code': run.returncode, 'event_order': order, 'results': results,
            'output_sha256': sha(stdout), 'scope': 'valid -> invalid token -> fresh identical request; one loaded process'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True, help='new result directory')
    p.add_argument('--engine', type=Path, default=ROOT / 'build-local/minimax-m2-cuda/bin/strata-minimax-m2-bench.exe')
    p.add_argument('--cuda-root', type=Path, default=ROOT / 'build-local/cuda-13.0')
    p.add_argument('--cases', nargs='+', choices=list(CASES), default=list(CASES))
    p.add_argument('--tokens', type=int, default=8)
    p.add_argument('--pipe-only', action='store_true', help='check error recovery/repeated requests instead of the logit corpus')
    args = p.parse_args()
    if not 1 <= args.tokens <= 256:
        p.error('--tokens must be 1..256')
    args.out.mkdir(parents=True, exist_ok=False)
    g = GGUFFile(args.model)
    contract = validate_loader_contract(g.metadata, g.tensors)
    template = renderer(g.metadata['tokenizer.chat_template'])
    env = runtime_environment(args.cuda_root)
    with args.model.open('rb') as stream:
        header_sha = hashlib.sha256(stream.read(g.header_end)).hexdigest()
    report = {'model': str(args.model), 'header_sha256': header_sha, 'engine_sha256': sha(args.engine),
              'source_revision': contract['source_revision'], 'cases': [],
              'scope': 'strict F32, FA off; sequential correctness comparisons, not controlled speed A/B'}
    if args.pipe_only:
        report['pipe'] = check_pipe(args, env, template)
        report['pass'] = report['pipe']['pass']
        (args.out / 'full-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        return 0 if report['pass'] else 1
    try:
        for name in args.cases:
            prompt = template.render(**text_context({'messages': [{'role': 'user', 'content': CASES[name]}],
                                                     'add_generation_prompt': True}))
            request = args.out / f'{name}-request.json'
            request.write_text(json.dumps({'prompt': prompt, 'max_tokens': args.tokens}, ensure_ascii=False), encoding='utf-8')
            results, values, hashes = [], [], []
            for mode in (2, 1):
                base = args.out / f'{name}-mode{mode}'
                output, logits = base.with_suffix('.json'), base.with_suffix('.f32')
                command = [str(args.engine.resolve()), '--gguf', str(args.model.resolve()), '--request', str(request),
                           '--output', str(output), '--logits', str(logits), '--mode', str(mode)]
                print(f'{name}: mode {mode}', flush=True)
                with base.with_suffix('.stdout.log').open('w', encoding='utf-8') as stdout, base.with_suffix('.stderr.log').open('w', encoding='utf-8') as stderr:
                    subprocess.run(command, env=env, stdout=stdout, stderr=stderr, check=True, timeout=600)
                result = json.loads(output.read_text(encoding='utf-8'))
                results.append(result)
                data = np.fromfile(logits, dtype='<f4')
                expected = result['result']['generated_tokens'] * 200064
                if data.size != expected or not np.isfinite(data).all():
                    raise ValueError(f'{name}: invalid logit file')
                values.append(data)
                hashes.append({'report_sha256': sha(output), 'logits_sha256': sha(logits)})
            a, b = values
            same_shape = a.shape == b.shape
            delta = a.astype(np.float64) - b if same_shape else None
            bits = same_shape and a.tobytes() == b.tobytes()
            same_tokens = results[0]['result']['token_ids'] == results[1]['result']['token_ids']
            audit = all(s['gpu_nodes'] and not s['rejected_cpu_nodes'] and not s['rejected_full_copies']
                        for r in results for s in (r['result']['prefill'], r['result']['decode'])
                        if s['compute_calls'])
            case = {'name': name, 'pass': bits and same_tokens and audit, 'elements': int(a.size),
                    'bit_exact': bits, 'token_ids_equal': same_tokens, 'gpu_only': audit,
                    'max_abs': float(abs(delta).max()) if same_shape else None,
                    'nmse': float(np.dot(delta, delta) / max(np.dot(b.astype(np.float64), b), 1e-30)) if same_shape else None,
                    'runs': results, 'hashes': hashes}
            report['cases'].append(case)
            (args.out / 'full-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f'{name}: {"PASS" if case["pass"] else "FAIL"}, bit_exact={bits}', flush=True)
        report['pass'] = len(report['cases']) == len(args.cases) and all(c['pass'] for c in report['cases'])
    except Exception as exc:
        report['pass'] = False
        report['error'] = str(exc)
        raise
    finally:
        (args.out / 'full-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return 0 if report['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
