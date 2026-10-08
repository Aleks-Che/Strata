"""Check bounded, naturally completed MiniMax responses and retained prefix logits.

The three requests match the MM27-12/17 corpus. This is a correctness run,
not a speed A/B: OS file-cache state and external load are uncontrolled.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.gguf_reader import GGUFFile
from tools.minimax_m2_loader_contract import validate_loader_contract
from tools.minimax_m2_template import renderer, text_context
from tools.run_minimax_m2 import PRECISION_ENV, runtime_environment
from tools.tune_minimax_m2_pipeline import audit, save, sha

CONFIG = {'name': 'completion-18-2-4', 'cache_mib': 18432, 'readers': 2, 'chunk_mib': 4}


def rows(path, results):
    count = sum(r['generated_tokens'] for r in results)
    if path.stat().st_size != count * 200064 * 4:
        raise ValueError('wrong logit file size: ' + str(path))
    return np.memmap(path, dtype='<f4', mode='r', shape=(count, 200064))


def compare(a, b):
    if a.shape != b.shape:
        return {'pass': False, 'bit_exact': False, 'elements': int(a.size)}
    exact = finite = True
    max_abs = 0.
    for offset in range(0, len(a), 16):
        x, y = a[offset:offset+16], b[offset:offset+16]
        finite &= bool(np.isfinite(x).all() and np.isfinite(y).all())
        exact &= bool(np.array_equal(x.view('<u4'), y.view('<u4')))
        max_abs = max(max_abs, float(np.max(np.abs(x.astype(np.float64) - y))))
    return {'pass': finite and exact, 'finite': finite, 'bit_exact': exact,
            'elements': int(a.size), 'max_abs': max_abs}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--tokens', type=int, default=1536)
    p.add_argument('--reference-logits', type=Path, required=True)
    p.add_argument('--reference-report', type=Path, required=True)
    p.add_argument('--engine', type=Path, default=ROOT/'build-local/minimax-m2-cuda/bin/strata-minimax-m2-bench.exe')
    p.add_argument('--cuda-root', type=Path, default=ROOT/'build-local/cuda-13.0')
    args = p.parse_args()
    if not 257 <= args.tokens <= 1705:
        p.error('--tokens must be 257..1705 (343 prompt tokens + output <= context 2048)')
    g = GGUFFile(args.model)
    contract = validate_loader_contract(g.metadata, g.tensors)
    template = renderer(g.metadata['tokenizer.chat_template'])
    texts = ['Explain in a few sentences why the sky appears blue during the day.'] * 2
    texts += ['Прочитай список и объясни на русском, как найти его сумму: ' + ', '.join(map(str, range(1, 97)))]
    requests = [{'prompt': template.render(**text_context({'messages': [{'role': 'user', 'content': t}],
                'add_generation_prompt': True})), 'max_tokens': args.tokens} for t in texts]
    args.out.mkdir(parents=True, exist_ok=False)
    save(args.out/'requests.json', requests)
    exe = args.out/'engine.exe'
    shutil.copyfile(args.engine, exe)
    sources = args.out/'sources'
    shutil.copytree(ROOT/'backends/minimax_m2', sources)
    for rel in ['tools/check_minimax_m2_completion.py', 'tools/tune_minimax_m2_pipeline.py',
                'tools/minimax_m2_template.py', 'tools/run_minimax_m2.py']:
        shutil.copyfile(ROOT/rel, sources/Path(rel).name)
    with args.model.open('rb') as f:
        header_sha = __import__('hashlib').sha256(f.read(g.header_end)).hexdigest()
    env = runtime_environment(args.cuda_root)
    env['STRATA_MM27_TOKENWISE'] = '0'
    cmd = [str(exe.resolve()), '--gguf', str(args.model.resolve()), '--request', str(args.out/'requests.json'),
           '--output', str(args.out/'generation.json'), '--logits', str(args.out/'generation.f32'),
           '--ctx', '2048', '--batch', '16', '--gpu-cache-mib', '18432', '--gpu-cache-allocator', 'arena',
           '--pipeline-readers', '2', '--pipeline-chunk-mib', '4', '--pipeline-lookahead', '1', '--pipeline-d2d-batch', '1']
    report = {'pass': False, 'model': str(args.model), 'model_header_sha256': header_sha,
              'source_revision': contract['source_revision'], 'engine_sha256': sha(exe),
              'driver_sha256': sha(Path(__file__)), 'precision_environment': dict(PRECISION_ENV, STRATA_MM27_TOKENWISE='0'),
              'configuration': CONFIG, 'tokens_requested': args.tokens, 'command': cmd,
              'scope': 'one process; KV cleared between requests; cache retained; uncontrolled OS cache/external load; sampled memory',
              'reference': {'report': str(args.reference_report), 'report_sha256': sha(args.reference_report),
                            'logits': str(args.reference_logits), 'logits_sha256': sha(args.reference_logits)}}
    try:
        save(args.out/'completion-report.json', report)
        print('completion: running three requests', flush=True)
        with (args.out/'stdout.log').open('w', encoding='utf-8') as out, (args.out/'stderr.log').open('w', encoding='utf-8') as err:
            run = subprocess.run(cmd, env=env, stdout=out, stderr=err, timeout=2400)
        report['exit_code'] = run.returncode
        if run.returncode:
            raise RuntimeError(f'engine exit {run.returncode}; see retained stderr')
        data = json.loads((args.out/'generation.json').read_text(encoding='utf-8'))
        results = data['results']
        report['gates'] = audit(data, CONFIG, args.tokens)
        actual = rows(args.out/'generation.f32', results)
        reference = json.loads(args.reference_report.read_text(encoding='utf-8'))['results']
        old = rows(args.reference_logits, reference)
        report['prefix_comparisons'] = []
        report['completions'] = []
        offset = old_offset = 0
        for i, (r, prior) in enumerate(zip(results, reference)):
            n, m = r['generated_tokens'], prior['generated_tokens']
            c = compare(actual[offset:offset+min(n, m)], old[old_offset:old_offset+m])
            c['token_ids_equal'] = r['token_ids'][:m] == prior['token_ids']
            c['pass'] &= c['token_ids_equal'] and n >= m
            c['request'] = i
            report['prefix_comparisons'].append(c)
            finite = all(np.isfinite(actual[j:min(j+16, offset+n)]).all() for j in range(offset, offset+n, 16))
            text = r['text']
            closed = text.count('</think>') == 1
            final = text.split('</think>', 1)[1].removesuffix('[e~[').strip() if closed else ''
            eos = (r['stop_reason'] == 'eos' and r['stop_token_id'] == 200020 and
                   r['token_ids'][-1] == 200020 and 200020 not in r['token_ids'][:-1])
            report['completions'].append({'request': i, 'generated_tokens': n, 'stop_reason': r['stop_reason'],
                'natural_eos': eos, 'reasoning_closed': closed, 'final': final, 'finite_logits': bool(finite),
                'max_tokens': r['max_tokens'], 'within_context': r['prompt_tokens']+r['max_tokens'] <= data['context'],
                'decode_tokens_per_second': r['decode_tokens_per_second'], 'request_ms': r['request_ms'],
                'pass': eos and closed and bool(final) and bool(finite)})
            offset += n
            old_offset += m
        n, m = results[0]['generated_tokens'], results[1]['generated_tokens']
        report['repeat'] = compare(actual[:n], actual[n:n+m])
        report['repeat']['token_ids_equal'] = results[0]['token_ids'] == results[1]['token_ids']
        report['repeat']['pass'] &= report['repeat']['token_ids_equal']
        report['generation_report_sha256'] = sha(args.out/'generation.json')
        report['generation_logits_sha256'] = sha(args.out/'generation.f32')
        report['pass'] = (len(reference) == len(results) == 3 and all(report['gates'].values()) and
                          all(c['pass'] for c in report['prefix_comparisons']+report['completions']) and report['repeat']['pass'])
        print(json.dumps({'pass': report['pass'], 'completions': report['completions']}, ensure_ascii=False), flush=True)
    except Exception as exc:
        report['error'] = str(exc)
        raise
    finally:
        save(args.out/'completion-report.json', report)
    return 0 if report['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
