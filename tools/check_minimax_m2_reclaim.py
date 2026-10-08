"""Compare a retained MM27-12 engine with whole-block reclaim on identical inputs.

This measures ordinary full-model workloads, with uncontrolled OS cache/external
load. The separate reclaim-check executable supplies deterministic pressure.
Never run another GPU test or a build concurrently with these timing runs.
"""
import argparse
import hashlib
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
from tools.tune_minimax_m2_pipeline import audit, save, sha, summarize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--baseline-dir', type=Path, required=True, help='retained engine.exe, backends/, manifest.json')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--engine', type=Path, default=ROOT / 'build-local/minimax-m2-cuda/bin/strata-minimax-m2-bench.exe')
    p.add_argument('--cuda-root', type=Path, default=ROOT / 'build-local/cuda-13.0')
    p.add_argument('--cache-mib', type=int, default=20480)
    p.add_argument('--chunk-mib', type=int, choices=(4, 8, 16), default=8)
    args = p.parse_args()
    if not 1 <= args.cache_mib <= 131072:
        p.error('invalid GPU cache cap')
    baseline = json.loads((args.baseline_dir / 'manifest.json').read_text())
    for rel, digest in baseline.items():
        if sha(args.baseline_dir / rel) != digest:
            raise ValueError('baseline hash mismatch: ' + rel)
    g = GGUFFile(args.model)
    validate_loader_contract(g.metadata, g.tensors)
    template = renderer(g.metadata['tokenizer.chat_template'])
    texts = ['Explain in a few sentences why the sky appears blue during the day.'] * 2
    texts += ['Прочитай список и объясни на русском, как найти его сумму: ' + ', '.join(map(str, range(1, 97)))]
    requests = [{'prompt': template.render(**text_context({'messages': [{'role': 'user', 'content': t}],
                 'add_generation_prompt': True})), 'max_tokens': 24} for t in texts]
    args.out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(args.baseline_dir, args.out / 'baseline')
    shutil.copyfile(args.engine, args.out / 'engine.exe')
    sources = args.out / 'sources'
    shutil.copytree(ROOT / 'backends/minimax_m2', sources / 'backends/minimax_m2')
    for rel in ('backends/step35/expert_cache.hpp', 'backends/hy3/gpu_arena.hpp',
                'tools/check_minimax_m2_reclaim.py', 'tools/tune_minimax_m2_pipeline.py'):
        dest = sources / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, dest)
    shutil.copytree(ROOT / 'backends/common', sources / 'backends/common')
    save(args.out / 'requests.json', requests)
    configs = [{'name': name, 'cache_mib': args.cache_mib, 'readers': 2, 'chunk_mib': args.chunk_mib}
               for name in ('legacy', 'whole-blocks')]
    with args.model.open('rb') as f:
        header_sha = hashlib.sha256(f.read(g.header_end)).hexdigest()
    report = {'pass': False, 'model': str(args.model), 'model_header_sha256': header_sha,
              'precision_environment': PRECISION_ENV, 'configurations': configs,
              'baseline_engine_sha256': sha(args.out / 'baseline/engine.exe'),
              'candidate_engine_sha256': sha(args.out / 'engine.exe'),
              'source_sha256': {s.relative_to(sources).as_posix(): sha(s) for s in sources.rglob('*') if s.is_file()},
              'attempts': [], 'runs': [], 'comparisons': [],
              'scope': '3 sequential A/B pairs, alternating order, fresh process per workload; sampled global memory; external load uncontrolled'}
    reference, reference_ids = None, None
    try:
        for repeat in range(3):
            for c in (configs if repeat % 2 == 0 else configs[::-1]):
                name = f"{repeat+1:02d}-{c['name']}"
                base = args.out / name
                exe = args.out / ('baseline/engine.exe' if c['name'] == 'legacy' else 'engine.exe')
                cmd = [str(exe.resolve()), '--gguf', str(args.model.resolve()), '--request', str(args.out / 'requests.json'),
                       '--output', str(base.with_suffix('.json')), '--logits', str(base.with_suffix('.f32')),
                       '--ctx', '2048', '--batch', '16', '--gpu-cache-allocator', 'arena',
                       '--gpu-cache-mib', str(args.cache_mib), '--pipeline-readers', '2',
                       '--pipeline-chunk-mib', str(args.chunk_mib), '--pipeline-lookahead', '1', '--pipeline-d2d-batch', '1']
                attempt = {'name': name, 'command': cmd}
                report['attempts'].append(attempt)
                save(args.out / 'suite-report.json', report)
                print(name + ': running', flush=True)
                with base.with_suffix('.stdout.log').open('w', encoding='utf-8') as out, base.with_suffix('.stderr.log').open('w', encoding='utf-8') as err:
                    run = subprocess.run(cmd, env=runtime_environment(args.cuda_root), stdout=out, stderr=err, timeout=1800)
                attempt['exit_code'] = run.returncode
                if run.returncode:
                    raise RuntimeError(f'{name}: exit {run.returncode}; retained inputs, logs and engine')
                data = json.loads(base.with_suffix('.json').read_text(encoding='utf-8'))
                gates = audit(data, c, 24)
                if c['name'] == 'whole-blocks':
                    gates['physical_budget'] = all(s['arena_reserved'] <= s['cache_limit']
                        for r in data['results'] for s in (r['prefill'], r['decode']))
                a = np.fromfile(base.with_suffix('.f32'), dtype='<f4')
                if a.size != sum(r['generated_tokens'] for r in data['results']) * 200064 or not np.isfinite(a).all():
                    raise ValueError('invalid logits: ' + name)
                ids = [r['token_ids'] for r in data['results']]
                if reference is None:
                    reference, reference_ids = base.with_suffix('.f32'), ids
                else:
                    b = np.fromfile(reference, dtype='<f4')
                    exact = a.shape == b.shape and a.tobytes() == b.tobytes()
                    report['comparisons'].append({'name': name, 'elements': int(a.size),
                        'bit_exact': exact, 'token_ids_equal': ids == reference_ids})
                    del b
                    if not exact or ids != reference_ids:
                        raise ValueError('numerical parity failed: ' + name)
                del a
                report['runs'].append({'name': name, 'repeat': repeat+1, 'configuration': c['name'],
                    'gates': gates, 'data': data, 'logits_sha256': sha(base.with_suffix('.f32'))})
                if not all(gates.values()):
                    raise ValueError(f'{name}: failed gates {gates}')
                report['summary'] = summarize(report['runs'], configs)
                save(args.out / 'suite-report.json', report)
                print(name + ': PASS; tok/s ' + ', '.join(f"{r['decode_tokens_per_second']:.3f}" for r in data['results']), flush=True)
        report['pass'] = True
    except Exception as exc:
        report['error'] = str(exc)
        raise
    finally:
        save(args.out / 'suite-report.json', report)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
