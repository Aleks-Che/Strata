"""Sequential offline DFlash/off comparison with full target-logit correctness gates.

Config array: name, depth (0..7), draft (required for depth>0), cache_mib.
The first run must be off unless --reference is supplied. Retain every failed
binary/logit/report; a divergent run is not eligible as a speedup result.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.gguf_reader import GGUFFile
from tools.inspect_minimax_m2_drafts import inspect
from tools.minimax_m2_template import renderer, text_context
from tools.run_minimax_m2 import PRECISION_ENV, runtime_environment


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def comparison(actual_dir, reference_dir):
    a = json.loads((actual_dir / 'report.json').read_text(encoding='utf-8'))
    b = json.loads((reference_dir / 'report.json').read_text(encoding='utf-8'))
    checks = []
    if len(a['results']) != len(b['results']):
        raise ValueError('different request counts')
    for i, (x, y) in enumerate(zip(a['results'], b['results'])):
        actual = np.fromfile(actual_dir / x['logits_file'], dtype='<f4')
        reference = np.fromfile(reference_dir / y['logits_file'], dtype='<f4')
        shape_ok = actual.size == len(x['ids']) * 200064 == reference.size
        ids_ok = x['ids'] == y['ids'] and x['stop_reason'] == y['stop_reason'] and x['prompt_tokens'] == y['prompt_tokens']
        item = {'request': i, 'shape_equal': shape_ok, 'token_ids_equal': ids_ok, 'elements': int(actual.size), 'pass': False}
        if shape_ok:
            delta = actual.astype(np.float64) - reference
            energy = np.square(reference.astype(np.float64)).sum()
            max_abs = float(np.abs(delta).max())
            nmse = float(np.square(delta).sum()/max(energy, 1e-30))
            finite = bool(np.isfinite(actual).all() and np.isfinite(reference).all())
            item.update(max_abs=max_abs, nmse=nmse, finite=finite, bit_exact=bool(np.array_equal(actual.view('<u4'), reference.view('<u4'))),
                        pass_=ids_ok and finite and max_abs <= 5e-4 and nmse <= 1e-7)
            item['pass'] = item.pop('pass_')
        checks.append(item)
    return checks


def summarize(runs, configs):
    summary = []
    for cfg in configs:
        selected = [r for r in runs if r['configuration'] == cfg['name'] and r.get('parity_pass', True)]
        if not selected:
            continue
        rows = [r for run in selected for r in run['data']['results']]
        phases = [s for r in rows for s in (r['prefill'], r['decode'])]
        total_proposed = sum(r['proposed'] for r in rows)
        total_accepted = sum(r['accepted'] for r in rows)
        summary.append({'configuration': cfg, 'runs': len(selected),
                        'median_aggregate_tok_s': statistics.median(sum(r['generated_tokens']-1 for r in run['data']['results'])*1000 /
                            sum(r['generation_ms'] for r in run['data']['results']) for run in selected),
                        'median_workload_seconds': statistics.median(sum(r['request_ms'] for r in run['data']['results'])/1000 for run in selected),
                        'request_tok_s': [statistics.median(run['data']['results'][i]['tokens_per_second'] for run in selected) for i in range(len(selected[0]['data']['results']))],
                        'accepted': total_accepted, 'proposed': total_proposed,
                        'acceptance_rate': total_accepted/total_proposed if total_proposed else None,
                        'emitted_per_cycle': sum(r['generated_tokens']-1 for r in rows)/sum(r['cycles'] for r in rows),
                        'vram_peak_percent': max(100*s['vram_used_peak']/s['vram_total'] for s in phases),
                        'ram_peak_percent': max(100*s['ram_used_peak']/s['ram_total'] for s in phases),
                        'draft_seconds': sum(r['draft_ms'] for r in rows)/1000,
                        'verify_seconds': sum(r['verify_ms'] for r in rows)/1000,
                        'catchup_seconds': sum(r['catchup_ms'] for r in rows)/1000,
                        'decode_h2d_gib_per_output': sum(r['decode']['h2d_bytes'] for r in rows)/sum(r['generated_tokens']-1 for r in rows)/2**30})
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--configs', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--tokens', type=int, choices=range(2, 257), default=32)
    p.add_argument('--repeats', type=int, choices=range(1, 6), default=1)
    p.add_argument('--reference', type=Path, help='Previous off run directory; same requests required')
    p.add_argument('--continue-on-parity-failure', action='store_true', help='Retain failed configurations and continue; exclude them from speed summaries, exit nonzero')
    p.add_argument('--engine', type=Path, default=ROOT / 'build-local/minimax-m2-cuda/bin/strata-minimax-m2-speculative-bench.exe')
    p.add_argument('--cuda-root', type=Path, default=ROOT / 'build-local/cuda-13.0')
    args = p.parse_args()
    configs = json.loads(args.configs.read_text(encoding='utf-8'))
    names = set()
    if not isinstance(configs, list) or not configs:
        p.error('expected nonempty configuration array')
    for c in configs:
        if set(c)-{'name', 'depth', 'draft', 'cache_mib', 'verify_tokenwise'} or not {'name', 'depth', 'cache_mib'} <= set(c):
            p.error('invalid configuration keys')
        if not isinstance(c['name'], str) or not c['name'] or len(c['name'])>64 or any(ch not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for ch in c['name']) or c['name'] in names:
            p.error('distinct lowercase configuration names required')
        if type(c['depth']) is not int or not 0 <= c['depth'] <= 7 or type(c['cache_mib']) is not int or not 0 <= c['cache_mib'] <= 24576 or bool(c['depth']) != bool(c.get('draft')):
            p.error('invalid depth/draft/cache configuration')
        names.add(c['name'])
        if type(c.get('verify_tokenwise', 0)) is not int or not 0 <= c.get('verify_tokenwise', 0) <= 3:
            p.error('invalid verification precision mode')
    if not args.reference and configs[0]['depth']:
        p.error('first configuration must be off')
    drafts = sorted({str(Path(c['draft']).resolve()) for c in configs if c['depth']})
    inventory = inspect(args.model, drafts, full_hash=True)
    template = renderer(GGUFFile(args.model).metadata['tokenizer.chat_template'])
    prompts = [('explanation', 'Explain in three sentences why the sky is blue.'),
               ('python', 'Write a Python function that returns the sum of even numbers in a list. Include a short example.'),
               ('russian', 'Кратко объясни, чем оперативная память отличается от видеопамяти.')]
    requests = [{'name': n, 'prompt': template.render(**text_context({'messages': [{'role': 'user', 'content': t}], 'add_generation_prompt': True})),
                 'max_tokens': args.tokens} for n, t in prompts]
    if args.reference:
        previous = json.loads((args.reference.resolve().parent / 'requests.json').read_text(encoding='utf-8'))
        if requests != previous:
            p.error('reference requests differ')
    args.out = args.out.resolve();args.out.mkdir(parents=True, exist_ok=False)
    save(args.out/'requests.json', requests);save(args.out/'configs.json', configs);save(args.out/'inspection.json', inventory)
    shutil.copyfile(args.engine, args.out/'engine.exe')
    shutil.copytree(ROOT/'backends/minimax_m2', args.out/'sources/backends/minimax_m2')
    shutil.copytree(ROOT/'backends/common', args.out/'sources/backends/common')
    for rel in ('backends/step35/expert_cache.hpp', 'backends/hy3/gpu_arena.hpp', 'tools/bench_minimax_m2_dflash.py'):
        dst = args.out/'sources'/rel;dst.parent.mkdir(parents=True, exist_ok=True);shutil.copyfile(ROOT/rel, dst)
    shutil.copyfile(args.engine.parent.parent/'minimax-m2-build-manifest.json', args.out/'build-manifest.json')
    report = {'pass': False, 'scope': 'offline greedy comparison, serial GPU runs; OS file cache and external load uncontrolled',
              'tokens_requested': args.tokens, 'repeats': args.repeats, 'precision_environment': PRECISION_ENV,
              'engine_sha256': sha(args.out/'engine.exe'), 'requests_sha256': sha(args.out/'requests.json'),
              'driver_sha256': sha(Path(__file__)), 'target_header_sha256': inventory['target_header_sha256'],
              'source_sha256': {x.relative_to(args.out/'sources').as_posix(): sha(x) for x in (args.out/'sources').rglob('*') if x.is_file()},
              'attempts': [], 'runs': [], 'comparisons': [], 'configurations': configs}
    reference = args.reference.resolve() if args.reference else None
    try:
        for repeat in range(args.repeats):
            for c in (configs if repeat % 2 == 0 else configs[::-1]):
                name = f"{repeat+1:02d}-{c['name']}";dest = args.out/name
                cmd = [str(args.out/'engine.exe'), '--model', str(args.model.resolve()), '--requests', str(args.out/'requests.json'),
                       '--out', str(dest), '--depth', str(c['depth']), '--cache-mib', str(c['cache_mib'])]
                cmd += ['--verify-tokenwise', str(c.get('verify_tokenwise', 0))]
                if c['depth']:
                    cmd += ['--draft', str(Path(c['draft']).resolve())]
                attempt = {'name': name, 'command': cmd};report['attempts'].append(attempt);save(args.out/'suite-report.json', report)
                print(name+': running', flush=True)
                with (args.out/f'{name}.stdout.log').open('w', encoding='utf-8') as out, (args.out/f'{name}.stderr.log').open('w', encoding='utf-8') as err:
                    run = subprocess.run(cmd, env=runtime_environment(args.cuda_root), stdout=out, stderr=err, timeout=1800)
                attempt['exit_code'] = run.returncode
                if run.returncode:
                    raise RuntimeError(f'{name}: exit {run.returncode}; saved executable/source/logs')
                data = json.loads((dest/'report.json').read_text(encoding='utf-8'))
                if not data['pass']:
                    raise ValueError('engine report failed')
                if reference is None:
                    reference = dest
                checks = comparison(dest, reference)
                report['comparisons'].append({'name': name, 'reference': str(reference), 'checks': checks})
                parity = all(x['pass'] for x in checks)
                report['runs'].append({'name': name, 'configuration': c['name'], 'repeat': repeat+1, 'data': data, 'parity_pass': parity,
                                       'logits_sha256': {x.name: sha(x) for x in dest.glob('*.f32')}})
                if not parity and not args.continue_on_parity_failure:
                    raise ValueError(f'{name}: full-logit or greedy parity failed; do not loosen tolerances')
                report['summary'] = summarize(report['runs'], configs);save(args.out/'suite-report.json', report)
                print(name+(': PASS; tok/s ' if parity else ': PARITY FAIL (timings ineligible); raw tok/s ')+', '.join(f"{r['tokens_per_second']:.3f}" for r in data['results'])+
                      '; acceptance '+str(sum(r['accepted'] for r in data['results']))+'/'+str(sum(r['proposed'] for r in data['results'])), flush=True)
        report['pass'] = all(r['parity_pass'] for r in report['runs'])
    except Exception as exc:
        report['error'] = str(exc)
        raise
    finally:
        save(args.out/'suite-report.json', report)
    return 0 if report['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
