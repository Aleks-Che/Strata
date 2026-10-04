"""Compare native MTP depths in separate, warmed pipe engines on this PC.

Uses the exact saved prompt/continuation. Never changes the working profile.
The first request warms routing/cache; the following requests are timed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve.server import StrataEngine, child_env, engine_args


def replace(args, flag, value):
    args = list(args)
    if flag in args:
        args[args.index(flag) + 1] = str(value)
    else:
        args += [flag, str(value)]
    return args


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--profile', type=Path, required=True)
    ap.add_argument('--reference', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--depths', default='0,1,2,3')
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--mtp-cache-mib', type=int, default=512)
    ap.add_argument('--chunk-mib', type=int, default=4)
    a = ap.parse_args()
    depths = [int(x) for x in a.depths.split(',')]
    if a.repeats < 2 or not depths or any(x not in range(4) for x in depths):
        ap.error('at least two measured repeats and depths 0..3 required')
    cfg = json.loads(a.profile.read_text(encoding='utf-8'))
    reference = json.loads(a.reference.read_text(encoding='utf-8'))
    ids, expected = reference['prompt_ids'], reference['generated_ids']
    a.output.parent.mkdir(parents=True, exist_ok=True)
    report = {'status': 'running', 'configuration': cfg, 'reference': str(a.reference.resolve()),
              'binary_sha256': hashlib.sha256(Path(cfg['exe']).read_bytes()).hexdigest(),
              'timing': '(output tokens - 1) * 1000 / DONE.decode_ms; excludes load/warmup/prefill, includes target sampling, draft, verify, repair and pipe writes',
              'scope': 'one saved prompt, sequential engines, OS file cache not cleared', 'variants': []}
    def save():
        a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    save()
    for depth in depths:
        variant = dict(cfg)
        for flag, value in (('--expert-pipeline', 1), ('--expert-chunk-mib', a.chunk_mib),
                            ('--mtp', depth), ('--mtp-cache-mib', a.mtp_cache_mib)):
            variant['args'] = replace(variant['args'], flag, value)
        variant['log'] = str(a.output.with_name(a.output.stem + f'-mtp{depth}.engine.log').resolve())
        result = {'depth': depth, 'configuration': variant, 'runs': [], 'status': 'running'}
        report['variants'].append(result)
        engine = None
        try:
            start = time.monotonic()
            engine = StrataEngine(variant['exe'], engine_args(variant), cwd=variant.get('cwd'),
                                  env=child_env(variant), log=variant['log'])
            result['startup_seconds'] = time.monotonic() - start
            result['info'] = engine.info
            assert engine.info['gpu_only'] == 1 and engine.info['expert_pipeline'] == 1
            assert engine.info['spec'] == depth
            for i in range(a.repeats + 1):
                tokens = [t for t in engine.generate(ids, len(expected), {'temperature': 0}, threading.Event()) if t is not None]
                done = dict(engine.last)
                assert done['finish'] == 'length' and len(tokens) == len(expected), done
                run = {'warmup': i == 0, 'tokens': tokens, 'equals_reference': tokens == expected, 'done': done,
                       'tokens_per_second': (len(tokens) - 1) * 1000 / done['decode_ms']}
                result['runs'].append(run)
                print(f"MTP {depth}, run {i}: {run['tokens_per_second']:.3f} tok/s, reference={run['equals_reference']}", flush=True)
                save()
            result['median_tokens_per_second'] = statistics.median(r['tokens_per_second'] for r in result['runs'] if not r['warmup'])
            result['status'] = 'pass' if all(r['equals_reference'] for r in result['runs']) else 'token_mismatch'
        except Exception as exc:
            result['status'], result['error'] = 'error', str(exc)
        finally:
            if engine:
                proc = engine.proc
                engine.close()
                result['exit_code'] = proc.returncode
                if proc.returncode != 0:
                    result['status'] = 'error'
                text = Path(variant['log']).read_text(encoding='utf-8', errors='replace')
                for marker, key in (('STRATA_GLM_MEMORY ', 'memory_snapshots'), ('STRATA_GLM_MTP ', 'mtp_statistics')):
                    result[key] = [json.loads(line.split(marker, 1)[1]) for line in text.splitlines() if marker in line]
            save()
    valid = [v for v in report['variants'] if v['status'] == 'pass']
    report['fastest_matching_depth'] = max(valid, key=lambda v: v['median_tokens_per_second'])['depth'] if valid else None
    report['status'] = 'pass' if len(valid) == len(depths) else 'incomplete_or_mismatch'
    save()
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
