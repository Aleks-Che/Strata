"""Compare CPU page placement with MTP and learned expert hints disabled.

Fresh engines, fixed saved A/B token references, separate startup/TTFT/wall/decode
timings. Never clears the OS file cache or edits the working launch profile.
"""
import argparse
from copy import deepcopy
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--profile', type=Path, required=True)
    ap.add_argument('--reference', type=Path, required=True)
    ap.add_argument('--second-reference', type=Path, required=True, help='Previous warm benchmark containing B-off')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--modes', default='0,1')
    ap.add_argument('--warmups', type=int, default=4)
    ap.add_argument('--repeats', type=int, default=5)
    ap.add_argument('--second-repeats', type=int, default=3)
    ap.add_argument('--cold-repeats', type=int, default=1)
    ap.add_argument('--diagnostics', type=int, choices=(0, 1), default=0)
    a = ap.parse_args()
    modes = [int(x) for x in a.modes.split(',')]
    if a.output.exists() or min(a.warmups, a.repeats, a.cold_repeats, a.second_repeats) < 1 or not modes or any(x not in (0, 1) for x in modes):
        ap.error('use an unused output path, positive counts and modes 0/1')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(a.profile.read_text(encoding='utf-8'))
    ref_a = json.loads(a.reference.read_text(encoding='utf-8'))
    ref_b = next(c for c in json.loads(a.second_reference.read_text(encoding='utf-8'))['cases'] if c['name'] == 'B-off')
    refs = {'A': (ref_a['prompt_ids'], ref_a['generated_ids']), 'B': (ref_b['prompt_ids'], ref_b['runs'][0]['tokens'])}
    assert cfg['args'][cfg['args'].index('--mtp') + 1] == '0'
    report = {'status': 'running', 'configuration': cfg,
              'binary_sha256': hashlib.sha256(Path(cfg['exe']).read_bytes()).hexdigest(),
              'method': {'modes': modes, 'warmups': a.warmups, 'repeats': a.repeats,
                         'second_repeats': a.second_repeats, 'cold_repeats': a.cold_repeats,
                         'diagnostics': a.diagnostics, 'os_file_cache_cleared': False,
                         'order': 'First engines run A warmups/timed repeats then B; later cold rounds run A once, reversing mode order each round.'},
              'cases': []}

    def save():
        a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    for round_id in range(a.cold_repeats):
        for mode in modes if round_id % 2 == 0 else list(reversed(modes)):
            name = f'r{round_id}-mode{mode}'
            variant = deepcopy(cfg)
            variant.setdefault('env', {}).update(STRATA_GLM_RAM_WARM_MODE=str(mode),
                STRATA_GLM_RAM_DIAGNOSTICS=str(a.diagnostics), STRATA_GLM_EXPERT_PROFILE='', STRATA_GLM_EXPERT_PROFILE_READ_ONLY='1')
            variant['log'] = str(a.output.with_name(a.output.stem + '-' + name + '.engine.log').resolve())
            assert not Path(variant['log']).exists(), 'use a fresh log basename'
            case = {'name': name, 'mode': mode, 'round': round_id, 'configuration': variant, 'runs': [], 'status': 'running'}
            report['cases'].append(case); save()
            engine = None
            print('START ' + name, flush=True)
            try:
                start = time.perf_counter()
                engine = StrataEngine(variant['exe'], engine_args(variant), cwd=variant.get('cwd'), env=child_env(variant), log=variant['log'])
                case['startup_seconds'] = time.perf_counter() - start
                case['info'] = engine.info
                assert engine.info['spec'] == 0 and engine.info['expert_warm_entries'] == 0 and engine.info['expert_ram_warm_mode'] == mode
                schedule = [('A', a.warmups + a.repeats), ('B', a.second_repeats)] if round_id == 0 else [('A', 1)]
                for prompt, count in schedule:
                    ids, expected = refs[prompt]
                    for i in range(count):
                        begin = time.perf_counter(); first = None; tokens = []
                        for token in engine.generate(ids, 64, {'temperature': 0}, threading.Event()):
                            if token is not None:
                                if first is None:
                                    first = time.perf_counter() - begin
                                    if i == 0: print(f'{name} {prompt}: first token at {first:.3f}s', flush=True)
                                tokens.append(token)
                        wall = time.perf_counter() - begin; done = dict(engine.last)
                        assert len(tokens) == 64 and done['finish'] == 'length' and tokens == expected, (name, prompt, i, done)
                        run = {'prompt': prompt, 'index': i, 'prompt_ids': ids, 'tokens': tokens,
                               'equals_saved_reference': True, 'done': done, 'first_token_seconds': first,
                               'request_wall_seconds': wall, 'tokens_per_second': 63000 / done['decode_ms']}
                        case['runs'].append(run); save()
                        print(f'{name} {prompt}#{i}: wall={wall:.3f}s decode={run["tokens_per_second"]:.3f} tok/s', flush=True)
                if round_id == 0:
                    timed = [r for r in case['runs'] if r['prompt'] == 'A'][a.warmups:]
                    rates = [r['tokens_per_second'] for r in timed]
                    case['warm_summary'] = {'median_tokens_per_second': statistics.median(rates),
                        'range': [min(rates), max(rates)], 'median_request_wall_seconds': statistics.median(r['request_wall_seconds'] for r in timed)}
                case['status'] = 'pass'
            except Exception as e:
                report['status'] = 'error'; case['status'] = 'error'; case['error'] = repr(e); raise
            finally:
                if engine:
                    process = engine.proc; engine.close(); case['exit_code'] = process.returncode
                    assert process.returncode == 0
                    lines = Path(variant['log']).read_text(encoding='utf-8', errors='replace').splitlines()
                    case['memory_snapshots'] = [json.loads(line.split('STRATA_GLM_MEMORY ', 1)[1]) for line in lines if 'STRATA_GLM_MEMORY ' in line]
                    assert len(case['memory_snapshots']) == len(case['runs']) + 1
                    if mode: assert case['memory_snapshots'][0]['ram_warm_skipped_gpu_bytes'] > 0
                save()
    report['status'] = 'pass'; save()


if __name__ == '__main__':
    main()
