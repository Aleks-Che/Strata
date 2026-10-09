"""Opt-in full-model resident-prefix audit. Run alone; retains full F32 logits.

The old executable is a same-model regression reference, not an independent
model oracle. No admission SHA is changed by this tool.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.check_minimax_m2_completion_pipe import result_checks
from tools.run_minimax_m2 import runtime_environment
from tools.strata_tokenizer import Tokenizer
from serve.winjob import contain


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def compare_logits(left, right):
    a, b = np.fromfile(left, dtype='<f4'), np.fromfile(right, dtype='<f4')
    if a.shape != b.shape:
        return {'pass': False, 'left_count': a.size, 'right_count': b.size}
    bits = a.view('<u4') != b.view('<u4')
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    return {'pass': finite and not bool(bits.any()), 'floats': a.size,
            'different_bits': int(bits.sum()), 'finite': finite,
            'max_abs': float(np.max(np.abs(a-b))) if a.size and finite else None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gguf', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--engine', type=Path, default=ROOT/'build-local/minimax-m2-cuda/bin/strata-minimax-m2-bench.exe')
    p.add_argument('--reference-engine', type=Path, required=True)
    p.add_argument('--cuda-root', type=Path, default=ROOT/'build-local/cuda-13.0')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    report = {'pass': False, 'scope': 'single resident prefix; exact tokens/full logits against fresh and old fresh',
              'model': str(args.gguf.resolve()), 'engine_sha256': sha(args.engine),
              'reference_engine_sha256': sha(args.reference_engine), 'runs': {}, 'sources': {}}
    for part in ['backends/minimax_m2', 'serve', 'tools']:
        files = (ROOT/part).glob('*') if part != 'backends/minimax_m2' else (ROOT/part).rglob('*')
        for file in files:
            if file.is_file() and file.suffix in ('.py', '.cpp', '.hpp', '.h', '.inc', '.cmake', '.txt', '.in'):
                rel = file.relative_to(ROOT)
                if part != 'backends/minimax_m2' and 'minimax' not in file.name and file.name not in ('server.py', 'winjob.py', 'strata_tokenizer.py'):
                    continue
                dest = args.out/'sources'/rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(file, dest)
                report['sources'][rel.as_posix()] = sha(file)
    shutil.copyfile(args.engine, args.out/'engine.exe')
    shutil.copyfile(args.reference_engine, args.out/'reference.exe')
    report['gpu'] = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                                           '--format=csv,noheader'], text=True).strip()
    env = runtime_environment(args.cuda_root)
    env['STRATA_MM27_TOKENWISE'] = '0'
    report['environment'] = {k: v for k, v in env.items() if k.startswith(('STRATA_', 'GGML_', 'LLAMA_', 'NVIDIA_'))}
    try:
        tok = Tokenizer.from_gguf(args.gguf)
        old = json.loads((ROOT/'build-local/minimax-m2-completion-01/requests.json').read_text(encoding='utf-8'))[0]
        reference = json.loads((ROOT/'build-local/minimax-m2-completion-01/generation.json').read_text(encoding='utf-8'))['results'][0]
        base = tok.encode(old['prompt'], parse_special=True)
        assert len(base) == 52
        branch = base.copy(); branch[35] = tok.encode('red')[0]
        long = (base+tok.encode(' The blue sky reflects scattered light.' * 150))[:512]
        long_branch = long.copy(); long_branch[497] = tok.encode('red')[0]
        cases = [('initial', base, 'a', 0), ('repeat', base, 'a', 48),
                 ('generated-tail', base+reference['token_ids'][:8], 'a', 48),
                 ('shorter', base[:33], 'a', 32), ('extend-after-shorter', base, 'a', 32),
                 ('branch', branch, 'a', 32), ('other-session', branch, 'b', 0),
                 ('return-evicted-session', branch, 'a', 0), ('anonymous', base, None, 0),
                 ('after-anonymous', base, 'a', 0), ('one-full-batch', base[:16], 'a', 0),
                 ('extend-full-batch', base, 'a', 16), ('long-initial', long, 'a', 48),
                 ('long-repeat', long, 'a', 496), ('long-extend', long+[base[-1]], 'a', 512),
                 ('long-branch', long_branch, 'a', 496)]
        report['cases'] = [{'name': name, 'prompt_tokens': len(ids), 'expected_reused': reuse} for name, ids, _, reuse in cases]
        fresh = [{'tokens': ids, 'max_tokens': 8} for _, ids, _, _ in cases]
        reused = [{**item, **({'session_key': hashlib.sha256(key.encode()).hexdigest()} if key else {})}
                  for item, (_, _, key, _) in zip(fresh, cases)]
        for name, binary, requests, extra in [('old-fresh', args.reference_engine, fresh[:2], []),
                                              ('fresh', args.engine, fresh, []),
                                              ('prefix', args.engine, reused, ['--prefix-cache', '1'])]:
            save(args.out/(name+'.requests.json'), requests)
            command = [str(binary.resolve()), '--gguf', str(args.gguf.resolve()), '--ctx', '2048', '--batch', '16',
                       '--mode', '2', '--gpu-cache-mib', '18432', '--gpu-cache-allocator', 'arena',
                       '--pipeline-readers', '2', '--pipeline-chunk-mib', '4', '--pipeline-lookahead', '1',
                       '--pipeline-d2d-batch', '1', '--request', str((args.out/(name+'.requests.json')).resolve()),
                       '--output', str((args.out/(name+'.json')).resolve()),
                       '--logits', str((args.out/(name+'.f32')).resolve()), *extra]
            print('START '+name, flush=True)
            with (args.out/(name+'.stdout.log')).open('wb') as out, (args.out/(name+'.stderr.log')).open('wb') as err:
                proc = subprocess.Popen(command, stdout=out, stderr=err, env=env,
                                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                contain(proc)
                try:
                    rc = proc.wait(timeout=1800)
                finally:
                    if proc.poll() is None:
                        proc.kill(); proc.wait(timeout=15)
                assert rc == 0, (name, rc)
            doc = json.loads((args.out/(name+'.json')).read_text(encoding='utf-8'))
            checks = [result_checks(result, doc) for result in doc['results']]
            report['runs'][name] = {'command': command, 'checks': checks, 'pid': proc.pid,
                                   'returncode': rc, 'pass': all(all(c.values()) for c in checks)}
            assert report['runs'][name]['pass'], checks
            save(args.out/'report.json', report)
            print('DONE '+name, flush=True)
        a = json.loads((args.out/'fresh.json').read_text(encoding='utf-8'))['results']
        b = json.loads((args.out/'prefix.json').read_text(encoding='utf-8'))['results']
        old_results = json.loads((args.out/'old-fresh.json').read_text(encoding='utf-8'))['results']
        report['old_tokens_equal'] = all(x['token_ids'] == y['token_ids'] for x, y in zip(old_results, a[:2]))
        prefix_bytes = (args.out/'old-fresh.f32').stat().st_size
        with (args.out/'fresh.f32').open('rb') as f:
            (args.out/'fresh-regression.f32').write_bytes(f.read(prefix_bytes))
        report['old_logits'] = compare_logits(args.out/'old-fresh.f32', args.out/'fresh-regression.f32')
        report['reuse_logits'] = compare_logits(args.out/'fresh.f32', args.out/'prefix.f32')
        for case, fresh_result, reused_result in zip(report['cases'], a, b):
            case.update(tokens_equal=fresh_result['token_ids'] == reused_result['token_ids'],
                        reused=reused_result['reused_tokens'], fresh_prefill_ms=fresh_result['prefill_ms'],
                        reused_prefill_ms=reused_result['prefill_ms'], fresh_request_ms=fresh_result['request_ms'],
                        reused_request_ms=reused_result['request_ms'])
            case['pass'] = (case['tokens_equal'] and case['reused'] == case['expected_reused'] and
                            reused_result['evaluated_prompt_tokens']+case['reused'] == case['prompt_tokens'] and
                            reused_result['kv_tokens'] == case['prompt_tokens']+reused_result['generated_tokens']-1 and
                            fresh_result['reused_tokens'] == 0)
        report['pass'] = (report['old_tokens_equal'] and report['old_logits']['pass'] and report['reuse_logits']['pass'] and
                          all(c['pass'] for c in report['cases']))
        assert report['pass'], 'prefix parity failed; see report and full logits'
    except BaseException:
        report['error'] = traceback.format_exc()
        raise
    finally:
        report['artifacts'] = {f.name: sha(f) for f in args.out.iterdir() if f.is_file() and f.name != 'report.json'}
        save(args.out/'report.json', report)


if __name__ == '__main__':
    main()
