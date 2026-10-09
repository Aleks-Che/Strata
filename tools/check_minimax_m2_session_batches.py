"""MM27-31: full-model prefix/archive parity against fresh at batch 1 or 8.

Windows diagnostic, one GPU process at a time. The boundary corpus uses
context1024 and prompts of 1/8..513 tokens, not a 2K/4K or speed benchmark.
All logits, source snapshots and independent memory samples are retained.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve.minimax_m2_engine import EXE_SHA256
from serve.winjob import contain
from tools.check_minimax_m2_prefix import save
from tools.check_minimax_m2_prefix_context import snapshot, NV
from tools.check_minimax_m2_sessions import compare_files
from tools.check_minimax_m2_sessions_context import inspect_case, sha
from tools.minimax_m2_memory_observer import MemoryObserver, memory_within_limit
from tools.run_minimax_m2 import runtime_environment
from tools.strata_tokenizer import Tokenizer


def make_corpus(base, long, replacement, batch):
    if batch not in (1, 8) or len(base) != 52 or len(long) != 513 or long[:52] != base:
        raise ValueError('expected batch1/8, base52 and its extension513')
    if replacement in (base[35], long[497]):
        raise ValueError('branch must change a token')
    branch = base.copy(); branch[35] = replacement
    long_branch = long[:512]; long_branch[497] = replacement
    # Explicit expected boundaries; this is not a call to the native planner.
    repeat, branch_reuse, extend, long_repeat, long_branch_reuse = {
        1: (51, 35, 33, 511, 497), 8: (48, 32, 32, 504, 496),
    }[batch]
    specs = [
        ('cold', base, 'a', 0, False, 8),
        ('resident', base, 'a', repeat, False, 8),
        ('branch', branch, 'a', branch_reuse, False, 8),
        ('shorter', base[:33], 'a', 32, False, 8),
        ('extend-shorter', base, 'a', extend, False, 8),
        ('one-full-batch', base[:batch], 'a', 0, False, 8),
        ('extend-full-batch', base, 'a', batch, False, 8),
        ('other-session', base, 'b', 0, False, 8),
        ('restored', base, 'a', repeat, True, 8),
        ('anonymous', base, None, 0, False, 8),
        ('after-anonymous', base, 'a', repeat, True, 8),
        ('long-cold', long[:512], 'long', 0, False, 8),
        ('short-restored', base, 'a', repeat, True, 8),
        ('long-restored', long[:512], 'long', long_repeat, True, 8),
        ('long-resident', long[:512], 'long', long_repeat, False, 8),
        ('long-extend', long, 'long', 512, False, 8),
        ('long-branch', long_branch, 'long', long_branch_reuse, False, 8),
        ('long-shorter', base, 'long', repeat, False, 8),
        ('one-cold', base, 'one', 0, False, 1),
        ('long-one-restored', base, 'long', repeat, True, 1),
        ('one-restored', base, 'one', repeat, True, 1),
        ('one-resident', base, 'one', repeat, False, 1),
        ('sample-resident', base, 'one', repeat, False, 64),
        ('short-before-sample-restore', base, 'a', repeat, True, 8),
        ('sample-restored', base, 'one', repeat, True, 64),
    ]
    fresh, requests, cases, indexes = [], [], [], {}
    for name, ids, key, reuse, restored, count in specs:
        req = {'tokens': ids, 'max_tokens': count}
        if name.startswith('sample-'):
            req['sampling'] = {'temperature': 1, 'top_p': .95, 'top_k': 40, 'seed': 42}
        identity = json.dumps(req, sort_keys=True)
        if identity not in indexes:
            indexes[identity] = len(fresh); fresh.append(req)
        requests.append({**req, **({'session_key': hashlib.sha256(key.encode()).hexdigest()} if key else {})})
        cases.append({'name': name, 'reference_index': indexes[identity], 'expected_reused': reuse,
                      'expected_restore': restored, 'prompt_tokens': len(ids)})
    return fresh, requests, cases


def audit_run(folder, name, requests, batch):
    """Re-read raw evidence, including complete logit length and all finite values."""
    data = json.loads((folder/(name+'.json')).read_text(encoding='utf-8'))
    assert len(data['results']) == len(requests), 'missing/extra native results'
    assert data['batch'] == batch and data['context'] == 1024
    assert data['strict_f32'] and data['kv'] == 'F32' and not data['flash_attention'] and not data['graphs']
    assert data['prefix_cache'] == (name == 'sessions')
    assert data['session_cache_mib'] == (768 if name == 'sessions' else 0)
    assert data['session_cache_slots'] == 4
    assert 'mm27-bounded-host-state-bulk' in data['patches']
    rows, offset, checks = [], 0, []
    for request, result in zip(requests, data['results']):
        rows.append(offset); offset += result['generated_tokens']
        assert result['prompt_tokens'] == len(request['tokens']) and result['max_tokens'] == request['max_tokens']
        assert 1 <= result['generated_tokens'] <= request['max_tokens']
        assert len(result['token_ids']) == result['generated_tokens']
        expected = {'expected_reused': result['reused_tokens'], 'expected_restore': result['session_restore']}
        if name == 'fresh':
            expected = {'expected_reused': 0, 'expected_restore': False}
        checks.append(inspect_case(result, data, expected))
    assert (folder/(name+'.f32')).stat().st_size == offset*NV*4, 'truncated/extra full logits'
    finite = compare_files(folder/(name+'.f32'), 0, folder/(name+'.f32'), 0, offset)
    assert finite['pass'] and all(all(c.values()) for c in checks), 'native or finite gate failed'
    return data, rows, checks


def compare_runs(folder, batch, cases):
    fresh_req = json.loads((folder/'fresh.requests.json').read_text(encoding='utf-8'))
    session_req = json.loads((folder/'sessions.requests.json').read_text(encoding='utf-8'))
    fresh, fresh_rows, _ = audit_run(folder, 'fresh', fresh_req, batch)
    sessions, rows, _ = audit_run(folder, 'sessions', session_req, batch)
    assert len(cases) == len(sessions['results'])
    comparisons = []
    for case, row, result, request in zip(cases, rows, sessions['results'], session_req):
        index = case['reference_index']; reference = fresh['results'][index]
        assert {k: v for k, v in request.items() if k != 'session_key'} == fresh_req[index]
        checks = inspect_case(result, sessions, case)
        checks['tokens'] = result['token_ids'] == reference['token_ids']
        checks['termination'] = result['stop_reason'] == reference['stop_reason']
        checks['sampling'] = result['sampling'] == reference['sampling']
        logits = compare_files(folder/'fresh.f32', fresh_rows[index], folder/'sessions.f32', row, result['generated_tokens'])
        comparisons.append({**case, 'checks': checks, 'logits': logits, 'generated_tokens': result['generated_tokens'],
                            'fresh_request_ms': reference['request_ms'], 'request_ms': result['request_ms'],
                            'session_ms': result['session_ms'], 'pass': all(checks.values()) and logits['pass']})
    return comparisons


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gguf', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--batch', type=int, choices=(1, 8), required=True)
    p.add_argument('--cuda-root', type=Path, default=ROOT/'build-local/cuda-13.0')
    args = p.parse_args(); args.out = args.out.resolve()
    binary = ROOT/'build-local/minimax-m2-cuda/bin/strata-minimax-m2-bench.exe'
    assert sha(binary) == EXE_SHA256
    args.out.mkdir(parents=True, exist_ok=False)
    report = {'pass': False, 'stage': 'MM27-31 boundary', 'batch': args.batch, 'context': 1024,
              'engine_sha256': EXE_SHA256, 'model': str(args.gguf.resolve()), 'runs': {}, 'cases': []}
    snapshot(args.out, report)
    for source in [Path(__file__), ROOT/'tools/test_minimax_m2_session_batches.py',
                   ROOT/'tools/check_minimax_m2_sessions.py', ROOT/'tools/check_minimax_m2_sessions_context.py',
                   ROOT/'tools/minimax_m2_memory_observer.py']:
        relative = source.resolve().relative_to(ROOT); dest = args.out/'sources'/relative
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, dest)
        report['sources'][relative.as_posix()] = sha(dest)
    shutil.copyfile(binary, args.out/'engine.exe')
    env = runtime_environment(args.cuda_root)
    env.update(STRATA_MM27_TOKENWISE='0', STRATA_MM27_STATE_BULK='1')
    report['environment'] = {k:v for k,v in env.items() if k.startswith(('STRATA_', 'GGML_', 'LLAMA_', 'NVIDIA_'))}
    report['gpu'] = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                                           '--format=csv,noheader'], text=True).strip()
    try:
        tok = Tokenizer.from_gguf(args.gguf)
        old = json.loads((ROOT/'build-local/minimax-m2-completion-01/requests.json').read_text(encoding='utf-8'))[0]
        base = tok.encode(old['prompt'], parse_special=True)
        long = (base + tok.encode(' The blue sky reflects scattered light.'*150))[:513]
        fresh, sessions, cases = make_corpus(base, long, tok.encode('red')[0], args.batch)
        report['cases'] = cases
        for name, requests in [('fresh', fresh), ('sessions', sessions)]:
            save(args.out/(name+'.requests.json'), requests)
            cmd = [str((args.out/'engine.exe').resolve()), '--gguf', str(args.gguf.resolve()), '--ctx', '1024',
                   '--batch', str(args.batch), '--mode', '2', '--prefix-cache', '1' if name == 'sessions' else '0',
                   '--session-cache-mib', '768' if name == 'sessions' else '0', '--session-cache-slots', '4',
                   '--gpu-cache-mib', '18432', '--gpu-cache-allocator', 'arena', '--pipeline-readers', '2',
                   '--pipeline-chunk-mib', '4', '--pipeline-lookahead', '1', '--pipeline-d2d-batch', '1',
                   '--request', str(args.out/(name+'.requests.json')), '--output', str(args.out/(name+'.json')),
                   '--logits', str(args.out/(name+'.f32'))]
            info = report['runs'][name] = {'command': cmd, 'pass': False}
            save(args.out/'report.json', report)
            print(f'START batch{args.batch} {name}: {len(requests)} requests', flush=True)
            with (args.out/(name+'.stdout.log')).open('wb') as out, (args.out/(name+'.stderr.log')).open('wb') as err:
                proc = subprocess.Popen(cmd, stdout=out, stderr=err, env=env, creationflags=subprocess.CREATE_NO_WINDOW)
                observer = None
                try:
                    if not contain(proc): raise RuntimeError('cannot contain native child')
                    info['pid'] = proc.pid
                    observer = MemoryObserver(args.out/(name+'.memory.jsonl'), on_error=lambda: proc.kill() if proc.poll() is None else None)
                    observer.start(); save(args.out/'report.json', report)
                    deadline = time.monotonic()+2400; last = -1
                    while proc.poll() is None:
                        if observer.error: raise RuntimeError(observer.error)
                        if time.monotonic() > deadline: raise TimeoutError('40 minute native limit')
                        f = args.out/(name+'.f32'); rows = f.stat().st_size//(NV*4) if f.exists() else 0
                        if rows != last:
                            print(f'batch{args.batch} {name}: {rows} full logit rows saved', flush=True); last = rows
                        time.sleep(2)
                    info['exit_code'] = proc.returncode
                    assert proc.returncode == 0, 'native failed; inspect stderr'
                finally:
                    if proc.poll() is None: proc.kill(); proc.wait(timeout=15)
                    info['process_exited'] = proc.poll() is not None
                    if observer:
                        observer.close(); info['observer'] = observer.summary()
                assert observer and observer.count and not observer.error
                samples = [json.loads(line) for line in (args.out/(name+'.memory.jsonl')).read_text().splitlines()]
                assert len(samples) == observer.count and all(memory_within_limit(m) for m in samples)
            _, _, info['native_checks'] = audit_run(args.out, name, requests, args.batch)
            info['pass'] = True; save(args.out/'report.json', report)
            print(f'DONE batch{args.batch} {name}', flush=True)
        report['cases'] = compare_runs(args.out, args.batch, cases)
        report['pass'] = all(c['pass'] for c in report['cases']) and all(r['pass'] for r in report['runs'].values())
        assert report['pass'], [c['name'] for c in report['cases'] if not c['pass']]
    except BaseException:
        report['error'] = traceback.format_exc(); print(report['error'], flush=True)
    finally:
        report['artifacts'] = {f.name: sha(f) for f in args.out.iterdir() if f.is_file() and f.name != 'report.json'}
        save(args.out/'report.json', report); print('REPORT pass='+str(report['pass']), flush=True)
    return 0 if report['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
