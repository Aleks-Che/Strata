"""MiniMax JSONL completion, input-boundary and real CTRL_BREAK recovery checks.

Windows only. A hidden console supervisor sends CTRL_BREAK solely to the
benchmark process group it creates; no other process or console is signalled.
Run after check_minimax_m2_completion.py, with no concurrent GPU benchmark.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_minimax_m2 import runtime_environment
from tools.tune_minimax_m2_pipeline import save, sha


def result_checks(result, header):
    checks = {'gpu_only': True, 'budget95': True, 'bytes': True, 'drained': True}
    for m in result['memory_samples']:
        checks['budget95'] &= (m['ram_total']-m['ram_available'] <= .95*m['ram_total'] and
                              m['vram_total']-m['vram_available'] <= .95*m['vram_total'])
    m = header['memory_before']
    for s in (result['prefill'], result['decode']):
        checks['gpu_only'] &= s['gpu_nodes'] > 0 and s['rejected_cpu_nodes'] == s['rejected_full_copies'] == 0
        checks['budget95'] &= s['sampled_ram_used_peak'] <= .95*m['ram_total'] and s['sampled_vram_used_peak'] <= .95*m['vram_total']
        checks['bytes'] &= (s['source_bytes'] == s['file_bytes'] == s['h2d_bytes'] == s['pipeline_h2d_bytes'] == s['pipeline_d2d_bytes'] and
                            s['h2d_bytes']+s['cache_hit_bytes'] == s['selected_bytes']+s['cache_guard_bytes'])
        checks['drained'] &= (s['pipeline_queued_bytes'] == s['pipeline_reader_owned_bytes'] == s['pipeline_unused_bytes'] == 0 and
                              s['cache_pending_matrices'] == s['ram_cache_readers'] == s['ram_cache_bytes'] == 0 and
                              s['arena_live'] == s['cache_resident'] <= s['arena_reserved'] <= 18432*2**20 and
                              s['arena_reserved'] <= s['cache_limit'])
    return checks


def worker(args):
    if not ctypes.windll.kernel32.GetConsoleCP():
        raise RuntimeError('the isolated supervisor needs its own hidden console')
    original = json.loads((args.source/'completion-report.json').read_text(encoding='utf-8'))
    # The Russian long case may fail the natural-completion gate while the
    # English reference is complete and numerically validated. Preserve that
    # failure; this check certifies only the selected English request.
    if not (all(original['gates'].values()) and original['repeat']['pass'] and
            original['completions'][0]['pass'] and all(c['pass'] for c in original['prefix_comparisons'])):
        raise ValueError('selected English reference and numerical gates must pass first')
    reference = json.loads((args.source/'generation.json').read_text(encoding='utf-8'))
    prompt = json.loads((args.source/'requests.json').read_text(encoding='utf-8'))[0]['prompt']
    expected = reference['results'][0]
    assert expected['generated_tokens'] > 257 and expected['stop_reason'] == 'eos'
    cmd = [str((args.out/'engine.exe').resolve()), '--gguf', reference['model'], '--pipe', '--ctx', '2048', '--batch', '16',
           '--gpu-cache-mib', '18432', '--gpu-cache-allocator', 'arena', '--pipeline-readers', '2',
           '--pipeline-chunk-mib', '4', '--pipeline-lookahead', '1', '--pipeline-d2d-batch', '1']
    report = {'pass': False, 'engine_sha256': sha(args.out/'engine.exe'), 'driver_sha256': sha(Path(__file__)),
              'reference_report': str(args.source/'generation.json'), 'reference_sha256': sha(args.source/'generation.json'),
              'command': cmd, 'cases': [], 'scope': 'one live JSONL process; targeted CTRL_BREAK after 257 emitted tokens; full token/text parity, not logit parity'}
    env = runtime_environment(args.cuda_root)
    env['STRATA_MM27_TOKENWISE'] = '0'
    inbox = queue.Queue()
    process = None
    try:
        with (args.out/'engine.stderr.log').open('w', encoding='utf-8') as stderr, (args.out/'events.jsonl').open('w', encoding='utf-8') as transcript:
            process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
                                       encoding='utf-8', env=env, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
            def reader():
                try:
                    for line in process.stdout:
                        inbox.put((time.monotonic(), line))
                finally:
                    inbox.put((time.monotonic(), None))
            thread = threading.Thread(target=reader, daemon=True)
            thread.start()
            def event(timeout=180):
                when, line = inbox.get(timeout=timeout)
                if line is None:
                    raise RuntimeError('unexpected engine EOF')
                transcript.write(line)
                transcript.flush()
                return when, json.loads(line)
            _, header = event(300)
            assert header['event'] == 'ready'
            report['header'] = header
            assert all(header[k] == reference[k] for k in ['context', 'batch', 'kv', 'strict_f32', 'flash_attention',
                       'graphs', 'gpu_cache_mib', 'gpu_cache_allocator', 'pipeline_readers', 'pipeline_chunk_mib',
                       'pipeline_lookahead', 'pipeline_d2d_batch', 'ram_cache_mib', 'cache_group_experts'])
            def request(name, r, cancel_after=None):
                start = time.monotonic()
                process.stdin.write(json.dumps(r, ensure_ascii=False)+'\n')
                process.stdin.flush()
                ids, signal_at = [], None
                while True:
                    remaining = 900-(time.monotonic()-start)
                    if remaining <= 0:
                        raise TimeoutError('request exceeded 900 seconds: '+name)
                    when, e = event(min(180, remaining))
                    if e['event'] == 'token':
                        ids.append(e['id'])
                        if cancel_after and len(ids) >= cancel_after and signal_at is None:
                            signal_at = time.monotonic()
                            if not ctypes.windll.kernel32.GenerateConsoleCtrlEvent(1, process.pid):
                                raise ctypes.WinError()
                    elif e['event'] in ('result', 'error'):
                        return {'name': name, 'request': r, 'streamed_token_ids': ids, 'terminal': e,
                                'elapsed_ms': (when-start)*1000,
                                'cancel_latency_ms': (when-signal_at)*1000 if signal_at else None}
                    else:
                        raise ValueError('unexpected event: '+str(e))
            def record(case):
                report['cases'].append(case)
                save(args.out/'pipe-report.json', report)
                print(case['name']+': '+('PASS' if case['pass'] else 'FAIL'), flush=True)
                if not case['pass']:
                    raise RuntimeError('failed gate: '+case['name'])
            invalid = [(f'max_{i}', {'prompt': prompt, 'max_tokens': value}) for i, value in enumerate(
                       [True, False, 0, -1, 1.5, '257', None, 2**64-1, 2049])]
            invalid += [('context_overflow', {'prompt': prompt, 'max_tokens': 2049-expected['prompt_tokens']}),
                        ('full_prompt_plus_one', {'tokens': [1]*2048, 'max_tokens': 1}),
                        ('empty_prompt', {'tokens': [], 'max_tokens': 1}),
                        ('invalid_token', {'tokens': [200064], 'max_tokens': 1})]
            for name, r in invalid:
                case = request(name, r)
                case['pass'] = case['terminal']['event'] == 'error' and not case['streamed_token_ids']
                record(case)
            def completed(name, r, full):
                case = request(name, r)
                result = case['terminal']
                checks = result_checks(result, header)
                want = expected['token_ids'] if full else expected['token_ids'][:8]
                checks['ids'] = case['streamed_token_ids'] == result['token_ids'] == want
                checks['counts'] = result['generated_tokens'] == len(want) and result['decode_forward_tokens'] == len(want)-1
                checks['stop'] = (result['stop_reason'] == ('eos' if full else 'length') and
                                   result['stop_token_id'] == (200020 if full else None))
                checks['budget'] = result['max_tokens'] == r.get('max_tokens', 8)
                if full:
                    checks['text'] = result['text'] == expected['text']
                    checks['EOS_exactly_once'] = want[-1] == 200020 and want.count(200020) == 1
                case['checks'] = checks
                case['pass'] = all(checks.values())
                record(case)
            completed('default_eight_after_invalid', {'prompt': prompt}, False)
            exact = {'prompt': prompt, 'max_tokens': 2048-expected['prompt_tokens']}
            completed('exact_context_budget_natural_EOS', exact, True)
            case = request('cancel_after_257', exact, cancel_after=257)
            ids = case['streamed_token_ids']
            case['pass'] = (case['terminal']['event'] == 'error' and 'cancelled' in case['terminal']['message'] and
                            257 <= len(ids) < expected['generated_tokens'] and ids == expected['token_ids'][:len(ids)] and
                            case['cancel_latency_ms'] is not None and 0 <= case['cancel_latency_ms'] < 30000)
            record(case)
            completed('full_response_after_cancel', {'prompt': prompt, 'max_tokens': original['tokens_requested']}, True)
            process.stdin.close()
            report['exit_code'] = process.wait(timeout=90)
            thread.join(timeout=5)
            leftovers = []
            while not inbox.empty():
                _, line = inbox.get_nowait()
                if line is not None:
                    leftovers.append(line)
            report['extra_events_after_last_result'] = leftovers
            report['pass'] = report['exit_code'] == 0 and not leftovers and all(c['pass'] for c in report['cases'])
    except Exception as exc:
        report['error'] = str(exc)
        raise
    finally:
        if process and process.poll() is None:
            process.kill()  # Only the test process created above, after failure.
            process.wait(timeout=30)
        save(args.out/'pipe-report.json', report)
    return 0 if report['pass'] else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True, help='completion directory with a passing English reference')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--engine', type=Path, default=ROOT/'build-local/minimax-m2-cuda/bin/strata-minimax-m2-bench.exe')
    p.add_argument('--cuda-root', type=Path, default=ROOT/'build-local/cuda-13.0')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = p.parse_args()
    if os.name != 'nt':
        p.error('this native Windows CTRL_BREAK check requires Windows')
    if args.worker:
        return worker(args)
    args.out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.engine, args.out/'engine.exe')
    shutil.copyfile(Path(__file__), args.out/Path(__file__).name)
    command = [sys.executable, '-X', 'utf8', str(Path(__file__).resolve()), '--source', str(args.source.resolve()),
               '--out', str(args.out.resolve()), '--cuda-root', str(args.cuda_root.resolve()), '--worker']
    save(args.out/'supervisor-command.json', command)
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = subprocess.SW_HIDE
    with (args.out/'supervisor.stdout.log').open('w', encoding='utf-8') as stdout, (args.out/'supervisor.stderr.log').open('w', encoding='utf-8') as stderr:
        # The worker bounds every read/request/exit wait and owns cleanup of
        # its child. Killing only the supervisor on an outer timeout would
        # bypass that cleanup and could leave the GPU child running.
        result = subprocess.run(command, stdout=stdout, stderr=stderr, startupinfo=startup,
                                creationflags=subprocess.CREATE_NEW_CONSOLE)
    print('pipe supervisor exit:', result.returncode)
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
