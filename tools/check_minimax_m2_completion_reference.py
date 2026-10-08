"""Compare a retained long MiniMax prefix with serial ReadFile/H2D, caches off.

This isolates GPU caching and the file pipeline. It is not an independent
model/framework oracle and does not turn a repeating completion into a PASS.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.check_minimax_m2_completion import compare, rows
from tools.run_minimax_m2 import runtime_environment
from tools.tune_minimax_m2_pipeline import save, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--request-index', type=int, default=2)
    p.add_argument('--tokens', type=int, default=1024)
    p.add_argument('--engine', type=Path, default=ROOT/'build-local/minimax-m2-cuda/bin/strata-minimax-m2-bench.exe')
    p.add_argument('--cuda-root', type=Path, default=ROOT/'build-local/cuda-13.0')
    args = p.parse_args()
    original = json.loads((args.source/'generation.json').read_text(encoding='utf-8'))
    requests = json.loads((args.source/'requests.json').read_text(encoding='utf-8'))
    if not 0 <= args.request_index < len(requests):
        p.error('request index is outside the reference corpus')
    prior = original['results'][args.request_index]
    if not 257 <= args.tokens <= prior['generated_tokens']:
        p.error('tokens must be 257..the retained generated length')
    request = dict(requests[args.request_index], max_tokens=args.tokens)
    args.out.mkdir(parents=True, exist_ok=False)
    save(args.out/'request.json', request)
    exe = args.out/'engine.exe'
    shutil.copyfile(args.engine, exe)
    shutil.copyfile(Path(__file__), args.out/Path(__file__).name)
    command = [str(exe.resolve()), '--gguf', original['model'], '--request', str(args.out/'request.json'),
               '--output', str(args.out/'generation.json'), '--logits', str(args.out/'generation.f32'),
               '--ctx', str(original['context']), '--batch', str(original['batch']), '--mode', '2',
               '--gpu-cache-mib', '0', '--ram-cache-mib', '0', '--pipeline-readers', '0']
    env = runtime_environment(args.cuda_root)
    env['STRATA_MM27_TOKENWISE'] = '0'
    report = {'pass': False, 'scope': 'numerical isolation of serial file/H2D vs cache+pipeline; not a natural-completion or independent-oracle gate',
              'command': command, 'engine_sha256': sha(exe), 'driver_sha256': sha(Path(__file__)),
              'reference': str(args.source/'generation.json'), 'reference_sha256': sha(args.source/'generation.json'),
              'request_index': args.request_index, 'tokens_requested': args.tokens}
    try:
        save(args.out/'reference-report.json', report)
        print('serial cache-off reference: running', flush=True)
        with (args.out/'stdout.log').open('w', encoding='utf-8') as stdout, (args.out/'stderr.log').open('w', encoding='utf-8') as stderr:
            run = subprocess.run(command, env=env, stdout=stdout, stderr=stderr, timeout=1800)
        report['exit_code'] = run.returncode
        if run.returncode:
            raise RuntimeError(f'serial engine exit {run.returncode}; retained diagnostics')
        data = json.loads((args.out/'generation.json').read_text(encoding='utf-8'))
        result = data['result']
        actual = rows(args.out/'generation.f32', [result])
        old = rows(args.source/'generation.f32', original['results'])
        offset = sum(r['generated_tokens'] for r in original['results'][:args.request_index])
        report['comparison'] = compare(actual, old[offset:offset+args.tokens])
        report['comparison']['token_ids_equal'] = result['token_ids'] == prior['token_ids'][:args.tokens]
        report['comparison']['pass'] &= report['comparison']['token_ids_equal']
        m = data['memory_before']
        checks = {'configuration': data['mode'] == 2 and data['expert_reader'] == 'file' and
                    data['gpu_cache_mib'] == data['ram_cache_mib'] == data['pipeline_readers'] == 0 and
                    data['strict_f32'] and data['kv'] == 'F32' and not data['flash_attention'] and not data['graphs'],
                  'gpu_only': True, 'budget95': True, 'bytes': True, 'serial_only': True}
        for s in (result['prefill'], result['decode']):
            checks['gpu_only'] &= s['gpu_nodes'] > 0 and s['rejected_cpu_nodes'] == s['rejected_full_copies'] == 0
            checks['budget95'] &= s['sampled_ram_used_peak'] <= .95*m['ram_total'] and s['sampled_vram_used_peak'] <= .95*m['vram_total']
            checks['bytes'] &= s['source_bytes'] == s['file_bytes'] == s['h2d_bytes'] == s['selected_bytes']
            checks['serial_only'] &= (s['staging_bytes'] == 16*2**20 and s['cache_resident'] == s['cache_hit_bytes'] ==
                                     s['pipeline_plans'] == s['pipeline_queued_bytes'] == s['ram_cache_bytes'] == 0)
        for sample in result['memory_samples']:
            checks['budget95'] &= (sample['ram_total']-sample['ram_available'] <= .95*sample['ram_total'] and
                                  sample['vram_total']-sample['vram_available'] <= .95*sample['vram_total'])
        report['checks'] = checks
        phrase = 'We can also show that the sum of numbers from 1 to 96 is 4656.'
        report['diagnostic'] = {'stop_reason': result['stop_reason'], 'reasoning_closed': '</think>' in result['text'],
                                 'repeated_phrase': phrase, 'phrase_count': result['text'].count(phrase),
                                 'generated_tokens': result['generated_tokens'], 'decode_tokens_per_second': result['decode_tokens_per_second']}
        report['generation_report_sha256'] = sha(args.out/'generation.json')
        report['generation_logits_sha256'] = sha(args.out/'generation.f32')
        report['pass'] = all(checks.values()) and report['comparison']['pass']
        print(json.dumps({'pass': report['pass'], 'comparison': report['comparison'], 'diagnostic': report['diagnostic']}, ensure_ascii=False), flush=True)
    except Exception as exc:
        report['error'] = str(exc)
        raise
    finally:
        save(args.out/'reference-report.json', report)
    return 0 if report['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
