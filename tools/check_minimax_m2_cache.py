"""Sequential strict cache A/B, with repeated and changed-topic requests.

Each process starts with an empty GPU cache. OS file-cache state is uncontrolled;
no claim of cold SSD throughput. All logits and greedy IDs must be bit-exact.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import shutil
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.gguf_reader import GGUFFile
from tools.minimax_m2_loader_contract import validate_loader_contract
from tools.minimax_m2_template import renderer, text_context
from tools.run_minimax_m2 import runtime_environment


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--cache-mib', type=int, default=18432)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--tokens', type=int, default=24)
    p.add_argument('--comparison', choices=['cache', 'allocator', 'reader', 'pipeline', 'lookahead', 'd2d'], default='cache')
    p.add_argument('--pipeline-readers', type=int, choices=[1, 2], default=1)
    p.add_argument('--pipeline-chunk-mib', type=int, choices=[4, 8, 16], default=8)
    p.add_argument('--candidate-reader', choices=['mmap', 'mmap-direct', 'mmap-decode'], default='mmap-direct')
    p.add_argument('--engine', type=Path, default=ROOT / 'build-local/minimax-m2-cuda/bin/strata-minimax-m2-bench.exe')
    p.add_argument('--cuda-root', type=Path, default=ROOT / 'build-local/cuda-13.0')
    args = p.parse_args()
    if not 2 <= args.tokens <= 256 or not 1 <= args.repeats <= 10 or not 1 <= args.cache_mib <= 131072:
        p.error('invalid tokens/repeats/cache size')
    args.out.mkdir(parents=True, exist_ok=False)
    variants = {'cache': ['off', 'on'], 'allocator': ['cuda', 'arena'], 'reader': ['file', args.candidate_reader],
                'pipeline': ['sync', 'pipeline'], 'lookahead': ['tensor', 'lookahead'], 'd2d': ['lookahead', 'batched']}[args.comparison]
    g = GGUFFile(args.model)
    contract = validate_loader_contract(g.metadata, g.tensors)
    template = renderer(g.metadata['tokenizer.chat_template'])
    def request(text):
        return {'prompt': template.render(**text_context({'messages': [{'role': 'user', 'content': text}],
                                                          'add_generation_prompt': True})), 'max_tokens': args.tokens}
    short = request('Explain in a few sentences why the sky appears blue during the day.')
    longer = request('Прочитай список и объясни на русском, как найти его сумму: ' + ', '.join(map(str, range(1, 97))))
    names = ['short_empty_cache', 'short_repeated', 'longer_new_topic']
    request_path = args.out / 'requests.json'
    request_path.write_text(json.dumps([short, short, longer], ensure_ascii=False), encoding='utf-8')
    with args.model.open('rb') as f:
        header_hash = hashlib.sha256(f.read(g.header_end)).hexdigest()
    report = {'model': str(args.model), 'header_sha256': header_hash, 'source_revision': contract['source_revision'],
              'engine_sha256': sha(args.engine), 'cache_mib': args.cache_mib, 'context': 2048, 'batch': 16,
              'tokens_requested': args.tokens, 'sampler': 'greedy', 'strict_f32': True, 'flash_attention': False,
              'comparison': args.comparison, 'variants': variants,
              'pipeline_readers': args.pipeline_readers, 'pipeline_chunk_mib': args.pipeline_chunk_mib,
              'scope': 'fresh process per variant; empty GPU cache at first request; OS file cache uncontrolled; file bytes are not SSD bytes',
              'names': names, 'runs': [], 'comparisons': []}
    dest = args.out / 'cache-ab-report.json'
    def save():
        dest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    try:
        retained = args.out / 'engine.exe'
        shutil.copyfile(args.engine, retained)
        report['retained_engine'] = str(retained)
        for repeat in range(args.repeats):
            pairs = {}
            for variant in (variants if repeat % 2 == 0 else variants[::-1]):
                base = args.out / f'{repeat + 1}-{variant}'
                output, logits = base.with_suffix('.json'), base.with_suffix('.f32')
                arena = args.comparison in ('reader', 'pipeline', 'lookahead', 'd2d') or variant == 'arena'
                pipelined = variant in ('pipeline', 'tensor', 'lookahead', 'batched')
                command = [str(retained.resolve()), '--gguf', str(args.model.resolve()), '--request', str(request_path),
                           '--output', str(output), '--logits', str(logits), '--ctx', '2048', '--batch', '16',
                           '--gpu-cache-mib', str(args.cache_mib if variant != 'off' else 0),
                           '--gpu-cache-allocator', 'arena' if arena else 'cuda',
                           '--expert-reader', variant if args.comparison == 'reader' else 'file']
                command += ['--pipeline-readers', str(args.pipeline_readers if pipelined else 0),
                            '--pipeline-chunk-mib', str(args.pipeline_chunk_mib)]
                if args.comparison in ('lookahead', 'd2d'):
                    command += ['--pipeline-lookahead', str(int(variant in ('lookahead', 'batched')))]
                if args.comparison == 'd2d':
                    command += ['--pipeline-d2d-batch', str(int(variant == 'batched'))]
                print(f'repeat {repeat + 1}: cache {variant}', flush=True)
                with base.with_suffix('.stdout.log').open('w', encoding='utf-8') as out, base.with_suffix('.stderr.log').open('w', encoding='utf-8') as err:
                    result = subprocess.run(command, env=runtime_environment(args.cuda_root), stdout=out, stderr=err, timeout=1800)
                if result.returncode:
                    raise RuntimeError(f'{base.name}: exit {result.returncode}; see stderr log')
                data = json.loads(output.read_text(encoding='utf-8'))
                expected_reader = variant if args.comparison == 'reader' else 'file'
                if data['expert_reader'] != expected_reader:
                    raise ValueError('unexpected reader configuration')
                if data['pipeline_readers'] != (args.pipeline_readers if pipelined else 0):
                    raise ValueError('unexpected pipeline configuration')
                if args.comparison == 'lookahead' and data['pipeline_lookahead'] != (variant == 'lookahead'):
                    raise ValueError('unexpected lookahead configuration')
                if args.comparison == 'd2d' and (not data['pipeline_lookahead'] or data['pipeline_d2d_batch'] != (variant == 'batched')):
                    raise ValueError('unexpected batched configuration')
                values = np.fromfile(logits, dtype='<f4')
                expected = sum(r['generated_tokens'] for r in data['results']) * 200064
                if values.size != expected or not np.isfinite(values).all():
                    raise ValueError('invalid logits')
                gpu_only = budget = accounting = True
                for r in data['results']:
                    for s in (r['prefill'], r['decode']):
                        gpu_only &= s['gpu_nodes'] > 0 and s['rejected_cpu_nodes'] == s['rejected_full_copies'] == 0
                        accounting &= s['source_bytes'] == s['h2d_bytes'] and s['h2d_bytes'] + s['cache_hit_bytes'] == s['selected_bytes'] + s['cache_guard_bytes']
                        accounting &= s['source_bytes'] == s['file_bytes'] + s['mmap_bytes']
                        if pipelined:
                            accounting &= s['pipeline_h2d_bytes'] == s['pipeline_d2d_bytes'] == s['h2d_bytes']
                            accounting &= s['pipeline_unused_bytes'] == s['pipeline_queued_bytes'] == s['pipeline_reader_owned_bytes'] == 0
                            accounting &= s['staging_bytes'] == s['pipeline_device_bytes'] == args.pipeline_chunk_mib * 4 * (1 << 20)
                            accounting &= s['pipeline_groups'] > 0 and 0 < s['pipeline_read_peak'] <= args.pipeline_readers
                        if variant in ('lookahead', 'batched'):
                            accounting &= s['pipeline_plan_peak'] == 3 and s['pipeline_lookahead_plans'] > 0
                            accounting &= s['pipeline_matrices'] == 3 * s['pipeline_plans']
                        if variant == 'batched':
                            accounting &= s['pipeline_copy_batches'] == s['pipeline_copy_fences'] == s['pipeline_scratch_fences'] == s['pipeline_matrices']
                            accounting &= s['pipeline_copy_fences'] < s['pipeline_copy_submissions'] and s['pipeline_abort_fences'] == 0
                        if expected_reader == 'file':
                            accounting &= s['mmap_bytes'] == s['host_working_set_limit'] == 0
                        elif expected_reader == 'mmap-decode':
                            accounting &= s['host_working_set_limit'] > 0
                            if s is r['prefill']:
                                accounting &= s['mmap_bytes'] == 0
                            else:
                                accounting &= s['mmap_bytes'] > 0
                        else:
                            accounting &= s['file_bytes'] == 0 and s['host_working_set_limit'] > 0
                            if expected_reader == 'mmap-direct':
                                accounting &= s['staging_bytes'] == 0
                        accounting &= s['arena_reserved'] <= args.cache_mib * (1 << 20)
                        if arena:
                            accounting &= s['arena_live'] == s['cache_resident'] <= s['arena_reserved']
                        mem = r['memory_samples'][0]
                        budget &= s['sampled_ram_used_peak'] <= .95 * mem['ram_total'] and s['sampled_vram_used_peak'] <= .95 * mem['vram_total']
                run = {'repeat': repeat + 1, 'variant': variant, 'command': command, 'exit_code': result.returncode,
                       'gpu_only': bool(gpu_only), 'budget95': bool(budget), 'byte_accounting': bool(accounting),
                       'logits_sha256': sha(logits), 'report_sha256': sha(output), 'data': data}
                report['runs'].append(run)
                pairs[variant] = (values, run)
                save()
                print('  decode tok/s: ' + ', '.join(f"{r['decode_tokens_per_second']:.3f}" for r in data['results']), flush=True)
            a, b = pairs[variants[0]][0], pairs[variants[1]][0]
            exact = a.shape == b.shape and a.tobytes() == b.tobytes()
            ids = [r['token_ids'] for r in pairs[variants[0]][1]['data']['results']] == [r['token_ids'] for r in pairs[variants[1]][1]['data']['results']]
            report['comparisons'].append({'repeat': repeat + 1, 'elements': int(a.size), 'bit_exact': exact, 'token_ids_equal': ids})
            save()
            if not exact or not ids:
                raise ValueError('cache changed logits or tokens')
        summary = []
        for i, name in enumerate(names):
            samples = {v: [x['data']['results'][i] for x in report['runs'] if x['variant'] == v] for v in variants}
            speeds = {v: statistics.median(r['decode_tokens_per_second'] for r in samples[v]) for v in samples}
            summary.append({'name': name, 'prompt_tokens': samples[variants[0]][0]['prompt_tokens'], 'decode_median_tok_s': speeds,
                            'speedup_percent': (speeds[variants[1]] / speeds[variants[0]] - 1) * 100,
                            'decode_hit_byte_percent': {v: statistics.median(100 * r['decode']['cache_hit_bytes'] / r['decode']['selected_bytes'] for r in samples[v]) for v in variants}})
        report['summary'] = summary
        report['pass'] = all(r[k] for r in report['runs'] for k in ('gpu_only', 'budget95', 'byte_accounting'))
    except Exception as exc:
        report['pass'] = False
        report['error'] = str(exc)
        raise
    finally:
        save()
    print(json.dumps(report['summary'], ensure_ascii=False, indent=2), flush=True)
    return 0 if report['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
