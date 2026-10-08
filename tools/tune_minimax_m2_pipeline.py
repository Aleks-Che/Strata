"""Sequential MiniMax cache/reader/chunk screening and repeated confirmation.

Every configuration uses strict F32 and the MM27-11 file pipeline, router
lookahead and tensor D2D batching. A new process starts each workload with an
empty GPU cache; OS file cache and external load are uncontrolled. Report
actual residency/limits, not only the requested cache cap. No defaults change.
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
from tools.minimax_m2_loader_contract import validate_loader_contract
from tools.minimax_m2_template import renderer, text_context
from tools.run_minimax_m2 import PRECISION_ENV, runtime_environment


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def configurations(path):
    result = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(result, list) or not 1 <= len(result) <= 32:
        raise ValueError('configs must be a list of 1..32 objects')
    names = set()
    for c in result:
        if set(c) != {'name', 'cache_mib', 'readers', 'chunk_mib'}:
            raise ValueError('config keys: name, cache_mib, readers, chunk_mib')
        name = c['name']
        if (not isinstance(name, str) or not 1 <= len(name) <= 64 or
                any(ch not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for ch in name) or name in names):
            raise ValueError('config names must be distinct lowercase labels')
        names.add(name)
        if (any(type(c[k]) is not int for k in ('cache_mib', 'readers', 'chunk_mib')) or
                not 1 <= c['cache_mib'] <= 131072 or c['readers'] not in (1, 2) or c['chunk_mib'] not in (4, 8, 16)):
            raise ValueError('invalid cache/readers/chunk values')
    return result


def audit(data, config, tokens):
    checks = {'configuration': data['gpu_cache_mib'] == config['cache_mib'] and
              data['gpu_cache_allocator'] == 'arena' and data['expert_reader'] == 'file' and
              data['pipeline_readers'] == config['readers'] and data['pipeline_chunk_mib'] == config['chunk_mib'] and
              data['pipeline_lookahead'] is True and data['pipeline_d2d_batch'] is True and
              data['strict_f32'] is True and data['kv'] == 'F32' and not data['flash_attention'] and
              not data['graphs'] and not data['mtp'] and not data['pipeline_trace'] and
              data['context'] == 2048 and data['batch'] == 16,
              'gpu_only': True, 'budget95': True, 'bytes': True, 'bounded_and_drained': True,
              'requests': len(data['results']) == 3}
    for r in data['results']:
        checks['requests'] &= (1 <= r['generated_tokens'] <= tokens and r['generated_tokens'] == len(r['token_ids']) and
                               (r['generated_tokens'] == tokens or r['stop_reason'] == 'eos'))
        for m in r['memory_samples']:
            checks['budget95'] &= (m['ram_total'] - m['ram_available'] <= .95 * m['ram_total'] and
                                  m['vram_total'] - m['vram_available'] <= .95 * m['vram_total'])
        m = r['memory_samples'][0]
        for s in (r['prefill'], r['decode']):
            checks['gpu_only'] &= s['gpu_nodes'] > 0 and s['rejected_cpu_nodes'] == s['rejected_full_copies'] == 0
            checks['budget95'] &= (s['sampled_ram_used_peak'] <= .95 * m['ram_total'] and
                                  s['sampled_vram_used_peak'] <= .95 * m['vram_total'])
            checks['bytes'] &= (s['source_bytes'] == s['file_bytes'] == s['h2d_bytes'] ==
                                s['pipeline_h2d_bytes'] == s['pipeline_d2d_bytes'] and
                                s['h2d_bytes'] + s['cache_hit_bytes'] == s['selected_bytes'] + s['cache_guard_bytes'] and
                                s['mmap_bytes'] == s['host_working_set_limit'] == 0)
            checks['bounded_and_drained'] &= (
                s['pipeline_unused_bytes'] == s['pipeline_queued_bytes'] == s['pipeline_reader_owned_bytes'] == 0 and
                s['staging_bytes'] == s['pipeline_device_bytes'] == config['chunk_mib'] * 4 * (1 << 20) and
                s['arena_live'] == s['cache_resident'] <= s['arena_reserved'] <= config['cache_mib'] * (1 << 20) and
                s['cache_resident'] <= s['cache_limit'] and s['pipeline_pending_fills_peak'] <= 256 and
                s['pipeline_plan_peak'] == 3 and s['pipeline_matrices'] == 3 * s['pipeline_plans'] and
                s['pipeline_copy_batches'] == s['pipeline_copy_fences'] == s['pipeline_scratch_fences'] == s['pipeline_matrices'] and
                s['pipeline_abort_fences'] == 0 and 0 < s['pipeline_read_peak'] <= config['readers'])
    return {k: bool(v) for k, v in checks.items()}


def summarize(runs, configs):
    summary = []
    for c in configs:
        rs = [r for r in runs if r['configuration'] == c['name']]
        if not rs:
            continue
        phases = [s for run in rs for r in run['data']['results'] for s in (r['prefill'], r['decode'])]
        m = rs[0]['data']['memory_before']
        row = {'configuration': c, 'runs': len(rs),
               'median_workload_seconds': statistics.median(sum(r['request_ms'] for r in run['data']['results']) / 1000 for run in rs),
               'ram_peak_percent': 100 * max(s['sampled_ram_used_peak'] for s in phases) / m['ram_total'],
               'vram_peak_percent': 100 * max(s['sampled_vram_used_peak'] for s in phases) / m['vram_total'], 'requests': []}
        for i in range(3):
            requests = [r['data']['results'][i] for r in rs]
            def med(f):
                return statistics.median(f(r) for r in requests)
            row['requests'].append({
                'prompt_tokens': requests[0]['prompt_tokens'],
                'generated_tokens': [r['generated_tokens'] for r in requests],
                'decode_tok_s': med(lambda r: r['decode_tokens_per_second']),
                'decode_range_tok_s': [min(r['decode_tokens_per_second'] for r in requests), max(r['decode_tokens_per_second'] for r in requests)],
                'ttft_ms': med(lambda r: r['ttft_ms']), 'request_ms': med(lambda r: r['request_ms']),
                'hit_byte_percent': med(lambda r: 100 * r['decode']['cache_hit_bytes'] / r['decode']['selected_bytes']),
                'h2d_gib_per_token': med(lambda r: r['decode']['h2d_bytes'] / r['decode_forward_tokens'] / 2**30),
                'cache_limit_gib': [r['decode']['cache_limit'] / 2**30 for r in requests],
                'cache_resident_gib': [r['decode']['cache_resident'] / 2**30 for r in requests],
                'arena_reserved_gib': [r['decode']['arena_reserved'] / 2**30 for r in requests]})
        summary.append(row)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--configs', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--repeats', type=int, choices=range(1, 11), default=1)
    p.add_argument('--tokens', type=int, choices=range(2, 257), default=24)
    p.add_argument('--reference-logits', type=Path)
    p.add_argument('--reference-report', type=Path)
    p.add_argument('--engine', type=Path, default=ROOT / 'build-local/minimax-m2-cuda/bin/strata-minimax-m2-bench.exe')
    p.add_argument('--cuda-root', type=Path, default=ROOT / 'build-local/cuda-13.0')
    args = p.parse_args()
    if bool(args.reference_logits) != bool(args.reference_report):
        p.error('provide both reference-logits and reference-report')
    configs = configurations(args.configs)
    g = GGUFFile(args.model)
    contract = validate_loader_contract(g.metadata, g.tensors)
    template = renderer(g.metadata['tokenizer.chat_template'])
    texts = ['Explain in a few sentences why the sky appears blue during the day.'] * 2
    texts += ['Прочитай список и объясни на русском, как найти его сумму: ' + ', '.join(map(str, range(1, 97)))]
    requests = [{'prompt': template.render(**text_context({'messages': [{'role': 'user', 'content': t}],
                  'add_generation_prompt': True})), 'max_tokens': args.tokens} for t in texts]
    args.out.mkdir(parents=True, exist_ok=False)
    save(args.out / 'requests.json', requests)
    save(args.out / 'configs.json', configs)
    exe = args.out / 'engine.exe'
    shutil.copyfile(args.engine, exe)
    sources = args.out / 'sources'
    shutil.copytree(ROOT / 'backends/minimax_m2', sources)
    shutil.copyfile(Path(__file__), sources / Path(__file__).name)
    with args.model.open('rb') as f:
        header_hash = hashlib.sha256(f.read(g.header_end)).hexdigest()
    report = {'pass': False, 'model': str(args.model), 'model_header_sha256': header_hash,
              'source_revision': contract['source_revision'], 'engine_sha256': sha(exe),
              'driver_sha256': sha(Path(__file__)), 'precision_environment': PRECISION_ENV,
              'configurations': configs, 'tokens_requested': args.tokens, 'repeats': args.repeats,
              'scope': 'sequential, fresh process per workload, alternating order; OS cache/external load uncontrolled; sampled memory peaks',
              'runs': [], 'comparisons': []}
    reference = args.reference_logits
    ref_data = json.loads(args.reference_report.read_text(encoding='utf-8')) if reference else None
    if reference:
        report['external_reference'] = {'logits': str(reference), 'sha256': sha(reference),
                                        'report': str(args.reference_report), 'report_sha256': sha(args.reference_report)}
    try:
        for repeat in range(args.repeats):
            for config in (configs if repeat % 2 == 0 else configs[::-1]):
                name = f"{repeat+1:02d}-{config['name']}"
                base = args.out / name
                cmd = [str(exe.resolve()), '--gguf', str(args.model.resolve()), '--request', str(args.out / 'requests.json'),
                       '--output', str(base.with_suffix('.json')), '--logits', str(base.with_suffix('.f32')),
                       '--ctx', '2048', '--batch', '16', '--gpu-cache-allocator', 'arena',
                       '--gpu-cache-mib', str(config['cache_mib']), '--pipeline-readers', str(config['readers']),
                       '--pipeline-chunk-mib', str(config['chunk_mib']), '--pipeline-lookahead', '1', '--pipeline-d2d-batch', '1']
                print(name + ': running', flush=True)
                with base.with_suffix('.stdout.log').open('w', encoding='utf-8') as out, base.with_suffix('.stderr.log').open('w', encoding='utf-8') as err:
                    run = subprocess.run(cmd, env=runtime_environment(args.cuda_root), stdout=out, stderr=err, timeout=1800)
                if run.returncode:
                    raise RuntimeError(f'{name}: exit {run.returncode}; retained stderr/executable for diagnosis')
                data = json.loads(base.with_suffix('.json').read_text(encoding='utf-8'))
                logits = np.fromfile(base.with_suffix('.f32'), dtype='<f4')
                expected = sum(r['generated_tokens'] for r in data['results']) * 200064
                if logits.size != expected or not np.isfinite(logits).all():
                    raise ValueError(f'{name}: invalid logits')
                gates = audit(data, config, args.tokens)
                report['runs'].append({'name': name, 'repeat': repeat+1, 'configuration': config['name'],
                                       'command': cmd, 'gates': gates, 'data': data,
                                       'logits_sha256': sha(base.with_suffix('.f32')), 'report_sha256': sha(base.with_suffix('.json'))})
                if reference is None:
                    reference, ref_data = base.with_suffix('.f32'), data
                else:
                    other = np.fromfile(reference, dtype='<f4')
                    exact = logits.shape == other.shape and logits.tobytes() == other.tobytes()
                    same_ids = [r['token_ids'] for r in data['results']] == [r['token_ids'] for r in ref_data['results']]
                    comparison = {'name': name, 'reference': str(reference), 'elements': int(logits.size),
                                  'bit_exact': exact, 'token_ids_equal': same_ids}
                    if not exact and logits.shape == other.shape:
                        index = int(np.flatnonzero(logits.view('<u4') != other.view('<u4'))[0])
                        comparison['first_mismatching_row_column'] = divmod(index, 200064)
                    report['comparisons'].append(comparison)
                    if not exact or not same_ids:
                        raise ValueError(f'{name}: numerical parity failed; do not loosen tolerances')
                if not all(gates.values()):
                    raise ValueError(f'{name}: failed gates {gates}')
                report['summary'] = summarize(report['runs'], configs)
                save(args.out / 'tuning-report.json', report)
                print(name + ': PASS; tok/s ' + ', '.join(f"{r['decode_tokens_per_second']:.3f}" for r in data['results']), flush=True)
        report['pass'] = True
    except Exception as exc:
        report['error'] = str(exc)
        raise
    finally:
        save(args.out / 'tuning-report.json', report)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
