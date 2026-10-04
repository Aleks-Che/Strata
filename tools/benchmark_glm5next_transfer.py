"""Compare warm selected GGUF ranges through the CUDA transport, without inference.

python -m tools.benchmark_glm5next_transfer --help
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import itertools
import json
import math
from pathlib import Path
import platform
import random
import re
import statistics
import subprocess

from tools.check_glm5next_transfer import MODES, digest, manifest
from tools.glm5next_expert_plan import integer, plan_expert_reads
from tools.setup_glm5next import inspect_model


def run_sample(checker, matrices, directory, chunk_bytes, mode, readers, decode, warmups, repeats, timeout):
    integer(readers, 'readers', 1)
    integer(warmups, 'warmups', 1)
    integer(repeats, 'repeats', 1)
    if readers > 4 or warmups > 5 or repeats > 20 or type(decode) is not bool:
        raise ValueError('Invalid benchmark options')
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('timeout must be finite and positive')
    payload = manifest(matrices, directory, chunk_bytes, mode)
    total = sum(m.bytes for m in matrices)
    guarded = total + 37 * (len(matrices) + 1)
    if guarded > 512 * 1024 * 1024:
        raise ValueError('Benchmark destinations exceed 512 MiB')
    command = [str(checker), '--benchmark', str(readers), str(int(decode)), str(warmups), str(repeats)]
    result = subprocess.run(command, input=payload, capture_output=True, check=True, timeout=timeout)
    lines = result.stdout.decode('ascii').splitlines()
    if len(lines) != repeats + 3:
        raise ValueError('Missing benchmark samples or verification')
    gpu = re.fullmatch(r'GPU ([0-9a-f]+) ([0-9]+) ([0-9]+)', lines[0])
    if not gpu:
        raise ValueError('Missing GPU identity')
    effective = 1 if decode and mode != 'native' else readers
    if lines[1] != (f'BENCH_CONFIG {readers} {effective} {int(decode)} {warmups} {repeats} '
                    f'{total} {guarded} {4*chunk_bytes} {4*chunk_bytes}'):
        raise ValueError('Benchmark configuration differs from request')
    samples = []
    fields = ('index', 'elapsed_ns', 'h2d_bytes', 'd2d_bytes', 'native_bytes', 'mmap_bytes',
              'chunks', 'read_us', 'slot_wait_us', 'consumer_wait_us', 'submit_us')
    chunks = sum((m.bytes + chunk_bytes - 1) // chunk_bytes for m in matrices)
    for index, line in enumerate(lines[2:-1]):
        if not re.fullmatch(r'SAMPLE(?: [0-9]+){11}', line):
            raise ValueError('Malformed sample')
        sample = dict(zip(fields, map(int, line.split()[1:])))
        if (sample['index'] != index or sample['elapsed_ns'] <= 0 or
                sample['h2d_bytes'] != total or sample['d2d_bytes'] != total or
                sample['native_bytes'] + sample['mmap_bytes'] != total or sample['chunks'] != chunks or
                (mode == 'native' and sample['native_bytes'] != total) or
                ((mode == 'mmap' or (mode == 'auto' and decode)) and sample['mmap_bytes'] != total)):
            raise ValueError('Benchmark sample disagrees with planned ranges/policy')
        samples.append(sample)
    comparisons = len(matrices) * (warmups + repeats)
    if lines[-1] != f'VERIFIED {comparisons}':
        raise ValueError('Benchmark did not verify every matrix')
    median_ns = statistics.median(s['elapsed_ns'] for s in samples)
    return {'mode': mode, 'phase': 'decode' if decode else 'prefill', 'readers': readers,
            'effective_readers': effective, 'gpu': bytes.fromhex(gpu[1]).decode('utf-8'),
            'cuda_runtime': int(gpu[2]), 'cuda_driver': int(gpu[3]), 'command': command,
            'byte_comparisons': comparisons, 'payload_bytes': total, 'destination_bytes': guarded,
            'pinned_bytes': 4*chunk_bytes, 'ring_bytes': 4*chunk_bytes, 'samples': samples,
            'median_ms': median_ns / 1e6, 'min_ms': min(s['elapsed_ns'] for s in samples) / 1e6,
            'max_ms': max(s['elapsed_ns'] for s in samples) / 1e6,
            'payload_gib_per_second': total / (1 << 30) / (median_ns / 1e9),
            'stderr_tail': result.stderr[-8192:].decode('utf-8', errors='replace')}


def benchmark(gguf, checker, layers, experts, chunk_bytes, warmups, repeats, timeout, seed):
    gguf, checker = Path(gguf).resolve(), Path(checker).resolve()
    if not layers or not experts:
        raise ValueError('Require layers and experts')
    report = inspect_model(gguf, tensor_details=True)
    matrices = [m for layer in dict.fromkeys(layers) for m in plan_expert_reads(report, layer, experts)]
    shards = []
    for shard in report['shard_details']:
        path = gguf.parent / shard['name']
        stat = path.stat()
        shards.append({'path': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                       'header_bytes': shard['header_end'], 'header_sha256': digest(path, shard['header_end'])})
    binary_hash = digest(checker)
    cases = list(itertools.product(MODES, (1, 2, 4), (False, True)))
    random.Random(seed).shuffle(cases)
    runs = []
    for mode, readers, decode in cases:
        run = run_sample(checker, matrices, gguf.parent, chunk_bytes, mode, readers,
                         decode, warmups, repeats, timeout)
        runs.append(run)
        print(f"{run['phase']} {mode} readers={readers}/{run['effective_readers']}: "
              f"{run['median_ms']:.3f} ms ({run['min_ms']:.3f}..{run['max_ms']:.3f})", flush=True)
    for shard in shards:
        stat = Path(shard['path']).stat()
        if (stat.st_size, stat.st_mtime_ns) != (shard['bytes'], shard['mtime_ns']):
            raise ValueError('Source changed during benchmark')
    if digest(checker) != binary_hash:
        raise ValueError('Checker changed during benchmark')
    if len({(r['gpu'], r['cuda_runtime'], r['cuda_driver']) for r in runs}) != 1:
        raise ValueError('GPU identity changed between runs')
    best = {}
    for phase in ('prefill', 'decode'):
        selected = min((r for r in runs if r['phase'] == phase), key=lambda r: r['median_ms'])
        best[phase] = {k: selected[k] for k in ('mode', 'readers', 'effective_readers', 'median_ms',
                                               'min_ms', 'max_ms', 'payload_gib_per_second')}
    return {'status': 'pass', 'first_shard': str(gguf), 'shards': shards,
            'checker': str(checker), 'checker_sha256': binary_hash,
            'layers': layers, 'experts': experts, 'chunk_bytes': chunk_bytes,
            'warmups': warmups, 'repeats': repeats, 'case_order_seed': seed,
            'matrix_count': len(matrices), 'quant_types': sorted({m.quant for m in matrices}),
            'matrices': [asdict(m) for m in matrices], 'runs': runs, 'lowest_observed_medians': best}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gguf', type=Path, required=True)
    parser.add_argument('--checker', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--layers', type=int, nargs='+', default=[3, 11, 45])
    parser.add_argument('--experts', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 287])
    parser.add_argument('--chunk-bytes', type=int, default=1024*1024)
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--seed', type=int, default=17)
    args = parser.parse_args(argv)
    if args.output.suffix.lower() != '.json':
        parser.error('Output must be a .json report')
    for source in [args.gguf, args.checker, *args.gguf.parent.glob('*.gguf')]:
        if args.output.resolve() == source.resolve() or (args.output.exists() and source.exists() and args.output.samefile(source)):
            parser.error('Output must not overwrite a source shard or checker')
    result = {'schema_version': 1, 'status': 'error', 'started_utc': datetime.now(timezone.utc).isoformat(),
              'platform': platform.platform(), 'processor': platform.processor(),
              'scope': 'Warm selected packed ranges, full payload and guards verified; no cache, compute or inference',
              'timed_region': 'pipeline start, all ordered transfers, finish and final consumer stream synchronization',
              'excluded_from_timer': ['allocations', 'stdio baseline', 'mapping warmup', 'warmup passes',
                                      'destination memset', 'D2H verification', 'byte comparison'],
              'warm_policy': 'All selected mapping bytes compared with stdio baseline before pipeline warmups; no cache flush',
              'physical_disk_reads_measured': False, 'runtime_defaults_changed': False,
              'limitations': ['Wall time includes CPU scheduling, source reads, H2D and D2D; not pure PCIe bandwidth',
                              'Decode mmap/auto uses one active reader regardless of configured reader count',
                              'No compute overlap timeline, model token rate, cold-storage or whole-model residency measurement',
                              'Lowest observed medians are local observations, not runtime recommendations']}
    try:
        result.update(benchmark(args.gguf, args.checker, args.layers, args.experts, args.chunk_bytes,
                                args.warmups, args.repeats, args.timeout, args.seed))
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        result['error'] = str(error)
    result['finished_utc'] = datetime.now(timezone.utc).isoformat()
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return 0 if result['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
