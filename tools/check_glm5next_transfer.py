"""Check real GLM GGUF expert bytes through the shared Windows CUDA pipeline.

python -m tools.check_glm5next_transfer --help
Read-only ranges, no model inference, dequantization or throughput benchmark.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

from tools.glm5next_expert_plan import integer, plan_expert_reads
from tools.setup_glm5next import inspect_model

MODES = {'mmap': 0, 'native': 1, 'auto': 2}


def digest(path, limit=None):
    sha = hashlib.sha256()
    with Path(path).open('rb') as source:
        while limit is None or limit > 0:
            data = source.read(1024 * 1024 if limit is None else min(limit, 1024 * 1024))
            if not data:
                if limit:
                    raise ValueError('Short header while hashing')
                break
            sha.update(data)
            if limit is not None:
                limit -= len(data)
    return sha.hexdigest()


def manifest(matrices, directory, chunk_bytes, mode, cache_identity=None, dispatch_check=False):
    integer(chunk_bytes, 'chunk_bytes', 1)
    if chunk_bytes > 16 * 1024 * 1024 or mode not in MODES or not 1 <= len(matrices) <= 4096:
        raise ValueError('Invalid transfer manifest limits or mode')
    lines = [f'GLM_RANGES_V1 {chunk_bytes} {MODES[mode]} {len(matrices)}']
    if dispatch_check and (cache_identity is None or len(matrices) % 3 or len(set(matrices)) != len(matrices)):
        raise ValueError('Dispatch requires identity and distinct complete triples')
    if cache_identity is not None:
        if not re.fullmatch(r'[0-9a-f]{64}', cache_identity):
            raise ValueError('Cache identity must be a SHA-256 hex string')
        protocol = 'GLM_DISPATCH_RANGES_V1' if dispatch_check else 'GLM_CACHE_RANGES_V1'
        lines = [f'{protocol} {chunk_bytes} {MODES[mode]} {len(matrices)} {cache_identity}']
    chunks = 0
    for m in matrices:
        integer(m.file_offset, 'file_offset')
        integer(m.bytes, 'matrix bytes', 1)
        if m.file_offset > (1 << 64) - 1 or m.bytes > 256 * 1024 * 1024:
            raise ValueError('Matrix exceeds native check limits')
        chunks += (m.bytes + chunk_bytes - 1) // chunk_bytes
        if chunks > 1048576:
            raise ValueError('Plan exceeds chunk metadata limit')
        path = (directory / m.shard).resolve()
        # An inspection report names basenames only; never accept parent traversal.
        if path.parent != directory.resolve():
            raise ValueError('Shard path must stay within model directory')
        lines.append(f'{m.file_offset} {m.bytes} {str(path).encode("utf-8").hex()}')
        if cache_identity is not None:
            for name in ('layer', 'expert'):
                if integer(getattr(m, name), name) > (1 << 31) - 1:
                    raise ValueError('Cache index exceeds int range')
            for name in ('columns', 'rows'):
                if integer(getattr(m, name), name, 1) > (1 << 64) - 1:
                    raise ValueError('Cache dimension exceeds uint64 range')
            if (m.branch not in ('main', 'mtp') or m.projection not in ('gate', 'up', 'down') or
                    not re.fullmatch(r'[A-Z0-9_]{1,64}', m.quant) or
                    m.file_offset + m.bytes > (1 << 64) - 1):
                raise ValueError('Invalid cache key fields')
            lines[-1] += (f' {m.branch} {m.layer} {m.expert} {m.projection} {m.quant}'
                          f' {m.columns} {m.rows} {m.shard.encode("utf-8").hex()}')
    return ('\n'.join(lines) + '\n').encode('ascii')


def run_transport(command, matrices, directory, chunk_bytes, mode, timeout, cache_identity=None, dispatch_check=False):
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('timeout must be finite and positive')
    payload = manifest(matrices, directory, chunk_bytes, mode, cache_identity, dispatch_check)
    result = subprocess.run(command, input=payload, capture_output=True, timeout=timeout, check=True)
    lines = result.stdout.decode('ascii').splitlines()
    cached = cache_identity is not None
    expected_lines = 1 + len(matrices)//3*4 if dispatch_check else len(matrices) + (3 if cached else 2)
    if len(lines) != expected_lines:
        raise ValueError('Transfer checker must return every matrix and final counters')
    gpu = re.fullmatch(r'GPU ([0-9a-f]+) ([0-9]+) ([0-9]+)', lines[0])
    if not gpu:
        raise ValueError('Missing CUDA device record')
    name = bytes.fromhex(gpu[1]).decode('utf-8')
    if not name or int(gpu[2]) <= 0 or int(gpu[3]) <= 0:
        raise ValueError('Invalid CUDA device record')
    if dispatch_check:
        scenarios = []
        for group in range(len(matrices)//3):
            a, c, d = matrices[group*3:group*3+3]
            delivered = 3*a.bytes + 4*c.bytes + 3*d.bytes
            for frequency in range(2):
                for decode in range(2):
                    line = lines[1+group*4+frequency*2+decode]
                    if not re.fullmatch(r'DISPATCH_OK(?: [0-9]+){9}', line):
                        raise ValueError('Malformed dispatch result')
                    g, f, dec, comparisons, h2d, d2d, native, mmap, unused = map(int, line.split()[1:])
                    source = native + mmap
                    if ((g, f, dec, comparisons) != (group, frequency, decode, 14) or
                            d2d != delivered or not delivered <= h2d <= source <= delivered+d.bytes or
                            (mode == 'native' and native != source) or
                            ((mode == 'mmap' or (mode == 'auto' and decode)) and mmap != source) or
                            unused > (d.bytes+chunk_bytes-1)//chunk_bytes):
                        raise ValueError('Dispatch stages, counters or reader policy disagree with plan')
                    scenarios.append({'group': group, 'policy': 'frequency' if frequency else 'lru',
                                      'decode': bool(decode), 'byte_comparisons': comparisons,
                                      'h2d_bytes': h2d, 'ring_d2d_bytes': d2d, 'native_bytes': native,
                                      'mmap_bytes': mmap, 'unused_prefetched_chunks': unused,
                                      'hits': 4, 'misses': 10, 'admissions': 5, 'bypasses': 5,
                                      'evictions': 3, 'invalidations': 2})
        return {'mode': mode, 'status': 'pass', 'matrix_count': len(matrices), 'gpu': name,
                'cuda_runtime': int(gpu[2]), 'cuda_driver': int(gpu[3]), 'scenarios': scenarios,
                'byte_comparisons': sum(s['byte_comparisons'] for s in scenarios),
                'stderr_tail': result.stderr[-8192:].decode('utf-8', errors='replace')}
    for i, m in enumerate(matrices):
        expected = f'CACHE_OK {i} {m.bytes} 5 1 4 2 1' if cached else f'OK {i} {m.bytes}'
        if lines[i + 1] != expected:
            raise ValueError(f'Missing, reordered or invalid matrix result at {i}')
    total_line = lines[-2] if cached else lines[-1]
    counter = re.fullmatch(r'TOTAL(?: ([0-9]+)){5}', total_line)
    if not counter:
        raise ValueError('Malformed transport counters')
    h2d, d2d, native, mmap, chunks = map(int, total_line.split()[1:])
    uploads = 4 if cached else 1
    total = uploads * sum(m.bytes for m in matrices)
    if (h2d != total or d2d != total or native + mmap != total or
            chunks != uploads * sum((m.bytes + chunk_bytes - 1) // chunk_bytes for m in matrices) or
            (mode == 'native' and native != total) or (mode == 'mmap' and mmap != total)):
        raise ValueError('Transport counters disagree with planned ranges/mode')
    cache_stats = {}
    if cached:
        n = len(matrices)
        if lines[-1] != f'CACHE_TOTAL {n} {4*n} {2*n} {n} {5*n}':
            raise ValueError('Cache lifecycle counters disagree with planned scenario')
        cache_stats = {'cache': {'hits': n, 'misses': 4*n, 'admissions': 4*n,
                                'evictions': 2*n, 'invalidations': n, 'byte_comparisons': 5*n,
                                'bypasses': 0, 'hit_source_bytes': 0, 'hit_h2d_bytes': 0}}
    return {'mode': mode, 'status': 'pass', 'matrix_count': len(matrices), 'gpu': name,
            'cuda_runtime': int(gpu[2]), 'cuda_driver': int(gpu[3]),
            'h2d_bytes': h2d, 'd2d_bytes': d2d, 'native_bytes': native, 'mmap_bytes': mmap,
            'chunks': chunks, **cache_stats,
            'stderr_tail': result.stderr[-8192:].decode('utf-8', errors='replace')}


def check(gguf, checker, layers, experts, modes, chunk_bytes, timeout, cache_check=False, *, dispatch_check=False):
    gguf, checker = Path(gguf).resolve(), Path(checker).resolve()
    if not layers or not experts or not modes or len(set(modes)) != len(modes):
        raise ValueError('Require layers, experts and distinct modes')
    if cache_check and dispatch_check:
        raise ValueError('Choose cache or dispatch check')
    report = inspect_model(gguf, tensor_details=True)
    matrices = [m for layer in dict.fromkeys(layers) for m in plan_expert_reads(report, layer, experts)]
    shards = []
    for s in report['shard_details']:
        path = gguf.parent / s['name']
        stat = path.stat()
        shards.append({'path': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                       'header_bytes': s['header_end'], 'header_sha256': digest(path, s['header_end'])})
    binary_hash = digest(checker)
    # Identifies this observed file set for the isolated checker. Not a full
    # payload digest or a production loader/session fingerprint.
    identity = hashlib.sha256(json.dumps(shards, sort_keys=True).encode('utf-8')).hexdigest() if cache_check or dispatch_check else None
    options = {'dispatch_check': True} if dispatch_check else {}
    runs = [run_transport([str(checker)], matrices, gguf.parent, chunk_bytes, mode, timeout, identity, **options) for mode in modes]
    for s in shards:
        stat = Path(s['path']).stat()
        if (stat.st_size, stat.st_mtime_ns) != (s['bytes'], s['mtime_ns']):
            raise ValueError('Source changed during transfer validation')
    if digest(checker) != binary_hash:
        raise ValueError('Checker changed during transfer validation')
    cache_info = {}
    if cache_check or dispatch_check:
        cache_info = {'cache_identity': identity,
                      'cache_identity_scope': 'SHA-256 of observed shard paths, sizes, mtimes and header hashes; not full weights',
                      'cache_scenario': {'policy': 'lru', 'budget': 'one matrix, recreated per range',
                                         'stages': ['cold_generation_1', 'hit_on_second_stream',
                                                    'evict_with_generation_2', 'reload_generation_1',
                                                    'invalidate_1_and_load_generation_3']}}
        if dispatch_check:
            cache_info['cache_scenario'] = {
                'policies': ['lru', 'frequency'], 'reader_policies': ['prefill', 'decode'],
                'budget': 'A+C bytes per consecutive triple; zero in last phase',
                'stages': ['warm_A_and_B_generation_2_alias_of_C', 'mixed_C_A_D_evict_B_pin_A_bypass_D',
                           'all_hit_C_A_without_sources', 'cancel_after_A_before_D',
                           'invalidate_generation_1_load_generation_3', 'zero_budget_A_C_D'],
                'byte_comparisons_per_triple_policy': 14,
                'source_counters_include_cancel_read_ahead': True,
                'ring_d2d_excludes_cache_to_destination': True}
    return {'status': 'pass', 'first_shard': str(gguf), 'shards': shards, **cache_info,
            'checker': str(checker), 'checker_sha256': binary_hash,
            'requested_layers': layers, 'requested_experts': experts, 'chunk_bytes': chunk_bytes,
            'matrix_count': len(matrices), 'quant_types': sorted({m.quant for m in matrices}),
            'matrices': [asdict(m) for m in matrices], 'runs': runs}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gguf', type=Path, required=True)
    parser.add_argument('--checker', type=Path, required=True)
    parser.add_argument('--layers', type=int, nargs='+', default=[3, 11, 45])
    parser.add_argument('--experts', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 287])
    parser.add_argument('--modes', choices=MODES, nargs='+', default=list(MODES))
    parser.add_argument('--chunk-bytes', type=int, default=262161)
    parser.add_argument('--timeout', type=float, default=120)
    checks = parser.add_mutually_exclusive_group()
    checks.add_argument('--cache-check', action='store_true', help='Check cache hits, forced eviction and generation reload for every range')
    checks.add_argument('--dispatch-check', action='store_true', help='Check mixed cache/transport dispatch, cancellation and reload on consecutive triples')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.suffix.lower() != '.json':
        parser.error('Output must be a .json report')
    for source in [args.gguf, args.checker, *args.gguf.parent.glob('*.gguf')]:
        if args.output.resolve() == source.resolve() or (args.output.exists() and source.exists()
                                                       and args.output.samefile(source)):
            parser.error('Output must not overwrite a source shard or checker')
    result = {'schema_version': 3 if args.dispatch_check else 2 if args.cache_check else 1, 'status': 'error',
              'scope': ('Selected real packed bytes via cache-aware dispatch; no GLM graph, dequantization or inference'
                        if args.dispatch_check else 'Selected real packed bytes via shared GPU transport and isolated cache; no GLM graph, dequantization or inference'
                        if args.cache_check else 'Selected real packed bytes via shared GPU transport; no GLM graph, dequantization, cache or inference'),
              'physical_disk_reads_measured': False}
    try:
        result.update(check(args.gguf, args.checker, args.layers, args.experts,
                            args.modes, args.chunk_bytes, args.timeout, args.cache_check, dispatch_check=args.dispatch_check))
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        if isinstance(exc, subprocess.CalledProcessError):
            result['checker_stdout_tail'] = (exc.stdout or b'')[-8192:].decode('utf-8', errors='replace')
            result['checker_stderr_tail'] = (exc.stderr or b'')[-8192:].decode('utf-8', errors='replace')
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'{result["status"]}: {result.get("matrix_count", 0)} matrices; {args.output}')
    return 0 if result['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
