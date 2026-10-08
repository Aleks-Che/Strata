"""Validate bounded CUDA H2D/compute intervals and exact logits on one request.

Diagnostic events add overhead. This is not a throughput benchmark.
The reference must contain the same prompt's first two F32 vocabulary rows.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_minimax_m2 import runtime_environment


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--request', type=Path, required=True)
    p.add_argument('--reference-logits', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--engine', type=Path, default=ROOT / 'build-local/minimax-m2-cuda/bin/strata-minimax-m2-bench.exe')
    p.add_argument('--cuda-root', type=Path, default=ROOT / 'build-local/cuda-13.0')
    p.add_argument('--pipeline-d2d-batch', type=int, choices=[0, 1], default=0)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    exe = args.out / 'engine.exe'
    shutil.copyfile(args.engine, exe)
    request = json.loads(args.request.read_text(encoding='utf-8'))
    if isinstance(request, list):
        request = request[0]
    request['max_tokens'] = 2
    (args.out / 'request.json').write_text(json.dumps(request, ensure_ascii=False), encoding='utf-8')
    command = [str(exe.resolve()), '--gguf', str(args.model.resolve()), '--request', str(args.out / 'request.json'),
               '--output', str(args.out / 'report.json'), '--logits', str(args.out / 'logits.f32'),
               '--ctx', '2048', '--batch', '16', '--gpu-cache-mib', '18432', '--gpu-cache-allocator', 'arena',
               '--pipeline-readers', '2', '--pipeline-chunk-mib', '4', '--pipeline-lookahead', '1',
               '--pipeline-d2d-batch', str(args.pipeline_d2d_batch),
               '--pipeline-trace', str(args.out / 'trace.json')]
    report = {'pass': False, 'command': command, 'engine_sha256': sha(exe),
              'reference': str(args.reference_logits), 'reference_sha256': sha(args.reference_logits),
              'scope': 'first four router plans; CUDA event diagnostics, not a throughput measurement'}
    try:
        with (args.out / 'stdout.log').open('w', encoding='utf-8') as out, (args.out / 'stderr.log').open('w', encoding='utf-8') as err:
            run = subprocess.run(command, env=runtime_environment(args.cuda_root), stdout=out, stderr=err, timeout=600)
        report['exit_code'] = run.returncode
        assert run.returncode == 0
        trace = json.loads((args.out / 'trace.json').read_text())
        data = json.loads((args.out / 'report.json').read_text())
        report.update(trace=trace, data=data)
        expected = np.fromfile(args.reference_logits, dtype='<f4', count=2 * 200064)
        actual = np.fromfile(args.out / 'logits.f32', dtype='<f4')
        assert expected.size == actual.size == 2 * 200064 and np.isfinite(actual).all()
        report['bit_exact'] = expected.tobytes() == actual.tobytes()
        assert report['bit_exact']
        assert len(trace['groups']) == 4
        h2d = compute = overlap = 0.
        for g in trace['groups']:
            assert not g['cancelled']
            h, c, d = (g[k] for k in ('h2d_intervals_ms', 'compute_intervals_ms', 'delivery_intervals_ms'))
            assert h and c and d
            for intervals in (h, c, d):
                assert all(np.isfinite(a) and np.isfinite(b) and 0 <= a <= b for a, b in intervals)
                assert all(b <= n + 1e-4 for (_, b), (n, _) in zip(intervals, intervals[1:]))
            intersection = sum(max(0., min(b, y) - max(a, x)) for a, b in h for x, y in c)
            assert abs(intersection - g['h2d_compute_overlap_ms']) < 1e-4
            assert 0 <= intersection <= min(sum(b-a for a, b in h), sum(b-a for a, b in c)) + 1e-4
            h2d += sum(b-a for a, b in h)
            compute += sum(b-a for a, b in c)
            overlap += intersection
        for s in (data['result']['prefill'], data['result']['decode']):
            if args.pipeline_d2d_batch:
                assert s['pipeline_copy_fences'] == s['pipeline_copy_batches'] == s['pipeline_matrices']
            assert s['pipeline_plan_peak'] == 3 and s['pipeline_matrices'] == 3 * s['pipeline_plans']
            assert s['pipeline_h2d_bytes'] == s['pipeline_d2d_bytes'] == s['h2d_bytes']
            assert s['pipeline_unused_bytes'] == s['pipeline_queued_bytes'] == s['pipeline_reader_owned_bytes'] == 0
        report.update(h2d_ms=h2d, compute_ms=compute, h2d_compute_overlap_ms=overlap, overlap_observed=overlap > 0, **{'pass': True})
    finally:
        (args.out / 'trace-check.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print({k: report[k] for k in ('pass', 'bit_exact', 'h2d_ms', 'compute_ms', 'h2d_compute_overlap_ms')})


if __name__ == '__main__':
    main()
