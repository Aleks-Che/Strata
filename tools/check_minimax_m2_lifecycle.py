"""Run sequential MiniMax context/recovery/pressure checks on this PC.

The full_pressure case targets 90% global RAM and adds bounded GPU pressure
(with a cache, up to 12 GiB while respecting the 95% guard), using a read-only
checkpoint mapping and disposable GPU allocations. Other cases test
near-full context windows using a repeated prefix and varied 33-token suffix.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_minimax_m2 import runtime_environment

CASES = {
    'fixture': ['--fixture', '--ctx', '4096', '--tokens', '4079', '--batch', '16'],
    'full_pressure': ['--ctx', '512', '--tokens', '31', '--batch', '8', '--pressure'],
    'context2048': ['--ctx', '2048', '--tokens', '2031', '--batch', '16', '--long-only'],
    'context4096': ['--ctx', '4096', '--tokens', '4079', '--batch', '16', '--long-only'],
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--cases', nargs='+', choices=list(CASES), default=list(CASES))
    p.add_argument('--gpu-cache-mib', type=int, default=0)
    p.add_argument('--gpu-cache-allocator', choices=['cuda', 'arena'], default='cuda')
    p.add_argument('--expert-reader', choices=['file', 'mmap', 'mmap-direct', 'mmap-decode'], default='file')
    p.add_argument('--engine', type=Path, default=ROOT / 'build-local/minimax-m2-cuda/bin/strata-minimax-m2-lifecycle-check.exe')
    p.add_argument('--cuda-root', type=Path, default=ROOT / 'build-local/cuda-13.0')
    p.add_argument('--pipeline-readers', type=int, choices=[0, 1, 2], default=0)
    p.add_argument('--pipeline-chunk-mib', type=int, choices=[4, 8, 16], default=8)
    p.add_argument('--pipeline-lookahead', type=int, choices=[0, 1], default=0)
    p.add_argument('--pipeline-d2d-batch', type=int, choices=[0, 1], default=0)
    args = p.parse_args()
    if args.pipeline_readers and args.expert_reader != 'file':
        p.error('pipeline requires file reader')
    if not 0 <= args.gpu_cache_mib <= 131072:
        p.error('invalid GPU cache cap')
    if any(c != 'fixture' for c in args.cases) and (not args.model or not args.model.is_file()):
        p.error('full-model cases require --model GGUF')
    args.out.mkdir(parents=True, exist_ok=False)
    env = runtime_environment(args.cuda_root)
    report = {'engine_sha256': sha(args.engine), 'cases': [], 'pass': False}
    try:
        # Preserve the actual binary, not only its hash: the original executable
        # from the two MM27-06 failures was lost when the harness was rebuilt.
        retained = args.out / 'engine.exe'
        shutil.copyfile(args.engine, retained)
        report['retained_engine'] = str(retained)
        source_dir = args.out / 'sources'
        source_dir.mkdir()
        report['source_sha256'] = {}
        for source in (ROOT / 'backends/minimax_m2').iterdir():
            if source.is_file():
                shutil.copyfile(source, source_dir / source.name)
                report['source_sha256'][source.relative_to(ROOT).as_posix()] = sha(source)
        for name in args.cases:
            directory = args.out / name
            command = [str(retained.resolve()), '--out', str(directory), *CASES[name]]
            command += ['--gpu-cache-mib', str(args.gpu_cache_mib), '--gpu-cache-allocator', args.gpu_cache_allocator]
            command += ['--expert-reader', args.expert_reader]
            command += ['--pipeline-readers', str(args.pipeline_readers), '--pipeline-chunk-mib', str(args.pipeline_chunk_mib), '--pipeline-lookahead', str(args.pipeline_lookahead), '--pipeline-d2d-batch', str(args.pipeline_d2d_batch)]
            if name != 'fixture':
                command += ['--gguf', str(args.model.resolve())]
            print(f'{name}: running', flush=True)
            with (args.out / f'{name}.stdout.log').open('w', encoding='utf-8') as out, (args.out / f'{name}.stderr.log').open('w', encoding='utf-8') as err:
                run = subprocess.run(command, env=env, stdout=out, stderr=err, timeout=3600)
            path = directory / 'lifecycle-report.json'
            result = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {'pass': False, 'error': 'missing report'}
            case = {'name': name, 'command': command, 'exit_code': run.returncode, 'result': result,
                    'pass': run.returncode == 0 and result.get('pass') is True,
                    'logits_sha256': {f.name: sha(f) for f in directory.glob('*.f32')}}
            report['cases'].append(case)
            (args.out / 'suite-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f'{name}: {"PASS" if case["pass"] else "FAIL"}', flush=True)
            if not case['pass']:
                break
        report['pass'] = len(report['cases']) == len(args.cases) and all(c['pass'] for c in report['cases'])
    except Exception as exc:
        report['error'] = str(exc)
        raise
    finally:
        (args.out / 'suite-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return 0 if report['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
