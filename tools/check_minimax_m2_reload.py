"""Run the MiniMax reload stress and retain the exact executable for reproducing failures.

The optional MM27-06 historical check applies to the pinned synthetic fixture on
the same strict RTX 5090 baseline. It does not relax or replace per-run comparisons.
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
from tools.run_minimax_m2 import PRECISION_ENV, runtime_environment

FIXTURE_SHA256 = '0fe45556a1d92c4c8783716ae1118e2278a84fc0693b98d85f45edf03e4981e9'
CONTINUATION_SHA256 = '662a8b04a40790b72cff94d4106f368fcfea7c77ec5a6816bc46380153648589'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--engine', type=Path, default=ROOT / 'build-local/minimax-m2-cuda/bin/strata-minimax-m2-reload-check.exe')
    p.add_argument('--cuda-root', type=Path, default=ROOT / 'build-local/cuda-13.0')
    p.add_argument('--check-mm27-06-baseline', action='store_true')
    p.add_argument('--expert-reader', choices=['file', 'mmap', 'mmap-direct', 'mmap-decode'], default='file')
    p.add_argument('--pipeline-readers', type=int, choices=[0, 1, 2], default=0)
    p.add_argument('--pipeline-chunk-mib', type=int, choices=[4, 8, 16], default=8)
    p.add_argument('--pipeline-lookahead', type=int, choices=[0, 1], default=0)
    p.add_argument('--pipeline-d2d-batch', type=int, choices=[0, 1], default=0)
    args = p.parse_args()
    if args.pipeline_readers and args.expert_reader != 'file':
        p.error('pipeline requires file reader')
    if not args.engine.is_file():
        p.error('reload-check executable missing; build it first')
    args.out.mkdir(parents=True, exist_ok=False)
    report = {'pass': False, 'precision_environment': PRECISION_ENV, 'checks': []}
    try:
        # A hash alone cannot reproduce a failure after the next rebuild.
        retained = args.out / 'engine.exe'
        shutil.copyfile(args.engine, retained)
        report['engine_sha256'] = sha(retained)
        snapshot = args.out / 'sources'
        snapshot.mkdir()
        report['source_sha256'] = {}
        for source in (ROOT / 'backends/minimax_m2').iterdir():
            if source.is_file():
                shutil.copyfile(source, snapshot / source.name)
                report['source_sha256'][source.relative_to(ROOT).as_posix()] = sha(source)
        for source in ('backends/step35/expert_cache.hpp', 'backends/hy3/gpu_arena.hpp',
                       'backends/common/expert_file.hpp', 'backends/common/device_memory.hpp',
                       'backends/common/vram_policy.hpp', 'backends/common/expert_frequency.hpp',
                       'tools/check_minimax_m2_reload.py', 'tools/run_minimax_m2.py'):
            report['source_sha256'][source] = sha(ROOT / source)
        command = [str(retained.resolve()), str((args.out / 'stress').resolve()), '--expert-reader', args.expert_reader]
        command += ['--pipeline-readers', str(args.pipeline_readers), '--pipeline-chunk-mib', str(args.pipeline_chunk_mib), '--pipeline-lookahead', str(args.pipeline_lookahead), '--pipeline-d2d-batch', str(args.pipeline_d2d_batch)]
        report['command'] = command
        print('reload stress: running nine sequential lifetimes', flush=True)
        with (args.out / 'stdout.log').open('w', encoding='utf-8') as out, (args.out / 'stderr.log').open('w', encoding='utf-8') as err:
            run = subprocess.run(command, env=runtime_environment(args.cuda_root), stdout=out, stderr=err, timeout=600)
        report['exit_code'] = run.returncode
        result = json.loads((args.out / 'stress/reload-report.json').read_text(encoding='utf-8'))
        report['result'] = result
        report['checks'].append({'name': 'all_55_cases', 'pass': run.returncode == 0 and result.get('pass') is True
                                 and result.get('cases') == 55 and len(result.get('tests', [])) == 55
                                 and all(t.get('pass') is True for t in result['tests'])})
        files = list((args.out / 'stress').glob('*.f32'))
        report['logits_sha256'] = {f.name: sha(f) for f in files}
        report['fixture_sha256'] = sha(args.out / 'stress/fixture.gguf')
        report['checks'].append({'name': 'ten_native_outputs_equal', 'pass': len(files) == 10
                                 and len(set(report['logits_sha256'].values())) == 1})
        if args.check_mm27_06_baseline:
            report['checks'].append({'name': 'historical_fixture_identity', 'pass': report['fixture_sha256'] == FIXTURE_SHA256})
            report['checks'].append({'name': 'historical_continuation_bits', 'pass':
                                    set(report['logits_sha256'].values()) == {CONTINUATION_SHA256}})
        report['pass'] = all(c['pass'] for c in report['checks'])
    except Exception as exc:
        report['error'] = str(exc)
        raise
    finally:
        (args.out / 'suite-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('reload stress: ' + ('PASS' if report['pass'] else 'FAIL'), flush=True)
    return 0 if report['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
