"""Check each supported MiniMax native-file pipeline configuration sequentially.

Retain the tested executable, source hashes and raw output. The default sync
reader regression runs first; these fixtures are correctness, not throughput.
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


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--engine', type=Path, default=ROOT / 'build-local/minimax-m2-cuda/bin/strata-minimax-m2-cache-check.exe')
    parser.add_argument('--cuda-root', type=Path, default=ROOT / 'build-local/cuda-13.0')
    parser.add_argument('--lookahead', action='store_true', help='check three-matrix lookahead as well as tensor mode')
    parser.add_argument('--d2d-batch', action='store_true', help='include tensor and lookahead batched copies')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    report = {'pass': False, 'cases': [], 'scope': 'synthetic bytes/logits/cache/fault/recovery; sequential GPU runs'}
    try:
        retained = args.out / 'engine.exe'
        shutil.copyfile(args.engine, retained)
        report['engine_sha256'] = sha(retained)
        report['source_sha256'] = {p.relative_to(ROOT).as_posix(): sha(p)
                                  for p in (ROOT / 'backends/minimax_m2').iterdir() if p.is_file()}
        for name in ('expert_pipeline.hpp', 'expert_slice.hpp', 'expert_file.hpp'):
            p = ROOT / 'backends/common' / name
            report['source_sha256'][p.relative_to(ROOT).as_posix()] = sha(p)
        configs = [(0, 8, 0, 0), *[(r, c, 0, 0) for r in (1, 2) for c in (4, 8, 16)]]
        if args.lookahead:
            configs += [(r, c, 1, 0) for r in (1, 2) for c in (4, 8, 16)]
        if args.d2d_batch:
            configs += [(r, c, l, 1) for r in (1, 2) for c in (4, 8, 16) for l in (0, 1)]
        for readers, chunk, lookahead, batched in configs:
            name = f'readers{readers}-chunk{chunk}' + ('-lookahead' if lookahead else '') + ('-batched' if batched else '')
            command = [str(retained.resolve()), str(args.out / name), '--pipeline-readers', str(readers), '--pipeline-chunk-mib', str(chunk)]
            if lookahead:
                command += ['--pipeline-lookahead', '1']
            if batched:
                command += ['--pipeline-d2d-batch', '1']
            print(name + ': running', flush=True)
            with (args.out / f'{name}.stdout.log').open('w', encoding='utf-8') as out, (args.out / f'{name}.stderr.log').open('w', encoding='utf-8') as err:
                run = subprocess.run(command, env=runtime_environment(args.cuda_root), stdout=out, stderr=err, timeout=600)
            result_path = args.out / name / 'cache-report.json'
            result = json.loads(result_path.read_text(encoding='utf-8')) if result_path.exists() else {'pass': False}
            expected = 168 if lookahead else 152 if readers else 504
            if batched:
                expected += 48
            case = {'name': name, 'command': command, 'exit_code': run.returncode, 'result': result,
                    'pass': run.returncode == 0 and result.get('pass') is True and result.get('cases') == expected}
            report['cases'].append(case)
            print(name + (': PASS' if case['pass'] else ': FAIL'), flush=True)
            if not case['pass']:
                break
        report['pass'] = len(report['cases']) == len(configs) and all(c['pass'] for c in report['cases'])
    finally:
        (args.out / 'suite-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return 0 if report['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
