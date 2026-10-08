"""Run the MiniMax DFlash loader and offline GPU correctness probe, sequentially.

Retains binaries, source, inputs, full logits and stderr even on failure. This is
not an end-to-end speculative decoding benchmark and never enables serving.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.gguf_reader import GGUFFile
from tools.inspect_minimax_m2_drafts import inspect
from tools.run_minimax_m2 import PRECISION_ENV, runtime_environment


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--draft', type=Path, action='append', required=True)
    p.add_argument('--gpu-draft', type=Path, action='append', default=[], help='Subset of --draft; omit for loader only')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--build', type=Path, default=ROOT / 'build-local/minimax-m2-cuda')
    p.add_argument('--cuda-root', type=Path, default=ROOT / 'build-local/cuda-13.0')
    args = p.parse_args()
    args.model = args.model.resolve()
    drafts = [d.resolve() for d in args.draft]
    gpu_drafts = [d.resolve() for d in args.gpu_draft]
    if len(set(drafts)) != len(drafts) or len(set(gpu_drafts)) != len(gpu_drafts) or not set(gpu_drafts) <= set(drafts):
        p.error('drafts must be unique; GPU drafts must be included in --draft')
    inspection = inspect(args.model, drafts, full_hash=True)
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    save(args.out / 'inspection.json', inspection)
    shutil.copytree(ROOT / 'backends/minimax_m2', args.out / 'sources/backends/minimax_m2')
    shutil.copytree(ROOT / 'backends/common', args.out / 'sources/backends/common')
    for rel in ('backends/step35/expert_cache.hpp', 'backends/hy3/gpu_arena.hpp',
                'tools/check_minimax_m2_dflash.py', 'tools/inspect_minimax_m2_drafts.py'):
        dst = args.out / 'sources' / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, dst)
    engines = ['dflash_loader'] + (['dflash-check'] if gpu_drafts else [])
    for name in engines:
        shutil.copyfile(args.build / 'bin' / f'strata-minimax-m2-{name}.exe', args.out / f'{name}.exe')
    shutil.copyfile(args.build / 'minimax-m2-build-manifest.json', args.out / 'build-manifest.json')
    report = {'pass': False, 'scope': 'loader, short GPU block/rollback/head-prefix checks; no useful tokens/s measurement',
              'precision_environment': PRECISION_ENV, 'target_header_sha256': inspection['target_header_sha256'],
              'target_full_sha256': None, 'drafts': [{k: d[k] for k in ('path', 'file_bytes', 'header_sha256', 'full_sha256')} for d in inspection['drafts']],
              'engine_sha256': {n: sha(args.out / f'{n}.exe') for n in engines},
              'source_sha256': {s.relative_to(args.out / 'sources').as_posix(): sha(s) for s in (args.out / 'sources').rglob('*') if s.is_file()},
              'attempts': [], 'loaders': [], 'negative_checks': [], 'gpu_runs': []}
    env = runtime_environment(args.cuda_root)

    def run(name, exe, arguments, expected_error=None):
        cmd = [str(args.out / f'{exe}.exe'), *map(str, arguments)]
        attempt = {'name': name, 'command': cmd}
        report['attempts'].append(attempt)
        save(args.out / 'suite-report.json', report)
        print(name+': running', flush=True)
        out, err = args.out / f'{name}.stdout.json', args.out / f'{name}.stderr.log'
        with out.open('w', encoding='utf-8') as stdout, err.open('w', encoding='utf-8') as stderr:
            result = subprocess.run(cmd, stdout=stdout, stderr=stderr, env=env, timeout=900)
        attempt['exit_code'] = result.returncode
        if expected_error:
            ok = result.returncode == 1 and expected_error in err.read_text(encoding='utf-8', errors='replace')
            report['negative_checks'].append({'name': name, 'pass': ok, 'expected_error': expected_error})
        else:
            ok = result.returncode == 0
        if not ok:
            raise RuntimeError(f'{name}: unexpected exit {result.returncode}; all artifacts retained')
        print(name+': PASS', flush=True)
        return out

    try:
        for i, draft in enumerate(drafts):
            result = json.loads(run(f'loader-{i}', 'dflash_loader', [args.model, draft]).read_text(encoding='utf-8'))
            report['loaders'].append(result)
        header = GGUFFile(drafts[0])
        with drafts[0].open('rb') as f:
            data = f.read(header.data_start)
        truncated = args.out / 'truncated.gguf'
        truncated.write_bytes(data)
        run('reject-truncated', 'dflash_loader', [args.model, truncated], 'invalid draft range:')
        needle = struct.pack('<5i', 2, 17, 31, 45, 60)
        if data.count(needle) != 1:
            raise ValueError('ambiguous layer metadata mutation')
        wrong_layers = args.out / 'wrong-layers.gguf'
        wrong_layers.write_bytes(data.replace(needle, struct.pack('<5i', 3, 17, 31, 45, 60)))
        run('reject-layers', 'dflash_loader', [args.model, wrong_layers], 'wrong target feature layers')
        needle = b'[PAD200054]'
        if data.count(needle) != 1:
            raise ValueError('ambiguous tokenizer mutation')
        wrong_vocab = args.out / 'wrong-vocab.gguf'
        wrong_vocab.write_bytes(data.replace(needle, b'[PAD200053]'))
        run('reject-vocab', 'dflash_loader', [args.model, wrong_vocab], 'tokens/merges differ')
        for i, draft in enumerate(gpu_drafts):
            dest = args.out / f'gpu-{i}'
            run(f'probe-{i}', 'dflash-check', [args.model, draft, dest])
            result = json.loads((dest / 'report.json').read_text(encoding='utf-8'))
            result['logits_sha256'] = {s.name: sha(s) for s in dest.glob('*.f32')}
            report['gpu_runs'].append(result)
        report['pass'] = all(v['pass'] for v in report['loaders']+report['negative_checks']+report['gpu_runs'])
    except Exception as exc:
        report['error'] = str(exc)
        raise
    finally:
        save(args.out / 'suite-report.json', report)
    return 0 if report['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
