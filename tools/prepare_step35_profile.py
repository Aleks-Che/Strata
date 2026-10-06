"""Create an isolated experimental Step HTTP profile in a NEW directory.

Does not select a default model, edit existing profiles or start a server.
Compute settings follow STEP-08/09; STEP-10 RAM checks set the cache cap.
The explicit optimized profile adds STEP-18 native allocation reuse/batching.
This is not a hardware autotuner.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.setup_step35 import inspect_model
from tools.strata_tokenizer import Tokenizer, extract
from tools.step35_loader_contract import LOADER_SHA
from serve.server import configured_template

PATCH_SET = 'cuda-f32-mmf-respect-tf32-override,cuda-routed-input-strides,step-sync-selected-copy-gpu-audit-mmap-demand,step-bounded-expert-cache,step-router-lookahead-pipeline,step-prefill-cache-admission'
FINGERPRINT = '37305e11f45387f044806d639588fe6ef9d4b10eeaad1dc8ca0f02f9cdae6502'


def load_tokenizer(path):
    path = Path(path)
    vocab = json.loads((path / 'vocab.json').read_text(encoding='utf8'))
    tokens = [None] * len(vocab)
    for token, token_id in vocab.items():
        if type(token_id) is not int or not 0 <= token_id < len(tokens) or tokens[token_id] is not None:
            raise ValueError('Invalid exported Step vocabulary IDs')
        tokens[token_id] = token
    config = json.loads((path / 'tokenizer.json').read_text(encoding='utf8'))
    return Tokenizer(tokens, (path / 'merges.txt').read_text(encoding='utf8').split('\n'),
                     json.loads((path / 'token_type.json').read_text(encoding='utf8')),
                     pre=config['pre'], special_ids=config['special_ids'])


def prepare(model, engine, directory, cuda_dir=None, optimized=False):
    engine, directory = Path(engine).resolve(), Path(directory).resolve()
    if directory.exists():
        raise ValueError('Destination exists; choose a new directory for the experimental profile')
    report = inspect_model(model)
    if report['structural_fingerprint_sha256'] != FINGERPRINT:
        raise ValueError('This experimental profile requires the reviewed Step UD-Q4_K_S shards')
    dirs = [str((Path(cuda_dir) / sub).resolve()) for sub in ('bin', 'bin/x64')
            if (Path(cuda_dir) / sub).is_dir()] if cuda_dir else []
    env = dict(os.environ)
    if dirs:
        key = 'PATH' if os.name == 'nt' else 'LD_LIBRARY_PATH'
        env[key] = os.pathsep.join(dirs + [env.get(key, '')])
    identity = json.loads(subprocess.run([str(engine), '--version'], env=env, check=True,
        capture_output=True, text=True, timeout=30, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)).stdout)
    expected = {'architecture': 'step35', 'engine': 'step35-native', 'protocol_version': 1,
                'source_sha': LOADER_SHA, 'patch_set': PATCH_SET}
    if identity != expected:
        raise ValueError('Expected the reviewed Step backend, source, patch set and pipe protocol v1')
    directory.mkdir(parents=True, exist_ok=False)
    extract(report['first_shard'], directory)
    tokenizer = load_tokenizer(directory / 'tokenizer')
    template = configured_template({'architecture': 'step35'}, tokenizer, directory / 'tokenizer')
    cfg = {'architecture': 'step35', 'model_name': 'step-3.7-flash', 'experimental': True,
           'exe': str(engine), 'cwd': str(ROOT), 'host': '127.0.0.1', 'gpu': 0,
           'tokenizer': str(directory / 'tokenizer'), 'lib_dirs': dirs,
           'log': str(directory / 'engine.log'), 'statistics_file': str(directory / 'usage.sqlite'),
           'args': ['--native', report['first_shard'], '--max-context', '4096', '--batch-size', '17',
                    '--copy-mode', 'pinned', '--kv', 'f32', '--expert-cache-mib', '8192',
                    '--expert-pipeline-readers', '1', '--expert-cache-prefill', 'off'],
           'sampling': {'temperature': 0}, 'reasoning_budget_tokens': 0,
           'backend_identity': identity, 'engine_sha256': hashlib.sha256(engine.read_bytes()).hexdigest(),
           'model_structural_fingerprint': report['structural_fingerprint_sha256'],
           'validated_eog_ids': sorted(template.resolve_stop_ids(tokenizer))}
    if optimized:
        cfg['args'] += ['--expert-cache-reuse', 'on', '--expert-pipeline-batch', 'on']
        cfg['experimental_optimizations'] = ['cache-reuse', 'pipeline-batch']
        cfg['fit_max_tokens'] = True
    profile = directory / 'step37.json'
    with profile.open('x', encoding='utf8') as stream:
        json.dump(cfg, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'engine', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--cuda-dir', type=Path)
    parser.add_argument('--optimized', action='store_true', help='requires the STEP-18 native engine; enables allocation reuse and batched copies, MTP off')
    args = parser.parse_args()
    try:
        profile = prepare(args.model, args.engine, args.output_dir, args.cuda_dir, optimized=args.optimized)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        parser.exit(1, f'Step profile preparation failed: {exc}\n')
    print(profile)
    print(f'Launch explicitly: python -m serve.server --engine strata --config "{profile}" --port 8093')


if __name__ == '__main__':
    main()
