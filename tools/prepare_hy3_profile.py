"""Export an isolated Hy3 HTTP profile into a NEW directory; no default changes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.inspect_hy3_gguf import inspect_model
from tools.strata_tokenizer import Tokenizer, extract
from tools.hy3_loader_contract import LOADER_SHA
from serve.server import configured_template

HEADER_SHA = 'f3307357f0b6ab163f188f7d19ba57d2960ca70a7491b511b23b2745a9500ca9'
PATCH_SET = 'hy3-mtp-load-flags,cuda-f32-mmf-respect-tf32-override,cuda-routed-input-strides,hy3-sync-selected-file-copy-gpu-audit-demand-mmap'


def load_tokenizer(path):
    path = Path(path)
    vocab = json.loads((path/'vocab.json').read_text(encoding='utf8'))
    tokens = [None]*len(vocab)
    for token, token_id in vocab.items():
        if type(token_id) is not int or not 0 <= token_id < len(tokens) or tokens[token_id] is not None:
            raise ValueError('invalid exported Hy3 vocabulary IDs')
        tokens[token_id] = token
    config = json.loads((path/'tokenizer.json').read_text(encoding='utf8'))
    return Tokenizer(tokens, (path/'merges.txt').read_text(encoding='utf8').split('\n'),
        json.loads((path/'token_type.json').read_text(encoding='utf8')),
        pre=config['pre'], special_ids=config['special_ids'])


def prepare(model, engine, directory, cuda_dir=None):
    model, engine, directory = Path(model).resolve(), Path(engine).resolve(), Path(directory).resolve()
    if directory.exists():
        raise ValueError('Destination exists; choose a new directory for the Hy3 profile')
    report = inspect_model(model)
    if report['header_sha256'] != HEADER_SHA:
        raise ValueError('This profile requires the reviewed Hy3-Q3_K_M-mtp header')
    dirs = [str((Path(cuda_dir)/sub).resolve()) for sub in ('bin', 'bin/x64')
            if (Path(cuda_dir)/sub).is_dir()] if cuda_dir else []
    env = dict(os.environ)
    if dirs:
        env['PATH'] = os.pathsep.join(dirs+[env.get('PATH', '')])
    identity = json.loads(subprocess.run([str(engine), '--version'], env=env, check=True,
        capture_output=True, text=True, timeout=30, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)).stdout)
    expected = {'architecture': 'hy_v3', 'engine': 'hy3-native', 'protocol_version': 1,
                'source_sha': LOADER_SHA, 'patch_set': PATCH_SET}
    if identity != expected:
        raise ValueError('Expected the reviewed Hy3 engine, source, patches and pipe protocol')
    directory.mkdir(parents=True, exist_ok=False)
    extract(model, directory)
    tokenizer = load_tokenizer(directory/'tokenizer')
    template = configured_template({'architecture': 'hy_v3'}, tokenizer, directory/'tokenizer')
    cfg = {'architecture': 'hy_v3', 'model_name': 'hy3', 'experimental': True,
        'exe': str(engine), 'cwd': str(ROOT), 'host': '127.0.0.1', 'gpu': 0,
        'tokenizer': str(directory/'tokenizer'), 'lib_dirs': dirs,
        'log': str(directory/'engine.log'), 'statistics_file': str(directory/'usage.sqlite'),
        'args': ['--native', str(model), '--max-context', '2048', '--batch-size', '17',
                 '--copy-mode', 'pinned', '--kv', 'f32'],
        'sampling': {'temperature': 0}, 'reasoning_budget_tokens': 0,
        'fit_max_tokens': True, 'backend_identity': identity,
        'engine_sha256': hashlib.sha256(engine.read_bytes()).hexdigest(), 'model_header_sha256': report['header_sha256'],
        'validated_eog_ids': sorted(template.resolve_stop_ids(tokenizer))}
    profile = directory/'hy3.json'
    with profile.open('x', encoding='utf8') as stream:
        json.dump(cfg, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('model', 'engine', 'output-dir'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--cuda-dir', type=Path)
    args = parser.parse_args()
    try:
        profile = prepare(args.model, args.engine, args.output_dir, args.cuda_dir)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        parser.exit(1, f'Hy3 profile preparation failed: {error}\n')
    print(profile)
    print(f'Launch explicitly: python -m serve.server --engine strata --config "{profile}" --port 8094')


if __name__ == '__main__':
    main()
