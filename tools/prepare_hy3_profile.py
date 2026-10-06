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
PATCH_SET = 'hy3-mtp-load-flags,cuda-f32-mmf-respect-tf32-override,cuda-routed-input-strides,hy3-sync-selected-file-copy-gpu-audit-demand-mmap,hy3-bounded-matrix-cache,hy3-bounded-pipeline,hy3-tensor-batch-copy,hy3-delivery-profile,hy3-disable-cuda-graphs,hy3-native-mtp,hy3-managed-ram-cache,hy3-frequency-ram-cache,hy3-gpu-prefill-policy'


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


def prepare(model, engine, directory, cuda_dir=None, cache_mib=0, pipeline_readers=0, pipeline_chunk_mib=4, pipeline_batch=False, mtp=0, ram_cache_mib=0, ram_cache_policy='frequency', gpu_cache_policy='all'):
    model, engine, directory = Path(model).resolve(), Path(engine).resolve(), Path(directory).resolve()
    if type(cache_mib) is not int or not 0 <= cache_mib <= 16384:
        raise ValueError('Hy3 expert cache must be 0..16384 MiB')
    if (type(pipeline_readers) is not int or not 0 <= pipeline_readers <= 2 or
            type(pipeline_chunk_mib) is not int or pipeline_chunk_mib not in (4,8,16) or
            (pipeline_readers and not cache_mib)):
        raise ValueError('Hy3 pipeline requires readers0..2, chunk4/8/16 MiB and nonzero cache cap')
    if type(pipeline_batch) is not bool or (pipeline_batch and not pipeline_readers):
        raise ValueError('Hy3 tensor batching requires a boolean option and pipeline readers')
    if type(mtp) is not int or not 0 <= mtp <= 3:
        raise ValueError('Hy3 MTP depth must be an integer 0..3')
    if ram_cache_mib != 'auto' and (type(ram_cache_mib) is not int or not 0 <= ram_cache_mib <= 1048576):
        raise ValueError('Hy3 RAM cache must be auto or an integer 0..1048576 MiB')
    if ram_cache_policy not in ('frequency','lru'):
        raise ValueError('Hy3 RAM cache policy must be frequency or lru')
    if gpu_cache_policy not in ('all','decode'):
        raise ValueError('Hy3 GPU cache policy must be all or decode')
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
    if gpu_cache_policy!='all':
        cfg['args'] += ['--gpu-cache-policy',gpu_cache_policy]
    if cache_mib:
        cfg['args'] += ['--expert-cache-mib', str(cache_mib)]
    if pipeline_readers:
        cfg['args'] += ['--pipeline-readers', str(pipeline_readers), '--pipeline-chunk-mib', str(pipeline_chunk_mib)]
    if pipeline_batch:
        cfg['args'] += ['--pipeline-batch', '1']
    if mtp:
        cfg['args'] += ['--mtp', str(mtp)]
    if ram_cache_mib:
        cfg['args'] += ['--ram-cache-policy', ram_cache_policy, '--ram-cache-mib', str(ram_cache_mib)]
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
    parser.add_argument('--expert-cache-mib', type=int, default=0, help='opt-in GPU cache cap (0..16384); runtime clamps to global free VRAM')
    parser.add_argument('--pipeline-readers', type=int, default=0, help='opt-in readers (1 or 2); requires nonzero expert cache')
    parser.add_argument('--pipeline-chunk-mib', type=int, default=4, choices=(4,8,16))
    parser.add_argument('--pipeline-batch', action='store_true', help='opt-in one delivery fence per expert tensor')
    parser.add_argument('--mtp', type=int, choices=(0,1,2,3), default=0, help='native resident MTP depth; active for greedy requests only')
    parser.add_argument('--ram-cache-mib', default='0', help='managed RAM matrix-chunk cache: auto or 0..1048576 MiB; dynamic 93% RAM target')
    parser.add_argument('--ram-cache-policy', choices=('frequency','lru'), default='frequency')
    parser.add_argument('--gpu-cache-policy', choices=('all','decode'), default='all')
    args = parser.parse_args()
    try:
        profile = prepare(args.model, args.engine, args.output_dir, args.cuda_dir, args.expert_cache_mib,
                          args.pipeline_readers, args.pipeline_chunk_mib, args.pipeline_batch, args.mtp,
                          'auto' if args.ram_cache_mib=='auto' else int(args.ram_cache_mib),args.ram_cache_policy,args.gpu_cache_policy)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        parser.exit(1, f'Hy3 profile preparation failed: {error}\n')
    print(profile)
    print(f'Launch explicitly: python -m serve.server --engine strata --config "{profile}" --port 8094')


if __name__ == '__main__':
    main()
