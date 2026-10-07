"""Run isolated synthetic MiMo checks with a global 95% RAM/VRAM ceiling.

No model argument: this tool never opens the user's full checkpoint. Each run
uses a new directory and records executable/manifest hashes and sampled memory.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gpu_memory():
    result = subprocess.run(
        ['nvidia-smi', '-i', '0', '--query-gpu=name,driver_version,memory.total,memory.used',
         '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True,
        timeout=10, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    name, driver, total, used = [p.strip() for p in result.stdout.strip().split(',')]
    return dict(name=name, driver=driver, total=int(float(total)*2**20), used=int(float(used)*2**20))


def memory(gpu):
    ram = psutil.virtual_memory()
    return dict(ram_total=ram.total, ram_available=ram.available, gpu=gpu)


def check_ceiling(sample):
    if sample['ram_available'] < sample['ram_total']*0.05 or sample['gpu']['used'] > sample['gpu']['total']*0.95:
        raise RuntimeError('95% global RAM/VRAM ceiling reached; only this check may be stopped')


def validate_result(result, manifest, kind, cache_fill_batch=False):
    for field, key in [('requested_revision', 'source_revision'), ('archive_sha256', 'archive_sha256'), ('patch_set', 'patches')]:
        if result.get(field) != manifest.get(key):
            raise ValueError('Binary result differs from build manifest: '+field)
    if result.get('status') == 'pass':
        cases = result.get('results', [])
        expected = {'kernels': 96, 'graph': 132, 'runtime': 244 if cache_fill_batch else 212}[kind]
        if result.get('case_count') != len(cases) or len(cases) != expected:
            raise ValueError('Incomplete MiMo numerical test coverage')
        if any(r.get('pass') is not True for r in cases):
            raise ValueError('A numerical case failed inside a passing report')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--kind', choices=['kernels', 'graph', 'runtime'], required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cuda-bin', type=Path, help='CUDA DLL directory to prepend to child PATH')
    parser.add_argument('--cache-slab-mib', type=int, choices=[0,16,32], default=0)
    parser.add_argument('--pipeline-batch', type=int, choices=[0,1], default=0)
    parser.add_argument('--cache-decay', type=int, choices=[0,16384,65536,131072], default=0)
    parser.add_argument('--cache-fill-batch', type=int, choices=[0,1], default=0)
    args = parser.parse_args()
    build = args.build.resolve()
    binary = build/'bin'/('strata-mimo2-'+args.kind+'-check'+('.exe' if os.name == 'nt' else ''))
    manifest = build/'mimo2-build-manifest.json'
    meta = json.loads(manifest.read_text(encoding='utf8'))
    if meta.get('cuda') is not True or (args.kind == 'runtime' and meta.get('runtime') is not True) or not binary.is_file():
        parser.error('compiled MiMo CUDA check and its manifest are required')
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    report = dict(status='error', kind=args.kind, scope='synthetic correctness, no inference throughput',
                  runner_command=[sys.executable, *sys.argv],
                  binary=str(binary), binary_sha256=sha(binary), manifest=meta,
                  manifest_sha256=sha(manifest), samples=[], sampling_seconds=0.5,
                  gpu_sampling_seconds=0.5, ceiling_fraction=0.95,
                  peak_scope='sampled system-wide usage includes other processes; not an allocation guarantee')
    env = os.environ.copy()
    env['STRATA_MIMO_CACHE_SLAB_MIB'] = str(args.cache_slab_mib)
    report['cache_slab_mib'] = args.cache_slab_mib
    env['STRATA_MIMO_PIPELINE_BATCH'] = str(args.pipeline_batch)
    report['pipeline_batch'] = args.pipeline_batch
    env['STRATA_MIMO_CACHE_DECAY'] = str(args.cache_decay)
    report['cache_decay'] = args.cache_decay
    env['STRATA_MIMO_CACHE_FILL_BATCH'] = str(args.cache_fill_batch)
    report['cache_fill_batch'] = args.cache_fill_batch
    env['NVIDIA_TF32_OVERRIDE'] = '0'
    env['GGML_CUDA_CUBLAS_COMPUTE_TYPE'] = 'f32'
    env['GGML_CUDA_DISABLE_GRAPHS'] = '1'
    env['LLAMA_GRAPH_REUSE_DISABLE'] = '1'
    if args.cuda_bin:
        env['PATH'] = str(args.cuda_bin.resolve())+os.pathsep+env.get('PATH', '')
    report['diagnostic_environment']={key:env.get(key) for key in
        ('GGML_CUDA_DISABLE_GRAPHS','GGML_CUDA_PDL','LLAMA_GRAPH_REUSE_DISABLE','NVIDIA_TF32_OVERRIDE','GGML_CUDA_CUBLAS_COMPUTE_TYPE')}
    result_path = directory/f'{args.kind}-report.json' if args.kind == 'kernels' else directory/'fixture'/f'{args.kind}-report.json'
    command = ([str(binary), '--output', str(result_path)] if args.kind == 'kernels' else
               [str(binary), str(directory/'fixture')])
    report['command'] = command
    child = None
    try:
        gpu = gpu_memory()
        first = memory(gpu)
        check_ceiling(first)
        report['before'] = first
        # Matrix fixtures allocate one small 256-expert tensor at a time;
        # graph fixtures hold several independent KV contexts concurrently.
        reserve = (512 if args.kind == 'kernels' else 1024)*2**20
        report['vram_admission_reserve_bytes'] = reserve
        if first['ram_available'] < 2*2**30 or first['gpu']['total']*.95-first['gpu']['used'] < reserve:
            raise RuntimeError('Insufficient RAM or VRAM reserve below the 95% global ceiling')
        with (directory/'stdout.log').open('wb') as stdout, (directory/'stderr.log').open('wb') as stderr:
            child = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env,
                                     creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            process = psutil.Process(child.pid)
            started = time.monotonic()
            while child.poll() is None:
                gpu = gpu_memory()
                sample = memory(gpu)
                sample['seconds'] = time.monotonic()-started
                try:
                    info = process.memory_info()
                    sample.update(process_rss=info.rss, process_peak_wset=getattr(info, 'peak_wset', None))
                except psutil.NoSuchProcess:
                    break
                report['samples'].append(sample)
                check_ceiling(sample)
                if sample['seconds'] > 600:
                    raise RuntimeError('synthetic check exceeded 600 seconds')
                time.sleep(0.5)
            report['exit_code'] = child.wait(timeout=10)
            report['wall_seconds'] = time.monotonic()-started
        if not result_path.is_file():
            raise RuntimeError(f"Native check exited with code {report['exit_code']} without a numerical report; see stderr.log")
        report['result'] = json.loads(result_path.read_text(encoding='utf8'))
        report['result_sha256'] = sha(result_path)
        validate_result(report['result'], meta, args.kind, bool(args.pipeline_batch and args.cache_fill_batch))
        report['status'] = 'pass' if report['exit_code'] == 0 and report['result']['status'] == 'pass' else 'fail'
    except Exception as error:
        report['error'] = str(error)
    finally:
        if child and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=10)
        (directory/'run-report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf8')
    print(report['status'], report.get('error', ''), directory/'run-report.json', flush=True)
    return int(report['status'] != 'pass')


if __name__ == '__main__':
    raise SystemExit(main())
