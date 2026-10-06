"""Run isolated synthetic Hy3 checks with a global 95% RAM/VRAM ceiling.

No model argument: this tool never opens the user's full checkpoint. Each run
uses a new directory and records executable/manifest hashes and sampled memory.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--kind', choices=['kernels', 'graph', 'runtime'], required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cuda-bin', type=Path, help='CUDA DLL directory to prepend to child PATH')
    args = parser.parse_args()
    build = args.build.resolve()
    binary = build/'bin'/('strata-hy3-'+args.kind+'-check'+('.exe' if os.name == 'nt' else ''))
    manifest = build/'hy3-build-manifest.json'
    meta = json.loads(manifest.read_text(encoding='utf8'))
    if not meta['cuda'] or not binary.is_file():
        parser.error('compiled Hy3 CUDA check and its manifest are required')
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    report = dict(status='error', kind=args.kind, scope='synthetic correctness, no inference throughput',
                  binary=str(binary), binary_sha256=sha(binary), manifest=meta,
                  manifest_sha256=sha(manifest), samples=[], sampling_seconds=0.5,
                  gpu_sampling_seconds=2, ceiling_fraction=0.95,
                  peak_scope='sampled system-wide usage includes other processes; not an allocation guarantee')
    env = os.environ.copy()
    env['NVIDIA_TF32_OVERRIDE'] = '0'
    if args.cuda_bin:
        env['PATH'] = str(args.cuda_bin.resolve())+os.pathsep+env.get('PATH', '')
    result_path = directory/'kernels-report.json' if args.kind == 'kernels' else directory/'fixture'/f'{args.kind}-report.json'
    command = [str(binary), '--output', str(result_path)] if args.kind == 'kernels' else [str(binary), str(directory/'fixture')]
    report['command'] = command
    child = None
    try:
        gpu = gpu_memory()
        first = memory(gpu)
        check_ceiling(first)
        report['before'] = first
        with (directory/'stdout.log').open('wb') as stdout, (directory/'stderr.log').open('wb') as stderr:
            child = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env,
                                     creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            process = psutil.Process(child.pid)
            started, index = time.monotonic(), 0
            while child.poll() is None:
                if index % 4 == 0:
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
                index += 1
            report['exit_code'] = child.wait(timeout=10)
            report['wall_seconds'] = time.monotonic()-started
        report['result'] = json.loads(result_path.read_text(encoding='utf8'))
        report['result_sha256'] = sha(result_path)
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
