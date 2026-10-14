"""Run an explicit DeepSeek request manifest; preserve inputs, output IDs and phase telemetry.

No OS cache purge. Physical disk counters include every process on the device.
Example: python tools/benchmark_deepseek4_stage5.py --config profile.json
         --requests requests.json --output results.json
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
import psutil

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from serve.server import StrataEngine,child_env

def expert_phase_counters(log: str, cfg: dict) -> dict:
    """Difference native counters, never infer a decode delta from process totals."""
    points = {}
    for line in log.splitlines():
        if not line.startswith(('STRATA_EXPERT_BOUNDARY ', 'STRATA_EXPERT_SNAPSHOT ', 'STRATA_EXPERT_TIMING ')):
            continue
        fields = dict(re.findall(r'(\w+)=([^\s]+)', line))
        phase = points.setdefault(int(fields['request']), {}).setdefault(fields['phase'], {})
        if 'backend' in fields:
            phase.setdefault(fields['backend'], {}).update({k: int(v) for k, v in fields.items() if k not in ('request', 'phase', 'backend')})
    def option(name, default):
        args = cfg['args']
        return int(args[args.index(name)+1]) if name in args else default
    target = option('--expert-cache-mib', 0)*2**20
    draft = option('--draft-expert-cache-mib', 0)*2**20
    result = {}
    for request, phases in points.items():
        deltas = {}
        for name, first, last in [('prefill', 'begin', 'prefill_end'), ('decode', 'prefill_end', 'end'), ('total', 'begin', 'end')]:
            if first not in phases or last not in phases:
                continue
            deltas[name] = {}
            for backend, after in phases[last].items():
                before = phases[first].get(backend, {})  # A newly created backend starts at zero.
                values = {k: v-before.get(k, 0) for k, v in after.items() if k != 'cache_bytes'}
                if any(v < 0 for v in values.values()):
                    raise ValueError('Counter reset inside a request: '+backend)
                capacity = after['cache_bytes']
                # Budget is only a label when unambiguous; retain backend IDs.
                role = 'target' if capacity == target and target != draft else 'draft' if capacity == draft and target != draft else 'unknown'
                deltas[name][backend] = {'role': role, 'cache_bytes': capacity, **values}
        result[str(request)] = {'snapshots': phases, 'deltas': deltas}
    return result

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--requests',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--timeout',type=int,default=600)
    ap.add_argument('--allow-cpu-lane',action='store_true')
    args=ap.parse_args()
    cfg=json.loads(args.config.read_text('utf-8'));cases=json.loads(args.requests.read_text('utf-8'))
    if cfg.get('architecture')!='deepseek4' or args.timeout<30:ap.error('DeepSeek config and timeout >=30 required')
    if args.output.exists():ap.error('Output exists; use a new artifact name')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result={'config':cfg,'exe_sha256':hashlib.sha256(Path(cfg['exe']).read_bytes()).hexdigest(),
        'os_file_cache_cleared':False,'physical_disk_scope':'entire devices, all processes',
        'requests':[],'samples':[],'logical_cpus':psutil.cpu_count(),'started_unix':time.time()}
    engine=None;proc=None;done=threading.Event();phase='loading';deadline=time.monotonic()+args.timeout
    def counters():
        cpu=proc.cpu_times();io=proc.io_counters()
        return {'time':time.monotonic(),'cpu_seconds':cpu.user+cpu.system,'process_read_bytes':io.read_bytes,
            'physical_reads':{k:v.read_bytes for k,v in psutil.disk_io_counters(perdisk=True).items()}}
    def delta(a,b):
        return {'seconds':b['time']-a['time'],'cpu_seconds':b['cpu_seconds']-a['cpu_seconds'],
            'process_read_bytes':b['process_read_bytes']-a['process_read_bytes'],
            'physical_read_bytes':{k:b['physical_reads'][k]-v for k,v in a['physical_reads'].items()}}
    def monitor():
        nonlocal proc
        while not done.wait(2):
            sample={'time':time.monotonic(),'phase':phase,'system_cpu_percent':psutil.cpu_percent(),
                'ram_available_bytes':psutil.virtual_memory().available}
            try:
                if proc is None and engine is not None:proc=psutil.Process(engine.proc.pid)
                if proc is not None:
                    mem=proc.memory_info();sample.update(process_cpu_percent=proc.cpu_percent()/psutil.cpu_count(),
                        rss_bytes=mem.rss,private_bytes=getattr(mem,'private',None))
                if time.monotonic()>deadline:
                    result['timeout']=phase
                    for child in psutil.Process().children(recursive=True):
                        if Path(child.exe()).resolve()==Path(cfg['exe']).resolve():child.kill()
                gpu=subprocess.check_output(['nvidia-smi',f'--id={cfg.get("gpu",0)}',
                    '--query-gpu=utilization.gpu,memory.used,temperature.gpu,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current','--format=csv,noheader,nounits'],
                    text=True,timeout=5,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)).strip().split(',')
                sample.update(gpu_percent=float(gpu[0]),vram_mib=float(gpu[1]),gpu_temperature=float(gpu[2]))
                sample.update(pcie_generation=gpu[3].strip(),pcie_max_generation=gpu[4].strip(),pcie_width=gpu[5].strip())
            except (psutil.Error,OSError,subprocess.SubprocessError,ValueError) as e:sample['error']=str(e)
            result['samples'].append(sample)
    watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
    try:
        env=child_env(cfg)
        # Avoid accidental experiment flags inherited from a developer shell.
        for k in list(env):
            if k.startswith(('STRATA_DS4_','STRATA_FIXED_','STRATA_PREFILL_','STRATA_SPEC_CHECK','STRATA_SPEC_SERIAL')) and k not in cfg.get('env',{}):del env[k]
        if 'GGML_CUDA_DISABLE_GRAPHS' not in cfg.get('env',{}):env.pop('GGML_CUDA_DISABLE_GRAPHS',None)
        start=time.monotonic();engine=StrataEngine(cfg['exe'],cfg['args'],cwd=cfg['cwd'],log=str(args.output.with_suffix('.log')),env=env)
        proc=psutil.Process(engine.proc.pid);proc.cpu_percent();result.update(load_seconds=time.monotonic()-start,engine_info=dict(engine.info))
        if engine.info.get('gpu_only')!=1 and not (args.allow_cpu_lane and engine.info.get('expert_compute')=='hybrid'):
            raise RuntimeError('Unexpected CPU execution policy')
        phase='idle_before';deadline=time.monotonic()+args.timeout;idle=counters();time.sleep(2);result['idle_before']=delta(idle,counters())
        for case in cases:
            ids=case['ids'];count=case['count'];name=case['name']
            if len(ids)+count+8>engine.max_context:raise ValueError('Request exceeds context: '+name)
            phase=name+'/prefill';deadline=time.monotonic()+args.timeout
            before=counters();prompt_end=None;first=None;output=[];cancel=threading.Event()
            for token in engine.generate(ids,count,{'temperature':0},cancel,session_id=case['session']):
                if prompt_end is None and engine.progress and engine.progress[0]>=len(ids)-1:
                    prompt_end=counters();phase=name+'/decode'
                if token is not None:
                    if first is None:first=counters();phase=name+'/decode'
                    output.append(token)
            after=counters();timing=dict(engine.last)
            row={**case,'output_ids':output,'timings':timing,'total':delta(before,after),
                'ttft_seconds':first['time']-before['time'] if first else None,
                'prefill_observed':delta(before,prompt_end) if prompt_end else None,
                'decode_observed':delta(prompt_end,after) if prompt_end else None,
                'decode_tokens_per_second':len(output)*1000/max(timing.get('decode_ms',0),.001),
                'vram_status':dict(engine.vram_status)}
            result['requests'].append(row)
            args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
            print(name,json.dumps({'wall':row['total']['seconds'],'prefill_ms':timing.get('prompt_ms'),'tps':row['decode_tokens_per_second'],
                'output':len(output),'reused':timing.get('reused'),
                'physical_read_gib':{k:v/2**30 for k,v in row['total']['physical_read_bytes'].items()}}),flush=True)
            if timing.get('finish')=='cancel':raise TimeoutError(name+' cancelled')
            if case.get('fresh') and timing.get('reused',0):raise RuntimeError('Unexpected KV reuse: '+name)
        phase='idle_after';idle=counters();time.sleep(2);result['idle_after']=delta(idle,counters())
    except BaseException as e:
        result['error']=repr(e);raise
    finally:
        done.set();watcher.join(timeout=8)
        if engine is not None:engine.close()
        log_path=args.output.with_suffix('.log')
        if log_path.exists():
            result['expert_phases']=expert_phase_counters(log_path.read_text(encoding='utf-8',errors='replace'),cfg)
        args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':main()
