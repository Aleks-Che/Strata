"""Fixed-input target-only cost diagnostic. No sampling or DSpark acceptance.

CUDA events require a STRATA_DEEPSEEK_COST_EXPERIMENT build; --events also
disables CUDA graphs. Event spans include stream stalls and host launch gaps.
Do not compare instrumented timings with normal engine throughput.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve.server import StrataEngine, child_env


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--tokens-json', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--events', action='store_true')
    ap.add_argument('--attention-stages', action='store_true', help='Also split private stable attention into prepare, MMA and fixup events')
    ap.add_argument('--expert-transport', action='store_true', help='Also record H2D, cache/ring D2D and host pipeline waits')
    ap.add_argument('--scheduler', action='store_true', help='Host scheduler timings; preserves CUDA graphs and adds no barriers')
    ap.add_argument('--timeout', type=int, default=600)
    args = ap.parse_args()
    if args.attention_stages and not args.events:
        ap.error('--attention-stages requires --events')
    if args.expert_transport and not args.events:
        ap.error('--expert-transport requires --events')
    cfg = json.loads(args.config.read_text('utf-8'))
    tokens = json.loads(args.tokens_json.read_text('utf-8'))
    if cfg.get('architecture') != 'deepseek4' or '--draft-model' in cfg['args']:
        ap.error('Requires a target-only DeepSeek profile')
    if not isinstance(tokens, list) or not 68 <= len(tokens) <= 4096 or any(type(x) is not int or x < 0 for x in tokens):
        ap.error('tokens-json must contain 68..4096 nonnegative integer IDs')
    if args.timeout < 30:
        ap.error('timeout must be >= 30')
    def setting(key, default):
        return int(cfg['args'][cfg['args'].index(key)+1]) if key in cfg['args'] else default
    if len(tokens) > setting('--batch-size', 512) or len(tokens)+9 > setting('--max-context', 8192):
        ap.error('Token workload must fit one batch and leave protocol context reserve')
    output = args.output.resolve()
    trace = output.with_suffix('.tsv')
    log = output.with_suffix('.log')
    stages = output.with_suffix('.attention.tsv')
    transport = output.with_suffix('.transport.tsv')
    scheduler = output.with_suffix('.scheduler.tsv')
    if any(p.exists() for p in [output, trace, log, stages, transport, scheduler]):
        ap.error('Choose fresh output paths')
    output.parent.mkdir(parents=True, exist_ok=True)
    env = child_env(cfg)
    if any(env.get(k) for k in ['STRATA_SPEC_TRACE_NODES', 'STRATA_PREFILL_PROBE']):
        ap.error('Disable activation and prefill observers for cost measurements')
    env['STRATA_FIXED_WORKLOAD'] = str(output)
    if args.events:
        env.update(STRATA_DS4_COST_PROFILE=str(trace), GGML_CUDA_DISABLE_GRAPHS='1')
    elif env.get('STRATA_DS4_COST_PROFILE'):
        ap.error('Use --events to enable event diagnostics explicitly')
    if args.attention_stages:
        if env.get('STRATA_DS4_FA_COMPACT') != '4':
            ap.error('--attention-stages requires private stable attention mode 4')
        env['STRATA_DS4_ATTENTION_STAGES'] = str(stages)
    elif env.get('STRATA_DS4_ATTENTION_STAGES'):
        ap.error('Use --attention-stages to enable attention event diagnostics explicitly')
    if args.expert_transport:
        env['STRATA_DS4_TRANSPORT_PROFILE'] = str(transport)
    elif env.get('STRATA_DS4_TRANSPORT_PROFILE'):
        ap.error('Use --expert-transport to enable transfer diagnostics explicitly')
    if args.scheduler:
        env['STRATA_DS4_SCHED_PROFILE'] = str(scheduler)
    elif env.get('STRATA_DS4_SCHED_PROFILE'):
        ap.error('Use --scheduler to enable host scheduler diagnostics explicitly')
    engine = None
    done = threading.Event()
    expired = threading.Event()
    samples = []
    start = time.monotonic()

    def watch():
        tracked = {}
        while not done.wait(2):
            sample = {'time': time.time(), 'system_cpu_percent': psutil.cpu_percent(),
                      'ram_available_bytes': psutil.virtual_memory().available}
            for child in psutil.Process().children(recursive=True):
                try:
                    if child.name().lower() != Path(cfg['exe']).name.lower():
                        continue
                    proc = tracked.setdefault(child.pid, child)
                    sample.update(pid=proc.pid, process_cpu_percent=proc.cpu_percent()/psutil.cpu_count())
                    if time.monotonic() - start > args.timeout:
                        expired.set(); proc.kill()
                except psutil.Error:
                    continue
            try:
                sample['gpu_util_percent,memory_mib'] = subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
                    text=True, timeout=5, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)).strip()
            except (OSError, subprocess.SubprocessError):
                pass
            samples.append(sample)

    watcher = threading.Thread(target=watch, daemon=True); watcher.start()
    try:
        engine = StrataEngine(cfg['exe'], cfg['args'], cwd=cfg['cwd'], log=str(log), env=env)
        if engine.info.get('gpu_only') != 1 or engine.info.get('speculative') != 'none':
            raise RuntimeError('Expected enforced GPU-only target, without draft')
        for _ in engine.generate(tokens, 1, {'temperature': 0}, threading.Event(), session_id='fixed-cost'):
            pass
        if expired.is_set():
            raise TimeoutError('Fixed workload exceeded time limit')
        if not output.exists():
            raise RuntimeError('Backend did not write a fixed workload report; rebuild the current DeepSeek backend')
        result = json.loads(output.read_text('utf-8'))
        if args.events and (not trace.exists() or trace.stat().st_size < 200):
            raise RuntimeError('No CUDA events: use a cost experiment build')
        if args.events:
            with trace.open('r', encoding='utf-8') as source:
                if not source.readline().startswith('graph\tphase\top\t'):
                    raise RuntimeError('Invalid CUDA event trace header')
        if args.attention_stages:
            if not stages.exists() or stages.stat().st_size < 200:
                raise RuntimeError('No attention stages: rebuild attention and cost experiments')
            with stages.open('r', encoding='utf-8') as source:
                if not source.readline().startswith('phase\tnode\tqueries\tkv\tbound\tblocks\tstage\t'):
                    raise RuntimeError('Invalid attention stage trace header')
        if args.expert_transport:
            if not transport.exists() or transport.stat().st_size < 200:
                raise RuntimeError('No expert transport events: rebuild the cost experiment')
            with transport.open('r', encoding='utf-8') as source:
                if source.readline() != 'phase\towner\tkind\tbytes\tms\n':
                    raise RuntimeError('Invalid expert transport trace header')
        if args.scheduler:
            if not scheduler.exists() or scheduler.stat().st_size < 200:
                raise RuntimeError('No host scheduler timings: rebuild the DeepSeek backend')
            with scheduler.open('r', encoding='utf-8') as source:
                if source.readline() != 'phase\tgraph\ttokens\tkind\tcalls\tbytes\tms\n':
                    raise RuntimeError('Invalid scheduler trace header')
        result.update(config=cfg, events=args.events, attention_stages=args.attention_stages,
                      scheduler=args.scheduler,
                      expert_transport=args.expert_transport, engine_info=engine.info,
                      elapsed_seconds=time.monotonic()-start, samples=samples,
                      executable_sha256=hashlib.sha256(Path(cfg['exe']).read_bytes()).hexdigest())
        output.write_text(json.dumps(result, indent=2)+'\n', 'utf-8')
        print(json.dumps({'output': str(output), 'runs': len(result['runs']), 'events': args.events}))
    finally:
        done.set(); watcher.join(timeout=8)
        if engine:
            engine.close()
        output.with_suffix('.samples.json').write_text(json.dumps(samples, indent=2)+'\n', 'utf-8')


if __name__ == '__main__':
    main()
