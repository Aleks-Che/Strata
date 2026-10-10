"""Bounded GPU-only benchmark: prefill, decode, cached repeat and CPU/GPU samples.

Run while other Strata engines are stopped. OS file cache is not cleared.
python tools/benchmark_deepseek4.py --config strata-deepseek4-ud-q3-k-xl.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import statistics
import sys
import threading
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve.deepseek import DeepSeekTemplate
from serve.server import StrataEngine, child_env
from tools.strata_tokenizer import Tokenizer


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--prompt-tokens', type=int, default=4096)
    ap.add_argument('--tokens', type=int, default=128)
    ap.add_argument('--generation-prompt', type=Path, help='UTF-8 file with the generation prompt')
    ap.add_argument('--alternate-generation-prompt', type=Path, help='After normal repeats, alternate another prompt with the original; separate RAM sessions, shared expert cache')
    ap.add_argument('--switch-rounds', type=int, default=2, help='Number of alternate/original pairs when an alternate prompt is supplied (2..5)')
    ap.add_argument('--repeats', type=int, default=1, help='Number of warm repeats of the same prompt')
    ap.add_argument('--prefill-repeats', type=int, default=0, help='Repeat full prefill in fresh sessions after decode; preserves expert cache, never restores KV')
    ap.add_argument('--timeout', type=int, default=240, help='Seconds per request, including prefill')
    ap.add_argument('--physical-disk', help='psutil disk name, e.g. PhysicalDrive6; records device-wide physical reads, including other processes')
    ap.add_argument('--output', type=Path, default=ROOT / 'bench/results' / f'deepseek4-{time.strftime("%Y%m%d-%H%M%S")}.json')
    args = ap.parse_args()
    if args.prompt_tokens < 64 or args.tokens < 1 or args.timeout < 30 or not 1 <= args.repeats <= 10 or not 0 <= args.prefill_repeats <= 10:
        ap.error('Require prompt-tokens >= 64, tokens >= 1, timeout >= 30, repeats in [1, 10], prefill-repeats in [0, 10]')
    if not 2 <= args.switch_rounds <= 5:
        ap.error('switch-rounds must be in [2, 5]')
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    if cfg.get('architecture') != 'deepseek4':
        ap.error('This benchmark requires a DeepSeek profile')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log = args.output.with_suffix('.log')
    tok = Tokenizer.from_gguf(cfg['args'][cfg['args'].index('--native') + 1])
    template = DeepSeekTemplate(Path(cfg['tokenizer']) / 'chat_template.jinja')
    result = {'config': cfg, 'os_file_cache_cleared': False, 'samples': [], 'requests': {}}
    def disk_counters():
        if not args.physical_disk:
            return None
        counters = psutil.disk_io_counters(perdisk=True).get(args.physical_disk)
        if counters is None:
            raise ValueError(f'Physical disk counter unavailable: {args.physical_disk}')
        return counters
    disk_counters()
    result['physical_disk'] = args.physical_disk
    result['physical_disk_scope'] = 'entire device, all processes' if args.physical_disk else None
    done = threading.Event()
    phase, deadline = 'loading', time.monotonic() + args.timeout
    engine = None

    def monitor():
        tracked = {}
        while not done.wait(2):
            sample = {'phase': phase, 'time': time.time(),
                      'ram_available_bytes': psutil.virtual_memory().available,
                      'system_cpu_percent': psutil.cpu_percent()}
            for child in psutil.Process().children(recursive=True):
                try:
                    if child.name().lower() != Path(cfg['exe']).name.lower():
                        continue
                    proc = tracked.setdefault(child.pid, child)
                    sample.update(pid=proc.pid, cpu_percent=proc.cpu_percent() / psutil.cpu_count(),
                                  rss_bytes=proc.memory_info().rss)
                    if time.monotonic() > deadline:
                        result['timeout'] = phase
                        proc.kill()
                except psutil.Error:
                    continue
            try:
                sample['gpu_util_percent,memory_mib'] = subprocess.check_output(
                    ['nvidia-smi', f'--id={cfg.get("gpu", 0)}', '--query-gpu=utilization.gpu,memory.used',
                     '--format=csv,noheader,nounits'], text=True, timeout=5,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)).strip()
            except (OSError, subprocess.SubprocessError):
                pass
            result['samples'].append(sample)

    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    try:
        started = time.monotonic()
        engine = StrataEngine(cfg['exe'], cfg['args'], cwd=cfg['cwd'], log=str(log), env=child_env(cfg))
        result['load_seconds'] = time.monotonic() - started
        result['engine_info'] = dict(engine.info)
        if engine.info.get('gpu_only') != 1:
            raise RuntimeError('The backend did not declare enforced GPU-only execution')
        if args.prompt_tokens + 16 > engine.max_context:
            raise ValueError('Prefill prompt exceeds configured context')
        prefill = ' '.join(f'Запись {i}: синий квадрат.' for i in range(args.prompt_tokens // 8 + 1))
        prefill += '\nОтветь одним словом: какого цвета квадраты?'
        generation = 'Напиши подробное объяснение на русском: как устроен кэш процессора и чем L1 отличается от L2 и L3.'
        if args.generation_prompt:
            generation = args.generation_prompt.read_text(encoding='utf-8')
        result['generation_prompt'] = generation
        alternate = args.alternate_generation_prompt.read_text(encoding='utf-8') if args.alternate_generation_prompt else None
        result['alternate_generation_prompt'] = alternate
        requests = [
            ('prefill', 'bench-context', prefill, 8),
            ('decode', 'bench-generation', generation, args.tokens),
        ] + [('repeat' if i==0 else f'repeat_{i+1}', 'bench-generation', generation, args.tokens)
             for i in range(args.repeats)]
        if alternate is not None:
            for i in range(args.switch_rounds):
                requests += [(f'switch_{i+1}_alternate', 'bench-alternate', alternate, args.tokens),
                             (f'switch_{i+1}_original', 'bench-generation', generation, args.tokens)]
        requests += [(f'full_prefill_{i+1}', f'bench-fresh-prefill-{i+1}', prefill, 8)
                     for i in range(args.prefill_repeats)]
        for name, session, prompt, count in requests:
            phase, deadline = name, time.monotonic() + args.timeout
            ids = tok.encode(template.render([{'role': 'user', 'content': prompt}], enable_thinking=False), parse_special=True)
            if name == 'prefill' or name.startswith('full_prefill_'):
                if len(ids) < args.prompt_tokens:
                    raise ValueError('Not enough tokens in generated benchmark prompt')
                ids = ids[:args.prompt_tokens - 16] + ids[-16:]
            if len(ids) + count + 8 > engine.max_context:
                raise ValueError('Prompt plus output exceeds configured context')
            started, output, cancel = time.monotonic(), [], threading.Event()
            cpu_before = psutil.Process(engine.proc.pid).cpu_times()
            disk_before = disk_counters()
            timer = threading.Timer(args.timeout - 10, cancel.set)
            timer.start()
            try:
                for token in engine.generate(ids, count, {'temperature': 0}, cancel, session_id=session):
                    if token is not None:
                        output.append(token)
            finally:
                timer.cancel()
            timings = dict(engine.last)
            entry = {'elapsed_seconds': time.monotonic() - started, 'timings': timings,
                     'token_ids': output, 'text': tok.decode(output)}
            cpu_after = psutil.Process(engine.proc.pid).cpu_times()
            entry['process_cpu_seconds'] = cpu_after.user + cpu_after.system - cpu_before.user - cpu_before.system
            if name.startswith('full_prefill_') and timings.get('reused', 0) != 0:
                raise RuntimeError('Full prefill unexpectedly reused sequence state')
            entry['vram_status'] = dict(getattr(engine, 'vram_status', {}) or {})
            if disk_before is not None:
                disk_after = disk_counters()
                entry['physical_read_bytes'] = disk_after.read_bytes - disk_before.read_bytes
                entry['physical_read_operations'] = disk_after.read_count - disk_before.read_count
            entry['decode_tokens_per_second'] = timings.get('generated', 0) * 1000 / max(timings.get('decode_ms', 0), 0.001)
            result['requests'][name] = entry
            print(name, json.dumps({k: v for k, v in entry.items() if k not in ('text', 'token_ids', 'vram_status')}, ensure_ascii=True), flush=True)
            if timings.get('finish') == 'cancel':
                raise TimeoutError(f'{name} exceeded benchmark time limit')
        repeats = [entry for name, entry in result['requests'].items() if name.startswith('repeat')]
        result['warm_median_tokens_per_second'] = statistics.median(entry['decode_tokens_per_second'] for entry in repeats)
        result['repeat_identical'] = all(result['requests']['decode']['token_ids'] == entry['token_ids'] for entry in repeats)
        if not result['repeat_identical']:
            raise RuntimeError('Greedy cached repeat differs from original output')
        if alternate is not None:
            switched = result['requests']
            for i in range(args.switch_rounds):
                if switched[f'switch_{i+1}_original']['token_ids'] != switched['decode']['token_ids']:
                    raise RuntimeError('Switching prompts changed original greedy output')
                if switched[f'switch_{i+1}_alternate']['token_ids'] != switched['switch_1_alternate']['token_ids']:
                    raise RuntimeError('Switching prompts changed alternate greedy output')
            result['switch_identical'] = True
        if args.prefill_repeats:
            full = [entry for name, entry in result['requests'].items() if name.startswith('full_prefill_')]
            result['full_prefill_median_ms'] = statistics.median(entry['timings']['prompt_ms'] for entry in full)
            if any(entry['token_ids'] != result['requests']['prefill']['token_ids'] for entry in full):
                raise RuntimeError('Repeated full prefill output differs')
    finally:
        # End sampling while weights/cache are still resident; otherwise a
        # teardown sample is incorrectly attributed to the final request.
        done.set()
        watcher.join(timeout=8)
        if engine:
            engine.close()
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(f'Results: {args.output}', flush=True)


if __name__ == '__main__':
    main()
