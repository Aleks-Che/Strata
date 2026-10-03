"""Bounded GPU-only benchmark: prefill, decode, cached repeat and CPU/GPU samples.

Run while other Strata engines are stopped. OS file cache is not cleared.
python tools/benchmark_deepseek4.py --config strata-deepseek4-ud-q3-k-xl.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
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
    ap.add_argument('--timeout', type=int, default=240, help='Seconds per request, including prefill')
    ap.add_argument('--output', type=Path, default=ROOT / 'bench/results' / f'deepseek4-{time.strftime("%Y%m%d-%H%M%S")}.json')
    args = ap.parse_args()
    if args.prompt_tokens < 64 or args.tokens < 1 or args.timeout < 30:
        ap.error('Require prompt-tokens >= 64, tokens >= 1 and timeout >= 30')
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    if cfg.get('architecture') != 'deepseek4':
        ap.error('This benchmark requires a DeepSeek profile')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log = args.output.with_suffix('.log')
    tok = Tokenizer.from_gguf(cfg['args'][cfg['args'].index('--native') + 1])
    template = DeepSeekTemplate(Path(cfg['tokenizer']) / 'chat_template.jinja')
    result = {'config': cfg, 'os_file_cache_cleared': False, 'samples': [], 'requests': {}}
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
                    if child.name() not in ('strata-deepseek4.exe', 'strata-deepseek4'):
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
        for name, session, prompt, count in (
            ('prefill', 'bench-context', prefill, 8),
            ('decode', 'bench-generation', generation, args.tokens),
            ('repeat', 'bench-generation', generation, args.tokens),
        ):
            phase, deadline = name, time.monotonic() + args.timeout
            ids = tok.encode(template.render([{'role': 'user', 'content': prompt}], enable_thinking=False), parse_special=True)
            if name == 'prefill':
                if len(ids) < args.prompt_tokens:
                    raise ValueError('Not enough tokens in generated benchmark prompt')
                ids = ids[:args.prompt_tokens - 16] + ids[-16:]
            if len(ids) + count + 8 > engine.max_context:
                raise ValueError('Prompt plus output exceeds configured context')
            started, output, cancel = time.monotonic(), [], threading.Event()
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
            entry['decode_tokens_per_second'] = timings.get('generated', 0) * 1000 / max(timings.get('decode_ms', 0), 0.001)
            result['requests'][name] = entry
            print(name, json.dumps({k: v for k, v in entry.items() if k not in ('text', 'token_ids')}, ensure_ascii=True), flush=True)
            if timings.get('finish') == 'cancel':
                raise TimeoutError(f'{name} exceeded benchmark time limit')
        result['repeat_identical'] = result['requests']['decode']['token_ids'] == result['requests']['repeat']['token_ids']
        if not result['repeat_identical']:
            raise RuntimeError('Greedy cached repeat differs from original output')
    finally:
        if engine:
            engine.close()
        done.set()
        watcher.join(timeout=8)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(f'Results: {args.output}', flush=True)


if __name__ == '__main__':
    main()
