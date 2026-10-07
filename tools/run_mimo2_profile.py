"""Console launcher for the pinned MiMo MIMO-17 Q4 MTP measurement."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time
from uuid import uuid4

from .check_mimo2_cuda import check_ceiling, gpu_memory, memory, sha
from .check_mimo2_engine import Engine
from .gguf_reader import GGUFFile
from .inspect_mimo2_drafts import inspect
from .mimo2_template import renderer
from .mimo2_tokenizer import from_gguf
from .summarize_mimo2_speculative import metrics


def resolve(base, value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_profile(path):
    profile = json.loads(path.read_text(encoding='utf8'))
    if profile.get('schema') != 1 or profile.get('memory_ceiling') != .95:
        raise ValueError('Expected profile schema1 and memory ceiling0.95')
    if not 1 <= profile['expert_cache_mib'] <= 14336 or profile['target_head_columns'] != 1:
        raise ValueError('Invalid MIMO-17 cache/head parameters')
    if profile['depth'] != 1 or profile['p_min'] != .7:
        raise ValueError('This launcher reproduces the measured depth1/p_min0.7 profile')
    paths = {key: resolve(path.parent, profile[key]) for key in ('model', 'draft', 'build', 'cuda_bin', 'benchmark_requests')}
    for key in ('model', 'draft', 'benchmark_requests'):
        if not paths[key].is_file():
            raise FileNotFoundError(f'{key}: {paths[key]}')
    if sha(paths['benchmark_requests']) != profile['benchmark_requests_sha256']:
        raise ValueError('The saved MIMO-17 benchmark requests changed')
    if not paths['cuda_bin'].is_dir():
        raise FileNotFoundError(f"CUDA DLL directory: {paths['cuda_bin']}")
    paths['binary'] = paths['build'] / 'bin/strata-mimo2-spec-check.exe'
    manifest = paths['build'] / 'mimo2-build-manifest.json'
    for file, expected in [(paths['binary'], profile['binary_sha256']), (manifest, profile['manifest_sha256'])]:
        if not file.is_file() or sha(file) != expected:
            raise ValueError(f'MIMO-17 snapshot missing or changed: {file}; no automatic substitution')
    metadata = json.loads(manifest.read_text(encoding='utf8'))
    if not metadata.get('cuda') or not metadata.get('speculative_probe'):
        raise ValueError('Expected the saved CUDA speculative probe')
    inspection = inspect(paths['model'], [paths['draft']])
    if inspection['drafts'][0]['contract']['kind'] != 'mtp':
        raise ValueError('Expected a compatible MTP sidecar')
    return profile, paths


def validate_request(request):
    count = request['predict']
    if type(count) is not int or not 1 <= count <= 128:
        raise ValueError('Число выходных токенов должно быть от1 до128.')
    if not request['tokens'] or len(request['tokens']) + count + 8 > 480:
        raise ValueError(f"Слишком длинный запрос: {len(request['tokens'])} токенов. "
                         f'При ответе до{count} допустимо не более{472-count}. Сократите текст или --max-tokens.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, default=Path(__file__).resolve().parents[1] / 'strata-mimo26-q4-mtp-10_93.json')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prompt', help='One independent prompt, then exit')
    mode.add_argument('--prompt-file', type=Path, help='UTF-8 text prompt, then exit')
    mode.add_argument('--benchmark', action='store_true', help='The saved 15 requests: 2 warmups +3 timed repeats per topic')
    mode.add_argument('--check', action='store_true', help='Check paths, snapshot SHA256 and GGUF contracts without loading weights')
    parser.add_argument('--max-tokens', type=int, help='Console output limit1..128; default128. Benchmark always32.')
    parser.add_argument('--timeout', type=int, default=1800, help='Load or individual request timeout, seconds')
    args = parser.parse_args()
    if args.timeout <= 0 or (args.max_tokens is not None and not 1 <= args.max_tokens <= 128):
        parser.error('Positive timeout and max-tokens1..128 required')
    if args.benchmark and args.max_tokens is not None:
        parser.error('--benchmark has fixed32-token requests; omit --max-tokens')
    engine = None
    report = None
    output = None
    try:
        profile, paths = load_profile(args.profile.resolve())
        count = args.max_tokens if args.max_tokens is not None else profile['default_max_tokens']
        if not 1 <= count <= 128:
            raise ValueError('Invalid default_max_tokens')
        print(profile['name'], flush=True)
        print('Профиль замера10,93 ток/с: Q4 MTP, depth1, cutoff0.7, cache до14ГиБ, RAM/VRAM до95%.', flush=True)
        print('Экспериментальная сборка MIMO-17: известное расхождение batch2 сохраняется. Скорость зависит от запроса и нагрузки.', flush=True)
        if args.check:
            print('OK: модель, Q4 MTP, CUDA directory, binary/manifest SHA256 и GGUF contracts проверены. Веса не загружались.')
            return 0
        tokenizer = from_gguf(paths['model'])
        template = renderer(GGUFFile(paths['model']).metadata['tokenizer.chat_template'])

        def request_for(text, name):
            prompt = template.render(messages=[dict(role='user', content=text)], enable_thinking=False, add_generation_prompt=True)
            request = dict(name=name, tokens=tokenizer.encode(prompt, True), predict=count,
                           depth=profile['depth'], p_min=profile['p_min'], warmup=False)
            validate_request(request)
            return request

        pending = None
        if args.benchmark:
            pending = json.loads(paths['benchmark_requests'].read_text(encoding='utf8'))
            if len(pending) != 15 or sum(not r['warmup'] for r in pending) != 9:
                raise ValueError('Expected the saved15-request benchmark with9 timed requests')
            for request in pending:
                validate_request(request)
                if request['predict'] != 32 or request['depth'] != 1 or request['p_min'] != .7:
                    raise ValueError('Benchmark request parameters changed')
        elif args.prompt is not None or args.prompt_file is not None:
            text = args.prompt if args.prompt is not None else args.prompt_file.read_text(encoding='utf-8-sig')
            pending = [request_for(text, 'prompt')]
        before = memory(gpu_memory())
        check_ceiling(before)
        root = Path(__file__).resolve().parents[1]
        output = root / 'build-local/mimo2-runs' / (datetime.now().strftime('%Y%m%d-%H%M%S-') + uuid4().hex[:8])
        output.mkdir(parents=True, exist_ok=False)
        command = [str(paths['binary']), '--model', str(paths['model']), '--kind', 'mtp', '--draft', str(paths['draft']),
                   '--expert-cache-mib', str(profile['expert_cache_mib']), '--target-head-columns', '1', '--memory-stages', '0']
        env = {k: v for k, v in os.environ.items() if not k.startswith('STRATA_MIMO_')}
        env.update(profile['environment'])
        env['PATH'] = str(paths['cuda_bin']) + os.pathsep + env.get('PATH', '')
        report = dict(status='running', profile=profile, command=command, before=before,
                      mode='benchmark' if args.benchmark else 'console', requests=[], samples=[])
        print(f'Логи: {output}\nЗагрузка модели и прогрев GPU…', flush=True)
        engine = Engine(command, env, output, args.timeout)
        if engine.read() != 'READY':
            raise RuntimeError('Unexpected engine startup response; see stdout.log/stderr.log')
        report['load_and_workspace_warmup_seconds'] = time.monotonic() - engine.start

        def generate(request):
            engine.timeout = time.monotonic() - engine.start + args.timeout
            message = dict(request)
            if args.benchmark:
                message['logits'] = str(output / f'{len(report["requests"])}.f32')
            engine.send(json.dumps(message))
            result = json.loads(engine.read())
            if result['rejected_cpu_nodes'] or result['pipeline_queued']:
                raise RuntimeError('GPU/drain audit failed')
            if (not result['target_head_columns'] or result['cache_request_mib'] != profile['expert_cache_mib'] or
                result['cache_slab_mib'] != 16 or result['cache_decay'] != 65536 or
                result['pipeline_batch'] != 1 or result['pipeline_packed_guards'] != 1 or
                result['pipeline_d2d_batch'] != 0 or result['cache_fill_batch'] or result['pipeline_early_host_refill']):
                raise RuntimeError('Engine settings differ from the measured profile')
            result['request'] = request
            visible = list(result['ids'])
            if visible and visible[-1] == 151645:
                visible.pop()
            result['text'] = tokenizer.decode(visible)
            report['requests'].append(result)
            print('\n' + result['text'], flush=True)
            prefix = 'Прогрев' if request.get('warmup') else 'Запрос'
            print(f"[{prefix}: {result['tokens_per_second']:.2f} ток/с decode; "
                  f"prefill {result['prefill_ms']/1000:.2f}с; MTP {result['accepted']}/{result['proposed']}]\n", flush=True)
            (output / 'run.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf8')

        if pending is not None:
            for request in pending:
                generate(request)
        else:
            print(f'Готово. Каждый запрос независимый, ответ до{count} токенов. Выход: /exit. Ответ появится целиком после расчёта.', flush=True)
            while True:
                try:
                    text = input('Вы> ')
                except EOFError:
                    break
                if text.strip().lower() in ('/exit', '/quit'):
                    break
                if not text.strip():
                    continue
                try:
                    request = request_for(text, f'prompt-{len(report["requests"])+1}')
                except ValueError as error:
                    print(error, flush=True)
                    continue
                generate(request)
        if args.benchmark:
            report['timed'] = metrics([r for r in report['requests'] if not r['request']['warmup']])
            print(f"Прогретая генерация: {report['timed']['tokens_per_second']:.3f} ток/с. Загрузка, prefill и первый токен исключены.")
        engine.close()
        if engine.process.returncode != 0:
            raise RuntimeError(f'Engine exited {engine.process.returncode}')
        report['status'] = 'complete'
        return 0
    except KeyboardInterrupt:
        if report is not None:
            report['status'] = 'interrupted'
        print('\nОстановка…', flush=True)
        return 130
    except Exception as error:
        if report is not None:
            report['status'], report['error'] = 'error', str(error)
        print(f'Ошибка: {error}', file=sys.stderr)
        return 1
    finally:
        if engine is not None:
            engine.close()
            report['samples'] = engine.samples
            report['exit_code'] = engine.process.returncode
        if report is not None:
            (output / 'run.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf8')


if __name__ == '__main__':
    sys.exit(main())
