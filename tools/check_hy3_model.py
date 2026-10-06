"""Full Hy3 MTP-off baseline: native vs pinned-file exact logits/greedy IDs.

New output directory, existing read-only GGUF, global 95% memory monitor.
Timings are sequential observations with OS caching, not a paired speed claim.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from check_hy3_engine import Engine
from check_hy3_template import renderer, TEMPLATE_SHA
from check_hy3_cuda import gpu_memory, memory, check_ceiling
from check_step35_model import Monitor
from gguf_reader import GGUFFile
from inspect_hy3_gguf import inspect_model
from strata_tokenizer import Tokenizer


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(engine):
    return [json.loads(line.split(' ', 1)[1]) for line in
            engine.stderr_path.read_text(encoding='utf8', errors='replace').splitlines()
            if line.startswith('STRATA_HY3_REQUEST ')]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine', type=Path, required=True)
    p.add_argument('--gguf', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--cuda-bin', type=Path, required=True)
    p.add_argument('--predict', type=int, default=16)
    p.add_argument('--smoke', action='store_true', help='one short request per mode, no lifecycle/long corpus')
    args = p.parse_args()
    if not 2 <= args.predict <= 64:
        p.error('predict must be 2..64')
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    model, binary = args.gguf.resolve(), args.engine.resolve()
    inventory = inspect_model(model)
    if inventory['metadata']['tokenizer.chat_template']['sha256'] != TEMPLATE_SHA:
        raise ValueError('unreviewed chat template')
    (directory/'inspection.json').write_text(json.dumps(inventory, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    meta = GGUFFile(model).metadata
    tokenizer, template = Tokenizer.from_gguf(model), renderer(meta['tokenizer.chat_template'])
    vocab = meta['tokenizer.ggml.tokens']
    texts = [('ru', 'Сколько будет 2 + 2? Ответь одной цифрой.')]
    if not args.smoke:
        texts += [('en', 'Name the first five prime numbers. Answer briefly.'),
                  ('zh', '中国的首都是哪里？请简短回答。'),
                  ('code', 'Write a Python function that adds two integers. Return code only.'),
                  ('numbers', 'Compute 123 + 456. Give only the number.'),
                  ('long', '\n'.join(f'Record {i}: the boat was blue and the car was green.' for i in range(24))+
                   '\nWhat color was the boat? Answer with one word.')]
    prompts = []
    for name, text in texts:
        rendered = template.render(messages=[{'role': 'user', 'content': text}], tools=None,
                                   reasoning_effort='no_think', add_generation_prompt=True,
                                   bos_token=vocab[meta['tokenizer.ggml.bos_token_id']],
                                   eos_token=vocab[meta['tokenizer.ggml.eos_token_id']])
        ids = tokenizer.encode(rendered, parse_special=True)
        if len(ids)+args.predict > 2048:
            raise ValueError('corpus exceeds baseline context')
        prompts.append(dict(name=name, text=text, rendered=rendered, ids=ids))
    report = dict(status='error', scope='full-model exact native/pinned-file logits and greedy parity; sequential timing observations',
                  model=str(model), engine_sha256=sha(binary), header_sha256=inventory['header_sha256'],
                  template_sha256=TEMPLATE_SHA, context=2048, batch=17, kv='f32', mtp=False,
                  cache=False, pipeline=False, cuda_fusion=False, tf32=False, file_cache_purged=False,
                  predict=args.predict, smoke=args.smoke, prompts=prompts, runs=[])
    report_path = directory/'model-report.json'
    def save():
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    os.environ['PATH'] = str(args.cuda_bin.resolve())+os.pathsep+os.environ.get('PATH', '')
    save()
    try:
        for mode in ('native', 'pinned'):
            work = directory/mode
            work.mkdir()
            engine, monitor = None, Monitor()
            record = dict(mode=mode, requests=[])
            report['runs'].append(record)
            try:
                check_ceiling(memory(gpu_memory()))
                print('Loading', mode, flush=True)
                start = time.perf_counter()
                engine = Engine(binary, model, work, mode=mode, logits=True, on_start=monitor.start)
                record.update(ready_seconds=time.perf_counter()-start, info=engine.info)
                save()
                if 'architecture=hy_v3' not in engine.info[0] or 'gpu_only=1' not in engine.info[0]:
                    raise ValueError('unexpected engine capabilities')
                print('READY', mode, round(record['ready_seconds'], 2), 's', flush=True)

                def generate(prompt, name):
                    last = time.perf_counter()
                    def progress(done, total):
                        nonlocal last
                        if time.perf_counter()-last > 20:
                            print(mode, name, 'prefill', done, '/', total, flush=True)
                            last = time.perf_counter()
                    result = engine.generate(prompt['ids'], args.predict, sampling='temperature=0', on_progress=progress)
                    raw = (work/'logits.f32').read_bytes()
                    if len(raw) != len(result['ids'])*len(vocab)*4:
                        raise ValueError('wrong logits length')
                    result.update(name=name, text=b''.join(tokenizer.token_bytes(i) for i in result['ids']).decode('utf8', errors='replace'),
                                  logits_sha256=hashlib.sha256(raw).hexdigest(), logits_bytes=len(raw), metrics=metrics(engine)[-1])
                    m = result['metrics']
                    if m['rejected_cpu_nodes'] or m['rejected_full_copies'] or not m['expert_nodes']:
                        raise ValueError('GPU-only/selected-copy audit failed')
                    if mode == 'pinned' and (m['source_bytes'] != m['h2d_bytes'] or m['staging_bytes'] != 16*2**20):
                        raise ValueError('pinned-file byte accounting failed')
                    result['decode_tokens_per_second'] = m['decode_steps']*1000/m['decode_forward_ms'] if m['decode_forward_ms'] else None
                    result['generation_tokens_per_second'] = len(result['ids'])*1000/m['generation_wall_ms'] if m['generation_wall_ms'] else None
                    reference = None
                    if name.endswith('_repeat'):
                        reference = record['requests'][0]
                    elif mode == 'pinned':
                        reference = next(r for r in report['runs'][0]['requests'] if r['name'] == name)
                    result['exact_reference'] = reference is None or (result['ids'] == reference['ids'] and
                        result['logits_sha256'] == reference['logits_sha256'] and result['logits_bytes'] == reference['logits_bytes'])
                    record['requests'].append(result)
                    (work/(name+'.f32')).write_bytes(raw)
                    save()
                    if not result['exact_reference']:
                        raise ValueError(mode+' '+name+' differs from reference')
                    print(mode, name, len(result['ids']), 'tokens, decode', result['decode_tokens_per_second'],
                          'tok/s, text', ascii(result['text']), flush=True)
                for prompt in prompts:
                    generate(prompt, prompt['name'])
                if not args.smoke:
                    generate(prompts[0], 'ru_repeat')
                    record['cancelled'] = engine.generate(prompts[-1]['ids'], 64, cancel=True)
                    if record['cancelled']['finish'] != 'cancel':
                        raise ValueError('STOP did not cancel')
                    generate(prompts[0], 'after_cancel_repeat')
                engine.close()
                record['exit_code'] = engine.process.returncode
                if record['exit_code']:
                    raise ValueError('engine exit failed')
            finally:
                try:
                    if engine and not engine.stderr.closed:
                        engine.close()
                finally:
                    monitor.close()
                    record.update(monitor_error=monitor.error, memory_samples=monitor.samples)
                    save()
            if monitor.error:
                raise ValueError(monitor.error)
        report['status'] = 'pass'
    except Exception as error:
        report['error'] = str(error)
    save()
    print(report['status'], report.get('error', ''), report_path, flush=True)
    return int(report['status'] != 'pass')


if __name__ == '__main__':
    raise SystemExit(main())
