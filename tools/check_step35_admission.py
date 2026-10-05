"""Compare optional Step prefill admission with exact logits and global memory guards.

Default: on/off/off/on, identical short/medium/short request sequences.
--long-only: one off run against the saved STEP-07 long-context reference;
its timing is an observation, not a paired benchmark.
Output directory must be new, so logs/references cannot overwrite inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

from check_step35_engine import Engine
from check_step35_model import Monitor
from check_step35_template import renderer, TEMPLATE_SHA
from gguf_reader import GGUFFile
from setup_step35 import first_shard
from strata_tokenizer import Tokenizer


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(engine):
    return [json.loads(line.split(' ', 1)[1]) for line in
            engine.stderr_path.read_text(encoding='utf8', errors='replace').splitlines()
            if line.startswith('STRATA_STEP_REQUEST ')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', type=Path, required=True)
    parser.add_argument('--gguf', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--context-reference', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--long-only', action='store_true')
    args = parser.parse_args()
    model = first_shard(args.gguf).resolve()
    reference = json.loads(args.reference.read_text(encoding='utf8'))
    context_ref = json.loads(args.context_reference.read_text(encoding='utf8'))
    for ref in (reference, context_ref):
        if ref['status'] != 'pass' or Path(ref['model']).resolve() != model:
            parser.error('passing numerical references for this model are required')
    if context_ref['context'] != 4096 or context_ref['batch'] != 17:
        parser.error('context reference must use context4096/batch17')
    meta = GGUFFile(model).metadata
    if hashlib.sha256(meta['tokenizer.chat_template'].encode()).hexdigest() != TEMPLATE_SHA:
        parser.error('unreviewed model template')
    vocab = meta['tokenizer.ggml.tokens']
    tokenizer = Tokenizer.from_gguf(model)
    body = '\n'.join(context_ref['long_prompt']['text'].splitlines()[:24])
    body += '\nWhich color was the boat? Answer briefly.'
    rendered = renderer(meta['tokenizer.chat_template']).render(
        messages=[{'role': 'user', 'content': body}], tools=None, add_generation_prompt=True,
        reasoning_effort='low', bos_token=vocab[meta['tokenizer.ggml.bos_token_id']],
        eos_token=vocab[meta['tokenizer.ggml.eos_token_id']])
    medium = tokenizer.encode(rendered, parse_special=True)
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    controls = {'A': reference['runs'][0]['requests'][0], 'B': reference['runs'][0]['requests'][1]}
    controls['long'] = next(r for r in context_ref['runs'][0]['requests'] if r['name'] == 'long')
    order = ['off'] if args.long_only else ['on', 'off', 'off', 'on']
    report = {'status': 'error', 'scope': 'long numerical check' if args.long_only else 'paired admission timings',
              'engine_sha256': sha(args.engine), 'reference_sha256': sha(args.reference),
              'context_reference_sha256': sha(args.context_reference), 'model': str(model),
              'context': 4096, 'batch': 17, 'kv': 'f32', 'cache': 'auto', 'readers': 1,
              'chunk_mib': 8, 'mtp': False, 'order': order, 'file_cache_purged': False,
              'medium_prompt': {'text': body, 'ids': medium}, 'runs': []}

    def save():
        (directory / 'admission-report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf8')

    try:
        for index, policy in enumerate(order):
            work = directory / f'{index}-{policy}'
            work.mkdir()
            run = {'policy': policy, 'requests': []}
            report['runs'].append(run)
            engine, monitor = None, Monitor()
            try:
                print(f'Loading run {index}, prefill admission={policy}', flush=True)
                started = time.perf_counter()
                engine = Engine(args.engine.resolve(), model, work, logits=True, on_start=monitor.start,
                                extra_args=['--max-context', '4096', '--expert-cache-mib', 'auto',
                                            '--expert-pipeline-readers', '1', '--expert-cache-prefill', policy])
                run.update(info=engine.info, ready_seconds=time.perf_counter()-started)

                def generate(name, key, ids, count):
                    last = time.perf_counter()

                    def progress(done, total):
                        nonlocal last
                        if time.perf_counter()-last >= 25:
                            print(f'run {index}/{policy} {name}: prefill {done}/{total}', flush=True)
                            last = time.perf_counter()

                    result = engine.generate(ids, count, sampling='temperature=0', on_progress=progress)
                    raw = (work / 'logits.f32').read_bytes()
                    result.update(name=name, key=key, logits_sha256=hashlib.sha256(raw).hexdigest(),
                                  logits_bytes=len(raw), metrics=metrics(engine)[-1])
                    run['requests'].append(result)
                    if key not in controls:
                        controls[key] = result
                        (work / f'{key}-reference.f32').write_bytes(raw)
                        result['establishes_control'] = True
                    expected = controls[key]
                    result['exact'] = (result['ids'] == expected['ids'] and
                                       result['logits_sha256'] == expected['logits_sha256'] and
                                       len(raw) == expected['logits_bytes'])
                    if not result['exact']:
                        raise ValueError(f'{name}: logits/IDs differ from control')
                    m = result['metrics']
                    if policy == 'off' and (m['prefill_cache_fill_bytes'] or not m['prefill_admission_skips']):
                        raise ValueError('off policy did not suppress prefill cache admission')
                    if policy == 'on' and m['prefill_admission_skips']:
                        raise ValueError('default policy unexpectedly skipped admission')
                    print(f'run {index}/{policy} {name}: exact {len(result["ids"])} output, '
                          f'prefill {result["prefill_ms"]/1000:.3f}s, wall {result["wall_seconds"]:.3f}s', flush=True)
                    save()

                a, b = (p['ids'] for p in reference['prompts'])
                generate('A_cold', 'A', a, reference['predict'])
                if args.long_only:
                    generate('long', 'long', context_ref['long_prompt']['ids'], 16)
                else:
                    generate('B', 'B', b, reference['predict'])
                    generate('medium', 'medium', medium, 16)
                generate('A_after_context', 'A', a, reference['predict'])
                generate('A_warm', 'A', a, reference['predict'])
                run['cancelled'] = engine.generate(context_ref['long_prompt']['ids'], 128, cancel=True)
                if run['cancelled']['finish'] != 'cancel':
                    raise ValueError('STOP failed')
                generate('A_after_cancel', 'A', a, reference['predict'])
                engine.close()
                run['exit_code'] = engine.process.returncode
                if run['exit_code']:
                    raise ValueError('engine exit failed')
            finally:
                if engine and not engine.stderr.closed:
                    engine.close()
                monitor.close()
                run.update(monitor_error=monitor.error, memory_samples=monitor.samples)
                if engine:
                    run['metrics'] = metrics(engine)
                save()
            if monitor.error:
                raise ValueError(monitor.error)
        report['status'] = 'pass'
    except Exception as error:
        report['error'] = str(error)
    save()
    print(report['status'], report.get('error', ''), flush=True)
    return int(report['status'] != 'pass')


if __name__ == '__main__':
    raise SystemExit(main())
