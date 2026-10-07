"""MiMo pipe baseline: fresh output directory, explicit model, global 95% ceiling.

native uses the pin's original selected mmap copies; pinned changes only transport.
This compares the same quantized graph, not an independent Transformers oracle.
"""
import argparse
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time

import numpy as np
import psutil

from .check_mimo2_cuda import check_ceiling, gpu_memory, memory, sha
from .gguf_reader import GGUFFile
from .mimo2_template import renderer


class Engine:
    def __init__(self, command, env, directory, timeout):
        self.directory, self.timeout = directory, timeout
        self.lines, self.records, self.samples = queue.Queue(), [], []
        self.start, self.next_sample = time.monotonic(), 0
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding='utf8', errors='replace', bufsize=1,
            env=env, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        self.ps = psutil.Process(self.process.pid)
        self.threads = []
        for kind, stream in [('stdout', self.process.stdout), ('stderr', self.process.stderr)]:
            def read(kind=kind, stream=stream):
                with (directory/(kind+'.log')).open('w', encoding='utf8') as log:
                    for line in stream:
                        log.write(line); log.flush()
                        if kind == 'stdout':
                            self.lines.put(line.strip())
                        elif line.startswith('STRATA_MIMO_REQUEST '):
                            self.records.append(json.loads(line.split(' ', 1)[1]))
            thread = threading.Thread(target=read, daemon=True)
            thread.start(); self.threads.append(thread)

    def send(self, line):
        self.process.stdin.write(line+'\n'); self.process.stdin.flush()

    def read(self):
        while True:
            now = time.monotonic()
            if now-self.start > self.timeout:
                raise TimeoutError('MiMo engine run exceeded timeout')
            if now >= self.next_sample:
                sample = memory(gpu_memory()); sample['seconds'] = now-self.start
                try:
                    info = self.ps.memory_info()
                    sample.update(process_rss=info.rss, process_peak_wset=getattr(info, 'peak_wset', None))
                    cpu = self.ps.cpu_times()
                    sample.update(process_cpu_user_seconds=cpu.user, process_cpu_system_seconds=cpu.system)
                except psutil.NoSuchProcess:
                    pass
                self.samples.append(sample); check_ceiling(sample); self.next_sample = now+1
            try:
                return self.lines.get(timeout=.1)
            except queue.Empty:
                if self.process.poll() is not None:
                    raise RuntimeError(f'engine exited {self.process.returncode}; see stderr.log')

    def request(self, ids, count, cancel_after_pp=False):
        self.send('GEN '+str(count)+' '+','.join(map(str, ids)))
        result = dict(ids=ids, requested=count, tokens=[], progress=[])
        cancelled = False
        while True:
            line = self.read()
            if line.startswith('PP '):
                result['progress'].append(int(line.split()[1]))
                if cancel_after_pp and not cancelled:
                    self.send('STOP'); cancelled = True
            elif line.startswith('T '):
                result['tokens'].append(int(line.split()[1]))
            elif line.startswith('DONE '):
                result['done'] = line
                if cancel_after_pp and ' cancel ' not in line:
                    raise AssertionError('cancel did not finish as cancel')
                if not cancel_after_pp and not any(' '+x+' ' in line for x in ('length', 'stop')):
                    raise AssertionError(line)
                return result
            elif line.startswith('ERR '):
                raise RuntimeError(line+'; see stderr.log')
            else:
                raise AssertionError('unexpected pipe response: '+line)

    def close(self):
        if self.process.poll() is None:
            try:
                self.send('QUIT'); self.process.wait(timeout=30)
            except (OSError, subprocess.TimeoutExpired):
                self.process.terminate()
                try: self.process.wait(timeout=10)
                except subprocess.TimeoutExpired: self.process.kill(); self.process.wait(timeout=10)
        for thread in self.threads: thread.join(timeout=10)


def compare_logits(a, b):
    if a.shape != b.shape or not a.size:
        raise ValueError('empty/mismatched logits')
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    delta = a.astype(np.float64)-b
    max_abs = float(np.max(np.abs(delta)))
    nmse = float(np.sum(delta*delta)/max(float(np.sum(b.astype(np.float64)**2)), 1e-30))
    exact = a.tobytes() == b.tobytes()
    return dict(pass_=finite and exact, finite=finite, bit_exact=exact, max_abs=max_abs, nmse=nmse, elements=int(a.size))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--cuda-bin', type=Path, required=True)
    p.add_argument('--mode', choices=['native', 'pinned', 'both'], default='both')
    p.add_argument('--suite', choices=['smoke', 'corpus', 'repeat', 'fixture'], default='smoke')
    p.add_argument('--repeats', type=int, choices=range(3, 11), default=3, help='Repeat suite request count; use4 for three warm requests')
    p.add_argument('--timeout', type=int, default=1200)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--expert-cache-mib', type=int, default=0, help='Cache applies to pinned mode; native stays the reference')
    p.add_argument('--cache-slab-mib', type=int, choices=[0,16,32], default=0)
    p.add_argument('--pipeline-batch', type=int, choices=[0,1], default=0)
    p.add_argument('--cache-decay', type=int, choices=[0,16384,65536,131072], default=0)
    p.add_argument('--cache-fill-batch', type=int, choices=[0,1], default=0)
    p.add_argument('--expert-cache-prefill', choices=['on', 'off'], default='on')
    p.add_argument('--expert-reader', choices=['file', 'mmap'], default='file')
    p.add_argument('--expert-readers', type=int, choices=[0, 1, 2], default=0)
    p.add_argument('--expert-chunk-mib', type=int, choices=[4, 8, 16], default=8)
    p.add_argument('--trace-graphs', type=int, default=0, help='Diagnostic CUDA timeline, excluded from speed comparisons')
    p.add_argument('--reference-dir', type=Path, help='Completed run containing reference-mode logits for the identical cases')
    p.add_argument('--reference-mode', choices=['native','pinned'], default='native', help='Explicit mode from the completed reference run')
    args = p.parse_args()
    build, model = args.build.resolve(), args.model.resolve()
    manifest_path = build/'mimo2-build-manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    if not manifest.get('runtime') or not manifest.get('cuda'):
        p.error('MiMo runtime build required')
    name = 'strata-mimo2-fixture-engine' if args.suite == 'fixture' else 'strata-mimo2'
    binary = build/'bin'/(name+'.exe')
    directory = args.output_dir.resolve(); directory.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy(); env['PATH'] = str(args.cuda_bin.resolve())+os.pathsep+env.get('PATH', '')
    env['STRATA_MIMO_CACHE_SLAB_MIB'] = str(args.cache_slab_mib)
    env['STRATA_MIMO_PIPELINE_BATCH'] = str(args.pipeline_batch)
    env['STRATA_MIMO_CACHE_DECAY'] = str(args.cache_decay)
    env['STRATA_MIMO_CACHE_FILL_BATCH'] = str(args.cache_fill_batch)
    report = dict(status='error', scope='GPU pipe with selected cache/transport; no MTP/API', cache_slab_mib=args.cache_slab_mib,
        model=str(model), model_bytes=model.stat().st_size, suite=args.suite, pipeline_batch=args.pipeline_batch, cache_decay=args.cache_decay, cache_fill_batch=args.cache_fill_batch,
        binary_sha256=sha(binary), manifest=manifest, manifest_sha256=sha(manifest_path),
        modes=[], comparisons=[], memory_ceiling=.95, memory_sampling_seconds=1)
    if args.suite == 'fixture':
        cases = [('short', list(range(9)), 8), ('swa128', [i%64 for i in range(145)], 4)]
        vocab = 64
    else:
        meta = GGUFFile(model).metadata; template = renderer(meta['tokenizer.chat_template'])
        questions = [('en', 'Reply with exactly: Hello world.'),
                     ('swa128', 'Remember this list: '+', '.join(str(i) for i in range(60))+'. What is the last number?')]
        if args.suite == 'corpus':
            questions += [('ru', 'Назови столицу Франции. Ответь одним словом.'),
                          ('zh', '法国的首都是哪里？请简短回答。'),
                          ('code', 'Write Python code: a function that adds two integers.'),
                          ('numbers', 'Compute 17 * 23. Reply with only the number.')]
        if args.suite == 'repeat':
            questions = [(f'en-repeat-{i+1}', 'Count from 1 to 20 separated by commas. Start immediately.') for i in range(args.repeats)]
        cases = [(name, template.render(messages=[dict(role='user', content=text)],
                    enable_thinking=False, add_generation_prompt=True),
                  32 if args.suite == 'repeat' else 12 if name.startswith('en') else 4)
                 for name, text in questions]
        vocab = 152576
    engine = None
    try:
        before = memory(gpu_memory()); check_ceiling(before); report['before'] = before
        for mode in (['native', 'pinned'] if args.mode == 'both' else [args.mode]):
            dest = directory/mode; dest.mkdir()
            logits = dest/'logits.f32'
            command = [str(binary), '--native', str(model), '--serve', '--max-context', '512',
                       '--batch-size', str(args.batch_size), '--copy-mode', mode, '--logits-file', str(logits)]
            command += ['--expert-cache-mib',str(args.expert_cache_mib if mode=='pinned' else 0),
                        '--expert-cache-prefill',args.expert_cache_prefill,'--expert-reader',args.expert_reader if mode=='pinned' else 'file']
            command += ['--expert-readers', str(args.expert_readers if mode=='pinned' else 0),
                        '--expert-chunk-mib', str(args.expert_chunk_mib)]
            if args.trace_graphs and mode=='pinned':
                command += ['--trace-graphs', str(args.trace_graphs), '--trace-file', str(dest/'cuda-trace.json')]
            entry = dict(mode=mode, command=command, cases=[]); report['modes'].append(entry)
            print('starting', mode, args.suite, flush=True)
            engine = Engine(command, env, dest, args.timeout)
            entry['samples'] = engine.samples
            while True:
                line = engine.read()
                if line.startswith('INFO '): entry['info'] = line
                elif line.startswith('READY '): break
            entry['load_wall_seconds'] = time.monotonic()-engine.start
            for name, text, count in cases:
                if isinstance(text, str):
                    engine.send('ENC 1 '+text.encode('utf8').hex()); line = engine.read()
                    if not line.startswith('IDS '): raise AssertionError(line)
                    ids = [int(n) for n in line.split()[1:]]
                else: ids = text
                if name == 'swa128' and len(ids) <= 128: raise AssertionError('long case must cross SWA128')
                print(mode, name, 'prompt tokens', len(ids), flush=True)
                case = engine.request(ids, count); case['name'] = name; entry['cases'].append(case)
                print(mode, name, case['done'], flush=True)
            # Lifecycle/error recovery with identical fresh prompt, independent of model text.
            short = entry['cases'][0]
            engine.send('GEN 1 -1'); line = engine.read()
            if not line.startswith('ERR '): raise AssertionError('invalid IDs accepted')
            recovered = engine.request(short['ids'], 1)
            if recovered['tokens'] != short['tokens'][:1]: raise AssertionError('error recovery changed fresh token')
            entry['error_recovery'] = recovered
            entry['cancel'] = engine.request(entry['cases'][1]['ids'], 8, cancel_after_pp=True)
            recovered = engine.request(short['ids'], 1)
            if recovered['tokens'] != short['tokens'][:1]: raise AssertionError('cancel recovery changed fresh token')
            entry['cancel_recovery'] = recovered
            engine.close(); entry['exit_code'] = engine.process.returncode; entry['requests'] = engine.records
            if entry['exit_code'] != 0: raise AssertionError('engine did not unload cleanly')
            engine = None
            if (dest/'cuda-trace.json').is_file():
                entry['cuda_trace'] = json.loads((dest/'cuda-trace.json').read_text(encoding='utf8'))
            if any(r['rejected_cpu_nodes'] or r['rejected_full_copies'] for r in entry['requests']):
                raise AssertionError('execution audit failed')
            if mode == 'pinned' and args.expert_cache_mib:
                for r in entry['requests']:
                    if not 0 <= r['cache_bytes'] <= r['cache_limit'] <= args.expert_cache_mib*2**20:
                        raise AssertionError('cache exceeded its live budget')
                    if r['finish'] in ('length', 'stop') and r['requested_bytes'] != r['h2d_bytes']+r['cache_hit_bytes']:
                        raise AssertionError('cache delivery byte accounting mismatch')
                    if r.get('pipeline_reader_owned',0) or r.get('pipeline_queued',0):
                        raise AssertionError('pipeline still owns queued work after request')
                entry['cache_audit'] = 'budget and delivered-byte accounting pass'
            entry['logits_rows'] = logits.stat().st_size//(vocab*4)
            entry['logits_sha256'] = sha(logits)
            if args.suite != 'fixture':
                from .mimo2_tokenizer import from_gguf
                tokenizer = from_gguf(model)
                for case in entry['cases']:
                    case['text'] = tokenizer.decode(case['tokens'])
        if len(report['modes']) == 2:
            left, right = report['modes']
            a = np.fromfile(directory/'native'/'logits.f32', dtype='<f4').reshape(-1, vocab)
            b = np.fromfile(directory/'pinned'/'logits.f32', dtype='<f4').reshape(-1, vocab)
            ai = bi = 0
            for x, y in zip(left['cases'], right['cases']):
                n, m = len(x['tokens']), len(y['tokens'])
                if x['ids'] != y['ids'] or x['tokens'] != y['tokens'] or n != m:
                    raise AssertionError('native/pinned input or greedy IDs differ')
                check = compare_logits(a[ai:ai+n], b[bi:bi+m]); check['name'] = x['name']
                report['comparisons'].append(check); ai += n; bi += m
                if not check['pass_']: raise AssertionError('native/pinned logits differ')
            report['parity'] = 'bit-exact logits and greedy IDs'
        if args.reference_dir:
            reference_path = args.reference_dir.resolve()
            reference = json.loads((reference_path/'engine-report.json').read_text(encoding='utf8'))
            if reference.get('status') != 'pass' or reference['model'] != str(model) or reference['model_bytes'] != model.stat().st_size:
                raise ValueError('reference must be a completed run for this model')
            ref = next(x for x in reference['modes'] if x['mode'] == args.reference_mode)
            expected = np.fromfile(reference_path/args.reference_mode/'logits.f32', dtype='<f4').reshape(-1, vocab)
            report['external_reference'] = str(reference_path)
            report['external_reference_mode'] = args.reference_mode
            report['external_comparisons'] = []
            for current in report['modes']:
                if len(ref['cases']) != len(current['cases']): raise ValueError('reference case count differs')
                actual = np.fromfile(directory/current['mode']/'logits.f32', dtype='<f4').reshape(-1, vocab)
                offset = 0
                for a_case, b_case in zip(current['cases'], ref['cases']):
                    if any(a_case[key] != b_case[key] for key in ('name', 'ids', 'tokens')):
                        raise AssertionError('external reference case/IDs differ')
                    n = len(a_case['tokens']); check = compare_logits(actual[offset:offset+n], expected[offset:offset+n]); offset += n
                    check.update(name=a_case['name'], mode=current['mode']); report['external_comparisons'].append(check)
                    if not check['pass_']: raise AssertionError('external reference logits differ')
            report['parity'] = 'bit-exact logits and greedy IDs against external '+args.reference_mode+' reference'
        report['status'] = 'pass'
    except Exception as error:
        report['error'] = str(error)
    finally:
        if engine:
            engine.close()
            report['last_exit_code'] = engine.process.returncode
            if report['modes']: report['modes'][-1]['requests'] = engine.records
        (directory/'engine-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(report['status'], report.get('error', ''), directory/'engine-report.json', flush=True)
    return int(report['status'] != 'pass')


if __name__ == '__main__':
    raise SystemExit(main())
