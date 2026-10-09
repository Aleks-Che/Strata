"""Opt-in real HTTP context4096/pressure/cancel audit; no speed or quality claim.

Uses the reviewed native engine and a separate bounded, read-only RAM/CUDA
holder. All processes and allocations are owned by this checker. Run alone.
"""
import argparse
from copy import deepcopy
import hashlib
import http.client
import json
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve.minimax_m2_api import MiniMaxAPITemplate
from serve.minimax_m2_engine import MiniMaxEngine, EXE_SHA256, HEADER_SHA256
from serve.server import Server, Service, make_handler, CTX_SLACK
from serve.winjob import contain
from tools.check_minimax_m2_completion_pipe import result_checks
from tools.check_minimax_m2_live_tools import collect_sse, save
from tools.run_minimax_m2 import runtime_environment
from tools.strata_tokenizer import Tokenizer


class Holder:
    def __init__(self, binary, gguf, cuda, directory):
        self.records = []
        self.lock = threading.Lock()
        self.errors = (directory/'holder.stderr.log').open('w', encoding='utf-8')
        self.proc = subprocess.Popen([str(binary), str(gguf)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=self.errors, text=True, encoding='utf-8', env=runtime_environment(cuda),
                                    creationflags=subprocess.CREATE_NO_WINDOW)
        if not contain(self.proc):
            self.proc.kill(); self.proc.wait(); self.errors.close()
            raise RuntimeError('cannot contain pressure holder')
        self.lines = queue.Queue()
        def pump():
            for line in self.proc.stdout:
                self.lines.put(line)
            self.lines.put(None)
        self.thread = threading.Thread(target=pump, daemon=True)
        self.thread.start()

    def expect(self, phase, timeout=300):
        end = time.monotonic()+timeout
        while True:
            line = self.lines.get(timeout=max(.01, end-time.monotonic()))
            if line is None:
                raise RuntimeError('pressure holder EOF')
            r = json.loads(line)
            r['monotonic'] = time.monotonic()
            self.records.append(r)
            if r['phase'] == 'progress':
                print('RAM pressure: touched '+str(round(r['ram_touched']/2**30, 1))+' GiB', flush=True)
                continue
            if r['phase'] != phase:
                raise RuntimeError('unexpected holder event: '+str(r))
            for axis in ('gpu', 'ram'):
                assert r[axis+'_total']*.05 <= r[axis+'_free'] <= r[axis+'_total'], r
            return r

    def command(self, command, phase):
        with self.lock:
            self.proc.stdin.write(command+'\n'); self.proc.stdin.flush()
            return self.expect(phase)

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.close()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill(); self.proc.wait(timeout=10)
        self.thread.join(5)
        self.proc.stdout.close()
        if not self.proc.stdin.closed:
            self.proc.stdin.close()
        self.errors.close()


def exact_prompt(svc, target):
    """Repeated filler with ordinary text and the real native chat template."""
    n = target
    for _ in range(12):
        body = 'Marker at start: CODE-4179. Ignore filler below.\n'+(' A'*n)+'\nReturn the marker.'
        req = {'model': 'minimax-m2.7', 'messages': [{'role': 'user', 'content': body}], 'temperature': 0, 'max_tokens': 8}
        messages, tools, kwargs = svc.normalize_request(req, 'openai')
        ids = svc.tok.encode(svc.template.render(messages, tools=tools, **kwargs), parse_special=True)
        if len(ids) == target:
            return req, ids
        n += target-len(ids)
        assert n > 0
    raise ValueError('cannot construct exact prompt token count')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gguf', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cuda-root', type=Path, default=ROOT/'build-local/cuda-13.0')
    parser.add_argument('--holder', type=Path, default=ROOT/'build-local/minimax-m2-cuda/bin/strata-minimax-m2-pressure-holder.exe')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    binary = ROOT/'build-local/minimax-m2-cuda/bin/strata-minimax-m2-bench.exe'
    report = {'pass': False, 'scope': 'real HTTP context4096 + external RAM/VRAM pressure; exact IDs, not logits/quality/speed A/B',
              'model': str(args.gguf.resolve()), 'engine_sha256': EXE_SHA256, 'header_sha256': HEADER_SHA256,
              'cases': [], 'sources': {}, 'ctx_slack': CTX_SLACK, 'monitor': [], 'monitor_error': None}
    sources = [*sorted((ROOT/'serve').glob('*.py')), *sorted((ROOT/'backends/minimax_m2').glob('*')),
               ROOT/'backends/common/device_memory.hpp', ROOT/'backends/glm5next/host_pages.hpp',
               ROOT/'tools/check_minimax_m2_live_tools.py', ROOT/'tools/check_minimax_m2_completion_pipe.py',
               ROOT/'tools/strata_tokenizer.py', ROOT/'serve/fixtures/minimax_m27_chat_template.jinja', Path(__file__).resolve()]
    for source in sources:
        if not source.is_file():
            continue
        relative = source.relative_to(ROOT)
        destination = args.out/'sources'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        report['sources'][relative.as_posix()] = hashlib.sha256(source.read_bytes()).hexdigest()
    for source, name in ((binary, 'engine.exe'), (args.holder, 'holder.exe')):
        shutil.copyfile(source, args.out/name)
        report[name+'_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    report['gpu'] = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
                                             encoding='utf-8', timeout=15).strip()
    holder = engine = server = server_thread = monitor_thread = None
    monitor_stop = threading.Event()
    def record():
        if holder:
            report['holder_records'] = list(holder.records)
        save(args.out/'report.json', report)
    try:
        holder = Holder(args.holder.resolve(), args.gguf.resolve(), args.cuda_root, args.out)
        report['holder_ready'] = holder.expect('ready')
        tok = Tokenizer.from_gguf(args.gguf)
        engine = MiniMaxEngine(args.gguf, binary, args.cuda_root, args.out/'native.log', context=4096, batch=16,
                              gpu_cache_mib=18432, pipeline_readers=2, pipeline_chunk_mib=4)
        report['header'], report['command'] = engine.header, engine.command
        native_pid = engine.header['native_pid']
        svc = Service(engine, tok, MiniMaxAPITemplate(ROOT/'serve/fixtures/minimax_m27_chat_template.jinja'), model_name='minimax-m2.7')
        server = Server(('127.0.0.1', 0), make_handler(svc))
        server_thread = threading.Thread(target=server.serve_forever, daemon=True); server_thread.start()
        long, long_ids = exact_prompt(svc, 4096-CTX_SLACK-8)
        over, over_ids = exact_prompt(svc, 4096)
        save(args.out/'long.prompt.json', {'ids': long_ids, 'text': tok.decode(long_ids)})
        short = {'model': 'minimax-m2.7', 'messages': [{'role': 'user', 'content': 'Explain in a few sentences why the sky appears blue during the day.'}],
                 'temperature': 0, 'max_tokens': 32}
        expected_short = json.loads((ROOT/'build-local/minimax-m2-completion-01/generation.json').read_text(encoding='utf-8'))['results'][0]['token_ids'][:32]
        report['short_reference_sha256'] = hashlib.sha256((ROOT/'build-local/minimax-m2-completion-01/generation.json').read_bytes()).hexdigest()
        print('READY context4096, long prompt '+str(len(long_ids))+' tokens, CUDA PID '+str(native_pid), flush=True)

        def monitor():
            try:
                while not monitor_stop.is_set():
                    sample = holder.command('SAMPLE', 'sample')
                    report['monitor'].append({**sample, 'native_request_id': engine._sequence})
                    monitor_stop.wait(.5)
            except Exception:
                report['monitor_error'] = traceback.format_exc()
                if engine._gate.locked():
                    engine._send({'command': 'cancel', 'request_id': engine._sequence})
        monitor_thread = threading.Thread(target=monitor, daemon=True); monitor_thread.start()

        def post(name, api, req, expected_status=200):
            print('START '+name, flush=True)
            save(args.out/(name+'.request.json'), req)
            prior = engine._sequence
            connection = http.client.HTTPConnection(*server.server_address, timeout=1000)
            started = time.monotonic()
            try:
                connection.request('POST', '/v1/chat/completions' if api == 'openai' else '/v1/messages',
                                   json.dumps(req).encode(), {'Content-Type': 'application/json'})
                response = connection.getresponse(); wire = response.read().decode('utf-8')
                (args.out/(name+'.response.txt')).write_text(wire, encoding='utf-8')
                row = {'name': name, 'status': response.status, 'wall_ms': 1000*(time.monotonic()-started),
                       'sequence_before': prior, 'sequence_after': engine._sequence, 'pass': False}
                report['cases'].append(row); record()
                assert response.status == expected_status, wire
                assert engine.alive() and engine.header['native_pid'] == native_pid
                if expected_status == 400:
                    assert engine._sequence == prior and 'context' in wire, wire
                else:
                    n = deepcopy(engine.last_result); row['native_result'] = n
                    assert n and not engine.last_error, engine.last_error
                    row['native_checks'] = result_checks(n, engine.header)
                    assert all(row['native_checks'].values()), row['native_checks']
                    answer = collect_sse(api, wire) if req.get('stream') else json.loads(wire)
                    save(args.out/(name+'.collected.json'), answer)
                    assert answer['usage']['prompt_tokens' if api == 'openai' else 'input_tokens'] == n['prompt_tokens']
                    assert answer['usage']['completion_tokens' if api == 'openai' else 'output_tokens'] == n['generated_tokens']
                    if api == 'openai':
                        assert answer['choices'][0]['finish_reason'] == ('stop' if n['stop_reason'] == 'eos' else 'length')
                    else:
                        assert answer['stop_reason'] == ('end_turn' if n['stop_reason'] == 'eos' else 'max_tokens')
                    assert engine._sequence == prior+1 and not engine._gate.locked()
                    if req['messages'] == short['messages']:
                        assert n['token_ids'] == expected_short
                assert not report['monitor_error'], report['monitor_error']
                row['pass'] = True; record()
                print('DONE '+name, flush=True)
                return row
            finally:
                connection.close()

        # Rejected requests must never reach CUDA, including stream=true.
        for api in ('openai', 'anthropic'):
            for stream in (False, True):
                post(api+'-'+str(stream)+'-one-token-over', api, {**long, 'max_tokens': 9, 'stream': stream}, 400)
                post(api+'-'+str(stream)+'-no-room', api, {**over, 'max_tokens': 0, 'stream': stream}, 400)
        a = post('openai-4k-boundary', 'openai', long)
        b = post('anthropic-4k-boundary', 'anthropic', {**long, 'stream': True})
        report['long_parity'] = a['native_result']['token_ids'] == b['native_result']['token_ids']
        assert report['long_parity'] and a['native_result']['prompt_tokens'] == b['native_result']['prompt_tokens'] == len(long_ids)
        warm = post('warm-short', 'openai', short)
        # Mapping touches clean checkpoint pages; it does not duplicate model RAM.
        report['pressure_ram'] = holder.command('RAM', 'ram'); record()
        report['pressure_gpu'] = holder.command('GPU', 'gpu'); record()
        stressed = post('under-pressure', 'anthropic', {**short, 'stream': True})
        before = warm['native_result']['decode']['arena_reserved']
        after = stressed['native_result']['decode']['arena_reserved']
        trims = sum(stressed['native_result'][p]['cache_pressure_trims'] for p in ('prefill', 'decode'))
        report['cache_pressure'] = {'before': before, 'after': after, 'trims': trims}
        assert before > after and trims > 0, report['cache_pressure']

        for api in ('openai', 'anthropic'):
            req = {**long, 'stream': api == 'anthropic'}
            name = api+'-pressure-prefill-disconnect'; save(args.out/(name+'.request.json'), req)
            conn = http.client.HTTPConnection(*server.server_address, timeout=10)
            prior = engine._sequence
            conn.request('POST', '/v1/chat/completions' if api == 'openai' else '/v1/messages', json.dumps(req), {'Content-Type': 'application/json'})
            end = time.monotonic()+10
            while engine._sequence == prior and time.monotonic() < end:
                time.sleep(.02)
            assert engine._sequence == prior+1 and engine._gate.locked()
            time.sleep(1)  # Intentionally cancel within the first long-prefill batch.
            started = time.monotonic(); conn.close()
            end = started+40
            while engine._gate.locked() and time.monotonic() < end:
                time.sleep(.05)
            row = {'name': name, 'native_error': deepcopy(engine.last_error), 'native_result': deepcopy(engine.last_result),
                   'cancel_ms': 1000*(time.monotonic()-started), 'native_pid': engine.header['native_pid'], 'pass': False}
            report['cases'].append(row); record()
            assert not engine._gate.locked() and engine.alive() and row['native_pid'] == native_pid
            assert row['native_result'] is None and row['native_error'] and 'cancelled' in row['native_error']['message']
            row['pass'] = True
            post(api+'-pressure-recovery', api, short)
        report['pressure_released'] = holder.command('FREE', 'freed')
        post('after-release', 'openai', short)
        pressure_samples = [m for m in report['monitor'] if m['gpu_bytes'] and m['ram_touched']]
        report['both_over85_samples'] = sum(all(1-m[a+'_free']/m[a+'_total'] >= .85 for a in ('gpu','ram')) for m in pressure_samples)
        assert report['both_over85_samples'] > 0, 'pressure never reached 85% on both physical memories'
        report['pass'] = all(c['pass'] for c in report['cases']) and not report['monitor_error']
    except Exception:
        report['error'] = traceback.format_exc()
        print(report['error'], flush=True)
    finally:
        monitor_stop.set()
        if monitor_thread:
            monitor_thread.join(310)
        if holder:
            holder.close(); report['holder_exit_code'] = holder.proc.returncode
        if server:
            server.shutdown()
        if engine:
            engine.close(); report['engine_alive_after_close'] = engine.alive()
        if server:
            server.server_close()
        if server_thread:
            server_thread.join(5)
        report['pass'] &= report.get('holder_exit_code') == 0 and report.get('engine_alive_after_close') is False
        record()
        print('REPORT pass='+str(report['pass']), flush=True)
    return 0 if report['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
