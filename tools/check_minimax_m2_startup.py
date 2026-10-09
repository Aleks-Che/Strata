"""Live isolated CLI startup/auth/unload/reload smoke; sequential with other GPU tests."""
import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve.minimax_m2_engine import MiniMaxEngine, EXE_SHA256, EVENTS_EXE_SHA256
from serve.winjob import contain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gguf', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--engine', type=Path, default=ROOT/'build-local/minimax-m2-cuda/bin/strata-minimax-m2-bench.exe')
    parser.add_argument('--pipeline-events', type=int, choices=(0, 2), default=0)
    args = parser.parse_args()
    digest = hashlib.sha256(args.engine.read_bytes()).hexdigest()
    assert digest in (EXE_SHA256, EVENTS_EXE_SHA256)
    assert not args.pipeline_events or digest == EVENTS_EXE_SHA256
    args.out.mkdir(parents=True, exist_ok=False)
    key = 'local-minimax-startup-fixture'
    cmd = [sys.executable, '-u', '-m', 'serve.minimax_m2_server', '--gguf', args.gguf, '--port', '0',
           '--api-key', key, '--ctx', '2048', '--batch', '16', '--gpu-cache-mib', '18432',
           '--pipeline-readers', '2', '--pipeline-chunk-mib', '4', '--log', str((args.out/'native.log').resolve())]
    cmd += ['--engine', str(args.engine.resolve()), '--pipeline-events', str(args.pipeline_events)]
    report = {'pass': False, 'cases': [], 'command': cmd, 'sources': {}, 'engine_sha256': digest,
              'pipeline_events': args.pipeline_events}
    for path in ['serve/minimax_m2_server.py', 'serve/minimax_m2_engine.py', 'serve/minimax_m2_worker.py',
                 'tools/check_minimax_m2_startup.py', 'serve/requirements-minimax.txt']:
        data = (ROOT/path).read_bytes()
        target = args.out/'sources'/path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        report['sources'][path] = hashlib.sha256(data).hexdigest()
    proc = thread = None
    with (args.out/'server.log').open('w', encoding='utf-8') as log:
        try:
            flags = {'creationflags': subprocess.CREATE_NO_WINDOW, 'startupinfo': MiniMaxEngine._hidden()} if os.name == 'nt' else {}
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=ROOT,
                                    encoding='utf-8', **flags)
            report['server_pid'] = proc.pid
            contain(proc)
            inbox = queue.Queue()
            def read():
                for line in proc.stdout:
                    log.write(line); log.flush()
                    if line.startswith('Experimental MiniMax API at '):
                        inbox.put(line)
                inbox.put(None)
            thread = threading.Thread(target=read, daemon=True)
            thread.start()
            line = inbox.get(timeout=180)
            assert line, 'server exited before ready'
            port = int(re.search(r':(\d+)\s*$', line)[1])

            def request(name, path, body=None, auth=True):
                conn = http.client.HTTPConnection('127.0.0.1', port, timeout=180)
                try:
                    headers = {'Content-Type': 'application/json'}
                    if auth:
                        headers['Authorization'] = 'Bearer '+key
                    start = time.monotonic()
                    conn.request('POST' if body is not None else 'GET', path,
                                 None if body is None else json.dumps(body), headers)
                    response = conn.getresponse()
                    payload = response.read().decode('utf-8')
                    (args.out/(name+'.txt')).write_text(payload, encoding='utf-8')
                    report['cases'].append({'name': name, 'status': response.status, 'wall_ms': 1000*(time.monotonic()-start)})
                    print(name+': '+str(response.status), flush=True)
                    return response.status, payload
                finally:
                    conn.close()

            req = {'model': 'minimax-m2.7', 'max_tokens': 8, 'temperature': 0, 'messages': [{'role': 'user',
                   'content': 'Explain in a few sentences why the sky appears blue during the day.'}]}
            assert request('auth-required', '/v1/chat/completions', req, auth=False)[0] == 401
            assert request('web-assets', '/')[0] == 200  # assets only, not a browser chat acceptance test
            answers = []
            for i in range(2):
                status, raw = request('prefix-'+str(i), '/v1/chat/completions', req)
                value = json.loads(raw)
                assert status == 200 and value['usage']['completion_tokens'] == 8
                assert value['choices'][0]['finish_reason'] == 'length'
                answers.append(value['choices'][0]['message'])
                assert request('unload-'+str(i), '/v1/unload', {})[0] == 200
            assert answers[0] == answers[1]
            report['pass'] = True
        except Exception:
            report['error'] = traceback.format_exc()
            raise
        finally:
            # Model is unloaded by HTTP on success; kill only this owned CLI.
            if proc and proc.poll() is None:
                proc.terminate(); proc.wait(timeout=15)
            if thread:
                thread.join(10)
            if proc:
                proc.stdout.close()
                report['server_stopped'] = proc.poll() is not None
            (args.out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
