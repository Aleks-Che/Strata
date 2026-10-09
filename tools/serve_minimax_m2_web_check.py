"""Owned live browser-test server with request/native evidence and a local STOP file.

Interact with the ordinary web UI at ready.json's loopback URL. Creating OUT/STOP
ends this server, cancelling active work and saving evidence. It also exits after
30 minutes. This helper does not operate the browser or call external tools.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve.minimax_m2_api import MiniMaxAPITemplate
from serve.minimax_m2_engine import MiniMaxEngine
from serve.server import Server, Service, make_handler
from tools.strata_tokenizer import Tokenizer


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gguf', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    evidence = {'scope': 'live browser evidence; UI observations recorded separately', 'requests': [], 'generations': [], 'sources': {}}
    for source in [*sorted((ROOT/'serve').glob('*.py')), *sorted((ROOT/'serve/web').glob('*')), Path(__file__).resolve()]:
        if source.is_file():
            name = source.relative_to(ROOT)
            target = args.out/'sources'/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            evidence['sources'][str(name)] = hashlib.sha256(target.read_bytes()).hexdigest()
    lock = threading.Lock()
    def record():
        with lock:
            save(args.out/'evidence.json', evidence)

    class ObservedEngine(MiniMaxEngine):
        def generate(self, ids, max_new, sampling, cancel, **kwargs):
            record_id = len(evidence['generations'])+1
            item = {'index': record_id, 'prompt_ids': list(ids), 'max_new': max_new, 'sampling': deepcopy(sampling)}
            evidence['generations'].append(item)
            record()
            gen = super().generate(ids, max_new, sampling, cancel, **kwargs)
            try:
                yield from gen
            except Exception:
                item['exception'] = traceback.format_exc()
                raise
            finally:
                gen.close()
                item.update(native_result=deepcopy(self.last_result), native_error=deepcopy(self.last_error),
                            cancelled=cancel.is_set(), engine_alive=self.alive(), native_pid=self.header['native_pid'])
                record()

    engine = server = thread = None
    try:
        tok = Tokenizer.from_gguf(args.gguf)
        engine = ObservedEngine(args.gguf, ROOT/'build-local/minimax-m2-cuda/bin/strata-minimax-m2-bench.exe',
                                ROOT/'build-local/cuda-13.0', args.out/'native.log', context=2048, batch=16,
                                gpu_cache_mib=18432, pipeline_readers=2, pipeline_chunk_mib=4)
        svc = Service(engine, tok, MiniMaxAPITemplate(ROOT/'serve/fixtures/minimax_m27_chat_template.jinja'), model_name='minimax-m2.7')
        svc.api_monitor = True
        base = make_handler(svc)
        class ObservedHandler(base):
            def _openai(self, req):
                evidence['requests'].append({'body': deepcopy(req), 'session_header': self.headers.get('X-Strata-Session-Id')})
                record()
                return super()._openai(req)
        server = Server(('127.0.0.1', 0), ObservedHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        ready = {'url': 'http://127.0.0.1:'+str(server.server_port), 'pid': os.getpid(), 'native_header': engine.header}
        save(args.out/'ready.json', ready)
        print(ready['url'], flush=True)
        end = time.monotonic()+1800
        while time.monotonic() < end and not (args.out/'STOP').exists():
            time.sleep(.2)
        evidence['stop_reason'] = 'STOP file' if (args.out/'STOP').exists() else '30 minute deadline'
    finally:
        if server:
            server.shutdown()
        if engine:
            engine.close()
        if server:
            server.server_close()
        if thread:
            thread.join(5)
        if engine:
            evidence['engine_alive_after_close'] = engine.alive()
        record()


if __name__ == '__main__':
    main()
