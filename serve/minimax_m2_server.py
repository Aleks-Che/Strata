"""Experimental MiniMax server using the isolated native JSONL engine."""
import argparse
from pathlib import Path

from serve.minimax_m2_api import MiniMaxAPITemplate
from serve.minimax_m2_engine import MiniMaxEngine, ROOT
from serve.minimax_m2_tools import _schemas
from serve.server import Server, Service, make_handler
from tools.strata_tokenizer import Tokenizer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gguf', type=Path, required=True)
    p.add_argument('--engine', type=Path, default=ROOT/'build-local/minimax-m2-cuda/bin/strata-minimax-m2-bench.exe')
    p.add_argument('--cuda-root', type=Path, default=ROOT/'build-local/cuda-13.0')
    p.add_argument('--log', type=Path, default=ROOT/'build-local/minimax-m2-server.stderr.log')
    p.add_argument('--ctx', type=int, default=512)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--gpu-cache-mib', type=int, default=0)
    p.add_argument('--pipeline-readers', type=int, default=0)
    p.add_argument('--pipeline-chunk-mib', type=int, default=8)
    p.add_argument('--pipeline-events', type=int, choices=(0, 2), default=0,
                   help='matrix events and async compute (2); requires readers and the reviewed events executable')
    p.add_argument('--prefix-cache', action='store_true', help='reuse complete prefill batches for explicit sessions')
    p.add_argument('--session-cache-mib', type=int, default=0, help='RAM cap for inactive session KV; requires --prefix-cache')
    p.add_argument('--session-cache-slots', type=int, default=4, help='maximum inactive session checkpoints (1..64)')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8080)
    p.add_argument('--api-key', default='')
    args = p.parse_args()
    if args.host != '127.0.0.1' and not args.api_key:
        p.error('a bind beyond 127.0.0.1 requires --api-key')
    if args.pipeline_events and not args.pipeline_readers:
        p.error('--pipeline-events requires --pipeline-readers')
    # Fail before loading CUDA weights if the isolated tools profile is incomplete.
    _schemas([{'type': 'function', 'function': {'name': 'startup_check',
               'parameters': {'type': 'object', 'properties': {}}}}])
    tok = Tokenizer.from_gguf(args.gguf)
    template = MiniMaxAPITemplate(ROOT/'serve/fixtures/minimax_m27_chat_template.jinja')
    template.resolve_stop_ids(tok)
    engine = MiniMaxEngine(args.gguf, args.engine, args.cuda_root, args.log, context=args.ctx, batch=args.batch,
                          gpu_cache_mib=args.gpu_cache_mib, pipeline_readers=args.pipeline_readers,
                          pipeline_chunk_mib=args.pipeline_chunk_mib, pipeline_events=args.pipeline_events, prefix_cache=args.prefix_cache,
                          session_cache_mib=args.session_cache_mib, session_cache_slots=args.session_cache_slots)
    server = None
    try:
        svc = Service(engine, tok, template, model_name='minimax-m2.7')
        svc.api_key = args.api_key
        server = Server((args.host, args.port), make_handler(svc))
        print('Experimental MiniMax API at http://'+args.host+':'+str(server.server_address[1]), flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        engine.close()
        if server:
            server.server_close()


if __name__ == '__main__':
    main()
