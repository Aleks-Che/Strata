"""Small native-protocol stand-in for subprocess lifecycle tests; no model/GPU."""
import json
import signal
import sys
import time

cancelled = False


def interrupt(*_):
    global cancelled
    cancelled = True


signal.signal(signal.SIGINT, interrupt)
if hasattr(signal, 'SIGBREAK'):
    signal.signal(signal.SIGBREAK, interrupt)


def emit(item):
    print(json.dumps(item), flush=True)


if sys.argv[1] == 'badready':
    print('not JSON', flush=True)
    time.sleep(10)
    sys.exit(0)
emit({'event': 'ready', 'context': 512})
for line in sys.stdin:
    cancelled = False
    req = json.loads(line)
    mode, limit = req['tokens'][0], req['max_tokens']
    if mode == 2:
        sys.exit(7)
    if mode == 3:
        print('invalid JSON', flush=True)
        continue
    if mode == 4:
        emit({'event': 'error', 'message': 'fixture native failure'})
        continue
    if mode in (5, 7):
        end = time.monotonic()+10
        while time.monotonic() < end and (not cancelled or mode == 7):
            time.sleep(.01)
    tokens = []
    for i in range(limit):
        if cancelled:
            break
        token = 200020 if mode == 1 and i == 1 or mode == 9 and i == 0 else 100+i
        tokens.append(token)
        emit({'event': 'token', 'id': token})
        if token == 200020 and mode != 9:
            break
        if mode == 6:
            time.sleep(.05)
    if cancelled:
        emit({'event': 'error', 'message': 'cancelled'})
        continue
    emit({'event': 'result', 'token_ids': tokens if mode != 8 else [999], 'generated_tokens': len(tokens),
          'prompt_tokens': len(req['tokens']), 'max_tokens': limit, 'decode_forward_tokens': len(tokens)-1,
          'reused_tokens': 0, 'evaluated_prompt_tokens': len(req['tokens']), 'kv_tokens': len(req['tokens'])+len(tokens)-1,
          'session_restore': False, 'session_ms': 0, 'session_saved_bytes': 0, 'session_restored_bytes': 0,
          'session_archive_bytes': 0, 'session_archive_entries': 0, 'session_archive_evictions': 0, 'session_archive_rejected': 0,
          'stop_reason': 'eos' if tokens[-1] == 200020 else 'length',
          'stop_token_id': 200020 if tokens[-1] == 200020 else None,
          'prefill_ms': 1, 'decode_ms': 2, 'request_ms': 3,
          'decode': {'cache_hits': 1, 'cache_misses': 2}})
