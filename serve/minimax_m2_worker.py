"""Private JSONL supervisor. A hidden Windows console isolates CTRL_BREAK.

One native request at a time; cancellation is addressed by request ID. Native
stdout is consumed through its terminal record before admitting the next one.
This module launches only the explicit child command supplied by its parent.
"""
import ctypes
import json
import os
import signal
import subprocess
import sys
import threading

from serve.winjob import contain

MAX_LINE = 16*1024*1024


def main():
    command = sys.argv[1:]
    if command[:1] == ['--']:
        command = command[1:]
    if not command:
        raise ValueError('missing native command')
    if os.name == 'nt' and not ctypes.windll.kernel32.GetConsoleCP():
        raise RuntimeError('MiniMax supervisor requires its private console')
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0)
    lock = threading.RLock()
    state = {'id': None, 'ready': False, 'failed': False, 'cancelled': False}
    reader = None

    def emit(event):
        with lock:
            sys.stdout.write(json.dumps(event, ensure_ascii=True, allow_nan=False)+'\n')
            sys.stdout.flush()

    def fail(message):
        with lock:
            state['failed'] = True
            emit({'event': 'transport_error', 'request_id': state['id'], 'message': message})

    def cancel(request_id):
        with lock:
            if state['id'] != request_id or state['cancelled'] or proc.poll() is not None:
                return
            state['cancelled'] = True
            if os.name == 'nt':
                if not ctypes.windll.kernel32.GenerateConsoleCtrlEvent(1, proc.pid):
                    raise ctypes.WinError()
            else:
                proc.send_signal(signal.SIGINT)

    def pump():
        try:
            while True:
                line = proc.stdout.readline(MAX_LINE+1)
                if not line:
                    fail('native EOF')
                    return
                if len(line) > MAX_LINE or not line.endswith(b'\n'):
                    raise ValueError('oversized or unterminated native event')
                event = json.loads(line)
                with lock:
                    if not isinstance(event, dict) or 'request_id' in event:
                        raise ValueError('invalid native envelope')
                    kind = event.get('event')
                    if kind == 'ready':
                        if state['ready'] or state['id'] is not None:
                            raise ValueError('unexpected native ready')
                        state['ready'] = True
                        emit({**event, 'request_id': None, 'native_pid': proc.pid})
                    elif kind in ('token', 'result', 'error') and state['id'] is not None:
                        event['request_id'] = state['id']
                        emit(event)
                        if kind in ('result', 'error'):
                            state['id'] = None
                    else:
                        raise ValueError('unsolicited native event')
        except Exception as exc:
            fail(str(exc))

    try:
        if os.name == 'nt' and not contain(proc):
            raise RuntimeError('cannot contain native child in Windows job')
        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        for line in sys.stdin.buffer:
            if len(line) > MAX_LINE:
                raise ValueError('oversized supervisor command')
            item = json.loads(line)
            with lock:
                if state['failed']:
                    break
                kind = item.get('command')
                if kind == 'quit':
                    break
                request_id = item.get('request_id')
                if type(request_id) is not int or request_id < 1:
                    raise ValueError('invalid supervisor request ID')
                if kind == 'cancel':
                    cancel(request_id)
                elif kind == 'generate':
                    if not state['ready'] or state['id'] is not None:
                        raise ValueError('native process is not idle and ready')
                    state['id'], state['cancelled'] = request_id, False
                    proc.stdin.write((json.dumps(item['request'], ensure_ascii=True, allow_nan=False)+'\n').encode())
                    proc.stdin.flush()
                else:
                    raise ValueError('unknown supervisor command')
    except Exception as exc:
        fail(str(exc))
        return 1
    finally:
        # Parent EOF/quit must not leave a resident CUDA child behind.
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=15)
        proc.stdin.close()
        if reader:
            reader.join(5)
        proc.stdout.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
