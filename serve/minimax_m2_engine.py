"""Resident Engine adapter for the reviewed MiniMax JSONL CUDA executable."""
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

from serve.minimax_m2_api import validate_sampling
from serve.server import EngineDied, EngineStuck
from serve.winjob import contain
from tools.inspect_minimax_m2_gguf import inspect_model
from tools.minimax_m2_loader_contract import LOADER_SHA
from tools.run_minimax_m2 import runtime_environment

ROOT = Path(__file__).resolve().parents[1]
EXE_SHA256 = '98e85e80e2dbaff5dc38f03f1ab4039835fb8d7e384f4491e1d700c989c5d3d8'
HEADER_SHA256 = '9a011b2ad6fb20db3bb0cdfe9af08665939ee1f0cb0609ffe2bfee7f31c825db'
MAX_LINE = 16*1024*1024


class NativeRequestError(ValueError):
    """A terminal native error; unlike a protocol error, the process is reusable."""


def check(ok, message):
    if not ok:
        raise ValueError('MiniMax: '+message)


class MiniMaxEngine:
    can_session_id = False
    can_cache_admin = False

    def __init__(self, model, binary, cuda_root, log_path, *, context=512, batch=8,
                 gpu_cache_mib=0, pipeline_readers=0, pipeline_chunk_mib=8,
                 prefix_cache=False, session_cache_mib=0, session_cache_slots=4,
                 startup_timeout=180, request_timeout=900, drain_timeout=30):
        check(type(context) is int and 256 <= context <= 4096, 'context must be 256..4096')
        check(type(batch) is int and 1 <= batch <= 16, 'batch must be 1..16')
        check(type(gpu_cache_mib) is int and 0 <= gpu_cache_mib <= 131072, 'invalid GPU cache cap')
        check(type(pipeline_readers) is int and pipeline_readers in (0, 1, 2), 'invalid pipeline readers')
        check(type(pipeline_chunk_mib) is int and pipeline_chunk_mib in (4, 8, 16), 'invalid pipeline chunk')
        check(type(prefix_cache) is bool, 'prefix_cache must be boolean')
        check(type(session_cache_mib) is int and 0 <= session_cache_mib <= 131072, 'invalid session cache cap')
        check(type(session_cache_slots) is int and 1 <= session_cache_slots <= 64, 'invalid session cache slots')
        check(not session_cache_mib or prefix_cache, 'session cache requires prefix_cache')
        check(all(type(t) in (int, float) and math.isfinite(t) and t > 0
                  for t in (startup_timeout, request_timeout, drain_timeout)), 'invalid timeout')
        self.model, self.binary = Path(model).resolve(), Path(binary).resolve()
        self.env = runtime_environment(cuda_root)
        self.env['STRATA_MM27_TOKENWISE'] = '0'
        self.env['PYTHONIOENCODING'] = 'utf-8'
        self.log_path = str(Path(log_path).resolve())
        self.options = {'context': context, 'batch': batch, 'gpu_cache_mib': gpu_cache_mib,
                        'pipeline_readers': pipeline_readers, 'pipeline_chunk_mib': pipeline_chunk_mib,
                        'prefix_cache': prefix_cache, 'session_cache_mib': session_cache_mib,
                        'session_cache_slots': session_cache_slots}
        self.can_session_id = prefix_cache
        self.startup_timeout, self.request_timeout, self.drain_timeout = startup_timeout, request_timeout, drain_timeout
        self._gate = threading.Lock()
        self._closing = threading.Event()
        self._sequence = 0
        self.proc = None
        self._log = None
        self._reader = None
        self.max_context = 0
        self.info, self.last, self.last_result = {}, None, None
        self.last_error, self.header, self.progress = None, None, None
        self.restart()

    def _admit(self):
        check(hashlib.sha256(self.binary.read_bytes()).hexdigest() == EXE_SHA256, 'unreviewed executable SHA-256')
        inventory = inspect_model(self.model)
        check(inventory['header_sha256'] == HEADER_SHA256, 'unreviewed model header')
        flags = {'startupinfo': self._hidden()} if os.name == 'nt' else {}
        version = json.loads(subprocess.check_output([str(self.binary), '--version'], env=self.env, timeout=20, **flags))
        check(version.get('architecture') == 'minimax-m2' and version.get('source_revision') == LOADER_SHA,
              'unreviewed executable provenance')
        self.version = version
        args = ['--gguf', str(self.model), '--pipe', '--ctx', str(self.options['context']), '--batch', str(self.options['batch']),
                '--mode', '2', '--gpu-cache-mib', str(self.options['gpu_cache_mib']), '--gpu-cache-allocator', 'arena',
                '--pipeline-readers', str(self.options['pipeline_readers']), '--pipeline-chunk-mib', str(self.options['pipeline_chunk_mib']),
                '--pipeline-lookahead', str(int(self.options['pipeline_readers'] > 0)),
                '--pipeline-d2d-batch', str(int(self.options['pipeline_readers'] > 0)),
                '--prefix-cache', str(int(self.options['prefix_cache'])),
                '--session-cache-mib', str(self.options['session_cache_mib']),
                '--session-cache-slots', str(self.options['session_cache_slots'])]
        return [str(self.binary), *args]

    @staticmethod
    def _hidden():
        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = subprocess.SW_HIDE
        return startup

    def _validate_ready(self, event):
        check(event.get('event') == 'ready' and event.get('request_id') is None, 'missing ready event')
        expected = {**self.version, **self.options, 'mode': 2, 'kv': 'F32', 'strict_f32': True,
                    'flash_attention': False, 'graphs': False, 'mtp': False, 'ram_cache_mib': 0,
                    'gpu_cache_allocator': 'arena', 'expert_reader': 'file', 'cache_group_experts': False,
                    'arena_block_mib': 64, 'arena_growth_reserve_mib': 0,
                    'pipeline_lookahead': self.options['pipeline_readers'] > 0,
                    'pipeline_d2d_batch': self.options['pipeline_readers'] > 0}
        check(all(type(event.get(k)) is type(v) and event[k] == v for k, v in expected.items()), 'ready config/provenance mismatch')
        check(Path(event['model']).resolve() == self.model, 'ready model mismatch')
        for key in ('memory_before', 'memory_loaded'):
            m = event[key]
            check(all(m[axis+'_total'] > 0 and 0 <= m[axis+'_total']-m[axis+'_available'] <= .95*m[axis+'_total']
                      for axis in ('ram', 'vram')), 'global memory exceeds 95%')

    def _pump(self, proc, inbox, stop):
        try:
            while not stop.is_set():
                line = proc.stdout.readline(MAX_LINE+1)
                if not line:
                    event = {'event': 'transport_error', 'message': 'supervisor EOF'}
                elif len(line) > MAX_LINE or not line.endswith(b'\n'):
                    event = {'event': 'transport_error', 'message': 'invalid supervisor line'}
                else:
                    try:
                        event = json.loads(line)
                        if not isinstance(event, dict):
                            raise ValueError('non-object event')
                    except (ValueError, UnicodeError) as exc:
                        event = {'event': 'transport_error', 'message': str(exc)}
                while not stop.is_set():
                    try:
                        inbox.put(event, timeout=.1)
                        break
                    except queue.Full:
                        pass
                if event.get('event') == 'transport_error':
                    return
        except (OSError, ValueError):
            if not stop.is_set():
                try:
                    inbox.put({'event': 'transport_error', 'message': 'supervisor read failed'}, timeout=.1)
                except queue.Full:
                    pass

    def _send(self, item):
        if not self.proc or self.proc.poll() is not None:
            raise EngineDied('MiniMax supervisor is not running')
        try:
            self.proc.stdin.write((json.dumps(item, allow_nan=False)+'\n').encode())
            self.proc.stdin.flush()
        except (OSError, ValueError) as exc:
            raise EngineDied('MiniMax supervisor write failed') from exc

    def _event(self, timeout):
        try:
            event = self._inbox.get(timeout=timeout)
        except queue.Empty:
            return None
        if event.get('event') == 'transport_error':
            raise EngineDied('MiniMax: '+event.get('message', 'transport error'))
        return event

    def _shutdown(self):
        self.max_context = 0
        proc = self.proc
        if not proc:
            return
        self._reader_stop.set()
        try:
            if proc.poll() is None:
                try:
                    self._send({'command': 'quit'})
                    proc.wait(timeout=5)
                except (EngineDied, subprocess.TimeoutExpired):
                    proc.kill(); proc.wait(timeout=10)
            proc.stdin.close()
            if self._reader:
                self._reader.join(5)
            proc.stdout.close()
        finally:
            self.proc = None
            if self._log:
                self._log.close(); self._log = None

    def restart(self):
        with self._gate:
            check(not self._closing.is_set(), 'engine is closing')
            self._shutdown()
            command = self._admit()
            self.command = command
            Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
            self._log = open(self.log_path, 'ab', buffering=0)
            self._inbox, self._reader_stop = queue.Queue(maxsize=64), threading.Event()
            flags = {'creationflags': subprocess.CREATE_NEW_CONSOLE, 'startupinfo': self._hidden()} if os.name == 'nt' else {}
            try:
                self.proc = subprocess.Popen([sys.executable, '-u', '-m', 'serve.minimax_m2_worker', '--', *command],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log, cwd=ROOT, env=self.env, **flags)
                contain(self.proc)
                self._reader = threading.Thread(target=self._pump, args=(self.proc, self._inbox, self._reader_stop), daemon=True)
                self._reader.start()
                header = self._event(self.startup_timeout)
                check(header is not None, 'startup timeout')
                self._validate_ready(header)
                self.header = header
                self.max_context = header['context']
                self.info = {'version': 'minimax-m2/'+LOADER_SHA, 'architecture': 'minimax-m2', 'sessions': False,
                             'prefix_cache': self.options['prefix_cache'], 'mtp': False,
                             'session_cache_mib': self.options['session_cache_mib'],
                             'session_cache_slots': self.options['session_cache_slots']}
                self.last = self.last_result = self.last_error = None
            except BaseException:
                self._shutdown()
                raise

    def alive(self):
        return self.proc is not None and self.proc.poll() is None and self.max_context > 0 and self._reader.is_alive()

    def unload(self):
        if not self._gate.acquire(blocking=False):
            raise EngineStuck('MiniMax request still owns the engine')
        try:
            self._shutdown()
        finally:
            self._gate.release()

    def close(self):
        """Cancel active work before shutting down the server's owned process."""
        self._closing.set()
        if not self._gate.acquire(timeout=self.drain_timeout+5):
            if self.proc and self.proc.poll() is None:
                self.proc.kill()
                self.proc.wait(timeout=10)
            raise EngineStuck('MiniMax consumer did not release the engine during shutdown')
        try:
            self._shutdown()
        finally:
            self._gate.release()

    def _terminal(self, event, received, ids, limit, allow_reuse=False):
        if event['event'] == 'error':
            self.last_error = event
            self.last = {'generated': len(received), 'reused': 0}
            return
        check(all(type(event.get(k)) is int for k in ('generated_tokens', 'prompt_tokens', 'max_tokens', 'decode_forward_tokens')) and
              event.get('token_ids') == received and event.get('generated_tokens') == len(received) and
              event.get('prompt_tokens') == len(ids) and event.get('max_tokens') == limit, 'terminal counts/IDs mismatch')
        check(0 < len(received) <= limit and event.get('decode_forward_tokens') == len(received)-1, 'invalid native token accounting')
        check(all(type(event.get(k)) is int for k in ('reused_tokens', 'evaluated_prompt_tokens', 'kv_tokens')) and
              0 <= event['reused_tokens'] < len(ids) and event['reused_tokens'] % self.options['batch'] == 0 and
              event['evaluated_prompt_tokens'] == len(ids)-event['reused_tokens'] and
              event['kv_tokens'] == len(ids)+len(received)-1 and
              (event['reused_tokens'] == 0 or self.can_session_id and allow_reuse), 'invalid native prefix accounting')
        archive_fields = ('session_saved_bytes', 'session_restored_bytes', 'session_archive_bytes',
                          'session_archive_entries', 'session_archive_evictions', 'session_archive_rejected')
        cap = self.options['session_cache_mib']*1024*1024
        check(all(type(event.get(k)) is int and event[k] >= 0 for k in archive_fields) and
              max(event['session_archive_bytes'], event['session_saved_bytes'], event['session_restored_bytes']) <= cap and
              event['session_archive_entries'] <= self.options['session_cache_slots'] and
              (event['session_archive_entries'] == 0) == (event['session_archive_bytes'] == 0) and
              type(event.get('session_restore')) is bool and
              event['session_restore'] == (event['session_restored_bytes'] > 0) and
              (not event['session_restore'] or allow_reuse and event['reused_tokens'] > 0) and
              (cap > 0 or all(event[k] == 0 for k in archive_fields)), 'invalid native session accounting')
        eos = received[-1] == 200020
        check(event.get('stop_reason') == ('eos' if eos else 'length') and
              event.get('stop_token_id') == (200020 if eos else None) and
              (200020 not in received[:-1]) and (eos or len(received) == limit), 'invalid native stop')
        check(all(type(event.get(k)) in (int, float) and math.isfinite(event[k]) and event[k] >= 0
                  for k in ('prefill_ms', 'decode_ms', 'request_ms', 'session_ms')), 'invalid native timings')
        hits, misses = event['decode']['cache_hits'], event['decode']['cache_misses']
        check(all(type(n) is int and n >= 0 for n in (hits, misses)), 'invalid native cache counters')
        self.last_result = event
        self.last = {'generated': len(received), 'reused': event['reused_tokens'], 'prompt_ms': event['prefill_ms'], 'decode_ms': event['decode_ms'],
                     'hits': hits, 'lookups': hits+misses}

    def generate(self, ids, max_new, sampling, cancel, embeddings=None, session_id=None):
        check(embeddings is None, 'media is unsupported')
        session_key = None
        if session_id is not None:
            check(self.can_session_id, 'session reuse requires prefix_cache')
            check(isinstance(session_id, str) and bool(session_id) and len(session_id.encode('utf-8')) <= 256 and
                  not any(ord(c) < 32 or ord(c) == 127 for c in session_id), 'invalid session ID')
            session_key = hashlib.sha256(session_id.encode('utf-8')).hexdigest()
        check(isinstance(ids, list) and ids and all(type(t) is int and 0 <= t < 200064 for t in ids), 'invalid token IDs')
        check(type(max_new) is int and max_new > 0 and len(ids)+max_new <= self.max_context, 'request exceeds context')
        validate_sampling(sampling or {})
        sample = {k: sampling[k] for k in ('temperature', 'top_p', 'top_k', 'seed') if k in (sampling or {})}
        if not self._gate.acquire(blocking=False):
            raise EngineStuck('concurrent MiniMax generation is unsupported')
        received, terminal, cancelled = [], False, False
        self.last_result = self.last_error = None
        self.last = {'generated': 0, 'reused': 0}
        started = time.monotonic()
        deadline = started+self.request_timeout
        self._sequence += 1
        request_id = self._sequence

        def interrupt():
            nonlocal cancelled, deadline
            if not cancelled:
                self._send({'command': 'cancel', 'request_id': request_id})
                cancelled = True
                deadline = time.monotonic()+self.drain_timeout

        def consume(event):
            nonlocal terminal
            check(event.get('request_id') == request_id, 'stale request ID')
            kind = event.get('event')
            if kind == 'token':
                token = event.get('id')
                check(type(token) is int and 0 <= token < 200064 and len(received) < max_new and
                      (not received or received[-1] != 200020), 'invalid or excess token event')
                received.append(token)
                return token
            check(kind in ('result', 'error'), 'unexpected request event')
            self._terminal(event, received, ids, max_new, allow_reuse=session_key is not None)
            terminal = True
            if kind == 'error' and not cancelled and not cancel.is_set():
                raise NativeRequestError('MiniMax native error: '+str(event.get('message')))
            return None

        try:
            if cancel.is_set() or self._closing.is_set():
                terminal = True  # no native request was submitted
                return
            native_request = {'tokens': ids, 'max_tokens': max_new, 'sampling': sample}
            if session_key is not None:
                native_request['session_key'] = session_key
            self._send({'command': 'generate', 'request_id': request_id, 'request': native_request})
            while not terminal:
                if cancel.is_set() or self._closing.is_set():
                    interrupt()
                if time.monotonic() >= deadline:
                    raise EngineDied('MiniMax request/cancel timeout')
                event = self._event(.25)
                if event is None:
                    yield None
                else:
                    token = consume(event)
                    if token is not None and not cancelled:
                        yield token
        except GeneratorExit:
            raise
        except NativeRequestError:
            raise
        except (EngineDied, ValueError, KeyError, TypeError) as exc:
            self._shutdown()
            raise EngineDied(str(exc)) from exc
        finally:
            try:
                if not terminal and self.proc is not None:
                    # EOS/limit already finishes natively; only early closes need
                    # cancellation. Drain under the request lock in all cases.
                    if not received or received[-1] != 200020 and len(received) < max_new:
                        interrupt()
                    end = time.monotonic()+self.drain_timeout
                    while not terminal:
                        if time.monotonic() >= end:
                            raise EngineDied('MiniMax terminal drain timed out')
                        event = self._event(.1)
                        if event is not None:
                            consume(event)
            except Exception as exc:
                self._shutdown()
                raise EngineDied(str(exc)) from exc
            finally:
                self._gate.release()
