"""Real supervisor/pipe/signal tests with a small deterministic child process."""
import ctypes
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest

from serve.minimax_m2_engine import MiniMaxEngine, NativeRequestError, ROOT, check
from serve.server import EngineDied, EngineStuck


class FixtureEngine(MiniMaxEngine):
    def _admit(self):
        return [sys.executable, '-u', str(ROOT/'serve/fixtures/minimax_jsonl_child.py'), self.model.name]

    def _validate_ready(self, event):
        check(event.get('event') == 'ready' and event.get('request_id') is None and event.get('context') == 512,
              'fixture ready mismatch')


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.engines = []

    def tearDown(self):
        for engine in self.engines:
            engine.close()
        self.tmp.cleanup()

    def engine(self, **kwargs):
        engine = FixtureEngine(kwargs.pop('model', 'normal'), 'unused', ROOT, Path(self.tmp.name)/'native.log',
                               startup_timeout=5, drain_timeout=2, **kwargs)
        self.engines.append(engine)
        return engine

    @staticmethod
    def run_request(engine, mode=0, count=4, cancel=None):
        return [t for t in engine.generate([mode], count, {}, cancel or threading.Event()) if t is not None]

    def test_length_eos_native_error_recovery_and_restart(self):
        e = self.engine()
        self.assertEqual(self.run_request(e), [100, 101, 102, 103])
        self.assertEqual(e.last['generated'], 4)
        self.assertEqual(self.run_request(e, 1), [100, 200020])
        self.assertEqual(e.last_result['request_id'], 2)
        with self.assertRaisesRegex(NativeRequestError, 'fixture native failure'):
            self.run_request(e, 4)
        self.assertTrue(e.alive())
        self.assertIsNone(e.last_result)
        self.assertEqual(self.run_request(e), [100, 101, 102, 103])
        old = e.proc
        e.unload()
        self.assertIsNotNone(old.poll())
        self.assertFalse(e.alive())
        e.restart()
        self.assertEqual(self.run_request(e), [100, 101, 102, 103])

    def test_early_generator_close_cancels_drains_and_recovers(self):
        e = self.engine()
        gen = e.generate([6], 100, {}, threading.Event())
        self.assertEqual(next(gen), 100)
        gen.close()
        self.assertIn('cancelled', e.last_error['message'])
        self.assertTrue(e.alive())
        self.assertEqual(self.run_request(e, 1), [100, 200020])

    def test_close_after_eos_or_limit_drains_result_without_cancellation(self):
        e = self.engine()
        for mode, count in [(1, 4), (0, 2)]:
            gen = e.generate([mode], count, {}, threading.Event())
            received = []
            while len(received) < 2:
                t = next(gen)
                if t is not None:
                    received.append(t)
            gen.close()
            self.assertEqual(e.last_result['token_ids'], received)
            self.assertIsNone(e.last_error)
        self.assertEqual(self.run_request(e), [100, 101, 102, 103])

    def test_cancel_during_prefill_and_already_cancelled(self):
        e = self.engine()
        cancel = threading.Event()
        gen = e.generate([5], 4, {}, cancel)
        self.assertIsNone(next(gen))
        cancel.set()
        self.assertEqual([t for t in gen if t is not None], [])
        self.assertIn('cancelled', e.last_error['message'])
        request_id = e.last_error['request_id']
        self.assertEqual(self.run_request(e, cancel=cancel), [])
        self.assertIsNone(e.last_error)
        self.assertEqual(self.run_request(e), [100, 101, 102, 103])
        self.assertGreater(e.last_result['request_id'], request_id)

    def test_unresponsive_cancel_retires_child_then_restart(self):
        e = self.engine()
        e.drain_timeout = .3
        old = e.proc
        cancel = threading.Event()
        gen = e.generate([7], 4, {}, cancel)
        self.assertIsNone(next(gen))
        cancel.set()
        with self.assertRaisesRegex(EngineDied, 'timeout'):
            list(gen)
        self.assertFalse(e.alive())
        self.assertIsNotNone(old.poll())
        e.restart()
        self.assertEqual(self.run_request(e), [100, 101, 102, 103])

    def test_request_deadline_retires_child(self):
        e = self.engine(request_timeout=.3)
        with self.assertRaisesRegex(EngineDied, 'timeout'):
            self.run_request(e, 7)
        self.assertFalse(e.alive())

    def test_ready_admission_provenance_config_and_memory(self):
        from copy import deepcopy
        e = self.engine()
        e.version = {'architecture': 'minimax-m2', 'source_revision': 'fixture', 'patches': 'fixture'}
        good = {**e.version, **e.options, 'event': 'ready', 'request_id': None, 'model': str(e.model),
                'mode': 2, 'kv': 'F32', 'strict_f32': True, 'flash_attention': False, 'graphs': False, 'mtp': False,
                'ram_cache_mib': 0, 'gpu_cache_allocator': 'arena', 'expert_reader': 'file', 'cache_group_experts': False,
                'arena_block_mib': 64, 'arena_growth_reserve_mib': 0, 'pipeline_lookahead': False, 'pipeline_d2d_batch': False,
                'memory_before': {'ram_total': 100, 'ram_available': 5, 'vram_total': 100, 'vram_available': 5},
                'memory_loaded': {'ram_total': 100, 'ram_available': 5, 'vram_total': 100, 'vram_available': 5}}
        MiniMaxEngine._validate_ready(e, good)
        for key, value in [('context', 2048), ('strict_f32', 1), ('source_revision', 'wrong'), ('model', 'wrong')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                MiniMaxEngine._validate_ready(e, {**good, key: value})
        for stage in ('memory_before', 'memory_loaded'):
            for axis in ('ram', 'vram'):
                bad = deepcopy(good)
                bad[stage][axis+'_available'] = 4
                with self.subTest(stage=stage, axis=axis), self.assertRaises(ValueError):
                    MiniMaxEngine._validate_ready(e, bad)

    def test_prefix_session_opt_in_digest_and_anonymous_request(self):
        import hashlib
        from unittest.mock import patch
        e = self.engine(prefix_cache=True)
        self.assertTrue(e.can_session_id)
        self.assertFalse(e.can_cache_admin)
        with patch.object(e, '_send', wraps=e._send) as send:
            list(e.generate([0], 4, {}, threading.Event(), session_id='диалог-1'))
            self.assertEqual(send.call_args.args[0]['request']['session_key'],
                             hashlib.sha256('диалог-1'.encode()).hexdigest())
            list(e.generate([0], 4, {}, threading.Event()))
            self.assertNotIn('session_key', send.call_args.args[0]['request'])
        for value in ('', 'x'*257, 'я'*129, '\n', '\x7f', 1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                list(e.generate([0], 4, {}, threading.Event(), session_id=value))
        disabled = self.engine()
        with self.assertRaisesRegex(ValueError, 'requires prefix_cache'):
            list(disabled.generate([0], 4, {}, threading.Event(), session_id='a'))

    def test_prefix_terminal_accounting_and_capability(self):
        e = self.engine(prefix_cache=True)
        self.run_request(e)
        good = {**e.last_result, 'prompt_tokens': 17, 'reused_tokens': 16, 'evaluated_prompt_tokens': 1, 'kv_tokens': 20}
        e._terminal(good, good['token_ids'], [0]*17, 4, allow_reuse=True)
        self.assertEqual(e.last['reused'], 16)
        for key, value in [('reused_tokens', True), ('reused_tokens', 17), ('reused_tokens', 7),
                           ('reused_tokens', -8), ('evaluated_prompt_tokens', 2), ('kv_tokens', 21)]:
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, 'prefix accounting'):
                e._terminal({**good, key: value}, good['token_ids'], [0]*17, 4, allow_reuse=True)
        with self.assertRaisesRegex(ValueError, 'prefix accounting'):
            e._terminal(good, good['token_ids'], [0]*17, 4)
        e.can_session_id = False
        with self.assertRaisesRegex(ValueError, 'prefix accounting'):
            e._terminal(good, good['token_ids'], [0]*17, 4, allow_reuse=True)

    def test_session_archive_limits_and_terminal_accounting(self):
        for opts in ({'session_cache_mib': 1}, {'session_cache_mib': True}, {'session_cache_mib': -1},
                     {'session_cache_mib': 131073}, {'session_cache_slots': 0}, {'session_cache_slots': 65},
                     {'session_cache_slots': True}):
            with self.subTest(opts=opts), self.assertRaises(ValueError):
                self.engine(**opts)
        e = self.engine(prefix_cache=True, session_cache_mib=16, session_cache_slots=2)
        self.run_request(e)
        good = {**e.last_result, 'prompt_tokens': 17, 'reused_tokens': 16, 'evaluated_prompt_tokens': 1, 'kv_tokens': 20,
                'session_restore': True, 'session_restored_bytes': 1024, 'session_saved_bytes': 1024,
                'session_archive_entries': 1, 'session_archive_bytes': 1536}
        e._terminal(good, good['token_ids'], [0]*17, 4, allow_reuse=True)
        for key, value in [('session_restore', 1), ('session_restored_bytes', 0), ('session_saved_bytes', 17*1024*1024),
                           ('session_archive_bytes', 0), ('session_archive_entries', 3), ('session_archive_evictions', -1),
                           ('session_archive_rejected', True), ('session_ms', float('nan'))]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                e._terminal({**good, key:value}, good['token_ids'], [0]*17, 4, allow_reuse=True)
        e.options['session_cache_mib'] = 0
        with self.assertRaisesRegex(ValueError, 'session accounting'):
            e._terminal(good, good['token_ids'], [0]*17, 4, allow_reuse=True)

    def test_malformed_eof_accounting_and_tokens_retire_process(self):
        e = self.engine()
        for mode in (2, 3, 8, 9):
            with self.subTest(mode=mode), self.assertRaises(EngineDied):
                self.run_request(e, mode)
            self.assertFalse(e.alive())
            e.restart()
            self.assertEqual(self.run_request(e), [100, 101, 102, 103])

    def test_stale_request_id_retires_process(self):
        e = self.engine()
        e._inbox.put({'event': 'token', 'id': 100, 'request_id': 999})
        with self.assertRaisesRegex(EngineDied, 'stale request ID'):
            self.run_request(e)
        self.assertFalse(e.alive())

    def test_concurrent_generate_and_unload_rejected(self):
        e = self.engine()
        gen = e.generate([6], 100, {}, threading.Event())
        self.assertEqual(next(gen), 100)
        with self.assertRaises(EngineStuck):
            self.run_request(e)
        with self.assertRaises(EngineStuck):
            e.unload()
        gen.close()
        self.assertEqual(self.run_request(e), [100, 101, 102, 103])

    def test_server_close_cancels_active_request(self):
        e = self.engine()
        gen = e.generate([5], 4, {}, threading.Event())
        self.assertIsNone(next(gen))
        errors = []
        def consume():
            try:
                list(gen)
            except Exception as exc:
                errors.append(exc)
        thread = threading.Thread(target=consume)
        thread.start()
        e.close()
        thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertFalse(e.alive())

    def test_input_rejections_do_not_poison_process(self):
        e = self.engine()
        for ids, budget, sampling in [([], 1, {}), ([True], 1, {}), ([200064], 1, {}),
                                     ([1], 512, {}), ([1], 1, {'temperature': float('nan')})]:
            with self.subTest(ids=ids, budget=budget, sampling=sampling), self.assertRaises(ValueError):
                list(e.generate(ids, budget, sampling, threading.Event()))
        self.assertEqual(self.run_request(e), [100, 101, 102, 103])

    def test_invalid_ready_and_unreviewed_binary_rejected(self):
        with self.assertRaises(EngineDied):
            self.engine(model='badready')
        binary = Path(self.tmp.name)/'fake.exe'
        binary.write_bytes(b'unreviewed')
        with self.assertRaisesRegex(ValueError, 'unreviewed executable'):
            MiniMaxEngine('unused', binary, ROOT, Path(self.tmp.name)/'log')

    @unittest.skipUnless(os.name == 'nt', 'Windows job containment')
    def test_supervisor_kill_does_not_orphan_native_child(self):
        e = self.engine()
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x100000, False, e.header['native_pid'])
        self.assertTrue(handle)
        try:
            e.proc.kill()
            e.proc.wait(timeout=5)
            self.assertEqual(kernel.WaitForSingleObject(handle, 5000), 0)
        finally:
            kernel.CloseHandle(handle)


if __name__ == '__main__':
    unittest.main()
