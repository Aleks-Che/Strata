"""Real-file runner contracts; scripted child outputs are not GPU evidence."""
import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools.check_glm5next_transfer import check, main, manifest, run_transport
from tools.glm5next_expert_plan import ExpertMatrix
from tools.test_setup_glm5next import model_fixture, write_shard


class TransferCheckTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.script = self.root / 'checker.py'
        self.matrices = [ExpertMatrix('main', 3, 0, p, 'IQ2_S', 256, 512,
                                      'weights with spaces.gguf', (1 << 33) + i * 100, 100)
                         for i, p in enumerate(('gate', 'up', 'down'))]
        self.output = 'GPU 46697874757265 13000 13000\nOK 0 100\nOK 1 100\nOK 2 100\nTOTAL 300 300 300 0 6\n'

    def emitter(self, output=None, exit_code=0, delay=0, expected=None):
        self.script.write_text('import sys,time\n'
                               'data=sys.stdin.buffer.read()\n'
                               f'expected={expected!r}\n'
                               'if expected is not None and data!=expected: sys.exit(9)\n'
                               f'time.sleep({delay!r})\n'
                               f'sys.stdout.write({self.output if output is None else output!r})\n'
                               f'sys.exit({exit_code})\n', encoding='utf-8')

    def run_check(self, mode='native', timeout=5):
        return run_transport([sys.executable, str(self.script)], self.matrices,
                             self.root, 64, mode, timeout)

    def test_manifest_exact_ranges_modes_and_unicode_paths(self):
        self.matrices[0] = replace(self.matrices[0], shard='модель.gguf')
        for mode, number in (('mmap', 0), ('native', 1), ('auto', 2)):
            expected = f'GLM_RANGES_V1 64 {number} 3\n'
            for m in self.matrices:
                expected += f'{m.file_offset} 100 {str((self.root / m.shard).resolve()).encode("utf-8").hex()}\n'
            self.assertEqual(manifest(self.matrices, self.root, 64, mode), expected.encode('ascii'))
        self.emitter(expected=manifest(self.matrices, self.root, 64, 'native'))
        result = self.run_check()
        self.assertEqual((result['gpu'], result['matrix_count'], result['chunks']), ('Fixture', 3, 6))

    def test_bad_results_never_pass(self):
        for output in ('', self.output+'OK 3 100\n', self.output.replace('OK 1 100\n', ''),
                       self.output.replace('OK 0 100', 'OK 1 100'),
                       self.output.replace('OK 0 100', 'OK 0 99'),
                       self.output.replace('GPU', 'CPU'), self.output.replace('46697874757265', 'f'),
                       self.output.replace('300 300 300 0 6', '299 300 300 0 6'),
                       self.output.replace('300 300 300 0 6', '300 300 299 0 6'),
                       self.output.replace('300 300 300 0 6', '300 300 300 0 5')):
            with self.subTest(output=output):
                self.emitter(output)
                with self.assertRaises(ValueError):
                    self.run_check()

    def test_path_specific_counters_and_auto(self):
        self.emitter()
        with self.assertRaises(ValueError):
            self.run_check('mmap')
        self.emitter(self.output.replace('300 300 300 0 6', '300 300 0 300 6'))
        self.assertEqual(self.run_check('mmap')['mmap_bytes'], 300)
        with self.assertRaises(ValueError):
            self.run_check('native')
        self.emitter(self.output.replace('300 300 300 0 6', '300 300 150 150 6'))
        self.assertEqual(self.run_check('auto')['native_bytes'], 150)

    def test_child_failure_timeout_and_invalid_timeout(self):
        self.emitter(exit_code=1)
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_check()
        self.emitter(delay=10)
        with self.assertRaises(subprocess.TimeoutExpired):
            self.run_check(timeout=0.1)
        for timeout in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                self.run_check(timeout=timeout)

    def test_manifest_limits_and_path_escape(self):
        for change in ({'bytes': 0}, {'bytes': 268435457}, {'file_offset': -1},
                       {'file_offset': 1 << 64}, {'shard': '../outside.gguf'}):
            with self.assertRaises(ValueError):
                manifest([replace(self.matrices[0], **change)], self.root, 64, 'native')
        for chunk in (0, True, 16 * 1024 * 1024 + 1):
            with self.assertRaises(ValueError):
                manifest(self.matrices, self.root, chunk, 'native')
        with self.assertRaises(ValueError):
            manifest([replace(self.matrices[0], bytes=1048577)], self.root, 1, 'native')
        with self.assertRaises(ValueError):
            manifest([], self.root, 64, 'native')

    def model(self):
        gguf = self.root / 'model.gguf'
        write_shard(gguf, *model_fixture())
        self.emitter()
        return gguf

    def test_report_includes_headers_binary_ranges_and_dedup(self):
        gguf = self.model()
        with patch('tools.check_glm5next_transfer.run_transport', return_value={'status': 'fixture'}) as run:
            result = check(gguf, self.script, [1, 2, 1], [1, 0, 1], ['native'], 64, 5)
        self.assertEqual(result['matrix_count'], 12)
        self.assertEqual({m['branch'] for m in result['matrices']}, {'main', 'mtp'})
        self.assertEqual(len(result['checker_sha256']), 64)
        self.assertEqual(len(result['shards'][0]['header_sha256']), 64)
        self.assertEqual(run.call_count, 1)

    def test_changed_source_or_checker_rejected(self):
        for changed in ('model', 'checker'):
            gguf = self.model()
            def mutate(*args):
                path = gguf if changed == 'model' else self.script
                with path.open('ab') as f:
                    f.write(b'changed')
                return {'status': 'fixture'}
            with patch('tools.check_glm5next_transfer.run_transport', side_effect=mutate):
                with self.assertRaisesRegex(ValueError, 'changed'):
                    check(gguf, self.script, [1], [0], ['native'], 64, 5)

    def test_cli_error_replaces_stale_success_and_protects_inputs(self):
        gguf = self.model()
        output = self.root / 'result.json'
        output.write_text('{"status":"pass"}', encoding='utf-8')
        args = ['--gguf', str(gguf), '--checker', str(self.script), '--output', str(output)]
        with patch('tools.check_glm5next_transfer.check', side_effect=ValueError('bad range')):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 1)
        self.assertEqual(json.loads(output.read_text(encoding='utf-8'))['status'], 'error')
        before = gguf.read_bytes()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(args[:-1]+[str(gguf)])
        self.assertEqual(gguf.read_bytes(), before)
        alias = self.root / 'alias.json'
        alias.hardlink_to(gguf)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(args[:-1]+[str(alias)])

    def cached_output(self):
        return ('GPU 46697874757265 13000 13000\n'
                + ''.join(f'CACHE_OK {i} 100 5 1 4 2 1\n' for i in range(3))
                + 'TOTAL 1200 1200 1200 0 24\nCACHE_TOTAL 3 12 6 3 15\n')

    def run_cached(self):
        return run_transport([sys.executable, str(self.script)], self.matrices,
                             self.root, 64, 'native', 5, 'a'*64)

    def test_cache_manifest_and_lifecycle_counters(self):
        payload = manifest(self.matrices, self.root, 64, 'native', 'a'*64)
        lines = payload.decode('ascii').splitlines()
        self.assertEqual(lines[0], f'GLM_CACHE_RANGES_V1 64 1 3 {"a"*64}')
        for m, line in zip(self.matrices, lines[1:]):
            self.assertEqual(line, f'{m.file_offset} {m.bytes} {str((self.root/m.shard).resolve()).encode("utf-8").hex()}'
                             f' {m.branch} {m.layer} {m.expert} {m.projection} {m.quant}'
                             f' {m.columns} {m.rows} {m.shard.encode("utf-8").hex()}')
        self.emitter(self.cached_output(), expected=payload)
        result = self.run_cached()
        self.assertEqual(result['h2d_bytes'], 1200)
        self.assertEqual(result['cache'], {'hits': 3, 'misses': 12, 'admissions': 12,
                                         'evictions': 6, 'invalidations': 3, 'byte_comparisons': 15,
                                         'bypasses': 0, 'hit_source_bytes': 0, 'hit_h2d_bytes': 0})

    def test_cache_results_reject_missing_stages_counters_and_old_protocol(self):
        valid = self.cached_output()
        for output in (self.output, valid.replace(' 5 1 4 2 1', ' 4 1 4 2 1', 1),
                       valid.replace('CACHE_OK 1', 'CACHE_OK 0'),
                       valid.replace('CACHE_TOTAL 3 12 6 3 15', 'CACHE_TOTAL 3 12 5 3 15'),
                       valid.replace('1200 1200 1200 0 24', '1500 1500 1500 0 30'),
                       valid.replace('CACHE_TOTAL 3 12 6 3 15\n', ''), valid+'extra\n'):
            with self.subTest(output=output):
                self.emitter(output)
                with self.assertRaises(ValueError):
                    self.run_cached()

    def test_cache_manifest_validates_complete_key(self):
        for identity in ('', 'a'*63, 'g'*64):
            with self.assertRaises(ValueError):
                manifest(self.matrices, self.root, 64, 'native', identity)
        for change in ({'branch': 'other'}, {'projection': 'other'}, {'quant': 'Q3_K extra'},
                       {'layer': -1}, {'expert': 1 << 31}, {'columns': 0}, {'rows': 1 << 64},
                       {'file_offset': (1 << 64) - 1}):
            with self.assertRaises(ValueError):
                manifest([replace(self.matrices[0], **change)], self.root, 64, 'native', 'a'*64)

    def test_cache_report_identity_covers_observed_file_set(self):
        gguf = self.model()
        with patch('tools.check_glm5next_transfer.run_transport', return_value={'status': 'fixture'}) as run:
            result = check(gguf, self.script, [1, 2], [0], ['native'], 64, 5, True)
            identity = result['cache_identity']
            self.assertEqual(run.call_args.args[-1], identity)
            self.assertEqual(len(identity), 64)
            with gguf.open('ab') as f:
                f.write(b'padding')
            changed = check(gguf, self.script, [1, 2], [0], ['native'], 64, 5, True)
            self.assertNotEqual(changed['cache_identity'], identity)
        self.assertEqual(len(result['cache_scenario']['stages']), 5)
        self.assertEqual({m['branch'] for m in result['matrices']}, {'main', 'mtp'})

    def test_cache_cli_failure_replaces_stale_success(self):
        gguf = self.model()
        output = self.root / 'cache.json'
        output.write_text('{"status":"pass"}', encoding='utf-8')
        with patch('tools.check_glm5next_transfer.check', side_effect=ValueError('cache stage missing')) as run:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(['--gguf', str(gguf), '--checker', str(self.script),
                                       '--cache-check', '--output', str(output)]), 1)
        self.assertTrue(run.call_args.args[-1])
        report = json.loads(output.read_text(encoding='utf-8'))
        self.assertEqual((report['schema_version'], report['status']), (2, 'error'))
        self.assertIn('cache stage missing', report['error'])

    def dispatch_output(self, mode='native'):
        return ('GPU 46697874757265 13000 13000\n' + ''.join(
            f'DISPATCH_OK 0 {frequency} {decode} 14 1020 1000 '
            + ('1040 0' if mode == 'native' or (mode == 'auto' and not decode) else '0 1040')
            + ' 1\n' for frequency in range(2) for decode in range(2)))

    def run_dispatch(self, mode='native'):
        return run_transport([sys.executable, str(self.script)], self.matrices,
                             self.root, 64, mode, 5, 'a'*64, True)

    def test_dispatch_modes_and_cancel_read_ahead(self):
        for mode in ('native', 'mmap', 'auto'):
            payload = manifest(self.matrices, self.root, 64, mode, 'a'*64, True)
            self.assertTrue(payload.startswith(b'GLM_DISPATCH_RANGES_V1 '))
            self.emitter(self.dispatch_output(mode), expected=payload)
            result = self.run_dispatch(mode)
            self.assertEqual(result['byte_comparisons'], 56)
            self.assertEqual(len(result['scenarios']), 4)
            self.assertEqual(result['scenarios'][0]['bypasses'], 5)
        self.emitter(self.dispatch_output('auto').replace('1040 0', '520 520'))
        self.assertEqual(self.run_dispatch('auto')['scenarios'][0]['mmap_bytes'], 520)

    def test_dispatch_results_reject_missing_reordered_stages_and_bad_counters(self):
        valid = self.dispatch_output()
        for output in (self.output, valid+'extra\n', valid.replace(' 14 ', ' 13 ', 1),
                       valid.replace('OK 0 0 0', 'OK 0 1 0'),
                       valid.replace('1020 1000', '1020 999'),
                       valid.replace('1040 0', '1101 0'),
                       valid.replace('1040 0', '1000 0'),
                       valid.replace('1040 0', '0 1040'),
                       valid.replace(' 1\n', ' 3\n')):
            with self.subTest(output=output):
                self.emitter(output)
                with self.assertRaises(ValueError):
                    self.run_dispatch()

    def test_dispatch_requires_distinct_triples_and_identity(self):
        for matrices, identity in ((self.matrices[:2], 'a'*64),
                                   ([self.matrices[0]]*3, 'a'*64), (self.matrices, None)):
            with self.assertRaises(ValueError):
                manifest(matrices, self.root, 64, 'native', identity, True)

    def test_dispatch_report_and_mutually_exclusive_cli_modes(self):
        gguf = self.model()
        with patch('tools.check_glm5next_transfer.run_transport', return_value={'status': 'fixture'}) as run:
            result = check(gguf, self.script, [1, 2], [0], ['auto'], 64, 5, dispatch_check=True)
        self.assertTrue(run.call_args.kwargs['dispatch_check'])
        self.assertEqual(len(result['cache_scenario']['stages']), 6)
        self.assertEqual(result['cache_scenario']['policies'], ['lru', 'frequency'])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(['--gguf', str(gguf), '--checker', str(self.script), '--cache-check',
                  '--dispatch-check', '--output', str(self.root/'report.json')])


if __name__ == '__main__':
    unittest.main()
