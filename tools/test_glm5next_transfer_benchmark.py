"""Benchmark protocol checks without CUDA or model downloads."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from tools import benchmark_glm5next_transfer as bench
from tools.glm5next_expert_plan import ExpertMatrix


class BenchmarkTests(unittest.TestCase):
    matrix = ExpertMatrix('main', 3, 0, 'gate', 'IQ2_S', 256, 256, 'part.gguf', 96, 82*256)

    def output(self, mode='mmap', decode=False):
        n = self.matrix.bytes
        native = n if mode == 'native' else 0
        effective = 1 if decode and mode != 'native' else 2
        lines = [f'GPU {b"test GPU".hex()} 13000 13000',
                 f'BENCH_CONFIG 2 {effective} {int(decode)} 1 2 {n} {n+74} 16384 16384',
                 f'SAMPLE 0 1000000 {n} {n} {native} {n-native} 6 100 200 300 400',
                 f'SAMPLE 1 3000000 {n} {n} {native} {n-native} 6 100 200 300 400',
                 'VERIFIED 3']
        return '\n'.join(lines).encode('ascii')

    def run_fixture(self, output=None, **kwargs):
        options = dict(chunk_bytes=4096, mode='mmap', readers=2, decode=False, warmups=1, repeats=2, timeout=10)
        options.update(kwargs)
        if output is None:
            output = self.output(options['mode'], options['decode'])
        with patch.object(bench.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, output, b'')) as run:
            result = bench.run_sample(Path('checker.exe'), [self.matrix], Path('.'), **options)
        return result, run

    def test_samples_and_effective_reader_policy(self):
        for mode in bench.MODES:
            for decode in (False, True):
                with self.subTest(mode=mode, decode=decode):
                    result, run = self.run_fixture(mode=mode, decode=decode)
                    self.assertEqual(result['median_ms'], 2)
                    self.assertEqual(result['min_ms'], 1)
                    self.assertEqual(result['max_ms'], 3)
                    self.assertEqual(result['effective_readers'], 1 if decode and mode != 'native' else 2)
                    self.assertEqual(result['byte_comparisons'], 3)
                    self.assertAlmostEqual(result['payload_gib_per_second'], self.matrix.bytes/(1 << 30)/.002)
                    self.assertEqual(run.call_args.kwargs['timeout'], 10)
                    self.assertTrue(run.call_args.kwargs['check'])

    def test_rejects_missing_or_corrupt_evidence(self):
        original = self.output().decode('ascii')
        n = self.matrix.bytes
        mutations = [original.replace('GPU ', 'BAD ', 1),
                     original.replace('BENCH_CONFIG 2 2', 'BENCH_CONFIG 2 1'),
                     original.replace('1000000', '0'),
                     original.replace('SAMPLE 1', 'SAMPLE 0'),
                     original.replace(f'{n} {n} 0 {n} 6', f'{n-1} {n} 0 {n} 6'),
                     original.replace(f'0 {n} 6', f'{n} 0 6'),
                     original.replace(' 6 100', ' 7 100'),
                     original.replace('VERIFIED 3', 'VERIFIED 2'),
                     original.replace('VERIFIED 3', ''), original + '\nEXTRA',
                     original.replace('1000000', 'nan')]
        for output in mutations:
            with self.subTest(output=output), self.assertRaises(ValueError):
                self.run_fixture(output.encode('ascii'))

    def test_limits_fail_before_subprocess(self):
        for kwargs in ({'readers': 0}, {'readers': 5}, {'readers': True}, {'decode': 1},
                       {'warmups': 0}, {'warmups': 6}, {'repeats': 0}, {'repeats': 21},
                       {'timeout': 0}, {'timeout': float('nan')}, {'timeout': float('inf')}):
            with self.subTest(kwargs=kwargs), patch.object(bench.subprocess, 'run') as run:
                options = dict(chunk_bytes=4096, mode='mmap', readers=2, decode=False, warmups=1, repeats=2, timeout=10)
                options.update(kwargs)
                with self.assertRaises(ValueError):
                    bench.run_sample('checker', [self.matrix], Path('.'), **options)
                run.assert_not_called()
        large = replace(self.matrix, bytes=256*1024*1024)
        with patch.object(bench.subprocess, 'run') as run, self.assertRaisesRegex(ValueError, '512 MiB'):
            bench.run_sample('checker', [large, large], Path('.'), 1024*1024, 'mmap', 1, False, 1, 1, 10)
        run.assert_not_called()

    def test_subprocess_failure_and_timeout_propagate(self):
        for error in (subprocess.CalledProcessError(1, 'checker'), subprocess.TimeoutExpired('checker', 1)):
            with self.subTest(error=error), patch.object(bench.subprocess, 'run', side_effect=error):
                with self.assertRaises(type(error)):
                    bench.run_sample('checker', [self.matrix], Path('.'), 4096, 'mmap', 1, False, 1, 1, 1)

    def test_failed_run_is_not_published_as_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);output = root/'report.json'
            with patch.object(bench, 'benchmark', side_effect=ValueError('bad byte evidence')):
                code = bench.main(['--gguf', str(root/'model.gguf'), '--checker', str(root/'checker.exe'),
                                   '--output', str(output)])
            report = json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(code, 1)
            self.assertEqual(report['status'], 'error')
            self.assertIn('bad byte evidence', report['error'])
            self.assertNotIn('lowest_observed_medians', report)

    def test_output_cannot_replace_checker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);checker = root/'checker.json';checker.write_bytes(b'preserve')
            with self.assertRaises(SystemExit), patch.object(bench, 'benchmark') as run:
                bench.main(['--gguf', str(root/'model.gguf'), '--checker', str(checker), '--output', str(checker)])
            run.assert_not_called()
            self.assertEqual(checker.read_bytes(), b'preserve')


if __name__ == '__main__':
    unittest.main()
