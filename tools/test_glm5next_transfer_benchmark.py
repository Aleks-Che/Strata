"""Benchmark protocol checks without CUDA or model downloads."""
from dataclasses import replace
import contextlib
import io
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

    def test_case_selection_bounds_and_seed(self):
        sizes = [262144, 1048576, 4194304, 16777216]
        cases = bench.benchmark_cases(sizes, ['mmap', 'native'], [1, 2], ['decode'], 19)
        expected = {(mode, readers, True, chunk) for mode in ('mmap', 'native')
                    for readers in (1, 2) for chunk in sizes}
        self.assertEqual(set(cases), expected)
        self.assertEqual(len(cases), len(expected))
        self.assertEqual(cases, bench.benchmark_cases(sizes, ['mmap', 'native'], [1, 2], ['decode'], 19))
        self.assertNotEqual(cases, bench.benchmark_cases(sizes, ['mmap', 'native'], [1, 2], ['decode'], 20))
        for kwargs in ({'chunk_sizes': []}, {'chunk_sizes': [1, 1]}, {'chunk_sizes': [0]},
                       {'chunk_sizes': [True]}, {'chunk_sizes': [16777217]}, {'chunk_sizes': list(range(1, 10))},
                       {'modes': []}, {'modes': ['unknown']}, {'modes': ['mmap', 'mmap']},
                       {'readers': [0]}, {'readers': [5]}, {'readers': [True]}, {'readers': [1, 1]},
                       {'phases': []}, {'phases': ['other']}, {'phases': ['decode', 'decode']}):
            options = dict(chunk_sizes=[1048576], modes=['mmap'], readers=[1], phases=['decode'], seed=19)
            options.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                bench.benchmark_cases(**options)

    def test_chunk_comparison_keeps_other_factors_fixed(self):
        base, _ = self.run_fixture()
        runs = []
        for mode in ('mmap', 'native'):
            for phase in ('prefill', 'decode'):
                for readers in (1, 2):
                    for chunk, ms in ((262144, 30), (1048576, 20), (4194304, 25)):
                        runs.append({**base, 'mode': mode, 'phase': phase, 'readers': readers,
                                     'effective_readers': 1 if phase == 'decode' and mode == 'mmap' else readers,
                                     'chunk_bytes': chunk, 'median_ms': ms, 'min_ms': ms-1, 'max_ms': ms+1,
                                     'pinned_bytes': 4*chunk, 'ring_bytes': 4*chunk})
        comparisons = bench.chunk_comparisons(list(reversed(runs)))
        self.assertEqual(len(comparisons), 8)
        for comparison in comparisons:
            self.assertEqual(comparison['baseline_chunk_bytes'], 1048576)
            self.assertEqual(comparison['lowest_observed_chunk_bytes'], 1048576)
            self.assertEqual([m['median_ratio_to_baseline'] for m in comparison['measurements']], [1.5, 1, 1.25])
            self.assertEqual([m['pinned_bytes'] for m in comparison['measurements']], [1048576, 4194304, 16777216])
        without_old_baseline = [r for r in runs if r['chunk_bytes'] != 1048576]
        for comparison in bench.chunk_comparisons(without_old_baseline):
            self.assertEqual(comparison['baseline_chunk_bytes'], 262144)
            self.assertEqual(comparison['lowest_observed_chunk_bytes'], 4194304)

    def test_filtered_single_phase_report_and_original_defaults(self):
        template, _ = self.run_fixture()

        def sample(checker, matrices, directory, chunk, mode, readers, decode, *args):
            return {**template, 'chunk_bytes': chunk, 'mode': mode, 'readers': readers,
                    'phase': 'decode' if decode else 'prefill', 'effective_readers': 1 if decode and mode != 'native' else readers,
                    'pinned_bytes': chunk*4, 'ring_bytes': chunk*4}

        with patch.object(bench, 'inspect_model', return_value={'shard_details': []}), \
                patch.object(bench, 'plan_expert_reads', return_value=[self.matrix]), \
                patch.object(bench, 'digest', return_value='a'*64), \
                patch.object(bench, 'run_sample', side_effect=sample) as run, contextlib.redirect_stdout(io.StringIO()):
            report = bench.benchmark('model.gguf', 'checker', [3], [0], 1048576, 1, 2, 10, 19,
                                     chunk_sizes=[262144, 1048576], modes=['native'], readers=[2], phases=['decode'])
            self.assertEqual(run.call_count, 2)
            self.assertEqual(set(report['lowest_observed_medians']), {'decode'})
            self.assertIsNone(report['chunk_bytes'])
            self.assertEqual(report['chunk_sizes'], [262144, 1048576])
            self.assertEqual(len(report['chunk_comparisons']), 1)
            self.assertEqual(report['runner_sha256'], 'a'*64)
            run.reset_mock()
            original = bench.benchmark('model.gguf', 'checker', [3], [0], 1048576, 1, 2, 10, 17)
            self.assertEqual(run.call_count, 18)
            self.assertEqual(original['chunk_bytes'], 1048576)
            self.assertEqual(original['chunk_sizes'], [1048576])
            self.assertEqual(set(original['lowest_observed_medians']), {'prefill', 'decode'})


if __name__ == '__main__':
    unittest.main()
