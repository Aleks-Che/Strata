"""Runner failures/cleanup; scripted children are not external VRAM evidence."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools.check_glm5next_memory import MIB, Worker, main, validate_stage


class MemoryCheckTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.script = self.root/'worker.py'

    def child(self, code, timeout=2):
        self.script.write_text(code, encoding='utf-8')
        return Worker([sys.executable, '-u', str(self.script)], timeout)

    def test_worker_protocol_and_clean_exit(self):
        with self.child('import sys,json\nprint(json.dumps({"phase":"ready"}))\n'
                        'assert input()=="QUIT"\n') as worker:
            self.assertEqual(worker.expect('ready')['phase'], 'ready')
            worker.finish()
        self.assertEqual(worker.process.returncode, 0)

    def test_early_exit_preserves_diagnostics_and_cleans_up(self):
        worker = self.child('import sys\nsys.stderr.write("fixture failure")\nsys.exit(7)\n')
        with self.assertRaisesRegex(ValueError, 'fixture failure'), worker:
            worker.expect('ready')
        self.assertEqual(worker.process.returncode, 7)

    def test_timeout_terminates_owned_child(self):
        worker = self.child('import time\ntime.sleep(30)\n', timeout=0.05)
        with self.assertRaises(TimeoutError), worker:
            worker.expect('ready')
        self.assertIsNotNone(worker.process.returncode)
        self.assertFalse(worker.reader.is_alive())

    def test_wrong_phase_extra_output_and_failed_exit_rejected(self):
        with self.child('print("{\\"phase\\":\\"wrong\\"}")\n') as worker:
            with self.assertRaises(ValueError):
                worker.expect('ready')
        for code in ('input()\nprint("extra")\n', 'import sys\ninput()\nsys.exit(3)\n'):
            with self.child(code) as worker:
                with self.assertRaises(ValueError):
                    worker.finish()

    def test_stage_counters_and_physical_memory_validation(self):
        protected = {'phase': 'protected', 'sample_valid': 1, 'target_bytes': 0,
                     'resident_bytes': 32*MIB, 'deferred_bytes': 32*MIB, 'trim_complete': 0,
                     'hits': 1, 'misses': 2, 'evictions': 1, 'free_bytes': 500*MIB, 'total_bytes': 1024*MIB,
                     'device_target_mib': 512}
        validate_stage(protected, 'protected')
        for change in ({'sample_valid': 0}, {'phase': 'warm'}, {'target_bytes': 64*MIB},
                       {'resident_bytes': 0}, {'deferred_bytes': 0}, {'trim_complete': 1},
                       {'hits': 0}, {'misses': 3}, {'evictions': 0}, {'free_bytes': -1},
                       {'total_bytes': 0}, {'free_bytes': 2048*MIB}, {'device_target_mib': 0}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_stage({**protected, **change}, 'protected')

    def test_report_requires_both_policies_and_replaces_stale_success(self):
        self.script.write_text('fixture', encoding='utf-8')
        output = self.root/'report.json'
        args = ['--checker', str(self.script), '--output', str(output)]
        with patch('tools.check_glm5next_memory.run_pair', return_value={'status': 'fixture'}) as run:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 0)
        self.assertEqual([call.args[1] for call in run.call_args_list], ['lru', 'frequency'])
        with patch('tools.check_glm5next_memory.run_pair', side_effect=ValueError('missing recovery')):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 1)
        report = json.loads(output.read_text(encoding='utf-8'))
        self.assertEqual(report['status'], 'error')
        self.assertIn('missing recovery', report['error'])

    def test_cli_protects_binary_and_validates_timeout(self):
        self.script.write_text('fixture', encoding='utf-8')
        alias = self.root/'alias.json'; alias.hardlink_to(self.script)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(['--checker', str(self.script), '--output', str(alias)])
        self.assertEqual(self.script.read_text(), 'fixture')
        for timeout in ('0', '-1', 'nan', 'inf'):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(['--checker', str(self.script), '--output', str(self.root/'report.json'), '--timeout', timeout])


if __name__ == '__main__':
    unittest.main()
