"""Comparison runner contracts with a scripted subprocess, not a numeric oracle."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from serve.glm5next import GLMTemplate
from tools.check_glm5next_tokenizer import check, compare_ids, main, oracle_version
from tools.glm5next_loader_contract import LOADER_SHA
from tools.glm5next_tokenizer_corpus import cases
from tools.test_glm5next_tokenizer import fixture
from tools.test_setup_glm5next import write_shard


class ByteTokens:
    tokens = list(range(256))

    def encode(self, text, parse_special=False):
        return [250] if text == '[gMASK]' and parse_special else list(text.encode('utf-8'))


class OracleCheckTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.script = self.root / 'scripted oracle.py'
        self.command = [sys.executable, str(self.script)]
        self.archive_hash = 'a' * 64
        self.version = f'requested_revision={LOADER_SHA}\narchive_sha256={self.archive_hash}\n'
        self.inputs = [('empty', ''), ('unicode-nul', 'П\x00'), ('special', '[gMASK]')]
        self.output = 'READY_TOKENIZER\nIDS\nIDS\nIDS 208 159 0\nIDS 208 159 0\nIDS 91 103 77 65 83 75 93\nIDS 250\n'

    def emitter(self, output=None, version=None, delay=0, exit_code=0, expected=None):
        self.script.write_text(
            'import sys, time\n'
            f'if "--version" in sys.argv:\n    sys.stdout.write({self.version if version is None else version!r})\n'
            'else:\n'
            f'    request = sys.stdin.buffer.read()\n    expected = {expected!r}\n'
            '    if expected is not None and request != expected: sys.exit(9)\n'
            f'    time.sleep({delay!r})\n'
            f'    sys.stdout.write({self.output if output is None else output!r})\n'
            '    sys.stderr.write("scripted fixture, not llama.cpp")\n'
            f'    sys.exit({exit_code})\n', encoding='utf-8')

    def compare(self, timeout=5):
        return compare_ids(ByteTokens(), self.inputs, self.command, 'path with spaces.gguf', timeout)

    def test_exact_ids_flags_empty_and_unicode_transport(self):
        self.emitter(expected=b'ENC 0 \nENC 1 \nENC 0 d09f00\nENC 1 d09f00\n'
                              b'ENC 0 5b674d41534b5d\nENC 1 5b674d41534b5d\nQUIT\n')
        report = self.compare()
        self.assertEqual((report['status'], report['case_count'], report['mismatch_count']), ('pass', 6, 0))
        self.assertEqual(report['cases'][-1]['oracle_ids'], [250])
        self.assertEqual(report['cases'][2]['text'], 'П\x00')
        self.assertIn('not llama.cpp', report['oracle_stderr_tail'])

    def test_id_mismatch_and_length_difference_are_failures(self):
        for old, new, difference in (('IDS 250', 'IDS 249', 0),
                                     ('IDS 250', 'IDS 250 1', 1),
                                     ('IDS 250', 'IDS', 0)):
            with self.subTest(new=new):
                self.emitter(self.output.replace(old, new))
                report = self.compare()
                self.assertEqual((report['status'], report['mismatch_count']), ('fail', 1))
                self.assertEqual(report['cases'][-1]['first_difference'], difference)
                self.assertEqual(report['cases'][-1]['python_ids'], [250])

    def test_protocol_errors_never_become_parity_success(self):
        for output in ('', self.output.replace('READY_TOKENIZER', 'READY'), self.output + 'IDS\n',
                       self.output.replace('IDS 250\n', ''), self.output.replace('IDS 250', 'ERR error'),
                       self.output.replace('IDS 250', 'IDS -1'), self.output.replace('IDS 250', 'IDS 256'),
                       self.output.replace('IDS 250', 'IDS 1.5')):
            with self.subTest(output=output[-30:]):
                self.emitter(output)
                with self.assertRaises(ValueError):
                    self.compare()

    def test_child_failure_and_timeout(self):
        self.emitter(exit_code=3)
        with self.assertRaises(subprocess.CalledProcessError):
            self.compare()
        self.emitter(delay=10)
        with self.assertRaises(subprocess.TimeoutExpired):
            self.compare(timeout=0.1)

    def test_provenance_checked_strictly(self):
        self.emitter()
        self.assertEqual(oracle_version(self.command, self.archive_hash.upper(), 5)['requested_revision'], LOADER_SHA)
        for version in (self.version.replace(LOADER_SHA, 'b' * 40),
                        self.version.replace(self.archive_hash, 'b' * 64),
                        self.version + 'archive_sha256=' + self.archive_hash + '\n',
                        self.version + 'unexpected=field\n', 'READY_TOKENIZER\n'):
            self.emitter(version=version)
            with self.subTest(version=version), self.assertRaises(ValueError):
                oracle_version(self.command, self.archive_hash, 5)
        with self.assertRaisesRegex(ValueError, '64-digit'):
            oracle_version(self.command, '', 5)

    def test_embedded_template_corpus_and_provenance_report(self):
        tok = fixture()
        source = (Path(__file__).parents[1] / 'serve/fixtures/glm53_chat_template.jinja').read_text(encoding='utf-8')
        gguf = self.root / 'metadata.gguf'
        meta = {'general.architecture': 'glm5next', 'tokenizer.ggml.model': 'gpt2',
                'tokenizer.ggml.pre': 'glm4', 'tokenizer.ggml.tokens': tok.tokens,
                'tokenizer.ggml.merges': [' '.join(pair) for pair in tok.ranks],
                'tokenizer.ggml.token_type': tok.token_types, 'tokenizer.chat_template': source}
        write_shard(gguf, meta, [])
        # This test validates corpus preparation/provenance, not oracle ID parity.
        with patch('tools.check_glm5next_tokenizer.oracle_version', return_value={'fixture': True}), \
             patch('tools.check_glm5next_tokenizer.compare_ids', return_value={'status': 'not-run'}) as compare:
            report = check(gguf, sys.executable, self.archive_hash, 5)
        inputs = dict(compare.call_args.args[1])
        self.assertEqual(len(inputs), 36)
        self.assertIn('Old thought', inputs['prompt/multi-turn/low/clear=0'])
        self.assertNotIn('Old thought', inputs['prompt/multi-turn/low/clear=1'])
        for clear in (0, 1):
            self.assertIn('Need two results', inputs[f'prompt/tool-turn/max/clear={clear}'])
            self.assertIn('<tool_response>А</tool_response><tool_response>中</tool_response>',
                          inputs[f'prompt/tool-turn/max/clear={clear}'])
        self.assertEqual(report['template_sha256'], 'a4fddbbf0b432101a296c17094f8bc5a2b0d30713b5b5cd92f86be78511aa724')
        self.assertEqual(report['gguf_header_bytes'], gguf.stat().st_size)
        self.assertEqual(len(report['oracle_binary_sha256']), 64)

    def test_error_report_replaces_previous_success_and_exit_code_is_nonzero(self):
        output = self.root / 'report.json'
        output.write_text('{"status":"pass"}')
        with patch('tools.check_glm5next_tokenizer.check', side_effect=ValueError('provenance mismatch')), \
             contextlib.redirect_stdout(io.StringIO()):
            code = main(['--gguf', str(self.root / 'model'), '--oracle', sys.executable,
                         '--archive-sha256', self.archive_hash, '--output', str(output)])
        self.assertEqual(code, 1)
        report = json.loads(output.read_text(encoding='utf-8'))
        self.assertEqual(report['status'], 'error')
        self.assertIn('provenance mismatch', report['error'])

    def test_report_cannot_overwrite_inputs(self):
        gguf = self.root / 'model.gguf'
        gguf.write_bytes(b'original')
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(['--gguf', str(gguf), '--oracle', sys.executable,
                  '--archive-sha256', self.archive_hash, '--output', str(gguf)])
        self.assertEqual(gguf.read_bytes(), b'original')

    def test_no_empty_corpus_or_invalid_timeout(self):
        with self.assertRaisesRegex(ValueError, 'Empty comparison'):
            compare_ids(ByteTokens(), [], self.command, 'unused', 5)
        for timeout in (0, -1, float('nan'), float('inf')):
            with self.assertRaisesRegex(ValueError, 'finite and positive'):
                check('unused', 'unused', self.archive_hash, timeout)


if __name__ == '__main__':
    unittest.main()
