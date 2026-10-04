"""Profile settings and non-overwrite contract; no model, GPU or downloads."""
import json
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.glm5next_loader_contract import LOADER_SHA
from tools.setup_glm5next import prepare_profile


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.exe = self.root / 'engine.exe'
        self.exe.write_bytes(b'fixture')
        self.profile = self.root / 'glm.json'
        self.report = {'first_shard': str(self.root / 'model.gguf')}

    def make(self, **kwargs):
        identity = {'architecture': 'glm5next', 'protocol_version': 1, 'source_sha': LOADER_SHA}
        with patch('tools.setup_glm5next.subprocess.run', return_value=SimpleNamespace(stdout=json.dumps(identity))), \
             patch('tools.strata_tokenizer.extract'):
            return prepare_profile(self.report, self.exe, self.profile, **kwargs)

    def test_requested_pipeline_mtp_and_memory_survive_export(self):
        cfg = self.make(ram_target_percent=95, vram_target_percent=95, expert_pipeline=1,
                        expert_chunk_mib=8, mtp=3, mtp_cache_mib=1024)
        self.assertEqual(json.loads(self.profile.read_text(encoding='utf-8')), cfg)
        for flag, value in (('--ram-target-percent', 95), ('--vram-target-percent', 95),
                            ('--expert-pipeline', 1), ('--expert-chunk-mib', 8), ('--mtp', 3), ('--mtp-cache-mib', 1024)):
            self.assertEqual(cfg['args'][cfg['args'].index(flag) + 1], str(value))

    def test_invalid_settings_leave_no_profile(self):
        for options in ({'mtp': 4}, {'mtp': 3, 'batch': 3}, {'mtp_cache_mib': -1},
                        {'expert_pipeline': 2}, {'expert_chunk_mib': 0}, {'expert_chunk_mib': 17},
                        {'ram_target_percent': 96}, {'vram_target_percent': 96}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.make(**options)
            self.assertFalse(self.profile.exists())

    def test_existing_profile_is_preserved(self):
        self.profile.write_text('original', encoding='utf-8')
        with self.assertRaises(ValueError):
            self.make()
        self.assertEqual(self.profile.read_text(encoding='utf-8'), 'original')


if __name__ == '__main__':
    unittest.main()
