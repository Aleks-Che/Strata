"""Hy3 profile isolation and admission, without real model allocations."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import prepare_hy3_profile as setup
from serve.test_hy3 import FIXTURE
from serve.test_hy3_http import Tokenizer


class ProfileTests(unittest.TestCase):
    def test_existing_destination_refused_before_inspection(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(setup, 'inspect_model') as inspect:
            with self.assertRaisesRegex(ValueError, 'Destination exists'):
                setup.prepare('model', 'engine', directory)
            inspect.assert_not_called()

    def test_reviewed_identity_header_and_isolated_profile(self):
        for variant in ('good', 'header', 'architecture', 'source_sha', 'protocol_version', 'patch_set'):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                base = Path(directory); exe = base/'engine.exe'; exe.write_bytes(b'engine')
                destination = base/'profile'
                report = {'header_sha256': setup.HEADER_SHA}
                identity = dict(architecture='hy_v3', engine='hy3-native', protocol_version=1,
                                source_sha=setup.LOADER_SHA, patch_set=setup.PATCH_SET)
                if variant == 'header': report['header_sha256'] = 'other'
                elif variant != 'good': identity[variant] = 'other'
                def extract(model, output):
                    (output/'tokenizer').mkdir()
                    (output/'tokenizer/chat_template.jinja').write_bytes(FIXTURE.read_bytes())
                with patch.object(setup, 'inspect_model', return_value=report), \
                     patch.object(setup.subprocess, 'run', return_value=SimpleNamespace(stdout=json.dumps(identity))), \
                     patch.object(setup, 'extract', side_effect=extract), \
                     patch.object(setup, 'load_tokenizer', return_value=Tokenizer()):
                    if variant != 'good':
                        with self.assertRaises(ValueError): setup.prepare('model', exe, destination)
                        self.assertFalse(destination.exists())
                    else:
                        profile = setup.prepare('model', exe, destination)
                        cfg = json.loads(profile.read_text(encoding='utf8'))
                        self.assertEqual(cfg['host'], '127.0.0.1')
                        self.assertEqual(cfg['architecture'], 'hy_v3')
                        self.assertEqual(cfg['validated_eog_ids'], [257])
                        self.assertTrue(cfg['fit_max_tokens'])
                        self.assertEqual(cfg['args'][-4:], ['--copy-mode', 'pinned', '--kv', 'f32'])
                        self.assertNotIn('--mtp', cfg['args'])
                        self.assertEqual(Path(cfg['tokenizer']).parent, destination)


if __name__ == '__main__':
    unittest.main()
