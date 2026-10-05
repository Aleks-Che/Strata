"""Profile admission/refusal tests without model weights or a GPU."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import prepare_step35_profile as setup
from serve.test_step35 import FIXTURE
from serve.test_step35_http import Tokenizer


class StepProfileTests(unittest.TestCase):
    def test_existing_destination_refused_without_inspection(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(setup, 'inspect_model') as inspect:
            with self.assertRaisesRegex(ValueError, 'Destination exists'):
                setup.prepare('model', 'engine', directory)
            inspect.assert_not_called()

    def test_admission_and_isolated_output(self):
        for variant in ('good', 'fingerprint', 'architecture', 'source_sha', 'patch_set', 'protocol_version'):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp); exe = base/'engine'; exe.write_bytes(b'engine')
                destination = base/'profile'
                report = {'first_shard': str(base/'model.gguf'), 'structural_fingerprint_sha256': setup.FINGERPRINT}
                identity = {'architecture': 'step35', 'engine': 'step35-native', 'protocol_version': 1,
                            'source_sha': setup.LOADER_SHA, 'patch_set': setup.PATCH_SET}
                if variant == 'fingerprint': report['structural_fingerprint_sha256'] = 'unreviewed'
                elif variant != 'good': identity[variant] = 'unreviewed'
                def extract(model, out):
                    (out/'tokenizer').mkdir()
                    (out/'tokenizer/chat_template.jinja').write_bytes(FIXTURE.read_bytes())
                with patch.object(setup, 'inspect_model', return_value=report), \
                     patch.object(setup.subprocess, 'run', return_value=SimpleNamespace(stdout=json.dumps(identity))), \
                     patch.object(setup, 'extract', side_effect=extract), \
                     patch.object(setup, 'load_tokenizer', return_value=Tokenizer()):
                    if variant != 'good':
                        with self.assertRaises(ValueError): setup.prepare('model', exe, destination)
                        self.assertFalse(destination.exists())
                    else:
                        path = setup.prepare('model', exe, destination)
                        cfg = json.loads(path.read_text(encoding='utf8'))
                        self.assertEqual(cfg['architecture'], 'step35')
                        self.assertEqual(cfg['validated_eog_ids'], [257, 258])
                        self.assertEqual(cfg['host'], '127.0.0.1')
                        self.assertEqual(Path(cfg['tokenizer']).parent, destination)
                        self.assertNotIn('--mtp', cfg['args'])
                        self.assertEqual(cfg['args'][-2:], ['--expert-cache-prefill', 'off'])
                        self.assertEqual(cfg['args'][cfg['args'].index('--expert-cache-mib')+1], '8192')


if __name__ == '__main__':
    unittest.main()
