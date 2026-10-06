"""Profile admission and startup selection; no GPU or model payloads."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from serve.frontend import ChatTemplate
from serve.deepseek import DeepSeekTemplate
from serve.glm5next import GLMTemplate
from serve.server import configured_template
from tools.glm5next_loader_contract import LOADER_SHA
from tools.setup_glm5next import prepare_profile


class ProfileTests(unittest.TestCase):
    def test_template_selection_and_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            tok = SimpleNamespace(pre='glm4', special_ids={k:i for i,k in enumerate(GLMTemplate.stop_token_keys)})
            with self.assertRaises(SystemExit):
                configured_template({'architecture':'glm5next'}, tok, path)
            (path/'chat_template.jinja').write_text('{{ messages }}', encoding='utf-8')
            self.assertIsInstance(configured_template({'architecture':'glm5next'},tok,path),GLMTemplate)
            tok.pre='qwen35'
            with self.assertRaises(SystemExit):
                configured_template({'architecture':'glm5next'},tok,path)
            tok.pre='glm4'; tok.special_ids={}
            with self.assertRaises(SystemExit):
                configured_template({'architecture':'glm5next'},tok,path)
            self.assertIsInstance(configured_template({'architecture':'qwen4exp'},tok,path),ChatTemplate)
            tok.pre='joyai-llm'
            self.assertIsInstance(configured_template({'architecture':'deepseek4'},tok,path),DeepSeekTemplate)
            tok.pre='glm4'
            with self.assertRaises(SystemExit):
                configured_template({'architecture':'deepseek4'},tok,path)
            with self.assertRaises(SystemExit):
                configured_template({'architecture':'unknown'},tok,path)

    def test_profile_identity_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); exe=root/'engine'; exe.touch(); profile=root/'strata-glm.json'
            report={'first_shard':str(root/'model.gguf')}
            identity={'architecture':'glm5next','protocol_version':1,'source_sha':LOADER_SHA}
            with patch('tools.setup_glm5next.subprocess.run') as run, patch('tools.strata_tokenizer.extract') as extract:
                for bad in ({**identity,'architecture':'deepseek4'},{**identity,'source_sha':'wrong'},[],{}):
                    run.return_value=SimpleNamespace(stdout=json.dumps(bad))
                    with self.assertRaises(ValueError): prepare_profile(report,exe,profile)
                    self.assertFalse(profile.exists()); extract.assert_not_called()
                run.return_value=SimpleNamespace(stdout=json.dumps(identity))
                cfg=prepare_profile(report,exe,profile)
                self.assertEqual(cfg['host'],'127.0.0.1')
                self.assertEqual(cfg['architecture'],'glm5next')
                self.assertEqual(cfg['args'],['--native',report['first_shard'],'--max-context','2048','--batch-size','16','--threads','4',
                    '--expert-pipeline','0','--expert-chunk-mib','4','--mtp','0','--mtp-cache-mib','512'])
                self.assertEqual(json.loads(profile.read_text()),cfg)
                extract.assert_called_once()
                with self.assertRaises(ValueError): prepare_profile(report,exe,profile)
                self.assertEqual(json.loads(profile.read_text()),cfg)

    def test_bad_settings_create_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); exe=root/'engine'; exe.touch()
            for values in ({'context':0},{'batch':0},{'threads':0},{'port':0},{'batch':4097},
                           {'ram_target_percent':96},{'vram_target_percent':-1},{'vram_target_percent':9}):
                profile=root/'new'/'profile.json'
                with self.assertRaises(ValueError): prepare_profile({'first_shard':'model'},exe,profile,**values)
                self.assertFalse(profile.parent.exists())

    def test_memory_targets_reach_engine_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); exe=root/'engine'; exe.touch()
            identity={'architecture':'glm5next','protocol_version':1,'source_sha':LOADER_SHA}
            with patch('tools.setup_glm5next.subprocess.run',return_value=SimpleNamespace(stdout=json.dumps(identity))), \
                 patch('tools.strata_tokenizer.extract'):
                cfg=prepare_profile({'first_shard':'model.gguf'},exe,root/'profile.json',
                                    ram_target_percent=90,vram_target_percent=95)
            for flag, value in (('--ram-target-percent', '90'), ('--vram-target-percent', '95')):
                self.assertEqual(cfg['args'][cfg['args'].index(flag)+1], value)


if __name__=='__main__': unittest.main()
