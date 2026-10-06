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
    def test_invalid_gpu_allocator_refused_before_inspection(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(setup, 'inspect_model') as inspect:
            for value in ('', 'pool', 1, None):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    setup.prepare('model','engine',Path(directory)/'new',gpu_cache_allocator=value)
            inspect.assert_not_called()
    def test_invalid_mtp_scratch_before_inspection(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(setup, 'inspect_model') as inspect:
            for shared,depth,placement in ((1,1,'streamed'),(True,0,'streamed'),(True,1,'resident'),(None,1,'streamed')):
                with self.subTest(shared=shared,depth=depth,placement=placement), self.assertRaises(ValueError):
                    setup.prepare('model','engine',Path(directory)/'new',mtp=depth,mtp_experts=placement,mtp_shared_scratch=shared)
            inspect.assert_not_called()
    def test_invalid_mtp_placement_before_inspection(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(setup, 'inspect_model') as inspect:
            for placement,depth in (('',1),('auto',1),(None,1),('streamed',0)):
                with self.subTest(placement=placement,depth=depth), self.assertRaises(ValueError):
                    setup.prepare('model','engine',Path(directory)/'new',mtp=depth,mtp_experts=placement)
            inspect.assert_not_called()
    def test_invalid_gpu_policy_refused_before_inspection(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(setup, 'inspect_model') as inspect:
            for value in ('', 'prefill', 1, None):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    setup.prepare('model','engine',Path(directory)/'new',gpu_cache_policy=value)
            inspect.assert_not_called()
    def test_invalid_ram_policy_refused_before_inspection(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(setup, 'inspect_model') as inspect:
            for value in ('', 'lfu', 1, None):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    setup.prepare('model','engine',Path(directory)/'new',ram_cache_policy=value)
            inspect.assert_not_called()
    def test_invalid_ram_cache_refused_before_inspection(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(setup, 'inspect_model') as inspect:
            for value in (-1,1048577,True,1.5,'1','bad',None):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    setup.prepare('model','engine',Path(directory)/'new',ram_cache_mib=value)
            inspect.assert_not_called()
    def test_invalid_mtp_refused_before_inspection(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(setup, 'inspect_model') as inspect:
            for value in (-1,4,True,1.5,'1',None):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    setup.prepare('model','engine',Path(directory)/'new',mtp=value)
            inspect.assert_not_called()
    def test_invalid_batch_refused_before_inspection(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(setup, 'inspect_model') as inspect:
            for value,readers in ((True,0),(1,2),('1',2),(None,2)):
                with self.subTest(value=value,readers=readers), self.assertRaises(ValueError):
                    setup.prepare('model','engine',Path(directory)/'new',cache_mib=8,
                                  pipeline_readers=readers,pipeline_batch=value)
            inspect.assert_not_called()
            self.assertFalse((Path(directory)/'new').exists())
    def test_invalid_pipeline_refused_before_inspection(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(setup, 'inspect_model') as inspect:
            for readers,chunk,cap in ((-1,4,8),(3,4,8),(True,4,8),(1,3,8),(1,4,0),(1,4.0,8)):
                with self.subTest(readers=readers,chunk=chunk,cap=cap), self.assertRaises(ValueError):
                    setup.prepare('model','engine',Path(directory)/'new',cache_mib=cap,
                                  pipeline_readers=readers,pipeline_chunk_mib=chunk)
            inspect.assert_not_called()
            self.assertFalse((Path(directory)/'new').exists())
    def test_invalid_cache_refused_before_inspection(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(setup, 'inspect_model') as inspect:
            for cap in (-1, 16385, True, 1.5, '8192'):
                with self.subTest(cap=cap), self.assertRaises(ValueError):
                    setup.prepare('model', 'engine', Path(directory)/'new', cache_mib=cap)
            inspect.assert_not_called()
            self.assertFalse((Path(directory)/'new').exists())

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
                        second = setup.prepare('model', exe, base/'cached', cache_mib=8192)
                        cached = json.loads(second.read_text(encoding='utf8'))
                        self.assertEqual(cached['args'][-2:], ['--expert-cache-mib', '8192'])
                        third=setup.prepare('model',exe,base/'pipeline',cache_mib=8192,pipeline_readers=1)
                        piped=json.loads(third.read_text(encoding='utf8'))
                        self.assertEqual(piped['args'][-4:],['--pipeline-readers','1','--pipeline-chunk-mib','4'])
                        fourth=setup.prepare('model',exe,base/'batch',cache_mib=8192,pipeline_readers=2,pipeline_batch=True)
                        batched=json.loads(fourth.read_text(encoding='utf8'))
                        self.assertEqual(batched['args'][-2:],['--pipeline-batch','1'])
                        for depth in (1,2,3):
                            mtp_profile=setup.prepare('model',exe,base/f'mtp{depth}',cache_mib=8192,pipeline_readers=2,mtp=depth)
                            mtp_cfg=json.loads(mtp_profile.read_text(encoding='utf8'))
                            self.assertEqual(mtp_cfg['args'][-2:],['--mtp',str(depth)])
                            self.assertEqual(mtp_cfg['sampling']['temperature'],0)
                        streamed=setup.prepare('model',exe,base/'mtp-streamed',mtp=1,mtp_experts='streamed')
                        args=json.loads(streamed.read_text(encoding='utf8'))['args']
                        self.assertEqual(args[args.index('--mtp-experts')+1],'streamed')
                        shared=setup.prepare('model',exe,base/'mtp-shared',mtp=1,mtp_experts='streamed',mtp_shared_scratch=True)
                        args=json.loads(shared.read_text(encoding='utf8'))['args']
                        self.assertEqual(args[args.index('--mtp-shared-scratch')+1],'1')
                        self.assertEqual(json.loads(profile.read_text(encoding='utf8')), cfg)
                        for cap in ('auto',8192):
                            ram=setup.prepare('model',exe,base/f'ram{cap}',cache_mib=8192,pipeline_readers=2,mtp=1,ram_cache_mib=cap)
                            self.assertEqual(json.loads(ram.read_text(encoding='utf8'))['args'][-2:],['--ram-cache-mib',str(cap)])
                            self.assertEqual(json.loads(ram.read_text(encoding='utf8'))['args'][-4:-2],['--ram-cache-policy','frequency'])
                        lru=setup.prepare('model',exe,base/'ram-lru',ram_cache_mib=8192,ram_cache_policy='lru')
                        self.assertEqual(json.loads(lru.read_text(encoding='utf8'))['args'][-4:],['--ram-cache-policy','lru','--ram-cache-mib','8192'])
                        gpu=setup.prepare('model',exe,base/'gpu-decode',cache_mib=11264,ram_cache_mib=65536,gpu_cache_policy='decode')
                        args=json.loads(gpu.read_text(encoding='utf8'))['args']
                        self.assertEqual(args[args.index('--gpu-cache-policy')+1],'decode')
                        arena=setup.prepare('model',exe,base/'gpu-arena',cache_mib=16384,gpu_cache_allocator='arena')
                        args=json.loads(arena.read_text(encoding='utf8'))['args']
                        self.assertEqual(args[args.index('--gpu-cache-allocator')+1],'arena')


if __name__ == '__main__':
    unittest.main()
