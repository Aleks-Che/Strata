"""Frontend identity must not imply runtime GPU, cache or speculative support."""
from copy import deepcopy
from pathlib import Path
import unittest

from serve.frontend import ChatTemplate
from serve.glm5next import GLMTemplate
from serve.server import Service, MockEngine, ByteTokenizer
from serve.test_glm5next_service import Tokenizer


class GLMInfoTests(unittest.TestCase):
    def service(self):
        tok = Tokenizer()
        template = GLMTemplate(Path(__file__).parent / 'fixtures/glm53_chat_template.jinja')
        return Service(MockEngine(tok, ''), tok, template, model_name='glm-test')

    def test_template_identity_does_not_claim_engine_features(self):
        svc = self.service()
        info = svc.engine_facts()
        self.assertEqual(info['architecture'], 'glm5next')
        self.assertEqual(info['model'], 'glm-test')
        self.assertEqual(info['reasoning']['efforts'], ['low', 'high', 'max'])
        for key in ('mtp', 'spec', 'speculative', 'expert_compute', 'expert_storage', 'expert_pipeline'):
            self.assertNotIn(key, info)

    def test_metrics_preserve_reported_facts_and_zero_values(self):
        svc = self.service()
        reported = {'architecture': 'glm5next', 'expert_compute': 'gpu', 'expert_storage': 'mmap',
                    'spec': 0, 'speculative': 'none', 'expert_pipeline': 0, 'expert_cache_mib': 0,
                    'expert_cached_matrices': 0}
        svc.engine.info = deepcopy(reported)
        info = svc.metrics()['engine']
        for key, value in reported.items():
            self.assertEqual(info[key], value)
        self.assertEqual(svc.engine.info, reported)
        self.assertEqual(info['reasoning']['default'], 'max')

    def test_engine_architecture_wins_but_service_identity_is_authoritative(self):
        svc = self.service()
        svc.engine.info = {'architecture': 'reported-test', 'model': 'stale', 'max_context': 1, 'images': True}
        facts = svc.engine_facts()
        self.assertEqual(facts['architecture'], 'reported-test')
        self.assertEqual(facts['model'], 'glm-test')
        self.assertEqual(facts['max_context'], svc.engine.max_context)
        self.assertFalse(facts['images'])

    def test_legacy_qwen_info_remains_available(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, ''), tok, ChatTemplate(Path(__file__).parent / 'chat_template.jinja'))
        svc.engine.info = {'spec': 4, 'mtp_max': 4, 'lookup': 1, 'expert_slots': 3511}
        facts = svc.engine_facts()
        for key, value in svc.engine.info.items():
            self.assertEqual(facts[key], value)
        self.assertIsNone(facts['architecture'])
        self.assertEqual(facts['reasoning']['default'], 'high')


if __name__ == '__main__':
    unittest.main()
