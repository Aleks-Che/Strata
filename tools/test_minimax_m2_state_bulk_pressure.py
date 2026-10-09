"""A fast eviction must not be confused with failure to reach the RAM target."""
import unittest
from tools.validate_minimax_m2_state_bulk_pressure import ram_sample_evidence

GIB=2**30


def sample(at,free):
    return {'monotonic':at,'ram_total':100*GIB,'ram_free':int(free*GIB),'commit_free':40*GIB}


class RamEvidenceTests(unittest.TestCase):
    def test_reclaim_between_target_and_next_poll(self):
        before,after=sample(99.954,6),sample(100.469,6.8)
        e=ram_sample_evidence([before,after],sample(100,5.8),2*GIB)
        self.assertTrue(e['target_verified'] and e['post_budget_short'])
        self.assertEqual(e['target_sample'],before)
        self.assertEqual(e['post_sample'],after)

    def test_no_independent_target_is_rejected(self):
        e=ram_sample_evidence([sample(99.9,8),sample(100.4,8)],sample(100,5.8),2*GIB)
        self.assertFalse(e['target_verified'])

    def test_distant_sample_cannot_prove_target(self):
        e=ram_sample_evidence([sample(99,6),sample(101,6)],sample(100,5.8),2*GIB)
        self.assertFalse(e['target_verified'])

    def test_post_target_ram_and_commit_must_prove_physical_shortage(self):
        for after in [sample(100.4,8),{**sample(100.4,6.8),'commit_free':6*GIB}]:
            e=ram_sample_evidence([sample(99.9,6),after],sample(100,5.8),2*GIB)
            self.assertTrue(e['target_verified'])
            self.assertFalse(e['post_budget_short'])


if __name__=='__main__':unittest.main()
