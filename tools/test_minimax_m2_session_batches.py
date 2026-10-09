"""Boundary audit expectations must exercise the intended paths at each batch."""
import unittest
from tools.check_minimax_m2_session_batches import make_corpus


class BatchCorpusTests(unittest.TestCase):
    def corpus(self, batch):
        return make_corpus(list(range(52)), list(range(513)), 1000, batch)

    def test_batch_specific_boundaries(self):
        for batch, expected in [(1, [51, 35, 33, 1, 511, 512, 497]),
                                (8, [48, 32, 32, 8, 504, 512, 496])]:
            _, _, cases = self.corpus(batch); found = {c['name']:c for c in cases}
            self.assertEqual([found[n]['expected_reused'] for n in ['resident','branch','extend-shorter',
                             'extend-full-batch','long-resident','long-extend','long-branch']], expected)
            self.assertEqual(found['one-full-batch']['expected_reused'], 0)

    def test_fresh_reference_has_same_input_and_sampler_without_session(self):
        for batch in (1, 8):
            fresh, requests, cases = self.corpus(batch)
            self.assertEqual(len(cases), 25)
            self.assertTrue(all('session_key' not in r for r in fresh))
            for request, case in zip(requests, cases):
                self.assertEqual({k:v for k,v in request.items() if k != 'session_key'}, fresh[case['reference_index']])
            self.assertEqual(sum(c['expected_restore'] for c in cases), 8)

    def test_single_output_sampling_and_branch_are_distinct(self):
        fresh, requests, cases = self.corpus(8)
        by_name = {c['name']:r for c,r in zip(cases,requests)}
        self.assertEqual(sum(r['max_tokens']==1 for r in requests), 4)
        for name in ['sample-resident','sample-restored']:
            self.assertEqual(by_name[name]['sampling']['seed'], 42)
        self.assertNotEqual(by_name['branch']['tokens'][35], by_name['cold']['tokens'][35])
        self.assertNotEqual(by_name['long-branch']['tokens'][497], by_name['long-cold']['tokens'][497])
        self.assertEqual(len(fresh), 9)

    def test_invalid_fixture_inputs_rejected(self):
        for base, long, replacement, batch in [(list(range(51)),list(range(513)),1000,8),
                                              (list(range(52)),list(range(512)),1000,8),
                                              (list(range(52)),list(range(513)),35,8),
                                              (list(range(52)),list(range(513)),1000,16)]:
            with self.assertRaises(ValueError): make_corpus(base,long,replacement,batch)


if __name__ == '__main__':
    unittest.main()
