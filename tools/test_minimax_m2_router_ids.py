"""Reject claimed reuse when native counters show fallback or missing copies."""
import copy
import unittest
from tools.check_minimax_m2_router_ids import router_checks


def result(enabled):
    s={'pipeline_matrices':6,'router_ids_published':6 if enabled else 0,
       'router_ids_hits':6 if enabled else 0,'router_ids_misses':0,'router_ids_bytes':192 if enabled else 0}
    return {'prefill':s,'decode':copy.deepcopy(s)}


class Evidence(unittest.TestCase):
    def test_off_on(self):
        for mode in [0,1]:self.assertTrue(all(router_checks(result(mode),mode).values()))

    def test_missing_reuse(self):
        for key in ['router_ids_published','router_ids_hits','router_ids_bytes']:
            r=result(1);r['decode'][key]-=1
            self.assertFalse(router_checks(r,1)['router_reuse'],key)

    def test_hidden_fallback(self):
        r=result(1);r['decode']['router_ids_misses']=1
        self.assertFalse(router_checks(r,1)['no_router_fallback'])

    def test_disabled_must_not_reuse(self):
        self.assertFalse(all(router_checks(result(1),0).values()))

    def test_one_output_no_decode(self):
        r=result(1);r['decode']={k:0 for k in r['decode']}
        self.assertTrue(all(router_checks(r,1).values()))
        r['decode']['router_ids_bytes']=32
        self.assertFalse(router_checks(r,1)['router_reuse'])


if __name__=='__main__':unittest.main()
