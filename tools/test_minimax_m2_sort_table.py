"""Fail closed on missing copies, live sources, hidden fallback or unbounded RAM."""
import copy
import unittest
from tools.check_minimax_m2_sort_table import table_checks


def result(mode):
    s={'pipeline_matrices':186,'sort_table_copies':186 if mode else 0,
       'sort_table_bytes':11904 if mode else 0,'sort_table_pending':0,
       'sort_table_pinned_bytes':16384 if mode else 0,'sort_table_peak_pending':4 if mode else 0,
       'sort_table_reuse_waits':0,'sort_table_drain_waits':0,'sort_table_fallbacks':0}
    return {'prefill':s,'decode':copy.deepcopy(s),'evaluated_prompt_tokens':1,'generated_tokens':2}


class Evidence(unittest.TestCase):
    def test_off_on(self):
        for mode in [0,1]:self.assertTrue(all(table_checks(result(mode),mode).values()))

    def test_each_failure(self):
        for key,value in [('sort_table_copies',185),('sort_table_bytes',11900),('sort_table_pending',1),
                          ('sort_table_pinned_bytes',32768),('sort_table_peak_pending',5),('sort_table_fallbacks',1)]:
            r=result(1);r['decode'][key]=value
            self.assertFalse(all(table_checks(r,1).values()),key)

    def test_disabled_no_staging(self):
        self.assertFalse(all(table_checks(result(1),0).values()))

    def test_single_output(self):
        r=result(1);r['generated_tokens']=1;r['decode']={k:0 for k in r['decode']}
        r['decode']['sort_table_pinned_bytes']=16384
        self.assertTrue(all(table_checks(r,1).values()))
        r['decode']['sort_table_bytes']=64
        self.assertFalse(all(table_checks(r,1).values()))

    def test_strided_input_does_not_inflate_table(self):
        r=result(1);r['prefill']['router_ids_bytes']=9018624
        r['evaluated_prompt_tokens']=64;r['prefill']['pipeline_matrices']=r['prefill']['sort_table_copies']=744
        r['prefill']['sort_table_bytes']=750336
        self.assertTrue(all(table_checks(r,1).values()))
        r['prefill']['sort_table_bytes']=2*r['prefill']['router_ids_bytes']
        self.assertFalse(table_checks(r,1)['table_bytes'])


if __name__=='__main__':unittest.main()
