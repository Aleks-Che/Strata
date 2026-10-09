"""CPU tests for the bounded full-logit audit and its no-decode accounting."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from tools.check_minimax_m2_prefix_context import compare_ranges, native_checks


class PrefixAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)/'logits.f32'

    def tearDown(self):
        self.tmp.cleanup()

    def test_exact_across_chunk_and_offset_boundaries(self):
        values=np.arange(13,dtype='<f4')/3
        np.concatenate(([99],values,[77],values)).astype('<f4').tofile(self.path)
        result=compare_ranges(self.path,1,15,13,chunk_floats=3)
        self.assertTrue(result['pass']); self.assertEqual(result['floats'],13)

    def test_single_bit_change_is_not_tolerated(self):
        values=np.ones(10,dtype='<f4'); values.view('<u4')[8]+=1; values.tofile(self.path)
        result=compare_ranges(self.path,0,5,5,chunk_floats=2)
        self.assertFalse(result['pass']); self.assertEqual(result['different_bits'],1)

    def test_signed_zero_difference_is_not_tolerated(self):
        np.array([0.,-0.],dtype='<f4').tofile(self.path)
        result=compare_ranges(self.path,0,1,1)
        self.assertFalse(result['pass']); self.assertEqual(result['max_abs'],0)

    def test_equal_nan_or_infinity_is_not_a_pass(self):
        for value in [float('nan'),float('inf'),-float('inf')]:
            np.array([value,value],dtype='<f4').tofile(self.path)
            result=compare_ranges(self.path,0,1,1)
            self.assertFalse(result['pass']); self.assertFalse(result['finite'])

    def test_empty_truncated_and_invalid_ranges_fail(self):
        self.path.write_bytes(b'\0'*7)
        for params in [(0,1,1),(0,0,0),(-1,0,1),(0,0,1,0)]:
            with self.subTest(params=params), self.assertRaises(ValueError):
                compare_ranges(self.path,*params)

    def test_only_single_output_allows_zero_decode_graphs(self):
        pre={'gpu_nodes':1,'compute_calls':1,'rejected_cpu_nodes':0,'rejected_full_copies':0}
        dec={k:0 for k in pre}
        result={'prefill':pre,'decode':dec,'generated_tokens':1,'prompt_tokens':52,'kv_tokens':52,
                'reused_tokens':48,'evaluated_prompt_tokens':4}
        with patch('tools.check_minimax_m2_prefix_context.result_checks',side_effect=lambda *_: {'gpu_only':False}):
            self.assertTrue(native_checks(result,{'batch':16})['gpu_only'])
            self.assertFalse(native_checks({**result,'generated_tokens':2},{'batch':16})['gpu_only'])
            self.assertFalse(native_checks({**result,'decode':{**dec,'compute_calls':1}},{'batch':16})['gpu_only'])
            self.assertFalse(native_checks({**result,'kv_tokens':53},{'batch':16})['kv_accounting'])


if __name__=='__main__':
    unittest.main()
