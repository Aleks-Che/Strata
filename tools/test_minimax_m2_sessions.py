"""Ensure the session audit rejects numerical changes and truncated evidence."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from tools.check_minimax_m2_sessions import compare_files, NV


class SessionAuditTests(unittest.TestCase):
    def test_exact_rows_and_single_bit_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/'a.f32',Path(tmp)/'b.f32'
            data=np.arange(NV*2,dtype='<f4');data.tofile(a);data[NV:].tofile(b)
            self.assertTrue(compare_files(a,1,b,0,1)['pass'])
            changed=data[NV:].copy();changed.view('<u4')[NV-1]^=1;changed.tofile(b)
            result=compare_files(a,1,b,0,1)
            self.assertFalse(result['pass']);self.assertEqual(result['different_bits'],1)

    def test_equal_nonfinite_rows_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/'a.f32';data=np.zeros(NV,dtype='<f4');data[-1]=np.nan;data.tofile(a)
            result=compare_files(a,0,a,0,1)
            self.assertFalse(result['pass']);self.assertFalse(result['finite'])

    def test_invalid_or_truncated_ranges_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/'a.f32';a.write_bytes(b'\0'*4)
            for first,second,rows in [(0,0,1),(0,0,0),(-1,0,1),(0,-1,1)]:
                with self.subTest(first=first,second=second,rows=rows),self.assertRaises(ValueError):
                    compare_files(a,first,a,second,rows)


if __name__=='__main__':unittest.main()
