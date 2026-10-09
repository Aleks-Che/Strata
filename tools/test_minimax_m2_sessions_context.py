"""CPU checks that the long-session audit cannot accept false restore evidence."""
from copy import deepcopy
import unittest
from unittest.mock import patch
from tools.check_minimax_m2_sessions_context import inspect_case


class SessionContextTests(unittest.TestCase):
    def setUp(self):
        self.r={'reused_tokens':4064,'session_restore':True,'session_restored_bytes':2048,
                'session_archive_bytes':1024,'session_archive_entries':1,
                'session_ms':3500.,'request_ms':8000.,'ttft_ms':5500.}
        self.h={'session_cache_mib':6144,'session_cache_slots':4}
        self.expected={'expected_reused':4064,'expected_restore':True}

    def checks(self,r):
        with patch('tools.check_minimax_m2_sessions_context.native_checks',return_value={'base':True}):
            return inspect_case(r,self.h,self.expected)

    def test_restore_requires_flag_bytes_and_expected_reuse(self):
        self.assertTrue(all(self.checks(self.r).values()))
        for key,value in [('session_restore',False),('session_restore',1),('session_restored_bytes',0),('reused_tokens',0)]:
            r=deepcopy(self.r);r[key]=value
            with self.subTest(key=key,value=value):self.assertFalse(all(self.checks(r).values()))

    def test_bounds_and_transfer_time_cannot_be_hidden(self):
        for key,value in [('session_archive_bytes',6144*2**20+1),('session_archive_bytes',-1),
                          ('session_archive_entries',5),('session_archive_entries',-1),
                          ('session_ms',-1),('session_ms',float('nan')),('session_ms',8001),('ttft_ms',3499)]:
            r=deepcopy(self.r);r[key]=value
            with self.subTest(key=key,value=value):self.assertFalse(all(self.checks(r).values()))

    def test_native_failures_are_preserved(self):
        with patch('tools.check_minimax_m2_sessions_context.native_checks',return_value={'base':False}):
            self.assertFalse(all(inspect_case(self.r,self.h,self.expected).values()))


if __name__=='__main__':unittest.main()
