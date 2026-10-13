"""Phase accounting checks; no model or GPU required."""
import unittest

from benchmark_deepseek4_stage5 import expert_phase_counters


class PhaseCountersTest(unittest.TestCase):
    cfg = {'args': ['--expert-cache-mib', '2', '--draft-expert-cache-mib', '1']}

    def test_new_backend_and_existing_history(self):
        log = '''STRATA_EXPERT_BOUNDARY request=2 phase=begin
STRATA_EXPERT_SNAPSHOT request=2 phase=begin backend=10 cache_bytes=2097152 h2d_bytes=100 cpu_jobs=2
STRATA_EXPERT_SNAPSHOT request=2 phase=prefill_end backend=10 cache_bytes=2097152 h2d_bytes=150 cpu_jobs=2
STRATA_EXPERT_SNAPSHOT request=2 phase=prefill_end backend=20 cache_bytes=1048576 h2d_bytes=30 cpu_jobs=0
STRATA_EXPERT_SNAPSHOT request=2 phase=end backend=10 cache_bytes=2097152 h2d_bytes=190 cpu_jobs=5
STRATA_EXPERT_SNAPSHOT request=2 phase=end backend=20 cache_bytes=1048576 h2d_bytes=50 cpu_jobs=1
'''
        d = expert_phase_counters(log, self.cfg)['2']['deltas']
        self.assertEqual(d['prefill']['10']['h2d_bytes'], 50)
        self.assertEqual(d['decode']['10']['h2d_bytes'], 40)
        self.assertEqual(d['total']['20']['h2d_bytes'], 50)
        self.assertEqual(d['decode']['10']['cpu_jobs'], 3)
        self.assertEqual(d['prefill']['20']['cpu_jobs'], 0)
        self.assertEqual(d['total']['10']['role'], 'target')
        self.assertEqual(d['total']['20']['role'], 'draft')
        for backend in ('10', '20'):
            self.assertEqual(d['total'][backend]['h2d_bytes'],
                             d['prefill'][backend]['h2d_bytes'] + d['decode'][backend]['h2d_bytes'])

    def test_incomplete_request_has_no_invented_decode(self):
        log = '''STRATA_EXPERT_BOUNDARY request=1 phase=begin
STRATA_EXPERT_SNAPSHOT request=1 phase=error backend=10 cache_bytes=2097152 h2d_bytes=80
'''
        self.assertEqual(expert_phase_counters(log, self.cfg)['1']['deltas'], {})

    def test_reject_reset(self):
        log = '''STRATA_EXPERT_SNAPSHOT request=1 phase=begin backend=10 cache_bytes=2097152 cpu_jobs=3
STRATA_EXPERT_SNAPSHOT request=1 phase=end backend=10 cache_bytes=2097152 cpu_jobs=2
'''
        with self.assertRaises(ValueError):
            expert_phase_counters(log, self.cfg)


if __name__ == '__main__':
    unittest.main()
