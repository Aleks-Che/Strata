"""Reject incomplete event/drain evidence before accepting a performance run."""
import copy
import unittest
from tools.check_minimax_m2_copy_events import event_checks


def result(mode):
    s={'pipeline_matrices':6,'pipeline_copy_batches':6,'pipeline_abort_fences':0,
       'pipeline_pending_copy':0,'pipeline_observer_fences':0,'compute_calls':2,
       'pipeline_copy_fences':0 if mode else 6,'pipeline_scratch_fences':0 if mode else 6,
       'pipeline_copy_events':6 if mode else 0,'pipeline_scratch_events':6 if mode else 0,
       'pipeline_scheduler_waits_skipped':6 if mode else 0,'pipeline_retire_checks':6 if mode else 0,
       'pipeline_retire_waits':1 if mode else 0,'async_compute_calls':8 if mode==2 else 0,
       'graph_exit_fences':2 if mode==2 else 0}
    return {'prefill':s,'decode':copy.deepcopy(s)}


class Evidence(unittest.TestCase):
    def test_all_modes(self):
        for mode in range(3):self.assertTrue(all(event_checks(result(mode),mode).values()))

    def test_missing_dependency_or_retirement(self):
        for key in ['pipeline_copy_events','pipeline_scratch_events','pipeline_scheduler_waits_skipped','pipeline_retire_checks']:
            r=result(2);r['decode'][key]=5
            self.assertFalse(event_checks(r,2)['matrix_dependencies'],key)

    def test_pending_copy_rejected(self):
        r=result(2);r['prefill']['pipeline_pending_copy']=1
        self.assertFalse(event_checks(r,2)['pending_copy_drained'])

    def test_async_work_requires_graph_exit_drain(self):
        r=result(2);r['decode']['graph_exit_fences']=1
        self.assertFalse(event_checks(r,2)['compute_mode'])

    def test_observer_cannot_hide_a_wait(self):
        r=result(2);r['decode']['pipeline_observer_fences']=1
        self.assertFalse(event_checks(r,2)['no_observer'])

    def test_old_mode_cannot_claim_async_speed(self):
        self.assertFalse(all(event_checks(result(2),0).values()))


if __name__=='__main__':unittest.main()
