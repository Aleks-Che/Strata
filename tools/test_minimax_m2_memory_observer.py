"""CPU failure-path checks for independent pressure evidence."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from tools.minimax_m2_memory_observer import MemoryObserver,memory_within_limit


def sample():
    return {'monotonic':1.,'ram_total':1000,'ram_free':60,'gpu_total':1000,'gpu_free':60,
            'commit_total':2000,'commit_free':1000}


class Sampler:
    def __init__(self,value=None,error=False):self.value=value or sample();self.error=error;self.closed=False
    def __call__(self):
        if self.error:raise OSError('observer fixture read failure')
        return self.value
    def close(self):self.closed=True


class ObserverTests(unittest.TestCase):
    def test_limit_and_invalid_samples(self):
        self.assertTrue(memory_within_limit(sample()))
        self.assertTrue(memory_within_limit({**sample(),'ram_free':50,'gpu_free':50}))
        for key,value in [('ram_free',49),('gpu_free',49),('ram_free',1001),('gpu_total',0),
                          ('ram_free',True),('gpu_free',float('nan')),('commit_free',2001),('commit_free',-1)]:
            with self.subTest(key=key,value=value):
                self.assertFalse(memory_within_limit({**sample(),key:value}))

    def test_guard_failure_is_retained_and_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            called=threading.Event();sampler=Sampler({**sample(),'ram_free':49})
            observer=MemoryObserver(Path(tmp)/'samples.jsonl',sampler=sampler,on_error=called.set,interval=.01)
            observer.start();self.assertTrue(called.wait(2));observer.close()
            self.assertIn('RAM/VRAM95',observer.error);self.assertTrue(sampler.closed)
            rows=[json.loads(x) for x in observer.path.read_text().splitlines()]
            self.assertEqual(len(rows),1);self.assertEqual(rows[0]['ram_free'],49)

    def test_sampler_failure_cannot_pass_as_empty_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            called=threading.Event();sampler=Sampler(error=True)
            observer=MemoryObserver(Path(tmp)/'samples.jsonl',sampler=sampler,on_error=called.set)
            observer.start();self.assertTrue(called.wait(2));observer.close()
            self.assertIn('fixture read failure',observer.error);self.assertEqual(observer.count,0)
            self.assertTrue(sampler.closed)

    def test_close_joins_before_artifact_is_hashed(self):
        with tempfile.TemporaryDirectory() as tmp:
            reached=threading.Event();sampler=Sampler()
            def sequence():reached.set();return 7
            observer=MemoryObserver(Path(tmp)/'samples.jsonl',sampler=sampler,sequence=sequence,interval=.01)
            observer.start();self.assertTrue(reached.wait(2));observer.close()
            data=observer.path.read_bytes();self.assertTrue(data.endswith(b'\n'))
            self.assertEqual(observer.count,len(data.splitlines()));self.assertIsNone(observer.error)
            self.assertTrue(observer.summary()['stopped'] and sampler.closed)
            self.assertTrue(all(json.loads(line)['native_sequence']==7 for line in data.splitlines()))


if __name__=='__main__':unittest.main()
