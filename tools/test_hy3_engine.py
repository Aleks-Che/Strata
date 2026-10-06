"""Harness regression: forced termination must not hide diagnostic reports."""
import subprocess
import sys
import tempfile
import unittest

from tools.check_hy3_engine import Engine


class EngineCleanupTest(unittest.TestCase):
    def test_close_after_forced_exit_with_buffered_stdin(self):
        engine = Engine.__new__(Engine)
        engine.stderr = tempfile.TemporaryFile(mode='w+', encoding='utf8')
        engine.process = subprocess.Popen(
            [sys.executable, '-u', '-c', 'import time; print("ready", flush=True); time.sleep(60)'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=engine.stderr,
            text=True, encoding='utf8', creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        try:
            self.assertEqual(engine.process.stdout.readline().strip(), 'ready')
            engine.process.stdin.write('pending input without flush')
            engine.process.terminate()
            engine.process.wait(timeout=10)
            engine.close()  # Closing stdin may raise EINVAL/BrokenPipe internally.
            self.assertTrue(engine.stderr.closed)
            self.assertTrue(engine.process.stdin.closed)
            self.assertTrue(engine.process.stdout.closed)
        finally:
            if engine.process.poll() is None:
                engine.process.kill()
                engine.process.wait(timeout=10)
            engine.close()


if __name__ == '__main__':
    unittest.main()
