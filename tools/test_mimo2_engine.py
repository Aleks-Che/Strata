"""CPU-only admission tests for full-model parity reports."""
import unittest
import numpy as np

from tools.check_mimo2_engine import compare_logits
from tools.check_mimo2_cuda import validate_result


class MiMoEngineReports(unittest.TestCase):
    def test_exact_transport_reference(self):
        a = np.array([[1, 2, -3], [4, 5, 6]], dtype=np.float32)
        self.assertTrue(compare_logits(a, a.copy())['pass_'])
        b = a.copy(); b[0, 0] = np.nextafter(b[0, 0], np.float32(2))
        self.assertFalse(compare_logits(a, b)['pass_'])
        self.assertGreater(compare_logits(a, b)['max_abs'], 0)

    def test_invalid_reference_not_a_pass(self):
        a = np.array([1, 2, 3], dtype=np.float32)
        for invalid in [np.array([], dtype=np.float32), a[:2]]:
            with self.assertRaises(ValueError): compare_logits(a, invalid)
        for value in [float('nan'), float('inf')]:
            b = a.copy(); b[0] = value
            with np.errstate(invalid='ignore'):
                self.assertFalse(compare_logits(b, b)['pass_'])

    def test_runtime_coverage_and_build_identity(self):
        manifest = dict(source_revision='pin', archive_sha256='archive', patches='patch')
        result = dict(status='pass', requested_revision='pin', archive_sha256='archive',
                      patch_set='patch', case_count=224, results=[{'pass': True} for _ in range(224)])
        validate_result(result, manifest, 'runtime')
        result['results'].pop(); result['case_count'] -= 1
        with self.assertRaises(ValueError): validate_result(result, manifest, 'runtime')


if __name__ == '__main__':
    unittest.main()
