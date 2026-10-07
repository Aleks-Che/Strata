"""CPU-only checks for CUDA runner resource limits and report admission."""
from copy import deepcopy
import unittest

from tools.check_mimo2_cuda import check_ceiling, validate_result


class MiMoCudaAdmission(unittest.TestCase):
    def test_d2d_batch_requires_error_recovery_coverage(self):
        manifest = {'source_revision': 'pin', 'archive_sha256': 'archive', 'patches': 'patches'}
        for fill, count in [(False, 228), (True, 260)]:
            result = dict(status='pass', requested_revision='pin', archive_sha256='archive', patch_set='patches',
                          case_count=count, results=[{'pass': True} for _ in range(count)])
            validate_result(result, manifest, 'runtime', fill, True)
            result['results'] = result['results'][:-4]
            result['case_count'] -= 4
            with self.assertRaises(ValueError):
                validate_result(result, manifest, 'runtime', fill, True)

    def test_fill_batch_requires_extended_runtime_coverage(self):
        manifest = {'source_revision': 'pin', 'archive_sha256': 'archive', 'patches': 'patches'}
        for enabled, count in [(False, 224), (True, 256)]:
            result = dict(status='pass', requested_revision='pin', archive_sha256='archive', patch_set='patches',
                          case_count=count, results=[{'pass': True} for _ in range(count)])
            validate_result(result, manifest, 'runtime', enabled)
            with self.assertRaises(ValueError):
                validate_result(result, manifest, 'runtime', not enabled)

    def test_global_memory_ceiling(self):
        base = {'ram_total': 10000, 'ram_available': 500, 'gpu': {'total': 10000, 'used': 9500}}
        check_ceiling(base)
        for key in ['ram', 'gpu']:
            bad = deepcopy(base)
            if key == 'ram':
                bad['ram_available'] -= 1
            else:
                bad['gpu']['used'] += 1
            with self.assertRaises(RuntimeError):
                check_ceiling(bad)

    def test_mismatched_or_incomplete_native_results(self):
        manifest = {'source_revision': 'pin', 'archive_sha256': 'archive', 'patches': 'patches'}
        result = {'status': 'pass', 'requested_revision': 'pin', 'archive_sha256': 'archive',
                  'patch_set': 'patches', 'case_count': 96, 'results': [{'pass': True} for _ in range(96)]}
        validate_result(result, manifest, 'kernels')
        for issue in ['revision', 'archive', 'patches', 'truncated', 'count', 'failed', 'wrong_kind']:
            bad = deepcopy(result)
            if issue == 'revision': bad['requested_revision'] = 'wrong'
            if issue == 'archive': bad['archive_sha256'] = 'wrong'
            if issue == 'patches': bad['patch_set'] = 'wrong'
            if issue == 'truncated': bad['results'].pop()
            if issue == 'count': bad['case_count'] -= 1
            if issue == 'failed': bad['results'][0]['pass'] = False
            with self.subTest(issue=issue), self.assertRaises(ValueError):
                validate_result(bad, manifest, 'graph' if issue == 'wrong_kind' else 'kernels')
        result['status'] = 'error'
        result['results'] = []
        result['case_count'] = 0
        validate_result(result, manifest, 'kernels')  # Preserve an authentic failed run.


if __name__ == '__main__':
    unittest.main()
