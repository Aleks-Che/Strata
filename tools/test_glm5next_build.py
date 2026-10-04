"""CMake archive preflight; no downloads, compiler, model or GPU."""
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


@unittest.skipUnless(shutil.which('cmake'), 'CMake required for archive preflight tests')
class GLMBuildTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.archive = self.root / 'source with spaces.tar.gz'
        # Preflight checks bytes only, not archive format/provenance/extraction.
        self.archive.write_bytes(b'archive preflight fixture\x00\xff')
        self.sha = hashlib.sha256(self.archive.read_bytes()).hexdigest()
        self.script = Path(__file__).resolve().parents[1] / 'backends/glm5next/SourceArchive.cmake'

    def run_check(self, digest=None, archive=None):
        return subprocess.run([shutil.which('cmake'),
                               f'-DSTRATA_GLM_ARCHIVE={archive or self.archive}',
                               f'-DSTRATA_GLM_ARCHIVE_SHA256={self.sha if digest is None else digest}',
                               '-P', str(self.script)], capture_output=True, text=True, timeout=15)

    def test_matching_hash_and_uppercase_accepted(self):
        for digest in (self.sha, self.sha.upper()):
            result = self.run_check(digest)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_or_malformed_hash_rejected(self):
        for digest in ('', 'a' * 63, 'a' * 65, 'g' * 64):
            result = self.run_check(digest)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('64-digit SHA-256', result.stderr)

    def test_changed_archive_rejected(self):
        self.archive.write_bytes(b'changed bytes')
        result = self.run_check()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('SHA-256 mismatch', result.stderr)

    def test_missing_file_and_directory_rejected(self):
        for path in (self.root / 'missing.tar.gz', self.root):
            result = self.run_check(archive=path)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('must name a local archive', result.stderr)


if __name__ == '__main__':
    unittest.main()
