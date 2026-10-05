"""Output guards for the real-shard checker; no GPU allocations or model needed."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class ShardOutputGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(__file__).resolve().parent.parent
        cls.binary=cls.root/'build-local/step35-cuda/bin/strata-step35-shards-check.exe'
        if not cls.binary.exists():raise unittest.SkipTest('build the Step shard checker first')

    def test_output_guards(self):
        with tempfile.TemporaryDirectory(prefix='step-shard-guards-',dir=self.root/'build-local') as directory:
            work=Path(directory).resolve()
            self.assertTrue(work.is_relative_to(self.root/'build-local'))
            source=work/'input.gguf';source.write_bytes(b'protected fixture input\x00\x01')
            digest=hashlib.sha256(source.read_bytes()).hexdigest()
            manifest=work/'manifest.json'
            manifest.write_text(json.dumps({'samples':[{'path':str(source),'tensor':'bad','expert':0}]}),encoding='utf8')
            original=manifest.read_bytes()
            alias=work/'aliased-report.json';os.link(source,alias)
            for output in (alias,manifest,work/'wrong-extension.gguf',work/'valid-report.json'):
                with self.subTest(output=output.name):
                    result=subprocess.run([str(self.binary),str(manifest),str(output)],capture_output=True,
                        text=True,timeout=10,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                    self.assertEqual(result.returncode,1,result.stderr)
                    self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),digest)
                    self.assertEqual(manifest.read_bytes(),original)
            self.assertFalse((work/'wrong-extension.gguf').exists())
            self.assertEqual(json.loads((work/'valid-report.json').read_text())['status'],'error')


if __name__=='__main__':unittest.main()
