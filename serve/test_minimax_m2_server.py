"""Startup admission must fail before expensive model/CUDA initialization."""
import contextlib
import io
import unittest
from unittest.mock import patch

from serve import minimax_m2_server as server


class StartupTests(unittest.TestCase):
    def test_external_bind_requires_key_before_model_load(self):
        with patch('sys.argv', ['server', '--gguf', 'missing', '--host', '0.0.0.0']), \
             patch.object(server.Tokenizer, 'from_gguf') as load, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as exit:
                server.main()
            self.assertEqual(exit.exception.code, 2)
            load.assert_not_called()

    def test_missing_tool_dependency_fails_before_model_load(self):
        with patch('sys.argv', ['server', '--gguf', 'missing']), \
             patch.object(server, '_schemas', side_effect=ValueError('missing jsonschema')), \
             patch.object(server.Tokenizer, 'from_gguf') as load:
            with self.assertRaisesRegex(ValueError, 'missing jsonschema'):
                server.main()
            load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
