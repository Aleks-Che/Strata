"""Startup admission must fail before expensive model/CUDA initialization."""
import contextlib
import io
import unittest
from unittest.mock import patch

from serve import minimax_m2_server as server


class StartupTests(unittest.TestCase):
    def test_router_ids_require_events_before_model_load(self):
        with patch('sys.argv', ['server', '--gguf', 'missing', '--router-host-ids']), \
             patch.object(server.Tokenizer, 'from_gguf') as load, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as exit:
                server.main()
            self.assertEqual(exit.exception.code, 2)
            load.assert_not_called()

    def test_router_ids_forwarded_and_default_off(self):
        for extra, enabled in [([], False), (['--router-host-ids', '--pipeline-events', '2', '--pipeline-readers', '2'], True)]:
            with patch('sys.argv', ['server', '--gguf', 'fixture', *extra]), \
                 patch.object(server.Tokenizer, 'from_gguf'), patch.object(server, '_schemas'), \
                 patch.object(server, 'MiniMaxAPITemplate'), patch.object(server, 'MiniMaxEngine') as engine, \
                 patch.object(server, 'Service'), patch.object(server, 'make_handler'), \
                 patch.object(server, 'Server') as listener, contextlib.redirect_stdout(io.StringIO()):
                listener.return_value.server_address = ('127.0.0.1', 8080)
                listener.return_value.serve_forever.side_effect = KeyboardInterrupt
                server.main()
                self.assertIs(engine.call_args.kwargs['router_host_ids'], enabled)
                engine.return_value.close.assert_called_once()

    def test_events_require_readers_before_model_load(self):
        with patch('sys.argv', ['server', '--gguf', 'missing', '--pipeline-events', '2']), \
             patch.object(server.Tokenizer, 'from_gguf') as load, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as exit:
                server.main()
            self.assertEqual(exit.exception.code, 2)
            load.assert_not_called()

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
