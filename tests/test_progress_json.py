import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from research.progress_json import write_progress


class ProgressJsonTest(unittest.TestCase):
    @unittest.skipUnless(os.name == 'nt', 'Windows sharing semantics')
    def test_real_windows_reader_lock_then_release(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'report.json'
            target.write_text('{"status":"old"}')
            probe = Path(folder) / 'probe.json'
            probe.write_text('{}')
            reader = target.open('r')
            release = None
            try:
                with self.assertRaises(PermissionError):
                    probe.replace(target)
                release = threading.Timer(0.15, reader.close)
                release.start()
                write_progress(target, {'status': 'completed'}, attempts=30, delay=0.02)
            finally:
                reader.close()
                if release is not None:
                    release.join()
            self.assertEqual(json.loads(target.read_text()), {'status': 'completed'})

    def test_transient_sharing_failure_retries_and_commits(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'report.json'
            target.write_text('{"status":"old"}')
            original = Path.replace
            calls = []
            def transient(source, destination):
                calls.append(source)
                if len(calls) < 3:
                    self.assertEqual(json.loads(target.read_text())['status'], 'old')
                    raise PermissionError('simulated sharing lock')
                return original(source, destination)
            with patch.object(Path, 'replace', transient):
                write_progress(target, {'status': 'completed'}, attempts=3, delay=0)
            self.assertEqual(len(calls), 3)
            self.assertEqual(json.loads(target.read_text()), {'status': 'completed'})

    def test_persistent_failure_preserves_both_documents(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'report.json'
            target.write_text('{"status":"old"}')
            with patch.object(Path, 'replace', side_effect=PermissionError('locked')) as replace:
                with self.assertRaises(PermissionError):
                    write_progress(target, {'status': 'new'}, attempts=2, delay=0)
            self.assertEqual(replace.call_count, 2)
            self.assertEqual(json.loads(target.read_text()), {'status': 'old'})
            self.assertEqual(json.loads(target.with_suffix('.json.tmp').read_text()), {'status': 'new'})


if __name__ == '__main__':
    unittest.main()
