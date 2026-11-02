"""Exercise OpenCode's compaction hook with recorded SDK-shaped task facts."""
import os
import shutil
import subprocess
import unittest


class TestTaskContext(unittest.TestCase):
    def test_continuation_keeps_constraints_failures_and_open_work(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node is required to exercise the OpenCode JavaScript plugin")
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        result = subprocess.run([node, os.path.join(root, "tests", "opencode_task_context.mjs")],
                                cwd=root, capture_output=True, text=True, timeout=20,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("all behavioral checks passed", result.stdout)
