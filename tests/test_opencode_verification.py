"""A checkpoint and an idle session cannot turn missing verification into success."""
import json
import os
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import apps.opencode.mcp as oc
import core.mcp as studio_mcp
from tests.test_opencode import Repo, PASSING


class TestVerification(Repo):
    def switch(self, on):
        os.makedirs(oc.STATE_DIR, exist_ok=True)
        with open(os.path.join(oc.STATE_DIR, oc.AGENTIC_FILE), "w", encoding="utf-8") as f:
            json.dump({"on": on}, f)

    def asks(self):
        return [p[2]["parts"][0]["text"] for p in self.fake.posts if p[0] == "ask"]

    def test_unavailable_tests_remain_unresolved_after_checkpoint_and_wait(self):
        self.switch(True)
        for line in ("Tests tests.test_hello did not finish in 120 seconds.",
                     "Tests could not be run: access denied"):
            with self.subTest(line=line):
                self.fake.scripts = [[("write", ("tests/test_hello.py", PASSING + "# new\n")),
                                      ("say", "done")]] + [[("say", "done")]] * oc.AGENTIC_ROUNDS
                with patch.object(oc, "run_tests", return_value=line) as run:
                    res = self.user().call_tool("opencode_ask", {"prompt": "change the test", "new_session": True})
                sid = oc.last_session()
                self.assertTrue(res.get("isError"), res)
                self.assertEqual(res["_meta"]["studio/opencode"]["state"], "needs_input")
                self.assertEqual(run.call_count, 1 + oc.AGENTIC_ROUNDS)
                self.assertEqual(len(oc.task(sid)["checkpoints"]), 1)
                self.assertEqual(oc.task(sid)["verification"]["result"], line)
                before = len(self.asks())
                waited = self.user().call_tool("opencode_wait", {"session_id": sid})
                self.assertEqual(waited["_meta"]["studio/opencode"]["state"], "needs_input")
                self.assertEqual(len(self.asks()), before)
                self.assertNotIn("is done.", self.text(waited))

    def test_failed_tests_cannot_disappear_when_the_repair_changes_nothing(self):
        self.switch(True)
        self.fake.scripts = [[("write", ("tests/test_hello.py", PASSING + "# new\n")), ("say", "done")]]
        with patch.object(oc, "run_tests", return_value="Tests FAILED: tests.test_hello (1 tests).") as run:
            res = self.user().call_tool("opencode_ask", {"prompt": "change the test"})
        self.assertEqual(res["_meta"]["studio/opencode"]["state"], "needs_input")
        self.assertEqual(run.call_count, 1 + oc.AGENTIC_ROUNDS)
        self.assertIn("the tests failed", self.text(res))

    def test_agentic_off_reports_unverified_work_and_new_ask_rechecks_it(self):
        self.switch(False)
        self.fake.scripts = [[("write", ("tests/test_hello.py", PASSING + "# new\n")), ("say", "done")]]
        with patch.object(oc, "run_tests", return_value="Tests could not be run: unavailable"):
            res = self.user().call_tool("opencode_ask", {"prompt": "change the test"})
        sid = oc.last_session()
        self.assertEqual(res["_meta"]["studio/opencode"]["state"], "needs_input")
        self.assertNotIn("repair limit", self.text(res))
        self.assertEqual(len(self.asks()), 1)
        waited = self.user().call_tool("opencode_wait", {"session_id": sid})
        self.assertEqual(waited["_meta"]["studio/opencode"]["state"], "needs_input")
        self.fake.scripts = [[("say", "Verification can run now.")]]
        with patch.object(oc, "run_tests", return_value="Tests passed: tests.test_hello (1 tests).") as run:
            res = self.user().call_tool("opencode_ask", {"prompt": "Run the tests again"})
        self.assertEqual(res["_meta"]["studio/opencode"]["state"], "done")
        self.assertEqual(run.call_args.args[1], ["tests.test_hello"])
        self.assertFalse(oc.task(sid).get("owed"))

    def test_unrelated_edit_also_rechecks_the_unresolved_test_modules(self):
        self.switch(False)
        self.fake.scripts = [[("write", ("hello.py", "print('bye')\n")), ("say", "done")]]
        with patch.object(oc, "run_tests", return_value="Tests FAILED: tests.test_hello (1 tests)."):
            self.user().call_tool("opencode_ask", {"prompt": "change the greeting"})
        self.fake.scripts = [[("write", ("README.md", "Updated notes\n")), ("say", "updated")]]
        with patch.object(oc, "run_tests", return_value="Tests passed: tests.test_hello (1 tests).") as run:
            res = self.user().call_tool("opencode_ask", {"prompt": "update the readme"})
        self.assertEqual(res["_meta"]["studio/opencode"]["state"], "done")
        self.assertEqual(run.call_args.args[1], ["tests.test_hello"])

    def test_copy_inspection_error_is_not_done_without_agentic(self):
        self.switch(False)
        self.fake.scripts = [[("say", "done")]]
        with patch.object(oc, "changed_files", side_effect=oc.OpenCodeError("git unavailable")):
            res = self.user().call_tool("opencode_ask", {"prompt": "check the task"})
        self.assertEqual(res["_meta"]["studio/opencode"]["state"], "needs_input")
        self.assertIn("could not be checked", self.text(res))


class TestTestRunner(unittest.TestCase):
    def test_timeout_ends_a_real_test_and_its_descendant(self):
        source = ("import subprocess, sys, time, unittest\n"
                  "class T(unittest.TestCase):\n"
                  "    def test_sleep(self):\n"
                  "        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], "
                  "creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))\n"
                  "        with open('child.pid', 'w') as f: f.write(str(child.pid))\n"
                  "        print('waiting for descendant', flush=True)\n"
                  "        time.sleep(60)\n")
        with tempfile.TemporaryDirectory() as root:
            os.mkdir(os.path.join(root, "tests"))
            for name, text in (("__init__.py", ""), ("test_sleep.py", source)):
                with open(os.path.join(root, "tests", name), "w", encoding="utf-8") as f:
                    f.write(text)
            with patch.object(oc, "TEST_TIMEOUT", 5), patch.object(oc, "TEST_LOOK", 0.1):
                line = oc.run_tests(root, ["tests.test_sleep"])
            self.assertIn("did not finish", line)
            self.assertIn("waiting for descendant", line)
            with open(os.path.join(root, "child.pid"), encoding="utf-8") as f:
                self.assertFalse(oc.procs.alive(int(f.read())), "a timed-out test's child survives")

    def test_timeout_preserves_diagnostics_and_kills_the_test_process(self):
        child = Mock()
        child.proc.communicate.side_effect = [subprocess.TimeoutExpired("tests", 1),
                                             (b"setup\nlast useful diagnostic", None)]
        with patch.object(oc.procs, "spawn", return_value=child), \
                patch.object(oc.time, "monotonic", side_effect=[0, oc.TEST_TIMEOUT + 1]), \
                patch.object(studio_mcp, "cancelled", return_value=False):
            line = oc.run_tests("copy", ["tests.test_hello"])
        self.assertIn("did not finish", line)
        self.assertIn("last useful diagnostic", line)
        child.terminate.assert_called_once()
        child.stop.assert_called_once_with(grace=0)

    def test_success_also_releases_the_job_holding_test_descendants(self):
        child = Mock()
        child.proc.communicate.return_value = (b"Ran 2 tests\nOK", None)
        child.proc.returncode = 0
        with patch.object(oc.procs, "spawn", return_value=child):
            line = oc.run_tests("copy", ["tests.test_hello"])
        self.assertIn("Tests passed", line)
        child.stop.assert_called_once_with(grace=0)

    def test_stop_releases_the_test_job_before_leaving(self):
        child = Mock()
        child.proc.communicate.side_effect = [subprocess.TimeoutExpired("tests", 1), (b"", None)]
        with patch.object(oc.procs, "spawn", return_value=child), \
                patch.object(studio_mcp, "cancelled", return_value=True):
            with self.assertRaises(oc.Stopped):
                oc.run_tests("copy", ["tests.test_hello"])
        child.stop.assert_called_once_with(grace=0)

    def test_checks_without_matching_tests_are_not_mislabelled_failed(self):
        self.assertIsNone(oc.verification_problem(["No test module matches the changed files, so none were run."]))
        self.assertIsNone(oc.verification_problem(["Tests passed: tests.test_hello (1 tests).", "Checkpoint saved."]))
