#!/usr/bin/env python3
"""The auto-update pass, against throwaway repositories on disk: a bare
"GitHub", a clone standing in for the workstation, and a second clone that
pushes. No network."""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import studio_update as upd

GIT = shutil.which("git")


def run(cwd, *args):
    subprocess.run(["git", "-C", cwd, *args], check=True, capture_output=True)


def commit(cwd, name, text):
    with open(os.path.join(cwd, name), "w") as f:
        f.write(text)
    run(cwd, "add", name)
    run(cwd, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", name)


@unittest.skipUnless(GIT, "git is not installed")
class UpdateTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        origin = os.path.join(self.dir, "origin.git")
        seed = os.path.join(self.dir, "seed")
        self.pc = os.path.join(self.dir, "pc")
        self.dev = os.path.join(self.dir, "dev")
        run(self.dir, "init", "-q", "--bare", "-b", "main", origin)
        run(self.dir, "init", "-q", "-b", "main", seed)
        commit(seed, "a.txt", "one\n")
        run(seed, "push", "-q", origin, "main")
        run(self.dir, "clone", "-q", origin, self.pc)
        run(self.dir, "clone", "-q", origin, self.dev)
        self._here, self._log = upd.HERE, upd.LOG
        upd.HERE, upd.LOG = self.pc, os.path.join(self.dir, "update.log")
        self.addCleanup(setattr, upd, "HERE", self._here)
        self.addCleanup(setattr, upd, "LOG", self._log)

    def read(self, name):
        with open(os.path.join(self.pc, name)) as f:
            return f.read()

    def logged(self):
        if not os.path.exists(upd.LOG):
            return ""
        with open(upd.LOG) as f:
            return f.read()

    def test_nothing_new_changes_nothing_and_logs_nothing(self):
        self.assertFalse(upd.update())
        self.assertEqual(self.logged(), "")

    def test_fast_forwards_when_github_is_ahead(self):
        commit(self.dev, "a.txt", "two\n")
        run(self.dev, "push", "-q")
        self.assertTrue(upd.update())
        self.assertEqual(self.read("a.txt"), "two\n")
        self.assertIn("Updated", self.logged())

    def test_leaves_local_commits_alone(self):
        commit(self.dev, "a.txt", "two\n")
        run(self.dev, "push", "-q")
        commit(self.pc, "b.txt", "mine\n")
        self.assertFalse(upd.update())
        self.assertEqual(self.read("a.txt"), "one\n")
        self.assertIn("not updating", self.logged())

    def test_leaves_local_edits_alone(self):
        commit(self.dev, "a.txt", "two\n")
        run(self.dev, "push", "-q")
        with open(os.path.join(self.pc, "a.txt"), "w") as f:
            f.write("edited here\n")
        self.assertFalse(upd.update())
        self.assertEqual(self.read("a.txt"), "edited here\n")
        self.assertIn("local edits", self.logged())

    def branch(self):
        return subprocess.run(["git", "-C", self.pc, "branch", "--show-current"],
                              capture_output=True, text=True).stdout.strip()

    def merged_branch(self):
        """The PC on a feature branch that GitHub then merges into main and
        moves past - the state that stopped updates."""
        run(self.pc, "checkout", "-q", "-b", "feature")
        commit(self.pc, "f.txt", "feature\n")
        run(self.pc, "push", "-q", "-u", "origin", "feature")
        run(self.dev, "pull", "-q", "origin", "feature")
        commit(self.dev, "a.txt", "two\n")
        run(self.dev, "push", "-q", "origin", "main")

    def test_even_a_merged_branch_is_never_switched_or_updated(self):
        self.merged_branch()
        st = upd.check()
        self.assertIn("Switch to main when ready", st["problem"])
        self.assertEqual(self.logged(), "")
        self.assertFalse(upd.update())
        self.assertEqual(self.branch(), "feature")
        self.assertEqual((self.read("a.txt"), self.read("f.txt")), ("one\n", "feature\n"))
        self.assertIn("will not switch branches", self.logged())

    def test_stays_on_a_branch_with_commits_main_lacks(self):
        self.merged_branch()
        commit(self.pc, "b.txt", "mine\n")
        self.assertFalse(upd.update())
        self.assertEqual(self.branch(), "feature")
        self.assertEqual(self.read("b.txt"), "mine\n")
        self.assertIn("on feature", self.logged())

    def test_stays_on_a_branch_with_local_edits(self):
        self.merged_branch()
        with open(os.path.join(self.pc, "f.txt"), "w") as f:
            f.write("edited here\n")
        self.assertFalse(upd.update())
        self.assertEqual(self.branch(), "feature")
        self.assertEqual(self.read("f.txt"), "edited here\n")
        self.assertIn("on feature", self.logged())

    def test_detached_head_is_left_alone(self):
        run(self.pc, "checkout", "--detach", "-q")
        commit(self.dev, "a.txt", "two\n")
        run(self.dev, "push", "-q")
        self.assertFalse(upd.update())
        self.assertEqual(self.branch(), "")
        self.assertEqual(self.read("a.txt"), "one\n")
        self.assertIn("detached HEAD", self.logged())

    def test_main_follows_remote_main_even_if_tracking_a_feature(self):
        run(self.dev, "checkout", "-q", "-b", "feature")
        commit(self.dev, "feature.txt", "not released\n")
        run(self.dev, "push", "-q", "-u", "origin", "feature")
        run(self.pc, "fetch", "-q")
        run(self.pc, "branch", "--set-upstream-to=origin/feature", "main")
        run(self.dev, "checkout", "-q", "main")
        commit(self.dev, "a.txt", "released\n")
        run(self.dev, "push", "-q")
        self.assertTrue(upd.update())
        self.assertEqual(self.read("a.txt"), "released\n")
        self.assertFalse(os.path.exists(os.path.join(self.pc, "feature.txt")))
        _, upstream = upd.git(GIT, "rev-parse", "--abbrev-ref", "@{upstream}")
        self.assertEqual(upstream, "origin/feature")

    def test_main_without_tracking_can_use_the_only_named_remote(self):
        run(self.pc, "remote", "rename", "origin", "studio")
        run(self.pc, "branch", "--unset-upstream")
        commit(self.dev, "a.txt", "two\n")
        run(self.dev, "push", "-q")
        self.assertTrue(upd.update())
        self.assertIn("studio/main", self.logged())

    def test_missing_remote_main_is_reported(self):
        run(self.dev, "push", "-q", "origin", "main:released")
        origin = os.path.join(self.dir, "origin.git")
        run(origin, "symbolic-ref", "HEAD", "refs/heads/released")
        run(self.dev, "push", "-q", "origin", "--delete", "main")
        self.assertIn("failed", upd.check()["problem"])
        self.assertFalse(upd.update())
        self.assertEqual(self.read("a.txt"), "one\n")

    def test_switch_between_check_and_pull_is_refused(self):
        commit(self.dev, "a.txt", "two\n")
        run(self.dev, "push", "-q")
        st = upd.check()
        run(self.pc, "checkout", "-q", "-b", "feature")
        with mock.patch.object(upd, "check", return_value=st):
            self.assertFalse(upd.pull()[0])
        self.assertEqual(self.branch(), "feature")
        self.assertEqual(self.read("a.txt"), "one\n")


    def test_check_lists_what_is_new_and_changes_nothing(self):
        commit(self.dev, "a.txt", "two\n")
        commit(self.dev, "c.txt", "three\n")
        run(self.dev, "push", "-q")
        st = upd.check()
        self.assertEqual(st["problem"], "")
        self.assertEqual((st["behind"], st["ahead"]), (2, 0))
        self.assertEqual(st["commits"], ["c.txt", "a.txt"])
        self.assertEqual(self.read("a.txt"), "one\n")
        self.assertEqual(self.logged(), "")

    def test_pull_says_what_happened(self):
        self.assertEqual(upd.pull(), (False, "Already up to date with origin/main."))
        commit(self.dev, "a.txt", "two\n")
        run(self.dev, "push", "-q")
        changed, msg = upd.pull()
        self.assertTrue(changed)
        self.assertIn("Reopen Studio Assist", msg)

    def remote_main(self):
        return subprocess.run(["git", "-C", os.path.join(self.dir, "origin.git"), "rev-parse",
                               "main"], capture_output=True, text=True).stdout.strip()

    def local_main(self):
        return subprocess.run(["git", "-C", self.pc, "rev-parse", "main"],
                              capture_output=True, text=True).stdout.strip()

    def test_pushes_commits_github_lacks(self):
        commit(self.pc, "b.txt", "mine\n")
        self.assertFalse(upd.update())           # the folder itself did not change
        self.assertEqual(self.remote_main(), self.local_main())
        self.assertIn("Pushed 1 commit(s) to origin/main", self.logged())

    def test_pulls_then_pushes_nothing_when_only_github_moved(self):
        commit(self.dev, "a.txt", "two\n")
        run(self.dev, "push", "-q")
        self.assertTrue(upd.update())
        self.assertNotIn("Pushed", self.logged())

    def test_never_pushes_when_both_sides_moved(self):
        commit(self.dev, "a.txt", "two\n")
        run(self.dev, "push", "-q")
        theirs = self.remote_main()
        commit(self.pc, "b.txt", "mine\n")
        self.assertFalse(upd.update())
        self.assertEqual(self.remote_main(), theirs)
        self.assertNotIn("Pushed", self.logged())

    def test_never_pushes_from_another_branch(self):
        run(self.pc, "checkout", "-q", "-b", "feature")
        commit(self.pc, "f.txt", "feature\n")
        before = self.remote_main()
        upd.update()
        self.assertEqual(self.remote_main(), before)

    def test_a_refused_push_is_logged(self):
        commit(self.pc, "b.txt", "mine\n")
        real = upd.git
        fake = lambda exe, *a, **k: (1, "denied") if a[:1] == ("push",) else real(exe, *a, **k)
        with mock.patch.object(upd, "git", side_effect=fake):
            self.assertFalse(upd.push()[0])
        self.assertNotEqual(self.remote_main(), self.local_main())
        self.assertIn("failed: denied", self.logged())


if __name__ == "__main__":
    unittest.main()
