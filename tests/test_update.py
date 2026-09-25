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

    def test_moves_off_a_merged_branch_onto_main(self):
        self.merged_branch()
        self.assertTrue(upd.update())
        self.assertEqual(self.branch(), "main")
        self.assertEqual((self.read("a.txt"), self.read("f.txt")), ("two\n", "feature\n"))
        self.assertIn("Moved from feature to main", self.logged())
        self.assertFalse(upd.update())              # and stays there quietly after
        commit(self.dev, "a.txt", "three\n")
        run(self.dev, "push", "-q", "origin", "main")
        self.assertTrue(upd.update())
        self.assertEqual(self.read("a.txt"), "three\n")

    def test_stays_on_a_branch_with_commits_main_lacks(self):
        self.merged_branch()
        commit(self.pc, "b.txt", "mine\n")
        self.assertFalse(upd.update())
        self.assertEqual(self.branch(), "feature")
        self.assertIn("1 commit(s) origin/main does not", self.logged())

    def test_stays_on_a_branch_with_local_edits(self):
        self.merged_branch()
        with open(os.path.join(self.pc, "f.txt"), "w") as f:
            f.write("edited here\n")
        self.assertFalse(upd.update())
        self.assertEqual(self.branch(), "feature")
        self.assertEqual(self.read("f.txt"), "edited here\n")
        self.assertIn("has local edits", self.logged())


if __name__ == "__main__":
    unittest.main()
