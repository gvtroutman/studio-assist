"""Nothing this app starts may outlive it: not the bridge, not what the bridge
started, not when a tab closes, not when the app is killed outright."""

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import studio_agent as eng
import studio_procs as procs

# A child that starts a grandchild, says both pids, and then waits on stdin
# the way a bridge does. The grandchild only sleeps: it is the `node` under
# `npx`, the process a plain Popen.kill() used to leave behind.
TREE = (
    "import subprocess, sys, time\n"
    "g = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
    "print('{\"grandchild\": %d}' % g.pid, flush=True)\n"
    "sys.stdin.read()\n"
    "time.sleep(120)\n"            # ignores end of input: stop() must still end it
)


def gone(pid, within=10.0):
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        if not procs.alive(pid):
            return True
        time.sleep(0.1)
    return False


class TestContainment(unittest.TestCase):
    def spawn_tree(self):
        child = procs.spawn([sys.executable, "-c", TREE], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, text=True,
                            creationflags=procs.NO_WINDOW)
        self.addCleanup(child.kill)
        grandchild = json.loads(child.proc.stdout.readline())["grandchild"]
        self.assertTrue(procs.alive(child.pid))
        self.assertTrue(procs.alive(grandchild))
        return child, grandchild

    def test_stop_ends_the_whole_tree_not_just_the_child(self):
        child, grandchild = self.spawn_tree()
        child.stop(grace=0.5)
        self.assertTrue(gone(child.pid))
        self.assertTrue(gone(grandchild), "the grandchild outlived stop()")
        self.assertNotIn(child, procs.live())

    def test_stop_twice_is_harmless(self):
        child, _ = self.spawn_tree()
        child.stop(0.2)
        child.stop(0.2)
        child.kill()

    def test_a_killed_parent_takes_its_tree_with_it(self):
        """The crash case: no atexit, no finally, no close(). The job handle
        dies with the parent and the kernel ends everything in the job."""
        parent_code = (
            "import subprocess, sys\n"
            "sys.path.insert(0, %r)\n"
            "import studio_procs\n"
            "c = studio_procs.spawn([sys.executable, '-c', %r], stdin=subprocess.PIPE,\n"
            "                       stdout=subprocess.PIPE, text=True)\n"
            "import json\n"
            "print(c.pid, json.loads(c.proc.stdout.readline())['grandchild'], flush=True)\n"
            "sys.stdin.read()\n"
        ) % (os.path.dirname(os.path.dirname(os.path.abspath(__file__))), TREE)
        parent = subprocess.Popen([sys.executable, "-c", parent_code],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  text=True, creationflags=procs.NO_WINDOW)
        try:
            child, grandchild = map(int, parent.stdout.readline().split())
            self.assertTrue(procs.alive(child) and procs.alive(grandchild))
            parent.kill()                 # TerminateProcess: nothing runs after it
            parent.wait(10)
            self.assertTrue(gone(child), "the child outlived a killed parent")
            self.assertTrue(gone(grandchild), "the grandchild outlived a killed parent")
        finally:
            for s in (parent.stdin, parent.stdout):
                s.close()

    def test_stop_all_ends_every_live_child(self):
        a, ga = self.spawn_tree()
        b, gb = self.spawn_tree()
        procs.stop_all(0.2)
        for pid in (a.pid, ga, b.pid, gb):
            self.assertTrue(gone(pid), pid)

    def test_alive_only_looks(self):
        self.assertTrue(procs.alive(os.getpid()))
        self.assertTrue(procs.alive(os.getpid()))  # still here: it did not kill us


class TestBridgeClients(unittest.TestCase):
    def test_closing_an_mcp_client_ends_the_tree_under_the_bridge(self):
        client = eng.MCPClient(sys.executable, ["-c", TREE])
        grandchild = client._inbox.get(timeout=20)["grandchild"]
        client.close(grace=0.5)
        self.assertTrue(gone(client.proc.pid))
        self.assertTrue(gone(grandchild))

    def test_a_com_host_makes_no_folder_until_it_is_used(self):
        """Every bridge module builds its host at import; the folder used to
        be made then, and one was left in %TEMP% per import."""
        import studio_com
        before = set(os.listdir(tempfile.gettempdir()))
        host = studio_com.ComHost("No.Such.ProgID", "Nothing", "Nothing.exe")
        self.assertIsNone(host.dir)
        host.close()
        after = set(os.listdir(tempfile.gettempdir()))
        self.assertFalse({n for n in after - before if n.startswith("studio_com_")})

    @unittest.skipUnless(sys.platform == "win32", "PowerShell worker")
    def test_a_used_com_host_removes_its_folder_and_worker_on_close(self):
        import studio_com
        host = studio_com.ComHost("Studio.NoSuchApp.Test", "Nothing", "Nothing.exe")
        try:
            host.run("return 1", timeout=20)
        except studio_com.ComError:
            pass
        folder, pid = host.dir, host.proc and host.proc.pid
        self.assertTrue(folder and os.path.isdir(folder))
        host.close()
        self.assertFalse(os.path.exists(folder))
        if pid:
            self.assertTrue(gone(pid))


if __name__ == "__main__":
    unittest.main()
