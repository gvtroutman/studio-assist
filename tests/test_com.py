"""The COM road into Photoshop and Illustrator (apps.adobe.com), without either
app: a stand-in worker answers each request line the way the PowerShell worker
does - `ok` after writing the answer file, `err <why>`, silence, or exiting."""

import os
import queue
import shutil
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import apps.adobe.com as com


class FakeWorker:
    """The worker's pipes. `answer(js)` decides, per request, what the answer
    file holds (None: nothing written) and which line comes back (None: none)."""

    def __init__(self, answer):
        self.answer = answer
        self.lines = queue.Queue()
        self.requests = []
        self.killed = False
        self.stdin = self
        self.stdout = self

    # proc
    def poll(self):
        return None

    # stdin
    def write(self, line):
        js_path, out_path = line.rstrip("\n").split("\t")
        self.requests.append((js_path, out_path))
        with open(js_path, encoding="utf-8") as f:
            text, reply = self.answer(f.read())
        if text is not None:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)
        if reply is not None:
            self.lines.put(reply + "\n")

    def flush(self):
        pass

    # stdout: blocks like a silent worker until the test ends
    def readline(self):
        return self.lines.get()

    # child (core.procs)
    def kill(self):
        self.killed = True

    def stop(self, grace):
        self.killed = True


class ComHostTest(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(com, "process_running", return_value=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.host = com.ComHost("Photoshop.Application", "Photoshop", "Photoshop.exe")
        self.addCleanup(self.host.close)

    def attach(self, answer):
        """Give the host a live stand-in worker instead of starting PowerShell."""
        worker = FakeWorker(answer)
        self.host.dir = tempfile.mkdtemp(prefix="studio_com_test_")
        self.host.js_path = os.path.join(self.host.dir, "call.jsx")
        self.host.out_path = os.path.join(self.host.dir, "answer.txt")
        self.host.child = self.host.proc = worker
        self.addCleanup(worker.lines.put, "")        # free a reader left blocking
        return worker

    def test_building_a_host_makes_no_temp_folder(self):
        self.assertIsNone(self.host.dir)
        self.host.close()                            # nothing to end, nothing to remove
        self.assertIsNone(self.host.dir)

    def test_a_call_runs_the_wrapped_body_and_decodes_its_answer(self):
        worker = self.attach(lambda js: ('{"name": "poster.psd", "w": 1920}', "ok"))
        self.assertEqual(self.host.run("return app.activeDocument.name;"),
                         {"name": "poster.psd", "w": 1920})
        js_path, out_path = worker.requests[0]
        self.assertEqual((js_path, out_path), (self.host.js_path, self.host.out_path))
        with open(js_path, encoding="utf-8") as f:
            sent = f.read()
        self.assertIn("return app.activeDocument.name;", sent)
        self.assertIn("function __J(v)", sent, "every call carries the serializer")

    def test_an_empty_answer_is_none(self):
        self.attach(lambda js: ("  \n", "ok"))
        self.assertIsNone(self.host.run("return;"))

    def test_a_thrown_script_error_is_a_sentence_with_its_line(self):
        self.attach(lambda js: ('{"__error": "No document open", "line": 4}', "ok"))
        with self.assertRaisesRegex(com.ComError, r"^No document open \(line 4\)$"):
            self.host.run("return app.activeDocument;")

    def test_an_answer_that_is_not_json_says_so(self):
        self.attach(lambda js: ("[object Document]", "ok"))
        with self.assertRaisesRegex(com.ComError, "not JSON: \\[object Document\\]"):
            self.host.run("return app.activeDocument;")

    def test_last_calls_answer_file_is_never_read_as_this_ones(self):
        answers = iter([('"first"', "ok"), (None, "ok")])
        self.attach(lambda js: next(answers))
        self.assertEqual(self.host.run("return 'first';"), "first")
        with self.assertRaisesRegex(com.ComError, "no answer file from Photoshop"):
            self.host.run("return 'second';")

    def test_a_com_refusal_is_explained_and_keeps_the_worker(self):
        worker = self.attach(lambda js: (None, "err Call was rejected by callee. "
                                               "(Exception from HRESULT: 0x80010001)"))
        with self.assertRaisesRegex(com.ComError, "Photoshop is busy"):
            self.host.run("return 1;")
        self.assertFalse(worker.killed)
        self.assertIs(self.host.proc, worker)

    def test_com_errors_become_sentences(self):
        explain = self.host._explain
        self.assertIn("could not be started through COM",
                      explain("Retrieving the COM class factory failed: 80080005"))
        self.assertIn("not registered on this machine (Photoshop.Application)",
                      explain("Invalid class string"))
        self.assertIn("PHOTOSHOP_PROGID", explain("Invalid class string"))
        self.assertEqual(explain("Type mismatch"), "Photoshop refused the call: Type mismatch")

    def test_a_silent_worker_is_killed_and_a_dialog_named(self):
        worker = self.attach(lambda js: (None, None))
        with self.assertRaisesRegex(com.ComError, "did not answer within 0 s. A dialog may be open"):
            self.host.run("return 1;", timeout=0)
        self.assertTrue(worker.killed)
        self.assertIsNone(self.host.proc, "the next call starts a fresh worker")

    def test_a_worker_that_exits_mid_call_is_dropped(self):
        worker = self.attach(lambda js: (None, ""))
        with self.assertRaisesRegex(com.ComError, "exited during the call"):
            self.host.run("return 1;")
        self.assertTrue(worker.killed)
        self.assertIsNone(self.host.proc)

    def test_a_cold_app_gets_the_launch_grace(self):
        self.attach(lambda js: (None, None))
        com.process_running.return_value = False
        with mock.patch.object(self.host, "_readline", return_value=None) as readline:
            with self.assertRaisesRegex(com.ComError, "within %d s" % (5 + com.LAUNCH_GRACE)):
                self.host.run("return 1;", timeout=5)
        readline.assert_called_once_with(5 + com.LAUNCH_GRACE)

    def test_close_ends_the_worker_and_removes_the_folder(self):
        worker = self.attach(lambda js: ('1', "ok"))
        self.host.run("return 1;")
        folder = self.host.dir
        self.host.close()
        self.assertTrue(worker.killed)
        self.assertFalse(os.path.exists(folder))
        self.assertIsNone(self.host.dir)


class HelpersTest(unittest.TestCase):
    def test_rgb_takes_six_hex_digits_with_or_without_the_hash(self):
        self.assertEqual(com.rgb("#FF8000"), (255, 128, 0))
        self.assertEqual(com.rgb("0a0b0c"), (10, 11, 12))

    def test_rgb_refuses_anything_else_as_a_com_error(self):
        # "-fffff" and " fffff" used to parse: int() takes a sign and spaces.
        for bad in ("-fffff", "+fffff", " fffff", "#fff", "#ffffff\n", "##ffffff",
                    "gggggg", "", None, 0xffffff):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(com.ComError, "colour must be #RRGGBB"):
                    com.rgb(bad)

    def test_strings_and_paths_become_extendscript_literals(self):
        self.assertEqual(com.js_str('say "hi"\n'), '"say \\"hi\\"\\n"')
        self.assertEqual(com.js_str(3), '"3"')
        path = com.js_path(os.path.join("renders", "out.png"))
        self.assertNotIn("\\\\", path)
        self.assertTrue(path.endswith('/renders/out.png"'))

    def test_wait_for_file_waits_for_the_size_to_settle(self):
        folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, folder, True)
        target = os.path.join(folder, "export.png")
        with open(target, "wb") as f:
            f.write(b"x" * 10)
        self.assertTrue(com.wait_for_file(target, timeout=2))
        self.assertFalse(com.wait_for_file(os.path.join(folder, "never.png"), timeout=0.3))


if __name__ == "__main__":
    unittest.main()
