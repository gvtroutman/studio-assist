#!/usr/bin/env python3
"""The diagnostics report. Nothing here touches the network or a display:
the host probe is swapped for one with a table of answers, because the whole
point of `--doctor` is that it runs when the rest of the app cannot."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import studio_agent as eng
import studio_doctor as doctor


class Fake:
    """A session, as much of one as `tab_rows` reads."""

    def __init__(self, name, prefix_tokens=None, window=None):
        self.app = type("App", (), {"name": name})()
        self.prefix_tokens, self.window = prefix_tokens, window


class DoctorTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self._real = os.environ.get("STUDIO_SETTINGS")
        os.environ["STUDIO_SETTINGS"] = os.path.join(self.dir, "settings.json")
        self._probe, self._alive = eng.probe_models, eng.host_alive
        self._window = eng.context_window
        self.addCleanup(self._restore)

    def _restore(self):
        eng.probe_models, eng.host_alive = self._probe, self._alive
        eng.context_window = self._window
        if self._real is None:
            os.environ.pop("STUDIO_SETTINGS", None)
        else:
            os.environ["STUDIO_SETTINGS"] = self._real

    def _host(self, ok=True, loaded=("m",), ids=("m",), vision=(), err=None,
              window=(32768, 262144)):
        eng.probe_models = lambda *a, **k: (ok, list(loaded), list(ids),
                                            list(vision), err)
        eng.context_window = lambda *a, **k: window
        eng.host_alive = lambda *a, **k: None

    def _roles(self, rows):
        return [r[2] for r in rows]

    # ------------------------------------------------------------- where
    def test_everything_is_kept_beside_the_settings_file(self):
        """STUDIO_SETTINGS moves the whole lot, which is what keeps a test
        run out of the window the user has open."""
        self.assertTrue(doctor.data_dir().startswith(self.dir))
        for path in (doctor.error_log_path(), doctor.tasks_dir()):
            self.assertTrue(path.startswith(self.dir), path)

    def test_an_unwritable_settings_path_is_called_out(self):
        os.environ["STUDIO_SETTINGS"] = os.path.join(
            self.dir, "wall", "settings.json")
        with open(os.path.join(self.dir, "wall"), "w") as f:
            f.write("")               # a file where the directory would have to be
        rows = doctor.storage_rows()
        self.assertIn("err", self._roles(rows))
        self.assertTrue(any("not writable" in r[1] for r in rows), rows)

    # -------------------------------------------------------------- host
    def test_an_unreachable_host_says_which_kind_of_unreachable(self):
        """The difference is the whole diagnosis, and a firewall makes both
        read as "timed out": a PC that is off answers nothing, a PC that is up
        with the server stopped answers the tailnet and drops the port."""
        self._host(ok=False, loaded=(), ids=(), err="timed out")
        eng.host_alive = lambda *a, **k: True
        rows = doctor.host_rows("http://h/v1")
        self.assertTrue(any("start the server" in r[1] for r in rows), rows)

        eng.host_alive = lambda *a, **k: False
        rows = doctor.host_rows("http://h/v1")
        self.assertTrue(any("asleep" in r[1] for r in rows), rows)
        self.assertEqual(max(self._roles(rows), key=["muted", "warn", "err"].index),
                         "err")

    def test_the_just_in_time_default_window_is_flagged(self):
        """8,192 is what LM Studio loads at when nobody says otherwise, and it
        is under some tabs' briefing alone - the root of "the model loops"."""
        self._host(window=(8192, 262144))
        rows = doctor.host_rows("http://h/v1")
        self.assertIn("warn", self._roles(rows))
        self.assertTrue(any("just-in-time default" in r[1] for r in rows), rows)

    def test_a_roomy_window_is_not_flagged(self):
        self._host(vision=("v",), window=(32768, 262144))
        rows = doctor.host_rows("http://h/v1")
        self.assertNotIn("warn", self._roles(rows))
        self.assertNotIn("err", self._roles(rows))

    def test_a_host_with_no_vision_model_is_worth_a_warning(self):
        self._host(vision=())
        self.assertIn("warn", self._roles(doctor.host_rows("http://h/v1")))
        self._host(vision=("v",))
        self.assertNotIn("warn", self._roles(doctor.host_rows("http://h/v1")))

    # -------------------------------------------------------------- tabs
    def test_a_tab_with_no_room_left_is_an_error_not_a_note(self):
        """A prefix that leaves under ROOM is the tab that loses its own tool
        results to truncation and calls the same read ten times."""
        rows = doctor.tab_rows([Fake("After Effects", 20000, 24576)])
        self.assertIn("err", self._roles(rows))
        self.assertTrue(any("truncation" in r[1] for r in rows), rows)

    def test_a_tab_with_room_reads_as_fine(self):
        rows = doctor.tab_rows([Fake("ComfyUI", 7707, 32768)])
        self.assertEqual(self._roles(rows), ["ok"])
        self.assertIn("25,061 left", rows[0][1])

    def test_a_tab_that_has_not_warmed_up_says_so_rather_than_guessing(self):
        rows = doctor.tab_rows([Fake("Chat")])
        self.assertEqual(self._roles(rows), ["muted"])
        self.assertIn("not measured", rows[0][1])

    # ------------------------------------------------------------ errors
    def test_recent_errors_are_newest_first_and_name_the_failure(self):
        doctor.log_error("Traceback...\nRuntimeError: the older one")
        doctor.log_error("Traceback...\nTclError: the newer one")
        rows = doctor.recent_errors()
        self.assertIn("the newer one", rows[0][1])
        self.assertIn("the older one", rows[1][1])

    def test_no_log_is_good_news_not_a_failure(self):
        self.assertEqual(self._roles(doctor.recent_errors()), ["ok"])

    # ------------------------------------------------------------ report
    def test_the_report_ranks_itself_for_an_exit_code(self):
        self._host(window=(8192, 262144))
        sections = doctor.report("http://h/v1", sessions=[])
        self.assertEqual(doctor.worst(sections), 1)       # worth a look
        self._host(ok=False, loaded=(), ids=(), err="timed out")
        self.assertEqual(doctor.worst(doctor.report("http://h/v1")), 2)

    def test_open_tabs_appear_only_when_there_are_sessions_to_report(self):
        self._host()
        self.assertNotIn("Open tabs", [t for t, _ in doctor.report("http://h/v1")])
        self.assertIn("Open tabs",
                      [t for t, _ in doctor.report("http://h/v1", sessions=[])])

    def test_the_console_rendering_marks_every_row(self):
        self._host(window=(8192, 262144))
        text = doctor.as_text(doctor.report("http://h/v1"))
        self.assertIn("Inference host", text)
        self.assertIn("!", text)          # the warning about the default window
        for line in text.splitlines():
            if line.startswith("  "):
                self.assertIn(line[2], doctor.MARKS.values(), line)




class ModuleBoundaryTest(unittest.TestCase):
    """The split is only worth having if it is enforced. These are the rules
    the modules were pulled out of studio_chat to keep."""

    HEADLESS = ("studio_doctor", "studio_files")

    def test_the_headless_modules_load_with_no_tkinter_at_all(self):
        """The real invariant, tested the real way. A guarded probe inside a
        function is fine and wanted - `python_rows` reports a missing tkinter,
        which is the whole point of being asked - so what matters is that
        nothing needs it at import time. Without this the split is a comment."""
        import importlib

        class Block:
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "tkinter" or fullname.startswith("tkinter."):
                    raise ImportError("no tkinter on this Python")
                return None

        saved = {k: v for k, v in sys.modules.items()
                 if k == "tkinter" or k.startswith("tkinter.") or k in self.HEADLESS}
        for k in saved:
            del sys.modules[k]
        sys.meta_path.insert(0, Block())
        try:
            for name in self.HEADLESS:
                with self.subTest(module=name):
                    importlib.import_module(name)       # must not raise
            rows = sys.modules["studio_doctor"].python_rows()
            tk_row = [r for r in rows if r[0] == "Tkinter"][0]
            self.assertEqual(tk_row[2], "err")
            self.assertIn("the window cannot open", tk_row[1])
        finally:
            sys.meta_path.pop(0)
            for k in list(sys.modules):
                if k in self.HEADLESS:
                    del sys.modules[k]
            sys.modules.update(saved)
            importlib.import_module("studio_doctor")

    def test_a_module_pulled_out_of_studio_chat_never_imports_it_back(self):
        import ast
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for name in self.HEADLESS + ("studio_ui",):
            with self.subTest(module=name):
                with open(os.path.join(here, name + ".py"), encoding="utf-8") as f:
                    tree = ast.parse(f.read())
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        self.assertNotIn("studio_chat", [a.name for a in node.names])
                    elif isinstance(node, ast.ImportFrom):
                        self.assertNotEqual(node.module, "studio_chat")

    def test_studio_chat_still_answers_for_the_names_it_re_exports(self):
        """The rest of the app reaches for these where it always did; moving
        them must not have been a rename."""
        import studio_chat
        for name in ("settings_path", "error_log_path", "log_error", "ERROR_LOG",
                     "LOG_MAX_BYTES", "is_picture", "image_dims",
                     "describe_attachment", "attachment_note", "this_pc",
                     "DARK", "LIGHT", "THEMES", "blend", "rounded", "clip",
                     "pretty_host", "Pill"):
            with self.subTest(name=name):
                self.assertTrue(hasattr(studio_chat, name))


if __name__ == "__main__":
    unittest.main()
