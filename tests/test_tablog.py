"""Each tab's own log (core.tablog): which tab a line belongs to, the lines
kept per tab, the activity log's `[tab]` column, and a bridge's stderr filed
under the tab that started it."""

import logging
import os
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import core.agent as eng
import core.doctor as doctor
import core.tablog as tablog


class TestTabLog(unittest.TestCase):
    def setUp(self):
        self.book = tablog.Book(keep=5)
        self.logger = logging.getLogger("studio.test_tablog")
        self.logger.addHandler(self.book)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

    def tearDown(self):
        self.logger.removeHandler(self.book)

    def texts(self, tab):
        return [e["text"] for e in self.book.lines(tab)]

    def test_a_line_is_filed_under_the_tab_its_thread_works_for(self):
        with tablog.working_for("comfyui"):
            self.logger.info("rendering")
        self.logger.info("the window's own")
        self.assertEqual(self.texts("comfyui"), ["rendering"])
        self.assertEqual(self.texts(None), ["the window's own"])

    def test_working_for_nests_and_restores(self):
        with tablog.working_for("resolve"):
            with tablog.working_for("chat"):
                self.assertEqual(tablog.current(), "chat")
            self.assertEqual(tablog.current(), "resolve")
        self.assertIsNone(tablog.current())

    def test_another_thread_does_not_inherit_the_tab(self):
        """What `MCPClient` has to work round: a thread started from inside a
        tab's work starts with no tab of its own."""
        seen = []
        with tablog.working_for("opencode"):
            t = threading.Thread(target=lambda: seen.append(tablog.current()))
            t.start()
            t.join()
        self.assertEqual(seen, [None])

    def test_an_explicit_tab_wins_over_the_thread(self):
        with tablog.working_for("resolve"):
            self.logger.info("for chat", extra={"tab": "chat"})
        self.assertEqual(self.texts("chat"), ["for chat"])
        self.assertEqual(self.texts("resolve"), [])

    def test_only_the_last_lines_are_kept_and_clear_empties_one_tab(self):
        with tablog.working_for("chat"):
            for i in range(8):
                self.logger.info("line %d", i)
        with tablog.working_for("resolve"):
            self.logger.info("kept")
        self.assertEqual(self.texts("chat"), ["line %d" % i for i in range(3, 8)])
        self.book.clear("chat")
        self.assertEqual(self.texts("chat"), [])
        self.assertEqual(self.texts("resolve"), ["kept"])

    def test_every_listener_hears_every_line_and_a_broken_one_costs_nothing(self):
        """Two windows (the tests build several) both hear; one whose queue
        is gone neither stops the others nor the line being filed."""
        first, second = [], []

        def broken(tab, entry):
            raise RuntimeError("queue gone")
        hear_first = lambda tab, entry: first.append((tab, entry["text"]))
        self.book.listen(hear_first)
        self.book.listen(broken)
        self.book.listen(lambda tab, entry: second.append((tab, entry["text"])))
        with tablog.working_for("chat"):
            self.logger.warning("careful")
        self.assertEqual(first, [("chat", "careful")])
        self.assertEqual(second, [("chat", "careful")])
        self.book.unlisten(hear_first)
        self.logger.info("still filed")
        self.assertEqual(self.texts(None), ["still filed"])
        self.assertEqual(len(first), 1)

    def test_tab_of_takes_every_id_an_event_carries(self):
        self.assertEqual(tablog.tab_of("chat"), "chat")
        self.assertEqual(tablog.tab_of(("chat", "abc123")), "chat")
        self.assertIsNone(tablog.tab_of(None))
        self.assertIsNone(tablog.tab_of(()))

    def test_a_line_reads_as_time_level_source_and_text(self):
        entry = {"time": time.mktime((2026, 9, 28, 12, 4, 31, 0, 0, -1)),
                 "level": "WARNING", "source": "studio.agent", "text": "tool x failed"}
        self.assertEqual(tablog.format_line(entry), "12:04:31  WARN  agent  tool x failed")
        entry.update(source="studio.tab", level="INFO", text="ready")
        self.assertEqual(tablog.format_line(entry), "12:04:31  INFO         ready")

    def test_install_opens_the_app_logger_once(self):
        logger = logging.getLogger("studio")
        before = (logger.level, list(logger.handlers), list(tablog.BOOK.listeners))
        hear = lambda tab, entry: None
        try:
            tablog.install(hear)
            tablog.install(hear)
            self.assertEqual(logger.handlers.count(tablog.BOOK), 1)
            self.assertEqual(tablog.BOOK.listeners.count(hear), 1)
            self.assertLessEqual(logger.level, logging.INFO)
        finally:
            logger.level, logger.handlers[:], tablog.BOOK.listeners[:] = before


class TestActivityLogColumn(unittest.TestCase):
    def test_the_activity_log_names_each_lines_tab(self):
        logger = logging.getLogger("studio")
        saved = (logger.level, list(logger.handlers), logger.propagate)
        real = os.environ.get("STUDIO_SETTINGS")
        folder = tempfile.mkdtemp()
        os.environ["STUDIO_SETTINGS"] = os.path.join(folder, "settings.json")
        try:
            path = doctor.start_activity_log()
            with tablog.working_for("resolve"):
                logging.getLogger("studio.agent").info("tool timeline 0.2s")
            logging.getLogger("studio").info("window line")
            for h in logger.handlers:
                h.flush()
            with open(path, encoding="utf-8") as f:
                text = f.read()
            self.assertIn("[resolve]", text)
            self.assertIn("[-]", text)
        finally:
            for h in logger.handlers:
                if h not in saved[1]:
                    h.close()
            logger.level, logger.handlers[:], logger.propagate = saved
            if real is None:
                os.environ.pop("STUDIO_SETTINGS", None)
            else:
                os.environ["STUDIO_SETTINGS"] = real


class TestBridgeStderr(unittest.TestCase):
    def test_a_bridges_stderr_is_filed_under_the_tab_that_started_it(self):
        book = tablog.Book()
        logger = logging.getLogger("studio.bridge")
        logger.addHandler(book)
        level = logger.level
        logger.setLevel(logging.INFO)
        try:
            with tablog.working_for("photoshop"):
                client = eng.MCPClient(sys.executable, [
                    "-c", "import sys; sys.stderr.write('COM worker ready\\n')"], quiet=True)
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline and not book.lines("photoshop"):
                time.sleep(0.05)
            client.close()
            self.assertEqual([e["text"] for e in book.lines("photoshop")], ["COM worker ready"])
        finally:
            logger.removeHandler(book)
            logger.setLevel(level)


if __name__ == "__main__":
    unittest.main()
