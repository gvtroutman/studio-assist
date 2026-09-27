"""Console windows opened outside the app, hidden and held in the Terminal tab.

The Holder and the diff are tested against fakes. The reader is tested live
against a console this test starts itself, hidden - never one of the user's.
The tab is tested in a real window with a fake Holder, and the watcher never
runs there: it only runs in the copy that holds the single-instance lock."""

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.consoles as consoles


class FakeDesktop:
    """Console windows as `windows()` reports them, and what was done to them."""

    def __init__(self):
        self.wins = {}
        self.hidden, self.shown = [], []

    def add(self, hwnd, pid, title="cmd", visible=True):
        self.wins[hwnd] = {"hwnd": hwnd, "pid": pid, "title": title, "visible": visible}

    def close(self, hwnd):
        self.wins.pop(hwnd, None)

    def list(self):
        return [dict(w) for w in self.wins.values()]

    def hide(self, hwnd):
        self.hidden.append(hwnd)
        if hwnd in self.wins:
            self.wins[hwnd]["visible"] = False

    def show(self, hwnd, front=False):
        self.shown.append(hwnd)
        if hwnd in self.wins:
            self.wins[hwnd]["visible"] = True

    def holder(self, ledger):
        return consoles.Holder(ledger=ledger, list_windows=self.list, hide_fn=self.hide,
                               show_fn=self.show, exists_fn=lambda h: h in self.wins)


class HolderTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.ledger = consoles.Ledger(os.path.join(self.dir, "held.json"))
        self.desk = FakeDesktop()
        self.h = self.desk.holder(self.ledger)

    def test_a_visible_console_is_hidden_and_held_and_written_down(self):
        self.desk.add(1, 100, "ComfyUI")
        taken, gone = self.h.sweep()
        self.assertEqual([t["hwnd"] for t in taken], [1])
        self.assertEqual(self.desk.hidden, [1])
        self.assertIn(1, self.h.held)
        self.assertEqual(self.ledger.load(), {1: 100})
        self.assertEqual(self.h.sweep(), ([], []), "a held one is not taken twice")

    def test_hidden_consoles_and_our_own_are_left_alone(self):
        self.desk.add(1, 100, visible=False)          # Logitech's, Epic's
        self.desk.add(2, os.getpid())
        self.assertEqual(self.h.sweep(), ([], []))
        self.assertEqual(self.desk.hidden, [])

    def test_a_given_back_window_is_not_taken_again_while_it_lives(self):
        self.desk.add(1, 100)
        self.h.sweep()
        self.h.release(1)
        self.assertEqual(self.desk.shown, [1])
        self.assertEqual(self.h.sweep(), ([], []))
        self.assertEqual(self.ledger.load(), {})
        self.desk.close(1)
        self.h.sweep()
        self.assertNotIn(1, self.h.released, "forgotten once it closes")

    def test_a_closed_console_is_reported_gone(self):
        self.desk.add(1, 100)
        self.h.sweep()
        self.desk.close(1)
        self.assertEqual(self.h.sweep(), ([], [1]))
        self.assertEqual(self.h.held, {})
        self.assertEqual(self.ledger.load(), {})

    def test_after_a_crash_the_next_copy_takes_back_what_was_hidden(self):
        self.desk.add(1, 100)
        self.desk.add(2, 200)
        self.h.sweep()
        # the app dies here: nothing shown again, the ledger still says 1 and 2
        self.desk.close(2)
        self.desk.add(3, 300, visible=False)           # hidden, but not by us
        again = self.desk.holder(self.ledger)
        self.assertEqual([t["hwnd"] for t in again.recover()], [1])
        self.assertEqual(self.ledger.load(), {1: 100})
        self.assertEqual(again.sweep(), ([], []))

    def test_a_reused_window_handle_is_not_mistaken_for_ours(self):
        self.ledger.save({1: 100})
        self.desk.add(1, 999, visible=False)
        self.assertEqual(self.desk.holder(self.ledger).recover(), [])

    def test_quitting_shows_everything_and_takes_nothing_after(self):
        self.desk.add(1, 100)
        self.desk.add(2, 200)
        self.h.sweep()
        self.assertEqual(sorted(self.h.stop()), [1, 2])
        self.assertEqual(sorted(self.desk.shown), [1, 2])
        self.desk.add(3, 300)
        self.assertEqual(self.h.sweep(), ([], []))
        self.assertEqual(self.ledger.load(), {})

    def test_nothing_held_writes_nothing(self):
        self.h.release_all()
        self.h.stop()
        self.assertFalse(os.path.exists(self.ledger.path))

    def test_a_ledger_that_is_not_json_is_empty(self):
        with open(self.ledger.path, "w") as f:
            f.write("{nope")
        self.assertEqual(self.ledger.load(), {})


class DiffTest(unittest.TestCase):
    def test_a_new_line_keeps_everything_before_it(self):
        self.assertEqual(consoles.diff(["a", "b"], ["a", "b", "c"]), (0, 2, ["c"]))

    def test_a_progress_bar_rewrites_only_its_line(self):
        self.assertEqual(consoles.diff(["a", "10%"], ["a", "20%"]), (0, 1, ["20%"]))

    def test_a_scrolled_buffer_drops_from_the_top(self):
        old = ["l%d" % i for i in range(10)]
        new = ["l%d" % i for i in range(3, 12)]
        self.assertEqual(consoles.diff(old, new), (3, 7, ["l10", "l11"]))

    def test_a_cleared_screen_is_a_rewrite(self):
        self.assertEqual(consoles.diff(["a", "b"], ["x"]), (0, 0, ["x"]))
        self.assertEqual(consoles.diff([], ["x"]), (0, 0, ["x"]))


# Waits for a line, says it back, then waits for Ctrl+C. "Ignore Ctrl+C" is
# inherited, and a test run from Git Bash or an IDE may have it set; a console
# started from Explorer, like ComfyUI's, does not, so the script clears it.
ECHO = (
    "import time, ctypes\n"
    "ctypes.windll.kernel32.SetConsoleCtrlHandler(None, False)\n"
    "print('ready', flush=True)\n"
    "s = input()\n"
    "print('got ' + s, flush=True)\n"
    "try:\n"
    "    time.sleep(60)\n"
    "except KeyboardInterrupt:\n"
    "    print('interrupted', flush=True)\n"
    "    time.sleep(60)\n"
)


@unittest.skipUnless(consoles.WINDOWS, "console windows are a Windows thing")
class LiveConsoleTest(unittest.TestCase):
    """A console of the test's own, started hidden, read and typed into
    through the reader process exactly as the tab does it."""

    def setUp(self):
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = 0                                # SW_HIDE: nothing flashes
        self.proc = subprocess.Popen([sys.executable, "-c", ECHO], startupinfo=si,
                                     creationflags=subprocess.CREATE_NEW_CONSOLE)
        self.reader = consoles.Reader()
        self.addCleanup(self.proc.wait, 5)
        self.addCleanup(self.proc.kill)
        self.addCleanup(self.reader.stop)
        self.win = self.wait(lambda: next((w for w in consoles.windows()
                                           if w["pid"] == self.proc.pid), None))
        self.assertIsNotNone(self.win, "its console window never appeared")
        self.assertFalse(self.win["visible"])

    def wait(self, fn, within=10.0):
        deadline = time.monotonic() + within
        while time.monotonic() < deadline:
            got = fn()
            if got:
                return got
            time.sleep(0.1)
        return None

    def screen(self):
        return self.reader.ask("read", self.win["hwnd"], self.win["pid"])["lines"]

    def shows(self, text):
        return self.wait(lambda: any(text in line for line in self.screen()))

    def test_read_type_and_interrupt(self):
        self.assertTrue(self.shows("ready"))
        self.reader.ask("type", self.win["hwnd"], self.win["pid"], text="hello there\n")
        self.assertTrue(self.shows("got hello there"))
        self.reader.ask("interrupt", self.win["hwnd"], self.win["pid"])
        self.assertTrue(self.shows("interrupted"))
        self.assertIsNone(self.proc.poll(), "the reader's Ctrl+C did not end it")
        self.assertIsNone(self.reader.child.proc.poll(), "nor the reader itself")

    def test_a_closed_console_is_an_error_and_the_reader_carries_on(self):
        self.assertTrue(self.shows("ready"))
        self.proc.kill()
        self.proc.wait(5)
        self.wait(lambda: not consoles.exists(self.win["hwnd"]))
        with self.assertRaises(OSError):
            self.screen()
        # the reader still answers
        with self.assertRaises(OSError) as e:
            self.reader.ask("read", 1, 1)
        self.assertIn("cannot reach", str(e.exception))


def _headless():
    try:
        import tkinter
        tkinter.Tk().destroy()
        return False
    except Exception:
        return True


class StubReader:
    def __init__(self):
        self.asked = []
        self.lines = ["C:\\>echo hi", "hi", "C:\\>"]

    def ask(self, op, hwnd, pid, **kw):
        self.asked.append((op, hwnd, kw))
        return {"ok": True, "lines": list(self.lines), "cursor": len(self.lines) - 1}

    def stop(self):
        pass


@unittest.skipIf(_headless(), "no display")
class TerminalTabTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import core.chat as studio_chat
        cls.mod = studio_chat
        cls.dir = tempfile.mkdtemp()
        cls._real_settings = os.environ.get("STUDIO_SETTINGS")
        os.environ["STUDIO_SETTINGS"] = os.path.join(cls.dir, "settings.json")
        with open(os.environ["STUDIO_SETTINGS"], "w") as f:
            json.dump({"tabs": ["chat"]}, f)
        cls._saved = {n: getattr(studio_chat.Chat, n) for n in
                      ("_boot_host", "_read_icons", "_boot_session")}
        studio_chat.Chat._boot_host = lambda self, *a, **k: None
        studio_chat.Chat._read_icons = lambda self: None
        studio_chat.Chat._boot_session = lambda self, s: None
        assert not studio_chat._MAIN, "the watcher must not run in a test"
        cls.app = studio_chat.Chat()
        cls.desk = FakeDesktop()
        cls.app.holder = cls.desk.holder(consoles.Ledger(os.path.join(cls.dir, "held.json")))
        for _ in range(10):
            cls.app.update()

    @classmethod
    def tearDownClass(cls):
        cls.app._quit()
        for n, fn in cls._saved.items():
            setattr(cls.mod.Chat, n, fn)
        if cls._real_settings is None:
            os.environ.pop("STUDIO_SETTINGS", None)
        else:
            os.environ["STUDIO_SETTINGS"] = cls._real_settings

    def pump(self, until=lambda: True, within=2.0):
        deadline = time.monotonic() + within
        while True:
            self.app.update()
            if until() or time.monotonic() > deadline:
                return until()
            time.sleep(0.02)

    def sweep(self):
        self.app.q.put(("consoles", None, self.app.holder.sweep()))
        self.pump(lambda: self.app.q.empty() and False, within=0.4)   # the pump is a timer

    def test_a_console_taken_opens_the_tab_mirrors_it_and_takes_input(self):
        app, desk = self.app, self.desk
        self.assertEqual(app.active, "chat")
        desk.add(11, 1100, "ComfyUI (Image Studio, 5090)")
        self.sweep()
        self.assertIn("terminals", app.sessions, "taking a console opens the tab")
        self.assertEqual(app.active, "chat", "beside the tab being looked at")
        view = app.sessions["terminals"].terminals
        self.assertIn("ComfyUI", view.note.cget("text"))
        view.reader = StubReader()

        app._select("terminals")
        self.assertFalse(app.composer.winfo_ismapped(), "no chat composer on a terminal")
        self.assertTrue(self.pump(lambda: "hi\n" in view.text.get("1.0", "end")))
        self.assertIn("1 console held", app.sessions["terminals"].status[0])

        view.entry.insert(0, "dir")
        view._send()
        self.assertTrue(self.pump(lambda: ("type", 11, {"text": "dir\n"}) in view.reader.asked))

        view.reader.lines = view.reader.lines + ["more"]
        self.assertTrue(self.pump(lambda: view.text.get("1.0", "end-1c").endswith("C:\\>\nmore")))

        # the console closes: its chip stays, marked, with what it last showed
        desk.close(11)
        self.sweep()
        self.assertIn(11, view.ended)
        self.assertEqual(view.b_show.text, "Remove")
        self.assertIn("more", view.text.get("1.0", "end"))
        view._show_window()                            # Remove, for a closed one
        self.assertEqual(view.order, [])
        self.assertEqual(desk.shown, [], "a closed window is not shown")

    def test_show_window_gives_it_back_and_off_gives_them_all_back(self):
        app, desk = self.app, self.desk
        desk.add(21, 2100, "cmd")
        desk.add(22, 2200, "powershell")
        self.sweep()
        view = app.sessions["terminals"].terminals
        view.reader = StubReader()
        view._select(21)
        view._show_window()
        self.assertIn(21, desk.shown)
        self.assertTrue(desk.wins[21]["visible"])
        self.assertEqual(view.order, [22])
        self.sweep()
        self.assertEqual(view.order, [22], "not taken again")

        app.hold_var.set(False)
        app._toggle_hold()
        self.assertTrue(desk.wins[22]["visible"])
        self.assertEqual(view.order, [])
        self.assertFalse(app.prefs.get("hold_consoles"))
        app.hold_var.set(True)
        app._toggle_hold()
        self.sweep()
        self.assertEqual(sorted(view.order), [21, 22], "on again takes them all")

        app._close_tab("terminals")
        self.assertTrue(desk.wins[21]["visible"] and desk.wins[22]["visible"])
        desk.close(21)
        desk.close(22)
        self.sweep()


if __name__ == "__main__":
    unittest.main()
