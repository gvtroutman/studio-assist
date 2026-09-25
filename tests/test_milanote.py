"""The Milanote tab: a window in a tab, and files dropped onto it.

No browser is started here. DevTools is a fake on a loopback socket speaking
real WebSocket frames, so the client is exercised end to end; the GUI half
stubs the browser the tab would have opened.
"""

import base64
import hashlib
import json
import os
import socket
import struct
import sys
import tempfile
import threading
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import studio_agent as eng
import studio_milanote as milanote


class FakeDevTools:
    """One WebSocket connection on a loopback port, answering each call with
    `answer(method, params)` and remembering what was asked."""

    def __init__(self, answer):
        self.answer = answer
        self.calls = []
        self.srv = socket.socket()
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(1)
        self.port = self.srv.getsockname()[1]
        self.url = "ws://127.0.0.1:%d/devtools/page/X" % self.port
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        conn, _ = self.srv.accept()
        self.srv.close()
        head = b""
        while b"\r\n\r\n" not in head:
            head += conn.recv(1024)
        key = [l.split(b":", 1)[1].strip() for l in head.split(b"\r\n")
               if l.lower().startswith(b"sec-websocket-key")][0]
        accept = base64.b64encode(hashlib.sha1(
            key + b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11").digest())
        conn.sendall(b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n"
                     b"Connection: Upgrade\r\nSec-WebSocket-Accept: " + accept + b"\r\n\r\n")
        try:
            while True:
                msg = json.loads(self._frame(conn))
                self.calls.append((msg["method"], msg["params"]))
                # An unrelated event first: the client must skip it.
                self._send(conn, json.dumps({"method": "Page.loadEventFired"}))
                reply = {"id": msg["id"]}
                result = self.answer(msg["method"], msg["params"])
                if isinstance(result, Exception):
                    reply["error"] = {"message": str(result)}
                else:
                    reply["result"] = result
                self._send(conn, json.dumps(reply))
        except (OSError, ValueError, IndexError):
            conn.close()

    @staticmethod
    def _frame(conn):
        def read(n):
            out = b""
            while len(out) < n:
                chunk = conn.recv(n - len(out))
                if not chunk:
                    raise OSError("closed")
                out += chunk
            return out
        b0, b1 = read(2)
        n = b1 & 0x7F
        if n == 126:
            n = struct.unpack(">H", read(2))[0]
        elif n == 127:
            n = struct.unpack(">Q", read(8))[0]
        assert b1 & 0x80, "a client frame must be masked"
        mask = read(4)
        return bytes(b ^ mask[i % 4] for i, b in enumerate(read(n))).decode()

    @staticmethod
    def _send(conn, text):
        data = text.encode()
        if len(data) < 126:
            head = bytes([0x81, len(data)])
        else:
            head = bytes([0x81, 126]) + struct.pack(">H", len(data))
        conn.sendall(head + data)


class TestDevTools(unittest.TestCase):
    def test_a_call_skips_events_and_returns_its_result(self):
        fake = FakeDevTools(lambda m, p: {"echo": p})
        dt = milanote.DevTools(fake.url)
        try:
            self.assertEqual(dt.call("Runtime.evaluate", {"expression": "1"}),
                             {"echo": {"expression": "1"}})
            # A long message crosses the 126-byte frame boundary both ways.
            long = "x" * 5000
            self.assertEqual(dt.call("Echo", {"s": long}), {"echo": {"s": long}})
        finally:
            dt.close()

    def test_an_error_is_raised_with_its_message(self):
        fake = FakeDevTools(lambda m, p: RuntimeError("no such method"))
        dt = milanote.DevTools(fake.url)
        try:
            with self.assertRaises(milanote.DevToolsError) as caught:
                dt.call("Nope.nothing")
            self.assertIn("no such method", str(caught.exception))
        finally:
            dt.close()

    def test_a_drop_is_enter_over_drop_with_the_paths(self):
        fake = FakeDevTools(lambda m, p: {})
        dt = milanote.DevTools(fake.url)
        try:
            milanote.drop_files(dt, [r"C:\a.png", r"C:\b.pdf"], 400, 300)
        finally:
            dt.close()
        self.assertEqual([p["type"] for _, p in fake.calls], ["dragEnter", "dragOver", "drop"])
        for method, params in fake.calls:
            self.assertEqual(method, "Input.dispatchDragEvent")
            self.assertEqual((params["x"], params["y"]), (400, 300))
            self.assertEqual(params["data"]["files"], [r"C:\a.png", r"C:\b.pdf"])


class TestBrowser(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def test_the_launch_is_an_app_window_on_its_own_profile_on_loopback(self):
        args = milanote.browser_args("chrome.exe", self.dir, "https://app.milanote.com/")
        self.assertIn("--app=https://app.milanote.com/", args)
        self.assertIn("--user-data-dir=" + self.dir, args)
        self.assertIn("--remote-debugging-port=0", args)
        self.assertIn("--remote-debugging-address=127.0.0.1", args)

    def test_the_port_is_read_from_the_profile(self):
        self.assertIsNone(milanote.read_port(self.dir))
        with open(os.path.join(self.dir, "DevToolsActivePort"), "w") as f:
            f.write("53257\n/devtools/browser/abc\n")
        self.assertEqual(milanote.read_port(self.dir), 53257)

    def test_no_browser_is_a_sentence_not_a_crash(self):
        b = milanote.Browser(exe="", profile=self.dir)
        b.exe = None
        with self.assertRaises(RuntimeError) as caught:
            b.start()
        self.assertIn("Chrome or Edge", str(caught.exception))

    def test_nothing_is_moved_before_it_is_embedded(self):
        """A top-level window fitted to (-left, -top) would sit on the corner
        of the desktop rather than in the tab."""
        b = milanote.Browser(exe="x", profile=self.dir)
        b.hwnd = 12345                    # not a window; fit must not touch it
        b.fit(800, 600)
        self.assertEqual(b.size, (800, 600))
        self.assertIsNone(b.parent)

    def test_an_upload_needs_files_and_an_open_window(self):
        b = milanote.Browser(exe="x", profile=self.dir)
        with self.assertRaises(RuntimeError):
            b.upload([os.path.join(self.dir, "missing.png")])
        present = os.path.join(self.dir, "card.txt")
        with open(present, "w") as f:
            f.write("hi")
        with self.assertRaises(RuntimeError) as caught:
            b.upload([present])
        self.assertIn("not open", str(caught.exception))

    def test_an_upload_lands_in_the_middle_of_the_board(self):
        present = os.path.join(self.dir, "card.txt")
        with open(present, "w") as f:
            f.write("hi")

        def answer(method, params):
            if method == "Runtime.evaluate":
                return {"result": {"value": [1000, 700, "https://app.milanote.com/board/1"]}}
            return {}
        fake = FakeDevTools(answer)
        b = milanote.Browser(exe="x", profile=self.dir)
        b.running = lambda: True
        b.port = lambda wait=0: fake.port
        milanote_page = milanote.milanote_page
        milanote.milanote_page = lambda port: {"url": "https://app.milanote.com/board/1",
                                               "webSocketDebuggerUrl": fake.url}
        try:
            said = b.upload([present])
        finally:
            milanote.milanote_page = milanote_page
        self.assertEqual(said, "dropped 1 file onto the board: card.txt")
        drops = [p for m, p in fake.calls if m == "Input.dispatchDragEvent"]
        self.assertEqual((drops[-1]["x"], drops[-1]["y"]), (500, 350))

    def test_an_upload_to_the_sign_in_page_says_to_sign_in(self):
        present = os.path.join(self.dir, "card.txt")
        with open(present, "w") as f:
            f.write("hi")
        fake = FakeDevTools(lambda m, p: {"result": {"value": [
            1000, 700, "https://app.milanote.com/login"]}})
        b = milanote.Browser(exe="x", profile=self.dir)
        b.running = lambda: True
        b.port = lambda wait=0: fake.port
        real = milanote.milanote_page
        milanote.milanote_page = lambda port: {"url": "https://app.milanote.com/login",
                                               "webSocketDebuggerUrl": fake.url}
        try:
            with self.assertRaises(RuntimeError) as caught:
                b.upload([present])
        finally:
            milanote.milanote_page = real
        self.assertIn("sign in", str(caught.exception))
        self.assertFalse([m for m, _ in fake.calls if m == "Input.dispatchDragEvent"])


class TestRegistry(unittest.TestCase):
    def test_milanote_is_a_tab_but_not_an_app(self):
        m = eng.TABS_BY_ID["milanote"]
        self.assertIs(m, eng.MILANOTE)
        self.assertTrue(m.panel)
        self.assertFalse(m.drivable or m.bridged or m.research)
        self.assertNotIn(m, eng.APPS)
        self.assertNotIn(m.name, eng.DRIVABLE)
        self.assertIs(eng.TABS[-1], eng.CHAT)
        self.assertTrue(m.installed() and m.running())
        self.assertEqual(m.chat_prompt(), "")
        with self.assertRaises(RuntimeError):
            m.connect()

    def test_no_other_tab_is_a_panel(self):
        # The Image Studio is a panel tab too, but one whose body is our own
        # form: it holds no other program's window.
        self.assertEqual([a.id for a in eng.TABS if a.panel], ["milanote", "image-studio"])
        self.assertEqual([a.id for a in eng.TABS if a.panel and not a.images], ["milanote"])


def _headless():
    try:
        import tkinter
        tkinter.Tk().destroy()
        return False
    except Exception:
        return True


class StubBrowser:
    def __init__(self):
        self.fitted, self.uploads, self.released, self.closed = [], [], False, False
        self.size = (1, 1)
        self.inset = (0, 0, 0, 0)

    def running(self):
        return not self.closed

    def embed(self, parent, w, h):
        self.parent = parent
        self.fitted.append((w, h))

    def fit(self, w, h):
        self.fitted.append((w, h))

    def focus(self):
        pass

    def measure(self):
        return self.inset

    def upload(self, paths):
        self.uploads.append(paths)
        return "dropped %d file onto the board: x" % len(paths)

    def release(self):
        self.released = True

    def close(self, grace=0):
        self.closed = True


@unittest.skipIf(_headless(), "no display")
class TestMilanoteTab(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import studio_chat
        cls.mod = studio_chat
        cls.dir = tempfile.mkdtemp()
        cls._real_settings = os.environ.get("STUDIO_SETTINGS")
        os.environ["STUDIO_SETTINGS"] = os.path.join(cls.dir, "settings.json")
        with open(os.environ["STUDIO_SETTINGS"], "w") as f:
            json.dump({"tabs": ["chat", "milanote"]}, f)
        cls._saved = {n: getattr(studio_chat.Chat, n) for n in
                      ("_boot_host", "_read_icons", "_open_panel", "_boot_session")}
        studio_chat.Chat._boot_host = lambda self, *a, **k: None
        studio_chat.Chat._read_icons = lambda self: None
        studio_chat.Chat._boot_session = lambda self, s: None
        cls.browsers = []

        def open_panel(self, s):          # what the worker posts, with a stub
            b = StubBrowser()
            cls.browsers.append(b)
            s.browser = b
            self.q.put(("panel", s.event_id, ("window", b)))
        studio_chat.Chat._open_panel = open_panel
        cls.app = studio_chat.Chat()
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

    def _pump(self, until=lambda: True):
        """Run the queue on this thread until `until()`, a second at most:
        the tab's workers post from threads of their own."""
        import time
        deadline = time.monotonic() + 1.0
        while True:
            self.app._drain()
            self.app.update()
            if until() or time.monotonic() > deadline:
                return
            time.sleep(0.01)

    def _milanote(self):
        if "milanote" not in self.app.sessions:
            self.app._add_tab("milanote")
        self.app._select("milanote")
        s = self.app.sessions["milanote"]
        if not (s.ready or s.booting):
            # test_agent's GUI tests replace Chat._ensure for the whole run;
            # this is what its panel branch does.
            s.booting = True
            self.app._open_panel(s)
        self._pump(lambda: s.ready)
        return s

    def test_the_tab_holds_the_window_and_no_composer(self):
        s = self._milanote()
        self.assertIsNone(s.view)
        self.assertTrue(s.ready)
        self.assertIsInstance(s.browser, StubBrowser)
        self.assertTrue(s.panel_host.winfo_ismapped())
        self.assertFalse(self.app.composer.winfo_ismapped())
        self.assertFalse(self.app.btn_fix.winfo_ismapped())
        self.assertEqual(self.app.btn_new.cget("state"), "disabled")
        self.assertEqual(self.app.btn_hist.cget("state"), "disabled")
        self.app._select("chat")
        self._pump()
        self.assertTrue(self.app.composer.winfo_ismapped())
        self.assertFalse(s.frame.winfo_ismapped())

    def test_a_message_for_the_tab_is_said_on_its_line(self):
        s = self._milanote()
        self.app._handle("sys", None, "the host is back")
        self.assertEqual(s.panel_note.cget("text"), "the host is back")
        self.app._handle("error", s.event_id, "it broke")
        self.assertEqual(s.panel_note.cget("text"), "it broke")
        # Buttons and shortcuts about a conversation do nothing here.
        self.app._on_new()
        self.app._on_send()
        self.app._resume_task()

    def test_upload_hands_the_chosen_files_to_the_window(self):
        s = self._milanote()
        real = self.mod.filedialog.askopenfilenames
        self.mod.filedialog.askopenfilenames = lambda **k: ("C:/a.png",)
        try:
            self.app._panel_upload(s)
            self._pump(lambda: "Dropped" in s.panel_note.cget("text"))
        finally:
            self.mod.filedialog.askopenfilenames = real
        self.assertEqual(s.browser.uploads, [["C:/a.png"]])
        self.assertIn("Dropped 1 file", s.panel_note.cget("text"))

    def test_closing_the_tab_takes_the_window_out_first(self):
        s = self._milanote()
        b = s.browser
        self.app._close_tab("milanote")
        self.assertTrue(b.released)
        self._pump(lambda: b.closed)
        self.assertTrue(b.closed)
        self.assertNotIn("milanote", self.app.sessions)


if __name__ == "__main__":
    unittest.main()
