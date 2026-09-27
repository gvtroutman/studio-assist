"""The Nodes view: a picture's graphs, and loading one into ComfyUI's page."""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import studio_comfy_view as cv
import studio_milanote as milanote

G1 = {"1": {"class_type": "UNETLoader", "inputs": {}},
      "9": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}}}
G2 = {"fc": {"class_type": "KSampler", "inputs": {}}}
G3 = {"h": {"class_type": "VAEDecode", "inputs": {}}}


class FakeDevTools:
    """Answers Runtime.evaluate as ComfyUI's page would: not ready for
    `not_ready` calls, then each load returns the next of `counts`."""

    def __init__(self, counts, not_ready=0, error=None, dialog=True):
        self.counts, self.not_ready, self.error = list(counts), not_ready, error
        self.calls, self.closed = [], False
        self.seq, self.ws = 0, FakeSocket(self, dialog)

    def call(self, method, params=None):
        self.calls.append((method, params or {}))
        if method != "Runtime.evaluate":
            return {}
        expr = params["expression"]
        if expr == cv.READY:
            self.not_ready -= 1
            return {"result": {"value": self.not_ready < 0}}
        if self.error:
            return {"exceptionDetails": {"exception": {"description": self.error}}}
        return {"result": {"value": self.counts.pop(0)}}

    def close(self):
        self.closed = True


class FakeSocket:
    """What `leave` talks to: a navigation or reload first raises the page's
    "Leave app?" (when `dialog`), and answers only once it is accepted."""

    def __init__(self, dt, dialog):
        self.dt, self.dialog, self.waiting, self.inbox = dt, dialog, None, []
        self.accepted = []

    def send(self, text):
        msg = json.loads(text)
        self.dt.calls.append((msg["method"], msg["params"]))
        if msg["method"] == "Page.handleJavaScriptDialog":
            self.accepted.append(msg["params"]["accept"])
            self.inbox += [{"id": msg["id"]}, {"id": self.waiting}]
        elif self.dialog:
            self.waiting = msg["id"]
            self.inbox.append({"method": "Page.javascriptDialogOpening",
                               "params": {"type": "beforeunload"}})
        else:
            self.inbox.append({"id": msg["id"]})

    def recv(self):
        return json.dumps(self.inbox.pop(0))


def browser(dt, url="http://127.0.0.1:8188"):
    b = cv.ComfyBrowser(url, exe="chrome.exe", profile="p")
    b.page = lambda: dt
    return b


class StepsTest(unittest.TestCase):
    def test_the_steps_are_the_graphs_in_the_order_they_ran(self):
        rec = {"graph": G1, "face_graph": G2, "paste_graph": None,
               "passes": [{"label": "Hands", "graph": G3}]}
        self.assertEqual(cv.graph_steps(rec), [("Picture", G1), ("Face pass", G2),
                                               ("Hands", G3)])

    def test_a_try_on_lists_its_garments_and_not_its_last_twice(self):
        rec = {"graph": G2, "dress": {"graphs": [G1, G2]}}
        self.assertEqual([label for label, _g in cv.graph_steps(rec)],
                         ["Try on 1", "Try on 2"])

    def test_a_record_without_graphs_has_no_steps(self):
        self.assertEqual(cv.graph_steps({"graph": None, "passes": [{"label": "x"}]}), [])
        self.assertEqual(cv.graph_steps({}), [])

    def test_an_origin_ignores_the_path_and_the_case(self):
        self.assertEqual(cv.origin("http://127.0.0.1:8188/"), cv.origin("HTTP://127.0.0.1:8188"))
        self.assertNotEqual(cv.origin("http://127.0.0.1:8188"),
                            cv.origin("http://100.127.17.38:8188"))


class ShowTest(unittest.TestCase):
    def test_a_graph_is_loaded_once_the_page_is_ready_and_named(self):
        dt = FakeDevTools([2], not_ready=2)
        self.assertEqual(browser(dt).show(G1, "abc · Picture"), 2)
        loads = [p["expression"] for m, p in dt.calls
                 if m == "Runtime.evaluate" and p["expression"] != cv.READY]
        self.assertEqual(len(loads), 1)
        self.assertIn('"abc \\u00b7 Picture"', loads[0])
        self.assertIn('"UNETLoader"', loads[0])
        self.assertTrue(dt.closed)

    def test_a_load_drawn_over_by_the_old_session_is_made_again(self):
        dt = FakeDevTools([39, 2])
        self.assertEqual(browser(dt).show(G1, "t"), 2)

    def test_another_backend_is_navigated_to_first(self):
        dt = FakeDevTools([2])
        b = browser(dt)
        b.show(G1, "t", "http://100.127.17.38:8188")
        self.assertEqual(dt.calls[1], ("Page.navigate", {"url": "http://100.127.17.38:8188"}))
        self.assertEqual(b.url, "http://100.127.17.38:8188")
        # The graphs there were unsaved workflows: the page asked, and was told Leave.
        self.assertEqual(dt.ws.accepted, [True])
        dt2 = FakeDevTools([2])
        b.page = lambda: dt2
        b.show(G1, "t", "http://100.127.17.38:8188/")
        self.assertNotIn("Page.navigate", [m for m, _p in dt2.calls])

    def test_a_reload_answers_leave_app_and_a_quiet_page_is_not_answered(self):
        dt = FakeDevTools([])
        browser(dt).reload()
        self.assertEqual([m for m, _p in dt.calls],
                         ["Page.enable", "Page.reload", "Page.handleJavaScriptDialog"])
        dt = FakeDevTools([], dialog=False)
        browser(dt).goto("http://100.127.17.38:8188")
        self.assertEqual(dt.ws.accepted, [])
        self.assertTrue(dt.closed)

    def test_a_page_error_is_a_sentence(self):
        dt = FakeDevTools([], error="TypeError: bad graph\n    at loadApiJson")
        with self.assertRaises(RuntimeError) as caught:
            browser(dt).show(G1, "t")
        self.assertEqual(str(caught.exception), "ComfyUI would not open the graph: "
                                                "TypeError: bad graph")
        self.assertTrue(dt.closed)

    def test_a_page_that_never_gets_ready_says_so(self):
        dt = FakeDevTools([], not_ready=10 ** 6)
        with self.assertRaises(RuntimeError) as caught:
            browser(dt)._wait_ready(dt, wait=0.2)
        self.assertIn("Is it running?", str(caught.exception))

    def test_the_page_is_the_one_on_this_comfyui(self):
        real = milanote.pages
        milanote.pages = lambda port: [{"url": "https://app.milanote.com/"},
                                       {"url": "http://127.0.0.1:8188/"}]
        try:
            self.assertEqual(browser(None).target(1)["url"], "http://127.0.0.1:8188/")
        finally:
            milanote.pages = real

    def test_its_window_has_a_profile_of_its_own(self):
        self.assertNotEqual(cv.profile_dir(), milanote.profile_dir())
        self.assertEqual(cv.ComfyBrowser.name, "ComfyUI")


def _headless():
    try:
        import tkinter
        tkinter.Tk().destroy()
        return False
    except Exception:
        return True


class StubComfy:
    """ComfyBrowser without Chrome: what was shown, where, and its end."""
    made = []

    def __init__(self, url):
        self.url, self.shown, self.went = url, [], []
        self.released = self.closed = False
        self.size, self.inset = (1, 1), (0, 0, 0, 0)
        StubComfy.made.append(self)

    def start(self):
        return 1

    def running(self):
        return not self.closed

    def show(self, graph, title, url=None):
        self.shown.append((title, url))
        return len(graph)

    def goto(self, url):
        self.went.append(url)

    def reload(self):
        pass

    def embed(self, parent, w, h):
        self.parent = parent

    def fit(self, w, h):
        pass

    def focus(self):
        pass

    def measure(self):
        return self.inset

    def release(self):
        self.released = True

    def close(self, grace=0):
        self.closed = True


@unittest.skipIf(_headless(), "no display")
class NodesTabTest(unittest.TestCase):
    """The ComfyUI tab's Chat | Nodes switch, and the Image Studio's Nodes
    landing there (studio_nodes_ui)."""

    @classmethod
    def setUpClass(cls):
        import tempfile
        import studio_chat
        import studio_nodes_ui
        cls.mod, cls.nodes_ui = studio_chat, studio_nodes_ui
        cls.dir = tempfile.mkdtemp()
        cls._real_settings = os.environ.get("STUDIO_SETTINGS")
        os.environ["STUDIO_SETTINGS"] = os.path.join(cls.dir, "settings.json")
        with open(os.environ["STUDIO_SETTINGS"], "w") as f:
            json.dump({"tabs": ["chat"]}, f)
        cls._saved = {n: getattr(studio_chat.Chat, n) for n in
                      ("_boot_host", "_read_icons", "_boot_session", "_ensure")}
        studio_chat.Chat._boot_host = lambda self, *a, **k: None
        studio_chat.Chat._read_icons = lambda self: None
        studio_chat.Chat._boot_session = lambda self, s: None
        studio_chat.Chat._ensure = lambda self, s: None
        cls._real_browser = studio_nodes_ui.comfy_view.ComfyBrowser
        studio_nodes_ui.comfy_view.ComfyBrowser = StubComfy
        cls._real_backends = studio_nodes_ui.NodesView._backends
        studio_nodes_ui.NodesView._backends = lambda self: [
            ("5090 Workstation", "http://127.0.0.1:8188"),
            ("3090 Server", "http://100.127.17.38:8188")]
        cls.app = studio_chat.Chat()
        for _ in range(10):
            cls.app.update()

    @classmethod
    def tearDownClass(cls):
        cls.app._quit()
        for n, fn in cls._saved.items():
            setattr(cls.mod.Chat, n, fn)
        cls.nodes_ui.comfy_view.ComfyBrowser = cls._real_browser
        cls.nodes_ui.NodesView._backends = cls._real_backends
        if cls._real_settings is None:
            os.environ.pop("STUDIO_SETTINGS", None)
        else:
            os.environ["STUDIO_SETTINGS"] = cls._real_settings

    def _pump(self, until=lambda: True):
        import time
        deadline = time.monotonic() + 2.0
        while True:
            self.app._drain()
            self.app.update()
            if until() or time.monotonic() > deadline:
                return
            time.sleep(0.01)

    def test_a_picture_opens_in_the_comfyui_tab_and_chat_comes_back(self):
        steps = [("Picture", G1), ("Face pass", G2)]
        self.app.open_nodes(steps, "http://127.0.0.1:8188", "rec1")
        s = self.app.sessions["comfyui"]
        v = s.nodes_view
        self._pump(lambda: s.browser is not None and s.browser.shown)
        self.assertEqual(self.app.active, "comfyui")
        self.assertTrue(v.on)
        self.assertEqual(s.browser.shown, [("rec1 · Picture", "http://127.0.0.1:8188")])
        self._pump(lambda: "2 nodes" in v.note.cget("text"))
        self.assertIn("rec1 · Picture: 2 nodes", v.note.cget("text"))
        # A window, not a conversation: no composer, no New chat.
        self.assertTrue(v.frame.winfo_ismapped())
        self.assertFalse(s.view.winfo_ismapped())
        self.assertFalse(self.app.composer.winfo_ismapped())
        self.assertEqual(self.app.btn_new.cget("state"), "disabled")
        # Another step, then the other ComfyUI by hand: the steps go.
        v.load_step(1)
        self._pump(lambda: len(s.browser.shown) == 2)
        self.assertEqual(s.browser.shown[-1][0], "rec1 · Face pass")
        v._switch("http://100.127.17.38:8188")
        self._pump(lambda: s.browser.went)
        self.assertEqual(s.browser.went, ["http://100.127.17.38:8188"])
        self.assertEqual(v.steps, [])
        self.assertFalse(v.b_step.winfo_ismapped())
        # Chat: the transcript and the composer back, the window kept.
        v.show_chat()
        self._pump()
        self.assertFalse(v.on)
        self.assertTrue(s.view.winfo_ismapped())
        self.assertFalse(v.frame.winfo_ismapped())
        self.assertTrue(self.app.composer.winfo_ismapped())
        self.assertEqual(self.app.btn_new.cget("state"), "normal")
        self.assertEqual(len(StubComfy.made), 1)
        # Closing the tab takes the window out of its frame, then ends it.
        b = s.browser
        self.app._close_tab("comfyui")
        self.assertTrue(b.released)
        self._pump(lambda: b.closed)
        self.assertTrue(b.closed)

    def test_only_the_comfyui_tab_has_a_nodes_view(self):
        self.app._add_tab("chat")
        self.assertIsNone(self.app.sessions["chat"].nodes_view)


if __name__ == "__main__":
    unittest.main()
