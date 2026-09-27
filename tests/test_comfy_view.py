"""The Nodes view: a picture's graphs, and loading one into ComfyUI's page."""

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

    def __init__(self, counts, not_ready=0, error=None):
        self.counts, self.not_ready, self.error = list(counts), not_ready, error
        self.calls, self.closed = [], False

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
        self.assertEqual(dt.calls[0], ("Page.navigate", {"url": "http://100.127.17.38:8188"}))
        self.assertEqual(b.url, "http://100.127.17.38:8188")
        dt2 = FakeDevTools([2])
        b.page = lambda: dt2
        b.show(G1, "t", "http://100.127.17.38:8188/")
        self.assertNotIn("Page.navigate", [m for m, _p in dt2.calls])

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


if __name__ == "__main__":
    unittest.main()
