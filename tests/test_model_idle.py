"""The model chip's pins and the idle unload (studio_chat), without a window."""
import os
import queue
import sys
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import studio_agent as eng
import studio_chat as chat


class Tab:
    def __init__(self, model, used_ago, busy=False):
        self.llm = type("L", (), {"model": model})()
        self.window = 65536
        self.used_at = time.time() - used_ago
        self.busy, self.booting = busy, False
        self.event_id = model + "-tab"


class Vision:
    model, needs_load = "vl", False


class Stub:
    host = "h"

    def __init__(self, tabs, vision=None):
        self.sessions = {i: t for i, t in enumerate(tabs)}
        self.vision = vision
        self.fit_lock = threading.Lock()
        self.q = queue.Queue()
        self.row = None

    def _host_healthy(self, role, last):
        self.row = last


class TestIdleUnload(unittest.TestCase):
    def setUp(self):
        self.unloaded = []
        real = (eng.loaded_instances, eng.unload_model)
        self.addCleanup(lambda: setattr(eng, "loaded_instances", real[0]) or
                        setattr(eng, "unload_model", real[1]))
        eng.loaded_instances = lambda host, m, timeout=5: [(m, 65536)]
        eng.unload_model = lambda host, i, timeout=60: self.unloaded.append(i)

    def run_idle(self, stub):
        chat.Chat._unload_idle(stub)

    def test_only_the_idle_model_goes(self):
        idle = chat.MODEL_IDLE_S + 60
        stub = Stub([Tab("coder", idle), Tab("small", 5)], Vision())
        self.run_idle(stub)
        self.assertEqual(self.unloaded, ["coder"])
        self.assertFalse(stub.vision.needs_load, "a tab is still in use: the vision model stays")
        self.assertIn("coder", stub.q.get_nowait()[2])

    def test_a_model_a_busy_tab_shares_stays(self):
        idle = chat.MODEL_IDLE_S + 60
        self.run_idle(Stub([Tab("coder", idle), Tab("coder", idle, busy=True)]))
        self.assertEqual(self.unloaded, [])

    def test_all_idle_takes_the_vision_model_too_and_marks_it_for_reload(self):
        idle = chat.MODEL_IDLE_S + 60
        stub = Stub([Tab("coder", idle)], Vision())
        self.run_idle(stub)
        self.assertEqual(sorted(self.unloaded), ["coder", "vl"])
        self.assertTrue(stub.vision.needs_load)


class TestPins(unittest.TestCase):
    def test_a_hand_edited_pin_file_is_cleaned(self):
        import json, tempfile
        path = os.path.join(tempfile.mkdtemp(), "s.json")
        with open(path, "w") as f:
            json.dump({"model_pins": {"opencode": "c", "x": 3}}, f)
        self.assertEqual(chat.Prefs(path).get("model_pins"), {"opencode": "c"})
        with open(path, "w") as f:
            json.dump({"model_pins": "nope"}, f)
        self.assertEqual(chat.Prefs(path).get("model_pins"), {})

    def test_a_pin_is_what_model_for_picks(self):
        oc = eng.APPS_BY_ID["opencode"]
        key = "STUDIO_MODEL_OPENCODE"
        old = os.environ.pop(key, None)
        try:
            os.environ[key] = "gpt-oss-20b"
            self.assertEqual(oc.model_for(["qwen3-coder-30b-a3b-instruct", "gpt-oss-20b"],
                                          "qwen3-coder-30b-a3b-instruct")[0], "gpt-oss-20b")
        finally:
            os.environ.pop(key, None)
            if old is not None:
                os.environ[key] = old


if __name__ == "__main__":
    unittest.main()
