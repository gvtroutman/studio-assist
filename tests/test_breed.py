"""Angles and Breed: the Kontext graphs, routing and one run (no ComfyUI)."""
import os
import random
import struct
import tempfile
import unittest
import zlib

import apps.image_studio.breed as sb
from apps.comfyui.mcp import ComfyError


def png(path, w, h):
    raw = b"".join(b"\x00" + b"\x00\x00\x00" * w for _ in range(h))

    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff))
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def classes(g):
    return sorted(n["class_type"] for n in g.values())


class TestGraphs(unittest.TestCase):
    def test_angle_edits_the_photo_it_was_given(self):
        g = sb.angle_graph("a.png", "profile left", 7)
        self.assertEqual(classes(g).count("LoadImage"), 1)
        self.assertEqual(g["r1_load"]["inputs"]["image"], "a.png")
        self.assertEqual(g["ks"]["inputs"]["latent_image"], ["r1_enc", 0])
        self.assertIn("side profile facing left", g["text"]["inputs"]["text"])
        self.assertIn("exact same face", g["text"]["inputs"]["text"])
        self.assertEqual(g["ks"]["inputs"]["seed"], 7)

    def test_breed_chains_both_parents_onto_an_empty_latent(self):
        g = sb.breed_graph("a.png", "b.png", (832, 1216), 3)
        self.assertEqual(g["r1_load"]["inputs"]["image"], "a.png")
        self.assertEqual(g["r2_load"]["inputs"]["image"], "b.png")
        self.assertEqual(g["r1_ref"]["inputs"]["conditioning"], ["text", 0])
        self.assertEqual(g["r2_ref"]["inputs"]["conditioning"], ["r1_ref", 0])
        self.assertEqual(g["guide"]["inputs"]["conditioning"], ["r2_ref", 0])
        self.assertEqual(g["ks"]["inputs"]["latent_image"], ["empty", 0])
        self.assertEqual((g["empty"]["inputs"]["width"], g["empty"]["inputs"]["height"]),
                         (832, 1216))

    def test_every_link_points_at_a_node(self):
        for g in (sb.angle_graph("a.png", "front", 1),
                  sb.breed_graph("a.png", "b.png", (1024, 1024), 1)):
            for node in g.values():
                for v in node["inputs"].values():
                    if isinstance(v, list):
                        self.assertIn(v[0], g)


class TestHelpers(unittest.TestCase):
    def test_angles_are_different_and_known(self):
        picked = sb.pick_angles(4, random.Random(1))
        self.assertEqual(len(set(picked)), 4)
        self.assertTrue(set(picked) <= set(sb.ANGLE_NAMES))
        self.assertEqual(len(sb.pick_angles(99)), len(sb.ANGLES))

    def test_lacks(self):
        self.assertEqual(sb.lacks(None), [])
        have = {k: set(v) for k, v in sb.FILES.items()}
        self.assertEqual(sb.lacks(have), [])
        have["diffusion_models"] = set()
        self.assertEqual(sb.lacks(have), [sb.KONTEXT])

    def test_size_keeps_the_parents_shape_at_about_a_megapixel(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "p.png")
            png(p, 400, 600)
            w, h = sb.size_for(p)
        self.assertEqual((w % 16, h % 16), (0, 0))
        self.assertAlmostEqual(w / h, 400 / 600, places=1)
        self.assertTrue(0.9e6 < w * h <= 1e6)


class FakeStudio:
    def __init__(self, backends, health, inventories):
        self._b, self.health, self.inventories = backends, health, inventories
        self.checked = []

    def backends(self):
        return self._b

    def check(self, b):
        self.checked.append(b["id"])


class TestRoute(unittest.TestCase):
    def setUp(self):
        self.full = {k: set(v) for k, v in sb.FILES.items()}
        self.b5090 = {"id": "5090", "name": "5090", "enabled": True, "roles": ["primary"]}
        self.b3090 = {"id": "3090", "name": "3090", "enabled": True, "roles": ["secondary"]}

    def test_primary_first(self):
        s = FakeStudio([self.b3090, self.b5090], {"5090": {"ok": True}, "3090": {"ok": True}},
                       {"5090": self.full, "3090": self.full})
        self.assertEqual(sb.route(s)[0]["id"], "5090")

    def test_skips_one_without_kontext_and_says_why(self):
        s = FakeStudio([self.b5090, self.b3090], {"5090": {"ok": True}, "3090": {"ok": True}},
                       {"5090": {}, "3090": self.full})
        self.assertEqual(sb.route(s)[0]["id"], "3090")
        s.inventories["3090"] = {}
        b, why = sb.route(s)
        self.assertIsNone(b)
        self.assertIn("lacks " + sb.KONTEXT, why)

    def test_offline_is_checked_once_then_named(self):
        s = FakeStudio([self.b5090], {}, {})
        b, why = sb.route(s)
        self.assertIsNone(b)
        self.assertEqual(s.checked, ["5090"])
        self.assertIn("5090 is offline", why)


class FakeClient:
    def __init__(self, entry):
        self.entry, self.cancelled = entry, []

    def queue_workflow(self, graph):
        return "p1"

    def watch(self):
        return self

    def close(self):
        pass

    def listen_for_progress(self, pid, on_event, stop=None, watch=None):
        on_event("progress", (3, 24, "ks"))
        return None if stop and stop() else self.entry

    def cancel_job(self, pid):
        self.cancelled.append(pid)

    def fetch(self, f):
        return b"PNG:" + f["filename"].encode()


class TestRun(unittest.TestCase):
    def test_returns_the_picture_and_reports_steps(self):
        steps = []
        c = FakeClient({"outputs": {"save": {"images": [
            {"filename": "x.png", "subfolder": "identity", "type": "output"}]}}})
        self.assertEqual(sb.run(c, {}, on_progress=lambda v, t: steps.append((v, t))),
                         b"PNG:x.png")
        self.assertEqual(steps, [(3, 24)])

    def test_stopped_cancels_its_prompt(self):
        c = FakeClient({})
        self.assertIsNone(sb.run(c, {}, stop=lambda: True))
        self.assertEqual(c.cancelled, ["p1"])

    def test_nothing_made_raises(self):
        with self.assertRaises(ComfyError):
            sb.run(FakeClient({"outputs": {}, "status": {}}), {})


if __name__ == "__main__":
    unittest.main()
