"""Angles and Blend: the Kontext graphs, routing and one run (no ComfyUI)."""
import os
import random
import struct
import tempfile
import unittest
import zlib

import apps.image_studio.blend as sb
import apps.image_studio.imagegen as ig
from apps.comfyui.mcp import ComfyError
from tests.test_imagegen import FakeClient as StudioClient, TempStudioMixin, settle


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
        g = sb.angle_graph("a.png", "left side", 7)
        self.assertEqual(classes(g).count("LoadImage"), 1)
        self.assertEqual(g["r1_load"]["inputs"]["image"], "a.png")
        self.assertEqual(g["ks"]["inputs"]["latent_image"], ["r1_enc", 0])
        self.assertIn("full side profile", g["text"]["inputs"]["text"])
        self.assertIn("facing the left edge of the picture", g["text"]["inputs"]["text"])
        self.assertIn("exact same face", g["text"]["inputs"]["text"])
        self.assertEqual(g["ks"]["inputs"]["seed"], 7)
        self.assertNotIn("flip_in", g)                   # a left view as it is

    def test_a_right_view_is_its_left_twin_in_a_mirror(self):
        g = sb.angle_graph("a.png", "right side", 7)
        self.assertEqual(g["text"]["inputs"]["text"],
                         sb.angle_graph("a.png", "left side", 7)["text"]["inputs"]["text"])
        self.assertEqual(g["flip_in"]["inputs"], {"image": ["r1_load", 0],
                                                  "flip_method": sb.FLIP})
        self.assertEqual(g["r1_scale"]["inputs"]["image"], ["flip_in", 0])
        self.assertEqual(g["flip_out"]["inputs"]["image"], ["dec", 0])
        self.assertEqual(g["save"]["inputs"]["images"], ["flip_out", 0])
        self.assertEqual([sb.mirrored(k) for k in ((1, 1, -1), (0, 0, 1), (-1, 0, 0))],
                         [True, False, False])

    def test_a_turn_from_above_or_below_is_turned_first_then_raised(self):
        self.assertEqual(len(sb.view_steps((0, 0, 1))), 1)       # level
        self.assertEqual(len(sb.view_steps((0, 1, 1))), 1)       # front from above
        self.assertEqual(len(sb.view_steps((0, -1, 0))), 1)      # straight below
        turn = sb.view_steps((-1, -1, 1))
        self.assertEqual(turn[0], sb.view_prompt((-1, 0, 1)))
        self.assertIn("worm's-eye view", turn[1])
        self.assertIn(sb.KEEP_TURN, turn[1])
        g = sb.angle_graph("a.png", "right side from above", 7)  # mirrored and raised
        self.assertIn("bird's-eye view", g["text2"]["inputs"]["text"])
        self.assertEqual(g["s2_scale"]["inputs"]["image"], ["flip_out", 0])
        self.assertEqual(g["ks2"]["inputs"]["latent_image"], ["s2_enc", 0])
        self.assertEqual(g["ks2"]["inputs"]["seed"], 8)
        self.assertEqual(g["save"]["inputs"]["images"], ["dec2", 0])
        self.assertEqual(sb.angle_graph("a.png", "left side from below", 7)[
            "s2_scale"]["inputs"]["image"], ["dec", 0])
        for name in sb.VIEW_NAMES:                               # every link resolves
            g = sb.angle_graph("a.png", name, sb.ig.MAX_SEED)
            self.assertLessEqual(g.get("ks2", g["ks"])["inputs"]["seed"], sb.ig.MAX_SEED)
            for node in g.values():
                for v in node["inputs"].values():
                    if isinstance(v, list):
                        self.assertIn(v[0], g, name)

    def test_blend_chains_both_photos_onto_an_empty_latent(self):
        g = sb.blend_graph("a.png", "b.png", (832, 1216), 3)
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
                  sb.blend_graph("a.png", "b.png", (1024, 1024), 1)):
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

    def test_the_26_views_of_a_cube_have_names_of_their_own(self):
        self.assertEqual(len(sb.VIEW_KEYS), 26)
        self.assertEqual(len(set(sb.VIEW_NAMES)), 26)
        self.assertEqual(sb.view_name((0, 0, 1)), "front")
        self.assertEqual(sb.view_name((1, 0, 1)), "front right")
        self.assertEqual(sb.view_name((-1, 0, 0)), "left side")
        self.assertEqual(sb.view_name((1, 1, -1)), "back right from above")
        self.assertEqual(sb.view_name((0, -1, 0)), "straight below")
        self.assertTrue(set(sb.DEFAULT_VIEWS) <= set(sb.VIEW_NAMES))

    def test_a_view_moves_the_camera_and_always_asks_for_the_left(self):
        for key in sb.VIEW_KEYS:
            self.assertNotIn("right", sb.view_prompt(key), key)
        self.assertIn("three-quarter view", sb.view_prompt((-1, 0, 1)))
        self.assertIn("over their shoulder", sb.view_prompt((-1, 0, -1)))
        above, below = sb.view_prompt((0, 1, 1)), sb.view_prompt((0, -1, 1))
        self.assertIn("bird's-eye view", above)
        self.assertIn("facing the camera straight on", above)
        self.assertIn("worm's-eye view", below)
        self.assertIn("straight down", sb.view_prompt((0, 1, 0)))
        self.assertIn("directly beneath", sb.view_prompt((0, -1, 0)))

    def test_the_preset_is_kept_and_read_back(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(sb.load_views(d), sb.DEFAULT_VIEWS)
            sb.save_views(["back left", "no such view", "front"], d)
            self.assertEqual(sb.load_views(d), ["back left", "front"])
            sb.save_views([], d)
            self.assertEqual(sb.load_views(d), [])
            with open(os.path.join(d, sb.PRESET_FILE), "w") as f:
                f.write("{broken")
            self.assertEqual(sb.load_views(d), sb.DEFAULT_VIEWS)

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


class KontextClient(StudioClient):
    """A ComfyUI with FLUX Kontext."""

    def inventory(self):
        inv = super().inventory()
        for kind, names in sb.FILES.items():
            inv[kind] = inv.get(kind, set()) | set(names)
        return inv


class TestWords(unittest.TestCase):
    def test_a_person_is_kept_and_two_pictures_of_anything_keep_nobody(self):
        self.assertEqual(sb.blend_words(), sb.BLEND + sb.KEEP)
        plain = sb.blend_words(False, "at dusk.")
        self.assertNotIn("person", plain)
        self.assertTrue(plain.endswith(" at dusk."), plain)
        g = sb.blend_graph("a.png", "b.png", (1024, 1024), 1, words=plain)
        self.assertEqual(g["text"]["inputs"]["text"], plain)
        self.assertEqual(sb.blend_graph("a.png", "b.png", (1024, 1024), 1)[
            "text"]["inputs"]["text"], sb.BLEND + sb.KEEP)

    def test_a_blend_is_read_whatever_was_kept(self):
        self.assertEqual(sb.clean_blend(None), {"images": [], "person": False, "words": ""})
        self.assertEqual(sb.clean_blend({"images": ["a", "", None, "b"], "person": 1,
                                         "words": "  in \n snow "}),
                         {"images": ["a", "b"], "person": True, "words": "in snow"})


class TestBlendJob(TempStudioMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.studio = ig.Studio(root=self.dir, notify=self.notified.append,
                                client_factory=KontextClient)

    def pic(self, name, size=(400, 600)):
        path = os.path.join(self.dir, name + ".png")
        png(path, *size)
        return path

    def blend(self, **over):
        return dict({"mode": "blend", "seed": 11, "backend": "auto",
                     "blend": {"images": [self.pic("a"), self.pic("b")],
                               "person": False, "words": "in snow"}}, **over)

    def test_a_blend_runs_on_the_queue_and_lands_in_history(self):
        jobs = self.studio.submit(self.blend())
        settle(jobs)
        job = jobs[0]
        self.assertEqual(job.status, "complete", job.detail)
        self.assertEqual(job.backend["id"], "5090")
        client = next(c for c in KontextClient.instances if c.graphs)
        self.assertEqual([os.path.basename(p) for p in client.uploads], ["a.png", "b.png"])
        g = client.graphs[0]
        self.assertEqual(g["r1_load"]["inputs"]["image"], "studio_a.png")
        self.assertEqual(g["r2_load"]["inputs"]["image"], "studio_b.png")
        self.assertEqual(g["ks"]["inputs"]["seed"], 11)
        self.assertEqual(g["text"]["inputs"]["text"], sb.blend_words(False, "in snow"))
        w, h = g["empty"]["inputs"]["width"], g["empty"]["inputs"]["height"]
        self.assertAlmostEqual(w / h, 400 / 600, places=1)      # the first picture's shape
        rec = self.studio.history.list()[0]
        self.assertEqual(rec["prompt"], "Blend: two pictures. in snow")
        self.assertEqual(rec["settings"]["mode"], "blend")
        self.assertEqual(list(rec["references"].values()), rec["settings"]["blend"]["images"])
        self.assertEqual((rec["width"], rec["height"], rec["seed"]), (w, h, 11))
        self.assertEqual([p["label"] for p in rec["passes"]], ["Blend"])
        self.assertTrue(os.path.isfile(rec["images"][0]))
        self.assertEqual(job.outputs, rec["images"])
        self.assertEqual(ig.summary(rec["settings"]), rec["prompt"])
        self.assertEqual([k for k, _ in ig.pipeline_stages(self.studio.lib, rec["settings"])],
                         ["sampling", "decoding", "complete"])

    def test_generate_again_remakes_it_and_a_new_seed_is_another(self):
        jobs = self.studio.submit(self.blend())
        settle(jobs)
        rec = self.studio.history.list()[0]
        again = ig.again(rec)
        self.assertEqual((again["seed"], again["prefer_backend"]), (11, "5090"))
        same = self.studio.submit(again)
        other = self.studio.submit(ig.again(rec, new_seed=True))
        settle(same + other)
        self.assertEqual([j.status for j in same + other], ["complete", "complete"])
        self.assertEqual(same[0].settings["seed"], 11)
        self.assertEqual(other[0].settings["seed_mode"], "random")
        self.assertEqual(len(self.studio.history.list()), 3)

    def test_what_cannot_be_blended_is_refused_in_words(self):
        a = self.pic("a")
        for images, says in (([a], "Choose two pictures"), ([a, a], "two different"),
                             ([a, os.path.join(self.dir, "gone.png")], "Not on this PC")):
            with self.assertRaises(ComfyError) as e:
                self.studio.submit(self.blend(blend={"images": images}))
            self.assertIn(says, str(e.exception))
        self.assertEqual(self.studio.queue.jobs, [])

    def test_no_backend_with_kontext_says_which_file_is_missing(self):
        self.studio = ig.Studio(root=self.dir, notify=self.notified.append,
                                client_factory=StudioClient)
        with self.assertRaises(ComfyError) as e:
            self.studio.submit(self.blend())
        self.assertIn("lacks " + sb.KONTEXT, str(e.exception))

    def test_a_picture_gone_before_its_turn_fails_the_job(self):
        s = self.blend()
        jobs = self.studio.submit(s)
        settle(jobs)
        os.remove(s["blend"]["images"][1])
        job = ig.Job(jobs[0].settings, jobs[0].backend)
        self.studio.queue.add(job)
        settle([job])
        self.assertEqual(job.status, "failed")
        self.assertIn("Not on this PC", job.detail)

    def test_cancelled_while_it_runs_keeps_nothing(self):
        import threading
        KontextClient.hold = threading.Event()
        StudioClient.hold = KontextClient.hold
        try:
            jobs = self.studio.submit(self.blend())
            self.studio.queue.cancel(jobs[0])
            settle(jobs)
        finally:
            StudioClient.hold = None
            KontextClient.hold = None
        self.assertEqual(jobs[0].status, "cancelled")
        self.assertEqual(self.studio.history.list(), [])


if __name__ == "__main__":
    unittest.main()
