"""The item pass: the form's Wearing pictures put on after the picture is
drawn, by FLUX.2 Klein; no apps, models or network."""
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import apps.image_studio.imagegen as ig
import apps.image_studio.wear as wear
from test_headswap import KleinClient, KLEIN
from test_imagegen import TempStudioMixin, PNG, settle, FakeClient

WORN_PNG = PNG + b"worn"              # what the item pass's run hands back


class TestWhereItGoes(unittest.TestCase):
    def test_a_name_says_where_it_is_worn_and_what_sam3_looks_for(self):
        self.assertEqual(wear.place_of("green flannel shirt"), ("body", "shirt"))
        self.assertEqual(wear.place_of("my lucky hat"), ("head", "hat"))
        self.assertEqual(wear.place_of("gold moon necklace"), ("neck", "necklace"))
        self.assertEqual(wear.place_of("white high-top sneakers"), ("feet", "sneakers"))
        self.assertEqual(wear.place_of("silver watch"), ("hand", "watch"))
        self.assertEqual(wear.place_of("round sunglasses"), ("face", "sunglasses"))
        self.assertEqual(wear.place_of("leather tote bag"), ("body", "bag"))
        # A name that says no kind of thing: SAM3 looks for the clothing.
        self.assertEqual(wear.place_of("Nike Air Max"), ("body", "clothing"))

    def test_the_body_goes_on_first_and_the_small_things_last(self):
        items = [{"name": n, "path": n} for n in ("bucket hat", "moon necklace",
                                                  "white sneakers", "flannel shirt")]
        self.assertEqual([i["name"] for i in wear.ordered(items)],
                         ["flannel shirt", "white sneakers", "moon necklace", "bucket hat"])

    def test_sam3_is_asked_once_for_every_item_and_what_they_are_fitted_to(self):
        items = [{"name": "flannel shirt", "path": "a"}, {"name": "white sneakers", "path": "b"},
                 {"name": "red shirt", "path": "c"}]
        self.assertEqual(wear.find_words(items), ["shirt", "sneakers", "person", "face", "foot"])

    def test_an_item_goes_on_the_main_person_not_a_passer_by(self):
        # The café bench (2026-10-01): SAM3's only "hat" was on a man behind her.
        boxes = [(1003, 328, 25, 35, "hat"), (150, 37, 858, 986, "person"),
                 (974, 329, 55, 157, "person"), (573, 127, 228, 283, "face")]
        box = wear.where(1024, 1024, {"name": "bucket hat"}, boxes)
        fx, fy, fw, fh = 573, 127, 228, 283
        self.assertEqual(box, (fx - fw * 0.5, fy - fh * 0.9, fw * 2, fh * 1.5))
        # A hat she wears is redrawn with her whole head under it.
        boxes.append((600, 60, 180, 120, "hat"))
        x, y, w, h = wear.where(1024, 1024, {"name": "bucket hat"}, boxes)
        self.assertLessEqual(x, fx - fw * 0.5)
        self.assertGreaterEqual(y + h, fy + fh * 0.6 - 1)

    def test_a_pair_is_one_crop_round_both(self):
        boxes = [(481, 1101, 83, 79, "sneakers"), (288, 1095, 86, 85, "sneakers"),
                 (269, 142, 312, 1042, "person"), (366, 168, 110, 123, "face")]
        self.assertEqual(wear.where(832, 1216, {"name": "green sneakers"}, boxes),
                         (288, 1095, 276, 85))

    def test_with_nothing_of_it_drawn_it_goes_where_it_is_worn(self):
        boxes = [(100, 50, 400, 900, "person"), (250, 80, 100, 120, "face")]
        x, y, w, h = wear.where(600, 1000, {"name": "moon necklace"}, boxes)
        self.assertGreater(y, 80 + 120 * 0.5)             # below the chin
        self.assertLess(y, 80 + 120 * 1.5)
        x, y, w, h = wear.where(600, 1000, {"name": "flannel shirt"}, boxes)
        self.assertGreater(y, 80)                         # below the face, not over it
        self.assertEqual((x, w), (100, 400))
        self.assertIsNone(wear.where(600, 1000, {"name": "hat"}, []))   # no one to wear it

    def test_the_crop_reaches_past_the_item_and_stays_inside_the_picture(self):
        crop = wear.crop_for(1024, 1024, (900, 900, 100, 100))
        self.assertEqual(crop["x"] + crop["width"], 1024)
        self.assertEqual(crop["y"] + crop["height"], 1024)
        self.assertGreaterEqual(crop["width"], wear.MIN_SIDE)
        tall = wear.crop_for(832, 1216, (300, 200, 200, 900))
        self.assertLessEqual(tall["height"], 1216)
        self.assertGreater(tall["height"], tall["width"])  # not a square: Klein draws its shape
        w, h = wear.drawn_size(tall)
        self.assertEqual((w % 16, h % 16), (0, 0))
        self.assertAlmostEqual(w * h / float(wear.AREA), 1.0, delta=0.06)

    def test_the_graph_draws_each_item_from_its_picture_and_blends_it_alone(self):
        items = [{"name": "flannel shirt", "path": "s.png", "noun": "shirt", "place": "body",
                  "picture": "studio_s.png", "crop": {"x": 100, "y": 200, "width": 400,
                                                      "height": 500}},
                 {"name": "moon necklace", "path": "n.png", "noun": "necklace", "place": "neck",
                  "picture": "studio_n.png", "crop": {"x": 0, "y": 100, "width": 200,
                                                      "height": 200}}]
        g = wear.item_graph("pic.png", items, 7, "x_wear", "sam3.pt", (832, 1216))
        self.assertEqual(g["i1_photo"]["inputs"]["image"], "studio_s.png")
        self.assertIn("flannel shirt", g["i1_text"]["inputs"]["text"])
        self.assertEqual(g["i1_word"]["inputs"]["text"], "shirt:%d" % wear.FIND)
        self.assertEqual(g["i2_word"]["inputs"]["text"], "necklace:%d" % wear.FIND)
        # The second is drawn on the picture the first left, seeded after it.
        self.assertEqual(g["i2_cut"]["inputs"]["image"], ["i1_put", 0])
        self.assertEqual(g["i2_noise"]["inputs"]["noise_seed"], 8)
        self.assertEqual(g["save"]["inputs"]["images"], ["i2_put", 0])
        # Klein sees the crop and the item's picture; it draws at the crop's shape.
        self.assertEqual(g["i1_pos2"]["inputs"]["latent"], ["i1_lat2", 0])
        self.assertEqual(g["i1_lat2"]["inputs"]["pixels"], ["i1_fit", 0])
        self.assertEqual((g["i1_empty"]["inputs"]["width"], g["i1_empty"]["inputs"]["height"]),
                         wear.drawn_size(items[0]["crop"]))
        # Blended through the item before and after, back where it was cut.
        self.assertEqual(g["i1_ma"]["inputs"]["image"], ["i1_cut", 0])
        self.assertEqual(g["i1_mb"]["inputs"]["image"], ["i1_small", 0])
        self.assertEqual((g["i1_put"]["inputs"]["x"], g["i1_put"]["inputs"]["y"]), (100, 200))
        # No colour moved toward the crop's: it would move the shirt's to the old one's.
        self.assertFalse(any(n["class_type"] == "ColorTransfer" for n in g.values()))
        # A thin thing reaches further than a shirt.
        self.assertGreater(g["i2_grow"]["inputs"]["expand"] / 200.0,
                           g["i1_grow"]["inputs"]["expand"] / 400.0)
        # No margin where the crop ends at the picture's edge (the necklace's left).
        self.assertEqual(g["i2_q2"]["inputs"]["x"], 0)
        self.assertGreater(g["i1_q2"]["inputs"]["x"], 0)
        self.assertLessEqual({n["class_type"] for n in g.values()} - {"SaveImage"},
                             wear.NODES | {"UNETLoader", "CLIPLoader", "VAELoader",
                                           "LoadImage", "CheckpointLoaderSimple",
                                           "CLIPTextEncode", "ConditioningZeroOut",
                                           "VAEEncode", "VAEDecode"})


class TestWearingSettings(unittest.TestCase):
    def test_the_list_is_cleaned_one_per_name(self):
        self.assertEqual(ig.clean_wearing([{"name": " hat ", "path": "a.png"},
                                           {"name": "HAT", "path": "b.png"},
                                           {"name": "", "path": "c.png"},
                                           {"name": "shoe"}, "junk"]),
                         [{"name": "hat", "path": "a.png"}])
        self.assertEqual(ig.clean_wearing(None), [])
        self.assertEqual(ig.default_settings()["wearing"], [])

    def test_what_is_worn_is_said_in_the_prompt_and_dresses_the_person(self):
        s = dict(ig.default_settings(), wearing=[{"name": "red plaid shirt", "path": "a.png"}])
        self.assertIn("wearing red plaid shirt", ig.person_text(s))
        self.assertTrue(ig.is_dressed(s))
        out = ig.outfit_of(dict(s, wearing=s["wearing"] + [{"name": "bucket hat",
                                                             "path": "b.png"}]))
        self.assertEqual(out["clothes"], [{"name": "red plaid shirt", "path": "a.png"}])
        self.assertEqual(out["accessories"], [{"name": "bucket hat", "path": "b.png"}])

    def test_a_character_keeps_what_it_wears(self):
        rec = ig.clean_character({"name": "A", "wearing": [{"name": "hat", "path": "h.png"}]})
        self.assertEqual(rec["wearing"], [{"name": "hat", "path": "h.png"}])

    def test_a_fix_redraws_without_putting_it_on_again(self):
        s = ig.Studio.fix_base(dict(ig.default_settings(),
                                    wearing=[{"name": "hat", "path": "h.png"}]))
        self.assertEqual(s["wearing"], [])


class TestItemPassInGenerate(TempStudioMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        WornClient.fail = False
        self.shirt = os.path.join(self.dir, "shirt.png")
        Path(self.shirt).write_bytes(PNG)

    def generate(self, client=None, **settings):
        self.studio.client_factory = client or WornClient
        self.studio.clients = {}
        s = dict(ig.default_settings(), model="z-image-turbo", backend="5090",
                 scene="A man in a park", critic_notes=False, hand_pass=False,
                 wearing=[{"name": "flannel shirt", "path": self.shirt}], **settings)
        jobs = self.studio.submit(s)
        settle(jobs)
        return jobs[0], FakeClient.instances[-1]

    def test_the_item_is_put_on_from_its_picture_after_the_picture_is_drawn(self):
        job, client = self.generate()
        self.assertEqual(job.status, "complete", job.detail)
        find = next(g for g in client.graphs if "p0d" in g)
        self.assertEqual(find["p0t"]["inputs"]["text"], "shirt:%d" % wear.FIND)
        worn = [g for g in client.graphs if "i1_ks" in g]
        self.assertEqual(len(worn), 1)
        self.assertEqual(worn[0]["i1_photo"]["inputs"]["image"], "studio_shirt.png")
        self.assertIn(self.shirt, client.uploads)
        # The words say it too, so the model draws a shirt there to begin with.
        self.assertIn("wearing flannel shirt", client.graphs[0]["10"]["inputs"]["text"])
        with open(job.record["images"][0], "rb") as f:
            saved = f.read()
        self.assertTrue(saved.endswith(b"worn"))        # the item pass's picture
        self.assertIn(b"tEXtComment\0" + wear.LICENSE_NOTE.encode("latin-1"), saved)
        self.assertIn("Item pass", [p["label"] for p in job.record["passes"]])
        self.assertIn("Item pass: the flannel shirt put on from its picture by FLUX.2 "
                      "Klein 9B.", job.record["notes"])
        self.assertEqual(job.record["license"], wear.LICENSE_NOTE)
        self.assertIn(("items", "Item pass"), ig.pipeline_stages(self.studio.lib, job.settings))

    def test_a_failed_item_pass_keeps_the_picture_as_drawn(self):
        WornClient.fail = True
        job, client = self.generate()
        self.assertEqual(job.status, "complete", job.detail)
        with open(job.record["images"][0], "rb") as f:
            self.assertFalse(f.read().endswith(b"worn"))
        self.assertNotIn("Item pass", [p["label"] for p in job.record["passes"]])
        self.assertTrue(any(n.startswith("The item pass could not run")
                            for n in job.record["notes"]), job.record["notes"])
        self.assertEqual(job.record["license"], "")

    def test_a_backend_without_klein_describes_it_in_words_and_says_so(self):
        plan = ig.compose(dict(ig.default_settings(), model="z-image-turbo", scene="A man",
                               wearing=[{"name": "flannel shirt", "path": self.shirt}]),
                          self.studio.lib, self.backend("5090"),
                          FakeClient(self.backend("5090")).inventory())
        self.assertEqual(plan.wear, [])
        self.assertTrue(any("Pictures of the flannel shirt" in w and wear.KLEIN in w
                            for w in plan.warnings), plan.warnings)
        self.assertIn("wearing flannel shirt", plan.prompt)

    def test_a_missing_picture_is_said_and_skipped(self):
        os.remove(self.shirt)
        job, client = self.generate()
        self.assertEqual(job.status, "complete", job.detail)
        self.assertFalse([g for g in client.graphs if "i1_ks" in g])


class WornClient(KleinClient):
    """A ComfyUI with SAM3 and Klein whose finder sees one man in a shirt."""
    fail = False

    def node_types(self):
        return super().node_types() | wear.NODES

    def listen_for_progress(self, pid, on_event, stop=None, timeout=0):
        graph = self.graphs[int(pid[3:]) - 1]
        if "p0d" in graph and "i1_ks" not in graph:
            found = {"shirt": [(300, 300, 300, 350)], "person": [(280, 140, 320, 900)],
                     "face": [(380, 170, 110, 120)]}
            outputs = {"7": {"text": ["1024"]}, "8": {"text": ["1024"]}}
            for k in (k for k in graph if k.endswith("t") and k.startswith("p")):
                word = graph[k]["inputs"]["text"].split(":")[0]
                outputs[k[:-1] + "v"] = {"text": [json.dumps([[
                    {"x": x, "y": y, "width": w, "height": h}
                    for x, y, w, h in found.get(word, [])]])]}
            return {"status": {"completed": True}, "outputs": outputs}
        if "i1_ks" in graph:
            if WornClient.fail:
                return {"status": {"messages": [["execution_error", {
                    "node_id": "i1_ks", "node_type": "SamplerCustomAdvanced",
                    "exception_type": "RuntimeError", "exception_message": "boom"}]]},
                    "outputs": {}}
            return {"status": {"completed": True}, "outputs": {"save": {"images": [
                {"filename": "x_wear_00001_.png", "subfolder": "ImageStudio",
                 "type": "output"}]}}}
        return super().listen_for_progress(pid, on_event, stop, timeout)

    def fetch(self, f):
        return WORN_PNG if "_wear_" in f["filename"] else super().fetch(f)


if __name__ == "__main__":
    unittest.main()
