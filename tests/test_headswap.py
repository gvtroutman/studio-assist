"""The Klein head swap before the final face swap; no apps, models or network."""
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import apps.image_studio.facefusion as ff
import apps.image_studio.headswap as hs
import apps.image_studio.imagegen as ig
from test_imagegen import TempStudioMixin, PNG, settle, FakeClient, FaceClient

HEAD_PNG = PNG + b"klein"             # what the head swap's run hands back
KLEIN = {kind: set(names) for kind, names in hs.FILES.items()}


class TestHeadSwapParts(unittest.TestCase):
    def test_a_backend_without_the_files_or_nodes_is_told_what_it_lacks(self):
        self.assertEqual(hs.lacks(KLEIN, hs.NODES), [])
        self.assertEqual(hs.lacks(dict(KLEIN, vae=set())), ["flux2-vae.safetensors"])
        self.assertEqual(hs.lacks(KLEIN, hs.NODES - {"Flux2Scheduler"}), ["Flux2Scheduler"])
        # An inventory never read is a backend that cannot, not one that may.
        self.assertEqual(hs.lacks(None), [hs.KLEIN])

    def test_each_head_is_the_face_facefusion_will_swap(self):
        with patch.object(hs.os.path, "isfile", return_value=True):
            small, big, right = (100, 100, 40, 40), (300, 100, 90, 100), (700, 100, 60, 60)
            a, b = {"name": "A", "references": ["a.png"]}, {"name": "B", "references": ["b.png"]}
            # One profile: the single face, or the biggest when SAM3 sees more.
            self.assertEqual(hs.targets(1000, 1000, [right, small, big], [a]), [(a, big)])
            # Two: left to right, as FaceFusion's face_index / face_count.
            self.assertEqual(hs.targets(1000, 1000, [right, big], [a, b]), [(a, big), (b, right)])
            # A scene's region picks the face; a region with no face has no head.
            there = dict(a, target_region=[.6, 0, 1, .5])
            nowhere = dict(b, target_region=[0, .8, 1, 1])
            self.assertEqual(hs.targets(1000, 1000, [small, big, right], [there, nowhere]),
                             [(there, right)])
            # No photo, no head: the face swap's own check says why.
            self.assertEqual(hs.targets(1000, 1000, [big], [{"name": "C", "references": []}]), [])
        self.assertEqual(hs.targets(1000, 1000, [big], [a]), [])       # the photo is gone

    def test_the_crop_has_room_for_the_hair_and_stays_inside_the_picture(self):
        crop = hs.head_crop(1024, 1024, (450, 300, 80, 90))
        self.assertEqual(crop["width"], int(90 * hs.CROP))
        self.assertEqual(crop["width"], crop["height"])
        centre = crop["y"] + crop["height"] / 2.0
        self.assertAlmostEqual(centre, 345 - 90 * hs.RISE, delta=1)     # higher than the face
        top = hs.head_crop(1024, 1024, (10, 5, 80, 90))
        self.assertEqual((top["x"], top["y"]), (0, 0))
        self.assertEqual(hs.head_crop(1024, 1024, (450, 300, 80, 90), crop=2, rise=.5),
                         {"x": 400, "y": 210, "width": 180, "height": 180})
        huge = hs.head_crop(800, 600, (100, 50, 500, 500))
        self.assertEqual((huge["width"], huge["x"] + huge["width"] <= 800), (600, True))

    def test_the_graph_redraws_each_head_from_its_photo_and_blends_the_head_alone(self):
        heads = [{"crop": {"x": 300, "y": 100, "width": 270, "height": 270}, "photo": "a.png"},
                 {"crop": {"x": 600, "y": 120, "width": 240, "height": 240}, "photo": "b.png"}]
        g = hs.head_graph("picture.png", heads, 7, "ImageStudio/x_head", "sam3.pt")
        self.assertEqual(g["unet"]["inputs"]["unet_name"], hs.KLEIN)
        self.assertEqual(g["clip"]["inputs"]["type"], "flux2")
        self.assertEqual(g["sigmas"]["inputs"]["steps"], hs.STEPS)
        self.assertEqual(g["h1_cut"]["inputs"]["crop_region"], heads[0]["crop"])
        self.assertEqual(g["h1_cut"]["inputs"]["image"], ["image", 0])
        # The picture first, the photo second, on the prompt and on its zeroed copy.
        self.assertEqual(g["h1_lat1"]["inputs"]["pixels"], ["h1_big", 0])
        self.assertEqual(g["h1_lat2"]["inputs"]["pixels"], ["h1_fit", 0])
        self.assertEqual(g["h1_photo"]["inputs"]["image"], "a.png")
        self.assertEqual(g["h1_pos2"]["inputs"]["conditioning"], ["h1_pos1", 0])
        self.assertEqual(g["h1_guide"]["inputs"]["positive"], ["h1_pos2", 0])
        self.assertEqual(g["h1_guide"]["inputs"]["negative"], ["h1_neg2", 0])
        self.assertEqual(g["h1_ks"]["inputs"]["latent_image"], ["empty", 0])
        # Drawn at SIDE, put back at the crop's own size and place, through the head.
        self.assertEqual(g["h1_small"]["inputs"]["width"], 270)
        # The head, the hair and a necklace, each before and after, as one mask.
        self.assertEqual([g["word%d" % w]["inputs"]["text"] for w in (0, 1, 2)],
                         ["head", "hair", "necklace"])
        self.assertEqual([(g["h1_m%d%s" % (w, k)]["inputs"]["conditioning"][0],
                           g["h1_m%d%s" % (w, k)]["inputs"]["image"][0])
                          for w in (0, 1, 2) for k in "ab"],
                         [("word0", "h1_cut"), ("word0", "h1_small"),
                          ("word1", "h1_cut"), ("word1", "h1_small"),
                          ("word2", "h1_cut"), ("word2", "h1_small")])
        self.assertEqual(g["h1_m4"]["inputs"]["mask"], ["h1_m2b_or", 0])
        # The old hair - the hair before, not Klein's - reaches further, for
        # its loose strands, and round: a curl to the side as one above.
        strands = g["h1_f0"]["inputs"]
        self.assertEqual((strands["mask"], strands["expand"], strands["tapered_corners"]),
                         (["h1_m1a", 0], hs.STRANDS * 270 // hs.SIDE, False))
        self.assertGreater(strands["expand"], g["h1_m4"]["inputs"]["expand"])
        self.assertEqual((g["h1_f0_or"]["inputs"]["destination"],
                          g["h1_f0_or"]["inputs"]["source"],
                          g["h1_f0_or"]["inputs"]["operation"]),
                         (["h1_m4", 0], ["h1_f0", 0], "or"))
        # A cord is thin: the picture's and Klein's, each further than the head.
        self.assertEqual([(g["h1_f%d" % f]["inputs"]["mask"], g["h1_f%d" % f]["inputs"]["expand"])
                          for f in (1, 2)],
                         [(["h1_m2a", 0], hs.CORD * 270 // hs.SIDE),
                          (["h1_m2b", 0], hs.CORD * 270 // hs.SIDE)])
        self.assertNotIn("h1_f3", g)              # Klein's own hair: no further
        self.assertEqual(g["h1_m5"]["inputs"]["destination"], ["h1_f2_or", 0])
        # Its colour moved towards the picture's, then put back.
        tone = g["h1_tone"]["inputs"]
        self.assertEqual((tone["image_target"], tone["image_ref"], tone["strength"]),
                         (["h1_small", 0], ["h1_cut", 0], hs.TONE))
        put = g["h1_put"]["inputs"]
        self.assertEqual((put["x"], put["y"], put["mask"], put["source"]),
                         (300, 100, ["h1_b2", 0], ["h1_tone", 0]))
        plain = hs.head_graph("picture.png", heads[:1], 7, "x", "sam3.pt",
                              words=["head:1"], tone=0)
        self.assertNotIn("h1_tone", plain)
        self.assertEqual(plain["h1_put"]["inputs"]["source"], ["h1_small", 0])
        self.assertEqual(plain["h1_m4"]["inputs"]["mask"], ["h1_m0b_or", 0])
        # No hair asked for, or no reach: the head's mask alone.
        self.assertNotIn("h1_f0", plain)
        self.assertEqual(plain["h1_m5"]["inputs"]["destination"], ["h1_m4", 0])
        bare = hs.head_graph("picture.png", heads[:1], 7, "x", "sam3.pt",
                             words=["head", "hair"], strands=0)
        self.assertNotIn("h1_f0", bare)
        # A photo cut to its head, when the head says where.
        cut = hs.head_graph("picture.png", [dict(heads[0], photo_crop={
            "x": 1, "y": 2, "width": 30, "height": 30})], 7, "x", "sam3.pt")
        self.assertEqual(cut["h1_fit"]["inputs"]["image"], ["h1_pcut", 0])
        self.assertEqual(cut["h1_pcut"]["inputs"]["image"], ["h1_photo", 0])
        self.assertEqual(g["h1_fit"]["inputs"]["image"], ["h1_photo", 0])
        self.assertLessEqual(g["h1_b1"]["inputs"]["sigma"], 10.0)
        # The second head is drawn on the first's picture, on its own seed.
        self.assertEqual(g["h2_cut"]["inputs"]["image"], ["h1_put", 0])
        self.assertEqual([g["h%d_noise" % k]["inputs"]["noise_seed"] for k in (1, 2)], [7, 8])
        self.assertEqual(g["save"]["inputs"]["images"], ["h2_put", 0])
        self.assertEqual(json.loads(json.dumps(g)), g)

    def test_sam3_is_asked_in_bare_words(self):
        # ComfyUI's SAM3 encoder reads a lone "hair:1" as that text, and SAM3
        # answered it with the whole person, or the glasses and a bottle.
        for word in hs.WORDS:
            self.assertNotIn(":", word)
        self.assertIn(hs.OLD_HAIR, hs.WORDS)
        # A count on the word is still the old hair's.
        g = hs.head_graph("picture.png", [{"crop": {"x": 0, "y": 0, "width": 512,
                                                    "height": 512}, "photo": "a.png"}],
                          7, "x", "sam3.pt", words=["head:2", "hair:2"])
        self.assertEqual(g["h1_f0"]["inputs"]["mask"], ["h1_m1a", 0])

    def test_the_soft_square_has_no_margin_where_the_crop_ends_at_the_pictures_edge(self):
        def square(crop, size=None):
            g = hs.head_graph("picture.png", [{"crop": crop, "photo": "a.png"}], 7, "x",
                              "sam3.pt", size=size)
            return (g["h1_q1"]["inputs"]["width"], g["h1_q1"]["inputs"]["height"],
                    g["h1_q2"]["inputs"]["x"], g["h1_q2"]["inputs"]["y"])
        pad = int(1000 * hs.EDGE)
        inside = {"x": 200, "y": 300, "width": 1000, "height": 1000}
        self.assertEqual(square(inside, (2000, 2000)),
                         (1000 - 2 * pad, 1000 - 2 * pad, pad, pad))
        # A close-up: the crop is the whole picture, and the top of the old
        # hair was kept by a margin there.
        whole = {"x": 0, "y": 0, "width": 1000, "height": 1000}
        self.assertEqual(square(whole, (1000, 1000)), (1000, 1000, 0, 0))
        corner = {"x": 0, "y": 500, "width": 1000, "height": 1000}
        self.assertEqual(square(corner, (1600, 1500)), (1000 - pad, 1000 - pad, 0, pad))
        # A caller that does not say how big the picture is: a margin all round.
        self.assertEqual(square(whole), (1000 - 2 * pad, 1000 - 2 * pad, pad, pad))

    def test_klein_is_told_what_the_picture_keeps_before_what_is_not_copied(self):
        keeps = hs.PROMPT.index("the necklace or jewellery image 1 wears")
        self.assertLess(keeps, hs.PROMPT.index("Nothing that is worn in image 2"))
        self.assertIn("every loose strand", hs.PROMPT)

    def test_the_swap_model_is_inswapper_unless_the_profile_names_another(self):
        self.assertEqual(ff.SWAP_MODEL, "inswapper_128")
        self.assertEqual(ff.model({}), "inswapper_128")
        self.assertEqual(ff.model({"swap_model": "hyperswap_1a_256"}), "hyperswap_1a_256")
        self.assertEqual(ff.model({"swap_model": "made_up"}), "inswapper_128")
        self.assertEqual(ig.clean_identity({"name": "X"})["swap_model"], "")
        seen = []

        def spawn(args, **kw):
            seen.append(args)
            raise RuntimeError("stop here")
        with patch.object(ff, "available", return_value=True), \
                patch.object(ff.studio_procs, "spawn", side_effect=spawn), \
                patch.object(ff.os.path, "isfile", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "stop here"):
                ff.swap(PNG, {"id": "p", "name": "P", "references": ["p.png"]})
        self.assertEqual(seen[0][seen[0].index("--model") + 1], "inswapper_128")
        self.assertEqual(seen[0][seen[0].index("--tone") + 1], str(ff.SWAP_TONE))


class KleinClient(FaceClient):
    """A ComfyUI with SAM3 and FLUX.2 Klein: its finders see one face."""
    fail_head = False

    def inventory(self):
        inv = super().inventory()
        for kind, names in KLEIN.items():
            inv[kind] = set(inv.get(kind) or ()) | names
        return inv

    def node_types(self):
        return set(FaceClient.NODES) | hs.NODES

    def listen_for_progress(self, pid, on_event, stop=None, timeout=0):
        graph = self.graphs[int(pid[3:]) - 1]
        if "p0d" in graph:
            outputs = {"7": {"text": ["1024"]}, "8": {"text": ["1024"]}}
            for k in (k for k in graph if k.endswith("t") and k.startswith("p")):
                found = [(450, 250, 80, 90)] if graph[k]["inputs"]["text"] == "face:8" else []
                outputs[k[:-1] + "v"] = {"text": [json.dumps([[
                    {"x": x, "y": y, "width": w, "height": h} for x, y, w, h in found]])]}
            return {"status": {"completed": True}, "outputs": outputs}
        if "h1_ks" in graph:
            on_event("progress", (2, 4, "h1_ks"))
            if KleinClient.fail_head:
                return {"status": {"messages": [["execution_error", {
                    "node_id": "h1_ks", "node_type": "SamplerCustomAdvanced",
                    "exception_type": "RuntimeError", "exception_message": "boom"}]]},
                    "outputs": {}}
            return {"status": {"completed": True}, "outputs": {"save": {"images": [
                {"filename": "x_head_00001_.png", "subfolder": "ImageStudio",
                 "type": "output"}]}}}
        return super().listen_for_progress(pid, on_event, stop, timeout)

    def fetch(self, f):
        return HEAD_PNG if "_head_" in f["filename"] else PNG


class TestHeadSwapInGenerate(TempStudioMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        KleinClient.fail_head = False
        self.photo = os.path.join(self.dir, "reference.png")
        Path(self.photo).write_bytes(PNG)
        self.studio.lib.save("identities", [{"id": "person", "name": "Person",
                                             "references": [self.photo],
                                             "use_references": False}])

    def generate(self, client=KleinClient, **settings):
        self.studio.client_factory = client
        self.studio.clients = {}
        swapped, self.pointed = [], []
        s = dict(ig.default_settings(), model="z-image-turbo", backend="5090",
                 scene="A portrait", identities=["person"], auto_refine=False,
                 hand_pass=False, **settings)
        with patch.object(ff, "available", return_value=True), \
                patch.object(ff, "swap", side_effect=lambda data, who, **k: (
                    swapped.append(data) or self.pointed.append(who.get("target_point"))
                    or (PNG, {"outside_mask_changed_pixels": 0}))):
            jobs = self.studio.submit(s)
            settle(jobs)
        return jobs[0], FakeClient.instances[-1], swapped

    def heads(self, client):
        return [g for g in client.graphs if "h1_ks" in g]

    def test_the_head_is_redrawn_from_the_photo_before_the_face_is_swapped(self):
        with patch.object(hs, "head_graph", wraps=hs.head_graph) as asked:
            job, client, swapped = self.generate()
        self.assertEqual(job.status, "complete", job.detail)
        find, head = client.graphs[1:3]
        self.assertEqual(find["p0t"]["inputs"]["text"], hs.FIND)
        self.assertNotIn("p1t", find)
        self.assertEqual(head["h1_cut"]["inputs"]["crop_region"],
                         hs.head_crop(1024, 1024, (450, 250, 80, 90)))
        self.assertEqual(head["h1_photo"]["inputs"]["image"], "studio_reference.png")
        self.assertIn(self.photo, client.uploads)
        self.assertNotIn("h2_ks", head)
        # The graph is told how big the picture is, for its edges.
        self.assertEqual(asked.call_args.kwargs.get("size"), (1024, 1024))
        # FaceFusion swaps the face on Klein's head, not on the generated one,
        # and is pointed at it: its own finder may see a passer-by as well.
        self.assertEqual(swapped, [HEAD_PNG])
        self.assertEqual(self.pointed, [[0.4785, 0.2881]])      # (450 + 40, 250 + 45) / 1024
        self.assertEqual(hs.middle(1024, 1024, (450, 250, 80, 90)),
                         tuple(self.pointed[0]))
        # The library's own record of the person is not pointed anywhere.
        self.assertNotIn("target_point", self.studio.lib.get("identities", "person"))
        self.assertIn("Head swap", [p["label"] for p in job.record["passes"]])
        self.assertIn("Head swap before the face swap: Person's head redrawn from their "
                      "photo by FLUX.2 Klein.", job.record["notes"])
        self.assertIn(("head_swap", "Head swap"), ig.pipeline_stages(self.studio.lib, job.settings))
        # The checkpoint kept before the faces is the picture as it was generated.
        kept = [r for r in self.studio.history.list() if r["id"].endswith("-generated")]
        self.assertFalse(kept)                # finished: the checkpoint became the result

    def test_a_failed_head_swap_leaves_the_face_swap_the_generated_picture(self):
        KleinClient.fail_head = True
        job, client, swapped = self.generate()
        self.assertEqual(job.status, "complete", job.detail)
        self.assertEqual(len(self.heads(client)), 1)
        self.assertEqual(swapped, [PNG])
        # No head was redrawn: the face is found as it ever was.
        self.assertEqual(self.pointed, [None])
        self.assertTrue(any(n.startswith("The head swap before the face swap could not run")
                            for n in job.record["notes"]), job.record["notes"])

    def test_a_backend_without_klein_swaps_the_face_alone_and_says_so(self):
        job, client, swapped = self.generate(client=FaceClient)
        self.assertEqual(job.status, "complete", job.detail)
        self.assertEqual(self.heads(client), [])
        self.assertEqual(swapped, [PNG])
        self.assertTrue(any(n.startswith("No head swap before the face swap: ")
                            and hs.KLEIN in n for n in job.record["notes"]), job.record["notes"])

    def test_the_head_swap_can_be_turned_off(self):
        job, client, swapped = self.generate(head_swap=False)
        self.assertEqual(job.status, "complete", job.detail)
        self.assertEqual(self.heads(client), [])
        self.assertEqual(swapped, [PNG])
        self.assertEqual(self.pointed, [None])
        self.assertFalse(any("head swap" in n.lower() for n in job.record["notes"]),
                         job.record["notes"])
        self.assertNotIn(("head_swap", "Head swap"),
                         ig.pipeline_stages(self.studio.lib, job.settings))

    def test_a_picture_saved_before_the_head_swap_has_it_on(self):
        self.assertTrue(ig.default_settings()["head_swap"])
        old = {k: v for k, v in ig.default_settings().items() if k != "head_swap"}
        old.update(identities=["person"])
        self.assertIn(("head_swap", "Head swap"), ig.pipeline_stages(self.studio.lib, old))


if __name__ == "__main__":
    unittest.main()
