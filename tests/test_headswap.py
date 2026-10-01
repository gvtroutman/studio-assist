"""The Klein head swap before the final face swap; no apps, models or network."""
import json
import os
import tempfile
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

    def test_a_klein_9b_lora_is_known_by_its_name_and_fits_only_the_9b(self):
        self.assertEqual(ig.guess_family("partner_klein-9b_head.safetensors"), hs.FAMILY)
        self.assertEqual(ig.guess_family("flux-2-klein-4b-lora.safetensors"), "flux2")
        self.assertIs(ig.compatibility("flux2", hs.FAMILY), False)
        self.assertIs(ig.compatibility(hs.FAMILY, hs.FAMILY), True)

    def test_a_licence_is_written_into_a_png_and_nothing_else(self):
        out = ig.png_text(PNG, "Comment", "Not for commercial use.")
        self.assertEqual(out[:8], PNG[:8])
        self.assertIn(b"tEXtComment\0Not for commercial use.", out)
        self.assertLess(out.index(b"tEXt"), out.index(b"IDAT") if b"IDAT" in out else len(out))
        self.assertEqual(ig.png_text(b"\xff\xd8 jpeg", "Comment", "x"), b"\xff\xd8 jpeg")

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
        # ...then what the head wears, before and after (test below).
        self.assertEqual(g["h1_wha_or"]["inputs"]["destination"], ["h1_m2b_or", 0])
        self.assertEqual(g["h1_m4"]["inputs"]["mask"], ["h1_whb_or", 0])
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
        self.assertEqual(g["h1_m6"]["inputs"]["mask"], ["h1_f2_or", 0])
        self.assertEqual(g["h1_m5"]["inputs"]["destination"], ["h1_m6", 0])
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
        self.assertEqual(plain["h1_m4"]["inputs"]["mask"], ["h1_whb_or", 0])
        self.assertEqual(plain["h1_wnb"]["inputs"]["mask"], ["h1_m0b", 0])   # near the head alone
        # No hair asked for, or no reach: the head's mask alone.
        self.assertNotIn("h1_f0", plain)
        self.assertEqual(plain["h1_m6"]["inputs"]["mask"], ["h1_m4", 0])
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

    def test_a_head_with_its_own_lora_is_drawn_through_it_and_named_by_its_trigger(self):
        heads = [{"crop": {"x": 300, "y": 100, "width": 270, "height": 270}, "photo": "a.png",
                  "lora": "partner_head_klein.safetensors", "strength": 0.9, "trigger": "partner"},
                 {"crop": {"x": 600, "y": 120, "width": 240, "height": 240}, "photo": "b.png"}]
        g = hs.head_graph("picture.png", heads, 7, "x", "sam3.pt")
        lora = g["h1_lora"]["inputs"]
        self.assertEqual((lora["model"], lora["lora_name"], lora["strength_model"]),
                         (["unet", 0], "partner_head_klein.safetensors", 0.9))
        self.assertEqual(g["h1_guide"]["inputs"]["model"], ["h1_lora", 0])
        self.assertIn("the head of partner, the person in image 2", g["h1_text"]["inputs"]["text"])
        self.assertEqual(g["h1_zero"]["inputs"]["conditioning"], ["h1_text", 0])
        self.assertEqual(g["h1_pos1"]["inputs"]["conditioning"], ["h1_text", 0])
        self.assertEqual(g["h1_neg1"]["inputs"]["conditioning"], ["h1_zero", 0])
        # The other person: Klein as it is, the shared prompt, no trigger.
        self.assertNotIn("h2_lora", g)
        self.assertEqual(g["h2_guide"]["inputs"]["model"], ["unet", 0])
        self.assertEqual(g["h2_pos1"]["inputs"]["conditioning"], ["pos", 0])
        self.assertEqual(g["pos"]["inputs"]["text"], hs.PROMPT)
        self.assertEqual(hs.prompt_for(None), hs.PROMPT)
        self.assertNotEqual(hs.prompt_for("partner"), hs.PROMPT)
        self.assertEqual(json.loads(json.dumps(g)), g)

    def test_a_profiles_head_lora_is_used_only_when_it_can_be(self):
        with tempfile.TemporaryDirectory() as d:
            lib = ig.Library(d)
            rec, _ = lib.import_lora({"file": "p_head.safetensors", "name": "P head",
                                      "category": "Identity", "family": "flux2-klein9b",
                                      "trigger": "pperson", "strength": 1.0})
            other, _ = lib.import_lora({"file": "p_flux1.safetensors", "name": "P flux",
                                        "category": "Identity", "family": "flux1"})
            # Made for Klein 4B by Build LoRA before 2026-10-01: not the 9B's shape.
            four, _ = lib.import_lora({"file": "p_head_4b.safetensors", "name": "P 4B",
                                       "category": "Identity", "family": "flux2"})
            p = {"name": "P", "head_lora": rec["id"]}
            inv = {"loras": {"p_head.safetensors"}}
            self.assertEqual(ig.head_lora(lib, p, "5090", inv),
                             ({"lora": "p_head.safetensors", "strength": 1.0,
                               "trigger": "pperson"}, None))
            self.assertEqual(ig.head_lora(lib, {"name": "P"}, "5090", inv), ({}, None))
            for profile, inventory, why in (
                    (dict(p, head_lora="gone"), inv, "not in the LoRA library"),
                    (dict(p, head_lora=other["id"]), inv, "not FLUX.2 Klein 9B"),
                    (dict(p, head_lora=four["id"]), inv,
                     "is for FLUX.2, not FLUX.2 Klein 9B: rebuild it with Build LoRA"),
                    (p, {"loras": set()}, "not on this backend")):
                own, note = ig.head_lora(lib, profile, "5090", inventory)
                self.assertEqual(own, {})
                self.assertIn(why, note)
        self.assertEqual(ig.clean_identity({"name": "X", "head_lora": "h"})["head_lora"], "h")
        self.assertEqual(ig.clean_identity({"name": "X"})["head_lora"], "")

    def test_what_the_head_wears_goes_with_it_only_near_the_head(self):
        # A flower crown outside SAM3's head and hair stayed as a ghost round
        # the new head; "headwear" also found a lily and a lace collar.
        head = {"crop": {"x": 0, "y": 0, "width": 1344, "height": 1344}, "photo": "a.png"}
        g = hs.head_graph("picture.png", [head], 7, "x", "sam3.pt")
        self.assertEqual(g["worn"]["inputs"]["text"], hs.WORN)
        self.assertNotIn(":", hs.WORN)
        near = int(1344 / hs.CROP * hs.WORN_NEAR)
        for k, src in (("a", "h1_cut"), ("b", "h1_small")):
            self.assertEqual((g["h1_w" + k]["inputs"]["image"],
                              g["h1_w" + k]["inputs"]["conditioning"]), ([src, 0], ["worn", 0]))
            # Near that picture's own head and hair, before and after alike.
            anchor = g["h1_w%sa1" % k]["inputs"]
            self.assertEqual((anchor["destination"], anchor["source"], anchor["operation"]),
                             (["h1_m0" + k, 0], ["h1_m1" + k, 0], "or"))
            self.assertEqual((g["h1_wn" + k]["inputs"]["mask"], g["h1_wn" + k]["inputs"]["expand"],
                              g["h1_wn" + k]["inputs"]["tapered_corners"]),
                             (["h1_w%sa1" % k, 0], near, False))
            only = g["h1_wh" + k]["inputs"]
            self.assertEqual((only["destination"], only["source"], only["operation"]),
                             (["h1_w" + k, 0], ["h1_wn" + k, 0], "multiply"))
        # Neither head nor hair asked for: nothing to be near, no headwear.
        alone = hs.head_graph("picture.png", [head], 7, "x", "sam3.pt", words=["necklace"])
        self.assertNotIn("worn", alone)
        self.assertNotIn("h1_wa", alone)
        self.assertEqual(json.loads(json.dumps(g)), g)

    def test_the_blends_edge_is_wide_and_leaves_all_it_covers_klein_s(self):
        # Klein's background is a few levels off the picture's: a 26 px edge
        # showed the old crown's outline in it.
        head = {"crop": {"x": 0, "y": 0, "width": 1344, "height": 1344}, "photo": "a.png"}
        g = hs.head_graph("picture.png", [head], 7, "x", "sam3.pt")
        wide = int(1344 * hs.FEATHER)
        grow = g["h1_m6"]["inputs"]
        self.assertEqual((grow["mask"], grow["expand"]), (["h1_f2_or", 0], wide))
        self.assertEqual(g["h1_bs"]["inputs"]["scale_by"], 1.0 / hs.SHRINK)
        blur = g["h1_b1"]["inputs"]
        self.assertEqual(blur["image"], ["h1_bs", 0])
        self.assertLessEqual(blur["blur_radius"], hs.BLUR_MAX)
        self.assertLessEqual(blur["sigma"], 10.0)
        # Two sigmas (at full size) in from the grown edge: what was covered
        # is still Klein's.
        self.assertGreaterEqual(wide, 2 * blur["sigma"] * hs.SHRINK - 1)
        self.assertEqual((g["h1_bu"]["inputs"]["width"], g["h1_bu"]["inputs"]["height"]),
                         (1344, 1344))
        self.assertEqual(g["h1_b2"]["inputs"]["image"], ["h1_bu", 0])
        # No feather: the short blur at full size, as before.
        flat = hs.head_graph("picture.png", [head], 7, "x", "sam3.pt", feather=0)
        self.assertNotIn("h1_m6", flat)
        self.assertNotIn("h1_bs", flat)
        self.assertEqual(flat["h1_b1"]["inputs"]["image"], ["h1_b0", 0])
        self.assertEqual(flat["h1_b2"]["inputs"]["image"], ["h1_b1", 0])

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
                      "photo by FLUX.2 Klein 9B.", job.record["notes"])
        # The 9B's licence goes with the picture: on its record and in the file.
        self.assertEqual(job.record["license"], hs.LICENSE_NOTE)
        with open(job.record["images"][0], "rb") as f:
            self.assertIn(b"tEXtComment\0" + hs.LICENSE_NOTE.encode("latin-1"), f.read())
        self.assertIn(("head_swap", "Head swap"), ig.pipeline_stages(self.studio.lib, job.settings))
        # The checkpoint kept before the faces is the picture as it was generated.
        kept = [r for r in self.studio.history.list() if r["id"].endswith("-generated")]
        self.assertFalse(kept)                # finished: the checkpoint became the result

    def test_the_persons_head_lora_draws_their_head_and_stays_out_of_the_picture(self):
        rec, _ = self.studio.lib.import_lora({
            "file": "person_head_klein.safetensors", "name": "Person head",
            "category": "Identity", "family": "flux2-klein9b", "trigger": "perperson",
            "strength": 1.0})
        self.studio.lib.save("loras")
        ident = self.studio.lib.get("identities", "person")
        ident["head_lora"] = rec["id"]
        self.studio.lib.save("identities")

        class WithLora(KleinClient):
            def inventory(self):
                inv = super().inventory()
                inv["loras"] = set(inv.get("loras") or ()) | {"person_head_klein.safetensors"}
                return inv
        job, client, swapped = self.generate(client=WithLora)
        self.assertEqual(job.status, "complete", job.detail)
        head = self.heads(client)[0]
        self.assertEqual(head["h1_lora"]["inputs"]["lora_name"], "person_head_klein.safetensors")
        self.assertEqual(head["h1_guide"]["inputs"]["model"], ["h1_lora", 0])
        self.assertIn("the head of perperson", head["h1_text"]["inputs"]["text"])
        self.assertIn("Head swap before the face swap: Person's head (with their head LoRA) "
                      "redrawn from their photo by FLUX.2 Klein 9B.", job.record["notes"])
        # A Klein LoRA is the head swap's alone: never in the picture's own stack.
        self.assertNotIn("person_head_klein", json.dumps(client.graphs[0]))
        self.assertNotIn("perperson", json.dumps(client.graphs[0]))
        # Not on the backend: Klein as it is, and the note says why.
        self.studio.inventories.clear()       # the next check reads the plain backend
        job, client, swapped = self.generate()
        self.assertEqual(job.status, "complete", job.detail)
        self.assertNotIn("h1_lora", self.heads(client)[0])
        self.assertIn("Person's head LoRA person_head_klein.safetensors is not on this backend.",
                      job.record["notes"])

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

    def test_a_head_swap_failing_on_the_second_picture_keeps_the_first(self):
        # Idea 69c4e1054c3a: the except returned every original picture, yet
        # the first one's note, heads and licence stayed - History claimed a
        # swap the picture lacked.
        for failure in (ig.ComfyError("gone"), KeyError("box")):
            with self.subTest(failure=type(failure).__name__):
                class FailSecond(KleinClient):
                    runs = 0

                    def listen_for_progress(self, pid, on_event, stop=None, timeout=0):
                        if "h1_ks" in self.graphs[int(pid[3:]) - 1]:
                            FailSecond.runs += 1
                            if FailSecond.runs == 2:
                                raise failure
                        return super().listen_for_progress(pid, on_event, stop, timeout)
                backend = self.backend("5090")
                client = FailSecond(backend)
                self.studio.inventories[backend["id"]] = client.inventory()
                job = ig.Job(dict(ig.default_settings(), backend="5090"), backend)
                with patch.object(ig.doctor, "log_error") as logged:
                    out = self.studio._head_swap(
                        job, client, {"seed": 1, "filename_prefix": "x"},
                        [("a.png", PNG), ("b.png", PNG)],
                        [self.studio.lib.get("identities", "person")], lambda *a: None)
                self.assertEqual(out, [("a.png", HEAD_PNG), ("b.png", PNG)])
                self.assertEqual(set(job.heads), {(0, "person")})
                self.assertEqual(job.license, hs.LICENSE_NOTE)
                # The failed run is not listed as a pass made.
                self.assertEqual([p["label"] for p in job.passes], ["Head swap"])
                self.assertEqual(sum(n.startswith("Head swap before the face swap: ")
                                     for n in job.notes), 1, job.notes)
                self.assertTrue(any("could not run" in n and "after the first 1" in n
                                    for n in job.notes), job.notes)
                # Only an error nobody planned for gets its traceback logged.
                self.assertEqual(logged.called, isinstance(failure, KeyError))
                if logged.called:
                    self.assertIn("Traceback", logged.call_args.args[0])

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
        # No Klein 9B in the picture, so no licence on it.
        self.assertEqual(job.record["license"], "")
        with open(job.record["images"][0], "rb") as f:
            self.assertNotIn(b"tEXt", f.read())
        self.assertNotIn(("head_swap", "Head swap"),
                         ig.pipeline_stages(self.studio.lib, job.settings))

    def test_a_picture_saved_before_the_head_swap_has_it_on(self):
        self.assertTrue(ig.default_settings()["head_swap"])
        old = {k: v for k, v in ig.default_settings().items() if k != "head_swap"}
        old.update(identities=["person"])
        self.assertIn(("head_swap", "Head swap"), ig.pipeline_stages(self.studio.lib, old))


if __name__ == "__main__":
    unittest.main()
