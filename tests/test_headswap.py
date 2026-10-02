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
        # The head and the hair, each before and after, all four as one mask.
        self.assertEqual([g["word%d" % w]["inputs"]["text"] for w in (0, 1)],
                         ["head:1", "hair:1"])
        self.assertEqual([(g["h1_m%d%s" % (w, k)]["inputs"]["conditioning"][0],
                           g["h1_m%d%s" % (w, k)]["inputs"]["image"][0])
                          for w in (0, 1) for k in "ab"],
                         [("word0", "h1_cut"), ("word0", "h1_small"),
                          ("word1", "h1_cut"), ("word1", "h1_small")])
        self.assertEqual(g["h1_m4"]["inputs"]["mask"], ["h1_m1b_or", 0])
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
        swapped = []
        s = dict(ig.default_settings(), model="z-image-turbo", backend="5090",
                 scene="A portrait", identities=["person"], auto_refine=False,
                 hand_pass=False, **settings)
        with patch.object(ff, "available", return_value=True), \
                patch.object(ff, "swap", side_effect=lambda data, *a, **k: (
                    swapped.append(data) or (PNG, {"outside_mask_changed_pixels": 0}))):
            jobs = self.studio.submit(s)
            settle(jobs)
        return jobs[0], FakeClient.instances[-1], swapped

    def heads(self, client):
        return [g for g in client.graphs if "h1_ks" in g]

    def test_the_head_is_redrawn_from_the_photo_before_the_face_is_swapped(self):
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
        # FaceFusion swaps the face on Klein's head, not on the generated one.
        self.assertEqual(swapped, [HEAD_PNG])
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
        self.assertFalse(any("head swap" in n.lower() for n in job.record["notes"]),
                         job.record["notes"])
        self.assertNotIn(("head_swap", "Head swap"),
                         ig.pipeline_stages(self.studio.lib, job.settings))

    # ------------------------------------------------- Retry face swap
    def kept(self):
        """A Generate whose face swap failed: -> the picture kept before it."""
        self.studio.client_factory = KleinClient
        self.studio.clients = {}
        s = dict(ig.default_settings(), model="z-image-turbo", backend="5090",
                 scene="A portrait", identities=["person"], auto_refine=False,
                 hand_pass=False)
        with patch.object(ff, "available", return_value=True), \
                patch.object(ff, "swap", side_effect=RuntimeError("failed")):
            jobs = self.studio.submit(s)
            settle(jobs)
        self.assertEqual(jobs[0].status, "failed")
        return self.studio.history.list()[0]

    def retry(self, kept):
        swapped = []
        with patch.object(ff, "available", return_value=True), \
                patch.object(ff, "swap", side_effect=lambda data, *a, **k: (
                    swapped.append(data) or (PNG, {"outside_mask_changed_pixels": 0}))):
            jobs = self.studio.submit(ig.retry_faces(kept))
            settle(jobs)
        return jobs[0], swapped

    def test_a_retried_face_swap_finishes_the_picture_as_the_job_would_have(self):
        whole, _, _ = self.generate()         # what an unbroken job runs and records
        kept = self.kept()
        client = FakeClient.instances[-1]
        before = len(client.graphs)
        job, swapped = self.retry(kept)
        self.assertEqual(job.status, "complete", job.detail)
        self.assertEqual(job.detail, "")
        self.assertEqual(job.backend["id"], "5090")       # on the lane it was made on
        # The head is redrawn again, and the face goes onto Klein's head.
        self.assertEqual(len([g for g in client.graphs[before:] if "h1_ks" in g]), 1)
        self.assertEqual(swapped, [HEAD_PNG])
        # Every pass the unbroken job ran, and a record that is the picture's own.
        labels = [p["label"] for p in whole.record["passes"]]
        self.assertIn("Eye pass", labels)
        self.assertEqual([p["label"] for p in job.record["passes"]], labels)
        self.assertEqual(job.record["workflow"], whole.record["workflow"])
        self.assertEqual(job.record["prompt"], whole.record["prompt"])
        self.assertEqual(job.record["graph"], kept["graph"])
        self.assertNotIn("mode", job.record["settings"])
        self.assertNotIn("face_finish", job.record["settings"])
        self.assertEqual(job.record["settings"]["scene"], "A portrait")
        self.assertIn("Finished by Retry face swap, from the picture kept before the face "
                      "swap.", job.record["notes"])
        # The kept picture became the result: History shows the two finished ones.
        self.assertEqual([r["id"] for r in self.studio.history.list()
                          if r["id"].endswith("-generated")], [])

    def test_a_retry_with_its_backend_down_swaps_the_face_alone_and_says_so(self):
        kept = self.kept()
        client = FakeClient.instances[-1]
        before = len(client.graphs)
        FakeClient.down = {"5090"}
        job, swapped = self.retry(kept)
        self.assertEqual(job.status, "complete", job.detail)
        self.assertEqual(job.backend["id"], ig.LOCAL_FACES["id"])
        self.assertEqual(client.graphs[before:], [])      # nothing was asked of ComfyUI
        self.assertEqual(swapped, [PNG])
        said = ("This retry swapped the face alone - no head swap before it and no eye, "
                "hand or glasses pass after: 5090 Workstation is not answering.")
        self.assertEqual(job.detail, said)
        self.assertIn(said, job.record["notes"])

    def test_a_retry_whose_settings_no_longer_compose_swaps_the_face_alone(self):
        kept = self.kept()
        kept["settings"]["model"] = "a model since removed"
        job, swapped = self.retry(kept)
        self.assertEqual(job.status, "complete", job.detail)
        self.assertEqual(swapped, [PNG])
        self.assertIn("No model called 'a model since removed' in the library", job.detail)

    def test_only_a_generated_picture_is_owed_more_than_the_swap(self):
        rec = {"images": ["a.png"], "seed": 3, "path": "r.json",
               "backend": {"id": "5090"}, "finish": {"profiles": [{"id": "person"}]},
               "settings": {"scene": "A portrait", "head_swap": False}}
        retry = ig.retry_faces(rec)
        self.assertEqual((retry["mode"], retry["seed"], retry["batch"]), ("faces", 3, 1))
        self.assertEqual(retry["face_finish"]["backend"], "5090")
        self.assertEqual((retry["scene"], retry["head_swap"]), ("A portrait", False))
        # A face-only swap's kept picture (Fix a spot, an earlier swap) is retried as it was.
        for settings in ({"mode": "fix", "scene": "x"}, {"mode": "faces"}, None):
            alone = ig.retry_faces(dict(rec, settings=settings))
            self.assertEqual(sorted(alone), ["batch", "face_finish", "mode", "seed"])
            self.assertNotIn("backend", alone["face_finish"])
        self.assertIsNone(self.studio.finish_backend(alone))
        self.assertEqual(self.studio.plan_route(alone)[0]["id"], ig.LOCAL_FACES["id"])
        # The job's strip shows what the retry runs: nothing is sampled again.
        lib = self.studio.lib
        self.assertEqual([k for k, _ in ig.pipeline_stages(lib, alone)],
                         ["face_swap", "complete"])
        self.assertEqual([k for k, _ in ig.pipeline_stages(lib, retry)],
                         ["face_swap", "eyes", "hands", "complete"])
        on = ig.retry_faces(dict(rec, settings={"scene": "A still life", "hand_pass": False}))
        self.assertEqual([k for k, _ in ig.pipeline_stages(lib, on)],
                         ["head_swap", "face_swap", "eyes", "complete"])

    def test_a_picture_saved_before_the_head_swap_has_it_on(self):
        self.assertTrue(ig.default_settings()["head_swap"])
        old = {k: v for k, v in ig.default_settings().items() if k != "head_swap"}
        old.update(identities=["person"])
        self.assertIn(("head_swap", "Head swap"), ig.pipeline_stages(self.studio.lib, old))


if __name__ == "__main__":
    unittest.main()
