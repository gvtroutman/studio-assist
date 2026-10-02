"""Recovery and cancellation regressions; no apps, models or network."""
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import apps.image_studio.facefusion as ff
import apps.image_studio.imagegen as ig
import apps.image_studio.scene.scene as sc
from test_imagegen import TempStudioMixin, PNG, settle, FakeClient, FaceClient


class TestFaceTargets(unittest.TestCase):
    def test_scene_builder_payload_carries_the_character_identity(self):
        scene = sc.new_scene()
        person = sc.new_object('person')
        person['character'] = 'character'
        scene['objects'].append(person)
        targets = sc.face_targets(scene, {'character': {'identity': 'person'}}, {})
        self.assertEqual(targets[0]['identity'], 'person')

    def test_lora_only_identities_do_not_require_a_face_swap(self):
        lib = SimpleNamespace(get=lambda *_: {'id': 'a', 'name': 'A', 'lora': 'a', 'references': []})
        self.assertEqual(ff.selected(lib, {'identities': ['a']}), [])

    def test_scene_regions_override_profile_selection_order(self):
        profiles = {"a": {"id": "a", "name": "A", "references": ["a.png"]},
                    "b": {"id": "b", "name": "B", "references": ["b.png"]}}
        lib = SimpleNamespace(get=lambda kind, key: profiles.get(key))
        settings = {"identities": ["a", "b"], "scene_faces": {"people": [
            {"id": "p1", "identity": "a", "region": [.6, .1, 1, .6]},
            {"id": "p2", "identity": "b", "region": [0, .1, .4, .6]}]}}
        selected = ff.selected(lib, settings)
        boxes = [[.1, .2, .3, .4], [.7, .2, .9, .4]]
        self.assertEqual([ff.target_face(boxes, region=p['target_region']) for p in selected], [1, 0])
        self.assertNotIn('target_region', profiles['a'])

    def test_ambiguous_extra_and_missing_faces_do_not_guess(self):
        boxes = [[.1, .2, .3, .4], [.4, .2, .6, .4], [.7, .2, .9, .4]]
        for options in ({}, {"region": [0, 0, 1, 1]}, {"index": 0, "count": 2},
                        {"region": [0, .8, 1, 1]}):
            with self.subTest(options=options), self.assertRaisesRegex(RuntimeError, 'Choose face'):
                ff.target_face(boxes, **options)
        self.assertEqual(ff.target_face(boxes, point=[.8, .3]), 2)

    def test_the_finish_passes_redraw_the_face_facefusion_swapped(self):
        small, big, right = (100, 100, 40, 40), (300, 100, 90, 100), (700, 100, 60, 60)
        # One profile: FaceFusion's single face, or the biggest when SAM3 sees more.
        self.assertEqual(ig.swapped_faces(1000, 1000, [big], [{}]), [big])
        self.assertEqual(ig.swapped_faces(1000, 1000, [right, small, big], [{}]), [big])
        # Two profiles: left to right, as FaceFusion's face_index/face_count.
        self.assertEqual(ig.swapped_faces(1000, 1000, [right, big], [{}, {}]), [big, right])
        # A scene's regions pick the face; a region with no face is skipped.
        self.assertEqual(ig.swapped_faces(1000, 1000, [small, big, right],
                                          [{"target_region": [.6, 0, 1, .5]},
                                           {"target_region": [0, .8, 1, 1]}]), [right])
        spot = ig.eye_spots([big])[0]
        self.assertEqual(spot["word"], ig.EYE_WORD)
        self.assertEqual(spot["size"], int(100 * ig.EYE_PAD))
        self.assertEqual(spot["box"], [300, 115, 90, 45])       # the eye band only
        self.assertEqual(ig.glasses_spots(1000, 1000, [(310, 125, 70, 20), (40, 900, 30, 10)],
                                          [big])[0]["word"], "glasses")
        self.assertEqual(len(ig.glasses_spots(1000, 1000, [(40, 900, 30, 10)], [big])), 0)

    def test_the_eye_pass_knows_whose_face_is_whose(self):
        left, right = (100, 100, 60, 60), (700, 100, 60, 60)
        self.assertEqual(ig.swapped_pairs(1000, 1000, [right, left], [{}, {}]),
                         [(0, left), (1, right)])
        # A profile whose region holds no face is skipped; the next keeps its own index.
        self.assertEqual(ig.swapped_pairs(1000, 1000, [left, right],
                                          [{"target_region": [0, .8, 1, 1]},
                                           {"target_region": [.6, 0, 1, .5]}]), [(1, right)])

    def test_the_eye_colour_is_read_from_a_look(self):
        for look, colour in (("blue", "blue"), ("dark brown", "dark brown"),
                             ("blue-grey", "blue-grey"), ("green, hooded", "green"),
                             ("Hazel", "hazel"), ("almond-shaped", ""), ("", ""), (None, "")):
            with self.subTest(look=look):
                self.assertEqual(ig.eye_colour(look), colour)

    def test_each_profile_takes_its_characters_eye_colour(self):
        chars = [{"id": "c1", "identity": "a", "looks": {"eyes": "blue"}},
                 {"id": "c2", "identity": "a", "looks": {"eyes": "green"}},
                 {"id": "c3", "identity": "b", "looks": {"eyes": "hazel"}},
                 {"id": "c4", "identity": "b", "looks": {"eyes": "hooded"}},
                 {"id": "c5", "looks": {"eyes": "grey"}}]
        lib = SimpleNamespace(all=lambda kind: chars)
        a, b, c = {"id": "a"}, {"id": "b"}, {"id": "c"}
        # The job's own character first; else the identity's characters when they agree.
        self.assertEqual(ig.profile_eyes(lib, {"character": "c2"}, [a, b]), ["green", "hazel"])
        self.assertEqual(ig.profile_eyes(lib, {}, [a, b]), ["", "hazel"])
        # No character of it: the form's own eyes for a lone profile, nothing for two.
        self.assertEqual(ig.profile_eyes(lib, {"eyes": "light blue"}, [c]), ["light blue"])
        self.assertEqual(ig.profile_eyes(lib, {"eyes": "blue"}, [c, b]), ["", "hazel"])

    def test_new_green_counts_only_green_the_pass_drew(self):
        w, h = 20, 10
        grey = bytes([90, 90, 90]) * (w * h)
        teal = bytearray(grey)
        green = bytearray(grey)
        for y in range(2, 5):
            for x in range(2, 6):            # 12 px of green in the left band
                green[(y * w + x) * 3:(y * w + x) * 3 + 3] = bytes([60, 110, 60])
        for x in range(12, 15):              # green already there before: not counted
            teal[(2 * w + x) * 3:(2 * w + x) * 3 + 3] = bytes([60, 110, 60])
            green[(2 * w + x) * 3:(2 * w + x) * 3 + 3] = bytes([60, 110, 60])
        before, after = sc.rgb_png(bytes(teal), w, h), sc.rgb_png(bytes(green), w, h)
        self.assertEqual(ig.new_green(before, after, w, h, [(0, 0, 10, 10), (10, 0, 10, 10)]),
                         [12, 0])
        self.assertIsNone(ig.new_green(before, after, w + 1, h, [(0, 0, 10, 10)]))
        self.assertIsNone(ig.new_green(b"not a png", after, w, h, [(0, 0, 10, 10)]))

    def test_an_eye_pass_with_a_green_blob_is_drawn_again(self):
        w, h = 8, 8
        before = sc.rgb_png(bytes([90, 90, 90]) * (w * h), w, h)

        def blob(n):                         # a picture with n px of new green
            rgb = bytearray([90, 90, 90]) * (w * h)
            for i in range(n):
                rgb[i * 3:i * 3 + 3] = bytes([60, 110, 60])
            return sc.rgb_png(bytes(rgb), w, h)

        def tries(greens, fail_at=None, bands=((0, 0, w, h),)):
            job = SimpleNamespace(passes=[], notes=[])
            seeds = []

            def run(reseed):
                if len(seeds) == fail_at:
                    job.passes.append({"label": "Eye pass", "seed": reseed})
                    raise ig.ComfyError("the run failed")
                seeds.append(reseed)
                job.passes.append({"label": "Eye pass", "seed": reseed})
                return [{"filename": "eyes_%d.png" % len(seeds)}]
            client = SimpleNamespace(fetch=lambda f: blob(greens[int(f["filename"][5]) - 1]))
            kept = ig.Studio._eye_tries(None, job, client, run, before, w, h, list(bands))
            return kept, seeds, job

        old = ig.EYE_GREEN_MAX
        ig.EYE_GREEN_MAX = 20
        try:
            # Clean at once: one try, nothing said.
            kept, seeds, job = tries([3])
            self.assertEqual((kept[0]["filename"], seeds, job.notes), ("eyes_1.png", [0], []))
            # A blob, then clean: the second kept, and only its graph in the record.
            kept, seeds, job = tries([40, 2])
            self.assertEqual(kept[0]["filename"], "eyes_2.png")
            self.assertEqual(seeds, [0, ig.EYE_RESEED])
            self.assertEqual([p["seed"] for p in job.passes], [ig.EYE_RESEED])
            self.assertIn("2 tries, the one with none kept", job.notes[0])
            # Blobs every time: EYE_TRIES, the least kept.
            kept, seeds, job = tries([40, 25, 33])
            self.assertEqual((kept[0]["filename"], len(seeds)), ("eyes_2.png", ig.EYE_TRIES))
            self.assertEqual(len(job.passes), 1)
            self.assertIn("the least (25 px)", job.notes[0])
            # A second try that fails keeps the first, and no graph of the failed one.
            kept, seeds, job = tries([40], fail_at=1)
            self.assertEqual((kept[0]["filename"], [p["seed"] for p in job.passes]),
                             ("eyes_1.png", [0]))
            # Eyes meant green (no bands to look at): one try, never fetched.
            kept, seeds, job = tries([40], bands=())
            self.assertEqual(seeds, [0])
        finally:
            ig.EYE_GREEN_MAX = old

    def test_same_identity_can_appear_at_two_scene_positions(self):
        lib = SimpleNamespace(get=lambda *_: {"id": "a", "name": "A", "references": ["a.png"]})
        selected = ff.selected(lib, {"scene_faces": {"people": [
            {"identity": "a", "region": [0, 0, .4, 1]},
            {"identity": "a", "region": [.6, 0, 1, 1]}]}})
        self.assertEqual(len(selected), 2)


class TestFinishRecovery(TempStudioMixin, unittest.TestCase):
    def profile(self):
        path = os.path.join(self.dir, 'reference.png')
        Path(path).write_bytes(PNG)
        self.studio.lib.save('identities', [{'id': 'person', 'name': 'Person',
            'references': [path], 'use_references': False}])
        return path

    def settings(self):
        return dict(ig.default_settings(), model='z-image-turbo', backend='5090',
                    scene='A portrait', identities=['person'], critic_notes=False)

    def test_failed_finish_keeps_picture_and_retries_offline_with_saved_profile(self):
        self.profile()
        with patch.object(ff, 'available', return_value=True), patch.object(ff, 'swap', side_effect=RuntimeError('failed')):
            jobs = self.studio.submit(self.settings())
            settle(jobs)
        job = jobs[0]
        self.assertEqual(job.status, 'failed')
        self.assertEqual(Path(job.outputs[0]).read_bytes(), PNG)
        record = self.studio.history.list()[0]
        self.assertEqual(record['finish']['profiles'][0]['name'], 'Person')
        self.studio.lib.save('identities', [])
        self.studio.lib.save('backends', [])
        with patch.object(ff, 'available', return_value=True), patch.object(ff, 'swap', return_value=(PNG, {})) as swap, \
                patch.object(self.studio, 'client', side_effect=AssertionError('No backend needed')):
            jobs = self.studio.submit(ig.retry_faces(record))
            settle(jobs)
        self.assertEqual(jobs[0].status, 'complete', jobs[0].detail)
        self.assertEqual(swap.call_args.args[1]['name'], 'Person')
        self.assertEqual(Path(job.outputs[0]).read_bytes(), PNG)
        # The retry finishes the kept checkpoint instead of saving another copy of it.
        kept = json.loads(Path(record['path']).read_text(encoding='utf-8'))
        self.assertEqual(kept['finish']['state'], 'complete')
        self.assertEqual(kept['finish']['results'], jobs[0].outputs)
        listed = self.studio.history.list()
        self.assertEqual(len(listed), 1)
        self.assertNotIn('finish', listed[0])

    def finish_job(self, glasses, hands=(), profile=True, tone=False, masks=None,
                   enhance=None, drawn=None, **settings):
        """Generate (with a face profile, unless `profile` is False) on a
        ComfyUI with SAM3, whose finder at the end sees one face, `glasses`
        and `hands` [(x, y, w, h)] or, with SAM3's score, [(x, y, w, h,
        score)]; `tone`, it has the tone-match node. `masks` is what the
        face swap's report says its masks were (none said, before them),
        `enhance` the face enhancer it says ran after it. `drawn`: it has
        the landmark node, which reads these faces."""
        if profile:
            self.profile()

        class FinishClient(FaceClient):
            def node_types(self):
                return set(FaceClient.NODES) | ({ig.TONE_NODE} if tone else set()) | {
                    "ConditioningCombine", "ConditioningSetMask"} | (   # a scene beard's regions
                    {ig.LANDMARK_NODE} if drawn is not None else set())

            def listen_for_progress(self, pid, on_event, stop=None, timeout=0):
                graph = self.graphs[int(pid[3:]) - 1]
                if "ll" in graph:             # the drawn face's landmarks
                    return {"status": {"completed": True},
                            "outputs": {"ll": {"text": [json.dumps(drawn)]}}}
                if "p0d" in graph:            # the finder at the end
                    said = {"face:8": [(450, 250, 80, 90)], "glasses:4": glasses,
                            "hand:8": hands}
                    outputs = {"7": {"text": ["1024"]}, "8": {"text": ["1024"]}}
                    for k in (k for k in graph if k.endswith("t") and k.startswith("p")):
                        outputs[k[:-1] + "v"] = {"text": [json.dumps([[
                            dict({"x": b[0], "y": b[1], "width": b[2], "height": b[3]},
                                 **({"score": b[4]} if len(b) > 4 else {}))
                            for b in said[graph[k]["inputs"]["text"]]]])]}
                    return {"status": {"completed": True}, "outputs": outputs}
                return super().listen_for_progress(pid, on_event, stop, timeout)
        self.studio.client_factory = FinishClient
        self.studio.clients = {}
        order = []
        s = dict(self.settings(), **settings)
        if not profile:
            s["identities"] = []
        report = dict({'outside_mask_changed_pixels': 0}, **({'masks': masks} if masks else {}))
        if enhance:
            report['enhance'] = enhance
        self.graphs_at_swap = []          # how many graphs ComfyUI had run at each swap
        with patch.object(ff, 'available', return_value=True), \
                patch.object(ff, 'swap', side_effect=lambda *a, **k: (
                    order.append('swap') or self.graphs_at_swap.append(
                        len(FakeClient.instances[-1].graphs)) or (PNG, dict(report)))):
            jobs = self.studio.submit(s)
            settle(jobs)
        client = FakeClient.instances[-1]
        return jobs[0], client, order

    def test_the_face_swap_is_followed_by_an_eye_pass_then_the_glasses(self):
        job, client, order = self.finish_job([(460, 272, 60, 22), (40, 900, 30, 10)])
        self.assertEqual(job.status, 'complete', job.detail)
        self.assertEqual(order, ['swap'])
        find, eyes, glasses = client.graphs[-3:]
        self.assertEqual([find["p0t"]["inputs"]["text"], find["p1t"]["inputs"]["text"]],
                         ig.FINISH_FIND)
        self.assertTrue(find["1"]["inputs"]["image"].startswith("studio_%s" % job.id))
        # The eyes: the whole face cropped, only SAM3's eyes in its eye band redrawn.
        region = eyes["fc1_1"]["inputs"]["crop_region"]
        self.assertLessEqual(region["x"], 450)
        self.assertGreaterEqual(region["x"] + region["width"], 530)
        self.assertEqual(eyes["fc1_s0"]["inputs"]["text"], ig.EYE_WORD)
        self.assertEqual(eyes["fc1_4"]["inputs"]["denoise"], ig.EYE_DENOISE)
        self.assertNotIn("fc2_1", eyes)
        self.assertIn("eyes", eyes[eyes["fc1_4"]["inputs"]["positive"][0]]["inputs"]["text"])
        # The glasses last, on the eye pass's picture; the pair off the face is left.
        self.assertEqual(glasses["fi"]["inputs"]["image"], "ImageStudio/faces_00001_.png [output]")
        self.assertEqual(glasses["fc1_s0"]["inputs"]["text"], "glasses")
        self.assertEqual(glasses["fc1_4"]["inputs"]["denoise"], ig.GLASSES_DENOISE)
        self.assertNotIn("fc2_1", glasses)
        self.assertNotIn("fc1_t", glasses)          # no tone curves on a swapped face
        notes = job.record["notes"]
        self.assertTrue(any(n.startswith("Eye pass after the face swap") for n in notes), notes)
        self.assertTrue(any(n.startswith("Glasses redrawn last: 1 pair") for n in notes), notes)

    def test_without_glasses_only_the_eyes_are_redrawn(self):
        job, client, _ = self.finish_job([])
        self.assertEqual(job.status, 'complete', job.detail)
        self.assertIn("p0d", client.graphs[-2])
        self.assertEqual(client.graphs[-1]["fc1_s0"]["inputs"]["text"], ig.EYE_WORD)
        self.assertIn("SAM3 found no glasses on the swapped face.", job.record["notes"])
        self.assertIn("SAM3 found no hands, so no hands pass was made.", job.record["notes"])

    def test_the_eye_pass_is_told_the_persons_eye_colour(self):
        self.studio.lib.save('characters', [{'id': 'c1', 'name': 'C', 'identity': 'person',
                                             'looks': {'eyes': 'blue'}}])
        job, client, _ = self.finish_job([])
        self.assertEqual(job.status, 'complete', job.detail)
        eyes = client.graphs[-1]
        said = eyes[eyes["fc1_4"]["inputs"]["positive"][0]]["inputs"]["text"]
        self.assertIn("round blue irises", said)
        self.assertIn("Eye pass told the eye colour: Person blue.", job.record["notes"])
        # No colour known: the eye pass as it was, and nothing said of it.
        self.studio.lib.save('characters', [])
        job, client, _ = self.finish_job([])
        eyes = client.graphs[-1]
        said = eyes[eyes["fc1_4"]["inputs"]["positive"][0]]["inputs"]["text"]
        self.assertIn(ig.EYE_WHAT, said)
        self.assertFalse(any(n.startswith("Eye pass told") for n in job.record["notes"]))

    def test_the_hands_are_redrawn_after_the_eyes_and_before_the_glasses(self):
        job, client, _ = self.finish_job([(460, 272, 60, 22)],
                                         hands=[(200, 600, 70, 80), (700, 620, 60, 70)])
        self.assertEqual(job.status, 'complete', job.detail)
        find, eyes, hands, glasses = client.graphs[-4:]
        self.assertEqual([find["p%dt" % i]["inputs"]["text"] for i in range(3)],
                         ig.FINISH_FIND + [ig.HAND_FIND])
        self.assertEqual(eyes["fc1_s0"]["inputs"]["text"], ig.EYE_WORD)
        # Both hands in one run, each its own crop, only SAM3's hand in it redrawn.
        self.assertEqual([hands["fc%d_s0" % i]["inputs"]["text"] for i in (1, 2)],
                         ["hand", "hand"])
        self.assertNotIn("fc3_1", hands)
        self.assertEqual(hands["fc1_4"]["inputs"]["denoise"], ig.HAND_DENOISE)
        self.assertIn("four fingers and a thumb",
                      hands[hands["fc1_4"]["inputs"]["positive"][0]]["inputs"]["text"])
        self.assertEqual(hands["fi"]["inputs"]["image"], "ImageStudio/faces_00001_.png [output]")
        self.assertNotIn("fc1_t", hands)            # no tone node on this ComfyUI
        self.assertEqual(glasses["fc1_s0"]["inputs"]["text"], "glasses")
        self.assertIn("Hands pass: 2 hands redrawn, denoise %s." % ig.HAND_DENOISE,
                      job.record["notes"])

    def test_a_picture_without_a_face_swap_still_gets_the_hands_pass(self):
        job, client, order = self.finish_job([(460, 272, 60, 22)], hands=[(200, 600, 70, 80)],
                                             profile=False, tone=True)
        self.assertEqual(job.status, 'complete', job.detail)
        self.assertEqual(order, [])
        find, hands = client.graphs[-2:]
        # Only the hands are looked for: no eyes or glasses without a swap.
        self.assertEqual(find["p0t"]["inputs"]["text"], ig.HAND_FIND)
        self.assertNotIn("p1t", find)
        self.assertEqual(hands["fc1_s0"]["inputs"]["text"], "hand")
        self.assertEqual(hands["fc1_t"]["inputs"]["amount"], ig.FIX_TONE)   # keeps the grade
        self.assertIn("Hands pass: 1 hand redrawn, denoise %s." % ig.HAND_DENOISE,
                      job.record["notes"])

    def test_each_pass_graph_is_kept_in_the_record_for_the_nodes_view(self):
        import apps.comfyui.view as comfy_view
        job, client, _ = self.finish_job([(460, 272, 60, 22)], hands=[(200, 600, 70, 80)])
        self.assertEqual(job.status, 'complete', job.detail)
        eyes, hands, glasses = client.graphs[-3:]
        self.assertEqual([p["label"] for p in job.record["passes"]],
                         ["Eye pass", "Hands", "Glasses"])
        self.assertEqual([p["graph"] for p in job.record["passes"]], [eyes, hands, glasses])
        self.assertEqual([label for label, _g in comfy_view.graph_steps(job.record)],
                         ["Picture", "Eye pass", "Hands", "Glasses"])

    def fail_hands(self, failure, fetch_fails=False):
        """finish_job with eyes and hands, where the hands pass raises
        `failure` (after it is listed, as `_run_pass` lists a run); with
        `fetch_fails`, the eye pass's result cannot be fetched either."""
        real = ig.Studio._run_pass

        def run_pass(studio, job, client, graph, say, label, status="refining"):
            if label != "Hands":
                return real(studio, job, client, graph, say, label, status=status)
            job.passes.append({"label": label, "graph": graph})
            if fetch_fails:
                client.fetch = Mock(side_effect=ig.ComfyError("ComfyUI went away"))
            raise failure
        with patch.object(ig.Studio, "_run_pass", run_pass), \
                patch.object(ig.doctor, "log_error") as logged:
            job, client, _ = self.finish_job([], hands=[(200, 600, 70, 80)])
        return job, logged

    def test_a_pass_failing_after_the_eyes_keeps_the_eyes_and_says_only_what_was_done(self):
        # Idea 3b7e25aaa3b1: the eye pass done, the hands erroring, the
        # untouched picture came back with the "Eye pass" note and graph kept.
        job, logged = self.fail_hands(ig.ComfyError("boom"))
        self.assertEqual(job.status, 'complete', job.detail)
        notes = job.record["notes"]
        self.assertTrue(any(n.startswith("Eye pass after the face swap") for n in notes), notes)
        self.assertFalse(any(n.startswith("Hands pass:") for n in notes), notes)
        self.assertTrue(any(n.endswith("pass after the face swap failed (boom); the passes "
                                       "finished before it are kept.") for n in notes), notes)
        self.assertEqual([p["label"] for p in job.record["passes"]], ["Eye pass"])
        self.assertFalse(logged.called)

    def test_a_pass_failing_whose_earlier_result_is_gone_drops_its_notes(self):
        job, _ = self.fail_hands(ig.ComfyError("boom"), fetch_fails=True)
        self.assertEqual(job.status, 'complete', job.detail)
        notes = job.record["notes"]
        self.assertFalse(any(n.startswith("Eye pass after the face swap") for n in notes), notes)
        self.assertTrue(any(n.endswith("pass after the face swap failed (boom); the picture "
                                       "is kept as it was before it.") for n in notes), notes)
        self.assertEqual(job.record["passes"], [])

    def test_an_unexpected_error_in_a_pass_still_saves_the_picture(self):
        # Idea 442b63523b89: only ComfyError/TemplateError/OSError were
        # caught; a KeyError failed the job and skipped History.
        job, logged = self.fail_hands(KeyError("parts"))
        self.assertEqual(job.status, 'complete', job.detail)
        self.assertEqual(len(self.studio.history.list()), 1)
        self.assertNotIn('finish', self.studio.history.list()[0])     # no checkpoint left pending
        self.assertTrue(any("failed (" in n for n in job.record["notes"]), job.record["notes"])
        self.assertIn("Traceback", logged.call_args.args[0])
        self.assertIn("KeyError", logged.call_args.args[0])

    def test_a_swap_that_went_behind_the_glasses_leaves_them_as_drawn(self):
        # Live, 2026-09-29: with FaceFusion's occlusion mask the frames are the
        # picture's own, to the pixel. Redrawn all the same they came back as
        # other glasses, and the likeness fell from 0.81 to 0.73.
        job, client, order = self.finish_job([(460, 272, 60, 22)],
                                             hands=[(200, 600, 70, 80)],
                                             masks=list(ff.SWAP_MASKS))
        self.assertEqual(job.status, 'complete', job.detail)
        self.assertIn("occlusion", ff.SWAP_MASKS)
        find, eyes, hands = client.graphs[-3:]
        self.assertEqual([find[k]["inputs"]["text"] for k in ("p0t", "p1t")],
                         ["face:8", ig.HAND_FIND])          # the glasses are not looked for
        self.assertNotIn("p2t", find)
        self.assertEqual([p["label"] for p in job.record["passes"]], ["Eye pass", "Hands"])
        self.assertTrue(any(n.startswith("No glasses pass") for n in job.record["notes"]),
                        job.record["notes"])
        self.assertNotIn(("glasses", "Glasses"),
                         ig.pipeline_stages(self.studio.lib, job.settings))
        # Asked for, they are redrawn whatever the swap kept; refused, never.
        job, client, _ = self.finish_job([(460, 272, 60, 22)], masks=list(ff.SWAP_MASKS),
                                         glasses_pass=True)
        self.assertEqual([p["label"] for p in job.record["passes"]], ["Eye pass", "Glasses"])
        self.assertIn(("glasses", "Glasses"), ig.pipeline_stages(self.studio.lib, job.settings))
        job, client, _ = self.finish_job([(460, 272, 60, 22)], glasses_pass=False)
        self.assertEqual([p["label"] for p in job.record["passes"]], ["Eye pass"])
        self.assertFalse(any("glasses" in n.lower() for n in job.record["notes"]),
                         job.record["notes"])

    def test_a_swap_the_face_enhancer_sharpened_gets_no_eye_pass(self):
        # The eye pass is there for the soft eyes a swap leaves. After a face
        # enhancer it cost likeness for eyes no sharper (live, 2026-09-30).
        self.assertTrue(ig.eye_pass([{'masks': ['box']}]))
        self.assertFalse(ig.eye_pass([{'enhance': 'gfpgan_1.4'}]))
        self.assertTrue(ig.eye_pass([{'enhance': 'gfpgan_1.4'}, {'enhance': None}]))
        job, client, _ = self.finish_job([(460, 272, 60, 22)], hands=[(200, 600, 70, 80)],
                                         masks=list(ff.SWAP_MASKS), enhance='gfpgan_1.4')
        self.assertEqual(job.status, 'complete', job.detail)
        find, hands = client.graphs[-2:]
        self.assertEqual(find["p0t"]["inputs"]["text"], ig.HAND_FIND)   # no face is looked for
        self.assertNotIn("p1t", find)
        self.assertEqual([p["label"] for p in job.record["passes"]], ["Hands"])
        self.assertTrue(any(n.startswith("No eye pass: the face enhancer")
                            for n in job.record["notes"]), job.record["notes"])
        # Nothing left to do at the end: nothing is looked for at all.
        job, client, _ = self.finish_job([], masks=list(ff.SWAP_MASKS), enhance='gfpgan_1.4',
                                         hand_pass=False)
        self.assertEqual(job.status, 'complete', job.detail)
        self.assertFalse(any("p0d" in g for g in client.graphs))
        self.assertEqual(job.record.get("passes") or [], [])
        # Glasses asked for: the face is found for them alone.
        job, client, _ = self.finish_job([(460, 272, 60, 22)], masks=list(ff.SWAP_MASKS),
                                         enhance='gfpgan_1.4', glasses_pass=True,
                                         hand_pass=False)
        self.assertEqual([p["label"] for p in job.record["passes"]], ["Glasses"])
        self.assertEqual([client.graphs[-2][k]["inputs"]["text"] for k in ("p0t", "p1t")],
                         ig.FINISH_FIND)
        # The strip shows the pass only when it will run.
        self.assertIn(("eyes", "Eye pass"), ig.pipeline_stages(self.studio.lib, job.settings))
        with patch.object(ff, 'enhancer', return_value='gfpgan_1.4'):
            self.assertNotIn(("eyes", "Eye pass"),
                             ig.pipeline_stages(self.studio.lib, job.settings))

    def test_the_hands_pass_takes_sam3s_surest_hands_and_leaves_them_as_they_were(self):
        # What SAM3 said of a carpenter at his bench (live, 2026-09-29): his two
        # hands, the forearm round each, and a thing a few pixels wide. Taken
        # biggest first the forearms won, and the hands in them were dropped
        # as their copies.
        hands = [(561, 655, 188, 187, 0.86), (316, 619, 165, 180, 0.83),
                 (448, 879, 30, 18, 0.65), (261, 534, 224, 268, 0.46),
                 (559, 557, 291, 287, 0.33)]
        job, client, _ = self.finish_job([], hands=hands, profile=False)
        self.assertEqual(job.status, 'complete', job.detail)
        graph = client.graphs[-1]
        self.assertIn("fc2_1", graph)
        self.assertNotIn("fc3_1", graph)
        areas = [graph["fc%d_a1" % i]["inputs"] for i in (1, 2)]
        crops = [graph["fc%d_1" % i]["inputs"]["crop_region"] for i in (1, 2)]
        for area, crop, (x, y, w, h, _s) in zip(areas, crops, hands[:2]):
            # The part redrawn is the hand's own box, grown, at the redraw's size.
            k = ig.FACE_EDIT / float(crop["width"])
            self.assertAlmostEqual(area["width"] / k, w * (1 + 2 * ig.FIX_AREA_GROW), delta=3)
        self.assertIn("Hands pass: 2 hands redrawn, denoise 0.4.", job.record["notes"])
        # A finish: gentle enough to leave a good hand as it was (0.6 took a
        # ring off, 0.4 did not), and sampled as a redraw is on this model.
        self.assertEqual(ig.HAND_DENOISE, 0.4)
        self.assertEqual(graph["fc1_4"]["inputs"]["denoise"], 0.4)
        self.assertEqual(graph["fc1_4"]["inputs"]["sampler_name"], "euler_ancestral")
        self.assertEqual(graph["fc1_4"]["inputs"]["scheduler"], "beta")
        self.assertEqual(graph["fc1_4"]["inputs"]["steps"], 8)
        # The picture itself is sampled as ever.
        first = client.graphs[0]
        self.assertEqual(first["40"]["inputs"]["sampler_name"], "res_multistep")

    def test_real_hands(self):
        hand, again, arm = (100, 100, 80, 90, "hand", 0.9), (104, 98, 84, 92, "hand", 0.7), \
            (60, 40, 200, 240, "hand", 0.55)
        far, unsure = (700, 600, 30, 36, "hand", 0.92), (500, 500, 90, 90, "hand", 0.45)
        kept = ig.real_hands(1024, 1024, [arm, unsure, again, far, hand])
        self.assertEqual(kept, [hand])
        # A hand beside another is its own; on a bigger picture a small one is smaller.
        other = (400, 100, 80, 90, "hand", 0.8)
        self.assertEqual(ig.real_hands(1024, 1024, [other, hand]), [hand, other])
        self.assertEqual(ig.real_hands(600, 600, [far]), [far])
        self.assertEqual(ig.real_hands(1024, 1024, []), [])

    def test_a_picture_of_no_one_gets_no_hands_pass(self):
        # SAM3 scored a red fox's paws 0.87 as hands, and the pass redrew
        # them as "a human hand with four fingers and a thumb".
        job, client, _ = self.finish_job([], hands=[(380, 900, 74, 51, 0.87)], profile=False,
                                         scene="A red fox sitting in fresh snow at dawn")
        self.assertEqual(job.status, 'complete', job.detail)
        self.assertFalse(any("p0d" in g for g in client.graphs))      # SAM3 is not asked
        self.assertEqual(job.record["passes"], [])
        self.assertIn("No hands pass: the picture's words name no person.",
                      job.record["notes"])
        self.assertNotIn(("hands", "Hand pass"),
                         ig.pipeline_stages(self.studio.lib, job.settings))
        self.assertIn(("hands", "Hand pass"), ig.pipeline_stages(
            self.studio.lib, dict(job.settings, scene="A man feeding a fox")))
        # A person is known by their trade too, and a thing is not a person.
        for scene in ("A chef plating a dish in a kitchen", "Two climbers on a ridge",
                      "A bride and groom on the steps", "Grandmother's hands kneading dough",
                      "The pianist at a grand piano"):
            self.assertTrue(ig.hand_pass({"scene": scene}), scene)
        for scene in ("A red bicycle against a white wall", "A lighthouse in a storm",
                      "A bowl of ramen on a wooden table", "Snow on a mountain pass"):
            self.assertFalse(ig.hand_pass({"scene": scene}), scene)
        self.assertTrue(ig.hand_pass({"scene": "A lighthouse", "identities": ["person"]}))
        self.assertTrue(ig.hand_pass({"scene": "A lighthouse", "subject": "a sailor"}))
        self.assertFalse(ig.hand_pass({"scene": "A chef", "hand_pass": False}))

    def test_the_hands_pass_can_be_turned_off(self):
        job, client, _ = self.finish_job([], hands=[(200, 600, 70, 80)], profile=False,
                                         hand_pass=False)
        self.assertEqual(job.status, 'complete', job.detail)
        self.assertFalse(any("p0d" in g for g in client.graphs))
        self.assertFalse(any("hands pass" in n.lower() for n in job.record["notes"]),
                         job.record["notes"])
        # With a swap, the eyes and glasses are still done, and no hands.
        job, client, _ = self.finish_job([(460, 272, 60, 22)], hands=[(200, 600, 70, 80)],
                                         hand_pass=False)
        find, eyes, glasses = client.graphs[-3:]
        self.assertNotIn("p2t", find)
        self.assertEqual(glasses["fc1_s0"]["inputs"]["text"], "glasses")

    def beard_region(self, box=(115, 75, 16, 10), side=256, name="beard_mask.png"):
        """A scene's beard as Scene Builder sends it: words and a side x side
        mask, white over `box` (x, y, w, h in the mask's pixels). The finder's
        face (450, 250, 80, 90) on 1024 is (112, 62, 20, 22) at 256."""
        import core.icons as studio_icons
        x, y, w, h = box
        px = bytearray(side * side * 4)
        for yy in range(side):
            for xx in range(side):
                v = 255 if x <= xx < x + w and y <= yy < y + h else 0
                px[(yy * side + xx) * 4:(yy * side + xx) * 4 + 4] = bytes((v, v, v, 255))
        path = os.path.join(self.dir, name)
        Path(path).write_bytes(studio_icons.png(bytes(px), side, side))
        return {"prompt": "closely cropped dense auburn short beard, high cheek coverage",
                "mask_path": path, "person_id": "p1", "kind": "facial_hair"}

    def test_a_scene_beard_is_redrawn_before_the_face_swap_inside_its_mask(self):
        region = self.beard_region()
        job, client, order = self.finish_job([(460, 272, 60, 22)], hands=[(200, 600, 70, 80)],
                                             character_regions=[region])
        self.assertEqual(job.status, 'complete', job.detail)
        self.assertEqual(order, ['swap'])
        first, beard, find, eyes, hands, glasses = client.graphs[-6:]
        # The beard pass is done when FaceFusion swaps (the user, 2026-10-02).
        self.assertEqual(self.graphs_at_swap, [len(client.graphs) - 4])
        self.assertEqual(first["p0t"]["inputs"]["text"], "face:8")
        self.assertEqual([k for k in first if k.endswith("t") and k.startswith("p")], ["p0t"])
        self.assertEqual(find["p0t"]["inputs"]["text"], "face:8")
        self.assertEqual(eyes["fc1_s0"]["inputs"]["text"], ig.EYE_WORD)
        # The beard on the picture before the swap, redrawn only inside the scene's mask.
        self.assertEqual(beard["fi"]["inputs"]["image"], first["1"]["inputs"]["image"])
        self.assertTrue(beard["fc1_a0"]["inputs"]["image"].startswith("studio_%s_0_beard0"
                                                                      % job.id))
        self.assertNotIn("fc1_s0", beard)           # the mask is the scene's, not SAM3's
        self.assertEqual(beard["fc1_4"]["inputs"]["denoise"], ig.BEARD_DENOISE)
        said = beard[beard["fc1_4"]["inputs"]["positive"][0]]["inputs"]["text"]
        self.assertIn("auburn short beard", said)
        region_px = beard["fc1_1"]["inputs"]["crop_region"]
        self.assertLessEqual(region_px["x"], 460)               # the mask's box, 460..524 x 300..340
        self.assertGreaterEqual(region_px["x"] + region_px["width"], 524)
        self.assertEqual(hands["fc1_s0"]["inputs"]["text"], "hand")
        self.assertEqual(glasses["fc1_s0"]["inputs"]["text"], "glasses")
        self.assertEqual([p["label"] for p in job.record["passes"]],
                         ["Beard", "Eye pass", "Hands", "Glasses"])
        self.assertIn("Beard pass before the face swap: 1 beard redrawn in its mask, "
                      "denoise %s." % ig.BEARD_DENOISE, job.record["notes"])
        stages = [k for k, _ in ig.pipeline_stages(self.studio.lib, job.settings)]
        self.assertLess(stages.index("beard"), stages.index("face_swap"))
        self.assertLess(stages.index("face_swap"), stages.index("eyes"))

    def test_a_scene_beard_without_a_face_swap_and_its_switch(self):
        region = self.beard_region()
        job, client, order = self.finish_job([], profile=False, scene="A bearded sailor",
                                             hand_pass=False, character_regions=[region])
        self.assertEqual(job.status, 'complete', job.detail)
        find, beard = client.graphs[-2:]
        self.assertEqual(find["p0t"]["inputs"]["text"], "face:8")
        self.assertNotIn("p1t", find)
        self.assertEqual([p["label"] for p in job.record["passes"]], ["Beard"])
        # Off, nothing is looked for; no beard chosen, no pass.
        job, client, _ = self.finish_job([], profile=False, scene="A bearded sailor",
                                         hand_pass=False, beard_pass=False,
                                         character_regions=[region])
        self.assertFalse(any("p0d" in g for g in client.graphs))
        self.assertEqual(ig.beard_regions({"character_regions": [
            dict(region, kind=None), dict(region, mask_path="gone.png")]}), [])

    def test_a_beard_off_the_drawn_face_is_left_and_said(self):
        mask = self.beard_region(box=(10, 200, 16, 10))   # the scene put the head elsewhere
        spots, off, said = ig.beard_spots(1024, 1024, [mask], [(450, 250, 80, 90)])
        self.assertEqual((spots, off, said), ([], 1, []))
        on = self.beard_region(name="on.png")
        (spot,), off, _ = ig.beard_spots(1024, 1024, [on], [(450, 250, 80, 90)])
        self.assertEqual(off, 0)
        self.assertEqual(spot["box"], [460, 300, 64, 40])
        self.assertEqual(spot["size"], int(64 * ig.BEARD_PAD))
        # The mask picture over the crop: white at the beard, black round it.
        crop = ig.fix_crops(1024, 1024, [spot])[0]
        rgba, w, h = __import__("core.icons", fromlist=["x"]).png_to_rgba(
            ig.beard_shape_png(spot, crop, 1024, 1024))
        self.assertEqual((w, h), (crop["width"], crop["height"]))
        mid = ((h // 2) * w + w // 2) * 4
        self.assertEqual(rgba[mid], 255)
        self.assertEqual(rgba[0], 0)
        job, client, _ = self.finish_job([], profile=False, scene="A bearded sailor",
                                         hand_pass=False, character_regions=[mask])
        self.assertEqual(job.status, 'complete', job.detail)
        self.assertEqual(job.record["passes"], [])
        self.assertIn("1 beard not redrawn: the face was not drawn where the scene put it.",
                      job.record["notes"])

    @staticmethod
    def face(unit=22.0, at=(490.0, 255.0)):
        """68 face points (iBUG, the scene's own template) in picture px:
        `unit` px per half eye gap, the eyes' middle `at`."""
        from apps.image_studio.scene import pose
        return [[at[0] + u * unit, at[1] + v * unit] for u, v in pose.FACE]

    def zone_mask(self, look, unit=22.0, at=(490.0, 255.0)):
        """A beard's landmark zones alone on a 1024 picture (no scene mask):
        -> a function telling whether the template point (u, v) is beard."""
        p = [tuple(q) for q in self.face(unit, at)]
        zones = ig.beard_zones(p, dict({"style": "short", "length": 0.15, "coverage": 0.8},
                                       **look))
        bufs = {}
        for k, polys in zones.items():
            bufs[k] = bytearray(1024 * 1024)
            for poly in polys:
                ig.fill_polygon(bufs[k], 1024, 1024, poly)

        def beard(u, v):
            i = int(at[1] + v * unit) * 1024 + int(at[0] + u * unit)
            return bool(bufs["fill"][i] and not bufs["clear"][i])
        return beard

    def test_the_drawn_face_s_landmarks_bound_a_beard(self):
        beard = self.zone_mask({})
        self.assertFalse(beard(0, 2.0))          # lips: kept clear
        self.assertTrue(beard(0, 1.42))          # upper lip: the moustache's place
        self.assertFalse(beard(0, 0.6))          # nose: above the moustache's upper edge
        self.assertTrue(beard(-1.5, 2.2))        # cheek below the upper beard line
        self.assertFalse(beard(-1.5, 0.9))       # cheek above it
        self.assertTrue(beard(1.5, 2.2))
        self.assertTrue(beard(0, 3.3))           # just under the chin, a short beard's length
        self.assertFalse(beard(0, 3.8))          # past it
        self.assertTrue(self.zone_mask({"length": 1.0})(0, 3.8))   # a long one reaches
        low = self.zone_mask({"coverage": 0.0})
        self.assertFalse(low(-1.5, 2.0))         # low cheek coverage: the line drops
        self.assertTrue(low(0, 2.9))             # the chin keeps it
        tache = self.zone_mask({"style": "moustache"})
        self.assertTrue(tache(0, 1.42))
        self.assertFalse(tache(-1.5, 2.2))
        self.assertFalse(tache(0, 2.9))
        goatee = self.zone_mask({"style": "goatee"})
        self.assertTrue(goatee(0, 2.9))
        self.assertTrue(goatee(0, 1.42))
        self.assertFalse(goatee(-1.5, 2.2))
        self.assertFalse(goatee(0, 2.0))
        # A cheek turned away is trimmed, never filled.
        p = [tuple(q) for q in self.face()]
        one = ig.beard_zones(p, {"style": "full", "length": 0.3, "coverage": 0.8},
                             (True, False))
        self.assertEqual(len(one["fill"]), 2)    # the upper lip and the near cheek

    def landmark_region(self, dots_at=(490.0, 255.0), **look):
        region = self.beard_region()
        region["face_dots"] = [[x / 1024.0, y / 1024.0] for x, y in self.face(at=dots_at)]
        region["beard"] = dict({"style": "short", "length": 0.15, "coverage": 0.8,
                                "density": 0.7, "color": ""}, **look)
        return region

    def test_the_scene_s_mask_is_fitted_to_the_drawn_face(self):
        drawn = [{"points": self.face(at=(530.0, 275.0)), "box": [0, 0, 1, 1]}]
        # The scene put the face 40 px left of and 20 px above where it was drawn.
        (spot,), off, said = ig.beard_spots(1024, 1024, [self.landmark_region()],
                                            [(450, 250, 80, 90)], drawn)
        self.assertEqual(off, 0)
        self.assertEqual(len(said), 1)
        self.assertTrue(said[0].startswith("Beard 1 fitted to the drawn face's landmarks: "
                                           "the scene's mask moved 45 px"), said[0])
        grey, mw, mh = spot["mask"]
        at = lambda u, v: grey[int((275 + v * 22) / 4) * mw + int((530 + u * 22) / 4)]  # noqa
        self.assertEqual(at(-0.9, 2.7), 255)     # the drawn face's cheek, not the scene's
        self.assertEqual(at(0, 2.0), 0)          # its lips
        # A drawn face twice the scene's: the zones alone, said so.
        big = [{"points": self.face(unit=44.0, at=(490.0, 230.0)), "box": [0, 0, 1, 1]}]
        _, _, said = ig.beard_spots(1024, 1024, [self.landmark_region()],
                                    [(400, 200, 180, 200)], big)
        self.assertIn("landmarks alone", said[0])
        # No drawn face near it: the scene's mask as it came.
        far = [{"points": self.face(at=(150.0, 255.0)), "box": [0, 0, 1, 1]}]
        (spot,), _, said = ig.beard_spots(1024, 1024, [self.landmark_region()],
                                          [(450, 250, 80, 90)], far)
        self.assertEqual(spot["box"], [460, 300, 64, 40])
        self.assertIn("kept as the scene made it", said[0])
        # Without the scene's face dots, or without the reader, nothing is fitted.
        _, _, said = ig.beard_spots(1024, 1024, [self.beard_region()], [(450, 250, 80, 90)],
                                    drawn)
        self.assertEqual(said, [])

    def test_the_beard_pass_reads_the_drawn_face_s_landmarks(self):
        region = self.landmark_region()
        drawn = [{"points": self.face(), "box": [444, 245, 536, 330], "yaw": 0, "score": 0.9}]
        job, client, _ = self.finish_job([], profile=False, scene="A bearded sailor",
                                         hand_pass=False, character_regions=[region],
                                         drawn=drawn)
        self.assertEqual(job.status, 'complete', job.detail)
        find, read, beard = client.graphs[-3:]
        self.assertEqual(read["ll"]["class_type"], ig.LANDMARK_NODE)
        self.assertEqual(json.loads(read["ll"]["inputs"]["boxes"]), [[450, 250, 80, 90]])
        self.assertEqual(read["li"]["inputs"]["image"], find["1"]["inputs"]["image"])
        self.assertTrue(any(n.startswith("Beard 1 fitted to the drawn face's landmarks: the "
                                         "scene's mask moved 0 px") for n in job.record["notes"]),
                        job.record["notes"])
        self.assertEqual([p["label"] for p in job.record["passes"]], ["Beard"])
        # A ComfyUI without the reader: the beard pass as before, and said.
        job, client, _ = self.finish_job([], profile=False, scene="A bearded sailor",
                                         hand_pass=False, character_regions=[region])
        self.assertEqual(job.status, 'complete', job.detail)
        self.assertFalse(any("ll" in g for g in client.graphs))
        self.assertTrue(any("lacks %s" % ig.LANDMARK_NODE in n for n in job.record["notes"]))
        self.assertEqual([p["label"] for p in job.record["passes"]], ["Beard"])

    def test_without_sam3_the_swap_is_kept_as_it_is(self):
        self.profile()
        with patch.object(ff, 'available', return_value=True), \
                patch.object(ff, 'swap', return_value=(PNG, {'outside_mask_changed_pixels': 0})):
            jobs = self.studio.submit(self.settings())
            settle(jobs)
        self.assertEqual(jobs[0].status, 'complete', jobs[0].detail)
        self.assertEqual(len(FakeClient.instances[-1].graphs), 1)
        self.assertTrue(any(n.startswith("No eye, hands or glasses pass after the face swap")
                            for n in jobs[0].record["notes"]))

    def test_face_only_fix_never_checks_comfyui(self):
        path = self.profile()
        self.studio.lib.save('backends', [])
        settings = {'mode': 'fix', 'backend': 'auto', 'fix': {'image': path,
                    'face_swap': 'person', 'face_point': [.25, .4]}}
        with patch.object(ff, 'available', return_value=True), patch.object(ff, 'swap', return_value=(PNG, {})) as swap, \
                patch.object(self.studio, 'client', side_effect=AssertionError('No backend needed')):
            jobs = self.studio.submit(settings)
            settle(jobs)
        self.assertEqual(jobs[0].status, 'complete', jobs[0].detail)
        self.assertEqual(swap.call_args.args[1]['target_point'], [.25, .4])

    def test_missing_references_and_runtime_are_in_preview(self):
        self.profile()
        self.studio.lib.get('identities', 'person')['references'] = []
        with patch.object(ff, 'available', return_value=False):
            errors = self.studio.preview(self.settings(), self.backend('5090')).errors
        self.assertTrue(any('FaceFusion' in e for e in errors))
        self.assertTrue(any('Image references' in e and 'reference' in e for e in errors))

    def test_cancel_returns_while_backend_interrupt_is_blocked(self):
        entered, release, returned = threading.Event(), threading.Event(), threading.Event()
        def interrupt(_):
            entered.set()
            release.wait(5)
        job = ig.Job({}, self.backend('5090'))
        job.prompt_id = 'prompt'
        job.status = 'sampling'
        with patch.object(self.studio, 'client', return_value=SimpleNamespace(cancel_job=interrupt)):
            worker = threading.Thread(target=lambda: (self.studio.queue.cancel(job), returned.set()))
            worker.start()
            try:
                self.assertTrue(entered.wait(1))
                self.assertTrue(returned.wait(1), 'Cancel blocked on the backend')
                self.assertTrue(job.cancel.is_set())
            finally:
                release.set()
                worker.join(2)


class TestSceneClose(unittest.TestCase):
    def test_retry_tick_cancels_its_pending_timer_and_never_starts_when_closing(self):
        from core.chat import Chat
        app = SimpleNamespace(closing=True, _stand_down=Mock(), _spawn=Mock())
        Chat._retry_host(app)
        app._stand_down.assert_called_once_with('host_timer')
        app._spawn.assert_not_called()

    def test_failed_save_and_cancelled_save_as_veto_final_close(self):
        from apps.image_studio.scene.ui import SceneBuilder
        sb = SceneBuilder.__new__(SceneBuilder)
        sb.win = Mock()
        sb.dirty = True
        sb.has_content = lambda: True
        sb._save_recovery = Mock()
        sb.save = Mock(return_value=False)
        with patch('apps.image_studio.scene.ui.messagebox.askyesnocancel', return_value=True):
            self.assertFalse(sb.close(final=True))
        sb.win.destroy.assert_not_called()

    def test_quit_and_tab_close_leave_state_intact_after_veto(self):
        from core.chat import Chat
        session = SimpleNamespace(images=SimpleNamespace(can_close=lambda: False))
        app = SimpleNamespace(closing=False, sessions={'images': session}, active='images')
        Chat._quit(app)
        self.assertFalse(app.closing)
        Chat._close_tab(app)
        self.assertIs(app.sessions['images'], session)

    def test_recovery_copy_is_valid_and_leaves_original_untouched(self):
        from apps.image_studio.scene.ui import SceneBuilder
        with tempfile.TemporaryDirectory() as folder:
            sb = SceneBuilder.__new__(SceneBuilder)
            sb.scene = sc.new_scene(details='A recovered room')
            sb.dirty = True
            sb.has_content = lambda: True
            sb.recovery_path = os.path.join(folder, 'recovery.scene.json')
            sb.status = Mock()
            sb._save_recovery()
            restored, problems = sc.load(sb.recovery_path)
            self.assertEqual(restored['details'], 'A recovered room')
            self.assertEqual(problems, [])


if __name__ == '__main__':
    unittest.main()
