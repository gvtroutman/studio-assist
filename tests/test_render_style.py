"""Chosen media and styles survive crop finishing; no models or network."""
import copy
import os
from pathlib import Path
import unittest

import apps.image_studio.imagegen as ig
from test_imagegen import FLUX_FILES, PNG, FaceClient, TempStudioMixin, settle
import test_finish_line as finish_fixture


def encoded_words(graph):
    return [node["inputs"]["text"] for node in graph.values()
            if node["class_type"] == "CLIPTextEncode"]


class TestDetailWords(unittest.TestCase):
    def test_unstyled_photographs_keep_the_measured_prompts(self):
        for words in ("a face", "a hand", "clear glasses", "a natural smile"):
            self.assertEqual(ig.detail_prompt(words), ig.FIX_PROMPT % words)
            self.assertEqual(ig.detail_prompt(words, {}, face=True), ig.FACE_PROMPT % words)
        self.assertEqual(ig.detail_prompt("A garden.", whole=True), "A garden.")

    def test_only_explicit_picture_medium_enters_a_crop(self):
        cases = {
            "A watercolor illustration of a woman beside a yellow submarine.":
                "watercolor illustration",
            "Draw an oil painting of a sailor beside a yellow submarine.": "oil painting",
            "Pixel art: a sailor beside a yellow submarine.": "Pixel art",
            "A woman beside a yellow submarine, in the style of an ink drawing.": "ink drawing",
            "A woman beside a yellow submarine, watercolor painting style.": "watercolor painting",
            "A photograph of a woman holding an oil painting of a submarine.": "",
            "An oil painting hangs behind a woman in a photograph.": "",
            "A woman wearing a T-shirt with a watercolor illustration of a submarine.": "",
            "A woman holding a camera in a gallery.": "",
        }
        for scene, medium in cases.items():
            with self.subTest(scene=scene):
                style = ig.detail_style("", scene)
                self.assertEqual(style, medium)
                words = ig.detail_prompt("a hand", {"detail_style": style})
                self.assertNotIn("submarine", words)
                if medium:
                    self.assertIn(medium, words)
                    self.assertNotIn("photo detail", words)

    def test_style_context_is_bounded_and_added_once(self):
        style = "Editorial illustration, clean ink linework, muted colour."
        values = {"detail_style": style}
        words = ig.detail_prompt("a hand. " + style, values)
        self.assertEqual(words.count("Editorial illustration"), 1)
        long = ig.detail_style("soft grain " * 200)
        self.assertLessEqual(len(long), ig.DETAIL_STYLE_CHARS)
        self.assertFalse(long.endswith(" "))
        self.assertEqual(ig.detail_style(style, "An oil painting of a woman."), style)


class TestComposedStyle(TempStudioMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.addCleanup(self.studio.close)

    def plan(self, inventory=None, **changes):
        settings = dict(ig.default_settings(), model="flux-dev", backend="5090",
                        subject="a woman", scene="Beside a yellow submarine.",
                        face_detail=True, hand_pass=False, seed=5)
        settings.update(changes)
        return ig.compose(settings, self.studio.lib, self.backend("5090"), inventory)

    def test_chosen_illustration_monochrome_and_analog_keep_their_look(self):
        for style in ("illustration", "black-and-white", "sx-70", "modern-photo"):
            with self.subTest(style=style):
                plan = self.plan(style=style, identities=["sitter"])
                self.assertEqual(plan.errors, [])
                selected = self.studio.lib.get("styles", style)["prompt"]
                self.assertEqual(plan.values["detail_style"], selected.rstrip(" ."))
                self.assertIn("yellow submarine", plan.prompt)
                self.assertIn("SITTERPERSON", plan.values["face_prompt"])
                self.assertNotIn("yellow submarine", plan.values["face_prompt"])
                self.assertIn(selected.rstrip(" ."), plan.values["face_prompt"])
                self.assertNotIn("fine pores", plan.values["face_prompt"])
                self.assertNotIn("real human face", plan.values["face_prompt"])

    def test_no_style_face_prompt_stays_identical(self):
        plan = self.plan()
        self.assertEqual(plan.values["detail_style"], "")
        self.assertEqual(plan.values["face_prompt"], ig.FACE_PROMPT % plan.prompt)

    def test_explicit_scene_medium_without_a_style_survives_finishing(self):
        plan = self.plan(style="none", scene="An oil painting of a woman beside a yellow submarine.")
        self.assertEqual(plan.values["detail_style"], "oil painting")
        self.assertIn("oil painting", plan.values["face_prompt"])
        self.assertNotIn("yellow submarine", plan.values["face_prompt"])

    def test_camera_chemistry_survives_with_no_selected_style(self):
        for camera, phrase in (("canon-ae1", "black and white"), ("sx-70", "warm, faded colour")):
            with self.subTest(camera=camera):
                plan = self.plan(style="none", camera_profile=camera)
                self.assertEqual(plan.errors, [])
                self.assertIn(phrase, plan.values["detail_style"])
                self.assertIn(phrase, plan.values["face_prompt"])
                self.assertNotIn("yellow submarine", plan.values["face_prompt"])
                self.assertNotIn("fine pores", plan.values["face_prompt"])

    def test_scene_camera_uses_its_saved_chemistry(self):
        plan = self.plan(style="none", camera_profile="digital-5d", scene_layout={
            "objects": [], "camera": {"profile": "deleted-camera",
                "chemistry": "Black and white instant film, lifted blacks and soft grain."}})
        self.assertIn("Black and white instant film", plan.values["face_prompt"])
        self.assertNotIn("Canon", plan.values["face_prompt"])

    def test_a_malformed_saved_camera_cannot_break_composing(self):
        for camera in (None, "old-camera", ["old-camera"], 1):
            with self.subTest(camera=camera):
                plan = self.plan(scene_layout={"objects": [], "camera": camera})
                self.assertEqual(plan.errors, [])
                self.assertEqual(plan.values["detail_style"], "")

    def test_only_an_applied_style_lora_keeps_its_trigger_in_face_redraws(self):
        styles = self.studio.lib.all("styles")
        for style in styles:
            if style["id"] == "sx70-authentic":
                style["trigger"] = "sx70 frame"
        self.studio.lib.save("styles", styles)
        absent = copy.deepcopy(FLUX_FILES)
        absent["loras"].discard("sx70.safetensors")
        cases = [("applied", None, {}, True),
                 ("incompatible", None, {"model": "z-image-turbo"}, False),
                 ("missing", absent, {}, False),
                 ("zero", None, {"style_strength": 0}, False)]
        for name, inventory, changes, applied in cases:
            with self.subTest(case=name):
                if inventory is not None:
                    inventory["checkpoints"] = {"sam3.pt"}
                plan = self.plan(inventory, style="sx70-authentic", **changes)
                self.assertEqual(plan.errors, [])
                self.assertEqual("sx70 frame" in plan.prompt, applied)
                self.assertEqual("sx70 frame" in plan.values["detail_style"], applied)
                graph = ig.face_graph(plan.workflow, plan.values, plan.loras, "made.png",
                                      [{"x": 20, "y": 20, "width": 128, "height": 128}],
                                      "oval.png", "test", faces=[{"words": "A woman with auburn hair."}])
                text = next(t for t in encoded_words(graph) if "auburn hair" in t)
                self.assertIn("SX-70 instant film", text)
                self.assertEqual("sx70 frame" in text, applied)
                self.assertNotIn("yellow submarine", text)

    def test_a_scene_face_uses_the_composed_style_during_a_real_job(self):
        styles = self.studio.lib.all("styles")
        for style in styles:
            if style["id"] == "sx70-authentic":
                style["trigger"] = "sx70 frame"
        self.studio.lib.save("styles", styles)
        FaceClient.fail_pass = False

        class MissingStyleClient(FaceClient):
            def inventory(self):
                inventory = copy.deepcopy(super().inventory())
                inventory["loras"].discard("sx70.safetensors")
                return inventory

        cases = [("incompatible", "z-image-turbo", None, FaceClient, False),
                 ("missing", "flux-dev", None, MissingStyleClient, False),
                 ("zero", "flux-dev", 0, FaceClient, False),
                 ("applied", "flux-dev", None, FaceClient, True)]
        for name, model, strength, client_class, applied in cases:
            with self.subTest(case=name):
                self.studio.client_factory = client_class
                self.studio.clients = {}
                self.studio.inventories.clear()
                self.studio.nodes.clear()
                jobs = self.studio.submit(dict(ig.default_settings(), model=model,
                    style="sx70-authentic", style_strength=strength, backend="5090",
                    seed=5, face_detail=True, hand_pass=False,
                    scene="A woman beside a yellow submarine.", scene_faces={
                        "people": [{"id": "p1", "name": "The woman", "at": [.435, .355],
                            "region": [.35, .25, .5, .45], "words": "A woman with auburn hair.",
                            "face": "", "from": ""}]}))
                settle(jobs)
                self.assertEqual(jobs[0].status, "complete", jobs[0].detail)
                words = encoded_words(jobs[0].record["face_graph"])
                person = next(text for text in words if "auburn hair" in text)
                self.assertIn("SX-70 instant film", person)
                self.assertEqual("sx70 frame" in person, applied)
                self.assertNotIn("yellow submarine", person)


class TestStyleInFinishing(TempStudioMixin, unittest.TestCase):
    # Reuse the real-pipeline fake backend without inheriting unrelated tests.
    profile = finish_fixture.TestFinishRecovery.profile
    settings = finish_fixture.TestFinishRecovery.settings
    finish_job = finish_fixture.TestFinishRecovery.finish_job
    beard_region = finish_fixture.TestFinishRecovery.beard_region

    def setUp(self):
        super().setUp()
        self.addCleanup(self.studio.close)

    def test_all_detail_passes_keep_selected_styles_and_their_own_subjects(self):
        self.studio.lib.save("styles", self.studio.lib.all("styles") + [{
            "id": "custom-film", "name": "Custom film",
            "prompt": "Soft sepia instant film, lifted blacks and coarse grain."}])
        for style in ("illustration", "black-and-white", "custom-film"):
            with self.subTest(style=style):
                job, client, _ = self.finish_job([(460, 272, 60, 22)],
                    hands=[(200, 600, 70, 80)], eyes="blue", expression="soft smile",
                    style=style, character_regions=[self.beard_region()],
                    scene="A man beside a yellow submarine.")
                self.assertEqual(job.status, "complete", job.detail)
                passes = job.record["passes"]
                self.assertEqual([p["label"] for p in passes],
                                 ["Eye pass", "Beard", "Smile pass", "Hands", "Glasses"])
                phrase = self.studio.lib.get("styles", style)["prompt"].rstrip(" .")
                for result in passes:
                    with self.subTest(style=style, pass_name=result["label"]):
                        words = encoded_words(result["graph"])
                        self.assertTrue(any(phrase in text for text in words), words)
                        self.assertFalse(any("photo detail" in text for text in words), words)
                        self.assertFalse(any("yellow submarine" in text for text in words), words)
                eye_words = " ".join(encoded_words(passes[0]["graph"]))
                self.assertIn("blue", eye_words)

    def test_an_explicit_scene_medium_reaches_the_hands_pass(self):
        job, _, _ = self.finish_job([], profile=False, hands=[(200, 600, 70, 80)],
            scene="A watercolor illustration of a man beside a yellow submarine.")
        self.assertEqual(job.status, "complete", job.detail)
        words = encoded_words(job.record["passes"][0]["graph"])
        self.assertTrue(any("watercolor illustration" in text for text in words), words)
        self.assertFalse(any("yellow submarine" in text for text in words), words)

    def test_a_fix_uses_its_style_without_repeating_the_scene(self):
        self.studio.client_factory = FaceClient
        FaceClient.fail_pass = False
        source = os.path.join(self.dir, "made.png")
        Path(source).write_bytes(PNG)
        jobs = self.studio.submit(dict(ig.default_settings(), mode="fix", backend="5090",
            model="z-image-turbo", scene="A man beside a yellow submarine.",
            style="illustration", seed=5, fix={"image": source, "target": "hand",
                "words": "a gloved hand", "spots": [{"x": 0, "y": 0, "size": 64}]}))
        settle(jobs)
        self.assertEqual(jobs[0].status, "complete", jobs[0].detail)
        graph = FaceClient.instances[-1].graphs[0]
        words = " ".join(encoded_words(graph))
        self.assertIn("Editorial illustration", words)
        self.assertIn("a gloved hand", words)
        self.assertNotIn("yellow submarine", words)
        self.assertNotIn("photo detail", words)


if __name__ == "__main__":
    unittest.main()
