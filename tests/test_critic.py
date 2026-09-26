"""The Visual Critic: reading the vision model's answer, planning the next
pass, keeping the three states apart, compiling the instructions, and the
refinement loop in Studio against a fake ComfyUI and a fake vision model.
Nothing here touches the network or a GPU."""

import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import studio_critic as critic  # noqa: E402
import studio_imagegen as ig  # noqa: E402
from test_imagegen import FaceClient, TempStudioMixin, settle  # noqa: E402


def ob(feature, status, category="identity", confidence=0.9, severity="minor",
       target="", correction="", value="", subject="character_a"):
    return {"feature": feature, "status": status, "category": category,
            "confidence": confidence, "severity": severity, "target": target,
            "correction": correction, "value": value, "subject": subject,
            "observation": feature + " seen"}


def result(*obs, needs=True):
    return critic.clean_result({"summary": "two people", "needs_refinement": needs,
                                "observations": list(obs)})


MUNICH = result(
    ob("pose", "MATCH", "body"), ob("background", "MATCH", "scene", subject="scene"),
    ob("lighting", "MATCH", "lighting", subject="scene"),
    ob("character_a face width", "MISMATCH", correction="widen the jaw and cheeks"),
    ob("character_a beard colour", "MISMATCH", correction="neutral dark brown beard"),
    ob("accordion fingers", "MISMATCH", "realism", target="hand",
       correction="correct the fingers, keep the grip"),
    ob("eye colour", "UNCERTAIN", confidence=0.2),
    ob("jacket", "NEW_USEFUL_DETAIL", "clothing",
       value="dark forest-green wool Bavarian jacket with brown horn buttons"))


class ReadingTest(unittest.TestCase):
    def test_json_is_found_inside_fences_and_chatter(self):
        got = critic._json_in('Sure!\n```json\n{"needs_refinement": false}\n```')
        self.assertEqual(got, {"needs_refinement": False})
        with self.assertRaises(ValueError):
            critic._json_in("It looks lovely.")

    def test_a_made_up_status_is_uncertain_and_confidence_is_clamped(self):
        r = critic.clean_result({"observations": [
            {"feature": "nose", "status": "kinda wrong", "confidence": 7},
            {"feature": "hat", "status": "mismatch", "confidence": "x", "category": "HATS"}]})
        self.assertEqual([o["status"] for o in r["observations"]], ["UNCERTAIN", "MISMATCH"])
        self.assertEqual(r["observations"][0]["confidence"], 1.0)
        self.assertEqual(r["observations"][1]["confidence"], 0.5)
        self.assertEqual(r["observations"][1]["category"], "realism")
        self.assertTrue(r["needs_refinement"])     # unsaid, but a mismatch is there


class PlannerTest(unittest.TestCase):
    def setUp(self):
        self.canon = critic.initial_canonical({"scene": "Munich"}, ())

    def test_faces_and_hands_are_fixed_locally_and_the_rest_preserved(self):
        plan = critic.plan_next_refinement(MUNICH, self.canon)
        self.assertTrue(plan["needs_pass"])
        self.assertEqual([(a["type"], a["target"]) for a in plan["actions"]],
                         [("FACE_CORRECTION", "face"), ("LOCAL_INPAINT", "hand")])
        self.assertEqual(plan["preserve"], ["pose", "background", "lighting"])
        self.assertEqual(len(plan["actions"][0]["corrections"]), 2)
        self.assertNotIn("eye colour", " ".join(plan["correct"]))

    def test_a_low_confidence_mismatch_is_never_corrected(self):
        plan = critic.plan_next_refinement(result(
            ob("nose shape", "MISMATCH", confidence=0.3)), self.canon)
        self.assertFalse(plan["needs_pass"])
        self.assertEqual(plan["ignored"], ["nose shape"])
        self.assertEqual(plan["actions"][0]["type"], "NO_CHANGE")

    def test_a_structural_failure_regenerates_and_nothing_else(self):
        plan = critic.plan_next_refinement(result(
            ob("face", "MISMATCH"),
            ob("environment", "MISMATCH", "scene", severity="major", subject="scene")),
            self.canon)
        self.assertEqual([a["type"] for a in plan["actions"]], ["FULL_REGENERATION"])
        self.assertEqual(len(plan["actions"][0]["corrections"]), 2)

    def test_a_touch_up_waits_for_the_local_fixes(self):
        both = result(ob("face", "MISMATCH"), ob("skin gloss", "MISMATCH", "lighting"))
        plan = critic.plan_next_refinement(both, self.canon)
        self.assertEqual([a["type"] for a in plan["actions"]], ["FACE_CORRECTION"])
        self.assertEqual(len(plan["deferred"]), 1)
        alone = critic.plan_next_refinement(result(
            ob("film texture", "MISMATCH", "camera", subject="camera")), self.canon)
        self.assertEqual([a["type"] for a in alone["actions"]], ["GLOBAL_REFINEMENT"])

    def test_no_meaningful_problems_stops(self):
        plan = critic.plan_next_refinement(result(ob("face", "MISMATCH"), needs=False),
                                           self.canon)
        self.assertFalse(plan["needs_pass"])


class StateTest(unittest.TestCase):
    def test_the_intent_cannot_be_rewritten(self):
        intent = critic.intent_from({"scene": "Two people in Munich"}, "Two people in Munich.")
        with self.assertRaises(TypeError):
            intent["prompt"] = "something else"

    def test_a_good_detail_is_kept_once_and_the_users_words_never_replaced(self):
        canon = critic.initial_canonical({"hair": "dark brown", "subject": "a man"},
                                         ("hair", "subject"))
        plan = critic.plan_next_refinement(MUNICH, canon)
        canon, changed = critic.merge_canonical(canon, plan["promote"])
        self.assertEqual(canon["characters"]["character_a"]["jacket"],
                         "dark forest-green wool Bavarian jacket with brown horn buttons")
        self.assertEqual(changed, ["characters.character_a.jacket"])
        again, changed = critic.merge_canonical(canon, plan["promote"])
        self.assertEqual(changed, [])                      # no growth on a repeat
        user = critic.clean_result({"observations": [ob(
            "hair", "NEW_USEFUL_DETAIL", value="auburn", confidence=0.99)]})["observations"]
        self.assertEqual(critic.plan_next_refinement(
            {"observations": user, "needs_refinement": False}, canon)["promote"], [])
        self.assertEqual(critic.merge_canonical(canon, user)[0]["characters"]
                         ["character_a"]["hair"], "dark brown")

    def test_two_spellings_of_a_feature_are_one_key(self):
        canon = critic.initial_canonical({}, ())
        for name, value in (("Hair colour", "dark"), ("hair_color", "dark neutral brown")):
            o = critic.clean_result({"observations": [ob(name, "NEW_USEFUL_DETAIL",
                                                         value=value)]})["observations"]
            canon, _ = critic.merge_canonical(canon, o)
        self.assertEqual(canon["characters"]["character_a"], {"hair_color":
                                                              "dark neutral brown"})

    def test_instructions_carry_the_intent_word_for_word_in_sections(self):
        intent = critic.intent_from({}, "A man plays accordion in Munich.")
        canon = critic.initial_canonical({"facial_hair": "brown beard"}, ("facial_hair",))
        text = critic.build_refinement_instructions(intent, canon, ["pose"],
                                                    ["Make the beard dark brown."])
        for head in ("ORIGINAL USER INTENT:\nA man plays accordion in Munich.",
                     "CANONICAL CHARACTER INFORMATION:", "PRESERVE:\ncurrent pose",
                     "CORRECT:\nMake the beard dark brown."):
            self.assertIn(head, text)
        prompt = critic.generator_prompt(intent, canon, ["Make the beard dark brown."],
                                         focus="hand")
        self.assertTrue(prompt.startswith("A close-up of the hand"))
        self.assertIn("A man plays accordion in Munich.", prompt)


class WholePictureGraphTest(unittest.TestCase):
    def test_a_crop_without_a_mask_is_laid_back_whole_at_its_own_size(self):
        wf = ig.load_workflow("flux_dev_baseline")
        values = dict(wf["defaults"], model="m", weight_dtype="default", clip_l="c",
                      t5="t", vae="v", prompt="p", negative="", seed=1,
                      face_prompt="p", face_denoise=0.2, encoder_device="default",
                      width=1216, height=832)
        g = ig.face_graph(wf, values, [], "x.png [output]", [{
            "x": 0, "y": 0, "width": 1216, "height": 832, "mask": False,
            "edit": (1232, 848)}], "oval.png", "out")
        self.assertEqual(g["fc1_2"]["inputs"]["height"], 848)
        self.assertEqual(g["fc1_6"]["inputs"]["height"], 832)
        self.assertNotIn("fc1_8", g)
        self.assertNotIn("mask", g["fc1_9"]["inputs"])
        self.assertEqual(g["fc1_1"]["inputs"]["crop_region"],
                         {"x": 0, "y": 0, "width": 1216, "height": 832})


class WindowTest(unittest.TestCase):
    """A just-in-time load is 8,192 tokens; past it LM Studio drops the start
    of the request - the picture - and the critic judged the prompt alone."""
    INTENT = {"prompt": "a woman on a hill", "scene": "", "camera": ""}

    def vision(self, window):
        v = FakeVision([{"needs_refinement": False, "observations": []}])
        v.fitted = []
        v.fit = lambda need: v.fitted.append(need) or window
        return v

    def test_the_window_is_fitted_to_the_picture_and_its_references(self):
        with tempfile.TemporaryDirectory() as d:
            ref = os.path.join(d, "r.jpg")
            with open(ref, "wb") as f:
                f.write(b"x")
            v = self.vision(32768)
            critic.analyze_generated_image(v, b"png", self.INTENT, {}, [ref, ref])
        self.assertGreater(v.fitted[0], 3 * critic.IMAGE_TOKENS)
        self.assertEqual(len(v.asked), 1)

    def test_an_answer_cut_off_keeps_the_observations_it_finished(self):
        text = ('{"summary": "s", "needs_refinement": true, "observations": ['
                '{"feature": "a", "status": "MATCH"}, {"feature": "b", "status": "MIS')
        got = critic._json_in(text)
        self.assertEqual([o["feature"] for o in got["observations"]], ["a"])

    def test_a_window_too_small_fails_rather_than_guessing(self):
        v = self.vision(4096)
        with self.assertRaises(ValueError) as e:
            critic.analyze_generated_image(v, b"png", self.INTENT, {}, [])
        self.assertIn("4,096", str(e.exception))
        self.assertEqual(v.asked, [])


class FakeVision:
    """Answers the critic's question from a script, one answer per look."""
    MIME = {".jpg": "image/jpeg"}

    def __init__(self, answers):
        self.answers, self.asked = list(answers), []
        self.llm = self

    def _encode(self, raw, mime):
        return "AA=="

    def chat(self, messages, max_tokens=0):
        self.asked.append(messages)
        return {"choices": [{"message": {"content": json.dumps(self.answers.pop(0))}}]}


class CriticClient(FaceClient):
    """FaceClient plus the finder's quick runs, read from /history."""
    def get_history(self, pid):
        return {"status": {"status_str": "success"}, "outputs": {
            "fd4": {"text": [json.dumps([[{"x": 400, "y": 300, "width": 90,
                                           "height": 110}]])]},
            "fd6": {"text": ["1024"]}, "fd7": {"text": ["1024"]}}}


class LoopTest(TempStudioMixin, unittest.TestCase):
    def run_with(self, answers, **extra):
        self.vision = FakeVision(answers) if answers is not None else None
        self.studio = ig.Studio(root=self.dir, notify=self.notified.append,
                                client_factory=CriticClient, vision=lambda: self.vision)
        FaceClient.fail_pass = False
        jobs = self.studio.submit(dict(ig.default_settings(), model="flux-dev",
                                       scene="A man plays accordion in Munich.",
                                       facial_hair="brown beard", backend="5090", seed=5,
                                       auto_refine=True, **extra))
        settle(jobs)
        self.assertEqual(jobs[0].status, "complete", jobs[0].detail)
        return jobs[0], FaceClient.instances[-1]

    def test_a_face_fault_is_redrawn_then_the_critic_stops(self):
        face = ob("character_a beard colour", "MISMATCH", correction="dark brown beard")
        job, client = self.run_with([
            {"needs_refinement": True, "observations": [ob("pose", "MATCH", "body"), face]},
            {"needs_refinement": False, "observations": [ob("pose", "MATCH", "body")]}])
        ref = job.record["refinement"]
        self.assertEqual([h["type"] for h in ref["history"]],
                         ["initial_generation", "face_correction"])
        self.assertEqual(ref["stopped"], "no meaningful problems left")
        self.assertEqual(len(self.vision.asked), 2)
        finder, redraw = client.graphs[1], client.graphs[2]
        self.assertEqual(finder["fd2"]["inputs"]["text"], "face:8")
        self.assertIn("dark brown beard", redraw["f10"]["inputs"]["text"])
        self.assertEqual(redraw["fc1_4"]["inputs"]["denoise"], 0.45)
        self.assertEqual(ref["intent"]["prompt"], job.record["prompt"])   # untouched
        with open(os.path.join(self.dir, "visual_critic.log"), encoding="utf-8") as f:
            log = f.read()
        self.assertIn("VISUAL CRITIC PASS 1", log)
        self.assertIn("Selected action:\nFACE_CORRECTION", log)

    def test_the_pass_limit_holds_when_the_critic_is_never_satisfied(self):
        face = ob("jaw", "MISMATCH", correction="broader jaw")
        answers = [{"needs_refinement": True, "observations": [face]}] * 5
        job, _ = self.run_with(answers, refine_passes=2)
        self.assertEqual(len(self.vision.asked), 2)
        self.assertEqual(job.record["refinement"]["stopped"], "pass limit reached")

    def test_a_structural_failure_makes_a_new_picture_from_the_compiled_prompt(self):
        job, client = self.run_with([
            {"needs_refinement": True, "observations": [
                ob("environment", "MISMATCH", "scene", severity="major", subject="scene",
                   correction="an old Munich street, not a beach")]},
            {"needs_refinement": False, "observations": []}])
        regen = client.graphs[1]
        self.assertIn("not a beach", regen["10"]["inputs"]["text"])
        self.assertIn("A man plays accordion in Munich.", regen["10"]["inputs"]["text"])
        self.assertNotEqual(regen["40"]["inputs"]["seed"], 5)

    def test_without_a_vision_model_the_picture_is_kept_with_a_warning(self):
        job, client = self.run_with(None)
        self.assertEqual(len(client.graphs), 1)
        self.assertTrue(any("vision model" in w for w in job.record["warnings"]))

    def test_a_critic_that_answers_nonsense_keeps_the_picture(self):
        self.vision = None
        job, client = self.run_with(["not json at all"])
        self.assertEqual(len(client.graphs), 1)
        self.assertEqual(job.record["refinement"]["stopped"], "critic failed")


if __name__ == "__main__":
    unittest.main()
