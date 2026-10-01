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

import apps.image_studio.critic as critic  # noqa: E402
import apps.image_studio.imagegen as ig  # noqa: E402
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
                                       auto_refine=True, hand_pass=False, **extra))
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
        job, client = self.run_with(answers, refine_passes=2)
        self.assertEqual(len(self.vision.asked), 3)      # two passes, and a look at the last
        self.assertEqual(len(client.graphs), 1 + 2 * 2)  # each pass: the finder, the redraw
        self.assertEqual(job.record["refinement"]["stopped"], "pass limit reached")
        self.assertEqual(job.record["refinement"]["left"], ["jaw"])

    def test_a_fault_still_there_is_redrawn_harder_then_left_to_the_user(self):
        jaw = ob("jaw", "MISMATCH", correction="broader jaw")

        def still(fix):
            return {"needs_refinement": True, "observations": [], "followups": [
                {"id": 1, "outcome": "PERSISTS", "observation": "jaw still narrow",
                 "correction": fix}]}
        job, client = self.run_with([{"needs_refinement": True, "observations": [jaw]},
                                     still("a wide square jaw"), still("wider")])
        ref = job.record["refinement"]
        self.assertEqual(len(self.vision.asked), 3)
        self.assertIn("FAULTS FOUND BEFORE", str(self.vision.asked[1]))
        self.assertIn("1. jaw", str(self.vision.asked[1]))
        first, second = client.graphs[2], client.graphs[4]
        self.assertEqual(first["fc1_4"]["inputs"]["denoise"], 0.45)
        self.assertEqual(second["fc1_4"]["inputs"]["denoise"], 0.6)
        self.assertIn("a wide square jaw", second["f10"]["inputs"]["text"])
        self.assertEqual(len(client.graphs), 5)          # tried twice, not a third time
        self.assertEqual(ref["stopped"], "faults left to the user")
        self.assertEqual(ref["left"], ["jaw"])
        self.assertEqual([(x["outcome"], x["tries"]) for x in ref["scores"]],
                         [("PERSISTS", 2)])
        self.assertEqual(ref["history"][1]["scores"][0]["outcome"], "PERSISTS")
        self.assertTrue(any("left to you" in n for n in job.record["notes"]))

    def test_a_pass_that_made_it_worse_is_taken_back(self):
        jaw = ob("jaw", "MISMATCH", correction="broader jaw")
        job, client = self.run_with([
            {"needs_refinement": True, "observations": [jaw]},
            {"needs_refinement": False, "observations": [], "followups": [
                {"id": 1, "outcome": "WORSE", "observation": "a seam across the chin"}]},
            {"needs_refinement": False, "observations": [], "followups": [
                {"id": 1, "outcome": "CLEARED"}]}])
        ref = job.record["refinement"]
        self.assertTrue(ref["history"][1]["taken_back"])
        # Redrawn again from the picture before, not from the damaged one.
        self.assertEqual(client.graphs[4]["fi"]["inputs"]["image"],
                         client.graphs[2]["fi"]["inputs"]["image"])
        self.assertEqual(ref["scores"][0]["outcome"], "CLEARED")
        self.assertEqual(ref["left"], [])

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

    def test_a_kept_scene_detail_goes_into_the_next_picture(self):
        jacket = ob("stage backdrop", "NEW_USEFUL_DETAIL", "scene", subject="scene",
                    value="red velvet curtain behind him")
        self.run_with([{"needs_refinement": False, "observations": [jacket]}])
        with open(os.path.join(self.dir, ig.CRITIC_MEMORY), encoding="utf-8") as f:
            self.assertIn("red velvet curtain", f.read())
        job, client = self.run_with([{"needs_refinement": False, "observations": []}])
        self.assertIn("red velvet curtain behind him", client.graphs[0]["10"]["inputs"]["text"])
        self.assertIn("red velvet curtain", str(self.vision.asked[0]))   # checked, not reinvented


class FaultsTest(unittest.TestCase):
    """The user's notes as faults, and what became of a fault by the next look."""
    SPOTS = [{"x": 40, "y": 40, "size": 64, "note": "thumb on the wrong side"},
             {"x": 90, "y": 40, "size": 64},
             {"x": 90, "y": 90, "size": 64, "photo": "hat.png"}]

    def test_the_users_notes_are_certain_faults_already_tried_once(self):
        a, b, c = critic.user_faults(self.SPOTS, "hand", note="six fingers")
        self.assertEqual((a["id"], a["spot"], a["source"], a["confidence"], a["tries"]),
                         (1, 0, "user", 1.0, 1))
        self.assertEqual(a["observation"], "thumb on the wrong side")
        self.assertEqual(b["observation"], "six fingers")     # the fix's note, for the rest
        self.assertTrue(a["redo"])
        self.assertFalse(c["redo"])                            # a photo is not redrawn
        (bare,) = critic.user_faults([{"x": 1, "y": 1, "size": 64}], "face")
        self.assertEqual(bare["observation"], "the face looked wrong")

    def test_the_question_asks_after_each_fault_by_number(self):
        intent = critic.intent_from({}, "A man waves.")
        faults = critic.user_faults(self.SPOTS[:1], "hand")
        text = critic.critic_prompt(intent, {}, faults, closeups=[1])
        self.assertIn("1. hand 1: thumb on the wrong side. [marked by the user]", text)
        self.assertIn("is a close-up of fault 1", text)
        self.assertNotIn("references", text)
        plain = critic.critic_prompt(intent, {})
        self.assertIn("references", plain)
        self.assertNotIn("FAULTS FOUND BEFORE", plain)

    def test_a_close_up_is_sent_in_place_of_the_references(self):
        v = FakeVision([{"observations": []}])
        with tempfile.TemporaryDirectory() as d:
            ref = os.path.join(d, "r.jpg")
            with open(ref, "wb") as f:
                f.write(b"x")
            critic.analyze_generated_image(
                v, b"png", {"prompt": "p", "scene": "", "camera": ""}, {}, [ref],
                faults=critic.user_faults(self.SPOTS[:1]), closeups=[(1, b"crop")])
        self.assertEqual(len(v.asked[0][0]["content"]), 3)     # words, picture, close-up

    def test_a_fix_is_scored_by_its_number_else_by_its_name_else_not_at_all(self):
        faults = critic.as_faults([ob("jaw", "MISMATCH"), ob("beard", "MISMATCH"),
                                   ob("hat", "MISMATCH"), ob("ear", "MISMATCH")], 1)
        self.assertEqual([(f["id"], f["tries"]) for f in faults],
                         [(1, 1), (2, 1), (3, 1), (4, 1)])
        look = critic.clean_result({"observations": [
            ob("Beard", "MATCH"), ob("hat", "MISMATCH", correction="a felt hat")],
            "followups": [{"id": 1, "outcome": "persists", "correction": "wider"},
                          {"id": "x"}, {"id": 9, "outcome": "CLEARED"}]})
        got = critic.score_fixes(faults, look)
        self.assertEqual([f["outcome"] for f in got],
                         ["PERSISTS", "CLEARED", "PERSISTS", "UNKNOWN"])
        self.assertEqual(got[0]["correction"], "wider")
        self.assertEqual(got[2]["correction"], "a felt hat")
        made_up = critic.clean_result({"followups": [{"id": 1, "outcome": "sort of"}]})
        self.assertEqual(critic.score_fixes(faults[:1], made_up)[0]["outcome"], "UNKNOWN")

    def test_a_users_note_is_not_reworded_by_the_critic(self):
        (f,) = critic.user_faults(self.SPOTS[:1], "hand")
        (got,) = critic.score_fixes([f], critic.clean_result({"followups": [
            {"id": 1, "outcome": "PERSISTS", "observation": "hand looks fine-ish",
             "correction": "thumb beside the index finger"}]}))
        self.assertEqual(got["observation"], "thumb on the wrong side")
        self.assertEqual(got["correction"], "thumb beside the index finger")

    def test_what_is_still_there_goes_again_until_it_was_tried_twice(self):
        scored = [dict(ob("jaw", "MISMATCH"), id=1, tries=1, outcome="PERSISTS"),
                  dict(ob("hat", "MISMATCH"), id=2, tries=2, outcome="WORSE"),
                  dict(ob("ear", "MISMATCH"), id=3, tries=1, outcome="CLEARED"),
                  dict(ob("eye", "MISMATCH"), id=4, tries=1, outcome="UNKNOWN"),
                  dict(ob("bag", "MISMATCH"), id=5, tries=1, outcome="PERSISTS", redo=False)]
        again, left = critic.carry(scored)
        self.assertEqual([f["feature"] for f in again], ["jaw"])
        self.assertEqual([f["feature"] for f in left], ["hat", "bag"])
        self.assertFalse(critic.went_wrong(scored))            # the ear was mended
        self.assertTrue(critic.went_wrong(scored[:2]))
        self.assertFalse(critic.went_wrong(scored[:1]))

    def test_a_redraw_is_harder_each_time_up_to_its_top(self):
        self.assertEqual(critic.harder(0.45, 0, 0.6), 0.45)
        self.assertEqual(critic.harder(0.45, 1, 0.6), 0.6)
        self.assertEqual(critic.harder(0.45, 2, 0.6), 0.6)
        self.assertEqual(critic.harder(0.6, 1, 0.6), 0.6)      # a hand's does not rise
        self.assertEqual(critic.harder(0.7, 1, 0.6), 0.7)      # nor fall under what was asked

    def test_a_carried_fault_is_planned_once_and_knows_its_tries(self):
        canon = critic.initial_canonical({}, ())
        carried = [dict(ob("jaw", "MISMATCH", correction="wider"), id=1, tries=1)]
        plan = critic.plan_next_refinement(
            result(ob("Jaw", "MISMATCH", correction="again"), needs=False), canon,
            carried=carried, closed=["jaw"])
        (action,) = plan["actions"]
        self.assertTrue(plan["needs_pass"])                    # whatever the critic said
        self.assertEqual((action["type"], action["tries"]), ("FACE_CORRECTION", 1))
        self.assertEqual(action["corrections"], ["jaw seen. wider."])
        self.assertEqual(critic.progress_text(action), "Refining faces again, harder")


class CloseUpClient(FaceClient):
    """FaceClient that also cuts the critic's close-ups."""
    def listen_for_progress(self, pid, on_event, stop=None, timeout=0):
        graph = self.graphs[int(pid[3:]) - 1]
        if "cs0" in graph:
            return {"status": {"completed": True}, "outputs": {
                k: {"images": [{"filename": k + ".png", "subfolder": "", "type": "temp"}]}
                for k in graph if k.startswith("cs")}}
        return super().listen_for_progress(pid, on_event, stop, timeout)


class FixCheckTest(TempStudioMixin, unittest.TestCase):
    """Fix a spot with the critic's check: the user's spots are the faults."""

    def fix_with(self, answers, client=CloseUpClient, **fix):
        self.vision = FakeVision(answers) if answers is not None else None
        self.studio = ig.Studio(root=self.dir, notify=self.notified.append,
                                client_factory=client, vision=lambda: self.vision)
        FaceClient.fail_pass = False
        src = os.path.join(self.dir, "made.png")
        with open(src, "wb") as f:
            f.write(ig.oval_png(256))
        s = self.studio.fix_base(dict(ig.default_settings(), model="flux-dev",
                                      scene="On a pier.", backend="5090"))
        s.update(mode="fix", seed=5, fix=dict({
            "image": src, "target": "hand", "strength": "light", "check": True,
            "spots": [{"x": 60, "y": 60, "size": 64, "note": "thumb on the wrong side"},
                      {"x": 180, "y": 180, "size": 64}]}, **fix))
        jobs = self.studio.submit(s)
        settle(jobs)
        self.assertEqual(jobs[0].status, "complete", jobs[0].detail)
        return jobs[0], FaceClient.instances[-1]

    def follow(self, *outcomes):
        return {"observations": [ob("sky", "MISMATCH", "scene", subject="scene")],
                "followups": [{"id": i, "outcome": o, "correction": "thumb by the index finger"}
                              for i, o in enumerate(outcomes, 1)]}

    def test_a_note_is_kept_with_its_spot(self):
        f = ig.clean_fix({"note": " six fingers ", "check": 1, "spots": [
            {"x": 5, "y": 6, "size": 100, "note": "x" * 500}]})
        self.assertEqual(f["note"], "six fingers")
        self.assertFalse(f["check"])                           # True alone turns it on
        self.assertEqual(len(f["spots"][0]["note"]), ig.FIX_NOTE_MAX)

    def test_without_the_check_a_fix_asks_no_one(self):
        job, client = self.fix_with([], check=False)
        self.assertEqual(len(client.graphs), 1)
        self.assertEqual(self.vision.asked, [])
        self.assertIsNone(job.record["refinement"])

    def test_the_spot_still_wrong_is_redrawn_again_and_the_other_left_alone(self):
        job, client = self.fix_with([self.follow("PERSISTS", "CLEARED"),
                                     self.follow("CLEARED")])
        fix, cut, redo, cut2 = client.graphs
        self.assertIn("fc2_4", fix)                            # the fix: both spots
        self.assertEqual(sorted(k for k in cut if k.startswith("cs")), ["cs0", "cs1"])
        asked = self.vision.asked[0][0]["content"]
        self.assertEqual(len(asked), 4)                        # words, picture, two close-ups
        self.assertIn("1. hand 1: thumb on the wrong side. [marked by the user]",
                      asked[0]["text"])
        self.assertIn("fc1_4", redo)
        self.assertNotIn("fc2_4", redo)                        # only the one still wrong
        self.assertEqual(redo["fc1_1"]["inputs"]["crop_region"],
                         fix["fc1_1"]["inputs"]["crop_region"])
        self.assertEqual(redo["fc1_4"]["inputs"]["denoise"], 0.6)   # light 0.45, harder
        said = [n["inputs"]["text"] for n in redo.values()
                if n["class_type"] == "CLIPTextEncode"]
        self.assertTrue(any("thumb by the index finger" in t for t in said))
        self.assertEqual(redo["fi"]["inputs"]["image"], "ImageStudio/faces_00001_.png [output]")
        self.assertEqual(list(cut2), ["cu", "cu0", "cs0"])     # only what was redrawn
        ref = job.record["refinement"]
        self.assertEqual(ref["history"][0]["type"], "fix")
        self.assertEqual([(x["spot"], x["outcome"]) for x in ref["scores"]],
                         [(0, "CLEARED"), (1, "CLEARED")])
        self.assertEqual(ref["stopped"], "no meaningful problems left")
        self.assertEqual(job.record["fix"]["spots"][0]["note"], "thumb on the wrong side")

    def test_what_the_critic_finds_elsewhere_is_not_touched_by_a_fix(self):
        job, client = self.fix_with([self.follow("CLEARED", "CLEARED")])
        self.assertEqual(len(client.graphs), 2)                # the fix, the close-ups
        self.assertEqual(job.record["refinement"]["history"][0]["scores"][0]["outcome"],
                         "CLEARED")

    def test_a_spot_wrong_after_two_redraws_is_left_to_the_user(self):
        job, client = self.fix_with([self.follow("PERSISTS", "CLEARED"),
                                     self.follow("PERSISTS"), self.follow("PERSISTS")])
        self.assertEqual(len(self.vision.asked), 2)
        self.assertEqual(job.record["refinement"]["left"], ["hand 1"])
        self.assertEqual(job.record["refinement"]["stopped"], "faults left to the user")

    def test_without_close_ups_the_critic_judges_by_the_whole_picture(self):
        job, client = self.fix_with([self.follow("CLEARED", "CLEARED")], client=FaceClient)
        self.assertEqual(len(self.vision.asked[0][0]["content"]), 2)

    def test_without_a_vision_model_the_fix_is_kept_with_a_warning(self):
        job, client = self.fix_with(None)
        self.assertEqual(len(client.graphs), 1)
        self.assertTrue(any("critic's check" in w for w in job.record["warnings"]))


def scored(feature, outcome, action="FACE_CORRECTION", denoise=0.45, **kw):
    return dict(ob(feature, "MISMATCH", **kw), id=1, tries=1, outcome=outcome,
                action=action, denoise=denoise)


class LedgerTest(unittest.TestCase):
    """What the scores add up to, and what the next picture takes from it."""

    def test_a_strength_that_seldom_mends_gives_way_to_one_that_does(self):
        led = {}
        for outcome in ("PERSISTS", "PERSISTS", "WORSE", "UNKNOWN"):
            led = critic.note_fixes(led, "flux-dev", [scored("jaw", outcome, target="face")])
        rec = led["fixes"]["flux-dev"]["identity/face"]["FACE_CORRECTION@0.45"]
        self.assertEqual(rec, {"tried": 3, "cleared": 0, "worse": 1})   # unknown: no evidence
        start = lambda top=0.6, model="flux-dev": critic.start_denoise(   # noqa: E731
            led, model, "identity/face", "FACE_CORRECTION", 0.45, top)
        self.assertEqual(start(), (0.45, ""))              # nothing harder is known to work
        for outcome in ("CLEARED", "CLEARED"):
            led = critic.note_fixes(led, "flux-dev", [scored("jaw", outcome, denoise=0.6,
                                                             target="face")])
        self.assertEqual(start()[0], 0.45)                 # two tries are not yet believed
        led = critic.note_fixes(led, "flux-dev", [scored("jaw", "PERSISTS", denoise=0.6,
                                                         target="face")])
        d, why = start()
        self.assertEqual(d, 0.6)
        self.assertIn("0.45 mended 0 of 3 before, 0.6 2 of 3", why)
        self.assertEqual(start(top=0.5)[0], 0.45)          # never past the top
        self.assertEqual(start(model="z-image-turbo")[0], 0.45)   # another model's record

    def test_a_strength_that_mends_is_kept(self):
        led = {}
        for outcome in ("CLEARED", "CLEARED", "PERSISTS"):
            led = critic.note_fixes(led, "m", [scored("jaw", outcome, target="face")])
        self.assertEqual(critic.start_denoise(led, "m", "identity/face", "FACE_CORRECTION",
                                              0.45, 0.6), (0.45, ""))

    def test_the_critic_must_repeat_itself_and_the_user_is_believed_at_once(self):
        hand = ob("fingers", "MISMATCH", "realism", target="hand", correction="five fingers")
        led = {}
        for n in (1, 2):
            led = critic.note_picture(led, "flux-dev", ["sitter"], "On a pier.", [hand])
            self.assertEqual(critic.recurring(led, "flux-dev", ["sitter"]), [])
        led = critic.note_picture(led, "flux-dev", ["sitter"], "On a pier.", [hand])
        (r,) = critic.recurring(led, "flux-dev", ["sitter"], "on a pier")
        self.assertEqual((r["kind"], r["weight"], r["seen"]), ("realism/hand", 3, 3))
        self.assertEqual(critic.first_line(r), "- hand (realism): wrong in 3 pictures "
                                               "before; last: fingers seen")
        (mark,) = critic.user_faults([{"x": 1, "y": 1, "size": 64, "note": "six fingers"}],
                                     "hand")
        once = critic.note_picture({}, "flux-dev", ["partner"], "", [mark], looked=False)
        (r,) = critic.recurring(once, "", ["partner"])
        self.assertEqual((r["weight"], r["user"], r["said"]), (3, 1, "six fingers"))
        self.assertIn("marked by the user", critic.first_line(r))
        self.assertEqual(once["faults"]["identities"]["partner"]["pictures"], 0)

    def test_a_fault_that_stopped_coming_back_is_unlearned(self):
        hand = ob("fingers", "MISMATCH", "realism", target="hand")
        led = {}
        for n in range(3):
            led = critic.note_picture(led, "m", ["sitter"], "", [hand])
        led = critic.note_picture(led, "m", ["sitter"], "", [])
        self.assertEqual(critic.recurring(led, "m", ["sitter"]), [])
        for n in range(2):
            led = critic.note_picture(led, "m", ["sitter"], "", [])
        self.assertEqual(led["faults"]["identities"]["sitter"],
                         {"pictures": 6, "kinds": {}})
        marked = critic.note_picture(led, "m", ["sitter"], "", [], looked=False)
        self.assertEqual(marked["faults"]["identities"]["sitter"]["pictures"], 6)

    def test_with_two_people_a_fault_is_the_models_not_a_persons(self):
        led = critic.note_picture({}, "m", ["a", "b"], "", [ob("jaw", "MISMATCH",
                                                                target="face")])
        self.assertEqual(sorted(led["faults"]), ["models"])
        self.assertEqual(critic.note_picture({}, "m", ["a"], "A pier.", []), {})

    def test_only_so_many_kinds_are_kept(self):
        found = [ob("x", "MISMATCH", "scene", target="thing %d" % i) for i in range(20)]
        led = critic.note_picture({}, "m", [], "", found)
        self.assertEqual(len(led["faults"]["models"]["m"]["kinds"]), critic.KINDS_MAX)

    def test_words_against_a_returning_fault_are_only_about_the_person(self):
        jaw = ob("jaw", "MISMATCH", target="face", correction="a broad square jaw.")
        hand = ob("fingers", "MISMATCH", "realism", target="hand", correction="five fingers")
        led = {}
        for n in range(3):
            led = critic.note_picture(led, "m", ["sitter"], "", [jaw, hand])
        self.assertEqual(critic.prevention(led, ["sitter"]),
                         [("identity/face", "a broad square jaw")])
        self.assertEqual(critic.prevention(led, ["sitter", "partner"]), [])
        self.assertEqual(critic.prevention(led, ["partner"]), [])

    def test_what_the_user_marks_on_a_passed_picture_is_a_blind_spot(self):
        marks = critic.user_faults([{"x": 1, "y": 1, "size": 64, "note": "six fingers"}],
                                   "hand")
        made = {"history": [{"type": "initial_generation"}], "scores": []}
        led = critic.note_blind({}, marks, made)
        self.assertEqual(critic.blind_checks(led), {"realism": ["six fingers"]})
        self.assertEqual(critic.note_blind({}, marks, None), {})          # it never looked
        flagged = dict(made, scores=[{"category": "realism", "target": "hand",
                                      "outcome": "PERSISTS"}])
        self.assertEqual(critic.blind_checks(critic.note_blind({}, marks, flagged)), {})
        fixed = {"history": [{"type": "fix"}], "scores": []}              # it saw only spots
        self.assertEqual(critic.blind_checks(critic.note_blind({}, marks, fixed)), {})
        fixed["scores"] = [{"category": "realism", "target": "hand", "outcome": "CLEARED"}]
        self.assertEqual(critic.blind_checks(critic.note_blind({}, marks, fixed)),
                         {"realism": ["six fingers"]})
        bare = critic.user_faults([{"x": 1, "y": 1, "size": 64}], "hand")
        self.assertEqual(critic.blind_checks(critic.note_blind({}, bare, made)),
                         {"realism": ["a wrong hand"]})

    def test_the_most_missed_are_the_ones_checked_for(self):
        made = {"history": [{"type": "initial_generation"}], "scores": []}
        led = {}
        for i, times in enumerate((1, 3, 2, 1, 4)):
            mark = critic.user_faults([{"x": 1, "y": 1, "size": 64, "note": "fault %d" % i}])
            for n in range(times):
                led = critic.note_blind(led, mark, made)
        self.assertEqual(critic.blind_checks(led), {"realism": ["fault 4", "fault 1",
                                                                "fault 2"]})

    def test_a_fix_tried_again_is_not_one_more_picture_with_the_fault(self):
        marks = critic.user_faults([{"x": 1, "y": 1, "size": 64, "note": "six fingers"}],
                                   "hand")
        led = critic.note_marked({}, "a.png", marks, "m", ["sitter"], "", None)
        again = critic.note_marked(led, "a.png", marks, "m", ["sitter"], "", None)
        self.assertEqual(again, led)
        other = critic.note_marked(led, "b.png", marks, "m", ["sitter"], "", None)
        kind = other["faults"]["identities"]["sitter"]["kinds"]["realism/hand"]
        self.assertEqual((kind["seen"], kind["weight"]), (2, 6))
        self.assertEqual(other["marked"], [["a.png", "realism/hand"],
                                           ["b.png", "realism/hand"]])

    def test_the_question_says_what_to_look_at_first_and_what_was_missed(self):
        intent = critic.intent_from({}, "A man waves.")
        text = critic.critic_prompt(
            intent, {}, first=[{"kind": "realism/hand", "seen": 4, "user": 1,
                                "said": "six fingers"}],
            missed={"realism": ["six fingers", "a wrong hand"]})
        self.assertIn("WRONG BEFORE in pictures like this one", text)
        self.assertIn("- hand (realism): wrong in 4 pictures before, marked by the user; "
                      "last: six fingers", text)
        self.assertIn("AI textures; missed before: six fingers; a wrong hand", text)
        self.assertLess(text.index("WRONG BEFORE"), text.index("Look at every category"))
        self.assertNotIn("missed before", critic.critic_prompt(intent, {}))


class LearningTest(TempStudioMixin, unittest.TestCase):
    """The ledger on disk, through Generate and Fix a spot."""
    HAND = ob("fingers", "MISMATCH", "realism", target="hand", correction="five fingers")
    FINE = {"needs_refinement": False, "observations": []}

    def generate(self, answers, client=CriticClient, **extra):
        self.vision = FakeVision(answers)
        self.studio = ig.Studio(root=self.dir, notify=self.notified.append,
                                client_factory=client, vision=lambda: self.vision)
        FaceClient.fail_pass = False
        jobs = self.studio.submit(dict(ig.default_settings(), model="flux-dev",
                                       scene="On a pier.", backend="5090", seed=5,
                                       auto_refine=True, hand_pass=False, **extra))
        settle(jobs)
        self.assertEqual(jobs[0].status, "complete", jobs[0].detail)
        return jobs[0], FaceClient.instances[-1]

    def ledger(self):
        return ig.load_critic_ledger(self.studio.lib)

    def test_what_a_redraw_did_is_filed_by_model_fault_tool_and_strength(self):
        self.generate([{"needs_refinement": True, "observations": [self.HAND]},
                       {"needs_refinement": False, "observations": [], "followups": [
                           {"id": 1, "outcome": "CLEARED"}]}])
        led = self.ledger()
        self.assertEqual(led["fixes"]["flux-dev"]["realism/hand"],
                         {"LOCAL_INPAINT@0.60": {"tried": 1, "cleared": 1, "worse": 0}})
        self.assertEqual(led["faults"]["models"]["flux-dev"]["kinds"]["realism/hand"]["seen"], 1)
        self.assertEqual(led["faults"]["scenes"]["on_a_pier"]["pictures"], 1)

    def test_the_fourth_picture_is_looked_at_for_what_went_wrong_in_three(self):
        for n in range(3):
            self.generate([{"needs_refinement": True, "observations": [self.HAND]},
                           dict(self.FINE, followups=[{"id": 1, "outcome": "CLEARED"}])])
            self.assertNotIn("WRONG BEFORE", str(self.vision.asked[0]))
        job, _ = self.generate([self.FINE])
        self.assertIn("- hand (realism): wrong in 3 pictures before",
                      self.vision.asked[0][0]["content"][0]["text"])
        self.assertTrue(any("looked first" in n for n in job.record["notes"]))

    def test_a_redraw_starts_where_the_ledger_says_it_mends(self):
        jaw = ob("jaw", "MISMATCH", target="face", correction="broader jaw")
        led = {}
        for outcome, d in (("PERSISTS", 0.45),) * 3 + (("CLEARED", 0.6),) * 3:
            led = critic.note_fixes(led, "flux-dev", [scored("jaw", outcome, denoise=d,
                                                             target="face")])
        self.studio = ig.Studio(root=self.dir, notify=self.notified.append)
        ig.save_critic_ledger(self.studio.lib, led)
        job, client = self.generate([{"needs_refinement": True, "observations": [jaw]},
                                     dict(self.FINE, followups=[{"id": 1,
                                                                 "outcome": "CLEARED"}])])
        self.assertEqual(client.graphs[2]["fc1_4"]["inputs"]["denoise"], 0.6)
        self.assertTrue(any("from the start" in n for n in job.record["notes"]))
        self.assertEqual(self.ledger()["fixes"]["flux-dev"]["identity/face"]
                         ["FACE_CORRECTION@0.60"]["tried"], 4)

    def test_a_persons_returning_fault_is_drawn_against_in_their_next_picture(self):
        jaw = ob("jaw", "MISMATCH", target="face", correction="a broad square jaw")
        led = {}
        for n in range(3):
            led = critic.note_picture(led, "flux-dev", ["sitter"], "", [jaw])
        ig.save_critic_ledger(self.studio.lib, led)
        job, client = self.generate([self.FINE], preset="identity", identities=["sitter"])
        self.assertIn("a broad square jaw", client.graphs[0]["10"]["inputs"]["text"])
        self.assertTrue(any("Drawn against" in n for n in job.record["notes"]))
        job, client = self.generate([self.FINE])          # a picture of no one known
        self.assertNotIn("a broad square jaw", client.graphs[0]["10"]["inputs"]["text"])

    def test_generate_again_replays_the_words_the_picture_was_made_with(self):
        # Idea 8ff4376dcd06: Again composed the prompt anew, and compose adds
        # what the critic learned since - the same seed on other words.
        job, _ = self.generate([self.FINE], preset="identity", identities=["sitter"])
        made = job.record["prompt"]
        led = {}
        jaw = ob("jaw", "MISMATCH", target="face", correction="a broad square jaw")
        for n in range(3):
            led = critic.note_picture(led, "flux-dev", ["sitter"], "", [jaw])
        ig.save_critic_ledger(self.studio.lib, led)
        client = FaceClient.instances[-1]             # the backend's one client, reused
        sent = len(client.graphs)
        jobs = self.studio.submit(ig.again(job.record))
        settle(jobs)
        self.assertEqual(jobs[0].status, "complete", jobs[0].detail)
        self.assertEqual(client.graphs[sent]["10"]["inputs"]["text"], made)
        self.assertNotIn("a broad square jaw", made)
        self.assertFalse(any("Drawn against" in n for n in jobs[0].record["notes"]))
        self.assertTrue(any("replayed" in n for n in jobs[0].record["notes"]))
        # A new seed is a variation: composed afresh, with what was learned
        # (the ledger again as it was: the replay's own clean picture is in it).
        ig.save_critic_ledger(self.studio.lib, led)
        sent = len(client.graphs)
        jobs = self.studio.submit(ig.again(job.record, new_seed=True))
        settle(jobs)
        self.assertIn("a broad square jaw", client.graphs[sent]["10"]["inputs"]["text"])
        # Fix a spot starts from the picture's settings, not its replayed words.
        self.assertNotIn("replay_prompt", self.studio.fix_base(ig.again(job.record)))

    def test_a_mark_on_a_picture_the_critic_passed_goes_into_its_checks(self):
        made, _ = self.generate([self.FINE])
        src = made.record["images"][0]
        s = self.studio.fix_base(made.record["settings"])
        s.update(mode="fix", seed=5, fix={"image": src, "target": "hand", "spots": [
            {"x": 1, "y": 1, "size": 64, "note": "six fingers"}]})
        jobs = self.studio.submit(s)
        settle(jobs)
        self.assertEqual(jobs[0].status, "complete", jobs[0].detail)
        led = self.ledger()
        self.assertEqual(critic.blind_checks(led), {"realism": ["six fingers"]})
        self.assertEqual(led["faults"]["models"]["flux-dev"]["kinds"]["realism/hand"]["user"], 1)
        self.generate([self.FINE])
        text = self.vision.asked[0][0]["content"][0]["text"]
        self.assertIn("missed before: six fingers", text)
        self.assertIn("- hand (realism): wrong in 1 picture before, marked by the user", text)

    def test_a_checked_fix_files_what_its_redraws_did(self):
        self.vision = FakeVision([{"observations": [], "followups": [
            {"id": 1, "outcome": "PERSISTS", "correction": "five fingers"}]},
            {"observations": [], "followups": [{"id": 1, "outcome": "CLEARED"}]}])
        self.studio = ig.Studio(root=self.dir, notify=self.notified.append,
                                client_factory=FaceClient, vision=lambda: self.vision)
        src = os.path.join(self.dir, "made.png")
        with open(src, "wb") as f:
            f.write(ig.oval_png(256))
        s = self.studio.fix_base(dict(ig.default_settings(), model="flux-dev",
                                      scene="On a pier.", backend="5090"))
        s.update(mode="fix", seed=5, fix={"image": src, "target": "hand", "check": True,
                                          "strength": "light", "spots": [
                                              {"x": 60, "y": 60, "size": 64}]})
        jobs = self.studio.submit(s)
        settle(jobs)
        self.assertEqual(jobs[0].status, "complete", jobs[0].detail)
        self.assertEqual(self.ledger()["fixes"]["flux-dev"]["realism/hand"], {
            "FIX@0.45": {"tried": 1, "cleared": 0, "worse": 0},
            "FIX@0.60": {"tried": 1, "cleared": 1, "worse": 0}})
        self.assertEqual(critic.blind_checks(self.ledger()), {})   # no record: never looked


class MemoryTest(unittest.TestCase):
    def test_a_person_is_remembered_by_identity_and_a_scene_by_its_words(self):
        state = {"characters": {"character_a": {"jacket": "green wool jacket"}},
                 "scene": {"backdrop": "brick wall"}, "camera": {}}
        mem = critic.remember({}, state, ["characters.character_a.jacket", "scene.backdrop"],
                              ["sitter"], "A man in Munich.")
        self.assertEqual(critic.recall(mem, ["sitter"]), [("jacket", "green wool jacket")])
        self.assertEqual(critic.recall(mem, (), "a man in munich"), [("backdrop", "brick wall")])
        self.assertEqual(critic.recall(mem, ["sitter"], set_keys=["Jacket"]), [])
        self.assertEqual(critic.recall(mem, ["sitter", "partner"]), [])   # whose? not said

    def test_with_two_people_a_character_detail_is_not_filed(self):
        state = {"characters": {"character_a": {"jacket": "x"}}, "scene": {}, "camera": {}}
        self.assertEqual(critic.remember({}, state, ["characters.character_a.jacket"],
                                         ["a", "b"]), {})


class KeptFileTest(unittest.TestCase):
    """Idea 943100ef2622: a ledger or memory that would not read loaded as {}
    and was saved over - everything the critic had learned, gone."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.lib = ig.Library(self.dir)
        self.path = os.path.join(self.dir, ig.CRITIC_LEDGER)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_a_change_is_read_changed_and_written(self):
        ig.change_kept(self.lib, ig.CRITIC_LEDGER, lambda led: dict(led, a=1))
        ig.change_kept(self.lib, ig.CRITIC_LEDGER, lambda led: dict(led, b=2))
        self.assertEqual(ig.load_critic_ledger(self.lib), {"a": 1, "b": 2})
        self.assertEqual(os.listdir(self.dir), [ig.CRITIC_LEDGER])   # no temp left

    def test_a_file_that_will_not_parse_is_set_aside_not_written_over(self):
        with open(self.path, "w", encoding="utf-8") as f:
            f.write('{"fixes": {"flux-dev": ')             # cut off
        from unittest.mock import patch
        with patch.object(ig.doctor, "log_error") as logged:
            ig.change_kept(self.lib, ig.CRITIC_LEDGER, lambda led: dict(led, a=1))
        with open(self.path + ".broken", encoding="utf-8") as f:
            self.assertEqual(f.read(), '{"fixes": {"flux-dev": ')
        self.assertEqual(ig.load_critic_ledger(self.lib), {"a": 1})
        self.assertIn(".broken", logged.call_args.args[0])

    def test_a_file_that_will_not_open_is_left_alone(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"kept": True}, f)
        from unittest.mock import patch
        real = open

        def held(path, *a, **k):                     # the other process has it
            if os.path.abspath(str(path)) == os.path.abspath(self.path) and "r" in (a[:1] or ("r",))[0]:
                raise PermissionError(13, "in use")
            return real(path, *a, **k)
        with patch("builtins.open", held), patch.object(ig.doctor, "log_error"):
            ig.change_kept(self.lib, ig.CRITIC_LEDGER, lambda led: {"erased": True})
        self.assertEqual(ig.load_critic_ledger(self.lib), {"kept": True})
        self.assertFalse(os.path.exists(self.path + ".broken"))


if __name__ == "__main__":
    unittest.main()
