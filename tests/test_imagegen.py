"""The Image Studio: the template adapter, composing a job, routing, the
library on disk, the job queue and history against a fake ComfyUI, and the tab
itself built in process. Nothing here touches the network or a GPU."""

import base64
import json
import os
import sys
import tempfile
import threading
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import studio_imagegen as ig  # noqa: E402

# A 1x1 PNG, for the pictures the fake ComfyUI "makes".
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")

FLUX_FILES = {"diffusion_models": {"flux1-dev.safetensors", "z_image_turbo_bf16.safetensors"},
              "text_encoders": {"clip_l.safetensors", "t5xxl_fp16.safetensors",
                                "t5xxl_fp8_e4m3fn.safetensors",
                                "t5xxl_fp8_e4m3fn_scaled.safetensors", "qwen_3_4b.safetensors"},
              "vae": {"ae.safetensors"}, "loras": {"gavin.safetensors", "sx70.safetensors",
                                                   "xl_thing.safetensors"},
              "checkpoints": set(), "clip_vision": set(), "style_models": set(),
              "controlnet": set(), "upscale_models": set()}


class FakeClient:
    """Answers what Studio asks of a ComfyUI, and records what it was sent."""
    instances = []
    down = set()                      # backend ids that do not answer
    hold = None                       # an Event a run waits on, to test cancel

    def __init__(self, backend):
        self.backend, self.url = backend, backend["url"].rstrip("/")
        self.graphs, self.uploads, self.cancelled, self.freed = [], [], [], 0
        FakeClient.instances.append(self)

    def health(self):
        if self.backend["id"] in FakeClient.down:
            return {"ok": False, "detail": "Cannot reach it", "queue": 0}
        return {"ok": True, "detail": "ComfyUI test", "vram_free": 20e9,
                "vram_total": 24e9, "queue": 0}

    def inventory(self):
        return {k: set(v) for k, v in FLUX_FILES.items()}

    def node_types(self):
        return {"UNETLoader", "DualCLIPLoader", "CLIPLoader", "VAELoader", "LoraLoader",
                "LoraLoaderModelOnly", "CLIPTextEncode", "FluxGuidance",
                "ConditioningZeroOut", "EmptySD3LatentImage", "KSampler", "VAEDecode",
                "SaveImage", "LoadImage", "VAEEncode", "ImageScaleBy", "VAEEncodeTiled",
                "VAEDecodeTiled", "ModelSamplingAuraFlow"}

    def upload_image(self, path):
        self.uploads.append(path)
        return "studio_" + os.path.basename(path)

    def queue_workflow(self, graph):
        self.graphs.append(graph)
        return "pid%d" % len(self.graphs)

    def listen_for_progress(self, pid, on_event, stop=None, timeout=0):
        on_event("queued", 0)
        on_event("executing", "1")
        on_event("executing", "40")
        for i in range(1, 5):
            on_event("progress", (i, 4, "40"))
        on_event("executing", "41")
        if FakeClient.hold is not None:
            while not FakeClient.hold.wait(0.02):
                if stop and stop():
                    return None
        if stop and stop():
            return None
        return {"status": {"completed": True}, "outputs": {"9": {"images": [
            {"filename": "ImageStudio_00001_.png", "subfolder": "", "type": "output"}]}}}

    def fetch(self, f):
        return PNG

    def cancel_job(self, pid):
        self.cancelled.append(pid)
        return True

    def free(self):
        self.freed += 1
        return True

    def get_queue(self):
        return {"queue_running": [], "queue_pending": []}


class FaceClient(FakeClient):
    """A ComfyUI with SAM3: the first run reports one small face, the second
    run is the face pass."""
    NODES = FakeClient.node_types(None) | ig.FACE_NODES
    fail_pass = False

    def inventory(self):
        return dict(super().inventory(), checkpoints={"sam3.pt"})

    def node_types(self):
        return set(FaceClient.NODES)

    def listen_for_progress(self, pid, on_event, stop=None, timeout=0):
        graph = self.graphs[int(pid[3:]) - 1]
        if "fs" in graph:
            on_event("progress", (1, 8, "fc1_4"))
            if FaceClient.fail_pass:
                return {"status": {"messages": [["execution_error", {
                    "node_id": "fc1_4", "node_type": "KSampler",
                    "exception_type": "RuntimeError", "exception_message": "boom"}]]},
                    "outputs": {}}
            return {"status": {"completed": True}, "outputs": {"fs": {"images": [
                {"filename": "faces_00001_.png", "subfolder": "ImageStudio", "type": "output"}]}}}
        entry = super().listen_for_progress(pid, on_event, stop, timeout)
        entry["outputs"].update({
            "fd4": {"text": [json.dumps([[{"x": 400, "y": 300, "width": 90, "height": 110}]])]},
            "fd6": {"text": ["1024"]}, "fd7": {"text": ["1024"]}})
        return entry


def settle(jobs, seconds=5):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if all(j.status in ig.FINISHED for j in jobs):
            return
        time.sleep(0.01)
    raise AssertionError("jobs did not finish: %s" % [(j.status, j.detail) for j in jobs])


class TempStudioMixin:
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        FakeClient.instances, FakeClient.down, FakeClient.hold = [], set(), None
        self.notified = []
        self.studio = ig.Studio(root=self.dir, notify=self.notified.append,
                                client_factory=FakeClient)
        lib = self.studio.lib
        lib.save("loras", [
            {"id": "gavin", "file": "gavin.safetensors", "name": "Gavin Identity",
             "category": "Identity", "trigger": "GAVINPERSON", "family": "flux1"},
            {"id": "sx70", "file": "sx70.safetensors", "name": "SX-70",
             "category": "Camera / Film", "family": "flux1", "strength": 0.55},
            {"id": "xl", "file": "xl_thing.safetensors", "name": "XL thing",
             "category": "Style", "family": "sdxl"},
            {"id": "mystery", "file": "mystery.safetensors", "name": "Mystery"},
        ])
        lib.save("identities", [
            {"id": "gavin", "name": "Gavin", "lora": "gavin", "trigger": "GAVINPERSON",
             "strength": 0.85},
            {"id": "lilya", "name": "Lilya", "trigger": "LILYAPERSON", "strength": 0.8}])
        styles = lib.all("styles") + [{"id": "sx70-authentic", "name": "SX-70 Authentic",
                                       "lora": "sx70", "strength": 0.55,
                                       "prompt": "SX-70 instant film.",
                                       "families": ["flux1"]}]
        lib.save("styles", styles)
        # The layered FLUX workflow (LoRAs, references, refine), kept for when
        # the baseline is known good; the engine's layering is tested on it.
        lib.save("models", lib.all("models") + [
            {"id": "flux-hq", "label": "FLUX.1 [dev] HQ", "family": "flux1",
             "workflow": "flux_hq",
             "values": {"model": "flux1-dev.safetensors", "weight_dtype": "default",
                        "clip_l": "clip_l.safetensors", "t5": "t5xxl_fp16.safetensors",
                        "vae": "ae.safetensors",
                        "clip_vision": "sigclip_vision_patch14_384.safetensors",
                        "style_model": "flux1-redux-dev.safetensors"},
             "backends": {"3090": {"t5": "t5xxl_fp8_e4m3fn.safetensors",
                                   "weight_dtype": "fp8_e4m3fn"}},
             "defaults": {"steps": 28, "guidance": 3.5, "sampler": "euler",
                          "scheduler": "simple", "width": 1024, "height": 1024}}])

    def backend(self, bid):
        return self.studio.backend(bid)


class TestFill(unittest.TestCase):
    def wf(self):
        return ig.load_workflow("flux_hq")

    def test_whole_placeholders_keep_their_type(self):
        g = ig.fill(self.wf(), {"model": "m", "clip_l": "c", "t5": "t", "vae": "v",
                                "prompt": "a fox", "seed": 7, "width": 832})
        self.assertEqual(g["40"]["inputs"]["seed"], 7)
        self.assertEqual(g["20"]["inputs"]["width"], 832)
        self.assertEqual(g["10"]["inputs"]["text"], "a fox")
        self.assertEqual(g["40"]["inputs"]["model"], ["1", 0])   # no LoRAs: the loader

    def test_lora_chain_goes_between_loader_and_consumers(self):
        g = ig.fill(self.wf(), {"model": "m", "clip_l": "c", "t5": "t", "vae": "v",
                                "prompt": "p", "seed": 1},
                    [("a.safetensors", 0.8), ("b.safetensors", 0.5)])
        self.assertEqual(g["lora1"]["inputs"]["model"], ["1", 0])
        self.assertEqual(g["lora1"]["inputs"]["clip"], ["2", 0])
        self.assertEqual(g["lora2"]["inputs"]["model"], ["lora1", 0])
        self.assertEqual(g["lora2"]["inputs"]["strength_model"], 0.5)
        self.assertEqual(g["40"]["inputs"]["model"], ["lora2", 0])
        self.assertEqual(g["10"]["inputs"]["clip"], ["lora2", 1])

    def test_model_only_chain_for_a_template_without_clip(self):
        g = ig.fill(ig.load_workflow("zimage_hq"), {"model": "m", "encoder": "e", "vae": "v",
                                                    "prompt": "p", "seed": 1},
                    [("z.safetensors", 1.0)])
        self.assertEqual(g["lora1"]["class_type"], "LoraLoaderModelOnly")
        self.assertEqual(g["4"]["inputs"]["model"], ["lora1", 0])

    def test_optional_nodes_follow_their_values(self):
        base = {"model": "m", "clip_l": "c", "t5": "t", "vae": "v", "prompt": "p", "seed": 1}
        plain = ig.fill(self.wf(), base)
        self.assertNotIn("21", plain)
        self.assertNotIn("45", plain)
        self.assertEqual(plain["9"]["inputs"]["images"], ["41", 0])
        full = ig.fill(self.wf(), dict(base, source_image="in.png", refine=True))
        self.assertNotIn("20", full)
        self.assertEqual(full["40"]["inputs"]["latent_image"], ["22", 0])
        self.assertEqual(full["9"]["inputs"]["images"], ["45", 0])

    def test_a_missing_value_is_named(self):
        with self.assertRaises(ig.TemplateError) as cm:
            ig.fill(self.wf(), {"clip_l": "c", "t5": "t", "vae": "v", "prompt": "p", "seed": 1})
        self.assertIn("model", str(cm.exception))

    def test_a_link_to_a_dropped_node_is_refused(self):
        wf = {"id": "bad", "graph": {
            "1": {"_when": "x", "class_type": "A", "inputs": {}},
            "2": {"class_type": "B", "inputs": {"in": ["1", 0]}}}}
        with self.assertRaises(ig.TemplateError):
            ig.fill(wf, {})

    def test_every_shipped_template_fills(self):
        for wf in ig.list_workflows():
            vals = {k: "x.safetensors" for k in (wf.get("files") or {})}
            vals.update(prompt="p", seed=1)
            loras = [("l.safetensors", 0.5)] if wf.get("lora_chain") else []
            for extra in ({}, {"refine": True, "source_image": "s.png"}):
                g = ig.fill(wf, dict(vals, **extra), loras)
                self.assertTrue(any(n["class_type"] == "SaveImage" for n in g.values()),
                                wf["id"])

    def test_the_flux_baseline_is_plain_text_to_image(self):
        wf = ig.load_workflow("flux_dev_baseline")
        g = ig.fill(wf, {"model": "flux1-dev.safetensors", "clip_l": "c", "t5": "t",
                         "vae": "ae", "prompt": "a fox", "seed": 9, "width": 832,
                         "height": 1216, "steps": 20, "guidance": 2.5,
                         "filename_prefix": "ImageStudio/job1"})
        classes = sorted(n["class_type"] for n in g.values())
        self.assertEqual(classes, sorted(
            ["UNETLoader", "DualCLIPLoader", "VAELoader", "CLIPTextEncode", "FluxGuidance",
             "ConditioningZeroOut", "EmptySD3LatentImage", "KSampler", "VAEDecode",
             "SaveImage"]))
        self.assertEqual((g["40"]["inputs"]["seed"], g["40"]["inputs"]["steps"]), (9, 20))
        self.assertEqual(g["11"]["inputs"]["guidance"], 2.5)
        self.assertEqual((g["20"]["inputs"]["width"], g["20"]["inputs"]["height"]), (832, 1216))
        self.assertEqual(g["9"]["inputs"]["filename_prefix"], "ImageStudio/job1")
        # With no LoRA and no refine, the layers leave the known-good graph as it was.
        self.assertEqual(g["40"]["inputs"]["model"], ["1", 0])
        self.assertEqual(g["10"]["inputs"]["clip"], ["2", 0])
        self.assertEqual(g["9"]["inputs"]["images"], ["41", 0])

    def test_face_boxes_and_crops(self):
        entry = {"outputs": {"fd4": {"text": [json.dumps([[
            {"x": 100, "y": 100, "width": 80, "height": 100},
            {"x": 600, "y": 50, "width": 8, "height": 9},            # texture, not a face
            {"x": 0, "y": 0, "width": 700, "height": 800}]])]},    # already full size
            "fd6": {"text": ["1024"]}, "fd7": {"text": ["1024"]}}}
        w, h, boxes = ig.face_boxes(entry)
        self.assertEqual((w, h, len(boxes)), (1024, 1024, 2))
        crops = ig.face_crops(w, h, boxes)
        self.assertEqual(len(crops), 1)
        self.assertEqual(crops[0]["width"], 200)
        self.assertIsNone(ig.face_boxes({"outputs": {}}))

    def test_the_face_graph_keeps_only_what_the_redraw_needs(self):
        wf = ig.load_workflow("flux_dev_baseline")
        crops = [{"x": 10, "y": 20, "width": 200, "height": 200},
                 {"x": 500, "y": 20, "width": 300, "height": 300}]
        g = ig.face_graph(wf, {"model": "m", "clip_l": "c", "t5": "t", "vae": "v",
                               "prompt": "p", "seed": 1, "face_prompt": "a face",
                               "refine": True},
                          [("gavin.safetensors", 0.8)], "pic.png [output]", crops, "oval.png",
                          "ImageStudio/x_faces")
        classes = {n["class_type"] for n in g.values()}
        self.assertFalse(classes & {"EmptySD3LatentImage", "VAEEncodeTiled"})
        for nid in ("20", "40", "41", "44", "9", "10", "11", "12"):
            self.assertNotIn(nid, g)
        self.assertEqual(g["fc1_4"]["inputs"]["model"], ["lora1", 0])
        self.assertEqual(g["f10"]["inputs"]["clip"], ["lora1", 1])
        self.assertEqual(g["fc2_4"]["inputs"]["positive"], ["f11", 0])
        self.assertEqual(g["fc2_1"]["inputs"]["image"], ["fc1_9", 0])   # composited so far
        self.assertEqual(g["fs"]["inputs"]["images"], ["fc2_9", 0])
        self.assertEqual((g["fc1_4"]["inputs"]["steps"], g["fc1_4"]["inputs"]["denoise"]),
                         (20, 0.4))

    def test_the_flux_baseline_takes_loras_and_a_refine_pass(self):
        wf = ig.load_workflow("flux_dev_baseline")
        g = ig.fill(wf, {"model": "m", "clip_l": "c", "t5": "t", "vae": "v", "prompt": "p",
                         "seed": 1, "refine": True}, [("gavin.safetensors", 0.85)])
        self.assertEqual(g["lora1"]["class_type"], "LoraLoader")
        self.assertEqual(g["40"]["inputs"]["model"], ["lora1", 0])
        self.assertEqual(g["44"]["inputs"]["model"], ["lora1", 0])
        self.assertEqual(g["10"]["inputs"]["clip"], ["lora1", 1])
        self.assertEqual(g["9"]["inputs"]["images"], ["45", 0])


class TestCompose(TempStudioMixin, unittest.TestCase):
    def plan(self, backend="5090", inventory=FLUX_FILES, nodes=None, **kw):
        s = ig.default_settings()
        s["model"] = "flux-hq"
        s.update(kw)
        return ig.compose(s, self.studio.lib, self.backend(backend), inventory, nodes=nodes)

    def test_flux_applies_identity_and_style_loras_and_refines(self):
        p = self.plan(model="flux-dev", scene="On a pier.", preset="hq_final",
                      identities=[{"id": "gavin", "strength": 0.9}],
                      loras=[{"id": "sx70", "strength": 0.5}])
        self.assertEqual(p.errors, [])
        self.assertEqual(p.loras, [("gavin.safetensors", 0.9), ("sx70.safetensors", 0.5)])
        self.assertTrue(p.values["refine"])
        self.assertTrue(p.prompt.startswith("GAVINPERSON"), p.prompt)

    def test_the_face_pass_needs_sam3_and_says_so(self):
        p = self.plan(model="flux-dev", scene="x", preset="hq_final")
        self.assertFalse(p.values["face_detail"])
        self.assertTrue(any("no SAM3 checkpoint" in w for w in p.warnings), p.warnings)
        p = self.plan(model="flux-dev", scene="x")              # not asked: not said
        self.assertFalse(any("SAM3" in w for w in p.warnings), p.warnings)

    def test_the_face_pass_is_planned_with_the_person_in_its_prompt(self):
        inv = dict(FLUX_FILES, checkpoints={"sam3.pt"})
        p = self.plan(model="flux-dev", scene="On a pier.", preset="identity",
                      identities=["gavin"], inventory=inv, nodes=FaceClient.NODES)
        self.assertEqual(p.errors, [])
        self.assertTrue(p.values["face_detail"])
        self.assertEqual(p.values["sam3"], "sam3.pt")
        self.assertIn("GAVINPERSON", p.values["face_prompt"])
        p = self.plan(model="flux-dev", scene="x", preset="identity", identities=["gavin"],
                      inventory=inv, nodes=FaceClient.NODES - {"SAM3_Detect"})
        self.assertFalse(p.values["face_detail"])
        self.assertTrue(any("SAM3_Detect" in w for w in p.warnings), p.warnings)
        p = self.plan(model="flux-dev", scene="x", preset="identity", identities=["gavin"],
                      inventory=inv, face_detail=False)
        self.assertFalse(p.values["face_detail"])

    def test_missing_nodes_are_named(self):
        p = self.plan(model="flux-dev", scene="x", nodes={"UNETLoader"})
        self.assertTrue(any("FluxGuidance" in e and "KSampler" in e for e in p.errors),
                        p.errors)

    def test_person_style_scene(self):
        p = self.plan(identities=[{"id": "gavin", "strength": 0.9}], style="sx70-authentic",
                      scene="At Munich Oktoberfest, raising a stein.", seed=3)
        self.assertEqual(p.errors, [])
        self.assertTrue(p.prompt.startswith("GAVINPERSON. At Munich Oktoberfest"))
        self.assertIn("SX-70 instant film", p.prompt)
        self.assertEqual(p.loras, [("gavin.safetensors", 0.9), ("sx70.safetensors", 0.55)])
        self.assertEqual(p.values["seed"], 3)

    def test_person_attributes_and_camera(self):
        p = self.plan(style="none", scene="On a pier at dusk.", subject="a woman in her 30s",
                      hair="auburn", eyes="green", build="slim", traits="freckles",
                      camera="85mm, shallow depth of field", anatomy=False)
        self.assertEqual(p.errors, [])
        self.assertEqual(p.prompt, "a woman in her 30s, slim build, green eyes, auburn hair, "
                                   "freckles. On a pier at dusk. 85mm, shallow depth of field.")

    def test_attributes_keep_their_own_nouns(self):
        self.assertEqual(ig.person_text({"hair": "long black hair", "eyes": "hazel eyes",
                                         "build": "about 70 kg"}),
                         "about 70 kg, hazel eyes, long black hair")

    def test_the_whole_character(self):
        self.assertEqual(ig.person_text({
            "subject": "a woman", "age": "in their 30s", "skin": "olive", "weight": -1,
            "stature": 2, "eyes": "green", "facial_hair": "", "hair": "auburn",
            "hair_style": "long wavy", "traits": "freckles", "expression": "neutral",
            "gaze": "looking at the camera", "top": "knit sweater", "bottom": "blue jeans",
            "footwear": "ankle boots", "accessories": "glasses, necklace"}),
            "a woman, in their 30s, olive skin, slim, tall, green eyes, long wavy auburn hair, "
            "freckles, neutral expression, looking at the camera, wearing knit sweater, "
            "blue jeans and ankle boots, with glasses and necklace")

    def test_hair_reads_as_one_phrase(self):
        self.assertEqual(ig.hair_text({"hair": "auburn", "hair_style": "in a bun"}),
                         "auburn hair in a bun")
        self.assertEqual(ig.hair_text({"hair": "grey", "hair_style": "buzz cut"}), "grey buzz cut")
        self.assertEqual(ig.hair_text({"hair": "black", "hair_style": "bald"}), "bald")
        self.assertEqual(ig.hair_text({"hair_style": "curly"}), "curly hair")

    def test_sliders_say_nothing_in_the_middle(self):
        self.assertEqual(ig.slider_word("weight", 0), "")
        self.assertEqual(ig.slider_word("weight", 3), "very heavyset")
        self.assertEqual(ig.slider_word("muscle", 9), "very muscular")      # clamped
        self.assertEqual(ig.slider_word("stature", -2), "short")

    def test_picks_toggle(self):
        self.assertEqual(ig.toggle("", "glasses", True), "glasses")
        self.assertEqual(ig.toggle("glasses", "necklace", True), "glasses, necklace")
        self.assertEqual(ig.toggle("Glasses, necklace", "glasses", True), "necklace")
        self.assertEqual(ig.toggle("sad", "angry", False), "angry")
        self.assertEqual(ig.toggle("angry", "angry", False), "")

    def test_every_expression_has_its_emoji(self):
        for pick in ig.SLOTS["expression"][3]:
            self.assertIn(pick, ig.EMOJI)
            self.assertTrue(ig.pick_label(pick).endswith(" " + pick))
        p = self.plan(style="none", subject="a man", expression="laughing")
        self.assertNotIn(ig.EMOJI["laughing"], p.prompt)

    def test_anatomy_constants_for_every_person(self):
        p = self.plan(style="none", subject="a woman", scene="Waving hello.")
        self.assertIn("every person has exactly two hands, each with four fingers and a "
                      "thumb, two feet, two eyes and a proportionate body", p.prompt)
        self.assertIn("extra fingers", p.negative)
        p = self.plan(style="none", scene="Two dancers on a stage.")
        self.assertIn("four fingers and a thumb", p.prompt)          # the scene names people
        p = self.plan(style="none", scene="A red bicycle against a white wall.")
        self.assertNotIn("fingers", p.prompt + p.negative)            # nobody in it
        p = self.plan(style="none", subject="a woman", anatomy=False)
        self.assertNotIn("fingers", p.prompt + p.negative)

    def test_identity_joins_the_described_person(self):
        p = self.plan(identities=["gavin"], style="none", hair="grey", scene="Reading.")
        self.assertTrue(p.prompt.startswith("GAVINPERSON, grey hair. Reading."), p.prompt)

    def test_a_person_alone_is_enough(self):
        p = self.plan(style="none", subject="an old fisherman", anatomy=False)
        self.assertEqual(p.errors, [])
        self.assertEqual(p.prompt, "an old fisherman.")

    def test_item_pictures_need_a_workflow_that_takes_them(self):
        pic = os.path.join(self.dir, "glasses.png")
        open(pic, "wb").close()
        p = self.plan(style="none", subject="a man", accessories="glasses",
                      item_refs={"glasses": pic, "hat": pic})
        self.assertEqual(p.images, {})
        self.assertTrue(any("Pictures of the glasses" in w and "no item reference input" in w
                            for w in p.warnings), p.warnings)
        self.assertFalse(any("hat" in w for w in p.warnings))          # not worn today

    def test_a_character_keeps_its_look_not_its_expression(self):
        rec = ig.clean_character({"name": "Mara", "identity": "lilya",
                                  "looks": {"hair": "auburn", "weight": -1, "muscle": 0,
                                            "expression": "sad", "top": "hoodie",
                                            "nonsense": "x"},
                                  "item_refs": {"hoodie": "C:/h.png", "": "x"}})
        self.assertEqual(rec["id"], "mara")
        self.assertEqual(rec["looks"], {"hair": "auburn", "weight": -1, "top": "hoodie"})
        self.assertEqual(rec["item_refs"], {"hoodie": "C:/h.png"})

    def test_random_looks_fill_the_creator(self):
        import random
        looks = ig.random_looks(random.Random(4))
        self.assertTrue(looks["subject"])
        self.assertNotIn("expression", looks)
        self.assertTrue(all(k in ig.CHARACTER_KEYS for k in looks))
        only = ig.random_looks(random.Random(4), sections=["Hair"])
        self.assertEqual(set(only), {"hair", "hair_style"})

    def test_two_people_in_one_picture(self):
        p = self.plan(identities=["gavin", "lilya"], scene="Dancing.")
        self.assertTrue(p.prompt.startswith("GAVINPERSON and LILYAPERSON. Dancing."))

    def test_identity_carries_no_style(self):
        p = self.plan(identities=["gavin"], scene="Portrait.", style="none")
        self.assertEqual([m["id"] for m in p.lora_meta], ["gavin"])
        self.assertNotIn("SX-70", p.prompt)

    def test_incompatible_lora_is_left_out_with_a_warning(self):
        p = self.plan(scene="x", loras=[{"id": "xl", "strength": 0.7}])
        self.assertEqual(p.loras, [])
        self.assertTrue(any("SDXL" in w and "left out" in w for w in p.warnings), p.warnings)

    def test_unknown_family_is_applied_with_a_warning(self):
        inv = dict(FLUX_FILES, loras=FLUX_FILES["loras"] | {"mystery.safetensors"})
        p = self.plan(inventory=inv, scene="x", loras=[{"id": "mystery", "strength": 0.4}])
        self.assertEqual(p.loras, [("mystery.safetensors", 0.4)])
        self.assertTrue(any("no model family" in w for w in p.warnings))

    def test_a_lora_missing_from_the_backend_is_named(self):
        p = self.plan(scene="x", loras=[{"id": "mystery", "strength": 0.4}])
        self.assertTrue(any("not on 5090" in w for w in p.warnings), p.warnings)

    def test_missing_model_file_is_an_error_with_the_fix(self):
        inv = dict(FLUX_FILES, diffusion_models=set())
        p = self.plan(inventory=inv, scene="x")
        self.assertTrue(any("flux1-dev.safetensors" in e and "Models" in e
                            and "models/diffusion_models" in e for e in p.errors), p.errors)

    def test_per_backend_filenames(self):
        p = self.plan(backend="3090", scene="x")
        self.assertEqual(p.values["t5"], "t5xxl_fp8_e4m3fn.safetensors")
        self.assertEqual(p.values["weight_dtype"], "fp8_e4m3fn")
        self.assertEqual(p.values["encoder_device"], "cpu")
        self.assertEqual(self.plan(scene="x").values["t5"], "t5xxl_fp16.safetensors")

    def test_per_backend_family_decides_compatibility(self):
        models = self.studio.lib.all("models")
        next(m for m in models if m["id"] == "flux-hq")["backends"]["5090"] = {
            "model": "flux2-dev.safetensors", "family": "flux2"}
        self.studio.lib.save("models", models)
        p = self.plan(inventory=None, identities=["gavin"], scene="x")
        self.assertEqual(p.family, "flux2")
        self.assertEqual(p.loras, [])
        self.assertTrue(any("FLUX.1 LoRA" in w for w in p.warnings), p.warnings)

    def test_references_go_where_the_workflow_takes_them(self):
        src = os.path.join(self.dir, "src.png")
        with open(src, "wb") as f:
            f.write(PNG)
        p = self.plan(scene="x", references={"source": src, "pose": src})
        self.assertEqual(p.images, {"source_image": src})
        self.assertEqual(p.values["denoise"], 0.7)
        self.assertTrue(any(w.startswith("Pose reference") for w in p.warnings), p.warnings)

    def test_a_pose_goes_through_the_controlnet_on_flux(self):
        src = os.path.join(self.dir, "pose.png")
        with open(src, "wb") as f:
            f.write(PNG)
        cn = "FLUX.1-dev-ControlNet-Union-Pro-2.0.safetensors"
        inv = dict(FLUX_FILES, controlnet={cn})
        p = self.plan(model="flux-dev", inventory=inv, scene="x", references={"pose": src},
                      pose={"points": [[0.5, 0.5]] * 18, "strength": 0.75})
        self.assertEqual(p.errors, [])
        self.assertEqual(p.images, {"pose_image": src})
        self.assertEqual((p.values["controlnet"], p.values["pose_strength"]), (cn, 0.75))
        g = ig.fill(p.workflow, dict(p.values, pose_image="pose.png"))
        self.assertEqual(g["52"]["class_type"], "ControlNetApplyAdvanced")
        self.assertEqual(g["40"]["inputs"]["positive"], ["52", 0])
        self.assertEqual(g["40"]["inputs"]["negative"], ["52", 1])
        self.assertEqual(g["52"]["inputs"]["strength"], 0.75)
        # Without the ControlNet on the backend the pose is left out, in words.
        p = self.plan(model="flux-dev", scene="x", references={"pose": src})
        self.assertEqual(p.errors, [])
        self.assertEqual(p.images, {})
        self.assertTrue(any("Pose reference" in w and cn in w for w in p.warnings),
                        p.warnings)
        # And with no pose, nothing of it is recorded or in the graph.
        p = self.plan(model="flux-dev", inventory=inv, scene="x")
        self.assertNotIn("controlnet", p.values)
        g = ig.fill(p.workflow, p.values)
        self.assertFalse({"50", "51", "52"} & set(g))
        self.assertEqual(g["40"]["inputs"]["positive"], ["11", 0])

    def test_redux_reference_needs_its_files(self):
        src = os.path.join(self.dir, "style.png")
        with open(src, "wb") as f:
            f.write(PNG)
        p = self.plan(scene="x", references={"style": src})
        self.assertEqual(p.images, {})
        self.assertTrue(any("lacks" in w for w in p.warnings))
        inv = dict(FLUX_FILES, clip_vision={"sigclip_vision_patch14_384.safetensors"},
                   style_models={"flux1-redux-dev.safetensors"})
        self.assertEqual(self.plan(inventory=inv, scene="x", references={"style": src}).images,
                         {"reference_image": src})

    def test_refine_is_capped_by_the_backend(self):
        p = self.plan(backend="3090", scene="x", preset="hq_final", width=1344, height=1344)
        self.assertTrue(p.values["refine"])
        self.assertLess(p.values["upscale"], 2.0)
        self.assertTrue(any("megapixels" in n for n in p.notes))

    def test_identity_preset_wants_a_person(self):
        self.assertTrue(self.plan(preset="identity", scene="x").errors)


class TestRouting(TempStudioMixin, unittest.TestCase):
    def test_role_decides_and_falls_back(self):
        bs = self.studio.backends()
        self.assertEqual([b["id"] for b in ig.route("identity", bs, {})], ["5090", "3090"])
        self.assertEqual([b["id"] for b in ig.route("upscale", bs, {})], ["3090", "5090"])
        down = {"5090": {"ok": False}}
        self.assertEqual([b["id"] for b in ig.route("identity", bs, down)], ["3090"])

    def test_disabled_and_modelless_backends_are_skipped(self):
        bs = [dict(b) for b in self.studio.backends()]
        bs[0]["enabled"] = False
        self.assertEqual([b["id"] for b in ig.route("identity", bs, {})], ["3090"])
        self.assertEqual(ig.route("identity", bs, {}, has_model=lambda b: False), [])

    def test_a_batch_spreads_over_both_machines(self):
        jobs = self.studio.submit(dict(ig.default_settings(), scene="x", batch=4))
        settle(jobs)
        self.assertEqual(sorted({j.backend["id"] for j in jobs}), ["3090", "5090"])
        self.assertEqual([j.settings["seed"] for j in jobs],
                         list(range(jobs[0].settings["seed"], jobs[0].settings["seed"] + 4)))

    def test_nothing_online_says_why(self):
        FakeClient.down = {"5090", "3090"}
        with self.assertRaises(ig.ComfyError) as cm:
            self.studio.submit(dict(ig.default_settings(), scene="x"))
        self.assertIn("5090 Workstation is offline", str(cm.exception))

    def test_auto_prefers_the_5090_and_falls_back_only_to_a_ready_3090(self):
        s = dict(ig.default_settings(), model="flux-dev", scene="x")
        self.studio.check_all()
        b, why = self.studio.plan_route(s)
        self.assertEqual(b["id"], "5090")
        self.assertIn("Auto → 5090 Workstation", why)
        FakeClient.down = {"5090"}
        self.studio.check_all()
        b, why = self.studio.plan_route(s)
        self.assertEqual(b["id"], "3090")
        self.assertIn("5090 Workstation is offline", why)
        # The 3090 without FLUX's files: nothing, and the files are named.
        self.studio.inventories["3090"]["diffusion_models"] = set()
        b, why = self.studio.plan_route(s)
        self.assertIsNone(b)
        self.assertIn("3090 Server is missing flux1-dev.safetensors", why)
        with self.assertRaises(ig.ComfyError):
            self.studio.submit(s)

    def test_readiness_names_each_missing_file_and_where(self):
        self.studio.check_all()
        self.studio.inventories["3090"]["text_encoders"] = {"clip_l.safetensors"}
        FakeClient.down = {"5090"}
        self.studio.check(self.backend("5090"))
        r = self.studio.readiness(self.studio.lib.get("models", "flux-dev"))
        self.assertEqual(r["5090"][0], "offline")
        self.assertEqual(r["3090"][0], "missing")
        self.assertIn("t5xxl_fp8_e4m3fn_scaled.safetensors (the text encoder, in "
                      "ComfyUI/models/text_encoders)", r["3090"][1])
        self.assertNotIn("clip_l", r["3090"][1])


class TestJobs(TempStudioMixin, unittest.TestCase):
    def test_a_job_runs_and_lands_in_history(self):
        room = []
        self.studio.make_room = room.append
        src = os.path.join(self.dir, "src.png")
        with open(src, "wb") as f:
            f.write(PNG)
        jobs = self.studio.submit(dict(ig.default_settings(), scene="A fox", backend="3090",
                                       model="flux-hq", identities=["gavin"],
                                       references={"source": src}, seed=42))
        settle(jobs)
        job = jobs[0]
        self.assertEqual(job.status, "complete", job.detail)
        self.assertEqual([b["id"] for b in room], ["3090"])       # LM Studio cleared first
        client = FakeClient.instances[-1]
        self.assertEqual(client.uploads, [src])
        graph = client.graphs[0]
        self.assertEqual(graph["21"]["inputs"]["image"], "studio_src.png")
        self.assertEqual(graph["40"]["inputs"]["seed"], 42)
        self.assertTrue(os.path.isfile(job.outputs[0]))
        self.assertTrue({"uploading", "running"} <= {n.status for n in self.notified}
                        or self.notified)
        rec = self.studio.history.list()[0]
        for key in ("prompt", "negative", "seed", "model", "loras", "identities", "style",
                    "workflow", "backend", "sampler", "steps", "guidance", "width",
                    "height", "created", "settings", "graph"):
            self.assertIn(key, rec)
        self.assertEqual(rec["seed"], 42)
        self.assertEqual(rec["loras"][0]["file"], "gavin.safetensors")
        self.assertEqual(rec["backend"]["id"], "3090")
        self.assertEqual(rec["settings"]["seed_mode"], "fixed")
        self.assertEqual(client.freed, 1)                          # queue empty: VRAM back

    def test_a_flux_job_records_its_files_and_output_name(self):
        jobs = self.studio.submit(dict(ig.default_settings(), model="flux-dev", scene="x",
                                       backend="5090"))
        settle(jobs)
        self.assertEqual(jobs[0].status, "complete", jobs[0].detail)
        rec = self.studio.history.list()[0]
        self.assertEqual(rec["workflow"], "flux_dev_baseline")
        self.assertEqual(rec["model"]["files"], {
            "model": "flux1-dev.safetensors", "clip_l": "clip_l.safetensors",
            "t5": "t5xxl_fp16.safetensors", "vae": "ae.safetensors"})
        self.assertEqual(rec["graph"]["9"]["inputs"]["filename_prefix"],
                         "ImageStudio/flux_dev_baseline_" + jobs[0].id)

    def face_studio(self):
        self.studio = ig.Studio(root=self.dir, notify=self.notified.append,
                                client_factory=FaceClient)
        FaceClient.fail_pass = False

    def test_the_face_pass_redraws_each_face_with_the_identity_lora(self):
        self.face_studio()
        jobs = self.studio.submit(dict(ig.default_settings(), model="flux-dev",
                                       preset="identity", identities=["gavin"],
                                       scene="On a pier.", backend="5090", seed=5))
        settle(jobs)
        job = jobs[0]
        self.assertEqual(job.status, "complete", job.detail)
        client = FaceClient.instances[-1]
        first, second = client.graphs
        self.assertEqual(first["fd3"]["inputs"]["image"], first["9"]["inputs"]["images"])
        self.assertEqual(second["fi"]["inputs"]["image"],
                         "ImageStudio_00001_.png [output]")
        self.assertEqual(second["fc1_4"]["inputs"]["model"], ["lora1", 0])
        self.assertEqual(second["fc1_4"]["inputs"]["seed"], 6)
        self.assertEqual(second["fc1_4"]["inputs"]["denoise"], 0.4)
        self.assertEqual(second["fc1_2"]["inputs"]["width"], ig.FACE_EDIT)
        self.assertIn("GAVINPERSON", second["f10"]["inputs"]["text"])
        self.assertNotIn("40", second)            # the picture is loaded, not made again
        self.assertNotIn("fc2_1", second)
        rec = self.studio.history.list()[0]
        self.assertEqual(rec["face_detail"]["redrawn"], 1)
        self.assertEqual(rec["face_graph"]["fs"]["inputs"]["filename_prefix"],
                         "ImageStudio/flux_dev_baseline_%s_faces" % job.id)
        self.assertTrue(any("Face pass: 1 face" in n for n in rec["notes"]), rec["notes"])

    def test_a_failed_face_pass_keeps_the_picture(self):
        self.face_studio()
        FaceClient.fail_pass = True
        jobs = self.studio.submit(dict(ig.default_settings(), model="flux-dev",
                                       preset="identity", identities=["gavin"],
                                       scene="x", backend="5090"))
        settle(jobs)
        self.assertEqual(jobs[0].status, "complete", jobs[0].detail)
        rec = self.studio.history.list()[0]
        self.assertIsNone(rec["face_graph"])
        self.assertEqual(len(rec["images"]), 1)
        self.assertTrue(any("face pass failed" in w and "boom" in w for w in rec["warnings"]),
                        rec["warnings"])

    def test_progress_reads_as_queued_loading_sampling_decoding(self):
        statuses = []
        job = ig.Job(ig.default_settings(), self.backend("5090"))
        job.plan = ig.Plan()
        job.plan.workflow = ig.load_workflow("flux_dev_baseline")
        graph = job.plan.workflow["graph"]

        def say(status=None, detail=None, progress=None):
            if status and (not statuses or statuses[-1][0] != status):
                statuses.append((status, detail))
        on = self.studio._progress(job, graph, say)
        on("queued", 1)
        on("executing", "1")
        on("executing", "40")
        on("progress", (5, 20, "40"))
        on("executing", "41")
        self.assertEqual([s for s, _ in statuses],
                         ["queued", "loading", "sampling", "decoding"])
        self.assertIn("1 ahead", statuses[0][1])
        self.assertIn("step 5 of 20 · 25% · node 40 KSampler", statuses[2][1])
        on("socket", "No live progress: refused")
        self.assertIn("No live progress: refused", job.notes)

    def test_generate_again_reproduces_and_new_seed_varies(self):
        rec = {"seed": 5, "steps": 20, "guidance": 3.5, "width": 1024, "height": 768,
               "sampler": "euler", "scheduler": "simple",
               "backend": {"id": "5090"},
               "settings": dict(ig.default_settings(), seed=5, seed_mode="random",
                                batch=1, batch_of=4)}
        s = ig.again(rec)
        self.assertEqual((s["seed"], s["seed_mode"]), (5, "fixed"))
        self.assertEqual((s["steps"], s["height"], s["prefer_backend"]), (20, 768, "5090"))
        self.assertNotIn("batch_of", s)
        self.assertEqual(ig.again(rec, new_seed=True)["seed"], -1)
        jobs = self.studio.submit(s)
        settle(jobs)
        self.assertEqual(jobs[0].settings["seed"], 5)
        self.assertEqual(jobs[0].backend["id"], "5090")

    def test_cancel_while_running(self):
        FakeClient.hold = threading.Event()
        jobs = self.studio.submit(dict(ig.default_settings(), scene="x", backend="5090"))
        deadline = time.monotonic() + 3
        while jobs[0].prompt_id is None and time.monotonic() < deadline:
            time.sleep(0.01)
        self.studio.queue.cancel(jobs[0])
        settle(jobs)
        self.assertEqual(jobs[0].status, "cancelled")
        self.assertEqual(FakeClient.instances[-1].cancelled, [jobs[0].prompt_id])
        self.assertEqual(self.studio.history.list(), [])

    def test_cancel_before_it_starts(self):
        FakeClient.hold = threading.Event()
        jobs = self.studio.submit(dict(ig.default_settings(), scene="x", backend="5090",
                                       batch=2))
        self.studio.queue.cancel(jobs[1])
        FakeClient.hold.set()
        settle(jobs)
        self.assertEqual([j.status for j in jobs], ["complete", "cancelled"])

    def test_a_plan_error_fails_the_job_in_words(self):
        jobs = self.studio.submit(dict(ig.default_settings(), scene="", backend="5090"))
        settle(jobs)
        self.assertEqual(jobs[0].status, "failed")
        self.assertIn("Describe the scene", jobs[0].detail)


class TestErrors(unittest.TestCase):
    def test_a_refused_workflow_names_the_missing_file_not_the_whole_list(self):
        detail = {"error": {"type": "prompt_outputs_failed_validation",
                            "message": "Prompt outputs failed validation"},
                  "node_errors": {"1": {"class_type": "UNETLoader", "errors": [
                      {"message": "Value not in list",
                       "details": "unet_name: 'flux1-dev.safetensors' not in "
                                  "['a.safetensors', 'b.safetensors']"}]}}}
        text = ig.explain(detail)
        self.assertIn("node 1 (UNETLoader): Value not in list - unet_name "
                      "'flux1-dev.safetensors' is not there (it has 2 others)", text)
        self.assertNotIn("a.safetensors", text)

    def test_a_failed_run_says_node_exception_and_message(self):
        entry = {"status": {"status_str": "error", "messages": [
            ["execution_start", {}],
            ["execution_error", {"node_id": "40", "node_type": "KSampler",
                                 "exception_type": "torch.OutOfMemoryError",
                                 "exception_message": "CUDA out of memory.\nTried 2 GB"}]]}}
        (text,) = ig.run_errors(entry)
        self.assertIn("node 40 (KSampler): torch.OutOfMemoryError: CUDA out of memory. "
                      "Tried 2 GB", text)
        self.assertIn("ran out of memory", text)

    def test_something_else_on_the_port_is_not_comfyui(self):
        c = ig.ComfyUIClient({"id": "x", "name": "X", "url": "http://127.0.0.1:1"})
        c.get_json = lambda path, timeout=None: {"ok": True}
        c.get_queue = lambda: {}
        h = c.health()
        self.assertFalse(h["ok"])
        self.assertIn("not ComfyUI", h["detail"])

    def test_nothing_listening_here_says_so_and_how_to_start_it(self):
        c = ig.ComfyUIClient({"id": "x", "name": "X", "url": "http://127.0.0.1:9",
                              "start": r"D:\ComfyUI\start.cmd"})
        h = c.health()
        self.assertFalse(h["ok"])
        self.assertIn("no ComfyUI is running here", h["detail"])
        self.assertIn(r"Start it: D:\ComfyUI\start.cmd", h["detail"])


class TestLibrary(unittest.TestCase):
    def test_junk_on_disk_costs_the_record_not_the_list(self):
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "identities.json"), "w") as f:
            json.dump([{"name": "Gavin", "strength": "lots"}, "junk", {"no": "name"},
                       {"name": "Gavin"}], f)
        with open(os.path.join(d, "backends.json"), "w") as f:
            f.write("{not json")
        lib = ig.Library(d)
        self.assertEqual([i["id"] for i in lib.all("identities")], ["gavin", "gavin-2"])
        self.assertEqual(lib.all("identities")[0]["strength"], 0.85)
        self.assertEqual([b["id"] for b in lib.all("backends")], ["5090", "3090"])
        self.assertTrue(lib.problems)

    def test_scan_adds_unknown_loras_with_guesses(self):
        lib = ig.Library(tempfile.mkdtemp())
        n = lib.merge_loras("3090", ["gavin_identity_flux.safetensors",
                                     "kodak_portra_sdxl.safetensors"])
        self.assertEqual(n, 2)
        a, b = lib.all("loras")
        self.assertEqual((a["category"], a["family"]), ("Identity", "flux1"))
        self.assertEqual((b["category"], b["family"]), ("Camera / Film", "sdxl"))
        self.assertEqual(lib.merge_loras("3090", ["gavin_identity_flux.safetensors"]), 0)

    def test_references_are_copied_into_the_studio(self):
        d = tempfile.mkdtemp()
        src = os.path.join(d, "me.png")
        with open(src, "wb") as f:
            f.write(PNG)
        lib = ig.Library(os.path.join(d, "studio"))
        kept = lib.keep_reference(src, "Gavin")
        self.assertTrue(kept.startswith(lib.root))
        self.assertEqual(kept, lib.keep_reference(src, "Gavin"))

    def test_every_default_style_has_its_example_picture(self):
        """The form shows styles as pictures; a default without one is a
        blank tile."""
        lib = ig.Library(tempfile.mkdtemp())
        for st in lib.all("styles"):
            self.assertTrue(ig.style_example(st), st["id"])

    def test_a_styles_own_example_wins_and_a_missing_one_falls_back(self):
        d = tempfile.mkdtemp()
        own = os.path.join(d, "mine.png")
        with open(own, "wb") as f:
            f.write(PNG)
        self.assertEqual(ig.style_example(ig.clean_style(
            {"name": "Cinema", "example": own})), own)
        self.assertEqual(ig.style_example(ig.clean_style(
            {"name": "Cinema", "example": os.path.join(d, "gone.png")})),
            os.path.join(ig.STYLE_EXAMPLES_DIR, "cinema.png"))
        self.assertIsNone(ig.style_example(ig.clean_style({"name": "Brand new"})))


def _headless():
    try:
        import tkinter
        tkinter.Tk().destroy()
        return False
    except Exception:
        return True


@unittest.skipIf(_headless(), "no display")
class TestImageStudioTab(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import studio_chat
        cls.mod = studio_chat
        cls.dir = tempfile.mkdtemp()
        cls._real_settings = os.environ.get("STUDIO_SETTINGS")
        os.environ["STUDIO_SETTINGS"] = os.path.join(cls.dir, "settings.json")
        with open(os.environ["STUDIO_SETTINGS"], "w") as f:
            json.dump({"tabs": ["chat", "image-studio"]}, f)
        cls._saved = {n: getattr(studio_chat.Chat, n) for n in
                      ("_boot_host", "_read_icons", "_boot_session")}
        studio_chat.Chat._boot_host = lambda self, *a, **k: None
        studio_chat.Chat._read_icons = lambda self: None
        studio_chat.Chat._boot_session = lambda self, s: None
        cls._client = ig.ComfyUIClient
        ig.ComfyUIClient = FakeClient
        cls.app = studio_chat.Chat()
        for _ in range(10):
            cls.app.update()

    @classmethod
    def tearDownClass(cls):
        cls.app._quit()
        ig.ComfyUIClient = cls._client
        for n, fn in cls._saved.items():
            setattr(cls.mod.Chat, n, fn)
        if cls._real_settings is None:
            os.environ.pop("STUDIO_SETTINGS", None)
        else:
            os.environ["STUDIO_SETTINGS"] = cls._real_settings

    def pump(self, until, seconds=5):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.app.update()
            if until():
                return
            time.sleep(0.01)
        raise AssertionError("condition not reached")

    def tab(self):
        self.app._select(("image-studio"))
        self.app.update()
        s = self.app.sessions["image-studio"]
        self.assertIsNotNone(s.images)
        return s, s.images

    def test_the_tab_is_a_form_with_no_composer(self):
        s, ui = self.tab()
        self.assertTrue(s.ready)
        self.assertFalse(self.app.composer.winfo_ismapped())
        self.pump(lambda: all(b["id"] in ui.studio.health for b in ui.studio.backends()))

    def test_generate_runs_a_job_and_reuse_puts_it_back(self):
        s, ui = self.tab()
        ui.scene.delete("1.0", "end")
        ui.scene.insert("1.0", "A red bicycle against a white wall")
        ui.adv["seed"].set("1234")
        ui.random_seed.set(False)
        ui.settings["model"] = "z-image-turbo"
        ui.settings["backend"] = "auto"
        n = len(ui.jobs)
        ui.generate()
        self.pump(lambda: len(ui.jobs) > n and ui.jobs[0].status in ig.FINISHED)
        job = ui.jobs[0]
        self.assertEqual(job.status, "complete", job.detail)
        self.assertEqual(job.settings["seed"], 1234)
        self.pump(lambda: ui.selected is not None)
        ui._show_list("history")
        rec = ui.studio.history.list()[0]
        ui.scene.delete("1.0", "end")
        ui.adv["seed"].set("")
        ui.apply(rec["settings"])
        self.assertEqual(ui.scene.get("1.0", "end").strip(), "A red bicycle against a white wall")
        self.assertEqual(ui.adv["seed"].get(), "1234")
        self.assertFalse(ui.random_seed.get())
        ui._show_list("queue")

    def test_warnings_show_before_generate(self):
        s, ui = self.tab()
        ui.settings["model"] = "flux-dev"
        ui.studio.lib.save("loras", [{"id": "xl", "file": "xl_thing.safetensors",
                                      "name": "XL thing", "family": "sdxl"}])
        ui._rebuild_choices()
        ui._add_lora("xl")
        self.assertIn("left out", ui.warn.cget("text"))
        ui._drop_lora(ui.loras[0])
        self.assertNotIn("left out", ui.warn.cget("text"))

    def test_a_drawn_pose_becomes_the_pose_reference(self):
        s, ui = self.tab()
        ui.settings["model"] = "flux-dev"
        ui.adv["width"].set("832")
        ui.adv["height"].set("1216")
        ed = ui.edit_pose()
        self.app.update()
        self.assertEqual(ed.size, (832, 1216))

        class Ev:
            def __init__(self, x, y, state=0, delta=0):
                self.x, self.y, self.state, self.delta = x, y, state, delta
        wrist = list(ed.points[7])
        at = (int(wrist[0] * ed.vw), int(wrist[1] * ed.vh))
        ed._press(Ev(*at))
        ed._move(Ev(at[0] + 20, at[1] - 40))
        self.assertLess(ed.points[7][1], wrist[1])
        ed._toggle(Ev(*[int(c) for c in (ed.points[17][0] * ed.vw, ed.points[17][1] * ed.vh)]))
        self.assertIn(17, ed.hidden)
        ed._preset("walking")
        ed._undo()
        self.assertIn(17, ed.hidden)
        ed.strength.set(0.8)
        ed._use()
        self.app.update()
        path = ui.refs["pose"]
        self.assertTrue(os.path.isfile(path))
        self.assertIn("stick figure", ui.ref_labels["pose"].cget("text"))
        got = ui.collect()
        self.assertEqual((got["pose"]["strength"], got["pose"]["hidden"]), (0.8, [17]))
        # A new size refits the figure at Generate rather than stretching it.
        ui.adv["width"].set("1024")
        ui.adv["height"].set("1024")
        ui._fit_pose()
        self.assertEqual((ui.pose["width"], ui.pose["height"]), (1024, 1024))
        self.assertNotEqual(ui.refs["pose"], path)
        ui._clear_ref("pose")
        self.assertIsNone(ui.pose)
        self.assertIsNone(ui.collect()["pose"])
        ui.apply(got)
        self.assertEqual(ui.pose["hidden"], [17])
        self.assertEqual(ui.refs["pose"], path)
        ui._clear_ref("pose")
        for k in ("width", "height"):
            ui.adv[k].set("")

    def test_editors_open_and_save(self):
        s, ui = self.tab()
        for open_ in (ui.edit_styles, ui.edit_loras, ui.edit_backends, ui.edit_models):
            ed = open_()
            self.app.update()
            ed._pick()
            ed.win.destroy()
        ed = ui.edit_identities()
        ed._new()
        kind, var = ed.widgets["name"]
        var.set("Lilya")
        ed.widgets["trigger"][1].set("LILYAPERSON")
        ed.widgets["strength"][1].set("0.8")
        ed._save()
        ed.win.destroy()
        rec = ui.studio.lib.get("identities", "lilya")
        self.assertEqual((rec["trigger"], rec["strength"]), ("LILYAPERSON", 0.8))
        self.assertIn("lilya", ui.idents)                # the form picked it up
        with open(os.path.join(ui.studio.lib.root, "identities.json")) as f:
            self.assertIn("LILYAPERSON", f.read())

    def test_a_character_goes_from_the_creator_to_the_form_and_history(self):
        s, ui = self.tab()
        ed = ui.edit_characters()
        ed._new()
        ed.name.set("Mara")
        ed.identity.set("")
        ed.vars["hair"].set("auburn")
        ed.vars["accessories"].set("glasses")
        ed.sl["weight"].set(-1)
        ed._show("Accessories")
        self.app.update()
        pic = os.path.join(tempfile.mkdtemp(), "glasses.png")
        import tkinter as tk
        tk.PhotoImage(master=self.app, width=4, height=4).write(pic, format="png")
        ed._set_item("glasses", pic)
        ed._changed()
        self.assertIn("slim", ed.sheet.cget("text"))
        self.assertIn("with glasses", ed.sheet.cget("text"))
        for tab in ed.SECTIONS:
            ed._show(tab)
            self.app.update()
        ed._randomize()                               # every look tab, from Constants
        ed.vars["hair"].set("auburn")
        ed.vars["accessories"].set("glasses")
        ed.sl["weight"].set(-1)
        ed._use()
        ed.win.destroy()
        rec = ui.studio.lib.get("characters", "mara")
        self.assertEqual(rec["looks"]["weight"], -1)
        self.assertIn("glasses", rec["item_refs"])
        self.assertEqual(ui.settings["character"], "mara")
        self.assertEqual(ui.text["hair"].get(), "auburn")
        ui.text["expression"].set("")
        ui._show_looks("Expression")
        faces = [w for w in ui.look_box.winfo_children()]
        self.assertTrue(faces)
        s_ = ui.collect()
        self.assertEqual((s_["character"], s_["weight"], s_["accessories"]),
                         ("mara", -1, "glasses"))
        self.assertIn("glasses", s_["item_refs"])
        ui.text["hair"].set("")
        ui.sliders["weight"].set(0)
        ui.apply(s_)                                  # Reuse Settings brings it back
        self.assertEqual((ui.text["hair"].get(), ui.sliders["weight"].get()), ("auburn", -1))
        for name, _ in ig.LOOKS:
            ui._show_looks(name)
            self.app.update()
        ui.save_as_character().win.destroy()

    def test_the_form_says_where_it_goes_and_refuses_what_cannot_run(self):
        s, ui = self.tab()
        self.pump(lambda: all(b["id"] in ui.studio.inventories for b in ui.studio.backends()))
        ui.settings.update(model="flux-dev", backend="auto")
        ui.scene.delete("1.0", "end")
        ui.scene.insert("1.0", "A lighthouse")
        ui._recheck()
        self.assertIn("Will run on 5090 Workstation", ui.route_note.cget("text"))
        ed = ui.edit_models()
        self.app.update()
        panel = " ".join(w.cget("text") for w in ed.form.winfo_children()[0].winfo_children())
        self.assertIn("✓  5090 Workstation: every file and node is there", panel)
        ed.win.destroy()
        saved = {bid: set(inv["diffusion_models"]) for bid, inv in ui.studio.inventories.items()}
        try:
            for inv in ui.studio.inventories.values():
                inv["diffusion_models"].discard("flux1-dev.safetensors")
            ui._recheck()
            self.assertIn("flux1-dev.safetensors", ui.warn.cget("text"))
            n = len(ui.jobs)
            ui.generate()
            self.app.update()
            self.assertEqual(len(ui.jobs), n)          # nothing was sent
            self.assertIn("missing flux1-dev.safetensors", ui.note.cget("text"))
        finally:
            for bid, files in saved.items():
                ui.studio.inventories[bid]["diffusion_models"] = files
            ui._recheck()

    def test_a_job_row_shows_the_stages(self):
        s, ui = self.tab()
        ui.settings.update(model="flux-dev", backend="5090")
        ui.scene.delete("1.0", "end")
        ui.scene.insert("1.0", "A fox")
        n = len(ui.jobs)
        ui.generate()
        self.pump(lambda: len(ui.jobs) > n and ui.jobs[0].status == "complete")
        w = ui.rows[ui.jobs[0].id]
        self.assertEqual([l.cget("text") for l in w["stages"]],
                         ["Queued", "Loading", "Sampling", "Decoding", "Complete"])

    def test_a_settled_tab_animates_nothing(self):
        s, ui = self.tab()
        self.pump(lambda: all(j.status in ig.FINISHED for j in ui.jobs))
        self.app.update()
        self.assertFalse([k for k in self.app.anim if k[0] == "images-clock"])


if __name__ == "__main__":
    unittest.main()
