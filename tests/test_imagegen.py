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


class PulidClient(FaceClient):
    """FaceClient with PuLID installed and its weights."""

    def node_types(self):
        return set(FaceClient.NODES) | ig.PULID_NODES

    def get_json(self, path, timeout=None):
        assert path == "/object_info/PulidFluxModelLoader", path
        return {"PulidFluxModelLoader": {"input": {"required": {
            "pulid_file": [["pulid_flux_v0.9.1.safetensors"]]}}}}


class PasteClient(PulidClient):
    """PulidClient with the real-face paste node; `report` is what it says."""
    report = []

    def node_types(self):
        return super().node_types() | {ig.PASTE_NODE}

    def listen_for_progress(self, pid, on_event, stop=None, timeout=0):
        graph = self.graphs[int(pid[3:]) - 1]
        if "pp" in graph:
            return {"status": {"completed": True}, "outputs": {
                "pp": {"text": [json.dumps(PasteClient.report)]},
                "ps": {"images": [{"filename": "real_00001_.png", "subfolder": "ImageStudio",
                                   "type": "output"}]}}}
        return super().listen_for_progress(pid, on_event, stop, timeout)


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

    def test_a_node_may_be_kept_by_any_of_several_values(self):
        wf = {"id": "t", "graph": {
            "1": {"_when": ["a", "b"], "class_type": "Shared", "inputs": {}},
            "2": {"class_type": "SaveImage", "inputs": {}}}}
        self.assertNotIn("1", ig.fill(wf, {}))
        self.assertIn("1", ig.fill(wf, {"b": "x.png"}))
        self.assertTrue(ig.uses(wf, "b"))

    def test_a_switch_may_fall_back_to_an_earlier_switch(self):
        wf = {"id": "t", "switches": {
            "first": {"when": "a", "then": ["A", 0], "else": ["P", 0]},
            "last": {"when": "b", "then": ["B", 0], "else": "{{first}}"}},
            "graph": {"P": {"class_type": "P", "inputs": {}},
                      "A": {"_when": "a", "class_type": "A", "inputs": {}},
                      "B": {"_when": "b", "class_type": "B", "inputs": {}},
                      "S": {"class_type": "S", "inputs": {"in": "{{last}}"}}}}
        self.assertEqual(ig.fill(wf, {})["S"]["inputs"]["in"], ["P", 0])
        self.assertEqual(ig.fill(wf, {"a": 1})["S"]["inputs"]["in"], ["A", 0])
        self.assertEqual(ig.fill(wf, {"a": 1, "b": 1})["S"]["inputs"]["in"], ["B", 0])

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
            if wf.get("built_by"):            # finished in code: tested with its builder
                continue
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

    def test_faces_are_matched_to_the_nearest_person_once(self):
        boxes = [(100, 100, 40, 40), (500, 100, 40, 40), (900, 900, 30, 30)]
        people = [{"name": "a", "at": [0.52, 0.13]}, {"name": "b", "at": [0.5, 0.11]},
                  {"name": "c", "at": [0.1, 0.9]}]
        got = ig.match_faces(1000, 1000, boxes, people)
        self.assertEqual({i: p["name"] for i, p in got.items()}, {1: "a"})   # b loses the tie
        # a big face with a likeness to draw is redrawn anyway
        self.assertEqual([i for i, _ in ig.indexed_crops(2000, 2000, [(0, 0, 700, 700)],
                                                         keep={0})], [0])
        self.assertEqual(ig.face_crops(2000, 2000, [(0, 0, 700, 700)]), [])

    def test_a_region_mask_is_white_over_its_region_at_the_frames_shape(self):
        import studio_icons
        rgba, w, h = studio_icons.png_to_rgba(ig.region_png([0.5, 0.0, 1.0, 0.5], 800, 1600))
        self.assertEqual((w, h), (32, 64))
        at = lambda x, y: rgba[(y * w + x) * 4]
        self.assertEqual((at(20, 10), at(5, 10), at(20, 40)), (255, 0, 0))

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

    def test_always_on_loras_join_every_picture_from_a_model_they_suit(self):
        lib = self.studio.lib
        recs = lib.all("loras")
        for r in recs:
            r["always"] = r["id"] in ("sx70", "xl")
        lib.save("loras", recs)
        self.assertTrue(lib.get("loras", "sx70")["always"])
        p = self.plan(model="flux-dev", scene="x")
        self.assertEqual(p.loras, [("sx70.safetensors", 0.55)])
        self.assertEqual(p.lora_meta[0]["why"], "always on")
        self.assertFalse(any("XL thing" in w for w in p.warnings), p.warnings)
        # Already added by hand: that strength wins, and it is not doubled.
        p = self.plan(model="flux-dev", scene="x", loras=[{"id": "sx70", "strength": 0.3}])
        self.assertEqual(p.loras, [("sx70.safetensors", 0.3)])

    def test_z_image_is_the_default_model(self):
        self.assertEqual(ig.default_settings()["model"], "z-image-turbo")

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
        self.assertTrue(any("Pictures of the glasses" in w and "takes no item pictures" in w
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

    def test_the_camera_leads_the_prompt_and_keeps_the_head_in(self):
        p = self.plan(model="flux-dev", scene="On a pier.", subject="a woman",
                      view={"shot": "waist", "turn": 90, "height": "low"})
        self.assertEqual(p.errors, [])
        self.assertTrue(p.prompt.startswith("Medium shot from the waist up"), p.prompt)
        self.assertIn("whole head in the frame", p.prompt)
        self.assertIn("subject's left", p.prompt)
        self.assertIn("low angle", p.prompt)
        self.assertNotIn("eye-level", self.plan(model="flux-dev", scene="x").prompt)
        # With a drawn pose the figure frames itself: only the height is said.
        src = os.path.join(self.dir, "pose.png")
        with open(src, "wb") as f:
            f.write(PNG)
        posed = self.plan(model="flux-dev", scene="x", references={"pose": src},
                          pose={"points": [[0.5, 0.5]] * 18},
                          view={"shot": "face", "turn": 180, "height": "high"})
        self.assertTrue(posed.prompt.startswith("High angle shot"), posed.prompt)
        self.assertNotIn("behind", posed.prompt)
        self.assertTrue(any("drawn pose decides the framing" in w for w in posed.warnings))

    def test_a_camera_snaps_to_its_steps(self):
        self.assertIsNone(ig.clean_view(None))
        self.assertEqual(ig.clean_view({"turn": -170, "shot": "?", "height": 3}),
                         {"shot": "full", "turn": 180, "height": "eye"})
        self.assertEqual(ig.clean_view({"turn": 100})["turn"], 90)
        self.assertEqual(ig.view_label({"shot": "head", "turn": -45, "height": "overhead"}),
                         "Head and shoulders, overhead, from their right, three-quarter")

    def test_a_pose_goes_through_the_controlnet_on_flux(self):
        src = os.path.join(self.dir, "pose.png")
        with open(src, "wb") as f:
            f.write(PNG)
        cn = "FLUX.1-dev-ControlNet-Union-Pro-2.0.safetensors"
        inv = dict(FLUX_FILES, controlnet={cn})
        p = self.plan(model="flux-dev", inventory=inv, scene="x", references={"pose": src},
                      pose={"points": [[0.5, 0.5]] * 18, "strength": 0.75,
                            "hands": {"right": {"shape": "fist"}}})
        self.assertEqual(p.errors, [])
        self.assertTrue(p.prompt.endswith("Right hand clenched in a fist."), p.prompt)
        self.assertEqual(p.values["prompt"], p.prompt)
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

    def test_a_depth_map_chains_after_the_pose_on_one_controlnet(self):
        pose, depth = (os.path.join(self.dir, n) for n in ("pose.png", "depth.png"))
        for path in (pose, depth):
            with open(path, "wb") as f:
                f.write(PNG)
        cn = "FLUX.1-dev-ControlNet-Union-Pro-2.0.safetensors"
        inv = dict(FLUX_FILES, controlnet={cn})
        p = self.plan(model="flux-dev", inventory=inv, scene="x",
                      references={"pose": pose, "composition": depth},
                      pose={"strength": 0.8}, composition={"strength": 0.4})
        self.assertEqual(p.errors, [])
        self.assertEqual(p.images, {"pose_image": pose, "composition_image": depth})
        self.assertEqual((p.values["pose_strength"], p.values["composition_strength"]),
                         (0.8, 0.4))
        g = ig.fill(p.workflow, dict(p.values, pose_image="p.png", composition_image="d.png"))
        self.assertEqual([n for n in g if g[n]["class_type"] == "ControlNetLoader"], ["50"])
        self.assertEqual(g["54"]["inputs"]["positive"], ["52", 0])
        self.assertEqual(g["54"]["inputs"]["control_net"], ["50", 0])
        self.assertEqual(g["40"]["inputs"]["positive"], ["54", 0])
        self.assertEqual(g["40"]["inputs"]["negative"], ["54", 1])
        self.assertEqual(g["40"]["inputs"]["denoise"], 1.0)
        # The depth map alone hangs off the prompt.
        p = self.plan(model="flux-dev", inventory=inv, scene="x",
                      references={"composition": depth})
        g = ig.fill(p.workflow, dict(p.values, composition_image="d.png"))
        self.assertNotIn("52", g)
        self.assertEqual(g["54"]["inputs"]["positive"], ["11", 0])
        self.assertEqual(g["40"]["inputs"]["positive"], ["54", 0])

    def test_flux_takes_a_source_picture_and_denoise_needs_one(self):
        src = os.path.join(self.dir, "frame.png")
        with open(src, "wb") as f:
            f.write(PNG)
        p = self.plan(model="flux-dev", scene="x", references={"source": src}, denoise=0.8)
        self.assertEqual((p.images, p.values["denoise"]), ({"source_image": src}, 0.8))
        g = ig.fill(p.workflow, dict(p.values, source_image="f.png"))
        self.assertNotIn("20", g)
        self.assertEqual(g["40"]["inputs"]["latent_image"], ["22", 0])
        self.assertEqual(g["40"]["inputs"]["denoise"], 0.8)
        # Denoise with nothing to start from would leave noise in the picture.
        p = self.plan(model="flux-dev", scene="x", denoise=0.8)
        self.assertEqual(p.values["denoise"], 1.0)
        self.assertTrue(any("Denoise 0.8" in n for n in p.notes), p.notes)

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

    def scene_faces(self, face):
        """A scene's person where FaceClient's one face is, and one who is not
        in the picture."""
        return {"likeness": 0.85, "people": [
            {"id": "p1", "name": "Lilya", "at": [0.43, 0.35], "words": "A woman in her 30s.",
             "face": face, "from": "", "region": [0.35, 0.25, 0.5, 0.45]},
            {"id": "p2", "name": "Gavin", "at": [0.9, 0.9], "words": "A man.", "face": "",
             "from": ""}]}

    def test_a_scene_face_is_redrawn_as_its_person_and_to_their_face_picture(self):
        self.studio = ig.Studio(root=self.dir, notify=self.notified.append,
                                client_factory=PulidClient)
        FaceClient.fail_pass = False
        face = os.path.join(self.dir, "lilya.png")
        with open(face, "wb") as f:
            f.write(PNG)
        jobs = self.studio.submit(dict(ig.default_settings(), model="flux-dev", scene="x",
                                       backend="5090", seed=5, face_detail=True,
                                       scene_faces=self.scene_faces(face)))
        settle(jobs)
        self.assertEqual(jobs[0].status, "complete", jobs[0].detail)
        client = PulidClient.instances[-1]
        first, second = client.graphs
        # Her face goes into the picture itself, over her head only.
        self.assertEqual(first["40"]["inputs"]["model"], ["pb_1", 0])
        self.assertEqual(first["pb_1"]["inputs"]["model"], ["1", 0])
        self.assertEqual(first["pb_1f"]["inputs"]["image"], "studio_lilya.png")
        self.assertEqual(first["pb_1k"]["inputs"]["image"], ["pb_1m", 0])
        mask = [u for u in client.uploads if "face_regions" in u]
        self.assertEqual(len(mask), 1)
        self.assertIn(face, client.uploads)
        k = second["fc1_4"]["inputs"]
        self.assertEqual(k["model"], ["fc1_p", 0])
        self.assertEqual(k["denoise"], 0.85)          # the scene's likeness
        self.assertEqual(second["fc1_p"]["inputs"]["image"], ["fc1_r", 0])
        self.assertEqual(second["fc1_r"]["inputs"]["image"], "studio_lilya.png")
        self.assertEqual(second["pl1"]["inputs"]["pulid_file"], "pulid_flux_v0.9.1.safetensors")
        # Her own words, on a copy of the face prompt's conditioning.
        pos = second[k["positive"][0]]
        text = second[pos["inputs"]["conditioning"][0]]["inputs"]["text"]
        self.assertIn("A woman in her 30s.", text)
        self.assertEqual(second["f10"]["inputs"]["text"].count("A woman in her 30s"), 0)
        rec = self.studio.history.list()[0]
        self.assertEqual(rec["face_detail"]["likeness"], ["Lilya"])
        self.assertTrue(any("Gavin's face was not found" in n for n in rec["notes"]))
        self.assertTrue(any("Likeness: Lilya" in n for n in rec["notes"]), rec["notes"])

    def test_without_pulid_a_scene_face_is_redrawn_from_its_words_and_says_so(self):
        self.face_studio()
        face = os.path.join(self.dir, "lilya.png")
        with open(face, "wb") as f:
            f.write(PNG)
        jobs = self.studio.submit(dict(ig.default_settings(), model="flux-dev", scene="x",
                                       backend="5090", seed=5, face_detail=True,
                                       scene_faces=self.scene_faces(face)))
        settle(jobs)
        first, second = FaceClient.instances[-1].graphs
        self.assertFalse([n for n in first if n.startswith("pb")])
        self.assertNotIn("fc1_p", second)
        self.assertEqual(second["fc1_4"]["inputs"]["denoise"], 0.4)
        rec = self.studio.history.list()[0]
        self.assertTrue(any("lacks PuLID" in w for w in rec["warnings"]), rec["warnings"])

    def real_scene(self, client, report, real=True):
        """A scene job whose face Lilya has two photos, on `client`."""
        self.studio = ig.Studio(root=self.dir, notify=self.notified.append,
                                client_factory=client)
        FaceClient.fail_pass = False
        PasteClient.report = report
        photos = []
        for name in ("lilya.png", "lilya_left.png"):
            photos.append(os.path.join(self.dir, name))
            with open(photos[-1], "wb") as f:
                f.write(PNG)
        faces = self.scene_faces(photos[0])
        faces["real"] = real
        faces["people"][0]["photos"] = photos
        jobs = self.studio.submit(dict(ig.default_settings(), model="flux-dev", scene="x",
                                       backend="5090", seed=5, face_detail=True,
                                       scene_faces=faces))
        settle(jobs)
        self.assertEqual(jobs[0].status, "complete", jobs[0].detail)
        return client.instances[-1], self.studio.history.list()[0]

    def test_a_real_face_is_pasted_last_and_the_pulid_picture_kept_beside_it(self):
        client, rec = self.real_scene(PasteClient, [
            {"name": "Lilya", "pasted": True, "reference": "studio_lilya_left.png",
             "difference": 3.5, "tolerance": 19.2, "confidence": 0.82}])
        first, second, third = client.graphs
        self.assertEqual(third["pi"]["inputs"]["image"], "ImageStudio/faces_00001_.png [output]")
        faces = json.loads(third["pp"]["inputs"]["faces"])
        self.assertEqual(faces, [{"name": "Lilya", "box": [400, 300, 90, 110],
                                  "references": ["studio_lilya.png", "studio_lilya_left.png"]}])
        self.assertEqual(third["pp"]["inputs"]["seed"], 5)
        self.assertTrue(third["ps"]["inputs"]["filename_prefix"].endswith("_real"))
        self.assertNotIn("KSampler", {n["class_type"] for n in third.values()})   # no redraw
        self.assertEqual(len(rec["images"]), 2)          # the pasted one first, PuLID's kept
        self.assertEqual(rec["paste_graph"], third)
        self.assertEqual(rec["face_detail"]["real"][0]["reference"],
                         os.path.join(self.dir, "lilya_left.png"))
        self.assertTrue(any("Real face: Lilya from lilya_left.png" in n for n in rec["notes"]),
                        rec["notes"])

    def test_a_face_no_photo_fits_is_left_as_pulid_drew_it(self):
        client, rec = self.real_scene(PasteClient, [
            {"name": "Lilya", "pasted": False, "why": "no photo at this angle (30 degrees off, "
                                                      "19 allowed)"}])
        self.assertEqual(len(client.graphs), 3)
        self.assertEqual(len(rec["images"]), 1)          # nothing pasted: nothing twice
        self.assertIsNone(rec["paste_graph"])
        self.assertTrue(any("Lilya kept as PuLID drew it - no photo at this angle" in n
                            for n in rec["notes"]), rec["notes"])

    def test_real_faces_need_the_node_and_the_scene_to_ask(self):
        client, rec = self.real_scene(PulidClient, [])
        self.assertEqual(len(client.graphs), 2)
        self.assertEqual(len(rec["images"]), 1)
        self.assertTrue(any("has no StudioFacePaste node" in n for n in rec["notes"]))
        client, rec = self.real_scene(PasteClient, [], real=False)
        self.assertEqual(len(client.graphs), 2)
        self.assertFalse(any("Real face" in n for n in rec["notes"]))

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

    def test_a_server_too_busy_to_answer_once_is_not_offline(self):
        """The 3090 finished five pictures and then read offline: its
        /system_stats had stalled past 5 s while it wrapped up a job."""
        c = ig.ComfyUIClient({"id": "x", "name": "X", "url": "http://10.0.0.9:8188"})
        waits = []
        def stats(path, timeout=None):
            waits.append(timeout)
            if len(waits) == 1:
                raise ig.Unreachable("Cannot reach X at http://10.0.0.9:8188. (timed out)")
            return {"system": {"comfyui_version": "0.37"}, "devices": [{}]}
        c.get_json = stats
        c.get_queue = lambda: {}
        h = c.health()
        self.assertTrue(h["ok"], h["detail"])
        self.assertEqual(waits, [5, 15])

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

    def test_outfit_presets_keep_only_clothes_and_start_with_some(self):
        lib = ig.Library(tempfile.mkdtemp())
        self.assertIn("Casual", [r["name"] for r in lib.all("outfits")])
        for r in lib.all("outfits"):
            self.assertTrue(r["looks"])
            self.assertLessEqual(set(r["looks"]), set(ig.OUTFIT_KEYS))
        lib.save("outfits", [{"name": "Red night", "looks": {
            "top": "red sweater", "hair": "black", "weight": 2, "footwear": " "}}])
        self.assertEqual(ig.Library(lib.root).all("outfits"),
                         [{"id": "red-night", "name": "Red night",
                           "looks": {"top": "red sweater"}}])

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

DRESS_FILES = {"diffusion_models": {"qwen_image_edit_2509_fp8_e4m3fn.safetensors"},
               "text_encoders": {"qwen_2.5_vl_7b_fp8_scaled.safetensors"},
               "vae": {"qwen_image_vae.safetensors"},
               "loras": {"Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors",
                         "clothes_tryon_qwen-edit-lora.safetensors"},
               "checkpoints": {"sam3.1_multiplex_fp16.safetensors"}}


class DressClient(FakeClient):
    """A ComfyUI with the Qwen edit model, the try-on LoRA and SAM3. A dress
    run with a head to find answers with its preview and one face; the run
    that saves answers with its picture."""
    NODES = FakeClient.node_types(None) | ig.DRESS_NODES | ig.FACE_NODES
    face = (380, 60, 70, 90)          # a full-length picture's face, x y w h

    def inventory(self):
        inv = super().inventory()
        for kind, files in DRESS_FILES.items():
            inv[kind] = inv.get(kind, set()) | files
        return inv

    def node_types(self):
        return set(DressClient.NODES)

    def listen_for_progress(self, pid, on_event, stop=None, timeout=0):
        graph = self.graphs[int(pid[3:]) - 1]
        ks = sorted(n for n in graph if n.endswith("_ks"))
        for n in ks:
            on_event("executing", n)
            on_event("progress", (6, 6, n))
        out = {}
        if "dp" in graph:
            out["dp"] = {"images": [{"filename": "dress_p.png", "subfolder": "",
                                     "type": "temp"}]}
        if "ds" in graph:
            out["ds"] = {"images": [{"filename": "dress_00001_.png", "subfolder": "ImageStudio",
                                     "type": "output"}]}
        if "fd4" in graph:
            x, y, w, h = DressClient.face
            out.update({"fd4": {"text": [json.dumps([[{"x": x, "y": y, "width": w,
                                                       "height": h}]])]},
                        "fd6": {"text": ["832"]}, "fd7": {"text": ["1216"]}})
        if not out:
            return super().listen_for_progress(pid, on_event, stop, timeout)
        return {"status": {"completed": True}, "outputs": out}


def ran(cls):
    """The graphs the 5090's client was sent (Studio makes one per backend)."""
    return next(c for c in cls.instances if c.backend["id"] == "5090" and c.graphs).graphs


def png_of(width, height):
    """A PNG header saying width x height: all picture_size reads."""
    import struct
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(
        ">II", width, height) + b"\x08\x02\x00\x00\x00"


class TestTryOn(TempStudioMixin, unittest.TestCase):
    def pic(self, name, size=(832, 1216)):
        path = os.path.join(self.dir, name + ".png")
        with open(path, "wb") as f:
            f.write(png_of(*size))
        return path

    def outfit(self, hair=True, acc=True):
        return ig.clean_outfit({
            "clothes": [{"name": "plaid shirt", "path": self.pic("shirt")},
                        {"name": "jeans", "path": self.pic("jeans")}],
            "hair": {"path": self.pic("hair")} if hair else None,
            "accessories": [{"name": "glasses", "path": self.pic("glasses")},
                            {"name": "wristwatch", "path": self.pic("watch")},
                            {"name": "gold necklace", "path": self.pic("necklace")}]
            if acc else []})

    def dress_studio(self):
        self.studio = ig.Studio(root=self.dir, notify=self.notified.append,
                                client_factory=DressClient)

    def test_picture_sizes_from_headers_alone(self):
        import struct
        self.assertEqual(ig.picture_size(png_of(832, 1216)), (832, 1216))
        jpeg = (b"\xff\xd8" + b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00" + b"\x00" * 9
                + b"\xff\xc0" + struct.pack(">HBHH", 17, 8, 1216, 832) + b"\x00" * 12)
        self.assertEqual(ig.picture_size(jpeg), (832, 1216))
        self.assertEqual(ig.picture_size(b"GIF89a" + struct.pack("<HH", 64, 32)), (64, 32))
        bmp = b"BM" + b"\x00" * 16 + struct.pack("<ii", 100, -50)
        self.assertEqual(ig.picture_size(bmp), (100, 50))
        webp = b"RIFF\x00\x00\x00\x00WEBPVP8X" + b"\x00" * 8 + (639).to_bytes(3, "little") \
            + (479).to_bytes(3, "little")
        self.assertEqual(ig.picture_size(webp), (640, 480))
        self.assertIsNone(ig.picture_size(b"not a picture at all"))

    def test_passes_clothes_then_body_then_head_in_words_that_work(self):
        passes = ig.dress_passes(self.outfit())
        self.assertEqual([(x["kind"], x["where"]) for x in passes],
                         [("clothes", "body"), ("accessories", "body"), ("hair", "head"),
                          ("accessories", "head")])
        self.assertEqual(passes[0]["prompt"], ig.TRYON_PROMPT)
        self.assertEqual([i["name"] for i in passes[1]["items"]], ["wristwatch"])
        self.assertEqual(passes[2]["prompt"], "The person in picture 1 now has the hair of the "
                                              "person in picture 2.")
        self.assertEqual(passes[3]["prompt"], "The person in picture 1 wears the glasses from "
                                              "picture 2 and the gold necklace from picture 3.")
        for x in passes:                 # "keep it the same" stopped every edit, live
            self.assertNotIn("same", x["prompt"].lower())
            self.assertNotIn("keep", x["prompt"].lower())
        words = ig.dress_passes(ig.clean_outfit({"hair": {"words": "short red hair"}}))
        self.assertEqual(words[0]["prompt"], "Change the person's hair to short red hair.")
        self.assertEqual(ig.dress_passes(ig.clean_outfit({})), [])

    def test_the_clothes_go_left_of_the_person_and_come_back_exactly(self):
        wf = ig.load_workflow(ig.DRESS_WORKFLOW)
        o = self.outfit(hair=False, acc=False)
        passes = ig.dress_passes(o)
        pictures = {x: "up_" + os.path.basename(x) for x in ig.outfit_pictures(o)}
        v = dict(ig.dress_values(wf, self.backend("5090")), seed=7)
        g = ig.dress_graph(wf, v, passes, "person.png", (832, 1216), pictures, "out")
        cw, ch = ig.panel_size(832, 1216, 0.5)
        self.assertEqual(g["d1_in"]["inputs"]["direction"], "right")
        self.assertEqual(g["d1_in"]["inputs"]["image2"], ["d1_p", 0])      # person on the right
        self.assertEqual((g["d1_p"]["inputs"]["width"], g["d1_p"]["inputs"]["height"]), (cw, ch))
        self.assertEqual(g["d1_f0"]["inputs"]["target_height"]
                         + g["d1_f1"]["inputs"]["target_height"], ch)       # a column as tall
        self.assertEqual(g["d1_cut"]["inputs"], {"image": ["d1_dec", 0], "width": cw,
                                                 "height": ch, "x": cw, "y": 0})
        self.assertEqual(g["d1_enc"]["inputs"]["pixels"], ["d1_in", 0])    # no resample between
        self.assertEqual(g["d1_ks"]["inputs"]["model"], ["9", 0])          # the try-on chain
        self.assertEqual(g["7"]["inputs"]["lora_name"], "clothes_tryon_qwen-edit-lora.safetensors")
        self.assertEqual(g["7"]["inputs"]["strength_model"], 1.5)
        self.assertEqual(g["d1_pos"]["inputs"]["prompt"], ig.TRYON_PROMPT)
        self.assertEqual((g["dz"]["inputs"]["width"], g["dz"]["inputs"]["height"]), (832, 1216))
        self.assertEqual(g["ds"]["inputs"]["filename_prefix"], "out")
        self.assertEqual(g["d1_ks"]["inputs"]["seed"], 7)
        hair = ig.dress_passes(ig.clean_outfit({"hair": {"words": "red hair"}}))
        g2 = ig.dress_graph(wf, v, hair, "person.png", (832, 1216), {}, "out")
        self.assertNotIn("7", g2)                     # no try-on LoRA without clothes
        self.assertEqual(g2["d1_ks"]["inputs"]["model"], ["6", 0])

    def test_the_head_crop_is_head_and_shoulders_or_nothing(self):
        crop = ig.head_region((380, 60, 70, 90), 832, 1216)
        self.assertLessEqual(crop["y"], 60 - 45)                   # the hair above
        self.assertGreaterEqual(crop["y"] + crop["height"], 60 + 90 + 2 * 90)   # the neck
        self.assertEqual((crop["width"] % 16, crop["height"] % 16), (0, 0))
        self.assertIsNone(ig.head_region((300, 200, 400, 500), 1024, 1024))    # a portrait
        wf = ig.load_workflow(ig.DRESS_WORKFLOW)
        o = self.outfit(acc=False)
        head = [x for x in ig.dress_passes(o) if x["where"] == "head"]
        pictures = {x: "up_" + os.path.basename(x) for x in ig.outfit_pictures(o)}
        g = ig.dress_head_graph(wf, dict(wf["defaults"], seed=3), head, "p.png [temp]",
                                (832, 1216), crop, "mask.png", (832, 1216), pictures, "out",
                                first=1, sam3="sam3.pt")
        self.assertEqual(g["hc"]["inputs"]["crop_region"], crop)
        self.assertEqual(g["hb"]["inputs"]["mask"], ["hp8", 0])    # the person, not the box
        self.assertEqual((g["hb"]["inputs"]["x"], g["hb"]["inputs"]["y"]), (crop["x"], crop["y"]))
        self.assertEqual(g["h1_ks"]["inputs"]["seed"], 4)
        self.assertEqual(g["h1_out"]["inputs"]["width"], crop["width"])

    def test_generate_draws_a_character_from_its_item_pictures(self):
        refs = {"leather jacket": self.pic("jacket"), "glasses": self.pic("glasses"),
                "hair": self.pic("hair"), "hat": self.pic("hat")}
        s = dict(ig.default_settings(), model="flux-dev", scene="On a pier.",
                 outerwear="leather jacket", accessories="glasses", item_refs=refs,
                 face_detail=True)
        inv = dict(FLUX_FILES, checkpoints={"sam3.pt"}, **KONTEXT_FILES)
        p = ig.compose(s, self.studio.lib, self.backend("5090"), inv)
        self.assertEqual(p.errors, [])
        # Clothes, then the hair, then accessories; the hat is not worn today.
        self.assertEqual([n for n, _ in p.items], ["leather jacket", "hair", "glasses"])
        self.assertEqual(p.references["item: glasses"], refs["glasses"])
        self.assertEqual(p.values["model"], "flux1-dev-kontext_fp8_scaled.safetensors")
        self.assertEqual(p.values["guidance"], 2.5)
        self.assertIn("The leather jacket, hair and glasses look exactly as in the reference",
                      p.prompt)
        self.assertNotIn("reference", p.values["face_prompt"])     # the face has none
        self.assertFalse(any("Pictures of the" in w for w in p.warnings), p.warnings)
        self.assertTrue(any("FLUX.1 Kontext" in n for n in p.notes), p.notes)
        mine = ig.compose(dict(s, guidance=4.0), self.studio.lib, self.backend("5090"), inv)
        self.assertEqual(mine.values["guidance"], 4.0)             # the form's wins
        short = ig.compose(s, self.studio.lib, self.backend("5090"), FLUX_FILES)
        self.assertEqual(short.items, [])
        self.assertEqual(short.values["model"], "flux1-dev.safetensors")
        self.assertTrue(any("flux1-dev-kontext_fp8_scaled.safetensors" in w
                            and "words alone" in w for w in short.warnings), short.warnings)
        plain = ig.compose(dict(s, item_refs={}), self.studio.lib, self.backend("5090"), inv)
        self.assertEqual((plain.items, plain.values["model"]), ([], "flux1-dev.safetensors"))
        self.assertNotIn("kontext_model", plain.values)

    def test_the_item_pictures_are_one_reference_on_the_prompt(self):
        wf = ig.load_workflow("flux_dev_baseline")
        g = ig.fill(wf, dict(wf["defaults"], model="m", clip_l="c", t5="t", vae="v",
                             prompt="p", seed=1))
        was = g["11"]["inputs"]["conditioning"]
        ig.add_item_refs(g, wf["items"], ["a.png", "b.png", "c.png"])
        self.assertEqual(g["11"]["inputs"]["conditioning"], ["ik3", 0])
        self.assertEqual(g["ik3"]["inputs"]["conditioning"], was)
        self.assertEqual(g["is3"]["inputs"]["image1"], ["is2", 0])
        self.assertEqual(g["is3"]["inputs"]["image2"], ["it3", 0])
        self.assertEqual(g["ik1"]["inputs"]["image"], ["is3", 0])
        self.assertEqual(g["ik2"]["inputs"]["vae"], ["3", 0])
        one = ig.fill(wf, dict(wf["defaults"], model="m", clip_l="c", t5="t", vae="v",
                               prompt="p", seed=1))
        ig.add_item_refs(one, wf["items"], ["a.png"])
        self.assertEqual(one["ik1"]["inputs"]["image"], ["it1", 0])
        self.assertFalse([n for n in one if n.startswith("is")])

    def test_a_try_on_job_dresses_the_body_then_the_head_and_lands_in_history(self):
        self.dress_studio()
        o = self.outfit()
        jobs = self.studio.submit({"mode": "dress", "seed": 11, "backend": "auto",
                                   "outfit": dict(o, person=self.pic("person"))})
        settle(jobs)
        job = jobs[0]
        self.assertEqual(job.status, "complete", job.detail)
        self.assertEqual(job.backend["id"], "5090")
        first, second = ran(DressClient)
        self.assertIn("dp", first)
        self.assertEqual([n for n in sorted(first) if n.endswith("_ks")], ["d1_ks", "d2_ks"])
        self.assertEqual(second["hi"]["inputs"]["image"], "dress_p.png [temp]")
        self.assertEqual([n for n in sorted(second) if n.endswith("_ks")], ["h1_ks", "h2_ks"])
        self.assertEqual(second["h1_ks"]["inputs"]["seed"], 13)     # after the body's two
        rec = self.studio.history.list()[0]
        self.assertEqual(rec["prompt"], "Try On: plaid shirt, jeans, the hair in the picture, "
                                        "glasses, wristwatch and gold necklace")
        self.assertEqual(rec["dress"]["head_crop"], second["hc"]["inputs"]["crop_region"])
        self.assertEqual([l["name"] for l in rec["loras"]], ["Lightning 4-step",
                                                             "Clothes Try On"])
        self.assertEqual(rec["settings"]["mode"], "dress")
        self.assertIn("Try On", ig.summary(rec["settings"]))
        again = ig.again(rec)
        self.assertEqual((again["seed"], again["prefer_backend"]), (11, "5090"))

    def test_without_sam3_everything_is_one_run_on_the_whole_picture(self):
        class NoSam(DressClient):
            def inventory(self):
                return dict(super().inventory(), checkpoints=set())
        self.studio = ig.Studio(root=self.dir, notify=self.notified.append,
                                client_factory=NoSam)
        jobs = self.studio.submit_dress({"outfit": dict(self.outfit(),
                                                        person=self.pic("person"))})
        settle(jobs)
        self.assertEqual(jobs[0].status, "complete", jobs[0].detail)
        self.assertEqual(len(ran(NoSam)), 1)
        rec = self.studio.history.list()[0]
        self.assertTrue(any("no SAM3" in n for n in rec["notes"]), rec["notes"])

    def test_a_try_on_needs_a_person_and_something_to_put_on(self):
        self.dress_studio()
        jobs = self.studio.submit_dress({"outfit": {"person": self.pic("person")}})
        settle(jobs)
        self.assertEqual(jobs[0].status, "failed")
        self.assertIn("Add something to put on", jobs[0].detail)
        jobs = self.studio.submit_dress({"outfit": dict(self.outfit(), person="")})
        settle(jobs)
        self.assertIn("Choose the person", jobs[0].detail)

    def test_nowhere_to_try_on_says_what_is_missing(self):
        jobs = None
        with self.assertRaises(ig.ComfyError) as e:     # FakeClient has no Qwen files
            jobs = self.studio.submit_dress({"outfit": dict(self.outfit(),
                                                            person=self.pic("person"))})
        self.assertIsNone(jobs)
        self.assertIn("clothes_tryon_qwen-edit-lora.safetensors", str(e.exception))

    def test_a_generated_picture_is_made_from_its_item_pictures(self):
        self.studio = ig.Studio(root=self.dir, notify=self.notified.append,
                                client_factory=KontextClient)
        refs = {"leather jacket": self.pic("jacket")}
        jobs = self.studio.submit(dict(ig.default_settings(), model="flux-dev",
                                       scene="On a pier.", outerwear="leather jacket",
                                       item_refs=refs, backend="5090", seed=5))
        settle(jobs)
        self.assertEqual(jobs[0].status, "complete", jobs[0].detail)
        graphs = ran(KontextClient)
        self.assertEqual(len(graphs), 1)                             # one pass, no Try On
        g = graphs[0]
        self.assertEqual(g["1"]["inputs"]["unet_name"], "flux1-dev-kontext_fp8_scaled.safetensors")
        self.assertEqual(g["it1"]["inputs"]["image"], "studio_jacket.png")
        self.assertEqual(g["11"]["inputs"]["conditioning"], ["ik3", 0])
        rec = self.studio.history.list()[0]
        self.assertEqual(rec["references"]["item: leather jacket"], refs["leather jacket"])
        self.assertEqual(rec["model"]["file"], "flux1-dev-kontext_fp8_scaled.safetensors")
        self.assertIsNone(rec["dress"])


KONTEXT_FILES = {"diffusion_models": FLUX_FILES["diffusion_models"]
                 | {"flux1-dev-kontext_fp8_scaled.safetensors"}}


class KontextClient(FakeClient):
    """A ComfyUI with FLUX Kontext and the nodes that give it a reference."""
    def inventory(self):
        return dict(super().inventory(), **KONTEXT_FILES)

    def node_types(self):
        return FakeClient.node_types(self) | ig.ITEM_NODES


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

    def test_the_camera_is_aimed_by_dragging_and_comes_back_with_reuse(self):
        s, ui = self.tab()
        self.assertIsNone(ui.collect()["view"])

        class Ev:
            def __init__(self, x, y):
                self.x, self.y = x, y
        a, k = ui.aim, ui.aim.k
        cx, cy, r = a.TOP
        a._press(Ev(int((cx + r) * k), int(cy * k)))             # round to their left
        a._press(Ev(int((a.FIG_X + a.REACH["head"]) * k), int(a.ROWS["high"] * k)))
        view = ui.collect()["view"]
        self.assertEqual(view, {"shot": "head", "turn": 90, "height": "high"})
        self.assertIn("Head and shoulders", ui.aim_note.cget("text"))
        ui._aimed(None)
        self.assertIsNone(ui.collect()["view"])
        ui.apply(dict(ig.default_settings(), view=view))
        self.assertEqual(ui.collect()["view"], view)

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
        # A click on a hand, not a drag, gives it its next shape; the menu agrees.
        hand = ed._hand_points()["right"]
        hx = int(sum(p[0] for p in hand) / len(hand) * ed.vw)
        hy = int(sum(p[1] for p in hand) / len(hand) * ed.vh)
        ed._press(Ev(hx, hy))
        ed._release(Ev(hx, hy))
        self.assertEqual(ed.hands["right"]["shape"], "open")
        self.assertIn("Open", ed.hand_pills["right"].cget("text"))
        ed._flip("left")
        ed._mirror()                                  # hands swap with the sides
        self.assertTrue(ed.hands["right"]["back"])
        self.assertEqual(ed.hands["left"]["shape"], "open")
        ed._undo()
        self.assertEqual(ed.hands["right"]["shape"], "open")
        for view in ("model", "figure"):
            ed._see(view)
            self.app.update()
        ed.strength.set(0.8)
        ed._use()
        self.app.update()
        path = ui.refs["pose"]
        self.assertTrue(os.path.isfile(path))
        self.assertIn("drawn figure", ui.ref_labels["pose"].cget("text"))
        got = ui.collect()
        self.assertEqual((got["pose"]["strength"], got["pose"]["hidden"]), (0.8, [17]))
        self.assertEqual(got["pose"]["hands"]["right"], {"shape": "open", "back": False})
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

    def test_a_civitai_link_imports_into_the_lora_library(self):
        import urllib.request
        import studio_images_ui
        sys.path.insert(0, HERE)
        from test_civitai import FakeCivitAI
        s, ui = self.tab()
        fake, real = FakeCivitAI(), urllib.request.urlopen
        urllib.request.urlopen = fake
        try:
            ed = ui.edit_loras()
            self.app.update()
            dlg = studio_images_ui.LoraImport(ui, ed)
            self.app.update()
            dlg.links.delete("1.0", "end")
            dlg.links.insert("1.0", "not a link")
            dlg.start()
            self.assertIn("Not a CivitAI link", dlg.msg.cget("text"))
            dlg.links.delete("1.0", "end")
            dlg.links.insert("1.0", "https://civitai.com/models/12345?modelVersionId=67890")
            dlg.dest = ""                        # the profile only; no download
            dlg.start()
            self.pump(lambda: not dlg.busy)
            self.assertIn("1 added", dlg.msg.cget("text"))
            rec = ui.studio.lib.lora_by_file("sx70_flux_v2.safetensors")
            self.assertEqual(rec["trigger"], "sx70 photo, polaroid frame")
            self.assertEqual(ed.records[ed.current]["id"], rec["id"])   # shown in the editor
            dlg.close()
            ed.win.destroy()
        finally:
            urllib.request.urlopen = real

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

    def test_item_pictures_are_chosen_on_the_form(self):
        s, ui = self.tab()
        pic = os.path.join(self.dir, "jacket.png")
        with open(pic, "wb") as f:
            f.write(PNG)
        ui.text["outerwear"].set("leather jacket")
        ui._show_looks("Clothes")
        ui._set_item("leather jacket", pic)
        self.app.update()
        kept = ui.collect()["item_refs"]["leather jacket"]
        self.assertNotEqual(kept, pic)                  # copied, so history keeps it
        self.assertTrue(os.path.isfile(kept))
        texts = [w.cget("text") for row in ui.look_box.winfo_children()
                 for w in row.winfo_children() if type(w).__name__ == "Label"]
        self.assertIn("leather jacket", texts)
        ui._set_item("leather jacket", None)
        self.assertNotIn("leather jacket", ui.collect()["item_refs"])
        ui._show_looks("Hair")
        self.app.update()
        # A Try On record from before stays out of the form, and says why.
        ui.reuse({"mode": "dress", "seed": 4})
        self.assertNotIn("dress", ui.collect())


if __name__ == "__main__":
    unittest.main()


class PersonCutoutTest(unittest.TestCase):
    """The Identities editor's Pick person: find everyone, pick one, cut out."""

    def test_people_found_orders_largest_first_and_drops_slivers(self):
        entry = {"outputs": {
            "5": {"text": [json.dumps([[{"x": 0, "y": 0, "width": 10, "height": 10},
                                        {"x": 50, "y": 0, "width": 100, "height": 200},
                                        {"x": 200, "y": 0, "width": 80, "height": 190}]])]},
            "7": {"text": ["400"]}, "8": {"text": ["300"]},
            "9": {"images": [{"filename": "p.png", "type": "temp"}]}}}
        w, h, boxes, pv = ig.people_found(entry)
        self.assertEqual((w, h), (400, 300))
        self.assertEqual(boxes, [(50, 0, 100, 200), (200, 0, 80, 190)])
        self.assertEqual(pv["filename"], "p.png")
        self.assertIsNone(ig.people_found({"outputs": {}}))

    def test_pick_box_prefers_smallest_containing_then_nearest(self):
        boxes = [(0, 0, 100, 100), (10, 10, 20, 20), (300, 300, 10, 10)]
        self.assertEqual(ig.pick_box(boxes, 15, 15), (10, 10, 20, 20))
        self.assertEqual(ig.pick_box(boxes, 50, 50), (0, 0, 100, 100))
        self.assertEqual(ig.pick_box(boxes, 290, 290), (300, 300, 10, 10))

    def test_cutout_region_pads_and_stays_inside(self):
        r = ig.cutout_region((0, 10, 100, 200), 150, 205)
        self.assertEqual(r, {"x": 0, "y": 2, "width": 108, "height": 203})

    def test_cutout_graph_crops_masks_and_saves(self):
        g = ig.cutout_graph("a.png", "sam3.pt", {"x": 1, "y": 2, "width": 3, "height": 4})
        self.assertEqual(g["6"]["inputs"]["width"], 3)
        self.assertEqual(g["7"]["inputs"]["mask"], ["5", 0])
        self.assertEqual(g["8"]["class_type"], "SaveImage")

    def test_find_then_cut_through_a_fake_backend(self):
        class Fake:
            def __init__(self, backend):
                self.backend, self.graphs = backend, []
                self.url = backend["url"].rstrip("/")

            def upload_image(self, path):
                return "up.png"

            def queue_workflow(self, graph):
                self.graphs.append(graph)
                return str(len(self.graphs))

            def get_history(self, pid):
                if pid == "1":
                    return {"outputs": {
                        "5": {"text": [json.dumps([{"x": 5, "y": 5, "width": 50,
                                                     "height": 90}])]},
                        "7": {"text": ["100"]}, "8": {"text": ["100"]},
                        "9": {"images": [{"filename": "pv.png", "type": "temp"}]}}}
                return {"outputs": {"8": {"images": [{"filename": "cut.png"}]}}}

            def fetch(self, f):
                return f["filename"].encode()

        with tempfile.TemporaryDirectory() as d:
            s = ig.Studio(root=d, client_factory=Fake)
            b = s.backends()[0]
            s.health[b["id"]] = {"ok": True}
            s.inventories[b["id"]] = {"checkpoints": {"sam3.pt", "flux.safetensors"}}
            found = s.find_people(__file__)
            self.assertEqual(found["boxes"], [(5, 5, 50, 90)])
            self.assertEqual(found["preview"], b"pv.png")
            self.assertEqual(s.cut_person(found, found["boxes"][0]), b"cut.png")
            region = s.client(b).graphs[1]["2"]["inputs"]["crop_region"]
            self.assertEqual(region, {"x": 2, "y": 2, "width": 56, "height": 96})
