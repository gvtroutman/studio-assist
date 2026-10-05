"""The Image Studio: the template adapter, composing a job, routing, the
library on disk, the job queue and history against a fake ComfyUI, and the tab
itself built in process. Nothing here touches the network or a GPU."""

import base64
import gc
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import apps.image_studio.imagegen as ig  # noqa: E402

# A 1x1 PNG, for the pictures the fake ComfyUI "makes".
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")

FLUX_FILES = {"diffusion_models": {"flux1-dev.safetensors", "z_image_turbo_bf16.safetensors"},
              "text_encoders": {"clip_l.safetensors", "t5xxl_fp16.safetensors",
                                "t5xxl_fp8_e4m3fn.safetensors",
                                "t5xxl_fp8_e4m3fn_scaled.safetensors", "qwen_3_4b.safetensors"},
              "vae": {"ae.safetensors"}, "loras": {"sitter.safetensors", "sx70.safetensors",
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
        # Tk things an earlier test left in a cycle are freed by whichever
        # thread the collector next runs on. On a lane's thread that is a Tk
        # call off the UI thread, and the lane stops there: a job left
        # "queued" for good, in a full run only (2026-09-30). Freed here,
        # on this thread, before a lane is started.
        gc.collect()
        self.dir = tempfile.mkdtemp()
        FakeClient.instances, FakeClient.down, FakeClient.hold = [], set(), None
        self.notified = []
        self.studio = ig.Studio(root=self.dir, notify=self.notified.append,
                                client_factory=FakeClient)
        lib = self.studio.lib
        lib.save("loras", [
            {"id": "sitter", "file": "sitter.safetensors", "name": "Sitter Identity",
             "category": "Identity", "trigger": "SITTERPERSON", "family": "flux1"},
            {"id": "sx70", "file": "sx70.safetensors", "name": "SX-70",
             "category": "Camera / Film", "family": "flux1", "strength": 0.55},
            {"id": "xl", "file": "xl_thing.safetensors", "name": "XL thing",
             "category": "Style", "family": "sdxl"},
            {"id": "mystery", "file": "mystery.safetensors", "name": "Mystery"},
        ])
        lib.save("identities", [
            {"id": "sitter", "name": "Sitter", "lora": "sitter", "trigger": "SITTERPERSON",
             "strength": 0.85},
            {"id": "partner", "name": "Partner", "trigger": "PARTNERPERSON", "strength": 0.8}])
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
    def test_klein_redraw_keeps_crop_schedule_masks_and_loras(self):
        wf = ig.load_workflow("klein9b_base")
        values = dict(wf["defaults"], model="klein", encoder="qwen", vae="vae",
                      prompt="a portrait", face_prompt="a face", negative="blur",
                      seed=ig.MAX_SEED, face_denoise=0.4, redraw_steps=20)
        crops = [{"x": 0, "y": 0, "width": 128, "height": 256,
                  "edit": (512, 1024), "head": False},
                 {"x": 128, "y": 0, "width": 128, "height": 256,
                  "edit": (768, 1024), "head": False}]
        for denoise, total, trim in ((0.4, 50, 30), (1.0, 20, 0), (0.0, 20, 20)):
            with self.subTest(denoise=denoise):
                g = ig.face_graph(wf, values, [("person.safetensors", 0.8)],
                    "made.png", crops, "oval.png", "out",
                    faces=[{"denoise": denoise}, {"denoise": denoise}])
                for i, width in ((1, 512), (2, 768)):
                    n = "fc%d_" % i
                    self.assertEqual(g[n + "sigmas"]["inputs"], {
                        "steps": total, "width": width, "height": 1024})
                    self.assertEqual(g[n + "trim"]["inputs"]["step"], trim)
                    self.assertEqual(g[n + "noise"]["inputs"]["noise_seed"], i - 1)
                    k = g[n + "4"]["inputs"]
                    self.assertEqual(k["latent_image"], [n + "3n", 0])
                    guider = g[n + "guider"]["inputs"]
                    adapter = g[guider["model"][0]]["inputs"]
                    self.assertEqual(adapter["lora_name"], "person.safetensors")
                    self.assertEqual(adapter["strength_model"], 0.8)
                    self.assertEqual(guider["negative"], ["12", 0])
                    self.assertEqual(guider["cfg"], 4.0)
                self.assertEqual(g["fc2_1"]["inputs"]["image"], ["fc1_9", 0])

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

    def test_zimage_pose_and_depth_patch_only_the_first_pass_model(self):
        wf = ig.load_workflow("zimage_hq")
        base = {"model": "m", "encoder": "e", "vae": "v", "prompt": "p", "seed": 1,
                "refine": True}
        plain = ig.fill(wf, base)
        self.assertEqual(plain["40"]["inputs"]["model"], ["4", 0])
        self.assertFalse({"50", "52", "54", "56"} & set(plain))
        for maps in ({"pose_image": "pose.png"}, {"composition_image": "depth.png"},
                     {"pose_image": "pose.png", "composition_image": "depth.png"}):
            g = ig.fill(wf, dict(base, **maps))
            self.assertEqual(g["50"]["class_type"], "ModelPatchLoader")
            self.assertEqual(g["50"]["inputs"]["name"], wf["defaults"]["model_patch"])
            self.assertEqual(g["40"]["inputs"]["model"], ["56", 0])
            last = "54" if "composition_image" in maps else "52"
            self.assertEqual(g["56"]["inputs"]["model"], [last, 0])
            if len(maps) == 2:                      # depth chained after pose
                self.assertEqual(g["54"]["inputs"]["model"], ["52", 0])
                self.assertEqual(g["52"]["inputs"]["strength"], 0.85)
            self.assertEqual(g["44"]["inputs"]["model"], ["4", 0])   # refine: unpatched
            self.assertEqual(g["4"]["inputs"]["model"], ["1", 0])
        self.assertEqual(wf["face_detail"]["model"], ["4", 0])       # face pass: unpatched

    def test_no_character_regions_leaves_the_graph_as_it_was(self):
        base = {"model": "m", "encoder": "e", "vae": "v", "prompt": "p", "seed": 1,
               "refine": True}
        with_field = ig.fill(ig.load_workflow("zimage_hq"), base)
        self.assertEqual(with_field["40"]["inputs"]["positive"], ["10", 0])
        self.assertEqual(with_field["44"]["inputs"]["positive"], ["10", 0])
        self.assertFalse([n for n in with_field if n.startswith("char")])

    def test_regional_conditioning_chains_any_number_of_characters(self):
        base = {"model": "m", "encoder": "e", "vae": "v", "prompt": "base scene", "seed": 1,
               "refine": True}
        for count in (1, 2, 3):
            regions = [{"prompt": "person %d" % i, "mask_var": "mask%d" % i}
                      for i in range(1, count + 1)]
            values = dict(base, character_regions=regions,
                         **{"mask%d" % i: "m%d.png" % i for i in range(1, count + 1)})
            g = ig.fill(ig.load_workflow("zimage_hq"), values)
            for i in range(1, count + 1):
                self.assertEqual(g["char%dmaskimg" % i]["inputs"]["image"], "m%d.png" % i)
                self.assertEqual(g["char%dmask" % i]["inputs"], {
                    "image": ["char%dmaskimg" % i, 0], "channel": "red"})
                self.assertEqual(g["char%dclip" % i]["inputs"]["text"], "person %d" % i)
                self.assertEqual(g["char%dcond" % i]["inputs"]["mask"], ["char%dmask" % i, 0])
            final = ["charcombine%d" % count, 0]
            self.assertEqual(g["40"]["inputs"]["positive"], final)
            self.assertEqual(g["44"]["inputs"]["positive"], final)
            # the base text still covers the whole frame, first in the chain
            first_combine = g["charcombine1"]["inputs"]
            self.assertEqual(first_combine["conditioning_1"], ["10", 0])
            self.assertEqual(g["10"]["inputs"]["text"], "base scene")

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
        # Each file by name: list_workflows leaves out one that will not read,
        # and a broken template must fail here, not be skipped.
        names = sorted(n[:-5] for n in os.listdir(ig.WORKFLOWS_DIR) if n.endswith(".json"))
        self.assertTrue(names)
        for wf in (ig.load_workflow(n) for n in names):
            if wf.get("built_by"):            # finished in code: tested with its builder
                continue
            vals = {k: "x.safetensors" for k in (wf.get("files") or {})}
            vals.update(prompt="p", seed=1)
            if wf.get("multi_identity"):
                vals.update(face1="person.png", identity_boxes="[[0.2,0.1,0.8,0.6]]")
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
        import core.icons as studio_icons
        rgba, w, h = studio_icons.png_to_rgba(ig.region_png([0.5, 0.0, 1.0, 0.5], 800, 1600))
        self.assertEqual((w, h), (32, 64))
        at = lambda x, y: rgba[(y * w + x) * 4]
        self.assertEqual((at(20, 10), at(5, 10), at(20, 40)), (255, 0, 0))

    def test_add_pulid_chains_one_persons_extra_photos_at_the_lower_weight(self):
        graph = {"ks": {"class_type": "KSampler", "inputs": {"model": ["1", 0]}}}
        ig.add_pulid(graph, "pulid.safetensors",
                     [(["a.png", "b.png", "c.png", "d.png"], "mask.png")])
        # capped at REFERENCE_PHOTOS_MAX (3): a fourth photo is dropped, not chained.
        self.assertNotIn("pb_1_4", graph)
        self.assertEqual(graph["pb_1f"]["inputs"]["image"], "a.png")
        self.assertEqual(graph["pb_1"]["inputs"]["weight"], ig.PULID_BASE_WEIGHT)
        self.assertEqual(graph["pb_1"]["inputs"]["model"], ["1", 0])
        self.assertEqual(graph["pb_1_2f"]["inputs"]["image"], "b.png")
        self.assertEqual(graph["pb_1_2"]["inputs"]["weight"], ig.PULID_EXTRA_WEIGHT)
        self.assertEqual(graph["pb_1_2"]["inputs"]["model"], ["pb_1", 0])
        self.assertEqual(graph["pb_1_3f"]["inputs"]["image"], "c.png")
        self.assertEqual(graph["pb_1_3"]["inputs"]["model"], ["pb_1_2", 0])
        # the same mask - one person, one region - covers every one of their photos.
        for n in ("pb_1", "pb_1_2", "pb_1_3"):
            self.assertEqual(graph[n]["inputs"]["attn_mask"], ["pb_1k", 0])
        self.assertEqual(graph["ks"]["inputs"]["model"], ["pb_1_3", 0])
        # a bare string (one photo) still works exactly as before.
        graph2 = {"ks": {"class_type": "KSampler", "inputs": {"model": ["1", 0]}}}
        ig.add_pulid(graph2, "pulid.safetensors", [("a.png", None)])
        self.assertEqual(graph2["pb_1"]["inputs"]["weight"], ig.PULID_BASE_WEIGHT)
        self.assertNotIn("pb_1_2", graph2)
        self.assertNotIn("attn_mask", graph2["pb_1"]["inputs"])

    def test_the_face_graph_keeps_only_what_the_redraw_needs(self):
        wf = ig.load_workflow("flux_dev_baseline")
        crops = [{"x": 10, "y": 20, "width": 200, "height": 200},
                 {"x": 500, "y": 20, "width": 300, "height": 300}]
        g = ig.face_graph(wf, {"model": "m", "clip_l": "c", "t5": "t", "vae": "v",
                               "prompt": "p", "seed": 1, "face_prompt": "a face",
                               "refine": True},
                          [("sitter.safetensors", 0.8)], "pic.png [output]", crops, "oval.png",
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
                         "seed": 1, "refine": True}, [("sitter.safetensors", 0.85)])
        self.assertEqual(g["lora1"]["class_type"], "LoraLoader")
        self.assertEqual(g["40"]["inputs"]["model"], ["lora1", 0])
        self.assertEqual(g["44"]["inputs"]["model"], ["lora1", 0])
        self.assertEqual(g["10"]["inputs"]["clip"], ["lora1", 1])
        self.assertEqual(g["9"]["inputs"]["images"], ["45", 0])


class TestCompose(TempStudioMixin, unittest.TestCase):
    def chest_lora(self, kind="chest_female"):
        lib = self.studio.lib
        lib.save("loras", lib.all("loras") + [{
            "id": "chest", "file": "chest.safetensors", "name": "Chest control",
            "family": "flux1", "body_control": kind, "strength": 2}])
        return dict(FLUX_FILES, loras=FLUX_FILES["loras"] | {"chest.safetensors"})

    def test_chest_slider_loads_signed_lora_strength_into_graph(self):
        inv = self.chest_lora()
        for step, strength in ((-3, -2), (1, 0.667), (3, 2)):
            p = self.plan(inventory=inv, subject="a woman", chest_size=step)
            self.assertEqual(p.errors, [])
            self.assertEqual(p.loras, [("chest.safetensors", strength)])
            graph = ig.fill(p.workflow, p.values, p.loras)
            self.assertEqual(graph["lora1"]["inputs"]["strength_model"], strength)
        self.assertEqual(self.plan(inventory=inv, subject="a woman", chest_size=0).loras, [])

    def test_male_chest_uses_size_words_without_negative_lora_strength(self):
        inv = self.chest_lora("chest_male")
        small = self.plan(inventory=inv, subject="a man", chest_size=-3)
        large = self.plan(inventory=inv, subject="a man", chest_size=3)
        self.assertEqual(small.loras, large.loras)
        self.assertIn("Flat Male Chest", small.prompt)
        self.assertIn("Large Male Chest", large.prompt)

    def test_chest_control_reads_scene_look_and_does_not_change_a_group(self):
        inv = self.chest_lora()
        woman = {"asset": "person", "look": {"subject": "a woman", "chest_size": 3}}
        p = self.plan(inventory=inv, scene="a portrait", scene_layout={"objects": [woman]})
        self.assertEqual(p.loras, [("chest.safetensors", 2)])
        p = self.plan(inventory=inv, scene="two people", scene_layout={"objects": [woman, woman]})
        self.assertEqual(p.loras, [])
        self.assertTrue(any("whole image" in w for w in p.warnings))

    def test_missing_or_wrong_subject_chest_lora_is_explained(self):
        self.chest_lora()
        for subject in ("a man", "a woman", "a person"):
            p = self.plan(subject=subject, chest_size=3)
            self.assertEqual(p.loras, [])
            self.assertTrue(any("words only" in w for w in p.warnings))

    def plan(self, backend="5090", inventory=FLUX_FILES, nodes=None, **kw):
        s = ig.default_settings()
        s["model"] = "flux-hq"
        s.update(kw)
        return ig.compose(s, self.studio.lib, self.backend(backend), inventory, nodes=nodes)

    def test_the_chosen_lens_is_said_after_the_camera(self):
        phone = dict(scene="a street market", camera_profile="galaxy-s25-ultra")
        self.assertIn("Shot on a 23mm wide-angle lens at f/1.7", self.plan(**phone).prompt)
        self.assertIn("Shot on a 67mm lens at f/2.4",
                      self.plan(camera_lens=67.0, **phone).prompt)
        # A lens the camera does not have falls back to its native one.
        self.assertIn("23mm", self.plan(camera_lens=40.0, **phone).prompt)
        # The phone shoots 4:3.
        p = self.plan(**phone)
        w, h = p.values["width"], p.values["height"]
        self.assertAlmostEqual(max(w, h) / min(w, h), 4 / 3, delta=0.06)
        # A camera with no lenses listed says none, as before; a scene's own
        # words carry its lens instead.
        self.assertNotIn(" lens at f/", self.plan(scene="x", camera_profile="leica-m6").prompt)
        self.assertNotIn("Shot on a 23mm", self.plan(scene_layout={"objects": []}, **phone).prompt)

    def test_the_shot_on_camera_says_its_words_and_shapes_the_picture(self):
        plain = self.plan(scene="a harbour at dusk")
        w, h = plain.values["width"], plain.values["height"]
        leica = self.plan(scene="a harbour at dusk", camera_profile="leica-m6")
        self.assertIn("Kodak Portra 400", leica.prompt)
        lw, lh = leica.values["width"], leica.values["height"]
        self.assertAlmostEqual(max(lw, lh) / min(lw, lh), 1.5, delta=0.06)
        self.assertEqual(lh > lw, h > w)                  # held the same way up
        sx = self.plan(scene="a harbour at dusk", camera_profile="sx-70")
        self.assertEqual(sx.values["width"], sx.values["height"])
        typed = self.plan(scene="a harbour", camera_profile="sx-70", width=1344, height=768)
        self.assertEqual((typed.values["width"], typed.values["height"]), (1344, 768))
        # A scene's words already carry its camera: not said twice.
        scene = self.plan(scene="x", camera_profile="leica-m6", scene_layout={"objects": []})
        self.assertNotIn("Portra", scene.prompt)
        self.assertNotIn("Leica", self.plan(scene="a harbour", camera_profile="none").prompt)
        self.assertEqual(ig.camera_size("3:2", 1024, 1024), (1216, 832))
        self.assertEqual(ig.camera_size("", 896, 1152), (896, 1152))

    def test_character_regions_reach_the_plan_as_masked_images(self):
        d = tempfile.mkdtemp()
        mask1, mask2 = os.path.join(d, "a.png"), os.path.join(d, "b.png")
        for path in (mask1, mask2):
            with open(path, "wb") as f:
                f.write(PNG)
        p = self.plan(model="z-image-turbo", scene="a beer hall", character_regions=[
            {"prompt": "the user, lederhosen", "mask_path": mask1},
            {"prompt": "Partner, blue dress", "mask_path": mask2}])
        self.assertEqual(p.errors, [])
        self.assertEqual(p.values["character_regions"],
                         [{"prompt": "the user, lederhosen", "mask_var": "char_mask_1"},
                          {"prompt": "Partner, blue dress", "mask_var": "char_mask_2"}])
        self.assertEqual(p.images["char_mask_1"], mask1)
        self.assertEqual(p.images["char_mask_2"], mask2)

    def test_a_missing_character_mask_is_skipped_with_a_warning(self):
        p = self.plan(model="z-image-turbo", scene="a beer hall", character_regions=[
            {"prompt": "the user, lederhosen", "mask_path": "C:\\gone.png"}])
        self.assertNotIn("character_regions", p.values)
        self.assertTrue(any("not on this PC" in w for w in p.warnings), p.warnings)

    def test_character_regions_on_a_workflow_without_regional_conditioning_are_explained(self):
        p = self.plan(model="flux-hq", scene="a beer hall", character_regions=[
            {"prompt": "the user, lederhosen", "mask_path": "C:\\gone.png"}])
        self.assertNotIn("character_regions", p.values)
        self.assertTrue(any("does not" in n for n in p.notes), p.notes)

    def test_flux_applies_identity_and_style_loras_and_refines(self):
        p = self.plan(model="flux-dev", scene="On a pier.", preset="hq_final",
                      identities=[{"id": "sitter", "strength": 0.9}],
                      loras=[{"id": "sx70", "strength": 0.5}])
        self.assertEqual(p.errors, [])
        self.assertEqual(p.loras, [("sitter.safetensors", 0.9), ("sx70.safetensors", 0.5)])
        self.assertTrue(p.values["refine"])
        self.assertTrue(p.prompt.startswith("SITTERPERSON"), p.prompt)

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

    def test_an_always_on_loras_trigger_reaches_the_prompt_same_as_a_picked_one(self):
        lib = self.studio.lib
        recs = lib.all("loras")
        for r in recs:
            if r["id"] == "sx70":
                r["always"] = True
                r["trigger"] = "sx70 photo, polaroid frame"
        lib.save("loras", recs)
        p = self.plan(model="flux-dev", scene="x")
        self.assertIn("sx70 photo, polaroid frame", p.prompt)

    def always_on(self, n=4, strength=0.8, missing=()):
        """`n` Z-Image LoRAs switched to Always on as Add-ons imports them,
        each with its trigger; those in `missing` are not on the backend.
        -> the backend's inventory."""
        lib = self.studio.lib
        names = ["snapshot", "detail", "instant", "afterdark", "skin", "grain"][:n]
        lib.save("loras", lib.all("loras") + [
            {"id": k, "file": k + ".safetensors", "name": k.title(), "family": "z-image",
             "strength": strength, "always": True, "trigger": "trig_" + k} for k in names])
        return dict(FLUX_FILES, loras=FLUX_FILES["loras"] | {
            k + ".safetensors" for k in names if k not in missing})

    def test_always_on_loras_share_what_the_model_takes(self):
        # Live on the 5090, 2026-09-29: four always-on LoRAs at the 0.8 each
        # was imported with (3.2) made a fox at dawn a night scene with a grid
        # across it, and a Scene Builder picture a smear; at 1.2 both were
        # clean pictures of what was asked.
        inv = self.always_on(4)
        p = self.plan(model="z-image-turbo", scene="A fox in snow.", inventory=inv)
        self.assertEqual(p.errors, [])
        self.assertEqual(ig.load_workflow("zimage_hq")["defaults"]["lora_budget"], 1.2)
        self.assertEqual([s for _, s in p.loras], [0.3, 0.3, 0.3, 0.3])
        self.assertEqual([m["strength"] for m in p.lora_meta], [0.3] * 4)
        said = [w for w in p.warnings if w.startswith("Always-on LoRAs turned down")]
        self.assertEqual(len(said), 1, p.warnings)
        self.assertIn("Snapshot to 0.3, Detail to 0.3, Instant to 0.3 and Afterdark to 0.3",
                      said[0])
        self.assertIn("come to 3.2", said[0])
        self.assertIn("takes about 1.2", said[0])
        for k in ("snapshot", "afterdark"):
            self.assertIn("trig_" + k, p.prompt)       # turned down, still given
        # One, or two that fit, are as the library has them: nothing is said.
        for n, strength in ((1, 0.8), (2, 0.6), (3, 0.4)):
            self.setUp()
            inv = self.always_on(n, strength)
            p = self.plan(model="z-image-turbo", scene="A fox in snow.", inventory=inv)
            self.assertEqual([s for _, s in p.loras], [strength] * n)
            self.assertFalse(any("turned down" in w for w in p.warnings), p.warnings)

    def test_a_lora_chosen_for_the_picture_keeps_its_strength(self):
        inv = self.always_on(4)
        p = self.plan(model="z-image-turbo", scene="A fox.", inventory=inv,
                      loras=[{"id": "snapshot", "strength": 0.9}])
        by = {m["id"]: (m["strength"], m["why"]) for m in p.lora_meta}
        self.assertEqual(by["snapshot"], (0.9, "added"))
        # The other three share what it leaves: 0.3 of 1.2, 0.1 each.
        self.assertEqual([by[k] for k in ("detail", "instant", "afterdark")],
                         [(0.1, "always on")] * 3)
        self.assertAlmostEqual(sum(s for s, _ in by.values()), 1.2, places=2)
        # Chosen LoRAs over what the model takes are said, not changed, and
        # leave the always-on ones no room.
        p = self.plan(model="z-image-turbo", scene="A fox.", inventory=inv,
                      loras=[{"id": "snapshot", "strength": 0.9},
                             {"id": "detail", "strength": 0.8}])
        self.assertEqual(p.loras, [("snapshot.safetensors", 0.9), ("detail.safetensors", 0.8)])
        self.assertTrue(any(w.startswith("The LoRAs chosen for this picture come to 1.7")
                            for w in p.warnings), p.warnings)
        self.assertTrue(any(w.startswith("Always-on LoRAs left out (Instant and Afterdark)")
                            for w in p.warnings), p.warnings)
        self.assertNotIn("trig_instant", p.prompt)
        self.assertIn("trig_snapshot", p.prompt)

    def test_the_budget_is_the_workflows_and_a_pictures_own_to_set(self):
        inv = self.always_on(4)
        # FLUX names none: its LoRAs are as they were.
        lib = self.studio.lib
        recs = lib.all("loras")
        for r in recs:
            if r["id"] in ("sitter", "sx70"):
                r["always"], r["strength"] = True, 0.9
        lib.save("loras", recs)
        p = self.plan(model="flux-dev", scene="x", inventory=inv)
        self.assertEqual(sorted(s for _, s in p.loras), [0.9, 0.9])
        self.assertFalse(any("turned down" in w for w in p.warnings), p.warnings)
        # A picture's own: 0 is no limit, another number is that one.
        p = self.plan(model="z-image-turbo", scene="x", inventory=inv, lora_budget=0)
        self.assertEqual([s for _, s in p.loras], [0.8] * 4)
        p = self.plan(model="z-image-turbo", scene="x", inventory=inv, lora_budget=2.0)
        self.assertEqual([s for _, s in p.loras], [0.5] * 4)

    def test_hold_loras(self):
        def stack(*items):
            return [[{"name": n}, s, why, n + ".safetensors"] for n, s, why in items]
        a = stack(("A", 0.8, "always on"), ("B", -0.8, "always on"))
        self.assertEqual(ig.hold_loras(a, 0, "M"), [])                  # no limit
        self.assertEqual(ig.hold_loras(a, 1.6, "M"), [])                # it fits
        self.assertEqual([x[1] for x in a], [0.8, -0.8])
        said = ig.hold_loras(a, 1.2, "M")                               # a slider's minus is kept
        self.assertEqual([x[1] for x in a], [0.6, -0.6])
        self.assertEqual(len(said), 1)
        b = stack(("Id", 1.0, "identity Ann"), ("Style", 0.5, "style S"))
        said = ig.hold_loras(b, 1.2, "M")
        self.assertEqual([x[1] for x in b], [1.0, 0.5])                 # chosen: said only
        self.assertEqual(len(said), 1)
        self.assertIn("1.5", said[0])

    def test_a_lora_left_out_does_not_leave_its_trigger_behind(self):
        # Live, 2026-09-29: a LoRA whose file was not on the machine was left
        # out, and its trained words ("... detailed skin pore ...") went into
        # the prompt of every picture all the same - a fox's too.
        inv = self.always_on(2, 0.5, missing=("detail",))
        p = self.plan(model="z-image-turbo", scene="A fox in snow.", inventory=inv)
        self.assertEqual(p.loras, [("snapshot.safetensors", 0.5)])
        self.assertTrue(any("Detail" in w and "left out" in w for w in p.warnings), p.warnings)
        self.assertEqual(p.prompt, "A fox in snow. trig_snapshot.")
        # Another family's, added to the form by an old record: the same.
        p = self.plan(model="z-image-turbo", scene="A fox in snow.", inventory=inv,
                      loras=[{"id": "sitter", "strength": 0.8}])
        self.assertNotIn("SITTERPERSON", p.prompt)

    def test_an_identitys_or_styles_trigger_goes_with_its_lora(self):
        # Idea 553c1022fa02: only added and always-on LoRAs lost their trigger
        # when left out; a person's FLUX LoRA on Z-Image still had its word said.
        lib = self.studio.lib
        styles = lib.all("styles")
        for st in styles:
            if st["id"] == "sx70-authentic":
                st["trigger"] = "sx70 frame"
        lib.save("styles", styles)
        chosen = dict(identities=[{"id": "sitter", "strength": 0.9}], style="sx70-authentic",
                      scene="On a pier.")
        p = self.plan(model="z-image-turbo", **chosen)      # both LoRAs are FLUX.1's
        self.assertEqual(p.loras, [])
        self.assertNotIn("SITTERPERSON", p.prompt)
        self.assertNotIn("sx70 frame", p.prompt)
        self.assertIn("SX-70 instant film", p.prompt)       # a style's words are words
        p = self.plan(model="flux-dev", **chosen)           # given: said
        self.assertIn("SITTERPERSON", p.prompt)
        self.assertIn("sx70 frame", p.prompt)
        # An identity with no LoRA keeps its trigger, as words.
        p = self.plan(model="z-image-turbo", identities=[{"id": "partner", "strength": 0.8}],
                      scene="On a pier.")
        self.assertIn("PARTNERPERSON", p.prompt)

    def test_a_saved_lora_mix_is_a_preset_on_its_built_in(self):
        lib = self.studio.lib
        lib.save("presets", [
            {"name": "Film look", "base": "hq_final",
             "loras": [{"id": "sx70", "strength": 0.4}, {"id": "sx70", "strength": 1},
                       {"id": "gone", "strength": 5}]},
            {"name": "Standard", "base": "nonsense"}])
        film, std = lib.all("presets")
        self.assertEqual(film["id"], "film-look")
        self.assertEqual(film["loras"], [{"id": "sx70", "strength": 0.4},
                                         {"id": "gone", "strength": 2.0}])
        self.assertEqual((std["id"], std["base"]), ("mix-standard", "standard"))
        info = ig.preset_info(lib, "film-look")
        self.assertTrue(info["custom"])
        self.assertEqual((info["label"], info["role"]), ("Film look", "hires"))
        self.assertIn("SX-70 0.4", info["about"])
        self.assertIn("gone (missing)", info["about"])
        self.assertFalse(ig.preset_info(lib, "hq_final")["custom"])
        self.assertEqual(ig.preset_info(lib, "no-such")["label"], "Standard")
        # The mix brings its built-in's values; its LoRAs come from the rows.
        p = self.plan(model="flux-dev", scene="x", preset="film-look",
                      loras=[{"id": "sx70", "strength": 0.4}])
        self.assertTrue(p.values["refine"])
        self.assertEqual(p.loras, [("sx70.safetensors", 0.4)])
        lib.save("presets", [{"name": "Me", "base": "identity"}])
        p = self.plan(model="flux-dev", scene="x", preset="me")
        self.assertTrue(any("needs a person" in e for e in p.errors), p.errors)

    def test_z_image_is_the_default_model(self):
        self.assertEqual(ig.default_settings()["model"], "z-image-turbo")

    def test_the_face_pass_needs_sam3_and_says_so(self):
        p = self.plan(model="flux-dev", scene="a woman", preset="hq_final")
        self.assertFalse(p.values["face_detail"])
        self.assertTrue(any("no SAM3 checkpoint" in w for w in p.warnings), p.warnings)
        p = self.plan(model="flux-dev", scene="a woman")        # not asked: not said
        self.assertFalse(any("SAM3" in w for w in p.warnings), p.warnings)

    def test_the_face_pass_is_for_pictures_of_people(self):
        # High Quality Final on "a red fox in snow": SAM3 found the fox's face
        # and the pass, which draws "a real human face ... natural lips and
        # teeth", gave it a person's mouth (live, 2026-09-30).
        inv = dict(FLUX_FILES, checkpoints={"sam3.pt"})
        for model in ("z-image-turbo", "flux-dev"):
            p = self.plan(model=model, scene="A red fox sitting in fresh snow at dawn.",
                          preset="hq_final", inventory=inv, nodes=FaceClient.NODES)
            self.assertEqual(p.errors, [])
            self.assertFalse(p.values["face_detail"], model)
            self.assertNotIn("sam3", p.values)
            self.assertIn("No face pass: the picture's words name no person.", p.notes)
            self.assertFalse(any("SAM3" in w or "face" in w.lower() for w in p.warnings),
                             p.warnings)
            self.assertTrue(p.values["refine"])                 # the rest of the preset stays
            for scene in ("A man feeding a fox.", "A chef plating a dish."):
                p = self.plan(model=model, scene=scene, preset="hq_final", inventory=inv,
                              nodes=FaceClient.NODES)
                self.assertTrue(p.values["face_detail"], (model, scene))
        # Someone chosen on the form is a person, whatever the scene says.
        p = self.plan(model="flux-dev", scene="In the snow.", subject="a woman",
                      preset="hq_final", inventory=inv, nodes=FaceClient.NODES)
        self.assertTrue(p.values["face_detail"])
        p = self.plan(model="flux-dev", scene="In the snow.", identities=["sitter"],
                      preset="identity", inventory=inv, nodes=FaceClient.NODES)
        self.assertTrue(p.values["face_detail"])
        # Not asked for: nothing is said.
        p = self.plan(model="flux-dev", scene="A red fox.", inventory=inv)
        self.assertFalse(any("face pass" in n.lower() for n in p.notes), p.notes)

    def test_the_face_pass_is_planned_with_the_person_in_its_prompt(self):
        inv = dict(FLUX_FILES, checkpoints={"sam3.pt"})
        p = self.plan(model="flux-dev", scene="On a pier.", preset="identity",
                      identities=["sitter"], inventory=inv, nodes=FaceClient.NODES)
        self.assertEqual(p.errors, [])
        self.assertTrue(p.values["face_detail"])
        self.assertEqual(p.values["sam3"], "sam3.pt")
        self.assertIn("SITTERPERSON", p.values["face_prompt"])
        p = self.plan(model="flux-dev", scene="x", preset="identity", identities=["sitter"],
                      inventory=inv, nodes=FaceClient.NODES - {"SAM3_Detect"})
        self.assertFalse(p.values["face_detail"])
        self.assertTrue(any("SAM3_Detect" in w for w in p.warnings), p.warnings)
        p = self.plan(model="flux-dev", scene="x", preset="identity", identities=["sitter"],
                      inventory=inv, face_detail=False)
        self.assertFalse(p.values["face_detail"])

    def test_missing_nodes_are_named(self):
        p = self.plan(model="flux-dev", scene="x", nodes={"UNETLoader"})
        self.assertTrue(any("FluxGuidance" in e and "KSampler" in e for e in p.errors),
                        p.errors)

    def test_person_style_scene(self):
        p = self.plan(identities=[{"id": "sitter", "strength": 0.9}], style="sx70-authentic",
                      scene="At Munich Oktoberfest, raising a stein.", seed=3)
        self.assertEqual(p.errors, [])
        self.assertTrue(p.prompt.startswith(
            "SITTERPERSON, fully clothed, wearing clothes suited to the scene. "
            "At Munich Oktoberfest"), p.prompt)
        self.assertIn("SX-70 instant film", p.prompt)
        self.assertEqual(p.loras, [("sitter.safetensors", 0.9), ("sx70.safetensors", 0.55)])
        self.assertEqual(p.values["seed"], 3)

    def test_person_attributes_and_camera(self):
        p = self.plan(style="none", scene="On a pier at dusk.", subject="a woman in her 30s",
                      hair="auburn", eyes="green", build="slim", traits="freckles",
                      camera="85mm, shallow depth of field", anatomy=False)
        self.assertEqual(p.errors, [])
        self.assertEqual(p.prompt, "a woman in her 30s, slim build, green eyes, auburn hair, "
                                   "freckles, fully clothed, wearing clothes suited to the "
                                   "scene. On a pier at dusk. 85mm, shallow depth of field.")

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

    def test_chest_size_is_in_both_subjects_prompts_and_character_looks(self):
        self.assertIn("chest_size", ig.CHARACTER_KEYS)
        for subject in ("a man", "a woman"):
            self.assertEqual(ig.person_text({"subject": subject, "chest_size": 0}), subject)
            self.assertIn("small chest", ig.person_text({"subject": subject, "chest_size": -2}))
            self.assertIn("full chest", ig.person_text({"subject": subject, "chest_size": 2}))

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
        self.assertIn("Every person has exactly two hands, each with four fingers and a "
                      "thumb, two feet, two eyes and a proportionate body", p.prompt)
        # "Drawn correctly: ..." made photographs illustrations (live, 2026-09-29).
        self.assertNotIn("drawn", p.prompt.lower())
        self.assertIn("extra fingers", p.negative)
        p = self.plan(style="none", scene="Two dancers on a stage.")
        self.assertIn("four fingers and a thumb", p.prompt)          # the scene names people
        p = self.plan(style="none", scene="A red bicycle against a white wall.")
        self.assertNotIn("fingers", p.prompt + p.negative)            # nobody in it
        p = self.plan(style="none", subject="a woman", anatomy=False)
        self.assertNotIn("fingers", p.prompt + p.negative)

    def test_a_workflow_can_leave_the_anatomy_constants_unsaid(self):
        # Z-Image Turbo: told of hands and fingers, it made them the picture
        # and cut the head off (live, 2026-09-29). The switch is still there,
        # so a note says why it did nothing.
        p = self.plan(model="z-image-turbo", style="none", subject="a woman",
                      scene="Waving hello.")
        self.assertEqual(p.errors, [])
        self.assertNotIn("fingers", p.prompt + p.negative)
        self.assertTrue(any("anatomy constants" in n for n in p.notes), p.notes)
        p = self.plan(model="z-image-turbo", style="none", subject="a woman", anatomy=False)
        self.assertFalse(any("anatomy constants" in n for n in p.notes), p.notes)
        p = self.plan(model="z-image-turbo", style="none", scene="A red bicycle.")
        self.assertFalse(any("anatomy constants" in n for n in p.notes), p.notes)

    def test_nothing_is_added_that_the_user_did_not_say(self):
        # A light and a small flaw, picked by the seed, were to be added to a
        # no-style prompt (2026-09-28) and never were: "No style" is itself a
        # style. Run live they cut heads off to show the flaw ("a loose thread
        # on a sleeve") and put a lamp in the picture ("a single lamp at
        # night"), so they are gone, with or without a style.
        for style in ("none", ""):
            for seed in (11, 22, 33, 44):
                p = self.plan(style=style, subject="a chef", scene="Plating a dish.",
                              seed=seed, anatomy=False)
                self.assertEqual(p.prompt, "a chef, fully clothed, wearing clothes suited "
                                           "to the scene. Plating a dish.", (style, seed))

    def test_identity_joins_the_described_person(self):
        p = self.plan(identities=["sitter"], style="none", hair="grey", scene="Reading.")
        self.assertTrue(p.prompt.startswith("SITTERPERSON, grey hair, fully clothed, wearing "
                                            "clothes suited to the scene. Reading."), p.prompt)

    def test_identity_descriptions_tell_two_undrawn_people_apart(self):
        """Without Scene Builder's own boxes, every person used to be labelled
        identically "from the left" - indistinguishable text that could make
        the model apply one person's features to the other's face."""
        alice = {"name": "Alice", "description": "red hair, freckles"}
        bob = {"name": "Bob", "description": "beard, glasses"}
        parts = ig.identity_description_text({}, None, [(alice, None), (bob, None)])
        self.assertEqual(len(parts), 3)          # Alice, Bob, the closing instruction
        self.assertNotEqual(parts[0], parts[1])
        self.assertIn("Alice", parts[0])
        self.assertIn("Bob", parts[1])
        self.assertIn("1 of 2", parts[0])
        self.assertIn("2 of 2", parts[1])

    def test_a_person_alone_is_enough(self):
        p = self.plan(style="none", subject="an old fisherman", anatomy=False)
        self.assertEqual(p.errors, [])
        self.assertEqual(p.prompt, "an old fisherman, fully clothed, wearing clothes suited "
                                   "to the scene.")

    def test_nobody_is_left_undressed(self):
        p = self.plan(style="none", subject="a woman", chest_size=2, anatomy=False)
        self.assertIn(ig.CLOTHED + ", " + ig.COVERED, p.prompt)
        p = self.plan(style="none", scene="A nude woman on a beach.", anatomy=False)
        self.assertEqual(p.prompt, "A nude woman on a beach. Fully clothed.")   # a floor
        p = self.plan(identities=["sitter"], style="none", scene="Reading.", anatomy=False)
        self.assertIn("SITTERPERSON, " + ig.CLOTHED + ", " + ig.COVERED, p.prompt)
        for dressed in ({"top": "hoodie"}, {"bottom": "blue jeans"},
                        {"scene": "A woman in a red dress."},
                        {"scene": "A man wearing a wetsuit."}):
            p = self.plan(style="none", subject="a person", anatomy=False, **dressed)
            self.assertNotIn(ig.COVERED, p.prompt, dressed)     # a garment is named
            self.assertIn(", " + ig.CLOTHED, p.prompt, dressed)  # and still said to be on
        p = self.plan(style="none", scene="A red bicycle on top of a hill.")
        self.assertNotIn(ig.COVERED, p.prompt)                  # nobody in it
        self.assertNotIn("clothed", p.prompt.lower())

    def test_the_clothing_floor_is_said_of_the_person_and_names_no_skin(self):
        # Live on the 5090, 2026-09-29, the same seeds: the sentence this
        # replaces ("Every person is clothed, the chest fully covered by their
        # clothing or swimwear") drew what it named - swimsuits in a garden,
        # a man knitting in his briefs, a woman walking a dog topless.
        for p in (self.plan(style="none", subject="a woman", chest_size=3,
                            scene="Walking a dog in a park.", anatomy=False),
                  self.plan(style="none", scene="A man knitting in an armchair.",
                            anatomy=False)):
            for word in ("chest fully", "swimwear", "covered", "Every person is"):
                self.assertNotIn(word, p.prompt)
        # A scene that describes its people itself has no one to hang the
        # floor on: it follows the scene, a sentence of its own, and no
        # "wearing..." without a wearer opens the prompt.
        p = self.plan(style="none", scene="A man knitting in an armchair.", anatomy=False)
        self.assertEqual(p.prompt, "A man knitting in an armchair. Fully clothed.")
        p = self.plan(model="z-image-turbo", style="none",
                      scene="An elderly man knitting a red scarf in an armchair.")
        self.assertEqual(p.prompt, "An elderly man knitting a red scarf in an armchair. "
                                   "Fully clothed.")

    def test_a_swimsuit_still_covers_the_chest(self):
        # Drawn topless on Z-Image Turbo, 2026-09-27: the swimsuit alone lost
        # to "very full chest" and "natural anatomy".
        p = self.plan(identities=["sitter"], style="none", subject="a woman", build="curvy",
                      chest_size=3, scene="sitterperson in a swimsuit")
        self.assertNotIn(ig.COVERED, p.prompt)                  # a garment is named
        self.assertIn("very full chest, " + ig.CLOTHED + ". sitterperson in a swimsuit",
                      p.prompt)
        self.assertNotIn("natural anatomy", p.prompt)
        self.assertNotIn("Anatomically", p.prompt)

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
        rec = ig.clean_character({"name": "Mara", "identity": "partner",
                                  "looks": {"hair": "auburn", "weight": -1, "muscle": 0,
                                            "expression": "sad", "top": "hoodie",
                                            "nonsense": "x"},
                                  "item_refs": {"hoodie": "C:/h.png", "": "x"}})
        self.assertEqual(rec["id"], "mara")
        self.assertEqual(rec["looks"], {"hair": "auburn", "weight": -1, "top": "hoodie"})
        self.assertEqual(rec["item_refs"], {"hoodie": "C:/h.png"})
        self.assertEqual(rec["faces"], [])
        self.assertEqual(ig.clean_character({"name": "M", "faces": ["C:/f.jpg", 3]})["faces"],
                         ["C:/f.jpg"])

    def test_random_looks_fill_the_creator(self):
        import random
        looks = ig.random_looks(random.Random(4))
        self.assertTrue(looks["subject"])
        self.assertNotIn("expression", looks)
        self.assertTrue(all(k in ig.CHARACTER_KEYS for k in looks))
        only = ig.random_looks(random.Random(4), sections=["Hair"])
        self.assertEqual(set(only), {"hair", "hair_style"})

    def test_two_people_in_one_picture(self):
        p = self.plan(identities=["sitter", "partner"], scene="Dancing.")
        self.assertTrue(p.prompt.startswith("SITTERPERSON and PARTNERPERSON, fully clothed, "
                                            "wearing clothes suited to the scene. Dancing."),
                        p.prompt)

    def test_identity_carries_no_style(self):
        p = self.plan(identities=["sitter"], scene="Portrait.", style="none")
        self.assertEqual([m["id"] for m in p.lora_meta], ["sitter"])
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
        p = self.plan(inventory=None, identities=["sitter"], scene="x")
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

    def test_on_klein_a_swapped_persons_face_is_from_their_photos_not_a_warning(self):
        # Klein takes no PuLID, but the head swap and face swap still put the
        # person's photos on after drawing: the plan said "the face comes from
        # the words" on a picture that went through both (2026-10-02).
        inv = dict(FLUX_FILES, diffusion_models={"flux-2-klein-base-9b.safetensors"},
                   text_encoders={"qwen_3_8b_fp8mixed.safetensors"},
                   vae={"flux2-vae.safetensors"})
        form = dict(model="klein-9b", inventory=inv, scene="a chef",
                    face_photos=[__file__], face_name="Partner")
        p = self.plan(identities=["partner"], **form)
        self.assertFalse(any("Face:" in w for w in p.warnings), p.warnings)
        self.assertIn("Face: Partner, from their photos by the head swap and face swap "
                      "after drawing.", p.notes)
        p = self.plan(identities=["partner"], head_swap=False, **form)
        self.assertIn("Face: Partner, from their photos by the face swap after drawing.",
                      p.notes)
        # No one to swap: the face really is the words, and that is said.
        p = self.plan(**form)
        self.assertTrue(any("comes from the words" in w for w in p.warnings), p.warnings)

    def test_klein_takes_a_pose_as_a_reference_picture_with_words_to_copy_it(self):
        # Klein has no ControlNet: the pose map is a reference latent on both
        # sides of its CFG, and the words before the prompt say to copy it.
        src = os.path.join(self.dir, "pose.png")
        with open(src, "wb") as f:
            f.write(PNG)
        inv = dict(FLUX_FILES, diffusion_models={"flux-2-klein-base-9b.safetensors"},
                   text_encoders={"qwen_3_8b_fp8mixed.safetensors"},
                   vae={"flux2-vae.safetensors"})
        p = self.plan(model="klein-9b", inventory=inv, scene="a dancer",
                      references={"pose": src}, pose={"strength": 0.6})
        self.assertEqual(p.errors, [])
        self.assertEqual(p.images, {"pose_image": src})
        self.assertFalse(any("Pose" in w for w in p.warnings), p.warnings)
        g = ig.fill(p.workflow, dict(p.values, pose_image="pose.png"))
        self.assertNotIn("10", g)
        self.assertTrue(g["13"]["inputs"]["text"].startswith("Apply the pose from image 1"))
        self.assertIn("a dancer", g["13"]["inputs"]["text"])
        self.assertEqual(g["50"]["inputs"]["image"], "pose.png")
        self.assertEqual(g["52"]["inputs"]["pixels"], ["51", 0])
        self.assertEqual(g["53"]["inputs"], {"conditioning": ["13", 0], "latent": ["52", 0]})
        self.assertEqual(g["54"]["inputs"], {"conditioning": ["12", 0], "latent": ["52", 0]})
        self.assertEqual((g["6"]["inputs"]["positive"], g["6"]["inputs"]["negative"]),
                         (["53", 0], ["54", 0]))
        # The negative words stay words: a switch named "negative" once put
        # its link in node 12's text, and ComfyUI refused the graph.
        self.assertIsInstance(g["12"]["inputs"]["text"], str)
        # Without a pose: the plain prompt, nothing of the pose in the graph.
        p = self.plan(model="klein-9b", inventory=inv, scene="a dancer")
        g = ig.fill(p.workflow, p.values)
        self.assertFalse({"13", "50", "51", "52", "53", "54"} & set(g))
        self.assertEqual((g["6"]["inputs"]["positive"], g["6"]["inputs"]["negative"]),
                         (["10", 0], ["12", 0]))
        self.assertIsInstance(g["12"]["inputs"]["text"], str)

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

    ESRGAN = "RealESRGAN_x4plus.safetensors"

    def test_the_refine_pass_enlarges_with_the_upscale_model_a_backend_has(self):
        # ComfyUI's own Z-Image upscaler recipe: RealESRGAN x4, then down to
        # the size asked, then the redraw. Lanczos alone gave the redraw a
        # blur to sharpen; the model alone smoothed skin and drew hairs on
        # knitwear, so the two are laid half and half (live, 2026-09-30).
        wf = ig.load_workflow("zimage_hq")
        self.assertEqual(wf["defaults"]["upscale_model"], self.ESRGAN)
        self.assertEqual(wf["defaults"]["refine_blend"], 0.5)
        inv = dict(FLUX_FILES, upscale_models={self.ESRGAN})
        p = self.plan(model="z-image-turbo", scene="x", preset="hq_final", inventory=inv,
                      nodes=FaceClient.NODES | ig.REFINE_MODEL_NODES)
        self.assertEqual(p.errors, [])
        self.assertFalse(any("Refine" in w for w in p.warnings), p.warnings)
        self.assertEqual((p.values["refine_model"], p.values["upscale_model"],
                          p.values["upscale"], p.values["refine_model_by"]),
                         (True, self.ESRGAN, 2.0, 0.5))
        g = ig.fill(p.workflow, p.values)
        self.assertEqual(g["46"]["inputs"], {"model_name": self.ESRGAN})
        self.assertEqual(g["47"]["inputs"], {"upscale_model": ["46", 0], "image": ["41", 0]})
        self.assertEqual((g["48"]["inputs"]["image"], g["48"]["inputs"]["scale_by"]),
                         (["47", 0], 0.5))
        self.assertEqual((g["42"]["inputs"]["image"], g["42"]["inputs"]["scale_by"]),
                         (["41", 0], 2.0))                          # lanczos's, beside it
        self.assertEqual(g["49"], {"class_type": "ImageBlend", "inputs": {
            "image1": ["48", 0], "image2": ["42", 0], "blend_factor": 0.5,
            "blend_mode": "normal"}})
        self.assertEqual(g["43"]["inputs"]["pixels"], ["49", 0])
        self.assertEqual(g["9"]["inputs"]["images"], ["45", 0])
        for nid in ("46", "47", "48", "49"):           # shown as "refining detail"
            self.assertIn(nid, p.workflow["stages"]["refining"])
        # The size a card holds still caps it, and the model's share follows.
        p = self.plan(backend="3090", model="z-image-turbo", scene="x", preset="hq_final",
                      width=1344, height=1344, inventory=inv)
        self.assertLess(p.values["upscale"], 2.0)
        self.assertEqual(p.values["refine_model_by"], round(p.values["upscale"] / 4, 6))
        # What the backend has is not known yet: asked for, as a pose's ControlNet is.
        p = self.plan(model="z-image-turbo", scene="x", preset="hq_final", inventory=None)
        self.assertTrue(p.values["refine_model"])
        # A preset that names no size of its own takes the workflow's (x1.5).
        p = self.plan(model="z-image-turbo", scene="a woman", preset="identity",
                      identities=["partner"], inventory=inv)
        self.assertEqual(p.errors, [])
        self.assertEqual(p.values["refine_model_by"], 0.375)
        g = ig.fill(p.workflow, p.values)
        self.assertEqual(g["48"]["inputs"]["scale_by"], 0.375)
        # The redraw after it is light and clean: measured live (2026-09-30),
        # the recipe's own dpmpp_2m_sde at 0.33 left scales on skin and kept
        # 0.59 of a face's likeness where euler_ancestral at 0.2 kept 0.84.
        k = g["44"]["inputs"]
        self.assertEqual((k["sampler_name"], k["scheduler"], k["steps"], k["denoise"]),
                         ("euler_ancestral", "beta", 5, 0.2))

    def test_without_the_upscale_model_the_refine_pass_is_lanczos_and_says_so(self):
        p = self.plan(model="z-image-turbo", scene="x", preset="hq_final")
        self.assertEqual(p.errors, [])                  # a finish, never a reason to fail
        self.assertTrue(p.values["refine"])
        self.assertNotIn("refine_model", p.values)
        self.assertNotIn("upscale_model", p.values)
        said = [w for w in p.warnings if w.startswith("Refine:")]
        self.assertEqual(len(said), 1, p.warnings)
        self.assertIn(self.ESRGAN, said[0])
        self.assertIn("ComfyUI/models/upscale_models", said[0])
        g = ig.fill(p.workflow, p.values)
        self.assertEqual(g["42"]["inputs"]["scale_by"], 2.0)
        self.assertEqual(g["43"]["inputs"]["pixels"], ["42", 0])
        for nid in ("46", "47", "48", "49"):
            self.assertNotIn(nid, g)
        # A blur takes a deeper redraw to become detail - unless the form says.
        self.assertEqual(g["44"]["inputs"]["denoise"], 0.25)
        p = self.plan(model="z-image-turbo", scene="x", preset="hq_final", refine_denoise=0.15)
        self.assertEqual(ig.fill(p.workflow, p.values)["44"]["inputs"]["denoise"], 0.15)
        # A ComfyUI without the nodes: the same, naming them.
        inv = dict(FLUX_FILES, upscale_models={self.ESRGAN})
        p = self.plan(model="z-image-turbo", scene="x", preset="hq_final", inventory=inv,
                      nodes=FaceClient.NODES)
        self.assertNotIn("refine_model", p.values)
        self.assertTrue(any("ImageUpscaleWithModel" in w for w in p.warnings), p.warnings)
        # No refine pass: nothing of it, and nothing said.
        p = self.plan(model="z-image-turbo", scene="x")
        self.assertNotIn("refine_model", p.values)
        self.assertNotIn("upscale_model", p.values)
        self.assertFalse(any("Refine" in w for w in p.warnings), p.warnings)
        # The model's readiness never waits on it.
        self.assertEqual(ig.missing_for(self.studio.lib.get("models", "z-image-turbo"),
                                        self.backend("5090"), FLUX_FILES), ([], []))

    def test_identity_preset_wants_a_person(self):
        self.assertTrue(self.plan(preset="identity", scene="x").errors)

    def test_preview_graph_uses_the_local_file_not_an_upload(self):
        src = os.path.join(self.dir, "frame.png")
        with open(src, "wb") as f:
            f.write(PNG)
        p = self.plan(model="flux-dev", scene="x", references={"source": src}, denoise=0.8)
        graph = ig.preview_graph(p)
        self.assertIsNotNone(graph)
        node = next(n for n in graph.values() if n["inputs"].get("image") == "frame.png")
        self.assertEqual(node["class_type"], "LoadImage")

    def test_preview_graph_is_none_without_a_workflow_or_with_errors(self):
        self.assertIsNone(ig.preview_graph(None))
        self.assertIsNone(ig.preview_graph(self.plan(preset="identity", scene="x")))


class TestStartBackend(TempStudioMixin, unittest.TestCase):
    """Start runs a backend's own start file - never the real one here."""

    def test_only_a_file_on_this_pc_can_be_started(self):
        cmd = os.path.join(self.dir, "Start ComfyUI.cmd")
        with open(cmd, "w") as f:
            f.write("@echo off\n")
        self.assertEqual(ig.Studio.start_file({"start": cmd}), cmd)
        self.assertEqual(ig.Studio.start_file({"start": '"%s"' % cmd}), cmd)
        self.assertIsNone(ig.Studio.start_file(
            {"start": "Start ComfyUI on the LLM PC with --listen."}))
        self.assertIsNone(ig.Studio.start_file({"start": cmd + ".gone"}))
        self.assertIsNone(ig.Studio.start_file({}))

    def test_a_cmd_starts_in_a_console_of_its_own_beside_itself(self):
        from unittest.mock import patch
        cmd = os.path.join(self.dir, "Start ComfyUI.cmd")
        with open(cmd, "w") as f:
            f.write("@echo off\n")
        b = dict(self.backend("5090"), start=cmd)
        with patch.object(ig.subprocess, "Popen") as popen:
            self.studio.start(b)
        args, kw = popen.call_args
        self.assertEqual(args[0], ["cmd", "/c", cmd])
        self.assertEqual(kw["cwd"], self.dir)
        self.assertEqual(kw["creationflags"], getattr(ig.subprocess, "CREATE_NEW_CONSOLE", 0))

    def test_no_second_start_while_the_first_console_is_open(self):
        # Idea 5902f8b35f18: after the UI's 180 s wait the Start button is
        # back while the first ComfyUI may still be loading; a second one on
        # the same port would fail over it.
        from unittest.mock import Mock, patch
        cmd = os.path.join(self.dir, "Start ComfyUI.cmd")
        with open(cmd, "w") as f:
            f.write("@echo off\n")
        b = dict(self.backend("5090"), start=cmd)
        console = Mock()
        console.poll.return_value = None                   # still open
        with patch.object(ig.subprocess, "Popen", return_value=console) as popen:
            self.studio.start(b)
            with self.assertRaises(ig.ComfyError) as caught:
                self.studio.start(b)
            self.assertEqual(popen.call_count, 1)
            self.assertIn("still open", str(caught.exception))
            console.poll.return_value = 1                  # closed: start again
            self.studio.start(b)
            self.assertEqual(popen.call_count, 2)

    def test_advice_is_not_run(self):
        from unittest.mock import patch
        with patch.object(ig.subprocess, "Popen") as popen:
            with self.assertRaises(ig.ComfyError) as caught:
                self.studio.start(self.backend("3090"))
        popen.assert_not_called()
        self.assertIn("--listen", str(caught.exception))


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

    def test_a_held_backend_takes_no_job_until_let_go(self):
        # Idea 575729b8a187: Build LoRA has the 5090's GPU; Generate, Fix and
        # Again must not load a FLUX beside the training.
        why = "5090 Workstation's GPU is building P's LoRA; send pictures there when it is done."
        self.studio.held["5090"] = why
        jobs = self.studio.submit(dict(ig.default_settings(), scene="x", batch=3))
        settle(jobs)
        self.assertEqual({j.backend["id"] for j in jobs}, {"3090"})
        with self.assertRaises(ig.ComfyError) as cm:                  # named by hand
            self.studio.submit(dict(ig.default_settings(), scene="x", backend="5090"))
        self.assertEqual(str(cm.exception), why)
        self.assertEqual(self.studio.plan_route(dict(ig.default_settings(), backend="5090")),
                         (None, why))
        FakeClient.down = {"3090"}                                     # nowhere else to go
        self.studio.check_all()
        with self.assertRaises(ig.ComfyError) as cm:
            self.studio.submit(dict(ig.default_settings(), scene="x"))
        self.assertIn("building P's LoRA", str(cm.exception))
        FakeClient.down = set()
        self.studio.check_all()
        del self.studio.held["5090"]
        jobs = self.studio.submit(dict(ig.default_settings(), scene="x", backend="5090"))
        settle(jobs)
        self.assertEqual(jobs[0].backend["id"], "5090")

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
    def test_a_refined_picture_says_what_enlarged_it_and_how_it_was_redrawn(self):
        esrgan = "RealESRGAN_x4plus.safetensors"

        class UpscaleClient(FakeClient):
            def inventory(self):
                return dict(super().inventory(), upscale_models={esrgan})

            def node_types(self):
                return super().node_types() | ig.REFINE_MODEL_NODES
        self.studio.client_factory = UpscaleClient
        jobs = self.studio.submit(dict(ig.default_settings(), scene="A lighthouse",
                                       backend="5090", model="z-image-turbo",
                                       preset="hq_final", face_detail=False, seed=7))
        settle(jobs)
        self.assertEqual(jobs[0].status, "complete", jobs[0].detail)
        graph = FakeClient.instances[-1].graphs[0]
        self.assertEqual(graph["46"]["class_type"], "UpscaleModelLoader")
        self.assertEqual(graph["43"]["inputs"]["pixels"], ["49", 0])
        self.assertEqual(self.studio.history.list()[0]["refine"], {
            "upscale": 2.0, "denoise": 0.2, "steps": 5, "sampler": "euler_ancestral",
            "scheduler": "beta", "model": esrgan})
        # On a ComfyUI without the model: lanczos, the deeper redraw, and no model named.
        self.studio.client_factory = FakeClient
        self.studio.clients, self.studio.inventories, self.studio.nodes = {}, {}, {}
        jobs = self.studio.submit(dict(ig.default_settings(), scene="A lighthouse",
                                       backend="5090", model="z-image-turbo",
                                       preset="hq_final", face_detail=False, seed=7))
        settle(jobs)
        self.assertEqual(jobs[0].status, "complete", jobs[0].detail)
        rec = self.studio.history.list()[0]
        self.assertEqual((rec["refine"]["model"], rec["refine"]["denoise"]), (None, 0.25))
        self.assertTrue(any(w.startswith("Refine:") for w in rec["warnings"]), rec["warnings"])

    def test_a_job_runs_and_lands_in_history(self):
        room = []
        self.studio.make_room = room.append
        src = os.path.join(self.dir, "src.png")
        with open(src, "wb") as f:
            f.write(PNG)
        jobs = self.studio.submit(dict(ig.default_settings(), scene="A fox", backend="3090",
                                       model="flux-hq", identities=["sitter"],
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
        self.assertEqual(rec["loras"][0]["file"], "sitter.safetensors")
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

    def fix_job(self, model="flux-dev", **fix):
        self.face_studio()
        src = os.path.join(self.dir, "made.png")
        with open(src, "wb") as f:
            f.write(PNG)
        w, h = ig.file_size_of(src)
        s = self.studio.fix_base(dict(ig.default_settings(), model=model, scene="On a pier.",
                                      backend="5090", face_detail=True, critic_notes=True))
        s.update(mode="fix", seed=5, fix=dict({"image": src, "target": "hand",
                                               "spots": [{"x": w // 2, "y": h // 2,
                                                          "size": 64}]}, **fix))
        jobs = self.studio.submit(s)
        settle(jobs)
        return jobs[0], FaceClient.instances[-1], src

    def test_a_fix_redraws_only_the_clicked_square(self):
        job, client, src = self.fix_job(strength="strong", words="left hand holding a cup")
        self.assertEqual(job.status, "complete", job.detail)
        (g,) = client.graphs                      # one run: no new picture is made
        self.assertNotIn("40", g)
        self.assertIn(src, client.uploads)
        self.assertEqual(g["fc1_4"]["inputs"]["denoise"], ig.FIX_STRENGTHS["strong"])
        self.assertIn("fc1_3n", g)                 # the oval's noise mask
        self.assertNotIn("fc1_h1", g)              # a hand is not blended as a head
        self.assertIn("left hand holding a cup", g["f10"]["inputs"]["text"])
        self.assertIn("five fingers", g["f10"]["inputs"]["text"])
        rec = self.studio.history.list()[0]
        self.assertEqual(rec["fix"]["spots"][0]["size"], 64)
        self.assertTrue(rec["prompt"].startswith("Fix left hand holding a cup"))
        again = ig.again(rec)
        self.assertEqual(again["mode"], "fix")    # Generate Again tries the fix again

    def test_a_z_image_picture_can_be_fixed(self):
        job, client, _ = self.fix_job(model="z-image-turbo")
        self.assertEqual(job.status, "complete", job.detail)
        g = client.graphs[0]
        self.assertEqual(g["f10"]["inputs"]["clip"], ["2", 0])   # Z-Image's own encoder
        self.assertEqual(g["fc1_4"]["inputs"]["cfg"], 1.0)

    def test_a_redraw_is_sampled_as_its_workflow_says_a_redraw_is(self):
        # Z-Image Turbo's own sampler adds no noise as it goes, and over what
        # is there it leaves specks on skin and beads in beards; of the clean
        # ones euler_ancestral on beta keeps a face nearest (ArcFace, live
        # 2026-09-29): its redraws are euler_ancestral's on beta.
        wf = ig.load_workflow("zimage_hq")
        self.assertEqual((wf["defaults"]["sampler"], wf["defaults"]["redraw_sampler"],
                          wf["defaults"]["redraw_scheduler"]),
                         ("res_multistep", "euler_ancestral", "beta"))
        job, client, _ = self.fix_job(model="z-image-turbo")
        k = client.graphs[0]["fc1_4"]["inputs"]
        self.assertEqual((k["sampler_name"], k["scheduler"], k["steps"]),
                         ("euler_ancestral", "beta", 8))
        # A sampler chosen for the picture is the picture's, not its redraws';
        # the steps are the picture's.
        values = dict(job.plan.values, sampler="euler", scheduler="simple", steps=9,
                      face_prompt="a hand", face_denoise=0.4, seed=1)
        g = ig.face_graph(wf, values, [], "made.png", ig.fix_crops(512, 512, [
            {"x": 200, "y": 200, "size": 64}]), "oval.png", "out")
        k = g["fc1_4"]["inputs"]
        self.assertEqual((k["sampler_name"], k["scheduler"], k["steps"]),
                         ("euler_ancestral", "beta", 9))
        # The Z-Image face pass is light: 0.3 keeps more of the face than 0.4
        # and still mends a 60 px face's eyes and teeth. FLUX keeps its 0.4.
        self.assertEqual(wf["defaults"]["face_denoise"], 0.3)
        inv = dict(FLUX_FILES, checkpoints={"sam3.pt"})
        for model, strength in (("z-image-turbo", 0.3), ("flux-dev", 0.4)):
            s = dict(ig.default_settings(), model=model, scene="a woman", preset="hq_final")
            p = ig.compose(s, self.studio.lib, self.backend("5090"), inv, nodes=FaceClient.NODES)
            self.assertEqual(p.errors, [])
            self.assertTrue(p.values["face_detail"])
            self.assertEqual(p.values["face_denoise"], strength, model)
        # Klein, base and distilled, has a face pass: one section, each
        # model's guidance its CFG (the base's real one, the distilled's 1),
        # the written negative, the native Flux2 scheduler; the base redraws in
        # 20 steps, not its 50.
        klein = ig.load_workflow("klein9b_base")
        inv = dict(FLUX_FILES, checkpoints={"sam3.pt"},
                   diffusion_models={"flux-2-klein-base-9b.safetensors",
                                     "flux-2-klein-9b-fp8.safetensors"},
                   text_encoders={"qwen_3_8b_fp8mixed.safetensors"},
                   vae={"flux2-vae.safetensors"})
        nodes = ig.face_nodes(klein) | set(FaceClient.NODES) | {
            n["class_type"] for n in klein["graph"].values()}
        for model, want in (("klein-9b", (20, 4.0)), ("klein-9b-distilled", (4, 1.0))):
            s = dict(ig.default_settings(), model=model, scene="a woman", preset="hq_final")
            p = ig.compose(s, self.studio.lib, self.backend("5090"), inv, nodes=nodes)
            self.assertEqual(p.errors, [])
            self.assertTrue(p.values["face_detail"], p.warnings)
            self.assertFalse(any("face pass" in w for w in p.warnings), p.warnings)
            values = dict(p.values, face_prompt="a face", face_denoise=0.4)
            g = ig.face_graph(klein, values, [], "made.png", ig.fix_crops(512, 512, [
                {"x": 200, "y": 200, "size": 64}]), "oval.png", "out")
            k = g["fc1_guider"]["inputs"]
            self.assertEqual(k["cfg"], want[1], model)
            self.assertEqual(g["fc1_sigmas"]["inputs"], {
                "steps": int(want[0] / 0.4), "width": ig.FACE_EDIT, "height": ig.FACE_EDIT})
            self.assertEqual(g["fc1_trim"]["inputs"]["step"], int(want[0] / 0.4) - want[0])
            self.assertEqual(g["fc1_sampler"]["inputs"]["sampler_name"], "euler")
            self.assertEqual(k["negative"], ["12", 0])
            self.assertEqual(g["fc1_4"]["class_type"], "SamplerCustomAdvanced")
            self.assertEqual(g["fc1_4"]["inputs"]["latent_image"], ["fc1_3n", 0])
            self.assertEqual(g["fc1_4"]["inputs"]["sigmas"], ["fc1_trim", 1])
            self.assertEqual(g["f10"]["inputs"]["clip"], ["2", 0])
            # A backend missing the crop scheduler split still makes the
            # initial picture, with a warning instead of a failed face pass.
            missing = ig.compose(s, self.studio.lib, self.backend("5090"), inv,
                                 nodes=nodes - {"SplitSigmas"})
            self.assertEqual(missing.errors, [])
            self.assertFalse(missing.values["face_detail"])
            self.assertTrue(any("SplitSigmas" in w for w in missing.warnings))
        # FLUX names none, and redraws with the picture's as before.
        self.assertNotIn("redraw_sampler", ig.load_workflow("flux_dev_baseline")["defaults"])
        job, client, _ = self.fix_job(model="flux-dev")
        k = client.graphs[0]["fc1_4"]["inputs"]
        self.assertEqual((k["sampler_name"], k["scheduler"], k["steps"]),
                         ("euler", "simple", 20))

    def test_generate_around_head_keeps_source_position_and_retry(self):
        lock = {"x": 40, "y": 40, "size": 64}
        job, client, src = self.fix_job(model="z-image-turbo", around_head=True,
            spots=[], locks=[lock], words="A woman in a blue coat in a garden")
        self.assertEqual(job.status, "complete", job.detail)
        (g,) = client.graphs
        w, h = ig.file_size_of(src)
        region = ig.lock_regions(w, h, [lock])[0]
        self.assertEqual(g["fc1_1"]["inputs"]["crop_region"],
                         {"x": 0, "y": 0, "width": w, "height": h})
        self.assertEqual(g["fc1_4"]["inputs"]["latent_image"], ["head_latent", 0])
        self.assertEqual(g["head_keep0_mask"]["inputs"]["operation"], "subtract")
        self.assertEqual(g["fl0_1"]["inputs"], {"image": ["fi", 0], "crop_region": region})
        self.assertEqual(g["fl0_2"]["inputs"]["x"], region["x"])
        self.assertEqual(g["fl0_2"]["inputs"]["y"], region["y"])
        self.assertFalse(g["fl0_2"]["inputs"]["resize_source"])
        self.assertEqual(g["fs"]["inputs"]["images"], ["fl0_2", 0])
        retry = ig.again(job.record)
        self.assertTrue(retry["fix"]["around_head"])
        self.assertEqual(retry["fix"]["locks"], [lock])
        self.assertEqual(retry["fix"]["image"], src)

    def test_generate_around_head_refuses_missing_head_mark(self):
        job, client, _ = self.fix_job(around_head=True, spots=[], words="A garden")
        self.assertEqual(job.status, "failed")
        self.assertIn("Mark the head", job.detail)
        self.assertEqual(client.graphs, [])

    def test_a_face_fix_blends_through_the_face_and_locks_are_laid_back(self):
        job, client, _ = self.fix_job(target="face", spots=[
            {"x": 40, "y": 40, "size": 64, "box": [30, 30, 20, 20]}],
            locks=[{"x": 10, "y": 10, "size": 64}])
        self.assertEqual(job.status, "complete", job.detail)
        g = client.graphs[0]
        self.assertIn("fc1_h1", g)                 # the face's true shape, not the oval
        self.assertEqual(g["fh2"]["inputs"]["text"], ig.FIX_FACE_MASK)
        self.assertEqual(g["fl0_1"]["inputs"]["image"], ["fi", 0])   # the original
        self.assertEqual(g["fs"]["inputs"]["images"], ["fl0_2", 0])  # laid back last
        self.assertEqual(self.studio.history.list()[0]["fix"]["locks"][0]["size"], 64)

    def test_a_fix_sees_the_photo_round_the_spot_and_takes_its_colours(self):
        FaceClient.NODES = FaceClient.NODES | {ig.TONE_NODE}
        self.addCleanup(setattr, FaceClient, "NODES", FaceClient.NODES - {ig.TONE_NODE})
        job, client, _ = self.fix_job()
        self.assertEqual(job.status, "complete", job.detail)
        g = client.graphs[0]
        t = g["fc1_t"]["inputs"]
        self.assertEqual((g["fc1_t"]["class_type"], t["image"], t["reference"], t["mask"]),
                         (ig.TONE_NODE, ["fc1_5", 0], ["fc1_2", 0], ["fc1_3h", 0]))
        self.assertEqual(t["amount"], ig.FIX_TONE)
        self.assertEqual(g["fc1_6"]["inputs"]["image"], ["fc1_t", 0])
        self.assertIn("fix_oval", g["fo"]["inputs"]["image"])

    def test_found_glasses_are_redrawn_by_their_own_outline(self):
        job, client, src = self.fix_job(target="other", words="glasses", spots=[
            {"x": 40, "y": 40, "size": 64, "box": [30, 30, 20, 20], "word": "glasses"}])
        with open(src, "wb") as f:                # a picture big enough for the box
            f.write(ig.oval_png(256))
        jobs = self.studio.submit(dict(job.settings))
        settle(jobs)
        self.assertEqual(jobs[0].status, "complete", jobs[0].detail)
        g = FaceClient.instances[-1].graphs[-1]
        self.assertEqual(g["fc1_s0"]["inputs"]["text"], "glasses")
        self.assertEqual(g["fc1_s1"]["inputs"]["image"], ["fc1_2", 0])   # in the original
        self.assertEqual(g["fc1_s3"]["inputs"]["source"], ["fc1_a2", 0])  # kept in the box
        self.assertEqual(g["fc1_3n"]["inputs"]["mask"], ["fc1_s3", 0])    # only it redrawn
        self.assertEqual(g["fc1_a3"]["inputs"]["mask"], ["fc1_s3", 0])    # only it blended

    def swap_job(self, spots, client=None, identities=(), **fix):
        """A fix whose spots may carry photos, on a 5090 with FLUX, Qwen and SAM3."""
        self.studio = ig.Studio(root=self.dir, notify=self.notified.append,
                                client_factory=client or SwapClient)
        if identities:
            self.studio.lib.save("identities", list(identities))
        src = os.path.join(self.dir, "made.png")
        with open(src, "wb") as f:
            f.write(ig.oval_png(256))
        hat = os.path.join(self.dir, "red hat.png")
        with open(hat, "wb") as f:
            f.write(PNG)
        s = self.studio.fix_base(dict(ig.default_settings(), model="flux-dev",
                                      scene="On a pier.", backend="5090"))
        s.update(mode="fix", seed=5, fix=dict({"image": src, "target": "other", "spots": [
            dict(sp, photo=hat) if sp.pop("hat", False) else sp for sp in spots]}, **fix))
        jobs = self.studio.submit(s)
        settle(jobs)
        sent = [c for c in (client or SwapClient).instances
                if c.backend["id"] == "5090" and c.graphs]
        return jobs[0], (sent[0].graphs if sent else []), hat

    def test_a_spot_with_a_photo_is_swapped_by_qwen(self):
        job, graphs, hat = self.swap_job([
            {"x": 60, "y": 40, "size": 64, "box": [40, 20, 40, 30], "word": "hat", "hat": True}])
        self.assertEqual(job.status, "complete", job.detail)
        (g,) = graphs                             # one run: the swap alone
        self.assertNotIn("fc1_4", g)              # not redrawn by FLUX
        self.assertEqual(g["sw1_c"]["inputs"]["image"], ["fi", 0])
        pos = g["sw1_pos"]["inputs"]
        self.assertEqual(pos["image1"], ["sw1_in", 0])        # the crop is picture 1
        self.assertEqual(g[pos["image2"][0]]["inputs"]["image"], "studio_red hat.png")
        self.assertIn("wears the hat from picture 2", pos["prompt"])
        self.assertEqual(g["sw1_m2"]["inputs"]["image"], ["sw1_out", 0])  # the new hat's outline
        self.assertEqual(g["sw1_b4"]["inputs"]["source"], ["sw1_out", 0])
        self.assertEqual(g["fs"]["inputs"]["images"], ["sw1_b4", 0])
        rec = self.studio.history.list()[0]
        self.assertEqual(rec["fix"]["spots"][0]["photo"], hat)
        self.assertIn("from a photo", rec["prompt"])

    def test_photo_spots_are_swapped_first_and_the_rest_redrawn_on_that(self):
        job, graphs, _ = self.swap_job([
            {"x": 60, "y": 40, "size": 64, "hat": True},
            {"x": 180, "y": 180, "size": 64}])
        self.assertEqual(job.status, "complete", job.detail)
        swap, redraw = graphs
        self.assertIn("sw1_b4", swap)
        self.assertNotIn("sw2_c", swap)
        self.assertEqual(swap["sw1_b3"]["inputs"]["image"], ["sw1_b2", 0])
        self.assertEqual(swap["sw1_b2"]["inputs"]["image"], ["fo", 0])   # a clicked square: the oval
        self.assertEqual(redraw["fi"]["inputs"]["image"],
                         "ImageStudio/fix_swap_00001_.png [output]")
        self.assertIn("fc1_4", redraw)
        self.assertNotIn("fc2_4", redraw)

    def test_a_photo_spot_needs_qwen_on_the_machine(self):
        job, _, _ = self.swap_job([{"x": 60, "y": 40, "size": 64, "hat": True}],
                                  client=FaceClient)
        self.assertEqual(job.status, "failed")
        self.assertIn("Qwen-Image-Edit", job.detail)
        self.assertIn("qwen_image_edit_2509", job.detail)

    def test_a_freehand_outline_is_the_only_part_changed(self):
        tri = [[40, 30], [100, 30], [70, 90]]
        spot = ig.outline_spot(tri)
        self.assertEqual((spot["x"], spot["y"]), (70, 60))
        self.assertEqual(spot["size"], 75)
        job, graphs, _ = self.swap_job([dict(spot), dict(ig.outline_spot(
            [[150, 150], [220, 150], [220, 220], [150, 220]]), hat=True)])
        self.assertEqual(job.status, "complete", job.detail)
        swap, redraw = graphs
        self.assertEqual(swap["sw1_b2"]["inputs"]["image"], ["sw1_b1", 0])
        self.assertEqual(swap["sw1_b0"]["inputs"]["mask"], ["sw1_m1", 0])
        shape = swap["sw1_m0"]["inputs"]["image"]
        self.assertTrue(shape.startswith("studio_") and "swap" in shape)
        self.assertEqual(redraw["fc1_a0"]["class_type"], "LoadImage")    # the drawn mask
        self.assertEqual(redraw["fc1_3n"]["inputs"]["mask"], ["fc1_a2", 0])
        self.assertEqual(redraw["fc1_a3"]["inputs"]["mask"], ["fc1_a2", 0])

    def test_outline_png_fills_inside_the_outline(self):
        import zlib
        raw = ig.outline_png([(2, 2), (8, 2), (8, 8), (2, 8)], 10, 10)
        self.assertEqual(ig.picture_size(raw), (10, 10))
        data = zlib.decompress(raw[raw.index(b"IDAT") + 4:raw.index(b"IEND") - 8])
        rows = [data[i * 11 + 1:(i + 1) * 11] for i in range(10)]
        self.assertEqual(rows[5], b"\x00\x00" + b"\xff" * 6 + b"\x00\x00")
        self.assertEqual(rows[0], b"\x00" * 10)
        self.assertEqual(sum(r.count(b"\xff") for r in rows), 36)

    def face_swap_job(self, spots, faces, refs=1, colour=True, installed=True):
        """swap_job with Sitter's face (`refs` reference pictures), on a 5090
        whose SAM3 finds `faces` (x, y, w, h) in the fixed picture and one
        face in each reference. FaceFusion counts as installed unless
        `installed` is False, whatever this checkout's .runtime holds."""
        from unittest.mock import patch
        class FaceFindClient(SwapClient):
            def node_types(self):
                return set(SwapClient.NODES) | ({"ColorTransfer"} if colour else set())

            def listen_for_progress(self, pid, on_event, stop=None, timeout=0):
                graph = self.graphs[int(pid[3:]) - 1]
                if "q0_v" not in graph:
                    return super().listen_for_progress(pid, on_event, stop, timeout)
                out = {}
                for i in range(len([k for k in graph if k.endswith("_v")])):
                    found = faces if i == 0 else [(300, 100, 200, 240)]
                    out.update({
                        "q%d_v" % i: {"text": [json.dumps([[
                            {"x": x, "y": y, "width": w, "height": h}
                            for x, y, w, h in found]])]},
                        "q%d_w" % i: {"text": ["256" if i == 0 else "860"]},
                        "q%d_h" % i: {"text": ["256" if i == 0 else "960"]}})
                return {"status": {"completed": True}, "outputs": out}
        paths = []
        for k in range(refs):
            paths.append(os.path.join(self.dir, "me%d.png" % k))
            with open(paths[-1], "wb") as f:
                f.write(PNG)
        with patch('apps.image_studio.facefusion.available', return_value=installed):
            job, graphs, _ = self.swap_job(spots, client=FaceFindClient, face_swap="sitter",
                                           identities=[{"id": "sitter", "name": "Sitter",
                                                        "references": paths}])
        return job, graphs, paths

    def test_a_fix_ends_with_a_face_swap_on_the_biggest_face(self):
        job, graphs, (me,) = self.face_swap_job([{"x": 60, "y": 60, "size": 64}],
                                                [(150, 20, 20, 24), (40, 100, 50, 60)])
        self.assertEqual(job.status, "complete", job.detail)
        redraw, find, swap = graphs               # spots first, then find, then swap
        self.assertIn("fc1_4", redraw)
        self.assertEqual(find["t"]["inputs"]["text"], "face:8")
        self.assertEqual(find["q0_i"]["inputs"]["image"], "ImageStudio/fix_00001_.png [output]")
        self.assertEqual(find["q1_i"]["inputs"]["image"], "studio_me0.png")
        self.assertEqual(swap["fi"]["inputs"]["image"], "ImageStudio/fix_00001_.png [output]")
        self.assertNotIn("sw2_c", swap)           # the biggest face only
        region = swap["sw1_c"]["inputs"]["crop_region"]
        self.assertLessEqual(region["x"], 40)
        self.assertGreaterEqual(region["x"] + region["width"], 90)
        pos = swap["sw1_pos"]["inputs"]
        self.assertIn("has the face of the person in picture 2", pos["prompt"])
        self.assertEqual(swap["sw1_t"]["inputs"]["text"], "face")   # blended by its outline
        ref = swap[pos["image2"][0]]["inputs"]              # the reference, cut to its face
        self.assertEqual(ref["crop_region"], ig.head_square(
            (300, 100, 200, 240), 860, 960, ig.FACE_SWAP_REF_PAD))
        self.assertEqual(swap[ref["image"][0]]["inputs"]["image"], "studio_me0.png")
        self.assertEqual(swap["sw1_tn"]["inputs"]["image_ref"], ["sw1_c", 0])  # its colours
        self.assertEqual(swap["sw1_b4"]["inputs"]["source"], ["sw1_tn", 0])
        rec = self.studio.history.list()[0]
        self.assertEqual(rec["fix"]["face_swap"], "sitter")
        self.assertIn("then a face swap", rec["prompt"])
        self.assertTrue(any("Sitter's face swapped in" in n for n in rec.get("notes", [])),
                        rec.get("notes"))

    def test_a_face_swap_uses_every_reference_picture(self):
        from unittest.mock import patch
        with patch('apps.image_studio.facefusion.swap', return_value=(PNG, {'outside_mask_changed_pixels': 0})) as swap:
            job, graphs, paths = self.face_swap_job([], [(40, 100, 50, 60)], refs=3)
        self.assertEqual(job.status, 'complete', job.detail)
        self.assertEqual(swap.call_args.args[1]['references'], paths)
        self.assertEqual(graphs, [])  # no Qwen redraw or SAM3 dependency
        self.assertEqual(job.record['facefusion'][0]['outside_mask_changed_pixels'], 0)

    def test_a_face_swap_alone_is_a_fix_and_needs_a_face(self):
        from unittest.mock import patch
        with patch('apps.image_studio.facefusion.swap', return_value=(PNG, {'outside_mask_changed_pixels': 0})):
            job, graphs, _ = self.face_swap_job([], [(40, 100, 50, 60)], colour=False)
        self.assertEqual(job.status, 'complete', job.detail)
        self.assertEqual(graphs, [])
        self.assertTrue(any('FaceFusion applied;' in n for n in job.notes))
        with patch('apps.image_studio.facefusion.swap', side_effect=RuntimeError('no face was found')):
            job, graphs, _ = self.face_swap_job([], [])
        self.assertEqual(job.status, 'failed')
        self.assertIn('no face', job.detail)
        self.assertEqual(graphs, [])

    def test_a_face_swap_fails_when_facefusion_is_not_installed(self):
        from unittest.mock import patch
        with patch('apps.image_studio.facefusion.swap') as swap:
            job, graphs, _ = self.face_swap_job([], [(40, 100, 50, 60)], installed=False)
        self.assertEqual(job.status, 'failed')
        self.assertIn('FaceFusion is not installed', job.detail)
        swap.assert_not_called()
        self.assertEqual(graphs, [])

    def test_a_face_swap_needs_an_identity_with_a_reference(self):
        job, graphs, _ = self.swap_job([], face_swap="partner", identities=[
            {"id": "partner", "name": "Partner"}])
        self.assertEqual(job.status, "failed")
        self.assertIn("Partner has no reference picture", job.detail)
        self.assertEqual(graphs, [])
        job, _, _ = self.swap_job([], face_swap="nobody")
        self.assertEqual(job.status, "failed")
        self.assertIn("nobody", job.detail)

    def test_swap_prompts(self):
        fix = {"target": "hand", "spots": [{"x": 1, "y": 1, "size": 64, "photo": "a.png"}]}
        self.assertEqual(ig.clean_fix(fix)["spots"][0]["photo"], "a.png")
        self.assertIn("like the hand in picture 2", ig.swap_prompt(fix, {}))
        self.assertIn("the cowboy hat from picture 2", ig.swap_prompt(
            dict(fix, target="other", words="cowboy hat"), {}))
        self.assertEqual(ig.fix_words(fix), "1 hand (all from a photo)")

    def test_without_the_tone_node_a_fix_says_so(self):
        job, client, _ = self.fix_job()
        self.assertEqual(job.status, "complete", job.detail)
        self.assertNotIn("fc1_t", client.graphs[0])
        self.assertTrue(any(ig.TONE_NODE in n for n in job.notes))

    def test_a_fix_without_spots_fails_in_words(self):
        job, _, _ = self.fix_job(spots=[])
        self.assertEqual(job.status, "failed")
        self.assertIn("Click the part", job.detail)

    def face_studio(self):
        self.studio = ig.Studio(root=self.dir, notify=self.notified.append,
                                client_factory=FaceClient)
        FaceClient.fail_pass = False

    def test_the_face_pass_redraws_each_face_with_the_identity_lora(self):
        self.face_studio()
        jobs = self.studio.submit(dict(ig.default_settings(), model="flux-dev", hand_pass=False,
                                       preset="identity", identities=["sitter"],
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
        self.assertIn("SITTERPERSON", second["f10"]["inputs"]["text"])
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
            {"id": "p1", "name": "Partner", "at": [0.43, 0.35], "words": "A woman in her 30s.",
             "face": face, "from": "", "region": [0.35, 0.25, 0.5, 0.45]},
            {"id": "p2", "name": "Sitter", "at": [0.9, 0.9], "words": "A man.", "face": "",
             "from": ""}]}

    def test_a_head_shape_goes_into_its_face_redraw_as_far_as_the_turn_agrees(self):
        import apps.image_studio.scene.scene as sc
        s = sc.new_scene("")
        p = sc.new_object("person")
        p["head"] = {"jaw_width": 0.9}
        s["objects"].append(p)
        s["frame"] = "square"
        s["camera"].update(target=[0.0, 1.3, 0.0], distance=2.6, lens=50.0)
        faces = self.scene_faces("")
        faces["head_depth"] = 0.4
        faces["people"][0].update(id=p["id"], head=p["head"], facing=0.5)

        def run(drawn_nose, pose_node=True):
            class HeadClient(FaceClient):
                def node_types(self):
                    return set(FaceClient.NODES) | ({ig.POSE_NODE} if pose_node else set())

                def listen_for_progress(self, pid, on_event, stop=None, timeout=0):
                    graph = self.graphs[int(pid[3:]) - 1]
                    if graph.get("2", {}).get("class_type") == ig.POSE_NODE:
                        if not pose_node:
                            raise ig.ComfyError("no such node")
                        pts = [[0, 0, 0.0]] * 133
                        pts[23], pts[39] = [410, 350, 0.9], [480, 350, 0.9]
                        pts[53] = [410 + 70 * drawn_nose, 360, 0.9]
                        return {"outputs": {"2": {"text": [json.dumps(
                            {"people": [{"box": [380, 280, 520, 700], "points": pts}]})]}}}
                    return super().listen_for_progress(pid, on_event, stop, timeout)
            self.studio = ig.Studio(root=self.dir, notify=self.notified.append,
                                    client_factory=HeadClient)
            jobs = self.studio.submit(dict(ig.default_settings(), model="flux-dev", scene="x",
                                           backend="5090", seed=5, face_detail=True, hand_pass=False,
                                           scene_layout=s, scene_faces=faces))
            settle(jobs)
            self.assertEqual(jobs[0].status, "complete", jobs[0].detail)
            return HeadClient.instances[-1].graphs[-1], self.studio.history.list()[0]

        second, rec = run(0.52)                            # drawn as the mannequin turns
        self.assertEqual(second["fc1_dc"]["inputs"]["strength"], 0.4)
        self.assertTrue(second["fc1_d"]["inputs"]["image"].startswith("studio_head_"))
        self.assertEqual(rec["face_detail"]["head_depth"], {"Partner": 0.4})
        second, rec = run(0.85)                            # about 29 degrees off: half
        self.assertEqual(second["fc1_dc"]["inputs"]["strength"], 0.2)
        self.assertTrue(any("half strength" in n for n in rec["notes"]), rec["notes"])
        second, rec = run(1.0)                             # turned the other way: none
        self.assertNotIn("fc1_dc", second)
        self.assertTrue(any("was not used" in n for n in rec["notes"]), rec["notes"])
        second, rec = run(0.5, pose_node=False)            # the turn unread: half
        self.assertEqual(second["fc1_dc"]["inputs"]["strength"], 0.2)
        faces["head_depth"] = 0.0                          # turned off in the scene
        second, rec = run(0.5)
        self.assertNotIn("fc1_dc", second)

    def test_a_scene_face_is_redrawn_as_its_person_and_to_their_face_picture(self):
        self.studio = ig.Studio(root=self.dir, notify=self.notified.append,
                                client_factory=PulidClient)
        FaceClient.fail_pass = False
        face = os.path.join(self.dir, "partner.png")
        with open(face, "wb") as f:
            f.write(PNG)
        jobs = self.studio.submit(dict(ig.default_settings(), model="flux-dev", scene="x",
                                       backend="5090", seed=5, face_detail=True, hand_pass=False,
                                       scene_faces=self.scene_faces(face)))
        settle(jobs)
        self.assertEqual(jobs[0].status, "complete", jobs[0].detail)
        client = PulidClient.instances[-1]
        first, second = client.graphs
        # Her face goes into the picture itself, over her head only.
        self.assertEqual(first["40"]["inputs"]["model"], ["pb_1", 0])
        self.assertEqual(first["pb_1"]["inputs"]["model"], ["1", 0])
        self.assertEqual(first["pb_1f"]["inputs"]["image"], "studio_partner.png")
        self.assertEqual(first["pb_1k"]["inputs"]["image"], ["pb_1m", 0])
        mask = [u for u in client.uploads if "face_regions" in u]
        self.assertEqual(len(mask), 1)
        self.assertIn(face, client.uploads)
        # The redraw is of the picture just made, not her face picture.
        self.assertEqual(second["fi"]["inputs"]["image"], "ImageStudio_00001_.png [output]")
        k = second["fc1_4"]["inputs"]
        self.assertEqual(k["model"], ["fc1_p", 0])
        self.assertEqual(k["denoise"], 0.85)          # the scene's likeness
        self.assertEqual(second["fc1_p"]["inputs"]["image"], ["fc1_r", 0])
        self.assertEqual(second["fc1_r"]["inputs"]["image"], "studio_partner.png")
        self.assertEqual(second["pl1"]["inputs"]["pulid_file"], "pulid_flux_v0.9.1.safetensors")
        # Her own words, on a copy of the face prompt's conditioning.
        pos = second[k["positive"][0]]
        text = second[pos["inputs"]["conditioning"][0]]["inputs"]["text"]
        self.assertIn("A woman in her 30s.", text)
        self.assertEqual(second["f10"]["inputs"]["text"].count("A woman in her 30s"), 0)
        rec = self.studio.history.list()[0]
        self.assertEqual(rec["face_detail"]["likeness"], ["Partner"])
        self.assertTrue(any("Sitter's face was not found" in n for n in rec["notes"]))
        self.assertTrue(any("Likeness: Partner" in n for n in rec["notes"]), rec["notes"])

    def test_a_face_picture_that_will_not_send_leaves_the_faces_to_their_words(self):
        # Two people: Partner's photo goes up for the face pass, Sitter's does not.
        # What is left must be the redraw the warning promises - from the words,
        # with no PuLID step pointing at loaders the graph does not have.
        class TwoFaces(PulidClient):
            def upload_image(self, path):
                if self.graphs and path.endswith("sitter.png"):
                    raise ig.ComfyError("upload refused")
                return super().upload_image(path)

            def listen_for_progress(self, pid, on_event, stop=None, timeout=0):
                entry = super().listen_for_progress(pid, on_event, stop, timeout)
                if "fd4" in entry["outputs"]:
                    entry["outputs"]["fd4"] = {"text": [json.dumps([[
                        {"x": 400, "y": 300, "width": 90, "height": 110},
                        {"x": 700, "y": 300, "width": 90, "height": 110}]])]}
                return entry
        self.studio = ig.Studio(root=self.dir, notify=self.notified.append,
                                client_factory=TwoFaces)
        FaceClient.fail_pass = False
        photos = []
        for name in ("partner", "sitter"):
            photos.append(os.path.join(self.dir, name + ".png"))
            with open(photos[-1], "wb") as f:
                f.write(PNG)
        faces = self.scene_faces(photos[0])
        faces["people"][1].update(face=photos[1], at=[0.73, 0.35],
                                  region=[0.65, 0.25, 0.8, 0.45])
        jobs = self.studio.submit(dict(ig.default_settings(), model="flux-dev", scene="x",
                                       backend="5090", seed=5, face_detail=True, hand_pass=False,
                                       scene_faces=faces))
        settle(jobs)
        self.assertEqual(jobs[0].status, "complete", jobs[0].detail)
        rec = self.studio.history.list()[0]
        self.assertTrue(any("could not be sent" in w for w in rec["warnings"]), rec["warnings"])
        second = TwoFaces.instances[-1].graphs[-1]
        self.assertIn("fs", second)                       # the face pass ran
        self.assertFalse([k for k, n in second.items() if n["class_type"] == "ApplyPulidFlux"])
        for nid, node in second.items():
            for value in node["inputs"].values():
                if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) \
                        and isinstance(value[1], int):
                    self.assertIn(value[0], second, "%s links to a missing node" % nid)
        self.assertEqual(rec["face_detail"]["likeness"], [])

    def test_a_face_facefusion_will_swap_is_not_also_drawn_with_pulid(self):
        from unittest.mock import patch
        self.studio = ig.Studio(root=self.dir, notify=self.notified.append,
                                client_factory=PulidClient)
        FaceClient.fail_pass = False
        face = os.path.join(self.dir, "partner.png")
        with open(face, "wb") as f:
            f.write(PNG)
        self.studio.lib.save("identities", [
            {"id": "partner", "name": "Partner", "trigger": "PARTNERPERSON", "strength": 0.8,
             "references": [face]}])
        scene_faces = dict(self.scene_faces(face))
        scene_faces["people"] = [dict(scene_faces["people"][0], identity="partner"),
                                 scene_faces["people"][1]]
        with patch("apps.image_studio.facefusion.available", return_value=True), \
                patch("apps.image_studio.facefusion.swap",
                     return_value=(PNG, {"outside_mask_changed_pixels": 0})):
            jobs = self.studio.submit(dict(ig.default_settings(), model="flux-dev", scene="x",
                                           backend="5090", seed=5, face_detail=True,
                                           hand_pass=False, scene_faces=scene_faces))
            settle(jobs)
        self.assertEqual(jobs[0].status, "complete", jobs[0].detail)
        client = PulidClient.instances[-1]
        second = client.graphs[1]           # the face pass, before FaceFusion's own swap
        # Partner is about to be swapped by FaceFusion, so the face pass leaves
        # her crop to the words - no PuLID photo redraw wasted on a face
        # that gets fully overwritten a moment later.
        self.assertNotIn("fc1_r", second)
        self.assertNotIn("pl1", second)
        rec = self.studio.history.list()[0]
        self.assertEqual(rec["face_detail"]["likeness"], [])
        self.assertFalse(any("Likeness: Partner" in n for n in rec["notes"]), rec["notes"])
        self.assertTrue(any("FaceFusion applied" in n for n in rec["notes"]), rec["notes"])
        self.assertTrue(any("Partner not drawn with PuLID" in n for n in rec["notes"]), rec["notes"])

    def test_a_face_the_head_swap_redraws_is_left_out_of_the_face_pass(self):
        """The head swap redraws the whole head after the face pass, so the
        face pass's redraw of that face would be thrown away: it is not
        made. With the head swap off, it is."""
        from unittest.mock import patch
        face = os.path.join(self.dir, "partner.png")
        with open(face, "wb") as f:
            f.write(PNG)

        def run(head_swap):
            self.studio = ig.Studio(root=self.dir, notify=self.notified.append,
                                    client_factory=FaceClient)
            FaceClient.fail_pass = False
            self.studio.lib.save("identities", [
                {"id": "partner", "name": "Partner", "references": [face]}])
            scene_faces = dict(self.scene_faces(""))
            scene_faces["people"] = [dict(scene_faces["people"][0], identity="partner"),
                                     scene_faces["people"][1]]
            with patch("apps.image_studio.facefusion.available", return_value=True), \
                    patch("apps.image_studio.facefusion.swap",
                          return_value=(PNG, {"outside_mask_changed_pixels": 0})), \
                    patch("apps.image_studio.headswap.lacks", return_value=[]), \
                    patch.object(ig.Studio, "_head_swap",
                                 lambda self, job, client, values, pictures, *a: pictures):
                jobs = self.studio.submit(dict(ig.default_settings(), model="flux-dev",
                                               scene="x", backend="5090", seed=5,
                                               face_detail=True, hand_pass=False,
                                               head_swap=head_swap, scene_faces=scene_faces))
                settle(jobs)
            self.assertEqual(jobs[0].status, "complete", jobs[0].detail)
            graphs = FaceClient.instances[-1].graphs
            return [g for g in graphs if "fs" in g], self.studio.history.list()[0]

        passes, rec = run(True)
        self.assertEqual(passes, [])                    # no face pass run at all
        self.assertTrue(any("1 face left to the head swap" in n for n in rec["notes"]),
                        rec["notes"])
        self.assertTrue(any("FaceFusion applied" in n for n in rec["notes"]), rec["notes"])
        passes, rec = run(False)
        self.assertEqual(len(passes), 1)
        self.assertFalse(any("left to the head swap" in n for n in rec["notes"]), rec["notes"])

    def test_a_characters_face_photos_draw_the_forms_person(self):
        """The plain form: the character's photos are its one person's face -
        in the picture itself over the whole frame, on the biggest face in
        the face pass, and to the real-face paste - with the pass forced on."""
        photos = []
        for name in ("partner.png", "partner_left.png"):
            photos.append(os.path.join(self.dir, name))
            with open(photos[-1], "wb") as f:
                f.write(PNG)
        self.studio = ig.Studio(root=self.dir, notify=self.notified.append,
                                client_factory=PasteClient)
        FaceClient.fail_pass = False
        PasteClient.report = [{"name": "Partner", "pasted": True, "reference": "studio_partner.png",
                               "difference": 2, "tolerance": 19}]
        jobs = self.studio.submit(dict(ig.default_settings(), model="flux-dev", scene="x",
                                       backend="5090", seed=5, face_photos=photos,
                                       face_name="Partner"))
        settle(jobs)
        self.assertEqual(jobs[0].status, "complete", jobs[0].detail)
        graphs = PasteClient.instances[-1].graphs
        first, second, third = graphs[:3]
        # After them, main's face finding on each reference photo - reads only.
        for g in graphs[3:]:
            self.assertIn("SAM3_Detect", {n.get("class_type") for n in g.values()})
        # Both of Partner's photos chain into PuLID: the first at full weight,
        # the second (a different angle) supporting it at PULID_EXTRA_WEIGHT.
        self.assertEqual(first["40"]["inputs"]["model"], ["pb_1_2", 0])
        self.assertEqual(first["pb_1f"]["inputs"]["image"], "studio_partner.png")
        self.assertEqual(first["pb_1"]["inputs"]["weight"], ig.PULID_BASE_WEIGHT)
        self.assertEqual(first["pb_1_2f"]["inputs"]["image"], "studio_partner_left.png")
        self.assertEqual(first["pb_1_2"]["inputs"]["weight"], ig.PULID_EXTRA_WEIGHT)
        self.assertEqual(first["pb_1_2"]["inputs"]["model"], ["pb_1", 0])
        self.assertNotIn("attn_mask", first["pb_1"]["inputs"])   # the whole frame: no mask
        self.assertNotIn("pb_1m", first)
        self.assertEqual(second["fc1_r"]["inputs"]["image"], "studio_partner.png")
        self.assertEqual(second["fc1_p"]["inputs"]["weight"], ig.PULID_WEIGHT)
        self.assertEqual(second["fc1_r2"]["inputs"]["image"], "studio_partner_left.png")
        self.assertEqual(second["fc1_p2"]["inputs"]["weight"], ig.PULID_EXTRA_WEIGHT)
        self.assertEqual(second["fc1_4"]["inputs"]["denoise"], ig.FORM_LIKENESS)
        self.assertEqual(json.loads(third["pp"]["inputs"]["faces"])[0]["references"],
                         ["studio_partner.png", "studio_partner_left.png"])
        rec = self.studio.history.list()[0]
        self.assertTrue(any("Face: Partner, from 2 photos" in n for n in rec["notes"]),
                        rec["notes"])
        self.assertTrue(any("Partner drawn from more than one of their photos" in n
                            for n in rec["notes"]), rec["notes"])
        self.assertEqual(rec["settings"]["face_photos"], photos)     # Generate Again has them

    def test_a_scene_face_wins_over_the_forms(self):
        self.assertEqual(ig.faces_of({"scene_faces": {"people": []}, "face_photos": [__file__]}),
                         {"people": []})
        self.assertEqual(ig.faces_of({"face_photos": ["/no/such.png"]}), {})
        who = ig.faces_of({"face_photos": [__file__]})["people"][0]
        self.assertEqual((who["at"], who["face"], who["name"]), (None, __file__, "the person"))

    def test_the_forms_person_is_the_biggest_face_left(self):
        boxes = [(10, 10, 20, 20), (500, 500, 80, 90), (100, 100, 60, 60)]
        scene = {"at": [0.02, 0.02], "name": "A"}
        form = {"at": None, "name": "B"}
        self.assertEqual(ig.match_faces(1000, 1000, boxes, [scene, form]),
                         {0: scene, 1: form})
        self.assertEqual(ig.match_faces(1000, 1000, [], [form]), {})

    def test_without_pulid_a_scene_face_is_redrawn_from_its_words_and_says_so(self):
        self.face_studio()
        face = os.path.join(self.dir, "partner.png")
        with open(face, "wb") as f:
            f.write(PNG)
        jobs = self.studio.submit(dict(ig.default_settings(), model="flux-dev", scene="x",
                                       backend="5090", seed=5, face_detail=True, hand_pass=False,
                                       scene_faces=self.scene_faces(face)))
        settle(jobs)
        first, second = FaceClient.instances[-1].graphs
        self.assertFalse([n for n in first if n.startswith("pb")])
        self.assertNotIn("fc1_p", second)
        self.assertEqual(second["fc1_4"]["inputs"]["denoise"], 0.4)
        rec = self.studio.history.list()[0]
        self.assertTrue(any("lacks PuLID" in w for w in rec["warnings"]), rec["warnings"])

    def real_scene(self, client, report, real=True):
        """A scene job whose face Partner has two photos, on `client`."""
        self.studio = ig.Studio(root=self.dir, notify=self.notified.append,
                                client_factory=client)
        FaceClient.fail_pass = False
        PasteClient.report = report
        photos = []
        for name in ("partner.png", "partner_left.png"):
            photos.append(os.path.join(self.dir, name))
            with open(photos[-1], "wb") as f:
                f.write(PNG)
        faces = self.scene_faces(photos[0])
        faces["real"] = real
        faces["people"][0]["photos"] = photos
        jobs = self.studio.submit(dict(ig.default_settings(), model="flux-dev", scene="x",
                                       backend="5090", seed=5, face_detail=True, hand_pass=False,
                                       scene_faces=faces))
        settle(jobs)
        self.assertEqual(jobs[0].status, "complete", jobs[0].detail)
        return client.instances[-1], self.studio.history.list()[0]

    def test_a_real_face_is_pasted_last_and_the_pulid_picture_kept_beside_it(self):
        client, rec = self.real_scene(PasteClient, [
            {"name": "Partner", "pasted": True, "reference": "studio_partner_left.png",
             "difference": 3.5, "tolerance": 19.2, "confidence": 0.82}])
        first, second, third = client.graphs
        self.assertEqual(third["pi"]["inputs"]["image"], "ImageStudio/faces_00001_.png [output]")
        faces = json.loads(third["pp"]["inputs"]["faces"])
        self.assertEqual(faces, [{"name": "Partner", "box": [400, 300, 90, 110],
                                  "references": ["studio_partner.png", "studio_partner_left.png"]}])
        self.assertEqual(third["pp"]["inputs"]["seed"], 5)
        self.assertTrue(third["ps"]["inputs"]["filename_prefix"].endswith("_real"))
        self.assertNotIn("KSampler", {n["class_type"] for n in third.values()})   # no redraw
        self.assertEqual(len(rec["images"]), 2)          # the pasted one first, PuLID's kept
        self.assertEqual(rec["paste_graph"], third)
        self.assertEqual(rec["face_detail"]["real"][0]["reference"],
                         os.path.join(self.dir, "partner_left.png"))
        self.assertTrue(any("Real face: Partner from partner_left.png" in n for n in rec["notes"]),
                        rec["notes"])

    def test_a_face_no_photo_fits_is_left_as_pulid_drew_it(self):
        client, rec = self.real_scene(PasteClient, [
            {"name": "Partner", "pasted": False, "why": "no photo at this angle (30 degrees off, "
                                                      "19 allowed)"}])
        self.assertEqual(len(client.graphs), 3)
        self.assertEqual(len(rec["images"]), 1)          # nothing pasted: nothing twice
        self.assertIsNone(rec["paste_graph"])
        self.assertTrue(any("Partner kept as PuLID drew it - no photo at this angle" in n
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
        jobs = self.studio.submit(dict(ig.default_settings(), model="flux-dev", hand_pass=False,
                                       preset="identity", identities=["sitter"],
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

    def test_add_after_close_finishes_the_job_instead_of_stranding_it(self):
        """add() and close() (app shutdown) can race: a job that lands in
        JobQueue after close() has already run its cancel pass used to sit
        in lane.waiting forever, its lane thread returning at once without
        ever touching it, never notified as finished."""
        self.studio.queue.close()
        job = ig.Job(dict(ig.default_settings(), scene="x"), self.backend("5090"))
        self.studio.queue.add(job)
        deadline = time.monotonic() + 2
        while job.status == "queued" and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(job.status, "cancelled")

    def test_a_plan_error_fails_the_job_in_words(self):
        jobs = self.studio.submit(dict(ig.default_settings(), scene="", backend="5090"))
        settle(jobs)
        self.assertEqual(jobs[0].status, "failed")
        self.assertIn("Describe the scene", jobs[0].detail)

    def test_a_job_that_crashes_logs_its_traceback(self):
        """The lane catches everything so it never dies with a job half done;
        the error log is then the only record, so it gets the whole trace -
        a bare repr named no line when a job failed on a TypeError."""
        from unittest import mock
        def crash(job, notify):
            raise TypeError("list indices must be integers or slices, not str")
        with mock.patch.object(ig.doctor, "log_error") as log, \
                mock.patch.object(self.studio, "run_job", crash):
            jobs = self.studio.submit(dict(ig.default_settings(), scene="x", backend="5090"))
            settle(jobs)
        self.assertEqual(jobs[0].status, "failed")
        self.assertIn("TypeError: list indices", jobs[0].detail)
        (text,), _ = log.call_args
        self.assertIn("Traceback (most recent call last)", text)
        self.assertIn("in crash", text)


class TestFixSpots(unittest.TestCase):
    def test_head_off_centre_in_landscape_frame_is_not_repositioned(self):
        wf = ig.load_workflow("flux_dev_baseline")
        values = {"model": "m", "clip_l": "c", "t5": "t", "vae": "v",
                  "prompt": "p", "seed": 1, "face_prompt": "A woman in a garden"}
        region = {"x": 1160, "y": 220, "width": 140, "height": 140}
        g = ig.around_head_graph(wf, values, [], "source.png", 1536, 1024,
                                 [region], "oval.png", "test")
        self.assertEqual(g["head_keep0_mask"]["inputs"]["x"], 942)
        self.assertEqual(g["head_keep0_mask"]["inputs"]["y"], 178)
        self.assertEqual(g["head_keep0"]["inputs"]["width"], 115)
        self.assertEqual(g["head_keep0"]["inputs"]["height"], 115)
        self.assertEqual(g["fl0_1"]["inputs"]["crop_region"], region)
        self.assertEqual(g["fl0_2"]["inputs"]["x"], 1160)
        self.assertEqual(g["fl0_2"]["inputs"]["y"], 220)
        self.assertEqual(g["fc1_6"]["inputs"]["width"], 1536)
        self.assertEqual(g["fc1_6"]["inputs"]["height"], 1024)

    def test_squares_stay_inside_the_picture(self):
        (c,) = ig.fix_crops(1000, 800, [{"x": 10, "y": 790, "size": 200}])
        self.assertEqual((c["x"], c["y"], c["width"]), (0, 600, 200))
        (c,) = ig.fix_crops(100, 80, [{"x": 50, "y": 40, "size": 500}])
        self.assertEqual((c["x"], c["y"], c["width"], c["height"]), (10, 0, 80, 80))

    def test_the_tone_node_moves_a_redraw_onto_the_originals_curves(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is ComfyUI's, not the app's")
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "comfy_nodes"))
        self.addCleanup(sys.path.pop, 0)
        import studio_matchtone as mt
        rng = np.random.default_rng(1)
        ref = rng.uniform(0.2, 0.6, (64, 64, 3)).astype(np.float32)
        drawn = np.clip(ref * 1.3 + 0.1, 0, 1)            # brighter, more contrast
        mask = np.ones((64, 64), np.float32)
        out = mt.match(drawn, ref, mask, 1.0)
        self.assertLess(abs(out.mean() - ref.mean()), 0.01)
        self.assertLess(abs(out.std() - ref.std()), 0.01)
        half = mt.match(drawn, ref, mask, 0.5)
        self.assertAlmostEqual(float(half.mean()), (drawn.mean() + out.mean()) / 2, places=2)
        self.assertIs(mt.match(drawn, ref, mask * 0, 1.0), drawn)   # nothing to fit on

    def test_a_found_accessory_is_redrawn_in_its_box_from_its_own_words(self):
        # 2026-09-26: glasses redrawn from the whole picture's prompt over the
        # oval of a head-sized crop came back as a tiny dancer.
        self.assertNotIn("dancing", ig.fix_prompt({"target": "other", "words": "glasses"},
                                                  "a woman dancing"))
        spot = {"x": 589, "y": 250, "size": 207, "box": [541, 198, 96, 103]}
        (crop,) = ig.fix_areas(ig.fix_crops(1024, 1024, [dict(spot, size=310)]), [spot])
        self.assertEqual(crop["area"], (73, 66, 236, 242))
        (face,) = ig.fix_areas(ig.fix_crops(1024, 1024, [spot], head=True), [spot])
        self.assertNotIn("area", face)                # a face keeps SAM3's own mask

    def test_what_find_finds_becomes_squares(self):
        g = ig.parts_graph("p.png", "sam3.pt", ig.FIX_FIND["other"])
        self.assertEqual(sum(n["class_type"] == "SAM3_Detect" for n in g.values()),
                         len(ig.FIX_FIND["other"]))
        said = {"outputs": {"7": {"text": ["1000"]}, "8": {"text": ["800"]},
                            "p0v": {"text": [json.dumps([[{"x": 100, "y": 100, "width": 40,
                                                           "height": 60}]])]},
                            "p1v": {"text": [json.dumps([{"x": 5, "y": 5, "width": 4,
                                                          "height": 4}])]}}}
        self.assertEqual(ig.parts_found(said, 2, ["glasses", "hat"])[2],
                         [(100, 100, 40, 60, "glasses")])
        self.assertEqual(ig.found_spots(1000, 800, [(100, 100, 40, 60, "glasses")],
                                        "other")[0]["word"], "glasses")
        width, height, boxes = ig.parts_found(said, 2)
        self.assertEqual((width, height, boxes), (1000, 800, [(100, 100, 40, 60)]))
        (sp,) = ig.found_spots(width, height, boxes, "hand")
        self.assertEqual((sp["x"], sp["y"], sp["size"], sp["box"]), (120, 130, 90,
                                                                      [100, 100, 40, 60]))
        self.assertEqual(ig.lock_regions(1000, 800, [{"x": 0, "y": 0, "size": 100}]),
                         [{"x": 0, "y": 0, "width": 100, "height": 100}])

    def test_a_fix_is_made_safe(self):
        f = ig.clean_fix({"target": "elbow", "strength": 9, "spots": [
            {"x": "5", "y": 6, "size": 100}, {"x": 1, "y": 1, "size": 10}, {"x": None}]})
        self.assertEqual(f["target"], "hand")
        self.assertEqual(f["strength"], 1.0)
        self.assertEqual(f["spots"], [{"x": 5, "y": 6, "size": 100}])
        self.assertEqual(ig.clean_fix({"strength": "light"})["strength"], 0.45)
        self.assertEqual(ig.fix_words({"spots": [{"x": 1, "y": 1, "size": 99}] * 2}), "2 hands")


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

    def test_a_body_that_is_not_json_is_a_comfyerror_not_a_crash(self):
        """A proxy's HTML error page, or ComfyUI cut off mid-restart, answers
        200 with a body that will not parse - the same kind of failure as an
        HTTP error, not a raw ValueError callers do not expect."""
        c = ig.ComfyUIClient({"id": "x", "name": "X", "url": "http://127.0.0.1:9"})
        with self.assertRaises(ig.ComfyError) as caught:
            c._parse("<html>502 Bad Gateway</html>")
        self.assertIn("X", str(caught.exception))
        self.assertIn("not JSON", str(caught.exception))

    def test_polling_survives_a_transient_error_and_an_explicit_null_status(self):
        """listen_for_progress used to only tolerate Unreachable (a dropped
        connection); a malformed body wrapped as ComfyError, or a history
        entry whose status is JSON null rather than missing, used to crash
        the polling thread instead of being read as busy-not-gone."""
        from unittest import mock
        c = ig.ComfyUIClient({"id": "x", "name": "X", "url": "http://127.0.0.1:9"})
        calls = []
        def get_history(prompt_id):
            calls.append(prompt_id)
            if len(calls) == 1:
                raise ig.ComfyError("X answered with something that is not JSON")
            return {"status": None, "outputs": {"9": {}}}
        c.get_history = get_history
        c.position = lambda pid: None
        events = []
        watch = ig.Watch(None, ig.queue.Queue(), "")
        with mock.patch.object(ig, "POLL_EVERY", 0.01):    # the second look, not 2 s on
            entry = c.listen_for_progress("pid", lambda k, d: events.append((k, d)),
                                          timeout=10, watch=watch)
        self.assertEqual(entry, {"status": None, "outputs": {"9": {}}})
        self.assertIn("busy", [k for k, _ in events])
        self.assertEqual(len(calls), 2)

    def _waiting_client(self, where):
        """A client whose prompt is never in /history; `where` is a list of
        what position() answers, one per look (the last one repeats)."""
        c = ig.ComfyUIClient({"id": "x", "name": "X", "url": "http://127.0.0.1:9"})
        c.get_history = lambda pid: None
        looks = []
        def position(pid):
            looks.append(pid)
            return where[min(len(looks), len(where)) - 1]
        c.position = position
        c.cancelled = []
        c.cancel_job = lambda pid: c.cancelled.append(pid) or True
        return c, looks

    def test_a_prompt_comfyui_lost_fails_in_seconds_not_at_the_deadline(self):
        """ComfyUI crashed or restarted mid-job: the prompt is in neither its
        history nor its queue, and the wait used to run on to JOB_TIMEOUT
        (30 minutes) saying only "quiet"."""
        from unittest import mock
        c, looks = self._waiting_client([-1, -1, None])
        watch = ig.Watch(None, ig.queue.Queue(), "")
        with mock.patch.object(ig, "POLL_EVERY", 0.01):
            with self.assertRaises(ig.ComfyError) as caught:
                c.listen_for_progress("pid", lambda k, d: None, timeout=30, watch=watch)
        self.assertIn("no longer has prompt pid", str(caught.exception))
        self.assertEqual(len(looks), 2 + ig.LOST_AFTER)
        self.assertEqual(c.cancelled, [])

    def test_a_prompt_seen_nowhere_fewer_times_than_lost_after_is_still_waited_on(self):
        """A prompt that ends between the /history and the /queue read is in
        neither once; a running one read again resets the count."""
        from unittest import mock
        c, looks = self._waiting_client([None] * (ig.LOST_AFTER - 1) + [-1] +
                                        [None] * (ig.LOST_AFTER - 1))
        entry = {"status": {"completed": True}, "outputs": {"9": {}}}
        c.get_history = lambda pid: entry if len(looks) == 2 * ig.LOST_AFTER - 1 else None
        watch = ig.Watch(None, ig.queue.Queue(), "")
        with mock.patch.object(ig, "POLL_EVERY", 0.01):
            got = c.listen_for_progress("pid", lambda k, d: None, timeout=30, watch=watch)
        self.assertEqual(got, entry)

    def test_a_wait_past_its_deadline_stops_the_prompt(self):
        """The deadline used to leave the prompt running on the GPU, so the
        lane's next job queued behind it and free() would not let go."""
        from unittest import mock
        c, _ = self._waiting_client([-1])
        watch = ig.Watch(None, ig.queue.Queue(), "")
        with mock.patch.object(ig, "POLL_EVERY", 0.01):
            with self.assertRaises(ig.ComfyError) as caught:
                c.listen_for_progress("pid", lambda k, d: None, timeout=0.05, watch=watch)
        self.assertEqual(c.cancelled, ["pid"])
        self.assertIn("it was stopped there", str(caught.exception))

    def test_a_quick_run_rides_out_a_busy_poll_and_names_the_failing_node(self):
        """SAM3's finders: one /history read that timed out while ComfyUI
        staged the model ended the run; and the error named no node."""
        from unittest import mock
        polls = []
        def get_history(pid):
            polls.append(pid)
            if len(polls) == 1:
                raise ig.Unreachable("Cannot reach X at http://10.0.0.9:8188. (timed out)")
            return {"status": {"status_str": "error", "messages": [["execution_error", {
                "node_id": "fd4", "exception_type": "RuntimeError",
                "exception_message": "no sam3 weights"}]]}}
        client = mock.Mock(backend={"name": "X"}, get_history=get_history)
        client.queue_workflow.return_value = "pid"
        graph = {"fd4": {"class_type": "SAM3Detect", "inputs": {}}}
        with mock.patch.object(ig.time, "sleep", lambda s: None):
            with self.assertRaises(ig.ComfyError) as caught:
                ig.Studio._run_quick(None, client, graph)
        self.assertEqual(len(polls), 2)
        self.assertIn("node fd4 (SAM3Detect): RuntimeError: no sam3 weights",
                      str(caught.exception))

    def test_a_quick_run_fails_at_once_when_refused_and_stops_at_its_deadline(self):
        from unittest import mock
        client = mock.Mock(backend={"name": "X"})
        client.get_history.side_effect = ig.Unreachable("Nothing is answering (refused)")
        with self.assertRaises(ig.Unreachable):
            ig.Studio._run_quick(None, client, {})
        self.assertEqual(client.get_history.call_count, 1)
        client = mock.Mock(backend={"name": "X"})
        client.queue_workflow.return_value = "pid"
        client.cancel_job.return_value = True
        with self.assertRaises(ig.ComfyError) as caught:
            ig.Studio._run_quick(None, client, {}, timeout=0)
        client.cancel_job.assert_called_once_with("pid")
        self.assertIn("stopped there", str(caught.exception))

    def test_an_upload_answered_with_something_not_json_is_a_comfyerror(self):
        import io
        from unittest import mock
        c = ig.ComfyUIClient({"id": "x", "name": "X", "url": "http://127.0.0.1:9"})
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a.png")
            with open(path, "wb") as f:
                f.write(b"png")
            with mock.patch.object(c, "_open", lambda req, t=None: io.BytesIO(
                    b"<html>502 Bad Gateway</html>")):
                with self.assertRaises(ig.ComfyError) as caught:
                    c.upload_image(path)
        self.assertIn("not JSON", str(caught.exception))
        self.assertEqual(c.uploaded, {})                    # nothing believed sent

    def test_a_template_that_will_not_read_is_logged_not_dropped_silently(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp:
            for name, text in (("good", '{"graph": {}}'), ("bad", '{"graph": {},}')):
                with open(os.path.join(tmp, name + ".json"), "w", encoding="utf-8") as f:
                    f.write(text)
            with mock.patch.object(ig.doctor, "log_error") as log:
                wfs = ig.list_workflows(tmp)
        self.assertEqual([w["id"] for w in wfs], ["good"])
        (text,), _ = log.call_args
        self.assertIn("bad.json", text)


class FakeWeb:
    """urlopen for fetch_picture: url -> bytes or (bytes, content type);
    gone.example is 404, private.example 403, anything else unreachable."""

    def __init__(self, pages, asked=None):
        self.pages, self.asked = pages, asked if asked is not None else []

    def __call__(self, req, timeout=None):
        import io
        import urllib.error
        url = req.full_url
        self.asked.append(url)
        for host, code in (("gone.example", 404), ("private.example", 403)):
            if host in url:
                raise urllib.error.HTTPError(url, code, "No", {}, None)
        if url not in self.pages:
            raise urllib.error.URLError("no such host")
        body = self.pages[url]
        body, ctype = body if isinstance(body, tuple) else (body, "image/whatever")

        class Answer(io.BytesIO):
            headers = {"Content-Type": ctype}

            def geturl(self):
                return url
        return Answer(body)


class TestLibrary(unittest.TestCase):
    def test_image_library_keeps_copies_deduplicates_and_survives_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            src = os.path.join(folder, "reference.png")
            with open(src, "wb") as f:
                f.write(PNG)
            lib = ig.Library(os.path.join(folder, "library"))
            record = lib.import_image(src)
            self.assertEqual(record, lib.import_image(src))
            os.remove(src)
            reopened = ig.Library(lib.root)
            self.assertEqual(reopened.all("images"), [record])
            self.assertEqual(record["name"], "reference.png")
            with open(record["path"], "rb") as f:
                self.assertEqual(f.read(), PNG)
            reopened.save_images([])
            self.assertEqual(ig.Library(lib.root).all("images"), [])
            self.assertTrue(os.path.isfile(record["path"]))

    def test_image_library_rejects_non_images_and_rolls_back_failed_save(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as folder:
            src = os.path.join(folder, "reference.png")
            lib = ig.Library(os.path.join(folder, "library"))
            with open(src, "w") as f:
                f.write("not an image")
            with self.assertRaises(ValueError):
                lib.import_image(src)
            with open(src, "wb") as f:
                f.write(PNG)
            with patch.object(ig.os, "replace", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    lib.import_image(src)
            self.assertEqual(lib.all("images"), [])
            self.assertEqual(ig.Library(lib.root).all("images"), [])

    def test_a_failed_save_of_any_kind_rolls_back_in_memory_too(self):
        """Only test_image_library_rejects_non_images_and_rolls_back_failed_save
        used to be true: Library.save() published the new records in memory
        before the atomic write, so a disk-full/read-only failure on any
        other kind (loras, identities, ...) left the in-memory list claiming
        an edit that was never actually saved."""
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as folder:
            lib = ig.Library(folder)
            before = lib.all("loras")
            with patch.object(ig.os, "replace", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    lib.save("loras", [{"id": "new", "file": "new.safetensors", "name": "New"}])
            self.assertEqual(lib.all("loras"), before)
            self.assertEqual(ig.Library(folder).all("loras"), before)

    def test_junk_on_disk_costs_the_record_not_the_list(self):
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "identities.json"), "w") as f:
            json.dump([{"name": "Sitter", "strength": "lots"}, "junk", {"no": "name"},
                       {"name": "Sitter"}], f)
        with open(os.path.join(d, "backends.json"), "w") as f:
            f.write("{not json")
        lib = ig.Library(d)
        self.assertEqual([i["id"] for i in lib.all("identities")], ["sitter", "sitter-2"])
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
        n = lib.merge_loras("3090", ["sitter_identity_flux.safetensors",
                                     "kodak_portra_sdxl.safetensors"])
        self.assertEqual(n, 2)
        a, b = lib.all("loras")
        self.assertEqual((a["category"], a["family"]), ("Identity", "flux1"))
        self.assertEqual((b["category"], b["family"]), ("Camera / Film", "sdxl"))
        self.assertEqual(lib.merge_loras("3090", ["sitter_identity_flux.safetensors"]), 0)

    def test_references_are_copied_into_the_studio(self):
        d = tempfile.mkdtemp()
        src = os.path.join(d, "me.png")
        with open(src, "wb") as f:
            f.write(PNG)
        lib = ig.Library(os.path.join(d, "studio"))
        kept = lib.keep_reference(src, "Sitter")
        self.assertTrue(kept.startswith(lib.root))
        self.assertEqual(kept, lib.keep_reference(src, "Sitter"))

    def test_a_picture_from_a_link_is_kept_like_an_uploaded_one(self):
        """A link's picture becomes a file under references/, named by its
        bytes as an upload is, so the same picture by link or by file is one
        file and a link that dies later breaks nothing."""
        d = tempfile.mkdtemp()
        src = os.path.join(d, "glasses.png")
        with open(src, "wb") as f:
            f.write(PNG)
        lib = ig.Library(os.path.join(d, "studio"))
        asked = []
        kept = lib.keep_link("example.com/glasses", "Ada items",
                             opener=FakeWeb({"https://example.com/glasses": PNG}, asked))
        self.assertEqual(asked, ["https://example.com/glasses"])
        self.assertTrue(kept.startswith(os.path.join(lib.root, "references", "ada-items")))
        self.assertTrue(kept.endswith(".png"))
        self.assertEqual(kept, lib.keep_reference(src, "Ada items"))

    def test_a_link_may_be_the_picture_a_page_a_search_result_or_data(self):
        jpeg = b"\xff\xd8\xff\xe0" + b"\0" * 32
        page = (b'<html><head><meta content="/img/dress.jpg?w=800&amp;q=1" '
                b'property="og:image"><meta name="twitter:image" content="x.png">'
                b'</head></html>')
        web = FakeWeb({"https://shop.example/dress": (page, "text/html; charset=utf-8"),
                       "https://shop.example/img/dress.jpg?w=800&q=1": jpeg,
                       "https://cdn.example/a.webp": b"RIFF\0\0\0\0WEBPVP8 ",
                       "https://blog.example/": (b"<html>nothing</html>", "text/html"),
                       "https://docs.example/a.pdf": (b"%PDF-1.4", "application/pdf"),
                       "https://phone.example/a": b"\0\0\0\x1cftypheic" + b"\0" * 8})
        self.assertEqual(ig.fetch_picture("https://shop.example/dress", web), (jpeg, ".jpg"))
        self.assertEqual(ig.fetch_picture(
            "https://www.google.com/imgres?imgurl=https%3A%2F%2Fcdn.example%2Fa.webp"
            "&imgrefurl=x", web)[1], ".webp")
        self.assertEqual(ig.fetch_picture(
            "data:image/png;base64," + base64.b64encode(PNG).decode(), web), (PNG, ".png"))
        for url, says in [("https://blog.example/", "copy its image address"),
                          ("https://docs.example/a.pdf", "application/pdf"),
                          ("https://phone.example/a", "HEIC"),
                          ("https://gone.example/a.png", "HTTP 404"),
                          ("https://private.example/a.png", "upload it instead"),
                          ("ftp://example.com/a.png", "Only http and https"),
                          ("https://me:pw@example.com/a.png", "password"),
                          ("   ", "Paste a link")]:
            with self.assertRaises(ig.LinkError) as err:
                ig.fetch_picture(url, web)
            self.assertIn(says, str(err.exception), url)
        self.assertEqual(ig.picture_ext(b"GIF89a..."), ".gif")
        self.assertEqual(ig.picture_ext(b"<svg/>"), "")

    def test_a_link_bigger_than_a_picture_is_refused(self):
        from unittest import mock
        big = b"\x89PNG\r\n\x1a\n" + b"\0" * 64
        with mock.patch.object(ig, "PICTURE_BYTES", 32):
            with self.assertRaises(ig.LinkError) as err:
                ig.fetch_picture("https://example.com/big.png",
                                 FakeWeb({"https://example.com/big.png": big}))
        self.assertIn("over", str(err.exception))

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

    def test_default_cameras_load_with_a_no_camera_choice_first(self):
        lib = ig.Library(tempfile.mkdtemp())
        cams = lib.all("camera_profiles")
        self.assertEqual(cams[0]["id"], "none")
        self.assertEqual(cams[0]["chemistry"], "")
        self.assertTrue(all(c["name"] and isinstance(c["chemistry"], str) for c in cams))

    def test_clean_camera_profile_drops_junk_and_keeps_a_blank_lens_optional(self):
        self.assertIsNone(ig.clean_camera_profile({"name": ""}))
        self.assertIsNone(ig.clean_camera_profile("not a dict"))
        cam = ig.clean_camera_profile({"name": "Leica M6", "chemistry": "Warm tones.",
                                       "lens": "35", "image": "", "notes": 7})
        self.assertEqual(cam, {"id": "leica-m6", "name": "Leica M6",
                               "chemistry": "Warm tones.", "lens": 35.0, "lenses": [],
                               "format": "3:2", "image": "", "notes": ""})
        # A camera's format is one of CAMERA_FORMATS or nothing; a starter
        # camera saved before formats existed takes its own.
        self.assertEqual(ig.clean_camera_profile({"name": "Mine", "format": "3x2"})["format"], "3:2")
        self.assertEqual(ig.clean_camera_profile({"name": "Mine", "format": "16x9"})["format"], "16:9")
        self.assertEqual(ig.clean_camera_profile({"name": "Mine", "format": "5:4"})["format"], "")
        self.assertEqual(ig.clean_camera_profile({"name": "Mine"})["format"], "")
        self.assertIsNone(ig.clean_camera_profile({"name": "No lens"})["lens"])
        # A lens out of the camera's own sane range is clamped, like a style's.
        self.assertEqual(ig.clean_camera_profile({"name": "Wild", "lens": 5000})["lens"], 300)

    def test_a_cameras_lenses_are_read_from_a_list_or_the_editors_lines(self):
        lines = "13 f/2.2 0.6x ultra-wide\n67mm f2.4 - 3x telephoto\n\n50\njunk\n13 again"
        self.assertEqual(ig.clean_lenses(lines), [
            {"name": "0.6x ultra-wide", "mm": 13.0, "f": 2.2},
            {"name": "3x telephoto", "mm": 67.0, "f": 2.4},
            {"name": "50mm", "mm": 50.0, "f": None}])
        # The editor shows them the way it reads them back.
        self.assertEqual(ig.clean_lenses(ig.lens_lines(ig.clean_lenses(lines))),
                         ig.clean_lenses(lines))
        self.assertEqual(ig.clean_lenses([{"mm": 5000, "f": 0.1}, "x", {"name": "no mm"}]),
                         [{"name": "300mm", "mm": 300.0, "f": 0.7}])
        # A camera with lenses and no native lens starts on its first.
        cam = ig.clean_camera_profile({"name": "Phone", "lenses": "23 f/1.7 Wide\n67 Tele"})
        self.assertEqual(cam["lens"], 23.0)

    def test_the_starter_cameras_include_the_phone_and_the_cinema_camera(self):
        cams = {c["id"]: c for c in ig.Library(tempfile.mkdtemp()).all("camera_profiles")}
        for cid in ("galaxy-s25-ultra", "canon-rebel-sl1", "sony-zv1", "bmpcc-6k-g2",
                    "nikon-n80", "sony-dsc-s650", "lomo-konstruktor-f"):
            cam = cams[cid]
            self.assertTrue(cam["lenses"], cid)
            self.assertIsNotNone(ig.lens_at(cam["lenses"], cam["lens"]), cid)  # native is one
            self.assertIn(cam["format"], ig.CAMERA_FORMATS[1:], cid)
            self.assertIn(cam["name"].lower(), cam["chemistry"].lower(), cid)
        self.assertEqual(cams["galaxy-s25-ultra"]["format"], "4:3")
        self.assertEqual(cams["bmpcc-6k-g2"]["format"], "16:9")

    def test_new_formats_reshape_to_the_scene_builders_frames(self):
        import apps.image_studio.scene.scene as sc
        self.assertEqual(ig.camera_size("4:3", 896, 1152), (896, 1152))
        self.assertEqual(ig.camera_size("4:3", 1024, 1024), (1152, 896))
        self.assertEqual(ig.camera_size("16:9", 1024, 1024), (1344, 768))
        for fmt in ig.CAMERA_FORMATS[1:]:
            self.assertIn(fmt, sc.FORMAT_FRAMES)

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


class SwapClient(DressClient):
    """A ComfyUI with FLUX, Qwen and SAM3 that answers a fix run ("fs") with
    its picture, named after the run's prefix."""

    def listen_for_progress(self, pid, on_event, stop=None, timeout=0):
        graph = self.graphs[int(pid[3:]) - 1]
        if "fs" not in graph:
            return super().listen_for_progress(pid, on_event, stop, timeout)
        prefix = graph["fs"]["inputs"]["filename_prefix"]
        name = os.path.basename(prefix).split("_")[0] + ("_swap" if prefix.endswith("_swap")
                                                         else "")
        return {"status": {"completed": True}, "outputs": {"fs": {"images": [
            {"filename": name + "_00001_.png", "subfolder": "ImageStudio", "type": "output"}]}}}


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

    def test_a_tag_said_in_the_scene_brings_its_picture(self):
        refs = {"glasses": self.pic("glasses"), "Red Dress": self.pic("dress"),
                "top hat": self.pic("hat"), "necklace": self.pic("necklace")}
        s = dict(ig.default_settings(), model="flux-dev", item_refs=refs,
                 scene="She pushes her Glasses up, in a red dress and a top hat.")
        o = ig.outfit_of(s)
        self.assertEqual([c["name"] for c in o["clothes"]], ["Red Dress"])
        self.assertEqual([a["name"] for a in o["accessories"]], ["glasses", "top hat"])
        # A word inside another is not said, and an unsaid tag stays out.
        self.assertFalse(ig.outfit_of(dict(s, scene="In sunglasses."))["accessories"])
        self.assertTrue(ig.says("round glasses", "glasses"))
        self.assertTrue(ig.says("two  top hats", "top hat"))
        self.assertFalse(ig.says("sunglasses", "glasses"))
        self.assertFalse(ig.says("anything", ""))
        # A slot's words count too, and a slot's own pick is not taken twice.
        worn = ig.outfit_of(dict(s, scene="", accessories="round glasses, necklace",
                                 top="summer dress"))
        self.assertEqual([a["name"] for a in worn["accessories"]], ["necklace", "glasses"])
        self.assertEqual(worn["clothes"], [])
        inv = dict(FLUX_FILES, **KONTEXT_FILES)
        p = ig.compose(s, self.studio.lib, self.backend("5090"), inv)
        self.assertEqual(p.errors, [])
        self.assertEqual([n for n, _ in p.items], ["Red Dress", "glasses", "top hat"])
        self.assertEqual(p.references["item: glasses"], refs["glasses"])
        self.assertIn("The Red Dress, glasses and top hat look exactly as in the reference",
                      p.prompt)

    def test_tags_are_the_things_on_the_person(self):
        refs = {k: self.pic(k.replace(" ", "_")) for k in
                ("glasses", "earring", "dress", "rose tattoo", "tattoo")}
        s = dict(ig.default_settings(), item_refs=refs, traits="freckles, tattoos",
                 accessories="hoop earrings", scene="She wears her dress and glasses.")
        o = ig.outfit_of(s)
        self.assertEqual([c["name"] for c in o["clothes"]], ["dress"])
        self.assertEqual(sorted(a["name"] for a in o["accessories"]),
                         ["earring", "glasses", "tattoo"])
        # Said on the skin by its own words, in the Traits slot or the scene.
        o = ig.outfit_of(dict(s, traits="a rose tattoo on the wrist", scene=""))
        self.assertEqual(sorted(a["name"] for a in o["accessories"]),
                         ["earring", "rose tattoo"])      # the longer tag, not both
        o = ig.outfit_of(dict(s, traits="a rose tattoo and a tattoo", scene=""))
        self.assertIn("tattoo", [a["name"] for a in o["accessories"]])

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
    def test_codex_finish_round_trip_keeps_protected_pixels_and_completes_pipeline(self):
        import apps.image_studio.ui as ui_mod
        from core.icons import png, png_to_rgba
        from unittest.mock import patch
        _, ui = self.tab()
        original = png(bytes([30, 40, 50, 255]) * 16, 4, 4)
        self.addCleanup(ui.codex_finish.set, ui.codex_finish.get())
        settings = dict(ig.default_settings(), scene="festival", codex_finish=True,
                        scene_faces={"people": [{"region": [0, 0, .5, .5]}]})
        job = ig.Job(settings, {"id": "5090", "name": "Test"})
        rec = {"id": "codex-ui-base", "created_ts": time.time(), "created": "2026-10-02T12:00:00",
               "settings": settings, "prompt": "festival"}
        ui.studio.save_result(job, rec, [("base.png", original)])
        ui.studio.queue._finish(job, "complete")
        ui.jobs.append(job)
        ui._select(("record", job.record))
        window = ui_mod.CodexFinishWindow(ui, job.record, job.outputs[0])
        self.addCleanup(window.win.destroy)
        clipboard = []
        with patch.object(ui.host, "_spawn", side_effect=lambda sid, fn: fn()), \
                patch.object(ui.host, "clipboard_clear", side_effect=clipboard.clear), \
                patch.object(ui.host, "clipboard_append", side_effect=clipboard.append), \
                patch.object(ui.host, "clipboard_get", side_effect=lambda: "".join(clipboard)):
            window.export()
            self.pump(lambda: not window.busy)
            self.assertIn("protected", ui.host.clipboard_get())
            self.assertIn(window.handoff["redacted_image"], ui.host.clipboard_get())
            self.assertNotIn(window.path, ui.host.clipboard_get())
            self.assertEqual(window.picture.get(0, 0), (96, 96, 96))
            with open(window.handoff["output"], "wb") as stream:
                stream.write(png(bytes([80, 60, 40, 255]) * 16, 4, 4))
            with patch.object(ui_mod.filedialog, "askopenfilename", return_value=window.handoff["output"]):
                window.take()
                self.pump(lambda: not window.busy)
        self.assertEqual(job.status, "complete")
        self.assertEqual(ui._selected_record()["workflow"], "codex-imagegen")
        with open(job.outputs[0], "rb") as stream:
            pixels, _, _ = png_to_rgba(stream.read())
        self.assertEqual(pixels[:8], bytes([30, 40, 50, 255]) * 2)
        self.assertEqual(pixels[8:16], bytes([80, 60, 40, 255]) * 2)
        ui.codex_finish.set(True)
        self.assertTrue(ui.collect()["codex_finish"])

    def test_head_photo_opens_position_lock_window_with_kept_source(self):
        from unittest.mock import patch
        import apps.image_studio.ui as ui_mod
        _, ui = self.tab()
        src = os.path.join(self.dir, "head-choice.png")
        with open(src, "wb") as f:
            f.write(PNG)
        before = list(ui.studio.lib.all("identities"))
        with patch.object(ui.host, "_spawn", side_effect=lambda sid, fn: fn()), \
                patch.object(ui_mod, "FixWindow") as window:
            ui._import_head_photo(src)
            self.pump(lambda: not ui.head_photo_busy)
        kept = window.call_args.args[1]
        self.assertNotEqual(kept, src)
        self.assertTrue(os.path.isfile(kept))
        self.assertEqual(window.call_args.kwargs, {"around_head": True})
        self.assertEqual(ui.studio.lib.all("identities"), before)

    def test_angles_asks_on_a_view_cube_and_keeps_the_pick_as_the_preset(self):
        from unittest.mock import patch
        import apps.image_studio.blend as sb
        import apps.image_studio.ui as ui_mod
        _, ui = self.tab()
        photos = []
        for n in range(2):
            photos.append(os.path.join(self.dir, "angle_src_%d.png" % n))
            with open(photos[-1], "wb") as f:
                f.write(PNG)
        sb.save_views(["back left", "front"])
        editor = type("Editor", (), {})()
        editor.owner, editor.win = ui, ui_mod.tk.Toplevel(self.app)
        self.addCleanup(editor.win.destroy)
        with patch.object(ui.host, "_spawn") as spawn:
            w = ui_mod.NewPhotos(editor, {"paths": photos, "sel": {0, 1}}, "angles", photos)
            spawn.assert_not_called()                    # it asks before it draws
            self.assertEqual(w.cube.chosen, ["back left", "front"])
            self.assertEqual(w.jobs(), [(v, p, v) for p in photos
                                        for v in ("back left", "front")])
            self.assertIn("4 photos a round", w.picked.cget("text"))
            w.cube.toggle("straight above")              # a click on the cube's top
            self.assertEqual(sb.load_views(), ["back left", "front", "straight above"])
            w.cube.set_chosen([])
            w.start()
            spawn.assert_not_called()                    # nothing picked, nothing run
            self.assertIn("Pick at least one", w.msg.cget("text"))
            w.cube.set_chosen(["right side"])
            w.start()
            spawn.assert_called_once()
        w.running = False
        w.close()

    def test_an_angles_round_clears_the_card_frees_kontext_and_always_ends(self):
        # Ideas 634988cde86d and c94e9a01ed0b: a round bypasses the queue, so
        # it never cleared LM Studio off a shared card nor freed Kontext after;
        # and an error other than ComfyError/OSError skipped done(), leaving
        # Make answering "Still making the last round" until reopened.
        from unittest.mock import Mock, patch
        import apps.image_studio.blend as sb
        import apps.image_studio.ui as ui_mod
        _, ui = self.tab()
        photo = os.path.join(self.dir, "angle_src.png")
        with open(photo, "wb") as f:
            f.write(PNG)
        editor = type("Editor", (), {})()
        editor.owner, editor.win = ui, ui_mod.tk.Toplevel(self.app)
        self.addCleanup(editor.win.destroy)
        backend = {"id": "kontext", "name": "Kontext PC", "url": "http://127.0.0.1:1",
                   "shares_llm_gpu": True, "release_vram": True}
        client = FakeClient(backend)
        room = Mock()
        with patch.object(ui.host, "_spawn", side_effect=lambda sid, fn: fn()), \
                patch.object(sb, "route", return_value=(backend, "")), \
                patch.object(ui.studio, "client", return_value=client), \
                patch.object(ui.studio, "make_room", room), \
                patch.object(sb, "run", side_effect=KeyError("images")), \
                patch.object(ui_mod.doctor, "log_error") as logged:
            w = ui_mod.NewPhotos(editor, {"paths": [photo], "sel": {0}}, "angles", [photo])
            w.cube.set_chosen(["front"])
            w.start()
            self.pump(lambda: not w.running)
        room.assert_called_once_with(backend)
        self.assertEqual(client.freed, 1)
        self.assertIn("KeyError", w.msg.cget("text"))
        self.assertIn("Traceback", logged.call_args.args[0])
        w.close()

    def test_blend_is_a_window_of_its_own_that_sends_a_job(self):
        from unittest.mock import patch
        import apps.image_studio.ui as ui_mod
        _, ui = self.tab()
        a, b = (os.path.join(self.dir, "blend_%s.png" % n) for n in "ab")
        for p in (a, b):
            with open(p, "wb") as f:
                f.write(PNG)
        w = ui.blend(a)
        self.addCleanup(lambda: w.win.winfo_exists() and w.win.destroy())
        self.assertIs(ui.blend(), w)                     # one window, raised
        with patch.object(ui.host, "_spawn") as spawn:
            w.start()
            spawn.assert_not_called()                    # one picture is not a blend
            self.assertIn("Choose two pictures", w.msg.cget("text"))
            ui.blend(b)                                  # the next goes in the empty place
            self.assertEqual(w.paths, [a, b])
            w.swap()
            self.assertEqual(w.paths, [b, a])
            w.words.set("in snow")
            w.start()
            sent = spawn.call_args.args[2]
        self.assertEqual(sent, {"mode": "blend", "seed": -1, "backend": "auto", "blend": {
            "images": [b, a], "person": False, "words": "in snow"}})
        self.assertEqual(ui.view, "queue")
        with patch.object(ui_mod, "ImageLibraryWindow") as library:
            w.from_library(1)
            library.call_args.kwargs["pick"](a)          # what the library's chooser calls
        self.assertEqual(w.paths, [b, a])
        ui.reuse(dict(sent, blend={"images": [a, b], "person": True, "words": "at dusk"}))
        self.assertEqual((w.paths, w.person.get(), w.words.get()), ([a, b], True, "at dusk"))

    def test_resting_on_a_thumbnail_shows_the_picture_larger(self):
        import apps.image_studio.ui as ui_mod
        _, ui = self.tab()
        big = os.path.join(self.dir, "peek_big.png")
        ui_mod.tk.PhotoImage(master=self.app, width=400, height=300).write(big, format="png")
        win = ui_mod.tk.Toplevel(self.app)
        self.addCleanup(win.destroy)
        box = ui._thumb(win, big)
        box.pack()
        self.pump(lambda: box.img.winfo_ismapped())
        pk = self.app._studio_peek
        self.assertIs(box._peek, box.img._peek)     # frame and picture are one picture
        box.img.event_generate("<Enter>", x=5, y=5)
        self.assertFalse(pk.shown())                # not before the pointer rests
        self.pump(pk.shown)
        img = pk.win.label.image
        self.assertEqual((img.width(), img.height()), (400, 300))   # never past its own size
        self.assertGreater(img.width(), ui.px(ui_mod.THUMB))
        box.img.event_generate("<ButtonPress-1>", x=5, y=5)
        self.assertFalse(pk.shown())                # a click still selects, and closes it
        # A thumbnail whose picture changes shows the new one; no picture, nothing.
        ui.set_thumb(box, None)
        box.img.event_generate("<Enter>", x=5, y=5)
        self.app.after(ui_mod.PEEK_DELAY + 100)
        self.app.update()
        self.assertFalse(pk.shown())
        pk.hide()
        # Leaving before the delay never opens it.
        ui.set_thumb(box, big)
        box.img.event_generate("<Enter>", x=5, y=5)
        box.img.event_generate("<Leave>", x=-50, y=-50)
        self.app.after(ui_mod.PEEK_DELAY + 100)
        self.app.update()
        self.assertFalse(pk.shown())

    def test_characters_tab_lists_identities_with_their_photo_count(self):
        _, ui = self.tab()
        photos = []
        for name in ("char_a.png", "char_b.png"):
            photos.append(os.path.join(self.dir, name))
            with open(photos[-1], "wb") as f:
                f.write(PNG)
        ui.studio.lib.save("identities", [{"name": "Partner", "references": photos}])
        try:
            ui._show_list("characters")
            self.assertEqual(ui.tab_chars.roles, ui.host.PILL_ROLES["option"])
            self.assertEqual(ui.tab_queue.roles, ui.host.PILL_ROLES["ghost"])

            def gather(w, kind):
                out = [w] if type(w).__name__ == kind else []
                for c in w.winfo_children():
                    out += gather(c, kind)
                return out
            shown = [w.cget("text") for w in gather(ui.list_box, "Label")]
            names = [e.get() for e in gather(ui.list_box, "Entry")]
            self.assertIn("Partner", names)
            self.assertTrue(any("2 reference photos" in t for t in shown), shown)
        finally:
            ui.studio.lib.save("identities", [])
            ui._show_list("queue")

    def test_new_character_can_be_renamed_from_the_tab(self):
        import apps.image_studio.ui as ui_mod
        _, ui = self.tab()
        before = {r["id"] for r in ui.studio.lib.all("identities")}
        try:
            ui._new_character()
            added = [r for r in ui.studio.lib.all("identities") if r["id"] not in before]
            self.assertEqual(len(added), 1)
            self.assertEqual(added[0]["name"], "New person")
            var = ui_mod.tk.StringVar(value="Sabine")
            ui._rename_character(added[0], var)
            rec = ui.studio.lib.get("identities", added[0]["id"])
            self.assertEqual(rec["name"], "Sabine")
        finally:
            ui.studio.lib.save("identities", [r for r in ui.studio.lib.all("identities")
                                              if r["id"] in before])
            ui._show_list("queue")

    def test_add_and_remove_a_characters_photo_from_the_tab(self):
        from unittest.mock import patch
        import apps.image_studio.ui as ui_mod
        _, ui = self.tab()
        src = os.path.join(self.dir, "new_char_photo.png")
        with open(src, "wb") as f:
            f.write(PNG)
        ui.studio.lib.save("identities", [{"name": "New Face"}])
        rec = next(r for r in ui.studio.lib.all("identities") if r["name"] == "New Face")
        try:
            with patch.object(ui_mod.filedialog, "askopenfilenames", return_value=(src,)), \
                    patch.object(ui.host, "_spawn", side_effect=lambda sid, fn: fn()), \
                    patch.object(ui_mod.face_finder, "problem", return_value="off in tests"):
                ui._add_character_photo(rec)
                self.pump(lambda: ui.studio.lib.get("identities", rec["id"])["references"])
            added = ui.studio.lib.get("identities", rec["id"])["references"]
            self.assertEqual(len(added), 1)
            self.assertNotEqual(added[0], src)          # copied into the library, not linked
            self.assertTrue(os.path.isfile(added[0]))
            with patch.object(ui_mod.messagebox, "askyesno", return_value=True):
                ui._remove_character_photo(ui.studio.lib.get("identities", rec["id"]), added[0])
            self.assertEqual(ui.studio.lib.get("identities", rec["id"])["references"], [])
        finally:
            ui.studio.lib.save("identities", [r for r in ui.studio.lib.all("identities")
                                              if r["id"] != rec["id"]])
            ui._show_list("queue")

    def test_head_window_requires_lock_and_queues_without_face_redraw(self):
        from unittest.mock import patch
        import apps.image_studio.ui as ui_mod
        _, ui = self.tab()
        src = os.path.join(self.dir, "head-window.png")
        with open(src, "wb") as f:
            f.write(PNG)
        settings = dict(ig.default_settings(), model="flux-dev", scene="A woman in a garden")
        window = ui_mod.FixWindow(ui, src, settings, around_head=True)
        with patch.object(ui.host, "_spawn") as spawn:
            window._redraw()
            spawn.assert_not_called()
            window.locks = [{"x": 50, "y": 50, "size": 64}]
            window._redraw()
        sent = spawn.call_args.args[2]
        self.assertTrue(sent["fix"]["around_head"])
        self.assertEqual(sent["fix"]["locks"], window.locks)
        self.assertEqual(sent["fix"]["spots"], [])
        self.assertEqual(sent["fix"]["face_swap"], "")
        self.assertEqual(sent["identities"], [])
        self.assertFalse(sent["critic_notes"])
        self.assertFalse(sent["hand_pass"])

    def test_image_library_is_past_generations_search_and_use_reaches_settings(self):
        s, ui = self.tab()
        before = ui.collect()
        self.addCleanup(lambda: ui.apply(before))
        ui.scene.delete("1.0", "end")
        ui.scene.insert("1.0", "A red bicycle against a white wall")
        ui.settings["model"] = "z-image-turbo"
        ui.settings["backend"] = "auto"
        n = len(ui.jobs)
        ui.generate()
        self.pump(lambda: len(ui.jobs) > n and ui.jobs[0].status in ig.FINISHED)
        self.assertEqual(ui.jobs[0].status, "complete", ui.jobs[0].detail)
        rec = ui.studio.history.list()[0]
        window = ui.image_library()
        self.addCleanup(window.win.destroy)
        found = window.selected()
        self.assertEqual(found["path"], rec["images"][0])
        self.assertIn("red bicycle", found["name"])
        window.search.set("no-match-at-all-xyz")
        self.assertIsNone(window.selected())
        window.search.set("red bicycle")
        for kind, label, _ in ig.REFERENCE_KINDS:
            window.role.set(label)
            window.use()
            self.assertEqual(ui.collect()["references"][kind], rec["images"][0])
        self.assertIsNone(ui.pose)

    @classmethod
    def setUpClass(cls):
        import core.chat as studio_chat
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

    def test_an_offline_backend_on_this_pc_has_a_start_button(self):
        """Offline + a start file here: Start launches it (patched - nothing
        real starts), the row says starting, and ready once it answers."""
        from unittest.mock import patch
        import apps.image_studio.ui as ui_mod
        import core.ui as core_ui
        _, ui = self.tab()
        cmd = os.path.join(self.dir, "Start ComfyUI.cmd")
        with open(cmd, "w") as f:
            f.write("@echo off\n")
        b = dict(ui.studio.backend("5090"), start=cmd)
        other = ui.studio.backend("3090")

        def words(cell):
            out = []
            for w in cell.winfo_children():
                try:
                    out.append(w.cget("text"))   # a Label, or a Pill's own cget
                except ui_mod.tk.TclError:
                    pass                         # the dot
            return out

        def texts():
            return [t for cell in ui.health_row.winfo_children() for t in words(cell)]

        FakeClient.down = {"5090", "3090"}
        polls = []

        def poll(*a, **k):
            polls.append(1)
            if len(polls) >= 2:
                FakeClient.down = set()   # it comes up on the second look
            return real_check(*a, **k)
        real_check = ui.studio.check
        try:
            with patch.object(ui.studio, "backends", return_value=[b, other]), \
                    patch.object(ui.studio, "backend", side_effect=lambda bid: b if bid == "5090" else other):
                ui.studio.check(b, full=False)
                ui.studio.check(other, full=False)
                ui._paint_health()
                starts = [w for cell in ui.health_row.winfo_children()
                          for w in cell.winfo_children() if isinstance(w, core_ui.Pill)
                          and w.cget("text") == "Start"]
                # The 3090's `start` is advice: no button for it.
                self.assertEqual(len(starts), 1)
                with patch.object(ig.subprocess, "Popen") as popen, \
                        patch.object(ui, "START_POLL", 0.01), \
                        patch.object(ui.studio, "check", side_effect=poll):
                    starts[0].command()
                    self.assertEqual(popen.call_args.args[0], ["cmd", "/c", cmd])
                    self.assertIn("starting", " ".join(texts()))
                    self.assertNotIn("Start", texts())
                    starts[0].command()             # a second click starts nothing more
                    self.assertEqual(popen.call_count, 1)
                    self.pump(lambda: not ui.starting and "ready" in " ".join(texts()))
                self.assertEqual(texts().count("Start"), 0)
                self.assertIn("is up", ui.note.cget("text"))
        finally:
            FakeClient.down = set()
            ui.studio.check_all()
            ui._paint_health()

    def test_generate_stays_visible_and_advanced_keeps_loras(self):
        _, ui = self.tab()
        ui._show_section("Image")
        if ui.adv_open:
            ui._toggle_advanced()
        self.app.update()
        self.assertTrue(ui.go.winfo_ismapped())
        self.assertFalse(ui.lora_box.winfo_ismapped())
        self.assertTrue(ui.scene.winfo_ismapped())
        self.assertFalse(ui.preset_pill.winfo_ismapped())
        before = ui.collect()
        for section in ui.sections:
            ui._show_section(section)
            self.app.update()
            self.assertEqual([k for k, box in ui.sections.items() if box.winfo_ismapped()],
                             [section])
            self.assertTrue(ui.go.winfo_ismapped())
            for pill in ui.section_pills.values():
                self.assertTrue(pill.winfo_ismapped())
                self.assertGreaterEqual(pill.winfo_width(), pill.winfo_reqwidth())
        ui._toggle_advanced()
        self.app.update()
        self.assertTrue(ui.adv_box.winfo_ismapped())
        self.assertTrue(ui.go.winfo_ismapped())
        self.assertEqual(ui.collect(), before)
        ui._toggle_advanced()
        ui._show_section("Image")

    def test_the_hands_pass_is_on_unless_unticked_and_a_picture_brings_it_back(self):
        _, ui = self.tab()
        self.assertTrue(ui.collect()["hand_pass"])
        ui.apply(dict(ui.collect(), hand_pass=False))
        self.assertFalse(ui.collect()["hand_pass"])
        ui.apply({k: v for k, v in ui.collect().items() if k != "hand_pass"})
        self.assertTrue(ui.collect()["hand_pass"])     # an older picture: it was on
        # The head swap before the face swap, likewise.
        self.assertTrue(ui.collect()["head_swap"])
        self.assertTrue(ui.collect()["scene_details_pass"])
        self.assertTrue(ui.collect()["smile_pass"])
        ui.apply(dict(ui.collect(), smile_pass=False))
        self.assertFalse(ui.collect()["smile_pass"])
        ui.apply({k: v for k, v in ui.collect().items() if k != "smile_pass"})
        self.assertTrue(ui.collect()["smile_pass"])
        ui.apply(dict(ui.collect(), scene_details_pass=False))
        self.assertFalse(ui.collect()["scene_details_pass"])
        ui.apply({k: v for k, v in ui.collect().items() if k != "scene_details_pass"})
        self.assertTrue(ui.collect()["scene_details_pass"])
        ui.apply(dict(ui.collect(), head_swap=False))
        self.assertFalse(ui.collect()["head_swap"])
        ui.apply({k: v for k, v in ui.collect().items() if k != "head_swap"})
        self.assertTrue(ui.collect()["head_swap"])
        # The glasses pass is the other way about: left to the swap (None)
        # unless ticked, and an older picture leaves it so.
        self.assertIsNone(ui.collect()["glasses_pass"])
        self.assertFalse(ig.glasses_pass(ui.collect(), [{"masks": ["box", "occlusion"]}]))
        ui.apply(dict(ui.collect(), glasses_pass=True))
        self.assertIs(ui.collect()["glasses_pass"], True)
        self.assertTrue(ig.glasses_pass(ui.collect(), [{"masks": ["box", "occlusion"]}]))
        ui.apply({k: v for k, v in ui.collect().items() if k != "glasses_pass"})
        self.assertIsNone(ui.collect()["glasses_pass"])
        texts = []

        def walk(w):
            for c in w.winfo_children():
                if c.winfo_class() == "Checkbutton":
                    texts.append(c.cget("text"))
                walk(c)
        walk(ui.frame_root if hasattr(ui, "frame_root") else self.app)
        self.assertIn("Redraw glasses after the face swap", texts)

    def test_the_picture_card_holds_no_prompt_only_a_failed_face_swap_note(self):
        _, ui = self.tab()
        src = os.path.join(self.dir, "shown.png")
        with open(src, "wb") as f:
            f.write(PNG)
        rec = {"id": "r1", "images": [src], "prompt": "partner. A woman on a meadow",
               "settings": {}, "warnings": ["Chest size is using words only."]}
        ui._select(("record", rec))
        self.app.update()
        self.assertFalse(ui.caption.winfo_ismapped())
        self.assertFalse(ui.act_retry_faces.winfo_ismapped())
        failed = dict(rec, id="r2", finish={"profiles": ["partner"], "state": "failed",
                                             "error": "No face found."})
        ui._select(("record", failed))
        self.app.update()
        self.assertTrue(ui.caption.winfo_ismapped())
        self.assertTrue(ui.act_retry_faces.winfo_ismapped())
        text = ui.caption.cget("text")
        self.assertIn("before the final face swap", text)
        self.assertIn("No face found.", text)
        self.assertNotIn("meadow", text)
        ui._select(("record", rec))
        self.app.update()
        self.assertFalse(ui.caption.winfo_ismapped())

    def test_fix_a_spot_marks_squares_and_queues_a_fix(self):
        import apps.image_studio.ui as ui_mod
        s, ui = self.tab()
        src = os.path.join(self.dir, "fixme.png")
        with open(src, "wb") as f:
            f.write(PNG)
        sent = []
        ui.host._spawn = lambda sid, fn, arg: sent.append(arg)
        self.addCleanup(lambda: delattr(ui.host, "_spawn"))
        fw = ui_mod.FixWindow(ui, src, dict(ig.default_settings(), scene="x"))
        self.pump(lambda: fw.img is not None)

        class Ev:
            def __init__(self, x, y, delta=0):
                self.x, self.y, self.delta = x, y, delta
        at = Ev(fw.ox + int(fw.w * fw.k / 2), fw.oy + int(fw.h * fw.k / 2))
        fw._redraw()
        self.assertEqual(sent, [])                # nothing marked: nothing sent
        fw._add(at)
        fw._wheel(Ev(at.x, at.y, 120))
        fw._pick("strength", "light")
        fw._redraw()
        (job,) = sent
        self.assertEqual(job["mode"], "fix")
        self.assertEqual(job["fix"]["strength"], "light")
        self.assertEqual(len(job["fix"]["spots"]), 1)
        self.assertGreaterEqual(job["fix"]["spots"][0]["size"], ig.FIX_MIN)
        self.assertFalse(job["fix"]["check"])     # the critic is asked only when chosen

    def test_fix_a_spot_takes_a_note_on_each_spot_and_asks_the_critic(self):
        import apps.image_studio.ui as ui_mod
        s, ui = self.tab()
        src = os.path.join(self.dir, "fixme.png")
        with open(src, "wb") as f:
            f.write(ig.oval_png(256))
        sent = []
        ui.host._spawn = lambda sid, fn, arg: sent.append(arg)
        self.addCleanup(lambda: delattr(ui.host, "_spawn"))
        fw = ui_mod.FixWindow(ui, src, dict(ig.default_settings(), scene="x"))
        self.pump(lambda: fw.img is not None)

        class Ev:
            def __init__(self, px, py):
                self.x, self.y = fw.ox + int(px * fw.k), fw.oy + int(py * fw.k)
        self.assertEqual(fw.note_where.cget("text"), "on every spot")
        fw.note.set("six fingers")                # nothing marked: a note on them all
        fw._add(Ev(60, 60))
        self.assertEqual(fw.note.get(), "")       # a new spot, its own note
        self.assertEqual(fw.note_where.cget("text"), "on spot 1")
        fw.note.set("thumb on the wrong side")
        fw._add(Ev(190, 190))
        self.assertEqual(fw.note_where.cget("text"), "on spot 2")
        fw.note.set("too small")
        fw._remove(Ev(190, 190))                  # back on the spot before
        self.assertEqual(fw.note.get(), "thumb on the wrong side")
        self.assertEqual(fw.note_where.cget("text"), "on spot 1")
        fw._add(Ev(190, 60))
        fw._pick("check", "on")
        fw._redraw()
        (job,) = sent
        fix = ig.clean_fix(job["fix"])
        self.assertTrue(fix["check"])
        self.assertEqual(fix["note"], "six fingers")
        self.assertEqual([sp.get("note") for sp in fix["spots"]],
                         ["thumb on the wrong side", None])

    def test_fix_a_spot_lassos_a_part_and_gives_it_a_photo(self):
        import apps.image_studio.ui as ui_mod
        s, ui = self.tab()
        src = os.path.join(self.dir, "fixme.png")
        with open(src, "wb") as f:
            f.write(ig.oval_png(256))
        sent = []
        ui.host._spawn = lambda sid, fn, arg: sent.append(arg)
        self.addCleanup(lambda: delattr(ui.host, "_spawn"))
        hat = os.path.join(self.dir, "hat.png")
        ui_mod.filedialog.askopenfilename = lambda **kw: hat
        self.addCleanup(lambda: delattr(ui_mod.filedialog, "askopenfilename"))
        fw = ui_mod.FixWindow(ui, src, dict(ig.default_settings(), scene="x"))
        self.pump(lambda: fw.img is not None)

        class Ev:
            def __init__(self, x, y):
                self.x, self.y = x, y

        def at(px, py):
            return Ev(fw.ox + int(px * fw.k), fw.oy + int(py * fw.k))
        fw._press(at(60, 60))
        for p in ((200, 60), (200, 200), (60, 200), (62, 62)):
            fw._drag(at(*p))
        fw._release(at(62, 62))
        (sp,) = fw.spots
        self.assertGreaterEqual(len(sp["outline"]), 4)
        self.assertAlmostEqual(sp["x"], 130, delta=4)
        fw._press(at(130, 130))                   # a click on it: its photo
        fw._release(at(130, 130))
        self.assertEqual(fw.spots[0]["photo"], hat)
        self.assertEqual(len(fw.spots), 1)
        fw._redraw()
        (job,) = sent
        spot = ig.clean_fix(job["fix"])["spots"][0]
        self.assertEqual(spot["photo"], hat)
        self.assertGreaterEqual(len(spot["outline"]), 4)
        self.assertEqual(job["fix"]["face_swap"], "")

    def test_fix_a_spot_can_end_with_a_face_swap_alone(self):
        import apps.image_studio.ui as ui_mod
        from unittest.mock import patch
        s, ui = self.tab()
        src = os.path.join(self.dir, "fixme.png")
        with open(src, "wb") as f:
            f.write(ig.oval_png(256))
        sent = []
        ui.host._spawn = lambda sid, fn, arg: sent.append(arg)
        self.addCleanup(lambda: delattr(ui.host, "_spawn"))
        fw = ui_mod.FixWindow(ui, src, dict(ig.default_settings(), scene="x"))
        fw._face_identity("sitter")
        self.assertIn("no reference picture", fw.msg.cget("text"))
        fw._face_identity("")
        fw._redraw()
        self.assertEqual(sent, [])                # nothing marked and no face
        fw._face_identity("sitter")
        fw._redraw()
        self.assertEqual(sent, [])                # missing reference is caught before queuing
        self.assertTrue(fw.win.winfo_exists())
        old_profiles = ui.studio.lib.all("identities")
        ui.studio.lib.save("identities", old_profiles + [
            {"id": "sitter", "name": "Sitter", "references": [src]}])
        try:
            with patch.object(ui_mod.ff, "available", return_value=True):
                fw._redraw()
        finally:
            ui.studio.lib.save("identities", old_profiles)
        (job,) = sent
        self.assertEqual(job["fix"]["face_swap"], "sitter")
        self.assertEqual(job["fix"]["spots"], [])

    def test_fix_a_spot_finds_in_one_click_and_locks(self):
        import apps.image_studio.ui as ui_mod
        s, ui = self.tab()
        src = os.path.join(self.dir, "fixme.png")
        with open(src, "wb") as f:
            f.write(PNG)
        sent = []
        ui.host._spawn = lambda sid, fn, *a: fn(*a) if not a else sent.append(a[0])
        self.addCleanup(lambda: delattr(ui.host, "_spawn"))
        ui._post = lambda what, fn: fn()
        self.addCleanup(lambda: delattr(ui, "_post"))
        ui.studio.find_parts = lambda path, kind: [{"x": 5, "y": 5, "size": 64,
                                                    "box": [1, 1, 8, 8]}]
        self.addCleanup(lambda: delattr(ui.studio, "find_parts"))
        fw = ui_mod.FixWindow(ui, src, dict(ig.default_settings(), scene="x"))
        self.pump(lambda: fw.img is not None)
        fw._find("face")
        self.assertEqual((len(fw.spots), fw.target), (1, "face"))
        fw._pick("mode", "lock")
        fw._find("hand")
        self.assertEqual(len(fw.locks), 1)
        fw._redraw()
        (job,) = sent
        self.assertEqual(job["fix"]["target"], "face")
        self.assertEqual(len(job["fix"]["locks"]), 1)
        self.assertEqual(job["fix"]["spots"][0]["box"], [1, 1, 8, 8])

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
        # A LoRA for another model never reaches the rows: it is set aside
        # and said, not applied and warned about (Add-ons).
        ui._add_lora("xl")
        self.assertEqual(ui.loras, [])
        self.assertIn("XL thing", ui.parked_note.cget("text"))
        self.assertNotIn("left out", ui.warn.cget("text"))
        ui.studio.lib.save("loras", [])
        ui._rebuild_choices()
        self.assertEqual(ui.parked, {})

    def test_only_loras_for_the_chosen_model_are_offered(self):
        s, ui = self.tab()
        lib = ui.studio.lib
        lib.save("loras", [
            {"id": "skin", "file": "skin.safetensors", "name": "Skin", "family": "z-image",
             "category": "Detail / Enhancement"},
            {"id": "grain", "file": "grain.safetensors", "name": "Grain", "family": "flux1",
             "category": "Style"},
            {"id": "edit", "file": "edit.safetensors", "name": "Edit", "family": "qwen-image"},
            {"id": "old", "file": "old.safetensors", "name": "Old", "family": "z-image",
             "enabled": False},
            {"id": "mystery", "file": "mystery.safetensors", "name": "Mystery"}])
        lib.save("presets", [{"id": "zmix", "name": "Z mix", "base": "standard",
                              "loras": [{"id": "skin", "strength": 0.5}]}])
        for r in list(ui.loras):
            ui._drop_lora(r)
        ui._set_model("z-image-turbo")
        ui._rebuild_choices()

        def offered():
            by_cat, unknown = ui.lora_menu_items()
            return ({c: [r["id"] for r in v] for c, v in by_cat.items()},
                    [r["id"] for r in unknown])
        self.assertEqual(offered(), ({"Detail / Enhancement": ["skin"]}, ["mystery"]))
        ui._add_lora("skin", 0.7)
        ui._add_lora("mystery", 0.3)
        # FLUX: the Z-Image row is set aside, the unknown one stays, the
        # Z-Image-only saved mix is not offered.
        ui._set_model("flux-dev")
        self.assertEqual(offered(), ({"Style": ["grain"]}, ["mystery"]))
        self.assertEqual([r["id"] for r in ui.loras], ["mystery"])
        self.assertEqual(ui.parked, {"skin": 0.7})
        self.assertTrue(ui.parked_note.winfo_manager())
        self.assertEqual(ui.collect()["loras"], [{"id": "mystery", "strength": 0.3}])
        # Back to Z-Image: it returns at its strength, and the note goes.
        ui._set_model("z-image-turbo")
        self.assertEqual(sorted((r["id"], r["var"].get()) for r in ui.loras),
                         [("mystery", 0.3), ("skin", 0.7)])
        self.assertEqual(ui.parked, {})
        self.assertFalse(ui.parked_note.winfo_manager())
        # Turned off in Add-ons: set aside too, back when turned on.
        lib.get("loras", "skin")["enabled"] = False
        ui._rebuild_choices()
        self.assertEqual([r["id"] for r in ui.loras], ["mystery"])
        lib.get("loras", "skin")["enabled"] = True
        ui._rebuild_choices()
        self.assertIn("skin", [r["id"] for r in ui.loras])
        for r in list(ui.loras):
            ui._drop_lora(r)
        lib.save("presets", [])
        lib.save("loras", [])
        ui._rebuild_choices()

    def test_saved_mixes_for_another_model_are_not_offered(self):
        s, ui = self.tab()
        lib = ui.studio.lib
        lib.save("loras", [{"id": "skin", "file": "skin.safetensors", "family": "z-image"}])
        lib.save("presets", [{"id": "zmix", "name": "Z mix", "base": "standard",
                              "loras": [{"id": "skin", "strength": 0.5}]}])
        seen = {}
        real = ui.choice

        def spy(parent, items, current, on_pick, **kw):
            if parent is ui.preset_row:
                seen["items"] = [v for v, _ in items if v]
            return real(parent, items, current, on_pick, **kw)
        ui.choice = spy
        try:
            ui._set_model("z-image-turbo")
            self.assertIn("zmix", seen["items"])
            ui._set_model("flux-dev")
            self.assertNotIn("zmix", seen["items"])
        finally:
            ui.choice = real
            lib.save("presets", [])
            lib.save("loras", [])
            ui._set_model("z-image-turbo")

    def test_addons_window_lists_toggles_and_installs(self):
        import apps.image_studio.ui as studio_images_ui
        from unittest.mock import patch
        s, ui = self.tab()
        lib = ui.studio.lib
        lib.save("loras", [
            {"id": "skin", "file": "skin.safetensors", "name": "Skin", "family": "z-image"},
            {"id": "grain", "file": "grain.safetensors", "name": "Grain", "family": "flux1"},
            {"id": "edit", "file": "edit.safetensors", "name": "Edit", "family": "sd15"},
            {"id": "mystery", "file": "mystery.safetensors", "name": "Mystery"}])
        ui.settings["model"] = "z-image-turbo"
        win = ui.open_addons()
        try:
            names = lambda: [k for k in win.pics]
            self.assertEqual(names(), ["skin", "mystery"])
            win.flip(lib.get("loras", "skin"), "enabled")
            self.assertFalse(ig.Library(lib.root).get("loras", "skin")["enabled"])
            self.assertIn("not offered", win.msg.cget("text"))
            win.flip(lib.get("loras", "skin"), "enabled")
            win.set_family(lib.get("loras", "mystery"), "flux1")
            self.assertEqual(names(), ["skin"])
            win.pick_model(win.OTHER)
            self.assertEqual(names(), ["edit"])
            win.pick_model("flux-dev")
            self.assertEqual(sorted(names()), ["grain", "mystery"])
            # Uninstall asks twice; a file only on another machine is turned off.
            rec = lib.get("loras", "grain")
            b = type("B", (), {"set": lambda self, **k: None,
                               "winfo_exists": lambda self: True})()
            win.uninstall(rec, b)
            self.assertIn("Recycle Bin", win.msg.cget("text"))
            self.assertEqual(win.armed, "grain")
            win.uninstall(rec, b)
            self.assertFalse(lib.get("loras", "grain")["enabled"])
            self.assertIn("turned off instead", win.msg.cget("text"))
            # The catalog: CivitAI's cards for the model, Install files it.
            card = {"model_id": 42, "version_id": 7, "name": "Film", "version": "v2",
                    "creator": "me", "base_model": "Flux.1 D", "family": "flux1",
                    "downloads": 1200, "category": "Style", "about": "grain",
                    "trigger": "film", "file": "film.safetensors", "size": 0,
                    "sha256": "ab" * 32, "preview_url": "",
                    "link": "https://civitai.com/models/42?modelVersionId=7"}
            calls = []

            def search(client, model, query="", sort="", cursor=""):
                calls.append((model["id"], query, sort, cursor))
                return [card], ""

            def install(lib_, client, c, folder, say=None, stop=None):
                return lib_.import_lora({"file": c["file"], "family": "flux1",
                                         "sha256": c["sha256"], "name": c["name"]})
            with patch.object(studio_images_ui.catalog, "search", side_effect=search), \
                    patch.object(studio_images_ui.catalog, "thumbnails", return_value={}), \
                    patch.object(studio_images_ui.catalog, "install", side_effect=install), \
                    patch.object(ui, "lora_folders", return_value=[("5090", "5090 - D:/l")]), \
                    patch.object(ui, "refresh_backends"):
                win.pick_tab("catalog")
                self.pump(lambda: win.cards)
                self.assertEqual(calls, [("flux-dev", "", "Most Downloaded", "")])
                self.assertEqual([c["version_id"] for c in win.cards], [7])
                _heading, row = win.box.winfo_children()   # its license group, then the card
                button = row.winfo_children()[1].winfo_children()[0]
                self.assertEqual(button.cget("text"), "Install")
                with patch.object(ui.studio, "backend",
                                  return_value={"id": "5090", "name": "5090",
                                                "lora_dir": "D:/l"}):
                    button.invoke()
                    self.pump(lambda: button.cget("text") == "Installed")
                self.assertIn("Installed Film", win.msg.cget("text"))
                self.assertIsNotNone(lib.lora_by_file("film.safetensors"))
                win.search()                           # shown as installed now
                self.pump(lambda: win.cards and not win.busy)
                self.assertEqual(calls[-1][0], "flux-dev")
        finally:
            win.close()
            lib.save("loras", [])
            self.pump(lambda: not self.app.q.qsize())

    def test_loras_saved_as_a_preset_come_back_when_it_is_picked(self):
        s, ui = self.tab()
        lib = ui.studio.lib
        lib.save("loras", [{"id": "a", "file": "a.safetensors", "name": "A"},
                           {"id": "b", "file": "b.safetensors", "name": "B"},
                           {"id": "c", "file": "c.safetensors", "name": "C"}])
        lib.save("presets", [])
        self.addCleanup(lambda: (lib.save("presets", []), ui._set_preset("standard")))
        ui._set_preset("identity")
        for r in list(ui.loras):
            ui._drop_lora(r)
        ui._add_lora("a", 0.5)
        ui._add_lora("b", 0.35)
        top = ui.save_preset()
        top.var.set("Real skin")
        top.ok()
        (mix,) = lib.all("presets")
        self.assertEqual(mix["base"], "identity")
        self.assertEqual(mix["loras"], [{"id": "a", "strength": 0.5},
                                        {"id": "b", "strength": 0.35}])
        self.assertEqual(ui.collect()["preset"], "real-skin")
        # A built-in takes the mix's rows away; a row added by hand stays.
        ui._add_lora("c", 1.0)
        ui._set_preset("standard")
        self.assertEqual([r["id"] for r in ui.loras], ["c"])
        ui._set_preset("real-skin")
        self.assertEqual([(r["id"], r["var"].get()) for r in ui.loras],
                         [("c", 1.0), ("a", 0.5), ("b", 0.35)])
        self.assertIn("A 0.5", ui.preset_about.cget("text"))
        self.assertTrue(ui.adv_open)
        # Saving under the same name replaces it; Delete keeps the rows.
        ui._drop_lora(ui.loras[0])
        top = ui.save_preset()
        self.assertEqual(top.var.get(), "Real skin")
        top.var.set("real SKIN")
        top.ok()
        self.assertEqual(len(lib.all("presets")), 1)
        ui.delete_preset()
        self.assertEqual(lib.all("presets"), [])
        self.assertEqual(ui.collect()["preset"], "identity")
        self.assertEqual([r["id"] for r in ui.loras], ["a", "b"])
        for r in list(ui.loras):
            ui._drop_lora(r)
        self.assertIsNone(ui.save_preset())              # nothing to save

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
        var.set("Partner")
        ed.widgets["trigger"][1].set("PARTNERPERSON")
        ed.widgets["strength"][1].set("0.8")
        ed._save()
        ed.win.destroy()
        rec = ui.studio.lib.get("identities", "partner")
        self.assertEqual((rec["trigger"], rec["strength"]), ("PARTNERPERSON", 0.8))
        self.assertIn("partner", ui.idents)                # the form picked it up
        with open(os.path.join(ui.studio.lib.root, "identities.json")) as f:
            self.assertIn("PARTNERPERSON", f.read())

    def test_profile_menu_selects_one_person_and_keeps_their_photo_list(self):
        s, ui = self.tab()
        ui.studio.lib.save("identities", [
            {"id": "one", "name": "One", "references": ["a.png", "b.png"],
             "avatar": "generated.png"},
            {"id": "two", "name": "Two", "references": ["c.png"]}])
        ui._rebuild_choices()
        ui._select_identity("one")
        self.assertTrue(ui.idents["one"][0].get())
        self.assertIn("2 reference photos", ui.identity_note.cget("text"))
        ui._set_model("withanyone")
        self.assertIn("WithAnyone uses the first photo", ui.identity_note.cget("text"))
        ui._set_model("z-image-turbo")
        self.assertNotIn("WithAnyone uses the first photo", ui.identity_note.cget("text"))
        ui._select_identity("two")
        self.assertFalse(ui.idents["one"][0].get())
        self.assertTrue(ui.idents["two"][0].get())
        ui._rebuild_choices()
        self.assertTrue(ui.idents["two"][0].get())
        ed = ui.edit_identities()
        self.assertEqual(ed.widgets["references"][1]["paths"], ["a.png", "b.png"])
        self.assertEqual(ed.widgets["avatar"][1].get(), "generated.png")
        ed.win.destroy()
        ui._select_identity("")
        self.assertFalse(any(v[0].get() for v in ui.idents.values()))

    def test_the_people_tab_has_one_person_dropdown_and_none_of_the_creators_look(self):
        s, ui = self.tab()
        ui.studio.lib.save("identities", [
            {"id": "gav", "name": "Gav", "references": ["a.png"]},
            {"id": "two", "name": "Two", "references": []}])
        ui.studio.lib.save("characters", [
            {"id": "mara", "name": "Mara", "identity": "gav", "looks": {"hair": "auburn"}},
            {"id": "bare", "name": "Bare", "identity": "", "looks": {}}])
        ui._rebuild_choices()
        texts = lambda box: [w.cget("text") for w in box.winfo_children()
                             if hasattr(w, "paint")]
        self.assertEqual(texts(ui.person_box), ["No one  ▾"])     # the one dropdown
        buttons = [t for w in ui.pc_box.winfo_children() for t in texts(w)]
        self.assertIn("Editor", buttons)
        self.assertIn("Image references", buttons)
        self.assertNotIn("Creator…", buttons)
        ticked = lambda: [i for i, v in ui.idents.items() if v[0].get()]
        ui._pick_from_people("c:mara")               # a character brings its face
        self.assertEqual((ui.settings["character"], ticked()), ("mara", ["gav"]))
        self.assertEqual(ui.person_pill.cget("text"), "Mara  ▾")
        self.assertEqual(ui.text["hair"].get(), "auburn")
        # Saved in the creator, the form's character's hidden look follows it.
        ui.studio.lib.save("characters", [
            {"id": "mara", "name": "Mara", "identity": "gav",
             "looks": {"hair": "black", "facial_hair": "heavy stubble"}},
            {"id": "bare", "name": "Bare", "identity": "", "looks": {}}])
        ui.text["expression"].set("laughing")
        ui._saved("characters")
        self.assertEqual((ui.text["hair"].get(), ui.text["facial_hair"].get()),
                         ("black", "heavy stubble"))
        self.assertEqual(ui.text["expression"].get(), "laughing")     # the picture's, kept
        ui._pick_from_people("i:two")                # a profile alone is not Mara
        self.assertEqual((ui.settings["character"], ticked()), ("", ["two"]))
        self.assertEqual(ui.person_pill.cget("text"), "Two  ▾")
        self.assertEqual((ui.text["hair"].get(), ui.text["facial_hair"].get()), ("", ""))
        ui._pick_from_people("c:bare")               # no face of its own: none
        self.assertEqual((ui.settings["character"], ticked()), ("bare", []))
        self.assertEqual(ui.person_pill.cget("text"), "Bare  ▾")
        ui._pick_from_people("")
        self.assertEqual((ui.settings["character"], ticked()), ("", []))
        self.assertEqual(ui.person_pill.cget("text"), "No one  ▾")
        tabs = texts(ui.look_tabs)
        # Who the person is is the creator's alone (the user, 2026-10-02).
        for hidden in ("Body", "Face", "Hair", "Accessories"):
            self.assertNotIn(hidden, tabs)
        self.assertEqual(tabs, ["Expression", "Clothes"])
        ui._show_looks("Face")                       # hidden: the first shown instead
        self.assertEqual(ui.look_section, "Expression")

    def test_a_character_tag_is_a_word_and_an_uploaded_picture(self):
        s, ui = self.tab()
        import tkinter as tk
        from unittest import mock
        import apps.image_studio.ui as siu
        pic = os.path.join(tempfile.mkdtemp(), "glasses.png")
        tk.PhotoImage(master=self.app, width=4, height=4).write(pic, format="png")
        ed = ui.edit_characters()
        ed._new()
        ed.name.set("Ada")
        ed._show("Tags")
        self.app.update()
        ed._add_tag()                                 # no word: nothing asked
        self.assertIn("Name the tag first", ed.msg.cget("text"))
        ed.tag_name.set("glasses")
        with mock.patch.object(siu.filedialog, "askopenfilename", return_value=""):
            ed._add_tag()                             # no picture: no tag
        self.assertEqual(ed.item_refs, {})
        with mock.patch.object(siu.filedialog, "askopenfilename", return_value=pic):
            ed._add_tag()
        kept = ed.item_refs["glasses"]
        self.assertNotEqual(kept, pic)                # copied into the library
        self.assertTrue(os.path.isfile(kept))
        self.assertEqual(ed.tag_name.get(), "")
        ed.tag_name.set("Glasses")                    # the same tag, a new picture
        with mock.patch.object(siu.filedialog, "askopenfilename", return_value=pic):
            ed._add_tag()
        self.assertEqual(list(ed.item_refs), ["glasses"])
        ed._randomize()                               # from Tags: every look tab
        self.assertTrue(ed._save())
        ed._new()                                     # another takes it from the library
        ed.name.set("Bea")
        ed._show("Tags")
        self.app.update()
        self.assertIn(("glasses", kept, "Ada"), ed._library_tags())
        ed._take_tag("glasses", kept)
        self.assertEqual(ed.item_refs, {"glasses": kept})
        self.assertNotIn("glasses", [n for n, _, _ in ed._library_tags()])
        ed._use()
        ed.win.destroy()
        self.assertEqual(ui.studio.lib.get("characters", "bea")["item_refs"],
                         {"glasses": kept})
        ui.scene.delete("1.0", "end")
        ui.scene.insert("1.0", "Bea puts her glasses on.")
        self.assertEqual(ig.outfit_of(ui.collect())["accessories"],
                         [{"name": "glasses", "path": kept}])

    def test_pictures_of_items_and_people_can_come_from_links(self):
        """The user: "i want to use urls for images of items and people". A tag,
        a form item and a profile photo each take a link: asked in a small
        window (the clipboard's link already in it), downloaded off the UI
        thread, kept under references/ as an upload is."""
        s, ui = self.tab()
        from unittest import mock
        asked = []

        def fetch(url, opener=None):
            asked.append(url)
            if "broken" in url:
                raise ig.LinkError("example.com answered HTTP 404 (Not Found).")
            return PNG, ".png"
        patched = mock.patch.object(ig, "fetch_picture", fetch)
        patched.start()
        self.addCleanup(patched.stop)

        ed = ui.edit_characters()
        ed._new()
        ed.name.set("Cy")
        ed._show("Tags")
        self.app.update()
        self.assertIsNone(ed._add_tag_link())          # no word: no link asked
        self.assertIn("Name the tag first", ed.msg.cget("text"))
        ed.tag_name.set("earrings")
        self.app.clipboard_clear()
        self.app.clipboard_append("https://shop.example/hoops.jpg")
        top = ed._add_tag_link()
        self.assertEqual(top.var.get(), "https://shop.example/hoops.jpg")
        top.var.set("https://example.com/broken.png")
        top.ok()
        self.pump(lambda: "No picture from that link" in ed.msg.cget("text"))
        self.assertEqual(ed.item_refs, {})              # no picture, no tag
        top = ed._add_tag_link()
        top.var.set("https://shop.example/hoops.jpg")
        top.ok()
        self.pump(lambda: "earrings" in ed.item_refs)
        kept = ed.item_refs["earrings"]
        self.assertTrue(kept.startswith(os.path.join(ui.studio.lib.root, "references")))
        self.assertTrue(os.path.isfile(kept))
        self.assertEqual(ed.tag_name.get(), "")
        self.assertIn("Tagged earrings", ed.msg.cget("text"))
        ed.win.destroy()

        ui._show_looks("Clothes")                       # an item on the form
        ui.text["top"].set("denim jacket")
        ui._show_looks("Clothes")
        self.app.update()
        top = ui._link_item("denim jacket")
        top.var.set("https://shop.example/jacket")
        top.ok()
        self.pump(lambda: "denim jacket" in ui.item_refs)
        self.assertTrue(os.path.isfile(ui.item_refs["denim jacket"]))

        rd = ui.edit_identities()                       # a person's photo
        rd._new()
        self.app.update()
        pics = rd.widgets["references"][1]
        top = rd._add_link(pics)
        top.var.set("https://example.com/me.jpg")
        top.ok()
        self.pump(lambda: len(pics["paths"]) == 1)
        self.assertTrue(pics["paths"][0].startswith(
            os.path.join(ui.studio.lib.root, "references")))
        self.assertEqual(asked[-1], "https://example.com/me.jpg")
        rd.win.destroy()

    def test_photos_dropped_on_the_identity_window_land_on_one_pad(self):
        """The user: "make the pictures for reference images of someone drag
        and drop into the identity window. turn the add photos, add folder,
        and add from link into a singular landing pad that spans the width of
        the identity description". The window takes drops (`core.filedrop`;
        its OLE side is tests/test_filedrop.py): the pad lights while a drag
        is over, files and a folder's pictures import, a link downloads."""
        from unittest import mock
        import tkinter as tk
        import core.filedrop as filedrop
        s, ui = self.tab()
        patched = [mock.patch.object(ig, "fetch_picture", lambda url, opener=None: (PNG, ".png")),
                   mock.patch("apps.image_studio.faces.crop_for_import",
                              lambda paths, *a, **k: (list(paths), ""))]
        for p in patched:
            p.start()
            self.addCleanup(p.stop)
        ui.studio.lib.save("identities", [])
        ed = ui.edit_identities()
        self.addCleanup(lambda: ed.win.winfo_exists() and ed.win.destroy())
        ed._dropped([os.path.join(self.dir, "x.png")], None)       # no one to add to
        self.assertIn("Make or choose a person first", ed.msg.cget("text"))
        ed._new()
        self.app.update()
        if filedrop.WINDOWS:
            self.assertIsNotNone(ed.drops, "the window takes no drops")
        pad, title = ed._pad
        texts, stack = [], [ed.form]
        while stack:
            w = stack.pop()
            stack.extend(w.winfo_children())
            try:
                texts.append(w.cget("text"))
            except tk.TclError:
                pass
        for gone in ("Add photos…", "Add folder…", "Add from link…"):
            self.assertNotIn(gone, texts)                  # one pad instead
        for link in ("Choose photos…", "Choose a folder…", "Paste a link…", "Remove"):
            self.assertIn(link, texts)
        description = ed.widgets["description"][1]
        self.assertEqual((pad.winfo_x(), pad.winfo_width()),
                         (description.winfo_x(), description.winfo_width()))
        self.assertEqual(title.cget("text"), "Drop photos or a folder here"
                         if ed.drops else "Click to choose photos")

        ed.drops.on_enter() if ed.drops else ed._light_pad(True)     # a drag comes over
        self.assertEqual(title.cget("text"), "Let go to add them")
        self.assertEqual(pad.cget("highlightbackground"), self.app.C["accent"])
        ed._light_pad(False)                                          # and goes
        self.assertEqual(pad.cget("highlightbackground"), self.app.C["border"])

        drop = tempfile.mkdtemp(dir=self.dir)
        folder = os.path.join(drop, "more")
        os.mkdir(folder)
        for i, name in enumerate(("one.png", os.path.join("more", "a.png"),
                                  os.path.join("more", "b.jpg"),
                                  os.path.join("more", "notes.txt"))):
            with open(os.path.join(drop, name), "wb") as f:
                f.write(PNG[:-1] + bytes([i]) if name.endswith("g") else b"words")
        pics = ed.widgets["references"][1]
        ed._dropped([os.path.join(drop, "one.png"), folder], None)
        self.pump(lambda: not pics.get("importing"))
        self.assertEqual(len(pics["paths"]), 3, ed.msg.cget("text"))   # not the .txt
        ed._dropped([], "https://example.com/her.png")                  # from a web page
        self.pump(lambda: len(pics["paths"]) == 4)
        self.assertTrue(all(p.startswith(os.path.join(ui.studio.lib.root, "references"))
                            for p in pics["paths"]))

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
        # The faces alone: no expression field or menu, no Looking row.
        shown, todo = [], list(faces)
        while todo:
            w = todo.pop()
            todo += w.winfo_children()
            shown.append(w)
        self.assertFalse([w for w in shown if w.winfo_class() == "Entry"])
        self.assertNotIn("Looking", [w.cget("text") for w in shown
                                     if w.winfo_class() == "Label"])
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

    def test_model_source_open_fetches_reviewable_results_off_thread(self):
        import apps.image_studio.ui as studio_images_ui
        from unittest.mock import patch
        _, ui = self.tab()
        result = {"checked": time.time(), "errors": [], "items": [
            {"title": "Example", "url": "https://civitai.com/models/123", "kind": "LORA",
             "why": "Image style", "details": "Review compatibility", "importable": True},
            {"title": "ComfyUI", "url": "https://github.com/Comfy-Org/ComfyUI/releases/latest",
             "kind": "Code / library", "why": "Backend changes", "details": "Release notes",
             "importable": False}]}
        threads = []
        def fetch(*args, **kwargs):
            threads.append(threading.current_thread())
            return result
        with patch.object(studio_images_ui.discovery, "discover", side_effect=fetch):
            dlg = ui.open_model_source("civitai")
            try:
                self.pump(lambda: not dlg.busy)
                self.assertEqual(len(dlg.results.winfo_children()), 2 + len(studio_images_ui.addons.CATALOG))
                self.assertTrue(all(t is not threading.main_thread() for t in threads))
                dlg.keep_link(result["items"][0])
                self.assertIn(result["items"][0]["url"], dlg.links.get("1.0", "end"))
                dlg.refresh_discoveries(force=True)
                dlg.win.destroy()
                self.pump(lambda: not self.app.q.qsize())
            finally:
                if dlg.win.winfo_exists():
                    dlg.win.destroy()

    def test_supported_addon_button_installs_into_selected_comfy_folder(self):
        import apps.image_studio.ui as studio_images_ui
        from unittest.mock import patch
        _, ui = self.tab()
        with tempfile.TemporaryDirectory() as root:
            for name in ("main.py", "folder_paths.py"):
                with open(os.path.join(root, name), "w") as stream:
                    stream.write("# fake backend\n")
            dlg = studio_images_ui.ModelSourceSettings(ui, "huggingface")
            try:
                card = dlg.results.winfo_children()[0]
                actions = card.winfo_children()[-1]
                button = actions.winfo_children()[0]
                self.assertEqual(button.cget("text"), "Add to app")
                with patch.object(studio_images_ui.filedialog, "askdirectory", return_value=root):
                    button.invoke()
                    self.pump(lambda: button.cget("text") == "Installed")
                target = os.path.join(root, "custom_nodes", "studio_matchtone", "__init__.py")
                self.assertTrue(os.path.isfile(target))
                self.assertIn("Restart ComfyUI", dlg.message.cget("text"))
            finally:
                dlg.win.destroy()

    def test_model_source_buttons_save_and_reopen(self):
        import apps.image_studio.ui as studio_images_ui
        from unittest.mock import patch
        _, ui = self.tab()
        with patch.dict(os.environ, {}, clear=True):
            for source, (_, domain, _) in studio_images_ui.model_sources.SOURCES.items():
                self.assertTrue(ui.source_buttons[source].winfo_ismapped())
                dlg = studio_images_ui.ModelSourceSettings(ui, source)
                try:
                    link = "https://%s/models/example" % domain
                    dlg.links.delete("1.0", "end")
                    dlg.links.insert("1.0", link)
                    dlg.token.set("test-key")
                    self.assertTrue(dlg.save())
                finally:
                    dlg.win.destroy()
                dlg = studio_images_ui.ModelSourceSettings(ui, source)
                try:
                    self.app.update()
                    self.assertEqual(dlg.links.get("1.0", "end").strip(), link)
                    self.assertEqual(dlg.token.get(), "test-key")
                    self.assertTrue(dlg.message.winfo_ismapped())
                finally:
                    dlg.win.destroy()

    def test_a_civitai_link_imports_into_the_lora_library(self):
        import urllib.request
        import apps.image_studio.ui as studio_images_ui
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

    def test_the_face_photos_are_the_identitys_and_the_form_shows_no_strip(self):
        """The user, 2026-09-27: face photos are managed in the identity builder
        only; the form keeps its one person dropdown. Picking a character or a
        profile sends that identity's reference photos with the picture."""
        import tkinter as tk
        s, ui = self.tab()
        folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, folder, True)
        pics = [os.path.join(folder, n) for n in ("front.png", "left.png", "other.png")]
        for pic in pics:
            tk.PhotoImage(master=self.app, width=4, height=4).write(pic, format="png")
        ui.studio.lib.save("identities", [
            {"id": "lil", "name": "Partner", "references": pics[:2]},
            {"id": "ann", "name": "Ann", "references": pics[2:]}])
        ui.studio.lib.save("characters", [
            {"id": "dirndl", "name": "Lil in a dirndl", "identity": "lil",
             "looks": {}, "item_refs": {}}])
        ui._rebuild_choices()
        for gone in ("face_box", "face_on", "_show_face"):
            self.assertFalse(hasattr(ui, gone), gone)
        ui._pick_from_people("c:dirndl")                       # a character
        got = ui.collect()
        self.assertEqual((got["face_photos"], got["face_name"]), (pics[:2], "Lil in a dirndl"))
        ui._pick_from_people("i:ann")                          # a profile alone
        self.assertEqual((ui.collect()["face_photos"], ui.collect()["face_name"]),
                         (pics[2:], "Ann"))
        ui._pick_from_people("")                               # no one
        self.assertEqual(ui.collect()["face_photos"], [])
        ui.apply(got)                                          # Reuse Settings
        self.assertEqual(ui.collect()["face_photos"], pics[:2])
        ed = ui.edit_characters()
        for gone in ("_add_faces", "_face_photos", "face_row"):
            self.assertFalse(hasattr(ed, gone), gone)
        ed.win.destroy()

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
        ui.scene.insert("1.0", "A man waving from a boat")
        n = len(ui.jobs)
        ui.generate()
        self.pump(lambda: len(ui.jobs) > n and ui.jobs[0].status == "complete")
        w = ui.rows[ui.jobs[0].id]
        self.assertEqual([l.cget("text") for l in w["stages"]],
                         ["Sampling", "Decoding", "Hand pass", "Complete"])
        # The hands pass is for people: a fox's paws are hands to SAM3, and
        # were redrawn as a hand's (live, 2026-09-29).
        ui.scene.delete("1.0", "end")
        ui.scene.insert("1.0", "A fox")
        n = len(ui.jobs)
        ui.generate()
        self.pump(lambda: len(ui.jobs) > n and ui.jobs[0].status == "complete")
        w = ui.rows[ui.jobs[0].id]
        self.assertEqual([l.cget("text") for l in w["stages"]],
                         ["Sampling", "Decoding", "Complete"])

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

    def test_what_they_wear_is_added_on_the_form_by_picture_and_name(self):
        s, ui = self.tab()
        pic = os.path.join(self.dir, "IMG_1234.png")
        with open(pic, "wb") as f:
            f.write(PNG)
        top = ui._name_wearing(pic)
        self.assertEqual(top.var.get(), "")          # a camera's file name is no name
        top.var.set("red plaid shirt")
        top.ok()
        self.app.update()
        worn = ui.collect()["wearing"]
        self.assertEqual([w["name"] for w in worn], ["red plaid shirt"])
        self.assertNotEqual(worn[0]["path"], pic)    # copied, so history keeps it
        self.assertTrue(os.path.isfile(worn[0]["path"]))
        texts = [w.cget("text") for row in ui.wear_box.winfo_children()
                 for w in row.winfo_children() if type(w).__name__ == "Label"]
        self.assertIn("red plaid shirt", texts)
        # The same name again replaces it; a picture's own name is offered.
        hat = os.path.join(self.dir, "bucket_hat.png")
        with open(hat, "wb") as f:
            f.write(PNG)
        top = ui._name_wearing(hat)
        self.assertEqual(top.var.get(), "bucket hat")
        top.ok()
        self.assertEqual([w["name"] for w in ui.collect()["wearing"]],
                         ["red plaid shirt", "bucket hat"])
        # Generate Again brings it back; × takes it off.
        settings = ui.collect()
        ui._drop_wearing(0)
        self.assertEqual([w["name"] for w in ui.collect()["wearing"]], ["bucket hat"])
        ui.reuse(settings)
        self.assertEqual([w["name"] for w in ui.collect()["wearing"]],
                         ["red plaid shirt", "bucket hat"])


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
                if "GetImageSize" in str(self.graphs[int(pid) - 1]):   # a people_graph
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
            # Cropped first: SAM3 looks in the crop, the cut lands in the photo.
            found = s.find_people(__file__, {"x": 300, "y": 40, "width": 100, "height": 100})
            g = s.client(b).graphs[2]
            self.assertEqual(g["4"]["inputs"]["image"], ["c", 0])
            self.assertEqual(g["9"]["inputs"]["images"], ["c", 0])
            s.cut_person(found, found["boxes"][0])
            region = s.client(b).graphs[3]["2"]["inputs"]["crop_region"]
            self.assertEqual(region, {"x": 302, "y": 42, "width": 56, "height": 96})
            # SAM3 is aimed at the picked person, in the cut's own pixels.
            self.assertEqual(s.client(b).graphs[3]["5"]["inputs"]["bboxes"],
                             {"x": 3, "y": 3, "width": 50, "height": 90})
            # look_at: the size and preview, no SAM3 run.
            photo = s.look_at(__file__)
            self.assertNotIn("4", s.client(b).graphs[4])
            self.assertEqual((photo["width"], photo["height"], photo["preview"]),
                             (100, 100, b"pv.png"))

    def test_crop_region_orders_clamps_and_drops_slips(self):
        self.assertEqual(ig.crop_region(300, 250, 100, -20, 400, 300),
                         {"x": 100, "y": 0, "width": 200, "height": 250})
        self.assertIsNone(ig.crop_region(10, 10, 30, 200, 400, 300))     # too thin
        self.assertIsNone(ig.crop_region(-5, -5, 500, 400, 400, 300))    # the whole photo

    def test_people_graph_without_sam3_only_sizes_and_previews(self):
        g = ig.people_graph("a.png", None)
        self.assertFalse({"2", "3", "4", "5", "c"} & set(g))
        self.assertEqual(g["9"]["inputs"]["images"], ["1", 0])
        self.assertEqual(ig.people_found({"outputs": {
            "7": {"text": ["10"]}, "8": {"text": ["20"]}}})[:3], (10, 20, []))
