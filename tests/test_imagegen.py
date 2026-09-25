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
                                "t5xxl_fp8_e4m3fn.safetensors", "qwen_3_4b.safetensors"},
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
        on_event("executing", "40")
        for i in range(1, 5):
            on_event("progress", (i, 4, "40"))
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
            for extra in ({}, {"refine": True, "source_image": "s.png"}):
                g = ig.fill(wf, dict(vals, **extra), [("l.safetensors", 0.5)])
                self.assertTrue(any(n["class_type"] == "SaveImage" for n in g.values()),
                                wf["id"])


class TestCompose(TempStudioMixin, unittest.TestCase):
    def plan(self, backend="5090", inventory=FLUX_FILES, **kw):
        s = ig.default_settings()
        s.update(kw)
        return ig.compose(s, self.studio.lib, self.backend(backend), inventory)

    def test_person_style_scene(self):
        p = self.plan(identities=[{"id": "gavin", "strength": 0.9}], style="sx70-authentic",
                      scene="At Munich Oktoberfest, raising a stein.", seed=3)
        self.assertEqual(p.errors, [])
        self.assertTrue(p.prompt.startswith("GAVINPERSON. At Munich Oktoberfest"))
        self.assertIn("SX-70 instant film", p.prompt)
        self.assertEqual(p.loras, [("gavin.safetensors", 0.9), ("sx70.safetensors", 0.55)])
        self.assertEqual(p.values["seed"], 3)

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
        self.assertTrue(any("flux1-dev.safetensors" in e and "Models" in e for e in p.errors))

    def test_per_backend_filenames(self):
        p = self.plan(backend="3090", scene="x")
        self.assertEqual(p.values["t5"], "t5xxl_fp8_e4m3fn.safetensors")
        self.assertEqual(p.values["weight_dtype"], "fp8_e4m3fn")
        self.assertEqual(p.values["encoder_device"], "cpu")
        self.assertEqual(self.plan(scene="x").values["t5"], "t5xxl_fp16.safetensors")

    def test_per_backend_family_decides_compatibility(self):
        models = self.studio.lib.all("models")
        models[0]["backends"]["5090"] = {"model": "flux2-dev.safetensors", "family": "flux2"}
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


class TestJobs(TempStudioMixin, unittest.TestCase):
    def test_a_job_runs_and_lands_in_history(self):
        room = []
        self.studio.make_room = room.append
        src = os.path.join(self.dir, "src.png")
        with open(src, "wb") as f:
            f.write(PNG)
        jobs = self.studio.submit(dict(ig.default_settings(), scene="A fox", backend="3090",
                                       identities=["gavin"], references={"source": src},
                                       seed=42))
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

    def test_generate_again_keeps_a_chosen_seed_and_rerolls_a_random_one(self):
        self.assertEqual(ig.again({"seed": 5, "seed_mode": "fixed"})["seed"], 5)
        self.assertEqual(ig.again({"seed": 5, "seed_mode": "random"})["seed"], -1)

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
        ui.generate()
        self.pump(lambda: ui.jobs and ui.jobs[0].status in ig.FINISHED)
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

    def test_a_settled_tab_animates_nothing(self):
        s, ui = self.tab()
        self.pump(lambda: all(j.status in ig.FINISHED for j in ui.jobs))
        self.app.update()
        self.assertFalse([k for k in self.app.anim if k[0] == "images-clock"])


if __name__ == "__main__":
    unittest.main()
