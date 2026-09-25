"""The ComfyUI bridge, against a fake ComfyUI.

No network: `urllib.request.urlopen` is replaced with a tiny in-memory ComfyUI
that answers the routes the bridge uses. What these prove is the contract the
registry entry and the prompt rely on - the graph comfy_generate builds, that
outputs land on this machine as files and as inline images, that ComfyUI's
error shapes reach the model as prose, and that the stdio server speaks what
studio_agent.MCPClient expects.
"""
import base64
import io
import json
import os
import shutil
import tempfile
import unittest
import urllib.error
import urllib.request
import zlib

import studio_agent as eng
import studio_comfy_mcp as comfy
import studio_tasks as tasks


def tiny_png():
    """A 1x1 grey PNG, so 'image' content is a real image."""
    def chunk(kind, data):
        c = kind + data
        return len(data).to_bytes(4, "big") + c + zlib.crc32(c).to_bytes(4, "big")
    ihdr = (1).to_bytes(4, "big") + (1).to_bytes(4, "big") + b"\x08\x00\x00\x00\x00"
    idat = zlib.compress(b"\x00\x80")
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat)
            + chunk(b"IEND", b""))


PNG = tiny_png()


class FakeComfy:
    """Just enough of ComfyUI's HTTP API, recording what it was asked."""

    def __init__(self):
        self.prompts = {}          # prompt_id -> graph
        self.history = {}          # prompt_id -> history entry
        self.uploads = []
        self.posts = []
        self.reject_next = None    # a (code, body) to answer the next /prompt with
        self.models_route = True   # older servers have no /models/<kind>
        self.checkpoints = ["sd_xl_base_1.0.safetensors", "dreamshaper_8.safetensors"]
        self.diffusion_models = ["z_image_turbo_bf16.safetensors"]
        self.text_encoders = ["qwen_3_4b.safetensors"]
        self.loras = ["detail.safetensors"]
        self.finish_after = 0      # history polls before a prompt "completes"
        self._polls = {}
        # What /queue says is running; someone else's run by default, which
        # is also what keeps a finished run from freeing the GPU under it.
        self.running = [[1, "run-1", {"5": {"class_type": "KSampler"}}]]
        self.free_vram = [20e9]    # /system_stats answers these in turn, the last for good
        self.faces = {}            # uploaded name -> SAM3's boxes, for PreviewAny to report
        self.size = (1254, 1254)   # what GetImageSize says of any picture
        self.shares = {}           # PreviewAny node id -> share of the 16x16 mask it prints
        self.extra = {}            # prompt_id -> the extra_data it was queued with

    def __call__(self, req, timeout=None):
        url = req if isinstance(req, str) else req.full_url
        method = "GET" if isinstance(req, str) else req.get_method()
        data = None if isinstance(req, str) else req.data
        path = url[len(comfy.COMFY_URL):]
        route, _, query = path.partition("?")
        q = dict(p.split("=", 1) for p in query.split("&") if p)
        body = self.route(method, route, q, data)
        if isinstance(body, bytes):
            return io.BytesIO(body)
        return io.BytesIO(json.dumps(body).encode("utf-8"))

    def node_outputs(self, graph):
        """What PreviewAny shows of SAM3's boxes and a picture's size, and a
        temp file for every other PreviewImage in the graph."""
        out = {}
        for nid, node in graph.items():
            if node["class_type"] == "PreviewImage" and nid != "7":
                out[nid] = {"images": [{"filename": "edit%s.png" % nid, "subfolder": "",
                                        "type": "temp"}]}
            if node["class_type"] != "PreviewAny":
                continue
            src, index = node["inputs"]["source"]
            if graph[src]["class_type"] == "GetImageSize":
                value = self.size[index]
            elif graph[src]["class_type"] == "ImageToMask":
                # torch prints a mask as tensor([[[0.1000, ...]]]).
                v = self.shares.get(nid, 0.1)
                out[nid] = {"text": ["tensor([[[%s]]])" % ", ".join(["%.4f" % v] * 256)]}
                continue
            else:
                # An uploaded picture by its name; a picture the graph made
                # (comfy_generate's, for the face pass) as "generated".
                pic = graph[graph[src]["inputs"]["image"][0]]
                image = (pic["inputs"]["image"] if pic["class_type"] == "LoadImage"
                         else "generated")
                value = [[{"x": x, "y": y, "width": w, "height": h, "score": 0.8}
                          for x, y, w, h in self.faces.get(image, [])]]
            out[nid] = {"text": [json.dumps(value)]}
        return out

    def fail(self, code, body):
        raise urllib.error.HTTPError("x", code, "err", {}, io.BytesIO(json.dumps(body).encode()))

    def route(self, method, route, q, data):
        if route == "/system_stats":
            free = self.free_vram.pop(0) if len(self.free_vram) > 1 else self.free_vram[0]
            return {"system": {"comfyui_version": "0.3.0"},
                    "devices": [{"name": "cuda:0 RTX 4090", "vram_total": 24e9, "vram_free": free}]}
        if route == "/queue" and method == "GET":
            return {"queue_running": self.running, "queue_pending": []}
        if route == "/queue":
            self.posts.append(("queue", json.loads(data)))
            return {}
        if route == "/free":
            self.posts.append(("free", json.loads(data)))
            return b""
        if route == "/interrupt":
            self.posts.append(("interrupt", None))
            return b""
        if route.startswith("/models/"):
            if not self.models_route:
                self.fail(404, "not found")
            return {"checkpoints": self.checkpoints,
                    "loras": self.loras,
                    "diffusion_models": self.diffusion_models,
                    "text_encoders": self.text_encoders,
                    "vae": ["ae.safetensors", "qwen_image_vae.safetensors"]}.get(route[8:], [])
        if route == "/object_info/CheckpointLoaderSimple":
            return {"CheckpointLoaderSimple": {"input": {"required": {
                "ckpt_name": [["fallback.safetensors"], {}]}}}}
        if route == "/object_info/KSampler":
            return {"KSampler": {"category": "sampling", "input": {"required": {
                "seed": ["INT", {"default": 0, "min": 0}],
                "sampler_name": [comfy.SAMPLERS, {}],
                "model": ["MODEL", {}]}}, "output": ["LATENT"], "output_name": ["LATENT"]}}
        if route == "/object_info/Nope":
            return {}
        if route == "/object_info":
            return {"KSampler": {"category": "sampling"}, "LoadImage": {"category": "image"},
                    "ImageUpscaleWithModel": {"category": "image/upscaling"}}
        if route == "/prompt":
            if self.reject_next:
                code, body = self.reject_next
                self.reject_next = None
                self.fail(code, body)
            graph = json.loads(data)["prompt"]
            pid = "p%d" % (len(self.prompts) + 1)
            self.prompts[pid] = graph
            self.extra[pid] = json.loads(data).get("extra_data")
            outputs = {"7": {"images": [
                {"filename": "StudioAssistant_00001_.png", "subfolder": "", "type": "output"},
                {"filename": "preview.png", "subfolder": "", "type": "temp"}]}}
            outputs.update(self.node_outputs(graph))
            self.history[pid] = {
                "prompt": [], "outputs": outputs,
                "status": {"status_str": "success", "completed": True, "messages": []}}
            return {"prompt_id": pid, "number": 1, "node_errors": {}}
        if route.startswith("/history/"):
            pid = route[9:]
            n = self._polls[pid] = self._polls.get(pid, 0) + 1
            if pid not in self.history or n <= self.finish_after:
                return {}
            return {pid: self.history[pid]}
        if route.startswith("/history"):
            return self.history
        if route == "/view":
            return PNG if q.get("filename", "").endswith(".png") else b"not png"
        if route == "/upload/image":
            self.uploads.append(data)
            return {"name": "sketch.png", "subfolder": "", "type": "input"}
        self.fail(404, {"error": "no route " + route})


class ComfyBridgeTest(unittest.TestCase):
    def setUp(self):
        self.fake = FakeComfy()
        self._real_open = urllib.request.urlopen
        urllib.request.urlopen = self.fake
        self.tmp = tempfile.mkdtemp()
        self._real_out = comfy.OUTPUT_DIR
        comfy.OUTPUT_DIR = self.tmp
        self._real_keep = comfy.KEEP_MODELS
        comfy.KEEP_MODELS = False         # the default, whatever this shell says

    def tearDown(self):
        urllib.request.urlopen = self._real_open
        comfy.OUTPUT_DIR = self._real_out
        comfy.KEEP_MODELS = self._real_keep
        shutil.rmtree(self.tmp, ignore_errors=True)

    def text(self, res):
        return res["content"][0]["text"]

    # ----------------------------------------------------------- the contract

    def test_tool_list_is_the_registry_entry_and_survives_sanitizing(self):
        names = {t["name"] for t in comfy.tool_list()}
        app = eng.APPS_BY_ID["comfyui"]
        everything = {t for names_ in app.groups.values() for t in names_}
        self.assertEqual(names, everything, "registry groups and bridge tools disagree")
        for t in comfy.tool_list():
            self.assertTrue(t["description"].strip())
            eng.sanitize_schema(t["inputSchema"])
            # The executor validates against the original schema; every tool's
            # schema has to be one the validator understands with no arguments.
            self.assertEqual(t["inputSchema"]["type"], "object")

    def test_read_only_tools_are_annotated_and_the_rest_are_not(self):
        """The executor decides from readOnlyHint which calls need a read-back.
        Unannotated, every comfy_ tool counted as an edit, so asking which
        models are installed left the model nagged to "inspect" its work."""
        hints = {t["name"]: t["annotations"]["readOnlyHint"] for t in comfy.tool_list()}
        for name in ("comfy_status", "comfy_list_models", "comfy_queue", "comfy_history",
                     "comfy_wait", "comfy_fetch_output"):
            self.assertTrue(hints[name], name)
            self.assertTrue(tasks.readonly(name, {}, {"annotations": {"readOnlyHint": True}}))
        for name in ("comfy_generate", "comfy_run_workflow", "comfy_upload_image",
                     "comfy_interrupt", "comfy_clear_queue"):
            self.assertFalse(hints[name], name)

    def test_a_generation_that_returns_its_picture_needs_no_second_inspection(self):
        """comfy_generate waits for the run and hands back the files and the
        image: that is the read-back. Before this, the executor treated it as
        an unverified edit and nagged the model to inspect after it had
        finished; the model could only answer with studio_task_update, and the
        conversation cycled on the task record until it stopped 'unverified'."""
        class Bridge:
            def call_tool(self, name, args):
                return comfy.call_tool(name, args)

        class LLM:
            def __init__(self):
                self.turn = 0
            def chat(self, messages, tools, max_tokens=None):
                self.turn += 1
                if self.turn == 1:
                    return {"choices": [{"finish_reason": "tool_calls", "message": {
                        "role": "assistant", "content": None, "tool_calls": [
                            {"id": "c1", "type": "function", "function": {
                                "name": "comfy_generate",
                                "arguments": json.dumps({"prompt": "a lighthouse"})}}]}}]}
                self.assertNotNag(messages)
                return {"choices": [{"finish_reason": "stop", "message": {
                    "role": "assistant", "content": "Saved; see the path above."}}]}
            def assertNotNag(self, messages):
                for m in messages:
                    if m.get("role") == "user" and "Before finishing, inspect" in m.get("content", ""):
                        raise AssertionError("executor asked for a read-back after a self-verifying generate")

        schemas = comfy.tool_list()
        tools = eng.to_openai_tools([t for t in schemas if t["name"] == "comfy_generate"])
        record = tasks.TaskRecord()
        ex = tasks.Executor(LLM(), Bridge(), tools, schemas=schemas, record=record)
        final = ex.run([{"role": "system", "content": "s"},
                        {"role": "user", "content": "draw a lighthouse"}], streaming=False)
        self.assertEqual(final, "Saved; see the path above.")
        self.assertEqual(record.status, "response complete; see recorded checks and limitations")
        entry = record.journal[-1]
        self.assertEqual(entry["name"], "comfy_generate")
        self.assertFalse(entry["read"])           # it is still an edit ...
        self.assertTrue(entry["verifies"])        # ... that carried its own evidence
        self.assertNotIn("Before finishing", json.dumps(ex.record.__dict__))

    def test_prompt_names_only_default_tools(self):
        # The registry test walks this too; here we also check the tools the
        # prompt *relies on* are in the default set, not merely not-absent.
        app = eng.APPS_BY_ID["comfyui"]
        for tool in ("comfy_list_models", "comfy_generate", "comfy_upload_image",
                     "comfy_wait", "comfy_status", "comfy_interrupt", "comfy_clear_queue"):
            self.assertIn(tool, app.tool_names())
        self.assertNotIn("comfy_run_workflow", app.tool_names())

    # ------------------------------------------------------------ generating

    def test_generate_builds_the_canonical_graph_and_brings_the_image_home(self):
        res = comfy.call_tool("comfy_generate", {
            "prompt": "a lighthouse at dusk", "negative": "blurry", "seed": 42,
            "width": 1152, "height": 896, "steps": 8, "cfg": 1.5, "realism": False,
            "checkpoint": "sd_xl_base_1.0.safetensors"})
        self.assertFalse(res["isError"])
        g = self.fake.prompts["p1"]
        self.assertEqual(g["1"]["inputs"]["ckpt_name"], "sd_xl_base_1.0.safetensors")
        self.assertEqual(g["2"]["inputs"], {"text": "a lighthouse at dusk", "clip": ["1", 1]})
        self.assertEqual(g["3"]["inputs"]["text"], "blurry")
        self.assertEqual(g["4"]["class_type"], "EmptyLatentImage")
        self.assertEqual((g["4"]["inputs"]["width"], g["4"]["inputs"]["height"]), (1152, 896))
        k = g["5"]["inputs"]
        self.assertEqual((k["seed"], k["steps"], k["cfg"], k["denoise"]), (42, 8, 1.5, 1.0))
        self.assertEqual(k["latent_image"], ["4", 0])
        self.assertEqual(g["7"]["class_type"], "SaveImage")
        # Result: seed for reproduction, the local path, and an inline image -
        # but not the temp preview.
        text = self.text(res)
        self.assertIn("seed: 42", text)
        self.assertIn("model: sd_xl_base_1.0.safetensors", text)
        path = os.path.join(self.tmp, "StudioAssistant_00001_.png")
        self.assertIn(path, text)
        self.assertTrue(os.path.isfile(path))
        images = [c for c in res["content"] if c["type"] == "image"]
        self.assertEqual(len(images), 1)
        self.assertEqual(base64.b64decode(images[0]["data"]), PNG)
        self.assertNotIn("preview.png", text)

    def test_generate_defaults_to_the_photographic_model_and_a_random_seed(self):
        """Z-Image out-draws an SD-era checkpoint for realism, so with both
        installed and nothing named, the split model is the default."""
        res = comfy.call_tool("comfy_generate", {"prompt": "x"})
        g = self.fake.prompts["p1"]
        self.assertEqual(g["1"]["inputs"]["unet_name"], "z_image_turbo_bf16.safetensors")
        self.assertGreaterEqual(g["5"]["inputs"]["seed"], 0)
        self.assertIn("seed: %d" % g["5"]["inputs"]["seed"], self.text(res))

    def test_a_non_generative_checkpoint_is_skipped_for_a_split_model(self):
        """The regression: the only checkpoint is sam3.1 (a segmentation model),
        which CheckpointLoaderSimple cannot generate from. The default must fall
        through to the Z-Image split model rather than pick it and fail every
        request. An explicit name still overrides."""
        self.fake.checkpoints = ["sam3.1_multiplex_fp16.safetensors"]
        res = comfy.call_tool("comfy_generate", {"prompt": "a duck"})
        self.assertFalse(res["isError"], self.text(res))
        self.assertEqual(self.fake.prompts["p1"]["1"]["class_type"], "UNETLoader")
        self.assertIn("Z-Image Turbo", self.text(res))
        # comfy_status and comfy_list_models say so, before anything is generated.
        self.assertIn("Z-Image Turbo", self.text(comfy.call_tool("comfy_status", {})))
        self.assertIn("None of these look like an image-generation checkpoint",
                      self.text(comfy.call_tool("comfy_list_models", {"kind": "checkpoints"})))
        # But if the user insists on it, it is used verbatim.
        comfy.call_tool("comfy_generate", {"prompt": "x",
                                           "checkpoint": "sam3.1_multiplex_fp16.safetensors"})
        # (p1 and p2 were the picture and its face detail run.)
        self.assertEqual(self.fake.prompts["p3"]["1"]["class_type"], "CheckpointLoaderSimple")

    def test_a_real_checkpoint_is_the_default_when_no_known_split_model_is(self):
        self.fake.checkpoints = ["dreamshaper_8.safetensors"]
        self.fake.diffusion_models = ["mystery_dit.safetensors"]
        comfy.call_tool("comfy_generate", {"prompt": "x"})
        self.assertEqual(self.fake.prompts["p1"]["1"]["inputs"]["ckpt_name"],
                         "dreamshaper_8.safetensors")

    def test_an_edit_model_is_never_the_text_to_image_default(self):
        """The regression: with Qwen-Image-Edit sorted ahead of Z-Image, every
        "draw a duck" ran the 20B edit model at 20 steps and cfg 2.5 - slow,
        soft, and not what it is for."""
        self.fake.checkpoints = []
        self.fake.diffusion_models = ["krea2_turbo_int8_convrot.safetensors",
                                      "qwen_image_edit_2509_fp8_e4m3fn.safetensors",
                                      "z_image_turbo_bf16.safetensors"]
        comfy.call_tool("comfy_generate", {"prompt": "a duck"})
        self.assertEqual(self.fake.prompts["p1"]["1"]["inputs"]["unet_name"],
                         "z_image_turbo_bf16.safetensors")
        self.fake.diffusion_models = ["qwen_image_edit_2509_fp8_e4m3fn.safetensors"]
        res = comfy.call_tool("comfy_generate", {"prompt": "a duck"})
        self.assertTrue(res["isError"])

    def test_krea2_gets_its_own_recipe(self):
        self.fake.diffusion_models = ["krea2_turbo_int8_convrot.safetensors"]
        self.fake.text_encoders = ["qwen3vl_4b_fp8_scaled.safetensors", "qwen_3_4b.safetensors"]
        comfy.call_tool("comfy_generate", {"prompt": "x"})
        g = self.fake.prompts["p1"]
        self.assertEqual(g["10"]["inputs"], {"clip_name": "qwen3vl_4b_fp8_scaled.safetensors",
                                             "type": "krea2", "device": "cpu"})
        self.assertEqual(g["11"]["inputs"]["vae_name"], "qwen_image_vae.safetensors")
        self.assertNotIn("12", g)                         # no shift node
        self.assertEqual(g["4"]["class_type"], "EmptyLatentImage")
        k = g["5"]["inputs"]
        self.assertEqual((k["steps"], k["cfg"], k["sampler_name"]), (8, 1.0, "euler"))

    def test_the_detail_pass_resamples_the_upscaled_picture(self):
        res = comfy.call_tool("comfy_generate", {"prompt": "a fisherman", "seed": 3,
                                                 "width": 832, "height": 1216})
        g = self.fake.prompts["p1"]
        self.assertEqual(g["13"]["class_type"], "ImageScaleBy")
        self.assertEqual(g["13"]["inputs"]["image"], ["6", 0])
        self.assertEqual(g["13"]["inputs"]["scale_by"], 1.5)
        self.assertEqual(g["14"]["inputs"]["pixels"], ["13", 0])
        # Tiled: a whole 2-4 MP frame through the VAE stalled ComfyUI for minutes.
        self.assertEqual((g["14"]["class_type"], g["16"]["class_type"]),
                         ("VAEEncodeTiled", "VAEDecodeTiled"))
        k = g["15"]["inputs"]
        self.assertEqual((k["latent_image"], k["denoise"], k["model"]), (["14", 0], 0.33, ["12", 0]))
        self.assertEqual(g["7"]["inputs"]["images"], ["16", 0])
        self.assertIn("detail pass: x1.5", self.text(res))
        self.assertIn("1248x1824", self.text(res))
        # Off on request, and never on image-to-image.
        comfy.call_tool("comfy_generate", {"prompt": "x", "hires": False})
        self.assertEqual(self.fake.prompts["p2"]["7"]["inputs"]["images"], ["6", 0])
        # Capped so a big base size is not blown past what the card holds.
        comfy.call_tool("comfy_generate", {"prompt": "x", "width": 2048, "height": 2048})
        self.assertNotIn("13", self.fake.prompts["p3"])

    def test_the_text_encoder_runs_on_the_cpu_unless_told_otherwise(self):
        """The encoder runs once a picture, the diffusion model every step: on
        the shared card, the encoder on the CPU made a picture in 41-49 s and
        the encoder on the GPU in 158 s."""
        comfy.call_tool("comfy_generate", {"prompt": "x"})
        self.assertEqual(self.fake.prompts["p1"]["10"]["inputs"]["device"], "cpu")
        real = comfy.ENCODER_ON_CPU
        comfy.ENCODER_ON_CPU = False
        try:
            comfy.call_tool("comfy_generate", {"prompt": "x"})
        finally:
            comfy.ENCODER_ON_CPU = real
        self.assertNotIn("device", self.fake.prompts["p2"]["10"]["inputs"])

    def test_realism_is_added_to_photographic_prompts_only(self):
        comfy.call_tool("comfy_generate", {"prompt": "a fisherman on a dock."})
        text = self.fake.prompts["p1"]["2"]["inputs"]["text"]
        self.assertTrue(text.startswith("a fisherman on a dock. Photorealistic photograph"))
        # cfg 1 never reads a negative: zero it, as the model's template does.
        self.assertEqual(self.fake.prompts["p1"]["3"], {
            "class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["2", 0]}})
        comfy.call_tool("comfy_generate", {"prompt": "a watercolor fox"})
        self.assertEqual(self.fake.prompts["p2"]["2"]["inputs"]["text"], "a watercolor fox")
        comfy.call_tool("comfy_generate", {"prompt": "a fox", "realism": False})
        self.assertEqual(self.fake.prompts["p3"]["2"]["inputs"]["text"], "a fox")
        # A checkpoint at real cfg gets a negative against the generated look.
        comfy.call_tool("comfy_generate", {"prompt": "a fox",
                                           "checkpoint": "sd_xl_base_1.0.safetensors"})
        self.assertIn("plastic skin", self.fake.prompts["p4"]["3"]["inputs"]["text"])

    def test_edit_builds_the_qwen_edit_graph_with_lightning(self):
        self.fake.diffusion_models = ["qwen_image_edit_2509_fp8_e4m3fn.safetensors",
                                      "z_image_turbo_bf16.safetensors"]
        self.fake.text_encoders = ["qwen_2.5_vl_7b_fp8_scaled.safetensors", "qwen_3_4b.safetensors"]
        self.fake.loras = ["Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors"]
        src = os.path.join(self.tmp, "room.png")
        with open(src, "wb") as fh:
            fh.write(PNG)
        res = comfy.call_tool("comfy_edit_image", {
            "image": src, "instruction": "make the walls green", "references": ["sofa.png"],
            "seed": 5, "whole_picture": True})
        self.assertFalse(res["isError"], self.text(res))
        self.assertEqual(len(self.fake.uploads), 1)          # the local path went up
        g = self.fake.prompts["p1"]
        self.assertEqual(g["1"]["inputs"]["unet_name"], "qwen_image_edit_2509_fp8_e4m3fn.safetensors")
        self.assertEqual(g["10"]["inputs"], {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
                                             "type": "qwen_image", "device": "cpu"})
        self.assertEqual(g["9"]["inputs"]["model"], ["1", 0])
        self.assertEqual(g["12"]["inputs"]["model"], ["9", 0])
        self.assertEqual(g["13"]["class_type"], "CFGNorm")
        self.assertEqual(g["21"]["class_type"], "FluxKontextImageScale")
        pos = g["2"]["inputs"]
        self.assertEqual(pos["prompt"], "make the walls green")
        self.assertEqual((pos["image1"], pos["image2"]), (["21", 0], ["22", 0]))
        self.assertEqual(g["22"]["inputs"]["image"], "sofa.png")
        # At cfg 1 the negative is never read: zeroed, not encoded a second time.
        self.assertEqual(g["3"], {"class_type": "ConditioningZeroOut",
                                  "inputs": {"conditioning": ["2", 0]}})
        k = g["5"]["inputs"]
        self.assertEqual((k["steps"], k["cfg"], k["model"], k["latent_image"]),
                         (4, 1.0, ["13", 0], ["4", 0]))
        self.assertEqual(g["7"]["inputs"]["images"], ["6", 0])   # a whole picture, saved as drawn
        self.assertIn("Lightning", self.text(res))
        # Full quality drops the LoRA and uses the template's 20 steps at cfg 4.
        comfy.call_tool("comfy_edit_image", {"image": "a.png", "instruction": "x", "fast": False,
                                             "whole_picture": True})
        g = self.fake.prompts["p2"]
        self.assertNotIn("9", g)
        self.assertEqual((g["5"]["inputs"]["steps"], g["5"]["inputs"]["cfg"]), (20, 4.0))
        self.assertEqual(g["3"]["inputs"]["prompt"], "")
        # photo_finish is Z-Image's detail pass as a second run. In the edit's
        # own graph ComfyUI kept the 19.5 GB edit model on the card while it
        # brought Z-Image in beside it, and one edit took 410 s. So the edit
        # ends in a preview, the GPU is freed, and the finish loads the preview.
        self.fake.running = []
        res = comfy.call_tool("comfy_edit_image", {"image": "a.png", "instruction": "x",
                                                   "photo_finish": True, "whole_picture": True})
        edit, finish = self.fake.prompts["p3"], self.fake.prompts["p4"]
        self.assertEqual(edit["7"], {"class_type": "PreviewImage",
                                     "inputs": {"images": ["6", 0]}})
        self.assertEqual({n["class_type"] for n in edit.values()} & {"SaveImage"}, set())
        self.assertEqual(finish["1"]["inputs"]["unet_name"], "z_image_turbo_bf16.safetensors")
        self.assertEqual(finish["31"]["inputs"]["image"], "preview.png [temp]")
        self.assertEqual(finish["13"]["inputs"]["image"], ["31", 0])
        self.assertEqual(finish["15"]["inputs"]["model"], ["12", 0])
        self.assertEqual(finish["15"]["inputs"]["denoise"], 0.25)
        self.assertEqual(finish["7"]["inputs"]["images"], ["16", 0])
        self.assertNotIn("42", finish)                   # no mask: all of it is redrawn
        self.assertEqual(self.fake.posts.count(("free", {"unload_models": True,
                                                         "free_memory": True})), 2)
        self.assertIn("photo finish", self.text(res))
        self.assertFalse(res["isError"], self.text(res))

    def local_edit_setup(self):
        self.fake.diffusion_models = ["qwen_image_edit_2509_fp8_e4m3fn.safetensors",
                                      "z_image_turbo_bf16.safetensors"]
        self.fake.text_encoders = ["qwen_2.5_vl_7b_fp8_scaled.safetensors", "qwen_3_4b.safetensors"]
        self.fake.loras = ["Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors"]
        self.fake.running = []

    def test_a_local_edit_keeps_the_original_outside_what_changed(self):
        """The edit model gives back the whole frame at ~1 MP, twice through
        the VAE: the "untouched" parts drift. So by default only what changed
        is taken from it, scaled back to the original's size and composited
        onto the original's own pixels."""
        self.local_edit_setup()
        res = comfy.call_tool("comfy_edit_image", {"image": "a.png", "instruction": "x"})
        self.assertFalse(res["isError"], self.text(res))
        edit, mask, save = (self.fake.prompts[p] for p in ("p1", "p2", "p3"))
        self.assertEqual(edit["7"]["class_type"], "PreviewImage")
        self.assertEqual(mask["30"]["inputs"]["image"], "a.png")
        self.assertEqual(mask["31"]["inputs"]["image"], "preview.png [temp]")
        self.assertEqual((mask["33"]["inputs"]["width"], mask["33"]["inputs"]["height"]),
                         (["32", 0], ["32", 1]))
        self.assertEqual(mask["50"]["inputs"]["blend_mode"], "difference")
        self.assertNotIn("40", mask)                     # no region, no SAM3
        comp = mask["64"]["inputs"]
        self.assertEqual((comp["destination"], comp["source"], comp["mask"]),
                         (["30", 0], ["33", 0], ["63", 0]))
        self.assertEqual(save["31"]["inputs"]["image"], "edit65.png [temp]")
        self.assertEqual(save["7"]["inputs"]["images"], ["31", 0])
        self.assertIn("kept the original outside what the edit changed (10%", self.text(res))
        # The saved picture carries all three runs, not only the last.
        self.assertIsNone(self.fake.extra["p1"])            # previews carry nothing
        meta = self.fake.extra["p3"]["extra_pnginfo"]["studio_pipeline"]
        self.assertEqual((meta["tool"], meta["arguments"]["instruction"]),
                         ("comfy_edit_image", "x"))
        self.assertEqual([st["prompt"] for st in meta["stages"]],
                         [edit, mask, save])
        self.assertEqual(meta["stages"][0]["server"], comfy.COMFY_URL)
        # A change over most of the frame is a global edit: kept whole.
        self.fake.shares = {"69": 0.8}
        res = comfy.call_tool("comfy_edit_image", {"image": "a.png", "instruction": "x"})
        self.assertEqual(self.fake.prompts["p6"]["31"]["inputs"]["image"], "preview.png [temp]")
        self.assertIn("changed 80% of the picture, so it is kept whole", self.text(res))

    def test_a_region_edit_takes_sam3s_mask_and_finishes_only_inside_it(self):
        self.local_edit_setup()
        self.fake.checkpoints = ["sam3.1_multiplex_fp16.safetensors"]
        self.fake.shares = {"69": 0.2, "89": 0.4}
        res = comfy.call_tool("comfy_edit_image", {"image": "a.png", "instruction": "make the "
                                                   "jacket red", "region": "jacket",
                                                   "photo_finish": True})
        self.assertFalse(res["isError"], self.text(res))
        mask, finish = self.fake.prompts["p2"], self.fake.prompts["p3"]
        self.assertEqual(mask["2"]["inputs"]["text"], "jacket:8")   # "jacket" alone finds one
        self.assertEqual((mask["40"]["inputs"]["image"], mask["41"]["inputs"]["image"]),
                         (["30", 0], ["33", 0]))            # before and after the edit
        self.assertEqual(mask["60"]["inputs"]["mask"], ["42", 0])
        self.assertEqual(mask["80"]["inputs"]["mask"], ["57", 0])
        self.assertEqual(finish["31"]["inputs"]["image"], "edit65.png [temp]")
        self.assertEqual(finish["32"]["inputs"]["image"], "edit66.png [temp]")
        self.assertEqual(finish["41"]["inputs"]["image"], ["16", 0])
        comp = finish["42"]["inputs"]
        self.assertEqual((comp["destination"], comp["source"], comp["mask"]),
                         (["30", 0], ["41", 0], ["33", 0]))
        self.assertEqual(finish["7"]["inputs"]["images"], ["42", 0])
        self.assertIn("outside the jacket (SAM3's mask, 20%", self.text(res))
        self.assertIn("inside the same mask", self.text(res))
        # SAM3 finds nothing: what changed is used instead, and said so.
        self.fake.shares = {"69": 0.0, "89": 0.3}
        res = comfy.call_tool("comfy_edit_image", {"image": "a.png", "instruction": "x",
                                                   "region": "unicorn"})
        self.assertEqual(self.fake.prompts["p6"]["31"]["inputs"]["image"], "edit85.png [temp]")
        self.assertIn("SAM3 found no 'unicorn'", self.text(res))

    def test_keep_takes_the_head_out_of_what_the_edit_may_change(self):
        """New clothes or a new body on the same person: the whole person is
        taken from the edit but the head, which stays the original's pixels."""
        self.local_edit_setup()
        self.fake.checkpoints = ["sam3.1_multiplex_fp16.safetensors"]
        self.fake.shares = {"69": 0.5, "89": 0.6, "36": 0.08}
        res = comfy.call_tool("comfy_edit_image", {"image": "a.png", "instruction": "x",
                                                   "region": "person", "keep": "head"})
        self.assertFalse(res["isError"], self.text(res))
        mask = self.fake.prompts["p2"]
        self.assertEqual(mask["3"]["inputs"]["text"], "head:8")
        self.assertEqual((mask["44"]["inputs"]["image"], mask["45"]["inputs"]["image"]),
                         (["30", 0], ["33", 0]))
        cut = mask["70"]["inputs"]
        self.assertEqual((cut["destination"], cut["source"], cut["operation"]),
                         (["63", 0], ["49", 0], "subtract"))
        self.assertEqual(mask["64"]["inputs"]["mask"], ["70", 0])
        self.assertEqual(mask["90"]["inputs"]["destination"], ["83", 0])   # the fallback too
        self.assertIn("the head (8% of the picture) is the original's own pixels", self.text(res))
        # A whole-picture edit with a kept head is the whole frame but the head.
        comfy.call_tool("comfy_edit_image", {"image": "a.png", "instruction": "make it night",
                                             "whole_picture": True, "keep": "face"})
        mask = self.fake.prompts["p5"]
        self.assertEqual(mask["43"]["class_type"], "SolidMask")
        self.assertEqual(mask["60"]["inputs"]["mask"], ["43", 0])
        self.assertEqual(self.fake.prompts["p6"]["31"]["inputs"]["image"], "edit65.png [temp]")

    def test_edit_without_an_edit_model_says_so(self):
        res = comfy.call_tool("comfy_edit_image", {"image": "a.png", "instruction": "x"})
        self.assertTrue(res["isError"])
        self.assertIn("no image-edit model", self.text(res))
        res = comfy.call_tool("comfy_edit_image", {"image": r"C:\nowhere\a.png",
                                                   "instruction": "x"})
        self.assertTrue(res["isError"])

    def test_face_swap_edits_each_face_cropped_and_stitches_it_back(self):
        """Edited whole, a face in a group photo is a few hundred of the edit
        model's pixels and came back a stranger. Each face is cropped, edited
        at full size against its match, and composited back where it was."""
        self.fake.checkpoints = ["sam3.1_multiplex_fp16.safetensors"]
        self.fake.diffusion_models = ["qwen_image_edit_2509_fp8_e4m3fn.safetensors"]
        self.fake.text_encoders = ["qwen_2.5_vl_7b_fp8_scaled.safetensors"]
        self.fake.loras = ["Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors"]
        self.fake.running = []
        # The scene: two people and a small face in the crowd, found right to left.
        self.fake.faces = {"old.png": [(763, 464, 100, 104), (254, 460, 102, 113),
                                       (992, 735, 16, 20)],
                           "us.png": [(292, 407, 130, 156), (730, 432, 128, 147)]}
        res = comfy.call_tool("comfy_face_swap", {"image": "old.png", "faces": "us.png",
                                                  "seed": 3})
        self.assertFalse(res["isError"], self.text(res))
        detect, edit, stitch = (self.fake.prompts[p] for p in ("p1", "p2", "p3"))
        self.assertEqual(detect["1"]["inputs"]["ckpt_name"], "sam3.1_multiplex_fp16.safetensors")
        # Left to right, the crowd face dropped: the man gets the left face.
        man, woman = edit["100"]["inputs"]["crop_region"], edit["120"]["inputs"]["crop_region"]
        self.assertTrue(man["x"] < 254 + 51 < man["x"] + man["width"])
        self.assertTrue(woman["x"] < 763 + 50 < woman["x"] + woman["width"])
        self.assertEqual(man["width"], man["height"])
        self.assertEqual(edit["102"]["inputs"]["bboxes"]["x"], 292)
        self.assertEqual(edit["122"]["inputs"]["bboxes"]["x"], 730)
        self.assertEqual(edit["103"]["inputs"]["image1"], ["101", 0])
        self.assertEqual(edit["103"]["inputs"]["image2"], ["102", 0])
        self.assertEqual(edit["106"]["inputs"]["steps"], 4)
        self.assertNotIn("SaveImage", {n["class_type"] for n in edit.values()})
        # The stitch loads each edit's preview and pastes it where it was cut.
        self.assertEqual(stitch["100"]["inputs"]["image"], "edit108.png [temp]")
        self.assertEqual(stitch["120"]["inputs"]["image"], "edit128.png [temp]")
        self.assertEqual((stitch["114"]["inputs"]["x"], stitch["114"]["inputs"]["y"]),
                         (man["x"], man["y"]))
        self.assertEqual(stitch["134"]["inputs"]["destination"], ["114", 0])
        self.assertEqual(stitch["9"]["inputs"]["images"], ["134", 0])
        self.assertIn("faces swapped: 2", self.text(res))
        # order swaps who gets which face.
        comfy.call_tool("comfy_face_swap", {"image": "old.png", "faces": "us.png",
                                            "order": [2, 1]})
        self.assertEqual(self.fake.prompts["p5"]["102"]["inputs"]["bboxes"]["x"], 730)
        res = comfy.call_tool("comfy_face_swap", {"image": "old.png", "faces": "us.png",
                                                  "order": [3]})
        self.assertTrue(res["isError"])
        res = comfy.call_tool("comfy_face_swap", {"image": "old.png", "faces": "nobody.png"})
        self.assertTrue(res["isError"])
        self.assertIn("No face found in nobody.png", self.text(res))

    def test_generate_redraws_each_face_at_full_size(self):
        """At 60 px a face is drawn smooth and waxy. With SAM3 installed the
        picture goes to a preview beside SAM3's face boxes, and a second run
        crops each face, resamples it at FACE_EDIT and blends it back through
        the oval - each crop taken from the picture as composited so far."""
        self.fake.checkpoints = ["sam3.1_multiplex_fp16.safetensors"]
        self.fake.running = []
        self.fake.size = (1824, 1248)
        # Two guests, a face too large to gain anything, and a speck of crowd.
        self.fake.faces = {"generated": [(300, 400, 60, 70), (700, 410, 58, 66),
                                         (1200, 100, 560, 600), (40, 40, 8, 9)]}
        res = comfy.call_tool("comfy_generate", {"prompt": "a wedding party", "seed": 5})
        self.assertFalse(res["isError"], self.text(res))
        first, second = self.fake.prompts["p1"], self.fake.prompts["p2"]
        self.assertEqual(first["7"]["class_type"], "PreviewImage")
        self.assertEqual(first["32"]["inputs"]["image"], first["7"]["inputs"]["images"])
        self.assertEqual(second["20"]["inputs"]["image"], "preview.png [temp]")
        crops = [n["inputs"]["crop_region"] for n in second.values()
                 if n["class_type"] == "ImageCropV2"]
        self.assertEqual(len(crops), 2)
        self.assertTrue(crops[0]["x"] < 330 < crops[0]["x"] + crops[0]["width"])
        self.assertEqual(second["100"]["inputs"]["image"], ["20", 0])
        self.assertEqual(second["120"]["inputs"]["image"], ["108", 0])
        self.assertEqual(second["103"]["inputs"]["denoise"], comfy.FACE_DENOISE)
        self.assertEqual(second["101"]["inputs"]["width"], comfy.FACE_EDIT)
        self.assertEqual(second["7"]["inputs"]["images"], ["128", 0])
        self.assertIn("a wedding party", second["2"]["inputs"]["text"])
        self.assertIn(b"IHDR", self.fake.uploads[-1])     # the oval went up
        self.assertIn("face detail: 2 face(s)", self.text(res))

        # No faces: the picture is saved as it was drawn.
        self.fake.faces = {}
        res = comfy.call_tool("comfy_generate", {"prompt": "a lighthouse"})
        save = self.fake.prompts["p4"]
        self.assertEqual(save["7"]["inputs"]["images"], ["20", 0])
        self.assertNotIn("face detail", self.text(res))
        # face_detail false, or no SAM3: one run, as before.
        comfy.call_tool("comfy_generate", {"prompt": "x", "face_detail": False})
        self.assertEqual(self.fake.prompts["p5"]["7"]["class_type"], "SaveImage")
        self.fake.checkpoints = ["dreamshaper_8.safetensors"]
        comfy.call_tool("comfy_generate", {"prompt": "x"})
        self.assertEqual(self.fake.prompts["p6"]["7"]["class_type"], "SaveImage")

    def test_face_swap_polishes_the_stitched_faces_with_z_image(self):
        """The edit model's face read as pasted onto an old photograph. With
        Z-Image installed the stitch goes to a preview and a fourth run
        redraws each swapped face lightly, keeping the likeness."""
        self.fake.checkpoints = ["sam3.1_multiplex_fp16.safetensors"]
        self.fake.diffusion_models = ["qwen_image_edit_2509_fp8_e4m3fn.safetensors",
                                      "z_image_turbo_bf16.safetensors"]
        self.fake.text_encoders = ["qwen_2.5_vl_7b_fp8_scaled.safetensors",
                                   "qwen_3_4b.safetensors"]
        self.fake.running = []
        self.fake.faces = {"old.png": [(254, 460, 102, 113)], "us.png": [(292, 407, 130, 156)]}
        res = comfy.call_tool("comfy_face_swap", {"image": "old.png", "faces": "us.png"})
        self.assertFalse(res["isError"], self.text(res))
        stitch, polish = self.fake.prompts["p3"], self.fake.prompts["p4"]
        self.assertEqual(stitch["9"]["class_type"], "PreviewImage")
        self.assertEqual(polish["20"]["inputs"]["image"], "edit9.png [temp]")
        self.assertEqual(polish["1"]["inputs"]["unet_name"], "z_image_turbo_bf16.safetensors")
        self.assertEqual(polish["103"]["inputs"]["denoise"], comfy.SWAP_DENOISE)
        self.assertEqual(polish["9"]["inputs"]["images"], ["108", 0])
        self.assertIn("face detail: 1 face(s)", self.text(res))
        # face_detail false: the stitch is the result, as before.
        comfy.call_tool("comfy_face_swap", {"image": "old.png", "faces": "us.png",
                                            "face_detail": False})
        self.assertEqual(self.fake.prompts["p7"]["9"]["class_type"], "SaveImage")

    def test_the_face_oval_is_a_valid_greyscale_png(self):
        png = comfy.oval_png(64)
        self.assertTrue(png.startswith(bytes([0x89]) + b"PNG"))
        rows = zlib.decompress(png[png.index(b"IDAT") + 4:png.index(b"IEND") - 8])
        self.assertEqual(len(rows), 64 * 65)
        centre, corner = rows[34 * 65 + 1 + 32], rows[1]
        self.assertEqual((centre, corner), (255, 0))

    def test_a_returned_picture_says_where_it_was_saved(self):
        res = comfy.call_tool("comfy_generate", {"prompt": "a fox"})
        image = [i for i in res["content"] if i["type"] == "image"][0]
        self.assertTrue(os.path.isfile(image["_meta"]["path"]))

    def test_upscale_redraws_at_the_larger_size(self):
        res = comfy.call_tool("comfy_upscale", {"image": "shot.png", "scale": 2})
        self.assertFalse(res["isError"], self.text(res))
        g = self.fake.prompts["p1"]
        self.assertEqual(g["8"]["inputs"]["image"], "shot.png")
        self.assertEqual(g["20"]["class_type"], "ImageScaleToTotalPixels")
        self.assertEqual(g["13"]["inputs"]["image"], ["20", 0])
        self.assertEqual(g["13"]["inputs"]["scale_by"], 2)
        self.assertEqual(g["15"]["inputs"]["denoise"], 0.3)
        self.assertEqual(g["7"]["inputs"]["images"], ["16", 0])

    def test_a_wait_rides_out_a_server_that_stops_answering(self):
        """ComfyUI stops answering HTTP while it stages a big model; that is a
        busy server, not a dead one, and the generate must not fail on it."""
        real = self.fake.route
        silent = [3]
        def stalls(method, r, q, data):
            if r.startswith("/history/") and silent[0]:
                silent[0] -= 1
                raise urllib.error.URLError("timed out")
            return real(method, r, q, data)
        self.fake.route = stalls
        real_sleep = comfy.time.sleep
        comfy.time.sleep = lambda s: None
        try:
            res = comfy.call_tool("comfy_generate", {"prompt": "x"})
        finally:
            comfy.time.sleep = real_sleep
        self.assertFalse(res["isError"], self.text(res))
        self.assertIn("took", self.text(res))

    def test_make_room_unloads_every_model_but_the_tabs_own(self):
        """The LLM PC's 3090 held the 30B and the vision model - 28 GB of a
        24 GB card - and ComfyUI saw 0.3 GB free: 254 s a picture. With them
        unloaded the same picture took 64 s."""
        listing = {"models": [
            {"key": "qwen3-coder-30b-a3b-instruct", "type": "llm",
             "loaded_instances": [{"id": "qwen3-coder-30b-a3b-instruct"},
                                  {"id": "qwen3-coder-30b-a3b-instruct:2"}]},
            {"key": "qwen2.5-vl-7b-instruct", "type": "llm",
             "loaded_instances": [{"id": "qwen2.5-vl-7b-instruct"}]},
            {"key": "qwen3.5-9b-deepseek-v4-flash", "type": "llm",
             "loaded_instances": [{"id": "qwen3.5-9b-deepseek-v4-flash"}]},
            {"key": "nomic-embed", "type": "embedding",
             "loaded_instances": [{"id": "nomic-embed"}]},
            {"key": "gemma", "type": "llm", "loaded_instances": []}]}
        unloaded = []
        def host(req, timeout=None):
            url = req if isinstance(req, str) else req.full_url
            if url.endswith("/api/v1/models"):
                return io.BytesIO(json.dumps(listing).encode())
            if url.endswith("/api/v1/models/unload"):
                unloaded.append(json.loads(req.data)["instance_id"])
                return io.BytesIO(b"{}")
            raise AssertionError(url)
        urllib.request.urlopen = host
        gone, err = eng.make_room("http://h:1234/v1", {"qwen3.5-9b-deepseek-v4-flash"})
        self.assertIsNone(err)
        self.assertEqual(gone, unloaded)
        self.assertEqual(unloaded, ["qwen3-coder-30b-a3b-instruct",
                                    "qwen3-coder-30b-a3b-instruct:2", "qwen2.5-vl-7b-instruct"])

    def test_only_rendering_tools_make_room_first(self):
        calls = []
        class Bridge:
            def call_tool(self, name, args):
                calls.append(name)
                return {}
            instructions = "x"
        app = eng.APPS_BY_ID["comfyui"]
        def room():
            calls.append("room")
            return 32768
        wrapped = eng.YieldGPU(Bridge(), app.gpu_tools, room,
                               lambda ctx: calls.append(("back", ctx)))
        wrapped.call_tool("comfy_status", {})
        wrapped.call_tool("comfy_generate", {})
        wrapped.settle()                  # the model is wanted again
        wrapped.call_tool("comfy_edit_image", {})
        wrapped.settle()
        self.assertEqual(calls, ["comfy_status", "room", "comfy_generate", ("back", 32768),
                                 "room", "comfy_edit_image", ("back", 32768)])
        self.assertEqual(wrapped.instructions, "x")
        # And the tab prefers a small tool-calling model, falling back to the
        # shared one when the host does not serve it.
        self.assertEqual(app.model_for(["qwen3.5-9b-deepseek-v4-flash", "big"], "big")[0],
                         "qwen3.5-9b-deepseek-v4-flash")
        self.assertEqual(app.model_for(["big"], "big")[0], "big")

    def test_the_tabs_model_comes_back_even_when_the_render_fails(self):
        calls = []
        class Broken:
            def call_tool(self, name, args):
                raise TimeoutError("bridge gone")
        wrapped = eng.YieldGPU(Broken(), {"comfy_generate"}, lambda: 16384,
                               lambda ctx: calls.append(ctx))
        with self.assertRaises(TimeoutError):
            wrapped.call_tool("comfy_generate", {})
        self.assertEqual(calls, [])       # away until the model is wanted...
        wrapped.settle()
        self.assertEqual(calls, [16384])  # ...and then back, failed render or not

    def test_the_picture_comes_back_before_the_model_does(self):
        """The model's reload after a render took 17 s, and the finished
        picture waited behind it, unseen; loading it beside the vision model's
        look at the picture doubled that look. So the result returns at once,
        the model comes back when it is next wanted - the executor settles
        before every request - and a render straight after a render leaves it
        away rather than loading it only to unload it again."""
        order = []
        class Bridge:
            def call_tool(self, name, args):
                order.append(name)
                return {"content": [{"type": "image", "data": "x"}]}
        windows = iter([32768, None])     # what it had; then, away, nothing
        wrapped = eng.YieldGPU(Bridge(), {"comfy_generate"},
                               lambda: order.append("room") or next(windows),
                               lambda ctx: order.append(("back", ctx)))
        result = wrapped.call_tool("comfy_generate", {})
        self.assertEqual(result["content"][0]["type"], "image")
        self.assertEqual(order, ["room", "comfy_generate"])       # not back yet
        wrapped.call_tool("comfy_generate", {})
        self.assertEqual(order, ["room", "comfy_generate", "room", "comfy_generate"])
        eng.settle(wrapped)
        self.assertEqual(order[-1], ("back", 32768))    # at the window it had first
        eng.settle(wrapped)                             # nothing away: nothing done
        eng.settle(object())                            # any other bridge is left alone
        self.assertEqual(len(order), 5)

    def test_a_made_picture_is_checked_against_the_brief_not_reviewed_for_flaws(self):
        """Asked to name defects, the vision model always found some - steam
        from a fox's mouth, fur not red enough - and the ComfyUI tab's model
        redrew the picture for each, twice in one request, against its own
        one-render rule. A tab whose pictures are what it made gets the check
        instead; a tab whose pictures are views of a project keeps the review."""
        self.assertTrue(eng.APPS_BY_ID["comfyui"].makes_pictures)
        self.assertFalse(any(a.makes_pictures for a in eng.APPS if a.id != "comfyui"))
        self.assertNotIn("defects", eng.Vision.CHECK)
        self.assertIn("Matches the brief.", eng.Vision.CHECK)
        asked = []
        class Eyes:
            def review(self, item, brief):
                asked.append("review")
                return "A fox. Defect: steam from its mouth."
            def check(self, item, brief):
                asked.append("check")
                return "A fox in the snow.\nMatches the brief."
        class Bridge:
            def call_tool(self, name, args):
                return {"content": [{"type": "text", "text": "saved fox.png"},
                                    {"type": "image", "mimeType": "image/png", "data": "x"}]}
        seen = []
        class LLM:
            replies = [{"role": "assistant", "content": "", "tool_calls": [
                           {"id": "c1", "type": "function", "function": {
                               "name": "comfy_generate", "arguments": "{\"prompt\": \"fox\"}"}}]},
                       {"role": "assistant", "content": "Here is your fox."}]
            def chat(self, messages, tools=None, max_tokens=None):
                seen.append(messages[-2]["content"] if len(messages) > 3 else "")
                return {"choices": [{"message": self.replies.pop(0), "finish_reason": "stop"}]}
        schemas = [t for t in comfy.tool_list() if t["name"] == "comfy_generate"]
        tools = eng.to_openai_tools(schemas)
        self.assertEqual(eng.run_agent(LLM(), Bridge(), tools, "a fox in the snow", "s",
                                       quiet=True, schemas=schemas, vision=Eyes(),
                                       makes_pictures=True), "Here is your fox.")
        self.assertEqual(asked, ["check"])
        self.assertIn("Matches the brief.", seen[-1])

    def test_a_step_waits_for_the_models_a_render_sent_away(self):
        """The executor asks the bridge to settle before every request to the
        model, so it never finds the model half-loaded and makes the host load
        a second copy just in time."""
        order = []
        class Bridge:
            def settle(self):
                order.append("settle")
            def call_tool(self, name, args):
                order.append(name)
                return {"content": [{"type": "text", "text": "ok"}]}
        class LLM:
            replies = [{"role": "assistant", "content": "", "tool_calls": [
                           {"id": "c1", "type": "function", "function": {
                               "name": "comfy_status", "arguments": "{}"}}]},
                       {"role": "assistant", "content": "Done."}]
            def stream(self, messages, tools, on_text):
                order.append("model")
                return self.replies.pop(0)
        tools = [{"type": "function", "function": {"name": "comfy_status", "parameters": {
            "type": "object", "properties": {}}}}]
        schemas = [{"name": "comfy_status", "inputSchema": {"type": "object", "properties": {}},
                    "annotations": {"readOnlyHint": True}}]
        ex = tasks.Executor(LLM(), Bridge(), tools, schemas=schemas)
        self.assertEqual(ex.run([{"role": "system", "content": "s"},
                                 {"role": "user", "content": "status?"}]), "Done.")
        self.assertEqual(order, ["settle", "model", "comfy_status", "settle", "model", "settle"])

    def test_give_back_reloads_only_a_model_that_is_gone(self):
        posted = []
        state = {"loaded": []}
        def host(req, timeout=None):
            url = req if isinstance(req, str) else req.full_url
            if url.endswith("/api/v1/models"):
                return io.BytesIO(json.dumps({"models": [{"key": "m", "type": "llm",
                    "loaded_instances": state["loaded"]}]}).encode())
            posted.append((url.rsplit("/", 1)[-1], json.loads(req.data)))
            return io.BytesIO(b"{}")
        urllib.request.urlopen = host
        self.assertIsNone(eng.give_back("http://h:1234/v1", "m", 32768))
        self.assertEqual(posted, [("load", {"model": "m", "context_length": 32768})])
        state["loaded"] = [{"id": "m"}]
        eng.give_back("http://h:1234/v1", "m", 32768)
        self.assertEqual(len(posted), 1)            # already back: no second instance

    def test_an_interrupted_prompt_says_so(self):
        comfy.call_tool("comfy_generate", {"prompt": "x", "wait": False})
        self.fake.history["p1"] = {"outputs": {}, "status": {
            "status_str": "error", "completed": False,
            "messages": [["execution_interrupted", {"node_id": "5"}]]}}
        res = comfy.call_tool("comfy_wait", {"prompt_id": "p1"})
        self.assertTrue(res["isError"])
        self.assertIn("interrupted", self.text(res))

    def test_a_finished_render_hands_the_gpu_back(self):
        """After a Z-Image render ComfyUI still held 11.7 GB of the card, where
        LM Studio could not see it: the 30B decoded at a third of its speed and
        the tab's model took 17 s to load instead of 3. A finished run with
        nothing queued behind it frees the GPU and waits for the memory to come
        back before it returns - the tab's model is loaded the moment it does."""
        self.fake.running = []
        self.fake.free_vram = [20e9, 4e9, 9e9, 19.5e9]    # before; then letting go
        slept = []
        real_sleep = comfy.time.sleep
        comfy.time.sleep = slept.append
        try:
            res = comfy.call_tool("comfy_generate", {"prompt": "x"})
        finally:
            comfy.time.sleep = real_sleep
        self.assertFalse(res["isError"], self.text(res))
        self.assertEqual(self.fake.posts, [("free", {"unload_models": True,
                                                     "free_memory": True})])
        self.assertEqual(self.fake.free_vram, [19.5e9])   # watched until it was back
        self.assertEqual(slept, [0.25, 0.25])
        # A run queued behind it wants the models where they are.
        self.fake.posts.clear()
        self.fake.running = [[2, "run-2", {}]]
        comfy.call_tool("comfy_generate", {"prompt": "x"})
        self.assertEqual(self.fake.posts, [])
        # And a ComfyUI with a card of its own keeps them (COMFYUI_KEEP_MODELS).
        self.fake.running = []
        comfy.KEEP_MODELS = True
        comfy.call_tool("comfy_generate", {"prompt": "x"})
        self.assertEqual(self.fake.posts, [])
        # A server that will not free costs speed, never the picture.
        comfy.KEEP_MODELS = False
        real = self.fake.route
        def refuses(method, r, q, data):
            if r == "/free":
                self.fake.fail(500, "no")
            return real(method, r, q, data)
        self.fake.route = refuses
        self.assertFalse(comfy.call_tool("comfy_generate", {"prompt": "x"})["isError"])

    def test_a_starved_gpu_is_named_in_the_result(self):
        real = self.fake.route
        def full(method, r, q, data):
            if r == "/system_stats":
                return {"devices": [{"name": "cuda:0 RTX 3090", "vram_total": 25.7e9,
                                     "vram_free": 0.3e9}]}
            return real(method, r, q, data)
        self.fake.route = full
        res = comfy.call_tool("comfy_generate", {"prompt": "x"})
        self.assertIn("0.3 of 26 GB free", self.text(res))

    def test_a_split_model_is_used_with_its_family_recipe_when_no_checkpoint_exists(self):
        """The LLM PC has Z-Image Turbo as three files and no checkpoint at all.
        With nothing named, generate must load the split model with the
        official recipe rather than fail for want of a checkpoint."""
        self.fake.checkpoints = []
        res = comfy.call_tool("comfy_generate", {"prompt": "x"})
        self.assertFalse(res["isError"], self.text(res))
        g = self.fake.prompts["p1"]
        self.assertEqual(g["1"], {"class_type": "UNETLoader", "inputs": {
            "unet_name": "z_image_turbo_bf16.safetensors", "weight_dtype": "default"}})
        self.assertEqual(g["10"]["inputs"], {"clip_name": "qwen_3_4b.safetensors", "type": "lumina2",
                                             "device": "cpu"})
        self.assertEqual(g["11"]["inputs"], {"vae_name": "ae.safetensors"})
        self.assertEqual(g["12"], {"class_type": "ModelSamplingAuraFlow",
                                   "inputs": {"model": ["1", 0], "shift": 3.0}})
        self.assertEqual(g["2"]["inputs"]["clip"], ["10", 0])
        self.assertEqual(g["4"]["class_type"], "EmptySD3LatentImage")
        k = g["5"]["inputs"]
        self.assertEqual((k["model"], k["steps"], k["cfg"], k["sampler_name"], k["scheduler"]),
                         (["12", 0], 8, 1.0, "res_multistep", "simple"))
        self.assertEqual(g["6"]["inputs"]["vae"], ["11", 0])
        self.assertIn("model: Z-Image Turbo", self.text(res))
        self.assertIn("steps: 8  cfg: 1.0  sampler: res_multistep/simple", self.text(res))
        # And the user is told, before generating, what will be used.
        self.assertIn("Z-Image Turbo", self.text(comfy.call_tool("comfy_status", {})))
        self.assertIn("z_image_turbo_bf16", self.text(comfy.call_tool("comfy_list_models", {})))

    def test_explicit_split_arguments_override_the_recipe(self):
        comfy.call_tool("comfy_generate", {
            "prompt": "x", "diffusion_model": "z_image_turbo_bf16.safetensors",
            "vae": "qwen_image_vae.safetensors", "text_encoder_type": "qwen_image",
            "shift": 1.5, "steps": 12, "cfg": 2.0})
        g = self.fake.prompts["p1"]
        self.assertEqual(g["1"]["class_type"], "UNETLoader")
        self.assertEqual(g["10"]["inputs"]["type"], "qwen_image")
        self.assertEqual(g["11"]["inputs"]["vae_name"], "qwen_image_vae.safetensors")
        self.assertEqual(g["12"]["inputs"]["shift"], 1.5)
        self.assertEqual((g["5"]["inputs"]["steps"], g["5"]["inputs"]["cfg"]), (12, 2.0))

    def test_nothing_to_generate_with_is_said_plainly(self):
        self.fake.checkpoints = []
        real = self.fake.route
        def no_split(method, r, q, data):
            return [] if r == "/models/diffusion_models" or r == "/models/unet" else real(method, r, q, data)
        self.fake.route = no_split
        res = comfy.call_tool("comfy_generate", {"prompt": "x"})
        self.assertTrue(res["isError"])
        self.assertIn("no checkpoints and no text-to-image diffusion models", self.text(res))

    def test_checkpoints_fall_back_to_the_loader_enum_on_old_servers(self):
        self.fake.models_route = False
        self.assertEqual(comfy.list_models("checkpoints"), ["fallback.safetensors"])

    def test_img2img_and_lora_rewire_the_graph(self):
        comfy.call_tool("comfy_generate", {"prompt": "x", "init_image": "sketch.png",
                                           "checkpoint": "sd_xl_base_1.0.safetensors",
                                           "denoise": 0.4, "lora": "detail.safetensors",
                                           "lora_strength": 0.7})
        g = self.fake.prompts["p1"]
        self.assertEqual(g["8"], {"class_type": "LoadImage", "inputs": {"image": "sketch.png"}})
        self.assertEqual(g["4"]["class_type"], "VAEEncode")
        self.assertEqual(g["4"]["inputs"], {"pixels": ["8", 0], "vae": ["1", 2]})
        self.assertEqual(g["5"]["inputs"]["denoise"], 0.4)
        self.assertEqual(g["9"]["inputs"]["lora_name"], "detail.safetensors")
        self.assertEqual(g["9"]["inputs"]["strength_model"], 0.7)
        self.assertEqual(g["5"]["inputs"]["model"], ["9", 0])
        self.assertEqual(g["2"]["inputs"]["clip"], ["9", 1])

    def test_a_second_output_with_the_same_name_is_not_overwritten(self):
        comfy.call_tool("comfy_generate", {"prompt": "x"})
        self.fake.route = self._different_png(self.fake.route)
        res = comfy.call_tool("comfy_generate", {"prompt": "y"})
        self.assertIn("StudioAssistant_00001_-p2.png", self.text(res))
        self.assertEqual(len(os.listdir(self.tmp)), 2)

    @staticmethod
    def _different_png(route):
        def patched(method, r, q, data):
            body = route(method, r, q, data)
            return body + b"\x00" if r == "/view" else body
        return patched

    def test_no_wait_returns_the_prompt_id_and_wait_collects_it(self):
        res = comfy.call_tool("comfy_generate", {"prompt": "x", "wait": False})
        self.assertIn("prompt_id p1", self.text(res))
        self.assertFalse(os.listdir(self.tmp))
        res = comfy.call_tool("comfy_wait", {"prompt_id": "p1"})
        self.assertIn("StudioAssistant_00001_.png", self.text(res))

    def test_wait_on_an_unknown_prompt_is_an_error_not_a_hang(self):
        res = comfy.call_tool("comfy_wait", {"prompt_id": "ghost"})
        self.assertTrue(res["isError"])
        self.assertIn("ghost", self.text(res))

    def test_a_timeout_hands_back_the_prompt_id(self):
        self.fake.finish_after = float("inf")
        real_sleep, real_clock = comfy.time.sleep, comfy.time.monotonic
        clock = [0.0]
        comfy.time.sleep = lambda s: clock.__setitem__(0, clock[0] + s)
        comfy.time.monotonic = lambda: clock[0]
        try:
            res = comfy.call_tool("comfy_generate", {"prompt": "x", "timeout": 5})
        finally:
            comfy.time.sleep, comfy.time.monotonic = real_sleep, real_clock
        self.assertTrue(res["isError"])
        self.assertIn("p1", self.text(res))
        self.assertIn("comfy_wait", self.text(res))

    def test_run_workflow_refuses_ui_format(self):
        res = comfy.call_tool("comfy_run_workflow", {"workflow": {"nodes": [], "links": []}})
        self.assertTrue(res["isError"])
        self.assertIn("API Format", self.text(res))
        res = comfy.call_tool("comfy_run_workflow", {"workflow": {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "a"}}}})
        self.assertFalse(res["isError"])
        self.assertEqual(self.fake.prompts["p1"]["1"]["class_type"], "CheckpointLoaderSimple")

    # -------------------------------------------------------- errors as prose

    def test_comfyui_validation_errors_name_the_node(self):
        self.fake.reject_next = (400, {
            "error": {"type": "prompt_outputs_failed_validation", "message": "Prompt outputs failed validation",
                      "details": ""},
            "node_errors": {"1": {"class_type": "CheckpointLoaderSimple", "errors": [
                {"message": "Value not in list", "details": "ckpt_name: 'sdxl' not in [...]"}]}}})
        res = comfy.call_tool("comfy_generate", {"prompt": "x", "checkpoint": "sdxl"})
        self.assertTrue(res["isError"])
        self.assertIn("HTTP 400", self.text(res))
        self.assertIn("CheckpointLoaderSimple", self.text(res))
        self.assertIn("ckpt_name", self.text(res))

    def test_unreachable_server_says_where_it_should_be(self):
        def down(req, timeout=None):
            raise urllib.error.URLError("connection refused")
        urllib.request.urlopen = down
        res = comfy.call_tool("comfy_status", {})
        self.assertTrue(res["isError"])
        self.assertIn(comfy.COMFY_URL, self.text(res))
        self.assertIn("--listen", self.text(res))

    def test_execution_errors_reach_the_model(self):
        comfy.call_tool("comfy_generate", {"prompt": "x", "wait": False})
        self.fake.history["p1"] = {"outputs": {}, "status": {
            "status_str": "error", "completed": False,
            "messages": [["execution_error", {"exception_type": "RuntimeError",
                                              "node_type": "KSampler",
                                              "exception_message": "CUDA out of memory"}]]}}
        res = comfy.call_tool("comfy_wait", {"prompt_id": "p1"})
        self.assertTrue(res["isError"])
        self.assertIn("CUDA out of memory", self.text(res))

    def test_unknown_tool_and_bad_arguments_are_errors_not_exceptions(self):
        self.assertTrue(comfy.call_tool("nope", {})["isError"])
        self.assertTrue(comfy.call_tool("comfy_node_info", {})["isError"])
        self.assertTrue(comfy.call_tool("comfy_generate", {"prompt": "  "})["isError"])

    # --------------------------------------------------------- the small ones

    def test_status_models_queue_history_and_control(self):
        self.assertIn("RTX 4090", self.text(comfy.call_tool("comfy_status", {})))
        self.assertIn("dreamshaper_8", self.text(comfy.call_tool("comfy_list_models", {})))
        self.assertIn("detail.safetensors",
                      self.text(comfy.call_tool("comfy_list_models", {"kind": "loras"})))
        self.assertIn("run-1", self.text(comfy.call_tool("comfy_queue", {})))
        comfy.call_tool("comfy_generate", {"prompt": "x"})
        self.assertIn("p1  success  1 file", self.text(comfy.call_tool("comfy_history", {})))
        comfy.call_tool("comfy_interrupt", {})
        comfy.call_tool("comfy_clear_queue", {})
        comfy.call_tool("comfy_clear_queue", {"prompt_ids": ["a", "b"]})
        self.assertEqual(self.fake.posts, [("interrupt", None), ("queue", {"clear": True}),
                                           ("queue", {"delete": ["a", "b"]})])

    def test_nodes_are_searchable_and_describable(self):
        self.assertIn("ImageUpscaleWithModel",
                      self.text(comfy.call_tool("comfy_search_nodes", {"query": "upscal"})))
        info = self.text(comfy.call_tool("comfy_node_info", {"node": "KSampler"}))
        self.assertIn("required seed: INT (default=0, min=0)", info)
        self.assertIn("one of %d values" % len(comfy.SAMPLERS), info)
        self.assertIn("outputs: 0=LATENT(LATENT)", info)
        self.assertTrue(comfy.call_tool("comfy_node_info", {"node": "Nope"})["isError"])

    def test_upload_sends_the_file_and_returns_the_name_to_use(self):
        src = os.path.join(self.tmp, "sketch.png")
        with open(src, "wb") as fh:
            fh.write(PNG)
        res = comfy.call_tool("comfy_upload_image", {"path": src})
        self.assertIn("'sketch.png'", self.text(res))
        self.assertIn(PNG, self.fake.uploads[0])
        self.assertIn(b'name="image"; filename="sketch.png"', self.fake.uploads[0])
        self.assertTrue(comfy.call_tool("comfy_upload_image",
                                        {"path": os.path.join(self.tmp, "no.png")})["isError"])

    # ------------------------------------------------------------ the wire

    def test_stdio_server_speaks_what_mcpclient_expects(self):
        lines = [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
             "params": {"name": "comfy_status", "arguments": {}}},
            {"jsonrpc": "2.0", "id": 4, "method": "resources/list"},
        ]
        out = io.StringIO()
        comfy.serve(io.StringIO("\n".join(json.dumps(m) for m in lines) + "\nnot json\n"), out)
        replies = [json.loads(l) for l in out.getvalue().splitlines()]
        # The unparseable line gets the JSON-RPC parse error, addressed to no id.
        self.assertEqual([r["id"] for r in replies], [1, 2, 3, 4, None])
        self.assertEqual(replies[4]["error"]["code"], -32700)
        self.assertEqual(replies[0]["result"]["serverInfo"]["name"], "studio-comfy-mcp")
        self.assertEqual(len(replies[1]["result"]["tools"]), len(comfy.TOOLS))
        self.assertIn("RTX 4090", replies[2]["result"]["content"][0]["text"])
        self.assertEqual(replies[3]["error"]["code"], -32601)

    def test_executor_validation_accepts_what_the_prompt_teaches(self):
        """The executor validates against the original schema; the argument
        shapes the prompt tells the model to use must pass it."""
        schema = {t["name"]: t["inputSchema"] for t in comfy.tool_list()}
        tasks.validate({"prompt": "x", "width": 1152, "height": 896, "seed": 7,
                        "sampler": "dpmpp_2m", "scheduler": "karras", "batch_size": 4},
                       schema["comfy_generate"])
        tasks.validate({"kind": "loras"}, schema["comfy_list_models"])
        with self.assertRaises(ValueError):
            tasks.validate({"kind": "lora"}, schema["comfy_list_models"])
        with self.assertRaises(ValueError):
            tasks.validate({"prompt": "x", "sampler": "euler_a"}, schema["comfy_generate"])


if __name__ == "__main__":
    unittest.main()
