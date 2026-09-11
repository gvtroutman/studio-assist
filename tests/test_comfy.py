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
        self.finish_after = 0      # history polls before a prompt "completes"
        self._polls = {}

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

    def fail(self, code, body):
        raise urllib.error.HTTPError("x", code, "err", {}, io.BytesIO(json.dumps(body).encode()))

    def route(self, method, route, q, data):
        if route == "/system_stats":
            return {"system": {"comfyui_version": "0.3.0"},
                    "devices": [{"name": "cuda:0 RTX 4090", "vram_total": 24e9, "vram_free": 20e9}]}
        if route == "/queue" and method == "GET":
            return {"queue_running": [[1, "run-1", {"5": {"class_type": "KSampler"}}]],
                    "queue_pending": []}
        if route == "/queue":
            self.posts.append(("queue", json.loads(data)))
            return {}
        if route == "/interrupt":
            self.posts.append(("interrupt", None))
            return b""
        if route.startswith("/models/"):
            if not self.models_route:
                self.fail(404, "not found")
            return {"checkpoints": self.checkpoints,
                    "loras": ["detail.safetensors"],
                    "diffusion_models": ["z_image_turbo_bf16.safetensors"],
                    "text_encoders": ["qwen_3_4b.safetensors"],
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
            self.history[pid] = {
                "prompt": [], "outputs": {"7": {"images": [
                    {"filename": "StudioAssistant_00001_.png", "subfolder": "", "type": "output"},
                    {"filename": "preview.png", "subfolder": "", "type": "temp"}]}},
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

    def tearDown(self):
        urllib.request.urlopen = self._real_open
        comfy.OUTPUT_DIR = self._real_out
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
            "width": 1152, "height": 896, "steps": 8, "cfg": 1.5,
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

    def test_generate_defaults_to_the_first_checkpoint_and_a_random_seed(self):
        res = comfy.call_tool("comfy_generate", {"prompt": "x"})
        g = self.fake.prompts["p1"]
        self.assertEqual(g["1"]["inputs"]["ckpt_name"], "sd_xl_base_1.0.safetensors")
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
        self.assertEqual(self.fake.prompts["p2"]["1"]["class_type"], "CheckpointLoaderSimple")

    def test_a_real_checkpoint_still_wins_over_a_split_model(self):
        """The fallback must not demote a genuine checkpoint: with both present,
        the all-in-one checkpoint is still the default."""
        self.fake.checkpoints = ["dreamshaper_8.safetensors"]
        comfy.call_tool("comfy_generate", {"prompt": "x"})
        self.assertEqual(self.fake.prompts["p1"]["1"]["inputs"]["ckpt_name"],
                         "dreamshaper_8.safetensors")

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
        self.assertEqual(g["10"]["inputs"], {"clip_name": "qwen_3_4b.safetensors", "type": "lumina2"})
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
        self.assertIn("no checkpoints and no diffusion models", self.text(res))

    def test_checkpoints_fall_back_to_the_loader_enum_on_old_servers(self):
        self.fake.models_route = False
        self.assertEqual(comfy.list_models("checkpoints"), ["fallback.safetensors"])

    def test_img2img_and_lora_rewire_the_graph(self):
        comfy.call_tool("comfy_generate", {"prompt": "x", "init_image": "sketch.png",
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
