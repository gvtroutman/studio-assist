#!/usr/bin/env python3
"""
studio_comfy_mcp - an MCP stdio bridge to a ComfyUI server.

ComfyUI is the one app in the registry that does not run on this workstation:
it sits on the same tailnet box as LM Studio, so its GPU does the image work
while the RTX 5090 here stays free for After Effects and Resolve. The bridge
runs *here*, like every other bridge, and talks to ComfyUI over its HTTP API:

    /system_stats /queue /prompt /history /view /upload/image /object_info

Generated images are pulled back to this machine (COMFYUI_OUTPUT_DIR, default
~/Pictures/ComfyUI) so the other tabs can import them, and returned as image
content so the chat window can show them inline.

Stdlib only. The protocol - framing, negotiation, validation, annotations,
progress and cancellation - is studio_mcp's; this file is the tools. Run it by
hand to see the tool list, or to check its own contract:

    python studio_comfy_mcp.py --list-tools
    python studio_comfy_mcp.py --check
"""

import base64
import json
import os
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

import studio_mcp

DEFAULT_URL = "http://100.127.17.38:8188"
COMFY_URL = os.environ.get("COMFYUI_URL", DEFAULT_URL).rstrip("/")
OUTPUT_DIR = os.environ.get(
    "COMFYUI_OUTPUT_DIR",
    os.path.join(os.path.expanduser("~"), "Pictures", "ComfyUI"))

MAX_INLINE_IMAGES = 4        # image content blocks per result; the rest are paths
MAX_WAIT = 600               # seconds a generate/wait call will block for
CLIENT_ID = uuid.uuid4().hex

# ComfyUI's model folders, as /models/<folder> names them. `kind` on
# comfy_list_models is an enum of these so the model cannot ask for "lora".
MODEL_KINDS = ["checkpoints", "diffusion_models", "text_encoders", "vae", "loras",
               "controlnet", "upscale_models", "embeddings", "clip", "unet", "clip_vision"]

SAMPLERS = ["euler", "euler_cfg_pp", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral",
            "lms", "dpmpp_2s_ancestral", "dpmpp_sde", "dpmpp_2m", "dpmpp_2m_sde",
            "dpmpp_3m_sde", "ddim", "uni_pc", "lcm", "deis", "ipndm", "res_multistep",
            "res_multistep_cfg_pp", "gradient_estimation", "er_sde", "sa_solver"]
SCHEDULERS = ["normal", "karras", "exponential", "sgm_uniform", "simple",
              "ddim_uniform", "beta", "linear_quadratic", "kl_optimal"]
# CLIPLoader's `type`: which text encoder family a split model's encoder is.
ENCODER_TYPES = ["lumina2", "qwen_image", "flux2", "chroma", "sd3", "wan", "hidream",
                 "pixart", "cosmos", "ltxv", "mochi", "hunyuan_image", "stable_diffusion"]

# Split models - a diffusion model, a text encoder and a VAE as three files
# rather than one checkpoint - each want their own sampling recipe, and the
# wrong one gives noise rather than an error. Matched on the diffusion model's
# filename, first hit wins; anything the caller passes overrides.
FAMILIES = [
    ("z_image", {"label": "Z-Image Turbo", "encoder_type": "lumina2", "encoder": "qwen_3",
                 "vae": "ae.safetensors", "shift": 3.0, "steps": 8, "cfg": 1.0,
                 "sampler": "res_multistep", "scheduler": "simple"}),
    ("qwen_image", {"label": "Qwen-Image", "encoder_type": "qwen_image", "encoder": "qwen_2.5_vl",
                    "vae": "qwen_image_vae.safetensors", "shift": 3.1, "steps": 20,
                    "cfg": 2.5, "sampler": "euler", "scheduler": "simple"}),
    ("flux", {"label": "Flux", "encoder_type": "flux2", "encoder": "t5",
              "vae": "ae.safetensors", "shift": None, "steps": 20, "cfg": 1.0,
              "sampler": "euler", "scheduler": "simple"}),
    ("chroma", {"label": "Chroma", "encoder_type": "chroma", "encoder": "t5",
                "vae": "ae.safetensors", "shift": None, "steps": 26, "cfg": 4.0,
                "sampler": "euler", "scheduler": "simple"}),
]
GENERIC_FAMILY = {"label": "split model", "encoder_type": "stable_diffusion", "encoder": "",
                  "vae": None, "shift": None, "steps": 20, "cfg": 6.0, "sampler": "euler",
                  "scheduler": "normal"}


class ComfyError(Exception):
    """Anything ComfyUI, or the network in front of it, refuses."""


# ------------------------------------------------------------------- HTTP

def _open(req, timeout):
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:2000]
        try:
            detail = json.loads(body)
        except ValueError:
            detail = body
        raise ComfyError("ComfyUI answered HTTP %d: %s" % (e.code, _explain(detail)))
    except urllib.error.URLError as e:
        raise ComfyError(
            "Cannot reach ComfyUI at %s (%s). It has to be running on the LLM PC, "
            "started with --listen so it accepts connections from this machine."
            % (COMFY_URL, e.reason))
    except OSError as e:
        raise ComfyError("Cannot reach ComfyUI at %s (%s)." % (COMFY_URL, e))


def _explain(detail):
    """ComfyUI's validation errors are nested; surface the messages, not the tree."""
    if not isinstance(detail, dict):
        return str(detail)
    parts = []
    err = detail.get("error")
    if isinstance(err, dict):
        parts.append(err.get("message", "") + (" - " + err["details"] if err.get("details") else ""))
    elif err:
        parts.append(str(err))
    for node, info in (detail.get("node_errors") or {}).items():
        for e in info.get("errors", []):
            parts.append("node %s (%s): %s %s" % (
                node, info.get("class_type", "?"), e.get("message", ""), e.get("details", "")))
    return "; ".join(p for p in parts if p) or json.dumps(detail)[:500]


def get_json(path, timeout=15):
    with _open(COMFY_URL + path, timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def post_json(path, payload, timeout=30):
    req = urllib.request.Request(
        COMFY_URL + path, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with _open(req, timeout) as r:
        body = r.read().decode("utf-8")
        return json.loads(body) if body.strip() else {}


def get_bytes(path, timeout=60):
    with _open(COMFY_URL + path, timeout) as r:
        return r.read()


def post_multipart(path, fields, filename, data, timeout=120):
    boundary = "----studio" + uuid.uuid4().hex
    body = b""
    for k, v in fields.items():
        body += ("--%s\r\nContent-Disposition: form-data; name=\"%s\"\r\n\r\n%s\r\n"
                 % (boundary, k, v)).encode("utf-8")
    body += ("--%s\r\nContent-Disposition: form-data; name=\"image\"; filename=\"%s\"\r\n"
             "Content-Type: application/octet-stream\r\n\r\n"
             % (boundary, filename)).encode("utf-8")
    body += data + ("\r\n--%s--\r\n" % boundary).encode("utf-8")
    req = urllib.request.Request(
        COMFY_URL + path, data=body, method="POST",
        headers={"Content-Type": "multipart/form-data; boundary=" + boundary})
    with _open(req, timeout) as r:
        return json.loads(r.read().decode("utf-8"))


# ------------------------------------------------------------- ComfyUI model

def list_models(kind):
    """Names ComfyUI would accept for a loader of this kind."""
    try:
        names = get_json("/models/" + urllib.parse.quote(kind))
        if isinstance(names, list):
            return names
    except ComfyError as e:
        if "HTTP 404" not in str(e):
            raise
    # Older servers have no /models route; the loader node's enum is the same list.
    loader = {"checkpoints": ("CheckpointLoaderSimple", "ckpt_name"),
              "loras": ("LoraLoader", "lora_name"),
              "vae": ("VAELoader", "vae_name"),
              "controlnet": ("ControlNetLoader", "control_net_name"),
              "upscale_models": ("UpscaleModelLoader", "model_name"),
              "unet": ("UNETLoader", "unet_name"),
              "diffusion_models": ("UNETLoader", "unet_name"),
              "clip": ("CLIPLoader", "clip_name"),
              "text_encoders": ("CLIPLoader", "clip_name"),
              "clip_vision": ("CLIPVisionLoader", "clip_name")}.get(kind)
    if not loader:
        return []
    info = get_json("/object_info/" + loader[0]).get(loader[0], {})
    spec = info.get("input", {}).get("required", {}).get(loader[1])
    return list(spec[0]) if spec and isinstance(spec[0], list) else []


def family_of(name):
    low = (name or "").lower()
    for key, preset in FAMILIES:
        if key in low:
            return dict(preset)
    return dict(GENERIC_FAMILY)


def pick_first(names, prefer=""):
    """First installed name containing `prefer`, else the first of them."""
    for n in names:
        if prefer and prefer.lower() in n.lower():
            return n
    return names[0] if names else None


# Files that live in models/checkpoints but are not txt2img base models: a
# segmentation model, an upscaler, a face model and so on. CheckpointLoaderSimple
# chokes on them, so they must not be picked as the default generator - the
# reason a lone sam3.1 checkpoint failed every "make a duck".
NON_GENERATIVE = ("sam", "segment", "florence", "yolo", "grounding", "depth",
                  "controlnet", "clip_vision", "clipvision", "upscal", "esrgan",
                  "codeformer", "gfpgan", "insightface", "instantid", "rembg",
                  "birefnet", "inpaint_only", "annotator", "preprocessor")


def is_generative_checkpoint(name):
    return not any(k in name.lower() for k in NON_GENERATIVE)


def is_known_family(name):
    return family_of(name)["label"] != GENERIC_FAMILY["label"]


def plan_model(a):
    """
    What to load, when the caller names nothing. A real all-in-one checkpoint is
    preferred; but a checkpoint whose name is a non-generative model (a lone
    sam3.1, an upscaler) is skipped in favour of a split model that has a known
    recipe, because such a checkpoint cannot make a picture and picking it fails
    every request. Explicit `checkpoint`/`diffusion_model` arguments always win.
    """
    if a.get("checkpoint"):
        return {"kind": "checkpoint", "checkpoint": a["checkpoint"], "label": a["checkpoint"]}
    unet = a.get("diffusion_model")
    if not unet:
        ckpts = list_models("checkpoints")
        unets = list_models("diffusion_models") or list_models("unet")
        good = [c for c in ckpts if is_generative_checkpoint(c)]
        known = [u for u in unets if is_known_family(u)]
        if good:                      # a plausible all-in-one checkpoint
            return {"kind": "checkpoint", "checkpoint": good[0], "label": good[0]}
        if known:                     # a split model we have a working recipe for
            unet = known[0]
        elif unets:                   # any diffusion model, generic recipe
            unet = unets[0]
        elif ckpts:                   # last resort: a checkpoint that looked unusable
            return {"kind": "checkpoint", "checkpoint": ckpts[0], "label": ckpts[0]}
        else:
            raise ComfyError("ComfyUI has no checkpoints and no diffusion models installed; "
                             "nothing can generate.")
    fam = family_of(unet)
    encoders = list_models("text_encoders") or list_models("clip")
    vaes = list_models("vae")
    encoder = a.get("text_encoder") or pick_first(encoders, fam["encoder"])
    vae = a.get("vae") or (fam["vae"] if fam["vae"] in vaes else pick_first(vaes))
    if not encoder or not vae:
        raise ComfyError("%s needs a text encoder and a VAE on ComfyUI; found encoders %s "
                         "and VAEs %s." % (unet, encoders, vaes))
    plan = dict(fam, kind="split", diffusion_model=unet, text_encoder=encoder, vae=vae)
    plan["encoder_type"] = a.get("text_encoder_type") or fam["encoder_type"]
    if a.get("shift") is not None:
        plan["shift"] = a["shift"]
    plan["label"] = "%s (%s + %s + %s)" % (fam["label"], unet, encoder, vae)
    return plan


def build_graph(a):
    """
    The canonical txt2img graph, with img2img and one LoRA as options. Node ids
    are strings; a link is [node_id, output_index]. Returns (graph, seed, plan).
    """
    seed = a.get("seed")
    if seed is None or seed < 0:
        seed = random.randint(0, 2**32 - 1)
    plan = plan_model(a)
    if plan["kind"] == "checkpoint":
        g = {"1": {"class_type": "CheckpointLoaderSimple",
                   "inputs": {"ckpt_name": plan["checkpoint"]}}}
        model, clip, vae = ["1", 0], ["1", 1], ["1", 2]
        latent_node = "EmptyLatentImage"
        defaults = {"steps": 20, "cfg": 6.0, "sampler": "euler", "scheduler": "normal"}
    else:
        g = {"1": {"class_type": "UNETLoader", "inputs": {
                 "unet_name": plan["diffusion_model"], "weight_dtype": "default"}},
             "10": {"class_type": "CLIPLoader", "inputs": {
                 "clip_name": plan["text_encoder"], "type": plan["encoder_type"]}},
             "11": {"class_type": "VAELoader", "inputs": {"vae_name": plan["vae"]}}}
        model, clip, vae = ["1", 0], ["10", 0], ["11", 0]
        if plan.get("shift") is not None:
            g["12"] = {"class_type": "ModelSamplingAuraFlow",
                       "inputs": {"model": model, "shift": plan["shift"]}}
            model = ["12", 0]
        latent_node = "EmptySD3LatentImage"
        defaults = plan
    if a.get("lora"):
        g["9"] = {"class_type": "LoraLoader", "inputs": {
            "lora_name": a["lora"],
            "strength_model": a.get("lora_strength", 1.0),
            "strength_clip": a.get("lora_strength", 1.0),
            "model": model, "clip": clip}}
        model, clip = ["9", 0], ["9", 1]
    g["2"] = {"class_type": "CLIPTextEncode", "inputs": {"text": a["prompt"], "clip": clip}}
    g["3"] = {"class_type": "CLIPTextEncode", "inputs": {"text": a.get("negative", ""), "clip": clip}}
    denoise = 1.0
    if a.get("init_image"):
        g["8"] = {"class_type": "LoadImage", "inputs": {"image": a["init_image"]}}
        g["4"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["8", 0], "vae": vae}}
        denoise = a.get("denoise", 0.6)
    else:
        g["4"] = {"class_type": latent_node, "inputs": {
            "width": a.get("width", 1024), "height": a.get("height", 1024),
            "batch_size": a.get("batch_size", 1)}}
    g["5"] = {"class_type": "KSampler", "inputs": {
        "seed": seed, "steps": a.get("steps", defaults["steps"]),
        "cfg": a.get("cfg", defaults["cfg"]),
        "sampler_name": a.get("sampler", defaults["sampler"]),
        "scheduler": a.get("scheduler", defaults["scheduler"]),
        "denoise": denoise, "model": model, "positive": ["2", 0], "negative": ["3", 0],
        "latent_image": ["4", 0]}}
    g["6"] = {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": vae}}
    g["7"] = {"class_type": "SaveImage", "inputs": {
        "filename_prefix": a.get("filename_prefix", "StudioAssistant"), "images": ["6", 0]}}
    return g, seed, plan


def submit(graph):
    res = post_json("/prompt", {"prompt": graph, "client_id": CLIENT_ID})
    pid = res.get("prompt_id")
    if not pid:
        raise ComfyError("ComfyUI did not queue the prompt: " + _explain(res))
    return pid


def history_entry(prompt_id):
    return get_json("/history/" + prompt_id).get(prompt_id)


def wait_for(prompt_id, timeout):
    """Poll history until the prompt finishes. Returns the history entry."""
    started = time.monotonic()
    deadline = started + max(1, min(timeout, MAX_WAIT))
    while True:
        entry = history_entry(prompt_id)
        if entry and (entry.get("status", {}).get("completed") or entry.get("outputs")):
            return entry
        if entry and entry.get("status", {}).get("status_str") == "error":
            return entry
        if studio_mcp.cancelled():
            raise ComfyError("Stopped waiting for prompt %s; it stays queued on ComfyUI. "
                             "comfy_wait collects it, comfy_interrupt stops it." % prompt_id)
        studio_mcp.progress("waiting on ComfyUI for %s" % prompt_id,
                            done=int(time.monotonic() - started), total=int(deadline - started))
        if time.monotonic() > deadline:
            raise ComfyError(
                "Prompt %s is still running after %ds. It stays queued on ComfyUI; call "
                "comfy_wait with this prompt_id to collect it, or comfy_queue to look."
                % (prompt_id, timeout))
        time.sleep(1.0)


def outputs_of(entry):
    """Every file a finished prompt wrote, flattened: [{filename, subfolder, type}]."""
    files = []
    for node_out in (entry.get("outputs") or {}).values():
        for key in ("images", "gifs", "videos", "audio", "files"):
            for f in node_out.get(key, []) or []:
                if f.get("type") == "temp":
                    continue          # previews, not outputs
                files.append({"filename": f.get("filename"), "subfolder": f.get("subfolder", ""),
                              "type": f.get("type", "output")})
    return files


def status_messages(entry):
    msgs = []
    for m in entry.get("status", {}).get("messages", []) or []:
        if isinstance(m, list) and len(m) == 2 and m[0] == "execution_error":
            d = m[1] or {}
            msgs.append("%s in %s: %s" % (d.get("exception_type", "error"),
                                          d.get("node_type", "?"), d.get("exception_message", "")))
    return msgs


def fetch(f, prompt_id=""):
    """Download one output to OUTPUT_DIR; returns (local_path, bytes)."""
    q = urllib.parse.urlencode({"filename": f["filename"], "subfolder": f.get("subfolder", ""),
                                "type": f.get("type", "output")})
    data = get_bytes("/view?" + q)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stem, ext = os.path.splitext(f["filename"])
    path = os.path.join(OUTPUT_DIR, f["filename"])
    if os.path.exists(path):
        with open(path, "rb") as fh:
            same = fh.read() == data
        if not same:
            path = os.path.join(OUTPUT_DIR, "%s-%s%s" % (stem, (prompt_id or uuid.uuid4().hex)[:8], ext))
    with open(path, "wb") as fh:
        fh.write(data)
    return path, data


def collect(prompt_id, entry):
    """Outputs of a finished prompt as an MCP result: text plus inline images."""
    files = outputs_of(entry)
    errors = status_messages(entry)
    if errors and not files:
        return result("Prompt %s failed: %s" % (prompt_id, "; ".join(errors)), error=True)
    content, lines = [], []
    for f in files:
        try:
            path, data = fetch(f, prompt_id)
        except ComfyError as e:
            lines.append("%s: could not download (%s)" % (f["filename"], e))
            continue
        lines.append(path)
        if f["filename"].lower().endswith(".png") and len(content) < MAX_INLINE_IMAGES:
            content.append({"type": "image", "mimeType": "image/png",
                            "data": base64.b64encode(data).decode("ascii")})
    text = "prompt_id: %s\n" % prompt_id
    text += ("Saved %d file(s) on this workstation:\n  " % len(lines) + "\n  ".join(lines)
             if lines else "The prompt finished but wrote no output files.")
    if errors:
        text += "\nComfyUI also reported: " + "; ".join(errors)
    return result(text, images=content)


# ----------------------------------------------------------------- the tools

def result(text, images=(), error=False):
    return {"content": [{"type": "text", "text": text}] + list(images), "isError": bool(error)}


def t_status(a):
    stats = get_json("/system_stats")
    q = get_json("/queue")
    lines = ["ComfyUI at %s is up." % COMFY_URL]
    sysinfo = stats.get("system", {})
    if sysinfo.get("comfyui_version"):
        lines.append("version %s" % sysinfo["comfyui_version"])
    for d in stats.get("devices", []):
        total, free = d.get("vram_total", 0), d.get("vram_free", 0)
        lines.append("GPU %s: %.1f GB VRAM, %.1f GB free" % (d.get("name", "?"), total / 1e9, free / 1e9))
    lines.append("queue: %d running, %d pending" % (len(q.get("queue_running", [])),
                                                    len(q.get("queue_pending", []))))
    try:
        lines.append("comfy_generate will use by default: " + plan_model({})["label"])
    except ComfyError as e:
        lines.append(str(e))
    lines.append("outputs are saved to %s" % OUTPUT_DIR)
    return result("\n".join(lines))


def t_list_models(a):
    kind = a.get("kind", "checkpoints")
    names = list_models(kind)
    if not names:
        hint = ""
        if kind == "checkpoints":
            split = list_models("diffusion_models")
            if split:
                hint = (" There are split models under diffusion_models (%s); comfy_generate "
                        "uses those on its own." % ", ".join(split))
        return result("No %s installed on ComfyUI.%s" % (kind, hint))
    note = ""
    if kind == "checkpoints" and not any(is_generative_checkpoint(n) for n in names):
        note = ("\nNone of these look like an image-generation checkpoint, so "
                "comfy_generate falls back to a split model under diffusion_models.")
    return result("%s (%d):\n  " % (kind, len(names)) + "\n  ".join(names) + note)


def t_search_nodes(a):
    query = a.get("query", "").lower()
    info = get_json("/object_info")
    hits = []
    for name, node in info.items():
        hay = (name + " " + str(node.get("display_name", "")) + " " + str(node.get("category", ""))).lower()
        if query in hay:
            hits.append("%s  [%s]" % (name, node.get("category", "")))
    hits.sort()
    if not hits:
        return result("No node matches %r." % query)
    more = "" if len(hits) <= 50 else "\n  ...and %d more; narrow the query" % (len(hits) - 50)
    return result("%d node(s):\n  " % len(hits) + "\n  ".join(hits[:50]) + more)


def t_node_info(a):
    name = a["node"]
    info = get_json("/object_info/" + urllib.parse.quote(name)).get(name)
    if not info:
        return result("ComfyUI has no node called %r. Use comfy_search_nodes." % name, error=True)
    lines = ["%s  [%s]" % (name, info.get("category", ""))]
    for section in ("required", "optional"):
        for key, spec in (info.get("input", {}).get(section) or {}).items():
            typ = spec[0] if isinstance(spec, list) and spec else "?"
            opts = spec[1] if isinstance(spec, list) and len(spec) > 1 and isinstance(spec[1], dict) else {}
            if isinstance(typ, list):
                shown = "one of %d values" % len(typ) if len(typ) > 12 else "|".join(map(str, typ))
            else:
                shown = str(typ)
            extra = ", ".join("%s=%s" % (k, opts[k]) for k in ("default", "min", "max", "step") if k in opts)
            lines.append("  %s %s: %s%s" % (section, key, shown, " (" + extra + ")" if extra else ""))
    outs = info.get("output", []) or []
    names = info.get("output_name", outs) or outs
    lines.append("  outputs: " + ", ".join("%d=%s(%s)" % (i, n, o) for i, (n, o) in enumerate(zip(names, outs))))
    return result("\n".join(lines))


def t_generate(a):
    if not a.get("prompt", "").strip():
        return result("prompt is required.", error=True)
    graph, seed, plan = build_graph(a)
    k = graph["5"]["inputs"]
    recipe = "model: %s\nseed: %d  steps: %s  cfg: %s  sampler: %s/%s\n" % (
        plan["label"], seed, k["steps"], k["cfg"], k["sampler_name"], k["scheduler"])
    pid = submit(graph)
    if not a.get("wait", True):
        return result(recipe + "Queued as prompt_id %s. Call comfy_wait to collect it." % pid)
    entry = wait_for(pid, a.get("timeout", 300))
    out = collect(pid, entry)
    out["content"][0]["text"] = recipe + out["content"][0]["text"]
    return out


def t_run_workflow(a):
    graph = a.get("workflow")
    if not isinstance(graph, dict) or not graph:
        return result("workflow must be an API-format graph: an object of node_id -> "
                      "{class_type, inputs}.", error=True)
    bad = [k for k, v in graph.items() if not (isinstance(v, dict) and "class_type" in v)]
    if bad:
        return result("Not API format - nodes %s have no class_type. Export the workflow "
                      "with 'Save (API Format)' in ComfyUI, not the ordinary save."
                      % ", ".join(bad), error=True)
    pid = submit(graph)
    if not a.get("wait", True):
        return result("Queued as prompt_id %s. Call comfy_wait to collect it." % pid)
    return collect(pid, wait_for(pid, a.get("timeout", 300)))


def t_wait(a):
    pid = a["prompt_id"]
    entry = history_entry(pid)
    if entry is None:
        q = get_json("/queue")
        queued = {row[1] for row in q.get("queue_running", []) + q.get("queue_pending", []) if len(row) > 1}
        if pid not in queued:
            return result("ComfyUI has no prompt %s - not queued and not in history." % pid, error=True)
    return collect(pid, wait_for(pid, a.get("timeout", 300)))


def t_queue(a):
    q = get_json("/queue")
    lines = []
    for label in ("queue_running", "queue_pending"):
        rows = q.get(label, [])
        lines.append("%s: %d" % (label.replace("queue_", ""), len(rows)))
        for row in rows[:20]:
            pid = row[1] if len(row) > 1 else "?"
            graph = row[2] if len(row) > 2 and isinstance(row[2], dict) else {}
            kinds = sorted({n.get("class_type", "?") for n in graph.values() if isinstance(n, dict)})
            lines.append("  %s  (%d nodes: %s)" % (pid, len(graph), ", ".join(kinds[:6])))
    return result("\n".join(lines))


def t_history(a):
    limit = max(1, min(int(a.get("limit", 5)), 50))
    hist = get_json("/history?max_items=%d" % limit)
    if not hist:
        return result("ComfyUI's history is empty.")
    lines = []
    for pid, entry in list(hist.items())[-limit:]:
        status = entry.get("status", {}).get("status_str", "?")
        files = outputs_of(entry)
        lines.append("%s  %s  %d file(s): %s" % (
            pid, status, len(files), ", ".join(f["filename"] for f in files[:6])))
        for m in status_messages(entry):
            lines.append("    " + m)
    return result("\n".join(lines))


def t_fetch_output(a):
    f = {"filename": a["filename"], "subfolder": a.get("subfolder", ""), "type": a.get("type", "output")}
    path, data = fetch(f)
    images = []
    if f["filename"].lower().endswith(".png"):
        images.append({"type": "image", "mimeType": "image/png",
                       "data": base64.b64encode(data).decode("ascii")})
    return result("Saved to %s (%d bytes)" % (path, len(data)), images=images)


def t_upload_image(a):
    path = a["path"]
    if not os.path.isfile(path):
        return result("No file at %s on this workstation." % path, error=True)
    with open(path, "rb") as fh:
        data = fh.read()
    res = post_multipart("/upload/image", {"overwrite": "true"}, os.path.basename(path), data)
    name = res.get("name", os.path.basename(path))
    if res.get("subfolder"):
        name = res["subfolder"] + "/" + name
    return result("Uploaded as %r - pass that as init_image to comfy_generate, or as a "
                  "LoadImage input." % name)


def t_interrupt(a):
    post_json("/interrupt", {})
    return result("Asked ComfyUI to interrupt the running prompt.")


def t_clear_queue(a):
    ids = a.get("prompt_ids") or []
    if ids:
        post_json("/queue", {"delete": ids})
        return result("Removed %d pending prompt(s) from the queue." % len(ids))
    post_json("/queue", {"clear": True})
    return result("Cleared every pending prompt. The one already running continues; "
                  "use comfy_interrupt to stop it.")


def _obj(props, required=()):
    return {"type": "object", "properties": props, "required": list(required),
            "additionalProperties": False}


def _s(desc, **kw):
    d = {"type": "string", "description": desc}
    d.update(kw)
    return d


def _i(desc, **kw):
    d = {"type": "integer", "description": desc}
    d.update(kw)
    return d


def _n(desc, **kw):
    d = {"type": "number", "description": desc}
    d.update(kw)
    return d


TOOLS = [
    ("comfy_status", t_status,
     "Is ComfyUI reachable, what GPU and VRAM it has, how long its queue is, and "
     "where outputs are saved on this workstation. Call this first if a tool "
     "reports it cannot reach ComfyUI.",
     _obj({})),
    ("comfy_list_models", t_list_models,
     "List the model files installed on ComfyUI of one kind: checkpoints (all-in-one "
     "base models), diffusion_models and text_encoders and vae (the three parts of a "
     "split model such as Z-Image, Qwen-Image or Flux), loras, controlnet, "
     "upscale_models, embeddings. Names come back exactly as the loader nodes want them.",
     _obj({"kind": _s("Which model folder to list. Default: checkpoints.", enum=MODEL_KINDS)})),
    ("comfy_search_nodes", t_search_nodes,
     "Find ComfyUI node types by a substring of their name or category, e.g. 'upscale', "
     "'controlnet', 'LoadImage'. Use before building a custom workflow.",
     _obj({"query": _s("Case-insensitive substring to look for.")}, ["query"])),
    ("comfy_node_info", t_node_info,
     "The inputs (with types and defaults) and outputs of one node type, by its exact "
     "class name such as KSampler or LoadImage.",
     _obj({"node": _s("Exact node class_type.")}, ["node"])),
    ("comfy_generate", t_generate,
     "Generate images with the standard graph (model -> prompt and negative -> "
     "KSampler -> VAE decode -> save). With no model named it uses the first "
     "checkpoint, or - when there are none - the first split model with the right "
     "recipe for its family (Z-Image Turbo: 8 steps, cfg 1, res_multistep/simple). "
     "Text-to-image by default; give init_image (a name from comfy_upload_image) for "
     "image-to-image. Blocks until done, downloads the results to this workstation "
     "and returns their paths plus the seed and settings used, so a result can be "
     "reproduced.",
     _obj({
         "prompt": _s("What to draw. Comma-separated descriptive phrases work best."),
         "negative": _s("What to avoid, e.g. 'blurry, text, watermark'. Default empty."),
         "checkpoint": _s("All-in-one checkpoint filename from comfy_list_models. Default: "
                          "the first installed, else a split model."),
         "diffusion_model": _s("Split model: filename from comfy_list_models "
                               "kind=diffusion_models. Picks the family's recipe."),
         "text_encoder": _s("Split model: text encoder filename from kind=text_encoders. "
                            "Default: chosen for the family."),
         "text_encoder_type": _s("Split model: the encoder family. Default: chosen for the "
                                 "family (lumina2 for Z-Image).", enum=ENCODER_TYPES),
         "vae": _s("Split model: VAE filename from kind=vae. Default: the family's."),
         "shift": _n("Split model: sampling shift. Default: the family's (3.0 for Z-Image).",
                     minimum=0, maximum=100),
         "width": _i("Pixels, multiple of 16. Default 1024.", minimum=64, maximum=4096),
         "height": _i("Pixels, multiple of 16. Default 1024.", minimum=64, maximum=4096),
         "steps": _i("Sampling steps. Default: the model's recipe (20 for a checkpoint, "
                     "8 for Z-Image Turbo).", minimum=1, maximum=150),
         "cfg": _n("Prompt adherence. Default: the model's recipe (6 for a checkpoint, 1 "
                   "for Z-Image Turbo, where the negative is then ignored).", minimum=0, maximum=30),
         "seed": _i("Fixed seed to reproduce or vary a result. Default: random.", minimum=-1),
         "sampler": _s("Sampler name. Default: the model's recipe.", enum=SAMPLERS),
         "scheduler": _s("Scheduler name. Default: the model's recipe.", enum=SCHEDULERS),
         "batch_size": _i("Images per call. Default 1.", minimum=1, maximum=8),
         "init_image": _s("Uploaded image name for image-to-image. Omit for text-to-image."),
         "denoise": _n("Image-to-image only: how much to change the init image, 0..1. "
                       "Default 0.6.", minimum=0, maximum=1),
         "lora": _s("LoRA filename from comfy_list_models kind=loras, applied to the checkpoint."),
         "lora_strength": _n("LoRA strength. Default 1.0.", minimum=-2, maximum=2),
         "filename_prefix": _s("Output filename prefix. Default StudioAssistant."),
         "wait": {"type": "boolean", "description": "Default true. false returns the "
                                                    "prompt_id at once; collect it with comfy_wait."},
         "timeout": _i("Seconds to wait before giving the prompt_id back. Default 300.",
                       minimum=5, maximum=MAX_WAIT),
     }, ["prompt"])),
    ("comfy_run_workflow", t_run_workflow,
     "Run any ComfyUI workflow in API format: an object of node_id -> {class_type, "
     "inputs}, where a link is [node_id, output_index]. This is what 'Save (API "
     "Format)' in ComfyUI exports. Use comfy_search_nodes and comfy_node_info to get "
     "input names right. Blocks until done and downloads the outputs like "
     "comfy_generate.",
     _obj({"workflow": {"type": "object", "description": "The API-format graph.",
                        "additionalProperties": True},
           "wait": {"type": "boolean", "description": "Default true."},
           "timeout": _i("Seconds to wait. Default 300.", minimum=5, maximum=MAX_WAIT)},
          ["workflow"])),
    ("comfy_wait", t_wait,
     "Wait for a queued prompt_id to finish and download its outputs to this workstation.",
     _obj({"prompt_id": _s("The prompt_id a generate call returned."),
           "timeout": _i("Seconds to wait. Default 300.", minimum=5, maximum=MAX_WAIT)},
          ["prompt_id"])),
    ("comfy_queue", t_queue,
     "What ComfyUI is running now and what is waiting, with prompt_ids.",
     _obj({})),
    ("comfy_history", t_history,
     "The most recent finished prompts and the files they produced, newest last.",
     _obj({"limit": _i("How many. Default 5, max 50.", minimum=1, maximum=50)})),
    ("comfy_fetch_output", t_fetch_output,
     "Download one file from ComfyUI's output folder to this workstation by its "
     "filename, as listed by comfy_history.",
     _obj({"filename": _s("Output filename, e.g. StudioAssistant_00003_.png."),
           "subfolder": _s("Subfolder if comfy_history showed one. Default none."),
           "type": _s("Default output.", enum=["output", "input", "temp"])},
          ["filename"])),
    ("comfy_upload_image", t_upload_image,
     "Send an image from this workstation to ComfyUI so it can be used as init_image "
     "in comfy_generate or in a LoadImage node. Returns the name to use.",
     _obj({"path": _s("Full path of a PNG or JPEG on this workstation.")}, ["path"])),
    ("comfy_interrupt", t_interrupt,
     "Stop the prompt ComfyUI is running right now. Pending prompts stay queued.",
     _obj({})),
    ("comfy_clear_queue", t_clear_queue,
     "Remove pending prompts: the ones listed, or all of them when none are given. "
     "Does not stop the running prompt.",
     _obj({"prompt_ids": {"type": "array", "items": {"type": "string"},
                          "description": "prompt_ids to remove. Omit to clear all pending."}})),
]

TOOLS_BY_NAME = {name: (fn, desc, schema) for name, fn, desc, schema in TOOLS}

# Tools that change nothing on ComfyUI. The executor reads this hint to decide
# which calls need a read-back; without it every comfy_ tool counts as an edit
# and the model is nagged to "inspect" after asking which models are installed.
# comfy_wait and comfy_fetch_output only observe a finished run and copy its
# files home, so they are the read-back for a generate that did not wait.
READ_ONLY = {"comfy_status", "comfy_list_models", "comfy_search_nodes", "comfy_node_info",
             "comfy_queue", "comfy_history", "comfy_wait", "comfy_fetch_output"}


# Hints past read-only, for the tools whose default would mislead: a generate
# adds files and deletes nothing, while interrupt and clear_queue throw work
# away. The rest are reads (READ_ONLY) or writes with the spec's defaults.
HINTS = {
    "comfy_generate": {"destructive": False},
    "comfy_run_workflow": {"destructive": False},
    "comfy_upload_image": {"destructive": False, "idempotent": True},
    "comfy_interrupt": {"destructive": True, "idempotent": True},
    "comfy_clear_queue": {"destructive": True, "idempotent": True},
}

SERVER = studio_mcp.Server(
    "studio-comfy-mcp", "1.1",
    studio_mcp.tools_from_table(TOOLS, read_only=READ_ONLY, **HINTS),
    errors=(ComfyError, KeyError, TypeError, ValueError),
    instructions="ComfyUI on %s. comfy_generate is the ordinary path: it builds the "
                 "graph, waits, and returns the picture. Call comfy_status first if a "
                 "tool reports it cannot reach ComfyUI." % COMFY_URL)


def tool_list():
    return [t.spec() for t in SERVER.tools]


def call_tool(name, arguments):
    """Call a tool from Python: every refusal is a result, never an exception."""
    try:
        return SERVER.call_tool(name, arguments)
    except studio_mcp.JSONRPCError as e:
        return result(e.message, error=True)


def serve(inp=None, out=None):
    SERVER.serve(inp, out)


if __name__ == "__main__":
    sys.exit(studio_mcp.main(SERVER))
