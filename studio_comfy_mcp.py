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
MAX_WAIT = 1200              # seconds a generate/wait call will block for
DEFAULT_WAIT = 900           # ...when the caller does not say. A starved GPU takes minutes.
# Below this much free VRAM, ComfyUI streams model weights from system RAM every
# step. On the LLM PC that is what LM Studio's resident models leave it, and it
# is most of the time a picture takes; each result says so rather than hide it.
LOW_VRAM = 8e9
# The text encoder runs once per picture; the diffusion model runs every step.
# On the shared 24 GB card, keeping the 8 GB encoder in VRAM beside the 12 GB
# model and the tab's LLM is what starved the rest, so it runs on the CPU from
# the LLM PC's 128 GB of RAM. COMFYUI_ENCODER_ON_GPU=1 puts it back.
ENCODER_ON_CPU = os.environ.get("COMFYUI_ENCODER_ON_GPU", "") not in ("1", "true", "yes")
# ComfyUI keeps its last run's models on the GPU until it needs the room for
# its own next run - and it cannot see LM Studio beside it on the same card.
# Measured after a Z-Image render: 11.7 GB still held and reported nowhere,
# the 30B decoding at 25-30 tokens a second instead of 73-81, the tab's 9B
# taking 17 s to load instead of 3 and the vision model ~30 s instead of 5.
# So a finished run with nothing queued behind it frees the GPU (`release`).
# It costs the next render nothing: 24.3 s with the models kept, 23.4 s after
# a free, and the memory is back in about a second. COMFYUI_KEEP_MODELS=1
# keeps them, for a ComfyUI with a card of its own.
KEEP_MODELS = os.environ.get("COMFYUI_KEEP_MODELS", "") in ("1", "true", "yes")
RELEASE_WAIT = 10            # seconds a finished run waits for the GPU to be let go
RELEASE_SLACK = 1.5e9        # ...until ComfyUI's free VRAM is this close to where it was
TILES = {"tile_size": 1024, "overlap": 128, "temporal_size": 64, "temporal_overlap": 8}
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
ENCODER_TYPES = ["lumina2", "krea2", "qwen_image", "flux2", "chroma", "sd3", "wan", "hidream",
                 "pixart", "cosmos", "ltxv", "mochi", "hunyuan_image", "stable_diffusion"]

# Split models - a diffusion model, a text encoder and a VAE as three files
# rather than one checkpoint - each want their own sampling recipe, and the
# wrong one gives noise rather than an error. Matched on the diffusion model's
# filename, first hit wins; anything the caller passes overrides. The recipes
# are ComfyUI's own templates for each model (image_z_image_turbo,
# image_krea2_turbo_t2i_int8, image_qwen_image_edit_2509).
#
# Order is also preference: with nothing named, text-to-image uses the first
# family installed, so the most photographic model leads. `edit` models take a
# picture and an instruction; they are never the text-to-image default - the
# Qwen edit model was, and made every "draw a duck" a slow, soft 20-step
# double-CFG render. `hires` is the detail pass (see build_graph).
FAMILIES = [
    ("z_image", {"label": "Z-Image Turbo", "encoder_type": "lumina2", "encoder": "qwen_3",
                 "vae": "ae.safetensors", "shift": 3.0, "steps": 8, "cfg": 1.0,
                 "sampler": "res_multistep", "scheduler": "simple",
                 "latent": "EmptySD3LatentImage", "photo": True,
                 "hires": {"scale": 1.5, "denoise": 0.33, "steps": 5,
                           "sampler": "dpmpp_2m_sde", "scheduler": "beta"}}),
    ("krea2", {"label": "Krea 2 Turbo", "encoder_type": "krea2", "encoder": "qwen3vl",
               "vae": "qwen_image_vae.safetensors", "shift": None, "steps": 8, "cfg": 1.0,
               "sampler": "euler", "scheduler": "simple", "latent": "EmptyLatentImage",
               "hires": {"scale": 1.5, "denoise": 0.3, "steps": 6,
                         "sampler": "euler", "scheduler": "simple"}}),
    ("qwen_image_edit", {"label": "Qwen-Image-Edit", "edit": True, "encoder_type": "qwen_image",
                         "encoder": "qwen_2.5_vl", "vae": "qwen_image_vae.safetensors",
                         "shift": 3.0, "steps": 20, "cfg": 4.0, "sampler": "euler",
                         "scheduler": "simple", "lightning": "lightning", "fast_steps": 4}),
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

# Realism. A photographic prompt gets a closing sentence that holds the model to
# a camera's rendering - the thing every "make it realistic" request is after -
# unless it already names another medium. And a checkpoint run at real CFG gets
# a negative against the looks that read as generated, when none is given.
# Z-Image and Krea at cfg 1 ignore a negative, so for them the sentence is it.
PHOTO_SUFFIX = ("Photorealistic photograph with true-to-life colour, natural light and "
                "shadow, real-world textures and fine detail, sharp focus.")
PHOTO_NEGATIVE = ("illustration, painting, drawing, cartoon, anime, CGI, 3d render, "
                  "plastic skin, airbrushed, oversaturated, overexposed, blurry, lowres, "
                  "jpeg artifacts, deformed hands, extra fingers, watermark, text")
NOT_PHOTO = ("illustration", "painting", "painted", "drawing", "sketch", "cartoon", "anime",
             "manga", "comic", "watercolor", "watercolour", "oil paint", "vector", "logo",
             "icon", "pixel art", "3d render", "low poly", "claymation", "line art",
             "charcoal", "pencil", "ink ", "sticker", "flat design", "isometric")


def wants_photo(prompt):
    low = prompt.lower()
    return not any(k in low for k in NOT_PHOTO)


class ComfyError(Exception):
    """Anything ComfyUI, or the network in front of it, refuses."""


class Unreachable(ComfyError):
    """No answer at all. While ComfyUI stages a 20 GB model through system RAM
    its HTTP server stops answering for half a minute or more, so a wait
    treats this as "busy", not "gone"."""


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
        raise Unreachable(
            "Cannot reach ComfyUI at %s (%s). It has to be running on the LLM PC, "
            "started with --listen so it accepts connections from this machine."
            % (COMFY_URL, e.reason))
    except OSError as e:
        raise Unreachable("Cannot reach ComfyUI at %s (%s)." % (COMFY_URL, e))


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


def is_edit_model(name):
    return bool(family_of(name).get("edit"))


def family_rank(name):
    low = (name or "").lower()
    for i, (key, _) in enumerate(FAMILIES):
        if key in low:
            return i
    return len(FAMILIES)


def plan_model(a):
    """
    What to load, when the caller names nothing. A split model with a known
    recipe comes first, the most photographic family first - Z-Image Turbo
    out-draws any SD-era checkpoint for realism, at 8 steps. Then a real
    all-in-one checkpoint; one whose name is a non-generative model (a lone
    sam3.1, an upscaler) is skipped, because it cannot make a picture and
    picking it fails every request. Edit models are never the default: they
    need a picture to edit. Explicit `checkpoint`/`diffusion_model` always win.
    """
    if a.get("checkpoint"):
        return {"kind": "checkpoint", "checkpoint": a["checkpoint"], "label": a["checkpoint"]}
    unet = a.get("diffusion_model")
    if not unet:
        ckpts = list_models("checkpoints")
        unets = [u for u in (list_models("diffusion_models") or list_models("unet"))
                 if not is_edit_model(u)]
        good = [c for c in ckpts if is_generative_checkpoint(c)]
        known = sorted((u for u in unets if is_known_family(u)), key=family_rank)
        if known:                     # a split model we have a working recipe for
            unet = known[0]
        elif good:                    # a plausible all-in-one checkpoint
            return {"kind": "checkpoint", "checkpoint": good[0], "label": good[0]}
        elif unets:                   # any diffusion model, generic recipe
            unet = unets[0]
        elif ckpts:                   # last resort: a checkpoint that looked unusable
            return {"kind": "checkpoint", "checkpoint": ckpts[0], "label": ckpts[0]}
        else:
            raise ComfyError("ComfyUI has no checkpoints and no text-to-image diffusion "
                             "models installed; nothing can generate.")
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


def new_seed(a):
    seed = a.get("seed")
    return random.randint(0, 2**32 - 1) if seed is None or seed < 0 else seed


def load_split(g, plan, ids=("1", "10", "11", "12"), lora=None):
    """UNET (+ LoRA) + text encoder + VAE (+ shift) into g, under `ids`;
    returns the (model, clip, vae) links."""
    unet, enc, vae, shift = ids
    g[unet] = {"class_type": "UNETLoader", "inputs": {
        "unet_name": plan["diffusion_model"], "weight_dtype": "default"}}
    g[enc] = {"class_type": "CLIPLoader", "inputs": {
        "clip_name": plan["text_encoder"], "type": plan["encoder_type"]}}
    if ENCODER_ON_CPU:
        g[enc]["inputs"]["device"] = "cpu"
    g[vae] = {"class_type": "VAELoader", "inputs": {"vae_name": plan["vae"]}}
    model = [unet, 0]
    if lora:
        g["9"] = {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": model, "lora_name": lora, "strength_model": 1.0}}
        model = ["9", 0]
    if plan.get("shift") is not None:
        g[shift] = {"class_type": "ModelSamplingAuraFlow",
                    "inputs": {"model": model, "shift": plan["shift"]}}
        model = [shift, 0]
    return model, [enc, 0], [vae, 0]


def detail_pass(g, pixels, model, positive, negative, vae, recipe, seed, first="13"):
    """
    The hi-res detail pass: upscale the decoded picture with lanczos, encode it
    and resample the last third of the schedule at the larger size. The model
    redraws pores, fabric weave, hair and foliage at the resolution they are
    seen at, instead of the soft, smeared micro-detail a 1-megapixel sample
    leaves - the single biggest step from "AI picture" to photograph. This is
    ComfyUI's own Z-Image upscaler recipe, less its ESRGAN model. Node ids
    start at `first`; returns the link to the final pixels.
    """
    n = int(first)
    up, enc, samp, dec = (str(n + i) for i in range(4))
    g[up] = {"class_type": "ImageScaleBy", "inputs": {
        "image": pixels, "upscale_method": "lanczos", "scale_by": recipe["scale"]}}
    # Tiled at this size: a whole 2-4 MP frame through the VAE beside a
    # resident diffusion model and an LLM ran ComfyUI out of VRAM, and its
    # fallback sat for six minutes on a pass that takes seconds in tiles.
    g[enc] = {"class_type": "VAEEncodeTiled", "inputs": dict(TILES, pixels=[up, 0], vae=vae)}
    g[samp] = {"class_type": "KSampler", "inputs": {
        "seed": seed, "steps": recipe["steps"], "cfg": 1.0,
        "sampler_name": recipe["sampler"], "scheduler": recipe["scheduler"],
        "denoise": recipe["denoise"], "model": model, "positive": positive,
        "negative": negative, "latent_image": [enc, 0]}}
    g[dec] = {"class_type": "VAEDecodeTiled", "inputs": dict(TILES, samples=[samp, 0], vae=vae)}
    return [dec, 0]


def hires_recipe(a, plan, w, h):
    """The detail pass to run, or None. On by default for a family that has
    one; `hires` false turns it off, `hires_scale` sizes it. The scale is
    capped so the pass never samples more than ~4.2 megapixels, where a
    24 GB card runs out and the models start to repeat themselves."""
    base = plan.get("hires")
    if not base or not a.get("hires", True) or a.get("init_image"):
        return None
    r = dict(base)
    if a.get("hires_scale"):
        r["scale"] = a["hires_scale"]
    r["scale"] = round(min(r["scale"], (4.2e6 / float(w * h)) ** 0.5), 3)
    if r["scale"] <= 1.05:
        return None
    if a.get("hires_denoise") is not None:
        r["denoise"] = a["hires_denoise"]
    return r


def input_image(ref):
    """
    A picture for a LoadImage node: a path on this workstation is uploaded
    first, anything else is taken to be a name already in ComfyUI's input
    folder. Saves the model a comfy_upload_image round trip per edit.
    """
    path = studio_mcp.local_path(ref)
    if os.path.isfile(path):
        return upload(path)
    if os.path.isabs(path) or ("\\" in path and ":" in path):
        raise ComfyError("No file at %s on this workstation." % path)
    return ref


def build_graph(a):
    """
    The canonical txt2img graph, with img2img, one LoRA and the detail pass as
    options. Node ids are strings; a link is [node_id, output_index].
    Returns (graph, seed, plan, notes) - notes say what was added on the
    caller's behalf, so the result can show it.
    """
    seed = new_seed(a)
    plan = plan_model(a)
    notes = []
    if plan["kind"] == "checkpoint":
        g = {"1": {"class_type": "CheckpointLoaderSimple",
                   "inputs": {"ckpt_name": plan["checkpoint"]}}}
        model, clip, vae = ["1", 0], ["1", 1], ["1", 2]
        latent_node = "EmptyLatentImage"
        defaults = {"steps": 20, "cfg": 6.0, "sampler": "euler", "scheduler": "normal"}
    else:
        g = {}
        model, clip, vae = load_split(g, plan)
        latent_node = plan.get("latent", "EmptySD3LatentImage")
        defaults = plan
    if a.get("lora"):
        g["9"] = {"class_type": "LoraLoader", "inputs": {
            "lora_name": a["lora"],
            "strength_model": a.get("lora_strength", 1.0),
            "strength_clip": a.get("lora_strength", 1.0),
            "model": model, "clip": clip}}
        model, clip = ["9", 0], ["9", 1]
    cfg = a.get("cfg", defaults["cfg"])
    prompt, negative = a["prompt"].strip(), a.get("negative", "")
    photo = a.get("realism", True) and wants_photo(prompt)
    if photo:
        prompt = prompt.rstrip(" .,;") + ". " + PHOTO_SUFFIX
        notes.append("realism: on")
        if not negative and cfg > 1.0:
            negative = PHOTO_NEGATIVE
    g["2"] = {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": clip}}
    if cfg <= 1.0:
        # At cfg 1 the sampler never reads the negative; a zeroed positive is
        # what the model's own template passes, and saves an encode.
        g["3"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["2", 0]}}
    else:
        g["3"] = {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": clip}}
    denoise = 1.0
    w, h = a.get("width", 1024), a.get("height", 1024)
    if a.get("init_image"):
        g["8"] = {"class_type": "LoadImage", "inputs": {"image": input_image(a["init_image"])}}
        g["4"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["8", 0], "vae": vae}}
        denoise = a.get("denoise", 0.6)
    else:
        g["4"] = {"class_type": latent_node, "inputs": {
            "width": w, "height": h, "batch_size": a.get("batch_size", 1)}}
    g["5"] = {"class_type": "KSampler", "inputs": {
        "seed": seed, "steps": a.get("steps", defaults["steps"]), "cfg": cfg,
        "sampler_name": a.get("sampler", defaults["sampler"]),
        "scheduler": a.get("scheduler", defaults["scheduler"]),
        "denoise": denoise, "model": model, "positive": ["2", 0], "negative": ["3", 0],
        "latent_image": ["4", 0]}}
    g["6"] = {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": vae}}
    out = ["6", 0]
    hr = hires_recipe(a, plan, w, h)
    if hr:
        out = detail_pass(g, out, model, ["2", 0], ["3", 0], vae, hr, seed)
        notes.append("detail pass: x%s at denoise %s -> %dx%d" % (
            hr["scale"], hr["denoise"], int(w * hr["scale"]), int(h * hr["scale"])))
    g["7"] = {"class_type": "SaveImage", "inputs": {
        "filename_prefix": a.get("filename_prefix", "StudioAssistant"), "images": out}}
    return g, seed, plan, notes


def submit(graph):
    res = post_json("/prompt", {"prompt": graph, "client_id": CLIENT_ID})
    pid = res.get("prompt_id")
    if not pid:
        raise ComfyError("ComfyUI did not queue the prompt: " + _explain(res))
    return pid


def history_entry(prompt_id):
    return get_json("/history/" + prompt_id, timeout=30).get(prompt_id)


def gpu_memory():
    """(free, total) VRAM in bytes as ComfyUI sees it, or (None, None).

    ComfyUI's own view of the card: on the LLM PC it does not see what LM
    Studio holds - with the 30B loaded it still reported 22 GB free - so a low
    figure is models ComfyUI itself is holding."""
    try:
        devices = get_json("/system_stats").get("devices", [])
    except ComfyError:
        return None, None
    for d in devices:
        return d.get("vram_free", 0), d.get("vram_total", 0)
    return None, None


def vram_note(free, total):
    """A sentence when ComfyUI's GPU is too full to hold a model, else ''."""
    if total and free is not None and free < LOW_VRAM:
        return ("ComfyUI's GPU had %.1f of %.0f GB free when this started - models loaded "
                "before it hold the rest - so model weights streamed from system RAM, "
                "which is most of the time this took." % (free / 1e9, total / 1e9))
    return ""


def run(graph, a, header, notes=(), started=None):
    """Submit, wait and collect, with the recipe and timing on top. `started`
    is when the work began, for a tool that ran something before this."""
    free, total = gpu_memory()
    note = vram_note(free, total)
    started = started or time.monotonic()
    pid = submit(graph)
    if not a.get("wait", True):
        return result(header + "Queued as prompt_id %s. Call comfy_wait to collect it." % pid)
    entry = wait_for(pid, a.get("timeout", DEFAULT_WAIT))
    out = collect(pid, entry)
    release(free)
    extra = "".join("%s\n" % n for n in notes)
    extra += "took %ds\n" % (time.monotonic() - started)
    if note:
        extra += note + "\n"
    out["content"][0]["text"] = header + extra + out["content"][0]["text"]
    return out


def release(baseline=None):
    """Unload ComfyUI's models from the GPU once it has nothing left to run,
    so the LLM host's models get the card back (see KEEP_MODELS). With the
    free VRAM ComfyUI saw before the run as `baseline`, wait - RELEASE_WAIT
    at most, about a second in practice - until it is back near that: the
    tab's model is loaded again the moment this returns, and a load racing
    the release met the very contention this is here to end. Best effort: a
    server that will not say, or will not free, costs speed only.
    -> True when it was asked to."""
    if KEEP_MODELS:
        return False
    try:
        q = get_json("/queue", timeout=5)
        if q.get("queue_running") or q.get("queue_pending"):
            return False                  # the next run wants the models
        post_json("/free", {"unload_models": True, "free_memory": True}, timeout=10)
    except ComfyError:
        return False
    if baseline:
        deadline = time.monotonic() + RELEASE_WAIT
        while time.monotonic() < deadline:
            free, _ = gpu_memory()
            if free is None or free >= baseline - RELEASE_SLACK:
                break
            time.sleep(0.25)
    return True


def wait_for(prompt_id, timeout):
    """Poll history until the prompt finishes. Returns the history entry."""
    started = time.monotonic()
    deadline = started + max(1, min(timeout, MAX_WAIT))
    silent_since = None
    while True:
        try:
            entry = history_entry(prompt_id)
            silent_since = None
        except Unreachable:
            # Busy staging a model, most likely: keep waiting, and say so.
            silent_since = silent_since or time.monotonic()
            entry = None
            if time.monotonic() > deadline:
                raise ComfyError(
                    "ComfyUI has not answered for %ds while running prompt %s. If it is "
                    "still up it keeps the prompt; comfy_wait collects it."
                    % (time.monotonic() - silent_since, prompt_id))
        if entry and (entry.get("status", {}).get("completed") or entry.get("outputs")):
            return entry
        if entry and entry.get("status", {}).get("status_str") == "error":
            return entry
        if studio_mcp.cancelled():
            raise ComfyError("Stopped waiting for prompt %s; it stays queued on ComfyUI. "
                             "comfy_wait collects it, comfy_interrupt stops it." % prompt_id)
        studio_mcp.progress(
            ("ComfyUI is busy (loading models?) on %s" if silent_since else
             "waiting on ComfyUI for %s") % prompt_id,
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
        if isinstance(m, list) and len(m) == 2 and m[0] == "execution_interrupted":
            msgs.append("interrupted (comfy_interrupt, or stopped on ComfyUI) before it "
                        "finished")
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
            # _meta.path is where the picture is on this workstation, so the
            # chat window can save, open and show the file it previews.
            content.append({"type": "image", "mimeType": "image/png",
                            "data": base64.b64encode(data).decode("ascii"),
                            "_meta": {"path": path}})
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
    graph, seed, plan, notes = build_graph(a)
    k = graph["5"]["inputs"]
    recipe = "model: %s\nseed: %d  steps: %s  cfg: %s  sampler: %s/%s\n" % (
        plan["label"], seed, k["steps"], k["cfg"], k["sampler_name"], k["scheduler"])
    return run(graph, a, recipe, notes)


def plan_edit(a):
    """The installed edit model, its encoder, VAE and (for fast) Lightning LoRA."""
    unets = list_models("diffusion_models") or list_models("unet")
    edits = [u for u in unets if is_edit_model(u)]
    unet = a.get("edit_model") or (edits[-1] if edits else None)
    if not unet:
        raise ComfyError("ComfyUI has no image-edit model installed (such as "
                         "qwen_image_edit_2509). comfy_generate with init_image restyles a "
                         "picture instead, but cannot follow an instruction.")
    fam = family_of(unet)
    if not fam.get("edit"):
        fam = dict(FAMILIES[[k for k, _ in FAMILIES].index("qwen_image_edit")][1])
    encoders = list_models("text_encoders") or list_models("clip")
    vaes = list_models("vae")
    plan = dict(fam, kind="split", diffusion_model=unet,
                text_encoder=pick_first(encoders, fam["encoder"]),
                vae=fam["vae"] if fam["vae"] in vaes else pick_first(vaes))
    if not plan["text_encoder"] or not plan["vae"]:
        raise ComfyError("%s needs a text encoder and a VAE on ComfyUI; found encoders %s "
                         "and VAEs %s." % (unet, encoders, vaes))
    plan["lora"] = None
    if a.get("fast", True):
        loras = [l for l in list_models("loras")
                 if fam["lightning"] in l.lower() and "edit" in l.lower()]
        plan["lora"] = loras[-1] if loras else None
    return plan


def photo_finish():
    """The Z-Image plan the detail pass borrows to finish an edit, or None."""
    unets = list_models("diffusion_models") or list_models("unet")
    photo = [u for u in unets if family_of(u).get("photo")]
    return plan_model({"diffusion_model": photo[0]}) if photo else None


def t_edit_image(a):
    instruction = a.get("instruction", "").strip()
    if not instruction:
        return result("instruction is required.", error=True)
    refs = [a["image"]] + list(a.get("references") or [])[:2]
    plan = plan_edit(a)
    seed = new_seed(a)
    g = {}
    model, clip, vae = load_split(g, plan, lora=plan["lora"])
    g["13"] = {"class_type": "CFGNorm", "inputs": {"model": model, "strength": 1.0}}
    model = ["13", 0]
    names = [input_image(r) for r in refs]
    # The picture being edited is scaled to the ~1 MP the model was trained at;
    # the latent starts from it, so the untouched parts come back unchanged.
    g["20"] = {"class_type": "LoadImage", "inputs": {"image": names[0]}}
    g["21"] = {"class_type": "FluxKontextImageScale", "inputs": {"image": ["20", 0]}}
    images = {"image1": ["21", 0]}
    for i, name in enumerate(names[1:], start=2):
        g[str(20 + i)] = {"class_type": "LoadImage", "inputs": {"image": name}}
        images["image%d" % i] = [str(20 + i), 0]
    g["2"] = {"class_type": "TextEncodeQwenImageEditPlus",
              "inputs": dict(images, clip=clip, vae=vae, prompt=instruction)}
    g["3"] = {"class_type": "TextEncodeQwenImageEditPlus",
              "inputs": dict(images, clip=clip, vae=vae, prompt="")}
    g["4"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["21", 0], "vae": vae}}
    fast = bool(plan["lora"])
    steps = a.get("steps", plan["fast_steps"] if fast else plan["steps"])
    cfg = a.get("cfg", 1.0 if fast else plan["cfg"])
    g["5"] = {"class_type": "KSampler", "inputs": {
        "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": plan["sampler"],
        "scheduler": plan["scheduler"], "denoise": 1.0, "model": model,
        "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0]}}
    g["6"] = {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": vae}}
    notes = []
    header = "model: %s%s\nseed: %d  steps: %s  cfg: %s\nedited: %s\n" % (
        plan["label"], " + " + plan["lora"] if plan["lora"] else "", seed, steps, cfg,
        ", ".join(names))
    z = photo_finish() if a.get("photo_finish", False) else None
    if a.get("photo_finish", False) and not z:
        notes.append("photo finish skipped: no Z-Image model installed")
    if z:
        return finish_edit(g, ["6", 0], z, instruction, seed, a, header, notes)
    g["7"] = {"class_type": "SaveImage", "inputs": {
        "filename_prefix": a.get("filename_prefix", "StudioEdit"), "images": ["6", 0]}}
    return run(g, a, header, notes)


def finish_edit(g, edited, z, instruction, seed, a, header, notes):
    """An edit and its photo finish - Z-Image redrawing the skin, cloth and
    grain the edit model leaves waxy, at 1.5x the size - as two runs, not one
    graph. In one graph ComfyUI kept the 19.5 GB edit model on the 24 GB card
    while it brought Z-Image's 11.7 GB in beside it: the finish sat in "Model
    Initializing" for minutes and then sampled at 7 s a step, 410 s for one
    edit. So the edit runs to a preview, the GPU is freed, and a second run
    loads that preview for the detail pass. A photo finish always waits."""
    started = time.monotonic()
    free, _ = gpu_memory()
    g["7"] = {"class_type": "PreviewImage", "inputs": {"images": edited}}
    pid = submit(g)
    entry = wait_for(pid, a.get("timeout", DEFAULT_WAIT))
    previews = [f for node in (entry.get("outputs") or {}).values()
                for f in node.get("images", []) or [] if f.get("type") == "temp"]
    if not previews:
        out = collect(pid, entry)         # the edit failed; ComfyUI says why
        release(free)
        return out
    release(free)
    f = previews[0]
    name = "%s/%s" % (f["subfolder"], f["filename"]) if f.get("subfolder") else f["filename"]
    g2 = {}
    zmodel, zclip, zvae = load_split(g2, z)
    g2["8"] = {"class_type": "LoadImage", "inputs": {"image": name + " [temp]"}}
    g2["2"] = {"class_type": "CLIPTextEncode", "inputs": {
        "text": instruction + ". " + PHOTO_SUFFIX, "clip": zclip}}
    g2["3"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["2", 0]}}
    r = dict(z["hires"], denoise=a.get("finish_denoise", 0.25))
    out = detail_pass(g2, ["8", 0], zmodel, ["2", 0], ["3", 0], zvae, r, seed)
    g2["7"] = {"class_type": "SaveImage", "inputs": {
        "filename_prefix": a.get("filename_prefix", "StudioEdit"), "images": out}}
    notes = list(notes) + ["photo finish: Z-Image detail pass x%s at denoise %s"
                           % (r["scale"], r["denoise"])]
    return run(g2, dict(a, wait=True), header, notes, started=started)


SAM3 = "sam3"                # the checkpoint that finds faces and heads
FACE_EDIT = 1024             # side each face crop is edited at
# Name no accessory here: listing "glasses", "facial hair" or "the hat" had the
# model add them - a hat and a moustache on a woman who had neither.
SWAP_INSTRUCTION = (
    "Replace the face and hair of the person in picture 1 with the face and hair of "
    "the person in picture 2, so that it is unmistakably the person from picture 2. "
    "Keep the position, angle and tilt of the head and everything else in picture 1 "
    "unchanged, and add nothing that is not already in picture 1 or picture 2. Match "
    "picture 1's lighting, colours, softness and film grain.")


def find_faces(names, prompt="face:8", threshold=0.3):
    """SAM3's face boxes and each picture's size, in one quick run: per
    uploaded name, (width, height, [(x, y, w, h), ...]) with the incidental
    faces - a crowd behind the subjects - dropped, left to right."""
    ckpt = pick_first(list_models("checkpoints"), SAM3)
    if not ckpt or SAM3 not in ckpt.lower():
        raise ComfyError("Face swap finds faces with SAM3, and ComfyUI has no sam3 "
                         "checkpoint installed.")
    g = {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
         "2": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1", 1]}}}
    for i, name in enumerate(names):
        n = 10 + 10 * i
        g[str(n)] = {"class_type": "LoadImage", "inputs": {"image": name}}
        g[str(n + 1)] = {"class_type": "SAM3_Detect", "inputs": {
            "model": ["1", 0], "image": [str(n), 0], "conditioning": ["2", 0],
            "threshold": threshold, "refine_iterations": 0, "individual_masks": True}}
        g[str(n + 2)] = {"class_type": "PreviewAny", "inputs": {"source": [str(n + 1), 1]}}
        g[str(n + 3)] = {"class_type": "GetImageSize", "inputs": {"image": [str(n), 0]}}
        g[str(n + 4)] = {"class_type": "PreviewAny", "inputs": {"source": [str(n + 3), 0]}}
        g[str(n + 5)] = {"class_type": "PreviewAny", "inputs": {"source": [str(n + 3), 1]}}
    entry = wait_for(submit(g), 300)
    out = entry.get("outputs") or {}

    def text(node):
        t = (out.get(str(node)) or {}).get("text") or []
        if not t:
            raise ComfyError("Face detection returned nothing: %s"
                             % ("; ".join(status_messages(entry)) or "no output"))
        return json.loads(t[0])

    found = []
    for i in range(len(names)):
        n = 10 + 10 * i
        boxes = text(n + 2)
        boxes = boxes[0] if boxes and isinstance(boxes[0], list) else boxes
        boxes = [(b["x"], b["y"], b["width"], b["height"]) for b in boxes or []]
        if boxes:
            big = max(w * h for _, _, w, h in boxes)
            boxes = sorted((b for b in boxes if b[2] * b[3] >= big / 4), key=lambda b: b[0])
        found.append((int(text(n + 4)), int(text(n + 5)), boxes))
    return found


def head_square(box, width, height, pad):
    """A square around a face, `pad` times its size and nudged up for the
    hair, kept inside the picture."""
    x, y, w, h = box
    side = int(min(max(w, h) * pad, width, height))
    cx, cy = x + w / 2.0, y + h / 2.0 - 0.12 * h
    x0 = int(min(max(cx - side / 2.0, 0), width - side))
    y0 = int(min(max(cy - side / 2.0, 0), height - side))
    return {"x": x0, "y": y0, "width": side, "height": side}


def t_face_swap(a):
    """Put the people of one picture into another, face for face, as crop,
    edit and stitch: each face is cut out of the scene with room for the
    hair, edited at 1024 px against the matching face from `faces`, and
    blended back through a soft head mask. Edited whole, a face in a group
    photo is a few hundred of the edit model's million pixels and comes
    back as a likeness at best; cropped, it gets all of them, and nothing
    outside the heads is touched."""
    started = time.monotonic()
    scene, people = input_image(a["image"]), input_image(a["faces"])
    (sw, sh, targets), (_, _, sources) = find_faces([scene, people])
    if not targets:
        return result("No face found in %s." % a["image"], error=True)
    if not sources:
        return result("No face found in %s." % a["faces"], error=True)
    order = a.get("order")
    if order:
        try:
            sources = [sources[int(i) - 1] for i in order]
        except (ValueError, IndexError):
            return result("order has %s, but %s has %d face(s)."
                          % (order, a["faces"], len(sources)), error=True)
    pairs = list(zip(targets, sources))
    plan = plan_edit(a)
    seed = new_seed(a)
    instruction = SWAP_INSTRUCTION
    if a.get("instruction"):
        instruction += " " + a["instruction"].strip()

    # 1. The edits, to previews: the 19.5 GB edit model and SAM3 are never
    # on the card together (see finish_edit).
    free, total = gpu_memory()
    g = {}
    model, clip, vae = load_split(g, plan, lora=plan["lora"])
    g["13"] = {"class_type": "CFGNorm", "inputs": {"model": model, "strength": 1.0}}
    model = ["13", 0]
    g["20"] = {"class_type": "LoadImage", "inputs": {"image": scene}}
    g["21"] = {"class_type": "LoadImage", "inputs": {"image": people}}
    fast = bool(plan["lora"])
    steps = a.get("steps", plan["fast_steps"] if fast else plan["steps"])
    cfg = a.get("cfg", 1.0 if fast else plan["cfg"])
    crops = []
    for i, (t, s) in enumerate(pairs):
        n = 100 + 20 * i
        crop = head_square(t, sw, sh, a.get("padding", 2.4))
        crops.append(crop)
        g[str(n)] = {"class_type": "ImageCropV2", "inputs": {"image": ["20", 0], "crop_region": crop}}
        g[str(n + 1)] = {"class_type": "ImageScale", "inputs": {
            "image": [str(n), 0], "upscale_method": "lanczos", "width": FACE_EDIT,
            "height": FACE_EDIT, "crop": "disabled"}}
        # The likeness: the person's head with some room, not the whole picture.
        g[str(n + 2)] = {"class_type": "CropByBBoxes", "inputs": {
            "image": ["21", 0], "bboxes": {"x": s[0], "y": s[1], "width": s[2], "height": s[3]},
            "output_width": FACE_EDIT, "output_height": FACE_EDIT,
            "padding": int(max(s[2], s[3]) * 0.45), "keep_aspect": "pad"}}
        images = {"image1": [str(n + 1), 0], "image2": [str(n + 2), 0]}
        g[str(n + 3)] = {"class_type": "TextEncodeQwenImageEditPlus",
                         "inputs": dict(images, clip=clip, vae=vae, prompt=instruction)}
        # At cfg 1 the sampler never reads the negative, and each encode runs
        # the 7B vision encoder over both pictures on the CPU.
        g[str(n + 4)] = ({"class_type": "ConditioningZeroOut",
                          "inputs": {"conditioning": [str(n + 3), 0]}} if cfg == 1 else
                         {"class_type": "TextEncodeQwenImageEditPlus",
                          "inputs": dict(images, clip=clip, vae=vae, prompt="")})
        g[str(n + 5)] = {"class_type": "VAEEncode", "inputs": {"pixels": [str(n + 1), 0], "vae": vae}}
        g[str(n + 6)] = {"class_type": "KSampler", "inputs": {
            "seed": seed + i, "steps": steps, "cfg": cfg, "sampler_name": plan["sampler"],
            "scheduler": plan["scheduler"], "denoise": 1.0, "model": model,
            "positive": [str(n + 3), 0], "negative": [str(n + 4), 0],
            "latent_image": [str(n + 5), 0]}}
        g[str(n + 7)] = {"class_type": "VAEDecode", "inputs": {"samples": [str(n + 6), 0], "vae": vae}}
        g[str(n + 8)] = {"class_type": "PreviewImage", "inputs": {"images": [str(n + 7), 0]}}
    entry = wait_for(submit(g), a.get("timeout", DEFAULT_WAIT))
    release(free)
    edited = {}
    for i in range(len(pairs)):
        imgs = (entry.get("outputs") or {}).get(str(100 + 20 * i + 8), {}).get("images") or []
        if imgs:
            f = imgs[0]
            edited[i] = ("%s/%s" % (f["subfolder"], f["filename"]) if f.get("subfolder")
                         else f["filename"]) + " [temp]"
    if len(edited) < len(pairs):
        return result("The face edit failed: %s" % ("; ".join(status_messages(entry))
                                                    or "no picture came back"), error=True)

    # 2. The stitch: each edited head back in its place, through a mask of the
    # head before and after (so the old hair goes too), grown and blurred, and
    # faded out towards the crop's edge so no seam can show.
    g = {"1": {"class_type": "CheckpointLoaderSimple",
               "inputs": {"ckpt_name": pick_first(list_models("checkpoints"), SAM3)}},
         "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "head:1", "clip": ["1", 1]}},
         "20": {"class_type": "LoadImage", "inputs": {"image": scene}}}
    last = ["20", 0]
    for i, crop in enumerate(crops):
        n, side = 100 + 20 * i, crop["width"]
        g[str(n)] = {"class_type": "LoadImage", "inputs": {"image": edited[i]}}
        g[str(n + 1)] = {"class_type": "ImageCropV2", "inputs": {"image": ["20", 0], "crop_region": crop}}
        g[str(n + 2)] = {"class_type": "ImageScale", "inputs": {
            "image": [str(n + 1), 0], "upscale_method": "lanczos", "width": FACE_EDIT,
            "height": FACE_EDIT, "crop": "disabled"}}
        for k, src in ((3, [str(n), 0]), (4, [str(n + 2), 0])):
            g[str(n + k)] = {"class_type": "SAM3_Detect", "inputs": {
                "model": ["1", 0], "image": src, "conditioning": ["2", 0], "threshold": 0.3,
                "refine_iterations": 2, "individual_masks": False}}
        g[str(n + 5)] = {"class_type": "MaskComposite", "inputs": {
            "destination": [str(n + 3), 0], "source": [str(n + 4), 0], "x": 0, "y": 0,
            "operation": "or"}}
        g[str(n + 6)] = {"class_type": "GrowMask", "inputs": {
            "mask": [str(n + 5), 0], "expand": 28, "tapered_corners": True}}
        g[str(n + 7)] = {"class_type": "SolidMask", "inputs": {
            "value": 1.0, "width": FACE_EDIT, "height": FACE_EDIT}}
        edge = FACE_EDIT // 8
        g[str(n + 8)] = {"class_type": "FeatherMask", "inputs": {
            "mask": [str(n + 7), 0], "left": edge, "top": edge, "right": edge, "bottom": edge}}
        g[str(n + 9)] = {"class_type": "MaskComposite", "inputs": {
            "destination": [str(n + 6), 0], "source": [str(n + 8), 0], "x": 0, "y": 0,
            "operation": "multiply"}}
        g[str(n + 10)] = {"class_type": "MaskToImage", "inputs": {"mask": [str(n + 9), 0]}}
        g[str(n + 11)] = {"class_type": "ImageBlur", "inputs": {
            "image": [str(n + 10), 0], "blur_radius": 31, "sigma": 10.0}}
        g[str(n + 12)] = {"class_type": "ImageToMask", "inputs": {"image": [str(n + 11), 0], "channel": "red"}}
        # The edit drifts brighter and warmer than an old print; matched to
        # the crop it replaces, the new head takes the photograph's colour.
        g[str(n + 15)] = {"class_type": "ColorTransfer", "inputs": {
            "image_target": [str(n), 0], "image_ref": [str(n + 2), 0],
            "method": "reinhard_lab", "source_stats": "per_frame", "strength": 1.0}}
        g[str(n + 13)] = {"class_type": "ImageScale", "inputs": {
            "image": [str(n + 15), 0], "upscale_method": "lanczos", "width": side,
            "height": side, "crop": "disabled"}}
        g[str(n + 14)] = {"class_type": "ImageCompositeMasked", "inputs": {
            "destination": last, "source": [str(n + 13), 0], "x": crop["x"], "y": crop["y"],
            "resize_source": False, "mask": [str(n + 12), 0]}}
        last = [str(n + 14), 0]
    g["9"] = {"class_type": "SaveImage", "inputs": {
        "filename_prefix": a.get("filename_prefix", "StudioFaceSwap"), "images": last}}
    header = ("model: %s%s\nseed: %d  steps: %s  cfg: %s\nfaces swapped: %d (left to right; "
              "%d found in the scene, %d in the faces picture)\n" % (
                  plan["label"], " + " + plan["lora"] if plan["lora"] else "", seed, steps,
                  cfg, len(pairs), len(targets), len(sources)))
    notes = [vram_note(free, total)] if vram_note(free, total) else []
    return run(g, dict(a, wait=True), header, notes, started=started)


def t_upscale(a):
    """Enlarge a picture and redraw its fine detail with the photographic model."""
    unets = list_models("diffusion_models") or list_models("unet")
    photo = [u for u in unets if family_of(u).get("photo")]
    if not photo:
        return result("Upscaling redraws detail with Z-Image, which is not installed on "
                      "ComfyUI.", error=True)
    plan = plan_model({"diffusion_model": photo[0]})
    seed = new_seed(a)
    g = {}
    model, clip, vae = load_split(g, plan)
    g["8"] = {"class_type": "LoadImage", "inputs": {"image": input_image(a["image"])}}
    # Normalise to ~1 MP first so `scale` means the same thing for any input,
    # and the pass never samples more than the card holds.
    g["20"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
        "image": ["8", 0], "upscale_method": "lanczos", "megapixels": 1.0,
        "resolution_steps": 16}}
    text = (a.get("prompt") or "A sharp, detailed, high-resolution photograph").strip()
    if a.get("realism", True) and wants_photo(text):
        text = text.rstrip(" .,;") + ". " + PHOTO_SUFFIX
    g["2"] = {"class_type": "CLIPTextEncode", "inputs": {"text": text, "clip": clip}}
    g["3"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["2", 0]}}
    r = dict(plan["hires"], scale=a.get("scale", 2.0), denoise=a.get("creativity", 0.3))
    out = detail_pass(g, ["20", 0], model, ["2", 0], ["3", 0], vae, r, seed)
    g["7"] = {"class_type": "SaveImage", "inputs": {
        "filename_prefix": a.get("filename_prefix", "StudioUpscale"), "images": out}}
    header = "model: %s\nseed: %d  scale: x%s of ~1 MP  creativity (denoise): %s\n" % (
        plan["label"], seed, r["scale"], r["denoise"])
    return run(g, a, header)


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
    return run(graph, a, "")


def t_wait(a):
    pid = a["prompt_id"]
    entry = history_entry(pid)
    if entry is None:
        q = get_json("/queue")
        queued = {row[1] for row in q.get("queue_running", []) + q.get("queue_pending", []) if len(row) > 1}
        if pid not in queued:
            return result("ComfyUI has no prompt %s - not queued and not in history." % pid, error=True)
    out = collect(pid, wait_for(pid, a.get("timeout", DEFAULT_WAIT)))
    release()
    return out


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


def upload(path):
    """Send a local file to ComfyUI's input folder; returns the name to load it by."""
    with open(path, "rb") as fh:
        data = fh.read()
    res = post_multipart("/upload/image", {"overwrite": "true"}, os.path.basename(path), data)
    name = res.get("name", os.path.basename(path))
    if res.get("subfolder"):
        name = res["subfolder"] + "/" + name
    return name


def t_upload_image(a):
    path = studio_mcp.local_path(a["path"])
    if not os.path.isfile(path):
        return result("No file at %s on this workstation." % path, error=True)
    return result("Uploaded as %r - pass that as init_image to comfy_generate, or as a "
                  "LoadImage input. (comfy_generate, comfy_edit_image and comfy_upscale "
                  "also take the local path directly.)" % upload(path))


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
     "Make pictures from a text prompt. With no model named it uses the most "
     "photographic one installed (Z-Image Turbo) with its own recipe, adds a realism "
     "sentence to photographic prompts, and runs a hi-res detail pass that redraws "
     "skin, fabric and texture at 1.5x the size - leave those defaults alone for the "
     "most realistic result. init_image (a local path or an uploaded name) restyles a "
     "picture; to change something specific in a picture use comfy_edit_image. Blocks "
     "until done, saves the results on this workstation and returns their paths, the "
     "seed and the settings used.",
     _obj({
         "prompt": _s("The picture, described as a photograph in plain sentences: subject "
                      "and action, setting, light, camera and lens, the textures that "
                      "make it real. Not instructions, not tag lists."),
         "negative": _s("What to avoid. Only read at cfg above 1; Z-Image ignores it."),
         "realism": {"type": "boolean", "description": "Default true: photographic prompts "
                     "get the realism sentence (and, at cfg above 1, a negative against "
                     "CGI looks). Set false for illustration, logos, paintings, cartoons."},
         "hires": {"type": "boolean", "description": "Default true where the model has a "
                   "detail pass. false for a quick draft at the base size."},
         "hires_scale": _n("Detail pass enlargement. Default 1.5; capped near 4 MP.",
                           minimum=1, maximum=2.5),
         "hires_denoise": _n("Detail pass strength. Default 0.33; higher invents more "
                             "detail, lower keeps the draft.", minimum=0.1, maximum=0.6),
         "checkpoint": _s("All-in-one checkpoint filename from comfy_list_models. Default: "
                          "a split model with a known recipe, else the first checkpoint."),
         "diffusion_model": _s("Split model: filename from comfy_list_models "
                               "kind=diffusion_models. Picks the family's recipe. Z-Image "
                               "is the photographic one, Krea 2 the stylised one."),
         "text_encoder": _s("Split model: text encoder filename from kind=text_encoders. "
                            "Default: chosen for the family."),
         "text_encoder_type": _s("Split model: the encoder family. Default: chosen for the "
                                 "family (lumina2 for Z-Image).", enum=ENCODER_TYPES),
         "vae": _s("Split model: VAE filename from kind=vae. Default: the family's."),
         "shift": _n("Split model: sampling shift. Default: the family's (3.0 for Z-Image).",
                     minimum=0, maximum=100),
         "width": _i("Pixels, multiple of 16, before the detail pass. Default 1024.",
                     minimum=64, maximum=4096),
         "height": _i("Pixels, multiple of 16, before the detail pass. Default 1024.",
                      minimum=64, maximum=4096),
         "steps": _i("Sampling steps. Default: the model's recipe (8 for Z-Image Turbo, "
                     "20 for a checkpoint).", minimum=1, maximum=150),
         "cfg": _n("Prompt adherence. Default: the model's recipe (1 for Z-Image Turbo, "
                   "6 for a checkpoint).", minimum=0, maximum=30),
         "seed": _i("Fixed seed to reproduce or vary a result. Default: random.", minimum=-1),
         "sampler": _s("Sampler name. Default: the model's recipe.", enum=SAMPLERS),
         "scheduler": _s("Scheduler name. Default: the model's recipe.", enum=SCHEDULERS),
         "batch_size": _i("Images per call. Default 1.", minimum=1, maximum=8),
         "init_image": _s("Image-to-image: a picture's path on this workstation, or a name "
                          "from comfy_upload_image. Omit for text-to-image."),
         "denoise": _n("Image-to-image only: how much to change the init image, 0..1. "
                       "Default 0.6.", minimum=0, maximum=1),
         "lora": _s("LoRA filename from comfy_list_models kind=loras, for this model's family."),
         "lora_strength": _n("LoRA strength. Default 1.0.", minimum=-2, maximum=2),
         "filename_prefix": _s("Output filename prefix. Default StudioAssistant."),
         "wait": {"type": "boolean", "description": "Default true. false returns the "
                                                    "prompt_id at once; collect it with comfy_wait."},
         "timeout": _i("Seconds to wait before giving the prompt_id back. Default %d."
                       % DEFAULT_WAIT, minimum=5, maximum=MAX_WAIT),
     }, ["prompt"])),
    ("comfy_edit_image", t_edit_image,
     "Change a picture by instruction and keep the rest of it: 'replace the sky with "
     "a storm', 'put her in a red coat', 'remove the car', 'make it night', 'relight "
     "from the left'. Up to two more pictures can be references ('put the jacket from "
     "picture 2 on the man in picture 1'). Uses Qwen-Image-Edit, fast (4 steps) by "
     "default. photo_finish true adds a Z-Image detail pass for the most photographic "
     "result, at extra time. Returns the edited picture like comfy_generate.",
     _obj({
         "image": _s("The picture to edit: its path on this workstation (sent up for "
                     "you) or a name from comfy_upload_image."),
         "instruction": _s("What to change, as one plain instruction; say what must stay "
                           "the same if it matters ('keep the face and pose')."),
         "references": {"type": "array", "maxItems": 2, "items": {"type": "string"},
                        "description": "Up to two more pictures, paths or uploaded "
                                       "names, called picture 2 and 3 in the instruction."},
         "photo_finish": {"type": "boolean", "description": "Default false. true runs a "
                          "Z-Image detail pass at 1.5x afterwards: sharper, more real "
                          "skin and texture, bigger file, extra time."},
         "finish_denoise": _n("photo_finish strength. Default 0.25.", minimum=0.1, maximum=0.5),
         "fast": {"type": "boolean", "description": "Default true: the 4-step Lightning "
                  "LoRA. false: 20 full steps at cfg 4, several times slower, sometimes "
                  "more faithful on hard edits."},
         "steps": _i("Override the step count.", minimum=1, maximum=60),
         "cfg": _n("Override cfg.", minimum=0, maximum=10),
         "seed": _i("Fixed seed. Default: random.", minimum=-1),
         "edit_model": _s("Edit model filename from kind=diffusion_models. Default: the "
                          "installed one."),
         "filename_prefix": _s("Output filename prefix. Default StudioEdit."),
         "wait": {"type": "boolean", "description": "Default true."},
         "timeout": _i("Seconds to wait. Default %d." % DEFAULT_WAIT, minimum=5,
                       maximum=MAX_WAIT),
     }, ["image", "instruction"])),
    ("comfy_face_swap", t_face_swap,
     "Put the faces of the people in one picture onto the people in another, so it "
     "looks like they were there: 'make this old photo us'. Each face in `image` is cut "
     "out large, given the matching face from `faces` with Qwen-Image-Edit, and blended "
     "back; everything but the heads is left as it was. Faces pair left to right. Takes "
     "a few minutes. Returns the finished picture like comfy_generate.",
     _obj({
         "image": _s("The scene whose people get new faces: a path on this workstation "
                     "or a name from comfy_upload_image."),
         "faces": _s("The picture of the people whose faces go in, same forms as image."),
         "order": {"type": "array", "items": {"type": "integer", "minimum": 1},
                   "description": "Which face in `faces` (1 = leftmost) goes on each "
                                  "face in `image`, left to right. Default [1, 2, ...]."},
         "instruction": _s("Anything to add to the swap instruction, such as 'keep her "
                           "own hair' or 'no glasses'."),
         "padding": _n("How much of the head around each face is redrawn, as a multiple "
                       "of the face's size. Default 2.4.", minimum=1.5, maximum=4),
         "fast": {"type": "boolean", "description": "Default true: 4 Lightning steps. "
                  "false: 20 full steps, much slower, sometimes a closer likeness."},
         "steps": _i("Override the step count.", minimum=1, maximum=60),
         "cfg": _n("Override cfg.", minimum=0, maximum=10),
         "seed": _i("Fixed seed. Default: random.", minimum=-1),
         "filename_prefix": _s("Output filename prefix. Default StudioFaceSwap."),
         "timeout": _i("Seconds to wait. Default %d." % DEFAULT_WAIT, minimum=5,
                       maximum=MAX_WAIT),
     }, ["image", "faces"])),
    ("comfy_upscale", t_upscale,
     "Enlarge a picture and redraw its fine detail with the photographic model, so it "
     "is sharp at the new size rather than stretched. The picture is brought to about "
     "1 megapixel, then scaled by `scale`.",
     _obj({
         "image": _s("The picture: its path on this workstation, or an uploaded name."),
         "scale": _n("Enlargement of the ~1 MP picture. Default 2 (about 4 MP).",
                     minimum=1.1, maximum=2.5),
         "creativity": _n("How much detail may be invented, 0.1..0.6. Default 0.3; above "
                          "0.4 faces and text start to change.", minimum=0.1, maximum=0.6),
         "prompt": _s("What the picture shows, one sentence. Helps the redraw. Optional."),
         "realism": {"type": "boolean", "description": "Default true. false for art."},
         "seed": _i("Fixed seed. Default: random.", minimum=-1),
         "filename_prefix": _s("Output filename prefix. Default StudioUpscale."),
         "wait": {"type": "boolean", "description": "Default true."},
         "timeout": _i("Seconds to wait. Default %d." % DEFAULT_WAIT, minimum=5,
                       maximum=MAX_WAIT),
     }, ["image"])),
    ("comfy_run_workflow", t_run_workflow,
     "Run any ComfyUI workflow in API format: an object of node_id -> {class_type, "
     "inputs}, where a link is [node_id, output_index]. This is what 'Save (API "
     "Format)' in ComfyUI exports. Use comfy_search_nodes and comfy_node_info to get "
     "input names right. Blocks until done and downloads the outputs like "
     "comfy_generate.",
     _obj({"workflow": {"type": "object", "description": "The API-format graph.",
                        "additionalProperties": True},
           "wait": {"type": "boolean", "description": "Default true."},
           "timeout": _i("Seconds to wait. Default %d." % DEFAULT_WAIT, minimum=5, maximum=MAX_WAIT)},
          ["workflow"])),
    ("comfy_wait", t_wait,
     "Wait for a queued prompt_id to finish and download its outputs to this workstation.",
     _obj({"prompt_id": _s("The prompt_id a generate call returned."),
           "timeout": _i("Seconds to wait. Default %d." % DEFAULT_WAIT, minimum=5, maximum=MAX_WAIT)},
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
    "comfy_edit_image": {"destructive": False},
    "comfy_face_swap": {"destructive": False},
    "comfy_upscale": {"destructive": False},
    "comfy_run_workflow": {"destructive": False},
    "comfy_upload_image": {"destructive": False, "idempotent": True},
    "comfy_interrupt": {"destructive": True, "idempotent": True},
    "comfy_clear_queue": {"destructive": True, "idempotent": True},
}

SERVER = studio_mcp.Server(
    "studio-comfy-mcp", "1.1",
    studio_mcp.tools_from_table(TOOLS, read_only=READ_ONLY, **HINTS),
    errors=(ComfyError, KeyError, TypeError, ValueError),
    instructions="ComfyUI on %s. comfy_generate makes a picture, comfy_edit_image "
                 "changes one by instruction, comfy_upscale enlarges one; each builds the "
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
