#!/usr/bin/env python3
"""
studio_imagegen - the Image Studio's engine: Person -> Style -> Scene ->
Reference -> Generate, with the ComfyUI graph built underneath.

No tkinter here; `studio_images_ui.py` is the tab, and this module can be
driven and tested without a window. The pieces, each apart from the next:

- **Backends.** Any number of ComfyUI servers, each an independent worker - the
  5090 on this PC and the 3090 on the LLM PC by default. Their VRAM is never
  pooled and their disks are never assumed to be shared: a picture reaches a
  backend by upload, a model by a filename that backend has.
- **`ComfyUIClient`**, one per backend. Every HTTP and WebSocket call to a
  ComfyUI lives in it; nothing else here opens a URL.
- **Logical assets.** A model or LoRA has one id in the app and a filename per
  backend (`resolve_model`, `lora_file`), because the two machines keep
  different files under different names.
- **Workflow templates** (`comfy_workflows/*.json`): ComfyUI API-format graphs
  with `{{placeholders}}`, optional nodes and a LoRA chain point. `fill()` is the
  adapter; replacing a workflow is editing a JSON file, not this module.
- **`compose()`** turns the form's settings into a plan for one backend: the
  prompt with identity triggers and style additions, the LoRA stack with
  compatibility warnings, the template's values. Pure, so it is tested.
- **`JobQueue`**: one lane (a thread) per backend, so both GPUs work at once;
  routing picks a lane from the preset's role.
- **`History`**: every finished picture with everything needed to make it again.

Stdlib only, like the rest of the app.
"""

import copy
import hashlib
import json
import os
import queue
import random
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

import studio_critic as critic
import studio_doctor as doctor
from studio_comfy_mcp import (FACE_EDIT, FACE_MIN, FACE_PAD, FACE_PROMPT, SAM3, ComfyError,
                              Unreachable, _explain, head_square, outputs_of, oval_png,
                              preview_of, status_messages)

HERE = os.path.dirname(os.path.abspath(__file__))
WORKFLOWS_DIR = os.path.join(HERE, "comfy_workflows")
STYLE_EXAMPLES_DIR = os.path.join(HERE, "style_examples")

MAX_SEED = 2 ** 32 - 1
JOB_TIMEOUT = 1800            # seconds a job may run before it is given up on
POSE_NODE = "StudioDWPoseKeypoints"   # comfy_nodes/studio_dwpose: a photo's pose points
PASTE_NODE = "StudioFacePaste"        # comfy_nodes/studio_facepaste: a person's own face
HEALTH_TTL = 30               # seconds a health reading is trusted when routing
QUIET_AFTER = 120             # seconds without a progress event before a job says so
MODEL_KINDS = ("diffusion_models", "checkpoints", "text_encoders", "vae", "loras",
               "clip_vision", "style_models", "controlnet", "upscale_models")


def studio_dir():
    """Beside the settings file, like everything else the app keeps, so
    STUDIO_SETTINGS moves it - which is how the tests stay out of it."""
    return os.path.join(doctor.data_dir(), "image-studio")


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-") or "item"


def unique_id(base, taken):
    base, n = slug(base), 2
    out = base
    while out in taken:
        out, n = "%s-%d" % (base, n), n + 1
    return out


# ============================================================ what things mean

ROLES = [
    ("primary", "Primary image generation"),
    ("flux", "FLUX / large models"),
    ("hires", "High-resolution generation"),
    ("identity", "Identity-heavy workflows"),
    ("interactive", "Interactive generation"),
    ("training", "LoRA training (later)"),
    ("secondary", "Secondary generation"),
    ("batch", "Batch jobs"),
    ("preprocess", "Preprocessing"),
    ("controlnet", "ControlNet preprocessing"),
    ("depth_pose", "Depth / pose maps"),
    ("caption", "Captioning"),
    ("upscale", "Upscaling"),
    ("background", "Background jobs"),
]
ROLE_NAMES = [r for r, _ in ROLES]

CATEGORIES = ["Identity", "Style", "Camera / Film", "Clothing", "Character",
              "Detail / Enhancement", "Other"]

# Model families. A LoRA is trained against one and applies to the families
# listed with it here and no others; anything unlisted is "unknown" and is
# applied with a warning rather than refused, since the table cannot know
# every model.
FAMILIES = {
    "flux1": "FLUX.1",
    "flux1-kontext": "FLUX.1 Kontext",
    "flux2": "FLUX.2",
    "sdxl": "SDXL",
    "sd15": "SD 1.5",
    "z-image": "Z-Image",
    "krea2": "Krea 2",
    "qwen-image": "Qwen-Image",
}
COMPATIBLE = {
    "flux1": {"flux1", "flux1-kontext"},
    "flux1-kontext": {"flux1-kontext", "flux1"},
}

REFERENCE_KINDS = [
    ("face", "Face", "Who the person is: a face to condition identity on"),
    ("pose", "Pose", "How the body is posed"),
    ("composition", "Composition", "How the frame is laid out"),
    ("style", "Style", "How the picture should look"),
    ("source", "Source image", "The picture to edit or vary"),
]
REFERENCE_NAMES = [k for k, _, _ in REFERENCE_KINDS]

PRESETS = {
    "standard": {"label": "Standard", "role": "interactive",
                 "about": "Text to image with the chosen model.",
                 "values": {"refine": False}},
    "identity": {"label": "Identity Portrait", "role": "identity",
                 "about": "A person from an identity profile, portrait framing, refined.",
                 "values": {"refine": True, "face_detail": True, "width": 896,
                            "height": 1152}},
    "hq_final": {"label": "High Quality Final", "role": "hires",
                 "about": "Generate, upscale, redraw detail at low denoise, then redraw "
                          "each face at full size.",
                 "values": {"refine": True, "upscale": 2.0, "face_detail": True}},
}
PRESET_ORDER = ["standard", "identity", "hq_final"]


def guess_family(filename):
    n = filename.lower()
    if "kontext" in n:
        return "flux1-kontext"
    if re.search(r"flux[._-]?2", n):
        return "flux2"
    if "flux" in n:
        return "flux1"
    if "z_image" in n or "zimage" in n or "z-image" in n:
        return "z-image"
    if "krea2" in n:
        return "krea2"
    if "qwen" in n:
        return "qwen-image"
    if "sdxl" in n or re.search(r"(^|[^a-z])xl([^a-z]|$)", n):
        return "sdxl"
    return ""


def guess_category(filename):
    n = filename.lower()
    for words, cat in ((("identity", "person", "face", "likeness"), "Identity"),
                       (("film", "polaroid", "sx70", "sx-70", "kodak", "portra", "cinema",
                         "camera", "lens"), "Camera / Film"),
                       (("style", "brush", "paint", "art"), "Style"),
                       (("cloth", "outfit", "dress", "jacket"), "Clothing"),
                       (("detail", "enhanc", "lightning", "turbo", "fix"),
                        "Detail / Enhancement"),
                       (("character", "char_"), "Character")):
        if any(w in n for w in words):
            return cat
    return "Other"


def pretty(filename):
    stem = os.path.splitext(os.path.basename(filename))[0]
    return re.sub(r"[_\-.]+", " ", stem).strip().title() or filename


def compatibility(lora_family, model_family):
    """True, False, or None when either family is unknown."""
    if not lora_family or not model_family:
        return None
    return model_family in COMPATIBLE.get(lora_family, {lora_family})


# ================================================================= records
# Everything on disk is re-validated on load: a hand-edited or wrecked file
# costs the record, never the tab (the settings file's rule).

def _num(v, kind, default, lo=None, hi=None):
    try:
        v = kind(v)
    except (TypeError, ValueError):
        return default
    if lo is not None:
        v = max(lo, v)
    if hi is not None:
        v = min(hi, v)
    return v


def _str(v, default=""):
    return v.strip() if isinstance(v, str) else default


def _strs(v):
    return [x.strip() for x in v if isinstance(x, str) and x.strip()] if isinstance(v, list) else []


def _map(v):
    """{backend id: filename} with junk dropped."""
    if not isinstance(v, dict):
        return {}
    return {k: x.strip() for k, x in v.items() if isinstance(k, str) and isinstance(x, str)
            and x.strip()}


def clean_backend(d):
    if not isinstance(d, dict) or not _str(d.get("url")):
        return None
    url = _str(d.get("url")).rstrip("/")
    if not re.match(r"^https?://", url):
        url = "http://" + url
    return {
        "id": slug(d.get("id") or d.get("name") or url),
        "name": _str(d.get("name")) or url,
        "url": url,
        "ws_url": _str(d.get("ws_url")),
        "enabled": d.get("enabled", True) is not False,
        "roles": [r for r in _strs(d.get("roles")) if r in ROLE_NAMES],
        "notes": _str(d.get("notes")),
        # The LLM PC's ComfyUI shares its card with LM Studio, whose models
        # ComfyUI cannot see; a job there clears them first (AGENTS.md).
        "shares_llm_gpu": d.get("shares_llm_gpu") is True,
        # ComfyUI keeps its last run's models on the card; this PC's 5090 is
        # also After Effects' and Resolve's, and the 3090 is LM Studio's.
        "release_vram": d.get("release_vram", True) is not False,
        "encoder_on_cpu": d.get("encoder_on_cpu") is True,
        "max_megapixels": _num(d.get("max_megapixels", 4.2), float, 4.2, 0.5, 64.0),
        "lora_dir": _str(d.get("lora_dir")),
        # How to start ComfyUI there, said when it does not answer.
        "start": _str(d.get("start")),
    }


def clean_model(d):
    if not isinstance(d, dict) or not _str(d.get("id")):
        return None
    backends = {}
    for bid, over in (d.get("backends") or {}).items() if isinstance(
            d.get("backends"), dict) else ():
        if over is None:
            backends[bid] = None          # explicitly not on that machine
        elif isinstance(over, dict):
            backends[bid] = {k: v for k, v in over.items() if isinstance(k, str)
                             and isinstance(v, (str, int, float))}
    values = d.get("values") if isinstance(d.get("values"), dict) else {}
    defaults = d.get("defaults") if isinstance(d.get("defaults"), dict) else {}
    return {
        "id": slug(d["id"]),
        "label": _str(d.get("label")) or d["id"],
        "family": _str(d.get("family")),
        "workflow": _str(d.get("workflow")),
        "values": {k: v for k, v in values.items() if isinstance(v, (str, int, float))},
        "backends": backends,
        "defaults": {k: v for k, v in defaults.items() if isinstance(v, (str, int, float, bool))},
        "notes": _str(d.get("notes")),
    }


def clean_lora(d):
    if not isinstance(d, dict) or not _str(d.get("file")):
        return None
    cat = _str(d.get("category"))
    return {
        "id": slug(d.get("id") or d["file"]),
        "file": _str(d["file"]),
        "files": _map(d.get("files")),
        "name": _str(d.get("name")) or pretty(d["file"]),
        "category": cat if cat in CATEGORIES else "Other",
        "trigger": _str(d.get("trigger")),
        "strength": _num(d.get("strength", 0.8), float, 0.8, -2.0, 2.0),
        "always": bool(d.get("always")),
        "family": _str(d.get("family")),
        "preview": _str(d.get("preview")),
        "notes": _str(d.get("notes")),
        "source": _str(d.get("source")),          # where it came from (a CivitAI page)
        "sha256": _str(d.get("sha256")).lower(),  # the file's, which is how CivitAI names it
    }


def clean_identity(d):
    if not isinstance(d, dict) or not _str(d.get("name")):
        return None
    return {
        "id": slug(d.get("id") or d["name"]),
        "name": _str(d["name"]),
        "lora": _str(d.get("lora")),
        "trigger": _str(d.get("trigger")),
        "strength": _num(d.get("strength", 0.85), float, 0.85, -2.0, 2.0),
        "references": _strs(d.get("references")),
        "reference_strength": _num(d.get("reference_strength", 0.6), float, 0.6, 0.0, 2.0),
        "use_references": d.get("use_references", True) is not False,
        "notes": _str(d.get("notes")),
    }


def clean_style(d):
    if not isinstance(d, dict) or not _str(d.get("name")):
        return None
    opt_num = lambda k, kind, lo, hi: (None if d.get(k) in (None, "")
                                       else _num(d.get(k), kind, None, lo, hi))
    return {
        "id": slug(d.get("id") or d["name"]),
        "name": _str(d["name"]),
        "lora": _str(d.get("lora")),
        "trigger": _str(d.get("trigger")),
        "strength": _num(d.get("strength", 0.6), float, 0.6, -2.0, 2.0),
        "prompt": _str(d.get("prompt")),
        "negative": _str(d.get("negative")),
        "sampler": _str(d.get("sampler")),
        "scheduler": _str(d.get("scheduler")),
        "guidance": opt_num("guidance", float, 0.0, 30.0),
        "steps": opt_num("steps", int, 1, 200),
        "width": opt_num("width", int, 256, 4096),
        "height": opt_num("height", int, 256, 4096),
        "families": [f for f in _strs(d.get("families"))],
        "example": _str(d.get("example")),
        "notes": _str(d.get("notes")),
    }


def clean_item_refs(v):
    """{item as the form names it: picture path}, junk dropped."""
    return {k.strip(): x.strip() for k, x in v.items() if isinstance(k, str) and k.strip()
            and isinstance(x, str) and x.strip()} if isinstance(v, dict) else {}


def clean_character(d):
    """A character from the creator: the look it keeps from picture to
    picture (`looks`, every CHARACTER_KEYS slot and slider it sets), the
    identity whose LoRA carries its face, and a picture per item it wears."""
    if not isinstance(d, dict) or not _str(d.get("name")):
        return None
    looks = d.get("looks") if isinstance(d.get("looks"), dict) else {}
    kept = {}
    for k in CHARACTER_KEYS:
        if k in SLIDER_KEYS:
            step = _num(looks.get(k, 0), int, 0, -SLIDER_SPAN, SLIDER_SPAN)
            if step:
                kept[k] = step
        elif _str(looks.get(k)):
            kept[k] = _str(looks.get(k))
    return {
        "id": slug(d.get("id") or d["name"]),
        "name": _str(d["name"]),
        "identity": _str(d.get("identity")),
        "looks": kept,
        "item_refs": clean_item_refs(d.get("item_refs")),
        "notes": _str(d.get("notes")),
    }


def clean_outfit(d):
    """A clothes preset: a name and what it puts in each OUTFIT_KEYS slot.
    Putting it on replaces all of those slots, so a slot it leaves empty
    is taken off (a summer outfit has no coat)."""
    if not isinstance(d, dict) or not _str(d.get("name")):
        return None
    looks = d.get("looks") if isinstance(d.get("looks"), dict) else {}
    return {
        "id": slug(d.get("id") or d["name"]),
        "name": _str(d["name"]),
        "looks": {k: _str(looks[k]) for k in OUTFIT_KEYS if _str(looks.get(k))},
    }


def style_example(style):
    """The picture that shows what `style` looks like: its own `example` when
    that file is there, else the one shipped for its id (the same cat photo
    through each default style), else None."""
    for path in (style.get("example"),
                 os.path.join(STYLE_EXAMPLES_DIR, (style.get("id") or "") + ".png")):
        if path and os.path.isfile(path):
            return path
    return None


CLEAN = {"backends": clean_backend, "models": clean_model, "loras": clean_lora,
         "identities": clean_identity, "styles": clean_style,
         "characters": clean_character, "outfits": clean_outfit}


def _default_backends():
    return [
        {"id": "5090", "name": "5090 Workstation",
         "url": os.environ.get("IMAGE_STUDIO_5090_URL", "http://127.0.0.1:8188"),
         "roles": ["primary", "flux", "hires", "identity", "interactive", "training"],
         "notes": "This PC. Its GPU is also After Effects' and Resolve's, so ComfyUI "
                  "lets go of VRAM when its queue empties.",
         "start": os.environ.get("IMAGE_STUDIO_5090_START",
                                 r"D:\ComfyUI\Start ComfyUI (Image Studio).cmd"),
         "max_megapixels": 6.0},
        {"id": "3090", "name": "3090 Server",
         "url": os.environ.get("COMFYUI_URL", "http://100.127.17.38:8188"),
         "roles": ["secondary", "batch", "preprocess", "controlnet", "depth_pose",
                   "caption", "upscale", "background"],
         "notes": "The LLM PC. Shares its 24 GB card with LM Studio: a job here unloads "
                  "LM Studio's models first, and the text encoder runs on the CPU.",
         "start": "Start ComfyUI on the LLM PC with --listen.",
         "shares_llm_gpu": True, "encoder_on_cpu": True, "max_megapixels": 4.2},
    ]


def _default_outfits():
    # Starters, worded so the Scene Builder's mannequin can draw them: its
    # colour words, shoe kinds and hats (studio_scene CLOTH, SHOES, HATS).
    return [
        {"id": "casual", "name": "Casual",
         "looks": {"top": "white t-shirt", "bottom": "blue jeans",
                   "footwear": "white sneakers"}},
        {"id": "smart", "name": "Smart",
         "looks": {"top": "white button-down shirt", "bottom": "charcoal tailored trousers",
                   "outerwear": "navy blazer", "footwear": "brown leather loafers",
                   "accessories": "wristwatch"}},
        {"id": "winter", "name": "Winter",
         "looks": {"top": "grey knit sweater", "bottom": "black jeans",
                   "outerwear": "camel wool overcoat", "footwear": "brown leather boots",
                   "accessories": "beanie, scarf"}},
        {"id": "summer", "name": "Summer",
         "looks": {"top": "white summer dress", "footwear": "tan sandals",
                   "accessories": "sunglasses"}},
        {"id": "site-ppe", "name": "Site PPE",
         "looks": {"top": "grey t-shirt", "bottom": "khaki cargo pants",
                   "outerwear": "hi-vis vest", "footwear": "brown work boots",
                   "accessories": "hard hat, safety glasses, gloves"}},
        {"id": "oktoberfest-dirndl", "name": "Oktoberfest dirndl",
         "looks": {"top": "green dirndl with a white blouse and a white apron",
                   "footwear": "black shoes", "accessories": "flower crown, beer steins"}},
        {"id": "oktoberfest-lederhosen", "name": "Oktoberfest lederhosen",
         "looks": {"top": "white linen shirt", "bottom": "lederhosen",
                   "footwear": "brown leather boots",
                   "accessories": "german hat, accordion"}},
    ]


def _default_models():
    # Filenames are the ones ComfyUI's own FLUX template downloads
    # (flux_dev_full_text_to_image); a machine with other names says so in its
    # `backends` entry (the Models editor). FLUX runs the baseline workflow
    # with LoRAs, refine and the face pass layered on, each off unless asked
    # for; references (Redux, image to image) are still flux_hq.json's.
    return [
        {"id": "flux-dev", "label": "FLUX.1 [dev]", "family": "flux1",
         "workflow": "flux_dev_baseline",
         "values": {"model": "flux1-dev.safetensors", "weight_dtype": "default",
                    "clip_l": "clip_l.safetensors", "t5": "t5xxl_fp16.safetensors",
                    "vae": "ae.safetensors"},
         "backends": {"3090": {"t5": "t5xxl_fp8_e4m3fn_scaled.safetensors",
                               "weight_dtype": "fp8_e4m3fn"}},
         "defaults": {"steps": 20, "guidance": 3.5, "sampler": "euler",
                      "scheduler": "simple", "width": 1024, "height": 1024},
         "notes": "FLUX.1 [dev]: text to image with LoRAs, refine and the face pass."},
        {"id": "z-image-turbo", "label": "Z-Image Turbo", "family": "z-image",
         "workflow": "zimage_hq",
         "values": {"model": "z_image_turbo_bf16.safetensors",
                    "encoder": "qwen_3_4b.safetensors", "vae": "ae.safetensors"},
         "defaults": {"steps": 8, "guidance": 1.0, "sampler": "res_multistep",
                      "scheduler": "simple", "width": 1024, "height": 1024},
         "notes": "Fast photographic model and the default: the 5090 when it has the "
                  "files, else the 3090."},
    ]


def _default_styles():
    # Prompt-only styles, so identity and style are separate controls from the
    # first day; a style LoRA is added to one in the Styles editor.
    return [
        {"id": "none", "name": "No style", "prompt": ""},
        {"id": "modern-photo", "name": "Modern photograph",
         "prompt": "Shot on a full-frame digital camera, 50mm lens, natural light, "
                   "true-to-life colour, fine detail."},
        {"id": "sx-70", "name": "SX-70 Polaroid",
         "prompt": "Polaroid SX-70 instant photograph: soft focus, warm faded colour, "
                   "lifted blacks, gentle vignetting, square format.",
         "width": 1024, "height": 1024},
        {"id": "cinema", "name": "Cinema still",
         "prompt": "Anamorphic 35mm film still, 2.39:1 framing, motivated practical "
                   "lighting, subtle halation, fine film grain.",
         "width": 1344, "height": 576},
        {"id": "black-and-white", "name": "Black and white",
         "prompt": "Black and white photograph on Kodak Tri-X, deep blacks, visible grain, "
                   "strong directional light."},
        {"id": "illustration", "name": "Illustration",
         "prompt": "Editorial illustration, clean ink linework, flat muted colour, "
                   "subtle paper texture."},
    ]


DEFAULTS = {"backends": _default_backends, "models": _default_models, "loras": list,
            "identities": list, "styles": _default_styles, "characters": list,
            "outfits": _default_outfits}


class Library:
    """The configuration lists (`CLEAN`), each `<kind>.json` under `studio_dir()`.
    A missing file is the defaults; a list the user edits is written whole,
    atomically. Best effort both ways: an unreadable file is the defaults and
    a line in `problems`, an unwritable one raises for the editor to say."""

    def __init__(self, root=None):
        self.root = root or studio_dir()
        self.problems = []
        self.data = {kind: self._load(kind) for kind in CLEAN}

    def _path(self, kind):
        return os.path.join(self.root, kind + ".json")

    def _load(self, kind):
        raw, path = None, self._path(kind)
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    raw = json.load(f)
            except (OSError, ValueError) as e:
                self.problems.append("%s could not be read (%s); using the defaults."
                                     % (path, e))
        if not isinstance(raw, list):
            raw = DEFAULTS[kind]()
        out, seen = [], set()
        for d in raw:
            rec = CLEAN[kind](d)
            if rec is None:
                continue
            rec["id"] = unique_id(rec["id"], seen)
            seen.add(rec["id"])
            out.append(rec)
        return out

    def save(self, kind, records=None):
        if records is not None:
            cleaned, seen = [], set()
            for d in records:
                rec = CLEAN[kind](d)
                if rec is not None:
                    rec["id"] = unique_id(rec["id"], seen)
                    seen.add(rec["id"])
                    cleaned.append(rec)
            self.data[kind] = cleaned
        os.makedirs(self.root, exist_ok=True)
        path = self._path(kind)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data[kind], f, indent=2)
        os.replace(tmp, path)

    def all(self, kind):
        return self.data[kind]

    def get(self, kind, rid):
        return next((r for r in self.data[kind] if r["id"] == rid), None)

    def lora_by_file(self, filename):
        for r in self.data["loras"]:
            if r["file"] == filename or filename in r["files"].values():
                return r
        return None

    def preview_dir(self):
        return os.path.join(self.root, "lora-previews")

    def import_lora(self, found):
        """A LoRA profile from outside (`studio_civitai.profile`) into the
        library, unsaved -> (record, added). The same file - by hash, else by
        filename - is the same record: what the user already filled in stays,
        and only empty fields are taken from the import. Keys starting with
        "_" are the importer's own and are dropped."""
        found = {k: v for k, v in found.items() if not k.startswith("_")}
        fname = _str(found.get("file"))
        if not found.get("category"):
            found["category"] = guess_category(fname + " " + _str(found.get("name")))
        if not found.get("family"):
            found["family"] = guess_family(fname)
        sha = _str(found.get("sha256")).lower()
        rec = next((r for r in self.data["loras"] if sha and r.get("sha256") == sha),
                   None) or (self.lora_by_file(fname) if fname else None)
        if rec is not None:
            for k, v in found.items():
                if k in rec and v not in ("", None) and (
                        rec[k] in ("", None) or (k == "category" and rec[k] == "Other")
                        or (k == "name" and rec[k] == pretty(rec["file"]))):
                    rec[k] = v
            return rec, False
        new = clean_lora(dict(found, id=found.get("name") or fname))
        if new is None:
            raise ValueError("a LoRA needs a filename")
        new["id"] = unique_id(new["id"], {r["id"] for r in self.data["loras"]})
        self.data["loras"].append(new)
        return new, True

    def keep_reference(self, path, owner):
        """A copy of a reference picture under the studio folder, so a profile
        does not break when the original is moved or deleted. -> the copy."""
        import shutil
        with open(path, "rb") as f:
            digest = hashlib.sha1(f.read()).hexdigest()[:16]
        folder = os.path.join(self.root, "references", slug(owner))
        os.makedirs(folder, exist_ok=True)
        dest = os.path.join(folder, digest + os.path.splitext(path)[1].lower())
        if not os.path.exists(dest):
            shutil.copyfile(path, dest)
        return dest

    def merge_loras(self, backend_id, filenames, lora_dir=""):
        """Add every LoRA a backend has that the library does not know, with
        a name, category and family guessed from the filename, and a preview
        from `lora_dir` when that folder is on this PC. -> number added."""
        added = 0
        taken = {r["id"] for r in self.data["loras"]}
        for name in filenames:
            rec = self.lora_by_file(name)
            if rec is not None:
                if not rec.get("preview") and lora_dir:
                    rec["preview"] = find_preview(lora_dir, name)
                continue
            rec = clean_lora({"file": name, "family": guess_family(name),
                              "category": guess_category(name),
                              "preview": find_preview(lora_dir, name) if lora_dir else ""})
            rec["id"] = unique_id(rec["id"], taken)
            taken.add(rec["id"])
            self.data["loras"].append(rec)
            added += 1
        return added


def find_preview(lora_dir, filename):
    """The picture a LoRA manager left beside the file, if any."""
    if not lora_dir or not os.path.isdir(lora_dir):
        return ""
    stem = os.path.splitext(os.path.join(lora_dir, filename))[0]
    for suffix in (".preview.png", ".png", ".preview.jpg", ".jpg", ".jpeg", ".webp"):
        if os.path.isfile(stem + suffix):
            return stem + suffix
    return ""


# ============================================================ ComfyUI client

NOT_IN_LIST = re.compile(r"^(\w+): '(.*?)' not in \[(.*)\]$", re.S)


def explain(detail):
    """ComfyUI's refusal of a workflow, in words: each node's error with its
    id and class, and a value missing from a list (a model file, a sampler)
    named exactly - without the list of everything the server does have."""
    if not isinstance(detail, dict) or not detail.get("node_errors"):
        return _explain(detail)
    parts = []
    err = detail.get("error")
    if isinstance(err, dict) and err.get("message"):
        parts.append(err["message"])
    for node, info in detail["node_errors"].items():
        for e in info.get("errors", []):
            what = " ".join(str(e.get("details") or "").split())
            m = NOT_IN_LIST.match(what)
            if m:
                n = len([x for x in m.group(3).split(",") if x.strip()])
                what = "%s %r is not there (it has %d other%s)" % (
                    m.group(1), m.group(2), n, "" if n == 1 else "s")
            parts.append("node %s (%s): %s%s" % (node, info.get("class_type", "?"),
                                                e.get("message", ""),
                                                " - " + what if what else ""))
    return "; ".join(p for p in parts if p)


class ComfyUIClient:
    """Every call the app makes to one ComfyUI server. One instance per
    backend; no shared state between them but the class."""

    def __init__(self, backend, timeout=15):
        self.backend = backend
        self.url = backend["url"].rstrip("/")
        self.ws_url = backend.get("ws_url") or (
            re.sub(r"^http", "ws", self.url) + "/ws")
        self.timeout = timeout
        self.client_id = uuid.uuid4().hex
        self.uploaded = set()
        self._nodes = None

    # --------------------------------------------------------------- HTTP
    def _open(self, req, timeout=None):
        try:
            return urllib.request.urlopen(req, timeout=timeout or self.timeout)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:2000]
            try:
                detail = json.loads(body)
            except ValueError:
                detail = body
            raise ComfyError("%s answered HTTP %d: %s"
                             % (self.backend["name"], e.code, explain(detail)))
        except (urllib.error.URLError, OSError) as e:
            reason = getattr(e, "reason", e)
            local = re.match(r"^https?://(127\.0\.0\.1|localhost)(:|/|$)", self.url)
            where = ("Nothing is answering at %s on this PC: no ComfyUI is running here "
                     "(Studio Assist itself is not a ComfyUI)." % self.url if local else
                     "Cannot reach %s at %s. ComfyUI has to be running there, started "
                     "with --listen." % (self.backend["name"], self.url))
            start = self.backend.get("start")
            raise Unreachable("%s (%s)%s" % (where, reason,
                                             " Start it: %s" % start if start else ""))

    def get_json(self, path, timeout=None):
        with self._open(self.url + path, timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def post_json(self, path, payload, timeout=None):
        req = urllib.request.Request(
            self.url + path, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        with self._open(req, timeout) as r:
            body = r.read().decode("utf-8")
            return json.loads(body) if body.strip() else {}

    # ---------------------------------------------------------- reading
    def health(self):
        """-> {"ok", "detail", "device", "vram_free", "vram_total", "queue"}."""
        try:
            try:
                stats = self.get_json("/system_stats", timeout=5)
            except Unreachable as e:
                # ComfyUI stops answering HTTP while it stages a model or
                # finishes a job: a busy server, not a dead one. A refusal
                # fails at once; only a timeout is worth the longer wait.
                if "timed out" not in str(e).lower():
                    raise
                stats = self.get_json("/system_stats", timeout=15)
            q = self.get_queue()
        except (ComfyError, ValueError) as e:
            return {"ok": False, "detail": str(e), "queue": 0}
        if not isinstance(stats, dict) or "comfyui_version" not in (stats.get("system") or {}):
            return {"ok": False, "queue": 0,
                    "detail": "Something answers at %s, but it is not ComfyUI (its "
                              "/system_stats has no comfyui_version)." % self.url}
        dev = (stats.get("devices") or [{}])[0]
        n = len(q.get("queue_running") or []) + len(q.get("queue_pending") or [])
        return {"ok": True, "detail": "ComfyUI %s" % stats.get("system", {}).get(
                    "comfyui_version", "?"),
                "device": dev.get("name", ""), "vram_free": dev.get("vram_free"),
                "vram_total": dev.get("vram_total"), "queue": n}

    def get_models(self, kind):
        names = self.get_json("/models/" + urllib.parse.quote(kind))
        return [n for n in names if isinstance(n, str)] if isinstance(names, list) else []

    def get_loras(self):
        return self.get_models("loras")

    def inventory(self):
        """{kind: set of filenames}; a kind the server does not have is empty."""
        out = {}
        for kind in MODEL_KINDS:
            try:
                out[kind] = set(self.get_models(kind))
            except ComfyError:
                out[kind] = set()
        return out

    def node_types(self, fresh=False):
        """Every node class the server has. Cached; `fresh` asks again, as a
        full check does, so a node installed since is seen."""
        if self._nodes is None or fresh:
            self._nodes = set(self.get_json("/object_info", timeout=60))
        return self._nodes

    def probe(self):
        """Every endpoint the studio uses, one by one: -> [(what, ok, detail)].
        /prompt is sent a graph with no outputs, which ComfyUI refuses before
        queueing anything - proof it validates, with nothing run."""
        out = []

        def step(what, fn):
            try:
                out.append((what, True, fn()))
            except Exception as e:
                out.append((what, False, str(e)))

        def stats():
            s = self.get_json("/system_stats", timeout=5)
            dev = (s.get("devices") or [{}])[0]
            return "ComfyUI %s, %s, %.1f of %.1f GB free" % (
                s["system"]["comfyui_version"], dev.get("name", "?"),
                (dev.get("vram_free") or 0) / 1e9, (dev.get("vram_total") or 0) / 1e9)

        def prompt():
            try:
                self.post_json("/prompt", {"prompt": {}, "client_id": self.client_id})
            except ComfyError as e:
                if "HTTP 400" in str(e):
                    return "validates (refused an empty graph, as it should)"
                raise
            raise ComfyError("accepted an empty graph - is this ComfyUI?")

        def socket():
            events = queue.Queue()
            watch = self.watch(events)
            if watch.error:
                raise ComfyError(watch.error)
            try:
                msg = events.get(timeout=5)   # ComfyUI greets a client with its status
                return "connected; first message %r" % msg.get("type")
            except queue.Empty:
                return "connected, but no greeting in 5 s"
            finally:
                watch.close()

        step("/system_stats", stats)
        step("/object_info", lambda: "%d node types" % len(self.node_types(fresh=True)))
        step("/prompt", prompt)
        step("/history", lambda: "%d recent entries" % len(
            self.get_json("/history?max_items=5")))
        step("websocket " + self.ws_url, socket)
        return out

    def get_queue(self):
        return self.get_json("/queue", timeout=5)

    def get_history(self, prompt_id):
        return self.get_json("/history/" + urllib.parse.quote(prompt_id)).get(prompt_id)

    def fetch(self, f):
        q = urllib.parse.urlencode({"filename": f["filename"],
                                    "subfolder": f.get("subfolder", ""),
                                    "type": f.get("type", "output")})
        with self._open(self.url + "/view?" + q, 60) as r:
            return r.read()

    # ---------------------------------------------------------- writing
    def upload_image(self, path):
        """Upload a local picture; -> the name ComfyUI's LoadImage takes. Named
        by content, so the same picture is sent once per backend however
        many jobs use it."""
        with open(path, "rb") as f:
            data = f.read()
        name = "studio_%s%s" % (hashlib.sha1(data).hexdigest()[:16],
                                os.path.splitext(path)[1].lower() or ".png")
        if name in self.uploaded:
            return name
        boundary = "----studio" + uuid.uuid4().hex
        body = b""
        for k, v in (("overwrite", "true"), ("type", "input")):
            body += ("--%s\r\nContent-Disposition: form-data; name=\"%s\"\r\n\r\n%s\r\n"
                     % (boundary, k, v)).encode()
        body += ("--%s\r\nContent-Disposition: form-data; name=\"image\"; filename=\"%s\"\r\n"
                 "Content-Type: application/octet-stream\r\n\r\n" % (boundary, name)).encode()
        body += data + ("\r\n--%s--\r\n" % boundary).encode()
        req = urllib.request.Request(
            self.url + "/upload/image", data=body, method="POST",
            headers={"Content-Type": "multipart/form-data; boundary=" + boundary})
        with self._open(req, 120) as r:
            res = json.loads(r.read().decode("utf-8"))
        name = res.get("name", name)
        if res.get("subfolder"):
            name = res["subfolder"] + "/" + name
        self.uploaded.add(name)
        return name

    def queue_workflow(self, graph):
        res = self.post_json("/prompt", {"prompt": graph, "client_id": self.client_id})
        pid = res.get("prompt_id")
        if not pid:
            raise ComfyError("%s did not queue the workflow: %s"
                             % (self.backend["name"], _explain(res)))
        return pid

    def cancel_job(self, prompt_id):
        """Stop one prompt and nothing else: interrupt it if it is the one
        running, take it out of the queue if it is waiting. A bare interrupt
        would stop whatever is running - the chat tab's render included."""
        try:
            q = self.get_queue()
        except ComfyError:
            return False
        running = [item[1] for item in q.get("queue_running") or [] if len(item) > 1]
        pending = [item[1] for item in q.get("queue_pending") or [] if len(item) > 1]
        if prompt_id in running:
            self.post_json("/interrupt", {"prompt_id": prompt_id})
            return True
        if prompt_id in pending:
            self.post_json("/queue", {"delete": [prompt_id]})
            return True
        return False

    def free(self):
        """Let go of VRAM when nothing else is queued here."""
        try:
            q = self.get_queue()
            if q.get("queue_running") or q.get("queue_pending"):
                return False
            self.post_json("/free", {"unload_models": True, "free_memory": True}, timeout=10)
            return True
        except ComfyError:
            return False

    # --------------------------------------------------------- progress
    def listen_for_progress(self, prompt_id, on_event, stop=None, timeout=JOB_TIMEOUT,
                            watch=None):
        """Wait for a prompt, reporting as it goes; -> its history entry, or
        None when `stop()` said to stop. `on_event(kind, data)` gets
        ("queued", prompts ahead), ("executing", node_id), ("cached", [node
        ids]), ("progress", (value, max, node_id)), ("busy", seconds silent)
        and ("socket", why) once when there is no live progress.

        `watch` is the socket opened *before* the prompt was queued (`watch()`),
        so its first events are not missed; without one it is opened here.
        Progress comes over ComfyUI's WebSocket; the end is read from /history
        either way, every two seconds, so a socket that drops or never opens
        costs the step counter and nothing else. A ComfyUI staging a model
        stops answering HTTP for half a minute: that is "busy", not gone."""
        own = watch is None
        if own:
            watch = self.watch()
        if watch.error:
            on_event("socket", watch.error)
        events = watch.events
        deadline = time.monotonic() + timeout
        next_poll, silent, started = 0.0, None, False
        heard = time.monotonic()      # the last event, for ("quiet", seconds)
        try:
            while True:
                if stop is not None and stop():
                    return None
                now = time.monotonic()
                if now >= next_poll:
                    next_poll = now + 2.0
                    try:
                        entry = self.get_history(prompt_id)
                        if not started and not entry:
                            ahead = self.position(prompt_id)
                            if ahead is None or ahead < 0:
                                started = ahead is not None
                            else:
                                on_event("queued", ahead)
                        silent = None
                    except Unreachable:
                        entry = None
                        silent = silent or now
                        on_event("busy", int(now - silent))
                    if entry and (entry.get("status", {}).get("completed")
                                  or entry.get("outputs")
                                  or entry.get("status", {}).get("status_str") == "error"):
                        return entry
                    if started and not watch.error and now - heard > QUIET_AFTER:
                        on_event("quiet", int(now - heard))
                if now > deadline:
                    raise ComfyError("%s has not finished prompt %s after %d s; it stays "
                                     "queued there." % (self.backend["name"], prompt_id, timeout))
                try:
                    msg = events.get(timeout=0.5)
                except queue.Empty:
                    continue
                data = msg.get("data") or {}
                if data.get("prompt_id") not in (None, prompt_id):
                    continue
                heard = time.monotonic()
                kind = msg.get("type")
                if kind == "executing" and data.get("node") is not None:
                    started = True
                    on_event("executing", str(data["node"]))
                elif kind == "execution_cached" and data.get("nodes"):
                    started = True
                    on_event("cached", [str(n) for n in data["nodes"]])
                elif kind == "progress":
                    started = True
                    on_event("progress", (data.get("value", 0), data.get("max", 0),
                                          str(data.get("node", ""))))
                elif kind in ("execution_success", "execution_error",
                              "execution_interrupted"):
                    next_poll = 0.0       # read the result now
        finally:
            if own:
                watch.close()

    def position(self, prompt_id):
        """Prompts ahead of this one in the queue; -1 while it runs; None
        when it is in neither list (finished, or never queued)."""
        q = self.get_queue()
        if any(len(i) > 1 and i[1] == prompt_id for i in q.get("queue_running") or []):
            return -1
        pending = sorted((i for i in q.get("queue_pending") or [] if len(i) > 1),
                         key=lambda i: i[0])
        for n, item in enumerate(pending):
            if item[1] == prompt_id:
                return n + len(q.get("queue_running") or [])
        return None

    def watch(self, events=None):
        """ComfyUI's WebSocket for this client, read on a thread of its own
        into `events` (blocking reads, so a frame is never cut by a timeout).
        -> a Watch; its `error` says why there is no socket, never silently."""
        events = events if events is not None else queue.Queue()
        try:
            from studio_milanote import WebSocket
            sep = "&" if "?" in self.ws_url else "?"
            ws = WebSocket(self.ws_url + sep + "clientId=" + self.client_id, timeout=5)
            ws.sock.settimeout(None)
        except Exception as e:
            return Watch(None, events, "No live progress: the WebSocket %s would not open "
                                       "(%s); the job is followed through /history instead."
                         % (self.ws_url, e))

        def read():
            try:
                while True:
                    text = ws.recv()
                    try:
                        msg = json.loads(text)
                    except (ValueError, TypeError):
                        continue          # a binary preview frame
                    if isinstance(msg, dict):
                        events.put(msg)
            except Exception:
                return                    # closed, by us or by the server

        threading.Thread(target=read, daemon=True).start()
        return Watch(ws, events, "")


class Watch:
    """An open (or failed) progress socket; see ComfyUIClient.watch."""

    def __init__(self, ws, events, error):
        self.ws, self.events, self.error = ws, events, error

    def close(self):
        if self.ws is not None:
            try:
                self.ws.close()
            except Exception:
                pass
            self.ws = None


# ======================================================= workflow templates

class TemplateError(ValueError):
    pass


def load_workflow(wid, folder=None):
    path = os.path.join(folder or WORKFLOWS_DIR, wid + ".json")
    try:
        with open(path, encoding="utf-8") as f:
            wf = json.load(f)
    except (OSError, ValueError) as e:
        raise TemplateError("Workflow template %s could not be read: %s" % (path, e))
    if not isinstance(wf.get("graph"), dict):
        raise TemplateError("Workflow template %s has no graph." % path)
    wf.setdefault("id", wid)
    return wf


def list_workflows(folder=None):
    folder = folder or WORKFLOWS_DIR
    out = []
    for name in sorted(os.listdir(folder)) if os.path.isdir(folder) else ():
        if name.endswith(".json"):
            try:
                out.append(load_workflow(name[:-5], folder))
            except TemplateError:
                pass
    return out


PLACEHOLDER = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")


def fill(wf, values, loras=()):
    """The adapter: a template and the values for this job -> an API-format
    graph ComfyUI will run.

    - `"{{name}}"` as a whole value becomes the value itself, typed (a seed
      stays an int, a link stays a list); inside a longer string it is text.
    - A node with `"_when": "x"` is kept only when x is set and truthy
      (`["x", "y"]`: when either is - one ControlNet loader for the pose
      and the depth map); `"_unless": "x"` the reverse.
    - `switches` name a link chosen by a value: `{"when": "x", "then": link,
      "else": link}`, then used as `"{{name}}"`. A branch may be
      `"{{other}}"`, a switch named before it: the depth ControlNet hangs
      off the pose's output when there is one, else off the prompt.
    - `lora_chain` names the model (and clip) outputs the LoRAs hang off;
      `{{model_out}}` / `{{clip_out}}` are the end of the chain.
    Anything left unfilled, or a link to a node that was dropped, is an error
    naming it: a broken template must not reach ComfyUI as a puzzle."""
    v = dict(wf.get("defaults") or {})
    v.update({k: x for k, x in values.items() if x is not None})
    graph = copy.deepcopy(wf["graph"])
    chain = wf.get("lora_chain") or {}
    if chain:
        model, clip = chain.get("model"), chain.get("clip")
        for i, (name, strength) in enumerate(loras):
            nid = "lora%d" % (i + 1)
            if clip:
                graph[nid] = {"class_type": "LoraLoader", "inputs": {
                    "model": model, "clip": clip, "lora_name": name,
                    "strength_model": strength, "strength_clip": strength}}
                model, clip = [nid, 0], [nid, 1]
            else:
                graph[nid] = {"class_type": "LoraLoaderModelOnly", "inputs": {
                    "model": model, "lora_name": name, "strength_model": strength}}
                model = [nid, 0]
        v["model_out"] = model
        if clip:
            v["clip_out"] = clip
    elif loras:
        raise TemplateError("Workflow %s takes no LoRAs." % wf["id"])
    for name, sw in (wf.get("switches") or {}).items():
        pick = sw["then"] if v.get(sw["when"]) else sw["else"]
        m = PLACEHOLDER.fullmatch(pick.strip()) if isinstance(pick, str) else None
        if m:
            if m.group(1) not in v:
                raise TemplateError("Workflow %s: switch %s names %s, which is not a "
                                    "switch before it." % (wf["id"], name, m.group(1)))
            pick = v[m.group(1)]
        v[name] = pick

    kept = {}
    for nid, node in graph.items():
        if "_when" in node and not any(v.get(w) for w in _names(node["_when"])):
            continue
        if "_unless" in node and v.get(node["_unless"]):
            continue
        kept[nid] = {"class_type": node["class_type"], "inputs": node.get("inputs", {})}
        if "_meta" in node:
            kept[nid]["_meta"] = node["_meta"]

    def sub(x, where):
        if isinstance(x, str):
            m = PLACEHOLDER.fullmatch(x.strip())
            if m:
                if m.group(1) not in v:
                    raise TemplateError("Workflow %s needs a value for %s (%s)."
                                        % (wf["id"], m.group(1), where))
                return v[m.group(1)]

            def one(mm):
                if mm.group(1) not in v:
                    raise TemplateError("Workflow %s needs a value for %s (%s)."
                                        % (wf["id"], mm.group(1), where))
                return str(v[mm.group(1)])
            return PLACEHOLDER.sub(one, x)
        if isinstance(x, list):
            return [sub(i, where) for i in x]
        return x

    for nid, node in kept.items():
        node["inputs"] = {k: sub(x, "node %s input %s" % (nid, k))
                          for k, x in node["inputs"].items()}
    for nid, node in kept.items():
        for k, x in node["inputs"].items():
            if (isinstance(x, list) and len(x) == 2 and isinstance(x[0], str)
                    and isinstance(x[1], int) and x[0] not in kept):
                raise TemplateError("Workflow %s: node %s input %s links to node %s, which "
                                    "is not in the graph." % (wf["id"], nid, k, x[0]))
    return kept


def missing_nodes(graph, available):
    return sorted({n["class_type"] for n in graph.values()} - set(available))


# ============================================================ the face pass
# The chat bridge's face detail pass (studio_comfy_mcp.face_detail), for a
# template that declares `face_detail`: SAM3 finds every face in the finished
# picture in the same run that makes it (`add_face_finder`), then a second run
# (`face_graph`) crops each face FACE_PAD times its size, redraws it at
# FACE_EDIT px with the job's own model, LoRAs and guidance at `face_denoise`,
# and blends it back through a soft oval. The identity LoRA is on the model
# that redraws the face, so the pass is where most of the likeness is drawn.
FACE_REDRAWN = 0.02            # the oval's weight beyond which the face pass redraws
FACE_NODES = {"CheckpointLoaderSimple", "SAM3_Detect", "PreviewAny", "GetImageSize",
              "ImageCropV2", "ImageScale", "ImageToMask", "ImageCompositeMasked", "LoadImage",
              "SetLatentNoiseMask", "ThresholdMask", "CLIPTextEncode", "MaskComposite",
              "GrowMask", "MaskToImage", "ImageBlur", "SolidMask"}
FACE_BOX_GROW = 0.15           # of its size the found face's box grows, as the part always blended
FACE_HEAD_GROW = 12            # px at FACE_EDIT the head's mask grows before it is softened
FACE_HEAD_SOFT = (21, 7.0)     # ImageBlur radius and sigma of its edge, at FACE_EDIT


# The Visual Critic's redraws (Studio._refine). A hand at 0.6 kept its shape
# in the reverted hand pass of 2026-09-25, and 0.85-0.9 left double hands, so
# a local fix stays at 0.6: it mends fingers, it does not re-pose them.
CRITIC_DENOISE = {"FACE_CORRECTION": 0.45, "LOCAL_INPAINT": 0.6,
                  "OBJECT_CORRECTION": 0.6, "GLOBAL_REFINEMENT": 0.2}
REGION_PAD = 1.6


def png_size(raw):
    """(width, height) from a PNG's header, or None."""
    if raw[:8] == b"\x89PNG\r\n\x1a\n" and len(raw) >= 24:
        return int.from_bytes(raw[16:20], "big"), int.from_bytes(raw[20:24], "big")
    return None


def _is_link(x):
    return isinstance(x, list) and len(x) == 2 and isinstance(x[0], str) and isinstance(x[1], int)


def add_face_finder(graph, sam3, pixels=None, prompt="face:8"):
    """Into a filled graph: SAM3's face boxes and the picture's size for its
    SaveImage's picture (or `pixels`), as PreviewAny text (read by
    `face_boxes`). `prompt` finds something else: "hand:4"."""
    if pixels is None:
        save = next(nid for nid, n in graph.items() if n["class_type"] == "SaveImage")
        pixels = graph[save]["inputs"]["images"]
    graph["fd1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": sam3}}
    graph["fd2"] = {"class_type": "CLIPTextEncode", "inputs": {"text": prompt,
                                                               "clip": ["fd1", 1]}}
    graph["fd3"] = {"class_type": "SAM3_Detect", "inputs": {
        "model": ["fd1", 0], "image": pixels, "conditioning": ["fd2", 0], "threshold": 0.3,
        "refine_iterations": 0, "individual_masks": True}}
    graph["fd4"] = {"class_type": "PreviewAny", "inputs": {"source": ["fd3", 1]}}
    graph["fd5"] = {"class_type": "GetImageSize", "inputs": {"image": pixels}}
    graph["fd6"] = {"class_type": "PreviewAny", "inputs": {"source": ["fd5", 0]}}
    graph["fd7"] = {"class_type": "PreviewAny", "inputs": {"source": ["fd5", 1]}}
    return graph


ITEM_NODES = {"LoadImage", "ImageStitch", "FluxKontextImageScale", "VAEEncode",
              "ReferenceLatent"}


def add_item_refs(graph, section, images):
    """Into a filled graph: the item pictures (LoadImage names, in order) as
    one reference for FLUX Kontext - side by side on white, scaled to a size
    Kontext was trained on, encoded, and set as the reference latent on the
    conditioning the template's `items` section names. One picture of them
    all, not a latent each: Kontext [dev] learnt from a single reference."""
    node, key = section["conditioning"]
    last = None
    for n, name in enumerate(images, 1):
        graph["it%d" % n] = {"class_type": "LoadImage", "inputs": {"image": name}}
        if last is None:
            last = ["it%d" % n, 0]
            continue
        graph["is%d" % n] = {"class_type": "ImageStitch", "inputs": {
            "image1": last, "image2": ["it%d" % n, 0], "direction": "right",
            "match_image_size": True, "spacing_width": 32, "spacing_color": "white"}}
        last = ["is%d" % n, 0]
    graph["ik1"] = {"class_type": "FluxKontextImageScale", "inputs": {"image": last}}
    graph["ik2"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["ik1", 0],
                                                          "vae": section["vae"]}}
    graph["ik3"] = {"class_type": "ReferenceLatent", "inputs": {
        "conditioning": graph[node]["inputs"][key], "latent": ["ik2", 0]}}
    graph[node]["inputs"][key] = ["ik3", 0]
    return graph


def face_boxes(entry):
    """What add_face_finder's nodes said -> (width, height, [(x, y, w, h)]),
    faces under FACE_MIN px dropped. None when the finder said nothing."""
    out = entry.get("outputs") or {}

    def text(node):
        t = (out.get(node) or {}).get("text") or []
        return json.loads(t[0]) if t else None
    try:
        boxes, width, height = text("fd4"), text("fd6"), text("fd7")
    except ValueError:
        return None
    if boxes is None or width is None or height is None:
        return None
    boxes = boxes[0] if boxes and isinstance(boxes[0], list) else boxes
    return int(width), int(height), [(b["x"], b["y"], b["width"], b["height"])
                                     for b in boxes or []
                                     if max(b["width"], b["height"]) >= FACE_MIN]


def face_crops(width, height, boxes, pad=FACE_PAD, keep=()):
    """The squares to redraw: every face whose padded square is smaller than
    FACE_EDIT (one already that big was drawn at full size), and every face
    whose box index is in `keep` (a face with a likeness to draw) whatever
    its size."""
    return [c for _, c in indexed_crops(width, height, boxes, pad, keep)]


def indexed_crops(width, height, boxes, pad=FACE_PAD, keep=()):
    """face_crops as [(box index, square)]."""
    out = []
    for i, b in enumerate(boxes):
        c = head_square(b, width, height, pad)
        if c["width"] < FACE_EDIT or i in keep:
            out.append((i, c))
    return out


# A scene says where each person's face is (studio_scene.face_targets); the
# finder's boxes are matched to them nearest first, a box taken by one person
# only, and only within FACE_MATCH of the face's size (or FACE_MATCH_FRAME of
# the frame, for a face the finder saw small): the model may place a head a
# little off the mannequin's, but a face across the frame is someone else's.
FACE_MATCH = 3.0
FACE_MATCH_FRAME = 0.06


def match_faces(width, height, boxes, people):
    """-> {box index: person} for the people (dicts with "at", as fractions
    of the frame) whose face the finder found."""
    pairs = []
    for i, (x, y, w, h) in enumerate(boxes):
        cx, cy = x + w / 2.0, y + h / 2.0
        reach = max(FACE_MATCH * max(w, h), FACE_MATCH_FRAME * max(width, height))
        for j, person in enumerate(people):
            px, py = person["at"][0] * width, person["at"][1] * height
            d = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
            if d <= reach:
                pairs.append((d, i, j))
    out, used = {}, set()
    for d, i, j in sorted(pairs):
        if i not in out and j not in used:
            out[i] = people[j]
            used.add(j)
    return out


# A face given a picture is redrawn with PuLID (lldacing's ComfyUI_PuLID_Flux_ll)
# on the redraw's model: InsightFace reads the picture's face and FLUX draws
# that face in the pose and light the crop already has. It needs the redraw to
# go deep (studio_scene.FACE_LIKENESS: at 0.45 the face barely moved, at 0.8-0.9
# it was the person, measured 2026-09-25) and only FLUX.1 takes it. The picture
# should show that one face: PuLID takes the biggest face in it.
PULID_NODES = {"PulidFluxModelLoader", "PulidFluxEvaClipLoader",
               "PulidFluxInsightFaceLoader", "ApplyPulidFlux"}
PULID_WEIGHT = 1.0
PULID_FAMILIES = {"flux1"}
# The same faces go into the picture itself, each confined to its person's
# head (`region`, a mask PuLID scales to the latent): the picture is then
# drawn with their heads, hair and skin, and the face pass only refines. A
# face drawn into a stranger's head at the end came out a sticker - a smooth
# pale face on a tan neck, a halo where the stranger's hair had been.
PULID_BASE_WEIGHT = 1.0
REGION_EDGE = 64                # px on a region mask's long edge


def region_png(region, width, height):
    """PNG bytes: white over `region` ([x0, y0, x1, y1] fractions) on black,
    at the frame's shape, REGION_EDGE px on its long edge."""
    import studio_icons
    k = REGION_EDGE / float(max(width, height))
    w, h = max(1, int(round(width * k))), max(1, int(round(height * k)))
    x0, y0 = int(region[0] * w), int(region[1] * h)
    x1, y1 = int(round(region[2] * w)), int(round(region[3] * h))
    px = bytearray(w * h * 4)
    for y in range(h):
        for x in range(w):
            v = 255 if x0 <= x < x1 and y0 <= y < y1 else 0
            px[(y * w + x) * 4:(y * w + x) * 4 + 4] = bytes((v, v, v, 255))
    return studio_icons.png(bytes(px), w, h)


def add_pulid(graph, pulid_file, faces, weight=PULID_BASE_WEIGHT):
    """Into a filled graph: one ApplyPulidFlux per (face picture, region
    mask) - LoadImage names - chained on the model its samplers share, each
    confined to its mask. Every KSampler on that model takes the chain."""
    samplers = [n for n in graph.values() if n["class_type"] == "KSampler"]
    if not samplers or not faces:
        return graph
    base = samplers[0]["inputs"]["model"]
    graph["pb1"] = {"class_type": "PulidFluxModelLoader", "inputs": {"pulid_file": pulid_file}}
    graph["pb2"] = {"class_type": "PulidFluxEvaClipLoader", "inputs": {}}
    graph["pb3"] = {"class_type": "PulidFluxInsightFaceLoader", "inputs": {"provider": "CUDA"}}
    last = base
    for i, (face, mask) in enumerate(faces, 1):
        n = "pb_%d" % i
        graph[n + "f"] = {"class_type": "LoadImage", "inputs": {"image": face}}
        graph[n + "m"] = {"class_type": "LoadImage", "inputs": {"image": mask}}
        graph[n + "k"] = {"class_type": "ImageToMask", "inputs": {"image": [n + "m", 0],
                                                                  "channel": "red"}}
        graph[n] = {"class_type": "ApplyPulidFlux", "inputs": {
            "model": last, "pulid_flux": ["pb1", 0], "eva_clip": ["pb2", 0],
            "face_analysis": ["pb3", 0], "image": [n + "f", 0], "weight": weight,
            "start_at": 0.0, "end_at": 1.0, "attn_mask": [n + "k", 0]}}
        last = [n, 0]
    for node in samplers:
        if node["inputs"]["model"] == base:
            node["inputs"]["model"] = last
    return graph


def _restated(g, links, prompt, text, tag):
    """The positive and negative links of `links` with every node between
    them and the CLIPTextEncode saying `prompt` copied (ids + `tag`) to say
    `text` instead: one face's own conditioning beside the shared one."""
    memo = {}

    def says(nid):
        if nid not in memo:
            n = g[nid]
            memo[nid] = ((n["class_type"] == "CLIPTextEncode" and n["inputs"].get("text") == prompt)
                         or any(says(x[0]) for x in n["inputs"].values() if _is_link(x)))
        return memo[nid]

    def copy_of(nid):
        if not says(nid):
            return nid
        new = nid + tag
        if new not in g:
            n = g[nid]
            inputs = {k: [copy_of(x[0]), x[1]] if _is_link(x) else x
                      for k, x in n["inputs"].items()}
            if n["class_type"] == "CLIPTextEncode" and inputs.get("text") == prompt:
                inputs["text"] = text
            g[new] = {"class_type": n["class_type"], "inputs": inputs}
        return new
    return ([copy_of(links["positive"][0]), links["positive"][1]],
            [copy_of(links["negative"][0]), links["negative"][1]])


def paste_graph(image, faces, seed, prefix):
    """The third run, after the face pass: `image` (a LoadImage name) with
    each person's own face put over theirs by PASTE_NODE where one of their
    photos (`faces`: [{"name", "box": [x, y, w, h], "references": [LoadImage
    names]}]) is turned close enough to the drawn head. No redraw after."""
    return {"pi": {"class_type": "LoadImage", "inputs": {"image": image}},
            "pp": {"class_type": PASTE_NODE, "inputs": {
                "image": ["pi", 0], "faces": json.dumps(faces), "seed": int(seed) % 2 ** 32}},
            "ps": {"class_type": "SaveImage", "inputs": {"images": ["pp", 0],
                                                         "filename_prefix": prefix}}}


def paste_report(entry):
    """PASTE_NODE's report from a finished run: [{"name", "pasted", ...}] or []."""
    text = ((entry.get("outputs") or {}).get("pp") or {}).get("text") or []
    try:
        report = json.loads(text[0])
    except (IndexError, TypeError, ValueError):
        return []
    return report if isinstance(report, list) else []


def face_graph(wf, values, loras, image, crops, oval, prefix, faces=None, pulid_file=None,
               boxes=None):
    """The second run: `image` (a LoadImage name) with each crop redrawn and
    blended back through `oval`, saved under `prefix`. The model, VAE and
    conditioning come from the template's `face_detail` section, filled like
    the rest (so the LoRA chain is the job's), keeping only the nodes they
    need. A crop may carry its own `edit` (width, height) to redraw at and
    `mask` False to lay the redraw back whole (the Visual Critic's
    whole-picture pass); a face crop has neither. `head` False (a hand, an
    object) blends back through the oval alone, not SAM3's head.

    `faces`, beside `crops`, says who each is (None for no one known):
    {"words": the person's own words for FACE_PROMPT, "image": a LoadImage
    name of their face picture or None, "denoise": this face's}. A face with
    an image is drawn to it through PuLID (`pulid_file`). `boxes`, beside
    `crops`, are the faces the finder found (x, y, w, h): each is blended
    back whole, whatever SAM3 makes of the head around it."""
    fd = wf["face_detail"]
    values = dict(wf.get("defaults") or {}, **{k: x for k, x in values.items() if x is not None})
    extra = dict(fd.get("nodes") or {})
    extra["fd_links"] = {"class_type": "_links", "inputs": {
        k: fd[k] for k in ("model", "vae", "positive", "negative")}}
    g = fill(dict(wf, graph=dict(wf["graph"], **extra)), values, loras)
    links = g.pop("fd_links")["inputs"]
    keep, todo = set(), [x[0] for x in links.values()]
    while todo:
        nid = todo.pop()
        if nid in keep:
            continue
        keep.add(nid)
        todo.extend(x[0] for x in g[nid]["inputs"].values() if _is_link(x))
    g = {k: g[k] for k in keep}
    g["fi"] = {"class_type": "LoadImage", "inputs": {"image": image}}
    g["fo"] = {"class_type": "LoadImage", "inputs": {"image": oval}}
    faces = list(faces or []) + [None] * (len(crops) - len(faces or []))
    # What is blended back is the head - the redrawn one and the one it
    # replaces, so no old hair is left around the new - found by SAM3 and
    # kept inside the oval (a neighbour's head in the crop is not). The oval
    # alone put a disc of redrawn background round every face (2026-09-25).
    if values.get("sam3"):
        g["fh1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": values["sam3"]}}
        g["fh2"] = {"class_type": "CLIPTextEncode", "inputs": {"text": "head",
                                                               "clip": ["fh1", 1]}}
        g["fh3"] = {"class_type": "ImageScale", "inputs": {
            "image": ["fo", 0], "upscale_method": "bilinear", "width": FACE_EDIT,
            "height": FACE_EDIT, "crop": "disabled"}}
        g["fh4"] = {"class_type": "ImageToMask", "inputs": {"image": ["fh3", 0],
                                                            "channel": "red"}}
    if any(f and f.get("image") for f in faces):
        g["pl1"] = {"class_type": "PulidFluxModelLoader", "inputs": {"pulid_file": pulid_file}}
        g["pl2"] = {"class_type": "PulidFluxEvaClipLoader", "inputs": {}}
        g["pl3"] = {"class_type": "PulidFluxInsightFaceLoader", "inputs": {"provider": "CUDA"}}
    seed, last = int(values["seed"]), ["fi", 0]
    for i, crop in enumerate(crops):
        n, side, tall = "fc%d_" % (i + 1), crop["width"], crop["height"]
        ew, eh = crop.get("edit") or (FACE_EDIT, FACE_EDIT)
        region = {k: crop[k] for k in ("x", "y", "width", "height")}
        face = faces[i] or {}
        model, positive, negative = links["model"], links["positive"], links["negative"]
        if face.get("words") and values.get("face_prompt"):
            positive, negative = _restated(g, links, values["face_prompt"],
                                           FACE_PROMPT % face["words"], "_" + n[:-1])
        if face.get("image"):
            g[n + "r"] = {"class_type": "LoadImage", "inputs": {"image": face["image"]}}
            g[n + "p"] = {"class_type": "ApplyPulidFlux", "inputs": {
                "model": model, "pulid_flux": ["pl1", 0], "eva_clip": ["pl2", 0],
                "face_analysis": ["pl3", 0], "image": [n + "r", 0], "weight": PULID_WEIGHT,
                "start_at": 0.0, "end_at": 1.0}}
            model = [n + "p", 0]
        g[n + "1"] = {"class_type": "ImageCropV2", "inputs": {"image": last,
                                                              "crop_region": region}}
        g[n + "2"] = {"class_type": "ImageScale", "inputs": {
            "image": [n + "1", 0], "upscale_method": "lanczos", "width": ew,
            "height": eh, "crop": "disabled"}}
        g[n + "3"] = {"class_type": "VAEEncode", "inputs": {"pixels": [n + "2", 0],
                                                            "vae": links["vae"]}}
        # Only the oval is redrawn: the crop around it is held as it is, so a
        # deep redraw (a likeness) cannot change the background it is blended
        # back onto - at 0.85 an unmasked crop came back as a visible disc.
        # The redrawn region is the oval's whole reach, hard-edged (a soft
        # noise mask left a pale ring half-redrawn); the soft oval blends it
        # back, fading out before that edge.
        # The whole-picture pass (`mask` False) is redrawn everywhere.
        latent = [n + "3", 0]
        if crop.get("mask") is not False:
            g[n + "3m"] = {"class_type": "ImageScale", "inputs": {
                "image": ["fo", 0], "upscale_method": "bilinear", "width": ew,
                "height": eh, "crop": "disabled"}}
            g[n + "3k"] = {"class_type": "ImageToMask", "inputs": {"image": [n + "3m", 0],
                                                                   "channel": "red"}}
            g[n + "3h"] = {"class_type": "ThresholdMask", "inputs": {"mask": [n + "3k", 0],
                                                                    "value": FACE_REDRAWN}}
            g[n + "3n"] = {"class_type": "SetLatentNoiseMask", "inputs": {
                "samples": latent, "mask": [n + "3h", 0]}}
            latent = [n + "3n", 0]
        g[n + "4"] = {"class_type": "KSampler", "inputs": {
            "seed": (seed + i + 1) % (MAX_SEED + 1), "steps": values["steps"], "cfg": 1.0,
            "sampler_name": values["sampler"], "scheduler": values["scheduler"],
            "denoise": face.get("denoise") or values["face_denoise"], "model": model,
            "positive": positive, "negative": negative,
            "latent_image": latent}}
        g[n + "5"] = {"class_type": "VAEDecode", "inputs": {"samples": [n + "4", 0],
                                                            "vae": links["vae"]}}
        g[n + "6"] = {"class_type": "ImageScale", "inputs": {
            "image": [n + "5", 0], "upscale_method": "lanczos", "width": side,
            "height": tall, "crop": "disabled"}}
        if crop.get("mask") is False:
            g[n + "9"] = {"class_type": "ImageCompositeMasked", "inputs": {
                "destination": last, "source": [n + "6", 0], "x": crop["x"], "y": crop["y"],
                "resize_source": False}}
            last = [n + "9", 0]
            continue
        blend = ["fo", 0]
        if values.get("sam3") and crop.get("head", True):
            for k, src in (("h1", [n + "5", 0]), ("h2", [n + "2", 0])):
                g[n + k] = {"class_type": "SAM3_Detect", "inputs": {
                    "model": ["fh1", 0], "image": src, "conditioning": ["fh2", 0],
                    "threshold": 0.3, "refine_iterations": 2, "individual_masks": False}}
            g[n + "h3"] = {"class_type": "MaskComposite", "inputs": {
                "destination": [n + "h1", 0], "source": [n + "h2", 0], "x": 0, "y": 0,
                "operation": "or"}}
            if boxes and i < len(boxes) and boxes[i]:
                # SAM3's "head" can come back as the hair alone, or holed
                # over the face (2026-09-25): the face's own box always is.
                kx, ky = FACE_EDIT / float(side), FACE_EDIT / float(tall)
                bx, by, bw, bh = boxes[i]
                gx, gy = bw * FACE_BOX_GROW / 2, bh * FACE_BOX_GROW / 2
                x0 = max(0, int((bx - gx - crop["x"]) * kx))
                y0 = max(0, int((by - gy - crop["y"]) * ky))
                x1 = min(FACE_EDIT, int((bx + bw + gx - crop["x"]) * kx))
                y1 = min(FACE_EDIT, int((by + bh - crop["y"]) * ky))   # not below the chin
                if x1 > x0 and y1 > y0:
                    g[n + "b0"] = {"class_type": "SolidMask", "inputs": {
                        "value": 0.0, "width": FACE_EDIT, "height": FACE_EDIT}}
                    g[n + "b1"] = {"class_type": "SolidMask", "inputs": {
                        "value": 1.0, "width": x1 - x0, "height": y1 - y0}}
                    g[n + "b2"] = {"class_type": "MaskComposite", "inputs": {
                        "destination": [n + "b0", 0], "source": [n + "b1", 0], "x": x0,
                        "y": y0, "operation": "or"}}
                    g[n + "b3"] = {"class_type": "MaskComposite", "inputs": {
                        "destination": [n + "h3", 0], "source": [n + "b2", 0], "x": 0,
                        "y": 0, "operation": "or"}}
            g[n + "h4"] = {"class_type": "GrowMask", "inputs": {
                "mask": [n + ("b3" if n + "b3" in g else "h3"), 0], "expand": FACE_HEAD_GROW,
                "tapered_corners": True}}
            g[n + "h5"] = {"class_type": "MaskComposite", "inputs": {
                "destination": [n + "h4", 0], "source": ["fh4", 0], "x": 0, "y": 0,
                "operation": "multiply"}}
            g[n + "h6"] = {"class_type": "MaskToImage", "inputs": {"mask": [n + "h5", 0]}}
            g[n + "h7"] = {"class_type": "ImageBlur", "inputs": {
                "image": [n + "h6", 0], "blur_radius": FACE_HEAD_SOFT[0],
                "sigma": FACE_HEAD_SOFT[1]}}
            blend = [n + "h7", 0]
        g[n + "7"] = {"class_type": "ImageScale", "inputs": {
            "image": blend, "upscale_method": "bilinear", "width": side,
            "height": tall, "crop": "disabled"}}
        g[n + "8"] = {"class_type": "ImageToMask", "inputs": {"image": [n + "7", 0],
                                                              "channel": "red"}}
        g[n + "9"] = {"class_type": "ImageCompositeMasked", "inputs": {
            "destination": last, "source": [n + "6", 0], "x": crop["x"], "y": crop["y"],
            "resize_source": False, "mask": [n + "8", 0]}}
        last = [n + "9", 0]
    g["fs"] = {"class_type": "SaveImage", "inputs": {"images": last, "filename_prefix": prefix}}
    return g


# ================================================================ dressing
# Try On: a person dressed from pictures - the clothes, the hair, the
# accessories. Each is a Qwen-Image-Edit 2509 pass, one after another in one
# run (`dress_graph`), on the loaders and model chains of qwen_dress.json.
# The clothes pass is kingroka's Clothes Try On LoRA as it was trained: one
# picture with the clothes on the left and the person on the right, and its
# own sentence as the prompt; the person's half is cut back out after. The
# LoRA does not do shoes, hats or accessories reliably (its author says so),
# so hair and accessories are Qwen's own multi-picture edit instead: the
# person as picture 1, what to put on them as pictures 2 and 3 (the words
# TextEncodeQwenImageEditPlus labels them with; "image 1" changed nothing).
DRESS_WORKFLOW = "qwen_dress"
TRYON_PROMPT = "put the clothes on the left onto the person on the right."
DRESS_KEEP = ("Keep the person's face, expression, pose, body and the background exactly "
              "as they are in picture 1")
CLOTHES_SLOTS = ("top", "bottom", "outerwear", "footwear")
HAIR_ITEM = "hair"                    # the item_refs key of a character's hair picture
ACCESSORIES_PER_PASS = 2              # pictures 2 and 3; picture 1 is the person
HEAD_SHARE = 0.6                      # a head crop bigger than this share is the whole picture
PERSON_GROW = 6                       # px the person's mask grows before its edge is softened
PERSON_SOFT = (15, 5.0)               # ImageBlur radius and sigma of that edge
DRESS_NODES = {"UNETLoader", "CLIPLoader", "VAELoader", "LoraLoaderModelOnly",
               "ModelSamplingAuraFlow", "CFGNorm", "LoadImage", "ImageScale",
               "ResizeAndPadImage", "ImageStitch", "ImageCrop", "ImageCropV2",
               "ImageToMask", "ImageCompositeMasked", "PreviewImage",
               # the head crop's finder and person mask (only with a SAM3 checkpoint)
               "CheckpointLoaderSimple", "CLIPTextEncode", "SAM3_Detect", "PreviewAny",
               "GetImageSize", "MaskComposite", "GrowMask", "MaskToImage", "ImageBlur",
               "TextEncodeQwenImageEditPlus", "ConditioningZeroOut", "VAEEncode", "KSampler",
               "VAEDecode", "SaveImage"}


def picture_size(data):
    """(width, height) of PNG, JPEG, WebP, GIF or BMP bytes, from the header
    alone; None for anything else. The dress pass lays the person out at
    their own shape, and stdlib has no image library to ask."""
    import struct
    if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24:
        return struct.unpack(">II", data[16:24])
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return struct.unpack("<HH", data[6:10])
    if data[:2] == b"BM" and len(data) >= 26:
        w, h = struct.unpack("<ii", data[18:26])
        return w, abs(h)
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        kind = data[12:16]
        if kind == b"VP8 " and len(data) >= 30:
            w, h = struct.unpack("<HH", data[26:30])
            return w & 0x3FFF, h & 0x3FFF
        if kind == b"VP8L" and len(data) >= 25:
            b = data[21:25]
            return (1 + (((b[1] & 0x3F) << 8) | b[0]),
                    1 + (((b[3] & 0xF) << 10) | (b[2] << 2) | ((b[1] & 0xC0) >> 6)))
        if kind == b"VP8X" and len(data) >= 30:
            return (1 + int.from_bytes(data[24:27], "little"),
                    1 + int.from_bytes(data[27:30], "little"))
        return None
    if data[:2] == b"\xff\xd8":
        i = 2
        while i + 9 < len(data):
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7 or marker == 0xFF:
                i += 1 if marker == 0xFF else 2
                continue
            length = struct.unpack(">H", data[i + 2:i + 4])[0]
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                h, w = struct.unpack(">HH", data[i + 5:i + 9])
                return w, h
            i += 2 + length
    return None


def file_size_of(path):
    """picture_size of a file on this PC, reading at most its first 256 KB
    (a JPEG's frame header can sit behind a large EXIF block)."""
    try:
        with open(path, "rb") as f:
            return picture_size(f.read(262144))
    except OSError:
        return None


def _item(d):
    if isinstance(d, dict) and _str(d.get("path")):
        return {"name": _str(d.get("name")) or os.path.splitext(
            os.path.basename(d["path"]))[0].replace("_", " "), "path": _str(d["path"])}
    return None


def clean_outfit(d):
    """What to dress a person in: {"clothes": [{"name", "path"}], "hair":
    {"path", "words"} | None, "accessories": [{"name", "path"}]}, junk
    dropped. Hair may be words alone; clothes and accessories are pictures."""
    d = d if isinstance(d, dict) else {}
    hair = d.get("hair") if isinstance(d.get("hair"), dict) else {}
    hair = {"path": _str(hair.get("path")), "words": _str(hair.get("words"))}
    return {"clothes": [x for x in map(_item, d.get("clothes") or []) if x],
            "hair": hair if hair["path"] or hair["words"] else None,
            "accessories": [x for x in map(_item, d.get("accessories") or []) if x]}


def outfit_of(settings):
    """The outfit a Generate job is dressed in: the character's pictures of
    what the form has them wearing today - garments in the Clothes slots,
    the rest as accessories - and the hair picture, when it has one."""
    refs = {k.lower(): (k, x) for k, x in clean_item_refs(settings.get("item_refs")).items()}
    out = {"clothes": [], "hair": None, "accessories": []}
    for k in ITEM_SLOTS:
        items = split_many(_field(settings, k)) if SLOTS[k][4] else [_field(settings, k)]
        for name in items:
            if name and name.lower() in refs:
                out["clothes" if k in CLOTHES_SLOTS else "accessories"].append(
                    {"name": refs[name.lower()][0], "path": refs[name.lower()][1]})
    if HAIR_ITEM in refs:
        out["hair"] = {"path": refs[HAIR_ITEM][1], "words": ""}
    return out


def outfit_pictures(outfit):
    """Every picture an outfit uses, in pass order."""
    return ([c["path"] for c in outfit["clothes"]]
            + ([outfit["hair"]["path"]] if outfit["hair"] and outfit["hair"]["path"] else [])
            + [a["path"] for a in outfit["accessories"]])


def outfit_text(outfit):
    """The outfit in words, for a job row and the record."""
    bits = [c["name"] for c in outfit["clothes"]]
    if outfit["hair"]:
        bits.append(outfit["hair"]["words"] or "the hair in the picture")
    bits += [a["name"] for a in outfit["accessories"]]
    return _and(bits)


# Accessories worn on the head, neck or face: drawn in the head crop. Any
# other (a watch, a belt, a bag) is drawn on the whole picture.
HEAD_WORDS = re.compile(
    r"\b(glasses|sunglasses|spectacles|goggles|earrings?|necklace|pendant|chain|choker|"
    r"caps?|hats?|beanie|fedora|beret|helmet|hood|headscarf|scarf|bandana|headband|"
    r"tiara|crown|ties?|bow tie|headphones|earbuds|mask|collar|piercing)\b", re.I)


def on_head(name):
    return bool(HEAD_WORDS.search(name or ""))


def _accessory_passes(items, where):
    out = []
    for i in range(0, len(items), ACCESSORIES_PER_PASS):
        group = items[i:i + ACCESSORIES_PER_PASS]
        out.append({"kind": "accessories", "where": where, "items": group,
                    "prompt": "The person in picture 1 wears %s." % _and(
                        ["the %s from picture %d" % (a["name"], n + 2)
                         for n, a in enumerate(group)])})
    return out


def dress_passes(outfit):
    """The edits, in order: every garment at once (the try-on LoRA), the
    accessories worn below the head, then the hair and the head's
    accessories, two to a pass. Each is marked `where` it is drawn: "body"
    (the whole picture) or "head" (the head crop, when there is one).

    The wording was found live (2026-09-25): "Give the person in picture 1
    the hairstyle shown in picture 2. Keep the face, pose, ... exactly as
    they are" changed nothing, seed after seed - any "keep it the same" or
    "same framing" clause and the edit was not made. "The person in picture
    1 now has the hair of the person in picture 2" and "...wears the glasses
    from picture 2" were made."""
    passes = []
    if outfit["clothes"]:
        passes.append({"kind": "clothes", "where": "body", "items": list(outfit["clothes"]),
                       "prompt": TRYON_PROMPT})
    acc = outfit["accessories"]
    passes += _accessory_passes([a for a in acc if not on_head(a["name"])], "body")
    hair = outfit["hair"]
    if hair and hair["path"]:
        passes.append({"kind": "hair", "where": "head",
                       "items": [{"name": "hair", "path": hair["path"]}],
                       "prompt": "The person in picture 1 now has the hair of the person in "
                                 "picture 2%s." % (": " + hair["words"] if hair["words"]
                                                   else "")})
    elif hair and hair["words"]:
        passes.append({"kind": "hair", "where": "head", "items": [],
                       "prompt": "Change the person's hair to %s." % hair["words"]})
    passes += _accessory_passes([a for a in acc if on_head(a["name"])], "head")
    return passes


def panel_size(width, height, megapixels):
    """The person's shape at `megapixels`, each side a multiple of 16."""
    scale = (megapixels * 1e6 / float(width * height)) ** 0.5
    return (max(256, int(round(width * scale / 16.0)) * 16),
            max(256, int(round(height * scale / 16.0)) * 16))


def head_region(box, width, height, share=None):
    """The head-and-shoulders box around a face (x, y, w, h) in a picture of
    width x height: hair and a hat above it, earrings beside it, a necklace
    or a tie below it. None when that is most of the picture already (a
    portrait): then the whole picture is the head's."""
    x, y, w, h = box
    side = max(w, h)
    x0, x1 = x + w / 2.0 - 1.7 * side, x + w / 2.0 + 1.7 * side
    y0, y1 = y - 1.1 * side, y + h + 2.6 * side
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(width, int(x1)), min(height, int(y1))
    cw, ch = (x1 - x0) // 16 * 16, (y1 - y0) // 16 * 16
    if cw < 64 or ch < 64 or cw * ch > (share or HEAD_SHARE) * width * height:
        return None
    return {"x": x0, "y": y0, "width": cw, "height": ch}


def soft_rect_png(size=256, feather=0.14):
    """A white rectangle on black fading to black over `feather` of each
    side, as a greyscale PNG: the head crop is blended back through it, so
    its edge never shows as a seam."""
    rows = []
    for yy in range(size):
        row = bytearray([0])
        for xx in range(size):
            e = min(xx + 0.5, size - xx - 0.5, yy + 0.5, size - yy - 0.5) / (feather * size)
            t = min(max(e, 0.0), 1.0)
            row.append(int(round(t * t * (3 - 2 * t) * 255)))
        rows.append(bytes(row))
    import struct
    import zlib

    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"".join(rows), 9)) + chunk(b"IEND", b""))


def _dress_start(wf, values, passes):
    """The loaders and model chains, filled; the try-on chain only when a
    pass puts clothes on. -> (graph, values with the defaults)."""
    clothes = any(p["kind"] == "clothes" for p in passes)
    g = fill(wf, dict(values, clothes=clothes))
    return g, dict(wf.get("defaults") or {}, **{k: x for k, x in values.items()
                                                  if x is not None})


def _dress_pass(g, d, ps, last, size, v, pictures, seed):
    """One pass into `g` under ids `d`*, on `last` (pixels of `size`).
    Sampled at the size it is drawn at, ~1 MP (what TextEncodeQwenImageEdit-
    Plus scales its reference to), each side a multiple of 16, so nothing is
    resampled between the canvas and the latent and the person's half is cut
    back out exactly; resampled, the cut landed a few pixels off and left a
    white edge (2026-09-25). -> the link to the result, at `size`."""
    clip, vae = ["2", 0], ["3", 0]
    mp = float(v["panel_megapixels"])
    loads = []
    for i, item in enumerate(ps["items"]):
        g[d + "i%d" % i] = {"class_type": "LoadImage", "inputs": {"image": pictures[item["path"]]}}
        loads.append([d + "i%d" % i, 0])
    if ps["kind"] == "clothes":
        # The garments in a column as tall as the person, each fitted into
        # its share on white, and the column left of the person: the
        # picture the try-on LoRA was trained on, at 1 MP in all.
        cw, ch = panel_size(size[0], size[1], float(v["tryon_megapixels"]) / 2)
        g[d + "p"] = {"class_type": "ImageScale", "inputs": {
            "image": last, "upscale_method": "lanczos", "width": cw, "height": ch,
            "crop": "disabled"}}
        k, column = len(loads), None
        for i, link in enumerate(loads):
            share = ch // k if i < k - 1 else ch - (ch // k) * (k - 1)
            g[d + "f%d" % i] = {"class_type": "ResizeAndPadImage", "inputs": {
                "image": link, "target_width": cw, "target_height": share,
                "padding_color": "white", "interpolation": "lanczos"}}
            if column is None:
                column = [d + "f%d" % i, 0]
            else:
                g[d + "s%d" % i] = {"class_type": "ImageStitch", "inputs": {
                    "image1": column, "image2": [d + "f%d" % i, 0], "direction": "down",
                    "match_image_size": False, "spacing_width": 0, "spacing_color": "white"}}
                column = [d + "s%d" % i, 0]
        g[d + "in"] = {"class_type": "ImageStitch", "inputs": {
            "image1": column, "image2": [d + "p", 0], "direction": "right",
            "match_image_size": False, "spacing_width": 0, "spacing_color": "white"}}
        model, images = ["9", 0], {"image1": [d + "in", 0]}
    else:
        pw, ph = panel_size(size[0], size[1], mp)
        g[d + "in"] = {"class_type": "ImageScale", "inputs": {
            "image": last, "upscale_method": "lanczos", "width": pw, "height": ph,
            "crop": "disabled"}}
        model = ["6", 0]
        images = {"image%d" % (i + 1): link for i, link in enumerate([[d + "in", 0]] + loads)}
    g[d + "pos"] = {"class_type": "TextEncodeQwenImageEditPlus", "inputs": dict(
        images, clip=clip, vae=vae, prompt=ps["prompt"])}
    g[d + "neg"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": [d + "pos", 0]}}
    g[d + "enc"] = {"class_type": "VAEEncode", "inputs": {"pixels": [d + "in", 0], "vae": vae}}
    g[d + "ks"] = {"class_type": "KSampler", "inputs": {
        "seed": seed % (MAX_SEED + 1), "steps": v["steps"], "cfg": 1.0,
        "sampler_name": v["sampler"], "scheduler": v["scheduler"], "denoise": 1.0,
        "model": model, "positive": [d + "pos", 0], "negative": [d + "neg", 0],
        "latent_image": [d + "enc", 0]}}
    g[d + "dec"] = {"class_type": "VAEDecode", "inputs": {"samples": [d + "ks", 0], "vae": vae}}
    out = [d + "dec", 0]
    if ps["kind"] == "clothes":
        g[d + "cut"] = {"class_type": "ImageCrop", "inputs": {
            "image": out, "width": cw, "height": ch, "x": cw, "y": 0}}
        out = [d + "cut", 0]
    g[d + "out"] = {"class_type": "ImageScale", "inputs": {
        "image": out, "upscale_method": "lanczos", "width": int(size[0]),
        "height": int(size[1]), "crop": "disabled"}}
    return [d + "out", 0]


def _dress_save(g, last, size, prefix):
    g["dz"] = {"class_type": "ImageScale", "inputs": {
        "image": last, "upscale_method": "lanczos", "width": int(size[0]),
        "height": int(size[1]), "crop": "disabled"}}
    g["ds"] = {"class_type": "SaveImage", "inputs": {"images": ["dz", 0],
                                                     "filename_prefix": prefix}}
    return g


def dress_graph(wf, values, passes, person, size, pictures, prefix, find_head=None):
    """The first run: `person` (a LoadImage name) of `size` (w, h), scaled
    to the working size (`panel_megapixels`), with each of `passes` drawn in
    order on the whole picture. With `find_head` (a SAM3 checkpoint) it
    ends in a preview of the working picture plus SAM3's face boxes, for
    dress_head_graph to crop from; else in the result scaled back to `size`
    and saved under `prefix`. `pictures` maps each item's path to its
    LoadImage name; pass n is seeded seed+n."""
    g, v = _dress_start(wf, values, passes)
    work = panel_size(size[0], size[1], float(v["panel_megapixels"]))
    g["p0"] = {"class_type": "LoadImage", "inputs": {"image": person}}
    g["p1"] = {"class_type": "ImageScale", "inputs": {
        "image": ["p0", 0], "upscale_method": "lanczos", "width": work[0], "height": work[1],
        "crop": "center"}}
    last = ["p1", 0]
    for n, ps in enumerate(passes):
        last = _dress_pass(g, "d%d_" % (n + 1), ps, last, work, v, pictures, int(v["seed"]) + n)
    if not find_head:
        return _dress_save(g, last, size, prefix)
    g["dp"] = {"class_type": "PreviewImage", "inputs": {"images": last}}
    return add_face_finder(g, find_head, pixels=last)


def dress_head_graph(wf, values, passes, image, work, crop, mask, size, pictures, prefix,
                     first=0, sam3=None):
    """The second run: `image` (the first run's preview, `work` sized) with
    `passes` drawn on `crop` of it - enlarged to the working size, dressed,
    shrunk back and blended in through `mask` (a LoadImage name of a soft
    rectangle) - or on the whole of it when `crop` is None; then scaled to
    `size` and saved. Pass n is seeded seed+first+n, after the first run's.

    With `sam3`, only the person is blended back: SAM3's "person" in the
    crop before and after the edit, grown and softened, times the soft
    rectangle. The edit redraws the crop's background a shade off (lighter,
    on the first live run), and through the rectangle alone that showed as
    a pale box behind the head; the background now keeps its own pixels."""
    g, v = _dress_start(wf, values, passes)
    g["hi"] = {"class_type": "LoadImage", "inputs": {"image": image}}
    if crop is None:
        last = ["hi", 0]
        for n, ps in enumerate(passes):
            last = _dress_pass(g, "h%d_" % (n + 1), ps, last, work, v, pictures,
                               int(v["seed"]) + first + n)
        return _dress_save(g, last, size, prefix)
    side = (crop["width"], crop["height"])
    g["hc"] = {"class_type": "ImageCropV2", "inputs": {"image": ["hi", 0], "crop_region": crop}}
    last = ["hc", 0]
    for n, ps in enumerate(passes):
        last = _dress_pass(g, "h%d_" % (n + 1), ps, last, side, v, pictures,
                           int(v["seed"]) + first + n)
    g["hm"] = {"class_type": "LoadImage", "inputs": {"image": mask}}
    g["hs"] = {"class_type": "ImageScale", "inputs": {
        "image": ["hm", 0], "upscale_method": "bilinear", "width": side[0], "height": side[1],
        "crop": "disabled"}}
    g["hk"] = {"class_type": "ImageToMask", "inputs": {"image": ["hs", 0], "channel": "red"}}
    blend = ["hk", 0]
    if sam3:
        g["hl"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": sam3}}
        g["ht"] = {"class_type": "CLIPTextEncode", "inputs": {"text": "person:4",
                                                              "clip": ["hl", 1]}}
        for nid, src in (("hp1", ["hc", 0]), ("hp2", last)):
            g[nid] = {"class_type": "SAM3_Detect", "inputs": {
                "model": ["hl", 0], "image": src, "conditioning": ["ht", 0],
                "threshold": 0.3, "refine_iterations": 2, "individual_masks": False}}
        g["hp3"] = {"class_type": "MaskComposite", "inputs": {
            "destination": ["hp1", 0], "source": ["hp2", 0], "x": 0, "y": 0, "operation": "or"}}
        g["hp4"] = {"class_type": "GrowMask", "inputs": {
            "mask": ["hp3", 0], "expand": PERSON_GROW, "tapered_corners": True}}
        g["hp5"] = {"class_type": "MaskToImage", "inputs": {"mask": ["hp4", 0]}}
        g["hp6"] = {"class_type": "ImageBlur", "inputs": {
            "image": ["hp5", 0], "blur_radius": PERSON_SOFT[0], "sigma": PERSON_SOFT[1]}}
        g["hp7"] = {"class_type": "ImageToMask", "inputs": {"image": ["hp6", 0],
                                                             "channel": "red"}}
        g["hp8"] = {"class_type": "MaskComposite", "inputs": {
            "destination": ["hp7", 0], "source": ["hk", 0], "x": 0, "y": 0,
            "operation": "multiply"}}
        blend = ["hp8", 0]
    g["hb"] = {"class_type": "ImageCompositeMasked", "inputs": {
        "destination": ["hi", 0], "source": last, "x": crop["x"], "y": crop["y"],
        "resize_source": False, "mask": blend}}
    return _dress_save(g, ["hb", 0], size, prefix)


def dress_values(wf, backend, steps=None):
    """The dress workflow's values on `backend` (its encoder device)."""
    v = dict(wf.get("defaults") or {})
    v["encoder_device"] = "cpu" if backend.get("encoder_on_cpu") else "default"
    if steps:
        v["steps"] = int(steps)
    return v


def dress_lacks(wf, outfit, inventory, nodes):
    """What a backend lacks to dress in `outfit`: lacks() without the try-on
    LoRA when there are no clothes to put on, and DRESS_NODES."""
    skip = set() if outfit["clothes"] else {"tryon"}
    out = lacks(wf, wf.get("defaults") or {}, inventory, None, skip=skip)
    if nodes is not None:
        out += [{"kind": "node", "name": n, "folder": "", "var": "",
                 "text": "the node %s (a newer ComfyUI has it)" % n}
                for n in sorted(DRESS_NODES - set(nodes))]
    return out


FOLDER_WORDS = {"diffusion_models": "diffusion model", "checkpoints": "checkpoint",
                "text_encoders": "text encoder", "vae": "VAE", "loras": "LoRA",
                "clip_vision": "CLIP vision model", "style_models": "style model",
                "controlnet": "ControlNet", "upscale_models": "upscale model"}


def _names(when):
    """A node's `_when`: one name or a list of them."""
    return when if isinstance(when, list) else [when]


def uses(wf, var):
    """Whether a template does anything with `var` (a node kept or dropped by
    it, or a switch on it) - so a setting it ignores is not recorded as done."""
    return (any(var in _names(n.get("_when")) or n.get("_unless") == var
                for n in wf["graph"].values())
            or any(sw.get("when") == var for sw in (wf.get("switches") or {}).values()))


def lacks(wf, values, inventory, nodes, skip=()):
    """What a backend is missing to run `wf` with `values`: [{"kind": "file" |
    "node", "name", "folder", "var", "text"}]. Files are checked when the
    backend's `inventory` is known and nodes when its `nodes` are; unknown
    is not reported as missing."""
    out = []
    if inventory is not None:
        for var, folder in (wf.get("files") or {}).items():
            if var in skip or var not in values:
                continue
            if values[var] not in inventory.get(folder, set()):
                out.append({"kind": "file", "name": values[var], "folder": folder, "var": var,
                            "text": "%s (the %s, in ComfyUI/models/%s)"
                                    % (values[var], FOLDER_WORDS.get(folder, folder), folder)})
    if nodes is not None:
        need = {n["class_type"] for n in wf["graph"].values()
                if "_when" not in n and "_unless" not in n}
        for name in sorted(need - set(nodes)):
            out.append({"kind": "node", "name": name, "folder": "", "var": "",
                        "text": "the node %s (a custom node pack ComfyUI lacks)" % name})
    return out


def missing_for(model, backend, inventory, nodes=None, workflow_loader=None):
    """Everything `backend` lacks to run `model` - the Models window, the
    model menu and routing all ask this. -> (problems, lacking) where
    `problems` are reasons it cannot run at all there (marked absent, a
    broken template) and `lacking` is `lacks()`'s list. Both empty: ready,
    as far as is known."""
    resolved = resolve_model(model, backend["id"])
    if resolved is None:
        return (["%s is marked as not on %s (Models)." % (model["label"], backend["name"])],
                [])
    values, _, wid = resolved
    try:
        wf = (workflow_loader or load_workflow)(wid)
    except TemplateError as e:
        return [str(e)], []
    return [], lacks(wf, values, inventory, nodes)


# ============================================================ composing a job

def default_settings():
    return {"preset": "standard", "model": "z-image-turbo", "backend": "auto",
            "identities": [], "style": "none", "style_strength": None,
            "scene": "", "camera": "", "negative": "", "loras": [], "references": {},
            "character": "", "item_refs": {}, "anatomy": True,
            "seed": -1, "seed_mode": "random", "steps": None, "guidance": None,
            "sampler": "", "scheduler": "", "width": None, "height": None,
            "denoise": None, "refine": None, "upscale": None, "refine_denoise": None,
            "face_detail": None, "batch": 1, "pose": None, "composition": None,
            "auto_refine": False, "refine_passes": 3,
            **{k: "" for k in SLOTS}, **{k: 0 for k, _, _ in SLIDERS}}


# The person, laid out the way a video game's character creator lays one out:
# sections of slots, each offering picks, every slot also taking free text.
# A slot is (setting, label, nouns, picks, many). A value that names none of
# `nouns` gets the first one: "auburn" -> "auburn hair", "neutral" ->
# "neutral expression". A `many` slot holds several picks, comma-separated
# ("glasses, silver necklace"), and a pick toggles in and out of it.
LOOKS = [
    ("Body", [
        ("subject", "Who", (), ["a woman", "a man", "a person", "a young woman",
                                "a young man", "an older woman", "an older man"], False),
        ("age", "Age", (), ["in their 20s", "in their 30s", "in their 40s", "in their 50s",
                            "in their 60s", "in their 70s"], False),
        ("skin", "Skin", ("skin", "complexion"),
         ["pale", "fair", "light", "olive", "tan", "light brown", "brown", "dark brown",
          "deep brown"], False),
        ("build", "Body type", ("build", "weight", "figure", "body", "physique", "lb", "kg",
                                "pound", "kilo"),
         ["athletic", "lean", "average", "curvy", "broad-shouldered", "stocky", "lanky",
          "petite"], False),
    ]),
    ("Face", [
        ("face", "Face shape", ("face", "jaw", "cheek", "chin"),
         ["oval", "round", "square", "heart-shaped", "long", "diamond-shaped",
          "sharp jawline", "high cheekbones"], False),
        ("eyes", "Eyes", ("eyes", "eye"),
         ["brown", "dark brown", "hazel", "amber", "green", "blue", "grey",
          "almond-shaped", "hooded"], False),
        ("brows", "Eyebrows", ("eyebrows", "brow"),
         ["thick", "thin", "arched", "straight", "bushy", "defined"], False),
        ("nose", "Nose", ("nose",), ["small", "straight", "button", "aquiline", "broad",
                                     "Roman"], False),
        ("lips", "Lips", ("lips", "lip", "mouth"), ["full", "thin", "wide", "bow-shaped"],
         False),
        ("facial_hair", "Facial hair", (),
         ["clean-shaven", "light stubble", "heavy stubble", "short beard", "full beard",
          "moustache", "goatee"], False),
        ("traits", "Marks / other", (),
         ["freckles", "dimples", "beauty mark", "rosy cheeks", "scar on the cheek",
          "tattoos", "nose piercing"], True),
    ]),
    ("Hair", [
        ("hair", "Colour", ("hair",),
         ["black", "dark brown", "brown", "light brown", "auburn", "red", "ginger",
          "strawberry blonde", "blonde", "platinum blonde", "grey", "silver", "white",
          "dyed pink", "dyed blue"], False),
        ("hair_style", "Style", ("hair",),
         ["very short", "short", "chin-length", "shoulder-length", "long", "very long",
          "straight", "wavy", "curly", "coily", "in a bun", "in a ponytail", "in braids",
          "slicked back", "pixie cut", "bob", "buzz cut", "afro", "locs", "bald"], False),
    ]),
    ("Expression", [
        ("expression", "Expression", ("expression", "smil", "laugh", "grin", "smirk",
                                      "frown", "pout", "scowl", "tear", "cry"),
         ["neutral", "soft smile", "broad smile", "laughing", "winking", "shy", "calm",
          "dreamy", "playful", "confident smirk", "serious", "thoughtful", "pensive",
          "surprised", "worried", "scared", "sad", "tearful", "angry", "determined",
          "disgusted", "tired"], False),
        ("gaze", "Looking", (),
         ["looking at the camera", "looking away", "looking over the shoulder",
          "looking up", "looking down", "eyes closed"], False),
    ]),
    ("Clothes", [
        ("top", "Top / dress", (),
         ["white t-shirt", "black t-shirt", "button-down shirt", "linen shirt",
          "knit sweater", "turtleneck", "hoodie", "blouse", "tank top", "polo shirt",
          "summer dress", "evening gown", "business suit", "tuxedo"], False),
        ("bottom", "Bottoms", (),
         ["blue jeans", "black jeans", "chinos", "tailored trousers", "shorts",
          "pleated skirt", "denim skirt", "leggings", "cargo pants"], False),
        ("outerwear", "Outerwear", (),
         ["denim jacket", "leather jacket", "trench coat", "wool overcoat", "blazer",
          "puffer jacket", "cardigan", "raincoat"], False),
        ("footwear", "Shoes", (),
         ["white sneakers", "leather boots", "ankle boots", "loafers", "heels", "sandals",
          "running shoes", "combat boots"], False),
    ]),
    ("Accessories", [
        ("accessories", "Accessories", (),
         ["glasses", "round glasses", "sunglasses", "earrings", "hoop earrings", "necklace",
          "pendant", "wristwatch", "bracelet", "rings", "baseball cap", "beanie", "fedora",
          "headscarf", "scarf", "tie", "bow tie", "belt", "backpack", "tote bag",
          "headphones", "gloves"], True),
    ]),
]
SLOTS = {s[0]: s for _, slots in LOOKS for s in slots}
# The face each expression pick shows on its chip. Never in the prompt.
EMOJI = {"neutral": "\U0001F610", "soft smile": "\U0001F642", "broad smile": "\U0001F601",
         "laughing": "\U0001F602", "winking": "\U0001F609", "shy": "\U0001F60A",
         "calm": "\U0001F60C", "dreamy": "\U0001F60D", "playful": "\U0001F61C",
         "confident smirk": "\U0001F60F", "serious": "\U0001F611",
         "thoughtful": "\U0001F914", "pensive": "\U0001F614", "surprised": "\U0001F62E",
         "worried": "\U0001F61F", "scared": "\U0001F628", "sad": "\U0001F641",
         "tearful": "\U0001F622", "angry": "\U0001F620", "determined": "\U0001F624",
         "disgusted": "\U0001F922", "tired": "\U0001F629",
         "looking at the camera": "\U0001F440", "looking up": "\U0001F644",
         "eyes closed": "\U0001F60C"}


def pick_label(pick):
    """A pick as its chip shows it: "\U0001F642 soft smile"."""
    return (EMOJI[pick] + " " + pick) if pick in EMOJI else pick

# The creator's sliders: (setting, label, words from lo to hi). The middle
# is 0 and says nothing; the form stores the step as an int.
SLIDERS = [
    ("weight", "Weight", ["very thin", "thin", "slim", "", "slightly heavy", "heavyset",
                          "very heavyset"]),
    ("muscle", "Muscle", ["frail", "soft", "untoned", "", "toned", "muscular",
                          "very muscular"]),
    # "stature", not "height": that is the picture's.
    ("stature", "Height", ["very short", "short", "a little short", "",
                          "a little tall", "tall", "very tall"]),
]
SLIDER_SPAN = 3                       # each slider runs -3..3
SLIDER_KEYS = [k for k, _, _ in SLIDERS]
SLIDER_SECTION = "Body"

# What changes picture to picture rather than person to person: a character
# does not keep these, and choosing one leaves them as they are.
PER_PICTURE = ("expression", "gaze")
CHARACTER_KEYS = [k for k in SLOTS if k not in PER_PICTURE] + [k for k, _, _ in SLIDERS]
# What a clothes preset (the `outfits` library) holds: the Clothes and
# Accessories sections' slots.
OUTFIT_KEYS = [k for name, slots in LOOKS if name in ("Clothes", "Accessories")
               for k, *_ in slots]
# The slots a reference picture can show: each item worn or carried.
ITEM_SLOTS = ("top", "bottom", "outerwear", "footwear", "accessories")


def _field(s, key):
    v = s.get(key)
    return v.strip().strip(",.").strip() if isinstance(v, str) else ""


def split_many(value):
    return [x.strip() for x in (value or "").split(",") if x.strip()]


def toggle(value, pick, many):
    """A pick clicked in a slot holding `value`: a single slot takes it (or
    lets go of it, clicked again); a `many` slot adds or drops it."""
    if not many:
        return "" if value.strip().lower() == pick.lower() else pick
    items = split_many(value)
    low = [x.lower() for x in items]
    if pick.lower() in low:
        del items[low.index(pick.lower())]
    else:
        items.append(pick)
    return ", ".join(items)


def slider_word(key, step):
    words = next(w for k, _, w in SLIDERS if k == key)
    try:
        step = int(round(float(step)))
    except (TypeError, ValueError):
        return ""
    return words[max(-SLIDER_SPAN, min(SLIDER_SPAN, step)) + SLIDER_SPAN]


def _noun(key, v):
    nouns = SLOTS[key][2]
    if v and nouns and not any(n in v.lower() for n in nouns):
        v += " " + nouns[0]
    return v


def _and(items):
    items = [x for x in items if x]
    return (", ".join(items[:-1]) + " and " + items[-1]) if len(items) > 1 else \
        "".join(items)


HAIR_NOUNS = ("bald", "buzz cut", "pixie cut", "bob", "afro", "locs", "crew cut", "mohawk",
              "undercut", "shaved head")
HAIR_AFTER = ("in ", "with ", "pulled ", "slicked ", "tied ")


def hair_text(s):
    """Colour and style as one phrase: "long wavy auburn hair", "auburn hair
    in a bun", "grey buzz cut", "bald"."""
    colour, style = _field(s, "hair"), _field(s, "hair_style")
    low = style.lower()
    if "bald" in low:
        return style
    if "hair" in colour.lower() or "hair" in low:
        return ", ".join(x for x in (style, _noun("hair", colour)) if x)
    if any(n in low for n in HAIR_NOUNS):
        return " ".join(x for x in (colour, style) if x)
    if not (colour or style):
        return ""
    if low.startswith(HAIR_AFTER):
        return " ".join(x for x in (colour, "hair", style) if x)
    return " ".join(x for x in (style, colour, "hair") if x)


def person_text(settings):
    """The person as prompt text, the creator's sections in order: who, body,
    face, hair, expression, clothes, accessories. "a woman, in their 30s, olive
    skin, slim, tall, green eyes, long auburn hair, freckles, soft smile,
    wearing a knit sweater and blue jeans, with glasses and a necklace"."""
    s = settings
    bits = [_field(s, k) for k in ("subject", "age")]
    bits += [_noun(k, _field(s, k)) for k in ("skin", "build")]
    bits += [slider_word(k, s.get(k)) for k, _, _ in SLIDERS]
    bits += [_noun(k, _field(s, k)) for k in ("face", "eyes", "brows", "nose", "lips",
                                              "facial_hair")]
    bits.append(hair_text(s))
    bits += [_field(s, "traits"), _noun("expression", _field(s, "expression")),
             _field(s, "gaze")]
    worn = _and([_field(s, k) for k in ("top", "bottom", "outerwear", "footwear")])
    bits.append("wearing " + worn if worn else "")
    carried = _and(split_many(_field(s, "accessories")))
    bits.append("with " + carried if carried else "")
    return ", ".join(b for b in bits if b)


def items_worn(settings):
    """Every item the person wears or carries, as the form names it."""
    out = []
    for k in ITEM_SLOTS:
        out += split_many(_field(settings, k)) if SLOTS[k][4] else [_field(settings, k)]
    return [x for x in out if x]


def random_looks(rng=random, sections=None):
    """A roll of the dice, the creator's Randomize: one pick per slot (a few
    left empty), every slider somewhere. `sections` limits it to those."""
    out = {}
    optional = {"facial_hair": 0.6, "traits": 0.5, "outerwear": 0.5, "brows": 0.5,
                "nose": 0.6, "lips": 0.5, "face": 0.4}
    for name, slots in LOOKS:
        if sections and name not in sections or name == "Expression":
            continue
        for key, _, _, picks, many in slots:
            if rng.random() < optional.get(key, 0.0):
                out[key] = ""
            elif key == "accessories":
                out[key] = ", ".join(rng.sample(picks, rng.randint(0, 2)))
            else:
                out[key] = rng.choice(picks)
    if not sections or SLIDER_SECTION in sections:
        for key, _, _ in SLIDERS:
            out[key] = rng.randint(-2, 2)
    return out


# The constants: what every person in every picture has, whoever they are
# and whatever the creator says about them. Diffusion models lose count of
# fingers and limbs, so the prompt says it outright. (part, positive,
# negative). Every shipped workflow samples at CFG 1, which ignores the
# negative prompt, so the positive sentence is what does the work there; the
# negative is for a model that reads one.
ANATOMY = [
    ("Hands", "two hands, each with four fingers and a thumb",
     "extra fingers, missing fingers, fused fingers, six fingers, extra hands, "
     "malformed hands"),
    ("Feet", "two feet", "extra feet, extra legs, missing feet"),
    ("Eyes", "two eyes", "extra eyes, third eye, misaligned eyes"),
    ("Body", "a proportionate body with natural anatomy",
     "extra limbs, extra arms, disproportionate body, elongated neck, deformed body"),
]
# A scene with nobody described still gets the constants when it names a person.
PEOPLE = re.compile(r"\b(wom[ae]n|m[ae]n|person|people|girls?|boys?|lady|ladies|guys?|"
                    r"kids?|child(ren)?|he|she|they|his|her|portrait|selfie|couple|"
                    r"crowd|dancers?|athletes?|workers?|someone|figure)\b", re.I)


def anatomy_text():
    parts = [pos for _, pos, _ in ANATOMY]
    return "Anatomically correct: every person has exactly " + _and(parts)


def anatomy_negative():
    return ", ".join(neg for _, _, neg in ANATOMY)


# The Camera diagram (`CameraAim` in the tab's module): where the camera is,
# as three steps - how much of the person is in the frame, which side of
# them it sees, and how high it is. Stored as settings["view"], {"shot",
# "turn", "height"}, or None when it is not set and the model frames the
# picture itself. Every shot but the face says the head is in the frame
# with room above it: FLUX, left to itself, crops the top of the head.
VIEW_SHOTS = [
    ("face", "Face", "Extreme close-up of the face, the whole face in the frame"),
    ("head", "Head and shoulders", "Close-up portrait of the head and shoulders, the "
     "top of the head in the frame"),
    ("waist", "Waist up", "Medium shot from the waist up, the whole head in the frame "
     "with space above it"),
    ("knees", "Knees up", "Medium-long shot from the knees up, the whole head in the "
     "frame with space above it"),
    ("full", "Full body", "Full body shot, the whole person from head to feet in the "
     "frame, space above the head and below the feet"),
    ("wide", "Wide", "Wide shot, the whole person small in the frame with the "
     "surroundings around them, space above the head"),
]
# Degrees the camera has gone round the person from their front, toward
# their left: 90 sees their left side, 180 their back.
VIEW_TURNS = {0: "seen from the front, facing the camera",
              45: "three-quarter view from the subject's left",
              90: "side profile view from the subject's left",
              135: "three-quarter back view from behind the subject's left",
              180: "seen from behind, back to the camera",
              -45: "three-quarter view from the subject's right",
              -90: "side profile view from the subject's right",
              -135: "three-quarter back view from behind the subject's right"}
VIEW_HEIGHTS = [
    ("overhead", "Overhead", "bird's-eye view, the camera overhead looking straight down"),
    ("high", "High", "high angle shot, the camera above eye level looking down"),
    ("eye", "Eye level", "eye-level camera"),
    ("low", "Low", "low angle shot, the camera below eye level looking up"),
    ("ground", "Ground", "worm's-eye view, the camera near the ground looking up"),
]
VIEW_DEFAULT = {"shot": "full", "turn": 0, "height": "eye"}
TURN_LABELS = {0: "front", 45: "their left, three-quarter", 90: "their left side",
               135: "behind their left", 180: "behind",
               -45: "their right, three-quarter", -90: "their right side",
               -135: "behind their right"}


def clean_view(view):
    """Anything -> {"shot", "turn", "height"} or None."""
    if not isinstance(view, dict):
        return None
    shots, heights = dict((k, t) for k, _, t in VIEW_SHOTS), dict(
        (k, t) for k, _, t in VIEW_HEIGHTS)
    try:
        turn = int(round(float(view.get("turn", 0)) / 45.0)) * 45
    except (TypeError, ValueError):
        turn = 0
    turn = (turn + 180) % 360 - 180 or 0
    turn = 180 if turn == -180 else turn
    return {"shot": view.get("shot") if view.get("shot") in shots else "full",
            "turn": turn,
            "height": view.get("height") if view.get("height") in heights else "eye"}


def view_label(view):
    """One line for the form: "Full body, eye level, from the front"."""
    v = clean_view(view)
    if not v:
        return ""
    shot = dict((k, lbl) for k, lbl, _ in VIEW_SHOTS)[v["shot"]]
    height = dict((k, lbl) for k, lbl, _ in VIEW_HEIGHTS)[v["height"]]
    return "%s, %s, from %s" % (shot, height.lower(), TURN_LABELS[v["turn"]])


def view_text(view, posed=False):
    """The camera's words, for the front of the prompt (FLUX weighs what
    comes first). A drawn pose already frames the figure and faces it, and
    words that disagree fight the ControlNet, so then only the height is said."""
    v = clean_view(view)
    if not v:
        return ""
    height = dict((k, t) for k, _, t in VIEW_HEIGHTS)[v["height"]]
    if posed:
        return height[0].upper() + height[1:]
    shot = dict((k, t) for k, _, t in VIEW_SHOTS)[v["shot"]]
    return ", ".join((shot, VIEW_TURNS[v["turn"]], height))


def has_person(settings, named=False):
    """Whether the picture has a person in it: one chosen or described, or
    the scene names one."""
    return bool(named or person_text(settings) or PEOPLE.search(_field(settings, "scene")))


def summary(settings):
    """One line for a job row before its prompt is composed."""
    if settings.get("mode") == "dress":
        return "Try On: " + outfit_text(clean_outfit(settings.get("outfit")))
    return ". ".join(x for x in (_field(settings, "scene"), person_text(settings),
                                 _field(settings, "camera")) if x)


def resolve_model(model, backend_id):
    """A logical model on one backend -> (values, family, workflow id), or
    None when that backend is marked as not having it."""
    over = model["backends"].get(backend_id, {})
    if over is None:
        return None
    values = dict(model["values"])
    values.update({k: x for k, x in over.items() if k not in ("family", "workflow")})
    return values, over.get("family") or model["family"], over.get("workflow") or model["workflow"]


def lora_file(lora, backend_id):
    return lora["files"].get(backend_id) or lora["file"]


class Plan:
    """What `compose` decided for one job on one backend."""

    def __init__(self):
        self.values, self.loras, self.images = {}, [], {}
        self.warnings, self.notes, self.errors = [], [], []
        self.prompt = self.negative = ""
        self.workflow = None
        self.family = ""
        self.lora_meta = []          # [{"name", "file", "strength", "category"}]
        self.references = {}         # kind -> local path actually used
        self.items = []              # [(item name, local path)]: Kontext's reference
        self.item_text = ""          # the prompt's line about them


ITEM_PROMPT = ("The %s %s exactly as in the reference picture, worn by the person. "
               "One person, not the reference picture itself.")


def plan_items(p, s, wf, v, backend, short, nodes):
    """Into `p` and `v`: the character's pictures of what it wears today
    (outfit_of: clothes, the hair picture, accessories) as references for the
    picture itself. A workflow with an `items` section makes it with FLUX
    Kontext, the pictures side by side as its reference (add_item_refs);
    otherwise, or on a backend without Kontext (`short` names the file), the
    words alone describe them, said once."""
    outfit = outfit_of(s)
    items = outfit["clothes"] + ([{"name": "hair", "path": outfit["hair"]["path"]}]
                                 if outfit["hair"] else []) + outfit["accessories"]
    if not items:
        return
    names = _and([i["name"] for i in items])
    gone = [i for i in items if not os.path.isfile(i["path"])]
    for i in gone:
        p.warnings.append("The picture of the %s (%s) is not on this PC any more; not "
                          "used." % (i["name"], i["path"]))
    items = [i for i in items if i not in gone]
    section = wf.get("items")
    why = ""
    if not section:
        why = "the %s workflow takes no item pictures" % wf.get("label", wf.get("id"))
    elif short:
        why = "%s lacks %s (FLUX.1 Kontext [dev], which draws from pictures)" % (
            backend["name"], short)
    elif nodes is not None and ITEM_NODES - set(nodes):
        why = "%s's ComfyUI lacks %s" % (backend["name"],
                                         ", ".join(sorted(ITEM_NODES - set(nodes))))
    if items and why:
        p.warnings.append("Pictures of the %s: %s, so the words alone describe %s."
                          % (names, why, "it" if len(items) == 1 else "them"))
        return
    if not items:
        return
    p.items = [(i["name"], i["path"]) for i in items]
    for name, path in p.items:
        p.references["item: " + name] = path
    v["model"] = v[section["model"]]
    v["weight_dtype"] = "default"
    if s.get("guidance") in (None, ""):
        v["guidance"] = (wf.get("defaults") or {}).get(section["guidance"], v.get("guidance"))
    p.item_text = ITEM_PROMPT % (_and([i["name"] for i in items]),
                                 "looks" if len(items) == 1 else "look")
    p.prompt = (p.prompt + " " + p.item_text).strip()
    v["prompt"] = p.prompt
    p.notes.append("Made with FLUX.1 Kontext [dev] from the pictures of the %s." % _and(
        [i["name"] for i in items]))


def compose(settings, lib, backend, inventory=None, workflow_loader=load_workflow,
            nodes=None):
    """The form's settings -> a Plan for `backend`. `inventory` is that
    backend's {kind: filenames} and `nodes` its node classes, when known; a
    file or node missing from them is an error for what the job cannot run
    without and a warning for what it can. Never raises for a user mistake:
    those land in `plan.errors`."""
    s = dict(default_settings())
    s.update(settings or {})
    p = Plan()
    bid = backend["id"]
    preset = PRESETS.get(s["preset"], PRESETS["standard"])
    model = lib.get("models", s["model"])
    if model is None:
        p.errors.append("No model called %r in the library." % s["model"])
        return p
    resolved = resolve_model(model, bid)
    if resolved is None:
        p.errors.append("%s is marked as not installed on %s." % (model["label"], backend["name"]))
        return p
    mvalues, family, wid = resolved
    p.family = family
    try:
        wf = workflow_loader(wid)
    except TemplateError as e:
        p.errors.append(str(e))
        return p
    p.workflow = wf

    style = lib.get("styles", s["style"]) if s["style"] else None
    idents = []
    for sel in s["identities"]:
        ident = lib.get("identities", sel.get("id") if isinstance(sel, dict) else sel)
        if ident is None:
            p.warnings.append("Identity %r is no longer in the library; left out." % sel)
            continue
        strength = sel.get("strength") if isinstance(sel, dict) else None
        idents.append((ident, ident["strength"] if strength is None else strength))
    if s["preset"] == "identity" and not idents:
        p.errors.append("Identity Portrait needs a person: choose one under Person.")

    # ------------------------------------------------------------ LoRAs
    stack = []                        # (lora record, strength, why)
    for ident, strength in idents:
        if ident["lora"]:
            rec = lib.get("loras", ident["lora"])
            if rec is None:
                p.warnings.append("%s's LoRA %r is not in the LoRA library."
                                  % (ident["name"], ident["lora"]))
            else:
                stack.append((rec, strength, "identity " + ident["name"]))
    if style and style["lora"]:
        rec = lib.get("loras", style["lora"])
        st = s["style_strength"] if s["style_strength"] is not None else style["strength"]
        if rec is None:
            p.warnings.append("Style %s's LoRA %r is not in the LoRA library."
                              % (style["name"], style["lora"]))
        else:
            stack.append((rec, st, "style " + style["name"]))
    for sel in s["loras"]:
        rec = lib.get("loras", sel.get("id")) if isinstance(sel, dict) else None
        if rec is None:
            continue
        stack.append((rec, sel.get("strength", rec["strength"]), "added"))
    # "Always on" LoRAs join every picture from a model they suit, at their
    # library strength, unless already in the stack. One for another family
    # is skipped without a warning: that is what "suits" means here.
    have_ids = {rec["id"] for rec, _, _ in stack}
    for rec in lib.all("loras"):
        if (rec.get("always") and rec["id"] not in have_ids
                and compatibility(rec["family"], family) is not False):
            stack.append((rec, rec["strength"], "always on"))

    if stack and not wf.get("lora_chain"):
        names = []
        for rec, strength, _ in stack:
            if strength and rec["name"] not in names:
                names.append(rec["name"])
        if names:
            p.warnings.append("The %s workflow takes no LoRAs: %s left out."
                              % (wf.get("label", wid), ", ".join(names)))
        stack = []
    seen = set()
    have = (inventory or {}).get("loras")
    for rec, strength, why in stack:
        if rec["id"] in seen:
            p.warnings.append("%s is enabled twice; used once." % rec["name"])
            continue
        seen.add(rec["id"])
        if not strength:
            continue
        fname = lora_file(rec, bid)
        ok = compatibility(rec["family"], family)
        if ok is False:
            p.warnings.append("%s is a %s LoRA and %s is %s: left out rather than applied "
                              "to a model it was not trained for."
                              % (rec["name"], FAMILIES.get(rec["family"], rec["family"]),
                                 model["label"], FAMILIES.get(family, family)))
            continue
        if ok is None:
            p.warnings.append("%s has no model family set; applied, but check it suits %s."
                              % (rec["name"], model["label"]))
        if have is not None and fname not in have:
            p.warnings.append("%s (%s) is not on %s; left out." % (rec["name"], fname,
                                                                  backend["name"]))
            continue
        p.loras.append((fname, round(float(strength), 3)))
        p.lora_meta.append({"id": rec["id"], "name": rec["name"], "file": fname,
                            "strength": round(float(strength), 3), "category": rec["category"],
                            "why": why})

    # ----------------------------------------------------------- prompt
    who = [i["trigger"] for i, _ in idents if i["trigger"]]
    scene = _field(s, "scene")
    person = person_text(s)
    named = " and ".join(t for t in who if t not in scene)
    posed = bool((s.get("references") or {}).get("pose"))
    parts = [x for x in (view_text(s.get("view"), posed),
                         ", ".join(x for x in (named, person) if x), scene,
                         _field(s, "camera")) if x]
    if posed and clean_view(s.get("view")):
        p.warnings.append("The drawn pose decides the framing and which way the person "
                          "faces; the Camera gives only its height. Zoom the figure "
                          "(mouse wheel in Draw…) to frame closer.")
    anatomy = s.get("anatomy") is not False and has_person(s, bool(idents))
    if anatomy:
        parts.append(anatomy_text())
    if style:
        extra = " ".join(x for x in (style["trigger"], style["prompt"]) if x)
        if extra:
            parts.append(extra)
    for rec, _, why in stack:
        if why == "added" and rec["trigger"] and rec["trigger"] not in " ".join(parts):
            parts.append(rec["trigger"])
    p.prompt = ". ".join(x.rstrip(" .") for x in parts if x) + ("." if parts else "")
    if s.get("prompt_override"):
        # The Visual Critic's regeneration: its compiled prompt, built round
        # the prompt above (studio_critic.generator_prompt), in its place.
        p.prompt = s["prompt_override"]
    if not p.prompt:
        p.errors.append("Describe the scene or the person, or choose a person.")
    p.negative = ", ".join(x for x in (s["negative"].strip(),
                                        style["negative"] if style else "",
                                        anatomy_negative() if anatomy else "") if x)
    if style and style["families"] and family not in style["families"]:
        p.warnings.append("The %s style was written for %s; %s may read it differently."
                          % (style["name"], ", ".join(FAMILIES.get(f, f) for f in style["families"]),
                             model["label"]))

    # ----------------------------------------------------------- values
    look = dict(model["defaults"])
    for k in ("sampler", "scheduler", "guidance", "steps", "width", "height"):
        if style and style.get(k) not in (None, ""):
            look[k] = style[k]
    look.update({k: x for k, x in preset["values"].items()})
    for k in ("steps", "guidance", "sampler", "scheduler", "width", "height", "denoise",
              "refine", "upscale", "refine_denoise", "face_detail"):
        if s.get(k) not in (None, ""):
            look[k] = s[k]
    v = dict(mvalues)
    # A file only an optional input needs (the pose ControlNet) may be named by
    # the workflow rather than the model, so a model saved before the input
    # existed still gets it; it is kept below only if that input is used.
    borrowed = [var for var in (wf.get("files") or {}) if var not in v
                and (wf.get("defaults") or {}).get(var)]
    v.update({var: wf["defaults"][var] for var in borrowed})
    v.update(look)
    v["prompt"], v["negative"] = p.prompt, p.negative
    v["seed"] = int(s["seed"]) % (MAX_SEED + 1)
    v["encoder_device"] = "cpu" if backend.get("encoder_on_cpu") else "default"
    for k in ("width", "height"):
        if k in v:
            v[k] = max(256, int(v[k]) // 16 * 16)
    # A size the card cannot hold is refused here, not sent: a 16384x16384
    # FLUX job on the 5090 ran out of memory in the VAE and took ComfyUI's
    # worker down with it (2026-09-25), leaving the prompt "running" for good.
    cap = backend.get("max_megapixels", 4.2)
    if "width" in v and "height" in v and v["width"] * v["height"] > cap * 1e6:
        p.errors.append("%dx%d is %.1f megapixels; %s is set to %.1f at most (Backends). "
                        "Make the picture smaller." % (v["width"], v["height"],
                                                      v["width"] * v["height"] / 1e6,
                                                      backend["name"], cap))

    # ------------------------------------------------------- references
    refs = {k: x for k, x in (s["references"] or {}).items() if x}
    if "face" not in refs:
        for ident, _ in idents:
            if ident["use_references"] and ident["references"]:
                refs["face"] = ident["references"][0]
                v.setdefault("face_strength", ident["reference_strength"])
                p.notes.append("Face reference from %s's profile." % ident["name"])
                break
    slots = wf.get("references") or {}
    needs = wf.get("needs") or {}

    def lacking(var):
        """The files `var`'s input needs that the backend lacks, named."""
        return ", ".join(str(v.get(f)) for f in needs.get(var, []) if inventory is not None
                         and v.get(f) not in inventory.get(wf.get("files", {}).get(f, ""),
                                                           set()))
    for kind in REFERENCE_NAMES:
        path = refs.get(kind)
        if not path:
            continue
        label = dict((k, l) for k, l, _ in REFERENCE_KINDS)[kind]
        var = slots.get(kind)
        if not var:
            text = ("%s reference: the %s workflow has no %s conditioning, so it was not "
                    "used." % (label, wf.get("label", wid), kind))
            (p.notes if kind == "face" and "face" not in s["references"] else
             p.warnings).append(text + (" The identity LoRA carries the likeness."
                                        if kind == "face" else ""))
            continue
        if var in p.images:
            p.warnings.append("%s reference: the workflow's %s input is already taken by "
                              "another reference; not used." % (label, var))
            continue
        if not os.path.isfile(path):
            p.warnings.append("%s reference %s is not on this PC any more; not used."
                              % (label, path))
            continue
        short = lacking(var)
        if short:
            p.warnings.append("%s reference: %s lacks %s, which it needs; not used."
                              % (label, backend["name"], short))
            continue
        p.images[var] = path
        p.references[kind] = path
    plan_items(p, s, wf, v, backend, lacking("items"), nodes)
    if "source_image" in p.images and s.get("denoise") in (None, ""):
        v["denoise"] = wf.get("source_denoise", 0.65)
    # Denoise below 1 over an empty latent leaves noise in the picture: it
    # only means something with a source picture to start from.
    if ("source_image" not in p.images and uses(wf, "source_image")
            and float(v.get("denoise") or 1.0) < 1.0):
        if s.get("denoise") not in (None, ""):
            p.notes.append("Denoise %s is for a source picture; there is none, so the "
                           "picture is made from noise (1.0)." % s["denoise"])
        v["denoise"] = 1.0
    comp = s.get("composition") if isinstance(s.get("composition"), dict) else {}
    if p.references.get("composition") and comp.get("strength") not in (None, ""):
        v["composition_strength"] = round(float(comp["strength"]), 3)
    pose = s.get("pose") if isinstance(s.get("pose"), dict) else {}
    if p.references.get("pose") and pose.get("strength") not in (None, ""):
        v["pose_strength"] = round(float(pose["strength"]), 3)
    if p.references.get("pose") and pose.get("hands"):
        import studio_pose
        hidden = set(pose.get("hidden") or ())
        seen = [None if i in hidden else x for i, x in enumerate(pose.get("points") or [])]
        words = studio_pose.hands_text(seen, pose["hands"])
        if words:
            p.prompt = (p.prompt + " " + words).strip()
            v["prompt"] = p.prompt
    for var in borrowed:
        if not any(var in fs and img in p.images for img, fs in needs.items()):
            v.pop(var, None)

    if v.get("refine") and not uses(wf, "refine"):
        if s.get("refine") or preset["values"].get("refine"):
            p.warnings.append("The %s workflow has no refine pass; the picture is made "
                              "without one." % wf.get("label", wid))
        v["refine"] = False
    # A refine pass is sized to what the backend's card holds.
    if v.get("refine") and "upscale" in v and "width" in v:
        cap = backend.get("max_megapixels", 4.2) * 1e6
        scale = min(float(v["upscale"]), (cap / float(v["width"] * v["height"])) ** 0.5)
        if scale < float(v["upscale"]) - 0.01:
            p.notes.append("Upscale held to x%.2f: %s is set to %.1f megapixels at most."
                           % (scale, backend["name"], cap / 1e6))
        v["upscale"] = round(scale, 3)
        if scale <= 1.05:
            v["refine"] = False

    # The face pass needs the template's face_detail section, a SAM3
    # checkpoint to find the faces and the stock nodes it is built from. It
    # is a finish, not the picture: without them the picture is made and the
    # pass is left out, said once.
    if v.get("face_detail"):
        asked = s.get("face_detail") or preset["values"].get("face_detail")
        ckpts = (inventory or {}).get("checkpoints") if inventory is not None else None
        sam = sorted(c for c in ckpts or () if SAM3 in c.lower())
        why = ""
        if not wf.get("face_detail"):
            why = "The %s workflow has no face pass" % wf.get("label", wid)
        elif ckpts is not None and not sam:
            why = ("%s has no SAM3 checkpoint (a file with sam3 in its name, in "
                   "ComfyUI/models/checkpoints), which finds the faces" % backend["name"])
        elif nodes is not None and FACE_NODES - set(nodes):
            why = "%s's ComfyUI lacks the node(s) %s" % (
                backend["name"], ", ".join(sorted(FACE_NODES - set(nodes))))
        if why:
            if asked:
                p.warnings.append(why + "; the picture is made without the face pass.")
            v["face_detail"] = False
        else:
            v["sam3"] = sam[0] if sam else None
            # The face is redrawn without the item pictures, so without their line.
            v["face_prompt"] = FACE_PROMPT % p.prompt.replace(p.item_text, "").strip()
            v.setdefault("face_denoise", (wf.get("defaults") or {}).get("face_denoise", 0.4))

    # ------------------------------------------- files and nodes it needs
    optional = {var for fs in needs.values() for var in fs}   # checked with their reference
    missing = lacks(wf, v, inventory, nodes, skip=optional)
    files = [m["text"] for m in missing if m["kind"] == "file"]
    if files:
        p.errors.append("%s is missing %s for %s: %s. Put each file in that folder on %s, "
                        "or map %s to that machine's filenames in Models."
                        % (backend["name"], "a file" if len(files) == 1 else
                           "%d files" % len(files), model["label"], "; ".join(files),
                           backend["name"], model["label"]))
    node_names = [m["name"] for m in missing if m["kind"] == "node"]
    if node_names:
        p.errors.append("%s's ComfyUI lacks the node%s %s that the %s workflow uses."
                        % (backend["name"], "" if len(node_names) == 1 else "s",
                           ", ".join(node_names), wf.get("label", wid)))
    p.values = v
    return p


# ================================================================ routing

def route(role, backends, health, has_model=None, load=None):
    """Backends that can take a job of `role`, best first. Enabled and not
    known to be down; those with the model (when known) only. Backends
    whose roles include `role` lead, then the rest - so a job falls back
    to the other machine rather than waiting on a dead one. `load` (backend
    id -> jobs waiting) breaks ties within a group."""
    load = load or {}
    ok = []
    for b in backends:
        if not b["enabled"]:
            continue
        h = health.get(b["id"])
        if h is not None and not h.get("ok"):
            continue
        if has_model is not None and not has_model(b):
            continue
        ok.append(b)
    first = [b for b in ok if role in b["roles"]]
    rest = [b for b in ok if role not in b["roles"]]
    return (sorted(first, key=lambda b: load.get(b["id"], 0))
            + sorted(rest, key=lambda b: load.get(b["id"], 0)))


# ================================================================ jobs

STATUSES = ("queued", "uploading", "loading", "sampling", "decoding", "running", "refining",
            "complete", "failed", "cancelled")
FINISHED = ("complete", "failed", "cancelled")
# The stages a job is shown moving through. "running" and "refining" are what
# a node no stage names reports; "uploading" counts as queued.
STAGES = ("queued", "loading", "sampling", "decoding", "complete")
# Node classes by the stage they are, for a template that names none. A
# sampler node that has sent no step yet is loading: ComfyUI moves the model
# onto the card inside the sampler, before its first step.
STAGE_OF_CLASS = {
    "loading": ("Loader", "CLIPTextEncode", "LoadImage", "FluxGuidance",
                "ConditioningZeroOut", "EmptySD3LatentImage", "EmptyLatentImage",
                "ModelSampling"),
    "sampling": ("KSampler", "SamplerCustom", "SamplerCustomAdvanced"),
    "decoding": ("VAEDecode", "SaveImage", "PreviewImage"),
}


def stage_of(wf, graph, node_id):
    """The stage a node belongs to: the template's `stages` first, then its
    class. -> one of STAGES, or "" for a node neither names."""
    for stage, ids in (wf.get("stages") or {}).items():
        if node_id in ids and stage in STAGES:
            return stage
    cls = (graph.get(node_id) or {}).get("class_type", "")
    for stage, marks in STAGE_OF_CLASS.items():
        if any(m in cls for m in marks):
            return stage
    return ""


class Job:
    def __init__(self, settings, backend):
        self.id = uuid.uuid4().hex[:12]
        self.settings = settings
        self.backend = backend
        self.status = "queued"
        self.detail = ""
        self.progress = None          # 0..1 while known
        self.created = time.time()
        self.started = self.finished = None
        self.prompt_id = None
        self.plan = None
        self.outputs = []             # local paths
        self.record = None            # the history record, once complete
        self.graph = None             # the graph as submitted
        self.face_graph = None        # the face pass's graph, when it ran
        self.paste_graph = None       # the real-face paste's graph, when it ran
        self.real_faces = None        # [{"name", "box", "photos"}] for the paste, from the face pass
        self.face = None              # {"found", "redrawn", "denoise"} when it ran
        self.dress = None             # {"outfit", "passes", "head_crop", ...} when dressed
        self.refinement = None        # the Visual Critic's passes, when it ran
        self.notes = []               # things said on the way (no live progress, ...)
        self.cancel = threading.Event()

    @property
    def seed(self):
        return self.settings.get("seed")

    def elapsed(self):
        if self.started is None:
            return 0.0
        return (self.finished or time.time()) - self.started


class Lane:
    """One backend's worker: a thread taking jobs off its own list, so the
    5090 and the 3090 each run one at a time and both at once."""

    def __init__(self, backend):
        self.backend = backend
        self.waiting = []
        self.current = None
        self.cv = threading.Condition()
        self.thread = None


class JobQueue:
    def __init__(self, studio, notify):
        self.studio = studio
        self.notify = notify          # notify(job) from any thread
        self.jobs = []
        self.lanes = {}
        self.lock = threading.Lock()
        self.closed = False

    def load(self):
        return {bid: len(l.waiting) + (1 if l.current else 0) for bid, l in self.lanes.items()}

    def add(self, job):
        with self.lock:
            self.jobs.append(job)
            lane = self.lanes.get(job.backend["id"])
            if lane is None:
                lane = self.lanes[job.backend["id"]] = Lane(job.backend)
            lane.backend = job.backend
        with lane.cv:
            lane.waiting.append(job)
            lane.cv.notify()
        if lane.thread is None or not lane.thread.is_alive():
            lane.thread = threading.Thread(target=self._work, args=(lane,), daemon=True)
            lane.thread.start()
        self.notify(job)

    def cancel(self, job):
        if job.status in FINISHED:
            return
        job.cancel.set()
        lane = self.lanes.get(job.backend["id"])
        if lane is not None:
            with lane.cv:
                if job in lane.waiting:
                    lane.waiting.remove(job)
                    self._finish(job, "cancelled", "cancelled before it started")
                    return
        if job.prompt_id:
            try:
                self.studio.client(job.backend).cancel_job(job.prompt_id)
            except ComfyError:
                pass

    def close(self):
        self.closed = True
        for job in list(self.jobs):
            if job.status not in FINISHED:
                self.cancel(job)
        for lane in self.lanes.values():
            with lane.cv:
                lane.cv.notify_all()

    def _finish(self, job, status, detail=""):
        job.status, job.detail = status, detail
        job.finished = time.time()
        if job.started is None:
            job.started = job.finished
        self.notify(job)

    def _work(self, lane):
        while not self.closed:
            with lane.cv:
                while not lane.waiting and not self.closed:
                    if not lane.cv.wait(timeout=60):
                        if not lane.waiting:
                            lane.thread = None
                            return
                if self.closed:
                    return
                job = lane.current = lane.waiting.pop(0)
            try:
                self.studio.run_job(job, self.notify)
            except Exception as e:           # never let a lane die with a job half done
                if job.status not in FINISHED:
                    self._finish(job, "failed", "%s: %s" % (type(e).__name__, e))
                doctor.log_error("Image Studio job %s failed:\n%r" % (job.id, e))
            finally:
                lane.current = None
            if not lane.waiting and lane.backend.get("release_vram"):
                self.studio.client(lane.backend).free()


# ================================================================ history

def run_errors(entry, graph=None):
    """Why a finished prompt made nothing, from its /history entry, in words:
    the node (id and class), the exception and its message, and for a
    missing file the exact name. [] when the entry says nothing."""
    out = []
    for m in (entry.get("status") or {}).get("messages") or []:
        if not (isinstance(m, list) and len(m) == 2):
            continue
        kind, d = m[0], m[1] or {}
        if kind == "execution_interrupted":
            out.append("interrupted before it finished")
        elif kind == "execution_error":
            nid = str(d.get("node_id", "?"))
            cls = d.get("node_type") or (graph or {}).get(nid, {}).get("class_type", "?")
            msg = " ".join(str(d.get("exception_message", "")).split())
            text = "ComfyUI failed at node %s (%s): %s: %s" % (
                nid, cls, d.get("exception_type", "error"), msg[:600])
            if "out of memory" in msg.lower() or "OutOfMemory" in str(d.get("exception_type")):
                text += " - the GPU ran out of memory; free VRAM or lower the size."
            out.append(text)
    if not out:
        out = [x for x in status_messages(entry)]
    return out


class History:
    """Every finished job: its pictures and a JSON record beside them under
    history/<date>/. The record holds the settings exactly as submitted, so
    Reuse Settings and Generate Again read it back rather than guess."""

    def __init__(self, root=None):
        self.root = root or os.path.join(studio_dir(), "history")

    def folder_for(self, when):
        return os.path.join(self.root, time.strftime("%Y-%m-%d", time.localtime(when)))

    def add(self, record, pictures):
        """record: dict; pictures: [(filename, bytes)]. -> the record, saved."""
        folder = self.folder_for(record["created_ts"])
        os.makedirs(folder, exist_ok=True)
        record["images"] = []
        for i, (name, data) in enumerate(pictures):
            ext = os.path.splitext(name)[1] or ".png"
            path = os.path.join(folder, "%s_%d%s" % (record["id"], i + 1, ext))
            with open(path, "wb") as f:
                f.write(data)
            record["images"].append(path)
        path = os.path.join(folder, record["id"] + ".json")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        os.replace(tmp, path)
        record["path"] = path
        return record

    def list(self, limit=200):
        """Newest first. A record that will not parse is skipped here; the
        pictures beside it stay on disk."""
        out = []
        if not os.path.isdir(self.root):
            return out
        for day in sorted(os.listdir(self.root), reverse=True):
            folder = os.path.join(self.root, day)
            if not os.path.isdir(folder):
                continue
            recs = []
            for name in os.listdir(folder):
                if not name.endswith(".json"):
                    continue
                try:
                    with open(os.path.join(folder, name), encoding="utf-8") as f:
                        rec = json.load(f)
                except (OSError, ValueError):
                    continue
                if isinstance(rec, dict) and isinstance(rec.get("settings"), dict):
                    rec["path"] = os.path.join(folder, name)
                    recs.append(rec)
            out.extend(sorted(recs, key=lambda r: r.get("created_ts", 0), reverse=True))
            if len(out) >= limit:
                break
        return out[:limit]


AGAIN_PINNED = ("steps", "guidance", "sampler", "scheduler", "width", "height")


def again(record, new_seed=False):
    """Settings for Generate Again from a history record (or {"settings":
    ...} for a job not finished yet): the same picture - the seed it was made
    with, the sampler values it resolved to (so a changed model default does
    not change it), and the backend it ran on, preferred when it can still
    take it. `new_seed` is the variation: everything the same but the seed."""
    s = copy.deepcopy(record.get("settings") or {})
    s["batch"] = 1
    s.pop("batch_of", None)
    if new_seed:
        s["seed"], s["seed_mode"] = -1, "random"
        return s
    seed = record.get("seed", s.get("seed"))
    if seed is not None and int(seed) >= 0:
        s["seed"], s["seed_mode"] = int(seed), "fixed"
    for k in AGAIN_PINNED:
        if s.get(k) in (None, "") and record.get(k) not in (None, ""):
            s[k] = record[k]
    if (record.get("backend") or {}).get("id"):
        s["prefer_backend"] = record["backend"]["id"]
    return s


# ======================================================= person cut-out
# The Identities editor's "Pick person": SAM3 finds everyone in a reference
# photo (`people_graph`), the user clicks one when there is more than one,
# and a second run (`cutout_graph`) crops that person and puts them on white,
# so the face reference is them and nobody beside them.
PEOPLE_PROMPT = "person:8"
CUTOUT_PAD = 0.04             # of the box's larger side, added around the person
CUTOUT_BG = 0xFFFFFF


def people_graph(image, sam3):
    """Everyone SAM3 finds in `image` (a LoadImage name), the picture's size,
    and a PNG of it for Tk to show (it reads no JPEG)."""
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": image}},
        "2": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": sam3}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": PEOPLE_PROMPT,
                                                         "clip": ["2", 1]}},
        "4": {"class_type": "SAM3_Detect", "inputs": {
            "model": ["2", 0], "image": ["1", 0], "conditioning": ["3", 0],
            "threshold": 0.3, "refine_iterations": 0, "individual_masks": True}},
        "5": {"class_type": "PreviewAny", "inputs": {"source": ["4", 1]}},
        "6": {"class_type": "GetImageSize", "inputs": {"image": ["1", 0]}},
        "7": {"class_type": "PreviewAny", "inputs": {"source": ["6", 0]}},
        "8": {"class_type": "PreviewAny", "inputs": {"source": ["6", 1]}},
        "9": {"class_type": "PreviewImage", "inputs": {"images": ["1", 0]}},
    }


def people_found(entry):
    """What people_graph said -> (width, height, [(x, y, w, h)] largest first,
    the preview's file dict or None). None when it said nothing."""
    out = entry.get("outputs") or {}

    def text(node):
        t = (out.get(node) or {}).get("text") or []
        return json.loads(t[0]) if t else None
    try:
        boxes, width, height = text("5"), text("7"), text("8")
    except ValueError:
        return None
    if width is None or height is None:
        return None
    boxes = boxes[0] if boxes and isinstance(boxes[0], list) else boxes or []
    boxes = sorted(((b["x"], b["y"], b["width"], b["height"]) for b in boxes),
                   key=lambda b: -b[2] * b[3])
    if boxes:                     # a head at the frame's edge is not a person to pick
        big = boxes[0][2] * boxes[0][3]
        boxes = [b for b in boxes if b[2] * b[3] >= big / 20]
    imgs = (out.get("9") or {}).get("images") or []
    return int(width), int(height), boxes, (imgs[0] if imgs else None)


def pick_box(boxes, x, y):
    """The box a click at (x, y) means: the smallest one holding it, else the
    one whose centre is nearest."""
    inside = [b for b in boxes if b[0] <= x <= b[0] + b[2] and b[1] <= y <= b[1] + b[3]]
    if inside:
        return min(inside, key=lambda b: b[2] * b[3])
    return min(boxes, key=lambda b: (b[0] + b[2] / 2.0 - x) ** 2 + (b[1] + b[3] / 2.0 - y) ** 2)


def cutout_region(box, width, height, pad=CUTOUT_PAD):
    x, y, w, h = box
    m = int(max(w, h) * pad)
    x0, y0 = max(0, int(x) - m), max(0, int(y) - m)
    x1, y1 = min(width, int(x + w) + m), min(height, int(y + h) + m)
    return {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0}


def cutout_graph(image, sam3, region, prefix="studio_person"):
    """`image` cropped to `region`, the main person in it masked by SAM3 and
    laid on white, saved under `prefix`. A neighbour's shoulder inside the
    crop is not the main person, so it goes white with the background."""
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": image}},
        "2": {"class_type": "ImageCropV2", "inputs": {"image": ["1", 0], "crop_region": region}},
        "3": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": sam3}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "person:1", "clip": ["3", 1]}},
        "5": {"class_type": "SAM3_Detect", "inputs": {
            "model": ["3", 0], "image": ["2", 0], "conditioning": ["4", 0],
            "threshold": 0.3, "refine_iterations": 2, "individual_masks": False}},
        "6": {"class_type": "EmptyImage", "inputs": {
            "width": region["width"], "height": region["height"], "batch_size": 1,
            "color": CUTOUT_BG}},
        "7": {"class_type": "ImageCompositeMasked", "inputs": {
            "destination": ["6", 0], "source": ["2", 0], "x": 0, "y": 0,
            "resize_source": False, "mask": ["5", 0]}},
        "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0],
                                                    "filename_prefix": prefix}},
    }


# ================================================================ the studio

class Studio:
    """The Image Studio without its window: library, clients, health, the
    queue and history. `make_room(backend)` is the host's hook to clear
    LM Studio off a shared GPU before a job there (Chat._images_make_room)."""

    def __init__(self, root=None, notify=lambda job: None, make_room=None,
                 client_factory=None, workflow_loader=load_workflow, vision=None):
        self.lib = Library(root)
        self.history = History(os.path.join(self.lib.root, "history"))
        self.client_factory = client_factory or ComfyUIClient   # read late: tests swap it
        self.workflow_loader = workflow_loader
        self.make_room = make_room
        self.vision = vision          # () -> studio_agent.Vision or None: the critic's eyes
        self.clients = {}
        self.health = {}              # backend id -> health dict (+ "at")
        self.inventories = {}         # backend id -> {kind: set}
        self.nodes = {}               # backend id -> set of node classes
        self.queue = JobQueue(self, notify)

    def backends(self):
        return self.lib.all("backends")

    def backend(self, bid):
        return self.lib.get("backends", bid)

    def client(self, backend):
        c = self.clients.get(backend["id"])
        if c is None or c.url != backend["url"].rstrip("/"):
            c = self.clients[backend["id"]] = self.client_factory(backend)
        c.backend = backend
        return c

    # ------------------------------------------------------------ health
    def check(self, backend, full=True):
        """Health (and, with `full`, the model inventory) of one backend.
        Network I/O: call it off the UI thread."""
        h = self.client(backend).health() if backend["enabled"] else {
            "ok": False, "detail": "disabled", "queue": 0}
        h["at"] = time.time()
        self.health[backend["id"]] = h
        if h["ok"] and full:
            c = self.client(backend)
            self.inventories[backend["id"]] = c.inventory()
            try:
                self.nodes[backend["id"]] = set(c.node_types(fresh=True))
            except TypeError:             # a client with no `fresh` (the tests')
                self.nodes[backend["id"]] = set(c.node_types())
            except ComfyError as e:
                h["detail"] += "; its node list would not load (%s)" % e
        return h

    def check_all(self, full=True):
        threads = [threading.Thread(target=self.check, args=(b, full), daemon=True)
                   for b in self.backends()]
        for t in threads:
            t.start()
        for t in threads:
            t.join(20)

    def scan_loras(self):
        """Merge every online backend's LoRA list into the library, and save.
        -> number added."""
        added = 0
        for b in self.backends():
            inv = self.inventories.get(b["id"])
            if inv is None:
                continue
            added += self.lib.merge_loras(b["id"], sorted(inv.get("loras", ())),
                                          b.get("lora_dir", ""))
        self.lib.save("loras")
        return added

    # ------------------------------------------------------ person cut-out
    def sam3_backend(self):
        """(backend, SAM3 checkpoint) on the first online backend that has
        one; checks the backends first if none is known. None if none."""
        for attempt in (0, 1):
            for b in self.backends():
                if not (self.health.get(b["id"]) or {}).get("ok"):
                    continue
                sam = sorted(c for c in (self.inventories.get(b["id"]) or {}).get(
                    "checkpoints", ()) if SAM3 in c.lower())
                if sam:
                    return b, sam[0]
            if not attempt:
                self.check_all()
        return None

    def _run_quick(self, client, graph, timeout=300):
        pid = client.queue_workflow(graph)
        end = time.time() + timeout
        while time.time() < end:
            entry = client.get_history(pid)
            if entry:                 # history holds a prompt once it has finished
                status = entry.get("status") or {}
                if status.get("status_str") == "error":
                    raise ComfyError("%s: %s" % (client.backend["name"], "; ".join(
                        str(m[1].get("exception_message", m[0])) for m in
                        status.get("messages") or [] if m[0] == "execution_error")
                        or "the run failed"))
                return entry
            time.sleep(0.5)
        raise ComfyError("%s took over %d s" % (client.backend["name"], timeout))

    def find_people(self, path):
        """Everyone in the picture at `path`. -> dict with backend, sam3, the
        uploaded name, width, height, boxes (largest first) and preview (the
        picture as PNG bytes). Network I/O: off the UI thread."""
        found = self.sam3_backend()
        if found is None:
            raise ComfyError("No online ComfyUI has a SAM3 checkpoint (a file with sam3 "
                             "in its name under checkpoints).")
        backend, sam3 = found
        c = self.client(backend)
        name = c.upload_image(path)
        said = people_found(self._run_quick(c, people_graph(name, sam3)))
        if said is None:
            raise ComfyError("SAM3 said nothing about the picture.")
        width, height, boxes, pv = said
        return {"backend": backend, "sam3": sam3, "image": name, "width": width,
                "height": height, "boxes": boxes, "preview": c.fetch(pv) if pv else None}

    def cut_person(self, found, box):
        """The person in `box` of what find_people found, on white: PNG bytes."""
        c = self.client(found["backend"])
        region = cutout_region(box, found["width"], found["height"])
        entry = self._run_quick(c, cutout_graph(found["image"], found["sam3"], region))
        imgs = ((entry.get("outputs") or {}).get("8") or {}).get("images") or []
        if not imgs:
            raise ComfyError("The cut-out came back empty.")
        return c.fetch(imgs[0])

    def missing(self, model, backend):
        """missing_for() with what is known of `backend`; no I/O."""
        return missing_for(model, backend, self.inventories.get(backend["id"]),
                           self.nodes.get(backend["id"]), self.workflow_loader)

    def has_model(self, model_id):
        """backend -> True when every file and node the model's workflow
        needs is known to be there. Unknown (not checked yet) counts as
        there - routing must not refuse on a guess - but a backend that has
        answered is held to its lists."""
        model = self.lib.get("models", model_id)

        def check(b):
            if model is None:
                return False
            problems, lacking = self.missing(model, b)
            return not problems and not lacking
        return check

    def readiness(self, model):
        """{backend id: (state, text)} for the Models window and the model
        menu: state is "ready", "missing", "offline", "disabled" or
        "unchecked", and text says exactly what is missing where."""
        out = {}
        for b in self.backends():
            h = self.health.get(b["id"])
            problems, lacking = self.missing(model, b)
            if problems:
                out[b["id"]] = ("missing", " ".join(problems))
            elif not b["enabled"]:
                out[b["id"]] = ("disabled", "disabled in Backends")
            elif h is None:
                out[b["id"]] = ("unchecked", "not checked yet")
            elif not h.get("ok"):
                out[b["id"]] = ("offline", "offline: " + h.get("detail", ""))
            elif b["id"] not in self.inventories:
                out[b["id"]] = ("unchecked", "online; its model list has not been read")
            elif lacking:
                out[b["id"]] = ("missing", "missing " + "; ".join(m["text"] for m in lacking))
            else:
                out[b["id"]] = ("ready", "every file and node is there")
        return out

    def plan_route(self, settings, load=None):
        """Where one job of `settings` would go, and why, from what is known -
        no I/O, so the form can say it before Generate. -> (backend or None,
        text). A named backend is taken as named (its gaps are compose()'s
        errors); Auto takes the backend the settings prefer (Generate Again's
        original), then those whose roles include the preset's, when each is
        up and has everything the model's workflow needs."""
        model = self.lib.get("models", settings.get("model"))
        label = model["label"] if model else settings.get("model")
        if settings.get("backend") not in (None, "", "auto"):
            b = self.backend(settings["backend"])
            if b is None:
                return None, "No backend called %r." % settings["backend"]
            return b, "%s (chosen by hand)." % b["name"]
        role = PRESETS.get(settings.get("preset"), PRESETS["standard"])["role"]
        if int(settings.get("batch") or 1) > 1 and role == "interactive":
            role = "batch"
        order = route(role, self.backends(), self.health,
                      self.has_model(settings.get("model")), load)
        if not order:
            return None, self.why_no_backend(settings)
        pick = order[0]
        prefer = settings.get("prefer_backend")
        why = []
        if prefer:
            if any(b["id"] == prefer for b in order):
                pick = self.backend(prefer)
                why.append("where this picture was made")
            else:
                gone = self.backend(prefer)
                why.append("%s, where it was made, cannot take it now, so the result "
                           "may differ slightly" % (gone["name"] if gone else prefer))
        if not why:
            why.append("preferred for %s" % PRESETS.get(settings.get("preset"),
                                                           PRESETS["standard"])["label"]
                       if role in pick["roles"] else "the only backend able to take it")
        passed = []                   # preferred for the role, but unable to take it
        for b in self.backends():
            if (b["enabled"] and role in b["roles"] and b["id"] != pick["id"]
                    and not any(o["id"] == b["id"] for o in order)):
                h = self.health.get(b["id"]) or {}
                passed.append("%s is offline" % b["name"] if not h.get("ok") else
                              "%s lacks files %s needs" % (b["name"], label))
        known = pick["id"] in self.inventories
        return pick, "Auto → %s: %s%s%s." % (
            pick["name"], "; ".join(why),
            "" if not known else "; has every file %s needs" % label,
            "" if not passed else " (" + "; ".join(passed) + ")")

    def pick_backends(self, settings, count=1):
        """The backend for each of `count` jobs. A named backend takes them
        all; Auto routes as plan_route says (batch when more than one),
        spreading a batch over every capable backend."""
        if settings.get("backend") not in (None, "", "auto"):
            b = self.backend(settings["backend"])
            if b is None:
                raise ComfyError("No backend called %r." % settings["backend"])
            return [b] * count
        for b in self.backends():
            h = self.health.get(b["id"])
            if b["enabled"] and (h is None or time.time() - h.get("at", 0) > HEALTH_TTL
                                 or (h.get("ok") and b["id"] not in self.inventories)):
                self.check(b)
        load = self.queue.load()
        picks = []
        for _ in range(count):
            if count == 1:
                b, why = self.plan_route(dict(settings, batch=1), load)
                if b is None:
                    raise ComfyError(why)
                return [b]
            role = PRESETS.get(settings.get("preset"), PRESETS["standard"])["role"]
            role = "batch" if role == "interactive" else role
            order = route(role, self.backends(), self.health,
                          self.has_model(settings.get("model")), load)
            if not order:
                raise ComfyError(self.why_no_backend(settings))
            order.sort(key=lambda b: load.get(b["id"], 0))  # least loaded capable machine
            picks.append(order[0])
            load[order[0]["id"]] = load.get(order[0]["id"], 0) + 1
        return picks

    def why_no_backend(self, settings):
        lines = []
        model = self.lib.get("models", settings.get("model"))
        if model is None:
            return "No model called %r in the library." % settings.get("model")
        for b in self.backends():
            h = self.health.get(b["id"]) or {}
            problems, lacking = self.missing(model, b)
            if not b["enabled"]:
                lines.append("%s is disabled" % b["name"])
            elif problems:
                lines.append(" ".join(problems).rstrip("."))
            elif not h.get("ok"):
                lines.append("%s is offline (%s)" % (b["name"], h.get("detail", "not checked")))
            elif lacking:
                lines.append("%s is missing %s" % (b["name"], "; ".join(m["text"]
                                                                       for m in lacking)))
        return "No backend can take this job: " + "; ".join(lines) + "."

    # ------------------------------------------------------------ submit
    def preview(self, settings, backend=None):
        """compose() against a backend, for the form's warnings; no I/O."""
        b = backend or next((x for x in self.backends() if x["enabled"]), None)
        if b is None:
            return None
        return compose(settings, self.lib, b, self.inventories.get(b["id"]),
                       self.workflow_loader, self.nodes.get(b["id"]))

    # ------------------------------------------------------------ poses
    def find_poses(self, path, stop=None):
        """The people in the picture at `path` and their pose points, from
        the first enabled backend that is up and has POSE_NODE: -> (the
        node's JSON, parsed; that backend). Not a job - a second or so on
        the CPU, no model loaded, nothing kept in History. Network I/O: call
        it off the UI thread. Raises ComfyError saying why no backend could."""
        why = []
        for b in self.backends():
            if not b["enabled"]:
                continue
            nodes = self.nodes.get(b["id"]) or set()
            if POSE_NODE not in nodes:          # a ComfyUI restarted since it was read
                h = self.check(b)
                nodes = self.nodes.get(b["id"]) or set()
                if not h.get("ok"):
                    why.append("%s is offline (%s)" % (b["name"], h.get("detail", "")))
                    continue
            if POSE_NODE not in nodes:
                why.append("%s has no %s node" % (b["name"], POSE_NODE))
                continue
            c = self.client(b)
            graph = {"1": {"class_type": "LoadImage",
                           "inputs": {"image": c.upload_image(path)}},
                     "2": {"class_type": POSE_NODE, "inputs": {"image": ["1", 0]}}}
            entry = c.listen_for_progress(c.queue_workflow(graph), lambda *a: None,
                                          stop=stop, timeout=180)
            if entry is None:
                raise ComfyError("Stopped before %s answered." % b["name"])
            text = ((entry.get("outputs") or {}).get("2") or {}).get("text") or []
            try:
                return json.loads(text[0]), b
            except (IndexError, TypeError, ValueError):
                raise ComfyError("%s could not find the pose: %s" % (
                    b["name"], "; ".join(run_errors(entry, graph)) or "it said nothing"))
        raise ComfyError(
            "No backend can find a pose in a photo: %s. The finder is a ComfyUI node "
            "of ours: copy comfy_nodes/studio_dwpose into ComfyUI's custom_nodes, put "
            "yolox_l.onnx and dw-ll_ucoco_384.onnx (huggingface.co/yzd-v/DWPose) in a "
            "'dwpose' model folder, and restart ComfyUI."
            % ("; ".join(why) or "no backend is enabled"))

    def submit(self, settings):
        """Queue the form's settings: one job per picture in the batch, each
        with its own seed so each picture's record says exactly how it was
        made. Routing does network I/O: call off the UI thread. -> [Job]."""
        if settings.get("mode") == "dress":
            return self.submit_dress(settings)
        s = copy.deepcopy(settings)
        count = max(1, min(int(s.get("batch") or 1), 64))
        seed = int(s.get("seed", -1))
        if seed < 0:
            s["seed_mode"] = "random"
            seed = random.randint(0, MAX_SEED)
        else:
            s["seed_mode"] = "fixed"
        jobs = []
        for i, b in enumerate(self.pick_backends(s, count)):
            one = copy.deepcopy(s)
            one["seed"] = (seed + i) % (MAX_SEED + 1)
            one["batch"] = 1
            one["batch_of"] = count
            job = Job(one, b)
            jobs.append(job)
            self.queue.add(job)
        return jobs

    # --------------------------------------------------------------- run
    def run_job(self, job, notify):
        """One job start to finish, on its lane's thread."""
        def say(status=None, detail=None, progress=None):
            if status:
                job.status = status
            if detail is not None:
                job.detail = detail
            job.progress = progress
            notify(job)

        b = job.backend
        client = self.client(b)
        job.started = time.time()
        if job.cancel.is_set():
            return self.queue._finish(job, "cancelled")
        say(detail="checking %s" % b["name"])
        h = self.check(b, full=b["id"] not in self.inventories)
        if not h["ok"]:
            return self.queue._finish(job, "failed", h["detail"])
        if job.settings.get("mode") == "dress":
            return self.run_dress(job, client, say)
        plan = compose(job.settings, self.lib, b, self.inventories.get(b["id"]),
                       self.workflow_loader, self.nodes.get(b["id"]))
        job.plan = plan
        if plan.errors:
            return self.queue._finish(job, "failed", " ".join(plan.errors))

        say("uploading" if plan.images or plan.items else None,
            "uploading references" if plan.images or plan.items else "")
        values = dict(plan.values)
        for var, path in plan.images.items():
            if job.cancel.is_set():
                return self.queue._finish(job, "cancelled")
            values[var] = client.upload_image(path)
        items = []
        for _, path in plan.items:
            if job.cancel.is_set():
                return self.queue._finish(job, "cancelled")
            items.append(client.upload_image(path))
        # One output name per job, so the file on the backend says which job
        # made it (ComfyUI adds _00001_ and the extension).
        values["filename_prefix"] = "ImageStudio/%s_%s" % (plan.workflow.get("id", "job"),
                                                           job.id)
        try:
            graph = fill(plan.workflow, values, plan.loras)
        except TemplateError as e:
            return self.queue._finish(job, "failed", str(e))
        if items:
            add_item_refs(graph, plan.workflow["items"], items)
        try:
            types = set(client.node_types())
        except ComfyError:
            types = None
        if not self._faces_into_picture(job, client, plan, graph, types,
                                        values.get("width"), values.get("height")):
            return self.queue._finish(job, "cancelled")
        lacking = missing_nodes(graph, types) if types is not None else []
        if values.get("face_detail"):
            if not values.get("sam3"):
                values["face_detail"] = False
                plan.warnings.append("%s's SAM3 checkpoint is not known; the picture is made "
                                     "without the face pass." % b["name"])
            elif types is not None and FACE_NODES - types:
                values["face_detail"] = False
                plan.warnings.append("%s's ComfyUI lacks %s; the picture is made without the "
                                     "face pass." % (b["name"], ", ".join(sorted(FACE_NODES
                                                                                 - types))))
            else:
                add_face_finder(graph, values["sam3"])
        if lacking:
            return self.queue._finish(job, "failed", "%s's ComfyUI lacks the node(s) %s that "
                                      "the %s workflow uses." % (
                                          b["name"], ", ".join(lacking),
                                          plan.workflow.get("label")))
        job.graph = graph
        if b.get("shares_llm_gpu") and self.make_room is not None:
            say(detail="clearing LM Studio off the GPU")
            try:
                self.make_room(b)
            except Exception as e:
                plan.notes.append("Could not clear the shared GPU (%s); this may be slow." % e)
        if job.cancel.is_set():
            return self.queue._finish(job, "cancelled")
        watch = client.watch() if hasattr(client, "watch") else None
        try:
            try:
                job.prompt_id = client.queue_workflow(graph)
            except Unreachable as e:
                return self.queue._finish(job, "failed", "The workflow could not be sent: %s"
                                          % e)
            except ComfyError as e:
                return self.queue._finish(job, "failed", "%s's ComfyUI rejected the workflow "
                                          "before running it. %s" % (b["name"], e))
            say("queued", "queued on %s" % b["name"], None)
            entry = client.listen_for_progress(
                job.prompt_id, self._progress(job, graph, say), stop=job.cancel.is_set,
                **({"watch": watch} if watch is not None else {}))
        finally:
            if watch is not None:
                watch.close()
        if entry is None:
            return self.queue._finish(job, "cancelled")
        files = [f for f in outputs_of(entry)]
        if not files:
            errors = run_errors(entry, graph)
            if any("interrupted" in e for e in errors) or job.cancel.is_set():
                return self.queue._finish(job, "cancelled")
            return self.queue._finish(job, "failed", "; ".join(errors) or
                                      "The workflow finished on %s without a picture."
                                      % b["name"])
        if values.get("face_detail") and not job.cancel.is_set():
            files, graph2 = self._face_pass(job, client, plan, values, entry, files, say)
            if graph2 is not None:
                job.face_graph = graph2
                if job.real_faces and not job.cancel.is_set():
                    files = self._real_faces(job, client, plan, values, files, say)
        if job.settings.get("auto_refine") and not job.cancel.is_set():
            files = self._refine(job, client, plan, values, files, say)
        if job.cancel.is_set():
            return self.queue._finish(job, "cancelled")
        say("decoding", "fetching the picture from %s" % b["name"], None)
        try:
            pictures = [(f["filename"], client.fetch(f)) for f in files]
        except ComfyError as e:
            return self.queue._finish(job, "failed", "The picture was made but could not be "
                                      "fetched from %s: %s" % (b["name"], e))
        job.record = self.history.add(self.record_for(job, graph), pictures)
        job.outputs = list(job.record["images"])
        job.progress = 1.0
        self.queue._finish(job, "complete",
                           "; ".join(plan.warnings[:1]) if plan.warnings else "")

    # ------------------------------------------------------------ try on
    def sam3_on(self, backend):
        """The SAM3 checkpoint on `backend`, from its last full check, or None."""
        inv = self.inventories.get(backend["id"]) or {}
        return next((c for c in sorted(inv.get("checkpoints") or ()) if SAM3 in c.lower()),
                    None)

    def _dress(self, job, client, outfit, person, size, say, finder=None):
        """Dress `person` (a LoadImage name on the job's backend) of `size`
        in `outfit`, on the lane's thread: the body run, then - when there is
        hair or a head accessory and SAM3 to find the head - the head run on
        a head-and-shoulders crop. `finder` (a SAM3 checkpoint) puts the
        face finder on the last run, for the face pass after. Sets job.dress.
        -> (entry, files) of the last run. Raises ComfyError."""
        b = job.backend
        wf = self.workflow_loader(DRESS_WORKFLOW)
        values = dress_values(wf, b)
        values["seed"] = int(job.settings.get("seed") or 0)
        passes = dress_passes(outfit)
        body = [x for x in passes if x["where"] == "body"]
        head = [x for x in passes if x["where"] == "head"]
        sam = self.sam3_on(b) if head else None
        prefix = "ImageStudio/dress_%s" % job.id
        job.dress = {"outfit": outfit, "passes": [dict(x, items=[i["name"] for i in x["items"]])
                                                  for x in passes],
                     "size": list(size), "head_crop": None, "graphs": []}
        if head and not sam:
            job.notes.append("%s has no SAM3 checkpoint to find the head, so the hair and "
                             "head accessories were drawn on the whole picture; in a "
                             "full-length picture they may not take." % b["name"])
        say("uploading", "uploading the outfit's pictures", None)
        pictures = {}
        for path in outfit_pictures(outfit):
            if job.cancel.is_set():
                raise ComfyError("cancelled")
            pictures[path] = client.upload_image(path)
        work = panel_size(size[0], size[1], float(values["panel_megapixels"]))
        first = body if sam else passes
        g1 = dress_graph(wf, values, first, person, size, pictures, prefix, find_head=sam)
        if not sam and finder:
            add_face_finder(g1, finder)
        entry = self._dress_run(job, client, g1, first, 0, say)
        if not sam:
            return entry, outputs_of(entry)
        found = face_boxes(entry)
        image = preview_of(entry, "dp")
        if image is None:
            raise ComfyError("the dress run gave no picture to dress the head on")
        crop = None
        if found and found[2]:
            box = max(found[2], key=lambda bx: bx[2] * bx[3])
            crop = head_region(box, work[0], work[1])
            job.dress["head_crop"] = crop
            if crop is None:
                job.notes.append("Try On: the head fills the picture, so the hair and "
                                 "head accessories were drawn on all of it.")
        else:
            job.notes.append("Try On: no face found, so the hair and head accessories were "
                             "drawn on the whole picture.")
        mask = os.path.join(self.lib.root, "dress_mask.png")
        if not os.path.isfile(mask):
            os.makedirs(self.lib.root, exist_ok=True)
            with open(mask, "wb") as fh:
                fh.write(soft_rect_png())
        g2 = dress_head_graph(wf, values, head, image, work, crop, client.upload_image(mask),
                              size, pictures, prefix, first=len(body), sam3=sam)
        if finder:
            add_face_finder(g2, finder)
        entry = self._dress_run(job, client, g2, head, len(body), say)
        return entry, outputs_of(entry)

    def _dress_run(self, job, client, graph, passes, first, say):
        """One Try On run, its progress said per pass. -> the /history
        entry. Raises ComfyError on a failed or cancelled run."""
        job.dress["graphs"].append(graph)
        names = {}
        for n, ps in enumerate(passes):
            words = {"clothes": "putting on the clothes", "hair": "doing the hair"}.get(
                ps["kind"]) or "putting on the " + _and([i["name"] for i in ps["items"]])
            for d in ("d%d_ks" % (n + 1), "h%d_ks" % (n + 1)):
                names[d] = words
        status = "sampling" if job.settings.get("mode") == "dress" else "refining"

        def on_event(kind, data):
            if kind == "executing" and data in names:
                say(status, "Try On · %s" % names[data], None)
            elif kind == "progress" and data[1]:
                value, total, nid = data
                say(status, "Try On · %s · step %d of %d" % (
                    names.get(nid, "dressing"), value, total), value / float(total))
            elif kind == "queued" and job.status in ("queued", "uploading"):
                say("queued", "queued on %s" % job.backend["name"], None)
        job.prompt_id = client.queue_workflow(graph)
        watch = client.watch() if hasattr(client, "watch") else None
        try:
            entry = client.listen_for_progress(
                job.prompt_id, on_event, stop=job.cancel.is_set,
                **({"watch": watch} if watch is not None else {}))
        finally:
            if watch is not None:
                watch.close()
        if entry is None:
            raise ComfyError("cancelled")
        if not outputs_of(entry) and preview_of(entry, "dp") is None:
            raise ComfyError("; ".join(run_errors(entry, graph)) or "the run made no picture")
        return entry

    def dress_route(self, settings):
        """The backend a Try On job goes to: the one named, else the one it
        was made on (Generate Again), else the first enabled, up, with
        everything the outfit needs - the primary (5090) first. No I/O. ->
        (backend or None, why)."""
        outfit = clean_outfit(settings.get("outfit"))
        try:
            wf = self.workflow_loader(DRESS_WORKFLOW)
        except TemplateError as e:
            return None, str(e)
        if settings.get("backend") not in (None, "", "auto"):
            order = [self.backend(settings["backend"])]
        else:
            order = sorted(self.backends(), key=lambda b: (
                b["id"] != settings.get("prefer_backend"), "primary" not in b["roles"]))
        why = []
        for b in order:
            if b is None or not b["enabled"]:
                continue
            h = self.health.get(b["id"]) or {}
            if not h.get("ok"):
                why.append("%s is offline" % b["name"])
                continue
            short = dress_lacks(wf, outfit, self.inventories.get(b["id"]),
                                self.nodes.get(b["id"]))
            if short:
                why.append("%s lacks %s" % (b["name"], "; ".join(m["text"] for m in short)))
                continue
            return b, "Try On will run on %s." % b["name"]
        return None, "No backend can dress this: " + ("; ".join(why) or "none is enabled") + "."

    def submit_dress(self, settings):
        """Queue a Try On job (network I/O: off the UI thread). -> [Job]."""
        s = copy.deepcopy(settings)
        s["mode"] = "dress"
        s["batch"] = 1
        if int(s.get("seed", -1)) < 0:
            s["seed"], s["seed_mode"] = random.randint(0, MAX_SEED), "random"
        else:
            s["seed_mode"] = "fixed"
        for b in self.backends():
            if b["enabled"] and (b["id"] not in self.health or b["id"] not in self.inventories):
                self.check(b)
        b, why = self.dress_route(s)
        if b is None:
            raise ComfyError(why)
        job = Job(s, b)
        self.queue.add(job)
        return [job]

    def run_dress(self, job, client, say):
        """A Try On job, start to finish, on its lane's thread."""
        b, s = job.backend, job.settings
        d = s.get("outfit") or {}
        outfit = clean_outfit(d)
        person = _str(d.get("person"))
        if not person:
            problem = "Choose the person to dress."
        elif not os.path.isfile(person):
            problem = "The picture of the person (%s) is not on this PC." % person
        elif not dress_passes(outfit):
            problem = "Add something to put on: clothes, hair or an accessory."
        else:
            gone = [x for x in outfit_pictures(outfit) if not os.path.isfile(x)]
            problem = "Not on this PC any more: %s." % ", ".join(gone) if gone else ""
        try:
            wf = self.workflow_loader(DRESS_WORKFLOW)
        except TemplateError as e:
            return self.queue._finish(job, "failed", str(e))
        short = dress_lacks(wf, outfit, self.inventories.get(b["id"]), self.nodes.get(b["id"]))
        if short and not problem:
            problem = "%s lacks %s." % (b["name"], "; ".join(m["text"] for m in short))
        size = file_size_of(person) if not problem else None
        if size is None and not problem:
            problem = "Could not read the size of %s (PNG, JPEG, WebP, GIF or BMP)." % person
        if problem:
            return self.queue._finish(job, "failed", problem)
        cap = b.get("max_megapixels", 4.2) * 1e6
        if size[0] * size[1] > cap:
            k = (cap / float(size[0] * size[1])) ** 0.5
            size = (int(size[0] * k) // 16 * 16, int(size[1] * k) // 16 * 16)
            job.notes.append("Saved at %dx%d: %s is set to %.1f megapixels at most."
                             % (size[0], size[1], b["name"], cap / 1e6))
        if b.get("shares_llm_gpu") and self.make_room is not None:
            say(detail="clearing LM Studio off the GPU")
            try:
                self.make_room(b)
            except Exception as e:
                job.notes.append("Could not clear the shared GPU (%s); this may be slow." % e)
        try:
            say("uploading", "uploading the person", None)
            entry, files = self._dress(job, client, outfit, client.upload_image(person), size,
                                       say)
        except (ComfyError, TemplateError, OSError) as e:
            if job.cancel.is_set():
                return self.queue._finish(job, "cancelled")
            return self.queue._finish(job, "failed", "Try On failed on %s: %s"
                                      % (b["name"], e))
        if job.cancel.is_set():
            return self.queue._finish(job, "cancelled")
        say("decoding", "fetching the picture from %s" % b["name"], None)
        try:
            pictures = [(f["filename"], client.fetch(f)) for f in files]
        except ComfyError as e:
            return self.queue._finish(job, "failed", "The picture was made but could not be "
                                      "fetched from %s: %s" % (b["name"], e))
        job.record = self.history.add(self.dress_record(job, wf, size), pictures)
        job.outputs = list(job.record["images"])
        job.progress = 1.0
        self.queue._finish(job, "complete")

    def dress_record(self, job, wf, size):
        """A Try On job's history record, in the fields a Generate record has."""
        s, b, v = job.settings, job.backend, dress_values(wf, job.backend)
        now = time.time()
        loras = [{"id": "", "name": "Lightning 4-step", "file": v["lightning"],
                  "strength": 1.0, "category": "Detail / Enhancement", "why": "speed"}]
        if any(x["kind"] == "clothes" for x in job.dress["passes"]):
            rec = self.lib.lora_by_file(v["tryon"]) or {}
            loras.append({"id": rec.get("id", ""), "name": rec.get("name") or "Clothes Try On",
                          "file": v["tryon"], "strength": v["tryon_strength"],
                          "category": "Clothing", "why": "try on"})
        outfit = job.dress["outfit"]
        refs = {"person": (s.get("outfit") or {}).get("person")}
        refs.update({"item: " + i["name"]: i["path"]
                     for i in outfit["clothes"] + outfit["accessories"]})
        if outfit["hair"] and outfit["hair"]["path"]:
            refs["hair"] = outfit["hair"]["path"]
        return {
            "id": time.strftime("%Y%m%d-%H%M%S", time.localtime(now)) + "-" + job.id[:6],
            "created": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)),
            "created_ts": now,
            "prompt": "Try On: " + outfit_text(outfit), "negative": "",
            "seed": s.get("seed"),
            "model": {"id": "try-on", "label": wf.get("label", "Try On"), "file": v["model"],
                      "family": "qwen-image",
                      "files": {var: v.get(var) for var in (wf.get("files") or {})}},
            "loras": loras, "identities": [], "style": None, "preset": "dress",
            "workflow": wf.get("id"), "workflow_label": wf.get("label"),
            "backend": {"id": b["id"], "name": b["name"], "url": b["url"]},
            "sampler": v["sampler"], "scheduler": v["scheduler"], "steps": v["steps"],
            "guidance": None, "width": size[0], "height": size[1], "denoise": 1.0,
            "refine": None, "face_detail": None, "references": refs,
            "warnings": [], "notes": list(job.notes),
            "duration": round(time.time() - job.started, 1),
            "prompt_id": job.prompt_id, "settings": s,
            "graph": job.dress["graphs"][-1], "face_graph": None, "dress": job.dress,
        }

    def _face_pass(self, job, client, plan, values, entry, files, say):
        """The second run of the face pass, on the lane's thread. -> (files,
        graph): the redrawn picture's files and the graph that made them, or
        the first run's files and None when there was nothing to redraw or
        the pass failed - the picture is never lost to its finish."""
        b = job.backend
        found = face_boxes(entry)
        if found is None:
            plan.warnings.append("The face finder said nothing; the picture is as made.")
            return files, None
        width, height, boxes = found
        scene = job.settings.get("scene_faces") or {}
        known = match_faces(width, height, boxes, scene.get("people") or [])
        # A person's own words stand for the whole prompt in their face's
        # redraw, so the style goes with them: without it an SX-70 picture
        # got smooth, grainless faces.
        style = (self.lib.get("styles", job.settings.get("style"))
                 if job.settings.get("style") else None)
        style = " ".join(x for x in (style["trigger"], style["prompt"]) if x) if style else ""
        for person in scene.get("people") or []:
            if person not in known.values():
                plan.notes.append("Face pass: %s's face was not found where the scene puts it%s."
                                  % (person["name"], ", so their face picture was not used"
                                     if person.get("face") else ""))
        pulid, why = self._pulid(client, plan, types=None) if any(
            p.get("face") for p in known.values()) else (None, "")
        if why:
            plan.warnings.append(why)
        likeness = {i for i, p in known.items() if p.get("face") and pulid}
        pairs = indexed_crops(width, height, boxes, keep=likeness)
        crops = [c for _, c in pairs]
        faces = []
        try:
            for i, _ in pairs:
                person = known.get(i)
                if person is None:
                    faces.append(None)
                    continue
                image = client.upload_image(person["face"]) if i in likeness else None
                words = " ".join(x for x in (person.get("words"), style) if x)
                faces.append({"words": words, "image": image,
                              "denoise": scene.get("likeness") if image else None,
                              "name": person["name"]})
        except (ComfyError, OSError) as e:
            plan.warnings.append("A face picture could not be sent (%s); the faces are "
                                 "redrawn from the words alone." % e)
            faces = [dict(f, image=None, denoise=None) if f else None for f in faces]
            faces += [None] * (len(crops) - len(faces))
        job.face = {"found": len(boxes), "redrawn": len(crops),
                    "denoise": values.get("face_denoise"),
                    "people": [f["name"] for f in faces if f],
                    "likeness": [f["name"] for f in faces if f and f.get("image")],
                    "likeness_denoise": scene.get("likeness")}
        if not crops:
            plan.notes.append("Face pass: %s" % ("no face found" if not boxes else
                                                 "every face was already drawn at full size"))
            return files, None
        f = files[0]
        image = "%s%s [%s]" % (f["subfolder"] + "/" if f.get("subfolder") else "",
                                f["filename"], f.get("type") or "output")
        oval = os.path.join(self.lib.root, "face_oval.png")
        try:
            if not os.path.isfile(oval):
                os.makedirs(self.lib.root, exist_ok=True)
                with open(oval, "wb") as fh:
                    fh.write(oval_png())
            graph = face_graph(plan.workflow, values, plan.loras, image, crops,
                               client.upload_image(oval), values["filename_prefix"] + "_faces",
                               faces=faces, pulid_file=pulid,
                               boxes=[boxes[i] for i, _ in pairs])
            say("refining", "redrawing %d face%s at %d px" % (
                len(crops), "" if len(crops) == 1 else "s", FACE_EDIT), None)
            pid = client.queue_workflow(graph)
        except (ComfyError, TemplateError, OSError) as e:
            plan.warnings.append("The face pass could not start (%s); the picture is as made."
                                 % e)
            return files, None
        job.prompt_id = pid
        n = len(crops)

        def on_event(kind, data):
            if kind != "progress" or not data[1]:
                return
            value, total, nid = data
            face = (nid or "").split("_")[0][2:] if (nid or "").startswith("fc") else "?"
            say("refining", "redrawing face %s of %d · step %d of %d" % (face, n, value, total),
                value / float(total))
        watch = client.watch() if hasattr(client, "watch") else None
        try:
            entry2 = client.listen_for_progress(
                pid, on_event, stop=job.cancel.is_set,
                **({"watch": watch} if watch is not None else {}))
        finally:
            if watch is not None:
                watch.close()
        if entry2 is None:
            return files, None                # cancelled; run_job says so
        files2 = outputs_of(entry2)
        if not files2:
            errors = run_errors(entry2, graph)
            plan.warnings.append("The face pass failed (%s); the picture is as made."
                                 % ("; ".join(errors) or "no picture"))
            return files, None
        plan.notes.append("Face pass: %d face%s redrawn at %d px, denoise %s" % (
            n, "" if n == 1 else "s", FACE_EDIT, values.get("face_denoise")))
        drawn = [f for f in faces if f and f.get("image")]
        if drawn:
            plan.notes.append("Likeness: %s drawn from their face picture%s at %s." % (
                ", ".join(f["name"] for f in drawn), "" if len(drawn) == 1 else "s",
                scene.get("likeness")))
        if scene.get("real"):
            job.real_faces = [{"name": p["name"], "box": list(boxes[i]),
                               "photos": p.get("photos") or ([p["face"]] if p.get("face") else [])}
                              for i, p in sorted(known.items())
                              if p.get("photos") or p.get("face")]
        return files2, graph

    # ------------------------------------------------------ Visual Critic
    def _refine(self, job, client, plan, values, files, say):
        """Automatic refinement, on the lane's thread: the vision model looks
        at the picture (studio_critic), what it finds right is left alone,
        and what it finds wrong is redrawn with the tool the fault calls for -
        the face pass's crop-redraw-blend for faces, hands and objects, the
        same at low denoise over the whole picture for a touch-up, and a new
        picture only for a structural failure. Up to `refine_passes` passes,
        stopping as soon as the critic finds nothing meaningful. Never loses
        the picture: any failure keeps the last good one. -> files."""
        s = job.settings
        try:
            vision = self.vision() if self.vision else None
        except Exception:
            vision = None
        if vision is None:
            plan.warnings.append("Automatic refinement needs a vision model on the LLM host "
                                 "and none is served; the picture is as made.")
            return files
        idents = [self.lib.get("identities", x.get("id") if isinstance(x, dict) else x)
                  for x in s.get("identities") or []]
        idents = [i for i in idents if i]
        style = self.lib.get("styles", s.get("style")) if s.get("style") else None
        intent = critic.intent_from(s, plan.prompt)
        canonical = critic.initial_canonical(s, SLOTS, [i["name"] for i in idents],
                                             style["name"] if style else None)
        refs = [plan.references["face"]] if plan.references.get("face") else []
        refs += [i["references"][0] for i in idents if i.get("references")
                 and i["references"][0] not in refs]
        passes = max(0, min(int(s.get("refine_passes") or critic.MAX_PASSES), 6))
        history = [{"pass": 0, "type": "initial_generation", "image": files[0]["filename"]}]
        log, stop = [], "pass limit reached"
        for n in range(1, passes + 1):
            if job.cancel.is_set():
                stop = "cancelled"
                break
            say("refining", "Analyzing result" + ("" if n == 1 else " again") + "...", None)
            try:
                raw = client.fetch(files[0])
                result = critic.analyze_generated_image(vision, raw, intent, canonical, refs)
            except Exception as e:
                plan.warnings.append("The Visual Critic could not read the picture (%s); "
                                     "it is kept as it was." % e)
                stop = "critic failed"
                break
            nxt = critic.plan_next_refinement(result, canonical)
            canonical, promoted = critic.merge_canonical(canonical, nxt["promote"])
            text = critic.log_text(n, result, nxt)
            log.append(text)
            self._critic_log(job, text)
            if not nxt["needs_pass"]:
                stop = "no meaningful problems left"
                break
            self._critic_log(job, critic.build_refinement_instructions(
                intent, canonical, nxt["preserve"], nxt["correct"]))
            done, errors = [], []
            for action in nxt["actions"]:
                if job.cancel.is_set():
                    break
                say("refining", critic.progress_text(action) + "...", None)
                try:
                    got = self._correct(job, client, plan, values, files, raw, action,
                                        intent, canonical, n, say)
                except (ComfyError, TemplateError, OSError, ValueError) as e:
                    errors.append("%s: %s" % (action["type"], e))
                    continue
                if got:
                    files = got
                    done.append(action)
            history.append({"pass": n, "type": " + ".join(a["type"].lower() for a in done)
                            or "none", "changes": [c for a in done for c in a["corrections"]],
                            "promoted": promoted, "errors": errors,
                            "image": files[0]["filename"]})
            for e in errors:
                plan.warnings.append("Refinement pass %d could not run %s" % (n, e))
            if not done:
                stop = "nothing could be corrected"
                break
        job.refinement = {"intent": dict(intent), "canonical": canonical,
                          "history": history, "stopped": stop, "log": log}
        made = [h for h in history[1:] if h["type"] != "none"]
        plan.notes.append("Visual Critic: %d refinement pass%s; stopped: %s." % (
            len(made), "" if len(made) == 1 else "es", stop))
        return files

    def _critic_log(self, job, text):
        """The debug log: image-studio/visual_critic.log, a block per look."""
        try:
            os.makedirs(self.lib.root, exist_ok=True)
            with open(os.path.join(self.lib.root, "visual_critic.log"), "a",
                      encoding="utf-8") as f:
                f.write("[%s job %s]\n%s\n\n" % (time.strftime("%Y-%m-%d %H:%M:%S"),
                                                job.id, text))
        except OSError:
            pass

    def _correct(self, job, client, plan, values, files, raw, action, intent, canonical,
                 n, say):
        """One planned correction on the current picture. -> the new files, or
        None when there was nothing to redraw (no face or hand found)."""
        kind, target, fixes = action["type"], action["target"], action["corrections"]
        b, wf = job.backend, plan.workflow
        if kind == "FULL_REGENERATION":
            return self._regenerate(job, client, critic.generator_prompt(
                intent, canonical, fixes), n, say)
        if not wf.get("face_detail"):
            raise ValueError("the %s workflow has no redraw section" % wf.get("label"))
        f = files[0]
        image = "%s%s [%s]" % (f["subfolder"] + "/" if f.get("subfolder") else "",
                                f["filename"], f.get("type") or "output")
        v = dict(values, seed=(int(values["seed"]) + 1000 * n) % (MAX_SEED + 1))
        if kind == "GLOBAL_REFINEMENT":
            w, h = png_size(raw) or (int(values["width"]), int(values["height"]))
            scale = min(1.0, (1.05e6 / float(w * h)) ** 0.5)
            crops = [{"x": 0, "y": 0, "width": w, "height": h, "mask": False,
                      "edit": (max(256, int(w * scale) // 16 * 16),
                               max(256, int(h * scale) // 16 * 16))}]
            v["face_prompt"] = critic.generator_prompt(intent, canonical, fixes)
        else:
            sam3 = self._sam3_of(b)
            if not sam3:
                raise ValueError("%s has no SAM3 checkpoint to find the %s"
                                 % (b["name"], target))
            found = face_boxes(self._run_quick(client, add_face_finder(
                {"fi": {"class_type": "LoadImage", "inputs": {"image": image}}}, sam3,
                ["fi", 0], ("face:8" if kind == "FACE_CORRECTION" else "%s:4" % target))))
            if not found or not found[2]:
                return None
            w, h, boxes = found
            if kind == "FACE_CORRECTION":
                crops = [head_square(bx, w, h, FACE_PAD) for bx in boxes]
                v["face_prompt"] = FACE_PROMPT % critic.generator_prompt(intent, canonical,
                                                                         fixes)
            else:
                crops = [dict(head_square(bx, w, h, REGION_PAD), head=False)
                         for bx in boxes[:4]]
                v["face_prompt"] = critic.generator_prompt(intent, canonical, fixes,
                                                           focus=target) + (
                    " " + anatomy_text() if target == "hand" else "")
            crops = [c for c in crops if c["width"] >= 64]
            if not crops:
                return None
        v["face_denoise"] = CRITIC_DENOISE[kind] if kind != "FACE_CORRECTION" else max(
            CRITIC_DENOISE[kind], float(values.get("face_denoise") or 0))
        oval = os.path.join(self.lib.root, "face_oval.png")
        if not os.path.isfile(oval):
            os.makedirs(self.lib.root, exist_ok=True)
            with open(oval, "wb") as fh:
                fh.write(oval_png())
        graph = face_graph(wf, v, plan.loras, image, crops, client.upload_image(oval),
                           "%s_pass%d_%s" % (values["filename_prefix"], n, kind.lower()))
        return self._run_pass(job, client, graph, say, critic.progress_text(action))

    def _regenerate(self, job, client, prompt, n, say):
        """A new picture from the compiled prompt and a new seed, for a
        structural failure only. The face pass is left to the critic."""
        b = job.backend
        s = copy.deepcopy(job.settings)
        s.update(prompt_override=prompt, face_detail=False, auto_refine=False,
                 seed=(int(s.get("seed") or 0) + 7919 * n) % (MAX_SEED + 1))
        p = compose(s, self.lib, b, self.inventories.get(b["id"]), self.workflow_loader,
                    self.nodes.get(b["id"]))
        if p.errors:
            raise ValueError(" ".join(p.errors))
        v = dict(p.values)
        for var, path in p.images.items():
            v[var] = client.upload_image(path)
        v["filename_prefix"] = "ImageStudio/%s_%s_regen%d" % (
            p.workflow.get("id", "job"), job.id, n)
        return self._run_pass(job, client, fill(p.workflow, v, p.loras), say,
                              "Regenerating the picture")

    def _sam3_of(self, backend):
        inv = self.inventories.get(backend["id"]) or {}
        sam = sorted(c for c in inv.get("checkpoints") or () if SAM3 in c.lower())
        return sam[0] if sam else None

    def _run_pass(self, job, client, graph, say, label):
        """Run one refinement graph to its end. -> files, or None if cancelled.
        Raises ComfyError when it ends without a picture."""
        job.prompt_id = client.queue_workflow(graph)

        def on_event(kind, data):
            if kind == "progress" and data[1]:
                say("refining", "%s · step %d of %d" % (label, data[0], data[1]),
                    data[0] / float(data[1]))
        watch = client.watch() if hasattr(client, "watch") else None
        try:
            entry = client.listen_for_progress(
                job.prompt_id, on_event, stop=job.cancel.is_set,
                **({"watch": watch} if watch is not None else {}))
        finally:
            if watch is not None:
                watch.close()
        if entry is None:
            return None
        out = outputs_of(entry)
        if not out:
            raise ComfyError("; ".join(run_errors(entry, graph)) or "no picture")
        return out

    def _real_faces(self, job, client, plan, values, files, say):
        """After the face pass, each person's own face over theirs where a
        photo of them fits the drawn head's angle (PASTE_NODE decides, face
        by face, and says why not). -> the files to keep: the pasted picture
        first and the face pass's beside it, or the face pass's alone when
        nothing was pasted or the paste could not run."""
        b = job.backend
        try:
            if PASTE_NODE not in set(client.node_types()):
                plan.notes.append(
                    "Real faces: %s has no %s node, so the faces are PuLID's. Copy "
                    "comfy_nodes/studio_facepaste into its custom_nodes and restart ComfyUI."
                    % (b["name"], PASTE_NODE))
                return files
            local, faces = {}, []
            for p in job.real_faces:
                names = []
                for path in p["photos"]:
                    names.append(client.upload_image(path))
                    local[names[-1]] = path
                faces.append({"name": p["name"], "box": p["box"], "references": names})
            f = files[0]
            image = "%s%s [%s]" % (f["subfolder"] + "/" if f.get("subfolder") else "",
                                    f["filename"], f.get("type") or "output")
            graph = paste_graph(image, faces, values["seed"],
                                values["filename_prefix"] + "_real")
            say("refining", "matching each face to its photos", None)
            pid = client.queue_workflow(graph)
            entry = client.listen_for_progress(pid, lambda *a: None, stop=job.cancel.is_set)
        except (ComfyError, OSError) as e:
            plan.warnings.append("The real faces could not be pasted (%s); the faces are "
                                 "PuLID's." % e)
            return files
        if entry is None:
            return files                      # cancelled; run_job says so
        report, out = paste_report(entry), outputs_of(entry)
        for r in report:
            if r.get("reference") in local:
                r["reference"] = local[r["reference"]]
        job.face = dict(job.face or {}, real=report)
        for r in report:
            if r.get("pasted"):
                plan.notes.append("Real face: %s from %s (%s degrees off, %s allowed)." % (
                    r["name"], os.path.basename(r.get("reference") or "their photo"),
                    r.get("difference"), r.get("tolerance")))
            else:
                plan.notes.append("Real face: %s kept as PuLID drew it - %s." % (
                    r.get("name"), r.get("why") or "no reason given"))
        if not out:
            errors = run_errors(entry, graph)
            plan.warnings.append("The real-face paste failed (%s); the faces are PuLID's."
                                 % ("; ".join(errors) or "no picture"))
            return files
        if not any(r.get("pasted") for r in report):
            return files
        job.paste_graph = graph
        return out + files

    def _faces_into_picture(self, job, client, plan, graph, types, w, h):
        """A scene's face pictures into the picture itself (`add_pulid`),
        each over its person's head. Said, never fatal: without PuLID the
        faces are still drawn to their pictures by the face pass, or from
        the words. -> False only when cancelled."""
        people = [p for p in (job.settings.get("scene_faces") or {}).get("people") or []
                  if p.get("face") and p.get("region")]
        if not people:
            return True
        pulid, why = self._pulid(client, plan, types)
        if not pulid:
            return True                   # the face pass says why, once
        if not w or not h:
            return True
        faces = []
        try:
            folder = os.path.join(self.lib.root, "face_regions")
            os.makedirs(folder, exist_ok=True)
            for person in people:
                if job.cancel.is_set():
                    return False
                data = region_png(person["region"], w, h)
                path = os.path.join(folder, hashlib.sha1(data).hexdigest()[:16] + ".png")
                if not os.path.isfile(path):
                    with open(path, "wb") as f:
                        f.write(data)
                faces.append((client.upload_image(person["face"]), client.upload_image(path)))
        except (ComfyError, OSError) as e:
            plan.warnings.append("The face pictures could not be sent (%s); the picture is "
                                 "drawn without them." % e)
            return True
        add_pulid(graph, pulid, faces)
        plan.notes.append("%s drawn from their face picture%s in the picture itself." % (
            ", ".join(p["name"] for p in people), "" if len(people) == 1 else "s"))
        return True

    def _pulid(self, client, plan, types=None):
        """-> (PuLID weights file, '') when the job's backend can draw a face
        to a picture, else (None, why not)."""
        wf = plan.workflow
        if not set(wf.get("families") or ()) & PULID_FAMILIES:
            return None, ("The %s workflow cannot draw a face to its picture (only FLUX.1 "
                          "can, through PuLID); the faces are redrawn from the words."
                          % wf.get("label", "chosen"))
        try:
            types = types or set(client.node_types())
            if PULID_NODES - types:
                return None, ("%s's ComfyUI lacks PuLID (%s), which draws a face to its "
                              "picture; the faces are redrawn from the words." % (
                                  client.backend["name"], ", ".join(sorted(PULID_NODES - types))))
            info = client.get_json("/object_info/PulidFluxModelLoader")
            files = info["PulidFluxModelLoader"]["input"]["required"]["pulid_file"][0]
        except (ComfyError, KeyError, IndexError, TypeError) as e:
            return None, "PuLID could not be asked about (%s); the faces are redrawn from the words." % e
        if not files:
            return None, ("%s has no PuLID weights (models/pulid); the faces are redrawn from "
                          "the words." % client.backend["name"])
        return files[0], ""

    def _progress(self, job, graph, say):
        """The on_event for one job: ComfyUI's events as Queued -> Loading ->
        Sampling -> Decoding, the node running, its step and percent."""
        wf, b = job.plan.workflow, job.backend
        refine = set((wf.get("stages") or {}).get("refining", ()))
        st = {"node": None, "stepped": set(), "socket": ""}

        def node_name(nid):
            return "node %s %s" % (nid, (graph.get(nid) or {}).get("class_type", "?"))

        def on_event(kind, data):
            if kind == "socket":
                st["socket"] = data
                job.notes.append(data)
            elif kind == "queued":
                if job.status in ("queued", "uploading"):
                    say("queued", "queued on %s, %s" % (
                        b["name"], "next" if data == 0 else "%d ahead" % data), None)
            elif kind == "cached":
                say(detail="reused from the last run: " + ", ".join(
                    node_name(n) for n in data[:4]), progress=job.progress)
            elif kind == "executing":
                st["node"] = data
                if data and data.startswith("fd") and data in graph:
                    say("refining", "finding faces · " + node_name(data), None)
                    return
                if data in refine:
                    say("refining", "refining detail · " + node_name(data), None)
                    return
                stage = stage_of(wf, graph, data)
                if stage == "sampling" and data not in st["stepped"]:
                    say("loading", "loading the model onto the GPU · " + node_name(data), None)
                elif stage in ("loading", "decoding"):
                    say(stage, node_name(data), None if stage == "loading" else 1.0)
                else:
                    say("running", node_name(data), job.progress)
            elif kind == "progress":
                value, total, nid = data
                nid = nid or st["node"]
                st["stepped"].add(nid)
                if total:
                    status = "refining" if nid in refine else "sampling"
                    say(status, "step %d of %d · %d%% · %s" % (
                        value, total, round(100.0 * value / total), node_name(nid)),
                        value / float(total))
            elif kind == "busy":
                say(detail="%s is not answering while it loads models (%ds)" % (b["name"], data),
                    progress=job.progress)
            elif kind == "quiet":
                say(detail="no word from %s for %ds (last: %s). A big VAE decode can take "
                           "this long; if it does not move, ComfyUI may be stuck - Cancel, "
                           "and check its console." % (
                               b["name"], data, node_name(st["node"]) if st["node"] else "?"),
                    progress=job.progress)
        return on_event

    def record_for(self, job, graph):
        p, s, b = job.plan, job.settings, job.backend
        model = self.lib.get("models", s.get("model")) or {}
        style = self.lib.get("styles", s.get("style")) if s.get("style") else None
        v = p.values
        now = time.time()
        idents = []
        for sel in s.get("identities") or []:
            ident = self.lib.get("identities", sel.get("id") if isinstance(sel, dict) else sel)
            if ident:
                idents.append({"id": ident["id"], "name": ident["name"],
                               "trigger": ident["trigger"],
                               "strength": (sel.get("strength") if isinstance(sel, dict)
                                            and sel.get("strength") is not None
                                            else ident["strength"])})
        return {
            "id": time.strftime("%Y%m%d-%H%M%S", time.localtime(now)) + "-" + job.id[:6],
            "created": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)),
            "created_ts": now,
            "prompt": p.prompt, "negative": p.negative, "seed": v.get("seed"),
            "model": {"id": model.get("id"), "label": model.get("label"),
                      "file": v.get("model"), "family": p.family,
                      "files": {var: v.get(var) for var in (p.workflow.get("files") or {})
                                if var in v},
                      "weight_dtype": v.get("weight_dtype")},
            "loras": p.lora_meta,
            "identities": idents,
            "style": ({"id": style["id"], "name": style["name"],
                       "strength": s.get("style_strength")} if style else None),
            "preset": s.get("preset"),
            "workflow": p.workflow.get("id"),
            "workflow_label": p.workflow.get("label"),
            "backend": {"id": b["id"], "name": b["name"], "url": b["url"]},
            "sampler": v.get("sampler"), "scheduler": v.get("scheduler"),
            "steps": v.get("steps"), "guidance": v.get("guidance"),
            "width": v.get("width"), "height": v.get("height"),
            "denoise": v.get("denoise"),
            "refine": ({"upscale": v.get("upscale"), "denoise": v.get("refine_denoise"),
                        "steps": v.get("refine_steps")} if v.get("refine") else None),
            "face_detail": job.face,
            "references": p.references,
            "warnings": p.warnings, "notes": p.notes + job.notes,
            "duration": round(time.time() - job.started, 1),
            "prompt_id": job.prompt_id,
            "settings": s,
            "graph": graph,
            "face_graph": job.face_graph,
            "refinement": job.refinement,
            "paste_graph": job.paste_graph,
            "dress": job.dress,
        }

    def close(self):
        self.queue.close()


def main(argv=None):
    """`python studio_imagegen.py --probe`: every endpoint on every backend,
    then what each model lacks where. Read-only: nothing is queued."""
    import argparse
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--probe", action="store_true", help="check every backend")
    args = ap.parse_args(argv)
    if not args.probe:
        ap.print_help()
        return 0
    studio = Studio()
    bad = 0
    for b in studio.backends():
        print("== %s  %s%s" % (b["name"], b["url"], "" if b["enabled"] else "  (disabled)"))
        for what, ok, detail in studio.client(b).probe():
            bad += not ok
            print("   %-4s %-28s %s" % ("ok" if ok else "FAIL", what, detail))
    studio.check_all()
    for m in studio.lib.all("models"):
        print("== %s" % m["label"])
        for bid, (state, text) in studio.readiness(m).items():
            print("   %-10s %-18s %s" % (state, studio.backend(bid)["name"], text))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
