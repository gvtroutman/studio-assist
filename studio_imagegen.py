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

import studio_doctor as doctor
from studio_comfy_mcp import (FACE_EDIT, FACE_MIN, FACE_PAD, FACE_PROMPT, SAM3, ComfyError,
                              Unreachable, _explain, head_square, outputs_of, oval_png,
                              status_messages)

HERE = os.path.dirname(os.path.abspath(__file__))
WORKFLOWS_DIR = os.path.join(HERE, "comfy_workflows")
STYLE_EXAMPLES_DIR = os.path.join(HERE, "style_examples")

MAX_SEED = 2 ** 32 - 1
JOB_TIMEOUT = 1800            # seconds a job may run before it is given up on
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
         "identities": clean_identity, "styles": clean_style}


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
         "notes": "Fast photographic model, installed on the 3090 today."},
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
            "identities": list, "styles": _default_styles}


class Library:
    """The five configuration lists, each `<kind>.json` under `studio_dir()`.
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
            stats = self.get_json("/system_stats", timeout=5)
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
    - A node with `"_when": "x"` is kept only when x is set and truthy;
      `"_unless": "x"` the reverse.
    - `switches` name a link chosen by a value: `{"when": "x", "then": link,
      "else": link}`, then used as `"{{name}}"`.
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
        v[name] = sw["then"] if v.get(sw["when"]) else sw["else"]

    kept = {}
    for nid, node in graph.items():
        if "_when" in node and not v.get(node["_when"]):
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
FACE_NODES = {"CheckpointLoaderSimple", "SAM3_Detect", "PreviewAny", "GetImageSize",
              "ImageCropV2", "ImageScale", "ImageToMask", "ImageCompositeMasked", "LoadImage"}


def _is_link(x):
    return isinstance(x, list) and len(x) == 2 and isinstance(x[0], str) and isinstance(x[1], int)


def add_face_finder(graph, sam3):
    """Into a filled graph: SAM3's face boxes and the picture's size for its
    SaveImage's picture, as PreviewAny text (read by `face_boxes`)."""
    save = next(nid for nid, n in graph.items() if n["class_type"] == "SaveImage")
    pixels = graph[save]["inputs"]["images"]
    graph["fd1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": sam3}}
    graph["fd2"] = {"class_type": "CLIPTextEncode", "inputs": {"text": "face:8",
                                                               "clip": ["fd1", 1]}}
    graph["fd3"] = {"class_type": "SAM3_Detect", "inputs": {
        "model": ["fd1", 0], "image": pixels, "conditioning": ["fd2", 0], "threshold": 0.3,
        "refine_iterations": 0, "individual_masks": True}}
    graph["fd4"] = {"class_type": "PreviewAny", "inputs": {"source": ["fd3", 1]}}
    graph["fd5"] = {"class_type": "GetImageSize", "inputs": {"image": pixels}}
    graph["fd6"] = {"class_type": "PreviewAny", "inputs": {"source": ["fd5", 0]}}
    graph["fd7"] = {"class_type": "PreviewAny", "inputs": {"source": ["fd5", 1]}}
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


def face_crops(width, height, boxes, pad=FACE_PAD):
    """The squares to redraw: every face whose padded square is smaller than
    FACE_EDIT (one already that big was drawn at full size)."""
    return [c for c in (head_square(b, width, height, pad) for b in boxes)
            if c["width"] < FACE_EDIT]


def face_graph(wf, values, loras, image, crops, oval, prefix):
    """The second run: `image` (a LoadImage name) with each crop redrawn and
    blended back through `oval`, saved under `prefix`. The model, VAE and
    conditioning come from the template's `face_detail` section, filled like
    the rest (so the LoRA chain is the job's), keeping only the nodes they
    need."""
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
    seed, last = int(values["seed"]), ["fi", 0]
    for i, crop in enumerate(crops):
        n, side = "fc%d_" % (i + 1), crop["width"]
        g[n + "1"] = {"class_type": "ImageCropV2", "inputs": {"image": last, "crop_region": crop}}
        g[n + "2"] = {"class_type": "ImageScale", "inputs": {
            "image": [n + "1", 0], "upscale_method": "lanczos", "width": FACE_EDIT,
            "height": FACE_EDIT, "crop": "disabled"}}
        g[n + "3"] = {"class_type": "VAEEncode", "inputs": {"pixels": [n + "2", 0],
                                                            "vae": links["vae"]}}
        g[n + "4"] = {"class_type": "KSampler", "inputs": {
            "seed": (seed + i + 1) % (MAX_SEED + 1), "steps": values["steps"], "cfg": 1.0,
            "sampler_name": values["sampler"], "scheduler": values["scheduler"],
            "denoise": values["face_denoise"], "model": links["model"],
            "positive": links["positive"], "negative": links["negative"],
            "latent_image": [n + "3", 0]}}
        g[n + "5"] = {"class_type": "VAEDecode", "inputs": {"samples": [n + "4", 0],
                                                            "vae": links["vae"]}}
        g[n + "6"] = {"class_type": "ImageScale", "inputs": {
            "image": [n + "5", 0], "upscale_method": "lanczos", "width": side,
            "height": side, "crop": "disabled"}}
        g[n + "7"] = {"class_type": "ImageScale", "inputs": {
            "image": ["fo", 0], "upscale_method": "bilinear", "width": side,
            "height": side, "crop": "disabled"}}
        g[n + "8"] = {"class_type": "ImageToMask", "inputs": {"image": [n + "7", 0],
                                                              "channel": "red"}}
        g[n + "9"] = {"class_type": "ImageCompositeMasked", "inputs": {
            "destination": last, "source": [n + "6", 0], "x": crop["x"], "y": crop["y"],
            "resize_source": False, "mask": [n + "8", 0]}}
        last = [n + "9", 0]
    g["fs"] = {"class_type": "SaveImage", "inputs": {"images": last, "filename_prefix": prefix}}
    return g


FOLDER_WORDS = {"diffusion_models": "diffusion model", "checkpoints": "checkpoint",
                "text_encoders": "text encoder", "vae": "VAE", "loras": "LoRA",
                "clip_vision": "CLIP vision model", "style_models": "style model",
                "controlnet": "ControlNet", "upscale_models": "upscale model"}


def uses(wf, var):
    """Whether a template does anything with `var` (a node kept or dropped by
    it, or a switch on it) - so a setting it ignores is not recorded as done."""
    return (any(n.get("_when") == var or n.get("_unless") == var
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
    return {"preset": "standard", "model": "flux-dev", "backend": "auto",
            "identities": [], "style": "none", "style_strength": None,
            "scene": "", "subject": "", "hair": "", "eyes": "", "build": "",
            "traits": "", "camera": "", "negative": "", "loras": [], "references": {},
            "seed": -1, "seed_mode": "random", "steps": None, "guidance": None,
            "sampler": "", "scheduler": "", "width": None, "height": None,
            "denoise": None, "refine": None, "upscale": None, "refine_denoise": None,
            "face_detail": None, "batch": 1}


# The person's attributes on the form: (setting, label, noun). A value that
# does not already name its noun gets it: "auburn" -> "auburn hair".
PERSON_FIELDS = [
    ("hair", "Hair", ("hair",)),
    ("eyes", "Eyes", ("eye",)),
    ("build", "Build / weight", ("build", "weight", "figure", "body", "physique",
                                 "lb", "kg", "pound", "kilo")),
    ("traits", "Other", ()),
]


def _field(s, key):
    v = s.get(key)
    return v.strip().strip(",.").strip() if isinstance(v, str) else ""


def person_text(settings):
    """The person as prompt text: who they are, then their attributes.
    "a woman in her 30s, auburn hair, green eyes, slim build"."""
    bits = [_field(settings, "subject")]
    for key, _, nouns in PERSON_FIELDS:
        v = _field(settings, key)
        if v and nouns and not any(n in v.lower() for n in nouns):
            v += " " + nouns[0] + ("s" if nouns[0] == "eye" else "")
        bits.append(v)
    return ", ".join(b for b in bits if b)


def summary(settings):
    """One line for a job row before its prompt is composed."""
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
    parts = [x for x in (", ".join(x for x in (named, person) if x), scene,
                         _field(s, "camera")) if x]
    if style:
        extra = " ".join(x for x in (style["trigger"], style["prompt"]) if x)
        if extra:
            parts.append(extra)
    for rec, _, why in stack:
        if why == "added" and rec["trigger"] and rec["trigger"] not in " ".join(parts):
            parts.append(rec["trigger"])
    p.prompt = ". ".join(x.rstrip(" .") for x in parts if x) + ("." if parts else "")
    if not p.prompt:
        p.errors.append("Describe the scene or the person, or choose a person.")
    p.negative = ", ".join(x for x in (s["negative"].strip(),
                                        style["negative"] if style else "") if x)
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
        short = [f for f in needs.get(var, []) if inventory is not None
                 and v.get(f) not in inventory.get(wf.get("files", {}).get(f, ""), set())]
        if short:
            p.warnings.append("%s reference: %s lacks %s, which it needs; not used."
                              % (label, backend["name"], ", ".join(str(v.get(f)) for f in short)))
            continue
        p.images[var] = path
        p.references[kind] = path
    if "source_image" in p.images and s.get("denoise") in (None, ""):
        v["denoise"] = wf.get("source_denoise", 0.65)

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
            v["face_prompt"] = FACE_PROMPT % p.prompt
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
        self.face = None              # {"found", "redrawn", "denoise"} when it ran
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


# ================================================================ the studio

class Studio:
    """The Image Studio without its window: library, clients, health, the
    queue and history. `make_room(backend)` is the host's hook to clear
    LM Studio off a shared GPU before a job there (Chat._images_make_room)."""

    def __init__(self, root=None, notify=lambda job: None, make_room=None,
                 client_factory=None, workflow_loader=load_workflow):
        self.lib = Library(root)
        self.history = History(os.path.join(self.lib.root, "history"))
        self.client_factory = client_factory or ComfyUIClient   # read late: tests swap it
        self.workflow_loader = workflow_loader
        self.make_room = make_room
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

    def submit(self, settings):
        """Queue the form's settings: one job per picture in the batch, each
        with its own seed so each picture's record says exactly how it was
        made. Routing does network I/O: call off the UI thread. -> [Job]."""
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
        plan = compose(job.settings, self.lib, b, self.inventories.get(b["id"]),
                       self.workflow_loader, self.nodes.get(b["id"]))
        job.plan = plan
        if plan.errors:
            return self.queue._finish(job, "failed", " ".join(plan.errors))

        say("uploading" if plan.images else None,
            "uploading references" if plan.images else "")
        values = dict(plan.values)
        for var, path in plan.images.items():
            if job.cancel.is_set():
                return self.queue._finish(job, "cancelled")
            values[var] = client.upload_image(path)
        # One output name per job, so the file on the backend says which job
        # made it (ComfyUI adds _00001_ and the extension).
        values["filename_prefix"] = "ImageStudio/%s_%s" % (plan.workflow.get("id", "job"),
                                                           job.id)
        try:
            graph = fill(plan.workflow, values, plan.loras)
        except TemplateError as e:
            return self.queue._finish(job, "failed", str(e))
        try:
            types = set(client.node_types())
        except ComfyError:
            types = None
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
        crops = face_crops(width, height, boxes)
        job.face = {"found": len(boxes), "redrawn": len(crops),
                    "denoise": values.get("face_denoise")}
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
                               client.upload_image(oval), values["filename_prefix"] + "_faces")
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
        return files2, graph

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
