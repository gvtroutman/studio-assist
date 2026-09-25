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
from studio_comfy_mcp import ComfyError, Unreachable, _explain, outputs_of, status_messages

HERE = os.path.dirname(os.path.abspath(__file__))
WORKFLOWS_DIR = os.path.join(HERE, "comfy_workflows")

MAX_SEED = 2 ** 32 - 1
JOB_TIMEOUT = 1800            # seconds a job may run before it is given up on
HEALTH_TTL = 30               # seconds a health reading is trusted when routing
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
                 "values": {"refine": True, "width": 896, "height": 1152}},
    "hq_final": {"label": "High Quality Final", "role": "hires",
                 "about": "Generate, upscale, then redraw detail at low denoise.",
                 "values": {"refine": True, "upscale": 2.0}},
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
        "notes": _str(d.get("notes")),
    }


CLEAN = {"backends": clean_backend, "models": clean_model, "loras": clean_lora,
         "identities": clean_identity, "styles": clean_style}


def _default_backends():
    return [
        {"id": "5090", "name": "5090 Workstation",
         "url": os.environ.get("IMAGE_STUDIO_5090_URL", "http://127.0.0.1:8188"),
         "roles": ["primary", "flux", "hires", "identity", "interactive", "training"],
         "notes": "This PC. Its GPU is also After Effects' and Resolve's, so ComfyUI "
                  "lets go of VRAM when its queue empties.",
         "max_megapixels": 6.0},
        {"id": "3090", "name": "3090 Server",
         "url": os.environ.get("COMFYUI_URL", "http://100.127.17.38:8188"),
         "roles": ["secondary", "batch", "preprocess", "controlnet", "depth_pose",
                   "caption", "upscale", "background"],
         "notes": "The LLM PC. Shares its 24 GB card with LM Studio: a job here unloads "
                  "LM Studio's models first, and the text encoder runs on the CPU.",
         "shares_llm_gpu": True, "encoder_on_cpu": True, "max_megapixels": 4.2},
    ]


def _default_models():
    # Filenames are the ones ComfyUI's own FLUX templates download; a machine
    # with other names says so in its `backends` entry (the Models editor).
    return [
        {"id": "flux-dev", "label": "FLUX.1 [dev]", "family": "flux1", "workflow": "flux_hq",
         "values": {"model": "flux1-dev.safetensors", "weight_dtype": "default",
                    "clip_l": "clip_l.safetensors", "t5": "t5xxl_fp16.safetensors",
                    "vae": "ae.safetensors",
                    "clip_vision": "sigclip_vision_patch14_384.safetensors",
                    "style_model": "flux1-redux-dev.safetensors"},
         "backends": {"3090": {"t5": "t5xxl_fp8_e4m3fn.safetensors",
                               "weight_dtype": "fp8_e4m3fn"}},
         "defaults": {"steps": 28, "guidance": 3.5, "sampler": "euler",
                      "scheduler": "simple", "width": 1024, "height": 1024},
         "notes": "The main FLUX model. Identity and style LoRAs trained on FLUX.1 apply."},
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
                             % (self.backend["name"], e.code, _explain(detail)))
        except (urllib.error.URLError, OSError) as e:
            reason = getattr(e, "reason", e)
            raise Unreachable("Cannot reach %s at %s (%s). ComfyUI has to be running there, "
                              "started with --listen if it is on another machine."
                              % (self.backend["name"], self.url, reason))

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
        except ComfyError as e:
            return {"ok": False, "detail": str(e), "queue": 0}
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

    def node_types(self):
        if self._nodes is None:
            self._nodes = set(self.get_json("/object_info", timeout=30))
        return self._nodes

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
    def listen_for_progress(self, prompt_id, on_event, stop=None, timeout=JOB_TIMEOUT):
        """Wait for a prompt, reporting as it goes; -> its history entry, or
        None when `stop()` said to stop. `on_event(kind, data)` gets
        ("executing", node_id), ("progress", (value, max, node_id)) and
        ("busy", seconds silent).

        Progress comes over ComfyUI's WebSocket when it will talk; the end is
        read from /history either way, every two seconds, so a socket that
        drops or never opens costs the step counter and nothing else. A
        ComfyUI staging a model stops answering HTTP for half a minute: that
        is "busy", not gone."""
        events = queue.Queue()
        ws = self._open_socket(events)
        deadline = time.monotonic() + timeout
        next_poll, silent = 0.0, None
        try:
            while True:
                if stop is not None and stop():
                    return None
                now = time.monotonic()
                if now >= next_poll:
                    next_poll = now + 2.0
                    try:
                        entry = self.get_history(prompt_id)
                        silent = None
                    except Unreachable:
                        entry = None
                        silent = silent or now
                        on_event("busy", int(now - silent))
                    if entry and (entry.get("status", {}).get("completed")
                                  or entry.get("outputs")
                                  or entry.get("status", {}).get("status_str") == "error"):
                        return entry
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
                if msg.get("type") == "executing" and data.get("node") is not None:
                    on_event("executing", str(data["node"]))
                elif msg.get("type") == "progress":
                    on_event("progress", (data.get("value", 0), data.get("max", 0),
                                          str(data.get("node", ""))))
                elif msg.get("type") in ("execution_success", "execution_error",
                                         "execution_interrupted"):
                    next_poll = 0.0       # read the result now
        finally:
            if ws is not None:
                ws.close()

    def _open_socket(self, events):
        """A reader thread on ComfyUI's WebSocket feeding `events`; -> the
        socket to close, or None when it will not open. Blocking reads on a
        thread of their own, so a frame is never cut by a timeout."""
        try:
            from studio_milanote import WebSocket
            sep = "&" if "?" in self.ws_url else "?"
            ws = WebSocket(self.ws_url + sep + "clientId=" + self.client_id, timeout=5)
            ws.sock.settimeout(None)
        except Exception:
            return None

        def read():
            try:
                while True:
                    text = ws.recv()
                    try:
                        msg = json.loads(text)
                    except ValueError:
                        continue          # a binary preview frame
                    if isinstance(msg, dict):
                        events.put(msg)
            except Exception:
                return                    # closed, by us or by the server

        threading.Thread(target=read, daemon=True).start()
        return ws


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


# ============================================================ composing a job

def default_settings():
    return {"preset": "standard", "model": "flux-dev", "backend": "auto",
            "identities": [], "style": "none", "style_strength": None,
            "scene": "", "negative": "", "loras": [], "references": {},
            "seed": -1, "seed_mode": "random", "steps": None, "guidance": None,
            "sampler": "", "scheduler": "", "width": None, "height": None,
            "denoise": None, "refine": None, "upscale": None, "refine_denoise": None,
            "batch": 1}


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


def compose(settings, lib, backend, inventory=None, workflow_loader=load_workflow):
    """The form's settings -> a Plan for `backend`. `inventory` is that
    backend's {kind: filenames} when known; a file missing from it is an
    error for what the job cannot run without and a warning for what it can.
    Never raises for a user mistake: those land in `plan.errors`."""
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
    scene = s["scene"].strip()
    parts = []
    if who and not all(t in scene for t in who):
        parts.append(" and ".join(t for t in who if t not in scene))
    if scene:
        parts.append(scene)
    if style:
        extra = " ".join(x for x in (style["trigger"], style["prompt"]) if x)
        if extra:
            parts.append(extra)
    for rec, _, why in stack:
        if why == "added" and rec["trigger"] and rec["trigger"] not in " ".join(parts):
            parts.append(rec["trigger"])
    p.prompt = ". ".join(x.rstrip(" .") for x in parts if x) + ("." if parts else "")
    if not p.prompt:
        p.errors.append("Describe the scene, or choose a person.")
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
              "refine", "upscale", "refine_denoise"):
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

    # --------------------------------------------------- files it needs
    if inventory is not None:
        for var, kind in (wf.get("files") or {}).items():
            if any(var in fs for fs in needs.values()):
                continue                  # checked with the reference it serves
            if var in v and v[var] not in inventory.get(kind, set()):
                p.errors.append("%s does not have %s (%s) for %s. Put it in ComfyUI's "
                                "models/%s folder there, or map %s to that machine's "
                                "filename in Models." % (backend["name"], v[var], var,
                                                         model["label"], kind, model["label"]))
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

STATUSES = ("queued", "uploading", "running", "refining", "complete", "failed", "cancelled")
FINISHED = ("complete", "failed", "cancelled")


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


def again(settings):
    """Settings for Generate Again: the same, with a new seed unless the seed
    was chosen by hand."""
    s = copy.deepcopy(settings)
    if s.get("seed_mode", "random") == "random":
        s["seed"] = -1
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
            self.inventories[backend["id"]] = self.client(backend).inventory()
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

    def has_model(self, model_id):
        """backend -> True when the model's main file is known to be there
        (unknown counts as there: routing must not refuse on a guess)."""
        model = self.lib.get("models", model_id)

        def check(b):
            if model is None:
                return False
            r = resolve_model(model, b["id"])
            if r is None:
                return False
            inv = self.inventories.get(b["id"])
            if inv is None:
                return True
            return r[0].get("model") in inv.get("diffusion_models", set()) | inv.get(
                "checkpoints", set())
        return check

    def pick_backends(self, settings, count=1):
        """The backend for each of `count` jobs. A named backend takes them
        all; Auto routes by the preset's role (batch when more than one),
        spreading a batch over every capable backend."""
        if settings.get("backend") not in (None, "", "auto"):
            b = self.backend(settings["backend"])
            if b is None:
                raise ComfyError("No backend called %r." % settings["backend"])
            return [b] * count
        for b in self.backends():
            h = self.health.get(b["id"])
            if b["enabled"] and (h is None or time.time() - h.get("at", 0) > HEALTH_TTL):
                self.check(b)
        role = PRESETS.get(settings.get("preset"), PRESETS["standard"])["role"]
        if count > 1:
            role = "batch" if role == "interactive" else role
        load = self.queue.load()
        picks = []
        for _ in range(count):
            order = route(role, self.backends(), self.health,
                          self.has_model(settings.get("model")), load)
            if not order:
                raise ComfyError(self.why_no_backend(settings))
            if count > 1:                 # least loaded among every capable machine
                order.sort(key=lambda b: load.get(b["id"], 0))
            picks.append(order[0])
            load[order[0]["id"]] = load.get(order[0]["id"], 0) + 1
        return picks

    def why_no_backend(self, settings):
        lines = []
        model = self.lib.get("models", settings.get("model"))
        for b in self.backends():
            h = self.health.get(b["id"]) or {}
            if not b["enabled"]:
                lines.append("%s is disabled" % b["name"])
            elif not h.get("ok"):
                lines.append("%s is offline (%s)" % (b["name"], h.get("detail", "not checked")))
            elif model is not None and not self.has_model(model["id"])(b):
                lines.append("%s does not have %s" % (b["name"], model["label"]))
        return "No backend can take this job: " + "; ".join(lines) + "."

    # ------------------------------------------------------------ submit
    def preview(self, settings, backend=None):
        """compose() against a backend, for the form's warnings; no I/O."""
        b = backend or next((x for x in self.backends() if x["enabled"]), None)
        if b is None:
            return None
        return compose(settings, self.lib, b, self.inventories.get(b["id"]),
                       self.workflow_loader)

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
                       self.workflow_loader)
        job.plan = plan
        if plan.errors:
            return self.queue._finish(job, "failed", " ".join(plan.errors))

        say("uploading", "uploading references" if plan.images else "")
        values = dict(plan.values)
        for var, path in plan.images.items():
            if job.cancel.is_set():
                return self.queue._finish(job, "cancelled")
            values[var] = client.upload_image(path)
        graph = fill(plan.workflow, values, plan.loras)
        try:
            lacking = missing_nodes(graph, client.node_types())
        except ComfyError:
            lacking = []
        if lacking:
            return self.queue._finish(job, "failed", "%s lacks the node(s) %s that the %s "
                                      "workflow uses." % (b["name"], ", ".join(lacking),
                                                          plan.workflow.get("label")))
        if b.get("shares_llm_gpu") and self.make_room is not None:
            say(detail="clearing LM Studio off the GPU")
            try:
                self.make_room(b)
            except Exception as e:
                plan.notes.append("Could not clear the shared GPU (%s); this may be slow." % e)
        if job.cancel.is_set():
            return self.queue._finish(job, "cancelled")
        job.prompt_id = client.queue_workflow(graph)
        say("running", "queued on %s" % b["name"], None)
        refine = set((plan.workflow.get("stages") or {}).get("refining", ()))
        state = {"node": None}

        def on_event(kind, data):
            if kind == "executing":
                state["node"] = data
                if data in refine and job.status != "refining":
                    say("refining", "refining detail", None)
                elif job.status == "running":
                    say(detail="running", progress=job.progress)
            elif kind == "progress":
                value, total, _ = data
                if total:
                    say(detail="step %d of %d" % (value, total), progress=value / float(total))
            elif kind == "busy":
                say(detail="%s is busy (loading models?) %ds" % (b["name"], data),
                    progress=job.progress)

        entry = client.listen_for_progress(job.prompt_id, on_event, stop=job.cancel.is_set)
        if entry is None:
            return self.queue._finish(job, "cancelled")
        errors = status_messages(entry)
        files = [f for f in outputs_of(entry)]
        if not files:
            if any("interrupted" in e for e in errors) or job.cancel.is_set():
                return self.queue._finish(job, "cancelled")
            return self.queue._finish(job, "failed", "; ".join(errors) or
                                      "The workflow finished without a picture.")
        pictures = [(f["filename"], client.fetch(f)) for f in files]
        job.record = self.history.add(self.record_for(job, graph), pictures)
        job.outputs = list(job.record["images"])
        self.queue._finish(job, "complete",
                           "; ".join(plan.warnings[:1]) if plan.warnings else "")

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
                      "file": v.get("model"), "family": p.family},
            "loras": p.lora_meta,
            "identities": idents,
            "style": ({"id": style["id"], "name": style["name"],
                       "strength": s.get("style_strength")} if style else None),
            "preset": s.get("preset"),
            "workflow": p.workflow.get("id"),
            "backend": {"id": b["id"], "name": b["name"], "url": b["url"]},
            "sampler": v.get("sampler"), "scheduler": v.get("scheduler"),
            "steps": v.get("steps"), "guidance": v.get("guidance"),
            "width": v.get("width"), "height": v.get("height"),
            "denoise": v.get("denoise"),
            "refine": ({"upscale": v.get("upscale"), "denoise": v.get("refine_denoise"),
                        "steps": v.get("refine_steps")} if v.get("refine") else None),
            "references": p.references,
            "warnings": p.warnings, "notes": p.notes,
            "duration": round(time.time() - job.started, 1),
            "prompt_id": job.prompt_id,
            "settings": s,
            "graph": graph,
        }

    def close(self):
        self.queue.close()
