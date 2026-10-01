#!/usr/bin/env python3
"""
studio_imagegen - the Image Studio's engine: Person -> Style -> Scene ->
Reference -> Generate, with the ComfyUI graph built underneath.

No tkinter here; `apps/image_studio/ui.py` is the tab, and this module can be
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

if __package__ in (None, ""):  # run as a script: import from the checkout
    import os as _os, sys as _sys
    _sys.path[0] = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))

import base64
import copy
import hashlib
import html
import json
import math
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

import apps.image_studio.critic as critic
import core.doctor as doctor
import apps.image_studio.scene.mannequin as mq
from apps.comfyui.mcp import (FACE_EDIT, FACE_MIN, FACE_PAD, FACE_PROMPT, SAM3, ComfyError,
                              Unreachable, _explain, head_square, outputs_of, oval_png,
                              preview_of, status_messages)

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WORKFLOWS_DIR = os.path.join(HERE, "comfy_workflows")
STYLE_EXAMPLES_DIR = os.path.join(HERE, "style_examples")

MAX_SEED = 2 ** 32 - 1
JOB_TIMEOUT = 1800            # seconds a job may run before it is given up on
POSE_NODE = "StudioDWPoseKeypoints"   # comfy_nodes/studio_dwpose: a photo's pose points
PASTE_NODE = "StudioFacePaste"        # comfy_nodes/studio_facepaste: a person's own face
HEALTH_TTL = 30               # seconds a health reading is trusted when routing
QUIET_AFTER = 120             # seconds without a progress event before a job says so
POLL_EVERY = 2.0              # seconds between looks at /history while a job runs
LOST_AFTER = 3                # answered looks finding a prompt nowhere before it is lost
MODEL_KINDS = ("diffusion_models", "checkpoints", "text_encoders", "vae", "loras",
               "clip_vision", "style_models", "controlnet", "upscale_models", "diffusers",
               "model_patches")


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
    "flux2-klein9b": "FLUX.2 Klein 9B",
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


def preset_info(lib, key):
    """A built-in preset, or a saved LoRA mix (the `presets` library) on top
    of its `base` built-in -> {label, role, about, values, base, loras,
    custom}. Unknown is Standard. A mix's LoRAs are what the form loads
    into its LoRA rows when it is picked; the rows, not the preset, are
    what a picture is made with."""
    if key in PRESETS:
        return dict(PRESETS[key], base=key, loras=[], custom=False)
    rec = lib.get("presets", key) if lib is not None and key else None
    if rec is None:
        return dict(PRESETS["standard"], base="standard", loras=[], custom=False)
    base = PRESETS[rec["base"]]
    names = []
    for sel in rec["loras"]:
        lora = lib.get("loras", sel["id"])
        names.append("%s %.2g" % (lora["name"] if lora else sel["id"] + " (missing)",
                                  sel["strength"]))
    about = rec["about"] or "%s, plus LoRAs: %s." % (
        base["label"], ", ".join(names) if names else "none")
    return dict(base, label=rec["name"], about=about, base=rec["base"],
                loras=[dict(sel) for sel in rec["loras"]], custom=True)


def guess_family(filename):
    n = filename.lower()
    if "kontext" in n:
        return "flux1-kontext"
    if "klein" in n and re.search(r"(^|[^0-9])9b", n):
        return "flux2-klein9b"
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


def model_families(model):
    """Every family a logical model runs as: its own, and any a backend's
    entry overrides it with."""
    fams = {model.get("family") or ""}
    fams.update((over or {}).get("family") or "" for over in model.get("backends", {}).values())
    return {f for f in fams if f}


def lora_fits(lora, model):
    """Does `lora` work with `model` wherever it runs? True, False, or None
    when the LoRA's family is not known. What the form offers and the
    Add-ons window files it under (AGENTS.md "Add-ons")."""
    fams = model_families(model) if model else set()
    if not lora.get("family") or not fams:
        return None
    return any(compatibility(lora["family"], f) for f in fams)


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
        "license": _str(d.get("license")),     # the model's terms, carried by each picture
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
        # Off in Add-ons: kept, but not offered on the form or added as Always on.
        "enabled": d.get("enabled", True) is not False,
        "body_control": _str(d.get("body_control")) if d.get("body_control") in
                        ("chest_female", "chest_male") else "",
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
        "description": _str(d.get("description")),
        "pool_photos": d.get("pool_photos") is True,
        "lora": _str(d.get("lora")),
        "trigger": _str(d.get("trigger")),
        "strength": _num(d.get("strength", 0.85), float, 0.85, -2.0, 2.0),
        # A Klein LoRA of their head (Build LoRA) for the head swap alone; never
        # in a picture's own LoRA stack, which is `lora`'s (`head_lora`).
        "head_lora": _str(d.get("head_lora")),
        "references": _strs(d.get("references")),
        "avatar": _str(d.get("avatar")),
        "face_swap": d.get("face_swap", True) is not False,
        # FaceFusion's swapper weight (studio_facefusion.SWAP_STRENGTH): 0.5 is
        # neutral, 1 strongest.
        "swap_strength": _num(d.get("swap_strength", 0.8), float, 0.8, 0.0, 1.0),
        # "" is studio_facefusion.SWAP_MODEL; a name picks another of SWAP_MODELS.
        "swap_model": _str(d.get("swap_model")),
        "reference_strength": _num(d.get("reference_strength", 0.6), float, 0.6, 0.0, 2.0),
        "use_references": d.get("use_references", True) is not False,
        "notes": _str(d.get("notes")),
        # The Scene Builder's head shape for them (`studio_mannequin.HEAD_SHAPE`).
        "head": mq.clean_head(d.get("head")),
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


def clean_camera_profile(d):
    """A camera in the Scene Builder's Camera body picker: its chemistry (film
    stock or digital colour science, in words) rides into the scene's camera
    words when it is chosen, and its native lens (optional) replaces the
    scene's. `image` is a picture of the camera itself, not a sample photo -
    a camera with none shows its name on a blank tile, as a style with no
    example does."""
    if not isinstance(d, dict) or not _str(d.get("name")):
        return None
    lens = d.get("lens")
    cid = slug(d.get("id") or d["name"])
    # Its frame's shape (`CAMERA_FORMATS`), which the scene's frame follows
    # when it is chosen. A library saved before formats existed takes the
    # starter camera's by id.
    fmt = d.get("format")
    if fmt is None:
        fmt = DEFAULT_CAMERA_FORMATS.get(cid, "")
    fmt = _str(fmt).replace(" ", "").replace("x", ":")
    return {
        "id": cid,
        "name": _str(d["name"]),
        "chemistry": _str(d.get("chemistry")),
        "lens": None if lens in (None, "") else _num(lens, float, None, 10, 300),
        "format": fmt if fmt in CAMERA_FORMATS else "",
        "image": _str(d.get("image")),
        "notes": _str(d.get("notes")),
    }


# A camera's frame shape: square (medium format 6x6, an SX-70) or 3:2 (35mm
# film and full-frame digital), held either way up. "" leaves the frame be.
CAMERA_FORMATS = ("", "1:1", "3:2")
DEFAULT_CAMERA_FORMATS = {"digital-5d": "3:2", "leica-m6": "3:2", "hasselblad-500cm": "1:1",
                          "canon-ae1": "3:2", "sx-70": "1:1", "sony-a7siii": "3:2"}
FORMAT_RATIO = {"1:1": 1.0, "3:2": 1.5}


def chosen_camera(lib, s):
    """The camera the form is set to (`camera_profile`, the deck above the
    Scene field), or None for none."""
    cid = s.get("camera_profile") or ""
    return lib.get("camera_profiles", cid) if cid and cid != "none" else None


def camera_profile_words(cp):
    """A camera's words for the prompt: its chemistry, named when the
    chemistry does not say which camera it is."""
    chem, name = cp["chemistry"].strip(), cp["name"].strip()
    if name and name.lower() not in chem.lower():
        chem = ("Shot on a %s. %s" % (name, chem)).strip()
    return chem


def camera_size(fmt, width, height):
    """(width, height) reshaped to a camera's format at about the same
    area, held the same way (a portrait size stays upright; a square one
    turns landscape for 3:2), in multiples of 64 - the long side from the
    short one, so 3:2 at a megapixel is 1216 x 832, the Scene Builder's frame.
    Unchanged for no format."""
    ratio = FORMAT_RATIO.get(fmt)
    if not ratio or not width or not height:
        return width, height
    area = float(width) * float(height)
    short = int(round((area / ratio) ** 0.5 / 64)) * 64
    long_ = int(short * ratio // 64) * 64
    return (short, long_) if height > width else (long_, short)


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
        "faces": _strs(d.get("faces")),
        "notes": _str(d.get("notes")),
    }


def character_faces(rec, lib=None):
    """Every photo of a character's face on this PC: its own face photos,
    then (when it has none) its identity's reference photos. The first is
    the one PuLID draws from; the real-face paste chooses among them all."""
    if not rec:
        return []
    out = [p for p in rec.get("faces") or [] if os.path.isfile(p)]
    if out or lib is None or not rec.get("identity"):
        return out
    ident = lib.get("identities", rec["identity"])
    if ident and ident.get("use_references", True):
        out = [p for p in ident.get("references") or [] if os.path.isfile(p)]
    return out


def clean_outfit_preset(d):
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


def clean_preset(d):
    """A saved LoRA mix shown in the form's Preset row: a name, the built-in
    preset it rides on (`base`: refine, face pass, size, routing) and the
    LoRAs with strengths it loads into the form."""
    if not isinstance(d, dict) or not _str(d.get("name")):
        return None
    loras, seen = [], set()
    for sel in d.get("loras") if isinstance(d.get("loras"), list) else ():
        if isinstance(sel, dict) and _str(sel.get("id")) and sel["id"] not in seen:
            seen.add(sel["id"])
            loras.append({"id": _str(sel["id"]),
                          "strength": _num(sel.get("strength", 0.8), float, 0.8, -2.0, 2.0)})
    base = _str(d.get("base"))
    rid = slug(d.get("id") or d["name"])
    return {
        "id": "mix-" + rid if rid in PRESETS else rid,     # never a built-in's key
        "name": _str(d["name"]),
        "base": base if base in PRESETS else "standard",
        "loras": loras,
        "about": _str(d.get("about")),
    }


# Thumbnails for pictures Tk cannot read (it reads PNG and GIF, and face
# photos are mostly JPEG): one PowerShell run over them all, System.Drawing
# turning each the way its EXIF says (phones store portraits on their side)
# and saving a PNG beside it. Stdlib Python has no JPEG decoder.
THUMB_SIDE = 96
_THUMB_PS = r"""
Add-Type -AssemblyName System.Drawing
foreach ($pair in $input) {
  $src, $dst = $pair -split '\|', 2
  try {
    $img = [System.Drawing.Image]::FromFile($src)
    if ($img.PropertyIdList -contains 274) {
      $o = [int]$img.GetPropertyItem(274).Value[0]
      $turn = @{3='Rotate180FlipNone'; 6='Rotate90FlipNone'; 8='Rotate270FlipNone'}[$o]
      if ($turn) { $img.RotateFlip($turn) }
    }
    $k = [Math]::Min(1.0, SIDE / [Math]::Max($img.Width, $img.Height))
    $w = [Math]::Max(1, [int]($img.Width * $k)); $h = [Math]::Max(1, [int]($img.Height * $k))
    $bmp = New-Object System.Drawing.Bitmap $w, $h
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.InterpolationMode = 'HighQualityBicubic'
    $g.DrawImage($img, 0, 0, $w, $h)
    $bmp.Save($dst, [System.Drawing.Imaging.ImageFormat]::Png)
    $g.Dispose(); $bmp.Dispose(); $img.Dispose()
  } catch { }
}
"""


def thumb_path(path):
    return os.path.splitext(path)[0] + ".thumb.png"


def thumbnails(paths, side=THUMB_SIDE):
    """{path: its PNG thumbnail, or None when none could be made}, making
    the missing ones in one PowerShell run. Blocks for it (about a second):
    call it off the UI thread."""
    import subprocess
    import core.procs as studio_procs
    todo = [p for p in paths if os.path.isfile(p) and not os.path.isfile(thumb_path(p))]
    if todo and os.name == "nt":
        try:
            child = studio_procs.spawn(
                ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
                 _THUMB_PS.replace("SIDE", str(int(side)))],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=studio_procs.NO_WINDOW)
            try:
                child.proc.communicate("\n".join("%s|%s" % (p, thumb_path(p)) for p in todo)
                                       .encode("utf-8"), timeout=60)
            finally:
                child.kill()
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return {p: (thumb_path(p) if os.path.isfile(thumb_path(p)) else None) for p in paths}


def style_example(style):
    """The picture that shows what `style` looks like: its own `example` when
    that file is there, else the one shipped for its id (the same cat photo
    through each default style), else None."""
    for path in (style.get("example"),
                 os.path.join(STYLE_EXAMPLES_DIR, (style.get("id") or "") + ".png")):
        if path and os.path.isfile(path):
            return path
    return None


def clean_image(d):
    if not isinstance(d, dict) or not _str(d.get("path")):
        return None
    name = _str(d.get("name")) or os.path.basename(d["path"])
    return {"id": slug(d.get("id") or name), "name": name,
            "path": _str(d["path"])}


CLEAN = {"images": clean_image, "backends": clean_backend, "models": clean_model, "loras": clean_lora,
         "identities": clean_identity, "styles": clean_style, "camera_profiles": clean_camera_profile,
         "characters": clean_character, "outfits": clean_outfit_preset, "presets": clean_preset}


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


KLEIN_LICENSE = ("Made with FLUX.2 [klein] 9B, under the FLUX Non-Commercial License: "
                 "not for commercial use.")


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
        {"id": "withanyone", "label": "Family photo (WithAnyone)", "family": "flux1",
         "workflow": "withanyone",
         "values": {"model": "flux1-dev.safetensors", "identity_model": "withanyone.safetensors",
                    "siglip": "siglip-base-patch16-256-i18n", "clip_l": "clip_l.safetensors",
                    "t5": "t5xxl_fp16.safetensors", "vae": "ae.safetensors"},
         "backends": {"3090": None},
         "defaults": {"steps": 25, "guidance": 4.0, "width": 1024, "height": 1024},
         "notes": "One to four people, one clear reference photo each. Scene Builder supplies "
                  "face positions; words describe poses and clothes. Requires the "
                  "StudioWithAnyone node and weights; initially enabled for the 32 GB 5090."},
        {"id": "z-image-turbo", "label": "Z-Image Turbo", "family": "z-image",
         "workflow": "zimage_hq",
         "values": {"model": "z_image_turbo_bf16.safetensors",
                    "encoder": "qwen_3_4b.safetensors", "vae": "ae.safetensors"},
         "defaults": {"steps": 8, "guidance": 1.0, "sampler": "res_multistep",
                      "scheduler": "simple", "width": 1024, "height": 1024},
         "notes": "Fast photographic model and the default: the 5090 when it has the "
                  "files, else the 3090."},
        {"id": "klein-9b", "label": "FLUX.2 Klein 9B", "family": "flux2-klein9b",
         "workflow": "klein9b_base",
         "values": {"model": "flux-2-klein-base-9b.safetensors",
                    "encoder": "qwen_3_8b_fp8mixed.safetensors", "vae": "flux2-vae.safetensors"},
         "backends": {"3090": None},
         "defaults": {"steps": 50, "guidance": 4.0, "sampler": "euler",
                      "width": 1024, "height": 1024},
         "license": KLEIN_LICENSE,
         "notes": "The undistilled Klein 9B a Build LoRA head LoRA is trained on, so a "
                  "person's LoRA shows here. About 40 s a picture on the 5090. "
                  "Non-commercial."},
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


def _default_camera_profiles():
    # Chemistry, not example photos: a camera is named and shot on real film
    # or a real sensor, so its own words say what that does to a picture.
    return [
        {"id": "none", "name": "No camera set", "chemistry": ""},
        {"id": "digital-5d", "name": "Canon 5D Mark IV",
         "chemistry": "Shot on a Canon 5D Mark IV: clean full-frame digital colour "
                      "science, natural skin tones, moderate dynamic range, minimal "
                      "noise, crisp fine detail.",
         "lens": 50.0},
        {"id": "leica-m6", "name": "Leica M6",
         "chemistry": "Shot on a Leica M6, a 35mm rangefinder loaded with Kodak Portra "
                      "400 colour negative film: warm, creamy skin tones, fine grain, "
                      "gentle highlight roll-off.",
         "lens": 35.0},
        {"id": "hasselblad-500cm", "name": "Hasselblad 500C/M",
         "chemistry": "Shot on a Hasselblad 500C/M medium format camera loaded with "
                      "Kodak Portra 160: ultra-smooth tonal gradation, shallow depth of "
                      "field, fine grain.",
         "lens": 50.0},
        {"id": "canon-ae1", "name": "Canon AE-1",
         "chemistry": "Shot on a Canon AE-1, 35mm black and white on Kodak Tri-X: deep "
                      "blacks, visible grain, high contrast, strong directional light.",
         "lens": 50.0},
        {"id": "sx-70", "name": "Polaroid SX-70",
         "chemistry": "Shot on a Polaroid SX-70 instant camera: soft focus, warm faded "
                      "colour, lifted blacks, gentle vignetting, square format.",
         "lens": 35.0},
        {"id": "sony-a7siii", "name": "Sony a7S III",
         "chemistry": "Shot on a Sony a7S III: low-light video-grade digital sensor, "
                      "clean high ISO, slightly cool colour science, smooth shadow "
                      "detail.",
         "lens": 35.0},
    ]


DEFAULTS = {"images": list, "backends": _default_backends, "models": _default_models, "loras": list,
            "identities": list, "styles": _default_styles, "camera_profiles": _default_camera_profiles,
            "characters": list, "outfits": _default_outfits, "presets": list}


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
        """Write `<kind>.json` atomically, then publish it in memory - never
        the other way round, or a failed write would leave `self.data[kind]`
        claiming records that are not actually on disk."""
        data = self.data[kind]
        if records is not None:
            cleaned, seen = [], set()
            for d in records:
                rec = CLEAN[kind](d)
                if rec is not None:
                    rec["id"] = unique_id(rec["id"], seen)
                    seen.add(rec["id"])
                    cleaned.append(rec)
            data = cleaned
        os.makedirs(self.root, exist_ok=True)
        path = self._path(kind)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
        self.data[kind] = data

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
        with open(path, "rb") as f:
            data = f.read()
        return self.keep_bytes(data, os.path.splitext(path)[1].lower(), owner)

    def import_identity_photos(self, paths, owner, existing=()):
        """Copy a curated set, counting duplicate bytes only once, in input order.

        File failures are reported individually so one bad photo cannot hide the
        rest. This performs disk IO and belongs on the editor's worker thread.
        """
        seen, added, errors = set(), [], []
        duplicates = 0
        for path in existing:
            try:
                with open(path, "rb") as f:
                    seen.add(hashlib.sha256(f.read()).digest())
            except OSError:
                pass
        for path in paths:
            try:
                with open(path, "rb") as f:
                    data = f.read()
                ext = picture_ext(data)
                if not ext:
                    raise ValueError("Not a supported image (PNG, JPEG, WebP, GIF or BMP).")
                digest = hashlib.sha256(data).digest()
                if digest in seen:
                    duplicates += 1
                    continue
                kept = self.keep_bytes(data, ext, owner)
                seen.add(digest)
                added.append(kept)
            except (OSError, ValueError) as exc:
                errors.append("%s: %s" % (os.path.basename(path), exc))
        return {"added": added, "duplicates": duplicates, "errors": errors}

    def import_image(self, path):
        """Keep a reusable picture; deduplicate by content, preserving its name.
        Only publish the new list in memory after it has been saved to disk.
        """
        with open(path, "rb") as f:
            data = f.read()
        ext = picture_ext(data)
        if not ext:
            raise ValueError("Choose a PNG, JPEG, WebP, GIF or BMP image.")
        kept = self.keep_bytes(data, ext, "image-library")
        return self.register_image(kept, os.path.basename(path))

    def register_image(self, kept, name):
        """Publish an already copied image (the UI copies on its worker)."""
        existing = next((r for r in self.all("images") if r["path"] == kept), None)
        if existing:
            return existing
        record = clean_image({"name": name, "path": kept})
        record["id"] = unique_id(record["id"], {r["id"] for r in self.all("images")})
        self.save_images(self.all("images") + [record])
        return record

    def save_images(self, records):
        self.save("images", records)

    def keep_bytes(self, data, ext, owner):
        """A picture's bytes under references/<owner>, named by their hash, so
        the same picture twice is one file. -> its path."""
        folder = os.path.join(self.root, "references", slug(owner))
        os.makedirs(folder, exist_ok=True)
        dest = os.path.join(folder, hashlib.sha1(data).hexdigest()[:16] + ext)
        if not os.path.exists(dest):
            with open(dest, "wb") as f:
                f.write(data)
        return dest

    def keep_link(self, url, owner, opener=None):
        """A picture from the web kept as `keep_reference` keeps a file: the
        link is downloaded (`fetch_picture`), and from then on the picture is
        a file under references/, so a link that dies later breaks nothing.
        -> its path; LinkError says in words why there is none."""
        data, ext = fetch_picture(url, opener)
        return self.keep_bytes(data, ext, owner)

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


# ============================================================ pictures from links

PICTURE_BYTES = 40 * 1024 * 1024      # a bigger "picture" is not one worth keeping
PICTURE_TIMEOUT = 30
# A browser's name: a share of image hosts and shops answer anything else with
# 403 or a bot check (the research bridge's search found the same).
PICTURE_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                 "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
# Where a page names its own picture, most telling first: a shop's product,
# a profile's photo, an article's lead picture.
PAGE_PICTURE = ("og:image:secure_url", "og:image:url", "og:image", "twitter:image",
                "twitter:image:src")


class LinkError(ValueError):
    """A link that gave no picture, said in words."""


def picture_ext(data):
    """The extension a picture's own first bytes say it is, or "" for
    anything that is not a picture ComfyUI will read."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return ".gif"
    if data[:2] == b"BM":
        return ".bmp"
    return ""


def picture_link(url):
    """A pasted link made one to fetch: http(s) (https when none is said),
    no credentials in it, and a Google Images result taken to the picture it
    shows. A data: link is returned as it is. -> url, or LinkError."""
    url = (url or "").strip().strip("<>\"'")
    if not url:
        raise LinkError("Paste a link to a picture first.")
    if url[:5].lower() == "data:":
        return url
    if "://" not in url:
        url = "https://" + url.lstrip("/")
    parts = urllib.parse.urlsplit(url)
    if parts.scheme.lower() not in ("http", "https"):
        raise LinkError("Only http and https links are downloaded, not %s." % parts.scheme)
    if "@" in parts.netloc:
        raise LinkError("A link with a name and password in it is not downloaded.")
    if not parts.hostname:
        raise LinkError("%s has no website in it." % url)
    shown = urllib.parse.parse_qs(parts.query).get("imgurl")
    if shown and "google." in parts.hostname and parts.path.endswith("/imgres"):
        return picture_link(shown[0])
    return url


def page_picture(text, base):
    """The picture a web page names as its own (`PAGE_PICTURE`, from its
    <meta> tags), as a whole link, or "". A shop's or a profile's page link
    is what most people copy, not the picture's."""
    found = {}
    for tag in re.findall(r"<meta\b[^>]*>", text, re.I):
        attrs = {k.lower(): v for k, _, v in re.findall(
            r"""([\w:-]+)\s*=\s*(["'])(.*?)\2""", tag, re.S)}
        key = (attrs.get("property") or attrs.get("name") or "").lower()
        if key in PAGE_PICTURE and attrs.get("content") and key not in found:
            found[key] = attrs["content"]
    for key in PAGE_PICTURE:
        if key in found:
            return urllib.parse.urljoin(base, html.unescape(found[key]).strip())
    return ""


def fetch_picture(url, opener=None, _page=True):
    """A picture's bytes from a link -> (bytes, extension). The link may be
    the picture's, a page that names one (`page_picture`, followed once), a
    Google Images result or a data: link. What comes back must be a picture
    by its own bytes (`picture_ext`), not by what the server says; anything
    else is a LinkError that says what came instead."""
    url = picture_link(url)
    ctype, final, host = "", url, urllib.parse.urlsplit(url).hostname
    if url[:5].lower() == "data:":
        head, _, body = url.partition(",")
        try:
            data = (base64.b64decode(body) if head.lower().endswith(";base64")
                    else urllib.parse.unquote_to_bytes(body))
        except ValueError:
            raise LinkError("That data: link is not a picture.")
    else:
        req = urllib.request.Request(url, headers={
            "User-Agent": PICTURE_AGENT, "Accept-Language": "en",
            "Accept": "image/png,image/jpeg,image/webp,image/*;q=0.9,text/html;q=0.5,"
                      "*/*;q=0.3"})
        try:
            with (opener or urllib.request.urlopen)(req, timeout=PICTURE_TIMEOUT) as r:
                data = r.read(PICTURE_BYTES + 1)
                final = r.geturl() if hasattr(r, "geturl") else url
                ctype = (r.headers.get("Content-Type") or "") if hasattr(r, "headers") else ""
        except urllib.error.HTTPError as e:
            raise LinkError("%s answered HTTP %d (%s)%s." % (
                host, e.code, e.reason,
                "; the site may not let pictures be downloaded, so save the picture "
                "and upload it instead" if e.code in (401, 403) else ""))
        except (urllib.error.URLError, OSError) as e:
            reason = getattr(e, "reason", e)
            if isinstance(reason, TimeoutError):
                raise LinkError("%s did not answer within %d seconds."
                                % (host, PICTURE_TIMEOUT))
            raise LinkError("Could not reach %s: %s" % (host, reason))
        if len(data) > PICTURE_BYTES:
            raise LinkError("That link is over %d MB; it is not a picture to keep."
                            % (PICTURE_BYTES // (1024 * 1024)))
    ext = picture_ext(data)
    if ext:
        return data, ext
    if _page and url[:5].lower() != "data:" and (
            "html" in ctype.lower() or data.lstrip()[:1] == b"<"):
        inner = page_picture(data.decode("utf-8", errors="replace"), final)
        if inner:
            return fetch_picture(inner, opener, _page=False)
        raise LinkError("That link is a web page with no picture of its own named in "
                        "it. Right-click the picture and copy its image address.")
    if data[4:12] in (b"ftypavif", b"ftypheic", b"ftypmif1", b"ftypheix"):
        raise LinkError("That picture is AVIF or HEIC, which ComfyUI cannot read. Save "
                        "it as PNG or JPEG and upload that.")
    raise LinkError("That link did not give a picture (PNG, JPEG, WebP, GIF or BMP)%s."
                    % ("; it gave %s" % ctype.split(";")[0] if ctype else ""))


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
            body = r.read().decode("utf-8")
        return self._parse(body)

    def post_json(self, path, payload, timeout=None):
        req = urllib.request.Request(
            self.url + path, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        with self._open(req, timeout) as r:
            body = r.read().decode("utf-8")
        return self._parse(body) if body.strip() else {}

    def _parse(self, body):
        """A 200 with a body that is not valid JSON - a proxy's HTML error
        page, or ComfyUI cut off mid-restart - is the same kind of failure
        as an HTTP error: a ComfyError, not a raw ValueError callers do not
        expect."""
        try:
            return json.loads(body)
        except ValueError as e:
            raise ComfyError("%s answered with something that is not JSON (%s): %s"
                             % (self.backend["name"], e, body[:200]))

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
        # ComfyUI's /models lists files, not Diffusers directories. The
        # WithAnyone node validates complete SigLIP folders and exposes them
        # as its actual input choices.
        try:
            info = self.get_json("/object_info/StudioWithAnyone")
            choices = info.get("StudioWithAnyone", {}).get("input", {}).get(
                "required", {}).get("siglip", [[]])[0]
            out["diffusers"].update(choices if isinstance(choices, list) else [])
        except ComfyError:
            pass
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
            res = self._parse(r.read().decode("utf-8"))
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
        stops answering HTTP for half a minute: that is "busy", not gone.

        A prompt that ComfyUI answers for but has nowhere - not in its history,
        not queued, not running - was lost in a crash or restart: after
        LOST_AFTER such looks that is a ComfyError, not a wait to the deadline.
        At the deadline the prompt is stopped there, so the next job does not
        queue behind it and VRAM can be let go."""
        own = watch is None
        if own:
            watch = self.watch()
        if watch.error:
            on_event("socket", watch.error)
        events = watch.events
        deadline = time.monotonic() + timeout
        next_poll, silent, started, missing = 0.0, None, False, 0
        heard = time.monotonic()      # the last event, for ("quiet", seconds)
        try:
            while True:
                if stop is not None and stop():
                    return None
                now = time.monotonic()
                if now >= next_poll:
                    next_poll = now + POLL_EVERY
                    try:
                        entry = self.get_history(prompt_id)
                        if not entry:
                            # Two reads, not one: a prompt that ends between
                            # them is in neither once, so only LOST_AFTER in a
                            # row count.
                            ahead = self.position(prompt_id)
                            missing = missing + 1 if ahead is None else 0
                            if ahead is not None and ahead < 0:
                                started = True
                            elif ahead is not None and not started:
                                on_event("queued", ahead)
                        silent = None
                    except (Unreachable, ComfyError):
                        # Not just a dropped connection: a malformed body from
                        # a server mid-restart, or a history/queue read that
                        # briefly errors, reads the same as busy - the poll
                        # two seconds later either recovers or the deadline
                        # above ends the job cleanly either way.
                        entry = None
                        silent = silent or now
                        on_event("busy", int(now - silent))
                    status = (entry or {}).get("status") or {}
                    if entry and (status.get("completed") or entry.get("outputs")
                                  or status.get("status_str") == "error"):
                        return entry
                    if missing >= LOST_AFTER:
                        raise ComfyError(
                            "%s no longer has prompt %s: it is not queued, running or "
                            "finished there, so ComfyUI restarted or dropped it. Its "
                            "console says why; run the job again once it is back."
                            % (self.backend["name"], prompt_id))
                    if started and not watch.error and now - heard > QUIET_AFTER:
                        on_event("quiet", int(now - heard))
                if now > deadline:
                    try:
                        stopped = self.cancel_job(prompt_id)
                    except ComfyError:
                        stopped = False
                    raise ComfyError("%s has not finished prompt %s after %d s; %s." % (
                        self.backend["name"], prompt_id, timeout,
                        "it was stopped there" if stopped else
                        "it could not be stopped there and may still be running"))
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
            from apps.milanote.milanote import WebSocket
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
            except TemplateError as e:
                # Left out of the menu, but never silently: a hand edit's
                # stray comma would otherwise just make a preset vanish.
                doctor.log_error("Image Studio: %s" % e)
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
    if wf.get("multi_identity"):
        sampler = graph[wf["multi_identity"]["sampler"]]["inputs"]
        if v.get("identity_pooling"):
            graph[wf["multi_identity"]["sampler"]]["class_type"] = "StudioWithAnyonePooled"
        for index in range(2, wf["multi_identity"]["max_people"] + 1):
            key = "face%d" % index
            if v.get(key):
                graph[key] = {"class_type": "LoadImage", "inputs": {"image": v[key]}}
                sampler[key] = [key, 0]
        for index, keys in enumerate(v.get("identity_reference_groups") or [], 1):
            previous = None
            for key in keys[1:]:
                graph[key] = {"class_type": "LoadImage", "inputs": {"image": v[key]}}
                nid = key + "_group"
                inputs = {"image": [key, 0]}
                if previous:
                    inputs["previous"] = previous
                graph[nid] = {"class_type": "StudioWithAnyoneReferences", "inputs": inputs}
                previous = [nid, 0]
            if previous:
                sampler["references%d" % index] = previous
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
    regional = wf.get("regional_conditioning")
    if regional and v.get("character_regions"):
        base_clip = regional["base_clip"]
        clip_in = graph[base_clip]["inputs"]["clip"]
        combined = [base_clip, 0]                  # the base text covers the whole frame
        for i, region in enumerate(v["character_regions"], 1):
            img, mask, clip_node, cond = ("char%dmaskimg" % i, "char%dmask" % i,
                                          "char%dclip" % i, "char%dcond" % i)
            graph[img] = {"class_type": "LoadImage",
                          "inputs": {"image": "{{%s}}" % region["mask_var"]}}
            graph[mask] = {"class_type": "ImageToMask",
                           "inputs": {"image": [img, 0], "channel": "red"}}
            graph[clip_node] = {"class_type": "CLIPTextEncode",
                                "inputs": {"clip": clip_in, "text": region["prompt"]}}
            graph[cond] = {"class_type": "ConditioningSetMask", "inputs": {
                "conditioning": [clip_node, 0], "mask": [mask, 0],
                "strength": 1.0, "set_cond_area": "default"}}
            combine = "charcombine%d" % i
            graph[combine] = {"class_type": "ConditioningCombine", "inputs": {
                "conditioning_1": combined, "conditioning_2": [cond, 0]}}
            combined = [combine, 0]
        for nid, key in regional["positive_in"]:
            graph[nid]["inputs"][key] = combined
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
FACE_DEPTH_STRENGTH = 0.5      # a face's own head depth map, in its redraw (`face_graph`)
FACE_DEPTH_END = 0.5           # ... over the first half of the steps: the shape, not the skin


# The Visual Critic's redraws (Studio._refine). A hand at 0.6 kept its shape
# in the reverted hand pass of 2026-09-25, and 0.85-0.9 left double hands, so
# a local fix stays at 0.6: it mends fingers, it does not re-pose them.
CRITIC_DENOISE = {"FACE_CORRECTION": 0.45, "LOCAL_INPAINT": 0.6,
                  "OBJECT_CORRECTION": 0.6, "GLOBAL_REFINEMENT": 0.2}
# How far a redraw may rise when the one before left the fault there
# (critic.harder). A hand's does not: past 0.6 it doubles, so its second try
# is a new seed and the critic's newer words, no more.
CRITIC_DENOISE_TOP = {"FACE_CORRECTION": 0.6, "LOCAL_INPAINT": 0.75,
                      "OBJECT_CORRECTION": 0.75, "GLOBAL_REFINEMENT": 0.3, "hand": 0.6}
REGION_PAD = 1.6


# Fix a spot: the user clicks the parts of a finished picture to redraw (a
# hand, most often). Each spot is a square round the click, redrawn by the
# face pass's crop-redraw-blend on the picture's own model and LoRAs, and
# blended back through the oval; everything outside the squares is kept
# pixel for pixel. The squares are the user's, so no SAM3 is needed.
FIX_TARGETS = {"hand": "a natural human hand with five fingers, clear knuckles and nails",
               "face": "a natural face with clear eyes and teeth",
               "other": ""}
FIX_STRENGTHS = {"light": 0.45, "medium": 0.65, "strong": 0.85}
# Only the thing fixed: the whole picture's prompt ("a woman, whole figure
# in view, dancing") redrew a glasses crop as a tiny dancer (2026-09-26).
FIX_PROMPT = "Close-up photo detail of %s, sharp and natural, matching the light and " \
             "colour around it."
FIX_AREA_GROW = 0.35                  # of its size Find's box grows: the most redrawn
FIX_SHAPE_GROW = 24                   # px at FACE_EDIT the found thing's own outline grows
FIX_AREA_SOFT = (15, 5.0)             # ImageBlur radius and sigma of its edge, at FACE_EDIT
FIX_MIN = 64                          # px: a smaller square is not worth redrawing
FIX_MAX_SPOTS = 8
FIX_NOTE_MAX = 200                    # characters of a note on a spot
# One-click Find: what SAM3 is asked for each kind, and how far round what it
# finds the square reaches. An accessory is not one thing to SAM3, so it is
# asked for each; a face square is padded like the face pass's.
FIX_FIND = {"hand": ["hand:8"], "face": ["face:8"],
            "other": ["glasses:4", "hat:4", "necklace:4", "earring:8", "bracelet:4",
                      "watch:4"]}
# Live on a dirndl (2026-09-26): "bag" came back as the apron and bodice, and
# one necklace as three boxes. An accessory bigger than this share of the
# picture is not one, and a box mostly inside a kept one of the same word
# is the same thing again.
FIX_FIND_MAX = {"other": 0.06, "hand": 0.12}
FIX_FIND_SAME = 0.6
FIX_FIND_PAD = {"hand": 1.5, "face": FACE_PAD, "other": 1.4}
FIX_FACE_MASK = "face"                # SAM3's word for a face fix's true blend mask
FIX_CONTEXT = 1.5                     # the crop redrawn is this x the spot: the photo round it
FIX_TONE = 0.85                       # how far the redraw's colour curves go to the original's
TONE_NODE = "StudioMatchTone"         # comfy_nodes/studio_matchtone
# A spot with a photo is not redrawn by the picture's model but swapped by
# Qwen-Image-Edit 2509 (qwen_dress.json's loaders): the crop as picture 1,
# the photo as picture 2, in the sentence shape Qwen follows ("keep X the
# same" clauses it ignores; see Try On). It works on a picture from any
# model. Its colours are not matched to the picture's: that would turn the
# photo's red hat into the old one's colour.
SWAP_PROMPTS = {"hand": "The hand in picture 1 is posed and looks like the %s in picture 2.",
                "face": "The person in picture 1 has the %s from picture 2.",
                "other": "The person in picture 1 wears the %s from picture 2, in place of "
                         "what is there."}
SWAP_NOUNS = {"hand": "hand", "face": "face", "other": "item"}
SWAP_GROW = 16                        # px the swapped thing's outline grows at the crop's size
# A fix can end with a face swap: an identity's face (its first reference
# picture) put on the picture's biggest face (found again after the spots
# are done) by the same Qwen swap.
FACE_SWAP_FIND = "face:8"
FACE_SWAP_LABEL = "Swapping in %s's face"
SWAP_SLOTS = 2                        # pictures Qwen 2509 takes beside the one it edits
# Every reference picture of the identity is used, each cut to its face
# (SAM3's biggest, FACE_SWAP_REF_PAD times it, with the hair): a half-body
# photo's face is a tenth of it, too little for Qwen to copy.
FACE_SWAP_REF_PAD = 1.8
FACE_SWAP_PROMPT = ("The person in picture 1 has the face of the person in %s: the same "
                    "eyes, nose, mouth, face shape, eyebrows and skin, turned and lit as "
                    "the face in picture 1.")
# After FaceFusion (Generate with a face profile) the swapped face's eyes
# come back soft and its glasses faint: the swap paints the new face over
# the frames. So two redraws follow it, on the picture's own model, each
# only where SAM3 finds the thing in the crop: the eyes, inside the face's
# eye band, then the glasses last, so nothing is drawn over them (the user,
# 2026-09-26: "it needs to do an eye pass and glasses last").
FINISH_FIND = ["face:8", "glasses:4"]
FINISH_WORDS = ["face", "glasses"]
EYE_WORD = "eye:2"
EYE_PAD = 1.6                         # the eye crop, of the face's longer side: the whole face
EYE_BAND = (0.15, 0.6)                # the eyes' rows, of the face box's height
EYE_DENOISE = 0.5
EYE_WHAT = ("clear, detailed eyes with round irises, dark pupils and lashes, both looking "
            "the same way")
# Live on Partner (2026-09-26): at 0.6, "glasses with ... clear lenses" came
# back with crisp frames but milky lenses over the eyes just drawn; 0.45
# and the lenses said to be glare-free kept the eyes seen through them.
GLASSES_DENOISE = 0.45
GLASSES_WHAT = ("thin metal glasses frames with perfectly clear, transparent lenses and no "
                "glare, the eyes sharp behind them")
# Every Generate of people then has its hands redrawn (the user, 2026-09-26: "a
# pass with natural hands"), before the glasses, which stay last. Only what
# SAM3 finds as a hand in each crop changes. It is a finish, so it must not
# cost a hand that was drawn well, and most are: on Z-Image Turbo 0.6 (the
# Critic's LOCAL_INPAINT, this pass's strength until 2026-09-29) took a ring
# off a finger, aged a florist's hand into scales and bent a guitarist's
# fingers off the frets, where 0.4 left each hand as it was and sharpened the
# small ones. A hand that is wrong is the Critic's or Fix a spot's, at 0.6.
HAND_FIND = "hand:8"
HAND_DENOISE = 0.4
HAND_WHAT = ("a natural, relaxed human hand with four fingers and a thumb, each finger "
             "separate and jointed, with clear knuckles and nails")
# Which of SAM3's hands are hands. It gives each box a score, and asked at
# 0.3 it also gives the forearm round a hand (0.33-0.57), a thing a few pixels
# wide (0.65) and the field a picture is of (0.48); a hand in plain view
# scores 0.78-0.97. Taken biggest first, the forearm won and the hand inside
# it was dropped as its copy: so they are taken by score, and a box that
# shares half the smaller of the two with a better one is that hand again.
# A fox's paws score as a hand's do (0.87), so the pass is for pictures whose
# words name a person (`has_person`), not for what SAM3 calls one.
HAND_SCORE = 0.5
HAND_SAME = 0.5
HAND_SMALL = 0.04                     # of the picture's longer side: a passer-by's hand
# A spot can be a freehand outline (the Fix a spot window's drag): only
# inside it changes. ComfyUI has no polygon mask node, so the outline is
# drawn here as a mask picture at the crop's size (`outline_png`) and
# uploaded; its square (for the crop) is its bounding box, padded.
FIX_OUTLINE_MAX = 400                 # points kept of an outline
FIX_OUTLINE_PAD = 1.25                # the square round an outline, of its longer side


def outline_png(points, width, height):
    """A greyscale PNG of width x height: white inside the closed outline
    `points` [(x, y)], black outside (even-odd, sampled at pixel centres)."""
    import struct
    import zlib
    edges = [(points[i], points[(i + 1) % len(points)]) for i in range(len(points))]
    edges = [(a, b) for a, b in edges if a[1] != b[1]]
    rows = []
    for yy in range(height):
        yc = yy + 0.5
        xs = sorted(a[0] + (yc - a[1]) * (b[0] - a[0]) / float(b[1] - a[1])
                    for a, b in edges if min(a[1], b[1]) <= yc < max(a[1], b[1]))
        row = bytearray(width + 1)            # the filter byte, then the pixels
        for x0, x1 in zip(xs[0::2], xs[1::2]):
            i0, i1 = max(0, int(x0 + 0.5)), min(width, int(x1 + 0.5))
            if i1 > i0:
                row[1 + i0:1 + i1] = b"\xff" * (i1 - i0)
        rows.append(bytes(row))

    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"".join(rows), 6)) + chunk(b"IEND", b""))


def outline_spot(points):
    """A freehand outline (picture px) -> the spot it makes: the square round
    its bounding box, padded by FIX_OUTLINE_PAD, at least FIX_MIN."""
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    side = int(max(FIX_MIN, max(max(xs) - min(xs), max(ys) - min(ys)) * FIX_OUTLINE_PAD))
    return {"x": int((min(xs) + max(xs)) / 2.0), "y": int((min(ys) + max(ys)) / 2.0),
            "size": side, "outline": [[int(x), int(y)] for x, y in points]}


def swap_prompt(fix, spot):
    """The Qwen sentence for a spot with a photo: its Find word, else the
    fix's words, else the target's noun."""
    f = clean_fix(fix)
    what = spot.get("word") or f["words"] or SWAP_NOUNS[f["target"]]
    return SWAP_PROMPTS[f["target"]] % what


def _spots(items, limit):
    out = []
    for sp in items or []:
        try:
            x, y, size = int(sp["x"]), int(sp["y"]), int(sp["size"])
        except (KeyError, TypeError, ValueError):
            continue
        if size < FIX_MIN:
            continue
        spot = {"x": x, "y": y, "size": size}
        try:                          # what Find found inside it (x, y, w, h)
            spot["box"] = [int(v) for v in sp["box"]][:4]
            if len(spot["box"]) != 4:
                del spot["box"]
        except (KeyError, TypeError, ValueError):
            pass
        if isinstance(sp.get("word"), str) and sp["word"].strip():
            spot["word"] = sp["word"].strip()[:40]
        try:                          # a freehand outline: [[x, y], ...] in the picture
            pts = [(int(p[0]), int(p[1])) for p in sp["outline"]][:FIX_OUTLINE_MAX]
            if len(pts) >= 3:
                spot["outline"] = [list(p) for p in pts]
        except (KeyError, TypeError, ValueError, IndexError):
            pass
        if isinstance(sp.get("photo"), str) and sp["photo"].strip():
            spot["photo"] = sp["photo"].strip()     # a picture of what goes there instead
        if isinstance(sp.get("note"), str) and sp["note"].strip():
            spot["note"] = sp["note"].strip()[:FIX_NOTE_MAX]   # what the user says is wrong
        out.append(spot)
    return out[:limit]


def clean_fix(fix):
    """The fix a job carries, made safe: {"image", "target", "words",
    "strength", "spots", "locks", "face_swap"} - spots are [{"x", "y",
    "size", "box"?}], centres and sides in the picture's own pixels (`box`
    what Find found in it); locks are squares of the same shape the fix may
    not change; `face_swap` the id of the identity whose face is swapped
    in last ("" for none). A spot's `note`, or the fix's for the spots
    without one, is what the user says is wrong there; with `check` the
    Visual Critic looks at the result and what is still wrong is redrawn
    again (`Studio._refine`)."""
    fix = fix if isinstance(fix, dict) else {}
    spots = _spots(fix.get("spots"), FIX_MAX_SPOTS)
    locks = _spots(fix.get("locks"), 16)
    target = fix.get("target") if fix.get("target") in FIX_TARGETS else "hand"
    strength = fix.get("strength")
    if strength in FIX_STRENGTHS:
        strength = FIX_STRENGTHS[strength]
    try:
        strength = min(1.0, max(0.2, float(strength)))
    except (TypeError, ValueError):
        strength = FIX_STRENGTHS["medium"]
    try:
        tone = min(1.0, max(0.0, float(fix.get("tone", FIX_TONE))))
    except (TypeError, ValueError):
        tone = FIX_TONE
    point = fix.get("face_point")
    if not (isinstance(point, (list, tuple)) and len(point) == 2
            and all(isinstance(x, (int, float)) and math.isfinite(x) and 0 <= x <= 1 for x in point)):
        point = None
    return {"image": _str(fix.get("image")), "target": target, "tone": tone,
            "around_head": fix.get("around_head") is True,
            "note": _str(fix.get("note"))[:FIX_NOTE_MAX], "check": fix.get("check") is True,
            "words": _str(fix.get("words")), "strength": strength, "spots": spots,
            "locks": locks, "face_swap": _str(fix.get("face_swap")), "face_point": point}


def fix_words(fix):
    """What a fix redraws, in words: "2 hands", "the collar", "1 spot from
    a photo"."""
    f = clean_fix(fix)
    n = len(f["spots"])
    if f["around_head"]:
        return "generate around the locked head"
    if not n and f["face_swap"]:
        return "a face swap"
    if f["words"]:
        return f["words"] + (", then a face swap" if f["face_swap"] else "")
    noun = {"hand": "hand", "face": "face"}.get(f["target"], "spot")
    out = "%d %s%s" % (n, noun, "" if n == 1 else "s")
    k = sum(1 for sp in f["spots"] if sp.get("photo"))
    if k:
        out += " (%s from a photo)" % ("all" if k == n else k)
    return out + (", then a face swap" if f["face_swap"] else "")


def fix_crops(width, height, spots, head=False):
    """Each spot -> the square face_graph redraws, kept inside the picture.
    `head` (a face fix with SAM3) blends back through the face's true mask."""
    crops = []
    for sp in spots:
        side = min(sp["size"], width, height)
        x = min(max(0, sp["x"] - side // 2), width - side)
        y = min(max(0, sp["y"] - side // 2), height - side)
        crops.append({"x": x, "y": y, "width": side, "height": side, "head": head})
    return crops


def fix_areas(crops, spots):
    """Into each crop whose spot Find made: `area`, the found box grown by
    FIX_AREA_GROW in the crop's pixels (x0, y0, x1, y1) - the only part
    redrawn and blended back. A face with SAM3 (`head`) keeps its own
    mask; a clicked square keeps the oval."""
    for crop, sp in zip(crops, spots):
        if sp.get("outline"):
            # Drawn by hand: only inside it changes, whatever the target.
            pts = [(x - crop["x"], y - crop["y"]) for x, y in sp["outline"]]
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            x0, y0 = max(0, int(min(xs))), max(0, int(min(ys)))
            x1, y1 = min(crop["width"], int(max(xs)) + 1), min(crop["height"], int(max(ys)) + 1)
            if x1 - x0 >= 8 and y1 - y0 >= 8:
                crop.update(area=(x0, y0, x1, y1), outline=pts)
            continue
        if crop.get("head") or not sp.get("box"):
            continue
        bx, by, bw, bh = sp["box"]
        gx, gy = bw * FIX_AREA_GROW, bh * FIX_AREA_GROW
        x0, y0 = max(0, int(bx - gx - crop["x"])), max(0, int(by - gy - crop["y"]))
        x1 = min(crop["width"], int(bx + bw + gx - crop["x"]))
        y1 = min(crop["height"], int(by + bh + gy - crop["y"]))
        if x1 - x0 >= 8 and y1 - y0 >= 8:
            crop["area"] = (x0, y0, x1, y1)
            if sp.get("word"):
                crop["word"] = sp["word"]
    return crops


def lock_regions(width, height, locks):
    """Each lock square -> the region of the original laid back last."""
    return [{k: c[k] for k in ("x", "y", "width", "height")}
            for c in fix_crops(width, height, locks)]


def around_head_graph(wf, values, loras, image, width, height, locks, oval, prefix):
    """Inpaint the original frame outside locked head squares, then restore
    those squares at native resolution. No resize or later pass can move them.
    """
    if not locks:
        raise ValueError("Mark the head to keep before generating around it.")
    scale = min(1.0, (1.05e6 / float(width * height)) ** 0.5)
    ew, eh = (max(64, int(v * scale) // 16 * 16) for v in (width, height))
    crop = {"x": 0, "y": 0, "width": width, "height": height,
            "mask": False, "edit": (ew, eh)}
    g = face_graph(wf, values, loras, image, [crop], oval, prefix, locks=locks)
    g["head_noise"] = {"class_type": "SolidMask", "inputs": {
        "value": 1.0, "width": ew, "height": eh}}
    mask = ["head_noise", 0]
    for i, r in enumerate(locks):
        x, y = int(r["x"] * ew / width), int(r["y"] * eh / height)
        right = min(ew, math.ceil((r["x"] + r["width"]) * ew / width))
        bottom = min(eh, math.ceil((r["y"] + r["height"]) * eh / height))
        key = "head_keep%d" % i
        g[key] = {"class_type": "SolidMask", "inputs": {
            "value": 1.0, "width": right - x, "height": bottom - y}}
        g[key + "_mask"] = {"class_type": "MaskComposite", "inputs": {
            "destination": mask, "source": [key, 0], "x": x, "y": y,
            "operation": "subtract"}}
        mask = [key + "_mask", 0]
    g["head_latent"] = {"class_type": "SetLatentNoiseMask", "inputs": {
        "samples": ["fc1_3", 0], "mask": mask}}
    g["fc1_4"]["inputs"]["latent_image"] = ["head_latent", 0]
    return g


def found_spots(width, height, boxes, kind):
    """What Find found (x, y, w, h) -> spots: a square round each, padded by
    FIX_FIND_PAD, at least FIX_MIN, largest first, at most FIX_MAX_SPOTS."""
    pad = FIX_FIND_PAD.get(kind, 1.5)
    out = []
    kept = []
    for b in sorted(boxes, key=lambda b: -b[2] * b[3]):
        bx, by, bw, bh = b[:4]
        if bw * bh > FIX_FIND_MAX.get(kind, 1.0) * width * height:
            continue
        word = b[4] if len(b) > 4 else None

        def inside(k):
            ix = max(0, min(bx + bw, k[0] + k[2]) - max(bx, k[0]))
            iy = max(0, min(by + bh, k[1] + k[3]) - max(by, k[1]))
            return ix * iy >= FIX_FIND_SAME * bw * bh
        if any((k[4] if len(k) > 4 else None) == word and inside(k) for k in kept):
            continue
        kept.append(b)
        side = int(min(max(FIX_MIN, max(bw, bh) * pad), width, height))
        sp = {"x": int(bx + bw / 2.0), "y": int(by + bh / 2.0), "size": side,
              "box": [int(bx), int(by), int(bw), int(bh)]}
        if len(b) > 4 and b[4]:
            sp["word"] = str(b[4])        # what SAM3 was asked: it finds the outline again
        out.append(sp)
    return out[:FIX_MAX_SPOTS]


def real_hands(width, height, boxes):
    """SAM3's hands, each (x, y, w, h, word, score) -> those worth a redraw,
    the surest first: scored HAND_SCORE or more, not smaller than HAND_SMALL
    of the picture, and not a better hand's box again (HAND_SAME)."""
    small = HAND_SMALL * max(width, height)
    kept = []
    for b in sorted(boxes, key=lambda b: -b[5]):
        x, y, w, h = b[:4]
        if b[5] < HAND_SCORE or (w * h) ** 0.5 < small:
            continue

        def same(k):
            ix = max(0, min(x + w, k[0] + k[2]) - max(x, k[0]))
            iy = max(0, min(y + h, k[1] + k[3]) - max(y, k[1]))
            return ix * iy >= HAND_SAME * min(w * h, k[2] * k[3])
        if not any(same(k) for k in kept):
            kept.append(b)
    return kept


def swapped_faces(width, height, boxes, profiles):
    """The faces (x, y, w, h) FaceFusion swapped for `profiles`, chosen as
    it chooses them (studio_facefusion.target_face over the faces left to
    right). One profile falls back to the biggest face when SAM3 sees more
    faces than FaceFusion did."""
    import apps.image_studio.facefusion as facefusion
    order = sorted(boxes, key=lambda b: b[0])
    norm = [[x / float(width), y / float(height), (x + w) / float(width), (y + h) / float(height)]
            for x, y, w, h in order]
    out = []
    for i, p in enumerate(profiles):
        try:
            k = facefusion.target_face(norm, region=p.get("target_region"),
                                       point=p.get("target_point"),
                                       index=i if len(profiles) > 1 else None,
                                       count=len(profiles))
        except RuntimeError:
            if len(profiles) != 1 or not order:
                continue
            k = order.index(max(order, key=lambda b: b[2] * b[3]))
        if order[k] not in out:
            out.append(order[k])
    return out


def eye_spots(faces):
    """Each face -> the eye pass's spot: a square round the whole face,
    its box the face's eye band, redrawn only where SAM3 finds the eyes."""
    top, bottom = EYE_BAND
    return [{"x": int(x + w / 2.0), "y": int(y + h / 2.0),
             "size": int(max(FIX_MIN, max(w, h) * EYE_PAD)),
             "box": [int(x), int(y + h * top), int(w), max(1, int(round(h * (bottom - top))))],
             "word": EYE_WORD} for x, y, w, h in faces]


def glasses_spots(width, height, glasses, faces):
    """What SAM3 found as glasses (x, y, w, h), those centred on one of
    `faces` -> spots, squared as Find squares them."""
    def on(g, f):
        cx, cy = g[0] + g[2] / 2.0, g[1] + g[3] / 2.0
        return f[0] <= cx <= f[0] + f[2] and f[1] <= cy <= f[1] + f[3]
    return found_spots(width, height, [tuple(g[:4]) + ("glasses",) for g in glasses
                                       if any(on(g, f) for f in faces)], "other")


def parts_graph(image, sam3, prompts):
    """SAM3 asked for each of `prompts` in `image` (a LoadImage name), with
    the picture's size, as PreviewAny text for `parts_found`."""
    g = {"1": {"class_type": "LoadImage", "inputs": {"image": image}},
         "2": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": sam3}},
         "6": {"class_type": "GetImageSize", "inputs": {"image": ["1", 0]}},
         "7": {"class_type": "PreviewAny", "inputs": {"source": ["6", 0]}},
         "8": {"class_type": "PreviewAny", "inputs": {"source": ["6", 1]}}}
    for i, p in enumerate(prompts):
        g["p%dt" % i] = {"class_type": "CLIPTextEncode", "inputs": {"text": p,
                                                                   "clip": ["2", 1]}}
        g["p%dd" % i] = {"class_type": "SAM3_Detect", "inputs": {
            "model": ["2", 0], "image": ["1", 0], "conditioning": ["p%dt" % i, 0],
            "threshold": 0.3, "refine_iterations": 0, "individual_masks": True}}
        g["p%dv" % i] = {"class_type": "PreviewAny", "inputs": {"source": ["p%dd" % i, 1]}}
    return g


def faces_graph(images, sam3, prompt=FACE_SWAP_FIND):
    """SAM3 asked for `prompt` in each of `images` (LoadImage names), with
    each picture's size, as PreviewAny text for `faces_found`."""
    g = {"l": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": sam3}},
         "t": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["l", 1]}}}
    for i, image in enumerate(images):
        n = "q%d_" % i
        g[n + "i"] = {"class_type": "LoadImage", "inputs": {"image": image}}
        g[n + "d"] = {"class_type": "SAM3_Detect", "inputs": {
            "model": ["l", 0], "image": [n + "i", 0], "conditioning": ["t", 0],
            "threshold": 0.3, "refine_iterations": 0, "individual_masks": True}}
        g[n + "v"] = {"class_type": "PreviewAny", "inputs": {"source": [n + "d", 1]}}
        g[n + "s"] = {"class_type": "GetImageSize", "inputs": {"image": [n + "i", 0]}}
        g[n + "w"] = {"class_type": "PreviewAny", "inputs": {"source": [n + "s", 0]}}
        g[n + "h"] = {"class_type": "PreviewAny", "inputs": {"source": [n + "s", 1]}}
    return g


def faces_found(entry, n):
    """What faces_graph said of its `n` pictures -> [(width, height, [(x, y,
    w, h)] biggest first) or None for a picture it said nothing of]."""
    out = entry.get("outputs") or {}

    def text(node):
        t = (out.get(node) or {}).get("text") or []
        return json.loads(t[0]) if t else None
    said = []
    for i in range(n):
        try:
            b, w, h = (text("q%d_%s" % (i, k)) for k in "vwh")
        except ValueError:
            b = w = h = None
        if w is None or h is None:
            said.append(None)
            continue
        b = b[0] if b and isinstance(b[0], list) else b or []
        boxes = [(x["x"], x["y"], x["width"], x["height"]) for x in b
                 if max(x["width"], x["height"]) >= 12]
        said.append((int(w), int(h), sorted(boxes, key=lambda x: -x[2] * x[3])))
    return said


def parts_found(entry, n, words=None, scores=False):
    """What parts_graph said about its `n` prompts -> (width, height,
    [(x, y, w, h)]), or None when it said nothing. With `words` (one per
    prompt) each box ends with the word that found it, and with `scores`
    too, with SAM3's score after the word (1.0 from a SAM3 that gives none)."""
    out = entry.get("outputs") or {}

    def text(node):
        t = (out.get(node) or {}).get("text") or []
        return json.loads(t[0]) if t else None
    try:
        width, height = text("7"), text("8")
        said = [text("p%dv" % i) for i in range(n)]
    except ValueError:
        return None
    if width is None or height is None:
        return None
    boxes = []
    for b, word in zip(said, words or [None] * n):
        b = b[0] if b and isinstance(b[0], list) else b or []
        boxes += [(x["x"], x["y"], x["width"], x["height"]) + ((word,) if word else ())
                  + ((_num(x.get("score"), float, 1.0),) if word and scores else ())
                  for x in b if max(x["width"], x["height"]) >= 12]
    return int(width), int(height), boxes


def fix_prompt(fix, prompt):
    """The close-up prompt a fix's squares are redrawn from."""
    f = clean_fix(fix)
    what = ", ".join(x for x in (f["words"], FIX_TARGETS[f["target"]]) if x) or "this detail"
    return FIX_PROMPT % what


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
    of the frame) whose face the finder found. A person whose "at" is None
    (the form's one person) is the biggest face left over."""
    pairs = []
    for i, (x, y, w, h) in enumerate(boxes):
        cx, cy = x + w / 2.0, y + h / 2.0
        reach = max(FACE_MATCH * max(w, h), FACE_MATCH_FRAME * max(width, height))
        for j, person in enumerate(people):
            if person.get("at") is None:
                continue
            px, py = person["at"][0] * width, person["at"][1] * height
            d = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
            if d <= reach:
                pairs.append((d, i, j))
    out, used = {}, set()
    for d, i, j in sorted(pairs):
        if i not in out and j not in used:
            out[i] = people[j]
            used.add(j)
    left = sorted((i for i in range(len(boxes)) if i not in out),
                  key=lambda i: -boxes[i][2] * boxes[i][3])
    for person in people:
        if person.get("at") is None and left:
            out[left.pop(0)] = person
    return out


# A person's head shape goes into their face's redraw as a depth map from the
# scene's camera (`_head_depths`) - only where the face was drawn turned about
# as the mannequin is: within HEAD_AGREE degrees whole, within HEAD_ROUGH at
# half strength, else not at all. The turn is read crudely, off where the nose
# sits between the face's two sides (`studio_scene._facing`), which a depth
# of NOSE_DEPTH before sides SIDE_HALF apart makes an angle.
HEAD_DENOISE = 0.6             # a shaped face's redraw, deeper than the face pass's 0.4
HEAD_AGREE = 15
HEAD_ROUGH = 30
FACE_SIDES_NOSE = (23, 39, 53)    # COCO-WholeBody: the jaw's two ends, the nose's tip
FACE_POINT_SCORE = 0.3


def facing_yaw(r):
    """A `_facing` (0..1, 0.5 ahead) -> degrees of turn, + to the picture's
    right; +-38 or so is as far as it reads, the nose past the sides."""
    return math.degrees(math.atan((r - 0.5) * 1.6))


def drawn_facing(people, box):
    """The `_facing` of the face DWPose found inside `box` (x, y, w, h), or
    None when it found none there sure enough."""
    x, y, w, h = box
    for p in people or []:
        pts = p.get("points") or []
        if len(pts) <= max(FACE_SIDES_NOSE):
            continue
        a, b, nose = (pts[k] for k in FACE_SIDES_NOSE)
        if min(a[2], b[2], nose[2]) < FACE_POINT_SCORE:
            continue
        if not (x <= nose[0] <= x + w and y <= nose[1] <= y + h):
            continue
        lo, hi = sorted((a[0], b[0]))
        if hi - lo < 1:
            continue
        return max(0.0, min(1.0, (nose[0] - lo) / (hi - lo)))
    return None


def head_gate(expected, drawn):
    """-> (0, 0.5 or 1, why): how much of a head's depth a face gets, by how
    far its drawn turn is from the mannequin's."""
    if expected is None:
        return 0.0, "is turned from the camera"
    if drawn is None:
        return 0.5, "could not be read for its turn"
    edge = lambda r: -1 if r < 0.05 else (1 if r > 0.95 else 0)   # noqa: E731
    if edge(expected) and edge(expected) == edge(drawn):
        return 1.0, ""                         # both past what the measure reads
    off = abs(facing_yaw(expected) - facing_yaw(drawn))
    if off <= HEAD_AGREE:
        return 1.0, ""
    if off <= HEAD_ROUGH:
        return 0.5, "was drawn turned about %d degrees from the mannequin's" % off
    return 0.0, "was drawn turned about %d degrees from the mannequin's" % off


# A character's face on the plain form. The form has one person, so the
# character's face photos are theirs: drawn into the whole frame (PuLID in
# the picture itself), matched to the biggest face the finder finds, then
# the face pass and the real-face paste as in a scene, at the Scene
# Builder's defaults (studio_scene.FACE_LIKENESS, REAL_FACES). A scene's own
# faces (`scene_faces`) win: it knows where each person stands.
FORM_LIKENESS = 0.6
FORM_REAL_FACES = True
WHOLE_FRAME = [0.0, 0.0, 1.0, 1.0]


def faces_of(settings):
    """-> the job's faces as the face pass takes them ({"likeness", "real",
    "people"}): the scene's, else the form's character's photos, else {}."""
    if settings.get("scene_faces"):
        return settings["scene_faces"]
    photos = [p for p in _strs(settings.get("face_photos")) if os.path.isfile(p)]
    if not photos:
        return {}
    return {"likeness": FORM_LIKENESS, "real": FORM_REAL_FACES,
            "people": [{"id": "form", "name": settings.get("face_name") or "the person",
                        "at": None, "region": list(WHOLE_FRAME), "words": "",
                        "face": photos[0], "from": "their face photos", "photos": photos}]}


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
# 2026-09-29: a single photo's PuLID embedding is what looked "pasted on" at
# high denoise (the user, 2026-09-25) - one angle, one lighting, over-trusted.
# Chaining more of the person's own photos onto the same masked region adds
# real signal (other angles, lighting) instead of just pushing one photo's
# weight higher. The first photo carries the full weight; each further one
# chains in at PULID_EXTRA_WEIGHT so an odd angle or poor light cannot swamp
# a clear photo - needs a live test to dial in against real faces.
PULID_EXTRA_WEIGHT = 0.5
REFERENCE_PHOTOS_MAX = 3        # photos of one person chained into PuLID at once
REGION_EDGE = 64                # px on a region mask's long edge


def region_png(region, width, height):
    """PNG bytes: white over `region` ([x0, y0, x1, y1] fractions) on black,
    at the frame's shape, REGION_EDGE px on its long edge."""
    import core.icons as studio_icons
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
    """Into a filled graph: one ApplyPulidFlux per (face picture(s), region
    mask) - LoadImage names - chained on the model its samplers share, each
    confined to its mask, or to nothing when the mask is None (the whole
    frame). Every KSampler on that model takes the chain.

    A person's entry may be one photo (a name) or several (a list, most
    recognisable first): each extra photo chains onto the same masked region
    at PULID_EXTRA_WEIGHT, capped at REFERENCE_PHOTOS_MAX, so more angles of
    the same person add signal without the first photo losing the lead."""
    samplers = [n for n in graph.values() if n["class_type"] == "KSampler"]
    if not samplers or not faces:
        return graph
    base = samplers[0]["inputs"]["model"]
    graph["pb1"] = {"class_type": "PulidFluxModelLoader", "inputs": {"pulid_file": pulid_file}}
    graph["pb2"] = {"class_type": "PulidFluxEvaClipLoader", "inputs": {}}
    graph["pb3"] = {"class_type": "PulidFluxInsightFaceLoader", "inputs": {"provider": "CUDA"}}
    last = base
    for i, (photos, mask) in enumerate(faces, 1):
        photos = [photos] if isinstance(photos, str) else list(photos)[:REFERENCE_PHOTOS_MAX]
        mask_link = None
        if mask is not None:
            graph["pb_%dm" % i] = {"class_type": "LoadImage", "inputs": {"image": mask}}
            graph["pb_%dk" % i] = {"class_type": "ImageToMask", "inputs": {
                "image": ["pb_%dm" % i, 0], "channel": "red"}}
            mask_link = ["pb_%dk" % i, 0]
        for j, face in enumerate(photos):
            n = "pb_%d" % i if j == 0 else "pb_%d_%d" % (i, j + 1)
            graph[n + "f"] = {"class_type": "LoadImage", "inputs": {"image": face}}
            graph[n] = {"class_type": "ApplyPulidFlux", "inputs": {
                "model": last, "pulid_flux": ["pb1", 0], "eva_clip": ["pb2", 0],
                "face_analysis": ["pb3", 0], "image": [n + "f", 0],
                "weight": weight if j == 0 else PULID_EXTRA_WEIGHT,
                "start_at": 0.0, "end_at": 1.0}}
            if mask_link is not None:
                graph[n]["inputs"]["attn_mask"] = mask_link
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
               boxes=None, locks=(), mask_word="head"):
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
    an image is drawn to it through PuLID (`pulid_file`); one with a
    "depth" (a LoadImage name: their head's depth map over the crop,
    `studio_scene.depth_crop_png`) is drawn to that shape through the
    workflow's ControlNet. `boxes`, beside
    `crops`, are the faces the finder found (x, y, w, h): each is blended
    back whole, whatever SAM3 makes of the head around it.

    `locks` (regions of the picture) are laid back from `image` last, so
    nothing inside them changes. `mask_word` is what SAM3 is asked for as
    a head crop's blend mask ("face" for Fix a spot's face)."""
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
        g["fh2"] = {"class_type": "CLIPTextEncode", "inputs": {"text": mask_word,
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
        if face.get("prompt") and values.get("face_prompt"):
            # A crop's whole prompt, its own: a marked spot redrawn again.
            positive, negative = _restated(g, links, values["face_prompt"],
                                           face["prompt"], "_" + n[:-1])
        elif face.get("words") and values.get("face_prompt"):
            positive, negative = _restated(g, links, values["face_prompt"],
                                           FACE_PROMPT % face["words"], "_" + n[:-1])
        images = face.get("images") or ([face["image"]] if face.get("image") else [])
        for j, img in enumerate(images[:REFERENCE_PHOTOS_MAX]):
            rn, pn = (n + "r", n + "p") if j == 0 else (n + "r%d" % (j + 1), n + "p%d" % (j + 1))
            g[rn] = {"class_type": "LoadImage", "inputs": {"image": img}}
            g[pn] = {"class_type": "ApplyPulidFlux", "inputs": {
                "model": model, "pulid_flux": ["pl1", 0], "eva_clip": ["pl2", 0],
                "face_analysis": ["pl3", 0], "image": [rn, 0],
                "weight": PULID_WEIGHT if j == 0 else PULID_EXTRA_WEIGHT,
                "start_at": 0.0, "end_at": 1.0}}
            model = [pn, 0]
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
        area = crop.get("area")
        if area:
            # Only what Find found (Fix a spot): a hard box redrawn, its
            # softened copy blended back - not the oval's whole reach.
            kx, ky = ew / float(side), eh / float(tall)
            ax0, ay0 = int(area[0] * kx), int(area[1] * ky)
            aw, ah = max(1, int(area[2] * kx) - ax0), max(1, int(area[3] * ky) - ay0)
            if crop.get("shape"):
                # A freehand outline: its own mask picture, drawn at the crop's size.
                g[n + "a0"] = {"class_type": "LoadImage", "inputs": {"image": crop["shape"]}}
                g[n + "a1"] = {"class_type": "ImageScale", "inputs": {
                    "image": [n + "a0", 0], "upscale_method": "bilinear", "width": ew,
                    "height": eh, "crop": "disabled"}}
                g[n + "a2"] = {"class_type": "ImageToMask", "inputs": {"image": [n + "a1", 0],
                                                                      "channel": "red"}}
            else:
                g[n + "a0"] = {"class_type": "SolidMask", "inputs": {
                    "value": 0.0, "width": ew, "height": eh}}
                g[n + "a1"] = {"class_type": "SolidMask", "inputs": {
                    "value": 1.0, "width": aw, "height": ah}}
                g[n + "a2"] = {"class_type": "MaskComposite", "inputs": {
                    "destination": [n + "a0", 0], "source": [n + "a1", 0], "x": ax0,
                    "y": ay0, "operation": "or"}}
            shape = n + "a2"
            if crop.get("word") and values.get("sam3"):
                # The thing's own outline, found again in the crop, grown a
                # little and kept inside the box: glasses are redrawn as
                # glasses, not as the block round them.
                if "fx1" not in g:
                    g["fx1"] = {"class_type": "CheckpointLoaderSimple",
                                "inputs": {"ckpt_name": values["sam3"]}}
                g[n + "s0"] = {"class_type": "CLIPTextEncode", "inputs": {
                    "text": crop["word"], "clip": ["fx1", 1]}}
                g[n + "s1"] = {"class_type": "SAM3_Detect", "inputs": {
                    "model": ["fx1", 0], "image": [n + "2", 0], "conditioning": [n + "s0", 0],
                    "threshold": 0.3, "refine_iterations": 2, "individual_masks": False}}
                g[n + "s2"] = {"class_type": "GrowMask", "inputs": {
                    "mask": [n + "s1", 0], "expand": FIX_SHAPE_GROW, "tapered_corners": True}}
                g[n + "s3"] = {"class_type": "MaskComposite", "inputs": {
                    "destination": [n + "s2", 0], "source": [n + "a2", 0], "x": 0, "y": 0,
                    "operation": "multiply"}}
                shape = n + "s3"
            g[n + "3n"] = {"class_type": "SetLatentNoiseMask", "inputs": {
                "samples": latent, "mask": [shape, 0]}}
            latent = [n + "3n", 0]
        elif crop.get("mask") is not False:
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
        if face.get("depth") and values.get("controlnet"):
            # Their head's shape, drawn for this crop from the scene's camera:
            # at FACE_EDIT the jaw and chin are hundreds of pixels, where in
            # the whole frame's depth map they were a few.
            if "fcn" not in g:
                g["fcn"] = {"class_type": "ControlNetLoader",
                            "inputs": {"control_net_name": values["controlnet"]}}
            g[n + "d"] = {"class_type": "LoadImage", "inputs": {"image": face["depth"]}}
            g[n + "dc"] = {"class_type": "ControlNetApplyAdvanced", "inputs": {
                "positive": positive, "negative": negative, "control_net": ["fcn", 0],
                "image": [n + "d", 0],
                "strength": face.get("depth_strength") or FACE_DEPTH_STRENGTH,
                "start_percent": 0.0, "end_percent": FACE_DEPTH_END, "vae": links["vae"]}}
            positive, negative = [n + "dc", 0], [n + "dc", 1]
        # A redraw over what is there takes the workflow's `redraw_sampler`
        # and `redraw_scheduler` when it names them. Z-Image Turbo's own
        # (res_multistep) adds no noise as it goes, and at every strength
        # from 0.3 to 0.6 it left specks and beads on skin; euler_ancestral
        # leaves skin, and drifts further from the face it redrew - least on
        # the beta scheduler (ArcFace, 2026-09-29; AGENTS.md "A redraw is
        # sampled as a redraw").
        g[n + "4"] = {"class_type": "KSampler", "inputs": {
            "seed": (seed + i + 1) % (MAX_SEED + 1),
            "steps": values.get("redraw_steps") or values["steps"], "cfg": 1.0,
            "sampler_name": values.get("redraw_sampler") or values["sampler"],
            "scheduler": values.get("redraw_scheduler") or values["scheduler"],
            "denoise": face.get("denoise") or values["face_denoise"], "model": model,
            "positive": positive, "negative": negative,
            "latent_image": latent}}
        g[n + "5"] = {"class_type": "VAEDecode", "inputs": {"samples": [n + "4", 0],
                                                            "vae": links["vae"]}}
        drawn = [n + "5", 0]
        redrawn = (shape if area else n + "3h")
        if values.get("match_tone") and redrawn in g:
            # The redraw's colours and tone curves moved to the original's
            # over the part redrawn: a hand keeps the picture's grade.
            g[n + "t"] = {"class_type": TONE_NODE, "inputs": {
                "image": drawn, "reference": [n + "2", 0], "mask": [redrawn, 0],
                "amount": float(values["match_tone"])}}
            drawn = [n + "t", 0]
        g[n + "6"] = {"class_type": "ImageScale", "inputs": {
            "image": drawn, "upscale_method": "lanczos", "width": side,
            "height": tall, "crop": "disabled"}}
        if crop.get("mask") is False:
            g[n + "9"] = {"class_type": "ImageCompositeMasked", "inputs": {
                "destination": last, "source": [n + "6", 0], "x": crop["x"], "y": crop["y"],
                "resize_source": False}}
            last = [n + "9", 0]
            continue
        blend = ["fo", 0]
        if area:
            g[n + "a3"] = {"class_type": "MaskToImage", "inputs": {"mask": [shape, 0]}}
            g[n + "a4"] = {"class_type": "ImageBlur", "inputs": {
                "image": [n + "a3", 0], "blur_radius": FIX_AREA_SOFT[0],
                "sigma": FIX_AREA_SOFT[1]}}
            blend = [n + "a4", 0]
        elif values.get("sam3") and crop.get("head", True):
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
    for i, region in enumerate(locks or ()):
        g["fl%d_1" % i] = {"class_type": "ImageCropV2", "inputs": {"image": ["fi", 0],
                                                                  "crop_region": dict(region)}}
        g["fl%d_2" % i] = {"class_type": "ImageCompositeMasked", "inputs": {
            "destination": last, "source": ["fl%d_1" % i, 0], "x": region["x"],
            "y": region["y"], "resize_source": False}}
        last = ["fl%d_2" % i, 0]
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


def _tag_re(tag):
    words = r"\s+".join(re.escape(w) for w in (tag or "").split())
    return re.compile(r"(?<![\w-])%s(?:e?s)?(?![\w-])" % words, re.I) if words else None


def says(text, tag):
    """Whether `text` names `tag` in words of its own, any case, a plural
    allowed: "glasses" is said in "round glasses" and "Glasses on", not in
    "sunglasses"."""
    found = _tag_re(tag)
    return found is not None and found.search(text or "") is not None


def outfit_of(settings):
    """The outfit a Generate job is dressed in: the character's pictures of
    what the form has them wearing today - garments in the Clothes slots,
    the rest as accessories - and the hair picture, when it has one.

    Each picture is a tag (the creator's Tags tab): something on the person
    - glasses, earrings, a dress, a tattoo - and its picture. A tag counts
    when a slot holds it exactly, or when its word is said in a slot ("round
    glasses" for glasses, Traits' "tattoos" for a tattoo) or in the scene
    ("she pushes her glasses up"). Marks on the skin are accessories here. One said only in the scene is clothing when its word names
    a garment the Clothes picks do ("red dress": a dress) and nothing worn on
    the head ("top hat"), else an accessory."""
    refs = {k.lower(): (k, x) for k, x in clean_item_refs(settings.get("item_refs")).items()}
    out = {"clothes": [], "hair": None, "accessories": []}
    used = set()
    for k in TAG_SLOTS:
        items = split_many(_field(settings, k)) if SLOTS[k][4] else [_field(settings, k)]
        for name in items:
            if name and name.lower() in refs and name.lower() not in used:
                used.add(name.lower())
                out["clothes" if k in CLOTHES_SLOTS else "accessories"].append(
                    {"name": refs[name.lower()][0], "path": refs[name.lower()][1]})
    # The longer tag first, and what it said is spent: "a rose tattoo" is
    # the rose tattoo's picture, not the plain tattoo's too.
    said = {k: _field(settings, k) for k in TAG_SLOTS + ("scene",)}
    garments = {x.split()[-1].lower() for k in CLOTHES_SLOTS for x in SLOTS[k][3]}
    for low, (name, path) in sorted(refs.items(), key=lambda kv: -len(kv[0])):
        if low == HAIR_ITEM or low in used:
            continue
        found = _tag_re(name)
        slot = next((k for k in TAG_SLOTS if found and found.search(said[k])), None)
        if found is None or slot is None and not found.search(said["scene"]):
            continue
        said = {k: found.sub(" ", v) for k, v in said.items()}
        used.add(low)
        clothes = (slot in CLOTHES_SLOTS if slot else
                   not on_head(name) and any(says(name, g) for g in garments))
        out["clothes" if clothes else "accessories"].append({"name": name, "path": path})
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
        if item.get("region"):            # only this part of the picture: a face
            g[d + "r%d" % i] = {"class_type": "ImageCropV2", "inputs": {
                "image": loads[-1], "crop_region": dict(item["region"])}}
            loads[-1] = [d + "r%d" % i, 0]
    if ps.get("sheet") and len(loads) > 1:
        # All the pictures side by side as one: Qwen takes three pictures in
        # all, so more references than two ride in one.
        for i in range(1, len(loads)):
            g[d + "h%d" % i] = {"class_type": "ImageStitch", "inputs": {
                "image1": loads[0], "image2": loads[i], "direction": "right",
                "match_image_size": True, "spacing_width": 16, "spacing_color": "white"}}
            loads[0] = [d + "h%d" % i, 0]
        loads = loads[:1]
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


def swap_graph(wf, values, image, crops, pictures, prompts, oval, prefix, sam3=None,
               locks=(), tone=0.0):
    """Fix a spot from photos: in `image` (a LoadImage name) each of `crops`
    (fix_crops, each with its spot's "photo" path, and fix_areas' `area` and
    `word` when Find made it) is cut out, swapped by Qwen-Image-Edit - the
    crop as picture 1, the photo (`pictures` maps its path to a LoadImage
    name) as picture 2, `prompts` beside `crops` - and blended back:

    - through the thing's own outline when Find named it and `sam3` is
      given: SAM3's word in the crop before and after (the new thing may be
      bigger than the old), grown SWAP_GROW px and softened;
    - else through Find's box grown by FIX_AREA_GROW, softened;
    - else (a clicked square) through `oval`, which reaches the spot alone.

    A crop may carry `photos` ([{"path", "region"?}], each cut to its
    region) in place of `photo`: more than SWAP_SLOTS go in side by side as
    one picture. With `tone` the swap's colour moves that far to the
    crop's (ColorTransfer, reinhard_lab).

    Qwen redraws the whole crop a shade off, so nothing outside the mask is
    kept from it. `locks` are laid back from `image` last; the result is
    saved under `prefix` by node "fs", like face_graph's."""
    g, v = _dress_start(wf, values, [])
    g["fi"] = {"class_type": "LoadImage", "inputs": {"image": image}}
    g["fo"] = {"class_type": "LoadImage", "inputs": {"image": oval}}
    if sam3 and any(c.get("word") for c in crops):
        g["sw_l"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": sam3}}
    last = ["fi", 0]
    for i, crop in enumerate(crops):
        n, side, tall = "sw%d_" % (i + 1), crop["width"], crop["height"]
        region = {k: crop[k] for k in ("x", "y", "width", "height")}
        g[n + "c"] = {"class_type": "ImageCropV2", "inputs": {"image": last,
                                                             "crop_region": region}}
        items = crop.get("photos") or [{"path": crop["photo"]}]
        ps = {"kind": "swap", "items": items, "prompt": prompts[i],
              "sheet": len(items) > SWAP_SLOTS}
        drawn = _dress_pass(g, n, ps, [n + "c", 0], (side, tall), v, pictures,
                            int(v["seed"]) + i)
        area = crop.get("area")
        if crop.get("shape"):                 # a freehand outline, drawn at the crop's size
            g[n + "m0"] = {"class_type": "LoadImage", "inputs": {"image": crop["shape"]}}
            g[n + "m1"] = {"class_type": "ImageToMask", "inputs": {"image": [n + "m0", 0],
                                                                  "channel": "red"}}
            shape = [n + "m1", 0]
        elif area and crop.get("word") and "sw_l" in g:
            g[n + "t"] = {"class_type": "CLIPTextEncode", "inputs": {
                "text": crop["word"], "clip": ["sw_l", 1]}}
            for k, src in (("m1", [n + "c", 0]), ("m2", drawn)):
                g[n + k] = {"class_type": "SAM3_Detect", "inputs": {
                    "model": ["sw_l", 0], "image": src, "conditioning": [n + "t", 0],
                    "threshold": 0.3, "refine_iterations": 2, "individual_masks": False}}
            g[n + "m3"] = {"class_type": "MaskComposite", "inputs": {
                "destination": [n + "m1", 0], "source": [n + "m2", 0], "x": 0, "y": 0,
                "operation": "or"}}
            g[n + "m4"] = {"class_type": "GrowMask", "inputs": {
                "mask": [n + "m3", 0], "expand": SWAP_GROW, "tapered_corners": True}}
            shape = [n + "m4", 0]
        elif area:
            g[n + "m0"] = {"class_type": "SolidMask", "inputs": {
                "value": 0.0, "width": side, "height": tall}}
            g[n + "m1"] = {"class_type": "SolidMask", "inputs": {
                "value": 1.0, "width": max(1, area[2] - area[0]),
                "height": max(1, area[3] - area[1])}}
            g[n + "m2"] = {"class_type": "MaskComposite", "inputs": {
                "destination": [n + "m0", 0], "source": [n + "m1", 0], "x": area[0],
                "y": area[1], "operation": "or"}}
            shape = [n + "m2", 0]
        else:
            shape = None
        if shape:
            g[n + "b0"] = {"class_type": "MaskToImage", "inputs": {"mask": shape}}
            g[n + "b1"] = {"class_type": "ImageBlur", "inputs": {
                "image": [n + "b0", 0], "blur_radius": FIX_AREA_SOFT[0],
                "sigma": FIX_AREA_SOFT[1]}}
            soft = [n + "b1", 0]
        else:
            soft = ["fo", 0]
        g[n + "b2"] = {"class_type": "ImageScale", "inputs": {
            "image": soft, "upscale_method": "bilinear", "width": side, "height": tall,
            "crop": "disabled"}}
        g[n + "b3"] = {"class_type": "ImageToMask", "inputs": {"image": [n + "b2", 0],
                                                              "channel": "red"}}
        if tone:
            # A face from a studio photo comes back pinker and brighter than
            # the picture: its colour moved to the crop it replaces (Lab
            # mean and spread over the whole crop). StudioMatchTone's
            # per-channel curves posterized a smooth face into cyan and
            # green blotches (live, 2026-09-26).
            g[n + "tn"] = {"class_type": "ColorTransfer", "inputs": {
                "image_target": drawn, "image_ref": [n + "c", 0],
                "method": "reinhard_lab", "source_stats": "per_frame",
                "strength": float(tone)}}
            drawn = [n + "tn", 0]
        g[n + "b4"] = {"class_type": "ImageCompositeMasked", "inputs": {
            "destination": last, "source": drawn, "x": crop["x"], "y": crop["y"],
            "resize_source": False, "mask": [n + "b3", 0]}}
        last = [n + "b4", 0]
    for i, region in enumerate(locks or ()):
        g["fl%d_1" % i] = {"class_type": "ImageCropV2", "inputs": {"image": ["fi", 0],
                                                                  "crop_region": dict(region)}}
        g["fl%d_2" % i] = {"class_type": "ImageCompositeMasked", "inputs": {
            "destination": last, "source": ["fl%d_1" % i, 0], "x": region["x"],
            "y": region["y"], "resize_source": False}}
        last = ["fl%d_2" % i, 0]
    g["fs"] = {"class_type": "SaveImage", "inputs": {"images": last, "filename_prefix": prefix}}
    return g


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
                "controlnet": "ControlNet", "upscale_models": "upscale model",
                "model_patches": "model patch (a Z-Image ControlNet)"}


# The refine pass's enlargement by a super-resolution model (a workflow's
# `refine_model`): ComfyUI's own nodes.
REFINE_MODEL_NODES = {"UpscaleModelLoader", "ImageUpscaleWithModel", "ImageBlend"}


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
            "scene": "", "camera": "", "camera_profile": "", "negative": "", "loras": [],
            "references": {},
            "character": "", "item_refs": {}, "face_photos": [], "face_name": "",
            "anatomy": True,
            "seed": -1, "seed_mode": "random", "steps": None, "guidance": None,
            "sampler": "", "scheduler": "", "width": None, "height": None,
            "denoise": None, "refine": None, "upscale": None, "refine_denoise": None,
            "face_detail": None, "batch": 1, "pose": None, "composition": None,
            "auto_refine": False, "refine_passes": 3, "hand_pass": True, "head_swap": True,
            "glasses_pass": None, "lora_budget": None,
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
    ("chest_size", "Chest size", ["very small chest", "small chest", "slightly smaller chest", "",
                                 "slightly fuller chest", "full chest", "very full chest"]),
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
# ...and where a tag can be said in the look: those, and the marks on the
# skin (Traits: "tattoos", "nose piercing").
TAG_SLOTS = ITEM_SLOTS + ("traits",)


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


# What the pipeline adds to the user's words is drawn, like any other words.
# Every shipped workflow samples at CFG 1: there is no negative prompt, and a
# thing named to rule it out is a thing named. Measured on the 5090
# (2026-09-29, the same seeds, only the added words changed): a sentence about
# the chest and swimwear dressed a gardener in a swimsuit, "Drawn correctly:"
# made photographs into illustrations, a sentence counting fingers made the
# hands the picture and cut the head off, and "a loose thread on a sleeve" made
# the sleeve the picture. So what is added is short, says what is there, and
# goes on the person it is about.

# The constants: what every person in every picture has, whoever they are
# and whatever the creator says about them. (part, positive, negative). A
# workflow whose model draws hands well without being told, and makes them
# the subject when it is told, leaves the sentence out (`"anatomy": false`:
# Z-Image Turbo). Every shipped workflow samples at CFG 1, which ignores the
# negative prompt; the negative is for a model that reads one.
ANATOMY = [
    ("Hands", "two hands, each with four fingers and a thumb",
     "extra fingers, missing fingers, fused fingers, six fingers, extra hands, "
     "malformed hands"),
    ("Feet", "two feet", "extra feet, extra legs, missing feet"),
    ("Eyes", "two eyes", "extra eyes, third eye, misaligned eyes"),
    ("Body", "a proportionate body",
     "extra limbs, extra arms, disproportionate body, elongated neck, deformed body"),
]
# A scene with nobody described still gets the constants when it names a person.
# It also decides whether the hands pass runs (`hand_pass`), so it knows a
# person by their trade, their family and what they are doing, where the word
# can mean nothing else: "a chef plating a dish" names no man or woman.
PEOPLE = re.compile(
    r"\b(wom[ae]n|m[ae]n|person|people|girls?|boys?|lady|ladies|guys?|"
    r"kids?|child(ren)?|he|she|they|his|her|portrait|selfie|couple|"
    r"crowd|dancers?|athletes?|workers?|someone|figure|"
    r"toddlers?|teenagers?|adults?|mothers?|fathers?|parents?|grand(mother|father|parent)s?|"
    r"daughters?|sons?|brothers?|sisters?|husbands?|wife|wives|brides?|grooms?|friends?|"
    r"famil(y|ies)|chefs?|farmers?|fisherm[ae]n|carpenters?|mechanics?|doctors?|nurses?|"
    r"teachers?|students?|soldiers?|sailors?|musicians?|pianists?|guitarists?|violinists?|"
    r"drummers?|singers?|actors?|actress(es)?|florists?|bakers?|barbers?|waiters?|"
    r"waitress(es)?|runners?|swimmers?|climbers?|cyclists?|hikers?|skiers?|surfers?|"
    r"tourists?|photographers?|scientists?|engineers?|astronauts?|firefighters?|"
    r"police(m[ae]n|wom[ae]n)|pedestrians?|commuters?|shoppers?|spectators?)\b", re.I)


# What a person wears when nothing names a garment: never nothing. The body
# and chest words alone read to FLUX as undressed. A scene that asks for bare
# skin gets the floor too; only a named garment replaces it. It is said of
# the person described, after them; with nobody described (the scene's own
# words are the person) there is no one to hang it on, and CLOTHED stands.
COVERED = "wearing clothes suited to the scene"
# ...and said for every person, dressed or not: "in a swimsuit" after "very
# full chest" was drawn topless on Z-Image Turbo (CFG 1: no negative prompt).
# Two words on the person, or a sentence of their own after a scene that
# describes its people itself. Until 2026-09-29 it was "Every person is
# clothed, the chest fully covered by their clothing or swimwear", which drew
# what it named: a man knitting in an armchair shirtless in briefs, a runner
# and a gardener in swimsuits (9 of 9 pictures), and a woman walking a dog
# topless (3 of 3). With these two words all of them came out dressed, and a
# swimsuit that was asked for was still a swimsuit.
CLOTHED = "fully clothed"
GARMENTS = re.compile(
    r"\b(wear(s|ing)?|dressed|clothe[sd]|clothing|outfits?|uniforms?|costumes?|"
    r"(t-?)?shirts?|blouses?|(tank|crop) tops?|sweaters?|jumpers?|hoodies?|cardigans?|vests?|"
    r"dress(es)?|gowns?|skirts?|suits?|tuxedos?|jackets?|coats?|blazers?|robes?|"
    r"kimonos?|sarees?|saris?|jeans|trousers|pants|shorts|leggings|overalls|"
    r"underwear|lingerie|bras?|briefs|boxers|panties|bikinis?|swimsuits?|"
    r"swim ?trunks|armou?r|pyjamas|pajamas|towel)\b", re.I)


def is_dressed(settings):
    """Whether the form or the scene names something the person wears."""
    return bool(any(_field(settings, k) for k in ("top", "bottom", "outerwear"))
                or GARMENTS.search(_field(settings, "scene")))


def anatomy_text():
    # Not "Drawn correctly: ...": on FLUX.1 [dev] and Z-Image Turbo alike the
    # word made a photograph an illustration (2026-09-29).
    parts = [pos for _, pos, _ in ANATOMY]
    return "Every person has exactly " + _and(parts)


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
    if settings.get("mode") == "fix":
        return "Fix: " + fix_words(settings.get("fix"))
    if settings.get("mode") == "blend":
        import apps.image_studio.blend as blend
        return blend.summary(settings)
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


def head_lora(lib, profile, backend_id, inventory):
    """A profile's Klein LoRA for the head swap (`head_lora`, made by Build
    LoRA) as `headswap.head_graph` takes it -> ({"lora", "strength",
    "trigger"} or {}, a note when it names one that cannot be used). A
    LoRA made for Klein 4B (family "flux2", Build LoRA before 2026-10-01)
    is refused: the 9B's layers are another shape."""
    import apps.image_studio.headswap as headswap
    rid = profile.get("head_lora")
    if not rid:
        return {}, None
    rec = lib.get("loras", rid)
    if rec is None:
        return {}, "%s's head LoRA %r is not in the LoRA library." % (profile["name"], rid)
    if compatibility(rec["family"], headswap.FAMILY) is False:
        return {}, "%s's head LoRA %s is for %s, not %s: rebuild it with Build LoRA." % (
            profile["name"], rec["name"], FAMILIES.get(rec["family"], rec["family"]),
            FAMILIES[headswap.FAMILY])
    name = lora_file(rec, backend_id)
    if inventory is not None and name not in (inventory.get("loras") or ()):
        return {}, "%s's head LoRA %s is not on this backend." % (profile["name"], name)
    return {"lora": name, "strength": rec["strength"], "trigger": rec["trigger"]}, None


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


CRITIC_MEMORY = "critic_memory.json"
CRITIC_MEMORY_LOCK = threading.Lock()  # two backends' lanes can both _refine at once


def load_critic_memory(lib):
    """The Visual Critic's kept details (studio_critic.remember), or {}."""
    root = getattr(lib, "root", None)
    if not root:
        return {}
    try:
        with open(os.path.join(root, CRITIC_MEMORY), encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_critic_memory(lib, memory):
    path = os.path.join(lib.root, CRITIC_MEMORY)
    try:
        os.makedirs(lib.root, exist_ok=True)
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(memory, f, indent=1, ensure_ascii=False)
        os.replace(path + ".tmp", path)
    except OSError:
        pass


# What the critic's scores add up to (studio_critic's ledger): which redraw
# mended what, which faults keep coming back, what the critic had passed.
CRITIC_LEDGER = "critic_ledger.json"
CRITIC_LEDGER_LOCK = threading.Lock()


def load_critic_ledger(lib):
    """The Visual Critic's ledger, or {}."""
    root = getattr(lib, "root", None)
    if not root:
        return {}
    try:
        with open(os.path.join(root, CRITIC_LEDGER), encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_critic_ledger(lib, ledger):
    path = os.path.join(lib.root, CRITIC_LEDGER)
    try:
        os.makedirs(lib.root, exist_ok=True)
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(ledger, f, indent=1, ensure_ascii=False)
        os.replace(path + ".tmp", path)
    except OSError:
        pass


def record_of_picture(path):
    """The History record a picture was saved with (its JSON is beside it,
    named by the record's id), or None: a picture from anywhere else."""
    stem = os.path.splitext(os.path.basename(path or ""))[0]
    try:
        with open(os.path.join(os.path.dirname(path), stem.rsplit("_", 1)[0] + ".json"),
                  encoding="utf-8") as f:
            rec = json.load(f)
    except (OSError, ValueError, TypeError):
        return None
    return rec if isinstance(rec, dict) else None


def identity_description_text(settings, library, identities):
    """Bind saved visual descriptions to the same people used for references.

    Scene-linked identities take precedence over stale form selections. Notes
    and avatars are deliberately not prompt inputs.
    """
    scene = settings.get("scene_faces")
    if scene is not None:
        people = [(library.get("identities", person.get("identity")), person.get("region"))
                  for person in scene.get("people") or []]
    else:
        people = [(ident, None) for ident, _ in identities]
    parts = []
    for index, (ident, region) in enumerate(people, 1):
        if not ident or not ident.get("description"):
            continue
        position = ""
        if isinstance(region, (list, tuple)) and len(region) == 4:
            try:
                center = (float(region[0]) + float(region[2])) / 2
                position = " on the left" if center < .4 else " on the right" if center > .6 else " in the center"
            except (TypeError, ValueError):
                pass
        elif len(people) > 1:
            position = " (number %d of %d, left to right)" % (index, len(people))
        parts.append("Person %d%s (%s), identifying appearance: %s" %
                     (index, position, ident["name"], ident["description"]))
    if parts:
        parts.append("Keep these facial proportions and distinguishing features. "
                     "Use the scene's requested clothing, pose and expression.")
    return parts


def plan_identities(p, settings, identities, values, library=None):
    """Keep each photo paired with its own position; never silently drop a person."""
    scene = settings.get("scene_faces")
    if scene is not None:
        people = []
        for person in scene.get("people") or []:
            ident = library.get("identities", person.get("identity")) if library else None
            people.append(dict(person, photos=(ident.get("references") if ident else
                                               person.get("photos")) or [],
                               pool_photos=bool(ident and ident.get("pool_photos"))))
    else:
        people = [{"name": ident["name"],
                   "face": (ident["references"] or [""])[0], "photos": ident["references"],
                   "pool_photos": ident.get("pool_photos", False)}
                  for ident, _ in identities]
        if not people and (settings.get("references") or {}).get("face"):
            people = [{"name": "Person", "face": settings["references"]["face"]}]
    limit = p.workflow["multi_identity"]["max_people"]
    if not 1 <= len(people) <= limit:
        p.errors.append("WithAnyone needs one to %d people with individual face photos; "
                        "this request has %d." % (limit, len(people)))
        return
    boxes, groups = [], []
    pooling = any(person.get("pool_photos") for person in people)
    values["identity_pooling"] = pooling
    width, height = int(values.get("width", 1024)), int(values.get("height", 1024))
    for index, person in enumerate(people, 1):
        name = person.get("name") or "Person %d" % index
        paths = list(dict.fromkeys([p for p in [person.get("face")] +
                                              list(person.get("photos") or []) if p]))
        if not paths or any(not os.path.isfile(path) for path in paths):
            p.errors.append("%s needs an existing face photo for WithAnyone. Set Face in "
                            "Scene Builder or add a photo to their identity profile." % name)
            continue
        use_set = person.get("pool_photos") or (
            settings.get("experimental_reference_groups", False) and not pooling)
        if len(paths) > 1 and not use_set:
            p.notes.append("%s: using the first reference photo. Combining multiple views is "
                           "experimental and failed likeness review; additional library photos "
                           "are kept but not blended." % name)
            paths = paths[:1]
        if len(paths) > 8:
            p.errors.append("%s has %d reference photos; WithAnyone supports up to 8 per person."
                            % (name, len(paths)))
            continue
        box = person.get("region")
        if scene is None:
            # Stable left-to-right arrangement for the form. Scene Builder
            # supplies explicit positions for a different composition.
            center = (index - 0.5) / len(people)
            half = min(0.18, 0.38 / len(people))
            box = [center - half, 0.15, center + half, 0.55]
        try:
            box = [float(x) for x in box]
            valid = (len(box) == 4 and all(math.isfinite(x) and 0 <= x <= 1 for x in box)
                     and (box[2] - box[0]) * width >= 4
                     and (box[3] - box[1]) * height >= 4)
        except (TypeError, ValueError):
            valid = False
        if not valid:
            p.errors.append("%s needs a visible face position inside the frame for WithAnyone." % name)
            continue
        keys = []
        for photo_index, path in enumerate(paths, 1):
            key = "face%d" % index + ("_ref%d" % photo_index if photo_index > 1 else "")
            p.images[key] = path
            p.references[key] = path
            keys.append(key)
        groups.append(keys)
        boxes.append(box)
        p.notes.append("WithAnyone person %d: %s, %d reference photo%s in one face region."
                       % (index, name, len(paths), "s" if len(paths) != 1 else ""))
        if len(paths) > 1:
            if pooling:
                p.notes.append("%s: primary photo supplies appearance; %d photos guide pooled identity."
                               % (name, len(paths)))
            p.warnings.append("Experimental %s for %s; likeness has not passed review." %
                              ("photo pooling" if pooling else "multi-photo blending", name))
    values["identity_boxes"] = json.dumps(boxes)
    values["identity_reference_groups"] = groups
    values["sampler"], values["scheduler"] = "WithAnyone", "flow matching"
    if width > 2048 or height > 2048:
        p.errors.append("WithAnyone supports at most 2048 pixels on either side.")
    if scene is not None:
        p.notes.append("Scene Builder supplies face positions. Poses, props and clothes "
                       "are described by the scene's words; this recipe takes no pose or depth map.")
    else:
        p.notes.append("People are placed left to right in selection order. Use Scene Builder "
                       "to set their positions.")


def chest_control_look(settings):
    """A whole-image LoRA can control one person, not different bodies in a group."""
    scene = settings.get("scene_layout")
    if isinstance(scene, dict):
        folks = [o for o in scene.get("objects", [])
                 if o.get("visible", True) and o.get("asset") in ("person", "crowd")]
        active = [o for o in folks if (o.get("look") or {}).get("chest_size")]
        if not active:
            return {}, ""
        if len(folks) != 1 or folks[0].get("asset") != "person":
            return {}, "Chest size LoRAs affect the whole image; separate chest sizes in a group need individual edits."
        return folks[0].get("look") or {}, ""
    return settings, ""


def chest_control_kind(look):
    subject = str(look.get("subject") or "").lower()
    female = bool(re.search(r"\b(woman|women|female)\b", subject))
    male = bool(re.search(r"\b(man|men|male)\b", subject))
    return "chest_female" if female and not male else "chest_male" if male and not female else ""


def hold_loras(applied, budget, label):
    """Hold a LoRA stack to `budget`: the total weight the model takes before
    its pictures break up (a workflow's `lora_budget`; 0 is no limit).
    `applied` is [[record, strength, why, file]], changed in place. -> the
    warnings to give.

    Only the "always on" ones are turned down, all by the same share, into
    what the others leave of the budget: nobody chose their sum, each was
    switched on alone in Add-ons at the strength it was imported with. A LoRA
    chosen for this picture (a row of the form, an identity's, a style's)
    keeps its strength, and a sum of those over the budget is said, not
    changed. With no room left the always-on ones are left out.

    Measured on the 5090 (2026-09-29, Z-Image Turbo, four always-on LoRAs,
    the same seeds): at 0.8 each (3.2) a fox at dawn was a night scene with a
    grid across it and a grey sweater was underwear; a Scene Builder picture
    (pose and depth maps) was a smear. At 2.0 it was a picture again, at 1.2
    a clean one."""
    total = sum(abs(a[1]) for a in applied)
    if not budget or total <= budget + 1e-9:
        return []
    always = [a for a in applied if a[2] == "always on"]
    chosen = sum(abs(a[1]) for a in applied if a[2] != "always on")
    said = []
    if chosen > budget + 1e-9:
        said.append("The LoRAs chosen for this picture come to %.3g, and %s takes about "
                    "%.3g before its pictures break up. Lower their strengths if this one "
                    "comes out dark, gridded or not as described." % (chosen, label, budget))
    if not always:
        return said
    names = _and([a[0]["name"] for a in always])
    room = budget - chosen
    if room <= 1e-9:
        applied[:] = [a for a in applied if a[2] != "always on"]
        said.append("Always-on LoRAs left out (%s): the LoRAs chosen for this picture "
                    "already come to %.3g, and %s takes about %.3g." % (
                        names, chosen, label, budget))
        return said
    share = room / sum(abs(a[1]) for a in always)
    for a in always:
        a[1] = round(a[1] * share, 3)
    said.append("Always-on LoRAs turned down to fit (%s): at full strength the LoRAs come "
                "to %.3g, and %s takes about %.3g before its pictures break up. Fewer "
                "always on (Add-ons) leaves each one stronger."
                % (_and(["%s to %.3g" % (a[0]["name"], a[1]) for a in always]), total,
                   label, budget))
    return said


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
    preset = preset_info(lib, s["preset"])
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
    if preset["base"] == "identity" and not idents:
        p.errors.append("Identity Portrait needs a person: choose one under Person.")

    # ------------------------------------------------------------ LoRAs
    stack = []                        # (lora record, strength, why)
    chest_look, chest_note = chest_control_look(s)
    chest_step = _num(chest_look.get("chest_size"), int, 0, -3, 3)
    chest_prompt = ""
    if chest_note:
        p.warnings.append(chest_note)
    if chest_step:
        kind = chest_control_kind(chest_look)
        candidates = [r for r in lib.all("loras") if r.get("body_control") == kind and kind
                      and compatibility(r["family"], family) is True]
        available = [r for r in candidates if inventory is None or
                     lora_file(r, bid) in inventory.get("loras", ())]
        if len(available) == 1:
            rec = available[0]
            strength = rec["strength"] * chest_step / 3 if kind == "chest_female" else rec["strength"]
            stack.append((rec, strength, "chest size"))
            if kind == "chest_male":
                chest_prompt = { -3: "Flat Male Chest", -2: "Flat Male Chest",
                                 -1: "Average Male Chest Pectorals", 1: "Medium Male Chest Pectorals",
                                 2: "Medium Perky Male Chest Pectorals",
                                 3: "Large Male Chest, 47-60 Inch Chest, Wide and Thick Chest"}[chest_step]
        else:
            reason = ("Choose a man or woman in Subject" if not kind else
                      "More than one matching chest LoRA is assigned in the library" if len(available) > 1 else
                      "No matching chest LoRA is installed on this backend and assigned in the LoRA library")
            p.warnings.append("Chest size is using words only. " + reason + ".")
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
    # is skipped without a warning: that is what "suits" means here. So is one
    # turned off in Add-ons.
    have_ids = {rec["id"] for rec, _, _ in stack}
    for rec in lib.all("loras"):
        if (rec.get("always") and rec.get("enabled", True) and rec["id"] not in have_ids
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
    applied = []                      # [record, strength, why, file]: what the model is given
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
        applied.append([rec, float(strength), why, fname])
    budget = s.get("lora_budget")
    if budget in (None, ""):
        budget = (wf.get("defaults") or {}).get("lora_budget")
    p.warnings.extend(hold_loras(applied, _num(budget, float, 0.0, 0.0), model["label"]))
    for rec, strength, why, fname in applied:
        p.loras.append((fname, round(strength, 3)))
        p.lora_meta.append({"id": rec["id"], "name": rec["name"], "file": fname,
                            "strength": round(strength, 3), "category": rec["category"],
                            "why": why})

    # ----------------------------------------------------------- prompt
    who = [i["trigger"] for i, _ in idents if i["trigger"]]
    scene = _field(s, "scene")
    person = person_text(s)
    named = " and ".join(t for t in who if t not in scene)
    posed = bool((s.get("references") or {}).get("pose"))
    someone = has_person(s, bool(idents))
    covered = COVERED if someone and not is_dressed(s) else ""
    # The form's camera body; a scene's own words already carry its camera.
    cam_body = chosen_camera(lib, s)
    body_words = (camera_profile_words(cam_body) if cam_body and not s.get("scene_layout")
                  else "")
    if body_words and body_words in scene:
        body_words = ""
    # The clothing floor is said of the person described, in their own
    # sentence; a scene that describes its people itself gets it after.
    described = bool(named or person)
    parts = [x for x in (view_text(s.get("view"), posed),
                         ", ".join(x for x in (named, person, CLOTHED if described else "",
                                               covered if described else "") if x), scene,
                         CLOTHED.capitalize() if someone and not described else "",
                         _field(s, "camera"), body_words) if x]
    descriptions = identity_description_text(s, lib, idents)
    parts.extend(descriptions)
    if descriptions:
        p.notes.append("Included saved identity descriptions with the reference photos.")
    if posed and clean_view(s.get("view")):
        p.warnings.append("The drawn pose decides the framing and which way the person "
                          "faces; the Camera gives only its height. Zoom the figure "
                          "(mouse wheel in Draw…) to frame closer.")
    # What the Visual Critic kept from earlier pictures of this person and
    # scene (critic_memory.json): every picture starts from it, so they agree.
    learned = critic.recall(load_critic_memory(lib), [i["id"] for i, _ in idents],
                            scene, [k for k in SLOTS if _field(s, k)] + ["scene", "camera"])
    if learned:
        parts.append(", ".join(v for _, v in learned))
        p.notes.append("Kept from earlier pictures: %s." % "; ".join(
            "%s %s" % (k.replace("_", " "), v) for k, v in learned))
    # And against what keeps going wrong in this person's pictures (the
    # critic's ledger): the fix that was asked for, said before it is needed.
    against = critic.prevention(load_critic_ledger(lib), [i["id"] for i, _ in idents])
    if against:
        parts.append(", ".join(v for _, v in against))
        p.notes.append("Drawn against faults seen before: %s." % "; ".join(
            "%s (%s)" % (v, k) for k, v in against))
    anatomy = s.get("anatomy") is not False and has_person(s, bool(idents))
    if anatomy and wf.get("anatomy") is False:
        anatomy = False
        p.notes.append("The anatomy constants are not said to %s: it draws hands well "
                       "untold, and told, made them the subject of the picture."
                       % model["label"])
    if anatomy:
        parts.append(anatomy_text())
    if style:
        extra = " ".join(x for x in (style["trigger"], style["prompt"]) if x)
        if extra:
            parts.append(extra)
    # A trigger is said for a LoRA the model is given, never for one left out
    # (not on this machine, another family's): its words alone are no LoRA.
    for rec, _, why, _ in applied:
        if why in ("added", "always on") and rec["trigger"] and rec["trigger"] not in " ".join(parts):
            parts.append(rec["trigger"])
    p.prompt = ". ".join(x.rstrip(" .") for x in parts if x) + ("." if parts else "")
    if chest_prompt and any(m["why"] == "chest size" for m in p.lora_meta):
        p.prompt += " " + chest_prompt + "."
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
    # The camera body's frame shape, over the model's size; a size typed in
    # Advanced (or a scene's frame) still wins below.
    if cam_body and cam_body.get("format") and "width" in look and "height" in look:
        look["width"], look["height"] = camera_size(cam_body["format"], look["width"],
                                                    look["height"])
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
    if wf.get("multi_identity"):
        plan_identities(p, s, idents, v, lib)
        if v.get("identity_pooling") and nodes is not None and "StudioWithAnyonePooled" not in nodes:
            p.errors.append("Update the WithAnyone node on %s and restart ComfyUI to use photo pooling."
                            % backend["name"])
        if nodes is not None and any(len(g) > 1 for g in v.get("identity_reference_groups", [])):
            if "StudioWithAnyoneReferences" not in nodes:
                p.errors.append("Update the WithAnyone node on %s and restart ComfyUI to use "
                                "multiple reference photos per person." % backend["name"])
        refs.pop("face", None)  # handled per person, never reduced to the first profile
    if "face" not in refs and not wf.get("multi_identity"):
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
    regions_in = s.get("character_regions") or []
    if regions_in and wf.get("regional_conditioning"):
        regions = []
        for i, region in enumerate(regions_in, 1):
            path = region.get("mask_path")
            if not path or not os.path.isfile(path):
                p.warnings.append("A character's mask picture is not on this PC any "
                                  "more; regional prompting skipped for it.")
                continue
            var = "char_mask_%d" % i
            p.images[var] = path
            regions.append({"prompt": region["prompt"], "mask_var": var})
        if regions:
            v["character_regions"] = regions
            p.notes.append("%d character%s given their own words and region of the "
                           "picture." % (len(regions), "" if len(regions) == 1 else "s"))
    elif regions_in:
        p.notes.append("Regional character prompting needs a workflow that supports it "
                       "(the %s workflow does not); used as ordinary words instead."
                       % wf.get("label", wid))
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
        import apps.image_studio.scene.pose as studio_pose
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
    # What enlarges the picture before the redraw: a super-resolution model
    # when the workflow names one (`refine_model`) and the backend has it,
    # else lanczos, and said - the redraw then sharpens a blur.
    v.pop("refine_model", None)
    rm = wf.get("refine_model") or {}
    var = rm.get("file")
    name = (v.get(var) or (wf.get("defaults") or {}).get(var)) if var else ""
    if v.get("refine") and name:
        folder = (wf.get("files") or {}).get(var, "upscale_models")
        there = inventory is None or name in inventory.get(folder, ())
        short = sorted(REFINE_MODEL_NODES - set(nodes)) if nodes is not None else []
        if there and not short:
            up = float(v.get("upscale") or (wf.get("defaults") or {}).get("upscale") or 1.5)
            v[var] = name
            v["refine_model"] = True
            v["refine_model_by"] = round(up / float(rm.get("scale") or 4), 6)
        else:
            v.pop(var, None)
            # A blur needs a deeper redraw to become detail than a picture
            # the model has already sharpened: the workflow's values for the
            # pass without it, where the user set none.
            for k, x in (rm.get("without") or {}).items():
                if s.get(k) in (None, ""):
                    v[k] = x
            p.warnings.append(
                "Refine: %s lacks %s, so the picture is enlarged by lanczos before its "
                "redraw, which is softer." % (backend["name"], (
                    "%s (the %s, in ComfyUI/models/%s)" % (
                        name, FOLDER_WORDS.get(folder, folder), folder) if not there
                    else "the node%s %s" % ("" if len(short) == 1 else "s",
                                            ", ".join(short)))))
    elif var:
        v.pop(var, None)

    # The face pass needs the template's face_detail section, a SAM3
    # checkpoint to find the faces and the stock nodes it is built from. It
    # is a finish, not the picture: without them the picture is made and the
    # pass is left out, said once.
    form_face = not s.get("scene_faces") and not wf.get("multi_identity") and faces_of(s)
    if form_face:
        who = form_face["people"][0]
        v["face_detail"] = True
        if set(wf.get("families") or ()) & PULID_FAMILIES:
            p.notes.append("Face: %s, from %d photo%s." % (
                who["name"], len(who["photos"]), "" if len(who["photos"]) == 1 else "s"))
        else:
            p.warnings.append("Face: only a FLUX.1 model can draw %s's face from their "
                              "photos; with %s the face comes from the words." % (
                                  who["name"], wf.get("label", wid)))
    if wf.get("multi_identity"):
        v["face_detail"] = False
        p.notes.append("WithAnyone draws the identities together. Face redraw, photo paste "
                       "and automatic refinement are off for this recipe.")
    # The face pass draws "a real human face ... natural lips and teeth" on
    # what SAM3 calls a face, and a fox's is one to it: High Quality Final
    # gave a fox a person's mouth (2026-09-30). Like the hands pass, it is
    # for pictures whose words name a person.
    if (v.get("face_detail") and not someone and not form_face
            and not (s.get("scene_faces") or {}).get("people")):
        v["face_detail"] = False
        p.notes.append("No face pass: the picture's words name no person.")
    if v.get("face_detail"):
        asked = (s.get("face_detail") or preset["values"].get("face_detail")
                 or bool(form_face))
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


def preview_graph(plan):
    """`plan.workflow` filled to look at, not to run - for the Nodes button
    on a form or a queued job that has not been submitted (and so has no
    `job.graph` yet): a reference shows its local file's name since nothing
    is uploaded, and the output name is a placeholder. -> the API-format
    graph, or None when there is nothing to build (an error, or no workflow
    chosen yet)."""
    if plan is None or not plan.workflow or plan.errors:
        return None
    values = dict(plan.values)
    for var, path in plan.images.items():
        values.setdefault(var, os.path.basename(path))
    values.setdefault("filename_prefix", "ImageStudio/preview")
    try:
        graph = fill(plan.workflow, values, plan.loras)
    except TemplateError:
        return None
    if plan.items:
        add_item_refs(graph, plan.workflow["items"],
                      [os.path.basename(path) for _, path in plan.items])
    return graph


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
            "face", "critic", "head_swap", "face_swap", "eyes", "hands", "glasses",
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


def hand_pass(settings):
    """Whether Generate ends with the hands pass: unless it is unticked, for
    a picture whose words name a person. A red fox's paws are hands to SAM3
    (scored 0.87), and were redrawn as a human hand's."""
    return (settings.get("hand_pass", True) is not False
            and has_person(settings, bool(settings.get("identities"))))


def glasses_pass(settings, reports=None):
    """Whether the glasses are redrawn after the face swap. The pass is there
    because the swap painted the new face over the frames (2026-09-26). With
    its occlusion mask (`facefusion.SWAP_MASKS`) it goes behind them and they
    stay as drawn, to the pixel; redrawn all the same they came back as other
    glasses (dark red frames as tortoiseshell) and cost the likeness 0.08
    (ArcFace 0.81 to 0.73, 2026-09-29). So: only after a swap that did not
    keep them (`reports`, the swaps' own; None before there are any), or
    when settings["glasses_pass"] says so either way."""
    import apps.image_studio.facefusion as facefusion
    if settings.get("glasses_pass") in (True, False):
        return settings["glasses_pass"]
    masks = ([r.get("masks") or () for r in reports] if reports
             else [facefusion.SWAP_MASKS])
    return not all("occlusion" in m for m in masks)


def pipeline_stages(lib, settings):
    """The stops a job's pipeline strip shows, in the order Generate runs
    them: the two ComfyUI stages every job goes through, then each optional
    finishing pass these settings turn on. Queued and loading are left off -
    obvious, not worth a stop on the strip."""
    import apps.image_studio.facefusion as facefusion
    if settings.get("mode") == "blend":       # one Kontext run, no finishing pass
        import apps.image_studio.blend as blend
        return list(blend.STAGES)
    stages = [("sampling", "Sampling")]
    if faces_of(settings):
        stages.append(("face", "Face pass"))
    if settings.get("auto_refine"):
        stages.append(("critic", "Critic"))
    stages.append(("decoding", "Decoding"))
    profiles = facefusion.selected(lib, settings)
    if profiles:
        if settings.get("head_swap", True) is not False:
            stages.append(("head_swap", "Head swap"))
        stages.append(("face_swap", "Face swap"))
        stages.append(("eyes", "Eye pass"))
    if hand_pass(settings):
        stages.append(("hands", "Hand pass"))
    if profiles and glasses_pass(settings):
        stages.append(("glasses", "Glasses"))
    stages.append(("complete", "Complete"))
    return stages


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
        self.passes = []              # [{"label", "graph"}] each `_run_pass`, in order
        self.facefusion = []          # verified final swaps, after all redraws
        self.real_faces = None        # [{"name", "box", "photos"}] for the paste, from the face pass
        self.face = None              # {"found", "redrawn", "denoise"} when it ran
        self.dress = None             # {"outfit", "passes", "head_crop", ...} when dressed
        self.refinement = None        # the Visual Critic's passes, when it ran
        self.notes = []               # things said on the way (no live progress, ...)
        self.license = ""             # a model's terms the picture carries (`headswap.LICENSE_NOTE`)
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
            if self.closed:
                return self._finish(job, "cancelled", "cancelled before it started")
            self.jobs.append(job)
            lane = self.lanes.get(job.backend["id"])
            if lane is None:
                lane = self.lanes[job.backend["id"]] = Lane(job.backend)
            lane.backend = job.backend
        with lane.cv:
            if self.closed:            # close() ran between the two locks above
                return self._finish(job, "cancelled", "cancelled before it started")
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
        job.detail = "Cancelling…"
        self.notify(job)
        lane = self.lanes.get(job.backend["id"])
        if lane is not None:
            with lane.cv:
                if job in lane.waiting:
                    lane.waiting.remove(job)
                    self._finish(job, "cancelled", "cancelled before it started")
                    return
        if job.prompt_id:
            # The caller can be Tk. Network timeouts must never hold its event loop.
            prompt_id = job.prompt_id
            def interrupt():
                try:
                    self.studio.client(job.backend).cancel_job(prompt_id)
                except ComfyError:
                    pass
            threading.Thread(target=interrupt, daemon=True).start()

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
                    # A job that reached lane.waiting between close()'s own
                    # cancel pass and this thread waking up would otherwise
                    # sit here forever, never notified as finished.
                    leftover, lane.waiting = lane.waiting, []
                    for job in leftover:
                        self._finish(job, "cancelled", "cancelled before it started")
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


def png_text(data, key, text):
    """`data` with a tEXt chunk `key`: `text` after its header (Latin-1, as
    PNG's tEXt is); anything not a PNG is handed back as it was."""
    import struct
    import zlib
    if data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        return data
    body = key.encode("latin-1") + b"\0" + text.encode("latin-1", "replace")
    chunk = (struct.pack(">I", len(body)) + b"tEXt" + body
             + struct.pack(">I", zlib.crc32(b"tEXt" + body) & 0xFFFFFFFF))
    end = 8 + 12 + struct.unpack(">I", data[8:12])[0]       # after IHDR
    return data[:end] + chunk + data[end:]


class History:
    """Every finished job: its pictures and a JSON record beside them under
    history/<date>/. The record holds the settings exactly as submitted, so
    Reuse Settings and Generate Again read it back rather than guess."""

    def __init__(self, root=None):
        self.root = root or os.path.join(studio_dir(), "history")

    def folder_for(self, when):
        return os.path.join(self.root, time.strftime("%Y-%m-%d", time.localtime(when)))

    def add(self, record, pictures):
        """record: dict; pictures: [(filename, bytes)]. -> the record, saved.
        A record's `license` is written into each PNG as its Comment, so
        the terms go wherever the file goes."""
        folder = self.folder_for(record["created_ts"])
        os.makedirs(folder, exist_ok=True)
        record["images"] = []
        for i, (name, data) in enumerate(pictures):
            ext = os.path.splitext(name)[1] or ".png"
            path = os.path.join(folder, "%s_%d%s" % (record["id"], i + 1, ext))
            if record.get("license"):
                data = png_text(data, "Comment", record["license"])
            with open(path, "wb") as f:
                f.write(data)
            record["images"].append(path)
        return self.update(record)

    def update(self, record):
        """Atomically update metadata without rewriting any image bytes."""
        folder = self.folder_for(record["created_ts"])
        path = os.path.join(folder, record["id"] + ".json")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in record.items() if k != "path"}, f, indent=2)
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
                # A face swap's kept base is left out once the swapped picture is saved.
                if (isinstance(rec, dict) and isinstance(rec.get("settings"), dict)
                        and (rec.get("finish") or {}).get("state") != "complete"):
                    rec["path"] = os.path.join(folder, name)
                    recs.append(rec)
            out.extend(sorted(recs, key=lambda r: r.get("created_ts", 0), reverse=True))
            if len(out) >= limit:
                break
        return out[:limit]


AGAIN_PINNED = ("steps", "guidance", "sampler", "scheduler", "width", "height")


LOCAL_FACES = {"id": "local-facefusion", "name": "FaceFusion on this PC", "url": "",
               "release_vram": False}


def local_faces(settings):
    if settings.get("mode") == "faces":
        return True
    fix = clean_fix(settings.get("fix"))
    return (settings.get("mode") == "fix" and bool(fix["face_swap"])
            and not fix["spots"] and not fix["locks"])


def retry_faces(record):
    """Retry only the finishing pass, using the saved picture and profile snapshot."""
    finish = record.get("finish") or {}
    if not record.get("images") or not finish.get("profiles"):
        raise ComfyError("This picture has no saved face pass to retry. Use Fix a spot.")
    return {"mode": "faces", "batch": 1, "seed": record.get("seed", -1),
            "face_finish": {"images": list(record["images"]),
                            "profiles": copy.deepcopy(finish["profiles"]),
                            "record": record.get("path")}}


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
# so the face reference is them and nobody beside them. The photo can be
# cropped first (`crop_region`): SAM3 then looks only there, and the boxes,
# the preview and the cut-out are all of the crop.
PEOPLE_PROMPT = "person:8"
CUTOUT_PAD = 0.04             # of the box's larger side, added around the person
CUTOUT_BG = 0xFFFFFF
CROP_MIN = 32                 # px a side: a smaller drag is a slip, not a crop


def people_graph(image, sam3, region=None):
    """Everyone SAM3 finds in `image` (a LoadImage name), the picture's size,
    and a PNG of it for Tk to show (it reads no JPEG). With `region` (x, y,
    width, height) all three are of that crop of it; with no `sam3` it
    finds nobody, only says the size and sends the PNG."""
    g = {"1": {"class_type": "LoadImage", "inputs": {"image": image}}}
    pic = ["1", 0]
    if region:
        g["c"] = {"class_type": "ImageCropV2", "inputs": {"image": pic, "crop_region": region}}
        pic = ["c", 0]
    if sam3:
        g.update({
            "2": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": sam3}},
            "3": {"class_type": "CLIPTextEncode", "inputs": {"text": PEOPLE_PROMPT,
                                                             "clip": ["2", 1]}},
            "4": {"class_type": "SAM3_Detect", "inputs": {
                "model": ["2", 0], "image": pic, "conditioning": ["3", 0],
                "threshold": 0.3, "refine_iterations": 0, "individual_masks": True}},
            "5": {"class_type": "PreviewAny", "inputs": {"source": ["4", 1]}}})
    g.update({
        "6": {"class_type": "GetImageSize", "inputs": {"image": pic}},
        "7": {"class_type": "PreviewAny", "inputs": {"source": ["6", 0]}},
        "8": {"class_type": "PreviewAny", "inputs": {"source": ["6", 1]}},
        "9": {"class_type": "PreviewImage", "inputs": {"images": pic}}})
    return g


def crop_region(x0, y0, x1, y1, width, height):
    """A dragged rectangle (corners in any order, the picture's pixels) as a
    crop_region inside a width x height picture; None when it is too small
    to be meant (under CROP_MIN px a side) or covers the whole picture."""
    x0, x1 = sorted((max(0, min(width, int(x0))), max(0, min(width, int(x1)))))
    y0, y1 = sorted((max(0, min(height, int(y0))), max(0, min(height, int(y1)))))
    if x1 - x0 < CROP_MIN or y1 - y0 < CROP_MIN:
        return None
    if (x0, y0, x1, y1) == (0, 0, width, height):
        return None
    return {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0}


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


def cutout_graph(image, sam3, region, prefix="studio_person", box=None):
    """`image` cropped to `region`, the main person in it masked by SAM3 and
    laid on white, saved under `prefix`. A neighbour's shoulder inside the
    crop is not the main person, so it goes white with the background.
    `box` (x, y, w, h in the crop's pixels) is the person picked: without it
    SAM3's "person:1" is whoever it scores highest, which in a tight crop
    was once a neighbour's hand at the edge rather than the woman filling
    it (2026-09-26)."""
    g = {
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
    if box:
        x, y, w, h = box
        g["5"]["inputs"]["bboxes"] = {"x": int(x), "y": int(y), "width": int(w),
                                      "height": int(h)}
    return g


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
            try:
                entry = client.get_history(pid)
            except Unreachable as e:
                # ComfyUI stops answering while it stages a model (SAM3's
                # first run): busy, as in health() - a refusal is not.
                if "timed out" not in str(e).lower():
                    raise
                entry = None
            if entry:                 # history holds a prompt once it has finished
                status = entry.get("status") or {}
                if status.get("status_str") == "error":
                    raise ComfyError("%s: %s" % (client.backend["name"], "; ".join(
                        run_errors(entry, graph)) or "the run failed"))
                return entry
            time.sleep(0.5)
        try:
            stopped = client.cancel_job(pid)
        except ComfyError:
            stopped = False
        raise ComfyError("%s took over %d s; %s." % (
            client.backend["name"], timeout, "the run was stopped there" if stopped
            else "it could not be stopped there and may still be running"))

    def look_at(self, path):
        """The picture at `path` as the crop window needs it: dict with
        backend, sam3, the uploaded name, width, height and preview (PNG
        bytes, since Tk reads no JPEG). No SAM3 run. Network I/O: off the
        UI thread."""
        backend, sam3 = self._sam3()
        c = self.client(backend)
        name = c.upload_image(path)
        said = people_found(self._run_quick(c, people_graph(name, None)))
        if said is None:
            raise ComfyError("ComfyUI could not read the picture.")
        width, height, _, pv = said
        return {"backend": backend, "sam3": sam3, "image": name, "width": width,
                "height": height, "preview": c.fetch(pv) if pv else None}

    def find_people(self, path, region=None):
        """Everyone in the picture at `path`, or in its `region` (a
        crop_region). -> dict with backend, sam3, the uploaded name, region,
        width, height, boxes (largest first) and preview (PNG bytes), the
        last four of the crop when there is one. Network I/O: off the UI
        thread."""
        backend, sam3 = self._sam3()
        c = self.client(backend)
        name = c.upload_image(path)
        said = people_found(self._run_quick(c, people_graph(name, sam3, region)))
        if said is None:
            raise ComfyError("SAM3 said nothing about the picture.")
        width, height, boxes, pv = said
        return {"backend": backend, "sam3": sam3, "image": name, "region": region,
                "width": width, "height": height, "boxes": boxes,
                "preview": c.fetch(pv) if pv else None}

    def _sam3(self):
        found = self.sam3_backend()
        if found is None:
            raise ComfyError("No online ComfyUI has a SAM3 checkpoint (a file with sam3 "
                             "in its name under checkpoints).")
        return found

    def find_parts(self, path, kind):
        """Fix a spot's one-click Find: the hands, faces or accessories
        (FIX_FIND[kind]) in the picture at `path` -> spots (found_spots).
        Network I/O: off the UI thread."""
        found = self.sam3_backend()
        if found is None:
            raise ComfyError("No online ComfyUI has a SAM3 checkpoint (a file with sam3 "
                             "in its name under checkpoints).")
        backend, sam3 = found
        c = self.client(backend)
        prompts = FIX_FIND.get(kind) or FIX_FIND["hand"]
        said = parts_found(self._run_quick(c, parts_graph(c.upload_image(path), sam3,
                                                          prompts)), len(prompts),
                           [p.split(":")[0] for p in prompts])
        if said is None:
            raise ComfyError("SAM3 said nothing about the picture.")
        width, height, boxes = said
        return found_spots(width, height, boxes, kind)

    def cut_person(self, found, box):
        """The person in `box` of what find_people found, on white: PNG bytes."""
        c = self.client(found["backend"])
        region = cutout_region(box, found["width"], found["height"])
        inner = (box[0] - region["x"], box[1] - region["y"], box[2], box[3])
        crop = found.get("region")
        if crop:                  # the box is of the crop; the cut is of the photo
            region = dict(region, x=region["x"] + crop["x"], y=region["y"] + crop["y"])
        entry = self._run_quick(c, cutout_graph(found["image"], found["sam3"], region,
                                                box=inner))
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
        if local_faces(settings):
            return dict(LOCAL_FACES), "FaceFusion on this PC; no ComfyUI needed."
        model = self.lib.get("models", settings.get("model"))
        label = model["label"] if model else settings.get("model")
        if settings.get("backend") not in (None, "", "auto"):
            b = self.backend(settings["backend"])
            if b is None:
                return None, "No backend called %r." % settings["backend"]
            return b, "%s (chosen by hand)." % b["name"]
        role = preset_info(self.lib, settings.get("preset"))["role"]
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
            why.append("preferred for %s" % preset_info(self.lib, settings.get("preset"))[
                "label"]
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
        if local_faces(settings):
            return [dict(LOCAL_FACES) for _ in range(count)]
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
            role = preset_info(self.lib, settings.get("preset"))["role"]
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
        import apps.image_studio.facefusion as facefusion
        if local_faces(settings):
            p = Plan()
            profiles, images = self.face_inputs(settings)
            p.errors.extend(facefusion.profile_errors(profiles))
            if not profiles:
                missing = clean_fix(settings.get("fix"))["face_swap"]
                p.errors.append("The identity %s is missing. Choose a person on the People tab."
                                % (missing or "for this picture"))
            if not images or any(not os.path.isfile(path) for path in images):
                p.errors.append("The source picture is missing. Choose an existing picture in History.")
            return p
        b = backend or next((x for x in self.backends() if x["enabled"]), None)
        if b is None:
            return None
        p = compose(settings, self.lib, b, self.inventories.get(b["id"]),
                    self.workflow_loader, self.nodes.get(b["id"]))
        if not (p.workflow or {}).get("multi_identity"):
            p.errors.extend(facefusion.profile_errors(facefusion.selected(self.lib, settings)))
        return p

    def face_inputs(self, settings):
        if settings.get("mode") == "faces":
            finish = settings.get("face_finish") or {}
            return copy.deepcopy(finish.get("profiles") or []), list(finish.get("images") or [])
        fix = clean_fix(settings.get("fix"))
        profile = self.lib.get("identities", fix["face_swap"])
        profiles = [dict(profile)] if profile else []
        if profiles and fix["face_point"] is not None:
            profiles[0]["target_point"] = fix["face_point"]
        return profiles, [fix["image"]]

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
        if settings.get("mode") == "blend":
            import apps.image_studio.blend as blend
            return blend.submit(self, settings)
        s = copy.deepcopy(settings)
        count = max(1, min(int(s.get("batch") or 1), 64))
        seed = int(s.get("seed", -1))
        if seed < 0:
            s["seed_mode"] = "random"
            seed = random.randint(0, MAX_SEED)
        else:
            s["seed_mode"] = "fixed"
        jobs = []
        backends = self.pick_backends(s, count)
        for i, b in enumerate(backends):
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
        if local_faces(job.settings):
            return self.run_profile_swap(job, clean_fix(job.settings.get("fix")), say)
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
        if job.settings.get("mode") == "fix":
            return self.run_fix(job, client, say)
        if job.settings.get("mode") == "blend":
            import apps.image_studio.blend as blend
            return blend.run_job(self, job, client, say)
        plan = compose(job.settings, self.lib, b, self.inventories.get(b["id"]),
                       self.workflow_loader, self.nodes.get(b["id"]))
        job.plan = plan
        import apps.image_studio.facefusion as facefusion
        profiles = ([] if (plan.workflow or {}).get("multi_identity") else
                    facefusion.selected(self.lib, job.settings))
        plan.errors.extend(facefusion.profile_errors(profiles))
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
        if not plan.workflow.get("multi_identity") and not self._faces_into_picture(
                job, client, plan, graph, types, values.get("width"), values.get("height")):
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
                if job.real_faces and not profiles and not job.cancel.is_set():
                    files = self._real_faces(job, client, plan, values, files, say)
        if (job.settings.get("auto_refine") and not plan.workflow.get("multi_identity")
                and not job.cancel.is_set()):
            files = self._refine(job, client, plan, values, files, say)
        if job.cancel.is_set():
            return self.queue._finish(job, "cancelled")
        say("decoding", "fetching the picture from %s" % b["name"], None)
        try:
            pictures = [(f["filename"], client.fetch(f)) for f in files]
        except ComfyError as e:
            return self.queue._finish(job, "failed", "The picture was made but could not be "
                                      "fetched from %s: %s" % (b["name"], e))
        if profiles:
            pictures = self.finish_profiles(
                job, graph, pictures, profiles, say,
                before=lambda made: self._head_swap(job, client, values, made, profiles, say))
            if pictures is None:
                return
        if not plan.workflow.get("multi_identity"):
            pictures = self._finish_passes(job, client, plan, values, pictures, profiles, say)
        self.save_result(job, self.record_for(job, graph), pictures)
        job.progress = 1.0
        self.queue._finish(job, "complete",
                           "; ".join(plan.warnings[:1]) if plan.warnings else "")

    def finish_profiles(self, job, graph, pictures, profiles, say, checkpoint=None, before=None):
        """Durable checkpoint before a fallible finishing step; never lose the base.
        A retry passes the checkpoint it retries, already saved. `before`
        (pictures -> pictures; Generate's head swap) runs after the
        checkpoint and before the faces, so the checkpoint is the picture
        as it was generated."""
        if checkpoint is None:
            record = copy.deepcopy(self.record_for(job, graph))
            record["id"] += "-generated"
            record["finish"] = {"state": "pending", "profiles": copy.deepcopy(profiles)}
            record["notes"] = record["notes"] + ["Generated picture saved before the final face swap."]
            job.record = self.history.add(record, pictures)
        else:
            record = job.record = checkpoint
        job.outputs = list(job.record["images"])
        say("face_swap", "Generated picture saved; applying faces")
        try:
            if before is not None:
                pictures = before(pictures)
                if job.cancel.is_set():
                    raise RuntimeError("Face swap cancelled.")
            result = self._apply_profiles(job, pictures, profiles, say)
            if job.cancel.is_set():
                raise RuntimeError("Face swap cancelled.")
            return result
        except (RuntimeError, OSError, ValueError) as error:
            detail = str(error) + " Generated picture kept in History. Retry face swap or use Fix a spot."
            record["finish"].update(state="cancelled" if job.cancel.is_set() else "failed", error=str(error))
            self._update_checkpoint(record)
            self.queue._finish(job, "cancelled" if job.cancel.is_set() else "failed", detail)
            return None

    def _update_checkpoint(self, record):
        try:
            self.history.update(record)
        except OSError as error:
            doctor.log_error("Could not update face checkpoint metadata: %s" % error)

    def save_result(self, job, record, pictures):
        checkpoint = job.record
        job.record = self.history.add(record, pictures)
        job.outputs = list(job.record["images"])
        if checkpoint and checkpoint.get("finish"):
            checkpoint["finish"].update(state="complete", results=list(job.outputs))
            self._update_checkpoint(checkpoint)

    def _head_swap(self, job, client, values, pictures, profiles, say):
        """Before the face swap, on the lane's thread: each profile's whole
        head redrawn from their first photo by FLUX.2 Klein (`headswap`), so
        the face FaceFusion swaps after lands on their head, hair and
        glasses and not the generated stranger's. Unless
        settings["head_swap"] is off. -> the pictures; those given when the
        backend cannot, on any failure, and on a cancel - a head swap is an
        extra, the picture is never lost to it."""
        import apps.image_studio.headswap as headswap
        if job.settings.get("head_swap", True) is False:
            return pictures
        b = job.backend
        sam = self.sam3_on(b)
        try:
            short = headswap.lacks(self.inventories.get(b["id"]),
                                   client.node_types() if sam else None)
        except ComfyError:
            short = []
        if not sam or short:
            job.notes.append("No head swap before the face swap: %s." % (
                "%s has no SAM3 checkpoint" % b["name"] if not sam else
                "%s lacks %s" % (b["name"], ", ".join(short))))
            return pictures
        folder = os.path.join(self.lib.root, "finish")
        out = []
        try:
            os.makedirs(folder, exist_ok=True)
            photos = {}
            for n, (filename, data) in enumerate(pictures):
                if job.cancel.is_set():
                    out.append((filename, data))
                    continue
                path = os.path.join(folder, "%s_head_%d.png" % (job.id, n))
                with open(path, "wb") as fh:
                    fh.write(data)
                image = client.upload_image(path)
                say(headswap.STATUS, "Finding the heads", None)
                job.prompt_id = client.queue_workflow(parts_graph(image, sam, [headswap.FIND]))
                entry = client.listen_for_progress(job.prompt_id, lambda kind, d: None,
                                                   stop=job.cancel.is_set)
                said = parts_found(entry or {}, 1)
                heads = []
                if said is not None:
                    width, height, boxes = said
                    for profile, face in headswap.targets(width, height, boxes, profiles):
                        photo = profile["references"][0]
                        if photo not in photos:
                            photos[photo] = client.upload_image(photo)
                        own, why = head_lora(self.lib, profile, b["id"],
                                             self.inventories.get(b["id"]))
                        if why and why not in job.notes:
                            job.notes.append(why)
                        heads.append(dict(own, crop=headswap.head_crop(width, height, face),
                                          photo=photos[photo], name=profile["name"]))
                if not heads:
                    if not job.cancel.is_set():
                        job.notes.append("SAM3 found no face of a chosen person, so no head "
                                         "swap was made before the face swap.")
                    out.append((filename, data))
                    continue
                graph = headswap.head_graph(image, heads, int(values.get("seed") or 0),
                                            values["filename_prefix"] + "_head", sam)
                files = self._run_pass(job, client, graph, say, headswap.LABEL,
                                       status=headswap.STATUS)
                if files is None:         # cancelled: the picture as it was
                    out.append((filename, data))
                    continue
                out.append((filename, client.fetch(files[0])))
                job.notes.append("Head swap before the face swap: %s redrawn from their "
                                 "photo by FLUX.2 Klein 9B." % ", ".join(
                                     "%s's head%s" % (h["name"], " (with their head LoRA)"
                                                      if h.get("lora") else "")
                                     for h in heads))
                job.license = headswap.LICENSE_NOTE
            return out
        except (ComfyError, Unreachable, OSError) as e:
            job.notes.append("The head swap before the face swap could not run (%s); the "
                             "faces are swapped on the picture as it was generated." % e)
            return pictures

    def _apply_profiles(self, job, pictures, profiles, say):
        """The final face swap, by FaceFusion (`facefusion.SWAP_MODEL`)."""
        import apps.image_studio.facefusion as facefusion
        result = []
        for filename, data in pictures:
            for index, profile in enumerate(profiles):
                if job.cancel.is_set():
                    raise RuntimeError("Face swap cancelled.")
                say("face_swap", "Applying %s's face" % profile["name"], None)
                options = ({"face_index": index, "face_count": len(profiles)}
                           if len(profiles) > 1 else {})
                data, report = facefusion.swap(data, profile, stop=job.cancel.is_set, **options)
                job.facefusion.append(report)
                job.notes.append("%s: FaceFusion applied; zero pixels changed outside the face mask."
                                 % profile["name"])
            result.append((os.path.splitext(filename)[0] + ".png", data))
        return result

    def run_profile_swap(self, job, fix, say):
        """An existing picture can receive a profile without a ComfyUI redraw."""
        profiles, images = self.face_inputs(job.settings)
        job.started = time.time()
        job.plan = Plan()
        job.plan.workflow = {"id": "facefusion", "label": "Apply identity"}
        job.plan.values = {"seed": job.settings.get("seed")}
        job.plan.prompt = "Apply faces: " + ", ".join(p["name"] for p in profiles)
        checkpoint = None
        try:
            problems = self.preview(job.settings).errors
            if problems:
                raise RuntimeError(" ".join(problems))
            pictures = []
            for path in images:
                with open(path, "rb") as source:
                    pictures.append((os.path.basename(path), source.read()))
            saved = (job.settings.get("face_finish") or {}).get("record")
            if saved:                 # a retry finishes the checkpoint it came from
                with open(saved, encoding="utf-8") as f:
                    checkpoint = dict(json.load(f), path=saved)
        except (OSError, RuntimeError, ValueError) as e:
            return self.queue._finish(job, "cancelled" if job.cancel.is_set() else "failed", str(e))
        pictures = self.finish_profiles(job, None, pictures, profiles, say, checkpoint)
        if pictures is None:
            return
        record = self.record_for(job, None)
        record["fix"] = fix
        self.save_result(job, record, pictures)
        job.progress = 1.0
        self.queue._finish(job, "complete")

    # ------------------------------------------------------------ fix a spot
    @staticmethod
    def fix_base(settings):
        """The settings a fix redraws with: the picture's own, less what made
        the picture and is not wanted in a close-up (references, pose, the
        face pass, the critic)."""
        s = {k: v for k, v in (settings or {}).items() if k not in ("mode", "fix")}
        s.update(references={}, item_refs={}, pose=None, composition=None,
                 face_detail=False, auto_refine=False, batch=1)
        return s

    def _fix_oval(self, client):
        """The fix oval, uploaded to `client`. The crop reaches FIX_CONTEXT
        times past the spot, so the model redraws it seeing the photo round
        it; the oval is shrunk to match, so only the spot changes."""
        oval = os.path.join(self.lib.root, "fix_oval.png")
        os.makedirs(self.lib.root, exist_ok=True)
        with open(oval, "wb") as fh:
            fh.write(oval_png(scale=1.0 / FIX_CONTEXT, centre=0.5))
        return client.upload_image(oval)

    def _finish_passes(self, job, client, plan, values, pictures, profiles, say):
        """The end of Generate, on the lane's thread, by the fix machinery on
        the picture's own model: after FaceFusion (`profiles`) each swapped
        face's eyes (EYE_WHAT); then the hands of a picture of people
        (HAND_WHAT, `real_hands`; the hands pass, unless
        settings["hand_pass"] is off); then the swapped faces' glasses last
        (GLASSES_WHAT), so nothing is drawn over them - when the swap
        painted over them (`glasses_pass`).
        -> the pictures; those given when a pass cannot run, fails or is
        cancelled - the picture is never lost to its finish."""
        hands = hand_pass(job.settings)
        if job.settings.get("hand_pass", True) is not False and not hands:
            job.notes.append("No hands pass: the picture's words name no person.")
        specs = bool(profiles) and glasses_pass(job.settings, job.facefusion)
        if profiles and not specs and job.settings.get("glasses_pass") is not False:
            job.notes.append("No glasses pass: the face swap went behind what was in front "
                             "of the face, so glasses are as they were drawn.")
        if not profiles and not hands:
            return pictures
        kinds = (["eye"] if profiles else []) + (["hands"] if hands else []) + (
            ["glasses"] if specs else [])
        named = "%s pass%s" % (", ".join(kinds[:-1]) + " or " + kinds[-1] if len(kinds) > 1
                               else kinds[0], " after the face swap" if profiles else "")
        b = job.backend
        sam = self.sam3_on(b)
        why, types = "", set()
        if not plan.workflow.get("face_detail"):
            why = "the %s workflow has no redraw section" % plan.workflow.get("label")
        elif not sam:
            why = "%s has no SAM3 checkpoint" % b["name"]
        else:
            try:
                types = set(client.node_types())
                lacks = FACE_NODES - types
            except ComfyError:
                lacks = set()
            if lacks:
                why = "%s's ComfyUI lacks %s" % (b["name"], ", ".join(sorted(lacks)))
        if why:
            job.notes.append("No %s: %s." % (named, why))
            return pictures
        asked = list(zip(FINISH_FIND, FINISH_WORDS))
        asked = ([a for a in asked if specs or a[1] != "glasses"] if profiles else []) + (
            [(HAND_FIND, "hand")] if hands else [])
        find, words = [a[0] for a in asked], [a[1] for a in asked]
        looking = (["eyes"] if profiles else []) + (["hands"] if hands else []) + (
            ["glasses"] if specs else [])
        v = dict(values, sam3=sam, match_tone=None)
        # A hand keeps the picture's grade; a swapped face gets no curves,
        # which posterize its skin.
        hand_tone = FIX_TONE if TONE_NODE in types else None
        folder = os.path.join(self.lib.root, "finish")
        out = []
        try:
            os.makedirs(folder, exist_ok=True)
            oval = self._fix_oval(client)
            for n, (filename, data) in enumerate(pictures):
                if job.cancel.is_set():
                    out.append((filename, data))
                    continue
                path = os.path.join(folder, "%s_%d.png" % (job.id, n))
                with open(path, "wb") as fh:
                    fh.write(data)
                image = client.upload_image(path)
                say("eyes" if profiles else "hands",
                    "Finding the %s" % " and ".join(looking), None)
                job.prompt_id = client.queue_workflow(parts_graph(image, sam, find))
                entry = client.listen_for_progress(job.prompt_id, lambda kind, d: None,
                                                   stop=job.cancel.is_set)
                said = parts_found(entry or {}, len(find), words, scores=True)
                if said is None:
                    if not job.cancel.is_set():
                        job.notes.append("SAM3 said nothing about the picture, so no %s was "
                                         "made." % named)
                    out.append((filename, data))
                    continue
                width, height, boxes = said
                passes, faces, found_hands, glasses = [], [], [], []
                if profiles:
                    faces = swapped_faces(width, height,
                                          [x[:4] for x in boxes if x[4] == "face"], profiles)
                    if faces:
                        eyes = eye_spots(faces)
                        passes.append(("Eye pass", fix_areas(fix_crops(width, height, eyes),
                                                             eyes),
                                       EYE_DENOISE, EYE_WHAT, "_eyes", None))
                    else:
                        job.notes.append("SAM3 found no swapped face, so no eye or glasses "
                                         "pass was made.")
                if hands:
                    found_hands = found_spots(width, height, real_hands(
                        width, height, [x for x in boxes if x[4] == "hand"]), "hand")
                    if found_hands:
                        crops = fix_crops(width, height, [
                            dict(sp, size=int(sp["size"] * FIX_CONTEXT)) for sp in found_hands])
                        passes.append(("Hands", fix_areas(crops, found_hands), HAND_DENOISE,
                                       HAND_WHAT, "_hands", hand_tone))
                    else:
                        job.notes.append("SAM3 found no hands, so no hands pass was made.")
                if faces and specs:
                    glasses = glasses_spots(width, height,
                                            [x for x in boxes if x[4] == "glasses"], faces)
                    if glasses:
                        crops = fix_crops(width, height, [
                            dict(sp, size=int(sp["size"] * FIX_CONTEXT)) for sp in glasses])
                        passes.append(("Glasses", fix_areas(crops, glasses), GLASSES_DENOISE,
                                       GLASSES_WHAT, "_glasses", None))
                    else:
                        job.notes.append("SAM3 found no glasses on the swapped face.")
                done = None
                for label, crops, denoise, what, tag, tone in passes:
                    graph = face_graph(plan.workflow, dict(v, face_prompt=FIX_PROMPT % what,
                                                           face_denoise=denoise,
                                                           match_tone=tone),
                                       plan.loras, image, crops, oval,
                                       values["filename_prefix"] + tag)
                    files = self._run_pass(job, client, graph, say, label,
                                           status=tag.lstrip("_"))
                    if files is None:     # cancelled: keep what is finished
                        break
                    done = files[0]
                    image = "%s%s [%s]" % (done["subfolder"] + "/" if done.get("subfolder")
                                           else "", done["filename"], done.get("type") or "output")
                    count, said = {
                        "_eyes": (len(faces), "Eye pass after the face swap: %d face%s, "
                                              "denoise %s."),
                        "_hands": (len(found_hands), "Hands pass: %d hand%s redrawn, "
                                                     "denoise %s."),
                        "_glasses": (len(glasses), "Glasses redrawn last: %d pair%s, "
                                                   "denoise %s.")}[tag]
                    job.notes.append(said % (count, "" if count == 1 else "s", denoise))
                if done is not None:
                    data = client.fetch(done)
                out.append((filename, data))
        except (ComfyError, TemplateError, OSError) as e:
            job.notes.append("The %s failed (%s); the picture is kept as it was before it."
                             % (named, e))
            return pictures
        return out

    def _outline_masks(self, job, client, crops, tag):
        """Each crop with a freehand outline gets `shape`: the outline drawn
        as a mask picture at the crop's size, uploaded to `client`."""
        for i, crop in enumerate(crops):
            if not crop.get("outline"):
                continue
            path = os.path.join(self.lib.root, "fix_shapes", "%s_%s_%d.png" % (job.id, tag, i))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as fh:
                fh.write(outline_png(crop["outline"], crop["width"], crop["height"]))
            crop["shape"] = client.upload_image(path)

    def run_fix(self, job, client, say):
        """Fix a spot, on the lane's thread: the squares the user clicked on a
        finished picture redrawn with its own model, LoRAs and prompt, and
        blended back. Squares given a photo are swapped for what the photo
        shows by Qwen-Image-Edit first (`swap_graph`), and the rest redrawn
        on that. With a `face_swap` identity the fix ends by swapping its
        face (from all its reference pictures) onto the picture's biggest face
        (`_face_swap_graph`); it may be the only step. A new picture in the
        history; the old one is kept."""
        b, s = job.backend, job.settings
        fix = clean_fix(s.get("fix"))
        src = fix["image"]
        around = fix["around_head"]
        swaps = [sp for sp in fix["spots"] if sp.get("photo")]
        plain = [sp for sp in fix["spots"] if not sp.get("photo")]
        face, who, refs = "", None, []
        problem = ""
        if fix["face_swap"]:
            who = self.lib.get("identities", fix["face_swap"])
            if who is None:
                problem = "The identity %s to swap the face from is gone." % fix["face_swap"]
            elif not who["references"]:
                problem = ("%s has no reference picture, so there is no face to swap in."
                           % who["name"])
            else:
                face, refs = who["name"], list(who["references"])
        if problem:
            pass
        elif not src or not os.path.isfile(src):
            problem = "The picture to fix (%s) is not on this PC." % (src or "none")
        elif around and (not fix["locks"] or fix["spots"] or fix["face_swap"]):
            problem = "Mark the head to keep; this mode cannot also redraw spots or swap faces."
        elif around and not fix["words"]:
            problem = "Describe the body and scene to generate around the head."
        elif not around and not fix["spots"] and not face:
            problem = "Click the part of the picture to redraw."
        else:
            gone = [sp["photo"] for sp in swaps if not os.path.isfile(sp["photo"])]
            if gone:
                problem = "The photo %s is not on this PC." % gone[0]
            elif face:
                refs = [r for r in refs if os.path.isfile(r)]
                if not refs:
                    problem = "None of %s's reference pictures is on this PC." % who["name"]
        size = file_size_of(src) if not problem else None
        if size is None and not problem:
            problem = "Could not read the size of %s." % src
        plan = swap_wf = None
        if not problem:
            base = self.fix_base(s)
            if around:
                base.update(identities=[], character="", scene_faces={}, hand_pass=False)
            plan = compose(base, self.lib, b, self.inventories.get(b["id"]),
                           self.workflow_loader, self.nodes.get(b["id"]))
            job.plan = plan
            if plan.errors:
                problem = " ".join(plan.errors)
            elif (plain or around) and not plan.workflow.get("face_detail"):
                problem = ("The %s workflow has no redraw section, so its pictures cannot "
                           "be fixed. Choose a redraw-capable model such as Z-Image HQ."
                           % plan.workflow.get("label"))
        if face and not problem and not self.sam3_on(b):
            problem = ("The face swap finds the face with SAM3, and %s has no SAM3 "
                       "checkpoint." % b["name"])
        if (swaps or face) and not problem:
            try:
                swap_wf = self.workflow_loader(DRESS_WORKFLOW)
            except TemplateError as e:
                problem = str(e)
            else:
                nodes = self.nodes.get(b["id"])
                missing = dress_lacks(swap_wf, {"clothes": []}, self.inventories.get(b["id"]),
                                      nodes)
                if nodes is not None and "SolidMask" not in nodes:
                    missing.append({"text": "the node SolidMask"})
                if missing:
                    problem = ("%s swapped by Qwen-Image-Edit, and %s lacks %s." % (
                        "A spot with a photo is" if swaps else "The face is", b["name"],
                        _and([m["text"] for m in missing])))
        if problem:
            return self.queue._finish(job, "failed", problem)
        values = dict(plan.values)
        try:
            say("uploading", "uploading the picture", None)
            for var, path in plan.images.items():
                values[var] = client.upload_image(path)
            image = client.upload_image(src)
            oval = self._fix_oval(client)
            locks = lock_regions(size[0], size[1], fix["locks"])
            values["filename_prefix"] = "ImageStudio/fix_%s" % job.id
            graphs = []
            if around:
                values.update(width=size[0], height=size[1], face_prompt=fix["words"] +
                              ". Keep the existing head in its original position and size; "
                              "build the body and scene around it.",
                              face_denoise=fix["strength"], sam3=None, match_tone=0)
                graphs.append(("Generating around the locked head", around_head_graph(
                    plan.workflow, values, plan.loras, image, size[0], size[1], locks,
                    oval, values["filename_prefix"])))
            if swaps:
                crops = fix_crops(size[0], size[1], [dict(sp, size=int(sp["size"] * FIX_CONTEXT))
                                                     for sp in swaps])
                fix_areas(crops, swaps)
                self._outline_masks(job, client, crops, "swap")
                pictures = {}
                for crop, sp in zip(crops, swaps):
                    crop["photo"] = sp["photo"]
                    if sp["photo"] not in pictures:
                        pictures[sp["photo"]] = client.upload_image(sp["photo"])
                sam = self.sam3_on(b) if any(c.get("word") for c in crops) else None
                sv = dict(dress_values(swap_wf, b), seed=values["seed"])
                graphs.append(("Swapping %d spot%s from %s" % (
                    len(swaps), "" if len(swaps) == 1 else "s",
                    "a photo" if len(pictures) == 1 else "photos"),
                    swap_graph(swap_wf, sv, image, crops, pictures,
                               [swap_prompt(fix, sp) for sp in swaps], oval,
                               values["filename_prefix"] + ("_swap" if plain else ""),
                               sam3=sam, locks=locks)))
            if plain:
                if fix["tone"] and TONE_NODE not in set(client.node_types()):
                    job.notes.append("%s's ComfyUI has no %s (comfy_nodes/studio_matchtone), "
                                     "so the redraw's colours are not matched to the picture."
                                     % (b["name"], TONE_NODE))
                else:
                    values["match_tone"] = fix["tone"]
                # A face is blended back through its true shape - SAM3's face in
                # the picture and in the redraw, plus the box Find found - not
                # the square's oval, so no redrawn skin or hair spills round it.
                sam = self.sam3_on(b) if fix["target"] == "face" else None
                values["sam3"] = sam
                if fix["target"] == "face" and not sam:
                    job.notes.append("%s has no SAM3 checkpoint, so the face is blended back "
                                     "through an oval, not its own shape." % b["name"])
                crops = fix_crops(size[0], size[1], [dict(sp, size=int(sp["size"] * FIX_CONTEXT))
                                                     for sp in plain], head=bool(sam))
                fix_areas(crops, plain)
                self._outline_masks(job, client, crops, "redraw")
                if any(c.get("word") for c in crops):    # found things redrawn by their outline
                    values["sam3"] = self.sam3_on(b)
                boxes = [sp.get("box") for sp in plain] if sam else None
                values["face_prompt"] = fix_prompt(fix, plan.prompt)
                values["face_denoise"] = fix["strength"]
                at = {id(sp): i for i, sp in enumerate(plain)}
                place = {n: at[id(sp)] for n, sp in enumerate(fix["spots"]) if id(sp) in at}

                def redraw(picture, faults=None, n=0):
                    """The spots redrawn on `picture`; with `faults` (the
                    critic's, pass `n`) only theirs, each harder than the
                    try before and from the critic's words for it."""
                    if faults is None:
                        return face_graph(plan.workflow, values, plan.loras, picture, crops,
                                          oval, values["filename_prefix"], boxes=boxes,
                                          locks=locks, mask_word=FIX_FACE_MASK)
                    which = [place[f["spot"]] for f in faults]
                    what = [", ".join(x for x in (f["correction"].strip().rstrip("."),
                                                  fix["words"], FIX_TARGETS[fix["target"]])
                                      if x) or "this detail" for f in faults]
                    return face_graph(
                        plan.workflow,
                        dict(values, seed=(int(values["seed"]) + 1000 * n) % (MAX_SEED + 1)),
                        plan.loras, picture, [crops[i] for i in which], oval,
                        "%s_redo%d" % (values["filename_prefix"], n),
                        faces=[{"prompt": FIX_PROMPT % w, "denoise": critic.harder(
                            fix["strength"], f["tries"], FIX_STRENGTHS["strong"])}
                            for w, f in zip(what, faults)],
                        boxes=[boxes[i] for i in which] if boxes else None, locks=locks,
                        mask_word=FIX_FACE_MASK)
                # Built once the swap run (if any) has made the picture it starts from.
                graphs.append(("Redrawing " + fix_words(dict(fix, spots=plain)), redraw))
            if face:
                photos = {r: client.upload_image(r) for r in refs}
                sv = dict(dress_values(swap_wf, b), seed=values["seed"])
                tone = fix["tone"]
                if tone and "ColorTransfer" not in set(client.node_types()):
                    tone = 0.0
                    job.notes.append("%s's ComfyUI has no ColorTransfer (update it), so the "
                                     "swapped face keeps the photos' colours." % b["name"])
                graphs.append((FACE_SWAP_LABEL % who["name"],
                               lambda picture: self._face_swap_graph(
                                   job, client, picture, refs, photos, swap_wf, sv, oval,
                                   values["filename_prefix"] + "_face", locks, tone)))
        except (ComfyError, TemplateError, OSError) as e:
            return self.queue._finish(job, "failed", "Could not set up the fix on %s: %s"
                                      % (b["name"], e))
        if b.get("shares_llm_gpu") and self.make_room is not None:
            say(detail="clearing LM Studio off the GPU")
            try:
                self.make_room(b)
            except Exception as e:
                job.notes.append("Could not clear the shared GPU (%s); this may be slow." % e)
        if job.cancel.is_set():
            return self.queue._finish(job, "cancelled")
        files = graph = None
        face_done = False
        for label, graph in graphs:
            if callable(graph):
                try:
                    graph = graph(image)
                except (TemplateError, ComfyError) as e:
                    if job.cancel.is_set():
                        return self.queue._finish(job, "cancelled")
                    return self.queue._finish(job, "failed", "Could not set up the fix on "
                                              "%s: %s" % (b["name"], e))
                if graph is None:         # the face swap found no face
                    if files is None:
                        return self.queue._finish(job, "failed", "SAM3 found no face in "
                                                  "the picture to swap.")
                    graph = job.graph
                    job.notes.append("SAM3 found no face after the fix, so none was "
                                     "swapped.")
                    continue
            job.graph = graph
            try:
                files = self._run_pass(job, client, graph, say, label)
            except ComfyError as e:
                if job.cancel.is_set():
                    return self.queue._finish(job, "cancelled")
                return self.queue._finish(job, "failed", "The fix failed on %s: %s"
                                          % (b["name"], e))
            if files is None or job.cancel.is_set():
                return self.queue._finish(job, "cancelled")
            face_done = bool(face) and label == FACE_SWAP_LABEL % who["name"]
            f = files[0]              # the next run starts from this one's picture
            image = "%s%s [%s]" % (f["subfolder"] + "/" if f.get("subfolder") else "",
                                    f["filename"], f.get("type") or "output")
        marks = []
        if fix["spots"] and not around:
            marks = [dict(f, action="FIX", denoise=fix["strength"]) for f in
                     critic.user_faults(fix["spots"], fix["target"], fix["note"])]
        if fix["check"] and marks:
            # The user's spots are faults the critic is asked after, their
            # notes what was wrong; what is still there is redrawn again.
            files = self._refine(job, client, plan, values, files, say, marked={
                "faults": marks, "redo": redraw if plain else None,
                "strength": fix["strength"],
                "crops": fix_crops(size[0], size[1], fix["spots"])})
            if job.cancel.is_set():
                return self.queue._finish(job, "cancelled")
        if marks:
            # What the user marked is filed whether or not the critic was
            # asked: a fault of this model and person, and, where the critic
            # had passed the picture, something it is blind to.
            passed = (record_of_picture(src) or {}).get("refinement")
            people = [x.get("id") if isinstance(x, dict) else x
                      for x in s.get("identities") or []]
            self._learn(lambda led: critic.note_marked(
                led, src, marks, _str(s.get("model")), people, s.get("scene") or "", passed))
        say("decoding", "fetching the picture from %s" % b["name"], None)
        try:
            pictures = [(f["filename"], client.fetch(f)) for f in files]
        except ComfyError as e:
            return self.queue._finish(job, "failed", "The picture was made but could not be "
                                      "fetched from %s: %s" % (b["name"], e))
        if fix["spots"]:
            plan.notes.append("Fixed %s at denoise %s from %s." % (
                fix_words(fix), fix["strength"], os.path.basename(src)))
        if swaps:
            plan.notes.append("Swapped from %s by Qwen-Image-Edit." % _and(
                sorted({os.path.basename(sp["photo"]) for sp in swaps})))
        if face_done:
            plan.notes.append("%s's face swapped in last by Qwen-Image-Edit, from %d "
                              "reference picture%s." % (who["name"], len(refs),
                                                        "" if len(refs) == 1 else "s"))
        rec = self.record_for(job, graph)
        rec["prompt"] = "Fix %s: %s" % (fix_words(fix), plan.prompt)
        rec["fix"] = fix
        job.record = self.history.add(rec, pictures)
        job.outputs = list(job.record["images"])
        job.progress = 1.0
        self.queue._finish(job, "complete")

    def _face_swap_graph(self, job, client, image, refs, photos, swap_wf, sv, oval, prefix,
                         locks, tone):
        """A fix's last run: SAM3 finds the faces in `image` (what the spots
        made, a LoadImage name) and in each of the identity's `refs` (paths;
        `photos` maps them to LoadImage names). The picture's biggest face -
        its subject - is swapped by swap_graph for the face in all of them,
        each cut to its own face, and blended back through SAM3's face
        before and after. None when SAM3 finds no face in the picture.
        Network I/O: it runs the finder."""
        sam = self.sam3_on(job.backend)
        names = [image] + [photos[r] for r in refs]
        job.prompt_id = client.queue_workflow(faces_graph(names, sam))
        entry = client.listen_for_progress(job.prompt_id, lambda kind, data: None,
                                           stop=job.cancel.is_set)
        if entry is None:
            raise ComfyError("cancelled")
        said = faces_found(entry, len(names))
        if said[0] is None:
            raise ComfyError("SAM3 said nothing about the picture.")
        width, height, boxes = said[0]
        point = clean_fix(job.settings.get("fix"))["face_point"]
        if point is not None:
            import apps.image_studio.facefusion as facefusion
            normalized = [[x / width, y / height, (x + w) / width, (y + h) / height]
                          for x, y, w, h in boxes]
            try:
                boxes = [boxes[facefusion.target_face(normalized, point=point)]]
            except RuntimeError as error:
                raise ComfyError(str(error)) from error
        spots = found_spots(width, height, boxes, "face")[:1]
        if not spots:
            return None
        for sp in spots:
            sp["word"] = "face"
        items = []
        for ref, got in zip(refs, said[1:]):
            if got and got[2]:
                items.append({"path": ref, "region": head_square(
                    got[2][0], got[0], got[1], FACE_SWAP_REF_PAD)})
        if not items:                     # no face found in them: the whole pictures
            items = [{"path": r} for r in refs]
            job.notes.append("SAM3 found no face in the reference pictures, so Qwen was "
                             "shown them whole.")
        crops = fix_crops(width, height, [dict(sp, size=int(sp["size"] * FIX_CONTEXT))
                                          for sp in spots])
        fix_areas(crops, spots)
        crops[0]["photos"] = items
        where = ("picture 2" if len(items) == 1 or len(items) > SWAP_SLOTS
                 else "pictures 2 and 3")
        return swap_graph(swap_wf, sv, image, crops, photos, [FACE_SWAP_PROMPT % where],
                          oval, prefix, sam3=sam, locks=locks, tone=tone)

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

    def _head_depths(self, client, plan, layout, image, width, height, pairs, boxes, known,
                     strength):
        """{box index: {"depth", "depth_strength"}} for the faces in `pairs`
        whose person has a head shape: their head's depth over the crop,
        from the scene's camera, at `strength` as far as the drawn face
        turns the way the mannequin's does (`head_gate`, read by DWPose off
        the first run's `image`). A face turned otherwise is left to its
        words: its head drawn the wrong way round is worse than none."""
        import apps.image_studio.scene.scene as sc
        scene, _ = sc.clean_scene(layout)
        fw, fh = sc.frame_size(scene)
        kx, ky = fw / float(width), fh / float(height)
        drawn = []
        try:
            graph = {"1": {"class_type": "LoadImage", "inputs": {"image": image}},
                     "2": {"class_type": POSE_NODE, "inputs": {"image": ["1", 0]}}}
            got = client.listen_for_progress(client.queue_workflow(graph), lambda *a: None,
                                             timeout=180)
            text = ((got or {}).get("outputs") or {}).get("2", {}).get("text") or []
            drawn = json.loads(text[0]).get("people") or [] if text else []
        except (ComfyError, ValueError, IndexError, TypeError) as e:
            plan.notes.append("Head shape: the drawn faces' turn could not be read (%s)." % e)
        out = {}
        for i, crop in pairs:
            person = known[i]
            gate, why = head_gate(person.get("facing"), drawn_facing(drawn, boxes[i]))
            if gate <= 0:
                plan.notes.append("Head shape: %s's face %s, so their head shape was not "
                                  "used." % (person["name"], why))
                continue
            region = (crop["x"] * kx, crop["y"] * ky, crop["width"] * kx, crop["height"] * ky)
            data = sc.depth_crop_png(scene, region, (FACE_EDIT, FACE_EDIT), person.get("id"))
            if not data:
                continue
            path = sc._write(data, "head", os.path.join(self.lib.root, "scenes", "renders"))
            try:
                out[i] = {"depth": client.upload_image(path),
                          "depth_strength": round(strength * gate, 3),
                          "head_denoise": HEAD_DENOISE}
            except (ComfyError, OSError) as e:
                plan.warnings.append("%s's head shape could not be sent (%s)."
                                     % (person["name"], e))
                continue
            if gate < 1:
                plan.notes.append("Head shape: %s's face %s, so their head shape is used at "
                                  "half strength." % (person["name"], why))
        return out

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
        scene = faces_of(job.settings)
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
        # A person FaceFusion will swap after gets that face drawn twice
        # otherwise: PuLID's likeness redraw here is thrown away the moment
        # FaceFusion replaces the same pixels (2026-09-28, the user: "this is a
        # waste of time otherwise"). Their crop is still redrawn for the
        # waxy-face fix, just from the words, not their photo.
        import apps.image_studio.facefusion as facefusion
        swapped = {p["person_id"] for p in facefusion.selected(self.lib, job.settings)
                  if p.get("person_id")}
        likely = {i for i, p in known.items() if p.get("id") not in swapped}
        pulid, why = self._pulid(client, plan, types=None) if any(
            p.get("face") for i, p in known.items() if i in likely) else (None, "")
        if why:
            plan.warnings.append(why)
        likeness = {i for i, p in known.items() if p.get("face") and pulid and i in likely}
        skipped = {p["name"] for i, p in known.items() if p.get("face") and p.get("id") in swapped}
        if skipped:
            plan.notes.append("Face pass: %s not drawn with PuLID - FaceFusion swaps their "
                              "face after." % _and(sorted(skipped)))
        layout = job.settings.get("scene_layout")
        head_k = scene.get("head_depth", 0) or 0
        shaped = ({i for i, p in known.items() if p.get("head")}
                  if layout and head_k > 0 and (values.get("controlnet") or (
                      plan.workflow.get("defaults") or {}).get("controlnet")) else set())
        pairs = indexed_crops(width, height, boxes, keep=likeness | shaped)
        crops = [c for _, c in pairs]
        f = files[0]
        picture = "%s%s [%s]" % (f["subfolder"] + "/" if f.get("subfolder") else "",
                                  f["filename"], f.get("type") or "output")
        heads = self._head_depths(client, plan, layout, picture, width, height,
                                  [(i, c) for i, c in pairs if i in shaped], boxes, known,
                                  head_k) if shaped & {i for i, _ in pairs} else {}
        faces = []
        try:
            for i, _ in pairs:
                person = known.get(i)
                if person is None:
                    faces.append(None)
                    continue
                images = []
                if i in likeness:
                    photos = [p for p in (person.get("photos") or [person["face"]])
                             if os.path.isfile(p)] or [person["face"]]
                    images = [client.upload_image(p) for p in photos[:REFERENCE_PHOTOS_MAX]]
                words = " ".join(x for x in (person.get("words"), style) if x)
                face = dict({"words": words, "image": images[0] if images else None,
                             "images": images,
                             "denoise": scene.get("likeness") if images else None,
                             "name": person["name"]}, **heads.get(i, {}))
                # A head shape needs the redraw deep enough to move the jaw.
                deep = face.pop("head_denoise", None)
                if deep:
                    face["denoise"] = max(face["denoise"] or 0, deep,
                                          values.get("face_denoise") or 0)
                faces.append(face)
        except (ComfyError, OSError) as e:
            plan.warnings.append("A face picture could not be sent (%s); the faces are "
                                 "redrawn from the words alone." % e)
            faces = [dict(f, image=None, denoise=None) if f else None for f in faces]
            faces += [None] * (len(crops) - len(faces))
        job.face = {"found": len(boxes), "redrawn": len(crops),
                    "denoise": values.get("face_denoise"),
                    "people": [f["name"] for f in faces if f],
                    "likeness": [f["name"] for f in faces if f and f.get("image")],
                    "likeness_denoise": scene.get("likeness"),
                    "head_depth": {f["name"]: f["depth_strength"] for f in faces
                                   if f and f.get("depth")}}
        if not crops:
            plan.notes.append("Face pass: %s" % ("no face found" if not boxes else
                                                 "every face was already drawn at full size"))
            return files, None
        oval = os.path.join(self.lib.root, "face_oval.png")
        try:
            if not os.path.isfile(oval):
                os.makedirs(self.lib.root, exist_ok=True)
                with open(oval, "wb") as fh:
                    fh.write(oval_png())
            graph = face_graph(plan.workflow, values, plan.loras, picture, crops,
                               client.upload_image(oval), values["filename_prefix"] + "_faces",
                               faces=faces, pulid_file=pulid,
                               boxes=[boxes[i] for i, _ in pairs])
            say("face", "redrawing %d face%s at %d px" % (
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
            say("face", "redrawing face %s of %d · step %d of %d" % (face, n, value, total),
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
        chained = [f["name"] for f in drawn if len(f.get("images") or []) > 1]
        if chained:
            plan.notes.append("%s drawn from more than one of their photos for a stronger "
                              "match." % _and(sorted(chained)))
        if scene.get("real"):
            job.real_faces = [{"name": p["name"], "box": list(boxes[i]),
                               "photos": p.get("photos") or ([p["face"]] if p.get("face") else [])}
                              for i, p in sorted(known.items())
                              if p.get("photos") or p.get("face")]
        return files2, graph

    # ------------------------------------------------------ Visual Critic
    def _refine(self, job, client, plan, values, files, say, marked=None):
        """Automatic refinement, on the lane's thread: the vision model looks
        at the picture (studio_critic), what it finds right is left alone,
        and what it finds wrong is redrawn with the tool the fault calls for -
        the face pass's crop-redraw-blend for faces, hands and objects, the
        same at low denoise over the whole picture for a touch-up, and a new
        picture only for a structural failure. Up to `refine_passes` passes,
        stopping as soon as the critic finds nothing meaningful. Never loses
        the picture: any failure keeps the last good one. -> files.

        Every look after a pass scores it (`critic.score_fixes`): a fault
        still there is redrawn again, harder; after `critic.MAX_TRIES` it is
        left to the user; a pass that damaged the picture and mended nothing
        is taken back. So the last pass is looked at too, a look with no
        redraw after it.

        `marked` is Fix a spot's check: {"faults" (`critic.user_faults`, each
        redrawn once by the fix), "redo" (picture, faults, n) -> graph, "crops"
        per spot}. Then only the user's spots are followed; what else the
        critic finds is logged and left alone."""
        s = job.settings
        try:
            vision = self.vision() if self.vision else None
        except Exception:
            vision = None
        if vision is None:
            plan.warnings.append("%s needs a vision model on the LLM host and none is "
                                 "served; the picture is as made."
                                 % ("The critic's check" if marked else "Automatic refinement"))
            return files
        idents = [self.lib.get("identities", x.get("id") if isinstance(x, dict) else x)
                  for x in s.get("identities") or []]
        idents = [i for i in idents if i]
        style = self.lib.get("styles", s.get("style")) if s.get("style") else None
        intent = critic.intent_from(s, plan.prompt)
        canonical = critic.initial_canonical(s, SLOTS, [i["name"] for i in idents],
                                             style["name"] if style else None)
        # Details kept from earlier pictures are known too, so the critic
        # checks them rather than inventing again (not locked: a better look
        # may replace them).
        memory = load_critic_memory(self.lib)
        for k, v in critic.recall(memory, [i["id"] for i in idents]):
            canonical["characters"].setdefault("character_a", {}).setdefault(k, v)
        for k, v in critic.recall(memory, (), s.get("scene") or ""):
            canonical["scene"].setdefault(k, v)
        refs =[plan.references["face"]] if plan.references.get("face") else []
        refs += [i["references"][0] for i in idents if i.get("references")
                 and i["references"][0] not in refs]
        passes = max(0, min(int(s.get("refine_passes") or critic.MAX_PASSES), 6))
        history = [{"pass": 0, "type": "fix" if marked else "initial_generation",
                    "image": files[0]["filename"]}]
        log, stop = [], "pass limit reached"
        faults = list(marked["faults"]) if marked else []   # redrawn, not yet looked at
        next_id = len(faults) + 1
        before = None                 # the picture before the last pass, to go back to
        scores, left = {}, []         # each fault's last score; those given up
        # What earlier pictures taught (the ledger): what to look at first,
        # what the critic had passed, and the strength a redraw starts at.
        ledger, model = load_critic_ledger(self.lib), _str(s.get("model"))
        who, where = [i["id"] for i in idents], s.get("scene") or ""
        first = [] if marked else critic.recurring(ledger, model, who, where)
        missed = critic.blind_checks(ledger)
        if first:
            plan.notes.append("Visual Critic looked first at what went wrong before: %s." %
                              _and([r["kind"] for r in first]))
        tried, found, looked = [], [], False   # for the ledger: redraws scored, faults found
        for n in range(1, passes + 2):
            closing = n > passes      # the look at the last pass: nothing is redrawn after
            if closing and not faults:
                break
            if job.cancel.is_set():
                stop = "cancelled"
                break
            say("critic", "Analyzing result" + ("" if n == 1 else " again") + "...", None)
            try:
                raw = client.fetch(files[0])
                close = self._closeups(job, client, files[0], marked, faults) if marked else []
                result = critic.analyze_generated_image(vision, raw, intent, canonical, refs,
                                                        faults=faults, closeups=close,
                                                        first=first, missed=missed)
            except Exception as e:
                plan.warnings.append("The Visual Critic could not read the picture (%s); "
                                     "it is kept as it was." % e)
                stop = "critic failed"
                break
            looked = True
            scored = critic.score_fixes(faults, result)
            tried += scored
            scores.update((f["id"], f) for f in scored)
            if scored:
                history[-1]["scores"] = critic.score_records(scored)
            if before is not None and critic.went_wrong(scored):
                files, history[-1]["taken_back"] = before, True
                plan.notes.append("Visual Critic: pass %d made the picture worse and mended "
                                  "nothing, so it was taken back." % (n - 1))
            carried, gone = critic.carry(scored)
            left += gone
            faults = []
            # A fix follows the user's spots alone.
            look = dict(result, observations=[], needs_refinement=False) if marked else result
            nxt = critic.plan_next_refinement(look, canonical, carried=carried,
                                              closed=[f["feature"] for f in scored])
            if marked and nxt["needs_pass"]:
                nxt["actions"] = [{"type": "LOCAL_INPAINT", "target": "marked spot%s" % (
                    "" if len(carried) == 1 else "s"), "faults": carried,
                    "corrections": nxt["correct"],
                    "tries": max(f["tries"] for f in carried)}]
            if nxt["needs_pass"]:
                found += [o for a in nxt["actions"] for o in a["faults"] if not o.get("tries")]
                for a in nxt["actions"]:
                    a["denoise"], why = self._critic_denoise(a, values, ledger, model, marked)
                    if why and not a.get("tries"):
                        plan.notes.append("Visual Critic redrew the %s at %s from the start: "
                                          "%s." % (a["target"] or "picture", a["denoise"], why))
            canonical, promoted = critic.merge_canonical(canonical, nxt["promote"])
            if promoted:
                # Read-modify-write on one shared file: two backends' lanes
                # can both be refining at once, and the second save must not
                # overwrite what the first just learned.
                with CRITIC_MEMORY_LOCK:
                    save_critic_memory(self.lib, critic.remember(
                        load_critic_memory(self.lib), canonical, promoted,
                        [i["id"] for i in idents], s.get("scene") or ""))
            text = critic.log_text(n, result, nxt, scored)
            log.append(text)
            self._critic_log(job, text)
            if closing:
                left += carried       # no pass is left to redraw them in
                break
            if not nxt["needs_pass"]:
                stop = "faults left to the user" if left else "no meaningful problems left"
                break
            self._critic_log(job, critic.build_refinement_instructions(
                intent, canonical, nxt["preserve"], nxt["correct"]))
            done, errors = [], []
            before = files
            for action in nxt["actions"]:
                if job.cancel.is_set():
                    break
                say("critic", critic.progress_text(action) + "...", None)
                try:
                    if marked:
                        f = files[0]
                        got = self._run_pass(job, client, marked["redo"](
                            "%s%s [%s]" % (f["subfolder"] + "/" if f.get("subfolder") else "",
                                           f["filename"], f.get("type") or "output"),
                            action["faults"], n), say, critic.progress_text(action),
                            status="critic")
                    else:
                        got = self._correct(job, client, plan, values, files, raw, action,
                                            intent, canonical, n, say)
                except (ComfyError, TemplateError, OSError, ValueError) as e:
                    errors.append("%s: %s" % (action["type"], e))
                    continue
                if got:
                    files = got
                    done.append(action)
            faults = critic.as_faults([dict(o, action="FIX" if marked else a["type"],
                                            denoise=self._fault_denoise(a, o, marked))
                                       for a in done for o in a["faults"]], next_id)
            next_id = max([next_id] + [f["id"] + 1 for f in faults])
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
                          "history": history, "stopped": stop, "log": log,
                          "scores": critic.score_records(scores.values()),
                          "left": [f["feature"] for f in left]}
        made = [h for h in history[1:] if h["type"] != "none"]
        plan.notes.append("Visual Critic: %d refinement pass%s; stopped: %s." % (
            len(made), "" if len(made) == 1 else "es", stop))
        if scores:
            plan.notes.append("Visual Critic's fixes: %s." % "; ".join(
                critic.score_lines(scores.values())))
        if left:
            plan.notes.append("Still wrong, and left to you (Fix a spot): %s." % _and(
                [f["feature"] for f in left]))
        if looked:
            # A fix's picture is filed by run_fix: the critic saw only its spots.
            self._learn(lambda led: critic.note_fixes(led, model, tried) if marked else
                        critic.note_picture(critic.note_fixes(led, model, tried), model,
                                            who, where, found))
        return files

    def _learn(self, change):
        """The ledger changed by `change` (ledger -> ledger) and saved. Two
        backends' lanes can both be learning: read, change and write as one."""
        with CRITIC_LEDGER_LOCK:
            save_critic_ledger(self.lib, change(load_critic_ledger(self.lib)))

    def _critic_denoise(self, action, values, ledger=None, model="", marked=None):
        """The strength of one planned redraw: its kind's own, or where the
        ledger says redraws of this fault start to mend on this model, and
        harder for each try that left the fault there. -> (denoise, why) -
        why "" unless the ledger moved the start. A fix's spots are redrawn
        at the strength the user chose, each as hard as its own tries ask."""
        kind, target = action["type"], action["target"]
        if marked or kind not in CRITIC_DENOISE:       # a fix; a new picture
            return (None if marked else 1.0), ""
        base = CRITIC_DENOISE[kind] if kind != "FACE_CORRECTION" else max(
            CRITIC_DENOISE[kind], float(values.get("face_denoise") or 0))
        top = CRITIC_DENOISE_TOP["hand" if target == "hand" else kind]
        why = ""
        if action.get("faults"):
            base, why = critic.start_denoise(ledger, model, critic.fault_kind(
                action["faults"][0]), kind, base, top)
        return critic.harder(base, action.get("tries"), top), why

    def _fault_denoise(self, action, fault, marked):
        """The strength one fault was just redrawn at, for the ledger."""
        if marked:
            return critic.harder(marked["strength"], fault.get("tries"),
                                 FIX_STRENGTHS["strong"])
        return action.get("denoise")

    def _closeups(self, job, client, f, marked, faults):
        """The marked spots of `faults`, each cut from the picture `f` for
        the critic to see large: a hand is a few dozen pixels of what the
        vision model is shown of the whole. -> [(fault id, PNG bytes)]; []
        when ComfyUI could not cut them, and the critic judges by the whole."""
        image = "%s%s [%s]" % (f["subfolder"] + "/" if f.get("subfolder") else "",
                                f["filename"], f.get("type") or "output")
        shown = [x for x in faults if x.get("spot") is not None][:critic.CLOSEUPS]
        if not shown:
            return []
        g = {"cu": {"class_type": "LoadImage", "inputs": {"image": image}}}
        for i, x in enumerate(shown):
            c = marked["crops"][x["spot"]]
            g["cu%d" % i] = {"class_type": "ImageCropV2", "inputs": {
                "image": ["cu", 0],
                "crop_region": {k: c[k] for k in ("x", "y", "width", "height")}}}
            g["cs%d" % i] = {"class_type": "PreviewImage", "inputs": {"images": ["cu%d" % i, 0]}}
        try:
            entry = client.listen_for_progress(client.queue_workflow(g), lambda kind, d: None,
                                               stop=job.cancel.is_set)
            out = (entry or {}).get("outputs") or {}
            return [(x["id"], client.fetch(out["cs%d" % i]["images"][0]))
                    for i, x in enumerate(shown)]
        except (ComfyError, KeyError, IndexError, TypeError):
            return []

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
        # A fault the last redraw left there is redrawn harder, not the same.
        v["face_denoise"] = action.get("denoise") or self._critic_denoise(action, values)[0]
        oval = os.path.join(self.lib.root, "face_oval.png")
        if not os.path.isfile(oval):
            os.makedirs(self.lib.root, exist_ok=True)
            with open(oval, "wb") as fh:
                fh.write(oval_png())
        graph = face_graph(wf, v, plan.loras, image, crops, client.upload_image(oval),
                           "%s_pass%d_%s" % (values["filename_prefix"], n, kind.lower()))
        return self._run_pass(job, client, graph, say, critic.progress_text(action),
                              status="critic")

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
                              "Regenerating the picture", status="critic")

    def _sam3_of(self, backend):
        inv = self.inventories.get(backend["id"]) or {}
        sam = sorted(c for c in inv.get("checkpoints") or () if SAM3 in c.lower())
        return sam[0] if sam else None

    def _run_pass(self, job, client, graph, say, label, status="refining"):
        """Run one refinement graph to its end. -> files, or None if cancelled.
        Raises ComfyError when it ends without a picture."""
        job.passes.append({"label": label, "graph": graph})
        job.prompt_id = client.queue_workflow(graph)

        def on_event(kind, data):
            if kind == "progress" and data[1]:
                say(status, "%s · step %d of %d" % (label, data[0], data[1]),
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
            say("face", "matching each face to its photos", None)
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
        people = [p for p in faces_of(job.settings).get("people") or []
                  if p.get("face") and p.get("region")]
        if not people:
            return True
        pulid, why = self._pulid(client, plan, types)
        if not pulid:
            return True                   # the face pass says why, once
        if not w or not h:
            return True
        faces, chained = [], []
        try:
            folder = os.path.join(self.lib.root, "face_regions")
            os.makedirs(folder, exist_ok=True)
            for person in people:
                if job.cancel.is_set():
                    return False
                photos = [p for p in (person.get("photos") or [person["face"]])
                         if os.path.isfile(p)] or [person["face"]]
                photos = photos[:REFERENCE_PHOTOS_MAX]
                if len(photos) > 1:
                    chained.append(person["name"])
                uploaded = [client.upload_image(p) for p in photos]
                if list(person["region"]) == WHOLE_FRAME:
                    # No mask: one the size of the picture's tokens does not
                    # fit when Kontext adds the item picture's (2026-09-26:
                    # "tensor a (8022) must match ... (3952)"), and a mask of
                    # everything masks nothing.
                    faces.append((uploaded, None))
                    continue
                data = region_png(person["region"], w, h)
                path = os.path.join(folder, hashlib.sha1(data).hexdigest()[:16] + ".png")
                if not os.path.isfile(path):
                    with open(path, "wb") as f:
                        f.write(data)
                faces.append((uploaded, client.upload_image(path)))
        except (ComfyError, OSError) as e:
            plan.warnings.append("The face pictures could not be sent (%s); the picture is "
                                 "drawn without them." % e)
            return True
        add_pulid(graph, pulid, faces)
        plan.notes.append("%s drawn from their face picture%s in the picture itself." % (
            ", ".join(p["name"] for p in people), "" if len(people) == 1 else "s"))
        if chained:
            plan.notes.append("%s drawn from more than one of their photos for a stronger "
                              "match." % _and(sorted(chained)))
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
                    say("face", "finding faces · " + node_name(data), None)
                    return
                if data in refine:
                    say("face", "refining detail · " + node_name(data), None)
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
                    status = "face" if nid in refine else "sampling"
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
        made = p.workflow.get("defaults") or {}
        now = time.time()
        idents = []
        for sel in s.get("identities") or []:
            ident = self.lib.get("identities", sel.get("id") if isinstance(sel, dict) else sel)
            if ident:
                idents.append({"id": ident["id"], "name": ident["name"],
                               "description": ident.get("description", ""),
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
            # What the pass ran with: a value the form left alone is the
            # workflow's own, which `fill` gave it.
            "refine": ({"upscale": v.get("upscale", made.get("upscale")),
                        "denoise": v.get("refine_denoise", made.get("refine_denoise")),
                        "steps": v.get("refine_steps", made.get("refine_steps")),
                        "sampler": v.get("refine_sampler", made.get("refine_sampler")),
                        "scheduler": v.get("refine_scheduler", made.get("refine_scheduler")),
                        "model": (v.get((p.workflow.get("refine_model") or {}).get("file"))
                                  if v.get("refine_model") else None)}
                       if v.get("refine") else None),
            "face_detail": job.face,
            "references": p.references,
            "warnings": p.warnings, "notes": p.notes + job.notes,
            # The model's own terms, and a pass's (the head swap's Klein) after them.
            "license": " ".join(dict.fromkeys(t for t in (model.get("license"), job.license)
                                              if t)),
            "duration": round(time.time() - job.started, 1),
            "prompt_id": job.prompt_id,
            "settings": s,
            "graph": graph,
            "face_graph": job.face_graph,
            "refinement": job.refinement,
            "paste_graph": job.paste_graph,
            "passes": list(job.passes),
            "facefusion": job.facefusion,
            "dress": job.dress,
        }

    def close(self):
        self.queue.close()


def main(argv=None):
    """`python apps/image_studio/imagegen.py --probe`: every endpoint on every backend,
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
