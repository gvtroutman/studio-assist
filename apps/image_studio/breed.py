"""New photos of an identity from its own references, by FLUX.1 Kontext [dev].

The user, 2026-09-28: "i would like to add breeding to reference images in
identity" - and then "breed is a seperate step. the angles are something every
photo has". So two things, both in the identity editor:

- Angles: any reference photo -> the same person seen from other angles
  (`ANGLES`), one Kontext edit of that photo per angle (`angle_graph`). The
  angles are the parts of a view cube, picked on one (`viewcube`), and the
  last pick is kept as the preset (`load_views`).
- Breed: two reference photos -> one new photo that mixes them
  (`breed_graph`): both go in as Kontext reference latents, the child is drawn
  on an empty latent at the first parent's shape, so it copies neither.

Nothing is scored or kept here: the editor shows the results and the person
picks which join the references. (`tools/make_variations.py` scores with
ArcFace; that needs ComfyUI's venv, and a person looking works as well.)

Stdlib only; the ComfyUI calls go through the studio's own client.
"""

import json
import os
import random

import apps.image_studio.imagegen as ig
from apps.comfyui.mcp import ComfyError, outputs_of

KONTEXT = "flux1-dev-kontext_fp8_scaled.safetensors"
FILES = {"diffusion_models": [KONTEXT],
         "text_encoders": ["clip_l.safetensors", "t5xxl_fp16.safetensors"],
         "vae": ["ae.safetensors"]}
STEPS = 24
GUIDANCE = 2.5
MEGAPIXELS = 1.0

KEEP = (" Keep this person's exact same face, face shape, facial features, body build, "
        "skin tone, eye colour, hairstyle and hair length, and glasses if they wear them. "
        "The same person, not a look-alike. A real photograph.")

# The views are the 26 parts of a view cube (`viewcube`), the way Bambu Studio
# asks which side of the model to look at (the user, 2026-09-29): 6 faces, 12
# edges, 8 corners. A view is where the camera stands, as a key (x, y, z) in
# the person's own frame, each -1, 0 or 1: x their right, y up, z in front of
# them. "Right" and "left" are always theirs.
VIEW_KEYS = [(x, y, z) for y in (0, 1, -1) for z in (1, 0, -1) for x in (0, 1, -1)
             if (x, y, z) != (0, 0, 0)]
PRESET_FILE = "angle_views.json"


def view_name(key):
    """'front', 'front right', 'right side', 'back left from above',
    'straight above' ..."""
    x, y, z = key
    side = {1: "right", -1: "left", 0: ""}[x]
    if z:
        flat = ("front" if z > 0 else "back") + (" " + side if side else "")
    else:
        flat = side + " side" if side else ""
    if not flat:
        return "straight above" if y > 0 else "straight below"
    return flat + {1: " from above", -1: " from below", 0: ""}[y]


VIEW_NAMES = [view_name(k) for k in VIEW_KEYS]
VIEW_OF = dict(zip(VIEW_NAMES, VIEW_KEYS))
DEFAULT_VIEWS = ["front left", "front right", "left side", "right side"]


# Kontext-style edits: what changes, then what stays. Head turns were the
# weakest edit in make_variations, so each says so plainly and moves the camera,
# not only the head - and says which edge of the picture they face, since
# "their left" and the picture's left are opposite ways round. With the camera
# at their right they face the picture's right. These are reference photos of
# a face, so from behind they still look back at the lens.
def view_prompt(key):
    x, y, z = key
    side = "right" if x > 0 else "left"
    if not x and not z:
        return ("Show this person from directly above, the camera looking straight "
                "down at them as they look up into the lens" if y > 0 else
                "Show this person from directly below, the camera looking straight "
                "up at them")
    if not x:
        flat = ("Show this person facing the camera straight on, looking into the lens"
                if z > 0 else
                "Show this person from directly behind, their back to the camera, "
                "turning their head to look back over their shoulder so their face "
                "shows")
    elif z > 0:
        flat = ("Show this person from a three-quarter angle: the camera has moved 45 "
                "degrees round to their %s, so we see more of the %s side of their "
                "face and they face towards the %s of the picture" % (side, side, side))
    elif not z:
        flat = ("Show this person in full side profile: the camera directly at their "
                "%s side, so we see the %s side of their face and they face the %s "
                "edge of the picture" % (side, side, side))
    else:
        flat = ("Show this person from behind and to their %s, their back half turned "
                "to the camera, looking back over their %s shoulder at the lens so "
                "their face is still seen" % (side, side))
    return flat + {1: ", from a high camera angle looking down at them as they look "
                      "up towards it", -1: ", from a low camera angle looking up at them",
                   0: ""}[y]


ANGLES = [(name, view_prompt(key)) for name, key in zip(VIEW_NAMES, VIEW_KEYS)]
ANGLE_NAMES = VIEW_NAMES

BREED = ("Make one new photograph of the same person who is in both of these pictures. "
         "Mix the two: take the pose and framing from one and the setting, light and "
         "clothes from the other, or blend them, so the new photo is like neither "
         "picture exactly.")


def angle_prompt(name):
    return dict(ANGLES)[name] + "." + KEEP


def pick_angles(n, rng=random):
    """`n` different angles, in a random order."""
    return rng.sample(ANGLE_NAMES, min(n, len(ANGLE_NAMES)))


def load_views(root=None):
    """The views picked last time (the preset), else DEFAULT_VIEWS. Names
    that are not views any more are dropped."""
    try:
        with open(os.path.join(root or ig.studio_dir(), PRESET_FILE), encoding="utf-8") as f:
            names = json.load(f)
    except (OSError, ValueError):
        return list(DEFAULT_VIEWS)
    if not isinstance(names, list):
        return list(DEFAULT_VIEWS)
    return [n for n in names if n in VIEW_OF]


def save_views(names, root=None):
    """Keep `names` as the preset, atomically. Raises OSError."""
    root = root or ig.studio_dir()
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, PRESET_FILE)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump([n for n in names if n in VIEW_OF], f)
    os.replace(path + ".tmp", path)


def lacks(inventory):
    """The model files a backend is missing for these edits (inventory as
    Studio.inventories keeps it; None = not read yet, so nothing known)."""
    if inventory is None:
        return []
    return [f for kind, names in FILES.items() for f in names
            if f not in (inventory.get(kind) or ())]


def size_for(path, megapixels=MEGAPIXELS):
    """A child's size: the parent's shape at ~1 MP, multiples of 16."""
    w, h = ig.file_size_of(path) or (1024, 1024)
    k = (megapixels * 1e6 / float(w * h)) ** 0.5
    return max(16, int(w * k) // 16 * 16), max(16, int(h * k) // 16 * 16)


def _loaders():
    return {
        "1": {"class_type": "UNETLoader", "inputs": {
            "unet_name": KONTEXT, "weight_dtype": "default"}},
        "2": {"class_type": "DualCLIPLoader", "inputs": {
            "clip_name1": FILES["text_encoders"][0], "clip_name2": FILES["text_encoders"][1],
            "type": "flux", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": FILES["vae"][0]}},
    }


def _reference(g, n, image, conditioning):
    """LoadImage -> Kontext scale -> latent -> onto `conditioning`. -> the
    ReferenceLatent's output and the latent (ids r<n>*)."""
    g["r%d_load" % n] = {"class_type": "LoadImage", "inputs": {"image": image}}
    g["r%d_scale" % n] = {"class_type": "FluxKontextImageScale", "inputs": {
        "image": ["r%d_load" % n, 0]}}
    g["r%d_enc" % n] = {"class_type": "VAEEncode", "inputs": {
        "pixels": ["r%d_scale" % n, 0], "vae": ["3", 0]}}
    g["r%d_ref" % n] = {"class_type": "ReferenceLatent", "inputs": {
        "conditioning": conditioning, "latent": ["r%d_enc" % n, 0]}}
    return ["r%d_ref" % n, 0], ["r%d_enc" % n, 0]


def _finish(g, prompt_node, conditioning, latent, seed, prefix, steps, guidance):
    g["guide"] = {"class_type": "FluxGuidance", "inputs": {
        "conditioning": conditioning, "guidance": guidance}}
    g["neg"] = {"class_type": "ConditioningZeroOut", "inputs": {
        "conditioning": [prompt_node, 0]}}
    g["ks"] = {"class_type": "KSampler", "inputs": {
        "model": ["1", 0], "seed": int(seed), "steps": steps, "cfg": 1.0,
        "sampler_name": "euler", "scheduler": "simple", "positive": ["guide", 0],
        "negative": ["neg", 0], "latent_image": latent, "denoise": 1.0}}
    g["dec"] = {"class_type": "VAEDecode", "inputs": {"samples": ["ks", 0], "vae": ["3", 0]}}
    g["save"] = {"class_type": "SaveImage", "inputs": {
        "filename_prefix": prefix, "images": ["dec", 0]}}
    return g


def angle_graph(image, angle, seed, prefix="identity/angle", steps=STEPS,
                guidance=GUIDANCE):
    """One photo (a LoadImage name) seen from `angle` (a name in ANGLES)."""
    g = _loaders()
    g["text"] = {"class_type": "CLIPTextEncode", "inputs": {
        "text": angle_prompt(angle), "clip": ["2", 0]}}
    cond, latent = _reference(g, 1, image, ["text", 0])
    return _finish(g, "text", cond, latent, seed, prefix, steps, guidance)


def breed_graph(image_a, image_b, size, seed, prefix="identity/breed", steps=STEPS,
                guidance=GUIDANCE):
    """Two photos (LoadImage names) -> one child of `size` (w, h)."""
    g = _loaders()
    g["text"] = {"class_type": "CLIPTextEncode", "inputs": {
        "text": BREED + KEEP, "clip": ["2", 0]}}
    cond, _ = _reference(g, 1, image_a, ["text", 0])
    cond, _ = _reference(g, 2, image_b, cond)
    g["empty"] = {"class_type": "EmptySD3LatentImage", "inputs": {
        "width": int(size[0]), "height": int(size[1]), "batch_size": 1}}
    return _finish(g, "text", cond, ["empty", 0], seed, prefix, steps, guidance)


def route(studio):
    """The backend these edits run on: enabled, up, with Kontext; the
    primary (5090) first. -> (backend or None, why)."""
    why = []
    for b in sorted(studio.backends(), key=lambda b: "primary" not in (b.get("roles") or [])):
        if not b.get("enabled"):
            continue
        if not (studio.health.get(b["id"]) or {}).get("ok"):
            studio.check(b)
        if not (studio.health.get(b["id"]) or {}).get("ok"):
            why.append("%s is offline" % b["name"])
            continue
        short = lacks(studio.inventories.get(b["id"]))
        if short:
            why.append("%s lacks %s" % (b["name"], ", ".join(short)))
            continue
        return b, ""
    return None, ("No backend can make these (FLUX Kontext): "
                  + ("; ".join(why) or "none is enabled") + ".")


def run(client, graph, stop=None, on_progress=None):
    """Queue `graph` and wait. -> the picture's bytes; None if `stop()` said
    to stop. Raises ComfyError when the run makes nothing."""
    def on_event(kind, data):
        if kind == "progress" and data[1] and on_progress is not None:
            on_progress(data[0], data[1])
    pid = client.queue_workflow(graph)
    watch = client.watch()
    try:
        entry = client.listen_for_progress(pid, on_event, stop=stop, watch=watch)
    finally:
        watch.close()
    if entry is None:
        client.cancel_job(pid)
        return None
    files = outputs_of(entry)
    if not files:
        raise ComfyError("; ".join(ig.run_errors(entry, graph)) or "the run made no picture")
    return client.fetch(files[0])
