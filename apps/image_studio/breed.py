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


FLIP = "y-axis: horizontally"           # ImageFlip's left-right mirror


def mirrored(key):
    """Is this view made from its left twin in a mirror? Live on Partner
    (2026-09-29, all 26, one seed) Kontext turned a face only towards the
    picture's left: each right view came out as its left twin, whatever the
    words said. So a view from their right is made in a mirror: the photo
    flipped, the left twin asked for, the result flipped back - which is
    their real right side, not a fake (a mirrored person's left is their
    right)."""
    return key[0] > 0


# Kontext-style edits: what changes, then what stays. Head turns were the
# weakest edit in make_variations, so each says so plainly and moves the camera,
# not only the head. What Kontext is asked is always a left view (`mirrored`):
# the camera at their left, so they face the picture's left - the way it turns
# them anyway. These are reference photos of a face, so from behind they
# still look back at the lens.
def view_steps(key):
    """The edits view `key` is made by, the words of each: one, or - for a
    turn seen from above or below - the turn level, then the camera's height
    on that picture. In one edit Kontext did one or the other: a turn from
    below came out facing the camera, a turn from above lost its turn. In
    two (live, 2026-09-29) the turn stayed. A right view's words are its left
    twin's (`mirrored`)."""
    x, y, z = key
    if not x and not z:
        return [STRAIGHT[y]]
    if not y or (not x and z > 0):
        return [view_prompt(key)]
    return [view_prompt((x, 0, z)), HEIGHT[y] % KEEP_TURN]


def view_prompt(key):
    """The words for view `key` in one edit - for a right view, its left
    twin's (`mirrored`)."""
    x, y, z = key
    if not x and not z:
        return STRAIGHT[y]
    if not x:
        flat = ("this person facing the camera straight on, looking into the lens"
                if z > 0 else
                "this person from directly behind, their back to the camera, turning "
                "their head to look back over their shoulder so their face shows")
    elif z > 0:
        flat = ("this person turned 45 degrees towards the left of the picture, in a "
                "three-quarter view: their nose points towards the left edge of the "
                "picture, their far cheek partly hidden, one ear visible")
    elif not z:
        flat = ("this person in full side profile facing the left edge of the picture, "
                "the camera directly beside them")
    else:
        flat = ("this person from behind and to one side, their back half turned to "
                "the camera, looking back over their shoulder towards the left of the "
                "picture so their face is still seen")
    return HEIGHT[y] % flat


# The camera's height. "From a high camera angle" alone came out, live, as
# the same camera with the head tilted. "Rotate the camera ... bird's-eye /
# worm's-eye view", "zoom out" and the ceiling lights are what made Kontext
# draw the body foreshortened from above and below (2026-09-29, four rounds
# on Partner; "from near the floor" and "the height of their waist" did not).
HEIGHT = {
    0: "Show %s",
    # "Their head tilted up towards the camera" turned every view from above
    # to face the lens; this keeps about 30 degrees of a turn (round 5). A
    # bird's-eye shot still pulls the face round - it is the weak row.
    1: ("Raise the camera high above this person to a bird's-eye view looking down at "
        "them from about 45 degrees overhead, without changing their pose: show %s, "
        "the top of their head and their shoulders seen from above, their face "
        "pointing the same way as before"),
    -1: ("Rotate the camera down to a worm's-eye view from below, looking up at this "
         "person from about 45 degrees under them, and show %s, their head tipped "
         "down towards the camera. Zoom out so they tower over the camera: their "
         "whole upper body and arms seen from underneath, the ceiling and ceiling "
         "lights far above them"),
}
KEEP_TURN = "this person keeping exactly the way they are turned and facing"
STRAIGHT = {
    1: ("Rotate the camera to a bird's-eye view directly above this person, looking "
        "straight down at the crown of their head; their face turned up to the lens "
        "and strongly foreshortened, the floor around their feet behind them"),
    -1: ("Rotate the camera to lie on the floor directly beneath this person, pointing "
         "straight up: zoom out so we see them from underneath, their body "
         "foreshortened as it rises above the lens, their face looking straight down "
         "into it, the ceiling behind them"),
}


ANGLES = [(name, " Then: ".join(view_steps(key)))
          for name, key in zip(VIEW_NAMES, VIEW_KEYS)]         # for reading, not sent
ANGLE_NAMES = VIEW_NAMES

BREED = ("Make one new photograph of the same person who is in both of these pictures. "
         "Mix the two: take the pose and framing from one and the setting, light and "
         "clothes from the other, or blend them, so the new photo is like neither "
         "picture exactly.")


def angle_prompt(name, step=0):
    """The words of edit `step` of view `name`, with what stays."""
    return view_steps(VIEW_OF[name])[step] + "." + KEEP


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
    """One photo (a LoadImage name) seen from `angle` (a name in ANGLES); a
    right view through a mirror both ways (`mirrored`), a turn from above or
    below in two edits (`view_steps`), the second on the first's picture."""
    key = VIEW_OF[angle]
    g = _loaders()
    g["text"] = {"class_type": "CLIPTextEncode", "inputs": {
        "text": angle_prompt(angle), "clip": ["2", 0]}}
    cond, latent = _reference(g, 1, image, ["text", 0])
    g = _finish(g, "text", cond, latent, seed, prefix, steps, guidance)
    picture = ["dec", 0]
    if mirrored(key):
        g["flip_in"] = {"class_type": "ImageFlip", "inputs": {
            "image": ["r1_load", 0], "flip_method": FLIP}}
        g["r1_scale"]["inputs"]["image"] = ["flip_in", 0]
        g["flip_out"] = {"class_type": "ImageFlip", "inputs": {
            "image": ["dec", 0], "flip_method": FLIP}}
        picture = ["flip_out", 0]
    if len(view_steps(key)) > 1:
        g["text2"] = {"class_type": "CLIPTextEncode", "inputs": {
            "text": angle_prompt(angle, 1), "clip": ["2", 0]}}
        g["s2_scale"] = {"class_type": "FluxKontextImageScale", "inputs": {"image": picture}}
        g["s2_enc"] = {"class_type": "VAEEncode", "inputs": {
            "pixels": ["s2_scale", 0], "vae": ["3", 0]}}
        g["s2_ref"] = {"class_type": "ReferenceLatent", "inputs": {
            "conditioning": ["text2", 0], "latent": ["s2_enc", 0]}}
        g["guide2"] = dict(g["guide"], inputs=dict(g["guide"]["inputs"],
                                                   conditioning=["s2_ref", 0]))
        g["neg2"] = {"class_type": "ConditioningZeroOut", "inputs": {
            "conditioning": ["text2", 0]}}
        g["ks2"] = dict(g["ks"], inputs=dict(
            g["ks"]["inputs"], seed=(int(seed) + 1) % (ig.MAX_SEED + 1), positive=["guide2", 0],
            negative=["neg2", 0], latent_image=["s2_enc", 0]))
        g["dec2"] = {"class_type": "VAEDecode", "inputs": {"samples": ["ks2", 0],
                                                           "vae": ["3", 0]}}
        picture = ["dec2", 0]
    g["save"]["inputs"]["images"] = picture
    return g


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
