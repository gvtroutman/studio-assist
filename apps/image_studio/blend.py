"""New photos of an identity from its own references, by FLUX.1 Kontext [dev].

The user, 2026-09-28: "i would like to add breeding to reference images in
identity" - and then "breed is a seperate step. the angles are something every
photo has". So two things, both in the identity editor. Breed was renamed
Blend on 2026-09-29 (his word: "rename breed to blend"):

- Angles: any reference photo -> the same person seen from other angles
  (`ANGLES`), one Kontext edit of that photo per angle (`angle_graph`). The
  angles are the parts of a view cube, picked on one (`viewcube`), and the
  last pick is kept as the preset (`load_views`).
- Blend: two reference photos -> one new photo that mixes them
  (`blend_graph`): both go in as Kontext reference latents, the blend is drawn
  on an empty latent at the first photo's shape, so it copies neither.

Nothing is scored or kept here: the editor shows the results and the person
picks which join the references. (`tools/make_variations.py` scores with
ArcFace; that needs ComfyUI's venv, and a person looking works as well.)

Blend is also a tool of its own, anywhere in the Image Studio (the user,
2026-09-29: "put real infrastructure behind it", and of what that could be,
"Blend anywhere"): any two pictures - the library's, History's, a file -
as a job of the studio's own queue (`submit`, `run_job`), kept in History
(`record`) with both pictures, the words and the seed, so Generate Again
remakes it. Those two pictures need not be of a person (`blend_words`).

Stdlib only; the ComfyUI calls go through the studio's own client.
"""

import copy
import json
import os
import random
import time

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

BLEND = ("Make one new photograph of the same person who is in both of these pictures. "
         "Mix the two: take the pose and framing from one and the setting, light and "
         "clothes from the other, or blend them, so the new photo is like neither "
         "picture exactly.")
# Two pictures of anything (Blend anywhere): nobody to keep.
BLEND_PICTURES = ("Make one new picture that blends these two pictures. Take the subject "
                  "and composition from one and the setting, light and colours from the "
                  "other, or mix them, so the new picture is like neither picture exactly.")


def clean_blend(d):
    """A job's `blend` as it is used: {"images": [paths], "person": whether
    both pictures are of one person, who is kept, "words": what to add}."""
    d = d if isinstance(d, dict) else {}
    images = d.get("images") if isinstance(d.get("images"), (list, tuple)) else []
    return {"images": [p for p in images if isinstance(p, str) and p],
            "person": bool(d.get("person")),
            "words": " ".join(str(d.get("words") or "").split())}


def blend_words(person=True, words=""):
    """What a blend is asked for: a new photo of the person in both
    pictures, or a blend of two pictures of anything, then `words`."""
    text = BLEND + KEEP if person else BLEND_PICTURES
    return text + (" " + words.rstrip(".") + "." if words else "")


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
    """A new photo's size: its source's shape at ~1 MP, multiples of 16."""
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


def blend_graph(image_a, image_b, size, seed, prefix="identity/blend", steps=STEPS,
                guidance=GUIDANCE, words=None):
    """Two photos (LoadImage names) -> one blend of `size` (w, h), asked for
    in `words` (`blend_words`; the same person in both when not given)."""
    g = _loaders()
    g["text"] = {"class_type": "CLIPTextEncode", "inputs": {
        "text": words or blend_words(), "clip": ["2", 0]}}
    cond, _ = _reference(g, 1, image_a, ["text", 0])
    cond, _ = _reference(g, 2, image_b, cond)
    g["empty"] = {"class_type": "EmptySD3LatentImage", "inputs": {
        "width": int(size[0]), "height": int(size[1]), "batch_size": 1}}
    return _finish(g, "text", cond, ["empty", 0], seed, prefix, steps, guidance)


def route(studio, settings=None):
    """The backend these edits run on: enabled, up, with Kontext; the one a
    job's `settings` name, else the one it was made on (Generate Again),
    else the primary (5090) first. -> (backend or None, why)."""
    s = settings or {}
    if s.get("backend") not in (None, "", "auto"):
        order = [studio.backend(s["backend"])]
    else:
        order = sorted(studio.backends(), key=lambda b: (
            b["id"] != s.get("prefer_backend"), "primary" not in (b.get("roles") or [])))
    why = []
    for b in order:
        if b is None or not b.get("enabled"):
            continue
        held = getattr(studio, "held", {}).get(b["id"])
        if held:                      # Build LoRA has its GPU
            why.append(held.rstrip("."))
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


# ============================================================ Blend anywhere
# A blend as a job of the studio's queue: settings {"mode": "blend", "seed",
# "backend", "blend": {"images": [a, b], "person", "words"}}. `Studio.submit`
# and `Studio.run_job` hand a job of this mode to `submit` and `run_job` here.
STAGES = [("sampling", "Sampling"), ("decoding", "Decoding"), ("complete", "Complete")]
LABEL = "Blend (FLUX Kontext)"


def summary(settings):
    """One line for a blend: its job's row, and its record's prompt."""
    d = clean_blend(settings.get("blend"))
    what = "the same person in two photos" if d["person"] else "two pictures"
    return "Blend: " + what + (". " + d["words"] if d["words"] else "")


def problem(blend):
    """Why these pictures cannot be blended, in words; "" when they can."""
    images = blend["images"]
    if len(images) != 2:
        return "Choose two pictures to blend."
    if os.path.normcase(os.path.abspath(images[0])) == os.path.normcase(
            os.path.abspath(images[1])):
        return "Choose two different pictures to blend."
    gone = [p for p in images if not os.path.isfile(p)]
    return "Not on this PC any more: %s." % ", ".join(gone) if gone else ""


def submit(studio, settings):
    """Queue a blend (network I/O: off the UI thread). -> [Job]. Raises
    ComfyError when the pictures or every backend cannot take it."""
    s = copy.deepcopy(settings)
    s["mode"], s["batch"] = "blend", 1
    s["blend"] = clean_blend(s.get("blend"))
    if int(s.get("seed", -1)) < 0:
        s["seed"], s["seed_mode"] = random.randint(0, ig.MAX_SEED), "random"
    else:
        s["seed_mode"] = "fixed"
    wrong = problem(s["blend"])
    if wrong:
        raise ComfyError(wrong)
    for b in studio.backends():
        if b["enabled"] and (b["id"] not in studio.health or b["id"] not in studio.inventories):
            studio.check(b)
    b, why = route(studio, s)
    if b is None:
        raise ComfyError(why)
    job = ig.Job(s, b)
    studio.queue.add(job)
    return [job]


def run_job(studio, job, client, say):
    """A blend job, start to finish, on its lane's thread."""
    b, s = job.backend, job.settings
    d = clean_blend(s.get("blend"))
    short = lacks(studio.inventories.get(b["id"]))
    wrong = problem(d) or ("%s lacks %s." % (b["name"], ", ".join(short)) if short else "")
    if wrong:
        return studio.queue._finish(job, "failed", wrong)
    size = size_for(d["images"][0])
    if b.get("shares_llm_gpu") and studio.make_room is not None:
        say(detail="clearing LM Studio off the GPU")
        try:
            studio.make_room(b)
        except Exception as e:
            job.notes.append("Could not clear the shared GPU (%s); this may be slow." % e)
    try:
        say("uploading", "uploading the two pictures", None)
        names = [client.upload_image(p) for p in d["images"]]
        graph = blend_graph(names[0], names[1], size, s["seed"],
                            prefix="ImageStudio/blend_%s" % job.id,
                            words=blend_words(d["person"], d["words"]))
        say("loading", "blending on %s" % b["name"], None)
        files = studio._run_pass(job, client, graph, say, "Blend", status="sampling")
    except (ComfyError, OSError) as e:
        if job.cancel.is_set():
            return studio.queue._finish(job, "cancelled")
        return studio.queue._finish(job, "failed", "Blend failed on %s: %s" % (b["name"], e))
    if files is None or job.cancel.is_set():
        return studio.queue._finish(job, "cancelled")
    say("decoding", "fetching the picture from %s" % b["name"], None)
    try:
        pictures = [(f["filename"], client.fetch(f)) for f in files[:1]]
    except ComfyError as e:
        return studio.queue._finish(job, "failed", "The picture was made but could not be "
                                    "fetched from %s: %s" % (b["name"], e))
    job.record = studio.history.add(record(job, size), pictures)
    job.outputs = list(job.record["images"])
    job.progress = 1.0
    studio.queue._finish(job, "complete")


def record(job, size):
    """A blend's history record, in the fields a Generate record has. Its
    graph is its one pass (`passes`), so `graph` stays empty and the Nodes
    view shows it once."""
    s, b = job.settings, job.backend
    now = time.time()
    a, c = s["blend"]["images"]
    return {
        "id": time.strftime("%Y%m%d-%H%M%S", time.localtime(now)) + "-" + job.id[:6],
        "created": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)),
        "created_ts": now,
        "prompt": summary(s), "negative": "", "seed": s.get("seed"),
        "model": {"id": "blend", "label": LABEL, "file": KONTEXT,
                  "family": "flux1-kontext",
                  "files": {kind: list(names) for kind, names in FILES.items()}},
        "loras": [], "identities": [], "style": None, "preset": "blend",
        "workflow": "blend", "workflow_label": LABEL,
        "backend": {"id": b["id"], "name": b["name"], "url": b["url"]},
        "sampler": "euler", "scheduler": "simple", "steps": STEPS,
        "guidance": GUIDANCE, "width": size[0], "height": size[1], "denoise": 1.0,
        "refine": None, "face_detail": None,
        "references": {"picture 1": a, "picture 2": c},
        "warnings": [], "notes": list(job.notes),
        "duration": round(time.time() - job.started, 1),
        "prompt_id": job.prompt_id, "settings": s,
        "graph": None, "face_graph": None, "passes": list(job.passes),
    }
