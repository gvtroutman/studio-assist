"""New photos of an identity from its own references, by FLUX.1 Kontext [dev].

The user, 2026-09-28: "i would like to add breeding to reference images in
identity" - and then "breed is a seperate step. the angles are something every
photo has". So two things, both in the identity editor:

- Angles: any reference photo -> the same person seen from other angles
  (`ANGLES`), one Kontext edit of that photo per angle (`angle_graph`).
- Breed: two reference photos -> one new photo that mixes them
  (`breed_graph`): both go in as Kontext reference latents, the child is drawn
  on an empty latent at the first parent's shape, so it copies neither.

Nothing is scored or kept here: the editor shows the results and the person
picks which join the references. (`tools/make_variations.py` scores with
ArcFace; that needs ComfyUI's venv, and a person looking works as well.)

Stdlib only; the ComfyUI calls go through the studio's own client.
"""

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

# Kontext-style edits: what changes, then what stays. Head turns were the
# weakest edit in make_variations, so each says so plainly and moves the camera,
# not only the head.
ANGLES = [
    ("three-quarter left", "Show this person from a three-quarter angle, their face "
                           "turned 45 degrees to their left, the camera moved to match"),
    ("three-quarter right", "Show this person from a three-quarter angle, their face "
                            "turned 45 degrees to their right, the camera moved to match"),
    ("profile left", "Show this person in full side profile facing left, the camera "
                     "directly beside them"),
    ("profile right", "Show this person in full side profile facing right, the camera "
                      "directly beside them"),
    ("front", "Show this person facing the camera straight on, looking into the lens"),
    ("from above", "Show this person from a high camera angle, looking down at them as "
                   "they look up towards the camera"),
    ("from below", "Show this person from a low camera angle, looking up at them"),
    ("over the shoulder", "Show this person looking back over their shoulder at the "
                          "camera, their body turned away"),
]
ANGLE_NAMES = [name for name, _ in ANGLES]

BREED = ("Make one new photograph of the same person who is in both of these pictures. "
         "Mix the two: take the pose and framing from one and the setting, light and "
         "clothes from the other, or blend them, so the new photo is like neither "
         "picture exactly.")


def angle_prompt(name):
    return dict(ANGLES)[name] + "." + KEEP


def pick_angles(n, rng=random):
    """`n` different angles, in a random order."""
    return rng.sample(ANGLE_NAMES, min(n, len(ANGLE_NAMES)))


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
