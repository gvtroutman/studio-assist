"""Make new photos of one identity from its reference photos (FLUX Kontext).

The user, 2026-09-27: "take the images in the reference folder and make new
versions out of it" - more, and more varied, pictures of the same person for
Build LoRA. Each chosen reference goes through FLUX.1 Kontext [dev] on the
5090's ComfyUI with a few edit prompts (a real setting and light instead of
the cut-out's white, other clothes, another expression, a turned head). Every
result is scored with ArcFace (insightface antelopev2, the model PuLID uses)
against the mean face of all the references, and kept only if it scores at
most DROP below its own source photo (scored against the other references).
Not one threshold for all: a real side profile scores 0.48 and a real front
face 0.81-0.88, so a floor low enough for profiles let a front-facing
variation through at 0.52 that visibly was a relative, not her.

Nothing is added to the profile: kept pictures land in
<library>/variations/<id>/kept (dropped ones in .../dropped) with a contact
sheet and report.json, for a person to look at first.

Run in ComfyUI's venv (it has PIL, cv2 and insightface):
  D:\\ComfyUI\\venv\\Scripts\\python.exe tools\\make_variations.py partner --photos 1,11,22
"""
import argparse
import json
import os
import random
import sys
import time
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import apps.comfyui.mcp as comfy  # noqa: E402

LIBRARY = Path(os.environ.get("APPDATA", "")) / "StudioAssistant" / "image-studio"
INSIGHTFACE = "D:/ComfyUI/models/insightface"
KEEP = (", while keeping her exact same face, facial features, glasses, eye colour "
        "and hairstyle")

# Edit instructions, Kontext-style: say what changes, then what stays.
PROMPTS = {
    "park": "Place this woman in a sunlit park with trees softly blurred behind her, "
            "natural daylight on her face, a real photograph" + KEEP,
    "street": "Place this woman on a city street at golden hour, warm low sunlight "
              "from the side, shallow depth of field, a real photograph" + KEEP,
    "cafe": "Place this woman indoors in a cosy cafe with warm lamp light and a "
            "blurred background, a candid photograph" + KEEP,
    "studio": "Photograph this woman against a plain mid-grey studio backdrop with "
              "soft window light from the left" + KEEP,
    "sweater": "Change her clothes to a plain oatmeal knit sweater and place her in a "
               "bright living room, a real photograph" + KEEP,
    "neutral": "Give her a calm, relaxed expression with her lips closed, in soft "
               "daylight in front of a plain wall, a real photograph" + KEEP,
    "turn": "Turn her head three-quarters to her left, looking past the camera, "
            "outdoors in soft overcast light, a real photograph" + KEEP,
    "laugh": "Make her laugh naturally with her eyes crinkled, outdoors on a "
             "terrace in soft evening light, a real photograph" + KEEP,
}
DEFAULT_PROMPTS = "park,street,studio,neutral"
DROP = 0.15


def kontext_graph(image, prompt, seed, steps, guidance, prefix):
    return {
        "1": {"class_type": "UNETLoader", "inputs": {
            "unet_name": "flux1-dev-kontext_fp8_scaled.safetensors",
            "weight_dtype": "default"}},
        "2": {"class_type": "DualCLIPLoader", "inputs": {
            "clip_name1": "clip_l.safetensors", "clip_name2": "t5xxl_fp16.safetensors",
            "type": "flux", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
        "4": {"class_type": "LoadImage", "inputs": {"image": image}},
        "5": {"class_type": "FluxKontextImageScale", "inputs": {"image": ["4", 0]}},
        "6": {"class_type": "VAEEncode", "inputs": {"pixels": ["5", 0], "vae": ["3", 0]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "8": {"class_type": "ReferenceLatent", "inputs": {
            "conditioning": ["7", 0], "latent": ["6", 0]}},
        "9": {"class_type": "FluxGuidance", "inputs": {
            "conditioning": ["8", 0], "guidance": guidance}},
        "10": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["7", 0]}},
        "11": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "seed": seed, "steps": steps, "cfg": 1.0,
            "sampler_name": "euler", "scheduler": "simple", "positive": ["9", 0],
            "negative": ["10", 0], "latent_image": ["6", 0], "denoise": 1.0}},
        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["3", 0]}},
        "13": {"class_type": "SaveImage", "inputs": {
            "filename_prefix": prefix, "images": ["12", 0]}},
    }


def run(graph, timeout=600):
    """Queue a graph on the local ComfyUI and return its one picture's bytes."""
    pid = comfy.submit(graph)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        entry = comfy.history_entry(pid)
        if entry and (entry.get("status", {}).get("completed") or entry.get("outputs")
                      or entry.get("status", {}).get("status_str") == "error"):
            files = comfy.outputs_of(entry)
            if not files:
                raise comfy.ComfyError("; ".join(comfy.status_messages(entry))
                                       or "no picture came back")
            f = files[0]
            q = urllib.parse.urlencode({"filename": f["filename"],
                                        "subfolder": f["subfolder"], "type": f["type"]})
            return comfy.get_bytes("/view?" + q)
        time.sleep(1.0)
    raise comfy.ComfyError("prompt %s still running after %ds" % (pid, timeout))


class Faces:
    """ArcFace embeddings (antelopev2's glintr100) of the biggest face."""

    def __init__(self):
        from insightface.app import FaceAnalysis
        self.app = FaceAnalysis(name="antelopev2", root=INSIGHTFACE,
                                allowed_modules=["detection", "recognition"],
                                providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=-1, det_size=(640, 640))

    def embed(self, path):
        import cv2
        import numpy as np
        picture = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if picture is None:
            return None
        faces = self.app.get(picture)
        if not faces:
            return None
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return face.normed_embedding


def unit(v):
    import numpy as np
    return v / np.linalg.norm(v)


def sheet(rows, path):
    """rows: [(source path, [(picture path, label, kept)])] -> one JPEG."""
    from PIL import Image, ImageDraw, ImageOps
    cell, top = 260, 22
    cols = 1 + max(len(r[1]) for r in rows)
    out = Image.new("RGB", (cols * cell, len(rows) * (cell + top)), "white")
    draw = ImageDraw.Draw(out)
    for r, (source, made) in enumerate(rows):
        for c, (p, label, kept) in enumerate([(source, "reference", True)] + made):
            im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
            im.thumbnail((cell - 10, cell - 10))
            x, y = c * cell, r * (cell + top)
            out.paste(im, (x + 5, y + top))
            draw.text((x + 5, y + 5), label, fill="black" if kept else "red")
    out.save(path, quality=90)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("person", help="identity id, e.g. partner")
    ap.add_argument("--photos", default="", help="1-based reference numbers, e.g. 1,11,22 "
                    "(default: all)")
    ap.add_argument("--prompts", default=DEFAULT_PROMPTS,
                    help="comma list from: " + ", ".join(PROMPTS))
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--guidance", type=float, default=2.5)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--drop", type=float, default=DROP,
                    help="keep a variation scoring at most this far below its source")
    args = ap.parse_args()

    profiles = json.loads((LIBRARY / "identities.json").read_text(encoding="utf-8"))
    profile = next((p for p in profiles if p.get("id") == args.person), None)
    if profile is None:
        sys.exit("No identity %r in %s" % (args.person, LIBRARY / "identities.json"))
    refs = [Path(p) for p in profile.get("references") or []]
    pick = ([refs[int(n) - 1] for n in args.photos.split(",") if n.strip()]
            if args.photos else refs)
    prompts = [k.strip() for k in args.prompts.split(",") if k.strip()]
    unknown = [k for k in prompts if k not in PROMPTS]
    if unknown:
        sys.exit("Unknown prompts: %s (known: %s)" % (", ".join(unknown), ", ".join(PROMPTS)))

    out = LIBRARY / "variations" / args.person
    (out / "kept").mkdir(parents=True, exist_ok=True)
    (out / "dropped").mkdir(parents=True, exist_ok=True)

    import numpy as np
    faces = Faces()
    embedded = {r: faces.embed(r) for r in refs}
    good = {r: e for r, e in embedded.items() if e is not None}
    if len(good) < 2:
        sys.exit("ArcFace found a face in only %d references" % len(good))
    centre = unit(np.mean(list(good.values()), axis=0))
    # Each reference against the mean of the others: how alike real photos score.
    own = {r: float(np.dot(e, unit(np.mean([o for q, o in good.items() if q != r], axis=0))))
           for r, e in good.items()}
    median = float(np.median(list(own.values())))
    print("References with a face: %d of %d; their own scores %.2f-%.2f (median %.2f). "
          "Keeping variations at most %.2f below their source." % (
              len(good), len(refs), min(own.values()), max(own.values()), median,
              args.drop), flush=True)

    rng = random.Random(args.seed)
    report, rows = [], []
    with comfy.on(comfy.LOCAL_URL):
        for ref in pick:
            name = comfy.post_multipart("/upload/image", {"overwrite": "true"},
                                        "variation_src_" + ref.name, ref.read_bytes())
            image = (name.get("subfolder") + "/" if name.get("subfolder") else "") + name["name"]
            made = []
            threshold = own.get(ref, median) - args.drop
            for key in prompts:
                seed = rng.randrange(2 ** 32)
                started = time.monotonic()
                data = run(kontext_graph(image, PROMPTS[key], seed, args.steps,
                                         args.guidance, "variations/" + args.person))
                tmp = out / ("_%s_%s_%d.png" % (ref.stem, key, seed))
                tmp.write_bytes(data)
                e = faces.embed(tmp)
                score = float(np.dot(e, centre)) if e is not None else None
                kept = score is not None and score >= threshold
                final = out / ("kept" if kept else "dropped") / tmp.name[1:]
                os.replace(tmp, final)
                label = "%s %s" % (key, "no face" if score is None else "%.2f" % score)
                made.append((final, label, kept))
                report.append({"reference": str(ref), "prompt": key, "text": PROMPTS[key],
                               "seed": seed, "score": score, "threshold": threshold,
                               "kept": kept,
                               "file": str(final)})
                print("%s  %-8s %-8s %s  (%.0f s)" % (
                    ref.name, key, label.split(" ", 1)[1], "kept" if kept else "dropped",
                    time.monotonic() - started), flush=True)
            rows.append((ref, made))

    stamp = time.strftime("%Y%m%d-%H%M%S")
    sheet(rows, out / ("sheet-%s.jpg" % stamp))
    (out / ("report-%s.json" % stamp)).write_text(json.dumps({
        "person": args.person, "drop": args.drop,
        "reference_scores": {str(k): v for k, v in own.items()},
        "variations": report}, indent=1), encoding="utf-8")
    kept = sum(1 for r in report if r["kept"])
    print("DONE %d of %d kept. Sheet: %s" % (kept, len(report),
                                            out / ("sheet-%s.jpg" % stamp)), flush=True)


if __name__ == "__main__":
    main()
