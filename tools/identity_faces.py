"""Find one person's face in photos, and cut photos down to them: the worker
behind importing an identity's reference photos (`faces.crop`) and Build
LoRA's head squares (`faces.find`).

Run in ComfyUI's venv (InsightFace with antelopev2, PIL, numpy, OpenCV - the
app has none of them) by `apps.image_studio.faces` with a job.json:

    {"mode": "find" | "crop", "refs": [...], "photos": [...],
     "insightface": <root holding models/antelopev2>, "cache": <json>,
     "out": <folder for crops>, "result": <json written at the end>}

Prints `FACE n total` as each photo is read, then `DONE` or `ERROR <words>`.

Who the person is: the mean ArcFace embedding of their faces, seeded by the
refs that hold exactly one face (else the photos that do), then refined three
times on the best-matching face of every picture. A crowd cannot pull it:
the 2026-10-01 wedding photos hold ~10 faces each, and a mean of every face
there was nobody. A face is theirs at SAME or more - on Partner's 135 photos
the other people's best faces scored 0.34 at most, hers 0.53 at least.

`find` gives each photo the person's face as the square OpenCV's cascade
would have drawn round it (HAAR_SCALE), so Build LoRA's head square (2.4 of
those, tuned on the cascade) is unchanged. `crop` cuts each photo to the
person's head and shoulders (KEEP_W x KEEP_H faces), slid inside the photo;
a photo they already fill (KEEP_WHOLE) is left whole.
"""
import base64
import json
import os
import sys

SAME = 0.42         # ArcFace cosine to the person's mean: theirs at this or more
REFINE = 0.45       # faces trusted enough to refine the mean with
HAAR_SCALE = 1.27   # the cascade's square over InsightFace's face width (median, 103 faces)
KEEP_W, KEEP_H = 4.0, 5.0   # the crop, in face widths: head, hair, shoulders
KEEP_ABOVE = 1.0    # face widths above the face's top kept for the hair
KEEP_WHOLE = 0.8    # a crop this much of the photo or more: keep the photo whole
MIN_SCORE = 0.5     # detector confidence


def say(text):
    print(text, flush=True)


# ------------------------------------------------- pure helpers (tested)
def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def unit(v):
    n = dot(v, v) ** 0.5 or 1.0
    return [x / n for x in v]


def mean(vs):
    return unit([sum(col) / len(vs) for col in zip(*vs)])


def person(ref_groups, photo_groups):
    """The person's mean embedding from each picture's faces (lists of
    embeddings) -> a unit vector, or None when no picture holds them alone."""
    seeds = [g[0] for g in ref_groups if len(g) == 1] or \
            [g[0] for g in photo_groups if len(g) == 1]
    if not seeds:
        return None
    c = mean(seeds)
    groups = [g for g in list(ref_groups) + list(photo_groups) if g]
    for _ in range(3):
        best = [max(g, key=lambda e: dot(e, c)) for g in groups]
        best = [e for e in best if dot(e, c) >= REFINE]
        if not best:
            break
        c = mean(best)
    return c


def theirs(faces, c):
    """The index of the person's face among `faces` (embeddings) and its
    similarity; index None when none is theirs."""
    if not faces or c is None:
        return None, 0.0
    sims = [dot(e, c) for e in faces]
    k = max(range(len(sims)), key=sims.__getitem__)
    return (k if sims[k] >= SAME else None), sims[k]


def haar_square(box):
    """InsightFace's face (x0, y0, x1, y1) -> the cascade-sized square round
    it, (x, y, w, h) - what train_identity_lora.head_square is tuned on."""
    x0, y0, x1, y1 = box
    side = (x1 - x0) * HAAR_SCALE
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    return [int(round(cx - side / 2)), int(round(cy - side / 2)), int(round(side)),
            int(round(side))]


def keep_box(width, height, box):
    """The head-and-shoulders crop round face `box` (x0, y0, x1, y1) in a
    width x height photo -> (left, top, right, bottom), or None when it
    would keep KEEP_WHOLE of the photo or more."""
    x0, y0, x1, y1 = box
    fw = x1 - x0
    w, h = min(width, KEEP_W * fw), min(height, KEEP_H * fw)
    if w * h >= KEEP_WHOLE * width * height:
        return None
    left = min(max((x0 + x1) / 2.0 - w / 2.0, 0), width - w)
    top = min(max(y0 - KEEP_ABOVE * fw, 0), height - h)
    return int(left), int(top), int(left + w), int(top + h)


# ------------------------------------------------------ the face reader
class Reader:
    """InsightFace's faces in a picture, through a cache keyed by the file's
    size and time: [{"box": [x0, y0, x1, y1], "emb": [512 floats]}]."""

    def __init__(self, root, cache_path):
        model = os.path.join(root, "models", "antelopev2", "glintr100.onnx")
        if not os.path.isfile(model):
            # FaceAnalysis would download 360 MB unasked into the wrong place.
            raise RuntimeError("antelopev2 is not at %s" % os.path.dirname(model))
        self.root, self.app = root, None
        self.cache_path = cache_path
        self.cache = {}
        if cache_path and os.path.isfile(cache_path):
            try:
                with open(cache_path, encoding="utf-8") as f:
                    self.cache = json.load(f)
            except (OSError, ValueError):
                self.cache = {}

    def model(self):
        if self.app is None:
            from insightface.app import FaceAnalysis
            self.app = FaceAnalysis(name="antelopev2", root=self.root,
                                    allowed_modules=["detection", "recognition"],
                                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            self.app.prepare(ctx_id=0, det_size=(640, 640))
        return self.app

    @staticmethod
    def stamp(path):
        st = os.stat(path)
        return [st.st_size, st.st_mtime_ns]

    def faces(self, path, image=None):
        import numpy as np
        key = os.path.normcase(os.path.abspath(path))
        hit = self.cache.get(key)
        if hit and hit.get("stamp") == self.stamp(path):
            return [{"box": f["box"], "emb": np.frombuffer(base64.b64decode(f["emb"]),
                                                            np.float16).astype(float).tolist()}
                    for f in hit["faces"]]
        im = image if image is not None else upright(path)
        bgr = np.ascontiguousarray(np.asarray(im)[:, :, ::-1])
        found = self.detect(bgr)
        out = [{"box": [float(v) for v in f.bbox],
                "emb": [float(v) for v in f.normed_embedding]} for f in found]
        self.cache[key] = {"stamp": self.stamp(path), "faces": [
            {"box": f["box"], "emb": base64.b64encode(
                np.asarray(f["emb"], np.float16).tobytes()).decode()} for f in out]}
        return out

    def detect(self, bgr):
        """Faces at 640, else at 1280 (small faces in a big photo), else with
        a grey border (a face cut by the photo's edge)."""
        import cv2
        app = self.model()
        faces = [f for f in app.get(bgr) if f.det_score >= MIN_SCORE]
        if not faces:
            app.det_model.input_size = (1280, 1280)
            try:
                faces = [f for f in app.get(bgr) if f.det_score >= MIN_SCORE]
            finally:
                app.det_model.input_size = (640, 640)
        if not faces:
            h, w = bgr.shape[:2]
            pad = max(w, h) // 4
            padded = cv2.copyMakeBorder(bgr, pad, pad, pad, pad, cv2.BORDER_CONSTANT,
                                        value=(128, 128, 128))
            faces = [f for f in app.get(padded) if f.det_score >= MIN_SCORE]
            for f in faces:
                b = f.bbox - pad
                f.bbox = [max(b[0], 0), max(b[1], 0), min(b[2], w), min(b[3], h)]
        return faces

    def save(self):
        if not self.cache_path:
            return
        tmp = self.cache_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.cache, f)
        os.replace(tmp, self.cache_path)


def upright(path):
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def run(job):
    reader = Reader(job["insightface"], job.get("cache"))
    refs = [p for p in job.get("refs") or () if os.path.isfile(p)]
    photos = list(job.get("photos") or ())
    todo = list(dict.fromkeys(refs + photos))
    read, total = {}, len(todo)
    for i, p in enumerate(todo, 1):
        try:
            read[p] = reader.faces(p)
        except Exception as exc:      # one unreadable photo must not stop the rest
            read[p] = exc
        say("FACE %d %d" % (i, total))
        if i % 20 == 0:
            reader.save()
    reader.save()

    def embs(p):
        got = read.get(p)
        return [f["emb"] for f in got] if isinstance(got, list) else []
    c = person([embs(p) for p in refs], [embs(p) for p in photos])
    out = {}
    for p in photos:
        got = read.get(p)
        if not isinstance(got, list):
            out[p] = {"box": None, "why": "unreadable: %s" % got}
            continue
        k, sim = theirs([f["emb"] for f in got], c)
        r = {"faces": len(got), "sim": round(sim, 3)}
        if c is None:
            r.update(box=None, why="no photo shows them alone")
        elif not got:
            r.update(box=None, why="no face")
        elif k is None:
            r.update(box=None, why="not found")    # not them, or too small or turned to tell
        else:
            r["box"] = haar_square(got[k]["box"])
            if job["mode"] == "crop":
                r["crop"] = cut(p, got[k]["box"], job["out"], len(out))
        out[p] = r
    return out


def cut(path, box, folder, index):
    """The photo cut to the person's head and shoulders, saved in `folder`
    -> the crop's path, or None when the photo is kept whole."""
    im = upright(path)
    keep = keep_box(im.size[0], im.size[1], box)
    if keep is None:
        return None
    stem, ext = os.path.splitext(os.path.basename(path))
    png = ext.lower() == ".png"
    dest = os.path.join(folder, "%03d_%s%s" % (index, stem, ".png" if png else ".jpg"))
    crop = im.crop(keep)
    if png:
        crop.save(dest)
    else:
        crop.save(dest, quality=95)
    return dest


def main():
    with open(sys.argv[1], encoding="utf-8") as f:
        job = json.load(f)
    try:
        if job.get("out"):
            os.makedirs(job["out"], exist_ok=True)
        out = run(job)
    except Exception as exc:
        say("ERROR %s" % exc)
        return 1
    with open(job["result"], "w", encoding="utf-8") as f:
        json.dump(out, f)
    say("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
