"""Find one person's face in photos, and cut photos down to them: the worker
behind importing an identity's reference photos (`faces.crop`) and Build
LoRA's head squares (`faces.find`).

Run in ComfyUI's venv (InsightFace with antelopev2, PIL, numpy, OpenCV - the
app has none of them) by `apps.image_studio.faces` with a job.json:

    {"mode": "find" | "crop" | "rate", "refs": [...], "photos": [...],
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
a photo they already fill (KEEP_WHOLE) is left whole. `rate` gives what
`faces.score` rates a photo on as LoRA training data: the training square's
side, the face's sharpness and light, its head angle, the other faces in the
square, whether the picture was made here (a cut-out on white, a Kontext
edit - from the graph in its PNG), and its near twins (`near`: ArcFace
similarity and the distance between the faces' dHashes).
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
MARGIN = 2.4        # train_identity_lora's head square, in cascade squares
NEAR = 0.88         # rate: twins worth reporting (faces.DUP_* decide)
VERSION = 2         # cache entries: 2 adds size, kind and each face's measures


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


def share_inside(square, box):
    """How much of `box` (x0, y0, x1, y1) lies inside `square` (same form)."""
    ix = max(0.0, min(square[2], box[2]) - max(square[0], box[0]))
    iy = max(0.0, min(square[3], box[3]) - max(square[1], box[1]))
    return ix * iy / max(1e-6, (box[2] - box[0]) * (box[3] - box[1]))


def hamming(a, b):
    """Bits apart of two hex dHashes."""
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def kind_of(prompt):
    """A picture's ComfyUI graph (its PNG "prompt" text) -> "cutout" (a
    person cut out onto white), "generated" (any other graph, e.g. a Kontext
    edit), or "photo" when it has none."""
    if not prompt:
        return "photo"
    try:
        types = {n.get("class_type") for n in json.loads(prompt).values()}
    except (ValueError, AttributeError):
        return "photo"
    if {"SAM3_Detect", "ImageCompositeMasked", "EmptyImage"} <= types:
        return "cutout"
    return "generated"


# ------------------------------------------------------ the face reader
class Reader:
    """InsightFace's faces in a picture, through a cache keyed by the file's
    size and time: [{"box": [x0, y0, x1, y1], "emb": [512 floats], "pose":
    [pitch, yaw, roll], "sharp": Laplacian variance at 256 px wide, "luma",
    "clip": share near black or white, "dhash": 16 hex}]; `meta(path)` is
    the picture's {"size": [w, h], "kind"}."""

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
                                    allowed_modules=["detection", "recognition",
                                                     "landmark_3d_68"],     # pose
                                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            self.app.prepare(ctx_id=0, det_size=(640, 640))
        return self.app

    @staticmethod
    def stamp(path):
        st = os.stat(path)
        return [st.st_size, st.st_mtime_ns]

    @staticmethod
    def key(path):
        return os.path.normcase(os.path.abspath(path))

    def faces(self, path):
        import numpy as np
        hit = self.cache.get(self.key(path))
        if hit and hit.get("stamp") == self.stamp(path) and hit.get("v") == VERSION:
            return [dict(f, emb=np.frombuffer(base64.b64decode(f["emb"]),
                                              np.float16).astype(float).tolist())
                    for f in hit["faces"]]
        im, prompt = upright(path, with_prompt=True)
        rgb = np.asarray(im)
        found = self.detect(np.ascontiguousarray(rgb[:, :, ::-1]))
        gray = np.asarray(im.convert("L"))
        out = []
        for f in found:
            box = [float(v) for v in f.bbox]
            pose = getattr(f, "pose", None)
            out.append(dict(measure(gray, box), box=box,
                            emb=[float(v) for v in f.normed_embedding],
                            pose=None if pose is None else [round(float(v), 1) for v in pose]))
        self.cache[self.key(path)] = {
            "v": VERSION, "stamp": self.stamp(path), "size": list(im.size),
            "kind": kind_of(prompt),
            "faces": [dict(f, emb=base64.b64encode(np.asarray(f["emb"], np.float16)
                                                   .tobytes()).decode()) for f in out]}
        return out

    def meta(self, path):
        hit = self.cache.get(self.key(path)) or {}
        return {"size": hit.get("size"), "kind": hit.get("kind", "photo")}

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
        """The cache to disk. A temp file of this worker's own: Rate photos
        beside an import or Build LoRA's face find run workers at once, and
        a shared name failed one's os.replace (WinError 5). A cache that will
        not save is a note - the faces are read again next time - never the
        job's failure."""
        if not self.cache_path:
            return
        tmp = "%s.%d.tmp" % (self.cache_path, os.getpid())
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.cache, f)
            os.replace(tmp, self.cache_path)
        except OSError as exc:
            say("The face cache was not saved (%s); its faces are read again next time." % exc)
            try:
                os.remove(tmp)
            except OSError:
                pass


def upright(path, with_prompt=False):
    """The picture upright as RGB (and, `with_prompt`, its PNG's ComfyUI
    graph text or None)."""
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        prompt = im.info.get("prompt")
        im = ImageOps.exif_transpose(im).convert("RGB")
    return (im, prompt) if with_prompt else im


def measure(gray, box):
    """Sharpness, light and a dHash of the face `box` in a grey picture."""
    import cv2
    import numpy as np
    h, w = gray.shape[:2]
    x0, y0 = max(0, int(box[0])), max(0, int(box[1]))
    x1, y1 = min(w, int(box[2])), min(h, int(box[3]))
    face = gray[y0:y1, x0:x1]
    if face.size == 0:
        return {"sharp": 0.0, "luma": 0.0, "clip": 1.0, "dhash": "0" * 16}
    fh = max(1, int(round(256.0 * face.shape[0] / face.shape[1])))
    small = cv2.resize(face, (256, fh), interpolation=cv2.INTER_AREA
                       if face.shape[1] > 256 else cv2.INTER_CUBIC)
    luma = small.astype(np.float32) / 255.0
    m = 0.2 * (x1 - x0)                       # a little round the face for the hash
    around = gray[max(0, int(y0 - m)):min(h, int(y1 + m)), max(0, int(x0 - m)):min(w, int(x1 + m))]
    tiny = cv2.resize(around, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
    bits = (tiny[:, 1:] > tiny[:, :-1]).flatten()
    return {"sharp": round(float(cv2.Laplacian(small, cv2.CV_64F).var()), 1),
            "luma": round(float(luma.mean()), 3),
            "clip": round(float(((luma > 0.98) | (luma < 0.02)).mean()), 3),
            "dhash": "%016x" % int("".join("1" if b else "0" for b in bits), 2)}


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
    out, mine = {}, {}
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
            elif job["mode"] == "rate":
                r.update(rated(reader.meta(p), got, k, r["box"]))
                mine[p] = got[k]
        out[p] = r
    if job["mode"] == "rate":         # near twins of the person's face, both ways
        names = list(mine)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                sim = dot(mine[a]["emb"], mine[b]["emb"])
                if sim >= NEAR:
                    apart = hamming(mine[a]["dhash"], mine[b]["dhash"])
                    out[a].setdefault("near", []).append([b, round(sim, 3), apart])
                    out[b].setdefault("near", []).append([a, round(sim, 3), apart])
    return out


def rated(meta, faces, k, square):
    """What `faces.score` needs of the person's face `faces[k]` in a picture
    of `meta`, whose cascade-sized square is `square`."""
    from train_identity_lora import head_square
    w, h = meta["size"]
    left, top, right, bottom = head_square(w, h, square)
    me = faces[k]
    width = me["box"][2] - me["box"][0]
    others = [round((f["box"][2] - f["box"][0]) / max(width, 1.0), 2)
              for j, f in enumerate(faces)
              if j != k and share_inside((left, top, right, bottom), f["box"]) > 0.25]
    pitch, yaw = (me["pose"] or [0.0, 0.0])[:2]
    return {"size": [w, h], "kind": meta["kind"], "side": right - left,
            "sharp": me["sharp"], "luma": me["luma"], "clip": me["clip"],
            "yaw": yaw, "pitch": pitch, "others": others}


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
        # Inside the try: a full disk is an ERROR, not half an answer
        # followed by a traceback. DONE only once the file is whole.
        with open(job["result"], "w", encoding="utf-8") as f:
            json.dump(out, f)
    except Exception as exc:
        say("ERROR %s" % exc)
        return 1
    say("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
