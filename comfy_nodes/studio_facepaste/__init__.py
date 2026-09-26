"""
studio_facepaste - a ComfyUI node that puts a person's real face, from one
of their photos, over the face drawn for them, for Studio Assist's Scene
Builder. Last of all, after the PuLID face pass, and only where it is sure.

Why: PuLID draws a face *like* the person's; at the size a full-length shot
gives a face (60 px or so) it came out thinner and younger than them, and a
deeper redraw put a halo round the head (2026-09-26). Their own pixels are
them - but only when their photo shows the head at the angle the picture
drew it: a front-on face pasted over a turned head came out doubled
(2026-09-25). So for each face:

  1. InsightFace (antelopev2, the one PuLID installs) finds the drawn face
     near where the face pass found it, with its 106 points and its angle,
     and does the same for each of the person's photos (the biggest face).
  2. The photo whose yaw and pitch are nearest the drawn face's is taken,
     if the difference is within a tolerance that grows as the face gets
     smaller: a 10 degree miss shows on a close-up and vanishes at 60 px
     (TOLERANCE). Roll does not count - the alignment turns the photo.
     Otherwise the face is left as PuLID drew it.
  3. The photo is aligned on the inner face (eyes, brows, nose, mouth) with
     a similarity transform, so the face keeps its own shape and width.
  4. What is kept: the face's outline up to a forehead built above the
     brows - forehead, cheeks, jaw; not hair, not ears.
  5. The photo's colour and exposure are moved to the drawn face's (LAB
     mean and spread over the kept part), its sharpness to the picture's,
     and the picture's grain is added to make up the difference.
  6. It is blended in over a feathered edge. No redraw after.

It answers with the picture and a JSON report (`ui.text` and a STRING):
per face, the photo chosen, both angles, the tolerance, the confidence,
and whether it was pasted and why not.

Needs nothing ComfyUI's venv lacks once PuLID is installed: insightface,
onnxruntime, OpenCV, numpy, and models/insightface/models/antelopev2. The
source of truth is comfy_nodes/studio_facepaste in the Studio Assist repo;
copy it into ComfyUI's custom_nodes and restart ComfyUI.
"""

import json
import math
import os

import numpy as np

try:
    import folder_paths
except ImportError:                    # run outside ComfyUI (tests, trials)
    folder_paths = None

# (face width in the picture, px -> degrees of yaw/pitch difference allowed)
TOLERANCE = ((48, 22.0), (100, 14.0), (160, 8.0), (260, 6.0))
MIN_SCORE = 0.5            # InsightFace's detection score, drawn face and photo
MIN_REF_SCALE = 0.75       # the photo's face at least this wide next to the drawn one
FIND_REACH = 1.2           # the drawn face must sit within this many face widths of the box
INNER = list(range(33, 106))   # 2d106: the inner face; 0-32 the outline, NOT in order
FOREHEAD = 0.55            # how far above the eyes the kept forehead reaches, eye-chin heights
ERODE = 0.04               # of the face's width, the kept part shrinks...
FEATHER = 0.07             # ...and its edge softens over this much (sigma)
SPREAD = (0.6, 1.6)        # the LAB spread may be scaled this far to match
BLURS = (0.0, 0.4, 0.7, 1.0, 1.4, 1.9)   # sharpness candidates, sigma in px
SHRINK_BLUR = 0.6          # a photo is blurred at most this x sqrt(1/scale^2 - 1): no more
                           # than shrinking it sharpened it. Uncapped, a 140 px face from
                           # a soft 190 px photo was blurred 0.7 more - its beard read as
                           # detail to match (2026-09-26)

_APP = {}


def _root():
    if folder_paths is not None:
        return folder_paths.get_folder_paths("insightface")[0]
    return os.environ.get("INSIGHTFACE_ROOT", "D:/ComfyUI/models/insightface")


def analyser():
    """antelopev2 without its recognition model, on the CPU: about a second
    a picture, and no VRAM taken from FLUX (the same reason as studio_dwpose)."""
    if "app" not in _APP:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="antelopev2", root=_root(),
                           allowed_modules=["detection", "landmark_3d_68", "landmark_2d_106"],
                           providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        _APP["app"] = app
    return _APP["app"]


def _face(f, shift=(0.0, 0.0), k=1.0):
    """An InsightFace face as a dict, its points mapped back by /k + shift."""
    pitch, yaw, roll = (float(a) for a in f.pose)
    pts = f.landmark_2d_106.astype(np.float64) / k + np.array(shift)
    box = f.bbox.astype(np.float64) / k + np.array(shift * 2)
    kps = f.kps.astype(np.float64) / k + np.array(shift)
    return {"points": pts, "kps": kps, "box": box, "width": float(box[2] - box[0]),
            "yaw": yaw, "pitch": pitch, "roll": roll, "score": float(f.det_score)}


def faces_in(bgr):
    return [_face(f) for f in analyser().get(bgr)]


def face_near(bgr, box):
    """The drawn face nearest `box` (x, y, w, h), found in a crop around it
    brought to ~200 px a face, so a small face gets good points."""
    import cv2
    x, y, w, h = box
    side = 3.0 * max(w, h)
    cx, cy = x + w / 2.0, y + h / 2.0
    x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    x1, y1 = int(min(bgr.shape[1], cx + side / 2)), int(min(bgr.shape[0], cy + side / 2))
    crop = bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    k = 600.0 / max(crop.shape[:2])
    crop = cv2.resize(crop, None, fx=k, fy=k,
                      interpolation=cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA)
    best = None
    for f in analyser().get(crop):
        face = _face(f, (x0, y0), k)
        fx, fy = (face["box"][0] + face["box"][2]) / 2, (face["box"][1] + face["box"][3]) / 2
        d = math.hypot(fx - cx, fy - cy)
        if d <= FIND_REACH * max(w, h) and (best is None or d < best[0]):
            best = (d, face)
    return best[1] if best else None


def tolerance(width):
    """Degrees of yaw/pitch difference allowed for a face this wide (px)."""
    t = TOLERANCE
    if width <= t[0][0]:
        return t[0][1]
    for (w0, d0), (w1, d1) in zip(t, t[1:]):
        if width <= w1:
            return d0 + (d1 - d0) * (width - w0) / float(w1 - w0)
    return t[-1][1]


def judge(target, ref):
    """-> (angle difference, tolerance, confidence 0..1, why not or '')."""
    diff = math.hypot(target["yaw"] - ref["yaw"], target["pitch"] - ref["pitch"])
    tol = tolerance(target["width"])
    conf = max(0.0, 1.0 - diff / tol)
    why = ""
    if target["score"] < MIN_SCORE:
        why = "the drawn face is unclear"
    elif ref["score"] < MIN_SCORE:
        why = "the photo's face is unclear"
    elif ref["width"] < MIN_REF_SCALE * target["width"]:
        why = "the photo's face is too small for this close-up"
    elif diff > tol:
        why = "no photo at this angle (%.0f degrees off, %.0f allowed)" % (diff, tol)
    return diff, tol, conf, why


def _similarity(src, dst):
    """Least squares rotation + uniform scale + shift taking src onto dst."""
    import cv2
    m, _ = cv2.estimateAffinePartial2D(src.astype(np.float32), dst.astype(np.float32),
                                       method=cv2.LMEDS)
    return m


def keep_polygon(pts, kps):
    """The kept part, from a face's 106 points and 5 key points (eyes, nose,
    mouth corners) in the picture: the hull of its outline (2d106 does not
    number the outline in order round the jaw) and a half-ellipse of
    forehead over the eyes, as wide as the face."""
    import cv2
    outline = pts[0:33]
    eyes = kps[0:2].mean(0)
    up = eyes - kps[3:5].mean(0)
    up = up / max(float(np.hypot(*up)), 1e-6)
    across = np.array([-up[1], up[0]])
    height = float(((eyes - outline) @ up).max())          # eyes to chin
    half = float(np.abs((outline - eyes) @ across).max())
    arc = [eyes + across * half * math.cos(t) + up * height * FOREHEAD * math.sin(t)
           for t in np.linspace(0, math.pi, 17)]
    hull = cv2.convexHull(np.vstack([outline, np.array(arc)]).astype(np.float32))
    return hull.reshape(-1, 2)


def _lab(rgb):
    import cv2
    return cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2LAB)


def _rgb(lab):
    import cv2
    return np.clip(cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2RGB), 0, 1)


def _grain(gray, where):
    """The grain's spread in `where`: the finest detail's median deviation,
    which the edges of eyes and glasses do not sway as a spread would."""
    import cv2
    hf = (gray - cv2.GaussianBlur(gray, (0, 0), 1.0))[where]
    return float(np.median(np.abs(hf - np.median(hf))) * 1.4826)


def paste_one(img, ref_rgb, target, ref, seed=0):
    """`img` (H, W, 3 float RGB) with `ref`'s face from `ref_rgb` put over
    `target`. -> (new picture, what was done)."""
    import cv2
    h, w = img.shape[:2]
    full = _similarity(ref["points"][INNER], target["points"][INNER])
    m, scale = full, math.hypot(full[0, 0], full[1, 0])
    src = ref_rgb
    if scale < 1.0:                              # shrink first: warpAffine aliases
        src = cv2.resize(ref_rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        m = m.copy()
        m[:, :2] /= scale
    warped = cv2.warpAffine(src.astype(np.float32), m, (w, h), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT)
    to = lambda p: p @ full[:, :2].T + full[:, 2]
    fw = target["width"]
    mask = np.zeros((h, w), np.float32)
    cv2.fillPoly(mask, [np.round(keep_polygon(to(ref["points"]), to(ref["kps"]))).astype(np.int32)], 1.0)
    er = max(1, int(round(ERODE * fw)))
    core = cv2.erode(mask, np.ones((2 * er + 1, 2 * er + 1), np.uint8))
    soft = cv2.GaussianBlur(core, (0, 0), max(0.8, FEATHER * fw))
    inside = core > 0.5
    if inside.sum() < 20:
        return img, {"pasted": False, "why": "the face is too small to paste"}

    # sharpness: blur the photo until its detail matches the picture's
    gi = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2GRAY)
    band = lambda g: (cv2.GaussianBlur(g, (0, 0), 1.0) - cv2.GaussianBlur(g, (0, 0), 2.5))[inside].std()
    want = band(gi)
    best = None
    cap = SHRINK_BLUR * math.sqrt(max(0.0, 1.0 / min(scale, 1.0) ** 2 - 1.0))
    for s in (b for b in BLURS if b <= cap):
        cand = cv2.GaussianBlur(warped, (0, 0), s) if s else warped
        d = abs(band(cv2.cvtColor(cand, cv2.COLOR_RGB2GRAY)) - want)
        if best is None or d < best[0]:
            best = (d, s, cand)
    _, blur, warped = best

    # colour and exposure: the photo's LAB mean and spread to the drawn face's
    lt, lr = _lab(img), _lab(warped)
    for c in range(3):
        mt, st = lt[..., c][inside].mean(), lt[..., c][inside].std()
        mr, sr = lr[..., c][inside].mean(), lr[..., c][inside].std()
        k = float(np.clip(st / sr, *SPREAD)) if sr > 1e-4 else 1.0
        lr[..., c] = (lr[..., c] - mr) * k + mt
    matched = _rgb(lr)

    # grain: what the picture has beyond the photo's, as luminance noise
    gt = _grain(gi, inside)
    gr = _grain(cv2.cvtColor(matched, cv2.COLOR_RGB2GRAY), inside)
    grain = float(math.sqrt(max(0.0, gt * gt - gr * gr)))
    if grain > 1e-4:
        rng = np.random.default_rng(seed)
        n = cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), 0.5)
        n *= grain / max(1e-6, n.std())
        matched = np.clip(matched + n[..., None], 0, 1)

    a = soft[..., None]
    return img * (1 - a) + matched * a, {"pasted": True, "blur": blur,
                                         "grain": round(grain, 4)}


def paste(img, faces, load_ref, seed=0):
    """img: (H, W, 3) float RGB 0..1. faces: [{"name", "box": [x, y, w, h],
    "references": [names]}]. load_ref(name) -> (H, W, 3) float RGB.
    -> (picture, report [dict per face])."""
    bgr = np.ascontiguousarray((img[:, :, ::-1] * 255).clip(0, 255).astype(np.uint8))
    refs, report = {}, []
    out = img.copy()
    for i, spec in enumerate(faces):
        r = {"name": spec.get("name", ""), "pasted": False}
        report.append(r)
        target = face_near(bgr, spec["box"])
        if target is None:
            r["why"] = "the drawn face was not found"
            continue
        r["face"] = {"yaw": round(target["yaw"], 1), "pitch": round(target["pitch"], 1),
                     "width": round(target["width"], 1)}
        best = None
        for name in spec.get("references") or []:
            if name not in refs:
                try:
                    rgb = load_ref(name)
                    found = faces_in(np.ascontiguousarray(
                        (rgb[:, :, ::-1] * 255).clip(0, 255).astype(np.uint8)))
                except Exception as e:           # a photo that will not load is skipped
                    refs[name] = (None, None, "could not be read (%s)" % e)
                    continue
                found.sort(key=lambda f: -f["width"])
                refs[name] = (rgb, found[0] if found else None, "" if found else "no face")
            rgb, ref, bad = refs[name]
            if ref is None:
                r.setdefault("skipped", []).append({"reference": name, "why": bad})
                continue
            diff, tol, conf, why = judge(target, ref)
            key = (bool(why), diff, -ref["width"])
            if best is None or key < best[0]:
                best = (key, name, rgb, ref, diff, tol, conf, why)
        if best is None:
            r["why"] = "no usable photo"
            continue
        _, name, rgb, ref, diff, tol, conf, why = best
        r.update({"reference": name, "reference_angle": {"yaw": round(ref["yaw"], 1),
                                                          "pitch": round(ref["pitch"], 1)},
                  "difference": round(diff, 1), "tolerance": round(tol, 1),
                  "confidence": round(conf, 2)})
        if why:
            r["why"] = why
            continue
        out, done = paste_one(out, rgb, target, ref, seed + i)
        r.update(done)
    return out, report


def _load_input(name):
    import cv2
    path = folder_paths.get_annotated_filepath(name)
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise IOError("cannot read %s" % name)
    return bgr[:, :, ::-1].astype(np.float32) / 255.0


class StudioFacePaste:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",),
                             "faces": ("STRING", {"multiline": True, "default": "[]"}),
                             "seed": ("INT", {"default": 0, "min": 0, "max": 2 ** 32 - 1})}}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "report")
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "Studio Assist"

    def run(self, image, faces, seed=0):
        import torch
        img = image[0].cpu().numpy().astype(np.float32)
        out, report = paste(img, json.loads(faces or "[]"), _load_input, seed)
        text = json.dumps(report)
        return {"ui": {"text": [text]},
                "result": (torch.from_numpy(out.astype(np.float32))[None], text)}


NODE_CLASS_MAPPINGS = {"StudioFacePaste": StudioFacePaste}
NODE_DISPLAY_NAME_MAPPINGS = {"StudioFacePaste": "Real face paste (Studio Assist)"}
