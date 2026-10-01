"""One person's face in photos, found by ArcFace: reference photos are cut
to the person as they are imported (`crop_for_import`), and Build LoRA's
head squares are drawn round the person rather than round the biggest face
(`find`).

Why (2026-10-01): Partner's identity took 126 wedding photos of 6720 x 4480,
most of them group shots. Build LoRA's worker found faces with OpenCV's
cascades, which skip any face under a twelfth of the photo's short side
(373 px there) - 49 photos went in whole, her face ~20 px of a 768 px
picture - and took the BIGGEST face, someone else's in 27 more. Only her
embedding tells her from the bridesmaids.

The work is `tools/identity_faces.py`, in ComfyUI's venv on this PC
(InsightFace and antelopev2 are ComfyUI's, for PuLID), run as a contained
child like the LoRA build. Faces read once are kept in a cache beside the
library (`face-cache.json`), so the next build or import reads only new
photos. This module is stdlib only.
"""

if __package__ in (None, ""):  # run as a script: import from the checkout
    import os as _os, sys as _sys
    _sys.path[0] = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))

import json
import math
import os
import re
import shutil
import subprocess
import tempfile

import core.procs as studio_procs

COMFYUI = os.environ.get("STUDIO_COMFYUI", r"D:\ComfyUI")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPT = os.path.join(ROOT, "tools", "identity_faces.py")
CACHE = "face-cache.json"       # in the library's folder


def python_exe(comfy=COMFYUI):
    return os.path.join(comfy, "venv", "Scripts" if os.name == "nt" else "bin",
                        "python.exe" if os.name == "nt" else "python")


def model_root(comfy=COMFYUI):
    """InsightFace's root: antelopev2 is in <root>/models/antelopev2."""
    return os.path.join(comfy, "models", "insightface")


def problem(comfy=COMFYUI):
    """What stops finding faces on this PC, in words; None when nothing does."""
    if not os.path.isfile(python_exe(comfy)):
        return "ComfyUI's venv is not at %s." % os.path.join(comfy, "venv")
    if not os.path.isfile(os.path.join(model_root(comfy), "models", "antelopev2",
                                       "glintr100.onnx")):
        return "InsightFace's antelopev2 is not in %s." % model_root(comfy)
    return None


def parse(line):
    """One line from the worker -> ("face", (n, total)) | ("done", "") |
    ("error", text) | ("note", text)."""
    line = line.strip()
    m = re.match(r"^FACE (\d+) (\d+)$", line)
    if m:
        return "face", (int(m.group(1)), int(m.group(2)))
    if line == "DONE":
        return "done", ""
    if line.startswith("ERROR "):
        return "error", line[6:].strip()
    return "note", line


class Job:
    """One run of the worker as a contained child. `run()` blocks (a worker
    thread's), calling `on("face", (n, total))`, and returns its answer
    {photo: {"box": [x, y, w, h] | None, "why": ..., "crop": path | None}}
    or raises RuntimeError in words. `stop()` from any thread ends it."""

    def __init__(self, mode, refs, photos, cache=None, out=None, comfy=COMFYUI):
        self.job = {"mode": mode, "refs": list(refs or ()), "photos": list(photos),
                    "cache": cache, "out": out, "insightface": model_root(comfy)}
        self.comfy = comfy
        self.child = None
        self.stopped = False

    def run(self, on=lambda kind, value: None):
        folder = tempfile.mkdtemp(prefix="studio-faces-")
        try:
            self.job["result"] = os.path.join(folder, "result.json")
            path = os.path.join(folder, "job.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.job, f)
            env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
            self.child = studio_procs.spawn(
                [python_exe(self.comfy), SCRIPT, path], cwd=folder,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, env=env, creationflags=studio_procs.NO_WINDOW)
            if self.stopped:
                self.child.kill()
            error, tail, done = None, [], False
            try:
                for raw in iter(self.child.proc.stdout.readline, b""):
                    kind, value = parse(raw.decode("utf-8", "replace"))
                    if kind == "face":
                        on(kind, value)
                    elif kind == "done":
                        done = True
                    elif kind == "error":
                        error = value
                    elif kind == "note" and value:
                        tail = (tail + [value])[-3:]
                self.child.proc.wait()
            finally:
                # Also when `on` raised: nothing reads the worker after this,
                # and it would hold InsightFace on the GPU unwatched (as
                # Build.run).
                self.child.kill()
            if self.stopped:
                raise RuntimeError("Stopped.")
            # DONE, not the file: the worker says it only once its answer is
            # whole; a worker that died writing it leaves half a file.
            if error or not done:
                raise RuntimeError(error or "the face finder stopped: %s"
                                   % (" | ".join(tail) or "it said nothing"))
            try:
                with open(self.job["result"], encoding="utf-8") as f:
                    return json.load(f)
            except (OSError, ValueError) as e:
                raise RuntimeError("the face finder's answer could not be read (%s)" % e)
        finally:
            shutil.rmtree(folder, ignore_errors=True)

    def stop(self):
        self.stopped = True
        if self.child is not None:
            self.child.kill()


def crop_for_import(paths, refs, cache, scratch, name="the person", on=lambda k, v: None,
                    comfy=COMFYUI):
    """The photos to import: each cut to the person's head and shoulders
    when it can be (the crop is written in `scratch`), else the photo as it
    is -> (paths, a note in words for the editor's status)."""
    paths = list(paths)
    if not paths:
        return paths, ""
    why = problem(comfy)
    if why:
        return paths, "Not cut to %s: %s" % (name, why)
    try:
        found = Job("crop", refs, paths, cache=cache, out=scratch, comfy=comfy).run(on)
    except (RuntimeError, OSError) as e:
        return paths, "Not cut to %s: %s" % (name, e)
    out, cut, close, missing = [], 0, 0, []
    for p in paths:
        r = found.get(p) or {}
        if r.get("crop"):
            out.append(r["crop"])
            cut += 1
        else:
            out.append(p)
            if r.get("box"):
                close += 1
            else:
                missing.append(r.get("why") or "not read")
    parts = []
    if cut:
        parts.append("%d cut to %s" % (cut, name))
    if close:
        parts.append("%d already close on them" % close)
    if missing:
        counts = {}
        for w in missing:
            w = "unreadable" if w.startswith("unreadable") else w
            counts[w] = counts.get(w, 0) + 1
        parts.append("%d kept whole (%s)" % (len(missing), ", ".join(
            "%s: %d" % kv for kv in sorted(counts.items()))))
    return out, "; ".join(parts)


# --------------------------------------------------------- LoRA ratings
# How good each photo is as Build LoRA's training data, judged on the head
# square the worker trains on (2026-10-01, the user: "a set of criteria that
# rates the images on best candidacy for lora", then "i want the ui to show
# me the ratings also" and "remove duplicates"). A photo the person is not
# found in scores 0: the build leaves it out.
WEIGHTS = {"likeness": 25, "resolution": 20, "sharpness": 20, "clean": 15,
           "light": 10, "real": 10}
# A cut-out's white and a Kontext edit's look are learned as the person's.
REAL = {"photo": 1.0, "cutout": 0.4, "generated": 0.5}
TRAIN_SIDE = 512        # lora_train.RESOLUTION: a smaller square is enlarged
# Twins (measured on Partner's 123): 0.95+ is the same face - an Angles or
# variations picture beside its source, only the background changed; 0.90
# with faces whose dHashes are 16 bits or fewer apart is a burst a moment
# apart; ~0.88 is another moment of the same day, worth keeping.
DUP_SAME, DUP_SIM, DUP_HASH = 0.95, 0.90, 16
TOP = 30                # the suggested set
ANGLES = ((-45, "profile left"), (-15, "three-quarter left"), (15, "front"),
          (45, "three-quarter right"), (999, "profile right"))


def _clip(v):
    return max(0.0, min(1.0, float(v)))


def angle(yaw):
    return next(name for edge, name in ANGLES if yaw <= edge)


def score(m):
    """One photo's rating from the worker's `rate` answer `m` -> {"score":
    0-100, "parts": {criterion: 0-1}, "flags": [words]}."""
    if not m.get("box"):
        return {"score": 0, "parts": {}, "flags": ["%s: left out of a LoRA" % (
            m.get("why") or "not found")]}
    flags = []
    like = _clip((m.get("sim", 0) - 0.35) / 0.4)
    side = m.get("side") or 0
    res = _clip((side - 160) / float(TRAIN_SIDE - 160))
    if side < 300:
        flags.append("small face: enlarged %.1fx to train" % (TRAIN_SIDE / float(max(side, 1))))
    lap = max(m.get("sharp") or 0, 1.0)
    sharp = _clip((math.log10(lap) - 1.4) / 1.0)
    if sharp < 0.35:
        flags.append("soft face")
    others = m.get("others") or []
    clean = 1.0 if not others else _clip(0.5 - 0.5 * max(others))
    if others:
        flags.append("%d other face%s in the square" % (len(others), "" if len(others) == 1
                                                         else "s"))
    luma, clipped = m.get("luma", 0.5), m.get("clip", 0)
    light = _clip(1 - abs(luma - 0.5) / 0.35) * _clip(1 - clipped * 4)
    if light < 0.4:
        flags.append("face too dark" if luma < 0.5 else "face too bright")
    kind = m.get("kind") or "photo"
    real = REAL.get(kind, 1.0)
    if kind == "cutout":
        flags.append("cut out on white")
    elif kind == "generated":
        flags.append("made here, not a photo")
    parts = {"likeness": like, "resolution": res, "sharpness": sharp, "clean": clean,
             "light": light, "real": real}
    total = sum(WEIGHTS[k] * v for k, v in parts.items())
    return {"score": int(round(total)), "parts": {k: round(v, 2) for k, v in parts.items()},
            "flags": flags}


def twins(m):
    """The photos the worker found to be `m`'s duplicate (see DUP_*)."""
    return [other for other, sim, apart in m.get("near") or ()
            if sim >= DUP_SAME or (sim >= DUP_SIM and apart <= DUP_HASH)]


def rate(found, paths, keep_first=True):
    """Every photo in `paths` rated -> {path: score() + {"dup_of": path or
    None, "top": bool, "angle": words}}. Of each set of duplicates the best
    stays and the rest are `dup_of` it; with `keep_first`, the first photo
    (the Primary) always stays. `top` is the suggested TOP: best first, with
    a bonus for a head angle the set has few of."""
    out = {p: dict(score(found.get(p) or {}), dup_of=None, top=False,
                   angle=angle((found.get(p) or {}).get("yaw") or 0)
                   if (found.get(p) or {}).get("box") else "")
           for p in paths}
    order = sorted(paths, key=lambda p: (not (keep_first and p == paths[0]),
                                         -out[p]["score"]))
    kept = set()
    for p in order:
        if out[p]["score"] == 0:
            continue
        twin = next((q for q in twins(found.get(p) or {}) if q in kept), None)
        if twin is not None:
            out[p]["dup_of"] = twin
            out[p]["flags"].append("duplicate of a better photo")
        else:
            kept.add(p)
    pool = [p for p in paths if p in kept]
    counts = {}
    for _ in range(min(TOP, len(pool))):
        best = max(pool, key=lambda p: out[p]["score"] + 12.0 / (1 + counts.get(out[p]["angle"], 0)))
        pool.remove(best)
        out[best]["top"] = True
        counts[out[best]["angle"]] = counts.get(out[best]["angle"], 0) + 1
    return out


def best_first(rated, paths):
    """`paths` sorted for the editor after a rating (the user: "it should auto
    sort them for the best pic to set as the primary"): first the Primary -
    the best-scoring front view, since the Primary is the face the picture
    is matched to, else the best of any angle - then the other usable photos
    best first, then the duplicates, then the photos the person is not found
    in. Ties keep their order."""
    def rank(p):
        r = rated[p]
        return (r["score"] == 0, bool(r["dup_of"]), -r["score"])
    usable = [p for p in paths if rated[p]["score"] and not rated[p]["dup_of"]]
    front = [p for p in usable if rated[p]["angle"] == "front"]
    first = max(front or usable, key=lambda p: rated[p]["score"], default=None)
    rest = sorted((p for p in paths if p != first), key=rank)
    return ([first] if first else []) + rest


def summary(rated):
    """Words for the editor's status after a rating."""
    n = len(rated)
    left = sum(1 for r in rated.values() if r["score"] == 0)
    dups = sum(1 for r in rated.values() if r["dup_of"])
    top = sum(1 for r in rated.values() if r["top"])
    return ("Rated %d photos: the best %d for a LoRA are marked ★; %d duplicate%s; "
            "%d left out (not found). Hover a photo for its details."
            % (n, top, dups, "" if dups == 1 else "s", left))


def detail(r):
    """One photo's rating in words, for the status line on hover."""
    if r["score"] == 0:
        return "0 · " + "; ".join(r["flags"])
    parts = " · ".join("%s %.2f" % kv for kv in r["parts"].items())
    return "%d%s · %s · %s%s" % (r["score"], " ★" if r["top"] else "", r["angle"], parts,
                                 (" · " + "; ".join(r["flags"])) if r["flags"] else "")


if __name__ == "__main__":
    print(problem() or "ready")
