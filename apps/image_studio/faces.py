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
            error, tail = None, []
            for raw in iter(self.child.proc.stdout.readline, b""):
                kind, value = parse(raw.decode("utf-8", "replace"))
                if kind == "face":
                    on(kind, value)
                elif kind == "error":
                    error = value
                elif kind == "note" and value:
                    tail = (tail + [value])[-3:]
            self.child.proc.wait()
            self.child.kill()
            if self.stopped:
                raise RuntimeError("Stopped.")
            if error or not os.path.isfile(self.job["result"]):
                raise RuntimeError(error or "the face finder stopped: %s" % " | ".join(tail))
            with open(self.job["result"], encoding="utf-8") as f:
                return json.load(f)
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


if __name__ == "__main__":
    print(problem() or "ready")
