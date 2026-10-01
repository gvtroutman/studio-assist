"""Build LoRA: an identity's reference photos -> a FLUX.2 Klein 9B LoRA of
their head, for the head swap (`headswap`, `Studio._head_swap`).

Klein is what redraws a person's head before the face swap, and the weak
stage: Klein 4B without a LoRA drew a look-alike (ArcFace 0.39 against
Partner's photos); with one trained on her face crops, 0.66, in about 6
minutes on the 5090 (2026-09-30; a FLUX.1 build took 90 and reached no
picture made with Z-Image). The head swap moved to Klein 9B on 2026-10-01,
and its LoRAs with it: a 4B LoRA does not load on the 9B. Same bench: 9B
with no LoRA 0.555, with her 9B LoRA 0.723 (12 of 12 at 0.5 or more, the
worst 0.56 against the 4B LoRA's 0.35); 9.0 minutes on the 5090 at about
21 GB.

The training itself is ai-toolkit's, in its own venv on this PC
(`D:\\ai-toolkit`), on Klein 9B's undistilled base (the LoRA then loads on
the distilled Klein ComfyUI runs), with Qwen3-8B as the text encoder and
ComfyUI's FLUX.2 VAE converted to ai-toolkit's layout -
`tools/prepare_klein_lora.py` sets those up once. The training itself
downloads nothing.

This module is stdlib only: it plans the run (`plan`, `config`), checks that
the toolkit is there (`problem`), finds the person's own face in every photo
(`faces.find`, ArcFace in ComfyUI's venv - the biggest face in a group shot
is often someone else's), and runs `tools/train_identity_lora.py` in the
toolkit's venv as a contained child (`run`), reading the `LEFT`/`KEPT`/
`STEP`/`DONE`/`ERROR` lines that script prints. That script crops the photos
to the head (PIL, which the app does not have) and starts ai-toolkit.

The finished file is copied into the 5090's LoRA folder; the caller adds it
to the library and makes it the person's `head_lora`.
"""

if __package__ in (None, ""):  # run as a script: import from the checkout
    import os as _os, sys as _sys
    _sys.path[0] = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))

import json
import os
import re
import subprocess
import sys
import time

import core.procs as studio_procs
from apps.image_studio import faces as face_finder

MIN_PHOTOS = 20
# Checkpoints of Partner's build scored the same from 500 steps to 1500 (head
# swap ArcFace 0.645 / 0.658 / 0.634 / 0.621 / 0.647 at 500-1500 by 250);
# 750 had the best mean and the best worst picture.
STEPS = 750
SAVE_EVERY = 250
RESOLUTION = 512
FAMILY = "flux2-klein9b"        # the library's family for Klein 9B (`imagegen.FAMILIES`)
ARCH = "flux2_klein_9b"         # ai-toolkit's name for it
TOOLKIT = os.environ.get("STUDIO_AI_TOOLKIT", r"D:\ai-toolkit")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPT = os.path.join(ROOT, "tools", "train_identity_lora.py")


def python_exe(toolkit=TOOLKIT):
    return os.path.join(toolkit, "venv", "Scripts" if os.name == "nt" else "bin",
                        "python.exe" if os.name == "nt" else "python")


def base_model(toolkit=TOOLKIT):
    """The folder of Klein's undistilled base (ai-toolkit's `name_or_path`)."""
    return os.path.join(toolkit, "models", "flux2-klein-base-9b")


def text_encoder(toolkit=TOOLKIT):
    """Qwen3-8B in Hugging Face form (Qwen/Qwen3-8B)."""
    return os.path.join(toolkit, "models", "qwen3-8b-te")


def vae(toolkit=TOOLKIT):
    """ComfyUI's FLUX.2 VAE in the layout ai-toolkit loads."""
    return os.path.join(toolkit, "models", "flux2-vae-bfl", "ae.safetensors")


def parts(toolkit=TOOLKIT):
    """Each file a build reads from the toolkit's models folder."""
    te = text_encoder(toolkit)
    return [os.path.join(base_model(toolkit), "flux-2-klein-base-9b.safetensors"),
            os.path.join(te, "config.json"), os.path.join(te, "tokenizer.json"),
            os.path.join(te, "model.safetensors.index.json"), vae(toolkit)]


def problem(toolkit=TOOLKIT):
    """What stops a build on this PC, in words; None when nothing does."""
    if not os.path.isfile(python_exe(toolkit)):
        return "ai-toolkit is not installed at %s (its venv is missing)." % toolkit
    if not os.path.isfile(os.path.join(toolkit, "run.py")):
        return "ai-toolkit at %s has no run.py." % toolkit
    missing = [p for p in parts(toolkit) if not os.path.isfile(p)]
    if missing:
        return ("ai-toolkit at %s lacks Klein's training files (%s). Run "
                "tools/prepare_klein_lora.py in the ai-toolkit venv first."
                % (toolkit, ", ".join(os.path.relpath(p, toolkit) for p in missing)))
    return None


def trigger_for(name):
    """A rare word for the person: the first letters of their name + "person",
    as Partner's `parperson`."""
    letters = re.sub(r"[^a-z]", "", (name or "").lower())[:3]
    return (letters or "idn") + "person"


def usable(paths):
    """The photos that still exist, in order, each once."""
    return [p for p in dict.fromkeys(paths or ()) if p and os.path.isfile(p)]


def plan(identity, lora_dir, toolkit=TOOLKIT, when=None, face_cache=None):
    """Everything one build needs, decided up front -> a spec dict. Raises
    ValueError, in words, when the person cannot be trained yet. With
    `face_cache` (the library's), the build finds the person's own face in
    each photo first (`Build.run`)."""
    photos = usable(identity.get("references"))
    if len(photos) < MIN_PHOTOS:
        raise ValueError("%s has %d usable photos; a LoRA needs at least %d."
                         % (identity.get("name") or "This person", len(photos), MIN_PHOTOS))
    if not lora_dir or not os.path.isdir(lora_dir):
        raise ValueError("No LoRA folder on this PC to put the LoRA in.")
    stamp = time.strftime("%Y%m%d-%H%M", time.localtime(when))
    rid = re.sub(r"[^a-z0-9]+", "-", (identity.get("id") or "person").lower()).strip("-")
    name = "%s_head_klein_%s" % (rid.replace("-", "_") or "person", stamp.replace("-", "_"))
    work = os.path.join(toolkit, "output", "%s-klein-%s" % (rid or "person", stamp))
    return {
        "identity": identity.get("id"),
        "person": identity.get("name") or rid,
        "name": name,
        "trigger": identity.get("trigger") or trigger_for(identity.get("name")),
        "photos": photos,
        "min_photos": MIN_PHOTOS,   # the script checks it again on the photos it can read
        "steps": STEPS,
        "work": work,
        "toolkit": toolkit,
        "base": base_model(toolkit),
        "text_encoder": text_encoder(toolkit),
        "vae": vae(toolkit),
        "resolution": RESOLUTION,
        "lora_out": os.path.join(lora_dir, name + ".safetensors"),
        "face_cache": face_cache,
    }


def caption(spec):
    return "a photo of %s" % spec["trigger"]


def config(spec):
    """ai-toolkit's job for `spec`: Klein 9B in bf16 (18 GB), its Qwen3-8B
    quantized to 8 bit (16 GB in bf16 beside it would not fit the 5090's
    32 while the captions are encoded; it is unloaded after), rank 16, lr
    1e-4, face crops at 512 only, no sampling."""
    return {"job": "extension", "config": {"name": spec["name"], "process": [{
        "type": "sd_trainer",
        "training_folder": os.path.join(spec["work"], "output"),
        "device": "cuda:0",
        "trigger_word": spec["trigger"],
        "network": {"type": "lora", "linear": 16, "linear_alpha": 16},
        "save": {"dtype": "float16", "save_every": SAVE_EVERY,
                 "max_step_saves_to_keep": 2, "push_to_hub": False},
        "datasets": [{"folder_path": os.path.join(spec["work"], "dataset"),
                      "caption_ext": "txt", "caption_dropout_rate": 0.05,
                      "shuffle_tokens": False, "cache_latents_to_disk": True,
                      "resolution": [spec["resolution"]],
                      # Windows: worker processes re-import the trainer and fail.
                      "num_workers": 0, "cache_latents_num_workers": 0}],
        "train": {"batch_size": 1, "steps": spec["steps"], "gradient_accumulation_steps": 1,
                  "train_unet": True, "train_text_encoder": False,
                  "gradient_checkpointing": True, "noise_scheduler": "flowmatch",
                  "optimizer": "adamw8bit", "lr": 1e-4, "dtype": "bf16",
                  "skip_first_sample": True, "disable_sampling": True,
                  "cache_text_embeddings": True, "unload_text_encoder": True},
        "model": {"arch": ARCH, "name_or_path": spec["base"], "vae_path": spec["vae"],
                  "quantize": False, "quantize_te": True, "low_vram": False},
    }]}, "meta": {"name": spec["name"], "version": "1.0"}}


def write_spec(spec):
    """The work folder, its spec (for the script) and ai-toolkit's job file.
    -> the spec's path."""
    os.makedirs(spec["work"], exist_ok=True)
    with open(os.path.join(spec["work"], "train.json"), "w", encoding="utf-8") as f:
        json.dump(config(spec), f, indent=2)
    path = os.path.join(spec["work"], "spec.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict(spec, caption=caption(spec)), f, indent=2)
    return path


def parse(line):
    """One line from the script -> ("step", (n, total)) | ("kept", (n, total)) |
    ("faces", (n, total)) | ("left", (n, total)) | ("done", path) |
    ("error", text) | ("note", text)."""
    line = line.strip()
    m = re.match(r"^(STEP|KEPT|FACES|LEFT) (\d+) (\d+)$", line)
    if m:
        return m.group(1).lower(), (int(m.group(2)), int(m.group(3)))
    if line.startswith("DONE "):
        return "done", line[5:].strip()
    if line.startswith("ERROR "):
        return "error", line[6:].strip()
    return "note", line


ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
BAR = re.compile(r"^(.*?)\s*\d+%\|")    # tqdm's "Loading weights:  42%|####"


def log_tail(path, lines=12, size=16384):
    """The last `lines` lines of ai-toolkit's train.log, as the Build LoRA
    window shows them: a tqdm bar redrawn in place (`\\r`) is one line, its
    latest, and a run of bars with the same label is its last. [] while the
    log is not there yet."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - size))
            text = f.read().decode("utf-8", "replace")
    except OSError:
        return []
    out = []
    for line in re.split(r"[\r\n]+", ANSI.sub("", text)):
        line = line.rstrip()
        if not line.strip():
            continue
        bar = BAR.match(line)
        if bar and out and BAR.match(out[-1]) and BAR.match(out[-1]).group(1) == bar.group(1):
            out[-1] = line
        else:
            out.append(line)
    return out[-lines:]


def eta(started, step, total, now=None):
    """Words for the time left, from the pace so far; "" before it is known."""
    if step < 5 or step >= total:
        return ""
    left = (time.time() if now is None else now) - started
    minutes = round(left / step * (total - step) / 60)
    if minutes >= 90:
        return "about %.1f h left" % (minutes / 60)
    return "about %d min left" % max(1, minutes)


class Build:
    """One training run as a contained child. `run()` blocks (a worker
    thread's), calling `on(kind, value)` with parse()'s kinds - and, while
    the person's face is found first, ("finding", (n, total)), or
    ("finder", words) when it could not be - and returns the LoRA's path or
    raises RuntimeError in words. `stop()` from any thread ends the whole
    tree - ai-toolkit's run.py included."""

    def __init__(self, spec):
        self.spec = spec
        self.child = None
        self.finder = None
        self.stopped = False
        self.step = (0, spec["steps"])
        self.started = None

    def find_faces(self, on):
        """The person's own face in each photo into the spec (`faces`), by
        ArcFace; when that cannot run, the script falls back on the biggest
        face and `on("finder", why)` says so."""
        why = face_finder.problem()
        if why is None:
            self.finder = face_finder.Job("find", [], self.spec["photos"],
                                          cache=self.spec["face_cache"])
            try:
                found = self.finder.run(lambda kind, value: on("finding", value))
            except (RuntimeError, OSError) as e:
                if self.stopped:
                    raise RuntimeError("Stopped.")
                why = str(e)
            else:
                self.spec["faces"] = {p: (found.get(p) or {}).get("box")
                                      for p in self.spec["photos"]}
                return
        on("finder", why)

    def run(self, on=lambda kind, value: None):
        if self.spec.get("face_cache") and not self.stopped:
            self.find_faces(on)
        if self.stopped:
            raise RuntimeError("Stopped.")
        spec_path = write_spec(self.spec)
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        self.child = studio_procs.spawn(
            [python_exe(self.spec["toolkit"]), SCRIPT, spec_path],
            cwd=self.spec["toolkit"], stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            env=env, creationflags=studio_procs.NO_WINDOW)
        if self.stopped:            # stop() came while starting
            self.child.kill()
        done, error, tail = None, None, []
        try:
            for raw in iter(self.child.proc.stdout.readline, b""):
                kind, value = parse(raw.decode("utf-8", "replace"))
                if kind == "step":
                    if self.started is None or value[0] <= 1:
                        self.started = time.time()
                    self.step = value
                elif kind == "kept":
                    self.spec["kept"] = value[0]    # what the LoRA was built from
                elif kind == "faces":
                    self.spec["faces"] = value[0]   # of those, cut to the head
                elif kind == "done":
                    done = value
                elif kind == "error":
                    error = value
                elif value:
                    tail = (tail + [value])[-5:]
                    continue
                on(kind, value)
            self.child.proc.wait()
        finally:
            # Also when `on` raised: nothing reads the child after this, and
            # an unwatched ai-toolkit kept 25 GB and the GPU while the button
            # offered a second build beside it (2026-10-01).
            self.child.kill()
        if self.stopped:
            raise RuntimeError("Stopped.")
        if done and os.path.isfile(done):
            return done
        raise RuntimeError(error or ("Training ended without a LoRA. Log: %s. %s" % (
            os.path.join(self.spec["work"], "train.log"), " | ".join(tail[-2:]))))

    def stop(self):
        self.stopped = True
        if self.finder is not None:
            self.finder.stop()
        if self.child is not None:
            self.child.kill()


def lora_record(spec):
    """The library's record for the finished file (`Library.import_lora`).
    Its trigger is what the head swap says for the person
    (`headswap.head_graph`)."""
    return {"file": os.path.basename(spec["lora_out"]),
            "name": "%s head (Klein)" % spec["person"],
            "category": "Identity", "family": FAMILY, "trigger": spec["trigger"],
            "strength": 1.0,
            "notes": "Klein 9B head-swap LoRA. Built from %d photos (%s cut to the head), "
                     "%d steps, ai-toolkit (%s)."
                     % (spec.get("kept", len(spec["photos"])), spec.get("faces", "?"),
                        spec["steps"], spec["work"])}


if __name__ == "__main__":
    print(problem() or "ready", file=sys.stdout)
