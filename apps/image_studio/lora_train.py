"""Build LoRA: an identity's reference photos -> a FLUX identity LoRA.

The training itself is ai-toolkit's, in its own venv on this PC
(`D:\\ai-toolkit`), on the FLUX.1-dev weights ComfyUI already has, converted
once to diffusers form by `tools/prepare_identity_lora.py` into
`models/flux-local-studio`. Nothing is downloaded.

This module is stdlib only: it plans the run (`plan`, `config`), checks that
the toolkit is there (`problem`), and runs `tools/train_identity_lora.py` in
the toolkit's venv as a contained child (`run`), reading the `STEP`/`DONE`/
`ERROR` lines that script prints. That script prepares the photos (PIL, which
the app does not have) and starts ai-toolkit's `run.py`.

The finished file is copied into the 5090's LoRA folder; the caller adds it
to the library and to the person.
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

MIN_PHOTOS = 20
STEPS = 2000
SAVE_EVERY = 250
TOOLKIT = os.environ.get("STUDIO_AI_TOOLKIT", r"D:\ai-toolkit")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPT = os.path.join(ROOT, "tools", "train_identity_lora.py")
BASE_PARTS = ("transformer", "text_encoder", "text_encoder_2", "vae")


def python_exe(toolkit=TOOLKIT):
    return os.path.join(toolkit, "venv", "Scripts" if os.name == "nt" else "bin",
                        "python.exe" if os.name == "nt" else "python")


def base_model(toolkit=TOOLKIT):
    return os.path.join(toolkit, "models", "flux-local-studio")


def problem(toolkit=TOOLKIT):
    """What stops a build on this PC, in words; None when nothing does."""
    if not os.path.isfile(python_exe(toolkit)):
        return "ai-toolkit is not installed at %s (its venv is missing)." % toolkit
    if not os.path.isfile(os.path.join(toolkit, "run.py")):
        return "ai-toolkit at %s has no run.py." % toolkit
    base = base_model(toolkit)
    missing = [p for p in BASE_PARTS if not os.path.isfile(os.path.join(base, p, ".complete"))]
    if missing:
        return ("The FLUX training copy at %s is incomplete (%s). Run "
                "tools/prepare_identity_lora.py in the ai-toolkit venv first."
                % (base, ", ".join(missing)))
    return None


def trigger_for(name):
    """A rare word for the person: the first letters of their name + "person",
    as Partner's `parperson`."""
    letters = re.sub(r"[^a-z]", "", (name or "").lower())[:3]
    return (letters or "idn") + "person"


def usable(paths):
    """The photos that still exist, in order, each once."""
    return [p for p in dict.fromkeys(paths or ()) if p and os.path.isfile(p)]


def plan(identity, lora_dir, toolkit=TOOLKIT, when=None):
    """Everything one build needs, decided up front -> a spec dict. Raises
    ValueError, in words, when the person cannot be trained yet."""
    photos = usable(identity.get("references"))
    if len(photos) < MIN_PHOTOS:
        raise ValueError("%s has %d usable photos; a LoRA needs at least %d."
                         % (identity.get("name") or "This person", len(photos), MIN_PHOTOS))
    if not lora_dir or not os.path.isdir(lora_dir):
        raise ValueError("No LoRA folder on this PC to put the LoRA in.")
    stamp = time.strftime("%Y%m%d-%H%M", time.localtime(when))
    rid = re.sub(r"[^a-z0-9]+", "-", (identity.get("id") or "person").lower()).strip("-")
    name = "%s_identity_%s" % (rid.replace("-", "_") or "person", stamp.replace("-", "_"))
    work = os.path.join(toolkit, "output", "%s-lora-%s" % (rid or "person", stamp))
    return {
        "identity": identity.get("id"),
        "person": identity.get("name") or rid,
        "name": name,
        "trigger": identity.get("trigger") or trigger_for(identity.get("name")),
        "photos": photos,
        "steps": STEPS,
        "work": work,
        "toolkit": toolkit,
        "base": base_model(toolkit),
        "lora_out": os.path.join(lora_dir, name + ".safetensors"),
    }


def caption(spec):
    return "a photo of %s" % spec["trigger"]


def config(spec):
    """ai-toolkit's job for `spec`: its own FLUX example's settings (rank 16,
    lr 1e-4, 512/768/1024 buckets), on the local FLUX copy, no sampling."""
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
                      "resolution": [512, 768, 1024],
                      # Windows: worker processes re-import the trainer and fail.
                      "num_workers": 0, "cache_latents_num_workers": 0}],
        "train": {"batch_size": 1, "steps": spec["steps"], "gradient_accumulation_steps": 1,
                  "train_unet": True, "train_text_encoder": False,
                  "gradient_checkpointing": True, "noise_scheduler": "flowmatch",
                  "optimizer": "adamw8bit", "lr": 1e-4, "dtype": "bf16",
                  "skip_first_sample": True, "disable_sampling": True,
                  "cache_text_embeddings": True, "unload_text_encoder": True},
        "model": {"name_or_path": spec["base"], "is_flux": True, "quantize": True,
                  "quantize_te": True, "low_vram": False},
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
    """One line from the script -> ("step", (n, total)) | ("done", path) |
    ("error", text) | ("note", text)."""
    line = line.strip()
    m = re.match(r"^STEP (\d+) (\d+)$", line)
    if m:
        return "step", (int(m.group(1)), int(m.group(2)))
    if line.startswith("DONE "):
        return "done", line[5:].strip()
    if line.startswith("ERROR "):
        return "error", line[6:].strip()
    return "note", line


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
    thread's), calling `on(kind, value)` with parse()'s kinds, and returns the
    LoRA's path or raises RuntimeError in words. `stop()` from any thread ends
    the whole tree - ai-toolkit's run.py included."""

    def __init__(self, spec):
        self.spec = spec
        self.child = None
        self.stopped = False
        self.step = (0, spec["steps"])
        self.started = None

    def run(self, on=lambda kind, value: None):
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
        for raw in iter(self.child.proc.stdout.readline, b""):
            kind, value = parse(raw.decode("utf-8", "replace"))
            if kind == "step":
                if self.started is None or value[0] <= 1:
                    self.started = time.time()
                self.step = value
            elif kind == "done":
                done = value
            elif kind == "error":
                error = value
            elif value:
                tail = (tail + [value])[-5:]
                continue
            on(kind, value)
        self.child.proc.wait()
        self.child.kill()
        if self.stopped:
            raise RuntimeError("Stopped.")
        if done and os.path.isfile(done):
            return done
        raise RuntimeError(error or ("Training ended without a LoRA. Log: %s. %s" % (
            os.path.join(self.spec["work"], "train.log"), " | ".join(tail[-2:]))))

    def stop(self):
        self.stopped = True
        if self.child is not None:
            self.child.kill()


def lora_record(spec, family="flux1"):
    """The library's record for the finished file (`Library.import_lora`)."""
    return {"file": os.path.basename(spec["lora_out"]),
            "name": "%s identity" % spec["person"],
            "category": "Identity", "family": family, "trigger": spec["trigger"],
            "strength": 1.0,
            "notes": "Built from %d photos, %d steps, ai-toolkit (%s)." % (
                len(spec["photos"]), spec["steps"], spec["work"])}


if __name__ == "__main__":
    print(problem() or "ready", file=sys.stdout)
