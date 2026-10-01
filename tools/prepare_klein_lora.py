"""Set up ai-toolkit once for Build LoRA's FLUX.2 Klein 4B head LoRAs.

Run in the ai-toolkit venv:
    D:\\ai-toolkit\\venv\\Scripts\\python.exe tools\\prepare_klein_lora.py

What `lora_train.parts` needs, each step skipped when its file is there:
1. Klein's undistilled base, flux-2-klein-base-4b.safetensors (7.75 GB,
   Apache 2.0, not gated), downloaded from black-forest-labs/FLUX.2-klein-base-4B.
2. Qwen3-4B as a Hugging Face folder: ComfyUI's own text encoder
   (qwen_3_4b.safetensors, the same tensors under the same names) hard-linked
   in, with Qwen's config and tokenizer files (about 12 MB) downloaded
   from Qwen/Qwen3-4B.
3. ComfyUI's FLUX.2 VAE (diffusers layout) converted to the layout
   ai-toolkit's flux2 loader reads, by ai-toolkit's own converter.
"""
import argparse
import os
import shutil
import sys
import urllib.request
from pathlib import Path

BASE_URL = ("https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B/resolve/main/"
            "flux-2-klein-base-4b.safetensors")
QWEN_URL = "https://huggingface.co/Qwen/Qwen3-4B/resolve/main/"
QWEN_FILES = ("config.json", "generation_config.json", "merges.txt", "tokenizer.json",
              "tokenizer_config.json", "vocab.json")


def fetch(url, dest):
    """`url` to `dest` through a .part file, so a broken download is never taken
    for the file."""
    part = dest.with_name(dest.name + ".part")
    print("Downloading", url, flush=True)
    with urllib.request.urlopen(url, timeout=60) as r, open(part, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    os.replace(part, dest)


def link_or_copy(src, dest):
    try:
        os.link(src, dest)            # same drive: no second copy of 8 GB
    except OSError:
        shutil.copyfile(src, dest)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--toolkit", type=Path, default=Path(r"D:\ai-toolkit"))
    ap.add_argument("--comfy-models", type=Path, default=Path(r"D:\ComfyUI-models"))
    args = ap.parse_args()
    models = args.toolkit / "models"

    base = models / "flux2-klein-base-4b" / "flux-2-klein-base-4b.safetensors"
    base.parent.mkdir(parents=True, exist_ok=True)
    if not base.is_file():
        fetch(BASE_URL, base)
    print("Klein base:", base, flush=True)

    te = models / "qwen3-4b-te"
    te.mkdir(parents=True, exist_ok=True)
    for name in QWEN_FILES:
        if not (te / name).is_file():
            fetch(QWEN_URL + name, te / name)
    weights = te / "model.safetensors"
    if not weights.is_file():
        src = args.comfy_models / "text_encoders" / "qwen_3_4b.safetensors"
        if not src.is_file():
            sys.exit("ComfyUI's Qwen3-4B text encoder is not at %s." % src)
        link_or_copy(src, weights)
    print("Text encoder:", te, flush=True)

    vae = models / "flux2-vae-bfl" / "ae.safetensors"
    if not vae.is_file():
        sys.path.insert(0, str(args.toolkit))
        from safetensors.torch import load_file, save_file
        from toolkit.models.v2.vae.flux2_kl import AutoEncoder, convert_diffusers_state_dict
        src = args.comfy_models / "vae" / "flux2-vae.safetensors"
        out = convert_diffusers_state_dict(load_file(str(src)))
        want = AutoEncoder(AutoEncoder.aitk_config_from_state_dict(out)).state_dict()
        bad = sorted(set(want) ^ set(out)) + [
            k for k in want if k in out and tuple(want[k].shape) != tuple(out[k].shape)]
        if bad:
            sys.exit("The converted VAE does not fit ai-toolkit's: %s" % ", ".join(bad[:5]))
        vae.parent.mkdir(parents=True, exist_ok=True)
        part = vae.with_name(vae.name + ".part")
        save_file({k: v.contiguous() for k, v in out.items()}, str(part))
        os.replace(part, vae)
    print("VAE:", vae, flush=True)
    print("ready", flush=True)


if __name__ == "__main__":
    main()
