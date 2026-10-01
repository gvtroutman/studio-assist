"""Set up ai-toolkit once for Build LoRA's FLUX.2 Klein 9B head LoRAs.

Run in the ai-toolkit venv:
    D:\\ai-toolkit\\venv\\Scripts\\python.exe tools\\prepare_klein_lora.py

What `lora_train.parts` needs, each step skipped when its file is there:
1. Klein 9B's undistilled base, flux-2-klein-base-9b.safetensors (18.2 GB),
   from black-forest-labs/FLUX.2-klein-base-9B. The repo is gated: its
   licence (FLUX Non-Commercial) is accepted on Hugging Face by the account
   whose token `hf auth login` saved, or the download is refused.
2. Qwen3-8B as a Hugging Face folder (16.4 GB, Apache 2.0), from
   Qwen/Qwen3-8B. ComfyUI's own copy is fp8, which ai-toolkit cannot train
   with, so unlike the 4B's it is not linked in.
3. ComfyUI's FLUX.2 VAE (diffusers layout) converted to the layout
   ai-toolkit's flux2 loader reads, by ai-toolkit's own converter. The 4B
   and the 9B share it.
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

BASE_REPO = "black-forest-labs/FLUX.2-klein-base-9B"
BASE_FILE = "flux-2-klein-base-9b.safetensors"
QWEN_REPO = "Qwen/Qwen3-8B"
QWEN_FILES = ["*.json", "*.safetensors", "merges.txt", "vocab.json"]


def fetch(repo, name, dest):
    """One file of a Hugging Face repo to `dest`, through a staging folder
    beside it, so a broken download is never taken for the file."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError
    stage = dest.parent / "_download"
    print("Downloading", repo, name, flush=True)
    try:
        path = hf_hub_download(repo, name, local_dir=stage)
    except GatedRepoError:
        sys.exit("%s is gated: accept its licence at https://huggingface.co/%s while "
                 "signed in, then run this again." % (repo, repo))
    os.replace(path, dest)
    shutil.rmtree(stage, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--toolkit", type=Path, default=Path(r"D:\ai-toolkit"))
    ap.add_argument("--comfy-models", type=Path, default=Path(r"D:\ComfyUI-models"))
    args = ap.parse_args()
    models = args.toolkit / "models"

    te = models / "qwen3-8b-te"
    if not (te / "model.safetensors.index.json").is_file():
        from huggingface_hub import snapshot_download
        print("Downloading", QWEN_REPO, flush=True)
        snapshot_download(QWEN_REPO, allow_patterns=QWEN_FILES, local_dir=te)
        shutil.rmtree(te / ".cache", ignore_errors=True)
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

    base = models / "flux2-klein-base-9b" / BASE_FILE
    base.parent.mkdir(parents=True, exist_ok=True)
    if not base.is_file():
        fetch(BASE_REPO, BASE_FILE, base)
    print("Klein base:", base, flush=True)
    print("ready", flush=True)


if __name__ == "__main__":
    main()
