"""Install the pinned WithAnyone recipe into a ComfyUI backend.

Run with that backend's Python, not Studio Assist's Python. Reuses existing
FLUX, text encoders, VAE and antelopev2. Downloads only WithAnyone and SigLIP.
Does not restart ComfyUI or change its Python packages.
"""
import argparse
import importlib
import json
from pathlib import Path
import shutil
import subprocess

REVISION = "6bb610caabc65e7b2fd37eb7eeb6cc5392347f36"
SOURCE = "https://github.com/okdalto/ComfyUI-WithAnyone.git"
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--library", type=Path)
    args = parser.parse_args()
    for module in ("torch", "transformers", "insightface", "onnxruntime", "cv2",
                   "einops", "safetensors", "huggingface_hub"):
        importlib.import_module(module)
    from huggingface_hub import hf_hub_download, snapshot_download

    checkout = ROOT / ".work" / "ComfyUI-WithAnyone"
    if not checkout.exists():
        subprocess.run(["git", "clone", SOURCE, str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "checkout", REVISION], check=True)
    actual = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if actual != REVISION:
        raise RuntimeError("Unexpected WithAnyone source revision")
    target = args.comfy.resolve() / "custom_nodes" / "studio_withanyone"
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "comfy_nodes" / "studio_withanyone" / "__init__.py", target)
    upstream = checkout / "WithAnyone"
    vendor = target / "vendor"
    shutil.copytree(upstream / "withanyone", vendor / "withanyone", dirs_exist_ok=True)
    for name in ("util.py", "LICENSE"):
        shutil.copy2(upstream / name, vendor / name)
    (vendor / "__init__.py").touch()
    (vendor / "SOURCE.json").write_text(json.dumps({"url": SOURCE, "revision": REVISION}), encoding="utf-8")
    print("Installed node code:", target, flush=True)
    print("Downloading WithAnyone weights (existing files are reused).", flush=True)
    hf_hub_download("WithAnyone/WithAnyone", "withanyone.safetensors",
                    local_dir=str(args.models / "diffusion_models"))
    print("Downloading SigLIP.", flush=True)
    # This directory is intentionally under ComfyUI/models, where /models/diffusers
    # discovers it, including on installations using extra_model_paths.yaml.
    snapshot_download("google/siglip-base-patch16-256-i18n",
                      local_dir=str(args.comfy / "models" / "diffusers" / "siglip-base-patch16-256-i18n"),
                      allow_patterns=["*.json", "*.safetensors", "*.model"])
    if args.library:
        import sys
        sys.path.insert(0, str(ROOT))
        import studio_imagegen as ig
        library = ig.Library(str(args.library))
        if not library.get("models", "withanyone"):
            model = next(m for m in ig._default_models() if m["id"] == "withanyone")
            library.save("models", library.all("models") + [model])
        print("Added Family photo (WithAnyone) to the model library.", flush=True)
    print("Installation complete. Restart ComfyUI when its queue is empty.", flush=True)


if __name__ == "__main__":
    main()
