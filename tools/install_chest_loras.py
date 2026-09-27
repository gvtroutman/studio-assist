"""Install the verified CivitAI chest controls into the local ComfyUI library."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import apps.image_studio.addons.civitai as civ
import apps.image_studio.imagegen as ig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", required=True)
    args = parser.parse_args()
    lib = ig.Library()
    client = civ.Client(civ.load_token(lib.root))
    for version, control, strength in ((2520278, "chest_female", 2.0),
                                       (3155339, "chest_male", 0.8)):
        info, model = client.lookup(str(version))
        profile = civ.profile(info, model)
        print("Installing", profile["name"], flush=True)
        civ.download(client, profile["_download"], args.folder, profile["file"], profile["sha256"])
        rec, _ = lib.import_lora(profile)
        rec.update(body_control=control, strength=strength, always=False)
        lib.save("loras")
        print("Installed and assigned:", rec["file"], flush=True)


if __name__ == "__main__":
    try:
        main()
    except civ.CivitAIError as exc:
        sys.exit(str(exc))
