"""A controlled A/B test for regional character prompting (AGENTS.md "Regional
character prompting"): the same saved scene, the same seed, model, LoRAs, pose
and depth maps, run twice - once as an ordinary one-paragraph prompt, once with
each named character given their own words and masked region of the picture
(scene.py's character_masks/scene_text_regional, imagegen.py's fill()
regional_conditioning block). Only the toggle differs, so any difference in the
two pictures is the technique, not the setup.

Not a unit test - a one-time comparison to run by hand against a live backend:

    python tools/ab_regional_prompt.py path\\to\\scene.json

Needs a scene with at least two people, each with a character assigned (the
Scene Builder's Character dropdown), and a live ComfyUI on the chosen backend.
Prints where each picture landed; look at both for the checklist regional
prompting is meant to fix: hair/clothing/identity bleed, wrong prop, the wrong
person doing something, background corruption, face quality, pose kept.
"""
import argparse
import json
import os
import sys
import time

if __package__ in (None, ""):  # run as a script: import from the checkout
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import apps.image_studio.imagegen as ig
import apps.image_studio.scene.scene as sc

TERMINAL = {"done", "failed", "cancelled"}


def run_variant(studio, scene, backend, model, seed, regional):
    """One arm of the test: the scene with `regional_prompting` forced to
    `regional`, submitted as one job with a fixed seed. -> the finished Job."""
    scene = json.loads(json.dumps(scene))       # a private copy per arm
    scene["regional_prompting"] = regional
    wf = studio.workflow_loader(model["workflow"])
    takes = {k for k in sc.MAP_KINDS if (wf.get("references") or {}).get(k)}
    maps, notes = sc.scene_maps(scene, takes)
    for n in notes:
        print("  note:", n)
    chars = {c["id"]: c for c in studio.lib.all("characters")}
    idents = {d["id"]: d for d in studio.lib.all("identities")}
    words, extra = sc.generation(scene, maps, chars, idents)
    settings = dict(ig.default_settings(), **extra)
    settings.update(model=model["id"], scene=words.text, seed=seed, seed_mode="fixed",
                    backend=backend["id"], batch=1)
    if regional and "character_regions" not in extra:
        print("  warning: fewer than two named characters - this arm is not regional.")
    job = studio.submit(settings)[0]
    while job.status not in TERMINAL:
        time.sleep(1)
    return job


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", help="a saved scene .json (Scene Builder's Save)")
    ap.add_argument("--backend", default="5090")
    ap.add_argument("--model", default="z-image-turbo")
    ap.add_argument("--seed", type=int, default=None, help="fixed for both arms; random if omitted")
    args = ap.parse_args()

    with open(args.scene, "r", encoding="utf-8") as f:
        scene, problems = sc.clean_scene(json.load(f))
    if problems:
        print("Scene file problems:", "; ".join(problems))
    if len(sc.people(scene)) < 2:
        print("This scene has fewer than two people; regional prompting has nothing to "
              "isolate. Add a second person and assign both a character first.")
        return 1

    studio = ig.Studio()
    backend = studio.backend(args.backend)
    model = studio.lib.get("models", args.model)
    if backend is None or model is None:
        print("Unknown backend or model: %r, %r" % (args.backend, args.model))
        return 1
    seed = args.seed if args.seed is not None else __import__("random").randint(0, ig.MAX_SEED)

    for regional, label in ((False, "current (one paragraph)"), (True, "regional (masked)")):
        print("Running the %s arm, seed %d..." % (label, seed))
        job = run_variant(studio, scene, backend, model, seed, regional)
        if job.status != "done":
            print("  %s: %s" % (job.status, job.detail))
            continue
        for path in job.outputs:
            print("  ->", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
