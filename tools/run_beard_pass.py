"""A live beard pass through Image Studio, in an isolated library.

A prepared scene's person becomes a character with a structured beard and
goes through Generate as the app runs it (the head swap and face swap from
their identity, then the finish passes), so the beard pass runs after the
identity pass with the drawn face's landmarks (StudioFaceLandmarks). Each
run keeps its settings, notes and final picture; each stage is in ComfyUI's
output/ImageStudio under the job's id (`_eyes` is the picture just before
the beard pass, `_beard` just after).

    python tools/run_beard_pass.py --character <id> --output "Scene Math Measurements/beard-live"
"""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from apps.image_studio import imagegen as ig
from apps.image_studio.scene import scene as sc

PREPARED = ROOT / "Scene Enhancements" / "prepared"
RUNS = [("oktoberfest - drinking from stein", "Drinker"),
        ("oktoberfest - wearing lederhosen and dirndl", "Man in lederhosen")]
BEARD = {"style": "short", "length": 0.2, "coverage": 0.8, "density": 0.75, "color": ""}


def save(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def isolated(output):
    source = ig.Library()
    library = output / "library"
    library.mkdir(exist_ok=True)
    for kind in ig.CLEAN:
        save(library / (kind + ".json"), copy.deepcopy(source.all(kind)))
    os.environ["STUDIO_SETTINGS"] = str(output / "settings.json")
    return ig.Studio(root=str(library))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--character", required=True,
                        help="the character id in the library to give the beard")
    parser.add_argument("--seed", type=int, default=261010)
    parser.add_argument("--denoise", type=float, help="BEARD_DENOISE for this run")
    parser.add_argument("--only", help="run only the scene whose name contains this")
    args = parser.parse_args()
    if args.denoise is not None:
        ig.BEARD_DENOISE = args.denoise
    runs = [r for r in RUNS if not args.only or args.only in r[0]]
    args.output.mkdir(parents=True, exist_ok=True)
    studio = isolated(args.output)
    backend = studio.lib.get("backends", "5090")
    try:
        client = studio.client(backend)
        if not client.health()["ok"]:
            raise RuntimeError("ComfyUI on the 5090 is not reachable")
        queue = client.get_queue()
        if queue.get("queue_running") or queue.get("queue_pending"):
            raise RuntimeError("The 5090 is busy; this run will not interrupt other jobs")
        studio.check(backend, full=True)
        if ig.LANDMARK_NODE not in studio.nodes[backend["id"]]:
            raise RuntimeError("ComfyUI lacks %s" % ig.LANDMARK_NODE)
        characters = {r["id"]: r for r in studio.lib.all("characters")}
        identities = {r["id"]: r for r in studio.lib.all("identities")}
        who = characters[args.character]
        report = {"character": args.character, "beard": BEARD, "seed": args.seed,
                  "beard_denoise": ig.BEARD_DENOISE, "runs": []}
        for name, person in runs:
            folder = args.output / name.replace(" ", "_")
            folder.mkdir(exist_ok=True)
            scene, problems = sc.load(str(PREPARED / (name + ".scene.json")))
            obj = next(o for o in scene["objects"] if o.get("name") == person)
            obj["character"] = args.character
            obj["look"] = dict(sc.character_look(who, obj.get("look") or {}), beard=dict(BEARD))
            save(folder / "input_scene.json", scene)
            maps, notes = sc.scene_maps(scene, {"pose", "composition"}, str(folder))
            words, extra = sc.generation(scene, maps, characters, identities)
            settings = dict(ig.default_settings(), **extra)
            settings.update(model="z-image-turbo", backend="5090", scene=words.text,
                            seed=args.seed, seed_mode="fixed", steps=8, batch=1,
                            auto_refine=False,
                            identities=[{"id": i, "strength": 0.85}
                                        for i in extra.get("scene_identities") or []])
            plan = studio.preview(settings, backend)
            if plan.errors:
                raise RuntimeError("Preflight: " + "; ".join(plan.errors))
            save(folder / "settings.json", settings)
            client.free()
            print("RENDER", name, flush=True)
            started = time.perf_counter()
            job = ig.Job(settings, backend)
            last = [None, 0]

            def progress(job):
                now = time.monotonic()
                if job.status != last[0] or now - last[1] > 20:
                    print("PROGRESS", job.status, job.detail, flush=True)
                    last[:] = [job.status, now]
            studio.run_job(job, progress)
            for path in job.outputs:
                shutil.copy(path, folder / ("final" + Path(path).suffix))
            row = {"job": job.id, "scene": name, "person": person, "status": job.status, "detail": job.detail,
                   "seconds": round(time.perf_counter() - started, 1), "outputs": job.outputs,
                   "notes": (job.record or {}).get("notes"),
                   "passes": (job.record or {}).get("passes"),
                   "scene_warnings": problems, "map_notes": notes}
            report["runs"].append(row)
            save(args.output / "report.json", report)
            print("DONE", name, job.status, row["seconds"], flush=True)
            for n in row["notes"] or []:
                print("  NOTE", n, flush=True)
    finally:
        studio.close()


if __name__ == "__main__":
    main()
