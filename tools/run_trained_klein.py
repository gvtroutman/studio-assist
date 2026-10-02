"""Repeat the frozen Klein geometry pilot with its identity's trained LoRA."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_scene_likeness import save, ensure_backend
from apps.image_studio import imagegen as ig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--identity", required=True,
                        help="the id of the identity whose trained head LoRA is added")
    args = parser.parse_args()
    previous = ROOT / "Scene Math Measurements/klein-likeness-measured"
    source_report = json.loads((previous / "report.json").read_text(encoding="utf-8"))
    if args.output.exists():
        raise RuntimeError("Choose a fresh output folder")
    args.output.mkdir(parents=True)
    library = args.output / "library"
    library.mkdir()
    for path in (previous / "library").glob("*.json"):
        (library / path.name).write_bytes(path.read_bytes())
    os.environ["STUDIO_SETTINGS"] = str(args.output / "settings.json")
    workflow = json.loads((previous / "klein_reference_workflow.json").read_text(encoding="utf-8"))
    save(args.output / "klein_reference_workflow.json", workflow)
    (args.output / "input_scene.json").write_bytes((previous / "input_scene.json").read_bytes())
    studio = ig.Studio(root=str(library), workflow_loader=lambda wid:
                      copy.deepcopy(workflow) if wid == "klein9b_base" else ig.load_workflow(wid))
    backend = studio.lib.get("backends", "5090")
    try:
        client = ensure_backend(studio, backend, start=True)
        studio.check(backend, full=True)
        identity = studio.lib.get("identities", args.identity)
        if identity is None:
            raise RuntimeError("No identity %r in the library" % args.identity)
        adapter, why = ig.head_lora(studio.lib, identity, backend["id"], studio.inventories[backend["id"]])
        if not adapter:
            raise RuntimeError(why or "No trained head LoRA assigned")
        report = {"method": "Frozen prior Klein pilot, same six settings and reference pixels. "
                  "Adds the identity's assigned trained 9B head LoRA at its saved strength and trigger. "
                  "No face replacement or detail. Three geometry pairs, alternating order.",
                  "parent_report": str(previous / "report.json"), "adapter": adapter,
                  "source_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in [ROOT / "apps/image_studio/imagegen.py", Path(__file__)]}, "runs": []}
        print("ADAPTER", json.dumps(adapter), flush=True)
        save(args.output / "report.json", report)
        for original in source_report["runs"]:
            arm, seed = original["arm"], original["seed"]
            folder = args.output / ("%s_%d" % (arm, seed))
            folder.mkdir()
            settings = json.loads((previous / folder.name / "settings.json").read_text(encoding="utf-8"))
            settings["loras"] = [{"id": identity["head_lora"], "strength": adapter["strength"]}]
            settings["replay_prompt"] = original["record"]["prompt"] + " A photograph of " + adapter["trigger"] + "."
            plan = studio.preview(settings, backend)
            if plan.errors or plan.loras != [(adapter["lora"], adapter["strength"])]:
                raise RuntimeError("LoRA preflight failed: " + str(plan.errors or plan.loras))
            if len(plan.images) != 3:
                raise RuntimeError("All three frozen reference inputs required")
            save(folder / "settings.json", settings)
            ensure_backend(studio, backend)
            client.free()
            print("RENDER", arm, seed, flush=True)
            job = ig.Job(settings, backend)
            started = time.perf_counter()
            last = [None, 0]
            def progress(job):
                now = time.monotonic()
                if job.status != last[0] or now - last[1] > 20:
                    print("PROGRESS", arm, seed, job.status, job.detail, flush=True)
                    last[:] = [job.status, now]
            studio.run_job(job, progress)
            row = {"arm": arm, "seed": seed, "status": job.status,
                   "seconds": time.perf_counter() - started, "detail": job.detail,
                   "outputs": job.outputs, "record": job.record, "maps": original["maps"]}
            report["runs"].append(row)
            save(args.output / "report.json", report)
            print("DONE", arm, seed, job.status, row["seconds"], flush=True)
            if job.status != "complete":
                raise RuntimeError(job.detail)
    finally:
        studio.close()


if __name__ == "__main__":
    main()
