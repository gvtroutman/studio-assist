"""Compare one photo against a saved identity's full library with a fixed recipe.

Uses the local ComfyUI backend. Saves both histories in a new output directory;
never edits the real library. Refuses to start while that backend is busy.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import studio_imagegen as ig


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--identity", required=True)
    ap.add_argument("--library", default=ig.studio_dir())
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--seed", type=int, default=240927)
    ap.add_argument("--size", type=int, default=768)
    args = ap.parse_args()
    if args.output.exists():
        ap.error("Choose a new output directory.")
    profiles = json.loads((Path(args.library) / "identities.json").read_text(encoding="utf-8"))
    identity = next(p for p in profiles if p["id"] == args.identity)
    if len(identity.get("references") or []) < 2:
        ap.error("This comparison needs at least two saved references.")
    studio = ig.Studio(root=str(args.output))
    backend = studio.backends()[0]
    client = studio.client(backend)
    summary = []
    try:
        for label, photos in (("first-photo", identity["references"][:1]),
                              ("all-photos", identity["references"])):
            queue = client.get_queue()
            if queue.get("queue_running") or queue.get("queue_pending"):
                raise RuntimeError("ComfyUI is busy; comparison stopped.")
            profile = copy.deepcopy(identity)
            profile["references"] = photos
            studio.lib.save("identities", [profile])
            settings = dict(ig.default_settings(), model="withanyone", identities=[identity["id"]],
                            scene=args.prompt, seed=args.seed, seed_mode="fixed", steps=25,
                            width=args.size, height=args.size, anatomy=False, face_detail=False,
                            auto_refine=False, hand_pass=False,
                            experimental_reference_groups=(label == "all-photos"))
            job = ig.Job(settings, backend)
            previous = [None]

            def progress(j):
                state = (j.status, j.detail)
                if state != previous[0]:
                    print(label, j.status, j.detail, flush=True)
                    previous[0] = state

            studio.run_job(job, progress)
            summary.append({"variant": label, "references": photos, "status": job.status,
                            "detail": job.detail, "outputs": job.outputs})
            (args.output / "comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            if job.status != "complete":
                raise RuntimeError(job.detail)
            print(label, job.outputs, flush=True)
            client.free()
    finally:
        studio.close()


if __name__ == "__main__":
    main()
