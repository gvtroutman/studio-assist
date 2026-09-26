"""Render a small WithAnyone trial through the same job path as Image Studio.

References stay local to the chosen ComfyUI backend. Outputs and history go
under --output; the user's library and saved scenes are not changed.
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import studio_imagegen as ig


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", action="append", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8188")
    ap.add_argument("--seed", type=int, default=240926)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--steps", type=int, default=25)
    args = ap.parse_args()
    studio = ig.Studio(root=args.output)
    backend = studio.backends()[0]
    backend["url"] = args.url
    if any(studio.client(backend).get_queue().get(k) for k in ("queue_running", "queue_pending")):
        raise RuntimeError("The backend is busy; run this trial after its queue finishes.")
    studio.lib.save("identities", [{"id": "person%d" % i, "name": "Person %d" % i,
                                    "references": [str(Path(path).resolve())]}
                                   for i, path in enumerate(args.ref, 1)])
    settings = {"model": "withanyone", "scene": args.prompt,
                "identities": [i["id"] for i in studio.lib.all("identities")],
                "width": args.width, "height": args.height, "seed": args.seed,
                "steps": args.steps, "anatomy": False, "auto_refine": False,
                "face_detail": False}
    job = ig.Job(settings, backend)
    previous = [None]

    def progress(j):
        state = (j.status, j.detail)
        if state != previous[0]:
            print(j.status, j.detail, flush=True)
            previous[0] = state

    try:
        studio.run_job(job, progress)
        print("Result:", job.status, job.detail, flush=True)
        for path in job.outputs:
            print(path, flush=True)
        if job.status != "complete":
            raise RuntimeError(job.detail)
    finally:
        studio.client(backend).free()
        studio.close()


if __name__ == "__main__":
    main()
