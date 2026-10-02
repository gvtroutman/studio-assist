"""Live A/B benchmark of HEAD and the working image pipeline, on an idle backend.

Writes isolated library snapshots, timings, graphs and images beneath --output.
Does not modify the user's library or clear ComfyUI's history. Uses /free only
with an empty queue to prevent cached renders from masquerading as fresh ones.
"""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import apps.image_studio.imagegen as current


def save(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def baseline(folder):
    result = subprocess.run(["git", "show", "HEAD:apps/image_studio/imagegen.py"],
                            cwd=ROOT, capture_output=True, check=True)
    path = folder / "baseline_imagegen.py"
    path.write_bytes(result.stdout)
    spec = importlib.util.spec_from_file_location("pipeline_before", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def instrument(module, folder):
    class Client(module.ComfyUIClient):
        def __init__(self, backend):
            super().__init__(backend)
            self.requests, self.events, self.graphs = [], [], []

        def _open(self, req, timeout=None):
            url = req.full_url if isinstance(req, urllib.request.Request) else req
            start = time.perf_counter()
            try:
                return super()._open(req, timeout)
            finally:
                self.requests.append({"path": url.split(self.url)[-1],
                                      "seconds": time.perf_counter() - start})

        def queue_workflow(self, graph):
            self.graphs.append(copy.deepcopy(graph))
            return super().queue_workflow(graph)

        def listen_for_progress(self, pid, on_event, **kwargs):
            def event(kind, data):
                self.events.append({"at": time.perf_counter(), "kind": kind, "data": data})
                on_event(kind, data)
            return super().listen_for_progress(pid, event, **kwargs)
    return Client


def idle(client):
    queue = client.get_queue()
    if queue.get("queue_running") or queue.get("queue_pending"):
        raise RuntimeError("Backend is busy; benchmark will not clear or interrupt other jobs")


def reset(client):
    idle(client)
    client.free()  # on this server free_memory resets execution caches too
    time.sleep(2)
    idle(client)


CASES = [
    ("fox", "A photograph of a red fox standing in fresh snow at dawn, pine trees "
     "in the background, soft morning light, eye level, full animal visible.",
     {"width": 1024, "height": 1024, "face_detail": False, "hand_pass": False}),
    ("portrait", "A photograph of a woman in her fifties wearing a grey knit sweater, "
     "seated at a wooden cafe table, both hands resting on the table, looking at "
     "the camera with a relaxed smile, soft window light, waist-up framing.",
     {"width": 832, "height": 1216, "refine": True, "upscale": 1.5,
      "face_detail": True, "hand_pass": True}),
    ("group", "A photograph of three adult friends, two women and a man, seated "
     "beside each other at a cafe table, each face clearly visible, talking and "
     "smiling, mugs on the table, hands visible, natural afternoon window light.",
     {"width": 1024, "height": 768, "face_detail": True, "hand_pass": True}),
]


def render(module, label, source, backend, output, case, seed):
    name, prompt, overrides = case
    folder = output / ("%s_%s_%d" % (label, name, seed))
    folder.mkdir()
    library = folder / "library"
    library.mkdir()
    for kind in current.CLEAN:
        save(library / (kind + ".json"), source.all(kind))
    os.environ["STUDIO_SETTINGS"] = str(folder / "settings.json")
    studio = module.Studio(root=str(library), client_factory=instrument(module, folder))
    client = studio.client(backend)
    reset(client)
    studio.check(backend, full=True)
    settings = dict(module.default_settings(), scene=prompt, seed=seed, backend=backend["id"],
                    auto_refine=False, batch=1, **overrides)
    # Direct execution on this thread avoids a lane freeing models before the
    # next measurement has observed the result. This is the real GUI runner.
    job = module.Job(settings, backend)
    print("START %s %s seed %d" % (label, name, seed), flush=True)
    started = time.perf_counter()
    studio.run_job(job, lambda job: None)
    elapsed = time.perf_counter() - started
    report = {"variant": label, "case": name, "seed": seed, "status": job.status,
              "detail": job.detail, "elapsed_seconds": elapsed, "outputs": job.outputs,
              "record": job.record, "http": client.requests, "events": client.events,
              "graphs": client.graphs}
    save(folder / "measurement.json", report)
    print("END %s %s %.2fs %s %s" % (label, name, elapsed, job.status, job.detail), flush=True)
    studio.close()
    reset(client)
    return report


def uploads(module, label, backend, picture, subfolder, count=10):
    client = module.ComfyUIClient(backend)
    original = client._open
    calls = []
    def opened(req, timeout=None):
        if isinstance(req, urllib.request.Request) and req.full_url.endswith("/upload/image"):
            calls.append(req.full_url)
            if subfolder:
                boundary = req.get_header("Content-type").split("boundary=")[-1]
                part = ("--%s\r\nContent-Disposition: form-data; name=\"subfolder\"\r\n\r\n%s\r\n"
                        % (boundary, subfolder)).encode()
                req = urllib.request.Request(req.full_url, data=part + req.data,
                                             headers=dict(req.header_items()), method="POST")
        return original(req, timeout)
    client._open = opened
    start = time.perf_counter()
    names = [client.upload_image(picture) for _ in range(count)]
    return {"variant": label, "case": "subfolder" if subfolder else "ordinary",
            "reuses": count, "http_uploads": len(calls),
            "elapsed_seconds": time.perf_counter() - start, "names": names}


def cancel_in_flight(module, label, backend, graph):
    """Pause after the server accepts /prompt, before its id reaches the job."""
    client = module.ComfyUIClient(backend)
    observer = current.ComfyUIClient(backend)
    reset(observer)
    accepted, release, stopped = threading.Event(), threading.Event(), threading.Event()
    original = client.queue_workflow
    state = {}
    def submit(graph):
        state["pid"] = original(graph)
        accepted.set()
        if not release.wait(20):
            raise RuntimeError("benchmark submission gate timed out")
        return state["pid"]
    client.queue_workflow = submit
    def worker():
        watch = client.watch()
        try:
            pid = client.queue_workflow(graph)
            state["result"] = client.listen_for_progress(pid, lambda *args: None,
                                                        stop=stopped.is_set, watch=watch)
        except Exception as error:
            state["error"] = str(error)
        finally:
            watch.close()
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    try:
        if not accepted.wait(20):
            raise RuntimeError("server did not accept cancellation benchmark")
        stopped.set()  # queue cannot interrupt here: it has no prompt id yet
        start = time.perf_counter()
        release.set()
        thread.join(30)
        if thread.is_alive():
            raise RuntimeError("cancellation worker did not finish")
        remaining = observer.position(state["pid"])
        report = {"variant": label, "prompt_id": state["pid"],
                  "worker_seconds": time.perf_counter() - start,
                  "position_after_stop": remaining, "error": state.get("error")}
        print("CANCEL %s %s" % (label, report), flush=True)
        return report
    finally:
        release.set()
        if state.get("pid"):
            observer.cancel_job(state["pid"])  # cleanup only this benchmark's prompt
            end = time.monotonic() + 60
            while observer.position(state["pid"]) is not None and time.monotonic() < end:
                time.sleep(1)
        reset(observer)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", default="5090")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = current.Library()
    backend = source.get("backends", args.backend)
    if not backend or backend.get("shares_llm_gpu"):
        raise RuntimeError("Choose a dedicated backend; this benchmark does not unload LM Studio")
    before = baseline(output)
    client = current.ComfyUIClient(backend)
    idle(client)
    report = {"backend": backend, "health": client.health(), "runs": [], "uploads": [],
              "cancel": [], "method": "Matched seeds, alternating arm order, real Studio.run_job; "
              "cold models and execution caches via /free before each run; copied user library; "
              "no selected identities or critic. Readiness check excluded from elapsed time."}
    save(output / "results.json", report)
    for repeat in range(args.repeats):
        seed = 4242 + repeat * 101
        order = [(before, "before"), (current, "after")]
        if repeat % 2:
            order.reverse()
        for case in CASES:
            for module, label in order:
                result = render(module, label, source, backend, output, case, seed)
                report["runs"].append(result)
                save(output / "results.json", report)
                if result["status"] != "complete":
                    raise RuntimeError("Render failed; see measurement.json")
    fox = next(r for r in report["runs"] if r["case"] == "fox")
    for subfolder in ("", "pipeline-benchmark/" + uuid.uuid4().hex[:8]):
        for module, label in ((before, "before"), (current, "after")):
            report["uploads"].append(uploads(module, label, backend, fox["outputs"][0], subfolder))
            save(output / "results.json", report)
    graph = fox["graphs"][0]
    for node in graph.values():
        if node["class_type"] == "SaveImage":
            node["inputs"]["filename_prefix"] = "ImageStudio/benchmark_cancel_" + uuid.uuid4().hex[:8]
    for module, label in ((before, "before"), (current, "after")):
        report["cancel"].append(cancel_in_flight(module, label, backend, graph))
        save(output / "results.json", report)
    print("RESULTS " + str(output / "results.json"), flush=True)


if __name__ == "__main__":
    main()
