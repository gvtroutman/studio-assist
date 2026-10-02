"""Matched-seed scene-math A/B through Image Studio, in an isolated library.

Both arms use the current meshes, model and pipeline. Only the landmark
deformation and ownership/feather algorithms differ. Final face replacement
is disabled because it would obscure the generated identity being measured.
"""
import argparse
from contextlib import ExitStack
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from apps.image_studio import imagegen as ig
from apps.image_studio.scene import scene as sc


def save(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def old_scene(output):
    path = output / "baseline_scene.py"
    path.write_bytes(subprocess.run(["git", "show", "HEAD:apps/image_studio/scene/scene.py"],
                                   cwd=ROOT, check=True, capture_output=True).stdout)
    spec = importlib.util.spec_from_file_location("scene_math_baseline_live", path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    old.render, old.frame_size = sc.render, sc.frame_size
    return old


def isolated(output):
    source = ig.Library()
    library = output / "library"
    library.mkdir(exist_ok=True)
    for kind in ig.CLEAN:
        records = copy.deepcopy(source.all(kind))
        if kind == "identities":
            for identity in records:
                identity["face_swap"] = False
        save(library / (kind + ".json"), records)
    os.environ["STUDIO_SETTINGS"] = str(output / "settings.json")
    return ig.Studio(root=str(library))


def ensure_backend(studio, backend, start=False):
    client = studio.client(backend)
    health = client.health()
    if not health["ok"] and start:
        print("STARTING through Image Studio:", backend["name"], flush=True)
        # The Image Studio start path remains responsible for launching it.
        # Hide its console so the benchmark does not take over the desktop.
        original = ig.subprocess.Popen
        def hidden(*args, **kwargs):
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            return original(*args, **kwargs)
        with patch.object(ig.subprocess, "Popen", side_effect=hidden):
            studio.start(backend)
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            time.sleep(2)
            health = client.health()
            if health["ok"]:
                break
        if not health["ok"]:
            raise RuntimeError("ComfyUI did not become reachable after Image Studio started it")
    if not health["ok"]:
        raise RuntimeError(health["detail"])
    queue = client.get_queue()
    if queue.get("queue_running") or queue.get("queue_pending"):
        raise RuntimeError("Backend is busy; benchmark will not interrupt other jobs")
    return client


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start-only", action="store_true")
    parser.add_argument("--klein", action="store_true", help="Klein Base native reference pilot")
    parser.add_argument("--seeds", nargs="+", type=int, default=[261002, 261003, 261004])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    studio = isolated(args.output)
    if args.klein:
        workflow = copy.deepcopy(ig.load_workflow("klein9b_base"))
        workflow["references"] = {"face": "face_image", "pose": "pose_image",
                                  "composition": "composition_image"}
        graph = workflow["graph"]
        positive, negative = ["10", 0], ["12", 0]
        for index, variable in enumerate(workflow["references"].values()):
            prefix = "ref%d" % index
            graph[prefix + "load"] = {"class_type": "LoadImage", "inputs": {
                "image": "{{%s}}" % variable}}
            graph[prefix + "scale"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
                "image": [prefix + "load", 0], "upscale_method": "lanczos", "megapixels": 0.25,
                "resolution_steps": 1}}
            graph[prefix + "encode"] = {"class_type": "VAEEncode", "inputs": {
                "pixels": [prefix + "scale", 0], "vae": ["3", 0]}}
            for side, conditioning in (("positive", positive), ("negative", negative)):
                graph[prefix + side] = {"class_type": "ReferenceLatent", "inputs": {
                    "conditioning": conditioning, "latent": [prefix + "encode", 0]}}
            positive, negative = [prefix + "positive", 0], [prefix + "negative", 0]
        graph["6"]["inputs"].update(positive=positive, negative=negative)
        save(args.output / "klein_reference_workflow.json", workflow)
        studio.workflow_loader = lambda wid: copy.deepcopy(workflow) if wid == "klein9b_base" else ig.load_workflow(wid)
    backend = studio.lib.get("backends", "5090")
    if not backend or backend.get("shares_llm_gpu"):
        raise RuntimeError("The dedicated 5090 Image Studio backend is required")
    try:
        client = ensure_backend(studio, backend, start=True)
        health = studio.check(backend, full=True)
        capabilities = {"health": health, "inventory": {k: sorted(v) for k, v in
            studio.inventories[backend["id"]].items()}, "nodes": sorted(studio.nodes[backend["id"]])}
        save(args.output / "capabilities.json", capabilities)
        print("READY", json.dumps({"health": health, "identity_nodes": sorted(
            set(capabilities["nodes"]) & {"ApplyPulidFlux", "StudioFacePaste", "StudioWithAnyone"})}), flush=True)
        if args.start_only:
            return
        if (args.output / "report.json").exists():
            raise RuntimeError("Choose a fresh output folder for a new trial")
        old = old_scene(args.output)
        scene_path = (ROOT / "Scene Math Measurements/likeness-pilot/input_scene.json" if args.klein
                      else ROOT / "Scene Enhancements/prepared/sound of music.scene.json")
        scene, problems = sc.load(str(scene_path))
        scene["real_faces"] = False
        # Retain the saved scene's actual head sliders and stage.
        characters = {r["id"]: r for r in studio.lib.all("characters")}
        identities = {r["id"]: r for r in studio.lib.all("identities")}
        save(args.output / "input_scene.json", scene)
        report = {"method": "Three paired seeds, alternating order, same current geometry and pipeline. "
                  "FLUX dev, scene pose/depth, PuLID and face detail; no FaceFusion or real-face paste. "
                  "Saved Sound of Music scene and head settings retained. Recognition is a proxy, not a human review.",
                  "scene_warnings": problems, "runs": [], "source_hashes": {str(p.relative_to(ROOT)):
                      hashlib.sha256(p.read_bytes()).hexdigest() for p in (
                          ROOT / "apps/image_studio/imagegen.py", ROOT / "apps/image_studio/scene/scene.py",
                          ROOT / "apps/image_studio/scene/distance.py")}}
        save(args.output / "report.json", report)
        if args.klein:
            report["method"] = "Three paired seeds, alternating order. Klein 9B Base, 50 steps CFG 4. " \
                "Native ReferenceLatent face photo, pose and depth, each 0.25 MP. Same current geometry " \
                "and locked prior pilot scene in both arms. No PuLID, ControlNet, regional masks, " \
                "face detail or replacement. This measures changed pose guides, not SDF mask conditioning."
        for index, seed in enumerate(args.seeds):
            for arm in (["before", "after"] if index % 2 == 0 else ["after", "before"]):
                ensure_backend(studio, backend)
                folder = args.output / ("%s_%d" % (arm, seed))
                folder.mkdir()
                with ExitStack() as stack:
                    if arm == "before":
                        stack.enter_context(patch.object(sc, "head_point", side_effect=lambda p, head=None: p))
                        stack.enter_context(patch.object(sc, "id_render", old.id_render))
                        stack.enter_context(patch.object(sc, "_character_mask_buffers", old._character_mask_buffers))
                    maps, notes = sc.scene_maps(scene, {"pose", "composition", "source"}, str(folder))
                    words, extra = sc.generation(scene, maps, characters, identities)
                    settings = dict(ig.default_settings(), **extra)
                    settings.update(model="flux-dev", backend="5090", scene=words.text,
                        seed=seed, seed_mode="fixed", steps=25, guidance=4.0,
                        auto_refine=False, hand_pass=False, head_swap=False,
                        glasses_pass=False, refine=False, batch=1)
                    if args.klein:
                        settings.update(model="klein-9b", steps=50, guidance=4.0,
                                        face_detail=False, sampler="euler")
                        settings["references"]["face"] = settings["scene_faces"]["people"][0]["face"]
                        settings["scene"] = ("Create a realistic photograph. Reference image 1 supplies the woman's "
                            "facial identity. Reference image 2 is a skeleton pose guide: match its body position "
                            "and framing. Reference image 3 is a depth guide: match the scene layout. "
                            "Render natural clothing, skin and scenery, without visible guide lines. " + words.text)
                    # Preflight rejects missing conditioning or identity models.
                    plan = studio.preview(settings, backend)
                    if plan.errors:
                        raise RuntimeError("Preflight: " + "; ".join(plan.errors))
                    if args.klein:
                        if len(plan.images) != 3:
                            raise RuntimeError("Klein requires all three native reference inputs")
                    else:
                        pulid, why = studio._pulid(client, plan, set(capabilities["nodes"]))
                        if not pulid:
                            raise RuntimeError("Identity conditioning required for comparison: " + why)
                    if not any(p.get("identity") and p.get("face") for p in settings["scene_faces"]["people"]):
                        raise RuntimeError("Scene has no linked identity with an existing face reference")
                    save(folder / "settings.json", settings)
                    client.free()
                    print("RENDER", arm, seed, flush=True)
                    started = time.perf_counter()
                    job = ig.Job(settings, backend)
                    last = [None, 0]
                    def progress(job):
                        now = time.monotonic()
                        if job.status != last[0] or now - last[1] > 20:
                            print("PROGRESS", arm, seed, job.status, job.detail, flush=True)
                            last[:] = [job.status, now]
                    studio.run_job(job, progress)
                    row = {"arm": arm, "seed": seed, "status": job.status,
                           "seconds": time.perf_counter() - started, "detail": job.detail,
                           "outputs": job.outputs, "record": job.record,
                           "maps": maps, "notes": notes, "warnings": job.plan.warnings if job.plan else []}
                    report["runs"].append(row)
                    save(args.output / "report.json", report)
                    print("DONE", arm, seed, job.status, row["seconds"], flush=True)
                    if job.status != "complete":
                        raise RuntimeError("Render failed: " + job.detail)
    finally:
        studio.close()


if __name__ == "__main__":
    main()
