"""Measure the scene mathematics against the pre-change algorithms.

The baseline ID renderer uses today's geometry, so unrelated mesh changes
cannot masquerade as a mathematical improvement. No GUI or GPU is used.
"""
import argparse
import copy
import importlib.util
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from apps.image_studio.scene import scene as sc
from apps.image_studio.scene.distance import signed_distance, feather_distance


def timed(call, count=3):
    samples = []
    for _ in range(count):
        start = time.perf_counter()
        result = call()
        samples.append(time.perf_counter() - start)
    return result, {"median_seconds": statistics.median(samples), "samples_seconds": samples}


def baseline(output):
    source = subprocess.run(["git", "show", "HEAD:apps/image_studio/scene/scene.py"],
                            cwd=ROOT, capture_output=True, check=True).stdout
    path = output / "baseline_scene.py"
    path.write_bytes(source)
    spec = importlib.util.spec_from_file_location("scene_before_math", path)
    before = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(before)
    # Isolate only the ownership algorithm; both arms use identical meshes.
    before.render, before.frame_size = sc.render, sc.frame_size
    return before


def grey(data, width, height):
    rgb = bytearray(width * height * 3)
    for k in range(3):
        rgb[k::3] = data
    return sc.rgb_png(bytes(rgb), width, height)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--scenes", nargs="+", default=[],
                        help="scene names in Scene Enhancements/prepared, without .scene.json")
    parser.add_argument("--mask-scene", help="the scene whose character masks are feathered "
                        "(default: the first of --scenes)")
    args = parser.parse_args()
    if not args.probe and not args.scenes:
        parser.error("--scenes is required unless --probe")
    args.output.mkdir(parents=True, exist_ok=args.probe)
    if args.probe:
        from apps.image_studio import imagegen as ig
        library = ig.Library()
        report = {"backends": [], "models": [{k: m.get(k) for k in
                  ("id", "name", "family", "workflow")} for m in library.all("models")],
                  "identities": [{"id": i["id"], "references": len(i.get("references") or []),
                                  "head": i.get("head")} for i in library.all("identities")]}
        for backend in library.all("backends"):
            entry = {k: backend.get(k) for k in ("id", "name", "url", "shares_llm_gpu")}
            try:
                client = ig.ComfyUIClient(backend)
                entry["queue"] = client.get_queue()
                entry["health"] = client.health()
                entry["inventory"] = {k: sorted(v) for k, v in client.inventory().items()}
                entry["identity_nodes"] = sorted(set(client.node_types()) & {
                    "ApplyPulidFlux", "PulidFluxModelLoader", "StudioWithAnyone",
                    "StudioFacePaste", "ModelPatchLoader"})
            except Exception as error:
                entry["error"] = str(error)
            report["backends"].append(entry)
        (args.output / "probe.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
        return
    before = baseline(args.output)
    report = {"method": "Same current geometry, before/after algorithms, three timed repetitions. "
              "Depth and reprojection evaluated on occupied pixels; default relief=1. "
              "SDF is discrete, half-pixel-corrected; not analytic mesh distance.", "scenes": []}
    for name in args.scenes:
        scene, _ = sc.load(str(ROOT / "Scene Enhancements" / "prepared" / (name + ".scene.json")))
        w, h = 320, 240
        folder = args.output / name
        folder.mkdir()
        with patch.object(sc, "frame_size", return_value=(w, h)):
            old, old_time = timed(lambda: before.id_render(scene, w, h))
            new, new_time = timed(lambda: sc.id_render(scene, w, h))
            depths = sc.depth_values(scene, w, h)
            cam = sc.Camera(scene["camera"], w, h)
            def errors(rendered):
                inst, _, _, world, _ = rendered
                wrong, maximum_pixel_error, maximum_depth_error, count = 0, 0, 0, 0
                for i, owner in enumerate(inst):
                    if not owner:
                        continue
                    p = world[i * 3:i * 3 + 3]
                    x, y, z = cam.project(p)
                    error = abs(1 / z - depths[i])
                    wrong += error > max(1e-7, depths[i] * 1e-6)
                    maximum_pixel_error = max(maximum_pixel_error,
                        math.hypot(x - (i % w + 0.5), y - (i // w + 0.5)))
                    maximum_depth_error = max(maximum_depth_error, error)
                    count += 1
                return {"visible_pixels": count, "pixels_with_wrong_depth": wrong,
                        "max_reprojection_error_pixels": maximum_pixel_error,
                        "max_inverse_depth_error": maximum_depth_error}
            current_pose = sc.pose_figures(scene, w, h)
            with patch.object(sc, "head_point", side_effect=lambda p, head=None: p):
                old_pose = sc.pose_figures(scene, w, h)
            displacements = [math.hypot((a[0] - b[0]) * w, (a[1] - b[1]) * h)
                for old_face, new_face in zip(old_pose, current_pose)
                for a, b in zip(old_face["face"], new_face["face"])]
            entry = {"scene": name, "size": [w, h], "before": dict(old_time, **errors(old)),
                     "after": dict(new_time, **errors(new)),
                     "landmark_movement_pixels": {"count": len(displacements),
                         "max": max(displacements, default=0),
                         "mean": statistics.mean(displacements) if displacements else 0}}
            for label, result in (("before", old), ("after", new)):
                (folder / (label + "_ids.png")).write_bytes(grey(result[0], w, h))
            (folder / "pose_after.png").write_bytes(sc.pose_png(scene))
            with patch.object(sc, "head_point", side_effect=lambda p, head=None: p):
                (folder / "pose_before.png").write_bytes(sc.pose_png(scene))
        report["scenes"].append(entry)
        print(json.dumps(entry), flush=True)
    # Real masks at normal generation resolution; isolate feather cost.
    scene, _ = sc.load(str(ROOT / "Scene Enhancements" / "prepared"
                           / ((args.mask_scene or args.scenes[0]) + ".scene.json")))
    w, h, masks = sc._character_mask_buffers(scene, 0)
    report["feather"] = []
    for char, mask in masks.items():
        old, old_time = timed(lambda: sc._feather(mask, w, h, 6))
        field, field_time = timed(lambda: signed_distance(mask, w, h))
        new, new_time = timed(lambda: feather_distance(field, 6))
        entry = {"character": char, "size": [w, h], "box_blur": old_time,
                 "distance_transform": field_time, "smoothstep": new_time,
                 "changed_alpha_pixels": sum(a != b for a, b in zip(old, new)),
                 "max_alpha_difference": max(abs(a - b) for a, b in zip(old, new)),
                 "hard_mask_mismatches": sum(bool(a) != (b > 0) for a, b in zip(mask, field))}
        for label, data in (("before", old), ("after", new)):
            (args.output / (char + "_mask_" + label + ".png")).write_bytes(grey(data, w, h))
        report["feather"].append(entry)
        print(json.dumps(entry), flush=True)
    rng = random.Random(42)
    cam = sc.Camera(dict(scene["camera"], yaw=0, pitch=0), 320, 240)
    maximum = [0, 0, 0]
    for _ in range(1000):
        yaw, pitch, roll = rng.uniform(-179, 179), rng.uniform(-85, 85), rng.uniform(-179, 179)
        angles = sc.head_angles(cam, cam.target, sc.euler(yaw, pitch, roll))
        for i, (key, expected) in enumerate(zip(("yaw", "pitch", "roll"), (yaw, -pitch, roll))):
            maximum[i] = max(maximum[i], abs(angles[key] - expected))
    report["head_angles"] = {"cases": 1000, "max_absolute_error_degrees": maximum}
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("REPORT", args.output / "report.json", flush=True)


if __name__ == "__main__":
    main()
