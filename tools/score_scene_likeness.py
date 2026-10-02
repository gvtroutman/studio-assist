"""CPU ArcFace scoring and contact sheet for a paired scene-generation pilot.

Run with the ComfyUI venv: it owns InsightFace, OpenCV and Pillow. This is
a recognition proxy against existing library photos, not a blinded study.
"""
import argparse
import json
from pathlib import Path
import statistics
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--release-backend", action="store_true")
    args = parser.parse_args()
    import cv2
    import numpy as np
    from insightface.app import FaceAnalysis
    from PIL import Image, ImageDraw, ImageOps, ImageFont
    analysis = FaceAnalysis(name="antelopev2", root="D:/ComfyUI/models/insightface",
        allowed_modules=["detection", "recognition", "landmark_3d_68"],
        providers=["CPUExecutionProvider"])
    analysis.prepare(ctx_id=-1, det_size=(640, 640))
    if args.check:
        print("CPU ArcFace ready", flush=True)
        return
    report = json.loads((args.output / "report.json").read_text(encoding="utf-8"))
    release = {"requested": args.release_backend}
    if args.release_backend:
        url = report["runs"][0]["record"]["backend"]["url"]
        def get_json(path):
            with urllib.request.urlopen(url + path, timeout=30) as response:
                return json.load(response)
        try:
            queue = get_json("/queue")
            if queue.get("queue_running") or queue.get("queue_pending"):
                release["status"] = "skipped: another job is using the backend"
            else:
                request = urllib.request.Request(url + "/free",
                    data=json.dumps({"unload_models": True, "free_memory": True}).encode(),
                    headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(request, timeout=30):
                    pass
                release["status"] = "unload and free request accepted"
                release["devices"] = get_json("/system_stats").get("devices")
        except Exception as error:
            release["status"] = "release could not be confirmed: " + str(error)
    identities = {p["id"]: p for p in json.loads((args.output / "library/identities.json").read_text(encoding="utf-8"))}
    def read_faces(path):
        with Image.open(path) as source:
            rgb = np.asarray(ImageOps.exif_transpose(source).convert("RGB"))
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return analysis.get(image), rgb.shape[1], rgb.shape[0]
    settings = [json.loads((args.output / ("%s_%d" % (r["arm"], r["seed"])) / "settings.json").read_text(
        encoding="utf-8")) for r in report["runs"]]
    # Keep an invalid pairing from looking like a mathematical improvement.
    recipe_fields = ("prompt", "negative", "model", "sampler", "scheduler", "steps",
                     "guidance", "width", "height", "seed", "loras")
    for seed in {r["seed"] for r in report["runs"]}:
        pair = {r["arm"]: r for r in report["runs"] if r["seed"] == seed}
        if set(pair) != {"before", "after"}:
            raise RuntimeError("Incomplete before/after pair for seed %d" % seed)
        for key in recipe_fields:
            if pair["before"]["record"].get(key) != pair["after"]["record"].get(key):
                raise RuntimeError("Unmatched recipe field %s at seed %d" % (key, seed))
    conditioned = {str(Path(p).resolve()).casefold() for setting in settings
        for person in setting["scene_faces"]["people"]
        for p in (person.get("photos") or [person.get("face")])[:3] if p}
    conditioned.update(str(Path(p["face"]).resolve()).casefold()
        for setting in settings for p in setting["scene_faces"]["people"] if p.get("face"))
    targets = settings[0]["scene_faces"]["people"]
    if len(targets) != 1 or not targets[0].get("identity"):
        raise RuntimeError("This pilot scorer requires exactly one explicitly linked scene person")
    identity = identities[targets[0]["identity"]]
    reference_vectors, reference_paths, failed = [], [], []
    for path in identity.get("references") or []:
        if str(Path(path).resolve()).casefold() in conditioned:
            continue
        found, _, _ = read_faces(path)
        if len(found) != 1:
            failed.append({"path": path, "face_count": len(found)})
            continue
        reference_vectors.append(found[0].normed_embedding)
        reference_paths.append(path)
    if len(reference_vectors) < 3:
        raise RuntimeError("Fewer than three unconditioned, unambiguous reference photographs")
    # A normalized centroid minimizes the influence of a single photo's angle.
    vectors = np.asarray(reference_vectors)
    centroid = vectors.mean(axis=0)
    centroid /= np.linalg.norm(centroid)
    scores = {"method": "CPU antelopev2 glintr100 ArcFace cosine to normalized mean of existing "
              "library reference embeddings. Excludes paths used as conditioning photos. "
              "Recognition proxy only: photos may be related or duplicated; no held-out study.",
              "identity": identity["id"], "reference_count": len(reference_paths),
              "references": reference_paths, "reference_failures": failed, "backend_release": release,
              "runs": [], "pairs": []}
    cells, run_embeddings = [], {}
    for run, setting in zip(report["runs"], settings):
        row = {"arm": run["arm"], "seed": run["seed"], "generation_seconds": run["seconds"]}
        if run["status"] != "complete" or len(run["outputs"]) != 1:
            row["error"] = "Generation not complete with exactly one output"
            scores["runs"].append(row)
            continue
        path = run["outputs"][0]
        row["image"] = path
        faces, width, height = read_faces(path)
        row["detected_faces"] = len(faces)
        region = setting["scene_faces"]["people"][0]["region"]
        hits = [f for f in faces if region[0] <= (f.bbox[0] + f.bbox[2]) / (2 * width) <= region[2]
                and region[1] <= (f.bbox[1] + f.bbox[3]) / (2 * height) <= region[3]]
        # One scene person and exactly one detected face is unambiguous even
        # if the model moved it away from the projected region; record that.
        face = hits[0] if len(hits) == 1 else (faces[0] if len(faces) == 1 else None)
        if face is None:
            row["error"] = "Missing or ambiguous target face"
        else:
            embedding = face.normed_embedding
            run_embeddings[(run["seed"], run["arm"])] = embedding
            row.update(cosine_to_reference_centroid=float(embedding @ centroid),
                mean_cosine_to_individual_references=float((vectors @ embedding).mean()),
                bbox=[float(x) for x in face.bbox],
                matched_projected_region=bool(len(hits) == 1),
                detected_head_pose=[float(x) for x in face.pose] if face.get("pose") is not None else None,
                planned_head_pose=setting["scene_faces"]["people"][0].get("head_pose"))
            box = face.bbox
            with Image.open(path) as source:
                pad = max(box[2] - box[0], box[3] - box[1]) * 0.35
                crop = source.convert("RGB").crop((max(0, box[0] - pad), max(0, box[1] - pad),
                    min(width, box[2] + pad), min(height, box[3] + pad)))
                cells.append((run["seed"], run["arm"], crop.copy(), row["cosine_to_reference_centroid"]))
        scores["runs"].append(row)
        print(json.dumps(row), flush=True)
    seeds = sorted({r["seed"] for r in scores["runs"]})
    for seed in seeds:
        pair = {r["arm"]: r for r in scores["runs"] if r["seed"] == seed}
        if set(pair) == {"before", "after"} and all("cosine_to_reference_centroid" in r for r in pair.values()):
            scores["pairs"].append({"seed": seed,
                "before": pair["before"]["cosine_to_reference_centroid"],
                "after": pair["after"]["cosine_to_reference_centroid"],
                "before_after_face_cosine": float(run_embeddings[(seed, "before")] @
                                                  run_embeddings[(seed, "after")]),
                "delta": pair["after"]["cosine_to_reference_centroid"] - pair["before"]["cosine_to_reference_centroid"]})
    if scores["pairs"]:
        scores["summary"] = {"paired_seeds": len(scores["pairs"]),
            "mean_before": statistics.mean(r["before"] for r in scores["pairs"]),
            "mean_after": statistics.mean(r["after"] for r in scores["pairs"]),
            "mean_delta": statistics.mean(r["delta"] for r in scores["pairs"]),
            "min_delta": min(r["delta"] for r in scores["pairs"]),
            "max_delta": max(r["delta"] for r in scores["pairs"]),
            "improved_seeds": sum(r["delta"] > 0 for r in scores["pairs"]),
            "limitation": "Single identity and scene, small paired-seed pilot; not statistical evidence of general improvement."}
        print(json.dumps(scores["summary"], indent=2), flush=True)
    (args.output / "likeness_scores.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")
    if cells:
        cell, label_h = 320, 46
        sheet = Image.new("RGB", (cell * 3, (cell + label_h) * len(seeds)), "#161616")
        draw = ImageDraw.Draw(sheet)
        font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 16)
        with Image.open(targets[0]["face"]) as source:
            ref = ImageOps.contain(ImageOps.exif_transpose(source).convert("RGB"), (cell, cell))
        for i, seed in enumerate(seeds):
            y = i * (cell + label_h)
            sheet.paste(ref, ((cell - ref.width) // 2, y + (cell - ref.height) // 2))
            draw.text((8, y + cell + 4), "Conditioning reference", fill="white", font=font)
            for _, arm, picture, score in [c for c in cells if c[0] == seed]:
                x = cell * (1 if arm == "before" else 2)
                picture = ImageOps.contain(picture, (cell, cell))
                sheet.paste(picture, (x + (cell - picture.width) // 2, y + (cell - picture.height) // 2))
                draw.text((x + 8, y + cell + 4), "%s | seed %d | cosine %.4f" % (arm, seed, score), fill="white", font=font)
        sheet.save(args.output / "face_comparison.jpg", quality=94)


if __name__ == "__main__":
    main()
