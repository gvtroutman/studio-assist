"""Audit and visualize the trained-LoRA effect using the two scored pilots."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
from PIL import Image, ImageDraw, ImageFont, ImageOps


def read(folder, name):
    return json.loads((folder / name).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    previous = Path("Scene Math Measurements/klein-likeness-measured")
    old_report, new_report = read(previous, "report.json"), read(args.output, "report.json")
    old_score, new_score = read(previous, "likeness_scores.json"), read(args.output, "likeness_scores.json")
    if old_score["references"] != new_score["references"]:
        raise RuntimeError("Recognition reference cohort changed")
    originals = {(r["arm"], r["seed"]): r for r in old_report["runs"]}
    original_scores = {(r["arm"], r["seed"]): r for r in old_score["runs"]}
    rows = []
    for run in new_report["runs"]:
        key = run["arm"], run["seed"]
        original = originals[key]
        for field in ("model", "sampler", "scheduler", "steps", "guidance", "width", "height", "seed", "negative"):
            if original["record"].get(field) != run["record"].get(field):
                raise RuntimeError("Changed recipe field: " + field)
        expected = original["record"]["prompt"] + " A photograph of " + new_report["adapter"]["trigger"] + "."
        if run["record"]["prompt"] != expected:
            raise RuntimeError("Prompt changed beyond the identity trigger")
        for kind in ("pose", "composition"):
            if hashlib.sha256(Path(original["maps"][kind]).read_bytes()).digest() != hashlib.sha256(Path(run["maps"][kind]).read_bytes()).digest():
                raise RuntimeError("Guide pixels changed")
    for scored in new_score["runs"]:
        original = original_scores[(scored["arm"], scored["seed"])]
        rows.append({"arm": scored["arm"], "seed": scored["seed"],
                     "without_lora": original["cosine_to_reference_centroid"],
                     "with_lora": scored["cosine_to_reference_centroid"],
                     "delta": scored["cosine_to_reference_centroid"] - original["cosine_to_reference_centroid"]})
    result = {"method": "Same seed, reference pixels and recipe; adds trained LoRA and its trigger. "
              "Same recognition cohort, which may include LoRA training photographs; not held-out validation.",
              "runs": rows, "mean_without_lora": statistics.mean(r["without_lora"] for r in rows),
              "mean_with_lora": statistics.mean(r["with_lora"] for r in rows),
              "mean_delta": statistics.mean(r["delta"] for r in rows),
              "improved_images": sum(r["delta"] > 0 for r in rows)}
    (args.output / "lora_comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    seeds = sorted({r["seed"] for r in rows})
    cell, label = 320, 48
    sheet = Image.new("RGB", (3 * cell, len(seeds) * (cell + label)), "#161616")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 16)
    settings = read(args.output / ("after_%d" % seeds[0]), "settings.json")
    with Image.open(settings["scene_faces"]["people"][0]["face"]) as image:
        reference = ImageOps.contain(ImageOps.exif_transpose(image).convert("RGB"), (cell, cell))
    for index, seed in enumerate(seeds):
        y = index * (cell + label)
        sheet.paste(reference, ((cell - reference.width) // 2, y + (cell - reference.height) // 2))
        draw.text((8, y + cell + 4), "Conditioning reference", font=font, fill="white")
        for column, title, scores in ((1, "Without LoRA", old_score), (2, "Trained LoRA", new_score)):
            scored = next(r for r in scores["runs"] if r["seed"] == seed and r["arm"] == "after")
            box = scored["bbox"]
            pad = max(box[2] - box[0], box[3] - box[1]) * 0.35
            with Image.open(scored["image"]) as image:
                crop = image.convert("RGB").crop((max(0, box[0] - pad), max(0, box[1] - pad),
                    min(image.width, box[2] + pad), min(image.height, box[3] + pad)))
                crop = ImageOps.contain(crop, (cell, cell))
            x = column * cell
            sheet.paste(crop, (x + (cell - crop.width) // 2, y + (cell - crop.height) // 2))
            draw.text((x + 8, y + cell + 4), "%s | %d | %.4f" % (title, seed, scored["cosine_to_reference_centroid"]), font=font, fill="white")
    sheet.save(args.output / "lora_comparison.jpg", quality=94)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
