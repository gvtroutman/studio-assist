"""Inspect a completed benchmark with Pillow from the bundled artifact runtime.

This reporting helper runs separately from Studio Assist; it adds no app dependency.
"""
import argparse
import hashlib
import html
import json
from pathlib import Path
import statistics

from PIL import Image, ImageChops, ImageDraw, ImageStat


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder")
    args = ap.parse_args()
    folder = Path(args.folder).resolve()
    data = json.loads((folder / "results.json").read_text(encoding="utf-8"))
    pairs = {}
    for run in data["runs"]:
        pairs.setdefault((run["case"], run["seed"]), {})[run["variant"]] = run
    comparisons = []
    tiles = []
    for (case, seed), pair in pairs.items():
        if set(pair) != {"before", "after"}:
            continue
        before = Image.open(pair["before"]["outputs"][0]).convert("RGB")
        after = Image.open(pair["after"]["outputs"][0]).convert("RGB")
        difference = ImageChops.difference(before, after)
        stats = ImageStat.Stat(difference)
        comparisons.append({"case": case, "seed": seed, "size": before.size,
                            "pixels_identical": difference.getbbox() is None,
                            "mean_absolute_channel_difference": statistics.mean(stats.mean),
                            "before_pixel_sha256": hashlib.sha256(before.tobytes()).hexdigest(),
                            "after_pixel_sha256": hashlib.sha256(after.tobytes()).hexdigest()})
        tile = Image.new("RGB", (1040, 650), "#15191f")
        draw = ImageDraw.Draw(tile)
        for x, label, picture in ((0, "before", before), (520, "after", after)):
            picture.thumbnail((500, 590), Image.Resampling.LANCZOS)
            tile.paste(picture, (x + (520 - picture.width) // 2, 45 + (590 - picture.height) // 2))
            draw.text((x + 15, 15), "%s | %s | seed %d | %.2f s" % (
                label.upper(), case, seed, pair[label]["elapsed_seconds"]), fill="white")
        target = folder / ("comparison_%s_%d.jpg" % (case, seed))
        tile.save(target, quality=94)
        tiles.append(target)
        if case in ("portrait", "group"):
            # Same geometric crops in both arms: top-middle (faces), middle
            # lower frame (hands). They are comparisons, not face detections.
            w, h = before.size
            regions = [(int(w*.2), int(h*.05), int(w*.8), int(h*.48)),
                       (int(w*.12), int(h*.48), int(w*.88), int(h*.88))]
            detail = Image.new("RGB", (1040, 1060), "#15191f")
            draw = ImageDraw.Draw(detail)
            for row, region in enumerate(regions):
                for x, label, picture in ((0, "before", before), (520, "after", after)):
                    crop = picture.crop(region)
                    crop.thumbnail((500, 480), Image.Resampling.LANCZOS)
                    detail.paste(crop, (x + (520-crop.width)//2, row*530 + 45))
                    draw.text((x+15, row*530+15), "%s %s seed %d %s" % (
                        label, case, seed, "faces" if row == 0 else "hands"), fill="white")
            detail.save(folder / ("details_%s_%d.jpg" % (case, seed)), quality=96)
    summary = []
    for case in sorted({key[0] for key, pair in pairs.items()
                        if set(pair) == {"before", "after"}}):
        row = {"case": case}
        for label in ("before", "after"):
            times = [r["elapsed_seconds"] for r in data["runs"]
                     if r["case"] == case and r["variant"] == label]
            row[label + "_seconds"] = times
            row[label + "_median"] = statistics.median(times)
        row["change_percent"] = 100 * (row["after_median"] / row["before_median"] - 1)
        summary.append(row)
    result = {"timing": summary, "pixel_comparisons": comparisons,
              "uploads": data["uploads"], "cancel": data["cancel"], "method": data["method"]}
    (folder / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    rows = "".join("<tr><td>%s</td><td>%.2f s</td><td>%.2f s</td><td>%+.1f%%</td></tr>" % (
        r["case"], r["before_median"], r["after_median"], r["change_percent"]) for r in summary)
    images = "".join('<figure><img src="%s"><figcaption>%s</figcaption></figure>' % (
        p.name, html.escape(p.stem)) for p in tiles)
    page = """<!doctype html><meta charset="utf-8"><title>Image pipeline benchmark</title>
    <style>body{font:16px system-ui;background:#15191f;color:#eee;max-width:1100px;margin:40px auto}
    table{border-collapse:collapse}td,th{padding:10px 24px;border-bottom:1px solid #444}
    img{max-width:100%%}figure{margin:30px 0}pre{white-space:pre-wrap}</style>
    <h1>Image pipeline benchmark</h1><p>%s</p>
    <p>Two seeds per case. Timings are medians; a small sample does not establish a speedup.
    Identical RGB pixel hashes verify that these transport changes preserve successful renders.</p>
    <table><tr><th>Case</th><th>Before</th><th>After</th><th>Change</th></tr>%s</table>
    <h2>Upload and cancellation measurements</h2><pre>%s</pre>
    <h2>Matched images</h2>%s""" % (html.escape(data["method"]), rows,
        html.escape(json.dumps({"uploads": data["uploads"], "cancel": data["cancel"]}, indent=2)), images)
    (folder / "report.html").write_text(page, encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
