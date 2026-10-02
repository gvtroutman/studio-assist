"""Create a compact, auditable text summary of the scene-math measurements."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from apps.image_studio.scene import scene as sc


def main():
    folder = ROOT / "Scene Math Measurements/geometry"
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    lines = ["Scene mathematics: measured results", "", report["method"], ""]
    for entry in report["scenes"]:
        before, after = entry["before"], entry["after"]
        lines += [entry["scene"],
            "  Wrong-depth pixels: %d/%d (%.2f%%) -> %d/%d" % (
                before["pixels_with_wrong_depth"], before["visible_pixels"],
                100 * before["pixels_with_wrong_depth"] / before["visible_pixels"],
                after["pixels_with_wrong_depth"], after["visible_pixels"]),
            "  Ownership runtime, median of 3: %.4fs -> %.4fs (%+.1f%%)" % (
                before["median_seconds"], after["median_seconds"],
                100 * (after["median_seconds"] / before["median_seconds"] - 1)),
            "  Maximum landmark movement at 320x240: %.4f pixels" %
                entry["landmark_movement_pixels"]["max"]]
        # Pixel-level difference of the actual rendered conditioning maps.
        old, w, h = sc.read_png(str(folder / entry["scene"] / "pose_before.png"))
        new, nw, nh = sc.read_png(str(folder / entry["scene"] / "pose_after.png"))
        if (w, h) != (nw, nh):
            raise ValueError("Comparison maps have different dimensions")
        changed = sum(old[i:i + 3] != new[i:i + 3] for i in range(0, len(old), 4))
        entry["pose_map_changed_pixels"] = {"size": [w, h], "changed": changed}
        lines += ["  Actual pose-map changed pixels: %d/%d at %dx%d" % (changed, w * h, w, h), ""]
    for entry in report["feather"]:
        old = entry["box_blur"]["median_seconds"]
        new = entry["distance_transform"]["median_seconds"] + entry["smoothstep"]["median_seconds"]
        lines += ["Feather, %s at %dx%d:" % (entry["character"], *entry["size"]),
                  "  %.3fs -> %.3fs (%.2fx); unchanged hard ownership (%d mismatches)" % (
                      old, new, new / old, entry["hard_mask_mismatches"]), ""]
    lines += ["Head-angle recovery: 1000 randomized rotations (yaw/roll +/-179, pitch +/-85 deg)",
              "  Max errors in yaw/pitch/roll: %s degrees" % report["head_angles"]["max_absolute_error_degrees"],
              "", "Scope:",
              "  These are geometry and CPU timings, not generated-face recognition scores.",
              "  Depth reference is the existing exact-relief depth pass, with the same scene geometry.",
              "  Pose maps are rasterized at the pose renderer's normal resolution.",
              "  Distances are discrete image-space distances, not analytic 3D distances.",
              "  Timings are a small workstation sample and have no confidence interval.", ""]
    (folder / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (folder / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
