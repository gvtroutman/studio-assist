"""Pose detector measurements and a contact sheet for compare_pose_strength."""
import json
import argparse
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from apps.image_studio import imagegen as ig
from apps.image_studio.scene import scene as sc
from tools.run_scene_likeness import save


def main():
    from PIL import Image, ImageDraw
    parser = argparse.ArgumentParser()
    parser.add_argument('--corrected', action='store_true')
    args = parser.parse_args()
    output = ROOT / 'Scene Math Measurements' / ('pose-contact-comparison' if args.corrected else 'pose-strength-comparison')
    arms = ('pose085', 'pose100') if args.corrected else ('scaled', 'full')
    report = json.loads((output / 'report.json').read_text(encoding='utf-8'))
    os.environ['STUDIO_SETTINGS'] = str(output / 'settings.json')
    studio = ig.Studio(root=str(output / 'library'))
    try:
        # Figures arrive far-to-near; select the foreground musician by
        # projected shoulder span, rather than the first (background) figure.
        figures = sc.pose_figures(report['scene'])
        def shoulder_span(figure):
            pts = figure['points']
            return math.dist(pts[2], pts[5]) if pts[2] and pts[5] else 0
        target = max(figures, key=shoulder_span)['points']
        # COCO -> OpenPose. Body only; head/hand dots have different roles.
        indices = {5: 5, 6: 2, 7: 6, 8: 3, 9: 7, 10: 4, 11: 11, 12: 8}
        detections = {}
        for row in report['runs']:
            path = row['outputs'][0]
            detected, backend = studio.find_poses(path)
            detections[(row['seed'], row['arm'])] = detected
            save(output / f"{row['arm']}_{row['seed']}" / 'detected_pose.json', detected)
        # Use the same confidently detected, visible joints for each seed's pair.
        for seed in sorted({r['seed'] for r in report['runs']}):
            people = {}
            for arm in arms:
                det = detections[(seed, arm)]
                def match(person):
                    pts = person['points']
                    ds = [math.hypot(pts[c][0] / det['width'] - target[o][0],
                                     pts[c][1] / det['height'] - target[o][1])
                          for c, o in indices.items() if target[o] and pts[c][2] >= 0.3]
                    return sum(ds) / len(ds) if ds else float('inf')
                people[arm] = min(det['people'], key=match) if det['people'] else None
            shared = [c for c, o in indices.items() if target[o] and
                      0 <= target[o][0] <= 1 and 0 <= target[o][1] <= 1 and
                      all(people[a] and people[a]['points'][c][2] >= 0.3 for a in people)]
            for row in (r for r in report['runs'] if r['seed'] == seed):
                det = detections[(seed, row['arm'])]
                pts = people[row['arm']]['points'] if people[row['arm']] else []
                errors = {str(c): math.hypot(pts[c][0] - target[indices[c]][0] * det['width'],
                                           pts[c][1] - target[indices[c]][1] * det['height'])
                          / math.hypot(det['width'], det['height']) * 100 for c in shared}
                row['pose_measurement'] = {'matched_coco_joints': shared,
                    'error_percent_frame_diagonal': errors,
                    'mean_error_percent': sum(errors.values()) / len(errors) if errors else None}
        save(output / 'report.json', report)
        tile_w, tile_h = 600, 435
        sheet = Image.new('RGB', (tile_w * 2, tile_h * 3), '#222222')
        draw = ImageDraw.Draw(sheet)
        def tile(path, x, y, label):
            im = Image.open(path).convert('RGB')
            im.thumbnail((tile_w, tile_h - 35))
            sheet.paste(im, (x + (tile_w - im.width) // 2, y + 35))
            draw.text((x + 12, y + 10), label, fill='white')
        tile(report['maps']['pose'], 0, 0, 'Target skeleton (identical in both arms)')
        tile(report['maps']['composition'], tile_w, 0, 'Target depth (strength unchanged at 0.413)')
        for y, seed in enumerate(sorted({r['seed'] for r in report['runs']}), start=1):
            for x, arm in enumerate(arms):
                row = next(r for r in report['runs'] if r['seed'] == seed and r['arm'] == arm)
                tile(row['outputs'][0], x * tile_w, y * tile_h,
                     f"Seed {seed}: {arm}, pose {row['pose_strength']}")
                print(seed, arm, row['pose_measurement'], flush=True)
        sheet.save(output / 'comparison.jpg', quality=95)
    finally:
        studio.close()


if __name__ == '__main__':
    main()
