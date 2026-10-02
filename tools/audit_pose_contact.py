"""Audit saved accordion contacts and prepare an isolated, fitted scene copy."""
import copy
import json
import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from apps.image_studio.scene import scene as sc


def main():
    output = ROOT / 'Scene Math Measurements/pose-contact-audit'
    output.mkdir(parents=True, exist_ok=True)
    scene, _ = sc.load(str(ROOT / 'Scene Enhancements/prepared/oktoberfest - playing accordion.scene.json'))
    person = next(o for o in scene['objects'] if o['asset'] == 'person')
    original = copy.deepcopy(person['pose']['controls'])
    look = person['look']
    before = sc.accordion_contacts(original, look)
    controls = copy.deepcopy(original)
    rng = random.Random(261002)
    for side in ('r', 'l'):
        keys = [f'arm_{side}_raise', f'arm_{side}_out', f'arm_{side}_bend', f'wrist_{side}_bend']
        def cost(c):
            gap = sc.accordion_contacts(c, look)[side]
            return gap * gap + 1e-8 * sum((c[k] - original[k]) ** 2 for k in keys)
        best = dict(controls)
        for start in range(12):
            candidate = dict(controls)
            for k in keys:
                candidate[k] = sc.pose_controls('accordion')[k] if start == 0 else (
                    original[k] if start == 1 else rng.uniform(*sc.CONTROL_RANGE[k]))
            for step in (30, 15, 7, 3, 1, 0.3):
                for iteration in range(30):
                    changed = False
                    for k in keys:
                        old, score = candidate[k], cost(candidate)
                        chosen = old
                        for sign in (-1, 1):
                            lo, hi = sc.CONTROL_RANGE[k]
                            candidate[k] = max(lo, min(hi, old + sign * step))
                            value = cost(candidate)
                            if value < score:
                                chosen, score = candidate[k], value
                        candidate[k] = chosen
                        changed |= chosen != old
                    if not changed:
                        break
            if cost(candidate) < cost(best):
                best = candidate
        controls = best
    person['pose'] = {'preset': '', 'controls': controls}
    after = sc.accordion_contacts(controls, look)
    audit = {'before_gap_metres': before, 'after_gap_metres': after,
             'control_changes': {k: [original[k], v] for k, v in controls.items() if v != original[k]},
             'note': 'Landmark proximity audit, not a collision or full hand-contact solver.'}
    (output / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    sc.save(scene, str(output / 'corrected.scene.json'))
    maps, notes = sc.scene_maps(scene, {'pose', 'composition'}, str(output))
    audit['maps'] = maps
    audit['notes'] = sc.scene_text(scene).notes + notes
    (output / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    main()
