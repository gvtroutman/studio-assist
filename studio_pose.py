#!/usr/bin/env python3
"""
studio_pose - a person's pose as an OpenPose stick figure, for the Image
Studio's pose ControlNet. No tkinter: the editor window is in
studio_images_ui (`PoseEditor`); this is what it edits and what it writes.

A pose is 18 OpenPose body keypoints (the COCO order every OpenPose ControlNet
was trained on), each [x, y] as a fraction of the picture's width and height,
or None for a joint that is not seen (a hand behind the back, the far ear in
profile). `render()` draws them the way OpenPose's own preprocessor does -
coloured limbs at 60% on black, full-colour joints - because that is the
picture the ControlNet reads; a figure drawn any other way is guessed at.

"Right" is the subject's right, as in OpenPose: facing the viewer, it is on
the picture's left.
"""

import hashlib
import json
import math
import os

import studio_icons

JOINTS = ["nose", "neck", "right shoulder", "right elbow", "right wrist",
          "left shoulder", "left elbow", "left wrist", "right hip", "right knee",
          "right ankle", "left hip", "left knee", "left ankle", "right eye", "left eye",
          "right ear", "left ear"]
# OpenPose's limbSeq, 0-based, and the colour each limb and joint is drawn in.
LIMBS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10),
         (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]
COLOURS = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
           (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
           (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
           (255, 0, 255), (255, 0, 170), (255, 0, 85)]
# Left and right swap under a mirror.
MIRROR = {2: 5, 3: 6, 4: 7, 8: 11, 9: 12, 10: 13, 14: 15, 16: 17}
MIRROR.update({b: a for a, b in list(MIRROR.items())})
# What hangs off each joint: dragging one carries these with it.
CHILDREN = {1: [0, 2, 5, 8, 11], 0: [14, 15, 16, 17], 2: [3], 3: [4], 5: [6], 6: [7],
            8: [9], 9: [10], 11: [12], 12: [13]}
RENDER_EDGE = 768             # px on the long edge; the ControlNet scales it to the latent

# Presets in body units: x across (0 is the middle, facing the viewer), y down
# from the top of the head, ankles near 1. `place()` fits them to a frame.
_HEAD = {0: (0, 0.06), 14: (-0.022, 0.045), 15: (0.022, 0.045), 16: (-0.048, 0.055),
         17: (0.048, 0.055), 1: (0, 0.16)}
_STAND = {**_HEAD, 2: (-0.11, 0.17), 3: (-0.14, 0.33), 4: (-0.15, 0.47),
          5: (0.11, 0.17), 6: (0.14, 0.33), 7: (0.15, 0.47),
          8: (-0.07, 0.5), 9: (-0.075, 0.73), 10: (-0.08, 0.96),
          11: (0.07, 0.5), 12: (0.075, 0.73), 13: (0.08, 0.96)}


def _pose(**changes):
    p = dict(_STAND)
    for k, v in changes.items():
        p[int(k[1:])] = v
    return p


PRESETS = [
    ("standing", "Standing", _STAND),
    ("arms_up", "Arms up", _pose(j3=(-0.16, 0.02), j4=(-0.2, -0.12), j6=(0.16, 0.02),
                                 j7=(0.2, -0.12))),
    ("hands_hips", "Hands on hips", _pose(j3=(-0.22, 0.3), j4=(-0.09, 0.45),
                                          j6=(0.22, 0.3), j7=(0.09, 0.45),
                                          j10=(-0.13, 0.96), j9=(-0.1, 0.73),
                                          j13=(0.13, 0.96), j12=(0.1, 0.73))),
    ("t_pose", "Arms out", _pose(j3=(-0.27, 0.17), j4=(-0.42, 0.17), j6=(0.27, 0.17),
                                 j7=(0.42, 0.17))),
    ("waving", "Waving", _pose(j6=(0.24, 0.12), j7=(0.25, -0.03))),
    ("arms_crossed", "Arms crossed", _pose(j3=(-0.15, 0.32), j4=(0.08, 0.3),
                                           j6=(0.15, 0.33), j7=(-0.08, 0.31))),
    ("walking", "Walking", _pose(j3=(-0.12, 0.33), j4=(-0.06, 0.46), j6=(0.17, 0.32),
                                 j7=(0.22, 0.44), j9=(-0.11, 0.72), j10=(-0.16, 0.95),
                                 j12=(0.09, 0.73), j13=(0.13, 0.94))),
    ("sitting", "Sitting", _pose(j3=(-0.15, 0.33), j4=(-0.12, 0.47), j6=(0.15, 0.33),
                                 j7=(0.12, 0.47), j8=(-0.08, 0.52), j9=(-0.12, 0.6),
                                 j10=(-0.12, 0.84), j11=(0.08, 0.52), j12=(0.12, 0.6),
                                 j13=(0.12, 0.84))),
    ("kneeling", "Kneeling", _pose(j9=(-0.08, 0.72), j10=(-0.08, 0.74),
                                   j12=(0.09, 0.6), j13=(0.1, 0.8))),
    ("profile", "Side view", {0: (0.06, 0.07), 14: (0.045, 0.05), 16: (-0.01, 0.06),
                              1: (0, 0.16), 2: (0.0, 0.17), 3: (0.02, 0.33),
                              4: (0.05, 0.47), 5: None, 6: None, 7: None, 15: None,
                              17: None, 8: (0, 0.5), 9: (0.03, 0.73), 10: (0.0, 0.96),
                              11: (-0.01, 0.5), 12: (-0.03, 0.73), 13: (-0.05, 0.95)}),
]
PRESET_NAMES = [k for k, _, _ in PRESETS]


def place(body, width, height, fill=0.82):
    """A preset in body units -> points in the frame: the figure `fill` of
    the frame's height (less when a wide figure would not fit across),
    centred."""
    pts = [body.get(i) for i in range(len(JOINTS))]
    seen = [p for p in pts if p]
    xs, ys = [p[0] for p in seen], [p[1] for p in seen]
    span_x, span_y = max(xs) - min(xs), max(ys) - min(ys)
    scale = min(fill * height / max(span_y, 1e-6), fill * width / max(span_x, 1e-6))
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    return [None if p is None else [round(0.5 + (p[0] - cx) * scale / width, 4),
                                    round(0.5 + (p[1] - cy) * scale / height, 4)]
            for p in pts]


def preset(name, width, height):
    body = dict((k, b) for k, _, b in PRESETS).get(name) or _STAND
    return place(body, width, height)


def clean(points):
    """Anything -> 18 points, [x, y] floats or None, or None if unusable."""
    if not isinstance(points, list) or len(points) != len(JOINTS):
        return None
    out = []
    for p in points:
        if p is None:
            out.append(None)
        elif (isinstance(p, (list, tuple)) and len(p) == 2
              and all(isinstance(c, (int, float)) for c in p)):
            out.append([float(p[0]), float(p[1])])
        else:
            return None
    return out if any(out) else None


def mirror(points):
    """Left for right, across the middle of the frame."""
    out = [None] * len(points)
    for i, p in enumerate(points):
        out[MIRROR.get(i, i)] = None if p is None else [round(1 - p[0], 4), p[1]]
    return out


def refit(points, old, new):
    """Points drawn in an `old` (w, h) frame, into a `new` one: the same
    figure at the same proportions, scaled to fit and centred, so changing
    the picture's size never stretches the person."""
    (ow, oh), (nw, nh) = old, new
    if not all((ow, oh, nw, nh)) or abs(ow / oh - nw / nh) < 1e-3:
        return [None if p is None else list(p) for p in points]
    k = min(nw / ow, nh / oh)
    return [None if p is None else
            [round(((p[0] - 0.5) * ow * k) / nw + 0.5, 4),
             round(((p[1] - 0.5) * oh * k) / nh + 0.5, 4)] for p in points]


def size_for(width, height, edge=RENDER_EDGE):
    k = edge / max(width, height)
    return max(64, int(round(width * k))), max(64, int(round(height * k)))


def render(points, width, height):
    """The OpenPose picture of `points` at width x height -> PNG bytes."""
    w, h = size_for(width, height)
    buf = bytearray(w * h * 4)
    buf[3::4] = b"\xff" * (w * h)
    stick = max(3, round(max(w, h) / 170))
    pix = [None if p is None else (p[0] * w, p[1] * h) for p in points]

    def blot(x0, y0, x1, y1, r, colour, alpha):
        """A capsule from (x0, y0) to (x1, y1), radius r, over what is there."""
        dx, dy = x1 - x0, y1 - y0
        ll = dx * dx + dy * dy or 1e-9
        lo_x, hi_x = max(0, int(min(x0, x1) - r)), min(w - 1, int(max(x0, x1) + r) + 1)
        lo_y, hi_y = max(0, int(min(y0, y1) - r)), min(h - 1, int(max(y0, y1) + r) + 1)
        rr = r * r
        keep = 1 - alpha
        cr, cg, cb = (c * alpha for c in colour)
        for y in range(lo_y, hi_y + 1):
            py = y + 0.5 - y0
            row = y * w * 4
            for x in range(lo_x, hi_x + 1):
                px = x + 0.5 - x0
                t = (px * dx + py * dy) / ll
                t = 0.0 if t < 0 else 1.0 if t > 1 else t
                ex, ey = px - t * dx, py - t * dy
                if ex * ex + ey * ey <= rr:
                    i = row + x * 4
                    buf[i] = int(buf[i] * keep + cr)
                    buf[i + 1] = int(buf[i + 1] * keep + cg)
                    buf[i + 2] = int(buf[i + 2] * keep + cb)

    for n, (a, b) in enumerate(LIMBS):
        if pix[a] and pix[b]:
            blot(pix[a][0], pix[a][1], pix[b][0], pix[b][1], stick, COLOURS[n], 0.6)
    for i, p in enumerate(pix):
        if p:
            blot(p[0], p[1], p[0], p[1], stick, COLOURS[i], 1.0)
    return studio_icons.png(bytes(buf), w, h)


def save(points, width, height, folder):
    """The picture of this pose at this size, under `folder`, named by what
    it is, so the same pose is drawn once. -> its path."""
    key = json.dumps([points, size_for(width, height)], sort_keys=True)
    path = os.path.join(folder, hashlib.sha1(key.encode()).hexdigest()[:16] + ".png")
    if not os.path.exists(path):
        os.makedirs(folder, exist_ok=True)
        tmp = path + ".part"
        with open(tmp, "wb") as f:
            f.write(render(points, width, height))
        os.replace(tmp, path)
    return path


def carried(joint):
    """`joint` and everything that hangs off it."""
    out, todo = [], [joint]
    while todo:
        j = todo.pop()
        out.append(j)
        todo.extend(CHILDREN.get(j, []))
    return out


def near(points, x, y, width, height, reach):
    """The joint nearest (x, y) px in a width x height view, within `reach`
    px, or None. Hidden joints count, so they can be shown again."""
    best, dist = None, reach
    for i, p in enumerate(points):
        if p is None:
            continue
        d = math.hypot(p[0] * width - x, p[1] * height - y)
        if d <= dist:
            best, dist = i, d
    return best
