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

import colorsys
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
RENDER_EDGE = 1024            # px on the long edge; the ControlNet scales it to the latent
DRAWING = 4                   # in the saved picture's name: a change to render() redraws

# Presets in body units: x across (0 is the middle, facing the viewer), y down
# from the top of the head, ankles near 1. `place()` fits them to a frame.
# The head is a real one's proportions (eyes about 1/27 of the height apart):
# doubling it, to spread the eyes, gave every picture a caricature's head.
_HEAD = {0: (0, 0.082), 14: (-0.0185, 0.06), 15: (0.0185, 0.06), 16: (-0.045, 0.068),
         17: (0.045, 0.068), 1: (0, 0.165)}
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
    ("profile", "Side view", {0: (0.035, 0.08), 14: (0.022, 0.06), 16: (-0.012, 0.068),
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


def _face_template():
    """The 68 face landmarks (the iBUG layout DWPose draws), for a face seen
    from the front: eyes at (-1, 0) and (1, 0), y down, nose tip near 1.1."""
    pts = []
    for i in range(17):                                   # jaw, ear to ear
        t = math.pi * (1 - i / 16)
        pts.append((2.1 * math.cos(t), 0.3 + 2.9 * math.sin(t)))
    for side in (-1, 1):                                  # brows
        xs = [-1.8, -1.45, -1.1, -0.75, -0.4] if side < 0 else [0.4, 0.75, 1.1, 1.45, 1.8]
        pts += [(x, -0.55 - 0.25 * math.cos((abs(x) - 1.1) * 2)) for x in xs]
    pts += [(0, -0.2 + 0.37 * i) for i in range(4)]       # bridge
    pts += [(x, 1.25 + 0.08 * (1 - abs(x) * 2)) for x in (-0.5, -0.25, 0, 0.25, 0.5)]
    for cx in (-1, 1):                                    # eyes
        pts += [(cx + 0.45 * math.cos(math.pi - a * math.pi / 3),
                 -0.17 * math.sin(math.pi - a * math.pi / 3)) for a in range(6)]
    pts += [(1.0 * math.cos(math.pi - a * math.pi / 6), 2.0 - 0.4 * math.sin(
        math.pi - a * math.pi / 6)) for a in range(12)]  # mouth
    pts += [(0.6 * math.cos(math.pi - a * math.pi / 4), 2.0 - 0.15 * math.sin(
        math.pi - a * math.pi / 4)) for a in range(8)]
    return pts


FACE = _face_template()


def face_points(points, width, height):
    """The face's landmark dots for a figure whose nose and both eyes are
    seen, as fractions of the frame, else []. DWPose draws these on every
    face it finds, and pose ControlNets learnt from its pictures: a body
    with no face dots came back seen from behind, on every seed tried
    (2026-09-25). So the figure gets a generic face, turned and sized by
    its eyes and put on the nose side of them."""
    nose, r_eye, l_eye = (points[i] if i < len(points) else None for i in (0, 14, 15))
    if not (nose and r_eye and l_eye):
        return []
    rx, ry = r_eye[0] * width, r_eye[1] * height
    lx, ly = l_eye[0] * width, l_eye[1] * height
    mx, my = (rx + lx) / 2, (ry + ly) / 2
    ex, ey = (lx - rx) / 2, (ly - ry) / 2           # template x: half the eye gap
    dx, dy = -ey, ex                                # template y: square to it
    if (nose[0] * width - mx) * dx + (nose[1] * height - my) * dy < 0:
        dx, dy = -dx, -dy                           # down is toward the nose
    return [((mx + u * ex + v * dx) / width, (my + u * ey + v * dy) / height)
            for u, v in FACE]


# ------------------------------------------------------------------- hands
# 21 points a hand, in DWPose's order: the wrist, then thumb, index, middle,
# ring and little finger, four points each from the knuckle out.
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
              (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
              (0, 17), (17, 18), (18, 19), (19, 20)]
HAND_JOINT = (0, 0, 255)      # DWPose draws every hand point blue
HAND_OF = {"right": (4, 3), "left": (7, 6)}   # wrist, elbow
# Each finger in the hand's own frame - y out along the forearm, x toward the
# thumb, the wrist at the origin, wrist to middle fingertip about 1: its
# knuckle, the angle it points (degrees toward the thumb) and its three bones.
FINGERS = [((0.09, 0.44), 8, (0.21, 0.12, 0.10)),      # index
           ((0.0, 0.46), 0, (0.23, 0.14, 0.11)),       # middle
           ((-0.085, 0.44), -7, (0.21, 0.13, 0.10)),   # ring
           ((-0.16, 0.39), -15, (0.16, 0.10, 0.09))]   # little
THUMB = ((0.1, 0.08), (0.16, 0.13, 0.11))
HAND_LENGTH = 0.72            # of the forearm, wrist to middle fingertip
_FIST = (90, 100, 60)
# A shape: how far each finger's three joints bend (degrees; a bend folds
# the finger toward the viewer, so its bones foreshorten and a fist's tips
# come back down onto the palm), how far the fingers fan (x their angles),
# and the thumb's angle and the two bends that carry it across the palm.
HAND_SHAPES = [
    ("relaxed", "Relaxed", {"curl": [(10, 20, 10), (15, 25, 15), (20, 30, 15), (25, 30, 15)],
                            "spread": 1.0, "thumb": (35, 10, 10)}),
    ("open", "Open", {"curl": [(0, 0, 0)] * 4, "spread": 1.9, "thumb": (58, 0, 0)}),
    ("fist", "Fist", {"curl": [_FIST] * 4, "spread": 0.6, "thumb": (18, 45, 35)}),
    ("grab", "Grabbing", {"curl": [(45, 55, 35)] * 4, "spread": 1.1, "thumb": (45, 15, 10)}),
    ("point", "Pointing", {"curl": [(0, 0, 0), _FIST, _FIST, _FIST], "spread": 0.8,
                           "thumb": (18, 45, 35)}),
    ("peace", "Peace", {"curl": [(0, 0, 0), (0, 0, 0), _FIST, _FIST], "spread": 3.2,
                        "thumb": (18, 45, 35)}),
    ("thumbs_up", "Thumbs up", {"curl": [_FIST] * 4, "spread": 0.6, "thumb": (80, -10, 0)}),
    ("ok", "OK", {"curl": [(40, 70, 50), (5, 5, 5), (8, 8, 5), (10, 10, 5)], "spread": 1.4,
                  "thumb": (22, 30, 25)}),
]
HAND_SHAPE_NAMES = [k for k, _, _ in HAND_SHAPES]
# What the prompt says of each shape. Fingers are ~70 px of a full-length
# picture, and the skeleton alone lost a peace sign and a thumbs up on the
# 5090 (2026-09-25); FLUX reads the words. A relaxed hand is not mentioned.
HAND_WORDS = {"open": "open with the fingers spread", "fist": "clenched in a fist",
              "grab": "curled as if gripping something", "point": "pointing with the index "
              "finger", "peace": "making a peace sign with two fingers",
              "thumbs_up": "giving a thumbs up", "ok": "making an OK sign"}


def hands_text(points, hands):
    """A sentence for the prompt about the hands that are seen and shaped:
    "Right hand making a peace sign, left hand clenched in a fist." or ""."""
    hands = clean_hands(hands)
    if not hands or not points:
        return ""
    parts = []
    for side in ("right", "left"):
        wrist, elbow = HAND_OF[side]
        words = HAND_WORDS.get(hands[side]["shape"])
        if words and wrist < len(points) and points[wrist] and points[elbow]:
            parts.append("%s hand %s" % (side, words))
    if not parts:
        return ""
    text = ", ".join(parts)
    return text[0].upper() + text[1:] + "."
DEFAULT_HANDS = {"right": {"shape": "relaxed", "back": False},
                 "left": {"shape": "relaxed", "back": False}}


def hand_shape(name):
    """A shape's 21 points in the hand's own frame."""
    spec = dict((k, s) for k, _, s in HAND_SHAPES).get(name) or HAND_SHAPES[0][2]
    pts = [(0.0, 0.0)]
    (tx, ty), bones = THUMB
    angle, bends = spec["thumb"][0], (0,) + tuple(spec["thumb"][1:])
    x, y = tx, ty
    pts.append((x, y))
    for bone, bend in zip(bones, bends):
        angle -= bend                        # a thumb curls across the palm
        a = math.radians(angle)
        x, y = x + bone * math.sin(a), y + bone * math.cos(a)
        pts.append((x, y))
    for ((kx, ky), aim, bones), curl in zip(FINGERS, spec["curl"]):
        a = math.radians(aim * spec["spread"])
        ux, uy = math.sin(a), math.cos(a)
        x, y, bend = kx, ky, 0
        pts.append((x, y))
        for bone, c in zip(bones, curl):
            bend += c                        # a finger curls toward the viewer
            k = bone * math.cos(math.radians(bend))
            x, y = x + ux * k, y + uy * k
            pts.append((x, y))
    return pts


def clean_hands(hands):
    """Anything -> {"right": {"shape", "back"}, "left": ...}, or None."""
    if not isinstance(hands, dict):
        return None
    out = {}
    for side in HAND_OF:
        h = hands.get(side) if isinstance(hands.get(side), dict) else {}
        out[side] = {"shape": h.get("shape") if h.get("shape") in HAND_SHAPE_NAMES
                     else "relaxed", "back": bool(h.get("back"))}
    return out


def mirror_hands(hands):
    return None if not hands else {"right": dict(hands["left"]), "left": dict(hands["right"])}


def hand_points(points, width, height, hands):
    """{side: 21 points as fractions of the frame} for each hand whose wrist
    and elbow are seen. The hand carries on from the forearm, HAND_LENGTH of
    it long; its palm faces the viewer unless `back`, with the thumb on the
    outside, as a person facing the viewer holds an open hand."""
    out = {}
    for side, spec in (clean_hands(hands) or {}).items():
        wi, ei = HAND_OF[side]
        w, e = points[wi], points[ei]
        if not (w and e):
            continue
        wx, wy, ex, ey = w[0] * width, w[1] * height, e[0] * width, e[1] * height
        length = math.hypot(wx - ex, wy - ey)
        if length < 1e-6:
            continue
        dx, dy = (wx - ex) / length, (wy - ey) / length
        px, py = (-dy, dx) if side == "right" else (dy, -dx)
        if spec["back"]:
            px, py = -px, -py
        s = length * HAND_LENGTH
        out[side] = [((wx + (u * px + v * dx) * s) / width, (wy + (u * py + v * dy) * s) / height)
                     for u, v in hand_shape(spec["shape"])]
    return out


def size_for(width, height, edge=RENDER_EDGE):
    k = edge / max(width, height)
    return max(64, int(round(width * k))), max(64, int(round(height * k)))


def render(points, width, height, hands=None):
    """The OpenPose picture of `points` at width x height -> PNG bytes, with
    DWPose's hands when `hands` says what shape they are in."""
    return render_figures([{"points": points, "hands": hands}], width, height)


def render_figures(figures, width, height):
    """Several people in one OpenPose picture, drawn in the order given (far
    to near, so the nearer one's limbs are on top). Each figure is
    {"points", "hands"?, "face"?}: `face` is its 68 dots as fractions of the
    frame when the caller knows better than `face_points` (the Scene Builder
    turns a real head in 3D), None to work them out, [] for none. -> PNG."""
    w, h = size_for(width, height)
    buf = bytearray(w * h * 4)
    buf[3::4] = b"\xff" * (w * h)
    stick = max(3, round(max(w, h) / 170))

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

    def draw(points, hands, face):
        pix = [None if p is None else (p[0] * w, p[1] * h) for p in points]
        for n, (a, b) in enumerate(LIMBS):
            if pix[a] and pix[b]:
                blot(pix[a][0], pix[a][1], pix[b][0], pix[b][1], stick, COLOURS[n], 0.6)
        for i, p in enumerate(pix):
            if p:
                blot(p[0], p[1], p[0], p[1], stick, COLOURS[i], 1.0)
        dot = max(1.2, stick / 3)
        for x, y in face_points(points, 1, 1) if face is None else face:
            blot(x * w, y * h, x * w, y * h, dot, (255, 255, 255), 1.0)
        # DWPose's hand: each bone its own hue round the colour wheel, the
        # points blue - sized to the hand as DWPose's are (radius 4 on a ~150
        # px hand). Twice that made a fist or a thumb a blue blob the model
        # could not read.
        for hand in hand_points(points, 1, 1, hands).values():
            at = [(x * w, y * h) for x, y in hand]
            span = math.hypot(at[12][0] - at[0][0], at[12][1] - at[0][1])
            span = max(span, math.hypot(at[9][0] - at[0][0], at[9][1] - at[0][1]) * 2)
            thin, knot = max(1.0, span * 0.018), max(1.5, span * 0.034)
            for n, (a, b) in enumerate(HAND_EDGES):
                rgb = tuple(int(c * 255) for c in
                            colorsys.hsv_to_rgb(n / len(HAND_EDGES), 1, 1))
                blot(at[a][0], at[a][1], at[b][0], at[b][1], thin, rgb, 1.0)
            for x, y in at:
                blot(x, y, x, y, knot, HAND_JOINT, 1.0)

    for fig in figures:
        draw(fig["points"], fig.get("hands"), fig.get("face"))
    return studio_icons.png(bytes(buf), w, h)


def save(points, width, height, folder, hands=None):
    """The picture of this pose at this size, under `folder`, named by what
    it is, so the same pose is drawn once. -> its path. A pose without hands
    keeps the name it had before hands existed, so Generate Again finds it."""
    hands = clean_hands(hands)
    key = json.dumps([DRAWING, points, size_for(width, height)] + ([hands] if hands else []),
                     sort_keys=True)
    path = os.path.join(folder, hashlib.sha1(key.encode()).hexdigest()[:16] + ".png")
    if not os.path.exists(path):
        os.makedirs(folder, exist_ok=True)
        tmp = path + ".part"
        with open(tmp, "wb") as f:
            f.write(render(points, width, height, hands))
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
