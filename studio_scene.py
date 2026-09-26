#!/usr/bin/env python3
"""
studio_scene - the Scene Builder's engine: a small 3D stage for blocking out a
picture before the Image Studio makes it.

Not a 3D package and not a slicer, though it is laid out like one: a person
and a couple of props on a floor, one camera, and the frame that camera sees.
That frame goes to the Image Studio as a `source` reference (image to image),
and the words written on each object go into the prompt as written. The
pieces:

- **The rig** (`JOINTS`): a mannequin of fifteen joints posed by forward
  kinematics from a handful of named controls (`CONTROLS`) - head, torso, and
  each hand and foot - and `POSES`, presets of those controls. Everything
  stands on its floor: an object's lowest point is put at its `position` y
  (`ground`), so a crouch drops the hips and a tipped drum lies on the floor.
- **Props** (`ASSETS`): a box and a cylinder, sized by scale in metres. What a
  prop *is* - a crate, a workbench, a gas cylinder - is its name and its
  description; the shape only holds its place in the frame.
- **The room** (`new_room`): the floor, and optionally four walls around the
  origin. Either can wear a picture - one the Image Studio makes from a few
  words (`texture_settings`), or a PNG from disk - repeated every `size`
  metres and drawn in perspective (`TexMap`). The room is the backdrop: it is
  drawn before every object, so a person always stands in front of a wall.
- **A person's look** (`look`, `character`): the Image Studio's character
  creator slots and sliders, per person - who, body, face, hair, expression,
  clothes, accessories - or a character's look copied on. It is said in that
  person's line, so two people in one picture each keep their own. The
  mannequin is built to it too (`body_shape`: weight, muscle, height) and
  wears its clothes, hat, glasses and hair (`outfit`), since the frame is what the
  picture copies.
- **The camera** orbits a target: yaw, pitch, distance and a lens in mm on a
  full-frame diagonal, so 35mm means what it does on a camera whatever the
  frame's shape (`Camera`).
- **The render** is one list of shaded polygons, painter's order, near-plane
  clipped (`render`). The window draws it on a Tk canvas; `png()` rasterises
  the same list at the generation size, which is how the viewport shows the
  exact frame the picture is made from.
- **The words** (`scene_text`): the scene's details, then each object in the
  frame as "Name (where it is in the frame, which way a person faces): a
  person's look. The description, verbatim", then the camera. An object outside the frame is
  left out and said so (`Words.notes`).

No tkinter here; `studio_scene_ui.py` is the window. Stdlib only.
"""

import copy
import hashlib
import json
import math
import os
import re
import struct
import zlib

import studio_icons

VERSION = 1
FULL_FRAME_DIAGONAL = 43.27    # mm; a lens is read against a full-frame sensor
NEAR = 0.05                    # m; the camera's near plane
FLOOR_REACH = 30.0             # m from the origin the floor is drawn to
TILE = 0.3                     # m; a prop's faces are cut to about this, for sorting
SKY = (201, 204, 209)
FLOOR = (143, 138, 132)
WALL = (184, 178, 170)
LIGHT = (-0.45, 0.8, 0.55)     # the one light, from above, front left
AMBIENT = 0.55

# Frames the generation is made at: FLUX and Z-Image sizes, all /16.
FRAMES = [
    ("square", "Square 1024 x 1024", 1024, 1024),
    ("portrait", "Portrait 896 x 1152", 896, 1152),
    ("landscape", "Landscape 1344 x 768", 1344, 768),
]
FRAME_SIZES = {k: (w, h) for k, _, w, h in FRAMES}
LENSES = (24, 35, 50, 85)
REDRAW = 0.7                   # denoise: how far the picture may move from the blockout

COLOURS = [                    # (hex, name) offered for any object
    ("#c9b8a6", "mannequin"), ("#e8c07a", "yellow"), ("#d9774b", "orange"),
    ("#b8423a", "red"), ("#4f7fb5", "blue"), ("#5f8f5a", "green"),
    ("#6e6e72", "grey"), ("#f2f2ef", "white"), ("#2c2c30", "black"),
    ("#8a6a4a", "wood"),
]


# ==================================================================== maths

def add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def mul(a, k):
    return (a[0] * k, a[1] * k, a[2] * k)


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def norm(a):
    n = math.sqrt(dot(a, a)) or 1.0
    return (a[0] / n, a[1] / n, a[2] / n)


def mat_mul(a, b):
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3))
                 for i in range(3))


def apply(m, v):
    return (m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
            m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
            m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2])


def column(m, i):
    return (m[0][i], m[1][i], m[2][i])


def euler(yaw=0.0, pitch=0.0, roll=0.0):
    """Ry(yaw) . Rx(pitch) . Rz(roll), degrees: roll first, yaw last. Y is up
    and a person faces +Z, so their left is +X."""
    y, p, r = math.radians(yaw), math.radians(pitch), math.radians(roll)
    cy, sy, cp, sp, cr, sr = (math.cos(y), math.sin(y), math.cos(p), math.sin(p),
                              math.cos(r), math.sin(r))
    ry = ((cy, 0, sy), (0, 1, 0), (-sy, 0, cy))
    rx = ((1, 0, 0), (0, cp, -sp), (0, sp, cp))
    rz = ((cr, -sr, 0), (sr, cr, 0), (0, 0, 1))
    return mat_mul(ry, mat_mul(rx, rz))


IDENTITY = euler()


def newell(pts):
    """A polygon's normal by Newell's method: sound for any planar polygon."""
    n = [0.0, 0.0, 0.0]
    for i, a in enumerate(pts):
        b = pts[(i + 1) % len(pts)]
        n[0] += (a[1] - b[1]) * (a[2] + b[2])
        n[1] += (a[2] - b[2]) * (a[0] + b[0])
        n[2] += (a[0] - b[0]) * (a[1] + b[1])
    return norm(tuple(n))


def centroid(pts):
    k = 1.0 / len(pts)
    return (sum(p[0] for p in pts) * k, sum(p[1] for p in pts) * k,
            sum(p[2] for p in pts) * k)


# ==================================================================== meshes
# A piece is a convex solid: a list of faces, each a list of points. `outward`
# turns every face's winding to face away from the piece's middle, so back
# faces can be culled with one rule however a piece was built.

def outward(faces):
    mid = centroid([p for f in faces for p in f])
    out = []
    for f in faces:
        if dot(newell(f), sub(centroid(f), mid)) < 0:
            f = f[::-1]
        out.append(f)
    return out


def prism(p0, p1, side, r0, r1, n=8):
    """A tapered prism from p0 to p1: an ellipse of radii r0 = (across, deep)
    at p0 and r1 at p1. `side` is the direction `across` runs."""
    d = norm(sub(p1, p0))
    a = sub(side, mul(d, dot(side, d)))           # `side`, square to the axis
    if dot(a, a) < 1e-8:
        a = cross(d, (0, 0, 1) if abs(d[2]) < 0.9 else (1, 0, 0))
    a = norm(a)
    b = cross(d, a)
    rings = []
    for p, (ra, rb) in ((p0, r0), (p1, r1)):
        ring = []
        for i in range(n):
            t = 2 * math.pi * (i + 0.5) / n
            ring.append(add(p, add(mul(a, ra * math.cos(t)), mul(b, rb * math.sin(t)))))
        rings.append(ring)
    lo, hi = rings
    faces = [[lo[i], lo[(i + 1) % n], hi[(i + 1) % n], hi[i]] for i in range(n)]
    faces += [lo[::-1], hi]
    return outward(faces)


def ellipsoid(c, m, radii, seg=8, rings=6):
    """An ellipsoid at `c`, axes the columns of `m`, of `radii`."""
    pts = []
    for j in range(rings + 1):
        v = math.pi * j / rings
        row = []
        for i in range(seg):
            u = 2 * math.pi * i / seg
            local = (radii[0] * math.sin(v) * math.cos(u), radii[1] * math.cos(v),
                     radii[2] * math.sin(v) * math.sin(u))
            row.append(add(c, apply(m, local)))
        pts.append(row)
    faces = []
    for j in range(rings):
        for i in range(seg):
            a, b = pts[j][i], pts[j][(i + 1) % seg]
            c2, d = pts[j + 1][(i + 1) % seg], pts[j + 1][i]
            f = [a, b, c2, d] if 0 < j < rings - 1 else ([a, c2, d] if j == 0 else [a, b, d])
            faces.append(f)
    return outward(faces)


def tiles(face, size=TILE):
    """A quad cut into tiles of about `size` m. Painter's order sorts by a
    face's middle, and the middle of a long wall is nearer the camera than a
    person standing by its far end: whole, the wall was drawn over them."""
    if len(face) != 4:
        return [face]
    a, b, c, d = face
    span = lambda p, q: math.sqrt(dot(sub(q, p), sub(q, p)))    # noqa: E731
    nu = max(1, min(12, int(math.ceil(max(span(a, b), span(d, c)) / size))))
    nv = max(1, min(12, int(math.ceil(max(span(a, d), span(b, c)) / size))))
    if nu == nv == 1:
        return [face]

    def at(u, v):
        top = add(a, mul(sub(b, a), u))
        bottom = add(d, mul(sub(c, d), u))
        return add(top, mul(sub(bottom, top), v))
    return [[at(i / nu, j / nv), at((i + 1) / nu, j / nv), at((i + 1) / nu, (j + 1) / nv),
             at(i / nu, (j + 1) / nv)] for i in range(nu) for j in range(nv)]


def box(lo, hi):
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
         (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    idx = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (3, 2, 6, 7), (0, 3, 7, 4), (1, 2, 6, 5)]
    return outward([[v[i] for i in f] for f in idx])


def cylinder(seg=16):
    """A unit cylinder standing on the floor: radius 0.5, height 1."""
    return prism((0, 0, 0), (0, 1, 0), (1, 0, 0), (0.5, 0.5), (0.5, 0.5), n=seg)


# ==================================================================== the rig
# (joint, parent, offset from the parent in its frame, m). Rest pose: standing,
# arms down, facing +Z, the pelvis at the origin.
JOINTS = [
    ("pelvis", None, (0, 0, 0)),
    ("spine", "pelvis", (0, 0.10, 0)),
    ("chest", "spine", (0, 0.24, 0)),
    ("neck", "chest", (0, 0.22, 0)),
    ("head", "neck", (0, 0.08, 0)),
    ("shoulder_l", "chest", (0.18, 0.17, 0)),
    ("elbow_l", "shoulder_l", (0, -0.29, 0)),
    ("wrist_l", "elbow_l", (0, -0.25, 0)),
    ("shoulder_r", "chest", (-0.18, 0.17, 0)),
    ("elbow_r", "shoulder_r", (0, -0.29, 0)),
    ("wrist_r", "elbow_r", (0, -0.25, 0)),
    ("hip_l", "pelvis", (0.10, -0.02, 0)),
    ("knee_l", "hip_l", (0, -0.44, 0)),
    ("ankle_l", "knee_l", (0, -0.42, 0)),
    ("hip_r", "pelvis", (-0.10, -0.02, 0)),
    ("knee_r", "hip_r", (0, -0.44, 0)),
    ("ankle_r", "knee_r", (0, -0.42, 0)),
]

# The easy controls: (part, key, label, lo, hi), degrees. A part is what a
# click on the mannequin selects; the window shows that part's controls.
PARTS = [("body", "Body"), ("head", "Head"), ("hand_l", "Left hand"),
         ("hand_r", "Right hand"), ("foot_l", "Left foot"), ("foot_r", "Right foot")]
PART_NAMES = dict(PARTS)
CONTROLS = [
    ("body", "bend", "Bend forward", -30, 90),
    ("body", "twist", "Twist (to their left +)", -60, 60),
    ("body", "lean", "Lean (to their left +)", -30, 30),
    ("head", "head_turn", "Turn (to their left +)", -80, 80),
    ("head", "head_nod", "Look down (+) / up (-)", -45, 60),
    ("head", "head_tilt", "Tilt", -35, 35),
]
for _side, _name in (("l", "left"), ("r", "right")):
    CONTROLS += [
        ("hand_" + _side, "arm_%s_raise" % _side, "Raise forward", -60, 180),
        ("hand_" + _side, "arm_%s_out" % _side, "Out to the side", -10, 100),
        ("hand_" + _side, "arm_%s_bend" % _side, "Bend elbow", 0, 150),
        ("foot_" + _side, "leg_%s_step" % _side, "Step forward", -45, 120),
        ("foot_" + _side, "leg_%s_out" % _side, "Out to the side", -10, 50),
        ("foot_" + _side, "leg_%s_bend" % _side, "Bend knee", 0, 150),
    ]
CONTROL_KEYS = [c[1] for c in CONTROLS]
CONTROL_RANGE = {c[1]: (c[3], c[4]) for c in CONTROLS}

POSES = [
    ("standing", "Standing", {"arm_l_out": 6, "arm_r_out": 6, "arm_l_bend": 8,
                              "arm_r_bend": 8}),
    ("walking", "Walking", {"leg_l_step": 22, "leg_r_step": -18, "leg_r_bend": 25,
                            "arm_l_raise": -18, "arm_r_raise": 22, "arm_l_bend": 15,
                            "arm_r_bend": 20, "arm_l_out": 5, "arm_r_out": 5}),
    ("reaching", "Reaching up", {"arm_r_raise": 165, "arm_r_out": 10, "arm_l_out": 8,
                                 "arm_l_bend": 10, "head_nod": -30}),
    ("pointing", "Pointing", {"arm_r_raise": 85, "arm_l_out": 6, "arm_l_bend": 10}),
    ("carrying", "Carrying", {"arm_l_raise": 35, "arm_r_raise": 35, "arm_l_bend": 75,
                              "arm_r_bend": 75, "arm_l_out": 12, "arm_r_out": 12}),
    ("working", "Working at a bench", {"bend": 22, "head_nod": 30, "arm_l_raise": 45,
                                       "arm_r_raise": 50, "arm_l_bend": 65,
                                       "arm_r_bend": 60, "arm_l_out": 10,
                                       "arm_r_out": 10}),
    ("crouching", "Crouching", {"leg_l_step": 105, "leg_r_step": 100, "leg_l_bend": 130,
                                "leg_r_bend": 130, "leg_l_out": 12, "leg_r_out": 12,
                                "bend": 35, "arm_l_raise": 40, "arm_r_raise": 40,
                                "arm_l_bend": 40, "arm_r_bend": 40, "head_nod": -15}),
    ("kneeling", "Kneeling on one knee", {"leg_l_step": 90, "leg_l_bend": 90,
                                          "leg_r_bend": 90, "arm_l_raise": 30,
                                          "arm_l_bend": 60, "arm_r_out": 6,
                                          "arm_r_bend": 10}),
    ("sitting", "Sitting", {"leg_l_step": 90, "leg_r_step": 90, "leg_l_bend": 90,
                            "leg_r_bend": 90, "arm_l_raise": 30, "arm_r_raise": 30,
                            "arm_l_bend": 50, "arm_r_bend": 50}),
]
POSE_NAMES = {k: label for k, label, _ in POSES}
POSE_VALUES = {k: v for k, _, v in POSES}


def pose_controls(preset):
    """Every control at its value in `preset`, 0 where the preset is silent."""
    values = POSE_VALUES.get(preset, {})
    return {k: float(values.get(k, 0)) for k in CONTROL_KEYS}


def joint_rotations(c):
    """The controls -> each joint's local rotation."""
    g = lambda k: float(c.get(k, 0) or 0)                 # noqa: E731
    rot = {
        "spine": euler(g("twist") / 2, g("bend") / 2, -g("lean") / 2),
        "chest": euler(g("twist") / 2, g("bend") / 2, -g("lean") / 2),
        "neck": euler(g("head_turn") / 3, g("head_nod") / 3, -g("head_tilt") / 3),
        "head": euler(g("head_turn") * 2 / 3, g("head_nod") * 2 / 3,
                      -g("head_tilt") * 2 / 3),
    }
    for side, sign in (("l", 1), ("r", -1)):
        rot["shoulder_" + side] = euler(0, -g("arm_%s_raise" % side),
                                        sign * g("arm_%s_out" % side))
        rot["elbow_" + side] = euler(0, -g("arm_%s_bend" % side), 0)
        rot["hip_" + side] = euler(0, -g("leg_%s_step" % side),
                                   sign * g("leg_%s_out" % side))
        rot["knee_" + side] = euler(0, g("leg_%s_bend" % side), 0)
    return rot


def skeleton(controls, root=IDENTITY, shape=None):
    """Forward kinematics: {joint: (position, world rotation)} with the
    pelvis at the origin and the whole rig turned by `root`. `shape`
    (`body_shape`) sets the shoulders and hips apart."""
    rot = joint_rotations(controls)
    shape = shape or REST_SHAPE
    wide = {"shoulder": shape["shoulders"], "hip": shape["hips"]}
    out = {}
    for name, parent, offset in JOINTS:
        local = rot.get(name, IDENTITY)
        k = wide.get(name.split("_")[0])
        if k:
            offset = (offset[0] * k, offset[1], offset[2])
        if parent is None:
            out[name] = ((0.0, 0.0, 0.0), mat_mul(root, local))
        else:
            pp, pm = out[parent]
            out[name] = (add(pp, apply(pm, offset)), mat_mul(pm, local))
    return out


# ==================================================================== the body
# A person's look sizes the mannequin: the Weight, Muscle and Height sliders
# (-3..3, as the Image Studio stores them) and a Body type word this table
# knows, as slider steps it adds. The picture is image to image from the
# frame, so a heavyset person drawn as the rest mannequin would be pulled
# thin again.
BUILDS = [
    ("athletic", {"muscle": 1.5, "weight": -0.5}), ("lean", {"weight": -1, "muscle": 0.5}),
    ("curvy", {"weight": 0.5, "curves": 2}), ("broad-shouldered", {"muscle": 1, "shoulders": 1}),
    ("stocky", {"weight": 1.5, "muscle": 1, "stature": -0.5}),
    ("lanky", {"weight": -1.5, "stature": 1}), ("petite", {"weight": -1, "stature": -1.5}),
    ("skinny", {"weight": -2}), ("slender", {"weight": -1}), ("slim", {"weight": -1}),
    ("thin", {"weight": -1.5}), ("chubby", {"weight": 1.5}), ("plump", {"weight": 1.5}),
    ("heavy", {"weight": 2}), ("heavyset", {"weight": 2}), ("overweight", {"weight": 2}),
    ("plus-size", {"weight": 2, "curves": 1}), ("fat", {"weight": 2.5}),
    ("obese", {"weight": 3}), ("muscular", {"muscle": 2}), ("bulky", {"muscle": 1.5, "weight": 1}),
    ("burly", {"muscle": 1.5, "weight": 1}), ("toned", {"muscle": 1}),
]


def body_shape(look=None):
    """A look -> the factors the mannequin is built with, 1 at rest:
    `height` (all of it), `fat` (girth everywhere), `belly`, `muscle`
    (chest, shoulders and limbs), and how far apart the `shoulders` and
    `hips` are."""
    look = look if isinstance(look, dict) else {}
    s = {k: _num(look.get(k), 0, -3, 3) for k in ("weight", "muscle", "stature")}
    s["curves"] = s["shoulders"] = 0.0
    build = str(look.get("build") or "").lower()
    for word, steps in BUILDS:
        if re.search(r"(?<![\w-])%s(?![\w-])" % re.escape(word), build):
            for k, v in steps.items():
                s[k] += v
    w, m, h = (max(-3.0, min(3.0, s[k])) for k in ("weight", "muscle", "stature"))
    fat = 1 + (0.17 if w > 0 else 0.08) * w
    muscle = 1 + 0.06 * m
    return {"height": 1 + 0.04 * h, "fat": fat, "belly": 1 + 0.1 * max(0.0, w),
            "muscle": muscle,
            "shoulders": 1 + 0.35 * (fat - 1) + 0.8 * (muscle - 1) + 0.04 * s["shoulders"],
            "hips": 1 + 0.55 * (fat - 1) + 0.05 * s["curves"]}


REST_SHAPE = {"height": 1.0, "fat": 1.0, "belly": 1.0, "muscle": 1.0, "shoulders": 1.0,
              "hips": 1.0}           # body_shape({}), which needs _num from below


# What a person wears, drawn on the mannequin from the look's Clothes words.
# Each piece of the body is a region; a garment colours the regions it
# covers, a skirt or a long coat adds a hem, boots add a shaft. The colour is
# the first colour word in the garment's words ("black leather jacket" is
# black, "leather jacket" leather), then `dark` / `light` before it.
CLOTH = [
    ("hi-vis", "#c8dc3c"), ("high-vis", "#c8dc3c"), ("hi vis", "#c8dc3c"),
    ("fluorescent", "#c8dc3c"), ("neon", "#c8dc3c"), ("black", "#27272b"),
    ("white", "#ecebe6"), ("ivory", "#e9e2cc"), ("cream", "#e9e0c6"), ("grey", "#8b8d91"),
    ("gray", "#8b8d91"), ("charcoal", "#45474c"), ("silver", "#b9bcc0"), ("navy", "#28324f"),
    ("denim", "#4d6a8f"), ("jeans", "#4d6a8f"), ("blue", "#3f6fb0"), ("teal", "#2f7f7f"),
    ("turquoise", "#3aa6a6"), ("green", "#4f7d4a"), ("olive", "#6b6b3a"),
    ("khaki", "#b3a37a"), ("chinos", "#b3a37a"), ("beige", "#cdbb9a"), ("camel", "#b08a5a"),
    ("tan", "#b48d64"), ("brown", "#6b4a33"), ("leather", "#3b2a22"),
    ("lederhosen", "#5b4030"), ("loden", "#4a5a3c"), ("red", "#b0342f"),
    ("burgundy", "#6e2233"), ("maroon", "#6e2233"), ("wine", "#6e2233"), ("pink", "#e39ab0"),
    ("orange", "#d9772f"), ("yellow", "#e2c23c"), ("mustard", "#c9a032"),
    ("purple", "#6b4a8f"), ("lavender", "#b4a4d6"), ("gold", "#c9a54a"),
    ("tuxedo", "#27272b"), ("suit", "#45474c"), ("trench", "#b3a37a"),
]
CLOTH_DEFAULT = {"top": "#8d97a3", "bottom": "#4c5566", "outerwear": "#5e564d",
                 "footwear": "#34302d", "hat": "#4a4540", "glasses": "#2e2a28",
                 "sunglasses": "#1c1c20", "goggles": "#b9c7cf", "alpine": "#4a5a3c",
                 "dirndl": "#7a2433", "apron": "#ecebe6", "blouse": "#ecebe6",
                 "accordion": "#a3262a", "stein": "#d9d3c2"}
# What the Shoes slot names, by shape: first match wins, anything else is a
# plain shoe (loafers, oxfords, brogues) on a thin dark sole.
SHOES = [
    ("heels", ("heels", "high heels", "heel", "stilettos", "stiletto", "pumps",
               "court shoes", "wedges", "wedge heels", "platform heels", "slingbacks")),
    ("boots", ("boot", "boots", "wellies", "wellingtons", "work boots")),
    ("sandals", ("sandals", "sandal", "flip-flops", "flip flops", "slides", "espadrilles",
                 "jandals", "thongs")),
    ("sneakers", ("sneakers", "sneaker", "trainers", "running shoes", "tennis shoes",
                  "high-tops", "high tops", "basketball shoes", "kicks")),
]
SOLE_LIGHT = "#dedcd6"           # a sneaker's sole, unless the words say otherwise
HEEL = 0.065                     # m a heel lifts the heel of the foot

# From the Accessories slot, on the head: (kind, the words that name it),
# first match wins, so "hard hat" is not read as a plain "hat". The kinds
# are shapes (`HAT_SHAPES`); the words say the rest.
HATS = [
    ("crown", ("flower crown", "floral crown", "flower wreath", "floral wreath",
               "crown of flowers", "wreath of flowers", "flower headband", "blumenkranz")),
    ("hard hat", ("hard hat", "hardhat", "helmet", "safety helmet")),
    ("top hat", ("top hat",)),
    ("cap", ("baseball cap", "cap", "trucker cap", "flat cap")),
    ("beanie", ("beanie", "headscarf", "head scarf", "bandana", "hijab", "turban",
                "beret", "hood", "balaclava", "headwrap", "do-rag", "durag")),
    ("alpine", ("alpine hat", "tyrolean hat", "tyrolean", "tirolean hat", "bavarian hat",
                "german hat", "trachten hat", "feathered hat", "hat with a feather",
                "loden hat", "gamsbart")),
    ("wide", ("sun hat", "sunhat", "cowboy hat", "stetson", "sombrero", "straw hat")),
    ("hat", ("hat", "fedora", "trilby", "panama", "bowler", "bucket hat", "boater")),
]
GLASSES = [
    ("sunglasses", ("sunglasses", "shades", "aviators")),
    ("goggles", ("goggles", "safety glasses", "safety specs")),
    ("glasses", ("glasses", "round glasses", "spectacles", "eyeglasses", "specs",
                 "reading glasses")),
]
# The crown as rings from the band up, (y, (across, deep)), and the brim's
# radii or None; metres in the head's frame, where the head is an ellipsoid at
# y 0.11, 0.115 m tall: the band sits above the eyes. A dome is three rings.
HAT_SHAPES = {
    "hard hat": ([(0.14, (0.102, 0.12)), (0.2, (0.096, 0.112)), (0.245, (0.07, 0.083)),
                  (0.262, (0.03, 0.036))], (0.118, 0.142)),
    "top hat": ([(0.15, (0.092, 0.107)), (0.36, (0.094, 0.109))], (0.13, 0.145)),
    "cap": ([(0.145, (0.092, 0.107)), (0.2, (0.084, 0.098)), (0.235, (0.045, 0.052))], None),
    "beanie": ([(0.14, (0.098, 0.114)), (0.2, (0.088, 0.102)), (0.24, (0.05, 0.058)),
                (0.252, (0.02, 0.024))], None),
    "wide": ([(0.15, (0.093, 0.108)), (0.25, (0.078, 0.09))], (0.2, 0.21)),
    "hat": ([(0.15, (0.093, 0.108)), (0.25, (0.076, 0.088))], (0.15, 0.165)),
    "alpine": ([(0.15, (0.093, 0.108)), (0.235, (0.078, 0.09)), (0.27, (0.045, 0.055))],
               (0.128, 0.143)),
}
# The flowers of a flower crown, round the head in turn, unless a colour is said.
CROWN_FLOWERS = ("#e39ab0", "#ecebe6", "#e2c23c", "#b0342f", "#b4a4d6", "#d9772f")
CROWN_LEAVES = "#4f7d4a"
# Things carried, from the Accessories slot: (kind, the words that name it).
# The words are sent as written; this only puts something in the hands.
HELD = [
    ("accordion", ("accordion", "accordian", "squeezebox", "concertina")),
    ("stein", ("beer stein", "beer steins", "stein", "steins", "beer mug", "beer mugs",
               "tankard", "tankards", "masskrug", "maßkrug", "maß", "pint of beer",
               "glass of beer", "beer glass", "beer glasses")),
]
BOTH_HANDS = ("steins", "mugs", "tankards", "glasses", "two", "both hands", "pair", "each hand")
TORSO = ("chest", "belly")
SLEEVES = ("upper_arm", "forearm")
LEGS = ("hips", "thigh", "shin")


def _said(text, *words):
    return any(re.search(r"(?<![\w-])%s(?![\w-])" % re.escape(w), text) for w in words)


def _near(text, word):
    """The words just before `word` in `text` ("a pink apron" -> "a pink
    apron"), for what colour one piece of a garment is."""
    m = re.search(r"((?:[\w-]+\s+){0,2})%s" % re.escape(word), text)
    return m.group(0) if m else ""


def _has_colour(text):
    return any(re.search(r"(?<![\w-])%s" % re.escape(w), text) for w, _ in CLOTH)


def cloth_colour(text, slot):
    """A garment's words -> its colour, (r, g, b)."""
    found = []
    for word, hexc in CLOTH:
        m = re.search(r"(?<![\w-])%s" % re.escape(word), text)
        if m:
            found.append((m.start(), -len(word), hexc))
    rgb = hex_rgb(min(found)[2] if found else CLOTH_DEFAULT[slot])
    if found:
        before = text[:min(found)[0]].split()
        if before and before[-1] in ("dark", "deep"):
            rgb = tuple(int(c * 0.6) for c in rgb)
        elif before and before[-1] in ("light", "pale", "pastel"):
            rgb = tuple(int(c + (255 - c) * 0.45) for c in rgb)
    return rgb


def _hem(text, default):
    """How far down the legs a skirt or coat reaches: 1 is the knee, 2 the
    ankle."""
    if _said(text, "gown", "maxi", "floor-length", "long"):
        return 1.95
    if _said(text, "midi"):
        return 1.4
    if _said(text, "mini"):
        return 0.55
    return default


def outfit(look=None):
    """The look's Clothes -> {"regions": {region: rgb}, "hems": [(reach, rgb,
    outer)], "boots": (height m, rgb) or None, "apron", "braces": rgb or None,
    "hat", "glasses", "hair", "shoes", "held": {kind: (sides, rgb)}}. Empty
    for a person with no clothes said, who stays the plain mannequin."""
    look = look if isinstance(look, dict) else {}
    said = {k: str(look.get(k) or "").strip().lower()
            for k in ("top", "bottom", "outerwear", "footwear")}
    regions, hems, boots, apron, braces = {}, [], None, None, None
    top = said["top"]
    dress = False
    if top:
        rgb = cloth_colour(top, "top")
        if _said(top, "dress", "gown", "frock", "sundress", "dirndl", "dirndls"):
            dress = True
            dirndl = _said(top, "dirndl", "dirndls")
            if dirndl:                    # "a green dirndl with a pink apron" is green
                own = _near(top, "dirndl")
                rgb = cloth_colour(own if _has_colour(own) else "", "dirndl")
            cover = TORSO + ("hips",)
            if _said(top, "long-sleeve", "long-sleeved", "long sleeve", "long sleeves"):
                cover += SLEEVES
            elif _said(top, "short-sleeve", "short-sleeved", "short sleeve", "t-shirt"):
                cover += ("upper_arm",)
            elif dirndl:                  # the blouse's puffed sleeves, white
                blouse = _near(top, "blouse")
                regions["upper_arm"] = cloth_colour(
                    blouse if _has_colour(blouse) else "", "blouse")
            hems.append((_hem(top, 1.35 if dirndl else 1.0), rgb, False))
            if dirndl or _said(top, "apron"):
                bib = _near(top, "apron")
                apron = cloth_colour(bib if _has_colour(bib) else "", "apron")
        elif _said(top, "suit", "tuxedo", "jumpsuit", "overalls", "boilersuit",
                   "coverall", "coveralls", "onesie"):
            cover = TORSO + SLEEVES + LEGS
        elif _said(top, "tank", "vest", "camisole", "cami", "sleeveless", "halter",
                   "tube top", "bikini", "bra"):
            cover = TORSO
        elif _said(top, "t-shirt", "tshirt", "tee", "polo", "short-sleeve",
                   "short-sleeved", "short sleeve"):
            cover = TORSO + ("upper_arm",)
        else:
            cover = TORSO + SLEEVES
        if _said(top, "turtleneck", "polo neck", "roll neck"):
            cover += ("neck",)
        regions.update(dict.fromkeys(cover, rgb))
    bottom = said["bottom"]
    if bottom:
        rgb = cloth_colour(bottom, "bottom")
        if _said(bottom, "skirt", "kilt", "sarong"):
            cover = ("hips",)
            hems.append((_hem(bottom, 0.9), rgb, False))
        elif _said(bottom, "shorts", "briefs", "trunks", "boxers", "swimsuit",
                   "lederhosen"):
            cover = ("hips", "thigh")
        else:
            cover = LEGS
        if dress:                         # under a dress, only the legs show
            cover = tuple(c for c in cover if c != "hips")
        regions.update(dict.fromkeys(cover, rgb))
        if _said(bottom, "lederhosen", "suspenders", "braces", "dungarees") and not dress:
            braces = rgb
    outer = said["outerwear"]
    if outer:
        rgb = cloth_colour(outer, "outerwear")
        sleeveless = _said(outer, "vest", "gilet", "waistcoat", "tabard", "poncho")
        regions.update(dict.fromkeys(TORSO + (() if sleeveless else SLEEVES), rgb))
        if _said(outer, "trench", "overcoat", "raincoat", "parka", "duster", "lab coat",
                 "long coat", "robe", "cloak"):
            hems.append((_hem(outer, 1.05), rgb, True))
            regions["hips"] = rgb
        elif _said(outer, "coat"):
            hems.append((_hem(outer, 0.6), rgb, True))
            regions["hips"] = rgb
    feet = said["footwear"]
    shoes = None
    if feet and not _said(feet, "barefoot", "bare feet", "none"):
        rgb = cloth_colour(feet, "footwear")
        kind = next((k for k, words in SHOES if _said(feet, *words)), "shoes")
        shoes = (kind, rgb)
        if kind != "sandals":             # a sandal shows the foot
            regions["foot"] = rgb
        if kind == "boots":
            boots = (0.09 if _said(feet, "ankle") else 0.24, rgb)
    hat = glasses = None
    held = {}
    for item in (str(look.get("accessories") or "").lower().split(",")):
        item = item.strip()
        kind = next((k for k, words in HELD if _said(item, *words)), None)
        if kind and kind not in held:
            sides = ("l", "r") if kind == "stein" and _said(item, *BOTH_HANDS) else ("r",)
            held[kind] = (sides, cloth_colour(item if _has_colour(item) else "", kind))
            continue                      # "a red accordion" is not a red hat
        kind = next((k for k, words in HATS if _said(item, *words)), None)
        if kind and not hat:
            rgb = cloth_colour(item, "hat")
            if kind == "hard hat" and not _has_colour(item):
                rgb = hex_rgb("#e2c23c")          # a hard hat unsaid is site yellow
            elif kind == "alpine" and not _has_colour(item):
                rgb = hex_rgb(CLOTH_DEFAULT["alpine"])    # loden green
            elif kind == "crown" and not _has_colour(item):
                rgb = None                                # flowers of every colour
            hat = (kind, rgb)
        kind = next((k for k, words in GLASSES if _said(item, *words)), None)
        if kind and not glasses:
            glasses = (kind, cloth_colour(item, kind))
    return {"regions": regions, "hems": hems, "boots": boots, "hat": hat,
            "glasses": glasses, "hair": hairdo(look), "shoes": shoes, "apron": apron,
            "braces": braces, "held": held}


# Hair, from the look's Hair section: the colour's words, and the style's
# for what shape to build. (words, rgb), matched as the clothes are: the
# first said wins, the longer of two at one place ("dark brown" not "brown").
HAIR_COLOURS = [
    ("black", "#1f1b19"), ("jet black", "#141212"), ("dark brown", "#3b2a20"),
    ("brunette", "#4a3325"), ("brown", "#5a3e2b"), ("light brown", "#8a6848"),
    ("chestnut", "#6b3d24"), ("auburn", "#7a3a22"), ("red", "#9a3b1f"),
    ("ginger", "#b5582a"), ("copper", "#a8552a"), ("strawberry blonde", "#c8905e"),
    ("dirty blonde", "#a88a5a"), ("blonde", "#d6b77a"), ("blond", "#d6b77a"),
    ("golden", "#d2aa5c"), ("platinum blonde", "#e6dcc0"), ("platinum", "#e6dcc0"),
    ("grey", "#9a9a98"), ("gray", "#9a9a98"), ("salt and pepper", "#6e6c6a"),
    ("silver", "#c0c0c2"), ("white", "#e8e6e0"), ("pink", "#e38fb0"),
    ("blue", "#4f7fc0"), ("purple", "#7a4fa0"), ("lilac", "#b49ad0"), ("green", "#4f9a6a"),
    ("teal", "#2f8f8f"),
]
HAIR_DEFAULT = "#3b2a20"
# How far hair falls, in the head's frame (the jaw is near y 0, the
# shoulders -0.13): the first length said wins.
HAIR_LENGTHS = [
    ("very long", -0.45), ("waist-length", -0.45), ("long", -0.3),
    ("shoulder-length", -0.12), ("shoulder length", -0.12), ("medium", -0.08),
    ("chin-length", 0.02), ("chin length", 0.02), ("bob", 0.02), ("lob", -0.06),
]


def hairdo(look=None):
    """The look's Hair -> {"rgb", "cap" (thickness m over the skull, 0 for
    none), "fall" (how low it hangs, or None), "volume", "tie" (bun,
    ponytail, braids or None)}, or None for bald or no hair said."""
    look = look if isinstance(look, dict) else {}
    colour = str(look.get("hair") or "").strip().lower()
    style = str(look.get("hair_style") or "").strip().lower()
    if not (colour or style) or _said(style, "bald", "shaved head", "clean-shaven head"):
        return None
    found = []
    for word, hexc in HAIR_COLOURS:
        m = re.search(r"(?<![\w-])%s" % re.escape(word), colour + " | " + style)
        if m:
            found.append((m.start(), -len(word), hexc))
    rgb = hex_rgb(min(found)[2] if found else HAIR_DEFAULT)
    fall = next((y for words, y in HAIR_LENGTHS if _said(style, words)), None)
    volume = (1.3 if _said(style, "curly", "coily", "kinky", "big", "voluminous")
              else 1.15 if _said(style, "wavy", "shaggy", "messy", "tousled") else 1.0)
    cap = 0.012 * volume
    if _said(style, "buzz cut", "buzzcut", "crew cut", "shaved", "cropped"):
        cap, fall = 0.004, None
    elif _said(style, "very short", "pixie cut", "pixie", "short", "slicked back",
               "undercut", "crew"):
        fall = None
    if _said(style, "afro"):
        cap, fall = 0.055, None
    tie = next((t for t, words in (("bun", ("bun", "topknot", "top knot", "updo", "chignon")),
                                   ("ponytail", ("ponytail", "pony tail")),
                                   ("braids", ("braids", "braid", "plaits", "pigtails")))
                if _said(style, *words)), None)
    if tie == "bun":
        fall = None
    if _said(style, "locs", "dreadlocks", "dreads") and fall is None:
        fall = -0.25
    return {"rgb": rgb, "cap": cap, "fall": fall, "volume": volume, "tie": tie}


def _shoe(shoes, at):
    """What a shoe adds under and over the foot, in the ankle's frame (the
    sole of the foot is at y -0.08, heel z -0.1 to toe 0.2): [(rgb, faces)].
    Everything stands on its floor, so a sole or a heel lifts the person
    by its height, as it does."""
    kind, rgb = shoes
    dark = tuple(int(c * 0.5) for c in rgb)
    slab = lambda lo, hi: outward([[at(p) for p in f] for f in box(lo, hi)])   # noqa: E731
    if kind == "heels":
        # The foot is tipped onto its toes (`toe_drop`): a thin sole under
        # the ball of the foot, and the heel post to the floor it stands on.
        toe = -0.08 - HEEL + 0.01          # the underside of the tipped toe
        floor = toe - 0.012
        return [(rgb, slab((-0.045, floor, 0.045), (0.045, toe + 0.002, 0.135))),
                (rgb, slab((-0.013, floor, -0.06), (0.013, -0.075, -0.034)))]
    if kind == "sandals":
        return [(rgb, slab((-0.05, -0.095, -0.065), (0.05, -0.078, 0.2))),
                (rgb, slab((-0.049, -0.085, 0.05), (0.049, -0.02, 0.075)))]   # the strap
    if kind == "sneakers":
        sole = hex_rgb(SOLE_LIGHT) if sum(rgb) < 600 else (200, 198, 192)
        return [(sole, slab((-0.05, -0.105, -0.068), (0.05, -0.076, 0.203)))]
    if kind == "boots":
        return [(dark, slab((-0.05, -0.1, -0.066), (0.05, -0.076, 0.2)))]
    return [(dark, slab((-0.047, -0.094, -0.062), (0.047, -0.076, 0.196)))]


def hairline(x, z):
    """How high hair starts on the head at (x, z) in its frame, z from the
    head's middle: high on the forehead, at the ears on the sides, low at
    the nape."""
    return 0.11 + 0.055 * (z / (math.hypot(x, z) or 1.0))


def _scalp(at, th, rings=4, n=14):
    """Hair over the skull, `th` thick: from a hairline high on the forehead
    and low at the nape up to the crown, lofted over the head's ellipsoid
    (centre (0, 0.11, 0.01), radii 0.085, 0.115, 0.1) in its frame."""
    cy, cz, rx, ry, rz = 0.11, 0.01, 0.085, 0.115, 0.1
    grid = []
    for j in range(rings + 1):
        ring = []
        for i in range(n):
            t = 2 * math.pi * i / n                       # 0 is the front
            low = hairline(math.sin(t), math.cos(t)) - 0.012    # under the head's edge
            y = low + (0.218 - low) * j / rings
            c = math.sqrt(max(0.0, 1 - ((y - cy) / ry) ** 2))
            q = (rx * c * math.sin(t), y - cy, rz * c * math.cos(t))
            nrm = norm((q[0] / rx ** 2, q[1] / ry ** 2, q[2] / rz ** 2))
            ring.append(at((q[0] + nrm[0] * th, cy + q[1] + nrm[1] * th,
                            cz + q[2] + nrm[2] * th)))
        grid.append(ring)
    faces = [[grid[j][i], grid[j][(i + 1) % n], grid[j + 1][(i + 1) % n], grid[j + 1][i]]
             for j in range(rings) for i in range(n)]
    faces.append(grid[-1])
    return outward(faces)


def person_pieces(controls, root=IDENTITY, shape=None, dressed=None):
    """The mannequin: [(part, faces, rgb or None)], pelvis at the origin,
    built to `shape` (`body_shape`) and wearing `dressed` (`outfit`); rgb
    None is the object's own colour. Parts are the control groups, so a
    click on a hand selects the hand's controls."""
    shape = shape or REST_SHAPE
    dressed = dressed or {}
    sk = skeleton(controls, root, shape)
    P = lambda j: sk[j][0]                                 # noqa: E731
    M = lambda j: sk[j][1]                                 # noqa: E731
    X = lambda j: column(M(j), 0)                          # noqa: E731
    at = lambda j, v: add(P(j), apply(M(j), v))            # noqa: E731
    fat, mus = shape["fat"] - 1, shape["muscle"] - 1
    curve = shape["hips"] - 0.55 * fat               # the hips beyond their girth
    wear = dressed.get("regions") or {}
    # A covered region is drawn a little fuller, as cloth over it is.
    k = lambda region, f: f * (1.04 if region in wear else 1.0)    # noqa: E731
    r = lambda rad, ka, kd=None: (rad[0] * ka, rad[1] * (ka if kd is None else kd))  # noqa
    hips = k("hips", 1 + 0.9 * fat)
    belly = k("belly", 1 + 0.9 * fat)
    chest = k("chest", 1 + 0.7 * fat + mus)
    neck = k("neck", 1 + 0.5 * fat + 0.6 * mus)
    upper = k("upper_arm", 1 + 0.7 * fat + 1.3 * mus)
    fore = k("forearm", 1 + 0.5 * fat + 0.8 * mus)
    thigh = k("thigh", 1 + 0.8 * fat + 0.8 * mus)
    shin = k("shin", 1 + 0.45 * fat + 0.6 * mus)
    # The head, less what a scalp of hair covers: a big face of it sorts in
    # front of the hair over it and shows through as a pale speck.
    hair = dressed.get("hair")
    hat = dressed.get("hat")
    scalp = bool(hair and hair["cap"] and not (hat and hat[0] != "crown"))
    skull = [[at("head", p) for p in f]
             for f in ellipsoid((0, 0.11, 0.01), IDENTITY, (0.085, 0.115, 0.1))
             if not (scalp and centroid(f)[1] > hairline(centroid(f)[0], centroid(f)[2] - 0.01))]
    shoes = dressed.get("shoes")
    toe_drop = HEEL if shoes and shoes[0] == "heels" else 0.0     # up on its toes
    # (part, region, faces); a region is what clothes cover, or the rgb of a
    # piece that is only clothes (a hem, a boot shaft).
    out = [
        ("body", "hips", prism(at("pelvis", (0, -0.07, 0)), P("spine"), X("pelvis"),
                               r((0.16, 0.10), hips * curve, hips),
                               r((0.15, 0.10), belly, belly))),
        ("body", "belly", prism(P("spine"), P("chest"), X("spine"),
                                r((0.14, 0.095), belly, belly * shape["belly"]),
                                r((0.155, 0.10), chest, chest))),
        ("body", "chest", prism(P("chest"), at("chest", (0, 0.20, 0)), X("chest"),
                                r((0.175, 0.11), chest * shape["shoulders"] ** 0.5, chest),
                                r((0.14, 0.085), chest * shape["shoulders"], chest))),
        ("head", "neck", prism(P("neck"), P("head"), X("neck"), r((0.05, 0.05), neck),
                               r((0.048, 0.048), neck), 6)),
        ("head", "head", skull),
    ]
    # The nose says which way the head faces: a small wedge on the front.
    nose = [at("head", v) for v in ((-0.016, 0.07, 0.1), (0.016, 0.07, 0.1),
                                    (0.016, 0.115, 0.1), (-0.016, 0.115, 0.1),
                                    (-0.01, 0.075, 0.135), (0.01, 0.075, 0.135),
                                    (0.01, 0.1, 0.13), (-0.01, 0.1, 0.13))]
    idx = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (3, 2, 6, 7), (0, 3, 7, 4), (1, 2, 6, 5)]
    out.append(("head", "head", outward([[nose[i] for i in f] for f in idx])))
    head_box = lambda lo, hi: outward([[at("head", p) for p in f]    # noqa: E731
                                       for f in box(lo, hi)])
    if hat and hat[0] == "crown":
        # Flowers round the head where a band would sit, on a ring of leaves.
        rgb = hat[1]
        out.append(("head", hex_rgb(CROWN_LEAVES), prism(
            at("head", (0, 0.168, 0.01)), at("head", (0, 0.184, 0.01)), X("head"),
            (0.084, 0.098), (0.08, 0.094), 12)))
        for i in range(10):
            t = 2 * math.pi * i / 10
            c = at("head", (0.083 * math.sin(t), 0.18, 0.01 + 0.097 * math.cos(t)))
            out.append(("head", rgb or hex_rgb(CROWN_FLOWERS[i % len(CROWN_FLOWERS)]),
                        ellipsoid(c, M("head"), (0.022, 0.02, 0.022), 6, 4)))
    elif hat:
        kind, rgb = hat
        rings, brim = HAT_SHAPES[kind]
        for (y0, r0), (y1, r1) in zip(rings, rings[1:]):
            out.append(("head", rgb, prism(at("head", (0, y0, 0.01)),
                                           at("head", (0, y1, 0.01)), X("head"), r0, r1, 12)))
        y0 = rings[0][0]
        if brim:
            out.append(("head", rgb, prism(at("head", (0, y0 - 0.004, 0.01)),
                                           at("head", (0, y0 + 0.008, 0.01)),
                                           X("head"), brim, brim, 16)))
        if kind == "cap":                 # the visor, forward over the eyes
            out.append(("head", rgb, head_box((-0.07, 0.14, 0.07), (0.07, 0.152, 0.2))))
        if kind == "alpine":              # a cord band and the feather at its side
            out.append(("head", tuple(int(c * 0.55) for c in rgb), prism(
                at("head", (0, 0.152, 0.01)), at("head", (0, 0.172, 0.01)), X("head"),
                (0.094, 0.109), (0.092, 0.107), 12)))
            out.append(("head", (216, 212, 200), prism(
                at("head", (0.085, 0.17, -0.03)), at("head", (0.098, 0.27, -0.06)),
                X("head"), (0.01, 0.025), (0.004, 0.012), 4)))
    if hair:
        rgb, v = hair["rgb"], hair["volume"]
        hp = lambda v3: at("head", v3)                     # noqa: E731
        if scalp:                                          # under a hat, the hat
            out.append(("head", rgb, _scalp(hp, hair["cap"])))
        if hair["fall"] is not None:
            # Down the back from behind the ears, clear of the back of the torso.
            y = hair["fall"]
            out.append(("head", rgb, prism(
                hp((0, 0.13, -0.035)), hp((0, y, -0.035 - 0.1 * min(1.0, (0.13 - y) / 0.4))),
                X("head"), (0.098 * v, 0.08 * v), (0.105 * v, 0.035 * v), 10)))
        if hair["tie"] == "bun":
            out.append(("head", rgb, ellipsoid(hp((0, 0.2, -0.09)), M("head"),
                                               (0.045 * v, 0.04 * v, 0.04 * v), 8, 5)))
        elif hair["tie"] == "ponytail":
            y = hair["fall"] if hair["fall"] is not None else 0.0
            out.append(("head", rgb, prism(hp((0, 0.17, -0.1)), hp((0, y, -0.15)), X("head"),
                                           (0.03 * v, 0.03 * v), (0.018 * v, 0.018 * v), 6)))
        elif hair["tie"] == "braids":
            y = hair["fall"] if hair["fall"] is not None else -0.2
            for sx in (1, -1):
                out.append(("head", rgb, prism(hp((sx * 0.06, 0.1, -0.06)),
                                               hp((sx * 0.075, y, -0.11)), X("head"),
                                               (0.02, 0.02), (0.015, 0.015), 6)))
    if dressed.get("glasses"):
        kind, rgb = dressed["glasses"]
        lo, hi = {"glasses": (0.11, 0.136), "sunglasses": (0.1, 0.14),
                  "goggles": (0.098, 0.146)}[kind]
        if kind == "goggles":
            lenses = [((-0.075, lo, 0.1), (0.075, hi, 0.122))]
        else:
            lenses = [((0.01, lo, 0.106), (0.064, hi, 0.116)),
                      ((-0.064, lo, 0.106), (-0.01, hi, 0.116)),
                      ((-0.012, hi - 0.01, 0.108), (0.012, hi - 0.003, 0.115))]
        arms = [((0.082, hi - 0.008, 0.0), (0.09, hi - 0.002, 0.108)),
                ((-0.09, hi - 0.008, 0.0), (-0.082, hi - 0.002, 0.108))]
        out += [("head", rgb, head_box(a, b)) for a, b in lenses + arms]
    shaft = dressed.get("boots") if "shin" not in wear else None   # trousers hide it
    for side in ("l", "r"):
        hand, foot = "hand_" + side, "foot_" + side
        # A boot's shaft is the shin's lower part, not a tube over it: the
        # shin's long faces sort in front of the shaft's and show through.
        shin_end = P("ankle_" + side)
        if shaft:
            up = norm(sub(P("knee_" + side), P("ankle_" + side)))
            shin_end = add(P("ankle_" + side), mul(up, shaft[0]))
        out += [
            (hand, "upper_arm", prism(P("shoulder_" + side), P("elbow_" + side),
                                      X("shoulder_" + side), r((0.052, 0.052), upper),
                                      r((0.042, 0.042), (upper + fore) / 2))),
            (hand, "forearm", prism(P("elbow_" + side), P("wrist_" + side), X("elbow_" + side),
                                    r((0.042, 0.042), fore), r((0.033, 0.03), fore))),
            (hand, "hand", prism(P("wrist_" + side), at("wrist_" + side, (0, -0.18, 0.01)),
                                 X("wrist_" + side), (0.045, 0.022), (0.04, 0.018), 6)),
            (foot, "thigh", prism(P("hip_" + side), P("knee_" + side), X("hip_" + side),
                                  r((0.078, 0.078), thigh), r((0.056, 0.056), shin))),
            (foot, "shin", prism(P("knee_" + side), shin_end, X("knee_" + side),
                                 r((0.052, 0.052), shin), r((0.04, 0.04), shin))),
            (foot, "foot", prism(at("ankle_" + side, (0, -0.045, -0.05)),
                                 at("ankle_" + side, (0, -0.045 - toe_drop, 0.19 - toe_drop)),
                                 X("ankle_" + side),
                                 r((0.045, 0.035), k("foot", 1)),
                                 r((0.042, 0.025), k("foot", 1)), 6)),
        ]
        if shoes:
            out += [(foot, rgb, faces) for rgb, faces in _shoe(
                shoes, lambda v, s=side: at("ankle_" + s, v))]
        if shaft:
            out.append((foot, shaft[1], prism(at("ankle_" + side, (0, -0.02, 0)), shin_end,
                                              X("knee_" + side), r((0.05, 0.05), shin),
                                              r((0.052, 0.052), shin))))
    # Braces over the chest: two straps down the front, a bar across them.
    if dressed.get("braces"):
        rgb, fz = dressed["braces"], 0.1 * chest + 0.012
        for sx in (1, -1):
            out.append(("body", rgb, prism(at("spine", (sx * 0.075, -0.02, 0.095 * belly + 0.012)),
                                           at("chest", (sx * 0.085, 0.19, fz)), X("chest"),
                                           (0.018, 0.006), (0.018, 0.006), 4)))
        out.append(("body", rgb, prism(at("chest", (-0.09, 0.1, fz + 0.004)),
                                       at("chest", (0.09, 0.1, fz + 0.004)), (0, 1, 0),
                                       (0.025, 0.006), (0.025, 0.006), 4)))
    held = dressed.get("held") or {}
    if "accordion" in held:
        # Across the chest, bellows between two ends; the Carrying pose puts
        # the hands on it.
        _, rgb = held["accordion"]
        z = 0.11 * chest + 0.13
        chest_box = lambda lo, hi: outward([[at("chest", p) for p in f]   # noqa: E731
                                            for f in box(lo, hi)])
        out += [("body", rgb, chest_box((-0.23, -0.2, z - 0.1), (-0.15, 0.12, z + 0.1))),
                ("body", rgb, chest_box((0.15, -0.2, z - 0.1), (0.23, 0.12, z + 0.1))),
                ("body", (236, 235, 230), chest_box((0.155, -0.18, z + 0.1),
                                                    (0.225, 0.1, z + 0.115)))]   # keys
        for i in range(8):                # the bellows' pleats, dark and light
            x0 = -0.15 + 0.3 * i / 8
            deep = 0.085 if i % 2 else 0.07
            out.append(("body", (42, 37, 34) if i % 2 else tuple(int(c * 0.45) for c in rgb),
                        chest_box((x0, -0.18, z - deep), (x0 + 0.3 / 8, 0.1, z + deep))))
    if "stein" in held:
        sides, rgb = held["stein"]
        for side in sides:
            # Upright whatever the arm does, in front of the palm.
            c = at("wrist_" + side, (0, -0.1, 0.085))
            lo, hi = add(c, (0, -0.085, 0)), add(c, (0, 0.085, 0))
            out.append(("hand_" + side, rgb, prism(lo, hi, (1, 0, 0), (0.048, 0.048),
                                                   (0.045, 0.045), 10)))
            out.append(("hand_" + side, (246, 242, 230), prism(
                hi, add(hi, (0, 0.022, 0)), (1, 0, 0), (0.046, 0.046), (0.04, 0.04), 10)))
    if dressed.get("apron"):
        # A plate down the front of the skirt, from the waist to the knee.
        fwd = column(M("pelvis"), 2)
        knee = mul(add(P("knee_l"), P("knee_r")), 0.5)
        top_z, low_z = 0.105 * hips + 0.02, 0.08 * thigh + 0.1
        a, b = at("pelvis", (0.11, 0.02, top_z)), at("pelvis", (-0.11, 0.02, top_z))
        low = add(add(knee, mul(fwd, low_z)), (0, 0.06, 0))
        wide = mul(X("pelvis"), 0.14)
        out.append(("body", dressed["apron"], [[a, add(low, wide), sub(low, wide), b],
                                               [b, sub(low, wide), add(low, wide), a]]))
    # Hems: a skirt or a coat below the waist, a flared tube from the hips to
    # a line across both legs `reach` of the way down (1 the knee, 2 the
    # ankle), so it follows a step or a seat.
    for reach, rgb, outer in dressed.get("hems") or ():
        line = []
        for side in ("l", "r"):
            a, b = (("hip_", "knee_") if reach <= 1 else ("knee_", "ankle_"))
            t = reach if reach <= 1 else reach - 1
            line.append(add(P(a + side), mul(sub(P(b + side), P(a + side)), t)))
        mid = mul(add(line[0], line[1]), 0.5)
        apart = math.sqrt(dot(sub(line[0], line[1]), sub(line[0], line[1]))) / 2
        flare = 0.03 + 0.035 * reach + (0.02 if outer else 0)
        top = at("pelvis", (0, 0.04 if outer else 0.0, 0))
        out.append(("body", rgb, prism(
            top, mid, X("pelvis"),
            (0.165 * hips * curve + (0.015 if outer else 0.005),
             0.105 * hips + (0.015 if outer else 0.005)),
            (apart + 0.075 * thigh + flare, 0.08 * thigh + flare), 12)))
    # Each piece's colour: what it wears, else the object's own.
    return [(part, faces, region if isinstance(region, tuple) else wear.get(region))
            for part, region, faces in out]


# ==================================================================== assets
# What the library offers. A prop's mesh is a unit shape standing on the
# floor, sized by the object's scale; the person is the rig above. Blender
# can make better ones later; the scene file names an asset by id, not by
# geometry, so a scene keeps opening when the geometry improves.
ASSETS = [
    {"id": "person", "label": "Person", "kind": "person", "name": "Person",
     "colour": "#c9b8a6", "scale": [1.0, 1.0, 1.0],
     "about": "A posable mannequin, 1.8 m. Describe who they are and what they are doing."},
    {"id": "box", "label": "Box", "kind": "prop", "name": "Crate", "colour": "#8a6a4a",
     "scale": [0.6, 0.6, 0.6],
     "about": "Any boxy thing: a crate, a bench, a cabinet, a wall. Size it in metres."},
    {"id": "cylinder", "label": "Cylinder", "kind": "prop", "name": "Drum",
     "colour": "#4f7fb5", "scale": [0.6, 0.9, 0.6],
     "about": "Any round thing: a drum, a gas cylinder, a post, a pipe laid down."},
]
ASSET = {a["id"]: a for a in ASSETS}
UNIT = {"box": box((-0.5, 0, -0.5), (0.5, 1, 0.5)), "cylinder": cylinder()}


def painted_pieces(obj):
    """An object's faces in the world: [(part, faces, rgb)], standing on its
    floor (its lowest point at position y). A person is built to their
    look's body (`body_shape`) and wears its clothes (`outfit`); what is
    not clothed is the object's colour."""
    rot = euler(*obj["rotation"])
    sx, sy, sz = obj["scale"]
    own = hex_rgb(obj["colour"])
    if obj["asset"] == "person":
        look = obj.get("look") or {}
        shape = body_shape(look)
        k = sx * shape["height"]
        pieces = [(part, [[mul(p, k) for p in f] for f in faces], rgb or own)
                  for part, faces, rgb in person_pieces(obj["pose"]["controls"], rot,
                                                        shape, outfit(look))]
    else:
        faces = [[apply(rot, (p[0] * sx, p[1] * sy, p[2] * sz)) for p in f]
                 for f in UNIT[obj["asset"]]]
        pieces = [("body", [t for f in faces for t in tiles(f)], own)]
    low = min(p[1] for _, faces, _ in pieces for f in faces for p in f)
    x, y, z = obj["position"]
    shift = (x, y - low, z)
    return [(part, [[add(p, shift) for p in f] for f in faces], rgb)
            for part, faces, rgb in pieces]


def object_pieces(obj):
    """An object's faces in the world: [(part, faces)]."""
    return [(part, faces) for part, faces, _ in painted_pieces(obj)]


def bounds(obj):
    pts = [p for _, faces in object_pieces(obj) for f in faces for p in f]
    lo = tuple(min(p[i] for p in pts) for i in range(3))
    hi = tuple(max(p[i] for p in pts) for i in range(3))
    return lo, hi


# ==================================================================== scene
# The room's surfaces: (key, label, what the picture is asked to be). The
# words the user writes go in the middle, as written.
SURFACES = [
    ("floor", "Floor",
     "A seamless, tileable floor texture seen from directly above, filling the whole "
     "frame: %s. Flat and evenly lit, no perspective, no horizon, no objects, no people."),
    ("wall", "Walls",
     "A seamless, tileable wall texture seen straight on, filling the whole frame: %s. "
     "Flat and evenly lit, no perspective, no floor or ceiling, no furniture, no people."),
]
SURFACE_NAMES = {k: label for k, label, _ in SURFACES}
TEXTURE_NEGATIVE = "people, person, furniture, perspective, horizon, text, watermark"
TEXTURE_SIDE = 256             # px; a picture is kept this size for the renderer
ROOM_LIMITS = {"width": (1.0, 40.0), "depth": (1.0, 40.0), "height": (1.5, 12.0)}


def new_room():
    """The floor, and walls around the origin (off until asked for). Each
    surface: the words its picture was asked for, the picture (a PNG path, ''
    for plain) and `size`, the metres one copy of the picture covers."""
    return {"walls": False, "width": 8.0, "depth": 8.0, "height": 3.0,
            "floor": {"prompt": "", "image": "", "size": 2.0},
            "wall": {"prompt": "", "image": "", "size": 3.0}}


def new_scene(details=""):
    return {"version": VERSION, "details": details, "frame": "portrait", "redraw": REDRAW,
            "camera": {"target": [0.0, 1.0, 0.0], "yaw": 0.0, "pitch": 6.0,
                       "distance": 4.2, "lens": 35.0},
            "room": new_room(), "objects": []}


def new_object(asset_id, taken=()):
    a = ASSET[asset_id]
    names = {o.get("name") for o in taken}
    name, n = a["name"], 2
    while name in names:
        name, n = "%s %d" % (a["name"], n), n + 1
    ids = {o.get("id") for o in taken}
    oid, n = asset_id, 2
    while oid in ids:
        oid, n = "%s-%d" % (asset_id, n), n + 1
    obj = {"id": oid, "asset": asset_id, "name": name, "description": "",
           "colour": a["colour"], "position": [0.0, 0.0, 0.0],
           "rotation": [0.0, 0.0, 0.0], "scale": list(a["scale"])}
    if a["kind"] == "person":
        obj["pose"] = {"preset": "standing", "controls": pose_controls("standing")}
        obj["character"] = ""
        obj["look"] = {}
    return obj


# ==================================================================== looks
# A person's look is the Image Studio's: every slot of `studio_imagegen.LOOKS`
# and every slider, kept sparse (only what is set). Imported lazily, as
# `scenes_dir` does: the engine is heavy and the renderer needs none of it.

def clean_look(d):
    """A saved look made safe: known slots as text, sliders as whole steps
    within their span, nothing unset kept."""
    import studio_imagegen as ig
    d = d if isinstance(d, dict) else {}
    out = {}
    for k in ig.SLOTS:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            out[k] = v.strip()
    for k in ig.SLIDER_KEYS:
        step = int(round(_num(d.get(k), 0, -ig.SLIDER_SPAN, ig.SLIDER_SPAN)))
        if step:
            out[k] = step
    return out


def character_look(rec, look=None):
    """A character's look put on a person, as the form does it: every slot
    and slider the character keeps (blank where it has none, so the last
    one's beard does not stay); the expression and gaze are the picture's,
    and stay."""
    import studio_imagegen as ig
    look = dict(look or {})
    for k in ig.CHARACTER_KEYS:
        look.pop(k, None)
    look.update((rec or {}).get("looks") or {})
    return clean_look(look)


def wear_outfit(rec, look=None):
    """A clothes preset put on a person: every clothes and accessories slot
    is the preset's, blank where it has none; the body, face and hair stay."""
    import studio_imagegen as ig
    look = dict(look or {})
    for k in ig.OUTFIT_KEYS:
        look.pop(k, None)
    look.update((rec or {}).get("looks") or {})
    return clean_look(look)


def outfit_looks(look):
    """What a person wears, as a preset keeps it."""
    import studio_imagegen as ig
    return {k: look[k] for k in ig.OUTFIT_KEYS if (look or {}).get(k)}


def look_text(obj):
    """The person's look in words, as the Image Studio says a person."""
    import studio_imagegen as ig
    return ig.person_text(obj.get("look") or {})


def _num(v, default, lo=None, hi=None):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    if x != x or x in (float("inf"), float("-inf")):
        return default
    if lo is not None:
        x = max(lo, x)
    if hi is not None:
        x = min(hi, x)
    return x


def _vec(v, default, lo=None, hi=None):
    v = v if isinstance(v, (list, tuple)) and len(v) == 3 else default
    return [_num(x, d, lo, hi) for x, d in zip(v, default)]


HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


def clean_object(d, taken=()):
    """A saved object made safe: an unknown asset is None (said by the
    caller), every number a number, every control within its range."""
    if not isinstance(d, dict) or d.get("asset") not in ASSET:
        return None
    base = new_object(d["asset"], taken)
    o = dict(base)
    o["id"] = str(d.get("id") or base["id"])
    if o["id"] in {t.get("id") for t in taken}:
        o["id"] = base["id"]
    o["name"] = str(d.get("name") or base["name"])
    o["description"] = str(d.get("description") or "")
    o["colour"] = d.get("colour") if HEX.match(str(d.get("colour") or "")) else base["colour"]
    o["position"] = _vec(d.get("position"), base["position"], -100, 100)
    o["rotation"] = _vec(d.get("rotation"), base["rotation"], -360, 360)
    o["scale"] = _vec(d.get("scale"), base["scale"], 0.05, 20)
    if "pose" in base:
        pose = d.get("pose") if isinstance(d.get("pose"), dict) else {}
        preset = pose.get("preset") if pose.get("preset") in POSE_VALUES else ""
        given = pose.get("controls") if isinstance(pose.get("controls"), dict) else {}
        start = pose_controls(preset or "standing")
        o["pose"] = {"preset": preset, "controls": {
            k: _num(given.get(k), start[k], *CONTROL_RANGE[k]) for k in CONTROL_KEYS}}
        o["character"] = str(d.get("character") or "")
        o["look"] = clean_look(d.get("look"))
    return o


def clean_scene(d):
    """-> (scene, problems). A file from an older or hand-edited scene opens
    with what can be read, and says what could not."""
    problems = []
    s = new_scene()
    if not isinstance(d, dict):
        return s, ["Not a scene file."]
    s["details"] = str(d.get("details") or "")
    s["frame"] = d.get("frame") if d.get("frame") in FRAME_SIZES else s["frame"]
    s["redraw"] = _num(d.get("redraw"), REDRAW, 0.05, 1.0)
    cam = d.get("camera") if isinstance(d.get("camera"), dict) else {}
    c = s["camera"]
    c["target"] = _vec(cam.get("target"), c["target"], -100, 100)
    c["yaw"] = _num(cam.get("yaw"), c["yaw"]) % 360
    c["pitch"] = _num(cam.get("pitch"), c["pitch"], -80, 85)
    c["distance"] = _num(cam.get("distance"), c["distance"], 0.3, 80)
    c["lens"] = _num(cam.get("lens"), c["lens"], 10, 300)
    room = d.get("room") if isinstance(d.get("room"), dict) else {}
    r = s["room"]
    r["walls"] = room.get("walls") is True
    for key, (lo, hi) in ROOM_LIMITS.items():
        r[key] = _num(room.get(key), r[key], lo, hi)
    for key, label, _ in SURFACES:
        given = room.get(key) if isinstance(room.get(key), dict) else {}
        r[key]["prompt"] = str(given.get("prompt") or "")
        r[key]["image"] = str(given.get("image") or "")
        r[key]["size"] = _num(given.get("size"), r[key]["size"], 0.25, 20)
        if r[key]["image"] and not os.path.isfile(r[key]["image"]):
            problems.append("The %s picture %s is missing, so it is drawn plain."
                            % (label.lower(), os.path.basename(r[key]["image"])))
    for raw in d.get("objects") or []:
        o = clean_object(raw, s["objects"])
        if o is None:
            problems.append("An object of unknown kind %r was left out."
                            % (raw.get("asset") if isinstance(raw, dict) else raw))
        else:
            s["objects"].append(o)
    return s, problems


def save(scene, path):
    s, _ = clean_scene(scene)
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=2)
    os.replace(tmp, path)
    return s


def load(path):
    """-> (scene, problems). Raises OSError or ValueError for a file that is
    not there or not JSON; the window says which."""
    with open(path, encoding="utf-8") as f:
        return clean_scene(json.load(f))


def scenes_dir():
    import studio_imagegen as ig        # lazily: the engine is heavy, this is a path
    return os.path.join(ig.studio_dir(), "scenes")


# ================================================================= textures

class Texture:
    """A picture for a surface: its pixels as 3-byte strings, row by row, so
    a scanline is a join; `mean` is the flat colour drawn while it bakes."""

    def __init__(self, rgba, w, h):
        self.w, self.h = w, h
        self.px = [bytes(rgba[i:i + 3]) for i in range(0, w * h * 4, 4)]
        n = float(w * h)
        self.mean = tuple(int(sum(rgba[c::4]) / n) for c in range(3))
        self._shaded = {}

    def shaded(self, k):
        """Mip levels at brightness `k`: [(pixels, w, h)], each a quarter the
        width of the one before, down to one pixel. Far floor drawn from the
        full picture shimmers into moire, which the picture would copy."""
        k = round(k, 3)
        if k not in self._shaded:
            px = self.px if k == 1.0 else [bytes(min(255, int(v * k)) for v in p)
                                           for p in self.px]
            levels = [(px, self.w, self.h)]
            while levels[-1][1] > 1 or levels[-1][2] > 1:
                src, w, h = levels[-1]
                nw, nh = max(1, w // 4), max(1, h // 4)
                out = []
                for y in range(nh):
                    rows = range(y * h // nh, max(y * h // nh + 1, (y + 1) * h // nh))
                    for x in range(nw):
                        cols = range(x * w // nw, max(x * w // nw + 1, (x + 1) * w // nw))
                        cell = [src[r * w + c] for r in rows for c in cols]
                        n = len(cell)
                        out.append(bytes(sum(p[i] for p in cell) // n for i in range(3)))
                levels.append((out, nw, nh))
            self._shaded[k] = levels
        return self._shaded[k]


_TEXTURES = {}


def texture(path):
    """The picture at `path` as a Texture, or None when there is none or it
    cannot be read (the surface is drawn plain). Cached by path and mtime."""
    if not path:
        return None
    try:
        key = (path, os.path.getmtime(path))
    except OSError:
        return None
    t = _TEXTURES.get(key)
    if t is None:
        try:
            with open(path, "rb") as f:
                rgba, w, h = studio_icons.png_to_rgba(f.read())
        except (OSError, ValueError, KeyError, IndexError, zlib.error, struct.error):
            return None
        t = _TEXTURES[key] = Texture(rgba, w, h)
    return t


def import_texture(src, folder=None):
    """A picture (the Image Studio's output, or a PNG the user chose) ->
    a `TEXTURE_SIDE` copy under `<scenes>/textures/`, named by content, and its
    path. The scene keeps that copy, so clearing History does not take the
    floor with it. Raises OSError, or ValueError for what is not a PNG this can
    read (JPEG, 16-bit, interlaced)."""
    with open(src, "rb") as f:
        data = f.read()
    try:
        rgba, w, h = studio_icons.png_to_rgba(data)
    except (KeyError, IndexError, zlib.error, struct.error) as e:
        raise ValueError("not a PNG this can read (%s)" % e)
    side = TEXTURE_SIDE
    tw, th = ((side, max(1, round(side * h / w))) if w >= h else
              (max(1, round(side * w / h)), side))
    tw, th = min(tw, w), min(th, h)
    out = bytearray(tw * th * 4)
    for y in range(th):
        rows = [min(h - 1, int((y + oy) * h / th)) * w for oy in (0.25, 0.75)]
        for x in range(tw):
            cols = [min(w - 1, int((x + ox) * w / tw)) for ox in (0.25, 0.75)]
            r = g = b = 0
            for row in rows:
                for col in cols:
                    i = (row + col) * 4
                    r, g, b = r + rgba[i], g + rgba[i + 1], b + rgba[i + 2]
            o = (y * tw + x) * 4
            out[o], out[o + 1], out[o + 2], out[o + 3] = r // 4, g // 4, b // 4, 255
    small = studio_icons.png(bytes(out), tw, th)
    folder = folder or os.path.join(scenes_dir(), "textures")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "texture_%s.png" % hashlib.sha1(small).hexdigest()[:16])
    if not os.path.isfile(path):
        with open(path, "wb") as f:
            f.write(small)
    return path


def texture_settings(scene, surface, model, backend="auto"):
    """The Image Studio settings that make `surface`'s picture from its
    words: text to image, square, and none of the form's person, identities,
    LoRAs or references - a floor has no face to redraw. `scene_texture` names
    the surface, so the finished job finds its way back to the builder."""
    import studio_imagegen as ig
    words = scene["room"][surface]["prompt"].strip().rstrip(" .")
    s = ig.default_settings()
    s.update(model=model, backend=backend, scene=dict(
        (k, w) for k, _, w in SURFACES)[surface] % words, negative=TEXTURE_NEGATIVE,
        anatomy=False, width=1024, height=1024, refine=False, face_detail=False,
        upscale=None, batch=1, scene_texture=surface)
    return s


def walls(room):
    """The four walls, each (quad, origin, along): a quad facing into the
    room, its top-left corner as seen from inside, and the direction along
    it. A picture's top row runs along the wall's top."""
    hw, hd, h = room["width"] / 2, room["depth"] / 2, room["height"]
    out = []
    for origin, along, length in (((-hw, h, -hd), (1, 0, 0), 2 * hw),     # back
                                  ((hw, h, -hd), (0, 0, 1), 2 * hd),      # right
                                  ((hw, h, hd), (-1, 0, 0), 2 * hw),      # front
                                  ((-hw, h, hd), (0, 0, -1), 2 * hd)):    # left
        end = add(origin, mul(along, length))
        quad = [origin, end, (end[0], 0.0, end[2]), (origin[0], 0.0, origin[2])]
        if dot(newell(quad), (-origin[0], 0, -origin[2])) < 0:
            quad = quad[::-1]
        out.append((quad, origin, along))
    return out


# ==================================================================== camera

class Camera:
    """An orbit camera: `target`, `yaw` and `pitch` (deg; 0/0 looks along -Z
    from +Z, level), `distance` (m) and `lens` (mm, full-frame diagonal).
    `width` x `height` is the frame in pixels."""

    def __init__(self, cam, width, height):
        self.target = tuple(cam["target"])
        yaw, pitch = math.radians(cam["yaw"]), math.radians(cam["pitch"])
        d = cam["distance"]
        self.eye = add(self.target, (d * math.cos(pitch) * math.sin(yaw), d * math.sin(pitch),
                                     d * math.cos(pitch) * math.cos(yaw)))
        self.f = norm(sub(self.target, self.eye))
        self.r = norm(cross(self.f, (0, 1, 0)))
        self.u = cross(self.r, self.f)
        self.w, self.h = width, height
        self.lens = cam["lens"]
        self.k = math.hypot(width, height) * cam["lens"] / FULL_FRAME_DIAGONAL

    def to_camera(self, p):
        q = sub(p, self.eye)
        return (dot(q, self.r), dot(q, self.u), dot(q, self.f))

    def to_screen(self, c):
        return (self.w / 2 + c[0] / c[2] * self.k, self.h / 2 - c[1] / c[2] * self.k)

    def project(self, p):
        """World -> (sx, sy, depth), or None behind the near plane."""
        c = self.to_camera(p)
        if c[2] < NEAR:
            return None
        sx, sy = self.to_screen(c)
        return sx, sy, c[2]

    def ray(self, sx, sy):
        """The world direction through frame pixel (sx, sy)."""
        return norm(add(self.f, add(mul(self.r, (sx - self.w / 2) / self.k),
                                    mul(self.u, -(sy - self.h / 2) / self.k))))

    def on_floor(self, sx, sy, height=0.0):
        """Where the ray through (sx, sy) meets the floor at `height`, or None."""
        d = self.ray(sx, sy)
        if abs(d[1]) < 1e-6:
            return None
        t = (height - self.eye[1]) / d[1]
        if t <= 0:
            return None
        return add(self.eye, mul(d, t))


def clip_near(pts):
    """Sutherland-Hodgman against the near plane, in camera space."""
    out = []
    for i, a in enumerate(pts):
        b = pts[(i + 1) % len(pts)]
        ina, inb = a[2] >= NEAR, b[2] >= NEAR
        if ina:
            out.append(a)
        if ina != inb:
            t = (NEAR - a[2]) / (b[2] - a[2])
            out.append(add(a, mul(sub(b, a), t)))
    return out


def shade(rgb, normal):
    k = AMBIENT + (1 - AMBIENT) * max(0.0, dot(normal, LIGHT_DIR))
    return tuple(min(255, int(c * k)) for c in rgb)


LIGHT_DIR = norm(LIGHT)


def hex_rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def frame_size(scene):
    return FRAME_SIZES.get(scene.get("frame"), FRAME_SIZES["portrait"])


class Poly:
    """One face on screen: points in frame pixels, a colour, its depth, and
    the object and part it belongs to (None for the room). `tex` is a
    `TexMap` when the face wears a picture; `rgb` is then its mean colour.
    `dim` (0-1) makes it a shadow: it darkens what is under it by that
    factor instead of painting over it, and `rgb` is only a stand-in."""
    __slots__ = ("pts", "rgb", "depth", "owner", "part", "tex", "dim")

    def __init__(self, pts, rgb, depth, owner, part, tex=None, dim=None):
        self.pts, self.rgb, self.depth, self.owner, self.part = pts, rgb, depth, owner, part
        self.tex, self.dim = tex, dim


class TexMap:
    """A picture laid on a plane, as the frame sees it. For pixel (x, y) the
    ray is d = a + b x + c y (unnormalised, frame pixels); it meets the plane
    at t = `c` / (d . n), and the point's place in the picture is
    (e + t d) . axis, for the along and down axes. Each dot product is linear
    in x and y, so a scanline costs a division and two products a pixel -
    perspective-correct without leaving the stdlib."""

    def __init__(self, cam, origin, along, down, normal, size, tex, k=1.0):
        per_m = tex.w / float(size)               # picture pixels a metre, both ways
        self.levels = [(px, w, h, per_m * w / tex.w) for px, w, h in tex.shaded(k)]
        self.fine = per_m / cam.k                 # picture pixels a frame pixel, x t/|d.n|
        a = add(cam.f, add(mul(cam.r, -cam.w / 2 / cam.k), mul(cam.u, cam.h / 2 / cam.k)))
        lin = lambda v: (dot(a, v), dot(cam.r, v) / cam.k, -dot(cam.u, v) / cam.k)  # noqa
        self.n, self.U, self.V = lin(normal), lin(along), lin(down)
        self.c = dot(sub(origin, cam.eye), normal)
        rel = sub(cam.eye, origin)
        self.eu, self.ev = dot(rel, along), dot(rel, down)

    def span(self, y, xa, xb):
        """Pixels xa..xb of row y, as RGB bytes."""
        yc = y + 0.5
        n0, nx = self.n[0] + self.n[2] * yc, self.n[1]
        u0, ux = self.U[0] + self.U[2] * yc, self.U[1]
        v0, vx = self.V[0] + self.V[2] * yc, self.V[1]
        c, eu, ev, fine, levels = self.c, self.eu, self.ev, self.fine, self.levels
        top = len(levels) - 1
        out = []
        for x in range(xa, xb + 1):
            xc = x + 0.5
            dn = n0 + nx * xc
            t = c / dn if dn else 0.0
            # How many picture pixels this one frame pixel spans, roughly
            # (further and more edge-on is more): each level is 4x fewer.
            spread = abs(t / dn) * fine if dn else 1e9
            lv = 0 if spread < 1 else 1 if spread < 4 else 2 if spread < 16 else 3
            px, tw, th, k = levels[min(lv, top)]
            i = int((eu + t * (u0 + ux * xc)) * k + 1e6) % tw
            j = int((ev + t * (v0 + vx * xc)) * k + 1e6) % th
            out.append(px[j * tw + i])
        return b"".join(out)


def render(scene, width=None, height=None):
    """The scene through its camera -> [Poly], far to near. The room is
    first - the floor, then the walls facing into it - clipped to the near
    plane; every object face that faces the camera follows, sorted by depth.
    The same list is drawn on the canvas and rasterised by `png`."""
    if width is None:
        width, height = frame_size(scene)
    cam = Camera(scene["camera"], width, height)
    polys = room_polys(scene, cam)
    tex = texture((scene.get("room") or new_room())["floor"]["image"])
    floor = tex.mean if tex else FLOOR
    faces = []
    for obj in scene["objects"]:
        pieces = painted_pieces(obj)
        polys += shadow_polys(pieces, cam, floor)
        for part, fs, rgb in pieces:
            for f in fs:
                n = newell(f)
                if dot(n, sub(centroid(f), cam.eye)) >= 0:
                    continue                     # faces away from the camera
                c = clip_near([cam.to_camera(p) for p in f])
                if len(c) < 3:
                    continue
                depth = sum(p[2] for p in c) / len(c)
                faces.append(Poly([cam.to_screen(p) for p in c], shade(rgb, n), depth,
                                  obj["id"], part))
    faces.sort(key=lambda p: -p.depth)
    return polys + faces


# Contact shadows: (metres grown past the footprint, factor it darkens by).
# Rings overlap, so the middle is the product of them all: dark where the
# sole meets the floor, fading out past it. Without them the mannequin reads
# as pasted on the floor, and the picture made from the frame draws the
# person hovering over it.
CONTACT = tuple((0.1 * (1 - i / 7.0) + 0.01, 0.93) for i in range(7))
AMBIENT_SHADOW = tuple((0.2 * (1 - i / 4.0) + 0.02, 0.96) for i in range(4))
CONTACT_REACH = 0.04           # m above its lowest point a part still touches
AMBIENT_REACH = 0.3            # m above it whose outline casts the faint shadow


def hull(pts):
    """The convex hull of 2D points, counter-clockwise (monotone chain)."""
    pts = sorted(set(pts))
    if len(pts) < 3:
        return pts
    cross2 = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])  # noqa

    def half(seq):
        out = []
        for p in seq:
            while len(out) >= 2 and cross2(out[-2], out[-1], p) <= 0:
                out.pop()
            out.append(p)
        return out[:-1]
    return half(pts) + half(reversed(pts))


def grown(outline, r, sides=16):
    """A floor outline grown by `r` metres all round, corners rounded."""
    ring = [(math.cos(2 * math.pi * i / sides) * r, math.sin(2 * math.pi * i / sides) * r)
            for i in range(sides)]
    return hull([(x + dx, z + dz) for x, z in outline for dx, dz in ring])


def shadow_polys(pieces, cam, floor=FLOOR):
    """The soft shadow an object leaves on the floor, as `dim` polys: a
    dark contact ring under each part that touches (each foot, a knee, a
    box's base) and a faint one under the whole body. Only for an object on
    the floor itself: one standing on a platform at y > 0 would have its
    shadow drawn under the platform, since the room is drawn first."""
    pts = [p for _, faces, _ in pieces for f in faces for p in f]
    if not pts or cam.eye[1] <= 0:
        return []
    low = min(p[1] for p in pts)
    if low > 0.02:
        return []
    touching = {}
    for part, faces, _ in pieces:
        for f in faces:
            for p in f:
                if p[1] <= low + CONTACT_REACH:
                    touching.setdefault(part, []).append((p[0], p[2]))
    whole = hull([(p[0], p[2]) for p in pts if p[1] <= low + AMBIENT_REACH])
    rings = [(whole, AMBIENT_SHADOW)] if len(whole) >= 3 else []
    rings += [(hull(foot), CONTACT) for foot in touching.values()]
    out = []
    for outline, steps in rings:
        if not outline:
            continue
        seen = 1.0
        for r, k in steps:
            seen *= k
            c = clip_near([cam.to_camera((x, 0.001, z)) for x, z in grown(outline, r)])
            if len(c) >= 3:
                out.append(Poly([cam.to_screen(p) for p in c],
                                tuple(int(v * seen) for v in floor), float("inf"),
                                None, "shadow", dim=k))
    return out


def room_polys(scene, cam):
    """The floor, then each wall whose face looks into the room from where
    the camera is. A camera outside the room sees through the near wall, as
    into a doll's house. The room is the backdrop, never sorted in among the
    objects: everything in the scene stands in front of it."""
    room = scene.get("room") or new_room()
    out = []
    r = FLOOR_REACH
    floor = clip_near([cam.to_camera(p) for p in ((-r, 0, -r), (r, 0, -r), (r, 0, r),
                                                  (-r, 0, r))])
    if len(floor) >= 3 and cam.eye[1] > 0:
        tex = texture(room["floor"]["image"])
        m = (TexMap(cam, (0, 0, 0), (1, 0, 0), (0, 0, 1), (0, 1, 0),
                    room["floor"]["size"], tex) if tex else None)
        out.append(Poly([cam.to_screen(c) for c in floor], tex.mean if tex else FLOOR,
                        float("inf"), None, None, m))
    if not room["walls"]:
        return out
    tex = texture(room["wall"]["image"])
    for quad, origin, along in walls(room):
        n = newell(quad)
        if dot(n, sub(origin, cam.eye)) >= 0:
            continue                             # its back is to the camera
        c = clip_near([cam.to_camera(p) for p in quad])
        if len(c) < 3:
            continue
        k = AMBIENT + (1 - AMBIENT) * max(0.0, dot(n, LIGHT_DIR))
        m = (TexMap(cam, origin, along, (0, -1, 0), n, room["wall"]["size"], tex, k)
             if tex else None)
        rgb = tuple(min(255, int(v * k)) for v in (tex.mean if tex else WALL))
        out.append(Poly([cam.to_screen(p) for p in c], rgb, float("inf"), None, "wall", m))
    return out


def grid_lines(scene, width, height, spacing=1.0, reach=10):
    """The floor's metre grid as screen segments, for the window only: the
    picture must not be told the floor is tiled. Inside the walls when there
    are walls, since the grid is drawn over the whole room."""
    cam = Camera(scene["camera"], width, height)
    room = scene.get("room") or new_room()
    if room["walls"]:
        hw, hd = room["width"] / 2, room["depth"] / 2
        lines = [((x, 0, -hd), (x, 0, hd), x == 0)
                 for x in range(int(math.ceil(-hw)), int(math.floor(hw)) + 1)]
        lines += [((-hw, 0, z), (hw, 0, z), z == 0)
                  for z in range(int(math.ceil(-hd)), int(math.floor(hd)) + 1)]
    else:
        lines = [(a, b, i == 0) for i in range(-reach, reach + 1)
                 for a, b in (((i * spacing, 0, -reach), (i * spacing, 0, reach)),
                              ((-reach, 0, i * spacing), (reach, 0, i * spacing)))]
    out = []
    for a, b, axis in lines:
        ca, cb = cam.to_camera(a), cam.to_camera(b)
        if ca[2] < NEAR and cb[2] < NEAR:
            continue
        if ca[2] < NEAR or cb[2] < NEAR:
            t = (NEAR - ca[2]) / (cb[2] - ca[2])
            cut = add(ca, mul(sub(cb, ca), t))
            ca, cb = (cut, cb) if ca[2] < NEAR else (ca, cut)
        out.append((cam.to_screen(ca), cam.to_screen(cb), axis))
    return out


# ==================================================================== raster

def rasterise(polys, width, height, sky=SKY, flat=False):
    """Painter's-order polygons -> RGB bytes. A scanline fill of convex
    polygons; every face the renderer makes is convex. A face with a picture
    is filled from it, unless `flat` (its mean colour: quick)."""
    buf = bytearray(bytes(sky) * (width * height))
    stride = width * 3
    for poly in polys:
        pts = poly.pts
        ys = [p[1] for p in pts]
        y0 = max(0, int(math.ceil(min(ys) - 0.5)))
        y1 = min(height - 1, int(math.floor(max(ys) - 0.5)))
        if y1 < y0:
            continue
        colour = bytes(poly.rgb)
        tex = None if flat else poly.tex
        dim = bytes(int(v * poly.dim) for v in range(256)) if poly.dim else None
        edges = [(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
        edges = [(a, b) if a[1] <= b[1] else (b, a) for a, b in edges if a[1] != b[1]]
        for y in range(y0, y1 + 1):
            yc = y + 0.5
            xs = [a[0] + (yc - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
                  for a, b in edges if a[1] <= yc < b[1]]
            if len(xs) < 2:
                continue
            xa = max(0, int(math.ceil(min(xs) - 0.5)))
            xb = min(width - 1, int(math.floor(max(xs) - 0.5)))
            if xb < xa:
                continue
            row = y * stride
            a, b = row + xa * 3, row + (xb + 1) * 3
            if dim:
                buf[a:b] = bytes(buf[a:b]).translate(dim)
            else:
                buf[a:b] = tex.span(y, xa, xb) if tex else colour * (xb - xa + 1)
    return bytes(buf)


def rgb_png(rgb, width, height):
    rgba = bytearray(width * height * 4)
    for i in range(3):
        rgba[i::4] = rgb[i::3]
    rgba[3::4] = b"\xff" * (width * height)
    return studio_icons.png(bytes(rgba), width, height)


def backdrop_png(scene, width, height):
    """The room alone at `width` x `height` - sky, floor and walls, with
    their pictures - for the window to put its object faces over."""
    polys = [p for p in render(scene, width, height) if p.owner is None]
    return rgb_png(rasterise(polys, width, height), width, height)


def png(scene):
    """The camera's frame at the generation size, as PNG bytes: exactly what
    the viewport shows inside its frame, less the grid and the selection."""
    w, h = frame_size(scene)
    return rgb_png(rasterise(render(scene, w, h), w, h), w, h)


def write_reference(scene, folder=None):
    """Render the frame to `<scenes>/renders/<hash>.png` and return the path.
    Named by content, so History's settings keep pointing at the picture that
    was sent, and the same frame twice is one file."""
    data = png(scene)
    folder = folder or os.path.join(scenes_dir(), "renders")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "scene_%s.png" % hashlib.sha1(data).hexdigest()[:16])
    if not os.path.isfile(path):
        with open(path, "wb") as f:
            f.write(data)
    return path


# ==================================================================== words

class Words:
    def __init__(self):
        self.text = ""
        self.notes = []


def placement(scene, obj):
    """Where an object sits in the frame, in words, or None outside it."""
    w, h = frame_size(scene)
    cam = Camera(scene["camera"], w, h)
    lo, hi = bounds(obj)
    mid = ((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2)
    p = cam.project(mid)
    if p is None or not (0 <= p[0] <= w and 0 <= p[1] <= h):
        return None
    x = p[0] / w
    where = ["left of frame" if x < 1 / 3 else "right of frame" if x > 2 / 3
             else "centre of frame"]
    d = scene["camera"]["distance"]
    if p[2] < d * 0.75:
        where.append("foreground")
    elif p[2] > d * 1.35:
        where.append("background")
    return where


def facing(scene, obj):
    """Which way a person faces, as the picture will show it."""
    w, h = frame_size(scene)
    cam = Camera(scene["camera"], w, h)
    fwd = apply(euler(*obj["rotation"]), (0, 0, 1))
    fwd = norm((fwd[0], 0, fwd[2]))
    to_cam = norm((cam.eye[0] - obj["position"][0], 0, cam.eye[2] - obj["position"][2]))
    c = dot(fwd, to_cam)
    if c > 0.7:
        return "facing the camera"
    if c < -0.7:
        return "back to the camera"
    side = "right" if dot(fwd, cam.r) > 0 else "left"
    if c > 0.2:
        return "three-quarter view, turned to frame %s" % side
    if c < -0.2:
        return "turned away, towards frame %s" % side
    return "in profile, facing frame %s" % side


def camera_words(scene):
    c = scene["camera"]
    lens = int(round(c["lens"]))
    kind = "wide-angle " if lens <= 28 else "telephoto " if lens >= 70 else ""
    if c["pitch"] > 35:
        angle = "a high angle looking down"
    elif c["pitch"] > 12:
        angle = "slightly above, looking down"
    elif c["pitch"] < -12:
        angle = "a low angle looking up"
    else:
        angle = "eye level"
    return "Shot from %s on a %dmm %slens" % (angle, lens, kind)


def scene_text(scene):
    """The scene's words for the prompt. Every description is kept exactly
    as written; only where things are in the frame is added around it."""
    out = Words()
    parts = []
    details = scene["details"].strip()
    if details:
        parts.append(details.rstrip())
    room = scene.get("room") or new_room()
    for key, label, _ in SURFACES:
        words = room[key]["prompt"].strip()
        if words and (key == "floor" or room["walls"]):
            parts.append("The %s: %s" % (label.lower(), words))
    for obj in scene["objects"]:
        where = placement(scene, obj)
        if where is None:
            out.notes.append("%s is outside the frame, so the words leave it out."
                             % obj["name"])
            continue
        about = []
        if obj["asset"] == "person":
            about.append("a person")
        about += where
        if obj["asset"] == "person":
            about.append(facing(scene, obj))
            preset = obj["pose"].get("preset")
            if preset in POSE_NAMES and preset != "standing":
                about.append(POSE_NAMES[preset].lower())
        line = "%s (%s)" % (obj["name"].strip() or ASSET[obj["asset"]]["label"],
                            ", ".join(about))
        desc = obj["description"].strip()
        look = look_text(obj) if obj["asset"] == "person" else ""
        said = ". ".join(x for x in (look, desc) if x)
        parts.append(line + (": " + said if said else ""))
        if not said:
            out.notes.append("%s has no description; the picture has only its shape and "
                             "name to go on." % obj["name"])
    parts.append(camera_words(scene))
    out.text = "\n".join(p if p.endswith((".", "!", "?")) else p + "." for p in parts)
    return out


def people(scene):
    return [o for o in scene["objects"] if o["asset"] == "person"]


def generation(scene, reference, characters=None):
    """What the Image Studio is handed: the words for its Scene field, and
    the settings Generate adds to the form's (the frame's size, the redraw
    strength, the reference, and the whole scene for History).

    A scene with people in it says every person's look in their own line,
    so the form's one person is blanked for this job - said twice, the
    picture gets an extra person or a blend of two, and so are the form's
    item pictures (compose matches them to the form's clothes, now blank;
    no workflow takes one yet, and the words carry the items). The identity
    whose LoRA carries a scene character's face rides along as
    `scene_identities`, for the form to add to its own. `characters` is
    {id: record}."""
    import studio_imagegen as ig
    w, h = frame_size(scene)
    words = scene_text(scene)
    s, _ = clean_scene(scene)
    extra = {"width": w, "height": h, "denoise": round(scene["redraw"], 3),
             "scene_layout": copy.deepcopy(s)}
    folks = people(s)
    if folks:
        extra.update({k: "" for k in ig.SLOTS})
        extra.update({k: 0 for k in ig.SLIDER_KEYS})
        extra["character"] = ""
        extra["item_refs"] = {}
        idents = []
        for o in folks:
            rec = (characters or {}).get(o["character"]) if o["character"] else None
            if rec and rec.get("identity") and rec["identity"] not in idents:
                idents.append(rec["identity"])
        extra["scene_identities"] = idents
    return words, reference, extra


# ==================================================================== pose from a photo
# A photo's pose, as the mannequin's controls. ComfyUI's `StudioDWPoseKeypoints`
# node (comfy_nodes/studio_dwpose) finds each person's COCO-WholeBody points in
# the picture; `fit_pose` searches the controls and the way the person faces
# for the pose whose joints, seen from the front with no perspective, fall on
# them. A flat picture cannot say how far a limb reaches towards the camera,
# so the search leans a little towards the rest pose and away from arms thrown
# back: the fit is a start to adjust, not a measurement.
SEEN = 0.3                    # a point DWPose is less sure of than this is not used
KP_JOINTS = {5: "shoulder_l", 6: "shoulder_r", 7: "elbow_l", 8: "elbow_r",
             9: "wrist_l", 10: "wrist_r", 11: "hip_l", 12: "hip_r",
             13: "knee_l", 14: "knee_r", 15: "ankle_l", 16: "ankle_r"}
KP_HEAD = {0: (0, 0.09, 0.135),                                 # the nose's tip
           1: (0.032, 0.125, 0.095), 2: (-0.032, 0.125, 0.095),  # the eyes
           3: (0.085, 0.10, 0.0), 4: (-0.085, 0.10, 0.0)}        # the ears
# Which points show each group of controls; a group none of them shows is
# left at rest and said so.
KP_GROUPS = [("head", "the head", (0, 1, 2, 3, 4), ("head_turn", "head_nod", "head_tilt")),
             ("body", "the body", (5, 6, 11, 12), ("bend", "twist", "lean"))]
for _side, _name, _o in (("l", "left", 0), ("r", "right", 1)):
    KP_GROUPS += [
        ("hand_" + _side, "the %s arm" % _name, (7 + _o, 9 + _o),
         ("arm_%s_raise" % _side, "arm_%s_out" % _side, "arm_%s_bend" % _side)),
        ("foot_" + _side, "the %s leg" % _name, (13 + _o, 15 + _o),
         ("leg_%s_step" % _side, "leg_%s_out" % _side, "leg_%s_bend" % _side))]
PRIOR = 1.5e-4                 # per (90 degrees from rest)^2, against a misfit in heights^2
BACKWARDS = 4e-4               # per (45 degrees)^2 an arm is swung back, or the body leans back
LIMB_TRIES = {
    "arm": [(r, o, b) for r in (-30, 20, 60, 100, 140, 175) for o in (0, 45, 90)
            for b in (0, 60, 120)],
    "leg": [(s, o, b) for s in (-30, 0, 30, 60, 90, 115) for o in (0, 25, 45)
            for b in (0, 45, 90, 135)],
}


class PoseFit:
    """What `fit_pose` found: `controls` (every CONTROL_KEYS), `yaw` (deg the
    person is turned from facing the camera; + is to frame right), `error`
    (the joints' root-mean-square distance from the photo's points, as a
    fraction of the person's height) and `unseen` (the parts the photo did
    not show, left at rest)."""

    def __init__(self, controls, yaw, error, unseen):
        self.controls, self.yaw, self.error, self.unseen = controls, yaw, error, unseen

    @property
    def rough(self):
        return self.error > 0.05


def photo_people(data):
    """The node's JSON (text or parsed) -> its people, most prominent first
    (the biggest box, weighed by how sure the finder was), each
    {"box": [x0, y0, x1, y1], "score", "points": [[x, y, score] * 133]}."""
    if isinstance(data, (str, bytes)):
        data = json.loads(data)
    folk = [p for p in (data or {}).get("people") or []
            if isinstance(p, dict) and len(p.get("points") or ()) >= 17
            and len(p.get("box") or ()) == 4]

    def size(p):
        b = p["box"]
        return (b[2] - b[0]) * (b[3] - b[1]) * p.get("score", 1)
    return sorted(folk, key=size, reverse=True)


def pose_points(controls, yaw=0.0, shape=None):
    """{COCO point index: world position} of the mannequin, pelvis at the
    origin, turned `yaw` degrees."""
    sk = skeleton(controls, euler(yaw), shape)
    out = {i: sk[j][0] for i, j in KP_JOINTS.items()}
    hp, hm = sk["head"]
    for i, v in KP_HEAD.items():
        out[i] = add(hp, apply(hm, v))
    return out


def _misfit(model, target):
    """Mean squared distance of the model's points, seen from the front and
    scaled and moved to fit best, from `target` [(index, x, y, weight)] (y
    down). None when no positive scale fits."""
    sw = sum(t[3] for t in target)
    mx = sum(model[i][0] * w for i, _, _, w in target) / sw
    my = sum(-model[i][1] * w for i, _, _, w in target) / sw
    dx = sum(x * w for _, x, _, w in target) / sw
    dy = sum(y * w for _, _, y, w in target) / sw
    num = den = 0.0
    for i, x, y, w in target:
        ax, ay = model[i][0] - mx, -model[i][1] - my
        num += w * (ax * (x - dx) + ay * (y - dy))
        den += w * (ax * ax + ay * ay)
    if den <= 0 or num <= 0:
        return None
    s = num / den
    err = 0.0
    for i, x, y, w in target:
        ex = s * (model[i][0] - mx) - (x - dx)
        ey = s * (-model[i][1] - my) - (y - dy)
        err += w * (ex * ex + ey * ey)
    return err / sw


def fit_pose(points, shape=None, box=None):
    """A person's 133 [x, y, score] (DWPose's order) -> PoseFit. `box` is
    their [x0, y0, x1, y1]; its height is the unit the fit is judged in.
    Raises ValueError when too little of the body is seen to fit anything."""
    seen = {i: (float(p[0]), float(p[1]), float(p[2])) for i, p in enumerate(points[:17])
            if len(p) >= 3 and p[2] >= SEEN}
    if sum(i in seen for i in (5, 6, 11, 12)) < 2 or len(seen) < 5:
        raise ValueError("The photo does not show enough of the person to pose from "
                         "(it needs the shoulders or hips and a few more joints).")
    ys = [p[1] for p in seen.values()]
    tall = max((box[3] - box[1]) if box else 0, max(ys) - min(ys), 1.0)
    target = [(i, x / tall, y / tall, s) for i, (x, y, s) in seen.items()]
    rest = pose_controls("standing")
    free, unseen = [], []
    for _part, name, idx, keys in KP_GROUPS:
        if any(i in seen for i in idx):
            free += keys
        else:
            unseen.append(name)

    core = [t for t in target if t[0] in (0, 1, 2, 3, 4, 5, 6, 11, 12)]

    def cost(state, target=target):
        m = _misfit(pose_points(state, state["_yaw"], shape), target)
        if m is None:
            return float("inf")
        prior = sum(((state[k] - rest[k]) / 90.0) ** 2 for k in free) * PRIOR
        back = [state["bend"]] + [state["arm_%s_raise" % s] for s in ("l", "r")]
        prior += sum(min(0.0, v) ** 2 for v in back) / 45.0 ** 2 * BACKWARDS
        return m + prior

    def move(state, k, d):
        trial = dict(state)
        if k == "_yaw":
            trial[k] = (state[k] + d + 180) % 360 - 180
        else:
            lo, hi = CONTROL_RANGE[k]
            trial[k] = min(hi, max(lo, state[k] + d))
        return trial

    def descend(state, keys, steps, target=target):
        """Coordinate descent: each key a step either way while that helps,
        then smaller steps."""
        best = cost(state, target)
        for step in steps:
            for _ in range(8):
                better = False
                for k in keys:
                    for d in (step, -step):
                        trial = move(state, k, d)
                        c = cost(trial, target)
                        if c < best - 1e-12:
                            state, best, better = trial, c, True
                            break
                if not better:
                    break
        return state, best

    def limbs(state):
        """Each limb from a spread of starts, the rest held: a limb towards
        or away from the camera looks alike from the front."""
        for _part, _name, _idx, limb in KP_GROUPS[2:]:
            if limb[0] not in free:
                continue
            kind = "arm" if limb[0].startswith("arm") else "leg"
            tries = LIMB_TRIES[kind] + [tuple(state[k] for k in limb)]
            ranked = sorted(tries, key=lambda v: cost(dict(state, **dict(zip(limb, v)))))
            state = min((descend(dict(state, **dict(zip(limb, v))), limb, (12, 6, 3))
                         for v in ranked[:2]), key=lambda r: r[1])[0]
        return state

    # Every way the person might face, on the head, shoulders and hips alone
    # (a limb still at rest would pull the torso to make up for it); then,
    # from the best three, the limbs, and all of it together, twice.
    keys = ["_yaw"] + free
    trunk = ["_yaw"] + [k for k in ("bend", "twist", "lean", "head_turn", "head_nod",
                                    "head_tilt") if k in free]
    starts = sorted((descend(dict(rest, _yaw=float(yaw)), trunk, (24, 12, 6), core)
                     for yaw in range(-180, 180, 30)), key=lambda r: r[1])
    results = []
    for state, _ in starts[:3]:
        for steps in ((12, 6, 3), (8, 4, 2, 1)):
            state = descend(limbs(state), keys, steps)[0]
        results.append((state, cost(state)))
    state, _ = min(results, key=lambda r: r[1])
    err = _misfit(pose_points(state, state["_yaw"], shape), target) or 0.0
    controls = {k: float(round(state[k])) for k in CONTROL_KEYS}
    return PoseFit(controls, float(round(state["_yaw"])), math.sqrt(err), unseen)
