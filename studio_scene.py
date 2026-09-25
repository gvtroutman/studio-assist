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
- **The camera** orbits a target: yaw, pitch, distance and a lens in mm on a
  full-frame diagonal, so 35mm means what it does on a camera whatever the
  frame's shape (`Camera`).
- **The render** is one list of shaded polygons, painter's order, near-plane
  clipped (`render`). The window draws it on a Tk canvas; `png()` rasterises
  the same list at the generation size, which is how the viewport shows the
  exact frame the picture is made from.
- **The words** (`scene_text`): the scene's details, then each object in the
  frame as "Name (where it is in the frame, which way a person faces): the
  description, verbatim", then the camera. An object outside the frame is
  left out and said so (`Words.notes`).

No tkinter here; `studio_scene_ui.py` is the window. Stdlib only.
"""

import copy
import hashlib
import json
import math
import os
import re

import studio_icons

VERSION = 1
FULL_FRAME_DIAGONAL = 43.27    # mm; a lens is read against a full-frame sensor
NEAR = 0.05                    # m; the camera's near plane
FLOOR_REACH = 30.0             # m from the origin the floor is drawn to
TILE = 0.3                     # m; a prop's faces are cut to about this, for sorting
SKY = (201, 204, 209)
FLOOR = (143, 138, 132)
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


def skeleton(controls, root=IDENTITY):
    """Forward kinematics: {joint: (position, world rotation)} with the
    pelvis at the origin and the whole rig turned by `root`."""
    rot = joint_rotations(controls)
    out = {}
    for name, parent, offset in JOINTS:
        local = rot.get(name, IDENTITY)
        if parent is None:
            out[name] = ((0.0, 0.0, 0.0), mat_mul(root, local))
        else:
            pp, pm = out[parent]
            out[name] = (add(pp, apply(pm, offset)), mat_mul(pm, local))
    return out


def person_pieces(controls, root=IDENTITY):
    """The mannequin: [(part, faces)], pelvis at the origin. Parts are the
    control groups, so a click on a hand selects the hand's controls."""
    sk = skeleton(controls, root)
    P = lambda j: sk[j][0]                                 # noqa: E731
    M = lambda j: sk[j][1]                                 # noqa: E731
    X = lambda j: column(M(j), 0)                          # noqa: E731
    at = lambda j, v: add(P(j), apply(M(j), v))            # noqa: E731
    out = [
        ("body", prism(at("pelvis", (0, -0.07, 0)), P("spine"), X("pelvis"),
                       (0.16, 0.10), (0.15, 0.10))),
        ("body", prism(P("spine"), P("chest"), X("spine"), (0.14, 0.095), (0.155, 0.10))),
        ("body", prism(P("chest"), at("chest", (0, 0.20, 0)), X("chest"),
                       (0.175, 0.11), (0.14, 0.085))),
        ("head", prism(P("neck"), P("head"), X("neck"), (0.05, 0.05), (0.048, 0.048), 6)),
        ("head", ellipsoid(at("head", (0, 0.11, 0.01)), M("head"), (0.085, 0.115, 0.1))),
    ]
    # The nose says which way the head faces: a small wedge on the front.
    nose = [at("head", v) for v in ((-0.016, 0.07, 0.1), (0.016, 0.07, 0.1),
                                    (0.016, 0.115, 0.1), (-0.016, 0.115, 0.1),
                                    (-0.01, 0.075, 0.135), (0.01, 0.075, 0.135),
                                    (0.01, 0.1, 0.13), (-0.01, 0.1, 0.13))]
    idx = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (3, 2, 6, 7), (0, 3, 7, 4), (1, 2, 6, 5)]
    out.append(("head", outward([[nose[i] for i in f] for f in idx])))
    for side in ("l", "r"):
        hand, foot = "hand_" + side, "foot_" + side
        out += [
            (hand, prism(P("shoulder_" + side), P("elbow_" + side), X("shoulder_" + side),
                         (0.052, 0.052), (0.042, 0.042))),
            (hand, prism(P("elbow_" + side), P("wrist_" + side), X("elbow_" + side),
                         (0.042, 0.042), (0.033, 0.03))),
            (hand, prism(P("wrist_" + side), at("wrist_" + side, (0, -0.18, 0.01)),
                         X("wrist_" + side), (0.045, 0.022), (0.04, 0.018), 6)),
            (foot, prism(P("hip_" + side), P("knee_" + side), X("hip_" + side),
                         (0.078, 0.078), (0.056, 0.056))),
            (foot, prism(P("knee_" + side), P("ankle_" + side), X("knee_" + side),
                         (0.052, 0.052), (0.04, 0.04))),
            (foot, prism(at("ankle_" + side, (0, -0.045, -0.05)),
                         at("ankle_" + side, (0, -0.045, 0.19)), X("ankle_" + side),
                         (0.045, 0.035), (0.042, 0.025), 6)),
        ]
    return out


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


def object_pieces(obj):
    """An object's faces in the world: [(part, faces)], standing on its floor
    (its lowest point at position y)."""
    rot = euler(*obj["rotation"])
    sx, sy, sz = obj["scale"]
    if obj["asset"] == "person":
        pieces = person_pieces(obj["pose"]["controls"], rot)
        pieces = [(part, [[mul(p, sx) for p in f] for f in faces]) for part, faces in pieces]
    else:
        faces = [[apply(rot, (p[0] * sx, p[1] * sy, p[2] * sz)) for p in f]
                 for f in UNIT[obj["asset"]]]
        pieces = [("body", [t for f in faces for t in tiles(f)])]
    low = min(p[1] for _, faces in pieces for f in faces for p in f)
    x, y, z = obj["position"]
    shift = (x, y - low, z)
    return [(part, [[add(p, shift) for p in f] for f in faces]) for part, faces in pieces]


def bounds(obj):
    pts = [p for _, faces in object_pieces(obj) for f in faces for p in f]
    lo = tuple(min(p[i] for p in pts) for i in range(3))
    hi = tuple(max(p[i] for p in pts) for i in range(3))
    return lo, hi


# ==================================================================== scene

def new_scene(details=""):
    return {"version": VERSION, "details": details, "frame": "portrait", "redraw": REDRAW,
            "camera": {"target": [0.0, 1.0, 0.0], "yaw": 0.0, "pitch": 6.0,
                       "distance": 4.2, "lens": 35.0},
            "objects": []}


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
    return obj


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
    the object and part it belongs to (None for the floor)."""
    __slots__ = ("pts", "rgb", "depth", "owner", "part")

    def __init__(self, pts, rgb, depth, owner, part):
        self.pts, self.rgb, self.depth, self.owner, self.part = pts, rgb, depth, owner, part


def render(scene, width=None, height=None):
    """The scene through its camera -> [Poly], far to near. The floor is
    first, clipped to the near plane; every object face that faces the camera
    follows, sorted by depth. The same list is drawn on the canvas and
    rasterised by `png`."""
    if width is None:
        width, height = frame_size(scene)
    cam = Camera(scene["camera"], width, height)
    polys = []
    r = FLOOR_REACH
    floor = [cam.to_camera(p) for p in ((-r, 0, -r), (r, 0, -r), (r, 0, r), (-r, 0, r))]
    floor = clip_near(floor)
    if len(floor) >= 3 and cam.eye[1] > 0:
        polys.append(Poly([cam.to_screen(c) for c in floor], FLOOR, float("inf"), None, None))
    faces = []
    for obj in scene["objects"]:
        rgb = hex_rgb(obj["colour"])
        for part, fs in object_pieces(obj):
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


def grid_lines(scene, width, height, spacing=1.0, reach=10):
    """The floor's metre grid as screen segments, for the window only: the
    picture must not be told the floor is tiled."""
    cam = Camera(scene["camera"], width, height)
    out = []
    for i in range(-reach, reach + 1):
        for a, b in (((i * spacing, 0, -reach), (i * spacing, 0, reach)),
                     ((-reach, 0, i * spacing), (reach, 0, i * spacing))):
            ca, cb = cam.to_camera(a), cam.to_camera(b)
            if ca[2] < NEAR and cb[2] < NEAR:
                continue
            if ca[2] < NEAR or cb[2] < NEAR:
                t = (NEAR - ca[2]) / (cb[2] - ca[2])
                cut = add(ca, mul(sub(cb, ca), t))
                ca, cb = (cut, cb) if ca[2] < NEAR else (ca, cut)
            out.append((cam.to_screen(ca), cam.to_screen(cb), i == 0))
    return out


# ==================================================================== raster

def rasterise(polys, width, height, sky=SKY):
    """Painter's-order polygons -> RGB bytes. A scanline fill of convex
    polygons; every face the renderer makes is convex."""
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
            buf[row + xa * 3:row + (xb + 1) * 3] = colour * (xb - xa + 1)
    return bytes(buf)


def png(scene):
    """The camera's frame at the generation size, as PNG bytes: exactly what
    the viewport shows inside its frame, less the grid and the selection."""
    w, h = frame_size(scene)
    rgb = rasterise(render(scene, w, h), w, h)
    rgba = bytearray(w * h * 4)
    for i in range(3):
        rgba[i::4] = rgb[i::3]
    rgba[3::4] = b"\xff" * (w * h)
    return studio_icons.png(bytes(rgba), w, h)


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
        parts.append(line + (": " + desc if desc else ""))
        if not desc:
            out.notes.append("%s has no description; the picture has only its shape and "
                             "name to go on." % obj["name"])
    parts.append(camera_words(scene))
    out.text = "\n".join(p if p.endswith((".", "!", "?")) else p + "." for p in parts)
    return out


def generation(scene, reference):
    """What the Image Studio is handed: the words for its Scene field, and
    the settings Generate adds to the form's (the frame's size, the redraw
    strength, the reference, and the whole scene for History)."""
    w, h = frame_size(scene)
    words = scene_text(scene)
    s, _ = clean_scene(scene)
    extra = {"width": w, "height": h, "denoise": round(scene["redraw"], 3),
             "scene_layout": copy.deepcopy(s)}
    return words, reference, extra
