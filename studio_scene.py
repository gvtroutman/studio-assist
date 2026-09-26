#!/usr/bin/env python3
"""
studio_scene - the Scene Builder's engine: a small 3D stage for blocking out a
picture before the Image Studio makes it.

Not a 3D package and not a slicer, though it is laid out like one: a person
and a couple of props on a floor, one camera, and the frame that camera sees.
What the picture is made from is what that frame means (`scene_maps`): a
pose map of every body's joints and a depth map of the whole frame, for the
FLUX ControlNet - and the grey frame itself as a `source` (image to image)
only when asked, or for a model with no ControlNet. The words written on
each object go into the prompt as written. The pieces:

- **The rig** (`JOINTS`): a figure of fifteen joints - a lofted torso
  in blocks and limbs (`loft`), an egg of a head, ball joints, mitten
  hands - an artist's wooden mannequin, posed by forward
  kinematics from a handful of named controls (`CONTROLS`) - head, torso, and
  each hand and foot - and `POSES`, presets of those controls. Everything
  stands on its floor: an object's lowest point is put at its `position` y
  (`ground`), so a crouch drops the hips and a tipped drum lies on the floor.
- **Shapes and props** (`ASSETS`, `MESHES`): box, cylinder, sphere, cone,
  frustum, capsule, wedge, pyramid and panel, and props made of a few parts -
  table, chair, bench, shelves, barrel, tree, bush, lamp post, parasol, car,
  fence - each sized by scale in metres. What a prop *is* is its name and its
  description; the shape is a stand-in that only holds its place in the
  frame. `STAND_INS` start a shape already named and sized.
- **A background crowd** (`crowd_members`): one object, many mannequins -
  each their own height, build, skin, clothes, pose and facing, dealt from a
  seed - said in the words as one line.
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
import random
import re
import struct
import zlib

import studio_icons
import studio_mannequin as mq
import studio_pose

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
# What the picture is made from (`scene_maps`), each 0 for off: the pose map's
# and the depth map's ControlNet strengths, and how much of the grey frame
# itself is kept (1 - denoise). The frame is off by default: image to image
# from it copies the mannequins' blocky look at any denoise that keeps the
# layout, which the two maps keep without it.
POSE_STRENGTH = 0.85
DEPTH_STRENGTH = 0.55
FRAME_KEEP = 0.0
FRAME_KEEP_MAX = 0.7
FALLBACK_KEEP = 0.3            # the frame kept for a model with no ControlNet (denoise 0.7)
FACE_LIKENESS = 0.6            # how far a face given a picture is redrawn at the end
REAL_FACES = True              # then their own face pasted over it, where a photo's angle fits
FACE_LIKENESS_RANGE = (0.3, 0.95)
DEPTH_EDGE = 512               # px on the depth map's long edge; the ControlNet scales it

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


def transpose(m):
    return tuple(tuple(m[j][i] for j in range(3)) for i in range(3))


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


def loft(p0, p1, side, profile, n=12):
    """A tube from p0 to p1 through rings of `profile`: [(t, (across, deep),
    shift[, shape])], t 0..1 along the way and `shift` how far the ring's
    middle sits off the axis in the `deep` direction - a calf behind the
    shin, a chest in front of the spine. `shape`, if given, is the ring's
    section (`studio_mannequin`): the angle round -> how far out, 1 on the
    ellipse. `side` is the direction `across` runs, as for `prism`, whose
    two rings are a loft's simplest."""
    d = norm(sub(p1, p0))
    a = sub(side, mul(d, dot(side, d)))
    if dot(a, a) < 1e-8:
        a = cross(d, (0, 0, 1) if abs(d[2]) < 0.9 else (1, 0, 0))
    a = norm(a)
    b = cross(d, a)
    rings = []
    for t, (ra, rb), shift, *shape in profile:
        c = add(add(p0, mul(sub(p1, p0), t)), mul(b, shift))
        ring = []
        for i in range(n):
            th = 2 * math.pi * (i + 0.5) / n
            k = shape[0](th) if shape else 1.0
            ring.append(add(c, add(mul(a, ra * k * math.cos(th)), mul(b, rb * k * math.sin(th)))))
        rings.append(ring)
    faces = [[lo[i], lo[(i + 1) % n], hi[(i + 1) % n], hi[i]]
             for lo, hi in zip(rings, rings[1:]) for i in range(n)]
    faces += [rings[0][::-1], rings[-1]]
    return outward(faces)


def ellipsoid(c, m, radii, seg=8, rings=6, warp=None):
    """An ellipsoid at `c`, axes the columns of `m`, of `radii`; `warp`, if
    given, reshapes each point in the ellipsoid's own frame first."""
    pts = []
    for j in range(rings + 1):
        v = math.pi * j / rings
        row = []
        for i in range(seg):
            u = 2 * math.pi * i / seg
            local = (radii[0] * math.sin(v) * math.cos(u), radii[1] * math.cos(v),
                     radii[2] * math.sin(v) * math.sin(u))
            if warp:
                local = warp(local)
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


def lathe(profile, seg=16):
    """A solid turned about the y axis: `profile` is [(radius, y)] from the
    bottom up. A radius of 0 is a point (a pole, a cone's tip); an end with
    a radius is capped flat."""
    rings = [[(r * math.cos(2 * math.pi * (i + 0.5) / seg), y,
               r * math.sin(2 * math.pi * (i + 0.5) / seg)) for i in range(seg)]
             for r, y in profile]
    faces = []
    for (r0, _), (r1, _), lo, hi in zip(profile, profile[1:], rings, rings[1:]):
        for i in range(seg):
            j = (i + 1) % seg
            if r0 < 1e-9:
                faces.append([lo[i], hi[j], hi[i]])
            elif r1 < 1e-9:
                faces.append([lo[i], lo[j], hi[i]])
            else:
                faces.append([lo[i], lo[j], hi[j], hi[i]])
    if profile[0][0] > 1e-9:
        faces.append(rings[0][::-1])
    if profile[-1][0] > 1e-9:
        faces.append(rings[-1])
    return outward(faces)


def cone(top=0.0, seg=16):
    """A unit cone on the floor: radius 0.5 at the base, `top` at height 1
    (a frustum when `top` is more than 0)."""
    return lathe([(0.5, 0.0), (top, 1.0)], seg)


def capsule(seg=16, rings=4):
    """A unit capsule standing on the floor: radius 0.5, height 1 end to end,
    so a round end is half its width. Stretched by scale, the ends go oval."""
    cap = [(0.5 * math.sin(math.pi / 2 * j / rings), 0.25 - 0.25 * math.cos(math.pi / 2 * j / rings))
           for j in range(rings + 1)]
    return lathe(cap + [(r, 1 - y) for r, y in reversed(cap)], seg)


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


def _jaw(p):
    """The skull's ellipsoid made a head: narrower at the jaw and the chin,
    the back of the neck in under the skull, the face a little flatter."""
    x, y, z = p
    v = y / 0.115
    if v < 0:
        x *= 1 - 0.32 * (-v) ** 1.5
        if z < 0:
            z *= 1 - 0.45 * (-v) ** 1.2
    if z > 0 and v > -0.6:
        z *= 1 - 0.06 * (1 - abs(x) / 0.085)
    return (x, y, z)


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
             for f in ellipsoid((0, 0.11, 0.01), IDENTITY, (0.085, 0.115, 0.1), 16, 12, _jaw)
             if not (scalp and centroid(f)[1] > hairline(centroid(f)[0], centroid(f)[2] - 0.01))]
    shoes = dressed.get("shoes")
    toe_drop = HEEL if shoes and shoes[0] == "heels" else 0.0     # up on its toes
    # (part, region, faces); a region is what clothes cover, or the rgb of a
    # piece that is only clothes (a hem, a boot shaft).
    # The artist's wooden mannequin: a chest block, an abdomen and a pelvis,
    # each lofted round and drawn in at its ends so the joins between them
    # read as joins; a smooth egg of a head with a jaw and no face. Up a
    # bone the `deep` side is the back.
    wide = shape["shoulders"]
    lf = lambda rows, ka, kd: [(t, (ra * ka, rb * kd), sh * kd)     # noqa: E731
                               for t, ra, rb, sh in rows]
    seat = mq.pelvis(0.06 + 0.04 * max(0.0, curve - 1))
    waist = mq.abdomen(0.02)
    out = [
        ("body", "hips", loft(at("pelvis", (0, -0.1, 0)), at("spine", (0, 0.015, 0)),
                              X("pelvis"), [row + (seat,) for row in lf(
            [(0, 0.09, 0.065, 0.0), (0.15, 0.14, 0.092, 0.004), (0.55, 0.16, 0.1, 0.006),
             (0.88, 0.152, 0.096, 0.004), (1, 0.13, 0.085, 0.0)], hips * curve, hips)], 20)),
        ("body", "belly", loft(at("spine", (0, 0.015, 0)), at("chest", (0, 0.0, 0)),
                               X("spine"), [
            (0, (0.12 * belly, 0.082 * belly), 0.0, waist),
            (0.2, (0.138 * belly, 0.092 * belly), 0.0, waist),
            (0.55, (0.14 * belly, 0.094 * belly * shape["belly"]),
             -0.006 * shape["belly"] - 0.04 * fat, waist),
            (1, (0.128 * belly, 0.086 * belly), 0.0, waist)], 20)),
        ("body", "chest", loft(at("chest", (0, 0.0, 0)), at("chest", (0, 0.23, 0)),
                               X("chest"), [
            (0, (0.122 * chest, 0.083 * chest), 0.0, mq.chest(0)),
            (0.12, (0.15 * chest, 0.096 * chest), -0.002, mq.chest(0.02)),
            (0.35, (0.162 * chest * wide ** 0.5, 0.1 * chest), -0.008, mq.chest(0.08)),
            (0.55, (0.172 * chest * wide ** 0.75, 0.102 * chest), -0.01, mq.chest(0.07)),
            (0.8, (0.18 * chest * wide, 0.094 * chest), -0.004, mq.chest(0.01)),
            (0.94, (0.158 * chest * wide, 0.074 * chest), 0.004, mq.chest(0)),
            (1, (0.08 * neck, 0.055 * neck), 0.006)], 24)),
        ("head", "neck", loft(at("neck", (0, -0.02, 0)), at("head", (0, 0.04, 0)), X("neck"), [
            (0, r((0.056, 0.054), neck), 0.0), (0.5, r((0.052, 0.05), neck), 0.0),
            (1, r((0.05, 0.048), neck), 0.0)], 12)),
        ("head", "head", skull),
    ]
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
    for side, sign in (("l", 1), ("r", -1)):
        hand, foot = "hand_" + side, "foot_" + side
        # A boot's shaft is the shin's lower part, not a tube over it: the
        # shin's long faces sort in front of the shaft's and show through.
        shin_end = P("ankle_" + side)
        if shaft:
            up = norm(sub(P("knee_" + side), P("ankle_" + side)))
            shin_end = add(P("ankle_" + side), mul(up, shaft[0]))
        out += [
            # Limbs are lofted too: a biceps in front of the upper arm, the
            # forearm full below the elbow, a calf behind the shin. Down a
            # bone the `deep` side is the front.
            (hand, "upper_arm", loft(P("shoulder_" + side), P("elbow_" + side),
                                     X("shoulder_" + side), [
                (0, r((0.056, 0.056), upper), 0.0), (0.4, r((0.057, 0.059), upper), 0.004),
                (0.75, r((0.05, 0.051), upper), 0.002),
                (1, r((0.041, 0.041), (upper + fore) / 2), 0.0)])),
            (hand, "forearm", loft(P("elbow_" + side), P("wrist_" + side), X("elbow_" + side), [
                (0, r((0.044, 0.044), fore), 0.0), (0.25, r((0.05, 0.047), fore), 0.0),
                (0.7, r((0.04, 0.036), fore), 0.0), (1, r((0.032, 0.025), fore), 0.0)])),
            # A mitten of a hand, the fingers as one, curled a little at the end.
            (hand, "hand", loft(P("wrist_" + side), at("wrist_" + side, (0, -0.19, 0.02)),
                                X("wrist_" + side), [
                (0, (0.028, 0.02), 0.0, mq.MITTEN), (0.3, (0.042, 0.02), 0.0, mq.MITTEN),
                (0.6, (0.044, 0.016), 0.0, mq.MITTEN), (0.88, (0.038, 0.012), 0.004, mq.MITTEN),
                (1, (0.022, 0.008), 0.008, mq.MITTEN)], 14)),
            (foot, "thigh", loft(P("hip_" + side), P("knee_" + side), X("hip_" + side), [
                (0, r((0.084, 0.084), thigh), 0.0), (0.3, r((0.084, 0.086), thigh), 0.004),
                (0.8, r((0.06, 0.062), (thigh + shin) / 2), 0.004),
                (1, r((0.055, 0.056), shin), 0.0)])),
            (foot, "shin", loft(P("knee_" + side), shin_end, X("knee_" + side), [
                (0, r((0.055, 0.055), shin), 0.0), (0.3, r((0.06, 0.064), shin), -0.01),
                (0.75, r((0.045, 0.047), shin), -0.003), (1, r((0.036, 0.036), shin), 0.0)])),
            (foot, "foot", loft(at("ankle_" + side, (0, -0.045, -0.055)),
                                at("ankle_" + side, (0, -0.045 - toe_drop, 0.19 - toe_drop)),
                                X("ankle_" + side), [
                (0, r((0.032, 0.03), k("foot", 1)), 0.005),
                (0.15, r((0.038, 0.035), k("foot", 1)), 0.0),
                (0.55, r((0.044, 0.033), k("foot", 1)), -0.002),
                (0.85, r((0.047, 0.024), k("foot", 1)), -0.011),
                (1, r((0.036, 0.018), k("foot", 1)), -0.017)], 10)),
            # Round joints where the limbs meet, so a bent arm or knee is
            # one limb and not two tubes with a gap between them.
            (hand, "upper_arm", ellipsoid(P("shoulder_" + side), M("shoulder_" + side),
                                          (0.058 * upper, 0.056 * upper, 0.056 * upper), 12, 8)),
            (hand, "forearm", ellipsoid(P("elbow_" + side), M("elbow_" + side),
                                        (0.043 * fore,) * 3, 12, 8)),
            (hand, "hand", ellipsoid(P("wrist_" + side), M("wrist_" + side),
                                     (0.032, 0.03, 0.026), 10, 6)),
            (hand, "hand", prism(at("wrist_" + side, (-sign * 0.03, -0.03, 0.012)),
                                 at("wrist_" + side, (-sign * 0.045, -0.08, 0.032)),
                                 X("wrist_" + side), (0.013, 0.012), (0.011, 0.01), 6)),
            (hand, "hand", prism(at("wrist_" + side, (-sign * 0.045, -0.08, 0.032)),
                                 at("wrist_" + side, (-sign * 0.045, -0.12, 0.04)),
                                 X("wrist_" + side), (0.011, 0.01), (0.008, 0.008), 6)),
            (foot, "shin", ellipsoid(P("knee_" + side), M("knee_" + side),
                                     (0.056 * shin, 0.058 * shin, 0.058 * shin), 12, 8)),
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
# What the library offers. A shape or a prop is a mesh in a unit box
# (x and z -0.5..0.5, y 0..1), standing on the floor, sized by the object's
# scale in metres; the person is the rig above, and a crowd is people. Blender
# can make better ones later; the scene file names an asset by id, not by
# geometry, so a scene keeps opening when the geometry improves. `group` is
# where the library lists it.
ASSETS = [
    {"id": "person", "label": "Person", "kind": "person", "group": "people",
     "name": "Person", "colour": "#c9b8a6", "scale": [1.0, 1.0, 1.0],
     "about": "A posable mannequin, 1.8 m. Describe who they are and what they are doing."},
    {"id": "crowd", "label": "Background crowd", "kind": "crowd", "group": "people",
     "name": "Crowd", "colour": "#c9b8a6", "scale": [1.0, 1.0, 1.0],
     "about": "Background people, as many as you like in an area. Describe them "
              "together: who they are, what they wear, what they are doing."},
    {"id": "box", "label": "Box", "kind": "prop", "group": "shape", "name": "Crate",
     "colour": "#8a6a4a", "scale": [0.6, 0.6, 0.6],
     "about": "Any boxy thing: a crate, a bench, a cabinet, a wall. Size it in metres."},
    {"id": "cylinder", "label": "Cylinder", "kind": "prop", "group": "shape",
     "name": "Drum", "colour": "#4f7fb5", "scale": [0.6, 0.9, 0.6],
     "about": "Any round thing: a drum, a gas cylinder, a post, a pipe laid down."},
    {"id": "sphere", "label": "Sphere", "kind": "prop", "group": "shape", "name": "Ball",
     "colour": "#b0342f", "scale": [0.5, 0.5, 0.5],
     "about": "Any round thing: a ball, a globe lamp, a boulder. Size it in metres."},
    {"id": "cone", "label": "Cone", "kind": "prop", "group": "shape", "name": "Cone",
     "colour": "#d9772f", "scale": [0.4, 0.7, 0.4],
     "about": "A traffic cone, a spire, a pile of sand. Size it in metres."},
    {"id": "wedge", "label": "Wedge", "kind": "prop", "group": "shape", "name": "Ramp",
     "colour": "#8b8d91", "scale": [1.0, 0.5, 1.5],
     "about": "A ramp or a slope, rising towards its back. Size it in metres."},
    {"id": "pyramid", "label": "Pyramid", "kind": "prop", "group": "shape",
     "name": "Pyramid", "colour": "#cdbb9a", "scale": [1.0, 0.8, 1.0],
     "about": "A pyramid, a roof, a pointed pile. Size it in metres."},
    {"id": "table", "label": "Table", "kind": "prop", "group": "prop", "name": "Table",
     "colour": "#8a6a4a", "scale": [1.4, 0.75, 0.8],
     "about": "A table on four legs. Say what it is and what is on it."},
    {"id": "chair", "label": "Chair", "kind": "prop", "group": "prop", "name": "Chair",
     "colour": "#8a6a4a", "scale": [0.45, 0.9, 0.45],
     "about": "A chair, its back behind it. Turn it to face the way it is sat in."},
    {"id": "bench", "label": "Bench", "kind": "prop", "group": "prop", "name": "Bench",
     "colour": "#8a6a4a", "scale": [1.8, 0.45, 0.35],
     "about": "A long bench, as at a beer garden table or in a park."},
    {"id": "barrel", "label": "Barrel", "kind": "prop", "group": "prop", "name": "Barrel",
     "colour": "#7a5536", "scale": [0.6, 0.9, 0.6],
     "about": "A barrel with hoops: a beer keg, a rain barrel. Lay it down with Tip."},
    {"id": "tree", "label": "Tree", "kind": "prop", "group": "prop", "name": "Tree",
     "colour": "#4f7d4a", "scale": [2.6, 4.5, 2.6],
     "about": "A leafy tree. Say what kind, and the season."},
    {"id": "bush", "label": "Bush", "kind": "prop", "group": "prop", "name": "Bush",
     "colour": "#4f7d4a", "scale": [1.2, 0.9, 1.0],
     "about": "A shrub or a hedge; stretch it wide for a hedge."},
    {"id": "lamp", "label": "Lamp post", "kind": "prop", "group": "prop",
     "name": "Lamp post", "colour": "#2e2a28", "scale": [0.4, 3.5, 0.4],
     "about": "A street lamp. Say whether it is lit."},
    {"id": "parasol", "label": "Parasol", "kind": "prop", "group": "prop",
     "name": "Parasol", "colour": "#e9e0c6", "scale": [2.4, 2.4, 2.4],
     "about": "A garden or market umbrella on a pole."},
    {"id": "car", "label": "Car", "kind": "prop", "group": "prop", "name": "Car",
     "colour": "#3f6fb0", "scale": [1.8, 1.45, 4.4],
     "about": "A car, its front towards +Z. Say the make, the era and its state."},
    {"id": "fence", "label": "Fence", "kind": "prop", "group": "prop", "name": "Fence",
     "colour": "#e9e2cc", "scale": [3.0, 1.1, 0.1],
     "about": "A fence of posts and rails. Stretch it as long as it needs to be."},
    {"id": "frustum", "label": "Frustum", "kind": "prop", "group": "shape",
     "name": "Lampshade", "colour": "#7d8a94", "scale": [0.4, 0.4, 0.4],
     "about": "A cone with its top cut off: a lampshade, a stool, a pedestal. Roll it "
              "180 for a bucket or pot."},
    {"id": "capsule", "label": "Capsule", "kind": "prop", "group": "shape",
     "name": "Bollard", "colour": "#6b6f76", "scale": [0.3, 1.0, 0.3],
     "about": "A rod with round ends: a bollard, a bolster, a rolled mat, a fire hydrant."},
    {"id": "plane", "label": "Panel", "kind": "prop", "group": "shape", "name": "Rug",
     "colour": "#7a4a5a", "scale": [2.0, 0.02, 1.4],
     "about": "A flat sheet: a rug, a poster, a sign, a screen. Tip it upright for a wall."},
    {"id": "shelves", "label": "Shelves", "kind": "prop", "group": "prop", "name": "Shelves",
     "colour": "#8b7355", "scale": [1.0, 1.8, 0.35],
     "about": "An open unit, open to +z: a bookcase, a shop shelf, a dresser, a rack."},
]
ASSET = {a["id"]: a for a in ASSETS}
ASSET_GROUPS = [("people", "People"), ("shape", "Shapes"), ("prop", "Props")]


def _slab(x0, y0, z0, x1, y1, z1):
    return box((x0, y0, z0), (x1, y1, z1))


def _post(x, z, y0, y1, r, n=8):
    return prism((x, y0, z), (x, y1, z), (1, 0, 0), (r, r), (r, r), n)


def _wedge():
    v = [(-0.5, 0, 0.5), (0.5, 0, 0.5), (0.5, 0, -0.5), (-0.5, 0, -0.5),
         (-0.5, 1, -0.5), (0.5, 1, -0.5)]
    return outward([[v[i] for i in f] for f in ((0, 1, 2, 3), (3, 2, 5, 4), (0, 4, 5, 1),
                                                (0, 3, 4), (1, 5, 2))])


def _pyramid():
    v = [(-0.5, 0, -0.5), (0.5, 0, -0.5), (0.5, 0, 0.5), (-0.5, 0, 0.5), (0, 1, 0)]
    return outward([[v[i] for i in f] for f in ((0, 1, 2, 3), (0, 1, 4), (1, 2, 4),
                                                (2, 3, 4), (3, 0, 4))])


def _legs(top, inset, r):
    return [_slab(sx * (0.5 - inset) - r, 0, sz * (0.5 - inset) - r,
                  sx * (0.5 - inset) + r, top, sz * (0.5 - inset) + r)
            for sx in (1, -1) for sz in (1, -1)]


def _car():
    tyre, glass, trim = "#1f1f22", "#6f8494", "#3a3a3e"
    wheels = []
    for sx in (1, -1):
        for sz in (1, -1):
            wheels.append((prism((sx * 0.49, 0.2, sz * 0.3), (sx * 0.38, 0.2, sz * 0.3),
                                 (0, 1, 0), (0.2, 0.065), (0.2, 0.065), 14), tyre))
            wheels.append((prism((sx * 0.5, 0.2, sz * 0.3), (sx * 0.49, 0.2, sz * 0.3),
                                 (0, 1, 0), (0.1, 0.033), (0.1, 0.033), 10), "#b9bcc0"))
    v = [(-0.44, 0.5, 0.18), (0.44, 0.5, 0.18), (0.44, 0.5, -0.36), (-0.44, 0.5, -0.36),
         (-0.38, 0.86, 0.06), (0.38, 0.86, 0.06), (0.38, 0.86, -0.28), (-0.38, 0.86, -0.28)]
    cab = outward([[v[i] for i in f] for f in ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
                                               (3, 2, 6, 7), (0, 3, 7, 4), (1, 2, 6, 5))])
    # The cab's sides are its windows and its roof is the body's colour. (A
    # pane laid over a face does not work: the painter's sort splits the two
    # into tiles and interleaves them in stripes.)
    roof = [f for f in cab if abs(norm(newell(f))[1]) >= 0.9]
    windows = [f for f in cab if abs(norm(newell(f))[1]) < 0.9]
    body = [(_slab(-0.5, 0.24, -0.48, 0.5, 0.5, 0.48), None),
            (_slab(-0.47, 0.14, -0.46, 0.47, 0.24, 0.46), None),         # the sills
            (_slab(-0.49, 0.14, 0.46, 0.49, 0.27, 0.5), trim),           # bumpers
            (_slab(-0.49, 0.14, -0.5, 0.49, 0.27, -0.46), trim)]
    lights = [(_slab(sx * 0.42 - 0.07, 0.36, 0.48, sx * 0.42 + 0.07, 0.43, 0.492), "#f4efd8")
              for sx in (1, -1)]
    lights += [(_slab(sx * 0.42 - 0.07, 0.36, -0.492, sx * 0.42 + 0.07, 0.42, -0.48), "#b0342f")
               for sx in (1, -1)]
    return body + [(roof, None), (windows, glass)] + lights + wheels


def _barrel():
    out = [(prism((0, 0, 0), (0, 0.5, 0), (1, 0, 0), (0.42, 0.42), (0.5, 0.5), 14), None),
           (prism((0, 0.5, 0), (0, 1, 0), (1, 0, 0), (0.5, 0.5), (0.42, 0.42), 14), None)]
    for y, r in ((0.1, 0.446), (0.3, 0.482), (0.7, 0.482), (0.9, 0.446)):
        out.append((prism((0, y - 0.025, 0), (0, y + 0.025, 0), (1, 0, 0), (r, r), (r, r),
                          14), "#3b2f28"))
    return out


def _fence():
    out = []
    for x in (-0.488, -0.244, 0, 0.244, 0.488):
        out.append((_slab(x - 0.012, 0, -0.35, x + 0.012, 0.92, 0.35), None))
        # A pointed cap: a four-sided prism, its corners on the post's.
        out.append((prism((x, 0.92, 0), (x, 1, 0), (1, 0, 0), (0.017, 0.495),
                          (0.0005, 0.005), 4), None))
    out += [(_slab(-0.5, y, -0.25, 0.5, y + 0.09, 0.25), None) for y in (0.3, 0.72)]
    # Pickets between the posts, on the rails' front.
    for i in range(20):
        x = -0.475 + 0.95 * (i + 0.5) / 20
        if min(abs(x - p) for p in (-0.488, -0.244, 0, 0.244, 0.488)) < 0.025:
            continue
        out.append((_slab(x - 0.008, 0.08, 0.25, x + 0.008, 0.88, 0.45), None))
    return out


def _lamp():
    return [(lathe([(0.34, 0), (0.34, 0.025), (0.2, 0.05), (0.13, 0.08), (0.1, 0.1)], 12), None),
            (_post(0, 0, 0.1, 0.86, 0.1), None),
            (lathe([(0.1, 0.86), (0.2, 0.875), (0.2, 0.89)], 10), None),
            (lathe([(0.2, 0.89), (0.36, 0.95)], 10), "#f2e6b0"),              # the glass
            (lathe([(0.46, 0.95), (0.46, 0.96), (0.08, 0.99), (0.0, 1.0)], 10), None)]


def _parasol():
    out = [(_post(0, 0, 0, 0.93, 0.018), "#8b8d91"),
           (_slab(-0.14, 0, -0.14, 0.14, 0.03, 0.14), "#8b8d91"),
           (lathe([(0.49, 0.76), (0.5, 0.79), (0.3, 0.87), (0.03, 0.925)], 16), None),
           (ellipsoid((0, 0.94, 0), IDENTITY, (0.025, 0.02, 0.025), 6, 4), "#8b8d91")]
    for i in range(8):                    # the ribs under the canopy
        t = 2 * math.pi * (i + 0.5) / 8
        out.append((prism((0, 0.66, 0), (0.46 * math.cos(t), 0.765, 0.46 * math.sin(t)),
                          (0, 1, 0), (0.006, 0.006), (0.004, 0.004), 4), "#8b8d91"))
    return out


def _tree():
    bark = "#5a3e2b"
    out = [(prism((0, 0, 0), (0, 0.55, 0), (1, 0, 0), (0.06, 0.06), (0.035, 0.035), 10), bark),
           (prism((0, 0, 0), (0, 0.04, 0), (1, 0, 0), (0.09, 0.09), (0.06, 0.06), 10), bark)]
    for t, y in ((0.3, 0.42), (2.4, 0.47), (4.3, 0.5)):
        out.append((prism((0, y - 0.06, 0), (0.22 * math.cos(t), y + 0.08, 0.22 * math.sin(t)),
                          (0, 1, 0), (0.018, 0.018), (0.01, 0.01), 6), bark))
    out.append((ellipsoid((0, 0.66, 0), IDENTITY, (0.4, 0.26, 0.4), 12, 7), None))
    for i, (r, y, s) in enumerate(((0.22, 0.62, 0.26), (0.24, 0.72, 0.24), (0.2, 0.6, 0.27),
                                   (0.23, 0.74, 0.25), (0.21, 0.66, 0.27))):
        t = 2 * math.pi * i / 5 + 0.4
        out.append((ellipsoid((r * math.cos(t), y, r * math.sin(t)), IDENTITY,
                              (s, s * 0.8, s), 10, 6), None))
    out.append((ellipsoid((0.04, 0.86, -0.03), IDENTITY, (0.3, 0.14, 0.3), 10, 5), None))
    return out


def _bush():
    out = [(ellipsoid((0, 0.45, 0), IDENTITY, (0.4, 0.45, 0.4), 12, 7), None)]
    for i in range(5):
        t = 2 * math.pi * i / 5
        out.append((ellipsoid((0.22 * math.cos(t), 0.36 + 0.08 * (i % 2), 0.22 * math.sin(t)),
                              IDENTITY, (0.27, 0.3, 0.27), 10, 6), None))
    return out


# Each shape and prop: [(faces, colour or None for the object's own)].
MESHES = {
    "box": [(_slab(-0.5, 0, -0.5, 0.5, 1, 0.5), None)],
    "cylinder": [(cylinder(), None)],
    "sphere": [(ellipsoid((0, 0.5, 0), IDENTITY, (0.5, 0.5, 0.5), 14, 9), None)],
    "cone": [(prism((0, 0, 0), (0, 1, 0), (1, 0, 0), (0.5, 0.5), (0.004, 0.004), 16), None)],
    "wedge": [(_wedge(), None)],
    "pyramid": [(_pyramid(), None)],
    "table": [(_slab(-0.5, 0.94, -0.5, 0.5, 1, 0.5), None)] + [
        (leg, None) for leg in _legs(0.94, 0.06, 0.035)] + [
        (_slab(x0, 0.84, z0, x1, 0.94, z1), None) for x0, z0, x1, z1 in (   # the apron
            (-0.41, 0.425, 0.41, 0.44), (-0.41, -0.44, 0.41, -0.425),
            (0.425, -0.41, 0.44, 0.41), (-0.44, -0.41, -0.425, 0.41))],
    "chair": [(_slab(-0.5, 0.47, -0.5, 0.5, 0.52, 0.5), None)] + [
        (leg, None) for leg in _legs(0.47, 0.06, 0.05)] + [
        # Back posts up from the back legs, a top rail and a slat between.
        (_slab(sx * 0.44 - 0.05, 0.52, -0.49, sx * 0.44 + 0.05, 1, -0.41), None)
        for sx in (1, -1)] + [
        (_slab(-0.39, y0, -0.48, 0.39, y1, -0.43), None) for y0, y1 in ((0.86, 0.98),
                                                                         (0.68, 0.74))] + [
        (_slab(-0.39, 0.14, sz * 0.44 - 0.02, 0.39, 0.18, sz * 0.44 + 0.02), None)
        for sz in (1, -1)],                                               # stretchers
    "bench": [(_slab(-0.5, 0.85, -0.5, 0.5, 1, 0.5), None)] + [
        (_slab(sx * 0.42 - 0.03, 0.06, -0.36, sx * 0.42 + 0.03, 0.85, 0.36), None)
        for sx in (1, -1)] + [
        (_slab(sx * 0.42 - 0.035, 0, -0.45, sx * 0.42 + 0.035, 0.06, 0.45), None)
        for sx in (1, -1)] + [                                            # the feet
        (_slab(-0.39, 0.25, -0.08, 0.39, 0.35, 0.08), None)],            # the stretcher
    "barrel": _barrel(),
    "tree": _tree(),
    "bush": _bush(),
    "lamp": _lamp(),
    "parasol": _parasol(),
    "car": _car(),
    "fence": _fence(),
    "frustum": [(cone(0.32), None)],
    "capsule": [(capsule(), None)],
    "plane": [(_slab(-0.5, 0, -0.5, 0.5, 1, 0.5), None)],
    "shelves": [(_slab(x0, y0, z0, x1, y1, z1), None) for x0, y0, z0, x1, y1, z1 in (
        (-0.5, 0, -0.5, -0.46, 1.0, 0.5), (0.46, 0, -0.5, 0.5, 1.0, 0.5),
        (-0.46, 0, -0.5, 0.46, 1.0, -0.44), (-0.46, 0, -0.5, 0.46, 0.03, 0.5),
        (-0.46, 0.33, -0.5, 0.46, 0.35, 0.5), (-0.46, 0.65, -0.5, 0.46, 0.67, 0.5),
        (-0.46, 0.97, -0.5, 0.46, 1.0, 0.5))],
}
UNIT = {k: [f for faces, _ in parts for f in faces] for k, parts in MESHES.items()}

# Box and Cylinder are stand-ins: the shape holds a place, the name says
# what it is. These start one already named and sized (label, asset, name,
# scale in m, colour).
STAND_INS = [
    ("Cabinet", "box", "Cabinet", [0.9, 1.9, 0.5], "#7b6a58"),
    ("Counter", "box", "Counter", [2.0, 0.9, 0.6], "#9a9186"),
    ("Bed", "box", "Bed", [1.5, 0.55, 2.0], "#cfc7ba"),
    ("Sofa", "box", "Sofa", [2.0, 0.85, 0.9], "#6a5f73"),
    ("Wall", "box", "Wall", [3.0, 2.5, 0.15], "#b8b2a8"),
    ("Door", "box", "Door", [0.9, 2.05, 0.05], "#6b4f36"),
    ("Drum", "cylinder", "Drum", [0.6, 0.9, 0.6], "#4f7fb5"),
    ("Post", "cylinder", "Post", [0.15, 2.5, 0.15], "#5b5b5b"),
    ("Round table", "cylinder", "Round table", [1.0, 0.75, 1.0], "#7b5a3c"),
    ("Tree trunk", "cylinder", "Tree", [0.4, 4.0, 0.4], "#5a4632"),
    ("Tree crown", "sphere", "Tree crown", [2.5, 2.2, 2.5], "#4f7a3c"),
]


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
    elif obj["asset"] == "crowd":
        pieces = crowd_pieces(obj)
        if not pieces:
            return []
    else:
        pieces = []
        for mesh, rgb in MESHES[obj["asset"]]:
            faces = [[apply(rot, (p[0] * sx, p[1] * sy, p[2] * sz)) for p in f]
                     for f in mesh]
            pieces.append(("body", [t for f in faces for t in tiles(f)],
                           hex_rgb(rgb) if rgb else own))
    low = min(p[1] for _, faces, _ in pieces for f in faces for p in f)
    x, y, z = obj["position"]
    shift = (x, y - low, z)
    return [(part, [[add(p, shift) for p in f] for f in faces], rgb)
            for part, faces, rgb in pieces]


# ==================================================================== crowds
# A background crowd is one object: `count` people stood in a `width` x
# `depth` area around its position, each a mannequin of their own - height,
# build, skin, clothes, pose and facing drawn from `seed` - so the picture
# gets many different people, not one person copied. Shuffle is a new seed.
# `wear` is a list of outfit presets' looks (copied on, like a character's)
# the crowd is dressed from; empty is everyday clothes.
CROWD_LIMITS = {"count": (1, 40), "width": (1.0, 30.0), "depth": (0.5, 30.0)}
CROWD_FACING = [("mixed", "Every way"), ("forward", "All one way"),
                ("inward", "Towards the middle"), ("outward", "Away from the middle")]
CROWD_ACTIVITY = [("mixed", "Mixed"), ("standing", "Standing about"),
                  ("walking", "Walking"), ("cheering", "Cheering, raising a glass")]
CROWD_SPACING = 0.6            # m between two people, at least
CROWD_SKIN = ("#f1d3bd", "#e8c0a0", "#d9a67c", "#c68b5e", "#a86f45", "#8a5534",
              "#6b3f25", "#4f2d1b")
CROWD_TOPS = ("t-shirt", "shirt", "sweater", "hoodie", "blouse", "polo shirt",
              "summer dress", "jacket")
CROWD_BOTTOMS = ("jeans", "trousers", "chinos", "skirt", "shorts")
CROWD_COLOURS = ("black", "white", "grey", "navy", "blue", "red", "green", "olive",
                 "khaki", "beige", "brown", "burgundy", "mustard", "teal", "pink", "charcoal")
CROWD_HAIR = ("black", "dark brown", "brown", "light brown", "auburn", "blonde", "grey",
              "red")
CROWD_STYLES = ("short", "short", "shoulder-length", "long", "in a ponytail", "in a bun",
                "buzz cut", "bald", "curly")


def new_crowd():
    return {"count": 12, "width": 6.0, "depth": 3.0, "seed": 1, "facing": "mixed",
            "activity": "mixed", "wear": [], "dressed": ""}


def clean_crowd(d):
    d = d if isinstance(d, dict) else {}
    c = new_crowd()
    c["count"] = int(round(_num(d.get("count"), c["count"], *CROWD_LIMITS["count"])))
    for k in ("width", "depth"):
        c[k] = _num(d.get(k), c[k], *CROWD_LIMITS[k])
    c["seed"] = int(_num(d.get("seed"), 1, 0, 2 ** 31))
    c["facing"] = d.get("facing") if d.get("facing") in dict(CROWD_FACING) else "mixed"
    c["activity"] = (d.get("activity") if d.get("activity") in dict(CROWD_ACTIVITY)
                     else "mixed")
    wear = d.get("wear") if isinstance(d.get("wear"), list) else []
    import studio_imagegen as ig
    c["wear"] = [{k: str(w[k]).strip() for k in ig.OUTFIT_KEYS
                  if isinstance(w.get(k), str) and w[k].strip()}
                 for w in wear[:20] if isinstance(w, dict)]
    c["wear"] = [w for w in c["wear"] if w]
    c["dressed"] = str(d.get("dressed") or "")[:80]
    return c


def _crowd_pose(rng, activity):
    kind = activity if activity != "mixed" else rng.choice(
        ("standing", "standing", "chatting", "walking", "walking", "cheering"))
    if kind == "walking":
        c = dict(POSE_VALUES["walking"])
        if rng.random() < 0.5:               # the other foot forward
            for a, b in (("leg_l_step", "leg_r_step"), ("leg_l_bend", "leg_r_bend"),
                         ("arm_l_raise", "arm_r_raise"), ("arm_l_bend", "arm_r_bend")):
                c[a], c[b] = c.get(b, 0), c.get(a, 0)
    elif kind == "cheering":
        side = rng.choice("lr")
        c = dict(POSE_VALUES["standing"])
        c.update({"arm_%s_raise" % side: rng.uniform(130, 170),
                  "arm_%s_bend" % side: rng.uniform(10, 50),
                  "arm_%s_out" % side: rng.uniform(5, 25)})
    elif kind == "chatting":
        side = rng.choice("lr")
        c = dict(POSE_VALUES["standing"])
        c.update({"arm_%s_raise" % side: rng.uniform(20, 50),
                  "arm_%s_bend" % side: rng.uniform(60, 100)})
    else:
        c = dict(POSE_VALUES["standing"])
        c.update(arm_l_bend=rng.uniform(5, 30), arm_r_bend=rng.uniform(5, 30),
                 leg_l_out=rng.uniform(0, 6), leg_r_out=rng.uniform(0, 6))
    c["head_turn"] = rng.uniform(-30, 30)
    c["head_nod"] = rng.uniform(-10, 12)
    out = pose_controls("standing")
    out.update({k: v for k, v in c.items() if k in out})
    return out, kind


def crowd_members(crowd):
    """The crowd's people: [{"at": (x, z) around the middle, "yaw", "controls",
    "look", "skin", "size"}], the same for the same settings every time."""
    rng = random.Random(crowd["seed"])
    w, d = crowd["width"], crowd["depth"]
    spots = []
    for _ in range(crowd["count"]):
        for _try in range(60):
            x, z = rng.uniform(-w / 2, w / 2), rng.uniform(-d / 2, d / 2)
            if all((x - a) ** 2 + (z - b) ** 2 >= CROWD_SPACING ** 2 for a, b in spots):
                spots.append((x, z))
                break
    out = []
    for x, z in spots:
        controls, kind = _crowd_pose(rng, crowd["activity"])
        facing = crowd["facing"]
        if facing == "forward":
            yaw = rng.uniform(-25, 25)
        elif facing in ("inward", "outward"):
            yaw = math.degrees(math.atan2(-x, -z)) + rng.uniform(-20, 20)
            if facing == "outward":
                yaw += 180
        else:
            yaw = rng.uniform(-180, 180)
        if crowd["wear"]:
            look = dict(rng.choice(crowd["wear"]))
        else:
            top = rng.choice(CROWD_TOPS)
            look = {"top": "%s %s" % (rng.choice(CROWD_COLOURS), top),
                    "footwear": rng.choice(("black shoes", "brown shoes", "white sneakers",
                                            "boots"))}
            if "dress" not in top:
                look["bottom"] = "%s %s" % (rng.choice(CROWD_COLOURS),
                                            rng.choice(CROWD_BOTTOMS))
        look.update(hair=rng.choice(CROWD_HAIR), hair_style=rng.choice(CROWD_STYLES),
                    weight=rng.choice((-1, 0, 0, 0, 1, 1, 2)),
                    stature=rng.choice((-1, 0, 0, 1)))
        out.append({"at": (x, z), "yaw": yaw, "controls": controls, "look": look,
                    "skin": hex_rgb(rng.choice(CROWD_SKIN)), "size": rng.uniform(0.94, 1.04)})
    return out


_CROWD_CACHE = {}


def crowd_pieces(obj):
    """The crowd's faces around the origin, [(part, faces, rgb)], each person
    on the floor; part is "m<n>", person by person, so each casts their own
    shadow. Cached by the object's settings: a drag of another object
    redraws a crowd many times unchanged."""
    key = json.dumps([obj["crowd"], obj["rotation"][0], obj["scale"][0]], sort_keys=True)
    hit = _CROWD_CACHE.get(key)
    if hit is not None:
        return hit
    turn = obj["rotation"][0]
    rot = euler(turn)
    pieces = []
    for i, m in enumerate(crowd_members(obj["crowd"])):
        shape = body_shape(m["look"])
        k = obj["scale"][0] * shape["height"] * m["size"]
        own = person_pieces(m["controls"], euler(turn + m["yaw"]), shape, outfit(m["look"]))
        faces = [(fs, rgb or m["skin"]) for _, fs, rgb in own]
        low = min(p[1] * k for fs, _ in faces for f in fs for p in f)
        at = apply(rot, (m["at"][0], 0, m["at"][1]))
        shift = (at[0], -low, at[2])
        pieces += [("m%d" % i, [[add(mul(p, k), shift) for p in f] for f in fs], rgb)
                   for fs, rgb in faces]
    if len(_CROWD_CACHE) > 64:
        _CROWD_CACHE.clear()
    _CROWD_CACHE[key] = pieces
    return pieces


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
    return {"version": VERSION, "details": details, "frame": "portrait",
            "pose_strength": POSE_STRENGTH, "depth_strength": DEPTH_STRENGTH,
            "frame_keep": FRAME_KEEP, "face_likeness": FACE_LIKENESS,
            "real_faces": REAL_FACES,
            "camera": {"target": [0.0, 1.0, 0.0], "yaw": 0.0, "pitch": 6.0,
                       "distance": 4.2, "lens": 35.0},
            "room": new_room(), "objects": [], "enrich": new_enrich()}


def new_enrich():
    """Enrich's memory: `added` details are in the words, `placed` became
    objects (the object's description carries the words), `seen` were
    offered and skipped, `never` the user asked not to be offered again."""
    return {"added": [], "placed": [], "seen": [], "never": []}


def new_object(asset_id, taken=(), stand_in=None):
    """A fresh object of an asset; `stand_in`, a label from STAND_INS, starts
    it named, sized and coloured as that thing."""
    a = dict(ASSET[asset_id])
    for label, asset, nm, scale, colour in STAND_INS:
        if label == stand_in and asset == asset_id:
            a.update(name=nm, scale=scale, colour=colour)
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
        obj["face"] = ""
    elif a["kind"] == "crowd":
        obj["crowd"] = new_crowd()
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
        o["face"] = str(d.get("face") or "")
        at = d.get("look_at")
        if isinstance(at, (list, tuple)) and len(at) == 3:
            o["look_at"] = _vec(at, [0.0, 1.6, 0.0], -100, 100)
    if "crowd" in base:
        o["crowd"] = clean_crowd(d.get("crowd"))
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
    # A scene saved before the maps had `redraw` (denoise from the frame); it
    # opens with the maps and no frame, as a new one does.
    s["pose_strength"] = _num(d.get("pose_strength"), POSE_STRENGTH, 0.0, 1.0)
    s["depth_strength"] = _num(d.get("depth_strength"), DEPTH_STRENGTH, 0.0, 1.0)
    s["frame_keep"] = _num(d.get("frame_keep"), FRAME_KEEP, 0.0, FRAME_KEEP_MAX)
    s["face_likeness"] = _num(d.get("face_likeness"), FACE_LIKENESS, *FACE_LIKENESS_RANGE)
    s["real_faces"] = bool(d.get("real_faces", REAL_FACES))
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
            if o.get("face") and not os.path.isfile(o["face"]):
                problems.append("%s's face picture %s is missing, so their face is drawn "
                                "from the words." % (o["name"], os.path.basename(o["face"])))
    given = d.get("enrich") if isinstance(d.get("enrich"), dict) else {}
    for key in s["enrich"]:
        raw = given.get(key) if isinstance(given.get(key), list) else []
        s["enrich"][key] = [str(x).strip()[:ENRICH_CHARS] for x in raw
                            if isinstance(x, str) and x.strip()][-ENRICH_KEEP:]
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


# ================================================================== history
# Undo is whole-scene snapshots, not inverse operations: a scene is a few KB
# of JSON (a crowd is its settings, not its people), and a snapshot cannot
# get out of step with an edit the way a hand-written inverse can. Each step
# is named from what differs between it and the one before (`change_label`),
# so no edit in the window has to say what it is.

OBJECT_CHANGES = [             # (field, how the step is named), first match wins
    ("look_at", "Point %s's eyes"), ("pose", "Pose %s"), ("look", "Change %s's look"), ("character", "Change %s's look"),
    ("crowd", "Change the crowd %s"), ("colour", "Colour %s"), ("position", "Move %s"),
    ("rotation", "Turn %s"), ("scale", "Size %s"), ("name", "Rename %s"),
    ("description", "Describe %s"),
]
SCENE_CHANGES = [("camera", "Move the camera"), ("room", "Change the floor and walls"),
                 ("frame", "Change the frame"), ("details", "Edit the scene details"),
                 ("pose_strength", "Change the pose strength"),
                 ("depth_strength", "Change the layout strength"),
                 ("frame_keep", "Change how much frame is kept")]


def change_label(before, after):
    """-> a few words for what changed from scene `before` to scene `after`."""
    was = {o["id"]: o for o in before.get("objects", [])}
    now = {o["id"]: o for o in after.get("objects", [])}
    added = [o for i, o in now.items() if i not in was]
    gone = [o for i, o in was.items() if i not in now]
    if added or gone:
        verb, some = ("Add", added) if added and not gone else \
            ("Delete", gone) if gone and not added else ("Change", added + gone)
        return "%s %s" % (verb, some[0]["name"] if len(some) == 1 else
                          "%d objects" % len(some))
    edited = [(was[i], o) for i, o in now.items() if o != was[i]]
    if len(edited) > 1:
        return "Change %d objects" % len(edited)
    if edited:
        a, b = edited[0]
        for key, words in OBJECT_CHANGES:
            if a.get(key) != b.get(key):
                return words % (a["name"] if key == "name" else b["name"])
        return "Change %s" % b["name"]
    if [o["id"] for o in before.get("objects", [])] != \
            [o["id"] for o in after.get("objects", [])]:
        return "Reorder the objects"
    for key, words in SCENE_CHANGES:
        if before.get(key) != after.get(key):
            return words
    return "Change the scene"


class History:
    """A scene's undo and redo: `steps` of (label, snapshot, selection),
    `at` the one the scene is in now. `record` after an edit, `undo` /
    `redo` / `go` to move, each handing back (scene, selection) to put up.
    The selection is the one the step was made with, so undoing a move
    selects what moved back. Snapshots are JSON text: compared as strings,
    and no step shares a list with the live scene."""
    LIMIT = 200

    def __init__(self, scene, sel=None, label="Start"):
        self.reset(scene, sel, label)

    @staticmethod
    def snap(scene):
        return json.dumps(scene, sort_keys=True)

    def reset(self, scene, sel=None, label="Start"):
        self.steps = [(label, self.snap(scene), sel)]
        self.at = 0
        self.saved = self.steps[0][1]

    def mark_saved(self, scene):
        self.saved = self.snap(scene)

    def unsaved(self, scene):
        return self.snap(scene) != self.saved

    def record(self, scene, sel=None):
        """-> the new step's label, or None when nothing changed. A step
        recorded after an undo throws the redo steps away, as everywhere."""
        now = self.snap(scene)
        prev = self.steps[self.at][1]
        if now == prev:
            return None
        label = change_label(json.loads(prev), scene)
        del self.steps[self.at + 1:]
        self.steps.append((label, now, sel))
        if len(self.steps) > self.LIMIT:
            del self.steps[:len(self.steps) - self.LIMIT]
        self.at = len(self.steps) - 1
        return label

    def can_undo(self):
        return self.at > 0

    def can_redo(self):
        return self.at < len(self.steps) - 1

    def labels(self):
        return [label for label, _, _ in self.steps]

    def go(self, i):
        """-> (scene, selection) at step `i`, or None if that is where it is.
        The selection is from the step nearest `i` that is crossed."""
        i = max(0, min(len(self.steps) - 1, i))
        if i == self.at:
            return None
        crossed = self.steps[i + 1] if i < self.at else self.steps[i]
        self.at = i
        return json.loads(self.steps[i][1]), crossed[2]

    def undo(self):
        return self.go(self.at - 1) if self.can_undo() else None

    def redo(self):
        return self.go(self.at + 1) if self.can_redo() else None


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
        if obj["asset"] == "crowd":          # each person their own shadow
            members = {}
            for piece in pieces:
                members.setdefault(piece[0], []).append(piece)
            for group in members.values():
                polys += shadow_polys(group, cam, floor)
        else:
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


def _write(data, prefix, folder=None):
    """PNG bytes -> `<scenes>/renders/<prefix>_<hash>.png`, its path. Named by
    content, so History's settings keep pointing at the picture that was
    sent, and the same picture twice is one file."""
    folder = folder or os.path.join(scenes_dir(), "renders")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "%s_%s.png" % (prefix, hashlib.sha1(data).hexdigest()[:16]))
    if not os.path.isfile(path):
        tmp = path + ".part"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    return path


def write_reference(scene, folder=None):
    """Render the frame and write it (`_write`) -> its path."""
    return _write(png(scene), "scene", folder)


# ==================================================================== maps
# What the picture is made from when the model's workflow has a ControlNet:
# not the grey frame - image to image copies its blocky mannequins at any
# denoise low enough to keep the layout - but what the frame means. Where
# each body's joints are (`pose_png`: OpenPose, drawn as the Image Studio's
# stick figure is) and how far every pixel is from the camera (`depth_png`).
# Both are seen through the one camera, so they fall exactly where the
# viewport's frame shows the mannequins and the props.
OPENPOSE_OF_COCO = {0: 0, 1: 15, 2: 14, 3: 17, 4: 16, 5: 5, 6: 2, 7: 6, 8: 3, 9: 7,
                    10: 4, 11: 11, 12: 8, 13: 12, 14: 9, 15: 13, 16: 10}
FACE_UNIT = 0.032              # m: half the gap between the eyes, studio_pose.FACE's unit
SEEN_FACING = -0.25            # a head point is drawn when it faces the camera this much
HIDDEN_BEHIND = 0.3            # m of something nearer at its pixel that hides a point


def rigs(obj):
    """The people in an object as [(skeleton, k, shift)]: a skeleton point p
    is at p * k + shift in the world, as `painted_pieces` places their faces.
    One for a person, one per member for a crowd, none for a prop."""
    def placed(controls, root, shape, look, k, x, y, z):
        low = min(p[1] for _, faces, _ in person_pieces(controls, root, shape, outfit(look))
                  for f in faces for p in f) * k
        return skeleton(controls, root, shape), k, (x, y - low, z)
    x, y, z = obj["position"]
    if obj["asset"] == "person":
        look = obj.get("look") or {}
        shape = body_shape(look)
        return [placed(obj["pose"]["controls"], euler(*obj["rotation"]), shape, look,
                       obj["scale"][0] * shape["height"], x, y, z)]
    if obj["asset"] != "crowd":
        return []
    turn = obj["rotation"][0]
    out = []
    for m in crowd_members(obj["crowd"]):
        shape = body_shape(m["look"])
        at = apply(euler(turn), (m["at"][0], 0, m["at"][1]))
        out.append(placed(m["controls"], euler(turn + m["yaw"]), shape, m["look"],
                          obj["scale"][0] * shape["height"] * m["size"],
                          x + at[0], y, z + at[2]))
    return out


# ==================================================================== eyes
# A person can be given a point to look at (`look_at`, world metres): the
# head's Turn and Look down controls are solved so the eyes meet it, and
# solved again whenever the person moves, turns or is re-posed. The head
# only, within its sliders' range: a point behind them leaves the head
# turned as far as it goes. A crowd has no eyes to point.
EYES = (0.0, 0.07, 0.09)       # in the head's frame, m


def eye_point(obj):
    """Where a person's eyes are in the world."""
    sk, k, shift = rigs(obj)[0]
    hp, hm = sk["head"]
    return add(mul(add(hp, apply(hm, EYES)), k), shift)


def aim_head(obj):
    """Turn a person's head to their `look_at`; False when they have none."""
    target = obj.get("look_at")
    if obj.get("asset") != "person" or not target:
        return False
    ctl = obj["pose"]["controls"]
    sk, k, shift = rigs(obj)[0]
    shape = body_shape(obj.get("look"))
    root = euler(*obj["rotation"])
    lo_t, hi_t = CONTROL_RANGE["head_turn"]
    lo_n, hi_n = CONTROL_RANGE["head_nod"]
    turn = nod = 0.0
    for _ in range(4):
        # The neck and head together are not exactly one yaw then one
        # pitch, so measure where the eyes point and correct the error.
        ctl["head_turn"], ctl["head_nod"] = turn, nod
        sk = skeleton(ctl, root, shape)
        hp, hm = sk["head"]
        cm = sk["chest"][1]
        eye = add(mul(add(hp, apply(hm, EYES)), k), shift)
        want = apply(transpose(cm), sub(target, eye))
        have = apply(transpose(cm), column(hm, 2))
        if dot(want, want) < 1e-6:
            break
        yaw = lambda v: math.degrees(math.atan2(v[0], v[2]))                  # noqa
        pitch = lambda v: math.degrees(math.atan2(-v[1], math.hypot(v[0], v[2])))  # noqa
        d_yaw = (yaw(want) - yaw(have) + 180) % 360 - 180
        turn = max(lo_t, min(hi_t, turn + d_yaw))
        nod = max(lo_n, min(hi_n, nod + pitch(want) - pitch(have)))
    ctl["head_turn"], ctl["head_nod"] = round(turn, 1), round(nod, 1)
    return True


# The Look at ring's head poses: (key, label, dx, dy) where (dx, dy) is the
# item's place on the ring on screen (right, down +), and the head turns and
# nods that way as the viewer sees it.
HEAD_POSES = [("ahead", "Ahead", 0, 0), ("up", "Up", 0, -1), ("up_right", "Up right", 1, -1),
              ("right", "Right", 1, 0), ("down_right", "Down right", 1, 1),
              ("down", "Down", 0, 1), ("down_left", "Down left", -1, 1),
              ("left", "Left", -1, 0), ("up_left", "Up left", -1, -1)]
HEAD_TURN, HEAD_NOD_UP, HEAD_NOD_DOWN = 60.0, 30.0, 35.0


def head_pose(scene, obj, key):
    """Put a person's head in one of HEAD_POSES, left and right as the
    camera sees them; any point they looked at is dropped."""
    _, _, dx, dy = next(h for h in HEAD_POSES if h[0] == key)
    w, h = frame_size(scene)
    cam = Camera(scene["camera"], w, h)
    their_left = apply(euler(*obj["rotation"]), (1, 0, 0))
    side = 1 if dot(their_left, cam.r) > 0 else -1     # their left on screen right?
    k = 0.75 if dx and dy else 1.0                     # a diagonal, a little of each
    ctl = obj["pose"]["controls"]
    ctl["head_turn"] = dx * side * HEAD_TURN * k
    ctl["head_nod"] = (HEAD_NOD_DOWN if dy > 0 else HEAD_NOD_UP) * dy * k
    ctl["head_tilt"] = 0.0
    obj.pop("look_at", None)
    obj["pose"]["preset"] = ""


def aim_heads(scene):
    """Every person with a point to look at, looking at it."""
    for obj in scene["objects"]:
        aim_head(obj)


def pose_figures(scene, width=None, height=None):
    """Every person the camera sees, as `studio_pose.render_figures` takes
    them, far to near: {"points": the 18 OpenPose points as fractions of the
    frame, None where not seen; "face": the 68 dots, or []; "depth": m}.

    Which head points are seen is decided the way DWPose would find them:
    the nose and eyes only on the side of the head facing the camera, the
    far ear hidden in profile. The face dots are a real face's, turned with
    the head in 3D, drawn whenever the nose is seen - in profile too, as
    DWPose draws them: a figure without them comes back seen from behind
    (studio_pose.face_points)."""
    if width is None:
        width, height = frame_size(scene)
    cam = Camera(scene["camera"], width, height)
    # What stands in front: a point with a surface more than HIDDEN_BEHIND
    # nearer than it at its pixel is hidden, as a photo would hide it - a
    # crowd member's limbs drawn through the person in front of them read
    # as that person's own and turned them round (2026-09-25). The body's
    # own surface is nearer than its joints by less than that.
    k_depth = DEPTH_EDGE / float(max(width, height))
    dw, dh = max(64, int(round(width * k_depth))), max(64, int(round(height * k_depth)))
    zb = depth_values(scene, dw, dh)

    def hidden(p):
        c = cam.to_camera(p)
        x, y = int(cam.to_screen(c)[0] * dw / width), int(cam.to_screen(c)[1] * dh / height)
        if not (0 <= x < dw and 0 <= y < dh) or zb[y * dw + x] <= 0:
            return False
        return 1.0 / zb[y * dw + x] < c[2] - HIDDEN_BEHIND
    out = []
    for obj in scene["objects"]:
        for sk, k, shift in rigs(obj):
            def world(p, k=k, shift=shift):
                return add(mul(p, k), shift)
            hp, hm = sk["head"]
            coco = {i: world(sk[j][0]) for i, j in KP_JOINTS.items()}
            coco.update({i: world(add(hp, apply(hm, v))) for i, v in KP_HEAD.items()})
            to_cam = norm(sub(cam.eye, world(hp)))
            fwd, side = column(hm, 2), column(hm, 0)     # the face looks +Z; +X is their left
            faces_way = {0: fwd, 1: norm(add(fwd, mul(side, 0.35))),
                         2: norm(add(fwd, mul(side, -0.35))), 3: side, 4: mul(side, -1)}
            pts = [None] * 18
            for i, p in coco.items():
                if i in faces_way and dot(faces_way[i], to_cam) <= SEEN_FACING:
                    continue
                s = cam.project(p)
                if s and not hidden(p):
                    pts[OPENPOSE_OF_COCO[i]] = [s[0] / width, s[1] / height]
            if pts[2] and pts[5]:                  # the neck, as OpenPose makes it
                pts[1] = [(pts[2][0] + pts[5][0]) / 2, (pts[2][1] + pts[5][1]) / 2]
            if not any(p and 0 <= p[0] <= 1 and 0 <= p[1] <= 1 for p in pts):
                continue
            face = []
            if pts[0]:
                for u, v in studio_pose.FACE:
                    local = (u * FACE_UNIT, KP_HEAD[1][1] - v * FACE_UNIT,
                             0.10 - 0.012 * u * u)
                    s = cam.project(world(add(hp, apply(hm, local))))
                    if s:
                        face.append((s[0] / width, s[1] / height))
            out.append({"points": pts, "face": face,
                        "depth": cam.to_camera(world(hp))[2]})
    out.sort(key=lambda f: -f["depth"])
    return out


def pose_png(scene):
    """The OpenPose picture of everyone in the frame, or None if no one is."""
    w, h = frame_size(scene)
    figures = pose_figures(scene, w, h)
    return studio_pose.render_figures(figures, w, h) if figures else None


def _fill_depth(zb, width, height, pts):
    """A convex face's 1/z into the z-buffer `zb`, nearest kept. `pts` are
    (sx, sy, 1/z): 1/z is linear across the screen for a flat face, so each
    pixel is a sum, not a division."""
    x0, y0, q0 = pts[0]
    best = None
    for i in range(1, len(pts) - 1):
        (x1, y1, q1), (x2, y2, q2) = pts[i], pts[i + 1]
        det = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if best is None or abs(det) > abs(best[0]):
            best = (det, x1, y1, q1, x2, y2, q2)
    det, x1, y1, q1, x2, y2, q2 = best
    if abs(det) < 1e-6:
        return                                   # edge-on
    b = ((q1 - q0) * (y2 - y0) - (q2 - q0) * (y1 - y0)) / det
    c = ((x1 - x0) * (q2 - q0) - (x2 - x0) * (q1 - q0)) / det
    a = q0 - b * x0 - c * y0
    ys = [p[1] for p in pts]
    top = max(0, int(math.ceil(min(ys) - 0.5)))
    bottom = min(height - 1, int(math.floor(max(ys) - 0.5)))
    edges = [(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
    edges = [(p, q) if p[1] <= q[1] else (q, p) for p, q in edges if p[1] != q[1]]
    for y in range(top, bottom + 1):
        yc = y + 0.5
        xs = [p[0] + (yc - p[1]) * (q[0] - p[0]) / (q[1] - p[1])
              for p, q in edges if p[1] <= yc < q[1]]
        if len(xs) < 2:
            continue
        xa = max(0, int(math.ceil(min(xs) - 0.5)))
        xb = min(width - 1, int(math.floor(max(xs) - 0.5)))
        q = a + b * (xa + 0.5) + c * yc
        for i in range(y * width + xa, y * width + xb + 1):
            if q > zb[i]:
                zb[i] = q
            q += b


def depth_values(scene, width, height):
    """1/z (1/m) of the nearest surface at each pixel, row by row, 0 where
    the camera sees only sky: the floor, the walls that face into the room,
    and every object, as `render` draws them but without the shadows."""
    cam = Camera(scene["camera"], width, height)
    room = scene.get("room") or new_room()
    faces = []
    if cam.eye[1] > 0:
        r = FLOOR_REACH
        faces.append([(-r, 0, -r), (r, 0, -r), (r, 0, r), (-r, 0, r)])
    if room["walls"]:
        faces += [quad for quad, origin, _ in walls(room)
                  if dot(newell(quad), sub(origin, cam.eye)) < 0]
    for obj in scene["objects"]:
        for _, fs, _ in painted_pieces(obj):
            faces += [f for f in fs if dot(newell(f), sub(centroid(f), cam.eye)) < 0]
    zb = [0.0] * (width * height)
    for f in faces:
        c = clip_near([cam.to_camera(p) for p in f])
        if len(c) >= 3:
            _fill_depth(zb, width, height, [cam.to_screen(p) + (1.0 / p[2],) for p in c])
    return zb


def depth_png(scene, edge=DEPTH_EDGE):
    """The frame as a depth map, the kind a depth ControlNet was trained on
    (Depth Anything's): grey by nearness - 1/z from the farthest thing seen
    (black) to the nearest (white), the sky black. Smaller than the frame,
    with the same shape: the ControlNet scales it to the picture."""
    w, h = frame_size(scene)
    k = edge / float(max(w, h))
    dw, dh = max(64, int(round(w * k))), max(64, int(round(h * k)))
    zb = depth_values(scene, dw, dh)
    seen = [q for q in zb if q > 0]
    lo, hi = (min(seen), max(seen)) if seen else (0.0, 1.0)
    span = (hi - lo) or 1.0
    grey = bytes(0 if q <= 0 else int(round(255 * (q - lo) / span)) for q in zb)
    rgb = bytearray(dw * dh * 3)
    for i in range(3):
        rgb[i::3] = grey
    return rgb_png(bytes(rgb), dw, dh)


MAP_KINDS = ("pose", "composition", "source")


def scene_maps(scene, takes, folder=None):
    """Draw and write what the picture is made from, for a model whose
    workflows take the reference kinds `takes` -> ({kind: path}, notes).

    - `pose`: the pose map, when the strength is above 0 and anyone is in
      the frame.
    - `composition`: the depth map, when its strength is above 0.
    - `source`: the grey frame, image to image, when some of it is kept - or
      for a model with neither ControlNet input, which has only the frame to
      go on (`FALLBACK_KEEP`, the old default)."""
    notes = []
    maps = {}
    controlled = "pose" in takes or "composition" in takes
    if "pose" in takes and scene["pose_strength"] > 0:
        data = pose_png(scene)
        if data:
            maps["pose"] = _write(data, "pose", folder)
        elif any(o["asset"] in ("person", "crowd") for o in scene["objects"]):
            notes.append("No one is in the frame, so no pose map was sent.")
    if "composition" in takes and scene["depth_strength"] > 0:
        maps["composition"] = _write(depth_png(scene), "depth", folder)
    if "source" in takes and (scene["frame_keep"] > 0 or not controlled):
        maps["source"] = write_reference(scene, folder)
    if not maps:
        notes.append("Pose, layout and frame are all off: the picture has only the words "
                     "to go on.")
    return maps, notes


# ==================================================================== words

class Words:
    def __init__(self):
        self.text = ""
        self.notes = []


def placement(scene, obj):
    """Where an object sits in the frame, in words, or None outside it. In
    it is any of its box on screen: its middle alone left a person framed
    head and shoulders out of the words, the middle being their hips below
    the frame. Left, centre or right is read off the part that is seen."""
    w, h = frame_size(scene)
    cam = Camera(scene["camera"], w, h)
    lo, hi = bounds(obj)
    corners = [cam.project((x, y, z)) for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
               for z in (lo[2], hi[2])]
    corners = [p for p in corners if p]
    if not corners:
        return None
    x0, x1 = max(0, min(p[0] for p in corners)), min(w, max(p[0] for p in corners))
    y0, y1 = max(0, min(p[1] for p in corners)), min(h, max(p[1] for p in corners))
    if x0 >= x1 or y0 >= y1:
        return None
    mid = ((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2)
    p = (0, 0, cam.to_camera(mid)[2])
    x = (x0 + x1) / 2 / w
    where = ["left of frame" if x < 1 / 3 else "right of frame" if x > 2 / 3
             else "centre of frame"]
    d = scene["camera"]["distance"]
    if p[2] < d * 0.75:
        where.append("foreground")
    elif p[2] > d * 1.35:
        where.append("background")
    return where


def _way(cam, fwd, at):
    """(cos to the camera, frame side) of a direction `fwd` from `at`, level."""
    fwd = norm((fwd[0], 0, fwd[2]))
    to_cam = norm((cam.eye[0] - at[0], 0, cam.eye[2] - at[2]))
    return dot(fwd, to_cam), "right" if dot(fwd, cam.r) > 0 else "left"


def facing(scene, obj):
    """Which way a person faces, as the picture will show it."""
    w, h = frame_size(scene)
    cam = Camera(scene["camera"], w, h)
    c, side = _way(cam, apply(euler(*obj["rotation"]), (0, 0, 1)), obj["position"])
    if c > 0.7:
        return "facing the camera"
    if c < -0.7:
        return "back to the camera"
    if c > 0.2:
        return "three-quarter view, turned to frame %s" % side
    if c < -0.2:
        return "turned away, towards frame %s" % side
    return "in profile, facing frame %s" % side


# ------------------------------------------------------------ a person, in words
# What the controls know that the look's words do not: how the person stands
# (`posture_words`), where their head looks against their body (`gaze_words`)
# and how much of them the frame shows (`framing_words`). Read off the posed
# skeleton rather than the sliders, so combinations come out as what they
# look like - a raised arm bent back is a hand above the head either way -
# and they agree with the pose map the ControlNet is given, which is drawn
# from the same skeleton: words that disagree with a ControlNet fight it.
LEG_POSES = ("walking", "crouching", "kneeling", "sitting")   # named: the legs are said
HEAD_TOP = 0.2                 # m from the head joint (the top of the neck) to the crown
BOTH_ARMS = {"hanging relaxed at the side": "arms relaxed at the sides",
             "raised above the head": "both arms raised above the head",
             "reaching forward at shoulder height": "both arms reaching forward",
             "stretched out to the side": "arms stretched out to the sides",
             "bent, the hand in front of the chest": "both arms bent, hands in front of the chest",
             "bent, the hand in front of the waist": "both arms bent, hands in front of the waist",
             "held forward": "both arms held forward",
             "swinging forward": "both arms swinging forward",
             "held out from the body": "arms held out from the body",
             "swinging back": "both arms swinging back"}


def _arm(sk, side, bent, hip_y):
    """One arm's place, in words, from the skeleton (+Z forward, +X their left)."""
    s, w = sk["shoulder_" + side][0], sk["wrist_" + side][0]
    ahead, out = w[2] - s[2], abs(w[0]) - abs(s[0])
    if w[1] > sk["head"][0][1] + HEAD_TOP:
        return "raised above the head"
    if w[1] > s[1] - 0.12:
        return ("reaching forward at shoulder height" if ahead >= out
                else "stretched out to the side")
    if ahead > 0.15:
        if bent:
            return "bent, the hand in front of the %s" % (
                "chest" if w[1] > (s[1] + hip_y) / 2 else "waist")
        return "held forward" if ahead > 0.3 else "swinging forward"
    if out > 0.18:
        return "held out from the body"
    if ahead < -0.15:
        return "swinging back"
    return "hanging relaxed at the side"


def posture_words(obj):
    """How a person stands, as phrases: the torso, the arms, the legs (unless
    a named pose that says them - kneeling, sitting - is chosen), and the
    head's nod and tilt (unless the look's Gaze says where they look).
    Left and right are theirs, as a caption says "her right hand"."""
    c = obj["pose"]["controls"]
    look = obj.get("look") or {}
    g = lambda k: float(c.get(k, 0) or 0)                 # noqa: E731
    sk = skeleton(c, IDENTITY, body_shape(look))
    out = []
    bend = g("bend")
    if bend >= 50:
        out.append("bent well forward at the waist")
    elif bend >= 18:
        out.append("leaning forward")
    elif bend <= -12:
        out.append("leaning back")
    if abs(g("lean")) >= 10:
        out.append("leaning to their %s" % ("left" if g("lean") > 0 else "right"))
    if abs(g("twist")) >= 20:
        out.append("shoulders turned to their %s" % ("left" if g("twist") > 0 else "right"))
    hip_y = sk["pelvis"][0][1]
    arms = {side: _arm(sk, side, g("arm_%s_bend" % side) >= 60, hip_y) for side in "rl"}
    if arms["r"] == arms["l"]:
        out.append(BOTH_ARMS[arms["r"]])
    else:
        out += ["right arm " + arms["r"], "left arm " + arms["l"]]
    if obj["pose"].get("preset") not in LEG_POSES:
        sl, sr = g("leg_l_step"), g("leg_r_step")
        bl, br = g("leg_l_bend"), g("leg_r_bend")
        if abs(sl - sr) >= 25:
            out.append("mid-stride, %s foot forward" % ("left" if sl > sr else "right"))
        elif min(g("leg_l_out"), g("leg_r_out")) >= 12:
            out.append("feet planted wide apart")
        elif abs(bl - br) >= 12:
            out.append("weight on the %s leg, the other knee relaxed"
                       % ("right" if bl > br else "left"))
    if not str(look.get("gaze") or "").strip():
        if g("head_nod") >= 20:
            out.append("looking down")
        elif g("head_nod") <= -15:
            out.append("looking up")
    if abs(g("head_tilt")) >= 15:
        out.append("head tilted")
    return out


def gaze_words(scene, obj):
    """Where a person's head looks, as the picture shows it, when that is
    not the way their body faces (a head turned back over the shoulder,
    towards the camera): '' otherwise, or when the look's Gaze says it."""
    if str((obj.get("look") or {}).get("gaze") or "").strip():
        return ""
    w, h = frame_size(scene)
    cam = Camera(scene["camera"], w, h)
    rig = rigs(obj)[0]
    hp, hm = rig[0]["head"]
    at = add(mul(hp, rig[1]), rig[2])
    body, _ = _way(cam, apply(euler(*obj["rotation"]), (0, 0, 1)), obj["position"])
    head, side = _way(cam, column(hm, 2), at)
    if abs(head - body) < 0.35:
        return ""
    if head > 0.7:
        return "head turned towards the camera"
    if head < -0.2:
        return "head turned away from the camera"
    return "looking towards frame %s" % side


def framing_words(scene, obj):
    """How much of a person the frame shows: 'whole figure in view', 'seen
    from the knees up', ... - '' when none of them is in it."""
    w, h = frame_size(scene)
    cam = Camera(scene["camera"], w, h)
    sk, k, shift = rigs(obj)[0]

    def seen(*joints):
        for j in joints:
            p = cam.project(add(mul(sk[j][0], k), shift))
            if p is None or not (0 <= p[0] <= w and 0 <= p[1] <= h):
                return False
        return True
    top = seen("head")
    if seen("ankle_l", "ankle_r"):
        return "whole figure in view" if top else "head out of the top of the frame"
    if seen("knee_l") or seen("knee_r"):
        return "seen from the knees up"
    if seen("pelvis"):
        return "seen from the waist up"
    if seen("shoulder_l") or seen("shoulder_r"):
        return "head and shoulders"
    return ""


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
        elif obj["asset"] == "crowd":
            n = len(crowd_members(obj["crowd"]))
            about.append("a background crowd of %d %s" % (n, "person" if n == 1 else "people"))
        about += where
        if obj["asset"] == "crowd":
            c = obj["crowd"]
            about.append({"mixed": "facing every way", "inward": "gathered, facing each other",
                          "outward": "facing outwards"}.get(c["facing"])
                         or facing(scene, obj))
            if c["activity"] != "mixed":
                about.append({"standing": "standing about", "walking": "walking",
                              "cheering": "cheering, raising a glass"}[c["activity"]])
        posture = ""
        if obj["asset"] == "person":
            about.append(facing(scene, obj))
            about += [x for x in (gaze_words(scene, obj), framing_words(scene, obj)) if x]
            preset = obj["pose"].get("preset")
            if preset in POSE_NAMES and preset != "standing":
                about.append(POSE_NAMES[preset].lower())
            posture = ", ".join(posture_words(obj))
            posture = posture[:1].upper() + posture[1:]
        line = "%s (%s)" % (obj["name"].strip() or ASSET[obj["asset"]]["label"],
                            ", ".join(about))
        desc = obj["description"].strip()
        look = look_text(obj) if obj["asset"] == "person" else ""
        said = ". ".join(x for x in (look, posture, desc) if x)
        parts.append(line + (": " + said if said else ""))
        said = look or desc
        if not said:
            out.notes.append("%s has no description; the picture has only its shape and "
                             "name to go on." % obj["name"])
    for detail in (scene.get("enrich") or {}).get("added") or []:
        if detail.strip():
            parts.append(detail.strip())
    parts.append(camera_words(scene))
    out.text = "\n".join(p if p.endswith((".", "!", "?")) else p + "." for p in parts)
    return out


def people(scene):
    return [o for o in scene["objects"] if o["asset"] == "person"]


# ==================================================================== faces
# The face pass (studio_imagegen) redraws every face in the finished picture
# close up. A scene knows who each face is: where each person's face falls in
# the frame, and their own words, so a face is redrawn as that person rather
# than as the whole prompt's blend of everyone - and, when the person has a
# face picture (their own, or their character's identity's), redrawn to be
# that face (PuLID), at the scene's `face_likeness`.
FACE_UP = 0.1                  # m from the head joint (the top of the neck) to the face's middle
# The part of the frame a person's face picture is drawn into in the picture
# itself (PuLID's attention mask), in head heights (head joint to crown)
# around the face: wide enough for the hair, short of the next person's head.
FACE_REGION = {"side": 1.1, "above": 0.5, "below": 0.6}


def face_picture(obj, characters=None, identities=None):
    """-> (path, where it came from) of the face a person is drawn with, or
    ('', ''): the person's own, else their character's identity's first
    reference picture."""
    if obj.get("face") and os.path.isfile(obj["face"]):
        return obj["face"], "their own face picture"
    rec = (characters or {}).get(obj.get("character")) if obj.get("character") else None
    ident = (identities or {}).get(rec.get("identity")) if rec and rec.get("identity") else None
    if ident and ident.get("use_references", True):
        for path in ident.get("references") or []:
            if os.path.isfile(path):
                return path, "%s's profile" % ident["name"]
    return "", ""


def face_photos(obj, characters=None, identities=None):
    """Every photo of a person's face there is, their own face picture
    first, then their character's identity's references: the real-face
    paste picks the one whose head is turned most like the drawn one's."""
    out = []
    if obj.get("face") and os.path.isfile(obj["face"]):
        out.append(obj["face"])
    rec = (characters or {}).get(obj.get("character")) if obj.get("character") else None
    ident = (identities or {}).get(rec.get("identity")) if rec and rec.get("identity") else None
    if ident and ident.get("use_references", True):
        out += [p for p in ident.get("references") or [] if os.path.isfile(p) and p not in out]
    return out


def face_targets(scene, characters=None, identities=None):
    """Every person whose face is in the frame: {"id", "name", "at": [x, y]
    (the face's middle, as fractions of the frame), "region": [x0, y0, x1,
    y1] (their head and hair, the same way), "words" (what they look
    like, their own description, the scene's details), "face" (a picture's
    path or ''), "from", "photos" (every photo of their face, `face_photos`)}. Crowds are left out: their faces are background."""
    w, h = frame_size(scene)
    cam = Camera(scene["camera"], w, h)
    details = scene["details"].strip().rstrip(".")
    out = []
    for obj in people(scene):
        sk, k, shift = rigs(obj)[0]
        hp, hm = sk["head"]
        up = column(hm, 1)
        p = cam.project(add(mul(add(hp, mul(up, FACE_UP)), k), shift))
        if p is None or not (0 <= p[0] <= w and 0 <= p[1] <= h):
            continue
        neck = cam.project(add(mul(hp, k), shift))
        crown = cam.project(add(mul(add(hp, mul(up, HEAD_TOP)), k), shift))
        head = (math.hypot(crown[0] - neck[0], crown[1] - neck[1])
                if neck and crown else 0.05 * h)
        top, bottom = min(neck[1], crown[1]) if neck and crown else p[1], (
            max(neck[1], crown[1]) if neck and crown else p[1])
        region = [max(0.0, (p[0] - FACE_REGION["side"] * head) / w),
                  max(0.0, (top - FACE_REGION["above"] * head) / h),
                  min(1.0, (p[0] + FACE_REGION["side"] * head) / w),
                  min(1.0, (bottom + FACE_REGION["below"] * head) / h)]
        said = [x.strip().rstrip(".") for x in (look_text(obj), obj["description"], details)
                if x and x.strip()]
        face, source = face_picture(obj, characters, identities)
        out.append({"id": obj["id"], "name": obj["name"],
                    "at": [round(p[0] / w, 4), round(p[1] / h, 4)],
                    "region": [round(x, 4) for x in region],
                    "words": ". ".join(said) + ("." if said else ""),
                    "face": face, "from": source,
                    "photos": face_photos(obj, characters, identities)})
    return out


def generation(scene, maps, characters=None, identities=None):
    """What the Image Studio is handed: the words for its Scene field, and
    the settings Generate adds to the form's (the frame's size, the maps
    from `scene_maps` as references with their strengths, the denoise when
    the frame is one, and the whole scene for History). -> (words, extra).

    `pose` and `composition` are always laid over the form's, None when not
    sent, so the form's own drawn figure (and its hands' words) does not
    ride along with a scene's.

    A scene with people in it says every person's look in their own line,
    so the form's one person is blanked for this job - said twice, the
    picture gets an extra person or a blend of two, and so are the form's
    item pictures (compose matches them to the form's clothes, now blank;
    no workflow takes one yet, and the words carry the items). The identity
    whose LoRA carries a scene character's face rides along as
    `scene_identities`, for the form to add to its own. `characters` and
    `identities` are {id: record}.

    A scene with people always gets the face pass, told who each face is
    (`scene_faces`, from `face_targets`): each redrawn in their own words,
    and to their face picture's likeness where they have one."""
    import studio_imagegen as ig
    w, h = frame_size(scene)
    words = scene_text(scene)
    s, _ = clean_scene(scene)
    extra = {"width": w, "height": h, "scene_layout": copy.deepcopy(s),
             "references": dict(maps), "pose": None, "composition": None}
    if "pose" in maps:
        extra["pose"] = {"strength": round(s["pose_strength"], 3)}
    if "composition" in maps:
        extra["composition"] = {"strength": round(s["depth_strength"], 3)}
    if "source" in maps:
        keep = s["frame_keep"]
        if "pose" not in maps and "composition" not in maps:
            keep = max(keep, FALLBACK_KEEP)
        extra["denoise"] = round(1 - keep, 3)
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
        extra["face_detail"] = True
        extra["scene_faces"] = {"likeness": round(s["face_likeness"], 3),
                                "real": s["real_faces"],
                                "people": face_targets(s, characters, identities)}
    return words, extra


# ---------------------------------------------------------------- enrich
# One lived-in detail at a time, from the model on the LLM PC, around the
# people and never between or on them. A detail with a body (a table, a
# stein, a passer-by) is placed in the scene as a box, cylinder or person,
# its words that object's description; light, haze and the like stay words.
# The model names a place (`WHERE`), never coordinates: where that is comes
# from the people and the camera here. The user adds, skips or bans each;
# the scene remembers all of it so the next offer is a new one.

ENRICH_KEEP = 40          # per list, newest kept
ENRICH_CHARS = 400
ENRICH_ANGLES = [
    "something left behind or in use on a surface (a glass, food, a bag, a jacket)",
    "a photographic imperfection (a person half cut off at the frame edge, uneven "
    "light, a soft out-of-focus foreground shape, glare, haze)",
    "people in the far background going about their own business, softly out of focus",
    "wear and weather on the place itself (worn ground, stains, puddles, scuffs, "
    "peeling paint, fallen leaves)",
    "light: where it comes from, its colour and how it falls unevenly on things",
    "signage, writing or decoration that belongs to this exact place, slightly worn",
    "a small sign of what happened a minute ago (a spill, a chair pushed back, "
    "crumpled napkins)",
]
# where -> (metres further from the camera than the people, side: -1 left,
# 1 right, 0 centred, None alternating; height of its lowest point)
WHERE = {"behind": (2.2, 0, 0.0), "behind_left": (1.6, -1, 0.0),
         "behind_right": (1.6, 1, 0.0), "far_background": (6.0, None, 0.0),
         "left": (0.0, -1, 0.0), "right": (0.0, 1, 0.0),
         "foreground_left": (-1.2, -1, 0.0), "foreground_right": (-1.2, 1, 0.0),
         "above": (1.2, 0, 2.4)}
SHAPES = ("box", "cylinder", "sphere", "cone", "frustum", "capsule", "wedge", "pyramid",
          "plane", "table", "chair", "bench", "shelves", "barrel", "tree", "bush", "lamp",
          "parasol", "car", "fence", "person", "none")
ENRICH_SYSTEM = """You help stage a photograph. You suggest ONE believable, lived-in detail \
that would make the scene feel like a real snapshot taken in that moment, not people \
pasted into a themed background.

Rules:
- Favour photographic imperfections and environmental storytelling over decorative \
clutter. A jacket over a bench, an empty glass, a stranger half cut off by the frame \
edge or uneven warm light is worth more than one more themed prop.
- Enrich AROUND the people. Never put anything or anyone between them, in front of \
their faces or in their hands, and never change who they are, their clothes or their \
pose.
- Be concrete and visual: materials, colours, state, how the light falls on it, how \
sharp or blurred it is. One or two sentences, 25 to 60 words, written as a line of \
an image prompt, not advice.
- Suggest something new: nothing already in the scene, nothing offered before, \
nothing like what the user banned.

The scene is blocked out in 3D with simple shapes, and your detail is placed in it:
- "shape": "box" (a bench, crate, stall, cabinet), "cylinder" (a barrel, post, \
stein, lamp, basket), "sphere" (a ball, bush, boulder, globe lamp), "cone" (a \
traffic cone, small tree, tent), "frustum" (a lampshade, stool, pedestal), \
"capsule" (a bollard, bolster, rolled mat), "wedge" (a ramp, slope, awning), \
"pyramid" (a roof, a pointed pile), "plane" (a rug, poster, sign board, puddle), \
"table", "chair", "bench", "shelves", "barrel" (a keg), "tree", "bush" (a shrub, a \
hedge), "lamp" (a street lamp), "parasol", "car", "fence", \
"person" (a whole passer-by, never a part of one), or \
"none" for what has no body of its own (light, haze, glare, stains, a hand or \
shoulder cut off by the frame edge, strings of bunting too thin to block out).
- "size": [width, height, depth] in metres, true to life (a table is about \
[1.8, 0.75, 0.8], a stein [0.1, 0.2, 0.1], a person [0.5, 1.75, 0.3]).
- "where": one of %s.
- "name": two or three words. "colour": its main colour as #rrggbb. \
"facing" (a person only): "camera", "side" or "away".

Answer with JSON only: {"detail": "<the prompt line>", "why": "<under 15 words>", \
"shape": "...", "name": "...", "size": [w, h, d], "colour": "#rrggbb", "where": "...", \
"facing": "..."}""" % ", ".join(WHERE)


def new_suggestion(detail, why=""):
    return {"detail": detail, "why": why, "shape": "none", "name": "", "size": None,
            "colour": "", "where": "behind", "facing": "camera"}


def enrich_angle(scene):
    """The kind of detail to ask for this time: a rotation, so the offers
    vary instead of settling on steins forever."""
    e = scene.get("enrich") or new_enrich()
    n = sum(len(e.get(k) or []) for k in ("added", "placed", "seen"))
    return ENRICH_ANGLES[n % len(ENRICH_ANGLES)]


def enrich_messages(scene):
    e = scene.get("enrich") or new_enrich()
    said = []
    for title, keys in (("Already added", ("added", "placed")),
                        ("Offered before, skipped", ("seen",)),
                        ("Never suggest these or anything like them", ("never",))):
        items = [x for k in keys for x in e.get(k) or []][-20:]
        if items:
            said.append("%s:\n%s" % (title, "\n".join("- " + x for x in items)))
    placed = {x.lower() for x in e.get("placed") or []}
    busy = {}
    for o in scene["objects"]:
        if o["description"].strip().lower() in placed:
            spot = " ".join(placement(scene, o) or [])
            if spot:
                busy[spot] = busy.get(spot, 0) + 1
    if busy:
        said.append("Places already used by added details (choose a different \"where\"): %s"
                    % ", ".join("%s (%d)" % kv for kv in sorted(busy.items())))
    user = ("The scene as the image prompt describes it:\n%s\n\n%s\n\nThis time, look for: %s."
            % (scene_text(scene).text, "\n\n".join(said) or "Nothing has been added yet.",
               enrich_angle(scene)))
    return [{"role": "system", "content": ENRICH_SYSTEM},
            {"role": "user", "content": user}]


def read_suggestion(text):
    """-> a suggestion (`new_suggestion`'s keys) from the model's reply, or
    None. Takes the JSON asked for, cleaned field by field, or failing that
    the first real line of prose, as words only."""
    text = re.sub(r"(?s)<think>.*?(</think>|$)", "", text or "").strip()
    m = re.search(r"(?s)\{.*\}", text)
    d = None
    if m:
        try:
            d = json.loads(m.group(0))
        except ValueError:
            d = None
    if isinstance(d, dict) and isinstance(d.get("detail"), str) and d["detail"].strip():
        why = d.get("why") if isinstance(d.get("why"), str) else ""
        out = new_suggestion(_tidy(d["detail"]), why.strip()[:160])
        if d.get("shape") in SHAPES:
            out["shape"] = d["shape"]
        if isinstance(d.get("name"), str):
            out["name"] = _tidy(d["name"])[:40]
        size = d.get("size")
        if (isinstance(size, list) and len(size) == 3
                and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in size)):
            out["size"] = [min(max(float(v), 0.05), 8.0) for v in size]
        if isinstance(d.get("colour"), str) and re.fullmatch(r"#[0-9a-fA-F]{6}", d["colour"]):
            out["colour"] = d["colour"].lower()
        if d.get("where") in WHERE:
            out["where"] = d["where"]
        if d.get("facing") in ("camera", "side", "away"):
            out["facing"] = d["facing"]
        return out
    for line in text.splitlines():
        line = _tidy(line)
        if len(line) > 12 and not line.startswith(("{", "}", "```")):
            return new_suggestion(line)
    return None


def _tidy(s):
    s = re.sub(r"^(?:suggested enrichment|detail|suggestion)\s*:\s*", "", s.strip(), flags=re.I)
    s = re.sub(r"^\s*(?:[-*\u2022]|\d+[.)])\s*", "", s)
    return s.strip().strip('"\'\u201c\u201d').strip()[:ENRICH_CHARS]


def _same(a, b):
    key = lambda s: re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
    return key(a) == key(b)


def suggest(scene, llm, tries=2):
    """-> one new suggestion from `llm` (anything with `chat(messages,
    max_tokens=)` returning an OpenAI reply). Raises RuntimeError when
    nothing usable came back."""
    e = scene.get("enrich") or new_enrich()
    old = [x for k in e for x in e[k]]
    for _ in range(tries):
        reply = llm.chat(enrich_messages(scene), max_tokens=900)
        msg = ((reply.get("choices") or [{}])[0].get("message") or {})
        got = read_suggestion(msg.get("content") or "")
        if got and not any(_same(got["detail"], x) for x in old):
            return got
    raise RuntimeError("The model offered nothing new for this scene. Try again, "
                       "or add a line to Scene details for it to build on.")


def _flat(v):
    return norm((v[0], 0.0, v[2]))


def _overlaps(a, b, gap=0.15):
    (alo, ahi), (blo, bhi) = a, b
    return (alo[0] < bhi[0] + gap and blo[0] < ahi[0] + gap and
            alo[2] < bhi[2] + gap and blo[2] < ahi[2] + gap)


def _screen_rect(cam, obj):
    """The frame rectangle an object's box covers, or None if behind the eye."""
    lo, hi = bounds(obj)
    pts = [cam.project((x, y, z)) for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
           for z in (lo[2], hi[2])]
    if any(p is None for p in pts):
        return None
    return (min(p[0] for p in pts), min(p[1] for p in pts),
            max(p[0] for p in pts), max(p[1] for p in pts))


def _hides(a, b, share=0.25):
    """Do frame rectangles a and b overlap by more than `share` of the smaller?"""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return False
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return w * h > share * max(small, 1e-9)


def _middle_in_frame(cam, obj, w, h):
    """An added detail's middle is in the frame. Stricter than `placement`,
    which counts any part on screen: a table 18 px in from the edge is not
    a detail anyone sees."""
    lo, hi = bounds(obj)
    p = cam.project(mul(add(lo, hi), 0.5))
    return p is not None and 0 <= p[0] <= w and 0 <= p[1] <= h


def _place(scene, sug):
    """-> a new object for the suggestion, placed clear of everything else
    and inside the frame, or None when it has no body or no such place was
    found (the caller keeps it as words then). It is not added to the scene.

    `where` is measured from the people: their middle, how far they spread
    across the frame, and the camera's own forward and right on the floor,
    so "behind_left" is behind them and to the left as the picture sees it
    from any orbit. A place that is taken is pushed further out; one that
    leaves the frame is pulled in towards the people."""
    if sug.get("shape") not in ASSET:
        return None
    w, h = frame_size(scene)
    cam = Camera(scene["camera"], w, h)
    fwd, right = _flat(cam.f), _flat(cam.r)
    # The subjects: a passer-by Enrich placed before is scenery, or the
    # people's middle drifts off towards the far background with each one.
    placed = [x.lower() for x in (scene.get("enrich") or {}).get("placed") or []]
    folks = [o for o in people(scene) if o["description"].strip().lower() not in placed]
    if folks:
        centre = tuple(sum(o["position"][i] for o in folks) / len(folks) for i in range(3))
    else:
        centre = (cam.target[0], 0.0, cam.target[2])
    spread = max([abs(dot(sub(o["position"], centre), right)) for o in folks] or [0.0]) + 0.6
    obj = new_object(sug["shape"], scene["objects"])
    size = sug.get("size")
    if sug["shape"] == "person":
        k = min(max((size[1] if size else 1.75) / 1.8, 0.5), 1.1)
        obj["scale"] = [round(k, 3)] * 3
    elif size:
        obj["scale"] = list(size)
    if sug.get("name"):
        taken, name, n = {o["name"] for o in scene["objects"]}, sug["name"], 2
        while name in taken:
            name, n = "%s %d" % (sug["name"], n), n + 1
        obj["name"] = name
    obj["description"] = sug["detail"]
    if sug.get("colour"):
        obj["colour"] = sug["colour"]
    where = sug.get("where") if sug.get("where") in WHERE else "behind"
    ahead, side, lift = WHERE[where]
    half = max(obj["scale"][0], obj["scale"][2]) / 2
    if side is None:
        side = -1 if len(scene["objects"]) % 2 else 1
        across = side * (0.3 * spread + 1.0)
    else:
        across = side * (spread + half + 0.4)

    def turn():
        """Face as asked, measured on the line to the lens from where it stands."""
        to_cam = _flat(sub(cam.eye, obj["position"]))
        across_ = (-to_cam[2], 0.0, to_cam[0])
        if dot(across_, right) < 0:
            across_ = mul(across_, -1)
        face = {"side": across_, "away": mul(to_cam, -1)}.get(sug.get("facing"), to_cam)
        obj["rotation"] = [round(math.degrees(math.atan2(face[0], face[2])), 1), 0.0, 0.0]
    # A narrow frame has little room beside the people, and more further
    # back: try the place, then the place pulled in towards them, then
    # both a step further from the camera. A taken spot is pushed out.
    others = [bounds(o) for o in scene["objects"]]
    subjects = [r for r in (_screen_rect(cam, o) for o in folks) if r]

    def clear():
        """Free on the floor, and in the frame neither hiding behind a
        subject nor standing in front of one."""
        if any(_overlaps(bounds(obj), b) for b in others):
            return False
        mine = _screen_rect(cam, obj)
        return mine is not None and not any(_hides(mine, r) for r in subjects)
    backs = (0.0, 0.8, 1.6, 2.6, 4.0) if ahead >= 0 else (0.0, 0.4, 0.9)
    for back in backs:
        for pull in (1.0, 0.85, 0.7, 0.55, 0.4):
            for nudge in range(4):
                a, s = ahead + back, across * pull
                if s:
                    s += math.copysign(0.6 * nudge, s)
                else:
                    a += 0.6 * nudge
                pos = add(add(centre, mul(fwd, a)), mul(right, s))
                obj["position"] = [round(pos[0], 2), lift, round(pos[2], 2)]
                turn()
                if not clear():
                    continue
                if _middle_in_frame(cam, obj, w, h):
                    return obj
                break          # free but out of frame: further out will not help
    return None


def place_suggestion(scene, sug):
    """-> a new object for the suggestion, placed clear of everyone and in
    frame; else the same thing behind the people on its side, else far off;
    else None (a thing with no body, or no room anywhere: it stays words).
    The object is not added to the scene."""
    where = sug.get("where") or "behind"
    side = "_left" if where.endswith("left") else "_right" if where.endswith("right") else ""
    for w in dict.fromkeys((where, "behind" + side, "far_background")):
        obj = _place(scene, dict(sug, where=w))
        if obj:
            return obj
    return None


def enrich_answer(scene, detail, answer):
    """Remember the user's answer: 'add' (in the words), 'placed' (an object
    now carries it), 'skip' or 'never'."""
    e = scene.setdefault("enrich", new_enrich())
    key = {"add": "added", "placed": "placed", "skip": "seen", "never": "never"}[answer]
    e[key] = (e.get(key, []) + [detail.strip()[:ENRICH_CHARS]])[-ENRICH_KEEP:]


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
    not show, left at rest); `scale`, the photo's pixels a metre at the
    person, and `pelvis`, where in the photo their pelvis falls (x, y px) -
    how big and where they stand, for `picture_scene`."""

    def __init__(self, controls, yaw, error, unseen, scale=0.0, pelvis=(0.0, 0.0)):
        self.controls, self.yaw, self.error, self.unseen = controls, yaw, error, unseen
        self.scale, self.pelvis = scale, pelvis

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
    placed = _laid_over(model, target)
    return placed and placed[0]


def _laid_over(model, target):
    """-> (the misfit, the scale, where the pelvis falls) for the model laid
    best over `target`: the scale in target units a metre, the pelvis (the
    model's origin) as a target (x, y). None when no positive scale fits."""
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
    return err / sw, s, (dx - s * mx, dy - s * my)


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
    err, scale, (px, py) = (_laid_over(pose_points(state, state["_yaw"], shape), target)
                            or (0.0, 0.0, (0.0, 0.0)))
    controls = {k: float(round(state[k])) for k in CONTROL_KEYS}
    return PoseFit(controls, float(round(state["_yaw"])), math.sqrt(err), unseen,
                   scale * tall, (px * tall, py * tall))


# ==================================================== a scene from a picture
# **From a picture…** makes a whole scene from one photo: every person the
# pose finder sees (the most prominent `PICTURE_PEOPLE`), each posed by
# `fit_pose` and stood where they are in it, and - when the vision model can
# look - the setting, the floor and each person's clothes and doing in words.
# Where a person stands comes from how big they are in the photo: the fit's
# scale (pixels a metre, which a bent or turned pose does not fool the way a
# box's height would) is how far they are from a lens of `PICTURE_LENS`, and
# the camera is level at the height that puts every pelvis at its own pose's
# height above the floor. A flat photo cannot say its lens or its tilt, so
# both are a start to adjust, as each pose is.
PICTURE_PEOPLE = 10
PICTURE_LENS = 35.0
PICTURE_SLOTS = ("subject", "age", "hair", "hair_style", "facial_hair", "top", "bottom",
                 "outerwear", "footwear", "accessories", "expression")
PICTURE_QUESTION = (
    "This photo, %d x %d pixels, is to be rebuilt as a 3D scene and photographed again. "
    "It shows %d people. Answer with JSON only, no other words, in this shape:\n"
    '{"setting": "one or two sentences for the picture: the place, the light, the time '
    'of day, what is around - not the people", "floor": "what the ground is, in a few '
    'words", "people": [{"box": [left, top, right, bottom], "name": "a short name for '
    'them, like Accordion player", "doing": "what they are doing and holding, in a few '
    'words", "subject": "a man, a woman, a young woman, an older man...", "age": "in '
    'their 30s...", "hair": "its colour", "hair_style": "", "facial_hair": "", "top": "", '
    '"bottom": "", "outerwear": "", "footwear": "", "accessories": "", "expression": ""}]}\n'
    "One entry for every person, including anyone seen from behind or cut off at an "
    "edge; the box is where they are in the photo, in its pixels. Leave a field empty "
    "when the photo does not show it.")
# How far apart (in photo widths) a described person's middle may be from a
# found one's and still be them. The vision model places a person across
# the frame well and up and down loosely, so across counts for more.
PICTURE_MATCH = 0.12


def picture_question(folk, width, height):
    """What the vision model is asked about a photo with `folk` in it. It
    is not given our boxes to number: a 7B model numbered them in its own
    order and put the band's clothes on the audience. It gives its own
    boxes, and `match_people` pairs them."""
    return PICTURE_QUESTION % (int(width or 0), int(height or 0), len(folk))


def picture_answer(text, width=0, height=0):
    """The vision model's reply -> {"setting", "floor", "people": [{...,
    "box": [x0, y0, x1, y1] as fractions of the photo, or absent}]}, every
    other value a string; what cannot be read is left out, never raised: a
    scene without the words is still a scene. A box of numbers no bigger
    than 100 in a bigger photo is read as percentages."""
    out = {"setting": "", "floor": "", "people": []}
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        data = json.loads(m.group(0)) if m else {}
    except ValueError:
        data = {}
    if not isinstance(data, dict):
        return out
    for k in ("setting", "floor"):
        if isinstance(data.get(k), str):
            out[k] = data[k].strip()
    for p in data.get("people") or []:
        if not isinstance(p, dict):
            continue
        said = {k: _said_word(k, v) for k, v in p.items() if isinstance(v, str)}
        said = {k: v for k, v in said.items() if v}
        box = p.get("box") or p.get("bbox_2d") or p.get("bbox")
        try:
            box = [float(v) for v in box][:4] if len(box) >= 4 else None
        except (TypeError, ValueError):
            box = None
        if box and box[2] > box[0] and box[3] > box[1]:
            pct = max(box) <= 100 and max(width, height) > 100
            w, h = (100.0, 100.0) if pct else (float(width or 1), float(height or 1))
            said["box"] = [box[0] / w, box[1] / h, box[2] / w, box[3] / h]
        out["people"].append(said)
    return out


UNSAID = ("", "none", "n/a", "na", "unknown", "not visible", "not shown", "unclear", "-")


def _said_word(key, value):
    """One field of the reply, cleaned: the example's trailing "..." (a 7B
    model copies "in their 30s..." back), nothing for "none" and the like,
    and a person as "a man", not "man", as the look's Who slot says it."""
    v = value.strip().rstrip(".…").strip()
    if v.lower() in UNSAID:
        return ""
    if key == "subject" and not re.match(r"(?i)(a|an|the)\s", v):
        v = ("an " if v[:1].lower() in "aeiou" else "a ") + v
    return v


def match_people(folk, said, width, height):
    """{n: what was said of the found person n (1 is folk[0])}: each
    described person paired with the found one whose middle is nearest,
    nearest pairs first, one each; too far apart is no one."""
    w, h = float(width or 1), float(height or 1)

    def mid(b):
        return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2

    pairs = []
    for n, p in enumerate(folk, 1):
        fx, fy = mid(p["box"])
        for i, s in enumerate(said):
            if "box" not in s:
                continue
            sx, sy = mid(s["box"])
            d = abs(fx / w - sx) + 0.35 * abs(fy / h - sy)
            if d <= PICTURE_MATCH:
                pairs.append((d, n, i))
    out, used = {}, set()
    for _, n, i in sorted(pairs):
        if n not in out and i not in used:
            out[n] = said[i]
            used.add(i)
    return out


def picture_frame(width, height):
    """The frame whose shape is nearest the photo's."""
    aspect = float(width) / max(1.0, float(height))
    return min(FRAMES, key=lambda f: abs(math.log(f[2] / float(f[3]) / aspect)))[0]


def pelvis_height(controls, yaw=0.0, shape=None):
    """How far the pelvis is above the floor in this pose (m)."""
    shape = shape or REST_SHAPE
    low = min(p[1] for _, faces, _ in person_pieces(controls, euler(yaw), shape)
              for f in faces for p in f)
    return -low * shape["height"]


def picture_scene(data, read=None, lens=PICTURE_LENS):
    """The pose finder's JSON for a photo (and `picture_answer` of it, or
    None) -> (scene, notes). Raises ValueError when no one in it can be
    posed: a scene of nobody is not what was asked for."""
    pw, ph = float(data.get("width") or 0), float(data.get("height") or 0)
    folk = photo_people(data)
    if not folk or pw <= 0 or ph <= 0:
        raise ValueError("The pose finder found no one in the picture.")
    read = read or {"setting": "", "floor": "", "people": []}
    words = match_people(folk[:PICTURE_PEOPLE], read["people"], pw, ph)
    notes = []
    if len(folk) > PICTURE_PEOPLE:
        notes.append("%d smaller people were left out; add a background crowd for them."
                     % (len(folk) - PICTURE_PEOPLE))
    scene = new_scene(read["setting"])
    scene["frame"] = picture_frame(pw, ph)
    fw, fh = FRAME_SIZES[scene["frame"]]
    g = max(fw / pw, fh / ph)                     # the frame covers the photo, centred
    ox, oy = (fw - pw * g) / 2, (fh - ph * g) / 2
    cam = Camera({"target": [0, 0, 0], "yaw": 0, "pitch": 0, "distance": 1,
                  "lens": lens}, fw, fh)
    placed = []
    for n, p in enumerate(folk[:PICTURE_PEOPLE], 1):
        try:
            fit = fit_pose(p["points"], None, p["box"])
        except ValueError:
            notes.append("Person %d shows too little of themselves to pose, so is left "
                         "out." % n)
            continue
        if fit.scale <= 0:
            continue
        sx, sy = fit.pelvis[0] * g + ox, fit.pelvis[1] * g + oy
        depth = cam.k / (fit.scale * g)
        ray = add(cam.f, add(mul(cam.r, (sx - fw / 2) / cam.k),
                             mul(cam.u, -(sy - fh / 2) / cam.k)))
        rel = mul(ray, depth)                         # from the eye, the eye at 0
        placed.append((n, fit, rel, depth, pelvis_height(fit.controls, fit.yaw)))
    if not placed:
        raise ValueError("No one in the picture shows enough of themselves to pose from.")
    # Near people count for more: a far one's height is a few pixels, and
    # the photo's tilt (taken as level) moves it the most.
    weight = [fit.scale ** 2 for _, fit, *_ in placed]
    eye_y = sum(w * (ph_ - rel[1]) for w, (_, _, rel, _, ph_) in zip(weight, placed))         / sum(weight)
    eye_y = min(30.0, max(0.2, eye_y))
    dist = placed[0][3]                   # the camera turns about the most prominent
    scene["camera"] = {"target": [0.0, round(eye_y, 3), 0.0], "yaw": 0.0, "pitch": 0.0,
                       "distance": round(dist, 3), "lens": lens}
    if read["floor"]:
        scene["room"]["floor"]["prompt"] = read["floor"]
    rough = []
    for n, fit, rel, _, _ in sorted(placed, key=lambda t: t[3], reverse=True):
        obj = new_object("person", scene["objects"])
        said = words.get(n) or {}
        if said.get("name"):
            obj["name"], k = said["name"][:40], 2
            while any(o["name"] == obj["name"] for o in scene["objects"]):
                obj["name"], k = "%s %d" % (said["name"][:40], k), k + 1
        elif len(placed) > 1:
            obj["name"] = "Person %d" % n
        obj["description"] = said.get("doing", "")
        obj["look"] = clean_look({k: said[k] for k in PICTURE_SLOTS if k in said})
        x, z = rel[0], dist + rel[2]
        obj["position"] = [round(x, 3), 0.0, round(z, 3)]
        toward = math.degrees(math.atan2(0.0 - x, dist - z))
        obj["rotation"][0] = round((toward + fit.yaw + 180) % 360 - 180, 1)
        obj["pose"] = {"preset": "", "controls": fit.controls}
        scene["objects"].append(obj)
        if fit.rough:
            rough.append(obj["name"])
    if rough:
        notes.append("The pose is rough for %s: adjust by hand." % ", ".join(rough))
    return scene, notes
