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
- **A prop's picture** (`picture`): instead of its grey shape, a prop can
  stand in the scene as a picture of what it is - one the Image Studio makes
  from its name and description (`picture_settings`), cut out of its plain
  background (`key_background`), or a PNG from disk. It is drawn as an
  upright card at the prop's place, always turned to the camera (`CutMap`),
  so the frame shows the thing itself rather than a box standing for it.
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
- **Instance, part, normal and world-position buffers** (`id_render`): the
  same rendered polygons, once more, as per-pixel buffers instead of
  shading - which body a pixel belongs to, which of that body's pieces,
  which way its surface faces, and the exact 3D point it is - and the
  sidecar that names the instance and part numbers. Not sent to
  generation; a render artifact for masking a region, regenerating one
  part, or compositing after the fact.

No tkinter here; `apps/image_studio/scene/ui.py` is the window. Stdlib only.
"""

import array
import bisect
import copy
import hashlib
import itertools
import json
import math
import os
import random
import re
import struct
import zlib

import core.icons as studio_icons
import apps.image_studio.scene.mannequin as mq
import apps.image_studio.scene.pose as studio_pose

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
    ("landscape_3x2", "3:2 landscape 1216 x 832", 1216, 832),
    ("portrait_2x3", "3:2 portrait 832 x 1216", 832, 1216),
    ("landscape_4x3", "4:3 landscape 1152 x 896", 1152, 896),
]
FRAME_SIZES = {k: (w, h) for k, _, w, h in FRAMES}
# A camera's format (`imagegen.CAMERA_FORMATS`) -> the frames it shoots,
# the way it is usually held first: a phone upright (Portrait is 4:3's
# 896 x 1152), a cinema camera wide (Landscape is 16:9's 1344 x 768).
FORMAT_FRAMES = {"1:1": ("square",), "3:2": ("landscape_3x2", "portrait_2x3"),
                 "4:3": ("portrait", "landscape_4x3"), "16:9": ("landscape",)}


def camera_frame(fmt, current):
    """The frame a camera of format `fmt` shoots, held the way `current`
    is (a portrait frame stays upright): `current` itself when it already
    fits, or when the camera has no format."""
    frames = FORMAT_FRAMES.get(fmt)
    if not frames or current in frames:
        return current
    w, h = FRAME_SIZES.get(current, FRAME_SIZES["portrait"])
    upright = [k for k in frames if FRAME_SIZES[k][1] > FRAME_SIZES[k][0]]
    return upright[0] if h > w and upright else frames[0]
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
# Off by default: a new, unproven layer (2026-09-28). Each named character's own
# words go to their own masked region of the picture instead of one shared
# paragraph, from the instance buffer id_render already computes - see
# character_masks/scene_text_regional. A workflow without regional_conditioning
# ignores it; so does a scene with fewer than two named characters.
REGIONAL_PROMPTING = False
FACE_LIKENESS_RANGE = (0.3, 0.95)
DEPTH_EDGE = 512               # px on the depth map's long edge; the ControlNet scales it
# A person's head shape (`mq.HEAD_SHAPE`) reaches the picture in the face
# pass, as their head's depth over the face's crop: at FACE_EDIT the jaw is
# hundreds of pixels. Its own strength, not the frame's depth: in a close-up
# the whole frame's depth strong enough to carry a jaw also drew the
# mannequin's egg head and ringed neck (2026-09-26).
HEAD_DEPTH = 0.35
HEAD_DEPTH_RANGE = (0.0, 0.8)
# How much of the depth strength a shot gets, by how tall the
# biggest face is in the frame (head, neck to crown, over the frame's
# height): a close-up gets little, the face pass carries its head.
FRAMING = [(0.22, "close-up", 0.45), (0.1, "medium shot", 0.75), (0.0, "wide shot", 1.0)]
DEPTH_RELIEF = 3.0             # each body's own depth stretched this much in the depth map
DEPTH_CLIP = (0.02, 0.995)     # the depth map's grey spans these shares of what is seen

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
        ("hand_" + _side, "wrist_%s_bend" % _side, "Bend wrist (to the palm +)", -70, 70),
        ("hand_" + _side, "fingers_%s_curl" % _side, "Curl fingers", 0, 90),
        ("hand_" + _side, "fingers_%s_spread" % _side, "Spread fingers", 0, 20),
        ("hand_" + _side, "thumb_%s_curl" % _side, "Thumb across the palm", 0, 90),
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
    # Right fingers on the keyboard, the left palm flat on the bass end.
    ("accordion", "Playing the accordion", {
        "arm_r_raise": 30, "arm_r_bend": 50, "wrist_r_bend": 70, "fingers_r_curl": 30,
        "fingers_r_spread": 8, "thumb_r_curl": 15, "arm_l_raise": -20, "arm_l_out": 40,
        "arm_l_bend": 120, "wrist_l_bend": -40, "fingers_l_curl": 12,
        "fingers_l_spread": 4}),
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
        rot["wrist_" + side] = euler(0, -g("wrist_%s_bend" % side), 0)
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
# A person's look sizes the mannequin: the Weight, Muscle, Chest size and Height sliders
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
    (chest, shoulders and limbs), `chest` (independent chest size), and how far apart the `shoulders` and
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
            "muscle": muscle, "chest": 1 + 0.1 * _num(look.get("chest_size"), 0, -3, 3),
            "shoulders": 1 + 0.35 * (fat - 1) + 0.8 * (muscle - 1) + 0.04 * s["shoulders"],
            "hips": 1 + 0.55 * (fat - 1) + 0.05 * s["curves"]}


REST_SHAPE = {"height": 1.0, "fat": 1.0, "belly": 1.0, "muscle": 1.0, "shoulders": 1.0,
              "hips": 1.0, "chest": 1.0}  # body_shape({}), which needs _num from below


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
                 "accordion": "#a3262a", "stein": "#d9d3c2",
                 "pretzel": "#9a5a26"}
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
    ("chef", ("chef hat", "chef's hat", "chefs hat", "toque", "toque blanche")),
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
    "chef": ([(0.15, (0.092, 0.107)), (0.23, (0.092, 0.107)),
              (0.25, (0.105, 0.12)), (0.45, (0.108, 0.123)),
              (0.46, (0.095, 0.11))], None),
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
    ("spatula", ("spatula", "fish slice", "slotted turner")),
    ("accordion", ("accordion", "accordian", "squeezebox", "concertina")),
    ("stein", ("beer stein", "beer steins", "stein", "steins", "beer mug", "beer mugs",
               "tankard", "tankards", "masskrug", "maßkrug", "maß", "pint of beer",
               "glass of beer", "beer glass", "beer glasses")),
    ("pretzel", ("pretzel", "pretzels", "brezel", "brezn", "breze")),
]
# How a hand takes what it holds, while its wrist and finger controls are all
# left at rest: the controls it gets, by their key less the side. The
# accordion's own hands are the "Playing the accordion" pose; this is only
# fingers ready for the keys (right) and flat for the bass end (left).
GRIP_KEYS = ("wrist_%s_bend", "fingers_%s_curl", "fingers_%s_spread", "thumb_%s_curl")
GRIPS = {"stein": (0, 55, 0, 60), "pretzel": (0, 35, 6, 70),
         "spatula": (0, 65, 0, 65),
         "accordion_r": (0, 30, 8, 15), "accordion_l": (0, 12, 4, 0)}


def gripped(controls, held):
    """`controls` with each hand that holds something and is left at rest
    taking it (`GRIPS`); a hand posed by its own controls keeps them."""
    out = dict(controls)
    for side in "lr":
        if any(float(out.get(k % side, 0) or 0) for k in GRIP_KEYS):
            continue
        kind = "accordion_" + side if "accordion" in held else next(
            (k for k in ("stein", "pretzel", "spatula") if side in held.get(k, ((),))[0]), None)
        if kind:
            out.update({k % side: float(v) for k, v in zip(GRIP_KEYS, GRIPS[kind])})
    return out
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
            if kind == "pretzel":         # the hand a stein leaves free
                taken = set(held.get("stein", ((),))[0])
                sides = ("l",) if "r" in taken else ("r",)
            held[kind] = (sides, cloth_colour(item if _has_colour(item) else "", kind))
            continue                      # "a red accordion" is not a red hat
        kind = next((k for k, words in HATS if _said(item, *words)), None)
        if kind and not hat:
            rgb = cloth_colour(item, "hat")
            if kind == "hard hat" and not _has_colour(item):
                rgb = hex_rgb("#e2c23c")          # a hard hat unsaid is site yellow
            elif kind == "chef" and not _has_colour(item):
                rgb = cloth_colour("white", "hat")
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


def _reshape(warp, p):
    """A point in the head's frame moved as `warp` moves the skull's, which
    works in the skull ellipsoid's frame (centre (0, 0.11, 0.01))."""
    x, y, z = warp((p[0], p[1] - 0.11, p[2] - 0.01))
    return (x, y + 0.11, z + 0.01)


def head_point(p, head=None):
    """Deform a head-frame landmark with the skull's identity warp.

    Landmarks already describe the neutral face (including its jaw), so
    only the identity deformation applies here, not the base `_jaw` shape.
    """
    warp = mq.head_warp(head)
    return _reshape(warp, p) if warp else p


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


# The fingers, index (by the thumb) to little: (across the palm, length, girth).
FINGERS = ((0.03, 0.078, 0.0095), (0.01, 0.086, 0.01),
           (-0.01, 0.08, 0.0095), (-0.029, 0.064, 0.0085))
KNUCKLE_Y = -0.098                   # the palm's end, down from the wrist
RELAXED = 10                         # each joint's curl at rest, degrees


def hand_joints(sign, grip):
    """A hand's 21 points in DWPose's order - the wrist, then thumb, index,
    middle, ring and little finger, four each from the base out - in the
    wrist's frame (down the hand is -y, the palm faces +z). `grip` is
    (curl, spread, thumb) in degrees. The mesh and the pose map both come
    from these, so what is drawn is what the ControlNet is told."""
    curl, spread, thumb = grip
    # The palm faces +z, so the thumb is on the hand's outer side (+x on
    # the left hand, -x on the right), away from the body at rest.
    # The thumb, its two joints swung from alongside the palm to across it.
    t = thumb / 90.0
    lerp = lambda a, b: tuple(u + (v - u) * t for u, v in zip(a, b))   # noqa: E731
    base = (sign * 0.03, -0.03, 0.012)
    knuckle = lerp((sign * 0.045, -0.08, 0.032), (sign * 0.03, -0.075, 0.058))
    tip = lerp((sign * 0.045, -0.12, 0.04), (0.0, -0.1, 0.078))
    pts = [(0.0, 0.0, 0.0), base, knuckle, mul(add(knuckle, tip), 0.5), tip]
    for i, (x, length, _) in enumerate(FINGERS):
        # Each of three joints bends by the curl, toward the palm; a spread
        # fans the fingers out from between the middle and ring fingers.
        fan = math.radians(spread * (1.5 - i) / 1.5) * sign
        p, bend = (x * sign, KNUCKLE_Y, 0.012), 0.0
        pts.append(p)
        for frac in (0.45, 0.3, 0.25):
            bend += math.radians(RELAXED + curl)
            d = (math.sin(fan) * math.cos(bend), -math.cos(fan) * math.cos(bend),
                 math.sin(bend))
            p = add(p, mul(d, length * frac))
            pts.append(p)
    return pts


def _fingers(part, side, sign, grip, at, side_x, side_z):
    """A hand's four fingers and thumb: [(part, "hand", faces)], through
    `at`, the wrist's frame."""
    pts = hand_joints(sign, grip)
    out = []
    for i, (_, _, w) in enumerate(FINGERS):
        chain = pts[5 + 4 * i: 9 + 4 * i]
        girth = w
        for p, q in zip(chain, chain[1:]):
            out.append((part, "hand", prism(at(p), at(q), side_x, (girth, girth * 0.9),
                                            (girth * 0.9, girth * 0.85), 6)))
            girth *= 0.88
    base, knuckle, tip = pts[1], pts[2], pts[4]
    out += [(part, "hand", prism(at(base), at(knuckle), side_z, (0.013, 0.012),
                                 (0.011, 0.01), 6)),
            (part, "hand", prism(at(knuckle), at(tip), side_z, (0.011, 0.01),
                                 (0.008, 0.008), 6))]
    return out


def person_pieces(controls, root=IDENTITY, shape=None, dressed=None, head=None):
    """The mannequin: [(part, faces, rgb or None)], pelvis at the origin,
    built to `shape` (`body_shape`), its head to `head` (`mq.HEAD_SHAPE`)
    and wearing `dressed` (`outfit`); rgb
    None is the object's own colour. Parts are the control groups, so a
    click on a hand selects the hand's controls."""
    shape = shape or REST_SHAPE
    dressed = dressed or {}
    held = dressed.get("held") or {}
    controls = gripped(controls, held)
    sk = skeleton(controls, root, shape)
    P = lambda j: sk[j][0]                                 # noqa: E731
    M = lambda j: sk[j][1]                                 # noqa: E731
    X = lambda j: column(M(j), 0)                          # noqa: E731
    at = lambda j, v: add(P(j), apply(M(j), v))            # noqa: E731
    grip = lambda side: tuple(float(controls.get(k % side, 0) or 0)   # noqa: E731
                              for k in GRIP_KEYS[1:])
    fat, mus = shape["fat"] - 1, shape["muscle"] - 1
    curve = shape["hips"] - 0.55 * fat               # the hips beyond their girth
    wear = dressed.get("regions") or {}
    # A covered region is drawn a little fuller, as cloth over it is.
    k = lambda region, f: f * (1.04 if region in wear else 1.0)    # noqa: E731
    r = lambda rad, ka, kd=None: (rad[0] * ka, rad[1] * (ka if kd is None else kd))  # noqa
    hips = k("hips", 1 + 0.9 * fat)
    # Weight fills a torso out more front to back than across, and chest
    # size is a bust in front, not a wider ribcage: scaled whole, a heavy
    # full-chested chest block stood out past the shoulders like a second
    # pair, and the picture drew two people stacked in it.
    belly = k("belly", 1 + 1.0 * fat)
    belly_deep = k("belly", 1 + 1.1 * fat)
    # The joins between the blocks fill out with weight: drawn in as on a
    # slim wooden mannequin, a heavy one read as three bodies stacked.
    join = 1 + 0.4 * max(0.0, fat)
    chest = k("chest", 1 + 0.45 * fat + mus)
    chest_deep = k("chest", 1 + 0.7 * fat + mus)
    bust = shape.get("chest", 1.0) - 1                  # -0.3 .. 0.3
    pecs = max(0.0, 0.08 + 0.9 * bust) if bust > 0 else 0.08 * (1 + bust / 0.3)
    # The top of the ribcage stays on the shoulder line, weight or not.
    top = shape["shoulders"] * (1 + 0.3 * (chest - 1))
    neck = k("neck", 1 + 0.5 * fat + 0.6 * mus) * mq.neck_scale(head)
    upper = k("upper_arm", 1 + 0.7 * fat + 1.3 * mus)
    fore = k("forearm", 1 + 0.5 * fat + 0.8 * mus)
    thigh = k("thigh", 1 + 0.8 * fat + 0.8 * mus)
    shin = k("shin", 1 + 0.45 * fat + 0.6 * mus)
    # The head, less what a scalp of hair covers: a big face of it sorts in
    # front of the hair over it and shows through as a pale speck.
    hair = dressed.get("hair")
    hat = dressed.get("hat")
    scalp = bool(hair and hair["cap"] and not (hat and hat[0] != "crown"))
    warp = mq.head_warp(head)
    shaped = (lambda p: warp(_jaw(p))) if warp else _jaw          # noqa: E731
    skull = [[at("head", p) for p in f]
             for f in ellipsoid((0, 0.11, 0.01), IDENTITY, (0.085, 0.115, 0.1), 16, 12, shaped)
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
            (0, (0.12 * belly * join, 0.082 * belly_deep), 0.0, waist),
            (0.2, (0.138 * belly, 0.092 * belly_deep), 0.0, waist),
            (0.55, (0.14 * belly, 0.094 * belly_deep * shape["belly"]),
             -0.006 * shape["belly"] - 0.04 * fat, waist),
            # Meets the chest block as at rest, however heavy: a wider belly
            # top under a narrower chest read as a second waist.
            (1, (0.128 * chest * join, 0.086 * (belly_deep + chest_deep) / 2), 0.0, waist)],
                               20)),
        ("body", "chest", loft(at("chest", (0, 0.0, 0)), at("chest", (0, 0.23, 0)),
                               X("chest"), [
            (0, (0.122 * chest * join, 0.083 * chest_deep), 0.0, mq.chest(0)),
            (0.12, (0.15 * chest, 0.096 * chest_deep), -0.002, mq.chest(0.25 * pecs)),
            (0.35, (0.162 * chest * wide ** 0.5, 0.1 * chest_deep * (1 + 0.3 * bust)),
             -0.008 - 0.06 * max(0.0, bust), mq.chest(pecs)),
            (0.55, (0.172 * chest * wide ** 0.75, 0.102 * chest_deep * (1 + 0.3 * bust)),
             -0.01 - 0.05 * max(0.0, bust), mq.chest(0.875 * pecs)),
            (0.8, (0.18 * top, 0.094 * chest_deep), -0.004, mq.chest(0.125 * pecs)),
            (0.94, (0.158 * top, 0.074 * chest_deep), 0.004, mq.chest(0)),
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
        # Hair follows the skull's shape: the point moved as the skull's is.
        def hp(v3):
            return at("head", _reshape(warp, v3) if warp else v3)
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
            # The palm; the four fingers and the thumb come after.
            (hand, "hand", loft(P("wrist_" + side), at("wrist_" + side, (0, -0.1, 0.012)),
                                X("wrist_" + side), [
                (0, (0.028, 0.02), 0.0, mq.MITTEN), (0.45, (0.042, 0.02), 0.0, mq.MITTEN),
                (1, (0.044, 0.016), 0.0, mq.MITTEN)], 14)),
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
            (foot, "shin", ellipsoid(P("knee_" + side), M("knee_" + side),
                                     (0.056 * shin, 0.058 * shin, 0.058 * shin), 12, 8)),
        ]
        out += _fingers(hand, side, sign, grip(side), lambda v, s=side: at("wrist_" + s, v),
                        X("wrist_" + side), column(M("wrist_" + side), 2))
        if shoes:
            out += [(foot, rgb, faces) for rgb, faces in _shoe(
                shoes, lambda v, s=side: at("ankle_" + s, v))]
        if shaft:
            out.append((foot, shaft[1], prism(at("ankle_" + side, (0, -0.02, 0)), shin_end,
                                              X("knee_" + side), r((0.05, 0.05), shin),
                                              r((0.052, 0.052), shin))))
    # Braces over the chest: two straps down the front, a bar across them.
    if dressed.get("braces"):
        rgb, fz = dressed["braces"], 0.1 * chest_deep * (1 + 0.3 * bust) + 0.012
        for sx in (1, -1):
            out.append(("body", rgb, prism(at("spine", (sx * 0.075, -0.02,
                                                        0.095 * belly_deep + 0.012)),
                                           at("chest", (sx * 0.085, 0.19, fz)), X("chest"),
                                           (0.018, 0.006), (0.018, 0.006), 4)))
        out.append(("body", rgb, prism(at("chest", (-0.09, 0.1, fz + 0.004)),
                                       at("chest", (0.09, 0.1, fz + 0.004)), (0, 1, 0),
                                       (0.025, 0.006), (0.025, 0.006), 4)))
    if "accordion" in held:
        # Across the chest, bellows between two ends; the Carrying pose puts
        # the hands on it.
        _, rgb = held["accordion"]
        z = accordion_front(chest_deep, bust)
        chest_box = lambda lo, hi: outward([[at("chest", p) for p in f]   # noqa: E731
                                            for f in box(lo, hi)])
        out += [("body", rgb, chest_box((-0.23, -0.2, z - 0.1), (-0.15, 0.12, z + 0.1))),
                ("body", rgb, chest_box((0.15, -0.2, z - 0.1), (0.23, 0.12, z + 0.1))),
                ("body", (236, 235, 230), chest_box((-0.225, -0.18, z + 0.1),
                                                    (-0.155, 0.1, z + 0.115)))]  # keys, at
        # their right (-x), for the right hand
        for i in range(8):                # the bellows' pleats, dark and light
            x0 = -0.15 + 0.3 * i / 8
            deep = 0.085 if i % 2 else 0.07
            out.append(("body", (42, 37, 34) if i % 2 else tuple(int(c * 0.45) for c in rgb),
                        chest_box((x0, -0.18, z - deep), (x0 + 0.3 / 8, 0.1, z + deep))))
    if "spatula" in held:
        sides, rgb = held["spatula"]
        for side in sides:
            # Handle inside the curled palm; blade extends along the fingers.
            def utensil_box(lo, hi):
                return outward([[at("wrist_" + side, p) for p in f] for f in box(lo, hi)])
            out.append(("hand_" + side, (48, 44, 40), utensil_box(
                (-0.014, -0.18, 0.022), (0.014, -0.035, 0.05))))
            out.append(("hand_" + side, rgb, utensil_box(
                (-0.009, -0.3, 0.03), (0.009, -0.17, 0.042))))
            # Three open slots between four metal tines, joined at both ends.
            for x in (-0.042, -0.016, 0.01, 0.036):
                out.append(("hand_" + side, rgb, utensil_box(
                    (x, -0.415, 0.03), (x + 0.006, -0.285, 0.036))))
            for y in (-0.415, -0.295):
                out.append(("hand_" + side, rgb, utensil_box(
                    (-0.042, y, 0.03), (0.042, y + 0.01, 0.036))))
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
    if "pretzel" in held:
        sides, rgb = held["pretzel"]
        for side in sides:
            # Pinched at its top between the thumb and the fingers, hanging
            # in the plane of the palm: a loop and the two arms crossed in it.
            w = lambda x, y, s=side: at("wrist_" + s, (x, -0.1 + y, 0.075))  # noqa: E731
            loop = [(0.055 * math.sin(2 * math.pi * i / 14),
                     -0.05 - 0.045 * math.cos(2 * math.pi * i / 14)) for i in range(14)]
            strands = list(zip(loop, loop[1:] + loop[:1]))
            strands += [((-0.03, -0.087), (0.028, -0.03)), ((0.03, -0.087), (-0.028, -0.03))]
            for (x0, y0), (x1, y1) in strands:
                out.append(("hand_" + side, rgb, prism(w(x0, y0), w(x1, y1),
                                                       column(M("wrist_" + side), 2),
                                                       (0.011, 0.011), (0.011, 0.011), 6)))
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
    {"id": "frying_pan", "label": "Frying pan", "kind": "prop", "group": "prop",
     "name": "Frying pan", "colour": "#343638", "scale": [0.42, 0.055, 0.68],
     "about": "An open shallow frying pan with a long handle towards +Z. Place it on a stove."},
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


def _frying_pan():
    # Bowl and handle fit the same unit box as every other prop. Each rim
    # segment is convex, so normals stay outward on the open inner wall too.
    out = [(prism((0, 0, -0.18), (0, 0.16, -0.18), (1, 0, 0),
                  (0.46, 0.28), (0.46, 0.28), 24), None)]
    for i in range(24):
        a, b = 2 * math.pi * i / 24, 2 * math.pi * (i + 1) / 24
        pts = [(rx * math.cos(t), y, -0.18 + rz * math.sin(t))
               for y in (0.16, 1.0) for rx, rz, t in (
                   (0.5, 0.32, a), (0.5, 0.32, b),
                   (0.455, 0.275, b), (0.455, 0.275, a))]
        out.append((outward([[pts[j] for j in f] for f in (
            (0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1),
            (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0))]), None))
    out.append((_slab(-0.04, 0.45, 0.09, 0.04, 0.7, 0.25), "#9a9c9e"))
    out.append((_slab(-0.05, 0.4, 0.23, 0.05, 0.8, 0.5), "#302c28"))
    return out


# Each shape and prop: [(faces, colour or None for the object's own)].
MESHES = {
    "frying_pan": _frying_pan(),
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

# ==================================================================== dressing
# A prop's details: what is on it, hung from it or painted on it - the
# maypole's wreath and its flags. Each is words, said after the prop's
# description, and unless it has no body of its own ("none": paint, carving,
# wear) simple pieces drawn on the prop, so the maps carry it as well. A piece
# is sized in metres, not by the prop's scale (a 20 m pole does not make a
# 20 m flag), and stands at a height on the prop, `at`, hugging its outline
# there (`girth`), so resizing the prop keeps the wreath at its top.
DRESS_PARTS = {               # part: (one, several) as the inspector says it
    "ring": ("a ring round it", "%d rings round it"),
    "flag": ("a flag standing out from it", "%d flags standing out from it"),
    "banner": ("a banner hanging down it", "%d banners hanging down it"),
    "sign": ("a board on its side", "%d boards on its sides"),
    "ball": ("a ball", "%d balls"), "box": ("a block", "%d blocks"),
    "cylinder": ("a cylinder", "%d cylinders"), "cone": ("a cone", "%d cones"),
    "none": ("", ""),
}
DRESS_AT = {"top": 1.0, "upper": 0.8, "middle": 0.5, "lower": 0.25, "base": 0.0,
            "along": None}           # along: spread up its length, side to side
DRESS_AT_WORDS = {"top": "at the top", "upper": "near the top", "middle": "halfway up",
                  "lower": "low down", "base": "at the foot", "along": "up its length"}
DRESS_SIZE = {"ring": [1.0, 0.15, 0.15], "flag": [0.6, 0.4, 0.02],
              "banner": [0.3, 1.5, 0.02], "sign": [0.5, 0.35, 0.04],
              "ball": [0.2, 0.2, 0.2], "box": [0.3, 0.3, 0.3],
              "cylinder": [0.2, 0.3, 0.2], "cone": [0.3, 0.4, 0.3]}
DRESS_COLOUR = "#d9d2c3"      # a piece the model gave no colour
DRESS_KEEP = 8                # details on one prop, at most
DRESS_COUNT = 12              # pieces of one detail, at most
DRESS_CHARS = 240
_GIRTH = {}
_SQ2 = math.sqrt(2)           # a 4-sided prism's corners are at 45 degrees


def new_dressing(text, part="none", at="top", count=1, size=None, colour=""):
    return {"text": text, "part": part, "at": at, "count": count,
            "size": list(size or DRESS_SIZE.get(part) or [0.3, 0.3, 0.3]), "colour": colour}


def clean_dressing(items):
    """A prop's saved details made safe: words kept, the rest within reach."""
    out = []
    for d in items if isinstance(items, list) else []:
        if not isinstance(d, dict) or not isinstance(d.get("text"), str) \
                or not d["text"].strip():
            continue
        part = d.get("part") if d.get("part") in DRESS_PARTS else "none"
        size = d.get("size")
        real = lambda v: (isinstance(v, (int, float)) and not isinstance(v, bool)  # noqa: E731
                          and math.isfinite(v))
        if not (isinstance(size, list) and len(size) == 3 and all(real(v) for v in size)):
            size = None
        count = int(round(d["count"])) if real(d.get("count")) else 1
        out.append(new_dressing(
            d["text"].strip()[:DRESS_CHARS], part,
            d.get("at") if d.get("at") in DRESS_AT else "top",
            min(max(count, 1), DRESS_COUNT),
            [min(max(float(v), 0.02), 8.0) for v in size] if size else None,
            d["colour"].lower() if HEX.match(str(d.get("colour") or "")) else ""))
    return out[:DRESS_KEEP]


def described(obj):
    """An object's description with its details after it, each as written."""
    words = obj["description"].strip()
    for d in obj.get("dressing") or []:
        text = d["text"].strip()
        if text:
            words = (words + (" " if words.endswith((".", "!", "?")) else ". ") + text
                     if words else text)
    return words


def dressing_label(d):
    """How the inspector says a detail is drawn: 'Drawn as 4 flags standing
    out from it, near the top', or 'Words only'."""
    one, many = DRESS_PARTS[d["part"]]
    if not one:
        return "Words only"
    return "Drawn as %s, %s" % (one if d["count"] == 1 else many % d["count"],
                                DRESS_AT_WORDS[d["at"]])


def girth(asset, t):
    """(x, z) half-widths of a prop's mesh in its unit box at height t (0 its
    foot, 1 its top): where its faces cross that height. The whole mesh's
    widest where nothing is there."""
    key = (asset, round(t, 3))
    if key not in _GIRTH:
        xs, zs = [], []
        for f in UNIT[asset]:
            for p, q in zip(f, f[1:] + f[:1]):
                if abs(p[1] - t) < 0.005:
                    xs.append(abs(p[0]))
                    zs.append(abs(p[2]))
                elif (p[1] - t) * (q[1] - t) < 0:
                    k = (t - p[1]) / (q[1] - p[1])
                    xs.append(abs(p[0] + (q[0] - p[0]) * k))
                    zs.append(abs(p[2] + (q[2] - p[2]) * k))
        if not xs:
            pts = [p for f in UNIT[asset] for p in f]
            xs, zs = [abs(p[0]) for p in pts], [abs(p[2]) for p in pts]
        _GIRTH[key] = (max(xs), max(zs))
    return _GIRTH[key]


def _reach(gx, gz, u):
    """How far out the prop's outline (an ellipse of half-widths gx, gz) is
    along the level direction u."""
    if gx < 1e-6 or gz < 1e-6:
        return max(gx, gz)
    return 1.0 / math.sqrt((u[0] / gx) ** 2 + (u[2] / gz) ** 2)


def dressing_pieces(obj):
    """A prop's details as [(faces, '#rrggbb')] in its own frame, before it
    is turned: metres, its foot at y 0, its middle at x = z = 0."""
    sx, sy, sz = obj["scale"]
    out = []
    for d in obj.get("dressing") or []:
        part, n = d["part"], d["count"]
        if part == "none":
            continue
        w, h, dp = d["size"]
        colour = d["colour"] or DRESS_COLOUR
        radial = part == "flag"
        if d["at"] == "along":          # up its length, side to side
            spots = [(0.3 + 0.6 * (i + 0.5) / n,
                      math.pi / 2 + math.pi * i if radial else 0.0) for i in range(n)]
        else:                           # round it at one height; one flag flies sideways
            first = math.pi / 2 if radial and n == 1 else 0.0
            spots = [(DRESS_AT[d["at"]], first + 2 * math.pi * i / n) for i in range(n)]
        if part == "ring":              # one ring a height, however many were asked
            spots = list(dict((t, (t, 0.0)) for t, _ in spots).values())
        for t, a in spots:
            u, v = (math.sin(a), 0.0, math.cos(a)), (math.cos(a), 0.0, -math.sin(a))
            tall = h if part in ("ring", "flag", "sign") else 0.0
            y = min(max(t * sy, tall / 2), sy - tall / 2) if tall < sy else sy / 2
            gx, gz = girth(obj["asset"], y / sy)
            gx, gz = gx * sx, gz * sz
            r = _reach(gx, gz, u)
            if part == "ring":
                rx, rz = max(gx + dp / 2, w / 2 - dp / 2), max(gz + dp / 2, w / 2 - dp / 2)
                seg = 16
                ring = [(rx * math.cos(2 * math.pi * i / seg), y,
                         rz * math.sin(2 * math.pi * i / seg)) for i in range(seg)]
                for p, q in zip(ring, ring[1:] + ring[:1]):
                    grow = mul(sub(q, p), 0.06)       # a little long, so no gap shows
                    out.append((prism(sub(p, grow), add(q, grow), (0, 1, 0),
                                      (h / 2 * _SQ2, dp / 2 * _SQ2),
                                      (h / 2 * _SQ2, dp / 2 * _SQ2), 4), colour))
            elif part == "flag":
                out.append((prism(add(mul(u, r), (0, y, 0)), add(mul(u, r + w), (0, y, 0)),
                                  (0, 1, 0), (h / 2 * _SQ2, dp / 2 * _SQ2),
                                  (h / 2 * _SQ2, dp / 2 * _SQ2), 4), colour))
            elif part in ("banner", "sign"):
                top = (min(max(t * sy, h), sy) if part == "banner"
                       else min(max(t * sy + h / 2, h), sy))
                at = mul(u, r + dp / 2)
                out.append((prism(add(at, (0, max(top - h, 0.0), 0)), add(at, (0, top, 0)),
                                  v, (w / 2 * _SQ2, dp / 2 * _SQ2),
                                  (w / 2 * _SQ2, dp / 2 * _SQ2), 4), colour))
            else:                       # a solid: on top, or round it at that height
                if d["at"] == "top" and n == 1:
                    mid, foot = (0.0, 0.0, 0.0), sy
                else:
                    mid = mul(u, r + max(w, dp) / 2)
                    foot = 0.0 if d["at"] == "base" else min(max(t * sy - h / 2, 0.0), sy)
                lo, hi = add(mid, (0, foot, 0)), add(mid, (0, foot + h, 0))
                if part == "ball":
                    out.append((ellipsoid(add(mid, (0, foot + h / 2, 0)), IDENTITY,
                                          (w / 2, h / 2, dp / 2), 10, 6), colour))
                elif part == "box":
                    out.append((prism(lo, hi, v, (w / 2 * _SQ2, dp / 2 * _SQ2),
                                      (w / 2 * _SQ2, dp / 2 * _SQ2), 4), colour))
                else:
                    tip = (0.004, 0.004) if part == "cone" else (w / 2, dp / 2)
                    out.append((prism(lo, hi, v, (w / 2, dp / 2), tip, 12), colour))
    return out


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
                                                        shape, outfit(look), obj.get("head"))]
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
        for mesh, rgb in dressing_pieces(obj):
            faces = [[apply(rot, p) for p in f] for f in mesh]
            pieces.append(("body", [t for f in faces for t in tiles(f)], hex_rgb(rgb)))
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
    import apps.image_studio.imagegen as ig
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
    return {"walls": False, "width": 8.0, "depth": 8.0, "height": 3.0, "grid": [],
            "floor": {"prompt": "", "image": "", "size": 2.0},
            "wall": {"prompt": "", "image": "", "size": 3.0}}


# ==================================================================== ground
# The floor can be shaped. `room["grid"]` is 4 x 4 handle heights, rows from
# the back (-z) to the front (+z, where the camera starts), columns from left
# (-x) to right, spread evenly over the room's width x depth round the origin
# (with walls or without). They pull one smooth surface, as a Bezier patch's
# control points do: every dot shapes a broad area, not one point, the
# surface stays between the lowest and highest dot, and raising the middle
# four makes a round hill. The patch runs through `smoothstep` first, so it
# meets the level floor round it without a crease: past the edges the floor
# carries the edge's height straight on. [] is the flat floor at 0, drawn as
# it always was. An object's position y stays where its lowest point is, so
# what stands on the floor is re-stood (`stand`) when it moves or the floor
# is reshaped.
GRID = 4                       # dots each way
GRID_HEIGHT = (-4.0, 6.0)      # m a dot can go down and up
GRID_CELLS = 12                # the shaped area is drawn as this many cells each way
GRID_OUTSIDE = (2.0, 6.0, 14.0)  # m past its edge the level floor is cut, for sorting
FLOOR_SHADE = 0.05             # a sloped face's brightness on a picture, in steps
FLOOR_PRESETS = [              # (key, label, rows back to front)
    ("flat", "Flat", [[0, 0, 0, 0]] * 4),
    ("hill", "Hill", [[0, 0, 0, 0], [0, 2.6, 2.6, 0], [0, 2.6, 2.6, 0], [0, 0, 0, 0]]),
    ("dip", "Dip", [[0, 0, 0, 0], [0, -2, -2, 0], [0, -2, -2, 0], [0, 0, 0, 0]]),
    ("ridge", "Ridge", [[0, 2.2, 2.2, 0]] * 4),
    ("rise", "Rise behind", [[2, 2, 2, 2], [2, 2, 2, 2], [0, 0, 0, 0], [0, 0, 0, 0]]),
    ("left", "Bank left", [[2, 2, 0, 0]] * 4),
    ("right", "Bank right", [[0, 0, 2, 2]] * 4),
    ("bowl", "Bowl", [[2, 2, 2, 2], [2, 0, 0, 2], [2, 0, 0, 2], [2, 2, 2, 2]]),
    ("rolling", "Rolling", [[0, 1.4, 0, 1.4], [1.4, 0, 1.4, 0], [0, 1.4, 0, 1.4],
                            [1.4, 0, 1.4, 0]]),
]
FLOOR_PRESET = {k: rows for k, _, rows in FLOOR_PRESETS}


def clean_grid(raw):
    """Handle heights made safe: 4 rows of 4 numbers in range, or [] for a
    flat floor (no such grid, or every dot at 0)."""
    if not isinstance(raw, (list, tuple)) or len(raw) != GRID:
        return []
    out = []
    for row in raw:
        if not isinstance(row, (list, tuple)) or len(row) != GRID:
            return []
        clean = []
        for v in row:
            try:
                v = float(v)
            except (TypeError, ValueError):
                return []
            if not math.isfinite(v):
                return []
            clean.append(round(max(GRID_HEIGHT[0], min(GRID_HEIGHT[1], v)), 3))
        out.append(clean)
    return out if any(v for row in out for v in row) else []


_FLOOR_SHAPE_WORDS = {
    "hill": "the ground rises into a hill",
    "dip": "the ground dips down in the middle",
    "ridge": "a ridge crosses the ground",
    "rise": "the ground rises up behind",
    "left": "the ground banks up to the left",
    "right": "the ground banks up to the right",
    "bowl": "the ground forms a shallow bowl, higher at the edges",
    "rolling": "the ground rolls gently, uneven underfoot",
}


def floor_shape_words(grid):
    """A short phrase for a shaped floor, so the words - the only thing this
    pipeline sends of the ground - say what the preview already shows.
    Matched by preset when the grid is exactly one; a hand-dragged shape
    falls back to a plain "not flat"."""
    grid = clean_grid(grid)
    if not grid:
        return ""
    for key, rows in FLOOR_PRESET.items():
        if key != "flat" and grid == rows:
            return _FLOOR_SHAPE_WORDS[key]
    return "the ground is uneven, not flat"


def _bernstein(t):
    """The four cubic weights at `t` (0-1), eased so the surface leaves its
    edge level."""
    t = min(1.0, max(0.0, t))
    s = t * t * (3 - 2 * t)
    r = 1 - s
    return (r * r * r, 3 * s * r * r, 3 * s * s * r, s * s * s)


class Ground:
    """The floor's height anywhere, from a room's grid (None: flat)."""

    def __init__(self, room):
        room = room or {}
        self.grid = clean_grid(room.get("grid"))
        self.w, self.d = float(room.get("width") or 8.0), float(room.get("depth") or 8.0)
        self.level = len({v for row in self.grid for v in row}) <= 1

    def at(self, x, z):
        if not self.grid:
            return 0.0
        if self.level:
            return self.grid[0][0]
        u0, u1, u2, u3 = _bernstein(x / self.w + 0.5)
        v0, v1, v2, v3 = _bernstein(z / self.d + 0.5)
        r0, r1, r2, r3 = [u0 * a + u1 * b + u2 * c + u3 * d for a, b, c, d in self.grid]
        return v0 * r0 + v1 * r1 + v2 * r2 + v3 * r3

    def slope(self, x, z, e=0.01):
        """The floor's upward normal at (x, z)."""
        dx = (self.at(x + e, z) - self.at(x - e, z)) / (2 * e)
        dz = (self.at(x, z + e) - self.at(x, z - e)) / (2 * e)
        return norm((-dx, 1.0, -dz))

    def cuts(self, lo, hi, half):
        """Where the floor is cut between `lo` and `hi` along one axis whose
        shaped area is -half..half: evenly across the area, and a few times
        beyond it (level there across, so only for the painter's sort). A
        level floor is its two ends."""
        if self.level:
            return [lo, hi]
        at = {lo, hi}
        at.update(-half + 2 * half * k / GRID_CELLS for k in range(GRID_CELLS + 1))
        at.update(s * (half + g) for s in (-1, 1) for g in GRID_OUTSIDE)
        return sorted(a for a in at if lo <= a <= hi)

    def xs(self, lo, hi):
        return self.cuts(lo, hi, self.w / 2)

    def zs(self, lo, hi):
        return self.cuts(lo, hi, self.d / 2)

    def faces(self, x0, x1, z0, z1):
        """The floor over x0..x1, z0..z1 as flat faces: [(points, normal)],
        each normal pointing up. A cell whose corners lie in a plane is one
        quad; any other is two triangles."""
        xs, zs = self.xs(x0, x1), self.zs(z0, z1)
        hs = [[self.at(x, z) for x in xs] for z in zs]
        out = []
        for j, (za, zb) in enumerate(zip(zs, zs[1:])):
            for i, (xa, xb) in enumerate(zip(xs, xs[1:])):
                a, b = (xa, hs[j][i], za), (xb, hs[j][i + 1], za)
                c, d = (xb, hs[j + 1][i + 1], zb), (xa, hs[j + 1][i], zb)
                if abs(a[1] + c[1] - b[1] - d[1]) < 1e-9:
                    polys = [[a, b, c, d]]
                else:
                    polys = [[a, b, c], [a, c, d]]
                for p in polys:
                    n = norm(newell(p))
                    out.append((p, n if n[1] > 0 else mul(n, -1)))
        return out


def floor_height(room, x, z):
    return Ground(room).at(x, z)


def above_floor(room, obj):
    """How high an object stands above the floor under it (m)."""
    x, y, z = obj["position"]
    return y - floor_height(room, x, z)


def stand(room, obj, lift=0.0):
    """Put an object `lift` above the floor under where it is."""
    x, _, z = obj["position"]
    obj["position"][1] = round(floor_height(room, x, z) + round(lift, 3), 4)


def shape_floor(scene, grid=None, **size):
    """Give the floor new handle heights (`grid`, [] for flat, None to keep
    them) and/or the room a new `width` / `depth`, which the shape spans;
    everything on the floor stays as high above it as it was."""
    room = scene["room"]
    lifts = [above_floor(room, o) for o in scene["objects"]]
    if grid is not None:
        room["grid"] = clean_grid(grid)
    for key, v in size.items():
        lo, hi = ROOM_LIMITS[key]
        room[key] = max(lo, min(hi, float(v)))
    for obj, lift in zip(scene["objects"], lifts):
        stand(room, obj, lift)


def new_scene(details=""):
    return {"version": VERSION, "details": details, "frame": "portrait",
            "pose_strength": POSE_STRENGTH, "depth_strength": DEPTH_STRENGTH,
            "frame_keep": FRAME_KEEP, "face_likeness": FACE_LIKENESS,
            "head_depth": HEAD_DEPTH,
            "real_faces": REAL_FACES,
            "regional_prompting": REGIONAL_PROMPTING,
            "camera": {"target": [0.0, 1.0, 0.0], "yaw": 0.0, "pitch": 6.0,
                       "distance": 4.2, "lens": 35.0, "profile": "", "body": "",
                       "format": "", "chemistry": "", "lenses": []},
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
    if a["kind"] == "prop":
        obj["picture"] = ""           # a cut-out PNG drawn in place of the shape
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
    import apps.image_studio.imagegen as ig
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
    from apps.image_studio.scene import beard
    facial_hair = beard.clean(d.get("beard"))
    if facial_hair is not None:
        out["beard"] = facial_hair
    return out


def character_look(rec, look=None):
    """A character's look put on a person, as the form does it: every slot
    and slider the character keeps (blank where it has none, so the last
    one's beard does not stay); the expression and gaze are the picture's,
    and stay."""
    import apps.image_studio.imagegen as ig
    look = dict(look or {})
    look.pop("beard", None)
    for k in ig.CHARACTER_KEYS:
        look.pop(k, None)
    look.update((rec or {}).get("looks") or {})
    return clean_look(look)


def follow_character(obj, rec):
    """A person's look made their character's as it is now, for who they
    are (imagegen.WHO_KEYS: body, face, hair; blank where the character has
    none), with the beard the character keeps or its facial hair pick names
    (beard.from_words). What they wear and their expression stay the
    scene's. `rec` None leaves the look. -> True when it changed."""
    import apps.image_studio.imagegen as ig
    from apps.image_studio.scene import beard
    if rec is None or obj.get("asset") != "person":
        return False
    looks = rec.get("looks") or {}
    look = dict(obj.get("look") or {})
    for k in ig.WHO_KEYS:
        look.pop(k, None)
    look.update((k, looks[k]) for k in ig.WHO_KEYS if k in looks)
    b = beard.clean(looks.get("beard")) or beard.from_words(look.get("facial_hair"))
    look.pop("beard", None)
    if b is not None:
        look["beard"] = b
    look = clean_look(look)
    if look == obj.get("look"):
        return False
    obj["look"] = look
    return True


def follow_characters(scene, characters):
    """Every person with a character (`characters` {id: record}) follows
    it (`follow_character`). -> the names of those whose look changed."""
    return [o["name"] for o in scene.get("objects") or []
            if o.get("asset") == "person" and o.get("character")
            and follow_character(o, (characters or {}).get(o["character"]))]


def wear_outfit(rec, look=None):
    """A clothes preset put on a person: every clothes and accessories slot
    is the preset's, blank where it has none; the body, face and hair stay."""
    import apps.image_studio.imagegen as ig
    look = dict(look or {})
    for k in ig.OUTFIT_KEYS:
        look.pop(k, None)
    look.update((rec or {}).get("looks") or {})
    return clean_look(look)


def outfit_looks(look):
    """What a person wears, as a preset keeps it."""
    import apps.image_studio.imagegen as ig
    return {k: look[k] for k in ig.OUTFIT_KEYS if (look or {}).get(k)}


def look_text(obj):
    """The person's look in words, as the Image Studio says a person."""
    import apps.image_studio.imagegen as ig
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
    if "picture" in base:
        o["picture"] = str(d.get("picture") or "")
        dressing = clean_dressing(d.get("dressing"))
        if dressing:
            o["dressing"] = dressing
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
        head = mq.clean_head(d.get("head"))
        if head:
            o["head"] = head
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
    s["head_depth"] = _num(d.get("head_depth"), HEAD_DEPTH, *HEAD_DEPTH_RANGE)
    s["real_faces"] = bool(d.get("real_faces", REAL_FACES))
    s["regional_prompting"] = bool(d.get("regional_prompting", REGIONAL_PROMPTING))
    cam = d.get("camera") if isinstance(d.get("camera"), dict) else {}
    c = s["camera"]
    c["target"] = _vec(cam.get("target"), c["target"], -100, 100)
    c["yaw"] = _num(cam.get("yaw"), c["yaw"]) % 360
    c["pitch"] = _num(cam.get("pitch"), c["pitch"], -80, 85)
    c["distance"] = _num(cam.get("distance"), c["distance"], 0.3, 80)
    c["lens"] = _num(cam.get("lens"), c["lens"], 10, 300)
    # Which camera profile shot this scene (Scene Builder's Camera body
    # picker) and the words it baked in when chosen - kept here so a saved
    # scene still says the same thing after the library entry changes or is
    # deleted (`apps.image_studio.scene.ui._set_camera_profile`).
    c["profile"] = str(cam.get("profile") or "")
    c["body"] = str(cam.get("body") or "")          # its name, for "Shot on" in the form
    c["format"] = cam.get("format") if cam.get("format") in FORMAT_FRAMES else ""
    c["chemistry"] = str(cam.get("chemistry") or "")
    import apps.image_studio.imagegen as ig
    c["lenses"] = ig.clean_lenses(cam.get("lenses"))      # the camera's, for its buttons
    s["frame"] = camera_frame(c["format"], s["frame"])    # the frame is the camera's
    room = d.get("room") if isinstance(d.get("room"), dict) else {}
    r = s["room"]
    r["walls"] = room.get("walls") is True
    for key, (lo, hi) in ROOM_LIMITS.items():
        r[key] = _num(room.get(key), r[key], lo, hi)
    r["grid"] = clean_grid(room.get("grid"))
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
            if o.get("picture") and not os.path.isfile(o["picture"]):
                problems.append("%s's picture %s is missing, so it is drawn as its shape."
                                % (o["name"], os.path.basename(o["picture"])))
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
    import apps.image_studio.imagegen as ig        # lazily: the engine is heavy, this is a path
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
    ("description", "Describe %s"), ("dressing", "Change %s's details"),
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
    # Reshaping the floor re-stands everything on it: one step, the floor's.
    if (before.get("room") or {}).get("grid") != (after.get("room") or {}).get("grid"):
        return "Shape the floor"
    edited =[(was[i], o) for i, o in now.items() if o != was[i]]
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
        self.alpha = bytes(rgba[3::4])
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
    rgba, w, h = shrink(*read_png(src), keep_alpha=False)
    return _keep(studio_icons.png(rgba, w, h), "texture", folder)


def read_png(path):
    """-> (rgba, w, h). Raises OSError, or ValueError for what is not a PNG
    this can read (JPEG, 16-bit, interlaced)."""
    with open(path, "rb") as f:
        data = f.read()
    try:
        return studio_icons.png_to_rgba(data)
    except (KeyError, IndexError, zlib.error, struct.error) as e:
        raise ValueError("not a PNG this can read (%s)" % e)


def shrink(rgba, w, h, side=TEXTURE_SIDE, keep_alpha=True):
    """RGBA down to at most `side` on its longer edge, four samples a pixel.
    Colour is weighted by alpha, so a cut-out's edge does not go dark."""
    tw, th = ((side, max(1, round(side * h / w))) if w >= h else
              (max(1, round(side * w / h)), side))
    tw, th = min(tw, w), min(th, h)
    out = bytearray(tw * th * 4)
    for y in range(th):
        rows = [min(h - 1, int((y + oy) * h / th)) * w for oy in (0.25, 0.75)]
        for x in range(tw):
            cols = [min(w - 1, int((x + ox) * w / tw)) for ox in (0.25, 0.75)]
            r = g = b = a = 0
            for row in rows:
                for col in cols:
                    i = (row + col) * 4
                    al = rgba[i + 3] if keep_alpha else 255
                    r, g, b = r + rgba[i] * al, g + rgba[i + 1] * al, b + rgba[i + 2] * al
                    a += al
            o = (y * tw + x) * 4
            if a:
                out[o], out[o + 1], out[o + 2] = r // a, g // a, b // a
            out[o + 3] = a // 4
    return bytes(out), tw, th


def _keep(data, kind, folder=None):
    """Write PNG `data` under `<scenes>/textures/`, named by content."""
    folder = folder or os.path.join(scenes_dir(), "textures")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "%s_%s.png" % (kind, hashlib.sha1(data).hexdigest()[:16]))
    if not os.path.isfile(path):
        with open(path, "wb") as f:
            f.write(data)
    return path


KEY_TOLERANCE = 60             # sum of |r|+|g|+|b| from the background still counted as it


def key_background(rgba, w, h, tolerance=KEY_TOLERANCE):
    """A picture of one thing on a plain background -> (rgba, w, h), the
    background made transparent and the picture cropped to the thing. The
    background is the border's median colour, flooded in from the edges, so
    a white mug on a white table-top keeps its inside. A picture whose
    border is not one colour (a photo, a scene) comes back as it was: better
    a card with a background than a thing cut to pieces."""
    border = [y * w + x for x in range(w) for y in (0, h - 1)] + \
             [y * w + x for y in range(h) for x in (0, w - 1)]
    bg = [sorted(rgba[i * 4 + c] for i in border)[len(border) // 2] for c in range(3)]
    br, bgg, bb = bg
    has_alpha = any(a < 255 for a in rgba[3::4])

    def background(i):
        o = i * 4
        return rgba[o + 3] < 128 or (not has_alpha and abs(rgba[o] - br) + abs(rgba[o + 1] - bgg)
                                     + abs(rgba[o + 2] - bb) <= tolerance)
    if sum(1 for i in border if background(i)) < 0.8 * len(border):
        return rgba, w, h
    gone = bytearray(w * h)
    todo = [i for i in border if background(i)]
    for i in todo:
        gone[i] = 1
    while todo:
        i = todo.pop()
        x, y = i % w, i // w
        for j in ((i - 1) if x else -1, (i + 1) if x < w - 1 else -1,
                  (i - w) if y else -1, (i + w) if y < h - 1 else -1):
            if j >= 0 and not gone[j] and background(j):
                gone[j] = 1
                todo.append(j)
    kept = [i for i in range(w * h) if not gone[i]]
    if len(kept) < 0.01 * w * h:
        return rgba, w, h
    xs, ys = [i % w for i in kept], [i // w for i in kept]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    cw, ch = x1 - x0 + 1, y1 - y0 + 1
    out = bytearray(cw * ch * 4)
    for y in range(ch):
        for x in range(cw):
            i = (y + y0) * w + x + x0
            o = (y * cw + x) * 4
            out[o:o + 3] = rgba[i * 4:i * 4 + 3]
            out[o + 3] = 0 if gone[i] else rgba[i * 4 + 3]
    return bytes(out), cw, ch


def import_cutout(src, folder=None):
    """A picture of a prop -> (path, width / height): shrunk, its background
    keyed out and cropped, kept beside the textures. Raises as `read_png`."""
    rgba, w, h = shrink(*read_png(src))
    rgba, w, h = key_background(rgba, w, h)
    return _keep(studio_icons.png(rgba, w, h), "cutout", folder), w / float(h)


PICTURE_WORDS = ("%s. The whole thing alone, centred and filling most of the frame, on a "
                 "plain pure white background; soft even studio light, no shadow on the "
                 "ground, nothing else in the frame.")
PICTURE_NEGATIVE = ("background scenery, room, floor, cast shadow, people, person, hands, "
                    "text, watermark, cropped, cut off")


def picture_settings(obj, model, backend="auto"):
    """The Image Studio settings that paint a prop from its name and
    description, as written: text to image on plain white, so the background
    keys out, and none of the form's person. `scene_picture` names the
    object, so the finished job finds its way back to the builder."""
    import apps.image_studio.imagegen as ig
    name = obj["name"].strip() or ASSET[obj["asset"]]["label"]
    desc = described(obj).rstrip(" .")
    s = ig.default_settings()
    s.update(model=model, backend=backend,
             scene=PICTURE_WORDS % ("%s: %s" % (name, desc) if desc else name),
             negative=PICTURE_NEGATIVE, anatomy=False, width=1024, height=1024,
             refine=False, face_detail=False, hand_pass=False, upscale=None, batch=1,
             scene_picture=obj["id"])
    return s


def texture_settings(scene, surface, model, backend="auto"):
    """The Image Studio settings that make `surface`'s picture from its
    words: text to image, square, and none of the form's person, identities,
    LoRAs or references - a floor has no face to redraw. `scene_texture` names
    the surface, so the finished job finds its way back to the builder."""
    import apps.image_studio.imagegen as ig
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
    it. A picture's top row runs along the wall's top. On a shaped floor a
    wall stands on it, `height` tall wherever it is, as several quads cut
    where the floor is (convex faces are all `rasterise` fills)."""
    hw, hd, h = room["width"] / 2, room["depth"] / 2, room["height"]
    g = Ground(room)
    out = []
    for (x, _, z), along, length in (((-hw, h, -hd), (1, 0, 0), 2 * hw),     # back
                                     ((hw, h, -hd), (0, 0, 1), 2 * hd),      # right
                                     ((hw, h, hd), (-1, 0, 0), 2 * hw),      # front
                                     ((-hw, h, hd), (0, 0, -1), 2 * hd)):    # left
        origin = (x, h + g.at(x, z), z)
        end = add(origin, mul(along, length))
        if along[2]:
            zs = g.zs(min(z, end[2]), max(z, end[2]))
            cuts = [(x, c) for c in (zs if along[2] > 0 else zs[::-1])]
        else:
            xs = g.xs(min(x, end[0]), max(x, end[0]))
            cuts = [(c, z) for c in (xs if along[0] > 0 else xs[::-1])]
        for (xa, za), (xb, zb) in zip(cuts, cuts[1:]):
            fa, fb = g.at(xa, za), g.at(xb, zb)
            quad = [(xa, fa + h, za), (xb, fb + h, zb), (xb, fb, zb), (xa, fa, za)]
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


def floor_point(cam, room, sx, sy, reach=2 * FLOOR_REACH, step=0.05):
    """Where the ray through (sx, sy) first meets the floor, shaped or
    flat, within `reach` m of the camera, or None."""
    g = Ground(room)
    if g.level:
        return cam.on_floor(sx, sy, g.at(0.0, 0.0))
    d, e = cam.ray(sx, sy), cam.eye
    above = lambda t: e[1] + d[1] * t - g.at(e[0] + d[0] * t, e[2] + d[2] * t)  # noqa: E731
    if above(0.0) <= 0:
        return None
    t = 0.0
    while t < reach:
        if above(t + step) <= 0:
            lo, hi = t, t + step
            for _ in range(24):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if above(mid) > 0 else (lo, mid)
            return add(e, mul(d, hi))
        t += step
    return None


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
    factor instead of painting over it, and `rgb` is only a stand-in.
    `cam` is `pts`' own camera-space (x, y, z=depth), None for the room and
    shadows; `nrm` is the face's unit world-space normal, one for the whole
    face (flat-shaded, so it never varies across it). `id_render` reads
    both - `cam` to interpolate world position, `nrm` to fill it straight
    in. A true depth pass determines visible ownership before filling the
    buffers, including room occlusion and cutout holes."""
    __slots__ = ("pts", "rgb", "depth", "owner", "part", "tex", "dim", "cam", "nrm")

    def __init__(self, pts, rgb, depth, owner, part, tex=None, dim=None, cam=None, nrm=None):
        self.pts, self.rgb, self.depth, self.owner, self.part = pts, rgb, depth, owner, part
        self.tex, self.dim = tex, dim
        self.cam, self.nrm = cam, nrm


def _bilinear(px, tw, th, uc, vc):
    """The picture `px` (`tw` x `th`), sampled at continuous texel
    coordinates (`uc`, `vc`), wrapped and blended across its four neighbours."""
    uc += 1e6
    vc += 1e6
    u0, v0 = int(uc), int(vc)
    fu, fv = uc - u0, vc - v0
    i0, i1 = u0 % tw, (u0 + 1) % tw
    j0, j1 = v0 % th, (v0 + 1) % th
    p00, p10 = px[j0 * tw + i0], px[j0 * tw + i1]
    p01, p11 = px[j1 * tw + i0], px[j1 * tw + i1]
    return bytes(int((p00[k] * (1 - fu) + p10[k] * fu) * (1 - fv) +
                      (p01[k] * (1 - fu) + p11[k] * fu) * fv) for k in range(3))


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
        self.plane(cam, origin, along, down, normal)

    def plane(self, cam, origin, along, down, normal):
        a = add(cam.f, add(mul(cam.r, -cam.w / 2 / cam.k), mul(cam.u, cam.h / 2 / cam.k)))
        lin = lambda v: (dot(a, v), dot(cam.r, v) / cam.k, -dot(cam.u, v) / cam.k)  # noqa
        self.n, self.U, self.V = lin(normal), lin(along), lin(down)
        self.c = dot(sub(origin, cam.eye), normal)
        rel = sub(cam.eye, origin)
        self.eu, self.ev = dot(rel, along), dot(rel, down)

    def paint(self, buf, row, y, xa, xb):
        buf[row + xa * 3:row + (xb + 1) * 3] = self.span(y, xa, xb)

    def span(self, y, xa, xb):
        """Pixels xa..xb of row y, as RGB bytes: bilinear within a mip level
        and blended between the two levels `spread` falls between, so the
        floor does not pop between sharp and blurred as it recedes, and a
        small picture (a flower on grass) does not alias into jagged dots."""
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
            # (further and more edge-on is more): each level is 4x fewer, so
            # that count in "levels" is log base 4 of the spread.
            spread = abs(t / dn) * fine if dn else 1e9
            lv = 0.0 if spread < 1 else min(top, math.log(spread, 4))
            lv0 = int(lv)
            frac = lv - lv0
            u = eu + t * (u0 + ux * xc)
            v = ev + t * (v0 + vx * xc)
            px0, tw0, th0, s0 = levels[lv0]
            c0 = _bilinear(px0, tw0, th0, u * s0, v * s0)
            lv1 = min(top, lv0 + 1)
            if frac and lv1 != lv0:
                px1, tw1, th1, s1 = levels[lv1]
                c1 = _bilinear(px1, tw1, th1, u * s1, v * s1)
                out.append(bytes(int(c0[i] * (1 - frac) + c1[i] * frac) for i in range(3)))
            else:
                out.append(c0)
        return b"".join(out)


class CutMap(TexMap):
    """A cut-out picture on an upright card: once, not repeated, and where
    it is transparent the frame keeps what is behind it."""

    def __init__(self, cam, origin, along, normal, width, height, tex):
        self.tex = tex
        self.ku, self.kv = tex.w / float(width), tex.h / float(height)
        self.plane(cam, origin, along, (0, -1, 0), normal)

    def sample(self, x, y):
        """Texture index and camera depth at a screen point, or None in a hole."""
        dn = self.n[0] + self.n[1] * x + self.n[2] * y
        if not dn:
            return None
        t = self.c / dn
        u = self.eu + t * (self.U[0] + self.U[1] * x + self.U[2] * y)
        v = self.ev + t * (self.V[0] + self.V[1] * x + self.V[2] * y)
        i, j = math.floor(u * self.ku), math.floor(v * self.kv)
        if t >= NEAR and 0 <= i < self.tex.w and 0 <= j < self.tex.h:
            index = j * self.tex.w + i
            if self.tex.alpha[index] >= 128:
                return index, t
        return None

    def paint(self, buf, row, y, xa, xb):
        yc = y + 0.5
        n0, nx = self.n[0] + self.n[2] * yc, self.n[1]
        u0, ux = self.U[0] + self.U[2] * yc, self.U[1]
        v0, vx = self.V[0] + self.V[2] * yc, self.V[1]
        c, eu, ev, ku, kv = self.c, self.eu, self.ev, self.ku, self.kv
        px, alpha, tw, th = self.tex.px, self.tex.alpha, self.tex.w, self.tex.h
        for x in range(xa, xb + 1):
            xc = x + 0.5
            dn = n0 + nx * xc
            if not dn:
                continue
            t = c / dn
            i = int((eu + t * (u0 + ux * xc)) * ku)
            j = int((ev + t * (v0 + vx * xc)) * kv)
            if 0 <= i < tw and 0 <= j < th and alpha[j * tw + i] >= 128:
                o = row + x * 3
                buf[o:o + 3] = px[j * tw + i]


def card(obj, cam, tex):
    """A prop's picture as a Poly: an upright card at its place, as wide
    and tall as the prop, turned square to the camera. None when any corner
    is behind the camera."""
    x, y, z = obj["position"]
    w, h = obj["scale"][0], obj["scale"][1]
    along = norm((cam.r[0], 0, cam.r[2]))
    top_left = (x - along[0] * w / 2, y + h, z - along[2] * w / 2)
    quad = [top_left, add(top_left, mul(along, w)),
            add(top_left, (along[0] * w, -h, along[2] * w)), add(top_left, (0, -h, 0))]
    c = [cam.to_camera(p) for p in quad]
    if any(p[2] < NEAR for p in c):
        return None
    normal = cross(along, (0, 1, 0))
    depth = cam.to_camera((x, y + h / 2, z))[2]
    return Poly([cam.to_screen(p) for p in c], tex.mean, depth, obj["id"], "body",
                CutMap(cam, top_left, along, normal, w, h, tex), cam=c, nrm=normal)


def render(scene, width=None, height=None):
    """The scene through its camera -> [Poly], far to near. The room is
    first - the floor, then the walls facing into it - clipped to the near
    plane; every object face that faces the camera follows, sorted by depth.
    The same list is drawn on the canvas and rasterised by `png`."""
    if width is None:
        width, height = frame_size(scene)
    cam = Camera(scene["camera"], width, height)
    polys = room_polys(scene, cam)
    room = scene.get("room") or new_room()
    tex = texture(room["floor"]["image"])
    floor = tex.mean if tex else FLOOR
    ground = Ground(room)
    faces = []
    for obj in scene["objects"]:
        tex = texture(obj.get("picture"))
        if tex is not None:
            poly = card(obj, cam, tex)
            if poly is not None:
                faces.append(poly)
            continue
        pieces = painted_pieces(obj)
        if obj["asset"] == "crowd":          # each person their own shadow
            members = {}
            for piece in pieces:
                members.setdefault(piece[0], []).append(piece)
            for group in members.values():
                polys += shadow_polys(group, cam, floor, ground)
        else:
            polys += shadow_polys(pieces, cam, floor, ground)
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
                                  obj["id"], part, cam=c, nrm=norm(n)))
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


def shadow_polys(pieces, cam, floor=FLOOR, ground=None):
    """The soft shadow an object leaves on the floor, as `dim` polys: a
    dark contact ring under each part that touches (each foot, a knee, a
    box's base) and a faint one under the whole body. Only for an object on
    the floor itself: one standing on a platform above it would have its
    shadow drawn under the platform, since the room is drawn first. On a
    shaped floor (`ground`) heights are above the floor under each point,
    and the rings lie on it."""
    g = ground or Ground(None)
    pts = [p for _, faces, _ in pieces for f in faces for p in f]
    if not pts or cam.eye[1] <= g.at(cam.eye[0], cam.eye[2]):
        return []
    if g.grid:
        # Under one object the floor is as good as a plane: the one touching
        # it under the object's middle, which is quick to ask of every point.
        mx = sum(p[0] for p in pts) / len(pts)
        mz = sum(p[2] for p in pts) / len(pts)
        h0, n = g.at(mx, mz), g.slope(mx, mz)
        gx, gz = -n[0] / n[1], -n[2] / n[1]
        up = lambda p: p[1] - h0 - gx * (p[0] - mx) - gz * (p[2] - mz)  # noqa: E731
    else:
        up = lambda p: p[1]  # noqa: E731
    low = min(up(p) for p in pts)
    if low > 0.02:
        return []
    touching = {}
    for part, faces, _ in pieces:
        for f in faces:
            for p in f:
                if up(p) <= low + CONTACT_REACH:
                    touching.setdefault(part, []).append((p[0], p[2]))
    whole = hull([(p[0], p[2]) for p in pts if up(p) <= low + AMBIENT_REACH])
    rings = [(whole, AMBIENT_SHADOW)] if len(whole) >= 3 else []
    rings += [(hull(foot), CONTACT) for foot in touching.values()]
    out = []
    for outline, steps in rings:
        if not outline:
            continue
        seen = 1.0
        for r, k in steps:
            seen *= k
            c = clip_near([cam.to_camera((x, g.at(x, z) + 0.001, z))
                           for x, z in grown(outline, r)])
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
    ground = Ground(room)
    r = FLOOR_REACH
    out = floor_polys(ground, cam, room["floor"], (-r, r, -r, r))
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
    if not ground.level:
        # A rise in the floor can stand in front of a wall's foot, and the
        # walls were drawn over it: the floor inside them again, on top.
        hw, hd = room["width"] / 2, room["depth"] / 2
        out += floor_polys(ground, cam, room["floor"], (-hw, hw, -hd, hd))
    return out


def floor_polys(ground, cam, face, box):
    """The floor over `box` (x0, x1, z0, z1) as faces towards the camera,
    far to near, each lit by its slope (level is as bright as the flat floor
    always was). A picture is laid on from straight above, x and z, so it
    runs on unbroken from face to face."""
    tex = texture(face["image"])
    flat = AMBIENT + (1 - AMBIENT) * max(0.0, LIGHT_DIR[1])
    out = []
    for pts, n in ground.faces(*box):
        if dot(n, sub(pts[0], cam.eye)) >= 0:
            continue                             # seen from below
        c = clip_near([cam.to_camera(p) for p in pts])
        if len(c) < 3:
            continue
        mid = centroid(pts)
        # Lit by the smooth surface's slope at its middle, not the face's
        # own: the two triangles of a cell then shade alike, not faceted.
        lit = ground.slope(mid[0], mid[2]) if ground.grid else n
        k = (AMBIENT + (1 - AMBIENT) * max(0.0, dot(lit, LIGHT_DIR))) / flat
        rgb = tuple(min(255, int(v * k)) for v in (tex.mean if tex else FLOOR))
        m = None
        if tex:
            p0 = pts[0]                          # the face's plane over x = z = 0
            origin = (0.0, p0[1] + (n[0] * p0[0] + n[2] * p0[2]) / n[1], 0.0)
            m = TexMap(cam, origin, (1, 0, 0), (0, 0, 1), n, face["size"], tex,
                       round(round(k / FLOOR_SHADE) * FLOOR_SHADE, 3))
        far = dot(sub(mid, cam.eye), sub(mid, cam.eye))
        out.append((-far, Poly([cam.to_screen(p) for p in c], rgb, float("inf"), None,
                               None, m)))
    out.sort(key=lambda t: t[0])
    return [p for _, p in out]


def grid_lines(scene, width, height, spacing=1.0, reach=10):
    """The floor's metre grid as screen segments, for the window only: the
    picture must not be told the floor is tiled. Inside the walls when there
    are walls, since the grid is drawn over the whole room. On a shaped
    floor it lies on it, each line in a segment per cut of the floor."""
    cam = Camera(scene["camera"], width, height)
    room = scene.get("room") or new_room()
    g = Ground(room)
    if room["walls"]:
        hw, hd = room["width"] / 2, room["depth"] / 2
        xs = [(x, x == 0) for x in range(int(math.ceil(-hw)), int(math.floor(hw)) + 1)]
        zs = [(z, z == 0) for z in range(int(math.ceil(-hd)), int(math.floor(hd)) + 1)]
        x0, x1, z0, z1 = -hw, hw, -hd, hd
    else:
        xs = zs = [(i * spacing, i == 0) for i in range(-reach, reach + 1)]
        x0, x1, z0, z1 = -reach, reach, -reach, reach

    def seen(p, q):
        """The floor between p and q faces the camera: the far side of a
        rise is out of sight, and its lines with it."""
        mid = mul(add(p, q), 0.5)
        return dot(g.slope(mid[0], mid[2]), sub(mid, cam.eye)) < 0
    lines = []
    for x, axis in xs:
        run = [(x, g.at(x, z), z) for z in g.zs(z0, z1)]
        lines += [(p, q, axis) for p, q in zip(run, run[1:]) if seen(p, q)]
    for z, axis in zs:
        run = [(x, g.at(x, z), z) for x in g.xs(x0, x1)]
        lines += [(p, q, axis) for p, q in zip(run, run[1:]) if seen(p, q)]
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

def _spans(poly, width, height):
    """(y, xa, xb) for every screen row a convex poly covers, in frame
    pixels - xa..xb inclusive, a row with no width left out. The scanline
    walk `rasterise` and `id_render` both fill from."""
    pts = poly.pts
    ys = [p[1] for p in pts]
    y0 = max(0, int(math.ceil(min(ys) - 0.5)))
    y1 = min(height - 1, int(math.floor(max(ys) - 0.5)))
    if y1 < y0:
        return
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
        if xb >= xa:
            yield y, xa, xb


def rasterise(polys, width, height, sky=SKY, flat=False):
    """Painter's-order polygons -> RGB bytes. A scanline fill of convex
    polygons; every face the renderer makes is convex. A face with a picture
    is filled from it, unless `flat` (its mean colour: quick)."""
    buf = bytearray(bytes(sky) * (width * height))
    stride = width * 3
    for poly in polys:
        colour = bytes(poly.rgb)
        tex = None if flat else poly.tex
        dim = bytes(int(v * poly.dim) for v in range(256)) if poly.dim else None
        for y, xa, xb in _spans(poly, width, height):
            row = y * stride
            a, b = row + xa * 3, row + (xb + 1) * 3
            if dim:
                buf[a:b] = bytes(buf[a:b]).translate(dim)
            elif tex:
                tex.paint(buf, row, y, xa, xb)
            else:
                buf[a:b] = colour * (xb - xa + 1)
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
    One for a person, one per member for a crowd, none for a prop. The
    skeleton also has "hands": {"l"/"r": 21 points} as `hand_joints` puts
    them, in the skeleton's space, each hand closed on what it holds."""
    def placed(controls, root, shape, look, k, x, y, z):
        dressed = outfit(look)
        low = min(p[1] for _, faces, _ in person_pieces(controls, root, shape, dressed)
                  for f in faces for p in f) * k
        controls = gripped(controls, dressed.get("held") or {})
        sk = skeleton(controls, root, shape)
        sk["hands"] = {}
        for side, sign in (("l", 1), ("r", -1)):
            wp, wm = sk["wrist_" + side]
            grip = tuple(float(controls.get(g % side, 0) or 0) for g in GRIP_KEYS[1:])
            sk["hands"][side] = [add(wp, apply(wm, v)) for v in hand_joints(sign, grip)]
        return sk, k, (x, y - low, z)
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
            head = obj.get("head") if obj["asset"] == "person" else None
            coco.update({i: world(add(hp, apply(hm, head_point(v, head))))
                         for i, v in KP_HEAD.items()})
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
                    s = cam.project(world(add(hp, apply(hm, head_point(local, head)))))
                    if s:
                        face.append((s[0] / width, s[1] / height))
            # The hands, as DWPose finds them: only where the wrist is seen,
            # a finger point hidden by what stands in front left out.
            hands = {}
            for side, name, wi in (("r", "right", 4), ("l", "left", 7)):
                if not pts[wi]:
                    continue
                hp2 = []
                for p in sk["hands"][side]:
                    s = cam.project(world(p))
                    hp2.append([s[0] / width, s[1] / height] if s and not hidden(world(p))
                               else None)
                hands[name] = hp2
            out.append({"id": obj["id"], "points": pts, "face": face, "hand_points": hands,
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


def depth_values(scene, width, height, relief=1.0, window=None):
    """1/z (1/m) of the nearest surface at each pixel, row by row, 0 where
    the camera sees only sky: the floor, the walls that face into the room,
    and every object, as `render` draws them but without the shadows.

    `relief` stretches each body's depth about its own middle (a person, a
    crowd member, a prop), where it is on screen unchanged: a body 0.3 m
    thick at 5 m is a flat cut-out in true 1/z, where Depth Anything - what
    the ControlNet learnt from - gives it rounded relief.
    `window` (x, y, w, h), a part of the frame in its own pixels, is drawn
    alone at `width` x `height` instead of the whole frame."""
    cam = Camera(scene["camera"], width, height)
    if window:
        fw, fh = frame_size(scene)
        wx, wy, ww, wh = window
        full = Camera(scene["camera"], fw, fh)
        kx, ky = width / float(ww), height / float(wh)
        cam.to_screen = lambda c: ((full.to_screen(c)[0] - wx) * kx,     # noqa: E731
                                   (full.to_screen(c)[1] - wy) * ky)
    room = scene.get("room") or new_room()
    r = FLOOR_REACH
    groups = [[pts for pts, n in Ground(room).faces(-r, r, -r, r)
               if dot(n, sub(pts[0], cam.eye)) < 0]]
    if room["walls"]:
        groups[0] += [quad for quad, origin, _ in walls(room)
                      if dot(newell(quad), sub(origin, cam.eye)) < 0]
    pictures = []
    for obj in scene["objects"]:
        tex = texture(obj.get("picture"))
        if tex is not None:
            pictures.append((obj, tex))
            continue
        bodies = {}
        for part, fs, _ in painted_pieces(obj):
            key = part if obj["asset"] == "crowd" else None
            bodies.setdefault(key, []).extend(
                f for f in fs if dot(newell(f), sub(centroid(f), cam.eye)) < 0)
        groups += bodies.values()
    zb = [0.0] * (width * height)
    for n, faces in enumerate(groups):
        cams = [clip_near([cam.to_camera(p) for p in f]) for f in faces]
        cams = [c for c in cams if len(c) >= 3]
        if not cams:
            continue
        mid = None
        if n and relief != 1.0:
            zs = [p[2] for c in cams for p in c]
            mid = (min(zs) + max(zs)) / 2
        for c in cams:
            _fill_depth(zb, width, height,
                        [cam.to_screen(p) + (1.0 / (p[2] if mid is None else
                                                    max(NEAR, mid + relief * (p[2] - mid))),)
                         for p in c])
    for obj, tex in pictures:
        poly = card(obj, full if window else cam, tex)
        if poly is None:
            continue
        pts = ([((x - wx) * kx, (y - wy) * ky) for x, y in poly.pts]
               if window else poly.pts)
        xa = max(0, math.floor(min(p[0] for p in pts)))
        xb = min(width, math.ceil(max(p[0] for p in pts)))
        ya = max(0, math.floor(min(p[1] for p in pts)))
        yb = min(height, math.ceil(max(p[1] for p in pts)))
        for y in range(ya, yb):
            for x in range(xa, xb):
                sx, sy = ((wx + (x + 0.5) / kx, wy + (y + 0.5) / ky)
                          if window else (x + 0.5, y + 0.5))
                hit = poly.tex.sample(sx, sy)
                if hit:
                    i = y * width + x
                    zb[i] = max(zb[i], 1.0 / hit[1])
    return zb


def depth_png(scene, edge=DEPTH_EDGE):
    """The frame as a depth map, the kind a depth ControlNet was trained on
    (Depth Anything's): grey by nearness - 1/z from the farthest thing seen
    (black) to the nearest (white), the sky black. Smaller than the frame,
    with the same shape: the ControlNet scales it to the picture."""
    w, h = frame_size(scene)
    k = edge / float(max(w, h))
    dw, dh = max(64, int(round(w * k))), max(64, int(round(h * k)))
    zb = depth_values(scene, dw, dh, DEPTH_RELIEF)
    seen = sorted(q for q in zb if q > 0)
    if seen:
        lo = seen[int(DEPTH_CLIP[0] * (len(seen) - 1))]
        hi = seen[int(DEPTH_CLIP[1] * (len(seen) - 1))]
    else:
        lo, hi = 0.0, 1.0
    span = (hi - lo) or 1.0
    grey = bytes(0 if q <= 0 else max(1, min(255, int(round(255 * (q - lo) / span))))
                 for q in zb)
    rgb = bytearray(dw * dh * 3)
    for i in range(3):
        rgb[i::3] = grey
    return rgb_png(bytes(rgb), dw, dh)


HEAD_REACH = 0.25              # m behind the nearest point a face crop's depth map spans


def depth_crop_png(scene, region, size, oid=None):
    """The depth map of one part of the frame, `region` (x, y, w, h) in the
    frame's pixels, drawn at `size` (w, h): a face crop for the face pass.
    Its grey spans only HEAD_REACH behind the nearest point - of the head of
    person `oid` when given, the shoulders in front of it white - so the
    face's own relief, jaw, cheek, chin, fills it, as Depth Anything would
    make of a close-up; what is further is black."""
    dw, dh = size
    zb = depth_values(scene, dw, dh, window=region)
    seen = [q for q in zb if q > 0]
    if not seen:
        return None
    hi = max(seen)
    obj = next((o for o in scene["objects"] if o["id"] == oid), None) if oid else None
    if obj is not None:
        cam = Camera(scene["camera"], *frame_size(scene))
        zs = [cam.to_camera(p)[2] for part, faces, _ in painted_pieces(obj) if part == "head"
              for f in faces for p in f]
        zs = [z for z in zs if z > NEAR]
        if zs:
            hi = 1.0 / min(zs)
    lo = 1.0 / (1.0 / hi + HEAD_REACH)
    span = (hi - lo) or 1.0
    grey = bytes(0 if q <= lo else min(255, int(round(255 * (q - lo) / span))) for q in zb)
    rgb = bytearray(dw * dh * 3)
    for i in range(3):
        rgb[i::3] = grey
    return rgb_png(bytes(rgb), dw, dh)


# ===================================================================== ids
# Which pixels belong to which thing, which way they face, and where they
# are - not guessed by a model after the fact (SAM over the finished
# picture, a depth net's relief) but read straight off the same polygons
# pose and depth already come from, at the same boundaries: a hand round a
# cup, hair over the background, two people touching. Not sent to
# generation - only pose and depth condition the ControlNet (`scene_maps`) -
# these are render artifacts for what comes after: a mask, a regional
# regeneration, replacing one person's clothes without touching their skin,
# compositing a corrected arm back in, or - from `world` - answering "what
# 3D point is this bad pixel" at all.
#
# Four buffers, `id_render`'s pass over `render`'s polygons, gated by
# true camera depth from `depth_values`:
# `instance` - which body a pixel belongs to (an object is one instance, a
# crowd one per member), one byte, 0 for the room and its shadows.
# `part` - which of that body's pieces (a crowd member is one piece,
# "body"), one byte, 0 the same way. Both: a part only gets a number when
# the render actually paints one of its pixels, so the numbering is only
# promised to agree with itself within one call, not across a moved camera.
# `normal` - the face's own unit world-space normal (flat-shaded, so it is
# the same at every pixel of one face) - three `array('f')` floats, (0,0,0)
# for the room. `world` - the exact world-space point under the pixel,
# perspective-correct across the face (`_plane`), not just its centre -
# three `array('f')` floats, (0,0,0) for the room.
ID_LIMIT = 255                 # the id buffers are one byte a pixel; the 256th body shares 255's


def _stable_triangle(xy):
    """The indices (0, i, i+1) of a convex screen poly's least edge-on
    triangle, for `_plane` to fit against - None if the whole poly is
    edge-on (a sliver, invisible either way)."""
    x0, y0 = xy[0]
    best = None
    for i in range(1, len(xy) - 1):
        x1, y1 = xy[i]
        x2, y2 = xy[i + 1]
        det = abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
        if best is None or det > best[0]:
            best = (det, i)
    if best is None or best[0] < 1e-6:
        return None
    return 0, best[1], best[1] + 1


def _plane(tri, xy, qs):
    """(a, b, c): q = a + b*x + c*y across the screen, fit to `qs` at the
    three screen points `xy` names by `tri` - affine across a flat, convex
    poly under perspective projection, the trick `_fill_depth` uses for
    1/z, reused here for camera-space x/z and y/z too."""
    i0, i1, i2 = tri
    x0, y0 = xy[i0]
    x1, y1 = xy[i1]
    x2, y2 = xy[i2]
    q0, q1, q2 = qs[i0], qs[i1], qs[i2]
    det = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    b = ((q1 - q0) * (y2 - y0) - (q2 - q0) * (y1 - y0)) / det
    c = ((x1 - x0) * (q2 - q0) - (x2 - x0) * (q1 - q0)) / det
    a = q0 - b * x0 - c * y0
    return a, b, c


def _instance_owners(scene):
    """{(object id, crowd member key or None): instance id}, 1 up in scene
    order, capped at ID_LIMIT. A crowd is one instance per member (its
    pieces are already parted "m0", "m1", ... by `crowd_pieces`); anything
    else is one instance for the whole object."""
    owners = {}
    for obj in scene["objects"]:
        if obj["asset"] == "crowd":
            for m in range(len(crowd_members(obj["crowd"]))):
                owners[(obj["id"], "m%d" % m)] = min(ID_LIMIT, len(owners) + 1)
        else:
            owners[(obj["id"], None)] = min(ID_LIMIT, len(owners) + 1)
    return owners


def _instance_meta(obj, member):
    """The sidecar entry for one instance: `member` is a crowd piece's part
    ("m3"), or None for a plain object."""
    if member is not None:
        return {"owner": obj["id"], "name": "%s %s" % (obj["name"], member[1:]),
                "type": "crowd_member"}
    meta = {"owner": obj["id"], "name": obj["name"], "type": obj["asset"]}
    if obj["asset"] == "person":
        meta["character"] = obj.get("character") or ""
    return meta


def id_render(scene, width=None, height=None):
    """-> (instance bytes, part bytes, normal floats, world floats,
    sidecar): every per-pixel truth buffer, a pass over `render`'s own
    polygons (`_spans`) - instance and part a byte a pixel, normal and
    world three `array('f')` floats a pixel (x, y, z in a pixel's 3 slots),
    all 0 for the room and its shadows. `sidecar` names the instance and
    part numbers -
    `{"instances": {"<id>": {"owner", "name", "type"[, "character"]}},
    "parts": {"<id>": {"<part id>": "<label>"}}}` - JSON-safe (string keys).

    `normal` is one vector for a whole face (flat-shaded, so it does not
    vary across it); true camera depth selects the nearest surface,
    independently of painter order. `world` is the exact point under each pixel,
    perspective-correct across the face, not just its centre - so a bad
    pixel's instance, part, facing and 3D position can all be read back."""
    if width is None:
        width, height = frame_size(scene)
    owners = _instance_owners(scene)
    by_id = {obj["id"]: obj for obj in scene["objects"]}
    cam = Camera(scene["camera"], width, height)
    ex, ey, ez = cam.eye
    rx, ry, rz = cam.r
    ux, uy, uz = cam.u
    fx, fy, fz = cam.f
    n = width * height
    inst, part = bytearray(n), bytearray(n)
    normal, world = array.array("f", [0.0]) * (n * 3), array.array("f", [0.0]) * (n * 3)
    # True camera depth, including room surfaces and transparent cards.
    # Average polygon depth cannot order intersecting limbs or two people
    # whose surfaces exchange which one is in front across the frame.
    nearest = depth_values(scene, width, height)
    parts, meta = {}, {}                   # instance id -> {label: part id}; -> sidecar entry
    for poly in render(scene, width, height):
        if poly.owner is None:
            continue
        crowd = bool(poly.part) and poly.part[0] == "m" and poly.part[1:].isdigit()
        iid = owners.get((poly.owner, poly.part if crowd else None))
        if iid is None:
            continue
        spans = list(_spans(poly, width, height))
        if not spans:
            continue                       # projects off-canvas: paints nothing, means nothing
        if iid not in meta:
            meta[iid] = _instance_meta(by_id[poly.owner], poly.part if crowd else None)
        label = "body" if crowd else poly.part
        seen = parts.setdefault(iid, {})
        pid = seen.get(label)
        if pid is None:
            pid = seen[label] = min(ID_LIMIT, len(seen) + 1)
        nx, ny, nz = poly.nrm or (0.0, 0.0, 0.0)
        tri = _stable_triangle(poly.pts) if poly.cam else None
        pz = px = py = None
        if tri:
            iz = [1.0 / c[2] for c in poly.cam]
            xz = [c[0] / c[2] for c in poly.cam]
            yz = [c[1] / c[2] for c in poly.cam]
            pz, px, py = (_plane(tri, poly.pts, iz), _plane(tri, poly.pts, xz),
                          _plane(tri, poly.pts, yz))
            az, bz, cz = pz
            ax, bx, cx = px
            ay, by, cy = py
        for y, xa, xb in spans:
            row = y * width
            yc = y + 0.5
            for x in range(xa, xb + 1):
                xc = x + 0.5
                if pz is None:
                    continue
                q = az + bz * xc + cz * yc
                if q <= 0 or q < nearest[row + x] - max(1e-8, abs(q) * 1e-7):
                    continue
                if isinstance(poly.tex, CutMap) and poly.tex.sample(xc, yc) is None:
                    continue
                inst[row + x], part[row + x] = iid, pid
                o = (row + x) * 3
                normal[o], normal[o + 1], normal[o + 2] = nx, ny, nz
                z = 1.0 / q
                cx_, cy_ = (ax + bx * xc + cx * yc) * z, (ay + by * xc + cy * yc) * z
                world[o] = ex + cx_ * rx + cy_ * ux + z * fx
                world[o + 1] = ey + cx_ * ry + cy_ * uy + z * fy
                world[o + 2] = ez + cx_ * rz + cy_ * uz + z * fz
    sidecar = {"instances": {str(iid): m for iid, m in meta.items()},
               "parts": {str(iid): {str(pid): label for label, pid in labels.items()}
                         for iid, labels in parts.items()}}
    return bytes(inst), bytes(part), normal, world, sidecar


def surface_render(scene, width=None, height=None):
    """Shared truth buffers plus numerical incidence, edges and illumination.

    Float maps are ephemeral analysis, never quantised PNGs or saved settings.
    Lighting is Lambertian incidence only, without shadows or material colour.
    """
    from apps.image_studio.scene.surface import analyse
    if width is None:
        width, height = frame_size(scene)
    inst, part, normals, world, sidecar = id_render(scene, width, height)
    cam = Camera(scene["camera"], width, height)
    maps = analyse(inst, part, normals, world, width, height, cam.eye, cam.f, LIGHT_DIR)
    maps.update(instance=inst, part=part, normal=normals, world=world,
                sidecar=sidecar, width=width, height=height)
    return maps


def identity_masks(scene, targets, long_edge=256):
    """{person id: PNG or None}: surface-aware masks for initial identity attention.

    Render once at bounded resolution for all people, at the scene's aspect.
    None means no eligible visible head: never substitute a rectangle that
    could condition an occluder. Form-only identity regions keep their own path.
    """
    from apps.image_studio.scene.surface import identity_mask
    targets = [p for p in targets if p.get("id")]
    if not targets:
        return {}
    fw, fh = frame_size(scene)
    k = min(1.0, long_edge / float(max(fw, fh)))
    w, h = max(1, round(fw * k)), max(1, round(fh * k))
    maps = surface_render(scene, w, h)
    by_owner = {m["owner"]: int(sid) for sid, m in maps["sidecar"]["instances"].items()
                if m["type"] == "person"}
    out = {}
    for person in targets:
        oid = person["id"]
        iid = by_owner.get(oid)
        # Identity belongs to the face, not the rear of a head. Profile views
        # remain eligible; the per-pixel incidence handles their grazing edge.
        if iid is None or (person.get("head_pose") or {}).get("facing_cosine", 1) < -0.1:
            out[oid] = None
            continue
        head_parts = {int(pid) for pid, label in maps["sidecar"]["parts"][str(iid)].items()
                      if label == "head"}
        alpha = identity_mask(maps["instance"], maps["part"], maps, w, h,
                              iid, head_parts, person["region"])
        if not any(alpha):
            out[oid] = None
            continue
        rgb = bytearray(w * h * 3)
        for channel in range(3):
            rgb[channel::3] = alpha
        out[oid] = rgb_png(bytes(rgb), w, h)
    return out


def instance_id_png(scene):
    """The instance buffer as a picture: its byte in all three channels, so
    it reads as flat grey regions and decodes exactly from any one of
    them."""
    inst = id_render(scene)[0]
    w, h = frame_size(scene)
    rgb = bytearray(w * h * 3)
    for i in range(3):
        rgb[i::3] = inst
    return rgb_png(bytes(rgb), w, h)


def part_id_png(scene):
    """The part buffer as a picture: red is the instance byte (so it reads
    the same as `instance_id_png`'s red channel), green the part byte, blue
    unused - each exact on its own channel, nothing blended across them."""
    inst, part = id_render(scene)[:2]
    w, h = frame_size(scene)
    rgb = bytearray(w * h * 3)
    rgb[0::3] = inst
    rgb[1::3] = part
    return rgb_png(bytes(rgb), w, h)


def normal_png(scene):
    """The normal buffer as a picture, the usual normal-map convention:
    each axis's [-1, 1] mapped to [0, 255], only for this picture - the
    buffer itself (`id_render`) keeps the real floats, not this
    quantising."""
    normal = id_render(scene)[2]
    w, h = frame_size(scene)
    rgb = bytearray(w * h * 3)
    for i in range(w * h):
        for k in range(3):
            rgb[i * 3 + k] = max(0, min(255, int(round((normal[i * 3 + k] + 1) * 127.5))))
    return rgb_png(bytes(rgb), w, h)


def id_map(scene):
    """The sidecar alone - what `instance_id_png`/`part_id_png`'s numbers
    mean - without rendering any buffer."""
    return id_render(scene)[4]


def _box_blur_1d(buf, length, count, base_stride, elem_stride, radius):
    """`count` lines of `length` values each - line `i` starts at
    `i * base_stride` and its values are `elem_stride` apart - box-averaged
    along their own length, clipped at each end (so an edge does not darken),
    via a running prefix sum."""
    out = bytearray(len(buf))
    for i in range(count):
        base = i * base_stride
        line = buf[base:base + length] if elem_stride == 1 else \
            bytes(buf[base + j * elem_stride] for j in range(length))
        prefix = [0]
        prefix.extend(itertools.accumulate(line))
        for x in range(length):
            lo, hi = max(0, x - radius), min(length - 1, x + radius)
            total = prefix[hi + 1] - prefix[lo]
            out[base + x * elem_stride] = total // (hi - lo + 1)
    return out


def _feather(buf, width, height, radius):
    """A 0/255 mask, softened `radius` pixels at its edge: a box blur along
    rows then columns, so a masked composite has no hard seam."""
    if radius <= 0:
        return buf
    rows = _box_blur_1d(buf, width, height, width, 1, radius)
    return _box_blur_1d(rows, height, width, 1, width, radius)


def _character_mask_buffers(scene, feather=6):
    """(width, height, {character id: bytearray}) - the pixels `character_masks`
    turns into PNGs, kept separate so a test can check a mask's own bytes
    without decoding a PNG back out."""
    inst, _, _, _, sidecar = id_render(scene)
    w, h = frame_size(scene)
    by_character = {}
    for iid_str, meta in sidecar["instances"].items():
        char = meta.get("character")
        if char:
            by_character.setdefault(char, set()).add(int(iid_str))
    buffers = {}
    for char, iids in by_character.items():
        table = bytes(255 if i in iids else 0 for i in range(256))
        hard = bytearray(inst.translate(table))
        if feather > 0:
            from apps.image_studio.scene.distance import signed_distance, feather_distance
            buffers[char] = feather_distance(signed_distance(hard, w, h), feather)
        else:
            buffers[char] = hard
    return w, h, buffers


def character_distance_fields(scene):
    """(width, height, {character: signed float pixels}) from visible ownership.

    Positive inside each character, negative outside, in frame pixels.
    Occluded geometry does not contribute to this visible boundary.
    """
    from apps.image_studio.scene.distance import signed_distance
    w, h, masks = _character_mask_buffers(scene, feather=0)
    return w, h, {char: signed_distance(mask, w, h) for char, mask in masks.items()}


def character_masks(scene, feather=6):
    """{character id: mask png bytes} - one soft-edged mask per named
    character in this scene, from the instance buffer `id_render` already
    computes: 255 where that character's own instance painted a pixel, 0
    elsewhere. Occlusion and overlap fall out for free (a covered character's
    hidden pixels were never painted with their id; hard masks never
    claim the same pixel, while their feathers may overlap). A crowd member
    or a person with no character
    assigned has no `"character"` in the sidecar, so it gets no mask -
    regional prompting is for named characters only."""
    w, h, buffers = _character_mask_buffers(scene, feather)
    masks = {}
    for char, buf in buffers.items():
        rgb = bytearray(w * h * 3)
        for i in range(3):
            rgb[i::3] = buf
        masks[char] = rgb_png(bytes(rgb), w, h)
    return masks


def beard_mask_buffers(scene, width=None, height=None, feather=2):
    """Visible beard coverage by person id, including unnamed people.

    Sample the existing world/ownership buffers in the posed head frame.
    Covered faces and rear heads therefore contribute no beard pixels.
    Feathering is clipped to the same person's visible head.
    """
    from apps.image_studio.scene import beard
    if width is None:
        width, height = frame_size(scene)
    candidates = {}
    for obj in scene["objects"]:
        b = beard.clean((obj.get("look") or {}).get("beard"))
        if obj["asset"] == "person" and b and b["style"] != "none" and b["density"] and b["coverage"]:
            sk, k, shift = rigs(obj)[0]
            origin, rotation = sk["head"]
            candidates[obj["id"]] = (b, k, shift, origin, transpose(rotation))
    if not candidates:
        return width, height, {}
    inst, parts, _, world, sidecar = id_render(scene, width, height)
    buffers = {}
    for sid, meta in sidecar["instances"].items():
        oid = meta["owner"]
        if oid not in candidates:
            continue
        b, k, shift, origin, inverse = candidates[oid]
        head_ids = {int(pid) for pid, label in sidecar["parts"][sid].items() if label == "head"}
        buf, visible = bytearray(width * height), bytearray(width * height)
        for i, iid in enumerate(inst):
            if iid != int(sid) or parts[i] not in head_ids:
                continue
            visible[i] = 255
            point = tuple(world[3 * i + j] for j in range(3))
            local = apply(inverse, sub(mul(sub(point, shift), 1 / k), origin))
            if beard.contains(local, b):
                buf[i] = 255
        if any(buf):
            if feather:
                buf = _feather(buf, width, height, feather)
                for i in range(len(buf)):
                    if not visible[i]:
                        buf[i] = 0
            buffers[oid] = buf
    return width, height, buffers


def beard_masks(scene, feather=2):
    """{person id: RGB mask PNG}, ready for regional conditioning."""
    w, h, buffers = beard_mask_buffers(scene, feather=feather)
    masks = {}
    for oid, buf in buffers.items():
        rgb = bytearray(w * h * 3)
        for channel in range(3):
            rgb[channel::3] = buf
        masks[oid] = rgb_png(bytes(rgb), w, h)
    return masks


def framing(scene):
    """-> (name, factor): the shot by its biggest face (FRAMING), and how
    much of the depth strength it gets."""
    tall = max((t["tall"] for t in face_targets(scene)), default=0.0)
    for least, name, factor in FRAMING:
        if tall >= least:
            return name, factor
    return FRAMING[-1][1:]


def head_box(scene, obj):
    """Where a person's head is in the frame, (x, y, w, h) in its pixels, or
    None when it is not in front of the camera."""
    w, h = frame_size(scene)
    cam = Camera(scene["camera"], w, h)
    pts = [cam.project(p) for part, faces, _ in painted_pieces(obj) if part == "head"
           for f in faces for p in f]
    pts = [p for p in pts if p]
    if not pts:
        return None
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))


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
    if not maps and "face_positions" in takes:
        notes.append("Face positions and scene descriptions are sent; body poses and props "
                     "are guided by words.")
    elif not maps:
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
             "raised, the hand at head height": "both arms raised, the hands at head height",
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
    # A hand between the chin and the crown, in front, is raised (a toast),
    # not at shoulder height: said as forward, the picture drew a second
    # forearm out in front as well as the raised arm the pose map shows.
    if w[1] > sk["head"][0][1] and ahead >= out:
        return "raised, the hand at head height"
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


HELD_WORDS = {"stein": "fingers wrapped round the beer stein",
              "spatula": "gripping a spatula",
              "pretzel": "pinching the pretzel between thumb and fingers"}


def accordion_front(chest_deep, bust):
    """Instrument centre in the chest frame, shared by mesh and contact checks."""
    return 0.11 * chest_deep * (1 + 0.3 * bust) + 0.06 * max(0.0, bust) + 0.13


def accordion_contacts(controls, look=None):
    """Nearest hand-landmark gaps to the playing surfaces, in rig metres.

    Uses the mesh's clothed chest depth, rig and finger points. Right
    fingertips meet the keyboard; left palm landmarks meet the bass end.
    This checks contact without changing the user's pose.
    """
    shape, dressed = body_shape(look), outfit(look or {})
    c = gripped(controls, {"accordion": (("r",), None)})
    sk = skeleton(c, IDENTITY, shape)
    cp, cm = sk["chest"]
    depth = 1 + 0.7 * (shape["fat"] - 1) + shape["muscle"] - 1
    if "chest" in (dressed.get("regions") or {}):
        depth *= 1.04
    z = accordion_front(depth, shape["chest"] - 1)
    surfaces = {"r": ((-0.225, -0.18, z + 0.1), (-0.155, 0.1, z + 0.115)),
                "l": ((0.23, -0.2, z - 0.1), (0.23, 0.12, z + 0.1))}
    out = {}
    for side, sign in (("r", -1), ("l", 1)):
        wp, wm = sk["wrist_" + side]
        grip = tuple(float(c.get(g % side, 0) or 0) for g in GRIP_KEYS[1:])
        pts = [apply(transpose(cm), sub(add(wp, apply(wm, p)), cp))
               for p in hand_joints(sign, grip)]
        lo, hi = surfaces[side]
        def gap(p):
            return math.sqrt(sum(max(lo[i] - p[i], 0, p[i] - hi[i]) ** 2 for i in range(3)))
        indices = (8, 12, 16, 20) if side == "r" else (5, 9, 13, 17)
        out[side] = min(gap(pts[i]) for i in indices)
    return out


def hand_words(controls, held, look=None):
    """What each hand is doing, as phrases, for the fingers the pose map's
    hands are too small to carry alone. A hand at rest says nothing."""
    c = gripped(controls, held)
    g = lambda k: float(c.get(k, 0) or 0)                 # noqa: E731
    if "accordion" in held:
        gaps = accordion_contacts(controls, look)
        return [phrase for side, phrase in (
            ("r", "right fingers on the accordion keys"),
            ("l", "left hand on the accordion's bass end")) if gaps[side] <= 0.035]
    out = []
    for side, name in (("r", "right"), ("l", "left")):
        kind = next((k for k in HELD_WORDS if side in held.get(k, ((),))[0]), None)
        curl, spread = g("fingers_%s_curl" % side), g("fingers_%s_spread" % side)
        if kind:
            words = HELD_WORDS[kind]
        elif curl >= 60:
            words = "clenched in a fist" if g("thumb_%s_curl" % side) >= 40 else "fingers curled in"
        elif curl >= 25:
            words = "fingers loosely curled"
        elif spread >= 12:
            words = "open with the fingers spread"
        else:
            continue
        out.append("%s hand %s" % (name, words))
    return out


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
    out += hand_words(c, outfit(look).get("held") or {}, look)
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
    import apps.image_studio.imagegen as ig
    c = scene["camera"]
    # At one of the camera's own lenses, its aperture is said too.
    lens = ig.lens_at(c.get("lenses"), c["lens"])
    if c["pitch"] > 35:
        angle = "a high angle looking down"
    elif c["pitch"] > 12:
        angle = "slightly above, looking down"
    elif c["pitch"] < -12:
        angle = "a low angle looking up"
    else:
        angle = "eye level"
    words = "Shot from %s on %s" % (angle, ig.lens_words(c["lens"], lens and lens["f"]))
    chemistry = (c.get("chemistry") or "").strip()
    body = (c.get("body") or "").strip()
    if body and body.lower() not in chemistry.lower():   # a camera whose words skip its name
        chemistry = ("Shot on a %s. %s" % (body, chemistry)).strip()
    return words + ". " + chemistry if chemistry else words


def in_frame(pts, w, h):
    """Whether screen points' box overlaps the w x h frame."""
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return bool(pts) and max(xs) > 0 and min(xs) < w and max(ys) > 0 and min(ys) < h


def room_seen(scene):
    """The room's surfaces the viewfinder shows: a subset of {"floor",
    "wall"}. Only these are described - a wall behind the camera, or a floor
    under a camera looking up at the sky, is not in the picture."""
    w, h = frame_size(scene)
    cam = Camera(scene["camera"], w, h)
    return {"wall" if p.part == "wall" else "floor"
            for p in room_polys(scene, cam) if in_frame(p.pts, w, h)}


def _join_parts(parts):
    return "\n".join(p if p.endswith((".", "!", "?")) else p + "." for p in parts)


def _scene_parts(scene):
    """(Words with only `.notes` set, [(character id or None, part text)]) -
    every line `scene_text` joins into one prompt, each tagged with the
    character it belongs to (None for room/details/camera/crowd/unassigned
    people) so `scene_text_regional` can split them without repeating this
    walk of `scene["objects"]`."""
    out = Words()
    parts = []
    details = scene["details"].strip()
    if details:
        parts.append((None, details.rstrip()))
    room = scene.get("room") or new_room()
    seen = room_seen(scene)
    for key, label, _ in SURFACES:
        words = room[key]["prompt"].strip()
        if not words or (key == "wall" and not room["walls"]):
            continue
        if key not in seen:
            out.notes.append("The %s %s outside the viewfinder, so the words leave %s out."
                             % ((label.lower(), "are", "them") if key == "wall"
                                else (label.lower(), "is", "it")))
            continue
        parts.append((None, "The %s: %s" % (label.lower(), words)))
    shape = floor_shape_words(room.get("grid")) if "floor" in seen else ""
    if shape:
        parts.append((None, shape))
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
            if "accordion" in (outfit(obj.get("look") or {}).get("held") or {}):
                gaps = accordion_contacts(obj["pose"]["controls"], obj.get("look"))
                missing = [name for side, name in (("r", "right"), ("l", "left"))
                           if gaps[side] > 0.035]
                if missing:
                    out.notes.append("%s: the %s hand%s %s not reach the accordion's "
                                     "playing surface; adjust the pose before generating." % (
                                         obj["name"], " and ".join(missing),
                                         "s" if len(missing) > 1 else "",
                                         "do" if len(missing) > 1 else "does"))
        line = "%s (%s)" % (obj["name"].strip() or ASSET[obj["asset"]]["label"],
                            ", ".join(about))
        desc = described(obj)
        look = look_text(obj) if obj["asset"] == "person" else ""
        said = ". ".join(x for x in (look, posture, desc) if x)
        char = obj.get("character") if obj["asset"] == "person" else None
        parts.append((char or None, line + (": " + said if said else "")))
        said = look or desc
        if not said:
            out.notes.append("%s has no description; the picture has only its shape and "
                             "name to go on." % obj["name"])
    for detail in (scene.get("enrich") or {}).get("added") or []:
        if detail.strip():
            parts.append((None, detail.strip()))
    parts.append((None, camera_words(scene)))
    return out, parts


def scene_text(scene):
    """The scene's words for the prompt. Every description is kept exactly
    as written; only where things are in the frame is added around it."""
    out, parts = _scene_parts(scene)
    out.text = _join_parts(p for _, p in parts)
    return out


def scene_text_regional(scene):
    """(base Words, {character id: text}) - like `scene_text`, but a person
    with a character assigned is pulled out of the shared text into their own
    entry (unchanged wording), so a regional prompt's base conditioning does
    not repeat what a masked region already says. A crowd or an unassigned
    person stays in the base text, same as `scene_text`."""
    out, parts = _scene_parts(scene)
    base, by_character = [], {}
    for char, text in parts:
        if char:
            by_character[char] = text if text.endswith((".", "!", "?")) else text + "."
        else:
            base.append(text)
    out.text = _join_parts(base)
    return out, by_character


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
    ('', ''): the person's own, else their character's first face photo,
    else their character's identity's first reference picture."""
    if obj.get("face") and os.path.isfile(obj["face"]):
        return obj["face"], "their own face picture"
    rec = (characters or {}).get(obj.get("character")) if obj.get("character") else None
    for path in (rec or {}).get("faces") or []:
        if os.path.isfile(path):
            return path, "%s's face photos" % rec["name"]
    ident = (identities or {}).get(rec.get("identity")) if rec and rec.get("identity") else None
    if ident and ident.get("use_references", True):
        for path in ident.get("references") or []:
            if os.path.isfile(path):
                return path, "%s's profile" % ident["name"]
    return "", ""


def face_photos(obj, characters=None, identities=None):
    """Every photo of a person's face there is, their own face picture
    first, then their character's face photos, then its identity's
    references: the real-face paste picks the one whose head is turned most
    like the drawn one's."""
    out = []
    if obj.get("face") and os.path.isfile(obj["face"]):
        out.append(obj["face"])
    rec = (characters or {}).get(obj.get("character")) if obj.get("character") else None
    out += [p for p in (rec or {}).get("faces") or [] if os.path.isfile(p) and p not in out]
    ident = (identities or {}).get(rec.get("identity")) if rec and rec.get("identity") else None
    if ident and ident.get("use_references", True):
        out += [p for p in ident.get("references") or [] if os.path.isfile(p) and p not in out]
    return out


FACE_SIDE = (0.08, 0.06, 0.0)  # the head's side at the ears, in its frame
NOSE_TIP = (0.0, 0.05, 0.1)


def head_angles(cam, centre, frame):
    """Camera-relative yaw, pitch, roll in degrees from the head frame.

    The viewing frame is centred on the head, including off-axis heads.
    Yaw is positive toward screen right, pitch positive looking up, roll
    positive counterclockwise on screen. At a vertical gaze Euler yaw/roll
    are coupled; choose roll zero. `facing_cosine` is 1 front-on, -1 away.
    """
    toward = norm(sub(cam.eye, centre))
    right = norm(sub(cam.r, mul(toward, dot(cam.r, toward))))
    up = cross(toward, right)
    forward = column(frame, 2)
    fx, fy, fz = dot(forward, right), dot(forward, up), dot(forward, toward)
    pitch = math.asin(max(-1.0, min(1.0, fy)))
    yaw = math.atan2(fx, fz)
    roll = (math.atan2(dot(column(frame, 0), up), dot(column(frame, 1), up))
            if math.hypot(fx, fz) > 1e-8 else 0.0)
    return {"yaw": round(math.degrees(yaw), 4),
            "pitch": round(math.degrees(pitch), 4),
            "roll": round(math.degrees(roll), 4), "facing_cosine": round(fz, 6)}


def _facing(a, b, nose):
    """Where the nose is between the face's two sides as the picture shows
    them, 0 at the left one, 1 at the right, 0.5 facing the camera; None
    when a point is behind it. DWPose's face points give the same measure
    of a drawn face (`studio_imagegen.drawn_facing`), so the two compare."""
    if not (a and b and nose):
        return None
    lo, hi = sorted((a[0], b[0]))
    if hi - lo < 1e-6:
        return 0.5
    return round(max(0.0, min(1.0, (nose[0] - lo) / (hi - lo))), 3)


def face_targets(scene, characters=None, identities=None):
    """Every person whose face is in the frame: {"id", "name", "at": [x, y]
    (the face's middle, as fractions of the frame), "region": [x0, y0, x1,
    y1] (their head and hair, the same way), "facing" (`_facing`), "head"
    (their head shape, {} for the mannequin's own), "tall" (the head, neck to
    crown, over the frame's height), "words" (what they look
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
        seen = lambda v: cam.project(add(mul(add(hp, apply(hm, head_point(v, obj.get("head")))), k), shift))  # noqa: E731
        facing = _facing(seen(FACE_SIDE), seen((-FACE_SIDE[0],) + FACE_SIDE[1:]),
                         seen(NOSE_TIP))
        out.append({"id": obj["id"], "name": obj["name"], "facing": facing,
                    "head_pose": head_angles(cam, add(mul(hp, k), shift), hm),
                    "identity": ((characters or {}).get(obj.get("character")) or {}).get("identity", ""),
                    "head": mq.clean_head(obj.get("head")), "tall": round(head / h, 4),
                    "at": [round(p[0] / w, 4), round(p[1] / h, 4)],
                    "region": [round(x, 4) for x in region],
                    "expression": str((obj.get("look") or {}).get("expression") or ""),
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
    and to their face picture's likeness where they have one.

    A person with a character is said as the character is now
    (`follow_characters`), not as the scene copied it."""
    import apps.image_studio.imagegen as ig
    if characters:
        scene = copy.deepcopy(scene)
        follow_characters(scene, characters)
    w, h = frame_size(scene)
    s, _ = clean_scene(scene)
    by_character = {}
    if s["regional_prompting"]:
        words, regional_text = scene_text_regional(scene)
        if len(regional_text) >= 2:            # one named character is not a bleed problem
            by_character = regional_text
        else:
            words = scene_text(scene)
    else:
        words = scene_text(scene)
    extra = {"width": w, "height": h, "scene_layout": copy.deepcopy(s),
             "references": dict(maps), "pose": None, "composition": None}
    shot, factor = framing(s)
    if factor < 1 and ("pose" in maps or "composition" in maps):
        words.notes.append("A %s: layout at %d%% of its strength, so the "
                           "mannequin is not drawn." % (shot, round(factor * 100)))
    if "pose" in maps:
        extra["pose"] = {"strength": round(s["pose_strength"], 3)}
    if "composition" in maps:
        extra["composition"] = {"strength": round(s["depth_strength"] * factor, 3)}
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
        extra["wearing"] = []
        extra["face_photos"], extra["face_name"] = [], ""
        idents = []
        for o in folks:
            rec = (characters or {}).get(o["character"]) if o["character"] else None
            if rec and rec.get("identity") and rec["identity"] not in idents:
                idents.append(rec["identity"])
        extra["scene_identities"] = idents
        extra["face_detail"] = True
        extra["scene_faces"] = {"likeness": round(s["face_likeness"], 3),
                                "head_depth": round(s["head_depth"], 3),
                                "real": s["real_faces"],
                                "people": face_targets(s, characters, identities)}
    if by_character:
        masks = character_masks(scene)
        extra["character_regions"] = [
            {"prompt": text, "mask_path": _write(masks[char], "character_mask")}
            for char, text in by_character.items() if char in masks]
    from apps.image_studio.scene import beard
    masks = beard_masks(s)
    # The face dots the pose map drew, so the beard pass can fit the mask to
    # the face as drawn (imagegen.shape_beard) and bound it by its landmarks.
    dots = {}
    for f in pose_figures(s) if masks else []:
        if f["face"] and f["id"] not in dots:
            dots[f["id"]] = [[round(x, 5), round(y, 5)] for x, y in f["face"]]
    for oid, mask in masks.items():
        person = next(o for o in folks if o["id"] == oid)
        extra.setdefault("character_regions", []).append({
            "prompt": beard.text(person["look"]["beard"]),
            "mask_path": _write(mask, "beard_mask"), "person_id": oid,
            "kind": "facial_hair", "beard": beard.clean(person["look"]["beard"]),
            "face_dots": dots.get(oid, [])})
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
                stand(scene.get("room"), obj, lift)
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


# A prop's details from the model (the Details box under a prop's
# description): what the user says it needs, or what would make it read as
# the real thing, each with how to block it out on the prop (`dressing_pieces`).
DRESS_SYSTEM = """You help stage a photograph. The user has picked one object in a \
scene blocked out in 3D and wants it dressed with the details that make it read as \
the real thing.

Rules:
- Details OF this object: what is on it, hung from it, fixed to it or painted on it, \
and its wear. Never a new object beside it, never people.
- If the user says what it needs, give exactly those, each made concrete. If not, \
suggest the one to three that matter most for this kind of object.
- Be concrete and visual: materials, colours, state. Each detail is 8 to 30 words, \
written as a line of an image prompt, not advice, and names the object.
- Nothing it already has.

Each detail is also blocked out on the object:
- "part": "ring" (a wreath, garland, hoop or band round it), "flag" (a flag or a \
board standing out from it), "banner" (cloth, ribbons or streamers hanging down it), \
"sign" (a plaque, board or picture flat on its side), "ball", "box", "cylinder", \
"cone" (a lantern, a knob, a crown, a finial), or "none" for what has no body of its \
own (paint, carving, stripes, rust, light).
- "at": "top", "upper", "middle", "lower", "base", or "along" (several spaced up its \
length, side to side).
- "count": how many pieces, 1 to %d.
- "size": [width, height, depth] of ONE piece in metres, true to life (a maypole \
wreath about [1.2, 0.25, 0.25], a flag [0.9, 0.6, 0.02]).
- "colour": its main colour as #rrggbb.

Answer with JSON only: {"details": [{"detail": "<the prompt line>", "part": "...", \
"at": "...", "count": 1, "size": [w, h, d], "colour": "#rrggbb"}]}""" % DRESS_COUNT


def dress_messages(scene, obj, ask=""):
    sx, sy, sz = obj["scale"]
    has = [d["text"] for d in obj.get("dressing") or []]
    lines = ["The object: %s, blocked out as a %s %.2g m wide, %.2g m tall, %.2g m deep."
             % (obj["name"].strip() or ASSET[obj["asset"]]["label"],
                ASSET[obj["asset"]]["label"].lower(), sx, sy, sz)]
    if obj["description"].strip():
        lines.append("Its description: %s" % obj["description"].strip())
    if has:
        lines.append("Details it already has:\n%s" % "\n".join("- " + x for x in has))
    if scene.get("details", "").strip():
        lines.append("The scene: %s" % scene["details"].strip())
    lines.append("The user asks for: %s" % ask.strip() if ask.strip() else
                 "The user has not said what it needs: suggest what it is missing.")
    return [{"role": "system", "content": DRESS_SYSTEM},
            {"role": "user", "content": "\n\n".join(lines)}]


def read_dressing(text):
    """-> [details] (`new_dressing`'s keys) from the model's reply: the JSON
    asked for, a bare list, or one detail; failing that the first real
    line of prose, as words only. [] when there is nothing."""
    text = re.sub(r"(?s)<think>.*?(</think>|$)", "", text or "").strip()
    m = re.search(r"(?s)[\[{].*[\]}]", text)
    d = None
    if m:
        try:
            d = json.loads(m.group(0))
        except ValueError:
            d = None
    if isinstance(d, dict):
        d = d.get("details") if isinstance(d.get("details"), list) else [d]
    if isinstance(d, list):
        got = [dict(x, text=_tidy(x["detail"])) for x in d
               if isinstance(x, dict) and isinstance(x.get("detail"), str)]
        for x in got:              # said after a full stop in the prop's line
            x["text"] = x["text"][:1].upper() + x["text"][1:]
        got = clean_dressing(got)
        if got:
            return got
    for line in text.splitlines():
        line = _tidy(line)
        if len(line) > 12 and not line.startswith(("{", "}", "[", "]", "```")):
            return [new_dressing(line[:DRESS_CHARS])]
    return []


def suggest_dressing(scene, obj, llm, ask=""):
    """-> the details `llm` offers for `obj`, none it already has. Raises
    RuntimeError when nothing usable came back."""
    reply = llm.chat(dress_messages(scene, obj, ask), max_tokens=900)
    msg = ((reply.get("choices") or [{}])[0].get("message") or {})
    has = [d["text"] for d in obj.get("dressing") or []]
    got = [d for d in read_dressing(msg.get("content") or "")
           if not any(_same(d["text"], x) for x in has)]
    if not got:
        raise RuntimeError("The model offered no new details for %s. Say what it needs "
                           "and ask again." % (obj["name"].strip() or "it"))
    return got


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
    'words", "indoors": true or false, "walls": "indoors, what the walls are, in a few '
    'words (material, colour, what hangs on them); outdoors, empty", "people": [{"box": [left, top, right, bottom], "name": "a short name for '
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
    out = {"setting": "", "floor": "", "walls": "", "indoors": False, "people": []}
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
    if isinstance(data.get("walls"), str):
        out["walls"] = _said_word("walls", data["walls"])
    indoors = data.get("indoors")
    if isinstance(indoors, str):
        indoors = indoors.strip().lower() in ("true", "yes", "indoors")
    # Walls said outdoors are a building's side, not a room round the camera.
    out["indoors"] = indoors is True
    if not out["indoors"]:
        out["walls"] = ""
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


# Room made round a photo taken indoors: this much (m) beyond the camera and
# the people, so the camera stands inside it and no one is in a wall.
PICTURE_ROOM_MARGIN = 1.5


def picture_room(room, placed, dist, eye_y, walls=""):
    """Four walls round an indoor photo's camera and people, and the walls'
    words. The room is centred on the origin, so each half-size is the
    farthest thing from it that side plus `PICTURE_ROOM_MARGIN`."""
    m = PICTURE_ROOM_MARGIN
    xs = [abs(rel[0]) for _, _, rel, _, _ in placed]
    zs = [abs(dist + rel[2]) for _, _, rel, _, _ in placed] + [dist]
    lo, hi = ROOM_LIMITS["width"]
    room["width"] = round(min(hi, max(lo, 2 * (max(xs) + m))), 2)
    lo, hi = ROOM_LIMITS["depth"]
    room["depth"] = round(min(hi, max(lo, 2 * (max(zs) + m))), 2)
    lo, hi = ROOM_LIMITS["height"]
    room["height"] = round(min(hi, max(lo, 3.0, eye_y + 0.8)), 2)
    room["walls"] = True
    if walls:
        room["wall"]["prompt"] = walls


def picture_scene(data, read=None, lens=PICTURE_LENS):
    """The pose finder's JSON for a photo (and `picture_answer` of it, or
    None) -> (scene, notes). Raises ValueError when no one in it can be
    posed: a scene of nobody is not what was asked for."""
    pw, ph = float(data.get("width") or 0), float(data.get("height") or 0)
    folk = photo_people(data)
    if not folk or pw <= 0 or ph <= 0:
        raise ValueError("The pose finder found no one in the picture.")
    read = dict({"setting": "", "floor": "", "walls": "", "indoors": False,
                 "people": []}, **(read or {}))
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
    if read["indoors"]:
        picture_room(scene["room"], placed, dist, eye_y, read["walls"])
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
