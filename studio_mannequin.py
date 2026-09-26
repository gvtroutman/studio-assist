#!/usr/bin/env python3
"""
studio_mannequin - the shapes the Scene Builder's person is sculpted from:
an artist's wooden mannequin, a part per bone.

A part is lofted along its bone (`studio_scene.loft`) through rings, and a
ring here is not an ellipse but a closed curve: a function of the angle
round the bone giving how far out the surface is, 1 on the ellipse. Up a
bone (the torso) the angle's positive half is the back; down one (a limb)
it is the front. Stdlib only, no tkinter, no geometry of its own.
"""

import math

FRONT_UP = -math.pi / 2        # the front, round a bone that runs up
BACK_UP = math.pi / 2


def _lobe(th, at, width):
    """A smooth bump round `at` (rad), `width` wide, 1 at its top."""
    d = math.atan2(math.sin(th - at), math.cos(th - at))
    return math.exp(-(d / width) ** 2)


def superellipse(p):
    """A section squarer than an ellipse as `p` goes past 2."""
    def k(th):
        c, s = abs(math.cos(th)), abs(math.sin(th))
        return (c ** p + s ** p) ** (-1.0 / p)
    return k


def chest(pecs):
    """The chest block: square-shouldered, the pectorals two lobes out in
    front with the breastbone a groove between them, the shoulder blades
    flat behind."""
    sq = superellipse(2.6)

    def k(th):
        out = 1 + pecs * (_lobe(th, FRONT_UP + 0.62, 0.42) + _lobe(th, FRONT_UP - 0.62, 0.42))
        out -= 0.5 * pecs * _lobe(th, FRONT_UP, 0.16)
        out -= 0.04 * _lobe(th, BACK_UP, 0.5)
        return sq(th) * out
    return k


def abdomen(abs_):
    """The waist block: a soft-cornered section, a little fuller in front."""
    sq = superellipse(2.3)
    return lambda th: sq(th) * (1 + abs_ * _lobe(th, FRONT_UP, 0.9))


def pelvis(seat):
    """The pelvis block: the seat two lobes behind, the front flat."""
    sq = superellipse(2.4)

    def k(th):
        out = 1 + seat * (_lobe(th, BACK_UP + 0.6, 0.45) + _lobe(th, BACK_UP - 0.6, 0.45))
        out -= 0.05 * _lobe(th, FRONT_UP, 0.6)
        return sq(th) * out
    return k


LIMB = superellipse(2.2)       # round, a touch firmer than a tube
MITTEN = superellipse(3.0)     # the hand: flat, fingers as one


# ===================================================================== head
# A person's head shape: (key, label, what 0 and 1 do). Each is 0..1 with
# 0.5 the mannequin's own head, so a head with no shape set is unchanged
# and a saved shape keeps only what is moved. They are the proportions
# that change the head's outline and its depth, not a face's features:
# the mannequin has none, and what the depth map cannot see the picture
# is not told. The same keys are an identity's `head`, so a person carries
# their head from scene to scene.
HEAD_SHAPE = [
    ("head_width", "Head width", "narrow", "wide"),
    ("face_length", "Face length", "short", "long"),
    ("cheek_width", "Cheekbones", "narrow", "wide"),
    ("jaw_width", "Jaw width", "narrow", "wide"),
    ("chin_width", "Chin width", "pointed", "broad"),
    ("chin_length", "Chin length", "short", "long"),
    ("skull_depth", "Back of head", "flat", "deep"),
    ("neck_width", "Neck width", "slender", "thick"),
]
HEAD_KEYS = [k for k, *_ in HEAD_SHAPE]


def clean_head(d):
    """A saved head shape made safe: known keys, each 0..1 to 0.01, and
    only those off the middle."""
    out = {}
    for k in HEAD_KEYS:
        try:
            x = float((d or {}).get(k, 0.5)) if isinstance(d, dict) else 0.5
        except (TypeError, ValueError):
            continue
        if x != x:
            continue
        x = round(max(0.0, min(1.0, x)), 2)
        if x != 0.5:
            out[k] = x
    return out


def _ramp(x, a, b):
    """0 at `a`, 1 at `b`, smooth between, flat beyond."""
    t = max(0.0, min(1.0, (x - a) / (b - a)))
    return t * t * (3 - 2 * t)


def head_warp(head):
    """The head's shape as a warp of a point on the skull, in the
    ellipsoid's own frame (y up from its middle, radius 0.115; +z the
    face), or None when the head is the mannequin's own."""
    head = clean_head(head)
    if not head:
        return None
    g = lambda k: head.get(k, 0.5) - 0.5          # noqa: E731  -0.5..0.5

    def warp(p):
        x, y, z = p
        v = y / 0.115                               # -1 the chin, 1 the crown
        front = _ramp(z, -0.02, 0.05)               # the face, not the back
        down = _ramp(-v, 0.25, 0.9)                 # the jaw, below the cheeks
        tip = _ramp(-v, 0.65, 1.0) * front          # the chin
        # Across: the whole head, a band at the cheekbones, the jaw, the chin.
        kx = 1 + 0.3 * g("head_width")
        kx *= 1 + 0.35 * g("cheek_width") * math.exp(-((v + 0.15) / 0.3) ** 2)
        kx *= 1 + 0.7 * g("jaw_width") * down
        kx *= 1 + 0.9 * g("chin_width") * tip
        x *= kx
        # Down: the face below the eyes longer, then the chin further.
        if v < 0:
            y *= 1 + 0.35 * g("face_length") + 0.25 * g("chin_length") * tip
            z += 0.03 * g("chin_length") * tip      # a long chin juts, too
        # Back: the skull behind the ears.
        if z < 0:
            z *= 1 + 0.4 * g("skull_depth") * _ramp(v, -0.6, -0.1)
        return (x, y, z)
    return warp


def neck_scale(head):
    """How much wider than the mannequin's own the neck is."""
    return 1 + 0.5 * (clean_head(head).get("neck_width", 0.5) - 0.5)
