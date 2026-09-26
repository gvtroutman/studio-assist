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
