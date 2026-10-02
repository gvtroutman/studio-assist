"""Discrete signed Euclidean distances for conditioning masks; stdlib only."""

import array
import math


def _edt_line(values):
    """Squared Euclidean distance transform: lower envelope of parabolas."""
    sites, edges = [0], [-math.inf, math.inf]
    for q in range(1, len(values)):
        while True:
            p = sites[-1]
            crossing = ((values[q] + q * q) - (values[p] + p * p)) / (2 * (q - p))
            if crossing > edges[-2]:
                break
            sites.pop()
            edges.pop()
        sites.append(q)
        edges[-1] = crossing
        edges.append(math.inf)
    out, k = [], 0
    for q in range(len(values)):
        while edges[k + 1] < q:
            k += 1
        p = sites[k]
        out.append((q - p) ** 2 + values[p])
    return out


def _squared_distance(binary, width, height, target):
    cap = width * width + height * height
    work = array.array("d", [0.0]) * len(binary)
    for y in range(height):
        start = y * width
        line = _edt_line([0 if binary[start + x] == target else cap for x in range(width)])
        work[start:start + width] = array.array("d", line)
    for x in range(width):
        line = _edt_line([work[y * width + x] for y in range(height)])
        for y, d in enumerate(line):
            work[y * width + x] = d
    return work


def signed_distance(mask, width, height):
    """Float pixels: positive inside, negative outside, zero at the edge.

    Exact Euclidean distance to the nearest opposite-class pixel centre,
    minus half a pixel to place the raster boundary between centres. This
    is a discrete boundary approximation, not distance to the 3D surface.
    No boundary beyond the image is invented: uniform masks saturate at
    the image diagonal. Runtime and storage are linear in pixel count.
    """
    if width <= 0 or height <= 0 or len(mask) != width * height:
        raise ValueError("Mask dimensions must match its pixels and be positive")
    binary = bytes(bool(v) for v in mask)
    if not any(binary) or all(binary):
        value = math.hypot(width, height) * (1 if binary[0] else -1)
        return array.array("f", [value]) * len(binary)
    inside = _squared_distance(binary, width, height, 0)
    outside = _squared_distance(binary, width, height, 1)
    return array.array("f", ((math.sqrt(inside[i]) - 0.5) if v else
                             -(math.sqrt(outside[i]) - 0.5)
                             for i, v in enumerate(binary)))


def feather_distance(field, radius):
    """Smoothstep across [-radius, radius], returning an 8-bit alpha mask."""
    if not math.isfinite(radius) or radius < 0:
        raise ValueError("Feather radius must be finite and nonnegative")
    if radius == 0:
        return bytearray(255 if d > 0 else 0 for d in field)
    out = bytearray(len(field))
    for i, d in enumerate(field):
        t = max(0.0, min(1.0, 0.5 + d / (2 * radius)))
        out[i] = round(255 * t * t * (3 - 2 * t))
    return out
