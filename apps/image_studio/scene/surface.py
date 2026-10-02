"""Numerical signals from visible world-space surfaces; stdlib only.

Normals remain floats. Grazing is not a silhouette, and normal variation
is not curvature: on flat-shaded meshes it mostly measures polygon creases.
Missing/room surfaces have valid=0, rather than an invented orientation.
"""

import array
import math


def analyse(instance, part, normals, world, width, height, eye, forward,
            light, grazing_width=0.2):
    """Return float maps, with undirected four-neighbour edges on both sides.

    facing is signed incidence; grazing/normal_edge are in [0,1].
    normal_edge is half the largest neighbouring normal-vector difference.
    depth_edge is relative camera-depth change; world_edge is distance in
    scene units. Ownership and part boundaries are separate byte maps.
    Geometric edges compare valid surfaces only, never zero-filled background.
    """
    n = width * height
    if width <= 0 or height <= 0 or len(instance) != n or len(part) != n or \
            len(normals) != 3 * n or len(world) != 3 * n:
        raise ValueError("Surface dimensions must match the buffers")
    if not math.isfinite(grazing_width) or grazing_width <= 0:
        raise ValueError("Grazing width must be finite and positive")

    def unit(v):
        length = math.sqrt(sum(x * x for x in v))
        return tuple(x / length for x in v) if length else (0.0, 0.0, 0.0)

    forward, light = unit(forward), unit(light)
    maps = {key: array.array("f", [0.0]) * n for key in
            ("facing", "grazing", "normal_edge", "depth_edge", "world_edge", "lighting")}
    maps.update({key: bytearray(n) for key in ("valid", "instance_edge", "part_edge")})
    directions = array.array("f", [0.0]) * (3 * n)
    depths = array.array("f", [0.0]) * n
    for i in range(n):
        if not instance[i]:
            continue
        o = i * 3
        normal = unit(normals[o:o + 3])
        toward = tuple(eye[k] - world[o + k] for k in range(3))
        if not any(normal) or not any(toward):
            continue
        view = unit(toward)
        facing = max(-1.0, min(1.0, sum(normal[k] * view[k] for k in range(3))))
        maps["valid"][i] = 1
        directions[o:o + 3] = array.array("f", normal)
        depths[i] = -sum(toward[k] * forward[k] for k in range(3))
        maps["facing"][i] = facing
        maps["grazing"][i] = max(0.0, 1.0 - abs(facing) / grazing_width)
        maps["lighting"][i] = max(0.0, sum(normal[k] * light[k] for k in range(3)))

    def edge(i, j):
        if instance[i] != instance[j]:
            maps["instance_edge"][i] = maps["instance_edge"][j] = 1
        elif instance[i] and part[i] != part[j]:
            maps["part_edge"][i] = maps["part_edge"][j] = 1
        if not (maps["valid"][i] and maps["valid"][j]):
            return
        a, b = i * 3, j * 3
        values = {
            "normal_edge": min(1.0, math.sqrt(sum(
                (directions[a + k] - directions[b + k]) ** 2 for k in range(3))) / 2),
            "depth_edge": abs(depths[i] - depths[j]) / max(abs(depths[i]), abs(depths[j]), 1e-8),
            "world_edge": math.sqrt(sum((world[a + k] - world[b + k]) ** 2 for k in range(3)))}
        for key, value in values.items():
            maps[key][i] = max(maps[key][i], value)
            maps[key][j] = max(maps[key][j], value)

    for y in range(height):
        for x in range(width):
            i = y * width + x
            if x + 1 < width:
                edge(i, i + 1)
            if y + 1 < height:
                edge(i, i + width)
    return maps


def identity_mask(instance, part, signals, width, height, iid, head_parts,
                  region, feather=1.5):
    """Visible head alpha, confidence-weighted and feathered only inward.

    Normal creases never cut holes. Only the ownership/part/region boundary
    limits the distance feather; grazing controls confidence and feather width.
    Returns bytes, never a rectangle fallback for an occluded head.
    """
    from .distance import signed_distance
    if not math.isfinite(feather) or feather < 0:
        raise ValueError("Feather must be finite and nonnegative")
    hard = bytearray(width * height)
    x0, y0, x1, y1 = region
    for i, owner in enumerate(instance):
        x, y = (i % width + 0.5) / width, (i // width + 0.5) / height
        if owner == iid and part[i] in head_parts and signals["valid"][i] and \
                x0 <= x < x1 and y0 <= y < y1:
            hard[i] = 255
    if not any(hard):
        return bytes(hard)
    distances = signed_distance(hard, width, height)
    out = bytearray(len(hard))
    for i, v in enumerate(hard):
        if not v:
            continue
        facing = max(0.0, signals["facing"][i])
        t = min(1.0, facing / 0.2)
        confidence = t * t * (3 - 2 * t)
        radius = feather * (1 + signals["grazing"][i])
        t = min(1.0, max(0.0, distances[i] / radius)) if radius else 1.0
        out[i] = round(255 * confidence * t * t * (3 - 2 * t))
    return bytes(out)
