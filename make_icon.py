#!/usr/bin/env python3
"""
Draw studio-assistant.ico, the shortcut and taskbar icon.

The app draws its app marks rather than shipping art (see AGENTS.md), and the
same rule applies here: this generates the .ico so the mark can be changed by
editing colours rather than by opening a paint program. Run it after a change:

    python make_icon.py

Stdlib only - struct for the ICO container, and studio_icons for the PNGs
inside it (the same writer the sidebar's app marks come back through).
"""

import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from studio_icons import png

SQUARE = (0xD9, 0x77, 0x57)   # ACCENT, the same orange as the Send button
MARK = (0x16, 0x15, 0x0F)     # the button's foreground, near-black
SIZES = (16, 24, 32, 48, 64, 128, 256)
SS = 4                        # supersample factor, then box-downsample


def rounded_hit(x, y, size, radius):
    """Point-in-rounded-square, in supersampled pixel coordinates."""
    r = radius
    cx = min(max(x, r), size - r)
    cy = min(max(y, r), size - r)
    if cx == x and cy == y:
        return True
    return (x - cx) ** 2 + (y - cy) ** 2 <= r * r


def triangle_hit(x, y, pts):
    """Point-in-triangle by consistent edge sign."""
    sign = None
    for i in range(3):
        ax, ay = pts[i]
        bx, by = pts[(i + 1) % 3]
        cross = (bx - ax) * (y - ay) - (by - ay) * (x - ax)
        if cross == 0:
            continue
        s = cross > 0
        if sign is None:
            sign = s
        elif s != sign:
            return False
    return True


def render(size):
    """-> RGBA bytes, anti-aliased by rendering big and averaging down."""
    big = size * SS
    radius = big * 0.22
    pts = [(big * 0.37, big * 0.27), (big * 0.37, big * 0.73), (big * 0.75, big * 0.50)]

    # 1 = square, 2 = mark, 0 = transparent
    cells = bytearray(big * big)
    for y in range(big):
        row = y * big
        for x in range(big):
            if not rounded_hit(x + 0.5, y + 0.5, big, radius):
                continue
            cells[row + x] = 2 if triangle_hit(x + 0.5, y + 0.5, pts) else 1

    out = bytearray(size * size * 4)
    area = SS * SS
    for y in range(size):
        for x in range(size):
            r = g = b = a = 0
            for dy in range(SS):
                base = (y * SS + dy) * big + x * SS
                for dx in range(SS):
                    cell = cells[base + dx]
                    if cell == 0:
                        continue
                    src = SQUARE if cell == 1 else MARK
                    r += src[0]
                    g += src[1]
                    b += src[2]
                    a += 255
            i = (y * size + x) * 4
            if a:
                # average over covered samples only, so edge pixels keep their
                # colour and fade in alpha instead of darkening toward black
                covered = a // 255
                out[i] = r // covered
                out[i + 1] = g // covered
                out[i + 2] = b // covered
                out[i + 3] = a // area
    return bytes(out)


def ico(blobs):
    """blobs: [(size, png bytes)]. PNG-in-ICO is fine on Vista and later."""
    head = struct.pack("<HHH", 0, 1, len(blobs))
    offset = 6 + 16 * len(blobs)
    entries = b""
    for size, data in blobs:
        entries += struct.pack("<BBBBHHII", size % 256, size % 256, 0, 0, 1, 32,
                               len(data), offset)
        offset += len(data)
    return head + entries + b"".join(d for _, d in blobs)


def main():
    blobs = []
    for size in SIZES:
        blobs.append((size, png(render(size), size, size)))
        print("  %dx%d" % (size, size), flush=True)
    with open("studio-assistant.ico", "wb") as f:
        f.write(ico(blobs))
    with open("studio-assistant-preview.png", "wb") as f:
        f.write(blobs[-1][1])  # 256px, for looking at
    print("wrote studio-assistant.ico")
    return 0


if __name__ == "__main__":
    sys.exit(main())
