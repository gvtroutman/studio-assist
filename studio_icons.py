#!/usr/bin/env python3
"""
Read an application's own icon out of its .exe, and write PNGs.

Windows keeps icons in the PE resource directory: an RT_GROUP_ICON lists the
sizes on offer, each pointing at an RT_ICON image that is either a PNG (Vista
and later, usually the 256px one) or a bottom-up DIB with a 1-bit AND mask
stapled underneath. This walks that structure with `struct` and `zlib` and
nothing else - the stdlib-only rule in AGENTS.md applies here as much as
anywhere, and a folder of extracted .pngs would rot the moment Adobe ships a
new year.

The two-letter badges stay as the fallback: an app whose icon cannot be read -
no resources, a packed exe, a path we cannot open - draws its mark instead.

    python studio_icons.py "C:\\Program Files\\...\\AfterFX.exe" out.png 64
"""

import os
import struct
import sys
import zlib

RT_ICON = 3
RT_GROUP_ICON = 14

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


# --------------------------------------------------------------------- PNG out

def png(pixels, width, height):
    """RGBA bytes -> a PNG file. make_icon.py draws the app's own mark with it."""
    stride = width * 4
    raw = b"".join(b"\x00" + pixels[y * stride:(y + 1) * stride]
                   for y in range(height))

    def chunk(tag, data):
        body = tag + data
        return (struct.pack(">I", len(data)) + body
                + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF))

    return (PNG_MAGIC
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9))
            + chunk(b"IEND", b""))


# ---------------------------------------------------------------- PE structure

class _PE:
    """Just enough of a PE file to reach its resources."""

    def __init__(self, f):
        self.f = f
        head = self._at(0, 0x40)
        if head[:2] != b"MZ":
            raise ValueError("not an executable")
        pe_off = struct.unpack_from("<I", head, 0x3C)[0]
        pe = self._at(pe_off, 24)
        if pe[:4] != b"PE\0\0":
            raise ValueError("not a PE image")
        n_sections = struct.unpack_from("<H", pe, 6)[0]
        opt_size = struct.unpack_from("<H", pe, 20)[0]
        opt_off = pe_off + 24
        magic = struct.unpack_from("<H", self._at(opt_off, 2), 0)[0]
        # PE32+ widens four optional-header fields, which pushes the data
        # directories 16 bytes further along. Nothing else here differs.
        dirs = opt_off + (112 if magic == 0x20B else 96)
        self.res_rva = struct.unpack("<II", self._at(dirs + 16, 8))[0]
        blob = self._at(opt_off + opt_size, 40 * n_sections)
        self.sections = []
        for i in range(n_sections):
            vsize, va, rawsize, raw = struct.unpack_from("<IIII", blob, 40 * i + 8)
            self.sections.append((va, max(vsize, rawsize), raw))
        if not self.res_rva:
            raise ValueError("no resource directory")

    def _at(self, off, n):
        self.f.seek(off)
        return self.f.read(n)

    def read(self, rva, n):
        for va, size, raw in self.sections:
            if va <= rva < va + size:
                return self._at(raw + (rva - va), n)
        raise ValueError("rva %#x is outside every section" % rva)

    # Offsets inside the resource tree are relative to the tree's own base -
    # except the leaf IMAGE_RESOURCE_DATA_ENTRY, whose OffsetToData is a plain
    # RVA. Mixing those two up is the classic way to read garbage here.
    def entries(self, offset):
        head = self.read(self.res_rva + offset, 16)
        n = (struct.unpack_from("<H", head, 12)[0]
             + struct.unpack_from("<H", head, 14)[0])
        blob = self.read(self.res_rva + offset + 16, 8 * n)
        return [struct.unpack_from("<II", blob, 8 * i) for i in range(n)]

    def by_id(self, offset, want):
        for name, child in self.entries(offset):
            if not name & 0x80000000 and name == want:
                return child
        return None

    def leaf(self, child):
        """Descend whatever subdirectories remain (language) to the bytes."""
        while child & 0x80000000:
            child = self.entries(child & 0x7FFFFFFF)[0][1]
        rva, size = struct.unpack("<II", self.read(self.res_rva + child, 8))
        return self.read(rva, size)


# -------------------------------------------------------------------- decoding

def dib_to_rgba(data):
    """A bottom-up DIB plus its AND mask -> (rgba, width, height)."""
    hsize, width, height2 = struct.unpack_from("<Iii", data, 0)
    bits, compression = struct.unpack_from("<HI", data, 14)
    used = struct.unpack_from("<I", data, 32)[0]
    if compression not in (0, 3):
        raise ValueError("compressed DIB")
    height = height2 // 2 or height2      # biHeight counts XOR + AND together

    palette = []
    pix = hsize
    if bits <= 8:
        count = used or (1 << bits)
        for i in range(count):
            b, g, r = data[hsize + 4 * i], data[hsize + 4 * i + 1], data[hsize + 4 * i + 2]
            palette.append((r, g, b))
        pix = hsize + 4 * count

    row_bytes = ((width * bits + 31) // 32) * 4
    mask_bytes = ((width + 31) // 32) * 4
    mask_at = pix + row_bytes * height
    have_mask = len(data) >= mask_at + mask_bytes * height

    out = bytearray(width * height * 4)
    opaque = False
    for y in range(height):
        src = pix + row_bytes * (height - 1 - y)     # rows run bottom to top
        msk = mask_at + mask_bytes * (height - 1 - y)
        dst = y * width * 4
        for x in range(width):
            if bits == 32:
                i = src + 4 * x
                b, g, r, a = data[i], data[i + 1], data[i + 2], data[i + 3]
            elif bits == 24:
                i = src + 3 * x
                b, g, r, a = data[i], data[i + 1], data[i + 2], 255
            elif bits == 8:
                r, g, b = palette[data[src + x]]
                a = 255
            elif bits == 4:
                byte = data[src + (x >> 1)]
                r, g, b = palette[byte >> 4 if not x & 1 else byte & 0xF]
                a = 255
            elif bits == 1:
                byte = data[src + (x >> 3)]
                r, g, b = palette[(byte >> (7 - (x & 7))) & 1]
                a = 255
            else:
                raise ValueError("%d-bit icon" % bits)
            if a:
                opaque = True
            if have_mask and (data[msk + (x >> 3)] >> (7 - (x & 7))) & 1:
                a = 0
            i = dst + 4 * x
            out[i], out[i + 1], out[i + 2], out[i + 3] = r, g, b, a

    if not opaque:
        # A 32-bit icon whose alpha channel is all zero is opaque as Windows
        # reads it - the AND mask is what cuts the shape out.
        for y in range(height):
            msk = mask_at + mask_bytes * (height - 1 - y)
            for x in range(width):
                bit = (data[msk + (x >> 3)] >> (7 - (x & 7))) & 1 if have_mask else 0
                out[(y * width + x) * 4 + 3] = 0 if bit else 255
    return bytes(out), width, height


def png_to_rgba(data):
    """The big entries are usually PNGs. 8-bit, non-interlaced, as icons are."""
    if data[:8] != PNG_MAGIC:
        raise ValueError("not a PNG")
    pos, idat, palette, trns = 8, [], None, None
    width = height = color = 0
    while pos + 8 <= len(data):
        size = struct.unpack_from(">I", data, pos)[0]
        tag = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + size]
        if tag == b"IHDR":
            width, height, depth, color, _c, _f, interlace = struct.unpack(
                ">IIBBBBB", body)
            if depth != 8 or interlace:
                raise ValueError("unsupported PNG: %d-bit%s"
                                 % (depth, ", interlaced" if interlace else ""))
        elif tag == b"PLTE":
            palette = body
        elif tag == b"tRNS":
            trns = body
        elif tag == b"IDAT":
            idat.append(body)
        elif tag == b"IEND":
            break
        pos += size + 12

    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color]
    raw = zlib.decompress(b"".join(idat))
    stride = width * channels
    lines = bytearray(stride * height)
    prev = bytearray(stride)
    at = 0
    for y in range(height):
        filt = raw[at]
        line = bytearray(raw[at + 1:at + 1 + stride])
        at += 1 + stride
        if filt:
            for i in range(stride):
                a = line[i - channels] if i >= channels else 0
                b = prev[i]
                c = prev[i - channels] if i >= channels else 0
                if filt == 1:
                    line[i] = (line[i] + a) & 0xFF
                elif filt == 2:
                    line[i] = (line[i] + b) & 0xFF
                elif filt == 3:
                    line[i] = (line[i] + ((a + b) >> 1)) & 0xFF
                else:
                    p = a + b - c
                    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                    near = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
                    line[i] = (line[i] + near) & 0xFF
        lines[y * stride:(y + 1) * stride] = line
        prev = line

    out = bytearray(width * height * 4)
    for i in range(width * height):
        s = i * channels
        if color == 6:
            out[i * 4:i * 4 + 4] = lines[s:s + 4]
            continue
        if color == 2:
            r, g, b, a = lines[s], lines[s + 1], lines[s + 2], 255
        elif color == 3:
            idx = lines[s]
            r, g, b = palette[3 * idx], palette[3 * idx + 1], palette[3 * idx + 2]
            a = trns[idx] if trns and idx < len(trns) else 255
        elif color == 4:
            r = g = b = lines[s]
            a = lines[s + 1]
        else:
            r = g = b = lines[s]
            a = 255
        out[i * 4], out[i * 4 + 1], out[i * 4 + 2], out[i * 4 + 3] = r, g, b, a
    return bytes(out), width, height


def resample(pixels, width, height, size):
    """
    Area-average down to size x size. Colours are weighted by alpha, so a
    half-covered edge pixel keeps the icon's colour instead of bleeding the
    transparent black underneath it.
    """
    if (width, height) == (size, size):
        return pixels
    out = bytearray(size * size * 4)
    for y in range(size):
        y0 = y * height // size
        y1 = max(y0 + 1, (y + 1) * height // size)
        for x in range(size):
            x0 = x * width // size
            x1 = max(x0 + 1, (x + 1) * width // size)
            r = g = b = a = n = 0
            for sy in range(y0, y1):
                row = sy * width * 4
                for sx in range(x0, x1):
                    i = row + sx * 4
                    al = pixels[i + 3]
                    r += pixels[i] * al
                    g += pixels[i + 1] * al
                    b += pixels[i + 2] * al
                    a += al
                    n += 1
            i = (y * size + x) * 4
            if a:
                out[i], out[i + 1], out[i + 2] = r // a, g // a, b // a
                out[i + 3] = a // n
    return bytes(out)


# ------------------------------------------------------------------ the way in

def best_entry(group, size):
    """
    Which image in the group to scale from: the smallest one at or above the
    target, so we shrink rather than blow up, and never the 256px monster when
    a 48px DIB will do - inflating and unfiltering that PNG costs ten times as
    much for a 26px badge.
    """
    count = struct.unpack_from("<H", group, 4)[0]
    entries = []
    for i in range(count):
        w, h, _colors, _res, _planes, bits, _bytes, ident = struct.unpack_from(
            "<BBBBHHIH", group, 6 + 14 * i)
        entries.append((w or 256, h or 256, bits, ident))
    fits = [e for e in entries if e[0] >= size] or entries
    return min(fits, key=lambda e: (e[0] >= 128, e[0]))


_CACHE = {}


def icon_png(exe, size=26):
    """
    PNG bytes for `exe`'s own icon at size x size, or None if it has none we
    can read. Never raises: a missing icon is a fallback, not a failure.
    """
    if not exe:
        return None
    try:
        key = (os.path.abspath(exe), os.path.getmtime(exe), size)
    except OSError:
        return None
    if key in _CACHE:
        return _CACHE[key]
    out = None
    try:
        with open(exe, "rb") as f:
            pe = _PE(f)
            groups = pe.by_id(0, RT_GROUP_ICON)
            icons = pe.by_id(0, RT_ICON)
            if groups is not None and icons is not None:
                groups &= 0x7FFFFFFF
                icons &= 0x7FFFFFFF
                # Windows shows the lowest-numbered group as the app's icon.
                first = min(n for n, _child in pe.entries(groups)
                            if not n & 0x80000000)
                group = pe.leaf(pe.by_id(groups, first))
                ident = best_entry(group, size)[3]
                data = pe.leaf(pe.by_id(icons, ident))
                if data[:8] == PNG_MAGIC:
                    rgba, w, h = png_to_rgba(data)
                else:
                    rgba, w, h = dib_to_rgba(data)
                out = png(resample(rgba, w, h, size), size, size)
    except Exception:
        out = None
    _CACHE[key] = out
    return out


def main(argv):
    if len(argv) < 2:
        return "usage: studio_icons.py <exe> [out.png] [size]"
    data = icon_png(argv[1], int(argv[3]) if len(argv) > 3 else 64)
    if not data:
        return "no icon could be read from %s" % argv[1]
    out = argv[2] if len(argv) > 2 else "icon.png"
    with open(out, "wb") as f:
        f.write(data)
    print("wrote %s, %d bytes" % (out, len(data)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
