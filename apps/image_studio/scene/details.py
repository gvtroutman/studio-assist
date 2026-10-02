"""Local flag-pattern and lettering repairs after a scene is generated.

Klein edits enlarged crops; the original SAM3 silhouette and selected box
confine the blend. A design swatch helps it draw regular blue-and-white lozenges rather
than asking the whole-picture model to draw them at background resolution.
Stdlib only; this does not render arbitrary text into a photograph.
"""
import math
import re

from apps.image_studio import wear
from core import icons

STATUS = "scene_details"
LABEL = "Flags and lettering"
NODES = wear.NODES
LICENSE_NOTE = wear.LICENSE_NOTE.replace("(the item pass)", "(the scene details pass)")
MAX_TARGETS = 8
MIN_SIDE = 24
FLAGS = re.compile(r"\b(?:flags?|bunting|pennants?)\b", re.I)
SIGNS = re.compile(r"\b(?:signs?|signage|banners?|lettering|inscriptions?)\b", re.I)
BAVARIAN = re.compile(r"\b(?:oktoberfest|bavarian|bavaria|bayern)\b", re.I)
OTHER_FLAG = re.compile(r"\b(?:german|american|british|french|italian|european|rainbow)\s+"
                        r"(?:flags?|bunting|pennants?)\b", re.I)
# The flag pattern redraw is off (2026-10-02, the user: "it's unnecessary"):
# flags are left as the picture drew them; lettering is still repaired.
FLAG_PATTERN = False
QUOTES = re.compile(r'["“]([^"”\n]{1,64})["”]|[\'‘]([^\'’\n]{1,64})[\'’]')


def requests(settings):
    """Repair only details actually described, using quoted lettering first.

    Saved scene descriptions and prop details are read independently, so
    a quotation in someone's description does not become a sign's wording.
    Without explicit wording, Oktoberfest signs receive a short fitting title.
    """
    if settings.get("scene_details_pass", True) is False or settings.get("mode"):
        return []
    layout = settings.get("scene_layout") or {}
    if not isinstance(layout, dict):
        layout = {}
    texts = [str(settings.get("scene") or ""), str(layout.get("details") or "")]
    for obj in layout.get("objects") or []:
        if not isinstance(obj, dict):
            continue
        if obj.get("asset") in ("person", "crowd"):
            continue
        texts.append(str(obj.get("description") or ""))
        texts.extend(str(d.get("text") or "") for d in obj.get("dressing") or []
                     if isinstance(d, dict))
    context = " ".join(texts)
    out = []
    if (FLAG_PATTERN and FLAGS.search(context) and BAVARIAN.search(context)
            and not OTHER_FLAG.search(context)):
        out.append({"noun": "flag", "pattern": "bavarian", "text": ""})
    # Scene Builder's flattened prompt repeats the prop descriptions. Read
    # their original fields for lettering rather than assigning that repeat
    # (or a character's quoted name) to a second sign.
    lettering = texts[1:] if layout.get("objects") else texts
    for text in lettering:
        if not SIGNS.search(text):
            continue
        quoted = [a or b for a, b in QUOTES.findall(text)]
        wording = quoted[0].strip() if len(quoted) == 1 else ""
        if not wording and not quoted and re.search(r"\boktoberfest\b", context, re.I):
            wording = "Oktoberfest"
        if not wording:
            continue
        noun = "banner" if re.search(r"\bbanners?\b", text, re.I) else "sign"
        req = {"noun": noun, "pattern": "", "text": wording}
        if req not in out:
            out.append(req)
    return out


def reference_size(box, edge=512):
    """Match the detected surface's aspect, keeping a bounded reference."""
    w, h = box[2:4]
    ratio = min(8.0, max(1 / 8.0, w / h))
    return (edge, max(64, round(edge / ratio))) if ratio >= 1 else (
        max(64, round(edge * ratio)), edge)


def lattice(x, y, pitch):
    """An affine rhombus tiling: fixed 5:8 diagonals, sheared to the right.

    Coordinates are in pixels, so a rectangular reference does not stretch
    the lozenges. Both families of parallel edges share a single phase.
    """
    u = (x - 0.25 * y) / pitch
    v = y / (1.6 * pitch)
    return (math.floor(u + v) + math.floor(u - v)) % 2


def pattern_png(size=512, height=None):
    """Area-sampled rhombi; preserve their shape across reference aspects."""
    width, height = int(size), int(height if height is not None else size)
    if not (1 <= width <= 2048 and 1 <= height <= 2048):
        raise ValueError("Pattern dimensions must be between 1 and 2048.")
    pitch = min(width, height) / 5.0
    pixels = bytearray(width * height * 4)
    # Four subpixel samples prevent ragged diagonal edges and phase bias.
    for y in range(height):
        for x in range(width):
            blue = sum(lattice(x + dx, y + dy, pitch)
                       for dx, dy in ((.25, .25), (.75, .25), (.25, .75), (.75, .75))) / 4
            at = (y * width + x) * 4
            pixels[at:at + 4] = bytes((*[round(255 + blue * (c - 255))
                                       for c in (35, 112, 192)], 255))
    return icons.png(bytes(pixels), width, height)


def plan(width, height, wanted, boxes):
    """Detected visible objects, largest first; never invent a repair region."""
    out, notes = [], []
    for ri, req in enumerate(wanted):
        noun = req["noun"]
        if sum(r["noun"] == noun for r in wanted) > 1:
            notes.append("Different wording was requested for several %ss; automatic "
                         "lettering was skipped rather than assigned to the wrong one." % noun)
            continue
        hits = sorted((b for b in boxes if len(b) >= 5 and b[4] == noun),
                      key=lambda b: -b[2] * b[3])
        kept = []
        for box in hits:
            x, y, w, h = box[:4]
            if not all(math.isfinite(v) for v in (x, y, w, h)) or min(w, h) < MIN_SIDE:
                continue
            x0, y0 = max(0, int(x)), max(0, int(y))
            x1, y1 = min(width, math.ceil(x + w)), min(height, math.ceil(y + h))
            if x1 <= x0 or y1 <= y0 or (x1 - x0) * (y1 - y0) > width * height * 0.35:
                continue
            # Duplicate SAM3 boxes for the same surface are one repair.
            if any(max(0, min(x1, k[2]) - max(x0, k[0])) *
                   max(0, min(y1, k[3]) - max(y0, k[1])) >
                   0.6 * min((x1 - x0) * (y1 - y0), (k[2] - k[0]) * (k[3] - k[1]))
                   for k in kept):
                continue
            kept.append((x0, y0, x1, y1))
            bounded = (x0, y0, x1 - x0, y1 - y0)
            out.append(dict(req, box=bounded, crop=wear.crop_for(width, height, bounded),
                            name=noun))
            if len(out) >= MAX_TARGETS:
                return out, notes
            # Reserve a slot for each later design/lettering request rather
            # than spending the whole bounded pass on the flags alone.
            if len(out) >= MAX_TARGETS - (len(wanted) - ri - 1):
                break
    return out, notes


def prompt(item):
    keep = (" Keep the object's outline, position, perspective, folds, material, "
            "lighting and shadows of image 1. Keep the people and background unchanged.")
    if item.get("pattern"):
        return ("Correct only the printed pattern on the flag in image 1. Use the "
                "regular alternating blue-and-white diagonal lozenges of image 2, "
                "evenly repeated across the cloth and following its folds and perspective. "
                "The white lozenges are white, not grey. No lettering or emblem." + keep)
    return ('Correct only the lettering on the %s in image 1. It reads exactly "%s", '
            'with this spelling, capitalization and word order. Clear, consistent, '
            'well-spaced painted letters on the existing surface, following its '
            'perspective. No additional letters, words or logos.' % (item["noun"], item["text"]) + keep)


def detail_graph(image, items, seed, prefix, sam3, size, pattern=None):
    """Reuse the item editor's crop/refine/composite, with detail-only prompts.

    Flags get the design swatch as image 2; lettering uses just its own crop
    as image 1. The blend is additionally bounded to the selected object,
    so another flag/sign inside the contextual crop cannot be overwritten.
    """
    if any(i.get("pattern") for i in items) and not pattern:
        raise ValueError("The flag pattern reference is missing.")
    planned = [dict(i, picture=pattern if i.get("pattern") else image) for i in items]
    g = wear.item_graph(image, planned, seed, prefix, sam3, size)
    for j, item in enumerate(items, 1):
        n, crop = "i%d_" % j, item["crop"]
        g[n + "text"]["inputs"]["text"] = prompt(item)
        g[n + "word"]["inputs"]["text"] = item["noun"]  # bare words for SAM3 masks
        g[n + "noise"]["inputs"]["noise_seed"] %= 2 ** 32
        if not item.get("pattern"):
            for suffix in ("photo", "fit", "lat2", "pos2", "neg2"):
                del g[n + suffix]
            g[n + "guide"]["inputs"].update(positive=[n + "pos1", 0], negative=[n + "neg1", 0])
        x, y, w, h = item["box"]
        g[n + "limit0"] = {"class_type": "SolidMask", "inputs": {
            "value": 0.0, "width": crop["width"], "height": crop["height"]}}
        g[n + "limit1"] = {"class_type": "SolidMask", "inputs": {
            "value": 1.0, "width": w, "height": h}}
        g[n + "limit2"] = {"class_type": "MaskComposite", "inputs": {
            "destination": [n + "limit0", 0], "source": [n + "limit1", 0],
            "x": x - crop["x"], "y": y - crop["y"], "operation": "or"}}
        # A print repair cannot extend the original cloth, replace its pole,
        # or leak into background holes. Use the original silhouette, eroded
        # slightly then softened inward; never union it with redrawn cloth.
        g[n + "ink0"] = {"class_type": "GrowMask", "inputs": {
            "mask": [n + "ma", 0], "expand": -1, "tapered_corners": True}}
        g[n + "ink1"] = {"class_type": "MaskToImage", "inputs": {"mask": [n + "ink0", 0]}}
        g[n + "ink2"] = {"class_type": "ImageBlur", "inputs": {
            "image": [n + "ink1", 0], "blur_radius": 2, "sigma": 0.7}}
        g[n + "ink3"] = {"class_type": "ImageToMask", "inputs": {
            "image": [n + "ink2", 0], "channel": "red"}}
        g[n + "ink4"] = {"class_type": "MaskComposite", "inputs": {
            "destination": [n + "ink3", 0], "source": [n + "ma", 0],
            "x": 0, "y": 0, "operation": "multiply"}}
        g[n + "ink5"] = {"class_type": "MaskComposite", "inputs": {
            "destination": [n + "ink4", 0], "source": [n + "limit2", 0],
            "x": 0, "y": 0, "operation": "multiply"}}
        g[n + "put"]["inputs"]["mask"] = [n + "ink5", 0]
        # Discard the clothing editor's before/after union and outward blend.
        for suffix in ("mb", "or", "grow", "q0", "q1", "q2", "wide", "m5",
                       "b0", "bs", "b1", "bu", "b2"):
            g.pop(n + suffix, None)
    return g
