"""The item pass: each picture on the form's Wearing list - a shirt, a
necklace, a hat, shoes - put on the person after the picture is drawn, by
FLUX.2 Klein 9B, whatever model drew it. Stdlib only.

Z-Image Turbo, the main model, takes no reference picture, so before this a
picture of an item reached it only as words. Here SAM3 finds where the item
goes (the garment the model drew from those words, else the part of the body
that wears it), that part is cut out and enlarged, Klein redraws it with the
item's picture beside it, and it is blended back through the item alone -
SAM3's mask of it before and after, grown and softened - so the face, the
pose and the rest of the picture are the picture's own. It is the head
swap's machinery (`headswap.head_graph`) aimed at an item: a crop of any
shape, not a square, and no colour moved toward the crop's, which would
move the item's colour toward the garment it replaces.
"""
import re

import apps.image_studio.headswap as headswap

KLEIN = headswap.KLEIN
FILES = headswap.FILES
NODES = (headswap.NODES - {"ColorTransfer"}) | {"ImageScale", "GrowMask", "MaskComposite",
                                               "SolidMask", "ImageBlur", "MaskToImage",
                                               "ImageToMask", "ImageScaleBy"}
LICENSE_NOTE = headswap.LICENSE_NOTE.replace("(the head swap)", "(the item pass)")
STEPS = headswap.STEPS
AREA = 1024 * 1024                        # what Klein draws a crop at, in pixels
STATUS = "items"
LABEL = "Item pass"

# Where a thing is worn, by its words. The first that matches decides; what
# matches none is worn on the body (a shirt, a dress, a bag). `noun` is what
# SAM3 looks for: the word that matched, else a garment word in the name,
# else "clothing".
PLACES = [
    ("head", r"hats?|caps?|beanies?|fedoras?|berets?|helmets?|hoods?|headscarf|bandanas?|"
             r"headbands?|tiaras?|crowns?|visors?|turbans?"),
    ("face", r"sunglasses|glasses|spectacles|goggles|earrings?|masks?|piercings?|"
             r"headphones|earbuds"),
    ("neck", r"necklaces?|pendants?|chains?|chokers?|lockets?|scarf|scarves|ties?|bow tie|"
             r"collars?|lanyards?"),
    ("hand", r"watch(?:es)?|bracelets?|bangles?|rings?|gloves?|mittens?"),
    ("feet", r"shoes?|boots?|sneakers?|trainers?|heels?|sandals?|loafers?|slippers?|socks?|"
             r"flats|clogs?|pumps?|flip[- ]flops?"),
]
GARMENTS = (r"t-shirt|shirt|blouse|top|tank top|sweater|jumper|hoodie|sweatshirt|cardigan|"
            r"jacket|coat|blazer|vest|waistcoat|dress|gown|skirt|pants|trousers|jeans|"
            r"shorts|leggings|suit|jumpsuit|overalls|dungarees|apron|uniform|kimono|robe|"
            r"belt|bag|handbag|purse|backpack|tote")
# The order items go on: the body first, the small things last, so a
# necklace is drawn over the new shirt and not under it.
ORDER = ["body", "feet", "hand", "neck", "face", "head"]
# More than one is worn: both shoes, gloves. Their boxes are one crop.
PAIRS = {"feet", "hand"}
# Thin things: a mask a cord wide is eaten from both sides by the blend's
# soft edge (headswap.CORD), so it reaches further, px at Klein's size.
THIN = re.compile(r"\b(necklaces?|pendants?|chains?|chokers?|lockets?|bracelets?|bangles?|"
                  r"watch(?:es)?|rings?|earrings?|lanyards?)\b", re.I)
CORD = headswap.CORD
GROW = 16                                 # px round the item, at Klein's size
CONTEXT = 0.35                            # the crop reaches this much past the item each side
MIN_SIDE = 192                            # px: no crop smaller than this
EDGE = 0.06                               # the soft frame's margin, of the crop's shorter side
FEATHER = 0.05                            # the blend's soft edge, of the crop's shorter side
SHRINK = headswap.SHRINK
BLUR_MAX = headswap.BLUR_MAX
FIND = 4                                  # SAM3's count for a word looked for

# What Klein is told. Image 1 is the crop, image 2 the item's picture.
PROMPT = ("The person in image 1 now wears the %s from image 2, exactly as it is in image "
          "2: the same colour, pattern, print, material, shape and every detail. It is worn "
          "naturally, fitted to their body and pose, in the light of image 1. Keep everything "
          "else in image 1 as it is: the face, hair, skin, hands, pose, the other clothes and "
          "the background. Nothing else in image 2 is copied: no person, skin, hair, other "
          "clothing or background.")


def place_of(name):
    """(place, noun) for an item's name: where it is worn, and the word SAM3
    looks for. "green flannel shirt" -> ("body", "shirt"); "my lucky hat" ->
    ("head", "hat")."""
    low = (name or "").lower()
    for place, words in PLACES:
        m = re.search(r"\b(%s)\b" % words, low)
        if m:
            return place, m.group(1)
    m = None
    for m in re.finditer(r"\b(%s)\b" % GARMENTS, low):
        pass
    if m is not None:
        return "body", m.group(1)
    # A name that says no kind of thing ("Nike Air Max"): SAM3 is asked for
    # what the person wears, as "max" would find nothing to blend through.
    return "body", "clothing"


def ordered(items):
    """`items` ([{"name", "path"}]) in the order they go on (ORDER)."""
    return sorted(items, key=lambda i: ORDER.index(place_of(i["name"])[0]))


def find_words(items):
    """What SAM3 is asked for on the whole picture, once for all the items:
    each item's noun, then the person and face it is fitted to, and the
    hands and feet when an item is worn there."""
    words = []
    for i in items:
        noun = place_of(i["name"])[1]
        if noun not in words:
            words.append(noun)
    places = {place_of(i["name"])[0] for i in items}
    anchors = ["person", "face"] + (["hand"] if "hand" in places else []) + (
        ["foot"] if "feet" in places else [])
    return words + [a for a in anchors if a not in words]


def _inside(box, around, slack=0.1):
    x, y, w, h = box[:4]
    ax, ay, aw, ah = around[:4]
    cx, cy = x + w / 2.0, y + h / 2.0
    return (ax - aw * slack <= cx <= ax + aw * (1 + slack)
            and ay - ah * slack <= cy <= ay + ah * (1 + slack))


def _union(boxes):
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    return (x0, y0, x1 - x0, y1 - y0)


def where(width, height, item, boxes):
    """Where `item` is worn in a picture, from SAM3's boxes on it ([(x, y, w,
    h, word)], `find_words`): its noun's box on the main person (the biggest
    person; both of a pair), else the part of them that wears it - a hat
    above the face, a necklace below it, a shirt on the whole person. ->
    (x, y, w, h), or None when there is no one to put it on."""
    place, noun = place_of(item["name"])

    def said(word):
        return [b for b in boxes if len(b) > 4 and b[4] == word]
    people = sorted(said("person"), key=lambda b: -b[2] * b[3])
    person = people[0][:4] if people else None
    faces = [b[:4] for b in said("face") if person is None or _inside(b, person, 0)]
    face = max(faces, key=lambda b: b[2] * b[3]) if faces else None
    if person is None and face is None:
        return None
    if person is None:                    # a close-up: the face is the person
        fx, fy, fw, fh = face
        person = (fx - fw, fy - fh * 0.5, fw * 3, fh * 4)
    # Someone else's: a box whose middle is in a smaller person's box. A
    # hat SAM3 found on a passer-by behind her is not hers (live, 2026-10-01).
    others = [b[:4] for b in people[1:] if b[2] * b[3] < person[2] * person[3] * 0.5]
    hits = [b[:4] for b in said(noun) if _inside(b, person)
            and not any(_inside(b, o, 0) for o in others)]
    near = None
    if place in ("head", "face", "neck") and face is not None:
        fx, fy, fw, fh = face
        near = {"head": (fx - fw * 0.5, fy - fh * 0.9, fw * 2, fh * 1.5),
                "face": (fx - fw * 0.15, fy - fh * 0.05, fw * 1.3, fh * 1.1),
                "neck": (fx - fw * 0.4, fy + fh * 0.7, fw * 1.8, fh * 1.4)}[place]
        # Worn by this face: its middle within a face of where it is worn.
        hits = [b for b in hits if _inside(b, (near[0] - fw, near[1] - fh, near[2] + 2 * fw,
                                              near[3] + 2 * fh), 0)]
    hits.sort(key=lambda b: -b[2] * b[3])
    if hits:
        found = _union(hits[:2]) if place in PAIRS else hits[0]
        # A hat is redrawn with the whole head under it, not the hat alone.
        return _union([found, near]) if place == "head" else found
    px, py, pw, ph = person
    if near is not None:
        return near
    if place == "hand":
        hands = [b[:4] for b in said("hand") if _inside(b, person)]
        if hands:
            return _union(sorted(hands, key=lambda b: -b[2] * b[3])[:2])
    if place == "feet":
        feet = [b[:4] for b in said("foot") if _inside(b, person)]
        if feet:
            return _union(sorted(feet, key=lambda b: -b[2] * b[3])[:2])
        return (px, py + ph * 0.82, pw, ph * 0.18)
    if place == "body" and face is not None:  # below the face, to the knees
        fx, fy, fw, fh = face
        top = min(py + ph, fy + fh * 0.8)
        return (px, top, pw, max(fh, (py + ph) - top))
    return person


def crop_for(width, height, box):
    """The rectangle round `box` that Klein redraws: CONTEXT past it on every
    side, at least MIN_SIDE, kept inside the picture. -> {"x", "y", "width",
    "height"} in whole pixels."""
    x, y, w, h = box
    pad = CONTEXT * max(w, h)
    w2 = min(width, max(MIN_SIDE, w + 2 * pad))
    h2 = min(height, max(MIN_SIDE, h + 2 * pad))
    left = min(max(x + w / 2.0 - w2 / 2.0, 0), width - w2)
    top = min(max(y + h / 2.0 - h2 / 2.0, 0), height - h2)
    return {"x": int(left), "y": int(top), "width": int(w2), "height": int(h2)}


def drawn_size(crop):
    """The size Klein draws a crop at: its shape, AREA pixels, sides in 16s."""
    ratio = crop["width"] / float(crop["height"])
    h = (AREA / ratio) ** 0.5
    w = h * ratio
    return max(16, int(round(w / 16.0)) * 16), max(16, int(round(h / 16.0)) * 16)


def plan(width, height, items, boxes):
    """Each item that has somewhere to go, in the order they go on:
    [{"name", "path", "noun", "place", "crop"}]; and the names of those that
    do not."""
    out, missed = [], []
    for item in ordered(items):
        box = where(width, height, item, boxes)
        if box is None:
            missed.append(item["name"])
            continue
        place, noun = place_of(item["name"])
        out.append(dict(item, noun=noun, place=place, crop=crop_for(width, height, box)))
    return out, missed


def item_graph(image, items, seed, prefix, sam3, size):
    """In `image` (a LoadImage name, `size` its width and height) each of
    `items` ([plan's, with "picture": the item's LoadImage name]) is cut out,
    drawn again by Klein at drawn_size with the item's picture beside it,
    shrunk back and blended in through SAM3's mask of the item's noun in the
    crop before and after, grown, inside a soft frame with no margin where
    the crop ends at the picture's edge. Each is drawn on the picture the
    last left; item n is seeded seed+n. Saved under `prefix` by node "save"."""
    g = {
        "unet": {"class_type": "UNETLoader", "inputs": {
            "unet_name": KLEIN, "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": FILES["text_encoders"][0], "type": "flux2", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": FILES["vae"][0]}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "image": {"class_type": "LoadImage", "inputs": {"image": image}},
        "sam": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": sam3}},
    }
    last = ["image", 0]
    for i, item in enumerate(items):
        n, crop = "i%d_" % (i + 1), item["crop"]
        cw, ch = crop["width"], crop["height"]
        dw, dh = drawn_size(crop)
        g[n + "text"] = {"class_type": "CLIPTextEncode", "inputs": {
            "clip": ["clip", 0], "text": PROMPT % item["name"]}}
        g[n + "zero"] = {"class_type": "ConditioningZeroOut", "inputs": {
            "conditioning": [n + "text", 0]}}
        g[n + "cut"] = {"class_type": "ImageCropV2", "inputs": {
            "image": last, "crop_region": dict(crop)}}
        g[n + "big"] = {"class_type": "ImageScale", "inputs": {
            "image": [n + "cut", 0], "upscale_method": "lanczos", "width": dw,
            "height": dh, "crop": "disabled"}}
        g[n + "photo"] = {"class_type": "LoadImage", "inputs": {"image": item["picture"]}}
        g[n + "fit"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
            "image": [n + "photo", 0], "upscale_method": "lanczos", "megapixels": 1.0,
            "resolution_steps": 1}}
        cond = {"pos": [n + "text", 0], "neg": [n + "zero", 0]}
        for k, pixels in (("1", [n + "big", 0]), ("2", [n + "fit", 0])):
            g[n + "lat" + k] = {"class_type": "VAEEncode", "inputs": {
                "pixels": pixels, "vae": ["vae", 0]}}
            for side_of in ("pos", "neg"):
                g[n + side_of + k] = {"class_type": "ReferenceLatent", "inputs": {
                    "conditioning": cond[side_of], "latent": [n + "lat" + k, 0]}}
                cond[side_of] = [n + side_of + k, 0]
        g[n + "guide"] = {"class_type": "CFGGuider", "inputs": {
            "model": ["unet", 0], "positive": cond["pos"], "negative": cond["neg"], "cfg": 1.0}}
        g[n + "noise"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": int(seed) + i}}
        g[n + "sigmas"] = {"class_type": "Flux2Scheduler", "inputs": {
            "steps": STEPS, "width": dw, "height": dh}}
        g[n + "empty"] = {"class_type": "EmptyFlux2LatentImage", "inputs": {
            "width": dw, "height": dh, "batch_size": 1}}
        g[n + "ks"] = {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": [n + "noise", 0], "guider": [n + "guide", 0], "sampler": ["sampler", 0],
            "sigmas": [n + "sigmas", 0], "latent_image": [n + "empty", 0]}}
        g[n + "dec"] = {"class_type": "VAEDecode", "inputs": {
            "samples": [n + "ks", 0], "vae": ["vae", 0]}}
        g[n + "small"] = {"class_type": "ImageScale", "inputs": {
            "image": [n + "dec", 0], "upscale_method": "lanczos", "width": cw,
            "height": ch, "crop": "disabled"}}
        # The item before and after, at the crop's own size: the old shirt
        # goes whole, and the new one is not cut where the old one ended.
        g[n + "word"] = {"class_type": "CLIPTextEncode", "inputs": {
            "text": "%s:%d" % (item["noun"], FIND), "clip": ["sam", 1]}}
        for k, src in (("a", [n + "cut", 0]), ("b", [n + "small", 0])):
            g[n + "m" + k] = {"class_type": "SAM3_Detect", "inputs": {
                "model": ["sam", 0], "image": src, "conditioning": [n + "word", 0],
                "threshold": 0.3, "refine_iterations": 2, "individual_masks": False}}
        g[n + "or"] = {"class_type": "MaskComposite", "inputs": {
            "destination": [n + "ma", 0], "source": [n + "mb", 0], "x": 0, "y": 0,
            "operation": "or"}}
        short = min(cw, ch)
        scale = short / float(min(dw, dh))
        reach = CORD if THIN.search(item["name"]) else 0
        grow = max(2, int(max(GROW, reach) * scale))
        g[n + "grow"] = {"class_type": "GrowMask", "inputs": {
            "mask": [n + "or", 0], "expand": grow, "tapered_corners": True}}
        through = [n + "grow", 0]
        # The soft frame: no hard line where the crop ends inside the picture.
        pad = max(1, int(short * EDGE))
        left = 0 if crop["x"] <= 0 else pad
        top = 0 if crop["y"] <= 0 else pad
        right = 0 if crop["x"] + cw >= size[0] else pad
        bottom = 0 if crop["y"] + ch >= size[1] else pad
        g[n + "q0"] = {"class_type": "SolidMask", "inputs": {
            "value": 0.0, "width": cw, "height": ch}}
        g[n + "q1"] = {"class_type": "SolidMask", "inputs": {
            "value": 1.0, "width": max(1, cw - left - right),
            "height": max(1, ch - top - bottom)}}
        g[n + "q2"] = {"class_type": "MaskComposite", "inputs": {
            "destination": [n + "q0", 0], "source": [n + "q1", 0], "x": left, "y": top,
            "operation": "or"}}
        wide = int(short * FEATHER)
        if wide >= 2 * SHRINK:
            g[n + "wide"] = {"class_type": "GrowMask", "inputs": {
                "mask": through, "expand": wide, "tapered_corners": True}}
            through = [n + "wide", 0]
        g[n + "m5"] = {"class_type": "MaskComposite", "inputs": {
            "destination": through, "source": [n + "q2", 0], "x": 0, "y": 0,
            "operation": "multiply"}}
        g[n + "b0"] = {"class_type": "MaskToImage", "inputs": {"mask": [n + "m5", 0]}}
        if wide >= 2 * SHRINK:
            sigma = min(10.0, wide / 2.0 / SHRINK)
            g[n + "bs"] = {"class_type": "ImageScaleBy", "inputs": {
                "image": [n + "b0", 0], "upscale_method": "area", "scale_by": 1.0 / SHRINK}}
            g[n + "b1"] = {"class_type": "ImageBlur", "inputs": {
                "image": [n + "bs", 0], "blur_radius": min(BLUR_MAX, max(1, int(2 * sigma + 1))),
                "sigma": sigma}}
            g[n + "bu"] = {"class_type": "ImageScale", "inputs": {
                "image": [n + "b1", 0], "upscale_method": "bilinear", "width": cw,
                "height": ch, "crop": "disabled"}}
            blurred = [n + "bu", 0]
        else:
            radius = min(BLUR_MAX, max(1, short // 50))
            g[n + "b1"] = {"class_type": "ImageBlur", "inputs": {
                "image": [n + "b0", 0], "blur_radius": radius,
                "sigma": min(10.0, max(0.1, radius / 2.0))}}
            blurred = [n + "b1", 0]
        g[n + "b2"] = {"class_type": "ImageToMask", "inputs": {
            "image": blurred, "channel": "red"}}
        g[n + "put"] = {"class_type": "ImageCompositeMasked", "inputs": {
            "destination": last, "source": [n + "small", 0], "x": crop["x"], "y": crop["y"],
            "resize_source": False, "mask": [n + "b2", 0]}}
        last = [n + "put", 0]
    g["save"] = {"class_type": "SaveImage", "inputs": {"images": last,
                                                       "filename_prefix": prefix}}
    return g


def lacks(inventory, nodes=None):
    """What a backend is missing for the item pass: Klein's files, then nodes."""
    out = headswap.lacks(inventory)
    if inventory is not None and nodes is not None:
        out += sorted(NODES - set(nodes))
    return out
