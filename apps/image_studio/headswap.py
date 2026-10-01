"""The head swap before the final face swap: FLUX.2 Klein redraws each
profile's whole head from their first photo, then FaceFusion puts their face
on it. Stdlib only.

FaceFusion changes the inside of a face and nothing else, so the head it
lands on - its shape, hair, glasses - stayed the generated stranger's. Klein
takes the picture and a photo as two references and redraws the head as the
photo's, at the picture's angle and light. On its own that is a look-alike,
not the person; the face swap after it is what makes it them.

Measured on Partner (2026-09-29, the two pictures and five photos of the
FaceFusion and BFS trials; ArcFace against her photos): no swap 0.11-0.17,
BFS on Qwen-Image-Edit 0.03, Klein alone 0.23-0.35, inswapper_128 alone
0.78-0.81, Klein then inswapper_128 0.77-0.85 - and a sharper face than
inswapper's alone. inswapper was trained on the ArcFace that scores it, so
its numbers flatter it; the user judged the pictures ("klein with inswapper
seems to work well enough"). The seed matters: one close-up scored 0.23 and
0.35 on two seeds.

Klein 9B since 2026-10-01 (the user: "id would like to try flux klein 9B", then
the 4B removed). On the 12 stranger pictures of Partner's LoRA bench, with no
LoRA: 4B 0.386 (3 of 12 at 0.5 or more), 9B 0.555 (8 of 12); her 4B LoRA
0.647. After the face swap 0.849 / 0.860 / 0.875. By eye the 9B keeps the
picture's expression, clothes and light where the 4B pasted in the photo's
smile and top. About 8 s a head on the 5090. The 9B is under the FLUX
Non-Commercial License, so every picture it touched says so (`LICENSE_NOTE`).
"""
import os

import apps.image_studio.facefusion as facefusion

KLEIN = "flux-2-klein-9b-fp8.safetensors"   # black-forest-labs/FLUX.2-klein-9b-fp8, gated
FILES = {"diffusion_models": [KLEIN],
         "text_encoders": ["qwen_3_8b_fp8mixed.safetensors"],   # Comfy-Org/flux2-klein-9B
         "vae": ["flux2-vae.safetensors"]}
# The library's family for a LoRA this Klein takes (`imagegen.FAMILIES`): a
# 4B LoRA ("flux2") does not fit the 9B's layers.
FAMILY = "flux2-klein9b"
LICENSE_NOTE = ("Made in part with FLUX.2 [klein] 9B (the head swap), under the FLUX "
                "Non-Commercial License: not for commercial use.")
NODES = {"ReferenceLatent", "Flux2Scheduler", "EmptyFlux2LatentImage", "CFGGuider",
         "SamplerCustomAdvanced", "RandomNoise", "KSamplerSelect", "ImageScaleToTotalPixels",
         "ImageCropV2", "ImageCompositeMasked", "SAM3_Detect", "ColorTransfer"}
STEPS = 4                                 # the distilled model's own count, cfg 1
SIDE = 1024                               # what Klein draws the crop at
# At 3 faces wide and a quarter of a face high, long generated hair ran out of
# the bottom of the crop and stayed on the chest (live, 2026-09-29).
CROP = 4.0                                # the crop's side / the face's: room for the hair
RISE = 0.0                                # how much of a face the crop sits above centre
FIND = "face:8"
# SAM3's words for what is blended back. "head" alone stops at the jaw: hair
# that Klein took off the shoulders came back from the picture underneath.
# The words are bare. ComfyUI's SAM3 encoder takes the count off "face:8" but
# reads a lone "hair:1" as that text, colon and all, and SAM3 answered it with
# the whole person on one picture (so the body was Klein's too, and a pendant
# gone), the glasses on a second and the glasses and a wine bottle on a third
# (live, 2026-09-29). One is the count of a bare word already.
# "necklace": Klein draws the whole of one where the old hair hid part of it,
# a few px off the picture's own, and at the edge of the hair the cord forked
# (2 pictures of 5). Where there is a necklace it is Klein's from end to end.
WORDS = ["head", "hair", "necklace"]
OLD_HAIR = "hair"                         # the word of WORDS whose mask before is the old hair
TONE = 0.5                                # how far Klein's colour moves to the crop's
GROW = 12                                 # px round the head, at the crop's size
# Loose strands lie outside SAM3's hair: round wavy hair they stayed in the
# air as pale curls beside a head whose hair is now pulled back. 32 px left
# the furthest of them, 48 none, on 5 pictures of 5; 64 began to take a
# pendant.
STRANDS = 48                              # px round the OLD hair, at the crop's size
# A thin thing's mask is thin, and the blend's soft edge eats it from both
# sides: GROW round a cord left the picture's own cord showing faintly
# beside Klein's.
CORD = 24
# How far round a word's mask the blend reaches when that is further than
# GROW: {word: (px round it before, px round it after)}, at the crop's size.
# The old hair's reach is `strands`.
REACH = {"necklace": (CORD, CORD)}
# What the head wears is blended with it. A flower crown the generated person
# wore (her profile's "flower crown") lay outside SAM3's head and hair: Klein
# drew her head without it, and the blend's edge left the old crown as a grey
# ghost round the new head, on 8 runs of 8 with her head LoRA (2 pictures x
# 4 seeds, 2026-10-01). SAM3 masks a crown alike for "headwear", "hat",
# "flower crown", "tiara" and "headband". On the 12 bench pictures with
# nothing on any head it also found hair, a lace collar and a lily, so it
# counts only within WORN_NEAR faces of that picture's own head and hair:
# before, so the old crown goes whole; after, so a hat Klein drew from the
# photo is not cut at the hair.
WORN = "headwear"
WORN_NEAR = 0.5                           # faces round the head and hair (crop side / CROP)
EDGE = 0.08                              # the soft square's margin, of the crop's side
BLUR_MAX = 31                             # ImageBlur's largest radius
# The blend's soft edge, as a share of the crop's side. Klein's background
# is a few levels off the picture's, so wherever the mask's edge crosses
# plain background its shape shows: past the old crown it was a grey crown
# round the new head (2026-10-01), a 26 px blur being too short to hide an
# outline. The mask is grown by FEATHER first, so all it covers is still
# Klein's, then blurred over as far again - at a quarter of the size, as
# ImageBlur stops at BLUR_MAX.
FEATHER = 0.08
SHRINK = 4                                # the wide blur runs at 1/SHRINK of the crop
STATUS = "head_swap"
LABEL = "Head swap"

# "Replace the head ... keep their hair as in image 2" kept the generated
# person's long blonde hair under a darker crown, on every seed: Klein holds
# on to image 1's hair until it is told to take it off, and where. Saying
# whose light falls on the head is what stopped a studio photo's flat pink
# face in a low sun. The last two sentences are for what Klein draws where the
# hair was. "...with no clothing, fabric, necklace or jewellery taken from
# image 2" took image 1's own necklace off as well, which a blend through the
# head and hair alone shows as a pendant on a cord cut short (9 runs of 9),
# and still copied the photo's floral top on 2 of 9. Saying first what image 1
# keeps, then what is not copied: the cord whole on 9 of 9, nothing of the
# photo's on 9 of 9 (3 pictures x 3 seeds, 2026-09-29).
PROMPT = ("The person in image 1 now has the head of the person in image 2: the same face, "
          "the same glasses, and the same hair as image 2 in colour, length and style. "
          "Remove the hair of image 1 completely, including any of it lying on the neck, "
          "shoulders and chest, and every loose strand of it. The head is lit by the light "
          "of image 1, with the same skin tone as the body in image 1. Keep the head angle, "
          "expression, pose and background of image 1. Below the neck everything is image "
          "1, unchanged: its clothes, and the necklace or jewellery image 1 wears, if any. "
          "Nothing that is worn in image 2 - clothing, fabric, necklace, jewellery - is "
          "copied.")


def prompt_for(trigger=None):
    """PROMPT, naming the person of image 2 by their LoRA's trigger word when
    they have one (`head_graph`'s "lora")."""
    if not trigger:
        return PROMPT
    return PROMPT.replace("the head of the person in image 2",
                          "the head of %s, the person in image 2" % trigger, 1)


def lacks(inventory, nodes=None):
    """What a backend is missing for a head swap: model files, then nodes.
    An inventory not read yet is a backend that cannot: the pass is an
    extra, never a guess."""
    if inventory is None:
        return list(FILES["diffusion_models"])
    out = [f for kind, names in FILES.items() for f in names
           if f not in (inventory.get(kind) or ())]
    if nodes is not None:
        out += sorted(NODES - set(nodes))
    return out


def targets(width, height, boxes, profiles):
    """Whose head is which: [(profile, (x, y, w, h))], each face chosen as
    FaceFusion will choose it (`facefusion.target_face` over the faces left
    to right), so the head redrawn is the one swapped after. One profile
    falls back to the biggest face. A profile with no face of its own, or
    no photo on disk, is left out."""
    order = sorted(boxes, key=lambda b: b[0])
    norm = [[x / float(width), y / float(height), (x + w) / float(width), (y + h) / float(height)]
            for x, y, w, h in order]
    out, taken = [], []
    for i, p in enumerate(profiles):
        photo = (p.get("references") or [None])[0]
        if not photo or not os.path.isfile(photo):
            continue
        try:
            k = facefusion.target_face(norm, region=p.get("target_region"),
                                       point=p.get("target_point"),
                                       index=i if len(profiles) > 1 else None,
                                       count=len(profiles))
        except RuntimeError:
            if len(profiles) != 1 or not order:
                continue
            k = order.index(max(order, key=lambda b: b[2] * b[3]))
        if k not in taken:
            taken.append(k)
            out.append((p, tuple(order[k])))
    return out


def middle(width, height, face):
    """The middle of a face, 0-1 across and down the picture: where the face
    swap is pointed after its head was redrawn (a profile's `target_point`)."""
    x, y, w, h = face
    return (round((x + w / 2.0) / float(width), 4), round((y + h / 2.0) / float(height), 4))


def head_crop(width, height, face, crop=None, rise=None):
    """The square round a face that Klein redraws: CROP faces wide, RISE of
    a face higher than centred, kept inside the picture."""
    x, y, w, h = face
    big = max(w, h)
    side = int(min(big * (CROP if crop is None else crop), width, height))
    left = int(min(max(x + w / 2.0 - side / 2.0, 0), width - side))
    top = int(min(max(y + h / 2.0 - side / 2.0 - big * (RISE if rise is None else rise), 0),
                  height - side))
    return {"x": left, "y": top, "width": side, "height": side}


def head_graph(image, heads, seed, prefix, sam3, words=None, tone=None, size=None,
               strands=None, feather=None):
    """In `image` (a LoadImage name) each of `heads` ([{"crop": head_crop,
    "photo": a LoadImage name}]) is cut out, enlarged to SIDE, redrawn by
    Klein with the photo beside it (cut to "photo_crop" when a head gives
    one) - through the person's own Klein LoRA when the head names one
    ("lora": file, "strength", "trigger": said in the prompt, `prompt_for`;
    Build LoRA's) - shrunk back, its colour moved `tone` of the way to the crop's
    (ColorTransfer, reinhard_lab; TONE), and blended in - through the head,
    hair and necklace alone: SAM3's `words` (WORDS) in the crop before and
    after (the new hair may be bigger or smaller than the old), grown and
    softened, with the old hair (OLD_HAIR, before) grown `strands`
    (STRANDS) further for its loose strands and a word of REACH as far as
    that says, inside a soft square that has
    no margin where the crop ends at the edge of the picture (`size`, its
    width and height; without it every side has one). Klein draws the
    whole crop again, so nothing outside that is kept from it. Each head
    is drawn on the picture the last left; head n is seeded seed+n. Saved
    under `prefix` by node "save"."""
    g = {
        "unet": {"class_type": "UNETLoader", "inputs": {
            "unet_name": KLEIN, "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": FILES["text_encoders"][0], "type": "flux2", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": FILES["vae"][0]}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": PROMPT}},
        "neg": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["pos", 0]}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "sigmas": {"class_type": "Flux2Scheduler", "inputs": {
            "steps": STEPS, "width": SIDE, "height": SIDE}},
        "empty": {"class_type": "EmptyFlux2LatentImage", "inputs": {
            "width": SIDE, "height": SIDE, "batch_size": 1}},
        "image": {"class_type": "LoadImage", "inputs": {"image": image}},
        "sam": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": sam3}},
    }
    words = list(WORDS if words is None else words)
    tone = TONE if tone is None else tone
    strands = STRANDS if strands is None else strands
    feather = FEATHER if feather is None else feather
    for w, word in enumerate(words):
        g["word%d" % w] = {"class_type": "CLIPTextEncode", "inputs": {
            "text": word, "clip": ["sam", 1]}}
    names = [word.split(":")[0].strip() for word in words]
    anchors = [w for w, name in enumerate(names) if name in ("head", OLD_HAIR)]
    if anchors:
        g["worn"] = {"class_type": "CLIPTextEncode", "inputs": {
            "text": WORN, "clip": ["sam", 1]}}
    last = ["image", 0]
    for i, head in enumerate(heads):
        n, crop = "h%d_" % (i + 1), head["crop"]
        side = crop["width"]
        g[n + "cut"] = {"class_type": "ImageCropV2", "inputs": {
            "image": last, "crop_region": dict(crop)}}
        g[n + "big"] = {"class_type": "ImageScale", "inputs": {
            "image": [n + "cut", 0], "upscale_method": "lanczos", "width": SIDE,
            "height": SIDE, "crop": "disabled"}}
        g[n + "photo"] = {"class_type": "LoadImage", "inputs": {"image": head["photo"]}}
        photo = [n + "photo", 0]
        if head.get("photo_crop"):        # the photo's head alone: no clothes to copy
            g[n + "pcut"] = {"class_type": "ImageCropV2", "inputs": {
                "image": photo, "crop_region": dict(head["photo_crop"])}}
            photo = [n + "pcut", 0]
        g[n + "fit"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
            "image": photo, "upscale_method": "lanczos", "megapixels": 1.0,
            "resolution_steps": 1}}
        cond = {"pos": ["pos", 0], "neg": ["neg", 0]}
        model = ["unet", 0]
        if head.get("lora"):
            g[n + "lora"] = {"class_type": "LoraLoaderModelOnly", "inputs": {
                "model": model, "lora_name": head["lora"],
                "strength_model": float(head.get("strength", 1.0))}}
            model = [n + "lora", 0]
        if head.get("trigger"):
            g[n + "text"] = {"class_type": "CLIPTextEncode", "inputs": {
                "clip": ["clip", 0], "text": prompt_for(head["trigger"])}}
            g[n + "zero"] = {"class_type": "ConditioningZeroOut", "inputs": {
                "conditioning": [n + "text", 0]}}
            cond = {"pos": [n + "text", 0], "neg": [n + "zero", 0]}
        for k, pixels in (("1", [n + "big", 0]), ("2", [n + "fit", 0])):
            g[n + "lat" + k] = {"class_type": "VAEEncode", "inputs": {
                "pixels": pixels, "vae": ["vae", 0]}}
            for side_of in ("pos", "neg"):
                g[n + side_of + k] = {"class_type": "ReferenceLatent", "inputs": {
                    "conditioning": cond[side_of], "latent": [n + "lat" + k, 0]}}
                cond[side_of] = [n + side_of + k, 0]
        g[n + "guide"] = {"class_type": "CFGGuider", "inputs": {
            "model": model, "positive": cond["pos"], "negative": cond["neg"], "cfg": 1.0}}
        g[n + "noise"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": int(seed) + i}}
        g[n + "ks"] = {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": [n + "noise", 0], "guider": [n + "guide", 0], "sampler": ["sampler", 0],
            "sigmas": ["sigmas", 0], "latent_image": ["empty", 0]}}
        g[n + "dec"] = {"class_type": "VAEDecode", "inputs": {
            "samples": [n + "ks", 0], "vae": ["vae", 0]}}
        g[n + "small"] = {"class_type": "ImageScale", "inputs": {
            "image": [n + "dec", 0], "upscale_method": "lanczos", "width": side,
            "height": side, "crop": "disabled"}}
        drawn = [n + "small", 0]
        if tone:
            g[n + "tone"] = {"class_type": "ColorTransfer", "inputs": {
                "image_target": drawn, "image_ref": [n + "cut", 0],
                "method": "reinhard_lab", "source_stats": "per_frame",
                "strength": float(tone)}}
            drawn = [n + "tone", 0]
        # Each of WORDS, before and after, at the crop's own size.
        found, far = None, []
        for w in range(len(words)):
            name = words[w].split(":")[0].strip()
            for at, (k, src) in enumerate((("a", [n + "cut", 0]), ("b", [n + "small", 0]))):
                key = "%sm%d%s" % (n, w, k)
                g[key] = {"class_type": "SAM3_Detect", "inputs": {
                    "model": ["sam", 0], "image": src, "conditioning": ["word%d" % w, 0],
                    "threshold": 0.3, "refine_iterations": 2, "individual_masks": False}}
                reach = (strands if name == OLD_HAIR and k == "a"
                         else REACH.get(name, (0, 0))[at])
                if reach:
                    far.append((key, int(reach)))
                if found is not None:
                    g[key + "_or"] = {"class_type": "MaskComposite", "inputs": {
                        "destination": found, "source": [key, 0], "x": 0, "y": 0,
                        "operation": "or"}}
                    found = [key + "_or", 0]
                else:
                    found = [key, 0]
        # What the head wears (WORN), where it is near that picture's own
        # head and hair: the old crown goes whole, a new hat stays whole.
        near = max(2, int(side / CROP * WORN_NEAR))
        for k, src in (("a", [n + "cut", 0]), ("b", [n + "small", 0])):
            if not anchors:
                break
            anchor = ["%sm%d%s" % (n, anchors[0], k), 0]
            for w in anchors[1:]:
                g["%sw%sa%d" % (n, k, w)] = {"class_type": "MaskComposite", "inputs": {
                    "destination": anchor, "source": ["%sm%d%s" % (n, w, k), 0],
                    "x": 0, "y": 0, "operation": "or"}}
                anchor = ["%sw%sa%d" % (n, k, w), 0]
            g[n + "w" + k] = {"class_type": "SAM3_Detect", "inputs": {
                "model": ["sam", 0], "image": src, "conditioning": ["worn", 0],
                "threshold": 0.3, "refine_iterations": 2, "individual_masks": False}}
            g[n + "wn" + k] = {"class_type": "GrowMask", "inputs": {
                "mask": anchor, "expand": near, "tapered_corners": False}}
            g[n + "wh" + k] = {"class_type": "MaskComposite", "inputs": {
                "destination": [n + "w" + k, 0], "source": [n + "wn" + k, 0],
                "x": 0, "y": 0, "operation": "multiply"}}
            g[n + "wh" + k + "_or"] = {"class_type": "MaskComposite", "inputs": {
                "destination": found, "source": [n + "wh" + k, 0], "x": 0, "y": 0,
                "operation": "or"}}
            found = [n + "wh" + k + "_or", 0]
        grow = max(2, GROW * side // SIDE)
        g[n + "m4"] = {"class_type": "GrowMask", "inputs": {
            "mask": found, "expand": grow, "tapered_corners": True}}
        through = [n + "m4", 0]
        for f, (key, reach) in enumerate(far):
            # What reaches further (the old hair's loose strands, a cord):
            # round, not tapered, so a curl out to one side is reached as
            # far as one above.
            g["%sf%d" % (n, f)] = {"class_type": "GrowMask", "inputs": {
                "mask": [key, 0], "expand": max(grow, reach * side // SIDE),
                "tapered_corners": False}}
            g["%sf%d_or" % (n, f)] = {"class_type": "MaskComposite", "inputs": {
                "destination": through, "source": ["%sf%d" % (n, f), 0], "x": 0, "y": 0,
                "operation": "or"}}
            through = ["%sf%d_or" % (n, f), 0]
        # The soft square: no head reaches the crop's edge with a hard line.
        # Where the crop's edge is the picture's there is no line to hide,
        # and a margin there kept the top of the old hair.
        pad = max(1, int(side * EDGE))
        left, top, right, bottom = pad, pad, pad, pad
        if size:
            left = 0 if crop["x"] <= 0 else pad
            top = 0 if crop["y"] <= 0 else pad
            right = 0 if crop["x"] + side >= size[0] else pad
            bottom = 0 if crop["y"] + side >= size[1] else pad
        g[n + "q0"] = {"class_type": "SolidMask", "inputs": {
            "value": 0.0, "width": side, "height": side}}
        g[n + "q1"] = {"class_type": "SolidMask", "inputs": {
            "value": 1.0, "width": max(1, side - left - right),
            "height": max(1, side - top - bottom)}}
        g[n + "q2"] = {"class_type": "MaskComposite", "inputs": {
            "destination": [n + "q0", 0], "source": [n + "q1", 0], "x": left, "y": top,
            "operation": "or"}}
        # The soft edge: grown by `feather` so the blur leaves all that was
        # covered at nearly 1 (two sigmas in), then blurred out as far again.
        wide = int(side * feather)
        if wide >= 2 * SHRINK:
            g[n + "m6"] = {"class_type": "GrowMask", "inputs": {
                "mask": through, "expand": wide, "tapered_corners": True}}
            through = [n + "m6", 0]
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
                "image": [n + "b1", 0], "upscale_method": "bilinear", "width": side,
                "height": side, "crop": "disabled"}}
            blurred = [n + "bu", 0]
        else:                             # a crop too small to need it: as before
            radius = min(BLUR_MAX, max(1, side // 50))
            g[n + "b1"] = {"class_type": "ImageBlur", "inputs": {
                "image": [n + "b0", 0], "blur_radius": radius,
                "sigma": min(10.0, max(0.1, radius / 2.0))}}
            blurred = [n + "b1", 0]
        g[n + "b2"] = {"class_type": "ImageToMask", "inputs": {
            "image": blurred, "channel": "red"}}
        g[n + "put"] = {"class_type": "ImageCompositeMasked", "inputs": {
            "destination": last, "source": drawn, "x": crop["x"], "y": crop["y"],
            "resize_source": False, "mask": [n + "b2", 0]}}
        last = [n + "put", 0]
    g["save"] = {"class_type": "SaveImage", "inputs": {"images": last,
                                                       "filename_prefix": prefix}}
    return g
