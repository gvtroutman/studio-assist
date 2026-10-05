"""Smile intent and mouth-only finishing regions. No Tk or model calls."""
import re

DENOISE = 0.35
LABEL = "Smile pass"
STATUS = "smile"
BASE = ("Keep this person's facial identity, age, skin tone and facial hair. "
        "Only refine the requested smile: natural lip contours, balanced mouth corners "
        "and realistic anatomy. Keep the head angle, nose, eyes, lighting and background.")
PROMPTS = {
    "closed": "A relaxed, gentle closed-mouth smile with the lips together. No visible teeth. ",
    "broad": "A natural broad smile, a believable upper row of individual teeth, "
             "subtle tooth variation, natural gums and lips. No extra rows of teeth. ",
    "laugh": "A natural laughing smile with a believable open mouth and individual teeth. "
             "Natural gums and lips, no extra rows of teeth. ",
    "smile": "A natural smile, keeping how open the mouth already is. If teeth are visible, "
             "keep a believable single upper row of individual teeth and natural gums. ",
}


def intent(text):
    """Smile shape, or nothing for a neutral/negative/non-smiling expression."""
    text = str(text or "").lower()
    if re.search(r"\b(?:no|not|without|never|stop)\s+(?:a\s+|any\s+)?(?:smil\w*|grin\w*|laugh\w*)"
                 r"|\b(?:unsmiling|frowning|scowling|neutral|serious)\b", text):
        return ""
    if not re.search(r"\b(?:smil\w*|grin\w*|laugh\w*)\b", text):
        return ""
    if re.search(r"\b(?:soft|gentle|subtle|slight|closed[- ]mouth)\b", text):
        return "closed"
    if re.search(r"\blaugh\w*\b", text):
        return "laugh"
    if re.search(r"\b(?:broad|wide|beaming|toothy|grin\w*)\b", text):
        return "broad"
    return "smile"


def requests(settings):
    if settings.get("smile_pass", True) is False:
        return []
    targets = (settings.get("scene_faces") or {}).get("people") or []
    layout = settings.get("scene_layout") or {}
    out = []
    if targets:
        objects = {o.get("id"): o for o in layout.get("objects") or [] if isinstance(o, dict)}
        for p in targets:
            obj = objects.get(p.get("id"), {})
            text = p.get("expression") or (obj.get("look") or {}).get("expression")
            # A person's own description, then shared scene details; never
            # flattened words containing another person's expression.
            if not text:
                text = obj.get("description") or layout.get("details") or ""
            shape = intent(text)
            if shape:
                out.append({"shape": shape, "region": p.get("region"),
                            "identity": p.get("identity"), "id": p.get("id")})
        return out
    from apps.image_studio.imagegen import has_person
    if not has_person(settings, bool(settings.get("identities") or settings.get("scene_identities"))):
        return []
    text = settings.get("expression") or settings.get("scene") or ""
    shape = intent(text)
    return [{"shape": shape, "region": None}] if shape else []


def spots(width, height, wanted, faces, profiles=()):
    """Only unambiguous requested faces, with a small mouth/lip band.

    Region targets never fall back to another face. Form intent applies to
    the swapped subject(s), or to a single detected face. Background faces
    are left alone when there is no way to identify the intended subject.
    """
    from apps.image_studio import facefusion
    ordered = sorted(faces, key=lambda b: b[0])
    norm = [[x / width, y / height, (x + w) / width, (y + h) / height]
            for x, y, w, h in ordered]
    selected, skipped = [], 0
    for req in wanted:
        region = req.get("region")
        if region:
            targets = [{"target_region": region}]
        elif profiles:
            targets = list(profiles)
        elif len(ordered) == 1:
            targets = [{}]
        else:
            skipped += 1
            continue
        for i, target in enumerate(targets):
            try:
                k = facefusion.target_face(norm, region=target.get("target_region"),
                    point=target.get("target_point"), index=i if len(targets) > 1 else None,
                    count=len(targets))
            except RuntimeError:
                skipped += 1
                continue
            if any(s["face"] == ordered[k] for s in selected):
                continue
            x, y, w, h = ordered[k]
            x0, y0 = max(0, int(x + w * 0.12)), max(0, int(y + h * 0.58))
            x1, y1 = min(width, int(x + w * 0.88)), min(height, int(y + h * 0.91))
            if x1 <= x0 or y1 <= y0:
                skipped += 1
                continue
            selected.append({"x": (x0 + x1) // 2, "y": (y0 + y1) // 2,
                "size": int(max(64, max(w, h) * 1.6)), "box": [x0, y0, x1 - x0, y1 - y0],
                "face": ordered[k], "prompt": PROMPTS[req["shape"]] + BASE})
    return selected, skipped


def crops(width, height, spots):
    """Context around a mouth, with no generic fix-area growth into the nose."""
    from apps.image_studio.imagegen import fix_crops
    out = fix_crops(width, height, spots)
    for crop, spot in zip(out, spots):
        x, y, w, h = spot["box"]
        crop["area"] = (max(0, int(x - crop["x"])), max(0, int(y - crop["y"])),
                        min(crop["width"], int(x + w - crop["x"])),
                        min(crop["height"], int(y + h - crop["y"])))
    return out
