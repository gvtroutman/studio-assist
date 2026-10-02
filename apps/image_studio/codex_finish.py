"""File handoff to Codex imagegen; no API key, model or ComfyUI required."""
import copy
import hashlib
import json
import math
import os
import time
import uuid

from apps.image_studio import imagegen as ig
from core import icons

DEFAULT_BRIEF = (
    "Finish this photograph: repair visible anatomy and object-contact errors, "
    "improve natural photographic detail and lighting. Preserve the composition, "
    "people's identities, age, body type, gender presentation, outfits and setting. "
    "Keep believable skin texture; avoid plastic skin, oversharpening and added text."
)
MAX_IMAGE_BYTES = 64 * 1024 * 1024
MAX_PIXELS = 16_000_000
REDACTION_RGBA = bytes((96, 96, 96, 255))


def pixel_boxes(regions, width, height):
    if not regions or not all(valid_region(r) for r in regions):
        raise ValueError("Mark the faces to protect before creating the handoff.")
    for x0, y0, x1, y1 in regions:
        yield (int(x0 * width), int(y0 * height), min(width, math.ceil(x1 * width)),
               min(height, math.ceil(y1 * height)))


def redact_faces(source, regions):
    """Replace protected pixels with opaque solid color; retain no PNG metadata."""
    pixels, width, height = icons.png_to_rgba(source)
    result = bytearray(pixels)
    for left, top, right, bottom in pixel_boxes(regions, width, height):
        row = REDACTION_RGBA * (right - left)
        for y in range(top, bottom):
            start = (y * width + left) * 4
            result[start:start + len(row)] = row
    return icons.png(result, width, height)


def face_regions(record):
    if (record.get("codex_finish") or {}).get("protected_regions"):
        return copy.deepcopy(record["codex_finish"]["protected_regions"])
    settings = record.get("settings") or {}
    return [list(p["region"]) for p in (settings.get("scene_faces") or {}).get("people") or []
            if valid_region(p.get("region"))]


def valid_region(region):
    return (isinstance(region, (list, tuple)) and len(region) == 4
            and all(isinstance(n, (int, float)) and math.isfinite(n) for n in region)
            and 0 <= region[0] < region[2] <= 1 and 0 <= region[1] < region[3] <= 1)


def restore_faces(source, edited, regions):
    """Exact source pixels within every protected rectangle, no face resampling."""
    original, width, height = icons.png_to_rgba(source)
    pixels, ew, eh = icons.png_to_rgba(edited)
    if (ew, eh) != (width, height):
        if abs((ew / eh) / (width / height) - 1) > .01:
            raise ValueError("The Codex image changed the canvas proportions. "
                             "Ask Codex to keep the original %sx%s framing." % (width, height))
        pixels = resize_rgba(pixels, ew, eh, width, height)
    result = bytearray(pixels)
    for left, top, right, bottom in pixel_boxes(regions, width, height):
        for y in range(top, bottom):
            start, end = (y * width + left) * 4, (y * width + right) * 4
            result[start:end] = original[start:end]
    return icons.png(result, width, height)


def resize_rgba(pixels, sw, sh, width, height):
    """Bilinear fit of the edited canvas only; original face pixels are never resized."""
    out = bytearray(width * height * 4)
    columns = []
    for x in range(width):
        sx = min(sw - 1, max(0, (x + .5) * sw / width - .5))
        x0 = int(sx)
        columns.append((x0, min(sw - 1, x0 + 1), sx - x0))
    for y in range(height):
        sy = min(sh - 1, max(0, (y + .5) * sh / height - .5))
        y0, dy = int(sy), sy - int(sy)
        arow, brow = y0 * sw * 4, min(sh - 1, y0 + 1) * sw * 4
        for x, (x0, x1, dx) in enumerate(columns):
            a, b, c, d = arow + x0 * 4, arow + x1 * 4, brow + x0 * 4, brow + x1 * 4
            dest = (y * width + x) * 4
            for channel in range(4):
                upper = pixels[a + channel] * (1 - dx) + pixels[b + channel] * dx
                lower = pixels[c + channel] * (1 - dx) + pixels[d + channel] * dx
                out[dest + channel] = round(upper * (1 - dy) + lower * dy)
    return bytes(out)


def read_picture(path):
    with open(path, "rb") as stream:
        data = stream.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("Choose a picture smaller than 64 MB.")
    size = ig.picture_size(data)
    if not size or min(size) <= 0:
        raise ValueError("Choose a PNG, JPEG, WebP, GIF or BMP picture.")
    if size[0] * size[1] > MAX_PIXELS:
        raise ValueError("Choose a picture with at most 16 megapixels.")
    return data, size


def export(root, record, source, brief=DEFAULT_BRIEF, regions=None):
    """Snapshot one selected image and its provenance in a fresh handoff folder."""
    if not record or not isinstance(record.get("settings"), dict):
        raise ValueError("Choose a finished picture with saved settings.")
    brief = brief.strip()
    if not brief:
        raise ValueError("Describe how you want the picture finished.")
    data, size = read_picture(source)
    regions = face_regions(record) if regions is None else regions
    if not regions or not all(valid_region(r) for r in regions):
        raise ValueError("Mark the faces to protect in Finish with Codex before sending.")
    redacted = redact_faces(data, regions)
    token = uuid.uuid4().hex
    folder = os.path.join(root, "codex-handoffs", token)
    private = os.path.join(root, "codex-private", token)
    os.makedirs(folder, exist_ok=False)
    os.makedirs(private, exist_ok=False)
    source_copy = os.path.join(private, "source.png")
    with open(source_copy, "wb") as stream:
        stream.write(data)
    public_image = os.path.join(folder, "redacted.png")
    with open(public_image, "wb") as stream:
        stream.write(redacted)
    output = os.path.join(folder, "finished.png")
    manifest = os.path.join(private, "handoff.json")
    snapshot = copy.deepcopy(record)
    request = (
        "Use imagegen to edit ONLY this face-redacted picture:\n" + public_image + "\n\n"
        "Privacy requirement: use only this image and this request. Do not search for, "
        "open or upload original pictures, identity references, local manifests or "
        "other files. The original faces stay local in Studio Assist.\n\n"
        "My finishing request:\n" + brief + "\n\n"
        + "Keep the exact %sx%s canvas and all people in exactly the same positions. " % size
        + "Leave the solid grey covers in place. Do not invent faces, heads or hair "
        "inside these protected normalized rectangles: "
        + json.dumps(regions) + ". Studio Assist will restore their exact original pixels "
        "on import, so keep lighting and geometry compatible around them.\n\n"
        "Inspect the result and save the finished image as PNG at:\n" + output
        + "\nLeave redacted.png unchanged. I will use Import finished "
        "image in Studio Assist to add the result to History.\n"
    )
    payload = {"version": 2, "source": os.path.abspath(source),
               "source_copy": source_copy, "source_sha256": hashlib.sha256(data).hexdigest(),
               "record": snapshot, "brief": brief, "output": output, "protected_regions": regions,
               "redacted_image": public_image, "redacted_sha256": hashlib.sha256(redacted).hexdigest()}
    with open(manifest, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    with open(os.path.join(folder, "request.md"), "w", encoding="utf-8") as stream:
        stream.write(request)
    return {"folder": folder, "manifest": manifest, "output": output, "request": request,
            "source": os.path.abspath(source), "redacted_image": public_image}


def output_folder(manifest):
    with open(manifest, encoding="utf-8") as stream:
        handoff = json.load(stream)
    output = handoff.get("output") if isinstance(handoff, dict) else None
    if not isinstance(output, str) or not output:
        raise ValueError("Choose a Studio Assist Codex handoff.json file.")
    return os.path.dirname(output)


def import_finished(history, manifest, picture):
    """Add an edited result, preserving source settings but not claiming old passes."""
    with open(manifest, encoding="utf-8") as stream:
        handoff = json.load(stream)
    if (not isinstance(handoff, dict) or handoff.get("version") not in (1, 2)
            or not isinstance(handoff.get("record"), dict)
            or not isinstance(handoff["record"].get("settings"), dict)):
        raise ValueError("Choose a Studio Assist Codex handoff.json file.")
    data, size = read_picture(picture)
    digest = hashlib.sha256(data).hexdigest()
    if digest == handoff.get("source_sha256"):
        raise ValueError("That is the original picture. Choose the finished image from Codex.")
    if digest == handoff.get("redacted_sha256"):
        raise ValueError("That is the covered handoff picture. Choose the finished image from Codex.")
    source = handoff["record"]
    source_data, _ = read_picture(handoff["source_copy"])
    if hashlib.sha256(source_data).hexdigest() != handoff.get("source_sha256"):
        raise ValueError("The handoff source changed. Create a new handoff from the original.")
    data = restore_faces(source_data, data, handoff.get("protected_regions"))
    size = ig.picture_size(data)
    digest = hashlib.sha256(data).hexdigest()
    now = time.time()
    record = {
        "id": time.strftime("%Y%m%d-%H%M%S", time.localtime(now)) + "-codex-" + uuid.uuid4().hex[:8],
        "created": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)),
        "created_ts": now, "settings": copy.deepcopy(source["settings"]),
        "prompt": handoff.get("brief") or DEFAULT_BRIEF,
        "workflow": "codex-imagegen", "workflow_label": "Finished with Codex imagegen",
        "model": {"label": "Codex imagegen"},
        "backend": {"id": "codex", "name": "Codex imagegen", "url": ""},
        "width": size[0], "height": size[1], "license": source.get("license") or "",
        "notes": ["Finished through a Codex handoff. Source settings are kept for reuse; "
                  "the imagegen edit cannot be repeated by a ComfyUI seed."],
        "codex_finish": {"source_id": source.get("id"), "source_record": source.get("path"),
                         "source_image": handoff.get("source"),
                         "source_sha256": handoff.get("source_sha256"),
                         "result_sha256": digest, "handoff": os.path.abspath(manifest),
                         "source_prompt": source.get("prompt") or "",
                         "brief": handoff.get("brief") or DEFAULT_BRIEF},
    }
    record["codex_finish"]["protected_regions"] = handoff["protected_regions"]
    record["notes"].append("Protected head regions keep their exact original pixels.")
    if handoff.get("version") == 2:
        record["codex_finish"]["faces_redacted_for_handoff"] = True
        record["notes"].append("Protected faces were covered in the image sent to Codex.")
    result = history.add(record, [("finished.png", data)])
    # Resolve the source through History, rather than trusting a manifest's path.
    for parent in history.list():
        if (parent.get("id") == source.get("id")
                and handoff.get("source") in (parent.get("images") or [])):
            state = parent.setdefault("codex_handoff", {})
            results = state.setdefault("results", {})
            results[handoff["source"]] = result["images"][0]
            remaining = [p for p in parent["images"] if p not in results]
            state["state"] = "pending" if remaining else "complete"
            history.update(parent)
            result["codex_finish"]["remaining"] = len(remaining)
            result["codex_finish"]["pipeline_results"] = [results.get(p, p) for p in parent["images"]]
            history.update(result)
            break
    return result
