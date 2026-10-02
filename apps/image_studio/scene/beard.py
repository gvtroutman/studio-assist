"""Per-person facial hair: normalized attributes, words and head-local coverage.

Length is a relative 0..1 scale, not metres. Coverage is the share of the
style's cheek/chin area; density describes the hairs, not mask opacity.
"""
import math

STYLES = ("none", "stubble", "short", "full", "moustache", "goatee")
DEFAULT = {"style": "short", "length": 0.15, "coverage": 0.8,
           "density": 0.65, "color": ""}


def clean(value):
    if not isinstance(value, dict) or value.get("style") not in STYLES:
        return None
    out = {"style": value["style"]}
    for key in ("length", "coverage", "density"):
        try:
            n = float(value.get(key, DEFAULT[key]))
        except (ValueError, TypeError, OverflowError):
            n = DEFAULT[key]
        out[key] = max(0.0, min(1.0, n)) if math.isfinite(n) else DEFAULT[key]
    color = value.get("color", "")
    out["color"] = color.strip()[:80] if isinstance(color, str) else ""
    return out


def text(value):
    b = clean(value)
    if b is None:
        return ""
    if b["style"] == "none" or b["density"] == 0 or b["coverage"] == 0:
        return "clean-shaven"
    name = {"stubble": "stubble", "short": "short beard", "full": "full beard",
            "moustache": "moustache", "goatee": "goatee"}[b["style"]]
    density = "sparse" if b["density"] < 0.35 else "dense" if b["density"] > 0.75 else "moderate-density"
    length = "closely cropped" if b["length"] < 0.25 else "trimmed" if b["length"] < 0.6 else "longer"
    coverage = "low cheek coverage" if b["coverage"] < 0.5 else "high cheek coverage"
    return " ".join(x for x in (length, density, b["color"], name) if x) + ", " + coverage


def contains(point, value):
    """Coverage on the visible lower face, in the posed head's local metres.

    The mannequin has no detailed mouth mesh. These anatomical bands leave
    the mouth and nose clear; world-surface sampling makes them follow its
    sculpted jaw. This is a coarse region, not individual hair geometry.
    """
    b = value
    if b["style"] == "none" or not b["coverage"] or not b["density"]:
        return False
    x, y, z = point
    if z <= 0.018 or y < -0.012 or y > 0.085:
        return False
    upper_lip = abs(x) < 0.036 and 0.045 <= y <= 0.062
    chin = abs(x) < 0.038 and y < 0.034
    cheek = abs(x) >= 0.03 and y < 0.025 + 0.052 * b["coverage"]
    if b["style"] == "moustache":
        return upper_lip
    if b["style"] == "goatee":
        return upper_lip or chin
    return upper_lip or chin or cheek
