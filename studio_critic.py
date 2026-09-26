"""The Visual Critic: a vision model looks at a picture the Image Studio just
made, says what is right and what is wrong, and the next pass fixes only what
is wrong. No tkinter and no network of its own: the vision model is handed in
(`studio_agent.Vision`), and `Studio._refine` in studio_imagegen runs the
passes with the tools the Image Studio already has.

Three kinds of state, kept apart on purpose:

- the **original intent** - what was asked, frozen (`intent_from`). Nothing the
  critic says is ever written into it;
- the **canonical state** - what is known about the picture: the people, the
  scene, the camera. It starts from the form and may take a detail the
  generator invented if the critic likes it (`merge_canonical`), one value per
  key, so it grows sideways, never by appending;
- the **corrections** - what the next pass fixes. Replaced after every look.
"""

import base64
import copy
import json
import re
import types

STATUSES = ("MATCH", "MISMATCH", "UNCERTAIN", "NEW_USEFUL_DETAIL")
CATEGORIES = ("identity", "body", "clothing", "scene", "interaction", "camera",
              "lighting", "realism")
ACTIONS = ("FACE_CORRECTION", "LOCAL_INPAINT", "OBJECT_CORRECTION", "GLOBAL_REFINEMENT",
           "FULL_REGENERATION", "NO_CHANGE")
MAX_PASSES = 3
MIN_CONFIDENCE = 0.6          # below this a mismatch is noted, never corrected
PROMOTE_CONFIDENCE = 0.75     # and below this an invented detail is not kept

# What each category is checked for. The model is small (a 7B VL at the time
# of writing); a list it can tick through reads better than an open question.
CHECKS = {
    "identity": "face shape, jaw and cheek width, nose, eyes, eyebrows, apparent age, "
                "facial hair, hair colour and style, skin, resemblance to the reference",
    "body": "height, build, shoulder width, proportions, pose, balance, stance, where "
            "the hands and feet are",
    "clothing": "type, colour, material, fit, accessories",
    "scene": "location, background, objects and where they are, architecture, "
             "perspective, scale",
    "interaction": "hands holding things correctly, grip, weight, contact, objects "
                   "passing through bodies",
    "camera": "angle, focal length, framing, depth of field, focus, motion blur, "
              "exposure, film texture, photographic realism",
    "lighting": "light and shadow direction, light on faces, whether the people match "
                "the background's light",
    "realism": "plastic or over-smoothed skin, cartoon look, fake hair, wrong hands or "
               "fingers, malformed anatomy, strange geometry, duplicated objects, AI "
               "textures",
}

CRITIC_PROMPT = """You are checking a generated picture against what was asked for.
The first picture is the generated one. Any pictures after it are references for
how a person must look.

WHAT WAS ASKED (the user's request, the source of truth):
%(intent)s

WHAT IS KNOWN ABOUT THE PICTURE:
%(canonical)s

Look at every category and check:
%(checks)s

Report what is RIGHT as well as what is wrong. Answer with JSON only, no prose:
{"summary": "one sentence on what the picture shows",
 "needs_refinement": true or false,
 "observations": [
  {"category": one of %(categories)s,
   "feature": "short name, e.g. character_a beard colour",
   "subject": "character_a, character_b, ... or scene or camera",
   "status": "MATCH" | "MISMATCH" | "UNCERTAIN" | "NEW_USEFUL_DETAIL",
   "confidence": 0.0 to 1.0,
   "severity": "minor" | "major",
   "observation": "what you see",
   "correction": "for a MISMATCH: what it should be instead, said as the fix",
   "target": "the one thing to fix in a word or two: face, hand, or the object's name",
   "value": "for a NEW_USEFUL_DETAIL: the detail, e.g. dark green wool jacket with horn buttons"}
 ]}

Rules: MATCH = it is right. MISMATCH = it differs from what was asked or known.
UNCERTAIN = you cannot tell (too small, hidden) - never guess. NEW_USEFUL_DETAIL =
something not asked for that fits well and is worth keeping. "major" is only for
a structural failure: the wrong composition, camera angle, place or environment,
a main subject missing or in the wrong place, a failed pose, badly wrong
perspective or light. Set needs_refinement to false when nothing meaningful is
wrong."""


# ============================================================ the three states

def intent_from(settings, prompt):
    """The original user intent: the words the user gave the form and the
    prompt they composed to. Frozen: a read-only view over a private copy."""
    s = settings or {}
    return types.MappingProxyType({
        "prompt": prompt or "",
        "scene": (s.get("scene") or "").strip(),
        "camera": (s.get("camera") or "").strip(),
    })


def _key(text):
    """One spelling per key: "Hair colour" and "hair_color" are the same key."""
    k = re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")
    return k.replace("colour", "color")


def initial_canonical(settings, look_keys=(), identities=(), style=None):
    """The canonical state as the form states it. Every value here came from
    the user, so each key is `locked`: a critic's guess never replaces it."""
    s = settings or {}
    person = {_key(k): str(s[k]).strip() for k in look_keys if str(s.get(k) or "").strip()}
    characters = {}
    if person or identities:
        a = dict(person)
        if identities:
            a["identity_reference"] = ", ".join(identities)
        characters["character_a"] = a
    scene = {"description": s["scene"].strip()} if (s.get("scene") or "").strip() else {}
    camera = {}
    if (s.get("camera") or "").strip():
        camera["description"] = s["camera"].strip()
    if style:
        camera["style"] = style
    state = {"characters": characters, "scene": scene, "camera": camera}
    state["locked"] = sorted(_paths(state))
    return state


def _paths(state):
    out = set()
    for cid, c in (state.get("characters") or {}).items():
        out.update("characters.%s.%s" % (cid, k) for k in c)
    for part in ("scene", "camera"):
        out.update("%s.%s" % (part, k) for k in state.get(part) or {})
    return out


def empty_corrections():
    return {"preserve": [], "correct": []}


# ================================================================ the critic

def _canonical_text(state):
    lines = []
    for cid, c in sorted((state.get("characters") or {}).items()):
        lines.append("%s: %s" % (cid, "; ".join("%s %s" % (k.replace("_", " "), v)
                                                for k, v in c.items())))
    for part in ("scene", "camera"):
        d = state.get(part) or {}
        if d:
            lines.append("%s: %s" % (part, "; ".join("%s %s" % (k.replace("_", " "), v)
                                                     for k, v in d.items())))
    return "\n".join(lines) or "(nothing beyond the request)"


def critic_prompt(intent, canonical):
    return CRITIC_PROMPT % {
        "intent": intent["prompt"] or intent["scene"] or "(no words given)",
        "canonical": _canonical_text(canonical),
        "checks": "\n".join("- %s: %s" % (k, v) for k, v in CHECKS.items()),
        "categories": "|".join(CATEGORIES)}


def _json_in(text):
    """The JSON object in a model's answer, fences and chatter around it or not."""
    text = re.sub(r"```(?:json)?", "", text or "")
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("the vision model answered without JSON")
    return json.loads(text[start:end + 1])


def _conf(v, default=0.5):
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return default


def clean_result(raw):
    """Whatever the model said -> a critic result every field of which is
    there and of its type. A status it made up is UNCERTAIN, never a fix."""
    raw = raw if isinstance(raw, dict) else {}
    obs = []
    for o in raw.get("observations") or []:
        if not isinstance(o, dict):
            continue
        status = str(o.get("status") or "").upper().replace(" ", "_")
        cat = str(o.get("category") or "").lower()
        obs.append({
            "category": cat if cat in CATEGORIES else "realism",
            "feature": str(o.get("feature") or o.get("observation") or "?")[:80],
            "subject": _key(o.get("subject") or "scene") or "scene",
            "status": status if status in STATUSES else "UNCERTAIN",
            "confidence": _conf(o.get("confidence")),
            "severity": "major" if str(o.get("severity")).lower() == "major" else "minor",
            "observation": str(o.get("observation") or "")[:300],
            "correction": str(o.get("correction") or "")[:300],
            "target": str(o.get("target") or "").strip().lower()[:40],
            "value": str(o.get("value") or "").strip()[:200],
        })
    needs = raw.get("needs_refinement")
    return {"summary": str(raw.get("summary") or "")[:400],
            "needs_refinement": needs if isinstance(needs, bool) else
            any(o["status"] == "MISMATCH" for o in obs),
            "observations": obs}


def analyze_generated_image(vision, generated_image, original_intent, canonical_state,
                            reference_images=(), max_tokens=1800):
    """The Visual Critic. `vision` is a studio_agent.Vision; `generated_image`
    the picture's PNG bytes; `reference_images` paths of pictures of the
    people (the first two are sent). -> a clean critic result (clean_result).
    Raises ValueError when the model gave no usable JSON."""
    content = [{"type": "text", "text": critic_prompt(original_intent, canonical_state)},
               _image_part(vision, generated_image, "image/png")]
    for path in list(reference_images)[:2]:
        try:
            with open(path, "rb") as f:
                raw = f.read()
        except OSError:
            continue
        mime = vision.MIME.get(path[path.rfind("."):].lower(), "image/png")
        content.append(_image_part(vision, raw, mime))
    response = vision.llm.chat([{"role": "user", "content": content}], max_tokens=max_tokens)
    text = (response["choices"][0]["message"].get("content") or "").strip()
    return clean_result(_json_in(text))


def _image_part(vision, raw, mime):
    data = vision._encode(raw, mime) if hasattr(vision, "_encode") else \
        base64.b64encode(raw).decode("ascii")
    return {"type": "image_url", "image_url": {"url": "data:%s;base64,%s" % (mime, data)}}


# =============================================================== the planner

HANDS = ("hand", "finger", "thumb", "grip", "wrist")
FACE_WORDS = ("face", "skin", "jaw", "cheek", "beard", "eye", "nose", "brow", "hair",
              "moustache", "age", "lips", "mouth")
STRUCTURAL = ("scene", "camera", "lighting", "body")


def action_for(o):
    """One confident mismatch -> (action, target)."""
    words = (o["feature"] + " " + o["target"]).lower()
    cat = o["category"]
    if o["severity"] == "major" and cat in STRUCTURAL:
        return "FULL_REGENERATION", ""
    if any(w in words for w in HANDS):
        return ("OBJECT_CORRECTION" if cat == "interaction" else "LOCAL_INPAINT"), "hand"
    if cat == "identity" or (cat == "realism" and any(w in words for w in FACE_WORDS)):
        return "FACE_CORRECTION", "face"
    if cat == "interaction" or (cat == "scene" and o["target"]):
        return "OBJECT_CORRECTION", o["target"] or "object"
    if cat in ("clothing", "body") and o["target"]:
        return "LOCAL_INPAINT", o["target"]
    return "GLOBAL_REFINEMENT", ""


def plan_next_refinement(critic_result, canonical_state, min_confidence=MIN_CONFIDENCE):
    """What to keep, what to fix, which tool fixes it, and whether a pass is
    wanted at all. -> {"preserve", "correct", "actions": [{"type", "target",
    "corrections"}], "ignored", "promote", "needs_pass", "deferred"}.

    A full regeneration, when any structural failure calls for one, is the
    only action. Otherwise the local fixes run together, and a whole-picture
    touch-up waits until nothing local is left: it would be redrawn over."""
    obs = critic_result["observations"]
    preserve = [o["feature"] for o in obs if o["status"] == "MATCH"]
    confident = [o for o in obs if o["status"] == "MISMATCH" and o["confidence"] >= min_confidence]
    ignored = [o["feature"] for o in obs if o["status"] == "MISMATCH"
               and o["confidence"] < min_confidence]
    locked = set(canonical_state.get("locked") or ())
    promote = [o for o in obs if o["status"] == "NEW_USEFUL_DETAIL" and o["value"]
               and o["confidence"] >= PROMOTE_CONFIDENCE
               and _detail_path(o) not in locked
               and not any(m["feature"] == o["feature"] for m in confident)]
    groups = {}
    for o in confident:
        kind, target = action_for(o)
        groups.setdefault((kind, target), []).append(_fix_text(o))
    actions = [{"type": k, "target": t, "corrections": c} for (k, t), c in groups.items()]
    deferred = []
    if any(a["type"] == "FULL_REGENERATION" for a in actions):
        actions = [{"type": "FULL_REGENERATION", "target": "",
                    "corrections": [c for a in actions for c in a["corrections"]]}]
    elif any(a["type"] != "GLOBAL_REFINEMENT" for a in actions):
        deferred = [c for a in actions if a["type"] == "GLOBAL_REFINEMENT"
                    for c in a["corrections"]]
        actions = [a for a in actions if a["type"] != "GLOBAL_REFINEMENT"]
    order = {k: i for i, k in enumerate(ACTIONS)}
    actions.sort(key=lambda a: order[a["type"]])
    needs = bool(critic_result["needs_refinement"] and actions)
    return {"preserve": preserve,
            "correct": [c for a in actions for c in a["corrections"]] if needs else [],
            "actions": actions if needs else [{"type": "NO_CHANGE", "target": "",
                                               "corrections": []}],
            "ignored": ignored, "promote": promote, "needs_pass": needs,
            "deferred": deferred if needs else []}


def _fix_text(o):
    fix = o["correction"].strip().rstrip(".")
    seen = o["observation"].strip().rstrip(".")
    return (seen + ". " + fix + ".") if fix and seen else (fix or seen) + "."


def _detail_path(o):
    subject = o["subject"]
    feature = _key(o["feature"].replace(subject.replace("_", " "), "")) or "detail"
    if subject.startswith("character"):
        return "characters.%s.%s" % (subject, feature)
    return "%s.%s" % ("camera" if o["category"] in ("camera", "lighting") else "scene", feature)


def merge_canonical(canonical_state, promotions):
    """A copy of the state with each promoted detail set at its one key. A
    key the user set is never replaced; a key the generator set before is
    (the newer look was the one judged good). -> (state, [paths set])."""
    state = copy.deepcopy(canonical_state)
    locked = set(state.get("locked") or ())
    changed = []
    for o in promotions:
        path = _detail_path(o)
        if path in locked:
            continue
        parts = path.split(".")
        where = state
        for p in parts[:-1]:
            where = where.setdefault(p, {})
        if where.get(parts[-1]) != o["value"]:
            where[parts[-1]] = o["value"]
            changed.append(path)
    return state, changed


# ======================================================== the prompt compiler

def build_refinement_instructions(original_intent, canonical_state, preserve, corrections):
    """The next pass's instructions in their sections. The intent goes in
    word for word; nothing here rewrites it."""
    chars = []
    for cid, c in sorted((canonical_state.get("characters") or {}).items()):
        chars.append("%s:\n%s" % (cid.replace("_", " ").title(),
                                  "\n".join("%s: %s" % (k.replace("_", " "), v)
                                            for k, v in c.items())))

    def block(d):
        return "\n".join("%s: %s" % (k.replace("_", " "), v) for k, v in (d or {}).items())
    sections = [("ORIGINAL USER INTENT", original_intent["prompt"]),
                ("CANONICAL CHARACTER INFORMATION", "\n\n".join(chars)),
                ("CANONICAL SCENE INFORMATION", block(canonical_state.get("scene"))),
                ("CANONICAL CAMERA INFORMATION", block(canonical_state.get("camera"))),
                ("PRESERVE", "\n".join("current " + p for p in preserve)),
                ("CORRECT", "\n".join(corrections))]
    return "FINAL GENERATION INSTRUCTIONS\n\n" + "\n\n".join(
        "%s:\n%s" % (h, body) for h, body in sections if body)


def generator_prompt(original_intent, canonical_state, corrections, focus=None):
    """What the diffusion model reads, from the same sections: prose, since
    FLUX reads no headings and samples at CFG 1, where "keep X" is not a
    control. What is kept is kept by what is left undrawn - the tool choice.
    `focus` ("face", "hand", an object) makes it a close-up's prompt."""
    known = []
    for cid, c in sorted((canonical_state.get("characters") or {}).items()):
        known.append("%s: %s" % (cid.replace("_", " "),
                                 ", ".join(str(v) for k, v in c.items()
                                           if k != "identity_reference")))
    for part in ("scene", "camera"):
        vals = [str(v) for k, v in (canonical_state.get(part) or {}).items()
                if str(v) not in original_intent["prompt"]]
        if vals:
            known.append(", ".join(vals))
    fixes = " ".join(c for c in corrections)
    head = ("A close-up of the %s from this photograph, in the same light, colour, focus "
            "and film texture as the rest of it. " % focus) if focus else ""
    return " ".join(x.strip() for x in (head, original_intent["prompt"], ". ".join(known)
                                        + ("." if known else ""), fixes) if x.strip())


# ================================================================== the log

def log_text(n, critic_result, plan):
    """The block for the log: what matched, what did not, what was chosen."""
    obs = critic_result["observations"]

    def lines(items):
        return "\n".join("- " + x for x in items) or "- (none)"
    mism = ["%s (%.2f)" % (o["feature"], o["confidence"]) for o in obs
            if o["status"] == "MISMATCH"]
    return "\n".join([
        "VISUAL CRITIC PASS %d" % n, "",
        "Summary: " + (critic_result["summary"] or "-"), "",
        "Matches:", lines(plan["preserve"]), "",
        "Mismatches:", lines(mism), "",
        "Uncertain:", lines(o["feature"] for o in obs if o["status"] == "UNCERTAIN"), "",
        "New useful details kept:", lines("%s: %s" % (o["feature"], o["value"])
                                          for o in plan["promote"]), "",
        "Ignored (low confidence):", lines(plan["ignored"]), "",
        "Selected action:",
        " + ".join(sorted({a["type"] for a in plan["actions"]}, key=ACTIONS.index)), ""])


PROGRESS = {"FACE_CORRECTION": "Refining faces", "LOCAL_INPAINT": "Correcting %s",
            "OBJECT_CORRECTION": "Correcting the %s", "GLOBAL_REFINEMENT":
            "Refining the whole picture", "FULL_REGENERATION": "Regenerating the picture"}


def progress_text(action):
    t = PROGRESS.get(action["type"], action["type"])
    return (t % (action["target"] or "detail")) if "%s" in t else t
