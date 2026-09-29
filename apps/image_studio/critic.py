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

And one thing carried from look to look, the **faults**: what the last pass
redrew, each with a number. The next look says what became of each
(`score_fixes`), so a fault that is still there is redrawn harder, not the
same way again, one that was tried `MAX_TRIES` times is left to the user, and
a redraw that made things worse is taken back. The user's own notes on a
picture (Fix a spot) are faults too (`user_faults`): certain, never guessed.
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
OUTCOMES = ("CLEARED", "PERSISTS", "WORSE", "UNKNOWN")
MAX_PASSES = 3
MAX_TRIES = 2                 # redraws of one fault; still wrong after them, it is the user's
HARDER = 0.15                 # denoise added for each redraw that left the fault there
CLOSEUPS = 3                  # marked spots shown to the critic large, beside the picture
MIN_CONFIDENCE = 0.6          # below this a mismatch is noted, never corrected
PROMOTE_CONFIDENCE = 0.75     # and below this an invented detail is not kept
# What one picture costs the vision model at most: LM Studio scales a large
# photo down before qwen2.5-vl reads it, and a 4.9 MB reference came to 4,082
# tokens (a 896x1152 render, 1,316). Measured 2026-09-26.
IMAGE_TOKENS = 4096

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

REFERENCES = ("Any pictures after it are references for\nhow a person must look.")
CLOSE = ("The %d after it %s of fault%s %s, in that order, as %s\nnow, for you to "
         "judge %s by.")

CRITIC_PROMPT = """You are checking a generated picture against what was asked for.
The first picture is the generated one. %(pictures)s

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

# The backward look: what the last pass redrew, asked after by number. The
# model is small, so each fault is one line and the answer one word.
FOLLOWUP_PROMPT = """

FAULTS FOUND BEFORE. Each has been redrawn since. Find each one in the picture
and say what became of it:
%(faults)s

Add to your JSON, beside "observations":
 "followups": [
  {"id": the fault's number,
   "outcome": "CLEARED" | "PERSISTS" | "WORSE" | "UNKNOWN",
   "observation": "what you see there now",
   "correction": "for PERSISTS or WORSE: what it should be instead, said as the fix"}
 ]

CLEARED = it is right now. PERSISTS = it is still wrong. WORSE = the redraw
damaged it: a seam, a double, a smear, a patch of the wrong colour. UNKNOWN =
you cannot tell - never guess. A fault the user marked was there: judge only
whether it still is. Do not list these faults again under "observations"."""


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
    The user, so each key is `locked`: a critic's guess never replaces it."""
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


def critic_prompt(intent, canonical, faults=(), closeups=()):
    """The question. `faults` are what the last pass redrew, asked after by
    number; `closeups` the numbers of those whose close-up is sent too."""
    pictures = REFERENCES
    if closeups:
        one = len(closeups) == 1
        pictures = CLOSE % (len(closeups), "is a close-up" if one else "are close-ups",
                            "" if one else "s", ", ".join(str(i) for i in closeups),
                            "it is" if one else "they are", "it" if one else "them")
    text = CRITIC_PROMPT % {
        "pictures": pictures,
        "intent": intent["prompt"] or intent["scene"] or "(no words given)",
        "canonical": _canonical_text(canonical),
        "checks": "\n".join("- %s: %s" % (k, v) for k, v in CHECKS.items()),
        "categories": "|".join(CATEGORIES)}
    if faults:
        text += FOLLOWUP_PROMPT % {"faults": "\n".join(
            "%d. %s%s" % (f["id"], fault_text(f),
                          " [marked by the user]" if f.get("source") == "user" else "")
            for f in faults)}
    return text


def _json_in(text):
    """The JSON object in a model's answer, fences and chatter around it or not."""
    text = re.sub(r"```(?:json)?", "", text or "")
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("the vision model answered without JSON")
    try:
        return json.loads(text[start:end + 1])
    except ValueError:
        pass
    # An answer cut off at max_tokens: keep every observation it finished.
    for cut in range(end, start, -1):
        if text[cut] == "}":
            try:
                return json.loads(text[start:cut + 1] + "]}")
            except ValueError:
                continue
    raise ValueError("the vision model's JSON could not be read")


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
    follow = []
    for f in raw.get("followups") or []:
        if not isinstance(f, dict):
            continue
        try:
            fid = int(f.get("id"))
        except (TypeError, ValueError):
            continue
        outcome = str(f.get("outcome") or "").upper().strip()
        follow.append({"id": fid,
                       "outcome": outcome if outcome in OUTCOMES else "UNKNOWN",
                       "observation": str(f.get("observation") or "")[:300],
                       "correction": str(f.get("correction") or "")[:300]})
    needs = raw.get("needs_refinement")
    return {"summary": str(raw.get("summary") or "")[:400],
            "needs_refinement": needs if isinstance(needs, bool) else
            any(o["status"] == "MISMATCH" for o in obs),
            "observations": obs, "followups": follow}


def analyze_generated_image(vision, generated_image, original_intent, canonical_state,
                            reference_images=(), max_tokens=3000, faults=(), closeups=()):
    """The Visual Critic. `vision` is a studio_agent.Vision; `generated_image`
    the picture's PNG bytes; `reference_images` paths of pictures of the
    people (the first two are sent). `faults` are what the last pass redrew,
    to be scored (`score_fixes`); `closeups` [(fault id, PNG bytes)] shows
    some of them large, and is sent instead of the references: a small model
    told two things about the pictures after the first mixes them up.
    -> a clean critic result (clean_result). Raises ValueError when the model
    gave no usable JSON."""
    closeups = list(closeups)[:CLOSEUPS]
    content = [{"type": "text", "text": critic_prompt(
        original_intent, canonical_state, faults, [i for i, _ in closeups])},
        _image_part(vision, generated_image, "image/png")]
    for _, raw in closeups:
        content.append(_image_part(vision, raw, "image/png"))
    if closeups:
        reference_images = ()
    for path in list(reference_images)[:2]:
        try:
            with open(path, "rb") as f:
                raw = f.read()
        except OSError:
            continue
        mime = vision.MIME.get(path[path.rfind("."):].lower(), "image/png")
        content.append(_image_part(vision, raw, mime))
    need = len(content[0]["text"]) // 3 + IMAGE_TOKENS * (len(content) - 1) + max_tokens
    fit = getattr(vision, "fit", None)
    window = fit(need) if fit else None
    if isinstance(window, int) and window < need:
        raise ValueError("the vision model's %s-token window cannot hold the picture, "
                         "its references and an answer (about %s)"
                         % ("{:,}".format(window), "{:,}".format(need)))
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


def plan_next_refinement(critic_result, canonical_state, min_confidence=MIN_CONFIDENCE,
                         carried=(), closed=()):
    """What to keep, what to fix, which tool fixes it, and whether a pass is
    wanted at all. -> {"preserve", "correct", "actions": [{"type", "target",
    "corrections", "faults", "tries"}], "ignored", "promote", "needs_pass",
    "deferred"}.

    A full regeneration, when any structural failure calls for one, is the
    only action. Otherwise the local fixes run together, and a whole-picture
    touch-up waits until nothing local is left: it would be redrawn over.

    `carried` are the faults still there after a redraw (`carry`): planned
    again beside what this look found, their action knowing how often it was
    tried. `closed` are features already scored this look; the critic naming
    one again as a new mismatch does not make it a second fault."""
    obs = critic_result["observations"]
    known = {_key(f["feature"]) for f in carried} | {_key(k) for k in closed}
    preserve = [o["feature"] for o in obs if o["status"] == "MATCH"]
    confident = list(carried) + [
        o for o in obs if o["status"] == "MISMATCH" and o["confidence"] >= min_confidence
        and _key(o["feature"]) not in known]
    ignored = [o["feature"] for o in obs if o["status"] == "MISMATCH"
               and o["confidence"] < min_confidence]
    locked = set(canonical_state.get("locked") or ())
    promote = [o for o in obs if o["status"] == "NEW_USEFUL_DETAIL" and o["value"]
               and o["confidence"] >= PROMOTE_CONFIDENCE
               and _detail_path(o) not in locked
               and not any(m["feature"] == o["feature"] for m in confident)]
    groups = {}
    for o in confident:
        groups.setdefault(action_for(o), []).append(o)

    def action(kind, target, faults):
        return {"type": kind, "target": target, "faults": faults,
                "corrections": [_fix_text(o) for o in faults],
                "tries": max(int(o.get("tries") or 0) for o in faults)}
    actions = [action(k, t, f) for (k, t), f in groups.items()]
    deferred = []
    if any(a["type"] == "FULL_REGENERATION" for a in actions):
        actions = [action("FULL_REGENERATION", "", [o for a in actions for o in a["faults"]])]
    elif any(a["type"] != "GLOBAL_REFINEMENT" for a in actions):
        deferred = [c for a in actions if a["type"] == "GLOBAL_REFINEMENT"
                    for c in a["corrections"]]
        actions = [a for a in actions if a["type"] != "GLOBAL_REFINEMENT"]
    order = {k: i for i, k in enumerate(ACTIONS)}
    actions.sort(key=lambda a: order[a["type"]])
    needs = bool((critic_result["needs_refinement"] or carried) and actions)
    return {"preserve": preserve,
            "correct": [c for a in actions for c in a["corrections"]] if needs else [],
            "actions": actions if needs else [{"type": "NO_CHANGE", "target": "",
                                               "corrections": [], "faults": [],
                                               "tries": 0}],
            "ignored": ignored, "promote": promote, "needs_pass": needs,
            "deferred": deferred if needs else []}


def _fix_text(o):
    fix = o["correction"].strip().rstrip(".")
    seen = o["observation"].strip().rstrip(".")
    return (seen + ". " + fix + ".") if fix and seen else (fix or seen) + "."


# ================================================================ the faults
# A fault is an observation that was redrawn, with a number ("id"), how often
# ("tries") and who found it ("source": "critic" or "user"). A marked spot's
# also says which spot ("spot") and whether it can be redrawn again ("redo").

def fault_text(f):
    """A fault in one line, as the critic is asked after it."""
    return "%s: %s" % (f["feature"], _fix_text(f)) if _fix_text(f) != "." else f["feature"]


def user_faults(spots, target="other", note=""):
    """Fix a spot's spots -> the faults they are, numbered from 1 in the
    spots' order. A spot's own `note` says what is wrong with it, else the
    fix's `note`, else only that the thing was marked. The user saw it, so it
    is certain, and it has been redrawn once: the fix did that."""
    noun = {"hand": "hand", "face": "face"}.get(target, "spot")
    out = []
    for i, sp in enumerate(spots):
        said = str(sp.get("note") or note or "").strip()[:200]
        out.append({"id": i + 1, "spot": i, "source": "user", "tries": 1,
                    "feature": "%s %d" % (sp.get("word") or noun, i + 1),
                    "category": "identity" if target == "face" else "realism",
                    "subject": "scene", "status": "MISMATCH", "confidence": 1.0,
                    "severity": "minor", "target": sp.get("word") or noun,
                    "observation": said or "the %s looked wrong" % noun,
                    "correction": "", "value": "", "redo": not sp.get("photo")})
    return out


def as_faults(observations, first_id):
    """What a pass redrew -> its faults: each numbered (kept, when it has a
    number already) and tried once more."""
    out = []
    for o in observations:
        f = dict(o, tries=int(o.get("tries") or 0) + 1)
        f.setdefault("source", "critic")
        if "id" not in f:
            f["id"] = first_id
            first_id += 1
        out.append(f)
    return out


def score_fixes(faults, critic_result):
    """What became of each fault, by this look. -> the faults, each with
    `outcome` (OUTCOMES) and, when it is still there, what is seen now and
    the fix in the critic's newer words. A fault the model did not answer
    for is read from its observations - the same feature called right or
    wrong - and is UNKNOWN without one: never a guess."""
    said = {f["id"]: f for f in critic_result.get("followups") or []}
    seen = {_key(o["feature"]): o for o in critic_result["observations"]}
    out = []
    for fault in faults:
        f, o = said.get(fault["id"]), seen.get(_key(fault["feature"]))
        outcome, now, fix = "UNKNOWN", "", ""
        if f:
            outcome, now, fix = f["outcome"], f["observation"], f["correction"]
        elif o and o["status"] == "MATCH":
            outcome, now = "CLEARED", o["observation"]
        elif o and o["status"] == "MISMATCH" and o["confidence"] >= MIN_CONFIDENCE:
            outcome, now, fix = "PERSISTS", o["observation"], o["correction"]
        scored = dict(fault, outcome=outcome)
        if outcome in ("PERSISTS", "WORSE"):
            # A user's note stays what is seen: it is what they said was wrong.
            if now and fault.get("source") != "user":
                scored["observation"] = now
            if fix:
                scored["correction"] = fix
        out.append(scored)
    return out


def carry(scored, max_tries=MAX_TRIES):
    """The scored faults still there -> (those to redraw again, those left
    to the user: tried `max_tries` times, or not redrawable)."""
    there = [f for f in scored if f["outcome"] in ("PERSISTS", "WORSE")]
    again = [f for f in there if f["tries"] < max_tries and f.get("redo", True)]
    return again, [f for f in there if f not in again]


def went_wrong(scored):
    """Whether the pass these faults were redrawn in is to be taken back: it
    damaged something and mended nothing."""
    return (any(f["outcome"] == "WORSE" for f in scored)
            and not any(f["outcome"] == "CLEARED" for f in scored))


def harder(denoise, tries, top):
    """The denoise of a redraw that follows `tries` that left the fault
    there: HARDER more for each, to `top` - never under what was asked."""
    return round(max(denoise, min(top, denoise + HARDER * max(0, int(tries or 0)))), 2)


def score_lines(scored):
    """The scores as the log and the job's notes say them."""
    return ["%s: %s%s" % (f["feature"], f["outcome"].lower(),
                          " after %d redraw%s" % (f["tries"], "" if f["tries"] == 1 else "s"))
            for f in scored]


def score_records(scored):
    """The scores as the record keeps them: what a later picture learns from."""
    return [{k: f.get(k) for k in ("id", "feature", "category", "target", "source",
                                   "tries", "outcome", "observation", "spot")
             if f.get(k) is not None} for f in scored]


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


# ================================================================ the memory
# What the critic promoted outlives the job: a person's kept details go with
# that identity, a scene's with its words, and every later picture of either
# is drawn with them, so the next picture starts where this one ended rather
# than inventing afresh. {"identities": {id: {key: value}}, "scenes": {scene
# key: {key: value}}}. The caller reads and writes the file.

def remember(memory, canonical_state, changed, identity_ids=(), scene=""):
    """A copy of `memory` with the promoted paths in `changed` filed. A
    person's detail is filed only when the picture had one identity:
    with two, character_a is not one of them."""
    memory = copy.deepcopy(memory or {})
    ids = list(identity_ids)
    for path in changed:
        parts = path.split(".")
        if parts[0] == "characters":
            if len(ids) != 1 or parts[1] != "character_a":
                continue
            value = (canonical_state["characters"].get("character_a") or {}).get(parts[2])
            bucket = memory.setdefault("identities", {}).setdefault(ids[0], {})
        else:
            skey = _key(scene)
            if not skey:
                continue
            value = (canonical_state.get(parts[0]) or {}).get(parts[1])
            bucket = memory.setdefault("scenes", {}).setdefault(skey, {})
        if value:
            bucket[parts[-1]] = value
    return memory


def recall(memory, identity_ids=(), scene="", set_keys=()):
    """The remembered details for a picture of these people in this scene,
    leaving out any key the form sets this time. -> [(key, value)]."""
    memory = memory or {}
    skip = {_key(k) for k in set_keys}
    out = []
    ids = list(identity_ids)
    if len(ids) == 1:
        out += sorted((memory.get("identities") or {}).get(ids[0], {}).items())
    if _key(scene):
        out += sorted((memory.get("scenes") or {}).get(_key(scene), {}).items())
    return [(k, v) for k, v in out if k not in skip and v]


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

def log_text(n, critic_result, plan, scored=()):
    """The block for the log: what became of the last pass's fixes, what
    matched, what did not, what was chosen."""
    obs = critic_result["observations"]

    def lines(items):
        return "\n".join("- " + x for x in items) or "- (none)"
    mism = ["%s (%.2f)" % (o["feature"], o["confidence"]) for o in obs
            if o["status"] == "MISMATCH"]
    return "\n".join([
        "VISUAL CRITIC PASS %d" % n, "",
        "Summary: " + (critic_result["summary"] or "-"), ""] + ([
            "Last pass's fixes:", lines(score_lines(scored)), ""] if scored else []) + [
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
    t = (t % (action["target"] or "detail")) if "%s" in t else t
    return t + (" again, harder" if action.get("tries") else "")
