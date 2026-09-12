#!/usr/bin/env python3
"""
studio_lessons - what the model learns after each task, per app.

A tab's model is small and starts every session knowing the app in general
and this bridge from its prompt. What it does not carry between sessions is
what went wrong last time and what the user said about it. This module is
that memory: a Notebook of one-line lessons per app, on disk beside the
settings, folded into the prompt at boot and, when one is added mid-session,
carried at the tail of the request until the next boot.

Lessons come from four places, each marked by `source`:

  user    - a message that begins "remember ..." / "from now on ..." is stored
            as it stands (`explicit_lesson`), and a correction ("no, ...",
            "I meant ...") marks the run for reflection (`looks_like_correction`).
  model   - the studio_remember tool, which the prompt tells the model to use
            for a convention it found or a correction it was given.
  error   - a validator refusal (`Executor.refusals`): an unsupported action,
            an unknown key, a wrong type. Those are facts about the contract
            the model got wrong once, and they are learned deterministically.
  review  - the end-of-task reflection (`reflect`): one extra, short request
            after a run that had errors or a correction in it, asking for one
            reusable sentence. Only runs when there was trouble - a clean run
            has nothing to teach.

Every entry is data, bounded, de-duplicated and best-effort on disk: a wrecked
or unwritable file costs a lesson, never the app.
"""

import json
import os
import re
import tempfile
import time

MAX_LESSONS = 40          # per app; the oldest low-value ones go first
LESSON_CHARS = 300        # one lesson, one sentence or two
BRIEF_CHARS = 4000        # the rendered block in the prompt
REFUSALS_PER_RUN = 3      # validator refusals learned from one run
REFLECT_CHARS = 24000     # of the conversation the reflection sees

# Higher survives longer when the notebook is full.
PRIORITY = {"user": 3, "model": 2, "review": 1, "error": 0}

REMEMBER_TOOL = {"type": "function", "function": {
    "name": "studio_remember",
    "description": ("Record one lesson for future tasks in this app: a correction the "
                    "user gave, a convention of this studio, a way a call fails and what "
                    "works instead, or a choice the user liked. One sentence, general "
                    "enough to apply next time - not a note about this task's ids. It "
                    "changes nothing in the project."),
    "parameters": {"type": "object", "additionalProperties": False,
                   "properties": {"lesson": {"type": "string", "minLength": 8,
                                             "maxLength": LESSON_CHARS,
                                             "description": "The lesson, as one sentence."}},
                   "required": ["lesson"]}}}

CORRECTION = re.compile(
    r"^\s*(no|nope|wrong|not that|that'?s (wrong|not (it|right|what))|actually|"
    r"i said|i meant|i asked for|don'?t|never|always|stop|undo|instead|not like that|"
    r"you (missed|forgot|ignored)|why did you|that (broke|isn'?t|is not))\b", re.I)
EXPLICIT = re.compile(
    r"^\s*(remember|from now on|in future|going forward|note that|note:|a rule:)"
    r"[:,\-]?\s*(that\s+|to\s+)?(?P<rest>.+)$", re.I | re.S)


def looks_like_correction(text):
    """Does this message read as the user correcting the previous answer?"""
    return bool(CORRECTION.match(text or ""))


def explicit_lesson(text):
    """The lesson in a message that states one outright, else None.
    "Remember that music goes on A3" -> "music goes on A3"."""
    m = EXPLICIT.match(text or "")
    if not m:
        return None
    rest = " ".join(m.group("rest").split()).strip(" .")
    return rest if len(rest) >= 8 else None


def normal(text):
    return " ".join(re.sub(r"[^\w\s]", " ", (text or "").lower()).split())


def clean(text):
    text = " ".join((text or "").split()).strip()
    if len(text) > LESSON_CHARS:
        text = text[:LESSON_CHARS - 1].rstrip() + "…"
    return text


class Notebook:
    """The lessons of one app, on disk as lessons/<app>.json beside the settings.

    `brief()` renders every lesson for the system prompt and remembers which
    ones it carried; `fresh()` renders only those added since, for the tail of
    a request mid-session - the system prompt is the head of the host's cached
    prefix and is not rewritten until the next boot.
    """

    def __init__(self, app_id, path=None):
        self.app_id = app_id
        self.path = path
        self.lessons = []
        self.problem = None
        self._carried = set()

    @classmethod
    def for_app(cls, app_id, base=None):
        if base is None:
            base = os.path.dirname(os.path.abspath(
                os.environ.get("STUDIO_SETTINGS") or
                os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~"),
                             "StudioAssistant", "settings.json")))
        return cls(app_id, os.path.join(base, "lessons", app_id + ".json"))

    # ------------------------------------------------------------------ disk

    def load(self):
        """Read the lessons back; returns a problem sentence or None."""
        self.lessons, self.problem = [], None
        if not self.path or not os.path.exists(self.path):
            return None
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict) or data.get("version") != 1:
                raise ValueError("unsupported lessons file")
            for item in data.get("lessons", []):
                text = clean(item.get("text", "")) if isinstance(item, dict) else ""
                if not text:
                    continue
                self.lessons.append({
                    "text": text,
                    "source": item.get("source") if item.get("source") in PRIORITY else "model",
                    "created": float(item.get("created") or 0),
                    "hits": int(item.get("hits") or 0)})
        except Exception as e:
            self.lessons = []
            self.problem = "%s: %s" % (os.path.basename(self.path), e)
        return self.problem

    def save(self):
        """Best-effort. Returns a note when the lessons could not be written."""
        if not self.path:
            return "lessons last for this session only; there is nowhere to save them"
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(self.path), suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump({"version": 1, "app": self.app_id, "lessons": self.lessons},
                              f, indent=1, ensure_ascii=False)
                os.replace(tmp, self.path)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
        except Exception as e:
            return "the lessons could not be saved: %s" % e
        return None

    # --------------------------------------------------------------- editing

    def find(self, text):
        key = normal(text)
        for lesson in self.lessons:
            if normal(lesson["text"]) == key:
                return lesson
        return None

    def add(self, text, source="model"):
        """Keep a lesson. Returns (lesson, note): the note says when it is a
        repeat of one already kept, or when it could not be saved."""
        text = clean(text)
        if len(text) < 8:
            raise ValueError("a lesson is at least a short sentence")
        if source not in PRIORITY:
            source = "model"
        existing = self.find(text)
        if existing is not None:
            existing["hits"] += 1
            if PRIORITY[source] > PRIORITY[existing["source"]]:
                existing["source"] = source
            self.save()
            return existing, "already kept"
        lesson = {"text": text, "source": source, "created": time.time(), "hits": 0}
        self.lessons.append(lesson)
        self._trim()
        return lesson, self.save()

    def _trim(self):
        # Full: drop the lesson that is least worth its line - lowest source
        # priority, then fewest repeats, then oldest - never the one just added.
        while len(self.lessons) > MAX_LESSONS:
            victim = min(self.lessons[:-1],
                         key=lambda l: (PRIORITY[l["source"]], l["hits"], l["created"]))
            self.lessons.remove(victim)

    def remove(self, text):
        lesson = self.find(text)
        if lesson is None:
            return False
        self.lessons.remove(lesson)
        self._carried.discard(normal(lesson["text"]))
        self.save()
        return True

    def learn_refusals(self, refusals):
        """Lessons from the validator's refusals in one run: (tool, message)
        pairs. Each is a fact about the contract, worth one line."""
        added = []
        for name, message in refusals[:REFUSALS_PER_RUN]:
            first = " ".join(str(message).split())
            text = "Calling %s: %s" % (name, first)
            lesson, note = self.add(text, "error")
            if note != "already kept":
                added.append(lesson)
        return added

    # ------------------------------------------------------------- rendering

    def ordered(self):
        return sorted(self.lessons, key=lambda l: (l["created"], l["text"]))

    @staticmethod
    def render(lessons):
        lines = ["- " + l["text"] for l in lessons]
        while lines and sum(len(x) + 1 for x in lines) > BRIEF_CHARS:
            lines.pop(0)                  # the oldest goes first
        return "\n".join(lines)

    def brief(self):
        """Every lesson, for the system prompt; marks them as carried."""
        lessons = self.ordered()
        self._carried = {normal(l["text"]) for l in lessons}
        return self.render(lessons)

    def fresh(self):
        """Lessons added since brief(), for the tail of a request."""
        return self.render([l for l in self.ordered() if normal(l["text"]) not in self._carried])


# ------------------------------------------------------------------ reflection

REFLECT_PROMPT = (
    "Look back over this task in %(app)s - what was asked, what failed, what the user "
    "corrected. Is there ONE lesson that would make a future task in this app go "
    "better: a correction the user gave, a call that failed and what worked instead, "
    "a convention of this studio you found? Reply with that lesson as one sentence "
    "beginning 'Lesson:', general enough to reuse and naming no ids from this task. "
    "If there is nothing worth keeping, reply exactly NONE. No tool calls.")


def reflect(llm, app_name, messages, max_chars=REFLECT_CHARS):
    """One short request after a troubled run: the conversation, then the
    question above. Returns the lesson text or None. The reply is never
    appended to the conversation, so it costs nothing next turn."""
    kept, size = [], 0
    for m in reversed(messages[1:]):
        # Whole exchanges only: a tool reply without its call is a malformed request.
        piece = len(json.dumps(m, ensure_ascii=False))
        if kept and size + piece > max_chars:
            break
        kept.insert(0, m)
        size += piece
    while kept and kept[0].get("role") == "tool":
        kept.pop(0)
    context = ([messages[0]] + kept +
               [{"role": "user", "content": REFLECT_PROMPT % {"app": app_name}}])
    reply = llm.chat(context, None, max_tokens=160)
    text = ((reply.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    return parse_reflection(text)


def parse_reflection(text):
    text = " ".join((text or "").split())
    if not text or text.upper().startswith("NONE"):
        return None
    m = re.search(r"lesson:\s*(.+)", text, re.I)
    if not m:
        return None
    lesson = m.group(1).strip().strip('"').strip()
    lesson = re.split(r"\s(?=NONE$)", lesson)[0]
    return clean(lesson) if len(lesson) >= 8 else None
