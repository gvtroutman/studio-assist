"""Ideas for updates to the app itself: a list kept beside the settings
(`ideas.json` in the data dir), shown by Help > Ideas for updates.

An idea is open, done or dropped. Nothing is ever lost by a click: done and
dropped ideas stay listed (dropped ones with the why, when one was given) so
a turned-down idea is not proposed again, and either can be reopened. Only
Delete takes a line away.

Two writers share the list: the user, in the window, and every tab's model,
through `studio_idea` (IDEA_TOOL, run by core/tasks.py's Executor) when it
hits a limit of the app itself. Each idea says who added it (`by`, and the
tab for a model's). Several tabs' workers and the window can each hold an
`Ideas`, so every edit re-reads the file under one lock before it writes -
otherwise the window would save its old list over an idea a model had just
added.

Claude Code sessions read the same file - its path is in docs/CODEMAP.md.
"""
import json
import os
import re
import tempfile
import threading
import time
import uuid

import core.doctor as doctor

STATUSES = ("open", "done", "dropped")
IDEA_CHARS = 300
MODEL_IDEAS_PER_RUN = 2   # a task that finds more has found a bad day, not ideas

LOCK = threading.Lock()   # the window and every tab's worker, in one process

IDEA_TOOL = {"type": "function", "function": {
    "name": "studio_idea",
    "description": ("Suggest one update to this app (Studio Assist) itself, for the user to "
                    "consider later: a tool this tab lacks, a step you had to work around, "
                    "something asked of you the app cannot do. Say what and why in one or "
                    "two sentences. Rarely - not for the user's project, not for a one-off "
                    "error. It changes nothing now."),
    "parameters": {"type": "object", "additionalProperties": False,
                   "properties": {"idea": {"type": "string", "minLength": 12,
                                           "maxLength": IDEA_CHARS,
                                           "description": "The update and why it is needed."}},
                   "required": ["idea"]}}}


def ideas_path():
    return os.path.join(doctor.data_dir(), "ideas.json")


def normal(text):
    return " ".join(re.sub(r"[^\w\s]", " ", (text or "").lower()).split())


class Ideas:
    broken = False                        # the file on disk would not read as JSON

    def __init__(self, path=None):
        self.path = path or ideas_path()
        self.items = []
        self.problem = self.load()

    def load(self):
        """A missing file is an empty list; a broken one is an empty list and a
        note, never a crash - and the next save sets it aside as .broken
        rather than writing over it. A file that could not be *opened* (held
        by a writer for an instant) is not broken: the list keeps what it
        had and says so, and no edit is made on top of it."""
        if not os.path.exists(self.path):
            self.items, self.broken = [], False
            return None
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
        except OSError as e:
            return "%s could not be read: %s" % (os.path.basename(self.path), e)
        except Exception as e:
            self.items, self.broken = [], True
            return "%s: %s" % (os.path.basename(self.path), e)
        items = []
        for item in (data.get("ideas", []) if isinstance(data, dict) else []):
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            status = item.get("status") if item.get("status") in STATUSES else "open"
            items.append({"id": str(item.get("id") or self._new_id()),
                          "text": text, "status": status,
                          "why": str(item.get("why") or ""),
                          "added": str(item.get("added") or ""),
                          "closed": str(item.get("closed") or ""),
                          "by": "model" if item.get("by") == "model" else "user",
                          "tab": str(item.get("tab") or "")})
        self.items, self.broken = items, False
        return None

    def save(self):
        """Atomic, like every other file the app keeps. Returns a note when the
        list could not be written, None when it was."""
        try:
            folder = os.path.dirname(os.path.abspath(self.path))
            os.makedirs(folder, exist_ok=True)
            if self.broken and os.path.exists(self.path):
                # The file that would not read is set aside, not written over.
                os.replace(self.path, self.path + ".broken")
            self.broken = False
            fd, tmp = tempfile.mkstemp(dir=folder, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump({"version": 1, "ideas": self.items}, f,
                              indent=1, ensure_ascii=False)
                os.replace(tmp, self.path)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
        except Exception as e:
            return "the ideas could not be saved: %s" % e
        self.problem = None
        return None

    def _new_id(self):
        return uuid.uuid4().hex[:12]      # the clock repeats within a tick on Windows

    def _find(self, idea_id):
        for item in self.items:
            if item["id"] == idea_id:
                return item
        return None

    def _edit(self, change):
        """Re-read, change, save - under the lock, so two writers in this
        process never save over each other. `change(self)` returns a note
        to stop without saving."""
        with LOCK:
            problem = self.load()
            if problem and not self.broken:
                return problem            # unreadable for now: change nothing
            stop = change(self)
            if stop:
                return stop
            return self.save()

    # ------------------------------------------------------------- editing
    # Each returns a note when something went wrong, None when it was kept,
    # so the window can say when a change did not land.

    def add(self, text, by="user", tab=""):
        text = " ".join(text.split())
        if not text:
            return None

        def change(book):
            book.items.append({"id": book._new_id(), "text": text, "status": "open",
                               "why": "", "added": time.strftime("%Y-%m-%d"), "closed": "",
                               "by": by, "tab": tab if by == "model" else ""})
        return self._edit(change)

    def mark(self, idea_id, status, why=""):
        if status not in STATUSES:
            return None

        def change(book):
            item = book._find(idea_id)
            if item is None:
                return "that idea is no longer on the list"
            item["status"] = status
            item["why"] = " ".join(why.split()) if status == "dropped" else ""
            item["closed"] = "" if status == "open" else time.strftime("%Y-%m-%d")
        return self._edit(change)

    def remove(self, idea_id):
        def change(book):
            item = book._find(idea_id)
            if item is None:
                return "that idea is no longer on the list"
            book.items.remove(item)
        return self._edit(change)

    def suggest(self, text, tab=""):
        """A model's idea, and the sentence it is told back. An idea already on
        the list is not added twice; one the user dropped is refused with
        their why, so the model learns it was turned down and not to ask again."""
        text = " ".join(text.split())[:IDEA_CHARS]
        self.load()
        same = next((i for i in self.items if normal(i["text"]) == normal(text)), None)
        if same is not None and same["status"] == "dropped":
            return ("The user already decided against this%s. Do not suggest it again."
                    % (": " + same["why"] if same["why"] else ""))
        if same is not None:
            return "Already on the list (%s); nothing added." % same["status"]
        problem = self.add(text, by="model", tab=tab)
        if problem:
            raise ValueError(problem)
        return ("Added to the user's ideas for updates: %s. Carry on with the task; "
                "the user reviews ideas later." % text)

    # ------------------------------------------------------------- reading

    def with_status(self, status):
        return [item for item in self.items if item["status"] == status]

    @staticmethod
    def who(item):
        """"you", "the After Effects model", "a model"."""
        if item.get("by") != "model":
            return "you"
        return "the %s model" % item["tab"] if item.get("tab") else "a model"

    def as_text(self):
        """The list as Markdown, for pasting into a Claude Code session."""
        lines = ["# Ideas for updates", ""]
        for status, head in (("open", "Open"), ("done", "Done"),
                             ("dropped", "Decided against")):
            items = self.with_status(status)
            if not items:
                continue
            lines += ["## " + head]
            for item in items:
                lines.append("- " + item["text"] + (
                    " (from %s)" % self.who(item) if item.get("by") == "model" else "") + (
                    " - why not: " + item["why"] if item["why"] else ""))
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"
