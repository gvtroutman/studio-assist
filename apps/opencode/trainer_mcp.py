#!/usr/bin/env python3
"""
studio_trainer_mcp - OpenCode's trainer, as an MCP server for Claude Code.

OpenCode codes with the small local model, and that model starts every task
knowing only its prompt and a notebook of one-line lessons. The trainer is
Claude, working in Claude Code on the user's own plan: the user asks there
("review the last OpenCode task", "teach it to run the tests"), and Claude
reads what the task did through this server and keeps or forgets lessons in
the notebook. Nothing here calls a model, and nothing here edits code.

What it reads, all on this PC:
  - the OpenCode tab's saved conversations (`tasks/opencode/*.json` beside
    the settings): what the user asked, what the local model did, and every
    OpenCode report, "refused: ..." lines included;
  - what a task changed: its worktree, or its merge commit (`tasks.json` in
    OpenCode's state folder, kept by apps/opencode/mcp.py);
  - OpenCode's own session, when its server is running.

What it writes: the lessons notebook the tab uses (core.lessons: everywhere,
the app, this folder), and `lessons.md`, which OpenCode reads through its
config. The window re-reads a notebook file changed on disk, so a lesson
kept here reaches the tab's next request without a restart.

Claude Code starts it from the repo's `.mcp.json`. By hand:

    python apps/opencode/trainer_mcp.py --list-tools
"""

if __package__ in (None, ""):  # run as a script: import from the checkout
    import os as _os, sys as _sys
    _sys.path[0] = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))

import glob
import json
import os
import re
import sys
import time
import types

import core.lessons as lessons
import core.mcp as studio_mcp
import apps.opencode.mcp as oc

APP_ID = "opencode"
WINDOW = 30000            # characters of a transcript per call
DIFF_CHARS = 60000
LESSON_LAYERS = {"everywhere": "every tab", "app": "OpenCode's tab",
                 "folder": "this folder"}
SOURCES = {"user": "the user said so", "trainer": "the trainer kept it",
           "model": "the local model kept it", "review": "reflected after a task",
           "error": "a refused call"}
SESSION_ID = re.compile(r"\bses_[A-Za-z0-9]+\b")
REFUSED = re.compile(r"^\s*refused: ", re.M)


class TrainerError(Exception):
    pass


# ------------------------------------------------------------------- where

def settings_dir():
    """Beside the settings, as the window keeps it (core.chat.settings_path)."""
    return os.path.dirname(os.path.abspath(
        os.environ.get("STUDIO_SETTINGS") or
        os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~"),
                     "StudioAssistant", "settings.json")))


def records_dir():
    return os.path.join(settings_dir(), "tasks", APP_ID)


def notebook():
    """The tab's own notebooks, as core.lessons.for_app builds them."""
    app = types.SimpleNamespace(id=APP_ID, workspace=oc.WORKSPACE)
    stack = lessons.for_app(app, settings_dir())
    stack.load()
    return stack


def publish(stack):
    """lessons.md, which OpenCode's config names under `instructions`."""
    return lessons.write_brief_file(stack, os.path.join(oc.STATE_DIR, "lessons.md"))


# --------------------------------------------------------------- the records

def _text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content
                         if isinstance(p, dict) and p.get("type") == "text")
    return ""


def load_record(task_id):
    if not re.fullmatch(r"[0-9a-f]{8,64}", task_id or ""):
        raise TrainerError("task_id is the id trainer_tasks lists (hex).")
    path = os.path.join(records_dir(), task_id + ".json")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except OSError:
        raise TrainerError("No saved OpenCode task %s. trainer_tasks lists them." % task_id)
    except ValueError as e:
        raise TrainerError("Task %s could not be read: %s" % (task_id, e))
    return data.get("record") or {}, data.get("messages") or []


def transcript(messages):
    """The tab's conversation as plain text: the user, the local model, its
    calls and OpenCode's reports. The system prompt is left out."""
    parts, named = [], {}
    for m in messages:
        role = m.get("role")
        if role == "system":
            continue
        text = _text(m.get("content")).strip()
        if role == "assistant" and m.get("tool_calls"):
            for c in m["tool_calls"]:
                named[c.get("id")] = (c.get("function") or {}).get("name")
            calls = ", ".join("%s(%s)" % ((c.get("function") or {}).get("name"),
                                          ((c.get("function") or {}).get("arguments") or "")[:600])
                              for c in m["tool_calls"])
            text = (text + "\n" if text else "") + "[calls " + calls + "]"
        if role == "tool":
            text = "[result of %s]\n%s" % (m.get("name") or named.get(m.get("tool_call_id"))
                                           or "a tool", text)
        if text:
            parts.append("%s: %s" % ({"user": "USER", "assistant": "LOCAL MODEL",
                                      "tool": "TOOL"}.get(role, str(role).upper()), text))
    return "\n\n".join(parts)


def trouble_in(record, messages):
    """What went wrong in a saved task, as short phrases; [] for a clean one."""
    signs = []
    status = record.get("status") or ""
    if status and not status.startswith("response complete"):
        signs.append("ended as: " + status[:80])
    if any(lessons.looks_like_correction(b) for b in (record.get("briefs") or [])[1:]):
        signs.append("the user corrected it")
    names = [(c.get("function") or {}).get("name") for m in messages
             for c in (m.get("tool_calls") or [])]
    if "opencode_undo" in names or "opencode_discard" in names:
        signs.append("the user undid or discarded work")
    refused = sum(len(REFUSED.findall(_text(m.get("content")))) for m in messages
                  if m.get("role") in ("tool", "assistant"))
    if refused:
        signs.append("%d step(s) refused" % refused)
    errors = sum(1 for m in messages if m.get("role") == "tool"
                 and re.match(r"\s*(tool error|error|failed|refused by the validator)",
                              _text(m.get("content")), re.I))
    if errors:
        signs.append("%d failing call(s)" % errors)
    return signs


def sessions_in(messages):
    seen = []
    for m in messages:
        for sid in SESSION_ID.findall(_text(m.get("content")) + json.dumps(
                m.get("tool_calls") or [])):
            if sid not in seen:
                seen.append(sid)
    return seen


def when(path):
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(path)))


# ----------------------------------------------------------------- the tools

def result(text, error=False):
    return studio_mcp.result(text, error=error)


def t_tasks(a):
    limit = max(1, min(int(a.get("limit") or 10), 50))
    paths = sorted(glob.glob(os.path.join(records_dir(), "*.json")),
                   key=os.path.getmtime, reverse=True)
    if not paths:
        return result("No OpenCode tasks are saved yet (%s)." % records_dir())
    rows = []
    for path in paths[:limit]:
        task_id = os.path.splitext(os.path.basename(path))[0]
        try:
            record, messages = load_record(task_id)
        except TrainerError as e:
            rows.append("%s  (unreadable: %s)" % (task_id, e))
            continue
        briefs = record.get("briefs") or []
        asked = " ".join((briefs[0] if briefs else "(nothing asked)").split())[:100]
        trouble = trouble_in(record, messages)
        sids = sessions_in(messages)
        rows.append("%s  %s  %d message(s)  %s\n    asked: %s%s%s" % (
            task_id, when(path), len(briefs),
            ("last session %s%s" % (sids[-1], " (of %d)" % len(sids) if len(sids) > 1 else ""))
            if sids else "no OpenCode session",
            asked, ("\n    later: " + " | ".join(" ".join(b.split())[:60] for b in briefs[1:4]))
            if len(briefs) > 1 else "",
            ("\n    trouble: " + "; ".join(trouble)) if trouble else "\n    clean"))
    return result("The OpenCode tab's saved tasks, newest first:\n\n" + "\n\n".join(rows))


def t_task(a):
    task_id = a.get("task_id")
    if not task_id:
        paths = sorted(glob.glob(os.path.join(records_dir(), "*.json")), key=os.path.getmtime)
        if not paths:
            raise TrainerError("No OpenCode tasks are saved yet.")
        task_id = os.path.splitext(os.path.basename(paths[-1]))[0]
    record, messages = load_record(task_id)
    text = transcript(messages)
    start = max(0, int(a.get("start") or 0))
    piece = text[start:start + WINDOW]
    sids = sessions_in(messages)
    head = "Task %s (%s), status: %s. Characters %d-%d of %d." % (
        task_id, ("sessions " + ", ".join(sids[-3:]) + (" and %d earlier" % (len(sids) - 3)
                                                         if len(sids) > 3 else "")) if sids
        else "no session",
        record.get("status") or "?", start, start + len(piece), len(text))
    if start + WINDOW < len(text):
        head += " Next: start=%d." % (start + WINDOW)
    return result(head + "\n\n" + (piece or "(nothing past this point)"))


def t_diff(a):
    sid = a.get("session_id") or oc.last_session()
    t = oc.task(sid)
    if not t:
        raise TrainerError("No task record for session %s: it was discarded, or it worked in "
                           "the folder directly (then trainer_task has what it did)." % sid)
    if t.get("merged"):
        code, out = oc._git("show", "--stat", "--patch", t["merged"], check=False)
        what = "merged as %s" % t["merged"][:8]
    elif t.get("isolated") and os.path.isdir(t.get("dir") or ""):
        oc._git("add", "-A", cwd=t["dir"], check=False)
        code, out = oc._git("diff", "--cached", "--stat", "--patch", t["base"], cwd=t["dir"],
                            check=False)
        what = "in its copy, not merged"
    else:
        raise TrainerError("Session %s's copy is gone and it was never merged." % sid)
    if code:
        raise TrainerError("git could not show it: %s" % out.strip()[-500:])
    out = out.strip() or "(no changes)"
    if len(out) > DIFF_CHARS:
        out = out[:DIFF_CHARS] + "\n... [cut at %d characters]" % DIFF_CHARS
    return result("Session %s (%s), \"%s\":\n%s" % (sid, what, t.get("title") or "", out))


def t_session(a):
    sid = a.get("session_id") or oc.last_session()
    try:
        return oc.t_get_session({"session_id": sid, "limit": int(a.get("limit") or 20)})
    except oc.OpenCodeError as e:
        raise TrainerError("OpenCode's server is not answering (%s). Its reports are in the "
                           "tab's transcript: trainer_task." % e)


def t_lessons(a):
    stack = notebook()
    rows = []
    for label, nb in stack.layers:
        for l in nb.ordered():
            rows.append("- %s\n    [%s; %s%s]" % (l["text"], LESSON_LAYERS.get(label, label),
                                                  SOURCES.get(l["source"], l["source"]),
                                                  ", came up %d more time(s)" % l["hits"]
                                                  if l["hits"] else ""))
    note = ("\n\nThe notebook could not be read in full: %s" % stack.problem) if stack.problem else ""
    return result(("Lessons the local model and OpenCode read before every task (%d):\n%s"
                   % (len(rows), "\n".join(rows)) if rows else "No lessons yet.") + note)


def t_keep(a):
    stack = notebook()
    lesson, note = stack.add(a["lesson"], "trainer", scope=a.get("scope") or "here")
    publish(stack)
    if note == "already kept":
        return result("Already in the notebook: %s" % lesson["text"])
    where = LESSON_LAYERS.get(stack.where(lesson["text"]), "?")
    return result("Kept for %s: %s%s" % (where, lesson["text"],
                                         (" (%s)" % note) if note else ""))


def t_forget(a):
    stack = notebook()
    found = stack.find(a["lesson"])
    if found is None:
        raise TrainerError("No lesson with that text; copy it exactly from trainer_lessons.")
    if found["source"] == "user" and not a.get("user_agreed"):
        raise TrainerError("That lesson is the user's own. Ask them first; forget it with "
                           "user_agreed only when they said yes.")
    stack.remove(a["lesson"])
    publish(stack)
    return result("Forgot: %s" % found["text"])


def _schema(props=None, required=()):
    return {"type": "object", "additionalProperties": False,
            "properties": props or {}, "required": list(required)}


LESSON_TEXT = {"type": "string", "minLength": 8, "maxLength": lessons.LESSON_CHARS}

TOOLS = [
    ("trainer_tasks", t_tasks,
     "List the OpenCode tab's saved tasks, newest first: id, when, what the user asked, "
     "the OpenCode sessions it used, and signs of trouble (refused steps, undo, "
     "corrections, errors). Start a review here.",
     _schema({"limit": {"type": "integer", "minimum": 1, "maximum": 50}})),
    ("trainer_task", t_task,
     "Read one saved task's whole conversation: the user's messages, what the local model "
     "said and called, and OpenCode's reports. No task_id: the newest. Long ones come in "
     "windows; the first line names the next start.",
     _schema({"task_id": {"type": "string"}, "start": {"type": "integer", "minimum": 0}})),
    ("trainer_diff", t_diff,
     "What an OpenCode session changed in the code: its merge commit, or the diff in its "
     "own copy if not merged. No session_id: the last one.",
     _schema({"session_id": {"type": "string"}})),
    ("trainer_session", t_session,
     "OpenCode's own messages in a session (its reasoning and tool steps), when its "
     "server is running. No session_id: the last one.",
     _schema({"session_id": {"type": "string"},
              "limit": {"type": "integer", "minimum": 1, "maximum": 60}})),
    ("trainer_lessons", t_lessons,
     "Every lesson in the notebook the local model and OpenCode read before each task, "
     "with where it applies and who kept it. Read before keeping or forgetting one.",
     _schema()),
    ("trainer_keep", t_keep,
     "Keep one lesson for the local model and OpenCode: one or two plain sentences it can "
     "act on next time (an instruction, not a story), naming no session ids or line "
     "numbers. scope 'here' is this folder; 'everywhere' is every tab of the app.",
     _schema({"lesson": LESSON_TEXT,
              "scope": {"type": "string", "enum": ["here", "everywhere"]}}, ["lesson"])),
    ("trainer_forget", t_forget,
     "Remove a lesson that is wrong, stale or contradicted - better than keeping a second "
     "that disagrees with it. Give its text exactly. The user's own lessons need "
     "user_agreed: true, and only after they said so.",
     _schema({"lesson": {"type": "string"}, "user_agreed": {"type": "boolean"}},
             ["lesson"])),
]

READ_ONLY = {"trainer_tasks", "trainer_task", "trainer_diff", "trainer_session",
             "trainer_lessons"}
HINTS = {"trainer_keep": {"destructive": False, "idempotent": True, "open_world": False},
         "trainer_forget": {"destructive": True, "idempotent": True, "open_world": False}}
for _name in READ_ONLY:
    HINTS[_name] = {"open_world": False}

SERVER = studio_mcp.Server(
    "studio-trainer-mcp", "1.0",
    studio_mcp.tools_from_table(TOOLS, read_only=READ_ONLY, **HINTS),
    errors=(TrainerError, KeyError, TypeError, ValueError),
    instructions=(
        "You are the trainer for OpenCode in Studio Assist. OpenCode codes with a small "
        "local model that starts every task knowing only its prompt and a notebook of "
        "lessons; you improve it by reading what its tasks did and keeping or forgetting "
        "lessons. You never edit the code for it here. To review: trainer_tasks, then "
        "trainer_task (and trainer_diff / trainer_session when the transcript is not "
        "enough), then trainer_lessons, then keep at most three lessons that would have "
        "changed the outcome - forgetting any they contradict. Look for refused steps and "
        "why, wasted reads of whole large files, edits in the wrong place, tests not run, "
        "and conventions of this codebase it did not know. The user's own lessons outrank "
        "yours."))


if __name__ == "__main__":
    sys.exit(studio_mcp.main(SERVER))
