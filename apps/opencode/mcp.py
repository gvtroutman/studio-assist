#!/usr/bin/env python3
"""
studio_opencode_mcp - an MCP stdio bridge to an OpenCode server on this PC.

OpenCode is a coding agent: it reads, edits and runs code on its own. Here it
codes with the studio's local model and works in ONE folder - the workspace,
by default this repository - and it changes nothing without the user saying
so. studio_agent.ServerSpec starts `opencode serve` with a config that lets it
read freely but makes every edit, every command and every fetch *ask*, and
refuses anything outside the workspace. This bridge is where those asks go:

    opencode_ask ── prompt_async ──► OpenCode works ... pauses on a permission
         │                                         │
         └─ elicitation/create ─► the user: Allow once / Always / Reject
                                                   │
                  /permission/{id}/reply ◄─────────┘  ... until the session is idle

The answer travels from the user to this bridge to OpenCode. The model that
called opencode_ask never sees the question and cannot answer it: a client
that cannot ask its user (no MCP elicitation) gets every step refused.

Agentic is the user's switch for working without those cards (the tab's
header button; see `agentic`). With it on, and only in a task's own copy, an
edit to a file in the copy and a command from a short list of tests and reads
are answered here, and an ask that ends with to-dos open or tests failing is
sent back to OpenCode. No tool sets the switch, and the merge stays the user's.

The server is on loopback with a password (the key file ServerSpec writes at
each start), because a coding agent's HTTP API is not something a web page in
a browser on this PC should be able to reach.

Stdlib only. The protocol - framing, negotiation, validation, annotations,
progress, cancellation, elicitation - is studio_mcp's; this file is the tools.

    python apps/opencode/mcp.py --list-tools
    python apps/opencode/mcp.py --check
"""

if __package__ in (None, ""):  # run as a script: import from the checkout
    import os as _os, sys as _sys
    _sys.path[0] = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))

import base64
import json
import fnmatch
import os
import re
import shlex
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import core.mcp as studio_mcp

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_URL = "http://127.0.0.1:4096"
OPENCODE_URL = os.environ.get("OPENCODE_URL", DEFAULT_URL).rstrip("/")
# The folder OpenCode works in. This repository unless told otherwise; the
# engine reads the same variable with the same default - keep them agreeing.
WORKSPACE = os.environ.get("OPENCODE_WORKSPACE") or HERE
# Where ServerSpec keeps OpenCode's config and the server password. Read on
# every request: the server may be (re)started after this bridge.
STATE_DIR = os.environ.get("OPENCODE_STATE") or os.path.join(
    os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "StudioAssistant", "opencode")
KEY_FILE = os.path.join(STATE_DIR, "server.key")
USERNAME = "opencode"

DEFAULT_WORK = 600           # seconds OpenCode may work (not counting the user) per call
MAX_WORK = 1800
POLL = 1.0                   # seconds between looks at a working session
SETTLE = 2.0                 # a task just handed over may not show as busy yet
EVENT_WAIT = 5.0             # ... between looks while the event stream is up
EVENT_TIMEOUT = 90           # a stream silent this long is taken as dropped
LOOK = 5.0                   # seconds between counting to-dos and changed files
STALL = 120                  # seconds with no sign of work before progress says "stuck?"
CONTEXT_HIGH = 80            # % of the window at which the report warns
TEST_TIMEOUT = 300           # seconds the tests of an ask's changes may run
TEST_LOOK = 2.0              # seconds between looks at tests that are running
AGENTIC_ROUNDS = 3           # times one ask goes back to OpenCode by itself, Agentic on
MAX_REPLY_CHARS = 6000       # what an ask hands back; the executor caps at 8000
MAX_FILE_CHARS = 6000        # leave room for headers under the executor's 8000 cap
MAX_LIST = 400               # list_files cap
MAX_DIFF_CHARS = 6000
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build",
             ".opencode", ".claude", ".runtime", ".work", ".studio-attachments"}

# Every server route this bridge uses, in one place. opencode_status checks
# them against the server's own OpenAPI document (/doc) and says which ones a
# newer or older OpenCode does not list, so a renamed route is a sentence in
# the transcript rather than a mystery 404 the model improvises around.
ROUTES = {
    "health": "/global/health",
    "doc": "/doc",
    "sessions": "/session",
    "session": "/session/{sessionID}",
    "status": "/session/status",
    "message": "/session/{sessionID}/message",
    "prompt": "/session/{sessionID}/prompt_async",
    "abort": "/session/{sessionID}/abort",
    "permissions": "/permission",
    "permission_reply": "/permission/{requestID}/reply",
    "questions": "/question",
    "question_reply": "/question/{requestID}/reply",
    "question_reject": "/question/{requestID}/reject",
    "diff": "/vcs/diff",
    "todo": "/session/{sessionID}/todo",
    "events": "/event",
}


class OpenCodeError(Exception):
    """Anything OpenCode, or the network in front of it, refuses."""


class Stopped(Exception):
    """The user stopped the work - a Stop, a closed tab, or Cancel on a step."""


# ------------------------------------------------------------------- HTTP

def password():
    pw = os.environ.get("OPENCODE_SERVER_PASSWORD")
    if pw:
        return pw
    try:
        with open(KEY_FILE, encoding="utf-8") as f:
            return f.read().strip() or None
    except OSError:
        return None


def _request(path, data=None, method=None, directory=None):
    # OpenCode serves any folder it is told of, one instance each: a task's
    # sessions, permissions and status live in that task's own copy.
    if directory:
        path += ("&" if "?" in path else "?") + "directory=" + urllib.parse.quote(directory, safe="")
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    pw = password()
    if pw:
        token = base64.b64encode(("%s:%s" % (USERNAME, pw)).encode("utf-8")).decode("ascii")
        headers["Authorization"] = "Basic " + token
    body = json.dumps(data).encode("utf-8") if data is not None else None
    return urllib.request.Request(OPENCODE_URL + path, data=body, headers=headers,
                                  method=method or ("POST" if body is not None else "GET"))


def _open(req, timeout):
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:2000]
        if e.code == 401:
            raise OpenCodeError(
                "OpenCode at %s refused the password. It was started by something other "
                "than this window, or before its last restart; the user restarts it with "
                "the Start OpenCode button." % OPENCODE_URL)
        try:
            detail = json.loads(body)
        except ValueError:
            detail = body
        raise OpenCodeError("OpenCode answered HTTP %d: %s" % (e.code, _explain(detail)))
    except urllib.error.URLError as e:
        raise OpenCodeError(
            "Cannot reach OpenCode at %s (%s). It is not running; the user starts it "
            "with the Start OpenCode button in this window." % (OPENCODE_URL, e.reason))
    except OSError as e:
        raise OpenCodeError("Cannot reach OpenCode at %s (%s)." % (OPENCODE_URL, e))


def _explain(detail):
    if isinstance(detail, dict):
        for key in ("message", "error", "data"):
            v = detail.get(key)
            if isinstance(v, str):
                return v
            if isinstance(v, dict) and isinstance(v.get("message"), str):
                return v["message"]
        return json.dumps(detail)[:500]
    return str(detail)


def _read(resp):
    with resp as r:
        body = r.read().decode("utf-8")
    return json.loads(body) if body.strip() else {}


def get_json(path, timeout=15, directory=None):
    return _read(_open(_request(path, directory=directory), timeout))


def post_json(path, payload, timeout=30, directory=None):
    return _read(_open(_request(path, payload, directory=directory), timeout))


def route(name, **ids):
    path = ROUTES[name]
    for k, v in ids.items():
        path = path.replace("{%s}" % k, urllib.parse.quote(str(v), safe=""))
    return path


# ------------------------------------------------------------- the server

def health():
    """(version, note). Older servers have no /global/health; /doc always exists."""
    try:
        h = get_json(ROUTES["health"], timeout=5)
        return str(h.get("version", "?")), ""
    except OpenCodeError as e:
        if "HTTP 404" not in str(e):
            raise
    doc = get_json(ROUTES["doc"], timeout=10)
    return str((doc.get("info") or {}).get("version", "?")), "no /global/health route"


def unknown_routes():
    """Routes this bridge uses that the server's OpenAPI document does not list."""
    try:
        doc = get_json(ROUTES["doc"], timeout=10)
    except OpenCodeError:
        return []
    # Compare with the parameter names blanked: a server that calls it {id}
    # where we say {sessionID} still has the route.
    listed = {_shape(p) for p in (doc.get("paths") or {})}
    if not listed:
        return []
    return sorted(r for k, r in ROUTES.items() if k != "doc" and _shape(r) not in listed)


def _shape(path):
    return re.sub(r"\{[^}]*\}", "{}", path).rstrip("/")


def _rows(data, key):
    return data if isinstance(data, list) else (data or {}).get(key, [])


def sessions():
    return _rows(get_json(ROUTES["sessions"]), "sessions")


def new_session(title="", directory=None):
    s = post_json(ROUTES["sessions"], {"title": title} if title else {}, directory=directory)
    if not s.get("id"):
        raise OpenCodeError("OpenCode created a session with no id: %s" % json.dumps(s)[:300])
    return s


def messages(session_id):
    return _rows(get_json(route("message", sessionID=session_id), timeout=30,
                          directory=task_dir(session_id)), "messages")


def statuses(directory=None):
    data = get_json(ROUTES["status"], timeout=10, directory=directory)
    return data if isinstance(data, dict) else {}


def pending_permissions(directory=None):
    return _rows(get_json(ROUTES["permissions"], timeout=10, directory=directory), "permissions")


def pending_questions(directory=None):
    return _rows(get_json(ROUTES["questions"], timeout=10, directory=directory), "questions")


def prompt(session_id, text):
    """Hand OpenCode a task and return at once; `run` follows it."""
    post_json(route("prompt", sessionID=session_id),
              {"parts": [{"type": "text", "text": text}]}, timeout=30,
              directory=task_dir(session_id))


def abort(session_id):
    post_json(route("abort", sessionID=session_id), {}, timeout=10,
              directory=task_dir(session_id))


def reply_permission(request_id, reply, message="", directory=None):
    payload = {"reply": reply}
    if message:
        payload["message"] = message
    post_json(route("permission_reply", requestID=request_id), payload, timeout=15,
              directory=directory)


def reply_question(request_id, answers, directory=None):
    post_json(route("question_reply", requestID=request_id), {"answers": answers}, timeout=15,
              directory=directory)


def reject_question(request_id, directory=None):
    post_json(route("question_reject", requestID=request_id), {}, timeout=15,
              directory=directory)


# ------------------------------------------------------- message summaries

def _info(m):
    return m.get("info") if isinstance(m.get("info"), dict) else m


def _role(m):
    return _info(m).get("role", "?")


def _parts(m):
    return m.get("parts") or []


def summarize(m, limit=4000):
    """One message as prose: its text, and one line per tool it ran or file it
    touched. Reasoning and step markers are noise to the model on this side."""
    lines, files = [], []
    for p in _parts(m):
        kind = p.get("type")
        if kind == "text" and p.get("text"):
            lines.append(p["text"].strip())
        elif kind == "tool":
            state = p.get("state") or {}
            title = state.get("title") or ""
            status = state.get("status") or ""
            lines.append("[%s %s%s]" % (p.get("tool", "tool"), status,
                                        (": " + title) if title else ""))
            err = state.get("error")
            if err:
                lines.append("  error: %s" % str(err)[:300])
        elif kind == "patch":
            files += [f for f in (p.get("files") or []) if isinstance(f, str)]
        elif kind == "file" and p.get("filename"):
            files.append(p["filename"])
    text = "\n".join(l for l in lines if l)
    if len(text) > limit:
        text = text[:limit] + "\n... [%d more characters]" % (len(text) - limit)
    if files:
        text += "\nFiles touched: " + ", ".join(sorted(set(rel(f) for f in files)))
    return text


def rel(path):
    """A path OpenCode names, as the workspace-relative one the user knows."""
    if not isinstance(path, str):
        return str(path)
    try:
        root = os.path.realpath(WORKSPACE)
        # A relative path is OpenCode's, so relative to the workspace - not
        # to wherever this bridge happened to be started.
        full = os.path.realpath(os.path.join(root, path))
        if full == root or full.startswith(root + os.sep):
            return os.path.relpath(full, root).replace(os.sep, "/")
        # A task's copy: <worktrees>/<task>/<path> is <path> to the user.
        copies = os.path.realpath(worktrees_dir())
        if full.startswith(copies + os.sep):
            inner = os.path.relpath(full, copies).replace(os.sep, "/")
            return inner.partition("/")[2] or inner
    except (OSError, ValueError):
        pass
    return path.replace("\\", "/")


def workspace_note(sid=None):
    t = task(sid) if sid else None
    if t and t.get("isolated"):
        return ("This task works in its own copy of %s (branch %s), from its last commit. "
                "Nothing reaches the user's folder until the user merges it: "
                "opencode_merge. opencode_undo takes back the last ask; opencode_discard "
                "drops the task." % (WORKSPACE, t.get("branch")))
    return "OpenCode works in %s." % WORKSPACE


# ------------------------------------------------------------- the asking

# What the user may answer to a permission. The labels are the buttons.
DECISIONS = [("once", "Allow once"), ("always", "Always allow"), ("reject", "Reject")]


def describe_permission(p):
    """(headline, meta) for one permission request: what the user reads, and
    the pieces a client that can show more (a diff, a command) draws itself."""
    kind = p.get("permission") or "?"
    meta = p.get("metadata") or {}
    patterns = [x for x in (p.get("patterns") or []) if isinstance(x, str)]
    always = [x for x in (p.get("always") or []) if isinstance(x, str)]
    shown = {"kind": kind, "patterns": patterns, "always": always}
    if kind == "edit":
        target = rel(meta.get("filepath") or (patterns[0] if patterns else "?"))
        headline = "OpenCode wants to edit %s." % target
        shown["file"] = target
        if isinstance(meta.get("diff"), str):
            shown["diff"] = meta["diff"]
    elif kind == "bash":
        cmd = meta.get("command") or (patterns[0] if patterns else "?")
        headline = "OpenCode wants to run a command:\n%s" % cmd
        shown["command"] = cmd
    elif kind in ("webfetch", "websearch"):
        what = meta.get("url") or meta.get("query") or ", ".join(patterns) or "?"
        headline = "OpenCode wants to %s %s." % ("fetch" if kind == "webfetch" else
                                                  "search the web for", what)
    elif kind == "external_directory":
        headline = ("OpenCode wants to reach outside its folder: %s."
                    % ", ".join(patterns or ["?"]))
    elif kind == "doom_loop":
        headline = ("OpenCode has made the same call several times in a row and wants "
                    "to go on: %s." % ", ".join(patterns or ["?"]))
    else:
        # A tool from an add-on (an MCP server's) or one this bridge does not
        # know by name: say which, and show what it was given if OpenCode said.
        what = [x for x in patterns if x != "*"]
        headline = "OpenCode wants to use the tool %s%s." % (kind, (" on " + ", ".join(what))
                                                             if what else "")
        details = {k: v for k, v in meta.items() if k not in ("diff",)}
        if details:
            shown["command"] = json.dumps(details, indent=1, ensure_ascii=False)[:1500]
    if always:
        # The bridge keeps the grant, not OpenCode (see `granted`): it lasts
        # as long as the task, survives a restart and can be taken back.
        shown["always_label"] = grant_label(kind, always) + " for this task"
    return headline, shown


def permission_schema(shown):
    always = shown.get("always_label")
    names = [label if value != "always" or not always else "Always allow %s" % always
             for value, label in DECISIONS]
    return {"type": "object",
            "properties": {
                "decision": {"type": "string", "title": "Decision",
                             "enum": [v for v, _ in DECISIONS], "enumNames": names},
                "note": {"type": "string", "title": "Note for OpenCode",
                         "description": "Optional: what to do instead, or why not."}},
            "required": ["decision"]}


def ask_permission(p):
    """The user's (decision, note) for one permission request. Raises Stopped."""
    headline, shown = describe_permission(p)
    try:
        res = studio_mcp.elicit(headline, permission_schema(shown),
                                meta={"studio/approval": shown})
    except studio_mcp.Declined:
        return ("reject", "No one could be asked to approve this step, so it was refused. "
                          "Do not try it another way."), False
    action = res.get("action")
    if action == "cancel":
        raise Stopped(headline)
    content = res.get("content") or {}
    decision = content.get("decision") if action == "accept" else "reject"
    if decision not in dict(DECISIONS):
        decision = "reject"
    return (decision, (content.get("note") or "").strip()), True


def question_schema(q):
    options = [o for o in (q.get("options") or []) if isinstance(o, dict) and o.get("label")]
    props = {}
    if q.get("multiple"):
        for i, o in enumerate(options):
            props["pick_%d" % i] = {"type": "boolean", "title": o["label"],
                                    "description": o.get("description", ""), "default": False}
    elif options:
        props["answer"] = {"type": "string", "title": q.get("header") or "Answer",
                           "enum": [o["label"] for o in options],
                           "enumNames": [o["label"] for o in options]}
    if q.get("custom", True) or not options:
        props["other"] = {"type": "string", "title": "Or in your own words"}
    return {"type": "object", "properties": props}, options


def ask_question(req):
    """Answers for one OpenCode question request, each a list of labels, or
    None when the user declined. Raises Stopped on cancel."""
    answers = []
    for q in req.get("questions") or []:
        schema, options = question_schema(q)
        try:
            res = studio_mcp.elicit("OpenCode asks: %s" % q.get("question", "?"), schema,
                                    meta={"studio/question": q})
        except studio_mcp.Declined:
            return None
        if res.get("action") == "cancel":
            raise Stopped(q.get("question", ""))
        if res.get("action") != "accept":
            return None
        c = res.get("content") or {}
        picked = [o["label"] for i, o in enumerate(options) if c.get("pick_%d" % i)]
        if isinstance(c.get("answer"), str) and c["answer"]:
            picked.append(c["answer"])
        if isinstance(c.get("other"), str) and c["other"].strip():
            picked.append(c["other"].strip())
        answers.append(picked)
    return answers


class Family:
    """Whether a session is the one we are following or one it spawned (a
    subagent's session asks in its own name). All of them live in `directory`,
    the task's folder."""

    def __init__(self, root, directory=None):
        self.root = root
        self.directory = directory
        self.parent = {}

    def __contains__(self, sid):
        seen = set()
        while sid and sid not in seen:
            if sid == self.root:
                return True
            seen.add(sid)
            if sid not in self.parent:
                try:
                    info = get_json(route("session", sessionID=sid), timeout=10,
                                    directory=self.directory)
                except OpenCodeError:
                    info = {}
                self.parent[sid] = info.get("parentID")
            sid = self.parent[sid]
        return False


class Events:
    """OpenCode's event stream for one folder. It wakes the follower the
    moment anything happens - a step to approve, the session going idle - so
    the follower looks every EVENT_WAIT seconds instead of every POLL. If the
    stream cannot be opened, or drops, `alive` goes false and it polls as
    before: the stream only makes it quicker, never decides anything."""

    QUIET = ("server.connected", "server.heartbeat")

    def __init__(self, directory=None):
        self.flag = threading.Event()
        self.alive = False
        self.last = time.monotonic()      # when the last real event came in
        self.resp = None
        try:
            self.resp = _open(_request(ROUTES["events"], directory=directory), EVENT_TIMEOUT)
            if "event-stream" not in (self.resp.headers.get("Content-Type") or ""):
                raise OpenCodeError("not an event stream")
        except (OpenCodeError, AttributeError):
            self.close()
            return
        self.alive = True
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        try:
            for raw in self.resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                try:
                    kind = json.loads(line[5:]).get("type", "")
                except (ValueError, AttributeError):
                    kind = ""
                if kind not in self.QUIET:
                    self.last = time.monotonic()
                    self.flag.set()
        except Exception:
            pass
        self.alive = False
        self.flag.set()

    def wait(self, seconds):
        if seconds > 0:
            self.flag.wait(seconds)
        self.flag.clear()

    def close(self):
        resp, self.resp = self.resp, None
        if resp is not None:
            try:
                resp.close()
            except Exception:
                pass


# ------------------------------------------------------------- the grants
# "Always allow" is kept here, per task, not given to OpenCode: OpenCode's
# own grant lasts until the server restarts, cannot be seen and cannot be
# taken back. The bridge answers OpenCode "once" every time and remembers
# The user's word itself - in the task record, so it outlives a restart,
# ends with the task, and opencode_grants / opencode_revoke show and drop it.

def grant_label(kind, patterns):
    patterns = [x for x in patterns or [] if isinstance(x, str)]
    if not patterns or patterns == ["*"]:
        return {"edit": "every edit", "bash": "every command"}.get(kind, "every %s" % kind)
    return "%s %s" % ({"edit": "edits to", "bash": "commands matching"}.get(kind, kind),
                      ", ".join(patterns))


def _matches(value, pattern):
    # OpenCode writes a command grant as "git status *": the command, then any
    # arguments or none.
    return (fnmatch.fnmatchcase(value, pattern)
            or (pattern.endswith(" *") and value == pattern[:-2]))


def granted(sid, p):
    """The user's standing grant in task `sid` that covers request `p`, or None."""
    kind = p.get("permission")
    wanted = [x for x in (p.get("patterns") or []) if isinstance(x, str)] or ["*"]
    for g in (task(sid) or {}).get("grants", []):
        if g.get("kind") == kind and all(any(_matches(w, gp) for gp in g.get("patterns", []))
                                         for w in wanted):
            return g
    return None


def add_grant(sid, p):
    kind = p.get("permission") or "?"
    patterns = ([x for x in (p.get("always") or []) if isinstance(x, str)]
                or [x for x in (p.get("patterns") or []) if isinstance(x, str)] or ["*"])
    grants = [g for g in (task(sid) or {}).get("grants", [])
              if not (g.get("kind") == kind and g.get("patterns") == patterns)]
    grants.append({"kind": kind, "patterns": patterns, "label": grant_label(kind, patterns),
                   "since": time.strftime("%Y-%m-%d %H:%M")})
    update_task(sid, grants=grants)


# ------------------------------------------------------------- agentic
# The user's switch for working without a card per step. The OpenCode tab's
# Agentic button writes it to a file in the state folder and it is read here
# on every look, like the password: no tool sets it, so the model that briefs
# OpenCode cannot turn it on. It holds only in a task's own copy, where every
# change can be taken back and nothing reaches the user's folder before they
# merge; a folder worked in place asks at every step, as before.
#
# What goes through: an edit to a file in the copy, and a command that is one
# of AGENTIC_COMMANDS with nothing of the shell's around it. What still asks:
# any other command, a fetch, an add-on's tool, a call repeated in a loop.
# It is a short list, not a sandbox: the tests it lets run are code OpenCode
# wrote, and no one read them first.

AGENTIC_FILE = "agentic.json"
# A program, and what must follow it. Reads and tests only: nothing that
# installs, fetches, deletes or changes history.
AGENTIC_COMMANDS = {
    "python": (("-m", "unittest"), ("-m", "pytest"), ("-m", "py_compile")),
    "pytest": ((),),
    "git": (("status",), ("diff",), ("log",), ("show",)),
    "ls": ((),), "dir": ((),), "cat": ((),), "type": ((),), "head": ((),), "tail": ((),),
    "grep": ((),), "rg": ((),), "findstr": ((),), "wc": ((),), "pwd": ((),),
}
AGENTIC_SAME = {"python3": "python", "py": "python"}
CMD_STYLE = {"dir", "type", "findstr"}        # their switches start with a slash
# Everything of the shell's but && between whole commands: a pipe, a redirect,
# a variable, a second command behind ; or &, and the brace the shell would
# turn into a path this check never saw.
SHELL_CHARS = re.compile(r"[;&|<>`${}\n\r]|%\w+%")
AGENTIC_STATUS = ("The user has Agentic on: in a task's copy, OpenCode's edits and its test "
                  "runs and reads go through without a card, and an ask that ends with "
                  "to-dos open or tests failing goes back to it, up to %d times. Any other "
                  "command, a fetch and every merge still wait for the user, and it is "
                  "refused anything outside its folder." % AGENTIC_ROUNDS)
AGENTIC_NOTE = ("(Agentic is on. You are working by yourself in a private copy of the "
                "project: your edits there and your test runs go through without the user "
                "approving each one, so do not stop to ask whether to go on. Write a to-do "
                "list, make the change, run the tests of what you changed, fix what fails, "
                "and end with which files and functions you changed and what the tests "
                "reported. The user reads the whole change before it is merged.)\n\n")


def agentic():
    """Whether the user has Agentic on."""
    try:
        with open(os.path.join(STATE_DIR, AGENTIC_FILE), encoding="utf-8") as f:
            return json.load(f).get("on") is True
    except (OSError, ValueError, AttributeError):
        return False


def agentic_task(sid):
    """The task of session `sid` when Agentic holds for it - it is on, and the
    task works in a copy of its own - or None."""
    t = task(sid)
    return t if t and t.get("isolated") and t.get("dir") and agentic() else None


def within(path, copy):
    """Whether `path` - absolute, or relative to the copy - is the task's
    copy or something in it."""
    root = os.path.normcase(os.path.realpath(copy))
    full = os.path.normcase(os.path.realpath(os.path.join(root, path)))
    return full == root or full.startswith(root + os.sep)


def in_copy(path, copy):
    """Whether `path` is a file of the task's copy, and not one of git's own."""
    root = os.path.normcase(os.path.realpath(copy))
    full = os.path.normcase(os.path.realpath(os.path.join(root, path)))
    return (full.startswith(root + os.sep)
            and os.path.relpath(full, root).split(os.sep)[0] != ".git")


def _readings(arg):
    """What the shell may make of an argument, quotes off. A backslash is a
    path's in cmd and inside quotes - the model writes cd "C:\\...\\copy" - but
    outside quotes Git Bash drops it and keeps what follows, so `.\\./x` is
    `../x` there: an unquoted argument is read both ways."""
    quote = arg[:1]
    whole = (len(arg) >= 2 and quote in "\"'" and arg.endswith(quote)
             and not re.search("[\"']", arg[1:-1]))
    bare = arg.replace('"', "").replace("'", "")
    return {bare.replace("\\", "/")} | (set() if whole else {re.sub(r"\\(.?)", r"\1", bare)})


def _outside(arg, copy, prog):
    """Whether a command's argument names a place outside the task's copy."""
    for piece in (p for reading in _readings(arg) for p in reading.split("=")):
        if prog in CMD_STYLE and re.fullmatch(r"/[A-Za-z?]([:\-].*)?", piece):
            continue                          # a switch: dir /b, findstr /s
        if piece.startswith("~"):
            return True
        drive = re.fullmatch(r"/([A-Za-z])(/.*)?", piece)     # Git Bash: /c/Users/...
        if drive and os.name == "nt":
            piece = "%s:%s" % (drive.group(1), drive.group(2) or "/")
        if re.match(r"[A-Za-z]:", piece):
            if os.name != "nt" or not within(piece, copy):
                return True
        elif (piece.startswith("/") or ".." in piece.split("/")) and not within(piece, copy):
            return True
    return False


def safe_command(cmd, copy):
    """Whether `cmd` may run in the task's copy without a card: each command
    in it (&& may join several) is a cd or one of AGENTIC_COMMANDS, no
    argument names a place outside the copy, and nothing else of the shell's
    is in it."""
    if (not isinstance(cmd, str) or not cmd.strip()
            or SHELL_CHARS.search(cmd.replace("&&", " "))):
        return False
    for part in cmd.split("&&"):
        try:
            argv = shlex.split(part, posix=False)
        except ValueError:
            return False                      # a quote left open
        if not argv:
            return False
        prog = argv[0].lower()
        prog = prog[:-4] if prog.endswith(".exe") else prog
        prog = AGENTIC_SAME.get(prog, prog)
        args = argv[1:]
        if prog == "cd":
            known = len(args) == 1
        else:
            known = any(tuple(args[:len(s)]) == s for s in AGENTIC_COMMANDS.get(prog, ()))
        if not known or any(_outside(a, copy, prog) for a in args):
            return False
    return True


def agentic_allows(sid, p):
    """What Agentic lets through without a card, in words for the report, or
    None when request `p` is the user's to decide."""
    t = agentic_task(sid)
    if not t:
        return None
    copy = t["dir"]
    kind = p.get("permission")
    meta = p.get("metadata") or {}
    patterns = [x for x in (p.get("patterns") or []) if isinstance(x, str)]
    if kind == "edit":
        files = [meta["filepath"]] if isinstance(meta.get("filepath"), str) else patterns
        if files and all(in_copy(f, copy) for f in files):
            return "edit " + ", ".join(rel(f) for f in files)
    elif kind == "bash":
        # OpenCode lists each command of a line in `patterns`, a leading cd
        # left out; the line as typed (`metadata.command`) has to pass as well.
        whole = meta.get("command")
        cmds = patterns + ([whole] if isinstance(whole, str) else [])
        if cmds and all(safe_command(c, copy) for c in cmds):
            return "run " + (" && ".join(patterns) or whole)
    return None


def todo_list(sid, directory=None):
    """The session's to-do list as [words, status] pairs; empty when it has
    none or cannot be read."""
    try:
        items = get_json(route("todo", sessionID=sid), timeout=5, directory=directory)
    except (OpenCodeError, ValueError):
        return []
    if not isinstance(items, list):
        return []
    return [[str(t.get("content") or "?"), str(t.get("status") or "")]
            for t in items if isinstance(t, dict)]


def open_todos(sid, t):
    """The to-dos this ask left open, in OpenCode's words. A list the ask
    never touched is an earlier ask's, given up with it, and is not counted:
    a session keeps its list until OpenCode writes another."""
    todos = todo_list(sid, t.get("dir"))
    if todos == t.get("todos"):
        return []
    return [words for words, status in todos if status not in ("completed", "cancelled")]


def unfinished(sid, lines, nudge=False):
    """Why an ask that ended is not finished, and the prompt that sends
    OpenCode back to it: (why, prompt), or None when it is finished, Agentic
    does not hold, or it has been sent back AGENTIC_ROUNDS times already.
    `lines` is what `after_ask` said; `nudge` is whether the ask wanted a
    change and none has been made."""
    t = agentic_task(sid)
    if not t or len(t.get("rounds", [])) >= AGENTIC_ROUNDS:
        return None
    failed = [l for l in lines if l.startswith("Tests FAILED")]
    if failed:
        return ("the tests failed",
                "The tests of what you changed failed. Fix the code - change a test only "
                "if the test itself is wrong - then run them again. If a failing test has "
                "nothing to do with your change, leave it alone and say so.\n\n" + failed[0])
    todo = open_todos(sid, t)
    if todo:
        return ("%d to-do(s) were open" % len(todo),
                "You stopped with %d to-do(s) still open:\n%s\nTick off any that are in "
                "fact done, then carry on with the first that is not; do not start over. "
                "When all are done, run the tests of what you changed and say which files "
                "and functions you changed."
                % (len(todo), "\n".join("- " + x[:200] for x in todo[:12])))
    if nudge and "nothing had changed" not in t.get("rounds", []):
        return ("nothing had changed",
                "Nothing in your copy has changed yet. If the task asks for a change, make "
                "it now: edit the file, then run its tests. If no change is needed, say "
                "why in a line or two.")
    return None


def send_back(sid, again):
    """Send OpenCode back to an ask that ended with work left: `again` is
    `unfinished`'s (why, prompt). Counted on the task, so the rounds of one
    ask stay within AGENTIC_ROUNDS over every call that follows it."""
    why, words = again
    update_task(sid, owed=None, rounds=(task(sid) or {}).get("rounds", []) + [why])
    studio_mcp.progress("back to OpenCode: " + why)
    prompt(sid, words)


# Words a request for reading starts with; anything else is taken as a change.
ASKS = frozenset(("what", "whats", "why", "how", "where", "which", "who", "when", "explain",
                  "describe", "list", "show", "tell", "is", "are", "does", "do", "did",
                  "summarize", "summarise", "review", "inspect", "diagnose", "find",
                  "check", "look", "read", "compare"))


def wants_change(text):
    """Whether a task reads as a change to make rather than a question."""
    first = (text.strip().splitlines() or [""])[0].strip().casefold()
    return (bool(first) and not first.endswith("?")
            and re.split(r"[^a-z]+", first.replace("'", ""), 1)[0] not in ASKS)


def settle(family, log):
    """Put every pending request of this session's family to the user - or
    answer it from a grant the user gave, or from Agentic - and return how
    many there were. Raises Stopped when the user stops."""
    n = 0
    where = family.directory
    for p in pending_permissions(where):
        if p.get("sessionID") not in family:
            continue
        n += 1
        headline = describe_permission(p)[0].split("\n")[0]
        what = headline.replace("OpenCode wants to ", "").rstrip(".:")
        g = granted(family.root, p)
        if g:
            reply_permission(p["id"], "once", "", where)
            log.append("allowed by the user's standing grant (%s): %s" % (g.get("label"), what))
            continue
        let = agentic_allows(family.root, p)
        if let:
            reply_permission(p["id"], "once", "", where)
            log.append("allowed by Agentic: %s" % let)
            continue
        (decision, note), asked = ask_permission(p)
        if decision == "always":
            add_grant(family.root, p)
        reply_permission(p["id"], "reject" if decision == "reject" else "once", note, where)
        word = {"once": "allowed", "always": "allowed from now on", "reject": "refused"}[decision]
        log.append("%s: %s%s" % (word, what, (" - note: " + note) if note else ""))
        if not asked:
            raise Stopped("no one could approve: " + headline)
    for q in pending_questions(where):
        if q.get("sessionID") not in family:
            continue
        n += 1
        answers = ask_question(q)
        if answers is None:
            reject_question(q["id"], where)
            log.append("declined to answer OpenCode's question")
        else:
            reply_question(q["id"], answers, where)
            log.append("answered OpenCode: %s" % "; ".join(", ".join(a) for a in answers))
    return n


def todo_count(sid, directory=None):
    """(done, total) of the session's to-do list, or None when it has none -
    the one real fraction there is for a coding task."""
    try:
        items = get_json(route("todo", sessionID=sid), timeout=5, directory=directory)
    except (OpenCodeError, ValueError):
        return None
    items = [t for t in items if isinstance(t, dict)] if isinstance(items, list) else []
    if not items:
        return None
    return (sum(1 for t in items if t.get("status") in ("completed", "cancelled")),
            len(items))


def files_changed(directory):
    """How many files a task's copy has changed since its last checkpoint;
    None for a session in the server's own folder."""
    if not directory:
        return None
    try:
        return len(changed_files(directory))
    except Exception:
        return None


def _span(seconds):
    seconds = int(seconds)
    return "%ds" % seconds if seconds < 60 else "%dm %02ds" % divmod(seconds, 60)


def progress_line(state, todos, files, decided, quiet):
    """The one line the chat shows beside a running ask."""
    parts = ["OpenCode is " + ("working" if state != "idle" else "finishing")]
    if todos:
        done, total = todos
        parts.append("%d/%d to-dos (%d%%)" % (done, total, 100 * done // total))
    if files is not None:
        parts.append("%d file(s) changed" % files)
    if decided:
        parts.append("%d step(s) decided" % decided)
    if quiet >= STALL:
        parts.append("nothing for %s - may be stuck; Stop ends it" % _span(quiet))
    else:
        parts.append("last activity %s ago" % _span(quiet))
    return "; ".join(parts)


class Clock:
    """The seconds of work one call may spend, over every round of it. Time
    The user spends deciding is not work and is never added."""

    def __init__(self, limit):
        self.limit, self.spent = limit, 0.0

    def add(self, seconds):
        """Count `seconds` of work; True once the limit is passed."""
        self.spent += seconds
        return self.spent > self.limit


def follow(sid, seen, clock, log=None):
    """Follow session `sid` until it is idle, putting each step it asks about
    to the user. Returns (state, report): state is "done", "stopped" or
    "working" (the work limit ran out). `seen` is the message ids that were
    there before; what came after is reported. `clock` is the work the call
    has left and `log` the steps decided so far, both shared by the rounds of
    one ask. Time the user spends deciding does not count."""
    directory = task_dir(sid)
    family = Family(sid, directory)
    events = Events(directory)
    log = [] if log is None else log
    started = time.monotonic()
    stopped = None
    idle_polls = 0
    # What progress shows: the to-do list as n/m, the files changed, and how
    # long since anything moved - an event, a to-do ticked, a file written -
    # so a stuck session reads as stuck rather than as endless dots.
    last_act, next_look, snap = started, 0.0, None
    said, say_by = None, 0.0
    try:
        while True:
            if studio_mcp.cancelled():
                raise Stopped("the user pressed Stop")
            tick = time.monotonic()
            asked = settle(family, log)
            if asked:
                idle_polls = 0
                last_act = time.monotonic()  # the user's time is not a stall
                continue                    # decided; look again before sleeping
            state = (statuses(directory).get(sid) or {}).get("type", "idle")
            if state == "idle":
                idle_polls += 1
                # A task just handed over may not be marked busy yet.
                if idle_polls >= 2 and time.monotonic() - started > SETTLE:
                    break
            else:
                idle_polls = 0
            now = time.monotonic()
            if now >= next_look:
                next_look = now + LOOK
                look = (todo_count(sid, directory), files_changed(directory))
                if look != snap:
                    snap, last_act = look, now
            last_act = max(last_act, events.last)
            todos, files = snap
            # Said when it changes, and every LOOK seconds so the client sees
            # the bridge alive. Every pass used to say it: five lines a second
            # while OpenCode streamed (765 in one live run of 141 s).
            line = progress_line(state, todos, files, len(log), now - last_act)
            if line != said or now >= say_by:
                said, say_by = line, now + LOOK
                studio_mcp.progress(line, *(todos or (None, None)))
            events.wait(max(POLL, EVENT_WAIT) if events.alive else POLL)
            if clock.add(time.monotonic() - tick):
                return "working", report(sid, seen, log)
    except Stopped as e:
        stopped = str(e)
        try:
            abort(sid)
        except OpenCodeError:
            pass
    finally:
        events.close()
    text = report(sid, seen, log)
    if stopped:
        return "stopped", "(%s)\n%s" % (stopped.split("\n")[0], text)
    return "done", text


def outcome(sid, state, text, work_limit):
    """A follow's (state, report) as the tool result the model reads. The
    state and session ride in `_meta` for a client that follows on by itself
    (the tab's Direct mode)."""
    if state == "working":
        res = result(
            "OpenCode is still working on session %s after %d seconds of work. Do "
            "not send the task again: call opencode_wait with this session_id to "
            "keep following it, or opencode_abort to stop it.\n%s"
            % (sid, work_limit, text), error=True)
    elif state == "unfinished":
        why, _, rest = text.partition("\n")
        res = result(
            "Session %s has work left (%s) and this call's %d seconds of work are "
            "used. Do not send the task again: call opencode_wait with this "
            "session_id to send OpenCode back to it, or tell the user where it "
            "stands.\n%s" % (sid, why, work_limit, rest), error=True)
    elif state == "stopped":
        why, _, rest = text.partition("\n")
        res = result("The user stopped OpenCode %s. Session %s is halted; what it had "
                     "already changed stays until it is undone or discarded.\n%s"
                     % (why, sid, rest), error=True)
    else:
        res = result("Session %s is done.\n%s" % (sid, text))
    res["_meta"] = {"studio/opencode": {"session": sid, "state": state}}
    return res


def run(sid, seen, work_limit):
    """follow() as a tool result."""
    state, text = follow(sid, seen, Clock(work_limit))
    return outcome(sid, state, text, work_limit)


def report(sid, seen, log):
    """What happened since `seen`: every step the user decided, then what
    OpenCode said and did, newest last, within MAX_REPLY_CHARS."""
    out = []
    if log:
        shown = log if len(log) <= 30 else ["... %d earlier decision(s)" % (len(log) - 30)] + log[-30:]
        decisions = "The user's decisions:\n" + "\n".join("- " + l for l in shown)
        out.append(decisions[:MAX_REPLY_CHARS // 2])
    chunks, error, msgs = [], None, []
    try:
        msgs = messages(sid)
        for m in msgs:
            if _info(m).get("id") in seen or _role(m) != "assistant":
                continue
            text = summarize(m, 2500)
            if text:
                chunks.append(text)
            err = _info(m).get("error")
            if err:
                error = _explain(err)
    except OpenCodeError as e:
        chunks.append("(could not read the session: %s)" % e)
    body = "\n".join(chunks) or "(OpenCode said nothing)"
    room = max(1500, MAX_REPLY_CHARS - sum(len(x) for x in out) - 200)
    if len(body) > room:
        body = "... [earlier steps cut]\n" + body[-room:]
    out.append("OpenCode:\n" + body)
    if error:
        out.append("OpenCode reported an error: " + error)
    ctx = context_report(msgs)
    if ctx:
        out.append(ctx)
    out.append(workspace_note(sid))
    return "\n\n".join(out)


# ------------------------------------------------------------ the context

def context_used(msgs):
    """(tokens the last request filled, times OpenCode compacted) for a
    session's messages. OpenCode records each answer's token counts; a
    compaction leaves a summary message or a compaction part."""
    used, compactions = None, 0
    for m in msgs:
        info = _info(m)
        if info.get("summary") is True or any(p.get("type") == "compaction" for p in _parts(m)):
            compactions += 1
        t = info.get("tokens")
        if info.get("role") == "assistant" and isinstance(t, dict):
            cache = t.get("cache") if isinstance(t.get("cache"), dict) else {}
            n = sum(x for x in (t.get("input"), t.get("output"), cache.get("read"),
                                cache.get("write")) if isinstance(x, int))
            if n:
                used = n
    return used, compactions


def context_report(msgs):
    """How full OpenCode's window is, and whether it has compacted - said,
    because a compaction silently drops the details of the task."""
    used, compactions = context_used(msgs)
    window = loaded_window()
    lines = []
    if used:
        if window:
            pct = 100 * used // window
            lines.append("Context: the last request filled about %d of %d tokens (%d%%)."
                         % (used, window, pct))
            if pct >= CONTEXT_HIGH:
                lines.append("That is close to full: OpenCode will soon compact and lose "
                             "details. Start a new session for the next unrelated step.")
        else:
            lines.append("Context: the last request filled about %d tokens." % used)
    if compactions:
        lines.append("OpenCode compacted this session %d time(s): its earlier history is a "
                     "summary now, so it may have lost details of the task. Repeat the goal, "
                     "the files and what is done when you continue it." % compactions)
    return "\n".join(lines)


# ------------------------------------------------------------- workspace

def inside(relpath, base=None):
    """The absolute path of `relpath` under the workspace - or `base`, a
    task's copy - or an error if it would leave it. OpenCode is refused
    anything outside; so is this tab."""
    if not isinstance(relpath, str):
        raise ValueError("path must be a string")
    r = relpath.replace("\\", "/").lstrip("/")
    root = os.path.realpath(base or WORKSPACE)
    full = os.path.realpath(os.path.join(root, r))
    if full != root and not full.startswith(root + os.sep):
        raise OpenCodeError("%r is outside OpenCode's folder, %s." % (relpath, WORKSPACE))
    return full


def list_files(relpath="", limit=MAX_LIST, base=None):
    root = inside(relpath, base)
    if not os.path.isdir(root):
        raise OpenCodeError("%s is not a folder in the workspace." % (relpath or "/"))
    out = []
    deadline = time.monotonic() + 3
    for folder, dirs, files in os.walk(root):
        if time.monotonic() >= deadline:
            return out, True
        dirs[:] = sorted(d for d in dirs if d.casefold() not in SKIP_DIRS)
        for f in sorted(files):
            p = os.path.join(folder, f)
            r = os.path.relpath(p, os.path.realpath(base or WORKSPACE)).replace(os.sep, "/")
            try:
                out.append("%s  (%d bytes)" % (r, os.path.getsize(p)))
            except OSError:
                out.append(r)
            if len(out) >= limit:
                return out, True
    return out, False


# ------------------------------------------------------------------ tasks
# A session is a task, and a task in a git repository gets a copy of its
# own: a git worktree on a branch of its own, under STATE_DIR. OpenCode
# works there, so the user's folder - and the app running from it, when the
# folder is this repo - is untouched until the user merges the task. Each
# ask ends with the tests of what it changed run in that copy, then a
# checkpoint commit on the task's branch, so the last ask can be undone.
# A merge is one squashed commit, so a merged task can be reverted.
# Merging, undoing and discarding each ask the user first, like every edit.
# A folder that is not a git repository gets no copy: OpenCode works in it
# directly, as before.

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
CHECKPOINT_ID = ("-c", "user.name=OpenCode", "-c", "user.email=opencode@localhost")


def _git(*args, cwd=None, timeout=120, check=True):
    try:
        p = subprocess.run(("git",) + args, cwd=cwd or WORKSPACE, capture_output=True,
                           stdin=subprocess.DEVNULL, timeout=timeout, creationflags=NO_WINDOW)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise OpenCodeError("git %s failed: %s" % (_verb(args), e))
    out = p.stdout.decode("utf-8", "replace")
    if not check:
        return p.returncode, out
    if p.returncode:
        err = (p.stderr.decode("utf-8", "replace") or out).strip()
        raise OpenCodeError("git %s failed: %s" % (_verb(args), err[:600]))
    return out


def _verb(args):
    return next((a for a in args if a not in CHECKPOINT_ID), "?")


def worktrees_dir():
    return os.path.join(STATE_DIR, "worktrees")


def _tasks_file():
    return os.path.join(STATE_DIR, "tasks.json")


def load_tasks():
    try:
        with open(_tasks_file(), encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_tasks(tasks):
    os.makedirs(STATE_DIR, exist_ok=True)
    tmp = _tasks_file() + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=1)
    os.replace(tmp, _tasks_file())


def task(sid):
    return load_tasks().get(sid) if sid else None


def update_task(sid, **fields):
    tasks = load_tasks()
    tasks.setdefault(sid, {}).update(fields)
    save_tasks(tasks)
    return tasks[sid]


def task_dir(sid):
    """The folder session `sid` works in, or None for the server's own."""
    t = task(sid)
    return t.get("dir") if t and t.get("isolated") else None


def is_repo():
    """Whether the workspace is the top of a git repository with a commit."""
    code, top = _git("rev-parse", "--show-toplevel", check=False)
    if code or not top.strip():
        return False
    if os.path.normcase(os.path.realpath(top.strip())) != os.path.normcase(
            os.path.realpath(WORKSPACE)):
        return False
    return _git("rev-parse", "--verify", "-q", "HEAD", check=False)[0] == 0


def start_task(title):
    """A new session for a new task, in a copy of its own when it can have one."""
    if not is_repo():
        s = new_session(title)
        update_task(s["id"], title=title, isolated=False, grants=[],
                    created=time.strftime("%Y-%m-%d %H:%M"))
        return s["id"]
    name = time.strftime("%Y%m%d-%H%M%S") + "-" + os.urandom(2).hex()
    copy = os.path.join(worktrees_dir(), name)
    branch = "opencode/" + name
    base = _git("rev-parse", "HEAD").strip()
    os.makedirs(worktrees_dir(), exist_ok=True)
    _git("worktree", "add", "-q", "-b", branch, copy, base)
    try:
        s = new_session(title, directory=copy)
    except OpenCodeError:
        _drop_copy(copy, branch)
        raise
    update_task(s["id"], title=title, isolated=True, dir=copy, branch=branch, base=base,
                checkpoints=[], grants=[], created=time.strftime("%Y-%m-%d %H:%M"))
    return s["id"]


def _drop_copy(copy, branch):
    _git("worktree", "remove", "--force", copy, check=False)
    _git("worktree", "prune", check=False)
    _git("branch", "-D", branch, check=False)


def changed_files(copy):
    """Paths changed in a task's copy since its last checkpoint."""
    out = _git("status", "--porcelain", "-uall", cwd=copy)
    files = []
    for line in out.splitlines():
        name = line[3:].strip()
        if " -> " in name:
            name = name.split(" -> ", 1)[1]
        files.append(name.strip('"'))
    return files


def checkpoint(sid, label):
    """Commit what the last ask changed on the task's branch; its sha, or None."""
    t = task(sid)
    if not t or not t.get("isolated"):
        return None
    copy = t["dir"]
    _git("add", "-A", cwd=copy)
    if _git("diff", "--cached", "--quiet", cwd=copy, check=False)[0] == 0:
        return None
    _git(*(CHECKPOINT_ID + ("commit", "-q", "-m", "checkpoint: " + label[:72])), cwd=copy)
    sha = _git("rev-parse", "HEAD", cwd=copy).strip()
    update_task(sid, checkpoints=t.get("checkpoints", []) + [{"sha": sha, "label": label[:120]}])
    return sha


# An app package whose tests are named after something else.
TEST_ALIASES = {"comfyui": "comfy", "image_studio": "imagegen"}
# Module names every app has; apps/<app>/mcp.py is tested by test_<app>, not test_mcp.
GENERIC_MODULES = {"mcp", "ui", "view"}


def tests_for(root, files):
    """The test modules that cover `files`: a changed test file itself, and
    tests/test_<name>.py for a changed module (studio_ and _mcp/_ui dropped).
    In apps/<app>/ the app's name is tried too: apps/opencode/mcp.py ->
    test_opencode, apps/comfyui/view.py -> test_comfy_view."""
    mods = []
    for f in files:
        if not f.endswith(".py"):
            continue
        name = os.path.basename(f)[:-3]
        parts = f.replace("\\", "/").split("/")
        app = parts[-2] if len(parts) >= 3 and parts[-3] == "apps" else None
        if name.startswith("test_"):
            candidates = [name]
        elif app:
            a = TEST_ALIASES.get(app, app)
            base = re.sub(r"_(mcp|ui)$", "", name)
            candidates = [] if name in GENERIC_MODULES else ["test_" + name]
            candidates += ["test_%s_%s" % (a, name), "test_%s_%s" % (a, base), "test_" + a]
        else:
            stem = name[7:] if name.startswith("studio_") else name
            candidates = ["test_" + stem, "test_" + re.sub(r"_(mcp|ui)$", "", stem)]
        for c in candidates:
            if os.path.isfile(os.path.join(root, "tests", c + ".py")):
                if "tests." + c not in mods:
                    mods.append("tests." + c)
                break
    return mods


def run_tests(root, mods):
    """Run `mods` in `root` and say how it went, in a few lines. It reports
    that it is still at it every TEST_LOOK seconds - a run longer than the
    client's patience is not a hung bridge - and ends with the user's Stop."""
    try:
        p = subprocess.Popen([sys.executable, "-m", "unittest"] + mods, cwd=root,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL,
                             env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
                             creationflags=NO_WINDOW)
    except OSError as e:
        return "Tests could not be run: %s" % e
    started = time.monotonic()
    while True:
        try:
            out = p.communicate(timeout=TEST_LOOK)[0]
            break
        except subprocess.TimeoutExpired:
            spent = time.monotonic() - started
            stop = studio_mcp.cancelled()
            if stop or spent > TEST_TIMEOUT:
                p.kill()
                p.communicate()
                if stop:
                    raise Stopped("the user pressed Stop")
                return "Tests %s did not finish in %d seconds." % (", ".join(mods),
                                                                   TEST_TIMEOUT)
            studio_mcp.progress("running the tests of what OpenCode changed (%s)"
                                % _span(spent))
    out = out.decode("utf-8", "replace").strip()
    ran = re.search(r"Ran (\d+) tests?", out)
    count = ran.group(1) if ran else "?"
    if p.returncode == 0:
        return "Tests passed: %s (%s tests)." % (", ".join(mods), count)
    return "Tests FAILED: %s (%s tests). Their output ends:\n%s" % (
        ", ".join(mods), count, out[-2500:])


def after_ask(sid, label):
    """What follows a finished ask in a task's copy: the tests of what it
    changed, then a checkpoint. Lines for the report."""
    t = task(sid)
    if not t or not t.get("isolated"):
        return []
    lines = []
    try:
        files = changed_files(t["dir"])
        if files:
            mods = tests_for(t["dir"], files)
            if mods:
                studio_mcp.progress("running the tests of what OpenCode changed")
                lines.append(run_tests(t["dir"], mods))
            else:
                lines.append("No test module matches the changed files, so none were run.")
            sha = checkpoint(sid, label)
            if sha:
                lines.append("Checkpoint %s saved on %s." % (sha[:8], t["branch"]))
        if t.get("undone"):
            update_task(sid, undone=False)
    except OpenCodeError as e:
        lines.append("Could not check the task's copy: %s" % e)
    return lines


def confirm(message, yes, detail=None, diff=None):
    """The user's yes to a step that changes their folder or drops work.
    False when they say no, or when no one can be asked."""
    shown = {"kind": "confirm"}
    if diff:
        shown["diff"] = diff
    elif detail:
        shown["command"] = detail
    schema = {"type": "object",
              "properties": {"decision": {"type": "string", "title": "Decision",
                                          "enum": ["yes", "no"], "enumNames": [yes, "Cancel"]}},
              "required": ["decision"]}
    try:
        res = studio_mcp.elicit(message, schema, meta={"studio/approval": shown})
    except studio_mcp.Declined:
        return False
    return (res.get("action") == "accept"
            and (res.get("content") or {}).get("decision") == "yes")


def _need_task(a):
    sid = a.get("session_id") or last_session()
    t = task(sid)
    if not sid or not t:
        raise OpenCodeError("No task to act on: give the session_id of one opencode_ask started.")
    if not t.get("isolated"):
        raise OpenCodeError("Session %s works in %s directly, not in a copy, so there is "
                            "nothing to merge, undo or discard." % (sid, WORKSPACE))
    return sid, t


def _idle_or_refuse(sid, t):
    if (statuses(t["dir"]).get(sid) or {}).get("type", "idle") != "idle":
        raise OpenCodeError("Session %s is still working. opencode_wait for it, or "
                            "opencode_abort it, first." % sid)


def t_merge(a):
    sid, t = _need_task(a)
    if t.get("merged"):
        return result("Session %s was already merged as %s." % (sid, t["merged"][:8]))
    _idle_or_refuse(sid, t)
    checkpoint(sid, "before merge")
    stat = _git("diff", "--stat", t["base"], "HEAD", cwd=t["dir"]).strip()
    if not stat:
        return result("Session %s changed nothing, so there is nothing to merge." % sid)
    files = [l.split("|")[0].strip() for l in stat.splitlines() if "|" in l]
    dirty = _git("status", "--porcelain", "--", *files).strip() if files else ""
    if dirty:
        return result("The user's folder has uncommitted changes in files this task also "
                      "changed, so it was not merged:\n%s\nThe user commits or sets those "
                      "aside first." % dirty, error=True)
    here = _git("rev-parse", "--abbrev-ref", "HEAD").strip()
    diff = _git("diff", t["base"], "HEAD", cwd=t["dir"])
    message = (a.get("message") or "OpenCode: %s" % (t.get("title") or sid)).strip()
    if not confirm("Merge OpenCode's work on \"%s\" into %s of %s, as one commit?\n%s"
                   % (t.get("title") or sid, here, WORKSPACE, stat),
                   "Merge", diff=diff[:60000]):
        return result("The user did not merge session %s; its work stays in its copy." % sid)
    code, out = _git("merge", "--squash", t["branch"], check=False)
    if code:
        _git("reset", "--merge", check=False)
        return result("The merge conflicted with work in %s, so nothing was changed:\n%s\n"
                      "OpenCode can bring the task up to date, or the user merges by hand "
                      "(branch %s)." % (here, out.strip()[-1500:], t["branch"]), error=True)
    _git("commit", "-q", "-m", message)
    sha = _git("rev-parse", "HEAD").strip()
    _drop_copy(t["dir"], t["branch"])
    update_task(sid, merged=sha, isolated=False, grants=[])
    return result("Merged session %s into %s as %s (%s). Its copy is removed; "
                  "opencode_undo reverts the merge." % (sid, here, sha[:8], message))


def t_undo(a):
    sid = a.get("session_id") or last_session()
    t = task(sid)
    if t and t.get("merged") and not t.get("isolated"):
        sha = t["merged"]
        stat = _git("show", "--stat", "--format=%s", sha).strip()
        if not confirm("Revert the merged OpenCode task %s in %s?\n%s" % (sha[:8], WORKSPACE, stat),
                       "Revert"):
            return result("The user kept the merge.")
        code, out = _git("revert", "--no-edit", sha, check=False)
        if code:
            _git("revert", "--abort", check=False)
            return result("Reverting %s conflicts with later work, so nothing changed:\n%s"
                          % (sha[:8], out.strip()[-1500:]), error=True)
        update_task(sid, merged=None)
        return result("Reverted the merge %s; the revert is commit %s."
                      % (sha[:8], _git("rev-parse", "--short", "HEAD").strip()))
    sid, t = _need_task(a)
    _idle_or_refuse(sid, t)
    checkpoint(sid, "last ask")
    t = task(sid)
    points = t.get("checkpoints", [])
    if not points:
        return result("Session %s has changed nothing yet; there is nothing to undo." % sid)
    back_to = points[-2]["sha"] if len(points) > 1 else t["base"]
    stat = _git("diff", "--stat", back_to, "HEAD", cwd=t["dir"]).strip()
    if not confirm("Undo OpenCode's last change in this task (%s)?\n%s"
                   % (points[-1]["label"], stat), "Undo"):
        return result("The user kept the change.")
    _git("reset", "-q", "--hard", back_to, cwd=t["dir"])
    _git("clean", "-q", "-fd", cwd=t["dir"])
    update_task(sid, checkpoints=points[:-1], undone=True)
    return result("Undid the last change in session %s; its copy is back at %s. OpenCode is "
                  "told on the next ask that it was undone." % (sid, back_to[:8]))


def t_discard(a):
    sid, t = _need_task(a)
    stat = _git("diff", "--stat", t["base"], cwd=t["dir"], check=False)[1].strip()
    if not confirm("Throw away OpenCode's task \"%s\" and its copy?%s"
                   % (t.get("title") or sid, ("\n" + stat) if stat else " It changed nothing."),
                   "Discard"):
        return result("The user kept the task.")
    try:
        abort(sid)
    except OpenCodeError:
        pass
    _drop_copy(t["dir"], t["branch"])
    tasks = load_tasks()
    tasks.pop(sid, None)
    save_tasks(tasks)
    if last_session() == sid:
        remember_session("")
    return result("Discarded session %s: its copy and branch are gone; the user's folder "
                  "was never touched." % sid)


def t_grants(a):
    sid = a.get("session_id") or last_session()
    grants = (task(sid) or {}).get("grants", [])
    # Agentic is not a grant: the user's switch, not theirs to revoke from here.
    also = ("\n" + AGENTIC_STATUS) if agentic_task(sid) else ""
    if not grants:
        return result("Session %s has no standing grants%s%s"
                      % (sid or "-", "." if also else ": every step is asked.", also))
    return result("Standing grants in session %s (allowed without asking):\n%s\n"
                  "opencode_revoke takes one back.%s" % (sid, "\n".join(
                      "%d. %s (since %s)" % (i + 1, g.get("label"), g.get("since", "?"))
                      for i, g in enumerate(grants)), also))


def t_revoke(a):
    sid = a.get("session_id") or last_session()
    t = task(sid)
    grants = (t or {}).get("grants", [])
    if not grants:
        return result("Session %s has no standing grants." % (sid or "-"))
    n = a.get("number")
    if n is None:
        update_task(sid, grants=[])
        return result("Took back all %d grant(s) in session %s; every step is asked again."
                      % (len(grants), sid))
    if not 1 <= int(n) <= len(grants):
        raise ValueError("number must be 1 to %d" % len(grants))
    gone = grants.pop(int(n) - 1)
    update_task(sid, grants=grants)
    return result("Took back \"%s\" in session %s." % (gone.get("label"), sid))


# ----------------------------------------------------------------- the tools

def result(text, error=False):
    res = {"content": [{"type": "text", "text": text}]}
    if error:
        res["isError"] = True
    return res


MIN_CONTEXT = 65536


def window_of(conf):
    """The context window OpenCode's config declares for its model, or None."""
    model = (conf.get("model") or "").split("/", 1)[-1]
    models = conf.get("provider", {}).get("lmstudio", {}).get("models", {})
    ctx = (models.get(model) or {}).get("limit", {}).get("context")
    return ctx if isinstance(ctx, int) and ctx > 0 else None


def _loaded_conf():
    try:
        with open(os.path.join(STATE_DIR, "opencode.json"), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def loaded_window():
    return window_of(_loaded_conf())


def context_note(conf):
    """A warning when the model is loaded with too small a window to code in
    this repo, or "". OpenCode's own prompt and tools take ~10k tokens and one
    of our larger files as much again; below this it compacts away the task."""
    ctx = window_of(conf)
    if not ctx or ctx >= MIN_CONTEXT:
        return ""
    return ("Warning: the model is loaded with a %d-token context window. OpenCode needs "
            "at least %d to keep a task in mind: in LM Studio on the LLM PC, raise the "
            "model's Context Length (or `lms load <model> --context-length %d`), then "
            "close and reopen Studio Assist." % (ctx, MIN_CONTEXT, MIN_CONTEXT))


def t_status(a):
    lines = []
    try:
        version, note = health()
        lines.append("OpenCode %s is running at %s%s."
                     % (version, OPENCODE_URL, (" (" + note + ")") if note else ""))
        missing = unknown_routes()
        if missing:
            lines.append("This server does not list the route(s) %s in its API document; "
                         "the bridge may need updating for this OpenCode version."
                         % ", ".join(missing))
        try:
            lines.append("%d session(s) so far." % len(sessions()))
            waiting = pending_permissions()
            if waiting:
                lines.append("%d step(s) are waiting for the user's approval." % len(waiting))
        except OpenCodeError as e:
            lines.append("Could not list sessions: %s" % e)
    except OpenCodeError as e:
        lines.append(str(e))
    lines.append(workspace_note())
    if is_repo():
        lines.append("Each new task gets its own copy of the folder (a git worktree), from "
                     "its last commit; the user merges it when it is right.")
    sid = last_session()
    if task(sid):
        grants = task(sid).get("grants", [])
        lines.append("The current task is session %s%s." % (
            sid, "; standing grants: " + "; ".join(g.get("label", "?") for g in grants)
            if grants else ""))
    cfg = os.path.join(STATE_DIR, "opencode.json")
    if os.path.isfile(cfg):
        try:
            with open(cfg, encoding="utf-8") as f:
                conf = json.load(f)
            model = conf.get("model")
            if model:
                lines.append("It codes with the model %s." % model)
            note = context_note(conf)
            if note:
                lines.append(note)
        except (OSError, ValueError):
            pass
    lines.append(AGENTIC_STATUS if agentic() and is_repo() else
                 "It reads freely; every edit, command and fetch waits for the user's "
                 "approval, and it is refused anything outside its folder.")
    return result("\n".join(lines))


def t_list_sessions(a):
    rows = sessions()
    if not rows:
        return result("No sessions yet. opencode_ask with no session_id starts one.")
    lines = []
    for s in rows[-50:]:
        when = (s.get("time") or {}).get("updated") or s.get("updated") or ""
        if isinstance(when, (int, float)):
            when = time.strftime("%Y-%m-%d %H:%M",
                                 time.localtime(when / 1000 if when > 1e11 else when))
        t = task(s.get("id"))
        state = ""
        if t:
            state = ("  [merged %s]" % t["merged"][:8] if t.get("merged") else
                     "  [in its copy, %d checkpoint(s), not merged]" % len(t.get("checkpoints", []))
                     if t.get("isolated") else "")
        lines.append("%s  %s  %s%s" % (s.get("id"), (s.get("title") or "(untitled)")[:60],
                                       when, state))
    return result("\n".join(lines))


def t_new_session(a):
    title = a.get("title") or ""
    sid = start_task(title)
    remember_session(sid)
    return result("Session %s created%s. Send work to it with opencode_ask.\n%s"
                  % (sid, (" - " + title) if title else "", workspace_note(sid)))


def _work_limit(a):
    return max(10, min(int(a.get("timeout") or DEFAULT_WORK), MAX_WORK))


def _last_path():
    return os.path.join(STATE_DIR, "last_session")


def last_session():
    """The session the last ask used, or "". Kept on disk so a follow-up
    continues the work even when the model forgets to pass the id back -
    a local model often does, and OpenCode then starts over knowing nothing."""
    try:
        with open(_last_path(), encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return ""


def remember_session(sid):
    try:
        os.makedirs(STATE_DIR, exist_ok=True)
        with open(_last_path(), "w", encoding="utf-8") as f:
            f.write(sid)
    except OSError:
        pass


def t_ask(a):
    text = a["prompt"]
    if not isinstance(text, str) or not text.strip():
        raise ValueError("prompt must be a non-empty string")
    sid = a.get("session_id") or ("" if a.get("new_session") else last_session())
    seen = None
    if sid:
        try:
            seen = {_info(m).get("id") for m in messages(sid)}
        except OpenCodeError:
            if a.get("session_id"):
                raise
            seen = None                        # the remembered one is gone
    if seen is not None and (statuses(task_dir(sid)).get(sid) or {}).get("type", "idle") != "idle":
        # A second task on a working session queues behind the first and the
        # report mixes the two; the last ask most likely timed out.
        return result("Session %s is still working on the last task. Do not send a new one: "
                      "call opencode_wait with this session_id to follow it, or opencode_abort "
                      "to stop it first." % sid, error=True)
    if seen is None:
        sid = start_task(text.strip().splitlines()[0][:60])
        seen = set()
    remember_session(sid)
    if (task(sid) or {}).get("undone"):
        text = ("(The user undid your last change in this task: those files are back as "
                "they were before it. Do not assume it is there.)\n\n" + text)
    if agentic_task(sid):
        # A new ask: its rounds start over, and the to-dos it finds are not its own.
        update_task(sid, rounds=[], owed=None, todos=todo_list(sid, task_dir(sid)))
        text = AGENTIC_NOTE + text
    prompt(sid, text)
    return finish(sid, seen, _work_limit(a), a["prompt"].strip().splitlines()[0][:80],
                  change=wants_change(a["prompt"]))


def finish(sid, seen, work_limit, label, change=False):
    """Follow an ask to its end; when it ends done, test and checkpoint what
    it changed. With Agentic on, an ask that ended with work left goes back
    to OpenCode (`unfinished`) - within this call's one work limit, and the
    count of rounds is the task's, so opencode_wait carries on where an ask
    ran out. `change` is whether the ask wanted one made. The result the
    model reads."""
    clock, log, extra = Clock(work_limit), [], []
    before = len((task(sid) or {}).get("checkpoints", []))
    while True:
        state, text = follow(sid, seen, clock, log)
        if state != "done":
            extra = []                        # an earlier round's tests are not this one's
            break
        tick = time.monotonic()
        try:
            extra = after_ask(sid, label)
        except Stopped as e:
            state, text = "stopped", "(%s)\n%s" % (e, text)
            break
        t = task(sid) or {}
        again = unfinished(sid, extra, change and len(t.get("checkpoints", [])) == before)
        if not again:
            break
        if clock.add(time.monotonic() - tick):
            # No work left in this call to follow another round: it is owed,
            # and opencode_wait sends it (the tests will not fail again by
            # themselves - what they ran is checkpointed).
            update_task(sid, owed=list(again))
            state, text = "unfinished", "%s\n%s" % (again[0], text)
            break
        send_back(sid, again)
    out = outcome(sid, state, text, work_limit)
    rounds = (task(sid) or {}).get("rounds", [])
    if rounds and agentic_task(sid):
        extra.append("Agentic sent OpenCode back to work %d time(s) by itself: %s.%s"
                     % (len(rounds), "; ".join(rounds),
                        " That is as often as it does; what is left is the user's call."
                        if len(rounds) >= AGENTIC_ROUNDS else ""))
    note = context_note(_loaded_conf())
    if note:
        extra.append(note)
    if extra:
        out.setdefault("content", []).append({"type": "text", "text": "\n".join(extra)})
    return out


def t_wait(a):
    sid = a["session_id"]
    msgs = messages(sid)
    # Report from the last thing the user asked, so a wait after a timed-out
    # ask returns the whole answer rather than only what came since.
    last_user = max((i for i, m in enumerate(msgs) if _role(m) == "user"), default=-1)
    seen = {_info(m).get("id") for m in msgs[:last_user + 1]}
    # A round the last call had no work left for is sent now, if Agentic still
    # holds and the session is at rest.
    t = agentic_task(sid)
    if (t and t.get("owed") and len(t.get("rounds", [])) < AGENTIC_ROUNDS
            and (statuses(t["dir"]).get(sid) or {}).get("type", "idle") == "idle"):
        send_back(sid, t["owed"])
    return finish(sid, seen, _work_limit(a), "waited-for work")


def t_get_session(a):
    sid = a["session_id"]
    limit = int(a.get("limit") or 6)
    rows = messages(sid)
    if not rows:
        return result("Session %s has no messages." % sid)
    chunks = []
    for m in rows[-limit:]:
        chunks.append("%s:\n%s" % (_role(m).upper(), summarize(m, 2500) or "(empty)"))
    return result("\n\n".join(chunks))


def t_abort(a):
    abort(a["session_id"])
    return result("Asked OpenCode to stop session %s. Whatever it had already changed "
                  "stays." % a["session_id"])


def _base(a):
    """The folder the file tools look in: the current task's copy while it
    has one, since that is where OpenCode's work is; else the workspace."""
    return task_dir(a.get("session_id") or last_session())


def t_changes(a):
    """What the current task changed in its copy, or else the uncommitted
    changes in the workspace, from git through OpenCode."""
    sid = a.get("session_id") or last_session()
    t = task(sid)
    if t and t.get("isolated"):
        return task_changes(sid, t, a.get("path"))
    rows = get_json(ROUTES["diff"] + "?mode=git", timeout=30)
    rows = rows if isinstance(rows, list) else []
    want = a.get("path")
    if want:
        want = rel(inside(want))
        rows = [r for r in rows if rel(r.get("file") or r.get("path") or "") == want]
        if not rows:
            return result("%s has no uncommitted changes." % want)
        patch = "\n".join((r.get("patch") or r.get("diff") or "") for r in rows)
        if len(patch) > MAX_DIFF_CHARS:
            patch = patch[:MAX_DIFF_CHARS] + "\n... [diff cut at %d characters]" % MAX_DIFF_CHARS
        return result(patch or "(no textual diff)")
    if not rows:
        return result("No uncommitted changes in %s." % WORKSPACE)
    lines = ["%s  %s  +%s -%s" % (r.get("status", "?"), rel(r.get("file") or r.get("path") or "?"),
                                  r.get("additions", "?"), r.get("deletions", "?"))
             for r in rows[:200]]
    return result("Uncommitted changes (git):\n" + "\n".join(lines))


def task_changes(sid, t, want=None):
    copy = t["dir"]
    _git("add", "-A", cwd=copy)
    if want:
        want = rel(inside(want, copy))
        patch = _git("diff", "--cached", t["base"], "--", want, cwd=copy)
        if not patch.strip():
            return result("%s is unchanged in session %s." % (want, sid))
        if len(patch) > MAX_DIFF_CHARS:
            patch = patch[:MAX_DIFF_CHARS] + "\n... [diff cut at %d characters]" % MAX_DIFF_CHARS
        return result(patch)
    rows = _git("diff", "--cached", "--numstat", t["base"], cwd=copy).strip().splitlines()
    if not rows:
        return result("Session %s has changed nothing yet." % sid)
    lines = []
    for row in rows[:200]:
        added, removed, name = (row.split("\t", 2) + ["", "", ""])[:3]
        lines.append("%s  +%s -%s" % (name, added, removed))
    return result("What session %s changed in its copy (not merged yet):\n%s"
                  % (sid, "\n".join(lines)))


def t_list_files(a):
    rows, truncated = list_files(a.get("path") or "", base=_base(a))
    if not rows:
        return result("Listing stopped early; narrow the path." if truncated else
                      "The folder is empty. %s" % workspace_note())
    kept, size = [], 0
    for row in rows:
        if size + len(row) + 1 > MAX_FILE_CHARS:
            truncated = True
            break
        kept.append(row)
        size += len(row) + 1
    text = "\n".join(kept)
    if truncated:
        text += "\n... more; list a subfolder."
    return result(text)


def t_read_file(a):
    full = inside(a["path"], _base(a))
    if not os.path.isfile(full):
        raise OpenCodeError("%s is not a file in the workspace." % a["path"])
    start, limit = a.get("start", 0), a.get("limit", MAX_FILE_CHARS)
    with open(full, encoding="utf-8", errors="replace") as f:
        # Character offsets, not byte seeks: UTF-8 and Windows newlines must
        # paginate exactly as they appear in the returned text.
        remaining = start
        while remaining:
            chunk = f.read(min(remaining, 65536))
            if not chunk:
                break
            remaining -= len(chunk)
        data = f.read(limit + 1)
    more = len(data) > limit
    data = data[:limit]
    header = ("Next page: opencode_read_file with the same path and start=%d.\n"
              % (start + len(data)) if more else "End of file.\n")
    return result(header + (data or "(empty page)"))


def t_search_files(a):
    """Bounded search: literal by default, or a regex with regex=true. Offsets
    feed directly into the file reader."""
    base = _base(a)
    root = inside(a.get("path") or "", base)
    if os.path.isfile(root):
        paths, partial = [os.path.relpath(root, os.path.realpath(base or WORKSPACE))], False
    else:
        rows, partial = list_files(a.get("path") or "", base=base)
        paths = [row.rsplit("  (", 1)[0] for row in rows]
    query = a["query"]
    if not query.strip():
        raise ValueError("query must contain text")
    if a.get("regex"):
        try:
            pattern = re.compile(query, re.IGNORECASE)
        except re.error as e:
            raise ValueError("Invalid regex: %s" % e)
        needle = None
        matched = pattern.search
    else:
        needle = query.casefold()
        matched = lambda line: needle in line.casefold()
    out, size, scanned = [], 0, 0
    deadline = time.monotonic() + 3
    for path in paths:
        if time.monotonic() >= deadline or scanned >= 8 * 1024 * 1024:
            partial = True
            break
        try:
            with open(inside(path, base), "rb") as f:
                raw = f.read(1024 * 1024 + 1)
            scanned += len(raw)
            if b"\0" in raw:
                continue
            if len(raw) > 1024 * 1024:
                partial = True
                raw = raw[:1024 * 1024]
            data = raw.decode("utf-8", "replace").replace("\r\n", "\n").replace("\r", "\n")
        except OSError:
            partial = True
            continue
        offset = 0
        for line_no, line in enumerate(data.splitlines(keepends=True), 1):
            if matched(line):
                row = "%s:%d start=%d: %s" % (path, line_no, offset, line.strip()[:240])
                if len(out) >= 40 or size + len(row) + 1 > MAX_FILE_CHARS:
                    return result("Partial search; narrow path or query.\n" + "\n".join(out))
                out.append(row)
                size += len(row) + 1
            offset += len(line)
    hint = (" This is literal search, not regex; pass regex=true for patterns such as \\bword\\b."
            if needle is not None and not out
            and any(t in needle for t in (".*", ".+", "\\b", "\\s")) else "")
    return result(("Partial search; narrow path or query.\n" if partial else "")
                  + ("\n".join(out) or "No matches in the text searched." + hint))


def _obj(props, required=()):
    return {"type": "object", "properties": props, "required": list(required),
            "additionalProperties": False}


def _s(desc, **kw):
    d = {"type": "string", "description": desc}
    d.update(kw)
    return d


def _i(desc, **kw):
    d = {"type": "integer", "description": desc}
    d.update(kw)
    return d


SESSION = {"type": "string", "description": "The session id. Default: the current task."}

TOOLS = [
    ("opencode_status", t_status,
     "Is OpenCode running, which version and model it has, how many sessions exist, "
     "whether steps are waiting for approval, and which folder it works in. Call this "
     "first if a tool reports it cannot reach OpenCode.",
     _obj({})),
    ("opencode_list_sessions", t_list_sessions,
     "List OpenCode's sessions - one per piece of work, each with its own history "
     "- with ids, titles and when they were last active.",
     _obj({})),
    ("opencode_new_session", t_new_session,
     "Start a fresh OpenCode session for a new piece of work. opencode_ask with no "
     "session_id does this on its own, so only call it to name the session.",
     _obj({"title": _s("A short title for the session.")})),
    ("opencode_ask", t_ask,
     "Give OpenCode a coding task in plain words and follow it to the end. It reads "
     "the code on its own; each edit, command or fetch it wants is shown to the user, "
     "who allows or refuses it - you are not asked and cannot answer for them. Returns "
     "the user's decisions and what OpenCode said and did. By "
     "default: it continues the last session, so OpenCode remembers earlier work. Set "
     "new_session for an unrelated job. When the user has turned Agentic on, edits and "
     "test runs inside the task's copy go through without a card, and an ask that ends "
     "with to-dos open or tests failing is sent back to OpenCode before this returns; "
     "only the user turns Agentic on or off.",
     _obj({"prompt": _s("The task, as you would brief a programmer: what to build or "
                        "change, in which files, and what done looks like."),
           "session_id": _s("Session to continue. Omit to continue the last one."),
           "new_session": {"type": "boolean",
                           "description": "Start a fresh session - only for an unrelated job."},
           "timeout": _i("Seconds OpenCode may work before this hands back, not counting "
                         "time the user spends deciding. Default %d." % DEFAULT_WORK,
                         minimum=10, maximum=MAX_WORK)},
          ["prompt"])),
    ("opencode_wait", t_wait,
     "Keep following a session that is still working, or has work left - after "
     "opencode_ask handed back at its timeout - putting each step it asks about to the "
     "user as opencode_ask does. Returns what it said and did since the last task it "
     "was given.",
     _obj({"session_id": _s("The session id."),
           "timeout": _i("Seconds of work to wait for. Default %d." % DEFAULT_WORK,
                         minimum=10, maximum=MAX_WORK)},
          ["session_id"])),
    ("opencode_get_session", t_get_session,
     "The most recent messages in a session: what was asked, what OpenCode replied, "
     "which tools it ran and which files it touched. Reads only; it does not wait.",
     _obj({"session_id": _s("The session id."),
           "limit": _i("How many messages, newest last. Default 6.", minimum=1, maximum=40)},
          ["session_id"])),
    ("opencode_abort", t_abort,
     "Stop the work OpenCode is doing in a session right now.",
     _obj({"session_id": _s("The session id.")}, ["session_id"])),
    ("opencode_changes", t_changes,
     "What is changed and not yet committed in OpenCode's folder, from git: each file "
     "with lines added and removed - or, given a path, that file's diff. Use it to check "
     "what an ask really changed before reporting.",
     _obj({"path": _s("One file's diff, relative to the folder. Omit for the list."),
           "session_id": SESSION})),
    ("opencode_list_files", t_list_files,
     "The files in OpenCode's folder - in the current task's copy while it has one - "
     "or one subfolder of it, with sizes.",
     _obj({"path": _s("Subfolder, relative to the folder. Default: all of it."),
           "session_id": SESSION})),
    ("opencode_read_file", t_read_file,
     "Read a page of text - from the current task's copy while it has one. Follow the "
     "next start offset to continue; search first to locate code.",
     _obj({"path": _s("File path relative to the folder, e.g. core/agent.py."),
           "start": _i("Character offset, starting at 0.", minimum=0, maximum=10000000),
           "limit": _i("Characters to return; default 6000.", minimum=1, maximum=6000),
           "session_id": SESSION}, ["path"])),
    ("opencode_search_files", t_search_files,
     "Find text in workspace files: literal (case insensitive) by default, or a regex with "
     "regex=true. Returns paths, lines and start offsets for opencode_read_file. Bounded "
     "search reports partial results; narrow path when needed.",
     _obj({"query": _s("Literal text, or a regex when regex=true.", minLength=1, maxLength=200),
           "regex": {"type": "boolean",
                     "description": "Treat query as a regex (case insensitive) instead of "
                                    "literal text. Default false."},
           "path": _s("File or subfolder to search. Omit for workspace."),
           "session_id": SESSION}, ["query"])),
    ("opencode_merge", t_merge,
     "Bring a finished task's work from its copy into the user's folder, as one commit. "
     "The user sees the diff and decides; you cannot merge for them. Call it when the "
     "user says the work is right, not before.",
     _obj({"session_id": SESSION,
           "message": _s("The commit message. Default: 'OpenCode: <task title>'.")})),
    ("opencode_undo", t_undo,
     "Take back the last ask's change in a task's copy - or, for a merged task, revert "
     "the merge commit. The user confirms first. OpenCode is told on the next ask.",
     _obj({"session_id": SESSION})),
    ("opencode_discard", t_discard,
     "Throw a task away: its copy and branch are deleted and nothing is merged. The "
     "user confirms first.",
     _obj({"session_id": SESSION})),
    ("opencode_grants", t_grants,
     "The steps the user said to always allow in a task, which run without asking.",
     _obj({"session_id": SESSION})),
    ("opencode_revoke", t_revoke,
     "Take back the user's standing grants in a task - one by its number from "
     "opencode_grants, or all of them - so those steps are asked again.",
     _obj({"session_id": SESSION,
           "number": _i("Which grant, from opencode_grants. Omit for all.", minimum=1)})),
]

TOOLS_BY_NAME = {name: (fn, desc, schema) for name, fn, desc, schema in TOOLS}

# Tools that change nothing. The executor reads this hint to decide which calls
# owe a read-back; opencode_ask is the edit, and its reply is not proof the code
# works, so it is not listed - the model is expected to check what came back.
READ_ONLY = {"opencode_status", "opencode_list_sessions", "opencode_get_session",
             "opencode_list_files", "opencode_read_file", "opencode_search_files",
             "opencode_changes", "opencode_grants"}


# Hints past read-only. opencode_ask and opencode_wait may lead to edits the
# user approves, which cannot be replayed for the same result; abort throws
# away work in progress but repeating it changes nothing more.
HINTS = {
    "opencode_new_session": {"destructive": False},
    "opencode_ask": {"destructive": True},
    "opencode_wait": {"destructive": True},
    "opencode_abort": {"destructive": True, "idempotent": True},
    "opencode_merge": {"destructive": True},
    "opencode_undo": {"destructive": True},
    "opencode_discard": {"destructive": True},
    "opencode_revoke": {"destructive": False, "idempotent": True},
}

SERVER = studio_mcp.Server(
    "studio-opencode-mcp", "3.1",
    studio_mcp.tools_from_table(TOOLS, read_only=READ_ONLY, **HINTS),
    errors=(OpenCodeError, KeyError, TypeError, ValueError, OSError),
    instructions="OpenCode codes in one folder with the local model, each task in a copy "
                 "of its own. opencode_ask briefs it and follows it to the end; every edit, "
                 "command and fetch it wants, and every merge, undo or discard, is put to the "
                 "user through MCP elicitation, never to the model. With the user's Agentic "
                 "switch on, edits and test runs inside a task's copy go through unasked; "
                 "no tool sets that switch. Call "
                 "opencode_status first if a tool reports it cannot reach OpenCode.")


def tool_list():
    return [t.spec() for t in SERVER.tools]


def call_tool(name, arguments):
    """Call a tool from Python: every refusal is a result, never an exception."""
    try:
        return SERVER.call_tool(name, arguments)
    except studio_mcp.JSONRPCError as e:
        return result(e.message, error=True)


def serve(inp=None, out=None):
    SERVER.serve(inp, out)


if __name__ == "__main__":
    sys.exit(studio_mcp.main(SERVER))
