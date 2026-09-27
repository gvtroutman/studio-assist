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

The server is on loopback with a password (the key file ServerSpec writes at
each start), because a coding agent's HTTP API is not something a web page in
a browser on this PC should be able to reach.

Stdlib only. The protocol - framing, negotiation, validation, annotations,
progress, cancellation, elicitation - is studio_mcp's; this file is the tools.

    python studio_opencode_mcp.py --list-tools
    python studio_opencode_mcp.py --check
"""

import base64
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import studio_mcp

HERE = os.path.dirname(os.path.abspath(__file__))
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


def _request(path, data=None, method=None):
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


def get_json(path, timeout=15):
    return _read(_open(_request(path), timeout))


def post_json(path, payload, timeout=30):
    return _read(_open(_request(path, payload), timeout))


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
    listed = set((doc.get("paths") or {}).keys())
    if not listed:
        return []
    return sorted(r for k, r in ROUTES.items() if k != "doc" and r not in listed)


def _rows(data, key):
    return data if isinstance(data, list) else (data or {}).get(key, [])


def sessions():
    return _rows(get_json(ROUTES["sessions"]), "sessions")


def new_session(title=""):
    s = post_json(ROUTES["sessions"], {"title": title} if title else {})
    if not s.get("id"):
        raise OpenCodeError("OpenCode created a session with no id: %s" % json.dumps(s)[:300])
    return s


def messages(session_id):
    return _rows(get_json(route("message", sessionID=session_id), timeout=30), "messages")


def statuses():
    data = get_json(ROUTES["status"], timeout=10)
    return data if isinstance(data, dict) else {}


def pending_permissions():
    return _rows(get_json(ROUTES["permissions"], timeout=10), "permissions")


def pending_questions():
    return _rows(get_json(ROUTES["questions"], timeout=10), "questions")


def prompt(session_id, text):
    """Hand OpenCode a task and return at once; `run` follows it."""
    post_json(route("prompt", sessionID=session_id),
              {"parts": [{"type": "text", "text": text}]}, timeout=30)


def abort(session_id):
    post_json(route("abort", sessionID=session_id), {}, timeout=10)


def reply_permission(request_id, reply, message=""):
    payload = {"reply": reply}
    if message:
        payload["message"] = message
    post_json(route("permission_reply", requestID=request_id), payload, timeout=15)


def reply_question(request_id, answers):
    post_json(route("question_reply", requestID=request_id), {"answers": answers}, timeout=15)


def reject_question(request_id):
    post_json(route("question_reject", requestID=request_id), {}, timeout=15)


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
        full = os.path.realpath(path)
        root = os.path.realpath(WORKSPACE)
        if full == root or full.startswith(root + os.sep):
            return os.path.relpath(full, root).replace(os.sep, "/")
    except (OSError, ValueError):
        pass
    return path.replace("\\", "/")


def workspace_note():
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
        shown["always_label"] = ("every edit in this session" if always == ["*"] and kind == "edit"
                                 else ", ".join(always))
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
    subagent's session asks in its own name)."""

    def __init__(self, root):
        self.root = root
        self.parent = {}

    def __contains__(self, sid):
        seen = set()
        while sid and sid not in seen:
            if sid == self.root:
                return True
            seen.add(sid)
            if sid not in self.parent:
                try:
                    info = get_json(route("session", sessionID=sid), timeout=10)
                except OpenCodeError:
                    info = {}
                self.parent[sid] = info.get("parentID")
            sid = self.parent[sid]
        return False


def settle(family, log):
    """Put every pending request of this session's family to the user, and
    return how many there were. Raises Stopped when the user stops."""
    n = 0
    for p in pending_permissions():
        if p.get("sessionID") not in family:
            continue
        n += 1
        headline = describe_permission(p)[0].split("\n")[0]
        (decision, note), asked = ask_permission(p)
        reply_permission(p["id"], decision, note)
        word = {"once": "allowed", "always": "allowed from now on", "reject": "refused"}[decision]
        log.append("%s: %s%s" % (word, headline.replace("OpenCode wants to ", "")
                                 .rstrip(".:"), (" - note: " + note) if note else ""))
        if not asked:
            raise Stopped("no one could approve: " + headline)
    for q in pending_questions():
        if q.get("sessionID") not in family:
            continue
        n += 1
        answers = ask_question(q)
        if answers is None:
            reject_question(q["id"])
            log.append("declined to answer OpenCode's question")
        else:
            reply_question(q["id"], answers)
            log.append("answered OpenCode: %s" % "; ".join(", ".join(a) for a in answers))
    return n


def run(sid, seen, work_limit):
    """Follow session `sid` until it is idle, putting each step it asks about
    to the user. `seen` is the message ids that were there before; what came
    after is summarized. Time the user spends deciding does not count."""
    family = Family(sid)
    log = []
    worked = 0.0
    started = time.monotonic()
    stopped = None
    idle_polls = 0
    try:
        while True:
            if studio_mcp.cancelled():
                raise Stopped("the user pressed Stop")
            tick = time.monotonic()
            asked = settle(family, log)
            if asked:
                idle_polls = 0
                continue                    # decided; look again before sleeping
            state = (statuses().get(sid) or {}).get("type", "idle")
            if state == "idle":
                idle_polls += 1
                # A task just handed over may not be marked busy yet.
                if idle_polls >= 2 and time.monotonic() - started > SETTLE:
                    break
            else:
                idle_polls = 0
            studio_mcp.progress("OpenCode is %s%s" % (
                "working" if state != "idle" else "finishing",
                "; %d step(s) decided" % len(log) if log else ""))
            time.sleep(POLL)
            worked += time.monotonic() - tick
            if worked > work_limit:
                return result(
                    "OpenCode is still working on session %s after %d seconds of work. Do "
                    "not send the task again: call opencode_wait with this session_id to "
                    "keep following it, or opencode_abort to stop it.\n%s"
                    % (sid, work_limit, report(sid, seen, log)), error=True)
    except Stopped as e:
        stopped = str(e)
        try:
            abort(sid)
        except OpenCodeError:
            pass
    text = report(sid, seen, log)
    if stopped:
        return result("The user stopped OpenCode (%s). Session %s is halted; whatever it "
                      "had already changed stays.\n%s" % (stopped.split("\n")[0], sid, text),
                      error=True)
    return result("Session %s is done.\n%s" % (sid, text))


def report(sid, seen, log):
    """What happened since `seen`: every step the user decided, then what
    OpenCode said and did, newest last, within MAX_REPLY_CHARS."""
    out = []
    if log:
        out.append("The user's decisions:\n" + "\n".join("- " + l for l in log))
    chunks, error = [], None
    try:
        for m in messages(sid):
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
    room = MAX_REPLY_CHARS - sum(len(x) for x in out) - 200
    if len(body) > room:
        body = "... [earlier steps cut]\n" + body[-room:]
    out.append("OpenCode:\n" + body)
    if error:
        out.append("OpenCode reported an error: " + error)
    out.append(workspace_note())
    return "\n\n".join(out)


# ------------------------------------------------------------- workspace

def inside(relpath):
    """The absolute path of `relpath` under the workspace, or an error if it
    would leave it. OpenCode is refused anything outside; so is this tab."""
    if not isinstance(relpath, str):
        raise ValueError("path must be a string")
    r = relpath.replace("\\", "/").lstrip("/")
    root = os.path.realpath(WORKSPACE)
    full = os.path.realpath(os.path.join(root, r))
    if full != root and not full.startswith(root + os.sep):
        raise OpenCodeError("%r is outside OpenCode's folder, %s." % (relpath, WORKSPACE))
    return full


def list_files(relpath="", limit=MAX_LIST):
    root = inside(relpath)
    if not os.path.isdir(root):
        raise OpenCodeError("%s is not a folder in the workspace." % (relpath or "/"))
    out = []
    deadline = time.monotonic() + 3
    for base, dirs, files in os.walk(root):
        if time.monotonic() >= deadline:
            return out, True
        dirs[:] = sorted(d for d in dirs if d.casefold() not in SKIP_DIRS)
        for f in sorted(files):
            p = os.path.join(base, f)
            r = os.path.relpath(p, os.path.realpath(WORKSPACE)).replace(os.sep, "/")
            try:
                out.append("%s  (%d bytes)" % (r, os.path.getsize(p)))
            except OSError:
                out.append(r)
            if len(out) >= limit:
                return out, True
    return out, False


# ----------------------------------------------------------------- the tools

def result(text, error=False):
    res = {"content": [{"type": "text", "text": text}]}
    if error:
        res["isError"] = True
    return res


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
    cfg = os.path.join(STATE_DIR, "opencode.json")
    if os.path.isfile(cfg):
        try:
            with open(cfg, encoding="utf-8") as f:
                model = json.load(f).get("model")
            if model:
                lines.append("It codes with the model %s." % model)
        except (OSError, ValueError):
            pass
    lines.append("It reads freely; every edit, command and fetch waits for the user's "
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
        lines.append("%s  %s  %s" % (s.get("id"), (s.get("title") or "(untitled)")[:60], when))
    return result("\n".join(lines))


def t_new_session(a):
    s = new_session(a.get("title") or "")
    return result("Session %s created%s. Send work to it with opencode_ask."
                  % (s["id"], (" - " + s["title"]) if s.get("title") else ""))


def _work_limit(a):
    return max(10, min(int(a.get("timeout") or DEFAULT_WORK), MAX_WORK))


def t_ask(a):
    text = a["prompt"]
    if not isinstance(text, str) or not text.strip():
        raise ValueError("prompt must be a non-empty string")
    sid = a.get("session_id")
    if sid:
        seen = {_info(m).get("id") for m in messages(sid)}
    else:
        sid = new_session(text.strip().splitlines()[0][:60])["id"]
        seen = set()
    prompt(sid, text)
    return run(sid, seen, _work_limit(a))


def t_wait(a):
    sid = a["session_id"]
    msgs = messages(sid)
    # Report from the last thing the user asked, so a wait after a timed-out
    # ask returns the whole answer rather than only what came since.
    last_user = max((i for i, m in enumerate(msgs) if _role(m) == "user"), default=-1)
    seen = {_info(m).get("id") for m in msgs[:last_user + 1]}
    return run(sid, seen, _work_limit(a))


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


def t_changes(a):
    """Uncommitted changes in the workspace, from git through OpenCode."""
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


def t_list_files(a):
    rows, truncated = list_files(a.get("path") or "")
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
    full = inside(a["path"])
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
    """Bounded literal search. Offsets feed directly into the file reader."""
    root = inside(a.get("path") or "")
    if os.path.isfile(root):
        paths, partial = [os.path.relpath(root, WORKSPACE)], False
    else:
        rows, partial = list_files(a.get("path") or "")
        paths = [row.rsplit("  (", 1)[0] for row in rows]
    needle = a["query"].casefold()
    if not needle.strip():
        raise ValueError("query must contain text")
    out, size, scanned = [], 0, 0
    deadline = time.monotonic() + 3
    for path in paths:
        if time.monotonic() >= deadline or scanned >= 8 * 1024 * 1024:
            partial = True
            break
        try:
            with open(inside(path), "rb") as f:
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
            if needle in line.casefold():
                row = "%s:%d start=%d: %s" % (path, line_no, offset, line.strip()[:240])
                if len(out) >= 40 or size + len(row) + 1 > MAX_FILE_CHARS:
                    return result("Partial search; narrow path or query.\n" + "\n".join(out))
                out.append(row)
                size += len(row) + 1
            offset += len(line)
    hint = (" This is literal search, not regex; try one exact term such as transcript."
            if not out and any(t in needle for t in (".*", ".+", "\\b", "\\s")) else "")
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
     "the user's decisions and what OpenCode said and did. Continue a piece of work by "
     "passing the session_id an earlier call returned; omit it to start a new session.",
     _obj({"prompt": _s("The task, as you would brief a programmer: what to build or "
                        "change, in which files, and what done looks like."),
           "session_id": _s("Session to continue. Omit for a new one."),
           "timeout": _i("Seconds OpenCode may work before this hands back, not counting "
                         "time the user spends deciding. Default %d." % DEFAULT_WORK,
                         minimum=10, maximum=MAX_WORK)},
          ["prompt"])),
    ("opencode_wait", t_wait,
     "Keep following a session that is still working - after opencode_ask handed back "
     "at its timeout - putting each step it asks about to the user as opencode_ask "
     "does. Returns what it said and did since the last task it was given.",
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
     _obj({"path": _s("One file's diff, relative to the folder. Omit for the list.")})),
    ("opencode_list_files", t_list_files,
     "The files in OpenCode's folder, or one subfolder of it, with sizes.",
     _obj({"path": _s("Subfolder, relative to the folder. Default: all of it.")})),
    ("opencode_read_file", t_read_file,
     "Read a page of text. Follow the next start offset to continue; search first to locate code.",
     _obj({"path": _s("File path relative to the folder, e.g. studio_agent.py."),
           "start": _i("Character offset, starting at 0.", minimum=0, maximum=10000000),
           "limit": _i("Characters to return; default 6000.", minimum=1, maximum=6000)}, ["path"])),
    ("opencode_search_files", t_search_files,
     "Find literal text (case insensitive) in workspace files. Returns paths, lines and start offsets "
     "for opencode_read_file. Bounded search reports partial results; narrow path when needed.",
     _obj({"query": _s("Literal text to find.", minLength=1, maxLength=200),
           "path": _s("File or subfolder to search. Omit for workspace.")}, ["query"])),
]

TOOLS_BY_NAME = {name: (fn, desc, schema) for name, fn, desc, schema in TOOLS}

# Tools that change nothing. The executor reads this hint to decide which calls
# owe a read-back; opencode_ask is the edit, and its reply is not proof the code
# works, so it is not listed - the model is expected to check what came back.
READ_ONLY = {"opencode_status", "opencode_list_sessions", "opencode_get_session",
             "opencode_list_files", "opencode_read_file", "opencode_search_files", "opencode_changes"}


# Hints past read-only. opencode_ask and opencode_wait may lead to edits the
# user approves, which cannot be replayed for the same result; abort throws
# away work in progress but repeating it changes nothing more.
HINTS = {
    "opencode_new_session": {"destructive": False},
    "opencode_ask": {"destructive": True},
    "opencode_wait": {"destructive": True},
    "opencode_abort": {"destructive": True, "idempotent": True},
}

SERVER = studio_mcp.Server(
    "studio-opencode-mcp", "2.0",
    studio_mcp.tools_from_table(TOOLS, read_only=READ_ONLY, **HINTS),
    errors=(OpenCodeError, KeyError, TypeError, ValueError, OSError),
    instructions="OpenCode codes in one folder with the local model. opencode_ask briefs "
                 "it and follows it to the end; every edit, command and fetch it wants is "
                 "put to the user through MCP elicitation, never to the model. Call "
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
