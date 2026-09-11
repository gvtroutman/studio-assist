#!/usr/bin/env python3
"""
studio_opencode_mcp - an MCP stdio bridge to an OpenCode server in a container.

OpenCode is a coding agent. Left to itself it reads and writes wherever it is
pointed, so this project never runs it on the workstation's own filesystem:
studio_agent.ContainerSpec starts it inside a Docker container whose only
mount is ONE folder - the workspace - and this bridge talks to the server that
container publishes on loopback:

    /global/health  /session  /session/{id}/message  /session/{id}/abort

The workspace is a folder on this machine (OPENCODE_WORKSPACE, default
%LOCALAPPDATA%\\StudioAssistant\\opencode-workspace) mounted at /workspace in
the container. Everything OpenCode makes lands there and nowhere else. The
file tools here read and write that same folder directly - confined to it, so
the model cannot reach past the sandbox from this side either.

Stdlib only. The protocol - framing, negotiation, validation, annotations,
progress and cancellation - is studio_mcp's; this file is the tools. Run it by
hand to see the tool list, or to check its own contract:

    python studio_opencode_mcp.py --list-tools
    python studio_opencode_mcp.py --check
"""

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import studio_mcp

DEFAULT_URL = "http://127.0.0.1:4096"
OPENCODE_URL = os.environ.get("OPENCODE_URL", DEFAULT_URL).rstrip("/")
WORKSPACE = os.environ.get(
    "OPENCODE_WORKSPACE",
    os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"),
                 "StudioAssistant", "opencode-workspace"))
CONTAINER_WORKSPACE = "/workspace"      # where the same folder is inside the container

MAX_WAIT = 900               # seconds an ask will block for
MAX_FILE_CHARS = 20000       # read_file cap; a model does not need a whole repo
MAX_LIST = 400               # list_files cap
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build",
             ".opencode"}

# Every server route this bridge uses, in one place. opencode_status checks
# them against the server's own OpenAPI document (/doc) and says which ones a
# newer or older OpenCode does not list, so a renamed route is a sentence in
# the transcript rather than a mystery 404 the model improvises around.
ROUTES = {
    "health": "/global/health",
    "doc": "/doc",
    "sessions": "/session",
    "message": "/session/{id}/message",
    "abort": "/session/{id}/abort",
}


class OpenCodeError(Exception):
    """Anything OpenCode, or the network in front of it, refuses."""


# ------------------------------------------------------------------- HTTP

def _open(req, timeout):
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:2000]
        try:
            detail = json.loads(body)
        except ValueError:
            detail = body
        raise OpenCodeError("OpenCode answered HTTP %d: %s" % (e.code, _explain(detail)))
    except urllib.error.URLError as e:
        raise OpenCodeError(
            "Cannot reach OpenCode at %s (%s). Its container is not running; the user "
            "starts it with the Start OpenCode button in this window."
            % (OPENCODE_URL, e.reason))
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


def get_json(path, timeout=15):
    with _open(OPENCODE_URL + path, timeout) as r:
        body = r.read().decode("utf-8")
        return json.loads(body) if body.strip() else {}


def post_json(path, payload, timeout=30):
    req = urllib.request.Request(
        OPENCODE_URL + path, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with _open(req, timeout) as r:
        body = r.read().decode("utf-8")
        return json.loads(body) if body.strip() else {}


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


def sessions():
    data = get_json(ROUTES["sessions"])
    return data if isinstance(data, list) else data.get("sessions", [])


def new_session(title=""):
    payload = {"title": title} if title else {}
    s = post_json(ROUTES["sessions"], payload)
    if not s.get("id"):
        raise OpenCodeError("OpenCode created a session with no id: %s" % json.dumps(s)[:300])
    return s


def messages(session_id):
    data = get_json(ROUTES["message"].format(id=session_id), timeout=30)
    return data if isinstance(data, list) else data.get("messages", [])


def ask(session_id, text, timeout):
    """One turn, synchronously. The server holds the request until the agent
    stops; a timeout here does not stop the agent, so the caller collects the
    rest with opencode_get_session rather than asking again."""
    return post_json(ROUTES["message"].format(id=session_id),
                     {"parts": [{"type": "text", "text": text}]}, timeout=timeout)


def abort(session_id):
    post_json(ROUTES["abort"].format(id=session_id), {}, timeout=10)


# ------------------------------------------------------- message summaries

def _role(m):
    info = m.get("info") if isinstance(m.get("info"), dict) else m
    return info.get("role", "?")


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
    text = "\n".join(lines)
    if len(text) > limit:
        text = text[:limit] + "\n... [%d more characters]" % (len(text) - limit)
    if files:
        text += "\nFiles touched: " + ", ".join(sorted(set(files)))
    return text


def workspace_note():
    return ("Files are in the workspace, %s on this PC (%s inside the container)."
            % (WORKSPACE, CONTAINER_WORKSPACE))


# ------------------------------------------------------------- workspace

def inside(rel):
    """The absolute path of `rel` under the workspace, or an error if it would
    leave it. The container can only see this folder; nor can the model."""
    if not isinstance(rel, str):
        raise ValueError("path must be a string")
    rel = rel.replace("\\", "/").lstrip("/")
    if rel.startswith(CONTAINER_WORKSPACE.lstrip("/") + "/") or rel == CONTAINER_WORKSPACE.lstrip("/"):
        rel = rel[len(CONTAINER_WORKSPACE.lstrip("/")):].lstrip("/")
    root = os.path.realpath(WORKSPACE)
    full = os.path.realpath(os.path.join(root, rel))
    if full != root and not full.startswith(root + os.sep):
        raise OpenCodeError("%r is outside the workspace. OpenCode and this tab can only "
                            "see %s; copy what it needs in with opencode_put_file."
                            % (rel, WORKSPACE))
    return full


def list_files(rel="", limit=MAX_LIST):
    root = inside(rel)
    if not os.path.isdir(root):
        raise OpenCodeError("%s is not a folder in the workspace." % (rel or "/"))
    out = []
    for base, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
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
        lines.append("OpenCode %s is running in its container at %s%s."
                     % (version, OPENCODE_URL, (" (" + note + ")") if note else ""))
        missing = unknown_routes()
        if missing:
            lines.append("This server does not list the route(s) %s in its API document; "
                         "the bridge may need updating for this OpenCode version."
                         % ", ".join(missing))
        try:
            n = len(sessions())
            lines.append("%d session(s) so far." % n)
        except OpenCodeError as e:
            lines.append("Could not list sessions: %s" % e)
    except OpenCodeError as e:
        lines.append(str(e))
    lines.append(workspace_note())
    cfg = os.path.join(WORKSPACE, "opencode.json")
    if os.path.isfile(cfg):
        try:
            with open(cfg, encoding="utf-8") as f:
                model = json.load(f).get("model")
            if model:
                lines.append("It codes with the model %s." % model)
        except (OSError, ValueError):
            pass
    lines.append("The container sees nothing else on this PC.")
    return result("\n".join(lines))


def t_list_sessions(a):
    rows = sessions()
    if not rows:
        return result("No sessions yet. opencode_ask with no session_id starts one.")
    lines = []
    for s in rows[-50:]:
        when = s.get("time", {}).get("updated") or s.get("updated") or ""
        if isinstance(when, (int, float)):
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(when / 1000 if when > 1e11 else when))
        lines.append("%s  %s  %s" % (s.get("id"), (s.get("title") or "(untitled)")[:60], when))
    return result("\n".join(lines))


def t_new_session(a):
    s = new_session(a.get("title") or "")
    return result("Session %s created%s. Send work to it with opencode_ask."
                  % (s["id"], (" - " + s["title"]) if s.get("title") else ""))


def t_ask(a):
    text = a["prompt"]
    if not isinstance(text, str) or not text.strip():
        raise ValueError("prompt must be a non-empty string")
    sid = a.get("session_id")
    if not sid:
        sid = new_session(text.strip().splitlines()[0][:60])["id"]
    timeout = int(a.get("timeout") or 600)
    timeout = max(10, min(timeout, MAX_WAIT))
    try:
        reply = ask(sid, text, timeout)
    except OpenCodeError as e:
        if "timed out" in str(e).lower():
            return result("OpenCode is still working on session %s after %d seconds. Do not "
                          "ask again: call opencode_get_session with this id to collect "
                          "its answer, or opencode_abort to stop it." % (sid, timeout),
                          error=True)
        raise
    body = summarize(reply) if isinstance(reply, dict) else str(reply)[:2000]
    info = reply.get("info", {}) if isinstance(reply, dict) else {}
    err = info.get("error")
    if err:
        body = (body + "\n" if body else "") + "OpenCode reported: " + _explain(err)
    return result("Session %s.\n%s\n%s" % (sid, body or "(no reply text)", workspace_note()),
                  error=bool(err))


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
    return result("Asked OpenCode to stop session %s. Whatever it had already written "
                  "to the workspace stays." % a["session_id"])


def t_list_files(a):
    rows, truncated = list_files(a.get("path") or "")
    if not rows:
        return result("The workspace is empty. %s" % workspace_note())
    text = "\n".join(rows)
    if truncated:
        text += "\n... more; list a subfolder."
    return result(text)


def t_read_file(a):
    full = inside(a["path"])
    if not os.path.isfile(full):
        raise OpenCodeError("%s is not a file in the workspace." % a["path"])
    with open(full, encoding="utf-8", errors="replace") as f:
        data = f.read(MAX_FILE_CHARS + 1)
    if len(data) > MAX_FILE_CHARS:
        data = data[:MAX_FILE_CHARS] + "\n... [truncated at %d characters]" % MAX_FILE_CHARS
    return result(data or "(empty file)")


def t_put_file(a):
    full = inside(a["path"])
    content = a.get("content", "")
    if not isinstance(content, str):
        raise ValueError("content must be a string")
    if os.path.isdir(full):
        raise OpenCodeError("%s is a folder." % a["path"])
    existed = os.path.isfile(full)
    if existed and not a.get("overwrite"):
        raise OpenCodeError("%s already exists; pass overwrite=true to replace it."
                            % a["path"])
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8", newline="") as f:
        f.write(content)
    rel = os.path.relpath(full, os.path.realpath(WORKSPACE)).replace(os.sep, "/")
    return result("%s %s (%d bytes). OpenCode sees it as %s/%s."
                  % ("Replaced" if existed else "Wrote", rel, len(content.encode("utf-8")),
                     CONTAINER_WORKSPACE, rel))


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
     "Is OpenCode's container running, which version and model it has, how many "
     "sessions exist, and where the workspace folder is on this PC. Call this first "
     "if a tool reports it cannot reach OpenCode.",
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
     "Give OpenCode a coding task in plain words and wait for it to finish: it reads, "
     "writes and runs code in the workspace and this returns what it said and did. "
     "Continue a piece of work by passing the session_id an earlier call returned; "
     "omit it to start a new session. Blocks until OpenCode stops; if it is still "
     "going at the timeout, collect the rest with opencode_get_session.",
     _obj({"prompt": _s("The task, as you would brief a programmer: what to build or "
                        "change, where in the workspace, and what done looks like."),
           "session_id": _s("Session to continue. Omit for a new one."),
           "timeout": _i("Seconds to wait before handing back. Default 600.",
                         minimum=10, maximum=MAX_WAIT)},
          ["prompt"])),
    ("opencode_get_session", t_get_session,
     "The most recent messages in a session: what was asked, what OpenCode replied, "
     "which tools it ran and which files it touched. Use it to collect an answer an "
     "opencode_ask timed out waiting for.",
     _obj({"session_id": _s("The session id."),
           "limit": _i("How many messages, newest last. Default 6.", minimum=1, maximum=40)},
          ["session_id"])),
    ("opencode_abort", t_abort,
     "Stop the work OpenCode is doing in a session right now.",
     _obj({"session_id": _s("The session id.")}, ["session_id"])),
    ("opencode_list_files", t_list_files,
     "The files in the workspace folder, or one subfolder of it, with sizes. This is "
     "everything OpenCode can see.",
     _obj({"path": _s("Subfolder, relative to the workspace. Default: the whole workspace.")})),
    ("opencode_read_file", t_read_file,
     "Read one file from the workspace, as text.",
     _obj({"path": _s("File path relative to the workspace, e.g. src/main.py.")}, ["path"])),
    ("opencode_put_file", t_put_file,
     "Write a text file into the workspace so OpenCode can work on it - the only way "
     "anything from outside gets in. Refuses to replace an existing file unless "
     "overwrite is true.",
     _obj({"path": _s("File path relative to the workspace. Folders are created."),
           "content": _s("The whole file's text."),
           "overwrite": {"type": "boolean", "description": "Replace an existing file. "
                                                            "Default false."}},
          ["path", "content"])),
]

TOOLS_BY_NAME = {name: (fn, desc, schema) for name, fn, desc, schema in TOOLS}

# Tools that change nothing. The executor reads this hint to decide which calls
# owe a read-back; opencode_ask is the edit, and its reply is not proof the code
# works, so it is not listed - the model is expected to read what came back.
READ_ONLY = {"opencode_status", "opencode_list_sessions", "opencode_get_session",
             "opencode_list_files", "opencode_read_file"}


# Hints past read-only. opencode_ask edits code in the workspace and cannot be
# replayed for the same result; put_file replaces only when told to; abort
# throws away work in progress but repeating it changes nothing more.
HINTS = {
    "opencode_new_session": {"destructive": False},
    "opencode_ask": {"destructive": True},
    "opencode_abort": {"destructive": True, "idempotent": True},
    "opencode_put_file": {"destructive": True, "idempotent": True},
}

SERVER = studio_mcp.Server(
    "studio-opencode-mcp", "1.1",
    studio_mcp.tools_from_table(TOOLS, read_only=READ_ONLY, **HINTS),
    errors=(OpenCodeError, KeyError, TypeError, ValueError, OSError),
    instructions="OpenCode runs in a container that sees only its workspace folder. "
                 "opencode_ask briefs it and waits; opencode_get_session collects an "
                 "answer a wait ran out on. Call opencode_status first if a tool "
                 "reports it cannot reach OpenCode.")


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
