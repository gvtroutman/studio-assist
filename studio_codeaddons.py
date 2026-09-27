"""
studio_codeaddons - the OpenCode tab's Add-ons: what the coding agent is given
beyond its own tools, kept by this app and written into OpenCode's config.
The Image Studio's Add-ons are LoRAs per model; these are the coding agent's.

Three kinds, each what OpenCode itself calls it:

  - "mcp"    an MCP server - a command OpenCode starts, or a URL - whose tools
             OpenCode gets. Every step that changes something is still asked.
             Catalog: the official MCP Registry (registry.modelcontextprotocol.io).
  - "plugin" an OpenCode plugin: an npm package OpenCode installs and loads at
             start. Catalog: npm, keyword `opencode-plugin`.
  - "skill"  a folder with a SKILL.md that OpenCode loads when a task calls for
             it. Catalog: GitHub repositories of skills (anthropics/skills).

The records live in `addons.json` in OpenCode's state folder, never in the
workspace; `config()` turns the enabled ones into the `mcp`, `plugin` and
`skills` sections of the opencode.json ServerSpec writes at each start, so an
add-on takes effect when OpenCode is (re)started. Turned off, it stays listed
and out of the config. Values the user types for an MCP server (keys, tokens)
are kept in that file, in the user's own profile, as OpenCode's config keeps
them.

Stdlib only; no tkinter here - the window is studio_codeaddons_ui.AddonsWindow.
"""

import base64
import json
import os
import re
import shutil
import urllib.error
import urllib.parse
import urllib.request

FILE = "addons.json"
KINDS = ("mcp", "plugin", "skill")
KIND_NAMES = {"mcp": "MCP servers", "plugin": "Plugins", "skill": "Skills"}
NAME = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")

MCP_REGISTRY = "https://registry.modelcontextprotocol.io/v0/servers"
NPM_SEARCH = "https://registry.npmjs.org/-/v1/search"
PLUGIN_KEYWORD = "opencode-plugin"
SKILL_REPOS = ["anthropics/skills"]
GITHUB_API = "https://api.github.com/repos/"
GITHUB_RAW = "https://raw.githubusercontent.com/"
TIMEOUT = 20
PAGE = 30


class AddonError(Exception):
    """A catalog or an install that did not work, in words for the window."""


# ================================================================ records

def path(state_dir):
    return os.path.join(state_dir, FILE)


def load(state_dir):
    """Every add-on record, in the order they were added. A missing or
    unreadable file is an empty list, never an error: the tab must start."""
    try:
        with open(path(state_dir), encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return []
    rows = data.get("addons") if isinstance(data, dict) else None
    return [r for r in (rows or []) if isinstance(r, dict) and r.get("kind") in KINDS
            and isinstance(r.get("name"), str)]


def save(state_dir, addons):
    os.makedirs(state_dir, exist_ok=True)
    tmp = path(state_dir) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"addons": list(addons)}, f, indent=2)
    os.replace(tmp, path(state_dir))


def key(addon):
    return "%s:%s" % (addon.get("kind"), addon.get("name"))


def add(state_dir, record):
    """Add `record`, replacing one of the same kind and name (a reinstall)."""
    record = dict(record, enabled=record.get("enabled", True))
    rows = [r for r in load(state_dir) if key(r) != key(record)]
    rows.append(record)
    save(state_dir, rows)
    return record


def set_enabled(state_dir, k, on):
    rows = load(state_dir)
    for r in rows:
        if key(r) == k:
            r["enabled"] = bool(on)
    save(state_dir, rows)


def remove(state_dir, k, send=None):
    """Take an add-on out. A skill this app downloaded goes to the Recycle Bin
    with its record; a folder the user pointed at is theirs and is left."""
    rows = load(state_dir)
    gone = [r for r in rows if key(r) == k]
    for r in gone:
        folder = r.get("path")
        if r.get("kind") == "skill" and r.get("downloaded") and folder and \
                os.path.isdir(folder) and _inside(folder, skills_dir(state_dir)):
            if send is None:
                from studio_catalog import recycle as send
            send(folder)
    save(state_dir, [r for r in rows if key(r) != k])


def _inside(p, root):
    p, root = os.path.realpath(p), os.path.realpath(root)
    return p == root or p.startswith(root + os.sep)


def config(addons):
    """The opencode.json sections for the enabled add-ons."""
    out = {}
    mcp, plugins, skills = {}, [], []
    for a in addons:
        if not a.get("enabled", True):
            continue
        kind = a.get("kind")
        if kind == "mcp" and isinstance(a.get("config"), dict) and NAME.match(a["name"]):
            mcp[a["name"]] = dict(a["config"], enabled=True)
        elif kind == "plugin" and a.get("package"):
            plugins.append(a["package"])
        elif kind == "skill" and a.get("path"):
            skills.append(a["path"])
    if mcp:
        out["mcp"] = mcp
        # OpenCode runs an MCP server's tools without asking unless told to;
        # a server can do anything, so each of its tools asks like an edit.
        out["permission"] = {"%s_*" % name: "ask" for name in mcp}
    if plugins:
        out["plugin"] = plugins
    if skills:
        out["skills"] = {"paths": skills}
    return out


def installed_keys(state_dir):
    return {key(r) for r in load(state_dir)}


# ================================================================ network

def _get(url, headers=None, raw=False):
    req = urllib.request.Request(url, headers=dict({"User-Agent": "StudioAssist",
                                                    "Accept": "application/json"},
                                                   **(headers or {})))
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            data = r.read()
    except urllib.error.HTTPError as e:
        if e.code == 403 and "github" in url:
            raise AddonError("GitHub refused: its hourly limit for unsigned requests is "
                             "spent. Try again later.")
        raise AddonError("%s answered HTTP %d." % (urllib.parse.urlsplit(url).netloc, e.code))
    except (urllib.error.URLError, OSError) as e:
        raise AddonError("Could not reach %s (%s)." % (urllib.parse.urlsplit(url).netloc,
                                                       getattr(e, "reason", e)))
    return data if raw else json.loads(data.decode("utf-8"))


# ------------------------------------------------------------- MCP servers

def short_name(registry_name):
    """The registry's `io.github.owner/thing` as a config key: `thing`."""
    tail = registry_name.rsplit("/", 1)[-1]
    tail = re.sub(r"[^A-Za-z0-9_.\-]+", "-", tail).strip("-.") or "server"
    return tail[:64]


def _placeholders(text):
    return re.findall(r"\{([A-Za-z0-9_\-]+)\}", text or "")


def mcp_ways(server):
    """How a registry entry can be run from here, best first: each a dict
    with a label, how it runs, and the values the user must supply. npm and
    PyPI packages speaking stdio run locally (npx, uvx); a streamable-HTTP or
    SSE remote is a URL. Anything else - a container image, a .NET tool, a
    package that needs positional arguments - is not offered."""
    ways = []
    for p in server.get("packages") or []:
        kind = p.get("registryType")
        transport = (p.get("transport") or {}).get("type")
        ident, version = p.get("identifier"), p.get("version")
        if transport != "stdio" or kind not in ("npm", "pypi") or not ident:
            continue
        if any(a.get("isRequired") and a.get("type") == "positional" and not a.get("value")
               for a in p.get("packageArguments") or []):
            continue
        pinned = version and version != "latest"
        if kind == "npm":
            command = ["npx", "-y", "%s@%s" % (ident, version) if pinned else ident]
        else:
            command = ["uvx", "%s==%s" % (ident, version) if pinned else ident]
        for a in p.get("packageArguments") or []:
            if a.get("type") == "named" and a.get("name") and a.get("value"):
                command += [a["name"], a["value"]]
            elif a.get("type") == "positional" and a.get("value"):
                command.append(a["value"])
        fields = [{"name": e.get("name"), "description": e.get("description") or "",
                   "required": bool(e.get("isRequired")), "secret": bool(e.get("isSecret")),
                   "default": e.get("default") or "", "where": "env"}
                  for e in p.get("environmentVariables") or [] if e.get("name")]
        ways.append({"label": "%s package, runs on this PC" % ("npm" if kind == "npm" else "PyPI"),
                     "type": "local", "command": command, "fields": fields})
    for r in server.get("remotes") or []:
        if r.get("type") not in ("streamable-http", "sse") or not r.get("url"):
            continue
        fields, headers = [], {}
        for h in r.get("headers") or []:
            if not h.get("name"):
                continue
            value = h.get("value") or "{%s}" % h["name"]
            headers[h["name"]] = value
            for ph in _placeholders(value):
                fields.append({"name": ph, "description": h.get("description") or "",
                               "required": bool(h.get("isRequired")),
                               "secret": bool(h.get("isSecret")), "default": "",
                               "where": "header"})
        for ph in _placeholders(r["url"]):
            fields.append({"name": ph, "description": "", "required": True, "secret": False,
                           "default": "", "where": "url"})
        ways.append({"label": "remote server (%s)" % r["type"], "type": "remote",
                     "url": r["url"], "headers": headers, "fields": fields})
    return ways


def search_mcp(query="", cursor=None, limit=PAGE):
    """(cards, next_cursor) from the MCP Registry: the latest version of each
    server that matches, with the ways it can be run from here."""
    params = {"limit": str(limit)}
    if query:
        params["search"] = query
    if cursor:
        params["cursor"] = cursor
    data = _get(MCP_REGISTRY + "?" + urllib.parse.urlencode(params))
    cards = []
    for row in data.get("servers") or []:
        s = row.get("server") or {}
        official = (row.get("_meta") or {}).get("io.modelcontextprotocol.registry/official") or {}
        if official.get("isLatest") is False or official.get("status") not in (None, "active"):
            continue
        cards.append({"kind": "mcp", "name": short_name(s.get("name", "")),
                      "title": s.get("title") or s.get("name", ""),
                      "registry_name": s.get("name", ""),
                      "description": s.get("description") or "",
                      "version": s.get("version") or "",
                      "url": ((s.get("repository") or {}).get("url") or s.get("websiteUrl") or ""),
                      "ways": mcp_ways(s)})
    return cards, (data.get("metadata") or {}).get("nextCursor")


def mcp_record(card, way, values):
    """The add-on record for running `card` the `way` chosen, with the values
    the user typed. Raises AddonError naming a required value left empty."""
    values = {k: (v or "").strip() for k, v in (values or {}).items()}
    for f in way.get("fields") or []:
        if f["required"] and not (values.get(f["name"]) or f.get("default")):
            raise AddonError("%s needs a value for %s." % (card["title"], f["name"]))

    def fill(text):
        for ph in _placeholders(text):
            text = text.replace("{%s}" % ph, values.get(ph, ""))
        return text

    if way["type"] == "local":
        env = {f["name"]: values.get(f["name"]) or f.get("default")
               for f in way.get("fields") or [] if values.get(f["name"]) or f.get("default")}
        cfg = {"type": "local", "command": list(way["command"])}
        if env:
            cfg["environment"] = env
    else:
        cfg = {"type": "remote", "url": fill(way["url"])}
        headers = {k: fill(v) for k, v in (way.get("headers") or {}).items()}
        headers = {k: v for k, v in headers.items() if v.strip() and v.strip() != "Bearer"}
        if headers:
            cfg["headers"] = headers
    return {"kind": "mcp", "name": card["name"], "title": card["title"],
            "description": card.get("description", ""), "version": card.get("version", ""),
            "source": card.get("registry_name") or card.get("url") or "", "config": cfg}


def hand_mcp(name, line):
    """An MCP server the user typed in: a URL, or a command line."""
    name = short_name(name or "")
    line = (line or "").strip()
    if not line:
        raise AddonError("Type the server's command line or its URL.")
    if re.match(r"^https?://", line):
        cfg = {"type": "remote", "url": line}
    else:
        import shlex
        cfg = {"type": "local", "command": shlex.split(line, posix=False)}
    return {"kind": "mcp", "name": name, "title": name, "description": "Added by hand.",
            "source": line, "config": cfg}


# ---------------------------------------------------------------- plugins

def search_plugins(query="", offset=0, size=PAGE):
    """(cards, next_offset) of npm packages tagged as OpenCode plugins, most
    downloaded first as npm ranks them."""
    text = "keywords:%s %s" % (PLUGIN_KEYWORD, query or "")
    data = _get(NPM_SEARCH + "?" + urllib.parse.urlencode(
        {"text": text.strip(), "size": size, "from": offset}))
    cards = []
    for o in data.get("objects") or []:
        p = o.get("package") or {}
        if not p.get("name"):
            continue
        cards.append({"kind": "plugin", "name": p["name"], "title": p["name"],
                      "description": p.get("description") or "",
                      "version": p.get("version") or "",
                      "downloads": ((o.get("downloads") or {}).get("monthly") or 0),
                      "url": ((p.get("links") or {}).get("npm") or
                              "https://www.npmjs.com/package/" + p["name"])})
    total = data.get("total") or 0
    nxt = offset + size
    return cards, (nxt if nxt < total else None)


def plugin_record(card):
    return {"kind": "plugin", "name": card["name"], "title": card["name"],
            "description": card.get("description", ""), "version": card.get("version", ""),
            "package": "%s@%s" % (card["name"], card["version"]) if card.get("version")
            else card["name"], "source": card.get("url", "")}


# ----------------------------------------------------------------- skills

def skills_dir(state_dir):
    return os.path.join(state_dir, "skills")


def _repo_tree(repo):
    info = _get(GITHUB_API + repo)
    branch = info.get("default_branch") or "main"
    tree = _get(GITHUB_API + repo + "/git/trees/%s?recursive=1" % urllib.parse.quote(branch))
    return branch, [t for t in tree.get("tree") or [] if t.get("type") == "blob"]


def list_skills(repo):
    """The skills in a GitHub repository: every folder holding a SKILL.md,
    with the files under it. Descriptions come later, one read each
    (`skill_front`), so a listing is two API calls."""
    branch, blobs = _repo_tree(repo)
    folders = sorted(os.path.dirname(b["path"]) for b in blobs
                     if b["path"].endswith("/SKILL.md") or b["path"] == "SKILL.md")
    cards = []
    for folder in folders:
        prefix = folder + "/" if folder else ""
        files = [b["path"] for b in blobs if b["path"].startswith(prefix)]
        name = short_name(os.path.basename(folder) or repo.split("/")[-1])
        cards.append({"kind": "skill", "name": name, "title": name, "description": "",
                      "repo": repo, "branch": branch, "folder": folder, "files": files,
                      "url": "https://github.com/%s/tree/%s/%s" % (repo, branch, folder)})
    return cards


def front_matter(text):
    """The name and description a SKILL.md opens with, as a dict."""
    out = {}
    m = re.match(r"^---\s*\r?\n(.*?)\r?\n---", text or "", re.S)
    lines = (m.group(1) if m else "").splitlines()
    for i, line in enumerate(lines):
        if line[:1].isspace():
            continue
        k, sep, v = line.partition(":")
        k, v = k.strip(), v.strip()
        if not sep or k not in ("name", "description"):
            continue
        if v in ("", ">", "|", ">-", "|-", ">+", "|+"):
            # A folded or literal block: the indented lines under the key.
            block = []
            for nxt in lines[i + 1:]:
                if nxt.strip() and not nxt[:1].isspace():
                    break
                block.append(nxt.strip())
            v = " ".join(b for b in block if b)
        out[k] = v.strip("\"'")
    return out


def _raw(card, rel):
    return GITHUB_RAW + "%s/%s/%s" % (card["repo"], card["branch"],
                                       urllib.parse.quote(rel))


def skill_front(card):
    prefix = card["folder"] + "/" if card["folder"] else ""
    text = _get(_raw(card, prefix + "SKILL.md"), raw=True).decode("utf-8", "replace")
    return front_matter(text)


def install_skill(state_dir, card, stop=None):
    """Download a skill's files into this app's skills folder and return its
    record. Files are fetched from raw.githubusercontent.com, one by one."""
    dest = os.path.join(skills_dir(state_dir), card["name"])
    tmp = dest + ".part"
    shutil.rmtree(tmp, ignore_errors=True)
    prefix = card["folder"] + "/" if card["folder"] else ""
    for rel in card["files"]:
        if stop is not None and stop.is_set():
            shutil.rmtree(tmp, ignore_errors=True)
            raise AddonError("Stopped.")
        inner = rel[len(prefix):]
        target = os.path.realpath(os.path.join(tmp, inner))
        if not _inside(target, tmp):
            continue                        # a path that would leave the folder
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "wb") as f:
            f.write(_get(_raw(card, rel), raw=True))
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    os.replace(tmp, dest)
    front = {}
    try:
        with open(os.path.join(dest, "SKILL.md"), encoding="utf-8") as f:
            front = front_matter(f.read())
    except OSError:
        pass
    return {"kind": "skill", "name": card["name"], "title": front.get("name") or card["name"],
            "description": front.get("description") or card.get("description", ""),
            "path": dest, "downloaded": True, "source": card.get("url", "")}


def folder_skill(folder):
    """A skill folder the user already has. It must hold a SKILL.md."""
    md = os.path.join(folder, "SKILL.md")
    if not os.path.isfile(md):
        raise AddonError("%s has no SKILL.md, so it is not a skill folder." % folder)
    with open(md, encoding="utf-8", errors="replace") as f:
        front = front_matter(f.read())
    name = short_name(front.get("name") or os.path.basename(os.path.normpath(folder)))
    return {"kind": "skill", "name": name, "title": front.get("name") or name,
            "description": front.get("description") or "", "path": os.path.abspath(folder),
            "downloaded": False, "source": os.path.abspath(folder)}


# ------------------------------------------------------- the running server

def live_status(url, key_file):
    """What the running OpenCode says of its MCP servers: {name: status dict}
    ("connected", "failed" with an error, "disabled", ...), or None when it
    is not running or will not say."""
    headers = {}
    try:
        with open(key_file, encoding="utf-8") as f:
            pw = f.read().strip()
        headers["Authorization"] = "Basic " + base64.b64encode(
            ("opencode:%s" % pw).encode()).decode()
    except OSError:
        pass
    req = urllib.request.Request(url.rstrip("/") + "/mcp", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            data = json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None
