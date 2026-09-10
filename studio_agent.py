#!/usr/bin/env python3
"""
studio_agent - a local LLM agent that drives creative apps.

Inference runs on the adjoining tailnet box (LM Studio, OpenAI-compatible).
Tools run here, on this workstation, over one MCP bridge per app:

  After Effects    npx @engine-room/after-effects-mcp  ->  CEP panel on :7777
  DaVinci Resolve  davinci-resolve-mcp (local venv)    ->  Resolve scripting API

Every app the agent can drive lives in APPS below. Adding one is a registry
entry, not a code change - see AGENTS.md.

Stdlib only. No pip installs.
"""

import argparse
import glob
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

DEFAULT_HOST = "http://100.127.17.38:1234/v1"
DEFAULT_MODEL = "qwen3-coder-30b-a3b-instruct"
MAX_TOOL_RESULT_CHARS = 8000

# Compound tools (Resolve's are all `action` + `params`) document their entire
# action list in the description - it *is* the API surface. Clipping at 1024
# silently amputated half of `timeline`'s actions and the model then invented
# them. Keep this generous; AE's descriptions are short and unaffected.
MAX_TOOL_DESC_CHARS = 4000

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def log(msg, quiet=False):
    if not quiet:
        print(msg, file=sys.stderr, flush=True)


class MCPClient:
    """Minimal MCP stdio client: newline-delimited JSON-RPC over a child process."""

    def __init__(self, command, args, quiet=False):
        exe = shutil.which(command)
        if not exe and os.path.isfile(command):
            exe = command  # registry entries may point straight at an interpreter
        if not exe:
            raise RuntimeError("could not find %r on PATH" % command)
        self.quiet = quiet
        self._id = 0
        self._lock = threading.Lock()
        self._inbox = queue.Queue()
        self.proc = subprocess.Popen(
            [exe] + list(args),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", bufsize=1, creationflags=NO_WINDOW,
        )
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._drain_stderr, daemon=True).start()

    def _read_stdout(self):
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                self._inbox.put(json.loads(line))
            except json.JSONDecodeError:
                pass  # server chatter that isn't protocol

    def _drain_stderr(self):
        for line in self.proc.stderr:
            if line.strip():
                log("  [mcp] " + line.rstrip(), self.quiet)

    def _send(self, payload):
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()

    def request(self, method, params=None, timeout=180):
        with self._lock:
            self._id += 1
            rid = self._id
        self._send({"jsonrpc": "2.0", "id": rid, "method": method,
                    "params": params or {}})
        deadline = time.time() + timeout
        stash = []
        try:
            while True:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                try:
                    msg = self._inbox.get(timeout=remaining)
                except queue.Empty:
                    break
                if msg.get("id") == rid:
                    if "error" in msg:
                        raise RuntimeError("MCP error: %s" % msg["error"])
                    return msg.get("result", {})
                stash.append(msg)  # notification or out-of-order reply
        finally:
            for m in stash:
                self._inbox.put(m)
        raise TimeoutError("no MCP reply to %s in %ss" % (method, timeout))

    def initialize(self, timeout=180):
        res = self.request("initialize", timeout=timeout, params={
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "studio_agent", "version": "1.0"},
        })
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return res

    def list_tools(self, timeout=180):
        tools, cursor = [], None
        while True:
            params = {"cursor": cursor} if cursor else {}
            res = self.request("tools/list", params, timeout=timeout)
            tools.extend(res.get("tools", []))
            cursor = res.get("nextCursor")
            if not cursor:
                return tools

    def call_tool(self, name, arguments):
        return self.request("tools/call", {"name": name, "arguments": arguments})

    def close(self):
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def mcp_result_to_text(result):
    """Flatten an MCP tool result into something a text model can read."""
    if not isinstance(result, dict):
        return str(result)
    parts = []
    for item in result.get("content", []) or []:
        kind = item.get("type")
        if kind == "text":
            parts.append(item.get("text", ""))
        elif kind == "image":
            parts.append("[image returned: %s, %d bytes base64 - not shown to a text model]"
                         % (item.get("mimeType", "?"), len(item.get("data", ""))))
        else:
            parts.append(json.dumps(item)[:500])
    text = "\n".join(p for p in parts if p) or json.dumps(result)[:1000]
    if result.get("isError"):
        text = "TOOL ERROR: " + text
    if len(text) > MAX_TOOL_RESULT_CHARS:
        text = text[:MAX_TOOL_RESULT_CHARS] + \
            "\n...[truncated %d chars - narrow the request]" % (len(text) - MAX_TOOL_RESULT_CHARS)
    return text


def sanitize_schema(node):
    """
    Make a draft-2020-12 schema digestible to LM Studio's grammar converter.

    The AE bridge describes fixed-length arrays the 2020-12 way - `prefixItems`
    alongside `"items": false` (meaning "nothing past the tuple"). LM Studio's
    converter is draft-07 shaped and rejects the boolean outright with
    "Unrecognized schema: false", which 400s the whole request. Rewriting the
    tuple as a bounded homogeneous array constrains the model identically and
    every converter understands it.
    """
    if isinstance(node, list):
        return [sanitize_schema(n) for n in node]
    if not isinstance(node, dict):
        return node

    out = {k: v for k, v in node.items() if k != "$schema"}

    prefix = out.pop("prefixItems", None)
    if prefix is not None:
        types = {p.get("type") for p in prefix if isinstance(p, dict)}
        out.setdefault("minItems", len(prefix))
        out.setdefault("maxItems", len(prefix))
        if len(types) == 1 and None not in types:
            out["items"] = prefix[0]
        else:
            out.pop("items", None)  # mixed tuple: leave the array unconstrained
    if isinstance(out.get("items"), bool):
        out.pop("items")

    if isinstance(out.get("items"), dict):
        out["items"] = sanitize_schema(out["items"])
    if isinstance(out.get("additionalProperties"), dict):
        out["additionalProperties"] = sanitize_schema(out["additionalProperties"])
    for key in ("properties", "$defs", "definitions"):
        if isinstance(out.get(key), dict):
            out[key] = {k: sanitize_schema(v) for k, v in out[key].items()}
    for key in ("anyOf", "oneOf", "allOf"):
        if isinstance(out.get(key), list):
            out[key] = [sanitize_schema(v) for v in out[key]]
    return out


def to_openai_tools(mcp_tools):
    out = []
    for t in mcp_tools:
        schema = sanitize_schema(t.get("inputSchema") or {"type": "object", "properties": {}})
        desc = (t.get("description") or "").strip()
        out.append({"type": "function", "function": {
            "name": t["name"],
            "description": desc[:MAX_TOOL_DESC_CHARS],
            "parameters": schema,
        }})
    return out


class LLM:
    def __init__(self, base_url, model, temperature=0.2, timeout=300):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    def chat(self, messages, tools=None, max_tokens=None):
        body = {"model": self.model, "messages": messages,
                "temperature": self.temperature}
        if max_tokens:
            body["max_tokens"] = max_tokens
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        req = urllib.request.Request(
            self.url, data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            raise RuntimeError("inference host %s: HTTP %s - %s"
                               % (self.url, e.code, e.read()[:400].decode("utf-8", "replace")))
        except urllib.error.URLError as e:
            raise RuntimeError("cannot reach inference host %s (%s). Is the tailnet up "
                               "and LM Studio serving?" % (self.url, e.reason))

    def stream(self, messages, tools=None, on_text=None):
        """Streamed completion. on_text(str) fires per token; returns the final message."""
        body = {"model": self.model, "messages": messages,
                "temperature": self.temperature, "stream": True}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        req = urllib.request.Request(
            self.url, data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            raise RuntimeError("inference host: HTTP %s - %s"
                               % (e.code, e.read()[:400].decode("utf-8", "replace")))
        except urllib.error.URLError as e:
            raise RuntimeError("cannot reach inference host %s (%s)" % (self.url, e.reason))

        content, calls = [], {}
        with resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                delta = ((chunk.get("choices") or [{}])[0]).get("delta") or {}
                piece = delta.get("content")
                if piece:
                    content.append(piece)
                    if on_text:
                        on_text(piece)
                for tc in delta.get("tool_calls") or []:
                    slot = calls.setdefault(tc.get("index", 0),
                                            {"id": None, "name": "", "args": ""})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] += fn["name"]
                    if fn.get("arguments"):
                        slot["args"] += fn["arguments"]

        msg = {"role": "assistant", "content": "".join(content)}
        if calls:
            msg["tool_calls"] = [
                {"id": c["id"] or ("call_%d" % i), "type": "function",
                 "function": {"name": c["name"], "arguments": c["args"]}}
                for i, c in sorted(calls.items())]
        return msg


# ---------------------------------------------------------------- health probes

def http_alive(url, timeout=2):
    """True when something answers - 405 counts, a websocket endpoint says that."""
    try:
        urllib.request.urlopen(url, timeout=timeout)
        return True
    except urllib.error.HTTPError:
        return True
    except Exception:
        return False


def process_running(image_name, timeout=8):
    """Resolve has no bridge port - its MCP server talks to it in-process."""
    try:
        out = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq %s" % image_name, "/NH"],
            capture_output=True, text=True, timeout=timeout,
            creationflags=NO_WINDOW).stdout or ""
    except Exception:
        return False
    return image_name.lower() in out.lower()


def newest_match(patterns):
    """First existing path across the globs, newest-looking name first."""
    for pattern in patterns:
        hits = sorted(glob.glob(pattern), reverse=True)
        for h in hits:
            if os.path.isfile(h):
                return h
    return None


# Best tool-callers on the box first; used when nothing is loaded yet.
PREFERRED_MODELS = [
    "qwen3-coder-30b-a3b-instruct", "seed-oss-36b-instruct", "qwen3.6-27b",
    "gpt-oss-20b", "gpt-oss-120b-distill-phi-4-14b", "qwen2.5-coder-14b-instruct",
]


def api_root(base_url):
    root = base_url.rstrip("/")
    return root[:-3].rstrip("/") if root.endswith("/v1") else root


def probe_models(base_url, timeout=8):
    """-> (reachable, loaded_id_or_None, [ids], error_or_None)"""
    try:  # LM Studio's REST API reports load state; plain /v1/models does not.
        with urllib.request.urlopen(api_root(base_url) + "/api/v0/models",
                                    timeout=timeout) as r:
            data = json.load(r).get("data", [])
        ids = [m.get("id") for m in data if m.get("id")]
        loaded = next((m.get("id") for m in data if m.get("state") == "loaded"), None)
        return True, loaded, ids, None
    except Exception:
        pass
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=timeout) as r:
            return True, None, [m.get("id") for m in json.load(r).get("data", [])], None
    except Exception as e:
        return False, None, [], str(e)


def pick_model(loaded, ids, want=None):
    """Prefer what the user asked for, then what is already in VRAM, then a known-good."""
    if want and want in ids:
        return want
    if loaded:
        return loaded
    for m in PREFERRED_MODELS:
        if m in ids:
            return m
    return ids[0] if ids else None


# --------------------------------------------------------------- the app registry

BASE_RULES = """
Rules that matter:
- Look before you write. Ask the project what is really there instead of guessing ids.
- Keep reads bounded. Prefer compact output; do not dump whole trees without need.
- After a write, verify it landed if the result is not self-evident.
- Work in small steps and stop when the user's request is satisfied.

When the task is done, reply with a short plain-text summary and no further tool calls."""

AE_PROMPT = """You are an agent operating a live After Effects session through tools.
The user watches every change happen; each call is a real undo step in their project.

Specific to After Effects:
- Identify layers by `id`, never by `index` - an index shifts on every insert.
- Call list_comps / get_comp / list_layers to learn real ids instead of guessing them.
- If a tool reports it cannot reach After Effects, say so plainly and stop; do not
  retry in a loop.
""" + BASE_RULES

RESOLVE_PROMPT = """You are an agent operating a live DaVinci Resolve session through tools.
The user watches every change happen in their project.

Specific to DaVinci Resolve:
- Every tool takes `action` (a string) and `params` (an object). Each tool's
  description lists its actions and the params they take - read it rather than
  inventing an action name.
- Identify media pool clips by `clip_id`. Identify timeline clips by `clip_id`, or
  by track_type + track_index + item_index.
- Resolve is page-based: colour work needs the Color page, node work the Fusion page.
  `resolve_control` with action "open_page" switches (edit, cut, color, fusion,
  fairlight, deliver).
- Start from `timeline` get_current, `media_pool` list and `project_manager`
  get_current - they tell you what is actually open.
- NEVER call `resolve_control` with action "quit". Closing Resolve mid-session costs
  the user unsaved work. If you believe Resolve must restart, say so and stop.
- If a tool reports it cannot reach DaVinci Resolve, say so plainly and stop; do not
  retry in a loop.
""" + BASE_RULES

CHAT_SUFFIX = """

This is a continuing conversation. The user may refer back to things you made
earlier - keep track of the ids you have seen so you do not re-derive them.
Answer questions directly without calling tools when no tool is needed."""


class AppSpec:
    """
    One drivable app: how to reach it, what to expose, how to talk about it.

    `probe` is a strategy string rather than a callable so the registry stays
    data the tests can walk: "port:7777" or "process:Resolve.exe".
    """

    def __init__(self, id, name, tab, code, fg, bg, exe_globs, probe, command,
                 args, bridge_label, groups, default_groups, system_prompt,
                 examples, launch_note=""):
        self.id = id
        self.name = name
        self.tab = tab                    # short label for a tab strip
        self.code = code                  # two-letter badge
        self.fg, self.bg = fg, bg
        self.exe_globs = exe_globs
        self.probe = probe
        self.command, self.args = command, list(args)
        self.bridge_label = bridge_label
        self.groups = groups
        self.default_groups = list(default_groups)
        self.system_prompt = system_prompt
        self.examples = list(examples)
        self.launch_note = launch_note

    def exe(self):
        return newest_match(self.exe_globs)

    def installed(self):
        return self.exe() is not None

    def running(self):
        kind, _, arg = self.probe.partition(":")
        if kind == "port":
            return http_alive("http://127.0.0.1:%s/" % arg)
        if kind == "process":
            return process_running(arg)
        return False

    def tool_names(self, group_names=None):
        wanted = set()
        for g in (group_names if group_names is not None else self.default_groups):
            wanted |= set(self.groups[g])
        return wanted

    def chat_prompt(self):
        return self.system_prompt + CHAT_SUFFIX

    def launch(self):
        exe = self.exe()
        if not exe:
            raise RuntimeError("%s is not installed where this agent looks for it"
                               % self.name)
        subprocess.Popen([exe], close_fds=True,
                         creationflags=getattr(subprocess, "DETACHED_PROCESS", 0))

    def __repr__(self):
        return "<AppSpec %s>" % self.id


AE_GROUPS = {
    "discover": [
        "list_comps", "get_comp", "get_comp_tree", "list_layers", "find_layers",
        "get_layer_full", "get_project_summary", "get_keyframes", "get_expression",
        "list_effects", "set_active_comp", "check_setup", "ae_guide",
    ],
    "create": [
        "create_comp", "create_text_layer", "create_shape_layer", "create_solid_layer",
        "create_null_layer", "create_adjustment_layer", "create_precomp_layer",
        "create_camera_layer", "create_light_layer", "create_footage_layer",
    ],
    "edit": [
        "set_transform", "set_layer", "set_text", "set_comp", "parent_layer",
        "reorder_layer", "duplicate_layer", "delete_layer", "duplicate_comp",
    ],
    "animate": [
        "add_keyframe", "remove_keyframe", "set_interpolation", "set_temporal_ease",
        "set_spatial_tangents", "set_expression", "clear_expression", "toggle_expression",
    ],
    "effects": [
        "add_effect", "remove_effect", "set_effect_param", "set_effect_enabled",
        "list_available_effects",
    ],
    "shapes": [
        "add_shape_content", "set_shape_path", "set_shape_property",
        "add_mask", "set_mask", "remove_mask", "add_text_animator",
    ],
    "inspect": ["screenshot_frame", "screenshot_layer", "diff_comp", "snapshot_comp"],
    "assets": ["import_footage", "export_mogrt", "purge_unused_footage", "place_audio_cues"],
    "raw": ["run_jsx", "run_batch"],
}

# Resolve ships 27 compound tools rather than ~76 small ones, so the groups are
# coarser - but the descriptions are long, and the whole set is still ~9k tokens.
RESOLVE_GROUPS = {
    "discover": [
        "resolve_control", "project_manager", "project_settings", "media_pool",
        "media_storage", "timeline", "timeline_item",
    ],
    "media": [
        "media_pool", "media_pool_item", "media_pool_item_markers", "folder",
        "media_storage",
    ],
    "timeline": [
        "timeline", "timeline_item", "timeline_markers", "timeline_item_markers",
        "timeline_item_takes", "timeline_ai",
    ],
    "color": [
        "timeline_item_color", "graph", "color_group", "gallery", "gallery_stills",
    ],
    "fusion": ["fusion_comp", "timeline_item_fusion"],
    "render": ["render", "render_presets"],
    "manage": [
        "project_manager", "project_manager_folders", "project_manager_cloud",
        "project_manager_database", "layout_presets",
    ],
}

RESOLVE_MCP = os.environ.get(
    "RESOLVE_MCP_DIR", os.path.join(os.path.expanduser("~"), "davinci-resolve-mcp"))

APPS = [
    AppSpec(
        id="after-effects",
        name="After Effects",
        tab="After Effects",
        code="Ae", fg="#9999FF", bg="#00005B",
        exe_globs=[r"C:\Program Files\Adobe\Adobe After Effects *\Support Files\AfterFX.exe"],
        probe="port:7777",
        command="npx", args=["-y", "@engine-room/after-effects-mcp"],
        bridge_label="127.0.0.1:7777",
        groups=AE_GROUPS,
        default_groups=["discover", "create", "edit", "animate", "effects"],
        system_prompt=AE_PROMPT,
        examples=[
            "What comps are in this project?",
            "Make a 1920x1080 title card, 5 seconds at 24fps",
            "Add a drop shadow to the text and fade it in over 12 frames",
        ],
        launch_note="After Effects also needs the ae-mcp panel under Window > Extensions.",
    ),
    AppSpec(
        id="resolve",
        name="DaVinci Resolve",
        tab="Resolve",
        code="Dv", fg="#F5A623", bg="#2B2B2B",
        exe_globs=[r"C:\Program Files\Blackmagic Design\DaVinci Resolve\Resolve.exe"],
        probe="process:Resolve.exe",
        command=os.path.join(RESOLVE_MCP, "venv", "Scripts", "python.exe"),
        args=[os.path.join(RESOLVE_MCP, "src", "server.py")],
        bridge_label="scripting API",
        groups=RESOLVE_GROUPS,
        default_groups=["discover", "media", "timeline", "color", "render"],
        system_prompt=RESOLVE_PROMPT,
        examples=[
            "What's on the current timeline?",
            "Add a marker at the playhead and call it Review",
            "Set up an H.264 render job and show me the queue",
        ],
        launch_note="Resolve takes a while to finish loading, and needs a project open.",
    ),
]

APPS_BY_ID = {a.id: a for a in APPS}
DEFAULT_APP = APPS[0].id


def get_app(app_id):
    try:
        return APPS_BY_ID[app_id]
    except KeyError:
        raise KeyError("unknown app %r; pick from %s"
                       % (app_id, ", ".join(APPS_BY_ID)))


def installed_apps():
    """Registry apps whose executable is actually on this machine."""
    return [a for a in APPS if a.installed()]


# ------------------------------------------------------- what is on this machine

ADOBE_DIR = r"C:\Program Files\Adobe"

# Adobe's own marks are two-letter badges in brand colours, so the UI draws them
# rather than scraping 32px icons out of the .exe files - crisp at any DPI.
PRODUCTS = [
    ("After Effects", "Ae", "After Effects", "#9999FF", "#00005B"),
    ("Premiere Pro", "Pr", "Premiere Pro", "#EA77FF", "#2A0634"),
    ("Photoshop", "Ps", "Photoshop", "#31A8FF", "#001E36"),
    ("Illustrator", "Ai", "Illustrator", "#FF9A00", "#330000"),
    ("Audition", "Au", "Audition", "#00E4BB", "#00312E"),
    ("Media Encoder", "Me", "Media Encoder", "#9999FF", "#1D1D2E"),
    ("Acrobat", "Ac", "Acrobat", "#FF5252", "#3B0000"),
]

OTHER_APPS = [
    (r"C:\Program Files\Blackmagic Design\DaVinci Resolve\Resolve.exe",
     "Dv", "DaVinci Resolve", "#F5A623", "#2B2B2B"),
]

# Derived, never hand-maintained: an app is drivable exactly when the registry
# has a bridge for it, so the sidebar cannot claim more than the agent can do.
DRIVABLE = {a.name: a.id for a in APPS}


def detect_apps():
    """Installed creative apps, newest label first. Pure filesystem, no registry."""
    try:
        entries = os.listdir(ADOBE_DIR)
    except OSError:
        entries = []
    found = []
    for match, code, name, fg, bg in PRODUCTS:
        hits = [e for e in entries if match.lower() in e.lower()]
        if not hits:
            continue
        years, beta = set(), False
        for e in hits:
            if "beta" in e.lower():
                beta = True
                continue
            m = re.search(r"(20\d\d)", e)
            if m:
                years.add(m.group(1))
        label = ", ".join(sorted(years, reverse=True))
        if beta:
            label = (label + ", Beta") if label else "Beta"
        found.append({"code": code, "name": name, "version": label, "fg": fg,
                      "bg": bg, "id": DRIVABLE.get(name),
                      "drivable": name in DRIVABLE})
    for path, code, name, fg, bg in OTHER_APPS:
        if os.path.exists(path):
            found.append({"code": code, "name": name, "version": "", "fg": fg,
                          "bg": bg, "id": DRIVABLE.get(name),
                          "drivable": name in DRIVABLE})
    return found


def run_agent(llm, mcp, tools, task, system_prompt, max_steps=25, quiet=False):
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": task}]
    for step in range(1, max_steps + 1):
        t0 = time.time()
        resp = llm.chat(messages, tools)
        choice = resp["choices"][0]
        msg = choice["message"]
        usage = resp.get("usage", {})
        log("  . step %d - %.1fs, %s prompt / %s completion tokens"
            % (step, time.time() - t0, usage.get("prompt_tokens", "?"),
               usage.get("completion_tokens", "?")), quiet)

        calls = msg.get("tool_calls") or []
        assistant = {"role": "assistant", "content": msg.get("content") or ""}
        if calls:
            assistant["tool_calls"] = calls
        messages.append(assistant)

        if not calls:
            return msg.get("content") or "(the model returned nothing)"

        for call in calls:
            fn = call["function"]
            name = fn["name"]
            raw = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw) if isinstance(raw, str) else raw
            except json.JSONDecodeError as e:
                result_text = ("TOOL ERROR: your arguments were not valid JSON (%s). "
                               "Re-issue the call with valid JSON." % e)
                log("  -> %s  [bad JSON args]" % name, quiet)
            else:
                preview = json.dumps(args)
                log("  -> %s %s" % (name, preview[:160] + ("..." if len(preview) > 160 else "")), quiet)
                try:
                    result_text = mcp_result_to_text(mcp.call_tool(name, args))
                except Exception as e:
                    result_text = "TOOL ERROR: %s" % e
                first = result_text.splitlines()[0] if result_text else ""
                log("     %s" % (first[:160] + ("..." if len(first) > 160 else "")), quiet)
            messages.append({"role": "tool", "tool_call_id": call.get("id", name),
                             "content": result_text})
    return "(stopped: hit the %d-step limit without a final answer)" % max_steps


def env_default(*names, fallback=None):
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return fallback


def main():
    p = argparse.ArgumentParser(
        description="Local LLM agent for creative apps. Inference on the tailnet, "
                    "tools on this PC.")
    p.add_argument("task", nargs="*", help="what to do; omit for an interactive session")
    p.add_argument("--app", default=DEFAULT_APP, choices=sorted(APPS_BY_ID),
                   help="which app to drive (default: %(default)s)")
    p.add_argument("--host", default=env_default("STUDIO_HOST", "AE_AGENT_HOST",
                                                 fallback=DEFAULT_HOST),
                   help="OpenAI-compatible base URL (default: %(default)s)")
    p.add_argument("--model", default=env_default("STUDIO_MODEL", "AE_AGENT_MODEL",
                                                  fallback=DEFAULT_MODEL))
    p.add_argument("--groups", default=None,
                   help="tool groups to expose; app-specific, see --list-groups")
    p.add_argument("--all-tools", action="store_true", help="expose every tool the app has")
    p.add_argument("--max-steps", type=int, default=25)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--list-tools", action="store_true", help="print exposed tools and exit")
    p.add_argument("--list-groups", action="store_true",
                   help="print the tool groups for every app and exit")
    p.add_argument("--quiet", action="store_true", help="hide the step trace")
    a = p.parse_args()

    if a.list_groups:
        for app in APPS:
            print("%s (--app %s)" % (app.name, app.id))
            for g, names in app.groups.items():
                mark = "*" if g in app.default_groups else " "
                print("  %s %-10s %s" % (mark, g, ", ".join(names)))
            print()
        print("* = on by default")
        return 0

    app = get_app(a.app)
    groups = [g.strip() for g in (a.groups or ",".join(app.default_groups)).split(",")
              if g.strip()]
    for g in groups:
        if g not in app.groups:
            p.error("unknown group %r for %s; pick from %s"
                    % (g, app.name, ", ".join(app.groups)))

    log(". connecting to the %s bridge..." % app.name, a.quiet)
    mcp = MCPClient(app.command, app.args, quiet=a.quiet)
    try:
        info = mcp.initialize()
        srv = info.get("serverInfo", {})
        log("  bridge up: %s %s" % (srv.get("name", "?"), srv.get("version", "")), a.quiet)

        all_tools = mcp.list_tools()
        if a.all_tools:
            wanted, chosen = None, all_tools
        else:
            wanted = app.tool_names(groups)
            chosen = [t for t in all_tools if t["name"] in wanted]

        log("  %d of %d tools exposed%s" % (len(chosen), len(all_tools),
            "" if a.all_tools else " (groups: %s)" % ",".join(groups)), a.quiet)

        if a.list_tools:
            for t in sorted(chosen, key=lambda x: x["name"]):
                print("%-26s %s" % (t["name"], (t.get("description") or "").split("\n")[0][:90]))
            missing = (wanted - {t["name"] for t in all_tools}) if wanted else set()
            if missing:
                print("\nnot served by this bridge: %s" % ", ".join(sorted(missing)))
            return 0

        tools = to_openai_tools(chosen)
        llm = LLM(a.host, a.model, a.temperature)
        log("  model: %s @ %s\n" % (a.model, a.host), a.quiet)

        if a.task:
            print(run_agent(llm, mcp, tools, " ".join(a.task), app.system_prompt,
                            a.max_steps, a.quiet))
            return 0

        print("studio_agent [%s] - interactive. Ctrl-C or 'exit' to quit.\n" % app.name)
        prompt = "%s> " % app.id
        while True:
            try:
                task = input(prompt).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if task.lower() in ("exit", "quit"):
                return 0
            if not task:
                continue
            try:
                print("\n" + run_agent(llm, mcp, tools, task, app.system_prompt,
                                       a.max_steps, a.quiet) + "\n")
            except Exception as e:
                print("error: %s\n" % e, file=sys.stderr)
    finally:
        mcp.close()


if __name__ == "__main__":
    sys.exit(main())
