#!/usr/bin/env python3
"""
ae_agent - a local LLM agent that drives After Effects.

Inference runs on the adjoining tailnet box (LM Studio, OpenAI-compatible).
Tools run here, on the Adobe PC, via the existing @engine-room/after-effects-mcp
bridge talking to the CEP panel inside AE on port 7777.

Stdlib only. No pip installs.
"""

import argparse
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

# Tool groups. A 3B-active MoE gets sloppy when shown all ~76 tools at once,
# so expose a working set by default and widen with --groups / --all-tools.
GROUPS = {
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
DEFAULT_GROUPS = ["discover", "create", "edit", "animate", "effects"]

SYSTEM_PROMPT = """You are an agent operating a live After Effects session through tools.
The user watches every change happen; each call is a real undo step in their project.

Rules that matter:
- Identify layers by `id`, never by `index` - an index shifts on every insert.
- Look before you write. Call list_comps / get_comp / list_layers to learn real ids
  instead of guessing them.
- Keep reads bounded. Prefer compact output; do not dump whole comp trees without need.
- After a write, verify it landed if the result is not self-evident.
- Work in small steps and stop when the user's request is satisfied.
- If a tool reports it cannot reach After Effects, say so plainly and stop; do not retry in a loop.

When the task is done, reply with a short plain-text summary and no further tool calls."""


def log(msg, quiet=False):
    if not quiet:
        print(msg, file=sys.stderr, flush=True)


class MCPClient:
    """Minimal MCP stdio client: newline-delimited JSON-RPC over a child process."""

    def __init__(self, command, args, quiet=False):
        exe = shutil.which(command)
        if not exe:
            raise RuntimeError("could not find %r on PATH" % command)
        self.quiet = quiet
        self._id = 0
        self._lock = threading.Lock()
        self._inbox = queue.Queue()
        self.proc = subprocess.Popen(
            [exe] + args,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
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
            "clientInfo": {"name": "ae_agent", "version": "1.0"},
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
            "description": desc[:1024],
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

AE_EXE = r"C:\Program Files\Adobe\Adobe After Effects 2026\Support Files\AfterFX.exe"
BRIDGE_URL = "http://127.0.0.1:7777/"

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


def ae_running(timeout=2):
    """True when something answers on the panel's port - 405 counts, it's a websocket."""
    try:
        urllib.request.urlopen(BRIDGE_URL, timeout=timeout)
        return True
    except urllib.error.HTTPError:
        return True
    except Exception:
        return False


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

DRIVABLE = {"After Effects"}  # the only app this agent has a bridge for


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
        found.append({"code": code, "name": name, "version": label,
                      "fg": fg, "bg": bg, "drivable": name in DRIVABLE})
    for path, code, name, fg, bg in OTHER_APPS:
        if os.path.exists(path):
            found.append({"code": code, "name": name, "version": "",
                          "fg": fg, "bg": bg, "drivable": name in DRIVABLE})
    return found


def launch_ae():
    if not os.path.exists(AE_EXE):
        raise RuntimeError("After Effects not found at %s" % AE_EXE)
    subprocess.Popen([AE_EXE], close_fds=True,
                     creationflags=getattr(subprocess, "DETACHED_PROCESS", 0))


def run_agent(llm, mcp, tools, task, max_steps=25, quiet=False):
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
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


def main():
    p = argparse.ArgumentParser(
        description="Local LLM agent for After Effects. Inference on the tailnet, tools on this PC.")
    p.add_argument("task", nargs="*", help="what to do; omit for an interactive session")
    p.add_argument("--host", default=os.environ.get("AE_AGENT_HOST", DEFAULT_HOST),
                   help="OpenAI-compatible base URL (default: %(default)s)")
    p.add_argument("--model", default=os.environ.get("AE_AGENT_MODEL", DEFAULT_MODEL))
    p.add_argument("--groups", default=",".join(DEFAULT_GROUPS),
                   help="tool groups to expose: %s" % ", ".join(GROUPS))
    p.add_argument("--all-tools", action="store_true", help="expose every AE tool")
    p.add_argument("--max-steps", type=int, default=25)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--list-tools", action="store_true", help="print exposed tools and exit")
    p.add_argument("--quiet", action="store_true", help="hide the step trace")
    a = p.parse_args()

    log(". connecting to the After Effects MCP bridge...", a.quiet)
    mcp = MCPClient("npx", ["-y", "@engine-room/after-effects-mcp"], quiet=a.quiet)
    try:
        info = mcp.initialize()
        srv = info.get("serverInfo", {})
        log("  bridge up: %s %s" % (srv.get("name", "?"), srv.get("version", "")), a.quiet)

        all_tools = mcp.list_tools()
        if a.all_tools:
            wanted, chosen = None, all_tools
        else:
            wanted = set()
            for g in [g.strip() for g in a.groups.split(",") if g.strip()]:
                if g not in GROUPS:
                    p.error("unknown group %r; pick from %s" % (g, ", ".join(GROUPS)))
                wanted |= set(GROUPS[g])
            chosen = [t for t in all_tools if t["name"] in wanted]

        log("  %d of %d tools exposed%s" % (len(chosen), len(all_tools),
            "" if a.all_tools else " (groups: %s)" % a.groups), a.quiet)

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
            print(run_agent(llm, mcp, tools, " ".join(a.task), a.max_steps, a.quiet))
            return 0

        print("ae_agent - interactive. Ctrl-C or 'exit' to quit.\n")
        while True:
            try:
                task = input("ae> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if task.lower() in ("exit", "quit"):
                return 0
            if not task:
                continue
            try:
                print("\n" + run_agent(llm, mcp, tools, task, a.max_steps, a.quiet) + "\n")
            except Exception as e:
                print("error: %s\n" % e, file=sys.stderr)
    finally:
        mcp.close()


if __name__ == "__main__":
    sys.exit(main())
