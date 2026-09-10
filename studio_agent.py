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
        self._request_lock = threading.Lock()
        self._inbox = queue.Queue()
        self.proc = subprocess.Popen(
            [exe] + list(args),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", bufsize=1, creationflags=NO_WINDOW,
        )
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._drain_stderr, daemon=True).start()

    def _read_stdout(self):
        try:
            for line in self.proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                    if isinstance(message, dict):
                        self._inbox.put(message)
                except json.JSONDecodeError:
                    pass  # server chatter that isn't protocol
        finally:
            self._inbox.put({"_closed": True})

    def _drain_stderr(self):
        for line in self.proc.stderr:
            if line.strip():
                log("  [mcp] " + line.rstrip(), self.quiet)

    def _send(self, payload):
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()

    def request(self, method, params=None, timeout=180):
        # One reader owns each response; concurrent calls cannot steal replies.
        with self._request_lock:
            return self._request(method, params, timeout)

    def _request(self, method, params=None, timeout=180):
        with self._lock:
            self._id += 1
            rid = self._id
        self._send({"jsonrpc": "2.0", "id": rid, "method": method,
                    "params": params or {}})
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                msg = self._inbox.get(timeout=remaining)
            except queue.Empty:
                break
            if msg.get("_closed"):
                self._inbox.put(msg)
                raise EOFError("The MCP bridge exited before returning a result.")
            if msg.get("id") == rid:
                if "error" in msg:
                    raise RuntimeError("MCP error: %s" % msg["error"])
                return msg.get("result", {})
            # Notifications and late replies to timed-out serialized requests
            # must not accumulate forever in the inbox.
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
    if result.get("structuredContent") is not None:
        parts.append(json.dumps(result["structuredContent"]))
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
            "description": desc,
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
        finish_reason, done = None, False
        with resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    done = True
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError as e:
                    raise RuntimeError("Malformed inference stream; no tools from this response were executed.") from e
                choice = (chunk.get("choices") or [{}])[0]
                finish_reason = choice.get("finish_reason") or finish_reason
                delta = choice.get("delta") or {}
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

        if not done or finish_reason not in ("stop", "tool_calls"):
            raise RuntimeError("Incomplete inference response (%s); no tools from this response were executed."
                               % (finish_reason or "connection ended"))
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
HOW TO WORK
- Look before you write. Ask the project what is really there instead of guessing
  ids, names or numbers.
- The tools you are given are the whole of what you can do. If nothing in the list
  fits, say so plainly - never invent a tool name, an action or an argument.
- Read a tool's description before its first use. The description is the contract,
  and a plausible-looking guess is the most common way these calls fail.
- Keep reads bounded. Prefer compact output; do not dump whole trees without need.
- After a write, verify it landed if the result is not self-evident - a small
  targeted read, not a full re-listing.
- Work in small steps and stop when the user's request is satisfied.
- Do not repeat a call that has already failed the same way. Report what happened.
- Ask first before anything destructive: deleting, overwriting or replacing
  something the user did not ask you to touch.
- The user is watching the app, not this transcript. Say what you did in their
  terms - what got made and where it is - not in tool names and ids.

When the task is done, reply with a short plain-text summary and no further tool calls."""

AE_PROMPT = """You are an agent operating a live After Effects session through tools.
The user watches every change happen; each call is a real undo step in their project.

HOW THE PROJECT IS SHAPED
- A project holds footage items and comps. A comp holds layers, stacked front to
  back. A layer whose source is another comp is a precomp - a reference to that
  comp, not a copy, so editing the precomp changes everywhere it is used.
- Comps are addressed by `compId`, layers by `layerId`. The tools hand these back
  when they create something and on every listing, and they stay valid for the life
  of the project.
- NEVER identify a layer by `index`. Index 1 is whatever sits on top at this
  instant, and every insert renumbers the rest. `id` survives that; `index` does not.
- Learn what is really there with list_comps, then get_comp and list_layers for the
  one you care about. find_layers matches by name when the user names a layer.
  get_layer_full is the deep read of a single layer - use it on the one layer you
  need, not on every layer in the comp.

UNITS - these are the ones that silently produce wrong output
- Colour is RGB 0..1, never 0..255: white is [1,1,1], mid grey [0.5,0.5,0.5], a warm
  orange roughly [0.95,0.6,0.15]. Pass 255-style numbers and you get pure white.
- Time is SECONDS everywhere, never frames. Frame 12 of a 24fps comp is 0.5. A
  comp's `duration` is seconds too.
- Opacity is 0..100. Scale is a percentage, so 100 is original size, not 1.
- Rotation is degrees. Position is [x,y] on a 2D layer, [x,y,z] on a 3D one.
- The comp origin is the TOP-LEFT corner and y grows downward, so the centre of a
  1920x1080 comp is [960,540] and "higher up the frame" means a SMALLER y.

MAKING THINGS
- create_comp takes width and height in pixels, duration in seconds and frameRate,
  and returns the new id. Its defaults are 1920x1080, 5 seconds, 30fps.
- create_text_layer puts the START OF THE FIRST BASELINE at `position`; anchorAlign
  'center' or 'right' move that reference point instead. It also pins tracking to 0
  so the layer does not silently inherit the user's Character panel. To centre a
  title, give the comp's centre x with anchorAlign 'center'.
- create_shape_layer leaves its origin at [0,0], which makes the layer's coordinate
  space the comp's - so every vertex and rectangle position you give afterwards is
  in plain comp pixels. Pass position 'center' only if you actually want After
  Effects' own spawn point, which shifts a drawing authored in comp coordinates by
  half a frame.
- A new shape layer is empty: creating the layer alone does not make visible
  artwork. Use add_shape_content for geometry AND a fill or stroke. Its arguments
  are compId, layerId, optional parentGroupPath, and a nested `content` object.
  For a centred rectangle in a 1920x1080 comp, first create the layer, then use its
  returned layerId for these two add_shape_content calls:
  {"compId": C, "layerId": L, "content": {"type": "rect", "name": "Box",
   "size": [400,240], "position": [960,540], "roundness": 0}}
  {"compId": C, "layerId": L, "content": {"type": "fill", "color": [1,0,0],
   "opacity": 100}}
  C and L stand for real ids returned by tools, not literal argument values.
  Adapt size, position and colour to the request and the actual comp dimensions.
  For a circle use type 'ellipse' with equal size dimensions. A custom 'path'
  takes `vertices`, not `points`, and `closed:true` for a closed outline.
- Keep independently coloured shapes in separate groups: add content
  {"type":"group","name":"Badge"}, then put its geometry and paint inside
  parentGroupPath ['Contents','Badge']. A fill or stroke paints the paths above
  it in that group. Add foreground groups before background groups.
- Edit existing content with set_shape_property (contentPath, property, value),
  or replace vertices with set_shape_path. Discover exact node names with
  get_layer_full using include:['shape'], shapeDetail:'compact'; request 'full'
  when exact property values are needed. Do not report a finished shape if only
  the empty layer succeeded; explain which geometry or paint step failed.
- Solids, nulls, adjustment layers, cameras and lights each have their own creating
  tool. Use the right one rather than faking a background with a text layer or a
  rig control with an invisible solid.
- Parent with parent_layer - a null is the usual rig - and restack with
  reorder_layer.

ANIMATING
- add_keyframe takes a `propertyPath` array: ['Transform','Position'],
  ['Transform','Opacity'], ['Effects','Gaussian Blur','Blurriness'].
- One keyframe is a static value. Movement needs at least two, at different times.
- Interpolation is 'linear', 'bezier' or 'hold'. After Effects' own default is
  linear and it looks mechanical - when the user asks for something smooth, or a
  fade that feels good, use bezier at both ends and add an ease for a firmer settle.
- set_transform with keyframe:true and a `time` is the shortcut for keyframing a
  transform property without spelling out its path.
- set_expression writes the expression AND evaluates it: if After Effects reports an
  error the call throws, so a result that comes back ok is an expression that really
  runs. Read the error, fix the text and call again; never leave a broken one behind.

EFFECTS
- add_effect takes a `matchName`, not the name shown in the Effects panel:
  'ADBE Gaussian Blur 2', 'ADBE Drop Shadow', 'ADBE Slider Control'. matchNames are
  stable across versions and languages; display names are neither.
- A wrong matchName fails immediately and costs nothing, so try the standard name
  first and fall back to list_available_effects only when it does not take.
- Set parameters with set_effect_param, and keyframe them through the
  ['Effects', <effect name>, <parameter>] path.

WHEN SOMETHING IS WRONG
- ae_guide(topic) is this bridge's own manual and covers traps no single tool schema
  shows. Read 'after-effects' before a first substantial build in a session, then
  'animation', 'shapes', 'text' or 'assembly' for the job in hand.
- If a tool reports it cannot reach After Effects, call check_setup and relay its
  nextSteps to the user word for word. Do not diagnose the CEP panel yourself and do
  not retry in a loop - this window has a Start After Effects button for the user.
""" + BASE_RULES

RESOLVE_PROMPT = """You are an agent operating a live DaVinci Resolve session through tools.
The user watches every change happen in their project.

HOW THESE TOOLS ARE SHAPED
- Every tool takes `action` (a string) and `params` (an object). One tool is a whole
  family of operations: {"action": "get_items", "params": {"track_type": "video",
  "index": 1}}.
- Each tool's description lists every action it accepts and the params that action
  takes. That list is the API surface - use an action from it. An invented action
  name is the commonest way these calls fail, and every tool fails the same way.
- Params are named, never positional, and they keep Resolve's own capitalisation:
  set_transform takes Pan, Tilt, ZoomX, ZoomY, RotationAngle; set_composite takes
  Opacity and CompositeMode; set_crop takes CropLeft and friends.

HOW THE PROJECT IS SHAPED
- A project holds one media pool and any number of timelines. The media pool is a
  folder tree ("Master", "Master/Selects"); a timeline is built out of pool clips.
- Media pool clips are addressed by `clip_id` - that is what media_pool_item takes,
  and what the bulk clip_ids arguments want. Timeline clips are addressed
  POSITIONALLY, by track_type + track_index + item_index, and that is how
  timeline_item and timeline_item_color find them. Do not pass a clip_id where a
  triple is wanted.
- track_index counts from 1, so V1 is track_index 1. item_index counts from 0, so
  the first clip on a track is 0. Swapping the two grades the wrong shot.
- Importing media puts a clip in the pool and does nothing else. It is not in the
  edit until append_to_timeline or create_timeline_from_clips puts it there.
- Time is FRAMES and TIMECODE, not seconds: markers are added at a frame, and
  timeline_markers get_current_timecode / set_current_timecode read and move the
  playhead.
- Marker colours are Resolve's colour names, not hex: Blue, Cyan, Green, Yellow,
  Red, Pink, Purple, Fuchsia, Rose, Lavender, Sky, Mint, Lemon, Sand, Cocoa, Cream.

WHERE TO START
- `project_manager` get_current says which project is open, `timeline` get_current
  which timeline, and `resolve_control` get_page which page is in front. Those three
  answer "what am I actually looking at".
- To see the media pool, use `folder` get_clips - optionally a path like
  "Master/Selects" - and `media_pool` get_current_folder. `media_pool` manages
  folders, timelines and imports; it has no listing action of its own.
- To see the edit, use `timeline` get_track_count for a track_type, then get_items
  for each track index you care about.

PAGES AND RENDERING
- Resolve is page-based and some work only exists on its page: grading on Color,
  node work on Fusion, delivery on Deliver. `resolve_control` open_page switches
  between edit, cut, color, fusion, fairlight and deliver.
- A render is: set_format_and_codec, then set_settings for the output directory and
  filename, then add_job - which returns a job_id - then start with that id.
  list_jobs shows the queue and is_rendering says whether one is running.
- Do not guess format and codec strings. get_formats lists what this install has,
  and get_codecs for a format lists what goes with it.

CARE
- Changes made through the scripting API do not reliably land in Resolve's undo
  stack. Treat deleted clips, replaced media and overwritten renders as permanent,
  and ask before doing one the user did not ask for.
- NEVER call `resolve_control` with action "quit". Closing Resolve mid-session costs
  the user unsaved work. If you believe Resolve must restart, say so and stop.
- If a tool reports it cannot reach DaVinci Resolve, say so plainly and stop; do not
  retry in a loop - this window has a Start DaVinci Resolve button for the user.
""" + BASE_RULES

CHAT_SUFFIX = """

This is a continuing conversation, in a window with one tab per app. You are this
app's tab: you see only its history and only its tools, and the user may be talking
to another app in another tab. The user may refer back to things you made earlier -
keep the ids you have already been given rather than re-deriving them, and re-read
only what may have changed since.
Answer questions directly without calling tools when no tool is needed."""


# The one tab with nothing behind it. It has no bridge and no tools, so the
# prompt's whole job is to keep the model from claiming otherwise: a confident
# "done - I added the layer" from a tab that cannot reach After Effects is worse
# than no answer at all.
CHAT_PROMPT = """You are the Chat tab of Studio Assistant: a plain conversation with
the local model, with no creative app behind it.

There is no bridge and there are no tools in this tab. You cannot open, read or change
a project in After Effects, DaVinci Resolve or anything else from here, and you must
never describe such a change as done. When the user wants work carried out in an app,
say so plainly and point them at that app's own tab, where the model is briefed on the
bridge and has its tools.

Be useful with what you do have: answer questions, explain how something in these
applications works, think an approach through, draft copy or a shot list, do the
arithmetic on frame rates, timecode and durations, and help the user decide what to
ask for in an app tab. This is a continuing conversation and the user may refer back
to earlier messages in it. Say when you are unsure rather than inventing specifics -
the reader is working to a deadline, and a confident wrong answer costs real time."""


class AppSpec:
    """
    One drivable app: how to reach it, what to expose, how to talk about it.

    `probe` is a strategy string rather than a callable so the registry stays
    data the tests can walk: "port:7777" or "process:Resolve.exe".
    """

    # False only for ChatSpec below. Anything that starts, probes, counts or
    # repairs a bridge asks this before assuming there is one.
    drivable = True

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
        return self.system_prompt + CHAT_SUFFIX + self.quality_rules()

    def cli_prompt(self):
        return self.system_prompt + self.quality_rules()

    def quality_rules(self):
        from studio_tasks import QUALITY_RULES
        return QUALITY_RULES

    def launch(self):
        exe = self.exe()
        if not exe:
            raise RuntimeError("%s is not installed where this agent looks for it"
                               % self.name)
        subprocess.Popen([exe], close_fds=True,
                         creationflags=getattr(subprocess, "DETACHED_PROCESS", 0))

    def __repr__(self):
        return "<%s %s>" % (type(self).__name__, self.id)


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
        default_groups=["discover", "create", "edit", "animate", "effects", "shapes", "inspect"],
        system_prompt=AE_PROMPT,
        examples=[
            "What comps are in this project?",
            "Make a 1920x1080 title card, 5 seconds at 24fps",
            "Add a centred red circle, 300 pixels across, to my comp",
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


class ChatSpec(AppSpec):
    """
    A tab with no app behind it: the model on its own, no bridge, no tools.

    It duck-types AppSpec - id, colours, prompt, examples - so `Session`, the tab
    strip and the transcript need no special case for it; everything
    bridge-shaped is empty, and `drivable` is False so nothing tries to start,
    probe or launch what is not there.

    Deliberately NOT a member of APPS: DRIVABLE is derived from that list, and
    the sidebar must not advertise chat as something this agent can drive.
    """

    drivable = False

    def __init__(self):
        AppSpec.__init__(
            self, id="chat", name="Chat", tab="Chat",
            code="Ch", fg="#ecebe8", bg="#3f4a5a",
            exe_globs=[], probe="", command=None, args=[],
            bridge_label="no bridge", groups={}, default_groups=[],
            system_prompt=CHAT_PROMPT,
            examples=[
                "What frame rate should I finish this in?",
                "How long is 240 frames at 23.976?",
                "Talk me through how to stage a lower third before I build it",
            ])

    def exe(self):
        return None

    def installed(self):
        return True                       # nothing to install, nothing to find

    def running(self):
        return True                       # the tab is the whole of it

    def tool_names(self, group_names=None):
        return set()

    def chat_prompt(self):
        # No CHAT_SUFFIX: it briefs an app tab on its bridge and its tools, and
        # this tab has neither. CHAT_PROMPT carries its own continuity note.
        return self.system_prompt

    def quality_rules(self):
        return ""                         # no tools, so no tool rules

    def launch(self):
        raise RuntimeError("Chat has no application to start.")


CHAT = ChatSpec()

# Everything that can be a tab, apps first. APPS stays the registry of drivable
# apps; TABS is what the tab strip and the new-tab menu offer.
TABS = APPS + [CHAT]
TABS_BY_ID = {a.id: a for a in TABS}


def get_app(app_id):
    try:
        return TABS_BY_ID[app_id]
    except KeyError:
        raise KeyError("unknown app %r; pick from %s"
                       % (app_id, ", ".join(TABS_BY_ID)))


def installed_apps():
    """Registry apps whose executable is actually on this machine."""
    return [a for a in APPS if a.installed()]


# ------------------------------------------------------- what is on this machine

ADOBE_DIR = r"C:\Program Files\Adobe"

# Each row carries where the product's own executable sits under its install
# folder: the UI reads the app's real icon straight out of that PE file
# (studio_icons.py), and falls back to the two-letter badge when it cannot.
# The globs are loose because a Beta install renames the exe after itself.
PRODUCTS = [
    ("After Effects", "Ae", "After Effects", "#9999FF", "#00005B",
     r"Support Files\AfterFX.exe"),
    ("Premiere Pro", "Pr", "Premiere Pro", "#EA77FF", "#2A0634",
     r"Adobe Premiere Pro*.exe"),
    ("Photoshop", "Ps", "Photoshop", "#31A8FF", "#001E36",
     r"Photoshop.exe"),
    ("Illustrator", "Ai", "Illustrator", "#FF9A00", "#330000",
     r"Support Files\Contents\Windows\Illustrator.exe"),
    ("Audition", "Au", "Audition", "#00E4BB", "#00312E",
     r"Adobe Audition*.exe"),
    ("Media Encoder", "Me", "Media Encoder", "#9999FF", "#1D1D2E",
     r"Adobe Media Encoder*.exe"),
    ("Acrobat", "Ac", "Acrobat", "#FF5252", "#3B0000",
     r"Acrobat\Acrobat.exe"),
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
    for match, code, name, fg, bg, exe_glob in PRODUCTS:
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
        exe = newest_match([os.path.join(ADOBE_DIR, e, exe_glob) for e in hits])
        found.append({"code": code, "name": name, "version": label, "fg": fg,
                      "bg": bg, "id": DRIVABLE.get(name), "exe": exe,
                      "drivable": name in DRIVABLE})
    for path, code, name, fg, bg in OTHER_APPS:
        if os.path.exists(path):
            found.append({"code": code, "name": name, "version": "", "fg": fg,
                          "bg": bg, "id": DRIVABLE.get(name), "exe": path,
                          "drivable": name in DRIVABLE})
    return found


def run_agent(llm, mcp, tools, task, system_prompt, max_steps=25, quiet=False,
              schemas=None, library=None):
    """One task, start to finish. `system_prompt` is final - see AppSpec.cli_prompt."""
    from studio_tasks import Executor, TaskRecord
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": task}]
    record = TaskRecord()
    record.briefs.append(task)
    def emit(kind, payload):
        if kind in ("tool", "tool_result", "sys"):
            log("  " + str(payload), quiet)
    return Executor(llm, mcp, tools, schemas=schemas, record=record,
                    emit=emit, library=library).run(messages, max_steps, streaming=False)


def converse(llm, mcp, tools, app, args, schemas=None):
    """The task given on the command line, or an interactive session if none.

    The CLI shares the GUI's library of made tools: same app, same directory
    beside the settings file, so a tool made in a tab is offered here too.
    """
    import studio_toolsmith as toolsmith
    system_prompt = app.cli_prompt()
    library = None
    if tools:
        library = toolsmith.Library.for_app(app.id)
        allowed, specs = toolsmith.contracts(tools, schemas)
        for problem in library.load(allowed, specs):
            log("  not offering a made tool - " + problem, args.quiet)
    if args.task:
        print(run_agent(llm, mcp, tools, " ".join(args.task), system_prompt,
                        args.max_steps, args.quiet, schemas=schemas, library=library))
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
            print("\n" + run_agent(llm, mcp, tools, task, system_prompt,
                                   args.max_steps, args.quiet, schemas=schemas,
                                   library=library) + "\n")
        except Exception as e:
            print("error: %s\n" % e, file=sys.stderr)


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
    p.add_argument("--app", default=DEFAULT_APP, choices=sorted(TABS_BY_ID),
                   help="which app to drive, or 'chat' for no app at all "
                        "(default: %(default)s)")
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
    if not app.drivable:
        # No bridge to start and no tools to expose: the model on its own.
        if a.groups or a.all_tools:
            p.error("%s has no bridge, so there are no tool groups to choose" % app.name)
        if a.list_tools:
            print("%s has no bridge and exposes no tools." % app.name)
            return 0
        llm = LLM(a.host, a.model, a.temperature)
        log("  model: %s @ %s\n" % (a.model, a.host), a.quiet)
        return converse(llm, None, [], app, a, schemas=[])

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

        return converse(llm, mcp, tools, app, a, schemas=chosen)
    finally:
        mcp.close()


if __name__ == "__main__":
    sys.exit(main())
