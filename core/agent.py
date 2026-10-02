#!/usr/bin/env python3
"""
studio_agent - a local LLM agent that drives creative apps.

Inference runs on the adjoining tailnet box (LM Studio, OpenAI-compatible).
Tools run here, on this workstation, over one MCP bridge per app:

  After Effects    npx @engine-room/after-effects-mcp  ->  CEP panel on :7777
  DaVinci Resolve  davinci-resolve-mcp (local venv)    ->  Resolve scripting API
  ComfyUI          apps/comfyui/mcp.py (this folder)   ->  HTTP API on the LLM PC

Every app the agent can drive lives in APPS below. Adding one is a registry
entry, not a code change - see AGENTS.md.

Stdlib only. No pip installs.
"""

if __package__ in (None, ""):  # run as a script: import from the checkout
    import os as _os, sys as _sys
    _sys.path[0] = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
    # This file is now split across core/agent_*.py, which import pieces of it
    # back by its real name (core.agent). Run as __main__, this module has no
    # entry under that name yet, so such an import would re-execute this whole
    # file from scratch, mid-way through its own first execution. Alias it.
    _sys.modules.setdefault("core.agent", _sys.modules[__name__])

import argparse
import base64
import glob
import logging
import ipaddress
import json
import os
import queue
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import core.mcp as studio_mcp
import core.procs as studio_procs
import core.tablog as tablog

DEFAULT_HOST = "http://100.127.17.38:1234/v1"
# The longest an MCP call may run while its bridge keeps reporting progress. A
# silent bridge still times out at the call's own timeout.
PROGRESS_CAP = 1800
# ComfyUI shares the inference box: its GPU does the image work so the 5090
# here stays free for rendering. Same variable the bridge reads.
COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://100.127.17.38:8188").rstrip("/")

# OpenCode runs on this machine as a child of this window, in one folder - this
# repository unless OPENCODE_WORKSPACE names another - and asks the user before
# every edit, command and fetch. It listens on loopback with a password.
# apps/opencode/mcp.py reads the same variables in its own process - keep
# them agreeing.
OPENCODE_URL = os.environ.get("OPENCODE_URL", "http://127.0.0.1:4096").rstrip("/")
OPENCODE_WORKSPACE = (os.environ.get("OPENCODE_WORKSPACE")
                      or os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
OPENCODE_STATE = os.environ.get("OPENCODE_STATE") or os.path.join(
    os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "StudioAssistant", "opencode")
DEFAULT_MODEL = "qwen3-coder-30b-a3b-instruct"
MAX_TOOL_RESULT_CHARS = 8000

# Compound tools (Resolve's are all `action` + `params`) document their entire
# action list in the description - it *is* the API surface. Clipping at 1024
# silently amputated half of `timeline`'s actions and the model then invented
# them. Keep this generous; AE's descriptions are short and unaffected.
MAX_TOOL_DESC_CHARS = 4000

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# The activity log; `studio_doctor.start_activity_log` gives it a file.
LOG = logging.getLogger("studio.agent")
BRIDGE_LOG = logging.getLogger("studio.bridge")   # what a bridge says on stderr


def log(msg, quiet=False):
    if not quiet:
        print(msg, file=sys.stderr, flush=True)


from core.agent_mcp_client import (
    Cancelled, MCPClient, mcp_result_to_text, sanitize_schema, to_openai_tools,
    HostUnreachable, ContextLimitError,
)


class LLM:
    """One model on the OpenAI-compatible host, and the draft model that runs
    ahead of it.

    `draft` is speculative decoding: a small model sharing the executing
    model's vocabulary guesses the next few tokens and the big one checks
    them in a single pass, so the answer reads the same and arrives sooner.
    LM Studio takes it per request as `draft_model` and loads it just in
    time; there is nothing to load beforehand. A pair the host refuses is
    dropped after one retry and explained in `draft_note`, so a bad pairing
    costs a line in the tab rather than every request. `drafted` is the
    running (accepted, offered) count of draft tokens when the host reports
    them, for the status line.
    """
    def __init__(self, base_url, model, temperature=0.2, timeout=300, draft=None):
        self.base_url = base_url
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.draft = draft
        self.draft_note = None
        self.drafted = (0, 0)

    def _body(self, messages, tools, **extra):
        body = {"model": self.model, "messages": messages,
                "temperature": self.temperature}
        if self.draft:
            body["draft_model"] = self.draft
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        body.update(extra)
        return body

    def _request(self, body):
        return urllib.request.Request(
            self.url, data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"})

    def _open(self, body):
        """POST and return the response, still open.

        An HTTP error with a draft model in the request is tried once more
        without it: the same request succeeding then is proof the pairing was
        the problem, and speculative decoding is switched off for this model
        rather than failing every request after. The retry failing too means
        the draft was not the trouble, and the original error is reported.
        """
        LOG.info("model %s request: %d messages, %d tools%s", self.model,
                 len(body.get("messages") or []), len(body.get("tools") or []),
                 ", streamed" if body.get("stream") else "")
        try:
            return urllib.request.urlopen(self._request(body), timeout=self.timeout)
        except urllib.error.HTTPError as e:
            LOG.warning("model %s: HTTP %s", self.model, e.code)
            detail = e.read()[:400].decode("utf-8", "replace")
            if body.get("draft_model"):
                plain = {k: v for k, v in body.items() if k != "draft_model"}
                try:
                    resp = urllib.request.urlopen(self._request(plain), timeout=self.timeout)
                except (urllib.error.HTTPError, urllib.error.URLError):
                    pass
                else:
                    self.draft_note = ("The host refused %s as a draft model for %s (HTTP %s - "
                                       "%s); speculative decoding is off for this model."
                                       % (self.draft, self.model, e.code, detail.strip()))
                    self.draft = None
                    return resp
            kind = (ContextLimitError if e.code in (400, 413) and
                    any(word in detail.lower() for word in ("context length", "context window", "context_length_exceeded"))
                    else RuntimeError)
            raise kind("inference host %s: HTTP %s - %s" % (self.url, e.code, detail))
        except urllib.error.URLError as e:
            raise HostUnreachable("cannot reach inference host %s (%s). Is the tailnet up "
                                  "and LM Studio serving?" % (self.url, e.reason))

    def _count(self, stats):
        """The draft's score, when the host says: LM Studio reports how many
        draft tokens were offered and how many the model kept."""
        if not isinstance(stats, dict) or "accepted_draft_tokens_count" not in stats:
            return
        kept, offered = self.drafted
        self.drafted = (kept + int(stats.get("accepted_draft_tokens_count") or 0),
                        offered + int(stats.get("total_draft_tokens_count") or 0))

    def chat(self, messages, tools=None, max_tokens=None):
        extra = {"max_tokens": max_tokens} if max_tokens else {}
        with self._open(self._body(messages, tools, **extra)) as r:
            data = json.load(r)
        self._count(data.get("stats"))
        return data

    def stream(self, messages, tools=None, on_text=None, max_tokens=None):
        """Streamed completion. on_text(str) fires per token; returns the final message."""
        limits = {"max_tokens": max_tokens} if max_tokens else {}
        resp = self._open(self._body(messages, tools, stream=True, **limits))

        content, calls = [], {}
        finish_reason, done, refused = None, False, None
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
                if isinstance(chunk, dict) and chunk.get("error") and not chunk.get("choices"):
                    # LM Studio refuses a streamed request with 200 OK and an
                    # `event: error` line - not an HTTP error, so `_open`'s
                    # draft retry never saw it and every reply read as
                    # "connection ended" (seen live 2026-09-27, draft refused).
                    err = chunk["error"]
                    refused = (err.get("message") if isinstance(err, dict) else str(err)) or "error"
                    break
                self._count(chunk.get("stats"))
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

        if refused is not None:
            if self.draft and not content and not calls:
                self.draft_note = ("The host refused %s as a draft model for %s (%s); "
                                   "speculative decoding is off for this model."
                                   % (self.draft, self.model, refused.strip()[:300]))
                self.draft = None
                return self.stream(messages, tools, on_text, max_tokens)
            raise RuntimeError("inference host %s refused the request: %s" % (self.url, refused))
        if finish_reason == "length":
            # The host stopped the model, not the model itself: the reply hit
            # the context window it was loaded with (or a response-length
            # limit set in LM Studio). Nothing partial is executed; say what
            # the host would not.
            raise ContextLimitError(
                "Incomplete inference response (length): %s ran out of room before its reply "
                "finished, and no tools from it were executed. The reply hit the context "
                "window the model was loaded with - or LM Studio's response-length limit, if "
                "one is set. On the LLM PC, reload %s with a larger context length (16384 or "
                "more) or increase its response limit. Saved task evidence can be continued "
                "through continuation notes; no partial tool call should be replayed."
                % (self.model, self.model))
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

TAILNET = ipaddress.ip_network("100.64.0.0/10")   # every Tailscale address


def host_alive(base_url, timeout=3):
    """Whether the machine behind the inference host answers at all, as
    distinct from its server: True, False, or None when it cannot be told.

    The difference is the whole diagnosis. A PC that is asleep, off or off
    the tailnet answers nothing; a PC that is up with LM Studio's server not
    running answers the tailnet and drops port 1234 - Windows Firewall
    drops a port with no listener rather than refusing it, so both read as
    "timed out" from the probe alone. A Tailscale peer is asked with
    `tailscale ping` (ICMP needs raw sockets; the disco ping needs nothing);
    any other address is not asked.
    """
    host = urllib.parse.urlsplit(base_url).hostname
    try:
        if ipaddress.ip_address(host) not in TAILNET:
            return None
    except ValueError:
        return None
    exe = shutil.which("tailscale")
    if not exe:
        return None
    try:
        r = subprocess.run([exe, "ping", "--c", "1", "--timeout", "%ds" % timeout, host],
                           capture_output=True, text=True, timeout=timeout + 5,
                           creationflags=NO_WINDOW)
    except Exception:
        return None
    return r.returncode == 0 and "pong" in r.stdout


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
    """Resolve has no bridge port - its MCP server talks to it in-process.

    CSV output: the table view cuts image names at 25 characters, which loses
    the ".exe" of "Adobe Premiere Pro (Beta).exe" and the match with it.
    """
    try:
        out = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq %s" % image_name, "/NH", "/FO", "CSV"],
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


# What makes a model a coder, by name. The OpenCode tab takes the best of
# these the host has (`best_coder`); a general model is its last resort.
CODER_HINTS = ("coder", "devstral", "codestral", "deepcoder", "codegemma", "starcoder",
               "codellama", "granite-code")
# A coder bigger than this does not fit the LLM PC's 24 GB at a usable quant.
CODER_MAX_B = 40


def model_size_b(model_id):
    """Total parameters in billions from a name ("qwen3-coder-30b-a3b" -> 30),
    or None. The active-parameter part of an MoE name ("a3b") is not it."""
    best = None
    for m in re.finditer(r"(?<![a-z0-9.])(\d+(?:\.\d+)?)b(?![a-z0-9])", model_id.lower()):
        n = float(m.group(1))
        best = n if best is None else max(best, n)
    return best


def best_coder(ids):
    """The strongest coding model among `ids`, or None: a named coder that
    fits, dense before MoE at similar size (every parameter works on every
    token, which is what keeps a long task in mind), then larger first."""
    def rank(mid):
        low = mid.lower()
        size = model_size_b(mid) or 0
        moe = bool(re.search(r"-a\d+(?:\.\d+)?b", low))
        return (size if not moe else size / 2, size)
    coders = [m for m in ids if m and any(h in m.lower() for h in CODER_HINTS)
              and not looks_vision(m) and "embed" not in m.lower()
              and (model_size_b(m) or 0) <= CODER_MAX_B]
    return max(coders, key=rank) if coders else None


def api_root(base_url):
    root = base_url.rstrip("/")
    return root[:-3].rstrip("/") if root.endswith("/v1") else root


# Names that mark a model as able to look at a picture, for a host whose model
# list carries no type. LM Studio's own list says "vlm" and needs no guessing.
VISION_HINTS = ("-vl", "vl-", "vision", "llava", "pixtral", "minicpm-v", "moondream",
                "gemma-3", "gemma3", "idefics", "florence", "internvl", "smolvlm")


def looks_vision(model_id):
    m = (model_id or "").lower()
    return any(h in m for h in VISION_HINTS)


def probe_models(base_url, timeout=8):
    """-> (reachable, [loaded ids], [ids], [vision ids], error_or_None)

    `loaded` is every model in VRAM, in the host's order - not the first one,
    which after the LLM PC restarts is whichever helper got loaded first: the
    vision model, or the draft. The vision ids are the served models that can
    take a picture in a message: what LM Studio types as "vlm", or, from a
    host that does not say, what the name suggests."""
    try:  # LM Studio's REST API reports load state and type; plain /v1/models does not.
        with urllib.request.urlopen(api_root(base_url) + "/api/v0/models",
                                    timeout=timeout) as r:
            data = json.load(r).get("data", [])
        ids = [m.get("id") for m in data if m.get("id")]
        loaded = [m.get("id") for m in data if m.get("id") and m.get("state") == "loaded"]
        vision = [m.get("id") for m in data
                  if m.get("id") and (m.get("type") == "vlm" or
                                      (not m.get("type") and looks_vision(m.get("id"))))]
        return True, loaded, ids, vision, None
    except Exception:
        pass
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=timeout) as r:
            ids = [m.get("id") for m in json.load(r).get("data", [])]
        return True, [], ids, [i for i in ids if looks_vision(i)], None
    except Exception as e:
        return False, [], [], [], str(e)


def in_vram(loaded):
    """`loaded` as a list, from a caller that has one id or none."""
    return [loaded] if isinstance(loaded, str) else list(loaded or [])


def helper_models():
    """Models the app has the host load for itself - a vision model, a draft -
    which being in VRAM says nothing about what the user wants to run."""
    return set(PREFERRED_VISION_MODELS) | {d for _, drafts in DRAFT_MODELS for d in drafts}


def pick_model(loaded, ids, want=None):
    """The executing model: what the user asked for; else the default when it
    is in VRAM; else whatever else is in VRAM that the app did not put there;
    else a known-good the host has; else the first.

    "What is in VRAM" came before the known-good list from the start, so a
    model loaded by hand in LM Studio is the one used. Once the window loaded a
    vision model of its own and LM Studio a draft, the first loaded id after
    an LLM-PC restart was qwen2.5-vl-7b-instruct, and every tab ran on it -
    a 7B at 8,192 that answered the warm-up with HTTP 500. Those are helpers,
    not choices.
    """
    loaded = in_vram(loaded)
    if want and want in ids:
        return want
    if DEFAULT_MODEL in loaded:
        return DEFAULT_MODEL
    helpers = helper_models()
    for m in loaded:
        if m not in helpers:
            return m
    for m in PREFERRED_MODELS:
        if m in ids:
            return m
    return ids[0] if ids else None


# The least a reply needs after the request's fixed prefix: a tool call with a
# prompt in it is a few hundred tokens, a thinking model's preamble a thousand,
# and every turn adds a tool result. Under this the tab is cut off before it can
# act - "Incomplete inference response (length)" on the first message.
MIN_ROOM = 2048


def context_window(base_url, model_id, timeout=5):
    """(loaded, maximum) context length in tokens of one model the host
    serves, from LM Studio's REST API; (None, None) from a host that does
    not say. `loaded` is what the model was loaded with, which is LM
    Studio's default for it and often far below `maximum`."""
    try:
        with urllib.request.urlopen(api_root(base_url) + "/api/v0/models",
                                    timeout=timeout) as r:
            data = json.load(r).get("data", [])
    except Exception:
        return None, None
    for m in data:
        if isinstance(m, dict) and m.get("id") == model_id:
            loaded, top = m.get("loaded_context_length"), m.get("max_context_length")
            return (loaded if isinstance(loaded, int) else None,
                    top if isinstance(top, int) else None)
    return None, None


def headroom_note(model, prompt_tokens, loaded, maximum=None):
    """What to tell a tab whose prefix leaves the model too little room to
    answer, or "" when there is room or nothing is known.

    `prompt_tokens` is the warm-up's `usage.prompt_tokens`: the exact cost of
    the briefing, the tools and one short message on this model's tokenizer.
    The fix is on the LLM PC, so the note names it in LM Studio's terms and
    as the `lms` command, with the numbers that justify it.
    """
    if not isinstance(prompt_tokens, int) or not isinstance(loaded, int):
        return ""
    room = loaded - prompt_tokens
    if room >= MIN_ROOM:
        return ""
    head = ("This tab's briefing and tools take %s of %s's %s-token context window, "
            "leaving %s for the conversation and each reply - the model will be cut "
            "off before it can finish (\"Incomplete inference response (length)\"). "
            % ("{:,}".format(prompt_tokens), model, "{:,}".format(loaded),
               "about {:,} tokens".format(room) if room > 0 else "nothing"))
    if isinstance(maximum, int) and maximum <= loaded:
        return head + ("That is all %s can take; this tab needs a model with a larger "
                       "context window." % model)
    want = max(16384, 2 * loaded)
    if isinstance(maximum, int):
        want = min(want, maximum)
    return head + ("On the LLM PC, reload %s with a context length of %s or more: eject it "
                   "in LM Studio and load it again with Context Length raised, or run "
                   "`lms load %s --context-length %d`.%s"
                   % (model, "{:,}".format(want), model, want,
                      " The model supports up to {:,}.".format(maximum)
                      if isinstance(maximum, int) else ""))


# Draft models for speculative decoding, by the family whose vocabulary they
# share; smallest first, because a draft earns its keep by being fast and the
# big model rejects what it gets wrong. A family is the id up to its first
# size or variant, so "qwen3-coder-30b-a3b-instruct" is qwen3 and "qwen3.6-27b"
# is not - a mismatched pair is refused by the host, and the LLM client drops
# it, but that is a retry the first request need not pay. Pin a pair the table
# does not know with STUDIO_DRAFT_MODEL.
DRAFT_MODELS = [
    ("qwen3", ["qwen3-0.6b", "qwen3-1.7b"]),
    ("qwen2.5-coder", ["qwen2.5-coder-0.5b-instruct", "qwen2.5-coder-1.5b-instruct"]),
    ("qwen2.5", ["qwen2.5-0.5b-instruct", "qwen2.5-1.5b-instruct"]),
    ("llama-3.1", ["llama-3.2-1b-instruct", "llama-3.2-3b-instruct"]),
    ("llama-3.3", ["llama-3.2-1b-instruct", "llama-3.2-3b-instruct"]),
    ("gemma-3", ["gemma-3-1b-it"]),
]
DRAFT_OFF = ("off", "none", "no", "0", "false")


def draft_family(model_id):
    """The DRAFT_MODELS family a model id belongs to, or None.

    The family has to be a whole part of the name - "qwen3" between dashes,
    the start or the end - not a prefix, or qwen3.6 would draft with qwen3's
    models. "meta-llama-3.1-8b-instruct" is llama-3.1; a publisher/ prefix
    is dropped first. The table is ordered, so qwen2.5-coder is found before
    qwen2.5 claims it."""
    m = (model_id or "").lower().rsplit("/", 1)[-1]
    for family, drafts in DRAFT_MODELS:
        if re.search(r"(^|-)%s(-|$)" % re.escape(family), m):
            return family
    return None


def pick_draft_model(executing, ids, want=None):
    """The draft model to run ahead of `executing`, or None for none.

    `want` is the pin: one of DRAFT_OFF turns speculative decoding off, a
    served id is used as given (the table need not know the pair), and one
    the host does not serve falls through to the table - a stale pin is not
    a reason to give the speed up. A model that is itself draft-sized gets
    no draft: there is nothing smaller worth running ahead of a 1.7B.
    """
    if want and want.strip().lower() in DRAFT_OFF:
        return None
    if want and want in ids and want != executing:
        return want
    drafts = dict(DRAFT_MODELS).get(draft_family(executing), [])
    if executing in (d for _, ds in DRAFT_MODELS for d in ds):
        return None
    for d in drafts:
        if d in ids and d != executing:
            return d
    return None


def resolve_draft(executing, ids):
    """(draft model or None, note to print once or "") for the executing model.

    STUDIO_DRAFT_MODEL is read here so every caller - the window, each tab
    with its own model, the CLI - resolves the same way."""
    want = os.environ.get("STUDIO_DRAFT_MODEL")
    draft = pick_draft_model(executing, ids, want)
    if want and want.strip().lower() in DRAFT_OFF:
        return None, ""
    if want and want not in ids:
        return draft, ("STUDIO_DRAFT_MODEL names %s, which the host does not serve; %s"
                       % (want, "drafting with " + draft if draft else "no draft model"))
    return draft, ""


# Vision models that describe a picture well and fit beside the executing model.
PREFERRED_VISION_MODELS = [
    "qwen3-vl-8b-instruct", "qwen2.5-vl-7b-instruct", "gemma-3-12b-it", "gemma-3-4b-it",
    "qwen3-vl-4b-instruct", "qwen2.5-vl-3b-instruct",
]


def pick_vision_model(vision_ids, executing, want=None, loaded=None):
    """The model that looks at pictures for the executing model.

    STUDIO_VISION_MODEL when the host serves it; else the executing model itself
    when it can see, which costs no second model in VRAM; else one already in
    VRAM, which costs no load; else a known-good vision model; else any the host
    has. None means nothing on the host can see, and every tab says so.
    """
    if want and want in vision_ids:
        return want
    if executing in vision_ids:
        return executing
    for m in in_vram(loaded):
        if m in vision_ids:
            return m
    for m in PREFERRED_VISION_MODELS:
        if m in vision_ids:
            return m
    return vision_ids[0] if vision_ids else None


class Vision:
    """The eyes of a text model: a vision-capable model on the same host.

    Every bridge answers a screenshot with an image, and the user attaches
    pictures to briefs, but the executing model reads text - so a frame it is
    handed is a placeholder unless something looks at it. This looks: it
    describes an attached picture for the brief, and reviews a returned frame
    against the brief so the executor's next step is informed by what is
    actually on screen rather than by a successful write.
    """
    DESCRIBE = ("Describe this picture for someone who cannot see it and has to "
                "recreate or work with it: subject, composition, every distinct "
                "shape and where it sits, colours, any text, anything notable. "
                "Be concrete and brief.")
    REVIEW = ("First say plainly what this frame shows. Then judge it against the "
              "brief: name concrete visual defects and the corrections, and say if "
              "the frame is not what was asked for at all - for instance the original "
              "picture placed unchanged when a remake was wanted. Do not claim to "
              "assess motion or audio from a still. Brief: ")
    # A picture a tool *made* is looked at differently from a frame of the
    # user's project. Asked to name defects, the vision model always finds
    # some - steam from a fox's mouth, fur "not red enough" - and the model
    # driving ComfyUI redrew the picture for each, at minutes a render and
    # against its own briefing's one render per request. So a made picture is
    # described and held to the brief, and nothing more.
    CHECK = ("This picture was just made for the brief below. Say in one or two "
             "sentences what it shows. Then, on a line of its own, write either "
             "\"Matches the brief.\" or \"Not what was asked:\" followed by the one "
             "thing that is plainly wrong - the wrong subject, the wrong number of "
             "things, text that was asked for missing or garbled. Do not list small "
             "flaws or suggest improvements; the user decides what to change next. "
             "Brief: ")
    MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif",
            ".webp": "image/webp", ".bmp": "image/bmp"}

    def __init__(self, base_url, model, timeout=300):
        self.model = model
        self.needs_load = False
        self.llm = LLM(base_url, model, timeout=timeout)

    def fit(self, prompt_tokens):
        """The model loaded with a window that holds `prompt_tokens` and an
        answer. A just-in-time load is 8,192 tokens, and past its window LM
        Studio drops the start of the request without a word - the picture
        with it - so a critic with two reference photos was judging a prompt
        it could only read. -> the window now, or None when the host does not
        say."""
        window, _ = fit_model(self.llm.base_url, self.model, prompt_tokens)
        return window

    def _ask(self, text, mime, data, max_tokens):
        response = self.llm.chat([{"role": "user", "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": "data:%s;base64,%s" % (mime, data)}}]}],
            max_tokens=max_tokens)
        return (response["choices"][0]["message"].get("content") or "").strip()

    @staticmethod
    def _encode(raw, mime):
        # Transparent PNGs go onto white first: the model sees alpha as black.
        if mime == "image/png":
            import core.icons as studio_icons
            raw = studio_icons.flatten_png(raw)
        return base64.b64encode(raw).decode("ascii")

    def describe(self, path):
        """What one picture file shows, in a sentence or a few."""
        return self.ask(path, self.DESCRIBE, 400) or "No description returned."

    def ask(self, path, question, max_tokens=400):
        """The model's answer to `question` about the picture file at `path`."""
        mime = self.MIME.get(os.path.splitext(path)[1].lower(), "image/png")
        with open(path, "rb") as f:
            data = self._encode(f.read(), mime)
        return self._ask(question, mime, data, max_tokens)

    def describe_all(self, paths):
        """The block `_turn` appends to a brief: one line per picture."""
        out = ["%s: %s" % (os.path.basename(p), self.describe(p)) for p in paths]
        return ("\n\nWhat the pictures show (described by the vision model %s):\n"
                % self.model + "\n".join(out))

    def review(self, item, brief, question=None, max_tokens=700):
        """An MCP image content item, judged against the task record."""
        mime = item.get("mimeType", "image/png")
        data = item["data"]
        if mime == "image/png":
            try:
                data = self._encode(base64.b64decode(data), mime)
            except (ValueError, TypeError):
                pass
        return self._ask((question or self.REVIEW) + json.dumps(brief), mime, data,
                         max_tokens) or "No assessment returned."

    def check(self, item, brief):
        """A picture a tool made, held to the brief without a list of flaws
        (see CHECK). Shorter to write, and nothing in it to redraw for."""
        return self.review(item, brief, self.CHECK, 200)


def load_model(base_url, model, timeout=600, context_length=None):
    """Ask LM Studio to load a model it has on disk. -> error text, or None.

    The vision model is picked from everything the host has downloaded, so it
    is usually not in VRAM when the window opens. LM Studio's REST API loads
    on request; a host without that endpoint still loads just-in-time on the
    first chat call, so a failure here is a note, never a stop.

    `context_length` is the window to load it with. Without it LM Studio uses
    the model's default, 8,192 for most - under every tab's prefix here. A
    model already loaded is not reloaded by this call: LM Studio starts a
    second instance beside the first, so `fit_model` unloads first.
    """
    body = {"model": model}
    if isinstance(context_length, int) and context_length > 0:
        body["context_length"] = context_length
    req = urllib.request.Request(
        api_root(base_url) + "/api/v1/models/load",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            json.load(r)
        return None
    except urllib.error.HTTPError as e:
        return "HTTP %s - %s" % (e.code, e.read()[:200].decode("utf-8", "replace"))
    except Exception as e:
        return str(e)


def loaded_instances(base_url, model, timeout=5):
    """[(instance_id, context_length)] of one model's loaded instances on the
    host, from LM Studio's /api/v1/models; [] when none, or from a host that
    does not say. The first instance's id is the model's; a second is
    "<model>:2", and requests by model id go to the first."""
    try:
        with urllib.request.urlopen(api_root(base_url) + "/api/v1/models",
                                    timeout=timeout) as r:
            models = json.load(r).get("models", [])
    except Exception:
        return []
    for m in models:
        if isinstance(m, dict) and m.get("key") == model:
            out = []
            for inst in m.get("loaded_instances") or []:
                if isinstance(inst, dict) and isinstance(inst.get("id"), str):
                    ctx = (inst.get("config") or {}).get("context_length")
                    out.append((inst["id"], ctx if isinstance(ctx, int) else None))
            return out
    return []


def unload_model(base_url, instance_id, timeout=60):
    """Unload one instance from the host. -> error text, or None."""
    req = urllib.request.Request(
        api_root(base_url) + "/api/v1/models/unload",
        data=json.dumps({"instance_id": instance_id}).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            json.load(r)
        return None
    except urllib.error.HTTPError as e:
        return "HTTP %s - %s" % (e.code, e.read()[:200].decode("utf-8", "replace"))
    except Exception as e:
        return str(e)


def make_room(base_url, keep, timeout=10):
    """Unload every language or vision model on the host except those in
    `keep`. -> ([unloaded instance ids], error text or None).

    ComfyUI shares the LLM PC's one GPU. With the 30B and the vision model
    resident it saw 0.3 GB free and streamed every diffusion model from system
    RAM - 200-330 s a picture, 35 s of it sampling. A tab whose tools render
    clears the host before each render; a tab whose model was unloaded is
    reloaded, at its own window, before its next turn (`Chat._turn`).
    Embedding models are left alone: they are small and something may be
    using them."""
    try:
        with urllib.request.urlopen(api_root(base_url) + "/api/v1/models",
                                    timeout=timeout) as r:
            models = json.load(r).get("models", [])
    except Exception as e:
        return [], str(e)
    gone, errors = [], []
    for m in models:
        if not isinstance(m, dict) or m.get("key") in keep or m.get("type") == "embedding":
            continue
        for inst in m.get("loaded_instances") or []:
            iid = inst.get("id") if isinstance(inst, dict) else None
            if not isinstance(iid, str) or iid in keep:
                continue
            err = unload_model(base_url, iid)
            if err:
                errors.append("%s: %s" % (iid, err))
            else:
                gone.append(iid)
    return gone, "; ".join(errors) or None


def give_back(base_url, model, context_length):
    """Reload `model` at `context_length` if a render unloaded it. -> error or None."""
    if loaded_instances(base_url, model):
        return None
    return load_model(base_url, model, context_length=context_length)


class YieldGPU:
    """A bridge whose `heavy` tools want the host's whole GPU: `before()` runs
    ahead of each of them and returns a token, `after(token)` runs once the
    tool returns - or raises. Everything else passes straight through.

    The tab's own model is idle while it waits on a render, so it leaves the
    GPU too: the Qwen edit model is 19.5 GB and does not fit beside even the
    9B, whose first step took 272 s there. Reloading the 9B from the host's
    RAM costs seconds.

    `after` does not run when the tool returns but when the model is next
    needed: whatever talks to the model calls `settle()` first (`Executor`
    before every request, `settle` below). Timed end to end, the picture used
    to sit finished and unseen for 17 s behind the model's reload; now it
    goes back the moment the render ends, and the vision model looks at it
    on a GPU with nothing else loading - LM Studio loading the two at once
    took 15 s for a check that takes 8 alone. A render straight after a
    render never brings the model back in between: it is still away."""

    def __init__(self, bridge, heavy, before, after=None):
        self.bridge, self.heavy = bridge, set(heavy)
        self.before, self.after = before, after
        self.away = None                  # (token,) while a render has the model away

    def call_tool(self, name, arguments, cancel=None):
        if name not in self.heavy:
            return self.bridge.call_tool(name, arguments, cancel=cancel)
        token = self.before()
        if self.away:
            token = self.away[0]          # still away: what it had before the first
        try:
            return self.bridge.call_tool(name, arguments, cancel=cancel)
        finally:
            if self.after:
                self.away = (token,)

    def settle(self):
        """Bring back what a render sent away, if anything is away. Cheap
        when nothing is."""
        away, self.away = self.away, None
        if away:
            self.after(away[0])

    def __getattr__(self, attr):
        return getattr(self.bridge, attr)


def settle(bridge):
    """Bring back the models a render sent away, before anything talks to
    the model - a `YieldGPU` bridge gives them back only when asked - and do
    nothing for any other bridge."""
    fn = getattr(bridge, "settle", None)
    if callable(fn):
        fn()


# What a conversation needs after the fixed prefix, when this app chooses the
# window: several tool exchanges, a screenshot's description, a reply.
ROOM = 8192


def wanted_context(prompt_tokens, maximum=None):
    """The window to load a model with for a prefix of `prompt_tokens`: a
    power of two, 16,384 at least, with ROOM after the prefix; never past the
    model's own maximum."""
    want = 16384
    need = (prompt_tokens if isinstance(prompt_tokens, int) else 0) + ROOM
    while want < need:
        want *= 2
    if isinstance(maximum, int) and maximum > 0:
        want = min(want, maximum)
    return want


def estimate_tokens(system_prompt, tools):
    """A prefix's cost before the host has counted it. Tool JSON tokenizes at
    about 3.5 characters a token (measured on the ComfyUI tab: 27,500 chars,
    7,707 tokens); 3 keeps the estimate on the high side."""
    return (len(system_prompt or "") + len(json.dumps(tools or []))) // 3


def fit_model(base_url, model, prompt_tokens, timeout=600, exact=True, keep=None):
    """Load `model` on the host with a window that fits a prefix of
    `prompt_tokens` - or reload it if the window it has leaves under MIN_ROOM.
    -> (context length now, note for the tab or "").

    `prompt_tokens` is exact after a warm-up (`usage.prompt_tokens`) and an
    `estimate_tokens` before one, which `exact=False` says so the note does
    not quote a guess as a count. Nothing is done when the window fits, when
    the model is already at its maximum (the note says it needs a bigger
    model), or when the host has no REST API (the note is the old advice:
    reload it by hand). A reload drops the host's prefix cache; the caller
    warms up again after.

    `keep`, when given, is what may stay on the host while `model` loads;
    everything else is unloaded first (`make_room`). LM Studio gives a model
    it loads onto an empty card the card, and one it loads beside another
    partly system memory, for as long as it stays loaded: on the LLM PC's
    24 GB, the 30B loaded after the vision model decoded at 20-31 tokens a
    second - still, after the vision model had gone - and at 75-79 loaded
    first. The vision model loads after it, and is the slower one for it.
    """
    loaded, maximum = context_window(base_url, model)
    if loaded is None and maximum is None:
        return None, ""       # a host that does not say: nothing to fit against
    if isinstance(loaded, int) and (not isinstance(prompt_tokens, int) or
                                    loaded - prompt_tokens >= MIN_ROOM):
        return loaded, ""
    if isinstance(loaded, int) and isinstance(maximum, int) and maximum <= loaded:
        return loaded, headroom_note(model, prompt_tokens, loaded, maximum)
    want = wanted_context(prompt_tokens, maximum)
    if isinstance(loaded, int) and want <= loaded:
        return loaded, headroom_note(model, prompt_tokens, loaded, maximum)
    if keep is not None:
        make_room(base_url, set(keep) | {model})
    for instance, _ in loaded_instances(base_url, model):
        err = unload_model(base_url, instance)
        if err:
            return loaded, ("Could not unload %s on the host to reload it with a larger "
                            "context window (%s). %s" % (model, err,
                            headroom_note(model, prompt_tokens, loaded, maximum)))
    err = load_model(base_url, model, timeout=timeout, context_length=want)
    if err:
        return loaded, ("Could not load %s with a %s-token context window (%s). %s"
                        % (model, "{:,}".format(want), err,
                           headroom_note(model, prompt_tokens, loaded, maximum)))
    if isinstance(loaded, int):
        return want, ("Reloaded %s with a %s-token context window: this tab's briefing and "
                      "tools take %s%s, and the %s it was loaded with left %s for the "
                      "conversation." % (model, "{:,}".format(want),
                      "" if exact else "about ", "{:,}".format(prompt_tokens),
                      "{:,}".format(loaded),
                      "about {:,} tokens".format(loaded - prompt_tokens)
                      if loaded > prompt_tokens else "nothing"))
    return want, "Loaded %s with a %s-token context window." % (model, "{:,}".format(want))


def resolve_vision(base_url, executing, vision_ids, loaded=None):
    """A `Vision` for this host, or None with the reason every tab should print.

    Picks from everything the host has, loaded or not; `Vision.needs_load` says
    whether the caller should `load_model` it before the first picture."""
    want = os.environ.get("STUDIO_VISION_MODEL")
    model = pick_vision_model(vision_ids, executing, want, loaded)
    if model:
        vision = Vision(base_url, model)
        vision.needs_load = model != executing and model not in in_vram(loaded)
        return vision, None
    why = ("STUDIO_VISION_MODEL names %s, which the host does not have" % want
           if want else "the host has no vision-capable model downloaded")
    return None, ("%s, so the model cannot see: attached pictures reach it as paths "
                  "and previews go unreviewed. Download a vision model in LM Studio "
                  "(one of %s) and connect again." % (why, ", ".join(PREFERRED_VISION_MODELS[:3])))


from core.agent_prompts import (
    BASE_RULES, LOOKUP_RULES, CREATIVE_RULES, CRAFT_EDITING, CRAFT_MOTION,
    CRAFT_DESIGN, CRAFT_IMAGES, AE_PROMPT, RESOLVE_PROMPT, COMFY_PROMPT,
    OPENCODE_PROMPT, PS_PROMPT, AI_PROMPT, PPRO_PROMPT, BRIDGE_PROMPT,
    CHAT_SUFFIX, CHAT_PROMPT, CHAT_RULES,
)

from core.agent_studio_brief import (
    STUDIO_BRIEF_CHARS, STUDIO_TEMPLATE, studio_brief_path, read_studio_brief,
    studio_section, about_section, lessons_section, RESEARCH_GROUPS,
    RESEARCH_TOOL_NAMES, research_client, Router,
)




class AppSpec:
    """
    One drivable app: how to reach it, what to expose, how to talk about it.

    `probe` is a strategy string rather than a callable so the registry stays
    data the tests can walk: "port:7777", "process:Resolve.exe" or, for an app
    on another machine, "url:http://host:port/".

    An app with no `exe_globs` is remote: nothing here to find or start, so it
    counts as installed, and `launch()` explains where it runs instead.
    """

    # False only for ChatSpec below: there is an application to find, probe,
    # launch and repair. Anything that would start or count a bridge asks
    # `bridged` instead - chat has one of those, in process.
    drivable = True
    bridged = True
    # True only for ServerSpec below: on this machine, started by this window
    # as a server with no window of its own.
    served = False
    # True only for BridgeSpec below: a bridge the user entered by hand.
    custom = False
    # Every app tab gets the research sidecar - this PC's files and the web,
    # read-only, in process - beside its bridge. False only for ChatSpec, whose
    # bridge *is* the research server.
    research = True
    # True only for PanelSpec below: the tab holds another program's window,
    # with no model and no bridge behind it.
    panel = False
    # True only for IMAGE_STUDIO below: a panel tab whose body is our own
    # form (apps/image_studio/ui.py) rather than another program's window.
    images = False
    # True only for TERMINALS below: a panel tab that mirrors console windows
    # opened outside the app (core/terminals_ui.py).
    terminals = False

    def __init__(self, id, name, tab, code, fg, bg, exe_globs, probe, command,
                 args, bridge_label, groups, default_groups, system_prompt,
                 examples, launch_note="", models=(), readback=(), review=None,
                 docs=(), craft="", gpu_tools=(), makes_pictures=False):
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
        # Models this app would rather drive than the shared one, best first.
        # ComfyUI shares its GPU with the inference box: a 30B model resident
        # beside a diffusion model is VRAM the pictures could have had, and the
        # tab's work - one generate call and a filename - does not need it.
        self.models = list(models)
        # Tools that render on the LLM PC's GPU: before each, every other model
        # is unloaded from the host so the render runs in VRAM (YieldGPU).
        self.gpu_tools = frozenset(gpu_tools)
        # True when the pictures this app's tools return are what they made -
        # a render, an edit - rather than a view of the user's project. The
        # vision model then checks one against the brief instead of listing
        # its flaws (Vision.check), because every flaw listed was a redraw.
        self.makes_pictures = bool(makes_pictures)
        # How to check a write landed, for the executor's read-back reminder:
        # (read tool, the id arguments it needs) pairs, most specific first. A
        # write that carried those ids is verified by that read with the same
        # values, so the reminder can name the exact call rather than "inspect
        # the target". Pairs whose ids the write lacks are skipped; a pair
        # with no ids fits any write.
        self.readback = [(tool, list(names)) for tool, names in readback]
        # The picture of the work: the read-only screenshot tool that returns
        # an image, and the id arguments it needs from an earlier call. With a
        # vision model served, the executor takes one itself when the model
        # says it is done with an unverified edit, and feeds the review back.
        self.review = (review[0], list(review[1])) if review else None
        # Where this app is documented, as (title, url) pairs the research
        # sidecar can read - so the model knows where to look before it guesses.
        self.docs = [(title, url) for title, url in docs]
        # How work in this kind of app is done well: the craft block the prompt
        # carries after the bridge's own conventions.
        self.craft = craft

    def model_for(self, ids, shared):
        """The model this app's tab should use, given what the host serves.

        STUDIO_MODEL_<APP> pins one for the app, then the registry preference
        list, then whatever the window is using. Only a model the host serves
        is chosen; a preference that is not served falls through, so a tab
        never fails to open because a small model was uninstalled. Returns
        (model, note) where note says why it differs from the shared one.
        """
        pinned = os.environ.get("STUDIO_MODEL_" + self.id.upper())
        if pinned:
            if pinned in ids:
                return pinned, "pinned by STUDIO_MODEL_%s" % self.id.upper()
            return shared, ("STUDIO_MODEL_%s names %s, which the host does not serve; "
                            "using %s" % (self.id.upper(), pinned, shared))
        for m in self.models:
            if m in ids:
                return m, ("this app prefers a small model so the GPU stays free for it"
                           if m != shared else "")
        if self.models:
            return shared, ("none of this app's preferred models (%s) is served; using %s"
                            % (", ".join(self.models), shared))
        return shared, ""

    def exe(self):
        return newest_match(self.exe_globs)

    @property
    def remote(self):
        return not self.exe_globs

    def installed(self):
        return self.remote or self.exe() is not None

    def running(self):
        kind, _, arg = self.probe.partition(":")
        if kind == "port":
            return http_alive("http://127.0.0.1:%s/" % arg)
        if kind == "process":
            return process_running(arg)
        if kind == "url":
            return http_alive(arg, timeout=4)
        return False

    def tool_names(self, group_names=None):
        wanted = set()
        for g in (group_names if group_names is not None else self.default_groups):
            wanted |= set(self.groups[g])
        return wanted

    def connect(self, quiet=True):
        """This app's bridge as an MCPClient-shaped object, not yet initialized.
        A subprocess for every app; ChatSpec runs its own bridge in process."""
        return MCPClient(self.command, self.args, quiet=quiet)

    def lookup_rules(self):
        """The research sidecar's briefing, with this app's documentation."""
        if not self.research:
            return ""
        docs = ""
        if self.docs:
            docs = ("\n- This app's own documentation, which fetch_page can read - start "
                    "there for how a feature, a script call or a setting works:\n" +
                    "\n".join("    %s: %s" % (title, url) for title, url in self.docs))
        return LOOKUP_RULES % {"docs": docs}

    def briefing(self):
        """What every prompt carries after the bridge's own conventions: the
        craft of this kind of work, how to be creative, and where to look."""
        return self.craft + CREATIVE_RULES + self.lookup_rules()

    def chat_prompt(self, studio="", lessons=""):
        """The GUI's system prompt. `studio` is the studio brief's text and
        `lessons` the app's notebook, rendered; both go last because they are
        the parts that change between sessions, and the host caches by prefix."""
        return (self.system_prompt + self.briefing() + CHAT_SUFFIX + self.quality_rules()
                + studio_section(studio) + lessons_section(lessons))

    def cli_prompt(self, studio="", lessons=""):
        return (self.system_prompt + self.briefing() + self.quality_rules()
                + studio_section(studio) + lessons_section(lessons))

    def quality_rules(self):
        from core.tasks import QUALITY_RULES
        return QUALITY_RULES

    def launch(self):
        if self.remote:
            raise RuntimeError("%s runs on another machine; this window cannot start "
                               "it. %s" % (self.name, self.launch_note))
        exe = self.exe()
        if not exe:
            raise RuntimeError("%s is not installed where this agent looks for it"
                               % self.name)
        subprocess.Popen([exe], close_fds=True,
                         creationflags=getattr(subprocess, "DETACHED_PROCESS", 0))

    def __repr__(self):
        return "<%s %s>" % (type(self).__name__, self.id)



class ServerSpec(AppSpec):
    """
    An app this window runs itself, as a local server with no window of its
    own: OpenCode, a coding agent. `installed()` is whether its program is
    here, `launch()` starts it as a contained child of this process - so it
    ends when the window does, like a bridge - and the probe is its loopback
    port. What keeps it in check is its config, not a sandbox: it works in
    one folder, reads freely, and every edit, command and fetch waits for the
    user (see `opencode_config`) - or, with the user's Agentic switch on
    (`agentic`), is answered by the bridge when it stays inside a task's copy.
    """

    served = True

    def __init__(self, workspace, state_dir, **kw):
        AppSpec.__init__(self, exe_globs=[], **kw)
        self.workspace = workspace
        self.state_dir = state_dir            # config, password, log - never the workspace
        self.child = None
        self._agentic = None                  # the switch, once read (see `agentic`)

    @property
    def remote(self):
        return False

    def exe(self):
        return None                           # no window, no icon to read

    def program(self):
        return opencode_exe()

    def fit_window(self, host, model):
        """Make sure `model` is loaded with at least OPENCODE_CONTEXT tokens
        (or its maximum) and return the window it has. Loaded smaller - or
        not at all, when the coder is not the model in use - OpenCode
        compacts its task away. Only this model is reloaded; the rest stay."""
        loaded, maximum = context_window(host, model)
        if loaded is None and maximum is None:
            return None                       # a host that does not say
        want = min(OPENCODE_CONTEXT, maximum) if isinstance(maximum, int) else OPENCODE_CONTEXT
        if isinstance(loaded, int) and loaded >= want:
            return loaded
        for instance, _ in loaded_instances(host, model):
            if unload_model(host, instance):
                return loaded
        if load_model(host, model, context_length=want):
            return loaded
        return context_window(host, model)[0] or want

    def model_for(self, ids, shared):
        """The best coder the host has (`best_coder`), for this tab's chat and
        OpenCode alike; STUDIO_MODEL_OPENCODE still pins one."""
        if os.environ.get("STUDIO_MODEL_" + self.id.upper()):
            return AppSpec.model_for(self, ids, shared)
        coder = best_coder(ids or [])
        if not coder:
            return shared, ""                 # no coder on the host: the shared model, silently
        return coder, ("the best coding model on the host" if coder != shared else "")

    def installed(self):
        return self.program() is not None

    @property
    def config_path(self):
        return os.path.join(self.state_dir, "opencode.json")

    @property
    def lessons_path(self):
        """What the tabs learned, as markdown OpenCode reads through `instructions`
        (studio_lessons.write_brief_file) - Direct mode has no model of ours."""
        return os.path.join(self.state_dir, "lessons.md")

    @property
    def key_path(self):
        return os.path.join(self.state_dir, "server.key")

    @property
    def agentic_path(self):
        """The user's Agentic switch, where the bridge reads it on every look
        (apps/opencode/mcp.py `agentic`). A file, so it holds across restarts;
        written by the window only, so no tool - and no model - sets it."""
        return os.path.join(self.state_dir, OPENCODE_AGENTIC)

    def agentic(self):
        """Whether the user has Agentic on: in a task's copy OpenCode's edits
        and test runs go through without a card. Read from the file once and
        kept - this window is the only writer, and the header asks on every
        status change."""
        if self._agentic is None:
            try:
                with open(self.agentic_path, encoding="utf-8") as f:
                    self._agentic = json.load(f).get("on") is True
            except (OSError, ValueError, AttributeError):
                self._agentic = False
        return self._agentic

    def set_agentic(self, on):
        os.makedirs(self.state_dir, exist_ok=True)
        tmp = self.agentic_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"on": bool(on), "since": time.strftime("%Y-%m-%d %H:%M")}, f)
        os.replace(tmp, self.agentic_path)
        self._agentic = bool(on)

    def write_config(self, host, model, ids, context=None):
        os.makedirs(self.state_dir, exist_ok=True)
        # The general brief goes to every folder; this repo adds its own.
        brief = [b for b in (OPENCODE_BRIEF_ANY, OPENCODE_BRIEF if own_repo(self.workspace) else None,
                             self.lessons_path)
                 if b and os.path.isfile(b)]
        cfg = opencode_config(host, model, ids, context, addons=load_addons(self.state_dir),
                              brief=brief)
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        return self.config_path

    def launch(self, host=None, model=None):
        """
        Write OpenCode's config and a fresh password, and start `opencode
        serve` in the workspace. Returns once it is started; the caller polls
        `running()` like any other app.
        """
        exe = self.program()
        if not exe:
            raise RuntimeError("OpenCode is not installed on this PC. Install it once, in "
                               "a terminal: npm install -g opencode-ai - then press "
                               "Start OpenCode again.")
        if not os.path.isdir(self.workspace):
            raise RuntimeError("OpenCode's folder %s does not exist." % self.workspace)
        host = host or env_default("STUDIO_HOST", "AE_AGENT_HOST", fallback=DEFAULT_HOST)
        _, loaded, ids, _, _ = probe_models(host)
        shared = pick_model(loaded, ids, model or env_default("STUDIO_MODEL", "AE_AGENT_MODEL"))
        chosen, _ = self.model_for(ids, shared or model or DEFAULT_MODEL)
        context = self.fit_window(host, chosen) if chosen else None
        self.write_config(host, chosen, ids, context)
        self.stop()
        # A new password each start, readable by this user only: the bridge
        # reads it from here, and nothing else on the PC - a web page in a
        # browser least of all - can drive the server without it.
        key = base64.urlsafe_b64encode(os.urandom(24)).decode("ascii")
        with open(self.key_path, "w", encoding="utf-8") as f:
            f.write(key)
        env = dict(os.environ, OPENCODE_CONFIG=self.config_path,
                   OPENCODE_SERVER_PASSWORD=key)
        env.pop("OPENCODE_CONFIG_CONTENT", None)
        # The tests OpenCode runs leave no __pycache__ in a task's copy: in a
        # folder with no .gitignore they were checkpointed and would be merged.
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        if own_repo(self.workspace):
            env["OPENCODE_DISABLE_PROJECT_CONFIG"] = "1"   # not the 60k-token AGENTS.md
        port = urllib.parse.urlsplit(OPENCODE_URL).port or 4096
        log_file = open(os.path.join(self.state_dir, "server.log"), "w",
                        encoding="utf-8", errors="replace")
        try:
            self.child = studio_procs.spawn(
                [exe, "serve", "--hostname", "127.0.0.1", "--port", str(port)],
                cwd=self.workspace, env=env, stdin=subprocess.DEVNULL,
                stdout=log_file, stderr=subprocess.STDOUT, creationflags=NO_WINDOW)
        finally:
            log_file.close()                  # the child holds its own handle

    def stop(self):
        """End the server this window started, and everything it started."""
        child, self.child = self.child, None
        if child is not None:
            try:
                child.stop(3.0)
            except Exception:
                pass


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

COMFY_GROUPS = {
    "discover": ["comfy_status", "comfy_list_models", "comfy_queue", "comfy_history"],
    "generate": ["comfy_generate", "comfy_edit_image", "comfy_face_swap", "comfy_upscale",
                 "comfy_upload_image",
                 "comfy_wait", "comfy_fetch_output"],
    "control": ["comfy_interrupt", "comfy_clear_queue"],
    # Arbitrary graphs and the node catalogue: powerful, verbose, and easy for a
    # small model to get wrong. Off by default; switch the group on for a session
    # that really needs a custom workflow.
    "workflows": ["comfy_run_workflow", "comfy_search_nodes", "comfy_node_info"],
}

OPENCODE_GROUPS = {
    "discover": ["opencode_status", "opencode_list_sessions", "opencode_get_session",
                 "opencode_changes", "opencode_list_files", "opencode_read_file",
                 "opencode_grants", "opencode_search_files"],
    # Nothing here edits by itself: OpenCode asks the user before each change.
    "work": ["opencode_new_session", "opencode_ask", "opencode_wait", "opencode_abort",
             "opencode_merge", "opencode_undo", "opencode_discard", "opencode_revoke"],
}

PS_GROUPS = {
    "discover": ["ps_status", "ps_list_documents", "ps_get_document", "ps_get_layer",
                 "ps_screenshot"],
    "create": ["ps_new_document", "ps_open", "ps_add_text_layer", "ps_add_fill_layer",
               "ps_place_file"],
    "edit": ["ps_set_layer", "ps_move_layer", "ps_reorder_layer", "ps_duplicate_layer",
             "ps_delete_layer", "ps_adjust_layer", "ps_resize_image", "ps_resize_canvas",
             "ps_crop"],
    "files": ["ps_save", "ps_save_as", "ps_close_document"],
    # The escape hatch: Photoshop's whole scripting surface. On by default because
    # the coder model writes usable ExtendScript and the prompt tells it to prefer
    # the shaped tools; switch the group off for a session that should not improvise.
    "script": ["ps_run_jsx"],
}

AI_GROUPS = {
    "discover": ["ai_status", "ai_list_documents", "ai_get_document", "ai_list_items",
                 "ai_get_item", "ai_screenshot"],
    "create": ["ai_new_document", "ai_open", "ai_add_text", "ai_add_shape", "ai_place_file",
               "ai_add_layer", "ai_add_artboard"],
    "edit": ["ai_set_item", "ai_transform_item", "ai_reorder_item", "ai_duplicate_item",
             "ai_delete_item", "ai_set_layer", "ai_activate_artboard"],
    "files": ["ai_save", "ai_save_as", "ai_close_document"],
    "script": ["ai_run_jsx"],
}

PPRO_GROUPS = {
    "discover": ["ppro_status", "ppro_get_project", "ppro_get_sequence", "ppro_get_clip",
                 "ppro_screenshot", "ppro_list_presets"],
    "create": ["ppro_open_project", "ppro_new_project", "ppro_import_files", "ppro_create_bin",
               "ppro_new_sequence", "ppro_add_to_sequence", "ppro_add_marker"],
    "edit": ["ppro_set_clip", "ppro_remove_clip", "ppro_set_clip_property", "ppro_add_effect",
             "ppro_razor", "ppro_set_track", "ppro_set_sequence", "ppro_set_item",
             "ppro_delete_bin"],
    "files": ["ppro_save", "ppro_save_as", "ppro_export", "ppro_close_project"],
    "script": ["ppro_run_jsx"],
}

# The Chat tab's bridge: this PC's files and the web, every tool a read. Both
# groups are on by default; the prompt teaches all five tools.
# The panel inside Premiere listens here; apps/adobe/premiere.py and the panel's
# main.js both read STUDIO_PREMIERE_PORT, so one variable moves every end.
PREMIERE_PORT = os.environ.get("STUDIO_PREMIERE_PORT") or "7787"

RESOLVE_MCP = os.environ.get(
    "RESOLVE_MCP_DIR", os.path.join(os.path.expanduser("~"), "davinci-resolve-mcp"))
HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def opencode_exe():
    """OpenCode's program, or None. npm puts a native opencode.exe inside the
    package and a .cmd shim on PATH; the .exe is started directly, so no
    cmd.exe sits between this window and the server."""
    roots = [os.environ.get("OPENCODE_BIN", "")]
    appdata = os.environ.get("APPDATA")
    if appdata:
        roots.append(os.path.join(appdata, "npm", "node_modules", "opencode-ai", "bin",
                                  "opencode.exe"))
    roots.append(os.path.join(os.path.expanduser("~"), ".opencode", "bin", "opencode.exe"))
    for p in roots:
        if p and os.path.isfile(p):
            return p
    return shutil.which("opencode")


# What OpenCode may do without asking: read, search and plan. Everything that
# changes something - an edit, a command, a fetch, a subagent's own edits -
# asks, and the bridge puts each ask to the user. Outside its folder it is
# refused outright. The user's "Always allow" on a step is OpenCode's own,
# and lasts until the server restarts.
OPENCODE_PERMISSIONS = {
    "read": "allow", "glob": "allow", "grep": "allow", "list": "allow", "lsp": "allow",
    "todowrite": "allow", "question": "allow", "skill": "allow", "task": "allow",
    "edit": "ask", "bash": "ask", "webfetch": "ask", "websearch": "ask",
    "doom_loop": "ask", "external_directory": "deny",
}
# The user's Agentic switch, a file in the state folder (ServerSpec.agentic).
# It loosens nothing above: OpenCode still asks, and with the switch on the
# bridge answers what stays inside a task's copy instead of showing a card.
OPENCODE_AGENTIC = "agentic.json"


def load_addons(state_dir):
    import apps.opencode.codeaddons as studio_codeaddons
    return studio_codeaddons.load(state_dir)


def addons_config(addons):
    import apps.opencode.codeaddons as studio_codeaddons
    return studio_codeaddons.config(addons)


# OpenCode puts the workspace's AGENTS.md whole into every request. This
# repo's is ~60k tokens - more than the model's window - so the task and
# everything OpenCode read got compacted away. On this repo it gets this short
# brief instead, and `launch()` sets OPENCODE_DISABLE_PROJECT_CONFIG so the
# root AGENTS.md is not loaded too.
OPENCODE_CONTEXT = 65536   # the window OpenCode's model is loaded with, at least
OPENCODE_BRIEF = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "docs", "OPENCODE.md")
OPENCODE_BRIEF_ANY = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "docs", "OPENCODE_ANY.md")
# Our own OpenCode plugin: a read with no line range on a long file gets the
# first lines and a note to grep and read around the match (apps/opencode/read_cap.js).
OPENCODE_READ_CAP = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "apps", "opencode", "read_cap.js")
# Our own MCP server "repo": repo_map outlines a folder or file with line
# ranges, repo_find says where a name is defined (apps/opencode/repomap.py).
OPENCODE_REPO_MAP = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "apps", "opencode", "repomap.py")


def own_repo(workspace):
    here = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.normcase(os.path.abspath(workspace)) == os.path.normcase(here)


def opencode_config(host, model, ids, context=None, addons=None, brief=None):
    """
    The opencode.json handed to `opencode serve` (by OPENCODE_CONFIG, so the
    workspace is not written to): the studio's LM Studio with the served
    models declared and one chosen, the permissions above, the add-ons
    The user has turned on, and `brief` (a file path, or a list of them) as
    its instructions.
    """
    base = host.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    models = {m: {"name": m} for m in (ids or [model]) if m}
    if model and model not in models:
        models[model] = {"name": model}
    if model and context:
        # OpenCode compacts the conversation before it outgrows the window
        # only if it knows the window.
        models[model]["limit"] = {"context": int(context), "output": min(8192, int(context) // 4)}
    cfg = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {"lmstudio": {"npm": "@ai-sdk/openai-compatible", "name": "LM Studio",
                                  "options": {"baseURL": base}, "models": models}},
        "model": "lmstudio/%s" % model if model else None,
        "permission": dict(OPENCODE_PERMISSIONS),
        "autoupdate": False,
        "share": "disabled",
        "plugin": ["file:///" + OPENCODE_READ_CAP.replace(os.sep, "/")],
    }
    if brief:
        cfg["instructions"] = [brief] if isinstance(brief, str) else list(brief)
    extra = addons_config(addons or [])
    # An add-on's permissions only add to the ones above; none is loosened.
    for k, v in extra.pop("permission", {}).items():
        cfg["permission"].setdefault(k, v)
    cfg["plugin"] += extra.pop("plugin", [])
    # Ours first; an add-on of the same name does not replace it.
    cfg["mcp"] = dict(extra.pop("mcp", {}))
    cfg["mcp"]["repo"] = {"type": "local", "command": [sys.executable, OPENCODE_REPO_MAP],
                          "enabled": True}
    cfg.update(extra)
    return cfg


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
        readback=[("get_layer_full", ["compId", "layerId"]), ("get_comp", ["compId"])],
        review=("screenshot_frame", ["compId"]),
        docs=[("After Effects scripting guide (the object model run_jsx drives)",
               "https://ae-scripting.docsforadobe.dev/"),
              ("Expression reference and examples",
               "https://aereference.com/expressions")],
        craft=CRAFT_MOTION,
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
        docs=[("DaVinci Resolve scripting API reference (what this bridge wraps)",
               "https://resolvedevdoc.readthedocs.io/en/latest/")],
        craft=CRAFT_EDITING,
    ),
    AppSpec(
        id="comfyui",
        name="ComfyUI",
        tab="ComfyUI",
        code="Cf", fg="#EEFF44", bg="#1430DA",   # studio_icons.COMFY_*
        exe_globs=[],                        # remote: lives on the LLM PC
        probe="url:%s/system_stats" % COMFYUI_URL,
        command=sys.executable,
        args=[os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "apps", "comfyui", "mcp.py")],
        bridge_label=COMFYUI_URL.split("//", 1)[-1],
        groups=COMFY_GROUPS,
        default_groups=["discover", "generate", "control"],
        system_prompt=COMFY_PROMPT,
        examples=[
            "Which checkpoints are installed?",
            "Make a moody lighthouse at dusk, 1152x896, and show me four variations",
            "Same seed, but make the sky orange",
            "Turn the sketch in my Pictures folder into a painted version",
        ],
        launch_note="Start ComfyUI on the LLM PC with --listen so it accepts connections "
                    "from this machine, then click Start again to re-check.",
        # A 9B that calls tools, so the 30B can leave the GPU to the pictures.
        # qwen3-1.7b was tried first and talked about generating instead of
        # calling comfy_generate; the 9B, tested live on this briefing, wrote
        # a photographer's prompt and called it. Not served -> the shared model.
        models=["qwen3.5-9b-deepseek-v4-flash"],
        gpu_tools=["comfy_generate", "comfy_edit_image", "comfy_face_swap", "comfy_upscale",
                   "comfy_run_workflow"],
        makes_pictures=True,
        docs=[("ComfyUI documentation", "https://docs.comfy.org/"),
              ("ComfyUI workflow examples", "https://comfyanonymous.github.io/ComfyUI_examples/")],
        craft=CRAFT_IMAGES,
    ),
    ServerSpec(
        id="opencode",
        name="OpenCode",
        tab="OpenCode",
        code="Oc", fg="#F0F0F0", bg="#3B3B3B",
        workspace=OPENCODE_WORKSPACE,
        state_dir=OPENCODE_STATE,
        probe="port:%s" % (urllib.parse.urlsplit(OPENCODE_URL).port or 4096),
        command=sys.executable,
        args=[os.path.join(HERE, "apps", "opencode", "mcp.py")],
        bridge_label="server on %s" % OPENCODE_URL.split("//", 1)[-1],
        groups=OPENCODE_GROUPS,
        default_groups=["discover", "work"],
        system_prompt=OPENCODE_PROMPT,
        examples=[
            "Have OpenCode explain how a tab's bridge is started",
            "Add a Copy button to each folded tool-call row in core/chat.py",
            "What has OpenCode changed that is not committed yet?",
            "Run the OpenCode bridge's tests and fix what fails",
        ],
        launch_note="It works in this app's own folder, reads freely, and asks you "
                    "before every edit, command and fetch - unless you turn Agentic on.",
        docs=[("OpenCode documentation", "https://opencode.ai/docs/")],
    ),
    AppSpec(
        id="photoshop",
        name="Photoshop",
        tab="Photoshop",
        code="Ps", fg="#31A8FF", bg="#001E36",
        exe_globs=[r"C:\Program Files\Adobe\Adobe Photoshop *\Photoshop.exe"],
        probe="process:Photoshop.exe",
        command=sys.executable,
        args=[os.path.join(HERE, "apps", "adobe", "photoshop.py")],
        bridge_label="COM scripting",
        groups=PS_GROUPS,
        default_groups=["discover", "create", "edit", "files", "script"],
        system_prompt=PS_PROMPT,
        examples=[
            "What layers are in this document?",
            "Make a 1920x1080 document with a dark blue band along the bottom",
            "Add the title 'Method & Form' in white, 96px, near the top left",
            "Place the latest ComfyUI picture and show me how it looks",
        ],
        launch_note="Photoshop takes a moment to finish loading; the first tool call "
                    "after that is slower while the bridge attaches.",
        readback=[("ps_get_layer", ["layer_id"]), ("ps_get_document", [])],
        review=("ps_screenshot", []),
        docs=[("Photoshop scripting reference (the object model ps_run_jsx drives)",
               "https://theiviaxx.github.io/photoshop-docs/")],
        craft=CRAFT_DESIGN,
    ),
    AppSpec(
        id="illustrator",
        name="Illustrator",
        tab="Illustrator",
        code="Ai", fg="#FF9A00", bg="#330000",
        exe_globs=[r"C:\Program Files\Adobe\Adobe Illustrator *\Support Files\Contents\Windows\Illustrator.exe"],
        probe="process:Illustrator.exe",
        command=sys.executable,
        args=[os.path.join(HERE, "apps", "adobe", "illustrator.py")],
        bridge_label="COM scripting",
        groups=AI_GROUPS,
        default_groups=["discover", "create", "edit", "files", "script"],
        system_prompt=AI_PROMPT,
        examples=[
            "What's on the artboard?",
            "Make a 1080x1080 artboard with a centred orange circle and a title under it",
            "Change every red fill to #2255AA",
            "Export the active artboard as a PNG at 2x to my Desktop",
        ],
        launch_note="Illustrator takes a moment to finish loading; the first tool call "
                    "after that is slower while the bridge attaches.",
        readback=[("ai_get_item", ["uuid"]), ("ai_get_document", [])],
        review=("ai_screenshot", []),
        docs=[("Illustrator scripting guide (the object model ai_run_jsx drives)",
               "https://ai-scripting.docsforadobe.dev/")],
        craft=CRAFT_DESIGN,
    ),
    AppSpec(
        id="premiere",
        name="Premiere Pro",
        tab="Premiere",
        code="Pr", fg="#E979FF", bg="#2A0634",
        # The Beta first: it is the Premiere this studio cuts in, and its exe has
        # a different name from a release build's.
        exe_globs=[r"C:\Program Files\Adobe\Adobe Premiere Pro (Beta)\Adobe Premiere Pro (Beta).exe",
                   r"C:\Program Files\Adobe\Adobe Premiere Pro *\Adobe Premiere Pro.exe"],
        probe="port:%s" % PREMIERE_PORT,
        command=sys.executable,
        args=[os.path.join(HERE, "apps", "adobe", "premiere.py")],
        bridge_label="127.0.0.1:%s" % PREMIERE_PORT,
        groups=PPRO_GROUPS,
        default_groups=["discover", "create", "edit", "files", "script"],
        system_prompt=PPRO_PROMPT,
        examples=[
            "What's on the timeline?",
            "Import the clips in my Footage folder and build a sequence from them",
            "Cut at the playhead and drop the second half's opacity to 50%",
            "Export the active sequence as H.264 to my Desktop",
        ],
        launch_note="Premiere Pro also needs the Studio Assist Bridge panel under Window > "
                    "Extensions; `python apps/adobe/premiere.py --install-panel` installs it.",
        readback=[("ppro_get_clip", ["clip_id"]), ("ppro_get_sequence", [])],
        review=("ppro_screenshot", []),
        docs=[("Premiere Pro scripting guide (the object model ppro_run_jsx drives)",
               "https://ppro-scripting.docsforadobe.dev/")],
        craft=CRAFT_EDITING,
    ),
]

APPS_BY_ID = {a.id: a for a in APPS}
DEFAULT_APP = APPS[0].id


class ChatSpec(AppSpec):
    """
    A tab with no creative app behind it: the model, and a bridge that reads.

    It duck-types AppSpec - id, colours, prompt, examples - so `Session`, the tab
    strip and the transcript need no special case for it. `drivable` is False:
    there is nothing to find, probe or launch, and the sidebar must not count it
    as an app. `bridged` stays True: its tools are studio_research_mcp's - this
    PC's files and the web, read-only - and `connect()` runs that bridge in this
    process through `studio_mcp.Loopback`, so there is no subprocess to start
    and nothing to fail. `command`/`args` still name the script, so the harness
    can check it like any bridge written here.

    Deliberately NOT a member of APPS: DRIVABLE is derived from that list, and
    the sidebar must not advertise chat as something this agent can drive.
    """

    drivable = False
    research = False                      # its bridge is the research server

    def __init__(self):
        AppSpec.__init__(
            self, id="chat", name="Chat", tab="Chat",
            code="Ch", fg="#ecebe8", bg="#3f4a5a",
            exe_globs=[], probe="", command=sys.executable,
            args=[os.path.join(HERE, "apps", "research", "mcp.py")],
            bridge_label="files and the web", groups=RESEARCH_GROUPS,
            default_groups=["files", "web"],
            system_prompt=CHAT_PROMPT,
            examples=[
                "What frame rate should I finish this in?",
                "Find the brief in my Documents folder and summarise it",
                "Look up the current Frame.io upload limits and cite the page",
                "Talk me through how to stage a lower third before I build it",
            ])

    def exe(self):
        return None

    def installed(self):
        return True                       # nothing to install, nothing to find

    def running(self):
        return True                       # the tab is the whole of it

    def connect(self, quiet=True):
        import apps.research.mcp as studio_research_mcp
        return studio_mcp.Loopback(studio_research_mcp.SERVER)

    def chat_prompt(self, studio="", lessons=""):
        # No CHAT_SUFFIX: it briefs an app tab on its app and the other tabs,
        # and CHAT_PROMPT carries its own version of that. CHAT_RULES replaces
        # QUALITY_RULES, which are about edits this tab cannot make. No lookup
        # rules either: CHAT_PROMPT teaches the same tools as its own.
        return (self.system_prompt + CREATIVE_RULES + CHAT_RULES
                + studio_section(studio) + lessons_section(lessons))

    def cli_prompt(self, studio="", lessons=""):
        return self.chat_prompt(studio, lessons)

    def quality_rules(self):
        return CHAT_RULES

    def launch(self):
        raise RuntimeError("Chat has no application to start.")


CHAT = ChatSpec()


class PanelSpec(AppSpec):
    """
    A tab that holds another program's window instead of a conversation: no
    model, no bridge, no transcript, no composer. Milanote is the one - a web
    app with no API to drive, so the tab is a container for it and for files
    dropped onto it (apps/milanote/milanote.py).

    `panel` is the flag the GUI asks. It duck-types AppSpec like ChatSpec does
    so the tab strip and its menu need no special case, and it is not in APPS
    for the same reason chat is not: DRIVABLE must not count it.
    """

    drivable = False
    bridged = False
    research = False
    panel = True

    def __init__(self, id, name, tab, code, fg, bg, url, note):
        AppSpec.__init__(
            self, id=id, name=name, tab=tab, code=code, fg=fg, bg=bg,
            exe_globs=[], probe="", command="", args=[], bridge_label="web app",
            groups={}, default_groups=[], system_prompt="", examples=[],
            launch_note=note)
        self.url = url

    def exe(self):
        return None

    def installed(self):
        return True

    def running(self):
        return True

    def chat_prompt(self, studio="", lessons=""):
        return ""                         # nothing here talks to a model

    def cli_prompt(self, studio="", lessons=""):
        return ""

    def connect(self, quiet=True):
        raise RuntimeError("%s is a window in a tab, with no bridge to connect to."
                           % self.name)

    def launch(self):
        raise RuntimeError(self.launch_note)


MILANOTE = PanelSpec(
    id="milanote", name="Milanote", tab="Milanote", code="Mn",
    fg="#ffffff", bg="#2f3542",
    url=os.environ.get("MILANOTE_URL", "https://app.milanote.com/"),
    note="Milanote opens inside its tab, in a Chrome or Edge window of its own.")


class ImagesSpec(PanelSpec):
    """The Image Studio: a form over ComfyUI - person, style, scene,
    references, generate - with the graph built underneath
    (apps/image_studio/imagegen.py) and the tab itself in apps/image_studio/ui.py. A panel
    tab like Milanote (no model, no bridge, no composer), but what it holds
    is ours, so there is no window to start."""
    images = True


IMAGE_STUDIO = ImagesSpec(
    id="image-studio", name="Image Studio", tab="Image Studio", code="IS",
    fg="#ffffff", bg="#5b3cc4", url="",
    note="The Image Studio sends its work to the ComfyUI backends listed under Backends.")

class TerminalSpec(PanelSpec):
    """Console windows opened outside the app - ComfyUI's, a cmd started by
    hand - hidden from the desktop and mirrored here (core/consoles.py,
    core/terminals_ui.py). The app opens this tab itself when it takes one."""
    terminals = True

    def __init__(self, **kw):
        PanelSpec.__init__(self, **kw)
        self.bridge_label = "consoles"


TERMINALS = TerminalSpec(
    id="terminals", name="Terminal", tab="Terminal", code=">_",
    fg="#ecebe8", bg="#232321", url="",
    note="Console windows opened outside the app are held in the Terminal tab.")

# Everything that can be a tab, apps first. APPS stays the registry of drivable
# apps; TABS is what the tab strip and the new-tab menu offer.
TABS = APPS + [MILANOTE, IMAGE_STUDIO, TERMINALS, CHAT]
TABS_BY_ID = {a.id: a for a in TABS}
from core.agent_bridges import (
    slug, split_command, BridgeSpec, _label, group_by_prefix, add_bridge,
    _rederive_drivable, remove_bridge, bridge_from_record, load_bridges,
    custom_bridges, get_app, installed_apps, ADOBE_DIR, PRODUCTS, OTHER_APPS,
    DRIVABLE, detect_apps,
)



from core.agent_cli import (
    ask_at_terminal, elicit_fields, elicit_at_terminal, answer_text,
    run_agent, learn_from_run, converse, env_default, _interrupt, main,
)




if __name__ == "__main__":
    sys.exit(main())
