"""MCP stdio transport: the JSON-RPC client every bridge is built on,
plus the schema/tool-list conversions to and from it. Split out of
core/agent.py (docs/CODEMAP.md) - a leaf module, nothing here calls
back into the rest of the engine."""
import json
import os
import queue
import shutil
import subprocess
import threading
import time

import core.mcp as studio_mcp
import core.procs as studio_procs
import core.tablog as tablog

from core.agent import (
    log, LOG, BRIDGE_LOG, NO_WINDOW, PROGRESS_CAP, MAX_TOOL_RESULT_CHARS,
    MAX_TOOL_DESC_CHARS,
)


class Cancelled(RuntimeError):
    """The user pressed Stop while an MCPClient call was in flight."""


class MCPClient:
    """Minimal MCP stdio client: newline-delimited JSON-RPC over a child process.

    Requests are serialized - one in flight, its reply owned by one reader.
    What the server said at `initialize` (revision, serverInfo, instructions)
    is kept on the client; the harness's `check` command reports it, and the
    executor never needs it. Server notifications - logging, progress - are
    written to the log as they pass, so a bridge that waits on a render is
    seen to be waiting.
    """

    on_elicit = None                      # see __init__
    on_progress = None                    # f(text): a call's progress, for a UI
    _eliciting = 0

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
        self._send_lock = threading.Lock()
        self._inbox = queue.Queue()
        self.protocol_version = None
        self.server_info = {}
        self.instructions = ""
        self.capabilities = {}
        # The bridge asking the user something (MCP elicitation): set before
        # initialize, it is called on a thread of its own with the request's
        # params and returns the result - {"action", "content"}. Unset, the
        # client does not claim it can ask, and a bridge must do without.
        self.on_elicit = None
        self._eliciting = 0               # questions open; no call times out under one
        # Contained: the bridge and everything it starts (npx's node, a COM
        # worker) end with close(), or with this process however it ends.
        self.child = studio_procs.spawn(
            [exe] + list(args),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", bufsize=1, creationflags=NO_WINDOW,
        )
        self.proc = self.child.proc
        # The tab this bridge was started for: its stderr goes to that tab's
        # log, from a thread that would not otherwise know (studio_tablog).
        self.tab = tablog.current()
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
        # What a bridge says on stderr is its only diagnostic, and the shortcut's
        # `pythonw.exe` has no console to print it to: it goes to the log too.
        with tablog.working_for(self.tab):
            for line in self.proc.stderr:
                if line.strip():
                    log("  [mcp] " + line.rstrip(), self.quiet)
                    BRIDGE_LOG.info("%s", line.rstrip()[:2000])

    def _send(self, payload):
        with self._send_lock:
            self.proc.stdin.write(json.dumps(payload) + "\n")
            self.proc.stdin.flush()

    def request(self, method, params=None, timeout=180, cancel=None):
        # One reader owns each response; concurrent calls cannot steal replies.
        with self._request_lock:
            return self._request(method, params, timeout, cancel)

    def _request(self, method, params=None, timeout=180, cancel=None):
        with self._lock:
            self._id += 1
            rid = self._id
        params = dict(params or {})
        if method == "tools/call":
            # Ask for progress under our own id; a bridge that waits reports on it.
            params.setdefault("_meta", {})["progressToken"] = rid
        self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        # A bridge that reports progress is working, not hung: each report
        # restarts the timeout, up to PROGRESS_CAP in all. Without this a
        # ComfyUI render that took four minutes was abandoned at three.
        deadline = time.monotonic() + timeout
        cap = time.monotonic() + max(timeout, PROGRESS_CAP)
        while True:
            # The user's Stop, not the bridge's own pace: checked every poll
            # (below, at most 1s apart) so a call that would otherwise run to
            # its full timeout - opencode_ask can take 600s - ends promptly.
            if cancel is not None and cancel.is_set():
                self._cancel(rid, "user stop")
                raise Cancelled("the user pressed Stop")
            if self._eliciting:
                # A person is deciding; the bridge is not hung. The clock
                # starts again from their answer.
                now = time.monotonic()
                deadline = max(deadline, now + timeout)
                cap = max(cap, now + timeout)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                msg = self._inbox.get(timeout=min(remaining, 1.0))
            except queue.Empty:
                continue
            if msg.get("_closed"):
                self._inbox.put(msg)
                raise EOFError("The MCP bridge exited before returning a result.")
            if msg.get("method") and msg.get("id") is not None:
                self._server_request(msg)
                continue
            if msg.get("id") == rid:
                if "error" in msg:
                    raise RuntimeError("MCP error: " + studio_mcp.error_text(msg["error"]))
                return msg.get("result", {})
            if msg.get("id") is None and msg.get("method"):
                note = msg.get("params") or {}
                self._notification(msg["method"], note)
                if (msg["method"] == "notifications/progress"
                        and note.get("progressToken") == rid):
                    deadline = min(time.monotonic() + timeout, cap)
            # Late replies to timed-out serialized requests must not
            # accumulate forever in the inbox.
        self._cancel(rid, "timeout")
        raise TimeoutError("no MCP reply to %s in %ss" % (method, timeout)
                           if time.monotonic() < cap else
                           "no MCP reply to %s in %ss, though it reported progress"
                           % (method, int(max(timeout, PROGRESS_CAP))))

    def _cancel(self, rid, reason):
        # Tell the bridge we stopped listening; one built on studio_mcp stops
        # waiting too, and never replies to a request nobody owns.
        try:
            self._send({"jsonrpc": "2.0", "method": "notifications/cancelled",
                        "params": {"requestId": rid, "reason": reason}})
        except Exception:
            pass

    def _server_request(self, msg):
        """A request from the bridge. ping is answered here; an elicitation is
        handed to `on_elicit` on a thread, so this loop goes on reading the
        call's progress while the user decides."""
        rid, method = msg["id"], msg["method"]
        if method == "ping":
            self._send({"jsonrpc": "2.0", "id": rid, "result": {}})
            return
        if method != "elicitation/create" or self.on_elicit is None:
            self._send({"jsonrpc": "2.0", "id": rid, "error": {
                "code": studio_mcp.METHOD_NOT_FOUND, "message": "unknown method %s" % method}})
            return
        handler = self.on_elicit
        with self._lock:
            self._eliciting += 1

        def answer():
            try:
                res = handler(msg.get("params") or {})
                reply = {"jsonrpc": "2.0", "id": rid,
                         "result": res if isinstance(res, dict) else {"action": "cancel"}}
            except Exception as e:
                reply = {"jsonrpc": "2.0", "id": rid, "error": {
                    "code": studio_mcp.INTERNAL_ERROR, "message": "%s: %s" % (type(e).__name__, e)}}
            finally:
                with self._lock:
                    self._eliciting -= 1
            try:
                self._send(reply)
            except Exception:
                pass                        # the bridge has gone; nobody to tell

        threading.Thread(target=answer, daemon=True, name="mcp-elicit").start()

    def _notification(self, method, params):
        if method == "notifications/message":
            log("  [mcp %s] %s" % (params.get("level", "info"),
                                   json.dumps(params.get("data"))[:300]), self.quiet)
        elif method == "notifications/progress":
            text = params.get("message") or "%s/%s" % (params.get("progress"),
                                                        params.get("total", "?"))
            log("  [mcp] ... " + str(text)[:200], self.quiet)
            if self.on_progress is not None:
                try:
                    self.on_progress(str(text)[:200])
                except Exception:
                    pass                    # a display must not break the call

    def initialize(self, timeout=180):
        res = self.request("initialize", timeout=timeout, params={
            "protocolVersion": studio_mcp.LATEST,
            "capabilities": {"elicitation": {"form": {}}} if self.on_elicit else {},
            "clientInfo": {"name": "studio_agent", "version": "1.1"},
        })
        self.protocol_version = res.get("protocolVersion")
        self.server_info = res.get("serverInfo") or {}
        self.instructions = res.get("instructions") or ""
        self.capabilities = res.get("capabilities") or {}
        if self.protocol_version not in studio_mcp.PROTOCOL_VERSIONS:
            log("  [mcp] bridge speaks protocol %s, which this client does not know; "
                "continuing on tools/list and tools/call" % self.protocol_version, self.quiet)
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

    def call_tool(self, name, arguments, cancel=None):
        t0 = time.monotonic()
        try:
            res = self.request("tools/call", {"name": name, "arguments": arguments},
                               cancel=cancel)
        except Exception as e:
            LOG.warning("tool %s failed after %.1fs: %s", name, time.monotonic() - t0, e)
            raise
        LOG.info("tool %s %s %.1fs%s", name, json.dumps(arguments)[:300],
                 time.monotonic() - t0,
                 " ERROR" if isinstance(res, dict) and res.get("isError") else "")
        return res

    def close(self, grace=3.0):
        """End of input asks the bridge to exit; after `grace` seconds its
        whole process tree is ended, not just the process we started."""
        self.child.stop(grace)


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
            parts.append("[image returned: %s, %d bytes - a text model sees only the "
                         "visual review below, if there is one]"
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


class HostUnreachable(RuntimeError):
    """Nothing answered at the inference host: the tailnet, the other PC or
    LM Studio's server is down, or the connection timed out. Distinct from
    an HTTP error, which is the host answering with a problem, because the
    GUI treats the two differently: this one gets a Connect button."""


class ContextLimitError(RuntimeError):
    """No complete model response exists; safe to compact and retry once."""


