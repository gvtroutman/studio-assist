#!/usr/bin/env python3
"""
studio_mcp - the MCP harness: what a bridge written here is built on, and what
any bridge - ours or installed - is held to.

Two halves in one stdlib-only file, because the same facts underlie both: the
JSON-RPC framing, the protocol revisions, the tool contract, the schema
validator.

The server half. `Server` speaks the Model Context Protocol over newline-
delimited JSON-RPC 2.0 on stdio, the transport `studio_agent.MCPClient` uses:

  - negotiates the protocol revision (2025-11-25 down to 2024-11-05) rather than
    pinning one, so a newer client gets a newer contract and an older one is not
    refused;
  - validates every `tools/call` against the tool's own inputSchema before the
    handler runs, and answers protocol mistakes - unknown tool, bad arguments,
    parse error, JSON-RPC batch - with the JSON-RPC error the spec names, while a
    tool's own failure is an `isError` result the model can read as prose;
  - carries every tool annotation the executor reads (`readOnlyHint` decides
    which calls owe a read-back), structured output with an `outputSchema`,
    server->client logging, progress on the client's `progressToken`, and
    cancellation a long tool can poll with `cancelled()`;
  - keeps stdout for the protocol: a stray `print` inside a tool lands on stderr,
    where the client logs it, instead of in the middle of a reply.

`Loopback` gives `MCPClient`'s interface over a `Server` in this process - no
subprocess, no pipes - so the executor can be run end to end against a real
bridge in a test.

The check half. `check_tools()` walks a tool list the way the executor and the
inference host will, and says what will go wrong before a model finds out:
schemas the grammar converter rewrites or rejects, descriptions the request
budget cannot carry, read-only hints that contradict a tool's name, groups that
name tools the bridge does not have, tools no group exposes, prompts that teach
a tool the tab was never given. `check_live()` does that against a running
bridge - ours or an installed one - and can call its harmless reads.

    python studio_mcp.py check --app comfyui             # start the bridge, report
    python studio_mcp.py check --app after_effects --call # ...and call its reads
    python studio_mcp.py check --command npx --args -y some-mcp-server
    python studio_mcp.py snapshot --app resolve tests/contracts/resolve.json
    python studio_mcp.py check --app resolve --snapshot tests/contracts/resolve.json

A bridge built on `Server` gets `main()` for free: `--list-tools` prints its
contract, `--describe` dumps it as JSON, `--check` runs the checks on itself.
"""

import argparse
import base64
import json
import math
import os
import queue
import re
import sys
import threading
import time
import traceback

# Revisions this harness speaks, newest first. `initialize` echoes the client's
# revision when it is one of these and otherwise offers the newest; the client
# then decides whether it can live with that. Nothing here depends on a feature
# that arrived after 2024-11-05, so an old client loses nothing but the
# annotations it would not read anyway.
PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
LATEST = PROTOCOL_VERSIONS[0]

# JSON-RPC 2.0 error codes, and the one MCP adds for a request the server has
# not been initialized for.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

LOG_LEVELS = ("debug", "info", "notice", "warning", "error", "critical", "alert",
              "emergency")

# The executor's own limits, restated here so a check can be run in a bridge
# process without importing the engine. Keep them equal to studio_agent's.
MAX_TOOL_DESC_CHARS = 4000
MAX_TOOL_RESULT_CHARS = 8000
# The executor bounds a request by characters: the brief and the tool contract
# are fixed, and what is left of REQUEST_CHARS carries the conversation. Under
# CONVERSATION_CHARS a couple of tool exchanges (each capped at
# MAX_TOOL_RESULT_CHARS) no longer fit beside the question that caused them.
REQUEST_CHARS = 100000
CONVERSATION_CHARS = 3 * MAX_TOOL_RESULT_CHARS

# Tool names per the 2025-11-25 revision: 1..128 of [A-Za-z0-9_.-]. Older
# bridges may break this; the model still has to be able to say the name.
TOOL_NAME = re.compile(r"^[A-Za-z0-9_.\-]{1,128}$")


class JSONRPCError(Exception):
    """A protocol-level refusal: sent as a JSON-RPC error, never as a result."""

    def __init__(self, code, message, data=None):
        Exception.__init__(self, message)
        self.code, self.message, self.data = code, message, data

    def wire(self):
        err = {"code": self.code, "message": self.message}
        if self.data is not None:
            err["data"] = self.data
        return err


def error_text(err):
    """One line for an MCP error object, however the server shaped it."""
    if isinstance(err, dict):
        text = "%s (%s)" % (err.get("message", "error"), err.get("code", "?"))
        if err.get("data") not in (None, "", {}, []):
            text += ": " + json.dumps(err["data"])[:300]
        return text
    return str(err)


# ------------------------------------------------------------- validation

def validate(value, schema, root=None, path="arguments"):
    """Validate a value against a JSON Schema without relaxing it for inference.

    Supports local refs, combinators, tuple arrays, objects and scalar bounds.
    Unknown annotation and format keywords are left to the bridge. The executor
    validates every call with this before dispatch; a `Server` validates again
    on arrival, so a bridge written here holds its contract whoever calls it.
    """
    root = schema if root is None else root
    if schema is False:
        raise ValueError(path + " is not allowed")
    if schema is True:
        return
    if "$ref" in schema:
        ref = schema["$ref"]
        if not ref.startswith("#/"):
            raise ValueError("unsupported external schema reference: " + ref)
        target = root
        for part in ref[2:].split("/"):
            target = target[part.replace("~1", "/").replace("~0", "~")]
        validate(value, target, root, path)
    for key in ("allOf", "anyOf", "oneOf"):
        if key in schema:
            passed = 0
            for branch in schema[key]:
                try:
                    validate(value, branch, root, path)
                    passed += 1
                except ValueError:
                    pass
            if ((key == "allOf" and passed != len(schema[key])) or
                    (key == "anyOf" and not passed) or (key == "oneOf" and passed != 1)):
                raise ValueError(path + " does not match " + key)
    if "not" in schema:
        try:
            validate(value, schema["not"], root, path)
        except ValueError:
            pass
        else:
            raise ValueError(path + " matches a forbidden schema")
    if "if" in schema:
        try:
            validate(value, schema["if"], root, path)
            branch = "then"
        except ValueError:
            branch = "else"
        validate(value, schema.get(branch, True), root, path)
    numeric = type(value) in (int, float)
    finite = numeric and (isinstance(value, int) or math.isfinite(value))
    types = {"object": isinstance(value, dict), "array": isinstance(value, list),
             "string": isinstance(value, str), "boolean": isinstance(value, bool),
             "null": value is None,
             "number": finite,
             "integer": finite and value == int(value)}
    want = schema.get("type")
    if want and not any(types.get(t, False) for t in (want if isinstance(want, list) else [want])):
        raise ValueError(path + " must be " + str(want))

    # JSON equality must distinguish true from 1.
    def equal(a, b):
        if type(a) in (int, float) and type(b) in (int, float):
            return a == b
        if type(a) is not type(b):
            return False
        if isinstance(a, dict):
            return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
        if isinstance(a, list):
            return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
        return a == b
    if "enum" in schema and not any(equal(value, v) for v in schema["enum"]):
        raise ValueError(path + " must be one of " + str(schema["enum"]))
    if "const" in schema and not equal(value, schema["const"]):
        raise ValueError(path + " has an invalid constant")
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                raise ValueError(path + "." + key + " is required")
        props = schema.get("properties", {})
        for key, val in value.items():
            patterns = [s for p, s in schema.get("patternProperties", {}).items()
                        if re.search(p, key)]
            if key in props:
                validate(val, props[key], root, path + "." + key)
            for pattern in patterns:
                validate(val, pattern, root, path + "." + key)
            if key not in props and not patterns:
                validate(val, schema.get("additionalProperties", True), root, path + "." + key)
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", math.inf):
            raise ValueError(path + " has the wrong number of items")
        prefix = schema.get("prefixItems", [])
        items = schema.get("items", True)
        if isinstance(items, list):
            prefix, items = items, schema.get("additionalItems", True)
        for i, val in enumerate(value):
            validate(val, prefix[i] if i < len(prefix) else items, root, path + "[%d]" % i)
        if schema.get("uniqueItems") and any(equal(a, b) for i, a in enumerate(value) for b in value[i + 1:]):
            raise ValueError(path + " must contain unique items")
    if type(value) in (int, float):
        if not finite:
            raise ValueError(path + " must be finite")
        for key, invalid in (("minimum", lambda b: value < b), ("maximum", lambda b: value > b),
                             ("exclusiveMinimum", lambda b: value <= b),
                             ("exclusiveMaximum", lambda b: value >= b)):
            if key in schema and invalid(schema[key]):
                raise ValueError(path + " violates " + key)
        if "multipleOf" in schema and not math.isclose(value / schema["multipleOf"], round(value / schema["multipleOf"]), abs_tol=1e-9):
            raise ValueError(path + " violates multipleOf")
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0) or len(value) > schema.get("maxLength", math.inf):
            raise ValueError(path + " has an invalid length")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            raise ValueError(path + " does not match the required pattern")


def sample(schema, root=None, depth=0):
    """A value shaped like the schema: enough to prove the schema can be walked.

    Required properties only, the first enum member, the declared default or
    example when there is one. Not a fuzzer - one value that the validator can
    be run over so a schema with a broken `$ref` or a mistyped keyword fails in
    a check rather than under a model's call.
    """
    root = schema if root is None else root
    if not isinstance(schema, dict) or depth > 12:
        return None
    if "$ref" in schema and schema["$ref"].startswith("#/"):
        target = root
        for part in schema["$ref"][2:].split("/"):
            target = target[part.replace("~1", "/").replace("~0", "~")]
        return sample(target, root, depth + 1)
    for key in ("const", "default"):
        if key in schema:
            return schema[key]
    if schema.get("examples"):
        return schema["examples"][0]
    if schema.get("enum"):
        return schema["enum"][0]
    want = schema.get("type")
    if isinstance(want, list):
        want = want[0] if want else None
    if want is None and "properties" not in schema:
        # A bare combinator: the first branch is the shape.
        for key in ("anyOf", "oneOf", "allOf"):
            if schema.get(key):
                return sample(schema[key][0], root, depth + 1)
        want = "array" if "items" in schema else "string"
    if want in (None, "object"):
        # An object whose branches add requirements (`oneOf: [{required: [a]},
        # {required: [b]}]` beside `properties`): take the first branch's, and
        # all of allOf's, so the sample satisfies the outer schema too.
        props = dict(schema.get("properties", {}))
        required = list(schema.get("required", []))
        for key in ("anyOf", "oneOf"):
            branch = (schema.get(key) or [None])[0]
            if isinstance(branch, dict):
                required += branch.get("required", [])
                props.update(branch.get("properties", {}))
        for branch in schema.get("allOf", []):
            if isinstance(branch, dict):
                required += branch.get("required", [])
                props.update(branch.get("properties", {}))
        return {k: sample(props.get(k, {}), root, depth + 1) for k in dict.fromkeys(required)}
    if want == "array":
        n = max(schema.get("minItems", 0), len(schema.get("prefixItems", [])))
        prefix = schema.get("prefixItems", [])
        items = schema.get("items", {})
        return [sample(prefix[i] if i < len(prefix) else items, root, depth + 1) for i in range(n)]
    if want == "string":
        return "x" * max(1, schema.get("minLength", 0))
    if want == "integer":
        return int(schema.get("minimum", schema.get("exclusiveMinimum", -1) + 1))
    if want == "number":
        return float(schema.get("minimum", schema.get("exclusiveMinimum", -1) + 1))
    if want == "boolean":
        return False
    return None


# ------------------------------------------------------------- tool results

def text_block(text):
    return {"type": "text", "text": text}


def image_block(data, mime_type):
    """An image content block from raw bytes or already-encoded base64."""
    if isinstance(data, (bytes, bytearray)):
        data = base64.b64encode(bytes(data)).decode("ascii")
    return {"type": "image", "data": data, "mimeType": mime_type}


def result(text=None, blocks=(), error=False, structured=None):
    """A tools/call result. `blocks` are content blocks after the text."""
    content = ([text_block(text)] if text is not None else []) + list(blocks)
    res = {"content": content}
    if structured is not None:
        res["structuredContent"] = structured
        if not content:
            # Older clients read only `content`; the spec asks for a text twin.
            res["content"] = [text_block(json.dumps(structured))]
    if error:
        res["isError"] = True
    return res


# ------------------------------------------------------------------- tools

class Tool:
    """One tool: its contract on the wire and the function behind it.

    `fn(arguments)` returns a tools/call result dict (see `result()`), or raises.
    Exceptions of the classes in `Server.errors` are the bridge's own refusals
    and reach the model as an `isError` result in the bridge's words; anything
    else is a bug, reported as an error result naming the exception with the
    traceback on stderr, and the server keeps serving.

    Annotations follow the spec's defaults - a tool is assumed to write, to be
    destructive, not idempotent, and to touch the outside world - so a read
    must say so: `read_only=True` sets the three hints a read implies.
    """

    def __init__(self, name, fn, description, input_schema, title=None,
                 output_schema=None, read_only=False, destructive=None,
                 idempotent=None, open_world=None):
        self.name, self.fn = name, fn
        self.description = description
        self.input_schema = input_schema or {"type": "object", "properties": {}}
        self.title = title
        self.output_schema = output_schema
        self.read_only = bool(read_only)
        self.destructive = (not read_only) if destructive is None else bool(destructive)
        self.idempotent = read_only if idempotent is None else bool(idempotent)
        self.open_world = True if open_world is None else bool(open_world)

    def spec(self):
        t = {"name": self.name, "description": self.description,
             "inputSchema": self.input_schema,
             "annotations": {"readOnlyHint": self.read_only,
                             "destructiveHint": self.destructive,
                             "idempotentHint": self.idempotent,
                             "openWorldHint": self.open_world}}
        if self.title:
            t["title"] = self.title
            t["annotations"]["title"] = self.title
        if self.output_schema:
            t["outputSchema"] = self.output_schema
        return t


def tools_from_table(table, read_only=(), **hints):
    """`Tool`s from the `(name, fn, description, schema)` rows our bridges keep.

    `read_only` names the rows that change nothing; `hints` are per-tool
    overrides, `{"comfy_clear_queue": {"destructive": True}}`.
    """
    out = []
    for name, fn, desc, schema in table:
        kw = dict(hints.get(name, {}))
        kw.setdefault("read_only", name in read_only)
        out.append(Tool(name, fn, desc, schema, **kw))
    return out


# --------------------------------------------------------- request context

_ctx = threading.local()


def progress(message=None, done=None, total=None):
    """Report progress on the call in flight, if the client asked for it.

    A no-op unless the request carried `_meta.progressToken`; a tool that
    waits - on a render, on a coding agent - calls this in its loop and the
    client sees the wait moving rather than a silent bridge.
    """
    server, token = getattr(_ctx, "server", None), getattr(_ctx, "token", None)
    if server is None or token is None:
        return
    params = {"progressToken": token, "progress": done if done is not None else time.time()}
    if total is not None:
        params["total"] = total
    if message:
        params["message"] = message
    server.notify("notifications/progress", params)


def cancelled():
    """True once the client cancelled the call in flight. Long tools poll it."""
    server, rid = getattr(_ctx, "server", None), getattr(_ctx, "rid", None)
    return server is not None and rid is not None and rid in server._cancelled


def log(level, data, logger=None):
    """Send a log message to the client, subject to the level it asked for."""
    server = getattr(_ctx, "server", None)
    if server is not None:
        server.log(level, data, logger)


# ------------------------------------------------------------------ server

class Server:
    """A Model Context Protocol server over newline-delimited JSON-RPC.

    `handle(msg)` is the whole protocol as a pure function - a message in, a
    reply (or None for a notification) out - so a test can drive it without
    stdio and `Loopback` can drive it without a subprocess. `serve()` puts it on
    a stream pair, reads on a thread so cancellations arrive while a tool runs,
    and writes replies and notifications under one lock.
    """

    def __init__(self, name, version, tools, instructions="", title=None,
                 errors=(), page_size=None, logger=None):
        self.name, self.version, self.title = name, version, title
        self.instructions = instructions
        self.tools = list(tools)
        self.by_name = {t.name: t for t in self.tools}
        if len(self.by_name) != len(self.tools):
            raise ValueError("duplicate tool names in " + name)
        self.errors = tuple(errors)
        self.page_size = page_size            # None: the whole list in one page
        self.logger = logger or name
        self.protocol = None                  # negotiated on initialize
        self.client = {}                      # the client's clientInfo
        self.initialized = False
        self.level = None                     # client's logging/setLevel; None: quiet
        self.sink = None                      # where notifications go, set by serve/Loopback
        self._inflight = set()                # request ids being handled right now
        self._cancelled = set()

    # ------------------------------------------------------------ outgoing

    def notify(self, method, params=None):
        if self.sink is not None:
            self.sink({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def log(self, level, data, logger=None):
        if level not in LOG_LEVELS:
            raise ValueError("unknown log level %r" % level)
        if self.level is None or LOG_LEVELS.index(level) < LOG_LEVELS.index(self.level):
            return
        self.notify("notifications/message",
                    {"level": level, "logger": logger or self.logger, "data": data})

    # ------------------------------------------------------------ protocol

    def initialize_result(self):
        info = {"name": self.name, "version": self.version}
        if self.title:
            info["title"] = self.title
        res = {"protocolVersion": self.protocol or LATEST,
               "capabilities": {"tools": {"listChanged": False}, "logging": {}},
               "serverInfo": info}
        if self.instructions:
            res["instructions"] = self.instructions
        return res

    def tool_list(self, cursor=None):
        specs = [t.spec() for t in self.tools]
        if not self.page_size:
            return {"tools": specs}
        try:
            start = int(cursor) if cursor is not None else 0
            if start < 0 or start > len(specs) or (cursor is not None and start % self.page_size):
                raise ValueError
        except (TypeError, ValueError):
            raise JSONRPCError(INVALID_PARAMS, "unknown cursor %r" % (cursor,))
        page = {"tools": specs[start:start + self.page_size]}
        if start + self.page_size < len(specs):
            page["nextCursor"] = str(start + self.page_size)
        return page

    def call_tool(self, name, arguments, rid=None, token=None):
        """Run one tool: contract first, then the handler, then the output.

        Unknown tool and arguments the schema refuses are protocol errors
        (-32602) - the executor validated before calling, so either means a
        caller that skipped it. Everything the handler does wrong is a result.
        """
        tool = self.by_name.get(name)
        if tool is None:
            raise JSONRPCError(INVALID_PARAMS, "Unknown tool: %s" % name,
                               {"tools": sorted(self.by_name)})
        arguments = {} if arguments is None else arguments
        if not isinstance(arguments, dict):
            raise JSONRPCError(INVALID_PARAMS, "arguments must be an object")
        try:
            validate(arguments, tool.input_schema)
        except ValueError as e:
            raise JSONRPCError(INVALID_PARAMS, "Invalid arguments for %s: %s" % (name, e))
        except (KeyError, TypeError) as e:
            raise JSONRPCError(INTERNAL_ERROR, "%s has a schema that cannot be checked: %r" % (name, e))

        _ctx.server, _ctx.rid, _ctx.token = self, rid, token
        real_stdout = sys.stdout
        sys.stdout = sys.stderr         # a print inside the tool must not reach the wire
        try:
            res = tool.fn(arguments)
        except self.errors as e:
            return result(str(e), error=True)
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            return result("%s failed inside the bridge: %s: %s" % (name, type(e).__name__, e),
                          error=True)
        finally:
            sys.stdout = real_stdout
            _ctx.server = _ctx.rid = _ctx.token = None
        return self._finish(tool, res)

    def _finish(self, tool, res):
        if not isinstance(res, dict) or not isinstance(res.get("content"), list):
            return result("%s returned something that is not a tool result: %r"
                          % (tool.name, res), error=True)
        if tool.output_schema and not res.get("isError"):
            structured = res.get("structuredContent")
            if structured is None:
                return result("%s declares an outputSchema but returned no structuredContent"
                              % tool.name, error=True)
            try:
                validate(structured, tool.output_schema, path="structuredContent")
            except ValueError as e:
                return result("%s returned output its schema refuses: %s" % (tool.name, e),
                              error=True)
            if not res["content"]:
                res["content"] = [text_block(json.dumps(structured))]
        return res

    def handle(self, msg):
        """One message in, one reply out - or None when nothing is owed."""
        if isinstance(msg, list):
            # Batching left the spec in 2025-06-18; answer the batch, not each item.
            return self._error(None, INVALID_REQUEST, "JSON-RPC batches are not supported")
        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
            return self._error(msg.get("id") if isinstance(msg, dict) else None,
                               INVALID_REQUEST, "not a JSON-RPC 2.0 message")
        rid, method = msg.get("id"), msg.get("method")
        if method is None:
            return None                     # a response to a request we never send
        params = msg.get("params")
        if params is None:
            params = {}
        if not isinstance(params, dict):
            return self._error(rid, INVALID_PARAMS, "params must be an object")
        if rid is None:
            self._notification(method, params)
            return None
        self._inflight.add(rid)
        try:
            res = self._request(rid, method, params)
            if rid in self._cancelled:
                return None                 # cancelled: the client stopped listening
        except JSONRPCError as e:
            return {"jsonrpc": "2.0", "id": rid, "error": e.wire()}
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            return self._error(rid, INTERNAL_ERROR, "%s: %s" % (type(e).__name__, e))
        finally:
            self._inflight.discard(rid)
            self._cancelled.discard(rid)
        return {"jsonrpc": "2.0", "id": rid, "result": res}

    def _request(self, rid, method, params):
        if method == "ping":
            return {}
        if method == "initialize":
            asked = params.get("protocolVersion")
            self.protocol = asked if asked in PROTOCOL_VERSIONS else LATEST
            self.client = params.get("clientInfo") or {}
            return self.initialize_result()
        if self.protocol is None:
            raise JSONRPCError(INVALID_REQUEST, "initialize first")
        if method == "tools/list":
            return self.tool_list(params.get("cursor"))
        if method == "tools/call":
            name = params.get("name")
            if not isinstance(name, str):
                raise JSONRPCError(INVALID_PARAMS, "tools/call needs a tool name")
            token = (params.get("_meta") or {}).get("progressToken")
            return self.call_tool(name, params.get("arguments"), rid, token)
        if method == "logging/setLevel":
            level = params.get("level")
            if level not in LOG_LEVELS:
                raise JSONRPCError(INVALID_PARAMS, "unknown log level %r" % (level,))
            self.level = level
            return {}
        raise JSONRPCError(METHOD_NOT_FOUND, "unknown method %s" % method)

    def _notification(self, method, params):
        if method == "notifications/initialized":
            self.initialized = True
        elif method == "notifications/cancelled":
            self.cancel(params.get("requestId"))
        # Anything else is a client's business we do not share in.

    def cancel(self, rid):
        """Mark a request cancelled. One that already finished is let be, as
        the spec asks; its reply may already be on the wire."""
        if rid in self._inflight:
            self._cancelled.add(rid)

    @staticmethod
    def _error(rid, code, message):
        return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}}

    # --------------------------------------------------------------- stdio

    def serve(self, inp=None, out=None):
        """Serve until the input closes. Defaults to stdin/stdout, made UTF-8."""
        own_stdio = inp is None and out is None
        inp = inp or sys.stdin
        out = out or sys.stdout
        if own_stdio:
            for stream in (inp, out):
                if hasattr(stream, "reconfigure"):
                    stream.reconfigure(encoding="utf-8", newline="\n")
        write_lock = threading.Lock()

        def send(msg):
            with write_lock:
                out.write(json.dumps(msg) + "\n")
                out.flush()

        self.sink = send
        inbox = queue.Queue()

        def read():
            # Cancellations are acted on here, while the main thread may be
            # deep in a tool; everything else waits its turn.
            try:
                for line in inp:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                    except ValueError:
                        inbox.put(("reply", self._error(None, PARSE_ERROR, "parse error")))
                        continue
                    if isinstance(msg, dict) and msg.get("method") == "notifications/cancelled":
                        self.cancel((msg.get("params") or {}).get("requestId"))
                        continue
                    inbox.put(("msg", msg))
            finally:
                inbox.put(None)

        threading.Thread(target=read, daemon=True, name="mcp-reader").start()
        try:
            while True:
                item = inbox.get()
                if item is None:
                    return
                kind, msg = item
                reply = msg if kind == "reply" else self.handle(msg)
                if reply is not None:
                    send(reply)
        finally:
            self.sink = None


class Loopback:
    """`MCPClient`'s interface over a `Server` in this process.

    The executor, the checks and the tests can run against a real bridge with
    no subprocess and no pipes. Notifications the server sends land in
    `notifications`.
    """

    def __init__(self, server):
        self.server = server
        self.notifications = []
        self.server.sink = self.notifications.append
        self._id = 0
        self.protocol_version = None
        self.server_info = {}
        self.instructions = ""
        self.capabilities = {}

    def request(self, method, params=None, timeout=None):
        self._id += 1
        reply = self.server.handle({"jsonrpc": "2.0", "id": self._id, "method": method,
                                    "params": params or {}})
        if reply is None:
            raise TimeoutError("no MCP reply to %s: the request was cancelled" % method)
        if "error" in reply:
            raise RuntimeError("MCP error: " + error_text(reply["error"]))
        return reply.get("result", {})

    def initialize(self, timeout=None):
        res = self.request("initialize", {"protocolVersion": LATEST, "capabilities": {},
                                          "clientInfo": {"name": "studio_mcp.Loopback",
                                                         "version": "1.0"}})
        self.server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"})
        self.protocol_version = res.get("protocolVersion")
        self.server_info = res.get("serverInfo") or {}
        self.instructions = res.get("instructions") or ""
        self.capabilities = res.get("capabilities") or {}
        return res

    def list_tools(self, timeout=None):
        tools, cursor = [], None
        while True:
            res = self.request("tools/list", {"cursor": cursor} if cursor else {})
            tools.extend(res.get("tools", []))
            cursor = res.get("nextCursor")
            if not cursor:
                return tools

    def call_tool(self, name, arguments):
        return self.request("tools/call", {"name": name, "arguments": arguments,
                                           "_meta": {"progressToken": self._id + 1}})

    def close(self):
        self.server.sink = None


# ------------------------------------------------------------------ checks

class Finding:
    LEVELS = ("error", "warn", "info")

    def __init__(self, level, where, text):
        assert level in self.LEVELS
        self.level, self.where, self.text = level, where, text

    def __repr__(self):
        return "%s %s: %s" % (self.level.upper(), self.where, self.text)


def _readonly_by_name(name, args, spec):
    """What the executor infers when it has to. Mirrors studio_tasks.readonly."""
    return (spec.get("annotations", {}).get("readOnlyHint") is True or
            name.startswith(("get_", "list_", "find_", "screenshot_")) or
            name in ("check_setup", "ae_guide", "diff_comp", "snapshot_comp"))


READ_NAMES = ("get_", "list_", "find_", "is_", "read_", "search_", "status", "check_")
WRITE_NAMES = ("set_", "add_", "create_", "delete_", "remove_", "put_", "clear_", "run_")

# Keywords LM Studio's grammar converter (and llama.cpp's, which it wraps) does
# not turn into grammar. They are not errors - the executor still validates
# them before a call - but the model is not steered by them, so a prompt that
# relies on one is relying on nothing.
UNSTEERED = ("if", "then", "else", "not", "patternProperties", "dependentSchemas",
             "dependentRequired", "unevaluatedProperties", "unevaluatedItems",
             "propertyNames", "contains", "minContains", "maxContains")


def _walk(node, seen=None):
    if isinstance(node, dict):
        for k, v in node.items():
            yield k, v
            for kv in _walk(v):
                yield kv
    elif isinstance(node, list):
        for v in node:
            for kv in _walk(v):
                yield kv


def check_tools(tools, groups=None, default_groups=None, prompt=None,
                sanitize=None, readonly=None, request_chars=REQUEST_CHARS):
    """Findings for a tool list, judged the way the executor and host will.

    `groups` / `default_groups` / `prompt` are the registry entry's, when the
    list is being checked against one. `sanitize` is the engine's
    `sanitize_schema`, and `readonly` its `readonly()`; both are passed in so a
    bridge process can run the schema-level checks without importing the
    engine.
    """
    out = []
    add = lambda level, where, text: out.append(Finding(level, where, text))
    readonly = readonly or _readonly_by_name
    names = [t.get("name") for t in tools]
    unhinted = []
    if not tools:
        add("error", "-", "the bridge exposes no tools")
    for name in {n for n in names if names.count(n) > 1}:
        add("error", name, "listed more than once")

    for t in tools:
        name = t.get("name")
        where = str(name)
        if not isinstance(name, str) or not TOOL_NAME.match(name):
            add("error", where, "name is not 1..128 of [A-Za-z0-9_.-]")
        desc = (t.get("description") or "").strip()
        if not desc:
            add("error", where, "no description; the model has nothing to choose it by")
        elif len(desc) > MAX_TOOL_DESC_CHARS:
            add("warn", where, "description is %d chars; the executor budgets %d per tool"
                % (len(desc), MAX_TOOL_DESC_CHARS))
        schema = t.get("inputSchema")
        if not isinstance(schema, dict):
            add("error", where, "inputSchema missing")
            continue
        if schema.get("type") not in ("object", None):
            add("error", where, "inputSchema must describe an object, not %r" % schema.get("type"))
        if "properties" not in schema and schema.get("type") == "object" and \
                not any(k in schema for k in ("$ref", "allOf", "anyOf", "oneOf")):
            add("info", where, "takes no arguments")
        try:
            validate(sample(schema), schema)
        except ValueError as e:
            add("info", where, "the schema refuses its own sample (%s); usually a pattern or a bound" % e)
        except Exception as e:
            add("error", where, "the executor cannot validate this schema: %s: %s" % (type(e).__name__, e))
        keys = {k for k, _ in _walk(schema)}
        unsteered = sorted(keys & set(UNSTEERED))
        if unsteered:
            add("info", where, "uses %s, which the grammar converter does not enforce" % ", ".join(unsteered))
        if any(k == "$ref" and isinstance(v, str) and not v.startswith("#/") for k, v in _walk(schema)):
            add("error", where, "an external $ref; the executor refuses those")
        if sanitize is not None:
            clean = sanitize(schema)
            changed = sorted(({k for k, _ in _walk(schema)} ^ {k for k, _ in _walk(clean)}) - {"$schema"})
            if changed:
                add("info", where, "rewritten for the grammar converter (%s)" % ", ".join(changed))
            if any(k == "items" and v is False for k, v in _walk(clean)):
                add("error", where, "'items': false survives sanitizing - LM Studio will 400 the request")
        out_schema = t.get("outputSchema")
        if out_schema is not None and (not isinstance(out_schema, dict) or out_schema.get("type") != "object"):
            add("error", where, "outputSchema must describe an object")

        ann = t.get("annotations") or {}
        ro = ann.get("readOnlyHint")
        if ro is True and ann.get("destructiveHint") is True:
            add("error", where, "annotated both read-only and destructive")
        if ro is None:
            unhinted.append(name)
            if name.startswith(READ_NAMES) and not readonly(name, {}, t):
                add("warn", where, "no readOnlyHint and the executor does not read the name as a "
                                   "read; every call will owe a read-back")
        elif ro is False and name.startswith(READ_NAMES) and "action" not in (schema.get("properties") or {}):
            add("warn", where, "named like a read but not annotated read-only; every call will owe a read-back")
        elif ro is True and name.startswith(WRITE_NAMES):
            add("warn", where, "named like a write but annotated read-only; the executor will skip its read-back")

    if unhinted:
        reads = [n for n in unhinted if readonly(n, {}, {})]
        add("info", "-", "%d of %d tools carry no readOnlyHint; the executor infers %d reads from "
                         "their names and treats the rest as writes" % (len(unhinted), len(tools), len(reads)))

    by_name = {t.get("name"): t for t in tools}
    if groups is not None:
        exposed_by_group = set()
        for gname, members in groups.items():
            if not members:
                add("error", "group " + gname, "empty")
            for m in members:
                if m not in by_name:
                    add("error", "group " + gname, "names %s, which the bridge does not expose" % m)
                exposed_by_group.add(m)
        for name in by_name:
            if name not in exposed_by_group:
                add("warn", name, "no group exposes it; the model can never call it")
        if default_groups is not None:
            missing = [g for g in default_groups if g not in groups]
            for g in missing:
                add("error", "default_groups", "names group %s, which does not exist" % g)
            working = [n for g in default_groups if g in groups for n in groups[g] if n in by_name]
            working = list(dict.fromkeys(working))
            if prompt:
                taught = {n for n in re.findall(r"\b[a-z][a-z0-9_]{3,}\b", prompt) if n in by_name}
                for n in sorted(taught - set(working)):
                    add("error", "prompt", "teaches %s, which is not in a default group" % n)
            # The executor's own arithmetic: the brief and the contract, as
            # the host sees them, come off the request; the rest is conversation.
            contract = len(json.dumps([{"type": "function", "function": {
                "name": n, "description": (by_name[n].get("description") or "").strip(),
                "parameters": (sanitize or (lambda x: x))(by_name[n].get("inputSchema") or {})}}
                for n in working]))
            brief = len(json.dumps([{"role": "system", "content": prompt or ""}]))
            left = request_chars - brief - contract
            text = ("%d tools: contract %d chars, brief %d, leaving %d of %d for conversation"
                    % (len(working), contract, brief, left, request_chars))
            if left < 0:
                add("error", "default groups", text + " - the executor will refuse every request")
            elif left < CONVERSATION_CHARS:
                add("warn", "default groups", text + " - under %d; a few tool results will "
                                                     "push the question out" % CONVERSATION_CHARS)
            else:
                add("info", "default groups", text)
    return out


def report(findings, out=None):
    """Print findings by level; True when none is an error."""
    out = out or sys.stdout
    for level in Finding.LEVELS:
        for f in findings:
            if f.level == level:
                out.write("  %-5s %-22s %s\n" % (level.upper(), f.where, f.text))
    counts = {lv: sum(1 for f in findings if f.level == lv) for lv in Finding.LEVELS}
    out.write("  %d error(s), %d warning(s), %d note(s)\n"
              % (counts["error"], counts["warn"], counts["info"]))
    return counts["error"] == 0


def check_live(client, app=None, call=False, out=None, sanitize=None, readonly=None):
    """Handshake with a bridge through `client`, then check what it exposes.

    `client` is an `MCPClient` or a `Loopback`, not yet initialized. `app` is
    the registry entry to judge it against, if any. With `call`, every tool
    annotated read-only that needs no arguments is called once and its
    answer's shape reported - the bridge's live path, on its harmless reads.
    """
    out = out or sys.stdout
    t0 = time.monotonic()
    init = client.initialize()
    out.write("  server     %s %s\n" % ((init.get("serverInfo") or {}).get("name", "?"),
                                        (init.get("serverInfo") or {}).get("version", "")))
    out.write("  protocol   %s%s\n" % (init.get("protocolVersion"),
                                       "" if init.get("protocolVersion") in PROTOCOL_VERSIONS
                                       else "  (not a revision this harness knows)"))
    out.write("  caps       %s\n" % ", ".join(sorted(init.get("capabilities") or {})))
    if init.get("instructions"):
        first = init["instructions"].strip().splitlines()[0]
        out.write("  instructs  %s (%d chars)\n" % (first[:80], len(init["instructions"])))
    tools = client.list_tools()
    out.write("  tools      %d in %.1fs\n" % (len(tools), time.monotonic() - t0))
    findings = check_tools(tools, groups=getattr(app, "groups", None),
                           default_groups=getattr(app, "default_groups", None),
                           prompt=getattr(app, "system_prompt", None),
                           sanitize=sanitize, readonly=readonly)
    if call:
        for t in tools:
            ann = t.get("annotations") or {}
            if ann.get("readOnlyHint") is not True or (t.get("inputSchema") or {}).get("required"):
                continue
            t1 = time.monotonic()
            try:
                res = client.call_tool(t["name"], {})
            except Exception as e:
                findings.append(Finding("error", t["name"], "call raised %s" % e))
                continue
            took = time.monotonic() - t1
            blocks = res.get("content") or []
            kinds = ", ".join(sorted({b.get("type", "?") for b in blocks})) or "no content"
            if res.get("isError"):
                text = next((b.get("text", "") for b in blocks if b.get("type") == "text"), "")
                findings.append(Finding("warn", t["name"], "isError in %.1fs: %s" % (took, text[:120])))
            else:
                findings.append(Finding("info", t["name"], "ok in %.1fs (%s)" % (took, kinds)))
    return tools, findings


# --------------------------------------------------------------------- CLI

def main(server, argv=None):
    """The command line every bridge built on `Server` gets."""
    argv = sys.argv[1:] if argv is None else argv
    if "--list-tools" in argv:
        for t in server.tools:
            print("%-24s %s" % (t.name, t.description.split(". ")[0]))
        return 0
    if "--describe" in argv:
        print(json.dumps({"initialize": server.initialize_result(),
                          "tools": [t.spec() for t in server.tools]}, indent=2))
        return 0
    if "--check" in argv:
        print("%s %s" % (server.name, server.version))
        return 0 if report(check_tools([t.spec() for t in server.tools])) else 1
    server.serve()
    return 0


def _cli(argv=None):
    p = argparse.ArgumentParser(prog="studio_mcp", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("check", "snapshot"):
        s = sub.add_parser(name)
        s.add_argument("--app", help="registry id (after-effects, resolve, comfyui, opencode, photoshop, illustrator)")
        s.add_argument("--command", help="start this bridge instead of a registry entry")
        s.add_argument("--args", nargs=argparse.REMAINDER, default=[])
        if name == "check":
            s.add_argument("--snapshot", help="check this recorded tool list instead of a live bridge")
            s.add_argument("--call", action="store_true", help="also call the harmless reads")
            s.add_argument("--in-process", action="store_true",
                           help="import our own bridge and run it here, no subprocess")
        else:
            s.add_argument("path", help="where to write the tool list")
    a = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")   # a bridge's instructions may not be cp1252

    import studio_agent as eng
    import studio_tasks
    app = None
    if a.app:
        app = eng.APPS_BY_ID.get(a.app)
        if app is None:
            p.error("no app %r; one of %s" % (a.app, ", ".join(eng.APPS_BY_ID)))
    if a.cmd == "check" and a.snapshot:
        with open(a.snapshot, encoding="utf-8") as fh:
            snap = json.load(fh)
        print("%s  (snapshot %s, recorded %s)" % (a.snapshot, snap.get("server", {}).get("name", "?"),
                                                  snap.get("recorded", "?")))
        findings = check_tools(snap["tools"], groups=getattr(app, "groups", None),
                               default_groups=getattr(app, "default_groups", None),
                               prompt=getattr(app, "system_prompt", None),
                               sanitize=eng.sanitize_schema, readonly=studio_tasks.readonly)
        return 0 if report(findings) else 1

    if a.command:
        client = eng.MCPClient(a.command, a.args, quiet=True)
    elif app is None:
        p.error("--app or --command is required")
    elif getattr(a, "in_process", False):
        client = Loopback(_import_bridge(app))
    else:
        client = eng.MCPClient(app.command, app.args, quiet=True)
    print("%s  (%s)" % (app.name if app else a.command,
                        "in process" if isinstance(client, Loopback) else "subprocess"))
    try:
        tools, findings = check_live(client, app, call=getattr(a, "call", False),
                                     sanitize=eng.sanitize_schema, readonly=studio_tasks.readonly)
        if a.cmd == "snapshot":
            snap = {"recorded": time.strftime("%Y-%m-%d"), "app": a.app,
                    "server": client.server_info, "protocolVersion": client.protocol_version,
                    "instructions": client.instructions, "tools": tools}
            with open(a.path, "w", encoding="utf-8") as fh:
                json.dump(snap, fh, indent=1)
            print("  wrote %d tools to %s" % (len(tools), a.path))
        return 0 if report(findings) else 1
    finally:
        client.close()


def _import_bridge(app):
    """The `Server` of one of our own bridges, from the script the registry names."""
    import importlib.util
    script = next((x for x in app.args if x.endswith(".py")), None)
    if not script or not os.path.isfile(script):
        raise SystemExit("%s is not a bridge written here; drop --in-process" % app.name)
    spec = importlib.util.spec_from_file_location(os.path.splitext(os.path.basename(script))[0], script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SERVER


if __name__ == "__main__":
    sys.exit(_cli())
