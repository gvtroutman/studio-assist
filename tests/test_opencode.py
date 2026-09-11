"""The OpenCode bridge, against a fake OpenCode server and a temp workspace.

No network and no Docker: `urllib.request.urlopen` is replaced with an
in-memory server answering the routes the bridge uses, and the workspace is a
temp folder. What these prove is the contract the registry entry and the
prompt rely on - that the bridge's tools are the registry's groups, that reads
are annotated as reads, that an ask returns what OpenCode said and did, that
the file tools cannot leave the workspace, that OpenCode's failures reach the
model as prose, and that the stdio server speaks what MCPClient expects.
"""
import io
import json
import os
import shutil
import tempfile
import unittest
import urllib.error
import urllib.request

import studio_agent as eng
import studio_opencode_mcp as oc
import studio_tasks as tasks


class FakeOpenCode:
    """Just enough of OpenCode's HTTP API, recording what it was asked."""

    def __init__(self):
        self.sessions = {}         # id -> session
        self.messages = {}         # id -> [message]
        self.posts = []
        self.health_route = True   # older servers have only /doc
        self.slow = False          # the next ask never returns
        self.error_next = None     # (code, body) for the next ask
        self.n = 0

    def __call__(self, req, timeout=None):
        url = req if isinstance(req, str) else req.full_url
        method = "GET" if isinstance(req, str) else req.get_method()
        data = None if isinstance(req, str) else req.data
        path = url[len(oc.OPENCODE_URL):]
        route, _, _ = path.partition("?")
        body = self.route(method, route, data)
        return io.BytesIO(json.dumps(body).encode("utf-8"))

    def fail(self, code, body):
        raise urllib.error.HTTPError("x", code, "err", {}, io.BytesIO(json.dumps(body).encode()))

    def route(self, method, route, data):
        if route == "/global/health":
            if not self.health_route:
                self.fail(404, {"message": "not found"})
            return {"healthy": True, "version": "1.2.3"}
        if route == "/doc":
            return {"info": {"version": "1.2.3"},
                    "paths": {"/global/health": {}, "/session": {}, "/session/{id}/message": {},
                              "/session/{id}/abort": {}, "/doc": {}}}
        if route == "/session" and method == "GET":
            return list(self.sessions.values())
        if route == "/session":
            self.n += 1
            sid = "ses_%d" % self.n
            payload = json.loads(data) if data else {}
            self.sessions[sid] = {"id": sid, "title": payload.get("title", ""),
                                  "time": {"updated": 1700000000000}}
            self.messages[sid] = []
            self.posts.append(("session", payload))
            return self.sessions[sid]
        parts = route.split("/")
        if len(parts) == 4 and parts[1] == "session":
            sid, action = parts[2], parts[3]
            if sid not in self.sessions:
                self.fail(404, {"message": "session not found"})
            if action == "message" and method == "GET":
                return self.messages[sid]
            if action == "message":
                payload = json.loads(data)
                self.posts.append(("ask", sid, payload))
                if self.slow:
                    raise urllib.error.URLError("timed out")
                if self.error_next:
                    code, body = self.error_next
                    self.error_next = None
                    self.fail(code, body)
                text = payload["parts"][0]["text"]
                self.messages[sid].append({"info": {"role": "user"},
                                           "parts": [{"type": "text", "text": text}]})
                reply = {"info": {"role": "assistant", "id": "msg_1"},
                         "parts": [{"type": "step-start"},
                                   {"type": "reasoning", "text": "hmm"},
                                   {"type": "tool", "tool": "write",
                                    "state": {"status": "completed", "title": "rename.py"}},
                                   {"type": "tool", "tool": "bash",
                                    "state": {"status": "error", "title": "python rename.py",
                                              "error": "exit 1"}},
                                   {"type": "patch", "files": ["rename.py"]},
                                   {"type": "text", "text": "I wrote rename.py for: " + text}]}
                self.messages[sid].append(reply)
                return reply
            if action == "abort":
                self.posts.append(("abort", sid))
                return True
        self.fail(404, {"message": "no route " + route})


class TestOpenCodeBridge(unittest.TestCase):
    def setUp(self):
        self.fake = FakeOpenCode()
        self._real_open = urllib.request.urlopen
        urllib.request.urlopen = self.fake
        self.tmp = tempfile.mkdtemp()
        self._real_ws = oc.WORKSPACE
        oc.WORKSPACE = os.path.join(self.tmp, "ws")
        os.makedirs(oc.WORKSPACE)

    def tearDown(self):
        urllib.request.urlopen = self._real_open
        oc.WORKSPACE = self._real_ws
        shutil.rmtree(self.tmp, ignore_errors=True)

    def text(self, res):
        return res["content"][0]["text"]

    # ----------------------------------------------------------- the contract

    def test_tool_list_is_the_registry_entry_and_survives_sanitizing(self):
        names = {t["name"] for t in oc.tool_list()}
        app = eng.APPS_BY_ID["opencode"]
        everything = {t for names_ in app.groups.values() for t in names_}
        self.assertEqual(names, everything, "registry groups and bridge tools disagree")
        for t in oc.tool_list():
            self.assertTrue(t["description"].strip())
            eng.sanitize_schema(t["inputSchema"])
            self.assertEqual(t["inputSchema"]["type"], "object")

    def test_read_only_tools_are_annotated_and_the_rest_are_not(self):
        hints = {t["name"]: t["annotations"]["readOnlyHint"] for t in oc.tool_list()}
        for name in ("opencode_status", "opencode_list_sessions", "opencode_get_session",
                     "opencode_list_files", "opencode_read_file"):
            self.assertTrue(hints[name], name)
            self.assertTrue(tasks.readonly(name, {}, {"annotations": {"readOnlyHint": True}}))
        for name in ("opencode_ask", "opencode_new_session", "opencode_abort",
                     "opencode_put_file"):
            self.assertFalse(hints[name], name)

    def test_prompt_relies_only_on_default_tools(self):
        app = eng.APPS_BY_ID["opencode"]
        for tool in ("opencode_status", "opencode_ask", "opencode_get_session",
                     "opencode_put_file", "opencode_list_files", "opencode_read_file",
                     "opencode_abort"):
            self.assertIn(tool, app.tool_names())
            self.assertIn(tool, app.system_prompt)

    def test_bridge_and_engine_agree_on_url_and_workspace(self):
        self.assertEqual(oc.OPENCODE_URL, eng.OPENCODE_URL)
        self.assertEqual(self._real_ws, eng.OPENCODE_WORKSPACE)
        self.assertEqual(eng.APPS_BY_ID["opencode"].workspace, eng.OPENCODE_WORKSPACE)

    # --------------------------------------------------------------- asking

    def test_ask_starts_a_session_and_returns_what_opencode_said_and_did(self):
        res = oc.call_tool("opencode_ask", {"prompt": "rename the frames"})
        self.assertFalse(res.get("isError"))
        out = self.text(res)
        self.assertIn("Session ses_1", out)
        self.assertIn("I wrote rename.py for: rename the frames", out)
        self.assertIn("[write completed: rename.py]", out)
        self.assertIn("[bash error: python rename.py]", out)
        self.assertIn("error: exit 1", out)
        self.assertIn("Files touched: rename.py", out)
        self.assertNotIn("hmm", out, "reasoning is not relayed")
        self.assertIn(oc.WORKSPACE, out)
        self.assertEqual(self.fake.sessions["ses_1"]["title"], "rename the frames")
        # Continuing passes the same id back.
        res = oc.call_tool("opencode_ask", {"prompt": "now add tests", "session_id": "ses_1"})
        self.assertIn("Session ses_1", self.text(res))
        self.assertEqual(len(self.fake.sessions), 1)
        self.assertEqual(self.fake.posts[-1][2]["parts"], [{"type": "text", "text": "now add tests"}])

    def test_a_timeout_hands_back_the_session_not_a_retry(self):
        self.fake.slow = True
        res = oc.call_tool("opencode_ask", {"prompt": "big job", "timeout": 10})
        self.assertTrue(res["isError"])
        self.assertIn("still working on session ses_1", self.text(res))
        self.assertIn("opencode_get_session", self.text(res))

    def test_get_session_collects_the_history(self):
        oc.call_tool("opencode_ask", {"prompt": "rename the frames"})
        res = oc.call_tool("opencode_get_session", {"session_id": "ses_1", "limit": 2})
        out = self.text(res)
        self.assertIn("USER:\nrename the frames", out)
        self.assertIn("ASSISTANT:\n", out)
        self.assertIn("Files touched: rename.py", out)
        res = oc.call_tool("opencode_get_session", {"session_id": "nope"})
        self.assertTrue(res["isError"])
        self.assertIn("HTTP 404", self.text(res))
        self.assertIn("session not found", self.text(res))

    def test_sessions_new_list_and_abort(self):
        self.assertIn("No sessions yet", self.text(oc.call_tool("opencode_list_sessions", {})))
        res = oc.call_tool("opencode_new_session", {"title": "Plot CSV"})
        self.assertIn("ses_1", self.text(res))
        out = self.text(oc.call_tool("opencode_list_sessions", {}))
        self.assertIn("ses_1", out)
        self.assertIn("Plot CSV", out)
        self.assertIn("2023-11-14", out)
        res = oc.call_tool("opencode_abort", {"session_id": "ses_1"})
        self.assertIn("stop session ses_1", self.text(res))
        self.assertEqual(self.fake.posts[-1], ("abort", "ses_1"))

    def test_opencode_errors_reach_the_model_as_prose(self):
        self.fake.error_next = (500, {"name": "ProviderError",
                                      "data": {"message": "model not found: lmstudio/x"}})
        res = oc.call_tool("opencode_ask", {"prompt": "hi"})
        self.assertTrue(res["isError"])
        self.assertIn("HTTP 500", self.text(res))
        self.assertIn("model not found", self.text(res))

    def test_unreachable_server_says_who_starts_it(self):
        def down(req, timeout=None):
            raise urllib.error.URLError("connection refused")
        urllib.request.urlopen = down
        for name, args in (("opencode_ask", {"prompt": "x"}), ("opencode_list_sessions", {})):
            res = oc.call_tool(name, args)
            self.assertTrue(res["isError"], name)
            self.assertIn("Start OpenCode button", self.text(res))
        # status still answers, with the workspace, so the model can say where it is
        res = oc.call_tool("opencode_status", {})
        self.assertFalse(res.get("isError"))
        self.assertIn("Cannot reach OpenCode", self.text(res))
        self.assertIn(oc.WORKSPACE, self.text(res))

    # --------------------------------------------------------------- status

    def test_status_reports_version_model_and_route_drift(self):
        with open(os.path.join(oc.WORKSPACE, "opencode.json"), "w") as f:
            json.dump({"model": "lmstudio/qwen"}, f)
        out = self.text(oc.call_tool("opencode_status", {}))
        self.assertIn("OpenCode 1.2.3 is running", out)
        self.assertIn("lmstudio/qwen", out)
        self.assertIn("0 session(s)", out)
        self.assertNotIn("does not list", out)
        self.assertIn("nothing else on this PC", out)
        # An older server without /global/health still answers, via /doc ...
        self.fake.health_route = False
        out = self.text(oc.call_tool("opencode_status", {}))
        self.assertIn("OpenCode 1.2.3 is running", out)
        self.assertIn("no /global/health route", out)
        # ... and a route the server's own document does not know is named.
        oc.ROUTES["abort"] = "/session/{id}/stop"
        try:
            out = self.text(oc.call_tool("opencode_status", {}))
            self.assertIn("/session/{id}/stop", out)
            self.assertIn("does not list", out)
        finally:
            oc.ROUTES["abort"] = "/session/{id}/abort"

    # ------------------------------------------------------------ workspace

    def test_file_tools_read_and_write_the_workspace_only(self):
        res = oc.call_tool("opencode_put_file", {"path": "src/a.py", "content": "print(1)\n"})
        self.assertFalse(res.get("isError"))
        self.assertIn("Wrote src/a.py", self.text(res))
        self.assertIn("/workspace/src/a.py", self.text(res))
        with open(os.path.join(oc.WORKSPACE, "src", "a.py")) as f:
            self.assertEqual(f.read(), "print(1)\n")
        # No silent overwrite.
        res = oc.call_tool("opencode_put_file", {"path": "src/a.py", "content": "x"})
        self.assertTrue(res["isError"])
        self.assertIn("overwrite", self.text(res))
        res = oc.call_tool("opencode_put_file", {"path": "src/a.py", "content": "x", "overwrite": True})
        self.assertIn("Replaced src/a.py", self.text(res))
        self.assertEqual(self.text(oc.call_tool("opencode_read_file", {"path": "src/a.py"})), "x")
        # Container-side paths are understood as workspace paths.
        self.assertEqual(self.text(oc.call_tool("opencode_read_file", {"path": "/workspace/src/a.py"})), "x")
        out = self.text(oc.call_tool("opencode_list_files", {}))
        self.assertIn("src/a.py  (1 bytes)", out)
        out = self.text(oc.call_tool("opencode_list_files", {"path": "src"}))
        self.assertIn("src/a.py", out)
        res = oc.call_tool("opencode_read_file", {"path": "src/none.py"})
        self.assertTrue(res["isError"])

    def test_paths_cannot_leave_the_workspace(self):
        outside = os.path.join(self.tmp, "secret.txt")
        with open(outside, "w") as f:
            f.write("no")
        for bad in ("../secret.txt", "..\\secret.txt", outside, "C:/Windows/win.ini",
                    "src/../../secret.txt"):
            with self.subTest(path=bad):
                res = oc.call_tool("opencode_read_file", {"path": bad})
                self.assertTrue(res["isError"])
                self.assertIn("outside the workspace", self.text(res))
                res = oc.call_tool("opencode_put_file", {"path": bad, "content": "x"})
                self.assertTrue(res["isError"])
                res = oc.call_tool("opencode_list_files", {"path": bad})
                self.assertTrue(res["isError"])
        with open(outside) as f:
            self.assertEqual(f.read(), "no")
        self.assertEqual(os.listdir(oc.WORKSPACE), [])

    def test_listing_skips_dependency_folders_and_is_bounded(self):
        for d in ("node_modules/x", ".git", "src"):
            os.makedirs(os.path.join(oc.WORKSPACE, d))
        open(os.path.join(oc.WORKSPACE, "node_modules", "x", "big.js"), "w").close()
        open(os.path.join(oc.WORKSPACE, "src", "ok.py"), "w").close()
        out = self.text(oc.call_tool("opencode_list_files", {}))
        self.assertIn("src/ok.py", out)
        self.assertNotIn("big.js", out)
        rows, truncated = oc.list_files("", limit=1)
        self.assertTrue(truncated)
        self.assertEqual(len(rows), 1)

    def test_unknown_tool_and_bad_arguments_are_errors_not_exceptions(self):
        self.assertTrue(oc.call_tool("opencode_nope", {})["isError"])
        self.assertTrue(oc.call_tool("opencode_ask", {})["isError"])
        self.assertTrue(oc.call_tool("opencode_ask", {"prompt": "  "})["isError"])
        self.assertTrue(oc.call_tool("opencode_read_file", {"path": 3})["isError"])
        self.assertTrue(oc.call_tool("opencode_put_file", {"path": "a", "content": 3})["isError"])

    # ------------------------------------------------------------ the wire

    def test_stdio_server_speaks_what_mcpclient_expects(self):
        lines = [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
             "params": {"name": "opencode_status", "arguments": {}}},
            {"jsonrpc": "2.0", "id": 4, "method": "resources/list"},
        ]
        out = io.StringIO()
        oc.serve(io.StringIO("\n".join(json.dumps(m) for m in lines) + "\nnot json\n"), out)
        replies = [json.loads(l) for l in out.getvalue().splitlines()]
        # The unparseable line gets the JSON-RPC parse error, addressed to no id.
        self.assertEqual([r["id"] for r in replies], [1, 2, 3, 4, None])
        self.assertEqual(replies[4]["error"]["code"], -32700)
        self.assertEqual(replies[0]["result"]["serverInfo"]["name"], "studio-opencode-mcp")
        self.assertEqual(len(replies[1]["result"]["tools"]), len(oc.TOOLS))
        self.assertIn("OpenCode 1.2.3", replies[2]["result"]["content"][0]["text"])
        self.assertEqual(replies[3]["error"]["code"], -32601)

    def test_executor_validation_accepts_what_the_prompt_teaches(self):
        schema = {t["name"]: t["inputSchema"] for t in oc.tool_list()}
        tasks.validate({"prompt": "x", "session_id": "ses_1", "timeout": 120}, schema["opencode_ask"])
        tasks.validate({"path": "src/a.py", "content": "x", "overwrite": True},
                       schema["opencode_put_file"])
        with self.assertRaises(ValueError):
            tasks.validate({"prompt": "x", "timeout": 5}, schema["opencode_ask"])
        with self.assertRaises(ValueError):
            tasks.validate({"content": "x"}, schema["opencode_put_file"])


if __name__ == "__main__":
    unittest.main()
