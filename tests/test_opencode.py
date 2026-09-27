"""The OpenCode bridge, against a fake OpenCode server and a temp folder.

No network and no OpenCode: `urllib.request.urlopen` is replaced with an
in-memory server that plays a session out step by step - working, stopping on
a permission or a question, going idle - the way the real one did when these
routes were read off OpenCode 1.18 (`/doc`) and driven by hand. What these
prove is the contract the registry entry and the prompt rely on: that every
step OpenCode asks about goes to the user through elicitation and the user's
answer is what OpenCode gets; that a client with no user refuses every step;
that a Stop halts the session; that the reply says what the user decided and
what OpenCode did; and that the stdio server speaks what MCPClient expects.
"""
import base64
import io
import json
import os
import shutil
import tempfile
import unittest
import urllib.error
import urllib.request

import studio_agent as eng
import studio_mcp
import studio_opencode_mcp as oc
import studio_tasks as tasks

EDIT = {"permission": "edit", "patterns": ["hello.py"],
        "metadata": {"filepath": "<ws>/hello.py",
                     "diff": "Index: x\n===\n--- x\n+++ x\n@@ -1 +1 @@\n-print('hi')\n+print('hello')\n"},
        "always": ["*"]}
BASH = {"permission": "bash", "patterns": ["python hello.py"],
        "metadata": {"command": "python hello.py"}, "always": ["python hello.py *"]}
QUESTION = {"questions": [{"question": "Which greeting?", "header": "Greeting",
                           "options": [{"label": "hello", "description": ""},
                                       {"label": "hi", "description": ""}]}]}


class FakeOpenCode:
    """Just enough of OpenCode's HTTP API, recording what it was asked.

    `script` is what the next task does: a list of ("permission", request),
    ("question", request) and ("say", text) steps, played in order. A pending
    request holds the session busy until it is answered, like the real one."""

    def __init__(self, ws):
        self.ws = ws
        self.sessions = {}         # id -> session
        self.messages = {}         # id -> [message]
        self.posts = []
        self.auth = []             # the Authorization header of each request
        self.script = [("say", "done")]
        self.running = {}          # sid -> remaining steps
        self.pending = None        # (kind, request)
        self.health_route = True
        self.error_next = None
        self.forever = False       # the session never finishes
        self.n = 0
        self.documented = set(oc.ROUTES.values())   # what its /doc lists

    def __call__(self, req, timeout=None):
        self.auth.append(req.get_header("Authorization"))
        path = req.full_url[len(oc.OPENCODE_URL):]
        route, _, query = path.partition("?")
        body = self.route(req.get_method(), route, req.data, query)
        return io.BytesIO(json.dumps(body).encode("utf-8"))

    def fail(self, code, body):
        raise urllib.error.HTTPError("x", code, "err", {}, io.BytesIO(json.dumps(body).encode()))

    def advance(self):
        """Play steps until one needs an answer or the script runs out."""
        for sid, steps in list(self.running.items()):
            while steps and self.pending is None:
                kind, what = steps.pop(0)
                self.n += 1
                if kind == "say":
                    self.messages[sid].append(
                        {"info": {"role": "assistant", "id": "msg_%d" % self.n},
                         "parts": [{"type": "reasoning", "text": "hmm"},
                                   {"type": "tool", "tool": "edit",
                                    "state": {"status": "completed", "title": "hello.py"}},
                                   {"type": "patch", "files": [os.path.join(self.ws, "hello.py")]},
                                   {"type": "text", "text": what}]})
                else:
                    req = json.loads(json.dumps(what).replace("<ws>", self.ws.replace("\\", "/")))
                    req.update(id=("per_%d" if kind == "permission" else "que_%d") % self.n,
                               sessionID=sid)
                    self.pending = (kind, req)
            if not steps and self.pending is None and not self.forever:
                del self.running[sid]

    def route(self, method, route, data, query):
        payload = json.loads(data) if data else {}
        if route == "/global/health":
            if not self.health_route:
                self.fail(404, {"message": "not found"})
            return {"healthy": True, "version": "1.2.3"}
        if route == "/doc":
            return {"info": {"version": "1.2.3"},
                    "paths": {r: {} for r in self.documented}}
        if route == "/session" and method == "GET":
            return list(self.sessions.values())
        if route == "/session":
            self.n += 1
            sid = "ses_%d" % self.n
            self.sessions[sid] = {"id": sid, "title": payload.get("title", ""),
                                  "time": {"updated": 1700000000000}}
            self.messages[sid] = []
            self.posts.append(("session", payload))
            return self.sessions[sid]
        if route == "/session/status":
            self.advance()
            return {sid: {"type": "busy"} for sid in self.running}
        if route == "/permission":
            self.advance()
            return [self.pending[1]] if self.pending and self.pending[0] == "permission" else []
        if route == "/question":
            self.advance()
            return [self.pending[1]] if self.pending and self.pending[0] == "question" else []
        if route.startswith("/permission/") or route.startswith("/question/"):
            rid = route.split("/")[2]
            assert self.pending and self.pending[1]["id"] == rid, (rid, self.pending)
            self.posts.append((route.split("/")[3], rid, payload))
            self.pending = None
            return True
        if route == "/vcs/diff":
            assert query == "mode=git", query
            return [{"file": "hello.py", "status": "modified", "additions": 1, "deletions": 1,
                     "patch": "@@ -1 +1 @@\n-print('hi')\n+print('hello')\n"}]
        parts = route.split("/")
        if len(parts) >= 3 and parts[1] == "session":
            sid = parts[2]
            if sid not in self.sessions:
                self.fail(404, {"message": "session not found"})
            action = parts[3] if len(parts) > 3 else ""
            if action == "":
                return {"id": sid, "parentID": self.sessions[sid].get("parentID")}
            if action == "message":
                return self.messages[sid]
            if action == "prompt_async":
                if self.error_next:
                    code, body = self.error_next
                    self.error_next = None
                    self.fail(code, body)
                self.posts.append(("ask", sid, payload))
                self.n += 1
                self.messages[sid].append({"info": {"role": "user", "id": "msg_%d" % self.n},
                                           "parts": payload["parts"]})
                self.running[sid] = list(self.script)
                return {}
            if action == "abort":
                self.posts.append(("abort", sid))
                self.running.pop(sid, None)
                self.pending = None
                return True
        self.fail(404, {"message": "no route " + route})


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._real = (oc.WORKSPACE, oc.KEY_FILE, oc.POLL, oc.SETTLE, urllib.request.urlopen,
                      oc.SERVER.client_caps, oc.STATE_DIR)
        oc.STATE_DIR = os.path.join(self.tmp, "state")
        oc.WORKSPACE = os.path.join(self.tmp, "ws")
        os.makedirs(oc.WORKSPACE)
        oc.KEY_FILE = os.path.join(self.tmp, "server.key")
        oc.POLL = oc.SETTLE = 0
        self.fake = FakeOpenCode(oc.WORKSPACE)
        urllib.request.urlopen = self.fake
        oc.SERVER.client_caps = {}
        self.asked = []

    def tearDown(self):
        (oc.WORKSPACE, oc.KEY_FILE, oc.POLL, oc.SETTLE, urllib.request.urlopen,
         oc.SERVER.client_caps, oc.STATE_DIR) = self._real
        oc.SERVER.sink = None
        shutil.rmtree(self.tmp, ignore_errors=True)

    def text(self, res):
        return res["content"][0]["text"]

    def user(self, *answers):
        """A client whose user gives `answers` in turn, recording each ask."""
        answers = list(answers)

        def on_elicit(params):
            self.asked.append(params)
            return answers.pop(0)
        client = studio_mcp.Loopback(oc.SERVER, on_elicit=on_elicit)
        client.initialize()
        return client

    def accept(self, **content):
        return {"action": "accept", "content": content}


class TestContract(Base):

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
                     "opencode_list_files", "opencode_read_file", "opencode_changes"):
            self.assertTrue(hints[name], name)
        for name in ("opencode_ask", "opencode_wait", "opencode_new_session", "opencode_abort"):
            self.assertFalse(hints[name], name)

    def test_nothing_the_model_can_call_approves_or_writes(self):
        """The user approves; the model has no tool that does. The old
        put_file wrote into the folder on the model's word alone."""
        names = {t["name"] for t in oc.tool_list()}
        self.assertNotIn("opencode_put_file", names)
        for name in names:
            self.assertNotRegex(name, "approve|permit|reply|allow|put|write")

    def test_prompt_relies_only_on_default_tools(self):
        app = eng.APPS_BY_ID["opencode"]
        for tool in ("opencode_status", "opencode_ask", "opencode_wait", "opencode_changes",
                     "opencode_abort"):
            self.assertIn(tool, app.tool_names())
            self.assertIn(tool, app.system_prompt)
        self.assertNotIn("opencode_put_file", app.system_prompt)

    def test_bridge_and_engine_agree_on_url_workspace_and_state(self):
        self.assertEqual(oc.OPENCODE_URL, eng.OPENCODE_URL)
        self.assertEqual(self._real[0], eng.OPENCODE_WORKSPACE)
        app = eng.APPS_BY_ID["opencode"]
        self.assertEqual(app.workspace, eng.OPENCODE_WORKSPACE)
        self.assertEqual(self._real[1], app.key_path)
        self.assertEqual(self._real[-1], app.state_dir)   # setUp moved STATE_DIR to tmp

    def test_every_request_carries_the_password_when_there_is_one(self):
        oc.call_tool("opencode_list_sessions", {})
        self.assertIsNone(self.fake.auth[-1])
        with open(oc.KEY_FILE, "w") as f:
            f.write("s3cret\n")
        oc.call_tool("opencode_list_sessions", {})
        self.assertEqual(self.fake.auth[-1],
                         "Basic " + base64.b64encode(b"opencode:s3cret").decode())

    def test_a_refused_password_says_who_restarts_it(self):
        def refuse(req, timeout=None):
            raise urllib.error.HTTPError("x", 401, "no", {}, io.BytesIO(b""))
        urllib.request.urlopen = refuse
        res = oc.call_tool("opencode_list_sessions", {})
        self.assertTrue(res["isError"])
        self.assertIn("refused the password", self.text(res))
        self.assertIn("Start OpenCode", self.text(res))


class TestApprovals(Base):

    def test_each_step_goes_to_the_user_and_their_answer_to_opencode(self):
        self.fake.script = [("permission", EDIT), ("permission", BASH), ("say", "Changed it.")]
        client = self.user(self.accept(decision="once"),
                           self.accept(decision="reject", note="do not run anything"))
        res = client.call_tool("opencode_ask", {"prompt": "say hello"})
        self.assertFalse(res.get("isError"), res)
        # What the user was shown: the file relative to the folder, its diff,
        # the command - and the choices, with what "always" would cover.
        edit, bash = self.asked
        self.assertIn("edit hello.py", edit["message"])
        shown = edit["_meta"]["studio/approval"]
        self.assertEqual(shown["file"], "hello.py")
        self.assertIn("+print('hello')", shown["diff"])
        decision = edit["requestedSchema"]["properties"]["decision"]
        self.assertEqual(decision["enum"], ["once", "always", "reject"])
        self.assertIn("Always allow every edit in this session", decision["enumNames"])
        self.assertIn("python hello.py", bash["message"])
        self.assertIn("python hello.py *", bash["requestedSchema"]["properties"]["decision"]
                      ["enumNames"][1])
        # What OpenCode was told.
        replies = [p for p in self.fake.posts if p[0] == "reply"]
        self.assertEqual(replies[0][2], {"reply": "once"})
        self.assertEqual(replies[1][2], {"reply": "reject", "message": "do not run anything"})
        # What the model is told: the decisions, then what OpenCode said and did.
        out = self.text(res)
        self.assertIn("Session ses_1 is done", out)
        self.assertIn("allowed: edit hello.py", out)
        self.assertIn("refused: run a command", out)
        self.assertIn("note: do not run anything", out)
        self.assertIn("Changed it.", out)
        self.assertIn("Files touched: hello.py", out)
        self.assertNotIn("hmm", out, "reasoning is not relayed")

    def test_decline_is_a_refusal_and_cancel_stops_the_session(self):
        self.fake.script = [("permission", EDIT), ("permission", BASH), ("say", "never")]
        client = self.user({"action": "decline"}, {"action": "cancel"})
        res = client.call_tool("opencode_ask", {"prompt": "x"})
        self.assertTrue(res["isError"])
        self.assertEqual([p[2]["reply"] for p in self.fake.posts if p[0] == "reply"], ["reject"])
        self.assertIn(("abort", "ses_1"), self.fake.posts)
        self.assertIn("The user stopped OpenCode", self.text(res))
        self.assertIn("refused: edit hello.py", self.text(res))

    def test_a_client_with_no_user_refuses_the_step_and_stops(self):
        """No elicitation - a CLI piped from a script, another MCP client -
        means nobody can approve, so nothing is changed on the model's word."""
        self.fake.script = [("permission", EDIT), ("say", "never")]
        res = oc.call_tool("opencode_ask", {"prompt": "x"})
        self.assertTrue(res["isError"])
        reply = [p for p in self.fake.posts if p[0] == "reply"][0][2]
        self.assertEqual(reply["reply"], "reject")
        self.assertIn("No one could be asked", reply["message"])
        self.assertIn(("abort", "ses_1"), self.fake.posts)
        self.assertIn("no one could approve", self.text(res))

    def test_opencodes_questions_are_the_users_to_answer(self):
        self.fake.script = [("question", QUESTION), ("say", "Used hello.")]
        client = self.user(self.accept(answer="hello"))
        res = client.call_tool("opencode_ask", {"prompt": "greet"})
        q = self.asked[0]
        self.assertIn("Which greeting?", q["message"])
        self.assertEqual(q["requestedSchema"]["properties"]["answer"]["enum"], ["hello", "hi"])
        self.assertIn(("reply", self.fake.posts[-1][1], {"answers": [["hello"]]}), self.fake.posts)
        self.assertIn("answered OpenCode: hello", self.text(res))

    def test_a_subagents_request_is_asked_and_a_strangers_is_not(self):
        self.fake.sessions["ses_child"] = {"id": "ses_child", "parentID": "ses_1"}
        self.fake.sessions["ses_other"] = {"id": "ses_other"}
        fam = oc.Family("ses_1")
        self.fake.sessions["ses_1"] = {"id": "ses_1"}
        self.assertIn("ses_1", fam)
        self.assertIn("ses_child", fam)
        self.assertNotIn("ses_other", fam)

    def test_timeout_message_points_at_wait(self):
        self.fake.forever = True
        self.fake.script = []
        sid = oc.new_session("x")["id"]
        oc.prompt(sid, "go")
        res = oc.run(sid, set(), work_limit=0)
        self.assertTrue(res["isError"])
        self.assertIn("opencode_wait", self.text(res))
        self.assertNotIn(("abort", sid), self.fake.posts)
        # ... and wait picks the same session up, asking as ask does.
        self.fake.forever = False
        self.fake.running[sid] = [("permission", EDIT), ("say", "finished")]
        client = self.user(self.accept(decision="always"))
        res = client.call_tool("opencode_wait", {"session_id": sid})
        self.assertIn("finished", self.text(res))
        self.assertIn("allowed from now on", self.text(res))

    def test_continuing_a_session_reports_only_the_new_work(self):
        client = self.user()
        client.call_tool("opencode_ask", {"prompt": "first"})
        self.fake.script = [("say", "second answer")]
        res = client.call_tool("opencode_ask", {"prompt": "second", "session_id": "ses_1"})
        out = self.text(res)
        self.assertIn("second answer", out)
        self.assertNotIn("done", out.replace("is done", ""))
        self.assertEqual(len(self.fake.sessions), 1)


    def test_a_follow_up_continues_the_last_session_unless_told_otherwise(self):
        # A local model often drops the session_id; OpenCode then started over
        # knowing nothing of the work it had just done.
        client = self.user()
        client.call_tool("opencode_ask", {"prompt": "first"})
        self.fake.script = [("say", "second answer")]
        client.call_tool("opencode_ask", {"prompt": "second"})
        self.assertEqual(len(self.fake.sessions), 1)
        client.call_tool("opencode_ask", {"prompt": "unrelated", "new_session": True})
        self.assertEqual(len(self.fake.sessions), 2)

    def test_a_small_context_window_is_named_with_its_fix(self):
        conf = eng.opencode_config("http://h:1/v1", "m1", ["m1"], 32768)
        note = oc.context_note(conf)
        self.assertIn("32768", note)
        self.assertIn("Context Length", note)
        self.assertEqual(oc.context_note(eng.opencode_config("http://h:1/v1", "m1", ["m1"], 131072)), "")
        self.assertEqual(oc.context_note(eng.opencode_config("http://h:1/v1", "m1", ["m1"])), "")


class TestOtherTools(Base):

    def test_get_session_collects_the_history(self):
        self.user().call_tool("opencode_ask", {"prompt": "rename the frames"})
        out = self.text(oc.call_tool("opencode_get_session", {"session_id": "ses_1", "limit": 2}))
        self.assertIn("USER:\nrename the frames", out)
        self.assertIn("ASSISTANT:\n", out)
        res = oc.call_tool("opencode_get_session", {"session_id": "nope"})
        self.assertTrue(res["isError"])
        self.assertIn("session not found", self.text(res))

    def test_sessions_new_list_and_abort(self):
        self.assertIn("No sessions yet", self.text(oc.call_tool("opencode_list_sessions", {})))
        self.assertIn("ses_1", self.text(oc.call_tool("opencode_new_session", {"title": "Plot"})))
        out = self.text(oc.call_tool("opencode_list_sessions", {}))
        self.assertIn("Plot", out)
        self.assertIn("2023-11-14", out)
        self.assertIn("stop session ses_1",
                      self.text(oc.call_tool("opencode_abort", {"session_id": "ses_1"})))

    def test_changes_lists_files_and_shows_one_diff(self):
        out = self.text(oc.call_tool("opencode_changes", {}))
        self.assertIn("modified  hello.py  +1 -1", out)
        out = self.text(oc.call_tool("opencode_changes", {"path": "hello.py"}))
        self.assertIn("+print('hello')", out)
        self.assertIn("no uncommitted changes",
                      self.text(oc.call_tool("opencode_changes", {"path": "other.py"})))

    def test_opencode_errors_reach_the_model_as_prose(self):
        self.fake.error_next = (500, {"name": "ProviderError",
                                      "data": {"message": "model not found: lmstudio/x"}})
        res = self.user().call_tool("opencode_ask", {"prompt": "hi"})
        self.assertTrue(res["isError"])
        self.assertIn("model not found", self.text(res))

    def test_unreachable_server_says_who_starts_it(self):
        def down(req, timeout=None):
            raise urllib.error.URLError("connection refused")
        urllib.request.urlopen = down
        for name, args in (("opencode_ask", {"prompt": "x"}), ("opencode_list_sessions", {})):
            res = oc.call_tool(name, args)
            self.assertTrue(res["isError"], name)
            self.assertIn("Start OpenCode button", self.text(res))
        res = oc.call_tool("opencode_status", {})
        self.assertFalse(res.get("isError"))
        self.assertIn("Cannot reach OpenCode", self.text(res))
        self.assertIn(oc.WORKSPACE, self.text(res))

    def test_status_reports_version_approvals_and_route_drift(self):
        out = self.text(oc.call_tool("opencode_status", {}))
        self.assertIn("OpenCode 1.2.3 is running", out)
        self.assertIn("0 session(s)", out)
        self.assertIn("waits for the user's approval", out)
        self.assertNotIn("does not list", out)
        self.fake.health_route = False
        self.assertIn("no /global/health route", self.text(oc.call_tool("opencode_status", {})))
        real = oc.ROUTES["abort"]
        oc.ROUTES["abort"] = "/session/{sessionID}/stop"
        try:
            out = self.text(oc.call_tool("opencode_status", {}))
            self.assertIn("/session/{sessionID}/stop", out)
        finally:
            oc.ROUTES["abort"] = real

    def test_file_tools_read_the_folder_only(self):
        os.makedirs(os.path.join(oc.WORKSPACE, "src"))
        with open(os.path.join(oc.WORKSPACE, "src", "a.py"), "w") as f:
            f.write("x")
        self.assertEqual(self.text(oc.call_tool("opencode_read_file", {"path": "src/a.py"})), "x")
        self.assertIn("src/a.py  (1 bytes)", self.text(oc.call_tool("opencode_list_files", {})))
        outside = os.path.join(self.tmp, "secret.txt")
        with open(outside, "w") as f:
            f.write("no")
        for bad in ("../secret.txt", "..\\secret.txt", outside, "C:/Windows/win.ini",
                    "src/../../secret.txt"):
            with self.subTest(path=bad):
                res = oc.call_tool("opencode_read_file", {"path": bad})
                self.assertTrue(res["isError"])
                self.assertIn("outside OpenCode's folder", self.text(res))

    def test_listing_skips_dependency_folders_and_is_bounded(self):
        for d in ("node_modules/x", ".git", ".claude/worktrees", "src"):
            os.makedirs(os.path.join(oc.WORKSPACE, d))
        open(os.path.join(oc.WORKSPACE, "node_modules", "x", "big.js"), "w").close()
        open(os.path.join(oc.WORKSPACE, ".claude", "worktrees", "copy.py"), "w").close()
        open(os.path.join(oc.WORKSPACE, "src", "ok.py"), "w").close()
        out = self.text(oc.call_tool("opencode_list_files", {}))
        self.assertIn("src/ok.py", out)
        self.assertNotIn("big.js", out)
        self.assertNotIn("copy.py", out)
        rows, truncated = oc.list_files("", limit=1)
        self.assertTrue(truncated)

    def test_unknown_tool_and_bad_arguments_are_errors_not_exceptions(self):
        self.assertTrue(oc.call_tool("opencode_nope", {})["isError"])
        self.assertTrue(oc.call_tool("opencode_ask", {})["isError"])
        self.assertTrue(oc.call_tool("opencode_ask", {"prompt": "  "})["isError"])
        self.assertTrue(oc.call_tool("opencode_read_file", {"path": 3})["isError"])

    def test_executor_validation_accepts_what_the_prompt_teaches(self):
        schema = {t["name"]: t["inputSchema"] for t in oc.tool_list()}
        tasks.validate({"prompt": "x", "session_id": "ses_1", "timeout": 120}, schema["opencode_ask"])
        tasks.validate({"session_id": "ses_1"}, schema["opencode_wait"])
        with self.assertRaises(ValueError):
            tasks.validate({"prompt": "x", "timeout": 5}, schema["opencode_ask"])


class TestTheWire(Base):

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
        self.assertEqual([r["id"] for r in replies], [1, 2, 3, 4, None])
        self.assertEqual(replies[0]["result"]["serverInfo"]["name"], "studio-opencode-mcp")
        self.assertEqual(len(replies[1]["result"]["tools"]), len(oc.TOOLS))
        self.assertIn("OpenCode 1.2.3", replies[2]["result"]["content"][0]["text"])


if __name__ == "__main__":
    unittest.main()
