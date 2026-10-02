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
import subprocess
import tempfile
import unittest
import urllib.error
import urllib.parse
import urllib.request

import core.agent as eng
import core.mcp as studio_mcp
import apps.opencode.mcp as oc
import core.tasks as tasks

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
    ("question", request), ("say", text), ("write", (path, text)) and
    ("todos", list) steps, played in order; a write lands in the folder the
    session was made for, and "<copy>" in a request is that folder. `scripts`
    holds what the prompts after this one do, one script each, before `script`
    is played again. A pending request holds the session busy until it is
    answered, like the real one."""

    def __init__(self, ws):
        self.ws = ws
        self.sessions = {}         # id -> session
        self.messages = {}         # id -> [message]
        self.posts = []
        self.auth = []             # the Authorization header of each request
        self.script = [("say", "done")]
        self.scripts = []          # the scripts of the prompts to come, in order
        self.todos = []
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
                if kind == "write":
                    folder = self.sessions[sid].get("directory") or self.ws
                    path = os.path.join(folder, what[0])
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(what[1])
                    continue
                if kind == "todos":
                    self.todos = what
                    continue
                if kind == "say":
                    self.messages[sid].append(
                        {"info": {"role": "assistant", "id": "msg_%d" % self.n},
                         "parts": [{"type": "reasoning", "text": "hmm"},
                                   {"type": "tool", "tool": "edit",
                                    "state": {"status": "completed", "title": "hello.py"}},
                                   {"type": "patch", "files": [os.path.join(self.ws, "hello.py")]},
                                   {"type": "text", "text": what}]})
                else:
                    copy = self.sessions[sid].get("directory") or self.ws
                    req = json.loads(json.dumps(what)
                                     .replace("<ws>", self.ws.replace("\\", "/"))
                                     .replace("<copy>", copy.replace("\\", "/")))
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
            folder = dict(p.split("=", 1) for p in query.split("&") if "=" in p).get("directory")
            self.sessions[sid] = {"id": sid, "title": payload.get("title", ""),
                                  "time": {"updated": 1700000000000},
                                  "directory": urllib.parse.unquote(folder) if folder else None}
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
            if action == "todo":
                return self.todos
            if action == "prompt_async":
                if self.error_next:
                    code, body = self.error_next
                    self.error_next = None
                    self.fail(code, body)
                self.posts.append(("ask", sid, payload))
                self.n += 1
                self.messages[sid].append({"info": {"role": "user", "id": "msg_%d" % self.n},
                                           "parts": payload["parts"]})
                self.running[sid] = list(self.scripts.pop(0) if self.scripts else self.script)
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
                     "opencode_list_files", "opencode_read_file", "opencode_search_files", "opencode_changes"):
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
                     "opencode_abort", "opencode_search_files", "opencode_read_file"):
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
        self.assertIn("Always allow every edit for this task", decision["enumNames"])
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

    def test_a_task_is_not_piled_onto_a_session_still_working(self):
        # After a timed-out ask, a second ask queued behind the first and the
        # report mixed the two.
        self.fake.forever = True
        self.fake.script = []
        sid = oc.new_session("x")["id"]
        oc.remember_session(sid)
        oc.prompt(sid, "go")
        res = self.user().call_tool("opencode_ask", {"prompt": "again"})
        self.assertTrue(res["isError"])
        self.assertIn("opencode_wait", self.text(res))
        self.assertEqual(len([p for p in self.fake.posts if p[0] == "ask"]), 1)

    def test_many_decisions_do_not_crowd_out_what_opencode_said(self):
        log = ["allowed: edit file_%d.py" % i for i in range(300)]
        sid = oc.new_session("x")["id"]
        self.fake.messages[sid].append({"info": {"role": "assistant", "id": "m"},
                                        "parts": [{"type": "text", "text": "ALL DONE"}]})
        out = oc.report(sid, set(), log)
        self.assertIn("ALL DONE", out)
        self.assertIn("270 earlier decision(s)", out)
        self.assertLess(len(out), oc.MAX_REPLY_CHARS + 500)

    def test_relative_paths_are_the_workspaces_whatever_the_cwd(self):
        cwd = os.getcwd()
        os.chdir(self.tmp)
        try:
            self.assertEqual(oc.rel("sub/a.py"), "sub/a.py")
            self.assertEqual(oc.rel(os.path.join(oc.WORKSPACE, "b.py")), "b.py")
        finally:
            os.chdir(cwd)

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
        # A server naming the parameter differently still has the route.
        self.fake.documented = [r.replace("{sessionID}", "{id}") for r in self.fake.documented]
        self.assertNotIn("does not list", self.text(oc.call_tool("opencode_status", {})))

    def test_file_tools_read_the_folder_only(self):
        os.makedirs(os.path.join(oc.WORKSPACE, "src"))
        with open(os.path.join(oc.WORKSPACE, "src", "a.py"), "w") as f:
            f.write("x")
        self.assertEqual(self.text(oc.call_tool("opencode_read_file", {"path": "src/a.py"})), "End of file.\nx")
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
        for d in ("node_modules/x", ".git", ".claude/worktrees", ".runtime/facefusion", "src"):
            os.makedirs(os.path.join(oc.WORKSPACE, d))
        with open(os.path.join(oc.WORKSPACE, ".runtime", "facefusion", "noise.py"), "w") as f:
            f.write("needle")
        open(os.path.join(oc.WORKSPACE, "node_modules", "x", "big.js"), "w").close()
        open(os.path.join(oc.WORKSPACE, ".claude", "worktrees", "copy.py"), "w").close()
        open(os.path.join(oc.WORKSPACE, "src", "ok.py"), "w").close()
        out = self.text(oc.call_tool("opencode_list_files", {}))
        self.assertIn("src/ok.py", out)
        self.assertNotIn("big.js", out)
        self.assertNotIn("copy.py", out)
        self.assertNotIn("noise.py", out)
        self.assertNotIn("noise.py", self.text(oc.call_tool("opencode_search_files", {"query": "needle"})))
        rows, truncated = oc.list_files("", limit=1)
        self.assertTrue(truncated)

    def test_read_pages_reassemble_unicode_and_windows_newlines(self):
        content = ("café λ\r\n" * 1800) + "last"
        with open(os.path.join(oc.WORKSPACE, "pages.txt"), "wb") as f:
            f.write(content.encode("utf-8"))
        expected = content.replace("\r\n", "\n")
        pages = []
        start = 0
        while True:
            text = self.text(oc.call_tool("opencode_read_file", {"path": "pages.txt", "start": start}))
            self.assertLess(len(text), eng.MAX_TOOL_RESULT_CHARS)
            header, body = text.split("\n", 1)
            pages.append(body)
            start += len(body)
            if header == "End of file.":
                break
            self.assertIn("start=%d" % start, header)
        self.assertEqual("".join(pages), expected)
        for args in ({"start": -1}, {"limit": 6001}, {"start": 10000001}):
            self.assertTrue(oc.call_tool("opencode_read_file", dict(path="pages.txt", **args))["isError"])

    def test_search_locates_code_beyond_old_reader_limit(self):
        prefix = "# café\r\n" * 4000
        with open(os.path.join(oc.WORKSPACE, "large.py"), "wb") as f:
            f.write((prefix + "def restart_server():\r\n    pass\r\n").encode("utf-8"))
        out = self.text(oc.call_tool("opencode_search_files", {"query": "RESTART_SERVER"}))
        start = len(prefix.replace("\r\n", "\n"))
        self.assertIn("large.py:4001 start=%d" % start, out)
        page = self.text(oc.call_tool("opencode_read_file", {"path": "large.py", "start": start}))
        self.assertIn("def restart_server():", page)
        self.assertTrue(oc.call_tool("opencode_search_files", {"query": "x", "path": "../"})["isError"])

    def test_search_files_supports_regex(self):
        with open(os.path.join(oc.WORKSPACE, "handlers.py"), "w") as f:
            f.write("def restart_server():\n    pass\n\n\ndef restart_client():\n    pass\n")
        literal = self.text(oc.call_tool("opencode_search_files", {"query": "restart_.*("}))
        self.assertIn("This is literal search, not regex", literal)
        self.assertIn("No matches", literal)
        out = self.text(oc.call_tool("opencode_search_files",
                                     {"query": r"def restart_\w+\(", "regex": True}))
        self.assertIn("handlers.py:1", out)
        self.assertIn("handlers.py:5", out)
        bad = oc.call_tool("opencode_search_files", {"query": "restart_(", "regex": True})
        self.assertTrue(bad["isError"])
        self.assertIn("Invalid regex", self.text(bad))

    def test_search_and_listing_fit_executor_budget_and_report_partial(self):
        for i in range(150):
            with open(os.path.join(oc.WORKSPACE, "%03d_%s.txt" % (i, "x" * 60)), "w") as f:
                f.write("match\n" * 100)
        for name, args in (("opencode_search_files", {"query": "match"}), ("opencode_list_files", {})):
            out = self.text(oc.call_tool(name, args))
            self.assertLess(len(out), eng.MAX_TOOL_RESULT_CHARS)
            self.assertTrue("Partial" in out or "more; list a subfolder" in out)

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


class TestProgress(Base):
    def test_todos_are_the_fraction(self):
        sid = self.fake.route("POST", "/session", b"{}", "")["id"]
        self.assertIsNone(oc.todo_count(sid))           # no list yet
        self.fake.todos = [{"content": "a", "status": "completed"},
                           {"content": "b", "status": "in_progress"},
                           {"content": "c", "status": "pending"}, "junk"]
        self.assertEqual(oc.todo_count(sid), (1, 3))
        self.assertIsNone(oc.todo_count("ses_gone"))    # 404 is not a crash

    def test_line_says_working_or_stuck(self):
        line = oc.progress_line("busy", (1, 4), 2, 0, 7)
        self.assertEqual(line, "OpenCode is working; 1/4 to-dos (25%); 2 file(s) changed; "
                               "last activity 7s ago")
        stuck = oc.progress_line("busy", None, None, 1, oc.STALL + 5)
        self.assertIn("1 step(s) decided", stuck)
        self.assertIn("nothing for 2m 05s - may be stuck", stuck)

    def test_files_changed_outside_a_copy_is_unknown(self):
        self.assertIsNone(oc.files_changed(None))


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




def git(cwd, *args):
    return subprocess.run(("git",) + args, cwd=cwd, capture_output=True, text=True,
                          check=True).stdout


class Repo(Base):
    """The workspace as a git repository with one commit and one test."""

    def setUp(self):
        Base.setUp(self)
        ws = oc.WORKSPACE
        git(ws, "init", "-q", "-b", "main")
        git(ws, "config", "user.name", "Test")
        git(ws, "config", "user.email", "test@example.com")
        git(ws, "config", "commit.gpgsign", "false")
        self.write("hello.py", "print('hi')\n")
        self.write("tests/__init__.py", "")
        self.write("tests/test_hello.py",
                   "import unittest\n\nclass T(unittest.TestCase):\n"
                   "    def test_it(self):\n        self.assertTrue(True)\n")
        git(ws, "add", "-A")
        git(ws, "commit", "-q", "-m", "start")

    def tearDown(self):
        subprocess.run(["git", "worktree", "prune"], cwd=oc.WORKSPACE, capture_output=True)
        Base.tearDown(self)

    def write(self, rel, text, root=None):
        path = os.path.join(root or oc.WORKSPACE, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def read(self, rel, root=None):
        with open(os.path.join(root or oc.WORKSPACE, rel), encoding="utf-8") as f:
            return f.read()

    def ask(self, client, prompt="change the greeting", *steps):
        self.fake.script = list(steps) or [("write", ("hello.py", "print('bye')\n")),
                                           ("say", "Changed it.")]
        return client.call_tool("opencode_ask", {"prompt": prompt})


class TestTasks(Repo):
    """A task in a git repository works in a copy of its own; the user's folder
    changes only when the user merges, and every change can be taken back."""

    def test_a_task_works_in_its_copy_then_tests_and_checkpoints(self):
        res = self.ask(self.user())
        out = "\n".join(c["text"] for c in res["content"])
        sid = oc.last_session()
        t = oc.task(sid)
        self.assertTrue(t["isolated"])
        self.assertEqual(self.fake.sessions[sid]["directory"], t["dir"])
        self.assertEqual(self.read("hello.py", t["dir"]), "print('bye')\n")
        self.assertEqual(self.read("hello.py"), "print('hi')\n", "the user's folder is untouched")
        self.assertIn("Tests passed: tests.test_hello (1 tests)", out)
        self.assertIn("Checkpoint", out)
        self.assertEqual(len(oc.task(sid)["checkpoints"]), 1)
        self.assertIn("own copy", out)
        changes = self.text(oc.call_tool("opencode_changes", {}))
        self.assertIn("hello.py  +1 -1", changes)
        self.assertIn("print('bye')", self.text(oc.call_tool("opencode_read_file",
                                                              {"path": "hello.py"})))

    def test_failing_tests_are_reported(self):
        res = self.ask(self.user(), "break the test",
                       ("write", ("tests/test_hello.py",
                                  "import unittest\n\nclass T(unittest.TestCase):\n"
                                  "    def test_it(self):\n        self.fail('boom')\n")),
                       ("say", "done"))
        out = "\n".join(c["text"] for c in res["content"])
        self.assertIn("Tests FAILED: tests.test_hello", out)
        self.assertIn("boom", out)

    def test_merge_is_the_users_and_lands_as_one_commit(self):
        client = self.user(self.accept(decision="no"), self.accept(decision="yes"))
        self.ask(client)
        sid = oc.last_session()
        copy = oc.task(sid)["dir"]
        res = client.call_tool("opencode_merge", {})
        self.assertIn("did not merge", self.text(res))
        self.assertEqual(self.read("hello.py"), "print('hi')\n")
        res = client.call_tool("opencode_merge", {"message": "Say bye"})
        self.assertIn("Merged session", self.text(res))
        self.assertEqual(self.read("hello.py"), "print('bye')\n")
        self.assertEqual(git(oc.WORKSPACE, "log", "-1", "--format=%s").strip(), "Say bye")
        self.assertFalse(os.path.isdir(copy), "the copy is removed")
        self.assertIn("+print('bye')", self.asked[-1]["_meta"]["studio/approval"]["diff"])

    def test_merge_refuses_over_the_users_own_uncommitted_edit(self):
        client = self.user(self.accept(decision="yes"))
        self.ask(client)
        self.write("hello.py", "print('mine')\n")
        res = client.call_tool("opencode_merge", {})
        self.assertTrue(res["isError"])
        self.assertIn("uncommitted changes", self.text(res))
        self.assertEqual(self.read("hello.py"), "print('mine')\n")
        self.assertEqual(self.asked, [], "the user was not even asked")

    def test_undo_takes_back_the_last_ask_and_opencode_is_told(self):
        client = self.user(self.accept(decision="yes"))
        self.ask(client)
        sid = oc.last_session()
        copy = oc.task(sid)["dir"]
        self.ask(client, "again", ("write", ("hello.py", "print('ciao')\n")), ("say", "ok"))
        self.assertEqual(self.read("hello.py", copy), "print('ciao')\n")
        res = client.call_tool("opencode_undo", {})
        self.assertIn("Undid the last change", self.text(res))
        self.assertEqual(self.read("hello.py", copy), "print('bye')\n")
        self.ask(client, "third", ("say", "ok"))
        last = [p for p in self.fake.posts if p[0] == "ask"][-1]
        self.assertIn("undid your last change", last[2]["parts"][0]["text"])
        self.assertFalse(oc.task(sid).get("undone"))

    def test_a_merged_task_can_be_reverted(self):
        client = self.user(self.accept(decision="yes"), self.accept(decision="yes"))
        self.ask(client)
        client.call_tool("opencode_merge", {})
        res = client.call_tool("opencode_undo", {})
        self.assertIn("Reverted the merge", self.text(res))
        self.assertEqual(self.read("hello.py"), "print('hi')\n")

    def test_discard_drops_the_copy_and_the_branch(self):
        client = self.user(self.accept(decision="yes"))
        self.ask(client)
        sid = oc.last_session()
        t = oc.task(sid)
        res = client.call_tool("opencode_discard", {})
        self.assertIn("Discarded", self.text(res))
        self.assertFalse(os.path.isdir(t["dir"]))
        self.assertNotIn(t["branch"], git(oc.WORKSPACE, "branch", "--list"))
        self.assertIsNone(oc.task(sid))
        self.assertEqual(self.read("hello.py"), "print('hi')\n")

    def test_no_one_to_confirm_means_nothing_is_merged(self):
        self.ask(self.user())
        oc.SERVER.sink = None                           # no client: no user to ask
        res = oc.call_tool("opencode_merge", {})
        self.assertIn("did not merge", self.text(res))
        self.assertEqual(self.read("hello.py"), "print('hi')\n")

    def test_a_folder_that_is_not_a_repo_works_in_place(self):
        shutil.rmtree(os.path.join(oc.WORKSPACE, ".git"),
                      onerror=lambda f, p, e: (os.chmod(p, 0o700), f(p)))
        self.ask(self.user())
        self.assertFalse(oc.task(oc.last_session())["isolated"])
        self.assertEqual(self.read("hello.py"), "print('bye')\n")
        res = oc.call_tool("opencode_merge", {})
        self.assertTrue(res["isError"])
        self.assertIn("not in a copy", self.text(res))


class TestGrants(Base):
    """"Always allow" is the bridge's to keep, per task - seen and revocable."""

    def test_always_is_kept_by_the_bridge_and_opencode_hears_once(self):
        self.fake.script = [("permission", EDIT), ("permission", EDIT), ("say", "done")]
        client = self.user(self.accept(decision="always"))
        out = self.text(client.call_tool("opencode_ask", {"prompt": "x"}))
        self.assertEqual(len(self.asked), 1, "the second edit is covered by the grant")
        replies = [p[2]["reply"] for p in self.fake.posts if p[0] == "reply"]
        self.assertEqual(replies, ["once", "once"])
        self.assertIn("standing grant (every edit)", out)
        self.assertIn("every edit", self.text(oc.call_tool("opencode_grants", {})))

    def test_a_revoked_grant_is_asked_again(self):
        self.fake.script = [("permission", EDIT), ("say", "done")]
        client = self.user(self.accept(decision="always"), self.accept(decision="once"))
        client.call_tool("opencode_ask", {"prompt": "x"})
        self.assertIn("Took back", self.text(oc.call_tool("opencode_revoke", {"number": 1})))
        self.assertIn("no standing grants", self.text(oc.call_tool("opencode_grants", {})))
        self.fake.script = [("permission", EDIT), ("say", "done")]
        client.call_tool("opencode_ask", {"prompt": "y"})
        self.assertEqual(len(self.asked), 2)

    def test_a_command_grant_covers_that_command_only(self):
        sid = oc.new_session("x")["id"]
        oc.update_task(sid, grants=[])
        oc.add_grant(sid, BASH)
        self.assertIsNotNone(oc.granted(sid, BASH))
        self.assertIsNotNone(oc.granted(sid, dict(BASH, patterns=["python hello.py -v"])))
        self.assertIsNone(oc.granted(sid, dict(BASH, patterns=["rm -rf ."])))
        self.assertIsNone(oc.granted(sid, EDIT))


EDIT_COPY = dict(EDIT, metadata=dict(EDIT["metadata"], filepath="<copy>/hello.py"))
RUN_TESTS = {"permission": "bash", "patterns": ["python -m unittest tests.test_hello"],
             "metadata": {"command": "python -m unittest tests.test_hello"},
             "always": ["python -m unittest *"]}
FETCH = {"permission": "webfetch", "patterns": ["https://example.com/"],
         "metadata": {"url": "https://example.com/"}, "always": ["*"]}
PASSING = ("import unittest\n\nclass T(unittest.TestCase):\n"
           "    def test_it(self):\n        self.assertTrue(True)\n")
FAILING = PASSING.replace("self.assertTrue(True)", "self.fail('boom')")


class TestAgentic(Repo):
    """Agentic is the user's switch. With it on, in a task's own copy, what
    stays inside the copy goes through without a card and an ask that ends
    with work left goes back to OpenCode; everything else is as before."""

    def setUp(self):
        Repo.setUp(self)
        self.switch(True)

    def switch(self, on):
        os.makedirs(oc.STATE_DIR, exist_ok=True)
        with open(os.path.join(oc.STATE_DIR, oc.AGENTIC_FILE), "w", encoding="utf-8") as f:
            json.dump({"on": on}, f)

    def out(self, res):
        return "\n".join(c["text"] for c in res["content"])

    def asks(self):
        return [p[2]["parts"][0]["text"] for p in self.fake.posts if p[0] == "ask"]

    def test_edits_and_tests_in_the_copy_go_through_without_a_card(self):
        res = self.ask(self.user(), "change the greeting",
                       ("permission", EDIT_COPY), ("write", ("hello.py", "print('bye')\n")),
                       ("permission", RUN_TESTS), ("say", "Changed it."))
        self.assertFalse(res.get("isError"), res)
        self.assertEqual(self.asked, [], "no card was shown")
        self.assertEqual([p[2] for p in self.fake.posts if p[0] == "reply"],
                         [{"reply": "once"}, {"reply": "once"}])
        out = self.out(res)
        self.assertIn("allowed by Agentic: edit hello.py", out)
        self.assertIn("allowed by Agentic: run python -m unittest tests.test_hello", out)
        self.assertIn("Tests passed: tests.test_hello", out)
        self.assertEqual(self.read("hello.py"), "print('hi')\n", "the user's folder is untouched")
        self.assertTrue(self.asks()[0].startswith(oc.AGENTIC_NOTE))
        self.assertTrue(self.asks()[0].endswith("change the greeting"))

    def test_off_every_step_is_asked_as_before(self):
        self.switch(False)
        res = self.ask(self.user(self.accept(decision="once"), self.accept(decision="once")),
                       "change the greeting",
                       ("permission", EDIT_COPY), ("permission", RUN_TESTS), ("say", "ok"))
        self.assertEqual(len(self.asked), 2)
        self.assertNotIn("Agentic", self.out(res))
        self.assertEqual(self.asks(), ["change the greeting"])
        # A switch file that cannot be read is off, not on.
        with open(os.path.join(oc.STATE_DIR, oc.AGENTIC_FILE), "w") as f:
            f.write("{not json")
        self.assertFalse(oc.agentic())
        with open(os.path.join(oc.STATE_DIR, oc.AGENTIC_FILE), "w") as f:
            json.dump({"on": "yes"}, f)
        self.assertFalse(oc.agentic(), "only true is on")

    def test_what_leaves_the_copy_or_is_not_on_the_list_is_still_the_users(self):
        refuse = self.accept(decision="reject")
        res = self.ask(self.user(refuse, refuse, refuse, refuse), "review the greeting",
                       ("permission", EDIT),                   # the user's folder, not the copy
                       ("permission", BASH),                   # python hello.py
                       ("permission", FETCH),
                       ("permission", dict(EDIT_COPY, metadata=dict(
                           EDIT_COPY["metadata"], filepath="<copy>/.git/config"))),
                       ("say", "ok"))
        self.assertEqual(len(self.asked), 4)
        self.assertNotIn("allowed by Agentic", self.out(res))
        self.assertEqual([p[2]["reply"] for p in self.fake.posts if p[0] == "reply"],
                         ["reject"] * 4)

    def test_no_user_to_ask_still_lets_the_copys_own_steps_through(self):
        """A client that cannot ask refuses every step that would need a card;
        what Agentic answers never needed one."""
        self.fake.script = [("permission", EDIT_COPY), ("say", "ok")]
        res = oc.call_tool("opencode_ask", {"prompt": "change the greeting"})
        self.assertFalse(res.get("isError"), res)
        self.fake.script = [("permission", BASH), ("say", "never")]
        res = oc.call_tool("opencode_ask", {"prompt": "run it"})
        self.assertTrue(res["isError"])
        self.assertIn("no one could approve", self.text(res))

    def test_a_folder_worked_in_place_asks_at_every_step(self):
        shutil.rmtree(os.path.join(oc.WORKSPACE, ".git"),
                      onerror=lambda f, p, e: (os.chmod(p, 0o700), f(p)))
        self.ask(self.user(self.accept(decision="once")), "change the greeting",
                 ("permission", EDIT), ("say", "ok"))
        self.assertEqual(len(self.asked), 1, "no copy, no undo: the user decides")
        self.assertEqual(self.asks(), ["change the greeting"])

    def test_the_commands_it_runs_unasked_are_tests_and_reads_inside_the_copy(self):
        copy = os.path.join(self.tmp, "copy")
        os.makedirs(os.path.join(copy, "core"))
        inside = copy.replace("\\", "/")
        for cmd in ("python -m unittest tests.test_hello",
                    "python -m unittest discover -s tests -p \"test_*.py\"",
                    "python3 -m unittest tests.test_hello -v",
                    "python -m pytest tests/test_hello.py -v", "pytest -q tests",
                    "py -m py_compile core/agent.py",
                    "Python.exe -m unittest",
                    "git status", "git diff --stat", "git log -5 --oneline", "git show HEAD",
                    "ls", "ls -la core", "cat core/agent.py", "grep -rn \"def fit\" core",
                    "rg fit_window", "head -40 core/agent.py", "wc -l core/agent.py",
                    "dir /b core", "findstr /s /i fit_window *.py", "pwd",
                    "cd core && python -m unittest tests.test_hello",
                    "cd %s && git status" % inside,
                    "cat %s/core/agent.py" % inside,
                    # As the model wrote it, live: the copy's own path, quoted.
                    "cd \"%s\" && python -m unittest tests.test_hello" % copy,
                    "cat \"%s\"" % os.path.join(copy, "core", "agent.py"),
                    "cat core\\agent.py"):
            self.assertTrue(oc.safe_command(cmd, copy), cmd)
        for cmd in ("", "   ", None,
                    "python hello.py", "python -c \"import os\"", "python -m pip install x",
                    "python -c \"import hello; print(repr(hello.greet('Ada')))\"",
                    "pytest --rootdir=/ tests", "python -m pytest ../other",
                    "python -munittest", "pip install requests", "npm install",
                    "rm -rf .", "del hello.py", "git commit -m x", "git push",
                    "git checkout main", "git reset --hard", "git -C .. status",
                    "curl https://example.com", "echo hi", "find . -delete",
                    "python -m unittest; rm -rf .", "python -m unittest & del x",
                    "python -m unittest || rm x", "python -m unittest | tee out.txt",
                    "python -m unittest > out.txt", "cat < hello.py",
                    "python -m unittest && rm -rf .", "cat `which python`",
                    "cat $(which python)", "cat $HOME/.ssh/id_rsa", "type %USERPROFILE%\\x",
                    "cat ../secret", "cat core/../../secret", "cat \"..\"/secret",
                    "cat ~/.ssh/id_rsa", "cat /etc/passwd", "cat C:/Windows/win.ini",
                    "cat /c/Windows/win.ini", "cat {..,}/secret",
                    # A backslash is a path's in cmd and one Git Bash drops:
                    # outside quotes both readings have to stay in the copy.
                    "cat .\\./secret", "cat ..\\secret", "cat \"..\\secret\"",
                    "cat \"..\\\\secret\"", "cat core\\..\\..\\secret",
                    "cd \"%s\" && git status" % self.tmp,
                    "cat \"C:\\Windows\\win.ini\"", "cat C:\\Windows\\win.ini",
                    "cat \"..\"\\secret", "type \\\\server\\share\\x",
                    "git diff --output=/tmp/x", "git diff --output=../x",
                    "cd .. && git status", "cd /", "cd", "cd a b",
                    "ls /s", "cat \"unclosed", "\"python\" -m unittest",
                    "./python -m unittest"):
            self.assertFalse(oc.safe_command(cmd, copy), cmd)

    def test_a_line_is_judged_whole_and_command_by_command(self):
        """OpenCode names each command of a line; one that is not on the list
        puts the whole line to the user, however it is reported."""
        sid = oc.start_task("x")
        hidden = dict(RUN_TESTS, metadata={"command": "python -m unittest && rm -rf ."})
        listed = dict(RUN_TESTS, patterns=["python -m unittest", "rm -rf ."], metadata={})
        bare = dict(RUN_TESTS, metadata={})
        self.assertIsNone(oc.agentic_allows(sid, hidden))
        self.assertIsNone(oc.agentic_allows(sid, listed))
        self.assertEqual(oc.agentic_allows(sid, bare), "run python -m unittest tests.test_hello")
        self.assertIsNone(oc.agentic_allows(sid, dict(RUN_TESTS, patterns=[], metadata={})))
        self.assertIsNone(oc.agentic_allows(sid, {"permission": "doom_loop", "patterns": ["*"]}))
        self.assertIsNone(oc.agentic_allows(sid, {"permission": "sequential_thinking",
                                                  "patterns": ["*"]}))

    def test_failing_tests_go_back_to_opencode_until_they_pass(self):
        self.fake.scripts = [[("write", ("tests/test_hello.py", FAILING)), ("say", "done")],
                             [("write", ("tests/test_hello.py", PASSING)), ("say", "fixed")]]
        res = self.user().call_tool("opencode_ask", {"prompt": "change the test"})
        self.assertFalse(res.get("isError"), res)
        first, second = self.asks()
        self.assertIn("The tests of what you changed failed", second)
        self.assertIn("boom", second)
        out = self.out(res)
        self.assertIn("fixed", out)
        self.assertIn("Tests passed: tests.test_hello", out)
        self.assertNotIn("Tests FAILED", out)
        self.assertIn("sent OpenCode back to work 1 time(s) by itself: the tests failed", out)
        self.assertEqual(len(oc.task(oc.last_session())["checkpoints"]), 2,
                         "each round can be undone by itself")

    def test_it_is_sent_back_only_so_often(self):
        n = [0]

        def worse():
            n[0] += 1
            return [("write", ("tests/test_hello.py", FAILING + "# %d\n" % n[0])),
                    ("say", "tried")]
        self.fake.scripts = [worse() for _ in range(oc.AGENTIC_ROUNDS + 3)]
        res = self.user().call_tool("opencode_ask", {"prompt": "change the test"})
        self.assertEqual(len(self.asks()), 1 + oc.AGENTIC_ROUNDS)
        out = self.out(res)
        self.assertIn("Tests FAILED", out)
        self.assertIn("sent OpenCode back to work %d time(s)" % oc.AGENTIC_ROUNDS, out)
        self.assertIn("the user's call", out)
        # The next ask starts its own count.
        self.fake.scripts = [[("write", ("tests/test_hello.py", PASSING)), ("say", "fixed")]]
        res = self.user().call_tool("opencode_ask", {"prompt": "make the test pass"})
        self.assertNotIn("sent OpenCode back", self.out(res))
        self.assertIn("Tests passed", self.out(res))

    def test_to_dos_left_open_send_it_back_and_an_earlier_list_does_not(self):
        plan = [{"content": "edit hello.py", "status": "pending"},
                {"content": "run the tests", "status": "pending"}]
        done = [dict(t, status="completed") for t in plan]
        self.fake.scripts = [[("todos", plan), ("say", "I will edit hello.py next.")],
                             [("write", ("hello.py", "print('bye')\n")), ("todos", done),
                              ("say", "Changed it.")]]
        res = self.user().call_tool("opencode_ask", {"prompt": "change the greeting"})
        first, second = self.asks()
        self.assertIn("2 to-do(s) still open", second)
        self.assertIn("- edit hello.py", second)
        self.assertIn("2 to-do(s) were open", self.out(res))
        # A list the ask never touched is an earlier ask's: not its work.
        self.fake.todos = plan
        self.fake.scripts = [[("write", ("hello.py", "print('ciao')\n")), ("say", "ok")]]
        self.user().call_tool("opencode_ask", {"prompt": "change it again"})
        self.assertEqual(len(self.asks()), 3)

    def test_a_change_not_made_is_asked_for_once_and_a_question_never(self):
        self.fake.script = [("say", "I would edit hello.py.")]
        res = self.user().call_tool("opencode_ask", {"prompt": "change the greeting"})
        first, second = self.asks()
        self.assertIn("Nothing in your copy has changed yet", second)
        self.assertIn("nothing had changed", self.out(res))
        self.user().call_tool("opencode_ask", {"prompt": "What does hello.py print?"})
        self.assertEqual(len(self.asks()), 3, "a question changes nothing, and need not")
        for text, change in (("change the greeting", True),
                             ("can you make the button blue", True),
                             ("Add a Copy button\nto each row", True),
                             ("what tools would be beneficial?", False),
                             ("how does the bridge start", False),
                             ("explain fit_window", False),
                             ("list all tooling", False),
                             ("What's in core?", False),
                             ("make it faster?", False),
                             ("", False)):
            self.assertEqual(oc.wants_change(text), change, text)

    def test_out_of_work_with_work_left_is_owed_to_the_wait(self):
        class Later:
            """The clock, moved on by what the tests are said to have taken."""
            by = 0.0

            def __getattr__(self, name):
                return getattr(real_time, name)

            def monotonic(self):
                return real_time.monotonic() + self.by
        real_time, real_after = oc.time, oc.after_ask
        later = Later()

        def slow(sid, label):
            later.by += 1000
            return real_after(sid, label)
        oc.time, oc.after_ask = later, slow
        try:
            self.fake.scripts = [[("write", ("tests/test_hello.py", FAILING)), ("say", "done")]]
            res = self.user().call_tool("opencode_ask", {"prompt": "change the test"})
        finally:
            oc.time, oc.after_ask = real_time, real_after
        sid = oc.last_session()
        self.assertTrue(res["isError"])
        self.assertIn("has work left (the tests failed)", self.text(res))
        self.assertIn("opencode_wait", self.text(res))
        self.assertEqual(res["_meta"]["studio/opencode"], {"session": sid, "state": "unfinished"})
        self.assertEqual(len(self.asks()), 1)
        self.assertEqual(oc.task(sid)["owed"][0], "the tests failed")
        self.fake.scripts = [[("write", ("tests/test_hello.py", PASSING)), ("say", "fixed")]]
        res = self.user().call_tool("opencode_wait", {"session_id": sid})
        self.assertFalse(res.get("isError"), res)
        self.assertIn("The tests of what you changed failed", self.asks()[1])
        self.assertIn("Tests passed", self.out(res))
        self.assertIn("1 time(s)", self.out(res))
        self.assertFalse(oc.task(sid).get("owed"))
        self.assertEqual(res["_meta"]["studio/opencode"]["state"], "done")

    def test_stop_ends_the_tests_and_the_run(self):
        real = (oc.TEST_LOOK, oc.run_tests, studio_mcp.cancelled)
        slow = ("import time, unittest\n\nclass T(unittest.TestCase):\n"
                "    def test_it(self):\n        time.sleep(60)\n")
        self.fake.script = [("write", ("tests/test_hello.py", slow)), ("say", "done")]
        testing = []

        def run_tests(root, mods):
            testing.append(mods)
            return real[1](root, mods)
        oc.TEST_LOOK = 0.2
        oc.run_tests = run_tests
        studio_mcp.cancelled = lambda: bool(testing)    # Stop comes while the tests run
        try:
            res = self.user().call_tool("opencode_ask", {"prompt": "change the test"})
        finally:
            oc.TEST_LOOK, oc.run_tests, studio_mcp.cancelled = real
        self.assertTrue(res["isError"])
        self.assertIn("The user stopped OpenCode", self.text(res))
        self.assertEqual(len(self.asks()), 1)
        self.assertEqual(res["_meta"]["studio/opencode"]["state"], "stopped")

    def test_status_and_grants_say_it_and_no_tool_sets_it(self):
        self.assertIn("The user has Agentic on", self.text(oc.call_tool("opencode_status", {})))
        self.ask(self.user())
        self.assertIn("The user has Agentic on", self.text(oc.call_tool("opencode_grants", {})))
        self.switch(False)
        self.assertNotIn("Agentic", self.text(oc.call_tool("opencode_status", {})))
        self.assertIn("every step is asked", self.text(oc.call_tool("opencode_grants", {})))
        for t in oc.tool_list():
            self.assertNotIn("agentic", t["name"].lower())
            self.assertNotIn("agentic", json.dumps(t["inputSchema"]).lower(), t["name"])
        self.assertEqual(oc.AGENTIC_FILE, eng.OPENCODE_AGENTIC)


class TestEventsAndContext(Base):

    def test_the_event_stream_wakes_the_follower_and_falls_back_when_it_drops(self):
        lines = [b'data: {"type":"server.connected","properties":{}}\n', b"\n",
                 b'data: {"type":"permission.asked","properties":{}}\n', b"\n"]

        class Stream:
            headers = {"Content-Type": "text/event-stream"}

            def __iter__(self):
                return iter(lines)

            def close(self):
                pass
        real = oc._open
        oc._open = lambda req, timeout: Stream()
        try:
            ev = oc.Events()
            self.assertTrue(ev.flag.wait(2), "an event woke it")
            for _ in range(50):              # the stream ends: polling takes over
                if not ev.alive:
                    break
                ev.flag.wait(0.05)
            self.assertFalse(ev.alive)
        finally:
            oc._open = real
        self.assertFalse(oc.Events().alive, "the fake server has no /event: poll as before")

    def test_the_report_says_how_full_the_context_is_and_when_it_compacted(self):
        os.makedirs(oc.STATE_DIR, exist_ok=True)
        with open(os.path.join(oc.STATE_DIR, "opencode.json"), "w") as f:
            json.dump(eng.opencode_config("http://h:1/v1", "m1", ["m1"], 65536), f)
        msgs = [{"info": {"role": "assistant", "id": "a", "summary": True}, "parts": []},
                {"info": {"role": "assistant", "id": "b",
                          "tokens": {"input": 50000, "output": 3000,
                                     "cache": {"read": 2000, "write": 0}}}, "parts": []}]
        self.assertEqual(oc.context_used(msgs), (55000, 1))
        out = oc.context_report(msgs)
        self.assertIn("55000 of 65536 tokens (83%)", out)
        self.assertIn("close to full", out)
        self.assertIn("compacted this session 1 time", out)
        self.assertEqual(oc.context_report([]), "")


class TestTestsFor(unittest.TestCase):
    def test_changed_files_map_to_their_tests_in_the_packaged_layout(self):
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, True)
        os.makedirs(os.path.join(root, "tests"))
        for t in ("mcp", "opencode", "comfy", "comfy_view", "imagegen", "agent", "update"):
            open(os.path.join(root, "tests", "test_%s.py" % t), "w").close()
        cases = {
            "apps/opencode/mcp.py": ["tests.test_opencode"],     # not core's test_mcp
            "apps/opencode/codeaddons_ui.py": ["tests.test_opencode"],
            "apps\\opencode\\mcp.py": ["tests.test_opencode"],
            "apps/comfyui/mcp.py": ["tests.test_comfy"],
            "apps/comfyui/view.py": ["tests.test_comfy_view"],
            "apps/image_studio/imagegen.py": ["tests.test_imagegen"],
            "apps/image_studio/ui.py": ["tests.test_imagegen"],
            "core/mcp.py": ["tests.test_mcp"],
            "core/agent.py": ["tests.test_agent"],
            "studio_update.py": ["tests.test_update"],
            "tests/test_mcp.py": ["tests.test_mcp"],
            "core/chat.py": [],
            "apps/opencode/read_cap.js": [],
        }
        for f, want in cases.items():
            self.assertEqual(oc.tests_for(root, [f]), want, f)

if __name__ == "__main__":
    unittest.main()
