"""OpenCode's trainer: the MCP server Claude Code uses to review OpenCode's
tasks and keep lessons (apps/opencode/trainer_mcp.py), and the notebook
files it shares with the window (core/lessons.py)."""

import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

import core.lessons as lessons
import apps.opencode.mcp as oc
import apps.opencode.trainer_mcp as tm


def call(name, args=None):
    """Through the protocol, as Claude Code calls it."""
    reply = tm.SERVER.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                              "params": {"name": name, "arguments": args or {}}})
    res = reply["result"]
    return res["content"][0]["text"], bool(res.get("isError"))


def tool_call(cid, name, args="{}"):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": args}}


class Base(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.state = os.path.join(self.dir, "state")
        os.makedirs(self.state)
        patches = [mock.patch.dict(os.environ, {"STUDIO_SETTINGS":
                                                os.path.join(self.dir, "settings.json")}),
                   mock.patch.object(oc, "STATE_DIR", self.state),
                   mock.patch.object(oc, "WORKSPACE", self.dir)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        # The notebooks are shared per file within a process; start clean.
        lessons._SHARED.clear()
        self.addCleanup(lessons._SHARED.clear)
        tm.SERVER.handle({"jsonrpc": "2.0", "id": 0, "method": "initialize",
                          "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                                     "clientInfo": {"name": "test", "version": "1"}}})

    def save_task(self, task_id, briefs, messages, status="response complete", age=0):
        d = tm.records_dir()
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, task_id + ".json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "record": {"id": task_id, "app_id": "opencode",
                                                "briefs": briefs, "status": status},
                       "messages": [{"role": "system", "content": "SECRET PROMPT"}] + messages}, f)
        t = 1_800_000_000 - age
        os.utime(path, (t, t))


class TestReading(Base):
    def test_tasks_newest_first_with_trouble(self):
        self.save_task("aaaa1111", ["add a button"], [
            {"role": "user", "content": "add a button"},
            {"role": "assistant", "content": "", "tool_calls": [tool_call("c1", "opencode_ask")]},
            {"role": "tool", "tool_call_id": "c1",
             "content": "Session ses_abc1 is done.\nallowed: edit core/chat.py"}], age=100)
        self.save_task("bbbb2222", ["rename x", "no, the other x"], [
            {"role": "user", "content": "rename x"},
            {"role": "assistant", "content": "", "tool_calls": [tool_call("c1", "opencode_ask")]},
            {"role": "tool", "tool_call_id": "c1",
             "content": "Session ses_def2 stopped.\nrefused: run pip install foo"},
            {"role": "assistant", "content": "", "tool_calls": [tool_call("c2", "opencode_undo")]},
            {"role": "tool", "tool_call_id": "c2", "content": "TOOL ERROR: busy"}],
            status="needs attention: x", age=0)
        text, error = call("trainer_tasks")
        self.assertFalse(error)
        self.assertLess(text.index("bbbb2222"), text.index("aaaa1111"))
        clean, troubled = text.split("aaaa1111")[1], text.split("aaaa1111")[0]
        self.assertIn("clean", clean)
        for sign in ("ended as: needs attention", "corrected", "undid", "1 step(s) refused",
                     "1 failing call"):
            self.assertIn(sign, troubled)
        self.assertIn("ses_def2", troubled)

    def test_task_transcript_names_tools_and_hides_the_prompt(self):
        self.save_task("cccc3333", ["fix it"], [
            {"role": "user", "content": "fix it"},
            {"role": "assistant", "content": "On it.",
             "tool_calls": [tool_call("c1", "opencode_ask", '{"prompt": "fix"}')]},
            {"role": "tool", "tool_call_id": "c1", "content": "x" * 50}])
        text, _ = call("trainer_task")           # no id: the newest
        self.assertIn("USER: fix it", text)
        self.assertIn('[calls opencode_ask({"prompt": "fix"})]', text)
        self.assertIn("[result of opencode_ask]", text)
        self.assertNotIn("SECRET PROMPT", text)

    def test_long_transcript_comes_in_windows(self):
        self.save_task("dddd4444", ["x"], [{"role": "user", "content": "y" * (tm.WINDOW + 500)}])
        text, _ = call("trainer_task", {"task_id": "dddd4444"})
        self.assertIn("Next: start=%d." % tm.WINDOW, text)
        text, _ = call("trainer_task", {"task_id": "dddd4444", "start": tm.WINDOW})
        self.assertNotIn("Next:", text)

    def test_bad_ids_are_refused_in_words(self):
        text, error = call("trainer_task", {"task_id": "..\\..\\settings"})
        self.assertTrue(error)
        self.assertIn("hex", text)
        text, error = call("trainer_task", {"task_id": "abcdef12"})
        self.assertTrue(error)
        self.assertIn("No saved OpenCode task", text)

    def test_diff_of_a_forgotten_session(self):
        text, error = call("trainer_diff", {"session_id": "ses_gone"})
        self.assertTrue(error)
        self.assertIn("No task record", text)

    def test_session_when_opencode_is_down(self):
        with mock.patch.object(oc, "t_get_session",
                               side_effect=oc.OpenCodeError("Cannot reach OpenCode")):
            text, error = call("trainer_session", {"session_id": "ses_1"})
        self.assertTrue(error)
        self.assertIn("trainer_task", text)


class TestTeaching(Base):
    def test_keep_goes_to_the_folder_and_reaches_opencode(self):
        text, error = call("trainer_keep", {"lesson": "Run tests.test_chat after editing core/chat.py."})
        self.assertFalse(error)
        self.assertIn("this folder", text)
        with open(os.path.join(self.state, "lessons.md"), encoding="utf-8") as f:
            self.assertIn("Run tests.test_chat after editing core/chat.py.", f.read())
        text, _ = call("trainer_lessons")
        self.assertIn("the trainer kept it", text)
        text, _ = call("trainer_keep", {"lesson": "Run tests.test_chat after editing core/chat.py."})
        self.assertIn("Already", text)

    def test_everywhere_scope(self):
        text, _ = call("trainer_keep", {"lesson": "Keep commit messages to one line.",
                                        "scope": "everywhere"})
        self.assertIn("every tab", text)

    def test_forget_and_the_users_own_lessons(self):
        stack = tm.notebook()
        stack.add("Answer in British English.", "user")
        stack.add("Grep before reading a long file.", "model")
        text, error = call("trainer_forget", {"lesson": "Answer in British English."})
        self.assertTrue(error)
        self.assertIn("user's own", text)
        text, error = call("trainer_forget", {"lesson": "Grep before reading a long file."})
        self.assertFalse(error)
        text, error = call("trainer_forget", {"lesson": "Answer in British English.",
                                              "user_agreed": True})
        self.assertFalse(error)
        self.assertIn("No lessons yet", call("trainer_lessons")[0])

    def test_schema_refuses_a_non_lesson(self):
        reply = tm.SERVER.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                  "params": {"name": "trainer_keep",
                                             "arguments": {"lesson": "short"}}})
        refused = "error" in reply or reply["result"].get("isError")
        self.assertTrue(refused)
        self.assertIn("No lessons yet", call("trainer_lessons")[0])


class TestSharedNotebookFile(unittest.TestCase):
    """The window and the trainer are two processes over one file: neither
    may save its old list over the other's lesson."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.path = os.path.join(self.dir, "f.json")

    def test_a_lesson_kept_elsewhere_survives_the_next_save(self):
        window = lessons.Notebook("f", self.path)
        window.load()
        window.add("First lesson from the window.", "model")
        window.brief()
        other = lessons.Notebook("f", self.path)        # the trainer's process
        other.load()
        other.add("A lesson the trainer kept.", "trainer")
        window.add("Second lesson from the window.", "model")
        again = lessons.Notebook("f", self.path)
        again.load()
        self.assertEqual({l["text"] for l in again.lessons},
                         {"First lesson from the window.", "A lesson the trainer kept.",
                          "Second lesson from the window."})

    def test_the_window_sees_it_on_its_next_request(self):
        window = lessons.Notebook("f", self.path)
        window.load()
        window.brief()
        other = lessons.Notebook("f", self.path)
        other.load()
        other.add("A lesson the trainer kept.", "trainer")
        self.assertIn("A lesson the trainer kept.", window.fresh())

    def test_trainer_outranks_the_model_not_the_user(self):
        nb = lessons.Notebook("f", self.path)
        nb.add("Use the repo map first.", "model")
        nb.add("Use the repo map first.", "trainer")
        self.assertEqual(nb.find("Use the repo map first.")["source"], "trainer")
        nb.add("Use the repo map first.", "user")
        self.assertEqual(nb.find("Use the repo map first.")["source"], "user")


if __name__ == "__main__":
    unittest.main()
