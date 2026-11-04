"""Live activity describes evidence; it cannot execute or approve tools."""
import unittest
import io
import json
import threading
from unittest.mock import Mock, patch

import core.agent as eng
import core.mcp as mcp
import apps.opencode.mcp as oc
from core.tasks import Executor

from core.activity import Activity, MAX_ITEMS, bridge_event, progress_event, tool_presentation, progress_presentation


class TestActivity(unittest.TestCase):
    def test_progress_badges_preserve_reported_facts_without_inventing_a_percent(self):
        text = "OpenCode is working; 5 file(s) changed; last activity 0s ago"
        badges, fraction = progress_presentation(text)
        self.assertEqual(badges, [("5 files changed", "muted"), ("Active now", "ok")])
        self.assertIsNone(fraction)
        badges, fraction = progress_presentation("OpenCode is working; 2/5 to-dos (40%); 1 file(s) changed")
        self.assertAlmostEqual(fraction, .4)
        self.assertIn(("1 file changed", "muted"), badges)
        self.assertIsNone(progress_presentation("9/0 to-dos (0%)")[1])
        self.assertIn(("No activity for 2m", "warn"), progress_presentation(
            "OpenCode is working; nothing for 2m - may be stuck; Stop ends it")[0])

    def test_readable_tool_labels_keep_source_and_context(self):
        title, context, icon, source = tool_presentation("opencode_ask", {"prompt": "Improve\nthe interface"})
        self.assertEqual((title, context, icon, source),
                         ("Work on your request", "Improve the interface", "code", "OpenCode"))
        self.assertEqual(tool_presentation("timeline", {"action": "add_marker"})[0], "Add marker")
        self.assertEqual(tool_presentation("custom_weird_tool", "incomplete JSON")[0], "Custom weird tool")

    def test_preparation_is_visible_without_claiming_execution(self):
        a = Activity()
        a.handle("tool_preparing", {"name": "read_file"})
        self.assertEqual(a.count, 0)
        self.assertIn("Preparing read_file", a.phase)
        a.handle("error", "cut off")
        a.handle("idle", False)
        self.assertEqual(a.phase, "Interrupted")
        self.assertFalse(a.active)

    def test_nested_same_name_results_pair_with_the_latest_running_call(self):
        a = Activity()
        for value in ("outer", "inner"):
            a.handle("tool", {"name": "read_file", "arguments": {"path": value}})
        a.handle("tool_result", {"name": "read_file", "text": "inner result", "status": "error"})
        self.assertEqual([i["status"] for i in a.items], ["running", "error"])
        a.handle("progress", "opening outer")
        self.assertIn("opening outer", a.phase)
        a.handle("tool_result", {"name": "read_file", "text": "outer result", "status": "ok"})
        a.handle("token", "text does not identify a tool")
        self.assertEqual(a.phase, "Writing reply…")
        self.assertEqual(a.count, 2)
        self.assertIn("inner result", a.as_text())
        self.assertIn("outer result", a.as_text())
        a.handle("idle", False)
        self.assertIn("1 failed tool call", a.phase)

    def test_bridge_call_ids_keep_overlapping_tools_separate(self):
        a = Activity()
        for ident in ("session:first", "session:second"):
            a.handle("bridge_activity", {"id": ident, "name": "bash", "status": "running"})
        a.handle("bridge_activity", {"id": "session:first", "name": "bash", "status": "ok", "text": "exit 0"})
        self.assertEqual(a.count, 2)
        self.assertEqual([i["status"] for i in a.items], ["ok", "running"])
        a.handle("idle", True)
        self.assertEqual(a.items[1]["status"], "unknown")
        self.assertEqual(a.phase, "Interrupted")
        self.assertIn("No result received", a.as_text())

    def test_finished_turn_and_next_turn_keep_distinct_tool_counts(self):
        a = Activity()
        a.handle("tool", {"name": "read_file", "arguments": {}})
        a.handle("tool_result", {"name": "read_file", "text": "ok"})
        a.handle("idle", False)
        self.assertEqual(a.phase, "Reply finished")
        a.handle("begin")
        self.assertEqual((a.count, a.items, a.phase), (0, [], "Thinking…"))

    def test_activity_bounds_detail_and_history_and_names_omissions(self):
        a = Activity()
        a.handle("tool", {"name": "first", "arguments": {}})
        a.handle("tool_result", {"name": "first", "text": "failed", "status": "error"})
        for _ in range(MAX_ITEMS + 5):
            a.handle("tool", {"name": "read_file", "arguments": {"path": "x" * 4000}})
            a.handle("tool_result", {"name": "read_file", "text": "y" * 4000})
        self.assertEqual(len(a.items), MAX_ITEMS)
        self.assertEqual(a.omitted, 6)
        self.assertIn("6 earlier tool", a.as_text())
        self.assertIn("clipped in the activity view", a.as_text())
        a.handle("idle", False)
        self.assertIn("1 failed tool call", a.phase)

    def test_invalid_progress_metadata_cannot_become_activity(self):
        for value in (None, [], {}, {"id": "a", "name": "bash", "status": "approved"},
                      {"id": "x" * 241, "name": "bash", "status": "running"}):
            self.assertIsNone(bridge_event(value))
        for params in ({"_meta": []}, {"_meta": "bad"}, None):
            self.assertIsNone(progress_event(params))


def stream(chunks, finish="tool_calls"):
    lines = ["data: " + json.dumps({"choices": [{"delta": c, "finish_reason": None}]}) for c in chunks]
    lines += ["data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": finish}]}), "data: [DONE]"]
    return io.BytesIO("\n".join(lines).encode("utf-8"))


class TestStreamingActivity(unittest.TestCase):
    def test_preparing_event_precedes_dispatch_and_text_survives(self):
        llm = eng.LLM("http://test/v1", "test")
        first = stream([{"content": "I'll read it."}, {"tool_calls": [{"index": 0, "id": "call-a",
                        "function": {"name": "read_file", "arguments": "{"}}]},
                        {"tool_calls": [{"index": 0, "function": {"arguments": "}"}}]}])
        second = stream([{"content": "Read it."}], "stop")
        bridge = Mock()
        bridge.call_tool.return_value = {"content": [{"type": "text", "text": "file contents"}]}
        events = []
        tools = [{"name": "read_file", "description": "Read", "inputSchema": {"type": "object"},
                  "annotations": {"readOnlyHint": True}}]
        with patch.object(llm, "_open", side_effect=[first, second]):
            ex = Executor(llm, bridge, eng.to_openai_tools(tools), schemas=tools,
                          emit=lambda kind, payload: events.append((kind, payload)))
            ex.run([{"role": "user", "content": "Read the file"}])
        kinds = [k for k, _ in events]
        self.assertLess(kinds.index("tool_preparing"), kinds.index("tool"))
        self.assertEqual(kinds.count("tool_preparing"), 1, "argument chunks cannot flood the UI")
        self.assertIn(("token", "I'll read it."), events)
        bridge.call_tool.assert_called_once()

    def test_stop_during_tool_arguments_cannot_dispatch_partial_json(self):
        llm = eng.LLM("http://test/v1", "test")
        cancel = threading.Event()
        events = []
        def emit(kind, payload):
            events.append((kind, payload))
            if kind == "tool_preparing":
                cancel.set()
        response = stream([{"tool_calls": [{"index": 0, "id": "a",
                           "function": {"name": "read_file", "arguments": "{"}}]},
                           {"tool_calls": [{"index": 0, "function": {"arguments": "}"}}]}])
        bridge = Mock()
        with patch.object(llm, "_open", return_value=response):
            ex = Executor(llm, bridge, [], cancel=cancel, emit=emit)
            ex.run([{"role": "user", "content": "Read"}])
        bridge.call_tool.assert_not_called()
        self.assertNotIn("tool", [k for k, _ in events])

    def test_mcp_activity_rejects_expired_tokens_and_display_failures_are_isolated(self):
        client = object.__new__(eng.MCPClient)
        client.quiet = True
        seen = []
        client.on_activity = seen.append
        params = {"progressToken": 4, "_meta": {"studio/toolActivity": {
            "id": "s:c", "name": "bash", "status": "running", "arguments": {"command": "test"}}}}
        client._notification("notifications/progress", params, request_id=5)
        self.assertFalse(seen)
        client._notification("notifications/progress", params, request_id=4)
        self.assertEqual(seen[0]["id"], "s:c")
        client.on_activity = Mock(side_effect=RuntimeError("closed view"))
        client._notification("notifications/progress", params, request_id=4)
        client.on_activity.assert_called_once()

    def test_loopback_delivers_tool_activity_without_affecting_the_result(self):
        def run(args):
            mcp.activity("s:c", "read", "running", args)
            mcp.activity("s:c", "read", "ok", args, "read it")
            return mcp.result("done")
        server = mcp.Server("activity-test", "1", [mcp.Tool(
            "observe", run, "Observe", {"type": "object"}, read_only=True)])
        client = mcp.Loopback(server)
        client.initialize()
        seen = []
        client.on_activity = seen.append
        self.assertIn("done", client.call_tool("observe", {})["content"][0]["text"])
        self.assertEqual([i["status"] for i in seen], ["running", "ok"])
        client.on_activity = Mock(side_effect=RuntimeError("view closed"))
        self.assertIn("done", client.call_tool("observe", {})["content"][0]["text"])
        client.close()


class TestOpenCodeActivity(unittest.TestCase):
    def part(self, status="running", sid="session", **kw):
        return {"type": "tool", "sessionID": sid, "messageID": "new", "id": "part",
                "callID": "call", "tool": "bash", "state": {"status": status,
                "input": {"command": "python -m unittest"}, "output": "exit 0"}, **kw}

    def test_part_updates_are_scoped_deduplicated_and_never_approve(self):
        feed = oc.ToolFeed({"session"}, {"old"})
        with patch.object(mcp, "activity") as notify, patch.object(oc, "reply_permission") as approve:
            feed.part(self.part(sid="unrelated"))
            feed.part(self.part("pending"))
            feed.part(self.part("completed", messageID="old"))
            self.assertFalse(notify.called)
            feed.part(self.part())
            feed.part(self.part())
            feed.part(self.part("completed"))
            self.assertEqual([c.args[2] for c in notify.call_args_list], ["running", "ok"])
            self.assertEqual(notify.call_args.args[0], "session:call")
            approve.assert_not_called()

    def test_polling_and_streaming_share_the_same_tool_identity(self):
        feed = oc.ToolFeed({"session"}, set())
        with patch.object(mcp, "activity") as notify:
            feed.part(self.part())
            feed.messages([{"info": {"id": "new", "role": "assistant"},
                            "parts": [self.part("error")]}], "session")
            self.assertEqual([c.args[2] for c in notify.call_args_list], ["running", "error"])
            self.assertEqual(len(feed.sent), 1)

    def test_sse_collects_latest_tool_states_and_leaves_text_out(self):
        events = object.__new__(oc.Events)
        events.parts, events.parts_lock = {}, threading.Lock()
        events.flag = threading.Event()
        events.alive = True
        updates = [{"type": "message.part.updated", "properties": {"part": self.part()}},
                   {"type": "message.part.updated", "properties": {"part": self.part("completed")}},
                   {"type": "message.part.updated", "properties": {"part": {"type": "text", "text": "private reasoning"}}}]
        events.resp = io.BytesIO("\n".join("data: " + json.dumps(e) for e in updates).encode())
        events._read()
        parts = events.drain_tools()
        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0]["state"]["status"], "completed")
        self.assertFalse(events.drain_tools())
        self.assertTrue(events.flag.is_set())


if __name__ == "__main__":
    unittest.main()
