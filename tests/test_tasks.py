"""Executor acceptance tests: fake inference and bridges, no creative apps."""
import argparse
import io
import json
import os
import queue
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import studio_agent as eng
import studio_tasks as tasks


def call(name, args=None, ident="c1"):
    return {"id": ident, "type": "function", "function": {
        "name": name, "arguments": json.dumps(args or {})}}


def answer(*calls, text=""):
    msg = {"role": "assistant", "content": text}
    if calls:
        msg["tool_calls"] = list(calls)
    return msg


def spec(name, schema=None, **kw):
    return {"name": name, "description": name,
            "inputSchema": schema or {"type": "object", "additionalProperties": False}, **kw}


class FakeLLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def stream(self, messages, tools, on_text):
        self.requests.append(messages)
        msg = next(self.responses)
        if msg.get("content"):
            on_text(msg["content"])
        return msg

    def chat(self, messages, tools):
        msg = self.stream(messages, tools, lambda _: None)
        return {"choices": [{"message": msg, "finish_reason": "tool_calls" if msg.get("tool_calls") else "stop"}]}


class TestExecutor(unittest.TestCase):
    def setup_run(self, responses, specs=None, bridge=None, **kw):
        specs = specs or [spec("create_comp"), spec("get_comp")]
        self.messages = [{"role": "system", "content": "rules"},
                         {"role": "user", "content": "Create a title"}]
        self.llm = FakeLLM(responses)
        self.bridge = bridge or Mock()
        if bridge is None:
            self.bridge.call_tool.return_value = {"content": [{"type": "text", "text": '{"id": 12}'}]}
        self.events = []
        self.executor = tasks.Executor(self.llm, self.bridge, eng.to_openai_tools(specs),
            schemas=specs, emit=lambda k, p: self.events.append((k, p)), **kw)
        return self.executor

    def test_edit_requires_readback_before_finishing(self):
        ex = self.setup_run([answer(call("create_comp")), answer(text="Done"),
                             answer(call("get_comp")), answer(text="Verified title")])
        self.assertEqual(ex.run(self.messages), "Verified title")
        self.assertEqual(self.bridge.call_tool.call_count, 2)
        self.assertIn("Before finishing", self.messages[5]["content"])

    def test_questions_need_no_tools_or_extra_verification(self):
        ex = self.setup_run([answer(text="24 fps means 24 frames per second.")])
        self.assertIn("24 fps", ex.run(self.messages))
        self.bridge.call_tool.assert_not_called()

    def test_disabled_tool_never_reaches_bridge(self):
        ex = self.setup_run([answer(call("run_jsx")), answer(text="Unavailable")])
        ex.run(self.messages)
        self.bridge.call_tool.assert_not_called()
        self.assertIn("not enabled", self.messages[3]["content"])

    def test_original_tuple_constraints_validate_before_dispatch(self):
        schema = {"type": "object", "required": ["p"], "properties": {"p": {
            "type": "array", "prefixItems": [{"type": "number", "maximum": 1},
                                                   {"type": "number", "minimum": 10}], "items": False}}}
        ex = self.setup_run([answer(call("set_transform", {"p": [0, 2]})), answer(text="Cannot do that")],
                            specs=[spec("set_transform", schema)])
        ex.run(self.messages)
        self.bridge.call_tool.assert_not_called()
        self.assertIn("p[1] must be >= 10, not 2", self.messages[3]["content"])

    def test_resolve_quit_blocked_even_if_exposed(self):
        ex = self.setup_run([answer(call("resolve_control", {"action": "quit"})), answer(text="Cannot quit")],
            specs=[spec("resolve_control", {"type": "object", "properties": {"action": {"type": "string"}}})])
        ex.run(self.messages)
        self.bridge.call_tool.assert_not_called()

    def test_documented_compound_actions_are_checked(self):
        s = spec("timeline", {"type": "object", "properties": {"action": {"type": "string"}}})
        s["description"] = "Timeline.\nActions:\n  get_name() -> {name}\n  set_name(name) -> {success}"
        ex = self.setup_run([answer(call("timeline", {"action": "invented"})), answer(text="Unavailable")], specs=[s])
        ex.run(self.messages)
        self.bridge.call_tool.assert_not_called()

    def test_stop_during_batch_skips_remaining_calls_and_repairs_history(self):
        cancel = threading.Event()
        def execute(*args):
            cancel.set()
            return {"content": [{"type": "text", "text": "ok"}]}
        bridge = Mock()
        bridge.call_tool.side_effect = execute
        ex = self.setup_run([answer(call("create_comp"), call("create_comp", ident="c2"))],
                            bridge=bridge, cancel=cancel)
        self.assertIn("Stopped", ex.run(self.messages))
        self.assertEqual(bridge.call_tool.call_count, 1)
        self.assertEqual([m["tool_call_id"] for m in self.messages if m["role"] == "tool"], ["c1", "c2"])
        self.assertIn("not executed", self.messages[-1]["content"])

    def test_repeated_error_is_not_dispatched_twice(self):
        bridge = Mock()
        bridge.call_tool.return_value = {"isError": True, "content": [{"type": "text", "text": "Bad target"}]}
        ex = self.setup_run([answer(call("create_comp")) for _ in range(3)], bridge=bridge)
        self.assertIn("repeated tool errors", ex.run(self.messages))
        self.assertEqual(bridge.call_tool.call_count, 1)

    def test_timeout_cannot_duplicate_write_even_in_later_turn(self):
        bridge = Mock()
        bridge.call_tool.side_effect = [TimeoutError("no response"), {"content": [{"type": "text", "text": "found"}]}]
        ex = self.setup_run([answer(call("create_comp")), answer(call("get_comp")),
                            answer(call("create_comp")), answer(text="Inspect manually")], bridge=bridge)
        ex.run(self.messages)
        self.assertEqual(bridge.call_tool.call_count, 2)
        self.assertEqual(ex.record.journal[0]["status"], "unknown")
        self.assertIn("unknown outcome", self.messages[-2]["content"])

    def test_incomplete_batch_envelope_prevents_all_dispatch(self):
        ex = self.setup_run([answer(call("create_comp"), call("create_comp"))])
        self.assertIn("invalid tool-call envelope", ex.run(self.messages))
        self.bridge.call_tool.assert_not_called()

    def test_disk_failure_before_edit_stops_execution(self):
        saves = Mock(side_effect=[None, OSError("disk full")])
        ex = self.setup_run([answer(call("create_comp"))], checkpoint=saves)
        with self.assertRaises(tasks.CheckpointError):
            ex.run(self.messages)
        self.bridge.call_tool.assert_not_called()
        self.assertEqual(self.messages[-1]["role"], "tool")
        self.assertEqual(self.messages[-1]["tool_call_id"], "c1")

    def test_disk_failure_after_edit_does_not_dispatch_next_call(self):
        saves = Mock(side_effect=[None, None, OSError("disk full"), None])
        ex = self.setup_run([answer(call("create_comp"), call("create_comp", ident="c2"))], checkpoint=saves)
        with self.assertRaises(tasks.CheckpointError):
            ex.run(self.messages)
        self.assertEqual(self.bridge.call_tool.call_count, 1)
        self.assertEqual([m["tool_call_id"] for m in self.messages if m["role"] == "tool"], ["c1", "c2"])

    def test_restored_task_must_read_before_writing(self):
        record = tasks.TaskRecord()
        record.status = "restored — inspect first"
        ex = self.setup_run([answer(call("create_comp")), answer(call("get_comp")),
                            answer(call("create_comp")), answer(call("get_comp")), answer(text="Done")], record=record)
        ex.run(self.messages)
        self.assertEqual(self.bridge.call_tool.call_count, 3)
        self.assertEqual(self.bridge.call_tool.call_args_list[0].args[0], "get_comp")

    def test_health_check_does_not_count_as_edit_verification(self):
        ex = self.setup_run([answer(call("create_comp")), answer(call("check_setup"))] +
                            [answer(text="Done") for _ in range(3)],
                            specs=[spec("create_comp"), spec("check_setup")])
        self.assertIn("unverified", ex.run(self.messages))

    def test_intent_saved_before_write(self):
        seen = []
        ex = self.setup_run([answer(call("create_comp")), answer(call("get_comp")), answer(text="Done")])
        ex.checkpoint = lambda: seen.append(json.loads(json.dumps(ex.record.journal)))
        ex.run(self.messages)
        self.assertEqual(seen[1][0]["status"], "running")
        self.assertEqual(seen[2][0]["status"], "ok")

    def test_json_encoded_error_is_treated_as_failure(self):
        bridge = Mock()
        bridge.call_tool.return_value = {"content": [{"type": "text", "text": '{"success": false, "error": "offline"}'}]}
        ex = self.setup_run([answer(call("create_comp")), answer(text="Offline")], bridge=bridge)
        ex.run(self.messages)
        self.assertEqual(ex.record.journal[0]["status"], "error")

    def test_preview_is_emitted_and_optional_critique_reaches_model(self):
        bridge = Mock()
        image = {"type": "image", "mimeType": "image/png", "data": "fake"}
        bridge.call_tool.return_value = {"content": [image]}
        ex = self.setup_run([answer(call("screenshot_frame")), answer(text="Needs adjustment")],
                            specs=[spec("screenshot_frame")], bridge=bridge,
                            vision=lambda image, brief: "Text is clipped at right edge")
        ex.run(self.messages)
        self.assertIn(("preview", image), self.events)
        self.assertIn("clipped", self.messages[3]["content"])

    def test_every_call_is_told_whole_with_its_outcome_under_its_name(self):
        """The transcript folds each call to its name, so the event carries
        the arguments entire - the script a run_jsx took, not 160 characters
        of it - and the result comes back under the same name with a status."""
        code = "\n".join("var line%d = app.project.item(%d);" % (i, i) for i in range(40))
        schema = {"type": "object", "required": ["code"], "additionalProperties": False,
                  "properties": {"code": {"type": "string"}}}
        ex = self.setup_run([answer(call("run_jsx", {"code": code})),
                             answer(call("no_such_tool")), answer(text="Done")],
                            specs=[spec("run_jsx", schema, annotations={"readOnlyHint": True})])
        ex.run(self.messages)
        calls = [p for k, p in self.events if k == "tool"]
        results = [p for k, p in self.events if k == "tool_result"]
        self.assertEqual([c["name"] for c in calls], ["run_jsx", "no_such_tool"])
        self.assertEqual(calls[0]["arguments"], {"code": code})
        self.assertIsNone(calls[0]["via"])
        self.assertEqual([(r["name"], r["status"]) for r in results],
                         [("run_jsx", "ok"), ("no_such_tool", "error")])
        self.assertEqual(results[0]["text"], '{"id": 12}')
        self.assertTrue(results[1]["text"].startswith("TOOL ERROR: tool is not enabled"))

    def test_a_call_the_executor_will_not_run_is_still_shown_as_one(self):
        cancel = threading.Event()
        bridge = Mock()
        def execute(*args):
            cancel.set()
            return {"content": [{"type": "text", "text": "ok"}]}
        bridge.call_tool.side_effect = execute
        ex = self.setup_run([answer(call("create_comp"), call("get_comp", ident="c2"))],
                            bridge=bridge, cancel=cancel)
        ex.run(self.messages)
        pairs = [(k, p["name"], p.get("status")) for k, p in self.events
                 if k in ("tool", "tool_result")]
        self.assertEqual(pairs, [("tool", "create_comp", None), ("tool_result", "create_comp", "ok"),
                                 ("tool", "get_comp", None), ("tool_result", "get_comp", "skipped")])

    def test_unverified_work_is_not_silently_accepted(self):
        ex = self.setup_run([answer(call("create_comp"))] + [answer(text="Done") for _ in range(3)])
        self.assertIn("remain unverified", ex.run(self.messages))


class TestPromisedWork(unittest.TestCase):
    """A reply that announces the call instead of making it does not end the
    run looking finished: one reminder, then the emptiness is named."""

    setup_run = TestExecutor.setup_run
    PROMISE = ("I've selected the first image. Now I'll generate an image of a man in "
               "lederhosen using this background. Once it completes, I'll provide the path.")

    def test_a_promise_is_nudged_once_and_the_call_then_runs(self):
        ex = self.setup_run([answer(text=self.PROMISE), answer(call("get_comp")),
                             answer(text="Here it is: C:\out\a.png")])
        self.assertEqual(ex.run(self.messages), "Here it is: C:\out\a.png")
        self.assertEqual(self.messages[3]["role"], "user")
        self.assertIn("called no tool", self.messages[3]["content"])
        self.assertEqual(self.bridge.call_tool.call_count, 1)
        self.assertIn(("sys", "The model described a step without calling a tool; asked once to make the call."),
                      self.events)

    def test_a_second_promise_ends_the_run_with_nothing_done_named(self):
        ex = self.setup_run([answer(text=self.PROMISE), answer(text="Now I will generate it.")])
        out = ex.run(self.messages)
        self.assertTrue(out.startswith("Nothing was done"), out)
        self.assertEqual(ex.record.status, out)
        self.bridge.call_tool.assert_not_called()
        # Two replies, one reminder, no third request.
        self.assertEqual(len(self.llm.requests), 2)

    def test_a_question_and_a_plain_answer_are_not_promises(self):
        for text in ("Which background should I use - I'll take the first unless you say?",
                     "The default model is Z-Image Turbo at 8 steps."):
            with self.subTest(text=text):
                ex = self.setup_run([answer(text=text)])
                self.assertEqual(ex.run(self.messages), text)
                self.assertEqual(len(self.llm.requests), 1)

    def test_a_sign_off_after_real_work_is_not_nudged(self):
        ex = self.setup_run([answer(call("get_comp")),
                             answer(text="Done. I'll be here if you want another seed.")])
        self.assertIn("another seed", ex.run(self.messages))
        self.assertEqual(len(self.llm.requests), 2)

    def test_with_no_tools_at_all_a_promise_is_just_a_reply(self):
        # setup_run substitutes its default tools for an empty list, so build
        # the toolless tab by hand: there is nothing it could have called.
        llm = FakeLLM([answer(text=self.PROMISE)])
        ex = tasks.Executor(llm, Mock(), [], schemas=[])
        messages = [{"role": "system", "content": "rules"}, {"role": "user", "content": "hi"}]
        self.assertEqual(ex.run(messages), self.PROMISE)
        self.assertEqual(len(llm.requests), 1)

    def test_the_detector(self):
        for text, hit in [("Next, I'll generate the image.", True),
                          ("Let me check the queue first.", True),
                          ("I will now upload the background.", True),
                          ("24 fps means 24 frames per second.", False),
                          ("Shall I go ahead and generate it?", False),
                          ("", False), (None, False)]:
            with self.subTest(text=text):
                self.assertEqual(tasks.announces_work(text), hit)


class TestVerifyingTheWrite(unittest.TestCase):
    """The read-back reminder names the exact call; with a vision model the
    executor looks at the work itself before accepting "done"."""

    ANY = {"type": "object", "additionalProperties": True}
    AE = [("get_layer_full", ["compId", "layerId"]), ("get_comp", ["compId"])]
    SPECS = [spec("set_transform", ANY), spec("get_layer_full", ANY), spec("get_comp", ANY),
             spec("screenshot_frame", ANY), spec("create_comp", ANY)]
    setup_run = TestExecutor.setup_run
    # messages: system, brief, the write, its result, "Done", then what the
    # executor put back - index 5.

    def test_the_reminder_names_the_read_and_the_ids_from_the_write(self):
        ex = self.setup_run([answer(call("set_transform", {"compId": 12, "layerId": 40, "opacity": 50})),
                             answer(text="Done"), answer(call("get_layer_full", ident="c2")),
                             answer(text="Verified")],
                            specs=self.SPECS, readback=self.AE)
        self.assertEqual(ex.run(self.messages), "Verified")
        hint = self.messages[5]["content"]
        self.assertIn("verify the edit you made with set_transform", hint)
        self.assertIn('call get_layer_full with {"compId": 12, "layerId": 40}', hint)

    def test_a_write_without_the_ids_gets_the_less_specific_read(self):
        # create_comp carries no compId; get_comp needs one; nothing in the
        # journal has one yet - so the reminder is the generic sentence.
        ex = self.setup_run([answer(call("create_comp", {"name": "Title"}))] +
                            [answer(text="Done") for _ in range(3)],
                            specs=self.SPECS, readback=self.AE)
        ex.run(self.messages)
        hint = self.messages[5]["content"]
        self.assertIn("inspect the target changed by create_comp", hint)
        self.assertNotIn("get_comp", hint)

    def test_a_read_that_needs_no_ids_fits_any_write(self):
        ex = self.setup_run([answer(call("ps_set_layer", {"layer_id": 7, "opacity": 40})),
                             answer(text="Done"), answer(text="Done"), answer(text="Done")],
                            specs=[spec("ps_set_layer", self.ANY), spec("ps_get_document", self.ANY)],
                            readback=[("ps_get_layer", ["layer_id"]), ("ps_get_document", [])])
        ex.run(self.messages)
        # ps_get_layer is not in this tab's tools, so the document read is named.
        self.assertIn("call ps_get_document with {}", self.messages[5]["content"])

    def test_with_a_vision_model_the_executor_looks_before_accepting_done(self):
        bridge = Mock()
        image = {"type": "image", "mimeType": "image/png", "data": "fake"}
        bridge.call_tool.side_effect = [{"content": [{"type": "text", "text": "ok"}]},
                                        {"content": [image]}]
        ex = self.setup_run([answer(call("set_transform", {"compId": 12, "layerId": 40})),
                             answer(text="Done"), answer(text="Done, and the review agrees")],
                            specs=self.SPECS, bridge=bridge, readback=self.AE,
                            review=("screenshot_frame", ["compId"]),
                            vision=lambda image, brief: "A red circle, centred, as asked")
        self.assertEqual(ex.run(self.messages), "Done, and the review agrees")
        self.assertEqual(bridge.call_tool.call_args_list[1].args, ("screenshot_frame", {"compId": 12}))
        shown = self.messages[5]["content"]
        self.assertIn("Automatic review of the work after set_transform", shown)
        self.assertIn("A red circle, centred", shown)
        self.assertIn(("preview", image), self.events)
        self.assertTrue(ex.record.journal[1]["auto"])
        self.assertNotIn("auto", ex.record.journal[0])

    def test_without_a_vision_model_nobody_looks_so_the_read_is_asked_for(self):
        bridge = Mock()
        bridge.call_tool.return_value = {"content": [{"type": "text", "text": "ok"}]}
        ex = self.setup_run([answer(call("set_transform", {"compId": 12, "layerId": 40}))] +
                            [answer(text="Done") for _ in range(3)],
                            specs=self.SPECS, bridge=bridge, readback=self.AE,
                            review=("screenshot_frame", ["compId"]))
        self.assertIn("remain unverified", ex.run(self.messages))
        self.assertEqual(bridge.call_tool.call_count, 1)
        self.assertIn("call get_layer_full", self.messages[5]["content"])

    def test_a_screenshot_that_fails_falls_back_to_the_reminder(self):
        bridge = Mock()
        bridge.call_tool.side_effect = [{"content": [{"type": "text", "text": "ok"}]},
                                        {"isError": True, "content": [{"type": "text", "text": "no comp"}]}]
        ex = self.setup_run([answer(call("set_transform", {"compId": 12, "layerId": 40}))] +
                            [answer(text="Done") for _ in range(3)],
                            specs=self.SPECS, bridge=bridge, readback=self.AE,
                            review=("screenshot_frame", ["compId"]), vision=lambda i, b: "x")
        self.assertIn("remain unverified", ex.run(self.messages))
        self.assertIn("call get_layer_full", self.messages[5]["content"])

    def test_a_review_tool_the_tab_does_not_offer_is_never_called(self):
        bridge = Mock()
        bridge.call_tool.return_value = {"content": [{"type": "text", "text": "ok"}]}
        ex = self.setup_run([answer(call("set_transform", {"compId": 12}))] +
                            [answer(text="Done") for _ in range(3)],
                            specs=[spec("set_transform", self.ANY)], bridge=bridge,
                            review=("screenshot_frame", ["compId"]), vision=lambda i, b: "x")
        ex.run(self.messages)
        self.assertEqual(bridge.call_tool.call_count, 1)
        self.assertIsNone(ex.review)

    def test_cli_uses_same_executor(self):
        llm = FakeLLM([answer(call("unknown")), answer(text="Not available")])
        bridge = Mock()
        self.assertEqual(eng.run_agent(llm, bridge, [], "task", "rules", quiet=True), "Not available")
        bridge.call_tool.assert_not_called()


class TestPlainChat(unittest.TestCase):
    """A tab with no bridge runs through the same executor, with nothing in it."""

    def test_a_tab_with_no_tools_is_offered_none_at_all(self):
        """Not even the internal ones: there is nothing to journal when
        nothing can be called."""
        self.assertEqual(tasks.inference_tools([]), [])
        names = [t["function"]["name"]
                 for t in tasks.inference_tools(eng.to_openai_tools([spec("get_comp")]))]
        self.assertIn("studio_task_update", names)

    def test_a_chat_turn_needs_no_bridge_and_carries_no_task_block(self):
        llm = FakeLLM([answer(text="240 frames at 23.976 is 10.01 seconds.")])
        messages = [{"role": "system", "content": eng.CHAT.chat_prompt()},
                    {"role": "user", "content": "how long is 240 frames"}]
        executor = tasks.Executor(llm, None, [])       # no bridge at all
        self.assertIn("10.01", executor.run(messages))
        sent = llm.requests[0]
        self.assertEqual(len(sent), 2)                 # system, then the question
        self.assertNotIn("Saved task context", json.dumps(sent))

    def test_a_chat_tab_still_cannot_call_a_tool(self):
        llm = FakeLLM([answer(call("create_comp")), answer(text="I cannot do that here.")])
        messages = [{"role": "system", "content": eng.CHAT.chat_prompt()},
                    {"role": "user", "content": "make a comp"}]
        tasks.Executor(llm, None, []).run(messages)
        result = [m for m in messages if m.get("role") == "tool"]
        self.assertIn("not enabled", result[0]["content"])

    def test_the_cli_chat_path_runs_without_a_bridge(self):
        llm = FakeLLM([answer(text="Finish at 23.976.")])
        args = argparse.Namespace(task=["what frame rate"], max_steps=5, quiet=True)
        with patch("sys.stdout", new=io.StringIO()) as out:
            self.assertEqual(eng.converse(llm, None, [], eng.CHAT, args), 0)
        self.assertIn("Finish at 23.976.", out.getvalue())
        self.assertEqual(llm.requests[0][0]["content"], eng.CHAT.cli_prompt())


class TestMemory(unittest.TestCase):
    def test_roundtrip_reconciles_crashed_batch_and_preserves_brief(self):
        record = tasks.TaskRecord()
        record.briefs = ["Use the exact words: Safety First"]
        record.journal = [{"status": "running", "name": "create_comp", "signature": "s"}]
        messages = [{"role": "system", "content": "old"}, answer(call("create_comp"), call("create_comp", ident="c2")),
                    {"role": "tool", "tool_call_id": "c1", "content": "ok"}]
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "task.json")
            record.save(path, messages)
            restored, history = tasks.TaskRecord.restore(path, "new")
        self.assertEqual(restored.briefs, record.briefs)
        self.assertEqual(restored.journal[0]["status"], "unknown")
        self.assertEqual(history[0]["content"], "new")
        self.assertEqual(history[-1]["tool_call_id"], "c2")

    def test_context_keeps_brief_and_complete_tool_exchanges(self):
        record = tasks.TaskRecord()
        record.briefs = ["Exact wording"]
        record.objects = {"title": "12"}
        messages = [{"role": "system", "content": "rules"}]
        for i in range(20):
            messages.extend([answer(call("get_comp", ident=str(i))),
                             {"role": "tool", "tool_call_id": str(i), "content": "x" * 100}])
        result = tasks.context_messages(messages, record, [], 1300)
        self.assertIn("Exact wording", result[-1]["content"])
        self.assertLess(len(result), len(messages))
        self.assertEqual(result[1]["role"], "assistant")
        self.assertEqual(result[-2]["tool_call_id"], "19")
        self.assertEqual(len(messages), 41)

    def test_the_task_record_is_the_last_message_so_the_history_stays_cached(self):
        """The record changes every turn; the host caches by prefix. Everything
        before the record in turn N must be, message for message, a prefix of
        turn N+1 - which it cannot be with the record at position 1."""
        record = tasks.TaskRecord()
        record.briefs = ["Make a title"]
        messages = [{"role": "system", "content": "rules"},
                    {"role": "user", "content": "Make a title"}]
        first = tasks.context_messages(messages, record, [])
        self.assertEqual(first[-1]["role"], "user")
        self.assertIn("Saved task context", first[-1]["content"])
        # A turn happens: the record changes and the history grows.
        record.status = "working"
        record.objects["title"] = "12"
        messages.extend([answer(call("get_comp", ident="1")),
                         {"role": "tool", "tool_call_id": "1", "content": "ok"}])
        second = tasks.context_messages(messages, record, [])
        self.assertEqual(second[:len(first) - 1], first[:-1])
        self.assertIn('"title": "12"', second[-1]["content"])
        self.assertNotIn("Saved task context", json.dumps(second[:-1]))

    def test_fixed_context_is_never_silently_clipped(self):
        record = tasks.TaskRecord()
        record.briefs = ["x" * 2000]
        with self.assertRaisesRegex(ValueError, "context budget"):
            tasks.context_messages([{"role": "system", "content": "rules"}], record, [], 1000)

    def test_record_updates_are_local_only(self):
        record = tasks.TaskRecord()
        bridge = Mock()
        llm = FakeLLM([answer(call("studio_task_update", {"objects": {"title": "42"},
                          "checks": [{"requirement": "5 seconds", "evidence": "duration 5"}]})), answer(text="Recorded")])
        ex = tasks.Executor(llm, bridge, [], record=record)
        ex.run([{"role": "system", "content": "rules"}])
        bridge.call_tool.assert_not_called()
        self.assertEqual(record.objects["title"], "42")


class TestTransport(unittest.TestCase):
    def test_truncated_stream_cannot_return_executable_calls(self):
        chunk = {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1",
                 "function": {"name": "create_comp", "arguments": "{}"}}]}}]}
        stream = io.BytesIO(("data: " + json.dumps(chunk) + "\n").encode())
        with patch.object(eng.urllib.request, "urlopen", return_value=stream):
            with self.assertRaisesRegex(RuntimeError, "Incomplete"):
                eng.LLM("http://fake", "model").stream([])

    def test_complete_stream_reassembles_call(self):
        chunks = [{"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1",
                  "function": {"name": "get_comp", "arguments": "{}"}}]}}]},
                  {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}]
        stream = io.BytesIO(("".join("data: " + json.dumps(c) + "\n" for c in chunks) + "data: [DONE]\n").encode())
        with patch.object(eng.urllib.request, "urlopen", return_value=stream):
            result = eng.LLM("http://fake", "model").stream([])
        self.assertEqual(result["tool_calls"][0]["function"]["name"], "get_comp")

    def test_dead_bridge_fails_without_timeout(self):
        client = eng.MCPClient.__new__(eng.MCPClient)
        client._lock, client._request_lock = threading.Lock(), threading.Lock()
        client._id = 0
        client._inbox = queue.Queue()
        client._inbox.put({"_closed": True})
        client._send = Mock()
        with self.assertRaises(EOFError):
            client.request("tools/call", timeout=100)


class TestSchemaValidation(unittest.TestCase):
    def test_descriptions_are_not_truncated_past_compatibility_limit(self):
        original = spec("timeline")
        original["description"] = "x" * 4500 + "\ncritical_last_action"
        converted = eng.to_openai_tools([original])
        self.assertEqual(converted[0]["function"]["description"], original["description"])

    def test_types_bounds_refs_and_extra_fields(self):
        schema = {"type": "object", "additionalProperties": False, "required": ["n"],
                  "$defs": {"number": {"type": "integer", "minimum": 1}},
                  "properties": {"n": {"$ref": "#/$defs/number"}}}
        tasks.validate({"n": 2}, schema)
        for value in ({}, {"n": True}, {"n": 0}, {"n": 2, "guess": 1}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                tasks.validate(value, schema)


class TestSessionRouting(unittest.TestCase):
    def test_old_generation_is_dropped_after_reopening_same_app(self):
        import studio_chat as chat
        old, new = chat.Session(eng.APPS[0]), chat.Session(eng.APPS[0])
        fake = Mock()
        fake.sessions = {new.id: new}
        chat.Chat._handle(fake, "token", old.event_id, "old result")
        fake._write.assert_not_called()
        self.assertNotEqual(old.event_id, new.event_id)
