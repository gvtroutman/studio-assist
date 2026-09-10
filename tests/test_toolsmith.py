"""Tools the model makes for itself, through the real executor.

A made tool is a shorthand for calls the tab already had. These tests exist to
prove it is only ever that: it cannot name a tool the tab was not given, cannot
skip validation against the bridge's own schema, and cannot keep its steps out
of the journal.
"""
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import Mock

import studio_agent as eng
import studio_tasks as tasks
import studio_toolsmith as toolsmith


SPECS = [
    {"name": "set_text", "description": "Set a text layer's contents.",
     "inputSchema": {"type": "object", "required": ["compId", "layerId", "text"],
         "additionalProperties": False,
         "properties": {"compId": {"type": "integer"}, "layerId": {"type": "integer"},
                        "text": {"type": "string"},
                        # a 2020-12 tuple, as the AE bridge really writes them
                        "colour": {"type": "array", "prefixItems": [
                            {"type": "number", "maximum": 1}], "items": False}}}},
    {"name": "get_layer_full", "description": "Read a layer back.",
     "inputSchema": {"type": "object", "required": ["layerId"],
         "additionalProperties": False,
         "properties": {"layerId": {"type": "integer"}}}},
    {"name": "resolve_control", "description": """Control the app.

Actions:
  get_version() -> object
  quit() -> object
""",
     "inputSchema": {"type": "object", "required": ["action"],
         "additionalProperties": False,
         "properties": {"action": {"type": "string", "enum": ["get_version", "quit"]}}}},
]


def call(name, args=None):
    return {"function": {"name": name, "arguments": json.dumps(args or {})}}


def definition(**over):
    made = {"name": "retitle_card",
            "description": "Set the title of a card and read the layer back.",
            "inputs": [{"name": "layer", "type": "integer", "description": "layer id"},
                       {"name": "words", "type": "string", "description": "the title"}],
            "steps": [{"tool": "set_text",
                       "arguments": {"compId": 1, "layerId": "{layer}", "text": "{words}"}},
                      {"tool": "get_layer_full", "arguments": {"layerId": "{layer}"}}]}
    made.update(over)
    return made


class ToolsmithTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.library = toolsmith.Library("after-effects", os.path.join(self.dir, "ae"))
        self.bridge = Mock()
        self.bridge.call_tool.return_value = {"content": [{"type": "text", "text": "done"}]}

    def executor(self, specs=SPECS, **kwargs):
        kwargs.setdefault("library", self.library)
        return tasks.Executor(None, self.bridge, eng.to_openai_tools(specs),
                              schemas=specs, **kwargs)

    def make(self, ex, **over):
        return ex._call(call("studio_tool_create", definition(**over)))[0]

    # ------------------------------------------------------------------ making
    def test_a_made_tool_is_offered_after_the_fixed_contract(self):
        ex = self.executor()
        before = [t["function"]["name"] for t in ex.tools]
        self.assertEqual(before[-2:], ["studio_task_update", "studio_tool_create"])
        text = self.make(ex)
        self.assertIn("retitle_card", text)
        names = [t["function"]["name"] for t in ex.tools]
        # Appended last, so making one re-prefills the tail and not the tool set.
        self.assertEqual(names, before + ["retitle_card"])
        self.assertEqual(names[:len(before)], before)

    def test_making_a_tool_touches_neither_project_nor_journal(self):
        ex = self.executor()
        text, wrote, read = ex._call(call("studio_tool_create", definition()))
        self.assertFalse(wrote)
        self.assertFalse(read)
        self.assertEqual(ex.record.journal, [])
        self.assertFalse(self.bridge.call_tool.called)
        self.assertIn("changed nothing in the project", text)

    def test_a_step_may_only_name_a_tool_this_tab_has(self):
        ex = self.executor()
        for tool in ("delete_comp", "studio_task_update", "inspect_layer", "retitle_card"):
            with self.assertRaises(ValueError) as caught:
                self.make(ex, steps=[{"tool": tool, "arguments": {}}])
            self.assertIn("not a tool you have", str(caught.exception))

    def test_a_made_tool_cannot_take_an_existing_name(self):
        ex = self.executor()
        for name in ("set_text", "studio_task_update", "studio_tool_create",
                     "inspect_layer", "verify_comp", "studio_workflow_capabilities"):
            with self.assertRaises(ValueError):
                self.make(ex, name=name)

    def test_overwriting_a_made_tool_is_deliberate(self):
        ex = self.executor()
        self.make(ex)
        with self.assertRaises(ValueError) as caught:
            self.make(ex)
        self.assertIn("replace", str(caught.exception))
        self.make(ex, replace=True, description="A second version of the same idea.")
        self.assertEqual(len(self.library.made), 1)

    def test_a_template_that_does_not_fit_its_tool_is_refused(self):
        ex = self.executor()
        with self.assertRaises(ValueError) as caught:          # required key missing
            self.make(ex, inputs=[],
                      steps=[{"tool": "set_text", "arguments": {"compId": 1}}])
        self.assertIn("does not fit set_text", str(caught.exception))
        with self.assertRaises(ValueError):                    # key the tool has never heard of
            self.make(ex, inputs=[], steps=[{"tool": "set_text", "arguments": {
                "compId": 1, "layerId": 2, "text": "hi", "colr": "red"}}])
        with self.assertRaises(ValueError):                    # input typed wrong for the slot
            self.make(ex, inputs=[{"name": "layer", "type": "string", "description": "id"},
                                  {"name": "words", "type": "string", "description": "t"}])

    def test_a_value_a_sample_cannot_satisfy_is_left_for_run_time(self):
        # The action is a placeholder, so the creation check sees "sample" against
        # an enum. That says nothing about the template; the real value is
        # validated on every call, like any other tool's.
        ex = self.executor()
        self.make(ex, name="app_version", inputs=[
                      {"name": "what", "type": "string", "description": "the action"}],
                  steps=[{"tool": "resolve_control", "arguments": {"action": "{what}"}}])
        self.assertIn("app_version", self.library.made)
        text, _, _ = ex._call(call("app_version", {"what": "explode"}))
        self.assertTrue(text.startswith("TOOL ERROR"))
        self.assertIn("must be one of", text)
        self.assertFalse(self.bridge.call_tool.called)

    def test_every_input_must_be_used_and_every_placeholder_declared(self):
        ex = self.executor()
        with self.assertRaises(ValueError) as caught:
            self.make(ex, inputs=definition()["inputs"] + [
                {"name": "spare", "type": "string", "description": "unused"}])
        self.assertIn("{spare}", str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            self.make(ex, steps=[{"tool": "get_layer_full",
                                  "arguments": {"layerId": "{layer}", "note": "{wrods}"}}])
        self.assertIn("{wrods}", str(caught.exception))

    def test_braces_that_are_not_placeholders_are_left_alone(self):
        # AE expressions are JavaScript. Their braces must not read as inputs.
        source = "if (x) {\n  value + 1;\n}"
        self.assertEqual(toolsmith.names_used(source), set())
        self.assertEqual(toolsmith.render(source, {"x": 1}), source)

    # ----------------------------------------------------------------- running
    def test_running_a_made_tool_calls_its_steps_in_order_with_real_types(self):
        ex = self.executor()
        self.make(ex)
        text, wrote, read = ex._call(call("retitle_card", {"layer": 12, "words": "Hello"}))
        self.assertEqual([c.args for c in self.bridge.call_tool.call_args_list],
                         [("set_text", {"compId": 1, "layerId": 12, "text": "Hello"}),
                          ("get_layer_full", {"layerId": 12})])
        report = json.loads(text)
        self.assertEqual(report["steps_run"], 2)
        self.assertEqual([r["tool"] for r in report["results"]],
                         ["set_text", "get_layer_full"])
        # It wrote and then read the same layer back, so nothing is left unverified.
        self.assertFalse(wrote)
        self.assertTrue(read)

    def test_a_made_tool_that_only_edits_still_owes_a_read(self):
        ex = self.executor()
        self.make(ex, name="just_write", steps=definition()["steps"][:1],
                  inputs=definition()["inputs"])
        _, wrote, read = ex._call(call("just_write", {"layer": 3, "words": "x"}))
        self.assertTrue(wrote)
        self.assertFalse(read)

    def test_every_step_is_journalled_and_says_what_it_ran_under(self):
        ex = self.executor()
        self.make(ex)
        ex._call(call("retitle_card", {"layer": 12, "words": "Hello"}))
        self.assertEqual([e["name"] for e in ex.record.journal],
                         ["set_text", "get_layer_full"])
        self.assertEqual({e.get("via") for e in ex.record.journal}, {"retitle_card"})
        # and a bare call afterwards is not attributed to it
        ex._call(call("get_layer_full", {"layerId": 12}))
        self.assertIsNone(ex.record.journal[-1].get("via"))

    def test_arguments_are_validated_against_the_original_bridge_schema(self):
        ex = self.executor()
        self.make(ex, name="tint_card", inputs=[
                      {"name": "shade", "type": "number", "description": "0..1"}],
                  steps=[{"tool": "set_text", "arguments": {
                      "compId": 1, "layerId": 2, "text": "x", "colour": ["{shade}"]}}])
        # The sanitized copy inference sees has lost the tuple's maximum; the
        # original has not, and the original is what a call is checked against.
        text, _, _ = ex._call(call("tint_card", {"shade": 4}))
        self.assertTrue(text.startswith("TOOL ERROR"))
        self.assertIn("colour[0]", text)
        self.assertFalse(self.bridge.call_tool.called)
        ex._call(call("tint_card", {"shade": 0.5}))
        self.assertTrue(self.bridge.call_tool.called)

    def test_its_own_inputs_are_validated_before_any_step_runs(self):
        ex = self.executor()
        self.make(ex)
        for args in ({"layer": "twelve", "words": "x"}, {"layer": 12},
                     {"layer": 12, "words": "x", "extra": 1}):
            with self.assertRaises(ValueError):
                ex._call(call("retitle_card", args))
        self.assertFalse(self.bridge.call_tool.called)

    def test_a_made_tool_cannot_close_resolve(self):
        ex = self.executor()
        self.make(ex, name="shut_it_down", inputs=[],
                  steps=[{"tool": "resolve_control", "arguments": {"action": "quit"}}])
        text, _, _ = ex._call(call("shut_it_down"))
        self.assertIn("TOOL ERROR", text)
        self.assertIn("unsaved work", text)
        self.assertFalse(self.bridge.call_tool.called)

    def test_a_failed_step_stops_the_rest_and_says_what_already_ran(self):
        ex = self.executor()
        self.make(ex)
        self.bridge.call_tool.side_effect = [
            {"content": [{"type": "text", "text": "done"}]}, RuntimeError("bridge fell over")]
        text, wrote, read = ex._call(call("retitle_card", {"layer": 12, "words": "Hello"}))
        self.assertTrue(text.startswith("TOOL ERROR"))
        self.assertIn("stopped at step 2 of 2", text)
        self.assertIn("earlier steps already ran", text)
        self.assertTrue(wrote)                 # the edit happened; the read did not
        self.assertFalse(read)

    def test_a_restored_task_must_still_be_inspected_first(self):
        record = tasks.TaskRecord()
        record.status = "restored"
        ex = self.executor(record=record)
        self.make(ex)
        text, _, _ = ex._call(call("retitle_card", {"layer": 12, "words": "Hello"}))
        self.assertIn("Inspect the current project", text)
        self.assertFalse(self.bridge.call_tool.called)

    def test_a_tab_with_no_tools_is_offered_no_tool_maker(self):
        self.assertEqual(tasks.inference_tools([], self.library), [])

    # ------------------------------------------------------------- persistence
    def test_a_made_tool_comes_back_in_the_next_session(self):
        self.make(self.executor())
        again = toolsmith.Library("after-effects", self.library.dir)
        allowed, specs = toolsmith.contracts(eng.to_openai_tools(SPECS), SPECS)
        self.assertEqual(again.load(allowed, specs), [])
        self.assertEqual([m.name for m in again.ordered()], ["retitle_card"])
        self.assertEqual(again.get("retitle_card").steps, definition()["steps"])

    def test_a_tool_built_on_a_tool_this_tab_no_longer_has_is_left_out(self):
        self.make(self.executor())
        narrowed = [s for s in SPECS if s["name"] != "set_text"]
        allowed, specs = toolsmith.contracts(eng.to_openai_tools(narrowed), narrowed)
        again = toolsmith.Library("after-effects", self.library.dir)
        problems = again.load(allowed, specs)
        self.assertEqual(len(problems), 1)
        self.assertIn("set_text", problems[0])
        self.assertEqual(again.ordered(), [])

    def test_a_wrecked_file_costs_one_tool_and_not_the_app(self):
        self.make(self.executor())
        with open(os.path.join(self.library.dir, "junk.json"), "w", encoding="utf-8") as f:
            f.write("{ not json")
        with open(os.path.join(self.library.dir, "wrong_name.json"), "w", encoding="utf-8") as f:
            json.dump(dict(definition(), version=1, name="other_name"), f)
        again = toolsmith.Library("after-effects", self.library.dir)
        problems = again.load()
        self.assertEqual(len(problems), 2)
        self.assertEqual([m.name for m in again.ordered()], ["retitle_card"])

    def test_nowhere_to_save_is_said_out_loud(self):
        self.library.dir = None
        text = self.make(self.executor())
        self.assertIn("this session only", text)
        self.assertIn("retitle_card", self.library.made)

    def test_forgetting_a_tool_removes_its_file(self):
        self.make(self.executor())
        path = self.library.path("retitle_card")
        self.assertTrue(os.path.exists(path))
        self.assertTrue(self.library.remove("retitle_card"))
        self.assertFalse(os.path.exists(path))
        self.assertEqual(self.library.ordered(), [])


if __name__ == "__main__":
    unittest.main()
