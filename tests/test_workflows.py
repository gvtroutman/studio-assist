"""Workflow adapters exercise the real executor with fake bridge contracts."""
import json
import unittest
from unittest.mock import Mock

import studio_agent as eng
import studio_tasks as tasks
import studio_workflows as workflows


def call(name, args=None):
    return {"function": {"name": name, "arguments": json.dumps(args or {})}}


class WorkflowTests(unittest.TestCase):
    def executor(self, **kwargs):
        # Deliberately include a 2020-12 tuple: aliases must validate against the
        # original schema, even though inference sees a sanitized version.
        specs = [{"name": "get_comp", "description": "Full contract " * 400,
                  "inputSchema": {"type": "object", "required": ["compId"],
                      "properties": {"compId": {"type": "integer"},
                                     "pair": {"type": "array", "prefixItems": [
                                         {"type": "number", "maximum": 1}], "items": False}},
                      "additionalProperties": False}}]
        bridge = Mock()
        bridge.call_tool.return_value = {"content": [{"type": "text", "text": '{"id": 7}'}]}
        return tasks.Executor(None, bridge, eng.to_openai_tools(specs), schemas=specs, **kwargs)

    def test_capabilities_do_not_claim_project_inspection(self):
        record = tasks.TaskRecord()
        record.status = "restored"
        ex = self.executor(record=record)
        result, wrote, read = ex._call(call("studio_workflow_capabilities"))
        report = json.loads(result)
        self.assertTrue(report["inspect_comp"]["available"])
        self.assertFalse(report["inspect_layer"]["available"])
        self.assertFalse(report["get_transcript"]["available"])
        self.assertEqual(len(report), 13)
        self.assertFalse(report["apply_miter_style"]["available"])
        self.assertFalse(report["apply_ellwood_style"]["available"])
        self.assertNotIn("apply_method_form_house_style", report)
        self.assertFalse(wrote or read)
        self.assertTrue(ex.must_inspect)
        ex.mcp.call_tool.assert_not_called()

    def test_inspection_dispatch_and_journal_use_bridge_identity(self):
        ex = self.executor()
        result, wrote, read = ex._call(call("inspect_comp", {"compId": 7}))
        ex.mcp.call_tool.assert_called_once_with("get_comp", {"compId": 7})
        self.assertIn('7', result)
        self.assertFalse(wrote)
        self.assertTrue(read)
        self.assertEqual(ex.record.journal[0]["name"], "get_comp")

    def test_original_schema_rejects_invalid_alias_arguments(self):
        ex = self.executor()
        with self.assertRaises(ValueError):
            ex._call(call("inspect_comp", {"compId": 7, "pair": [2]}))
        ex.mcp.call_tool.assert_not_called()

    def test_alias_cannot_repeat_failed_underlying_call(self):
        ex = self.executor()
        ex.mcp.call_tool.return_value = {"isError": True, "content": []}
        ex._call(call("get_comp", {"compId": 7}))
        with self.assertRaisesRegex(ValueError, "already failed"):
            ex._call(call("inspect_comp", {"compId": 7}))
        self.assertEqual(ex.mcp.call_tool.call_count, 1)

    def test_disabled_bridge_schema_does_not_enable_workflow(self):
        ex = tasks.Executor(None, Mock(), [], schemas=[{"name": "get_comp"}])
        with self.assertRaisesRegex(ValueError, "not enabled"):
            ex._call(call("inspect_comp", {"compId": 7}))
        ex.mcp.call_tool.assert_not_called()

    def test_persistence_failure_prevents_inspection(self):
        ex = self.executor(checkpoint=Mock(side_effect=OSError("disk full")))
        with self.assertRaises(tasks.CheckpointError):
            ex._call(call("inspect_comp", {"compId": 7}))
        ex.mcp.call_tool.assert_not_called()

    def test_tool_contract_matches_warmup_and_preserves_description(self):
        ex = self.executor()
        self.assertEqual(ex.tools, tasks.inference_tools(ex.bridge_tools))
        alias = next(t for t in ex.tools if t["function"]["name"] == "inspect_comp")
        original = ex.bridge_tools[0]["function"]
        self.assertEqual(alias["function"]["parameters"], original["parameters"])
        self.assertTrue(alias["function"]["description"].endswith(original["description"]))
        names = {t["function"]["name"] for t in ex.tools}
        self.assertFalse(names.intersection(workflows.PENDING))


if __name__ == "__main__":
    unittest.main()
