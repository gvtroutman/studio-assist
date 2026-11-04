"""Saved OpenCode tool outcomes stay evidence and can be read without repeating them."""
import unittest
from unittest.mock import patch

import apps.opencode.mcp as oc
import core.agent as eng


def message(parts, sid="ses_one"):
    return {"info": {"role": "assistant", "sessionID": sid}, "parts": parts}


def tool(output="", **state):
    return {"type": "tool", "tool": "bash", "callID": "call_one",
            "sessionID": "ses_one", "state": {
                "status": "completed", "input": {"command": "python -m unittest"},
                "metadata": {"exit": 0}, "output": output, **state}}


def text(result):
    return "\n".join(p["text"] for p in result["content"] if p["type"] == "text")


class TestResultSummary(unittest.TestCase):
    def test_failed_shell_exit_survives_a_confident_reply(self):
        out = oc.summarize(message([tool("FAILED (failures=1)", metadata={"exit": 1}),
                                   {"type": "text", "text": "Tests passed."}]))
        self.assertIn("python -m unittest", out)
        self.assertIn("exit=1", out)
        self.assertIn("FAILED (failures=1)", out)
        self.assertIn("tool completion is not a passing check", out)
        self.assertIn("call_id=call_one", out)

    def test_empty_missing_and_compacted_output_have_distinct_meanings(self):
        empty = oc.summarize(message([tool()]))
        self.assertIn("saved result is empty", empty)
        absent = tool()
        absent["state"].pop("output")
        missing = oc.summarize(message([absent]))
        self.assertIn("No saved output or error; result unknown", missing)
        compacted = oc.summarize(message([tool("OLD SUCCESS", time={"compacted": 12})]))
        self.assertIn("cleared by compaction; result unknown", compacted)
        self.assertNotIn("OLD SUCCESS", compacted)

    def test_long_output_and_final_text_keep_both_ends_with_recovery_hint(self):
        out = oc.summarize(message([
            tool("OUTPUT HEAD\n" + "x" * 3000 + "\nFAILED FOOTER"),
            {"type": "text", "text": "ANSWER HEAD\n" + "y" * 3000 + "\nWORK REMAINS"},
        ]), 2500)
        self.assertLessEqual(len(out), 2500)
        for phrase in ("OUTPUT HEAD", "FAILED FOOTER", "opencode_tool_result",
                       "session_id=ses_one", "call_id=call_one", "ANSWER HEAD", "WORK REMAINS",
                       "characters omitted"):
            self.assertIn(phrase, out)

    def test_unknown_exit_does_not_become_success(self):
        out = oc.summarize(message([tool("OK", metadata={})]))
        self.assertIn("exit=unknown", out)
        self.assertIn("not a passing check", out)

    def test_tiny_excerpt_is_still_bounded(self):
        self.assertLessEqual(len(oc._excerpt("x" * 100, 5)), 5)

    def test_combined_session_history_keeps_the_final_warning_inside_budget(self):
        messages = [message([{"type": "text", "text": "x" * 2500}]) for _ in range(5)]
        messages.append(message([{"type": "text", "text": "LAST MESSAGE\n" + "y" * 2300
                                 + "\nFINAL WARNING: tests failed"}]))
        with patch.object(oc, "messages", return_value=messages):
            out = text(oc.call_tool("opencode_get_session", {"session_id": "ses_one"}))
        self.assertLessEqual(len(out), oc.MAX_REPLY_CHARS)
        self.assertIn("FINAL WARNING: tests failed", out)
        self.assertIn("limit=1", out)


class TestSavedToolResult(unittest.TestCase):
    def read(self, parts, **args):
        with patch.object(oc, "messages", return_value=[message(parts)]) as saved:
            result = oc.call_tool("opencode_tool_result", {
                "session_id": "ses_one", "call_id": "call_one", **args})
        saved.assert_called_once_with("ses_one")
        return result

    def test_exact_output_windows_are_stable_and_recoverable(self):
        output = "first\n" + "0123456789\u00e9" * 900 + "\nlast"
        one = self.read([tool(output)], limit=6000)
        self.assertFalse(one.get("isError"), one)
        header, first = text(one).split("\n\n", 1)
        self.assertEqual(first, output[:6000])
        self.assertIn("start=6000", header)
        self.assertLessEqual(len(text(one)), 6600)
        two = self.read([tool(output)], start=6000)
        self.assertEqual(text(two).split("\n\n", 1)[1], output[6000:])
        self.assertEqual(text(self.read([tool(output)], start=6000)), text(two))

    def test_error_text_is_recovered_without_running_a_tool(self):
        part = tool(status="error", error="permission refused\nretry requires approval")
        part["state"].pop("output")
        out = self.read([part])
        self.assertIn("status=error", text(out))
        self.assertEqual(text(out).split("\n\n", 1)[1], part["state"]["error"])

    def test_output_and_error_are_both_preserved(self):
        out = self.read([tool("stdout", status="error", error="failed")])
        self.assertEqual(text(out).split("\n\n", 1)[1], "stdout\n\nTool error:\nfailed")

    def test_empty_result_is_known_empty_and_missing_is_unknown(self):
        empty = self.read([tool()])
        self.assertFalse(empty.get("isError"), empty)
        self.assertIn("Empty saved result", text(empty))
        missing = tool()
        missing["state"].pop("output")
        out = self.read([missing])
        self.assertTrue(out["isError"])
        self.assertIn("result unknown", text(out))

    def test_compacted_output_is_never_returned_as_saved_evidence(self):
        out = self.read([tool("STALE PRIVATE OUTPUT", time={"compacted": 1})])
        self.assertTrue(out["isError"])
        self.assertIn("cleared by compaction; result unknown", text(out))
        self.assertNotIn("STALE PRIVATE OUTPUT", text(out))

    def test_other_sessions_and_nonmatching_call_ids_are_excluded(self):
        other_message = message([tool("OTHER SESSION")], "ses_two")
        other_part = tool("OTHER PART")
        other_part["sessionID"] = "ses_two"
        mismatch = tool("OTHER CALL")
        mismatch["callID"] = "call_one_suffix"
        with patch.object(oc, "messages", return_value=[other_message, message([other_part, mismatch])]):
            out = oc.call_tool("opencode_tool_result", {"session_id": "ses_one", "call_id": "call_one"})
        self.assertTrue(out["isError"])
        self.assertIn("No saved tool call", text(out))
        for secret in ("OTHER SESSION", "OTHER PART", "OTHER CALL"):
            self.assertNotIn(secret, text(out))

    def test_duplicate_call_id_is_ambiguous(self):
        out = self.read([tool("one"), tool("two")])
        self.assertTrue(out["isError"])
        self.assertIn("ambiguous", text(out))

    def test_offsets_and_limits_are_validated_and_end_is_explicit(self):
        for args in ({"start": -1}, {"limit": 0}, {"limit": 6001}):
            with self.subTest(args=args):
                out = oc.call_tool("opencode_tool_result", {
                    "session_id": "ses_one", "call_id": "call_one", **args})
                self.assertTrue(out["isError"])
        out = self.read([tool("small")], start=10)
        self.assertIn("past the end", text(out))
        self.assertEqual(text(out).split("\n\n", 1)[1], "")

    def test_contract_is_read_only_and_available_to_the_tab(self):
        spec = next(t for t in oc.tool_list() if t["name"] == "opencode_tool_result")
        self.assertTrue(spec["annotations"]["readOnlyHint"])
        self.assertIn("opencode_tool_result", eng.OPENCODE_GROUPS["discover"])

    def test_saved_tool_header_fields_are_bounded(self):
        part = tool("output", status="s" * 20000)
        part["tool"] = "t" * 20000
        out = self.read([part])
        self.assertLessEqual(len(text(out)), 600)
        self.assertIn("characters omitted", text(out))


if __name__ == "__main__":
    unittest.main()
