"""Model choices and retries keep the last working state until verified."""

import copy
import io
import json
import os
import queue
import unittest
import urllib.error
from types import SimpleNamespace
from unittest.mock import Mock, patch

from core import agent as eng
from core import chat


def sse(*chunks):
    lines = ["data: " + json.dumps(chunk) for chunk in chunks]
    return ("\n".join(lines + ["data: [DONE]"]) + "\n").encode()


def refused(message):
    return sse({"error": {"message": message}})


def streamed_ok():
    return sse({"choices": [{"delta": {"content": "done"}, "finish_reason": "stop"}]})


def http_refusal():
    return urllib.error.HTTPError("http://host/v1/chat/completions", 400, "bad", {},
                                  io.BytesIO(b"draft vocabulary mismatch"))


class TestDraftRecovery(unittest.TestCase):
    def setUp(self):
        self.llm = eng.LLM("http://host/v1", "executor", draft="draft")
        self.messages = [{"role": "user", "content": "Inspect the scene."}]
        self.calls = []

    def host(self, *responses):
        remaining = iter(responses)

        def open_request(request, timeout=None):
            self.calls.append(json.loads(request.data))
            response = next(remaining)
            if callable(response):
                response = response()
            if isinstance(response, BaseException):
                raise response
            return io.BytesIO(response)

        mocked = patch("urllib.request.urlopen", side_effect=open_request)
        mocked.start()
        self.addCleanup(mocked.stop)

    def assert_pair_kept(self):
        self.assertEqual(self.llm.draft, "draft")
        self.assertIsNone(self.llm.draft_note)

    def test_two_stream_refusals_keep_pair_and_initial_error(self):
        self.host(refused("draft refused"), refused("executor refused"))
        with self.assertRaisesRegex(RuntimeError, "draft refused"):
            self.llm.stream(self.messages)
        self.assertEqual([call.get("draft_model") for call in self.calls], ["draft", None])
        self.assert_pair_kept()

    def test_incomplete_plain_retry_keeps_pair(self):
        self.host(refused("draft refused"), b'data: {"choices": []}\n')
        with self.assertRaisesRegex(RuntimeError, "Incomplete inference response"):
            self.llm.stream(self.messages)
        self.assertEqual(len(self.calls), 2)
        self.assert_pair_kept()

    def test_plain_retry_preserves_context_error_type(self):
        self.host(refused("draft refused"),
                  sse({"choices": [{"delta": {}, "finish_reason": "length"}]}))
        with self.assertRaises(eng.ContextLimitError):
            self.llm.stream(self.messages)
        self.assert_pair_kept()

    def test_partial_text_is_never_replayed(self):
        self.host(sse({"choices": [{"delta": {"content": "partial"}}]},
                      {"error": {"message": "stream failed"}}))
        text = []
        with self.assertRaisesRegex(RuntimeError, "stream failed"):
            self.llm.stream(self.messages, on_text=text.append)
        self.assertEqual((len(self.calls), text), (1, ["partial"]))
        self.assert_pair_kept()

    def test_partial_tool_preparation_is_never_replayed(self):
        self.host(sse({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "call", "function": {"name": "read_scene", "arguments": "{"}}
        ]}}]}, {"error": {"message": "stream failed"}}))
        prepared = []
        with self.assertRaisesRegex(RuntimeError, "stream failed"):
            self.llm.stream(self.messages, on_tool=prepared.append)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(prepared[0]["id"], "call")
        self.assert_pair_kept()

    def test_http_retry_then_stream_refusal_has_no_third_attempt(self):
        self.host(http_refusal(), refused("executor refused"))
        with self.assertRaisesRegex(RuntimeError, "executor refused"):
            self.llm.stream(self.messages)
        self.assertEqual(len(self.calls), 2)
        self.assert_pair_kept()

    def test_http_retry_then_broken_json_does_not_disable_draft(self):
        self.host(http_refusal(), b"not JSON")
        with self.assertRaises(json.JSONDecodeError):
            self.llm.chat(self.messages)
        self.assert_pair_kept()

    def test_http_retry_then_error_payload_does_not_disable_draft(self):
        self.host(http_refusal(), b'{"error": "executor refused"}')
        self.assertEqual(self.llm.chat(self.messages), {"error": "executor refused"})
        self.assert_pair_kept()

    def test_http_retry_then_complete_stream_disables_verified_pair(self):
        self.host(http_refusal(), streamed_ok())
        self.assertEqual(self.llm.stream(self.messages)["content"], "done")
        self.assertEqual(len(self.calls), 2)
        self.assertIsNone(self.llm.draft)
        self.assertIn("draft vocabulary mismatch", self.llm.draft_note)

    def test_stream_retry_uses_request_settings_and_keeps_new_pair(self):
        def changed():
            self.llm.model, self.llm.draft = "new executor", "new draft"
            self.llm.temperature = 0.9
            return refused("old pair refused")

        tools = [{"type": "function", "function": {"name": "read_scene"}}]
        self.host(changed, streamed_ok())
        self.assertEqual(self.llm.stream(self.messages, tools, max_tokens=77)["content"], "done")
        first, retry = self.calls
        self.assertEqual(retry, {key: value for key, value in first.items() if key != "draft_model"})
        self.assertEqual((retry["model"], retry["temperature"], retry["max_tokens"]),
                         ("executor", 0.2, 77))
        self.assertEqual((self.llm.model, self.llm.draft), ("new executor", "new draft"))
        self.assertIsNone(self.llm.draft_note)

    def test_http_retry_does_not_disable_a_new_pair(self):
        def changed():
            self.llm.model, self.llm.draft = "new executor", "new draft"
            return http_refusal()

        self.host(changed, b'{"choices": [{"message": {"content": "done"}}]}')
        self.llm.chat(self.messages)
        self.assertEqual([call["model"] for call in self.calls], ["executor", "executor"])
        self.assertEqual(self.llm.draft, "new draft")
        self.assertIsNone(self.llm.draft_note)


class TestModelSelection(unittest.TestCase):
    def test_missing_pin_preserves_registry_preference(self):
        app = copy.copy(eng.APPS_BY_ID["comfyui"])
        app.models = ["small"]
        with patch.dict(os.environ, {"STUDIO_MODEL_COMFYUI": "missing"}):
            model, note = app.model_for(["shared", "small"], "shared")
            self.assertEqual(model, "small")
            self.assertIn("does not serve", note)
            self.assertIn("using small", note)
            self.assertEqual(os.environ["STUDIO_MODEL_COMFYUI"], "missing")

    def test_missing_coder_pin_preserves_best_coder_choice(self):
        app = eng.APPS_BY_ID["opencode"]
        with patch.dict(os.environ, {"STUDIO_MODEL_OPENCODE": "missing"}):
            model, note = app.model_for(["shared", "qwen3-coder-30b-a3b-instruct"], "shared")
        self.assertEqual(model, "qwen3-coder-30b-a3b-instruct")
        self.assertIn("does not serve", note)
        self.assertIn("coding", note)


class Prefs:
    def __init__(self):
        self.pins = {"chat": "old"}
        self.saved = []

    def get(self, key):
        return self.pins

    def set(self, **values):
        self.pins = values["model_pins"]
        self.saved.append(dict(self.pins))


class TestModelSwitch(unittest.TestCase):
    """Exercise state changes through the real handlers without constructing Tk."""
    def setUp(self):
        self.old = eng.LLM("http://host/v1", "old")
        self.new = eng.LLM("http://host/v1", "new")
        self.session = SimpleNamespace(id="chat", app=eng.CHAT, event_id=("chat", "generation"),
                                       llm=self.old, busy=False, booting=False, closed=False,
                                       prefix_tokens=2500, window=8192,
                                       messages=[{"role": "system", "content": "brief"}],
                                       record=object(), tools=[], library=None)
        self.owner = SimpleNamespace(prefs=Prefs(), sessions={"chat": self.session},
                                     model_ids=["old", "new"], q=queue.Queue(), host="http://host/v1",
                                     _llm_for=Mock(return_value=self.new), _warm=Mock(),
                                     _apply_status=Mock(), _spawn=Mock())

    def events(self):
        result = []
        while not self.owner.q.empty():
            result.append(self.owner.q.get_nowait())
        return result

    def test_stale_menu_choice_does_not_change_preferences_or_environment(self):
        with patch.dict(os.environ, {"STUDIO_MODEL_CHAT": "old"}):
            chat.Chat._pin_model(self.owner, self.session, "uninstalled")
            self.assertEqual(os.environ["STUDIO_MODEL_CHAT"], "old")
        self.assertEqual(self.owner.prefs.pins, {"chat": "old"})
        self.assertFalse(self.owner.prefs.saved)
        self.owner._spawn.assert_not_called()
        self.assertIn("no longer serves", self.events()[0][2])

    def test_busy_closed_and_replaced_tabs_cannot_save_a_stale_pin(self):
        for state in ("busy", "closed", "replaced"):
            with self.subTest(state=state), patch.dict(os.environ, {"STUDIO_MODEL_CHAT": "old"}):
                self.session.busy = state == "busy"
                self.session.closed = state == "closed"
                self.owner.sessions["chat"] = object() if state == "replaced" else self.session
                chat.Chat._pin_model(self.owner, self.session, "new")
                self.assertEqual(os.environ["STUDIO_MODEL_CHAT"], "old")
                self.assertFalse(self.owner.prefs.saved)
                self.owner._spawn.assert_not_called()

    def test_failed_switch_restores_client_and_context_without_resetting_task(self):
        messages, record = self.session.messages, self.session.record
        self.owner._warm.side_effect = eng.HostUnreachable("host unavailable")
        self.session.busy = True
        with self.assertRaises(eng.HostUnreachable):
            chat.Chat._swap_model(self.owner, self.session)
        self.assertIs(self.session.llm, self.old)
        self.assertEqual((self.session.prefix_tokens, self.session.window), (2500, 8192))
        self.assertIs(self.session.messages, messages)
        self.assertIs(self.session.record, record)
        self.assertFalse(self.session.busy)
        self.assertFalse(any(kind == "status" and value[0] == "ready"
                             for kind, _, value in self.events()))

    def test_recovery_does_not_overwrite_a_newer_client(self):
        newer = eng.LLM("http://host/v1", "newer")

        def warm(session, strict=False):
            session.llm = newer
            session.prefix_tokens, session.window = 900, 4096
            raise RuntimeError("old switch failed")

        self.owner._warm.side_effect = warm
        with self.assertRaisesRegex(RuntimeError, "old switch failed"):
            chat.Chat._swap_model(self.owner, self.session)
        self.assertIs(self.session.llm, newer)
        self.assertEqual((self.session.prefix_tokens, self.session.window), (900, 4096))

    def test_successful_switch_warms_before_reporting_ready(self):
        chat.Chat._swap_model(self.owner, self.session)
        self.owner._warm.assert_called_once_with(self.session, strict=True)
        self.assertIs(self.session.llm, self.new)
        self.assertTrue(any(kind == "status" and value == ("ready", "ok", False)
                            for kind, _, value in self.events()))

    def test_same_model_adopts_shared_client_without_rewarming(self):
        self.new.model = self.old.model
        chat.Chat._swap_model(self.owner, self.session)
        self.assertIs(self.session.llm, self.new)
        self.owner._warm.assert_not_called()
        self.assertEqual((self.session.prefix_tokens, self.session.window), (2500, 8192))

    def test_chip_distinguishes_requested_pin_from_actual_fallback(self):
        self.assertEqual(chat.Chat._model_mode(self.owner, self.session), "Pinned")
        self.owner.prefs.pins["chat"] = "missing"
        self.assertEqual(chat.Chat._model_mode(self.owner, self.session), "Fallback")
        self.owner.prefs.pins.clear()
        self.assertEqual(chat.Chat._model_mode(self.owner, self.session), "Auto")

    def test_warmup_failure_is_strict_only_for_model_switch(self):
        self.owner._fit = Mock(return_value=(False, ""))
        self.owner._host_lost = Mock()
        self.owner._draft_check = Mock()
        self.session.llm.chat = Mock(side_effect=eng.HostUnreachable("host unavailable"))
        with patch.object(eng, "loaded_instances", return_value=[]):
            with self.assertRaises(eng.HostUnreachable):
                chat.Chat._warm(self.owner, self.session, strict=True)
            chat.Chat._warm(self.owner, self.session)
        self.owner._draft_check.assert_called_once_with(self.session, self.old)


if __name__ == "__main__":
    unittest.main()
