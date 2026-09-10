"""
Offline tests. No network, no creative apps, no model.

    python -m unittest discover -s tests -v

Anything needing a display skips itself when there isn't one.
"""

import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import studio_agent as eng


class TestSchemaSanitizing(unittest.TestCase):
    """LM Studio 400s the whole request on 2020-12 tuple syntax. See AGENTS.md."""

    def test_tuple_becomes_bounded_array(self):
        out = eng.sanitize_schema({
            "type": "array", "minItems": 3, "maxItems": 3, "items": False,
            "prefixItems": [{"type": "number"}, {"type": "number"},
                            {"type": "number"}],
        })
        self.assertNotIn("prefixItems", out)
        self.assertEqual(out["items"], {"type": "number"})
        self.assertEqual((out["minItems"], out["maxItems"]), (3, 3))

    def test_bounds_inferred_when_absent(self):
        out = eng.sanitize_schema({
            "type": "array", "items": False,
            "prefixItems": [{"type": "number"}, {"type": "number"}],
        })
        self.assertEqual((out["minItems"], out["maxItems"]), (2, 2))

    def test_mixed_tuple_left_unconstrained(self):
        out = eng.sanitize_schema({
            "type": "array", "items": False,
            "prefixItems": [{"type": "string"}, {"type": "number"}],
        })
        self.assertNotIn("items", out)
        self.assertEqual(out["maxItems"], 2)

    def test_boolean_items_never_survives(self):
        for val in (True, False):
            self.assertNotIn("items", eng.sanitize_schema({"type": "array",
                                                           "items": val}))

    def test_recurses_into_nested_structures(self):
        out = eng.sanitize_schema({
            "type": "object",
            "properties": {
                "color": {"type": "array", "items": False,
                          "prefixItems": [{"type": "number"}]},
                "nested": {"type": "object", "properties": {
                    "pos": {"type": "array", "items": False,
                            "prefixItems": [{"type": "number"},
                                            {"type": "number"}]}}},
            },
            "anyOf": [{"type": "array", "items": False,
                       "prefixItems": [{"type": "number"}]}],
        })
        self.assertEqual(out["properties"]["color"]["items"], {"type": "number"})
        self.assertEqual(
            out["properties"]["nested"]["properties"]["pos"]["maxItems"], 2)
        self.assertNotIn("prefixItems", out["anyOf"][0])

    def test_schema_key_dropped(self):
        self.assertNotIn("$schema", eng.sanitize_schema(
            {"$schema": "https://json-schema.org/draft/2020-12/schema",
             "type": "object"}))

    def test_plain_schema_untouched(self):
        src = {"type": "object", "properties": {"n": {"type": "integer"}},
               "required": ["n"]}
        self.assertEqual(eng.sanitize_schema(src), src)


class TestModelChoice(unittest.TestCase):
    def test_explicit_request_wins(self):
        self.assertEqual(eng.pick_model("a", ["a", "b"], want="b"), "b")

    def test_falls_back_to_loaded_when_request_absent(self):
        self.assertEqual(eng.pick_model("a", ["a", "b"], want="nope"), "a")

    def test_prefers_known_good_when_nothing_loaded(self):
        ids = ["random-7b", eng.PREFERRED_MODELS[0]]
        self.assertEqual(eng.pick_model(None, ids), eng.PREFERRED_MODELS[0])

    def test_last_resort_is_first_available(self):
        self.assertEqual(eng.pick_model(None, ["only-one"]), "only-one")

    def test_no_models_is_none(self):
        self.assertIsNone(eng.pick_model(None, []))


class TestToolResults(unittest.TestCase):
    def test_text_parts_joined(self):
        self.assertEqual(
            eng.mcp_result_to_text({"content": [{"type": "text", "text": "a"},
                                                {"type": "text", "text": "b"}]}),
            "a\nb")

    def test_error_flagged_for_the_model(self):
        out = eng.mcp_result_to_text({"isError": True,
                                      "content": [{"type": "text", "text": "boom"}]})
        self.assertTrue(out.startswith("TOOL ERROR:"))

    def test_long_output_truncated(self):
        big = "x" * (eng.MAX_TOOL_RESULT_CHARS + 500)
        out = eng.mcp_result_to_text({"content": [{"type": "text", "text": big}]})
        self.assertLess(len(out), len(big))
        self.assertIn("truncated", out)

    def test_image_described_not_dumped(self):
        out = eng.mcp_result_to_text({"content": [{"type": "image", "data": "A" * 999,
                                                   "mimeType": "image/png"}]})
        self.assertIn("image returned", out)
        self.assertNotIn("AAAA", out)


class TestToolConversion(unittest.TestCase):
    def test_openai_conversion_shape(self):
        tools = eng.to_openai_tools([{"name": "t", "description": "d",
                                      "inputSchema": {"type": "object"}}])
        self.assertEqual(tools[0]["type"], "function")
        self.assertEqual(tools[0]["function"]["name"], "t")

    def test_missing_schema_tolerated(self):
        tools = eng.to_openai_tools([{"name": "t"}])
        self.assertEqual(tools[0]["function"]["parameters"]["type"], "object")

    def test_compound_tool_action_list_survives(self):
        """
        Resolve's tools document every action in the description - it is the API
        surface, not prose. A 1024-char clip amputated half of `timeline` and the
        model then invented action names. See AGENTS.md.
        """
        desc = "Timeline operations.\n" + "\n".join(
            "  action_%03d(x) -> {ok}" % i for i in range(100))
        self.assertGreater(len(desc), 1024)
        out = eng.to_openai_tools([{"name": "timeline", "description": desc}])
        kept = out[0]["function"]["description"]
        self.assertIn("action_099", kept)
        self.assertLessEqual(len(kept), eng.MAX_TOOL_DESC_CHARS)


class TestAppRegistry(unittest.TestCase):
    """The registry is the only place an app is described. Keep it walkable."""

    def test_ids_unique_and_indexed(self):
        ids = [a.id for a in eng.APPS]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), set(eng.APPS_BY_ID))
        self.assertIn(eng.DEFAULT_APP, eng.APPS_BY_ID)

    def test_every_app_is_fully_described(self):
        for app in eng.APPS:
            with self.subTest(app=app.id):
                self.assertTrue(app.name and app.tab and app.code)
                self.assertTrue(app.fg.startswith("#") and app.bg.startswith("#"))
                self.assertTrue(app.command)
                self.assertTrue(app.exe_globs)
                self.assertTrue(app.examples)
                self.assertTrue(app.system_prompt.strip())
                self.assertIn("no further tool calls", app.system_prompt)

    def test_default_groups_exist_and_resolve(self):
        for app in eng.APPS:
            with self.subTest(app=app.id):
                for g in app.default_groups:
                    self.assertIn(g, app.groups)
                self.assertTrue(app.tool_names())

    def test_groups_never_empty(self):
        for app in eng.APPS:
            for name, tools in app.groups.items():
                with self.subTest(app=app.id, group=name):
                    self.assertTrue(tools)

    def test_probe_strategy_is_understood(self):
        for app in eng.APPS:
            kind, _, arg = app.probe.partition(":")
            with self.subTest(app=app.id):
                self.assertIn(kind, ("port", "process"))
                self.assertTrue(arg)

    def test_chat_prompt_extends_the_cli_prompt(self):
        for app in eng.APPS:
            self.assertTrue(app.chat_prompt().startswith(app.system_prompt))
            self.assertIn("continuing conversation", app.chat_prompt())

    def test_resolve_is_told_not_to_quit_the_host_app(self):
        """resolve_control has a quit action; losing the user's session is not ours."""
        resolve = eng.APPS_BY_ID["resolve"]
        self.assertIn("quit", resolve.system_prompt)
        self.assertIn("NEVER", resolve.system_prompt)

    def test_after_effects_is_told_to_identify_layers_by_id(self):
        """An index shifts on every insert; the prompt has to keep saying so."""
        ae = eng.APPS_BY_ID["after-effects"]
        self.assertIn("`index`", ae.system_prompt)
        self.assertIn("NEVER", ae.system_prompt)

    def test_prompts_only_name_tools_the_app_exposes(self):
        """
        The prompts brief the model on real tools by name. Move one out of
        default_groups and the prompt is left telling the model to call something
        that is not in its tool list - which it answers by inventing a call. So:
        anything a prompt names must be in that app's working set.
        """
        for app in eng.APPS:
            everything = {t for names in app.groups.values() for t in names}
            for tool in sorted(everything - app.tool_names()):
                with self.subTest(app=app.id, tool=tool):
                    self.assertIsNone(
                        re.search(r"\b%s\b" % re.escape(tool), app.system_prompt),
                        "prompt names %s, which default_groups does not expose" % tool)

    def test_unknown_app_names_the_alternatives(self):
        with self.assertRaises(KeyError) as ctx:
            eng.get_app("nope")
        self.assertIn("after-effects", str(ctx.exception))


class TestDetection(unittest.TestCase):
    def test_detect_apps_shape(self):
        for a in eng.detect_apps():
            self.assertEqual({"code", "name", "version", "fg", "bg", "id", "drivable"},
                             set(a))
            self.assertTrue(a["fg"].startswith("#"))

    def test_drivable_is_derived_from_the_registry(self):
        self.assertEqual(eng.DRIVABLE, {a.name: a.id for a in eng.APPS})

    def test_drivable_rows_carry_a_registry_id(self):
        for a in eng.detect_apps():
            if a["drivable"]:
                self.assertIn(a["id"], eng.APPS_BY_ID)
            else:
                self.assertIsNone(a["id"])

    def test_newest_match_returns_none_when_nothing_exists(self):
        self.assertIsNone(eng.newest_match([r"Z:\nothing\here\*.exe"]))


class TestEnvDefaults(unittest.TestCase):
    def test_first_set_name_wins(self):
        os.environ.pop("STUDIO_TEST_A", None)
        os.environ["STUDIO_TEST_B"] = "b"
        try:
            self.assertEqual(
                eng.env_default("STUDIO_TEST_A", "STUDIO_TEST_B", fallback="f"), "b")
        finally:
            os.environ.pop("STUDIO_TEST_B", None)

    def test_fallback_when_none_set(self):
        self.assertEqual(eng.env_default("STUDIO_TEST_MISSING", fallback="f"), "f")


def _headless():
    try:
        import tkinter
        tkinter.Tk().destroy()
        return False
    except Exception:
        return True


@unittest.skipIf(_headless(), "no display")
class TestGui(unittest.TestCase):
    """Layout invariants and per-tab isolation. See AGENTS.md on pack order."""

    @classmethod
    def setUpClass(cls):
        import studio_chat
        cls.mod = studio_chat
        # No npx, no Resolve venv, no network - and always every tab, so the
        # switching tests do not depend on what is installed on this machine.
        cls._real_installed = eng.installed_apps
        eng.installed_apps = lambda: list(eng.APPS)
        studio_chat.Chat._boot_host = lambda self: None
        studio_chat.Chat._ensure = lambda self, s: None
        cls.app = studio_chat.Chat()
        for _ in range(15):
            cls.app.update()

    @classmethod
    def tearDownClass(cls):
        cls.app.destroy()
        eng.installed_apps = cls._real_installed

    def _bottom_of(self, widget):
        return (widget.winfo_rooty() - self.app.winfo_rooty()
                + widget.winfo_height())

    def test_composer_visible_at_every_size(self):
        for w, h in ((880, 520), (900, 560), (1180, 820), (1400, 900), (890, 300)):
            with self.subTest(size="%dx%d" % (w, h)):
                self.app.geometry("%dx%d" % (w, h))
                for _ in range(15):
                    self.app.update()
                self.app.update_idletasks()
                win_h = self.app.winfo_height()
                self.assertTrue(self.app.input.winfo_ismapped())
                self.assertTrue(self.app.btn_send.winfo_ismapped())
                self.assertLessEqual(self._bottom_of(self.app.input), win_h)
                self.assertLessEqual(self._bottom_of(self.app.btn_send), win_h)
                self.assertGreater(self.app.input.winfo_height(), 5)

    def test_connections_stay_visible_under_a_long_app_list(self):
        """A full Adobe install is eight rows; the live status must survive it."""
        self.app.geometry("880x520")
        for _ in range(15):
            self.app.update()
        self.app.update_idletasks()
        win_h = self.app.winfo_height()
        self.assertEqual(set(self.app.conn), {"host"} | set(self.app.order))
        for key, (dot, lbl) in self.app.conn.items():
            with self.subTest(row=key):
                self.assertTrue(lbl.winfo_ismapped())
                self.assertLessEqual(self._bottom_of(lbl), win_h)

    def test_one_tab_per_app(self):
        self.assertEqual(set(self.app.tab_ui), set(self.app.sessions))
        self.assertEqual(len(self.app.order), len(eng.APPS))
        for sid in self.app.order:
            self.assertTrue(self.app.tab_ui[sid]["tab"].winfo_ismapped())

    def test_exactly_one_transcript_is_mapped(self):
        for sid in self.app.order:
            self.app._select(sid)
            for _ in range(5):
                self.app.update()
            mapped = [i for i in self.app.order
                      if self.app.sessions[i].frame.winfo_ismapped()]
            self.assertEqual(mapped, [sid])

    def test_switching_tabs_retitles_the_launch_button(self):
        for sid in self.app.order:
            self.app._select(sid)
            self.app.update()
            self.assertIn(self.app.sessions[sid].app.name,
                          self.app.btn_fix.cget("text"))

    def test_transcripts_do_not_bleed_between_tabs(self):
        a, b = self.app.order[0], self.app.order[1]
        self.app._handle("sys", a, "only-in-the-first-tab")
        self.app.update()
        self.assertIn("only-in-the-first-tab",
                      self.app.sessions[a].view.get("1.0", "end"))
        self.assertNotIn("only-in-the-first-tab",
                         self.app.sessions[b].view.get("1.0", "end"))

    def test_each_session_has_its_own_history_and_prompt(self):
        prompts = []
        for sid in self.app.order:
            s = self.app.sessions[sid]
            self.assertEqual(s.messages[0]["role"], "system")
            self.assertIn(s.app.name.split()[0], s.messages[0]["content"])
            prompts.append(s.messages[0]["content"])
        self.assertEqual(len(set(prompts)), len(self.app.order))
        histories = [id(self.app.sessions[i].messages) for i in self.app.order]
        self.assertEqual(len(set(histories)), len(self.app.order))

    def test_markdown_rendered_not_literal(self):
        sid = self.app.order[0]
        self.app._handle("stream_start", sid, None)
        for tok in ("Made **Title** at ", "`1920x1080`."):
            self.app._handle("token", sid, tok)
        self.app._handle("stream_end", sid, None)
        self.app.update()
        view = self.app.sessions[sid].view
        body = view.get("1.0", "end")
        self.assertNotIn("**", body)
        self.assertIn("Made Title at 1920x1080.", body)
        self.assertTrue(view.tag_ranges("b"))
        self.assertTrue(view.tag_ranges("code"))

    def test_events_for_an_unknown_session_land_on_the_active_tab(self):
        """Host-level failures have no app of their own; they must still be seen."""
        self.app._select(self.app.order[0])
        self.app._handle("error", None, "host-level-problem")
        self.app.update()
        self.assertIn("host-level-problem",
                      self.app.sessions[self.app.order[0]].view.get("1.0", "end"))

    def test_no_method_shadows_tkinter_internals(self):
        import tkinter
        clashes = [n for n in vars(self.mod.Chat)
                   if not n.startswith("__")
                   and (hasattr(tkinter.Misc, n) or hasattr(tkinter.Tk, n))]
        self.assertEqual(clashes, [], "shadowing Tk internals breaks the widget")

    def test_sidebar_labels_do_not_wrap_mid_token(self):
        self.assertEqual(self.mod.pretty_host("http://100.127.17.38:1234/v1"),
                         "100.127.17.38:1234")
        self.assertEqual(self.mod.clip("short", 24), "short")
        self.assertEqual(len(self.mod.clip("x" * 40, 24)), 24)


class TestSingleInstance(unittest.TestCase):
    def test_second_claim_refused(self):
        import studio_chat
        port = 57999
        self.assertTrue(studio_chat.claim_single_instance(port))
        self.assertFalse(studio_chat.claim_single_instance(port))


if __name__ == "__main__":
    unittest.main(verbosity=2)
