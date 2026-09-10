"""
Offline tests. No network, no creative apps, no model.

    python -m unittest discover -s tests -v

Anything needing a display skips itself when there isn't one.
"""

import json
import os
import re
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import studio_agent as eng
import studio_icons as icons


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

    def test_default_ae_tools_can_finish_and_edit_a_visible_shape(self):
        # A layer container alone cannot fulfil "make a circle". Guard the
        # complete workflow, independently of how the groups are organised.
        ae = eng.APPS_BY_ID["after-effects"]
        required = {"list_comps", "get_comp", "create_shape_layer",
                    "add_shape_content", "set_shape_path", "set_shape_property",
                    "get_layer_full"}
        self.assertFalse(required - ae.tool_names(),
                         "Default AE tools must include geometry and paint")

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


class TestPlainChat(unittest.TestCase):
    """The one tab with no app behind it: a conversation and nothing else."""

    def test_it_is_a_tab_but_never_a_drivable_app(self):
        """DRIVABLE is derived from APPS, and the sidebar must not offer chat
        as something this agent can drive."""
        self.assertNotIn(eng.CHAT, eng.APPS)
        self.assertNotIn("chat", eng.APPS_BY_ID)
        self.assertNotIn(eng.CHAT.name, eng.DRIVABLE)
        self.assertIn(eng.CHAT, eng.TABS)
        self.assertIs(eng.TABS_BY_ID["chat"], eng.CHAT)
        self.assertIs(eng.get_app("chat"), eng.CHAT)

    def test_nothing_tries_to_start_probe_or_find_it(self):
        self.assertFalse(eng.CHAT.drivable)
        self.assertTrue(all(app.drivable for app in eng.APPS))
        self.assertIsNone(eng.CHAT.exe())
        self.assertTrue(eng.CHAT.installed())    # nothing to install
        self.assertTrue(eng.CHAT.running())      # the tab is the whole of it
        with self.assertRaises(RuntimeError):
            eng.CHAT.launch()

    def test_it_offers_no_tools_and_no_groups(self):
        self.assertEqual(eng.CHAT.tool_names(), set())
        self.assertEqual(eng.CHAT.groups, {})
        self.assertEqual(eng.CHAT.default_groups, [])

    def test_the_prompt_carries_no_tool_rules(self):
        """The app suffixes brief a tab on its bridge and its tools. This tab
        has neither, and the rules would be describing something absent."""
        prompt = eng.CHAT.chat_prompt()
        self.assertEqual(prompt, eng.CHAT.system_prompt)
        self.assertEqual(prompt, eng.CHAT.cli_prompt())
        for absent in ("studio_task_update", "studio_workflow_capabilities",
                       "TASK QUALITY"):
            self.assertNotIn(absent, prompt)

    def test_the_prompt_forbids_claiming_work_it_cannot_do(self):
        """A confident "done - I added the layer" from a tab that cannot reach
        After Effects is worse than no answer at all."""
        prompt = eng.CHAT.chat_prompt()
        self.assertIn("no bridge", prompt)
        self.assertIn("never describe such a change as done", prompt)
        self.assertIn("Chat", prompt)


class TestDetection(unittest.TestCase):
    def test_detect_apps_shape(self):
        for a in eng.detect_apps():
            self.assertEqual({"code", "name", "version", "fg", "bg", "id", "exe",
                              "drivable"}, set(a))
            self.assertTrue(a["fg"].startswith("#"))

    def test_exe_path_is_real_when_found(self):
        """The sidebar reads each app's icon out of this file; a stale path
        would silently fall back to the drawn badge and hide the breakage."""
        for a in eng.detect_apps():
            if a["exe"] is not None:
                with self.subTest(app=a["name"]):
                    self.assertTrue(os.path.isfile(a["exe"]), a["exe"])

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


class TestIcons(unittest.TestCase):
    """
    The sidebar shows each app's real icon, read out of its .exe. python.exe is
    the one PE file every machine running these tests is guaranteed to have.
    """

    def test_reads_an_icon_out_of_a_real_exe(self):
        data = icons.icon_png(sys.executable, 26)
        self.assertTrue(data and data.startswith(icons.PNG_MAGIC))
        self.assertEqual(struct.unpack(">II", data[16:24]), (26, 26))

    def test_size_is_whatever_was_asked_for(self):
        for size in (16, 20, 48):
            data = icons.icon_png(sys.executable, size)
            with self.subTest(size=size):
                self.assertEqual(struct.unpack(">II", data[16:24]), (size, size))

    def test_a_missing_icon_is_never_an_error(self):
        """It is a fallback to the drawn badge, so nothing here may raise."""
        self.assertIsNone(icons.icon_png(None))
        self.assertIsNone(icons.icon_png(r"Z:\nothing\here.exe"))
        self.assertIsNone(icons.icon_png(__file__))          # not a PE at all

    def test_png_round_trip(self):
        px = bytes([255, 0, 0, 255, 0, 255, 0, 128,
                    0, 0, 255, 255, 9, 9, 9, 0])
        back, w, h = icons.png_to_rgba(icons.png(px, 2, 2))
        self.assertEqual((w, h), (2, 2))
        self.assertEqual(back, px)

    def test_resample_averages_rather_than_drops(self):
        black_and_white = bytes([0, 0, 0, 255, 255, 255, 255, 255] * 2)
        out = icons.resample(black_and_white, 2, 2, 1)
        self.assertEqual(out[0], 127)      # not 0 and not 255
        self.assertEqual(out[3], 255)

    def test_group_entry_choice_prefers_a_small_source(self):
        """256px entries are PNGs; inflating one for a 26px badge is waste."""
        group = struct.pack("<HHH", 0, 1, 3) + b"".join(
            struct.pack("<BBBBHHIH", w % 256, w % 256, 0, 0, 1, 32, 0, i)
            for i, w in enumerate((256, 48, 16)))
        self.assertEqual(icons.best_entry(group, 26)[0], 48)
        self.assertEqual(icons.best_entry(group, 12)[0], 16)


class TestPrefs(unittest.TestCase):
    """Settings live outside the checkout; nothing here may reach the real one."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "settings.json")

    def _prefs(self):
        import studio_chat
        return studio_chat.Prefs(self.path)

    def test_round_trip(self):
        p = self._prefs()
        p.set(theme="light", tabs=["resolve"], pinned=["Photoshop"], hidden=["Acrobat"])
        again = self._prefs()
        self.assertEqual(again.get("theme"), "light")
        self.assertEqual(again.get("tabs"), ["resolve"])
        self.assertEqual(again.get("pinned"), ["Photoshop"])

    def test_missing_file_is_the_default(self):
        p = self._prefs()
        self.assertEqual(p.get("theme"), "dark")
        self.assertIsNone(p.get("tabs"))     # None means "every installed app"

    def test_a_hand_wrecked_file_cannot_stop_the_window_opening(self):
        for junk in ("{not json", json.dumps([1, 2, 3]),
                     json.dumps({"theme": "chartreuse", "hidden": "nope",
                                 "tabs": 7})):
            with open(self.path, "w", encoding="utf-8") as f:
                f.write(junk)
            with self.subTest(junk=junk[:20]):
                p = self._prefs()
                self.assertIn(p.get("theme"), ("dark", "light"))
                self.assertEqual(p.get("hidden"), [])
                self.assertIsNone(p.get("tabs"))

    def test_an_unwritable_path_is_survivable(self):
        import studio_chat
        p = studio_chat.Prefs(os.path.join(self.dir, "settings.json", "no", "x.json"))
        p.set(theme="light")                 # must not raise


class TestThemes(unittest.TestCase):
    def test_both_palettes_carry_every_role(self):
        """A role missing from one palette is a KeyError mid theme switch."""
        import studio_chat
        self.assertEqual(set(studio_chat.DARK), set(studio_chat.LIGHT))
        for palette in studio_chat.THEMES.values():
            for role, value in palette.items():
                with self.subTest(role=role):
                    self.assertRegex(value, r"^#[0-9a-f]{6}$")

    def test_named_themes_are_the_ones_on_offer(self):
        import studio_chat
        self.assertEqual([k for k, _label in studio_chat.THEME_NAMES],
                         list(studio_chat.THEMES))


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
        # Settings are a real file under %APPDATA%; a test run must not touch
        # the one the user's own window is reading.
        cls.dir = tempfile.mkdtemp()
        cls._real_settings = os.environ.get("STUDIO_SETTINGS")
        os.environ["STUDIO_SETTINGS"] = os.path.join(cls.dir, "settings.json")
        # No npx, no Resolve venv, no network - and always every tab, so the
        # switching tests do not depend on what is installed on this machine.
        cls._real_installed = eng.installed_apps
        eng.installed_apps = lambda: list(eng.APPS)
        studio_chat.Chat._boot_host = lambda self: None
        studio_chat.Chat._ensure = lambda self, s: None
        studio_chat.Chat._read_icons = lambda self: None
        cls.app = studio_chat.Chat()
        for _ in range(15):
            cls.app.update()

    @classmethod
    def tearDownClass(cls):
        cls.app.destroy()
        eng.installed_apps = cls._real_installed
        if cls._real_settings is None:
            os.environ.pop("STUDIO_SETTINGS", None)
        else:
            os.environ["STUDIO_SETTINGS"] = cls._real_settings

    def setUp(self):
        """Every test starts from every tab open - chat included - looking at
        the first."""
        for app in eng.TABS:
            self.app._add_tab(app.id)      # a tab already open is just selected
        self.app._select(eng.APPS[0].id)
        self.app.update()

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
        # Two rows whatever the registry grows to: the shared host, and every
        # bridge behind one entry. The per-bridge detail is in its menu.
        self.assertEqual(set(self.app.conn), {"host", "bridges"})
        for key, (lead, lbl) in self.app.conn.items():
            with self.subTest(row=key):
                self.assertTrue(lbl.winfo_ismapped())
                self.assertLessEqual(self._bottom_of(lbl), win_h)

    def test_the_bridges_row_counts_bridges_not_tabs(self):
        """A chat tab has no bridge; counting it would report one of two
        bridges missing when nothing is missing at all."""
        self.app._sync_bridges()
        _lead, lbl = self.app.conn["bridges"]
        bridges = [i for i in self.app.order if eng.TABS_BY_ID[i].drivable]
        before = lbl.cget("text")
        self.assertIn(str(len(bridges)), before)
        self.app._close_tab(eng.CHAT.id)
        self.app._sync_bridges()
        self.assertEqual(lbl.cget("text"), before)

    def test_one_tab_per_app(self):
        self.assertEqual(set(self.app.tab_ui), set(self.app.sessions))
        self.assertEqual(len(self.app.order), len(eng.TABS))
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
            if not self.app.sessions[sid].app.drivable:
                # Nothing to start, so the button has no business showing.
                self.assertFalse(self.app.btn_fix.winfo_ismapped())
                continue
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

    def test_closing_a_tab_takes_its_session_with_it(self):
        sid = self.app.order[-1]
        self.app._close_tab(sid)
        self.app.update()
        self.assertNotIn(sid, self.app.order)
        self.assertNotIn(sid, self.app.sessions)
        self.assertNotIn(sid, self.app.tab_ui)
        self.assertNotEqual(self.app.active, sid)

    def test_a_closed_tab_marks_its_session_so_a_late_bridge_shuts_down(self):
        """A bridge that finishes starting after the tab went would otherwise
        leave its subprocess running for the rest of the session."""
        sid = self.app.order[-1]
        s = self.app.sessions[sid]
        self.app._close_tab(sid)
        self.assertTrue(s.closed)

    def test_reopening_a_tab_is_a_fresh_conversation(self):
        sid = self.app.order[0]
        self.app.sessions[sid].messages.append({"role": "user", "content": "hello"})
        self.app._close_tab(sid)
        self.app._add_tab(sid)
        self.app.update()
        self.assertEqual(self.app.active, sid)
        self.assertEqual(len(self.app.sessions[sid].messages), 1)

    def test_asking_for_an_open_tab_switches_rather_than_duplicates(self):
        before = list(self.app.order)
        self.app._add_tab(before[-1])
        self.assertEqual(self.app.order, before)
        self.assertEqual(self.app.active, before[-1])

    def test_closing_every_tab_leaves_a_panel_that_can_still_speak(self):
        for sid in list(self.app.order):
            self.app._close_tab(sid)
        self.app.update()
        self.assertEqual(self.app.order, [])
        self.assertIsNone(self.app.active)
        self.assertIsNone(self.app.cur())
        self.assertTrue(self.app.empty.winfo_ismapped())
        self.assertEqual(str(self.app.btn_send.cget("state")), "disabled")
        # a host-level failure still has to reach the user (AGENTS.md)
        self.app._handle("error", None, "host-level-problem")
        self.app.update()
        self.assertIn("host-level-problem", self.app.empty_msg.cget("text"))
        self.app._on_send()          # must not raise with nothing open

    def test_a_chat_tab_is_ready_without_starting_a_bridge(self):
        """No MCP subprocess, no tools - but still warmed against its own
        prompt prefix, which is the whole reason the first reply is quick."""
        warmed = []

        class OneReply:
            def chat(self, messages, tools, max_tokens=None):
                warmed.append((messages, tools))
                return {"choices": [{"message": {"content": "ready"}}]}

        self.app.host_ready.set()
        real_llm, self.app.llm = self.app.llm, OneReply()
        try:
            s = self.app.sessions[eng.CHAT.id]
            self.app._boot_session(s)
        finally:
            self.app.llm = real_llm
        self.app._drain()              # the queue, on the UI thread, now
        self.app.update()
        self.assertTrue(s.ready)
        self.assertIsNone(s.mcp)
        self.assertEqual(s.tools, [])
        self.assertEqual(len(warmed), 1)
        messages, tools = warmed[0]
        self.assertEqual(tools, [])                    # nothing to offer
        self.assertEqual(messages[0]["content"], eng.CHAT.chat_prompt())
        self.assertEqual(s.bridge[0], "ok")
        self.assertNotIn("Bridge connected", s.view.get("1.0", "end"))

    def test_bridge_only_actions_say_why_they_do_nothing_in_chat(self):
        """Both are menu items, always enabled. Silence would read as a bug."""
        s = self.app.sessions[eng.CHAT.id]
        self.app._select(s.id)
        was_ready, s.ready = s.ready, True
        try:
            self.app._on_fix()
            self.app._capabilities()
        finally:
            s.ready = was_ready
        body = s.view.get("1.0", "end")
        self.assertIn("no app to start", body)
        self.assertIn("no capabilities", body)
        self.assertFalse(s.busy)

    def test_events_for_a_closed_tab_are_dropped(self):
        """A turn can still be in flight; it must not write into another app."""
        gone, other = self.app.order[-1], self.app.order[0]
        self.app._close_tab(gone)
        self.app._select(other)
        self.app._handle("sys", gone, "from-a-closed-tab")
        self.app.update()
        self.assertNotIn("from-a-closed-tab",
                         self.app.sessions[other].view.get("1.0", "end"))

    def test_open_tabs_are_remembered(self):
        sid = self.app.order[-1]
        self.app._close_tab(sid)
        self.assertEqual(self.app.prefs.get("tabs"), list(self.app.order))
        self.assertNotIn(sid, self.mod.Prefs(self.app.prefs.path).get("tabs"))

    def test_hiding_an_app_removes_its_row_and_sticks(self):
        rows = [a["name"] for a in self.app.detected]
        if not rows:
            self.skipTest("no creative apps on this machine")
        self.app._hide_app(rows[0])
        self.app.update()
        self.assertIn(rows[0], self.mod.Prefs(self.app.prefs.path).get("hidden"))
        self.assertNotIn(rows[0], self._sidebar_names())
        self.app._show_app(rows[0])
        self.assertIn(rows[0], self._sidebar_names())

    def test_pinning_moves_an_app_to_the_top(self):
        rows = [a["name"] for a in self.app.detected]
        if len(rows) < 2:
            self.skipTest("needs two creative apps")
        self.app._pin_app(rows[-1])
        self.app.update()
        self.assertEqual(self._sidebar_names()[0], self.mod.clip(rows[-1], self.mod.APP_NAME_CHARS))
        self.app._pin_app(rows[-1])          # same control unpins
        self.assertEqual(self.app.prefs.get("pinned"), [])

    def _sidebar_names(self):
        """The visible app list: each row's title, in the order it is drawn."""
        import tkinter
        out = []
        for row in self.app.applist.winfo_children():
            for box in row.winfo_children():
                if isinstance(box, tkinter.Frame):
                    labels = [w for w in box.winfo_children()
                              if isinstance(w, tkinter.Label)]
                    if labels:
                        out.append(labels[0].cget("text"))
                    break
        return out

    def _labels(self, menu):
        """Separators have no -label, so ask each entry what it is first."""
        return [menu.entrycget(i, "label") for i in range(menu.index("end") + 1)
                if menu.type(i) != "separator"]

    def test_the_new_tab_menu_offers_every_app_and_marks_the_open_ones(self):
        labels = self._labels(self.app._menu_tabs())
        self.assertEqual(len(labels), len(eng.TABS))
        for app in eng.TABS:
            self.assertTrue(any(app.name in l for l in labels), app.name)
        self.assertEqual(len([l for l in labels if "(open)" in l]),
                         len(self.app.order))

    def test_the_bridges_menu_names_every_bridge_whatever_is_open(self):
        """'What bridges exist' is a registry question, not a tab question."""
        self.app._close_tab(self.app.order[-1])
        labels = self._labels(self.app._menu_bridges())
        for app in eng.APPS:
            self.assertTrue(any(app.name in l for l in labels), app.name)
        self.assertTrue(any("no tab open" in l for l in labels))

    def test_the_add_menu_lists_what_was_hidden(self):
        rows = [a["name"] for a in self.app.detected]
        if not rows:
            self.skipTest("no creative apps on this machine")
        self.assertIn("Nothing is hidden", self._labels(self.app._menu_hidden()))
        self.app._hide_app(rows[0])
        self.assertIn(rows[0], self._labels(self.app._menu_hidden()))
        self.app._show_app(rows[0])

    def test_theme_switch_repaints_the_live_window(self):
        self.app._theme("light")
        self.app.update()
        self.assertEqual(self.app.cget("bg"), self.mod.LIGHT["bg"])
        self.assertEqual(self.app.sessions[self.app.order[0]].view.cget("bg"),
                         self.mod.LIGHT["bg"])
        self.assertEqual(self.app.prefs.get("theme"), "light")
        self.app._theme("dark")
        self.app.update()
        self.assertEqual(self.app.cget("bg"), self.mod.DARK["bg"])
        self.assertEqual(self.app.lbl_status.cget("bg"), self.mod.DARK["head"])

    def test_a_tools_window_open_across_a_switch_is_repainted(self):
        sid = eng.APPS[0].id
        self.app._tools_window(sid)
        self.app._theme("light")
        self.app.update()
        view = self.app.tool_views[sid]
        self.assertEqual(view.cget("bg"), self.mod.LIGHT["bg"])
        self.assertEqual(view.tag_cget("group", "foreground"),
                         self.mod.LIGHT["accent"])
        self.app._theme("dark")
        self.app.windows[("tools", sid)].destroy()
        self.app.update()

    def test_a_theme_switch_keeps_the_transcripts(self):
        sid = self.app.order[0]
        self.app._handle("sys", sid, "survives-a-repaint")
        self.app._theme("light")
        self.app._theme("dark")
        self.app.update()
        self.assertIn("survives-a-repaint",
                      self.app.sessions[sid].view.get("1.0", "end"))

    def test_status_colours_are_roles_not_hex(self):
        """Anything that puts a colour on the queue has to survive a theme
        switch, which means naming a palette role rather than a literal."""
        for s in self.app.sessions.values():
            with self.subTest(app=s.id):
                self.assertIn(s.status[1], self.mod.DARK)
                self.assertIn(s.bridge[0], self.mod.DARK)

    def test_no_method_shadows_tkinter_internals(self):
        import tkinter
        clashes = [n for n in vars(self.mod.Chat)
                   if not n.startswith("__")
                   and (hasattr(tkinter.Misc, n) or hasattr(tkinter.Tk, n))]
        self.assertEqual(clashes, [], "shadowing Tk internals breaks the widget")

    def test_stop_button_sets_cancellation_without_touching_composer(self):
        s = self.app.cur()
        original_status = s.status
        try:
            s.busy = True
            s.cancel.clear()
            self.app.input.delete("1.0", "end")
            self.app.input.insert("1.0", "next request")
            self.app._apply_status()
            self.assertEqual(self.app.btn_send.cget("text"), "Stop")
            self.app._on_send()
            self.assertTrue(s.cancel.is_set())
            self.assertEqual(self.app.input.get("1.0", "end").strip(), "next request")
            self.assertEqual(self.app.btn_send.cget("text"), "Stopping…")
        finally:
            s.busy = False
            s.cancel.clear()
            s.status = original_status
            self.app.input.delete("1.0", "end")
            self.app._apply_status()

    def test_old_worker_cannot_write_to_reopened_tab(self):
        s = self.app.cur()
        old_event_id, app_id = s.event_id, s.id
        self.app._close_tab(app_id)
        self.app._add_tab(app_id)
        replacement = self.app.cur()
        self.app._handle("token", old_event_id, "stale-generation-token")
        self.assertNotIn("stale-generation-token", replacement.view.get("1.0", "end"))

    def test_preview_image_remains_owned_by_session(self):
        import base64
        s = self.app.cur()
        png = icons.png(bytes([255, 0, 0, 255]) * 4, 2, 2)
        count = len(s.preview_images)
        self.app._handle("preview", s.event_id, {"type": "image", "mimeType": "image/png",
                                               "data": base64.b64encode(png).decode()})
        self.assertEqual(len(s.preview_images), count + 1)
        self.assertTrue(s.view.image_names())

    def test_gui_executor_saves_completed_task(self):
        from test_tasks import FakeLLM, answer
        s = self.app.cur()
        original_llm = self.app.llm
        original_messages, original_record = s.messages, s.record
        try:
            s.reset()
            s.record.briefs.append("Explain frame rate")
            s.messages.append({"role": "user", "content": "Explain frame rate"})
            self.app.llm = FakeLLM([answer(text="Frames per second.")])
            self.app._turn(s)
            self.app.update()
            with open(self.app._task_path(s), encoding="utf-8") as f:
                saved = json.load(f)
            self.assertEqual(saved["record"]["app_id"], s.id)
            self.assertEqual(saved["messages"][-1]["content"], "Frames per second.")
        finally:
            self.app.llm = original_llm
            s.messages, s.record = original_messages, original_record

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
