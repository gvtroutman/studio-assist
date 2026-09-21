"""What the app learns and asks: the notebook, the reflection, studio_ask,
studio_remember, the research sidecar on every tab, and the studio brief.
Fake inference and bridges; no creative apps, no network, no model."""
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import Mock

import studio_agent as eng
import studio_lessons as lessons
import studio_tasks as tasks
from test_tasks import FakeLLM, answer, call, spec


class TestNotebook(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.book = lessons.Notebook.for_app("resolve", self.dir)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_lessons_survive_a_reload_and_are_deduplicated(self):
        self.book.add("Music goes on A3.", "user")
        self.book.add("music goes on A3", "model")      # the same, said again
        again = lessons.Notebook.for_app("resolve", self.dir)
        self.assertIsNone(again.load())
        self.assertEqual([l["text"] for l in again.lessons], ["Music goes on A3."])
        self.assertEqual(again.lessons[0]["hits"], 1)
        self.assertEqual(again.lessons[0]["source"], "user")   # the higher source wins

    def test_a_wrecked_file_costs_the_lessons_never_the_app(self):
        os.makedirs(os.path.dirname(self.book.path))
        with open(self.book.path, "w") as f:
            f.write("{not json")
        self.assertIn("resolve.json", self.book.load())
        self.assertEqual(self.book.lessons, [])
        self.assertEqual(self.book.brief(), "")

    def test_a_full_notebook_drops_the_least_valuable_first(self):
        for i in range(lessons.MAX_LESSONS):
            self.book.add("Calling timeline: refusal number %d" % i, "error")
        self.book.add("The user likes sentence case.", "user")
        self.book.add("Calling timeline: one more refusal", "error")
        texts = [l["text"] for l in self.book.lessons]
        self.assertEqual(len(texts), lessons.MAX_LESSONS)
        self.assertIn("The user likes sentence case.", texts)
        self.assertIn("Calling timeline: one more refusal", texts)     # never the newest
        self.assertNotIn("Calling timeline: refusal number 0", texts)  # the oldest error

    def test_brief_carries_everything_and_fresh_only_what_came_after(self):
        self.book.add("First lesson kept.", "user")
        brief = self.book.brief()
        self.assertEqual(brief, "- First lesson kept.")
        self.assertEqual(self.book.fresh(), "")
        self.book.add("Second lesson, mid-session.", "model")
        self.assertEqual(self.book.fresh(), "- Second lesson, mid-session.")
        self.assertIn("Second lesson", self.book.brief())          # the next boot has both
        self.assertEqual(self.book.fresh(), "")

    def test_forgetting_removes_from_disk_too(self):
        self.book.add("Keep this for a moment.", "user")
        self.assertTrue(self.book.remove("keep this for a moment"))
        self.assertFalse(self.book.remove("keep this for a moment"))
        again = lessons.Notebook.for_app("resolve", self.dir)
        again.load()
        self.assertEqual(again.lessons, [])

    def test_refusals_become_lessons_a_few_per_run(self):
        refusals = [("timeline", "unsupported action 'list'; supported actions: get_items"),
                    ("timeline", "unsupported action 'list'; supported actions: get_items"),
                    ("folder", "unknown key 'path_'; did you mean path?"),
                    ("media_pool", "x"), ("render", "y"), ("graph", "z")]
        added = self.book.learn_refusals(refusals)
        self.assertEqual([l["text"] for l in added],
                         ["Calling timeline: unsupported action 'list'; supported actions: get_items",
                          "Calling folder: unknown key 'path_'; did you mean path?"])
        self.assertTrue(all(l["source"] == "error" for l in added))
        self.assertEqual(len(self.book.lessons), 2)

    def test_the_rendered_block_is_bounded_and_keeps_the_newest(self):
        for i in range(60):
            self.book.add("Lesson %02d " % i + "x" * 250, "user")
        brief = self.book.brief()
        self.assertLessEqual(len(brief), lessons.BRIEF_CHARS)
        self.assertIn("Lesson %02d" % (lessons.MAX_LESSONS + 19), brief)   # the last added


class TestReadingTheUser(unittest.TestCase):
    def test_a_stated_rule_is_kept_as_the_user_put_it(self):
        self.assertEqual(lessons.explicit_lesson("Remember that music always goes on A3."),
                         "music always goes on A3")
        self.assertEqual(lessons.explicit_lesson("from now on, use 25 fps for UK deliveries"),
                         "use 25 fps for UK deliveries")
        self.assertIsNone(lessons.explicit_lesson("Remember"))
        self.assertIsNone(lessons.explicit_lesson("Make a title card"))

    def test_a_correction_is_recognised_a_request_is_not(self):
        for text in ("No, I meant the second clip", "That's not what I asked for",
                     "actually put it on V2", "Don't touch the music track", "wrong track"):
            self.assertTrue(lessons.looks_like_correction(text), text)
        for text in ("Make a title card", "Nothing fancy, a plain cut", "How long is the edit?"):
            self.assertFalse(lessons.looks_like_correction(text), text)

    def test_reflection_reply_is_read_strictly(self):
        self.assertEqual(lessons.parse_reflection("Lesson: check the track count before appending."),
                         "check the track count before appending.")
        self.assertEqual(lessons.parse_reflection('Sure. Lesson: "Ask which sequence first."'),
                         "Ask which sequence first.")
        self.assertIsNone(lessons.parse_reflection("NONE"))
        self.assertIsNone(lessons.parse_reflection("none - nothing to keep"))
        self.assertIsNone(lessons.parse_reflection("I did the task well."))   # no Lesson: marker
        self.assertIsNone(lessons.parse_reflection(""))

    def test_reflection_asks_once_with_whole_exchanges_and_no_tools(self):
        llm = FakeLLM([answer(text="Lesson: read the timeline before appending.")])
        llm.chat = lambda messages, tools, max_tokens=None: (
            llm.requests.append((messages, tools, max_tokens)) or
            {"choices": [{"message": answer(text="Lesson: read the timeline before appending.")}]})
        messages = [{"role": "system", "content": "rules"},
                    {"role": "user", "content": "append the clip"},
                    answer(call("timeline", {"action": "append"})),
                    {"role": "tool", "tool_call_id": "c1", "content": "TOOL ERROR: no track"},
                    answer(text="It failed.")]
        self.assertEqual(lessons.reflect(llm, "DaVinci Resolve", messages),
                         "read the timeline before appending.")
        sent, tools, max_tokens = llm.requests[0]
        self.assertIsNone(tools)
        self.assertTrue(max_tokens)
        self.assertEqual(sent[0], messages[0])
        self.assertEqual(sent[1:-1], messages[1:])
        self.assertIn("DaVinci Resolve", sent[-1]["content"])
        self.assertEqual(len(messages), 5)                    # nothing appended

    def test_reflection_never_starts_on_a_tool_reply(self):
        llm = Mock()
        llm.chat.return_value = {"choices": [{"message": answer(text="NONE")}]}
        messages = [{"role": "system", "content": "rules"}]
        for i in range(40):
            messages.append(answer(call("get_comp", ident="c%d" % i), text="x" * 900))
            messages.append({"role": "tool", "tool_call_id": "c%d" % i, "content": "y" * 900})
        self.assertIsNone(lessons.reflect(llm, "AE", messages, max_chars=6000))
        sent = llm.chat.call_args[0][0]
        self.assertEqual(sent[1]["role"], "assistant")
        self.assertLess(len(sent), len(messages))


class TestExecutorLearnsAndAsks(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.book = lessons.Notebook.for_app("after-effects", self.dir)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def run_with(self, responses, specs=None, **kw):
        specs = specs or [spec("create_comp"), spec("get_comp")]
        self.messages = [{"role": "system", "content": "rules"},
                         {"role": "user", "content": "Make a title"}]
        self.llm = FakeLLM(responses)
        self.bridge = Mock()
        self.bridge.call_tool.return_value = {"content": [{"type": "text", "text": '{"id": 12}'}]}
        self.events = []
        self.executor = tasks.Executor(self.llm, self.bridge, eng.to_openai_tools(specs),
            schemas=specs, emit=lambda k, p: self.events.append((k, p)),
            notebook=self.book, **kw)
        return self.executor

    def test_the_internal_tools_include_ask_and_remember(self):
        ex = self.run_with([answer(text="hi")])
        names = [t["function"]["name"] for t in ex.tools]
        self.assertEqual(names, ["create_comp", "get_comp", "studio_task_update",
                                 "studio_tool_create", "studio_ask", "studio_remember"])

    def test_a_question_ends_the_run_and_the_answer_continues_it(self):
        asked = {"question": "Which frame rate?", "options": [
            {"label": "24 fps", "description": "film look"}, {"label": "25 fps"}]}
        ex = self.run_with([answer(call("studio_ask", asked)), answer(text="24 it is.")])
        self.assertEqual(ex.run(self.messages), "Which frame rate?")
        self.assertEqual(ex.asked["options"][1], {"label": "25 fps"})
        self.assertFalse(ex.asked["multiple"])
        self.assertEqual([p for k, p in self.events if k == "ask"], [ex.asked])
        self.assertIn("waiting for the user's answer", ex.record.status)
        self.assertTrue(ex.record.status.startswith("response complete"))
        self.bridge.call_tool.assert_not_called()
        self.assertIn("next message", self.messages[-1]["content"])   # the tool reply
        self.assertEqual(len(self.llm.requests), 1)                   # no second turn
        # The answer is an ordinary next message.
        self.messages.append({"role": "user", "content": "24 fps"})
        self.assertEqual(ex.run(self.messages), "24 it is.")

    def test_a_question_needs_two_to_five_choices(self):
        ex = self.run_with([answer(call("studio_ask", {"question": "Which?", "options": [{"label": "a"}]})),
                            answer(text="ok")])
        ex.run(self.messages)
        self.assertIsNone(ex.asked)
        self.assertIn("TOOL ERROR", self.messages[3]["content"])
        self.assertEqual([k for k, _ in self.events if k == "ask"], [])

    def test_remember_writes_the_notebook_and_rides_the_tail_until_the_next_boot(self):
        self.messages_prompt_lessons = self.book.brief()          # nothing yet: carried = {}
        ex = self.run_with([answer(call("studio_remember", {"lesson": "Titles are sentence case here."})),
                            answer(text="Noted.")])
        self.assertEqual(ex.run(self.messages), "Noted.")
        self.assertEqual([l["text"] for l in self.book.lessons], ["Titles are sentence case here."])
        self.assertIn("Kept for future tasks", self.messages[3]["content"])
        self.assertIn(("sys", "Remembered: Titles are sentence case here."), self.events)
        # The first request had no lesson; the second carries it last, after
        # the task record, and the system prompt is untouched.
        first, second = self.llm.requests
        self.assertNotIn("sentence case", json.dumps(first))
        self.assertIn("LESSONS FROM EARLIER WORK", second[-1]["content"])
        self.assertIn("Titles are sentence case here.", second[-1]["content"])
        self.assertTrue(second[-2]["content"].startswith("Saved task context"))
        self.assertEqual(second[0], first[0])
        self.assertEqual(self.messages[0]["content"], "rules")
        # Booted again, it is in the prompt, and no longer fresh.
        self.assertIn("sentence case", eng.APPS_BY_ID["after-effects"].chat_prompt(lessons=self.book.brief()))
        self.assertEqual(self.book.fresh(), "")

    def test_a_tab_without_a_notebook_refuses_to_remember(self):
        ex = self.run_with([answer(call("studio_remember", {"lesson": "Something worth keeping."})),
                            answer(text="ok")])
        ex.notebook = None
        ex.run(self.messages)
        self.assertIn("keeps no notebook", self.messages[3]["content"])

    def test_validator_refusals_are_collected_and_flag_trouble(self):
        specs = [spec("create_comp", {"type": "object", "additionalProperties": False,
                                      "properties": {"name": {"type": "string"}}})]
        ex = self.run_with([answer(call("create_comp", {"nmae": "x"})), answer(text="gave up")],
                           specs=specs)
        ex.run(self.messages)
        self.assertEqual(len(ex.refusals), 1)
        self.assertEqual(ex.refusals[0][0], "create_comp")
        self.assertIn("nmae", ex.refusals[0][1])
        self.assertTrue(ex.trouble)
        self.bridge.call_tool.assert_not_called()

    def test_a_clean_run_has_no_trouble_and_teaches_nothing(self):
        ex = self.run_with([answer(call("get_comp")), answer(text="fine")])
        ex.run(self.messages)
        self.assertFalse(ex.trouble)
        self.assertEqual(ex.refusals, [])
        llm = Mock()
        kept = eng.learn_from_run(ex, self.messages, self.book, llm, "After Effects")
        self.assertEqual(kept, [])
        llm.chat.assert_not_called()                              # no reflection

    def test_a_troubled_run_reflects_once_and_keeps_the_lesson(self):
        specs = [spec("create_comp", {"type": "object", "additionalProperties": False,
                                      "properties": {"name": {"type": "string"}}})]
        ex = self.run_with([answer(call("create_comp", {"nmae": "x"})), answer(text="gave up")],
                           specs=specs)
        ex.run(self.messages)
        llm = Mock()
        llm.chat.return_value = {"choices": [{"message": answer(text="Lesson: create_comp takes name, not nmae.")}]}
        kept = eng.learn_from_run(ex, self.messages, self.book, llm, "After Effects")
        self.assertEqual(len(kept), 2)                           # the refusal, the reflection
        self.assertTrue(kept[0].startswith("Calling create_comp:"))
        self.assertEqual(kept[1], "create_comp takes name, not nmae.")
        self.assertEqual(llm.chat.call_count, 1)
        sources = {l["source"] for l in self.book.lessons}
        self.assertEqual(sources, {"error", "review"})

    def test_a_correction_from_the_user_reflects_even_when_no_call_failed(self):
        ex = self.run_with([answer(text="Moved it.")])
        self.messages[1]["content"] = "No, I meant the clip on V2"
        ex.record.briefs.append(self.messages[1]["content"])
        ex.run(self.messages)
        llm = Mock()
        llm.chat.return_value = {"choices": [{"message": answer(text="NONE")}]}
        self.assertEqual(eng.learn_from_run(ex, self.messages, self.book, llm, "Resolve"), [])
        self.assertEqual(llm.chat.call_count, 1)

    def test_a_stated_rule_is_kept_without_asking_the_model(self):
        ex = self.run_with([answer(text="Understood.")])
        ex.record.briefs.append("Remember that music always goes on A3")
        ex.run(self.messages)
        llm = Mock()
        kept = eng.learn_from_run(ex, self.messages, self.book, llm, "Resolve")
        self.assertEqual(kept, ["music always goes on A3"])
        self.assertEqual(self.book.lessons[0]["source"], "user")
        llm.chat.assert_not_called()

    def test_learning_never_costs_the_result(self):
        ex = self.run_with([answer(text="done")])
        ex.run(self.messages)
        ex.trouble = True
        llm = Mock()
        llm.chat.side_effect = ConnectionError("host away")
        self.assertEqual(eng.learn_from_run(ex, self.messages, self.book, llm, "AE"), [])


class TestResearchSidecar(unittest.TestCase):
    def test_every_app_tab_has_the_sidecar_and_chat_is_the_sidecar(self):
        for app in eng.APPS:
            self.assertTrue(app.research, app.id)
        self.assertFalse(eng.CHAT.research)
        self.assertEqual(eng.RESEARCH_TOOL_NAMES,
                         {"list_folder", "find_files", "read_file", "search_web", "fetch_page"})

    def test_the_router_sends_each_name_to_its_owner(self):
        bridge, sidecar = Mock(), Mock()
        bridge.instructions = "the bridge's"
        router = eng.Router(bridge, sidecar)
        router.call_tool("fetch_page", {"url": "x"})
        router.call_tool("create_comp", {"name": "x"})
        sidecar.call_tool.assert_called_once_with("fetch_page", {"url": "x"})
        bridge.call_tool.assert_called_once_with("create_comp", {"name": "x"})
        self.assertEqual(router.instructions, "the bridge's")
        router.close()
        bridge.close.assert_called_once()
        sidecar.close.assert_not_called()

    def test_the_sidecar_is_the_real_research_server_in_process(self):
        client = eng.research_client()
        self.assertEqual({t["name"] for t in client.list_tools()}, set(eng.RESEARCH_TOOL_NAMES))
        for tool in client.list_tools():
            self.assertTrue(tool.get("annotations", {}).get("readOnlyHint"), tool["name"])

    def test_a_page_or_a_file_never_verifies_an_edit(self):
        for name in eng.RESEARCH_TOOL_NAMES:
            self.assertFalse(tasks.verification_read(name, {}), name)
        self.assertTrue(tasks.verification_read("get_comp", {}))

    def test_every_app_names_documentation_the_sidecar_can_read(self):
        for app in eng.APPS:
            with self.subTest(app=app.id):
                self.assertTrue(app.docs, "no docs")
                for title, url in app.docs:
                    self.assertTrue(url.startswith("https://"))
                    # helpx.adobe.com answers the bridge with 403: search only.
                    self.assertNotIn("helpx.adobe.com", url)
                    self.assertIn(url, app.chat_prompt())
                self.assertIn("LOOKING THINGS UP", app.chat_prompt())
                for tool in eng.RESEARCH_TOOL_NAMES:
                    self.assertIn(tool, app.chat_prompt())

    def test_media_apps_carry_editing_craft_and_the_rest_their_own(self):
        by_id = eng.APPS_BY_ID
        for app_id in ("resolve", "premiere"):
            self.assertIn("HOW AN EDIT IS CUT", by_id[app_id].chat_prompt())
            self.assertIn("J-cut", by_id[app_id].chat_prompt())
        self.assertIn("HOW MOTION WORK IS BUILT", by_id["after-effects"].chat_prompt())
        self.assertIn("HOW DESIGN WORK IS BUILT", by_id["photoshop"].chat_prompt())
        self.assertIn("HOW IMAGES ARE MADE", by_id["comfyui"].chat_prompt())
        for app in eng.TABS:
            self.assertIn("CREATIVE WORK", app.chat_prompt())
            self.assertIn("studio_ask", app.chat_prompt())


class TestStudioBrief(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_no_file_means_no_section(self):
        path = eng.studio_brief_path(self.dir)
        self.assertEqual(path, os.path.join(self.dir, "studio.md"))
        self.assertEqual(eng.read_studio_brief(path), "")
        self.assertEqual(eng.studio_section(""), "")
        self.assertNotIn("ABOUT THIS STUDIO", eng.APPS[0].chat_prompt())

    def test_the_brief_and_the_lessons_come_last_in_that_order(self):
        path = eng.studio_brief_path(self.dir)
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Method & Form\nWe cut brand films at 25 fps.\n")
        text = eng.read_studio_brief(path)
        for app in eng.TABS:
            prompt = app.chat_prompt(studio=text, lessons="- Music goes on A3.")
            self.assertTrue(prompt.startswith(app.system_prompt))
            about, kept = prompt.index("ABOUT THIS STUDIO"), prompt.index("LESSONS FROM EARLIER WORK")
            self.assertLess(prompt.index("CREATIVE WORK"), about)
            self.assertLess(about, kept)
            self.assertIn("25 fps", prompt)
            self.assertIn("Music goes on A3", prompt)
            self.assertEqual(prompt, app.chat_prompt(text, "- Music goes on A3."))

    def test_a_long_brief_is_clipped_and_says_so(self):
        section = eng.studio_section("x" * (eng.STUDIO_BRIEF_CHARS + 100))
        self.assertIn("longer", section)
        self.assertLess(len(section), eng.STUDIO_BRIEF_CHARS + 200)

    def test_the_template_never_reaches_the_model(self):
        self.assertIn("## Who", eng.STUDIO_TEMPLATE)
        self.assertNotIn("## Who", eng.APPS[0].chat_prompt())


class TestTerminalAnswers(unittest.TestCase):
    asked = {"question": "Which?", "options": [{"label": "A"}, {"label": "B"}, {"label": "C"}],
             "multiple": False}

    def test_a_number_is_its_label_and_text_is_itself(self):
        self.assertEqual(eng.answer_text(self.asked, "2"), "B")
        self.assertEqual(eng.answer_text(self.asked, "neither, use D"), "neither, use D")
        self.assertEqual(eng.answer_text(self.asked, "9"), "9")
        self.assertEqual(eng.answer_text(dict(self.asked, multiple=True), "1, 3"), "A; C")
        self.assertEqual(eng.answer_text(self.asked, "1, 3"), "1, 3")   # not multiple

    def test_the_console_form_prints_choices_and_reads_one(self):
        import io
        from contextlib import redirect_stdout
        out = io.StringIO()
        with redirect_stdout(out):
            reply = eng.ask_at_terminal(self.asked, answer=lambda prompt: "3")
        self.assertEqual(reply, "C")
        self.assertIn("1. A", out.getvalue())
        self.assertIn("Which?", out.getvalue())


class TestGuiForms(unittest.TestCase):
    """The question form, the studio editor and the lessons window, driven
    in-process - never with synthetic keystrokes (see AGENTS.md)."""

    @classmethod
    def setUpClass(cls):
        import studio_chat
        cls.mod = studio_chat
        cls.dir = tempfile.mkdtemp()
        cls._real_settings = os.environ.get("STUDIO_SETTINGS")
        os.environ["STUDIO_SETTINGS"] = os.path.join(cls.dir, "settings.json")
        cls._real_installed = eng.installed_apps
        eng.installed_apps = lambda: list(eng.APPS)
        studio_chat.Chat._boot_host = lambda self: None
        studio_chat.Chat._ensure = lambda self, s: None
        studio_chat.Chat._read_icons = lambda self: None
        cls._real_fit = (eng.loaded_instances, eng.fit_model)
        eng.loaded_instances = lambda *a, **k: [("m", 8192)]
        eng.fit_model = lambda *a, **k: (8192, "")
        try:
            cls.app = studio_chat.Chat()
        except Exception as e:                # no display
            raise unittest.SkipTest("no display: %s" % e)
        for _ in range(15):
            cls.app.update()

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "app", None) is not None:
            cls.app.destroy()
        eng.installed_apps = cls._real_installed
        eng.loaded_instances, eng.fit_model = cls._real_fit
        if cls._real_settings is None:
            os.environ.pop("STUDIO_SETTINGS", None)
        else:
            os.environ["STUDIO_SETTINGS"] = cls._real_settings
        shutil.rmtree(cls.dir, ignore_errors=True)

    def setUp(self):
        self.app._add_tab("after-effects")
        self.app._select("after-effects")
        self.s = self.app.sessions["after-effects"]
        self.s.busy = False
        self.app._on_new()                # a fresh conversation every time
        self.s.ready = True
        self.sent = []
        self._real_spawn = self.app._spawn
        self.app._spawn = lambda sid, fn, *a: self.sent.append(a)
        self.app.update()

    def tearDown(self):
        self.app._spawn = self._real_spawn
        self.s.busy = False

    def test_every_session_starts_with_its_own_prompt_and_notebook(self):
        self.assertIsNotNone(self.s.notebook)
        self.assertEqual(self.s.messages[0]["content"], self.s.prompt())
        self.assertIsNotNone(self.app.sessions["chat"].notebook)   # chat keeps one too

    def test_a_click_on_a_choice_is_the_next_message(self):
        asked = {"question": "Which frame rate?", "multiple": False,
                 "options": [{"label": "24 fps", "description": "film"}, {"label": "25 fps"}]}
        self.app._handle("ask", self.s.event_id, asked)
        self.app.update()
        self.assertEqual(len(self.s.ask_buttons), 3)              # two choices and "something else"
        self.assertTrue(all(b.winfo_exists() for b in self.s.ask_buttons))
        self.s.ask_buttons[1].invoke()
        self.assertEqual(self.s.messages[-1], {"role": "user", "content": "25 fps"})
        self.assertEqual(self.s.record.briefs[-1], "25 fps")
        self.assertEqual(len(self.sent), 1)
        self.assertTrue(self.s.busy)
        self.assertEqual(self.s.ask_buttons, [])                  # settled
        self.assertIn("25 fps", self.s.view.get("1.0", "end"))

    def test_ticked_choices_go_together_and_nothing_ticked_sends_nothing(self):
        asked = {"question": "Which platforms?", "multiple": True,
                 "options": [{"label": "Instagram"}, {"label": "YouTube"}, {"label": "LinkedIn"}]}
        self.app._handle("ask", self.s.event_id, asked)
        self.app.update()
        boxes, send = self.s.ask_buttons[:3], self.s.ask_buttons[3]
        send.invoke()
        self.assertEqual(self.sent, [])
        boxes[0].invoke()
        boxes[2].invoke()
        send.invoke()
        self.assertEqual(self.s.messages[-1]["content"], "Instagram; LinkedIn")
        self.assertEqual(len(self.sent), 1)

    def test_a_typed_message_settles_an_open_question(self):
        asked = {"question": "Which?", "multiple": False,
                 "options": [{"label": "A"}, {"label": "B"}]}
        self.app._handle("ask", self.s.event_id, asked)
        buttons = list(self.s.ask_buttons)
        self.app.input.insert("1.0", "neither, use C")
        self.app._on_send()
        self.assertEqual(self.s.messages[-1]["content"], "neither, use C")
        self.assertEqual(self.s.ask_buttons, [])
        self.assertEqual(str(buttons[0].cget("state")), "disabled")
        buttons[0].invoke()                                       # a dead form sends nothing
        self.assertEqual(len(self.sent), 1)

    def test_the_studio_brief_is_saved_and_reaches_idle_tabs_now(self):
        self.app._studio_window()
        editor = self.app.studio_editor
        self.assertIn("## Who", editor.get("1.0", "end"))        # seeded with the template
        editor.delete("1.0", "end")
        editor.insert("1.0", "# Method & Form\nBrand films, 25 fps, sentence case.")
        win = self.app.windows["studio"]
        for child in win.winfo_children():
            if isinstance(child, self.mod.tk.Button):
                child.invoke()
        with open(self.app._studio_path(), encoding="utf-8") as f:
            self.assertIn("sentence case", f.read())
        self.assertIn("sentence case", self.app.studio)
        self.assertIn("ABOUT THIS STUDIO", self.s.messages[0]["content"])
        self.assertIn("sentence case", self.s.messages[0]["content"])
        # A tab mid-conversation keeps its prompt until New chat.
        self.s.messages.append({"role": "user", "content": "hello"})
        editor.delete("1.0", "end")
        editor.insert("1.0", "# Changed\nNow 24 fps.")
        for child in win.winfo_children():
            if isinstance(child, self.mod.tk.Button):
                child.invoke()
        self.assertNotIn("24 fps", self.s.messages[0]["content"])
        self.app._on_new()
        self.assertIn("24 fps", self.s.messages[0]["content"])
        win.destroy()

    def test_the_lessons_window_lists_and_forgets(self):
        self.s.notebook.add("Titles are sentence case here.", "user")
        self.app._lessons_window()
        text = self.app.lessons_view.get("1.0", "end")
        self.assertIn("Titles are sentence case here.", text)
        self.assertIn("you said so", text)
        win = self.app.windows[("lessons", "after-effects")]
        buttons = [w for w in self.app.lessons_view.winfo_children()
                   if isinstance(w, self.mod.tk.Button)]
        self.assertEqual(len(buttons), 1)
        buttons[0].invoke()
        self.assertEqual(self.s.notebook.lessons, [])
        self.assertIn("Nothing kept yet", self.app.lessons_view.get("1.0", "end"))
        self.app.windows[("lessons", "after-effects")].destroy()


if __name__ == "__main__":
    unittest.main()
