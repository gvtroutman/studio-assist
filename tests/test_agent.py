"""
Offline tests. No network, no creative apps, no model.

    python -m unittest discover -s tests -v

Anything needing a display skips itself when there isn't one.
"""

import io
import json
import os
import re
import shutil
import struct
import sys
import tempfile
import tkinter as tk
import unittest
import urllib.error

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

    def test_the_apps_own_helpers_in_vram_are_not_the_users_choice(self):
        """After the LLM PC restarts, the first thing in VRAM is whatever
        helper got loaded first - the vision model, or a draft. Every tab
        then ran on qwen2.5-vl-7b-instruct, which answers a tool call with
        advice to open File Explorer."""
        ids = ["qwen2.5-vl-7b-instruct", "qwen3-1.7b", eng.DEFAULT_MODEL, "gpt-oss-20b"]
        self.assertEqual(eng.pick_model(["qwen2.5-vl-7b-instruct"], ids), eng.DEFAULT_MODEL)
        self.assertEqual(eng.pick_model(["qwen3-1.7b", "qwen2.5-vl-7b-instruct"], ids),
                         eng.DEFAULT_MODEL)
        # the default in VRAM wins over a helper listed before it
        self.assertEqual(eng.pick_model(["qwen2.5-vl-7b-instruct", eng.DEFAULT_MODEL], ids),
                         eng.DEFAULT_MODEL)
        # a model loaded by hand is still the one used when the default is not loaded
        self.assertEqual(eng.pick_model(["qwen2.5-vl-7b-instruct", "gpt-oss-20b"], ids),
                         "gpt-oss-20b")
        # and STUDIO_MODEL still beats everything
        self.assertEqual(eng.pick_model([eng.DEFAULT_MODEL], ids, want="gpt-oss-20b"),
                         "gpt-oss-20b")

    def test_the_probe_reports_every_loaded_model(self):
        import urllib.request

        class Resp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def fake_open(url, timeout=None):
            return Resp(json.dumps({"data": [
                {"id": "qwen2.5-vl-7b-instruct", "type": "vlm", "state": "loaded"},
                {"id": "big-30b", "type": "llm", "state": "loaded"},
                {"id": "other", "type": "llm", "state": "not-loaded"}]}).encode("utf-8"))
        real = urllib.request.urlopen
        urllib.request.urlopen = fake_open
        self.addCleanup(setattr, urllib.request, "urlopen", real)
        ok, loaded, ids, vision, err = eng.probe_models("http://h:1234/v1")
        self.assertTrue(ok)
        self.assertEqual(loaded, ["qwen2.5-vl-7b-instruct", "big-30b"])
        self.assertEqual(vision, ["qwen2.5-vl-7b-instruct"])
        # the vision picker still prefers one already in VRAM, from the list
        self.assertEqual(eng.pick_vision_model(["other-vl", "qwen2.5-vl-7b-instruct"], "big-30b",
                                               loaded=loaded), "qwen2.5-vl-7b-instruct")


class TestVisionChoice(unittest.TestCase):
    """Every bridge answers a screenshot with a picture and the executing model
    reads text, so a vision model is resolved from the host like the executing
    one is - never left to an environment variable nobody set."""

    def test_pin_wins_when_served(self):
        self.assertEqual(eng.pick_vision_model(["a-vl", "b-vl"], "coder", want="b-vl"), "b-vl")

    def test_executing_model_sees_for_itself(self):
        # No second model in VRAM when the one already there can look.
        self.assertEqual(eng.pick_vision_model(["qwen3-vl-8b-instruct", "gemma-3-4b-it"],
                                               "gemma-3-4b-it"), "gemma-3-4b-it")

    def test_prefers_known_good_then_anything(self):
        good = eng.PREFERRED_VISION_MODELS[0]
        self.assertEqual(eng.pick_vision_model(["odd-vl", good], "coder"), good)
        self.assertEqual(eng.pick_vision_model(["odd-vl"], "coder"), "odd-vl")

    def test_one_already_in_vram_beats_a_load(self):
        good = eng.PREFERRED_VISION_MODELS[0]
        self.assertEqual(eng.pick_vision_model(["odd-vl", good], "coder", loaded="odd-vl"),
                         "odd-vl")

    def test_resolve_says_when_a_load_is_needed(self):
        real = os.environ.pop("STUDIO_VISION_MODEL", None)
        try:
            v, _ = eng.resolve_vision("http://h/v1", "coder", ["odd-vl"], loaded="coder")
            self.assertTrue(v.needs_load)
            v, _ = eng.resolve_vision("http://h/v1", "coder", ["odd-vl"], loaded="odd-vl")
            self.assertFalse(v.needs_load)
            v, _ = eng.resolve_vision("http://h/v1", "odd-vl", ["odd-vl"], loaded=None)
            self.assertFalse(v.needs_load)   # the executing model sees for itself
        finally:
            if real:
                os.environ["STUDIO_VISION_MODEL"] = real

    def test_load_model_posts_to_lm_studio_and_reports_failure(self):
        import urllib.request
        import urllib.error
        import io
        calls = []

        class Resp(io.BytesIO):
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False

        def fake_open(req, timeout=None):
            calls.append((req.full_url, json.loads(req.data)))
            if "bad" in req.data.decode():
                raise urllib.error.HTTPError(req.full_url, 404, "nope", {}, io.BytesIO(b"no such model"))
            return Resp(b'{"status": "loaded"}')

        real = urllib.request.urlopen
        urllib.request.urlopen = fake_open
        try:
            self.assertIsNone(eng.load_model("http://h:1234/v1", "odd-vl"))
            self.assertEqual(calls[0], ("http://h:1234/api/v1/models/load", {"model": "odd-vl"}))
            err = eng.load_model("http://h:1234/v1", "bad-vl")
            self.assertIn("404", err)
            self.assertIn("no such model", err)
        finally:
            urllib.request.urlopen = real

    def test_unserved_pin_falls_through(self):
        self.assertEqual(eng.pick_vision_model(["odd-vl"], "coder", want="nope"), "odd-vl")

    def test_nothing_served_is_none(self):
        self.assertIsNone(eng.pick_vision_model([], "coder"))

    def test_resolve_explains_absence(self):
        real = os.environ.pop("STUDIO_VISION_MODEL", None)
        try:
            vision, note = eng.resolve_vision("http://h/v1", "coder", [])
            self.assertIsNone(vision)
            self.assertIn("cannot see", note)
            os.environ["STUDIO_VISION_MODEL"] = "ghost"
            vision, note = eng.resolve_vision("http://h/v1", "coder", ["odd-vl"])
            # A pin the host does not serve is not a reason to go blind.
            self.assertEqual(vision.model, "odd-vl")
            self.assertIsNone(note)
        finally:
            os.environ.pop("STUDIO_VISION_MODEL", None)
            if real:
                os.environ["STUDIO_VISION_MODEL"] = real

    def test_name_hints_when_host_gives_no_type(self):
        self.assertTrue(eng.looks_vision("Qwen2.5-VL-7B-Instruct"))
        self.assertTrue(eng.looks_vision("gemma-3-12b-it"))
        self.assertFalse(eng.looks_vision("qwen3-coder-30b-a3b-instruct"))

    def test_transparent_pictures_go_onto_white(self):
        """A vision model sees alpha as black, so a black glyph on nothing is
        described as a black square. Opaque pictures pass through untouched."""
        glyph = bytes([0, 0, 0, 255]) * 2 + bytes([0, 0, 0, 0]) * 2   # 2x2: black row, clear row
        flat = icons.flatten_png(icons.png(glyph, 2, 2))
        rgba, w, h = icons.png_to_rgba(flat)
        self.assertEqual(rgba[:8], bytes([0, 0, 0, 255]) * 2)
        self.assertEqual(rgba[8:], bytes([255, 255, 255, 255]) * 2)
        half = bytes([0, 0, 0, 128])
        rgba, _, _ = icons.png_to_rgba(icons.flatten_png(icons.png(half, 1, 1)))
        self.assertEqual(tuple(rgba), (127, 127, 127, 255))
        opaque = icons.png(bytes([10, 20, 30, 255]) * 4, 2, 2)
        self.assertIs(icons.flatten_png(opaque), opaque)
        self.assertEqual(icons.flatten_png(b"not a png"), b"not a png")

    def test_vision_describes_and_reviews_through_the_host(self):
        sent = []

        class FakeLLM:
            def chat(self, messages, tools=None, max_tokens=None):
                sent.append(messages[0]["content"])
                return {"choices": [{"message": {"content": "  a red bridge  "}}]}

        v = eng.Vision("http://h/v1", "odd-vl")
        v.llm = FakeLLM()
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "bridge.png")
        with open(path, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n")
        try:
            block = v.describe_all([path])
            self.assertIn("bridge.png: a red bridge", block)
            self.assertIn("odd-vl", block)
            self.assertEqual(sent[0][1]["image_url"]["url"][:22], "data:image/png;base64,")
            review = v.review({"data": "AAAA", "mimeType": "image/jpeg"}, {"briefs": ["remake"]})
            self.assertEqual(review, "a red bridge")
            self.assertIn("remake", sent[1][0]["text"])
            self.assertTrue(sent[1][1]["image_url"]["url"].startswith("data:image/jpeg"))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestPerAppModel(unittest.TestCase):
    """An app may prefer a small model over the shared one. The preference
    is a preference: it never stops a tab opening. No app in the registry
    uses it today - ComfyUI did, and its 1.7B talked about generating
    instead of calling comfy_generate - so the tests give one to a spec."""

    def setUp(self):
        self.comfy = eng.APPS_BY_ID["comfyui"]
        self._models = self.comfy.models
        self.comfy.models = ["small-1.7b", "small-1.5b"]
        self._pin = os.environ.pop("STUDIO_MODEL_COMFYUI", None)

    def tearDown(self):
        self.comfy.models = self._models
        if self._pin is not None:
            os.environ["STUDIO_MODEL_COMFYUI"] = self._pin
        else:
            os.environ.pop("STUDIO_MODEL_COMFYUI", None)

    def test_a_preferred_small_model_the_host_serves_is_chosen(self):
        model, note = self.comfy.model_for(["big-30b", "small-1.7b"], "big-30b")
        self.assertEqual(model, "small-1.7b")
        self.assertIn("small", note)

    def test_the_first_served_preference_wins_in_registry_order(self):
        first, second = self.comfy.models[:2]
        self.assertEqual(self.comfy.model_for(["big", second], "big")[0], second)
        self.assertEqual(self.comfy.model_for(["big", second, first], "big")[0], first)

    def test_an_unserved_preference_falls_back_to_the_shared_model_and_says_so(self):
        model, note = self.comfy.model_for(["big-30b"], "big-30b")
        self.assertEqual(model, "big-30b")
        self.assertIn("none of this app's preferred models", note)

    def test_the_pin_wins_when_served_and_is_explained_when_not(self):
        os.environ["STUDIO_MODEL_COMFYUI"] = "my-pick"
        self.assertEqual(self.comfy.model_for(["my-pick", "big"], "big"),
                         ("my-pick", "pinned by STUDIO_MODEL_COMFYUI"))
        model, note = self.comfy.model_for(["big"], "big")
        self.assertEqual(model, "big")
        self.assertIn("does not serve", note)

    def test_apps_without_a_preference_use_the_shared_model_silently(self):
        for app in eng.APPS:
            if not app.models:
                self.assertEqual(app.model_for(["x", "big"], "big"), ("big", ""))

    def test_comfyui_shares_the_model_the_other_tabs_use(self):
        """The small-model rule made the tab unusable: qwen3-1.7b under the
        ComfyUI briefing and nineteen tools described the generation it was
        about to make and never called it, or was cut off. The shared 30B
        made the picture. A host that cannot hold both pins one with
        STUDIO_MODEL_COMFYUI."""
        self.comfy.models = self._models
        self.assertEqual(self.comfy.models, [])
        self.assertEqual(self.comfy.model_for(["big-30b", "qwen3-1.7b"], "big-30b"),
                         ("big-30b", ""))


class TestDraftChoice(unittest.TestCase):
    """Speculative decoding: a small model of the executing model's family
    runs ahead of it. The pair is resolved from what the host serves, as the
    executing and vision models are, and never guessed across families - a
    pair with different vocabularies is refused by the host."""

    IDS = ["qwen3-coder-30b-a3b-instruct", "qwen3-0.6b", "qwen3-1.7b",
           "qwen2.5-coder-14b-instruct", "qwen2.5-coder-0.5b-instruct", "gpt-oss-20b"]
    BIG = "qwen3-coder-30b-a3b-instruct"

    def setUp(self):
        self._pin = os.environ.pop("STUDIO_DRAFT_MODEL", None)

    def tearDown(self):
        os.environ.pop("STUDIO_DRAFT_MODEL", None)
        if self._pin is not None:
            os.environ["STUDIO_DRAFT_MODEL"] = self._pin

    def test_the_smallest_served_model_of_the_family(self):
        self.assertEqual(eng.pick_draft_model(self.BIG, self.IDS), "qwen3-0.6b")
        self.assertEqual(eng.pick_draft_model("qwen2.5-coder-14b-instruct", self.IDS),
                         "qwen2.5-coder-0.5b-instruct")
        without = [i for i in self.IDS if i != "qwen3-0.6b"]
        self.assertEqual(eng.pick_draft_model(self.BIG, without), "qwen3-1.7b")

    def test_a_family_is_a_whole_name_not_a_prefix(self):
        self.assertEqual(eng.draft_family(self.BIG), "qwen3")
        self.assertEqual(eng.draft_family("lmstudio-community/Qwen3-30B-A3B"), "qwen3")
        self.assertEqual(eng.draft_family("meta-llama-3.1-8b-instruct"), "llama-3.1")
        self.assertEqual(eng.draft_family("qwen2.5-coder-14b-instruct"), "qwen2.5-coder")
        # qwen3.6 is not qwen3, and gemma-3n is not gemma-3: different vocabularies
        self.assertIsNone(eng.draft_family("qwen3.6-27b"))
        self.assertIsNone(eng.draft_family("gemma-3n-e4b-it"))
        self.assertIsNone(eng.draft_family("gpt-oss-20b"))

    def test_no_family_or_nothing_served_means_no_draft(self):
        self.assertIsNone(eng.pick_draft_model("gpt-oss-20b", self.IDS))
        self.assertIsNone(eng.pick_draft_model("qwen3.6-27b", self.IDS + ["qwen3.6-27b"]))
        self.assertIsNone(eng.pick_draft_model(self.BIG, [self.BIG, "gpt-oss-20b"]))

    def test_a_draft_sized_model_runs_alone(self):
        # The ComfyUI tab's 1.7B: nothing smaller is worth running ahead of it.
        self.assertIsNone(eng.pick_draft_model("qwen3-1.7b", self.IDS))
        self.assertIsNone(eng.pick_draft_model("qwen3-0.6b", self.IDS))

    def test_the_pin_wins_when_served_and_off_is_off(self):
        self.assertEqual(eng.pick_draft_model(self.BIG, self.IDS, want="qwen3-1.7b"), "qwen3-1.7b")
        # a pair the table does not know is the user's call
        self.assertEqual(eng.pick_draft_model("gpt-oss-20b", self.IDS, want="qwen3-0.6b"),
                         "qwen3-0.6b")
        for off in ("off", "none", "0", "OFF", " no "):
            self.assertIsNone(eng.pick_draft_model(self.BIG, self.IDS, want=off), off)

    def test_resolve_reads_the_pin_and_explains_one_the_host_lacks(self):
        self.assertEqual(eng.resolve_draft(self.BIG, self.IDS), ("qwen3-0.6b", ""))
        self.assertEqual(eng.resolve_draft("gpt-oss-20b", self.IDS), (None, ""))
        os.environ["STUDIO_DRAFT_MODEL"] = "ghost-0.5b"
        draft, note = eng.resolve_draft(self.BIG, self.IDS)
        self.assertEqual(draft, "qwen3-0.6b")   # a stale pin is not a reason to go slow
        self.assertIn("ghost-0.5b", note)
        self.assertIn("qwen3-0.6b", note)
        draft, note = eng.resolve_draft("gpt-oss-20b", self.IDS)
        self.assertIsNone(draft)
        self.assertIn("no draft model", note)
        os.environ["STUDIO_DRAFT_MODEL"] = "off"
        self.assertEqual(eng.resolve_draft(self.BIG, self.IDS), (None, ""))


class TestSpeculativeRequests(unittest.TestCase):
    """The draft model rides on every request as LM Studio's `draft_model`,
    which the host loads just in time; a pair it refuses is dropped after one
    retry and explained, rather than failing every request after it."""

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def setUp(self):
        import urllib.request
        self.calls = []
        self.refuse = None        # (status, text) for a request carrying a draft
        self.refuse_all = False   # ...or for every request
        self.stats = None
        self.cut = False          # the host stops the reply with "length"
        real = urllib.request.urlopen

        def fake_open(req, timeout=None):
            body = json.loads(req.data)
            self.calls.append(body)
            if self.cut:
                chunks = [{"choices": [{"delta": {"content": "<think>"}, "finish_reason": None}]},
                          {"choices": [{"delta": {}, "finish_reason": "length"}]}]
                lines = ["data: " + json.dumps(c) for c in chunks] + ["data: [DONE]"]
                return self.Resp("\n".join(lines).encode("utf-8"))
            if self.refuse and (self.refuse_all or body.get("draft_model")):
                code, text = self.refuse
                raise urllib.error.HTTPError(req.full_url, code, "bad", {},
                                             io.BytesIO(text.encode("utf-8")))
            msg = {"role": "assistant", "content": "ok"}
            if body.get("stream"):
                chunks = [{"choices": [{"delta": {"content": "ok"}, "finish_reason": None}]},
                          {"choices": [{"delta": {}, "finish_reason": "stop"}]}]
                if self.stats:
                    chunks[-1]["stats"] = self.stats
                lines = ["data: " + json.dumps(c) for c in chunks] + ["data: [DONE]"]
                return self.Resp("\n".join(lines).encode("utf-8"))
            reply = {"choices": [{"message": msg, "finish_reason": "stop"}]}
            if self.stats:
                reply["stats"] = self.stats
            return self.Resp(json.dumps(reply).encode("utf-8"))
        urllib.request.urlopen = fake_open
        self.addCleanup(setattr, urllib.request, "urlopen", real)
        self.llm = eng.LLM("http://h:1234/v1", "big-30b", draft="tiny-0.6b")
        self.messages = [{"role": "user", "content": "hi"}]

    def test_the_draft_rides_on_every_request_and_only_when_there_is_one(self):
        self.llm.chat(self.messages, max_tokens=1)
        self.llm.stream(self.messages, on_text=lambda _: None)
        self.assertEqual([c.get("draft_model") for c in self.calls], ["tiny-0.6b"] * 2)
        self.assertEqual(self.calls[0]["max_tokens"], 1)
        self.assertTrue(self.calls[1]["stream"])
        plain = eng.LLM("http://h:1234/v1", "big-30b")
        plain.chat(self.messages)
        self.assertNotIn("draft_model", self.calls[-1])

    def test_a_refused_pair_is_dropped_once_and_explained(self):
        self.refuse = (400, "draft model vocabulary does not match")
        data = self.llm.chat(self.messages)
        self.assertEqual(data["choices"][0]["message"]["content"], "ok")
        # the same request, once more, without the draft
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.calls[0]["draft_model"], "tiny-0.6b")
        self.assertNotIn("draft_model", self.calls[1])
        self.assertEqual(self.calls[0]["messages"], self.calls[1]["messages"])
        self.assertIsNone(self.llm.draft)
        for word in ("tiny-0.6b", "big-30b", "400", "vocabulary"):
            self.assertIn(word, self.llm.draft_note)
        # and never again: the next request goes out plain, first time
        self.llm.stream(self.messages, on_text=lambda _: None)
        self.assertEqual(len(self.calls), 3)
        self.assertNotIn("draft_model", self.calls[2])

    def test_an_error_that_is_not_the_drafts_keeps_the_draft_and_the_error(self):
        self.refuse, self.refuse_all = (400, "Unrecognized schema: false"), True
        with self.assertRaises(RuntimeError) as cm:
            self.llm.chat(self.messages)
        self.assertIn("Unrecognized schema", str(cm.exception))
        self.assertIn("400", str(cm.exception))
        self.assertEqual(len(self.calls), 2)          # tried once without, no better
        self.assertEqual(self.llm.draft, "tiny-0.6b")
        self.assertIsNone(self.llm.draft_note)

    def test_a_length_cutoff_names_the_model_and_the_window(self):
        # The ComfyUI tab on qwen3-1.7b at 8,192 tokens: the prefix left the
        # model no room, the host stopped it with "length", and the bare
        # finish_reason told the user nothing they could act on.
        self.cut = True
        with self.assertRaises(RuntimeError) as cm:
            self.llm.stream(self.messages, on_text=lambda _: None)
        text = str(cm.exception)
        self.assertIn("Incomplete inference response (length)", text)
        for word in ("big-30b", "context", "no tools from it were executed", "16384"):
            self.assertIn(word, text)

    def test_the_hosts_score_is_kept_when_it_reports_one(self):
        self.stats = {"draft_model": "tiny-0.6b", "total_draft_tokens_count": 10,
                      "accepted_draft_tokens_count": 8, "tokens_per_second": 50}
        self.llm.chat(self.messages)
        self.assertEqual(self.llm.drafted, (8, 10))
        self.llm.stream(self.messages, on_text=lambda _: None)
        self.assertEqual(self.llm.drafted, (16, 20))
        # a host that says nothing, or an empty stats object: nothing counted
        self.stats = {"tokens_per_second": 50}
        self.llm.chat(self.messages)
        self.stats = None
        self.llm.chat(self.messages)
        self.assertEqual(self.llm.drafted, (16, 20))


class TestHeadroom(unittest.TestCase):
    """A tab whose prefix fills the model's window is told at warm-up, with
    the numbers and the fix, rather than cut off on its first message."""

    def test_too_little_room_is_named_with_the_fix(self):
        note = eng.headroom_note("qwen3-1.7b", 7178, 8192, 32768)
        for word in ("7,178", "8,192", "about 1,014 tokens", "qwen3-1.7b",
                     "lms load qwen3-1.7b --context-length 16384", "up to 32,768"):
            self.assertIn(word, note)

    def test_enough_room_or_no_numbers_says_nothing(self):
        self.assertEqual(eng.headroom_note("m", 3000, 8192, 32768), "")
        self.assertEqual(eng.headroom_note("m", 8192 - eng.MIN_ROOM, 8192), "")
        self.assertEqual(eng.headroom_note("m", None, 8192), "")
        self.assertEqual(eng.headroom_note("m", 7000, None), "")

    def test_a_prefix_past_the_window_leaves_nothing(self):
        self.assertIn("leaving nothing", eng.headroom_note("m", 9000, 8192, 32768))

    def test_a_model_already_at_its_maximum_needs_replacing_not_reloading(self):
        note = eng.headroom_note("tiny", 7178, 8192, 8192)
        self.assertIn("needs a model with a larger context window", note)
        self.assertNotIn("lms load", note)

    def test_the_asked_for_window_never_exceeds_the_models_own(self):
        self.assertIn("--context-length 12288", eng.headroom_note("m", 7000, 8192, 12288))

    def test_the_window_comes_from_the_hosts_model_list(self):
        import urllib.request
        real = urllib.request.urlopen

        class Resp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        seen = []

        def fake_open(url, timeout=None):
            seen.append(url)
            return Resp(json.dumps({"data": [
                {"id": "qwen3-1.7b", "state": "loaded", "loaded_context_length": 8192,
                 "max_context_length": 32768},
                {"id": "other", "state": "not-loaded"}]}).encode("utf-8"))
        urllib.request.urlopen = fake_open
        self.addCleanup(setattr, urllib.request, "urlopen", real)
        self.assertEqual(eng.context_window("http://h:1234/v1", "qwen3-1.7b"), (8192, 32768))
        self.assertEqual(seen, ["http://h:1234/api/v0/models"])
        self.assertEqual(eng.context_window("http://h:1234/v1", "other"), (None, None))
        self.assertEqual(eng.context_window("http://h:1234/v1", "absent"), (None, None))

    def test_a_host_that_does_not_answer_gives_no_numbers(self):
        import urllib.request
        real = urllib.request.urlopen

        def gone(url, timeout=None):
            raise urllib.error.URLError("timed out")
        urllib.request.urlopen = gone
        self.addCleanup(setattr, urllib.request, "urlopen", real)
        self.assertEqual(eng.context_window("http://h:1234/v1", "m"), (None, None))


class FakeHost:
    """LM Studio's REST API, as much of it as a fit touches: the v0 list
    with each model's window, the v1 list with its instances, load and
    unload. Records every write it is sent."""

    def __init__(self, loaded=8192, maximum=262144, instances=("big-30b",), fail_load=None):
        self.loaded, self.maximum = loaded, maximum
        self.instances = list(instances)
        self.fail_load = fail_load
        self.calls = []

    def __call__(self, req, timeout=None):
        import urllib.request
        url = req.full_url if isinstance(req, urllib.request.Request) else req
        body = json.loads(req.data) if getattr(req, "data", None) else None
        if url.endswith("/api/v0/models"):
            entry = {"id": "big-30b", "max_context_length": self.maximum,
                     "state": "loaded" if self.instances else "not-loaded"}
            if self.instances and self.loaded:
                entry["loaded_context_length"] = self.loaded
            return self._resp({"data": [entry, {"id": "other", "state": "not-loaded"}]})
        if url.endswith("/api/v1/models"):
            return self._resp({"models": [{"key": "big-30b", "loaded_instances": [
                {"id": i, "config": {"context_length": self.loaded}} for i in self.instances]}]})
        if url.endswith("/api/v1/models/unload"):
            self.calls.append(("unload", body))
            self.instances.remove(body["instance_id"])
            return self._resp({"instance_id": body["instance_id"]})
        if url.endswith("/api/v1/models/load"):
            self.calls.append(("load", body))
            if self.fail_load:
                raise urllib.error.HTTPError(url, 400, "bad", {}, io.BytesIO(self.fail_load.encode()))
            self.instances.append(body["model"])
            self.loaded = body.get("context_length")
            return self._resp({"status": "loaded", "instance_id": body["model"]})
        raise AssertionError("unexpected url " + url)

    @staticmethod
    def _resp(payload):
        class Resp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False
        return Resp(json.dumps(payload).encode("utf-8"))


class TestFitModel(unittest.TestCase):
    """The window a model is loaded with is the app's to set. LM Studio loads
    at its default - 8,192 for the 30B - and every tab's prefix is most or
    all of that: the ComfyUI tab measured 7,707 tokens before a word was
    said, After Effects about 20,000. Under that the model lost its own tool
    results to truncation and called list_folder ten times running. The
    same call that loads the vision model takes a context_length, so the app
    loads, or reloads, the model with one that fits."""

    H = "http://h:1234/v1"

    def host(self, **kw):
        import urllib.request
        fake = FakeHost(**kw)
        real = urllib.request.urlopen
        urllib.request.urlopen = fake
        self.addCleanup(setattr, urllib.request, "urlopen", real)
        return fake

    def test_the_wanted_window_is_a_power_of_two_with_room_after_the_prefix(self):
        self.assertEqual(eng.wanted_context(7707), 16384)
        self.assertEqual(eng.wanted_context(20000), 32768)
        self.assertEqual(eng.wanted_context(31000), 65536)
        self.assertEqual(eng.wanted_context(None), 16384)
        self.assertEqual(eng.wanted_context(20000, maximum=24576), 24576)
        self.assertEqual(eng.wanted_context(1000, maximum=8192), 8192)

    def test_the_estimate_is_on_the_high_side_of_the_measurement(self):
        # 27,506 characters of ComfyUI briefing and tools cost 7,707 tokens.
        tools = [{"type": "function", "function": {"name": "t", "parameters": {}}}]
        chars = 11374 + len(json.dumps(tools))
        self.assertGreaterEqual(eng.estimate_tokens("x" * 11374, tools), chars // 3)
        self.assertEqual(eng.estimate_tokens("", []), 0)

    def test_a_window_that_fits_is_left_alone(self):
        fake = self.host(loaded=32768)
        self.assertEqual(eng.fit_model(self.H, "big-30b", 7707), (32768, ""))
        self.assertEqual(fake.calls, [])
        # and with no count yet there is nothing to fit against
        fake = self.host(loaded=8192)
        self.assertEqual(eng.fit_model(self.H, "big-30b", None), (8192, ""))
        self.assertEqual(fake.calls, [])

    def test_a_short_window_is_unloaded_and_loaded_again_larger(self):
        fake = self.host(loaded=8192)
        now, note = eng.fit_model(self.H, "big-30b", 7707)
        self.assertEqual(now, 16384)
        self.assertEqual(fake.calls, [("unload", {"instance_id": "big-30b"}),
                                      ("load", {"model": "big-30b", "context_length": 16384})])
        for word in ("Reloaded big-30b", "16,384", "7,707", "8,192", "about 485 tokens"):
            self.assertIn(word, note)

    def test_a_model_not_yet_loaded_is_loaded_with_the_window_not_the_default(self):
        fake = self.host(loaded=None, instances=())
        now, note = eng.fit_model(self.H, "big-30b", 20000)
        self.assertEqual(now, 32768)
        self.assertEqual(fake.calls, [("load", {"model": "big-30b", "context_length": 32768})])
        self.assertIn("Loaded big-30b with a 32,768-token context window", note)

    def test_every_instance_goes_before_the_reload(self):
        # A load of a model already loaded is a second instance, and requests
        # by model id keep going to the first: both have to go.
        fake = self.host(loaded=8192, instances=("big-30b", "big-30b:2"))
        eng.fit_model(self.H, "big-30b", 7707)
        self.assertEqual([c for c in fake.calls if c[0] == "unload"],
                         [("unload", {"instance_id": "big-30b"}),
                          ("unload", {"instance_id": "big-30b:2"})])

    def test_a_model_at_its_maximum_is_not_reloaded_but_named(self):
        fake = self.host(loaded=8192, maximum=8192)
        now, note = eng.fit_model(self.H, "big-30b", 7707)
        self.assertEqual((now, fake.calls), (8192, []))
        self.assertIn("needs a model with a larger context window", note)

    def test_a_host_that_says_nothing_is_left_alone(self):
        import urllib.request

        def gone(req, timeout=None):
            raise urllib.error.URLError("no such endpoint")
        real = urllib.request.urlopen
        urllib.request.urlopen = gone
        self.addCleanup(setattr, urllib.request, "urlopen", real)
        self.assertEqual(eng.fit_model(self.H, "big-30b", 7707), (None, ""))

    def test_a_load_that_fails_falls_back_to_the_advice(self):
        fake = self.host(loaded=8192, fail_load="out of memory")
        now, note = eng.fit_model(self.H, "big-30b", 7707)
        self.assertEqual(now, 8192)
        self.assertIn("Could not load big-30b with a 16,384-token context window", note)
        self.assertIn("out of memory", note)
        self.assertIn("lms load big-30b --context-length 16384", note)

    def test_load_model_sends_the_window_when_given_one(self):
        fake = self.host(loaded=None, instances=())
        self.assertIsNone(eng.load_model(self.H, "big-30b", context_length=32768))
        self.assertIsNone(eng.load_model(self.H, "big-30b"))
        self.assertEqual([c[1] for c in fake.calls],
                         [{"model": "big-30b", "context_length": 32768}, {"model": "big-30b"}])

    def test_instances_come_from_the_v1_list(self):
        self.host(loaded=8192, instances=("big-30b", "big-30b:2"))
        self.assertEqual(eng.loaded_instances(self.H, "big-30b"),
                         [("big-30b", 8192), ("big-30b:2", 8192)])
        self.assertEqual(eng.loaded_instances(self.H, "absent"), [])


class TestHostUnreachable(unittest.TestCase):
    """Nothing answering is its own error, still a RuntimeError for anyone
    catching that, because the GUI answers it with Connect rather than with
    the message alone."""

    def test_a_url_error_is_host_unreachable(self):
        import urllib.request
        real = urllib.request.urlopen

        def gone(req, timeout=None):
            raise urllib.error.URLError(TimeoutError("timed out"))
        urllib.request.urlopen = gone
        self.addCleanup(setattr, urllib.request, "urlopen", real)
        llm = eng.LLM("http://h:1234/v1", "big-30b")
        with self.assertRaises(eng.HostUnreachable) as cm:
            llm.chat([{"role": "user", "content": "hi"}])
        self.assertIsInstance(cm.exception, RuntimeError)
        self.assertIn("cannot reach inference host", str(cm.exception))
        self.assertIn("timed out", str(cm.exception))
        with self.assertRaises(eng.HostUnreachable):
            llm.stream([{"role": "user", "content": "hi"}])


class TestHostAlive(unittest.TestCase):
    """The tailnet is asked whether the LLM PC is there at all - only a
    Tailscale address, only with the CLI on PATH, and never with a network
    call from a test."""

    def setUp(self):
        self.ran = []
        real_run, real_which = eng.subprocess.run, eng.shutil.which
        self.addCleanup(setattr, eng.subprocess, "run", real_run)
        self.addCleanup(setattr, eng.shutil, "which", real_which)
        eng.shutil.which = lambda name: r"C:\Program Files\Tailscale\tailscale.exe"

        class Done:
            def __init__(self, code, out):
                self.returncode, self.stdout = code, out
        self.answer = Done(0, "pong from desktop-4rkl10b (100.127.17.38) via 192.168.40.43:41641 in 2ms\n")

        def fake_run(args, **kw):
            self.ran.append(args)
            return self.answer
        eng.subprocess.run = fake_run
        self.Done = Done

    def test_a_tailnet_peer_is_pinged(self):
        self.assertIs(eng.host_alive("http://100.127.17.38:1234/v1"), True)
        args = self.ran[-1]
        self.assertTrue(args[0].endswith("tailscale.exe"))
        self.assertEqual(args[1:3], ["ping", "--c"])
        self.assertEqual(args[-1], "100.127.17.38")
        self.answer = self.Done(1, "")
        self.assertIs(eng.host_alive("http://100.127.17.38:1234/v1"), False)

    def test_anything_else_is_not_asked(self):
        for url in ("http://192.168.40.43:1234/v1", "http://llm.local:1234/v1",
                    "http://localhost:1234/v1"):
            with self.subTest(url=url):
                self.assertIsNone(eng.host_alive(url))
        self.assertEqual(self.ran, [])
        eng.shutil.which = lambda name: None
        self.assertIsNone(eng.host_alive("http://100.127.17.38:1234/v1"))
        self.assertEqual(self.ran, [])


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
                # A remote or container app has no .exe to find here, and must
                # say where it runs instead.
                self.assertTrue(app.exe_globs or ((app.remote or app.container)
                                                  and app.launch_note))
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
                self.assertIn(kind, ("port", "process", "url"))
                self.assertTrue(arg)
                if kind == "url":
                    self.assertTrue(app.remote, "only a remote app probes a URL")
                    self.assertTrue(arg.startswith("http"))

    def test_remote_app_is_installed_but_never_launched_from_here(self):
        """ComfyUI lives on the LLM PC: it is always 'installed', so it gets a
        tab, and launch() must explain rather than hunt for an .exe."""
        comfy = eng.APPS_BY_ID["comfyui"]
        self.assertTrue(comfy.remote)
        self.assertIsNone(comfy.exe())
        self.assertTrue(comfy.installed())
        self.assertIn(comfy, eng.installed_apps())
        with self.assertRaises(RuntimeError) as ctx:
            comfy.launch()
        self.assertIn("another machine", str(ctx.exception))
        for app in eng.APPS:
            if app.exe_globs:
                self.assertFalse(app.remote)

    def test_container_app_runs_here_behind_docker(self):
        """OpenCode is on this machine but never on its bare disk: no .exe, not
        remote, 'installed' when Docker is, and started by this window as a
        container that is handed the workspace folder and nothing else."""
        oc = eng.APPS_BY_ID["opencode"]
        self.assertTrue(oc.container)
        self.assertFalse(oc.remote)
        self.assertIsNone(oc.exe())
        self.assertEqual(oc.installed(), eng.docker_exe() is not None)
        self.assertEqual(oc.command, eng.sys.executable)
        self.assertEqual(os.path.basename(oc.args[0]), "studio_opencode_mcp.py")
        self.assertTrue(os.path.isfile(oc.args[0]))
        self.assertTrue(os.path.isfile(os.path.join(oc.dockerfile, "Dockerfile")))
        for app in eng.APPS:
            self.assertFalse(app.remote and app.container, app.id)

    def test_container_is_given_the_workspace_and_nothing_else(self):
        args = eng.docker_run_args(r"C:\ws", image="img", port=4096)
        mounts = [args[i + 1] for i, a in enumerate(args) if a == "-v"]
        binds = [m for m in mounts if ":" in m and not m.startswith(eng.OPENCODE_HOME_VOLUME)]
        self.assertEqual(binds, [r"C:\ws:/workspace"], "only the workspace is bind-mounted")
        self.assertIn("%s:/home/node" % eng.OPENCODE_HOME_VOLUME, mounts)
        ports = [args[i + 1] for i, a in enumerate(args) if a == "-p"]
        self.assertEqual(ports, ["127.0.0.1:4096:4096"], "loopback only")
        self.assertIn("--cap-drop", args)
        self.assertIn("no-new-privileges", args)
        self.assertNotIn("--privileged", args)
        self.assertEqual(args[-1], "img")
        for a in args:
            self.assertNotIn(r"C:\Users", a)

    def test_opencode_config_points_at_the_studio_host(self):
        cfg = eng.opencode_config("http://100.127.17.38:1234/v1", "m1", ["m1", "m2"])
        prov = cfg["provider"]["lmstudio"]
        self.assertEqual(prov["options"]["baseURL"], "http://100.127.17.38:1234/v1")
        self.assertEqual(set(prov["models"]), {"m1", "m2"})
        self.assertEqual(cfg["model"], "lmstudio/m1")
        # Loopback inside the container is the container: rewrite to the host.
        cfg = eng.opencode_config("http://127.0.0.1:1234", "m1", [])
        self.assertEqual(cfg["provider"]["lmstudio"]["options"]["baseURL"],
                         "http://host.docker.internal:1234/v1")
        self.assertIn("m1", cfg["provider"]["lmstudio"]["models"])

    def test_launch_builds_once_then_runs_the_container(self):
        oc = eng.APPS_BY_ID["opencode"]
        calls = []
        images = {"present": False}

        def fake_docker(*args, timeout=0):
            calls.append(list(args))
            if args[0] == "image":
                if not images["present"]:
                    raise RuntimeError("docker image inspect failed: No such image")
                return "sha256:abc\n"
            if args[0] == "build":
                images["present"] = True
            return ""

        tmp = tempfile.mkdtemp()
        real = (eng.docker, eng.docker_exe, eng.probe_models, oc.workspace)
        eng.docker, eng.docker_exe = fake_docker, lambda: "docker"
        eng.probe_models = lambda host, timeout=8: (True, "m1", ["m1", "m2"], [], None)
        oc.workspace = os.path.join(tmp, "ws")
        try:
            oc.launch(host="http://100.127.17.38:1234/v1")
            kinds = [c[0] for c in calls]
            self.assertEqual(kinds, ["image", "build", "rm", "run"])
            self.assertEqual(calls[1][-1], oc.dockerfile)
            self.assertEqual(calls[-1], eng.docker_run_args(oc.workspace, oc.image))
            with open(os.path.join(oc.workspace, "opencode.json"), encoding="utf-8") as f:
                cfg = json.load(f)
            self.assertEqual(cfg["model"], "lmstudio/m1")
            calls.clear()
            oc.launch(host="http://100.127.17.38:1234/v1")
            self.assertEqual([c[0] for c in calls], ["image", "rm", "run"], "built once")
            eng.docker_exe = lambda: None
            with self.assertRaises(RuntimeError) as ctx:
                oc.launch()
            self.assertIn("Docker Desktop", str(ctx.exception))
        finally:
            eng.docker, eng.docker_exe, eng.probe_models, oc.workspace = real
            shutil.rmtree(tmp, ignore_errors=True)

    def test_comfy_bridge_is_the_stdlib_server_beside_the_engine(self):
        comfy = eng.APPS_BY_ID["comfyui"]
        self.assertEqual(comfy.command, eng.sys.executable)
        self.assertTrue(os.path.isfile(comfy.args[0]))
        self.assertEqual(os.path.basename(comfy.args[0]), "studio_comfy_mcp.py")
        self.assertIn(eng.COMFYUI_URL, comfy.probe)

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
    """The one tab with no app behind it: a conversation, and a bridge that
    reads this PC's files and the web but changes nothing."""

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
        self.assertTrue(all(app.drivable and app.bridged for app in eng.APPS))
        self.assertIsNone(eng.CHAT.exe())
        self.assertTrue(eng.CHAT.installed())    # nothing to install
        self.assertTrue(eng.CHAT.running())      # the tab is the whole of it
        with self.assertRaises(RuntimeError):
            eng.CHAT.launch()

    def test_its_bridge_runs_in_process_and_only_reads(self):
        """No subprocess: connect() is a Loopback over the research bridge's
        own Server, and every tool it offers is annotated read-only - the
        executor must never owe a read-back in this tab."""
        import studio_mcp
        import studio_research_mcp as research
        self.assertTrue(eng.CHAT.bridged)
        client = eng.CHAT.connect()
        try:
            self.assertIsInstance(client, studio_mcp.Loopback)
            self.assertIs(client.server, research.SERVER)
            client.initialize()
            tools = client.list_tools()
        finally:
            client.close()
        self.assertEqual({t["name"] for t in tools}, eng.CHAT.tool_names())
        self.assertEqual(eng.CHAT.tool_names(), {n for g in eng.RESEARCH_GROUPS.values() for n in g})
        for t in tools:
            self.assertIs(t["annotations"]["readOnlyHint"], True, t["name"])
        # the script is named too, so the harness can check it like any bridge here
        self.assertTrue(eng.CHAT.args[0].endswith("studio_research_mcp.py"))
        self.assertTrue(os.path.isfile(eng.CHAT.args[0]))

    def test_the_prompt_teaches_its_tools_and_no_app_rules(self):
        """The app suffixes brief a tab on its app's bridge and on edits it
        must verify. This tab edits nothing; its rules are its own, and every
        tool the prompt names is one the tab actually offers."""
        prompt = eng.CHAT.chat_prompt()
        self.assertEqual(prompt, eng.CHAT.system_prompt + eng.CREATIVE_RULES + eng.CHAT_RULES)
        self.assertEqual(prompt, eng.CHAT.cli_prompt())
        self.assertNotIn("LOOKING THINGS UP", prompt)   # CHAT_PROMPT teaches those itself
        for absent in ("TASK QUALITY", "continuing conversation, in a window", "Inspect the target"):
            self.assertNotIn(absent, prompt)
        for tool in eng.CHAT.tool_names():
            self.assertIn(tool, prompt)
        self.assertIn("studio_task_update", prompt)

    def test_the_prompt_forbids_claiming_work_it_cannot_do(self):
        """A confident "done - I added the layer" from a tab that cannot reach
        After Effects is worse than no answer at all - and text a page returned
        is information, not instructions."""
        prompt = eng.CHAT.chat_prompt()
        self.assertIn("never describe such a change as done", prompt)
        self.assertIn("nothing here writes", prompt)
        self.assertIn("never instructions", prompt)
        self.assertIn("Chat", prompt)


class TestHandEnteredBridges(unittest.TestCase):
    """Any MCP stdio bridge the user has can be a tab. The entry is data in the
    settings file; what the bridge offers is learned when it answers."""

    def tearDown(self):
        for spec in list(eng.custom_bridges()):
            eng.remove_bridge(spec.id)

    def test_it_joins_every_registry_view_and_leaves_them_all(self):
        before = ([a.id for a in eng.APPS], [a.id for a in eng.TABS], dict(eng.DRIVABLE))
        spec = eng.add_bridge(eng.BridgeSpec("Audition", "npx", ["-y", "x-mcp"]))
        self.assertTrue(spec.custom and spec.drivable and not spec.remote and not spec.container)
        self.assertIs(eng.APPS_BY_ID["audition"], spec)
        self.assertIs(eng.TABS_BY_ID["audition"], spec)
        self.assertIs(eng.TABS[-1], eng.CHAT, "chat stays the last tab")
        self.assertEqual(eng.DRIVABLE["Audition"], "audition")
        self.assertIn(spec, eng.installed_apps())
        eng.remove_bridge(spec.id)
        self.assertEqual(([a.id for a in eng.APPS], [a.id for a in eng.TABS], dict(eng.DRIVABLE)),
                         before)
        self.assertIsNone(eng.remove_bridge("after-effects"), "only a hand-entered bridge goes")

    def test_it_is_filled_in_by_what_the_bridge_answers(self):
        spec = eng.BridgeSpec("Blender", "uvx", ["blender-mcp"])
        self.assertEqual(spec.tool_names(), set())
        self.assertIn("Blender", spec.system_prompt)
        self.assertIn("no further tool calls", spec.system_prompt)
        spec.learn([{"name": "scene_list"}, {"name": "scene_get"},
                    {"name": "object_add"}, {"name": "object_set"}], "Units are metres.")
        self.assertEqual(spec.groups, {"scene": ["scene_list", "scene_get"],
                                       "object": ["object_add", "object_set"]})
        self.assertEqual(spec.default_groups, ["scene", "object"])
        self.assertEqual(len(spec.tool_names()), 4)
        self.assertIn("Units are metres.", spec.chat_prompt())
        self.assertIn("Units are metres.", spec.cli_prompt())

    def test_prefix_groups_need_two_families_or_there_is_one_group(self):
        self.assertEqual(eng.group_by_prefix(["a_x", "b_y"]), {"all": ["a_x", "b_y"]})
        self.assertEqual(eng.group_by_prefix(["one", "two", "three"]), {"all": ["one", "two", "three"]})
        self.assertEqual(eng.group_by_prefix([]), {})
        self.assertEqual(list(eng.group_by_prefix(["a_1", "a_2", "b_1", "b_2"])), ["a", "b"])

    def test_no_probe_means_nothing_to_check_and_no_exe_means_nothing_to_start(self):
        spec = eng.BridgeSpec("Thing", "npx", [])
        self.assertTrue(spec.running())
        self.assertTrue(spec.installed())
        with self.assertRaises(RuntimeError) as ctx:
            spec.launch()
        self.assertIn("no program path", str(ctx.exception))
        probed = eng.BridgeSpec("Thing", "npx", [], probe="process:NoSuchThing.exe")
        self.assertFalse(probed.running())

    def test_records_round_trip_and_junk_is_skipped(self):
        spec = eng.BridgeSpec("Premiere Pro", r"C:\tools\node.exe", ["server.js"],
                              exe=r"C:\x\Premiere.exe", probe="process:Premiere.exe")
        rec = spec.record()
        again = eng.bridge_from_record(rec)
        self.assertEqual(again.record(), rec)
        self.assertEqual(again.exe_globs, [r"C:\x\Premiere.exe"])
        for junk in ("nope", {}, {"name": "x"}, {"command": "y"}, {"name": " ", "command": "y"},
                     {"name": "x", "command": ""}):
            self.assertIsNone(eng.bridge_from_record(junk), junk)
        loaded = eng.load_bridges([rec, "junk", {"name": "B", "command": "c", "args": "not a list"}])
        self.assertEqual([s.id for s in loaded], ["premiere-pro", "b"])
        self.assertEqual(loaded[1].args, [])

    def test_a_hand_entered_bridge_never_shadows_one_written_here(self):
        spec = eng.bridge_from_record({"name": "Photoshop", "command": "npx"})
        self.assertEqual(spec.id, "photoshop-bridge")
        self.assertFalse(eng.APPS_BY_ID["photoshop"].custom)

    def test_command_lines_split_the_windows_way(self):
        self.assertEqual(eng.split_command(r'"C:\Program Files\x\python.exe" s.py --a "b c"'),
                         (r"C:\Program Files\x\python.exe", ["s.py", "--a", "b c"]))
        self.assertEqual(eng.split_command("npx -y some-mcp"), ("npx", ["-y", "some-mcp"]))

    def test_the_sidebar_learns_of_it(self):
        eng.add_bridge(eng.BridgeSpec("Blender", "uvx", ["blender-mcp"]))
        rows = {r["name"]: r for r in eng.detect_apps()}
        self.assertTrue(rows["Blender"]["drivable"])
        self.assertEqual(rows["Blender"]["version"], "bridge")
        self.assertFalse(rows["Blender"]["remote"])
        # a bridge named for an app that already has one written here gets a
        # tab, but the sidebar row stays with the bridge written here
        eng.add_bridge(eng.BridgeSpec("After Effects", "npx", ["other-ae-mcp"], id="ae2"))
        rows = {r["name"]: r for r in eng.detect_apps()}
        self.assertEqual(eng.DRIVABLE["After Effects"], "after-effects")
        if "After Effects" in rows:
            self.assertEqual(rows["After Effects"]["id"], "after-effects")
        eng.remove_bridge("ae2")
        self.assertEqual(eng.DRIVABLE["After Effects"], "after-effects")

    def test_the_cli_takes_a_command_line(self):
        import subprocess
        out = subprocess.run([sys.executable, "studio_agent.py", "--help"], capture_output=True,
                             text=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.assertIn("--mcp", out.stdout)


class TestComBridges(unittest.TestCase):
    """Photoshop and Illustrator are driven through COM scripting by bridges
    written here. Nothing in these tests reaches PowerShell or an app: the
    host's `run` is replaced and the tool bodies are checked as text."""

    def setUp(self):
        import studio_photoshop_mcp as ps
        import studio_illustrator_mcp as ai
        self.ps, self.ai = ps, ai
        self.calls = []
        self._real = (ps.HOST.run, ai.HOST.run, ps.HOST.running, ai.HOST.running)

    def tearDown(self):
        self.ps.HOST.run, self.ai.HOST.run, self.ps.HOST.running, self.ai.HOST.running = self._real

    def _answer(self, host, value, running=True):
        def run(body, timeout=None, setup="", teardown=""):
            self.calls.append(body)
            return value
        host.run = run
        host.running = lambda: running

    def test_registry_entries_point_at_the_scripts_beside_the_engine(self):
        for app_id, script in (("photoshop", "studio_photoshop_mcp.py"),
                               ("illustrator", "studio_illustrator_mcp.py")):
            app = eng.APPS_BY_ID[app_id]
            self.assertEqual(app.command, eng.sys.executable)
            self.assertEqual(os.path.basename(app.args[0]), script)
            self.assertTrue(os.path.isfile(app.args[0]))
            self.assertTrue(app.probe.startswith("process:"))

    def test_status_never_starts_a_closed_app(self):
        self._answer(self.ps.HOST, {"running": True}, running=False)
        res = self.ps.call_tool("ps_status", {})
        self.assertEqual(self.calls, [], "status must not touch COM when the app is closed")
        self.assertIn("not running", res["content"][0]["text"])
        self.assertFalse(res.get("isError"))

    def test_layers_are_addressed_by_id_and_a_bad_id_is_a_sentence(self):
        import studio_com
        self._answer(self.ps.HOST, {"layer_id": 7, "name": "Title", "kind": "text", "visible": True,
                                    "opacity": 100, "blend_mode": "normal", "locked": False,
                                    "depth": 0, "bounds": [10, 20, 110, 60]})
        res = self.ps.call_tool("ps_set_layer", {"layer_id": 7, "opacity": 50})
        self.assertIn("__layer(d, 7)", self.calls[-1])
        self.assertIn("layer_id 7", res["content"][0]["text"])
        res = self.ps.call_tool("ps_set_layer", {"layer_id": 7})
        self.assertTrue(res["isError"])
        self.assertIn("nothing to set", res["content"][0]["text"])

        def refuse(body, **kw):
            raise studio_com.ComError("No layer with layer_id 99")
        self.ps.HOST.run = refuse
        res = self.ps.call_tool("ps_delete_layer", {"layer_id": 99})
        self.assertTrue(res["isError"])
        self.assertIn("layer_id 99", res["content"][0]["text"])

    def test_save_as_refuses_to_overwrite_unless_told(self):
        self._answer(self.ps.HOST, {"name": "a.psd", "path": None})
        tmp = tempfile.mkdtemp()
        here = os.path.join(tmp, "taken.png")
        open(here, "wb").close()
        self.addCleanup(shutil.rmtree, tmp, True)
        res = self.ps.call_tool("ps_save_as", {"path": here, "format": "png"})
        self.assertTrue(res["isError"])
        self.assertIn("overwrite=true", res["content"][0]["text"])
        self.assertEqual(self.calls, [])

    def test_illustrator_flips_y_so_the_model_sees_y_down(self):
        self._answer(self.ai.HOST, {"uuid": "42", "type": "path", "name": "", "layer": "Layer 1",
                                    "hidden": False, "locked": False, "opacity": 100,
                                    "bounds": [10, 20, 60, 50]})
        self.ai.call_tool("ai_add_shape", {"kind": "rectangle", "x": 10, "y": 20, "width": 50, "height": 30})
        self.assertIn("rectangle(-(20), 10, 50, 30)", self.calls[-1])
        self.ai.call_tool("ai_set_item", {"uuid": "42", "x": 5, "y": 7})
        self.assertIn("-(7)", self.calls[-1])
        self.ai.call_tool("ai_transform_item", {"uuid": "42", "dx": 3, "dy": 4, "rotate": 10})
        self.assertIn("translate(3, -(4))", self.calls[-1])
        self.assertIn("rotate(-10", self.calls[-1])
        res = self.ai.call_tool("ai_reorder_item", {"uuid": "42", "position": "above"})
        self.assertTrue(res["isError"])
        self.assertIn("relative_to", res["content"][0]["text"])

    def test_the_prelude_serializes_what_extendscript_cannot(self):
        import studio_com
        js = studio_com.script("return 1", setup="SETUP;", teardown="TEARDOWN;")
        for needle in ("function __J(", "SETUP;", "TEARDOWN;", "__error", "return 1"):
            self.assertIn(needle, js)
        host = studio_com.ComHost("No.Such.ProgID", "Nothing", "Nothing.exe")
        self.assertIsNone(host._decode("  "))
        self.assertEqual(host._decode('{"a": [1, 2]}'), {"a": [1, 2]})
        with self.assertRaises(studio_com.ComError) as ctx:
            host._decode('{"__error": "boom", "line": 3}')
        self.assertIn("boom", str(ctx.exception))
        self.assertIn("line 3", str(ctx.exception))
        self.assertIn("not registered", host._explain("Invalid class string 80040154"))
        self.assertIn("busy", host._explain("Call was rejected by callee 80010001"))

    def test_process_check_survives_tasklist_truncating_long_image_names(self):
        """tasklist's table view cuts image names at 25 characters, which lost the
        ".exe" of Premiere Beta's; both helpers ask for CSV, which does not."""
        import studio_com
        seen = []

        class Out:
            stdout = '"Adobe Premiere Pro (Beta).exe","1","Console","1","10 K"\n'

        def fake_run(cmd, **kw):
            seen.append(cmd)
            return Out()
        real = studio_com.subprocess.run, eng.subprocess.run
        studio_com.subprocess.run = eng.subprocess.run = fake_run
        try:
            self.assertTrue(studio_com.process_running("Adobe Premiere Pro (Beta).exe"))
            self.assertTrue(eng.process_running("Adobe Premiere Pro (Beta).exe"))
        finally:
            studio_com.subprocess.run, eng.subprocess.run = real
        for cmd in seen:
            self.assertEqual(cmd[-2:], ["/FO", "CSV"], cmd)

    def test_a_progid_nobody_registered_is_a_sentence_not_a_hang(self):
        """The one test that runs the PowerShell worker. No app is named, so
        nothing starts; the COM error comes back as prose within seconds."""
        import studio_com
        if not shutil.which("powershell.exe"):
            self.skipTest("no PowerShell")
        host = studio_com.ComHost("Studio.NoSuchApp.Test", "Nothing", "Nothing.exe")
        try:
            with self.assertRaises(studio_com.ComError) as ctx:
                host.run("return 1", timeout=20)
            self.assertIn("Nothing", str(ctx.exception))
        finally:
            host.close()


class TestDetection(unittest.TestCase):
    def test_detect_apps_shape(self):
        for a in eng.detect_apps():
            self.assertEqual({"code", "name", "version", "fg", "bg", "id", "exe",
                              "drivable", "remote"}, set(a))
            self.assertTrue(a["fg"].startswith("#"))

    def test_remote_apps_are_listed_without_an_exe(self):
        """ComfyUI has nothing on this disk; the row comes from the registry
        alone, flagged for the sidebar's second group."""
        rows = {a["name"]: a for a in eng.detect_apps()}
        for app in eng.APPS:
            if app.remote:
                with self.subTest(app=app.name):
                    self.assertIn(app.name, rows)
                    self.assertTrue(rows[app.name]["remote"])
                    self.assertIsNone(rows[app.name]["exe"])
                    self.assertTrue(rows[app.name]["drivable"])

    def test_container_apps_sit_with_this_pc_and_say_so(self):
        """OpenCode runs here, in Docker: the row is in this PC's group, not the
        LLM PC's, with 'container' where a version year would go."""
        rows = {a["name"]: a for a in eng.detect_apps()}
        for app in eng.APPS:
            if app.container:
                with self.subTest(app=app.name):
                    self.assertIn(app.name, rows)
                    self.assertFalse(rows[app.name]["remote"])
                    self.assertEqual(rows[app.name]["version"], "container")
                    self.assertIsNone(rows[app.name]["exe"])
                    self.assertTrue(rows[app.name]["drivable"])

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


class TestErrorLog(unittest.TestCase):
    """The app's only forensic record: the shortcut starts it with
    `pythonw.exe`, so there is no console a traceback could reach instead."""

    def setUp(self):
        import studio_chat
        self.mod = studio_chat
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)
        self._real = os.environ.get("STUDIO_SETTINGS")
        os.environ["STUDIO_SETTINGS"] = os.path.join(self.dir, "settings.json")
        self.addCleanup(self._restore)
        self.path = studio_chat.error_log_path()

    def _restore(self):
        if self._real is None:
            os.environ.pop("STUDIO_SETTINGS", None)
        else:
            os.environ["STUDIO_SETTINGS"] = self._real

    def test_it_writes_beside_the_settings_and_says_where(self):
        self.assertEqual(self.mod.log_error("first entry"), self.path)
        with open(self.path, encoding="utf-8") as f:
            self.assertIn("first entry", f.read())

    def test_it_rolls_over_instead_of_growing_without_end(self):
        """Months of a workstation tool's failures must not become one file
        nobody can open. One generation back is as far as anyone looks."""
        self.mod.log_error("x" * (self.mod.LOG_MAX_BYTES + 1))
        self.mod.log_error("after the rollover")
        self.assertTrue(os.path.exists(self.path + ".1"))
        with open(self.path, encoding="utf-8") as f:
            kept = f.read()
        self.assertIn("after the rollover", kept)
        self.assertLess(len(kept), self.mod.LOG_MAX_BYTES)

    def test_an_unwritable_log_costs_the_entry_not_the_app(self):
        os.environ["STUDIO_SETTINGS"] = os.path.join(
            self.dir, "settings.json", "no", "settings.json")
        with open(os.path.join(self.dir, "settings.json"), "w") as f:
            f.write("")               # a file where the directory would have to be
        self.assertIsNone(self.mod.log_error("nowhere to put this"))

    def test_a_startup_failure_is_logged_rather_than_silent(self):
        """Without this, a failure before the window exists looked exactly
        like double-clicking the shortcut and nothing happening."""
        shown = []
        real = self.mod.messagebox.showerror
        self.mod.messagebox.showerror = lambda *a, **k: shown.append(a)
        try:
            self.mod.fail_visibly("starting up", "Traceback...\nTclError: no display")
        finally:
            self.mod.messagebox.showerror = real
        with open(self.path, encoding="utf-8") as f:
            body = f.read()
        self.assertIn("starting up", body)
        self.assertIn("TclError: no display", body)


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
        # Connect runs the real one against a stubbed probe; see the tests.
        cls.real_boot_host = staticmethod(studio_chat.Chat._boot_host)
        studio_chat.Chat._boot_host = lambda self, *a, **k: None
        studio_chat.Chat._ensure = lambda self, s: None
        studio_chat.Chat._read_icons = lambda self: None
        # A tab's boot fits the model's window on the host: no host here.
        cls._real_fit = (eng.loaded_instances, eng.fit_model)
        eng.loaded_instances = lambda *a, **k: [("m", 8192)]
        eng.fit_model = lambda *a, **k: (8192, "")
        cls.app = studio_chat.Chat()
        for _ in range(15):
            cls.app.update()

    @classmethod
    def tearDownClass(cls):
        # `_quit`, not `destroy`: the app's own shutdown stands the timers down
        # first. A bare destroy left them armed, and the run ended with Tk
        # complaining about "invalid command name ..._drain".
        cls.app._quit()
        eng.installed_apps = cls._real_installed
        eng.loaded_instances, eng.fit_model = cls._real_fit
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

    def test_the_bridges_row_counts_every_bridge_chats_included(self):
        """The chat tab's bridge runs in process, but it is a bridge with
        tools: the row counts it, and closing the tab drops the count."""
        self.app._sync_bridges()
        _lead, lbl = self.app.conn["bridges"]
        bridges = [i for i in self.app.order if eng.TABS_BY_ID[i].bridged]
        self.assertIn(eng.CHAT.id, bridges)
        self.assertIn(str(len(bridges)), lbl.cget("text"))
        self.app._close_tab(eng.CHAT.id)
        self.app._sync_bridges()
        self.assertIn(str(len(bridges) - 1), lbl.cget("text"))

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

    def test_tool_calls_fold_to_their_name_and_open_on_the_script(self):
        """A call is one row: the tool's name to read, the call as the model
        made it - the whole script - and its result behind a click."""
        sid = self.app.order[0]
        s = self.app.sessions[sid]
        view = s.view
        self.app._on_new()
        script = "\n".join("var layer%d = comp.layer(%d);" % (i, i) for i in range(30))
        self.app._handle("tool", s.event_id, {"name": "run_jsx", "via": None,
                                              "arguments": {"code": script, "timeout": 60}})
        self.app._handle("tool_result", s.event_id, {"name": "run_jsx", "status": "ok",
                                                     "text": '{"layer": 1}'})
        self.app.update()
        body = view.get("1.0", "end")
        self.assertIn("run_jsx", body)
        self.assertIn(script, body)                      # entire, not clipped
        self.assertIn("result: {\"layer\": 1}", body)
        self.assertNotIn("\u2026", body)                  # the ellipsis left with the result
        n = s.call_seq
        # Folded: the body is elided, the header is not, and the header is clickable.
        self.assertEqual(str(view.tag_cget("body:%d" % n, "elide")), "1")
        head = view.tag_ranges("call:%d" % n)[0]
        self.assertIn("call_head", view.tag_names(head))
        self.assertIsNotNone(view.bbox(head))
        self.assertIsNone(view.bbox(view.tag_ranges("body:%d" % n)[0]))
        self.assertEqual(view.get(view.tag_ranges("mark:%d" % n)[0]), self.app.g["closed"])
        # Opened: the script is on screen and the chevron turned.
        self.app._toggle_call(view, n)
        self.app.update()
        self.assertEqual(str(view.tag_cget("body:%d" % n, "elide")), "0")
        self.assertIsNotNone(view.bbox(view.tag_ranges("body:%d" % n)[0]))
        self.assertEqual(view.get(view.tag_ranges("mark:%d" % n)[0]), self.app.g["open"])
        self.app._toggle_call(view, n)
        self.assertEqual(str(view.tag_cget("body:%d" % n, "elide")), "1")

    def test_a_failed_call_says_so_on_its_row_and_a_step_sits_in(self):
        sid = self.app.order[0]
        s = self.app.sessions[sid]
        view = s.view
        self.app._on_new()
        self.app._handle("tool", s.event_id, {"name": "timeline", "via": None,
                                              "arguments": {"action": "add_marker", "params": {"frame": 25}}})
        self.app._handle("tool_result", s.event_id, {"name": "timeline", "status": "error",
                                                     "text": "TOOL ERROR: no such frame"})
        self.app._handle("tool", s.event_id, {"name": "set_text", "via": "lower_third",
                                              "arguments": {"layerId": 4, "text": "Producer"}})
        self.app._handle("tool_result", s.event_id, {"name": "set_text", "status": "ok", "text": "ok"})
        self.app._handle("tool", s.event_id, {"name": "get_comp", "via": None, "arguments": {}})
        self.app._handle("tool_result", s.event_id, {"name": "get_comp", "status": "skipped",
                                                     "text": "Cancelled before dispatch"})
        self.app.update()
        failed = view.tag_ranges("call_failed")
        self.assertEqual(view.get(failed[0], failed[1]), " failed")
        self.assertEqual(view.get(failed[2], failed[3]), " not run")
        # The compound tool's header names its action; the step is indented.
        first = view.tag_ranges("call:%d" % (s.call_seq - 2))[0]
        self.assertIn("timeline add_marker", view.get(first, str(first) + " lineend"))
        step = view.tag_ranges("call:%d" % (s.call_seq - 1))[0]
        self.assertIn("call_step", view.tag_names(step))
        self.assertNotIn("call_step", view.tag_names(first))
        # New chat takes the rows and their tags with it.
        self.app._on_new()
        self.assertFalse([t for t in view.tag_names() if t.startswith(("call:", "body:", "mark:"))])
        self.assertEqual(s.call_seq, 0)
        self.assertEqual(s.open_calls, {})

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

    def test_a_chat_tab_is_ready_without_starting_a_subprocess(self):
        """Its bridge is in process - a Loopback, no pipes - and its tools are
        the research bridge's; still warmed against its own prompt prefix and
        the same tool list a real message uses, which is the whole reason the
        first reply is quick."""
        import studio_mcp
        import studio_tasks as tasks
        warmed = []

        class OneReply:
            model = "shared"

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
        self.assertIsInstance(s.mcp, studio_mcp.Loopback)
        self.assertEqual({t["function"]["name"] for t in s.tools}, eng.CHAT.tool_names())
        self.assertIsNotNone(s.library)                # it has tools to make tools from
        self.assertEqual(len(warmed), 1)
        messages, tools = warmed[0]
        self.assertEqual(tools, tasks.inference_tools(s.tools, s.library))
        self.assertIn("studio_task_update", [t["function"]["name"] for t in tools])
        self.assertEqual(messages[0]["content"], eng.CHAT.chat_prompt())
        self.assertEqual(s.bridge[0], "ok")
        self.assertIn("5 tools", s.bridge[1])
        body = s.view.get("1.0", "end")
        self.assertNotIn("Bridge connected", body)
        self.assertIn("within reach", body)
        s.close()
        self.assertIsNone(s.mcp)

    def test_a_booting_tab_fits_the_models_window_and_warms_up_again_after_a_reload(self):
        """Before the warm-up the model is loaded with a window from an
        estimate; the warm-up's reply carries the exact prefix cost, and a
        window still short is reloaded on that and the warm-up paid once
        more - the reload threw the host's prefix cache away."""
        fits, warmed = [], []

        def fit(host, model, prompt_tokens, timeout=600, exact=True):
            fits.append((model, prompt_tokens, exact))
            if len(fits) == 2:
                return 16384, "Reloaded shared with a 16,384-token context window: test"
            return 8192, ""

        class Counts:
            model, base_url = "shared", "http://h:1234/v1"

            def chat(self, messages, tools, max_tokens=None):
                warmed.append(len(tools))
                return {"choices": [{"message": {"content": "ready"}}],
                        "usage": {"prompt_tokens": 7707}}

        real_fit, eng.fit_model = eng.fit_model, fit
        real_llm, self.app.llm = self.app.llm, Counts()
        self.app.host_ready.set()
        try:
            s = self.app.sessions[eng.CHAT.id]
            self.app._boot_session(s)
        finally:
            eng.fit_model, self.app.llm = real_fit, real_llm
        self.app._drain()
        self.app.update()
        self.assertTrue(s.ready)
        self.assertEqual([m for m, _, _ in fits], ["shared", "shared"])
        self.assertIsInstance(fits[0][1], int)          # the estimate, before the warm-up
        self.assertFalse(fits[0][2])                    # and said to be one
        self.assertEqual(fits[1][1:], (7707, True))     # the host's count, after it
        self.assertEqual(len(warmed), 2)
        self.assertIn("Reloaded shared with a 16,384-token context window",
                      s.view.get("1.0", "end"))

    def test_no_reload_while_another_tab_is_mid_request(self):
        """A reload cuts off whatever the host is answering. With another
        tab busy the tab gets the advice instead, and is fitted on its next
        boot."""
        touched = []
        real_fit, eng.fit_model = eng.fit_model, lambda *a, **k: touched.append(a) or (8192, "")
        real_window = eng.context_window
        eng.context_window = lambda host, model, timeout=5: (8192, 32768)

        class Shared:
            model, base_url = "shared", "http://h:1234/v1"
        other = self.app.sessions[eng.APPS[0].id]
        other.busy = True
        s = self.app.sessions[eng.CHAT.id]
        real_llm, s.llm = s.llm, Shared()
        try:
            reloaded, note = self.app._fit(s, 7707)
        finally:
            other.busy = False
            s.llm = real_llm
            eng.fit_model, eng.context_window = real_fit, real_window
        self.assertFalse(reloaded)
        self.assertEqual(touched, [])
        self.assertIn("lms load shared --context-length 16384", note)

    def test_the_comfyui_tab_gets_its_own_small_model_and_the_rest_share(self):
        """One host, but not one model: the ComfyUI tab drives a small model
        when the host serves one, so the diffusion model has the GPU. Other
        tabs keep the shared handle - a new LLM object per tab would be a new
        cached prefix per tab for no reason."""
        real_llm, real_ids = self.app.llm, self.app.model_ids
        self.app.llm = eng.LLM(self.app.host, "big-30b", temperature=0.3, timeout=99)
        spec, real_models = eng.APPS_BY_ID["comfyui"], eng.APPS_BY_ID["comfyui"].models
        spec.models, small = ["small-1.7b"], "small-1.7b"
        try:
            comfy, ae = self.app.sessions["comfyui"], self.app.sessions[eng.APPS[0].id]
            self.app.model_ids = ["big-30b", small]
            chosen = self.app._llm_for(comfy)
            self.assertEqual(chosen.model, small)
            self.assertEqual((chosen.temperature, chosen.timeout), (0.3, 99))
            self.assertIs(self.app._llm_for(ae), self.app.llm)
            self.app._drain()
            self.app.update()
            self.assertIn("Model for this tab: " + small, comfy.view.get("1.0", "end"))
            self.assertNotIn("Model for this tab", ae.view.get("1.0", "end"))
            # the small model gone from the host: the tab still opens, on the shared one
            self.app.model_ids = ["big-30b"]
            self.assertIs(self.app._llm_for(comfy), self.app.llm)
        finally:
            self.app.llm, self.app.model_ids = real_llm, real_ids
            spec.models = real_models

    def test_a_tab_with_its_own_model_gets_its_own_draft(self):
        """The draft pairs with the executing model, so a tab that departs
        from the shared model resolves a draft for its own - and a 1.7B gets
        none unless the pin insists."""
        real_llm, real_ids = self.app.llm, self.app.model_ids
        pin = os.environ.pop("STUDIO_DRAFT_MODEL", None)
        big, small = "qwen3-coder-30b-a3b-instruct", "qwen3-1.7b"
        spec, real_models = eng.APPS_BY_ID["comfyui"], eng.APPS_BY_ID["comfyui"].models
        spec.models = [small]
        self.app.llm = eng.LLM(self.app.host, big, draft="qwen3-0.6b")
        self.app.model_ids = [big, small, "qwen3-0.6b"]
        try:
            comfy = self.app.sessions["comfyui"]
            chosen = self.app._llm_for(comfy)
            self.assertEqual((chosen.model, chosen.draft), (small, None))
            os.environ["STUDIO_DRAFT_MODEL"] = "qwen3-0.6b"
            self.assertEqual(self.app._llm_for(comfy).draft, "qwen3-0.6b")
            os.environ["STUDIO_DRAFT_MODEL"] = "ghost"
            chosen = self.app._llm_for(comfy)
            self.assertIsNone(chosen.draft)
            self.app._drain()
            self.app.update()
            self.assertIn("STUDIO_DRAFT_MODEL names ghost", comfy.view.get("1.0", "end"))
        finally:
            self.app.llm, self.app.model_ids = real_llm, real_ids
            spec.models = real_models
            os.environ.pop("STUDIO_DRAFT_MODEL", None)
            if pin is not None:
                os.environ["STUDIO_DRAFT_MODEL"] = pin

    def test_the_inference_row_names_the_draft_and_its_score(self):
        """Speculative decoding can backfire - a draft the model keeps
        rejecting is slower than none - so the row says which draft runs and,
        once the host reports it, how much of its work is kept."""
        real_llm, real_ids = self.app.llm, self.app.model_ids
        lbl = self.app.conn["host"][1]
        self.app.llm = eng.LLM(self.app.host, "big-30b", draft="qwen3-0.6b")
        self.app.model_ids = ["big-30b", "qwen3-0.6b"]
        try:
            self.app._host_line("ok", "sees: eyes-vl")
            self.app._drain()
            lines = lbl.cget("text").split("\n")
            self.assertEqual(lines[1:], ["2 models", "big-30b", "sees: eyes-vl", "draft: qwen3-0.6b"])
            self.app.llm.drafted = (78, 100)
            self.app._host_line()                # role and last line kept
            self.app._drain()
            lines = lbl.cget("text").split("\n")
            self.assertEqual(lines[-2:], ["sees: eyes-vl", "draft: qwen3-0.6b · 78%"])
            self.assertTrue(all(len(line) <= 24 for line in lines), lines)
            self.app.llm.draft = None
            self.app._host_line()
            self.app._drain()
            self.assertNotIn("draft", lbl.cget("text"))
            self.assertIn("sees: eyes-vl", lbl.cget("text"))
        finally:
            self.app.llm, self.app.model_ids = real_llm, real_ids

    def test_a_refused_draft_is_said_once_in_the_tab_it_happened_in(self):
        real_llm = self.app.llm
        s = self.app.sessions[eng.APPS[0].id]
        self.app.llm = eng.LLM(self.app.host, "big-30b", draft=None)
        self.app.llm.draft_note = "The host refused tiny as a draft model for big-30b (HTTP 400 - no); speculative decoding is off for this model."
        try:
            self.app._draft_check(s, self.app.llm)
            self.app._draft_check(s, self.app.llm)
            self.app._drain()
            self.app.update()
            body = s.view.get("1.0", "end")
            self.assertEqual(body.count("The host refused tiny"), 1)
            self.assertIsNone(self.app.llm.draft_note)
        finally:
            self.app.llm = real_llm

    def test_start_says_why_it_does_nothing_in_chat(self):
        """A menu item, always enabled. Silence would read as a bug - and
        there is no app behind this tab to start."""
        s = self.app.sessions[eng.CHAT.id]
        self.app._select(s.id)
        was_ready, s.ready = s.ready, True
        try:
            self.app._on_fix()
        finally:
            s.ready = was_ready
        body = s.view.get("1.0", "end")
        self.assertIn("no app to start", body)
        self.assertFalse(s.busy)

    def test_a_remote_app_is_checked_not_started(self):
        """ComfyUI runs on the LLM PC. The button re-probes and says where to
        start it; nothing hunts for an .exe, and the status says 'reachable'."""
        s = self.app.sessions["comfyui"]
        self.app._select(s.id)
        self.app.update()
        self.assertEqual(self.app.btn_fix.cget("text"), "Check ComfyUI")
        real_running = eng.AppSpec.running
        eng.AppSpec.running = lambda self: False
        try:
            self.app._fix(s)
            self.app._drain()
        finally:
            eng.AppSpec.running = real_running
        body = s.view.get("1.0", "end")
        self.assertIn("not answering", body)
        self.assertIn("--listen", body)
        self.assertNotIn("Launching", body)
        self.assertEqual(s.status[0], "ComfyUI is not reachable")
        self.assertTrue(s.status[2], "the button stays, to check again")

    def test_the_error_log_lives_beside_the_settings_not_in_the_source_tree(self):
        """Every test run that exercised _guard appended its deliberate
        HostUnreachable tracebacks to the source tree's log, beside the
        user's real ones. It goes where STUDIO_SETTINGS puts everything."""
        path = self.mod.error_log_path()
        self.assertEqual(os.path.dirname(path), self.dir)
        self.assertEqual(os.path.basename(path), self.mod.ERROR_LOG)
        self.app._log("Traceback: a test's own")
        with open(path, encoding="utf-8") as f:
            self.assertIn("a test's own", f.read())
        self.assertFalse(os.path.exists(os.path.join(self.mod.HERE, "tests", self.mod.ERROR_LOG)))

    def test_a_container_app_is_started_not_checked(self):
        """OpenCode runs here, so its button starts it - and when Docker is
        missing the reason reaches the transcript as prose, not a traceback."""
        s = self.app.sessions["opencode"]
        self.app._select(s.id)
        self.app.update()
        self.assertEqual(self.app.btn_fix.cget("text"), "Start OpenCode")
        real_running, real_exe = eng.AppSpec.running, eng.docker_exe
        eng.AppSpec.running = lambda self: False
        eng.docker_exe = lambda: None
        logged = []
        self.app._log = logged.append
        try:
            self.app._guard(s.event_id, self.app._fix, s)
            self.app._drain()
        finally:
            eng.AppSpec.running, eng.docker_exe = real_running, real_exe
            del self.app._log
        body = s.view.get("1.0", "end")
        self.assertIn("Launching OpenCode", body)
        self.assertIn("Docker Desktop is not installed", body)
        self.assertNotIn("RuntimeError", body)
        self.assertNotIn("Traceback", body)
        # A refusal is not a crash: nothing for the error log, and the header
        # falls back from "launching" so the button comes back.
        self.assertEqual(logged, [])
        self.assertEqual(s.status[0], "OpenCode is not running")
        self.assertTrue(s.status[2], "the button stays, to try again")

    def _host_state(self):
        """Everything a probe writes, so a test can put it back."""
        a = self.app
        return (a.llm, a.model_ids, a.vision, a.vision_note, a.draft_note,
                a.host_role, a.host_last, a.host_ok)

    def _restore_host(self, state):
        a = self.app
        (a.llm, a.model_ids, a.vision, a.vision_note, a.draft_note,
         a.host_role, a.host_last, a.host_ok) = state
        if a.host_timer is not None:      # no quiet probe firing into a later test
            a.after_cancel(a.host_timer)
            a.host_timer = None

    def _stuck(self, s):
        """Boot a tab with no model: how every tab ends up when the probe at
        startup times out."""
        self.app.host_ready.set()
        self.app._boot_session(s)
        self.app._drain()
        self.app.update()

    def test_a_tab_stuck_without_the_host_offers_connect(self):
        """The startup probe is one call with a short timeout, and a PC still
        waking fails it. That used to leave every tab at 'no inference host'
        with no button, and reopening the window as the only way on."""
        state = self._host_state()
        s = self.app.sessions[eng.CHAT.id]
        self.app._select(s.id)
        self.app.llm = None
        try:
            self._stuck(s)
            self.assertFalse(s.ready)
            self.assertTrue(s.host_down)
            self.assertEqual(s.status, ("no inference host - trying again", "err", True))
            self.assertEqual(self.app.btn_fix.cget("text"), "Connect")
            self.assertTrue(self.app.btn_fix.winfo_ismapped())
            self.assertIsNotNone(self.app.host_timer, "a tab stuck is a retry scheduled")
        finally:
            s.host_down = False
            self._restore_host(state)

    def test_connect_probes_again_and_starts_the_tab_you_are_looking_at(self):
        """Connect is the probe again. Failing, the tab keeps its button and
        the transcript says what to check; answering, the stuck tabs are
        unstuck - the active one starts now, a ready one that lost the host
        mid-conversation is ready again, the rest wait to be selected."""
        state = self._host_state()
        real_probe, real_alive = eng.probe_models, eng.host_alive
        eng.host_alive = lambda host, timeout=3: None
        a, b, c = (self.app.sessions[i] for i in self.app.order[:3])
        ensured = []
        self.app.llm = None
        try:
            self.app._select(a.id)
            self._stuck(a)
            self._stuck(b)
            c.ready, c.host_down = True, True   # lost the host after a turn
            c.status = ("inference host unreachable", "err", True)
            self.app._ensure = ensured.append   # after _select, which ensures too

            eng.probe_models = lambda host, timeout=8: (False, None, [], [], "timed out")
            self.app._connect_host()
            self.assertEqual(a.status[0],
                             "connecting to the inference host" + self.mod.ELLIPSIS)
            self.assertFalse(self.app.btn_fix.winfo_ismapped(), "no second click meanwhile")
            self.real_boot_host(self.app, False)   # what _connect_host spawned
            self.app._drain()
            self.app.update()
            self.assertIsNone(self.app.llm)
            self.assertEqual(a.status, ("no inference host - trying again", "err", True))
            self.assertEqual(self.app.btn_fix.cget("text"), "Connect")
            self.assertTrue(a.host_down and b.host_down and c.host_down)
            body = a.view.get("1.0", "end")
            self.assertIn("Cannot reach the inference host", body)
            self.assertIn("tries again every 30 seconds", body)
            self.assertIn("Connect", body)
            self.assertIn("unreachable", self.app.conn["host"][1].cget("text"))
            self.assertEqual(ensured, [])
            self.assertIsNotNone(self.app.host_timer)

            eng.probe_models = lambda host, timeout=8: (True, "m1", ["m1", "m2"], [], None)
            self.app._connect_host()
            self.real_boot_host(self.app, False)
            self.app._drain()
            self.app.update()
            self.assertEqual(self.app.llm.model, "m1")
            self.assertEqual(ensured, [a], "the active tab starts; the rest when selected")
            self.assertFalse(a.host_down or b.host_down or c.host_down)
            self.assertEqual(a.status, ("not started", "muted", False))
            self.assertEqual(b.status, ("not started", "muted", False))
            self.assertEqual(b.bridge[0], "faint", "no longer 'no model' on its dot")
            self.assertEqual(c.status, ("ready", "muted", False))
            self.assertFalse(self.app.btn_fix.winfo_ismapped())
            self.assertIn("Connected to the inference host", a.view.get("1.0", "end"))
            row = self.app.conn["host"][1].cget("text")
            self.assertIn("m1", row)
            self.assertIn("2 models", row)
            self.assertNotIn("unreachable", row)
            self.assertFalse(self.app.host_booting)
        finally:
            eng.probe_models, eng.host_alive = real_probe, real_alive
            self.app.__dict__.pop("_ensure", None)
            for s in (a, b, c):
                s.ready, s.host_down = False, False
            self._restore_host(state)

    def test_the_window_keeps_trying_while_the_host_is_down(self):
        """The LLM PC drops off on a timer. A failed probe schedules a quiet
        one - the row and the status move, the transcript does not fill with
        the same paragraph every half minute - and the loop ends by itself
        once nothing is waiting on the host."""
        state = self._host_state()
        real_probe, real_alive = eng.probe_models, eng.host_alive
        eng.probe_models = lambda host, timeout=8: (False, None, [], [], "timed out")
        eng.host_alive = lambda host, timeout=3: None
        spawned = []
        self.app._spawn = lambda *a: spawned.append(a)
        s = self.app.cur()
        self.app.llm = None
        try:
            self.app._host_probed(False)
            first = self.app.host_timer
            self.assertIsNotNone(first)
            self.app._host_probed(False)
            self.assertEqual(self.app.host_timer, first, "one pending probe, not two")

            self.app.host_timer = None    # as the timer firing does
            self.app._retry_host()
            self.assertEqual(spawned, [(None, self.app._boot_host, False, True)])
            self.assertIsNone(self.app.host_timer, "the probe itself re-arms, on failure")

            s.host_down = True
            before = s.view.get("1.0", "end")
            self.real_boot_host(self.app, False, True)   # what the retry spawned
            self.app._drain()
            self.app.update()
            self.assertEqual(s.view.get("1.0", "end"), before, "a quiet probe says nothing")
            self.assertEqual(s.status, ("no inference host - trying again", "err", True))
            self.assertIn("unreachable", self.app.conn["host"][1].cget("text"))
            self.assertIsNotNone(self.app.host_timer, "...and the next is scheduled")

            spawned.clear()
            self.app.host_timer = None
            self.app.host_booting = True  # Connect is already probing
            self.app._retry_host()
            self.assertEqual(spawned, [])
            self.app.host_booting = False
            self.app.llm = eng.LLM(self.app.host, "m1")
            s.host_down = False           # nothing waits on the host any more
            self.app._retry_host()
            self.assertEqual(spawned, [])
            self.assertIsNone(self.app.host_timer)
        finally:
            eng.probe_models, eng.host_alive = real_probe, real_alive
            self.app.__dict__.pop("_spawn", None)
            s.host_down = False
            self._restore_host(state)

    def test_unreachable_is_explained_by_which_half_is_down(self):
        """A locked PC keeps serving; a sleeping one does not; a restarted
        one comes back without LM Studio's server. The probe cannot tell a
        PC that is gone from one that is up with nothing listening - the
        firewall drops both - so the tailnet is asked and the paragraph says
        which, and what to change over there."""
        real_alive = eng.host_alive
        try:
            eng.host_alive = lambda host, timeout=3: True
            up = self.app._explain_unreachable("timed out")
            self.assertIn("LM Studio's server is not answering", up)
            self.assertIn("service on login", up)
            self.assertIn("A locked screen does not stop it", up)
            eng.host_alive = lambda host, timeout=3: False
            gone = self.app._explain_unreachable("timed out")
            self.assertIn("not answering at all", gone)
            self.assertIn("standby-timeout-ac 0", gone)
            eng.host_alive = lambda host, timeout=3: None
            unknown = self.app._explain_unreachable("timed out")
            self.assertIn("Cannot reach the inference host", unknown)
            for text in (up, gone, unknown):
                self.assertIn("tries again every 30 seconds", text)
                self.assertIn("Connect", text)
                self.assertTrue(text.endswith("(timed out)"))
        finally:
            eng.host_alive = real_alive

    def test_a_second_probe_keeps_the_model_the_tabs_booted_on(self):
        """Connect again with the host fine: the same model keeps its client,
        so the tabs that hold it stay in step with the Inference row."""
        state = self._host_state()
        real_probe = eng.probe_models
        self.app.llm = eng.LLM(self.app.host, "m1")
        try:
            eng.probe_models = lambda host, timeout=8: (True, "m1", ["m1"], [], None)
            before = self.app.llm
            self.real_boot_host(self.app, False)
            self.app._drain()
            self.assertIs(self.app.llm, before)
            eng.probe_models = lambda host, timeout=8: (True, "m2", ["m2"], [], None)
            self.real_boot_host(self.app, False)
            self.app._drain()
            self.assertIsNot(self.app.llm, before)
            self.assertEqual(self.app.llm.model, "m2")
        finally:
            eng.probe_models = real_probe
            self._restore_host(state)

    def test_a_turn_that_loses_the_host_says_so_and_offers_connect(self):
        """Mid-conversation the host can go away too. The row goes red, the
        header offers Connect, and the next request that gets through - a
        Connect or just sending again - puts both back."""
        from test_tasks import FakeLLM, answer
        state = self._host_state()
        s = self.app.cur()
        saved = s.messages, s.record, s.ready
        self.app.llm = eng.LLM(self.app.host, "m1")
        self.app.model_ids = ["m1"]

        class Gone:
            model = "m1"

            def stream(self, messages, tools, on_text):
                raise eng.HostUnreachable("cannot reach inference host h (timed out)")

        class Back(FakeLLM):
            model = "m1"                  # the Inference row names it
        try:
            self.app._host_healthy("ok", "sees: eyes")
            self.app._drain()
            s.reset()
            s.ready = True
            s.record.briefs.append("hello")
            s.messages.append({"role": "user", "content": "hello"})
            self.app.llm = Gone()
            self.app._guard(s.event_id, self.app._turn, s)
            self.app._drain()
            self.app.update()
            self.assertTrue(s.host_down)
            self.assertEqual(s.status, ("inference host unreachable - trying again", "err", True))
            self.assertEqual(self.app.btn_fix.cget("text"), "Connect")
            self.assertIsNotNone(self.app.host_timer, "the window will try by itself")
            self.assertTrue(self.app.btn_fix.winfo_ismapped())
            self.assertIn("cannot reach inference host", s.view.get("1.0", "end"))
            self.assertNotIn("Traceback", s.view.get("1.0", "end"))
            self.assertIn("unreachable", self.app.conn["host"][1].cget("text"))
            self.assertEqual(self.app.dot_role[self.app.conn["host"][0]], "err")

            # Not a correction - "again" reads as one and would start the
            # reflection, one more request the fake has no answer for.
            s.record.briefs.append("once more please")
            s.messages.append({"role": "user", "content": "once more please"})
            self.app.llm = Back([answer(text="Back.")])
            self.app._guard(s.event_id, self.app._turn, s)
            self.app._drain()
            self.app.update()
            self.assertFalse(s.host_down)
            self.assertEqual(s.status[0], "ready")
            self.assertFalse(self.app.btn_fix.winfo_ismapped())
            self.assertIn("sees: eyes", self.app.conn["host"][1].cget("text"))
            self.assertEqual(self.app.dot_role[self.app.conn["host"][0]], "ok")
        finally:
            s.messages, s.record, s.ready = saved
            s.host_down = False
            self._restore_host(state)

    def test_the_inference_row_and_the_bridges_menu_offer_connect(self):
        """The row is a menu like the Bridges row; its one action is Connect,
        worded for the state, and disabled while a probe runs."""
        state = self._host_state()
        try:
            self.app.llm = None
            m = self.app._menu_host()
            self.assertEqual(m.entrycget(0, "label"), "Connect to the inference host")
            self.assertEqual(str(m.entrycget(0, "state")), "normal")
            self.app.llm = eng.LLM(self.app.host, "m1")
            m = self.app._menu_host()
            self.assertEqual(m.entrycget(0, "label"), "Connect again")
            self.app.host_booting = True
            m = self.app._menu_host()
            self.assertIn("Connecting", m.entrycget(0, "label"))
            self.assertEqual(str(m.entrycget(0, "state")), "disabled")
            self.app._connect_host()              # a probe already running: nothing
        finally:
            self.app.host_booting = False
            self._restore_host(state)
        labels = [self.app.m_bridge.entrycget(i, "label")
                  for i in range(self.app.m_bridge.index("end") + 1)
                  if self.app.m_bridge.type(i) == "command"]
        self.assertIn("Connect to the inference host", labels)

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
        rows = [a["name"] for a in self.app.detected if not a["remote"]]
        if len(rows) < 2:
            self.skipTest("needs two creative apps")
        self.app._pin_app(rows[-1])
        self.app.update()
        self.assertEqual(self._sidebar_names()[0], self.mod.clip(rows[-1], self.mod.APP_NAME_CHARS))
        self.app._pin_app(rows[-1])          # same control unpins
        self.assertEqual(self.app.prefs.get("pinned"), [])

    def test_remote_apps_sit_under_their_own_heading(self):
        """ComfyUI runs on the LLM PC: its rows are drawn last, below a second
        heading, and hiding them takes the heading with them."""
        import tkinter
        remote = [a["name"] for a in self.app.detected if a["remote"]]
        if not remote:
            self.skipTest("no remote app in the registry")
        names = self._sidebar_names()
        clipped = [self.mod.clip(n, self.mod.APP_NAME_CHARS) for n in remote]
        self.assertEqual(names[-len(remote):], clipped)

        def caps():
            return [w.cget("text") for w in self.app.applist.winfo_children()
                    if isinstance(w, tkinter.Label)]
        self.assertIn("ON %s" % self.mod.LLM_PC, caps())
        for n in remote:
            self.app._hide_app(n)
        self.assertNotIn("ON %s" % self.mod.LLM_PC, caps())
        for n in remote:
            self.app._show_app(n)

    def _sidebar_names(self):
        """The visible app list: each row's title, in the order it is drawn.

        Searched rather than walked by a fixed depth: a row is a canvas with
        its content on a frame placed inside it, so the name box is a
        grandchild now and would be a great-grandchild the next time the row
        gains a wrapper. The name box is the first frame holding two labels -
        the pin and hide glyphs each sit in a one-label slot of their own."""
        import tkinter

        def box_of(widget):
            for child in widget.winfo_children():
                if isinstance(child, tkinter.Frame):
                    labels = [w for w in child.winfo_children()
                              if isinstance(w, tkinter.Label)]
                    if len(labels) >= 2:
                        return labels[0]
                found = box_of(child)
                if found is not None:
                    return found
            return None

        out = []
        for row in self.app.applist.winfo_children():
            title = box_of(row)
            if title is not None:
                out.append(title.cget("text"))
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

    # Tk names its one documented override hook like any other method, so the
    # blanket check cannot tell it from an accident. Overriding it is the
    # supported way to catch a failing callback; each entry here is a decision,
    # not an oversight, and anything not listed is still a clash.
    DELIBERATE_TK_OVERRIDES = {"report_callback_exception"}

    def test_no_method_shadows_tkinter_internals(self):
        import tkinter
        clashes = [n for n in vars(self.mod.Chat)
                   if not n.startswith("__")
                   and n not in self.DELIBERATE_TK_OVERRIDES
                   and (hasattr(tkinter.Misc, n) or hasattr(tkinter.Tk, n))]
        self.assertEqual(clashes, [], "shadowing Tk internals breaks the widget")

    def test_the_deliberate_overrides_are_really_tk_hooks(self):
        """The allowlist must not become a place to hide a genuine clash: a
        name in it that Tk does not define is a typo waiting to do nothing."""
        import tkinter
        for name in self.DELIBERATE_TK_OVERRIDES:
            with self.subTest(name=name):
                self.assertTrue(hasattr(tkinter.Misc, name) or hasattr(tkinter.Tk, name))
                self.assertIn(name, vars(self.mod.Chat))

    def test_the_pump_stays_a_single_timer_however_it_is_entered(self):
        """Every hand-called `_drain` in this file used to arm a tick beside
        the live one; they piled up and Tk named each orphan on the way out
        ("invalid command name ..._drain"). One in, one armed."""
        self.app._drain()
        first = self.app.drain_timer
        self.app._drain()
        pending = self.app.tk.splitlist(self.app.tk.call("after", "info"))
        self.assertIn(self.app.drain_timer, pending)
        self.assertNotIn(first, pending)

    def test_a_failing_event_leaves_the_pump_running(self):
        """The queue pump is the only path from the worker threads to the UI.
        An event whose handling raised used to escape past the reschedule, so
        the pump stopped for good: every tab went quiet at once - no tokens,
        no status, no idle - while the threads kept filling a queue nobody
        read. `pythonw.exe` has no stderr, so nothing said why."""
        sid, seen, real = self.app.order[0], [], self.app._handle

        def explode(kind, s, payload):
            if payload == "boom":
                raise RuntimeError("a widget said no")
            seen.append(payload)
            return real(kind, s, payload)

        self.app._handle = explode
        try:
            self.app.q.put(("sys", sid, "boom"))
            self.app.q.put(("sys", sid, "delivered-after-the-failure"))
            self.app._drain()
        finally:
            self.app._handle = real
        self.assertIn("delivered-after-the-failure", seen)
        self.assertIsNotNone(self.app.drain_timer)

    def test_a_failing_event_is_logged_and_named_in_the_transcript(self):
        """Surviving is not enough - a failure nobody can see is the hobby
        version. It goes to the error log, and the tab says so in words."""
        sid, logged, real = self.app.order[0], [], self.app._handle
        real_log, self.app._log = self.app._log, logged.append

        def explode(kind, s, payload):
            raise RuntimeError("a widget said no")

        self.app._handle = explode
        try:
            self.app.q.put(("sys", sid, "boom"))
            self.app._drain()
        finally:
            self.app._handle, self.app._log = real, real_log
        self.assertTrue(any("a widget said no" in t for t in logged), logged)
        self.assertIn("a widget said no",
                      self.app.cur().view.get("1.0", "end"))

    def test_a_failing_callback_reaches_the_same_place(self):
        """Tk's hook for a menu command or button that raised. Its default
        prints to a stderr the shortcut does not give the app."""
        logged, real_log = [], self.app._log
        self.app._log = logged.append
        try:
            self.app.report_callback_exception(
                RuntimeError, RuntimeError("the button said no"), None)
        finally:
            self.app._log = real_log
        self.assertTrue(any("the button said no" in t for t in logged), logged)

    def _save_a_task(self, s, brief, steps=1, broken=False):
        import studio_tasks as tasks
        folder = os.path.dirname(self.app._task_path(s))
        os.makedirs(folder, exist_ok=True)
        self.addCleanup(shutil.rmtree, folder, True)
        if broken:
            path = os.path.join(folder, "0" * 32 + ".json")
            with open(path, "w") as f:
                f.write("{not json")
            return path
        record = tasks.TaskRecord()
        record.app_id = s.id
        record.briefs = [brief]
        record.journal = [{"call": "x", "status": "ok"} for _ in range(steps)]
        path = os.path.join(folder, record.id + ".json")
        record.save(path, [{"role": "system", "content": "s"},
                           {"role": "user", "content": brief}])
        return path

    def test_saved_tasks_lists_what_was_asked_not_hex_names(self):
        """The old file dialog showed a folder of 32-character names and no
        way to tell which was which, so in practice nothing was resumed."""
        s = self.app.cur()
        self._save_a_task(s, "make a pic of a duck", steps=13)
        try:
            self.app._resume_task()
            body = self.app.tasks_view.get("1.0", "end")
            self.assertIn("make a pic of a duck", body)
            self.assertIn("13 steps", body)
        finally:
            self.app.windows[("tasks", s.id)].destroy()

    def test_the_conversation_already_on_screen_is_not_offered(self):
        """Resuming it would replace it with a checkpoint of itself."""
        s = self.app.cur()
        s.record.briefs = ["the one already open"]
        s.record.save(self.app._task_path(s), s.messages)
        self.addCleanup(shutil.rmtree,
                        os.path.dirname(self.app._task_path(s)), True)
        try:
            self.app._resume_task()
            self.assertNotIn("the one already open",
                             self.app.tasks_view.get("1.0", "end"))
        finally:
            self.app.windows[("tasks", s.id)].destroy()
            s.record.briefs = []

    def test_a_task_that_will_not_parse_is_shown_and_cannot_be_resumed(self):
        """Hiding it is how someone comes to believe the app lost their work."""
        s = self.app.cur()
        self._save_a_task(s, "", broken=True)
        try:
            self.app._resume_task()
            view = self.app.tasks_view
            self.assertIn("nothing asked yet", view.get("1.0", "end"))
            labels = [w.cget("text") for w in view.winfo_children()
                      if isinstance(w, (tk.Button, self.mod.Pill))]
            self.assertEqual(labels, ["delete"])
        finally:
            self.app.windows[("tasks", s.id)].destroy()

    def test_diagnostics_opens_saying_so_and_then_paints_the_report(self):
        """The probe talks to the host, so the window has to open before the
        answer does - a tailnet timeout must not look like a frozen app."""
        import studio_doctor
        fake = [("Inference host", [("Model", "qwen3-coder-30b", "ok"),
                                    ("Context window", "8,192 loaded", "warn")])]
        real = studio_doctor.report
        studio_doctor.report = lambda *a, **k: fake
        try:
            self.app._diagnostics_window()
            self.assertIn("Checking", self.app.diag_view.get("1.0", "end"))
            for _ in range(15):
                self.app.update()
            self.app._paint_diagnostics(fake)     # as the pump would
            body = self.app.diag_view.get("1.0", "end")
            self.assertIn("Inference host", body)
            self.assertIn("qwen3-coder-30b", body)
            self.app._copy_diagnostics()
            self.assertIn("8,192 loaded", self.app.clipboard_get())
        finally:
            studio_doctor.report = real
            self.app.windows[("diagnostics", "")].destroy()

    def test_diagnostics_survives_being_painted_after_it_is_closed(self):
        """The report arrives on the queue; the window it was asked for may
        have been shut in the meantime, and the pump must not care."""
        self.app._diagnostics_window()
        self.app.windows[("diagnostics", "")].destroy()
        self.app.update()
        self.app._paint_diagnostics([("Anything", [("x", "y", "ok")])])

    def test_the_pump_does_not_re_arm_while_the_window_is_closing(self):
        """`_quit` cancels the timers; a tick already in flight must not put
        one back, or `after` fires against a destroyed widget - which is what
        printed "invalid command name ..._drain" over a window already gone."""
        before = self.app.drain_timer
        self.app.closing = True
        try:
            self.app._drain()
            self.assertIs(self.app.drain_timer, before)
        finally:
            self.app.closing = False

    # ------------------------------------------------------------- animation
    # One `after` tick drives everything that moves, for the reason the queue
    # pump is one tick: a timer per animation is a timer per orphan on the way
    # out. These hold the discipline that follows from that.

    def test_the_animator_is_one_timer_armed_only_while_something_moves(self):
        """A window with nothing happening in it must not wake up fourteen
        times a second to redraw nothing - and two animations must not mean
        two timers."""
        self.app.anim.clear()
        self.app._stand_down("anim_timer")
        self.assertIsNone(self.app.anim_timer)
        self.app._animate("a", lambda frame: None)
        first = self.app.anim_timer
        self.assertIsNotNone(first)
        self.app._animate("b", lambda frame: None)
        self.assertIs(self.app.anim_timer, first, "a second animation, no second timer")
        self.app._anim_tick()
        pending = self.app.tk.splitlist(self.app.tk.call("after", "info"))
        self.assertIn(self.app.anim_timer, pending)
        self.assertNotIn(first, pending, "the old tick is stood down, not left behind")
        self.app._unanimate("a")
        self.app._unanimate("b")
        self.app._anim_tick()
        self.assertIsNone(self.app.anim_timer, "nothing left to draw, no tick armed")

    def test_an_animation_paints_immediately_rather_than_next_frame(self):
        """Registering is also a paint: a status that changes must not wait
        ANIM_MS to say so, or every state change reads as lag."""
        seen = []
        self.app._animate("probe", lambda frame: seen.append(frame))
        try:
            self.assertEqual(seen, [self.app.anim_frame])
        finally:
            self.app._unanimate("probe")

    def test_a_one_frame_animation_drops_itself_without_a_tick(self):
        """A `draw` that returns False on its first paint - the wipe at the
        end of its run, an arc already where it was going - is over before the
        timer ever sees it, and must not arm one."""
        self.app.anim.clear()
        self.app._stand_down("anim_timer")
        self.app._animate("once", lambda frame: False)
        self.assertNotIn("once", self.app.anim)
        self.assertIsNone(self.app.anim_timer)

    def test_a_failing_animation_is_dropped_and_the_others_keep_drawing(self):
        """Same rule as the queue pump: one bad frame must not stop the tick.
        A widget destroyed under its own animation is ordinary and silent; an
        animation that raises anything else is a bug and is reported."""
        drawn, logged = [], []
        real_log, self.app._log = self.app._log, logged.append

        def gone(frame):
            raise tk.TclError("invalid command name .!canvas")

        try:
            self.app._animate("gone", gone)
            self.app._animate("broken", lambda frame: 1 / 0)
            self.app._animate("fine", drawn.append)
            self.app._anim_tick()
            self.assertNotIn("gone", self.app.anim)
            self.assertNotIn("broken", self.app.anim)
            self.assertIn("fine", self.app.anim)
            self.assertTrue(drawn)
            self.assertIsNotNone(self.app.anim_timer)
            self.assertTrue(any("ZeroDivisionError" in t for t in logged), logged)
            self.assertFalse(any("TclError" in t for t in logged),
                             "a destroyed widget is not worth a log entry")
        finally:
            self.app._unanimate("fine")
            self.app._log = real_log

    def test_the_animator_does_not_re_arm_while_the_window_is_closing(self):
        before = self.app.anim_timer
        self.app.closing = True
        try:
            self.app._animate("late", lambda frame: None)
            self.assertIs(self.app.anim_timer, before)
        finally:
            self.app.closing = False
            self.app._unanimate("late")

    def test_a_trailing_ellipsis_is_what_makes_a_label_animate(self):
        """The whole protocol: text ending in one is text describing something
        still happening, and the dots count. Anything else is shown once and
        its animation dropped, which is how a finished state stops moving."""
        shown = []
        self.app._ellipsis("probe", "working" + self.mod.ELLIPSIS, shown.append)
        self.assertIn("probe", self.app.anim)
        for _ in range(3 * self.mod.ELLIPSIS_FRAMES + 1):
            self.app._anim_tick()
        self.assertTrue(all(t.startswith("working.") for t in shown), shown[:4])
        self.assertEqual({t[len("working"):] for t in shown}, {".", "..", "..."})
        self.app._ellipsis("probe", "ready", shown.append)
        self.assertNotIn("probe", self.app.anim)
        self.assertEqual(shown[-1], "ready")

    def test_every_status_the_window_shows_is_a_role_and_never_a_bare_ellipsis(self):
        """The header renders the dots itself, so the literal character must
        never reach the label - and a status still carries a palette role."""
        for text, role, _fixable in (
                ("working" + self.mod.ELLIPSIS, "warn", False),
                ("ready", "muted", False)):
            with self.subTest(text=text):
                s = self.app.cur()
                s.status = (text, role, False)
                self.app._apply_status()
                self.assertNotIn(self.mod.ELLIPSIS, self.app.lbl_status.cget("text"))
                self.assertEqual(self.app.lbl_status.cget("fg"), self.app.C[role])

    def test_blend_walks_between_two_colours(self):
        """A generator expression per colour closes over the comprehension's
        loop variable, so both ends come out as the second colour and every
        pulse in the window goes flat without an error. It did. This is why
        the channels are built eagerly."""
        blend = self.mod.blend
        self.assertEqual(blend("#000000", "#ffffff", 0.0), "#000000")
        self.assertEqual(blend("#000000", "#ffffff", 1.0), "#ffffff")
        self.assertEqual(blend("#000000", "#ffffff", 0.25), "#404040")
        self.assertEqual(blend("#102030", "#302010", 0.5), "#202020")
        self.assertEqual(blend("#000000", "#ffffff", 4.0), "#ffffff", "clamped")
        self.assertNotEqual(blend("#232321", "#302f2c", 0.5), "#302f2c")

    def test_the_bridges_row_is_a_drawn_arc_and_not_a_font_glyph(self):
        """MDL2's chain link says "two things fastened together", which is not
        what a bridge is; and a glyph cannot show a span going up. The drawn
        arc can, so the link codepoint is gone from the glyph table."""
        self.assertNotIn("link", self.app.g)
        lead, _lbl = self.app.conn["bridges"]
        self.assertIsInstance(lead, tk.Canvas)
        self.assertIn(lead, self.app.arcs)
        self.assertTrue(lead.find_all(), "the arc draws itself")

    def test_the_arc_draws_itself_across_only_when_the_state_changes(self):
        """One pass on a change, then it settles. There is no state it can sit
        in spinning: a bridge that never starts would otherwise animate for the
        life of the window."""
        lead, _lbl = self.app.conn["bridges"]
        key = ("arc", str(lead))
        self.app._arc_state(lead, "faint")
        self.app._unanimate(key)
        self.app._arc_state(lead, "ok")
        self.assertIn(key, self.app.anim, "a new state draws itself across")
        for _ in range(self.mod.SWEEP_FRAMES + 2):
            self.app._anim_tick()
        self.assertNotIn(key, self.app.anim, "and then settles")
        self.app._arc_state(lead, "ok")
        self.assertNotIn(key, self.app.anim, "the same state again is not a change")

    def test_a_theme_switch_repaints_the_arc_rather_than_reconfiguring_it(self):
        """A canvas that plotted its own palette colours cannot be told a new
        one with config() - same reason the app marks and the dots are
        repainted. A role missing from a palette would be a KeyError mid-switch."""
        lead, _lbl = self.app.conn["bridges"]
        was = self.app.prefs.get("theme")
        try:
            for name in ("light", "dark"):
                self.app._theme(name)
                # The switch re-syncs the row, so the role is whatever the
                # bridges are actually doing - but it is always a role both
                # palettes hold, or the switch is a KeyError halfway through.
                self.assertIn(self.app.arcs[lead], self.mod.DARK)
                self.assertIn(self.app.arcs[lead], self.mod.LIGHT)
                self.assertTrue(lead.find_all(), "repainted, not reconfigured")
                self.assertNotIn(self.mod.DARK["side"] if name == "light" else
                                 self.mod.LIGHT["side"],
                                 [lead.itemcget(i, "fill") for i in lead.find_all()],
                                 "no colour left over from the other palette")
        finally:
            self.app._theme(was)

    def test_a_call_waiting_on_a_result_shows_dots_and_gives_them_up(self):
        """The ellipsis a row used to wait behind never moved, so a call that
        took a minute looked the same as one that had hung."""
        s = self.app.cur()
        self.app._clear_view(s)
        windows = len(s.view.window_names())
        self.app._show_call(s, {"name": "get_comp", "arguments": {}, "via": None})
        self.assertEqual(len(s.view.window_names()), windows + 1)
        n, = s.open_calls["get_comp"]
        dots = s.pending[n]
        self.assertIn(("dots", str(dots)), self.app.anim)
        self.app._show_result(s, {"name": "get_comp", "text": "ok", "status": "ok"})
        self.assertNotIn(("dots", str(dots)), self.app.anim)
        self.assertNotIn(n, s.pending)

    def test_a_call_row_written_while_thinking_survives_the_dots_going(self):
        """The dots sit at the end, and `_end_thinking` deletes from its mark
        to the end - so a row written past them would be inside what it
        deletes. `_show_call` takes them down first, not last."""
        s = self.app.cur()
        self.app._clear_view(s)
        self.app._begin_thinking(s)
        self.app._show_call(s, {"name": "get_comp", "arguments": {}, "via": None})
        self.assertIsNone(s.thinking)
        self.assertIn("get_comp", s.view.get("1.0", "end"))

    def test_a_failed_call_still_says_so_in_words(self):
        """The dots go; the word that replaces them is the one the row had
        before any of this."""
        s = self.app.cur()
        self.app._clear_view(s)
        self.app._show_call(s, {"name": "set_text", "arguments": {}, "via": None})
        self.app._show_result(s, {"name": "set_text", "text": "no", "status": "error"})
        self.assertIn("set_text failed", s.view.get("1.0", "end"))

    def test_a_picture_tool_holds_the_space_its_picture_will_fill(self):
        """A generation is the longest wait in the app and the one with most to
        show for it, so the waiting happens where the result will be."""
        import base64
        s = self.app.cur()
        self.app._clear_view(s)
        self.app._show_call(s, {"name": "comfy_generate", "arguments": {}, "via": None})
        self.assertEqual(len(s.stages), 1)
        _tag, stage = s.stages[0]
        self.assertIn(("stage", str(stage)), self.app.anim)
        before = s.view.index("end-1c")
        png = icons.png(bytes([0, 0, 0, 255]) * 4, 2, 2)
        self.app._handle("preview", s.event_id,
                         {"type": "image", "mimeType": "image/png",
                          "data": base64.b64encode(png).decode()})
        self.assertEqual(s.stages, [], "the picture took the placeholder's place")
        self.assertNotIn(("stage", str(stage)), self.app.anim)
        self.assertLessEqual(float(s.view.index("end-1c").split(".")[0]),
                             float(before.split(".")[0]) + 1,
                             "the placeholder came out as the picture went in")

    def test_a_placeholder_nothing_arrives_for_is_taken_down_by_the_result(self):
        """Guessing from a tool's name costs nothing in either direction, and
        this is the half that makes that true."""
        s = self.app.cur()
        self.app._clear_view(s)
        self.app._show_call(s, {"name": "ppro_screenshot", "arguments": {}, "via": None})
        self.assertEqual(len(s.stages), 1)
        self.app._show_result(s, {"name": "ppro_screenshot", "text": "no display",
                                  "status": "error"})
        self.assertEqual(s.stages, [])
        self.assertFalse([k for k in self.app.anim if k[0] == "stage"])

    def test_which_tool_names_are_taken_to_make_a_picture(self):
        makes = self.mod.makes_a_picture
        for name in ("comfy_generate", "comfy_run_workflow", "comfy_wait",
                     "ppro_screenshot", "screenshot_frame", "screenshot_layer",
                     "render", "comfy_fetch_output"):
            self.assertTrue(makes(name), name)
        for name in ("get_comp", "list_folder", "set_text", "comfy_list_models",
                     "timeline", "read_file"):
            self.assertFalse(makes(name), name)

    def test_the_thinking_dots_come_and_go_without_a_trace(self):
        """They sit at the end of the transcript, so anything written next
        would land underneath them. Every event that writes takes them down
        first, and what is left has to be exactly what was there before."""
        s = self.app.cur()
        self.app._clear_view(s)
        self.app._write(s, "a line\n", "sys")
        before = s.view.get("1.0", "end")
        self.app._begin_thinking(s)
        self.assertIsNotNone(s.thinking)
        _mark, dots = s.thinking
        self.assertIn(("dots", str(dots)), self.app.anim)
        self.app._begin_thinking(s)
        self.assertIs(s.thinking[1], dots, "idempotent: one set of dots, not two")
        self.app._end_thinking(s)
        self.assertIsNone(s.thinking)
        self.assertNotIn(("dots", str(dots)), self.app.anim)
        self.assertEqual(s.view.get("1.0", "end"), before)
        self.assertFalse(s.view.window_names())

    def test_every_transcript_event_takes_the_thinking_dots_down_first(self):
        """The set is a list, so it can go stale. Anything that reaches
        `_write`, `_show_call` or the rest belongs in it."""
        s = self.app.cur()
        for kind in self.mod.WRITES_TO_TRANSCRIPT:
            with self.subTest(kind=kind):
                self.app._clear_view(s)
                self.app._begin_thinking(s)
                payload = {"sys": "note", "error": "bad", "token": "x",
                           "ready": None, "stream_start": None,
                           "tool": {"name": "t", "arguments": {}, "via": None},
                           "tool_result": {"name": "t", "text": "r"},
                           "ask": {"question": "which?", "options": ["a", "b"]},
                           "preview": {"data": ""}}[kind]
                try:
                    self.app._handle(kind, s.id, payload)
                except Exception:
                    pass          # a malformed payload is not what is on trial
                self.assertIsNone(s.thinking, kind)
        self.app._clear_view(s)

    def test_clearing_a_transcript_stands_its_animations_down(self):
        """New chat destroys every widget embedded in it; the animations that
        were drawing them must not be left for the next frame to trip over."""
        s = self.app.cur()
        self.app._clear_view(s)
        self.app._show_call(s, {"name": "comfy_generate", "arguments": {}, "via": None})
        self.app._begin_thinking(s)
        # This session's widgets by name: `anim` is the whole window's, and
        # another tab's dots are none of this test's business.
        live = ([("dots", str(w)) for w in s.pending.values()]
                + [("stage", str(c)) for _tag, c in s.stages]
                + [("dots", str(s.thinking[1]))])
        self.assertEqual(len(live), 3)
        for key in live:
            self.assertIn(key, self.app.anim)
        self.app._clear_view(s)
        self.assertEqual(s.pending, {})
        self.assertEqual(s.stages, [])
        self.assertIsNone(s.thinking)
        self.assertFalse([k for k in self.app.anim if k in live])

    def test_a_busy_tab_pulses_its_own_dot_and_stops_when_it_is_done(self):
        """The run you started is very often not the tab you are looking at,
        so a tab says for itself that it is working."""
        s = self.app.cur()
        dot = self.app.tab_ui[s.id]["dot"]
        key = ("dot", str(dot))
        was = s.busy
        try:
            s.busy = True
            self.app._paint_tab(s.id)
            self.assertIn(key, self.app.anim)
            self.assertEqual(self.app.dot_role[dot], s.bridge[0],
                             "the settled role survives, for a theme switch")
            s.busy = False
            self.app._paint_tab(s.id)
            self.assertNotIn(key, self.app.anim)
        finally:
            s.busy = was
            self.app._paint_tab(s.id)

    def test_a_settled_window_animates_nothing(self):
        """The rule the rest of this section exists to protect: every
        animation is tied to something actually in progress, so a window
        nobody is asking anything of draws nothing and arms no timer. A tab
        stuck at "not started" because the host went away is settled, not
        busy - that one was a live animation for the life of the window."""
        s = self.app.cur()
        was = (s.busy, s.booting, s.bridge, self.app.host_booting)
        try:
            for t in self.app.sessions.values():
                t.busy = t.booting = False
                self.app._handle("bridge", t.id, ("faint", "%s\nnot started"
                                                  % t.app.bridge_label))
            self.app.host_booting = False
            self.app._handle("host", None, ("faint", None))
            self.app._handle("host_probed", None, False)
            s.status = ("ready", "muted", False)
            self.app._apply_status()
            for _ in range(self.mod.SWEEP_FRAMES + 2):
                self.app._anim_tick()
            self.assertEqual(self.app.anim, {},
                             "something is animating that nothing is doing")
            self.assertIsNone(self.app.anim_timer)
        finally:
            s.busy, s.booting, s.bridge, self.app.host_booting = was
            self.app._stand_down("host_timer")

    # ---------------------------------------------------------------- corners
    def test_nothing_in_the_window_is_a_tk_button(self):
        """Tk's Button is a rectangle and nothing on it bends, so one of them
        among the rounded controls is not a style choice - it is the only
        shape Tk would give. Every button goes through `_button`, which makes
        a `Pill`. Read from the source because the dialogs that hold most of
        them are not open."""
        import studio_ui
        for module in (self.mod, studio_ui):
            with self.subTest(module=module.__name__):
                with open(module.__file__, encoding="utf-8") as f:
                    body = f.read()
                self.assertNotIn("tk.Button(", body,
                                 "use _button(); see PILL_ROLES for the kinds")

    def test_the_window_says_its_name_once(self):
        """The title bar has it, and Windows repeats it on the taskbar and in
        Alt-Tab. The header said it again directly underneath, which bought
        nothing - the header's job is to say what this tab is doing."""
        self.assertEqual(self.app.title(), self.mod.APP_NAME)
        head = self.app.lbl_status.master
        texts = []
        for child in head.winfo_children():
            try:
                texts.append(str(child.cget("text")))
            except tk.TclError:
                pass
        self.assertNotIn(self.mod.APP_NAME, texts)

    def test_every_pill_is_repainted_on_a_theme_switch(self):
        """A Pill plotted its palette colours into canvas items; `_theme` can
        reconfigure a widget but not a shape, so it repaints them instead."""
        was = self.app.prefs.get("theme")
        try:
            for name in ("light", "dark"):
                self.app._theme(name)
                self.assertTrue(self.app.pills)
                for pill in self.app.pills:
                    self.assertIs(pill.C, self.app.C, pill.cget("text"))
        finally:
            self.app._theme(was)

    def test_a_pictures_masked_corners_follow_the_theme(self):
        """The corners are rounded by painting them in the background they
        sit on - Tk will not clip an image to a shape. Switch the palette
        without redrawing them and every picture in the transcript keeps four
        blots of the old one."""
        import base64
        s = self.app.cur()
        self.app._clear_view(s)
        was = self.app.prefs.get("theme")
        self.app._theme("dark")
        png = icons.png(bytes([255, 255, 255, 255]) * 4, 2, 2)
        self.app._handle("preview", s.event_id,
                         {"type": "image", "mimeType": "image/png",
                          "data": base64.b64encode(png).decode()})
        canvas = s.view.nametowidget(s.view.window_names()[-1])
        try:
            def masks():
                return {canvas.itemcget(i, "fill")
                        for i in canvas.find_withtag("mask")}
            self.assertEqual(masks(), {self.mod.DARK["bg"]})
            self.app._theme("light")
            self.assertEqual(masks(), {self.mod.LIGHT["bg"]})
        finally:
            self.app._theme(was)
            self.app._clear_view(s)

    def test_a_dead_widgets_repaint_is_swept_rather_than_raising(self):
        """Chips are rebuilt on every attach and every send, and each one
        registers a repaint. Left in the list they would grow with the
        session and raise in the middle of a theme switch."""
        gone = tk.Frame(self.app)
        self.app._repaint_on_theme(gone, lambda: None)
        self.assertIn(gone, [w for w, _d in self.app.repaints])
        gone.destroy()
        self.app._forget()
        self.assertNotIn(gone, [w for w, _d in self.app.repaints])

    def test_a_left_reading_pill_takes_its_width_and_wraps(self):
        """An option in the question form is a full-width row that happens to
        be clickable, so it reads from the left and a long label wraps rather
        than running off the end."""
        holder = tk.Frame(self.app, width=self.app._px(260))
        holder.pack_propagate(False)
        pill = self.mod.Pill(holder, "a label long enough that it has to wrap "
                                     "onto a second line to fit in this row",
                             lambda: None, self.app.f_body,
                             self.app.PILL_ROLES["option"], anchor="w")
        pill.pack(fill="x")
        try:
            holder.update()
            pill.paint(self.app.C)
            holder.update()
            one_line = self.app.f_body.metrics("linespace") + 2 * pill.pady
            self.assertGreater(int(pill.cget("height")), one_line, "it wrapped")
            self.assertLessEqual(pill.winfo_width(), self.app._px(260))
        finally:
            holder.destroy()

    def test_a_field_is_an_entry_inside_a_drawn_outline(self):
        """Tk's own `highlightthickness` can only draw a rectangle, and a
        square field between rounded buttons is what gives a window away."""
        holder = tk.Frame(self.app)
        var = tk.StringVar(value="npx -y some-mcp")
        entry = self.app._entry(holder, var)
        try:
            entry.master.pack(fill="x")
            holder.update()
            self.assertIsInstance(entry.master, tk.Canvas)
            self.assertEqual(entry.get(), "npx -y some-mcp")
            self.assertEqual(int(entry.cget("highlightthickness")), 0,
                             "the outline is drawn, not Tk's rectangle")
            self.assertIn(entry.master, [w for w, _d in self.app.repaints])
        finally:
            holder.destroy()
            self.app._forget()

    def test_quitting_leaves_no_animation_armed(self):
        """A second window, so the real one survives the test. `after` against
        a destroyed widget is what printed "invalid command name" over a
        window that was already gone."""
        other = self.mod.Chat()
        try:
            other._animate("x", lambda frame: None)
            self.assertIsNotNone(other.anim_timer)
        finally:
            other._quit()
        self.assertIsNone(other.anim_timer)
        self.assertEqual(other.anim, {})

    def test_the_send_pill_fires_on_release_and_only_inside(self):
        """Press and release, not one click: a button that reacts only once
        the work has started reads as one that missed the press, and letting
        go somewhere else has to mean you changed your mind."""
        fired = []
        pill = self.mod.Pill(self.app, "Go", lambda: fired.append(1), self.app.f_bold,
                             ("accent", "accent_fg", "accent_dk", "border", "faint"))
        pill.paint(self.app.C)
        pill.update()
        w, h = pill.winfo_width(), pill.winfo_height()
        try:
            pill._press(None)
            self.assertTrue(pill.down)
            self.assertEqual(fired, [], "nothing has happened yet")
            pill._release(type("Ev", (), {"x": w // 2, "y": h // 2})())
            self.assertFalse(pill.down)
            self.assertEqual(fired, [1])
            pill._press(None)
            pill._release(type("Ev", (), {"x": w + 40, "y": h // 2})())
            self.assertEqual(fired, [1], "let go outside: cancelled")
            # Off and back on with the button still held re-arms it, so a
            # wobble between press and release is not a click thrown away.
            pill._press(None)
            pill._light(False, type("Ev", (), {"state": 0x0100})())
            self.assertFalse(pill.down)
            pill._light(True, type("Ev", (), {"state": 0x0100})())
            self.assertTrue(pill.down, "came back with the button down")
            pill._release(type("Ev", (), {"x": w // 2, "y": h // 2})())
            self.assertEqual(fired, [1, 1])
            # ...and merely hovering, with nothing held, presses nothing.
            pill._light(True, type("Ev", (), {"state": 0})())
            self.assertFalse(pill.down)
        finally:
            pill.destroy()

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
            # The label is animated now: the button says Stopping and counts
            # dots, so it is the stem that is fixed, not the whole string.
            self.assertTrue(self.app.btn_send.cget("text").startswith("Stopping"),
                            self.app.btn_send.cget("text"))
            self.assertIn("send", self.app.anim)
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
        """A picture arrives on a canvas of its own now, not as a bare image:
        that is what the wipe-in is drawn on. The session still has to hold the
        PhotoImage, or Tk drops it the moment it is collected."""
        import base64
        s = self.app.cur()
        png = icons.png(bytes([255, 0, 0, 255]) * 4, 2, 2)
        count, windows = len(s.preview_images), len(s.view.window_names())
        self.app._handle("preview", s.event_id, {"type": "image", "mimeType": "image/png",
                                               "data": base64.b64encode(png).decode()})
        self.assertEqual(len(s.preview_images), count + 1)
        self.assertEqual(len(s.view.window_names()), windows + 1)

    def _picture(self, name, kind="png"):
        """A real PNG Tk wrote itself, or a JPEG that is only a header - the
        dimensions live in the header, and that is all attaching reads."""
        path = os.path.join(self.dir, name)
        if kind == "png":
            photo = tk.PhotoImage(width=320, height=180, master=self.app)
            photo.write(path, format="png")
        else:
            self._blob(path, b"\xff\xd8\xff\xe0\x00\x04\x00\x00"           # APP0
                             b"\xff\xc0\x00\x11\x08\x04\x38\x07\x80\x03"
                             b"\x01\x22\x00\x02\x11\x01\x03\x11\x01\xff\xd9")
        return path

    @staticmethod
    def _blob(path, data):
        with open(path, "wb") as fh:
            fh.write(data)

    def test_picture_dimensions_come_from_the_header(self):
        self.assertEqual(self.mod.image_dims(self._picture("a.png")), (320, 180))
        self.assertEqual(self.mod.image_dims(self._picture("b.jpg", "jpg")), (1920, 1080))
        gif = os.path.join(self.dir, "c.gif")
        self._blob(gif, b"GIF89a\x40\x01\xf0\x00" + b"\x00" * 6)
        self.assertEqual(self.mod.image_dims(gif), (320, 240))
        txt = os.path.join(self.dir, "d.txt")
        self._blob(txt, b"not a picture")
        self.assertIsNone(self.mod.image_dims(txt))
        self.assertIsNone(self.mod.image_dims(os.path.join(self.dir, "missing.png")))
        line = self.mod.describe_attachment(self._picture("frame 12.png"))
        self.assertIn("frame 12.png (320 x 180, 1 KB PNG) at ", line)
        self.assertTrue(line.endswith(os.path.join(self.dir, "frame 12.png")))

    def test_attached_pictures_ride_with_the_message_in_every_tab(self):
        """One attach control on the shared composer: chips appear above the
        input, the composer stays whole, removing one hides the strip, and
        sending puts the paths - not the pixels - into the brief and the
        picture into the transcript. The same on a tab with no tools."""
        png, jpg = self._picture("board.png"), self._picture("ref.jpg", "jpg")
        self.app.geometry("900x560")
        for app_id in (eng.APPS[0].id, eng.CHAT.id):
            with self.subTest(tab=app_id):
                self.app._select(app_id)
                s = self.app.cur()
                self.app._add_attachments([png, jpg, png])          # once each
                for _ in range(10):
                    self.app.update()
                self.assertEqual(self.app.attachments, [png, jpg])
                self.assertTrue(self.app.chips.winfo_ismapped())
                self.assertEqual(len(self.app.chips.winfo_children()), 2)
                self.assertTrue(self.app.input.winfo_ismapped())
                self.assertTrue(self.app.btn_send.winfo_ismapped())
                self.assertTrue(self.app.btn_attach.winfo_ismapped())
                self.assertLessEqual(self._bottom_of(self.app.btn_send), self.app.winfo_height())
                self.app._drop_attachment(jpg)
                self.app.update()
                self.assertEqual(len(self.app.chips.winfo_children()), 1)
                spawned = []
                original = self.app._spawn
                self.app._spawn = lambda *a: spawned.append(a)
                messages, record, ready = s.messages, s.record, s.ready
                try:
                    s.reset()
                    s.ready = True
                    # Counted by what the session owns, not by what is on the
                    # transcript: sending also puts the thinking dots there.
                    pictures = len(s.preview_images)
                    self.app.input.insert("1.0", "Match this")
                    self.app._on_send()
                    self.app.update()
                    self.assertEqual(spawned[0][1:], (self.app._turn, s, [png]))
                    brief = s.messages[-1]["content"]
                    self.assertTrue(brief.startswith("Match this\n\nAttached files and folders"))
                    self.assertIn("board.png (320 x 180, 1 KB PNG) at " + png, brief)
                    self.assertNotIn("base64", brief)
                    self.assertEqual(s.record.briefs[-1], brief)
                    self.assertEqual(self.app.attachments, [])
                    self.assertFalse(self.app.chips.winfo_ismapped())
                    self.assertEqual(self.app.input.get("1.0", "end").strip(), "")
                    self.assertEqual(len(s.preview_images), pictures + 1)
                    # a picture alone is a message
                    s.busy = False                      # the mocked turn never ended
                    self.app._add_attachments([jpg])
                    self.app._on_send()
                    self.assertTrue(s.messages[-1]["content"].startswith("Take a look"))
                    self.assertIn("[ref.jpg]", s.view.get("1.0", "end"))   # no Tk decoder
                finally:
                    s.busy = False
                    self.app.attachments = []
                    self.app._paint_chips()
                    s.messages, s.record, s.ready = messages, record, ready
                    self.app._spawn = original

    def test_a_container_tab_is_handed_a_copy_it_can_reach(self):
        """OpenCode sees one folder. A picture from anywhere else is copied in,
        and the brief names the path inside the container, not the one here."""
        png = self._picture("sketch.png")
        spec = eng.APPS_BY_ID["opencode"]
        real = spec.workspace
        spec.workspace = os.path.join(self.dir, "ws")
        try:
            note = self.mod.attachment_note([png], spec)
            copy = os.path.join(spec.workspace, "attachments", "sketch.png")
            self.assertTrue(os.path.exists(copy))
            self.assertIn("/workspace/attachments/sketch.png", note)
            self.assertIn(copy, note)
            self.assertNotIn(png + ")", note)
            self.assertEqual(self.mod.attachment_note([], spec), "")
            plain = self.mod.attachment_note([png], eng.APPS[0])
            self.assertIn(png, plain)
            self.assertNotIn("/workspace", plain)
            # A folder is copied whole, and named by its folder path inside.
            src = os.path.join(self.dir, "refs")
            os.makedirs(os.path.join(src, "inner"))
            self._blob(os.path.join(src, "inner", "a.txt"), b"a")
            note = self.mod.attachment_note([src + os.sep], spec)
            self.assertTrue(os.path.exists(os.path.join(spec.workspace, "attachments",
                                                        "refs", "inner", "a.txt")))
            self.assertIn("/workspace/attachments/refs)", note)
            self.assertIn("refs (folder, 0 files, 1 folders)", note)
        finally:
            spec.workspace = real

    def test_any_file_or_folder_can_be_attached(self):
        """Files travel as paths whatever their type or size, and a folder
        goes as a bounded listing so the model can name what is in it
        without a tool call. Only pictures reach the vision model."""
        big = os.path.join(self.dir, "render.mov")
        with open(big, "wb") as fh:
            fh.seek(self.mod.ATTACH_LIMIT + 1)
            fh.write(b"\0")
        folder = os.path.join(self.dir, "footage")
        os.makedirs(os.path.join(folder, "audio"))
        for i in range(self.mod.LIST_LIMIT + 5):
            self._blob(os.path.join(folder, "clip%03d.mp4" % i), b"x")
        png = self._picture("still.png")
        line = self.mod.describe_attachment(big)
        self.assertTrue(line.startswith("render.mov (200.0 MB MOV) at "))
        listing = self.mod.describe_attachment(folder)
        self.assertTrue(listing.startswith("footage (folder, %d files, 1 folders) at %s\n"
                                           % (self.mod.LIST_LIMIT + 5, folder)))
        self.assertIn("\n    audio/\n    clip000.mp4\n", listing)
        self.assertIn("\n    ... and 6 more", listing)
        self.assertNotIn("clip044", listing)
        self.app._select(eng.APPS[0].id)
        s = self.app.cur()
        self.app._add_attachments([big, folder, png])
        self.app.update()
        self.assertEqual(self.app.attachments, [big, folder, png])
        self.assertEqual(len(self.app.chips.winfo_children()), 3)
        self.assertTrue(self.app.btn_folder.winfo_ismapped())
        spawned = []
        original = self.app._spawn
        self.app._spawn = lambda *a: spawned.append(a)
        messages, record, ready = s.messages, s.record, s.ready
        try:
            s.reset()
            s.ready = True
            self.app.input.insert("1.0", "Cut these together")
            self.app._on_send()
            self.assertEqual(spawned[0][1:], (self.app._turn, s, [png]))
            brief = s.messages[-1]["content"]
            self.assertIn("- " + line, brief)
            self.assertIn("- " + listing, brief)
            self.assertIn("[footage%s]" % os.sep, s.view.get("1.0", "end"))
        finally:
            s.busy = False
            self.app.attachments = []
            self.app._paint_chips()
            s.messages, s.record, s.ready = messages, record, ready
            self.app._spawn = original

    def test_vision_description_lands_in_the_brief_before_the_turn(self):
        """When the host serves a vision model the worker asks it what the
        pictures show and appends that to the brief - the executing model
        reads text. A vision failure is one line, and the turn still runs.
        With no vision model the tab says so, once per turn with pictures."""
        from test_tasks import FakeLLM, answer

        class FakeVision:
            model = "some-vl"
            def __init__(self, fn):
                self.describe_all = fn
            def review(self, item, brief):
                return "looks fine"

        png = self._picture("still.png")
        s = self.app.cur()
        original_llm, saved = self.app.llm, (s.messages, s.record)
        original_vision = self.app.vision
        try:
            s.reset()
            brief = "Match this" + self.mod.attachment_note([png], s.app)
            s.messages.append({"role": "user", "content": brief})
            s.record.briefs.append(brief)
            self.app.llm = FakeLLM([answer(text="Done.")])
            self.app.vision = FakeVision(lambda paths: "\n\nWhat the pictures show: a blue card")
            self.app._turn(s, [png])
            self.assertTrue(s.messages[1]["content"].endswith("a blue card"))
            self.assertEqual(s.record.briefs[-1], s.messages[1]["content"])
            self.assertEqual(s.messages[-1]["content"], "Done.")
            s.reset()
            s.messages.append({"role": "user", "content": brief})
            s.record.briefs.append(brief)
            self.app.llm = FakeLLM([answer(text="Done anyway.")])
            def fail(paths):
                raise RuntimeError("host busy")
            self.app.vision = FakeVision(fail)
            self.app._turn(s, [png])
            self.assertEqual(s.messages[1]["content"], brief)
            self.assertEqual(s.messages[-1]["content"], "Done anyway.")
            s.reset()
            s.messages.append({"role": "user", "content": brief})
            s.record.briefs.append(brief)
            self.app.llm = FakeLLM([answer(text="Blind.")])
            self.app.vision = None
            self.app._turn(s, [png])
            self.app._drain()
            self.assertIn("No vision model is served", s.view.get("1.0", "end"))
            self.assertEqual(s.messages[-1]["content"], "Blind.")
        finally:
            self.app.vision = original_vision
            self.app.llm = original_llm
            s.messages, s.record = saved

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

    def test_a_bridge_entered_by_hand_becomes_a_tab_a_row_and_a_setting(self):
        """The connect dialog's values go through _save_bridge; a bad set is a
        sentence back to the dialog, a good one is a new drivable row, an open
        tab and a record in the settings file. Forgetting undoes all three."""
        script = os.path.join(os.path.dirname(eng.__file__), "studio_comfy_mcp.py")
        line = '"%s" "%s"' % (sys.executable, script)
        self.assertIn("name", self.app._save_bridge({"name": " ", "command": line}))
        self.assertIn("command line", self.app._save_bridge({"name": "Blender", "command": ""}))
        self.assertIn("not on PATH", self.app._save_bridge({"name": "Blender",
                                                            "command": "no-such-thing-xyz"}))
        self.assertIn("process:", self.app._save_bridge({"name": "Blender", "command": line,
                                                         "probe": "magic:x"}))
        self.assertIsNone(self.app._save_bridge({"name": "Blender", "command": line,
                                                 "probe": "process:blender.exe"}))
        try:
            spec = eng.APPS_BY_ID["blender"]
            self.assertTrue(spec.custom)
            self.assertEqual(spec.command, sys.executable)
            self.assertEqual(spec.args, [script])
            self.assertEqual(self.app.active, "blender")
            self.assertIn("blender", self.app.sessions)
            self.assertIn("Blender", self._sidebar_names())
            saved = self.mod.Prefs(self.app.prefs.path).get("bridges")
            self.assertEqual([b["name"] for b in saved], ["Blender"])
            # the dialog itself builds, prefilled, for a new row and for an edit
            win = self.app._bridge_dialog(row={"name": "Premiere Pro", "exe": ""})
            win.destroy()
            win = self.app._bridge_dialog(spec=spec)
            win.destroy()
            # the Bridges menu offers the new bridge's tools window
            self.app._fill_bridge_menu()
            labels = [self.app.m_bridge.entrycget(i, "label")
                      for i in range(self.app.m_bridge.index("end") + 1)
                      if self.app.m_bridge.type(i) == "command"]
            self.assertIn("Blender tools...", labels)
        finally:
            if "blender" in eng.APPS_BY_ID:
                self.app._forget_bridge(eng.APPS_BY_ID["blender"])
        self.assertNotIn("blender", eng.APPS_BY_ID)
        self.assertNotIn("blender", self.app.sessions)
        self.assertNotIn("Blender", self._sidebar_names())
        self.assertEqual(self.mod.Prefs(self.app.prefs.path).get("bridges"), [])

    def test_a_learned_bridge_rewrites_the_prompt_before_the_warm_up(self):
        """What a hand-entered bridge offers is only known once it answers, so
        _boot_bridge fills the entry in and the session's system prompt - made
        at Session() from the empty entry - is replaced with the learned one."""
        spec = eng.add_bridge(eng.BridgeSpec("Blender", sys.executable, ["-c", "pass"]))

        class FakeClient:
            instructions = "Blender counts in metres."

            def __init__(self, *a, **kw):
                pass

            def initialize(self, timeout=None):
                return {}

            def list_tools(self, timeout=None):
                return [{"name": "scene_list", "description": "d", "inputSchema": {"type": "object"}},
                        {"name": "scene_get", "description": "d", "inputSchema": {"type": "object"}},
                        {"name": "obj_add", "description": "d", "inputSchema": {"type": "object"}},
                        {"name": "obj_del", "description": "d", "inputSchema": {"type": "object"}}]

            def close(self):
                pass

        real = eng.MCPClient
        eng.MCPClient = FakeClient
        try:
            self.app._add_tab("blender")
            s = self.app.sessions["blender"]
            before = s.messages[0]["content"]
            self.app._boot_bridge(s)
            self.assertTrue(spec.learned)
            self.assertEqual(s.groups, ["scene", "obj"])
            # The bridge's four, then the research sidecar's beside them.
            names = [t["function"]["name"] for t in s.tools]
            self.assertEqual(names[:4], ["scene_list", "scene_get", "obj_add", "obj_del"])
            self.assertEqual(set(names[4:]), set(eng.RESEARCH_TOOL_NAMES))
            self.assertNotEqual(s.messages[0]["content"], before)
            self.assertIn("Blender counts in metres.", s.messages[0]["content"])
            self.assertEqual(s.messages[0]["content"], s.prompt())
            self.assertTrue(s.messages[0]["content"].startswith(spec.chat_prompt()))
        finally:
            eng.MCPClient = real
            self.app._forget_bridge(spec)

    def test_tabs_fold_to_their_marks_rather_than_fall_off_the_edge(self):
        """Seven labelled tabs do not fit a small window. Rather than let the
        packer push the last ones off unmapped, every tab but the active one
        folds to mark and dot; wide again, they unfold."""
        self.app.geometry("%dx%d" % (self.app._px(880), self.app._px(560)))
        self.app.update()
        self.app._fit_tabs()
        self.app.update()
        for sid, ui in self.app.tab_ui.items():
            self.assertTrue(ui["tab"].winfo_ismapped(), sid)
            self.assertTrue(ui["mark"].winfo_ismapped(), sid)
        active = self.app.tab_ui[self.app.active]
        self.assertTrue(active["label"].winfo_ismapped(), "the active tab keeps its name")
        folded = [sid for sid, ui in self.app.tab_ui.items() if ui["compact"]]
        self.assertTrue(folded, "nothing folded at the minimum width")
        self.assertNotIn(self.app.active, folded)
        # selecting a folded tab unfolds it and folds the one that was active
        target = folded[0]
        self.app._select(target)
        self.app.update()
        self.assertTrue(self.app.tab_ui[target]["label"].winfo_ismapped())
        self.app.geometry("%dx%d" % (self.app._px(1900), self.app._px(820)))
        self.app.update()
        self.app._fit_tabs()
        self.app.update()
        self.assertFalse([sid for sid, ui in self.app.tab_ui.items() if ui["compact"]])
        self.app.geometry("%dx%d" % (self.app._px(1180), self.app._px(820)))
        self.app.update()

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
        self.addCleanup(studio_chat._LOCK.close)
        self.assertFalse(studio_chat.claim_single_instance(port))


if __name__ == "__main__":
    unittest.main(verbosity=2)
