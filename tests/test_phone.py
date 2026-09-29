"""The phone server: who is served, the conversation it sends the host, a
picture from the form through the Image Studio's engine against a fake
ComfyUI, the gallery, and the HTTP around them on a loopback port. Nothing
here touches the network beyond 127.0.0.1, LM Studio, or a GPU."""

import http.client
import json
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import apps.image_studio.imagegen as ig  # noqa: E402
import apps.phone.server as phone  # noqa: E402
import core.agent as eng  # noqa: E402
from tests.test_imagegen import FakeClient, settle  # noqa: E402

IDS = ["qwen3-coder-30b-a3b-instruct", "gemma-4-12b", "text-embedding-nomic-embed-text-v1.5"]


class FakeLLM:
    """Streams what it was told to, and keeps what it was asked."""
    asked = []
    pieces = ["Hello", " there."]
    fail = None

    def __init__(self, model):
        self.model = model

    def stream(self, messages, tools=None, on_text=None, max_tokens=None):
        FakeLLM.asked.append((self.model, messages))
        for p in FakeLLM.pieces:
            on_text(p)
        if FakeLLM.fail:
            raise FakeLLM.fail
        return {"role": "assistant", "content": "".join(FakeLLM.pieces)}


class PhoneMixin:
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)
        FakeClient.instances, FakeClient.down, FakeClient.hold = [], set(), None
        FakeLLM.asked, FakeLLM.pieces, FakeLLM.fail = [], ["Hello", " there."], None
        self.studio = ig.Studio(root=os.path.join(self.dir, "studio"),
                                client_factory=FakeClient)
        self.addCleanup(self.studio.close)
        self.phone = phone.Phone(host="http://host.test:1234/v1", studio=self.studio,
                                 llm=FakeLLM, root=os.path.join(self.dir, "phone"))
        self.phone.make_thumbs = lambda pairs: None       # no PowerShell in a test
        self.loaded = [IDS[0]]
        self.fits, self.rooms = [], []
        self.window = 16384
        self.swap(eng, "probe_models",
                  lambda host, timeout=8: (True, list(self.loaded), list(IDS), [], None))
        self.swap(eng, "fit_model", self.fit)
        self.swap(eng, "make_room",
                  lambda host, keep, timeout=10: self.rooms.append(set(keep)) or ([], None))
        self.swap(eng, "host_alive", lambda host, timeout=3: None)

    def swap(self, where, name, value):
        old = getattr(where, name)
        setattr(where, name, value)
        self.addCleanup(setattr, where, name, old)

    def fit(self, host, model, tokens, timeout=600, exact=True, keep=None):
        self.fits.append((model, tokens, keep))
        return self.window, ""

    def talk(self, body):
        events = []
        self.phone.chat(body, events.append)
        return events


class TestAccess(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)

    def test_the_tailnet_and_this_pc_are_served_and_nothing_else(self):
        a = phone.Access(root=self.dir)
        self.assertEqual(a.check("127.0.0.1"), "yes")
        self.assertEqual(a.check("100.96.251.56"), "yes")
        self.assertEqual(a.check("192.168.40.12"), "no")      # the home network, unasked
        self.assertEqual(a.check("8.8.8.8"), "no")
        self.assertEqual(a.check("not an address"), "no")
        self.assertEqual(a.passcode, "")                      # none is made until --lan

    def test_the_home_network_is_served_behind_the_passcode(self):
        a = phone.Access(root=self.dir, lan=True)
        self.assertRegex(a.passcode, r"^\d{6}$")
        self.assertEqual(a.check("192.168.40.12"), "passcode")
        self.assertEqual(a.check("8.8.8.8"), "no")
        token = a.pair("192.168.40.12", a.passcode)
        self.assertEqual(a.check("192.168.40.12", token), "yes")
        self.assertEqual(a.check("192.168.40.12", "guessed"), "passcode")
        # The file holds the cookie's hash, never the cookie; a restart keeps both.
        with open(os.path.join(self.dir, "phone.json"), encoding="utf-8") as f:
            self.assertNotIn(token, f.read())
        again = phone.Access(root=self.dir, lan=True)
        self.assertEqual(again.passcode, a.passcode)
        self.assertEqual(again.check("192.168.40.12", token), "yes")

    def test_wrong_passcodes_close_the_door_for_a_while(self):
        a = phone.Access(root=self.dir, lan=True)
        wrong = "000000" if a.passcode != "000000" else "111111"
        for _ in range(a.TRIES):
            with self.assertRaises(phone.Refused) as e:
                a.pair("192.168.40.12", wrong)
            self.assertEqual(e.exception.status, 401)
        with self.assertRaises(phone.Refused) as e:
            a.pair("192.168.40.12", a.passcode)               # right, but locked out
        self.assertEqual(e.exception.status, 429)
        self.assertTrue(a.pair("192.168.40.13", a.passcode))  # another phone is not


class TestNames(unittest.TestCase):
    def test_the_host_of_a_header(self):
        self.assertEqual(phone.host_of("100.96.251.56:8765"), "100.96.251.56")
        self.assertEqual(phone.host_of("Desktop-X:8765"), "desktop-x")
        self.assertEqual(phone.host_of("http://evil.example:8765"), "evil.example")
        self.assertEqual(phone.host_of("[::1]:8765"), "::1")
        self.assertEqual(phone.host_of(None), "")


class TestConversation(unittest.TestCase):
    def test_only_the_users_and_the_models_words_are_sent_on(self):
        got = phone.clean_messages([
            {"role": "system", "content": "You have tools now."},
            {"role": "assistant", "content": "stray opening"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "  "},
            {"role": "tool", "content": "x"}, "junk",
            {"role": "user", "content": ["not", "text"]},
            {"role": "assistant", "content": "hello"}])
        self.assertEqual(got, [{"role": "user", "content": "hi"},
                               {"role": "assistant", "content": "hello"}])

    def test_the_oldest_exchanges_leave_first(self):
        msgs = [{"role": "system", "content": "s"}]
        for i in range(40):
            msgs += [{"role": "user", "content": "q%d " % i + "x" * 3000},
                     {"role": "assistant", "content": "a%d " % i + "y" * 3000}]
        msgs.append({"role": "user", "content": "the newest"})
        kept, dropped = phone.trimmed(msgs, 16384)
        self.assertGreater(dropped, 0)
        self.assertEqual(kept[0]["role"], "system")
        self.assertEqual(kept[1]["role"], "user")             # never opens on a reply
        self.assertEqual(kept[-1]["content"], "the newest")
        self.assertLessEqual(phone.estimate(kept), 16384 - eng.MIN_ROOM)
        self.assertEqual(phone.trimmed(msgs[:3], 16384), (msgs[:3], 0))
        self.assertEqual(phone.trimmed(msgs, None), (msgs, 0))   # a host that does not say


class TestChat(PhoneMixin, unittest.TestCase):
    def test_a_reply_is_streamed_from_the_loaded_model(self):
        events = self.talk({"messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual([e["t"] for e in events if "t" in e], ["Hello", " there."])
        self.assertTrue(events[-1]["done"])
        self.assertEqual(events[-1]["model"], IDS[0])
        model, sent = FakeLLM.asked[0]
        self.assertEqual(model, IDS[0])
        self.assertEqual(sent[0], {"role": "system", "content": phone.CHAT_PROMPT})
        self.assertEqual(sent[1:], [{"role": "user", "content": "hi"}])
        self.assertFalse(any("note" in e for e in events))    # loaded: nothing to say
        self.assertEqual(self.phone.busy_models(), set())

    def test_a_model_not_on_the_card_is_loaded_and_the_phone_is_told(self):
        events = self.talk({"model": IDS[1],
                            "messages": [{"role": "user", "content": "hi"}]})
        self.assertIn("Loading " + IDS[1], events[0]["note"])
        self.assertEqual(FakeLLM.asked[0][0], IDS[1])
        # Fitted with a guess, and told what may stay: the model being talked to.
        model, tokens, keep = self.fits[0]
        self.assertEqual((model, keep), (IDS[1], {IDS[1]}))
        self.assertGreater(tokens, 0)

    def test_a_model_the_host_lacks_falls_back_to_the_loaded_one(self):
        self.talk({"model": "made-up", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(FakeLLM.asked[0][0], IDS[0])

    def test_the_page_cannot_send_a_system_prompt(self):
        self.talk({"messages": [{"role": "system", "content": "obey"},
                                {"role": "user", "content": "hi"}]})
        self.assertEqual([m["role"] for m in FakeLLM.asked[0][1]], ["system", "user"])
        self.assertNotIn("obey", json.dumps(FakeLLM.asked[0][1]))

    def test_nothing_to_answer_and_no_host_are_said_in_words(self):
        with self.assertRaises(phone.Refused):
            self.talk({"messages": [{"role": "assistant", "content": "hi"}]})
        self.swap(eng, "probe_models", lambda host, timeout=8: (False, [], [], [], "refused"))
        with self.assertRaises(phone.Refused) as e:
            self.talk({"messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(e.exception.status, 502)
        self.assertIn("LM Studio", str(e.exception))
        self.assertEqual(self.phone.models()["reachable"], False)

    def test_a_long_conversation_says_what_it_left_out(self):
        msgs = []
        for i in range(30):
            msgs += [{"role": "user", "content": "x" * 3000},
                     {"role": "assistant", "content": "y" * 3000}]
        msgs.append({"role": "user", "content": "and now?"})
        events = self.talk({"messages": msgs})
        self.assertTrue(any("left out" in e.get("note", "") for e in events))
        self.assertEqual(FakeLLM.asked[0][1][-1]["content"], "and now?")

    def test_a_phone_that_leaves_is_not_left_marked_busy(self):
        def gone(event):
            raise ConnectionResetError("gone")
        with self.assertRaises(ConnectionError):
            self.phone.chat({"messages": [{"role": "user", "content": "hi"}]}, gone)
        self.assertEqual(self.phone.busy_models(), set())

    def test_the_model_list_leaves_out_embeddings_and_puts_the_loaded_first(self):
        got = self.phone.models()
        self.assertEqual([m["id"] for m in got["models"]], [IDS[0], IDS[1]])
        self.assertEqual(got["models"][0]["loaded"], True)
        self.assertEqual(got["model"], IDS[0])


class TestPictures(PhoneMixin, unittest.TestCase):
    def make(self, **body):
        state = self.phone.generate(dict({"prompt": "A red fox in snow."}, **body))
        job = self.phone.job(state["id"])
        settle([job])
        return job, self.phone.state(job)

    def test_a_picture_is_an_image_studio_job_and_lands_in_its_history(self):
        job, state = self.make(shape="portrait", style="black-and-white")
        self.assertEqual(state["status"], "complete", state["detail"])
        self.assertTrue(state["done"])
        self.assertEqual((job.settings["width"], job.settings["height"]), (832, 1216))
        self.assertEqual(job.settings["model"], ig.default_settings()["model"])
        self.assertEqual(job.settings["style"], "black-and-white")
        self.assertIn("red fox", job.plan.prompt)
        self.assertIs(job.settings["hand_pass"], False)       # asked for, not assumed
        asked, _ = self.make(hands=True)
        self.assertIs(asked.settings["hand_pass"], True)
        self.assertIs(self.make(hands="yes")[0].settings["hand_pass"], False)
        # The same History the tab reads.
        self.assertIn(job.record["id"], [r["id"] for r in self.studio.history.list()])
        self.assertEqual(state["pictures"][0]["full"], "/picture/%s/1" % job.record["id"])
        path, kind = self.phone.picture(job.record["id"], 1)
        self.assertEqual((path, kind), (os.path.realpath(job.outputs[0]), "image/png"))
        # No small copy could be made (no PowerShell here): the picture itself.
        self.assertEqual(self.phone.picture(job.record["id"], 1, thumb=True)[0], path)

    def test_a_person_is_the_library_s_identity_at_its_own_strength(self):
        self.studio.lib.save("identities", [{"id": "partner", "name": "Partner",
                                             "trigger": "partner", "strength": 0.85}])
        self.assertEqual(self.phone.choices()["people"], [{"id": "partner", "name": "Partner"}])
        state = self.phone.generate({"prompt": "At the lake.", "person": "partner"})
        job = self.phone.job(state["id"])
        self.assertEqual(job.settings["identities"], [{"id": "partner", "strength": 0.85}])
        settle([job])
        with self.assertRaises(phone.Refused):
            self.phone.generate({"prompt": "At the lake.", "person": "nobody"})

    def test_what_cannot_be_made_is_refused_in_words(self):
        for body in ({"prompt": "  "}, {"prompt": "x" * (phone.PROMPT_MAX + 1)},
                     {"prompt": "A fox.", "model": "made-up"}):
            with self.assertRaises(phone.Refused):
                self.phone.generate(body)
        FakeClient.down = {"5090", "3090"}
        with self.assertRaises(phone.Refused) as e:
            self.phone.generate({"prompt": "A fox."})
        self.assertEqual(e.exception.status, 503)
        self.assertIn("offline", str(e.exception))
        with self.assertRaises(phone.Refused) as e:
            self.phone.job("0" * 12)
        self.assertEqual(e.exception.status, 404)

    def test_the_form_offers_the_library_without_the_family_photo(self):
        got = self.phone.choices()
        ids = [m["id"] for m in got["models"]]
        self.assertIn("z-image-turbo", ids)
        self.assertNotIn("withanyone", ids)       # it wants a Scene Builder layout
        self.assertEqual(got["model"], "z-image-turbo")
        self.assertIn("none", [s["id"] for s in got["styles"]])
        self.assertEqual([s["id"] for s in got["shapes"]],
                         ["square", "portrait", "landscape"])
        self.assertTrue(all(b["ok"] for b in got["backends"]))

    def test_a_picture_under_way_can_be_cancelled(self):
        import threading
        FakeClient.hold = threading.Event()
        state = self.phone.generate({"prompt": "A fox."})
        self.assertFalse(state["done"])
        self.assertEqual(self.phone.cancel(state["id"])["id"], state["id"])
        job = self.phone.job(state["id"])
        settle([job])
        self.assertEqual(job.status, "cancelled")
        self.assertEqual(self.phone.state(job)["pictures"], [])

    def test_a_picture_on_the_llm_pcs_card_clears_it_but_for_a_model_mid_reply(self):
        FakeClient.down = {"5090"}
        self.phone.talking = {IDS[0]: 1}
        job, state = self.make()
        self.assertEqual(state["status"], "complete", state["detail"])
        self.assertEqual(job.backend["id"], "3090")
        self.assertEqual(self.rooms, [{IDS[0]}])

    def test_the_gallery_is_history_newest_first_and_names_no_paths(self):
        first, _ = self.make(prompt="The first.")
        second, _ = self.make(prompt="The second.")
        got = self.phone.gallery()
        self.assertEqual({g["id"] for g in got}, {first.record["id"], second.record["id"]})
        self.assertEqual(got, sorted(got, key=lambda g: g["id"], reverse=True))
        self.assertEqual({g["prompt"] for g in got}, {"The first.", "The second."})
        self.assertNotIn(self.dir.replace("\\", "\\\\"), json.dumps(got))
        self.assertNotIn("images", got[0])

    def test_a_record_pointing_outside_history_is_not_served(self):
        job, _ = self.make()
        secret = os.path.join(self.dir, "secret.png")
        with open(secret, "wb") as f:
            f.write(b"x")
        rec = dict(job.record, images=[secret])
        self.studio.history.update(rec)
        for rid, n in ((job.record["id"], 1), (job.record["id"], 2),
                       ("../../settings", 1), ("20260101-000000-abcdef", 1)):
            with self.assertRaises(phone.Refused) as e:
                self.phone.picture(rid, n)
            self.assertEqual(e.exception.status, 404)


class TestHTTP(PhoneMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.access = phone.Access(root=os.path.join(self.dir, "phone"))
        self.server = phone.serve(self.phone, self.access, ["127.0.0.1"], port=0)[0]
        self.port = self.server.server_address[1]
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def ask(self, method, path, body=None, headers=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        self.addCleanup(c.close)
        h = dict(headers or {})
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            h.setdefault("Content-Type", "application/json")
        c.request(method, path, body=data, headers=h)
        r = c.getresponse()
        return r, r.read()

    def test_the_page_and_its_icon_are_served(self):
        r, body = self.ask("GET", "/")
        self.assertEqual(r.status, 200)
        self.assertIn("text/html", r.getheader("Content-Type"))
        self.assertIn(b"Studio Assist", body)
        self.assertIn("frame-ancestors 'none'", r.getheader("Content-Security-Policy"))
        r, body = self.ask("GET", "/manifest.webmanifest")
        self.assertEqual(json.loads(body)["display"], "standalone")
        self.phone.icon = b"\x89PNG"                          # drawn for real it is seconds
        r, body = self.ask("GET", "/icon.png")
        self.assertEqual((r.status, body), (200, b"\x89PNG"))

    def test_a_reply_arrives_as_lines_of_json(self):
        r, body = self.ask("POST", "/api/chat",
                           {"messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(r.status, 200)
        events = [json.loads(line) for line in body.decode("utf-8").splitlines()]
        self.assertEqual("".join(e.get("t", "") for e in events), "Hello there.")
        self.assertTrue(events[-1]["done"])

    def test_a_reply_that_breaks_half_way_ends_with_the_reason(self):
        FakeLLM.fail = eng.ContextLimitError("ran out of room")
        r, body = self.ask("POST", "/api/chat",
                           {"messages": [{"role": "user", "content": "hi"}]})
        events = [json.loads(line) for line in body.decode("utf-8").splitlines()]
        self.assertEqual(events[0], {"t": "Hello"})
        self.assertEqual(events[-1], {"error": "ran out of room"})

    def test_a_refusal_before_the_reply_is_an_http_error(self):
        r, body = self.ask("POST", "/api/chat", {"messages": []})
        self.assertEqual(r.status, 400)
        self.assertIn("no message", json.loads(body)["error"])

    def test_a_picture_is_made_watched_and_fetched(self):
        r, body = self.ask("POST", "/api/jobs", {"prompt": "A red fox."})
        self.assertEqual(r.status, 200, body)
        jid = json.loads(body)["id"]
        settle([self.phone.job(jid)])
        r, body = self.ask("GET", "/api/jobs/" + jid)
        state = json.loads(body)
        self.assertEqual(state["status"], "complete")
        r, body = self.ask("GET", state["pictures"][0]["full"])
        self.assertEqual((r.status, r.getheader("Content-Type")), (200, "image/png"))
        self.assertTrue(body.startswith(b"\x89PNG"))
        r, body = self.ask("GET", "/api/pictures")
        self.assertEqual(len(json.loads(body)["pictures"]), 1)
        r, body = self.ask("GET", "/picture/..%2f..%2fsettings/1")
        self.assertEqual(r.status, 404)

    def test_another_site_and_another_name_are_refused(self):
        body = {"prompt": "A red fox."}
        r, _ = self.ask("POST", "/api/jobs", body, {"Origin": "http://evil.example"})
        self.assertEqual(r.status, 403)
        r, _ = self.ask("POST", "/api/jobs", body, {"Origin": "null"})
        self.assertEqual(r.status, 403)
        r, _ = self.ask("GET", "/api/models", headers={"Host": "evil.example:8765"})
        self.assertEqual(r.status, 403)
        r, _ = self.ask("POST", "/api/jobs", body,
                        {"Origin": "http://127.0.0.1:%d" % self.port})
        self.assertEqual(r.status, 200)
        self.assertEqual(len(self.phone.jobs), 1)             # only the last was made
        settle(list(self.phone.jobs.values()))
        # A form post from a page cannot be JSON without asking first.
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        self.addCleanup(c.close)
        c.request("POST", "/api/jobs", body=b'{"prompt": "x"}',
                  headers={"Content-Type": "text/plain"})
        self.assertEqual(c.getresponse().status, 415)

    def test_an_unpaired_phone_gets_the_page_and_nothing_else(self):
        self.access.check = lambda address, cookie="": "yes" if cookie == "paired" \
            else "passcode"
        self.access.pair = lambda address, passcode: "paired" if passcode == "123456" \
            else (_ for _ in ()).throw(phone.Refused("That is not the passcode.", 401))
        self.assertEqual(self.ask("GET", "/")[0].status, 200)
        self.assertEqual(json.loads(self.ask("GET", "/api/access")[1]), {"paired": False})
        for method, path, body in (("GET", "/api/models", None), ("GET", "/api/pictures", None),
                                   ("POST", "/api/jobs", {"prompt": "x"}),
                                   ("POST", "/api/chat", {"messages": []})):
            self.assertEqual(self.ask(method, path, body)[0].status, 401, path)
        self.assertEqual(self.ask("POST", "/api/pair", {"passcode": "000000"})[0].status, 401)
        r, _ = self.ask("POST", "/api/pair", {"passcode": "123456"})
        self.assertEqual(r.status, 200)
        cookie = r.getheader("Set-Cookie")
        self.assertIn("HttpOnly", cookie)
        self.assertIn("SameSite=Strict", cookie)
        r, _ = self.ask("GET", "/api/models", headers={"Cookie": cookie.split(";")[0]})
        self.assertEqual(r.status, 200)

    def test_a_stranger_is_refused_everything(self):
        self.access.check = lambda address, cookie="": "no"
        self.assertEqual(self.ask("GET", "/")[0].status, 403)
        self.assertEqual(self.ask("POST", "/api/pair", {"passcode": "1"})[0].status, 403)


class TestPlan(unittest.TestCase):
    def setUp(self):
        self.old = (phone.tailnet_address, phone.tailnet_name, phone.lan_addresses)
        self.addCleanup(lambda: setattr(phone, "tailnet_address", self.old[0]))
        self.addCleanup(lambda: setattr(phone, "tailnet_name", self.old[1]))
        self.addCleanup(lambda: setattr(phone, "lan_addresses", self.old[2]))
        phone.tailnet_address = lambda: "100.96.251.56"
        phone.tailnet_name = lambda: "desktop-x.tail1234.ts.net"
        phone.lan_addresses = lambda: ["192.168.40.10"]

    def test_by_default_only_the_tailnet_and_this_pc_are_listened_on(self):
        binds, links = phone.plan()
        self.assertEqual(binds, ["127.0.0.1", "100.96.251.56"])
        self.assertEqual(links, ["http://100.96.251.56:8765", "http://desktop-x:8765"])

    def test_without_tailscale_only_this_pc(self):
        phone.tailnet_address = lambda: None
        self.assertEqual(phone.plan(), (["127.0.0.1"], []))

    def test_lan_listens_everywhere_and_links_the_home_address_too(self):
        binds, links = phone.plan(lan=True, port=9000)
        self.assertEqual(binds, ["0.0.0.0"])
        self.assertEqual(links, ["http://100.96.251.56:9000", "http://desktop-x:9000",
                                 "http://192.168.40.10:9000"])


if __name__ == "__main__":
    unittest.main()
