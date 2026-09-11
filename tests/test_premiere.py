"""
The Premiere Pro bridge: a CEP panel that evaluates what studio_premiere_mcp.py
posts to it. Nothing here reaches Premiere: the host's `run` is replaced and
the tool bodies are checked as text, and the one HTTP server a test starts is a
fake panel on loopback that answers what the test tells it to.
"""

import http.server
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import studio_agent as eng               # noqa: E402
import studio_cep                        # noqa: E402
import studio_premiere_mcp as ppro       # noqa: E402


class FakePanel:
    """The panel's HTTP contract, with a canned answer instead of ExtendScript."""

    def __init__(self, answer="null", status=200, delay=0.0):
        self.answer, self.status, self.delay = answer, status, delay
        self.scripts = []
        panel = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _reply(self, code, obj):
                body = json.dumps(obj).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                self._reply(200, {"bridge": "studio-premiere", "app": "PPRO", "version": "27.0", "port": 0})

            def do_POST(self):
                if not self.headers.get("Content-Type", "").startswith("application/json"):
                    return self._reply(415, {"error": "Content-Type must be application/json"})
                data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                panel.scripts.append(data["script"])
                time.sleep(panel.delay)
                if panel.status != 200:
                    return self._reply(panel.status, {"error": "boom"})
                self._reply(200, {"result": panel.answer})

        class Server(http.server.HTTPServer):
            def handle_error(self, request, client_address):
                # A client that gave up (the timeout test) leaves the handler
                # writing to a dead socket; that is the point, not a failure.
                if not isinstance(sys.exc_info()[1], (ConnectionError, OSError)):
                    super().handle_error(request, client_address)

        self.server = Server(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = "http://127.0.0.1:%d" % self.server.server_address[1]

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class TestRegistry(unittest.TestCase):
    def test_the_entry_points_at_the_script_and_probes_the_panel_port(self):
        app = eng.APPS_BY_ID["premiere"]
        self.assertEqual(app.command, eng.sys.executable)
        self.assertEqual(os.path.basename(app.args[0]), "studio_premiere_mcp.py")
        self.assertTrue(os.path.isfile(app.args[0]))
        self.assertEqual(app.probe, "port:" + eng.PREMIERE_PORT)
        self.assertIn(eng.PREMIERE_PORT, app.bridge_label)
        self.assertTrue(app.exe_globs[0].endswith("Adobe Premiere Pro (Beta).exe"),
                        "the Beta is looked for first; its exe is named differently")
        self.assertFalse(app.remote)
        self.assertIn("--install-panel", app.launch_note)

    def test_the_prompt_teaches_the_units_that_fail_silently(self):
        prompt = eng.PPRO_PROMPT
        for needle in ("SECONDS", "count from 1", "[0.5, 0.5]", "clip_id", "item_id", "ppro_status"):
            self.assertIn(needle, prompt)

    def test_groups_and_bridge_expose_the_same_tools(self):
        exposed = {t["name"] for t in ppro.tool_list()}
        grouped = {n for g in eng.PPRO_GROUPS.values() for n in g}
        self.assertEqual(exposed, grouped)
        self.assertEqual(ppro.READ_ONLY, set(eng.PPRO_GROUPS["discover"]))


class TestPanel(unittest.TestCase):
    def test_the_manifest_targets_premiere_and_enables_node(self):
        root = ET.parse(ppro.HOST.panel_src + "/CSXS/manifest.xml").getroot()
        self.assertEqual(root.get("ExtensionBundleId"), ppro.PANEL_ID)
        hosts = [h.get("Name") for h in root.iter("Host")]
        self.assertEqual(hosts, ["PPRO"])
        params = [p.text for p in root.iter("Parameter")]
        self.assertIn("--enable-nodejs", params)
        main_path = next(root.iter("MainPath")).text
        self.assertTrue(os.path.isfile(os.path.join(ppro.HOST.panel_src, main_path)))
        with open(os.path.join(ppro.HOST.panel_src, "main.js"), encoding="utf-8") as fh:
            js = fh.read()
        self.assertIn('server.listen(port, "127.0.0.1"', js, "loopback only")
        self.assertIn("application/json", js, "a browser page cannot post that without a preflight")
        self.assertIn("STUDIO_PREMIERE_PORT", js, "the panel and the bridge read the same variable")

    def test_install_copies_the_panel_under_cep_extensions(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        real = os.environ.get("APPDATA")
        os.environ["APPDATA"] = tmp
        try:
            self.assertFalse(ppro.HOST.panel_installed())
            dest = ppro.HOST.install_panel()
            self.assertTrue(ppro.HOST.panel_installed())
            self.assertEqual(dest, os.path.join(tmp, "Adobe", "CEP", "extensions", ppro.PANEL_ID))
            self.assertTrue(os.path.isfile(os.path.join(dest, "main.js")))
            # A second install replaces the first rather than failing on it.
            ppro.HOST.install_panel()
        finally:
            if real is None:
                del os.environ["APPDATA"]
            else:
                os.environ["APPDATA"] = real

    def test_unreachable_is_explained_by_what_is_actually_missing(self):
        host = studio_cep.CepHost("http://127.0.0.1:1", "Premiere Pro", "Nothing.exe",
                                  ppro.PANEL_ID, ppro.HOST.panel_src)
        host.running = lambda: False
        self.assertIn("not running", host.explain_unreachable())
        host.running = lambda: True
        host.panel_installed = lambda: False
        self.assertIn("--install-panel", host.explain_unreachable())
        host.panel_installed = lambda: True
        real = studio_cep.debug_mode_missing
        studio_cep.debug_mode_missing = lambda versions=None: ["12"]
        try:
            self.assertIn("PlayerDebugMode", host.explain_unreachable())
            self.assertIn("CSXS.12", host.explain_unreachable())
        finally:
            studio_cep.debug_mode_missing = real
        studio_cep.debug_mode_missing = lambda versions=None: []
        try:
            self.assertIn("Window > Extensions", host.explain_unreachable("refused"))
        finally:
            studio_cep.debug_mode_missing = real


class TestTransport(unittest.TestCase):
    """The real HTTP road, against a fake panel on loopback."""

    def host(self, panel):
        h = studio_cep.CepHost(panel.url, "Premiere Pro", "Nothing.exe", ppro.PANEL_ID, ppro.HOST.panel_src)
        h.running = lambda: True
        return h

    def test_a_body_is_wrapped_posted_and_decoded(self):
        panel = FakePanel(answer='{"a": [1, 2]}')
        self.addCleanup(panel.close)
        self.assertEqual(self.host(panel).run("return {a: [1, 2]}", setup="SETUP;"), {"a": [1, 2]})
        js = panel.scripts[-1]
        for needle in ("function __J(", "SETUP;", "__error", "return {a: [1, 2]}"):
            self.assertIn(needle, js)
        self.assertEqual(self.host(panel).ping()["bridge"], "studio-premiere")

    def test_a_script_error_and_a_parse_error_are_sentences(self):
        panel = FakePanel(answer='{"__error": "No clip with clip_id 9", "line": 4}')
        self.addCleanup(panel.close)
        with self.assertRaises(studio_cep.CepError) as ctx:
            self.host(panel).run("return 1")
        self.assertIn("clip_id 9", str(ctx.exception))
        self.assertIn("line 4", str(ctx.exception))
        panel.answer = "EvalScript error."
        with self.assertRaises(studio_cep.CepError) as ctx:
            self.host(panel).run("return 1")
        self.assertIn("syntax", str(ctx.exception))
        panel.status = 500
        with self.assertRaises(studio_cep.CepError) as ctx:
            self.host(panel).run("return 1")
        self.assertIn("HTTP 500", str(ctx.exception))

    def test_a_silent_panel_is_a_dialog_not_a_hang(self):
        panel = FakePanel(answer="1", delay=1.5)
        self.addCleanup(panel.close)
        with self.assertRaises(studio_cep.CepError) as ctx:
            self.host(panel).run("return 1", timeout=0.3)
        self.assertIn("dialog", str(ctx.exception))

    def test_nothing_listening_is_explained(self):
        panel = FakePanel()
        url = panel.url
        panel.close()
        h = studio_cep.CepHost(url, "Premiere Pro", "Nothing.exe", ppro.PANEL_ID, ppro.HOST.panel_src)
        h.running = lambda: False
        with self.assertRaises(studio_cep.CepError) as ctx:
            h.run("return 1")
        self.assertIn("not running", str(ctx.exception))


class TestTools(unittest.TestCase):
    """Tool bodies as text, with the host's `run` replaced."""

    def setUp(self):
        self.calls = []
        self._real = (ppro.HOST.run, ppro.HOST.ping, ppro.running)

    def tearDown(self):
        ppro.HOST.run, ppro.HOST.ping, ppro.running = self._real

    def answer(self, value, running=True, ping=None):
        def run(body, timeout=None, setup="", teardown=""):
            self.calls.append(body)
            return value
        ppro.HOST.run = run
        ppro.running = lambda: running
        if ping is not None:
            ppro.HOST.ping = ping

    def test_status_never_touches_the_panel_when_premiere_is_closed(self):
        self.answer({"running": True}, running=False)
        res = ppro.call_tool("ppro_status", {})
        self.assertEqual(self.calls, [])
        self.assertIn("not running", res["content"][0]["text"])
        self.assertFalse(res.get("isError"))

    def test_status_relays_why_the_panel_is_not_answering(self):
        def ping(timeout=3):
            raise studio_cep.CepError("its bridge panel is not installed")
        self.answer({}, running=True, ping=ping)
        res = ppro.call_tool("ppro_status", {})
        self.assertEqual(self.calls, [])
        self.assertIn("not installed", res["content"][0]["text"])
        self.assertFalse(res.get("isError"), "a status is an answer, not a failure")
        self.assertFalse(res["structuredContent"]["reachable"])

    def test_clips_are_addressed_by_id_and_a_bad_id_is_a_sentence(self):
        clip = {"clip_id": "c1", "name": "shot", "track": "V1", "kind": "video", "track_index": 1,
                "index": 0, "start": 0, "end": 4, "in_point": 0, "out_point": 4, "duration": 4}
        self.answer(clip)
        res = ppro.call_tool("ppro_set_clip", {"clip_id": "c1", "start": 2.5})
        self.assertIn('__clip(sq, "c1")', self.calls[-1])
        self.assertIn("c.start = __time(2.5)", self.calls[-1])
        self.assertIn("clip_id c1", res["content"][0]["text"])
        res = ppro.call_tool("ppro_set_clip", {"clip_id": "c1"})
        self.assertTrue(res["isError"])
        self.assertIn("nothing to set", res["content"][0]["text"])

        def refuse(body, **kw):
            raise studio_cep.CepError("No clip with clip_id zz in sequence Cut")
        ppro.HOST.run = refuse
        res = ppro.call_tool("ppro_remove_clip", {"clip_id": "zz"})
        self.assertTrue(res["isError"])
        self.assertIn("clip_id zz", res["content"][0]["text"])

    def test_tracks_count_from_one_on_the_way_in(self):
        self.answer({"track": "V2", "kind": "video", "index": 2, "name": "Video 2", "clips": 0, "muted": True})
        ppro.call_tool("ppro_set_track", {"kind": "video", "track_index": 2, "muted": True})
        self.assertIn('__track(sq, "video", 2)', self.calls[-1])
        self.assertIn("setMute(true ? 1 : 0)", self.calls[-1])
        self.answer({"sequence": "Cut", "time": 0, "added": []})
        ppro.call_tool("ppro_add_to_sequence", {"item_id": "i1", "video_track": 2, "audio_track": 3,
                                                 "mode": "overwrite"})
        self.assertIn("var v = 2, au = 3;", self.calls[-1])
        self.assertIn("overwriteClip(it, __time(t), v - 1, au - 1)", self.calls[-1])

    def test_a_keyframe_needs_a_time_and_a_flat_value_clears_them(self):
        self.answer({"clip": {"clip_id": "c1", "name": "shot", "track": "V1", "start": 0, "end": 1},
                     "effect": "Motion", "property": "Scale", "value": 50, "keyframed": True})
        ppro.call_tool("ppro_set_clip_property", {"clip_id": "c1", "effect": "Motion", "property": "Scale",
                                                   "value": 50, "time": 1.5})
        self.assertIn("setValueAtKey(t, v, true)", self.calls[-1])
        self.assertIn("__time(1.5)", self.calls[-1])
        ppro.call_tool("ppro_set_clip_property", {"clip_id": "c1", "effect": "Motion", "property": "Position",
                                                   "value": [0.5, 0.5]})
        self.assertIn("var v = [0.5, 0.5];", self.calls[-1])
        self.assertIn("setTimeVarying(false)", self.calls[-1])
        self.assertNotIn("addKey", self.calls[-1])

    def test_get_sequence_prints_ids_seconds_and_timecode(self):
        self.answer({"name": "Cut", "width": 1920, "height": 1080, "fps": 25, "duration": 10, "playhead": 2.48,
                     "active": True, "in_point": None, "out_point": None,
                     "tracks": [{"track": "V1", "name": "Video 1", "clips": 1, "muted": False, "locked": True}],
                     "clips": [{"clip_id": "abc", "name": "shot", "track": "V1", "kind": "video", "track_index": 1,
                                "index": 0, "start": 0, "end": 4, "in_point": 1, "out_point": 5, "duration": 4}],
                     "markers": [{"name": "Review", "start": 3, "end": 3, "comments": "", "type": "Comment"}]})
        res = ppro.call_tool("ppro_get_sequence", {"sequence": "Cut"})
        text = res["content"][0]["text"]
        self.assertIn('__seq("Cut")', self.calls[-1])
        self.assertIn("playhead at 2.48s (00:00:02:12)", text)
        self.assertIn("clip_id abc", text)
        self.assertIn("locked", text)
        self.assertIn("source 1-5s", text)
        self.assertIn("marker 'Review' at 3s", text)

    def test_save_as_and_export_refuse_to_overwrite_unless_told(self):
        self.answer({"name": "p", "path": None})
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        taken = os.path.join(tmp, "taken.prproj")
        open(taken, "wb").close()
        res = ppro.call_tool("ppro_save_as", {"path": taken})
        self.assertTrue(res["isError"])
        self.assertIn("overwrite=true", res["content"][0]["text"])
        self.assertEqual(self.calls, [])
        preset = os.path.join(tmp, "H264 Match Source - High bitrate.epr")
        open(preset, "wb").close()
        out = os.path.join(tmp, "out.mp4")
        open(out, "wb").close()
        res = ppro.call_tool("ppro_export", {"preset": preset, "output_path": out})
        self.assertTrue(res["isError"])
        self.assertIn("overwrite=true", res["content"][0]["text"])
        self.assertEqual(self.calls, [])

    def test_export_resolves_a_preset_name_and_names_the_ambiguity(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        for folder, name in (("4E49434B_48323634", "Match Source - High bitrate"),
                             ("3F3F3F3F_4D6F6F56", "Match Source - High bitrate"),
                             ("4E49434B_48323634", "YouTube 1080p Full HD")):
            os.makedirs(os.path.join(tmp, folder), exist_ok=True)
            open(os.path.join(tmp, folder, name + ".epr"), "wb").close()
        real = ppro.PRESET_GLOBS
        ppro.PRESET_GLOBS = [os.path.join(tmp, "*", "*.epr")]
        self.addCleanup(setattr, ppro, "PRESET_GLOBS", real)
        presets = ppro.list_presets()
        self.assertEqual({p["format"] for p in presets}, {"H264", "MooV"})
        self.assertEqual(len(ppro.list_presets("youtube")), 1)
        res = ppro.call_tool("ppro_list_presets", {"search": "h264"})
        self.assertIn("H264: YouTube 1080p Full HD", res["content"][0]["text"])

        self.answer({"sequence": "Cut", "output": "x", "result": "No Error", "seconds": 3})
        res = ppro.call_tool("ppro_export", {"preset": "Match Source - High bitrate",
                                             "output_path": os.path.join(tmp, "a.mp4")})
        self.assertTrue(res["isError"])
        self.assertIn("2 presets match", res["content"][0]["text"])
        self.assertEqual(self.calls, [])
        res = ppro.call_tool("ppro_export", {"preset": "nothing like this",
                                             "output_path": os.path.join(tmp, "a.mp4")})
        self.assertTrue(res["isError"])
        self.assertIn("ppro_list_presets", res["content"][0]["text"])

        # A unique name resolves to its path; queue=true goes to Media Encoder.
        self.answer({"sequence": "Cut", "job": "j1", "output": os.path.join(tmp, "b.mp4")})
        res = ppro.call_tool("ppro_export", {"preset": "YouTube 1080p Full HD",
                                             "output_path": os.path.join(tmp, "b.mp4"), "queue": True})
        self.assertFalse(res.get("isError"))
        self.assertIn("YouTube 1080p Full HD.epr", self.calls[-1].replace("/", os.sep))
        self.assertIn("encodeSequence", self.calls[-1])
        self.assertIn("job j1", res["content"][0]["text"])

    def test_a_sequence_is_made_from_a_preset_through_qe_never_the_dialog(self):
        """app.project.createNewSequence opens the New Sequence dialog in Premiere 27
        and hangs the panel behind it; QE's newSequence with a preset is silent, and
        only takes the preset as File.fsName."""
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        for name in ("HD 1080p 23.976 fps", "HD 1080p 25 fps", "HD 1080p 29.97 fps", "HD 1080p 50 fps",
                     "HD 1080p 59.94 fps", "Social Media Square 1x1 30 fps"):
            open(os.path.join(tmp, name + ".sqpreset"), "wb").close()
        real = ppro.SEQ_PRESET_GLOBS
        ppro.SEQ_PRESET_GLOBS = [os.path.join(tmp, "*.sqpreset")]
        self.addCleanup(setattr, ppro, "SEQ_PRESET_GLOBS", real)
        self.assertTrue(ppro.sequence_preset({}).endswith("HD 1080p 25 fps.sqpreset"))
        self.assertTrue(ppro.sequence_preset({"fps": 24}).endswith("HD 1080p 23.976 fps.sqpreset"))
        self.assertTrue(ppro.sequence_preset({"fps": 30}).endswith("HD 1080p 29.97 fps.sqpreset"))
        self.assertTrue(ppro.sequence_preset({"preset": "square"}).endswith("Square 1x1 30 fps.sqpreset"))
        with self.assertRaises(studio_cep.CepError):
            ppro.sequence_preset({"preset": "hd 1080p"})            # ambiguous
        self.answer({"name": "Cut", "width": 1280, "height": 720, "fps": 50, "active": True, "note": ""})
        res = ppro.call_tool("ppro_new_sequence", {"name": "Cut", "width": 1280, "height": 720, "fps": 50})
        body = self.calls[-1]
        self.assertNotIn("createNewSequence(", body)
        self.assertIn("__qe().project.newSequence(", body)
        self.assertIn(".fsName", body)
        self.assertIn("HD 1080p 50 fps", body)
        self.assertIn("st.videoFrameWidth = 1280", body)
        self.assertIn("__TPS / 50", body)
        self.assertIn("1280x720 @ 50fps", res["content"][0]["text"])
        ppro.call_tool("ppro_new_sequence", {"name": "Cut", "item_ids": ["a", "b"]})
        self.assertIn("createNewSequenceFromClips", self.calls[-1])

    def test_paths_reach_premiere_as_native_file_names(self):
        """QE refuses a forward-slash path string and appends .png itself."""
        self.answer({"sequence": "Cut", "time": 1, "timecode": "00:00:01:00", "width": 1, "height": 1})
        real = ppro.PREVIEW_DIR
        ppro.PREVIEW_DIR = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, ppro.PREVIEW_DIR, True)
        self.addCleanup(setattr, ppro, "PREVIEW_DIR", real)
        real_wait = ppro.com.wait_for_file
        def fake_wait(path, timeout=30):
            with open(path, "wb") as fh:
                fh.write(b"png")
            return True
        ppro.com.wait_for_file = fake_wait
        self.addCleanup(setattr, ppro.com, "wait_for_file", real_wait)
        res = ppro.call_tool("ppro_screenshot", {"time": 1})
        self.assertFalse(res.get("isError"), res["content"][0]["text"])
        self.assertIn("exportFramePNG(__tc(sq, t), new File(", self.calls[-1])
        self.assertNotIn('.png").fsName', self.calls[-1], "QE appends the extension itself")
        self.assertEqual(res["content"][1]["type"], "image")

    def test_a_marker_end_is_a_number_of_seconds(self):
        """marker.end refuses a Time object; a plain number works."""
        self.answer({"name": "Fix", "start": 7, "end": 8, "comments": "", "sequence": "Cut"})
        ppro.call_tool("ppro_add_marker", {"time": 7, "name": "Fix", "duration": 1, "color": 3})
        self.assertIn("m.end = t + 1;", self.calls[-1])
        self.assertNotIn("m.end = __time", self.calls[-1])
        self.assertIn("setColorByIndex(3)", self.calls[-1])

    def test_import_checks_the_files_exist_before_touching_premiere(self):
        self.answer([])
        res = ppro.call_tool("ppro_import_files", {"paths": [r"C:\no\such\file.mov"]})
        self.assertTrue(res["isError"])
        self.assertIn("No file at", res["content"][0]["text"])
        self.assertEqual(self.calls, [])

    def test_run_jsx_returns_what_the_script_returned(self):
        self.answer({"n": 3})
        res = ppro.call_tool("ppro_run_jsx", {"code": "return {n: app.project.sequences.numSequences}"})
        self.assertEqual(json.loads(res["content"][0]["text"]), {"n": 3})
        self.assertIn("app.project.sequences.numSequences", self.calls[-1])


if __name__ == "__main__":
    unittest.main()
