import io
import json
import os
import sys
import tempfile
import unittest
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import studio_agent as eng
import studio_appinfo as appinfo


class Page:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return io.BytesIO(json.dumps(self.body).encode("utf-8"))

    def __exit__(self, *a):
        return False


def opener_for(pages, seen=None):
    def open_(req, timeout=None):
        title = urllib.parse.unquote(req.full_url.rsplit("/", 1)[-1])
        if seen is not None:
            seen.append(title)
        if title not in pages:
            raise OSError("404")
        return Page(pages[title])
    return open_


def page(extract, kind="standard"):
    return {"type": kind, "extract": extract,
            "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/X"}}}


class FakeApp:
    def __init__(self, id, name, exe=None):
        self.id, self.name, self.tab, self._exe = id, name, name, exe

    def exe(self):
        return self._exe


class TestAppInfo(unittest.TestCase):
    def setUp(self):
        self.base = tempfile.mkdtemp()

    def test_known_app_profile_renders_and_caches(self):
        app = FakeApp("premiere", "Premiere Pro")
        seen = []
        data = appinfo.refresh(app, self.base, opener_for(
            {"Adobe_Premiere_Pro": page("Adobe Premiere Pro is a video editing app.")}, seen),
            now=1000.0)
        text = appinfo.render(data)
        self.assertIn("Premiere Pro tab", text)
        self.assertIn("video editing", text)
        self.assertEqual(appinfo.load("premiere", self.base)["wiki_checked"], 1000.0)
        # Within the month: no second fetch.
        appinfo.refresh(app, self.base, opener_for({}, seen), now=2000.0)
        self.assertEqual(seen, ["Adobe_Premiere_Pro"])

    def test_failed_fetch_keeps_old_overview(self):
        app = FakeApp("resolve", "DaVinci Resolve")
        appinfo.refresh(app, self.base, opener_for({"DaVinci_Resolve": page("An editor.")}), now=appinfo.WIKI_MAX_AGE + 1)
        data = appinfo.refresh(app, self.base, opener_for({}), now=3 * appinfo.WIKI_MAX_AGE)
        self.assertEqual(data["wiki"], "An editor.")

    def test_unknown_app_needs_a_software_page(self):
        app = FakeApp("custom", "Blender")
        pages = {"Blender_(software)": page("Blender is a 3D computer graphics software tool."),
                 "Blender": page("A blender is a kitchen appliance.")}
        self.assertIn("3D", appinfo.overview(app, opener_for(pages))[0])
        app = FakeApp("custom2", "Mixer")
        self.assertIsNone(appinfo.overview(app, opener_for(
            {"Mixer": page("A mixer is a kitchen appliance.")})))
        self.assertIsNone(appinfo.overview(app, opener_for(
            {"Mixer": page("Mixer may refer to software or...", "disambiguation")})))

    def test_no_page_app_is_not_fetched(self):
        seen = []
        self.assertIsNone(appinfo.overview(FakeApp("opencode", "OpenCode"), opener_for({}, seen)))
        self.assertEqual(seen, [])

    def test_release_labels_beta_and_year(self):
        app = FakeApp("x", "X", r"C:\Program Files\Adobe\Adobe Premiere Pro (Beta)\nothere.exe")
        self.assertIsNone(appinfo.release(app))      # no file, no claim
        self.assertEqual(appinfo.render({}), "")

    def test_section_goes_in_the_prompt(self):
        self.assertIn("ABOUT THE APP THIS TAB DRIVES", eng.about_section("- x"))
        self.assertEqual(eng.about_section(""), "")


if __name__ == "__main__":
    unittest.main()
