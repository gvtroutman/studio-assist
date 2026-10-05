import io
import json
import os
import sys
import unittest
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.icon_search as icon_search
import core.icons as icons

PNG = icons.png(bytes([0, 0, 200, 255]) * 16, 4, 4)


def page(title, mime, thumb, index, license="Public domain"):
    return {"title": "File:" + title, "index": index, "imageinfo": [{
        "mime": mime, "thumburl": thumb, "descriptionurl": "https://commons.wikimedia.org/x",
        "extmetadata": {"LicenseShortName": {"value": license}}}]}


def opener(body, seen=None):
    def open_(req, timeout=None):
        if seen is not None:
            seen.append(req)
        if isinstance(body, Exception):
            raise body
        return io.BytesIO(body if isinstance(body, bytes)
                          else json.dumps(body).encode("utf-8"))
    return open_


THUMB = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ab/%s/250px-%s.png"


class TestSearch(unittest.TestCase):
    def test_drawings_come_first_and_photos_are_left_out(self):
        pages = [page("Shot.png", "image/png", THUMB % ("s", "s"), 1),
                 page("HQ.jpg", "image/jpeg", THUMB % ("h", "h"), 2),
                 page("Logo.svg", "image/svg+xml", THUMB % ("l", "l"), 3, "GPL")]
        seen = []
        hits = icon_search.search("blender logo", opener=opener(
            {"query": {"pages": pages}}, seen))
        self.assertEqual([h["title"] for h in hits], ["Logo.svg", "Shot.png"])
        self.assertEqual(hits[0]["license"], "GPL")
        sent = urllib.parse.parse_qs(urllib.parse.urlsplit(seen[0].full_url).query)
        self.assertEqual(sent["gsrsearch"], ["blender logo filetype:drawing|bitmap"])
        self.assertEqual(sent["gsrnamespace"], ["6"])
        self.assertTrue(seen[0].get_header("User-agent"))

    def test_a_thumbnail_off_wikimedia_is_never_offered(self):
        pages = [page("Elsewhere.svg", "image/svg+xml", "https://evil.example/x.png", 1),
                 page("Plain.svg", "image/svg+xml", "http://upload.wikimedia.org/x.png", 2)]
        self.assertEqual(icon_search.search("x", opener=opener(
            {"query": {"pages": pages}})), [])

    def test_nothing_found_and_nothing_typed_are_empty(self):
        self.assertEqual(icon_search.search("zzz", opener=opener({"batchcomplete": True})), [])
        self.assertEqual(icon_search.search("   ", opener=opener(OSError("unused"))), [])

    def test_no_network_says_so(self):
        with self.assertRaises(icon_search.SearchError) as got:
            icon_search.search("x", opener=opener(OSError("offline")))
        self.assertIn("Wikimedia Commons", str(got.exception))

    def test_the_words_an_app_and_a_button_start_with(self):
        self.assertEqual(icon_search.words_for("  Adobe   Photoshop "), "Adobe Photoshop logo")
        self.assertEqual(icon_search.words_for("Send", logo=False), "Send icon")


class TestFetch(unittest.TestCase):
    def test_a_png_from_wikimedia_is_fetched(self):
        self.assertEqual(icon_search.fetch(THUMB % ("a", "a"), opener=opener(PNG)), PNG)

    def test_anything_else_is_refused(self):
        for url, body in (("https://evil.example/a.png", PNG),
                          (THUMB % ("a", "a"), b"<html>not a picture</html>"),
                          (THUMB % ("a", "a"), PNG + bytes(icon_search.MAX_BYTES)),
                          (THUMB % ("a", "a"), OSError("offline"))):
            with self.assertRaises(icon_search.SearchError):
                icon_search.fetch(url, opener=opener(body))

    def test_only_https_on_wikimedia_hosts_is_ours(self):
        self.assertTrue(icon_search.ours("https://upload.wikimedia.org/x.png"))
        self.assertTrue(icon_search.ours("https://thumb.wikimedia.org/x.png"))
        for url in ("http://upload.wikimedia.org/x.png", "https://wikimedia.org.evil.example/x",
                    "https://evilwikimedia.org/x", "", None, "file:///C:/x.png"):
            self.assertFalse(icon_search.ours(url), url)


if __name__ == "__main__":
    unittest.main()
