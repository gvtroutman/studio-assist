"""Add-ons (studio_catalog): which LoRAs fit which model, the CivitAI catalog
filtered to a model's base models, safe thumbnails, and uninstalling to the
Recycle Bin. CivitAI is a table of canned answers; the one live piece is the
PowerShell picture conversion, run on a BMP this test writes."""

import base64
import os
import struct
import sys
import tempfile
import unittest
import urllib.parse
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import apps.image_studio.addons.catalog as catalog  # noqa: E402
import apps.image_studio.addons.civitai as civitai  # noqa: E402
import apps.image_studio.imagegen as ig  # noqa: E402

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
JPEG_ISH = b"\xff\xd8\xff\xe0 not really a jpeg"


def bmp(w=4, h=3):
    """A 24-bit BMP, which Windows' System.Drawing reads."""
    row = (b"\x00\x80\xff" * w).ljust((w * 3 + 3) & ~3, b"\x00")
    pixels = row * h
    head = struct.pack("<2sIHHI", b"BM", 54 + len(pixels), 0, 0, 54)
    info = struct.pack("<IiiHHIIiiII", 40, w, h, 1, 24, 0, len(pixels), 2835, 2835, 0, 0)
    return head + info + pixels


def version(vid, base, level=1, name="v1"):
    return {"id": vid, "name": name, "baseModel": base, "trainedWords": ["trig"],
            "description": "<p>Does a thing.</p>",
            "files": [{"name": "lora_%d.safetensors" % vid, "primary": True, "sizeKB": 2048,
                       "hashes": {"SHA256": ("%064x" % vid).upper()}}],
            "images": [{"url": "https://image.civitai.com/x/u/original=true/%d-spicy.jpeg" % vid,
                        "nsfwLevel": 8, "type": "image"},
                       {"url": "https://image.civitai.com/x/u/original=true/%d.jpeg" % vid,
                        "nsfwLevel": level, "type": "image"}]}


HIT = {"id": 42, "name": "Hands", "type": "LORA", "creator": {"username": "maker"},
       "stats": {"downloadCount": 307110}, "tags": ["hands", "tool"],
       "modelVersions": [version(3, "SDXL 1.0"), version(2, "Flux.1 D"),
                         version(1, "ZImageBase")]}


class FakeClient:
    def __init__(self, hits=(HIT,), nxt="c2"):
        self.hits, self.nxt, self.calls = list(hits), nxt, []

    def search(self, bases, query="", sort="", cursor=""):
        self.calls.append((tuple(bases), query, sort, cursor))
        return self.hits, self.nxt


def lib_with(loras, root=None):
    lib = ig.Library(root or tempfile.mkdtemp())
    lib.save("loras", loras)
    return lib


class TestFits(unittest.TestCase):
    def setUp(self):
        self.lib = ig.Library(tempfile.mkdtemp())
        self.flux = self.lib.get("models", "flux-dev")
        self.zit = self.lib.get("models", "z-image-turbo")

    def test_a_lora_fits_only_its_family_and_kontext_pairs_with_flux(self):
        self.assertTrue(ig.lora_fits({"family": "flux1"}, self.flux))
        self.assertFalse(ig.lora_fits({"family": "flux1"}, self.zit))
        self.assertTrue(ig.lora_fits({"family": "flux1-kontext"}, self.flux))
        self.assertFalse(ig.lora_fits({"family": "qwen-image"}, self.zit))
        self.assertIsNone(ig.lora_fits({"family": ""}, self.zit))
        self.assertIsNone(ig.lora_fits({"family": "flux1"}, None))

    def test_a_backend_that_runs_the_model_as_another_family_counts(self):
        model = dict(self.zit, backends={"3090": {"family": "krea2"}})
        self.assertEqual(ig.model_families(model), {"z-image", "krea2"})
        self.assertTrue(ig.lora_fits({"family": "krea2"}, model))

    def test_enabled_defaults_on_and_off_is_kept(self):
        self.assertTrue(ig.clean_lora({"file": "a.safetensors"})["enabled"])
        self.assertFalse(ig.clean_lora({"file": "a.safetensors", "enabled": False})["enabled"])

    def test_a_lora_turned_off_is_not_added_as_always_on(self):
        lib = lib_with([{"id": "skin", "file": "skin.safetensors", "family": "z-image",
                         "always": True, "enabled": False}])
        backend = lib.all("backends")[0]
        plan = ig.compose(dict(ig.default_settings(), model="z-image-turbo",
                               scene="a cat"), lib, backend)
        self.assertEqual(plan.loras, [])
        lib.get("loras", "skin")["enabled"] = True
        plan = ig.compose(dict(ig.default_settings(), model="z-image-turbo",
                               scene="a cat"), lib, backend)
        self.assertEqual([f for f, _ in plan.loras], ["skin.safetensors"])

    def test_installed_is_split_per_model(self):
        lib = lib_with([
            {"id": "z", "file": "z.safetensors", "name": "Zed", "family": "z-image"},
            {"id": "off", "file": "o.safetensors", "name": "Alpha", "family": "z-image",
             "enabled": False},
            {"id": "f", "file": "f.safetensors", "family": "flux1"},
            {"id": "q", "file": "q.safetensors", "family": "qwen-image"},
            {"id": "u", "file": "u.safetensors"}])
        fits, unknown = catalog.sorted_for(lib, lib.get("models", "z-image-turbo"))
        self.assertEqual([r["id"] for r in fits], ["z", "off"])      # on first
        self.assertEqual([r["id"] for r in unknown], ["u"])
        self.assertEqual([r["id"] for r in catalog.fits_none(lib)], ["q"])

    def test_where_installed_reads_the_checked_lists(self):
        rec = ig.clean_lora({"file": "a.safetensors", "files": {"3090": "sub/a.safetensors"}})
        backends = [{"id": "5090", "name": "5090"}, {"id": "3090", "name": "3090"},
                    {"id": "x", "name": "unchecked"}]
        inv = {"5090": {"loras": {"a.safetensors"}}, "3090": {"loras": {"a.safetensors"}}}
        self.assertEqual(catalog.where_installed(rec, backends, inv), (["5090"], ["3090"]))


class TestCatalog(unittest.TestCase):
    def test_the_card_is_the_newest_version_for_the_model(self):
        c = catalog.card(HIT, {"z-image"})
        self.assertEqual((c["version_id"], c["family"], c["base_model"]),
                         (1, "z-image", "ZImageBase"))
        self.assertEqual(c["file"], "lora_1.safetensors")
        self.assertEqual(c["sha256"], "%064x" % 1)
        self.assertEqual(c["link"], "https://civitai.com/models/42?modelVersionId=1")
        self.assertEqual(c["downloads"], 307110)
        self.assertEqual(c["trigger"], "trig")
        self.assertEqual(catalog.card(HIT, {"flux1"})["version_id"], 2)
        self.assertIsNone(catalog.card(HIT, {"qwen-image"}))

    def test_only_a_pg_picture_is_shown_and_asked_for_small(self):
        c = catalog.card(HIT, {"z-image"})
        self.assertEqual(c["preview_url"],
                         "https://image.civitai.com/x/u/width=320/1.jpeg")
        mature = dict(HIT, modelVersions=[version(9, "ZImageTurbo", level=4)])
        self.assertEqual(catalog.card(mature, {"z-image"})["preview_url"], "")

    def test_search_asks_for_the_models_base_names(self):
        lib = ig.Library(tempfile.mkdtemp())
        client = FakeClient()
        cards, nxt = catalog.search(client, lib.get("models", "z-image-turbo"), "hands",
                                    "Newest", "c1")
        self.assertEqual(client.calls, [(("ZImageTurbo", "ZImageBase"), "hands", "Newest",
                                         "c1")])
        self.assertEqual([c["version_id"] for c in cards], [1])
        self.assertEqual(nxt, "c2")
        self.assertEqual(catalog.bases_for(lib.get("models", "flux-dev")), ("Flux.1 D",))
        self.assertEqual(catalog.search(client, dict(lib.get("models", "flux-dev"),
                                                     family="mystery"))[0], [])

    def test_the_client_builds_civitais_query(self):
        seen = []

        class Resp:
            def __init__(self, body):
                self.body = body

            def read(self):
                return self.body

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def opener(req, timeout=None):
            seen.append(req.full_url)
            return Resp(b'{"items": [{"id": 1}], "metadata": {"nextCursor": "n"}}')
        items, nxt = civitai.Client(opener=opener).search(
            ("ZImageTurbo", "ZImageBase"), " skin ", "Bogus", "c")
        q = urllib.parse.parse_qs(urllib.parse.urlsplit(seen[0]).query)
        self.assertEqual(q["baseModels"], ["ZImageTurbo", "ZImageBase"])
        self.assertEqual((q["types"], q["nsfw"], q["query"], q["cursor"], q["sort"]),
                         (["LORA"], ["false"], ["skin"], ["c"], ["Most Downloaded"]))
        self.assertEqual((items, nxt), ([{"id": 1}], "n"))

    def test_installed_as_matches_hash_then_version_then_file(self):
        c = catalog.card(HIT, {"z-image"})
        by_hash = lib_with([{"id": "h", "file": "other.safetensors", "sha256": c["sha256"]}])
        self.assertEqual(catalog.installed_as(by_hash, c)["id"], "h")
        by_version = lib_with([{"id": "v", "file": "x.safetensors",
                                "source": "https://civitai.com/models/42?modelVersionId=1"}])
        self.assertEqual(catalog.installed_as(by_version, c)["id"], "v")
        other_version = lib_with([{"id": "v", "file": "x.safetensors",
                                   "source": "https://civitai.com/models/42?modelVersionId=12"}])
        self.assertIsNone(catalog.installed_as(other_version, c))
        by_file = lib_with([{"id": "f", "file": "lora_1.safetensors"}])
        self.assertEqual(catalog.installed_as(by_file, c)["id"], "f")

    def test_install_is_the_importers_link_path(self):
        c = catalog.card(HIT, {"z-image"})
        with mock.patch.object(civitai, "import_link", return_value=("rec", True)) as imp:
            self.assertEqual(catalog.install("lib", "client", c, "D:/loras"), ("rec", True))
        self.assertEqual(imp.call_args[0][:4], ("lib", "client", c["link"], "D:/loras"))

    def test_human_counts(self):
        self.assertEqual([catalog.human_count(n) for n in (7, 1500, 187485, 2300000)],
                         ["7", "1.5k", "187k", "2.3M"])


class TestPictures(unittest.TestCase):
    def test_png_is_kept_and_anything_else_is_converted_once(self):
        folder = tempfile.mkdtemp()
        bodies = {"https://p/1.png": PNG, "https://p/2.jpg": JPEG_ISH}

        def opened(client, url):
            class R:
                def read(self_):
                    return bodies[url]

                def __enter__(self_):
                    return self_

                def __exit__(self_, *a):
                    return False
            if url not in bodies:
                raise civitai.CivitAIError("gone")
            return R()

        def fake_png(pairs, side=catalog.THUMB):
            for src, dest in pairs:
                with open(src, "rb") as f:
                    self.assertEqual(f.read(), JPEG_ISH)
                with open(dest, "wb") as f:
                    f.write(PNG)
            return [d for _, d in pairs]
        with mock.patch.object(civitai, "_open_download", side_effect=opened), \
                mock.patch.object(catalog, "to_png", side_effect=fake_png) as conv:
            got = catalog.thumbnails(None, {1: "https://p/1.png", 2: "https://p/2.jpg",
                                            3: "https://p/gone.jpg", 4: ""}, folder)
            self.assertEqual(sorted(got), [1, 2])
            self.assertEqual(conv.call_count, 1)
            again = catalog.thumbnails(None, {1: "x", 2: "y"}, folder)   # cached
            self.assertEqual(sorted(again), [1, 2])
        self.assertEqual(sorted(os.listdir(folder)), ["1.png", "2.png"])  # no temp left

    def test_installed_previews_pass_png_through(self):
        folder = tempfile.mkdtemp()
        png, jpg = os.path.join(folder, "a.png"), os.path.join(folder, "b.jpg")
        for path, body in ((png, PNG), (jpg, JPEG_ISH)):
            with open(path, "wb") as f:
                f.write(body)
        recs = [{"id": "a", "preview": png}, {"id": "b", "preview": jpg},
                {"id": "c", "preview": ""}]
        with mock.patch.object(catalog, "to_png", return_value=[]) as conv:
            self.assertEqual(catalog.previews(recs, folder), {"a": png})
        (pairs,), _ = conv.call_args
        self.assertEqual([s for s, _ in pairs], [jpg])

    @unittest.skipUnless(sys.platform == "win32", "System.Drawing is Windows'")
    def test_windows_converts_a_picture_to_png(self):
        folder = tempfile.mkdtemp()
        src, dest = os.path.join(folder, "in.bmp"), os.path.join(folder, "out.png")
        with open(src, "wb") as f:
            f.write(bmp(400, 200))
        self.assertEqual(catalog.to_png([(src, dest)], side=100), [dest])
        with open(dest, "rb") as f:
            head = f.read(24)
        self.assertEqual(head[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(struct.unpack(">II", head[16:24]), (100, 50))


class TestUninstall(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.lib = lib_with([{"id": "a", "file": "a.safetensors", "name": "A"},
                             {"id": "b", "file": "b.safetensors", "name": "B"}],
                            os.path.join(self.dir, "lib"))
        self.loras = os.path.join(self.dir, "loras")
        os.makedirs(self.loras)
        with open(os.path.join(self.loras, "a.safetensors"), "wb") as f:
            f.write(b"x")
        self.backends = [{"id": "5090", "lora_dir": self.loras},
                         {"id": "3090", "lora_dir": ""}]

    def test_the_file_goes_to_the_bin_and_the_record_out(self):
        sent = []
        paths = catalog.uninstall(self.lib, self.lib.get("loras", "a"), self.backends,
                                  send=sent.append)
        self.assertEqual(paths, [os.path.join(self.loras, "a.safetensors")])
        self.assertEqual(sent, paths)
        self.assertEqual([r["id"] for r in ig.Library(self.lib.root).all("loras")], ["b"])

    def test_no_file_on_this_pc_changes_nothing(self):
        with self.assertRaises(ValueError):
            catalog.uninstall(self.lib, self.lib.get("loras", "b"), self.backends,
                              send=self.fail)
        self.assertEqual(len(self.lib.all("loras")), 2)

    def test_users_are_named_before_uninstalling(self):
        self.lib.save("styles", [{"id": "s", "name": "Grainy", "lora": "a"}])
        self.lib.save("presets", [{"id": "p", "name": "Mix", "loras": [{"id": "a"}]}])
        self.assertEqual(catalog.users_of(self.lib, self.lib.get("loras", "a")),
                         ["style Grainy", "preset Mix"])


if __name__ == "__main__":
    unittest.main()
