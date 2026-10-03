"""Add-ons (studio_catalog): which LoRAs fit which model, the CivitAI catalog
filtered to a model's base models, safe thumbnails, and uninstalling to the
Recycle Bin. CivitAI is a table of canned answers; the one live piece is the
PowerShell picture conversion, run on a BMP this test writes."""

import base64
import json
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
            {"id": "q", "file": "q.safetensors", "family": "sd15"},
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

    def test_a_card_carries_civitais_license(self):
        lic = civitai.license_of
        self.assertEqual(lic({"allowCommercialUse": ["Image", "Rent"], "allowNoCredit": True}),
                         ("commercial", "Sell pictures OK"))
        self.assertEqual(lic({"allowCommercialUse": "Image", "allowNoCredit": False}),
                         ("commercial", "Sell pictures OK, credit the creator"))
        self.assertEqual(lic({"allowCommercialUse": ["RentCivit"]})[0], "custom")
        self.assertEqual(lic({"allowCommercialUse": ["None"]}),
                         ("noncommercial", "No commercial use"))
        self.assertEqual(lic({"allowCommercialUse": []})[0], "noncommercial")
        self.assertEqual(lic({}), ("unstated", ""))
        c = catalog.card(dict(HIT, allowCommercialUse=["Image"]), {"z-image"})
        self.assertEqual(c["license_group"], "commercial")
        self.assertEqual(catalog.card(HIT, {"z-image"})["license_group"], "unstated")

    def test_cards_are_listed_by_license_keeping_order_inside_a_group(self):
        cards = [{"n": 1, "license_group": "noncommercial"}, {"n": 2},
                 {"n": 3, "license_group": "commercial"}, {"n": 4, "license_group": "custom"},
                 {"n": 5, "license_group": "commercial"}]
        self.assertEqual([c["n"] for c in catalog.by_license(cards)], [3, 5, 4, 1, 2])

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


class TestLicenses(unittest.TestCase):
    def test_a_base_model_takes_its_familys_license_unless_set(self):
        self.assertEqual(ig.model_license({"family": "z-image"}), ("commercial", "Apache 2.0"))
        self.assertEqual(ig.model_license({"family": "flux1"})[0], "custom")
        self.assertEqual(ig.model_license({"family": "krea2"}), ("unstated", ""))
        klein = {"family": "flux2-klein9b", "license": ig.KLEIN_LICENSE}
        self.assertEqual(ig.model_license(klein), ("noncommercial", ig.KLEIN_LICENSE))
        self.assertEqual(ig.model_license({"family": "flux1", "license_group": "commercial"}),
                         ("commercial", ""))

    def test_models_are_listed_by_license_keeping_their_order(self):
        lib = ig.Library(tempfile.mkdtemp())
        ordered = [m["id"] for m in ig.by_license(lib.all("models"), ig.model_license)]
        self.assertEqual(ordered[:2], ["z-image-turbo", "qwen-image"])
        self.assertEqual(ordered[2:4], ["flux-dev", "withanyone"])
        self.assertEqual(ordered[4:], ["klein-9b", "klein-9b-distilled"])

    def test_records_keep_a_license_and_drop_a_made_up_group(self):
        rec = ig.clean_lora({"file": "a.safetensors", "license_group": "commercial",
                             "license": "Sell pictures OK"})
        self.assertEqual((rec["license_group"], rec["license"]),
                         ("commercial", "Sell pictures OK"))
        self.assertEqual(ig.clean_lora({"file": "a.safetensors",
                                        "license_group": "free"})["license_group"], "")
        self.assertEqual(ig.lora_license({"license_group": ""}), ("unstated", ""))
        self.assertEqual(ig.clean_model({"id": "m", "license_group": "noncommercial"})[
            "license_group"], "noncommercial")

    def test_installed_loras_are_listed_by_license(self):
        lib = lib_with([{"file": "a.safetensors", "name": "A"},
                        {"file": "b.safetensors", "name": "B", "license_group": "noncommercial"},
                        {"file": "c.safetensors", "name": "C", "license_group": "commercial"}])
        self.assertEqual([r["name"] for r in catalog.by_license(lib.all("loras"))],
                         ["C", "B", "A"])

    def test_an_import_takes_the_license_only_from_a_whole_model(self):
        whole = dict(HIT, allowCommercialUse=["Image"], allowNoCredit=False)
        p = civitai.profile(version(1, "ZImageBase"), whole)
        self.assertEqual((p["license_group"], p["license"]),
                         ("commercial", "Sell pictures OK, credit the creator"))
        partial = civitai.profile(dict(version(1, "ZImageBase"), model={"name": "Hands"}))
        self.assertNotIn("license_group", partial)


class TestCheckpoints(unittest.TestCase):
    def test_only_commercial_architectures_are_searched(self):
        self.assertEqual(sorted(catalog.checkpoint_families()),
                         ["qwen-image", "sd15", "sdxl", "z-image"])

    def test_search_keeps_checkpoints_you_may_sell_pictures_from(self):
        sell = dict(HIT, id=1, allowCommercialUse=["Image"])
        nc = dict(HIT, id=2, allowCommercialUse=["None"])
        client = FakeClient(hits=[sell, nc])
        calls = []
        client.search = lambda bases, q, s, c, types="LORA": (
            calls.append((tuple(bases), types)) or ([sell, nc], ""))
        cards, nxt, dropped = catalog.checkpoint_search(client, "sdxl")
        self.assertEqual(([c["model_id"] for c in cards], dropped), ([1], 1))
        self.assertEqual(calls, [(("SDXL 1.0",), "Checkpoint")])

    def test_install_adds_a_model_copied_from_the_same_family(self):
        lib = ig.Library(tempfile.mkdtemp())
        c = catalog.card(dict(HIT, allowCommercialUse=["Image"]), {"z-image"})
        c["name"] = "Better Z"
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(civitai, "download") as dl:
            rec = catalog.checkpoint_install(lib, None, c, d)
        self.assertEqual(dl.call_args[0][3], c["file"])
        self.assertEqual((rec["label"], rec["family"], rec["values"]["model"]),
                         ("Better Z", "z-image", c["file"]))
        self.assertEqual(rec["workflow"], lib.get("models", "z-image-turbo")["workflow"])
        self.assertEqual(ig.model_license(rec)[0], "commercial")
        with self.assertRaises(civitai.CivitAIError):
            catalog.checkpoint_install(lib, None, dict(c, family="sd15"), "x")

    def test_a_qwen_checkpoint_needs_no_qwen_model_in_the_library(self):
        lib = ig.Library(tempfile.mkdtemp())
        lib.save("models", [m for m in lib.all("models") if m["family"] != "qwen-image"])
        c = catalog.card(dict(HIT, allowCommercialUse=["Image"]), {"z-image"})
        c.update(family="qwen-image", name="Real Qwen")
        with tempfile.TemporaryDirectory() as d, mock.patch.object(civitai, "download"):
            rec = catalog.checkpoint_install(lib, None, c, d)
        self.assertEqual((rec["workflow"], rec["values"]["encoder"]),
                         ("qwen_image", "qwen_2.5_vl_7b_fp8_scaled.safetensors"))

    def test_qwen_joins_an_existing_library_once(self):
        root = tempfile.mkdtemp()
        lib = ig.Library(root)
        lib.save("models", [m for m in lib.all("models") if m["id"] != "qwen-image"])
        lib = ig.Library(root)
        self.assertIsNotNone(lib.get("models", "qwen-image"))
        lib.save("models", [m for m in lib.all("models") if m["id"] != "qwen-image"])
        self.assertIsNone(ig.Library(root).get("models", "qwen-image"))      # stays removed

    def test_the_qwen_workflow_builds(self):
        wf = ig.load_workflow("qwen_image")
        self.assertEqual(wf["families"], ["qwen-image"])
        self.assertEqual(wf["graph"]["2"]["inputs"]["type"], "qwen_image")

    QWEN_FILES = {"diffusion_models": {"qwen_image_fp8_e4m3fn.safetensors"},
                  "text_encoders": {"qwen_2.5_vl_7b_fp8_scaled.safetensors"},
                  "vae": {"qwen_image_vae.safetensors"},
                  "controlnet": {"Qwen-Image-InstantX-ControlNet-Union.safetensors"},
                  "checkpoints": {"sam3.pt"}}

    def qwen_plan(self, inventory):
        studio = ig.Studio(root=tempfile.mkdtemp())
        pose = os.path.join(studio.lib.root, "pose.png")
        with open(pose, "wb") as f:
            f.write(PNG)
        s = ig.default_settings()
        s.update(model="qwen-image", scene="a woman waving on a beach",
                 references={"pose": pose}, face_detail=True)
        return ig.compose(s, studio.lib, studio.backend("5090"), inventory)

    def test_qwen_takes_a_pose_through_its_controlnet_and_has_a_face_pass(self):
        p = self.qwen_plan(self.QWEN_FILES)
        self.assertEqual((p.errors, p.warnings), ([], []))
        self.assertIn("pose_image", p.images)
        self.assertTrue(p.values["face_detail"])
        g = ig.fill(p.workflow, dict(p.values, pose_image="pose.png"), p.loras)
        self.assertEqual(g["40"]["inputs"]["positive"], ["52", 0])
        self.assertEqual(g["52"]["class_type"], "ControlNetApplyAdvanced")

    def test_qwen_chains_composition_after_pose_and_refines(self):
        studio = ig.Studio(root=tempfile.mkdtemp())
        refs = {}
        for kind in ("pose", "composition"):
            refs[kind] = os.path.join(studio.lib.root, kind + ".png")
            with open(refs[kind], "wb") as f:
                f.write(PNG)
        s = ig.default_settings()
        s.update(model="qwen-image", scene="a woman on a beach", references=refs, refine=True)
        p = ig.compose(s, studio.lib, studio.backend("5090"), self.QWEN_FILES)
        self.assertEqual((p.errors, p.warnings), ([], []))
        self.assertEqual(sorted(p.images), ["composition_image", "pose_image"])
        g = ig.fill(p.workflow, dict(p.values, pose_image="a", composition_image="b"),
                    p.loras)
        self.assertEqual(g["54"]["inputs"]["positive"], ["52", 0])     # depth after pose
        self.assertEqual(g["40"]["inputs"]["positive"], ["54", 0])
        self.assertEqual(g["9"]["inputs"]["images"], ["45", 0])        # the refined one
        # The refine redraws the enlarged picture unposed, on the plain model.
        self.assertEqual((g["44"]["inputs"]["model"], g["44"]["inputs"]["positive"]),
                         (["4", 0], ["10", 0]))
        self.assertNotIn("{{", json.dumps(g))

    def test_qwen_starts_from_a_source_picture(self):
        studio = ig.Studio(root=tempfile.mkdtemp())
        src = os.path.join(studio.lib.root, "source.png")
        with open(src, "wb") as f:
            f.write(PNG)
        s = ig.default_settings()
        s.update(model="qwen-image", scene="a woman on a beach", references={"source": src})
        p = ig.compose(s, studio.lib, studio.backend("5090"), self.QWEN_FILES)
        self.assertEqual((p.errors, p.warnings), ([], []))
        self.assertEqual(p.values["denoise"], 0.65)
        g = ig.fill(p.workflow, dict(p.values, source_image="s.png"), p.loras)
        self.assertEqual(g["40"]["inputs"]["latent_image"], ["22", 0])
        self.assertEqual(g["40"]["inputs"]["denoise"], 0.65)
        self.assertNotIn("20", g)
        plain = ig.fill(p.workflow, dict(p.values, source_image=None, denoise=1.0), p.loras)
        self.assertEqual(plain["40"]["inputs"]["latent_image"], ["20", 0])

    def test_qwen_says_truly_where_a_persons_face_comes_from(self):
        studio = ig.Studio(root=tempfile.mkdtemp())
        photo = os.path.join(studio.lib.root, "face.png")
        with open(photo, "wb") as f:
            f.write(PNG)
        studio.lib.save("identities", [{"id": "sam", "name": "Sam", "references": [photo]}])
        s = ig.default_settings()
        s.update(model="qwen-image", scene="Sam on a beach", identities=["sam"])
        p = ig.compose(s, studio.lib, studio.backend("5090"), self.QWEN_FILES)
        said = " ".join(p.notes + p.warnings)
        self.assertNotIn("LoRA carries", said)
        self.assertIn("head swap and face swap", said)

    def test_qwen_without_its_controlnet_says_so_and_drops_the_pose(self):
        p = self.qwen_plan(dict(self.QWEN_FILES, controlnet=set()))
        self.assertNotIn("pose_image", p.images)
        self.assertTrue(any("ControlNet-Union" in w for w in p.warnings))

    def test_the_qwen_face_pass_redraws_on_the_unposed_model(self):
        wf = ig.load_workflow("qwen_image")
        v = dict(wf["defaults"], model="m", encoder="e", vae="v", prompt="p", seed=1,
                 face_prompt="a face", sam3="sam3.pt")
        g = ig.face_graph(wf, v, [], "pic.png", [{"x": 0, "y": 0, "width": 256,
                                                  "height": 256}], "oval.png", "F")
        samplers = [n for n in g.values() if n["class_type"] == "KSampler"]
        self.assertEqual([(n["inputs"]["model"], n["inputs"]["cfg"]) for n in samplers],
                         [(["4", 0], 2.5)])
        self.assertFalse(any("ControlNet" in n["class_type"] for n in g.values()))


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
