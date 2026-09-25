"""LoRA profiles from CivitAI: reading links, the API's answers turned into a
library record, a safetensors file's own header, the hash lookup, and the
download that puts the file in a backend's folder. CivitAI is a table of
canned answers here; nothing touches the network."""

import base64
import hashlib
import io
import json
import os
import struct
import sys
import tempfile
import unittest
import urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import studio_civitai as civitai  # noqa: E402
import studio_imagegen as ig  # noqa: E402

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
LORA_BYTES = b"not really weights, but bytes with a hash"
LORA_SHA = hashlib.sha256(LORA_BYTES).hexdigest()

VERSION = {
    "id": 67890, "modelId": 12345, "name": "v2.0", "baseModel": "Flux.1 D",
    "trainedWords": ["sx70 photo", "polaroid frame", "sx70 photo"],
    "description": "<p>Second pass.</p>",
    "downloadUrl": "https://civitai.com/api/download/models/67890",
    "files": [
        {"name": "training_data.zip", "type": "Training Data", "sizeKB": 10},
        {"name": "sx70_flux_v2.safetensors", "type": "Model", "primary": True,
         "sizeKB": len(LORA_BYTES) / 1024.0, "hashes": {"SHA256": LORA_SHA.upper()},
         "downloadUrl": "https://civitai.com/api/download/models/67890"},
    ],
    "images": [
        {"url": "https://image.civitai.com/x/spicy.jpeg", "nsfwLevel": 8, "type": "image"},
        {"url": "https://image.civitai.com/x/clip.mp4", "nsfwLevel": 1, "type": "video"},
        {"url": "https://image.civitai.com/x/tame.png", "nsfwLevel": 1, "type": "image"},
    ],
}
MODEL = {
    "id": 12345, "name": "SX-70 Instant Film", "type": "LORA",
    "tags": ["photography", "style", "polaroid"],
    "description": "<p>An <b>instant film</b> look.</p><p>Use at 0.7&ndash;0.9.</p>",
    "creator": {"username": "filmnerd"},
    "modelVersions": [{"id": 67890}, {"id": 11111}],
}


class Response(io.BytesIO):
    def __init__(self, data, headers=None):
        super().__init__(data)
        self.headers = headers or {}


class FakeCivitAI:
    """The routes the importer uses. `seen` records each request."""

    def __init__(self):
        self.seen, self.routes = [], {
            "https://civitai.com/api/v1/model-versions/67890": VERSION,
            "https://civitai.com/api/v1/models/12345": MODEL,
            "https://civitai.com/api/v1/model-versions/by-hash/" + LORA_SHA.upper(): VERSION,
            "https://image.civitai.com/x/tame.png": PNG,
            "https://civitai.com/api/download/models/67890": LORA_BYTES,
        }
        self.need_key = set()

    def __call__(self, req, timeout=None):
        url = req.full_url
        self.seen.append((url, req.get_header("Authorization")))
        if url in self.need_key and not req.get_header("Authorization"):
            raise urllib.error.HTTPError(url, 401, "Unauthorized", {}, None)
        if url not in self.routes:
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
        body = self.routes[url]
        if isinstance(body, (dict, list)):
            return Response(json.dumps(body).encode(), {"Content-Type": "application/json"})
        return Response(body, {"Content-Type": "application/octet-stream",
                               "Content-Length": str(len(body))})


def safetensors(path, meta, payload=b""):
    head = json.dumps({"__metadata__": meta} if meta is not None else {}).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(head)) + head + payload)
    return path


class TestLinks(unittest.TestCase):
    def test_every_way_of_naming_a_version(self):
        for text, want in [
            ("https://civitai.com/models/12345/sx-70?modelVersionId=67890", (12345, 67890)),
            ("civitai.com/models/12345", (12345, None)),
            ("https://civitai.com/api/download/models/67890?type=Model", (None, 67890)),
            ("https://civitai.green/models/12345?modelVersionId=67890", (12345, 67890)),
            ("urn:air:flux1:lora:civitai:12345@67890", (12345, 67890)),
            ("  67890 ", (None, 67890)),
        ]:
            with self.subTest(text=text):
                ref = civitai.parse_link(text)
                self.assertEqual((ref["model"], ref["version"]), want)

    def test_what_is_not_civitai_is_none(self):
        for text in ("", "https://huggingface.co/models/123", "hello",
                     "https://civitai.com/user/someone", "https://notcivitai.com/models/1"):
            with self.subTest(text=text):
                self.assertIsNone(civitai.parse_link(text))


class TestProfile(unittest.TestCase):
    def test_a_version_becomes_a_library_record(self):
        p = civitai.profile(VERSION, MODEL)
        self.assertEqual(p["file"], "sx70_flux_v2.safetensors")    # the primary, not the zip
        self.assertEqual(p["name"], "SX-70 Instant Film (v2.0)")
        self.assertEqual(p["trigger"], "sx70 photo, polaroid frame")
        self.assertEqual(p["family"], "flux1")
        self.assertEqual(p["category"], "Camera / Film")
        self.assertEqual(p["sha256"], LORA_SHA)
        self.assertEqual(p["source"], "https://civitai.com/models/12345?modelVersionId=67890")
        self.assertIn("Base model: Flux.1 D", p["notes"])
        self.assertIn("By filmnerd", p["notes"])
        self.assertIn("Second pass.", p["notes"])
        self.assertNotIn("<p>", p["notes"])
        self.assertEqual(p["_preview_url"], "https://image.civitai.com/x/tame.png")

    def test_base_models_map_to_families(self):
        for base, fam in [("Flux.1 D", "flux1"), ("Flux.1 Kontext", "flux1-kontext"),
                          ("Flux.2 D", "flux2"), ("SDXL 1.0", "sdxl"), ("Pony", "sdxl"),
                          ("Illustrious", "sdxl"), ("SD 1.5", "sd15"),
                          ("ZImageTurbo", "z-image"), ("Z-Image Turbo", "z-image"),
                          ("Qwen", "qwen-image"), ("Wan Video", "")]:
            with self.subTest(base=base):
                self.assertEqual(civitai.family_of(base), fam)


class TestFiles(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def test_the_header_is_read_and_a_non_safetensors_refused(self):
        path = safetensors(os.path.join(self.dir, "a.safetensors"), {
            "ss_output_name": "gavin_v3", "ss_base_model_version": "flux1",
            "modelspec.trigger_phrase": "GAVINPERSON"})
        p = civitai.profile_from_header(civitai.read_header(path), "a.safetensors")
        self.assertEqual((p["name"], p["family"], p["trigger"]),
                         ("gavin_v3", "flux1", "GAVINPERSON"))
        self.assertEqual(civitai.read_header(safetensors(
            os.path.join(self.dir, "b.safetensors"), None)), {})
        junk = os.path.join(self.dir, "c.safetensors")
        with open(junk, "wb") as f:
            f.write(b"\xff" * 64)
        with self.assertRaises(civitai.CivitAIError):
            civitai.read_header(junk)

    def test_copy_into_names_it_as_comfyui_will(self):
        loras = os.path.join(self.dir, "loras")
        inside = os.path.join(loras, "people")
        os.makedirs(inside)
        a = safetensors(os.path.join(inside, "me.safetensors"), {})
        self.assertEqual(civitai.copy_into(a, loras), os.path.join("people", "me.safetensors"))
        b = safetensors(os.path.join(self.dir, "else.safetensors"), {})
        self.assertEqual(civitai.copy_into(b, loras), "else.safetensors")
        self.assertTrue(os.path.isfile(os.path.join(loras, "else.safetensors")))
        self.assertEqual(civitai.copy_into(b, loras), "else.safetensors")   # same file: fine
        os.makedirs(os.path.join(self.dir, "x"))
        c = safetensors(os.path.join(self.dir, "x", "else.safetensors"), {"k": "v"})
        with self.assertRaises(civitai.CivitAIError):
            civitai.copy_into(c, loras)                                   # different file


class TestImport(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.lib = ig.Library(os.path.join(self.dir, "studio"))
        self.fake = FakeCivitAI()
        self.client = civitai.Client(opener=self.fake)
        self.said = []

    def test_a_link_adds_a_record_with_its_preview_and_is_saved(self):
        rec, added = civitai.import_link(self.lib, self.client,
                                         "https://civitai.com/models/12345", say=self.said.append)
        self.assertTrue(added)
        self.assertEqual(rec["file"], "sx70_flux_v2.safetensors")
        self.assertEqual(rec["trigger"], "sx70 photo, polaroid frame")
        self.assertTrue(rec["preview"].startswith(self.lib.preview_dir()))
        with open(rec["preview"], "rb") as f:
            self.assertEqual(f.read(), PNG)
        again = ig.Library(self.lib.root).get("loras", rec["id"])
        self.assertEqual(again["sha256"], LORA_SHA)
        self.assertEqual(again["source"], rec["source"])
        # No file was asked for, so nothing was downloaded.
        self.assertNotIn("https://civitai.com/api/download/models/67890",
                         [u for u, _ in self.fake.seen])

    def test_a_second_import_keeps_what_the_user_wrote(self):
        rec, _ = civitai.import_link(self.lib, self.client, "67890")
        rec["name"], rec["trigger"], rec["notes"] = "Mine", "my words", ""
        again, added = civitai.import_link(self.lib, self.client, "67890")
        self.assertFalse(added)
        self.assertIs(again, rec)
        self.assertEqual((rec["name"], rec["trigger"]), ("Mine", "my words"))
        self.assertIn("From CivitAI", rec["notes"])                # empty: filled
        self.assertEqual(len(self.lib.all("loras")), 1)

    def test_a_scanned_record_is_filled_in_not_duplicated(self):
        self.lib.merge_loras("5090", ["sx70_flux_v2.safetensors"])
        rec, added = civitai.import_link(self.lib, self.client, "67890")
        self.assertFalse(added)
        self.assertEqual(rec["name"], "SX-70 Instant Film (v2.0)")   # the guess was replaced
        self.assertEqual(rec["trigger"], "sx70 photo, polaroid frame")

    def test_download_lands_whole_checked_and_with_the_key(self):
        folder = os.path.join(self.dir, "loras")
        self.fake.need_key.add("https://civitai.com/api/download/models/67890")
        with self.assertRaises(civitai.CivitAIError) as cm:
            civitai.import_link(self.lib, self.client, "67890", folder)
        self.assertIn("API key", str(cm.exception))
        self.assertEqual(os.listdir(folder) if os.path.isdir(folder) else [], [])
        keyed = civitai.Client("secret", opener=self.fake)
        civitai.import_link(self.lib, keyed, "67890", folder)
        with open(os.path.join(folder, "sx70_flux_v2.safetensors"), "rb") as f:
            self.assertEqual(f.read(), LORA_BYTES)
        self.assertIn(("https://civitai.com/api/download/models/67890", "Bearer secret"),
                      self.fake.seen)

    def test_a_damaged_download_leaves_nothing(self):
        folder = os.path.join(self.dir, "loras")
        self.fake.routes["https://civitai.com/api/download/models/67890"] = b"truncated"
        with self.assertRaises(civitai.CivitAIError):
            civitai.import_link(self.lib, self.client, "67890", folder)
        self.assertEqual(os.listdir(folder), [])
        self.assertEqual(self.lib.all("loras"), [])

    def test_a_file_is_found_by_its_hash(self):
        path = safetensors(os.path.join(self.dir, "renamed.safetensors"),
                           {"ss_output_name": "whatever"})
        # The hash CivitAI knows is the file's own; make the table say so.
        sha = civitai.sha256_of(path)
        self.fake.routes["https://civitai.com/api/v1/model-versions/by-hash/" + sha.upper()] \
            = VERSION
        rec, added = civitai.import_file(self.lib, self.client, path)
        self.assertTrue(added)
        self.assertEqual(rec["file"], "renamed.safetensors")       # its name, not CivitAI's
        self.assertEqual(rec["name"], "SX-70 Instant Film (v2.0)")
        self.assertEqual(rec["sha256"], sha)
        self.assertTrue(rec["preview"])

    def test_an_unknown_or_offline_file_uses_its_header(self):
        path = safetensors(os.path.join(self.dir, "mine.safetensors"),
                           {"modelspec.title": "My Face", "modelspec.trigger_phrase": "ZQX",
                            "modelspec.architecture": "flux-1-dev/lora"})
        rec, _ = civitai.import_file(self.lib, self.client, path, say=self.said.append)
        self.assertEqual((rec["name"], rec["trigger"], rec["family"]),
                         ("My Face", "ZQX", "flux1"))
        self.assertTrue(any("does not know" in s for s in self.said))

        def offline(req, timeout=None):
            raise urllib.error.URLError("no route to host")
        other = safetensors(os.path.join(self.dir, "other.safetensors"), {"modelspec.title": "B"})
        rec, _ = civitai.import_file(self.lib, civitai.Client(opener=offline), other,
                                     say=self.said.append)
        self.assertEqual(rec["name"], "B")
        self.assertTrue(any("Not looked up" in s for s in self.said))

    def test_a_file_already_in_a_lora_folder_keeps_its_subfolder(self):
        loras = os.path.join(self.dir, "loras")
        os.makedirs(os.path.join(loras, "people"))
        path = safetensors(os.path.join(loras, "people", "me.safetensors"), {})
        rec, _ = civitai.import_file(self.lib, self.client, path, lora_dirs=[loras],
                                     lookup=False)
        self.assertEqual(rec["file"], os.path.join("people", "me.safetensors"))
        self.assertEqual([u for u, _ in self.fake.seen], [])       # lookup off: no calls

    def test_the_key_is_kept_and_the_environment_wins(self):
        root = self.lib.root
        self.assertEqual(civitai.load_token(root), "")
        civitai.save_token(root, " abc ")
        self.assertEqual(civitai.load_token(root), "abc")
        os.environ[civitai.TOKEN_ENV] = "from-env"
        try:
            self.assertEqual(civitai.load_token(root), "from-env")
        finally:
            del os.environ[civitai.TOKEN_ENV]


if __name__ == "__main__":
    unittest.main()
