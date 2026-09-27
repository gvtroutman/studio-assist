import json
import os
import tempfile
import unittest
from unittest.mock import patch

import apps.image_studio.addons.civitai as studio_civitai
import apps.image_studio.model_sources as sources


class ModelSourcesTests(unittest.TestCase):
    def test_saved_keys_links_and_civitai_importer_share_settings(self):
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {}, clear=True):
            for source, (_, domain, _) in sources.SOURCES.items():
                link = "https://%s/models/example" % domain
                sources.save(root, source, [link, "", link], " example-key ")
                self.assertEqual(sources.load(root, source),
                                 {"links": [link], "token": "example-key"})
            self.assertEqual(studio_civitai.load_token(root), "example-key")

    def test_invalid_host_does_not_replace_saved_settings(self):
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {}, clear=True):
            sources.save(root, "huggingface", [], "original")
            for link in ["https://huggingface.co.evil.test/model", "http://huggingface.co/a",
                         "https://key@huggingface.co/a", "not a link"]:
                with self.assertRaises(ValueError):
                    sources.save(root, "huggingface", [link], "replacement")
            self.assertEqual(sources.load(root, "huggingface")["token"], "original")

    def test_environment_key_is_not_persisted_or_overwrites_saved_key(self):
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {}, clear=True):
            sources.save(root, "huggingface", [], "saved")
            with patch.dict(os.environ, {"HF_TOKEN": "environment"}):
                sources.save(root, "huggingface", ["https://huggingface.co/org/model"], "environment")
                self.assertEqual(sources.load(root, "huggingface")["token"], "environment")
                with open(sources.path(root, "huggingface"), encoding="utf-8") as stream:
                    self.assertEqual(json.load(stream)["token"], "saved")
            self.assertEqual(sources.load(root, "huggingface")["token"], "saved")
