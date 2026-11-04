import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from core import local_config


class LocalConfigTests(unittest.TestCase):
    def test_missing_invalid_or_non_string_configuration_uses_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "studio.local.json"
            with patch.object(local_config, "CONFIG_PATH", path):
                self.assertEqual(local_config.default("COMFYUI_URL", "fallback"), "fallback")
                for content in ("{", "[]", '{"COMFYUI_URL": 123}', '{"COMFYUI_URL": " "}'):
                    path.write_text(content, encoding="utf-8")
                    self.assertEqual(local_config.default("COMFYUI_URL", "fallback"), "fallback")

    def test_endpoint_is_read_without_changing_environment(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "studio.local.json"
            path.write_text(json.dumps({"COMFYUI_URL": " http://example.test:8188 "}), encoding="utf-8")
            with patch.object(local_config, "CONFIG_PATH", path):
                self.assertEqual(local_config.default("COMFYUI_URL", "fallback"), "http://example.test:8188")
                self.assertEqual(local_config.default("STUDIO_HOST", "fallback"), "fallback")
