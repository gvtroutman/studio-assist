from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import studio_addons as addons


class AddonTests(unittest.TestCase):
    def backend(self, root):
        root = Path(root)
        (root / "main.py").write_text("# backend")
        (root / "folder_paths.py").write_text("# paths")
        return root

    def test_install_all_supported_nodes_and_repeat_without_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.backend(folder)
            for addon in addons.CATALOG:
                result = addons.install(addon, root)
                target = Path(result["path"])
                self.assertEqual(target.read_bytes(),
                                 (addons.ROOT / "comfy_nodes" / addon / "__init__.py").read_bytes())
                self.assertTrue(addons.install(addon, root)["unchanged"])

    def test_update_preserves_previous_file_and_other_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.backend(folder)
            target = root / "custom_nodes/studio_matchtone"
            target.mkdir(parents=True)
            (target / "__init__.py").write_bytes(b"old version")
            (target / "notes.txt").write_text("keep me")
            result = addons.install("studio_matchtone", root)
            self.assertEqual(Path(result["backup"]).read_bytes(), b"old version")
            self.assertEqual((target / "notes.txt").read_text(), "keep me")

    def test_invalid_addon_and_wrong_folder_write_nothing(self):
        with tempfile.TemporaryDirectory() as folder:
            for addon in ("../../escape", "studio_matchtone"):
                with self.assertRaises(ValueError):
                    addons.install(addon, folder)
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_failed_replace_keeps_original(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.backend(folder)
            target = root / "custom_nodes/studio_matchtone"
            target.mkdir(parents=True)
            (target / "__init__.py").write_bytes(b"old version")
            with patch.object(addons.os, "replace", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    addons.install("studio_matchtone", root)
            self.assertEqual((target / "__init__.py").read_bytes(), b"old version")
            self.assertEqual(len(list(target.iterdir())), 2)  # original and retained backup
