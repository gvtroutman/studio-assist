"""Reference-set preparation without a model or live image backend."""
import base64
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import apps.image_studio.imagegen as ig
from apps.image_studio.ui import RecordEditor

PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==')


class IdentityImportTests(unittest.TestCase):
    def test_duplicates_are_by_content_and_bad_files_do_not_drop_good_ones(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first, duplicate, bad = [root / p for p in ('first.png', 'copy.JPG', 'bad.png')]
            first.write_bytes(PNG)
            duplicate.write_bytes(PNG)
            bad.write_text('not a picture')
            lib = ig.Library(str(root / 'library'))
            result = lib.import_identity_photos([str(p) for p in
                (first, bad, duplicate, root / 'missing.jpg')], 'Person')
            self.assertEqual(len(result['added']), 1)
            self.assertEqual(result['duplicates'], 1)
            self.assertEqual(len(result['errors']), 2)
            first.unlink()
            self.assertEqual(Path(result['added'][0]).read_bytes(), PNG)
            again = lib.import_identity_photos([str(duplicate)], 'Renamed person', result['added'])
            self.assertEqual(again, {'added': [], 'duplicates': 1, 'errors': []})

    def test_primary_changes_order_without_losing_other_views(self):
        editor = RecordEditor.__new__(RecordEditor)
        editor._draw_paths = mock.Mock()
        editor.status = mock.Mock()
        pics = {'paths': ['side', 'front', 'other'], 'sel': {1}}
        editor._primary_path(pics)
        self.assertEqual(pics['paths'], ['front', 'side', 'other'])
        self.assertEqual(pics['sel'], {0})
        pics['sel'] = {0, 1}
        editor._primary_path(pics)
        editor.status.assert_called_once()
        self.assertEqual(pics['paths'], ['front', 'side', 'other'])

    def test_save_cannot_publish_an_incomplete_import(self):
        editor = RecordEditor.__new__(RecordEditor)
        editor._imports = 1
        editor.status = mock.Mock()
        editor._store = mock.Mock()
        self.assertFalse(editor._save())
        editor._store.assert_not_called()
