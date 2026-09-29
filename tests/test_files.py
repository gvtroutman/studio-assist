"""Attachments described for a model (core.files), without a display: the
JPEG header walk in `image_dims` on files that are cut off or padded, and a
folder listing's bound. The Tk-side attachment tests live in test_agent."""

import os
import shutil
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import core.files as files

SOF = b"\xff\xc0\x00\x11\x08\x04\x38\x07\x80\x03\x01\x22\x00\x02\x11\x01\x03\x11\x01"  # 1920 x 1080


class TestImageDims(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)

    def dims(self, data):
        """image_dims on a file holding `data`, run on a thread so a header walk
        that never ends fails the test instead of hanging the suite."""
        path = os.path.join(self.dir, "p.jpg")
        with open(path, "wb") as fh:
            fh.write(data)
        out = []
        t = threading.Thread(target=lambda: out.append(files.image_dims(path)), daemon=True)
        t.start()
        t.join(5)
        self.assertFalse(t.is_alive(), "image_dims never returned on %r" % data)
        return out[0]

    def test_a_jpeg_cut_off_after_a_marker_is_no_picture_and_does_not_hang(self):
        # A half-downloaded file: the loop used to seek back onto the same marker forever.
        self.assertIsNone(self.dims(b"\xff\xd8\xff\xe0"))
        self.assertIsNone(self.dims(b"\xff\xd8\xff\xe0\x00"))
        self.assertIsNone(self.dims(b"\xff\xd8\xff\xe0\x00\x00\x00\x00"))

    def test_a_jpeg_cut_off_inside_its_frame_header_is_no_picture(self):
        self.assertIsNone(self.dims(b"\xff\xd8" + SOF[:6]))

    def test_fill_bytes_before_a_marker_are_skipped(self):
        self.assertEqual(self.dims(b"\xff\xd8\xff\xff\xff" + SOF[1:] + b"\xff\xd9"), (1920, 1080))

    def test_a_whole_jpeg_still_reads(self):
        app0 = b"\xff\xe0\x00\x04\x00\x00"
        self.assertEqual(self.dims(b"\xff\xd8" + app0 + SOF + b"\xff\xd9"), (1920, 1080))


class TestDescribeFolder(unittest.TestCase):
    def test_a_long_folder_is_listed_to_the_limit_with_the_rest_counted(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        os.mkdir(os.path.join(d, "sub"))
        for i in range(files.LIST_LIMIT + 5):
            open(os.path.join(d, "f%03d.txt" % i), "w").close()
        text = files.describe_folder(d)
        lines = text.splitlines()
        self.assertIn("(folder, %d files, 1 folders)" % (files.LIST_LIMIT + 5), lines[0])
        self.assertEqual(lines[1], "    sub/")          # folders first
        self.assertEqual(len(lines), 1 + files.LIST_LIMIT + 1)
        self.assertEqual(lines[-1], "    ... and 6 more")


if __name__ == "__main__":
    unittest.main()
