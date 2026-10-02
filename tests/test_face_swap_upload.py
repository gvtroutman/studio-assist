"""Face swap photos: pictures from outside the Studio given a person's face
(`ig.keep_uploads`, `ig.upload_swap`, `ui.FaceSwapWindow`). No FaceFusion,
ComfyUI or network: the swap is patched."""
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import apps.image_studio.facefusion as ff
import apps.image_studio.imagegen as ig
import test_imagegen as base               # its tab tests are not run again from here
from test_imagegen import PNG, TempStudioMixin, settle

JPEG = b"\xff\xd8\xff\xe0 not really a jpeg, but its first bytes say so"


class KeepUploads(TempStudioMixin, unittest.TestCase):
    def file(self, name, data):
        path = os.path.join(self.dir, name)
        Path(path).write_bytes(data)
        return path

    def test_pictures_are_kept_by_hash_as_png(self):
        a = self.file("a.png", PNG)
        twice = self.file("again.png", PNG)
        kept, errors = ig.keep_uploads(self.studio.lib, [a, twice])
        self.assertEqual(errors, [])
        (path,) = kept                         # the same picture twice is one
        self.assertEqual(Path(path).read_bytes(), PNG)
        self.assertEqual(os.path.basename(os.path.dirname(path)), ig.UPLOADS)

    def test_anything_but_png_is_turned_into_one_or_said(self):
        jpg = self.file("phone.jpg", JPEG)
        turned = []

        def convert(pairs):
            turned.extend(pairs)
            for _, dest in pairs:
                Path(dest).write_bytes(PNG)
        kept, errors = ig.keep_uploads(self.studio.lib, [jpg], convert=convert)
        self.assertEqual((errors, len(turned)), ([], 1))
        self.assertEqual(kept, [turned[0][1]])
        self.assertTrue(kept[0].endswith(".jpg.png"))
        # Turned already: not again.
        ig.keep_uploads(self.studio.lib, [jpg], convert=convert)
        self.assertEqual(len(turned), 1)
        # One that cannot be turned is an error, not a picture without a preview.
        other = self.file("other.jpg", JPEG + b"!")
        kept, errors = ig.keep_uploads(self.studio.lib, [other], convert=lambda pairs: [])
        self.assertEqual(kept, [])
        self.assertIn("other.jpg: could not be turned into a PNG", errors[0])

    def test_a_folder_gives_its_pictures_and_a_non_picture_is_named(self):
        folder = os.path.join(self.dir, "trip")
        os.makedirs(os.path.join(folder, "deeper"))
        Path(folder, "one.png").write_bytes(PNG)
        Path(folder, "notes.txt").write_text("not a picture")
        Path(folder, "deeper", "two.png").write_bytes(PNG + b"2")
        fake = self.file("fake.png", b"text pretending")
        kept, errors = ig.keep_uploads(self.studio.lib, [folder, fake])
        self.assertEqual(len(kept), 1)                       # not the sub-folder's
        self.assertEqual(errors, ["fake.png: not a picture (PNG, JPEG, WebP, GIF or BMP)"])

    def test_a_dead_link_is_said(self):
        def opener(*a, **k):
            raise ig.LinkError("That link gave no picture.")
        with patch.object(self.studio.lib, "keep_link", side_effect=opener):
            kept, errors = ig.keep_uploads(self.studio.lib, url="https://example.com/x")
        self.assertEqual((kept, errors), ([], ["That link gave no picture."]))


class UploadSwap(TempStudioMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.ref = os.path.join(self.dir, "reference.png")
        Path(self.ref).write_bytes(PNG)
        self.studio.lib.save("identities", [
            {"id": "sitter", "name": "Sitter", "references": [self.ref]},
            {"id": "partner", "name": "Partner", "references": [self.ref]}])
        self.photo = os.path.join(self.dir, "photo.png")
        Path(self.photo).write_bytes(PNG)

    def test_one_face_needs_no_click_and_marks_point_at_faces(self):
        lib = self.studio.lib
        s = ig.upload_swap(lib, self.photo, [], "sitter")
        self.assertEqual(s["mode"], "faces")
        self.assertTrue(ig.local_faces(s))
        (p,) = s["face_finish"]["profiles"]
        self.assertEqual(p["name"], "Sitter")
        self.assertNotIn("target_point", p)
        s = ig.upload_swap(lib, self.photo, [{"identity": "partner", "point": [0.7, 0.3]},
                                             {"identity": "sitter", "point": [0.2, 0.4]}],
                           "sitter")
        self.assertEqual([(p["name"], p["target_point"]) for p in s["face_finish"]["profiles"]],
                         [("Partner", [0.7, 0.3]), ("Sitter", [0.2, 0.4])])
        self.assertNotIn("target_point", lib.get("identities", "partner"))   # a copy
        with self.assertRaises(ValueError):
            ig.upload_swap(lib, self.photo, [], "")
        with self.assertRaises(ValueError):
            ig.upload_swap(lib, self.photo, [{"identity": "gone", "point": [0.5, 0.5]}])

    def test_the_swap_runs_on_this_pc_and_history_keeps_both(self):
        s = ig.upload_swap(self.studio.lib, self.photo,
                           [{"identity": "partner", "point": [0.5, 0.5]}])
        with patch.object(ff, "available", return_value=True), \
                patch.object(ff, "swap", return_value=(PNG, {})) as swap, \
                patch.object(self.studio, "client", side_effect=AssertionError("no ComfyUI")):
            jobs = self.studio.submit(s)
            settle(jobs)
        self.assertEqual(jobs[0].status, "complete", jobs[0].detail)
        self.assertEqual(jobs[0].backend["id"], ig.LOCAL_FACES["id"])
        self.assertEqual(swap.call_args.args[1]["target_point"], [0.5, 0.5])
        # History lists the swapped picture; the upload's checkpoint, finished, is not.
        (listed,) = self.studio.history.list()
        self.assertNotIn("finish", listed)
        self.assertIn("Partner: FaceFusion applied", " ".join(listed["notes"]))

    def test_a_failed_swap_keeps_the_upload_for_a_retry(self):
        s = ig.upload_swap(self.studio.lib, self.photo, [], "sitter")
        with patch.object(ff, "available", return_value=True), \
                patch.object(ff, "swap", side_effect=RuntimeError("The target face is "
                                                                  "ambiguous or missing.")):
            jobs = self.studio.submit(s)
            settle(jobs)
        self.assertEqual(jobs[0].status, "failed")
        self.assertIn("Uploaded picture kept in History", jobs[0].detail)
        (kept,) = self.studio.history.list()
        self.assertIn("Uploaded picture saved before the final face swap.", kept["notes"])
        retry = ig.retry_faces(kept)
        self.assertEqual(retry["mode"], "faces")
        self.assertNotIn("backend", retry["face_finish"])     # a swap alone, as it was
        with patch.object(ff, "available", return_value=True), \
                patch.object(ff, "swap", return_value=(PNG, {})):
            jobs = self.studio.submit(retry)
            settle(jobs)
        self.assertEqual(jobs[0].status, "complete", jobs[0].detail)
        self.assertEqual(json.loads(Path(kept["path"]).read_text(encoding="utf-8"))
                         ["finish"]["state"], "complete")


class FaceSwapWindowTest(unittest.TestCase):
    """The window on the real tab, as TestImageStudioTab builds it."""
    setUpClass = base.TestImageStudioTab.__dict__["setUpClass"]
    tearDownClass = base.TestImageStudioTab.__dict__["tearDownClass"]
    pump = base.TestImageStudioTab.pump
    tab = base.TestImageStudioTab.tab

    def test_pictures_marks_and_one_job_each(self):
        import apps.image_studio.ui as ui_mod
        _, ui = self.tab()
        lib = ui.studio.lib
        ref = os.path.join(self.dir, "ref.png")
        Path(ref).write_bytes(PNG)
        old = lib.all("identities")
        lib.save("identities", [{"id": "sitter", "name": "Sitter", "references": [ref]},
                                {"id": "partner", "name": "Partner", "references": [ref]}])
        self.addCleanup(lambda: lib.save("identities", old))
        a, b = (os.path.join(self.dir, n) for n in ("a.png", "b.png"))
        Path(a).write_bytes(PNG)
        Path(b).write_bytes(PNG + b"b")
        spawned = []
        ui.host._spawn = lambda sid, fn, *args: spawned.append((fn, args))
        self.addCleanup(lambda: delattr(ui.host, "_spawn"))
        win = ui.face_swap()
        self.assertIs(ui.face_swap(), win)                 # one window, raised
        self.addCleanup(lambda: win.win.destroy())
        self.assertEqual(win.person, "sitter")
        win.start()
        self.assertIn("Add a picture first", win.msg.cget("text"))
        win.take([a, b])
        (fn, _), = spawned
        spawned.clear()
        fn()                                               # the import, here
        self.pump(lambda: len(win.pictures) == 2 and win.img is not None)
        self.assertEqual(win.at, 0)
        for pic in win.pictures:
            self.assertEqual(os.path.basename(os.path.dirname(pic["path"])), ig.UPLOADS)

        class Ev:
            def __init__(self, x, y):
                self.x, self.y = x, y
        middle = Ev(win.ox + win.img.width() // 2, win.oy + win.img.height() // 2)
        win._pick_person("partner")
        win._mark(middle)
        self.assertEqual(win.pictures[0]["marks"],
                         [{"identity": "partner", "point": win._to_pic(middle)}])
        win._pick_person("sitter")
        win._mark(middle)                                  # the same face: now Sitter's
        self.assertEqual([m["identity"] for m in win.pictures[0]["marks"]], ["sitter"])
        win._unmark(middle)
        self.assertEqual(win.pictures[0]["marks"], [])
        win._mark(middle)
        win.show(1)                                        # no marks: Person's face
        with patch.object(ui_mod.ff, "available", return_value=True):
            win.start()
        (fn, _), = spawned
        sent = []
        with patch.object(ui, "_submit", side_effect=sent.append):
            fn()
        self.assertEqual([s["mode"] for s in sent], ["faces", "faces"])
        first, second = (s["face_finish"]["profiles"] for s in sent)
        self.assertEqual([p["id"] for p in first], ["sitter"])
        self.assertIn("target_point", first[0])
        self.assertEqual([p["id"] for p in second], ["sitter"])
        self.assertNotIn("target_point", second[0])
        win.remove()
        self.assertEqual((len(win.pictures), win.at), (1, 0))


if __name__ == "__main__":
    unittest.main()
