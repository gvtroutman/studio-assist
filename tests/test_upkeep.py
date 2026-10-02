"""The Image Studio's housekeeping: what its jobs leave behind is removed, and
nothing else. Every folder here is a temporary one; no ComfyUI, no network."""
import io
import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import apps.image_studio.imagegen as ig
from test_imagegen import TempStudioMixin, PNG, settle

DAY = 86400
JOB = "0123456789ab"                   # a job's id: 12 hex digits


def aged(path, days, data=PNG):
    """A file at `path`, last written `days` ago."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    when = time.time() - days * DAY
    os.utime(path, (when, when))
    return str(path)


class TestWhatIsTheAppsToRemove(unittest.TestCase):
    def test_a_jobs_pictures_and_uploads_are_known_by_their_names(self):
        for name in ("zimage_hq_%s_00001_.png", "flux_dev_baseline_%s_faces_00001_.png",
                     "zimage_hq_%s_pass1_face_correction_00001_.png", "fix_%s_face_00002_.png",
                     "dress_%s_00001_.png", "zimage_hq_%s_regen2_00001_.png"):
            self.assertTrue(ig.MADE.match(name % JOB), name)
        # A test picture under a name of its own, ComfyUI's own, another format.
        for name in ("livetest_1790470158_glasses_00001_.png", "dresstest_av5_77_00001_.png",
                     "ComfyUI_00001_.png", "headtest_broad_00001_.png",
                     "zimage_hq_%s_00001_.json" % JOB, "%s_00001_.png" % JOB):
            self.assertFalse(ig.MADE.match(name), name)
        self.assertTrue(ig.SENT.match("studio_0123456789abcdef.png"))
        self.assertTrue(ig.SENT.match("studio_0123456789abcdef.jpg"))
        for name in ("studio_face_oval.png", "studio_face_oval_75_555.png", "swatch_cat.jpg",
                     "variation_src_0123456789abcdef.png", "studio_0123456789abcdef"):
            self.assertFalse(ig.SENT.match(name), name)

    def test_only_a_named_folder_of_a_comfyui_on_this_pc_is_tidied(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "output"))

            def backend(**k):
                return ig.clean_backend(dict({"id": "b", "url": "http://127.0.0.1:8188"}, **k))
            self.assertEqual(ig.comfy_folder(backend(folder=d)), d)
            self.assertEqual(ig.comfy_folder(backend(folder=d, url="http://localhost:8188")), d)
            # Not named: never guessed, not even from where it is started.
            self.assertIsNone(ig.comfy_folder(backend(start=os.path.join(d, "start.cmd"))))
            # Another machine's, whatever folder it names; a folder that is not ComfyUI's.
            self.assertIsNone(ig.comfy_folder(backend(folder=d, url="http://100.127.17.38:8188")))
            self.assertIsNone(ig.comfy_folder(backend(folder=os.path.join(d, "output"))))
        b = ig.clean_backend({"url": "http://127.0.0.1:8188"})
        self.assertEqual((b["folder"], b["keep_days"]), ("", ig.KEEP_DAYS))
        # The Backends form hands a number back as a float, or None when cleared.
        self.assertEqual(backend(keep_days=3.0)["keep_days"], 3)
        self.assertEqual(backend(keep_days=None)["keep_days"], ig.KEEP_DAYS)
        self.assertEqual(backend(keep_days=-2)["keep_days"], 0)


class TestTidy(TempStudioMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.comfy = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.comfy, ignore_errors=True)
        self.out = os.path.join(self.comfy, "output", "ImageStudio")
        self.inp = os.path.join(self.comfy, "input")
        os.makedirs(self.out)
        os.makedirs(self.inp)

    def backend(self, **k):
        return ig.clean_backend(dict({"id": "5090", "name": "5090 Workstation",
                                      "url": "http://127.0.0.1:8188",
                                      "folder": self.comfy}, **k))

    def fill(self):
        """ComfyUI's folders after a week and a day. -> (what is the app's
        and old, everything else)."""
        old = [aged(os.path.join(self.out, "zimage_hq_%s_00001_.png" % JOB), 8),
               aged(os.path.join(self.out, "zimage_hq_%s_hands_00001_.png" % JOB), 30),
               aged(os.path.join(self.inp, "studio_0123456789abcdef.png"), 8)]
        keep = [aged(os.path.join(self.out, "zimage_hq_aaaaaaaaaaaa_00001_.png"), 6),
                aged(os.path.join(self.out, "livetest_1790470158_eyes_00001_.png"), 90),
                aged(os.path.join(self.out, "_exp", "zimage_hq_%s_00001_.png" % JOB), 90),
                aged(os.path.join(self.comfy, "output", "zimage_hq_%s_00001_.png" % JOB), 90),
                aged(os.path.join(self.inp, "studio_fedcba9876543210.png"), 1),
                aged(os.path.join(self.inp, "studio_face_oval.png"), 90),
                aged(os.path.join(self.inp, "swatch_cat.jpg"), 90),
                aged(os.path.join(self.inp, "lora_l1lya", "studio_0123456789abcdef.png"), 90)]
        return old, keep

    def test_old_job_pictures_and_uploads_go_and_nothing_else(self):
        old, keep = self.fill()
        done = self.studio.tidy(self.backend())
        self.assertEqual((done["output"], done["input"], done["temp"]), (2, 1, 0))
        self.assertEqual(done["bytes"], 3 * len(PNG))
        self.assertEqual([p for p in old if os.path.exists(p)], [])
        self.assertEqual([p for p in keep if not os.path.exists(p)], [])

    def test_the_days_kept_are_the_backends_and_none_keeps_everything(self):
        old, keep = self.fill()
        for b in (self.backend(keep_days=0), self.backend(folder=""),
                  self.backend(url="http://100.127.17.38:8188")):
            done = self.studio.tidy(b)
            self.assertEqual((done["output"], done["input"]), (0, 0), b)
        # Twenty days: only the picture of a month ago is old enough.
        done = self.studio.tidy(self.backend(keep_days=20))
        self.assertEqual((done["output"], done["input"]), (1, 0))
        self.assertEqual(sorted(os.path.basename(p) for p in old if os.path.exists(p)),
                         ["studio_0123456789abcdef.png", "zimage_hq_%s_00001_.png" % JOB])
        self.assertEqual([p for p in keep if not os.path.exists(p)], [])

    def test_a_picture_written_only_to_be_uploaded_does_not_outlive_the_upload(self):
        client = self.studio.client(self.studio.backend("5090"))
        name = self.studio._upload_made(client, "finish", "%s_0.png" % JOB, PNG)
        self.assertEqual(name, "studio_%s_0.png" % JOB)
        self.assertEqual(os.listdir(os.path.join(self.dir, "finish")), [])
        # Nor a failed one.
        with patch.object(client, "upload_image", side_effect=ig.ComfyError("gone")):
            with self.assertRaises(ig.ComfyError):
                self.studio._upload_made(client, "fix_shapes", "%s_redraw_0.png" % JOB, PNG)
        self.assertEqual(os.listdir(os.path.join(self.dir, "fix_shapes")), [])

    def test_what_a_dead_job_left_in_the_library_goes_after_a_day(self):
        old = [aged(os.path.join(self.dir, "finish", "%s_0.png" % JOB), 2),
               aged(os.path.join(self.dir, "finish", "%s_head_0.png" % JOB), 2),
               aged(os.path.join(self.dir, "fix_shapes", "%s_swap_1.png" % JOB), 2)]
        keep = [aged(os.path.join(self.dir, "finish", "%s_1.png" % JOB), 0),
                aged(os.path.join(self.dir, "finish", "mine.png"), 9),
                aged(os.path.join(self.dir, "fix_oval.png"), 9)]
        done = self.studio.tidy(self.backend(folder=""))
        self.assertEqual((done["temp"], done["output"], done["input"]), (3, 0, 0))
        self.assertEqual([p for p in old if os.path.exists(p)], [])
        self.assertEqual([p for p in keep if not os.path.exists(p)], [])

    def test_a_lane_tidies_after_its_last_job_and_not_again_for_hours(self):
        self.fill()
        backends = [dict(b, folder=self.comfy) if b["id"] == "5090" else b
                    for b in self.studio.backends()]
        self.studio.lib.save("backends", backends)
        jobs = self.studio.submit(dict(ig.default_settings(), scene="x", backend="5090"))
        settle(jobs)
        deadline = time.monotonic() + 3
        while "5090" not in self.studio.tidied and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(jobs[0].status, "complete", jobs[0].detail)
        self.assertFalse(os.path.exists(os.path.join(self.out, "zimage_hq_%s_00001_.png" % JOB)))
        with patch.object(self.studio, "tidy") as tidy:
            self.assertIsNone(self.studio.upkeep(self.studio.backend("5090")))
            tidy.assert_not_called()
            self.studio.tidied["5090"] -= ig.TIDY_EVERY + 1
            self.studio.upkeep(self.studio.backend("5090"))
            tidy.assert_called_once()

    def test_housekeeping_that_fails_is_logged_and_costs_nothing(self):
        with patch.object(self.studio, "tidy", side_effect=RuntimeError("boom")), \
                patch.object(ig.doctor, "log_error") as log:
            self.assertIsNone(self.studio.upkeep(self.backend()))
        self.assertIn("RuntimeError: boom", log.call_args.args[0])
        self.assertIn("Traceback", log.call_args.args[0])

    def test_a_file_that_will_not_go_is_left_for_the_next_tidy(self):
        old, _ = self.fill()
        real = os.remove

        def remove(path):
            if path.endswith("_hands_00001_.png"):
                raise PermissionError("open in a viewer")
            real(path)
        with patch.object(ig.os, "remove", side_effect=remove):
            done = self.studio.tidy(self.backend())
        self.assertEqual((done["output"], done["input"]), (1, 1))
        self.assertTrue(os.path.exists(old[1]))


class Answer(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class TestAnUploadIsSentAgain(unittest.TestCase):
    def test_the_same_picture_is_sent_once_until_it_may_have_been_tidied_away(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "photo.png")
            Path(path).write_bytes(PNG)
            client = ig.ComfyUIClient({"id": "b", "name": "B", "url": "http://127.0.0.1:1"})
            sent = []

            def answer(req, timeout=None):
                sent.append(req)
                body = req.data.split(b'filename="')[1].split(b'"')[0].decode()
                return Answer(json.dumps({"name": body}).encode())
            with patch.object(client, "_open", side_effect=answer):
                name = client.upload_image(path)
                self.assertTrue(ig.SENT.match(name), name)
                self.assertEqual(client.upload_image(path), name)
                self.assertEqual(len(sent), 1)
                # Older than UPLOAD_TTL: the file may be gone from ComfyUI's input.
                client.uploaded[name] -= ig.UPLOAD_TTL + 1
                self.assertEqual(client.upload_image(path), name)
                self.assertEqual(len(sent), 2)
                self.assertEqual(client.upload_image(path), name)
                self.assertEqual(len(sent), 2)
        # A day kept is the least a backend can ask for: longer than an upload is believed.
        self.assertLess(ig.UPLOAD_TTL, DAY)


class TestTheCriticsLogRollsOver(TempStudioMixin, unittest.TestCase):
    def test_past_its_size_the_log_becomes_the_one_before_and_starts_again(self):
        job = ig.Job({}, self.studio.backend("5090"))
        path = os.path.join(self.dir, ig.CRITIC_LOG)
        with patch.object(ig, "CRITIC_LOG_MAX", 200):
            self.studio._critic_log(job, "first look")
            self.assertFalse(os.path.exists(path + ".1"))
            with open(path, "a", encoding="utf-8") as f:
                f.write("x" * 300)
            self.studio._critic_log(job, "second look")
            before = Path(path + ".1").read_text(encoding="utf-8")
            self.assertIn("first look", before)
            now = Path(path).read_text(encoding="utf-8")
            self.assertIn("second look", now)
            self.assertNotIn("first look", now)
            # One generation back, no more: the next rollover replaces it.
            with open(path, "a", encoding="utf-8") as f:
                f.write("y" * 300)
            self.studio._critic_log(job, "third look")
            self.assertNotIn("first look", Path(path + ".1").read_text(encoding="utf-8"))
            self.assertEqual(sorted(n for n in os.listdir(self.dir) if n.startswith(
                ig.CRITIC_LOG)), [ig.CRITIC_LOG, ig.CRITIC_LOG + ".1"])

    def test_a_log_that_will_not_roll_is_still_written_to(self):
        job = ig.Job({}, self.studio.backend("5090"))
        path = os.path.join(self.dir, ig.CRITIC_LOG)
        Path(path).write_text("z" * 300, encoding="utf-8")
        with patch.object(ig, "CRITIC_LOG_MAX", 200), \
                patch.object(ig.os, "replace", side_effect=PermissionError("held")):
            self.studio._critic_log(job, "a look")
        self.assertIn("a look", Path(path).read_text(encoding="utf-8"))

    def test_it_is_as_big_as_the_error_log_may_get(self):
        self.assertEqual(ig.CRITIC_LOG_MAX, ig.doctor.LOG_MAX_BYTES)


if __name__ == "__main__":
    unittest.main()
