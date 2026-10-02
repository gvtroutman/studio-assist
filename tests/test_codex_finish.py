"""Codex handoff provenance and History round trip, without network or models."""
import copy
import json
import os
import tempfile
import unittest

from apps.image_studio import codex_finish as cf, imagegen as ig
from core.icons import png, png_to_rgba


class CodexFinishTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = self.temp.name
        self.source = os.path.join(self.root, "original.png")
        self.original = png(bytes([30, 40, 50, 255]) * 4, 2, 2)
        with open(self.source, "wb") as stream:
            stream.write(self.original)
        self.record = {"id": "base", "settings": {"scene": "a festival", "seed": 42},
                       "prompt": "Two people at a festival", "graph": {"old": {}},
                       "passes": [{"old": True}], "license": "Keep attribution"}
        self.record["settings"]["scene_faces"] = {"people": [{"region": [0, 0, .5, .5]}]}

    def test_export_and_import_preserve_source_and_do_not_claim_old_passes(self):
        before = copy.deepcopy(self.record)
        h = cf.export(self.root, self.record, self.source, "Repair hands")
        with open(h["manifest"], encoding="utf-8") as stream:
            payload = json.load(stream)
        self.assertEqual(payload["record"], before)
        self.assertIn(h["output"], h["request"])
        self.assertIn("Repair hands", h["request"])
        with open(payload["source_copy"], "rb") as stream:
            self.assertEqual(stream.read(), self.original)
        self.assertNotEqual(os.path.dirname(payload["source_copy"]), h["folder"])
        with open(h["redacted_image"], "rb") as stream:
            masked, width, height = png_to_rgba(stream.read())
        self.assertEqual((width, height), (2, 2))
        self.assertEqual(masked[:4], cf.REDACTION_RGBA)
        self.assertEqual(masked[4:], bytes([30, 40, 50, 255]) * 3)
        with open(h["output"], "wb") as stream:
            stream.write(png(bytes([80, 60, 40, 255]) * 4, 2, 2))
        history = ig.History(os.path.join(self.root, "history"))
        result = cf.import_finished(history, h["manifest"], h["output"])
        self.assertEqual(result["settings"], before["settings"])
        self.assertEqual(result["codex_finish"]["source_id"], "base")
        self.assertTrue(result["codex_finish"]["faces_redacted_for_handoff"])
        self.assertEqual((result["width"], result["height"]), (2, 2))
        with open(result["images"][0], "rb") as stream:
            pixels, _, _ = png_to_rgba(stream.read())
        self.assertEqual(pixels[:4], bytes([30, 40, 50, 255]))
        self.assertEqual(pixels[4:], bytes([80, 60, 40, 255]) * 3)
        self.assertEqual(result["license"], before["license"])
        self.assertNotIn("graph", result)
        self.assertNotIn("passes", result)
        self.assertEqual(history.list()[0]["codex_finish"], result["codex_finish"])
        self.assertEqual(self.record, before)
        with open(self.source, "rb") as stream:
            self.assertEqual(stream.read(), self.original)
        with self.assertRaisesRegex(ig.ComfyError, "Codex"):
            ig.again(result)

    def test_export_is_fresh_each_time_and_requires_a_brief(self):
        first = cf.export(self.root, self.record, self.source)
        second = cf.export(self.root, self.record, self.source)
        self.assertNotEqual(first["folder"], second["folder"])
        with self.assertRaisesRegex(ValueError, "Describe"):
            cf.export(self.root, self.record, self.source, "  ")

    def test_original_and_non_image_cannot_be_imported(self):
        h = cf.export(self.root, self.record, self.source)
        history = ig.History(os.path.join(self.root, "history"))
        with self.assertRaisesRegex(ValueError, "original"):
            cf.import_finished(history, h["manifest"], self.source)
        with open(h["output"], "w") as stream:
            stream.write("not an image")
        with self.assertRaisesRegex(ValueError, "picture"):
            cf.import_finished(history, h["manifest"], h["output"])
        self.assertEqual(history.list(), [])

    def test_invalid_manifest_cannot_create_history(self):
        path = os.path.join(self.root, "bad.json")
        with open(path, "w") as stream:
            json.dump({"version": 99}, stream)
        with self.assertRaisesRegex(ValueError, "handoff"):
            cf.import_finished(ig.History(self.root), path, self.source)

    def test_resized_output_and_missing_protection_are_refused(self):
        h = cf.export(self.root, self.record, self.source)
        with open(h["output"], "wb") as stream:
            stream.write(png(bytes([80, 60, 40, 255]) * 6, 3, 2))
        with self.assertRaisesRegex(ValueError, "canvas"):
            cf.import_finished(ig.History(self.root), h["manifest"], h["output"])
        with self.assertRaisesRegex(ValueError, "faces"):
            cf.export(self.root, self.record, self.source, regions=[])

    def test_matching_proportions_fit_without_resizing_original_face_pixels(self):
        edited = png(bytes([80, 60, 40, 255]) * 16, 4, 4)
        result = cf.restore_faces(self.original, edited, [[0, 0, .5, .5]])
        pixels, width, height = png_to_rgba(result)
        self.assertEqual((width, height), (2, 2))
        self.assertEqual(pixels[:4], bytes([30, 40, 50, 255]))
        self.assertEqual(pixels[4:], bytes([80, 60, 40, 255]) * 3)

    def test_shared_folder_and_request_exclude_originals_and_metadata(self):
        self.record["prompt"] = "PRIVATE_IDENTITY_DESCRIPTION"
        self.record["references"] = ["PRIVATE_FACE_PHOTO.jpg"]
        h = cf.export(self.root, self.record, self.source, "Repair hands")
        self.assertEqual(set(os.listdir(h["folder"])), {"redacted.png", "request.md"})
        self.assertIn(h["redacted_image"], h["request"])
        self.assertNotIn(self.source, h["request"])
        self.assertNotIn(h["manifest"], h["request"])
        self.assertNotIn("PRIVATE_IDENTITY_DESCRIPTION", h["request"])
        self.assertNotIn("PRIVATE_FACE_PHOTO", h["request"])
        self.assertEqual(cf.output_folder(h["manifest"]), h["folder"])

    def test_redaction_is_opaque_and_strips_embedded_original_metadata(self):
        width, height = 8, 6
        original = bytes(c for y in range(height) for x in range(width)
                         for c in (x * 8, y * 9, x + y, 100))
        source = ig.png_text(png(original, width, height), "Comment", "PRIVATE_FACE_METADATA")
        regions = [[.13, .17, .51, .52], [.4, .4, .9, .9]]
        redacted = cf.redact_faces(source, regions)
        self.assertNotIn(b"PRIVATE_FACE_METADATA", redacted)
        pixels, _, _ = png_to_rgba(redacted)
        covered = {(x, y) for l, t, r, b in cf.pixel_boxes(regions, width, height)
                   for y in range(t, b) for x in range(l, r)}
        for y in range(height):
            for x in range(width):
                offset = (y * width + x) * 4
                expected = cf.REDACTION_RGBA if (x, y) in covered else original[offset:offset + 4]
                self.assertEqual(pixels[offset:offset + 4], expected)

    def test_original_can_be_removed_after_export_and_faces_still_restore_locally(self):
        h = cf.export(self.root, self.record, self.source)
        os.remove(self.source)
        with open(h["output"], "wb") as stream:
            stream.write(png(bytes([80, 60, 40, 255]) * 4, 2, 2))
        result = cf.import_finished(ig.History(os.path.join(self.root, "history")),
                                    h["manifest"], h["output"])
        with open(result["images"][0], "rb") as stream:
            pixels, _, _ = png_to_rgba(stream.read())
        self.assertEqual(pixels[:4], bytes([30, 40, 50, 255]))

    def test_unedited_redacted_picture_cannot_complete_the_pipeline(self):
        h = cf.export(self.root, self.record, self.source)
        with self.assertRaisesRegex(ValueError, "covered"):
            cf.import_finished(ig.History(self.root), h["manifest"], h["redacted_image"])

    def test_pipeline_saves_handoff_and_waits_without_holding_a_lane(self):
        from types import SimpleNamespace
        lib = SimpleNamespace(root=self.root)
        history = ig.History(os.path.join(self.root, "history"))
        studio = SimpleNamespace(lib=lib, history=history)
        job = ig.Job({"codex_finish": True}, {"id": "test"})
        record = dict(self.record, created_ts=1, created="1970-01-01")
        ig.Studio.save_result(studio, job, record, [("made.png", self.original)])
        queue = SimpleNamespace(notify=lambda job: None)
        ig.JobQueue._finish(queue, job, "complete")
        self.assertEqual(job.status, "awaiting_codex")
        self.assertIn(job.status, ig.FINISHED)
        self.assertEqual(job.record["codex_handoff"]["state"], "pending")
        h = job.record["codex_handoff"]["handoffs"][0]
        self.assertTrue(os.path.isfile(h["manifest"]))
        with open(h["output"], "wb") as stream:
            stream.write(png(bytes([80, 60, 40, 255]) * 4, 2, 2))
        result = cf.import_finished(history, h["manifest"], h["output"])
        self.assertEqual(result["codex_finish"]["remaining"], 0)
        parent = next(r for r in history.list() if r["id"] == "base")
        self.assertEqual(parent["codex_handoff"]["state"], "complete")
        stages = ig.pipeline_stages(SimpleNamespace(get=lambda *a: None), {"codex_finish": True,
                                                                            "hand_pass": False})
        self.assertEqual(stages[-2:], [("codex", "Codex finish"), ("complete", "Complete")])


if __name__ == "__main__":
    unittest.main()
