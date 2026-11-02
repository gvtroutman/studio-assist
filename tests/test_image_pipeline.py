"""Image pipeline transport and finishing regressions; no server or GPU."""
import io
import json
import os
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock, patch

import apps.image_studio.imagegen as ig
import apps.image_studio.blend as blend
from tests.test_imagegen import FakeClient, FaceClient, PNG, TempStudioMixin


def failed_entry(status=None):
    return {"status": status if status is not None else {
        "status_str": "error", "completed": False, "messages": [["execution_error", {
            "node_id": "redraw", "node_type": "KSampler", "exception_type": "RuntimeError",
            "exception_message": "CUDA out of memory"}]]},
        "outputs": {"save": {"images": [{"filename": "partial.png", "type": "output"}]}}}


class TestCompletion(TempStudioMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.addCleanup(self.studio.close)

    def generate(self, client_factory, **settings):
        self.studio.client_factory = client_factory
        values = dict(ig.default_settings(), model="flux-dev", scene="A portrait",
                      face_detail=False, hand_pass=False, auto_refine=False)
        values.update(settings)
        job = ig.Job(values, self.backend("5090"))
        self.studio.run_job(job, Mock())
        return job

    def test_a_failed_generation_cannot_complete_with_a_partial_picture(self):
        class FailedClient(FakeClient):
            def listen_for_progress(self, *args, **kwargs):
                return failed_entry()
        with patch.object(FailedClient, "fetch", return_value=PNG) as fetched:
            job = self.generate(FailedClient)
        self.assertEqual(job.status, "failed", job.detail)
        self.assertIn("node redraw (KSampler)", job.detail)
        fetched.assert_not_called()
        self.assertEqual(self.studio.history.list(), [])

    def test_a_backend_interruption_is_cancelled_even_if_it_saved_an_image(self):
        class InterruptedClient(FakeClient):
            def listen_for_progress(self, *args, **kwargs):
                return failed_entry({"status_str": "error", "messages": [
                    ["execution_interrupted", {}]]})
        job = self.generate(InterruptedClient)
        self.assertEqual(job.status, "cancelled", job.detail)
        self.assertEqual(job.outputs, [])
        self.assertEqual(self.studio.history.list(), [])

    def test_a_failed_face_redraw_keeps_the_base_and_records_no_redraw(self):
        class FailedFaceClient(FaceClient):
            def listen_for_progress(self, pid, *args, **kwargs):
                if "fs" in self.graphs[int(pid[3:]) - 1]:
                    return failed_entry()
                return super().listen_for_progress(pid, *args, **kwargs)
        with patch.object(FailedFaceClient, "fetch", return_value=PNG) as fetched:
            job = self.generate(FailedFaceClient, face_detail=True)
        self.assertEqual(job.status, "complete", job.detail)
        self.assertIsNone(job.face_graph)
        self.assertIsNone(job.face)
        self.assertEqual(fetched.call_args.args[0]["filename"], "ImageStudio_00001_.png")
        self.assertTrue(any("face pass failed" in warning for warning in job.record["warnings"]))


class TestResultBoundaries(unittest.TestCase):
    def test_a_status_only_failure_is_explained(self):
        self.assertTrue(ig.run_errors(failed_entry({"status_str": "error"})))
        for status in (None, {}, {"completed": True, "status_str": "success"}):
            self.assertEqual(ig.run_errors({"status": status}), [])

    def test_try_on_rejects_saved_images_and_previews_from_a_failed_run(self):
        for outputs in ({"save": {"images": [{"filename": "partial.png"}]}},
                        {"dp": {"images": [{"filename": "partial.png", "type": "temp"}]}}):
            with self.subTest(outputs=outputs):
                entry = dict(failed_entry(), outputs=outputs)
                client = Mock()
                client.listen_for_progress.return_value = entry
                job = ig.Job({}, {"name": "Test"})
                job.dress = {"graphs": []}
                with self.assertRaisesRegex(ig.ComfyError, "node redraw"):
                    ig.Studio._dress_run(None, job, client, {}, [], 0, Mock())
                client.watch.return_value.close.assert_called_once()

    def test_a_failed_real_face_paste_keeps_the_previous_face_metadata(self):
        client = Mock()
        client.node_types.return_value = {ig.PASTE_NODE}
        client.upload_image.return_value = "reference.png"
        client.queue_workflow.return_value = "paste-id"
        client.listen_for_progress.return_value = failed_entry()
        job = ig.Job({}, {"name": "Test"})
        job.face = {"redrawn": 1}
        job.real_faces = [{"name": "Person", "photos": [__file__], "box": [1, 2, 3, 4]}]
        plan = ig.Plan()
        files = [{"filename": "base.png"}]
        with patch.object(ig, "paste_report", return_value=[{"name": "Person", "pasted": True}]):
            result = ig.Studio._real_faces(None, job, client, plan,
                                          {"seed": 1, "filename_prefix": "test"}, files, Mock())
        self.assertEqual(result, files)
        self.assertEqual(job.face, {"redrawn": 1})
        self.assertEqual(job.prompt_id, "paste-id")
        self.assertIsNone(job.paste_graph)
        self.assertEqual(plan.notes, [])
        self.assertTrue(plan.warnings)
        client.watch.return_value.close.assert_called_once()
        names = [call[0] for call in client.mock_calls]
        self.assertLess(names.index("watch"), names.index("queue_workflow"))

    def test_angles_and_blend_reject_a_partial_picture(self):
        client = Mock()
        client.listen_for_progress.return_value = failed_entry()
        with self.assertRaisesRegex(ig.ComfyError, "node redraw"):
            blend.run(client, {})
        client.fetch.assert_not_called()
        client.watch.return_value.close.assert_called_once()

    def test_quick_finders_reject_error_messages_even_without_a_status_string(self):
        client = Mock(backend={"name": "Test"})
        client.get_history.return_value = failed_entry({"messages": [["execution_error", {
            "node_id": "find", "exception_message": "finder failed"}]]})
        with self.assertRaisesRegex(ig.ComfyError, "finder failed"):
            ig.Studio._run_quick(None, client, {})

    def test_try_on_and_blend_watch_before_submission_and_close_on_rejection(self):
        for mode in ("dress", "blend"):
            with self.subTest(mode=mode):
                calls = []
                watch = SimpleNamespace(close=lambda: calls.append("close"))
                def submit(graph):
                    calls.append("submit")
                    raise ig.ComfyError("rejected")
                client = SimpleNamespace(watch=lambda: calls.append("watch") or watch,
                                         queue_workflow=submit)
                job = ig.Job({}, {"name": "Test"})
                job.dress = {"graphs": []}
                with self.assertRaisesRegex(ig.ComfyError, "rejected"):
                    if mode == "dress":
                        ig.Studio._dress_run(None, job, client, {}, [], 0, Mock())
                    else:
                        blend.run(client, {})
                self.assertEqual(calls, ["watch", "submit", "close"])


class TestImagePipeline(unittest.TestCase):
    def client(self):
        return ig.ComfyUIClient({"id": "test", "name": "Test GPU",
                                 "url": "http://127.0.0.1:9"})

    def test_a_message_only_failure_settles_without_waiting_for_the_deadline(self):
        client = self.client()
        entry = {"status": {"messages": [["execution_error", {
            "node_id": "find", "exception_message": "finder failed"}]]}, "outputs": {}}
        client.get_history = Mock(return_value=entry)
        client.position = Mock()
        client.cancel_job = Mock()
        watch = ig.Watch(None, ig.queue.Queue(), "")
        result = client.listen_for_progress("pid", Mock(), timeout=0, watch=watch)
        self.assertEqual(result, entry)
        client.position.assert_not_called()
        client.cancel_job.assert_not_called()

    def test_a_renamed_upload_is_reused_by_content(self):
        client = self.client()
        response = {"name": "renamed.png", "subfolder": "references"}
        opened = Mock(side_effect=lambda *args: io.BytesIO(json.dumps(response).encode()))
        with tempfile.TemporaryDirectory() as folder:
            paths = [os.path.join(folder, name) for name in ("one.png", "two.png")]
            for path in paths:
                with open(path, "wb") as f:
                    f.write(b"same picture")
            with patch.object(client, "_open", opened):
                names = [client.upload_image(path) for path in paths]
                self.assertEqual(names, ["references/renamed.png"] * 2)
                self.assertEqual(opened.call_count, 1)
                with open(paths[0], "wb") as f:
                    f.write(b"changed picture")
                client.upload_image(paths[0])
                self.assertEqual(opened.call_count, 2)

    def test_cancellation_reaches_a_prompt_accepted_after_stop_was_clicked(self):
        client = self.client()
        client.cancel_job = Mock(return_value=True)
        watch = ig.Watch(None, ig.queue.Queue(), "")
        result = client.listen_for_progress("just-accepted", Mock(), stop=lambda: True,
                                            watch=watch)
        self.assertIsNone(result)
        client.cancel_job.assert_called_once_with("just-accepted")

    def test_two_callers_share_one_upload(self):
        client = self.client()
        sending, release, second = threading.Event(), threading.Event(), threading.Event()
        def upload(req, timeout):
            sending.set()
            if not release.wait(5):
                raise AssertionError("upload was never released")
            return io.BytesIO(b'{"name": "shared.png"}')
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "photo.png")
            with open(path, "wb") as f:
                f.write(b"picture")
            with patch.object(client, "_open", side_effect=upload) as opened, \
                    ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(client.upload_image, path)
                try:
                    self.assertTrue(sending.wait(5))
                    def other():
                        second.set()
                        return client.upload_image(path)
                    other_result = pool.submit(other)
                    self.assertTrue(second.wait(5))
                finally:
                    release.set()
                self.assertEqual(first.result(5), "shared.png")
                self.assertEqual(other_result.result(5), "shared.png")
                self.assertEqual(opened.call_count, 1)

    def test_stop_during_submission_cancels_the_new_refinement_prompt(self):
        client = self.client()
        sending, release = threading.Event(), threading.Event()
        job = ig.Job({}, {"id": "test"})
        watch = ig.Watch(None, ig.queue.Queue(), "")
        def submit(graph):
            sending.set()
            if not release.wait(5):
                raise AssertionError("submission was never released")
            return "new-prompt"
        with patch.object(client, "watch", return_value=watch), \
                patch.object(client, "queue_workflow", side_effect=submit), \
                patch.object(client, "cancel_job", return_value=True) as cancelled, \
                ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(ig.Studio._run_pass, None, job, client, {}, Mock(), "Hands")
            try:
                self.assertTrue(sending.wait(5))
                self.assertIsNone(job.prompt_id)
                job.cancel.set()
            finally:
                release.set()
            self.assertIsNone(result.result(5))
            cancelled.assert_called_once_with("new-prompt")

    def test_a_failed_cancellation_still_closes_its_owned_socket(self):
        client = self.client()
        client.cancel_job = Mock(side_effect=ig.ComfyError("server unavailable"))
        watch = Mock(error="", events=ig.queue.Queue())
        with patch.object(client, "watch", return_value=watch):
            self.assertIsNone(client.listen_for_progress("pid", Mock(), stop=lambda: True))
        watch.close.assert_called_once()
        client.cancel_job.assert_called_once_with("pid")

    def run_pass(self, client, cancelled=False):
        job = ig.Job({}, {"id": "test"})
        if cancelled:
            job.cancel.set()
        return ig.Studio._run_pass(None, job, client, {}, Mock(), "Hands")

    def test_refinement_watches_before_submission_and_always_closes(self):
        calls = []
        watch = SimpleNamespace(close=lambda: calls.append("close"))
        def submit(graph):
            calls.append("submit")
            raise ig.ComfyError("rejected graph")
        client = SimpleNamespace(watch=lambda: calls.append("watch") or watch,
                                 queue_workflow=submit)
        with self.assertRaises(ig.ComfyError):
            self.run_pass(client)
        self.assertEqual(calls, ["watch", "submit", "close"])

    def test_a_cancelled_refinement_is_not_submitted(self):
        client = Mock()
        self.assertIsNone(self.run_pass(client, cancelled=True))
        client.watch.assert_not_called()
        client.queue_workflow.assert_not_called()

    def test_a_refinement_with_an_image_and_an_error_is_not_successful(self):
        entry = {"outputs": {"save": {"images": [{"filename": "partial.png"}]}},
                 "status": {"status_str": "error", "messages": [["execution_error", {
                     "node_id": "redraw", "node_type": "KSampler",
                     "exception_type": "RuntimeError", "exception_message": "out of memory"}]]}}
        client = Mock()
        client.listen_for_progress.return_value = entry
        with self.assertRaisesRegex(ig.ComfyError, "node redraw.*out of memory"):
            self.run_pass(client)
        client.watch.return_value.close.assert_called_once()

    def test_successful_refinement_accepts_legacy_and_null_status(self):
        for status in (None, {}, {"completed": True}):
            with self.subTest(status=status):
                client = Mock()
                client.listen_for_progress.return_value = {
                    "status": status, "outputs": {"save": {"images": [{"filename": "done.png"}]}}}
                self.assertEqual(self.run_pass(client), [{"filename": "done.png",
                                                          "subfolder": "", "type": "output"}])
                client.watch.return_value.close.assert_called_once()

    def test_interrupted_or_failed_refinement_rejects_partial_outputs(self):
        for status in ({"status_str": "error"},
                       {"messages": [["execution_interrupted", {}]]}):
            with self.subTest(status=status):
                client = Mock()
                client.listen_for_progress.return_value = {
                    "status": status, "outputs": {"save": {"images": [{"filename": "partial.png"}]}}}
                with self.assertRaises(ig.ComfyError):
                    self.run_pass(client)


if __name__ == "__main__":
    unittest.main()
