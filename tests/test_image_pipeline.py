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


class TestImagePipeline(unittest.TestCase):
    def client(self):
        return ig.ComfyUIClient({"id": "test", "name": "Test GPU",
                                 "url": "http://127.0.0.1:9"})

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
