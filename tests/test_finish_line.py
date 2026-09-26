"""Recovery and cancellation regressions; no apps, models or network."""
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import studio_facefusion as ff
import studio_imagegen as ig
import studio_scene as sc
from test_imagegen import TempStudioMixin, PNG, settle


class TestFaceTargets(unittest.TestCase):
    def test_scene_builder_payload_carries_the_character_identity(self):
        scene = sc.new_scene()
        person = sc.new_object('person')
        person['character'] = 'character'
        scene['objects'].append(person)
        targets = sc.face_targets(scene, {'character': {'identity': 'person'}}, {})
        self.assertEqual(targets[0]['identity'], 'person')

    def test_lora_only_identities_do_not_require_a_face_swap(self):
        lib = SimpleNamespace(get=lambda *_: {'id': 'a', 'name': 'A', 'lora': 'a', 'references': []})
        self.assertEqual(ff.selected(lib, {'identities': ['a']}), [])

    def test_scene_regions_override_profile_selection_order(self):
        profiles = {"a": {"id": "a", "name": "A", "references": ["a.png"]},
                    "b": {"id": "b", "name": "B", "references": ["b.png"]}}
        lib = SimpleNamespace(get=lambda kind, key: profiles.get(key))
        settings = {"identities": ["a", "b"], "scene_faces": {"people": [
            {"id": "p1", "identity": "a", "region": [.6, .1, 1, .6]},
            {"id": "p2", "identity": "b", "region": [0, .1, .4, .6]}]}}
        selected = ff.selected(lib, settings)
        boxes = [[.1, .2, .3, .4], [.7, .2, .9, .4]]
        self.assertEqual([ff.target_face(boxes, region=p['target_region']) for p in selected], [1, 0])
        self.assertNotIn('target_region', profiles['a'])

    def test_ambiguous_extra_and_missing_faces_do_not_guess(self):
        boxes = [[.1, .2, .3, .4], [.4, .2, .6, .4], [.7, .2, .9, .4]]
        for options in ({}, {"region": [0, 0, 1, 1]}, {"index": 0, "count": 2},
                        {"region": [0, .8, 1, 1]}):
            with self.subTest(options=options), self.assertRaisesRegex(RuntimeError, 'Choose face'):
                ff.target_face(boxes, **options)
        self.assertEqual(ff.target_face(boxes, point=[.8, .3]), 2)

    def test_same_identity_can_appear_at_two_scene_positions(self):
        lib = SimpleNamespace(get=lambda *_: {"id": "a", "name": "A", "references": ["a.png"]})
        selected = ff.selected(lib, {"scene_faces": {"people": [
            {"identity": "a", "region": [0, 0, .4, 1]},
            {"identity": "a", "region": [.6, 0, 1, 1]}]}})
        self.assertEqual(len(selected), 2)


class TestFinishRecovery(TempStudioMixin, unittest.TestCase):
    def profile(self):
        path = os.path.join(self.dir, 'reference.png')
        Path(path).write_bytes(PNG)
        self.studio.lib.save('identities', [{'id': 'person', 'name': 'Person',
            'references': [path], 'use_references': False}])
        return path

    def settings(self):
        return dict(ig.default_settings(), model='z-image-turbo', backend='5090',
                    scene='A portrait', identities=['person'], auto_refine=False)

    def test_failed_finish_keeps_picture_and_retries_offline_with_saved_profile(self):
        self.profile()
        with patch.object(ff, 'available', return_value=True), patch.object(ff, 'swap', side_effect=RuntimeError('failed')):
            jobs = self.studio.submit(self.settings())
            settle(jobs)
        job = jobs[0]
        self.assertEqual(job.status, 'failed')
        self.assertEqual(Path(job.outputs[0]).read_bytes(), PNG)
        record = self.studio.history.list()[0]
        self.assertEqual(record['finish']['profiles'][0]['name'], 'Person')
        self.studio.lib.save('identities', [])
        self.studio.lib.save('backends', [])
        with patch.object(ff, 'available', return_value=True), patch.object(ff, 'swap', return_value=(PNG, {})) as swap, \
                patch.object(self.studio, 'client', side_effect=AssertionError('No backend needed')):
            jobs = self.studio.submit(ig.retry_faces(record))
            settle(jobs)
        self.assertEqual(jobs[0].status, 'complete', jobs[0].detail)
        self.assertEqual(swap.call_args.args[1]['name'], 'Person')
        self.assertEqual(Path(job.outputs[0]).read_bytes(), PNG)
        # The retry finishes the kept checkpoint instead of saving another copy of it.
        kept = json.loads(Path(record['path']).read_text(encoding='utf-8'))
        self.assertEqual(kept['finish']['state'], 'complete')
        self.assertEqual(kept['finish']['results'], jobs[0].outputs)
        listed = self.studio.history.list()
        self.assertEqual(len(listed), 1)
        self.assertNotIn('finish', listed[0])

    def test_face_only_fix_never_checks_comfyui(self):
        path = self.profile()
        self.studio.lib.save('backends', [])
        settings = {'mode': 'fix', 'backend': 'auto', 'fix': {'image': path,
                    'face_swap': 'person', 'face_point': [.25, .4]}}
        with patch.object(ff, 'available', return_value=True), patch.object(ff, 'swap', return_value=(PNG, {})) as swap, \
                patch.object(self.studio, 'client', side_effect=AssertionError('No backend needed')):
            jobs = self.studio.submit(settings)
            settle(jobs)
        self.assertEqual(jobs[0].status, 'complete', jobs[0].detail)
        self.assertEqual(swap.call_args.args[1]['target_point'], [.25, .4])

    def test_missing_references_and_runtime_are_in_preview(self):
        self.profile()
        self.studio.lib.get('identities', 'person')['references'] = []
        with patch.object(ff, 'available', return_value=False):
            errors = self.studio.preview(self.settings(), self.backend('5090')).errors
        self.assertTrue(any('FaceFusion' in e for e in errors))
        self.assertTrue(any('Manage profiles' in e and 'reference' in e for e in errors))

    def test_cancel_returns_while_backend_interrupt_is_blocked(self):
        entered, release, returned = threading.Event(), threading.Event(), threading.Event()
        def interrupt(_):
            entered.set()
            release.wait(5)
        job = ig.Job({}, self.backend('5090'))
        job.prompt_id = 'prompt'
        job.status = 'sampling'
        with patch.object(self.studio, 'client', return_value=SimpleNamespace(cancel_job=interrupt)):
            worker = threading.Thread(target=lambda: (self.studio.queue.cancel(job), returned.set()))
            worker.start()
            try:
                self.assertTrue(entered.wait(1))
                self.assertTrue(returned.wait(1), 'Cancel blocked on the backend')
                self.assertTrue(job.cancel.is_set())
            finally:
                release.set()
                worker.join(2)


class TestSceneClose(unittest.TestCase):
    def test_retry_tick_cancels_its_pending_timer_and_never_starts_when_closing(self):
        from studio_chat import Chat
        app = SimpleNamespace(closing=True, _stand_down=Mock(), _spawn=Mock())
        Chat._retry_host(app)
        app._stand_down.assert_called_once_with('host_timer')
        app._spawn.assert_not_called()

    def test_failed_save_and_cancelled_save_as_veto_final_close(self):
        from studio_scene_ui import SceneBuilder
        sb = SceneBuilder.__new__(SceneBuilder)
        sb.win = Mock()
        sb.dirty = True
        sb.has_content = lambda: True
        sb._save_recovery = Mock()
        sb.save = Mock(return_value=False)
        with patch('studio_scene_ui.messagebox.askyesnocancel', return_value=True):
            self.assertFalse(sb.close(final=True))
        sb.win.destroy.assert_not_called()

    def test_quit_and_tab_close_leave_state_intact_after_veto(self):
        from studio_chat import Chat
        session = SimpleNamespace(images=SimpleNamespace(can_close=lambda: False))
        app = SimpleNamespace(closing=False, sessions={'images': session}, active='images')
        Chat._quit(app)
        self.assertFalse(app.closing)
        Chat._close_tab(app)
        self.assertIs(app.sessions['images'], session)

    def test_recovery_copy_is_valid_and_leaves_original_untouched(self):
        from studio_scene_ui import SceneBuilder
        with tempfile.TemporaryDirectory() as folder:
            sb = SceneBuilder.__new__(SceneBuilder)
            sb.scene = sc.new_scene(details='A recovered room')
            sb.dirty = True
            sb.has_content = lambda: True
            sb.recovery_path = os.path.join(folder, 'recovery.scene.json')
            sb.status = Mock()
            sb._save_recovery()
            restored, problems = sc.load(sb.recovery_path)
            self.assertEqual(restored['details'], 'A recovered room')
            self.assertEqual(problems, [])


if __name__ == '__main__':
    unittest.main()
