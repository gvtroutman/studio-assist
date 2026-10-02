"""Identity ordering, refusal and the complete job path, without GPU or network."""
import json
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import apps.image_studio.imagegen as ig
from test_imagegen import FakeClient, PNG


class WithAnyoneClient(FakeClient):
    def inventory(self):
        inv = super().inventory()
        inv["diffusion_models"].add("withanyone.safetensors")
        inv["diffusers"] = {"siglip-base-patch16-256-i18n"}
        return inv

    def node_types(self, **kwargs):
        return super().node_types() | {"StudioWithAnyone", "StudioWithAnyoneReferences", "StudioWithAnyonePooled"}


class WithAnyoneTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.lib = ig.Library(self.tmp.name)
        self.backend = self.lib.all("backends")[0]
        self.photos = []
        for name in ("left", "right"):
            path = os.path.join(self.tmp.name, name + ".png")
            with open(path, "wb") as f:
                f.write(PNG)
            self.photos.append(path)
        self.settings = {"model": "withanyone", "scene": "Two people eating pretzels.",
                         "experimental_reference_groups": True,
                         "face_detail": True, "critic_notes": True,
                         "scene_faces": {"real": True, "people": [
                             {"name": "Left", "face": self.photos[0], "region": [.1, .2, .4, .5]},
                             {"name": "Right", "face": self.photos[1], "region": [.6, .2, .9, .5]}]}}

    def plan(self):
        client = WithAnyoneClient(self.backend)
        return ig.compose(self.settings, self.lib, self.backend, client.inventory(),
                          nodes=client.node_types())

    def test_scene_people_keep_their_own_photo_and_position(self):
        p = self.plan()
        self.assertEqual(p.errors, [])
        self.assertEqual(p.images, dict(zip(("face1", "face2"), self.photos)))
        self.assertEqual(json.loads(p.values["identity_boxes"]),
                         [[.1, .2, .4, .5], [.6, .2, .9, .5]])
        self.assertFalse(p.values["face_detail"])
        g = ig.fill(p.workflow, dict(p.values, face1="uploaded-left", face2="uploaded-right"))
        self.assertEqual(g["40"]["inputs"]["face2"], ["face2", 0])
        self.assertEqual(g["face2"]["inputs"]["image"], "uploaded-right")
        # Upstream's "resemblance in form" end: the reference face as it looks.
        self.assertEqual(g["40"]["inputs"]["siglip_weight"], 1.0)

    def test_descriptions_follow_scene_identities_not_stale_form_selection(self):
        self.lib.save('identities', [
            {'name': 'Left', 'description': 'Broad cheeks and a rounded chin.', 'references': self.photos[:1]},
            {'name': 'Other', 'description': 'A narrow jaw.', 'references': self.photos[1:]}])
        self.settings['identities'] = ['other']
        self.settings['scene_faces']['people'][0]['identity'] = 'left'
        p = self.plan()
        self.assertIn('Person 1 on the left (Left)', p.prompt)
        self.assertIn('Broad cheeks and a rounded chin.', p.prompt)
        self.assertNotIn('A narrow jaw.', p.prompt)

    def test_description_survives_save_and_enters_form_prompt_without_notes(self):
        self.settings.pop('scene_faces')
        description = 'Rounded cheeks.\nFine rectangular glasses and reddish brown hair.'
        self.lib.save('identities', [{'name': 'Left', 'description': description,
            'notes': 'Private bookkeeping, not appearance.', 'references': self.photos[:1]}])
        self.assertEqual(ig.Library(self.tmp.name).get('identities', 'left')['description'], description)
        self.settings['identities'] = ['left']
        p = self.plan()
        self.assertIn(description, p.prompt)
        self.assertNotIn('Private bookkeeping', p.prompt)
        self.assertEqual(p.values['prompt'], p.prompt)

    def test_missing_photo_refuses_instead_of_using_a_stranger(self):
        self.settings["scene_faces"]["people"][1]["face"] = "missing.png"
        self.assertIn("Right needs", " ".join(self.plan().errors))

    def test_too_many_people_are_not_truncated(self):
        self.settings["scene_faces"]["people"] *= 3
        self.assertIn("has 6", " ".join(self.plan().errors))

    def test_scene_builder_accepts_face_positions_without_controlnet(self):
        import apps.image_studio.scene.scene as sc
        from apps.image_studio.scene.ui import SceneBuilder
        studio = SimpleNamespace(lib=self.lib, workflow_loader=ig.load_workflow)
        builder = SimpleNamespace(owner=SimpleNamespace(studio=studio))
        self.assertEqual(SceneBuilder.takes(builder, "withanyone"), {"face_positions"})
        scene = sc.clean_scene({})[0]
        maps, notes = sc.scene_maps(scene, {"face_positions"}, self.tmp.name)
        self.assertEqual(maps, {})
        self.assertIn("Face positions", " ".join(notes))

    def test_siglip_readiness_uses_node_choices_not_the_empty_file_list(self):
        client = ig.ComfyUIClient(self.backend)
        choices = {"StudioWithAnyone": {"input": {"required": {
            "siglip": [["siglip-base-patch16-256-i18n"]]}}}}
        with patch.object(client, "get_models", return_value=[]), \
             patch.object(client, "get_json", return_value=choices):
            self.assertEqual(client.inventory()["diffusers"], {"siglip-base-patch16-256-i18n"})

    def test_invalid_box_is_refused(self):
        self.settings["scene_faces"]["people"][0]["region"] = [.8, .2, .1, .5]
        self.assertIn("Left needs a visible", " ".join(self.plan().errors))

    def test_profiles_are_all_used_in_order_without_a_scene(self):
        self.settings.pop("scene_faces")
        self.lib.save("identities", [{"id": name, "name": name, "references": [path]}
                                      for name, path in zip(("left", "right"), self.photos)])
        self.settings["identities"] = ["right", "left"]
        p = self.plan()
        self.assertEqual(p.errors, [])
        self.assertEqual(p.images["face1"], self.photos[1])
        self.assertEqual(p.images["face2"], self.photos[0])

    def test_readiness_names_missing_identity_weights_and_node(self):
        _, missing = ig.missing_for(self.lib.get("models", "withanyone"), self.backend,
                                    FakeClient(self.backend).inventory(),
                                    FakeClient(self.backend).node_types())
        self.assertIn("withanyone.safetensors", [m["name"] for m in missing])
        self.assertIn("StudioWithAnyone", [m["name"] for m in missing])

    def test_job_uploads_both_people_and_never_redraws_the_result(self):
        studio = ig.Studio(root=self.tmp.name, client_factory=WithAnyoneClient)
        self.addCleanup(studio.close)
        job = ig.Job(self.settings, self.backend)
        with patch.object(studio, "_faces_into_picture", side_effect=AssertionError("PuLID")), \
             patch.object(studio, "_face_pass", side_effect=AssertionError("redraw")), \
             patch.object(studio, "_check_fix", side_effect=AssertionError("critic")):
            studio.run_job(job, lambda j: None)
        self.assertEqual(job.status, "complete", job.detail)
        self.assertEqual(studio.client(self.backend).uploads, self.photos)
        self.assertEqual(len(studio.client(self.backend).graphs), 1)
        self.assertEqual(job.record["references"]["face2"], self.photos[1])

    def test_selected_profiles_do_not_require_or_run_facefusion(self):
        import apps.image_studio.facefusion as ff
        self.settings.pop("scene_faces")
        self.lib.save("identities", [{"id": "left", "name": "Left",
                                      "references": self.photos, "face_swap": True}])
        self.settings["identities"] = ["left"]
        self.settings["hand_pass"] = True
        studio = ig.Studio(root=self.tmp.name, client_factory=WithAnyoneClient)
        self.addCleanup(studio.close)
        job = ig.Job(self.settings, self.backend)
        with patch.object(ff, "available", return_value=False), \
             patch.object(ff, "swap", side_effect=AssertionError("FaceFusion")), \
             patch.object(studio, "_finish_passes", side_effect=AssertionError("finishing")):
            preview = studio.preview(self.settings, self.backend)
            self.assertEqual(preview.errors, [])
            studio.run_job(job, lambda j: None)
        self.assertEqual(job.status, "complete", job.detail)
        self.assertEqual(studio.client(self.backend).uploads, self.photos)
        self.assertEqual(len(studio.client(self.backend).graphs), 1)
        self.assertFalse(job.record.get("facefusion"))
        graph = studio.client(self.backend).graphs[0]
        self.assertEqual(graph["40"]["inputs"]["references1"], ["face1_ref2_group", 0])
        self.assertNotIn("face2", graph["40"]["inputs"])
        self.assertEqual(job.record["references"]["face1_ref2"], self.photos[1])

    def test_two_profiles_keep_photo_groups_and_positions_separate(self):
        self.settings.pop("scene_faces")
        self.lib.save("identities", [
            {"id": "left", "name": "Left", "references": self.photos + self.photos[:1]},
            {"id": "right", "name": "Right", "references": self.photos[::-1]}])
        self.settings["identities"] = ["left", "right"]
        p = self.plan()
        self.assertEqual(p.errors, [])
        self.assertEqual(p.values["identity_reference_groups"],
                         [["face1", "face1_ref2"], ["face2", "face2_ref2"]])
        self.assertEqual(len(json.loads(p.values["identity_boxes"])), 2)
        graph = ig.fill(p.workflow, dict(p.values, **p.images))
        self.assertEqual(graph["40"]["inputs"]["references2"], ["face2_ref2_group", 0])
        self.assertEqual(graph["face2_ref2"]["inputs"]["image"], self.photos[0])

    def test_scene_uses_linked_library_even_when_legacy_references_disabled(self):
        self.lib.save("identities", [{"id": "left", "name": "Left",
            "references": self.photos, "use_references": False}])
        self.settings["scene_faces"]["people"] = [dict(
            self.settings["scene_faces"]["people"][0], identity="left", face="")]
        p = self.plan()
        self.assertEqual(p.errors, [])
        self.assertEqual(list(p.images.values()), self.photos)
        self.assertEqual(json.loads(p.values["identity_boxes"]), [[.1, .2, .4, .5]])

    def test_missing_secondary_reference_is_not_silently_dropped(self):
        self.settings["scene_faces"]["people"][0]["photos"] = ["missing.png"]
        self.assertIn("Left needs", " ".join(self.plan().errors))

    def test_old_backend_reports_required_node_update(self):
        self.settings["scene_faces"]["people"][0]["photos"] = self.photos
        client = WithAnyoneClient(self.backend)
        p = ig.compose(self.settings, self.lib, self.backend, client.inventory(),
                       nodes=client.node_types() - {"StudioWithAnyoneReferences"})
        self.assertIn("restart ComfyUI", " ".join(p.errors))

    def test_all_six_photos_feed_one_person(self):
        photos = self.photos[:]
        for i in range(4):
            path = os.path.join(self.tmp.name, "extra%d.png" % i)
            Path(path).write_bytes(PNG)
            photos.append(path)
        self.settings.pop("scene_faces")
        self.lib.save("identities", [{"id": "left", "name": "Left", "references": photos}])
        self.settings["identities"] = ["left"]
        p = self.plan()
        self.assertEqual(p.errors, [])
        self.assertEqual(list(p.images.values()), photos)
        self.assertEqual(len(json.loads(p.values["identity_boxes"])), 1)
        graph = ig.fill(p.workflow, dict(p.values, **p.images))
        self.assertEqual(graph["face1_ref6_group"]["inputs"]["previous"], ["face1_ref5_group", 0])
        self.assertEqual(graph["40"]["inputs"]["references1"], ["face1_ref6_group", 0])

    def test_rejected_multi_photo_blend_is_not_used_by_default(self):
        self.settings.pop("experimental_reference_groups")
        self.settings.pop("scene_faces")
        self.lib.save("identities", [{"id": "left", "name": "Left", "references": self.photos}])
        self.settings["identities"] = ["left"]
        client = WithAnyoneClient(self.backend)
        p = ig.compose(self.settings, self.lib, self.backend, client.inventory(),
                       nodes=client.node_types() - {"StudioWithAnyoneReferences"})
        self.assertEqual(p.errors, [])
        self.assertEqual(list(p.images.values()), self.photos[:1])
        self.assertEqual(self.lib.get("identities", "left")["references"], self.photos)
        self.assertIn("not blended", " ".join(p.notes))
        graph = ig.fill(p.workflow, dict(p.values, **p.images))
        self.assertNotIn("references1", graph["40"]["inputs"])

    def test_profile_pooling_selects_new_node_and_keeps_other_person_primary_only(self):
        self.settings.pop('scene_faces')
        self.lib.save('identities', [
            {'name': 'Left', 'references': self.photos, 'pool_photos': True},
            {'name': 'Right', 'references': self.photos[::-1]}])
        self.settings['identities'] = ['left', 'right']
        p = self.plan()
        self.assertEqual(p.errors, [])
        self.assertEqual(p.values['identity_reference_groups'], [['face1', 'face1_ref2'], ['face2']])
        graph = ig.fill(p.workflow, dict(p.values, **p.images))
        self.assertEqual(graph['40']['class_type'], 'StudioWithAnyonePooled')
        self.assertNotIn('references2', graph['40']['inputs'])
        client = WithAnyoneClient(self.backend)
        old = ig.compose(self.settings, self.lib, self.backend, client.inventory(),
                         nodes=client.node_types() - {'StudioWithAnyonePooled'})
        self.assertIn('restart ComfyUI to use photo pooling', ' '.join(old.errors))

    def test_scene_linked_profile_pooling_works_without_experiment_flag(self):
        self.settings.pop('experimental_reference_groups')
        self.lib.save('identities', [{'name': 'Left', 'references': self.photos, 'pool_photos': True}])
        self.settings['scene_faces']['people'][0]['identity'] = 'left'
        p = self.plan()
        self.assertEqual(p.errors, [])
        self.assertTrue(p.values['identity_pooling'])
        self.assertEqual(p.values['identity_reference_groups'][0], ['face1', 'face1_ref2'])


class ReferenceGroupingTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / "comfy_nodes/studio_withanyone/references.py"
        spec = importlib.util.spec_from_file_location("withanyone_reference_groups", path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def test_consensus_preserves_single_photo_and_raw_embedding_scale(self):
        consensus = self.module.identity_consensus
        self.assertEqual(consensus([[3, 4]]), [3, 4])
        self.assertEqual(consensus([[3, 4], [3, 4]]), [3, 4])
        # Magnitude must not bias the direction toward the second photo.
        pooled = consensus([[2, 0], [0, 4]])
        self.assertAlmostEqual(pooled[0], pooled[1])
        self.assertAlmostEqual(sum(x*x for x in pooled), 9)
        self.assertEqual(pooled, consensus([[0, 4], [2, 0]]))

    def test_consensus_refuses_invalid_or_cancelling_vectors(self):
        for rows in ([], [[]], [[0, 0]], [[1], [1, 2]],
                     [[float('nan')]], [[float('inf')]], [[1, 0], [-1, 0]]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.module.identity_consensus(rows)

    def test_different_sizes_share_person_region_without_averaging(self):
        photos = [SimpleNamespace(shape=(1, 256, 256, 3)),
                  SimpleNamespace(shape=(1, 1200, 800, 3)),
                  SimpleNamespace(shape=(1, 640, 960, 3))]
        boxes = [[.1, .2, .4, .5], [.6, .2, .9, .5]]
        refs = self.module.grouped_references(
            [photos[0], photos[2], None, None], [(photos[1],), None, None, None], boxes)
        self.assertIs(refs[0][0], photos[0])
        self.assertIs(refs[1][0], photos[1])
        self.assertEqual([r[1] for r in refs], [boxes[0], boxes[0], boxes[1]])
        self.assertEqual([r[2:] for r in refs], [(1, 1), (1, 2), (2, 1)])

    def test_refuses_unassigned_photos_batches_and_excess_references(self):
        photo = SimpleNamespace(shape=(1, 256, 256, 3))
        with self.assertRaisesRegex(ValueError, "primary"):
            self.module.grouped_references([photo, None], [None, (photo,)], [[0, 0, 1, 1]])
        with self.assertRaisesRegex(ValueError, "more than"):
            self.module.grouped_references([photo], [(photo,) * 8], [[0, 0, 1, 1]])
        with self.assertRaisesRegex(ValueError, "still image"):
            self.module.grouped_references([SimpleNamespace(shape=(2, 256, 256, 3))],
                                           [None], [[0, 0, 1, 1]])


if __name__ == "__main__":
    unittest.main()
