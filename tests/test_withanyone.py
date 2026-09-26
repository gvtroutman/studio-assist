"""Identity ordering, refusal and the complete job path, without GPU or network."""
import json
import os
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import studio_imagegen as ig
from test_imagegen import FakeClient, PNG


class WithAnyoneClient(FakeClient):
    def inventory(self):
        inv = super().inventory()
        inv["diffusion_models"].add("withanyone.safetensors")
        inv["diffusers"] = {"siglip-base-patch16-256-i18n"}
        return inv

    def node_types(self, **kwargs):
        return super().node_types() | {"StudioWithAnyone"}


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
                         "face_detail": True, "auto_refine": True,
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

    def test_missing_photo_refuses_instead_of_using_a_stranger(self):
        self.settings["scene_faces"]["people"][1]["face"] = "missing.png"
        self.assertIn("Right needs", " ".join(self.plan().errors))

    def test_too_many_people_are_not_truncated(self):
        self.settings["scene_faces"]["people"] *= 3
        self.assertIn("has 6", " ".join(self.plan().errors))

    def test_scene_builder_accepts_face_positions_without_controlnet(self):
        import studio_scene as sc
        from studio_scene_ui import SceneBuilder
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
             patch.object(studio, "_refine", side_effect=AssertionError("critic")):
            studio.run_job(job, lambda j: None)
        self.assertEqual(job.status, "complete", job.detail)
        self.assertEqual(studio.client(self.backend).uploads, self.photos)
        self.assertEqual(len(studio.client(self.backend).graphs), 1)
        self.assertEqual(job.record["references"]["face2"], self.photos[1])


if __name__ == "__main__":
    unittest.main()
