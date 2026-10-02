"""Surface-map geometry and identity attention; no backend or GUI."""
import math
import os
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from apps.image_studio.scene import scene as sc
from apps.image_studio.scene.surface import analyse, identity_mask
import apps.image_studio.imagegen as ig
from core.icons import png_to_rgba


class TestSurfaceSignals(unittest.TestCase):
    def signals(self, inst, parts, normals, world, w, h):
        return analyse(bytes(inst), bytes(parts), normals, world, w, h,
                       (0, 0, 0), (0, 0, 1), (0, 0, -1))

    def test_perspective_incidence_and_light_are_distinct(self):
        maps = self.signals([1, 1, 1], [1] * 3, [0, 0, -2] * 3,
                            [0, 0, 2, 2, 0, 2, 20, 0, 2], 3, 1)
        for actual, expected in zip(maps["facing"], (1, 1 / math.sqrt(2), 1 / math.sqrt(101))):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(list(maps["lighting"]), [1] * 3)
        self.assertEqual(list(maps["normal_edge"]), [0] * 3)
        self.assertEqual(list(maps["depth_edge"]), [0] * 3)
        self.assertGreater(maps["grazing"][2], 0.4)

    def test_crease_does_not_imply_depth_or_ownership_boundary(self):
        maps = self.signals([1, 1], [1, 1], [0, 0, -1, 1, 0, 0],
                            [0, 0, 2, 0.1, 0, 2], 2, 1)
        for value in maps["normal_edge"]:
            self.assertAlmostEqual(value, math.sqrt(2) / 2)
        self.assertEqual(list(maps["depth_edge"]), [0, 0])
        self.assertEqual(list(maps["instance_edge"]), [0, 0])
        self.assertEqual(list(maps["part_edge"]), [0, 0])

    def test_occlusion_and_parts_and_background_are_separate(self):
        maps = self.signals([1, 1, 2, 0], [1, 2, 1, 0], [0, 0, -1] * 3 + [0] * 3,
                            [0, 0, 2, 0.1, 0, 2, 0.2, 0, 4, 0, 0, 0], 4, 1)
        self.assertEqual(list(maps["valid"]), [1, 1, 1, 0])
        self.assertEqual(list(maps["part_edge"]), [1, 1, 0, 0])
        self.assertEqual(list(maps["instance_edge"]), [0, 1, 1, 1])
        self.assertEqual(list(maps["depth_edge"]), [0, 0.5, 0.5, 0])
        self.assertEqual(maps["normal_edge"][3], 0)
        self.assertEqual(maps["grazing"][3], 0)

    def test_edges_never_wrap_rows(self):
        maps = self.signals([1, 1, 2, 2], [1] * 4, [0, 0, -1] * 4,
                            [0, 0, 2, 10, 0, 2, 0, 1, 2, 10, 1, 2], 2, 2)
        self.assertEqual(list(maps["instance_edge"]), [1] * 4)
        self.assertEqual(list(maps["world_edge"]), [10] * 4)
        with self.assertRaises(ValueError):
            self.signals([1], [1], [], [], 1, 1)

    def test_mask_feathers_inward_and_reduces_grazing_confidence(self):
        inst = bytes([2] * 7 + [2, 1, 1, 1, 1, 1, 2] * 5 + [2] * 7)
        parts = bytes(1 if v == 1 else 2 for v in inst)
        signals = {"valid": bytes([1] * 49), "facing": [1.0] * 49, "grazing": [0.0] * 49}
        alpha = identity_mask(inst, parts, signals, 7, 7, 1, {1}, [0, 0, 1, 1])
        self.assertEqual(alpha[24], 255)
        self.assertGreater(alpha[8], 0)
        self.assertLess(alpha[8], alpha[24])
        self.assertTrue(all(alpha[i] == 0 for i, owner in enumerate(inst) if owner != 1))
        signals["facing"][24] = 0.1
        signals["grazing"][24] = 0.5
        grazing = identity_mask(inst, parts, signals, 7, 7, 1, {1}, [0, 0, 1, 1])
        self.assertLess(grazing[24], alpha[24])
        signals["facing"][24] = -1
        self.assertEqual(identity_mask(inst, parts, signals, 7, 7, 1, {1}, [0, 0, 1, 1])[24], 0)

    def test_real_scene_mask_stays_on_its_visible_head(self):
        scene = sc.new_scene("")
        scene["objects"].append(sc.new_object("person"))
        targets = sc.face_targets(scene)
        maps = sc.surface_render(scene, 192, 256)
        with patch.object(sc, "surface_render", return_value=maps):
            data = sc.identity_masks(scene, targets)[targets[0]["id"]]
        self.assertIsNotNone(data)
        pixels, w, h = png_to_rgba(data)
        self.assertTrue(any(pixels[0::4]))
        for i, alpha in enumerate(pixels[0::4]):
            if alpha:
                sid = str(maps["instance"][i])
                self.assertEqual(maps["sidecar"]["instances"][sid]["owner"], targets[0]["id"])
                self.assertEqual(maps["sidecar"]["parts"][sid][str(maps["part"][i])], "head")
        targets[0]["head_pose"]["facing_cosine"] = -1
        with patch.object(sc, "surface_render", return_value=maps):
            self.assertIsNone(sc.identity_masks(scene, targets)[targets[0]["id"]])

    def test_occluded_head_has_no_rectangle_fallback(self):
        scene = sc.new_scene("")
        scene["objects"].append(sc.new_object("person"))
        targets = sc.face_targets(scene)
        hidden = dict(instance=b"\0", part=b"\0", sidecar={"instances": {}, "parts": {}})
        with patch.object(sc, "surface_render", return_value=hidden) as render:
            masks = sc.identity_masks(scene, targets * 2)
        render.assert_called_once()
        self.assertIsNone(masks[targets[0]["id"]])

    def test_real_profile_and_occlusion_without_buffer_stubs(self):
        scene = sc.new_scene("")
        person = sc.new_object("person")
        scene["objects"].append(person)
        for yaw in (0, -90, 90, 180):
            with self.subTest(yaw=yaw):
                person["rotation"][0] = yaw
                targets = sc.face_targets(scene)
                data = sc.identity_masks(scene, targets)[person["id"]]
                if yaw == 180:
                    self.assertIsNone(data)
                else:
                    self.assertIsNotNone(data)
                    pixels, w, h = png_to_rgba(data)
                    self.assertEqual(max(w, h), 256)
                    self.assertTrue(any(pixels[0::4]))
        person["rotation"][0] = 0
        blocker = sc.new_object("box", scene["objects"])
        blocker.update(position=[0, 0, 1], scale=[3, 3, 0.5])
        scene["objects"].append(blocker)
        targets = sc.face_targets(scene)
        self.assertIsNone(sc.identity_masks(scene, targets)[person["id"]])

    def test_camera_translation_and_rotation_preserve_signals(self):
        inst, part = bytes([1, 1]), bytes([1, 1])
        normals, world = [0, 0, -1] * 2, [0, 0, 2, 1, 0, 2]
        original = analyse(inst, part, normals, world, 2, 1,
                           (0, 0, 0), (0, 0, 1), (0, 0, -1))
        # Rotate 90 degrees around Y, then translate both camera and scene.
        moved = analyse(inst, part, [-1, 0, 0] * 2, [12, 20, 30, 12, 20, 29], 2, 1,
                        (10, 20, 30), (1, 0, 0), (-1, 0, 0))
        for key in original:
            for before, after in zip(original[key], moved[key]):
                self.assertAlmostEqual(before, after)


class TestIdentityConsumer(unittest.TestCase):
    def test_real_scene_builds_a_masked_pulid_graph_and_rear_head_builds_none(self):
        with tempfile.TemporaryDirectory() as folder:
            studio = ig.Studio.__new__(ig.Studio)
            studio.lib = SimpleNamespace(root=folder)
            studio._pulid = Mock(return_value=("pulid.safetensors", ""))
            layout = sc.new_scene("")
            person = sc.new_object("person")
            person["face"] = os.path.join(folder, "reference.png")
            with open(person["face"], "wb") as f:
                f.write(sc.rgb_png(bytes([128, 128, 128]), 1, 1))
            layout["objects"].append(person)
            client = Mock()
            client.upload_image.side_effect = lambda path: path
            for yaw in (0, 180):
                person["rotation"][0] = yaw
                targets = sc.face_targets(layout)
                job = SimpleNamespace(settings={"scene_layout": layout,
                                                "scene_faces": {"people": targets}},
                                      cancel=threading.Event())
                plan = SimpleNamespace(notes=[], warnings=[])
                graph = {"base": {"class_type": "ModelLoader", "inputs": {}},
                         "sample": {"class_type": "KSampler", "inputs": {"model": ["base", 0]}}}
                self.assertTrue(studio._faces_into_picture(job, client, plan, graph, set(), 768, 1024))
                self.assertEqual(plan.warnings, [])
                if yaw == 180:
                    self.assertEqual(graph["sample"]["inputs"]["model"], ["base", 0])
                    self.assertNotIn("pb_1", graph)
                else:
                    self.assertEqual(graph["sample"]["inputs"]["model"], ["pb_1", 0])
                    self.assertEqual(graph["pb_1"]["inputs"]["attn_mask"], ["pb_1k", 0])
                    self.assertEqual(graph["pb_1k"]["inputs"]["channel"], "red")
                    with open(graph["pb_1m"]["inputs"]["image"], "rb") as f:
                        pixels, w, h = png_to_rgba(f.read())
                    self.assertEqual(max(w, h), 256)
                    self.assertTrue(any(pixels[0::4]))
                    self.assertTrue(any(0 < v < 255 for v in pixels[0::4]))

    def test_surface_mask_is_uploaded_and_hidden_head_is_skipped(self):
        with tempfile.TemporaryDirectory() as folder:
            studio = ig.Studio.__new__(ig.Studio)
            studio.lib = SimpleNamespace(root=folder)
            studio._pulid = Mock(return_value=("pulid.safetensors", ""))
            people = [dict(id="visible", name="Visible", face="face.png", region=[0.1, 0.1, 0.5, 0.5]),
                      dict(id="hidden", name="Hidden", face="hidden.png", region=[0.5, 0.1, 0.9, 0.5])]
            job = SimpleNamespace(settings={"scene_layout": sc.new_scene(""),
                                            "scene_faces": {"people": people}}, cancel=threading.Event())
            plan = SimpleNamespace(notes=[], warnings=[])
            client = Mock()
            client.upload_image.side_effect = lambda path: path
            mask = sc.rgb_png(bytes([128] * 12), 2, 2)
            with patch.object(sc, "identity_masks", return_value={"visible": mask, "hidden": None}), \
                    patch.object(ig, "add_pulid") as add:
                self.assertTrue(studio._faces_into_picture(job, client, plan, {}, set(), 800, 1200))
                faces = add.call_args.args[2]
                self.assertEqual(len(faces), 1)
                with open(faces[0][1], "rb") as f:
                    self.assertEqual(f.read(), mask)
            self.assertTrue(any("Hidden: no visible" in note for note in plan.notes))
            self.assertFalse(any(call.args[0] == "hidden.png" for call in client.upload_image.call_args_list))

    def test_scene_full_frame_region_still_uses_surface_mask(self):
        with tempfile.TemporaryDirectory() as folder:
            studio = ig.Studio.__new__(ig.Studio)
            studio.lib = SimpleNamespace(root=folder)
            studio._pulid = Mock(return_value=("pulid.safetensors", ""))
            person = dict(id="person", name="Person", face="face.png", region=ig.WHOLE_FRAME)
            job = SimpleNamespace(settings={"scene_layout": sc.new_scene(""),
                                            "scene_faces": {"people": [person]}}, cancel=threading.Event())
            plan, client = SimpleNamespace(notes=[], warnings=[]), Mock()
            client.upload_image.side_effect = lambda path: path
            with patch.object(sc, "identity_masks", return_value={"person": b"surface mask"}), \
                    patch.object(ig, "add_pulid") as add:
                studio._faces_into_picture(job, client, plan, {}, set(), 800, 1200)
            self.assertIsNotNone(add.call_args.args[2][0][1])


if __name__ == "__main__":
    unittest.main()
