"""Geometry invariants independent of Tk, a model, or a running backend."""

import math
import random
import unittest
from unittest.mock import patch

from apps.image_studio.scene import scene as sc
from apps.image_studio.scene.distance import signed_distance, feather_distance


class TestDistanceFields(unittest.TestCase):
    def test_matches_brute_force_euclidean_distance(self):
        rng = random.Random(17)
        for w, h in ((7, 5), (1, 9), (9, 1)):
            mask = bytes(rng.randrange(2) for _ in range(w * h))
            field = signed_distance(mask, w, h)
            for i, value in enumerate(mask):
                x, y = i % w, i // w
                distances = [math.hypot(x - j % w, y - j // w)
                             for j, other in enumerate(mask) if other != value]
                expected = (min(distances) - 0.5) * (1 if value else -1)
                self.assertAlmostEqual(field[i], expected, places=5)

    def test_symmetry_saturation_and_monotone_feather(self):
        field = signed_distance(bytes([0] * 5 + [255] * 5), 10, 1)
        alpha = feather_distance(field, 3)
        self.assertEqual(list(alpha), sorted(alpha))
        for a, b in zip(alpha, reversed(alpha)):
            self.assertEqual(a + b, 255)
        self.assertEqual(feather_distance(field, 0), bytes([0] * 5 + [255] * 5))
        for value in (0, 255):
            uniform = signed_distance(bytes([value] * 12), 4, 3)
            self.assertEqual(set(uniform), {5 if value else -5})
        with self.assertRaises(ValueError):
            signed_distance(b"", 0, 1)
        with self.assertRaises(ValueError):
            signed_distance(b"\0", 2, 2)


class TestHeadGeometry(unittest.TestCase):
    def test_landmark_uses_same_deformation_as_skull(self):
        head = {"jaw_width": 0.9, "face_length": 0.8}
        p = (0.05, 0.025, 0.10)
        actual = sc.head_point(p, head)
        expected = sc.mq.head_warp(head)((p[0], p[1] - 0.11, p[2] - 0.01))
        self.assertEqual(actual, (expected[0], expected[1] + 0.11, expected[2] + 0.01))
        self.assertGreater(actual[0], p[0])
        self.assertLess(actual[1], p[1])
        self.assertEqual(sc.head_point(p), p)

    def test_camera_relative_angles_and_off_axis_gaze(self):
        cam = sc.Camera(sc.new_scene("")["camera"], 320, 240)
        centre = cam.target
        toward = sc.norm(sc.sub(cam.eye, centre))
        frame = tuple(
            tuple(v[i] for v in (cam.r, cam.u, toward)) for i in range(3))
        angles = sc.head_angles(cam, centre, frame)
        self.assertEqual((angles["yaw"], angles["pitch"], angles["roll"]), (0, 0, 0))
        self.assertEqual(angles["facing_cosine"], 1)
        # Camera level, centred on the head, gives controlled axis tests.
        cam = sc.Camera(dict(sc.new_scene("")["camera"], yaw=0, pitch=0), 320, 240)
        for yaw in (-70, 0, 55, 180):
            angles = sc.head_angles(cam, cam.target, sc.euler(yaw))
            self.assertAlmostEqual(abs(angles["yaw"]), abs(yaw), places=3)
            self.assertAlmostEqual(angles["facing_cosine"], math.cos(math.radians(yaw)), places=5)
        for yaw, pitch, roll in ((30, 25, -18), (-40, -35, 22)):
            angles = sc.head_angles(cam, cam.target, sc.euler(yaw, pitch, roll))
            self.assertAlmostEqual(angles["yaw"], yaw, places=3)
            self.assertAlmostEqual(angles["pitch"], -pitch, places=3)
            self.assertAlmostEqual(angles["roll"], roll, places=3)
        off_axis = sc.add(cam.target, (1, 0, 0))
        self.assertGreater(sc.head_angles(cam, off_axis, sc.IDENTITY)["yaw"], 0)

    def test_face_targets_carry_angles(self):
        scene = sc.new_scene("")
        person = sc.new_object("person")
        scene["objects"].append(person)
        target = sc.face_targets(scene)[0]
        self.assertEqual(set(target["head_pose"]), {"yaw", "pitch", "roll", "facing_cosine"})

    def test_pose_landmarks_project_the_deformed_head(self):
        scene = sc.new_scene("")
        person = sc.new_object("person")
        person["head"] = {"jaw_width": 0.9, "face_length": 0.8}
        scene["objects"].append(person)
        w, h = 96, 128
        cam = sc.Camera(scene["camera"], w, h)
        sk, scale, shift = sc.rigs(person)[0]
        hp, hm = sk["head"]
        u, v = sc.studio_pose.FACE[8]  # chin
        local = (u * sc.FACE_UNIT, sc.KP_HEAD[1][1] - v * sc.FACE_UNIT,
                 0.10 - 0.012 * u * u)
        local = sc.head_point(local, person["head"])
        p = sc.add(sc.mul(sc.add(hp, sc.apply(hm, local)), scale), shift)
        expected = cam.project(p)
        actual = sc.pose_figures(scene, w, h)[0]["face"][8]
        self.assertAlmostEqual(actual[0], expected[0] / w)
        self.assertAlmostEqual(actual[1], expected[1] / h)

    def test_character_distance_fields_preserve_visible_ownership(self):
        scene = sc.new_scene("")
        person = sc.new_object("person")
        person["character"] = "person"
        scene["objects"].append(person)
        with patch.object(sc, "frame_size", return_value=(64, 80)):
            w, h, hard = sc._character_mask_buffers(scene, 0)
            _, _, fields = sc.character_distance_fields(scene)
        self.assertEqual((w, h), (64, 80))
        self.assertEqual(feather_distance(fields["person"], 0), hard["person"])


class TestSurfaceOwnership(unittest.TestCase):
    def test_crossing_surfaces_follow_pixel_depth_not_draw_order(self):
        scene = sc.new_scene("")
        scene["objects"] = [sc.new_object("box"), sc.new_object("box")]
        scene["objects"][1]["id"] = "second"
        xy = [(0, 0), (4, 0), (4, 4), (0, 4)]
        # The sloping face is nearer on the left, farther on the right.
        slopes = [1, 4, 4, 1]
        polys = [sc.Poly(xy, (255, 0, 0), 2.5, scene["objects"][0]["id"], "body",
                         cam=[(0, 0, z) for z in slopes], nrm=(0, 0, 1)),
                 sc.Poly(xy, (0, 255, 0), 2, "second", "body",
                         cam=[(0, 0, 2)] * 4, nrm=(0, 1, 0))]
        depths = [max(1 - 0.1875 * (x + 0.5), 0.5) for y in range(4) for x in range(4)]
        for order in (polys, list(reversed(polys))):
            with patch.object(sc, "render", return_value=order), \
                    patch.object(sc, "depth_values", return_value=depths):
                inst, _, normals, _, _ = sc.id_render(scene, 4, 4)
            self.assertEqual(inst, bytes([1, 1, 1, 2] * 4))
            self.assertEqual(tuple(normals[:3]), (0, 0, 1))
            self.assertEqual(tuple(normals[9:12]), (0, 1, 0))


if __name__ == "__main__":
    unittest.main()
