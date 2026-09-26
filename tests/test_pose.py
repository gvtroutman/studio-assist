"""The pose: OpenPose stick figures, and the picture the pose ControlNet reads."""

import math
import os
import struct
import sys
import tempfile
import unittest
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import studio_pose as sp  # noqa: E402


def pixels(png):
    """PNG bytes from studio_icons.png -> (width, height, RGBA rows)."""
    w, h = struct.unpack(">II", png[16:24])
    pos, data = 8, b""
    while pos < len(png):
        n = struct.unpack(">I", png[pos:pos + 4])[0]
        if png[pos + 4:pos + 8] == b"IDAT":
            data += png[pos + 8:pos + 8 + n]
        pos += 12 + n
    raw = zlib.decompress(data)
    stride = w * 4 + 1
    return w, h, [raw[y * stride + 1:(y + 1) * stride] for y in range(h)]


class TestPose(unittest.TestCase):
    def test_every_preset_has_every_joint_in_the_frame(self):
        for name in sp.PRESET_NAMES:
            pts = sp.preset(name, 832, 1216)
            self.assertEqual(len(pts), 18, name)
            for p in pts:
                if p is not None:
                    self.assertTrue(0 <= p[0] <= 1 and 0 <= p[1] <= 1, (name, p))
        self.assertIsNone(sp.preset("profile", 1024, 1024)[5])      # the far shoulder

    def test_a_preset_keeps_its_proportions_in_any_frame(self):
        tall, wide = sp.preset("standing", 800, 1200), sp.preset("standing", 1200, 800)

        def ratio(pts, w, h):              # shoulder width over neck-to-ankle, in px
            return ((pts[5][0] - pts[2][0]) * w) / ((pts[10][1] - pts[1][1]) * h)
        self.assertAlmostEqual(ratio(tall, 800, 1200), ratio(wide, 1200, 800), places=2)

    def test_mirror_swaps_sides_and_undoes_itself(self):
        pts = sp.preset("waving", 1024, 1024)
        m = sp.mirror(pts)
        self.assertAlmostEqual(m[4][0], 1 - pts[7][0], places=4)    # right wrist <- left
        self.assertEqual(m[4][1], pts[7][1])
        again = sp.mirror(m)
        for a, b in zip(again, pts):
            self.assertAlmostEqual(a[0], b[0], places=3)

    def test_refit_never_stretches_the_figure(self):
        pts = sp.preset("standing", 1024, 1024)
        out = sp.refit(pts, (1024, 1024), (832, 1216))
        # Same shape in pixels: the shoulder-to-ankle ratio holds.
        def shape(p, w, h):
            return ((p[5][0] - p[2][0]) * w) / ((p[10][1] - p[1][1]) * h)
        self.assertAlmostEqual(shape(pts, 1024, 1024), shape(out, 832, 1216), places=3)
        self.assertEqual(sp.refit(pts, (512, 512), (1024, 1024)), pts)   # same shape frame

    def test_render_draws_openpose_colours_on_black(self):
        pts = sp.preset("standing", 832, 1216)
        w, h, rows = pixels(sp.render(pts, 832, 1216))
        self.assertEqual(max(w, h), sp.RENDER_EDGE)
        self.assertAlmostEqual(w / h, 832 / 1216, places=2)
        self.assertEqual(rows[0][:4], b"\x00\x00\x00\xff")                 # a black corner
        nx, ny = int(pts[0][0] * w), int(pts[0][1] * h)                     # the nose is red
        self.assertEqual(tuple(rows[ny][nx * 4:nx * 4 + 3]), sp.COLOURS[0])

    def test_a_hidden_joint_and_its_limbs_are_not_drawn(self):
        pts = sp.preset("arms_up", 1024, 1024)
        pts[7] = None
        w, h, rows = pixels(sp.render(pts, 1024, 1024))
        wrist = sp.preset("arms_up", 1024, 1024)[7]
        x, y = int(wrist[0] * w), int(wrist[1] * h)
        self.assertEqual(rows[y][x * 4:x * 4 + 3], b"\x00\x00\x00")

    def test_a_seen_face_gets_its_landmark_dots(self):
        # Without them the ControlNet drew every figure from behind.
        pts = sp.preset("standing", 832, 1216)
        dots = sp.face_points(pts, 832, 1216)
        self.assertEqual(len(dots), 68)
        nose, eye_y = pts[0], pts[14][1]
        chin = dots[8]
        self.assertAlmostEqual(chin[0], nose[0], places=3)
        self.assertGreater(chin[1], nose[1])                  # below the nose, not above
        self.assertLess(chin[1], pts[1][1])                   # and above the neck
        self.assertTrue(all(abs(x - nose[0]) < 0.1 for x, _ in dots))
        w, h, rows = pixels(sp.render(pts, 832, 1216))
        x, y = int(chin[0] * w), int(chin[1] * h)
        self.assertEqual(tuple(rows[y][x * 4:x * 4 + 3]), (255, 255, 255))
        # A face turned away or in profile gets none.
        self.assertEqual(sp.face_points(sp.preset("profile", 1024, 1024), 1024, 1024), [])
        pts[15] = None
        self.assertEqual(sp.face_points(pts, 832, 1216), [])

    def test_save_names_the_picture_by_what_it_is(self):
        d = tempfile.mkdtemp()
        pts = sp.preset("waving", 1024, 1024)
        a = sp.save(pts, 1024, 1024, d)
        self.assertEqual(sp.save(pts, 1024, 1024, d), a)
        self.assertNotEqual(sp.save(sp.mirror(pts), 1024, 1024, d), a)
        with open(a, "rb") as f:
            self.assertEqual(f.read(8), b"\x89PNG\r\n\x1a\n")

    def test_dragging_a_joint_carries_what_hangs_off_it(self):
        self.assertEqual(sorted(sp.carried(3)), [3, 4])                  # elbow, wrist
        self.assertEqual(sorted(sp.carried(1)), list(range(18)))        # the neck: all
        self.assertEqual(sp.carried(4), [4])

    def test_clean_refuses_what_is_not_a_pose(self):
        self.assertIsNone(sp.clean([[0, 0]] * 3))
        self.assertIsNone(sp.clean([None] * 18))
        self.assertIsNone(sp.clean(["x"] * 18))
        self.assertEqual(sp.clean([[1, 2]] + [None] * 17)[0], [1.0, 2.0])

    def test_every_hand_shape_has_21_points_and_a_fist_is_closed(self):
        def reach(shape):                  # the fingertips' mean distance from the wrist
            pts = sp.hand_shape(shape)
            return sum(math.hypot(*pts[i]) for i in (8, 12, 16, 20)) / 4
        for name in sp.HAND_SHAPE_NAMES:
            self.assertEqual(len(sp.hand_shape(name)), 21, name)
        self.assertLess(reach("fist"), reach("relaxed"))
        self.assertLess(reach("relaxed"), reach("open"))
        point = sp.hand_shape("point")
        self.assertGreater(math.hypot(*point[8]), math.hypot(*point[12]))   # index out

    def test_hands_follow_the_forearm_and_turn_over(self):
        pts = sp.preset("standing", 1024, 1024)
        palm = sp.hand_points(pts, 1024, 1024, {"right": {"shape": "open"}})
        back = sp.hand_points(pts, 1024, 1024, {"right": {"shape": "open", "back": True}})
        self.assertEqual(set(palm), {"right", "left"})             # the left is relaxed
        wrist = pts[4]
        self.assertAlmostEqual(palm["right"][0][0], wrist[0], places=4)
        self.assertGreater(palm["right"][12][1], wrist[1])          # the arm hangs: fingers down
        # Palm to the viewer, a hanging right hand's thumb is on the outside
        # (the picture's left); turned over, it is on the inside.
        self.assertLess(palm["right"][4][0], wrist[0])
        self.assertGreater(back["right"][4][0], wrist[0])
        pts[3] = None                                               # no elbow, no hand
        self.assertNotIn("right", sp.hand_points(pts, 1024, 1024, sp.DEFAULT_HANDS))

    def test_hands_are_drawn_and_named_into_the_picture(self):
        d = tempfile.mkdtemp()
        pts = sp.preset("t_pose", 1024, 1024)
        bare = sp.save(pts, 1024, 1024, d)
        self.assertEqual(sp.save(pts, 1024, 1024, d, None), bare)    # old poses keep their name
        fist = sp.save(pts, 1024, 1024, d, {"right": {"shape": "fist"}})
        self.assertNotIn(fist, (bare, sp.save(pts, 1024, 1024, d, {"right": {"shape": "open"}})))
        w, h, rows = pixels(sp.render(pts, 1024, 1024, sp.DEFAULT_HANDS))
        tip = sp.hand_points(pts, 1, 1, sp.DEFAULT_HANDS)["right"][12]
        x, y = int(tip[0] * w), int(tip[1] * h)
        self.assertEqual(tuple(rows[y][x * 4:x * 4 + 3]), sp.HAND_JOINT)

    def test_the_prompt_names_only_shaped_hands_that_are_seen(self):
        pts = sp.preset("standing", 1024, 1024)
        self.assertEqual(sp.hands_text(pts, {"right": {"shape": "peace"}}),
                         "Right hand making a peace sign with two fingers.")
        self.assertEqual(sp.hands_text(pts, sp.DEFAULT_HANDS), "")
        pts[4] = None
        self.assertEqual(sp.hands_text(pts, {"right": {"shape": "fist"},
                                             "left": {"shape": "fist"}}),
                         "Left hand clenched in a fist.")
        self.assertEqual(sp.mirror_hands({"right": {"shape": "fist", "back": False},
                                          "left": {"shape": "open", "back": True}})["right"],
                         {"shape": "open", "back": True})

    def test_near_picks_the_closest_joint_within_reach(self):
        pts = sp.preset("standing", 1000, 1000)
        x, y = pts[4][0] * 500, pts[4][1] * 500
        self.assertEqual(sp.near(pts, x + 3, y, 500, 500, 10), 4)
        self.assertIsNone(sp.near(pts, 2, 2, 500, 500, 10))


if __name__ == "__main__":
    unittest.main()
