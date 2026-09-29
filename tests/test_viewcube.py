"""The view cube for Angles: its parts, its turning, and clicks on it."""
import unittest
from collections import Counter

import apps.image_studio.blend as sb
import apps.image_studio.viewcube as vc
from core.ui import DARK


class TestGeometry(unittest.TestCase):
    def test_every_part_of_the_cube_is_a_view(self):
        per = Counter(k for k, _, _ in vc.CELLS)
        self.assertEqual(len(vc.CELLS), 54)
        self.assertEqual(set(per), set(sb.VIEW_KEYS))
        for key, n in per.items():                  # face 1, edge 2, corner 3
            self.assertEqual(n, sum(1 for v in key if v))

    def test_a_cell_lies_on_its_face_and_leans_its_way(self):
        for key, normal, corners in vc.CELLS:
            axis = [abs(v) for v in normal].index(1)
            self.assertTrue(all(p[axis] == normal[axis] for p in corners))
            self.assertEqual(key[axis], normal[axis])

    def test_seen_from_in_front_their_right_is_on_the_left(self):
        d, r, up = vc.basis(0, 0)
        self.assertAlmostEqual(vc.dot(d, (0, 0, 1)), 1)
        self.assertAlmostEqual(vc.dot(r, (1, 0, 0)), -1)
        self.assertAlmostEqual(vc.dot(up, (0, 1, 0)), 1)
        self.assertEqual({n for _, n, _, _ in vc.facing(0, 0)}, {(0, 0, 1)})

    def test_home_shows_front_their_right_and_top(self):
        faces = {n for _, n, _, _ in vc.facing(*vc.HOME)}
        self.assertEqual(faces, {(0, 0, 1), (1, 0, 0), (0, 1, 0)})


class TestWidget(unittest.TestCase):
    def setUp(self):
        import tkinter as tk
        try:
            self.root = tk.Tk()
        except tk.TclError:
            self.skipTest("no display")
        self.root.withdraw()
        self.heard = []
        self.cube = vc.ViewCube(self.root, DARK, 200, ["front", "nonsense"],
                                self.heard.append)
        self.cube.pack()
        self.root.update_idletasks()

    def tearDown(self):
        self.root.destroy()

    def click(self, x, y, to=None):
        ev = type("E", (), {"x": x, "y": y})
        self.cube._press(ev)
        if to:
            self.cube._drag(type("E", (), {"x": to[0], "y": to[1]}))
        self.cube._release(type("E", (), {"x": (to or (x, y))[0], "y": (to or (x, y))[1]}))

    def test_unknown_names_are_dropped(self):
        self.assertEqual(self.cube.chosen, ["front"])

    def test_a_click_on_a_face_middle_picks_and_drops_that_view(self):
        self.cube.yaw, self.cube.pitch = 0, 0
        self.cube.draw()
        self.click(100, 100)
        self.assertEqual(self.heard[-1], [])            # front was picked: dropped
        self.click(100, 100)
        self.assertEqual(self.heard[-1], ["front"])

    def test_a_corner_of_the_front_is_the_corner_view(self):
        self.cube.yaw, self.cube.pitch = 0, 0
        self.cube.draw()
        # upper left on screen, seen from in front, is their right, up
        self.assertEqual(self.cube.key_at(100 - 50, 100 - 50), (1, 1, 1))

    def test_a_drag_turns_the_cube_and_picks_nothing(self):
        yaw = self.cube.yaw
        self.click(100, 100, to=(160, 100))
        self.assertNotEqual(self.cube.yaw, yaw)
        self.assertEqual(self.heard, [])


if __name__ == "__main__":
    unittest.main()
