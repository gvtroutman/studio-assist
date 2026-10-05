"""The drawing arithmetic in core.ui, without a window: colour blends, the
jumping dots' wave, and the browser tab's outline. `rounded` and `palette` are
covered in test_agent's TestPrefs; these are what nothing else checked."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import core.ui as ui


class FakeCanvas:
    """Keeps each polygon's points and options instead of drawing it."""

    def __init__(self):
        self.polygons = []

    def create_polygon(self, *args, **kw):
        flat = []
        for a in args:
            flat.extend(a if isinstance(a, (list, tuple)) else [a])
        self.polygons.append((list(zip(flat[0::2], flat[1::2])), kw))
        return len(self.polygons)


class Scaled:
    """Set ui.ROUNDING for one test and put it back afterwards."""

    def rounding(self, value):
        real = ui.ROUNDING
        self.addCleanup(setattr, ui, "ROUNDING", real)
        ui.ROUNDING = value


class TestFitChars(unittest.TestCase):
    """A name clipped to the pixels a dragged rail has for it."""

    class Mono:                       # ten pixels a character
        def measure(self, s):
            return 10 * len(s)

    def test_a_name_that_fits_is_whole_and_one_that_does_not_is_clipped(self):
        font = self.Mono()
        self.assertEqual(ui.fit_chars("Photoshop", font, 200), 9)
        n = ui.fit_chars("Adobe Premiere Pro 2026", font, 120)
        self.assertEqual(n, 12)
        self.assertLessEqual(font.measure(ui.clip("Adobe Premiere Pro 2026", n)), 120)

    def test_no_room_still_leaves_a_stub(self):
        self.assertEqual(ui.fit_chars("Illustrator", self.Mono(), 0), 4)


class TestBlend(unittest.TestCase):
    def test_the_ends_are_the_two_colours_not_both_the_second(self):
        # The bug the comment in `blend` describes: lazily built channels made
        # every blend come out as `b`, with no error anywhere.
        self.assertEqual(ui.blend("#102030", "#f0e0d0", 0.0), "#102030")
        self.assertEqual(ui.blend("#102030", "#f0e0d0", 1.0), "#f0e0d0")
        self.assertEqual(ui.blend("#000000", "#ffffff", 0.5), "#808080")

    def test_each_channel_moves_on_its_own(self):
        self.assertEqual(ui.blend("#ff0000", "#0000ff", 0.25), "#bf0040")

    def test_t_outside_zero_to_one_stops_at_the_ends(self):
        self.assertEqual(ui.blend("#102030", "#f0e0d0", -3), "#102030")
        self.assertEqual(ui.blend("#102030", "#f0e0d0", 7), "#f0e0d0")

    def test_upper_case_hex_blends_and_comes_out_lower(self):
        self.assertEqual(ui.blend("#FFFFFF", "#FFFFFF", 0.3), "#ffffff")


class TestJumps(unittest.TestCase):
    F, S = ui.JUMP_FRAMES, ui.JUMP_STAGGER

    def test_a_dot_sits_rises_to_the_top_and_comes_back(self):
        self.assertAlmostEqual(ui.jumps(0)[0], 0.0)
        self.assertAlmostEqual(ui.jumps(self.F)[0], 1.0)
        self.assertAlmostEqual(ui.jumps(2 * self.F)[0], 0.0)

    def test_the_fall_mirrors_the_rise(self):
        for k in range(self.F + 1):
            self.assertAlmostEqual(ui.jumps(k)[0], ui.jumps(2 * self.F - k)[0])

    def test_each_dot_is_the_one_before_it_a_stagger_later(self):
        for frame in range(-5, 3 * self.F):
            lift = ui.jumps(frame, n=4)
            for i in range(1, 4):
                self.assertAlmostEqual(lift[i], ui.jumps(frame - i * self.S, n=4)[0])

    def test_every_height_is_between_the_line_and_the_top(self):
        for frame in range(-20, 40):
            for up in ui.jumps(frame, n=5):
                self.assertTrue(-1e-9 <= up <= 1 + 1e-9, (frame, up))

    def test_metrics_grow_with_the_font_and_keep_a_small_one_visible(self):
        class Font:
            def __init__(self, ascent):
                self.ascent = ascent

            def metrics(self, what):
                return self.ascent

        size, stride, rise = ui.jump_metrics(Font(2))
        self.assertEqual(size, 3)                       # never under 3 px
        self.assertGreater(stride, size)                # the dots never touch
        big = ui.jump_metrics(Font(40))
        self.assertGreater(big[0], size)
        self.assertEqual(ui.jump_width(Font(40)), 2 * big[1] + big[0])


class TestBrowserTab(Scaled, unittest.TestCase):
    def test_the_feet_reach_the_flare_past_each_side(self):
        self.rounding(1.0)
        c = FakeCanvas()
        ui.browser_tab(c, 10, 0, 110, 30, 8, fill="#123456")
        pts, kw = c.polygons[0]
        flare = ui.tab_flare(8)
        self.assertEqual(kw, {"fill": "#123456"})
        self.assertAlmostEqual(min(x for x, _ in pts), 10 - flare)
        self.assertAlmostEqual(max(x for x, _ in pts), 110 + flare)
        self.assertAlmostEqual(min(y for _, y in pts), 0)
        self.assertAlmostEqual(max(y for _, y in pts), 30)
        # It starts and ends on the bottom line, out at the feet's tips.
        for (x, y), want in ((pts[0], 10 - flare), (pts[-1], 110 + flare)):
            self.assertAlmostEqual(x, want)
            self.assertAlmostEqual(y, 30)

    def test_a_corner_bigger_than_the_tab_stops_at_half_its_height(self):
        self.rounding(1.0)
        c = FakeCanvas()
        ui.browser_tab(c, 10, 0, 110, 30, 40)
        pts, _ = c.polygons[0]
        self.assertAlmostEqual(min(x for x, _ in pts), 10 - 15)
        self.assertAlmostEqual(max(x for x, _ in pts), 110 + 15)

    def test_square_corners_draw_a_plain_rectangle(self):
        self.rounding(0.0)
        c = FakeCanvas()
        ui.browser_tab(c, 10, 0, 110, 30, 8)
        self.assertEqual(c.polygons[0][0], [(10, 0), (110, 0), (110, 30), (10, 30)])
        self.assertEqual(ui.tab_flare(8), 0)

    def test_the_flare_follows_the_corners_setting(self):
        self.rounding(1.5)
        self.assertEqual(ui.tab_flare(8), 12)


class TestLifted(Scaled, unittest.TestCase):
    def test_the_surface_is_drawn_last_over_two_darker_shadows(self):
        self.rounding(1.0)
        c = FakeCanvas()
        ui.lifted(c, 100, 40, 6, "#eeeeee", "#808080", tags="card")
        self.assertEqual(len(c.polygons), 3)
        shadows, (surface, top) = c.polygons[:2], c.polygons[2]
        self.assertEqual(top["fill"], "#eeeeee")
        self.assertEqual(max(y for _, y in surface), 38)   # 2 px left for the shadow
        for pts, kw in shadows:
            self.assertLess(kw["fill"], "#808080")          # darker than what it sits on
            self.assertEqual(kw["tags"], "card")
        self.assertEqual(max(y for _, y in shadows[0][0]), 40)


if __name__ == "__main__":
    unittest.main()
