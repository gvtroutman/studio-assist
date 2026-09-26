"""The Scene Builder: the rig and its poses, the camera, the frame rendered to
a PNG, scene files, the words sent with the frame, and the window driven in
process through the Image Studio against a fake ComfyUI. No network, no GPU."""

import json
import math
import os
import struct
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import studio_icons  # noqa: E402
import studio_imagegen as ig  # noqa: E402
import studio_scene as sc  # noqa: E402
from test_imagegen import FLUX_FILES, FakeClient, TempStudioMixin, _headless  # noqa: E402


def staged(*assets):
    s = sc.new_scene("A steel workshop, overcast light")
    for a in assets:
        s["objects"].append(sc.new_object(a, s["objects"]))
    return s


def png_size(data):
    assert data[:8] == studio_icons.PNG_MAGIC
    return struct.unpack(">II", data[16:24])


def two_tone(path, a, b, n=64):
    """A PNG split down the middle: `a` on the left, `b` on the right."""
    px = bytearray()
    for _y in range(n):
        for x in range(n):
            px += bytes(a if x < n // 2 else b) + b"\xff"
    with open(path, "wb") as f:
        f.write(studio_icons.png(bytes(px), n, n))
    return path


def pixel(scene, x, y):
    w, h = sc.frame_size(scene)
    rgb = sc.rasterise(sc.render(scene, w, h), w, h)
    i = (int(y) * w + int(x)) * 3
    return tuple(rgb[i:i + 3])


class TestRig(unittest.TestCase):
    def test_every_pose_stands_on_its_floor(self):
        for key, _, _ in sc.POSES:
            with self.subTest(pose=key):
                o = sc.new_object("person")
                o["pose"] = {"preset": key, "controls": sc.pose_controls(key)}
                lo, hi = sc.bounds(o)
                self.assertAlmostEqual(lo[1], 0.0, places=9)
                self.assertTrue(1.0 < hi[1] < 2.3, hi)
                o["position"] = [0, 0.8, 0]                    # standing on a platform
                self.assertAlmostEqual(sc.bounds(o)[0][1], 0.8, places=9)

    def test_a_crouch_drops_the_hips_and_a_reach_raises_the_hand(self):
        stand = sc.skeleton(sc.pose_controls("standing"))
        crouch = sc.skeleton(sc.pose_controls("crouching"))
        reach = sc.skeleton(sc.pose_controls("reaching"))
        foot = lambda sk: min(sk["ankle_l"][0][1], sk["ankle_r"][0][1])   # noqa: E731
        self.assertLess(crouch["pelvis"][0][1] - foot(crouch),
                        stand["pelvis"][0][1] - foot(stand) - 0.3)
        self.assertGreater(reach["wrist_r"][0][1], reach["head"][0][1])
        self.assertLess(stand["wrist_r"][0][1], stand["pelvis"][0][1])

    def test_the_controls_move_the_parts_they_name(self):
        c = sc.pose_controls("standing")
        c["arm_l_raise"] = 90                                   # forward is +Z
        sk = sc.skeleton(c)
        self.assertGreater(sk["wrist_l"][0][2], 0.4)
        c["head_turn"] = 90                                     # to their left, +X
        nose = sc.apply(sc.skeleton(c)["head"][1], (0, 0, 1))
        self.assertGreater(nose[0], 0.9)
        c["leg_r_step"] = 90
        self.assertGreater(sc.skeleton(c)["ankle_r"][0][2], 0.3)

    def test_every_control_belongs_to_a_part_the_viewport_can_pick(self):
        o = sc.new_object("person")
        parts = {part for part, _ in sc.object_pieces(o)}
        self.assertEqual(parts, {p for p, _ in sc.PARTS})
        self.assertEqual({c[0] for c in sc.CONTROLS}, parts)


class TestLookAt(unittest.TestCase):
    def facing(self, obj):
        sk = sc.rigs(obj)[0][0]
        way = sc.norm(sc.sub(obj["look_at"], sc.eye_point(obj)))
        return sc.dot(way, sc.column(sk["head"][1], 2))

    def test_the_head_turns_to_the_point_and_follows_a_move(self):
        p = sc.new_object("person")
        p["rotation"] = [30.0, 0.0, 0.0]
        p["look_at"] = [2.0, 1.2, 3.0]
        self.assertTrue(sc.aim_head(p))
        self.assertGreater(self.facing(p), 0.999)
        p["position"] = [1.5, 0.0, 0.0]
        sc.aim_heads({"objects": [p]})
        self.assertGreater(self.facing(p), 0.999)
        p["look_at"] = [0.0, 1.6, -3.0]                        # behind: as far as it goes
        sc.aim_head(p)
        self.assertEqual(abs(p["pose"]["controls"]["head_turn"]), 80)

    def test_a_crowd_or_a_person_without_a_point_is_left_alone(self):
        self.assertFalse(sc.aim_head(sc.new_object("person")))
        self.assertFalse(sc.aim_head(dict(sc.new_object("crowd"), look_at=[0, 1, 1])))

    def test_the_point_is_saved_and_named_in_history(self):
        p = sc.new_object("person")
        self.assertNotIn("look_at", sc.clean_object(p))
        p2 = dict(p, look_at=[1, 2, 3])
        self.assertEqual(sc.clean_object(p2)["look_at"], [1.0, 2.0, 3.0])
        a, b = {"objects": [p]}, {"objects": [p2]}
        self.assertEqual(sc.change_label(a, b), "Point Person's eyes")


class TestBodyAndClothes(unittest.TestCase):
    def person(self, **look):
        o = sc.new_object("person")
        o["look"] = look
        return o

    def width(self, o):
        lo, hi = sc.bounds(o)
        return hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]

    def test_the_sliders_size_the_mannequin_and_it_still_stands_on_its_floor(self):
        rest = self.width(self.person())
        heavy = self.width(self.person(weight=3))
        thin = self.width(self.person(weight=-3))
        self.assertGreater(heavy[0], rest[0] * 1.1)
        self.assertGreater(heavy[2], rest[2] * 1.2)             # the belly
        self.assertLess(thin[0], rest[0])
        self.assertGreater(self.width(self.person(muscle=3))[0], rest[0])
        self.assertAlmostEqual(self.width(self.person(stature=3))[1], rest[1] * 1.12,
                               places=6)
        self.assertLess(self.width(self.person(stature=-3))[1], rest[1])
        for look in ({"weight": 3}, {"stature": -3, "weight": -3}, {"muscle": 3}):
            self.assertAlmostEqual(sc.bounds(self.person(**look))[0][1], 0.0, places=9)

    def test_a_body_type_word_counts_as_slider_steps(self):
        self.assertGreater(sc.body_shape({"build": "stocky"})["fat"], 1)
        self.assertLess(sc.body_shape({"build": "petite"})["height"], 1)
        self.assertEqual(sc.body_shape({"build": "average"}), sc.body_shape({}))
        # Words are matched whole: "broad-shouldered" is not "broad", and a
        # slider already at the end is not pushed past it.
        self.assertEqual(sc.body_shape({"weight": 3, "build": "obese"})["fat"],
                         sc.body_shape({"weight": 3})["fat"])

    def test_no_clothes_is_the_plain_mannequin_in_its_colour(self):
        o = self.person(weight=2)
        own = sc.hex_rgb(o["colour"])
        # Skin is the object's colour; the eyes are their own.
        self.assertEqual({rgb for _, _, rgb in sc.painted_pieces(o)},
                         {own, sc.EYE_WHITE, sc.EYE_DARK})

    def test_clothes_colour_what_they_cover(self):
        o = self.person(top="white t-shirt", bottom="blue jeans", footwear="black boots")
        colours = {rgb for _, _, rgb in sc.painted_pieces(o)}
        for words, slot in (("white t-shirt", "top"), ("blue jeans", "bottom"),
                            ("black boots", "footwear")):
            self.assertIn(sc.cloth_colour(words, slot), colours)
        self.assertIn(sc.hex_rgb(o["colour"]), colours)         # the forearms, the head
        dressed = sc.outfit(o["look"])
        self.assertEqual(dressed["regions"]["upper_arm"], sc.cloth_colour("white", "top"))
        self.assertNotIn("forearm", dressed["regions"])         # short sleeves
        self.assertIsNotNone(dressed["boots"])
        # A dress and a long coat hang below the waist; trousers do not.
        self.assertEqual(dressed["hems"], [])
        self.assertEqual(len(sc.outfit({"top": "evening gown"})["hems"]), 1)
        self.assertEqual(len(sc.outfit({"outerwear": "trench coat"})["hems"]), 1)
        gown = sc.outfit({"top": "evening gown"})["hems"][0][0]
        self.assertGreater(gown, sc.outfit({"top": "summer dress"})["hems"][0][0])
        sleeved = sc.outfit({"top": "long-sleeve summer dress"})    # long sleeves, not long
        self.assertIn("forearm", sleeved["regions"])
        self.assertLess(sleeved["hems"][0][0], gown)

    def test_hats_and_glasses_come_from_the_accessories(self):
        dressed = sc.outfit({"accessories": "sunglasses, red baseball cap, wristwatch"})
        self.assertEqual(dressed["hat"], ("cap", sc.cloth_colour("red", "hat")))
        self.assertEqual(dressed["glasses"][0], "sunglasses")      # not plain "glasses"
        self.assertEqual(sc.outfit({"accessories": "hard hat"})["hat"][0], "hard hat")
        self.assertEqual(sc.outfit({"accessories": "hard hat"})["hat"][1],
                         sc.hex_rgb("#e2c23c"))                      # site yellow
        self.assertEqual(sc.outfit({"accessories": "white hard hat"})["hat"][1],
                         sc.cloth_colour("white", "hat"))
        self.assertEqual(sc.outfit({"accessories": "round glasses"})["glasses"][0], "glasses")
        bare = sc.outfit({"accessories": "necklace, headphones, tote bag"})
        self.assertEqual((bare["hat"], bare["glasses"]), (None, None))

        # They are on the head: taller with a top hat, clicked as the head,
        # and the person still stands on the floor.
        plain = self.person()
        hatted = self.person(accessories="top hat, glasses")
        self.assertGreater(self.width(hatted)[1], self.width(plain)[1] + 0.1)
        self.assertAlmostEqual(sc.bounds(hatted)[0][1], 0.0, places=9)
        count = lambda o, part=None: len([1 for p, _, _ in sc.painted_pieces(o)   # noqa
                                          if part in (None, p)])
        added = count(hatted) - count(plain)
        self.assertGreater(added, 0)
        self.assertEqual(count(hatted, "head") - count(plain, "head"), added)

    def test_each_kind_of_shoe_has_its_shape(self):
        kind = lambda words: sc.outfit({"footwear": words})["shoes"][0]   # noqa: E731
        self.assertEqual(kind("black high heels"), "heels")
        self.assertEqual(kind("white sneakers"), "sneakers")
        self.assertEqual(kind("running shoes"), "sneakers")
        self.assertEqual(kind("ankle boots"), "boots")
        self.assertEqual(kind("sandals"), "sandals")
        self.assertEqual(kind("loafers"), "shoes")
        self.assertIsNone(sc.outfit({"footwear": "barefoot"})["shoes"])
        # A sandal shows the foot; a shoe covers it.
        self.assertNotIn("foot", sc.outfit({"footwear": "brown sandals"})["regions"])
        self.assertIn("foot", sc.outfit({"footwear": "loafers"})["regions"])
        # A sole lifts the person, a heel more; they still stand on the floor.
        tall = lambda words: sc.bounds(self.person(footwear=words))[1][1]  # noqa: E731
        self.assertGreater(tall("loafers"), tall(""))
        self.assertGreater(tall("high heels"), tall("sneakers") + 0.03)
        self.assertAlmostEqual(sc.bounds(self.person(footwear="heels"))[0][1], 0.0, places=9)
        # A boot's shaft replaces the shin's lower part, and trousers hide it.
        count = lambda o: len(sc.painted_pieces(o))                       # noqa: E731
        booted = self.person(footwear="leather boots", bottom="shorts")
        shod = self.person(footwear="loafers", bottom="shorts")
        self.assertEqual(count(booted), count(shod) + 2)
        self.assertEqual(count(self.person(footwear="leather boots", bottom="jeans")),
                         count(self.person(footwear="loafers", bottom="jeans")))

    def test_hair_comes_from_the_hair_section(self):
        self.assertIsNone(sc.hairdo({}))
        self.assertIsNone(sc.hairdo({"hair": "black", "hair_style": "bald"}))
        long = sc.hairdo({"hair": "dark brown", "hair_style": "very long wavy"})
        self.assertEqual(long["rgb"], sc.hex_rgb("#3b2a20"))        # not plain "brown"
        self.assertEqual(long["fall"], -0.45)                       # not plain "long"
        self.assertGreater(long["volume"], 1)
        self.assertIsNone(sc.hairdo({"hair": "blonde", "hair_style": "pixie cut"})["fall"])
        self.assertEqual(sc.hairdo({"hair": "auburn", "hair_style": "in a bun"})["tie"], "bun")
        buzz = sc.hairdo({"hair_style": "buzz cut"})
        self.assertLess(buzz["cap"], sc.hairdo({"hair": "black"})["cap"])
        self.assertEqual(buzz["rgb"], sc.hex_rgb(sc.HAIR_DEFAULT))

        # Drawn on the head, in its colour; long hair hangs lower.
        plain, short = self.person(), self.person(hair="red", hair_style="short")
        low = lambda o: min(p[1] for part, fs, rgb in sc.painted_pieces(o)   # noqa: E731
                            if rgb == sc.hex_rgb("#9a3b1f") for f in fs for p in f)
        hair = [part for part, _, rgb in sc.painted_pieces(short) if rgb == sc.hex_rgb("#9a3b1f")]
        self.assertEqual(set(hair), {"head"})
        longer = self.person(hair="red", hair_style="long")
        self.assertLess(low(longer), low(short) - 0.2)
        self.assertEqual(sc.bounds(short)[0][1], sc.bounds(plain)[0][1])
        # Under a hat the scalp is the hat's, and the head keeps its crown.
        hatted = self.person(hair="red", hair_style="short", accessories="beanie")
        self.assertNotIn(sc.hex_rgb("#9a3b1f"), {rgb for _, _, rgb in sc.painted_pieces(hatted)})

    def test_a_garment_is_the_colour_it_names_first(self):
        black = sc.hex_rgb("#27272b")
        self.assertEqual(sc.cloth_colour("black leather jacket", "outerwear"), black)
        self.assertEqual(sc.cloth_colour("leather jacket", "outerwear"),
                         sc.hex_rgb("#3b2a22"))
        self.assertEqual(sc.cloth_colour("jeans", "bottom"), sc.hex_rgb("#4d6a8f"))
        self.assertLess(sum(sc.cloth_colour("dark green hoodie", "top")),
                        sum(sc.cloth_colour("green hoodie", "top")))
        self.assertEqual(sc.cloth_colour("hoodie", "top"), sc.hex_rgb(sc.CLOTH_DEFAULT["top"]))

    def test_the_frame_shows_the_clothes(self):
        s = staged("person")
        s["objects"][0]["look"] = {"top": "red sweater"}
        w, h = sc.frame_size(s)
        polys = [p for p in sc.render(s, w, h) if p.owner]
        red = lambda rgb: rgb[0] > 2 * rgb[1] and rgb[0] > 2 * rgb[2]   # noqa: E731
        self.assertTrue(any(red(p.rgb) for p in polys))
        s["objects"][0]["look"] = {}
        self.assertFalse(any(red(p.rgb) for p in sc.render(s, w, h) if p.owner))


class TestCamera(unittest.TestCase):
    def test_the_target_is_the_middle_of_the_frame(self):
        s = staged()
        cam = sc.Camera(s["camera"], 896, 1152)
        x, y, depth = cam.project(tuple(s["camera"]["target"]))
        self.assertAlmostEqual(x, 448)
        self.assertAlmostEqual(y, 576)
        self.assertAlmostEqual(depth, s["camera"]["distance"])

    def test_behind_the_camera_is_nothing_and_the_floor_ray_inverts(self):
        s = staged()
        cam = sc.Camera(s["camera"], 1024, 1024)
        self.assertIsNone(cam.project(sc.add(cam.eye, sc.mul(cam.f, -1))))
        x, y, _ = cam.project((0.7, 0.0, 1.2))
        hit = cam.on_floor(x, y)
        for a, b in zip(hit, (0.7, 0.0, 1.2)):
            self.assertAlmostEqual(a, b, places=6)

    def test_a_longer_lens_is_a_tighter_frame(self):
        s = staged()
        wide = sc.Camera(dict(s["camera"], lens=24), 1024, 1024).project((1, 1, 0))
        long = sc.Camera(dict(s["camera"], lens=85), 1024, 1024).project((1, 1, 0))
        self.assertGreater(abs(long[0] - 512), abs(wide[0] - 512))


class TestRender(unittest.TestCase):
    def test_the_png_is_the_frame_at_the_generation_size(self):
        s = staged("person", "box")
        for key, _, w, h in sc.FRAMES:
            with self.subTest(frame=key):
                s["frame"] = key
                data = sc.png(s)
                self.assertEqual(png_size(data), (w, h))

    def test_the_frame_shows_what_is_in_it(self):
        s = staged("person")
        w, h = sc.frame_size(s)
        rgb = sc.rasterise(sc.render(s, w, h), w, h)
        px = lambda x, y: tuple(rgb[(y * w + x) * 3:(y * w + x) * 3 + 3])   # noqa: E731
        self.assertEqual(px(2, 2), sc.SKY)
        self.assertEqual(px(2, h - 3), sc.FLOOR)
        self.assertNotIn(px(w // 2, h // 2), (sc.SKY, sc.FLOOR))   # the person's chest

    def test_every_face_names_its_object_and_part(self):
        s = staged("person", "cylinder")
        polys = sc.render(s)
        self.assertIsNone(polys[0].owner)                     # the floor, drawn first
        owners = {(p.owner, p.part) for p in polys[1:]}
        self.assertIn(("person", "hand_r"), owners)
        self.assertIn(("cylinder", "body"), owners)
        depths = [p.depth for p in polys[1:]]
        self.assertEqual(depths, sorted(depths, reverse=True))

    def test_a_person_by_the_far_end_of_a_long_wall_is_not_hidden_by_it(self):
        """A wall running away from the camera, a person standing by its far
        end: the wall's middle is nearer than the person, so drawn whole it
        went on top of them."""
        s = staged("person", "box")
        person, wall = s["objects"]
        wall.update(scale=[0.2, 2.5, 10], position=[1.0, 0, -5], colour="#4f7fb5")
        person["position"] = [0.75, 0, -7]
        w, h = sc.frame_size(s)
        rgb = sc.rasterise(sc.render(s, w, h), w, h)
        x, y, _ = sc.Camera(s["camera"], w, h).project((0.75, 1.3, -7))   # their chest
        i = (int(y) * w + int(x)) * 3
        r, g, b = rgb[i:i + 3]
        self.assertGreater(r, b, "the wall's blue is over the person's chest")
        self.assertEqual(sc.bounds(wall)[0][1], 0.0)            # tiling keeps its shape

    def test_an_object_through_the_camera_is_clipped_not_a_crash(self):
        s = staged("box")
        s["objects"][0]["scale"] = [20, 3, 20]                # the camera is inside it
        self.assertEqual(png_size(sc.png(s)), sc.frame_size(s))

    def test_a_shadow_plants_what_stands_on_the_floor(self):
        """Without a contact shadow the picture made from the frame draws
        the person hovering over the ground."""
        s = staged("person")
        w, h = sc.frame_size(s)
        cam = sc.Camera(s["camera"], w, h)
        x, y, _ = cam.project((0.3, 0, 0.05))               # beside the right foot
        shadows = [p for p in sc.render(s, w, h) if p.dim]
        self.assertTrue(shadows)
        self.assertTrue(all(p.owner is None and p.part == "shadow" for p in shadows))
        under = pixel(s, x, y)
        self.assertLess(sum(under), sum(sc.FLOOR))
        self.assertEqual(pixel(s, 2, h - 3), sc.FLOOR)        # far from it, no shadow

    def test_a_lifted_foot_casts_no_contact_shadow(self):
        s = staged("person")
        pieces = sc.painted_pieces(s["objects"][0])
        cam = sc.Camera(s["camera"], *sc.frame_size(s))
        both = len(sc.shadow_polys(pieces, cam))
        s["objects"][0]["pose"]["controls"]["leg_l_step"] = 40
        one = len(sc.shadow_polys(sc.painted_pieces(s["objects"][0]), cam))
        self.assertEqual(both - one, len(sc.CONTACT))

    def test_an_object_up_on_a_platform_leaves_no_shadow_on_the_floor(self):
        s = staged("box")
        s["objects"][0]["position"][1] = 0.8
        self.assertFalse([p for p in sc.render(s) if p.dim])

    def test_the_window_draws_a_shadow_as_its_flat_stand_in(self):
        s = staged("person")
        shadows = [p for p in sc.render(s) if p.dim]
        inner = shadows[-1]
        self.assertLess(sum(inner.rgb), sum(shadows[0].rgb))  # darker inwards

    def test_the_reference_is_named_by_its_content(self):
        s = staged("person")
        d = tempfile.mkdtemp()
        a = sc.write_reference(s, d)
        self.assertEqual(sc.write_reference(s, d), a)
        s["objects"][0]["position"][0] = 1.0
        self.assertNotEqual(sc.write_reference(s, d), a)


class TestMaps(unittest.TestCase):
    """The pose and depth maps the FLUX ControlNet is given instead of the frame."""

    def figure(self, yaw=0.0):
        s = staged("person")
        s["objects"][0]["rotation"][0] = yaw
        figs = sc.pose_figures(s)
        self.assertEqual(len(figs), 1)
        return s, figs[0]

    def test_a_person_facing_the_camera_has_every_point_and_a_face(self):
        s, fig = self.figure()
        pts = fig["points"]
        self.assertTrue(all(pts), pts)
        self.assertLess(pts[2][0], pts[5][0])      # their right is on the picture's left
        self.assertEqual(pts[1], [(pts[2][0] + pts[5][0]) / 2, (pts[2][1] + pts[5][1]) / 2])
        self.assertLess(pts[0][1], pts[1][1])      # the nose above the neck
        self.assertLess(pts[1][1], pts[10][1])     # the neck above the ankles
        self.assertEqual(len(fig["face"]), 68)
        # Every point is on the mannequin as the frame shows it.
        w, h = sc.frame_size(s)
        cam = sc.Camera(s["camera"], w, h)
        seen = [cam.project(p) for _, fs, _ in sc.painted_pieces(s["objects"][0])
                for f in fs for p in f]
        xs, ys = [p[0] / w for p in seen], [p[1] / h for p in seen]
        for x, y in pts:
            self.assertTrue(min(xs) <= x <= max(xs) and min(ys) <= y <= max(ys), (x, y))

    def test_turned_away_the_face_is_hidden_and_the_ears_are_not(self):
        _, fig = self.figure(180)
        pts = fig["points"]
        self.assertEqual((pts[0], pts[14], pts[15]), (None, None, None))
        self.assertTrue(pts[16] and pts[17])
        self.assertEqual(fig["face"], [])
        self.assertGreater(pts[2][0], pts[5][0])   # seen from behind, right is right

    def test_in_profile_the_far_eye_and_ear_go_and_the_face_stays(self):
        _, fig = self.figure(90)
        pts = fig["points"]
        self.assertEqual(sum(p is None for p in (pts[14], pts[15])), 1)
        self.assertEqual(sum(p is None for p in (pts[16], pts[17])), 1)
        self.assertEqual(len(fig["face"]), 68)

    def test_a_crowd_is_a_figure_each_and_far_ones_come_first(self):
        s = staged("person", "crowd")
        s["objects"][0]["position"] = [-1.4, 0, 0]
        s["objects"][1]["position"] = [0.9, 0, -4]
        s["objects"][1]["crowd"].update(count=5, width=2, depth=1)
        figs = sc.pose_figures(s)
        self.assertEqual(len(figs), 6)
        depths = [f["depth"] for f in figs]
        self.assertEqual(depths, sorted(depths, reverse=True))

    def test_what_stands_in_front_hides_the_joints_behind_it(self):
        s = staged("person", "person")
        s["objects"][1]["position"] = [0.25, 0, -2.5]      # half behind the first
        figs = sc.pose_figures(s)
        self.assertEqual(len(figs), 2)
        far, near = figs
        self.assertTrue(all(near["points"]))              # the one in front is whole
        self.assertTrue(any(p is None for p in far["points"]))
        self.assertTrue(any(far["points"]))               # the one behind is not gone
        s["objects"][1]["position"] = [0, 0, -2.5]         # straight behind: the body goes
        far = sc.pose_figures(s)[0]["points"]
        self.assertEqual([far[i] for i in (1, 2, 5, 8, 11)], [None] * 5)

    def test_someone_out_of_the_frame_is_left_out(self):
        s = staged("person")
        s["objects"][0]["position"] = [30, 0, 0]
        self.assertEqual(sc.pose_figures(s), [])
        self.assertIsNone(sc.pose_png(s))

    def test_one_figure_draws_as_the_stick_figure_editor_does(self):
        import studio_pose as sp
        pts = sp.preset("standing", 512, 768)
        self.assertEqual(sp.render(pts, 512, 768),
                         sp.render_figures([{"points": pts}], 512, 768))

    def test_the_depth_map_is_nearer_brighter_and_the_sky_black(self):
        s = staged("person", "box")
        s["objects"][1]["position"] = [0.8, 0, -4]
        w, h = 90, 116
        zb = sc.depth_values(s, w, h)
        cam = sc.Camera(s["camera"], w, h)

        def at(obj):
            lo, hi = sc.bounds(obj)
            x, y, _ = cam.project(((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2,
                                   (lo[2] + hi[2]) / 2))
            return zb[int(y) * w + int(x)]
        person, box = (at(o) for o in s["objects"])
        self.assertGreater(box, 0)
        self.assertGreater(person, box)
        self.assertEqual(zb[:w], [0.0] * w)                       # the top row is sky
        self.assertEqual(png_size(sc.depth_png(s)), (398, 512))  # portrait, long edge 512

    def test_scene_maps_follows_the_strengths_and_the_model(self):
        d = tempfile.mkdtemp()
        s = staged("person")
        maps, notes = sc.scene_maps(s, set(sc.MAP_KINDS), d)
        self.assertEqual((sorted(maps), notes), (["composition", "pose"], []))
        self.assertTrue(os.path.basename(maps["pose"]).startswith("pose_"))
        self.assertTrue(os.path.basename(maps["composition"]).startswith("depth_"))
        with open(maps["pose"], "rb") as f:
            self.assertEqual(png_size(f.read()), (796, 1024))
        s["frame_keep"] = 0.2
        self.assertIn("source", sc.scene_maps(s, set(sc.MAP_KINDS), d)[0])
        s["frame_keep"] = 0.0
        # A model with no ControlNet has only the frame.
        self.assertEqual(list(sc.scene_maps(s, {"source"}, d)[0]), ["source"])
        s["pose_strength"] = s["depth_strength"] = 0
        maps, notes = sc.scene_maps(s, set(sc.MAP_KINDS), d)
        self.assertEqual(maps, {})
        self.assertTrue(any("only the words" in n for n in notes), notes)
        s = staged("box")                          # props only: no pose map, and no fuss
        self.assertEqual(sc.scene_maps(s, set(sc.MAP_KINDS), d), (
            {"composition": sc.scene_maps(s, {"composition"}, d)[0]["composition"]}, []))
        s = staged("person")
        s["objects"][0]["position"] = [30, 0, 0]
        maps, notes = sc.scene_maps(s, {"pose"}, d)
        self.assertEqual(maps, {})
        self.assertTrue(any("No one is in the frame" in n for n in notes), notes)

    def test_an_old_scene_with_redraw_opens_with_the_maps(self):
        s, problems = sc.clean_scene({"redraw": 0.55})
        self.assertEqual(problems, [])
        self.assertNotIn("redraw", s)
        self.assertEqual((s["pose_strength"], s["depth_strength"], s["frame_keep"]),
                         (sc.POSE_STRENGTH, sc.DEPTH_STRENGTH, sc.FRAME_KEEP))


class TestLibraryAssets(unittest.TestCase):
    """Shapes, props and the background crowd."""

    def test_every_asset_stands_on_the_floor_and_is_drawn(self):
        for a in sc.ASSETS:
            with self.subTest(asset=a["id"]):
                self.assertIn(a["group"], dict(sc.ASSET_GROUPS))
                s = staged(a["id"])
                obj = s["objects"][0]
                obj["position"] = [0.5, 0.25, -1.0]
                self.assertAlmostEqual(sc.bounds(obj)[0][1], 0.25)
                self.assertTrue([p for p in sc.render(s) if p.owner == obj["id"]])
                again, problems = sc.clean_scene(json.loads(json.dumps(s)))
                self.assertEqual(problems, [])
                self.assertEqual(again["objects"][0]["asset"], a["id"])

    def test_a_prop_is_sized_in_metres(self):
        s = staged("table")
        s["objects"][0]["scale"] = [2.0, 0.75, 0.8]
        lo, hi = sc.bounds(s["objects"][0])
        self.assertAlmostEqual(hi[0] - lo[0], 2.0, places=3)
        self.assertAlmostEqual(hi[1] - lo[1], 0.75, places=3)

    def test_a_props_parts_keep_their_own_colours(self):
        tree = staged("tree")["objects"][0]
        colours = {rgb for _, _, rgb in sc.painted_pieces(tree)}
        self.assertIn(sc.hex_rgb("#5a3e2b"), colours)             # the trunk
        self.assertIn(sc.hex_rgb(tree["colour"]), colours)        # the leaves: its colour

    def test_a_crowd_is_many_different_people_the_same_each_time(self):
        crowd = sc.new_crowd()
        crowd.update(count=20, width=8, depth=4)
        a, b = sc.crowd_members(crowd), sc.crowd_members(dict(crowd))
        self.assertEqual(len(a), 20)
        self.assertEqual(a, b)
        for i, m in enumerate(a):
            self.assertLessEqual(abs(m["at"][0]), 4)
            self.assertLessEqual(abs(m["at"][1]), 2)
            for n in a[i + 1:]:
                gap = math.hypot(m["at"][0] - n["at"][0], m["at"][1] - n["at"][1])
                self.assertGreaterEqual(gap, sc.CROWD_SPACING)
        self.assertGreater(len({json.dumps(m["look"], sort_keys=True) for m in a}), 10)
        crowd["seed"] = 2
        self.assertNotEqual(sc.crowd_members(crowd), a)

    def test_a_crowd_can_be_dressed_from_presets(self):
        crowd = sc.new_crowd()
        crowd["wear"] = [{"top": "dirndl"}, {"bottom": "lederhosen"}]
        tops = {m["look"].get("top") or m["look"].get("bottom")
                for m in sc.crowd_members(crowd)}
        self.assertEqual(tops, {"dirndl", "lederhosen"})

    def test_too_many_for_the_floor_is_as_many_as_fit(self):
        crowd = sc.new_crowd()
        crowd.update(count=40, width=1.0, depth=0.5)
        self.assertLess(len(sc.crowd_members(crowd)), 40)

    def test_a_crowd_is_said_in_one_line_and_is_not_the_scenes_people(self):
        s = staged("crowd")
        s["objects"][0]["position"] = [0, 0, -3]
        s["objects"][0]["description"] = "Oktoberfest revellers, laughing."
        s["objects"][0]["crowd"].update(count=7, activity="cheering")
        text = sc.scene_text(s).text
        self.assertIn("Crowd (a background crowd of 7 people", text)
        self.assertIn("cheering, raising a glass", text)
        self.assertIn("Oktoberfest revellers, laughing.", text)
        self.assertEqual(sc.people(s), [])
        _, extra = sc.generation(s, {})
        self.assertNotIn("subject", extra)                     # the form's person stays

    def test_each_person_in_a_crowd_casts_their_own_shadow(self):
        s = staged("crowd")
        s["objects"][0]["crowd"].update(count=5, width=6, depth=2)
        cam = sc.Camera(s["camera"], *sc.frame_size(s))
        pieces = sc.painted_pieces(s["objects"][0])
        whole = len(sc.shadow_polys(pieces, cam))
        each = len([p for p in sc.render(s) if p.dim])
        self.assertGreater(each, whole)

    def test_a_damaged_crowd_opens_within_its_limits(self):
        c = sc.clean_crowd({"count": 900, "width": -3, "facing": "sideways",
                            "wear": [{"top": "dirndl", "hair": "red"}, "junk", {}]})
        self.assertEqual((c["count"], c["width"], c["facing"]), (40, 1.0, "mixed"))
        self.assertEqual(c["wear"], [{"top": "dirndl"}])


class TestRoom(unittest.TestCase):
    """The floor and walls, and the pictures they wear."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def test_the_walls_face_into_the_room_and_hide_their_backs(self):
        room = dict(sc.new_room(), walls=True)
        for quad, origin, along in sc.walls(room):
            n = sc.newell(quad)
            self.assertGreater(sc.dot(n, sc.sub((0, 1, 0), sc.centroid(quad))), 0)
            self.assertAlmostEqual(origin[1], room["height"])
        s = staged()
        s["room"]["walls"] = True
        s["camera"].update(pitch=85, distance=20, lens=24)      # overhead: every wall
        self.assertEqual(len([p for p in sc.render(s) if p.part == "wall"]), 4)
        s["camera"].update(pitch=30, distance=12)               # outside: a doll's house
        self.assertEqual(len([p for p in sc.render(s) if p.part == "wall"]), 3)
        s["room"]["walls"] = False
        self.assertFalse([p for p in sc.render(s) if p.part == "wall"])

    def test_the_room_is_behind_everything_in_it(self):
        s = staged("box")
        s["room"].update(walls=True, depth=2.0)            # the back wall 1 m behind a crate
        polys = sc.render(s)
        kinds = [p.owner is None for p in polys]
        self.assertEqual(kinds, sorted(kinds, reverse=True))

    def test_a_picture_is_laid_on_the_floor_and_repeats(self):
        s = staged()
        s["camera"].update(pitch=89.0, distance=4.0, target=[0.0, 0.0, 0.0], lens=35)
        before = pixel(s, 448, 576)
        self.assertEqual(before, sc.FLOOR)
        s["room"]["floor"].update(image=sc.import_texture(
            two_tone(os.path.join(self.dir, "f.png"), (200, 0, 0), (0, 0, 200)), self.dir),
            size=2.0)
        # Looking straight down on the origin: x = 0 is the picture's left
        # edge, so just right of the middle is red and just left, a copy's
        # right half, is blue.
        self.assertEqual(pixel(s, 470, 576), (200, 0, 0))
        self.assertEqual(pixel(s, 426, 576), (0, 0, 200))
        flat = sc.rasterise(sc.render(s), 896, 1152, flat=True)
        self.assertEqual(tuple(flat[(576 * 896 + 470) * 3:][:3]), (100, 0, 100))   # the mean

    def test_a_missing_picture_is_drawn_plain_and_said(self):
        s = staged()
        s["room"]["floor"]["image"] = os.path.join(self.dir, "gone.png")
        self.assertEqual(sc.render(s)[0].rgb, sc.FLOOR)
        _, problems = sc.clean_scene(s)
        self.assertTrue(any("floor picture gone.png is missing" in p for p in problems))

    def test_a_picture_is_kept_small_and_named_by_content(self):
        src = two_tone(os.path.join(self.dir, "big.png"), (10, 20, 30), (40, 50, 60), n=600)
        a = sc.import_texture(src, self.dir)
        self.assertEqual(a, sc.import_texture(src, self.dir))
        with open(a, "rb") as f:
            self.assertEqual(png_size(f.read()), (sc.TEXTURE_SIDE, sc.TEXTURE_SIDE))
        with open(os.path.join(self.dir, "x.jpg"), "wb") as f:
            f.write(b"\xff\xd8\xff\xe0 not a png")
        with self.assertRaises(ValueError):
            sc.import_texture(os.path.join(self.dir, "x.jpg"), self.dir)

    def test_the_room_saves_and_older_files_open_with_a_plain_floor(self):
        s = staged()
        s["room"].update(walls=True, width=5.5, height=99)
        s["room"]["wall"]["prompt"] = "whitewashed brick"
        path = os.path.join(self.dir, "r.scene.json")
        sc.save(s, path)
        back, problems = sc.load(path)
        self.assertEqual(problems, [])
        self.assertEqual((back["room"]["walls"], back["room"]["width"]), (True, 5.5))
        self.assertEqual(back["room"]["height"], sc.ROOM_LIMITS["height"][1])
        self.assertEqual(back["room"]["wall"]["prompt"], "whitewashed brick")
        old = dict(s)
        del old["room"]
        self.assertEqual(sc.clean_scene(old)[0]["room"], sc.new_room())

    def test_the_surfaces_words_go_into_the_prompt_as_written(self):
        s = staged()
        s["room"]["floor"]["prompt"] = "Polished concrete, worn YELLOW lines"
        s["room"]["wall"]["prompt"] = "whitewashed brick"
        text = sc.scene_text(s).text
        self.assertIn("The floor: Polished concrete, worn YELLOW lines.", text)
        self.assertNotIn("brick", text)                     # no walls, no wall words
        s["room"]["walls"] = True
        self.assertIn("The walls: whitewashed brick.", sc.scene_text(s).text)

    def test_the_grid_stays_inside_the_walls(self):
        s = staged()
        s["room"].update(walls=True, width=4.0, depth=4.0)
        self.assertEqual(len(sc.grid_lines(s, 896, 1152)), 10)


class TestHistory(unittest.TestCase):
    """Undo and redo are whole-scene snapshots, each step named from what
    differs between it and the one before."""

    def test_undo_and_redo_walk_the_steps_and_hand_back_copies(self):
        s = staged()
        h = sc.History(s)
        self.assertFalse(h.can_undo() or h.can_redo())
        self.assertIsNone(h.record(s))                        # nothing changed
        box = sc.new_object("box", s["objects"])
        s["objects"].append(box)
        self.assertEqual(h.record(s, box["id"]), "Add Crate")
        box["position"][0] = 1.5
        self.assertEqual(h.record(s, box["id"]), "Move Crate")
        scene, sel = h.undo()
        self.assertEqual(scene["objects"][0]["position"][0], 0.0)
        self.assertEqual(sel, box["id"])                      # what moved back
        scene["objects"][0]["position"][0] = 9                # a copy, not a step
        scene, _ = h.undo()
        self.assertEqual(scene["objects"], [])
        self.assertIsNone(h.undo())
        scene, _ = h.redo()
        self.assertEqual(scene["objects"][0]["position"][0], 0.0)
        self.assertEqual(h.labels(), ["Start", "Add Crate", "Move Crate"])

    def test_an_edit_after_undo_drops_the_redo_steps(self):
        s = staged("box")
        h = sc.History(s)
        s["frame"] = "landscape"
        h.record(s)
        s, _ = h.undo()
        s["details"] = "A quarry"
        self.assertEqual(h.record(s), "Edit the scene details")
        self.assertFalse(h.can_redo())
        self.assertEqual(h.labels(), ["Start", "Edit the scene details"])

    def test_the_steps_are_bounded(self):
        s = staged()
        h = sc.History(s)
        for i in range(h.LIMIT + 20):
            s["details"] = "Step %d" % i
            h.record(s)
        self.assertEqual(len(h.steps), h.LIMIT)
        self.assertEqual(h.at, h.LIMIT - 1)

    def test_unsaved_is_against_the_saved_scene_not_the_edit_count(self):
        s = staged("box")
        h = sc.History(s)
        self.assertFalse(h.unsaved(s))
        s["objects"][0]["colour"] = "#ff0000"
        h.record(s)
        self.assertTrue(h.unsaved(s))
        s, _ = h.undo()
        self.assertFalse(h.unsaved(s))                        # back where it was saved

    def test_steps_are_named_for_what_changed(self):
        base = staged("person", "box")

        def said(edit):
            after = json.loads(json.dumps(base))
            edit(after)
            return sc.change_label(base, after)
        person = lambda s: s["objects"][0]  # noqa: E731
        self.assertEqual(said(lambda s: s["objects"].pop(1)), "Delete Crate")
        self.assertEqual(said(lambda s: person(s)["pose"].update(preset="")), "Pose Person")
        self.assertEqual(said(lambda s: person(s)["look"].update(hair="red")),
                         "Change Person's look")
        self.assertEqual(said(lambda s: person(s)["rotation"].__setitem__(0, 30)),
                         "Turn Person")
        self.assertEqual(said(lambda s: person(s).update(name="Ada")), "Rename Person")
        self.assertEqual(said(lambda s: s["camera"].update(yaw=40)), "Move the camera")
        self.assertEqual(said(lambda s: s["room"].update(walls=True)),
                         "Change the floor and walls")
        self.assertEqual(said(lambda s: [o["position"].__setitem__(0, 2)
                                         for o in s["objects"]]), "Change 2 objects")


class TestSceneFile(unittest.TestCase):
    def test_save_and_open_keep_everything(self):
        s = staged("person", "box")
        s["objects"][0]["description"] = "kneeling, welding; helmet down, gloves on"
        s["objects"][0]["pose"] = {"preset": "kneeling",
                                   "controls": sc.pose_controls("kneeling")}
        s["objects"][1].update(name="Workbench", scale=[1.8, 0.9, 0.8], colour="#8a6a4a")
        s["camera"].update(yaw=30.0, lens=50.0)
        s["frame"], s["pose_strength"], s["frame_keep"] = "landscape", 0.62, 0.15
        path = os.path.join(tempfile.mkdtemp(), "shop.scene.json")
        sc.save(s, path)
        back, problems = sc.load(path)
        self.assertEqual(problems, [])
        self.assertEqual(back, sc.clean_scene(s)[0])
        self.assertEqual(back["objects"][0]["description"],
                         "kneeling, welding; helmet down, gloves on")

    def test_a_damaged_file_opens_with_what_can_be_read(self):
        s, problems = sc.clean_scene({
            "frame": "huge", "pose_strength": "lots", "frame_keep": 5, "camera": {"lens": -5, "pitch": 400},
            "objects": [{"asset": "spaceship"}, "junk",
                        {"asset": "person", "id": "p", "position": [1, "x", 2],
                         "pose": {"preset": "flying", "controls": {"arm_l_raise": 999}}},
                        {"asset": "box", "id": "p", "colour": "red"}]})
        self.assertEqual(s["frame"], "portrait")
        self.assertEqual((s["pose_strength"], s["frame_keep"]),
                         (sc.POSE_STRENGTH, sc.FRAME_KEEP_MAX))
        self.assertEqual((s["camera"]["lens"], s["camera"]["pitch"]), (10, 85))
        self.assertEqual(len(problems), 2)
        person, box = s["objects"]
        self.assertEqual(person["position"], [1.0, 0.0, 2.0])
        self.assertEqual(person["pose"]["controls"]["arm_l_raise"], 180)
        self.assertEqual(person["pose"]["preset"], "")
        self.assertNotEqual(box["id"], "p")                  # ids stay unique
        self.assertEqual(box["colour"], sc.ASSET["box"]["colour"])


class TestWords(unittest.TestCase):
    def test_descriptions_are_sent_as_written(self):
        s = staged("person", "cylinder")
        p, c = s["objects"]
        p["description"] = "Kneeling to weld; face shield DOWN, hi-vis vest, gloves."
        c.update(name="Gas cylinder", description="acetylene, valve open  (hose attached)")
        c["position"] = [-1.0, 0, 0]
        words = sc.scene_text(s)
        self.assertIn("A steel workshop, overcast light.", words.text)
        self.assertIn(". Kneeling to weld; face shield DOWN, hi-vis vest, gloves.", words.text)
        self.assertIn("Gas cylinder (left of frame): acetylene, valve open  (hose attached).",
                      words.text)
        self.assertIn("a person, centre of frame, facing the camera", words.text)
        self.assertTrue(ig.has_person({"scene": words.text}, False))   # anatomy constants
        self.assertTrue(words.text.endswith("Shot from eye level on a 35mm lens."))
        self.assertEqual(words.notes, [])

    def test_what_is_outside_the_frame_is_left_out_and_said(self):
        s = staged("person", "box")
        s["objects"][1].update(name="Crate", description="an open crate",
                               position=[0, 0, 20])           # behind the camera
        words = sc.scene_text(s)
        self.assertNotIn("open crate", words.text)
        self.assertTrue(any("Crate is outside the frame" in n for n in words.notes))
        self.assertTrue(any("Person has no description" in n for n in words.notes))

    def test_which_way_a_person_faces(self):
        s = staged("person")
        p = s["objects"][0]
        for yaw, words in ((0, "facing the camera"), (180, "back to the camera"),
                           (90, "in profile, facing frame right"),
                           (-90, "in profile, facing frame left")):
            p["rotation"][0] = yaw
            self.assertEqual(sc.facing(s, p), words, yaw)

    def posed(self, preset="standing", **controls):
        s = staged("person")
        p = s["objects"][0]
        p["pose"] = {"preset": preset, "controls": dict(sc.pose_controls(preset), **controls)}
        return s, p

    def test_the_posture_is_read_off_the_posed_body(self):
        for preset, words in (
                ("standing", ["arms relaxed at the sides"]),
                ("reaching", ["right arm raised above the head",
                              "left arm hanging relaxed at the side", "looking up"]),
                ("pointing", ["right arm reaching forward at shoulder height",
                              "left arm hanging relaxed at the side"]),
                ("carrying", ["both arms bent, hands in front of the chest"]),
                ("working", ["leaning forward", "both arms bent, hands in front of the waist",
                             "looking down"])):
            _, p = self.posed(preset)
            self.assertEqual(sc.posture_words(p), words, preset)

    def test_the_legs_are_said_unless_a_named_pose_says_them(self):
        _, p = self.posed(leg_l_bend=25)
        self.assertIn("weight on the right leg, the other knee relaxed", sc.posture_words(p))
        _, p = self.posed(leg_r_step=30, leg_l_step=-10)
        self.assertIn("mid-stride, right foot forward", sc.posture_words(p))
        _, p = self.posed("kneeling")
        self.assertFalse([w for w in sc.posture_words(p) if "leg" in w or "foot" in w])
        _, p = self.posed(twist=30, lean=-15, arm_l_out=90)
        words = sc.posture_words(p)
        self.assertIn("shoulders turned to their left", words)
        self.assertIn("leaning to their right", words)
        self.assertIn("left arm stretched out to the side", words)

    def test_the_looks_gaze_outranks_the_heads_words(self):
        s, p = self.posed(head_nod=30)
        self.assertIn("looking down", sc.posture_words(p))
        p["rotation"][0] = 90
        p["pose"]["controls"]["head_turn"] = -80        # over the shoulder, to the camera
        self.assertEqual(sc.gaze_words(s, p), "head turned towards the camera")
        p["look"] = {"gaze": "looking at the camera"}
        self.assertNotIn("looking down", sc.posture_words(p))
        self.assertEqual(sc.gaze_words(s, p), "")
        p["look"] = {}
        p["pose"]["controls"]["head_turn"] = 0
        self.assertEqual(sc.gaze_words(s, p), "")        # the head goes the body's way

    def test_how_much_of_the_person_the_frame_shows(self):
        s, p = self.posed()
        self.assertEqual(sc.framing_words(s, p), "whole figure in view")
        for distance, aim, words in ((1.8, 1.2, "seen from the knees up"),
                                     (1.0, 1.35, "seen from the waist up"),
                                     (0.6, 1.6, "head and shoulders")):
            s["camera"].update(distance=distance, target=[0, aim, 0])
            self.assertEqual(sc.framing_words(s, p), words, distance)
        # Framed this close the person's middle is below the frame; they are
        # still in the words.
        self.assertIn("Person (a person, centre of frame, facing the camera, "
                      "head and shoulders)", sc.scene_text(s).text)

    def test_the_camera_in_words(self):
        s = staged()
        s["camera"].update(pitch=45, lens=24)
        self.assertEqual(sc.camera_words(s),
                         "Shot from a high angle looking down on a 24mm wide-angle lens")

    def test_each_person_says_their_own_look(self):
        s = staged("person", "person", "box")
        a, b, box = s["objects"]
        a.update(name="Welder", description="grinding a seam; face shield DOWN.",
                 look={"subject": "a man", "age": "in their 40s", "facial_hair": "full beard",
                       "top": "hi-vis vest", "stature": 2, "expression": "determined"})
        b.update(name="Apprentice", position=[0.9, 0, 0],
                 look={"subject": "a young woman", "hair": "auburn",
                       "hair_style": "in a ponytail"})
        box["position"] = [-0.9, 0, 0]
        text = sc.scene_text(s).text
        self.assertIn("Welder (a person, centre of frame, facing the camera, whole figure in "
                      "view): a man, in their 40s, tall, full beard, determined expression, "
                      "wearing hi-vis vest. Arms relaxed at the sides. grinding a seam; face "
                      "shield DOWN.", text)
        self.assertIn("Apprentice (a person, ", text)
        self.assertIn("): a young woman, auburn hair in a ponytail. Arms relaxed", text)
        self.assertNotIn("Apprentice has no description", " ".join(sc.scene_text(s).notes))

    def test_a_look_is_saved_and_cleaned(self):
        s = staged("person")
        s["objects"][0].update(character="ada", look={
            "hair": " auburn ", "weight": 9, "muscle": "x", "eyes": "", "bogus": "yes",
            "accessories": "glasses, necklace"})
        back, problems = sc.clean_scene(json.loads(json.dumps(s)))
        self.assertEqual(problems, [])
        self.assertEqual(back["objects"][0]["character"], "ada")
        self.assertEqual(back["objects"][0]["look"], {"hair": "auburn", "weight": 3,
                                                      "accessories": "glasses, necklace"})
        old = staged("person")                               # a scene from before looks
        del old["objects"][0]["look"], old["objects"][0]["character"]
        back, _ = sc.clean_scene(old)
        self.assertEqual((back["objects"][0]["look"], back["objects"][0]["character"]),
                         ({}, ""))
        self.assertNotIn("look", sc.new_object("box"))

    def test_a_character_copies_its_look_but_not_the_expression(self):
        rec = {"looks": {"subject": "a woman", "hair": "black", "weight": -1}}
        look = sc.character_look(rec, {"facial_hair": "full beard", "expression": "shy",
                                       "muscle": 2})
        self.assertEqual(look, {"subject": "a woman", "hair": "black", "weight": -1,
                                "expression": "shy"})

    def test_a_costume_is_drawn_not_only_said(self):
        d = sc.outfit({"top": "green dirndl with a pink apron", "bottom": "lederhosen"})
        self.assertEqual(d["regions"]["chest"], sc.hex_rgb("#4f7d4a"))     # the dirndl's
        self.assertEqual(d["apron"], sc.hex_rgb("#e39ab0"))                # the apron's
        self.assertEqual(d["regions"]["upper_arm"], sc.hex_rgb("#ecebe6"))  # the blouse
        self.assertIsNone(d["braces"])                   # lederhosen under a dress: hidden
        lh = sc.outfit({"bottom": "lederhosen"})
        self.assertEqual(lh["braces"], sc.hex_rgb("#5b4030"))
        self.assertNotIn("shin", lh["regions"])                            # knee length
        acc = sc.outfit({"accessories": "flower crown, red accordion, two beer steins"})
        self.assertEqual(acc["hat"], ("crown", None))                      # every colour
        self.assertEqual(acc["held"]["accordion"], (("r",), sc.hex_rgb("#b0342f")))
        self.assertEqual(acc["held"]["stein"][0], ("l", "r"))
        self.assertEqual(sc.outfit({"accessories": "beer stein"})["held"]["stein"][0],
                         ("r",))
        self.assertEqual(sc.outfit({"accessories": "german hat"})["hat"][0], "alpine")
        self.assertEqual(sc.outfit({"accessories": "hat"})["hat"][0], "hat")

    def test_costume_pieces_are_on_the_mannequin(self):
        plain = staged("person")["objects"][0]
        dressed = staged("person")["objects"][0]
        dressed["look"] = {"top": "dirndl", "accessories": "flower crown, accordion, steins"}
        count = lambda o, part: sum(len(f) for p, f, _ in sc.painted_pieces(o)   # noqa: E731
                                    if p == part)
        for part in ("head", "body", "hand_l", "hand_r"):
            self.assertGreater(count(dressed, part), count(plain, part), part)
        dressed["look"]["accessories"] = "flower crown"
        self.assertEqual(sc.bounds(dressed)[0][1], 0.0)          # still on the floor

    def test_an_outfit_replaces_the_clothes_and_keeps_the_person(self):
        look = {"subject": "a man", "hair": "grey", "top": "hoodie",
                "outerwear": "raincoat", "accessories": "glasses", "weight": 2}
        rec = {"looks": {"top": "white t-shirt", "bottom": "blue jeans"}}
        self.assertEqual(sc.wear_outfit(rec, look),
                         {"subject": "a man", "hair": "grey", "weight": 2,
                          "top": "white t-shirt", "bottom": "blue jeans"})
        self.assertEqual(sc.outfit_looks(look), {"top": "hoodie", "outerwear": "raincoat",
                                                 "accessories": "glasses"})

    def test_people_in_the_scene_blank_the_forms_person(self):
        s = staged("person")
        s["objects"][0].update(character="ada", look={"subject": "a woman"})
        chars = {"ada": {"id": "ada", "identity": "ada-face", "item_refs": {"x": "/y.png"}}}
        _, extra = sc.generation(s, {}, chars)
        self.assertEqual((extra["subject"], extra["hair"], extra["weight"],
                          extra["character"], extra["item_refs"]), ("", "", 0, "", {}))
        self.assertEqual(extra["scene_identities"], ["ada-face"])
        _, props = sc.generation(staged("box"), {})
        self.assertNotIn("subject", props)                   # no people: the form's stays
        self.assertNotIn("scene_identities", props)

    def test_generation_carries_size_strengths_and_the_scene(self):
        s = staged("person")
        s["frame"], s["pose_strength"], s["depth_strength"] = "landscape", 0.7, 0.4
        s["frame_keep"] = 0.2
        maps = {"pose": "/x/p.png", "composition": "/x/d.png", "source": "/x/f.png"}
        words, extra = sc.generation(s, maps)
        self.assertEqual((extra["width"], extra["height"], extra["denoise"]), (1344, 768, 0.8))
        self.assertEqual(extra["references"], maps)
        self.assertEqual((extra["pose"], extra["composition"]),
                         ({"strength": 0.7}, {"strength": 0.4}))
        # The frame alone (a model with no ControlNet) keeps at least the old
        # default, and the form's drawn pose is blanked.
        _, extra = sc.generation(s, {"source": "/x/f.png"})
        self.assertEqual((extra["denoise"], extra["pose"], extra["composition"]),
                         (round(1 - sc.FALLBACK_KEEP, 3), None, None))
        _, extra = sc.generation(s, {"pose": "/x/p.png"})
        self.assertNotIn("denoise", extra)                     # nothing to denoise from
        self.assertEqual(extra["scene_layout"]["objects"][0]["id"], "person")
        s["objects"][0]["name"] = "changed"
        self.assertEqual(extra["scene_layout"]["objects"][0]["name"], "Person")


    def test_people_get_the_face_pass_told_where_each_face_is(self):
        s = staged("person", "person")
        s["objects"][1]["position"] = [0.8, 0.0, 0.0]
        s["objects"][0]["description"] = "Laughing."
        _, extra = sc.generation(s, {"pose": "/x/p.png"})
        self.assertTrue(extra["face_detail"])
        faces = extra["scene_faces"]
        self.assertEqual(faces["likeness"], sc.FACE_LIKENESS)
        a, b = faces["people"]
        self.assertLess(a["at"][0], b["at"][0])              # left to right as placed
        self.assertTrue(0.05 < a["at"][1] < 0.5)             # a face is high in the frame
        x0, y0, x1, y1 = a["region"]                         # the head, round the face
        self.assertTrue(x0 < a["at"][0] < x1 and y0 < a["at"][1] < y1)
        self.assertLess(x1, b["region"][2])
        self.assertLess(x1 - x0, 0.3)
        self.assertIn("Laughing.", a["words"])
        self.assertIn("steel workshop", a["words"])
        self.assertEqual((a["face"], b["face"]), ("", ""))
        s["objects"][1]["position"] = [40.0, 0.0, 0.0]       # out of the frame
        _, extra = sc.generation(s, {"pose": "/x/p.png"})
        self.assertEqual(len(extra["scene_faces"]["people"]), 1)

    def test_a_face_picture_is_the_persons_own_or_their_identitys(self):
        folder = tempfile.mkdtemp()
        own, ref = os.path.join(folder, "own.png"), os.path.join(folder, "ref.jpg")
        for path in (own, ref):
            with open(path, "wb") as f:
                f.write(b"x")
        s = staged("person")
        obj = s["objects"][0]
        obj["character"] = "lil"
        chars = {"lil": {"id": "lil", "identity": "lilya"}}
        idents = {"lilya": {"id": "lilya", "name": "Lilya", "references": [ref],
                            "use_references": True}}
        self.assertEqual(sc.face_picture(obj, chars, idents), (ref, "Lilya's profile"))
        obj["face"] = own
        _, extra = sc.generation(s, {"pose": "/x/p.png"}, chars, idents)
        self.assertEqual(extra["scene_faces"]["people"][0]["face"], own)
        # every photo of them goes to the real-face paste, their own first
        self.assertEqual(extra["scene_faces"]["people"][0]["photos"], [own, ref])
        self.assertIs(extra["scene_faces"]["real"], True)
        s["real_faces"] = False
        again, _ = sc.clean_scene(json.loads(json.dumps(s)))
        self.assertIs(again["real_faces"], False)
        self.assertIs(sc.clean_scene({})[0]["real_faces"], sc.REAL_FACES)
        s["real_faces"] = True
        # kept through a save, and said when the file has gone
        again, problems = sc.clean_scene(json.loads(json.dumps(s)))
        self.assertEqual(again["objects"][0]["face"], own)
        os.remove(own)
        _, problems = sc.clean_scene(json.loads(json.dumps(s)))
        self.assertTrue(any("face picture" in p for p in problems), problems)


class TestIntoCompose(TempStudioMixin, unittest.TestCase):
    """The frame and the words through the Image Studio's own compose()."""

    def settings(self, model, takes, **scene):
        s = staged("person")
        s.update(scene)
        s["objects"][0]["description"] = "checking a gauge, hard hat and hi-vis on"
        maps, _ = sc.scene_maps(s, takes, tempfile.mkdtemp())
        words, extra = sc.generation(s, maps)
        st = dict(ig.default_settings(), model=model, scene=words.text, **extra)
        return st, maps

    def test_a_model_with_no_controlnet_takes_the_frame(self):
        st, maps = self.settings("z-image-turbo", {"source"})
        self.assertEqual(list(maps), ["source"])
        plan = ig.compose(st, self.studio.lib, self.backend("3090"), FLUX_FILES)
        self.assertEqual(plan.errors, [])
        self.assertEqual(plan.references.get("source"), maps["source"])
        self.assertEqual((plan.values["width"], plan.values["height"]), (896, 1152))
        self.assertEqual(plan.values["denoise"], round(1 - sc.FALLBACK_KEEP, 3))
        self.assertIn("checking a gauge, hard hat and hi-vis on", plan.prompt)
        self.assertIn(ig.anatomy_text(), plan.prompt)

    def test_the_forms_person_is_not_said_twice(self):
        s = staged("person")
        s["objects"][0]["look"] = {"subject": "an older man", "hair": "grey"}
        maps, _ = sc.scene_maps(s, {"source"}, tempfile.mkdtemp())
        words, extra = sc.generation(s, maps)
        st = dict(ig.default_settings(), model="z-image-turbo", scene=words.text,
                  subject="a woman", hair="auburn")
        st.update(extra)
        plan = ig.compose(st, self.studio.lib, self.backend("3090"), FLUX_FILES)
        self.assertIn("an older man, grey hair", plan.prompt)
        self.assertNotIn("a woman", plan.prompt)
        self.assertNotIn("auburn", plan.prompt)

    def test_flux_takes_the_pose_and_depth_maps_and_no_frame(self):
        cn = "FLUX.1-dev-ControlNet-Union-Pro-2.0.safetensors"
        inv = dict(FLUX_FILES, controlnet={cn})
        st, maps = self.settings("flux-dev", set(sc.MAP_KINDS), pose_strength=0.8,
                                 depth_strength=0.45)
        self.assertEqual(sorted(maps), ["composition", "pose"])
        plan = ig.compose(st, self.studio.lib, self.backend("5090"), inv)
        self.assertEqual(plan.errors, [])
        self.assertEqual(plan.images, {"pose_image": maps["pose"],
                                       "composition_image": maps["composition"]})
        self.assertEqual((plan.values["pose_strength"], plan.values["composition_strength"]),
                         (0.8, 0.45))
        g = ig.fill(plan.workflow, dict(plan.values, pose_image="p.png",
                                        composition_image="d.png"))
        self.assertEqual(g["40"]["inputs"]["denoise"], 1.0)     # from noise: no frame
        self.assertEqual(g["40"]["inputs"]["latent_image"], ["20", 0])
        # Some of the frame kept pins the props too, image to image on top.
        st, maps = self.settings("flux-dev", set(sc.MAP_KINDS), frame_keep=0.2)
        plan = ig.compose(st, self.studio.lib, self.backend("5090"), inv)
        self.assertEqual(plan.images["source_image"], maps["source"])
        self.assertEqual(plan.values["denoise"], 0.8)
        # Without the ControlNet file the backend says so and goes on in words.
        st, _ = self.settings("flux-dev", set(sc.MAP_KINDS))
        plan = ig.compose(st, self.studio.lib, self.backend("5090"), FLUX_FILES)
        self.assertEqual(plan.images, {})
        self.assertTrue(any("lacks" in w and cn in w for w in plan.warnings), plan.warnings)

    def test_a_floor_is_text_to_image_with_nothing_of_the_person(self):
        s = staged()
        s["room"]["floor"]["prompt"] = "oily concrete, drain in the corner."
        st = sc.texture_settings(s, "floor", "flux-dev")
        plan = ig.compose(st, self.studio.lib, self.backend("5090"), FLUX_FILES)
        self.assertEqual(plan.errors, [])
        self.assertIn("seen from directly above", plan.prompt)
        self.assertIn(": oily concrete, drain in the corner. Flat", plan.prompt)
        self.assertNotIn(ig.anatomy_text(), plan.prompt)
        self.assertEqual(plan.references, {})
        self.assertEqual((plan.values["width"], plan.values["height"]), (1024, 1024))
        self.assertEqual(st["scene_texture"], "floor")


def photo_of(controls, yaw, scale=300, drop=()):
    """A pose finder's 133 points for the mannequin in `controls`, turned
    `yaw`, seen from the front in picture pixels; `drop` points unseen."""
    pts = [[0.0, 0.0, 0.0] for _ in range(133)]
    for i, p in sc.pose_points(controls, yaw).items():
        if i not in drop:
            pts[i] = [600 + scale * p[0], 500 - scale * p[1], 0.9]
    return pts


def joints_apart(a, b):
    """The farthest any joint of pose `a` is from its place in pose `b`, in
    metres, as a front view shows them (x and y): what a photo can show."""
    pa, pb = sc.pose_points(*a), sc.pose_points(*b)
    return max(math.hypot(pa[i][0] - pb[i][0], pa[i][1] - pb[i][1]) for i in pa)


class TestPoseFromPhoto(unittest.TestCase):
    def test_a_pose_comes_back_from_its_own_picture(self):
        for preset, yaw in (("walking", 0), ("pointing", 40), ("sitting", -70),
                            ("carrying", 160)):
            with self.subTest(pose=preset, yaw=yaw):
                c = sc.pose_controls(preset)
                fit = sc.fit_pose(photo_of(c, yaw))
                self.assertLess(abs((fit.yaw - yaw + 180) % 360 - 180), 15)
                self.assertLess(joints_apart((fit.controls, fit.yaw), (c, yaw)), 0.07)
                self.assertFalse(fit.rough)
                self.assertEqual(fit.unseen, [])
                self.assertEqual(set(fit.controls), set(sc.CONTROL_KEYS))

    def test_the_fit_does_not_care_where_or_how_big_the_person_is(self):
        c = sc.pose_controls("working")
        small = sc.fit_pose(photo_of(c, 30, scale=80))
        big = sc.fit_pose(photo_of(c, 30, scale=900))
        self.assertLess(joints_apart((small.controls, small.yaw), (big.controls, big.yaw)),
                        0.05)

    def test_a_part_the_photo_does_not_show_stays_at_rest_and_is_said(self):
        c = sc.pose_controls("walking")
        fit = sc.fit_pose(photo_of(c, 0, drop=(13, 15)))        # the left knee and ankle
        self.assertEqual(fit.unseen, ["the left leg"])
        rest = sc.pose_controls("standing")
        for k in ("leg_l_step", "leg_l_out", "leg_l_bend"):
            self.assertEqual(fit.controls[k], rest[k])

    def test_too_little_of_a_person_is_refused_in_words(self):
        pts = photo_of(sc.pose_controls("standing"), 0, drop=range(3, 17))
        with self.assertRaisesRegex(ValueError, "does not show enough"):
            sc.fit_pose(pts)

    def test_the_most_prominent_person_comes_first(self):
        pts = [[0, 0, 0]] * 133
        data = json.dumps({"people": [
            {"box": [0, 0, 50, 100], "score": 0.9, "points": pts},
            {"box": [0, 0, 300, 600], "score": 0.8, "points": pts},
            {"box": [0, 0, 1], "score": 0.99, "points": pts},          # malformed
            {"box": [0, 0, 400, 800], "score": 0.9, "points": pts[:5]}]})  # too few
        folk = sc.photo_people(data)
        self.assertEqual([p["box"][2] for p in folk], [300, 50])
        self.assertEqual(sc.photo_people({"people": []}), [])


def picture_of(people, distance=5.0, eye_y=1.6, lens=35.0, size=(1344, 768)):
    """A pose finder's answer for a photo of `people` [(x, z, preset, yaw)]
    stood in a scene, taken by a level camera at `distance` from the origin."""
    w, h = size
    cam = sc.Camera({"target": [0, eye_y, 0], "yaw": 0, "pitch": 0, "distance": distance,
                     "lens": lens}, w, h)
    folk = []
    for x, z, preset, yaw in people:
        c = sc.pose_controls(preset)
        pelvis = (x, sc.pelvis_height(c, yaw), z)
        pts = [[0.0, 0.0, 0.0] for _ in range(133)]
        for i, p in sc.pose_points(c, yaw).items():
            sx, sy, _ = cam.project(sc.add(pelvis, p))
            pts[i] = [sx, sy, 0.9]
        xs, ys = [p[0] for p in pts[:17]], [p[1] for p in pts[:17]]
        folk.append({"box": [min(xs), min(ys) - 20, max(xs), max(ys)], "score": 0.9,
                     "points": pts})
    return {"width": w, "height": h, "people": folk}


class TestSceneFromPicture(unittest.TestCase):
    TRUTH = [(-1.0, 0.0, "standing", 0), (1.2, -2.0, "walking", 40),
             (0.2, 1.0, "pointing", -30)]

    def test_everyone_stands_where_they_were_turned_as_they_were(self):
        scene, notes = sc.picture_scene(picture_of(self.TRUTH))
        self.assertEqual(notes, [])
        self.assertEqual(scene["frame"], "landscape")
        cam = scene["camera"]
        self.assertAlmostEqual(cam["target"][1], 1.6, delta=0.1)     # the eye's height
        self.assertEqual(cam["pitch"], 0.0)
        eye = sc.Camera(cam, *sc.frame_size(scene)).eye
        got = sorted((o["position"][0], o["position"][2] - eye[2], o) for o in
                     scene["objects"])
        for (x, z, preset, yaw), (gx, gz, o) in zip(sorted(self.TRUTH), got):
            with self.subTest(preset=preset):
                self.assertAlmostEqual(gx, x, delta=0.15)
                self.assertAlmostEqual(gz, z - 5.0, delta=0.25)      # from the eye
                toward = math.degrees(math.atan2(eye[0] - gx, eye[2] - o["position"][2]))
                turned = (o["rotation"][0] - toward + 180) % 360 - 180
                self.assertLess(abs(turned - yaw), 15)
                self.assertEqual(o["position"][1], 0.0)
        self.assertEqual(sc.clean_scene(scene)[1], [])               # a scene that saves

    def boxed(self, n, data):
        """Where found person `n` (1 is the most prominent) is, as a reply
        gives it: in the photo's pixels, its height a little off, as the
        vision model's are."""
        b = sc.photo_people(data)[n - 1]["box"]
        return [round(b[0]) + 6, round(b[1]) + 30, round(b[2]) - 4, round(b[3]) - 10]

    def test_the_words_go_to_the_setting_the_floor_and_each_person(self):
        data = picture_of(self.TRUTH)
        reply = json.dumps({"setting": "A beer tent at night", "floor": "wooden boards",
                            "people": [
            {"box": self.boxed(2, data), "name": "Walker", "doing": "carrying two steins",
             "top": "white linen shirt", "bottom": "lederhosen", "hair": "", "age": "none",
             "subject": "man", "mood": "happy"},
            {"box": self.boxed(1, data), "name": "Walker", "age": "in their 30s..."},
            {"box": [0, 0, 5, 5], "name": "Nobody"},                  # no one is there
            "junk", {"name": "Boxless"}]})
        read = sc.picture_answer("Sure! ```json\n%s\n```" % reply, 1344, 768)
        self.assertEqual(len(read["people"]), 4)
        scene, _ = sc.picture_scene(data, read)
        self.assertEqual(scene["details"], "A beer tent at night")
        self.assertEqual(scene["room"]["floor"]["prompt"], "wooden boards")
        walker = next(o for o in scene["objects"] if o["name"] == "Walker")
        self.assertEqual(walker["description"], "carrying two steins")
        self.assertEqual(walker["look"], {"top": "white linen shirt", "bottom": "lederhosen",
                                          "subject": "a man"})   # no "mood" slot, no "none"
        other = next(o for o in scene["objects"] if o["name"] == "Walker 2")
        self.assertEqual(other["look"], {"age": "in their 30s"})  # the example's "..." gone
        self.assertEqual(len(scene["objects"]), 3)
        self.assertIn("Person 3", [o["name"] for o in scene["objects"]])

    def test_the_vision_models_own_order_does_not_matter(self):
        """It is matched by where each person is, not by number: a 7B model
        listed the band first and put its clothes on the audience."""
        data = picture_of(self.TRUTH)
        said = [{"box": [x / 1344.0 for x in self.boxed(n, data)], "name": "P%d" % n}
                for n in (3, 1, 2)]
        for s in said:
            s["box"] = [s["box"][0], s["box"][1] * 1344 / 768, s["box"][2],
                        s["box"][3] * 1344 / 768]
        got = sc.match_people(sc.photo_people(data), said, 1344, 768)
        self.assertEqual({n: s["name"] for n, s in got.items()},
                         {1: "P1", 2: "P2", 3: "P3"})

    def test_a_box_in_percentages_is_read_as_such(self):
        read = sc.picture_answer('{"people": [{"box": [10, 20, 30, 40], "name": "A"}]}',
                                 2000, 1000)
        self.assertEqual(read["people"][0]["box"], [0.1, 0.2, 0.3, 0.4])
        read = sc.picture_answer('{"people": [{"bbox_2d": [200, 100, 600, 1000]}]}',
                                 2000, 1000)
        self.assertEqual(read["people"][0]["box"], [0.1, 0.1, 0.3, 1.0])

    def test_a_reply_that_is_not_json_leaves_the_words_empty(self):
        for text in ("I can't see the people.", "{broken", "", None, "[1, 2]"):
            self.assertEqual(sc.picture_answer(text),
                             {"setting": "", "floor": "", "people": []})

    def test_the_question_gives_the_photos_size_and_how_many(self):
        q = sc.picture_question([{"box": [0, 0, 50, 100]}, {"box": [100, 50, 200, 200]}],
                                200, 150)
        self.assertIn("200 x 150 pixels", q)
        self.assertIn("It shows 2 people", q)

    def test_the_frame_is_the_photos_shape(self):
        self.assertEqual(sc.picture_frame(1100, 1000), "square")
        self.assertEqual(sc.picture_frame(4000, 3000), "landscape")     # 4:3 is nearer 7:4
        self.assertEqual(sc.picture_frame(3000, 4000), "portrait")
        self.assertEqual(sc.picture_frame(1920, 1080), "landscape")

    def test_no_one_to_pose_is_refused_in_words(self):
        with self.assertRaisesRegex(ValueError, "found no one"):
            sc.picture_scene({"width": 100, "height": 100, "people": []})
        few = picture_of(self.TRUTH[:1])
        for i in range(3, 17):
            few["people"][0]["points"][i] = [0, 0, 0]
        with self.assertRaisesRegex(ValueError, "enough of themselves"):
            sc.picture_scene(few)

    def test_past_the_limit_the_smaller_people_are_left_out_and_said(self):
        crowd = [(-4 + i * 0.8, -1.0 - (i % 3), "standing", 0)
                 for i in range(sc.PICTURE_PEOPLE + 2)]
        scene, notes = sc.picture_scene(picture_of(crowd, distance=9.0))
        self.assertEqual(len(scene["objects"]), sc.PICTURE_PEOPLE)
        self.assertIn("2 smaller people were left out", notes[0])


class PoseClient(FakeClient):
    """A ComfyUI with the pose finder, answering with one person."""
    answer = None

    def node_types(self):
        return FakeClient.node_types(self) | {"LoadImage", ig.POSE_NODE}

    def listen_for_progress(self, pid, on_event, stop=None, timeout=0):
        return {"status": {"completed": True},
                "outputs": {"2": {"text": [json.dumps(PoseClient.answer)]}}}


class TestFindPoses(TempStudioMixin, unittest.TestCase):
    def photo(self):
        path = os.path.join(self.dir, "pose.png")
        with open(path, "wb") as f:
            f.write(b"not really a png")
        return path

    def test_a_backend_without_the_finder_is_named_with_how_to_add_it(self):
        with self.assertRaisesRegex(ig.ComfyError, "custom_nodes.*restart ComfyUI"):
            self.studio.find_poses(self.photo())

    def test_the_finder_is_one_load_and_one_node_and_its_json_comes_back(self):
        self.studio.client_factory = PoseClient
        PoseClient.answer = {"width": 10, "height": 20, "people": []}
        data, b = self.studio.find_poses(self.photo())
        self.assertEqual(data["height"], 20)
        graph = PoseClient.instances[-1].graphs[-1]
        self.assertEqual(graph["2"], {"class_type": ig.POSE_NODE,
                                      "inputs": {"image": ["1", 0]}})
        self.assertEqual(graph["1"]["class_type"], "LoadImage")
        self.assertTrue(b["enabled"])


def sc_room():
    import studio_scene_ui
    return studio_scene_ui.ROOM


class Ev:
    def __init__(self, x, y, state=0, keysym=""):
        self.x, self.y, self.state, self.keysym = x, y, state, keysym


@unittest.skipIf(_headless(), "no display")
class TestSceneBuilderWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import studio_chat
        cls.mod = studio_chat
        cls.dir = tempfile.mkdtemp()
        cls._real_settings = os.environ.get("STUDIO_SETTINGS")
        os.environ["STUDIO_SETTINGS"] = os.path.join(cls.dir, "settings.json")
        with open(os.environ["STUDIO_SETTINGS"], "w") as f:
            json.dump({"tabs": ["chat", "image-studio"]}, f)
        cls._saved = {n: getattr(studio_chat.Chat, n) for n in
                      ("_boot_host", "_read_icons", "_boot_session")}
        studio_chat.Chat._boot_host = lambda self, *a, **k: None
        studio_chat.Chat._read_icons = lambda self: None
        studio_chat.Chat._boot_session = lambda self, s: None
        cls._client = ig.ComfyUIClient
        ig.ComfyUIClient = FakeClient
        cls.app = studio_chat.Chat()
        cls.app.geometry("1400x900")
        for _ in range(10):
            cls.app.update()

    @classmethod
    def tearDownClass(cls):
        cls.app._quit()
        ig.ComfyUIClient = cls._client
        for n, fn in cls._saved.items():
            setattr(cls.mod.Chat, n, fn)
        if cls._real_settings is None:
            os.environ.pop("STUDIO_SETTINGS", None)
        else:
            os.environ["STUDIO_SETTINGS"] = cls._real_settings

    def pump(self, until, seconds=5):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.app.update()
            if until():
                return
            time.sleep(0.01)
        raise AssertionError("condition not reached")

    def builder(self):
        self.app._select("image-studio")
        self.app.update()
        ui = self.app.sessions["image-studio"].images
        sb = ui.build_scene()
        sb.win.geometry("1240x780")
        for _ in range(5):
            self.app.update()
        self.addCleanup(self._close, ui, sb)
        return ui, sb

    def _close(self, ui, sb):
        sb.dirty = False
        if sb.win.winfo_exists():
            sb.close()
        sb = ui.scene_builder                                 # one a test opened anew
        if sb is not None:
            sb.dirty = False
            sb.close()
        self.assertIsNone(ui.scene_builder)

    def centre_of(self, sb, tag):
        items = sb.canvas.find_withtag(tag)
        self.assertTrue(items, tag)
        pts = sb.canvas.coords(items[-1])
        xs, ys = pts[0::2], pts[1::2]
        return sum(xs) / len(xs), sum(ys) / len(ys)

    def test_one_window_starting_from_the_forms_scene(self):
        self.app._select("image-studio")
        ui = self.app.sessions["image-studio"].images
        ui.scene.delete("1.0", "end")
        ui.scene.insert("1.0", "A loading dock at dawn")
        ui, sb = self.builder()
        self.assertEqual(sb.scene["details"], "A loading dock at dawn")
        self.assertIs(ui.build_scene(), sb)                   # raised, not a second one

    def test_the_builder_goes_with_its_form(self):
        """It writes into the Image Studio's form, so closing the tab closes
        it - on the UI thread, since Session.close runs on a worker."""
        ui, sb = self.builder()
        ui.release()
        self.assertIsNone(ui.scene_builder)
        self.assertFalse(sb.win.winfo_exists())
        self.assertIsNot(ui.build_scene(), sb)

    def test_the_library_adds_shapes_props_and_stand_ins(self):
        ui, sb = self.builder()
        post = sb.add("cylinder", "Post")
        chair = sb.add("chair")
        self.app.update()
        self.assertEqual((post["name"], post["scale"]), ("Post", [0.15, 2.5, 0.15]))
        self.assertTrue(sb.canvas.find_withtag("o:" + chair["id"]) or
                        any(p.owner == chair["id"] for p in sc.render(sb.scene)))
        self.assertEqual(sb.stand_in_btn.winfo_ismapped(), 1)

    def test_a_photo_poses_the_person_and_turns_them_as_in_it(self):
        ui, sb = self.builder()
        person = sb.add("person")
        person["position"] = [1.5, 0.0, 0.0]            # off the camera's axis
        c = sc.pose_controls("pointing")
        other = photo_of(sc.pose_controls("standing"), 0, scale=60)
        answer = {"people": [{"box": [0, 0, 60, 100], "score": 0.9, "points": other},
                             {"box": [300, 0, 900, 900], "score": 0.9,
                              "points": photo_of(c, 40)}]}
        real = ui.studio.find_poses
        ui.studio.find_poses = lambda path, stop=None: (answer, {"name": "5090"})
        self.addCleanup(setattr, ui.studio, "find_poses", real)
        self.assertTrue(sb.pose_from_photo(person["id"], path="C:/photos/point.jpg"))
        self.assertIn(person["id"], sb.posing)
        self.assertFalse(sb.pose_from_photo(person["id"], path="C:/again.jpg"))  # one at a time
        self.pump(lambda: person["id"] not in sb.posing, 30)
        self.assertEqual(person["pose"]["preset"], "")
        got = (person["pose"]["controls"], 40)
        self.assertLess(joints_apart(got, (c, 40)), 0.07)       # the big one, not the small
        eye = sc.Camera(sb.scene["camera"], *sc.frame_size(sb.scene)).eye
        toward = math.degrees(math.atan2(eye[0] - 1.5, eye[2]))
        turned = (person["rotation"][0] - toward + 180) % 360 - 180
        self.assertLess(abs(turned - 40), 15)               # + is to frame right
        text = sb.msg.cget("text")
        self.assertIn("Posed Person from point.jpg", text.replace(person["name"], "Person"))
        self.assertIn("most prominent of 2 people", text)

        ui.studio.find_poses = lambda path, stop=None: ({"people": []}, {"name": "5090"})
        sb.pose_from_photo(person["id"], path="C:/photos/empty.jpg")
        self.pump(lambda: person["id"] not in sb.posing, 10)
        self.assertIn("found no one in empty.jpg", sb.msg.cget("text"))

    def test_a_picture_makes_a_new_scene_with_its_people_and_words(self):
        ui, sb = self.builder()
        sb.add("crate" if "crate" in sc.ASSET else "box")
        sb.dirty = False                                      # nothing to ask about
        answer = picture_of(TestSceneFromPicture.TRUTH)
        real = ui.studio.find_poses
        ui.studio.find_poses = lambda path, stop=None: (answer, {"name": "5090"})
        self.addCleanup(setattr, ui.studio, "find_poses", real)

        class Eyes:
            asked = []

            def ask(self, path, question, max_tokens=400):
                Eyes.asked.append((path, question))
                box = sc.photo_people(answer)[0]["box"]
                return json.dumps({"setting": "An alpine meadow",
                                   "people": [{"box": box, "name": "Hiker"}]})
        self.app.vision, was = Eyes(), getattr(self.app, "vision", None)
        self.addCleanup(setattr, self.app, "vision", was)
        self.assertTrue(sb.from_picture("C:/photos/meadow.jpg"))
        self.assertFalse(sb.from_picture("C:/photos/again.jpg"))   # one at a time
        self.pump(lambda: not sb.picturing, 60)
        self.assertEqual(sb.scene["details"], "An alpine meadow")
        self.assertEqual(len(sb.scene["objects"]), 3)             # the crate is gone
        self.assertIn("Hiker", [o["name"] for o in sb.scene["objects"]])
        self.assertIn("It shows 3 people", Eyes.asked[0][1])
        self.assertIsNone(sb.path)
        self.assertTrue(sb.dirty)
        self.assertIn("Made a scene from meadow.jpg: 3 people", sb.msg.cget("text"))
        self.assertEqual(sb.history.labels()[-1], "From meadow.jpg")

        self.app.vision = None                                  # no eyes: still a scene
        sb.dirty = False
        sb.from_picture("C:/photos/meadow.jpg")
        self.pump(lambda: not sb.picturing, 60)
        self.assertEqual(sb.scene["details"], "")
        self.assertIn("no vision model is connected", sb.msg.cget("text"))

        before = sb.scene
        ui.studio.find_poses = lambda path, stop=None: ({"people": []}, {"name": "5090"})
        sb.from_picture("C:/photos/empty.jpg")
        self.pump(lambda: not sb.picturing, 10)
        self.assertIn("Could not make a scene from empty.jpg", sb.msg.cget("text"))
        self.assertIs(sb.scene, before)                           # the scene is kept

    def test_undo_and_redo_put_the_scene_back(self):
        ui, sb = self.builder()
        self.assertEqual(sb.undo_pill.state, "disabled")
        person = sb.add("person")
        sb.remember()                                         # the edit has settled
        self.assertEqual(sb.undo_pill.state, "normal")
        x, y = self.centre_of(sb, "p:body")
        start = list(person["position"])
        sb._press(Ev(x, y))
        sb._motion(Ev(x + 60, y))
        self.assertEqual(sb.history.labels()[-1], "Add Person")   # not mid-drag
        sb._release(Ev(x + 60, y))
        self.assertEqual(sb.history.labels()[-1], "Move Person")  # one step per drag
        moved = list(sb.obj(person["id"])["position"])
        self.assertNotEqual(moved, start)

        self.assertTrue(sb.undo())
        self.assertEqual(sb.obj(person["id"])["position"], start)
        self.assertEqual(sb.sel, person["id"])
        self.assertAlmostEqual(sb.vars["x"][0].get(), start[0], places=2)
        self.assertIn("Undid: Move Person", sb.msg.cget("text"))
        self.assertEqual(sb.redo_pill.state, "normal")
        self.assertTrue(sb.redo())
        self.assertEqual(sb.obj(person["id"])["position"], moved)

        sb.go_to(0)                                           # from the History menu
        self.assertEqual(sb.scene["objects"], [])
        self.assertEqual(sb.lb.size(), 2)                     # the scene and room rows
        self.assertIsNone(sb.sel)
        self.assertFalse(sb.dirty)                            # as the window opened
        self.assertTrue(sb.redo())
        self.assertTrue(sb.dirty)

    def test_a_slider_drag_is_one_step_and_undo_takes_a_pending_edit(self):
        import tkinter as tk
        ui, sb = self.builder()
        sb.add("box")
        sb.remember()
        steps = len(sb.history.steps)
        want = str(sb.vars["x"][0])
        scale = next(w for row in sb.panel.winfo_children() for w in row.winfo_children()
                     if isinstance(w, tk.Scale) and str(w.cget("variable")) == want)
        start = sb.obj()["position"][0]
        for v in (0.5, 1.0, 1.5, 2.0):
            scale.set(start + v)
            self.app.update()
        self.assertEqual(len(sb.history.steps), steps)        # still settling
        sb.undo()                                             # records it, then undoes it
        self.assertEqual(sb.obj()["position"][0], start)
        self.assertEqual(sb.history.labels()[-1], "Move Crate")
        self.assertEqual(len(sb.history.steps), steps + 1)

    def test_ctrl_z_is_the_scenes_except_in_a_text_box(self):
        import tkinter as tk
        ui, sb = self.builder()
        sb.add("box")
        sb.remember()
        sb.canvas.focus_force()
        self.app.update()
        sb.canvas.event_generate("<Control-z>")
        self.app.update()
        self.assertEqual(sb.scene["objects"], [])
        sb.canvas.event_generate("<Control-y>")
        self.app.update()
        self.assertEqual(len(sb.scene["objects"]), 1)
        ev = Ev(0, 0)
        ev.widget = tk.Text(sb.panel)
        self.assertIsNone(sb._undo_key(ev, sb.undo))
        self.assertEqual(len(sb.scene["objects"]), 1)

    def test_a_new_or_opened_scene_starts_a_new_history(self):
        ui, sb = self.builder()
        sb.add("box")
        sb.remember()
        sb.dirty = False
        sb.new()
        self.assertEqual(sb.history.labels(), ["New scene"])
        self.assertEqual(sb.undo_pill.state, "disabled")
        sb.add("barrel")
        path = os.path.join(self.dir, "undo.scene.json")
        self.assertTrue(sb.save(path))
        sb.remember()
        sb.undo()
        self.assertTrue(sb.dirty)                             # the saved scene had a barrel
        sb.redo()
        self.assertFalse(sb.dirty)
        self.assertTrue(sb.open(path))
        self.assertEqual(sb.history.labels(), ["Opened undo.scene.json"])

    def test_click_a_hand_to_pose_it_and_drag_to_move(self):
        ui, sb = self.builder()
        person = sb.add("person")
        self.app.update()
        x, y = self.centre_of(sb, "p:hand_r")
        oid, part = sb.hit(x, y)
        self.assertEqual((oid, part), (person["id"], "hand_r"))
        sb._press(Ev(x, y))
        sb._release()
        self.assertEqual((sb.sel, sb.part), (person["id"], "hand_r"))
        self.assertIn("arm_r_raise", sb.vars)
        self.assertNotIn("head_turn", sb.vars)
        sb._set_pose("pointing")
        self.assertEqual(person["pose"]["controls"]["arm_r_raise"], 85)
        self.assertAlmostEqual(sb.vars["arm_r_raise"][0].get(), 85)
        sb._reset_part()                                       # the right hand only
        self.assertEqual(person["pose"]["controls"]["arm_r_raise"], 0)
        self.assertEqual(person["pose"]["preset"], "")        # no longer the preset
        import tkinter as tk
        sb._set_pose("pointing")
        want = str(sb.vars["arm_r_raise"][0])
        scale = next(w for row in sb.panel.winfo_children() for w in row.winfo_children()
                     if isinstance(w, tk.Scale) and str(w.cget("variable")) == want)
        scale.set(40)                                          # as a drag of the slider
        self.app.update()
        self.assertEqual(person["pose"]["controls"]["arm_r_raise"], 40)
        self.assertEqual(person["pose"]["preset"], "")
        self.assertIn("Custom", sb.pose_pill.cget("text"))

        x, y = self.centre_of(sb, "p:body")
        before = list(person["position"])
        sb._press(Ev(x, y))
        sb._motion(Ev(x + 60, y))
        sb._release(Ev(x + 60, y))
        self.assertGreater(person["position"][0], before[0] + 0.1)   # to frame right
        self.assertLess(person["position"][0], before[0] + 1.5)      # pixels, not metres
        self.assertEqual(person["position"][1], before[1])           # still on its floor
        sb.set_tool("rotate")
        sb._press(Ev(*self.centre_of(sb, "p:body")))
        sb._motion(Ev(x + 160, y))
        sb._release()
        self.assertNotEqual(person["rotation"][0], 0)
        sb.set_tool("move")
        self.assertAlmostEqual(sb.vars["x"][0].get(), person["position"][0], places=2)

    def test_look_at_opens_a_ring_of_head_poses(self):
        ui, sb = self.builder()
        a = sb.add("person")
        b = sb.add("person")
        b["position"] = [1.2, 0.0, -0.5]
        a["position"] = [-0.6, 0.0, 0.0]
        sb.changed()
        self.app.update()
        sb.set_tool("look")

        def ring_item(key):
            return next((x, y) for k, _, x, y in sb._ring_items() if k == key)

        def open_ring(person):
            sb._press(Ev(*self.centre_of(sb, "o:" + person["id"])))
            sb._release()
            self.assertEqual((sb.sel, sb.ring["oid"]), (person["id"], person["id"]))
            self.assertTrue(sb.canvas.find_withtag("ring:ahead"))

        open_ring(a)                                           # a faces the camera
        sb._press(Ev(*ring_item("right")))                     # frame right
        sb._release()
        self.assertIsNone(sb.ring)
        self.assertFalse(sb.canvas.find_withtag("ring"))
        self.assertEqual(a["pose"]["controls"]["head_turn"], sc.HEAD_TURN)  # their left
        open_ring(a)
        sb._press(Ev(*ring_item("down")))
        sb._release()
        self.assertEqual(a["pose"]["controls"]["head_turn"], 0)
        self.assertGreater(a["pose"]["controls"]["head_nod"], 20)
        self.assertIn("looks down", sb.msg.cget("text"))

        a["rotation"][0] = 180.0                               # turned away: sides swap
        sc.head_pose(sb.scene, a, "right")
        self.assertEqual(a["pose"]["controls"]["head_turn"], -sc.HEAD_TURN)
        a["rotation"][0] = 0.0
        sb.changed()

        open_ring(a)                                           # Point, then b
        sb._press(Ev(*ring_item("point")))
        sb._release()
        sb._press(Ev(*self.centre_of(sb, "o:" + b["id"])))
        sb._release()
        self.assertEqual(sb.sel, a["id"])
        self.assertGreater(a["look_at"][0], 0.5)               # at b, not the floor
        self.assertGreater(a["pose"]["controls"]["head_turn"], 10)
        self.assertIn("looks at " + b["name"], sb.msg.cget("text"))
        open_ring(a)                                           # a pose drops the point
        sb._press(Ev(*ring_item("ahead")))
        sb._release()
        self.assertNotIn("look_at", a)
        open_ring(a)
        sb._press(Ev(*ring_item("camera")))
        sb._release()
        self.assertEqual(a["look_at"], [round(v, 3) for v in sb.camera().eye])
        open_ring(b)                                           # Esc closes, then lets go
        sb._key(Ev(0, 0, keysym="Escape"))
        self.assertIsNone(sb.ring)
        self.assertEqual(sb.sel, b["id"])
        sb._key(Ev(0, 0, keysym="Escape"))
        self.assertIsNone(sb.sel)
        crowd = sb.add("crowd")
        self.app.update()
        sb._press(Ev(*self.centre_of(sb, "o:" + crowd["id"])))
        self.assertIsNone(sb.ring)

    def test_a_person_keeps_their_look_in_the_inspector(self):
        ui, sb = self.builder()
        for k, var in ui.text.items():
            var.set("")
        for var in ui.sliders.values():
            var.set(0)
        ui.settings["character"] = ""
        ui.text["subject"].set("a woman")
        ui.text["hair"].set("auburn")
        first = sb.add("person")                               # takes the form's look
        self.assertEqual(first["look"], {"subject": "a woman", "hair": "auburn"})
        second = sb.add("person")                              # the second does not
        self.assertEqual(second["look"], {})

        sb._look_tab("Hair")
        self.assertIn("hair_style", sb.look_vars)
        sb.look_vars["hair_style"].set("buzz cut")
        sb.look_changed()                                      # as a key in its field
        self.assertEqual(second["look"], {"hair_style": "buzz cut"})
        self.assertIn("buzz cut", sb.words_label.cget("text"))

        # Clothes and the body are drawn at once, not only said.
        colours = lambda: {sb.canvas.itemcget(i, "fill")             # noqa: E731
                           for i in sb.canvas.find_withtag("o:" + second["id"])}
        before = colours()
        sb._look_tab("Clothes")
        sb.look_vars["top"].set("red sweater")
        sb.look_changed()
        self.assertTrue(colours() - before)
        self.assertTrue(sb.dirty)
        del second["look"]["top"]

        ui.studio.lib.save("characters", [{"id": "ada", "name": "Ada", "identity": "",
                                           "looks": {"subject": "a woman", "hair": "black",
                                                     "weight": -1}}])
        second["look"]["expression"] = "shy"
        sb._set_character("ada")
        self.assertEqual(second["character"], "ada")
        self.assertEqual(second["name"], "Ada")                # was still "Person 2"
        self.assertEqual(second["look"], {"subject": "a woman", "hair": "black",
                                          "weight": -1, "expression": "shy"})
        sb._clear_look()
        self.assertEqual((second["look"], second["character"]), ({}, ""))
        ui.studio.lib.save("characters", [])

    def test_save_and_put_on_an_outfit_preset(self):
        ui, sb = self.builder()
        ui.studio.lib.save("outfits", [])
        self.addCleanup(ui.studio.lib.save, "outfits", ig._default_outfits())
        a, b = sb.add("person"), sb.add("person")
        a["look"] = {"subject": "a woman", "top": "red sweater", "footwear": "heels"}
        sb.select(a["id"])
        sb._save_outfit("Red night")
        self.assertEqual([(r["id"], r["looks"]) for r in ui.studio.lib.all("outfits")],
                         [("red-night", {"top": "red sweater", "footwear": "heels"})])
        b["look"] = {"subject": "a man", "top": "hoodie", "bottom": "shorts"}
        sb.select(b["id"])
        sb._put_on_outfit("red-night")
        self.assertEqual(b["look"], {"subject": "a man", "top": "red sweater",
                                     "footwear": "heels"})
        self.assertTrue(sb.dirty)
        self.assertEqual(sb.outfit_name, "Red night")
        from unittest import mock
        with mock.patch("studio_scene_ui.messagebox.askyesno", return_value=True):
            sb._delete_outfit("red NIGHT")
        self.assertEqual(ui.studio.lib.all("outfits"), [])
        self.assertEqual(b["look"]["top"], "red sweater")    # wearers keep their clothes

    def test_a_background_crowd_in_the_window(self):
        ui, sb = self.builder()
        crowd = sb.add("crowd")
        self.assertIn("crowd_count", sb.vars)
        self.assertTrue(sb.canvas.find_withtag("o:" + crowd["id"]))
        seed = crowd["crowd"]["seed"]
        sb._shuffle_crowd(crowd)
        self.assertNotEqual(crowd["crowd"]["seed"], seed)
        preset = ui.studio.lib.all("outfits")[0]
        sb._dress_crowd(crowd, preset["id"])
        self.assertEqual(crowd["crowd"]["wear"], [preset["looks"]])
        sb._dress_crowd(crowd, "*")
        self.assertEqual(len(crowd["crowd"]["wear"]), len(ui.studio.lib.all("outfits")))
        sb._dress_crowd(crowd, "")
        self.assertEqual(crowd["crowd"]["wear"], [])
        self.assertIn("background crowd", sb.words_label.cget("text"))
        for asset in ("tree", "sphere"):
            obj = sb.add(asset)
            self.assertIn("size0", sb.vars)                      # sized in metres
            self.assertEqual(sb.sel, obj["id"])

    def test_empty_space_orbits_and_the_wheel_zooms(self):
        ui, sb = self.builder()
        sb._press(Ev(5, 5))
        self.assertIsNone(sb.sel)
        yaw = sb.scene["camera"]["yaw"]
        sb._motion(Ev(105, 5))
        sb._release()
        self.assertNotEqual(sb.scene["camera"]["yaw"], yaw)
        d = sb.scene["camera"]["distance"]
        sb._zoom(1)
        self.assertGreater(sb.scene["camera"]["distance"], d)

    def test_make_a_floor_and_walls_from_words(self):
        """Make sends the words to the Image Studio as text to image, and the
        finished picture is put on the surface and drawn in the viewport."""
        ui, sb = self.builder()
        sb.select(sc_room())
        self.assertIn("floor_size", sb.vars)
        self.assertFalse(sb.make_texture("floor"))              # no words yet
        sb.scene["room"]["floor"]["prompt"] = "polished concrete"
        n = len(ui.jobs)
        self.assertTrue(sb.make_texture("floor"))
        self.pump(lambda: len(ui.jobs) > n and ui.jobs[0].status in ig.FINISHED)
        job = ui.jobs[0]
        self.assertEqual(job.status, "complete", job.detail)
        self.assertEqual(job.settings["scene_texture"], "floor")
        self.assertEqual(job.settings["references"], {})
        self.assertIn("polished concrete", job.settings["scene"])
        self.pump(lambda: sb.scene["room"]["floor"]["image"])
        self.assertTrue(os.path.isfile(sb.scene["room"]["floor"]["image"]))
        self.assertNotIn("floor", sb.making)
        self.assertTrue(sb.dirty)
        self.pump(lambda: any(sb.canvas.type(i) == "image" for i in sb.canvas.find_all()))
        sb.texture_done(job)                                    # told twice: used once
        self.assertNotIn("floor", sb.making)

        sb.scene["room"]["wall"]["prompt"] = "whitewashed brick"
        self.assertTrue(sb.make_texture("wall"))
        self.assertTrue(sb.scene["room"]["walls"])              # Make walls puts them up
        self.pump(lambda: sb.scene["room"]["wall"]["image"])
        self.assertIn("The walls: whitewashed brick.", sb.words_label.cget("text"))
        sb._set_model("z-image-turbo")
        self.assertEqual(sb.check(), "")                        # a room alone is enough

    def test_save_reopen_and_generate_through_the_image_studio(self):
        ui, sb = self.builder()
        sb.scene["details"] = "Steel workshop, overcast"
        p = sb.add("person")
        p["description"] = "welding a beam; helmet down, leather gloves"
        sb._set_pose("kneeling")
        b = sb.add("box")
        b.update(name="Workbench", description="steel bench, vice closed")
        path = os.path.join(tempfile.mkdtemp(), "shop.scene.json")
        self.assertTrue(sb.save(path))
        self.assertFalse(sb.dirty)
        sb.scene = sc.new_scene()
        self.assertTrue(sb.open(path))
        self.assertEqual([o["name"] for o in sb.scene["objects"]], ["Person", "Workbench"])
        self.assertEqual(sb.scene["objects"][0]["pose"]["preset"], "kneeling")

        ui.settings["model"] = "flux-dev"                   # pose and depth maps
        self.assertEqual(sb.takes("flux-dev"), set(sc.MAP_KINDS))
        self.assertEqual(sb.check(), "")
        self.assertEqual(sb.takes("no-such-model"), set())

        sb._set_model("z-image-turbo")                      # the frame alone
        ui.random_seed.set(False)
        ui.adv["seed"].set("77")
        n = len(ui.jobs)
        self.assertTrue(sb.generate())
        self.pump(lambda: len(ui.jobs) > n and ui.jobs[0].status in ig.FINISHED)
        job = ui.jobs[0]
        self.assertEqual(job.status, "complete", job.detail)
        st = job.settings
        self.assertTrue(os.path.isfile(st["references"]["source"]))
        with open(st["references"]["source"], "rb") as f:
            self.assertEqual(png_size(f.read()), (896, 1152))
        self.assertEqual((st["width"], st["height"], st["denoise"]),
                         (896, 1152, round(1 - sc.FALLBACK_KEEP, 3)))
        self.assertEqual((st["pose"], st["composition"]), (None, None))
        self.assertIn("welding a beam; helmet down, leather gloves", st["scene"])
        self.assertIn("Workbench (", st["scene"])
        self.assertEqual(st["scene_file"], path)
        self.assertEqual((st["subject"], st["character"]), ("", ""))   # the scene's people only
        self.assertEqual(len(st["scene_layout"]["objects"]), 2)
        self.assertIn("welding a beam", ui.scene.get("1.0", "end"))   # on the form too
        rec = ui.studio.history.list()[0]
        self.assertEqual(rec["settings"]["scene_layout"]["objects"][0]["pose"]["preset"],
                         "kneeling")
        # A plain Generate from the form afterwards carries no scene.
        self.assertNotIn("scene_layout", ui.collect())
        self.assertNotIn("source", ui.collect()["references"])   # the form is left alone

        # On FLUX the maps take the form's pose and source slots for the job;
        # its other references stay.
        style = os.path.join(tempfile.mkdtemp(), "look.png")
        two_tone(style, (200, 40, 40), (40, 40, 200))
        ui._set_ref("style", style)
        ui._set_ref("source", style)
        sb._set_model("flux-dev")
        n = len(ui.jobs)
        self.assertTrue(sb.generate())
        self.pump(lambda: len(ui.jobs) > n and ui.jobs[0].status in ig.FINISHED)
        refs = ui.jobs[0].settings["references"]
        self.assertEqual(sorted(refs), ["composition", "pose", "style"])
        self.assertEqual(refs["style"], style)
        self.assertEqual(ui.jobs[0].settings["pose"], {"strength": sc.POSE_STRENGTH})


class FakeLLM:
    def __init__(self, *replies):
        self.replies, self.sent = list(replies), []

    def chat(self, messages, max_tokens=None):
        self.sent.append(messages)
        return {"choices": [{"message": {"content": self.replies.pop(0)}}]}


class TestShapesAndProps(unittest.TestCase):
    """The library past box and cylinder: primitives, compound props under one
    transform, and stand-ins that start a shape named and sized."""

    def test_every_prop_stands_on_the_floor_at_its_scale(self):
        for a in sc.ASSETS:
            if a["kind"] != "prop":
                continue
            with self.subTest(asset=a["id"]):
                obj = sc.new_object(a["id"])
                obj["position"] = [1.0, 0.3, -2.0]
                lo, hi = sc.bounds(obj)
                self.assertAlmostEqual(lo[1], 0.3, places=6)
                size = [h - l for l, h in zip(lo, hi)]
                # Within its scale, and most of its height (a lamp post's
                # pole, a car's cab, a parasol's top do not fill the box).
                self.assertGreater(size[1], 0.8 * obj["scale"][1])
                for got, want in zip(size, obj["scale"]):
                    self.assertLessEqual(got, want * 1.03 + 0.01)

    def test_every_face_turns_outward(self):
        for key in ("box", "cylinder", "sphere", "cone", "frustum", "capsule", "wedge",
                    "pyramid", "plane"):
            with self.subTest(shape=key):
                faces = sc.UNIT[key]
                mid = sc.centroid([p for f in faces for p in f])
                for f in faces:
                    self.assertGreaterEqual(
                        sc.dot(sc.newell(f), sc.sub(sc.centroid(f), mid)), -1e-9)

    def test_a_compound_prop_is_one_object_that_turns_as_one(self):
        s = staged("table")
        t = s["objects"][0]
        t["rotation"] = [90.0, 0.0, 0.0]
        lo, hi = sc.bounds(t)
        self.assertAlmostEqual(hi[0] - lo[0], 0.8, delta=0.03)     # its depth, now across
        self.assertAlmostEqual(hi[2] - lo[2], 1.4, delta=0.03)
        self.assertEqual({p.owner for p in sc.render(s)[1:] if p.owner}, {"table"})
        saved, problems = sc.clean_scene(json.loads(json.dumps(s)))
        self.assertEqual(problems, [])
        self.assertEqual(saved["objects"][0]["asset"], "table")
        self.assertNotIn("parts", saved["objects"][0])            # by id, not geometry

    def test_a_stand_in_starts_named_and_sized(self):
        s = staged()
        cab = sc.new_object("box", s["objects"], "Cabinet")
        self.assertEqual((cab["name"], cab["scale"]), ("Cabinet", [0.9, 1.9, 0.5]))
        s["objects"].append(cab)
        again = sc.new_object("box", s["objects"], "Cabinet")
        self.assertEqual(again["name"], "Cabinet 2")
        self.assertEqual(sc.new_object("box", (), "Drum")["name"], "Crate")  # not a box's
        for label, aid, _, _, _ in sc.STAND_INS:
            self.assertIn(aid, sc.ASSET, label)
        self.assertIn("Cabinet", sc.scene_text(dict(s, objects=[dict(cab, description="Grey "
                                                                     "steel, doors shut.")])).text)

    def test_enrich_can_place_the_new_shapes(self):
        got = sc.read_suggestion('{"detail": "A bush in a pot.", "shape": "sphere", '
                                 '"size": [0.8, 0.8, 0.8], "where": "behind_right"}')
        self.assertEqual(got["shape"], "sphere")
        s = staged("person")
        obj = sc.place_suggestion(s, got)
        self.assertEqual(obj["asset"], "sphere")
        self.assertEqual(obj["scale"], [0.8, 0.8, 0.8])


class EnrichTest(unittest.TestCase):
    def scene(self):
        s = staged("person")
        s["details"] = "A Munich street during Oktoberfest"
        return s

    def test_reads_json_prose_and_thinking(self):
        got = sc.read_suggestion('<think>hmm</think>{"detail": "A pretzel basket on a far '
                                 'table.", "why": "food", "shape": "box", "size": [1.8, 0.75, '
                                 '0.8], "colour": "#8A6A4A", "where": "behind_left"}')
        self.assertEqual((got["detail"], got["why"], got["shape"], got["where"], got["colour"]),
                         ("A pretzel basket on a far table.", "food", "box", "behind_left",
                          "#8a6a4a"))
        self.assertEqual(got["size"], [1.8, 0.75, 0.8])
        prose = sc.read_suggestion("Suggested enrichment: - A jacket over a bench.")
        self.assertEqual((prose["detail"], prose["shape"]), ("A jacket over a bench.", "none"))
        self.assertIsNone(sc.read_suggestion(""))
        # Nonsense fields fall back rather than reaching the scene.
        bad = sc.read_suggestion('{"detail": "A lamp post.", "shape": "spaceship", "size": [1, "x"],'
                                 ' "where": "inside Gavin", "colour": "red"}')
        self.assertEqual((bad["shape"], bad["size"], bad["where"], bad["colour"]),
                         ("none", None, "behind", ""))

    def two_people(self):
        s = self.scene()
        s["objects"].append(sc.new_object("person", s["objects"]))
        s["objects"][0]["position"] = [-0.5, 0.0, 0.0]
        s["objects"][1]["position"] = [0.5, 0.0, 0.0]
        return s

    def test_places_around_the_people_as_the_camera_sees_it(self):
        s = self.two_people()
        cam = sc.Camera(s["camera"], *sc.frame_size(s))
        for where in sc.WHERE:
            sug = sc.new_suggestion("A long wooden festival table.")
            # Close to the lens a table would cover their legs; a basket fits.
            size = [0.4, 0.3, 0.3] if where.startswith("foreground") else [1.8, 0.75, 0.8]
            sug.update(shape="box", size=size, where=where, name="Table")
            obj = sc.place_suggestion(s, sug)
            self.assertIsNotNone(obj, where)
            self.assertIsNotNone(sc.placement(s, obj), where)
            for other in s["objects"]:     # never on top of anyone, nor hidden or hiding
                self.assertFalse(sc._overlaps(sc.bounds(obj), sc.bounds(other), gap=0), where)
                self.assertFalse(sc._hides(sc._screen_rect(cam, obj),
                                           sc._screen_rect(cam, other)), where)
            depth = sc.dot(sc.sub(obj["position"], (0, 0, 0)), sc._flat(cam.f))
            across = sc.dot(obj["position"], sc._flat(cam.r))
            if where.startswith("behind") or where == "far_background":
                self.assertGreater(depth, 0.5, where)
            if where.startswith("foreground"):
                self.assertLess(depth, 0, where)
            if where.endswith("left"):
                self.assertLess(across, -0.5, where)
            if where.endswith("right"):
                self.assertGreater(across, 0.5, where)
        above = sc.new_suggestion("Bunting.")
        above.update(shape="box", size=[3, 0.1, 0.1], where="above")
        self.assertEqual(sc.place_suggestion(s, above)["position"][1], 2.4)

    def test_placed_object_carries_the_words_and_a_second_goes_elsewhere(self):
        s = self.two_people()
        sug = sc.new_suggestion("Two half-full steins on a far table.")
        sug.update(shape="box", size=[1.8, 0.75, 0.8], where="behind_right", name="Beer table")
        first = sc.place_suggestion(s, sug)
        s["objects"].append(first)
        second = sc.place_suggestion(s, sug)
        self.assertEqual(second["name"], "Beer table 2")
        self.assertFalse(sc._overlaps(sc.bounds(first), sc.bounds(second), gap=0))
        self.assertIn("Beer table (", sc.scene_text(s).text)
        self.assertIn("Two half-full steins on a far table.", sc.scene_text(s).text)
        self.assertEqual(first["colour"], sc.ASSET["box"]["colour"])

    def test_a_passer_by_faces_as_asked_and_bodiless_stays_words(self):
        s = self.two_people()
        sug = sc.new_suggestion("A blurred couple in dirndl and lederhosen.")
        sug.update(shape="person", size=[0.5, 1.7, 0.3], where="far_background")
        obj = sc.place_suggestion(s, sug)
        self.assertEqual(obj["asset"], "person")
        self.assertEqual(sc.facing(s, obj), "facing the camera")
        sug["facing"] = "side"
        self.assertIn("profile", sc.facing(s, sc.place_suggestion(s, sug)))
        light = sc.new_suggestion("Warm uneven tent light.")
        self.assertIsNone(sc.place_suggestion(s, light))
        # A table too big for the foreground goes behind them, same side.
        big = sc.new_suggestion("A long table.")
        big.update(shape="box", size=[1.8, 0.75, 0.8], where="foreground_left")
        self.assertIsNone(sc._place(s, big))
        self.assertEqual(sc.placement(s, sc.place_suggestion(s, big))[0], "left of frame")

    def test_a_placed_passer_by_does_not_move_the_subjects(self):
        s = self.two_people()
        walker = sc.new_suggestion("A blurred stranger crossing far behind.")
        walker.update(shape="person", where="far_background")
        s["objects"].append(sc.place_suggestion(s, walker))
        sc.enrich_answer(s, walker["detail"], "placed")
        mat = sc.new_suggestion("A curled beer mat on the cobbles.")
        mat.update(shape="box", size=[0.3, 0.05, 0.3], where="foreground_left")
        self.assertGreater(sc.place_suggestion(s, mat)["position"][2], 0.5)

    def test_prompt_carries_scene_history_and_rotates(self):
        s = self.scene()
        first = sc.enrich_angle(s)
        sc.enrich_answer(s, "Blue-and-white bunting overhead.", "never")
        sc.enrich_answer(s, "Two half-full steins on a far table.", "skip")
        user = sc.enrich_messages(s)[1]["content"]
        self.assertIn("Oktoberfest", user)
        self.assertIn("bunting", user)
        self.assertIn("Never suggest", user)
        self.assertNotEqual(sc.enrich_angle(s), first)

    def test_suggest_refuses_repeats_then_gives_up(self):
        s = self.scene()
        sc.enrich_answer(s, "A jacket draped over a bench.", "skip")
        llm = FakeLLM('{"detail": "a jacket draped over a bench"}',
                      '{"detail": "Condensation beading on a stein at the frame edge."}')
        self.assertEqual(sc.suggest(s, llm)["detail"],
                         "Condensation beading on a stein at the frame edge.")
        with self.assertRaises(RuntimeError):
            sc.suggest(s, FakeLLM('{"detail": "A jacket draped over a bench."}',
                                  "{}"))

    def test_added_goes_into_words_and_survives_a_save(self):
        s = self.scene()
        sc.enrich_answer(s, "Warm tent light spills unevenly across the cobbles.", "add")
        text = sc.scene_text(s).text
        self.assertIn("Warm tent light spills unevenly across the cobbles.", text)
        self.assertLess(text.index("cobbles"), text.index("Shot from"))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "a.scene.json")
            sc.save(s, path)
            back, problems = sc.load(path)
        self.assertEqual(back["enrich"]["added"], s["enrich"]["added"])
        self.assertEqual(problems, [])
        old, _ = sc.clean_scene({"objects": [], "enrich": {"added": [3, "  ", "x" * 999]}})
        self.assertEqual(old["enrich"]["added"], ["x" * sc.ENRICH_CHARS])


if __name__ == "__main__":
    unittest.main()
