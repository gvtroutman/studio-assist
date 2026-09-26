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
        self.assertEqual({rgb for _, _, rgb in sc.painted_pieces(o)}, {own})

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

    def test_the_reference_is_named_by_its_content(self):
        s = staged("person")
        d = tempfile.mkdtemp()
        a = sc.write_reference(s, d)
        self.assertEqual(sc.write_reference(s, d), a)
        s["objects"][0]["position"][0] = 1.0
        self.assertNotEqual(sc.write_reference(s, d), a)


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


class TestSceneFile(unittest.TestCase):
    def test_save_and_open_keep_everything(self):
        s = staged("person", "box")
        s["objects"][0]["description"] = "kneeling, welding; helmet down, gloves on"
        s["objects"][0]["pose"] = {"preset": "kneeling",
                                   "controls": sc.pose_controls("kneeling")}
        s["objects"][1].update(name="Workbench", scale=[1.8, 0.9, 0.8], colour="#8a6a4a")
        s["camera"].update(yaw=30.0, lens=50.0)
        s["frame"], s["redraw"] = "landscape", 0.62
        path = os.path.join(tempfile.mkdtemp(), "shop.scene.json")
        sc.save(s, path)
        back, problems = sc.load(path)
        self.assertEqual(problems, [])
        self.assertEqual(back, sc.clean_scene(s)[0])
        self.assertEqual(back["objects"][0]["description"],
                         "kneeling, welding; helmet down, gloves on")

    def test_a_damaged_file_opens_with_what_can_be_read(self):
        s, problems = sc.clean_scene({
            "frame": "huge", "redraw": "lots", "camera": {"lens": -5, "pitch": 400},
            "objects": [{"asset": "spaceship"}, "junk",
                        {"asset": "person", "id": "p", "position": [1, "x", 2],
                         "pose": {"preset": "flying", "controls": {"arm_l_raise": 999}}},
                        {"asset": "box", "id": "p", "colour": "red"}]})
        self.assertEqual(s["frame"], "portrait")
        self.assertEqual(s["redraw"], sc.REDRAW)
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
        self.assertIn(": Kneeling to weld; face shield DOWN, hi-vis vest, gloves.", words.text)
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
        self.assertIn("Welder (a person, centre of frame, facing the camera): a man, in their "
                      "40s, tall, full beard, determined expression, wearing hi-vis vest. "
                      "grinding a seam; face shield DOWN.", text)
        self.assertIn("Apprentice (a person, ", text)
        self.assertIn("): a young woman, auburn hair in a ponytail.", text)
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

    def test_people_in_the_scene_blank_the_forms_person(self):
        s = staged("person")
        s["objects"][0].update(character="ada", look={"subject": "a woman"})
        chars = {"ada": {"id": "ada", "identity": "ada-face", "item_refs": {"x": "/y.png"}}}
        _, _, extra = sc.generation(s, "/x/ref.png", chars)
        self.assertEqual((extra["subject"], extra["hair"], extra["weight"],
                          extra["character"], extra["item_refs"]), ("", "", 0, "", {}))
        self.assertEqual(extra["scene_identities"], ["ada-face"])
        _, _, props = sc.generation(staged("box"), "/x/ref.png")
        self.assertNotIn("subject", props)                   # no people: the form's stays
        self.assertNotIn("scene_identities", props)

    def test_generation_carries_size_strength_and_the_scene(self):
        s = staged("person")
        s["frame"], s["redraw"] = "landscape", 0.55
        words, ref, extra = sc.generation(s, "/x/ref.png")
        self.assertEqual((extra["width"], extra["height"], extra["denoise"]), (1344, 768, 0.55))
        self.assertEqual(extra["scene_layout"]["objects"][0]["id"], "person")
        s["objects"][0]["name"] = "changed"
        self.assertEqual(extra["scene_layout"]["objects"][0]["name"], "Person")


class TestIntoCompose(TempStudioMixin, unittest.TestCase):
    """The frame and the words through the Image Studio's own compose()."""

    def settings(self, model):
        s = staged("person")
        s["objects"][0]["description"] = "checking a gauge, hard hat and hi-vis on"
        ref = sc.write_reference(s, tempfile.mkdtemp())
        words, ref, extra = sc.generation(s, ref)
        st = dict(ig.default_settings(), model=model, scene=words.text,
                  references={"source": ref}, **extra)
        return st, ref

    def test_a_workflow_with_a_source_input_takes_the_frame(self):
        st, ref = self.settings("z-image-turbo")
        plan = ig.compose(st, self.studio.lib, self.backend("3090"), FLUX_FILES)
        self.assertEqual(plan.errors, [])
        self.assertEqual(plan.references.get("source"), ref)
        self.assertEqual((plan.values["width"], plan.values["height"]), (896, 1152))
        self.assertEqual(plan.values["denoise"], sc.REDRAW)
        self.assertIn("checking a gauge, hard hat and hi-vis on", plan.prompt)
        self.assertIn(ig.anatomy_text(), plan.prompt)

    def test_the_forms_person_is_not_said_twice(self):
        s = staged("person")
        s["objects"][0]["look"] = {"subject": "an older man", "hair": "grey"}
        ref = sc.write_reference(s, tempfile.mkdtemp())
        words, ref, extra = sc.generation(s, ref)
        st = dict(ig.default_settings(), model="z-image-turbo", scene=words.text,
                  references={"source": ref}, subject="a woman", hair="auburn")
        st.update(extra)
        plan = ig.compose(st, self.studio.lib, self.backend("3090"), FLUX_FILES)
        self.assertIn("an older man, grey hair", plan.prompt)
        self.assertNotIn("a woman", plan.prompt)
        self.assertNotIn("auburn", plan.prompt)

    def test_a_workflow_without_one_says_the_frame_goes_unused(self):
        st, _ = self.settings("flux-dev")
        plan = ig.compose(st, self.studio.lib, self.backend("5090"), FLUX_FILES)
        self.assertNotIn("source", plan.references)
        self.assertTrue(any("Source image" in w for w in plan.warnings), plan.warnings)

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

        ui.settings["model"] = "flux-dev"                   # takes no source picture
        self.assertIn("frame would not be used", sb.check())
        self.assertFalse(sb.generate())

        sb._set_model("z-image-turbo")
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
        self.assertEqual((st["width"], st["height"], st["denoise"]), (896, 1152, sc.REDRAW))
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


if __name__ == "__main__":
    unittest.main()
