"""The Scene Builder: the rig and its poses, the camera, the frame rendered to
a PNG, scene files, the words sent with the frame, and the window driven in
process through the Image Studio against a fake ComfyUI. No network, no GPU."""

import json
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

    def test_a_workflow_without_one_says_the_frame_goes_unused(self):
        st, _ = self.settings("flux-dev")
        plan = ig.compose(st, self.studio.lib, self.backend("5090"), FLUX_FILES)
        self.assertNotIn("source", plan.references)
        self.assertTrue(any("Source image" in w for w in plan.warnings), plan.warnings)


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
        self.assertEqual(len(st["scene_layout"]["objects"]), 2)
        self.assertIn("welding a beam", ui.scene.get("1.0", "end"))   # on the form too
        rec = ui.studio.history.list()[0]
        self.assertEqual(rec["settings"]["scene_layout"]["objects"][0]["pose"]["preset"],
                         "kneeling")
        # A plain Generate from the form afterwards carries no scene.
        self.assertNotIn("scene_layout", ui.collect())


if __name__ == "__main__":
    unittest.main()
