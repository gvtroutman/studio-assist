"""Structured facial hair, visible geometry masks and generation wiring."""
import copy
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from apps.image_studio.scene import beard, scene as sc
from apps.image_studio import imagegen as ig


def staged():
    scene = sc.new_scene()
    obj = sc.new_object("person")
    obj["look"] = {"facial_hair": "full beard", "beard": dict(beard.DEFAULT, color="brown")}
    scene["objects"] = [obj]
    return scene, obj


class TestBeard(unittest.TestCase):
    def test_saved_attributes_are_bounded_and_copied(self):
        scene, obj = staged()
        obj["look"]["beard"].update(length=2, density=-1, coverage=float("nan"))
        saved, _ = sc.clean_scene(scene)
        b = saved["objects"][0]["look"]["beard"]
        self.assertEqual(b["length"], 1)
        self.assertEqual(b["density"], 0)
        self.assertEqual(b["coverage"], beard.DEFAULT["coverage"])
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "beard.scene.json")
            sc.save(saved, path)
            loaded = sc.load(path)
            self.assertEqual(loaded[0]["objects"][0]["look"]["beard"], b)
        self.assertIsNone(beard.clean({"style": "unknown"}))
        self.assertIsNone(beard.clean("short"))

    def test_structured_words_override_legacy_and_reach_face_pass(self):
        scene, obj = staged()
        said = sc.look_text(obj)
        self.assertIn("brown short beard", said)
        self.assertNotIn("full beard", said)
        self.assertEqual(ig.person_text({"facial_hair": "short beard"}), "short beard")
        obj["look"]["beard"]["style"] = "none"
        self.assertIn("clean-shaven", sc.look_text(obj))
        obj["look"]["beard"]["style"] = "short"
        self.assertIn("brown short beard", sc.face_targets(scene)[0]["words"])
        switched = sc.character_look({"looks": {"hair": "black"}}, obj["look"])
        self.assertNotIn("beard", switched)

    def test_style_bands_exclude_eyes_mouth_and_rear(self):
        b = dict(beard.DEFAULT)
        self.assertTrue(beard.contains((0, 0.02, 0.08), b))
        self.assertTrue(beard.contains((0, 0.053, 0.1), b))
        self.assertFalse(beard.contains((0, 0.038, 0.1), b))
        self.assertFalse(beard.contains((0, 0.1, 0.1), b))
        self.assertFalse(beard.contains((0, 0.02, -0.08), b))
        b["style"] = "moustache"
        self.assertFalse(beard.contains((0, 0.02, 0.08), b))
        self.assertTrue(beard.contains((0, 0.053, 0.1), b))
        b.update(style="short", coverage=0.1)
        self.assertFalse(beard.contains((0.05, 0.065, 0.08), b))
        b["coverage"] = 1
        self.assertTrue(beard.contains((0.05, 0.065, 0.08), b))

    def test_masks_are_confined_to_visible_person_heads(self):
        scene, obj = staged()
        _, _, masks = sc.beard_mask_buffers(scene, 256, 256, feather=2)
        self.assertIn(obj["id"], masks)
        inst, parts, _, _, meta = sc.id_render(scene, 256, 256)
        for i, value in enumerate(masks[obj["id"]]):
            if value:
                self.assertEqual(meta["instances"][str(inst[i])]["owner"], obj["id"])
                self.assertEqual(meta["parts"][str(inst[i])][str(parts[i])], "head")
        obj["rotation"][0] = 180
        self.assertEqual(sc.beard_mask_buffers(scene, 256, 256, feather=0)[2], {})

    def test_hidden_person_gets_no_mask_and_neighbours_do_not_share_one(self):
        scene, obj = staged()
        other = sc.new_object("person", scene["objects"])
        other["look"] = copy.deepcopy(obj["look"])
        other["position"] = [0, 0, -2]
        scene["objects"].append(other)
        _, _, masks = sc.beard_mask_buffers(scene, 256, 256, feather=0)
        self.assertNotIn(other["id"], masks)
        other["position"] = [0.6, 0, 0]
        _, _, masks = sc.beard_mask_buffers(scene, 256, 256, feather=0)
        self.assertEqual(set(masks), {obj["id"], other["id"]})
        self.assertFalse(any(a and b for a, b in zip(masks[obj["id"]], masks[other["id"]])))

    def test_single_unnamed_person_gets_regional_generation_input(self):
        scene, obj = staged()
        self.assertFalse(scene["regional_prompting"])
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(sc, "_write", side_effect=lambda data, prefix: self.write(folder, prefix, data)):
                words, extra = sc.generation(scene, {})
            region = extra["character_regions"][0]
            self.assertEqual(region["person_id"], obj["id"])
            self.assertEqual(region["kind"], "facial_hair")
            self.assertTrue(os.path.isfile(region["mask_path"]))
            self.assertIn("brown short beard", region["prompt"])
            self.assertIn("brown short beard", words.text)
            # For the beard pass to fit to the drawn face: the pose map's dots, the look.
            self.assertEqual(len(region["face_dots"]), 68)
            self.assertEqual(region["face_dots"], [[round(x, 5), round(y, 5)] for x, y in
                                                   sc.pose_figures(scene)[0]["face"]])
            self.assertEqual(region["beard"]["style"], "short")
        obj["look"].pop("beard")
        with patch.object(sc, "id_render", side_effect=AssertionError("legacy looks need no extra render")):
            self.assertEqual(sc.beard_masks(scene), {})

    @staticmethod
    def write(folder, prefix, data):
        path = os.path.join(folder, prefix + ".png")
        with open(path, "wb") as stream:
            stream.write(data)
        return path


if __name__ == "__main__":
    unittest.main()
