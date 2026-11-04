"""Character sources across scenes and finishing passes; no network or GPU."""
import copy
import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import apps.image_studio.facefusion as ff
import apps.image_studio.imagegen as ig
import apps.image_studio.wear as wear
from apps.image_studio.scene import scene as sc
from test_imagegen import FakeClient, PNG, TempStudioMixin, settle
from test_wear import WornClient


class TestSceneReferences(unittest.TestCase):
    def scene(self):
        scene = sc.new_scene("A workshop")
        chars = {}
        for key, x, shirt in (("ada", -.6, "red shirt"), ("ben", .6, "blue shirt")):
            chars[key] = {"id": key, "name": key.title(), "looks": {"subject": "a person"},
                          "item_refs": {shirt: key + ".png", "green coat": "coat.png"}}
            obj = sc.new_object("person", scene["objects"])
            obj.update(character=key, name=key.title(), position=[x, 0, 0], look={"top": shirt})
            scene["objects"].append(obj)
        return scene, chars

    def test_each_characters_item_picture_stays_paired_across_scenes(self):
        scene, chars = self.scene()
        original = copy.deepcopy(scene)
        for detail in ("A workshop", "A garden"):
            scene["details"] = detail
            _, extra = sc.generation(scene, {}, chars)
            targets = extra["scene_wear"]
            self.assertEqual([p["items"] for p in targets], [
                [{"name": "red shirt", "path": "ada.png"}],
                [{"name": "blue shirt", "path": "ben.png"}]])
            faces = {p["id"]: p for p in extra["scene_faces"]["people"]}
            for p in targets:
                self.assertEqual(p["region"], faces[p["person_id"]]["region"])
            self.assertEqual(extra["item_refs"], {})
            self.assertEqual(extra["wearing"], [])
        self.assertEqual(scene["objects"], original["objects"])
        chars["ada"]["item_refs"]["red shirt"] = "new.png"
        self.assertEqual(targets[0]["items"][0]["path"], "ada.png")  # frozen job settings

    def test_scene_outfit_overrides_do_not_restore_old_character_clothes(self):
        scene, chars = self.scene()
        scene["objects"][0]["look"]["top"] = "green coat"
        chars["ada"]["wearing"] = [{"name": "red shirt", "path": "wearing.png"}]
        _, extra = sc.generation(scene, {}, chars)
        self.assertEqual(extra["scene_wear"][0]["items"], [{"name": "green coat", "path": "coat.png"}])
        scene["objects"][0]["look"]["top"] = "red shirt"
        _, extra = sc.generation(scene, {}, chars)
        self.assertEqual(extra["scene_wear"][0]["items"][0]["path"], "wearing.png")

    def test_another_person_and_shared_scene_words_do_not_activate_a_tag(self):
        scene, chars = self.scene()
        scene["details"] = "A green coat hangs on the wall"
        chars["ada"]["item_refs"]["blue shirt"] = "wrong.png"
        _, extra = sc.generation(scene, {}, chars)
        self.assertEqual(extra["scene_wear"][0]["items"], [{"name": "red shirt", "path": "ada.png"}])

    def test_out_of_frame_people_have_no_item_target(self):
        scene, chars = self.scene()
        scene["objects"][1]["position"] = [40, 0, 0]
        _, extra = sc.generation(scene, {}, chars)
        self.assertEqual([p["name"] for p in extra["scene_wear"]], ["Ada"])
        scene["objects"] = []
        _, extra = sc.generation(scene, {}, chars)
        self.assertEqual(extra["scene_wear"], [])


class TestTargetedItems(unittest.TestCase):
    boxes = [(100, 50, 250, 800, "person"), (500, 50, 400, 900, "person"),
             (160, 80, 100, 120, "face"), (640, 80, 100, 120, "face"),
             (120, 220, 210, 320, "shirt"), (550, 220, 300, 350, "shirt")]

    def item(self):
        return {"name": "red shirt", "path": "red.png", "target_region": [.1, .05, .35, .25]}

    def test_scene_region_selects_the_smaller_person(self):
        self.assertEqual(wear.where(1000, 1000, self.item(), self.boxes), (120, 220, 210, 320))
        planned, missed = wear.plan(1000, 1000, [self.item()], self.boxes)
        self.assertEqual(missed, [])
        self.assertEqual(planned[0]["bounds"], self.boxes[0][:4])
        self.assertEqual(planned[0]["exclude"], [self.boxes[1][:4]])

    def test_missing_ambiguous_and_invalid_targets_never_choose_the_largest(self):
        for region in ([.3, .3, .4, .4], [0, 0, 1, 1], [1, 0, 0, 1], None):
            with self.subTest(region=region):
                item = dict(self.item(), target_region=region)
                self.assertIsNone(wear.where(1000, 1000, item, self.boxes))
                planned, missed = wear.plan(1000, 1000, [item], self.boxes)
                self.assertEqual(planned, [])
                self.assertEqual(missed, ["red shirt"])

    def test_overlapping_person_detections_are_skipped(self):
        boxes = self.boxes + [(140, 40, 250, 900, "person")]
        self.assertIsNone(wear.where(1000, 1000, self.item(), boxes))

    def test_missing_garment_uses_this_person_body(self):
        boxes = [b for b in self.boxes if b[4] != "shirt"]
        x, y, w, h = wear.where(1000, 1000, self.item(), boxes)
        self.assertEqual((x, w), (100, 250))
        self.assertGreater(y, 80)

    def test_final_mask_cannot_spill_into_another_person_after_feathering(self):
        item = dict(self.item(), noun="shirt", place="body", picture="red.png",
                    crop={"x": 0, "y": 0, "width": 10, "height": 10},
                    bounds=(1.2, 1.2, 6.6, 6.6), exclude=[(5.2, 0, 4.8, 10)])
        graph = wear.item_graph("base.png", [item], 7, "test", "sam3.pt", (10, 10))

        def mask(link):
            # Worst case: the blurred item mask covers the whole crop. Evaluate
            # the stock mask operations after it; ownership still bounds edits.
            if link == ["i1_b2", 0]:
                return [[1.0] * 10 for _ in range(10)]
            node = graph[link[0]]
            p = node["inputs"]
            if node["class_type"] == "SolidMask":
                return [[p["value"]] * p["width"] for _ in range(p["height"])]
            dest, src = mask(p["destination"]), mask(p["source"])
            for y, row in enumerate(src):
                for x, value in enumerate(row):
                    dy, dx = y + p["y"], x + p["x"]
                    if 0 <= dy < len(dest) and 0 <= dx < len(dest[0]):
                        old = dest[dy][dx]
                        dest[dy][dx] = {"or": max(old, value), "multiply": old * value,
                                        "subtract": max(0, old - value)}[p["operation"]]
            return dest

        result = mask(graph["i1_put"]["inputs"]["mask"])
        changed = {(x, y) for y, row in enumerate(result) for x, value in enumerate(row) if value}
        self.assertEqual(changed, {(x, y) for x in range(2, 5) for y in range(2, 7)})


class TestCharacterPipeline(TempStudioMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.shirt = os.path.join(self.dir, "red.png")
        self.other = os.path.join(self.dir, "blue.png")
        for path in (self.shirt, self.other):
            Path(path).write_bytes(PNG)

    def settings(self):
        return dict(ig.default_settings(), model="z-image-turbo", backend="5090",
                    scene="Two people wearing shirts", face_detail=False, hand_pass=False,
                    auto_refine=False, scene_wear=[
                        {"person_id": "ada", "name": "Ada", "region": [.1, .05, .35, .25],
                         "items": [{"name": "red shirt", "path": self.shirt}]},
                        {"person_id": "ben", "name": "Ben", "region": [.5, .05, .9, .25],
                         "items": [{"name": "blue shirt", "path": self.other}]}])

    def test_scene_pictures_reach_the_correct_crops_and_history(self):
        self.studio.client_factory = SceneWearClient
        jobs = self.studio.submit(self.settings())
        settle(jobs)
        job = jobs[0]
        self.assertEqual(job.status, "complete", job.detail)
        client = FakeClient.instances[-1]
        graph = next(g for g in client.graphs if "i1_ks" in g)
        self.assertEqual(graph["i1_photo"]["inputs"]["image"], "studio_red.png")
        self.assertEqual(graph["i2_photo"]["inputs"]["image"], "studio_blue.png")
        self.assertLess(graph["i1_cut"]["inputs"]["crop_region"]["x"],
                        graph["i2_cut"]["inputs"]["crop_region"]["x"])
        self.assertIn("i1_owned", graph)
        self.assertIn("i2_owned", graph)
        self.assertIn(("items", "Item pass"), ig.pipeline_stages(self.studio.lib, job.settings))
        self.assertIn(self.shirt, job.record["references"].values())
        self.assertIn(self.other, job.record["references"].values())
        self.assertEqual(job.record["settings"]["scene_wear"], self.settings()["scene_wear"])

    def test_missing_picture_or_backend_is_reported_and_other_targets_survive(self):
        s = self.settings()
        Path(self.shirt).unlink()
        inventory = SceneWearClient(self.backend("5090")).inventory()
        plan = ig.compose(s, self.studio.lib, self.backend("5090"), inventory,
                          nodes=SceneWearClient(self.backend("5090")).node_types())
        self.assertEqual([p["name"] for p in plan.scene_wear], ["Ben"])
        self.assertTrue(any("Ada's picture" in w for w in plan.warnings))
        plan = ig.compose(s, self.studio.lib, self.backend("5090"),
                          FakeClient(self.backend("5090")).inventory())
        self.assertEqual(plan.scene_wear, [])
        self.assertTrue(any("Ben's item pictures" in w and wear.KLEIN in w for w in plan.warnings))

    def test_missing_target_keeps_the_picture_and_records_no_item_pass(self):
        class NoFaces(SceneWearClient):
            faces = []
        self.studio.client_factory = NoFaces
        jobs = self.studio.submit(self.settings())
        settle(jobs)
        self.assertEqual(jobs[0].status, "complete", jobs[0].detail)
        self.assertFalse([g for g in FakeClient.instances[-1].graphs if "i1_ks" in g])
        self.assertNotIn("Item pass", [p["label"] for p in jobs[0].record["passes"]])
        self.assertTrue(any("no unambiguous person" in n for n in jobs[0].record["notes"]))

    def test_a_fix_clears_scene_item_passes(self):
        self.assertEqual(ig.Studio.fix_base(self.settings())["scene_wear"], [])

    def test_final_face_sources_follow_scene_and_form_snapshots(self):
        self.studio.lib.save("identities", [{"id": "person", "name": "Person",
                                             "references": [self.other]}])
        person = {"id": "ada", "identity": "person", "face": self.shirt,
                  "photos": [self.shirt, self.other], "region": [.1, .05, .35, .25]}
        selected = ff.selected(self.studio.lib, {"scene_faces": {"people": [person]}})
        self.assertEqual(selected[0]["references"], [self.shirt, self.other])
        self.assertEqual(selected[0]["person_id"], "ada")
        self.assertEqual(selected[0]["target_region"], person["region"])
        # A form's captured character photos win over a subsequently edited profile.
        selected = ff.selected(self.studio.lib, {"identities": ["person"],
                                               "face_photos": [self.shirt]})
        self.assertEqual(selected[0]["references"], [self.shirt])
        self.assertEqual(self.studio.lib.get("identities", "person")["references"], [self.other])

    def test_legacy_scenes_keep_profile_sources_and_multiple_profiles_do_not_share_a_snapshot(self):
        self.studio.lib.save("identities", [
            {"id": "ada", "name": "Ada", "references": [self.shirt]},
            {"id": "ben", "name": "Ben", "references": [self.other]}])
        selected = ff.selected(self.studio.lib, {"scene_faces": {"people": [
            {"identity": "ada", "region": [.1, .05, .35, .25]}]}})
        self.assertEqual(selected[0]["references"], [self.shirt])
        selected = ff.selected(self.studio.lib, {"identities": ["ada", "ben"],
                                               "face_photos": ["unassigned.png"]})
        self.assertEqual([p["references"] for p in selected], [[self.shirt], [self.other]])

    def test_saved_form_photos_still_apply_after_live_profile_photos_are_removed(self):
        self.studio.lib.save("identities", [{"id": "ada", "name": "Ada", "references": [],
                                             "lora": "trained-face"}])
        selected = ff.selected(self.studio.lib, {"identities": ["ada"],
                                               "face_photos": [self.shirt]})
        self.assertEqual(selected[0]["references"], [self.shirt])

    def test_disabling_one_of_two_profiles_does_not_assign_an_unbound_snapshot(self):
        self.studio.lib.save("identities", [
            {"id": "ada", "name": "Ada", "references": [self.shirt]},
            {"id": "ben", "name": "Ben", "references": [self.other], "face_swap": False}])
        selected = ff.selected(self.studio.lib, {"identities": ["ada", "ben"],
                                               "face_photos": ["unassigned.png"]})
        self.assertEqual(selected[0]["references"], [self.shirt])


class SceneWearClient(WornClient):
    faces = [(160, 80, 100, 120), (640, 80, 100, 120)]

    def listen_for_progress(self, pid, on_event, stop=None, timeout=0):
        graph = self.graphs[int(pid[3:]) - 1]
        if "p0d" in graph and "i1_ks" not in graph:
            found = {"shirt": [(120, 220, 210, 320), (550, 220, 300, 350)],
                     "person": [(100, 50, 250, 800), (500, 50, 400, 900)], "face": self.faces}
            outputs = {"7": {"text": ["1000"]}, "8": {"text": ["1000"]}}
            for key in (k for k in graph if k.endswith("t") and k.startswith("p")):
                word = graph[key]["inputs"]["text"].split(":")[0]
                outputs[key[:-1] + "v"] = {"text": [json.dumps([[
                    {"x": x, "y": y, "width": w, "height": h}
                    for x, y, w, h in found.get(word, [])]])]}
            return {"status": {"completed": True}, "outputs": outputs}
        return super().listen_for_progress(pid, on_event, stop, timeout)


if __name__ == "__main__":
    unittest.main()
