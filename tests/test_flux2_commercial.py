"""The commercial FLUX.2 baseline against ComfyUI's shipped reference."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from apps.image_studio import imagegen as ig

ROOT = Path(__file__).resolve().parents[1]


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


class TestFlux2Commercial(unittest.TestCase):
    def test_reference_wiring_survives_subgraph_expansion(self):
        reference = read("tests/fixtures/flux2_klein4b_official.json")
        sub = next(s for s in reference["definitions"]["subgraphs"]
                   if s["name"] == "Text to Image (Flux.2 Klein 4B Distilled)")
        graph = read("workflows/flux2_klein4b_commercial_api.json")
        self.assertEqual(len(graph), len(sub["nodes"]) + 2)
        originals = {n["id"]: n for n in sub["nodes"]}
        for nid, node in originals.items():
            self.assertEqual(graph[str(nid + 100)]["class_type"], node["type"])
        for link in sub["links"]:
            origin, target = link["origin_id"], link["target_id"]
            if origin < 0 or target < 0:
                continue
            key = originals[target]["inputs"][link["target_slot"]]["name"]
            self.assertEqual(graph[str(target + 100)]["inputs"][key],
                             [str(origin + 100), link["origin_slot"]])
        self.assertEqual(graph["174"]["inputs"]["text"], ["76", 0])
        self.assertEqual(graph["9"]["inputs"]["images"], ["165", 0])
        self.assertEqual(graph["162"]["inputs"]["steps"], 4)
        self.assertEqual(graph["163"]["inputs"]["cfg"], 1)
        self.assertEqual(graph["176"]["class_type"], "ConditioningZeroOut")

    def test_image_studio_fills_the_same_api_graph(self):
        api = read("workflows/flux2_klein4b_commercial_api.json")
        wf = ig.load_workflow("klein4b_distilled")
        values = {"model": api["170"]["inputs"]["unet_name"],
                  "encoder": api["171"]["inputs"]["clip_name"],
                  "vae": api["172"]["inputs"]["vae_name"],
                  "prompt": api["76"]["inputs"]["value"], "seed": 0}
        self.assertEqual(ig.fill(wf, values), api)
        changed = ig.fill(wf, dict(values, seed=ig.MAX_SEED, width=896, height=1152))
        self.assertEqual(changed["173"]["inputs"]["noise_seed"], ig.MAX_SEED)
        self.assertEqual(changed["168"]["inputs"]["value"], 896)
        self.assertEqual(changed["169"]["inputs"]["value"], 1152)
        for nid in ("162", "166"):
            self.assertEqual(changed[nid]["inputs"]["width"], ["168", 0])
            self.assertEqual(changed[nid]["inputs"]["height"], ["169", 0])

    def test_importable_ui_has_every_api_edge_and_no_disabled_nodes(self):
        ui = read("workflows/flux2_klein4b_commercial.json")
        api = read("workflows/flux2_klein4b_commercial_api.json")
        nodes = {str(n["id"]): n for n in ui["nodes"]}
        self.assertEqual(set(nodes), set(api))
        for node in nodes.values():
            self.assertEqual(node["mode"], 0)
        for _lid, origin, slot, target, input_slot, _kind in ui["links"]:
            key = nodes[str(target)]["inputs"][input_slot]["name"]
            self.assertEqual(api[str(target)]["inputs"][key], [str(origin), slot])

    def test_four_billion_family_is_separate_and_commercial(self):
        model = next(m for m in ig._default_models() if m["id"] == "klein-4b-distilled")
        self.assertEqual(ig.model_license(model)[0], "commercial")
        self.assertEqual(ig.guess_family(model["values"]["model"]), "flux2-klein4b")
        self.assertEqual(ig.guess_family("flux-2-klein-base-4b.safetensors"), "flux2-klein4b")
        self.assertEqual(ig.guess_family("flux-2-klein-9b-fp8.safetensors"), "flux2-klein9b")
        self.assertEqual(ig.guess_family("flux2-dev.safetensors"), "flux2")
        self.assertEqual(ig.model_license({"family": "flux2-klein9b"})[0], "noncommercial")

    def test_reference_digest_is_pinned(self):
        wf = ig.load_workflow("klein4b_distilled")
        digest = hashlib.sha256((ROOT / "tests/fixtures/flux2_klein4b_official.json").read_bytes())
        self.assertEqual(digest.hexdigest(), wf["provenance"]["sha256"])

    def test_existing_library_gets_the_model_once(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "models.json"
            path.write_text(json.dumps([{"id": "my-model", "label": "My model",
                "family": "flux1", "workflow": "flux_dev_baseline", "values": {}}]))
            lib = ig.Library(folder)
            self.assertIsNotNone(lib.get("models", "my-model"))
            self.assertIsNotNone(lib.get("models", "klein-4b-distilled"))
            reopened = ig.Library(folder)
            self.assertEqual(sum(m["id"] == "klein-4b-distilled"
                                 for m in reopened.all("models")), 1)
            reopened.save("models", [m for m in reopened.all("models")
                                     if m["id"] != "klein-4b-distilled"])
            self.assertIsNone(ig.Library(folder).get("models", "klein-4b-distilled"))

    def test_removed_model_stays_removed_in_a_new_library(self):
        with tempfile.TemporaryDirectory() as folder:
            lib = ig.Library(folder)
            lib.save("models", [m for m in lib.all("models") if m["id"] != "klein-4b-distilled"])
            self.assertIsNone(ig.Library(folder).get("models", "klein-4b-distilled"))

    def test_missing_four_billion_model_never_uses_nine_billion(self):
        with tempfile.TemporaryDirectory() as folder:
            lib = ig.Library(folder)
            backend = lib.get("backends", "5090")
            settings = dict(ig.default_settings(), model="klein-4b-distilled",
                            scene="A ceramic mug on a table", face_detail=False,
                            hand_pass=False, auto_refine=False, real=False)
            inventory = {"diffusion_models": {"flux-2-klein-9b-fp8.safetensors"},
                         "text_encoders": {"qwen_3_4b.safetensors"},
                         "vae": {"flux2-vae.safetensors"}, "loras": set()}
            plan = ig.compose(settings, lib, backend, inventory)
            self.assertTrue(any("flux-2-klein-4b-fp8.safetensors" in e for e in plan.errors))
            inventory["diffusion_models"].add("flux-2-klein-4b-fp8.safetensors")
            plan = ig.compose(settings, lib, backend, inventory)
            self.assertEqual(plan.errors, [])
            graph = ig.fill(plan.workflow, plan.values, plan.loras)
            self.assertEqual(graph["170"]["inputs"]["unet_name"], "flux-2-klein-4b-fp8.safetensors")
            self.assertEqual(len(graph), 16)


if __name__ == "__main__":
    unittest.main()
