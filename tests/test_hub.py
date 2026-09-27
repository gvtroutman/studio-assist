"""studio_hub: Hugging Face LoRAs and GitHub ComfyUI plugins for Add-ons.
The services are canned answers; installs go to a temporary folder."""

import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import studio_hub as hub  # noqa: E402


class Answer(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def opener(table):
    def open_(req, timeout=None):
        for key, body in table.items():
            if key in req.full_url:
                return Answer(body if isinstance(body, bytes) else json.dumps(body).encode())
        raise AssertionError("unexpected " + req.full_url)
    return open_


class HuggingFace(unittest.TestCase):
    def test_search_filters_to_the_model_base(self):
        seen = []
        rows = [{"id": "a/b", "downloads": 5, "likes": 1, "tags": ["license:mit"]},
                {"id": "bad id", "downloads": 9}]

        def open_(req, timeout=None):
            seen.append(req.full_url)
            return Answer(json.dumps(rows).encode())
        cards = hub.hf_search({"family": "flux1"}, "hands", opener=open_)
        self.assertEqual([c["id"] for c in cards], ["a/b"])
        self.assertEqual(cards[0]["license"], "mit")
        self.assertIn("base_model%3Aadapter%3Ablack-forest-labs%2FFLUX.1-dev", seen[0])

    def test_pick_largest_top_level_safetensors(self):
        info = {"siblings": [{"rfilename": "x/deep.safetensors", "size": 99},
                             {"rfilename": "small.safetensors", "size": 1},
                             {"rfilename": "big.safetensors", "size": 5,
                              "lfs": {"sha256": "AB"}},
                             {"rfilename": "README.md", "size": 50}]}
        self.assertEqual(hub.hf_pick_file(info), ("big.safetensors", 5, "ab"))
        self.assertIsNone(hub.hf_pick_file({"siblings": []}))

    def test_download_checks_hash(self):
        body = b"lora bytes"
        with tempfile.TemporaryDirectory() as d:
            dest = os.path.join(d, "x.safetensors")
            with self.assertRaises(hub.HubError):
                hub._download("https://huggingface.co/f", dest, "0" * 64, 0, "",
                              lambda t: None, None, opener({"/f": body}))
            self.assertFalse(os.listdir(d))
            sha = hashlib.sha256(body).hexdigest()
            hub._download("https://huggingface.co/f", dest, sha, 0, "",
                          lambda t: None, None, opener({"/f": body}))
            with open(dest, "rb") as f:
                self.assertEqual(f.read(), body)


def zipball(files):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    return buf.getvalue()


class GitHub(unittest.TestCase):
    def comfy(self, d):
        for n in ("main.py", "folder_paths.py"):
            open(os.path.join(d, n), "w").close()
        return d

    def test_search(self):
        data = {"total_count": 1, "items": [
            {"full_name": "o/nodes", "description": "x", "stargazers_count": 3,
             "default_branch": "dev"},
            {"full_name": "o/old", "archived": True}]}
        cards, more = hub.gh_search("face", opener=opener({"search/repositories": data}))
        self.assertEqual([c["id"] for c in cards], ["o/nodes"])
        self.assertEqual((cards[0]["branch"], more), ("dev", 0))

    def test_install_unpacks_into_custom_nodes(self):
        card = {"id": "o/nodes", "name": "nodes", "branch": "main"}
        body = zipball({"o-nodes-abc/__init__.py": "X = 1\n",
                        "o-nodes-abc/sub/a.py": "", "o-nodes-abc/sub/": ""})
        with tempfile.TemporaryDirectory() as d:
            path = hub.gh_install(card, self.comfy(d), opener=opener({"zipball": body}))
            self.assertEqual(path, os.path.join(d, "custom_nodes", "nodes"))
            self.assertTrue(os.path.isfile(os.path.join(path, "sub", "a.py")))
            self.assertTrue(hub.gh_installed(d, card))
            with self.assertRaises(hub.HubError):     # never overwrites
                hub.gh_install(card, d, opener=opener({"zipball": body}))

    def test_install_refuses_escaping_paths(self):
        card = {"id": "o/evil", "name": "evil", "branch": "main"}
        body = zipball({"top/../../escape.py": ""})
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(hub.HubError):
                hub.gh_install(card, self.comfy(d), opener=opener({"zipball": body}))
            self.assertFalse(os.path.exists(os.path.join(d, "custom_nodes", "evil")))
            self.assertFalse(os.path.exists(os.path.join(d, "escape.py")))

    def test_not_comfy(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(hub.HubError):
                hub.custom_nodes(d)

    def test_folder_remembered(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(hub.load_comfy_folder(d), "")
            hub.save_comfy_folder(d, "C:/ComfyUI")
            self.assertEqual(hub.load_comfy_folder(d), "C:/ComfyUI")


if __name__ == "__main__":
    unittest.main()
