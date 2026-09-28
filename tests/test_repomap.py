import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import apps.opencode.repomap as repomap
import core.mcp as studio_mcp

PY = '''"""doc"""
LIMIT = 5


class Box:
    @property
    def size(self, scale):
        return 1


async def fetch(url, timeout=3):
    pass
'''
JS = "export const ReadCap = async () => {\n}\nfunction helper(a) {}\n"


def text(res):
    return res["content"][0]["text"]


class RepoMapTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = self.root = self.tmp.name
        os.makedirs(os.path.join(root, "pkg"))
        os.makedirs(os.path.join(root, "__pycache__"))
        with open(os.path.join(root, "pkg", "a.py"), "w") as f:
            f.write(PY)
        with open(os.path.join(root, "pkg", "b.js"), "w") as f:
            f.write(JS)
        with open(os.path.join(root, "__pycache__", "junk.py"), "w") as f:
            f.write("def junk(): pass\n")

    def tearDown(self):
        self.tmp.cleanup()

    def call(self, tool, **args):
        return repomap.SERVER.call_tool(tool, args)

    def test_file_outline_has_line_ranges_decorators_and_methods(self):
        out = text(self.call("map", path=os.path.join(self.root, "pkg", "a.py")))
        self.assertIn("(12 lines)", out)
        self.assertIn("const LIMIT  L2", out)
        self.assertIn("class Box  L5-8", out)
        self.assertIn("    def size(scale)  L6-8", out, "decorator line starts the span")
        self.assertIn("def fetch(url, timeout)  L11-12", out)

    def test_folder_map_skips_caches_and_honours_depth(self):
        out = text(self.call("map", path=self.root))
        self.assertIn("pkg/a.py  (12 lines)", out)
        self.assertIn("class Box", out)
        self.assertIn("def ReadCap", out)
        self.assertNotIn("size", out, "methods only at depth 2")
        self.assertNotIn("junk", out)
        self.assertIn("def size", text(self.call("map", path=self.root, depth=2)))
        self.assertNotIn("class", text(self.call("map", path=self.root, depth=0)))

    def test_find_names_file_span_and_owner(self):
        out = text(self.call("find", name="size", path=self.root))
        self.assertEqual(out, "pkg/a.py:6-8  def Box.size(scale)")
        self.assertIn("pkg/b.js:3", text(self.call("find", name="helper", path=self.root)))
        self.assertIn("No definition", text(self.call("find", name="nope", path=self.root)))

    def test_refusals_are_results(self):
        res = self.call("map", path=os.path.join(self.root, "missing"))
        self.assertTrue(res.get("isError"))

    def test_broken_python_still_maps(self):
        p = os.path.join(self.root, "bad.py")
        with open(p, "w") as f:
            f.write("def x(:\n")
        self.assertIn("does not parse", text(self.call("map", path=p)))

    def test_tool_contracts_pass_the_checker(self):
        self.assertFalse([f for f in studio_mcp.check_tools([t.spec() for t in repomap.SERVER.tools])
                          if getattr(f, "level", "") == "error"])


if __name__ == "__main__":
    unittest.main()
