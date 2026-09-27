"""OpenCode's Add-ons: the records, what they put in OpenCode's config, and the
three catalogs read from canned answers shaped like the live ones (the MCP
Registry's v0 servers list, npm's search, GitHub's tree and raw files, as read
on 2026-09-27). No network: `ca._get` is replaced."""
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.agent as eng              # noqa: E402
import apps.opencode.codeaddons as ca          # noqa: E402

REGISTRY = {"servers": [
    {"server": {"name": "io.github.acme/notes-mcp", "title": "Notes", "version": "1.2.0",
                "description": "Notes for agents.",
                "repository": {"url": "https://github.com/acme/notes-mcp"},
                "packages": [{"registryType": "npm", "identifier": "@acme/notes-mcp",
                              "version": "1.2.0", "transport": {"type": "stdio"},
                              "environmentVariables": [
                                  {"name": "NOTES_TOKEN", "isRequired": True, "isSecret": True,
                                   "description": "API token"},
                                  {"name": "NOTES_DIR", "default": "notes"}]},
                             {"registryType": "oci", "identifier": "acme/notes",
                              "transport": {"type": "stdio"}}],
                "remotes": [{"type": "streamable-http", "url": "https://notes.example/mcp",
                             "headers": [{"name": "Authorization", "value": "Bearer {token}",
                                          "isRequired": True, "isSecret": True}]}]},
     "_meta": {"io.modelcontextprotocol.registry/official": {"status": "active",
                                                            "isLatest": True}}},
    {"server": {"name": "io.github.acme/notes-mcp", "version": "1.1.0"},
     "_meta": {"io.modelcontextprotocol.registry/official": {"isLatest": False}}},
    {"server": {"name": "com.example/py-thing", "version": "0.3",
                "packages": [{"registryType": "pypi", "identifier": "py-thing",
                              "version": "0.3", "transport": {"type": "stdio"}}]},
     "_meta": {}},
    {"server": {"name": "com.example/needs-args", "version": "1",
                "packages": [{"registryType": "npm", "identifier": "needs-args",
                              "transport": {"type": "stdio"},
                              "packageArguments": [{"type": "positional", "isRequired": True,
                                                    "name": "dir"}]}]},
     "_meta": {}}],
    "metadata": {"nextCursor": "page2", "count": 4}}

NPM = {"total": 45, "objects": [
    {"package": {"name": "opencode-dcp", "version": "3.2.0", "description": "Prunes.",
                 "links": {"npm": "https://www.npmjs.com/package/opencode-dcp"}},
     "downloads": {"monthly": 39602}}]}

TREE = {"tree": [
    {"type": "blob", "path": "skills/pdf/SKILL.md"},
    {"type": "blob", "path": "skills/pdf/scripts/fill.py"},
    {"type": "blob", "path": "skills/pdf/../../evil.txt"},
    {"type": "blob", "path": "README.md"},
    {"type": "tree", "path": "skills/pdf"}]}

SKILL_MD = b"---\nname: pdf\ndescription: >\n  Work with PDFs.\n  Fill forms.\n---\n# PDF\n"


class Canned:
    def __init__(self):
        self.urls = []

    def __call__(self, url, headers=None, raw=False):
        self.urls.append(url)
        if url.startswith(ca.MCP_REGISTRY):
            return REGISTRY
        if url.startswith(ca.NPM_SEARCH):
            return NPM
        if url == ca.GITHUB_API + "anthropics/skills":
            return {"default_branch": "main"}
        if "/git/trees/" in url:
            return TREE
        if url.startswith(ca.GITHUB_RAW):
            return SKILL_MD if url.endswith("SKILL.md") else b"print('fill')\n"
        raise ca.AddonError("no canned answer for " + url)


class TestCodeAddons(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.real = ca._get
        self.canned = ca._get = Canned()

    def tearDown(self):
        ca._get = self.real
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_records_turn_off_and_remove(self):
        self.assertEqual(ca.load(self.dir), [])
        ca.add(self.dir, {"kind": "plugin", "name": "p", "package": "p@1"})
        ca.add(self.dir, {"kind": "plugin", "name": "p", "package": "p@2"})   # a reinstall
        rows = ca.load(self.dir)
        self.assertEqual([r["package"] for r in rows], ["p@2"])
        self.assertTrue(rows[0]["enabled"])
        ca.set_enabled(self.dir, "plugin:p", False)
        self.assertEqual(ca.config(ca.load(self.dir)), {})
        ca.set_enabled(self.dir, "plugin:p", True)
        self.assertEqual(ca.config(ca.load(self.dir)), {"plugin": ["p@2"]})
        ca.remove(self.dir, "plugin:p")
        self.assertEqual(ca.load(self.dir), [])
        with open(ca.path(self.dir), "w") as f:
            f.write("{broken")
        self.assertEqual(ca.load(self.dir), [], "an unreadable file never stops the tab")

    def test_config_makes_every_server_tool_ask(self):
        rows = [{"kind": "mcp", "name": "notes", "config": {"type": "local", "command": ["x"]}},
                {"kind": "mcp", "name": "bad name!", "config": {"type": "local"}},
                {"kind": "skill", "name": "pdf", "path": r"C:\s\pdf"}]
        cfg = ca.config(rows)
        self.assertEqual(cfg["mcp"], {"notes": {"type": "local", "command": ["x"],
                                                "enabled": True}})
        self.assertEqual(cfg["permission"], {"notes_*": "ask"})
        self.assertEqual(cfg["skills"], {"paths": [r"C:\s\pdf"]})
        # ... and in OpenCode's whole config, an add-on only adds permissions.
        full = eng.opencode_config("http://h:1/v1", "m", ["m"], addons=rows + [
            {"kind": "mcp", "name": "edit", "config": {"type": "local", "command": ["y"]}}])
        self.assertEqual(full["permission"]["notes_*"], "ask")
        self.assertEqual(full["permission"]["edit"], "ask")
        self.assertEqual(full["permission"]["external_directory"], "deny")
        self.assertIn("notes", full["mcp"])

    def test_the_registry_offers_only_the_latest_and_what_can_run_here(self):
        cards, nxt = ca.search_mcp("notes")
        self.assertEqual(nxt, "page2")
        self.assertIn("search=notes", self.canned.urls[-1])
        names = [c["registry_name"] for c in cards]
        self.assertEqual(names.count("io.github.acme/notes-mcp"), 1, "old versions are hidden")
        notes = cards[0]
        self.assertEqual(notes["name"], "notes-mcp")
        labels = [w["label"] for w in notes["ways"]]
        self.assertEqual(len(labels), 2, "npm and the remote; the container image is not offered")
        npm, remote = notes["ways"]
        self.assertEqual(npm["command"], ["npx", "-y", "@acme/notes-mcp@1.2.0"])
        self.assertEqual([f["name"] for f in npm["fields"]], ["NOTES_TOKEN", "NOTES_DIR"])
        self.assertTrue(npm["fields"][0]["secret"])
        self.assertEqual([f["name"] for f in remote["fields"]], ["token"])
        py = [c for c in cards if c["registry_name"] == "com.example/py-thing"][0]
        self.assertEqual(py["ways"][0]["command"], ["uvx", "py-thing==0.3"])
        needs = [c for c in cards if c["registry_name"] == "com.example/needs-args"][0]
        self.assertEqual(needs["ways"], [], "a required positional argument is not guessed")

    def test_an_mcp_record_carries_the_users_values(self):
        notes = ca.search_mcp()[0][0]
        npm, remote = notes["ways"]
        with self.assertRaises(ca.AddonError) as ctx:
            ca.mcp_record(notes, npm, {})
        self.assertIn("NOTES_TOKEN", str(ctx.exception))
        rec = ca.mcp_record(notes, npm, {"NOTES_TOKEN": " t0k "})
        self.assertEqual(rec["config"], {"type": "local",
                                         "command": ["npx", "-y", "@acme/notes-mcp@1.2.0"],
                                         "environment": {"NOTES_TOKEN": "t0k",
                                                         "NOTES_DIR": "notes"}})
        self.assertEqual(rec["source"], "io.github.acme/notes-mcp")
        rec = ca.mcp_record(notes, remote, {"token": "abc"})
        self.assertEqual(rec["config"], {"type": "remote", "url": "https://notes.example/mcp",
                                         "headers": {"Authorization": "Bearer abc"}})
        hand = ca.hand_mcp("my server", "npx -y thing --flag")
        self.assertEqual(hand["name"], "my-server")
        self.assertEqual(hand["config"]["command"], ["npx", "-y", "thing", "--flag"])
        self.assertEqual(ca.hand_mcp("r", "https://x/mcp")["config"],
                         {"type": "remote", "url": "https://x/mcp"})

    def test_plugins_from_npm(self):
        cards, nxt = ca.search_plugins("prune")
        self.assertIn("keywords%3Aopencode-plugin+prune", self.canned.urls[-1])
        self.assertEqual(nxt, 30)
        self.assertEqual(ca.plugin_record(cards[0])["package"], "opencode-dcp@3.2.0")
        self.assertEqual(cards[0]["downloads"], 39602)

    def test_skills_listed_downloaded_and_recycled(self):
        cards = ca.list_skills("anthropics/skills")
        self.assertEqual([c["name"] for c in cards], ["pdf"])
        self.assertEqual(ca.skill_front(cards[0]),
                         {"name": "pdf", "description": "Work with PDFs. Fill forms."})
        rec = ca.install_skill(self.dir, cards[0])
        self.assertTrue(os.path.isfile(os.path.join(rec["path"], "scripts", "fill.py")))
        self.assertFalse(os.path.exists(os.path.join(self.dir, "evil.txt")),
                         "a path that leaves the folder is not written")
        self.assertEqual(rec["description"], "Work with PDFs. Fill forms.")
        ca.add(self.dir, rec)
        sent = []
        ca.remove(self.dir, "skill:pdf", send=sent.append)
        self.assertEqual(sent, [rec["path"]], "a downloaded skill goes to the Recycle Bin")
        # A folder the user pointed at is theirs: its record goes, the folder stays.
        mine = os.path.join(self.dir, "mine")
        os.makedirs(mine)
        with open(os.path.join(mine, "SKILL.md"), "wb") as f:
            f.write(SKILL_MD)
        rec = ca.folder_skill(mine)
        self.assertFalse(rec["downloaded"])
        ca.add(self.dir, rec)
        sent.clear()
        ca.remove(self.dir, "skill:pdf", send=sent.append)
        self.assertEqual(sent, [])
        self.assertTrue(os.path.isdir(mine))
        with self.assertRaises(ca.AddonError):
            ca.folder_skill(self.dir)

    def test_front_matter_reads_plain_quoted_and_folded(self):
        self.assertEqual(ca.front_matter('---\nname: "x"\ndescription: one line\n---\n'),
                         {"name": "x", "description": "one line"})
        self.assertEqual(ca.front_matter("---\r\nname: y\r\ndescription: |\r\n  a\r\n  b\r\n---"),
                         {"name": "y", "description": "a b"})
        self.assertEqual(ca.front_matter("no front matter"), {})


class TestAddonsWindow(unittest.TestCase):
    """The window over a temp state folder, with the catalogs canned and no
    OpenCode running."""

    @classmethod
    def setUpClass(cls):
        import core.chat as studio_chat
        cls.dir = tempfile.mkdtemp()
        os.environ["STUDIO_SETTINGS"] = os.path.join(cls.dir, "settings.json")
        cls.real = (eng.installed_apps, studio_chat.Chat._boot_host, studio_chat.Chat._ensure,
                    studio_chat.Chat._read_icons, ca._get, ca.live_status)
        eng.installed_apps = lambda: list(eng.APPS)
        studio_chat.Chat._boot_host = lambda self, *a, **k: None
        studio_chat.Chat._ensure = lambda self, s: None
        studio_chat.Chat._read_icons = lambda self: None
        ca._get = Canned()
        ca.live_status = lambda url, key: None
        cls.app = studio_chat.Chat()
        cls.spec = eng.APPS_BY_ID["opencode"]
        cls.real_state = cls.spec.state_dir
        cls.spec.state_dir = os.path.join(cls.dir, "state")

    @classmethod
    def tearDownClass(cls):
        import core.chat as studio_chat
        cls.app._quit()
        cls.spec.state_dir = cls.real_state
        (eng.installed_apps, studio_chat.Chat._boot_host, studio_chat.Chat._ensure,
         studio_chat.Chat._read_icons, ca._get, ca.live_status) = cls.real
        os.environ.pop("STUDIO_SETTINGS", None)
        shutil.rmtree(cls.dir, ignore_errors=True)

    def pump(self, until, tries=100):
        import time
        for _ in range(tries):
            self.app._drain()
            self.app.update()
            if until():
                return True
            time.sleep(0.02)
        return False

    def texts(self, w):
        out = []
        for c in w.winfo_children():
            try:
                out.append(c.cget("text"))
            except Exception:
                pass
            out += self.texts(c)
        return out

    def test_the_button_is_on_opencodes_tab_only(self):
        self.app._select("opencode")
        self.app.update()
        self.assertTrue(self.app.btn_addons.winfo_ismapped())
        self.app._select(eng.APPS[0].id)
        self.app.update()
        self.assertFalse(self.app.btn_addons.winfo_ismapped())

    def test_install_from_the_catalog_then_turn_off_and_remove(self):
        self.app._select("opencode")
        self.app._open_code_addons()
        w = self.app.code_addons
        try:
            self.assertIn("None yet.", " ".join(self.texts(w.list)))
            w.pick(kind="plugin", tab="catalog")
            self.assertTrue(self.pump(lambda: "opencode-dcp" in self.texts(w.list)))
            install = [b for b in w.list.winfo_children()[0].winfo_children()[-1].winfo_children()
                       if getattr(b, "cget", None) and b.cget("text") == "Install"][0]
            install.invoke()
            self.assertEqual(ca.load(self.spec.state_dir)[0]["package"], "opencode-dcp@3.2.0")
            self.assertIn("Installed", self.texts(w.list))
            self.assertIn("It is used when OpenCode starts.", w.msg.cget("text"))
            # An MCP server that needs a key asks for it before it is filed.
            w.pick(kind="mcp", tab="catalog")
            self.assertTrue(self.pump(lambda: "Notes" in self.texts(w.list)))
            card = w.list.winfo_children()[0]
            [b for b in card.winfo_children()[-1].winfo_children()
             if b.cget("text") == "Install"][0].invoke()
            self.assertIn("NOTES_TOKEN", self.texts(card))
            self.assertEqual(len(ca.load(self.spec.state_dir)), 1, "nothing filed yet")
            # Installed: turn off, then remove - the second click, not the first.
            w.pick(kind="plugin", tab="installed")
            row = w.list.winfo_children()[0].winfo_children()[-1]
            [b for b in row.winfo_children() if b.cget("text") == "Turn off"][0].invoke()
            self.assertFalse(ca.load(self.spec.state_dir)[0]["enabled"])
            row = w.list.winfo_children()[0].winfo_children()[-1]
            [b for b in row.winfo_children() if b.cget("text") == "Remove"][0].invoke()
            self.assertEqual(len(ca.load(self.spec.state_dir)), 1)
            row = w.list.winfo_children()[0].winfo_children()[-1]
            [b for b in row.winfo_children()
             if b.cget("text") == "Click again to remove"][0].invoke()
            self.assertEqual(ca.load(self.spec.state_dir), [])
        finally:
            w.close()


if __name__ == "__main__":
    unittest.main()
