"""The Chat tab's bridge: this PC's files and the web, read-only.

No network: `urllib.request.urlopen` is replaced with a fake that answers the
URLs a test names. Files are made in a temp folder. What these prove is the
contract the chat prompt relies on - every tool reads and nothing writes, long
texts come back in windows the model can page through, secrets are refused by
name, a walk over a big tree stops and says so, a page becomes readable text,
and the whole thing runs through the harness in process with no findings.
"""
import http.client
import io
import json
import os
import shutil
import tempfile
import unittest
import urllib.error
import urllib.request
import zipfile

import studio_agent as eng
import studio_mcp
import studio_research_mcp as research
import studio_tasks as tasks


def text_of(res):
    return res["content"][0]["text"]


def parsed(res):
    return json.loads(text_of(res))


class FakeResponse(io.BytesIO):
    def __init__(self, data, ctype="text/html; charset=utf-8", url=""):
        super().__init__(data)
        self.headers = http.client.HTTPMessage()
        self.headers["Content-Type"] = ctype
        self._url = url
        self.status = 200

    def geturl(self):
        return self._url

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


class FakeWeb:
    """urlopen: a table of URL prefix -> (bytes, content-type) or an exception."""

    def __init__(self):
        self.pages = {}
        self.requests = []

    def add(self, url, data, ctype="text/html; charset=utf-8", final=None):
        self.pages[url] = (data if isinstance(data, bytes) else data.encode("utf-8"), ctype, final or url)

    def __call__(self, req, timeout=None):
        url = req.full_url
        self.requests.append(req)
        for prefix, page in self.pages.items():
            if url.startswith(prefix):
                if isinstance(page, Exception):
                    raise page
                data, ctype, final = page
                return FakeResponse(data, ctype, final)
        raise urllib.error.URLError("no route to host")


PAGE = """<!doctype html><html><head><title> Wiggle  Expressions </title>
<style>body{color:red}</style><script>var x = 1;</script></head>
<body><nav><a href="/">Home</a></nav>
<h1>Wiggle</h1><p>The wiggle expression adds <b>random</b> motion.   Two arguments.</p>
<ul><li>frequency</li><li>amplitude</li></ul>
<pre>    wiggle(2, 30);</pre>
<p>See <a href="guide.html">the guide</a> and <a href="#top">top</a> and
<a href="javascript:void(0)">nothing</a>.</p>
<script>document.write("never")</script>
</body></html>"""

DDG = """<html><body>
<div class="result"><h2 class="result__title"><a class="result__a" rel="nofollow"
 href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fwiggle&amp;rut=abc">Wiggle guide</a></h2>
<a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fwiggle">Learn the <b>wiggle</b> expression.</a></div>
<div class="result"><h2 class="result__title"><a class="result__a" href="https://direct.example.org/page">Direct</a></h2>
<div class="result__snippet">A direct link.</div></div>
</body></html>"""


class ResearchCase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.web = FakeWeb()
        self._real_open = urllib.request.urlopen
        urllib.request.urlopen = self.web
        self.addCleanup(setattr, urllib.request, "urlopen", self._real_open)

    def path(self, *parts):
        return os.path.join(self.dir, *parts)

    def write(self, rel, content, mode="w"):
        p = self.path(*rel.split("/"))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, mode, **({} if "b" in mode else {"encoding": "utf-8"})) as f:
            f.write(content)
        return p

    def call(self, name, **args):
        return research.call_tool(name, args)


class TestFolders(ResearchCase):
    def test_a_listing_is_folders_first_hidden_out_and_never_a_path_leak(self):
        self.write("b.txt", "b")
        self.write("A.md", "a")
        self.write(".hidden", "h")
        os.mkdir(self.path("sub"))
        out = parsed(self.call("list_folder", path=self.dir))
        self.assertEqual(out["path"], self.dir)
        self.assertEqual([e["name"] for e in out["entries"]], ["sub", "A.md", "b.txt"])
        self.assertEqual((out["folders"], out["files"]), (1, 2))
        self.assertEqual(out["entries"][1]["size"], "1 KB")
        self.assertIn("modified", out["entries"][0])
        out = parsed(self.call("list_folder", path=self.dir, show_hidden=True))
        self.assertIn(".hidden", [e["name"] for e in out["entries"]])

    def test_a_long_listing_is_capped_and_says_so(self):
        for i in range(research.LIST_CAP + 5):
            self.write("f%03d.txt" % i, "x")
        out = parsed(self.call("list_folder", path=self.dir))
        self.assertEqual(len(out["entries"]), research.LIST_CAP)
        self.assertEqual(out["omitted"], 5)
        self.assertIn("find_files", out["note"])

    def test_a_missing_or_private_folder_is_a_sentence(self):
        res = self.call("list_folder", path=self.path("nope"))
        self.assertTrue(res["isError"])
        self.assertIn("nothing exists", text_of(res))
        res = self.call("list_folder", path=self.path(".ssh"))
        self.assertTrue(res["isError"])
        self.assertIn("key material", text_of(res))
        res = self.call("list_folder", path=self.write("file.txt", "x"))
        self.assertTrue(res["isError"])
        self.assertIn("not a folder", text_of(res))

    def test_the_home_folder_is_the_default_and_tilde_expands(self):
        home = os.path.expanduser("~")
        self.assertEqual(research.resolve("~"), os.path.abspath(home))
        self.assertEqual(research.resolve('"%s"' % self.dir), self.dir)

    def test_a_path_the_model_double_escaped_still_finds_the_folder(self):
        """qwen3-1.7b wrote "C:\\\\Users\\\\..." in its arguments JSON for a
        folder it had been given as C:\\Users\\...: the tool saw two
        backslashes, found nothing, and the model learned a platitude from
        the error. Collapsed only when that makes something exist."""
        self.write("sub/a.txt", "a")
        doubled = self.path("sub").replace("\\", "\\\\")
        self.assertEqual(research.resolve(doubled), self.path("sub"))
        out = parsed(self.call("list_folder", path=doubled))
        self.assertEqual(out["files"], 1)
        # a path that is right is never touched, nor one that is wrong either way
        self.assertEqual(studio_mcp.local_path(self.path("sub")), self.path("sub"))
        missing = self.path("nope").replace("\\", "\\\\")
        self.assertEqual(studio_mcp.local_path(missing), missing)
        self.assertEqual(studio_mcp.local_path(None), None)


class TestFind(ResearchCase):
    def setUp(self):
        super().setUp()
        self.write("shoot/day1/A001.mp4", "")
        self.write("shoot/day2/A002.MP4", "")
        self.write("shoot/day2/notes.txt", "")
        self.write("brief final.docx", "")
        self.write("node_modules/x/y.mp4", "")
        self.write(".ssh/id_rsa.mp4", "")
        self.write("keys/server.pem", "")

    def names(self, out):
        return sorted(os.path.basename(m["path"]) for m in out["matches"])

    def test_a_pattern_is_case_insensitive_and_skips_what_it_should(self):
        out = parsed(self.call("find_files", root=self.dir, pattern="*.mp4"))
        self.assertEqual(self.names(out), ["A001.mp4", "A002.MP4"])
        self.assertNotIn("partial", out)
        self.assertTrue(all(m["kind"] == "file" and "size" in m for m in out["matches"]))

    def test_a_plain_word_matches_anywhere_in_the_name(self):
        out = parsed(self.call("find_files", root=self.dir, pattern="brief"))
        self.assertEqual(self.names(out), ["brief final.docx"])
        self.assertEqual(out["pattern"], "*brief*")
        out = parsed(self.call("find_files", root=self.dir, pattern="day"))
        self.assertEqual(self.names(out), ["day1", "day2"])
        self.assertEqual({m["kind"] for m in out["matches"]}, {"folder"})

    def test_key_material_never_matches(self):
        out = parsed(self.call("find_files", root=self.dir, pattern="*.pem"))
        self.assertEqual(out["matches"], [])
        res = self.call("find_files", root=self.path(".ssh"), pattern="*")
        self.assertTrue(res["isError"])

    def test_the_limit_stops_the_walk_and_the_answer_says_so(self):
        out = parsed(self.call("find_files", root=self.dir, pattern="*", limit=2))
        self.assertEqual(len(out["matches"]), 2)
        self.assertTrue(out["partial"])
        self.assertIn("limit of 2", out["note"])
        res = self.call("find_files", root=self.dir, pattern="*", limit=research.FIND_CAP + 1)
        self.assertTrue(res["isError"])          # the schema, not the walk, refuses

    def test_a_walk_that_runs_out_of_time_is_partial_not_hung(self):
        real, research.WALK_SECONDS = research.WALK_SECONDS, -1
        try:
            out = parsed(self.call("find_files", root=self.dir, pattern="*.txt"))
        finally:
            research.WALK_SECONDS = real
        self.assertTrue(out["partial"])
        self.assertIn("seconds", out["note"])


class TestRead(ResearchCase):
    def test_text_comes_back_whole_with_a_header(self):
        p = self.write("notes.txt", "hello\nworld\n")
        out = text_of(self.call("read_file", path=p))
        head, body = out.split("\n\n", 1)
        self.assertIn(p, head)
        self.assertIn("12 characters, complete", head)
        self.assertEqual(body, "hello\nworld\n")

    def test_a_long_file_comes_in_windows_that_say_what_to_ask_for_next(self):
        p = self.write("long.txt", "x" * 1000)
        out = text_of(self.call("read_file", path=p, max_chars=400))
        self.assertIn("characters 0-400 of 1000", out)
        self.assertIn("start=400", out)
        self.assertEqual(out.split("\n\n", 1)[1], "x" * 400)
        out = text_of(self.call("read_file", path=p, start=800, max_chars=400))
        self.assertIn("characters 800-1000 of 1000 (the end)", out)
        res = self.call("read_file", path=p, start=5000)
        self.assertTrue(res["isError"])
        self.assertIn("past the end", text_of(res))

    def test_encodings_and_boms_are_handled(self):
        p = self.write("bom.txt", b"\xef\xbb\xbfcaf\xc3\xa9", "wb")
        self.assertTrue(text_of(self.call("read_file", path=p)).endswith("café"))
        p = self.write("u16.txt", "café".encode("utf-16"), "wb")
        self.assertTrue(text_of(self.call("read_file", path=p)).endswith("café"))
        p = self.write("cp.txt", "caf\xe9".encode("cp1252"), "wb")
        self.assertTrue(text_of(self.call("read_file", path=p)).endswith("café"))

    def test_a_binary_file_is_named_not_read(self):
        p = self.write("clip.mp4", b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 100, "wb")
        res = self.call("read_file", path=p)
        self.assertTrue(res["isError"])
        self.assertIn("binary file (MP4", text_of(res))
        self.assertIn(p, text_of(res))

    def test_a_word_document_is_its_paragraphs(self):
        doc = ("<w:document xmlns:w='http://schemas.openxmlformats.org/wordprocessingml/2006/main'>"
               "<w:body><w:p><w:r><w:t>Title</w:t></w:r></w:p>"
               "<w:p><w:r><w:t xml:space='preserve'>One </w:t></w:r><w:r><w:t>two</w:t></w:r>"
               "<w:r><w:tab/><w:t>three</w:t></w:r></w:p></w:body></w:document>")
        p = self.path("brief.docx")
        with zipfile.ZipFile(p, "w") as z:
            z.writestr("word/document.xml", doc)
        out = text_of(self.call("read_file", path=p))
        self.assertTrue(out.endswith("Title\nOne two\tthree"))
        p = self.write("bad.docx", "not a zip")
        res = self.call("read_file", path=p)
        self.assertTrue(res["isError"])
        self.assertIn("Word document", text_of(res))

    def test_refusals_are_sentences(self):
        for path, expect in ((self.path("nope.txt"), "Nothing exists"),
                             (self.dir, "list_folder"),
                             (self.path("secrets", "id.pem"), "key material"),
                             (self.path(".aws", "credentials"), "key material")):
            res = self.call("read_file", path=path)
            self.assertTrue(res["isError"], path)
            self.assertIn(expect, text_of(res))
        real, research.READ_BYTES = research.READ_BYTES, 10
        try:
            res = self.call("read_file", path=self.write("big.txt", "x" * 50))
        finally:
            research.READ_BYTES = real
        self.assertTrue(res["isError"])
        self.assertIn("reads files up to", text_of(res))


class TestFetch(ResearchCase):
    def test_a_page_becomes_readable_text_with_title_and_links(self):
        self.web.add("https://example.com/ae/", PAGE)
        out = text_of(self.call("fetch_page", url="https://example.com/ae/", links=True))
        self.assertIn("URL: https://example.com/ae/", out)
        self.assertIn("Title: Wiggle Expressions", out)
        self.assertIn("# Wiggle", out)
        self.assertIn("adds random motion. Two arguments.", out)
        self.assertIn("- frequency\n- amplitude", out)
        self.assertIn("    wiggle(2, 30);", out)
        for gone in ("color:red", "var x", "never"):
            self.assertNotIn(gone, out)
        links = out.split("Links:\n", 1)[1]
        self.assertIn("- Home  https://example.com/", links)
        self.assertIn("- the guide  https://example.com/ae/guide.html", links)
        self.assertNotIn("javascript", links)
        self.assertNotIn("#top", links)
        self.assertNotIn("Links:", text_of(self.call("fetch_page", url="https://example.com/ae/")))

    def test_the_request_looks_like_a_browser_and_a_bare_host_gets_https(self):
        self.web.add("https://example.com", PAGE)
        self.call("fetch_page", url="example.com")
        req = self.web.requests[-1]
        self.assertEqual(req.full_url, "https://example.com")
        self.assertTrue(req.get_header("User-agent", "").startswith("Mozilla/5.0"))

    def test_long_pages_page_and_truncation_is_said(self):
        self.web.add("https://example.com/long", "<p>" + "word " * 3000 + "</p>")
        out = text_of(self.call("fetch_page", url="https://example.com/long", max_chars=500))
        self.assertIn("characters 0-500 of", out)
        self.assertIn("start=500", out)
        real, research.FETCH_BYTES = research.FETCH_BYTES, 100
        try:
            out = text_of(self.call("fetch_page", url="https://example.com/long"))
        finally:
            research.FETCH_BYTES = real
        self.assertIn("the end is missing", out)

    def test_text_types_come_back_as_text_and_binaries_are_refused(self):
        self.web.add("https://api.example.com/v", '{"version": "25.1"}', "application/json")
        out = text_of(self.call("fetch_page", url="https://api.example.com/v"))
        self.assertIn("Type: application/json", out)
        self.assertIn('{"version": "25.1"}', out)
        self.web.add("https://example.com/manual.pdf", b"%PDF-1.7 ...", "application/pdf")
        res = self.call("fetch_page", url="https://example.com/manual.pdf")
        self.assertTrue(res["isError"])
        self.assertIn("application/pdf", text_of(res))
        self.assertIn("does not read", text_of(res))

    def test_the_charset_header_wins_and_the_final_url_is_reported(self):
        self.web.add("https://example.com/latin", "<p>caf\xe9</p>".encode("cp1252"),
                     "text/html; charset=windows-1252", final="https://example.com/latin/index.html")
        out = text_of(self.call("fetch_page", url="https://example.com/latin"))
        self.assertIn("café", out)
        self.assertIn("URL: https://example.com/latin/index.html", out)

    def test_failures_are_sentences_not_tracebacks(self):
        self.web.pages["https://example.com/gone"] = urllib.error.HTTPError(
            "https://example.com/gone", 404, "Not Found", None, None)
        self.web.pages["https://slow.example.com"] = urllib.error.URLError(TimeoutError())
        for url, expect in (("https://example.com/gone", "HTTP 404"),
                            ("https://slow.example.com", "did not answer within"),
                            ("https://nowhere.example.net", "Could not reach"),
                            ("ftp://example.com/x", "Only http and https"),
                            ("https://user:pw@example.com/", "credentials"),
                            ("", "URL is required")):
            res = self.call("fetch_page", url=url)
            self.assertTrue(res["isError"], url)
            self.assertIn(expect, text_of(res))


class TestSearch(ResearchCase):
    def test_results_are_title_url_and_snippet_with_redirects_unwrapped(self):
        self.web.add(research.SEARCH_URL, DDG)
        out = parsed(self.call("search_web", query="after effects wiggle", limit=5))
        self.assertEqual(out["query"], "after effects wiggle")
        self.assertEqual(out["results"], [
            {"title": "Wiggle guide", "url": "https://example.com/wiggle",
             "snippet": "Learn the wiggle expression."},
            {"title": "Direct", "url": "https://direct.example.org/page", "snippet": "A direct link."}])
        self.assertIn("not facts", out["note"])
        self.assertIn("q=after+effects+wiggle", self.web.requests[-1].full_url)

    def test_the_limit_and_an_empty_answer(self):
        self.web.add(research.SEARCH_URL, DDG)
        self.assertEqual(len(parsed(self.call("search_web", query="x", limit=1))["results"]), 1)
        self.web.add(research.SEARCH_URL, "<html><body>No results.</body></html>")
        out = parsed(self.call("search_web", query="qzxv"))
        self.assertEqual(out["results"], [])
        self.assertIn("No results", out["note"])

    def test_a_bot_check_is_named_not_parsed(self):
        self.web.add(research.SEARCH_URL, "<html><body>anomaly detected, please solve the challenge</body></html>")
        res = self.call("search_web", query="x")
        self.assertTrue(res["isError"])
        self.assertIn("refused this request as automated", text_of(res))
        res = self.call("search_web", query="   ")
        self.assertTrue(res["isError"])


class TestContract(ResearchCase):
    """What the harness, the registry and the executor hold the bridge to."""

    def test_every_tool_is_a_read_and_the_registry_names_them_all(self):
        specs = research.tool_list()
        self.assertEqual({t["name"] for t in specs}, eng.CHAT.tool_names())
        for t in specs:
            ann = t["annotations"]
            self.assertEqual((ann["readOnlyHint"], ann["destructiveHint"], ann["idempotentHint"]),
                             (True, False, True), t["name"])
            self.assertTrue(tasks.readonly(t["name"], {}, t), t["name"])
        for name in ("list_folder", "find_files", "read_file"):
            self.assertIs(next(t for t in specs if t["name"] == name)["annotations"]["openWorldHint"], False)
        for name in ("fetch_page", "search_web"):
            self.assertIs(next(t for t in specs if t["name"] == name)["annotations"]["openWorldHint"], True)

    def test_the_harness_finds_nothing_wrong_with_it(self):
        findings = studio_mcp.check_tools(research.tool_list(), groups=eng.CHAT.groups,
                                          default_groups=eng.CHAT.default_groups,
                                          prompt=eng.CHAT.system_prompt,
                                          sanitize=eng.sanitize_schema, readonly=tasks.readonly)
        self.assertEqual([f.text for f in findings if f.level in ("error", "warn")], [])

    def test_the_executor_reads_through_the_loopback_and_owes_no_read_back(self):
        """A chat turn that reads a file: the call is journaled as a read, the
        result reaches the model, and the run ends without the executor asking
        for an inspection - there was no edit to inspect."""
        p = self.write("brief.txt", "Deliver at 23.976.")
        client = eng.CHAT.connect()
        client.initialize()
        schemas = client.list_tools()
        tools = eng.to_openai_tools(schemas)
        replies = iter([
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {
                "name": "read_file", "arguments": json.dumps({"path": p})}}]},
            {"role": "assistant", "content": "The brief says 23.976."}])
        seen = []

        class LLM:
            def stream(self, messages, tools_, on_text):
                seen.append(messages)
                return next(replies)

        record = tasks.TaskRecord()
        messages = [{"role": "system", "content": eng.CHAT.chat_prompt()},
                    {"role": "user", "content": "what frame rate does the brief want"}]
        try:
            out = tasks.Executor(LLM(), client, tools, schemas, record=record).run(messages)
        finally:
            client.close()
        self.assertEqual(out, "The brief says 23.976.")
        self.assertEqual(len(seen), 2)                   # no third pass asking to inspect
        self.assertEqual(record.journal[-1]["name"], "read_file")
        self.assertTrue(record.journal[-1]["read"])
        self.assertEqual(record.journal[-1]["status"], "ok")
        self.assertIn("Deliver at 23.976.", [m for m in messages if m.get("role") == "tool"][0]["content"])
        self.assertTrue(record.status.startswith("response complete"))

    def test_the_command_line_describes_the_same_contract(self):
        real = eng.sys.stdout
        eng.sys.stdout = io.StringIO()
        try:
            self.assertEqual(studio_mcp.main(research.SERVER, ["--describe"]), 0)
            described = json.loads(eng.sys.stdout.getvalue())
        finally:
            eng.sys.stdout = real
        self.assertEqual(described["tools"], research.tool_list())
        self.assertEqual(described["initialize"]["serverInfo"]["name"], "studio-research-mcp")


if __name__ == "__main__":
    unittest.main()
