"""core/ideas.py: the list behind Help > Ideas for updates."""
import json
import os
import shutil
import tempfile
import unittest
import unittest.mock

import core.ideas as ideas


class TestIdeas(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "ideas.json")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_no_file_is_an_empty_list(self):
        book = ideas.Ideas(self.path)
        self.assertEqual(book.items, [])
        self.assertIsNone(book.problem)

    def test_an_added_idea_is_on_disk_and_blank_ones_are_not(self):
        book = ideas.Ideas(self.path)
        self.assertIsNone(book.add("  Batch  renders\nfrom History "))
        book.add("   ")
        again = ideas.Ideas(self.path)
        self.assertEqual([i["text"] for i in again.items], ["Batch renders from History"])
        self.assertEqual(again.items[0]["status"], "open")
        self.assertTrue(again.items[0]["added"])

    def test_done_dropped_and_reopened(self):
        book = ideas.Ideas(self.path)
        book.add("one")
        book.add("two")
        one, two = (i["id"] for i in book.items)
        book.mark(one, "done", "ignored for done")
        book.mark(two, "dropped", " too slow  on the 4090 ")
        again = ideas.Ideas(self.path)
        self.assertEqual([i["text"] for i in again.with_status("done")], ["one"])
        self.assertEqual(again.with_status("done")[0]["why"], "")
        self.assertEqual(again.with_status("dropped")[0]["why"], "too slow on the 4090")
        self.assertTrue(again.with_status("dropped")[0]["closed"])
        again.mark(two, "open")
        self.assertEqual(again.with_status("open")[0]["why"], "")
        self.assertEqual(again.with_status("open")[0]["closed"], "")

    def test_remove(self):
        book = ideas.Ideas(self.path)
        book.add("gone")
        book.remove(book.items[0]["id"])
        self.assertEqual(ideas.Ideas(self.path).items, [])

    def test_a_broken_file_is_set_aside_not_written_over(self):
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("{not json")
        book = ideas.Ideas(self.path)
        self.assertEqual(book.items, [])
        self.assertIn("ideas.json", book.problem)
        book.add("fresh")
        with open(self.path + ".broken", encoding="utf-8") as f:
            self.assertEqual(f.read(), "{not json")
        with open(self.path, encoding="utf-8") as f:
            self.assertEqual(json.load(f)["ideas"][0]["text"], "fresh")

    def test_a_failed_save_says_so(self):
        book = ideas.Ideas(os.path.join(self.dir, "ideas.json", "under-a-file"))
        with open(os.path.join(self.dir, "ideas.json"), "w") as f:
            f.write("x")
        self.assertIn("could not be saved", book.add("anything"))

    def test_a_models_idea_says_which_tab(self):
        book = ideas.Ideas(self.path)
        reply = book.suggest("A tool for blend modes;  run_jsx was the only way.", "After Effects")
        self.assertTrue(reply.startswith("Added"))
        item = ideas.Ideas(self.path).items[0]
        self.assertEqual((item["by"], item["tab"]), ("model", "After Effects"))
        self.assertEqual(ideas.Ideas.who(item), "the After Effects model")
        self.assertEqual(ideas.Ideas.who({"by": "model", "tab": ""}), "a model")
        self.assertEqual(ideas.Ideas.who({"by": "user"}), "you")
        self.assertIn("(from the After Effects model)", book.as_text())

    def test_a_model_is_told_about_duplicates_and_what_was_dropped(self):
        book = ideas.Ideas(self.path)
        book.add("Batch renders from History")
        book.add("A per-hand redraw pass")
        book.mark(book.items[1]["id"], "dropped", "it looked bad")
        self.assertIn("Already on the list (open)", book.suggest("batch renders from history!"))
        reply = book.suggest("A per-hand redraw pass.")
        self.assertIn("already decided against this: it looked bad", reply)
        self.assertIn("Do not suggest it again", reply)
        self.assertEqual(len(ideas.Ideas(self.path).items), 2)

    def test_two_writers_never_save_over_each_other(self):
        window = ideas.Ideas(self.path)                         # opened first
        ideas.Ideas(self.path).suggest("An idea from a tab's worker.", "Premiere")
        window.add("An idea typed in the window")               # its list is stale
        self.assertEqual([i["text"] for i in ideas.Ideas(self.path).items],
                         ["An idea from a tab's worker.", "An idea typed in the window"])
        worker = ideas.Ideas(self.path)
        window.mark(window.items[0]["id"], "done")
        worker.add("Third")
        self.assertEqual([i["status"] for i in ideas.Ideas(self.path).items],
                         ["done", "open", "open"])

    def test_a_file_that_cannot_be_opened_is_not_edited_over(self):
        book = ideas.Ideas(self.path)
        book.add("kept")
        real_open = open

        def busy(path, *a, **k):
            if path == self.path:
                raise PermissionError("in use")
            return real_open(path, *a, **k)
        with unittest.mock.patch("builtins.open", busy):
            self.assertIn("could not be read", book.add("lost?"))
        self.assertEqual([i["text"] for i in ideas.Ideas(self.path).items], ["kept"])
        self.assertFalse(os.path.exists(self.path + ".broken"))

    def test_as_text_is_markdown_by_status(self):
        book = ideas.Ideas(self.path)
        book.add("keep")
        book.add("nope")
        book.mark(book.items[1]["id"], "dropped", "overkill")
        text = book.as_text()
        self.assertIn("## Open\n- keep", text)
        self.assertIn("## Decided against\n- nope - why not: overkill", text)
        self.assertNotIn("## Done", text)


if __name__ == "__main__":
    unittest.main()
