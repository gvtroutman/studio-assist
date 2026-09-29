"""Help > Ideas for updates: a list of what to build or fix next in the
app itself, kept in `ideas.json` (core/ideas.py). A mixin, like the other
windows split out of core/chat.py (docs/CODEMAP.md)."""
import os
import tkinter as tk

import core.ideas as ideas


class ChatIdeasMixin:
    def _ideas_window(self):
        """Type an idea, press Enter. Each open one can be marked done or
        dropped; dropping asks why, in a field under the line, because the why
        is what stops the same idea coming back. Copy puts the whole list on
        the clipboard as Markdown, for a Claude Code session."""
        key = "ideas"
        win = self.windows.get(key)
        if win is not None and win.winfo_exists():
            win.deiconify()
            win.lift()
            self.ideas_entry.focus_set()
            return
        self.ideas = ideas.Ideas()
        self.ideas_dropping = None        # the id whose "why not" field is open
        win = tk.Toplevel(self)
        self.windows[key] = win
        win.title("Ideas for updates")
        win.geometry("%dx%d" % (self._px(600), self._px(560)))
        self._skin(win, bg="bg")
        pad = self._px(16)

        self._skin(tk.Label(win, text="Something to build or fix next in the app. "
                                      "Enter adds it.", font=self.f_small, anchor="w"),
                   bg="bg", fg="faint").pack(fill="x", padx=pad, pady=(self._px(12), 0))
        top = tk.Frame(win)
        self._skin(top, bg="bg")
        top.pack(fill="x", padx=pad, pady=(self._px(4), self._px(6)))
        var = tk.StringVar()
        entry = self._entry(top, var)

        def add(_ev=None):
            self._ideas_note(self.ideas.add(var.get()))
            var.set("")
            self._ideas_fill()
            return "break"
        self._button(top, "Add", add, kind="accent").pack(side="right", padx=(self._px(8), 0))
        entry.master.pack(side="left", fill="x", expand=True)
        entry.bind("<Return>", add)
        self.ideas_entry, self.ideas_add = entry, add

        foot = tk.Frame(win)
        self._skin(foot, bg="bg")
        foot.pack(side="bottom", fill="x", padx=pad, pady=(0, self._px(12)))
        self._button(foot, "Copy list", self._copy_ideas).pack(side="right")
        note = self._skin(tk.Label(foot, text="", font=self.f_small, anchor="w",
                                   justify="left", wraplength=self._px(420)),
                          bg="bg", fg="faint")
        note.pack(side="left", fill="x", expand=True)
        self.ideas_note = note

        bar = tk.Scrollbar(win, highlightthickness=0, bd=0, width=11)
        self._skin(bar, bg="bg", troughcolor="bg", activebackground="faint")
        bar.pack(side="right", fill="y")
        view = tk.Text(win, font=self.f_body, wrap="word", bd=0, padx=18, pady=6,
                       yscrollcommand=bar.set, state="disabled", cursor="arrow",
                       highlightthickness=0)
        self._skin(view, bg="bg", fg="text", selectbackground="sel")
        view.pack(side="left", fill="both", expand=True)
        bar.config(command=view.yview)
        self._ideas_tags(view)
        self._repaint_on_theme(view, lambda: self._ideas_tags(view))
        self.ideas_view = view            # for the tests
        self._ideas_note(self.ideas.problem)
        self._ideas_fill()
        entry.focus_set()
        self._ideas_watch(win, self._ideas_stamp())

    def _ideas_stamp(self):
        try:
            return os.path.getmtime(self.ideas.path)
        except OSError:
            return None                   # no file yet

    def _ideas_watch(self, win, seen):
        """A tab's model can add an idea while the window is open, from its
        worker. Every two seconds the file's stamp is compared, and the list
        redrawn when it moved - but not while a "why not" field is open, which
        a redraw would take away mid-sentence."""
        if self.closing or not win.winfo_exists():
            return
        stamp = self._ideas_stamp()
        if stamp != seen:
            if self.ideas_dropping is not None:
                stamp = seen              # redrawn once the field closes
            else:
                self._ideas_note(self.ideas.load())
                self._ideas_fill()
        win.after(2000, lambda: self._ideas_watch(win, stamp))

    def _ideas_note(self, problem, said=None):
        """The foot of the window says where the list lives, or what went wrong
        keeping it - a failed save must not look like a kept idea."""
        note = getattr(self, "ideas_note", None)
        if note is None or not note.winfo_exists():
            return
        note.config(text=problem or said or "Kept in %s" % self.ideas.path)
        self._skin(note, bg="bg", fg="err" if problem else "faint")

    def _ideas_fill(self):
        """Draw the list again from `self.ideas`. Deleting the text destroys
        the buttons embedded in it, so each redraw starts clean."""
        view = getattr(self, "ideas_view", None)
        if view is None or not view.winfo_exists():
            return
        view.config(state="normal")
        view.delete("1.0", "end")
        opened = self.ideas.with_status("open")
        view.insert("end", "Open  ·  %d\n" % len(opened), "group")
        if not opened:
            view.insert("end", "Nothing open. Type an idea above and press Enter.\n", "desc")
        for item in opened:
            self._ideas_line(view, item, (("done", "done"), ("drop", None)))
            if item["id"] == self.ideas_dropping:
                self._ideas_why(view, item)
        for status, head in (("done", "Done"), ("dropped", "Decided against")):
            items = self.ideas.with_status(status)
            if not items:
                continue
            view.insert("end", "%s  ·  %d\n" % (head, len(items)), "group")
            for item in reversed(items):  # the latest closed first
                self._ideas_line(view, item, (("reopen", "open"), ("delete", "delete")))
        view.config(state="disabled")

    def _ideas_tags(self, view):
        """Tag colours are copied out of the palette, so the window re-reads
        them on a theme switch (`_repaint_on_theme`)."""
        self._tool_tags(view)             # "group" (the headings) and "desc"
        view.tag_configure("idea", foreground=self.C["text"], font=self.f_body,
                           lmargin1=8, lmargin2=8, rmargin=12, spacing1=10)
        view.tag_configure("closed", foreground=self.C["faint"], font=self.f_body,
                           lmargin1=8, lmargin2=8, rmargin=12, spacing1=10)
        view.tag_configure("meta", foreground=self.C["muted"], font=self.f_small,
                           lmargin1=8, spacing3=4)

    def _ideas_line(self, view, item, actions):
        """The idea on its own line, so a long one wraps under itself; the
        dates and the buttons on the line below."""
        view.insert("end", item["text"] + "\n", "idea" if item["status"] == "open" else "closed")
        meta = "added %sby %s" % (item["added"] + " " if item["added"] else "",
                                  self.ideas.who(item))
        if item["closed"]:
            meta += "%s%s %s" % ("  ·  " if meta else "",
                                 "done" if item["status"] == "done" else "dropped",
                                 item["closed"])
        if item["why"]:
            meta += "%swhy not: %s" % ("  ·  " if meta else "", item["why"])
        view.insert("end", meta + "   ", "meta")
        for label, status in actions:
            view.window_create("end", window=self._ideas_button(view, label, item["id"], status),
                               padx=self._px(2), align="center")
        view.insert("end", "\n", "meta")

    def _ideas_button(self, parent, label, idea_id, status):
        def act():
            if status is None:            # drop: ask why first, under the line
                self.ideas_dropping = idea_id
            elif status == "delete":
                self._ideas_note(self.ideas.remove(idea_id))
            else:
                self._ideas_note(self.ideas.mark(idea_id, status))
            self._ideas_fill()
        return self._button(parent, label, act, kind="ghost", bg="bg",
                            font=self.f_small, padx=self._px(9),
                            pady=self._px(1), r=self._px(8))

    def _ideas_why(self, view, item):
        """An inline "why not?" under an idea being dropped. Empty is allowed;
        a why is better, it is what the next person proposing it will read."""
        row = tk.Frame(view)
        self._skin(row, bg="bg")
        var = tk.StringVar()
        entry = self._entry(row, var)

        def drop(_ev=None):
            self.ideas_dropping = None
            self._ideas_note(self.ideas.mark(item["id"], "dropped", var.get()))
            self._ideas_fill()
            return "break"

        def cancel(_ev=None):
            self.ideas_dropping = None
            self._ideas_fill()
            return "break"
        self._button(row, "Cancel", cancel, font=self.f_small).pack(side="right", padx=(self._px(6), 0))
        self._button(row, "Drop", drop, kind="accent", font=self.f_small).pack(side="right", padx=(self._px(6), 0))
        entry.master.config(width=self._px(320))
        entry.master.pack(side="left", fill="x", expand=True)
        entry.bind("<Return>", drop)
        entry.bind("<Escape>", cancel)
        view.insert("end", "Why not? (optional)\n", "meta")
        view.window_create("end", window=row, padx=self._px(8))
        view.insert("end", "\n", "meta")
        self.ideas_why, self.ideas_drop = entry, drop     # for the tests
        entry.after_idle(entry.focus_set)

    def _copy_ideas(self):
        """The list as Markdown on the clipboard, to paste into a session."""
        self.clipboard_clear()
        self.clipboard_append(self.ideas.as_text())
        n = len(self.ideas.with_status("open"))
        self._ideas_note(None, "Copied, with %d open idea%s." % (n, "" if n == 1 else "s"))
