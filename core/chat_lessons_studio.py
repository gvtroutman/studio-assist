"""The studio brief editor and the per-app lessons window. Split out
of core/chat.py (docs/CODEMAP.md) as a mixin."""
import os
import tkinter as tk

import core.agent as eng


class ChatLessonsMixin:
    def _studio_path(self):
        return eng.studio_brief_path(self._data_dir())

    def _studio_window(self):
        """The studio brief, in an editor. Saved, it is the last part of every
        tab's briefing: a tab whose conversation has not started takes it at
        once, the rest on their next New chat - rewriting a prompt mid-way
        would throw away the host's cached prefix and the transcript's sense."""
        key = "studio"
        win = self.windows.get(key)
        if win is not None and win.winfo_exists():
            win.deiconify()
            win.lift()
            return
        win = tk.Toplevel(self)
        self.windows[key] = win
        win.title("About this studio")
        win.geometry("%dx%d" % (self._px(640), self._px(560)))
        self._skin(win, bg="bg")
        self._skin(tk.Label(win, text="What every tab is told about this studio before any "
                                      "task. Plain text or Markdown; saved to %s."
                                      % self._studio_path(),
                            font=self.f_small, anchor="w", justify="left",
                            wraplength=self._px(600)),
                   bg="bg", fg="faint").pack(fill="x", padx=self._px(16), pady=(self._px(12), 0))
        bar = tk.Scrollbar(win, highlightthickness=0, bd=0, width=11)
        self._skin(bar, bg="bg", troughcolor="bg", activebackground="faint")
        bar.pack(side="right", fill="y")
        editor = tk.Text(win, font=self.f_body, wrap="word", bd=0, padx=14, pady=12,
                         undo=True, yscrollcommand=bar.set, highlightthickness=0,
                         insertwidth=2)
        self._skin(editor, bg="card", fg="text", insertbackground="text", selectbackground="sel")
        editor.pack(fill="both", expand=True, padx=self._px(16), pady=self._px(12))
        bar.config(command=editor.yview)
        editor.insert("1.0", self.studio or eng.STUDIO_TEMPLATE)
        note = self._skin(tk.Label(win, text="", font=self.f_small, anchor="w"),
                          bg="bg", fg="muted")
        note.pack(fill="x", padx=self._px(16))

        def save():
            text = editor.get("1.0", "end").strip()
            if text == eng.STUDIO_TEMPLATE.strip():
                text = ""                 # the template itself is not a brief
            try:
                os.makedirs(os.path.dirname(self._studio_path()), exist_ok=True)
                with open(self._studio_path(), "w", encoding="utf-8") as f:
                    f.write(text + ("\n" if text else ""))
            except OSError as e:
                note.config(text="Could not save: %s" % e)
                return
            self.studio = text
            fresh = 0
            for s in self.sessions.values():
                s.studio = text
                if not s.busy and len(s.messages) == 1:
                    s.messages[0] = {"role": "system", "content": s.prompt()}
                    fresh += 1
            waiting = len(self.sessions) - fresh
            note.config(text="Saved. %s" % (
                "Every tab has it." if not waiting else
                "%d tab%s take%s it on their next New chat (Ctrl+N)."
                % (waiting, "" if waiting == 1 else "s", "s" if waiting == 1 else "")))
        button = self._button(win, "Save", save, kind="accent", font=self.f_body)
        button.pack(anchor="e", padx=self._px(16), pady=(0, self._px(12)))
        self.studio_editor = editor       # for the tests

    def _lessons_window(self):
        """Every lesson kept for the current tab's app, each with a way to
        forget it - a lesson learned from a bad run should not need a text
        editor to get rid of."""
        s = self.cur()
        if s is None:
            return
        if s.notebook is None:
            self._write(s, "This tab keeps no lessons.\n", "sys")
            return
        key = ("lessons", s.id)
        win = self.windows.get(key)
        if win is not None and win.winfo_exists():
            win.destroy()                 # rebuilt: a forget changed the list
        win = tk.Toplevel(self)
        self.windows[key] = win
        win.title("Lessons - %s" % s.app.name)
        win.geometry("%dx%d" % (self._px(560), self._px(480)))
        self._skin(win, bg="bg")
        bar = tk.Scrollbar(win, highlightthickness=0, bd=0, width=11)
        self._skin(bar, bg="bg", troughcolor="bg", activebackground="faint")
        bar.pack(side="right", fill="y")
        view = tk.Text(win, font=self.f_body, wrap="word", bd=0, padx=18, pady=14,
                       yscrollcommand=bar.set, state="disabled", cursor="arrow",
                       highlightthickness=0)
        self._skin(view, bg="bg", fg="text", selectbackground="sel")
        view.pack(side="left", fill="both", expand=True)
        bar.config(command=view.yview)
        self._tool_tags(view)
        view.config(state="normal")
        kept = s.notebook.ordered()
        if not kept:
            view.insert("end", "Nothing kept yet for %s.\n\n" % s.app.name, "group")
            view.insert("end", "Lessons arrive after a task: a call the validator refused, "
                               "a correction you gave, a sentence the model chose to keep "
                               "(studio_remember), or something you told it to remember "
                               "- start a message with \"remember\" or \"from now on\".\n",
                        "desc")
        else:
            view.insert("end", "%d lesson%s: those for every tab, then %s's%s, each oldest "
                               "first.\n" % (len(kept), "" if len(kept) == 1 else "s", s.app.name,
                                             ", then this folder's"
                                             if getattr(s.app, "workspace", None) else ""), "group")
            for lesson in kept:
                view.insert("end", "\n")
                view.window_create("end", window=self._forget_lesson_button(view, s, lesson["text"]))
                view.insert("end", "  " + lesson["text"] + "\n", "name")
                view.insert("end", "      %s%s%s\n" % (
                    self._layer_name(s, lesson["text"]) + "  ·  ",
                    {"user": "you said so", "trainer": "the trainer (Claude) kept it",
                     "model": "the model kept it", "review": "reflected after a task",
                     "error": "a refused call"}[lesson["source"]],
                    "  ·  came up %d more time%s" % (lesson["hits"], "" if lesson["hits"] == 1 else "s")
                    if lesson["hits"] else ""), "desc")
        view.config(state="disabled")
        self.lessons_view = view          # for the tests

    # --------------------------------------------------------- diagnostics
    def _forget_lesson_button(self, parent, s, text):
        def forget():
            if s.busy:
                return                    # the worker may be writing the notebook
            s.notebook.remove(text)
            self._publish_lessons(s)
            self._lessons_window()
        return self._button(parent, "forget", forget, kind="ghost", bg="card",
                            font=self.f_small, padx=self._px(9),
                            pady=self._px(1), r=self._px(8))

