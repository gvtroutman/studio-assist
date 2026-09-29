"""The Diagnostics window: the same facts `--doctor` prints from a
console, read live and shown in the app. Split out of core/chat.py
(docs/CODEMAP.md) as a mixin."""
import tkinter as tk

import core.doctor as doctor


class ChatDiagnosticsMixin:
    def _diag_tags(self, view):
        """Tag colours are copied out of the palette, so a Diagnostics window
        left open across a theme switch has to be told again - the same
        contract as `_tool_tags`, and `_theme` calls both."""
        view.tag_configure("head", foreground=self.C["accent"], font=self.f_cap,
                           spacing1=16, spacing3=6)
        view.tag_configure("label", foreground=self.C["faint"], font=self.f_mono,
                           lmargin1=8)
        for role in ("ok", "warn", "err", "muted", "text"):
            view.tag_configure(role, foreground=self.C[role], font=self.f_mono,
                               lmargin2=26, rmargin=12, spacing3=2)

    def _diagnostics_window(self):
        """One place to look when a tab is behaving oddly, instead of three:
        is the host up, which model is actually loaded, how much of its window
        each tab's briefing has already eaten, and what went wrong last.

        The same facts `--doctor` prints from a console, because they come
        from the same `studio_doctor.report()`. The probe talks to the host
        over the network, so it runs on a worker and the window opens saying
        so rather than freezing for the length of a tailnet timeout."""
        key = ("diagnostics", "")
        win = self.windows.get(key)
        if win is not None and win.winfo_exists():
            win.lift()
        else:
            win = tk.Toplevel(self)
            self.windows[key] = win
            win.title("Diagnostics")
            win.geometry("%dx%d" % (self._px(720), self._px(560)))
            self._skin(win, bg="bg")

            foot = tk.Frame(win)
            self._skin(foot, bg="bg")
            foot.pack(side="bottom", fill="x", padx=14, pady=(0, 12))
            for label, command in (("Copy", self._copy_diagnostics),
                                   ("Check again", self._refresh_diagnostics)):
                button = self._button(foot, label, command)
                button.pack(side="right", padx=(6, 0))

            bar = tk.Scrollbar(win, highlightthickness=0, bd=0, width=11)
            self._skin(bar, bg="bg", troughcolor="bg", activebackground="faint")
            bar.pack(side="right", fill="y")
            view = tk.Text(win, font=self.f_body, wrap="word", bd=0, padx=18,
                           pady=14, yscrollcommand=bar.set, state="disabled",
                           cursor="arrow", highlightthickness=0)
            self._skin(view, bg="bg", fg="text", selectbackground="sel")
            view.pack(side="left", fill="both", expand=True)
            bar.config(command=view.yview)
            self._diag_tags(view)
            self.diag_view = view
        self._refresh_diagnostics()

    def _refresh_diagnostics(self):
        self._paint_diagnostics(None)
        # `sessions` is read on the worker, but only for numbers the UI thread
        # wrote and never mutates in place; the network is the slow part.
        tabs = [self.sessions[i] for i in self.order if i in self.sessions]
        self._spawn(None, self._probe_diagnostics, tabs)

    def _probe_diagnostics(self, tabs):
        model = self.llm.model if self.llm is not None else None
        self.q.put(("diagnostics", None, doctor.report(self.host, model, tabs)))

    def _paint_diagnostics(self, sections):
        """`sections` is None while the probe is still out."""
        view = getattr(self, "diag_view", None)
        if view is None or not view.winfo_exists():
            return
        self.diag_text = doctor.as_text(sections) if sections else ""
        view.config(state="normal")
        view.delete("1.0", "end")
        if sections is None:
            view.insert("end", "Checking…\n", "muted")
        for title, rows in sections or []:
            view.insert("end", "\n%s\n" % title, "head")
            for label, value, role in rows:
                view.insert("end", "%-16s" % label, "label")
                view.insert("end", value + "\n", role)
        view.config(state="disabled")

    def _copy_diagnostics(self):
        """So a report can be pasted somewhere it will be read, rather than
        retyped off a screen."""
        text = getattr(self, "diag_text", "")
        if not text:
            return
        self.clipboard_clear()
        self.clipboard_append(text)

