"""The Terminal tab: console windows opened outside the app, hidden from the
desktop and mirrored here, one at a time, with a line to type into them.

`studio_consoles.py` finds, hides and reads the windows; the app's watcher
(`Chat._watch_consoles`) takes each new one about a second after it opens,
whether or not this tab is open, and opens this tab for it. What is on screen
here is a copy of the console's text, read twice a second while the tab is
looked at - the real window stays alive, hidden, whatever happens to the app.
"""

import threading
import tkinter as tk

import studio_consoles as consoles

POLL_S = 0.5                      # how often the console on screen is read
TITLE_CHARS = 32

HINT = ("Consoles opened outside the app are hidden and shown here. "
        "Show window puts one back on the desktop.")


def clip(text, n=TITLE_CHARS):
    text = " ".join(text.split()) or "Console"
    return text if len(text) <= n else text[:n - 1] + "…"


class TerminalsView:
    def __init__(self, host, session):
        self.host, self.s = host, session
        self.reader = consoles.Reader()
        self.order = []               # hwnds with a chip, in the order they came
        self.ended = set()            # hwnds of consoles that have closed, kept to read
        self.titles = {}              # hwnd -> its window title, kept past its closing
        self.lines = {}               # hwnd -> the lines last read from it
        self.current = None           # the hwnd on screen
        self.painted = []             # the lines in the Text, for `diff`
        self.last_error = None
        self.stopping = threading.Event()
        self.poller = None
        self._build(session.frame)

    # ------------------------------------------------------------------ build
    def _build(self, frame):
        h = self.host
        bar = h._skin(tk.Frame(frame), bg="bg")
        bar.pack(side="top", fill="x", padx=18, pady=(0, 6))
        self.b_show = h._button(bar, "Show window", self._show_window)
        self.b_show.pack(side="right")
        self.b_break = h._button(bar, "Ctrl+C", self._interrupt)
        self.b_break.pack(side="right", padx=(0, 6))
        self.chips = h._skin(tk.Frame(bar), bg="bg")
        self.chips.pack(side="left", fill="x", expand=True)
        self.note = tk.Label(frame, font=h.f_ui, anchor="w", text=HINT)
        h._skin(self.note, bg="bg", fg="muted")
        self.note.pack(side="top", fill="x", padx=18, pady=(0, 6))

        entry_row = h._skin(tk.Frame(frame), bg="bg")
        entry_row.pack(side="bottom", fill="x", padx=14, pady=(0, 14))
        h._button(entry_row, "Send", self._send, kind="accent").pack(side="right")
        self.entry = tk.Entry(entry_row, font=h.f_mono, bd=0, highlightthickness=1)
        h._skin(self.entry, bg="card", fg="text", insertbackground="text",
                highlightbackground="border", highlightcolor="accent")
        self.entry.pack(side="left", fill="x", expand=True, padx=(0, 8), ipady=5)
        self.entry.bind("<Return>", lambda ev: (self._send(), "break")[1])

        body = h._skin(tk.Frame(frame), bg="card")
        body.pack(side="top", fill="both", expand=True, padx=14, pady=(0, 8))
        ybar = tk.Scrollbar(body, highlightthickness=0, bd=0, width=11)
        xbar = tk.Scrollbar(body, orient="horizontal", highlightthickness=0, bd=0, width=11)
        for b in (ybar, xbar):
            h._skin(b, bg="card", troughcolor="card", activebackground="faint")
        ybar.pack(side="right", fill="y")
        xbar.pack(side="bottom", fill="x")
        self.text = tk.Text(body, font=h.f_mono, wrap="none", bd=0, padx=12, pady=10,
                            state="disabled", highlightthickness=0,
                            yscrollcommand=ybar.set, xscrollcommand=xbar.set)
        h._skin(self.text, bg="card", fg="text", selectbackground="sel",
                insertbackground="text")
        self.text.pack(side="left", fill="both", expand=True)
        ybar.config(command=self.text.yview)
        xbar.config(command=self.text.xview)
        # Typing with the mirror focused goes to the line below, where it is sent.
        self.text.bind("<Key>", self._redirect)
        self._paint_chips()

    # --------------------------------------------------------------- lifecycle
    def start(self):
        """First view of the tab (`Chat._ensure`): start reading."""
        if self.poller is None:
            self.poller = threading.Thread(target=self._poll, daemon=True)
            self.poller.start()
        self.refresh()

    def focus(self):
        self.entry.focus_set()

    def can_close(self):
        return True

    def release(self):
        """The tab is closing: every console back on the desktop."""
        self.host.holder.release_all()

    def close(self):
        self.stopping.set()
        self.reader.stop()

    # ------------------------------------------------------------------ events
    def _post(self, what, arg=None):
        self.host.q.put(("terminals", self.s.event_id, (what, arg)))

    def handle(self, payload):
        what, arg = payload
        if what == "screen":
            hwnd, lines = arg
            self.lines[hwnd] = lines
            if hwnd == self.current:
                self._paint(lines)
            if self.last_error:
                self.last_error = None
                self.say(HINT)
        elif what == "error":
            hwnd, text = arg
            if hwnd == self.current and text != self.last_error:
                self.last_error = text
                self.say(text[:1].upper() + text[1:], "err")
        elif what == "said":
            self.say(*arg)

    def say(self, text, role="muted"):
        self.note.config(text=text)
        self.host._skin(self.note, bg="bg", fg=role)

    def refresh(self, gone=()):
        """The watcher took or lost windows: chips follow `holder.held`."""
        held = self.host.holder.snapshot()
        for hwnd in gone:
            if hwnd in self.order:
                self.ended.add(hwnd)
        for hwnd, info in held.items():
            if hwnd not in self.order:
                self.order.append(hwnd)
            self.ended.discard(hwnd)
            self.titles[hwnd] = info.get("title") or self.titles.get(hwnd, "")
        for hwnd in list(self.order):     # given back from elsewhere (holding turned off)
            if hwnd not in held and hwnd not in self.ended:
                self._forget(hwnd)
        if self.current not in self.order:
            self._select(self.order[0] if self.order else None)
        elif self.current in gone:
            self._select(self.current)    # its button and note say it has closed
        else:
            self._paint_chips()

    def count(self):
        return len(self.host.holder.held)

    # ------------------------------------------------------------------ reading
    def _poll(self):
        """Off the UI thread: the console on screen, read while it is looked at."""
        while not self.stopping.wait(POLL_S):
            hwnd = self.current
            info = self.host.holder.held.get(hwnd) if hwnd is not None else None
            if info is None or self.host.active != self.s.id:
                continue
            try:
                out = self.reader.ask("read", hwnd, info["pid"])
            except (OSError, ValueError) as e:
                if not self.stopping.is_set():
                    self._post("error", (hwnd, str(e)))
                continue
            self._post("screen", (hwnd, out["lines"]))

    def _paint(self, new):
        """Only what changed: a console that printed one line, or scrolled, is
        not three thousand lines rewritten under the user's selection."""
        t = self.text
        at_end = t.yview()[1] >= 0.999
        drop, keep, tail = consoles.diff(self.painted, new)
        t.config(state="normal")
        if drop:
            t.delete("1.0", "%d.0" % (drop + 1))
        if keep:
            t.delete("%d.end" % keep, "end")
            if tail:
                t.insert("end", "\n" + "\n".join(tail))
        else:
            t.delete("1.0", "end")
            t.insert("1.0", "\n".join(tail))
        t.config(state="disabled")
        self.painted = list(new)
        if at_end:
            t.see("end")

    # ----------------------------------------------------------------- choosing
    def _select(self, hwnd):
        self.current = hwnd
        self.painted = []
        self.last_error = None
        self._paint(self.lines.get(hwnd, []))
        ended = hwnd in self.ended
        self.b_show.set(text="Remove" if ended else "Show window")
        self.b_break.set(state="disabled" if ended or hwnd is None else "normal")
        if hwnd is None:
            self.say("No consoles held. One opened outside the app will appear here.")
        elif ended:
            self.say("%s has closed. What it last showed is kept here; Remove clears it."
                     % clip(self.titles.get(hwnd, "")))
        else:
            self.say(HINT)
        self._paint_chips()

    def _paint_chips(self):
        for w in self.chips.winfo_children():
            w.destroy()
        for hwnd in self.order:
            label = clip(self.titles.get(hwnd, "")) + ("  (closed)" if hwnd in self.ended else "")
            self.host._button(self.chips, label, lambda h=hwnd: self._select(h),
                              kind="accent" if hwnd == self.current else "quiet").pack(
                side="left", padx=(0, 6))

    def _forget(self, hwnd):
        self.order.remove(hwnd)
        self.ended.discard(hwnd)
        self.lines.pop(hwnd, None)
        self.titles.pop(hwnd, None)

    # ------------------------------------------------------------------ actions
    def _show_window(self):
        hwnd = self.current
        if hwnd is None:
            return
        if hwnd not in self.ended:
            self.host.holder.release(hwnd, front=True)
        self._forget(hwnd)
        self._select(self.order[0] if self.order else None)
        self.host._consoles_changed()

    def _ask(self, op, **kw):
        hwnd = self.current
        info = self.host.holder.held.get(hwnd) if hwnd is not None else None
        if info is None:
            return

        def work():
            try:
                self.reader.ask(op, hwnd, info["pid"], **kw)
            except (OSError, ValueError) as e:
                self._post("error", (hwnd, str(e)))
        threading.Thread(target=work, daemon=True).start()

    def _send(self):
        text = self.entry.get()
        self.entry.delete(0, "end")
        self._ask("type", text=text + "\n")    # an empty line is Enter: "press any key"

    def _interrupt(self):
        self._ask("interrupt")

    def _redirect(self, ev):
        if ev.state & 0x4:                     # Ctrl: copy and select-all stay the mirror's
            return None
        if ev.char and ev.char.isprintable():
            self.entry.focus_set()
            self.entry.insert("end", ev.char)
            return "break"
        if ev.keysym == "Return":
            self.entry.focus_set()
            self._send()
            return "break"
        return None
