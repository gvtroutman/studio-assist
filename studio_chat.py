#!/usr/bin/env python3
"""
Studio Assistant - a chat window that drives creative apps with a local model.

One tab per app. Each tab owns its own MCP bridge, its own tool set, its own
system prompt and its own conversation - switching tabs switches which app you
are talking to, and nothing leaks between them.

Bridges start lazily: opening a tab for the first time is what launches that
app's bridge and pays its warm-up, so a session that only ever touches After
Effects never spawns Resolve's server.

Inference runs on the tailnet box. Nothing to install: Tkinter ships with
Python, and the engine is stdlib only.
"""

import json
import os
import queue
import re
import socket
import sys
import threading
import time
import traceback

try:
    import tkinter as tk
    from tkinter import font as tkfont
    from tkinter import messagebox
except ImportError:
    sys.exit("Tkinter is missing from this Python install; reinstall Python with tcl/tk.")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import studio_agent as eng

APP_NAME = "Studio Assistant"
ERROR_LOG = "studio_assistant_error.log"

# ------------------------------------------------------------------ appearance
BG = "#141413"
SIDE = "#1a1a18"
HEAD = "#1a1a18"
CARD = "#232321"
BORDER = "#302f2c"
TEXT = "#ecebe8"
MUTED = "#928d86"
FAINT = "#6b6862"
ACCENT = "#d97757"
ACCENT_DK = "#c26343"
OK = "#5fb87f"
WARN = "#e0a458"
ERRC = "#e0685c"

SIDEBAR_W = 236

MAX_HISTORY = 40
MAX_STEPS = 25


def pretty_host(url):
    """100.127.17.38:1234 - the scheme and /v1 are noise in a 236px rail."""
    s = url.replace("https://", "").replace("http://", "").rstrip("/")
    return s[:-3].rstrip("/") if s.endswith("/v1") else s


def clip(s, n):
    """Truncate rather than let a label wrap mid-word."""
    return s if len(s) <= n else s[:n - 1] + "…"


def rounded(canvas, x1, y1, x2, y2, r, **kw):
    """Rounded rectangle - Tk's canvas has no primitive for it."""
    pts = [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r, x2, y2,
           x2 - r, y2, x1 + r, y2, x1, y2, x1, y2 - r, x1, y1 + r, x1, y1]
    return canvas.create_polygon(pts, smooth=True, **kw)


class Session:
    """
    One tab. Everything an app-specific conversation needs, kept apart from
    every other app's: bridge, tools, history, transcript and busy state.
    """

    def __init__(self, app):
        self.app = app
        self.mcp = None
        self.tools = []
        self.messages = [{"role": "system", "content": app.chat_prompt()}]
        self.busy = False
        self.ready = False
        self.booting = False
        self.status = ("not started", MUTED, False)
        self.bridge = (FAINT, "%s\nnot started" % app.bridge_label)
        self.frame = None
        self.view = None
        self._stream_open = False
        self._stream_buf = []
        self._asst_start = "1.0"

    @property
    def id(self):
        return self.app.id

    def reset(self):
        self.messages = [{"role": "system", "content": self.app.chat_prompt()}]
        self._stream_open = False
        self._stream_buf = []

    def close(self):
        if self.mcp:
            try:
                self.mcp.close()
            except Exception:
                pass
            self.mcp = None


class Chat(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_NAME)
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w, h = min(1180, int(sw * 0.72)), min(820, int(sh * 0.80))
        self.geometry("%dx%d+%d+%d" % (w, h, (sw - w) // 2, max(0, (sh - h) // 3)))
        self.minsize(880, 520)
        self.configure(bg=BG)

        self.q = queue.Queue()
        self.llm = None
        self.host = eng.env_default("STUDIO_HOST", "AE_AGENT_HOST",
                                    fallback=eng.DEFAULT_HOST)
        self.want_model = eng.env_default("STUDIO_MODEL", "AE_AGENT_MODEL")
        self.host_ready = threading.Event()

        # Tabs for apps that are actually installed. If none are, show them all
        # anyway - each tab will then say plainly that its app is missing.
        apps = eng.installed_apps() or list(eng.APPS)
        self.sessions = {a.id: Session(a) for a in apps}
        self.order = [a.id for a in apps]
        self.active = self.order[0]

        self._fonts()
        self._build()
        self.after(40, self._drain)
        self._spawn(None, self._boot_host)
        self._select(self.active)
        self.protocol("WM_DELETE_WINDOW", self._quit)

    def _fonts(self):
        self.f_ui = tkfont.Font(family="Segoe UI", size=10)
        self.f_bold = tkfont.Font(family="Segoe UI", size=10, weight="bold")
        self.f_title = tkfont.Font(family="Segoe UI Semibold", size=12)
        self.f_small = tkfont.Font(family="Segoe UI", size=8)
        self.f_cap = tkfont.Font(family="Segoe UI", size=8, weight="bold")
        self.f_badge = tkfont.Font(family="Segoe UI", size=9, weight="bold")
        self.f_mono = tkfont.Font(family="Consolas", size=9)
        self.f_body = tkfont.Font(family="Segoe UI", size=11)
        self.f_body_b = tkfont.Font(family="Segoe UI", size=11, weight="bold")

    # ------------------------------------------------------------------- layout
    def _build(self):
        head = tk.Frame(self, bg=HEAD, height=52)
        head.pack(side="top", fill="x")
        head.pack_propagate(False)
        tk.Frame(self, bg=BORDER, height=1).pack(side="top", fill="x")

        tk.Label(head, text=APP_NAME, bg=HEAD, fg=TEXT, font=self.f_title
                 ).pack(side="left", padx=(18, 10))
        self.lbl_status = tk.Label(head, text="starting", bg=HEAD, fg=MUTED,
                                   font=self.f_ui)
        self.lbl_status.pack(side="left")
        self.btn_new = tk.Button(head, text="New chat", command=self._on_new,
                                 font=self.f_ui, bg=CARD, fg=TEXT, relief="flat",
                                 activebackground=BORDER, activeforeground=TEXT,
                                 padx=12, pady=4, cursor="hand2", bd=0)
        self.btn_new.pack(side="right", padx=(6, 18))
        self.btn_fix = tk.Button(head, text="Start app", command=self._on_fix,
                                 font=self.f_ui, bg=ACCENT, fg="#16150f", relief="flat",
                                 activebackground=ACCENT_DK, padx=12, pady=4,
                                 cursor="hand2", bd=0)

        main = tk.Frame(self, bg=BG)
        main.pack(side="top", fill="both", expand=True)

        side = tk.Frame(main, bg=SIDE, width=SIDEBAR_W)
        side.pack(side="left", fill="y")
        side.pack_propagate(False)
        tk.Frame(main, bg=BORDER, width=1).pack(side="left", fill="y")
        self._build_sidebar(side)

        right = tk.Frame(main, bg=BG)
        right.pack(side="left", fill="both", expand=True)

        # Fixed-size widgets are packed BEFORE the expanding transcript, on
        # purpose. In Tk an expanding sibling packed first claims the leftover
        # space and shoves later fixed-size widgets off the edge - which is
        # exactly how the composer used to disappear until the window was
        # resized. Composer, then tab strip, then the transcript stack.
        composer = tk.Frame(right, bg=BG)
        composer.pack(side="bottom", fill="x", padx=18, pady=(8, 16))
        self._build_composer(composer)

        strip = tk.Frame(right, bg=BG)
        strip.pack(side="top", fill="x", padx=14, pady=(10, 0))
        self._build_tabs(strip)
        tk.Frame(right, bg=BORDER, height=1).pack(side="top", fill="x")

        self.stack = tk.Frame(right, bg=BG)
        self.stack.pack(side="top", fill="both", expand=True)
        for sid in self.order:
            self._build_transcript(self.sessions[sid])

    def _build_transcript(self, s):
        s.frame = tk.Frame(self.stack, bg=BG)
        bar = tk.Scrollbar(s.frame, bg=BG, troughcolor=BG, activebackground=FAINT,
                           highlightthickness=0, bd=0, width=11)
        bar.pack(side="right", fill="y")
        s.view = tk.Text(s.frame, bg=BG, fg=TEXT, font=self.f_body, wrap="word", bd=0,
                         padx=22, pady=16, yscrollcommand=bar.set, state="disabled",
                         cursor="arrow", selectbackground="#3d3b37",
                         insertbackground=TEXT, highlightthickness=0)
        s.view.pack(side="left", fill="both", expand=True)
        bar.config(command=s.view.yview)
        self._tags(s.view)
        self._welcome(s)

    def _tags(self, v):
        v.tag_configure("role_user", foreground=ACCENT, font=self.f_cap,
                        spacing1=18, spacing3=6)
        v.tag_configure("role_asst", foreground="#8fb0c9", font=self.f_cap,
                        spacing1=18, spacing3=6)
        v.tag_configure("user", background=CARD, lmargin1=16, lmargin2=16, rmargin=16,
                        spacing1=6, spacing3=10, borderwidth=0)
        v.tag_configure("asst", lmargin1=16, lmargin2=16, rmargin=60, spacing2=4,
                        spacing3=10)
        v.tag_configure("tool", foreground=FAINT, font=self.f_mono, lmargin1=22,
                        lmargin2=36, rmargin=16, spacing1=2, spacing3=2)
        v.tag_configure("err", foreground=ERRC, lmargin1=16, lmargin2=16, rmargin=40,
                        spacing3=10)
        v.tag_configure("sys", foreground=MUTED, lmargin1=16, lmargin2=16, spacing3=8)
        v.tag_configure("hint", foreground=FAINT, lmargin1=26, lmargin2=26, spacing3=5)
        # last: these win on font when combined with a block tag
        v.tag_configure("b", font=self.f_body_b)
        v.tag_configure("code", font=self.f_mono, foreground="#d7d1c9")

    def _cap(self, parent, text):
        tk.Label(parent, text=text, bg=SIDE, fg=FAINT, font=self.f_cap,
                 anchor="w").pack(fill="x", padx=18, pady=(18, 8))

    def _badge(self, parent, code, fg, bg, size=26, canvas_bg=SIDE):
        c = tk.Canvas(parent, width=size, height=size, bg=canvas_bg,
                      highlightthickness=0, bd=0)
        rounded(c, 1, 1, size - 1, size - 1, 7, fill=bg, outline=bg)
        c.create_text(size / 2, size / 2 + 1, text=code, fill=fg,
                      font=self.f_badge)
        return c

    def _dot(self, parent, color, size=8, bg=SIDE):
        c = tk.Canvas(parent, width=size + 2, height=size + 2, bg=bg,
                      highlightthickness=0, bd=0)
        c.create_oval(1, 1, size, size, fill=color, outline=color)
        return c

    def _set_dot(self, canvas, color):
        canvas.itemconfig(1, fill=color, outline=color)

    def _hook_click(self, widget, fn):
        """Children swallow clicks, so bind the whole subtree."""
        widget.bind("<Button-1>", fn)
        for child in widget.winfo_children():
            self._hook_click(child, fn)

    # ---------------------------------------------------------------- tab strip
    def _build_tabs(self, strip):
        self.tab_ui = {}
        for sid in self.order:
            app = self.sessions[sid].app
            tab = tk.Frame(strip, bg=BG, cursor="hand2")
            tab.pack(side="left", padx=(0, 4))
            rule = tk.Frame(tab, bg=BORDER, height=2)
            rule.pack(side="bottom", fill="x")
            inner = tk.Frame(tab, bg=BG)
            inner.pack(side="top", padx=12, pady=(6, 7))
            badge = self._badge(inner, app.code, app.fg, app.bg, size=20, canvas_bg=BG)
            badge.pack(side="left")
            lbl = tk.Label(inner, text=app.tab, bg=BG, fg=MUTED, font=self.f_ui)
            lbl.pack(side="left", padx=(8, 8))
            dot = self._dot(inner, FAINT, bg=BG)
            dot.pack(side="left")
            self.tab_ui[sid] = {"tab": tab, "rule": rule, "label": lbl, "dot": dot,
                                "bgs": [tab, inner, lbl, badge, dot]}
            self._hook_click(tab, lambda ev, i=sid: self._select(i))

    def _paint_tab(self, sid):
        ui = self.tab_ui[sid]
        s = self.sessions[sid]
        on = sid == self.active
        for w in ui["bgs"]:
            w.config(bg=CARD if on else BG)
        ui["label"].config(fg=TEXT if on else MUTED)
        ui["rule"].config(bg=ACCENT if on else BORDER)
        self._set_dot(ui["dot"], s.bridge[0])

    def _select(self, sid):
        self.active = sid
        for i in self.order:
            self.sessions[i].frame.pack_forget()
        self.sessions[sid].frame.pack(side="top", fill="both", expand=True)
        for i in self.order:
            self._paint_tab(i)
        self._apply_status()
        self._ensure(self.sessions[sid])
        self.input.focus_set()

    # ----------------------------------------------------------------- sidebar
    def _build_sidebar(self, side):
        # Connections are built first and anchored to the bottom. The app list
        # runs to eight rows on a full Adobe install and would otherwise push
        # the live status clean off the panel - and the status is the half that
        # changes. Fixed widget before the expanding one, same rule as the
        # composer.
        conns = tk.Frame(side, bg=SIDE)
        conns.pack(side="bottom", fill="x", pady=(0, 14))
        tk.Frame(side, bg=BORDER, height=1).pack(side="bottom", fill="x", padx=14)

        listing = tk.Frame(side, bg=SIDE)
        listing.pack(side="top", fill="both", expand=True)

        self._cap(listing, "ON THIS PC")
        self.app_dots = {}
        apps = eng.detect_apps()
        for a in apps:
            row = tk.Frame(listing, bg=SIDE)
            row.pack(fill="x", padx=14, pady=2)
            self._badge(row, a["code"], a["fg"], a["bg"]).pack(side="left")
            box = tk.Frame(row, bg=SIDE)
            box.pack(side="left", fill="x", expand=True, padx=(9, 0))
            tk.Label(box, text=a["name"], bg=SIDE, fg=TEXT, font=self.f_ui,
                     anchor="w").pack(fill="x")
            sub = a["version"] or "installed"
            if a["drivable"]:
                sub += "  ·  drivable"
            tk.Label(box, text=sub, bg=SIDE, fg=FAINT, font=self.f_small,
                     anchor="w").pack(fill="x")
            if a["drivable"]:
                dot = self._dot(row, FAINT)
                dot.pack(side="right", padx=(4, 2))
                self.app_dots[a["id"]] = dot
        if not apps:
            tk.Label(listing, text="no creative apps found", bg=SIDE, fg=FAINT,
                     font=self.f_small).pack(padx=18, anchor="w")

        self._cap(conns, "CONNECTIONS")
        self.conn = {"host": self._conn_row(conns, "Inference", pretty_host(self.host))}
        for sid in self.order:
            app = self.sessions[sid].app
            self.conn[sid] = self._conn_row(conns, "%s bridge" % app.tab,
                                            "%s\nnot started" % app.bridge_label)

    def _conn_row(self, side, title, detail):
        row = tk.Frame(side, bg=SIDE)
        row.pack(fill="x", padx=14, pady=3)
        dot = self._dot(row, FAINT)
        dot.pack(side="left", padx=(6, 0), pady=(3, 0), anchor="n")
        box = tk.Frame(row, bg=SIDE)
        box.pack(side="left", fill="x", expand=True, padx=(10, 0))
        tk.Label(box, text=title, bg=SIDE, fg=TEXT, font=self.f_ui,
                 anchor="w").pack(fill="x")
        # no wraplength: these are pre-clipped, and char wrapping split IPs mid-number
        lbl = tk.Label(box, text=detail, bg=SIDE, fg=FAINT, font=self.f_small,
                       anchor="w", justify="left")
        lbl.pack(fill="x")
        return dot, lbl

    def _build_composer(self, composer):
        shell = tk.Frame(composer, bg=BORDER)
        shell.pack(fill="x")
        inner = tk.Frame(shell, bg=CARD)
        inner.pack(fill="x", padx=1, pady=1)
        # button first, then the expanding input - same rule as above
        self.btn_send = tk.Button(inner, text="Send", command=self._on_send,
                                  font=self.f_bold, bg=ACCENT, fg="#16150f",
                                  relief="flat", activebackground=ACCENT_DK,
                                  padx=18, pady=6, cursor="hand2", bd=0)
        self.btn_send.pack(side="right", padx=10, pady=10)
        self.input = tk.Text(inner, height=2, bg=CARD, fg=TEXT, font=self.f_body,
                             wrap="word", bd=0, padx=14, pady=11,
                             insertbackground=ACCENT, selectbackground="#3d3b37",
                             highlightthickness=0)
        self.input.pack(side="left", fill="both", expand=True)
        self.input.bind("<Return>", self._on_return)
        self.input.bind("<Control-Tab>", self._on_next_tab)
        self.input.focus_set()
        tk.Label(composer, text="Enter to send   ·   Shift+Enter for a new line"
                                "   ·   Ctrl+Tab to switch app",
                 bg=BG, fg=FAINT, font=self.f_small, anchor="w"
                 ).pack(fill="x", pady=(6, 0))

    def _welcome(self, s):
        self._write(s, "%s. Try:\n" % s.app.name, "sys")
        for e in s.app.examples:
            self._write(s, e + "\n", "hint")

    # -------------------------------------------------------------- view writes
    def _write(self, s, text, tag):
        s.view.config(state="normal")
        s.view.insert("end", text, tag)
        s.view.config(state="disabled")
        s.view.see("end")

    def _role(self, s, name, tag):
        self._write(s, "\n%s\n" % name, tag)

    _MD = re.compile(r"\*\*(.+?)\*\*|`([^`\n]+)`")

    def _insert_md(self, s, text, base):
        """Light markdown: **bold** and `code`. Models emit it whether asked or not."""
        pos = 0
        for m in self._MD.finditer(text):
            if m.start() > pos:
                self._write(s, text[pos:m.start()], base)
            if m.group(1) is not None:
                self._write(s, m.group(1), (base, "b"))
            else:
                self._write(s, m.group(2), (base, "code"))
            pos = m.end()
        if pos < len(text):
            self._write(s, text[pos:], base)

    def _apply_status(self):
        """The header always describes the tab you are looking at."""
        s = self.sessions[self.active]
        text, color, fixable = s.status
        self.lbl_status.config(text=text, fg=color)
        self.btn_fix.config(text="Start %s" % s.app.name)
        self._show_fix(fixable)
        self.btn_send.config(text="…" if s.busy else "Send",
                             state="disabled" if s.busy else "normal")

    def _show_fix(self, show):
        if show:
            self.btn_fix.pack(side="right", padx=4)
        else:
            self.btn_fix.pack_forget()

    # ----------------------------------------------------------------- threading
    def _spawn(self, sid, fn, *a):
        threading.Thread(target=self._guard, args=(sid, fn) + a, daemon=True).start()

    def _guard(self, sid, fn, *a):
        """No traceback ever reaches the user - it goes to the transcript as prose."""
        try:
            fn(*a)
        except Exception as e:
            self.q.put(("error", sid, "%s: %s" % (type(e).__name__, e)))
            self.q.put(("trace", sid, traceback.format_exc()))
            self.q.put(("idle", sid, None))

    def _drain(self):
        try:
            while True:
                kind, sid, payload = self.q.get_nowait()
                self._handle(kind, sid, payload)
        except queue.Empty:
            pass
        self.after(40, self._drain)

    def _handle(self, kind, sid, payload):
        # A sid of None means "whatever tab the user is looking at" - startup
        # errors from the shared inference host have no app of their own.
        s = self.sessions.get(sid) or self.sessions[self.active]

        if kind == "status":
            s.status = payload
            if s.id == self.active:
                self._apply_status()
        elif kind == "host":
            color, detail = payload
            dot, lbl = self.conn["host"]
            self._set_dot(dot, color)
            if detail:
                lbl.config(text=detail)
        elif kind == "bridge":
            s.bridge = payload
            color, detail = payload
            dot, lbl = self.conn[s.id]
            self._set_dot(dot, color)
            if detail:
                lbl.config(text=detail)
            if s.id in self.app_dots:
                self._set_dot(self.app_dots[s.id], color)
            self._paint_tab(s.id)
        elif kind == "sys":
            self._write(s, payload + "\n", "sys")
        elif kind == "tool":
            self._write(s, "  " + payload + "\n", "tool")
        elif kind == "tool_result":
            self._write(s, "     " + payload + "\n", "tool")
        elif kind == "stream_start":
            self._role(s, s.app.tab.upper(), "role_asst")
            s._asst_start = s.view.index("end-1c")
            s._stream_buf = []
            s._stream_open = True
        elif kind == "token":
            s._stream_buf.append(payload)
            self._write(s, payload, "asst")
        elif kind == "stream_end":
            if s._stream_open:
                raw = "".join(s._stream_buf).rstrip()
                s.view.config(state="normal")
                s.view.delete(s._asst_start, "end-1c")
                s.view.config(state="disabled")
                self._insert_md(s, raw + "\n", "asst")
                s._stream_buf = []
                s._stream_open = False
        elif kind == "error":
            s._stream_open = False
            self._write(s, payload + "\n", "err")
        elif kind == "trace":
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   ERROR_LOG), "a", encoding="utf-8") as f:
                f.write("\n---- %s ----\n%s" % (time.strftime("%Y-%m-%d %H:%M:%S"),
                                                payload))
        elif kind == "ready":
            self._write(s, "Connected and warmed up - replies land in a few seconds.\n",
                        "sys")
        elif kind == "idle":
            s.busy = False
            if s.id == self.active:
                self._apply_status()

    # --------------------------------------------------------------- preflight
    def _boot_host(self):
        """Shared across every tab: one inference host, one model."""
        self.q.put(("status", None, ("checking the inference host", MUTED, False)))
        ok, loaded, ids, err = eng.probe_models(self.host)
        if not ok:
            self.q.put(("host", None, (ERRC, "%s\nunreachable" % pretty_host(self.host))))
            self.q.put(("error", None,
                        "Cannot reach the inference host at %s.\n"
                        "Check that the other PC is awake, Tailscale is up on both "
                        "ends, and LM Studio's server is started.\n(%s)"
                        % (self.host, err)))
            self.host_ready.set()
            return
        model = eng.pick_model(loaded, ids, self.want_model)
        if not model:
            self.q.put(("host", None, (ERRC, "%s\nno models" % pretty_host(self.host))))
            self.q.put(("error", None, "The inference host is up but serving no models. "
                                       "Load one in LM Studio and reopen this window."))
            self.host_ready.set()
            return
        self.llm = eng.LLM(self.host, model)
        self.q.put(("host", None, (OK, "%s\n%d models\n%s"
                                   % (pretty_host(self.host), len(ids), clip(model, 24)))))
        self.host_ready.set()

    def _ensure(self, s):
        """First view of a tab is what starts that app's bridge."""
        if s.ready or s.booting:
            return
        s.booting = True
        self._spawn(s.id, self._boot_session, s)

    def _boot_session(self, s):
        sid = s.id
        try:
            if not self.host_ready.is_set():
                self.q.put(("status", sid, ("waiting for the inference host", MUTED, False)))
                self.host_ready.wait(timeout=240)
            if self.llm is None:
                self.q.put(("status", sid, ("no inference host", ERRC, False)))
                self.q.put(("bridge", sid, (ERRC, "%s\nno model" % s.app.bridge_label)))
                return

            self.q.put(("status", sid, ("starting the %s bridge" % s.app.name,
                                        MUTED, False)))
            try:
                mcp = eng.MCPClient(s.app.command, s.app.args, quiet=True)
                mcp.initialize(timeout=75)
                allt = mcp.list_tools(timeout=45)
            except Exception as e:
                self.q.put(("status", sid, ("bridge did not start", ERRC, False)))
                self.q.put(("bridge", sid, (ERRC, "%s\nfailed to start" % s.app.bridge_label)))
                self.q.put(("error", sid, self._bridge_help(s.app, e)))
                return
            s.mcp = mcp

            wanted = s.app.tool_names()
            s.tools = eng.to_openai_tools([t for t in allt if t["name"] in wanted])

            # Prefill dominates the first call - a full tool schema set takes about a
            # minute cold. Pay it here against the exact prompt prefix a real message
            # will use, so the first question comes back in seconds. Each app has its
            # own prefix, so each tab warms up the first time it is opened.
            self.q.put(("status", sid, ("warming up the model, about a minute",
                                        WARN, False)))
            try:
                self.llm.chat([{"role": "system", "content": s.app.chat_prompt()},
                               {"role": "user", "content": "Say ready."}],
                              s.tools, max_tokens=1)
            except Exception:
                pass  # warming is an optimisation; failing here is not fatal
            s.ready = True
            self._refresh_bridge(s)
            self.q.put(("ready", sid, None))
        finally:
            s.booting = False
            self.q.put(("idle", sid, None))

    def _bridge_help(self, app, err):
        if app.id == "after-effects":
            return ("Could not start the After Effects bridge.\n"
                    "Usually this is npx on a cold cache, or Node missing from PATH. "
                    "Close this window and open it again - the second start is normally "
                    "quick. If it keeps failing, run this once in a terminal to see the "
                    "real error:\n"
                    "    npx -y @engine-room/after-effects-mcp --help\n\n(%s)" % err)
        return ("Could not start the %s bridge.\n"
                "It is expected at:\n    %s\nIf that path is wrong, set RESOLVE_MCP_DIR "
                "to the checkout and reopen this window.\n\n(%s)"
                % (app.name, app.command, err))

    def _refresh_bridge(self, s):
        n = len(s.tools)
        if s.app.running():
            self.q.put(("status", s.id, ("connected", OK, False)))
            self.q.put(("bridge", s.id, (OK, "%s\n%d tools" % (s.app.bridge_label, n))))
        else:
            self.q.put(("status", s.id, ("%s is not running" % s.app.name, WARN, True)))
            self.q.put(("bridge", s.id, (WARN, "%s\nnot running" % s.app.bridge_label)))

    def _on_fix(self):
        s = self.sessions[self.active]
        if s.busy:
            return
        s.busy = True
        self._apply_status()
        self._spawn(s.id, self._fix, s)

    def _fix(self, s):
        try:
            if not s.app.running():
                self.q.put(("sys", s.id, "Launching %s..." % s.app.name))
                self.q.put(("status", s.id, ("launching %s" % s.app.name, WARN, False)))
                s.app.launch()
                for _ in range(60):
                    if s.app.running():
                        self.q.put(("sys", s.id, "%s is up. %s"
                                    % (s.app.name, s.app.launch_note)))
                        break
                    time.sleep(2)
                else:
                    self.q.put(("error", s.id,
                                "%s did not come up within two minutes. %s"
                                % (s.app.name, s.app.launch_note)))
            self._refresh_bridge(s)
        finally:
            self.q.put(("idle", s.id, None))

    # ------------------------------------------------------------------ sending
    def _on_return(self, ev):
        if ev.state & 0x0001:  # Shift+Enter = newline
            return None
        self._on_send()
        return "break"

    def _on_next_tab(self, ev=None):
        i = (self.order.index(self.active) + 1) % len(self.order)
        self._select(self.order[i])
        return "break"

    def _on_new(self):
        s = self.sessions[self.active]
        if s.busy:
            return
        s.reset()
        s.view.config(state="normal")
        s.view.delete("1.0", "end")
        s.view.config(state="disabled")
        self._welcome(s)

    def _on_send(self):
        s = self.sessions[self.active]
        if s.busy:
            return
        task = self.input.get("1.0", "end").strip()
        if not task:
            return
        if not s.ready:
            self._write(s, "%s is still starting up - give it a moment.\n"
                        % s.app.tab, "err")
            return
        self.input.delete("1.0", "end")
        self._role(s, "YOU", "role_user")
        self._write(s, task + "\n", "user")
        s.messages.append({"role": "user", "content": task})
        s.busy = True
        s.status = ("working", WARN, False)
        self._apply_status()
        self._spawn(s.id, self._turn, s)

    def _trim(self, s):
        if len(s.messages) > MAX_HISTORY + 1:
            keep = s.messages[1:][-MAX_HISTORY:]
            while keep and keep[0].get("role") == "tool":
                keep.pop(0)  # never open on an orphaned tool reply
            s.messages = s.messages[:1] + keep

    def _turn(self, s):
        sid = s.id
        for _ in range(MAX_STEPS):
            self._trim(s)
            started = {"v": False}

            def on_text(piece, st=started):
                if not st["v"]:
                    st["v"] = True
                    self.q.put(("stream_start", sid, None))
                self.q.put(("token", sid, piece))

            msg = self.llm.stream(s.messages, s.tools, on_text)
            self.q.put(("stream_end", sid, None))
            s.messages.append(msg)

            calls = msg.get("tool_calls") or []
            if not calls:
                if not (msg.get("content") or "").strip():
                    self.q.put(("error", sid, "The model returned an empty reply."))
                self._refresh_bridge(s)
                self.q.put(("idle", sid, None))
                return

            for call in calls:
                fn = call["function"]
                name = fn["name"]
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError as e:
                    out = ("TOOL ERROR: arguments were not valid JSON (%s). "
                           "Re-issue with valid JSON." % e)
                    self.q.put(("tool", sid, "%s  [bad arguments]" % name))
                else:
                    preview = json.dumps(args)
                    self.q.put(("tool", sid, "%s %s" % (
                        name, preview[:110] + ("..." if len(preview) > 110 else ""))))
                    try:
                        out = eng.mcp_result_to_text(s.mcp.call_tool(name, args))
                    except Exception as e:
                        out = "TOOL ERROR: %s" % e
                    # collapse to one line - raw JSON's first line is often just "["
                    flat = " ".join(out.split())
                    self.q.put(("tool_result", sid,
                                flat[:110] + ("..." if len(flat) > 110 else "")))
                s.messages.append({"role": "tool",
                                   "tool_call_id": call.get("id", name),
                                   "content": out})
        self.q.put(("sys", sid, "Stopped - the agent hit its step limit for this turn."))
        self._refresh_bridge(s)
        self.q.put(("idle", sid, None))

    def _quit(self):
        for s in self.sessions.values():
            s.close()
        self.destroy()


_LOCK = None


def claim_single_instance(port=57733):
    """One app, one set of MCP servers. A second copy would spawn rival bridges."""
    global _LOCK
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", port))
        s.listen(1)
        _LOCK = s
        return True
    except OSError:
        s.close()
        return False


def main():
    try:  # crisp text on a high-DPI display
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    if not claim_single_instance():
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo(APP_NAME, "%s is already running.\n\n"
                                      "Look for its window on the taskbar." % APP_NAME)
        root.destroy()
        return
    Chat().mainloop()


if __name__ == "__main__":
    main()
