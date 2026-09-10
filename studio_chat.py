#!/usr/bin/env python3
"""
Studio Assistant - a chat window that drives creative apps with a local model.

One tab per app. Each tab owns its own MCP bridge, its own tool set, its own
system prompt and its own conversation - switching tabs switches which app you
are talking to, and nothing leaks between them. Tabs are opened and closed from
the tab strip, and which ones are open is remembered between runs.

Bridges start lazily: opening a tab for the first time is what launches that
app's bridge and pays its warm-up, so a session that only ever touches After
Effects never spawns Resolve's server.

Inference runs on the tailnet box. Nothing to install: Tkinter ships with
Python, and the engine is stdlib only.
"""

import base64
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import studio_agent as eng
import studio_icons as icons

APP_NAME = "Studio Assistant"
ERROR_LOG = "studio_assistant_error.log"
ICON_FILE = os.path.join(HERE, "studio-assistant.ico")

# ------------------------------------------------------------------ appearance
# Widgets are built against role names, never literal hex, and every one of them
# is registered in `Chat.skin`. That is what makes Preferences able to repaint
# the running window instead of rebuilding it - a rebuild would throw away every
# transcript. Anything that puts a colour on the queue sends a role name too.
DARK = {
    "bg": "#141413", "side": "#1a1a18", "head": "#1a1a18", "card": "#232321",
    "hover": "#262623", "border": "#302f2c", "text": "#ecebe8",
    "muted": "#928d86", "faint": "#6b6862", "accent": "#d97757",
    "accent_dk": "#c26343", "accent_fg": "#16150f", "ok": "#5fb87f",
    "warn": "#e0a458", "err": "#e0685c", "sel": "#3d3b37", "code": "#d7d1c9",
    "asst": "#8fb0c9",
}

LIGHT = {
    "bg": "#fbfaf8", "side": "#f1eee9", "head": "#f1eee9", "card": "#e5e1d8",
    "hover": "#e5e1d8", "border": "#d7d1c6", "text": "#23211d",
    "muted": "#66615a", "faint": "#8f8981", "accent": "#c2582f",
    "accent_dk": "#a44821", "accent_fg": "#fffaf6", "ok": "#2f7d52",
    "warn": "#96650f", "err": "#b23b30", "sel": "#d8d2c5", "code": "#4c4740",
    "asst": "#2c6a91",
}

THEMES = {"dark": DARK, "light": LIGHT}
THEME_NAMES = [("dark", "Dark"), ("light", "Light")]

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


APP_NAME_CHARS = 16                       # sidebar rows, before the ellipsis


def app_subtitle(a):
    """The second line of a sidebar row. Also what the rail is measured on."""
    sub = a["version"] or "installed"
    return sub + "  ·  drivable" if a["drivable"] else sub


def this_pc():
    """The machine name, for the sidebar heading - this is the PC being driven."""
    try:
        return socket.gethostname().upper()
    except Exception:
        return "THIS PC"


def settings_path():
    return os.environ.get("STUDIO_SETTINGS") or os.path.join(
        os.environ.get("APPDATA") or os.path.expanduser("~"),
        "StudioAssistant", "settings.json")


class Prefs:
    """
    A small JSON file under %APPDATA%. Best-effort in both directions: a
    read-only or locked-down profile costs you the preference, never the app.
    """

    DEFAULTS = {"theme": "dark", "tabs": None, "pinned": [], "hidden": []}

    def __init__(self, path=None):
        self.path = path or settings_path()
        self.data = dict(self.DEFAULTS)
        try:
            with open(self.path, encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                self.data.update(loaded)
        except Exception:
            pass
        # Anything on disk is a suggestion, not a contract - a hand-edited file
        # must not be able to stop the window opening.
        if self.data.get("theme") not in THEMES:
            self.data["theme"] = "dark"
        for key in ("pinned", "hidden"):
            got = self.data.get(key)
            # `isinstance(got, list)` first: a bare string is iterable, and
            # "nope" would otherwise hide four apps called n, o, p and e.
            self.data[key] = ([x for x in got if isinstance(x, str)]
                              if isinstance(got, list) else [])
        tabs = self.data.get("tabs")
        self.data["tabs"] = ([x for x in tabs if isinstance(x, str)]
                             if isinstance(tabs, list) else None)

    def get(self, key):
        return self.data.get(key)

    def set(self, **kw):
        self.data.update(kw)
        self.save()

    def save(self):
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2)
        except Exception:
            pass


class Session:
    """
    One tab. Everything an app-specific conversation needs, kept apart from
    every other app's: bridge, tools, history, transcript and busy state.
    """

    def __init__(self, app):
        self.app = app
        self.mcp = None
        self.tools = []
        self.catalog = []                 # every tool the bridge offers
        self.messages = [{"role": "system", "content": app.chat_prompt()}]
        self.busy = False
        self.ready = False
        self.booting = False
        self.closed = False
        self.status = ("not started", "muted", False)
        self.bridge = ("faint", "%s\nnot started" % app.bridge_label)
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
        try:  # the app's own mark, in the title bar and on the taskbar
            self.iconbitmap(default=ICON_FILE)
        except Exception:
            pass
        # Tk scales fonts by the screen's DPI but not the pixel numbers handed
        # to pack() and geometry(). On a 150% display that combination clipped
        # the sidebar - the text grew, the rail holding it did not.
        self.scale = max(1.0, self.winfo_fpixels("1i") / 96.0)
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w, h = min(self._px(1180), int(sw * 0.72)), min(self._px(820), int(sh * 0.80))
        self.geometry("%dx%d+%d+%d" % (w, h, (sw - w) // 2, max(0, (sh - h) // 3)))
        self.minsize(min(self._px(880), int(sw * 0.9)),
                     min(self._px(520), int(sh * 0.9)))

        self.prefs = Prefs()
        self.C = dict(THEMES[self.prefs.get("theme")])
        self.skin = {}                    # widget -> {tk option: palette role}
        self.dot_role = {}                # canvas -> palette role
        self.marks = {}                   # icon key -> [(canvas, size, spec)]
        self.photos = {}                  # (icon key, size) -> PhotoImage
        self.windows = {}                 # ("tools", app id) / "prefs" -> Toplevel
        self.tool_views = {}              # app id -> that window's Text
        self.configure(bg=self.C["bg"])

        self.q = queue.Queue()
        self.llm = None
        self.host = eng.env_default("STUDIO_HOST", "AE_AGENT_HOST",
                                    fallback=eng.DEFAULT_HOST)
        self.want_model = eng.env_default("STUDIO_MODEL", "AE_AGENT_MODEL")
        self.host_ready = threading.Event()

        self.detected = eng.detect_apps()
        self.hidden = list(self.prefs.get("hidden"))
        self.pinned = list(self.prefs.get("pinned"))

        apps = self._opening_tabs()
        self.sessions = {a.id: Session(a) for a in apps}
        self.order = [a.id for a in apps]
        self.active = self.order[0] if self.order else None

        self._fonts()
        self._metrics()
        self._build()
        self._menus()
        self.after(40, self._drain)
        self._spawn(None, self._read_icons)
        self._spawn(None, self._boot_host)
        self._select(self.active)
        self.protocol("WM_DELETE_WINDOW", self._quit)

    def _opening_tabs(self):
        """
        Which tabs to open: what was open last time, else every installed app.
        A registry entry that has since disappeared is dropped silently.
        """
        want = self.prefs.get("tabs")
        if want is not None:
            return [eng.APPS_BY_ID[i] for i in want if i in eng.APPS_BY_ID]
        return eng.installed_apps() or list(eng.APPS)

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
        # Segoe MDL2 Assets is Windows' own icon font - a pin and a chain link
        # drawn by the OS beat anything hand-plotted at 9pt. Fall back to
        # punctuation if it is somehow missing.
        have = "Segoe MDL2 Assets" in set(tkfont.families())
        self.f_glyph = tkfont.Font(family="Segoe MDL2 Assets" if have else "Segoe UI",
                                   size=9)
        # By codepoint: these are private-use characters that paste into an
        # editor as blanks, and MDL2 is documented by its hex codes anyway.
        mdl2 = {"pin": 0xE718, "unpin": 0xE77A, "close": 0xE8BB,
                "add": 0xE710, "link": 0xE71B, "more": 0xE70D}
        plain = {"pin": 0x2191, "unpin": 0x2193, "close": 0x00D7,
                 "add": 0x002B, "link": 0x21C4, "more": 0x02C5}
        self.g = {k: chr(v) for k, v in (mdl2 if have else plain).items()}

    def _px(self, n):
        """A pixel count designed at 96dpi, in this display's pixels."""
        return int(round(n * self.scale))

    def _metrics(self):
        """
        Sizes that have to hold text. The rail is measured against the rows it
        is actually going to draw - version lines run from "Beta" to
        "2024, 2025, 2026, Beta" - so nothing is clipped at any display scale.
        """
        self.marks_px = {"row": self._px(26), "tab": self._px(20),
                         "menu": self._px(16)}
        self.dot_px = self._px(8)
        widest = 0
        for a in self.detected:
            widest = max(widest, self.f_ui.measure(clip(a["name"], APP_NAME_CHARS)),
                         self.f_small.measure(app_subtitle(a)))
        # mark, both glyph buttons, the status dot and every gap between them
        self.side_w = max(self._px(SIDEBAR_W), widest + self._px(122))

    # ------------------------------------------------------------------ theming
    def _skin(self, widget, **roles):
        """Remember which palette role each colour came from, so switching
        theme is a re-read of the same widgets rather than a rebuild."""
        self.skin[widget] = roles
        try:
            widget.config(**{k: self.C[v] for k, v in roles.items()})
        except tk.TclError:
            pass
        return widget

    def _forget(self):
        """Sweep out destroyed widgets. The app list is rebuilt on every pin,
        hide and theme change, and each rebuild leaves its rows behind."""
        for reg in (self.skin, self.dot_role):
            for widget in list(reg):
                try:
                    if not widget.winfo_exists():
                        del reg[widget]
                except tk.TclError:
                    reg.pop(widget, None)
        for key, entries in list(self.marks.items()):
            self.marks[key] = [e for e in entries if e[0].winfo_exists()]

    def _theme(self, name):
        if name not in THEMES:
            return
        self.C = dict(THEMES[name])
        self.prefs.set(theme=name)
        self.configure(bg=self.C["bg"])
        self._forget()
        for widget in list(self.skin):
            try:
                widget.config(**{k: self.C[v] for k, v in self.skin[widget].items()})
            except tk.TclError:
                self.skin.pop(widget, None)
        for canvas in list(self.dot_role):
            self._set_dot(canvas, self.dot_role[canvas])
        for s in self.sessions.values():
            self._tags(s.view)
        for app_id, view in list(self.tool_views.items()):
            if view.winfo_exists():
                self._tool_tags(view)
            else:
                del self.tool_views[app_id]
        self._build_apps()                # rows carry their own hover colours
        for sid in self.order:
            self._paint_tab(sid)
        self._apply_status()
        self._sync_bridges()
        self._menus()                     # menu colours are set at build time
        if callable(self.windows.get("prefs_paint")):
            self.windows["prefs_paint"]()

    # ------------------------------------------------------------------- menus
    def _menus(self):
        """A real menu bar. Windows draws the bar itself; the drop-downs take
        our colours, so they follow the theme and the bar does not."""
        bar = tk.Menu(self, tearoff=0)

        def menu():
            return tk.Menu(bar, tearoff=0, bg=self.C["card"], fg=self.C["text"],
                           activebackground=self.C["accent"],
                           activeforeground=self.C["accent_fg"],
                           bd=0, activeborderwidth=0)

        m_file = menu()
        m_file.add_command(label="New chat", accelerator="Ctrl+N",
                           command=self._on_new)
        m_file.add_command(label="New tab...", accelerator="Ctrl+T",
                           command=lambda: self._tab_menu(self.btn_add))
        m_file.add_command(label="Close tab", accelerator="Ctrl+W",
                           command=self._close_tab)
        m_file.add_separator()
        m_file.add_command(label="Preferences...", accelerator="Ctrl+,",
                           command=self._prefs_window)
        m_file.add_separator()
        m_file.add_command(label="Exit", command=self._quit)
        bar.add_cascade(label="File", menu=m_file)

        m_view = menu()
        self.theme_var = tk.StringVar(value=self.prefs.get("theme"))
        for key, label in THEME_NAMES:
            m_view.add_radiobutton(label="%s mode" % label, value=key,
                                   variable=self.theme_var,
                                   command=lambda k=key: self._theme(k))
        m_view.add_separator()
        m_view.add_command(label="Next app", accelerator="Ctrl+Tab",
                           command=self._on_next_tab)
        bar.add_cascade(label="View", menu=m_view)

        m_bridge = menu()
        for app in eng.APPS:
            m_bridge.add_command(label="%s tools..." % app.name,
                                 command=lambda i=app.id: self._tools_window(i))
        m_bridge.add_separator()
        m_bridge.add_command(label="Start the current app", command=self._on_fix)
        bar.add_cascade(label="Bridges", menu=m_bridge)

        m_help = menu()
        m_help.add_command(label="About %s" % APP_NAME, command=self._about)
        bar.add_cascade(label="Help", menu=m_help)

        self.config(menu=bar)
        # Bound on the input as well, returning "break", so Tk's own Text
        # bindings (Ctrl+T transposes characters) never also fire.
        for seq, fn in (("<Control-n>", self._on_new),
                        ("<Control-t>", lambda: self._tab_menu(self.btn_add)),
                        ("<Control-w>", self._close_tab),
                        ("<Control-comma>", self._prefs_window)):
            self.bind_all(seq, lambda ev, f=fn: (f(), "break")[1])
            self.input.bind(seq, lambda ev, f=fn: (f(), "break")[1])
        self.bind_all("<Control-Tab>", self._on_next_tab)
        self.input.bind("<Control-Tab>", self._on_next_tab)

    def _about(self):
        messagebox.showinfo(
            APP_NAME,
            "%s\n\nOne tab per creative app, driven by a local model.\n\n"
            "Inference: %s\nBridges run here on %s."
            % (APP_NAME, pretty_host(self.host), this_pc()), parent=self)

    # ------------------------------------------------------------------- layout
    def _build(self):
        head = tk.Frame(self, height=52)
        self._skin(head, bg="head")
        head.pack(side="top", fill="x")
        head.pack_propagate(False)
        self._skin(tk.Frame(self, height=1), bg="border").pack(side="top", fill="x")

        self._skin(tk.Label(head, text=APP_NAME, font=self.f_title),
                   bg="head", fg="text").pack(side="left", padx=(18, 10))
        self.lbl_status = tk.Label(head, text="starting", font=self.f_ui)
        self._skin(self.lbl_status, bg="head", fg="muted")
        self.lbl_status.pack(side="left")
        self.btn_new = tk.Button(head, text="New chat", command=self._on_new,
                                 font=self.f_ui, relief="flat", padx=12, pady=4,
                                 cursor="hand2", bd=0)
        self._skin(self.btn_new, bg="card", fg="text", activebackground="border",
                   activeforeground="text")
        self.btn_new.pack(side="right", padx=(6, 18))
        self.btn_fix = tk.Button(head, text="Start app", command=self._on_fix,
                                 font=self.f_ui, relief="flat", padx=12, pady=4,
                                 cursor="hand2", bd=0)
        self._skin(self.btn_fix, bg="accent", fg="accent_fg",
                   activebackground="accent_dk", activeforeground="accent_fg")

        main = self._skin(tk.Frame(self), bg="bg")
        main.pack(side="top", fill="both", expand=True)

        side = tk.Frame(main, width=self.side_w)
        self._skin(side, bg="side")
        side.pack(side="left", fill="y")
        side.pack_propagate(False)
        self._skin(tk.Frame(main, width=1), bg="border").pack(side="left", fill="y")
        self._build_sidebar(side)

        right = self._skin(tk.Frame(main), bg="bg")
        right.pack(side="left", fill="both", expand=True)

        # Fixed-size widgets are packed BEFORE the expanding transcript, on
        # purpose. In Tk an expanding sibling packed first claims the leftover
        # space and shoves later fixed-size widgets off the edge - which is
        # exactly how the composer used to disappear until the window was
        # resized. Composer, then tab strip, then the transcript stack.
        composer = self._skin(tk.Frame(right), bg="bg")
        composer.pack(side="bottom", fill="x", padx=18, pady=(8, 16))
        self._build_composer(composer)

        strip = self._skin(tk.Frame(right), bg="bg")
        strip.pack(side="top", fill="x", padx=14, pady=(10, 0))
        self._build_tabs(strip)
        self._skin(tk.Frame(right, height=1), bg="border").pack(side="top", fill="x")

        self.stack = self._skin(tk.Frame(right), bg="bg")
        self.stack.pack(side="top", fill="both", expand=True)
        self._build_empty()
        for sid in self.order:
            self._build_transcript(self.sessions[sid])

    def _build_empty(self):
        """What the stack shows with every tab closed. Host-level errors have
        to land somewhere even then, so this panel can speak."""
        self.empty = self._skin(tk.Frame(self.stack), bg="bg")
        box = self._skin(tk.Frame(self.empty), bg="bg")
        box.pack(expand=True)
        self._skin(tk.Label(box, text="No app open", font=self.f_title),
                   bg="bg", fg="muted").pack(pady=(0, 6))
        self._skin(tk.Label(box, text="Open a tab for the app you want to talk to.",
                            font=self.f_ui), bg="bg", fg="faint").pack()
        btn = tk.Button(box, text="Choose an app", font=self.f_ui, relief="flat",
                        padx=14, pady=5, cursor="hand2", bd=0)
        btn.config(command=lambda: self._tab_menu(btn))
        self._skin(btn, bg="accent", fg="accent_fg", activebackground="accent_dk",
                   activeforeground="accent_fg")
        btn.pack(pady=14)
        self.empty_msg = tk.Label(box, text="", font=self.f_ui, wraplength=420,
                                  justify="center")
        self._skin(self.empty_msg, bg="bg", fg="err")
        self.empty_msg.pack(pady=(4, 0))

    def _build_transcript(self, s):
        s.frame = self._skin(tk.Frame(self.stack), bg="bg")
        bar = tk.Scrollbar(s.frame, highlightthickness=0, bd=0, width=11)
        self._skin(bar, bg="bg", troughcolor="bg", activebackground="faint")
        bar.pack(side="right", fill="y")
        s.view = tk.Text(s.frame, font=self.f_body, wrap="word", bd=0, padx=22,
                         pady=16, yscrollcommand=bar.set, state="disabled",
                         cursor="arrow", highlightthickness=0)
        self._skin(s.view, bg="bg", fg="text", selectbackground="sel",
                   insertbackground="text")
        s.view.pack(side="left", fill="both", expand=True)
        bar.config(command=s.view.yview)
        self._tags(s.view)
        self._welcome(s)

    def _tags(self, v):
        C = self.C
        v.tag_configure("role_user", foreground=C["accent"], font=self.f_cap,
                        spacing1=18, spacing3=6)
        v.tag_configure("role_asst", foreground=C["asst"], font=self.f_cap,
                        spacing1=18, spacing3=6)
        v.tag_configure("user", background=C["card"], lmargin1=16, lmargin2=16,
                        rmargin=16, spacing1=6, spacing3=10, borderwidth=0)
        v.tag_configure("asst", lmargin1=16, lmargin2=16, rmargin=60, spacing2=4,
                        spacing3=10)
        v.tag_configure("tool", foreground=C["faint"], font=self.f_mono, lmargin1=22,
                        lmargin2=36, rmargin=16, spacing1=2, spacing3=2)
        v.tag_configure("err", foreground=C["err"], lmargin1=16, lmargin2=16,
                        rmargin=40, spacing3=10)
        v.tag_configure("sys", foreground=C["muted"], lmargin1=16, lmargin2=16,
                        spacing3=8)
        v.tag_configure("hint", foreground=C["faint"], lmargin1=26, lmargin2=26,
                        spacing3=5)
        # last: these win on font when combined with a block tag
        v.tag_configure("b", font=self.f_body_b)
        v.tag_configure("code", font=self.f_mono, foreground=C["code"])

    # ------------------------------------------------------------ small widgets
    def _cap(self, parent, text, bg="side"):
        lbl = tk.Label(parent, text=text, font=self.f_cap, anchor="w")
        self._skin(lbl, bg=bg, fg="faint")
        return lbl

    def _mark(self, parent, spec, size=26, bg="side"):
        """
        An app's mark: its real icon, read out of its own .exe, and the drawn
        two-letter badge until (or unless) that arrives. Registered by key so
        the icon can be swapped in from the loader thread.
        """
        c = tk.Canvas(parent, width=size, height=size, highlightthickness=0, bd=0)
        self._skin(c, bg=bg)
        self.marks.setdefault(spec["key"], []).append((c, size, spec))
        self._draw_mark(c, size, spec)
        return c

    def _draw_mark(self, c, size, spec):
        c.delete("all")
        photo = self.photos.get((spec["key"], size))
        if photo is not None:
            c.create_image(size / 2, size / 2, image=photo)
            return
        rounded(c, 1, 1, size - 1, size - 1, 7, fill=spec["bg"], outline=spec["bg"])
        c.create_text(size / 2, size / 2 + 1, text=spec["code"], fill=spec["fg"],
                      font=self.f_badge)

    def _dot(self, parent, role, size=None, bg="side"):
        size = size or self.dot_px
        c = tk.Canvas(parent, width=size + 2, height=size + 2, highlightthickness=0,
                      bd=0)
        self._skin(c, bg=bg)
        c.create_oval(1, 1, size, size, fill=self.C[role], outline=self.C[role])
        self.dot_role[c] = role
        return c

    def _set_dot(self, canvas, role):
        self.dot_role[canvas] = role
        try:
            canvas.itemconfig(1, fill=self.C[role], outline=self.C[role])
        except tk.TclError:
            self.dot_role.pop(canvas, None)

    def _glyph(self, parent, name, command, bg="side", fg="faint", tip=None):
        """A one-character button in Windows' icon font. Cheaper than an image
        and it stays crisp at any DPI, same reasoning as the drawn badges."""
        lbl = tk.Label(parent, text=self.g[name], font=self.f_glyph, cursor="hand2",
                       padx=3)
        self._skin(lbl, bg=bg, fg=fg)
        lbl.bind("<Button-1>", lambda ev: (command(lbl), "break")[1])
        lbl.bind("<Enter>", lambda ev: lbl.config(fg=self.C["text"]))
        lbl.bind("<Leave>", lambda ev: lbl.config(fg=self.C[fg]))
        if tip:
            self._tip(lbl, tip)
        return lbl

    def _tip(self, widget, text):
        """A plain tooltip - these controls are small and their meaning is not
        guessable from a 9pt glyph."""
        state = {"win": None}

        def show(_ev=None):
            if state["win"] or not widget.winfo_exists():
                return
            win = tk.Toplevel(widget)
            win.wm_overrideredirect(True)
            win.configure(bg=self.C["border"])
            tk.Label(win, text=text, font=self.f_small, bg=self.C["card"],
                     fg=self.C["text"], padx=7, pady=3).pack(padx=1, pady=1)
            win.wm_geometry("+%d+%d" % (widget.winfo_rootx(),
                                        widget.winfo_rooty() + widget.winfo_height() + 4))
            state["win"] = win

        def hide(_ev=None):
            if state["win"]:
                state["win"].destroy()
                state["win"] = None

        widget.bind("<Enter>", show, add="+")
        widget.bind("<Leave>", hide, add="+")
        widget.bind("<Button-1>", hide, add="+")

    def _hook_click(self, widget, fn):
        """Children swallow clicks, so bind the whole subtree."""
        widget.bind("<Button-1>", fn)
        for child in widget.winfo_children():
            self._hook_click(child, fn)

    def _hover(self, row, widgets, base, lit):
        """
        Light a whole row on hover. <Leave> also fires when the pointer moves
        onto a child, so check where it actually went before unlighting.
        """
        def paint(role):
            for w in widgets:
                try:
                    w.config(bg=self.C[role])
                except tk.TclError:
                    pass

        def leave(ev):
            under = row.winfo_containing(ev.x_root, ev.y_root)
            while under is not None:
                if under is row:
                    return
                under = getattr(under, "master", None)
            paint(base)

        for w in widgets:
            w.bind("<Enter>", lambda ev: paint(lit), add="+")
            w.bind("<Leave>", leave, add="+")

    def _popup(self, menu, widget):
        """Post a menu under the control that opened it, and always let go."""
        try:
            menu.tk_popup(widget.winfo_rootx(),
                          widget.winfo_rooty() + widget.winfo_height())
        finally:
            menu.grab_release()

    def _menu_item(self, menu, label, key, command):
        """A menu row carrying the app's own icon, once we have read it."""
        extra = {}
        photo = self.photos.get((key, self.marks_px["menu"]))
        if photo is not None:
            extra = {"image": photo, "compound": "left"}
        menu.add_command(label=label, command=command, **extra)

    def _menu(self):
        return tk.Menu(self, tearoff=0, bg=self.C["card"], fg=self.C["text"],
                       activebackground=self.C["accent"],
                       activeforeground=self.C["accent_fg"], bd=0,
                       activeborderwidth=0)

    # ---------------------------------------------------------------- tab strip
    def _build_tabs(self, strip):
        self.tab_ui = {}
        # Tabs live in their own frame so the + button trails them without
        # having to be re-packed every time a tab opens or closes.
        self.tabbar = self._skin(tk.Frame(strip), bg="bg")
        self.tabbar.pack(side="left")
        self.btn_add = self._glyph(strip, "add", self._tab_menu, bg="bg",
                                   tip="Open a tab for another app")
        self.btn_add.pack(side="left", padx=(4, 0), pady=(6, 7))
        for sid in self.order:
            self._make_tab(sid)

    def _make_tab(self, sid):
        app = self.sessions[sid].app
        tab = self._skin(tk.Frame(self.tabbar, cursor="hand2"), bg="bg")
        tab.pack(side="left", padx=(0, 4))
        rule = self._skin(tk.Frame(tab, height=2), bg="border")
        rule.pack(side="bottom", fill="x")
        inner = self._skin(tk.Frame(tab), bg="bg")
        inner.pack(side="top", padx=(10, 8), pady=(6, 7))
        mark = self._mark(inner, self._spec_for(app), self.marks_px["tab"],
                          bg="bg")
        mark.pack(side="left")
        lbl = tk.Label(inner, text=app.tab, font=self.f_ui)
        self._skin(lbl, bg="bg", fg="muted")
        lbl.pack(side="left", padx=(8, 8))
        dot = self._dot(inner, "faint", bg="bg")
        dot.pack(side="left")
        close = tk.Label(inner, text=self.g["close"], font=self.f_glyph,
                         cursor="hand2", padx=2)
        self._skin(close, bg="bg", fg="faint")
        close.pack(side="left", padx=(8, 0))
        self.tab_ui[sid] = {"tab": tab, "rule": rule, "label": lbl, "dot": dot,
                            "close": close,
                            "bgs": [tab, inner, lbl, mark, dot, close]}
        self._hook_click(tab, lambda ev, i=sid: self._select(i))
        # after _hook_click, so the close glyph keeps its own handler
        close.bind("<Button-1>", lambda ev, i=sid: (self._close_tab(i), "break")[1])

    def _spec_for(self, app):
        """The mark for a registry app - keyed by app id, same as its sidebar row."""
        return {"key": app.id, "code": app.code, "fg": app.fg, "bg": app.bg}

    def _paint_tab(self, sid):
        ui = self.tab_ui[sid]
        s = self.sessions[sid]
        on = sid == self.active
        for w in ui["bgs"]:
            w.config(bg=self.C["card"] if on else self.C["bg"])
        ui["label"].config(fg=self.C["text"] if on else self.C["muted"])
        ui["close"].config(fg=self.C["muted"] if on else self.C["faint"])
        ui["rule"].config(bg=self.C["accent"] if on else self.C["border"])
        self._set_dot(ui["dot"], s.bridge[0])

    def _menu_tabs(self):
        """Which app to talk to. Every drivable app is offered; one already
        open switches to it rather than opening a second."""
        m = self._menu()
        for app in eng.APPS:
            open_now = app.id in self.sessions
            self._menu_item(m, "%s%s" % (app.name, "   (open)" if open_now else ""),
                            app.id, lambda i=app.id: self._add_tab(i))
        return m

    def _tab_menu(self, widget=None):
        self._popup(self._menu_tabs(), widget or self.btn_add)

    def _add_tab(self, app_id):
        if app_id in self.sessions:
            self._select(app_id)
            return
        s = Session(eng.APPS_BY_ID[app_id])
        self.sessions[app_id] = s
        self.order.append(app_id)
        self._build_transcript(s)
        self._make_tab(app_id)
        self._select(app_id)
        self._remember_tabs()
        self._sync_bridges()

    def _close_tab(self, sid=None):
        sid = sid or self.active
        if sid is None:
            return
        s = self.sessions.pop(sid)
        s.closed = True
        self.order.remove(sid)
        ui = self.tab_ui.pop(sid)
        ui["tab"].destroy()
        s.frame.destroy()
        # Shutting an MCP subprocess down can block for a moment; a turn still
        # in flight keeps running and its events are dropped by _handle.
        self._spawn(None, s.close)
        if self.active == sid:
            self.active = self.order[0] if self.order else None
            self._select(self.active)
        else:
            self._sync_bridges()
        self._remember_tabs()

    def _remember_tabs(self):
        self.prefs.set(tabs=list(self.order))

    def _select(self, sid):
        self.active = sid
        for i in self.order:
            self.sessions[i].frame.pack_forget()
        self.empty.pack_forget()
        if sid is None:
            self.empty.pack(side="top", fill="both", expand=True)
        else:
            self.sessions[sid].frame.pack(side="top", fill="both", expand=True)
        for i in self.order:
            self._paint_tab(i)
        self._apply_status()
        self._sync_bridges()
        if sid is not None:
            self._ensure(self.sessions[sid])
            self.input.focus_set()

    def _on_next_tab(self, ev=None):
        if len(self.order) > 1:
            i = (self.order.index(self.active) + 1) % len(self.order)
            self._select(self.order[i])
        return "break"

    # ----------------------------------------------------------------- sidebar
    def _build_sidebar(self, side):
        # Connections are built first and anchored to the bottom. The app list
        # runs to eight rows on a full Adobe install and would otherwise push
        # the live status clean off the panel - and the status is the half that
        # changes. Fixed widget before the expanding one, same rule as the
        # composer.
        conns = self._skin(tk.Frame(side), bg="side")
        conns.pack(side="bottom", fill="x", pady=(0, 14))
        self._skin(tk.Frame(side, height=1), bg="border").pack(
            side="bottom", fill="x", padx=14)

        listing = self._skin(tk.Frame(side), bg="side")
        listing.pack(side="top", fill="both", expand=True)

        header = self._skin(tk.Frame(listing), bg="side")
        header.pack(fill="x", padx=(18, 12), pady=(18, 8))
        self._cap(header, "ON %s" % clip(this_pc(), 22)).pack(side="left")
        self._glyph(header, "add", self._unhide_menu,
                    tip="Add an app back to this list").pack(side="right")

        # A full Adobe install plus Resolve is eight rows, and a short window
        # cut the last one in half. The list scrolls inside a canvas; the bar
        # only appears when it is actually needed.
        wrap = self._skin(tk.Frame(listing), bg="side")
        wrap.pack(side="top", fill="both", expand=True)
        self.applist_bar = tk.Scrollbar(wrap, highlightthickness=0, bd=0, width=9)
        self._skin(self.applist_bar, bg="border", troughcolor="side",
                   activebackground="faint")
        canvas = tk.Canvas(wrap, highlightthickness=0, bd=0)
        self._skin(canvas, bg="side")
        canvas.pack(side="left", fill="both", expand=True)
        self.applist_canvas = canvas
        self.applist = self._skin(tk.Frame(canvas), bg="side")
        item = canvas.create_window(0, 0, window=self.applist, anchor="nw")
        self.applist_bar.config(command=canvas.yview)
        canvas.config(yscrollcommand=self.applist_bar.set)

        def fit(_ev=None):
            canvas.itemconfig(item, width=canvas.winfo_width())
            canvas.config(scrollregion=canvas.bbox("all"))
            over = self.applist.winfo_reqheight() > canvas.winfo_height()
            if over and not self.applist_bar.winfo_ismapped():
                self.applist_bar.pack(side="right", fill="y")
            elif not over and self.applist_bar.winfo_ismapped():
                self.applist_bar.pack_forget()
                canvas.yview_moveto(0)

        canvas.bind("<Configure>", fit)
        self.applist.bind("<Configure>", fit)

        def wheel(ev):
            if self.applist_bar.winfo_ismapped():
                canvas.yview_scroll(-1 if ev.delta > 0 else 1, "units")

        # bind_all while the pointer is over the rail: rows are separate
        # widgets and the wheel would otherwise only reach whichever one it
        # happens to be over. Nothing else in the window binds the wheel.
        canvas.bind("<Enter>", lambda ev: canvas.bind_all("<MouseWheel>", wheel))
        canvas.bind("<Leave>", lambda ev: canvas.unbind_all("<MouseWheel>"))
        self._build_apps()

        self._cap(conns, "CONNECTIONS").pack(fill="x", padx=18, pady=(18, 8))
        self.conn = {"host": self._conn_row(conns, "Inference",
                                            pretty_host(self.host))}
        self.conn["bridges"] = self._conn_row(
            conns, "Bridges", "not started", glyph="link",
            command=self._bridges_menu)

    def _build_apps(self):
        """
        The list is the user's, not the machine's: pinned apps first, hidden
        ones gone entirely. Detection still decides what may appear at all.
        """
        for w in self.applist.winfo_children():
            w.destroy()
        self._forget()
        self.app_dots = {}

        rows = [a for a in self.detected if a["name"] not in self.hidden]
        rows.sort(key=lambda a: self.pinned.index(a["name"])
                  if a["name"] in self.pinned else len(self.pinned))
        for a in rows:
            self._app_row(a)
        if not rows:
            msg = ("every app is hidden" if self.detected
                   else "no creative apps found")
            self._skin(tk.Label(self.applist, text=msg, font=self.f_small),
                       bg="side", fg="faint").pack(padx=18, pady=2, anchor="w")

    def _app_row(self, a):
        name = a["name"]
        pinned = name in self.pinned
        row = self._skin(tk.Frame(self.applist), bg="side")
        row.pack(fill="x", padx=(10, 8), pady=1)
        spec = {"key": a["id"] or name, "code": a["code"], "fg": a["fg"],
                "bg": a["bg"]}
        mark = self._mark(row, spec, self.marks_px["row"])
        mark.pack(side="left", padx=(4, 0))

        # Every fixed-size widget before the expanding one. The name box claims
        # whatever is left over, so packing it first shoves the pin, the hide
        # and the status dot off the right-hand edge - the same pack-order trap
        # that hid the Send button. Right to left: hide, pin, dot.
        hide = self._glyph(row, "close", lambda w, n=name: self._hide_app(n),
                           tip="Remove %s from this list" % name)
        hide.pack(side="right")
        pin = self._glyph(row, "unpin" if pinned else "pin",
                          lambda w, n=name: self._pin_app(n),
                          fg="accent" if pinned else "faint",
                          tip="Unpin %s" % name if pinned else "Pin %s to the top" % name)
        pin.pack(side="right")
        dot = None
        if a["drivable"]:
            dot = self._dot(row, "faint")
            dot.pack(side="right", padx=(4, 2))
            self.app_dots[a["id"]] = dot

        box = self._skin(tk.Frame(row), bg="side")
        box.pack(side="left", fill="x", expand=True, padx=(9, 0))
        title = self._skin(tk.Label(box, text=clip(name, APP_NAME_CHARS),
                                    font=self.f_ui, anchor="w"), bg="side", fg="text")
        title.pack(fill="x")
        subtitle = self._skin(tk.Label(box, text=app_subtitle(a), font=self.f_small,
                                       anchor="w"), bg="side", fg="faint")
        subtitle.pack(fill="x")

        widgets = [row, box, title, subtitle, mark, hide, pin]
        if dot is not None:
            widgets.append(dot)
            row.config(cursor="hand2")
            for w in (row, box, title, subtitle, mark):
                w.bind("<Button-1>", lambda ev, i=a["id"]: self._add_tab(i))
        self._hover(row, widgets, "side", "hover")
        for w in widgets:
            w.bind("<Button-3>", lambda ev, r=a: self._app_context(ev, r), add="+")

    def _app_context(self, ev, a):
        name = a["name"]
        m = self._menu()
        if a["drivable"]:
            m.add_command(label="Chat with %s" % name,
                          command=lambda i=a["id"]: self._add_tab(i))
            m.add_separator()
        m.add_command(label="Unpin" if name in self.pinned else "Pin to the top",
                      command=lambda n=name: self._pin_app(n))
        m.add_command(label="Remove from the list",
                      command=lambda n=name: self._hide_app(n))
        try:
            m.tk_popup(ev.x_root, ev.y_root)
        finally:
            m.grab_release()

    def _pin_app(self, name):
        if name in self.pinned:
            self.pinned.remove(name)
        else:
            self.pinned.append(name)
        self.prefs.set(pinned=list(self.pinned))
        self._build_apps()

    def _hide_app(self, name):
        if name not in self.hidden:
            self.hidden.append(name)
        self.prefs.set(hidden=list(self.hidden))
        self._build_apps()

    def _show_app(self, name):
        if name in self.hidden:
            self.hidden.remove(name)
        self.prefs.set(hidden=list(self.hidden))
        self._build_apps()

    def _menu_hidden(self):
        m = self._menu()
        if self.hidden:
            for name in self.hidden:
                m.add_command(label=name, command=lambda n=name: self._show_app(n))
            m.add_separator()
            m.add_command(label="Show all", command=self._show_all)
        else:
            m.add_command(label="Nothing is hidden", state="disabled")
        return m

    def _unhide_menu(self, widget):
        self._popup(self._menu_hidden(), widget)

    def _show_all(self):
        self.hidden = []
        self.prefs.set(hidden=[])
        self._build_apps()

    # ------------------------------------------------------------- connections
    def _conn_row(self, side, title, detail, glyph=None, command=None):
        row = self._skin(tk.Frame(side), bg="side")
        row.pack(fill="x", padx=14, pady=3)
        if glyph:
            lead = tk.Label(row, text=self.g[glyph], font=self.f_glyph)
            self._skin(lead, bg="side", fg="faint")
            lead.pack(side="left", padx=(1, 0), pady=(2, 0), anchor="n")
        else:
            lead = self._dot(row, "faint")
            lead.pack(side="left", padx=(6, 0), pady=(3, 0), anchor="n")
        box = self._skin(tk.Frame(row), bg="side")
        box.pack(side="left", fill="x", expand=True, padx=(10, 0))
        head = self._skin(tk.Frame(box), bg="side")
        head.pack(fill="x")
        self._skin(tk.Label(head, text=title, font=self.f_ui, anchor="w"),
                   bg="side", fg="text").pack(side="left")
        # no wraplength: these are pre-clipped, and char wrapping split IPs mid-number
        lbl = tk.Label(box, text=detail, font=self.f_small, anchor="w",
                       justify="left")
        self._skin(lbl, bg="side", fg="faint")
        lbl.pack(fill="x")
        if command:
            more = tk.Label(head, text=self.g["more"], font=self.f_glyph)
            self._skin(more, bg="side", fg="faint")
            more.pack(side="left", padx=(6, 0))
            row.config(cursor="hand2")
            self._hook_click(row, lambda ev: command(row))
            self._hover(row, [row, box, head, lbl, lead, more], "side", "hover")
        return lead, lbl

    def _sync_bridges(self):
        """One row for every bridge at once: how many are up, how many tools
        they are offering. Which bridges exist is in the menu behind it."""
        lead, lbl = self.conn["bridges"]
        live = [s for s in self.sessions.values() if s.ready]
        tools = sum(len(s.tools) for s in live)
        roles = [s.bridge[0] for s in self.sessions.values()]
        if not self.sessions:
            role, detail = "faint", "no tab open\n%d available" % len(eng.APPS)
        elif live:
            role = "ok" if len(live) == len(self.sessions) else "warn"
            detail = "%d of %d connected\n%d tools" % (len(live), len(self.sessions),
                                                       tools)
        else:
            role = "err" if "err" in roles else "faint"
            detail = "%d bridge%s\nnot started" % (len(self.sessions),
                                                   "" if len(self.sessions) == 1 else "s")
        lead.config(fg=self.C[role])
        lbl.config(text=detail)

    def _menu_bridges(self):
        """What bridges exist, and what each one can currently do."""
        m = self._menu()
        for app in eng.APPS:
            s = self.sessions.get(app.id)
            if s is None:
                note = "no tab open"
            elif s.ready:
                note = "%d tools" % len(s.tools)
            else:
                note = s.bridge[1].splitlines()[-1]
            self._menu_item(m, "%s  —  %s" % (app.name, note), app.id,
                            lambda i=app.id: self._tools_window(i))
        m.add_separator()
        m.add_command(label="Bridges run here on %s" % this_pc(), state="disabled")
        return m

    def _bridges_menu(self, widget):
        self._popup(self._menu_bridges(), widget)

    def _tool_tags(self, view):
        """Tag colours are copied out of the palette, so a tools window left
        open across a theme switch has to be told again."""
        view.tag_configure("group", foreground=self.C["accent"], font=self.f_cap,
                           spacing1=16, spacing3=6)
        view.tag_configure("name", foreground=self.C["text"], font=self.f_mono,
                           lmargin1=8, spacing1=6)
        view.tag_configure("off", foreground=self.C["faint"], font=self.f_mono,
                           lmargin1=8, spacing1=6)
        view.tag_configure("desc", foreground=self.C["muted"], font=self.f_small,
                           lmargin1=8, lmargin2=8, rmargin=12, spacing3=4)

    def _tools_window(self, app_id):
        """
        Everything the bridge exposes, grouped the way the registry groups it,
        and marked with what this chat actually offers the model. The working
        set is deliberately smaller than the bridge - see AGENTS.md.
        """
        app = eng.APPS_BY_ID[app_id]
        key = ("tools", app_id)
        win = self.windows.get(key)
        if win is not None and win.winfo_exists():
            win.deiconify()
            win.lift()
            return
        win = tk.Toplevel(self)
        self.windows[key] = win
        win.title("%s tools" % app.name)
        win.geometry("%dx%d" % (self._px(560), self._px(620)))
        self._skin(win, bg="bg")

        s = self.sessions.get(app_id)
        catalog = {t["name"]: (t.get("description") or "").strip()
                   for t in (s.catalog if s else [])}
        exposed = app.tool_names()

        head = self._skin(tk.Frame(win), bg="head")
        head.pack(side="top", fill="x")
        self._mark(head, self._spec_for(app), self.marks_px["row"],
                   bg="head").pack(
            side="left", padx=14, pady=12)
        box = self._skin(tk.Frame(head), bg="head")
        box.pack(side="left", pady=12)
        self._skin(tk.Label(box, text="%s bridge" % app.name, font=self.f_title,
                            anchor="w"), bg="head", fg="text").pack(fill="x")
        if catalog:
            note = "%s  ·  %d tools, %d offered to the model" % (
                app.bridge_label, len(catalog), len(exposed & set(catalog)))
        else:
            note = ("%s  ·  not started, so this is the registry's own list"
                    % app.bridge_label)
        self._skin(tk.Label(box, text=note, font=self.f_small, anchor="w"),
                   bg="head", fg="faint").pack(fill="x")
        self._skin(tk.Frame(win, height=1), bg="border").pack(side="top", fill="x")

        bar = tk.Scrollbar(win, highlightthickness=0, bd=0, width=11)
        self._skin(bar, bg="bg", troughcolor="bg", activebackground="faint")
        bar.pack(side="right", fill="y")
        view = tk.Text(win, font=self.f_body, wrap="word", bd=0, padx=18, pady=14,
                       yscrollcommand=bar.set, state="disabled", cursor="arrow",
                       highlightthickness=0)
        self._skin(view, bg="bg", fg="text", selectbackground="sel")
        view.pack(side="left", fill="both", expand=True)
        bar.config(command=view.yview)
        self.tool_views[app_id] = view
        self._tool_tags(view)

        view.config(state="normal")
        seen = set()
        for group, names in app.groups.items():
            on = group in app.default_groups
            view.insert("end", "%s%s\n" % (group.upper(),
                                           "" if on else "   (not in this chat)"),
                        "group")
            for name in names:
                if name in seen:
                    continue
                seen.add(name)
                view.insert("end", name + "\n", "name" if on else "off")
                desc = catalog.get(name, "")
                if desc:
                    view.insert("end", clip(" ".join(desc.split()), 400) + "\n",
                                "desc")
        extra = sorted(set(catalog) - seen)
        if extra:
            view.insert("end", "ALSO ON THIS BRIDGE   (not grouped)\n", "group")
            for name in extra:
                view.insert("end", name + "\n", "off")
                view.insert("end", clip(" ".join(catalog[name].split()), 400) + "\n",
                            "desc")
        view.config(state="disabled")

    # ---------------------------------------------------------------- composer
    def _build_composer(self, composer):
        shell = self._skin(tk.Frame(composer), bg="border")
        shell.pack(fill="x")
        inner = self._skin(tk.Frame(shell), bg="card")
        inner.pack(fill="x", padx=1, pady=1)
        # button first, then the expanding input - same rule as above
        self.btn_send = tk.Button(inner, text="Send", command=self._on_send,
                                  font=self.f_bold, relief="flat", padx=18, pady=6,
                                  cursor="hand2", bd=0)
        self._skin(self.btn_send, bg="accent", fg="accent_fg",
                   activebackground="accent_dk", activeforeground="accent_fg")
        self.btn_send.pack(side="right", padx=10, pady=10)
        self.input = tk.Text(inner, height=2, font=self.f_body, wrap="word", bd=0,
                             padx=14, pady=11, highlightthickness=0)
        self._skin(self.input, bg="card", fg="text", insertbackground="accent",
                   selectbackground="sel")
        self.input.pack(side="left", fill="both", expand=True)
        self.input.bind("<Return>", self._on_return)
        self.input.focus_set()
        hint = tk.Label(composer, text="Enter to send   ·   Shift+Enter for a new line"
                                       "   ·   Ctrl+Tab to switch app",
                        font=self.f_small, anchor="w")
        self._skin(hint, bg="bg", fg="faint")
        hint.pack(fill="x", pady=(6, 0))

    def _welcome(self, s):
        self._write(s, "%s. Try:\n" % s.app.name, "sys")
        for e in s.app.examples:
            self._write(s, e + "\n", "hint")

    # ------------------------------------------------------------- preferences
    def _prefs_window(self):
        win = self.windows.get("prefs")
        if win is not None and win.winfo_exists():
            win.deiconify()
            win.lift()
            return
        win = tk.Toplevel(self)
        self.windows["prefs"] = win
        win.title("Preferences")
        win.resizable(False, False)
        win.transient(self)
        self._skin(win, bg="bg")

        body = self._skin(tk.Frame(win), bg="bg")
        body.pack(fill="both", expand=True, padx=22, pady=18)
        self._cap(body, "APPEARANCE", bg="bg").pack(fill="x", pady=(0, 10))

        cards = self._skin(tk.Frame(body), bg="bg")
        cards.pack(fill="x")
        shells = {}
        for key, label in THEME_NAMES:
            shell = self._skin(tk.Frame(cards), bg="border")
            shell.pack(side="left", padx=(0, 12))
            inner = tk.Frame(shell, bg=THEMES[key]["bg"], cursor="hand2")
            inner.pack(padx=2, pady=2)
            # A working miniature of the window, painted in the palette it
            # selects - the honest way to show what the choice does.
            c = tk.Canvas(inner, width=118, height=74, bg=THEMES[key]["bg"],
                          highlightthickness=0, bd=0)
            c.pack(padx=8, pady=(8, 4))
            p = THEMES[key]
            c.create_rectangle(0, 0, 34, 74, fill=p["side"], outline=p["side"])
            c.create_rectangle(0, 0, 118, 13, fill=p["head"], outline=p["head"])
            for i, y in enumerate((24, 36, 48)):
                c.create_rectangle(6, y, 28, y + 6, fill=p["card"], outline=p["card"])
            c.create_rectangle(44, 24, 108, 44, fill=p["card"], outline=p["card"])
            c.create_rectangle(44, 52, 84, 62, fill=p["border"], outline=p["border"])
            c.create_rectangle(90, 50, 110, 64, fill=p["accent"], outline=p["accent"])
            tk.Label(inner, text=label, font=self.f_ui, bg=THEMES[key]["bg"],
                     fg=THEMES[key]["text"]).pack(pady=(0, 8))
            for w in (inner, c) + tuple(inner.winfo_children()):
                w.bind("<Button-1>", lambda ev, k=key: self._theme(k))
            shells[key] = shell

        def paint():
            for key, shell in shells.items():
                chosen = self.prefs.get("theme") == key
                try:
                    shell.config(bg=self.C["accent"] if chosen else self.C["border"])
                except tk.TclError:
                    pass
            self.theme_var.set(self.prefs.get("theme"))

        # registered so a theme change from the menu relights the right card
        self.windows["prefs_paint"] = paint
        paint()

        self._cap(body, "SIDEBAR", bg="bg").pack(fill="x", pady=(22, 8))
        count = tk.Label(body, font=self.f_small, anchor="w")
        self._skin(count, bg="bg", fg="faint")
        count.pack(fill="x")
        btn = tk.Button(body, text="Show every app again", font=self.f_ui,
                        relief="flat", padx=12, pady=4, cursor="hand2", bd=0)
        self._skin(btn, bg="card", fg="text", activebackground="border",
                   activeforeground="text")
        btn.pack(anchor="w", pady=(8, 0))

        def refresh_count():
            n = len(self.hidden)
            count.config(text="%s hidden from the app list."
                         % ("Nothing is" if not n else
                            "%d app%s" % (n, "" if n == 1 else "s")))
            btn.config(state="disabled" if not n else "normal")

        btn.config(command=lambda: (self._show_all(), refresh_count()))
        refresh_count()

        close = tk.Button(body, text="Close", command=win.destroy, font=self.f_ui,
                          relief="flat", padx=16, pady=5, cursor="hand2", bd=0)
        self._skin(close, bg="accent", fg="accent_fg", activebackground="accent_dk",
                   activeforeground="accent_fg")
        close.pack(anchor="e", pady=(22, 0))

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

    def cur(self):
        """The session the user is looking at, or None with every tab closed."""
        return self.sessions.get(self.active)

    def _apply_status(self):
        """The header always describes the tab you are looking at."""
        s = self.cur()
        if s is None:
            self.lbl_status.config(text="no app open", fg=self.C["muted"])
            self._show_fix(False)
            self.btn_send.config(text="Send", state="disabled")
            self.btn_new.config(state="disabled")
            return
        text, role, fixable = s.status
        self.lbl_status.config(text=text, fg=self.C[role])
        self.btn_fix.config(text="Start %s" % s.app.name)
        self._show_fix(fixable)
        self.btn_new.config(state="normal")
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

    def _read_icons(self):
        """
        Off the UI thread: an app's icon lives inside its .exe, and a cold read
        of a 500MB Photoshop binary is not something to do while the window is
        trying to open. Badges are drawn until these land.
        """
        row, tab, menu = (self.marks_px["row"], self.marks_px["tab"],
                          self.marks_px["menu"])
        jobs = [(a["id"] or a["name"], a["exe"], row) for a in self.detected]
        for app in eng.APPS:
            exe = app.exe()
            jobs += [(app.id, exe, tab), (app.id, exe, menu)]
        for key, exe, size in jobs:
            data = icons.icon_png(exe, size)
            if data:
                self.q.put(("icon", None, (key, size, data)))

    def _handle(self, kind, sid, payload):
        # A sid of None means "whatever tab the user is looking at" - startup
        # errors from the shared inference host have no app of their own. A sid
        # whose tab has been closed is dropped: a turn still finishing must not
        # write into somebody else's transcript.
        if sid is not None and sid not in self.sessions:
            return
        s = self.sessions.get(sid) or self.cur()

        if kind == "icon":
            key, size, data = payload
            # PhotoImage has to be built on the UI thread, and something must
            # keep a reference or Tk drops the image the moment it is collected.
            try:
                self.photos[(key, size)] = tk.PhotoImage(
                    data=base64.b64encode(data).decode("ascii"), master=self)
            except tk.TclError:
                return
            live = []
            for canvas, csize, spec in self.marks.get(key, []):
                try:
                    if not canvas.winfo_exists():
                        continue
                    if csize == size:
                        self._draw_mark(canvas, csize, spec)
                except tk.TclError:
                    continue
                live.append((canvas, csize, spec))
            self.marks[key] = live
            return

        if s is None:                     # every tab is closed
            if kind in ("error", "sys"):
                self.empty_msg.config(text=payload)
            elif kind == "trace":
                self._log(payload)
            return

        if kind == "status":
            s.status = payload
            if s.id == self.active:
                self._apply_status()
        elif kind == "host":
            role, detail = payload
            dot, lbl = self.conn["host"]
            self._set_dot(dot, role)
            if detail:
                lbl.config(text=detail)
        elif kind == "bridge":
            s.bridge = payload
            role, detail = payload
            if s.id in self.app_dots:
                self._set_dot(self.app_dots[s.id], role)
            self._paint_tab(s.id)
            self._sync_bridges()
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
            self._log(payload)
        elif kind == "ready":
            self._write(s, "Connected and warmed up - replies land in a few seconds.\n",
                        "sys")
            self._sync_bridges()
        elif kind == "idle":
            s.busy = False
            if s.id == self.active:
                self._apply_status()

    def _log(self, text):
        with open(os.path.join(HERE, ERROR_LOG), "a", encoding="utf-8") as f:
            f.write("\n---- %s ----\n%s" % (time.strftime("%Y-%m-%d %H:%M:%S"), text))

    # --------------------------------------------------------------- preflight
    def _boot_host(self):
        """Shared across every tab: one inference host, one model."""
        self.q.put(("status", None, ("checking the inference host", "muted", False)))
        ok, loaded, ids, err = eng.probe_models(self.host)
        if not ok:
            self.q.put(("host", None, ("err", "%s\nunreachable" % pretty_host(self.host))))
            self.q.put(("error", None,
                        "Cannot reach the inference host at %s.\n"
                        "Check that the other PC is awake, Tailscale is up on both "
                        "ends, and LM Studio's server is started.\n(%s)"
                        % (self.host, err)))
            self.host_ready.set()
            return
        model = eng.pick_model(loaded, ids, self.want_model)
        if not model:
            self.q.put(("host", None, ("err", "%s\nno models" % pretty_host(self.host))))
            self.q.put(("error", None, "The inference host is up but serving no models. "
                                       "Load one in LM Studio and reopen this window."))
            self.host_ready.set()
            return
        self.llm = eng.LLM(self.host, model)
        self.q.put(("host", None, ("ok", "%s\n%d models\n%s"
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
                self.q.put(("status", sid, ("waiting for the inference host",
                                            "muted", False)))
                self.host_ready.wait(timeout=240)
            if self.llm is None:
                self.q.put(("status", sid, ("no inference host", "err", False)))
                self.q.put(("bridge", sid, ("err", "%s\nno model" % s.app.bridge_label)))
                return

            self.q.put(("status", sid, ("starting the %s bridge" % s.app.name,
                                        "muted", False)))
            try:
                mcp = eng.MCPClient(s.app.command, s.app.args, quiet=True)
                mcp.initialize(timeout=75)
                allt = mcp.list_tools(timeout=45)
            except Exception as e:
                self.q.put(("status", sid, ("bridge did not start", "err", False)))
                self.q.put(("bridge", sid, ("err", "%s\nfailed to start"
                                            % s.app.bridge_label)))
                self.q.put(("error", sid, self._bridge_help(s.app, e)))
                return
            s.mcp = mcp
            if s.closed:
                # The tab was closed while this bridge was starting, so nothing
                # was there to shut down yet. Do it now or the subprocess
                # outlives the tab for the rest of the session.
                s.close()
                return
            # the whole catalogue, for the tools window; the model sees the
            # working set only
            s.catalog = [{"name": t["name"], "description": t.get("description", "")}
                         for t in allt]

            wanted = s.app.tool_names()
            s.tools = eng.to_openai_tools([t for t in allt if t["name"] in wanted])

            # Prefill dominates the first call - a full tool schema set takes about a
            # minute cold. Pay it here against the exact prompt prefix a real message
            # will use, so the first question comes back in seconds. Each app has its
            # own prefix, so each tab warms up the first time it is opened.
            self.q.put(("status", sid, ("warming up the model, about a minute",
                                        "warn", False)))
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
            self.q.put(("status", s.id, ("connected", "ok", False)))
            self.q.put(("bridge", s.id, ("ok", "%s\n%d tools" % (s.app.bridge_label, n))))
        else:
            self.q.put(("status", s.id, ("%s is not running" % s.app.name, "warn", True)))
            self.q.put(("bridge", s.id, ("warn", "%s\nnot running" % s.app.bridge_label)))

    def _on_fix(self):
        s = self.cur()
        if s is None or s.busy:
            return
        s.busy = True
        self._apply_status()
        self._spawn(s.id, self._fix, s)

    def _fix(self, s):
        try:
            if not s.app.running():
                self.q.put(("sys", s.id, "Launching %s..." % s.app.name))
                self.q.put(("status", s.id, ("launching %s" % s.app.name, "warn", False)))
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

    def _on_new(self):
        s = self.cur()
        if s is None or s.busy:
            return
        s.reset()
        s.view.config(state="normal")
        s.view.delete("1.0", "end")
        s.view.config(state="disabled")
        self._welcome(s)

    def _on_send(self):
        s = self.cur()
        if s is None or s.busy:
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
        s.status = ("working", "warn", False)
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
