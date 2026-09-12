#!/usr/bin/env python3
"""
Studio Assistant - a chat window that drives creative apps with a local model.

One tab per app, plus a Chat tab with no app behind it. Each tab owns its own
MCP bridge, its own tool set, its own system prompt and its own conversation -
switching tabs switches which app you are talking to, and nothing leaks between
them. Chat is the same window with the bridge and the tools left out. Tabs are opened and closed from
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
import shutil
import socket
import sys
import threading
import time
import traceback
import uuid

try:
    import tkinter as tk
    from tkinter import font as tkfont
    from tkinter import messagebox, filedialog
except ImportError:
    sys.exit("Tkinter is missing from this Python install; reinstall Python with tcl/tk.")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import studio_agent as eng
import studio_icons as icons
import studio_tasks as tasks
import studio_toolsmith as toolsmith

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

# Apps this machine may have that no bridge here drives, and where a bridge
# for each has been seen. Hints for the connect dialog, not commands: each of
# these installs a CEP panel inside the app and starts differently, so the
# command line has to come from the one the user actually installed.
SUGGESTED_BRIDGES = {
    "Premiere Pro": "Bridges seen for Premiere Pro: antipaster/Adobe-Premiere-Pro-MCP and "
                    "leancoderkavy/premiere-pro-mcp on GitHub. Both need their CEP panel "
                    "installed in Premiere (Window > Extensions) and the panel open; the "
                    "command line is in each README.",
    "Audition": "No standalone Audition bridge is known. mikechambers/adb-mcp on GitHub "
                "covers several Adobe apps through one proxy; check whether its current "
                "release lists Audition before installing it.",
    "Media Encoder": "Media Encoder is normally driven from Premiere or After Effects "
                     "(their render queues hand off to it) rather than by a bridge of its "
                     "own. A Premiere bridge that lists AME tools is the usual route.",
}

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


class Pill(tk.Canvas):
    """
    A rounded button. Tk's Button is a rectangle and nothing on it bends, so
    this draws its own: a smoothed polygon with the label on top. `roles` are
    palette role names; `paint(C)` is called with the palette on every theme
    switch, and `set()` covers what _apply_status used to config() on the
    Button - the text and whether it takes clicks.
    """

    def __init__(self, parent, text, command, font, roles, padx=18, pady=6, r=12,
                 **kw):
        tk.Canvas.__init__(self, parent, highlightthickness=0, bd=0, cursor="hand2",
                           **kw)
        self.command, self.font, self.roles = command, font, roles
        self.padx, self.pady, self.r = padx, pady, r
        self.text, self.lit, self.C = text, False, None
        self.bind("<Button-1>", self._click)
        self.bind("<Enter>", lambda ev: self._light(True))
        self.bind("<Leave>", lambda ev: self._light(False))

    @property
    def state(self):
        return str(tk.Canvas.cget(self, "state")) or "normal"

    def cget(self, key):
        """Reads like the Button it replaced: `text` and `state` answer."""
        return self.text if key == "text" else tk.Canvas.cget(self, key)

    def _click(self, _ev):
        if self.state == "normal":
            self.command()
        return "break"

    def _light(self, on):
        self.lit = on
        if self.C:
            self.paint(self.C)

    def set(self, text=None, state=None):
        if text is not None:
            self.text = text
        if state is not None:
            self.config(state=state)      # the canvas's own option
        if self.C:
            self.paint(self.C)

    def paint(self, C):
        self.C = C
        bg, fg, active, off, off_fg = (C[r] for r in self.roles)
        w = self.font.measure(self.text) + 2 * self.padx
        h = self.font.metrics("linespace") + 2 * self.pady
        self.config(width=w, height=h)
        self.delete("all")
        fill = off if self.state != "normal" else active if self.lit else bg
        rounded(self, 0, 0, w, h, self.r, fill=fill, outline=fill)
        self.create_text(w / 2, h / 2, text=self.text, font=self.font,
                         fill=off_fg if self.state != "normal" else fg)
        self.config(cursor="hand2" if self.state == "normal" else "arrow")


APP_NAME_CHARS = 16                       # sidebar rows, before the ellipsis
LLM_PC = "LLM PC"                         # the sidebar's second group: remote apps


def app_subtitle(a):
    """The second line of a sidebar row. Also what the rail is measured on."""
    sub = a["version"] or ("remote" if a.get("remote") else "installed")
    return sub + "  ·  drivable" if a["drivable"] else sub


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff")
ATTACH_TYPES = [("All files", "*.*"), ("Pictures", " ".join("*" + e for e in IMAGE_EXTS))]
PREVIEWABLE = (".png", ".gif")            # what Tk 8.6 can decode without PIL
ATTACH_LIMIT = 200_000_000                # bytes; a picture, not a video
LIST_LIMIT = 40                           # folder entries named in the brief


def is_picture(path):
    """Whether an attachment is a picture - the ones the vision model is
    asked about and the transcript tries to show."""
    return os.path.splitext(path)[1].lower() in IMAGE_EXTS


def image_dims(path):
    """(width, height) from the file header, or None. PNG, GIF and JPEG only -
    the formats a camera, a screenshot or an export actually produces - and
    read without decoding, so a 200 MB TIFF costs nothing to attach."""
    try:
        with open(path, "rb") as f:
            head = f.read(32)
            if head[:8] == b"\x89PNG\r\n\x1a\n":
                return (int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big"))
            if head[:6] in (b"GIF87a", b"GIF89a"):
                return (int.from_bytes(head[6:8], "little"), int.from_bytes(head[8:10], "little"))
            if head[:2] == b"\xff\xd8":
                f.seek(2)
                while True:
                    marker = f.read(2)
                    if len(marker) < 2 or marker[0] != 0xFF:
                        return None
                    if marker[1] in (0xD8, 0x01) or 0xD0 <= marker[1] <= 0xD7:
                        continue
                    size = int.from_bytes(f.read(2), "big")
                    if marker[1] in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                                     0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                        sof = f.read(5)
                        return (int.from_bytes(sof[3:5], "big"), int.from_bytes(sof[1:3], "big"))
                    f.seek(size - 2, 1)
    except (OSError, ValueError, IndexError):
        pass
    return None


def describe_folder(path):
    """A folder as a heading and a bounded listing: what is in it, so the
    model can name a file without a tool call, and how much was left out."""
    try:
        names = sorted(os.listdir(path), key=str.lower)
    except OSError as e:
        return "%s (folder, unreadable: %s) at %s" % (os.path.basename(path) or path,
                                                     e.strerror or e, path)
    files = [n for n in names if os.path.isfile(os.path.join(path, n))]
    dirs = [n for n in names if os.path.isdir(os.path.join(path, n))]
    head = "%s (folder, %d files, %d folders) at %s" % (
        os.path.basename(path) or path, len(files), len(dirs), path)
    shown = [n + "/" for n in dirs] + files
    lines = ["    " + n for n in shown[:LIST_LIMIT]]
    if len(shown) > LIST_LIMIT:
        lines.append("    ... and %d more" % (len(shown) - LIST_LIMIT))
    return "\n".join([head] + lines)


def describe_attachment(path):
    """One line a text model can act on: name, size, dimensions when it is a
    picture, and the path every bridge on this PC opens files by. A folder
    gets its listing."""
    if os.path.isdir(path):
        return describe_folder(path)
    ext = os.path.splitext(path)[1].lstrip(".").upper() or "file"
    try:
        size = os.path.getsize(path)
    except OSError:
        size = 0
    detail = ["%.1f MB" % (size / 1e6) if size >= 1e6 else "%d KB" % max(1, size // 1000)]
    dims = image_dims(path)
    if dims:
        detail.insert(0, "%d x %d" % dims)
    return "%s (%s %s) at %s" % (os.path.basename(path), ", ".join(detail), ext, path)


def attachment_note(paths, app):
    """The paragraph appended to the brief when files or folders are attached.
    Bridges on this PC take the path as it is; the OpenCode container sees
    only its workspace, so attachments are copied in and named by the path
    the container will see."""
    if not paths:
        return ""
    lines = []
    if getattr(app, "container", False):
        folder = os.path.join(app.workspace, "attachments")
        os.makedirs(folder, exist_ok=True)
        for p in paths:
            name = os.path.basename(os.path.normpath(p))
            dest = os.path.join(folder, name)
            if os.path.abspath(dest) != os.path.abspath(p):
                if os.path.isdir(p):
                    shutil.copytree(p, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(p, dest)
            lines.append("- %s (copied into the workspace; the container sees it as "
                         "/workspace/attachments/%s)" % (describe_attachment(dest), name))
        head = "Attached files and folders, copied into the workspace:"
    else:
        head = ("Attached files and folders - on this PC; tools that take a path "
                "(import, place, upload, open, read_file) take these paths as written:")
        lines = ["- " + describe_attachment(p) for p in paths]
    return "\n\n" + head + "\n" + "\n".join(lines)


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

    DEFAULTS = {"theme": "dark", "tabs": None, "pinned": [], "hidden": [], "bridges": []}

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
        # Bridges the user entered by hand. Each record is re-validated by the
        # engine when it is turned into a registry entry; here only the shape.
        bridges = self.data.get("bridges")
        self.data["bridges"] = ([x for x in bridges if isinstance(x, dict)]
                                if isinstance(bridges, list) else [])

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
        self.generation = uuid.uuid4().hex
        self.cancel = threading.Event()
        self.record = tasks.TaskRecord()
        self.record.app_id = app.id
        self.schemas = []
        self.llm = None                   # this tab's model; set by Chat at boot
        self.library = None               # tools the model made, set by Chat
        self.groups = list(app.default_groups)
        self.preview_images = []
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

    @property
    def event_id(self):
        return (self.app.id, self.generation)

    def reset(self):
        self.messages = [{"role": "system", "content": self.app.chat_prompt()}]
        self.record = tasks.TaskRecord()
        self.record.app_id = self.app.id
        self.cancel.clear()
        self.preview_images.clear()
        self._stream_open = False
        self._stream_buf = []

    def close(self):
        self.cancel.set()
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
        self.attachments = []             # file and folder paths waiting in the composer
        self.windows = {}                 # ("tools", app id) / "prefs" -> Toplevel
        self.tool_views = {}              # app id -> that window's Text
        self.configure(bg=self.C["bg"])

        self.q = queue.Queue()
        self.llm = None
        self.model_ids = []               # what the host serves, for per-app picks
        self.vision = None                # eng.Vision, or None when nothing served can see
        self.vision_note = None           # why, said once per tab
        self.host = eng.env_default("STUDIO_HOST", "AE_AGENT_HOST",
                                    fallback=eng.DEFAULT_HOST)
        self.want_model = eng.env_default("STUDIO_MODEL", "AE_AGENT_MODEL")
        self.host_ready = threading.Event()

        eng.load_bridges(self.prefs.get("bridges"))
        self.detected = eng.detect_apps()
        self.hidden = list(self.prefs.get("hidden"))
        self.pinned = list(self.prefs.get("pinned"))

        apps = self._opening_tabs()
        self.sessions = {a.id: self._session(a) for a in apps}
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

    def _session(self, app):
        """A tab's session - the one place one is made, at startup and from the
        new-tab menu alike, so every tab with a bridge gets the library of tools
        the model made for it, beside its settings and tasks. (Restored tabs
        once missed it, and the tool maker worked only in tabs opened later.)"""
        s = Session(app)
        if app.bridged:
            s.library = toolsmith.Library.for_app(app.id, self._data_dir())
        return s

    def _opening_tabs(self):
        """
        Which tabs to open: what was open last time, else every installed app
        plus chat. A registry entry that has since disappeared is dropped silently.
        """
        want = self.prefs.get("tabs")
        if want is not None:
            return [eng.TABS_BY_ID[i] for i in want if i in eng.TABS_BY_ID]
        # Chat needs nothing installed, so it is always worth opening - and on a
        # machine with neither app it is the only tab that can answer anything.
        return (eng.installed_apps() or list(eng.APPS)) + [eng.CHAT]

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
                "add": 0xE710, "link": 0xE71B, "more": 0xE70D, "picture": 0xEB9F,
                "file": 0xE8A5, "folder": 0xE8B7}
        plain = {"pin": 0x2191, "unpin": 0x2193, "close": 0x00D7,
                 "add": 0x002B, "link": 0x21C4, "more": 0x02C5, "picture": 0x25A3,
                 "file": 0x2750, "folder": 0x25AD}
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
        self.marks_px = {"row": self._px(30), "tab": self._px(22),
                         "menu": self._px(18)}
        self.dot_px = self._px(8)
        widest = 0
        for a in self.detected:
            widest = max(widest, self.f_ui.measure(clip(a["name"], APP_NAME_CHARS)),
                         self.f_small.measure(app_subtitle(a)))
        # mark, both glyph buttons, the status dot and every gap between them
        self.side_w = max(self._px(SIDEBAR_W), widest + self._px(126))

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
        self.btn_send.paint(self.C)
        self.composer_paint()
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
        m_file.add_command(label="Resume saved task...", command=self._resume_task)
        m_file.add_command(label="Task progress...", command=self._task_progress)
        m_file.add_command(label="New tab...", accelerator="Ctrl+T",
                           command=lambda: self._tab_menu(self.btn_add))
        m_file.add_command(label="Close tab", accelerator="Ctrl+W",
                           command=self._close_tab)
        m_file.add_separator()
        m_file.add_command(label="Connect an MCP bridge...",
                           command=lambda: self._bridge_dialog())
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

        # Rebuilt each time it opens: a bridge connected by hand is a new entry.
        m_bridge = menu()
        self.m_bridge = m_bridge

        def fill_bridges():
            m_bridge.delete(0, "end")
            for app in eng.APPS:
                m_bridge.add_command(label="%s tools..." % app.name,
                                     command=lambda i=app.id: self._tools_window(i))
            m_bridge.add_separator()
            m_bridge.add_command(label="Choose capabilities for current tab...",
                                 command=self._capabilities)
            m_bridge.add_command(label="Start the current app", command=self._on_fix)
            m_bridge.add_separator()
            m_bridge.add_command(label="Connect an MCP bridge...",
                                 command=lambda: self._bridge_dialog())
        self._fill_bridge_menu = fill_bridges
        fill_bridges()
        m_bridge.config(postcommand=fill_bridges)
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
                        ("<Control-o>", self._on_attach),
                        ("<Control-O>", self._on_attach_folder),
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
        c.create_image(size / 2 + 1, size / 2 + 1, image=self._disc(role, size))
        self.dot_role[c] = role
        return c

    def _disc(self, role, size):
        """The status dot as an antialiased image - a canvas oval this small
        comes out as an octagon. Cached by colour, since a theme switch
        changes what every role means."""
        key = ("disc", self.C[role], size)
        photo = self.photos.get(key)
        if photo is None:
            data = icons.disc_png(self.C[role], size)
            photo = tk.PhotoImage(data=base64.b64encode(data).decode("ascii"),
                                  master=self)
            self.photos[key] = photo
        return photo

    def _set_dot(self, canvas, role):
        self.dot_role[canvas] = role
        try:
            size = int(canvas.cget("width")) - 2
            canvas.itemconfig(1, image=self._disc(role, size))
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

    def _hover(self, row, widgets, base, lit, on=None):
        """
        Light a whole row on hover. <Leave> also fires when the pointer moves
        onto a child, so check where it actually went before unlighting.
        `on`, if given, is told whether the row is lit - for controls that
        only appear while the pointer is over the row.
        """
        def paint(role):
            for w in widgets:
                try:
                    w.config(bg=self.C[role])
                except tk.TclError:
                    pass
            if on is not None:
                on(role == lit)

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
        strip.bind("<Configure>", self._fit_tabs)

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
                            "close": close, "mark": mark, "compact": False,
                            "bgs": [tab, inner, lbl, mark, dot, close]}
        self._hook_click(tab, lambda ev, i=sid: self._select(i))
        # after _hook_click, so the close glyph keeps its own handler
        close.bind("<Button-1>", lambda ev, i=sid: (self._close_tab(i), "break")[1])
        self._tip(mark, app.name)         # the name, once the label is folded away
        self._fit_tabs()

    def _compact_tab(self, sid, on):
        """Fold a tab down to its mark and dot, or unfold it. Pack order is
        mark, label, dot, close; the label goes back in before the dot."""
        ui = self.tab_ui[sid]
        if ui["compact"] == on:
            return
        ui["compact"] = on
        if on:
            ui["label"].pack_forget()
            ui["close"].pack_forget()
        else:
            ui["label"].pack(side="left", padx=(8, 8), before=ui["dot"])
            ui["close"].pack(side="left", padx=(8, 0))

    def _fit_tabs(self, _ev=None):
        """
        Seven apps do not fit across a small window as labelled tabs, and Tk's
        packer answers overflow by pushing the last tabs off the edge, unmapped
        and unreachable. So the strip folds instead: when the labelled row is
        wider than the strip, every tab but the active one drops to its mark
        and status dot - the mark carries the name as a tooltip - and unfolds
        again when there is room. Called on every add, close, select and resize.
        """
        strip = self.tabbar.master
        avail = strip.winfo_width() - self.btn_add.winfo_reqwidth() - 8
        if avail <= 1 or not self.tab_ui:
            return                        # not laid out yet; <Configure> will call back
        # Measured from the parts, not the packed tab: a part's requested width
        # is known at once, while the tab's own needs an idle pass - and an idle
        # pass from inside a <Configure> handler re-enters this method.
        # Paddings are the literal ones _make_tab packs with.
        def labelled(ui):
            return (ui["mark"].winfo_reqwidth() + ui["dot"].winfo_reqwidth() + 22
                    + ui["label"].winfo_reqwidth() + ui["close"].winfo_reqwidth() + 24)
        fold = sum(labelled(ui) for ui in self.tab_ui.values()) > avail
        for sid in self.tab_ui:
            self._compact_tab(sid, fold and sid != self.active)

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
        """Which app to talk to. Every drivable app is offered, and Chat - the
        model with no app behind it; one already open switches to it rather
        than opening a second."""
        m = self._menu()
        for app in eng.TABS:
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
        s = self._session(eng.TABS_BY_ID[app_id])
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
        s.cancel.set()
        self.order.remove(sid)
        ui = self.tab_ui.pop(sid)
        ui["tab"].destroy()
        s.frame.destroy()
        self._fit_tabs()
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
        self._fit_tabs()
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
        # Pinning orders a row within its own group: a remote app pinned to the
        # top still lives on the LLM PC, and the heading has to stay true.
        rows.sort(key=lambda a: self.pinned.index(a["name"])
                  if a["name"] in self.pinned else len(self.pinned))
        local = [a for a in rows if not a.get("remote")]
        remote = [a for a in rows if a.get("remote")]
        for a in local:
            self._app_row(a)
        if not local:
            msg = ("every app is hidden"
                   if any(not a.get("remote") for a in self.detected)
                   else "no creative apps found")
            self._skin(tk.Label(self.applist, text=msg, font=self.f_small),
                       bg="side", fg="faint").pack(padx=18, pady=2, anchor="w")
        # The second group, under its own heading. The first heading is the
        # rail's header and stays put; this one comes and goes with its rows.
        if remote:
            self._cap(self.applist, "ON %s" % LLM_PC).pack(
                fill="x", padx=(18, 12), pady=(16, 8))
            for a in remote:
                self._app_row(a)

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
        # The glyphs only appear while the pointer is over the row. Each sits
        # in a slot frozen at the glyph's own size, so unmapping the glyph
        # leaves the slot - and the dot beside it - exactly where they were.
        # (Painting the glyph in the row colour instead leaves a ClearType
        # ghost.) A pinned app's pin is state, and stays put.
        def slot():
            f = self._skin(tk.Frame(row), bg="side")
            f.pack(side="right")
            return f
        hide_slot, pin_slot = slot(), slot()
        hide = self._glyph(hide_slot, "close", lambda w, n=name: self._hide_app(n),
                           tip="Remove %s from this list" % name)
        pin = self._glyph(pin_slot, "unpin" if pinned else "pin",
                          lambda w, n=name: self._pin_app(n),
                          fg="accent" if pinned else "faint",
                          tip="Unpin %s" % name if pinned else "Pin %s to the top" % name)
        for f, g in ((hide_slot, hide), (pin_slot, pin)):
            f.config(width=g.winfo_reqwidth(), height=g.winfo_reqheight())
            f.pack_propagate(False)
            g.pack()

        def reveal(lit):
            for g in (hide,) if pinned else (hide, pin):
                if lit:
                    g.pack()
                else:
                    g.pack_forget()
        reveal(False)
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

        widgets = [row, box, title, subtitle, mark, hide, pin, hide_slot, pin_slot]
        if dot is not None:
            widgets.append(dot)
            row.config(cursor="hand2")
            for w in (row, box, title, subtitle, mark):
                w.bind("<Button-1>", lambda ev, i=a["id"]: self._add_tab(i))
        self._hover(row, widgets, "side", "hover", on=reveal)
        for w in widgets:
            w.bind("<Button-3>", lambda ev, r=a: self._app_context(ev, r), add="+")

    def _app_context(self, ev, a):
        name = a["name"]
        m = self._menu()
        spec = eng.APPS_BY_ID.get(a["id"]) if a["drivable"] else None
        if a["drivable"]:
            m.add_command(label="Chat with %s" % name,
                          command=lambda i=a["id"]: self._add_tab(i))
            if spec is not None and spec.custom:
                m.add_command(label="Edit the bridge...",
                              command=lambda sp=spec: self._bridge_dialog(spec=sp))
                m.add_command(label="Forget the bridge",
                              command=lambda sp=spec: self._forget_bridge(sp))
            m.add_separator()
        else:
            # No bridge written here. The user may have one - installed from
            # anywhere - and can connect it as a command line.
            m.add_command(label="Connect an MCP bridge for %s..." % name,
                          command=lambda r=a: self._bridge_dialog(row=r))
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
        they are offering. Which bridges exist is in the menu behind it.

        Every tab with a bridge counts, the chat tab's in-process one included:
        it is connected or it is not, and its tools are tools."""
        lead, lbl = self.conn["bridges"]
        bridged = [s for s in self.sessions.values() if s.app.bridged]
        live = [s for s in bridged if s.ready]
        tools = sum(len(s.tools) for s in live)
        roles = [s.bridge[0] for s in bridged]
        if not bridged:
            role, detail = "faint", "no tab open\n%d available" % len(eng.TABS)
        elif live:
            role = "ok" if len(live) == len(bridged) else "warn"
            detail = "%d of %d connected\n%d tools" % (len(live), len(bridged),
                                                       tools)
        else:
            role = "err" if "err" in roles else "faint"
            detail = "%d bridge%s\nnot started" % (len(bridged),
                                                   "" if len(bridged) == 1 else "s")
        lead.config(fg=self.C[role])
        lbl.config(text=detail)

    def _menu_bridges(self):
        """What bridges exist, and what each one can currently do."""
        m = self._menu()
        for app in eng.TABS:
            if not app.bridged:
                continue
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

        library = s.library if s is not None else None
        if library is None:
            # No tab open, or one that has not booted: read the app's made tools
            # off disk and list them all, with no tool set to check them against.
            library = toolsmith.Library.for_app(app_id, self._data_dir())
            library.load()
        made = library.ordered()
        if made:
            view.insert("end", "MADE BY THE MODEL   (in this app's tabs)\n", "group")
            for tool in made:
                view.insert("end", tool.name, "name")
                view.insert("end", "   ")
                view.window_create("end",
                                   window=self._forget_button(view, app_id, tool.name))
                view.insert("end", "\n")
                view.insert("end", clip(" ".join(tool.description.split()), 400)
                            + "\nruns " + tool.summary() + "\n", "desc")
        view.config(state="disabled")

    def _forget_button(self, parent, app_id, name):
        """Made tools are the one part of this window the user can change: one
        the model built badly should not need a text editor to get rid of."""
        def forget():
            s = self.sessions.get(app_id)
            if s is not None and s.busy:
                return                    # never change the tool list mid-turn
            library = (s.library if s is not None and s.library is not None else
                       toolsmith.Library.for_app(app_id, self._data_dir()))
            library.remove(name)
            win = self.windows.pop(("tools", app_id), None)
            if win is not None and win.winfo_exists():
                win.destroy()
            self._tools_window(app_id)
        button = tk.Button(parent, text="forget", command=forget, bd=0, relief="flat",
                           font=self.f_small, padx=6, pady=0, cursor="hand2")
        return self._skin(button, bg="card", fg="muted", activebackground="hover",
                          activeforeground="text")

    # ---------------------------------------------------------------- composer
    def _build_composer(self, composer):
        # The input's rounded outline is drawn on a canvas, with the real
        # widgets in a frame placed on top of it. The frame sits far enough
        # inside the curve (INSET against radius R) that its square corners
        # never poke out; the canvas follows the frame's height, so the
        # outline grows with the chip strip and shrinks back with it.
        INSET, R = 7, 14
        shell = tk.Canvas(composer, highlightthickness=0, bd=0)
        self._skin(shell, bg="bg")
        shell.pack(fill="x")
        inner = self._skin(tk.Frame(shell), bg="card")
        item = shell.create_window(INSET, INSET, window=inner, anchor="nw")

        def paint(_ev=None):
            w = shell.winfo_width()
            h = inner.winfo_reqheight() + 2 * INSET
            shell.config(height=h)
            shell.itemconfig(item, width=max(1, w - 2 * INSET))
            shell.delete("frame")
            rounded(shell, 0, 0, w, h, R, fill=self.C["border"],
                    outline=self.C["border"], tags="frame")
            rounded(shell, 1, 1, w - 1, h - 1, R - 1, fill=self.C["card"],
                    outline=self.C["card"], tags="frame")
            shell.tag_lower("frame")
        shell.bind("<Configure>", paint)
        inner.bind("<Configure>", paint)
        self.composer_paint = paint

        # Pictures waiting to go with the next message sit above the input,
        # one chip each; the strip is packed only while it has something in it.
        self.chips = self._skin(tk.Frame(inner), bg="card")
        self.row = self._skin(tk.Frame(inner), bg="card")
        self.row.pack(fill="x")
        row = self.row
        # button first, then the expanding input - same rule as above
        self.btn_send = Pill(row, "Send", self._on_send, self.f_bold,
                             ("accent", "accent_fg", "accent_dk", "border", "faint"))
        self._skin(self.btn_send, bg="card")
        self.btn_send.paint(self.C)
        self.btn_send.pack(side="right", padx=(10, 4), pady=4)
        self.btn_attach = self._glyph(row, "file", self._on_attach, bg="card",
                                      tip="Attach files (Ctrl+O)")
        self.btn_attach.pack(side="left", padx=(12, 0))
        self.btn_folder = self._glyph(row, "folder", self._on_attach_folder, bg="card",
                                      tip="Attach a folder (Ctrl+Shift+O)")
        self.btn_folder.pack(side="left", padx=(4, 0))
        self.input = tk.Text(row, height=2, font=self.f_body, wrap="word", bd=0,
                             padx=14, pady=11, highlightthickness=0)
        self._skin(self.input, bg="card", fg="text", insertbackground="accent",
                   selectbackground="sel")
        self.input.pack(side="left", fill="both", expand=True)
        self.input.bind("<Return>", self._on_return)
        self.input.focus_set()
        hint = tk.Label(composer, text="Enter to send   ·   Shift+Enter for a new line"
                                       "   ·   Ctrl+O to attach files"
                                       "   ·   Ctrl+Tab to switch app",
                        font=self.f_small, anchor="w")
        self._skin(hint, bg="bg", fg="faint")
        hint.pack(fill="x", pady=(6, 0))

    # ------------------------------------------------------------- attachments
    def _on_attach(self, _widget=None):
        paths = filedialog.askopenfilenames(parent=self, title="Attach files",
                                            filetypes=ATTACH_TYPES)
        if paths:
            self._add_attachments(paths)

    def _on_attach_folder(self, _widget=None):
        path = filedialog.askdirectory(parent=self, title="Attach a folder", mustexist=True)
        if path:
            self._add_attachments([path])

    def _add_attachments(self, paths):
        """Queue files and folders for the next message. Bad paths are refused
        here, with a line in the transcript, rather than at send time. Only
        pictures have a size limit: they are the attachments whose bytes are
        read (by the vision model); anything else travels as its path."""
        s = self.cur()
        for p in paths:
            p = os.path.abspath(p)
            if p in self.attachments:
                continue
            try:
                size = 0 if os.path.isdir(p) else os.path.getsize(p)
            except OSError as e:
                if s:
                    self._write(s, "Could not attach %s: %s\n" % (p, e.strerror or e), "err")
                continue
            if is_picture(p) and size > ATTACH_LIMIT:
                if s:
                    self._write(s, "Not attaching %s: %.0f MB is a file to import, not a "
                                   "picture to talk about.\n" % (p, size / 1e6), "err")
                continue
            self.attachments.append(p)
        self._paint_chips()

    def _drop_attachment(self, path):
        if path in self.attachments:
            self.attachments.remove(path)
        self._paint_chips()

    def _paint_chips(self):
        for w in self.chips.winfo_children():
            w.destroy()
        if not self.attachments:
            self.chips.pack_forget()
            return
        for p in self.attachments:
            chip = self._skin(tk.Frame(self.chips), bg="side")
            chip.pack(side="left", padx=(12, 0), pady=(10, 0))
            dims = image_dims(p)
            text = os.path.basename(p) + ("  %d\u00d7%d" % dims if dims else "")
            if os.path.isdir(p):
                text = self.g["folder"] + "  " + os.path.basename(p)
            lbl = tk.Label(chip, text=clip(text, 44), font=self.f_small, padx=8, pady=3)
            self._skin(lbl, bg="side", fg="text")
            lbl.pack(side="left")
            self._tip(lbl, p)
            self._glyph(chip, "close", lambda _w, p=p: self._drop_attachment(p),
                        bg="side", tip="Remove").pack(side="left", padx=(0, 4))
        self.chips.pack(fill="x", before=self.row)

    def _show_attachment(self, s, path):
        """The attachment in the transcript under the message it went with - a
        picture inline where Tk can decode it, a name where it cannot, a
        folder by name with a trailing separator."""
        if os.path.isdir(path):
            self._write(s, "[%s%s]\n" % (os.path.basename(path), os.sep), "hint")
            return
        if path.lower().endswith(PREVIEWABLE):
            try:
                self._show_preview(s, {"file": path})
                return
            except Exception:
                pass
        self._write(s, "[%s]\n" % os.path.basename(path), "hint")

    def _welcome(self, s):
        if not s.app.drivable:
            # Say the one thing this tab is not, before the model has to.
            self._write(s, "Chat with the model - no app attached, so nothing here can "
                           "open or change a project. It can read files and folders on "
                           "this PC and look things up on the web. Try:\n", "sys")
        else:
            self._write(s, "%s. Try:\n" % s.app.name, "sys")
        for e in s.app.examples:
            self._write(s, e + "\n", "hint")

    # ---------------------------------------------------- bridges entered by hand
    def _bridge_dialog(self, row=None, spec=None):
        """
        Connect any MCP stdio bridge as an app tab. `row` is a sidebar app with
        no bridge written here (the name, program and process are filled in);
        `spec` is an existing hand-entered bridge to edit. The dialog is built
        by this method and applied by `_save_bridge`, so a test can drive it
        without posting a window.
        """
        win = tk.Toplevel(self)
        win.title("Connect an MCP bridge")
        win.resizable(False, False)
        win.transient(self)
        self._skin(win, bg="bg")
        body = self._skin(tk.Frame(win), bg="bg")
        body.pack(fill="both", expand=True, padx=22, pady=18)

        name = (spec.name if spec else row["name"] if row else "")
        exe = (spec.exe_path if spec else (row or {}).get("exe") or "")
        probe = spec.probe if spec else ("process:" + os.path.basename(exe) if exe else "")
        command = " ".join([spec.command] + [
            ('"%s"' % a if " " in a else a) for a in spec.args]) if spec else ""
        if spec and " " in spec.command:
            command = '"%s"' % spec.command + command[len(spec.command):]

        self._cap(body, "BRIDGE", bg="bg").pack(fill="x", pady=(0, 8))
        intro = ("Any MCP server that speaks over stdio can drive a tab. Enter the command "
                 "line that starts it - the same one its README puts in an MCP client "
                 "config. Its tools and its own instructions are read when the tab opens.")
        lbl = tk.Label(body, text=intro, font=self.f_small, anchor="w", justify="left",
                       wraplength=self._px(460))
        self._skin(lbl, bg="bg", fg="muted")
        lbl.pack(fill="x", pady=(0, 12))

        fields = {}

        def field(key, label, value, hint):
            self._skin(tk.Label(body, text=label, font=self.f_ui, anchor="w"),
                       bg="bg", fg="text").pack(fill="x")
            var = tk.StringVar(value=value)
            entry = tk.Entry(body, textvariable=var, font=self.f_ui, bd=0,
                             highlightthickness=1, width=58)
            self._skin(entry, bg="card", fg="text", insertbackground="accent",
                       highlightbackground="border", highlightcolor="accent")
            entry.pack(fill="x", ipady=5, pady=(3, 2))
            h = tk.Label(body, text=hint, font=self.f_small, anchor="w", justify="left",
                         wraplength=self._px(460))
            self._skin(h, bg="bg", fg="faint")
            h.pack(fill="x", pady=(0, 10))
            fields[key] = var
            return entry

        first = field("name", "App", name, "What the tab and the sidebar call it.")
        cmd = field("command", "Command line", command,
              'e.g.  npx -y some-mcp-server   or   "C:\\path\\python.exe" server.py')
        field("exe", "Program (optional)", exe,
              "Path of the app's .exe, so the Start button can launch it and the tab "
              "gets its icon.")
        field("probe", "Running check (optional)", probe,
              "process:<Name.exe>, port:<number> or url:<http://...> - how to tell the "
              "app is up. Leave empty if the bridge itself is the only evidence.")
        if row and row["name"] in SUGGESTED_BRIDGES:
            tip = tk.Label(body, text=SUGGESTED_BRIDGES[row["name"]], font=self.f_small,
                           anchor="w", justify="left", wraplength=self._px(460))
            self._skin(tip, bg="bg", fg="muted")
            tip.pack(fill="x", pady=(0, 10))

        err = tk.Label(body, text="", font=self.f_small, anchor="w", justify="left",
                       wraplength=self._px(460))
        self._skin(err, bg="bg", fg="err")
        err.pack(fill="x")

        def save():
            values = {k: v.get() for k, v in fields.items()}
            problem = self._save_bridge(values, replacing=spec)
            if problem:
                err.config(text=problem)
            else:
                win.destroy()

        row_b = self._skin(tk.Frame(body), bg="bg")
        row_b.pack(fill="x", pady=(12, 0))
        ok = tk.Button(row_b, text="Connect" if spec is None else "Save", command=save,
                       font=self.f_ui, relief="flat", padx=16, pady=5, cursor="hand2", bd=0)
        self._skin(ok, bg="accent", fg="accent_fg", activebackground="accent_dk",
                   activeforeground="accent_fg")
        ok.pack(side="right")
        cancel = tk.Button(row_b, text="Cancel", command=win.destroy, font=self.f_ui,
                           relief="flat", padx=12, pady=5, cursor="hand2", bd=0)
        self._skin(cancel, bg="card", fg="text", activebackground="border",
                   activeforeground="text")
        cancel.pack(side="right", padx=(0, 8))
        win.bind("<Return>", lambda ev: save())
        win.bind("<Escape>", lambda ev: win.destroy())
        (first if not name else cmd).focus_set()
        return win

    def _save_bridge(self, values, replacing=None):
        """
        Register a bridge from the dialog's values and open its tab. Returns a
        sentence for the dialog when the values will not do, else None. The
        command line is split the way a shell would; the program and the
        running check are optional.
        """
        name = (values.get("name") or "").strip()
        line = (values.get("command") or "").strip()
        exe = (values.get("exe") or "").strip().strip('"')
        probe = (values.get("probe") or "").strip()
        if not name:
            return "Give the app a name."
        if not line:
            return "Enter the command line that starts the bridge."
        owner = eng.APPS_BY_ID.get(eng.DRIVABLE.get(name))
        if owner is not None and not owner.custom:
            return "%s already has a bridge written into this program; give this one another name." % name
        try:
            command, args = eng.split_command(line)
        except ValueError as e:
            return "That command line does not parse: %s" % e
        if not (shutil.which(command) or os.path.isfile(command)):
            return "%r is not on PATH and is not a file. Use the full path." % command
        if exe and not os.path.isfile(exe):
            return "There is no program at %s." % exe
        if probe and probe.partition(":")[0] not in ("process", "port", "url"):
            return "The running check must start with process:, port: or url:."
        if replacing is not None:
            if replacing.id in self.sessions:
                self._close_tab(replacing.id)
            eng.remove_bridge(replacing.id)
        rec = {"name": name, "command": command, "args": args, "exe": exe, "probe": probe}
        if replacing is not None:
            rec["id"] = replacing.id
        spec = eng.bridge_from_record(rec)
        if spec is None:
            return "Those values do not make a bridge."
        if spec.id in self.sessions:
            self._close_tab(spec.id)      # an older entry with the same id
        eng.add_bridge(spec)
        self._remember_bridges()
        self.detected = eng.detect_apps()
        self._build_apps()
        self._add_tab(spec.id)
        return None

    def _forget_bridge(self, spec):
        if spec.id in self.sessions:
            self._close_tab(spec.id)
        eng.remove_bridge(spec.id)
        self._remember_bridges()
        self.detected = eng.detect_apps()
        self._build_apps()

    def _remember_bridges(self):
        self.prefs.set(bridges=[a.record() for a in eng.custom_bridges()])

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
            self.btn_send.set(text="Send", state="disabled")
            self.btn_new.config(state="disabled")
            return
        text, role, fixable = s.status
        self.lbl_status.config(text=text, fg=self.C[role])
        self.btn_fix.config(text=("Check %s" if s.app.remote else "Start %s") % s.app.name)
        self._show_fix(fixable)
        self.btn_new.config(state="normal")
        self.btn_send.set(text="Stopping…" if s.busy and s.cancel.is_set() else
                          "Stop" if s.busy else "Send",
                          state="disabled" if s.busy and s.cancel.is_set() else "normal")

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
        if isinstance(sid, tuple):
            app_id, generation = sid
            live = self.sessions.get(app_id)
            if live is None or live.generation != generation:
                return
            sid = app_id
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
        elif kind == "preview":
            self._show_preview(s, payload)
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
            if s.app.drivable:
                note = "Bridge connected; ready for a task.\n"
            elif s.app.bridged:           # chat: tools, but no app behind them
                note = "Ready; files on this PC and the web are within reach.\n"
            else:
                note = "Ready.\n"
            self._write(s, note, "sys")
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
        ok, loaded, ids, vision_ids, err = eng.probe_models(self.host)
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
        self.model_ids = ids
        # Every bridge answers a screenshot with a picture and every tab takes
        # attachments; the executing model reads text. One vision model on the
        # same host serves every tab, and its absence is a status, not a silence.
        self.vision, self.vision_note = eng.resolve_vision(self.host, model, vision_ids, loaded)
        def host_row(role, last):
            self.q.put(("host", None, (role, "%s\n%d models\n%s\n%s"
                                       % (pretty_host(self.host), len(ids), clip(model, 24), last))))
        if not self.vision:
            host_row("warn", "no vision model")
            self.host_ready.set()
            return
        seeing = "sees: " + clip(self.vision.model, 18)
        if not self.vision.needs_load:
            host_row("ok", seeing)
            self.host_ready.set()
            return
        # The tabs can boot meanwhile: a picture before the load finishes is
        # loaded just-in-time by the host, only slower.
        host_row("muted", "loading " + clip(self.vision.model, 16))
        self.host_ready.set()
        err = eng.load_model(self.host, self.vision.model)
        if err:
            self.q.put(("sys", None, "Could not load the vision model %s on the host (%s); "
                                     "it will be loaded on first use instead."
                                     % (self.vision.model, err)))
        host_row("ok", seeing)

    def _llm_for(self, s):
        """The shared model, unless this app prefers one the host serves.

        A tab's model is fixed at boot: the executor, both warm-ups and the
        cached prefix on the host all have to agree, and swapping mid-session
        would throw the prefix away. Explains a departure once, in the tab.
        """
        shared = getattr(self.llm, "model", None)
        if shared is None:
            return self.llm
        model, note = s.app.model_for(self.model_ids, shared)
        if note:
            self.q.put(("sys", s.event_id, "Model for this tab: %s (%s)." % (model, note)))
        if model == shared:
            return self.llm
        return eng.LLM(self.host, model, self.llm.temperature, self.llm.timeout)

    def _ensure(self, s):
        """First view of a tab is what starts that app's bridge."""
        if s.ready or s.booting:
            return
        s.booting = True
        self._spawn(s.event_id, self._boot_session, s)

    def _boot_session(self, s):
        sid = s.event_id
        try:
            if not self.host_ready.is_set():
                self.q.put(("status", sid, ("waiting for the inference host",
                                            "muted", False)))
                self.host_ready.wait(timeout=240)
            if self.llm is None:
                self.q.put(("status", sid, ("no inference host", "err", False)))
                self.q.put(("bridge", sid, ("err", "%s\nno model" % s.app.bridge_label)))
                return
            s.llm = self._llm_for(s)
            if self.vision_note:
                self.q.put(("sys", sid, self.vision_note))

            if s.app.bridged:
                self._boot_bridge(s)
                if s.mcp is None:         # it reported its own failure
                    return
                if s.closed:
                    return

            # Prefill dominates the first call - a full tool schema set takes about a
            # minute cold. Pay it here against the exact prompt prefix a real message
            # will use, so the first question comes back in seconds. Each tab has its
            # own prefix, so each warms up the first time it is opened.
            self.q.put(("status", sid, ("warming up the model%s"
                                        % (", about a minute" if s.tools else ""),
                                        "warn", False)))
            try:
                s.llm.chat([{"role": "system", "content": s.app.chat_prompt()},
                            {"role": "user", "content": "Say ready."}],
                           tasks.inference_tools(s.tools, s.library), max_tokens=1)
            except Exception as e:
                self.q.put(("sys", sid, "Warm-up did not finish; the first request may be slower. " + str(e)))
            s.ready = True
            if s.app.bridged:
                self._refresh_bridge(s)
            else:
                self.q.put(("status", sid, ("ready", "ok", False)))
                self.q.put(("bridge", sid, ("ok", "no bridge\nthe model on its own")))
            self.q.put(("ready", sid, None))
        finally:
            s.booting = False
            self.q.put(("idle", sid, None))

    def _boot_bridge(self, s):
        """
        Start this tab's bridge and work out what it offers. On failure it says
        why and leaves `s.mcp` None, which is how the caller knows to stop.
        """
        sid = s.event_id
        self.q.put(("status", sid, ("starting the %s bridge" % s.app.name,
                                    "muted", False)))
        mcp = None
        try:
            mcp = s.app.connect()
            mcp.initialize(timeout=75)
            allt = mcp.list_tools(timeout=45)
        except Exception as e:
            if mcp is not None:
                mcp.close()
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

        s.schemas = allt
        if s.app.custom:
            # Nothing was known about this bridge until it answered: its groups
            # come from its tool names and its briefing from `instructions`, so
            # the system prompt - built at Session() - is rebuilt now, before
            # the warm-up pays for the prefix a real message will use.
            s.app.learn(allt, mcp.instructions)
            s.groups = list(s.app.default_groups)
            s.messages[0] = {"role": "system", "content": s.app.chat_prompt()}
        wanted = s.app.tool_names(s.groups)
        s.tools = eng.to_openai_tools([t for t in allt if t["name"] in wanted])
        self._load_library(s)
        missing = wanted - {t["name"] for t in allt}
        if missing:
            self.q.put(("sys", sid, "This bridge does not provide: " + ", ".join(sorted(missing))))

    def _load_library(self, s):
        """Read back the tools the model made for this app, against the tools
        this tab is currently offering. A made tool whose steps are no longer
        exposed, or whose file has been hand-edited into nonsense, is left out
        and said aloud - it must never reach the model as a tool that cannot run.
        """
        if s.library is None:
            return
        allowed, specs = toolsmith.contracts(s.tools, s.schemas)
        try:
            problems = s.library.load(allowed, specs)
        except Exception as e:
            self.q.put(("sys", s.event_id, "Could not read the tools made here: %s" % e))
            return
        if problems:
            self.q.put(("sys", s.event_id, "Not offering %d tool(s) made here: %s"
                        % (len(problems), "; ".join(problems))))

    def _bridge_help(self, app, err):
        if app.custom:
            return ("Could not start the bridge you connected for %s.\n"
                    "Its command line is:\n    %s\nRun that in a terminal to see the real "
                    "error, or right-click %s in the sidebar to edit or forget the bridge."
                    "\n\n(%s)" % (app.name, " ".join([app.command] + app.args), app.name, err))
        if app.id in ("photoshop", "illustrator", "premiere"):
            return ("Could not start the %s bridge.\n"
                    "It is a script beside this program and needs only Python (and "
                    "PowerShell); run this in a terminal to see the real error:\n"
                    "    python %s --list-tools\n\n(%s)" % (app.name, app.args[0], err))
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
        if not s.app.bridged:
            return
        n = len(s.tools)
        if s.app.running():           # always, for a tab with no app to run
            self.q.put(("status", s.event_id, ("connected", "ok", False)))
            self.q.put(("bridge", s.event_id, ("ok", "%s\n%d tools" % (s.app.bridge_label, n))))
        elif s.app.remote:
            # Nothing here can start it; the button re-probes instead.
            self.q.put(("status", s.event_id, ("%s is not reachable" % s.app.name, "warn", True)))
            self.q.put(("bridge", s.event_id, ("warn", "%s\nnot reachable" % s.app.bridge_label)))
        else:
            self.q.put(("status", s.event_id, ("%s is not running" % s.app.name, "warn", True)))
            self.q.put(("bridge", s.event_id, ("warn", "%s\nnot running" % s.app.bridge_label)))

    def _on_fix(self):
        s = self.cur()
        if s is None or s.busy:
            return
        if not s.app.drivable:
            self._write(s, "This tab has no app to start - it is the model on "
                           "its own.\n", "sys")
            return
        s.busy = True
        s.cancel.clear()
        self._apply_status()
        self._spawn(s.event_id, self._fix, s)

    def _fix(self, s):
        try:
            if s.app.remote:
                # A remote app cannot be launched from here: re-probe, and say
                # where it has to be started if it still does not answer.
                if not s.app.running():
                    self.q.put(("sys", s.event_id, "%s is not answering at %s. %s"
                                % (s.app.name, s.app.bridge_label, s.app.launch_note)))
            elif not s.app.running():
                self.q.put(("sys", s.event_id, "Launching %s..." % s.app.name))
                self.q.put(("status", s.event_id, ("launching %s" % s.app.name, "warn", False)))
                try:
                    s.app.launch()
                except RuntimeError as e:
                    # launch() refuses in prose (nothing installed, no program
                    # path): that is an answer for the transcript, not a crash
                    # for the log, and the header must fall back to "not
                    # running" so the button comes back.
                    self.q.put(("error", s.event_id, str(e)))
                    self._refresh_bridge(s)
                    return
                for _ in range(60):
                    if s.cancel.is_set():
                        return
                    if s.app.running():
                        self.q.put(("sys", s.event_id, "%s is up. %s"
                                    % (s.app.name, s.app.launch_note)))
                        break
                    time.sleep(2)
                else:
                    self.q.put(("error", s.event_id,
                                "%s did not come up within two minutes. %s"
                                % (s.app.name, s.app.launch_note)))
            self._refresh_bridge(s)
        finally:
            self.q.put(("idle", s.event_id, None))

    # ------------------------------------------------------------------ sending
    def _data_dir(self):
        return os.path.dirname(os.path.abspath(self.prefs.path))

    def _task_path(self, s):
        return os.path.join(self._data_dir(), "tasks", s.id, s.record.id + ".json")

    def _resume_task(self):
        s = self.cur()
        if s is None or s.busy:
            return
        path = filedialog.askopenfilename(parent=self, title="Resume a saved task",
            initialdir=os.path.dirname(self._task_path(s)),
            filetypes=[("Studio task", "*.json")])
        if not path:
            return
        try:
            record, messages = tasks.TaskRecord.restore(path, s.app.chat_prompt())
            if record.app_id != s.id:
                raise ValueError("This task belongs to a different app. Open its tab to resume it.")
            # Keep the current conversation recoverable when replacing it.
            if s.record.briefs:
                s.record.save(self._task_path(s), s.messages)
            s.record, s.messages = record, messages
            s.cancel.clear()
            s.view.config(state="normal")
            s.view.delete("1.0", "end")
            s.view.config(state="disabled")
            for msg in messages:
                if msg.get("role") in ("user", "assistant") and isinstance(msg.get("content"), str):
                    self._role(s, "YOU" if msg["role"] == "user" else s.app.tab.upper(), "role_user" if msg["role"] == "user" else "role_asst")
                    self._write(s, msg["content"] + "\n", "user" if msg["role"] == "user" else "asst")
            self._write(s, "Task restored. Send a message to continue; the current project must be inspected first.\n", "sys")
        except Exception as e:
            self._write(s, "Could not restore task: %s\n" % e, "err")

    def _task_progress(self):
        s = self.cur()
        if s is None:
            return
        # The worker owns the mutable record while busy; show a stable file snapshot.
        try:
            with open(self._task_path(s), encoding="utf-8") as f:
                record = json.load(f)["record"]
        except (OSError, ValueError, KeyError):
            record = {"status": "No saved progress yet."}
        win = tk.Toplevel(self)
        win.title("Task progress — " + s.app.tab)
        view = self._skin(tk.Text(win, wrap="word", font=self.f_body,
                                  width=75, height=28), bg="bg", fg="text")
        view.pack(fill="both", expand=True)
        view.insert("end", str(record.get("status", "No saved progress yet.")) + "\n\n")
        for key, label in (("briefs", "Your requests"), ("plan", "Plan"), ("issues", "Open issues")):
            items = record.get(key, [])
            if items:
                view.insert("end", label + "\n")
                for i, item in enumerate(items, 1):
                    view.insert("end", "%d. %s\n" % (i, item))
                view.insert("end", "\n")
        if record.get("checks"):
            view.insert("end", "Acceptance checks (agent observations)\n")
            for check in record["checks"]:
                view.insert("end", str(check.get("requirement", "")) + "\n  " +
                            (check.get("evidence") or "Not verified yet") + "\n")
            view.insert("end", "\n")
        view.insert("end", "Saved task: " + self._task_path(s))
        view.config(state="disabled")

    def _capabilities(self):
        s = self.cur()
        if s is None or s.busy or s.booting or not s.ready:
            return
        if not s.app.groups:
            self._write(s, "This tab has no bridge, so there are no capabilities "
                           "to choose. Open an app tab for that.\n", "sys")
            return
        win = tk.Toplevel(self)
        win.title("Capabilities — " + s.app.tab)
        self._skin(win, bg="bg")
        variables = {}
        for group in s.app.groups:
            var = tk.BooleanVar(value=group in s.groups)
            variables[group] = var
            control = tk.Checkbutton(win, text=group.capitalize(), variable=var,
                                     font=self.f_body)
            self._skin(control, bg="bg", fg="text", selectcolor="card",
                       activebackground="bg", activeforeground="text")
            control.pack(anchor="w", padx=self._px(16), pady=self._px(3))
            # Default tools remain available because the app prompt names them.
            if group in s.app.default_groups:
                control.config(state="disabled")
        def apply():
            if self.sessions.get(s.id) is not s or s.busy:
                win.destroy()
                return
            s.groups = [g for g, var in variables.items() if var.get()]
            wanted = s.app.tool_names(s.groups)
            s.tools = eng.to_openai_tools([t for t in s.schemas if t["name"] in wanted])
            # A made tool built on a tool this tab no longer offers is dropped
            # from the model's list rather than failing when it is called.
            self._load_library(s)
            s.busy = True
            s.cancel.clear()
            s.status = ("warming selected capabilities", "warn", False)
            self._apply_status()
            self._spawn(s.event_id, self._warm_capabilities, s)
            win.destroy()
        button = self._skin(tk.Button(win, text="Apply", command=apply),
                            bg="accent", fg="accent_fg")
        button.pack(padx=self._px(16), pady=self._px(12))

    def _warm_capabilities(self, s):
        try:
            s.llm.chat([{"role": "system", "content": s.app.chat_prompt()},
                        {"role": "user", "content": "Say ready."}],
                       tasks.inference_tools(s.tools, s.library), max_tokens=1)
            self.q.put(("sys", s.event_id, "Selected capabilities are ready."))
        finally:
            self.q.put(("status", s.event_id, ("ready", "ok", False)))
            self.q.put(("idle", s.event_id, None))

    def _show_preview(self, s, item):
        try:
            if item.get("file"):
                photo = tk.PhotoImage(file=item["file"], master=self)
            else:
                data = item.get("data", "")
                if len(data) > 20_000_000:
                    raise ValueError("preview exceeds the display size limit")
                photo = tk.PhotoImage(data=data, master=self)
            factor = max(1, (photo.width() + self._px(639)) // self._px(640),
                         (photo.height() + self._px(359)) // self._px(360))
            if factor > 1:
                photo = photo.subsample(factor)
            s.preview_images.append(photo)
            s.view.config(state="normal")
            s.view.image_create("end", image=photo)
            s.view.insert("end", "\n")
            s.view.config(state="disabled")
            s.view.see("end")
        except Exception as e:
            if item.get("file"):
                raise
            self._write(s, "Preview could not be displayed: %s\n" % e, "sys")

    def _on_return(self, ev):
        if ev.state & 0x0001:  # Shift+Enter = newline
            return None
        if self.cur() is None or not self.cur().busy:
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
        if s is None:
            return
        if s.busy:
            s.cancel.set()
            s.status = ("stopping after the current operation; completed edits remain", "warn", False)
            self._apply_status()
            return
        task = self.input.get("1.0", "end").strip()
        attached = list(self.attachments)
        if not task and not attached:
            return
        if not s.ready:
            self._write(s, "%s is still starting up - give it a moment.\n"
                        % s.app.tab, "err")
            return
        task = task or "Take a look at what I attached."
        try:
            note = attachment_note(attached, s.app)
        except OSError as e:
            self._write(s, "Could not hand the attachments over: %s\n" % e, "err")
            return
        self.input.delete("1.0", "end")
        self.attachments = []
        self._paint_chips()
        self._role(s, "YOU", "role_user")
        self._write(s, task + "\n", "user")
        for p in attached:
            self._show_attachment(s, p)
        s.messages.append({"role": "user", "content": task + note})
        s.record.briefs.append(task + note)
        s.cancel.clear()
        s.busy = True
        s.status = ("working", "warn", False)
        self._apply_status()
        # The vision model is asked about pictures only; the rest are paths.
        self._spawn(s.event_id, self._turn, s, [p for p in attached if is_picture(p)])

    def _turn(self, s, pictures=()):
        sid = s.event_id
        started = {"value": False}
        def emit(kind, payload):
            if kind == "token":
                if not started["value"]:
                    self.q.put(("stream_start", sid, None))
                    started["value"] = True
            elif kind == "stream_end":
                started["value"] = False
            self.q.put((kind, sid, payload))
        def checkpoint():
            s.record.save(self._task_path(s), s.messages)
        try:
            if pictures and self.vision:
                # The executing model reads text. Put what the pictures show
                # into the brief itself, so it survives checkpoints and resume.
                emit("status", ("looking at the pictures", "muted", True))
                try:
                    s.messages[-1]["content"] += self.vision.describe_all(pictures)
                    s.record.briefs[-1] = s.messages[-1]["content"]
                except Exception as e:
                    emit("sys", "The vision model could not describe the pictures (%s); "
                                "the model has their paths only." % e)
            elif pictures:
                emit("sys", "No vision model is served, so the model has only the names "
                            "and paths of the pictures - it cannot see what is in them.")
            executor = tasks.Executor(s.llm or self.llm, s.mcp, s.tools, schemas=s.schemas,
                record=s.record, cancel=s.cancel, emit=emit, checkpoint=checkpoint,
                vision=self.vision.review if self.vision else None, library=s.library)
            executor.run(s.messages, MAX_STEPS)
        except Exception:
            s.record.status = "interrupted; inspect project state before continuing"
            try:
                checkpoint()
            except Exception:
                pass
            raise
        finally:
            self.q.put(("stream_end", sid, None))
            complete = s.record.status.startswith("response complete")
            status = "stopped" if s.cancel.is_set() else "ready" if complete else "needs attention"
            self.q.put(("status", sid, (status, "muted" if complete else "warn", False)))
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
