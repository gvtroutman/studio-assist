#!/usr/bin/env python3
"""
Studio Assist - a chat window that drives creative apps with a local model.

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
import math
import os
import queue
import re
import shutil
import socket
import subprocess
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
import studio_doctor as doctor
import studio_files as files
import studio_lessons as lessons
import studio_milanote as milanote
import studio_images_ui as images_ui
import studio_procs as procs
import studio_ui as ui
import studio_icons as icons
import studio_tasks as tasks
import studio_update as updater
import studio_toolsmith as toolsmith

APP_NAME = "Studio Assist"
# The log, the icon and %LOCALAPPDATA%\StudioAssistant keep their old spelling
# on purpose: those are paths on disk, not the name on the window, and renaming
# them would orphan the settings, lessons and task records already written under
# them. The name the user reads is this one constant.
ERROR_LOG = doctor.ERROR_LOG
LOG_MAX_BYTES = doctor.LOG_MAX_BYTES
ICON_FILE = os.path.join(HERE, "studio-assistant.ico")

# The palette, the primitives drawn from it and the rounded `Pill` live in
# studio_ui: appearance alone, and the one part of the window that is.
# Re-exported here because this is where the rest of the app reaches for them.
DARK, LIGHT = ui.DARK, ui.LIGHT
THEMES, THEME_NAMES = ui.THEMES, ui.THEME_NAMES
blend, rounded, clip, pretty_host = ui.blend, ui.rounded, ui.clip, ui.pretty_host
Pill = ui.Pill

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
CALL_TEXT_LIMIT = 12_000                  # chars of one argument or result shown in a folded call row

# ------------------------------------------------------------------- animation
# Everything that moves is driven from one `after` tick, for the same reason
# the queue pump is: a timer per animation is a timer per orphan on the way
# out, and Tk names every one of them. The tick is armed only while something
# is registered, so a window with nothing happening in it costs nothing.
ANIM_MS = 70                              # one frame
ELLIPSIS = "…"                       # the marker: a label ending in this is still working
SEND = "→"                           # the send button's label while idle: an arrow, not a word
PANEL_HINT = ("Sign in here once and this tab keeps it. Upload files drops "
              "them onto the board you have open.")
REVEAL_FRAMES = 7                         # a picture wipes in over about half a second
SWEEP_FRAMES = 26                         # one pass of the arc across a bridge row
RIPPLE_FRAMES = 24                        # one pulse out of a placeholder's centre dot: about 1.7s
RIPPLE_SPEED = 0.55                       # grid steps the ripple travels per frame
RIPPLE_WIDTH = 1.8                        # grid steps a dot takes to swell and settle as it passes
SHADES = 12                               # brightness steps a pulse is quantised to, so the disc cache stays small

# Events that put something at the end of a transcript, and so have to take the
# thinking dots down before they do. Kept beside the handler that reads it.
WRITES_TO_TRANSCRIPT = frozenset((
    "sys", "tool", "tool_result", "preview", "ask", "stream_start", "token",
    "error", "ready"))


def makes_a_picture(name):
    """Does this tool's name say it is about to produce an image? The answer
    decides whether a placeholder is drawn where the picture will land, and
    being wrong costs nothing in either direction: a placeholder no image
    arrives for is cleared when the call returns, and a generator not guessed
    here simply appears without one. So the list is generous on purpose."""
    return any(word in name.lower() for word in
               ("generate", "render", "workflow", "screenshot", "snapshot",
                "preview", "thumbnail", "comfy_wait", "comfy_fetch"))


APP_NAME_CHARS = 16                       # sidebar rows, before the ellipsis
DRAIN_MS = 40                             # the pump, while events flow
DRAIN_IDLE_MS = 160                       # ...and while nothing is happening
QUIT_GRACE_S = 3.0                        # every bridge's time to exit, together
HOST_RETRY_MS = 30000                     # between probes while the host is down
UPDATE_FIRST_MS = 8000                    # the first look at GitHub, after start-up settles
UPDATE_EVERY_MS = 15 * 60 * 1000          # ...and again while the window is open
LLM_PC = "LLM PC"                         # the sidebar's second group: remote apps


def app_subtitle(a):
    """The second line of a sidebar row. Also what the rail is measured on."""
    sub = a["version"] or ("remote" if a.get("remote") else "installed")
    return sub + "  ·  drivable" if a["drivable"] else sub


# Attachments - reading a picture's header, describing a folder, copying into
# a container's workspace - live in studio_files: none of it is about the
# window, and it is worth testing without a display. Re-exported here because
# this is where the rest of the app reaches for them.
IMAGE_EXTS = files.IMAGE_EXTS
ATTACH_TYPES = files.ATTACH_TYPES
PREVIEWABLE = files.PREVIEWABLE
ATTACH_LIMIT = files.ATTACH_LIMIT
LIST_LIMIT = files.LIST_LIMIT
is_picture = files.is_picture
image_dims = files.image_dims
describe_folder = files.describe_folder
describe_attachment = files.describe_attachment
attachment_note = files.attachment_note
this_pc = files.this_pc


# Where the app keeps its things, and the one writer for the error log, live
# in studio_doctor: `--doctor` has to work when the window will not, so that
# module knows the layout of %APPDATA%\StudioAssistant and imports no tkinter.
# Re-exported here because this is where the rest of the app reaches for them.
settings_path = doctor.settings_path
error_log_path = doctor.error_log_path
log_error = doctor.log_error


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
        self.notebook = None              # lessons kept for this app, set by Chat
        self.studio = ""                  # the studio brief's text, set by Chat
        self.sidecar = None               # the research bridge, in process
        self.sidecar_names = frozenset()  # its tools, riding beside the bridge's
        self.ask_buttons = []             # the question form waiting for a click
        self.call_seq = 0                 # folded call rows in the transcript, numbered
        self.open_calls = {}              # tool name -> rows still awaiting a result
        self.pending = {}                 # row number -> its animated dots, while it waits
        self.stages = []                  # placeholders where a picture is expected to land
        self.stage_seq = 0                # numbers their tags, like call_seq numbers rows
        self.thinking = None              # the dots shown while the model has the floor
        self.groups = list(app.default_groups)
        self.preview_images = []
        self.status = ("not started", "muted", False)
        self.bridge = ("faint", "%s\nnot started" % app.bridge_label)
        self.host_down = False            # stuck without the inference host; Connect fixes it
        # What this tab's fixed prefix costs, and the window behind it. Measured
        # at the warm-up and kept, because "prompt_tokens against
        # loaded_context_length" is the first thing to look at when a tab talks
        # instead of calling a tool, or repeats one call - and until now the
        # app measured it, acted on it and forgot it. Diagnostics shows it.
        self.prefix_tokens = None
        self.window = None
        self.frame = None
        self.view = None
        self.hero = None                  # the app's mark and name while nothing is said
        self.browser = None               # a panel tab's window (studio_milanote.Browser)
        self.panel_host = None            # ...the frame it is held in
        self.panel_note = None            # ...and the line above it that speaks
        self.images = None                # the Image Studio's form (studio_images_ui)
        self._stream_open = False
        self._stream_buf = []
        self._asst_start = "1.0"

    @property
    def id(self):
        return self.app.id

    @property
    def event_id(self):
        return (self.app.id, self.generation)

    def prompt(self):
        """This tab's system prompt: the app's, with the studio brief and every
        lesson kept for the app. Built at boot and on New chat, never mid-way -
        it is the head of the host's cached prefix."""
        kept = self.notebook.brief() if self.notebook is not None else ""
        return self.app.chat_prompt(self.studio, kept)

    def offered(self, wanted):
        """The tools this tab offers the model: the bridge's in `wanted`, then
        the research sidecar's, in schema order."""
        return eng.to_openai_tools([t for t in self.schemas
                                    if t["name"] in wanted or t["name"] in self.sidecar_names])

    def reset(self):
        self.messages = [{"role": "system", "content": self.prompt()}]
        self.ask_buttons = []
        self.call_seq = 0
        self.open_calls = {}
        # The widgets these held were embedded in the transcript, and clearing
        # it destroyed them; their animations drop themselves the next frame.
        self.pending = {}
        self.stages = []
        self.stage_seq = 0
        self.thinking = None
        self.record = tasks.TaskRecord()
        self.record.app_id = self.app.id
        self.cancel.clear()
        self.preview_images.clear()
        self._stream_open = False
        self._stream_buf = []

    def close(self):
        self.cancel.set()
        browser, self.browser = self.browser, None
        if browser is not None:
            browser.close()
        images, self.images = self.images, None
        if images is not None:
            images.close()                # cancels its jobs on the backends
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
        self.draft_note = None            # a STUDIO_DRAFT_MODEL the host lacks, said once per tab
        self.host_role, self.host_last = "faint", ""   # the Inference row, re-posted per turn
        self.host_ok = None               # the row as the probe left it, restored after a loss
        self.host = eng.env_default("STUDIO_HOST", "AE_AGENT_HOST",
                                    fallback=eng.DEFAULT_HOST)
        self.want_model = eng.env_default("STUDIO_MODEL", "AE_AGENT_MODEL")
        self.host_ready = threading.Event()
        self.fit_lock = threading.Lock()   # one tab fits the shared model at a time
        self.host_booting = False         # a probe is running; Connect waits its turn
        self.host_timer = None            # the next quiet probe, while the host is down
        self.drain_timer = None           # the queue pump's next tick
        self.anim_timer = None            # the one animation tick; None while nothing moves
        self.anim = {}                    # key -> draw(frame); see _animate
        self.anim_frame = 0
        self.arcs = {}                    # canvas -> role, for the drawn bridge arc
        self.pills = []                   # every Pill in the window, for _theme
        self.row_role = {}                # rail row canvas -> the role it is drawn in
        self.repaints = []                # (widget, draw) for shapes _theme must redraw
        self.closing = False              # set by _quit, so no timer outlives the window

        # What the user wrote about the studio: File > About this studio...
        self.studio = eng.read_studio_brief(self._studio_path())

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
        self.drain_timer = self.after(40, self._drain)
        self.restart = False              # set by an update; main() relaunches
        self.updating = False
        self.update_timer = self.after(UPDATE_FIRST_MS, self._update_tick)
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
        s.studio = self.studio
        if app.bridged:
            s.library = toolsmith.Library.for_app(app.id, self._data_dir())
            s.notebook = lessons.Notebook.for_app(app.id, self._data_dir())
            s.notebook.load()             # a problem is said at boot, in the tab
        s.messages[0] = {"role": "system", "content": s.prompt()}
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
        # Inter when it is installed, Segoe UI when it is not - Tk would
        # otherwise substitute a font of its own choosing without a word. A
        # static Inter install names its semibold as a family of its own; a
        # variable one does not, and Tk can only ask it for bold.
        have = set(tkfont.families())
        sans = "Inter" if "Inter" in have else "Segoe UI"
        semi = next((f for f in (sans + " SemiBold", sans + " Semibold")
                     if f in have), None)
        self.f_ui = tkfont.Font(family=sans, size=10)
        self.f_bold = tkfont.Font(family=sans, size=10, weight="bold")
        self.f_title = (tkfont.Font(family=semi, size=13) if semi else
                        tkfont.Font(family=sans, size=13, weight="bold"))
        self.f_small = tkfont.Font(family=sans, size=8)
        self.f_cap = tkfont.Font(family=sans, size=8, weight="bold")
        self.f_badge = tkfont.Font(family=sans, size=9, weight="bold")
        # An empty tab's name and badge, in the middle of the transcript.
        self.f_hero = (tkfont.Font(family=semi, size=20) if semi else
                       tkfont.Font(family=sans, size=20, weight="bold"))
        self.f_hero_badge = tkfont.Font(family=sans, size=22, weight="bold")
        self.f_mono = tkfont.Font(family="Consolas", size=9)
        self.f_body = tkfont.Font(family=sans, size=11)
        self.f_body_b = tkfont.Font(family=sans, size=11, weight="bold")
        # Segoe MDL2 Assets is Windows' own icon font - a pin drawn by the OS
        # beats anything hand-plotted at 9pt. Fall back to punctuation if it is
        # somehow missing. The bridges' mark is not in here: it is `_arc`,
        # which has a state to show and no glyph can.
        have = "Segoe MDL2 Assets" in set(tkfont.families())
        self.f_glyph = tkfont.Font(family="Segoe MDL2 Assets" if have else "Segoe UI",
                                   size=9)
        # By codepoint: these are private-use characters that paste into an
        # editor as blanks, and MDL2 is documented by its hex codes anyway.
        mdl2 = {"pin": 0xE718, "unpin": 0xE77A, "close": 0xE8BB,
                "add": 0xE710, "more": 0xE70D, "picture": 0xEB9F,
                "file": 0xE8A5, "folder": 0xE8B7, "closed": 0xE76C, "open": 0xE70D}
        plain = {"pin": 0x2191, "unpin": 0x2193, "close": 0x00D7,
                 "add": 0x002B, "more": 0x02C5, "picture": 0x25A3,
                 "file": 0x2750, "folder": 0x25AD, "closed": 0x203A, "open": 0x02C5}
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
                         "menu": self._px(18), "hero": self._px(72)}
        self.dot_px = self._px(8)
        # A tab chip's inset and corner radius. The frame inside it must clear
        # the curve - a corner of radius r bulges r*(1 - 1/root 2) past the
        # inset, so the pad is comfortably over a third of the radius - and
        # `_fit_tabs` measures against the same pad rather than a literal.
        self.tab_pad, self.tab_r = self._px(5), self._px(9)
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
        for reg in (self.skin, self.dot_role, self.arcs, self.row_role):
            for widget in list(reg):
                try:
                    if not widget.winfo_exists():
                        del reg[widget]
                except tk.TclError:
                    reg.pop(widget, None)
        for key, entries in list(self.marks.items()):
            self.marks[key] = [e for e in entries if e[0].winfo_exists()]
        # Chips are rebuilt on every attach and every send, and each one
        # registers a repaint; swept here the list tracks the window instead
        # of the session's history of it.
        self.pills = [p for p in self.pills if p.winfo_exists()]
        self.repaints = [(w, d) for w, d in self.repaints if w.winfo_exists()]

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
        # Drawn canvases are repainted, never reconfigured - the arc is made of
        # palette colours it plotted itself, same as the app marks and the dots.
        for canvas in list(self.arcs):
            self._paint_arc(canvas)
        for pill in self.pills:           # `_forget` above swept both lists
            pill.paint(self.C)
        for _widget, draw in self.repaints:
            try:
                draw()
            except tk.TclError:
                pass
        self.composer_paint()
        for s in self.sessions.values():
            if s.view is not None:
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
        m_file.add_command(label="Chat history...", accelerator="Ctrl+H",
                           command=self._resume_task)
        m_file.add_command(label="Task progress...", command=self._task_progress)
        m_file.add_command(label="Lessons for this tab...", command=self._lessons_window)
        m_file.add_command(label="About this studio...", command=self._studio_window)
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
            m_bridge.add_command(label="Connect to the inference host",
                                 command=self._connect_host)
            m_bridge.add_separator()
            m_bridge.add_command(label="Connect an MCP bridge...",
                                 command=lambda: self._bridge_dialog())
        self._fill_bridge_menu = fill_bridges
        fill_bridges()
        m_bridge.config(postcommand=fill_bridges)
        bar.add_cascade(label="Bridges", menu=m_bridge)

        m_help = menu()
        m_help.add_command(label="Diagnostics...", command=self._diagnostics_window)
        m_help.add_command(label="Check for updates...",
                           command=lambda: self._check_updates(quiet=False))
        m_help.add_separator()
        m_help.add_command(label="About %s" % APP_NAME, command=self._about)
        bar.add_cascade(label="Help", menu=m_help)

        self.config(menu=bar)
        # Bound on the input as well, returning "break", so Tk's own Text
        # bindings (Ctrl+T transposes characters) never also fire.
        for seq, fn in (("<Control-n>", self._on_new),
                        ("<Control-h>", self._resume_task),
                        ("<Control-t>", lambda: self._tab_menu(self.btn_add)),
                        ("<Control-w>", self._close_tab),
                        ("<Control-o>", self._on_attach),
                        ("<Control-O>", self._on_attach_folder),
                        ("<Control-comma>", self._prefs_window)):
            self.bind_all(seq, lambda ev, f=fn: (f(), "break")[1])
            self.input.bind(seq, lambda ev, f=fn: (f(), "break")[1])
        self.bind_all("<Control-Tab>", self._on_next_tab)
        self.input.bind("<Control-Tab>", self._on_next_tab)

    # ----------------------------------------------------------------- updates
    def _update_tick(self):
        """Look at GitHub now and again while the window is open. The header's
        Update button appears when there is something to pull."""
        if self.closing:
            return
        self._check_updates(quiet=True)
        self.update_timer = self.after(UPDATE_EVERY_MS, self._update_tick)

    def _check_updates(self, quiet=True):
        def work():
            try:
                st = updater.check()
            except Exception as e:        # a git that hangs past its timeout
                st = {"behind": 0, "ahead": 0, "commits": [], "upstream": "",
                      "problem": "Could not check GitHub: %s" % e}
            self.q.put(("update", None, (st, quiet)))
        threading.Thread(target=work, daemon=True).start()

    def _show_update(self, st, quiet):
        if st["behind"] and not self.updating:
            self.btn_update.set(text="Update (%d)" % st["behind"])
            self.btn_update.pack(side="right", padx=(6, 0), after=self.btn_hist)
        else:
            self.btn_update.pack_forget()
        if quiet:
            return
        if st["problem"]:
            messagebox.showwarning(APP_NAME, st["problem"], parent=self)
        elif st["behind"]:
            self._on_update(st)
        else:
            messagebox.showinfo(APP_NAME, "Up to date with %s." % st["upstream"],
                                parent=self)

    def _on_update(self, st=None):
        """Say what is new, then fast-forward. Only ever a fast-forward: work
        on this PC is never merged over (studio_update.py)."""
        if self.updating:
            return
        if st is None:
            self._check_updates(quiet=False)
            return
        new = st["commits"][:12]
        more = len(st["commits"]) - len(new)
        text = ("%d new change%s on GitHub (%s):\n\n%s%s\n\nUpdate now?"
                % (st["behind"], "" if st["behind"] == 1 else "s", st["upstream"],
                   "\n".join("\u2022 " + c for c in new),
                   "\n\u2026and %d more" % more if more > 0 else ""))
        if not messagebox.askyesno(APP_NAME, text, parent=self):
            return
        self.updating = True
        self.btn_update.set(text="Updating" + ELLIPSIS)

        def work():
            try:
                result = updater.pull()
            except Exception as e:
                result = (False, "The update failed: %s" % e)
            self.q.put(("updated", None, result))
        threading.Thread(target=work, daemon=True).start()

    def _updated(self, changed, msg):
        self.updating = False
        if not changed:
            self.btn_update.set(text="Update")
            messagebox.showwarning(APP_NAME, msg, parent=self)
            return
        self.btn_update.pack_forget()
        if messagebox.askyesno(APP_NAME, "Updated. This window is still running the "
                               "old version.\n\nRestart %s now?" % APP_NAME, parent=self):
            self.restart = True
            self._quit()

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

        # The window's name is in the title bar, where Windows also puts it on
        # the taskbar and in Alt-Tab. Repeating it here said it twice, one line
        # under the other, and bought nothing: the header's job is to say what
        # this tab is doing. So the status starts the row.
        self.lbl_status = tk.Label(head, text="starting", font=self.f_title)
        self._skin(self.lbl_status, bg="head", fg="muted")
        self.lbl_status.pack(side="left", padx=(18, 0))
        # The status's jumping dots, when it describes something still going
        # on: a canvas beside the label, since a Label's text cannot move. As
        # tall as the label and plotted on its baseline, so the two line up
        # without either knowing where the other is.
        self.status_jump = tk.Canvas(head, highlightthickness=0, bd=0,
                                     width=ui.jump_width(self.f_title) + 2)
        self._skin(self.status_jump, bg="head")
        self.btn_new = self._button(head, "New chat", self._on_new, bg="head")
        self.btn_new.pack(side="right", padx=(6, 18))
        # Every tab keeps its app's past conversations; this is the way back
        # to them, beside the button that starts the next one.
        self.btn_hist = self._button(head, "History", self._resume_task, bg="head")
        self.btn_hist.pack(side="right", padx=(6, 0))
        self.btn_fix = self._button(head, "Start app", self._on_fix, bg="head",
                                    kind="accent")
        # Shown only when GitHub has commits this folder does not (_show_update).
        self.btn_update = self._button(head, "Update", self._on_update, bg="head",
                                       kind="accent")

        main = self._skin(tk.Frame(self), bg="bg")
        main.pack(side="top", fill="both", expand=True)

        side = tk.Frame(main, width=self.side_w)
        self._skin(side, bg="side")
        side.pack(side="left", fill="y")
        side.pack_propagate(False)
        self._build_sidebar(side)

        right = self._skin(tk.Frame(main), bg="bg")
        right.pack(side="left", fill="both", expand=True)

        # Fixed-size widgets are packed BEFORE the expanding transcript, on
        # purpose. In Tk an expanding sibling packed first claims the leftover
        # space and shoves later fixed-size widgets off the edge - which is
        # exactly how the composer used to disappear until the window was
        # resized. Composer, then tab strip, then the transcript stack.
        composer = self._skin(tk.Frame(right), bg="bg")
        composer.pack(side="bottom", fill="x", padx=24, pady=(8, 20))
        self.composer = composer          # a panel tab has none; _select hides it
        self._build_composer(composer)

        strip = self._skin(tk.Frame(right), bg="bg")
        strip.pack(side="top", fill="x", padx=14, pady=(10, 6))
        self.strip = strip
        self._build_tabs(strip)

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
        btn = self._button(box, "Choose an app", lambda: None, kind="accent")
        btn.command = lambda: self._tab_menu(btn)
        btn.pack(pady=14)
        self.empty_msg = tk.Label(box, text="", font=self.f_ui, wraplength=420,
                                  justify="center")
        self._skin(self.empty_msg, bg="bg", fg="err")
        self.empty_msg.pack(pady=(4, 0))

    def _build_transcript(self, s):
        if s.app.images:
            self._build_images(s)
            return
        if s.app.panel:
            self._build_panel(s)
            return
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
        self._build_hero(s)
        self._welcome(s)

    def _build_hero(self, s):
        """What a tab shows before anything is said: the app's mark and name
        in the middle of the transcript. Placed over the Text rather than
        written into it, so it sits in the middle at any window height and the
        boot lines keep their place at the top."""
        s.hero = self._skin(tk.Frame(s.frame), bg="bg")
        self._mark(s.hero, self._spec_for(s.app), self.marks_px["hero"],
                   bg="bg").pack()
        self._skin(tk.Label(s.hero, text=s.app.name, font=self.f_hero),
                   bg="bg", fg="text").pack(pady=(self._px(14), 0))
        if not s.app.drivable:
            # Say the one thing this tab is not, before the model has to.
            note = tk.Label(s.hero, font=self.f_ui, justify="center",
                            wraplength=self._px(380),
                            text="No app attached - it reads files and folders on "
                                 "this PC and looks things up on the web.")
            self._skin(note, bg="bg", fg="muted").pack(pady=(self._px(6), 0))

    def _hide_hero(self, s):
        if s.hero is not None:
            s.hero.place_forget()

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
        # A tool call is one folded row: the header in "tool", the glyph that
        # opens it, a red word when it failed, and the body - the call as the
        # model made it, then the result - hidden by a per-row elide tag until
        # the header is clicked. Steps of a made tool sit one indent in.
        v.tag_configure("glyph", font=self.f_glyph)
        v.tag_configure("call_failed", foreground=C["err"])
        v.tag_configure("call_step", lmargin1=40, lmargin2=54)
        v.tag_configure("tool_body", foreground=C["muted"], font=self.f_mono, lmargin1=36,
                        lmargin2=36, rmargin=16, spacing3=2)
        v.tag_configure("tool_body_step", lmargin1=54, lmargin2=54)
        v.tag_bind("call_head", "<Button-1>", self._on_call_click)
        v.tag_bind("call_head", "<Enter>", lambda e: e.widget.config(cursor="hand2"))
        v.tag_bind("call_head", "<Leave>", lambda e: e.widget.config(cursor="arrow"))
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
    # Two kinds of button, and nothing else. `(bg, fg, hover, disabled,
    # disabled fg)` in the order `Pill.paint` reads them.
    PILL_ROLES = {"accent": ("accent", "accent_fg", "accent_dk", "border", "faint"),
                  "quiet": ("card", "text", "border", "card", "faint"),
                  # an option row in the question form: reads as a choice, not
                  # as a control, until the pointer is on it
                  "option": ("bg", "text", "hover", "card", "faint"),
                  # a way out, offered without competing with the choices
                  "ghost": ("card", "muted", "hover", "card", "faint")}

    def _button(self, parent, text, command, kind="quiet", bg="bg", font=None,
                **kw):
        """Every button in this window is a `Pill`. Tk's Button is a rectangle
        and nothing on it bends, so one square control among rounded ones is
        not a style choice, it is the only shape Tk would give. `Pill` draws
        its own, and registering it here is what lets `_theme` repaint it -
        a canvas that plotted its own palette colours cannot be told a new one
        with config()."""
        pill = Pill(parent, text, command, font or self.f_ui,
                    self.PILL_ROLES[kind], **kw)
        pill.disc = self._disc_colour     # for the jumping dots a "…" label draws
        self._skin(pill, bg=bg)
        self.pills.append(pill)
        pill.paint(self.C)
        return pill

    def _repaint_on_theme(self, widget, draw):
        """Register a shape that must be drawn again when the palette changes.

        `_theme` re-reads `skin` and reconfigures widgets, which is enough for
        anything whose colour is a Tk option. It is no use at all to a canvas
        that plotted palette colours into items of its own - a rounded card, a
        chip, a field's outline, the corners masked off a picture. The tab
        chips and the rail rows are already redrawn by `_paint_tab` and
        `_build_apps`; everything else that draws itself belongs here, and is
        swept by widget so a dead one cannot raise mid-switch."""
        self.repaints.append((widget, draw))

    def _entry(self, parent, var, bg="bg"):
        """A text field with a rounded outline: the entry itself on a canvas
        that draws the border, exactly as the composer's input sits inside
        one. Tk's own `highlightthickness` can only draw a rectangle, and a
        square field between rounded buttons is the one shape that gives the
        window away. Returns the entry; its canvas is `entry.master` and that
        is what the caller packs."""
        INSET, R = self._px(6), self._px(10)
        shell = tk.Canvas(parent, highlightthickness=0, bd=0)
        self._skin(shell, bg=bg)
        entry = tk.Entry(shell, textvariable=var, font=self.f_ui, bd=0,
                         highlightthickness=0)
        self._skin(entry, bg="card", fg="text", insertbackground="accent")
        item = shell.create_window(INSET, INSET, window=entry, anchor="nw")
        focused = {"on": False}

        def paint(_ev=None):
            w = shell.winfo_width()
            h = entry.winfo_reqheight() + 2 * INSET
            shell.config(height=h)
            shell.itemconfig(item, width=max(1, w - 2 * INSET))
            shell.delete("box")
            # At rest the field is a raised surface, not a box; the accent
            # ring comes back only to say where typing will go.
            if focused["on"]:
                edge = self.C["accent"]
                rounded(shell, 0, 0, w, h, R, fill=edge, outline=edge, tags="box")
                rounded(shell, 1, 1, w - 1, h - 1, R - 1, fill=self.C["card"],
                        outline=self.C["card"], tags="box")
            else:
                ui.lifted(shell, w, h, R, self.C["card"], self.C[bg], "box")
            shell.tag_lower("box")

        shell.bind("<Configure>", paint)
        entry.bind("<FocusIn>", lambda ev: (focused.__setitem__("on", True), paint()))
        entry.bind("<FocusOut>", lambda ev: (focused.__setitem__("on", False), paint()))
        self._repaint_on_theme(shell, paint)
        return entry

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
        big = size >= self.marks_px["hero"]
        rounded(c, 1, 1, size - 1, size - 1, size * 0.22 if big else 7,
                fill=spec["bg"], outline=spec["bg"])
        c.create_text(size / 2, size / 2 + 1, text=spec["code"], fill=spec["fg"],
                      font=self.f_hero_badge if big else self.f_badge)

    def _dots(self, parent, bg="bg", role="muted", n=3):
        """Three dots jumping in a wave: something is still running, and this
        is where its answer will land. Embedded in the transcript, so it says
        it in the place the user is already looking rather than only in the
        header. The same motion as a status ending in an ellipsis
        (`ui.jumps`), at the transcript's dot size.

        Drawn as the status dot is - `_disc` renders a PNG because a canvas
        oval this small comes out as an octagon. A dot is a touch brighter the
        higher it is, which is what keeps a low one from reading as a stray
        full stop; every shade is cached by colour, so a run of these costs a
        handful of small images and not one per frame. The animation reads
        `self.C` live, which is what makes a theme switch mid-run correct
        without a repaint hook."""
        size = self.dot_px
        gap = size + self._px(5)
        rise = size                       # room above for the hop, and no more
        c = tk.Canvas(parent, width=gap * (n - 1) + size + 2,
                      height=size + rise + 2, highlightthickness=0, bd=0)
        self._skin(c, bg=bg)
        ground = rise + size / 2 + 1      # a resting dot's centre
        ids = [c.create_image(gap * i + size / 2 + 1, ground) for i in range(n)]

        def draw(frame):
            for i, (item, up) in enumerate(zip(ids, ui.jumps(frame, n))):
                c.coords(item, gap * i + size / 2 + 1, ground - up * rise)
                c.itemconfig(item, image=self._shade(role, bg, 0.55 + 0.45 * up,
                                                     size))
        self._animate(("dots", str(c)), draw)
        return c

    def _shade(self, role, bg, level, size):
        """A disc of `role` at `level` brightness over `bg`. Quantised to
        SHADES steps before it is blended, so the cache holds a fixed dozen
        images per pair of roles rather than a new one for every frame of
        every pulse - and at this size the steps are not tellable apart."""
        level = round(min(1.0, max(0.0, level)) * SHADES) / float(SHADES)
        return self._disc_colour(blend(self.C[bg], self.C[role], level), size)

    def _arc(self, parent, bg="side", role="faint"):
        """
        The bridges' mark: a span, drawn. MDL2's chain link says "a link" -
        two things fastened together - and that is not what a bridge is or
        what this row reports. An arc between two banks is, and it is the one
        shape in the window that can also show its own state: a bridge that is
        starting draws itself across from left to right, over and over, and
        stops as a finished span once it is up.

        Hand-plotted rather than a glyph, so it owes nothing to a font being
        installed; a drawn canvas is repainted on a theme switch rather than
        reconfigured, same as the app marks, which is what `self.arcs` is for.
        """
        w, h = self._px(18), self._px(13)
        c = tk.Canvas(parent, width=w, height=h, highlightthickness=0, bd=0)
        self._skin(c, bg=bg)
        self.arcs[c] = role
        self._paint_arc(c, role)
        return c

    def _paint_arc(self, c, role=None, span=1.0):
        """Draw the arc in the palette's `role`, `span` of the way across. The
        piers stay put while the deck is still being built, so a partial arc
        reads as one going up rather than as one that is broken."""
        role = self.arcs[c] = role or self.arcs.get(c, "faint")
        c.delete("all")
        w, h = int(c.cget("width")), int(c.cget("height"))
        pad, thick = self._px(1), max(1, self._px(1.6))
        base = h - self._px(2)            # the line the span lands on
        rise = self._px(6)                # shallow: a deck, not a rooftop
        colour = self.C[role]
        # The span it is going to be, behind the span it is: while the arc
        # draws itself across, the rest of the crossing is already faintly
        # there, which is what makes a partial one read as building rather
        # than as broken.
        ghost = blend(self.C[role], self.C["side"], 0.72)
        # Walked as a polyline rather than drawn as an arc item, because a
        # partial span is then simply a matter of stopping early. Shallow on
        # purpose: at 18 by 13 pixels a tall one reads as a chevron, and the
        # piers and abutments that would fix that only thicken the middle -
        # both were tried at size before this was settled on.
        steps = 28
        pts = [(pad + (w - 2 * pad) * (i / steps),
                base - rise * math.sin(math.pi * (i / steps)))
               for i in range(steps + 1)]
        c.create_line(*[xy for p in pts for xy in p], fill=ghost, width=thick,
                      smooth=True, capstyle="round")
        cut = max(2, int(round(len(pts) * min(1.0, max(0.0, span)))))
        c.create_line(*[xy for p in pts[:cut] for xy in p], fill=colour,
                      width=thick, smooth=True, capstyle="round")

    def _arc_state(self, c, role):
        """Paint the arc in `role`. A role it is not already in draws itself
        across once and settles: the row is told its state on every bridge
        event, and a span building is a better account of one coming up than
        a colour appearing fully formed. One pass, then the animation drops
        itself - there is no state this can sit in spinning, which matters
        because a bridge that never starts would otherwise animate for ever."""
        if self.arcs.get(c) == role:
            return
        self.arcs[c] = role
        start = self.anim_frame

        def draw(frame):
            step = frame - start + 1
            self._paint_arc(c, role, step / float(SWEEP_FRAMES))
            return step < SWEEP_FRAMES

        self._animate(("arc", str(c)), draw)

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
        return self._disc_colour(self.C[role], size)

    def _disc_colour(self, colour, size):
        """The same disc, by literal colour rather than by role: a pulse needs
        the shades between two roles and the palette holds only the ends."""
        key = ("disc", colour, size)
        photo = self.photos.get(key)
        if photo is None:
            data = icons.disc_png(colour, size)
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

    def _paint_app_dot(self, s):
        """The rail's dot for one app: its bridge's state, breathing while
        that tab is still starting up. `booting` and not the bridge role,
        because a tab can sit at "not started" indefinitely - the host coming
        back resets every stuck tab, and the ones you are not looking at wait
        to be selected. Those are settled, not busy, and must not animate."""
        dot = self.app_dots.get(s.id)
        if dot is not None:
            self._pulse_dot(dot, s.bridge[0], "side", s.booting)

    def _pulse_dot(self, canvas, role, bg, busy):
        """A status dot that breathes while its tab is working, and sits still
        the rest of the time. The dot already carries what the bridge is doing;
        this is the other half - whether that tab is mid-run - and it is worth
        saying on the tab itself, because the run you started is very often not
        the tab you are looking at now.

        `dot_role` still holds the settled role, so a theme switch repaints it
        correctly whether it is pulsing or not."""
        key = ("dot", str(canvas))
        if not busy:
            self._unanimate(key)
            self._set_dot(canvas, role)
            return
        self.dot_role[canvas] = role
        size = int(canvas.cget("width")) - 2
        self._animate(key, lambda frame: canvas.itemconfig(
            1, image=self._shade(role, bg, 0.3 + 0.7 * (
                0.5 + 0.5 * math.sin(frame * 0.3)), size)))

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

    def _hover(self, row, widgets, base, lit, on=None, draw=None):
        """
        Light a whole row on hover. <Leave> also fires when the pointer moves
        onto a child, so check where it actually went before unlighting.
        `on`, if given, is told whether the row is lit - for controls that
        only appear while the pointer is over the row. `draw`, if given, is
        handed the role instead of a widget: a row whose highlight is a
        rounded shape has it drawn on a canvas, and a canvas cannot be told a
        colour it plotted itself with config().

        `row` takes the bindings whether or not it is in `widgets`, so a
        drawn row's own canvas - which must keep its background - still
        notices the pointer arriving over its margin.
        """
        def paint(role):
            for w in widgets:
                try:
                    w.config(bg=self.C[role])
                except tk.TclError:
                    pass
            if draw is not None:
                draw(role)
            if on is not None:
                on(role == lit)

        def leave(ev):
            under = row.winfo_containing(ev.x_root, ev.y_root)
            while under is not None:
                if under is row:
                    return
                under = getattr(under, "master", None)
            paint(base)

        for w in ([row] + list(widgets) if row not in widgets else widgets):
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
        """
        A tab is a rounded chip drawn on its own canvas, with the mark, label,
        dot and close glyph in a frame placed on top of it - the same shape
        the composer's outline is made with, and for the same reason: a Tk
        Frame is a rectangle and nothing on it bends.

        The frame sits `PAD` inside the curve so its square corners never poke
        out of it, the canvas follows the frame's requested size, and the
        active tab's accent bar is drawn rather than packed, inset by the
        radius so it stays inside the shape.
        """
        app = self.sessions[sid].app
        PAD, R = self.tab_pad, self.tab_r
        tab = tk.Canvas(self.tabbar, highlightthickness=0, bd=0, cursor="hand2")
        self._skin(tab, bg="bg")
        tab.pack(side="left", padx=(0, 4))
        inner = self._skin(tk.Frame(tab), bg="bg")
        item = tab.create_window(PAD, PAD, window=inner, anchor="nw")
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

        def paint(_ev=None):
            w = inner.winfo_reqwidth() + 2 * PAD
            h = inner.winfo_reqheight() + 2 * PAD
            tab.config(width=w, height=h)
            tab.delete("chip")
            on = sid == self.active
            fill = self.C["card" if on else "bg"]
            rounded(tab, 0, 0, w, h, R, fill=fill, outline=fill, tags="chip")
            if on:
                tab.create_line(R, h - self._px(2), w - R, h - self._px(2),
                                fill=self.C["accent"], width=self._px(2),
                                capstyle="round", tags="chip")
            tab.tag_lower("chip")

        inner.bind("<Configure>", paint)
        self.tab_ui[sid] = {"tab": tab, "label": lbl, "dot": dot, "paint": paint,
                            "close": close, "mark": mark, "compact": False,
                            "bgs": [inner, lbl, mark, dot, close]}
        paint()
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
            # 4 for the gap the tab is packed with, 16 for the label's own
            # padding and 8 for the close glyph's - and the chip's inset,
            # which unlike those scales with the display.
            return (ui["mark"].winfo_reqwidth() + ui["dot"].winfo_reqwidth()
                    + ui["label"].winfo_reqwidth() + ui["close"].winfo_reqwidth()
                    + 2 * self.tab_pad + 28)
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
        ui["paint"]()                     # the chip is drawn, so it is repainted
        self._pulse_dot(ui["dot"], s.bridge[0], "card" if on else "bg", s.busy)

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
        if s.browser is not None:
            s.browser.release()           # out of the frame before it goes
        if s.images is not None:
            s.images.release()            # the Scene Builder goes with its form
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
        panel = sid is not None and self.sessions[sid].app.panel
        if panel:
            self.composer.pack_forget()
        elif not self.composer.winfo_ismapped():
            self.composer.pack(side="bottom", fill="x", padx=24, pady=(8, 20),
                               before=self.strip)
        for i in self.order:
            self._paint_tab(i)
        self._fit_tabs()
        self._apply_status()
        self._sync_bridges()
        if sid is not None:
            self._ensure(self.sessions[sid])
            if panel:
                self._focus_panel(self.sessions[sid])
            else:
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
        conns.pack(side="bottom", fill="x", pady=(14, 14))

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
                                            pretty_host(self.host),
                                            command=self._host_menu)}
        self.conn["bridges"] = self._conn_row(
            conns, "Bridges", "not started", arc=True,
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
        """
        One app in the rail. The row is a canvas so its hover highlight can be
        a rounded shape; everything in it lives in a frame placed on top,
        inset far enough that the frame's square corners stay inside the
        curve. The canvas keeps the rail's own background and draws the
        highlight - `_hover` is given `draw` for exactly that.
        """
        name = a["name"]
        pinned = name in self.pinned
        # The inset is what keeps the frame's square corners inside the curve,
        # and it is also height every row now costs: nine rows at four pixels
        # a side pushed the last one off the rail. Three is over the third of
        # the radius the geometry needs, and the gap it leaves between rows
        # replaces the `pady` they used to be packed with.
        PAD, R = self._px(3), self._px(9)
        shell = tk.Canvas(self.applist, highlightthickness=0, bd=0)
        self._skin(shell, bg="side")
        shell.pack(fill="x", padx=(8, 6))
        row = self._skin(tk.Frame(shell), bg="side")
        item = shell.create_window(PAD, PAD, window=row, anchor="nw")

        def paint(role="side"):
            w, h = shell.winfo_width(), row.winfo_reqheight() + 2 * PAD
            shell.config(height=h)
            shell.itemconfig(item, width=max(1, w - 2 * PAD))
            shell.delete("chip")
            fill = self.C[role]
            rounded(shell, 0, 0, w, h, R, fill=fill, outline=fill, tags="chip")
            shell.tag_lower("chip")

        shell.bind("<Configure>", lambda ev: paint(self.row_role.get(shell, "side")))
        row.bind("<Configure>", lambda ev: paint(self.row_role.get(shell, "side")))
        self.row_role[shell] = "side"

        def draw(role):
            self.row_role[shell] = role
            paint(role)

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
            shell.config(cursor="hand2")
            row.config(cursor="hand2")
            for w in (shell, row, box, title, subtitle, mark):
                w.bind("<Button-1>", lambda ev, i=a["id"]: self._add_tab(i))
        # `shell` is the row for hover purposes but keeps the rail's own
        # background: the highlight it shows is drawn, not configured.
        self._hover(shell, widgets, "side", "hover", on=reveal, draw=draw)
        for w in widgets + [shell]:
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
    def _conn_row(self, side, title, detail, arc=False, command=None):
        row = self._skin(tk.Frame(side), bg="side")
        row.pack(fill="x", padx=14, pady=3)
        if arc:
            lead = self._arc(row)
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
        self._arc_state(lead, role)
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

    def _menu_host(self):
        """The inference host, and the one thing to do about it: connect
        again. Reopening the window used to be the only way back after a
        probe that timed out; this is that, without losing the tabs."""
        m = self._menu()
        if self.host_booting:
            m.add_command(label="Connecting to the inference host...", state="disabled")
        else:
            m.add_command(label="Connect to the inference host" if self.llm is None
                          else "Connect again", command=self._connect_host)
        m.add_separator()
        m.add_command(label="Inference runs on %s" % pretty_host(self.host),
                      state="disabled")
        return m

    def _host_menu(self, widget):
        self._popup(self._menu_host(), widget)

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
        return self._button(parent, "forget", forget, kind="ghost", bg="card",
                            font=self.f_small, padx=self._px(9),
                            pady=self._px(1), r=self._px(8))

    # ---------------------------------------------------------------- composer
    def _build_composer(self, composer):
        # The input's rounded outline is drawn on a canvas, with the real
        # widgets in a frame placed on top of it. The frame sits far enough
        # inside the curve (INSET against radius R) that its square corners
        # never poke out; the canvas follows the frame's height, so the
        # outline grows with the chip strip and shrinks back with it.
        INSET, R = 7, 16
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
            ui.lifted(shell, w, h, R, self.C["card"], self.C["bg"], "frame")
            shell.tag_lower("frame")
        shell.bind("<Configure>", paint)
        inner.bind("<Configure>", paint)
        self.composer_paint = paint

        # Pictures waiting to go with the next message sit above the input,
        # one chip each; the strip is packed only while it has something in it.
        self.chips = self._skin(tk.Frame(inner), bg="card")
        # The text takes the card's width; the controls sit in a row under it,
        # attach on the left and send on the right - the layout of every
        # composer people already know, rather than buttons wedged beside the
        # text. `self.row` is the text's slot, which the chips go in above.
        self.row = self._skin(tk.Frame(inner), bg="card")
        self.row.pack(fill="x")
        self.input = tk.Text(self.row, height=2, font=self.f_body, wrap="word", bd=0,
                             padx=12, pady=10, highlightthickness=0)
        self._skin(self.input, bg="card", fg="text", insertbackground="accent",
                   selectbackground="sel")
        self.input.pack(fill="both", expand=True)
        self.input.bind("<Return>", self._on_return)
        self.input.focus_set()
        # Tk's Text has no placeholder, so it is a label over the empty text,
        # shown and hidden on `<<Modified>>` - which fires on any change,
        # typed or programmatic, once its flag is cleared each time.
        ghost = tk.Label(self.input, text="Ask %s%s" % (APP_NAME, ELLIPSIS),
                         font=self.f_body, cursor="xterm", bd=0, padx=0, pady=0)
        self._skin(ghost, bg="card", fg="faint")
        ghost.bind("<Button-1>", lambda ev: self.input.focus_set())

        def placeholder(_ev=None):
            if self.input.compare("end-1c", "==", "1.0"):
                ghost.place(x=12, y=10)
            else:
                ghost.place_forget()
            self.input.edit_modified(False)
        self.input.bind("<<Modified>>", placeholder)
        placeholder()

        bar = self._skin(tk.Frame(inner), bg="card")
        bar.pack(fill="x", pady=(0, 4))
        # right-hand side first: fixed widgets before anything that expands
        self.btn_send = self._button(bar, SEND, self._on_send, kind="accent",
                                     bg="card", font=self.f_body_b, padx=8,
                                     pady=5, round=True)
        self.btn_send.pack(side="right", padx=(8, 4))
        self._tip(self.btn_send, "Send (Enter)")
        hint = tk.Label(bar, text="Enter to send  ·  Shift+Enter for a new line",
                        font=self.f_small)
        self._skin(hint, bg="card", fg="faint")
        hint.pack(side="right")
        self.btn_attach = self._button(bar, "+  Attach", self._on_attach,
                                       kind="ghost", bg="card", padx=10, pady=4,
                                       r=self._px(10))
        self.btn_attach.pack(side="left", padx=(2, 0))
        self._tip(self.btn_attach, "Attach files (Ctrl+O)")
        self.btn_folder = self._glyph(bar, "folder", self._on_attach_folder, bg="card",
                                      tip="Attach a folder (Ctrl+Shift+O)")
        self.btn_folder.pack(side="left", padx=(4, 0))

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
        # A chip is round, like everything else that holds something: the
        # label and its close glyph sit on a canvas that draws the shape, far
        # enough inside the curve that their square corners stay in it.
        PAD, R = self._px(4), self._px(11)
        for p in self.attachments:
            shell = tk.Canvas(self.chips, highlightthickness=0, bd=0)
            self._skin(shell, bg="card")
            shell.pack(side="left", padx=(12, 0), pady=(10, 0))
            chip = self._skin(tk.Frame(shell), bg="sel")
            item = shell.create_window(PAD, PAD, window=chip, anchor="nw")
            dims = image_dims(p)
            text = os.path.basename(p) + ("  %d\u00d7%d" % dims if dims else "")
            if os.path.isdir(p):
                text = self.g["folder"] + "  " + os.path.basename(p)
            lbl = tk.Label(chip, text=clip(text, 44), font=self.f_small, padx=8, pady=3)
            self._skin(lbl, bg="sel", fg="text")
            lbl.pack(side="left")
            self._tip(lbl, p)
            self._glyph(chip, "close", lambda _w, p=p: self._drop_attachment(p),
                        bg="sel", tip="Remove").pack(side="left", padx=(0, 4))

            def paint(_ev=None, shell=shell, chip=chip):
                w = chip.winfo_reqwidth() + 2 * PAD
                h = chip.winfo_reqheight() + 2 * PAD
                shell.config(width=w, height=h)
                shell.delete("chip")
                fill = self.C["sel"]
                rounded(shell, 0, 0, w, h, R, fill=fill, outline=fill, tags="chip")
                shell.tag_lower("chip")

            chip.bind("<Configure>", paint)
            self._repaint_on_theme(shell, paint)
            paint()
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
        """An empty tab: its app's mark and name, until the first message."""
        s.hero.place(relx=0.5, rely=0.42, anchor="center")

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
            entry = self._entry(body, var)
            entry.master.pack(fill="x", pady=(3, 2))
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
        ok = self._button(row_b, "Connect" if spec is None else "Save", save,
                          kind="accent")
        ok.pack(side="right")
        cancel = self._button(row_b, "Cancel", win.destroy)
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
        # The card and the ring that marks the chosen one are both drawn, so
        # both can be round; the miniature inside sits clear of the curve.
        RING, R = self._px(4), self._px(12)
        for key, label in THEME_NAMES:
            shell = tk.Canvas(cards, highlightthickness=0, bd=0, cursor="hand2")
            self._skin(shell, bg="bg")
            shell.pack(side="left", padx=(0, 12))
            inner = tk.Frame(shell, bg=THEMES[key]["bg"], cursor="hand2")
            item = shell.create_window(RING, RING, window=inner, anchor="nw")
            # A working miniature of the window, painted in the palette it
            # selects - the honest way to show what the choice does - and
            # drawn with the same rounded shapes the window itself uses.
            c = tk.Canvas(inner, width=118, height=74, bg=THEMES[key]["bg"],
                          highlightthickness=0, bd=0)
            c.pack(padx=8, pady=(8, 4))
            p = THEMES[key]
            c.create_rectangle(0, 0, 34, 74, fill=p["side"], outline=p["side"])
            c.create_rectangle(0, 0, 118, 13, fill=p["head"], outline=p["head"])
            for y in (24, 36, 48):
                rounded(c, 6, y, 28, y + 6, 3, fill=p["card"], outline=p["card"])
            rounded(c, 44, 24, 108, 44, 5, fill=p["card"], outline=p["card"])
            rounded(c, 44, 52, 84, 62, 4, fill=p["border"], outline=p["border"])
            rounded(c, 90, 50, 110, 64, 5, fill=p["accent"], outline=p["accent"])
            tk.Label(inner, text=label, font=self.f_ui, bg=THEMES[key]["bg"],
                     fg=THEMES[key]["text"]).pack(pady=(0, 8))
            for w in (shell, inner, c) + tuple(inner.winfo_children()):
                w.bind("<Button-1>", lambda ev, k=key: self._theme(k))
            shells[key] = (shell, inner)
            # The card has no size until Tk lays the miniature out, and the
            # window sizes itself to what it can see: without this the whole
            # of Preferences came up collapsed around two one-pixel canvases.
            inner.bind("<Configure>", lambda ev: paint())

        def paint():
            for key, (shell, inner) in shells.items():
                chosen = self.prefs.get("theme") == key
                try:
                    w = inner.winfo_reqwidth() + 2 * RING
                    h = inner.winfo_reqheight() + 2 * RING
                    shell.config(width=w, height=h)
                    shell.delete("ring")
                    ring = self.C["accent"] if chosen else self.C["border"]
                    rounded(shell, 0, 0, w, h, R, fill=ring, outline=ring,
                            tags="ring")
                    shell.tag_lower("ring")
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
        btn = self._button(body, "Show every app again", lambda: None)
        btn.pack(anchor="w", pady=(8, 0))

        def refresh_count():
            n = len(self.hidden)
            count.config(text="%s hidden from the app list."
                         % ("Nothing is" if not n else
                            "%d app%s" % (n, "" if n == 1 else "s")))
            btn.set(state="disabled" if not n else "normal")

        btn.command = lambda: (self._show_all(), refresh_count())
        refresh_count()

        close = self._button(body, "Close", win.destroy, kind="accent")
        close.pack(anchor="e", pady=(22, 0))

    # -------------------------------------------------------------- view writes
    def _write(self, s, text, tag):
        if tag == "err":
            self._hide_hero(s)            # a paragraph that matters, never under it
        s.view.config(state="normal")
        s.view.insert("end", text, tag)
        s.view.config(state="disabled")
        s.view.see("end")

    def _role(self, s, name, tag):
        self._write(s, "\n%s\n" % name, tag)

    def _clear_view(self, s):
        """Empty a transcript - the folded call rows' own tags with it, and
        the rows still waiting for a result, which now has nowhere to land.

        The animated widgets embedded in it go the same way: deleting the text
        destroys them, and their animations are stood down here rather than
        left for the next frame to trip over."""
        for widget in list(s.pending.values()) + [c for _tag, c in s.stages]:
            self._unanimate(("dots", str(widget)))
            self._unanimate(("stage", str(widget)))
        if s.thinking is not None:
            self._unanimate(("dots", str(s.thinking[1])))
        s.view.config(state="normal")
        s.view.delete("1.0", "end")
        s.view.config(state="disabled")
        for tag in s.view.tag_names():
            if tag.partition(":")[0] in ("call", "mark", "status", "body", "stage"):
                s.view.tag_delete(tag)
        s.open_calls = {}
        s.pending = {}
        s.stages = []
        s.thinking = None

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

    # ------------------------------------------------------------ tool calls
    def _show_call(self, s, payload):
        """A tool call as one folded row: the tool's name to read at a glance,
        and behind a click the call as the model made it - the script, for
        the tools that take one - with its result once that arrives. The
        whole header line takes the click; the body is elided under its own
        tag until then."""
        # First, and not last: the thinking dots sit at the end of the
        # transcript, so a row written past them would be inside the range
        # `_end_thinking` deletes. `_handle` has already done this for the
        # event that got here; a result nothing announced arrives through
        # `_show_result` instead, and this is what covers that route.
        self._end_thinking(s)
        s.call_seq += 1
        n = s.call_seq
        name, args = payload["name"], payload["arguments"]
        head = name
        if isinstance(args, dict) and isinstance(args.get("action"), str):
            head += " " + args["action"]  # a compound tool: say which action
        step = bool(payload.get("via"))
        row = ("tool", "call_head", "call:%d" % n) + (("call_step",) if step else ())
        body = ("tool_body", "body:%d" % n) + (("tool_body_step",) if step else ())
        v = s.view
        v.tag_configure("body:%d" % n, elide=True)
        v.config(state="normal")
        v.insert("end", self.g["closed"], row + ("glyph", "mark:%d" % n))
        v.insert("end", " " + head, row)
        # The ellipsis this row used to wait behind was three characters that
        # never moved, and a call that takes a minute looked the same as one
        # that had hung. The dots are the same promise, kept visibly.
        v.insert("end", " ", row + ("status:%d" % n,))
        s.pending[n] = self._embed(v, self._dots(v), row + ("status:%d" % n,))
        v.insert("end", "\n", row)
        v.insert("end", self._call_text(args), body)
        v.config(state="disabled")
        v.see("end")
        s.open_calls.setdefault(name, []).append(n)
        # A tool whose name says it is about to make a picture gets the space
        # that picture will fill, right away, rather than a blank transcript
        # for however long a generation takes.
        if makes_a_picture(name):
            self._stage(s)

    def _show_result(self, s, payload):
        """The outcome of the latest call of that name still waiting for one:
        the header loses its ellipsis - or gains a word when the call failed -
        and the result joins the folded body under the arguments."""
        name, status = payload["name"], payload.get("status", "ok")
        waiting = s.open_calls.get(name)
        if not waiting:                   # a result nothing announced: its own row
            self._show_call(s, {"name": name, "arguments": {}, "via": None})
            waiting = s.open_calls[name]
        n = waiting.pop()
        v = s.view
        v.config(state="normal")
        rng = v.tag_ranges("status:%d" % n)
        if rng:
            at = v.index(rng[0])
            # Deleting the range destroys the dots embedded in it; stop drawing
            # them first rather than leave the next frame to find out.
            dots = s.pending.pop(n, None)
            if dots is not None:
                self._unanimate(("dots", str(dots)))
            v.delete(rng[0], rng[1])
            word = {"error": " failed", "skipped": " not run"}.get(status, "")
            if word:
                v.insert(at, word, ("tool", "call_head", "call:%d" % n, "call_failed"))
        rng = v.tag_ranges("body:%d" % n)
        if rng:
            body = ("tool_body", "body:%d" % n)
            if "tool_body_step" in v.tag_names(rng[0]):
                body += ("tool_body_step",)
            v.insert(rng[-1], "\n" + self._field("result", payload["text"]) + "\n", body)
        v.config(state="disabled")
        # Whatever this call was going to produce, it has produced it. A stage
        # still standing was a guess that did not pay off, or a generation that
        # failed; either way it stops waiting for a picture.
        self._clear_stages(s)
        if s.busy:
            # The model has the floor again. Between a result and whatever it
            # does next is the longest silence in a run, and the one that used
            # to look most like nothing happening.
            self._begin_thinking(s)

    # ------------------------------------------- what is happening, in the transcript
    def _round_off(self, c, w, h, r):
        """Round the corners of whatever is already drawn on `c` by covering
        each one in the background it sits on. A PhotoImage has no alpha to
        mask with and Tk will not clip an item to a shape, so the corners are
        painted out instead - four arcs' worth of background, over the picture
        rather than under it, which at these radii is indistinguishable from
        the picture having been rounded."""
        c.delete("mask")
        bg, steps = self.C["bg"], 10
        for cx, cy, sx, sy in ((0, 0, 1, 1), (w, 0, -1, 1),
                               (w, h, -1, -1), (0, h, 1, -1)):
            # The arc's centre is one radius inward on both axes; the quarter
            # it covers runs between the two points where it meets the edges.
            ox, oy = cx + sx * r, cy + sy * r
            ax, ay, bx, by = 0.0, -sy * r, -sx * r, 0.0
            pts = [cx, cy]
            for i in range(steps + 1):
                t = (i / steps) * (math.pi / 2)
                pts += [ox + ax * math.cos(t) + bx * math.sin(t),
                        oy + ay * math.cos(t) + by * math.sin(t)]
            c.create_polygon(pts, fill=bg, outline=bg, tags="mask")

    def _embed(self, v, widget, tags=(), at=None):
        """Put a widget into the transcript - at `at`, or at the end - carrying
        `tags`. An embedded window takes one character's place, so it is tagged
        by that character's range rather than at insert time the way text is."""
        at = v.index("end-1c") if at is None else v.index(at)
        v.window_create(at, window=widget, align="center", padx=self._px(2))
        for tag in tags:
            v.tag_add(tag, at, "%s+1c" % at)
        return widget

    def _begin_thinking(self, s):
        """Dots at the end of the transcript while the model has the floor and
        has not said anything yet. The first token of a reply can be a long way
        off - a cold prefix is about a minute - and until it arrives the only
        sign of life was a word in the header at the other end of the window.

        Idempotent: every path that could start one calls this, and a run makes
        several in a row."""
        if s.thinking is not None or s.closed:
            return
        v = s.view
        v.config(state="normal")
        v.mark_set("thinking:%d" % s.call_seq, "end-1c")
        v.mark_gravity("thinking:%d" % s.call_seq, "left")
        s.thinking = ("thinking:%d" % s.call_seq,
                      self._embed(v, self._dots(v, role="accent"), ("sys",)))
        v.config(state="disabled")
        v.see("end")

    def _end_thinking(self, s):
        """Take the dots away again, from the mark they were laid down at.
        Anything written after them would sit below them, so every kind of
        event that writes to a transcript ends the thinking first."""
        if s.thinking is None:
            return
        mark, dots = s.thinking
        s.thinking = None
        self._unanimate(("dots", str(dots)))
        v = s.view
        try:
            v.config(state="normal")
            v.delete(mark, "end-1c")
            v.mark_unset(mark)
            v.config(state="disabled")
        except tk.TclError:
            pass                          # the transcript was cleared under it

    def _stage(self, s):
        """The space a picture is about to fill, held open and lit while it is
        made. A generation is the longest thing this app waits for and the one
        with the most to show for it, so the wait happens where the result will
        be rather than in a status line."""
        v = s.view
        w, h = self._px(320), self._px(180)
        c = tk.Canvas(v, width=w, height=h, highlightthickness=0, bd=0)
        self._skin(c, bg="bg")
        # A grid of dots with one in the middle that pulses, each pulse rippling
        # out through the rest - after Motion's staggered grid, where every
        # cell starts a beat later the further it is from the origin. It reads
        # as "being made" rather than as "broken", and unlike a progress bar it
        # claims no fraction nobody here knows. Odd counts on both axes, so
        # there is a dot at the exact centre to start from; the strip under
        # the grid is the caption's.
        cols, rows, pitch = 13, 5, self._px(22)
        top = self._px(16)
        ox, oy = (w - (cols - 1) * pitch) / 2.0, top + pitch / 2.0
        mid_c, mid_r = cols // 2, rows // 2
        dots = [(ox + i * pitch, oy + j * pitch,
                 math.hypot(i - mid_c, j - mid_r))
                for j in range(rows) for i in range(cols)]
        small, big = self._px(4), self._px(10)

        def swell(frame, d):
            """0 at rest, 1 at the crest: how far the ripple has this dot
            raised, `d` grid steps from the centre. A raised cosine either
            side of the ripple's front, so a dot eases in and back out. The
            front starts a ripple's width short of the centre so the centre
            dot swells into its pulse rather than popping."""
            front = (frame % RIPPLE_FRAMES) * RIPPLE_SPEED - RIPPLE_WIDTH
            off = abs(d - front) / RIPPLE_WIDTH
            return 0.0 if off >= 1 else 0.5 + 0.5 * math.cos(math.pi * off)

        def draw(frame):
            c.delete("all")
            rounded(c, 1, 1, w - 1, h - 1, self._px(10), fill=self.C["card"],
                    outline=self.C["card"])
            # A dot at rest is a step up from the card - `border` alone was
            # invisible against it, the way the old sheen once was - and one
            # at the crest is lit toward the text. The centre is the one
            # coloured dot: the origin, as in Motion's example. Quantised to
            # SHADES steps, which is what keeps the disc cache to a few dozen
            # images however long a generation runs.
            rest = blend(self.C["card"], self.C["faint"], 0.35)
            lit = blend(self.C["card"], self.C["text"], 0.55)
            for x, y, d in dots:
                k = round(swell(frame, d) * (SHADES - 1)) / float(SHADES - 1)
                if d == 0:
                    size = int(round(small + 2 + (big + 2 - small - 2) * k))
                    colour = blend(self.C["accent_dk"], self.C["accent"], k)
                else:
                    size = int(round(small + (big - small) * k))
                    colour = blend(rest, lit, k)
                c.create_image(x, y, image=self._disc_colour(colour, size))
            f, caption = self.f_small, "making a picture"
            tw, gap = f.measure(caption), f.measure(" ")
            x = (w - tw - gap - ui.jump_width(f)) / 2.0
            y = (oy + (rows - 1) * pitch + h) / 2.0
            c.create_text(x, y, font=f, fill=self.C["faint"], text=caption,
                          anchor="w")
            ui.draw_jumps(c, x + tw + gap, y - f.metrics("linespace") / 2.0
                          + f.metrics("ascent"), ui.jumps(frame),
                          self._disc_colour(self.C["faint"], ui.jump_metrics(f)[0]),
                          f)

        # Tagged, not marked: the whole placeholder - its blank lines with it -
        # has to come out in one piece when the picture lands in its place, and
        # a tag range is how every other removable run in this transcript is
        # found again.
        s.stage_seq += 1
        tag = "stage:%d" % s.stage_seq
        v.config(state="normal")
        at = v.index("end-1c")
        v.insert("end", "\n", "sys")
        self._embed(v, c, ("sys",))
        v.insert("end", "\n", "sys")
        v.tag_add(tag, at, "end-1c")
        v.config(state="disabled")
        v.see("end")
        self._animate(("stage", str(c)), draw)
        s.stages.append((tag, c))
        return c

    def _take_stage(self, s):
        """Take the oldest placeholder out of the transcript and say where it
        was, so a picture can land exactly where its space was being held.
        None when nothing was expecting one - a bridge can hand back an image
        for a call whose name gave no hint."""
        if not s.stages:
            return None
        tag, c = s.stages.pop(0)
        self._unanimate(("stage", str(c)))
        rng = s.view.tag_ranges(tag)
        if not rng:
            return None                   # the transcript was cleared under it
        at = s.view.index(rng[0])
        s.view.config(state="normal")
        s.view.delete(rng[0], rng[-1])
        s.view.tag_delete(tag)
        s.view.config(state="disabled")
        return at

    def _clear_stages(self, s):
        """Take down every placeholder still standing. One a picture arrived
        for is already gone; these are the ones nothing came for - a guess from
        the tool's name that did not pay off, or a generation that failed."""
        while s.stages:
            self._take_stage(s)

    def _call_text(self, args):
        """The arguments as the model made them, one per line - a value with
        lines of its own, a script, printed whole under its name."""
        if not isinstance(args, dict):    # not an object: shown as it was written
            return self._field("arguments", str(args)) + "\n"
        if not args:
            return "no arguments\n"
        return "\n".join(self._field(k, v) for k, v in args.items()) + "\n"

    def _field(self, key, value):
        """`key: value` on one line while it fits; a longer one, or one with
        lines of its own, under the key. JSON - an object argument, or the
        result most bridges return - is laid out once it is too long for a line."""
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                value = parsed if isinstance(parsed, (dict, list)) else value
            except ValueError:
                pass
        if not isinstance(value, str):
            value = json.dumps(value, ensure_ascii=False)
            if len(value) > 72:
                value = json.dumps(json.loads(value), ensure_ascii=False, indent=2)
        value = value.rstrip()
        if len(value) > CALL_TEXT_LIMIT:
            value = (value[:CALL_TEXT_LIMIT].rstrip()
                     + "\n\u2026 %d more characters; the whole text is in the task record."
                     % (len(value) - CALL_TEXT_LIMIT))
        if "\n" in value or len(value) > 72:
            return "%s:\n%s" % (key, value)
        return "%s: %s" % (key, value)

    def _on_call_click(self, ev):
        v = ev.widget
        for tag in v.tag_names("@%d,%d" % (ev.x, ev.y)):
            if tag.startswith("call:"):
                self._toggle_call(v, int(tag[5:]))
                break
        return "break"                    # not a place to start a selection

    def _toggle_call(self, v, n):
        """Open a folded call row, or close it: flip the body's elide and turn
        the chevron to match."""
        body = "body:%d" % n
        hidden = str(v.tag_cget(body, "elide")) in ("1", "true")
        v.tag_configure(body, elide=not hidden)
        rng = v.tag_ranges("mark:%d" % n)
        if rng:
            at = v.index(rng[0])
            tags = v.tag_names(rng[0])
            v.config(state="normal")
            v.delete(rng[0], rng[1])
            v.insert(at, self.g["open" if hidden else "closed"], tags)
            v.config(state="disabled")
        if hidden and rng:
            # Opened near the fold, the body would land below it. Bring as
            # much of it up as fits, the header staying in view.
            shown = v.tag_ranges(body)
            if shown:
                v.see(shown[-1])
            v.see(at)

    def _show_status(self, text, role, lift):
        """The header's status, with its jumping dots beside it while `lift`
        says it is still going on, and without them once it is not."""
        self.lbl_status.config(text=text, fg=self.C[role])
        c = self.status_jump
        if lift is None:
            c.pack_forget()
            return
        if not c.winfo_ismapped():
            c.pack(side="left", after=self.lbl_status, padx=(3, 0))
        f = self.f_title
        h = self.lbl_status.winfo_reqheight()
        c.config(height=h)
        c.delete("all")
        ui.draw_jumps(c, 1, (h - f.metrics("linespace")) / 2.0 + f.metrics("ascent"),
                      lift, self._disc_colour(self.C[role], ui.jump_metrics(f)[0]), f)

    def cur(self):
        """The session the user is looking at, or None with every tab closed."""
        return self.sessions.get(self.active)

    def _apply_status(self):
        """The header always describes the tab you are looking at."""
        s = self.cur()
        if s is None:
            self._ellipsis("status", "no app open",
                           lambda t, lift: self._show_status(t, "muted", lift))
            self._show_fix(False)
            self._ellipsis("send", SEND,
                           lambda t, lift: self.btn_send.set(text=t, lift=lift))
            self.btn_send.set(state="disabled")
            self.btn_new.set(state="disabled")
            self.btn_hist.set(state="disabled")
            return
        text, role, fixable = s.status
        # A status ending in an ellipsis is one still happening, and the dots
        # jump so the window never sits looking stalled while it works. See
        # `_ellipsis`; the marker is written at the point the status is made.
        self._ellipsis("status", text,
                       lambda t, lift, r=role: self._show_status(t, r, lift))
        if s.host_down:
            self.btn_fix.set(text="Connect")
        else:
            self.btn_fix.set(text=("Check %s" if s.app.remote else "Start %s") % s.app.name)
        self._show_fix(fixable and not s.app.panel)
        self.btn_new.set(state="disabled" if s.app.panel else "normal")
        self.btn_hist.set(state="disabled" if s.busy or s.app.panel else "normal")
        stopping = s.busy and s.cancel.is_set()
        self._ellipsis("send",
                       "Stopping" + ELLIPSIS if stopping else "Stop" if s.busy else SEND,
                       lambda t, lift: self.btn_send.set(text=t, lift=lift))
        self.btn_send.set(state="disabled" if stopping else "normal")
        # A tab still working says so on its own tab, not only in the header:
        # the run you started is often not the tab you are looking at.
        if s.id in self.tab_ui:
            self._paint_tab(s.id)

    def _host_probed(self, ok):
        """A probe is over. Answered, every tab that was stuck without the
        host is unstuck - the active one starts now, a ready one is ready
        again, the rest start when selected. Not, they keep their Connect,
        and the window tries again by itself: the LLM PC drops off on a
        timer, and nobody should have to press a button every time it
        comes back."""
        # The probe is over, whatever it found: the row's dot settles. This
        # runs on the UI thread after the worker has cleared `host_booting`,
        # which the row's own event may well have been posted before.
        dot, _lbl = self.conn["host"]
        self._pulse_dot(dot, self.dot_role.get(dot, "faint"), "side", False)
        stuck = [t for t in self.sessions.values() if t.host_down]
        for t in stuck:
            if not ok:
                t.status = ("no inference host - trying again", "err", True)
                continue
            t.host_down = False
            if t.ready:
                t.status = ("ready", "muted", False)
            else:
                t.status = ("not started", "muted", False)
                self._handle("bridge", t.id, ("faint", "%s\nnot started" % t.app.bridge_label))
        self._apply_status()
        s = self.cur()
        if ok and s is not None and s in stuck:
            self._ensure(s)
        if not ok:
            self._arm_retry()

    def _host_wanted(self):
        """Is anything waiting on the host: no model yet, or a tab stuck."""
        return self.llm is None or any(t.host_down for t in self.sessions.values())

    def _arm_retry(self):
        """One quiet probe, HOST_RETRY_MS from now, unless one is pending."""
        if self.host_timer is None and not self.closing:
            self.host_timer = self.after(HOST_RETRY_MS, self._retry_host)

    def _retry_host(self):
        self.host_timer = None
        if self.host_booting or not self._host_wanted():
            return                        # Connect is at it, or nothing needs it
        self._spawn(None, self._boot_host, False, True)

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
        """The one pump carrying the worker threads' events to the UI, and it
        must never stop. `_handle` touches widgets, images and transcripts;
        anything it raised used to escape past the reschedule below, so the
        pump died - permanently, and in silence. Every tab went quiet at once
        (no tokens, no status, no idle, the Stop button stuck on) while the
        threads kept filling a queue nobody read, and the traceback went to a
        stderr `pythonw.exe` does not have. One bad event is now logged and
        dropped; the next tick runs either way."""
        handled = False
        try:
            while True:
                try:
                    event = self.q.get_nowait()
                except queue.Empty:
                    break
                handled = True
                try:
                    self._handle(*event)
                except Exception:
                    self._report("handling a %s event" % (event[0],),
                                 traceback.format_exc())
        finally:
            # Not while the window is going away: `after` on a destroyed widget
            # raises, and a tick already in flight would outlive `destroy`.
            # Stand the old tick down first, so the pump is a single timer
            # however it is entered - a hand-called `_drain` used to arm one
            # beside the live one, and every extra was a callback left pending
            # against a window that would be destroyed under it.
            # Fast while anything is moving; an idle window polls at a rate
            # nobody can see but the CPU can - a quarter of the wake-ups.
            if not self.closing:
                self._stand_down("drain_timer")
                busy = handled or any(t.busy for t in self.sessions.values())
                self.drain_timer = self.after(DRAIN_MS if busy else DRAIN_IDLE_MS,
                                              self._drain)

    def _stand_down(self, timer):
        """Cancel a pending `after` by attribute name and forget it. An id that
        has already fired is not an error worth having."""
        handle = getattr(self, timer, None)
        if handle is not None:
            try:
                self.after_cancel(handle)
            except Exception:
                pass
            setattr(self, timer, None)

    # ----------------------------------------------------------------- animation
    def _animate(self, key, draw):
        """Register `draw(frame)` to be called every frame under `key`, and
        paint it once now so nothing waits a tick to appear. Registering the
        same key again replaces the old callback, which is what makes this
        safe to call from `_apply_status` and friends - they run on every
        event, and an animation is a property of the state, not of the event
        that announced it. A `draw` that returns False is dropped.

        One timer for all of them, armed only while `self.anim` has something
        in it: the same reasoning as the queue pump, plus the part the pump
        does not need - a window sitting idle must not wake up 14 times a
        second to redraw nothing."""
        self.anim[key] = draw
        # Guarded exactly as a frame from the tick is, and for a sharper
        # reason: `_animate` is called from `_apply_status` and friends, which
        # run on every event, so an animation that raised on its first paint
        # would take the event with it.
        if self._draw_once(key, draw):
            self._arm_anim()

    def _unanimate(self, key):
        """Stop an animation. Unknown keys are not an error - callers stop
        things that may never have started."""
        self.anim.pop(key, None)

    def _arm_anim(self):
        if self.anim and self.anim_timer is None and not self.closing:
            self.anim_timer = self.after(ANIM_MS, self._anim_tick)

    def _draw_once(self, key, draw):
        """One frame of one animation, and nothing it does may escape. True
        while the animation is still wanted; False once it has been dropped,
        whether because it said it was finished, because the widget under it
        went away, or because it was broken."""
        try:
            if draw(self.anim_frame) is not False:
                return True
        except tk.TclError:
            pass                          # the widget was destroyed under it
        except Exception:
            self.anim.pop(key, None)
            self._report("drawing the %s animation" % (key,),
                         traceback.format_exc())
            return False
        self.anim.pop(key, None)
        return False

    def _anim_tick(self):
        """One frame. This is a timer, and a timer that raises is a timer that
        does not re-arm - the same way the pump used to die.

        The old tick is stood down before the next is armed, so the animator
        is a single timer however it is entered. A hand-called `_anim_tick`
        - the tests are full of them - used to arm one beside the live one,
        and Tk named every orphan on the way out."""
        self._stand_down("anim_timer")
        self.anim_frame += 1
        for key, draw in list(self.anim.items()):
            self._draw_once(key, draw)
        self._arm_anim()

    def _ellipsis(self, key, text, show):
        """Call `show(text, lift)`, and if the text ends in an ellipsis keep
        calling it every frame: the stem without the ellipsis, and `lift` the
        heights of three jumping dots to draw after it (`ui.jumps`). Text that
        does not end in one is shown once with `lift` None.

        The trailing ellipsis is the whole protocol. A status, a button label
        or a caption that ends in one is one describing something still
        happening, so the author writes `"working" + ELLIPSIS` at the point
        where they know that and nothing else has to be told. Anything not
        ending in one is shown once and its animation dropped, which is how a
        finished state stops moving without a second call."""
        if not text.endswith(ELLIPSIS):
            self._unanimate(key)
            show(text, None)
            return
        stem = text[:-1]
        self._animate(key, lambda frame: show(stem, ui.jumps(frame)))

    def report_callback_exception(self, exc, val, tb):
        """Tk's own hook for a callback that raised - a menu command, a button,
        a binding. The default prints to stderr, and the shortcut starts the
        app with `pythonw.exe`, which has none, so a broken control was simply
        dead and said nothing. Route it where `_drain` sends its own."""
        self._report("handling a click",
                     "".join(traceback.format_exception(exc, val, tb)))

    def _report(self, doing, trace):
        """A failure on the UI side. Nothing in here may raise: this is the
        path that keeps the pump alive, and it is often reached because some
        widget is already unhappy. The log is the record; the transcript line
        is so the user learns from the app rather than from its silence."""
        try:
            self._log("While %s:\n%s" % (doing, trace))
        except Exception:
            pass
        try:
            s = self.cur()
            if s is not None:
                self._write(s, "Something went wrong %s: %s\nThe app is still "
                               "running and the details are in %s\n"
                            % (doing, trace.strip().rsplit("\n", 1)[-1],
                               error_log_path()), "err")
        except Exception:
            pass

    def _read_icons(self):
        """
        Off the UI thread: an app's icon lives inside its .exe, and a cold read
        of a 500MB Photoshop binary is not something to do while the window is
        trying to open. Badges are drawn until these land.
        """
        row, tab, menu, hero = (self.marks_px["row"], self.marks_px["tab"],
                                self.marks_px["menu"], self.marks_px["hero"])
        jobs = [(a["id"] or a["name"], a["exe"], row) for a in self.detected]
        for app in eng.APPS:
            exe = app.exe()
            jobs += [(app.id, exe, tab), (app.id, exe, menu), (app.id, exe, hero)]
        for key, exe, size in jobs:
            # No .exe (ComfyUI is on the LLM PC): its mark is drawn instead.
            data = icons.icon_png(exe, size) or icons.drawn_png(key, size)
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

        if kind == "update":              # the window's, like "icon" below
            self._show_update(*payload)
            return
        if kind == "updated":
            self._updated(*payload)
            return

        if kind == "diagnostics":
            # Like "icon", this belongs to the window rather than to a tab:
            # the report is about the whole installation, and it must still
            # arrive when every tab is closed.
            self._paint_diagnostics(payload)
            return

        if s is not None and s.app.panel and kind not in (
                "host", "host_probed", "host_retry"):
            self._panel_event(s, kind, payload)
            return

        if s is None:                     # every tab is closed
            if kind in ("error", "sys"):
                self.empty_msg.config(text=payload)
            elif kind == "trace":
                self._log(payload)
            return

        # Anything about to write into a transcript takes the thinking dots
        # down first: they sit at the end, and whatever is written next would
        # otherwise land underneath them. `_show_call` and `_show_result` put
        # them back when the model still has work to do.
        if kind in WRITES_TO_TRANSCRIPT:
            self._end_thinking(s)

        if kind == "status":
            s.status = payload
            if s.id == self.active:
                self._apply_status()
        elif kind == "host":
            role, detail = payload
            dot, lbl = self.conn["host"]
            # Pulsing while a probe is out: the LLM PC drops off on a timer and
            # the window keeps trying by itself, which is worth being able to
            # see rather than having to infer from a row that never changes.
            self._pulse_dot(dot, role, "side", self.host_booting)
            if detail:
                lbl.config(text=detail)
        elif kind == "host_probed":
            self._host_probed(payload)
        elif kind == "host_retry":
            self._arm_retry()
        elif kind == "bridge":
            s.bridge = payload
            role, detail = payload
            self._paint_app_dot(s)
            self._paint_tab(s.id)
            self._sync_bridges()
        elif kind == "sys":
            self._write(s, payload + "\n", "sys")
        elif kind == "tool":
            self._show_call(s, payload)
        elif kind == "tool_result":
            self._show_result(s, payload)
        elif kind == "preview":
            self._show_preview(s, payload)
        elif kind == "ask":
            self._show_ask(s, payload)
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
            self._end_thinking(s)
            self._clear_stages(s)
            self._paint_app_dot(s)        # `booting` is cleared just before this

            if s.id == self.active:
                self._apply_status()
            elif s.id in self.tab_ui:
                self._paint_tab(s.id)     # its dot stops pulsing even unwatched

    def _log(self, text):
        # One writer, so the rollover is in one place. A failure to write costs
        # the entry, not the app - the transcript already has it as prose.
        log_error(text)

    # --------------------------------------------------------------- preflight
    def _boot_host(self, first=True, quiet=False):
        """Shared across every tab: one inference host, one model.

        Run at startup, again by Connect, and every HOST_RETRY_MS by the
        window itself while the host is down. The probe is one HTTP call
        with a short timeout, and a PC still waking or a server not yet
        started fails it once; that used to cost the window, because nothing
        probed again. Now `host_ready` says the first probe is over,
        `host_booting` keeps two probes from running at once, and
        `host_probed` tells the UI thread to give the tabs stuck without a
        host their start - or to schedule the next try. A `quiet` probe is
        one of those tries: it moves the row and the status, and writes
        nothing into the transcript unless it succeeds.
        """
        self.host_booting = True
        # Only the row's dot changes here: the detail is left alone, because a
        # probe that fails should not blank out what the last good one said.
        self.q.put(("host", None, (self.host_role, None)))
        ok = False
        try:
            ok = self._probe_host(first, quiet)
        finally:
            self.host_booting = False
            self.host_ready.set()
            self.q.put(("host_probed", None, ok))
        # The vision model is not loaded here: it goes on after a tab's own
        # model (`_boot_session`, `_load_vision`), never before it.

    def _probe_host(self, first, quiet=False):
        """-> True with `self.llm` set. A probe that fails after an earlier
        one succeeded leaves the model in place: the tabs that have it keep
        working the moment the host is back, without another Connect."""
        if first:
            self.q.put(("status", None, ("checking the inference host" + ELLIPSIS, "muted", False)))
        ok, loaded, ids, vision_ids, err = eng.probe_models(self.host)
        if not ok:
            self.host_role, self.host_last = "err", "unreachable"
            self.q.put(("host", None, ("err", "%s\nunreachable" % pretty_host(self.host))))
            if not quiet:
                self.q.put(("error", None, self._explain_unreachable(err)))
            return False
        model = eng.pick_model(loaded, ids, self.want_model)
        if not model:
            self.host_role, self.host_last = "err", "no models"
            self.q.put(("host", None, ("err", "%s\nno models" % pretty_host(self.host))))
            if not quiet:
                self.q.put(("error", None, "The inference host is up but serving no models. "
                                           "Load one in LM Studio; this window tries again "
                                           "every %d seconds, and Connect tries now."
                                           % (HOST_RETRY_MS // 1000)))
            return False
        # Speculative decoding: a small model of the same family runs ahead of
        # this one when the host serves it. Nothing to load - LM Studio takes
        # it per request - and a pair the host refuses is dropped by the client.
        # The same model on a second probe keeps its client: the tabs booted
        # on it hold that object, and its draft score is the row's.
        draft, self.draft_note = eng.resolve_draft(model, ids)
        if self.llm is None or self.llm.model != model:
            self.llm = eng.LLM(self.host, model, draft=draft)
        self.model_ids = ids
        # Every bridge answers a screenshot with a picture and every tab takes
        # attachments; the executing model reads text. One vision model on the
        # same host serves every tab, and its absence is a status, not a silence.
        self.vision, self.vision_note = eng.resolve_vision(self.host, model, vision_ids, loaded)
        if not first:
            self.q.put(("sys", None, "Connected to the inference host at %s: %s, %d model%s served."
                        % (pretty_host(self.host), model, len(ids), "" if len(ids) == 1 else "s")))
        if not self.vision:
            self._host_healthy("warn", "no vision model")
        else:
            # Loaded or not: it goes on after the first tab's model, and a
            # picture before then is loaded just in time by the host.
            self._host_healthy("ok", "sees: " + clip(self.vision.model, 18))
        return True

    def _explain_unreachable(self, err):
        """Which half is down. The probe alone cannot say: Windows Firewall
        drops a port with no listener rather than refusing it, so a PC that
        is asleep and a PC that is up with LM Studio's server not running
        both read as "timed out". The tailnet can say, so it is asked."""
        retry = ("This window tries again every %d seconds; Connect - the button, "
                 "or the Inference row - tries now." % (HOST_RETRY_MS // 1000))
        alive = eng.host_alive(self.host)
        if alive is True:
            what = ("The LLM PC is up, but LM Studio's server is not answering at %s.\n"
                    "Start the server in LM Studio - and to have it back after a restart "
                    "with nobody signed in, turn on its service on login (Settings > "
                    "Developer). A locked screen does not stop it; sleep does."
                    % pretty_host(self.host))
        elif alive is False:
            what = ("The LLM PC is not answering at all at %s - asleep, off, or off "
                    "the tailnet.\nWake it, and if it sleeps on a timer, turn that off "
                    "there (powercfg /change standby-timeout-ac 0); locking is fine."
                    % pretty_host(self.host))
        else:
            what = ("Cannot reach the inference host at %s.\n"
                    "Check that the other PC is awake, Tailscale is up on both ends, "
                    "and LM Studio's server is started." % self.host)
        return "%s\n%s\n(%s)" % (what, retry, err)

    def _load_vision(self):
        """The vision model onto the host, after a tab's own model has gone on
        - never before it. LM Studio gives the card to the model it loads
        first and the one it loads beside it partly system memory for as
        long as it stays loaded: loaded at the probe, ahead of every tab,
        the vision model left the 30B decoding at 26 tokens a second where
        it does 79. Off the path a tab waits on; one tab asks, the rest find
        it done."""
        vision = self.vision
        # Under the fit lock, like every load: a tab fitting its model now
        # would find the card half taken, and two tabs asking at once would
        # load it twice.
        with self.fit_lock:
            if vision is None or not vision.needs_load:
                return
            vision.needs_load = False
            self._host_healthy("muted", "loading " + clip(vision.model, 16))
            err = eng.load_model(self.host, vision.model)
        if err:
            self.q.put(("sys", None, "Could not load the vision model %s on the host (%s); "
                                     "it will be loaded on first use instead."
                                     % (vision.model, err)))
        self._host_healthy("ok", "sees: " + clip(vision.model, 18))

    def _host_healthy(self, role, last):
        """The Inference row with the host answering - kept, so that a loss
        mid-conversation can be undone by the next request that gets through
        rather than by another probe."""
        self.host_ok = (role, last)
        self._host_line(role, last)

    def _host_lost(self):
        """The Inference row, after a request found nothing at the host."""
        self._host_line("err", "unreachable")

    def _host_back(self):
        """...and after the next one that got through."""
        if self.host_role == "err" and self.host_ok:
            self._host_line(*self.host_ok)

    def _connect_host(self):
        """Connect: the header button while a tab is stuck without the host,
        the Inference row, and the Bridges menu. Probes again and, when the
        host answers, starts the tab you are looking at; the others start
        when selected, as they always have. Reopening the window used to be
        the only way to get here."""
        if self.host_booting:
            return
        for t in self.sessions.values():
            if t.host_down:
                t.status = ("connecting to the inference host" + ELLIPSIS, "muted", False)
        self._apply_status()
        self._spawn(None, self._boot_host, False)

    def _host_line(self, role=None, last=None):
        """The Inference row: host, how many models, the shared model, what
        sees for it, and the draft model running ahead of it - with the share
        of its guesses the model kept, once the host has reported any. Posted
        from the worker at boot and again after a turn, so the score is live.
        """
        if role is not None:
            self.host_role = role
        if last is not None:
            self.host_last = last
        if self.llm is None:
            return
        lines = [pretty_host(self.host), "%d models" % len(self.model_ids),
                 clip(self.llm.model, 24), self.host_last]
        draft = getattr(self.llm, "draft", None)
        if draft:
            kept, offered = getattr(self.llm, "drafted", (0, 0))
            lines.append("draft: %s · %d%%" % (clip(draft, 11), 100 * kept // offered)
                         if offered else "draft: " + clip(draft, 17))
        self.q.put(("host", None, (self.host_role, "\n".join(lines))))

    def _fit(self, s, prompt_tokens, exact=True):
        """Load, or reload, this tab's model on the host with a window that
        holds its prefix and a conversation. -> (reloaded, note).

        LM Studio loads a model at its default, 8,192 for most, and every tab's
        briefing and tools take most or all of that: the model then loses its
        own tool results to truncation and asks again, or is cut off. The host
        can be told the window at load time, so the app tells it - the same
        call that loads the vision model. A reload while another tab is
        mid-request would cut that request off, so then this tab gets the
        advice instead and is fitted on its next boot. Two tabs booting at
        once take turns: the second looks again once the first has loaded,
        and finds nothing to do - a load beside a load is a second instance."""
        if any(o.busy for o in self.sessions.values() if o is not s):
            loaded, top = eng.context_window(self.host, s.llm.model)
            s.window = loaded
            return False, eng.headroom_note(s.llm.model, prompt_tokens, loaded, top)
        with self.fit_lock:
            before, _ = eng.context_window(self.host, s.llm.model)
            # A load goes onto an empty card (`fit_model`'s `keep`): the model
            # loaded first gets the GPU, and the one beside it system memory.
            now, note = eng.fit_model(self.host, s.llm.model, prompt_tokens, exact=exact,
                                      keep=())
            reloaded = before != now and not note.startswith("Could not")
            vision = self.vision
            if reloaded and vision is not None and vision.model != s.llm.model:
                # It went with the rest; `_load_vision` puts it back after.
                vision.needs_load = not eng.loaded_instances(self.host, vision.model)
        s.window = now
        return reloaded, note

    def _make_room(self, s):
        """Before a render: every model off the shared GPU - this tab's own
        too, which only waits while the picture is made - so the render runs
        in VRAM rather than streaming from system RAM. A model another tab is
        mid-request on stays. Says what it did once a tab. -> the context
        length to reload this tab's model with afterwards."""
        own = getattr(s.llm or self.llm, "model", None)
        keep = {getattr(o.llm or self.llm, "model", None)
                for o in self.sessions.values() if o is not s and o.busy}
        with self.fit_lock:
            ctx = next((c for _, c in eng.loaded_instances(self.host, own)), None)
            gone, err = eng.make_room(self.host, keep)
        if gone and not getattr(s, "room_said", False):
            s.room_said = True
            self.q.put(("sys", s.event_id,
                        "Unloaded %s from the LLM PC while %s renders, so the picture gets "
                        "the whole GPU; this tab's model comes back after each render, other "
                        "tabs reload theirs when next used." % (", ".join(gone), s.app.name)))
        elif err:
            self.q.put(("sys", s.event_id, "Could not free the GPU for %s (%s); this "
                                           "render may be slow." % (s.app.name, err)))
        return ctx or s.window

    def _give_back(self, s, ctx):
        """After a render: this tab's model back, at the window it had. Not
        straight away - `eng.YieldGPU` calls this when the model is next
        needed, once the picture is on screen and the vision model has
        looked at it on a GPU with nothing else loading. The vision model
        goes first: LM Studio loads a model into an empty card in 3 s and
        beside another in 9 to 16, whichever of the two comes second, so
        the check, the unload and the reload take 11 s in that order and
        20 s with both resident."""
        own = getattr(s.llm or self.llm, "model", None)
        busy = [o for o in self.sessions.values() if o is not s and o.busy]
        keep = {getattr(o.llm or self.llm, "model", None) for o in busy} | {own}
        if busy and self.vision is not None:
            keep.add(self.vision.model)   # another tab may be looking through it
        with self.fit_lock:
            eng.make_room(self.host, keep)
            err = eng.give_back(self.host, own, ctx)
        if err:
            self.q.put(("sys", s.event_id, "Could not reload %s after the render (%s)."
                        % (own, err)))

    def _reload_if_unloaded(self, s):
        """A tab whose model was unloaded - by another tab making room, or by
        hand in LM Studio - gets it back at its own window before its turn.
        Left to the host, the next request would load it just in time at the
        8,192 default and truncate this tab's briefing."""
        llm = s.llm or self.llm
        if llm is None or s.window is None:
            return
        loaded, top = eng.context_window(self.host, llm.model)
        if loaded is not None or top is None:     # loaded, or a host that does not say
            return
        self.q.put(("status", s.event_id, ("loading %s on the LLM PC%s"
                                           % (llm.model, ELLIPSIS), "muted", True)))
        prefix = s.prefix_tokens or eng.estimate_tokens(s.messages[0]["content"], s.tools)
        self._fit(s, prefix, exact=bool(s.prefix_tokens))

    def _headroom(self, s, reply):
        """After a warm-up: the prefix's exact token cost, which the host
        reports as `usage.prompt_tokens`, against the window the model was
        loaded with. A tab that cannot fit a reply gets the model reloaded
        with a window that can, now, rather than cut off on its first
        message; a host that gives neither number says nothing. -> whether
        the model was reloaded, so the caller warms up again."""
        used = ((reply or {}).get("usage") or {}).get("prompt_tokens")
        if not isinstance(used, int):
            return False
        s.prefix_tokens = used            # exact, and kept for Diagnostics
        reloaded, note = self._fit(s, used)
        if note:
            self.q.put(("sys", s.event_id, note))
        return reloaded

    def _draft_check(self, s, llm):
        """After a request: a pair the host refused is said once, in the tab
        where it happened, and the Inference row stops naming a draft the
        shared model no longer runs - or shows how the one it runs is doing."""
        note = getattr(llm, "draft_note", None)
        if note:
            llm.draft_note = None
            self.q.put(("sys", s.event_id, note))
        if llm is self.llm and (note or getattr(llm, "draft", None)):
            self._host_line()

    def _llm_for(self, s):
        """The shared model, unless this app prefers one the host serves.

        A tab's model is fixed at boot: the executor, both warm-ups and the
        cached prefix on the host all have to agree, and swapping mid-session
        would throw the prefix away. Explains a departure once, in the tab.
        A model of the tab's own gets a draft model of its own, resolved the
        same way as the shared one's.
        """
        shared = getattr(self.llm, "model", None)
        if shared is None:
            return self.llm
        model, note = s.app.model_for(self.model_ids, shared)
        if note:
            self.q.put(("sys", s.event_id, "Model for this tab: %s (%s)." % (model, note)))
        if model == shared:
            return self.llm
        draft, note = eng.resolve_draft(model, self.model_ids)
        if note:
            self.q.put(("sys", s.event_id, note))
        return eng.LLM(self.host, model, self.llm.temperature, self.llm.timeout, draft=draft)

    def _ensure(self, s):
        """First view of a tab is what starts that app's bridge."""
        if s.ready or s.booting:
            return
        s.booting = True
        if s.app.images:
            s.booting, s.ready = False, True   # nothing to start: the form is ours
            s.status = ("Image Studio ready", "muted", False)
            s.bridge = ("ok", "%s\nready" % s.app.bridge_label)
            self._apply_status()
            s.images.start()
            return
        if s.app.panel:
            s.status = ("opening %s" % s.app.name + ELLIPSIS, "muted", False)
            self._apply_status()
            self._spawn(s.event_id, self._open_panel, s)
            return
        self._spawn(s.event_id, self._boot_session, s)

    def _boot_session(self, s):
        sid = s.event_id
        try:
            if not self.host_ready.is_set():
                self.q.put(("status", sid, ("waiting for the inference host" + ELLIPSIS,
                                            "muted", False)))
                self.host_ready.wait(timeout=240)
            if self.llm is None:
                # Set before the status is posted: the header reads it to
                # label the button Connect rather than Start <app>. The
                # retry loop may have stopped with no tab to serve; this
                # tab wants it.
                s.host_down = True
                self.q.put(("status", sid, ("no inference host - trying again", "err", True)))
                self.q.put(("bridge", sid, ("err", "%s\nno model" % s.app.bridge_label)))
                self.q.put(("host_retry", None, None))
                return
            s.host_down = False
            s.llm = self._llm_for(s)
            if self.vision_note:
                self.q.put(("sys", sid, self.vision_note))
            if self.draft_note and s.llm is self.llm:
                self.q.put(("sys", sid, self.draft_note))
            if s.notebook is not None and s.notebook.problem:
                self.q.put(("sys", sid, "Could not read this app's lessons: " + s.notebook.problem))
            elif s.notebook is not None and s.notebook.lessons:
                self.q.put(("sys", sid, "%d lesson%s from earlier work in %s are in the briefing "
                                        "(File > Lessons for this tab)."
                            % (len(s.notebook.lessons), "" if len(s.notebook.lessons) == 1 else "s",
                               s.app.name)))

            if s.app.bridged:
                self._boot_bridge(s)
                if s.mcp is None:         # it reported its own failure
                    return
                if s.closed:
                    return

            # The model has to be in VRAM with a window that holds this tab's
            # prefix before the warm-up pays for that prefix: loaded here, with
            # the window from an estimate, rather than just in time by the
            # warm-up at LM Studio's default and reloaded straight after.
            offered = tasks.inference_tools(s.tools, s.library)
            if not eng.loaded_instances(self.host, s.llm.model):
                self.q.put(("status", sid, ("loading %s on the host%s"
                                            % (s.llm.model, ELLIPSIS),
                                            "warn", False)))
            _, note = self._fit(s, eng.estimate_tokens(s.messages[0]["content"], offered),
                                exact=False)
            if note:
                self.q.put(("sys", sid, note))
            # Prefill dominates the first call - a full tool schema set is 9,000 to
            # 20,000 tokens, seconds on a model with the card to itself and over a
            # minute on one loaded beside another (see `_fit`). Pay it here against
            # the exact prompt prefix a real message will use, so the first question
            # comes back in seconds. Each tab has its own prefix, so each warms up
            # the first time it is opened. The reply carries the prefix's exact
            # cost; a window the estimate got wrong is fitted on it and the warm-up
            # paid once more.
            for attempt in (1, 2):
                self.q.put(("status", sid, ("warming up the model" + ELLIPSIS,
                                            "warn", False)))
                try:
                    reply = s.llm.chat([s.messages[0], {"role": "user", "content": "Say ready."}],
                                       offered, max_tokens=1)
                except eng.HostUnreachable as e:
                    self._host_lost()
                    self.q.put(("sys", sid, "Warm-up did not finish; the first request may be slower. " + str(e)))
                    break
                except Exception as e:
                    self.q.put(("sys", sid, "Warm-up did not finish; the first request may be slower. " + str(e)))
                    break
                self._host_back()
                if attempt == 2 or not self._headroom(s, reply):
                    break
            self._draft_check(s, s.llm)
            s.ready = True
            if s.app.bridged:
                self._refresh_bridge(s)
            else:
                self.q.put(("status", sid, ("ready", "ok", False)))
                self.q.put(("bridge", sid, ("ok", "no bridge\nthe model on its own")))
            self.q.put(("ready", sid, None))
            # Now the vision model, after this tab's own and not before it -
            # except beside a tab that renders, which clears the card for every
            # picture and has the vision model look at it on its own.
            if self.vision is not None and self.vision.needs_load and not s.app.gpu_tools:
                self._spawn(None, self._load_vision)
        finally:
            s.booting = False
            self.q.put(("idle", sid, None))

    def _boot_bridge(self, s):
        """
        Start this tab's bridge and work out what it offers. On failure it says
        why and leaves `s.mcp` None, which is how the caller knows to stop.
        """
        sid = s.event_id
        self.q.put(("status", sid, ("starting the %s bridge%s" % (s.app.name, ELLIPSIS),
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
        if s.app.research:
            # The research sidecar - this PC's files and the web, read-only, in
            # process - rides beside the bridge: its tools after the bridge's,
            # its schemas with them, and one Router in front of both so the
            # executor sees one client.
            try:
                s.sidecar = eng.research_client()
                extra = s.sidecar.list_tools()
            except Exception as e:
                extra = []
                self.q.put(("sys", sid, "The files-and-web tools are not available in this tab: %s" % e))
            s.schemas = allt + extra
            s.sidecar_names = frozenset(t["name"] for t in extra)
            if extra:
                s.mcp = eng.Router(mcp, s.sidecar)
        if getattr(s.app, "gpu_tools", None):
            s.mcp = eng.YieldGPU(s.mcp, s.app.gpu_tools, lambda: self._make_room(s),
                                 lambda ctx: self._give_back(s, ctx))
        if s.app.custom:
            # Nothing was known about this bridge until it answered: its groups
            # come from its tool names and its briefing from `instructions`, so
            # the system prompt - built at Session() - is rebuilt now, before
            # the warm-up pays for the prefix a real message will use.
            s.app.learn(allt, mcp.instructions)
            s.groups = list(s.app.default_groups)
            s.messages[0] = {"role": "system", "content": s.prompt()}
        wanted = s.app.tool_names(s.groups)
        s.tools = s.offered(wanted)
        self._load_library(s)
        missing = wanted - {t["name"] for t in allt}
        if missing:
            self.q.put(("sys", sid, "This bridge does not provide: " + ", ".join(sorted(missing))))

    # ------------------------------------------------------------------ panels
    def _build_panel(self, s):
        """A panel tab's body: a line of controls over the frame another
        program's window is held in. No transcript, no hero and no composer -
        the window is the tab. See studio_milanote.py."""
        s.frame = self._skin(tk.Frame(self.stack), bg="bg")
        bar = self._skin(tk.Frame(s.frame), bg="bg")
        bar.pack(side="top", fill="x", padx=18, pady=(0, 8))
        self._button(bar, "Upload files", lambda: self._panel_upload(s),
                     kind="accent").pack(side="left")
        self._button(bar, "Reload", lambda: self._panel_reload(s)).pack(
            side="left", padx=(6, 0))
        s.panel_note = tk.Label(bar, font=self.f_ui, anchor="w", text=PANEL_HINT)
        self._skin(s.panel_note, bg="bg", fg="muted")
        s.panel_note.pack(side="left", fill="x", expand=True, padx=(12, 0))
        s.panel_host = tk.Frame(s.frame, bd=0, highlightthickness=0)
        self._skin(s.panel_host, bg="card")
        s.panel_host.pack(side="top", fill="both", expand=True, padx=14, pady=(0, 14))
        s.panel_host.bind("<Configure>",
                          lambda ev: self._fit_panel(s, ev.width, ev.height))

    def _build_images(self, s):
        """The Image Studio tab: our own form, in the place a panel tab's
        window goes. Its note line is the panel's, so an error from a worker
        (`_guard`) is said there like on any panel tab."""
        s.frame = self._skin(tk.Frame(self.stack), bg="bg")
        s.images = images_ui.ImageStudio(self, s)
        s.panel_note = s.images.note

    def _images_make_room(self, backend):
        """Before an Image Studio job on a ComfyUI that shares the LLM PC's
        GPU: LM Studio's models off the card, as `_make_room` does for the
        ComfyUI tab - a model another tab is mid-request on stays, and the
        rest reload themselves when next used (`_reload_if_unloaded`)."""
        if not self.host:
            return
        keep = {getattr(o.llm or self.llm, "model", None)
                for o in self.sessions.values() if o.busy}
        with self.fit_lock:
            eng.make_room(self.host, keep)

    def _fit_panel(self, s, width, height):
        if s.browser is not None:
            s.browser.fit(width, height)

    def _focus_panel(self, s):
        if s.browser is not None:
            s.browser.focus()

    def _panel_say(self, s, text, role="muted"):
        if s.panel_note is not None:
            s.panel_note.config(text=text)
            self._skin(s.panel_note, bg="bg", fg=role)

    def _open_panel(self, s):
        """Off the UI thread: start the tab's window. `_panel_event` adopts
        it into the tab on the UI thread, which owns the frame."""
        browser = milanote.Browser(url=s.app.url)
        s.browser = browser
        try:
            browser.start()
        except Exception as e:
            s.browser = None
            browser.close(0)
            self.q.put(("panel", s.event_id, ("failed", str(e))))
            return
        if s.closed:                      # the tab went while the window came
            s.browser = None
            browser.close()
            return
        self.q.put(("panel", s.event_id, ("window", browser)))

    def _measure_panel(self, s, browser):
        """How much of the window is its own frame, so fitting it can clip
        that off. Twice: the frame settles a moment after it is adopted."""
        for wait in (1.0, 3.0):
            time.sleep(wait)
            if browser is not s.browser:
                return
            try:
                before = browser.inset
                if browser.measure() != before:
                    self.q.put(("panel", s.event_id, ("measured", browser)))
            except Exception:
                return                    # it keeps its title bar; nothing else is wrong

    def _panel_event(self, s, kind, payload):
        """`_handle` for a panel tab. What would be written into a transcript
        is said on its note line; the rest is about a conversation it has not."""
        if kind == "panel":
            what, arg = payload
            if what == "window" and arg is s.browser:
                s.panel_host.update_idletasks()
                arg.embed(s.panel_host.winfo_id(), s.panel_host.winfo_width(),
                          s.panel_host.winfo_height())
                s.booting, s.ready = False, True
                s.status = ("%s is open" % s.app.name, "muted", False)
                s.bridge = ("ok", "%s\nopen" % s.app.bridge_label)
                if s.id == self.active:
                    self._focus_panel(s)
                self._spawn(s.event_id, self._measure_panel, s, arg)
            elif what == "measured" and arg is s.browser:
                arg.fit(*arg.size)
            elif what == "failed":
                s.booting = False
                s.status = ("could not open %s" % s.app.name, "err", False)
                s.bridge = ("err", "%s\nnot open" % s.app.bridge_label)
                self._panel_say(s, arg[:1].upper() + arg[1:], "err")
            elif what == "note":
                self._panel_say(s, *arg)
            self._paint_app_dot(s)
            if s.id in self.tab_ui:
                self._paint_tab(s.id)
            if s.id == self.active:
                self._apply_status()
        elif kind == "images":
            if s.images is not None:
                s.images.handle(payload)
        elif kind in ("error", "sys"):
            self._panel_say(s, payload, "err" if kind == "error" else "muted")
        elif kind == "trace":
            self._log(payload)
        elif kind == "status":
            s.status = payload
            if s.id == self.active:
                self._apply_status()
        elif kind == "idle":
            s.busy = False
            if s.id == self.active:
                self._apply_status()

    def _panel_upload(self, s):
        if s.browser is None or not s.ready:
            self._panel_say(s, "%s is still opening." % s.app.name, "warn")
            return
        paths = filedialog.askopenfilenames(parent=self, title="Upload to %s" % s.app.name)
        if not paths:
            return
        self._panel_say(s, "Uploading %d file%s" % (len(paths), "" if len(paths) == 1 else "s")
                        + ELLIPSIS)
        self._spawn(s.event_id, self._upload_panel, s, list(paths))

    def _upload_panel(self, s, paths):
        browser = s.browser
        try:
            if browser is None:
                raise RuntimeError("%s is not open in this tab" % s.app.name)
            said, role = browser.upload(paths), "ok"
        except Exception as e:
            said, role = str(e), "err"
        self.q.put(("panel", s.event_id, ("note", (said[:1].upper() + said[1:], role))))

    def _panel_reload(self, s):
        """Reload the page; a window that has gone is opened again."""
        if s.browser is not None and s.browser.running():
            self._spawn(s.event_id, self._reload_panel, s)
        elif not s.booting:
            s.browser, s.ready = None, False
            self._panel_say(s, "Opening %s again" % s.app.name + ELLIPSIS)
            self._ensure(s)

    def _reload_panel(self, s):
        browser = s.browser
        try:
            browser.reload()
            said, role = PANEL_HINT, "muted"
        except Exception as e:
            said, role = str(e), "err"
        self.q.put(("panel", s.event_id, ("note", (said[:1].upper() + said[1:], role))))

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
        if s is None or s.busy or s.app.panel:
            return
        if s.host_down:                   # the button reads Connect
            self._connect_host()
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
                self.q.put(("status", s.event_id,
                            ("launching %s%s" % (s.app.name, ELLIPSIS), "warn", False)))
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
        """Every task saved for this tab's app, newest first, each showing
        what was asked of it and how far it got.

        This was a file dialog pointed at a folder of 32-character hex names:
        you could see that you had eleven saved tasks and not one of them told
        you which was which, so in practice nothing was ever resumed. Every
        message checkpoints its tab, so this is the app's memory of the work -
        it is worth being able to read. Nothing here is deleted on the app's
        own initiative; the button asks first."""
        s = self.cur()
        if s is None or s.busy or s.app.panel:
            return
        key = ("tasks", s.id)
        win = self.windows.get(key)
        if win is not None and win.winfo_exists():
            win.destroy()                 # rebuilt: a resume or a delete changed the list
        win = tk.Toplevel(self)
        self.windows[key] = win
        win.title("History - %s" % s.app.name)
        win.geometry("%dx%d" % (self._px(660), self._px(520)))
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
        saved = tasks.TaskRecord.summaries(os.path.dirname(self._task_path(s)))
        # Not the conversation already on screen - resuming that is a no-op
        # that would replace it with a checkpoint of itself.
        here = s.record.id + ".json"
        saved = [t for t in saved if os.path.basename(t["path"]) != here]
        if not saved:
            view.insert("end", "No past conversations with %s yet.\n\n" % s.app.name, "group")
            view.insert("end", "Every message is saved with the tab it was sent in, so a "
                               "conversation shows up here as soon as you ask for "
                               "something and start a new one.\n",
                        "desc")
        else:
            view.insert("end", "%d past conversation%s with %s, newest first.\n"
                        % (len(saved), "" if len(saved) == 1 else "s", s.app.name),
                        "group")
            for task in saved:
                view.insert("end", "\n")
                if not task["problem"]:
                    view.window_create("end",
                                       window=self._task_button(view, s, task, "open"))
                view.window_create("end",
                                   window=self._task_button(view, s, task, "delete"))
                view.insert("end", "  " + self._task_title(task) + "\n", "name")
                detail = task["problem"] or "%d step%s  ·  %s" % (
                    task["steps"], "" if task["steps"] == 1 else "s",
                    (task["status"] or "no status recorded")[:80])
                view.insert("end", "      %s  ·  %s\n" % (
                    time.strftime("%d %b %Y, %H:%M", time.localtime(task["when"])),
                    detail), "off" if task["problem"] else "desc")
        view.config(state="disabled")
        self.tasks_view = view            # for the tests

    @staticmethod
    def _task_title(task):
        """The first line of what was asked, which is how a person recognises
        a task. A record saved before anything was asked has no brief."""
        lines = [l for l in (task["brief"] or "").strip().splitlines() if l.strip()]
        return (lines[0][:110] if lines else "(nothing asked yet)")

    def _task_button(self, parent, s, task, kind):
        def resume():
            if s.busy:
                return
            win = self.windows.get(("tasks", s.id))
            if win is not None and win.winfo_exists():
                win.destroy()
            self._restore_task(s, task["path"])

        def delete():
            # Saved conversations are the user's own work. The app never
            # removes one on its own - not on a timer, not to keep a folder
            # tidy - and when asked it asks again first.
            if not messagebox.askyesno(
                    "Delete conversation",
                    "Delete this conversation?\n\n%s\n\nThis cannot be undone."
                    % self._task_title(task), parent=parent):
                return
            try:
                os.unlink(task["path"])
            except OSError as e:
                self._write(s, "Could not delete that saved task: %s\n" % e, "err")
            self._resume_task()           # rebuilt without it
        return self._button(parent, kind, resume if kind == "open" else delete,
                            kind="ghost", bg="card", font=self.f_small,
                            padx=self._px(9), pady=self._px(1), r=self._px(8))

    def _restore_task(self, s, path):
        try:
            record, messages = tasks.TaskRecord.restore(path, s.prompt())
            if record.app_id != s.id:
                raise ValueError("This task belongs to a different app. Open its tab to resume it.")
            # Keep the current conversation recoverable when replacing it.
            if s.record.briefs:
                s.record.save(self._task_path(s), s.messages)
            s.record, s.messages = record, messages
            s.cancel.clear()
            self._clear_view(s)
            self._hide_hero(s)
            for msg in messages:
                if msg.get("role") in ("user", "assistant") and isinstance(msg.get("content"), str):
                    self._role(s, "YOU" if msg["role"] == "user" else s.app.tab.upper(), "role_user" if msg["role"] == "user" else "role_asst")
                    self._write(s, msg["content"] + "\n", "user" if msg["role"] == "user" else "asst")
            self._write(s, "Reopened from history. Send a message to carry on; the current project must be inspected first.\n", "sys")
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
        if s is None or s.busy or s.booting or not s.ready or s.app.panel:
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
            s.tools = s.offered(wanted)
            # A made tool built on a tool this tab no longer offers is dropped
            # from the model's list rather than failing when it is called.
            self._load_library(s)
            s.busy = True
            s.cancel.clear()
            s.status = ("warming selected capabilities" + ELLIPSIS, "warn", False)
            self._apply_status()
            self._spawn(s.event_id, self._warm_capabilities, s)
            win.destroy()
        button = self._button(win, "Apply", apply, kind="accent")
        button.pack(padx=self._px(16), pady=self._px(12))

    def _warm_capabilities(self, s):
        try:
            reply = s.llm.chat([s.messages[0], {"role": "user", "content": "Say ready."}],
                               tasks.inference_tools(s.tools, s.library), max_tokens=1)
            self.q.put(("sys", s.event_id, "Selected capabilities are ready."))
            self._draft_check(s, s.llm)
            self._headroom(s, reply)
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
            self._end_thinking(s)
            # Where the placeholder was standing, if one was; otherwise the end
            # of the transcript, the way an attachment goes in.
            at = self._take_stage(s) or s.view.index("end-1c")
            s.view.config(state="normal")
            canvas = self._reveal(s.view, photo)
            canvas.bind("<Button-3>", lambda ev: self._picture_menu(ev, item), add="+")
            canvas.bind("<Double-Button-1>", lambda ev: self._open_picture(item), add="+")
            self._embed(s.view, canvas, (), at)
            s.view.insert("%s+1c" % at, "\n")
            s.view.config(state="disabled")
            s.view.see("end")
        except Exception as e:
            if item.get("file"):
                raise
            self._write(s, "Preview could not be displayed: %s\n" % e, "sys")

    @staticmethod
    def _picture_path(item):
        """The file a preview shows, when it is on this workstation: the
        preview's own file, or where the bridge saved it (`_meta.path`)."""
        for path in (item.get("file"), (item.get("_meta") or {}).get("path")):
            if path and os.path.isfile(path):
                return path
        return None

    @staticmethod
    def _picture_bytes(item):
        path = Chat._picture_path(item)
        if path:
            with open(path, "rb") as f:
                return f.read()
        return base64.b64decode(item.get("data", ""))

    def _picture_menu(self, ev, item):
        """Right-click on a picture: save it, open it, find it. The preview is
        shrunk to fit the transcript; each of these is the full picture."""
        path = self._picture_path(item)
        m = self._menu()
        m.add_command(label="Save picture as" + ELLIPSIS, command=lambda: self._save_picture(item))
        m.add_command(label="Open", command=lambda: self._open_picture(item))
        if path:
            m.add_command(label="Show in folder",
                          command=lambda: self._show_in_folder(path))
            m.add_command(label="Copy path", command=lambda: (
                self.clipboard_clear(), self.clipboard_append(path)))
        try:
            m.tk_popup(ev.x_root, ev.y_root)
        finally:
            m.grab_release()

    @staticmethod
    def _show_in_folder(path):
        # Not procs.spawn: the Explorer window is the user's, and must outlive
        # this app rather than be stopped with its children.
        import subprocess
        subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])

    def _save_picture(self, item):
        path = self._picture_path(item)
        ext = os.path.splitext(path)[1] if path else ".png"
        name = os.path.basename(path) if path else "picture" + ext
        target = filedialog.asksaveasfilename(
            parent=self, title="Save picture", initialfile=name, defaultextension=ext,
            initialdir=os.path.join(os.path.expanduser("~"), "Pictures"),
            filetypes=[("Picture", "*" + ext), ("All files", "*.*")])
        if not target:
            return
        try:
            with open(target, "wb") as f:
                f.write(self._picture_bytes(item))
        except (OSError, ValueError) as e:
            messagebox.showerror("Save picture", "Could not save it: %s" % e, parent=self)

    def _open_picture(self, item):
        """In the default viewer. A picture with no file of its own - a
        screenshot a bridge sent as data - is written to a temporary one."""
        path = self._picture_path(item)
        try:
            if not path:
                import tempfile
                fd, path = tempfile.mkstemp(suffix=".png", prefix="studio-picture-")
                with os.fdopen(fd, "wb") as f:
                    f.write(self._picture_bytes(item))
                item["file"] = path
            os.startfile(path)
        except (OSError, ValueError) as e:
            messagebox.showerror("Open picture", "Could not open it: %s" % e, parent=self)

    def _reveal(self, v, photo):
        """A picture on a canvas, wiped in from the left over about half a
        second. Tk cannot fade an image - PhotoImage has no alpha to animate -
        but it can uncover one, and a picture that arrives rather than appears
        is the difference between a window that is working and a window that
        blinked. One pass, then the cover is gone and the canvas is a picture
        like any other."""
        w, h = photo.width(), photo.height()
        c = tk.Canvas(v, width=w, height=h, highlightthickness=0, bd=0)
        self._skin(c, bg="bg")
        c.create_image(0, 0, image=photo, anchor="nw")
        r = self._px(10)
        self._round_off(c, w, h, r)
        # The corners are painted in the background they sit on, so a theme
        # switch would leave four blots of the old one on every picture in
        # the transcript.
        self._repaint_on_theme(c, lambda: self._round_off(c, w, h, r))
        cover = c.create_rectangle(0, 0, w, h, fill=self.C["bg"], width=0)
        start = self.anim_frame

        def draw(frame):
            step = frame - start + 1
            if step >= REVEAL_FRAMES:
                c.delete(cover)
                return False
            # Eased, so it arrives rather than stops: a linear wipe at this
            # length reads as a scan line.
            t = 1 - (1 - step / float(REVEAL_FRAMES)) ** 2
            c.coords(cover, w * t, 0, w, h)

        self._animate(("reveal", str(c)), draw)
        return c

    def _on_return(self, ev):
        if ev.state & 0x0001:  # Shift+Enter = newline
            return None
        if self.cur() is None or not self.cur().busy:
            self._on_send()
        return "break"

    def _on_new(self):
        s = self.cur()
        if s is None or s.busy or s.app.panel:
            return
        s.reset()
        self._clear_view(s)
        self._welcome(s)

    def _on_send(self):
        s = self.cur()
        if s is None or s.app.panel:
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
        self._send(s, task, attached, note)

    def _send(self, s, task, attached=(), note=""):
        """One message into a tab's conversation - typed, or a click on the
        model's question - and the turn that answers it. An open question form
        is closed either way: a typed reply is an answer too."""
        self._settle_ask(s)
        self._hide_hero(s)
        self._role(s, "YOU", "role_user")
        self._write(s, task + "\n", "user")
        for p in attached:
            self._show_attachment(s, p)
        s.messages.append({"role": "user", "content": task + note})
        s.record.briefs.append(task + note)
        s.cancel.clear()
        s.busy = True
        s.status = ("working" + ELLIPSIS, "warn", False)
        self._apply_status()
        self._begin_thinking(s)
        # The vision model is asked about pictures only; the rest are paths.
        self._spawn(s.event_id, self._turn, s, [p for p in attached if is_picture(p)])

    # ------------------------------------------------------ the model's question
    def _show_ask(self, s, asked):
        """studio_ask, as a form in the transcript: a button per option - boxes
        to tick when several may apply - and one for an answer of the user's
        own. A click is the user's next message; the form then stays, greyed,
        so the transcript still shows what was asked and chosen."""
        self._settle_ask(s)
        view = s.view
        # The card is drawn, so its corners can be round; the form is a frame
        # on top of it, inset clear of the curve. Same shape as a tab chip or
        # a rail row - see `_make_tab`.
        PAD, R = self._px(10), self._px(14)
        shell = tk.Canvas(view, highlightthickness=0, bd=0)
        self._skin(shell, bg="bg")
        frame = self._skin(tk.Frame(shell, padx=self._px(4), pady=self._px(2)),
                           bg="card")
        shell.create_window(PAD, PAD, window=frame, anchor="nw")

        def paint(_ev=None):
            w = frame.winfo_reqwidth() + 2 * PAD
            h = frame.winfo_reqheight() + 2 * PAD
            shell.config(width=w, height=h)
            shell.delete("card")
            fill = self.C["card"]
            rounded(shell, 0, 0, w, h, R, fill=fill, outline=fill, tags="card")
            shell.tag_lower("card")

        frame.bind("<Configure>", paint)
        self._repaint_on_theme(shell, paint)
        wrap = self._px(520)
        self._skin(tk.Label(frame, text=asked["question"], font=self.f_body,
                            wraplength=wrap, justify="left", anchor="w"),
                   bg="card", fg="text").pack(fill="x", pady=(0, self._px(6)))
        buttons = []

        def answer(text):
            if self.sessions.get(s.id) is not s or s.busy:
                return
            self._send(s, text)

        if asked.get("multiple"):
            picks = []
            for option in asked["options"]:
                var = tk.BooleanVar(master=self, value=False)
                text = option["label"]
                if option.get("description"):
                    text += "  -  " + option["description"]
                box = tk.Checkbutton(frame, text=text, variable=var, font=self.f_body,
                                     anchor="w", justify="left", wraplength=wrap,
                                     cursor="hand2", bd=0, highlightthickness=0)
                self._skin(box, bg="card", fg="text", selectcolor="bg",
                           activebackground="card", activeforeground="text")
                box.pack(fill="x")
                picks.append((option["label"], var))
                buttons.append(box)

            def send_picks():
                chosen = [label for label, var in picks if var.get()]
                if chosen:
                    answer("; ".join(chosen))
            go = self._button(frame, "Send these", send_picks, kind="accent",
                              bg="card", font=self.f_body, padx=self._px(14),
                              pady=self._px(5), r=self._px(11))
            go.pack(anchor="w", pady=(self._px(6), 0))
            buttons.append(go)
        else:
            for option in asked["options"]:
                # A full-width row that is also a button: `anchor="w"` gives a
                # pill the packer's width and a left-read label that wraps.
                button = self._button(
                    frame, option["label"], lambda t=option["label"]: answer(t),
                    kind="option", bg="card", font=self.f_body, anchor="w",
                    padx=self._px(11), pady=self._px(5), r=self._px(10))
                button.pack(fill="x", pady=(0, self._px(2)))
                buttons.append(button)
                if option.get("description"):
                    self._skin(tk.Label(frame, text=option["description"], font=self.f_small,
                                        wraplength=wrap, justify="left", anchor="w"),
                               bg="card", fg="faint").pack(fill="x", padx=(10, 0),
                                                           pady=(0, self._px(4)))
        other = self._button(frame, "Something else…", self.input.focus_set,
                             kind="ghost", bg="card", font=self.f_small,
                             padx=self._px(9), pady=self._px(3), r=self._px(9))
        other.pack(anchor="w", pady=(self._px(4), 0))
        buttons.append(other)

        view.config(state="normal")
        view.insert("end", "\n")
        view.window_create("end", window=shell, padx=self._px(4))
        view.insert("end", "\n")
        view.config(state="disabled")
        # The form has no height until Tk lays it out, so a scroll now stops
        # short of it; scroll again once it has one.
        view.see("end")
        view.after_idle(lambda: view.winfo_exists() and view.see("end"))
        s.ask_buttons = buttons

    def _settle_ask(self, s):
        """Grey the question form once it has been answered - by a click or by
        a typed message - so it cannot send a second answer."""
        for button in s.ask_buttons:
            try:
                # A Pill has to be repainted to look disabled; the tick boxes
                # beside it are still Tk's own and answer to config().
                if isinstance(button, Pill):
                    button.set(state="disabled")
                else:
                    button.config(state="disabled")
            except tk.TclError:
                pass
        s.ask_buttons = []

    def _turn(self, s, pictures=()):
        sid = s.event_id
        started = {"value": False}
        lost = False
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
            # A run that ended in an error can leave a render's model away;
            # it comes back at the window it had.
            eng.settle(s.mcp)
            self._reload_if_unloaded(s)
            if pictures and self.vision:
                # The executing model reads text. Put what the pictures show
                # into the brief itself, so it survives checkpoints and resume.
                emit("status", ("looking at the pictures" + ELLIPSIS, "muted", True))
                try:
                    s.messages[-1]["content"] += self.vision.describe_all(pictures)
                    s.record.briefs[-1] = s.messages[-1]["content"]
                except Exception as e:
                    emit("sys", "The vision model could not describe the pictures (%s); "
                                "the model has their paths only." % e)
            elif pictures:
                emit("sys", "No vision model is served, so the model has only the names "
                            "and paths of the pictures - it cannot see what is in them.")
            look = None
            if self.vision:
                look = self.vision.check if s.app.makes_pictures else self.vision.review
            executor = tasks.Executor(s.llm or self.llm, s.mcp, s.tools, schemas=s.schemas,
                record=s.record, cancel=s.cancel, emit=emit, checkpoint=checkpoint,
                vision=look, library=s.library,
                readback=s.app.readback, review=s.app.review, notebook=s.notebook)
            executor.run(s.messages, MAX_STEPS)
            self._learn(s, executor, emit)
        except Exception as e:
            s.record.status = "interrupted; inspect project state before continuing"
            try:
                checkpoint()
            except Exception:
                pass
            lost = isinstance(e, eng.HostUnreachable)
            raise
        finally:
            self.q.put(("stream_end", sid, None))
            self._draft_check(s, s.llm or self.llm)
            if lost:
                # The host went away mid-conversation: the row says so, the
                # button offers Connect, and the window keeps trying. Sending
                # again would also do.
                s.host_down = True
                self._host_lost()
                self.q.put(("status", sid, ("inference host unreachable - trying again", "err", True)))
                self.q.put(("host_retry", None, None))
            else:
                s.host_down = False
                self._host_back()
                complete = s.record.status.startswith("response complete")
                status = "stopped" if s.cancel.is_set() else "ready" if complete else "needs attention"
                self.q.put(("status", sid, (status, "muted" if complete else "warn", False)))
            self.q.put(("idle", sid, None))

    def _learn(self, s, executor, emit):
        """What the run leaves in the app's notebook, said in the tab. The
        reflection is one more short request, made only after a run with an
        error in it or a brief that read as a correction - a clean run has
        nothing to teach and pays nothing."""
        if s.notebook is None or s.cancel.is_set():
            return
        brief = s.record.briefs[-1] if s.record.briefs else ""
        if executor.trouble or lessons.looks_like_correction(brief):
            emit("status", ("thinking about what to remember" + ELLIPSIS, "muted", True))
        for text in eng.learn_from_run(executor, s.messages, s.notebook,
                                       s.llm or self.llm, s.app.name):
            emit("sys", "Lesson kept for %s: %s" % (s.app.name, text))

    # ------------------------------------------------ the studio and the lessons
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
            view.insert("end", "%d lesson%s, oldest first. Every tab for %s carries them.\n"
                        % (len(kept), "" if len(kept) == 1 else "s", s.app.name), "group")
            for lesson in kept:
                view.insert("end", "\n")
                view.window_create("end", window=self._forget_lesson_button(view, s, lesson["text"]))
                view.insert("end", "  " + lesson["text"] + "\n", "name")
                view.insert("end", "      %s%s\n" % (
                    {"user": "you said so", "model": "the model kept it",
                     "review": "reflected after a task", "error": "a refused call"}[lesson["source"]],
                    "  ·  came up %d more time%s" % (lesson["hits"], "" if lesson["hits"] == 1 else "s")
                    if lesson["hits"] else ""), "desc")
        view.config(state="disabled")
        self.lessons_view = view          # for the tests

    # --------------------------------------------------------- diagnostics
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

    def _forget_lesson_button(self, parent, s, text):
        def forget():
            if s.busy:
                return                    # the worker may be writing the notebook
            s.notebook.remove(text)
            self._lessons_window()
        return self._button(parent, "forget", forget, kind="ghost", bg="card",
                            font=self.f_small, padx=self._px(9),
                            pady=self._px(1), r=self._px(8))

    def _quit(self):
        if self.closing:
            return                        # a signal and the close box, together
        # Flag first, then cancel: a tick that fires between the two sees the
        # flag and does not re-arm. Leaving them armed is what printed
        # "invalid command name ..._drain" over a window that was already gone.
        self.closing = True
        for s in self.sessions.values():
            if s.images is not None:
                s.images.release()        # on this thread: an unsaved scene asks first
        self.anim.clear()
        for timer in ("drain_timer", "host_timer", "anim_timer", "update_timer"):
            self._stand_down(timer)
        # Off the screen at once; the bridges get their grace behind it, all
        # together. One at a time, each allowed seconds to exit, was a window
        # frozen for as long as six tabs took.
        try:
            self.withdraw()
        except tk.TclError:
            pass
        closers = [threading.Thread(target=s.close, daemon=True)
                   for s in self.sessions.values()]
        for t in closers:
            t.start()
        deadline = time.monotonic() + QUIT_GRACE_S
        for t in closers:
            t.join(max(0.0, deadline - time.monotonic()))
        procs.stop_all(0)                 # whatever did not go, and its tree
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


def fail_visibly(doing, trace):
    """Say something when there is no window to say it in. A failure while the
    app was still building itself - a font, an icon, a settings file Tk choked
    on - used to look exactly like double-clicking the shortcut and nothing
    happening at all, because `pythonw.exe` has no console to print to. Log it,
    then put it on the screen, because the log is no use to someone who has not
    been told there is one."""
    path = log_error("While %s:\n%s" % (doing, trace))
    try:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            APP_NAME, "%s ran into a problem while %s.\n\n%s\n\n%s"
            % (APP_NAME, doing, trace.strip().rsplit("\n", 1)[-1],
               "The details are in\n" + path if path
               else "The error log could not be written."))
        root.destroy()
    except Exception:
        pass


def main():
    # Before anything that needs a window. `--doctor` is the answer to "I
    # clicked it and nothing happened": it opens no window, takes no
    # single-instance lock, and so runs beside a copy that is already up.
    if "--doctor" in sys.argv[1:]:
        sections = doctor.report()
        print(doctor.as_text(sections))
        raise SystemExit(doctor.worst(sections))
    try:  # crisp text on a high-DPI display
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    # A worker started outside `Chat._spawn` has no `_guard` around it; without
    # this its traceback dies with the thread.
    threading.excepthook = lambda args: log_error(
        "In thread %s:\n%s" % (args.thread and args.thread.name,
                               "".join(traceback.format_exception(
                                   args.exc_type, args.exc_value, args.exc_traceback))))
    if not claim_single_instance():
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo(APP_NAME, "%s is already running.\n\n"
                                      "Look for its window on the taskbar." % APP_NAME)
        root.destroy()
        return
    try:
        app = Chat()
        # Ctrl+C / Ctrl+Break in a console, or a SIGTERM: the same orderly
        # quit as the close box. Tk's loop runs Python only in callbacks, and
        # the pump's timer is one, so the handler runs within a tick.
        procs.on_shutdown(lambda: app.after(0, app._quit))
        app.mainloop()
    except Exception:
        fail_visibly("starting up", traceback.format_exc())
        raise SystemExit(1)
    if app.restart:
        relaunch()


def relaunch():
    """Start the updated code once this copy has let go of the lock."""
    if _LOCK is not None:
        _LOCK.close()
    subprocess.Popen([sys.executable, os.path.abspath(__file__)] + sys.argv[1:],
                     cwd=HERE, close_fds=True)


if __name__ == "__main__":
    main()
