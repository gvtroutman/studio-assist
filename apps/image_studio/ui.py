#!/usr/bin/env python3
"""
studio_images_ui - the Image Studio tab: Person -> Style -> Scene -> Reference
-> Generate, over `studio_imagegen`.

A collaborator of `studio_chat.Chat`, not a piece of it (AGENTS.md: further
splitting is extraction, one owner of its own state at a time). The window
hosts it as a panel tab and lends it what every widget there goes through -
`_skin` for palette roles, `_button` for Pills, `_entry` for fields, `_spawn`
for work off the UI thread, `q` for the way back, `_animate` for the one tick.
Worker threads never touch a widget: a job's changes arrive as ("images", sid,
payload) events through the window's pump and land in `handle()`.
"""

import colorsys
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
import tkinter as tk
from tkinter import filedialog, messagebox

import apps.image_studio.addons.catalog as catalog
import apps.image_studio.addons.hub as hub
import apps.image_studio.addons.civitai as civitai
import apps.image_studio.model_sources as model_sources
import apps.image_studio.addons.discovery as discovery
import apps.image_studio.addons.nodes as addons
import apps.comfyui.view as comfy_view
import apps.image_studio.imagegen as ig
import apps.image_studio.facefusion as ff
import apps.image_studio.blend as sb
import apps.image_studio.viewcube as viewcube
import apps.image_studio.lora_train as lt
import apps.image_studio.scene.pose as sp
import apps.image_studio.scene.ui as studio_scene_ui

THUMB = 72                    # px, before the display's scale
STYLE_TILE = 104              # px, before the display's scale; the examples are 208
CAMERA_CARD = 150             # px, the Shot on card's long edge, before the display's scale
HISTORY_PAGE = 40
# The look sections the People tab shows. Body and Accessories are the
# Editor's (CharacterCreator) alone: a character's still reach the prompt.
FORM_LOOKS = [(name, slots) for name, slots in ig.LOOKS
              if name not in ("Body", "Accessories")]
ELLIPSIS = "…"          # the window's marker for "still happening": it animates
ADVANCED = [                  # (setting, label, kind)
    ("seed", "Seed", "int"),
    ("steps", "Steps", "int"),
    ("guidance", "Guidance", "float"),
    ("sampler", "Sampler", "text"),
    ("scheduler", "Scheduler", "text"),
    ("width", "Width", "int"),
    ("height", "Height", "int"),
    ("denoise", "Denoise", "float"),
    ("upscale", "Upscale factor", "float"),
    ("refine_denoise", "Refine denoise", "float"),
    ("batch", "Batch size", "int"),
]
STATUS_ROLE = {"queued": "muted", "uploading": "accent", "loading": "accent",
               "sampling": "accent", "decoding": "accent", "running": "accent",
               "refining": "accent", "face": "accent", "critic": "accent",
               "head_swap": "accent", "face_swap": "accent", "eyes": "accent",
               "hands": "accent",
               "glasses": "accent", "complete": "ok", "failed": "err", "cancelled": "faint"}
STATUS_TEXT = {"face": "Face pass", "critic": "Critic", "head_swap": "Head swap",
               "face_swap": "Face swap",
               "eyes": "Eye pass", "hands": "Hand pass", "glasses": "Glasses"}
# A status to the key on the job's own pipeline strip (ig.pipeline_stages) it
# lights up. Queued/uploading/loading run before the strip's first stop, so
# nothing is lit yet.
STAGE_KEY = {"queued": None, "uploading": None, "loading": None, "running": "sampling",
             "sampling": "sampling", "face": "face", "critic": "critic",
             "decoding": "decoding", "head_swap": "head_swap", "face_swap": "face_swap",
             "eyes": "eyes",
             "hands": "hands", "glasses": "glasses", "complete": "complete"}
READY_MARK = {"ready": "✓", "missing": "✗", "offline": "○", "disabled": "–",
              "unchecked": "?"}


def open_path(path, select=False):
    if sys.platform == "win32":
        if select:
            subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])
        else:
            os.startfile(path)            # noqa - Windows only, like the rest of the app


def photo(path, box, profile=False):
    """A PhotoImage of `path` shrunk by a whole factor to fit `box` px, or None.
    Tk reads PNG and GIF itself; anything else has no preview. `profile`
    shows a profile photo's normalized thumbnail (prepare_previews) when
    there is one; a picture shown as itself never gets it."""
    if profile:
        small = ff.preview_path(path)
        path = small if os.path.isfile(small) else path
    try:
        img = tk.PhotoImage(file=path)
    except (tk.TclError, OSError):
        return None
    k = max(1, -(-max(img.width(), img.height()) // max(1, box)))
    return img.subsample(k) if k > 1 else img


def photo_at(path, side, master):
    """A PhotoImage of `path` at `side` px on its long edge, near enough, or
    None. Tk scales only by whole factors, so this zooms by a and subsamples
    by b for the a/b (both at most 8) closest to the ratio it needs. Made in
    `master`'s interpreter: without one it belongs to the first Tk root."""
    try:
        img = tk.PhotoImage(master=master, file=path)
    except (tk.TclError, OSError):
        return None
    want = side / max(1, img.width(), img.height())
    a, b = min(((a, b) for a in range(1, 9) for b in range(1, 9)),
               key=lambda ab: (abs(ab[0] / ab[1] - want), ab[0]))
    if a > 1:
        img = img.zoom(a)
    return img.subsample(b) if b > 1 else img


PEEK = 520                    # px, the hover preview's long edge, before the display's scale
PEEK_DELAY = 350              # ms the pointer rests on a picture before its preview opens


class Peek:
    """A larger copy of a picture while the pointer rests on its thumbnail.
    One borderless window per Tk root, opened after PEEK_DELAY beside the
    pointer and kept on the screen; leaving the picture, clicking or turning
    the wheel closes it. Wire a widget with `peek(widget, path)`."""
    GAP = 18                  # px between the pointer and the preview

    def __init__(self, root):
        self.root = root
        self.win = None
        self.source = None        # what the pointer is resting on
        self.timer = None
        self.cache = {}           # (path, mtime, side) -> PhotoImage, the last few

    def side(self):
        scale = self.root.winfo_fpixels("1i") / 96.0
        return max(64, min(int(PEEK * scale), int(self.root.winfo_screenheight() * 0.8)))

    def image(self, path):
        """The picture at PEEK, or at its own size where that is smaller:
        Tk only enlarges by repeating pixels."""
        try:
            with open(path, "rb") as f:
                size = ig.picture_size(f.read(64))
            side = min(self.side(), max(size)) if size else self.side()
            key = (path, os.path.getmtime(path), side)
        except OSError:
            return None
        if key not in self.cache:
            img = photo_at(path, key[2], self.root)
            if img is None:
                return None
            if len(self.cache) >= 8:
                self.cache.pop(next(iter(self.cache)))
            self.cache[key] = img
        return self.cache[key]

    def enter(self, ev, source):
        if self.source is source and (self.timer or self.shown()):
            return                # from a thumbnail's frame onto its picture
        self.hide()
        self.source = source
        # On the root, which also cancels it: a widget's own `after` would
        # leave the command in its list, and its destroy fails deleting it again.
        widget = ev.widget
        self.timer = self.root.after(PEEK_DELAY, lambda: self.show(widget))

    def leave(self, ev):
        try:
            under = ev.widget.winfo_containing(ev.x_root, ev.y_root)
        except (tk.TclError, KeyError):
            under = None
        if under is None or getattr(under, "_peek", None) is not self.source:
            self.hide()

    def shown(self):
        return self.win is not None and self.win.winfo_exists() and \
            self.win.state() != "withdrawn"

    def show(self, widget):
        self.timer = None
        if not widget.winfo_exists():
            return
        path = self.source() if callable(self.source) else self.source
        img = self.image(path) if path else None
        if img is None:
            return
        if self.win is None or not self.win.winfo_exists():
            self.win = tk.Toplevel(self.root)
            self.win.withdraw()
            self.win.overrideredirect(True)
            try:
                self.win.attributes("-topmost", True)
            except tk.TclError:
                pass
            self.win.label = tk.Label(self.win, bd=0, highlightthickness=1,
                                      highlightbackground="#808080", bg="#000000")
            self.win.label.pack()
        self.win.label.config(image=img)
        self.win.label.image = img
        self.move(widget.winfo_pointerx(), widget.winfo_pointery())
        self.win.deiconify()
        self.win.lift()

    def move(self, x, y):
        """Right of and centred on the pointer, or left of it where the right
        runs off the screen."""
        img = self.win.label.image
        w, h = img.width() + 2, img.height() + 2
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        left = x + self.GAP
        if left + w > sw and x < sw:
            left = x - self.GAP - w
        top = y - h // 2
        if 0 <= y < sh:
            top = max(0, min(top, sh - h))
        self.win.geometry("+%d+%d" % (left, top))

    def motion(self, ev):
        if self.shown():
            self.move(ev.x_root, ev.y_root)

    def hide(self, _ev=None):
        if self.timer is not None:
            try:
                self.root.after_cancel(self.timer)
            except tk.TclError:
                pass
            self.timer = None
        self.source = None
        if self.win is not None and self.win.winfo_exists():
            self.win.withdraw()


def peek(widget, path):
    """Hovering `widget` shows `path` larger in a Peek. `path` may be a
    callable, read when the preview opens, for a thumbnail whose picture
    changes; widgets given the same `path` object count as one picture."""
    root = widget._root()
    pk = getattr(root, "_studio_peek", None)
    if pk is None:
        pk = root._studio_peek = Peek(root)
    widget._peek = path
    widget.bind("<Enter>", lambda ev: pk.enter(ev, path), add="+")
    widget.bind("<Leave>", pk.leave, add="+")
    widget.bind("<Motion>", pk.motion, add="+")
    widget.bind("<ButtonPress>", pk.hide, add="+")
    widget.bind("<MouseWheel>", pk.hide, add="+")
    widget.bind("<Destroy>", lambda ev: pk.hide() if ev.widget is widget and
                pk.source is path else None, add="+")
    return widget


class CameraAim:
    """The Camera row's two diagrams. From above, the camera is dragged round
    the person: which side of them it sees. From the side, it is dragged up
    and down (how high it is) and in and out (how much of them is in the
    frame). Both snap to studio_imagegen's VIEW_* steps. Not set until first
    touched, since then the model frames the picture itself; `get()` is
    settings["view"]."""
    W, H = 380, 170           # before the display's scale
    SPLIT = 168               # the side view starts here
    TOP = (84, 92, 60)        # the view from above: centre x, y and the orbit's radius
    # The view from the side: the person faces right at FIG_X, head at the top
    # of FIG, feet at the bottom; the camera's row is its height, its
    # distance from the person its shot.
    FIG_X = SPLIT + 30
    FIG = {"top": 48, "eye": 57, "neck": 66, "hip": 106, "knee": 133, "foot": 160}
    ROWS = {"overhead": 12, "high": 32, "eye": 57, "low": 112, "ground": 156}
    REACH = {"face": 42, "head": 66, "waist": 90, "knees": 114, "full": 138, "wide": 162}
    SPAN = {"face": (47, 68), "head": (44, 82), "waist": (42, 110), "knees": (41, 138),
            "full": (40, 164), "wide": (34, 168)}

    def __init__(self, owner, parent, on_change):
        self.o, self.on_change = owner, on_change
        self.view = None
        self.k = owner.px(100) / 100.0
        self.cv = tk.Canvas(parent, width=int(self.W * self.k), height=int(self.H * self.k),
                            highlightthickness=0, bd=0, cursor="hand2")
        owner.skin(self.cv, bg="card")
        self.cv.bind("<Button-1>", self._press)
        self.cv.bind("<B1-Motion>", self._drag)
        self.dragging = None
        owner.host._repaint_on_theme(self.cv, self.draw)
        self.draw()

    def get(self):
        return dict(self.view) if self.view else None

    def set(self, view):
        self.view = ig.clean_view(view)
        self.draw()

    # ---------------------------------------------------------------- input
    def _press(self, ev):
        x = ev.x / self.k
        self.dragging = "top" if x < self.SPLIT else "side"
        self._drag(ev)

    def _drag(self, ev):
        x, y = ev.x / self.k, ev.y / self.k
        v = dict(self.view or ig.VIEW_DEFAULT)
        if self.dragging == "top":
            cx, cy, _ = self.TOP
            if math.hypot(x - cx, y - cy) < 6:
                return
            # 0 is in front (below the person), toward their left is toward +x
            v["turn"] = math.degrees(math.atan2(x - cx, y - cy))
        elif self.dragging == "side":
            d = x - self.FIG_X
            v["shot"] = min(self.REACH, key=lambda s: abs(self.REACH[s] - d))
            v["height"] = min(self.ROWS, key=lambda h: abs(self.ROWS[h] - y))
        else:
            return
        v = ig.clean_view(v)
        if v != self.view:
            self.view = v
            self.draw()
            self.on_change()

    # ------------------------------------------------------------- drawing
    def draw(self):
        C, k, cv = self.o.host.C, self.k, self.cv
        cv.delete("all")
        on = self.view is not None
        v = self.view or ig.VIEW_DEFAULT
        ink, cam = C["muted"], C["accent"] if on else C["faint"]
        font = self.o.host.f_small

        def P(*xy):
            return [n * k for n in xy]

        cv.create_line(*P(self.SPLIT, 8, self.SPLIT, self.H - 8), fill=C["border"])
        cv.create_text(*P(8, 8), text="FROM ABOVE", anchor="nw", fill=C["faint"], font=font)
        cv.create_text(*P(self.SPLIT + 8, 8), text="FROM THE SIDE", anchor="nw",
                       fill=C["faint"], font=font)

        # From above: the orbit, the person (shoulders, head, nose toward the
        # front), the camera on the orbit looking in.
        cx, cy, r = self.TOP
        cv.create_oval(*P(cx - r, cy - r, cx + r, cy + r), outline=C["border"], dash=(2, 3))
        cv.create_text(*P(cx + 12, cy + r + 3), text="front", anchor="w", fill=C["faint"],
                       font=font)
        cv.create_oval(*P(cx - 22, cy - 7, cx + 22, cy + 7), fill=ink, outline=ink)
        cv.create_oval(*P(cx - 27, cy - 4, cx - 19, cy + 4), fill=ink, outline=ink)
        cv.create_oval(*P(cx + 19, cy - 4, cx + 27, cy + 4), fill=ink, outline=ink)
        cv.create_oval(*P(cx - 7, cy - 7, cx + 7, cy + 7), fill=C["card"], outline=ink,
                       width=2 * k)
        cv.create_polygon(*P(cx - 4, cy + 6, cx + 4, cy + 6, cx, cy + 13), fill=ink)
        a = math.radians(v["turn"])
        px, py = cx + r * math.sin(a), cy + r * math.cos(a)
        self._frustum(px, py, cx, cy, 14, cam)
        self._camera(px, py, cx - px, cy - py, cam)

        # From the side: the person in profile, facing right; what is in the
        # frame marked beside their back; the camera where it was put.
        F, fx = self.FIG, self.FIG_X
        top, bot = self.SPAN[v["shot"]]
        cv.create_line(*P(fx - 16, top, fx - 16, bot), fill=cam, width=3 * k)
        cv.create_line(*P(fx - 20, top, fx - 12, top), fill=cam, width=2 * k)
        cv.create_line(*P(fx - 20, bot, fx - 12, bot), fill=cam, width=2 * k)
        cv.create_oval(*P(fx - 8, F["top"], fx + 8, F["top"] + 16), outline=ink,
                       width=2 * k)
        cv.create_line(*P(fx + 7, F["eye"] - 1, fx + 11, F["eye"] + 2, fx + 7, F["eye"] + 4),
                       fill=ink, width=2 * k)
        cv.create_line(*P(fx, F["top"] + 16, fx, F["hip"]), fill=ink, width=2 * k)
        cv.create_line(*P(fx, F["neck"] + 4, fx + 5, F["hip"] - 14, fx + 3, F["hip"] + 4),
                       fill=ink, width=2 * k)
        cv.create_line(*P(fx, F["hip"], fx + 2, F["knee"], fx, F["foot"], fx + 9, F["foot"]),
                       fill=ink, width=2 * k)
        cv.create_line(*P(self.SPLIT + 6, F["foot"] + 1, self.W - 6, F["foot"] + 1),
                       fill=C["border"])
        sx, sy = fx + self.REACH[v["shot"]], self.ROWS[v["height"]]
        self._frustum(sx, sy, fx + 6, (top + bot) / 2, (bot - top) / 2, cam, True)
        self._camera(sx, sy, fx + 6 - sx, (top + bot) / 2 - sy, cam)

    def _frustum(self, x, y, tx, ty, half, colour, upright=False):
        """Two dashed lines from the lens to either edge of what it sees: `half`
        either side of (tx, ty), square to the line of sight, or straight up
        and down when `upright` (the side view frames a stretch of the body)."""
        d = math.hypot(tx - x, ty - y) or 1
        nx, ny = (0, 1) if upright else (-(ty - y) / d, (tx - x) / d)
        for s in (-1, 1):
            self.cv.create_line(x * self.k, y * self.k, (tx + s * nx * half) * self.k,
                                (ty + s * ny * half) * self.k, fill=colour, dash=(3, 3))

    def _camera(self, x, y, dx, dy, colour):
        """A camera at (x, y) whose lens points along (dx, dy)."""
        k, d = self.k, math.hypot(dx, dy) or 1
        ux, uy = dx / d, dy / d                      # forward
        vx, vy = -uy, ux                             # across

        def at(f, s):
            return [(x + ux * f + vx * s) * k, (y + uy * f + vy * s) * k]

        body = at(-11, -6) + at(-11, 6) + at(1, 6) + at(1, -6)
        lens = at(1, -3) + at(1, 3) + at(7, 5) + at(7, -5)
        self.cv.create_polygon(*body, fill=colour, outline=colour)
        self.cv.create_polygon(*lens, fill=colour, outline=colour)


class ImageStudio:
    def __init__(self, host, session):
        self.host, self.s = host, session
        self.studio = ig.Studio(notify=self._notify, make_room=host._images_make_room,
                                vision=lambda: getattr(host, "vision", None))
        self.settings = ig.default_settings()
        self.idents = {}              # identity id -> (BooleanVar, DoubleVar, scale row)
        self.loras = []               # [{"id", "var", "row"}]
        # LoRAs taken off the rows because they do not work with the chosen
        # model (or are off in Add-ons): id -> strength, back when they fit.
        self.parked = {}
        self.refs = {}                # kind -> local path
        self.pose = None              # the drawn stick figure (PoseEditor), or None
        self.planned_size = (1024, 1024)   # the picture's size, as last composed
        self.adv = {}                 # setting -> StringVar
        self.text = {}                # look slot and camera setting -> StringVar
        self.sliders = {}             # body slider keys -> IntVar
        self.item_refs = {}           # item -> picture, from the character
        self.look_section = FORM_LOOKS[0][0]
        self.face_photos = []         # the picked character's face photos (ig.character_faces)
        self.face_name = ""
        self.anatomy = tk.BooleanVar(value=True)
        self.hints = {}               # setting -> Label
        self.rows = {}                # job id -> row widgets
        self.jobs = []                # this session's jobs, newest first
        self.view = "queue"
        self.selected = None          # ("job", Job) | ("record", dict)
        self.keep = []                # PhotoImages Tk must not lose
        self.pending_preview = None
        self.shown_history = HISTORY_PAGE
        self.random_seed = tk.BooleanVar(value=True)
        self.refine = tk.BooleanVar(value=False)
        self.refine_set = False       # the user touched it; the preset no longer decides
        self.faces = tk.BooleanVar(value=False)
        self.faces_set = False        # likewise for the face pass
        self.auto_refine = tk.BooleanVar(value=False)   # the Visual Critic
        self.hand_pass = tk.BooleanVar(value=True)      # the hands redrawn last
        self.head_swap = tk.BooleanVar(value=True)      # the head redrawn before the face swap
        self.glasses_pass = tk.BooleanVar(value=False)  # the glasses redrawn whatever the swap kept
        self.adv_open = False
        self.scene_builder = None     # the Scene Builder window, while it is open
        self.lora_build = None        # the identity LoRA being trained (lt.Build)
        self._build(session.frame)
        for problem in self.studio.lib.problems:
            self.say(problem, "warn")

    def start(self):
        """First view of the tab (`Chat._ensure`): ask the backends how they
        are. Not at construction - a tab is built before it is looked at,
        and building must not reach the network."""
        self.refresh_backends()
        self._prepare_profiles()

    # ================================================================ plumbing
    def _notify(self, job):
        """From any thread: a job changed."""
        self.host.q.put(("images", self.s.event_id, ("job", job)))

    def _post(self, what, arg=None):
        self.host.q.put(("images", self.s.event_id, (what, arg)))

    def handle(self, payload):
        what, arg = payload
        if what == "job":
            self._job_changed(arg)
        elif what == "health":
            self._paint_health()
            self._rebuild_models()
            self._recheck()
        elif what == "said":
            self.say(*arg)
        elif what == "submitted":
            for job in arg:
                if job not in self.jobs:
                    self.jobs.insert(0, job)
            self._show_list("queue")
        elif what == "editor-reload":
            if arg.win.winfo_exists():
                arg.reload()
        elif what == "editor-refresh":
            if arg.win.winfo_exists():
                arg.refresh()
        elif what == "call":
            arg()
        elif what == "import-said":
            dlg, text, role = arg
            if dlg.win.winfo_exists():
                dlg.status(text, role)
        elif what == "import-done":
            dlg, editor, rid, text, role = arg
            if dlg.win.winfo_exists():
                dlg.finished(text, role)
            if editor is not None and editor.win.winfo_exists():
                editor.reload(rid)
            self._rebuild_choices()
            self._recheck()
        elif what == "library":
            self._rebuild_choices()
            self._recheck()

    def px(self, n):
        return self.host._px(n)

    def skin(self, w, **roles):
        return self.host._skin(w, **roles)

    def label(self, parent, text="", role="text", font=None, bg="bg", **kw):
        lbl = tk.Label(parent, text=text, font=font or self.host.f_ui, anchor="w",
                       justify="left", **kw)
        return self.skin(lbl, bg=bg, fg=role)

    def frame(self, parent, bg="bg"):
        return self.skin(tk.Frame(parent, bd=0, highlightthickness=0), bg=bg)

    def cap(self, parent, text, bg="bg"):
        c = self.host._cap(parent, text.upper(), bg=bg)
        c.pack(side="top", fill="x", pady=(self.px(12), self.px(3)))
        return c

    def button(self, parent, text, command, kind="quiet", bg="bg", **kw):
        return self.host._button(parent, text, command, kind=kind, bg=bg, **kw)

    def switch(self, parent, var, command=None, bg="bg"):
        return self.host._switch(parent, var, command=command, bg=bg)

    def say(self, text, role="muted"):
        self.note.config(text=text)
        self.skin(self.note, bg="bg", fg=role)

    def scrolled(self, parent, bg="bg"):
        """A vertically scrolling frame: a canvas holding it, a bar beside it,
        the wheel while the pointer is over it. -> (outer, inner)."""
        outer = self.frame(parent, bg)
        bar = tk.Scrollbar(outer, highlightthickness=0, bd=0, width=11)
        self.skin(bar, bg=bg, troughcolor=bg, activebackground="faint")
        bar.pack(side="right", fill="y")
        canvas = tk.Canvas(outer, highlightthickness=0, bd=0, yscrollcommand=bar.set)
        self.skin(canvas, bg=bg)
        canvas.pack(side="left", fill="both", expand=True)
        bar.config(command=canvas.yview)
        inner = self.frame(canvas, bg)
        item = canvas.create_window(0, 0, window=inner, anchor="nw")
        inner.bind("<Configure>", lambda ev: canvas.config(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda ev: canvas.itemconfig(item, width=ev.width))

        def wheel(ev):
            if canvas.winfo_height() < inner.winfo_height():
                canvas.yview_scroll(int(-ev.delta / 120), "units")
        def leave(ev):
            # Leave also fires going into a child; only a real exit unbinds.
            under = outer.winfo_containing(ev.x_root, ev.y_root)
            if under is None or not str(under).startswith(str(outer)):
                canvas.unbind_all("<MouseWheel>")
        outer.bind("<Enter>", lambda ev: canvas.bind_all("<MouseWheel>", wheel))
        outer.bind("<Leave>", leave)
        inner.canvas = canvas
        return outer, inner

    def choice(self, parent, items, current, on_pick, bg="bg", width=None, images=None,
               kind="option", **kw):
        """A dropdown: a Pill that posts a menu of (value, label) pairs. The
        window has no ttk and wants none (palette roles, Pills everywhere).
        `kw` goes to the Pill: anchor="w" for one as wide as its row."""
        labels = dict(items)
        pill = self.button(parent, labels.get(current, current or "Choose") + "  ▾",
                           lambda: None, kind=kind, bg=bg, **kw)

        def post():
            menu = tk.Menu(pill, tearoff=0)
            self.skin(menu, bg="card", fg="text", activebackground="sel",
                      activeforeground="text")
            for value, text in items:
                if value is None:
                    menu.add_separator()
                else:
                    extra = {"image": images[value], "compound": "left"} if images and value in images else {}
                    menu.add_command(label=text, command=lambda v=value: pick(v), **extra)
            menu.tk_popup(pill.winfo_rootx(), pill.winfo_rooty() + pill.winfo_height())

        def pick(value):
            pill.set(text=labels.get(value, value) + "  ▾")
            on_pick(value)
        pill.command = post
        return pill

    def slider(self, parent, var, lo=0.0, hi=1.5, bg="bg", command=None):
        sc = tk.Scale(parent, variable=var, from_=lo, to=hi, resolution=0.05,
                      orient="horizontal", showvalue=True, bd=0, highlightthickness=0,
                      sliderrelief="flat", sliderlength=self.px(14), width=self.px(8),
                      font=self.host.f_small, command=command)
        return self.skin(sc, bg=bg, fg="muted", troughcolor="border",
                         activebackground="accent")

    # ================================================================ layout
    def _build(self, root):
        head = self.frame(root)
        head.pack(side="top", fill="x", padx=self.px(18), pady=(0, self.px(6)))
        self.manage_pill = self.button(head, "Manage ▾", self._manage_menu)
        self.manage_pill.pack(side="right")
        self.button(head, "App store", lambda: self.open_addons("catalog"),
                    kind="accent").pack(side="right", padx=(0, self.px(6)))
        self.source_buttons = {}
        for source, (name, _domain, _env) in model_sources.SOURCES.items():
            button = self.button(head, name, lambda key=source: self.open_model_source(key))
            button.pack(side="right", padx=(0, self.px(6)))
            self.source_buttons[source] = button
        self.health_row = self.frame(head)
        self.health_row.pack(side="left", fill="x", expand=True)

        self.note = self.label(root, "", "muted")
        self.note.pack(side="top", fill="x", padx=self.px(20), pady=(0, self.px(4)))

        body = self.frame(root)
        body.pack(side="top", fill="both", expand=True, padx=self.px(14),
                  pady=(0, self.px(14)))
        # Fixed before expanding (AGENTS.md: pack order).
        left = self.frame(body)
        left.config(width=self.px(390))
        left.pack_propagate(False)
        left.pack(side="left", fill="y")
        actions = self.frame(left)
        actions.pack(side="bottom", fill="x")
        self.section_bar = self.frame(left)
        self.section_bar.pack(side="top", fill="x", pady=(0, self.px(8)))
        scroll, self.form = self.scrolled(left)
        scroll.pack(side="top", fill="both", expand=True)
        right = self.frame(body)
        right.pack(side="left", fill="both", expand=True, padx=(self.px(14), 0))
        self._build_form(self.form, actions)
        self._build_right(right)

    # -------------------------------------------------------------- the form
    def _build_form(self, f, actions):
        pad = {"padx": (self.px(6), self.px(10))}
        self.sections = {name: self.frame(f) for name in
                         ("Image", "People", "References", "Settings")}
        self.section_pills = {}
        for name in self.sections:
            pill = self.button(self.section_bar, name,
                               lambda n=name: self._show_section(n),
                               font=self.host.f_small, padx=self.px(9))
            pill.pack(side="left", padx=(0, self.px(3)))
            self.section_pills[name] = pill
        f = self.sections["Image"]
        self.cap(f, "Prompt").pack(**pad)
        setup = self.sections["Settings"]
        self.cap(setup, "Preset").pack(**pad)
        self.preset_row = self.frame(setup)
        self.preset_row.pack(side="top", fill="x", **pad)
        self.preset_mix = set()       # LoRA ids the picked saved mix put in the rows
        self.preset_about = self.label(setup, ig.PRESETS["standard"]["about"], "faint",
                                       self.host.f_small, wraplength=self.px(380))
        self._build_presets()

        self.cap(setup, "Model and backend").pack(**pad)
        self.model_row = self.frame(setup)
        self.model_row.pack(side="top", fill="x", **pad)

        # Shot on: the cameras as cards to flip through, above the Scene
        # field. The card showing is the camera - for the form's Generate
        # (its words and frame shape) and the Scene Builder's alike.
        self.cap(f, "Shot on").pack(**pad)
        self.camera_deck = self.frame(f)
        self.camera_deck.pack(side="top", fill="x", pady=(0, self.px(8)), **pad)
        self._build_camera_deck()

        shell = self.frame(f, "card")
        shell.pack(side="top", fill="x", **pad)
        self.scene = tk.Text(shell, height=5, wrap="word", bd=0, highlightthickness=0,
                             font=self.host.f_ui, padx=self.px(8), pady=self.px(6),
                             undo=True)
        self.skin(self.scene, bg="card", fg="text", insertbackground="accent",
                  selectbackground="sel")
        self.scene.pack(fill="x")
        self.scene.bind("<KeyRelease>", lambda ev: self._recheck())
        self.scene.bind("<Control-Return>", lambda ev: (self.generate(), "break")[1])
        # As wide as the scene box above it: the way into the Scene Builder
        # is the scene's own, not a small aside beside a hint.
        srow = self.frame(f)
        srow.pack(side="top", fill="x", pady=(self.px(4), 0), **pad)
        self.button(srow, "Scene Builder…", self.build_scene).pack(
            side="left", fill="x", expand=True, padx=(0, self.px(4)))
        self.button(srow, "Image library…", self.image_library).pack(
            side="left", fill="x", expand=True)
        self.pc_box = pb = self.sections["People"]
        self.cap(pb, "Person").pack(**pad)
        # One dropdown for the person, characters and profiles both
        # (_rebuild_choices fills it), and under it the two ways to edit them.
        self.person_box = self.frame(pb)
        self.person_box.pack(side="top", fill="x", **pad)
        prow = self.frame(pb)
        prow.pack(side="top", fill="x", pady=(self.px(6), 0), **pad)
        self.button(prow, "Editor", lambda: self.edit_characters()).pack(side="left")
        self.button(prow, "Image references", self.edit_identities).pack(side="right")
        for key in ig.SLOTS:
            self.text[key] = tk.StringVar()
        for key in ig.SLIDER_KEYS:
            self.sliders[key] = tk.IntVar(value=0)
        self.look_tabs = self.frame(pb)
        self.look_tabs.pack(side="top", fill="x", pady=(self.px(10), 0), **pad)
        self.look_box = self.frame(pb)
        self.look_box.pack(side="top", fill="x", **pad)
        self._show_looks(self.look_section)
        b = tk.Checkbutton(pb, text="Anatomy constants", variable=self.anatomy, anchor="w",
                           font=self.host.f_ui, bd=0, highlightthickness=0,
                           command=self._recheck)
        self.skin(b, bg="bg", fg="text", activebackground="bg", selectcolor="card",
                  activeforeground="text")
        b.pack(side="top", fill="x", pady=(self.px(8), 0), **pad)

        self.cap(pb, "Camera").pack(**pad)
        self.aim = CameraAim(self, pb, self._aimed)
        self.aim.cv.pack(side="top", anchor="w", **pad)
        arow = self.frame(pb)
        arow.pack(side="top", fill="x", pady=(self.px(4), 0), **pad)
        self.aim_note = self.label(arow, "", "muted", self.host.f_small)
        self.aim_note.pack(side="left", fill="x", expand=True)
        self.aim_clear = self.button(arow, "Clear", lambda: self._aimed(None), kind="ghost")
        self._aimed(recheck=False)
        self.text["camera"] = tk.StringVar()
        e = self.host._entry(pb, self.text["camera"])
        e.master.pack(side="top", fill="x", **pad)
        e.bind("<KeyRelease>", lambda ev: self._recheck())
        self.label(pb, "Lens, light, film: \u201c85mm, shallow depth of field, "
                   "golden hour\u201d.", "faint", self.host.f_small,
                   wraplength=self.px(380)).pack(side="top", fill="x", **pad)

        self.cap(f, "Style").pack(**pad)
        self.style_box = self.frame(f)
        self.style_box.pack(side="top", fill="x", **pad)

        self.ref_box = self.frame(self.sections["References"])
        self.ref_box.pack(side="top", fill="x", **pad)
        self._build_refs()

        arow = self.frame(setup)
        arow.pack(side="top", fill="x", pady=(self.px(12), 0), **pad)
        self.adv_pill = self.button(arow, "Advanced  ▸",
                                    self._toggle_advanced, kind="ghost")
        self.adv_pill.pack(side="left")
        self.adv_box = self.frame(setup)
        self.cap(self.adv_box, "LoRAs").pack(**pad)
        self.lora_box = self.frame(self.adv_box)
        self.lora_box.pack(side="top", fill="x", **pad)
        self.parked_note = self.label(self.adv_box, "", "faint", self.host.f_small,
                                      wraplength=self.px(380))
        lrow = self.frame(self.adv_box)
        lrow.pack(side="top", fill="x", pady=(self.px(4), 0), **pad)
        self.lora_row = lrow
        self.add_lora_pill = self.button(lrow, "Add LoRA  ▾", self._post_lora_menu)
        self.add_lora_pill.pack(side="left")
        self.button(lrow, "Add-ons…", self.open_addons, kind="ghost").pack(
            side="left", padx=(self.px(6), 0))
        self.button(lrow, "Save as preset…", self.save_preset, kind="ghost").pack(
            side="left", padx=(self.px(6), 0))

        self._build_advanced(self.adv_box)

        self.warn = self.label(actions, "", "warn", self.host.f_small, wraplength=self.px(350))
        self.warn.pack(side="top", fill="x", pady=(self.px(10), 0), **pad)
        self.readiness_pill = self.button(actions, "Resolve readiness issues ▾",
                                          self._readiness_menu, kind="ghost")
        self.readiness_pill.pack(side="top", anchor="w", **pad)
        # Where the job will go, and why, before Generate (plan_route).
        self.route_note = self.label(setup, "", "muted", self.host.f_small,
                                     wraplength=self.px(380))
        self.route_note.pack(side="top", fill="x", pady=(self.px(6), 0), **pad)
        grow = self.frame(actions)
        grow.pack(side="top", fill="x", pady=(self.px(10), self.px(18)), **pad)
        self.go = self.button(grow, "Generate", self.generate, kind="accent")
        self.go.pack(side="top", fill="x")
        # The Visual Critic (studio_critic): the vision model checks the
        # picture and the faults it finds are redrawn, up to three passes.
        b = tk.Checkbutton(setup, text="Automatic refinement", variable=self.auto_refine,
                           anchor="w", font=self.host.f_small, bd=0, highlightthickness=0,
                           wraplength=self.px(380), justify="left")
        self.skin(b, bg="bg", fg="muted", activebackground="bg", selectcolor="card",
                  activeforeground="text")
        b.pack(side="top", fill="x", pady=(0, self.px(4)), **pad)
        # The hands pass (Studio._finish_passes): every hand SAM3 finds is
        # redrawn at the end of Generate, before the glasses.
        b = tk.Checkbutton(setup, text="Natural hands pass", variable=self.hand_pass,
                           anchor="w", font=self.host.f_small, bd=0, highlightthickness=0,
                           wraplength=self.px(380), justify="left")
        self.skin(b, bg="bg", fg="muted", activebackground="bg", selectcolor="card",
                  activeforeground="text")
        b.pack(side="top", fill="x", pady=(0, self.px(4)), **pad)
        # The head swap (Studio._head_swap): with a face profile, FLUX.2 Klein
        # redraws the whole head from their photo before FaceFusion swaps the face.
        b = tk.Checkbutton(setup, text="Head swap before the face swap",
                           variable=self.head_swap,
                           anchor="w", font=self.host.f_small, bd=0, highlightthickness=0,
                           wraplength=self.px(380), justify="left")
        self.skin(b, bg="bg", fg="muted", activebackground="bg", selectcolor="card",
                  activeforeground="text")
        b.pack(side="top", fill="x", pady=(0, self.px(4)), **pad)
        # The glasses pass (ig.glasses_pass): the face swap now goes behind
        # the frames, so they are left as drawn unless this is ticked.
        b = tk.Checkbutton(setup, text="Redraw glasses after the face swap",
                           variable=self.glasses_pass,
                           anchor="w", font=self.host.f_small, bd=0, highlightthickness=0,
                           wraplength=self.px(380), justify="left")
        self.skin(b, bg="bg", fg="muted", activebackground="bg", selectcolor="card",
                  activeforeground="text")
        b.pack(side="top", fill="x", pady=(0, self.px(12)), **pad)
        self._rebuild_choices()
        self._show_section("Image")

    def _show_section(self, name):
        for key, box in self.sections.items():
            box.pack_forget()
            pill = self.section_pills[key]
            pill.roles = self.host.PILL_ROLES["accent" if key == name else "quiet"]
            pill.paint(self.host.C)
        self.sections[name].pack(side="top", fill="x")
        self.form.canvas.yview_moveto(0)
        self.section = name

    def open_model_source(self, source):
        dialog = ModelSourceSettings(self, source)
        dialog.refresh_discoveries()
        return dialog

    def _readiness_menu(self):
        menu = self.host._menu()
        for label, action in (("Reference photos and face swaps…", self.edit_identities),
                              ("Models and required files…", self.edit_models),
                              ("Backends…", self.edit_backends),
                              ("Check connections", self.refresh_backends)):
            menu.add_command(label=label, command=action)
        pill = self.readiness_pill
        menu.tk_popup(pill.winfo_rootx(), pill.winfo_rooty() + pill.winfo_height())

    def _manage_menu(self):
        menu = self.host._menu()
        menu.add_command(label="Models…", command=self.edit_models)
        menu.add_command(label="Backends…", command=self.edit_backends)
        menu.add_separator()
        menu.add_command(label="Check connections", command=self.refresh_backends)
        p = self.manage_pill
        menu.tk_popup(p.winfo_rootx(), p.winfo_rooty() + p.winfo_height())

    def _rebuild_choices(self):
        """Everything drawn from the library: models, backends, people, styles,
        LoRA rows. Called again after any editor saves."""
        lib = self.studio.lib
        self._rebuild_models()
        self._build_presets()
        chars = lib.all("characters")
        if self.settings["character"] not in {c["id"] for c in chars}:
            self.settings["character"] = ""
        for w in self.person_box.winfo_children():
            w.destroy()
        old = {k: (b.get(), s.get()) for k, (b, s, _) in self.idents.items()}
        self.idents = {}
        portraits = {}
        for ident in lib.all("identities"):
            on, st = old.get(ident["id"], (False, ident["strength"]))
            bvar, svar = tk.BooleanVar(value=on), tk.DoubleVar(value=st)
            sc = self.frame(self.person_box)  # retains legacy strength settings, no control needed
            self.idents[ident["id"]] = (bvar, svar, sc)
            path = ident.get("avatar") or next(iter(ident["references"]), "")
            img = photo(path, self.px(40), profile=True) if path else None
            if img:
                portraits[ident["id"]] = img
        self.identity_photos = portraits
        # Characters first (a look and a face), then the profiles not already
        # wrapped by a character (a face alone); a character shows its
        # profile's portrait.
        people = [("", "No one")] + [("c:" + c["id"], c["name"]) for c in chars]
        wrapped = {c.get("identity") for c in chars}
        idents = [i for i in lib.all("identities") if i["id"] not in wrapped]
        if chars and idents:
            people.append((None, ""))
        people += [("i:" + i["id"], i["name"]) for i in idents]
        images = {"i:" + iid: img for iid, img in portraits.items()}
        images.update({"c:" + c["id"]: portraits[c["identity"]] for c in chars
                       if c.get("identity") in portraits})
        self.person_pill = self.choice(self.person_box, people, "", self._pick_from_people,
                                       images=images, kind="quiet", anchor="w",
                                       font=self.host.f_title, pady=self.px(9))
        self.person_pill.pack(side="top", fill="x")
        head_row = self.frame(self.person_box)
        head_row.pack(side="top", fill="x")
        self.button(head_row, "Choose head photo…", self.choose_head_photo).pack(side="left")
        self.identity_note = self.label(self.person_box, "", "muted", self.host.f_small)
        self.identity_note.pack(side="top", anchor="w", pady=(self.px(3), 0))
        self._show_identity()

        for w in self.style_box.winfo_children():
            w.destroy()
        styles = lib.all("styles")
        if self.settings["style"] not in {st["id"] for st in styles}:
            self.settings["style"] = styles[0]["id"] if styles else ""
        self._build_style_tiles(styles)
        self.button(self.style_box, "Styles…", self.edit_styles, kind="ghost").pack(
            side="top", anchor="w", pady=(self.px(4), 0))
        self.style_strength = tk.DoubleVar(value=0.6)
        self.style_scale_row = self.frame(self.style_box)
        self.style_scale_row.pack(side="top", fill="x")
        self._set_style(self.settings["style"], recheck=False)

        kept = [(r["id"], r["var"].get()) for r in self.loras]
        for r in self.loras:
            r["row"].destroy()
        self.loras = []
        for lid, strength in kept:
            if lib.get("loras", lid):
                self._add_lora(lid, strength, recheck=False)
        self._refit_loras()

    def _rebuild_models(self):
        """The model and backend choosers. Each model says which online
        machines have it, once the backends have answered - a model nobody
        has installed should not look like a choice that will work."""
        lib = self.studio.lib
        for w in self.model_row.winfo_children():
            w.destroy()
        models = []
        for m in lib.all("models"):
            text = "%s  (%s)" % (m["label"], ig.FAMILIES.get(m["family"], m["family"] or "?"))
            ready = self.studio.readiness(m)
            if all(state == "unchecked" for state, _ in ready.values()):
                text += "  — not checked yet"
            else:
                where = [self.studio.backend(bid)["name"] for bid, (state, _) in ready.items()
                         if state == "ready"]
                text += "  — " + ("ready on " + ", ".join(where) if where else
                                  "not ready anywhere: Models… says what is missing")
            models.append((m["id"], text))
        if models and self.settings["model"] not in dict(models):
            self.settings["model"] = models[0][0]
        self.choice(self.model_row, models, self.settings["model"],
                    self._set_model).pack(side="top", anchor="w")
        backs = [("auto", "Auto (route by preset)")] + [(b["id"], b["name"])
                                                        for b in lib.all("backends")]
        if self.settings["backend"] not in dict(backs):
            self.settings["backend"] = "auto"
        self.choice(self.model_row, backs, self.settings["backend"],
                    lambda v: self._set("backend", v)).pack(side="top", anchor="w",
                                                           pady=(self.px(4), 0))

    def _set(self, key, value):
        self.settings[key] = value
        self._recheck()

    def _set_model(self, value):
        """Another model: the LoRA rows and saved mixes that do not work with
        it are set aside, and those set aside that do come back."""
        self.settings["model"] = value
        self._refit_loras()
        self._build_presets()
        self._show_identity()
        self._recheck()

    def _build_presets(self):
        """The Preset row: the built-ins, then the saved LoRA mixes, and a
        Delete beside a saved one. Built again when a mix is saved or
        deleted."""
        for w in self.preset_row.winfo_children():
            w.destroy()
        lib = self.studio.lib
        # A saved mix none of whose LoRAs work with the model is not offered.
        model = lib.get("models", self.settings["model"])
        mixes = [m for m in lib.all("presets") if not m["loras"] or any(
            ig.lora_fits(r, model) is not False for r in
            filter(None, (lib.get("loras", x["id"]) for x in m["loras"])))]
        items = [(k, ig.PRESETS[k]["label"]) for k in ig.PRESET_ORDER]
        if mixes:
            items += [(None, "")] + [(m["id"], m["name"]) for m in mixes]
        if self.settings["preset"] not in dict(items):
            self.settings["preset"] = "standard"
        self.preset_pill = self.choice(self.preset_row, items, self.settings["preset"],
                                       self._set_preset)
        self.preset_pill.pack(side="left")
        info = ig.preset_info(lib, self.settings["preset"])
        if info["custom"]:
            self.button(self.preset_row, "Delete", self.delete_preset, kind="ghost").pack(
                side="left", padx=(self.px(6), 0))
        self.preset_about.config(text=info["about"])

    def _set_preset(self, key):
        """A preset picked. A saved mix replaces the LoRA rows the last mix
        put there with its own (a row added by hand stays) and opens
        Advanced so they show; a built-in takes the last mix's rows away."""
        self.settings["preset"] = key
        info = ig.preset_info(self.studio.lib, key)
        for r in [r for r in self.loras if r["id"] in self.preset_mix]:
            r["row"].destroy()
            self.loras.remove(r)
        for lid in self.preset_mix:
            self.parked.pop(lid, None)
        self.preset_mix = set()
        for sel in info["loras"]:
            if self.studio.lib.get("loras", sel["id"]):
                for r in [r for r in self.loras if r["id"] == sel["id"]]:
                    r["row"].destroy()
                    self.loras.remove(r)
                self.parked.pop(sel["id"], None)
                self._add_lora(sel["id"], sel["strength"], recheck=False)
                self.preset_mix.add(sel["id"])
        self._show_parked()
        if info["loras"] and not self.adv_open:
            self._toggle_advanced()
        self._build_presets()
        if not self.refine_set:
            self.refine.set(bool(info["values"].get("refine")))
        if not self.faces_set:
            self.faces.set(bool(info["values"].get("face_detail")))
        self._recheck()

    def save_preset(self):
        """The LoRA rows, with their strengths, saved as a preset under a
        name; the same name replaces that preset. It rides on the built-in
        preset now chosen (or the chosen mix's)."""
        rows = [{"id": r["id"], "strength": round(r["var"].get(), 2)} for r in self.loras]
        if not rows:
            self.say("Add a LoRA or two first (Add LoRA), then save them as a preset.",
                     "warn")
            return None
        lib = self.studio.lib
        info = ig.preset_info(lib, self.settings["preset"])

        def done(name):
            old = next((m for m in lib.all("presets")
                        if m["name"].lower() == name.lower()), None)
            rec = {"id": old["id"] if old else "", "name": name, "base": info["base"],
                   "loras": rows}
            try:
                lib.save("presets", [m for m in lib.all("presets") if m is not old] + [rec])
            except OSError as e:
                self.say("Could not save the preset: %s" % e, "err")
                return
            saved = next(m for m in lib.all("presets") if m["name"] == name)
            self.settings["preset"] = saved["id"]
            self.preset_mix = {r["id"] for r in rows}
            self._build_presets()
            self.say("Saved preset %s. Pick it under Preset to load these LoRAs." % name,
                     "muted")
        return self._ask_name("Save LoRAs as a preset", "Preset name",
                              info["label"] if info["custom"] else "", done)

    def delete_preset(self):
        lib = self.studio.lib
        rec = lib.get("presets", self.settings["preset"])
        if rec is None:
            return
        try:
            lib.save("presets", [m for m in lib.all("presets") if m["id"] != rec["id"]])
        except OSError as e:
            self.say("Could not delete the preset: %s" % e, "err")
            return
        self.preset_mix = set()         # its LoRAs stay in the rows until taken off
        self.settings["preset"] = rec["base"]
        self._build_presets()
        self._recheck()
        self.say("Deleted preset %s; its LoRAs are still in the rows." % rec["name"],
                 "muted")

    def _ask_name(self, title, label, value, then, parent=None, ok_text="Save"):
        """A small window asking for a name; `then(name)` on Save or Return."""
        top = tk.Toplevel(parent or self.host)
        top.title(title)
        top.transient(parent or self.host)
        self.skin(top, bg="bg")
        self.label(top, label, "text").pack(side="top", anchor="w", padx=self.px(12),
                                            pady=(self.px(10), self.px(4)))
        var = tk.StringVar(value=value)
        e = self.host._entry(top, var)
        e.master.pack(side="top", fill="x", padx=self.px(12))
        row = self.frame(top)
        row.pack(side="top", fill="x", padx=self.px(12), pady=self.px(10))

        def ok(_ev=None):
            name = var.get().strip()
            if name:
                top.destroy()
                then(name)
        self.button(row, ok_text, ok, kind="accent").pack(side="right")
        self.button(row, "Cancel", top.destroy, kind="ghost").pack(
            side="right", padx=(0, self.px(6)))
        e.bind("<Return>", ok)
        e.bind("<Escape>", lambda _ev: top.destroy())
        e.focus_set()
        top.var, top.ok = var, ok       # for the tests
        return top

    def from_link(self, parent, what, owner, then, status):
        """A picture of `what` from a link instead of a file: ask for the link
        (the clipboard's, when it holds one), download it off the UI thread
        into references/<owner> (`Library.keep_link`), then `then(path)` on
        the UI thread while `parent` is still open. `status(text, role)` says
        it is downloading, and why when nothing came."""
        def got(url):
            status("Downloading the picture of the %s%s" % (what, ELLIPSIS))

            def work():
                try:
                    path = self.studio.lib.keep_link(url, owner)
                except (ig.LinkError, OSError) as e:
                    msg = "No picture from that link: %s" % e
                    self._post("call", lambda: parent.winfo_exists() and status(msg, "err"))
                    return

                def done():
                    if parent.winfo_exists():
                        status("Downloaded the picture of the %s." % what, "ok")
                        then(path)
                self._post("call", done)
            self.host._spawn(self.s.event_id, work)
        try:
            clip = parent.clipboard_get().strip()
        except tk.TclError:
            clip = ""
        looks = ("\n" not in clip and len(clip) < 8192 and
                 re.match(r"(?i)(https?://|data:image/)\S+$", clip))
        return self._ask_name("A picture from a link", "Link to a picture of the %s "
                              "(or to a page that shows it)" % what,
                              clip if looks else "", got, parent=parent, ok_text="Download")

    def _build_style_tiles(self, styles):
        """The styles as a grid of pictures: each one the same photo in that
        style (`ig.style_example`), its name under it, a click to choose it.
        A style with no picture shows its name on a blank tile."""
        grid = self.frame(self.style_box)
        grid.pack(side="top", fill="x")
        self.style_tiles, self.style_photos = {}, []
        side = self.px(STYLE_TILE)
        for n, st in enumerate(styles):
            tile = tk.Frame(grid, bd=0, highlightthickness=self.px(2), cursor="hand2")
            self.skin(tile, bg="bg", highlightbackground="border")
            tile.grid(row=n // 3, column=n % 3, padx=(0, self.px(6)), pady=(0, self.px(6)),
                      sticky="n")
            path = ig.style_example(st)
            img = photo_at(path, side, grid) if path else None
            if img is not None:
                self.style_photos.append(img)
                pic = peek(tk.Label(tile, image=img, bd=0), path)
            else:
                pic = self.frame(tile, "card")
                pic.config(width=side, height=side)
                pic.pack_propagate(False)
                self.label(pic, st["name"], "faint", self.host.f_small, bg="card",
                           wraplength=side - self.px(8)).pack(expand=True)
            pic.pack(side="top")
            name = self.label(tile, st["name"], "text", self.host.f_small, wraplength=side)
            name.config(anchor="center", justify="center")
            name.pack(side="top", fill="x", pady=(self.px(2), self.px(2)))
            for w in (tile, pic, name, *pic.winfo_children()):
                w.bind("<Button-1>", lambda ev, sid=st["id"]: self._set_style(sid))
            self.style_tiles[st["id"]] = tile
        if not styles:
            self.label(grid, "No styles yet.", "faint").pack(side="left")

    def _set_style(self, sid, recheck=True):
        self.settings["style"] = sid
        for w in self.style_scale_row.winfo_children():
            w.destroy()
        st = self.studio.lib.get("styles", sid)
        for tid, tile in self.style_tiles.items():
            self.skin(tile, bg="bg", highlightbackground="accent" if tid == sid else "border",
                      highlightcolor="accent" if tid == sid else "border")
        if st and st["lora"]:
            self.style_strength.set(st["strength"])
            self.label(self.style_scale_row, "LoRA strength", "faint",
                       self.host.f_small).pack(side="left")
            self.slider(self.style_scale_row, self.style_strength, 0.0, 1.5,
                        command=lambda _v: self._recheck()).pack(
                side="left", fill="x", expand=True, padx=(self.px(8), 0))
        if recheck:
            self._recheck()

    def _toggle_ident(self, iid):
        self._show_identity()
        self._recheck()

    def choose_head_photo(self):
        """Choose the source frame, then mark the head that must stay put."""
        if getattr(self, "head_photo_busy", False):
            return
        path = filedialog.askopenfilename(parent=self.host, title="Choose a clear photo with one face",
                   filetypes=[("Pictures", "*.png *.jpg *.jpeg *.webp *.gif *.bmp")])
        if path:
            self._import_head_photo(path)

    def _import_head_photo(self, path):
        self.head_photo_busy = True
        self.say("Importing head photo…", "muted")
        def work():
            result = self.studio.lib.import_identity_photos([path], "head-photos")
            source = next(iter(result["added"]), "")
            if source and not source.lower().endswith(".png"):
                try:
                    size = ig.file_size_of(source)
                    dest = source + ".png"
                    if not size or not catalog.to_png([(source, dest)], side=max(size)):
                        raise ValueError("Could not open this format. Save the photo as PNG and select it again.")
                    source = dest
                except (OSError, ValueError) as e:
                    result["errors"].append(str(e))
                    source = ""
            def done():
                self.head_photo_busy = False
                if not source:
                    return self.say("Could not import head photo: " + "; ".join(result["errors"]), "err")
                FixWindow(self, source, self.collect(), around_head=True)
            self._post("call", done)
        self.host._spawn(self.s.event_id, work)

    def _pick_from_people(self, key):
        """The person dropdown: "c:<id>" a character (its look and its
        face), "i:<id>" a profile alone, "" no one."""
        if key.startswith("c:"):
            return self._set_character(key[2:])
        self.settings["character"] = ""
        # A profile alone: its own reference photos are the face; no one, none.
        ident = self.studio.lib.get("identities", key[2:]) if key else None
        self.face_photos = ig.character_faces({"identity": key[2:]}, self.studio.lib) if ident else []
        self.face_name = ident["name"] if ident and self.face_photos else ""
        self._select_identity(key[2:])

    def _character(self):
        cid = self.settings["character"]
        return self.studio.lib.get("characters", cid) if cid else None

    def _face_of(self, rec):
        """A character's profile, if it still exists."""
        return rec["identity"] if rec and rec.get("identity") in self.idents else ""

    def _select_identity(self, iid):
        # One dropdown shows one person: another face is not the character.
        if self._character() is not None and self._face_of(self._character()) != iid:
            self.settings["character"] = ""
        for key, (bv, _, _) in self.idents.items():
            bv.set(key == iid)
        self._show_identity()
        self._recheck()

    def _show_identity(self):
        chosen = [self.studio.lib.get("identities", iid) for iid, (bv, _, _) in self.idents.items()
                  if bv.get()]
        rec = self._character()
        names = rec["name"] if rec else ", ".join(i["name"] for i in chosen)
        self.person_pill.set(text=(names or "No one") + "  ▾")
        count = sum(len(set(i["references"])) for i in chosen)
        model = self.studio.lib.get("models", self.settings["model"])
        action = ("WithAnyone uses the first photo" if model and model.get("workflow") == "withanyone"
                  else "face applied automatically")
        if model and model.get("workflow") == "withanyone" and any(i.get("pool_photos") for i in chosen):
            action = "WithAnyone photo pooling enabled (experimental)"
        self.identity_note.config(text=("%d reference photos · %s" % (count, action)
                                       if count else "Add photos in Image references." if chosen
                                       else "Choose a person for this picture."))

    def _prepare_profiles(self):
        profiles = self.studio.lib.all("identities")
        if not ff.available():
            return
        def work():
            if ff.prepare_previews(profiles):
                self._post("call", self._rebuild_choices)
        self.host._spawn(self.s.event_id, work)

    # ------------------------------------------------------------- the look
    def _set_character(self, cid):
        """Put a character's look on the form: every slot and slider it keeps
        (blank where it has none, so the last one's beard does not stay),
        its item pictures, and its identity ticked (or none, when it has
        none). The expression is the picture's, and stays."""
        self.settings["character"] = cid
        rec = self._character()
        if rec is not None:
            for key in ig.CHARACTER_KEYS:
                if key in self.sliders:
                    self.sliders[key].set(int(rec["looks"].get(key, 0)))
                else:
                    self.text[key].set(rec["looks"].get(key, ""))
            self.item_refs = dict(rec["item_refs"])
            self._select_identity(self._face_of(rec))
            # Their face photos go with the picture: the identity builder's
            # reference photos (ig.character_faces). Nothing shows on the form.
            self.face_photos = ig.character_faces(rec, self.studio.lib)
            self.face_name = rec["name"]
            self._show_looks(self.look_section)
        else:
            self.face_photos, self.face_name = [], ""
        self._show_identity()
        self._recheck()

    def look_rows(self, parent, slots, vars_, changed, chips=False, bg="bg"):
        """Rows for look slots: a label, the field, and the picks - a menu on
        the ▾ beside the field, or (`chips`) a grid of them under it, the
        chosen ones lit. -> a function that relights the chips."""
        lights = []
        for key, label, _, picks, many in slots:
            row = self.frame(parent, bg)
            row.pack(side="top", fill="x", pady=(self.px(2), 0))
            self.label(row, label, "muted", bg=bg, width=12).pack(side="left", anchor="n")
            var = vars_[key]
            box = self.frame(row, bg)
            box.pack(side="left", fill="x", expand=True)
            line = self.frame(box, bg)
            line.pack(side="top", fill="x")

            def pick(p, k=key, m=many):
                vars_[k].set(ig.toggle(vars_[k].get(), p, m))
                changed()
            if not chips:
                menu_pill = self.button(line, "▾", lambda: None, kind="ghost", bg=bg)
                menu_pill.pack(side="right", padx=(self.px(4), 0))
                menu_pill.command = (lambda pl=menu_pill, ps=picks, k=key, m=many, f=pick:
                                     self._post_picks(pl, ps, vars_[k], m, f))
            entry = self.host._entry(line, var, bg=bg)
            entry.master.pack(side="left", fill="x", expand=True)
            entry.bind("<KeyRelease>", lambda ev: changed())
            if not chips and key != "expression":
                continue
            # The form shows expressions as faces alone, big enough to read;
            # the creator names every pick. Rows of a fixed count, packed
            # left, so a long pick does not widen a whole column.
            faces = not chips
            cols = 6 if faces else 3
            font = self.emoji_font() if faces else self.host.f_small
            pills, line = [], None
            for i, p in enumerate(picks):
                if i % cols == 0:
                    line = self.frame(box, bg)
                    line.pack(side="top", fill="x", pady=(self.px(3), 0))
                text = ig.EMOJI.get(p, p) if faces else ig.pick_label(p)
                pl = self.button(line, text, lambda p=p, f=pick: f(p), bg=bg, font=font,
                                 padx=self.px(6 if faces else 8), pady=self.px(2))
                pl.pack(side="left", padx=(0, self.px(3)))
                pills.append((p, pl))
            lights.append((var, many, pills))

        def relight():
            for var, many, pills in lights:
                now = [x.lower() for x in (ig.split_many(var.get()) if many
                                           else [var.get().strip()])]
                for p, pl in pills:
                    kind = "accent" if p.lower() in now else "quiet"
                    if pl.roles is not self.host.PILL_ROLES[kind]:
                        pl.roles = self.host.PILL_ROLES[kind]
                        if pl.C:
                            pl.paint(pl.C)
        relight()
        return relight

    def emoji_font(self):
        """The faces' font: Windows draws emoji from Segoe UI Emoji, and Tk
        falls back to whatever has the glyph elsewhere."""
        if getattr(self, "_emoji_font", None) is None:
            import tkinter.font as tkfont
            self._emoji_font = tkfont.Font(root=self.host, family="Segoe UI Emoji", size=15)
        return self._emoji_font

    def _post_picks(self, pill, picks, var, many, pick):
        menu = tk.Menu(pill, tearoff=0)
        self.skin(menu, bg="card", fg="text", activebackground="sel", activeforeground="text")
        now = [x.lower() for x in (ig.split_many(var.get()) if many else [var.get().strip()])]
        for p in picks:
            menu.add_command(label=("✓ " if p.lower() in now else "    ") + ig.pick_label(p),
                             command=lambda p=p: pick(p))
        menu.tk_popup(pill.winfo_rootx(), pill.winfo_rooty() + pill.winfo_height())

    def slider_rows(self, parent, vars_, changed, bg="bg"):
        """Body proportions as the game's sliders: -3..3, a word
        beside each, nothing said at the middle."""
        for key, label, _ in ig.SLIDERS:
            row = self.frame(parent, bg)
            row.pack(side="top", fill="x", pady=(self.px(2), 0))
            self.label(row, label, "muted", bg=bg, width=12).pack(side="left")
            word = self.label(row, "", "faint", self.host.f_small, bg=bg, width=14)
            word.pack(side="right")

            def moved(_v=None, k=key, w=word):
                w.config(text=ig.slider_word(k, vars_[k].get()) or "average")
                changed()
            sc = tk.Scale(row, variable=vars_[key], from_=-ig.SLIDER_SPAN,
                          to=ig.SLIDER_SPAN, resolution=1, orient="horizontal",
                          showvalue=False, bd=0, highlightthickness=0, sliderrelief="flat",
                          sliderlength=self.px(14), width=self.px(8), command=moved)
            self.skin(sc, bg=bg, fg="muted", troughcolor="border", activebackground="accent")
            sc.pack(side="left", fill="x", expand=True, padx=(self.px(6), self.px(6)))
            word.config(text=ig.slider_word(key, vars_[key].get()) or "average")

    def _show_looks(self, section):
        """One section of the look on the form at a time, as the creator's
        tabs are - the ones in FORM_LOOKS."""
        if section not in dict(FORM_LOOKS):
            section = FORM_LOOKS[0][0]
        self.look_section = section
        for box in (self.look_tabs, self.look_box):
            for w in box.winfo_children():
                w.destroy()
        for i, (name, _) in enumerate(FORM_LOOKS):
            self.button(self.look_tabs, name, lambda n=name: self._show_looks(n),
                        kind="accent" if name == section else "quiet",
                        font=self.host.f_small, padx=self.px(6), pady=self.px(2)).grid(
                row=0, column=i, sticky="ew", padx=(0, self.px(3)),
                pady=(self.px(3), self.px(2)))
        for col in range(len(FORM_LOOKS)):
            self.look_tabs.columnconfigure(col, weight=1)
        relight = [None]

        def changed():
            if relight[0]:
                relight[0]()
            self._recheck()
        if section == ig.SLIDER_SECTION:
            self.slider_rows(self.look_box, self.sliders, changed)
        relight[0] = self.look_rows(self.look_box, dict(ig.LOOKS)[section], self.text,
                                    changed)
        if section in ("Clothes", "Accessories", "Hair"):
            keys = [sl[0] for sl in dict(ig.LOOKS)[section]]
            items = ([ig.HAIR_ITEM] if section == "Hair" else
                     ig.items_worn({k: self.text[k].get() for k in keys}))
            self.item_rows(self.look_box, items, self.item_refs, self._choose_item,
                           self._set_item, link=self._link_item)

    def item_rows(self, p, items, refs, choose, drop, title="ITEM PICTURES",
                  empty="Choose an item above to give it a picture.", link=None):
        """A row per item worn - its picture, name and file, Picture…, Link…
        and ×: what the glasses or the necklace actually look like. The form's
        look tabs and the creator's both show these, and the creator's Tags
        tab every one it has; Generate draws from them."""
        o, host = self, self.host
        o.label(p, title, "faint", host.f_small).pack(
            side="top", fill="x", pady=(o.px(12), o.px(2)))
        if not items:
            o.label(p, empty, "faint", host.f_small).pack(side="top", fill="x")
            return
        for item in items:
            row = o.frame(p)
            row.pack(side="top", fill="x", pady=(0, o.px(3)))
            path = refs.get(item)
            img = photo(path, o.px(40)) if path and os.path.isfile(path) else None
            if img is not None:
                o.keep.append(img)
                peek(tk.Label(row, image=img, bd=0), path).pack(side="left",
                                                                padx=(0, o.px(6)))
            o.label(row, item, "text", width=18).pack(side="left")
            o.label(row, os.path.basename(path) if path else "no picture", "faint",
                    host.f_small).pack(side="left", fill="x", expand=True)
            if path:
                o.button(row, "×", lambda i=item: drop(i, None), kind="ghost").pack(
                    side="right")
            if link is not None:
                o.button(row, "Link…", lambda i=item: link(i)).pack(
                    side="right", padx=(o.px(4), 0))
            o.button(row, "Picture…", lambda i=item: choose(i)).pack(
                side="right", padx=(o.px(4), 0))

    def _link_item(self, item):
        rec = self.studio.lib.get("characters", self.settings["character"])
        return self.from_link(self.host, item, (rec["name"] if rec else "form") + " items",
                              lambda path: self._set_item(item, path), self.say)

    def _choose_item(self, item):
        path = filedialog.askopenfilename(parent=self.host, title="A picture of the " + item,
                                          filetypes=[("Pictures", "*.png *.jpg *.jpeg *.webp"),
                                                     ("All files", "*.*")])
        if path:
            self._set_item(item, path)

    def _set_item(self, item, path):
        """A picture for one item, on the form: copied under references/ as
        the creator's are, so the history record's path stays good. Save as…
        keeps it in a character."""
        if path:
            rec = self.studio.lib.get("characters", self.settings["character"])
            try:
                path = self.studio.lib.keep_reference(
                    path, (rec["name"] if rec else "form") + " items")
            except OSError as e:
                return self.say("Could not copy %s: %s" % (path, e), "err")
            self.item_refs[item] = path
        else:
            self.item_refs.pop(item, None)
        self._show_looks(self.look_section)
        self._recheck()

    def collect_looks(self):
        out = {k: v.get().strip() for k, v in self.text.items() if k in ig.SLOTS}
        out.update({k: int(v.get()) for k, v in self.sliders.items()})
        return out

    # ------------------------------------------------------------ references
    def _build_refs(self):
        self.ref_labels = {}
        for kind, label, about in ig.REFERENCE_KINDS:
            row = self.frame(self.ref_box)
            row.pack(side="top", fill="x", pady=(0, self.px(14)))
            self.label(row, label, "text").pack(side="top", anchor="w")
            name = self.label(row, "—", "faint", self.host.f_small,
                              wraplength=self.px(340))
            name.pack(side="top", fill="x", pady=(self.px(3), self.px(4)))
            buttons = self.frame(row)
            buttons.pack(side="top", fill="x")
            clear = self.button(buttons, "Clear", lambda k=kind: self._clear_ref(k),
                                kind="ghost")
            clear.pack(side="right")
            self.button(buttons, "Choose…", lambda k=kind, a=about: self._pick_ref(k, a),
                        kind="quiet").pack(side="left")
            self.button(buttons, "Library…", lambda k=kind: self.image_library(k),
                        kind="quiet").pack(side="left", padx=(self.px(4), 0))
            if kind == "pose":
                self.button(buttons, "Draw pose…", self.edit_pose, kind="quiet").pack(
                    side="left", padx=(self.px(4), 0))
            self.ref_labels[kind] = name

    def _pick_ref(self, kind, about):
        path = filedialog.askopenfilename(
            parent=self.host, title=about,
            filetypes=[("Pictures", "*.png *.jpg *.jpeg *.webp *.bmp"), ("All files", "*.*")])
        if path:
            if kind == "pose":
                self.pose = None      # a picture of its own replaces the drawn one
            self._set_ref(kind, path)

    def image_library(self, kind="source"):
        window = getattr(self, "_image_library", None)
        if window is None or not window.win.winfo_exists():
            window = self._image_library = ImageLibraryWindow(self, kind)
        else:
            window.role.set(dict((k, label) for k, label, _ in ig.REFERENCE_KINDS)[kind])
            window.reload()
            window.win.lift()
        return window

    def use_library_image(self, kind, path):
        if kind == "pose":
            self.pose = None
        self._set_ref(kind, path)
        self._show_section("References")

    def _clear_ref(self, kind):
        if kind == "pose":
            self.pose = None
        self._set_ref(kind, None)

    def _set_ref(self, kind, path):
        if path:
            self.refs[kind] = path
        else:
            self.refs.pop(kind, None)
        text = os.path.basename(path) if path else "—"
        if kind == "pose" and path and self.pose:
            text = "drawn figure · strength %.2f" % self.pose.get("strength", 0.9)
        self.ref_labels[kind].config(text=text)
        self._recheck()

    # ------------------------------------------------------------------ pose
    def edit_pose(self):
        return PoseEditor(self)

    def use_pose(self, pose):
        """From the pose editor: this stick figure is the pose reference."""
        self.pose = pose
        self._fit_pose()

    def _fit_pose(self):
        """The drawn pose's picture at the size the job will be. A size
        changed since it was drawn refits the figure (the same proportions,
        scaled and centred) rather than stretching it with the frame."""
        if not self.pose:
            return
        self._recheck()
        w, h = self.planned_size
        pw, ph = self.pose.get("width") or w, self.pose.get("height") or h
        if (pw, ph) != (w, h):
            self.pose = dict(self.pose, width=w, height=h,
                             points=sp.refit(self.pose["points"], (pw, ph), (w, h)))
        hidden = set(self.pose.get("hidden") or ())
        points = [None if i in hidden else p for i, p in enumerate(self.pose["points"])]
        folder = os.path.join(self.studio.lib.root, "poses")
        try:
            path = sp.save(points, w, h, folder, self.pose.get("hands"))
        except OSError as e:
            self.say("Could not write the pose picture in %s: %s" % (folder, e), "err")
            return
        self._set_ref("pose", path)

    # ----------------------------------------------------------------- LoRAs
    def _lora_model(self):
        return self.studio.lib.get("models", self.settings["model"])

    def _offered(self, rec):
        """Is this LoRA offered with the chosen model? On in Add-ons, and not
        made for another family (one with no family set is offered, marked)."""
        return rec.get("enabled", True) and ig.lora_fits(rec, self._lora_model()) is not False

    def lora_menu_items(self):
        """What Add LoRA offers for the chosen model -> ({category: [record]},
        [records with no family set]). LoRAs for other models are not in it."""
        by_cat, unknown = {}, []
        for r in self.studio.lib.all("loras"):
            if not self._offered(r):
                continue
            if ig.lora_fits(r, self._lora_model()) is None:
                unknown.append(r)
            else:
                by_cat.setdefault(r["category"], []).append(r)
        return by_cat, unknown

    def _post_lora_menu(self):
        menu = tk.Menu(self.add_lora_pill, tearoff=0)
        self.skin(menu, bg="card", fg="text", activebackground="sel", activeforeground="text")
        by_cat, unknown = self.lora_menu_items()
        model = self._lora_model()
        label = model["label"] if model else "this model"

        def submenu(title, recs):
            sub = tk.Menu(menu, tearoff=0)
            self.skin(sub, bg="card", fg="text", activebackground="sel",
                      activeforeground="text")
            for r in sorted(recs, key=lambda r: r["name"].lower()):
                sub.add_command(label=r["name"], command=lambda i=r["id"]: self._add_lora(i))
            menu.add_cascade(label=title, menu=sub)
        if not by_cat and not unknown:
            menu.add_command(label="Nothing installed for %s yet" % label, state="disabled")
        for cat in ig.CATEGORIES:
            if cat in by_cat:
                submenu(cat, by_cat[cat])
        if unknown:
            menu.add_separator()
            submenu("Model not set - may not work", unknown)
        menu.add_separator()
        menu.add_command(label="Find more for %s%s" % (label, ELLIPSIS),
                         command=lambda: self.open_addons("catalog"))
        p = self.add_lora_pill
        menu.tk_popup(p.winfo_rootx(), p.winfo_rooty() + p.winfo_height())

    def open_addons(self, tab="installed"):
        return AddonsWindow(self, tab)

    def _refit_loras(self):
        """Rows for LoRAs that do not suit the chosen model go to `parked`;
        parked ones that suit it now come back, at their strength."""
        lib = self.studio.lib
        for r in list(self.loras):
            rec = lib.get("loras", r["id"])
            if rec is not None and not self._offered(rec):
                self.parked[r["id"]] = r["var"].get()
                r["row"].destroy()
                self.loras.remove(r)
        for lid, strength in list(self.parked.items()):
            rec = lib.get("loras", lid)
            if rec is None:
                del self.parked[lid]
            elif self._offered(rec):
                del self.parked[lid]
                self._add_lora(lid, strength, recheck=False)
        self._show_parked()

    def _show_parked(self):
        lib = self.studio.lib
        names = [lib.get("loras", lid)["name"] for lid in self.parked if lib.get("loras", lid)]
        if not names:
            self.parked_note.pack_forget()
            return
        model = self._lora_model()
        self.parked_note.config(text="Set aside for %s (made for another model, or off in "
                                "Add-ons): %s" % (model["label"] if model else "this model",
                                                  ", ".join(names)))
        self.parked_note.pack(side="top", fill="x", before=self.lora_row,
                              padx=(self.px(6), self.px(10)))

    def _add_lora(self, lid, strength=None, recheck=True):
        rec = self.studio.lib.get("loras", lid)
        if rec is None or any(r["id"] == lid for r in self.loras):
            return
        if not self._offered(rec):
            self.parked[lid] = rec["strength"] if strength is None else strength
            self._show_parked()
            return
        var = tk.DoubleVar(value=rec["strength"] if strength is None else strength)
        row = self.frame(self.lora_box)
        row.pack(side="top", fill="x")
        entry = {"id": lid, "var": var, "row": row}
        self.button(row, "×", lambda: self._drop_lora(entry), kind="ghost").pack(
            side="right")
        self.label(row, rec["name"], "text", width=16).pack(side="left")
        self.slider(row, var, -1.0, 2.0, command=lambda _v: self._recheck()).pack(
            side="left", fill="x", expand=True)
        self.loras.append(entry)
        if recheck:
            self._recheck()

    def _drop_lora(self, entry):
        entry["row"].destroy()
        self.loras.remove(entry)
        self._recheck()

    # -------------------------------------------------------------- advanced
    def _build_advanced(self, box):
        grid = self.frame(box)
        grid.pack(side="top", fill="x")
        for i, (key, label, kind) in enumerate(ADVANCED):
            self.label(grid, label, "muted").grid(row=i, column=0, sticky="w",
                                                  pady=self.px(1))
            var = tk.StringVar()
            entry = self.host._entry(grid, var)
            entry.master.grid(row=i, column=1, sticky="we", padx=(self.px(8), 0))
            entry.bind("<KeyRelease>", lambda ev: self._recheck())
            self.adv[key] = var
            hint = self.label(grid, "", "faint", self.host.f_small)
            hint.grid(row=i, column=2, sticky="w", padx=(self.px(8), 0))
            self.hints[key] = hint
        grid.columnconfigure(1, weight=1)
        srow = self.frame(box)
        srow.pack(side="top", fill="x", pady=(self.px(4), 0))
        for text, var, cmd in (("New seed each time", self.random_seed, self._recheck),
                               ("Refine pass (upscale + low-denoise redraw)", self.refine,
                                self._touch_refine),
                               ("Face pass (redraw each face at full size; needs SAM3)",
                                self.faces, self._touch_faces)):
            b = tk.Checkbutton(srow, text=text, variable=var, command=cmd, anchor="w",
                               font=self.host.f_ui, bd=0, highlightthickness=0)
            self.skin(b, bg="bg", fg="text", activebackground="bg", selectcolor="card",
                      activeforeground="text")
            b.pack(side="top", fill="x")
        self.button(srow, "Random seed", self._roll_seed, kind="ghost").pack(
            side="top", anchor="w")
        self.label(box, "Negative prompt (models run at CFG 1 ignore it)", "muted").pack(
            side="top", fill="x", pady=(self.px(6), 0))
        self.neg = tk.StringVar()
        e = self.host._entry(box, self.neg)
        e.master.pack(side="top", fill="x")
        self.label(box, "Blank fields use the model's and style's defaults, shown beside "
                   "them.", "faint", self.host.f_small, wraplength=self.px(380)).pack(
            side="top", fill="x", pady=(self.px(4), 0))

    def _touch_refine(self):
        self.refine_set = True
        self._recheck()

    def _touch_faces(self):
        self.faces_set = True
        self._recheck()

    def _roll_seed(self):
        self.adv["seed"].set(str(ig.random.randint(0, ig.MAX_SEED)))
        self.random_seed.set(False)
        self._recheck()

    def _toggle_person(self):
        self._show_section("Image" if self.section == "People" else "People")

    def _toggle_advanced(self):
        self.adv_open = not self.adv_open
        self.adv_pill.set(text="Advanced  " +
                         ("▾" if self.adv_open else "▸"))
        if self.adv_open:
            self.adv_box.pack(side="top", fill="x", padx=(self.px(6), self.px(10)),
                              after=self.adv_pill.master)
        else:
            self.adv_box.pack_forget()

    # ------------------------------------------------------------- settings
    def collect(self):
        """The form as settings, exactly what history stores."""
        s = dict(self.settings)
        s["scene"] = self.scene.get("1.0", "end").strip()
        for key, var in self.text.items():
            s[key] = var.get().strip()
        for key, var in self.sliders.items():
            s[key] = int(var.get())
        s["item_refs"] = dict(self.item_refs)
        s["face_photos"] = list(self.face_photos)
        s["face_name"] = self.face_name if s["face_photos"] else ""
        s["anatomy"] = bool(self.anatomy.get())
        s["negative"] = self.neg.get().strip()
        s["identities"] = [{"id": iid, "strength": round(sv.get(), 3)}
                           for iid, (bv, sv, _) in self.idents.items() if bv.get()]
        st = self.studio.lib.get("styles", s["style"])
        s["style_strength"] = (round(self.style_strength.get(), 3)
                               if st and st["lora"] else None)
        s["loras"] = [{"id": r["id"], "strength": round(r["var"].get(), 3)}
                      for r in self.loras]
        s["references"] = dict(self.refs)
        s["pose"] = dict(self.pose) if self.pose and "pose" in self.refs else None
        s["view"] = self.aim.get()
        s["refine"] = bool(self.refine.get())
        s["face_detail"] = bool(self.faces.get())
        s["auto_refine"] = bool(self.auto_refine.get())
        s["hand_pass"] = bool(self.hand_pass.get())
        s["head_swap"] = bool(self.head_swap.get())
        # Ticked: always. Unticked: only after a swap that painted over them.
        s["glasses_pass"] = True if self.glasses_pass.get() else None
        for key, _, kind in ADVANCED:
            raw = self.adv[key].get().strip()
            if not raw:
                s[key] = None
                continue
            try:
                s[key] = int(float(raw)) if kind == "int" else float(raw) if kind == "float" \
                    else raw
            except ValueError:
                s[key] = None
        s["batch"] = s.get("batch") or 1
        if self.random_seed.get() or s.get("seed") is None:
            s["seed"] = -1
        return s

    def apply(self, settings):
        """Put saved settings back in the form (Reuse Settings)."""
        s = dict(ig.default_settings())
        s.update(settings)
        for key in ("preset", "model", "backend", "style"):
            self.settings[key] = s.get(key) or self.settings[key]
        self.settings["character"] = s.get("character") or ""
        if (s.get("camera_profile") or "none") != self.settings.get("camera_profile"):
            self.set_camera(s.get("camera_profile") or "none")
        chosen = {d["id"]: d.get("strength") for d in s.get("identities") or []
                  if isinstance(d, dict)}
        for iid, (bv, sv, _) in self.idents.items():
            bv.set(iid in chosen)
            if chosen.get(iid) is not None:
                sv.set(chosen[iid])
        self._show_identity()
        self.scene.delete("1.0", "end")
        self.scene.insert("1.0", s.get("scene") or "")
        for key, var in self.text.items():
            var.set(s.get(key) or "")
        for key, var in self.sliders.items():
            var.set(int(s.get(key) or 0))
        self.item_refs = ig.clean_item_refs(s.get("item_refs"))
        self.face_photos = ig._strs(s.get("face_photos"))
        self.face_name = s.get("face_name") or ""
        self.anatomy.set(s.get("anatomy") is not False)
        self.neg.set(s.get("negative") or "")
        pose = s.get("pose") if isinstance(s.get("pose"), dict) else None
        self.pose = dict(pose) if pose and sp.clean(pose.get("points")) else None
        self.aim.set(s.get("view"))
        self._aimed(recheck=False)
        for kind in ig.REFERENCE_NAMES:
            self._set_ref(kind, (s.get("references") or {}).get(kind))
        for key, _, _ in ADVANCED:
            v = s.get(key)
            self.adv[key].set("" if v is None or key == "batch" and v == 1 else str(v))
        fixed = s.get("seed_mode") == "fixed" or s.get("seed", -1) >= 0
        self.random_seed.set(not fixed)
        if fixed:
            self.adv["seed"].set(str(s.get("seed")))
        self.refine.set(bool(s.get("refine")))
        self.refine_set = True
        self.faces.set(bool(s.get("face_detail")))
        self.faces_set = True
        self.auto_refine.set(bool(s.get("auto_refine")))
        self.hand_pass.set(s.get("hand_pass", True) is not False)
        self.head_swap.set(s.get("head_swap", True) is not False)
        self.glasses_pass.set(s.get("glasses_pass") is True)
        for r in list(self.loras):
            r["row"].destroy()
        self.loras = []
        self.parked = {}
        self._rebuild_choices()
        for iid, (bv, sv, sc) in self.idents.items():
            bv.set(iid in chosen)
            if chosen.get(iid) is not None:
                sv.set(chosen[iid])
            if bv.get():
                sc.pack(side="right", fill="x", expand=True, padx=(self.px(8), 0))
        self._show_looks(self.look_section)
        if s.get("style_strength") is not None:
            self.style_strength.set(s["style_strength"])
        for d in s.get("loras") or []:
            if isinstance(d, dict):
                self._add_lora(d.get("id"), d.get("strength"), recheck=False)
        self.preset_mix = set()
        self._build_presets()
        if not self.adv_open:
            self._toggle_advanced()
        self._show_section("Settings")
        self._recheck()
        self.say("Settings loaded from history. Change anything, then Generate.", "muted")

    def _aimed(self, view=False, recheck=True):
        """The Camera diagram changed (or is cleared, with view None): say
        what it now asks for, and offer Clear while it asks for anything."""
        if view is None:
            self.aim.set(None)
        text = ig.view_label(self.aim.get())
        self.aim_note.config(text=text or "Not set: the model frames the picture. "
                             "Drag the camera to aim it.")
        self.skin(self.aim_note, bg="bg", fg="text" if text else "faint")
        if text:
            self.aim_clear.pack(side="right")
        else:
            self.aim_clear.pack_forget()
        if recheck:
            self._recheck()

    def _recheck(self):
        """Compose against the backend the job would go to, for the warnings
        and the defaults beside the advanced fields. No I/O."""
        s = self.collect()
        b, why = self.studio.plan_route(s)
        if b is None:
            self.route_note.config(text="")
            self.warn.config(text="• " + why)
            self.skin(self.warn, bg="bg", fg="err")
            b = next((x for x in self.studio.backends() if x["enabled"]), None)
            if b is None:
                if hasattr(self, "act_nodes") and not self._selected_steps()[0]:
                    self.act_nodes.set(state="disabled")
                return
            p = self.studio.preview(s, b)
            v = p.values
        else:
            p = self.studio.preview(s, b)
            lines = p.errors + p.warnings
            self.warn.config(text="\n".join("• " + x for x in lines) if lines else "")
            self.skin(self.warn, bg="bg", fg="err" if p.errors else "warn")
            h = self.studio.health.get(b["id"])
            self.route_note.config(text="Will run on " + why.split("Auto → ", 1)[-1]
                                   + ("" if h else " Not checked yet: Check asks it."))
            self.skin(self.route_note, bg="bg", fg="err" if p.errors else "muted")
            v = p.values
        if hasattr(self, "act_nodes"):
            self.act_nodes.set(state="normal" if (self._selected_steps()[0]
                                                  or ig.preview_graph(p)) else "disabled")
        try:
            self.planned_size = (int(v.get("width") or 1024), int(v.get("height") or 1024))
        except (TypeError, ValueError):
            pass
        for key, _, _ in ADVANCED:
            val = v.get(key)
            if key == "seed":
                val = "random" if self.random_seed.get() else ""
            elif key == "batch":
                val = 1
            self.hints[key].config(text="" if val in (None, "") else "default %s" % val)

    # ============================================================== generate
    def generate(self, extra=None, base=None):
        """Refuses, in words, what cannot run where it would go: a missing
        file or node, no backend able to take it. Routing that has not heard
        from the backends yet is left to submit(), which asks them. `extra`
        is laid over the form's settings for this job alone - the Scene
        Builder's frame size, denoise and scene. `base` replaces the form
        altogether - the Scene Builder's floor and wall pictures, which want
        none of its person. True when it was sent on."""
        self._fit_pose()
        s = dict(base) if base is not None else self.collect()
        s.update(extra or {})
        b, why = self.studio.plan_route(s)
        known = all(bk["id"] in self.studio.health for bk in self.studio.backends()
                    if bk["enabled"])
        if b is None and known:
            self.say(why, "err")
            return False
        if b is not None:
            p = self.studio.preview(s, b)
            if p is not None and p.errors:
                self.say("Not sent. " + " ".join(p.errors), "err")
                return False
        self.say("Routing" + ELLIPSIS, "muted")
        self.host._spawn(self.s.event_id, self._submit, s)
        return True

    def build_scene(self, path=None):
        """The Scene Builder: one window, raised if it is already open."""
        sb = self.scene_builder
        if sb is not None and sb.win.winfo_exists():
            sb.win.lift()
            if path:
                sb.open(path)
            return sb
        self.scene_builder = studio_scene_ui.SceneBuilder(self, path)
        self.show_shot_on()
        return self.scene_builder

    # ------------------------------------------------------------ Shot on
    # The cameras as a deck of cards above the Scene field: one card at a
    # time, flipped with ‹ › (or the wheel over it). The card showing is the
    # camera, `settings["camera_profile"]`. With the Scene Builder open the
    # two are one choice: a flip here sets the scene's camera, and a camera
    # chosen there (or a scene opened with one) turns the deck to it.
    def _camera_cards(self):
        return self.studio.lib.all("camera_profiles")

    def _build_camera_deck(self):
        for w in self.camera_deck.winfo_children():
            w.destroy()
        cards = self._camera_cards()
        if not cards:
            self.label(self.camera_deck, "No cameras yet.", "faint").pack(side="left")
            return
        ids = [c["id"] for c in cards]
        cid = self.settings.get("camera_profile") or ""
        if cid not in ids:
            cid = "none" if "none" in ids else ids[0]
            self.settings["camera_profile"] = cid
        i = ids.index(cid)
        cp = cards[i]
        side = self.px(CAMERA_CARD)
        row = self.frame(self.camera_deck)
        row.pack(side="top", anchor="w")
        self.camera_prev = self.button(row, "‹", lambda: self._flip_camera(-1), kind="ghost")
        self.camera_prev.pack(side="left", fill="y")
        card = tk.Frame(row, bd=0, highlightthickness=self.px(2))
        self.skin(card, bg="card", highlightbackground="accent")
        card.pack(side="left", padx=self.px(6))
        img = (photo_at(cp["image"], side, card)
               if cp.get("image") and os.path.isfile(cp["image"]) else None)
        self.camera_photo = img
        if img is not None:
            pic = tk.Label(card, image=img, bd=0)
            self.skin(pic, bg="card")
        else:
            pic = self.frame(card, "card")
            pic.config(width=side, height=side * 2 // 3)
            pic.pack_propagate(False)
            self.label(pic, cp["name"], "faint", self.host.f_small, bg="card",
                       wraplength=side - self.px(8)).pack(expand=True)
        pic.pack(side="top")
        self.camera_next = self.button(row, "›", lambda: self._flip_camera(1), kind="ghost")
        self.camera_next.pack(side="left", fill="y")
        about = [x for x in (("%dmm" % cp["lens"]) if cp.get("lens") else "",
                             cp.get("format") or "") if x]
        self.camera_name = self.label(self.camera_deck, cp["name"], "text", self.host.f_ui)
        self.camera_name.pack(side="top", anchor="w", pady=(self.px(4), 0))
        self.camera_about = self.label(
            self.camera_deck, " · ".join(about + ["%d of %d" % (i + 1, len(cards))]),
            "muted", self.host.f_small)
        self.camera_about.pack(side="top", anchor="w")
        foot = self.frame(self.camera_deck)
        foot.pack(side="top", anchor="w")
        self.button(foot, "Cameras…", self.edit_camera_profiles, kind="ghost").pack(side="left")
        for w in (card, pic, *pic.winfo_children()):
            w.bind("<MouseWheel>", lambda ev: self._flip_camera(-1 if ev.delta > 0 else 1))

    def _flip_camera(self, step):
        cards = self._camera_cards()
        if not cards:
            return
        ids = [c["id"] for c in cards]
        cid = self.settings.get("camera_profile") or ""
        i = ids.index(cid) if cid in ids else 0
        self.set_camera(ids[(i + step) % len(ids)])

    def set_camera(self, cid):
        """The deck's choice: the form's camera, and the open scene's."""
        self.settings["camera_profile"] = cid
        self._build_camera_deck()
        sb = self.scene_builder
        if sb is not None and (sb.scene["camera"].get("profile") or "none") != cid:
            sb._set_camera_profile(cid)
        self._recheck()

    def show_shot_on(self):
        """The Scene Builder's camera changed (or it opened or closed): turn
        the deck to it."""
        sb = self.scene_builder
        if sb is None or getattr(self, "camera_deck", None) is None:
            return
        cid = sb.scene["camera"].get("profile") or "none"
        if cid != self.settings.get("camera_profile"):
            self.settings["camera_profile"] = cid
            try:
                self._build_camera_deck()
            except tk.TclError:
                pass

    def _submit(self, s):
        """Off the UI thread: routing may check a backend's health."""
        try:
            jobs = self.studio.submit(s)
        except ig.ComfyError as e:
            self._post("said", (str(e), "err"))
            self._post("health")
            return
        self._post("submitted", jobs)
        self._post("health")

    # ============================================================== backends
    def refresh_backends(self):
        self.say("Checking the backends" + ELLIPSIS, "muted")
        self.host._spawn(self.s.event_id, self._check_all)

    def _check_all(self):
        self.studio.check_all()
        up, down = [], []
        for b in self.studio.backends():
            h = self.studio.health.get(b["id"]) or {}
            if h.get("ok"):
                up.append(b["name"])
            elif b["enabled"]:
                down.append("%s: %s" % (b["name"], h.get("detail", "no answer")))
        self._post("health")
        text = ("Online: " + ", ".join(up)) if up else "No backend answered."
        if down:
            text += "  Offline - " + "  ".join(down)
        self._post("said", (text, "warn" if down else "muted"))

    def _paint_health(self):
        for w in self.health_row.winfo_children():
            w.destroy()
        for b in self.studio.backends():
            h = self.studio.health.get(b["id"])
            if not b["enabled"]:
                role, text = "faint", "disabled"
            elif h is None:
                role, text = "faint", "not checked"
            elif h.get("ok"):
                free, total = h.get("vram_free"), h.get("vram_total")
                role = "ok"
                text = "ready"
                if total:
                    text += " · %.0f/%.0f GB free" % ((free or 0) / 1e9, total / 1e9)
                busy = self.studio.queue.load().get(b["id"], 0)
                if busy or h.get("queue"):
                    text += " · %d queued" % max(busy, h.get("queue", 0))
            else:
                role, text = "err", "offline"
            cell = self.frame(self.health_row)
            cell.pack(side="left", padx=(0, self.px(16)))
            dot = tk.Canvas(cell, width=self.px(10), height=self.px(10),
                            highlightthickness=0, bd=0)
            self.skin(dot, bg="bg")
            dot.create_oval(1, 1, self.px(10) - 1, self.px(10) - 1,
                            fill=self.host.C[role], outline="")
            dot.pack(side="left")
            self.label(cell, b["name"], "text", self.host.f_bold).pack(
                side="left", padx=(self.px(6), self.px(4)))
            lbl = self.label(cell, text, "muted", self.host.f_small)
            lbl.pack(side="left")
            if h is not None and not h.get("ok"):
                lbl.bind("<Button-1>", lambda ev, d=h.get("detail", ""): self.say(d, "err"))
                lbl.config(cursor="hand2")

    # ============================================================ right side
    def _build_right(self, right):
        # A grid, so the picture and the list share the height 3:2 whatever
        # the window's size and the display's scale. Neither frame lets what
        # it holds size it (pack_propagate off): a picture as large as its
        # file pushed the list off the window when this was packed.
        right.rowconfigure(0, weight=3)
        right.rowconfigure(2, weight=2)
        right.columnconfigure(0, weight=1)
        top = self.frame(right, "card")
        top.grid_propagate(False)
        top.pack_propagate(False)
        top.grid(row=0, column=0, sticky="nsew")
        acts = self.frame(top, "card")
        acts.pack(side="bottom", fill="x", padx=self.px(10), pady=(self.px(4), self.px(10)))
        self.act_again = self.button(acts, "Generate again  ▾", self._again_menu, bg="card")
        self.act_fix = self.button(acts, "Fix a spot", self._fix_selected, bg="card")
        self.act_blend = self.button(acts, "Blend" + ELLIPSIS, self._blend_selected, bg="card")
        self.act_again.pack(side="left")
        self.act_fix.pack(side="right")
        self.act_blend.pack(side="right", padx=(0, self.px(4)))
        for p in (self.act_again, self.act_fix):
            p.set(state="disabled")
        # The prompt and settings are in the Queue / History row below, so the
        # picture takes this card's whole height; the caption shows only when
        # a face swap failed, with the Retry button (packed in `_select`).
        self.caption = self.label(top, "", "muted", self.host.f_small, bg="card")
        self.act_retry_faces = self.button(top, "Retry face swap", self._retry_faces, bg="card")
        self.wrap(self.caption, top, self.px(24))
        self.preview = tk.Label(top, bd=0, highlightthickness=0, text="Nothing yet",
                                font=self.host.f_ui, cursor="hand2")
        self.skin(self.preview, bg="card", fg="faint")
        self.preview.pack(side="top", fill="both", expand=True, pady=self.px(8))
        self.preview.bind("<Double-Button-1>", lambda ev: self._open_selected())
        self.preview.bind("<Button-3>", self._picture_menu)
        self.preview.bind("<Configure>", lambda ev: self._repaint_preview())
        # Nodes floats over the picture's top-right corner, not in this row -
        # it opens ComfyUI's own editor, not an action on the picture itself.
        # Built last, so it stacks above the preview label under it.
        self.act_nodes = self.button(top, "Nodes", self._show_nodes, bg="card")
        self.act_nodes.place(relx=1.0, x=-self.px(8), y=self.px(8), anchor="ne")
        self.act_nodes.set(state="disabled")

        tabs = self.frame(right)
        tabs.grid(row=1, column=0, sticky="ew", pady=(self.px(10), self.px(4)))
        self.tab_queue = self.button(tabs, "Queue", lambda: self._show_list("queue"),
                                     kind="option")
        self.tab_hist = self.button(tabs, "History", lambda: self._show_list("history"),
                                    kind="ghost")
        self.tab_chars = self.button(tabs, "Characters", lambda: self._show_list("characters"),
                                     kind="ghost")
        self.tab_queue.pack(side="left")
        self.tab_hist.pack(side="left", padx=(self.px(6), 0))
        self.tab_chars.pack(side="left", padx=(self.px(6), 0))
        outer, self.list_box = self.scrolled(right)
        outer.pack_propagate(False)
        outer.grid(row=2, column=0, sticky="nsew")
        self._show_list("queue")

    def wrap(self, label, within, less):
        """Wrap `label` at the width `within` actually has, less `less` px -
        a fixed wraplength is wrong at every size but one."""
        within.bind("<Configure>", lambda ev: label.config(
            wraplength=max(self.px(80), ev.width - less)), add="+")

    def _picture_menu(self, ev):
        path = self.pending_preview
        if not path:
            return
        menu = tk.Menu(self.preview, tearoff=0)
        self.skin(menu, bg="card", fg="text", activebackground="sel", activeforeground="text")
        menu.add_command(label="Open", command=self._open_selected)
        menu.add_command(label="Fix a spot" + ELLIPSIS, command=self._fix_selected)
        menu.add_command(label="Blend with" + ELLIPSIS, command=self._blend_selected)
        if self._selected_steps()[0]:
            menu.add_command(label="Show nodes", command=self._show_nodes)
        menu.add_command(label="Show in folder", command=lambda: self._open_selected(True))
        menu.add_command(label="Copy path", command=lambda: (
            self.host.clipboard_clear(), self.host.clipboard_append(path)))
        menu.tk_popup(ev.x_root, ev.y_root)

    def _show_list(self, which):
        self.view = which
        self.tab_queue.roles = self.host.PILL_ROLES["option" if which == "queue" else "ghost"]
        self.tab_hist.roles = self.host.PILL_ROLES["option" if which == "history" else "ghost"]
        self.tab_chars.roles = self.host.PILL_ROLES["option" if which == "characters" else "ghost"]
        for p in (self.tab_queue, self.tab_hist, self.tab_chars):
            p.paint(self.host.C)
        for w in self.list_box.winfo_children():
            w.destroy()
        self.rows = {}
        self.keep = [k for k in self.keep if getattr(k, "_preview", False)]
        if which == "queue":
            if not self.jobs:
                self.label(self.list_box, "No jobs this session. Generate adds one; "
                           "History has everything made before.", "faint",
                           wraplength=self.px(420)).pack(
                    side="top", fill="x", padx=self.px(8), pady=self.px(8))
            for job in self.jobs:
                self._job_row(job)
        elif which == "characters":
            self._show_characters()
        else:
            records = self.studio.history.list(self.shown_history + 1)
            if not records:
                self.label(self.list_box, "Nothing generated yet.", "faint").pack(
                    side="top", fill="x", padx=self.px(8), pady=self.px(8))
            for rec in records[:self.shown_history]:
                self._record_row(rec)
            if len(records) > self.shown_history:
                self.button(self.list_box, "Show more", self._more_history,
                            kind="ghost").pack(side="top", pady=self.px(6))
        self.list_box.canvas.yview_moveto(0)

    def _more_history(self):
        self.shown_history += HISTORY_PAGE
        self._show_list("history")

    # ------------------------------------------------------------ Characters
    def _show_characters(self):
        """Every character (Library > Identities) as a card: their photos to
        add to or click off, a name to rename by. LoRA, face swap and notes
        stay in the fuller editor, opened per card - this tab is for the
        photos every character needs, not the tuning a few of them get."""
        top = self.frame(self.list_box)
        top.pack(side="top", fill="x", padx=(0, self.px(4)), pady=(0, self.px(6)))
        self.button(top, "New character", self._new_character, bg="card").pack(side="left")
        idents = self.studio.lib.all("identities")
        if not idents:
            self.label(self.list_box, "No characters yet. New character adds one, then "
                       "drop in their photos below.", "faint", wraplength=self.px(420)).pack(
                side="top", fill="x", padx=self.px(8), pady=self.px(8))
        for rec in idents:
            self._character_row(rec)

    def _character_row(self, rec):
        row = self.frame(self.list_box, "card")
        row.pack(side="top", fill="x", pady=(0, self.px(6)), padx=(0, self.px(4)))
        head = self.frame(row, "card")
        head.pack(side="top", fill="x", padx=self.px(8), pady=(self.px(8), self.px(2)))
        var = tk.StringVar(value=rec.get("name") or "")
        e = self.host._entry(head, var)
        e.master.pack(side="left", fill="x", expand=True)
        e.bind("<Return>", lambda ev, r=rec, v=var: self._rename_character(r, v))
        e.bind("<FocusOut>", lambda ev, r=rec, v=var: self._rename_character(r, v))
        self.button(head, "More settings" + ELLIPSIS, lambda r=rec: self._character_settings(r),
                   kind="ghost", bg="card").pack(side="right")
        photos = [p for p in rec.get("references") or [] if os.path.isfile(p)]
        self.label(row, "%d reference photo%s%s" % (
            len(photos), "" if len(photos) == 1 else "s",
            " - the first is Primary" if photos else ""), "muted", self.host.f_small,
            bg="card").pack(side="top", anchor="w", padx=self.px(8))
        grid = self.frame(row, "card")
        grid.pack(side="top", fill="x", padx=self.px(8), pady=(self.px(4), self.px(8)))
        side = self.px(72)
        shown = photos[:6]

        def tile(col, ring="hover"):
            f = tk.Frame(grid, width=side, height=side, bd=0, highlightthickness=self.px(2))
            self.skin(f, bg="hover", highlightbackground=ring, highlightcolor=ring)
            f.grid_propagate(False)
            f.grid(row=0, column=col, padx=self.px(2))
            return f

        for i, p in enumerate(shown):
            box = tile(i, "accent" if i == 0 else "hover")
            img = photo(p, side, profile=True)
            if img is not None:
                self.keep.append(img)
                lbl = peek(tk.Label(box, image=img, bd=0), p)
            else:
                lbl = tk.Label(box, text=os.path.basename(p), font=self.host.f_small,
                               wraplength=side - self.px(8))
            self.skin(lbl, bg="hover", fg="muted")
            lbl.place(relx=0.5, rely=0.5, anchor="center")
            for w in (box, lbl):
                w.bind("<Button-1>", lambda ev, r=rec, path=p: self._remove_character_photo(r, path))
        col = len(shown)
        if len(photos) > 6:
            self.label(grid, "+%d more - More settings…" % (len(photos) - 6), "faint",
                      self.host.f_small, bg="card").grid(row=0, column=col, padx=self.px(6),
                                                         sticky="w")
            col += 1
        add = tile(col)
        plus = tk.Label(add, text="+ Add", font=self.host.f_small)
        self.skin(plus, bg="hover", fg="accent")
        plus.place(relx=0.5, rely=0.5, anchor="center")
        for w in (add, plus):
            w.bind("<Button-1>", lambda ev, r=rec: self._add_character_photo(r))

    def _new_character(self):
        names = {r.get("name") for r in self.studio.lib.all("identities")}
        name, i = "New person", 2
        while name in names:
            name, i = "New person %d" % i, i + 1
        try:
            self.studio.lib.save("identities", self.studio.lib.all("identities") +
                                 [{"name": name}])
        except OSError as e:
            return self.say("Could not add a new character: %s" % e, "err")
        self._saved("identities")
        self._show_list("characters")

    def _rename_character(self, rec, var):
        name = var.get().strip()
        if not name or name == (rec.get("name") or ""):
            var.set(rec.get("name") or "")
            return
        rec["name"] = name
        try:
            self.studio.lib.save("identities", self.studio.lib.all("identities"))
        except OSError as e:
            return self.say("Could not rename: %s" % e, "err")
        self._saved("identities")
        self._show_list("characters")

    def _add_character_photo(self, rec):
        paths = filedialog.askopenfilenames(parent=self.host, filetypes=[
            ("Pictures", "*.png *.jpg *.jpeg *.webp"), ("All files", "*.*")])
        if not paths:
            return
        rid, owner = rec.get("id"), rec.get("name") or "person"
        existing = list(rec.get("references") or [])
        self.say("Adding %s's photo%s" % (owner, "" if len(paths) == 1 else "s") + ELLIPSIS,
                 "muted")

        def work():
            try:
                result = self.studio.lib.import_identity_photos(paths, owner, existing)
            except OSError as e:
                result = {"added": [], "duplicates": 0, "errors": [str(e)]}

            def done():
                # save() replaces every record with a freshly-cleaned copy, so
                # a save elsewhere while this copy ran (another card's rename
                # or photo add) leaves `rec` stale - look the person up by id,
                # on the library's current list, rather than trust the object.
                current = self.studio.lib.get("identities", rid)
                if current is None:
                    return    # deleted from elsewhere while the copy ran
                refs = current.setdefault("references", [])
                refs.extend(p for p in result["added"] if p not in refs)
                try:
                    self.studio.lib.save("identities", self.studio.lib.all("identities"))
                except OSError as e:
                    return self.say("Could not save the new photo(s): %s" % e, "err")
                self._saved("identities")
                if self.view == "characters":
                    self._show_list("characters")
                if result["errors"]:
                    self.say("Added %d, %d unreadable: %s" % (
                        len(result["added"]), len(result["errors"]),
                        "; ".join(result["errors"][:2])), "warn")
            self._post("call", done)
        self.host._spawn(self.s.event_id, work)

    def _remove_character_photo(self, rec, path):
        if not messagebox.askyesno("Remove photo", "Remove this photo from %s's references?"
                                   % (rec.get("name") or "this person"), parent=self.host):
            return
        rec["references"] = [p for p in rec.get("references") or [] if p != path]
        try:
            self.studio.lib.save("identities", self.studio.lib.all("identities"))
        except OSError as e:
            return self.say("Could not save: %s" % e, "err")
        self._saved("identities")
        self._show_list("characters")

    def _character_settings(self, rec):
        editor = self.edit_identities()
        for i, r in enumerate(editor.records):
            if r.get("id") == rec.get("id"):
                editor.lb.selection_clear(0, "end")
                editor.lb.selection_set(i)
                editor.lb.see(i)
                editor._pick()
                break

    def _thumb(self, parent, path, bg="card"):
        """A fixed square holding the picture. A Frame, because a Label with
        no image counts its width and height in characters and lines."""
        size = self.px(THUMB)
        box = tk.Frame(parent, width=size, height=size, bd=0, highlightthickness=0)
        self.skin(box, bg="hover")
        box.pack_propagate(False)
        box.img = tk.Label(box, bd=0, highlightthickness=0)
        self.skin(box.img, bg="hover")
        box.img.pack(expand=True)
        box.path = None
        larger = lambda: box.path
        peek(box, larger)
        peek(box.img, larger)
        self.set_thumb(box, path)
        return box

    def set_thumb(self, box, path):
        box.path = path
        img = photo(path, self.px(THUMB)) if path else None
        if img is not None:
            self.keep.append(img)
            box.img.config(image=img)

    # ---------------------------------------------------------------- jobs
    def _job_row(self, job):
        row = self.frame(self.list_box, "card")
        row.pack(side="top", fill="x", pady=(0, self.px(6)), padx=(0, self.px(4)))
        thumb = self._thumb(row, job.outputs[0] if job.outputs else None)
        thumb.pack(side="left", padx=self.px(6), pady=self.px(6))
        right = self.frame(row, "card")
        right.pack(side="left", fill="both", expand=True, pady=self.px(6))
        top = self.frame(right, "card")
        top.pack(side="top", fill="x")
        cancel = self.button(top, "Cancel", lambda: self.studio.queue.cancel(job),
                             kind="ghost", bg="card")
        cancel.pack(side="right", padx=(0, self.px(6)))
        status = self.label(top, "", "muted", self.host.f_bold, bg="card")
        status.pack(side="left")
        elapsed = self.label(top, "", "faint", self.host.f_small, bg="card")
        elapsed.pack(side="left", padx=(self.px(8), 0))
        s = job.settings
        prompt = ig.summary(s).replace("\n", " ")
        self.wrap(self.label(right, prompt[:140] + ("\u2026" if len(prompt) > 140 else ""),
                             "text", bg="card"), right, self.px(12))
        model = self.studio.lib.get("models", s.get("model")) or {}
        preset = ig.preset_info(self.studio.lib, s.get("preset"))["label"]
        meta = self.label(right, "", "faint", self.host.f_small, bg="card")
        meta.pack(side="top", fill="x")
        self.wrap(meta, right, self.px(12))
        strip = self.frame(right, "card")
        strip.pack(side="top", fill="x", pady=(self.px(3), 0))
        stages, stage_keys = [], []
        for i, (key, label) in enumerate(ig.pipeline_stages(self.studio.lib, s)):
            if i:
                self.label(strip, "→", "faint", self.host.f_small, bg="card").pack(
                    side="left", padx=self.px(3))
            lbl = self.label(strip, label, "faint", self.host.f_small, bg="card")
            lbl.pack(side="left")
            stages.append(lbl)
            stage_keys.append(key)
        bar = tk.Canvas(right, height=self.px(4), highlightthickness=0, bd=0)
        self.skin(bar, bg="card")
        bar.pack(side="top", fill="x", pady=(self.px(4), 0), padx=(0, self.px(8)))
        detail = self.label(right, "", "faint", self.host.f_small, bg="card")
        detail.pack(side="top", fill="x")
        self.wrap(detail, right, self.px(12))
        widgets = {"row": row, "thumb": thumb, "status": status, "elapsed": elapsed,
                   "bar": bar, "detail": detail, "cancel": cancel, "meta": meta,
                   "stages": stages, "stage_keys": stage_keys, "strip": strip,
                   "base": "%s · %s · %s · seed %s" % (
                       preset, model.get("label", s.get("model")), job.backend["name"],
                       s.get("seed")) if s.get("mode") not in ("dress", "blend") else
                   "%s · %s · seed %s" % ("Try On" if s["mode"] == "dress" else "Blend",
                                          job.backend["name"], s.get("seed"))}
        if ig.local_faces(s):
            widgets["base"] = "Face swap · this PC"
        for w in (row, right, thumb, thumb.img, status, meta, detail):
            w.bind("<Button-1>", lambda ev: self._select(("job", job)))
        self.rows[job.id] = widgets
        self._paint_job(job)

    def _paint_job(self, job):
        w = self.rows.get(job.id)
        if w is None:
            return
        cancelling = job.cancel.is_set() and job.status not in ig.FINISHED
        w["status"].config(text="Cancelling…" if cancelling else
                           STATUS_TEXT.get(job.status, job.status.capitalize()))
        if cancelling:
            w["cancel"].set(state="disabled")
        self.skin(w["status"], bg="card", fg=STATUS_ROLE.get(job.status, "muted"))
        loras = ", ".join("%s %.2f" % (l["name"], l["strength"])
                          for l in (job.plan.lora_meta if job.plan else []))
        w["meta"].config(text=w["base"] + (" · " + loras if loras else ""))
        w["detail"].config(text=job.detail or "")
        self.skin(w["detail"], bg="card", fg="err" if job.status == "failed" else "faint")
        key = STAGE_KEY.get(job.status, False)
        if key is False:                  # failed or cancelled: the strip has said its piece
            w["strip"].pack_forget()
        else:
            keys = w["stage_keys"]
            at = keys.index(key) if key in keys else -1
            for i, lbl in enumerate(w["stages"]):
                role = "ok" if i < at or job.status == "complete" else (
                    "accent" if i == at else "faint")
                self.skin(lbl, bg="card", fg=role)
                lbl.config(font=self.host.f_bold if i == at and job.status != "complete"
                           else self.host.f_small)
        w["elapsed"].config(text="%ds" % job.elapsed() if job.started else "")
        bar = w["bar"]
        bar.delete("all")
        width = bar.winfo_width()
        if job.status not in ig.FINISHED and width > 1:
            bar.create_rectangle(0, 0, width, self.px(4), fill=self.host.C["border"],
                                 outline="")
            if job.progress is not None:
                bar.create_rectangle(0, 0, int(width * job.progress), self.px(4),
                                     fill=self.host.C["accent"], outline="")
        if job.status in ig.FINISHED:
            w["cancel"].pack_forget()
            bar.pack_forget()

    def _job_changed(self, job):
        if job not in self.jobs:
            self.jobs.insert(0, job)
            if self.view == "queue":
                self._show_list("queue")
        if job.outputs and job.id in self.rows:
            self.set_thumb(self.rows[job.id]["thumb"], job.outputs[0])
            if self.selected is None or self.selected[0] == "job":
                self._select(("job", job))
        self._paint_job(job)
        running = [j for j in self.jobs if j.status not in ig.FINISHED]
        if running:
            self.s.status = ("%d image job%s running" % (len(running), "" if len(running) == 1
                                                          else "s") + ELLIPSIS, "accent", False)
        else:
            self.s.status = ("Image Studio ready", "muted", False)
        if self.s.id == self.host.active:
            self.host._apply_status()
        if running:
            self.host._animate(("images-clock", self.s.event_id), self._tick)
        if job.status in ig.FINISHED and (job.settings.get("scene_texture")
                                          or job.settings.get("scene_picture")):
            sb = self.scene_builder
            if sb is not None and sb.win.winfo_exists():
                sb.texture_done(job)
        if job.status in ig.FINISHED:
            self._paint_health()
            if job.status == "failed":
                self.say("A job on %s failed: %s" % (job.backend["name"], job.detail), "err")
                if self.selected is None or self.selected[0] == "job":
                    self._select(("job", job))
                    if not job.outputs:
                        self.preview.config(image="", text="Failed - the reason is below")
                        self.skin(self.preview, bg="card", fg="err")

    def _tick(self, _frame):
        """The elapsed clocks and bars, on the window's one tick, and only
        while a job is unfinished."""
        live = [j for j in self.jobs if j.status not in ig.FINISHED]
        for job in live:
            self._paint_job(job)
        return bool(live)

    # --------------------------------------------------------------- history
    def _record_row(self, rec):
        row = self.frame(self.list_box, "card")
        row.pack(side="top", fill="x", pady=(0, self.px(6)), padx=(0, self.px(4)))
        thumb = self._thumb(row, (rec.get("images") or [None])[0])
        thumb.pack(side="left", padx=self.px(6), pady=self.px(6))
        right = self.frame(row, "card")
        right.pack(side="left", fill="both", expand=True, pady=self.px(6))
        btns = self.frame(right, "card")
        btns.pack(side="top", fill="x")
        self.button(btns, "Repeat seed", lambda: self._again(rec), bg="card").pack(
            side="right", padx=(0, self.px(6)))
        self.button(btns, "Reuse", lambda: self.reuse(rec["settings"]), bg="card").pack(
            side="right", padx=(0, self.px(4)))
        self.label(btns, rec.get("created", "")[5:16].replace("T", " "), "muted",
                   self.host.f_small, bg="card").pack(side="left")
        prompt = (rec.get("prompt") or "").replace("\n", " ")
        for text, role, font in ((prompt[:160] + ("\u2026" if len(prompt) > 160 else ""),
                                  "text", None), (self.describe(rec), "faint",
                                                  self.host.f_small),
                                 ("; ".join(rec.get("warnings") or []), "faint",
                                  self.host.f_small),
                                 (rec.get("license") or "", "faint", self.host.f_small)):
            if not text:
                continue
            lbl = self.label(right, text, role, font, bg="card")
            lbl.pack(side="top", fill="x")
            self.wrap(lbl, right, self.px(12))
        for w in (row, right, thumb, thumb.img):
            w.bind("<Button-1>", lambda ev: self._select(("record", rec)))

    def describe(self, rec):
        if rec.get("workflow") == "facefusion":
            return "Face swap on this PC" + (" · %ss" % rec["duration"] if rec.get("duration") else "")
        bits = [(rec.get("model") or {}).get("label") or "?",
                (rec.get("backend") or {}).get("name") or "?",
                "seed %s" % rec.get("seed"),
                "%sx%s" % (rec.get("width"), rec.get("height")),
                "%s steps" % rec.get("steps"),
                "%s/%s" % (rec.get("sampler"), rec.get("scheduler"))]
        if rec.get("guidance") is not None:
            bits.append("guidance %s" % rec["guidance"])
        if rec.get("refine"):
            bits.append("refined x%s at %s%s" % (
                rec["refine"].get("upscale"), rec["refine"].get("denoise"),
                " (%s)" % os.path.splitext(rec["refine"]["model"])[0]
                if rec["refine"].get("model") else ""))
        if (rec.get("face_detail") or {}).get("redrawn"):
            bits.append("%d face(s) redrawn at %s" % (rec["face_detail"]["redrawn"],
                                                     rec["face_detail"].get("denoise")))
        if rec.get("identities"):
            bits.append("people: " + ", ".join("%s %.2f" % (i["name"], i["strength"] or 0)
                                               for i in rec["identities"]))
        if rec.get("style"):
            bits.append("style: " + rec["style"]["name"])
        if rec.get("loras"):
            bits.append("LoRAs: " + ", ".join("%s %.2f" % (l["name"], l["strength"])
                                              for l in rec["loras"]))
        if rec.get("duration"):
            bits.append("%ss" % rec["duration"])
        return " · ".join(bits)

    # ------------------------------------------------------------- selection
    def _select(self, item):
        self.selected = item
        if item[0] == "job":
            path = item[1].outputs[0] if item[1].outputs else None
        else:
            path = (item[1].get("images") or [None])[0]
        self.pending_preview = path
        self._repaint_preview()
        rec = self._selected_record()
        if rec and (rec.get("finish") or {}).get("profiles") and rec["finish"].get("state") != "complete":
            error = rec["finish"].get("error")
            self.caption.config(text="Generated image kept before the final face swap."
                                + (" " + error if error else ""))
            self.caption.pack(side="bottom", fill="x", padx=self.px(12), before=self.preview)
            self.act_retry_faces.pack(side="bottom", anchor="w", padx=self.px(12),
                                     pady=self.px(4), before=self.preview)
            self.act_retry_faces.set(state="disabled" if self._face_pass_busy(rec) else "normal")
        else:
            self.caption.pack_forget()
            self.act_retry_faces.pack_forget()
        self.act_again.set(state="normal" if rec or item[0] == "job" else "disabled")
        self.act_fix.set(state="normal" if path and os.path.isfile(path) else "disabled")
        self.act_nodes.set(state="normal" if (self._selected_steps()[0]
                                              or self._form_steps()[0]) else "disabled")

    @staticmethod
    def clip(text, n=180):
        text = " ".join(text.split())
        return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + "\u2026"

    def _repaint_preview(self):
        path = self.pending_preview
        if not path:
            return
        box = min(self.preview.winfo_width(), self.preview.winfo_height())
        img = photo(path, max(box, self.px(200)) - self.px(8))
        if img is None:
            self.preview.config(image="", text="No preview for %s" % os.path.basename(path))
            return
        img._preview = True
        self.keep = [k for k in self.keep if not getattr(k, "_preview", False)] + [img]
        self.preview.config(image=img, text="")

    def _selected_record(self):
        if self.selected is None:
            return None
        if self.selected[0] == "record":
            return self.selected[1]
        return self.selected[1].record

    def _selected_settings(self):
        rec = self._selected_record()
        if rec:
            return rec["settings"]
        if self.selected and self.selected[0] == "job":
            return self.selected[1].settings
        return None

    def _again(self, rec, new_seed=False):
        """Repeat saved parameters; library entries and backend files remain live."""
        s = ig.again(rec, new_seed)
        self.say(("Same settings, new seed" if new_seed else
                  "Repeating seed %s" % s.get("seed")) +
                 ". Uses current profiles, styles and model files; the result can differ.", "muted")
        self.host._spawn(self.s.event_id, self._submit, s)

    def _face_pass_busy(self, rec):
        """A job still finishing `rec`: its own first pass, or a retry of it."""
        return any(j.status not in ig.FINISHED and (
            (j.record and j.record["id"] == rec["id"])
            or (rec.get("path") and (j.settings.get("face_finish") or {}).get("record") == rec["path"]))
            for j in self.jobs)

    def _retry_faces(self):
        rec = self._selected_record()
        if rec:
            if self._face_pass_busy(rec):
                return self.say("This face pass is still running. Cancel it before retrying.", "muted")
            try:
                settings = ig.retry_faces(rec)
            except ig.ComfyError as error:
                return self.say(str(error), "err")
            self.say("Applying the saved face profiles to the kept picture; no regeneration.", "muted")
            self.host._spawn(self.s.event_id, self._submit, settings)

    def _again_menu(self):
        """Repeat seed, New seed and Reuse settings, one menu under one
        button rather than three pills fighting the row for space."""
        menu = tk.Menu(self.act_again, tearoff=0)
        self.skin(menu, bg="card", fg="text", activebackground="sel", activeforeground="text")
        menu.add_command(label="Repeat seed", command=self._again_selected)
        menu.add_command(label="New seed", command=lambda: self._again_selected(True))
        menu.add_command(label="Reuse settings", command=self._reuse_selected)
        menu.tk_popup(self.act_again.winfo_rootx(),
                      self.act_again.winfo_rooty() + self.act_again.winfo_height())

    def _again_selected(self, new_seed=False):
        rec = self._selected_record()
        if rec:
            return self._again(rec, new_seed)
        s = self._selected_settings()
        if s:
            self._again({"settings": s}, new_seed)

    def _reuse_selected(self):
        s = self._selected_settings()
        if s:
            self.reuse(s)

    def reuse(self, settings):
        """Reuse Settings: put them back on the form. A Try On, from before
        item pictures went into the picture itself, has no form to go back
        to; Generate Again still remakes it."""
        if settings.get("mode") == "dress":
            return self.say("That was a Try On, which the form no longer has; Generate "
                            "Again remakes it.", "warn")
        if settings.get("mode") == "blend":       # a blend's form is its own window
            return self.blend(settings=settings)
        self.apply(self.studio.fix_base(settings) if settings.get("mode") == "fix"
                   else settings)

    def _fix_selected(self):
        """Fix a spot on the picture shown: mark the parts to redraw."""
        path, s = self.pending_preview, self._selected_settings()
        if not path or not os.path.isfile(path) or s is None:
            return self.say("Choose a finished picture to fix.", "warn")
        FixWindow(self, path, s)

    def _blend_selected(self):
        """Blend, with the picture shown (if there is one) as its first."""
        path = self.pending_preview
        self.blend(path if path and os.path.isfile(path) else None)

    def blend(self, first=None, settings=None):
        """The Blend window (`BlendWindow`): one, raised if it is already
        open, taking `first` as its first picture or a blend's `settings`
        (Reuse settings) as everything it holds."""
        window = getattr(self, "_blend_window", None)
        if window is None or not window.win.winfo_exists():
            window = self._blend_window = BlendWindow(self)
        window.take(first, settings)
        window.win.lift()
        return window

    # ============================================================ the Nodes view
    def _selected_steps(self):
        """-> ([(label, graph)], ComfyUI url, name) of the picture selected;
        ([], None, None) when it has none yet - a job still waiting for its
        lane composes nothing until it starts, so it falls back to a
        preview graph built locally for the backend it is assigned to."""
        item = self.selected
        if item is None:
            return [], None, None
        rec = self._selected_record()
        if rec:
            fields, backend, name = rec, rec.get("backend") or {}, rec.get("id")
            steps = comfy_view.graph_steps(fields)
            url = backend.get("url")
            return (steps, url, name) if steps and url else ([], None, None)
        if item[0] != "job":
            return [], None, None
        job = item[1]
        fields, backend, name = comfy_view.job_fields(job), job.backend or {}, job.id
        steps = comfy_view.graph_steps(fields)
        url = backend.get("url")
        if steps and url:
            return steps, url, name
        if not backend.get("url") or job.settings.get("mode") == "blend":
            return [], None, None         # a blend is not composed from the form
        graph = ig.preview_graph(self.studio.preview(job.settings, backend))
        return ([("Pipeline", graph)], backend["url"], name) if graph else ([], None, None)

    def _form_steps(self):
        """The pipeline the form would submit right now, on the backend Auto
        would send it to - composed locally, nothing sent to ComfyUI. ([],
        None, None) when there is not enough on the form to build one."""
        s = self.collect()
        b, _why = self.studio.plan_route(s)
        if b is None:
            b = next((x for x in self.studio.backends() if x["enabled"]), None)
        if b is None:
            return [], None, None
        graph = ig.preview_graph(self.studio.preview(s, b))
        return ([("Pipeline", graph)], b["url"], "This picture") if graph else ([], None, None)

    def _show_nodes(self):
        """The picture's graphs in ComfyUI's own editor: the ComfyUI tab's
        Nodes view (studio_nodes_ui), where each step can be opened. The
        picture selected in Queue or History first; the form itself
        (whatever Generate would send right now) otherwise - so Nodes
        always shows the pipeline connected to the instance that will run
        it, built only now, on demand."""
        steps, url, name = self._selected_steps()
        if not steps:
            steps, url, name = self._form_steps()
        if not steps:
            return self.say("Not enough set to show a pipeline yet.", "warn")
        self.host.open_nodes(steps, url, name)

    def _open_selected(self, select=False):
        path = self.pending_preview
        if path and os.path.exists(path):
            open_path(path, select)

    # =============================================================== editors
    def _saved(self, kind):
        self._rebuild_choices()
        if kind == "identities":
            self._prepare_profiles()
        if kind == "backends":
            self.studio.clients.clear()
            self._paint_health()
            self.refresh_backends()
        if kind == "camera_profiles":
            self._build_camera_deck()
        if kind == "camera_profiles" and self.scene_builder is not None:
            try:
                if self.scene_builder.win.winfo_exists():
                    self.scene_builder._inspect()
            except tk.TclError:
                pass
        self._recheck()

    def edit_backends(self):
        return RecordEditor(self, "backends", "Backends", [
            ("name", "Name", "text"),
            ("url", "ComfyUI API URL", "text"),
            ("ws_url", "WebSocket URL (blank: derived)", "text"),
            ("enabled", "Enabled", "bool"),
            ("roles", "Roles (what Auto sends here)", ("multi", ig.ROLES)),
            ("shares_llm_gpu", "Shares its GPU with LM Studio (clear it before a job)", "bool"),
            ("release_vram", "Free VRAM when its queue empties", "bool"),
            ("encoder_on_cpu", "Run the text encoder on the CPU", "bool"),
            ("max_megapixels", "Largest refine size (megapixels)", "number"),
            ("lora_dir", "LoRA folder, if it is on this PC (for previews)", "text"),
            ("notes", "Notes", "long"),
        ], template={"name": "New backend", "url": "http://127.0.0.1:8188",
                     "roles": ["secondary"]})

    def edit_models(self):
        wfs = [(w["id"], w.get("label", w["id"])) for w in ig.list_workflows()
               if not w.get("built_by")]      # a finish (Try On), not a model's workflow
        return RecordEditor(self, "models", "Models", [
            ("label", "Name", "text"),
            ("id", "Logical id (what history records)", "text"),
            ("family", "Model family", ("choice", [("", "unknown")] + list(ig.FAMILIES.items()))),
            ("workflow", "Workflow template", ("choice", wfs)),
            ("values", "Files and settings, every machine (name = value)", "kv"),
            ("backends", "Per machine: what differs there", "per_backend"),
            ("defaults", "Defaults (steps, guidance, sampler, width...)", "kv"),
            ("notes", "Notes", "long"),
        ], template={"id": "new-model", "label": "New model", "family": "flux1",
                     "workflow": "flux_dev_baseline", "values": {"model": "model.safetensors"}},
            extra=("Check backends", self._recheck_models), info=self._model_status)

    def _model_status(self, rec):
        """The Models window's answer to "can I use this, and where": per
        backend, ready or exactly which files (and folders) or nodes are
        missing, for the record as it stands in the editor."""
        model = ig.clean_model(rec)
        if model is None:
            return [("Give the model an id to check it.", "faint")]
        out = []
        for bid, (state, text) in self.studio.readiness(model).items():
            b = self.studio.backend(bid)
            role = {"ready": "ok", "missing": "err", "offline": "warn"}.get(state, "faint")
            out.append(("%s  %s: %s" % (READY_MARK.get(state, "?"), b["name"], text), role))
        wf = model.get("workflow")
        out.append(("Workflow: comfy_workflows/%s.json" % wf, "faint"))
        return out

    def _recheck_models(self, editor):
        editor.status("Checking the backends" + ELLIPSIS)

        def work():
            self.studio.check_all()
            self._post("health")
            self.host.q.put(("images", self.s.event_id, ("editor-refresh", editor)))
        self.host._spawn(self.s.event_id, work)

    def edit_loras(self):
        return RecordEditor(self, "loras", "LoRA library", [
            ("name", "Friendly name", "text"),
            ("file", "Filename", "text"),
            ("files", "Filename on a machine where it differs", "per_backend_file"),
            ("category", "Category", ("choice", [(c, c) for c in ig.CATEGORIES])),
            ("trigger", "Trigger phrase", "text"),
            ("strength", "Recommended strength", "number"),
            ("body_control", "Chest control (female: signed strength; male: size words)",
             ("choice", [("", "None"), ("chest_female", "Female chest slider"),
                         ("chest_male", "Male chest sizes")])),
            ("always", "Always on (every picture from a model it suits)", "bool"),
            ("family", "Trained for", ("choice", [("", "unknown")] + list(ig.FAMILIES.items()))),
            ("preview", "Preview image", "path"),
            ("source", "Where it came from", "text"),
            ("notes", "Notes", "long"),
        ], template={"file": "new_lora.safetensors", "category": "Other"},
            extra=[("Import from CivitAI…", lambda ed: LoraImport(self, ed)),
                   ("Scan backends", self._scan_loras)],
            label=lambda r: "%s — %s%s" % (r["category"], r["name"],
                                            "  (always on)" if r.get("always") else ""))

    def lora_folders(self):
        """[(backend id, "name — folder")] for every backend whose LoRA folder
        is on this PC: the only places a file can be put."""
        return [(b["id"], "%s — %s" % (b["name"], b["lora_dir"]))
                for b in self.studio.backends()
                if b.get("lora_dir") and os.path.isdir(b["lora_dir"])]

    def _scan_loras(self, editor):
        editor.status("Scanning" + ELLIPSIS)

        def work():
            self.studio.check_all()
            n = self.studio.scan_loras()
            self._post("library")
            self._post("said", ("LoRA scan: %d new in the library." % n, "muted"))
            self.host.q.put(("images", self.s.event_id, ("editor-reload", editor)))
        self.host._spawn(self.s.event_id, work)

    def head_lora_choices(self):
        """The LoRAs a head swap can take: Klein 9B ones, or unknown."""
        return [("", "none")] + [(r["id"], "%s (%s)" % (r["name"], r["category"]))
                                 for r in self.studio.lib.all("loras")
                                 if ig.compatibility(r["family"], lt.FAMILY) is not False]

    def edit_identities(self):
        loras = [("", "none")] + [(r["id"], "%s (%s)" % (r["name"], r["category"]))
                                  for r in self.studio.lib.all("loras")]
        return RecordEditor(self, "identities", "Identities", [
            ("name", "Name", "text"),
            ("description", "Identity description (used with the photos)", "long"),
            ("references", "Reference photos — first photo is Primary", "paths"),
            ("pool_photos", "Pool photos for WithAnyone (experimental)", "bool"),
            ("avatar", "Profile picture (optional; a generated picture is fine)", "path"),
            ("face_swap", "Final FaceFusion swap (not used by WithAnyone)", "bool"),
            ("swap_strength", "Face swap strength (0.5 is their face in full; more adds "
                              "nothing, less keeps some of the drawn face)", "number"),
            ("head_lora", "Head swap LoRA (FLUX.2 Klein 9B; Build LoRA makes it)",
             ("choice", self.head_lora_choices())),
            ("notes", "Notes", "long"),
            ("lora", "Identity LoRA", ("choice", loras)),
            ("trigger", "Trigger token", "text"),
            ("strength", "Default LoRA strength", "number"),
            ("use_references", "Use first photo as face reference (other models)", "bool"),
            ("reference_strength", "Face reference strength", "number"),
        ], template={"name": "New person", "strength": 0.85, "face_swap": True,
                      "swap_strength": 0.8},
            extra=[("Use this person", self._use_profile), ("Pick person…", self._pick_person)])

    def _use_profile(self, editor):
        if editor.current is None:
            return
        if not editor._save():
            return
        self._select_identity(editor.records[editor.current]["id"])
        editor.win.destroy()

    # ------------------------------------------------------------ Build LoRA
    def build_lora(self, editor):
        """The person's saved photos -> their own FLUX.2 Klein LoRA of their
        head (`lora_train`), trained by ai-toolkit on this PC's GPU in the
        background. One build at a time; a second click offers to stop it.
        When done, the file goes into this PC's LoRA folder, joins the
        library, and becomes the person's head swap LoRA (`head_lora`)."""
        run = self.lora_build
        if run is not None:
            n, total = run.step
            if messagebox.askyesno("Build LoRA", "A LoRA for %s is being built (step %d of "
                                   "%d). Stop it?" % (run.spec["person"], n, total),
                                   parent=editor.win):
                run.stop()
            return
        if editor.current is None:
            return
        w = editor.widgets.get("references")
        count = len(lt.usable(w[1]["paths"])) if w else 0
        if count < lt.MIN_PHOTOS:
            return editor.status("A LoRA needs at least %d photos of the person; this "
                                 "one has %d." % (lt.MIN_PHOTOS, count), "err")
        problem = lt.problem()
        if problem:
            return editor.status(problem, "err")
        folders = [b for b in self.studio.backends() if b.get("lora_dir")
                   and os.path.isdir(b["lora_dir"])]
        folders.sort(key=lambda b: not re.match(r"^https?://(127\.0\.0\.1|localhost)[:/]",
                                                b.get("url", "")))
        if not folders:
            return editor.status("No backend has a LoRA folder on this PC.", "err")
        if not editor._save():
            return
        ident = self.studio.lib.get("identities", editor.records[editor.current]["id"])
        try:
            spec = lt.plan(ident, folders[0]["lora_dir"])
        except ValueError as e:
            return editor.status(str(e), "err")
        if not messagebox.askokcancel("Build LoRA", (
                "Train a head LoRA for %s from %d photos?\n\nIt takes about 9 minutes "
                "and uses the GPU: ComfyUI's models are unloaded first, and pictures "
                "on %s should wait until it is done. Keep Studio Assist open; closing it "
                "stops the training.\n\nWhen it finishes it becomes %s's head swap LoRA "
                "(FLUX.2 Klein 9B, trigger word \"%s\"): the head swap before the face swap "
                "draws them with it.") % (
                    spec["person"], len(spec["photos"]), folders[0]["name"],
                    spec["person"], spec["trigger"]), parent=editor.win):
            return
        run = self.lora_build = lt.Build(spec)
        backend = folders[0]
        self.say("Building %s's LoRA: preparing %d photos" % (spec["person"],
                                                              len(spec["photos"])) + ELLIPSIS)
        editor.status("LoRA build started; progress shows under Generate.")

        def progress(kind, value):
            if kind == "kept" and value[0] < value[1]:
                self._post("said", ("Building %s's LoRA: %d of %d photos could not be read; "
                                    "training on the other %d." % (
                                        spec["person"], value[1] - value[0], value[1],
                                        value[0]), "warn"))
            elif kind == "faces" and value[0] < value[1]:
                self._post("said", ("Building %s's LoRA: no face found in %d of %d photos; "
                                    "those train whole." % (
                                        spec["person"], value[1] - value[0], value[1]), "muted"))
            elif kind == "step":
                n, total = value
                left = lt.eta(run.started, n, total) if run.started else ""
                text = ("Building %s's LoRA: step %d of %d" % (spec["person"], n, total)
                        + (" · " + left if left else ""))
                self._post("said", (text, "muted"))

        def finished(path, error):
            self.lora_build = None
            if error:
                return self.say("LoRA for %s not built: %s" % (spec["person"], error),
                                "muted" if run.stopped else "err")
            self._attach_lora(spec, editor)
            self.say("%s's head LoRA is ready and set as their head swap LoRA (%s)."
                     % (spec["person"], os.path.basename(path)), "ok")
            self.refresh_backends()      # ComfyUI's LoRA list now has the file

        def work():
            path, error = None, "the build stopped unexpectedly"
            try:
                try:
                    self.studio.client(backend).free()
                except Exception:
                    pass                 # offline: nothing is holding the GPU
                path, error = run.run(progress), None
            except (RuntimeError, OSError) as e:
                error = str(e)
            finally:
                # Always: a build that died any other way must not leave
                # `lora_build` set, or Build LoRA offers to stop it until a restart.
                self._post("call", lambda: finished(path, error))
        self.host._spawn(self.s.event_id, work)

    def _attach_lora(self, spec, editor):
        """A finished build into the library and onto its person - and onto
        the open editor's copy of them, so its next Save keeps it."""
        lib = self.studio.lib
        rec, _ = lib.import_lora(lt.lora_record(spec))
        lib.save("loras")
        ident = lib.get("identities", spec["identity"])
        if ident is not None:
            ident["head_lora"] = rec["id"]
            lib.save("identities")
        if editor.win.winfo_exists():
            choices = {"lora": [("", "none")] + [(r["id"], "%s (%s)" % (r["name"], r["category"]))
                                                 for r in lib.all("loras")],
                       "head_lora": self.head_lora_choices()}
            editor.fields = [(k, label, ("choice", choices[k]) if k in choices else kind)
                             for k, label, kind in editor.fields]
            shown = (editor.current is not None and
                     editor.records[editor.current].get("id") == spec["identity"])
            if shown:
                editor._store()
            for r in editor.records:
                if r.get("id") == spec["identity"]:
                    r["head_lora"] = rec["id"]
            if shown:
                editor._build_form()
        self._saved("loras")
        self._saved("identities")

    # ------------------------------------------------------- person cut-out
    def _pick_person(self, editor):
        """A photo - the selected reference, else one chosen from disk -
        cropped by a drag, then cut down to one person on white: SAM3 finds
        everyone in the crop; with more than one, a click says who. The
        cut-out takes the photo's place in the list (the front for a photo
        from disk), so it is the face reference when the photo was; a listed
        photo stays after it."""
        w = editor.widgets.get("references")
        pics = w[1] if w else None
        if pics is None:
            return
        if pics["sel"]:
            path = pics["paths"][min(pics["sel"])]
        else:
            path = filedialog.askopenfilename(
                parent=editor.win, title="A photo with the person in it", filetypes=[
                    ("Pictures", "*.png *.jpg *.jpeg *.webp"), ("All files", "*.*")])
            if not path:
                return
        rec = editor.records[editor.current]
        editor.status("Opening the photo" + ELLIPSIS)

        def later(fn):
            self._post("call", lambda: editor.win.winfo_exists() and fn())

        def cut(found, box):
            try:
                data = self.studio.cut_person(found, box)
                tmp = os.path.join(self.studio.lib.root, "references", "_cutout.png")
                os.makedirs(os.path.dirname(tmp), exist_ok=True)
                with open(tmp, "wb") as f:
                    f.write(data)
                kept = self.studio.lib.keep_reference(tmp, rec.get("name") or "person")
                os.remove(tmp)
            except (ig.ComfyError, OSError) as e:
                later(lambda: editor.status("The cut-out failed: %s" % e, "err"))
                return

            def done():
                if kept in pics["paths"]:
                    pics["paths"].remove(kept)
                at = pics["paths"].index(path) if path in pics["paths"] else 0
                pics["paths"].insert(at, kept)
                pics["sel"] = set()
                editor._draw_paths(pics)
                editor.status("Cut out. Save keeps it.")
            later(done)

        def find(region):
            try:
                found = self.studio.find_people(path, region)
            except (ig.ComfyError, OSError) as e:
                later(lambda: editor.status(str(e), "err"))
                return
            boxes = found["boxes"]
            if not boxes:
                later(lambda: editor.status("SAM3 found nobody in %s." % (
                    "that crop" if region else "that photo"), "warn"))
            elif len(boxes) == 1:
                later(lambda: editor.status("One person; cutting them out" + ELLIPSIS))
                cut(found, boxes[0])
            else:
                later(lambda: self._choose_person(editor, found, lambda b: (
                    editor.status("Cutting them out" + ELLIPSIS),
                    self.host._spawn(self.s.event_id, cut, found, b))))

        def cropped(region):
            editor.status("Finding the people in the %s" % ("crop" if region else "photo")
                          + ELLIPSIS)
            self.host._spawn(self.s.event_id, find, region)

        def look():
            try:
                photo = self.studio.look_at(path)
            except (ig.ComfyError, OSError) as e:
                later(lambda: editor.status(str(e), "err"))
                return
            later(lambda: (editor.status("Drag round the part of the photo to look in."),
                           self._crop_photo(editor, photo, cropped)))
        self.host._spawn(self.s.event_id, look)

    def _show_found(self, parent, found, name, title, heading):
        """A Toplevel over `parent` called `title`, `heading` above found's
        preview (written to references/`name` for Tk) on a canvas;
        -> (window, canvas, k), k the canvas px per picture px."""
        top = tk.Toplevel(parent)
        top.title(title)
        top.transient(parent)
        self.skin(top, bg="bg")
        self.label(top, heading, "text").pack(side="top", fill="x", padx=self.px(12),
                                              pady=(self.px(10), 0))
        img = None
        if found["preview"]:
            tmp = os.path.join(self.studio.lib.root, "references", name)
            try:
                os.makedirs(os.path.dirname(tmp), exist_ok=True)
                with open(tmp, "wb") as f:
                    f.write(found["preview"])
                img = photo_at(tmp, self.px(640), top)
            except OSError:
                img = None
        k = img.width() / float(found["width"]) if img else self.px(640) / float(
            max(found["width"], found["height"]))
        cw, ch = int(found["width"] * k), int(found["height"] * k)
        cv = tk.Canvas(top, width=cw, height=ch, bd=0, highlightthickness=0)
        self.skin(cv, bg="card")
        cv.pack(side="top", padx=self.px(12), pady=self.px(10))
        if img:
            top._img = img
            cv.create_image(0, 0, image=img, anchor="nw")
        return top, cv, k

    def _crop_photo(self, editor, photo, then):
        """A window with the photo: a drag marks the part to look in, the
        rest dimmed. `then` gets its crop_region, or None for the whole
        photo; Cancel or closing the window calls nothing."""
        top, cv, k = self._show_found(editor.win, photo, "_photo.png", "Crop the photo",
                                      "Drag round the person, or use the whole photo.")
        cv.configure(cursor="crosshair")
        cw, ch = int(cv["width"]), int(cv["height"])
        accent = self.host.C["accent"]
        drag = {"from": None, "to": None}
        # Four dimming panels round the crop, and its outline; moved on each drag.
        shade = [cv.create_rectangle(0, 0, 0, 0, fill="black", outline="", stipple="gray50",
                                     state="hidden") for _ in range(4)]
        edge = cv.create_rectangle(0, 0, 0, 0, outline=accent, width=self.px(2),
                                   state="hidden")

        def region():
            if not drag["from"] or not drag["to"]:
                return None
            (x0, y0), (x1, y1) = drag["from"], drag["to"]
            return ig.crop_region(x0 / k, y0 / k, x1 / k, y1 / k,
                                  photo["width"], photo["height"])

        def draw():
            r = region()
            state = "normal" if r else "hidden"
            if r:
                x0, y0 = r["x"] * k, r["y"] * k
                x1, y1 = x0 + r["width"] * k, y0 + r["height"] * k
                for item, box in zip(shade, ((0, 0, cw, y0), (0, y1, cw, ch),
                                             (0, y0, x0, y1), (x1, y0, cw, y1))):
                    cv.coords(item, *box)
                cv.coords(edge, x0, y0, x1, y1)
            for item in shade + [edge]:
                cv.itemconfigure(item, state=state)
            use.set(state="normal" if r else "disabled")

        def press(ev):
            drag["from"], drag["to"] = (ev.x, ev.y), None
            draw()

        def move(ev):
            drag["to"] = (min(max(ev.x, 0), cw), min(max(ev.y, 0), ch))
            draw()

        def finish(r):
            top.destroy()
            then(r)
        cv.bind("<ButtonPress-1>", press)
        cv.bind("<B1-Motion>", move)
        cv.bind("<ButtonRelease-1>", move)
        row = self.frame(top)
        row.pack(side="top", fill="x", padx=self.px(12), pady=(0, self.px(10)))
        use = self.button(row, "Use crop", lambda: region() and finish(region()))
        use.pack(side="right")
        self.button(row, "Whole photo", lambda: finish(None), kind="ghost").pack(
            side="right", padx=(0, self.px(4)))
        self.button(row, "Cancel", top.destroy, kind="ghost").pack(side="left")
        top.bind("<Return>", lambda ev: region() and finish(region()))
        top.bind("<Escape>", lambda ev: top.destroy())
        draw()
        return top

    def _choose_person(self, editor, found, then):
        """A window with the photo (or its crop) and a numbered box round
        each person; a click picks the one under it."""
        top, cv, k = self._show_found(
            editor.win, found, "_people.png", "Who is it?",
            "%d people. Click the one this identity is." % len(found["boxes"]))
        cv.configure(cursor="hand2")
        accent = self.host.C["accent"]
        for n, (x, y, w, h) in enumerate(found["boxes"], 1):
            cv.create_rectangle(x * k, y * k, (x + w) * k, (y + h) * k, outline=accent,
                                width=self.px(2))
            cv.create_text(x * k + self.px(6), y * k + self.px(4), text=str(n), anchor="nw",
                           fill=accent, font=self.host.f_ui)

        def click(ev):
            box = ig.pick_box(found["boxes"], ev.x / k, ev.y / k)
            top.destroy()
            then(box)
        cv.bind("<Button-1>", click)

    def edit_characters(self, looks=None):
        """The character creator; `looks` starts a new character from them."""
        return CharacterCreator(self, looks)

    def save_as_character(self):
        return self.edit_characters(dict(self.collect_looks(), item_refs=self.item_refs))

    def edit_styles(self):
        loras = [("", "none")] + [(r["id"], "%s (%s)" % (r["name"], r["category"]))
                                  for r in self.studio.lib.all("loras")]
        return RecordEditor(self, "styles", "Styles", [
            ("name", "Name", "text"),
            ("lora", "Style LoRA", ("choice", loras)),
            ("trigger", "Trigger phrase", "text"),
            ("strength", "Default LoRA strength", "number"),
            ("prompt", "Prompt additions", "long"),
            ("negative", "Negative additions", "long"),
            ("sampler", "Sampler", "text"),
            ("scheduler", "Scheduler", "text"),
            ("guidance", "Guidance", "number"),
            ("steps", "Steps", "number"),
            ("width", "Width", "number"),
            ("height", "Height", "number"),
            ("families", "Written for", ("multi", list(ig.FAMILIES.items()))),
            ("example", "Example picture (PNG; the tile on the form)", "path"),
            ("notes", "Notes", "long"),
        ], template={"name": "New style"})

    def edit_camera_profiles(self):
        """Cameras for the Scene Builder's Camera body picker: a name, its
        chemistry (film stock or digital colour science, in words), and
        optionally the native lens it sets and a picture of the camera for
        its button."""
        return RecordEditor(self, "camera_profiles", "Cameras", [
            ("name", "Name", "text"),
            ("chemistry", "Chemistry (film stock or colour science, in words)", "long"),
            ("lens", "Native lens (mm; sets the scene's lens when chosen)", "number"),
            ("format", "Frame shape: 1:1 or 3:2 (sets the scene's frame when chosen)", "text"),
            ("image", "Picture of the camera (PNG; the button in Scene Builder)", "path"),
            ("notes", "Notes", "long"),
        ], template={"name": "New camera"})

    def can_close(self):
        sb = self.scene_builder
        if sb is not None and sb.win.winfo_exists():
            return sb._ask_save()     # release() -> close() writes the recovery copy
        return True

    def release(self, confirmed=False):
        """On the UI thread, before the tab's frame goes (`Chat._close_tab`,
        `Chat._quit`): the Scene Builder writes into this form, so it cannot
        outlive it. `close()` runs on a worker thread and must not touch Tk."""
        sb = self.scene_builder
        try:
            if sb is not None and sb.win.winfo_exists():
                if not sb.close(final=True, confirmed=confirmed):
                    return False
        except tk.TclError:
            pass
        self.scene_builder = None
        return True

    def close(self):
        self.studio.close()


class ImageLibraryWindow:
    """Every picture you have generated, newest first, with explicit roles in
    the existing generation plan - so a past result becomes a new source,
    style or pose reference without any separate saving step. With `pick`
    it is a chooser instead: the selected picture's path goes to `pick` and
    the window closes (the Blend window's Library buttons)."""

    def __init__(self, owner, kind="source", pick=None):
        self.owner, self.studio, self.pick = owner, owner.studio, pick
        self.records = []
        self.image = None
        o = owner
        self.win = win = tk.Toplevel(o.host)
        win.title("Image library")
        win.transient(o.host)
        o.skin(win, bg="bg")
        win.geometry("%dx%d" % (o.px(780), o.px(580)))
        bar = o.frame(win)
        bar.pack(fill="x", padx=o.px(12), pady=o.px(12))
        self.search = tk.StringVar(master=win)
        entry = o.host._entry(bar, self.search)
        entry.master.pack(side="left", fill="x", expand=True)
        self.search.trace_add("write", lambda *_: self.reload())
        o.label(win, "Search by prompt. Everything you have generated is here.",
                "muted", o.host.f_small).pack(fill="x", padx=o.px(12))
        body = o.frame(win)
        body.pack(fill="both", expand=True, padx=o.px(12), pady=o.px(12))
        left = o.frame(body)
        left.pack(side="left", fill="both", expand=True)
        scroll = tk.Scrollbar(left)
        scroll.pack(side="right", fill="y")
        self.listbox = tk.Listbox(left, exportselection=False, bd=0, highlightthickness=0,
                                 font=o.host.f_ui, yscrollcommand=scroll.set)
        o.skin(self.listbox, bg="card", fg="text", selectbackground="sel", selectforeground="text")
        self.listbox.pack(fill="both", expand=True)
        scroll.config(command=self.listbox.yview)
        self.listbox.bind("<<ListboxSelect>>", lambda _: self.preview())
        right = o.frame(body)
        right.pack(side="left", fill="both", padx=(o.px(12), 0))
        self.picture = o.label(right, "Select an image", "muted", width=30)
        self.picture.config(anchor="center")
        self.picture.pack(fill="both", expand=True)
        peek(self.picture, lambda: (self.selected() or {}).get("path") if self.image else None)
        self.detail = o.label(right, "", "muted", o.host.f_small, wraplength=o.px(280))
        self.detail.pack(fill="x", pady=o.px(8))
        foot = o.frame(win)
        foot.pack(fill="x", padx=o.px(12), pady=(0, o.px(12)))
        labels = [label for _, label, _ in ig.REFERENCE_KINDS]
        self.role = tk.StringVar(master=win, value=dict((k, l) for k, l, _ in ig.REFERENCE_KINDS)[kind])
        if pick is None:
            o.label(foot, "Use as").pack(side="left")
            menu = tk.OptionMenu(foot, self.role, *labels)
            o.skin(menu, bg="card", fg="text", activebackground="sel", activeforeground="text")
            menu.pack(side="left", padx=o.px(8))
            o.button(foot, "Use selected image", self.use, kind="accent").pack(side="right")
            o.button(foot, "Blend" + ELLIPSIS, self.blend).pack(side="right",
                                                                padx=(0, o.px(4)))
        else:
            o.button(foot, "Use this picture", self.use, kind="accent").pack(side="right")
        self.message = o.label(win, "", "muted", o.host.f_small, wraplength=o.px(730))
        self.message.pack(fill="x", padx=o.px(12), pady=(0, o.px(12)))
        self.reload()

    def status(self, text, role="muted"):
        self.message.config(text=text)
        self.owner.skin(self.message, bg="bg", fg=role)

    def selected(self):
        sel = self.listbox.curselection()
        return self.records[sel[0]] if sel else None

    def reload(self, rid=None):
        current = self.selected()
        rid = rid or (current or {}).get("id")
        query = self.search.get().strip().casefold()
        self.records = []
        for rec in self.studio.history.list():
            prompt = rec.get("prompt") or ""
            if query and query not in prompt.casefold():
                continue
            images = rec.get("images") or []
            for i, path in enumerate(images):
                name = self.owner.clip(prompt, 70) or "(no prompt)"
                if len(images) > 1:
                    name += " (%d/%d)" % (i + 1, len(images))
                self.records.append({"id": "%s_%d" % (rec["id"], i), "name": name,
                                     "path": path,
                                     "when": (rec.get("created") or "")[5:16].replace("T", " ")})
        self.listbox.delete(0, "end")
        for r in self.records:
            missing = "" if os.path.isfile(r["path"]) else " (missing)"
            self.listbox.insert("end", (r["when"] + "  " + r["name"] + missing).strip())
        if self.records:
            at = next((i for i, r in enumerate(self.records) if r["id"] == rid), 0)
            self.listbox.selection_set(at)
            self.listbox.see(at)
        self.preview()

    def preview(self):
        rec = self.selected()
        self.image = photo(rec["path"], self.owner.px(280), profile=True) if rec else None
        self.picture.config(image=self.image or "", width=0 if self.image else 30,
                            text="" if self.image else "Preview unavailable" if rec else
                            "Generate an image to get started")
        self.detail.config(text=rec["name"] if rec else "")

    def use(self):
        rec = self.selected()
        if rec is None:
            return self.status("Select an image first.")
        if not os.path.isfile(rec["path"]):
            return self.status("This image is missing.", "err")
        if self.pick is not None:
            self.win.destroy()
            return self.pick(rec["path"])
        kind = next(k for k, label, _ in ig.REFERENCE_KINDS if label == self.role.get())
        self.owner.use_library_image(kind, rec["path"])
        self.status("Selected as %s. Check References, then Generate." % self.role.get(), "ok")

    def blend(self):
        """The Blend window, with the selected picture as its first."""
        rec = self.selected()
        if rec is not None and not os.path.isfile(rec["path"]):
            return self.status("This image is missing.", "err")
        self.owner.blend(rec["path"] if rec else None)


class BlendWindow:
    """Blend anywhere (`blend`): two pictures - the library's, a file, the
    one shown in the tab - into a new one. Blend queues a job like
    Generate's (`blend.submit`), so the picture arrives in Queue and is kept
    in History with both pictures, the words and the seed; Generate Again
    remakes it and Reuse settings opens this window on it. The identity
    editor's Blend (`NewPhotos`) is the other way in: there the picked
    blends join the person's references."""
    SIDE = 220

    def __init__(self, owner):
        self.owner = o = owner
        host = o.host
        self.paths = [None, None]
        self.win = win = tk.Toplevel(host)
        win.title("Blend")
        win.transient(host)
        host._skin(win, bg="bg")
        win.geometry("%dx%d" % (host._px(560), host._px(490)))
        foot = o.frame(win)
        foot.pack(side="bottom", fill="x", padx=o.px(12), pady=o.px(12))
        o.button(foot, "Blend", self.start, kind="accent").pack(side="right")
        o.button(foot, "Swap", self.swap, kind="ghost").pack(side="right", padx=(0, o.px(4)))
        self.msg = o.label(foot, "", "muted", host.f_small, wraplength=o.px(360))
        self.msg.pack(side="left", fill="x", expand=True)
        pair = o.frame(win)
        pair.pack(side="top", fill="x", padx=o.px(12), pady=(o.px(12), 0))
        self.slots = [self._slot(pair, i) for i in (0, 1)]
        o.label(win, "The new picture takes picture 1's shape, at about a megapixel.",
                "faint", host.f_small).pack(side="top", anchor="w", padx=o.px(12),
                                            pady=(o.px(6), 0))
        row = o.frame(win)
        row.pack(side="top", fill="x", padx=o.px(12), pady=(o.px(10), 0))
        self.person = tk.BooleanVar(master=win, value=False)
        o.switch(row, self.person).pack(side="left")
        o.label(row, "The same person is in both: keep their face", "text").pack(
            side="left", padx=(o.px(8), 0))
        o.label(win, "Anything to add (optional)", "muted", host.f_small).pack(
            side="top", anchor="w", padx=o.px(12), pady=(o.px(10), o.px(2)))
        self.words = tk.StringVar(master=win)
        host._entry(win, self.words).master.pack(side="top", fill="x", padx=o.px(12))
        self.status("Choose two pictures, then Blend.")

    def _slot(self, parent, i):
        """One picture's place: its square, its name, and where to take it
        from. -> {"pic", "name"}."""
        o, host, side = self.owner, self.owner.host, self.owner.px(self.SIDE)
        col = o.frame(parent)
        col.pack(side="left", expand=True)
        o.label(col, "Picture %d" % (i + 1), "muted", host.f_small).pack(side="top", anchor="w")
        box = tk.Frame(col, width=side, height=side, bd=0, highlightthickness=0)
        o.skin(box, bg="card")
        box.pack_propagate(False)
        box.pack(side="top")
        pic = tk.Label(box, bd=0, highlightthickness=0, text="None yet", font=host.f_small)
        o.skin(pic, bg="card", fg="faint")
        pic.pack(expand=True)
        peek(pic, lambda: self.paths[i])
        name = o.label(col, "", "faint", host.f_small, wraplength=side)
        name.pack(side="top", anchor="w")
        btns = o.frame(col)
        btns.pack(side="top", anchor="w", pady=(o.px(4), 0))
        o.button(btns, "Library" + ELLIPSIS, lambda: self.from_library(i)).pack(side="left")
        o.button(btns, "File" + ELLIPSIS, lambda: self.from_file(i)).pack(
            side="left", padx=(o.px(4), 0))
        return {"pic": pic, "name": name}

    def status(self, text, role="muted"):
        if self.win.winfo_exists():
            self.msg.config(text=text)
            self.owner.skin(self.msg, bg="bg", fg=role)

    def take(self, first=None, settings=None):
        """Fill the window: a blend's `settings` whole, else `first` into
        the first place that is empty (the first when neither is)."""
        if settings is not None:
            d = sb.clean_blend(settings.get("blend"))
            self.paths = (d["images"] + [None, None])[:2]
            self.person.set(d["person"])
            self.words.set(d["words"])
        elif first:
            self.paths[1 if self.paths[0] and not self.paths[1] else 0] = first
        self.draw()

    def set(self, i, path):
        if self.win.winfo_exists() and path:
            self.paths[i] = path
            self.draw()
            self.win.lift()

    def from_library(self, i):
        ImageLibraryWindow(self.owner, pick=lambda path: self.set(i, path))

    def from_file(self, i):
        self.set(i, filedialog.askopenfilename(parent=self.win, filetypes=[
            ("Pictures", "*.png *.jpg *.jpeg *.webp *.bmp"), ("All files", "*.*")]))

    def swap(self):
        self.paths.reverse()
        self.draw()

    def draw(self):
        o = self.owner
        for slot, path in zip(self.slots, self.paths):
            img = photo(path, o.px(self.SIDE)) if path and os.path.isfile(path) else None
            if img is not None:
                o.keep.append(img)
            slot["pic"].image = img
            slot["pic"].config(image=img or "", text="" if img else (
                "None yet" if not path else "No preview" if os.path.isfile(path)
                else "Missing"))
            slot["name"].config(text=os.path.basename(path) if path else "")

    def settings(self):
        """The job Blend sends: a new seed each time."""
        return {"mode": "blend", "seed": -1, "backend": "auto",
                "blend": {"images": [p for p in self.paths if p],
                          "person": bool(self.person.get()), "words": self.words.get()}}

    def start(self):
        o, s = self.owner, self.settings()
        wrong = sb.problem(sb.clean_blend(s["blend"]))
        if wrong:
            return self.status(wrong)
        if o.lora_build is not None:
            return self.status("A LoRA is being built and has the GPU; wait for it.", "err")
        self.status("Sent. It is in the Queue now, and in History once it is made; "
                    "Blend again for another.", "ok")
        o._show_list("queue")
        o.host._spawn(o.s.event_id, o._submit, s)


class NewPhotos:
    """Angles or Blend (`blend`) for the identity editor: new photos of the
    person drawn on the Kontext backend, shown as they come; a click picks
    one, Add puts the picked ones into the references (Save keeps them).
    Again makes another round; closing the window stops the run.
    Angles asks first: the views are picked on a view cube (`viewcube`),
    kept as the preset for next time, and Make draws each selected photo
    from each of them."""
    PER_PHOTO = 4                 # the views Surprise me picks

    def __init__(self, editor, pics, mode, parents):
        self.editor, self.pics, self.mode, self.parents = editor, pics, mode, parents
        o = self.owner = editor.owner
        host = o.host
        self.made, self.sel, self.stop = [], set(), threading.Event()
        self.running = False
        self.cube = None
        self.folder = tempfile.mkdtemp(prefix="studio-%s-" % mode)
        win = self.win = tk.Toplevel(editor.win)
        win.title("Blend" if mode == "blend" else "Angles")
        win.transient(editor.win)
        host._skin(win, bg="bg")
        win.geometry("%dx%d" % (host._px(640 if mode == "blend" else 860),
                                host._px(560 if mode == "blend" else 600)))
        win.protocol("WM_DELETE_WINDOW", self.close)
        top = o.frame(win)
        top.pack(side="top", fill="x", padx=o.px(12), pady=(o.px(12), 0))
        o.label(top, "Blending" if mode == "blend" else "From", "muted", host.f_small).pack(
            side="left", padx=(0, o.px(6)))
        for p in parents[:6]:
            img = photo(p, o.px(64), profile=True)
            if img is not None:
                o.keep.append(img)
                peek(tk.Label(top, image=img, bd=0), p).pack(side="left", padx=o.px(2))
        foot = o.frame(win)
        foot.pack(side="bottom", fill="x", padx=o.px(12), pady=o.px(12))
        o.button(foot, "Add to references", self.add, kind="accent").pack(side="right")
        o.button(foot, "Again" if mode == "blend" else "Make", self.start).pack(
            side="right", padx=(0, o.px(4)))
        o.button(foot, "Stop", self.stop.set, kind="ghost").pack(
            side="right", padx=(0, o.px(4)))
        self.msg = o.label(foot, "", "muted", host.f_small)
        self.msg.pack(side="left", fill="x", expand=True)
        if mode != "blend":
            self._build_cube(win)
        outer, self.grid = o.scrolled(win)
        outer.pack(side="top", fill="both", expand=True, padx=o.px(12), pady=(o.px(8), 0))
        if mode == "blend":
            self.start()
        else:
            self.status("Click the sides of the cube to look from, then Make. "
                        "Drag it to turn it.")

    def _build_cube(self, win):
        """The view cube down the left, with what is picked under it."""
        o, host = self.owner, self.owner.host
        side = o.frame(win)
        side.pack(side="left", fill="y", padx=(o.px(12), 0), pady=(o.px(8), 0))
        o.label(side, "Look from", "muted", host.f_small).pack(side="top", anchor="w")
        self.cube = viewcube.ViewCube(side, lambda: host.C, o.px(200), sb.load_views(),
                                      self._views_changed, font=host.f_small)
        self.cube.pack(side="top", pady=(o.px(4), 0))
        o.label(side, "Right and left are theirs.", "faint", host.f_small).pack(
            side="top", anchor="w")
        row = o.frame(side)
        row.pack(side="top", fill="x", pady=(o.px(6), 0))
        o.button(row, "Surprise me", lambda: self.cube.set_chosen(
            sb.pick_angles(self.PER_PHOTO))).pack(side="left")
        o.button(row, "Clear", lambda: self.cube.set_chosen([]), kind="ghost").pack(
            side="left", padx=(o.px(4), 0))
        o.button(row, "Turn back", self.cube.home, kind="ghost").pack(
            side="left", padx=(o.px(4), 0))
        self.picked = o.label(side, "", "text", host.f_small, wraplength=o.px(200))
        self.picked.pack(side="top", anchor="w", pady=(o.px(6), 0))
        self._views_changed(self.cube.chosen, save=False)

    def _views_changed(self, names, save=True):
        n = len(names) * len(self.parents)
        self.picked.config(text="\n".join(names) + (
            "\n\n%d photo%s a round" % (n, "" if n == 1 else "s") if names
            else "Nothing picked yet."))
        if save:
            try:
                sb.save_views(names)
            except OSError:
                pass                  # a preset is a convenience; the pick still stands

    def status(self, text, role="muted"):
        if self.win.winfo_exists():
            self.msg.config(text=text)
            self.owner.skin(self.msg, bg="bg", fg=role)

    def jobs(self):
        """[(label, parent, angle or None)] for one round."""
        if self.mode == "blend":
            return [("blend", None, None)]
        views = self.cube.chosen if self.cube is not None else sb.pick_angles(self.PER_PHOTO)
        return [(angle, p, angle) for p in self.parents for angle in views]

    def start(self):
        if self.running:
            return self.status("Still making the last round; Stop ends it.")
        jobs, studio = self.jobs(), self.owner.studio
        if not jobs:
            return self.status("Pick at least one side of the cube first.")
        self.running = True
        self.stop.clear()
        self.status("Finding a backend with FLUX Kontext" + ELLIPSIS)

        def say(text, role="muted"):
            self.owner._post("call", lambda: self.status(text, role))

        def work():
            error = None
            try:
                backend, why = sb.route(studio)
                if backend is None:
                    raise ig.ComfyError(why)
                client = studio.client(backend)
                names = {p: client.upload_image(p) for p in self.parents}
                size = sb.size_for(self.parents[0])
                for n, (label, parent, angle) in enumerate(jobs):
                    if self.stop.is_set():
                        break
                    seed = random.randint(0, ig.MAX_SEED)
                    graph = (sb.blend_graph(names[self.parents[0]], names[self.parents[1]],
                                            size, seed) if angle is None
                             else sb.angle_graph(names[parent], angle, seed))
                    head = "%s %d of %d on %s" % ("Blending" if angle is None else
                                                  "Angle: " + angle, n + 1, len(jobs),
                                                  backend["name"])
                    say(head + ELLIPSIS)
                    data = sb.run(client, graph, self.stop.is_set,
                                  lambda v, t, head=head: say("%s · step %d of %d"
                                                              % (head, v, t)))
                    if data is None:
                        break
                    path = os.path.join(self.folder, "%s_%d.png" % (
                        re.sub(r"\W+", "-", label), seed))
                    with open(path, "wb") as f:
                        f.write(data)
                    self.owner._post("call", lambda p=path, l=label: self.arrived(p, l))
            except (ig.ComfyError, OSError) as e:
                error = str(e)

            def done():
                self.running = False
                if error:
                    self.status(error, "err")
                elif self.stop.is_set():
                    self.status("Stopped.")
                else:
                    self.status("Click the ones to keep, then Add to references.", "ok")
            self.owner._post("call", done)
        self.owner.host._spawn(self.owner.s.event_id, work)

    def arrived(self, path, label):
        self.made.append((path, label))
        self.draw()

    def draw(self, cols=4):
        if not self.win.winfo_exists():
            return
        o, host, side = self.owner, self.owner.host, self.owner.px(140)
        for w in self.grid.winfo_children():
            w.destroy()
        for i, (path, label) in enumerate(self.made):
            tile = tk.Frame(self.grid, bd=0, highlightthickness=o.px(3))
            ring = "accent" if i in self.sel else "bg"
            o.skin(tile, bg="card", highlightbackground=ring, highlightcolor=ring)
            tile.grid(row=i // cols, column=i % cols, padx=o.px(2), pady=o.px(2))
            img = photo(path, side)
            if img is not None:
                o.keep.append(img)
                lbl = peek(tk.Label(tile, image=img, bd=0, width=side, height=side), path)
            else:                          # sizes in characters without an image
                lbl = tk.Label(tile, text="no preview", bd=0, font=host.f_small,
                               width=12, height=6)
            o.skin(lbl, bg="card", fg="muted")
            lbl.pack()
            cap = o.label(tile, label, "muted", host.f_small, bg="card")
            cap.pack()
            for w in (tile, lbl, cap):
                w.bind("<Button-1>", lambda ev, i=i: self.toggle(i))

    def toggle(self, i):
        self.sel ^= {i}
        self.draw()

    def add(self):
        picked = [self.made[i][0] for i in sorted(self.sel)]
        if not picked:
            return self.status("Click the photos to keep first.")
        if not self.pics["grid"].winfo_exists():
            return self.status("The person's form was closed; open it again.", "err")
        self.editor._import_paths(self.pics, picked)
        self.made = [m for i, m in enumerate(self.made) if i not in self.sel]
        self.sel = set()
        self.draw()
        self.status("Added %d. Save the person to keep them." % len(picked), "ok")

    def close(self):
        self.stop.set()
        self.win.destroy()
        if not self.running and not self.pics.get("importing"):
            shutil.rmtree(self.folder, ignore_errors=True)


class RecordEditor:
    """One editor for every list the studio keeps: the records on the left,
    a form built from `fields` on the right. `fields` is (key, label, kind)
    with kind one of text, long, number, bool, path, paths, kv, per_backend,
    per_backend_file, ("choice", pairs) or ("multi", pairs)."""

    def __init__(self, owner, kind, title, fields, template, extra=None, label=None,
                 info=None):
        self.owner, self.kind, self.fields = owner, kind, fields
        self.info = info              # rec -> [(text, role)], shown above the fields
        self.template, self.label_of = template, label or (
            lambda r: r.get("name") or r.get("label") or r.get("id"))
        host = owner.host
        self.records = [dict(r) for r in owner.studio.lib.all(kind)]
        self.current = None
        win = self.win = tk.Toplevel(host)
        win.title(title)
        win.transient(host)
        host._skin(win, bg="bg")
        win.geometry("%dx%d" % (host._px(820), host._px(620)))
        left = owner.frame(win)
        left.pack(side="left", fill="y", padx=owner.px(12), pady=owner.px(12))
        self.lb = tk.Listbox(left, width=30, bd=0, highlightthickness=0, activestyle="none",
                             font=host.f_ui, exportselection=False)
        host._skin(self.lb, bg="card", fg="text", selectbackground="sel",
                   selectforeground="text")
        self.lb.pack(side="top", fill="y", expand=True)
        self.lb.bind("<<ListboxSelect>>", lambda ev: self._pick())
        btns = owner.frame(left)
        btns.pack(side="top", fill="x", pady=(owner.px(8), 0))
        owner.button(btns, "New", self._new).pack(side="left")
        owner.button(btns, "Duplicate", self._dup).pack(side="left", padx=(owner.px(4), 0))
        owner.button(btns, "Delete", self._delete, kind="ghost").pack(
            side="left", padx=(owner.px(4), 0))
        for text, fn in ([extra] if isinstance(extra, tuple) else extra or []):
            owner.button(left, text, lambda fn=fn: fn(self)).pack(
                side="top", anchor="w", pady=(owner.px(8), 0))
        right = owner.frame(win)
        right.pack(side="left", fill="both", expand=True, pady=owner.px(12),
                   padx=(0, owner.px(12)))
        foot = owner.frame(right)
        foot.pack(side="bottom", fill="x", pady=(owner.px(8), 0))
        owner.button(foot, "Save", self._save, kind="accent").pack(side="right")
        self.msg = owner.label(foot, "", "muted", host.f_small)
        self.msg.pack(side="left", fill="x", expand=True)
        outer, self.form = owner.scrolled(right)
        outer.pack(side="top", fill="both", expand=True)
        self.widgets = {}
        self._reload_list(0 if self.records else None)

    def status(self, text, role="muted"):
        self.msg.config(text=text)
        self.owner.skin(self.msg, bg="bg", fg=role)

    def reload(self, select_id=None):
        self.records = [dict(r) for r in self.owner.studio.lib.all(self.kind)]
        at = next((i for i, r in enumerate(self.records) if r.get("id") == select_id), 0)
        self.current = None
        self._reload_list(at if self.records else None)
        self.status("Library updated.")

    def refresh(self):
        """Redraw the form - its status panel above all - keeping edits."""
        self._store()
        self._build_form()
        self.status("Checked.")

    def _reload_list(self, select):
        self.lb.delete(0, "end")
        for r in self.records:
            self.lb.insert("end", self.label_of(r))
        if select is not None and self.records:
            self.lb.selection_clear(0, "end")
            self.lb.selection_set(select)
            self.lb.see(select)
        self._pick()

    def _pick(self):
        if self.current is not None:
            self._store()
        sel = self.lb.curselection()
        self.current = sel[0] if sel else None
        self._build_form()

    # ----------------------------------------------------------------- form
    def _build_form(self):
        o, host = self.owner, self.owner.host
        for w in self.form.winfo_children():
            w.destroy()
        self.widgets = {}
        if self.current is None:
            o.label(self.form, "Nothing here yet. New adds one.", "faint").pack(
                side="top", anchor="w")
            return
        rec = self.records[self.current]
        backends = o.studio.backends()
        if self.info is not None:
            box = o.frame(self.form, "card")
            box.pack(side="top", fill="x", pady=(0, o.px(4)))
            for text, role in self.info(rec):
                lbl = o.label(box, text, role, host.f_small, bg="card",
                              wraplength=o.px(380))
                lbl.pack(side="top", fill="x", padx=o.px(8), pady=o.px(2))
                o.wrap(lbl, box, o.px(20))
        parent = self.form
        for key, label, kind in self.fields:
            if self.kind == 'identities' and key == 'lora':
                advanced = o.frame(self.form)
                def toggle(box=advanced):
                    if box.winfo_manager():
                        box.pack_forget()
                    else:
                        box.pack(side='top', fill='x')
                o.button(self.form, 'Advanced settings', toggle, kind='ghost').pack(side='top', anchor='w')
                parent = advanced
            o.label(parent, label, "muted", host.f_small).pack(
                side="top", fill="x", pady=(o.px(8), o.px(2)))
            val = rec.get(key)
            if kind in ("text", "number", "path"):
                var = tk.StringVar(value="" if val is None else str(val))
                row = o.frame(parent)
                row.pack(side="top", fill="x")
                if kind == "path":
                    o.button(row, "Choose…", lambda v=var: self._choose(v)).pack(
                        side="right", padx=(o.px(6), 0))
                    if self.kind == "identities":
                        o.button(row, "Link…", lambda v=var: self._link_path(v)).pack(
                            side="right", padx=(o.px(6), 0))
                e = host._entry(row, var)
                e.master.pack(side="left", fill="x", expand=True)
                self.widgets[key] = (kind, var)
                if kind == "path" and val and os.path.isfile(val):
                    img = photo(val, o.px(160), profile=self.kind == "identities")
                    if img is not None:
                        o.keep.append(img)
                        peek(tk.Label(parent, image=img, bd=0), val).pack(
                            side="top", anchor="w", pady=o.px(4))
            elif kind == "long":
                t = tk.Text(parent, height=8 if key == "description" else 3, wrap="word", bd=0, highlightthickness=0,
                            font=host.f_ui, padx=o.px(6), pady=o.px(4))
                o.skin(t, bg="card", fg="text", insertbackground="accent")
                t.insert("1.0", val or "")
                t.pack(side="top", fill="x")
                self.widgets[key] = (kind, t)
            elif kind == "bool":
                var = tk.BooleanVar(value=bool(val))
                o.switch(parent, var).pack(side="top", anchor="w")
                self.widgets[key] = (kind, var)
            elif isinstance(kind, tuple) and kind[0] == "choice":
                var = tk.StringVar(value=val or "")
                o.choice(parent, kind[1], val or "", var.set).pack(side="top", anchor="w")
                self.widgets[key] = ("choice", var)
            elif isinstance(kind, tuple) and kind[0] == "multi":
                have, vars_ = set(val or []), {}
                grid = o.frame(parent)
                grid.pack(side="top", fill="x")
                for i, (name, text) in enumerate(kind[1]):
                    var = tk.BooleanVar(value=name in have)
                    b = tk.Checkbutton(grid, text=text, variable=var, anchor="w",
                                       font=host.f_small, bd=0, highlightthickness=0)
                    o.skin(b, bg="bg", fg="text", activebackground="bg", selectcolor="card",
                           activeforeground="text")
                    b.grid(row=i // 2, column=i % 2, sticky="w")
                    vars_[name] = var
                self.widgets[key] = ("multi", vars_)
            elif kind == "paths":
                # A grid of thumbnails; a click selects one for Remove.
                pics = {"paths": list(val or []), "sel": set(), "grid": o.frame(parent)}
                pics["grid"].pack(side="top", fill="x")
                self._draw_paths(pics)
                row = o.frame(parent)
                row.pack(side="top", fill="x", pady=(o.px(4), 0))
                o.button(row, "Add photos…", lambda p=pics: self._add_paths(p)).pack(
                    side="left")
                if self.kind == "identities":
                    o.button(row, "Add folder…", lambda p=pics: self._add_folder(p)).pack(
                        side="left", padx=(o.px(4), 0))
                    o.button(row, "Add from link…", lambda p=pics: self._add_link(p)).pack(
                        side="left", padx=(o.px(4), 0))
                    actions = o.frame(parent)
                    actions.pack(side="top", fill="x")
                    o.button(actions, "Use as primary", lambda p=pics: self._primary_path(p)).pack(
                        side="left")
                    o.button(actions, "Build LoRA" + ELLIPSIS,
                             lambda: self.owner.build_lora(self)).pack(
                        side="left", padx=(o.px(4), 0))
                    o.button(actions, "Angles" + ELLIPSIS,
                             lambda p=pics: self._new_photos(p, "angles")).pack(
                        side="left", padx=(o.px(4), 0))
                    o.button(actions, "Blend" + ELLIPSIS,
                             lambda p=pics: self._new_photos(p, "blend")).pack(
                        side="left", padx=(o.px(4), 0))
                    o.label(parent, "Use clear photos of the same person, one face per photo.\n"
                            "Choose a clear front view as Primary. Enable pooling to let the "
                            "other photos guide identity too. Build LoRA trains the "
                            "person's own FLUX LoRA from %d or more photos.\n"
                            "Angles draws the selected photos from the sides you pick on "
                            "a view cube; Blend mixes two selected photos into a new "
                            "one." % lt.MIN_PHOTOS,
                            "muted", host.f_small).pack(
                                side="top", anchor="w")
                o.button(row, "Remove", lambda p=pics: self._remove_paths(p),
                         kind="ghost").pack(side="left", padx=(o.px(4), 0))
                self.widgets[key] = ("paths", pics)
            elif kind == "kv":
                t = tk.Text(parent, height=5, wrap="none", bd=0, highlightthickness=0,
                            font=host.f_mono, padx=o.px(6), pady=o.px(4))
                o.skin(t, bg="card", fg="text", insertbackground="accent")
                t.insert("1.0", "\n".join("%s = %s" % kv for kv in (val or {}).items()))
                t.pack(side="top", fill="x")
                self.widgets[key] = ("kv", t)
            elif kind in ("per_backend", "per_backend_file"):
                per = {}
                for b in backends:
                    row = o.frame(parent)
                    row.pack(side="top", fill="x", pady=(o.px(2), 0))
                    o.label(row, b["name"], "text", width=18).pack(side="left", anchor="n")
                    over = (val or {}).get(b["id"], {} if kind == "per_backend" else "")
                    if kind == "per_backend_file":
                        var = tk.StringVar(value=over or "")
                        e = host._entry(row, var)
                        e.master.pack(side="left", fill="x", expand=True)
                        per[b["id"]] = var
                        continue
                    absent = tk.BooleanVar(value=over is None)
                    cb = tk.Checkbutton(row, text="not on this machine", variable=absent,
                                        font=host.f_small, bd=0, highlightthickness=0)
                    o.skin(cb, bg="bg", fg="muted", activebackground="bg",
                           selectcolor="card", activeforeground="text")
                    cb.pack(side="right", anchor="n")
                    t = tk.Text(row, height=3, wrap="none", bd=0, highlightthickness=0,
                                font=host.f_mono, padx=o.px(6), pady=o.px(4))
                    o.skin(t, bg="card", fg="text", insertbackground="accent")
                    t.insert("1.0", "\n".join("%s = %s" % kv for kv in (over or {}).items()))
                    t.pack(side="left", fill="x", expand=True)
                    per[b["id"]] = (absent, t)
                self.widgets[key] = (kind, per)

    def _choose(self, var):
        path = filedialog.askopenfilename(parent=self.win, filetypes=[
            ("Pictures", "*.png *.jpg *.jpeg *.webp"), ("All files", "*.*")])
        if path:
            var.set(path)

    def _draw_paths(self, pics, cols=4):
        """The photos of a `paths` field as a grid of tiles; the selected ones
        framed in the accent colour. Tk previews PNG and GIF only, so any
        other file shows as its name on a blank tile."""
        o, host = self.owner, self.owner.host
        grid, side = pics["grid"], o.px(110)
        for w in grid.winfo_children():
            w.destroy()
        if not pics["paths"]:
            o.label(grid, "No photos yet.", "faint", host.f_small).grid(row=0, column=0,
                                                                        sticky="w")
            return
        for i, p in enumerate(pics["paths"]):
            tile = tk.Frame(grid, bd=0, highlightthickness=o.px(3))
            ring = "accent" if i in pics["sel"] else "bg"
            o.skin(tile, bg="card", highlightbackground=ring, highlightcolor=ring)
            tile.grid(row=i // cols, column=i % cols, padx=o.px(2), pady=o.px(2))
            img = photo(p, side, profile=self.kind == "identities") if os.path.isfile(p) else None
            if img is not None:
                o.keep.append(img)
                lbl = peek(tk.Label(tile, image=img, bd=0, width=side, height=side), p)
            else:
                name = os.path.basename(p) + ("" if os.path.isfile(p) else "\n(missing)")
                lbl = tk.Label(tile, text=name, bd=0, font=host.f_small,
                               wraplength=side - o.px(8), width=12, height=6)
            o.skin(lbl, bg="card", fg="muted")
            lbl.pack()
            if self.kind == "identities" and i == 0:
                o.label(tile, "Primary", "accent", host.f_small).pack()
            for w in (tile, lbl):
                w.bind("<Button-1>", lambda ev, i=i: self._toggle_path(pics, i))

    def _toggle_path(self, pics, i):
        pics["sel"] ^= {i}
        self._draw_paths(pics)

    def _remove_paths(self, pics):
        pics["paths"] = [p for i, p in enumerate(pics["paths"]) if i not in pics["sel"]]
        pics["sel"] = set()
        self._draw_paths(pics)

    def _add_paths(self, pics):
        paths = filedialog.askopenfilenames(parent=self.win, filetypes=[
            ("Pictures", "*.png *.jpg *.jpeg *.webp"), ("All files", "*.*")])
        self._import_paths(pics, paths)

    def _new_photos(self, pics, mode):
        """Angles of the selected photos, or a blend of the two selected."""
        chosen = [pics["paths"][i] for i in sorted(pics["sel"])
                  if os.path.isfile(pics["paths"][i])]
        if mode == "blend" and len(chosen) != 2:
            return self.status("Select exactly two photos to blend.")
        if mode == "angles" and not chosen:
            return self.status("Select the photos to see from other angles.")
        if self.owner.lora_build is not None:
            return self.status("A LoRA is being built and has the GPU; wait for it.", "err")
        NewPhotos(self, pics, mode, chosen)

    def _primary_path(self, pics):
        if len(pics["sel"]) != 1:
            self.status("Select one photo to use as Primary.")
            return
        index = next(iter(pics["sel"]))
        pics["paths"].insert(0, pics["paths"].pop(index))
        pics["sel"] = {0}
        self._draw_paths(pics)

    def _add_folder(self, pics):
        folder = filedialog.askdirectory(parent=self.win, title="Photos of one person")
        if folder:
            self._import_paths(pics, folder=folder)

    def _import_paths(self, pics, paths=(), folder=None):
        if pics.get("importing") or (not paths and not folder):
            return
        rec = self.records[self.current]
        owner = rec.get("name") or "person"
        existing = list(pics["paths"])
        pics["importing"] = True
        self._imports = getattr(self, "_imports", 0) + 1
        self.status("Importing reference photos" + ELLIPSIS)

        def done(result):
            pics["importing"] = False
            self._imports -= 1
            if not self.win.winfo_exists():
                return
            if not pics["grid"].winfo_exists():
                if any(r is rec for r in self.records):
                    refs = rec.setdefault("references", [])
                    refs.extend(p for p in result["added"] if p not in refs)
                    # The same record may have been opened again while copying.
                    if self.current is not None and self.records[self.current] is rec:
                        self._store()
                        rec["references"] = list(dict.fromkeys(rec["references"] + refs))
                        self._build_form()
                    self.status("Import finished for %s: %d added, %d duplicates, %d unreadable."
                                " Save to keep the changes." % (owner, len(result["added"]),
                                result["duplicates"], len(result["errors"])))
                return
            for path in result["added"]:
                if path not in pics["paths"]:
                    pics["paths"].append(path)
            self._draw_paths(pics)
            message = "%d added · %d duplicates · %d unreadable" % (
                len(result["added"]), result["duplicates"], len(result["errors"]))
            if result["errors"]:
                message += "\n" + "\n".join(result["errors"][:3])
            self.status(message, "err" if result["errors"] else "muted")

        def work():
            try:
                candidates = paths
                if folder:
                    with os.scandir(folder) as entries:
                        candidates = sorted((e.path for e in entries if e.is_file()
                            and os.path.splitext(e.name)[1].lower() in
                            (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")),
                            key=lambda p: (p.casefold(), p))
                result = self.owner.studio.lib.import_identity_photos(candidates, owner, existing)
            except OSError as exc:
                result = {"added": [], "duplicates": 0, "errors": [str(exc)]}
            self.owner._post("call", lambda: done(result))
        self.owner.host._spawn(self.owner.s.event_id, work)

    def _add_link(self, pics):
        """A photo of the person from a link, added to the list when it has
        come, as `_add_paths` adds a file."""
        rec = self.records[self.current]

        def then(path):
            if not pics["grid"].winfo_exists():     # another record's form now
                return
            if path not in pics["paths"]:
                pics["paths"].append(path)
            self._draw_paths(pics)
        return self.owner.from_link(self.win, "person", rec.get("name") or "person", then,
                                    self.status)

    def _link_path(self, var):
        rec = self.records[self.current]
        return self.owner.from_link(self.win, "person", rec.get("name") or "person",
                                    var.set, self.status)

    @staticmethod
    def _parse_kv(text):
        out = {}
        for line in text.splitlines():
            if "=" not in line:
                continue
            k, v = (x.strip() for x in line.split("=", 1))
            if not k:
                continue
            for cast in (int, float):
                try:
                    v = cast(v)
                    break
                except ValueError:
                    pass
            if v in ("true", "false"):
                v = v == "true"
            out[k] = v
        return out

    def _store(self):
        """The form back into the record it shows."""
        if self.current is None or self.current >= len(self.records):
            return
        rec = self.records[self.current]
        for key, (kind, w) in self.widgets.items():
            if kind in ("text", "path", "choice"):
                rec[key] = w.get().strip()
            elif kind == "number":
                raw = w.get().strip()
                try:
                    rec[key] = float(raw) if raw else None
                except ValueError:
                    pass
            elif kind == "long":
                rec[key] = w.get("1.0", "end").strip()
            elif kind == "bool":
                rec[key] = bool(w.get())
            elif kind == "multi":
                rec[key] = [n for n, v in w.items() if v.get()]
            elif kind == "paths":
                rec[key] = list(w["paths"])
            elif kind == "kv":
                rec[key] = self._parse_kv(w.get("1.0", "end"))
            elif kind == "per_backend_file":
                rec[key] = {bid: v.get().strip() for bid, v in w.items() if v.get().strip()}
            elif kind == "per_backend":
                rec[key] = {bid: (None if absent.get() else self._parse_kv(t.get("1.0", "end")))
                            for bid, (absent, t) in w.items()}
                rec[key] = {k: v for k, v in rec[key].items() if v is None or v}

    def _new(self):
        self._store()
        self.records.append(dict(self.template))
        self.current = None
        self._reload_list(len(self.records) - 1)

    def _dup(self):
        if self.current is None:
            return
        self._store()
        copy_ = dict(self.records[self.current])
        for k in ("name", "label"):
            if copy_.get(k):
                copy_[k] += " copy"
        copy_["id"] = (copy_.get("id") or "item") + "-copy"
        self.records.append(copy_)
        self.current = None
        self._reload_list(len(self.records) - 1)

    def _delete(self):
        if self.current is None:
            return
        del self.records[self.current]
        self.current = None
        self._reload_list(min(len(self.records) - 1, 0) if self.records else None)
        self.status("Deleted - Save to keep it deleted.", "warn")

    def _save(self):
        if getattr(self, "_imports", 0):
            self.status("Wait for the photo import to finish before saving.")
            return False
        self._store()
        try:
            self.owner.studio.lib.save(self.kind, self.records)
        except OSError as e:
            self.status("Could not save: %s" % e, "err")
            return
        idx = self.current
        self.records = [dict(r) for r in self.owner.studio.lib.all(self.kind)]
        self.current = None
        self._reload_list(idx if idx is not None and idx < len(self.records) else None)
        self.status("Saved.", "ok")
        self.owner._saved(self.kind)
        return True


class FixWindow:
    """Fix a spot: the picture large. A drag round a part draws a freehand
    outline, and only inside it is redrawn; a click marks a square instead.
    A click on a marked spot gives it a photo of what goes there (a hat, a
    hand), swapped in by Qwen-Image-Edit; a second click takes the photo off.
    The wheel sizes the square under the pointer
    (or the next one), a right-click takes one away. Redraw queues a fix job
    (studio_imagegen `run_fix`): only the squares change, the rest of the
    picture is kept pixel for pixel, and the result is a new picture in the
    history beside the old one.

    Find hands / face / accessories asks SAM3 to mark them in one click. In
    Lock mode a click (or Find) marks a square the fix may not change: the
    original is laid back over it last. An identity's face (the Face swap
    row) is swapped onto the picture's biggest face after everything else,
    and needs no spots.

    What is wrong is a note on the spot marked last (or, after Find, on
    them all): what the Visual Critic is asked after when Afterwards is
    "Critic checks it", and a spot still wrong is then redrawn again."""

    TARGETS = [("hand", "Hand"), ("face", "Face"), ("other", "Accessory / other")]
    STRENGTHS = [("light", "Light"), ("medium", "Medium"), ("strong", "Strong")]
    MODES = [("redraw", "Redraw"), ("lock", "Lock (keep as is)")]
    CHECKS = [("off", "Leave as made"), ("on", "Critic checks it")]
    FINDS = [("hand", "Find hands"), ("face", "Find face"), ("other", "Find accessories")]

    def __init__(self, owner, path, settings, around_head=False):
        self.owner = o = owner
        self.around_head = around_head
        host = owner.host
        self.path, self.settings = path, settings
        self.spots = []                   # [{"x", "y", "size"}] in the picture's pixels
        self.locks = []                   # the same, kept as they are
        self.target, self.strength, self.mode = "hand", "medium", "redraw"
        self.check = "off"                # "on": the Visual Critic looks at the result
        self.current = None               # the spot the note is on; None: on them all
        self.wide = ""                    # the note on them all
        if around_head:
            self.target, self.strength, self.mode = "other", "strong", "lock"
        self.finding = False
        self.hover = None
        self.face_point = None
        self.choosing_face = False
        self.img = None
        try:
            full = tk.PhotoImage(master=host, file=path)
            self.w, self.h = full.width(), full.height()
        except (tk.TclError, OSError):
            self.w = self.h = 0
        self.size = max(ig.FIX_MIN, int(max(self.w, self.h) * 0.16))
        win = self.win = tk.Toplevel(host)
        win.title("Generate around head" if around_head else "Fix a spot")
        win.transient(host)
        host._skin(win, bg="bg")
        win.geometry("%dx%d" % (host._px(860), host._px(820)))

        foot = o.frame(win)
        foot.pack(side="bottom", fill="x", padx=o.px(12), pady=(0, o.px(12)))
        self.go = o.button(foot, "Generate around head" if around_head else "Redraw",
                           self._redraw, kind="accent")
        self.go.pack(side="right")
        o.button(foot, "Clear", self._clear, kind="ghost").pack(side="right",
                                                               padx=(0, o.px(6)))
        self.msg = o.label(foot, "", "muted", host.f_small)
        self.msg.pack(side="left", fill="x", expand=True)

        opts = o.frame(win)
        opts.pack(side="bottom", fill="x", padx=o.px(12), pady=(o.px(8), o.px(8)))
        self.pills = {}
        row = o.frame(opts)
        row.pack(side="top", fill="x", pady=(0, o.px(4)))
        o.label(row, "Find", "muted", width=10).pack(side="left")
        self.find_btns = []
        for kind, text in self.FINDS:
            p = o.button(row, text, lambda k=kind: self._find(k), bg="card")
            p.pack(side="left", padx=(0, o.px(4)))
            self.find_btns.append(p)
        if around_head:
            row.pack_forget()
        for key, items, label in (("mode", self.MODES, "A click"),
                                  ("target", self.TARGETS, "Redraw"),
                                  ("strength", self.STRENGTHS, "Change"),
                                  ("check", self.CHECKS, "Afterwards")):
            row = o.frame(opts)
            row.pack(side="top", fill="x", pady=(0, o.px(4)))
            o.label(row, label, "muted", width=10).pack(side="left")
            for value, text in items:
                p = o.button(row, text, lambda k=key, v=value: self._pick(k, v),
                             kind="ghost")
                p.pack(side="left", padx=(0, o.px(4)))
                self.pills[(key, value)] = p
            if around_head and key != "strength":
                row.pack_forget()
        row = o.frame(opts)
        row.pack(side="top", fill="x")
        o.label(row, "Describe", "muted", width=10).pack(side="left")
        self.words = tk.StringVar()
        if around_head:
            self.words.set(settings.get("scene") or "")
        e = host._entry(row, self.words)
        e.master.pack(side="left", fill="x", expand=True)
        row = o.frame(opts)
        row.pack(side="top", fill="x", pady=(o.px(4), 0))
        o.label(row, "Wrong", "muted", width=10).pack(side="left")
        self.note = tk.StringVar()
        self.note.trace_add("write", lambda *_: self._noted())
        self.note_where = o.label(row, "", "faint", host.f_small)
        self.note_where.pack(side="right", padx=(o.px(6), 0))
        e = host._entry(row, self.note)
        e.master.pack(side="left", fill="x", expand=True)
        self._note_on(None)
        if around_head:
            row.pack_forget()
        row = o.frame(opts)
        row.pack(side="top", fill="x", pady=(o.px(4), 0))
        o.label(row, "Face swap", "muted", width=10).pack(side="left")
        self.face = ""                    # the identity whose face is swapped in last
        idents = [(i["id"], i["name"]) for i in o.studio.lib.all("identities")]
        self.face_pill = o.choice(row, [("", "None")] + idents, "", self._face_identity)
        self.face_pill.pack(side="left")
        o.button(row, "Choose face…", self._choose_face, kind="ghost").pack(
            side="left", padx=o.px(6))
        if not idents:
            o.label(row, "  No identities yet (Library > Identities).", "faint",
                    host.f_small).pack(side="left")
        if around_head:
            row.pack_forget()

        self.canvas = tk.Canvas(win, bd=0, highlightthickness=0, cursor="crosshair")
        host._skin(self.canvas, bg="card")
        self.canvas.pack(side="top", fill="both", expand=True, padx=o.px(12),
                         pady=(o.px(12), 0))
        self.canvas.bind("<Configure>", lambda ev: self._layout())
        self.canvas.bind("<Button-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<ButtonRelease-1>", self._release)
        self.canvas.bind("<Button-3>", self._remove)
        self.canvas.bind("<Motion>", self._move)
        self.canvas.bind("<Leave>", lambda ev: self._move(None))
        self.canvas.bind("<MouseWheel>", self._wheel)
        self._paint_pills()
        self._status()

    # ------------------------------------------------------------- picture
    def _layout(self):
        """Fit the picture to the canvas; `k` is screen px per picture px."""
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if not self.w or cw < 20 or ch < 20:
            return
        side = min(cw, ch * self.w / float(self.h)) if self.w >= self.h else \
            min(ch, cw * self.h / float(self.w))
        self.img = photo_at(self.path, int(side), self.owner.host)
        if self.img is None:
            return
        self.k = self.img.width() / float(self.w)
        self.ox = (cw - self.img.width()) // 2
        self.oy = (ch - self.img.height()) // 2
        self._draw()

    def _to_pic(self, ev):
        if self.img is None:
            return None
        x, y = (ev.x - self.ox) / self.k, (ev.y - self.oy) / self.k
        if 0 <= x < self.w and 0 <= y < self.h:
            return int(x), int(y)
        return None

    def _marks(self):
        return self.locks if self.mode == "lock" else self.spots

    def _under(self, at, marks=None):
        """The index of the square `at` (picture px) is inside, or None -
        among the marks of the current mode unless `marks` is given."""
        marks = self._marks() if marks is None else marks
        for i in range(len(marks) - 1, -1, -1):
            sp = marks[i]
            if abs(sp["x"] - at[0]) <= sp["size"] / 2 and abs(sp["y"] - at[1]) <= sp["size"] / 2:
                return i
        return None

    def _draw(self):
        c, C = self.canvas, self.owner.host.C
        c.delete("all")
        if self.img is None:
            c.create_text(c.winfo_width() // 2, c.winfo_height() // 2, fill=C["faint"],
                          text="No preview for %s" % os.path.basename(self.path))
            return

        def square(sp, **kw):
            if self.around_head:
                r = ig.lock_regions(self.w, self.h, [sp])[0]
                c.create_rectangle(self.ox + r["x"] * self.k, self.oy + r["y"] * self.k,
                                   self.ox + (r["x"] + r["width"]) * self.k,
                                   self.oy + (r["y"] + r["height"]) * self.k, **kw)
                return
            half = sp["size"] * self.k / 2
            x, y = self.ox + sp["x"] * self.k, self.oy + sp["y"] * self.k
            c.create_rectangle(x - half, y - half, x + half, y + half, **kw)
        c.create_image(self.ox, self.oy, image=self.img, anchor="nw")
        if self.face_point is not None:
            x = self.ox + self.face_point[0] * self.w * self.k
            y = self.oy + self.face_point[1] * self.h * self.k
            c.create_oval(x - 8, y - 8, x + 8, y + 8, outline=C["ok"], width=2)
        for sp in self.locks:
            square(sp, outline=C["ok"], width=max(2, self.owner.px(2)), dash=(6, 3))
            c.create_text(self.ox + sp["x"] * self.k, self.oy + (sp["y"] - sp["size"] / 2)
                          * self.k - self.owner.px(8), text="locked", fill=C["ok"],
                          font=self.owner.host.f_small)
        line = max(2, self.owner.px(2))
        for n, sp in enumerate(self.spots, 1):
            label = str(n)
            if sp.get("photo"):
                label += " · " + os.path.basename(sp["photo"])
            if sp.get("note"):
                label += " · " + (sp["note"] if len(sp["note"]) <= 28
                                  else sp["note"][:27] + ELLIPSIS)
            if sp.get("outline"):
                flat = [v for x, y in sp["outline"]
                        for v in (self.ox + x * self.k, self.oy + y * self.k)]
                c.create_polygon(*flat, outline=C["accent"], fill="", width=line)
                top = min(y for _, y in sp["outline"])
            else:
                square(sp, outline=C["accent"], width=line)
                top = sp["y"] - sp["size"] / 2
            c.create_text(self.ox + sp["x"] * self.k, self.oy + top * self.k
                          - self.owner.px(8), text=label, fill=C["accent"],
                          font=self.owner.host.f_small)
        trail = getattr(self, "trail", None)
        if trail and len(trail) >= 2:
            c.create_line(*[v for p in trail for v in p], fill=C["accent"], width=line)
            return
        if self.hover is not None and self._under(self.hover) is None:
            square({"x": self.hover[0], "y": self.hover[1], "size": self.size},
                   outline=C["ok" if self.mode == "lock" else "muted"], dash=(4, 3))

    # --------------------------------------------------------------- mouse
    DRAG = 6                          # screen px a press moves before it is a lasso

    def _press(self, ev):
        self.press = (ev.x, ev.y)
        self.trail = []               # the lasso so far, in screen px

    def _drag(self, ev):
        if getattr(self, "press", None) is None:
            return
        if not self.trail:
            if abs(ev.x - self.press[0]) + abs(ev.y - self.press[1]) < self.DRAG:
                return
            self.trail = [self.press]
        last = self.trail[-1]
        if abs(ev.x - last[0]) + abs(ev.y - last[1]) >= 3:
            self.trail.append((ev.x, ev.y))
            self._draw()

    def _release(self, ev):
        press, trail = getattr(self, "press", None), getattr(self, "trail", [])
        self.press, self.trail = None, []
        if press is None:
            return
        if self.choosing_face:
            at = self._to_pic(ev)
            if at is not None:
                self.face_point = [at[0] / self.w, at[1] / self.h]
                self.choosing_face = False
                self._draw()
                self._status("Face selected. The swap will use the face under this mark.")
            return
        if len(trail) >= 3:
            return self._lasso(trail)
        self._add(ev)

    def _lasso(self, trail):
        """A freehand outline, closed: only inside it is redrawn (or, in Lock
        mode, kept - as the square round it)."""
        pts = []
        for x, y in trail:
            px = min(self.w - 1, max(0, int((x - self.ox) / self.k)))
            py = min(self.h - 1, max(0, int((y - self.oy) / self.k)))
            pts.append((px, py))
        step = max(1, len(pts) // ig.FIX_OUTLINE_MAX + 1)
        spot = ig.outline_spot(pts[::step])
        spot["size"] = min(spot["size"], max(ig.FIX_MIN, min(self.w, self.h)))
        marks = self._marks()
        if marks is self.spots and len(marks) >= ig.FIX_MAX_SPOTS:
            return self._status("Eight at a time; redraw these first.", "warn")
        if marks is self.locks:
            spot.pop("outline")
        marks.append(spot)
        if marks is self.spots:
            self._note_on(len(marks) - 1)
        self._draw()
        self._status()

    def _note_on(self, i):
        """The Wrong field now writes on spot `i` (None: on every spot
        without a note of its own) and shows what is written there, and
        beside it which spot that is."""
        self.current = i
        self.note_where.config(text="on every spot" if i is None else "on spot %d" % (i + 1))
        self.showing = True           # showing it writes nothing
        try:
            self.note.set(self.wide if i is None else self.spots[i].get("note", ""))
        finally:
            self.showing = False

    def _noted(self):
        if getattr(self, "showing", False):
            return
        text = self.note.get().strip()[:ig.FIX_NOTE_MAX]
        if self.current is None or self.current >= len(self.spots):
            self.wide = text
            return
        sp = self.spots[self.current]
        if text:
            sp["note"] = text
        else:
            sp.pop("note", None)
        self._draw()

    def _add(self, ev):
        at = self._to_pic(ev)
        if at is None:
            return
        i = self._under(at)
        if i is not None:
            if self.mode != "lock":
                self._photo(i)
            return
        marks = self._marks()
        if marks is self.spots and len(marks) >= ig.FIX_MAX_SPOTS:
            return self._status("Eight at a time; redraw these first.", "warn")
        marks.append({"x": at[0], "y": at[1], "size": self.size})
        if marks is self.spots:
            self._note_on(len(marks) - 1)
        self._draw()
        self._status()

    def _photo(self, i):
        """A click on a marked spot: a photo of what goes there instead
        (Qwen-Image-Edit swaps it in), or, with one already, take it off."""
        sp = self.spots[i]
        if sp.get("photo"):
            sp.pop("photo")
            self._draw()
            return self._status("Spot %d: photo taken off; it is redrawn." % (i + 1))
        path = filedialog.askopenfilename(
            parent=self.win, title="A photo of what goes in spot %d" % (i + 1),
            filetypes=[("Pictures", "*.png *.jpg *.jpeg *.webp"), ("All files", "*.*")])
        if path:
            sp["photo"] = path
            self._draw()
            self._status("Spot %d becomes what %s shows." % (i + 1, os.path.basename(path)))

    def _remove(self, ev):
        at = self._to_pic(ev)
        if at is None:
            return
        for marks in (self.spots, self.locks):     # whichever is under it
            i = self._under(at, marks)
            if i is not None:
                del marks[i]
                if marks is self.spots:
                    self._note_on(len(marks) - 1 if marks else None)
                self._draw()
                return self._status()

    def _move(self, ev):
        self.hover = self._to_pic(ev) if ev is not None else None
        self._draw()

    def _wheel(self, ev):
        grow = 1.1 if ev.delta > 0 else 1 / 1.1
        at = self._to_pic(ev)
        i = self._under(at) if at else None
        top = max(ig.FIX_MIN, min(self.w, self.h))
        if i is not None:
            sp = self._marks()[i]
            sp["size"] = int(min(top, max(ig.FIX_MIN, sp["size"] * grow)))
        else:
            self.size = int(min(top, max(ig.FIX_MIN, self.size * grow)))
        self._draw()

    # ------------------------------------------------------------- options
    def _pick(self, key, value):
        setattr(self, key, value)
        self._paint_pills()
        self._draw()

    def _find(self, kind):
        """One click: SAM3 marks every hand, face or accessory. In Redraw
        mode they replace the squares and set what is redrawn; in Lock mode
        they are locked."""
        if self.finding:
            return
        self.finding = True
        for p in self.find_btns:
            p.set(state="disabled")
        noun = dict(self.FINDS)[kind][5:]
        self._status("Finding the %s%s" % (noun, ELLIPSIS))
        owner, mode = self.owner, self.mode

        def later(fn):
            owner._post("call", lambda: self.win.winfo_exists() and fn())

        def done(spots, err=None):
            self.finding = False
            for p in self.find_btns:
                p.set(state="normal")
            if err:
                return self._status(err, "err")
            if not spots:
                return self._status("SAM3 found no %s in the picture." % noun, "warn")
            if mode == "lock":
                self.locks.extend(spots)
            else:
                self.spots = spots
                self._note_on(None)
                self._pick("target", kind)
            self._draw()
            self._status("Found %d. %s" % (len(spots), "Locked." if mode == "lock" else
                         "Right-click any you want left alone."))

        def work():
            try:
                spots = owner.studio.find_parts(self.path, kind)
            except (ig.ComfyError, OSError) as e:
                msg = str(e)
                return later(lambda: done(None, msg))
            later(lambda: done(spots))
        owner.host._spawn(owner.s.event_id, work)

    def _paint_pills(self):
        host = self.owner.host
        for (key, value), p in self.pills.items():
            p.roles = host.PILL_ROLES["option" if getattr(self, key) == value else "ghost"]
            p.paint(host.C)

    def _face_identity(self, ident):
        """Last of all the fix swaps this identity's face (its first
        reference picture) onto the picture's biggest face."""
        self.face = ident
        if not ident:
            return self._status()
        rec = self.owner.studio.lib.get("identities", ident) or {}
        if not rec.get("references"):
            return self._status("%s has no reference picture to take the face from."
                                % rec.get("name", ident), "warn")
        self._status("Last, the face becomes %s's." % rec["name"])

    def _choose_face(self):
        self.choosing_face = True
        self._status("Click inside the face that should receive the selected identity.")

    def _status(self, text=None, role="muted"):
        n = len(self.spots)
        if text is None and self.around_head:
            text = ("Click the head; scroll to cover the entire head and hair. "
                    "Right-click removes a mark. Describe the new body and scene below."
                    if not self.locks else
                    "The marked square stays unchanged at this size and position. "
                    "The surrounding image will be generated.")
        if text is None:
            text = ("Drag round each part to redraw (or click for a square), or Find. "
                    "Right-click removes one." if not n else
                    "%d marked; only inside them changes. Click one to give it a photo "
                    "of what goes there." % n)
            if self.locks:
                text += " %d locked." % len(self.locks)
        self.msg.config(text=text)
        self.owner.skin(self.msg, bg="bg", fg=role)

    def _clear(self):
        self.spots, self.locks, self.wide = [], [], ""
        self._note_on(None)
        self._draw()
        self._status()

    def _redraw(self):
        if self.around_head and (not self.locks or not self.words.get().strip()):
            return self._status("Mark the head to keep and describe the body and scene first.", "warn")
        if not self.around_head and not self.spots and not self.face:
            return self._status("Click the part of the picture to redraw first, or "
                                "choose whose face to swap in.", "warn")
        s = self.owner.studio.fix_base(self.settings)
        s.update(mode="fix", seed=-1, fix={
            "image": self.path, "target": self.target, "strength": self.strength,
            "words": self.words.get().strip(), "spots": [dict(sp) for sp in self.spots],
            "locks": [dict(sp) for sp in self.locks], "face_swap": self.face,
            "face_point": self.face_point, "note": self.wide,
            "check": self.check == "on" and bool(self.spots)})
        if self.around_head:
            s.update(identities=[], character="", scene_faces={}, hand_pass=False)
            s["fix"].update(around_head=True, tone=0)
            model = self.owner.studio.lib.get("models", s.get("model"))
            try:
                wf = self.owner.studio.workflow_loader(model["workflow"]) if model else {}
            except ig.TemplateError as e:
                return self._status(str(e), "warn")
            if not wf.get("face_detail"):
                return self._status("Choose a redraw-capable model such as Z-Image HQ in the main form, "
                                    "then reopen Choose head photo.", "warn")
        if ig.local_faces(s):
            problems = self.owner.studio.preview(s).errors
            if problems:
                return self._status(" ".join(problems), "warn")
        self.owner.say("Fixing %s%s" % (ig.fix_words(s["fix"]), ELLIPSIS), "muted")
        self.owner.host._spawn(self.owner.s.event_id, self.owner._submit, s)
        self.win.destroy()


class CharacterCreator:
    """The character creator, laid out like a video game's: the characters on
    the left; on the right a name, the identity that carries the face, and
    tabs - Body (with the sliders), Face, Hair, Clothes, Accessories - of
    picks to click, each slot also taking free text; a picture per item worn;
    Randomize; and the character sheet, the prompt text it makes, underneath.
    Tags are the things on the character - glasses, earrings, a dress, a
    tattoo - each a word and its picture (`item_refs`), used whenever the
    word is said (`ig.outfit_of`). The anatomy constants have a tab of their own, locked:
    every character has them. Expressions are not here - they are the
    picture's, on the form."""

    SECTIONS = [name for name, _ in ig.LOOKS if name != "Expression"] + ["Tags", "Constants"]

    def __init__(self, owner, looks=None):
        self.owner = o = owner
        host = owner.host
        self.lib = owner.studio.lib
        self.records = [dict(r) for r in self.lib.all("characters")]
        self.current = None
        self.section = self.SECTIONS[0]
        self.vars = {k: tk.StringVar() for k in ig.CHARACTER_KEYS if k in ig.SLOTS}
        self.sl = {k: tk.IntVar(value=0) for k in ig.SLIDER_KEYS}
        self.name = tk.StringVar()
        self.identity = tk.StringVar()
        self.item_refs = {}
        self.tag_name = tk.StringVar()
        win = self.win = tk.Toplevel(host)
        win.title("Character creator")
        win.transient(host)
        host._skin(win, bg="bg")
        win.geometry("%dx%d" % (host._px(1020), host._px(680)))

        left = o.frame(win)
        left.pack(side="left", fill="y", padx=o.px(12), pady=o.px(12))
        self.lb = tk.Listbox(left, width=24, bd=0, highlightthickness=0, activestyle="none",
                             font=host.f_ui, exportselection=False)
        host._skin(self.lb, bg="card", fg="text", selectbackground="sel",
                   selectforeground="text")
        self.lb.pack(side="top", fill="y", expand=True)
        self.lb.bind("<<ListboxSelect>>", lambda ev: self._pick())
        btns = o.frame(left)
        btns.pack(side="top", fill="x", pady=(o.px(8), 0))
        o.button(btns, "New", self._new).pack(side="left")
        o.button(btns, "Duplicate", self._dup).pack(side="left", padx=(o.px(4), 0))
        o.button(btns, "Delete", self._delete, kind="ghost").pack(side="left",
                                                                  padx=(o.px(4), 0))

        right = o.frame(win)
        right.pack(side="left", fill="both", expand=True, pady=o.px(12),
                   padx=(0, o.px(12)))
        foot = o.frame(right)
        foot.pack(side="bottom", fill="x", pady=(o.px(8), 0))
        o.button(foot, "Use on the form", self._use, kind="accent").pack(side="right")
        o.button(foot, "Save", self._save).pack(side="right", padx=(0, o.px(6)))
        self.msg = o.label(foot, "", "muted", host.f_small)
        self.msg.pack(side="left", fill="x", expand=True)
        sheet = o.frame(right, "card")
        sheet.pack(side="bottom", fill="x", pady=(o.px(8), 0))
        o.label(sheet, "CHARACTER SHEET", "faint", host.f_small, bg="card").pack(
            side="top", fill="x", padx=o.px(10), pady=(o.px(6), 0))
        self.sheet = o.label(sheet, "", "text", bg="card", wraplength=o.px(560))
        self.sheet.pack(side="top", fill="x", padx=o.px(10), pady=(o.px(2), o.px(8)))

        top = o.frame(right)
        top.pack(side="top", fill="x")
        o.label(top, "Name", "muted", width=12).pack(side="left")
        e = host._entry(top, self.name)
        e.master.pack(side="left", fill="x", expand=True)
        e.bind("<KeyRelease>", lambda ev: self._changed())
        self.ident_row = o.frame(right)
        self.ident_row.pack(side="top", fill="x", pady=(o.px(6), 0))
        self.tabs = o.frame(right)
        self.tabs.pack(side="top", fill="x", pady=(o.px(10), o.px(4)))
        outer, self.panel = o.scrolled(right)
        outer.pack(side="top", fill="both", expand=True)

        if looks is not None:
            refs = looks.pop("item_refs", {}) if isinstance(looks, dict) else {}
            self.records.append({"name": "New character", "looks": looks,
                                 "item_refs": dict(refs)})
        self._reload_list(len(self.records) - 1 if self.records else None)

    # ---------------------------------------------------------------- state
    def status(self, text, role="muted"):
        self.msg.config(text=text)
        self.owner.skin(self.msg, bg="bg", fg=role)

    def looks(self):
        out = {k: v.get().strip() for k, v in self.vars.items()}
        out.update({k: int(v.get()) for k, v in self.sl.items()})
        return out

    def _store(self):
        if self.current is None or self.current >= len(self.records):
            return
        rec = self.records[self.current]
        rec.update(name=self.name.get().strip(), identity=self.identity.get(),
                   looks=self.looks(), item_refs=dict(self.item_refs))

    def _load(self, rec):
        looks = rec.get("looks") or {}
        for k, v in self.vars.items():
            v.set(looks.get(k, "") or "")
        for k, v in self.sl.items():
            v.set(int(looks.get(k, 0) or 0))
        self.name.set(rec.get("name") or "")
        self.identity.set(rec.get("identity") or "")
        self.item_refs = dict(rec.get("item_refs") or {})

    def _reload_list(self, select):
        self.lb.delete(0, "end")
        for r in self.records:
            self.lb.insert("end", r.get("name") or "(no name)")
        self.current = None
        if select is not None and self.records:
            self.lb.selection_clear(0, "end")
            self.lb.selection_set(select)
            self.lb.see(select)
        self._pick(store=False)

    def _pick(self, store=True):
        if store:
            self._store()
        sel = self.lb.curselection()
        self.current = sel[0] if sel else None
        if self.current is not None:
            self._load(self.records[self.current])
        self._build()

    # ----------------------------------------------------------------- form
    def _build(self):
        o, host = self.owner, self.owner.host
        self.relight = None           # the last tab's chips are about to go
        for box in (self.ident_row, self.tabs, self.panel):
            for w in box.winfo_children():
                w.destroy()
        if self.current is None:
            o.label(self.panel, "No characters yet. New makes one.", "faint").pack(
                side="top", anchor="w")
            self.sheet.config(text="")
            return
        o.label(self.ident_row, "Face (identity)", "muted", width=12).pack(side="left")
        idents = [("", "none: the words alone")] + [(i["id"], i["name"])
                                                    for i in self.lib.all("identities")]
        o.choice(self.ident_row, idents, self.identity.get(), self.identity.set).pack(
            side="left")
        for i, name in enumerate(self.SECTIONS):
            o.button(self.tabs, ("\U0001F512 " if name == "Constants" else "") + name,
                     lambda n=name: self._show(n),
                     kind="accent" if name == self.section else "quiet").pack(
                side="left", padx=(0, o.px(4)))
        o.button(self.tabs, "\U0001F3B2 Randomize", self._randomize, kind="ghost").pack(
            side="right")
        p = self.panel
        if self.section == "Constants":
            o.label(p, "Every character has these, and every picture with a person in "
                    "it says so (the form's Anatomy constants).", "muted",
                    wraplength=o.px(560)).pack(side="top", fill="x", pady=(0, o.px(6)))
            for part, pos, _ in ig.ANATOMY:
                row = o.frame(p, "card")
                row.pack(side="top", fill="x", pady=(0, o.px(3)))
                o.label(row, "\U0001F512  " + part, "text", bg="card", width=12).pack(
                    side="left", padx=o.px(8), pady=o.px(4))
                o.label(row, pos, "muted", bg="card").pack(side="left", fill="x")
            self._sheet()
            return
        if self.section == "Tags":
            self._tags(p)
            self._sheet()
            return
        if self.section == ig.SLIDER_SECTION:
            o.slider_rows(p, self.sl, self._changed)
        slots = [sl for sl in dict(ig.LOOKS)[self.section] if sl[0] in self.vars]
        self.relight = o.look_rows(p, slots, self.vars, self._changed, chips=True)
        if self.section in ("Clothes", "Accessories"):
            self._item_pictures(p, [sl[0] for sl in slots])
        elif self.section == "Hair":
            self._item_pictures(p, [], [ig.HAIR_ITEM])
        self._sheet()

    def _item_pictures(self, p, keys, items=None):
        """A picture for each item this section has the character wearing.
        Hair has one picture of its own (`items` = ["hair"])."""
        items = items or ig.items_worn({k: self.vars[k].get() for k in keys})
        self.owner.item_rows(p, items, self.item_refs, self._choose_item, self._set_item,
                             link=self._link_item)

    def _tags(self, p):
        """Every tag the character has, and a row to make one: a word and the
        picture it stands for, uploaded then (a tag without a picture is not
        made) or taken from a tag another character in the library has."""
        o, host = self.owner, self.owner.host
        o.label(p, "A tag is something on the person - glasses, earrings, a dress, a "
                "tattoo - and a picture of it. Say its word in the scene or the look "
                "(\"glasses\") and Generate draws it from the picture.",
                "muted", wraplength=o.px(560)).pack(side="top", fill="x", pady=(0, o.px(6)))
        row = o.frame(p)
        row.pack(side="top", fill="x", pady=(0, o.px(4)))
        o.label(row, "New tag", "muted", width=12).pack(side="left")
        e = host._entry(row, self.tag_name)
        e.master.pack(side="left", fill="x", expand=True)
        e.bind("<Return>", lambda ev: self._add_tag())
        o.button(row, "Upload picture…", self._add_tag, kind="accent").pack(
            side="left", padx=(o.px(6), 0))
        o.button(row, "From link…", self._add_tag_link).pack(side="left",
                                                            padx=(o.px(4), 0))
        shared = self._library_tags()
        if shared:
            row = o.frame(p)
            row.pack(side="top", fill="x", pady=(0, o.px(4)))
            o.label(row, "From library", "muted", width=12).pack(side="left")
            o.choice(row, [(i, "%s (%s)" % (name, who))
                           for i, (name, _, who) in enumerate(shared)], None,
                     lambda i: self._take_tag(*shared[i][:2])).pack(side="left")
        o.item_rows(p, sorted(self.item_refs, key=str.lower), self.item_refs,
                    self._choose_item, self._set_item, title="TAGS",
                    empty="No tags yet. Name one above and upload its picture.",
                    link=self._link_item)

    def _tag_key(self, name):
        """The key a tag is kept under: the one it already has, in any case."""
        return next((k for k in self.item_refs if k.lower() == name.lower()), name)

    def _new_tag_name(self):
        name = " ".join(self.tag_name.get().replace(",", " ").split())
        if not name:
            self.status("Name the tag first: the word you will say, like glasses.", "warn")
        return name

    def _tagged(self, name):
        if self._tag_key(name) in self.item_refs:
            self.tag_name.set("")
            self.status("Tagged %s. Say it in the scene and Generate uses the picture."
                        % name, "ok")

    def _add_tag(self):
        name = self._new_tag_name()
        if name:
            self._choose_item(self._tag_key(name))
            self._tagged(name)

    def _add_tag_link(self):
        """A tag whose picture is on the web: the word, then its link. The
        tag is made only once the picture has come."""
        name = self._new_tag_name()
        if name:
            return self._link_item(self._tag_key(name), lambda: self._tagged(name))

    def _library_tags(self):
        """[(tag, picture, character)] the other characters in the library
        have and this one does not, each picture once."""
        mine = {k.lower() for k in self.item_refs}
        out, seen = [], set()
        for i, rec in enumerate(self.records):
            if i == self.current:
                continue
            for name, path in sorted((rec.get("item_refs") or {}).items()):
                if (name.lower() in mine or (name.lower(), path) in seen
                        or not os.path.isfile(path)):
                    continue
                seen.add((name.lower(), path))
                out.append((name, path, rec.get("name") or "(no name)"))
        return out

    def _take_tag(self, name, path):
        """Another character's tag, its picture already under references/."""
        self.item_refs[self._tag_key(name)] = path
        self.status("Tagged %s, from the library." % name, "ok")
        self._build()

    def _choose_item(self, item):
        path = filedialog.askopenfilename(parent=self.win, title="A picture of the " + item,
                                          filetypes=[("Pictures", "*.png *.jpg *.jpeg *.webp"),
                                                     ("All files", "*.*")])
        if path:
            self._set_item(item, path)

    def _link_item(self, item, after=None):
        """A picture for `item` from a link (`ImageStudio.from_link`)."""
        at = self.current

        def then(path):
            if self.current != at:          # another character since: not theirs
                return
            self._set_item(item, path)
            if after:
                after()
        return self.owner.from_link(self.win, item, (self.name.get() or "character")
                                    + " items", then, self.status)

    def _set_item(self, item, path):
        if path:
            try:
                path = self.lib.keep_reference(path, (self.name.get() or "character")
                                               + " items")
            except OSError as e:
                self.status("Could not copy %s: %s" % (path, e), "err")
                return
            self.item_refs[item] = path
        else:
            self.item_refs.pop(item, None)
        self._build()

    def _show(self, section):
        self._store()
        self.section = section
        self._build()

    def _changed(self):
        if getattr(self, "relight", None):
            self.relight()
        if self.current is not None:
            name = self.name.get().strip() or "(no name)"
            if self.lb.get(self.current) != name:
                self.lb.delete(self.current)
                self.lb.insert(self.current, name)
                self.lb.selection_set(self.current)
        self._sheet()

    def _sheet(self):
        text = ig.person_text(self.looks())
        self.sheet.config(text=text or "Pick anything above and the character reads "
                                       "here, as the prompt will.")

    def _randomize(self):
        """A roll of the dice for the tab shown (every look tab, from Tags or
        Constants)."""
        tab = None if self.section in ("Tags", "Constants") else [self.section]
        for k, v in ig.random_looks(sections=tab).items():
            (self.sl if k in self.sl else self.vars)[k].set(v)
        self._build()

    # ---------------------------------------------------------------- list
    def _new(self):
        self._store()
        self.records.append({"name": "New character", "looks": {}})
        self._reload_list(len(self.records) - 1)

    def _dup(self):
        if self.current is None:
            return
        self._store()
        rec = dict(self.records[self.current])
        rec["name"] = (rec.get("name") or "Character") + " copy"
        rec.pop("id", None)
        self.records.append(rec)
        self._reload_list(len(self.records) - 1)

    def _delete(self):
        if self.current is None:
            return
        del self.records[self.current]
        self._reload_list(0 if self.records else None)
        self.status("Deleted - Save to keep it deleted.", "warn")

    def _save(self):
        self._store()
        try:
            self.lib.save("characters", self.records)
        except OSError as e:
            self.status("Could not save: %s" % e, "err")
            return False
        idx = self.current
        self.records = [dict(r) for r in self.lib.all("characters")]
        self._reload_list(idx if idx is not None and idx < len(self.records) else None)
        self.status("Saved.", "ok")
        self.owner._saved("characters")
        return True

    def _use(self):
        """Save, then put this character on the form."""
        if self.current is None or not self._save():
            return
        cid = self.records[self.current]["id"]
        self.owner.settings["character"] = cid
        self.owner._rebuild_choices()
        self.owner._set_character(cid)
        self.status("Saved, and on the form.", "ok")


class PoseEditor:
    """The pose, as a wooden artist's mannequin to drag: a torso, tapered
    limbs on ball joints, a head that shows which way it faces, and hands
    with fingers. Dragging a joint carries what hangs off it (an elbow
    brings the forearm and hand), Shift moves the joint alone, dragging the
    empty frame moves the whole figure and the wheel resizes it. A click on
    a hand gives it its next shape (Relaxed, Open, Fist, ...), as do the menus
    beside the frame, which also turn a hand palm or back to the viewer.
    Right-click hides a joint the picture should not show - an arm behind
    the back, the far ear in profile - or shows it again. The person's right
    is drawn darker; facing the viewer, it is on the frame's left.

    The mannequin is for the eye. What the ControlNet reads is the skeleton
    `studio_pose.render` draws from the same points - OpenPose's body, DWPose's
    face and hands - and "What the model sees" shows it. Use pose hands the
    pose to the form (`use_pose`)."""

    VIEW = 520                    # px on the frame's long edge, before the display's scale
    REACH = 12                    # px from a joint that still picks it up
    BG = "#1d1f23"
    WOOD = {"right": "#a8784c", "left": "#d0a271", "torso": "#bf8f60", "head": "#c99a69"}
    LINE = "#5c3d22"
    # Limb radii as fractions of the torso's length (neck to hips): (start, end).
    LIMB_R = {(2, 3): (0.11, 0.085), (3, 4): (0.085, 0.062), (5, 6): (0.11, 0.085),
              (6, 7): (0.085, 0.062), (8, 9): (0.15, 0.11), (9, 10): (0.11, 0.075),
              (11, 12): (0.15, 0.11), (12, 13): (0.11, 0.075)}

    def __init__(self, owner):
        self.owner = o = owner
        host = owner.host
        o._recheck()
        self.size = w, h = owner.planned_size
        pose = owner.pose
        self.strength = tk.DoubleVar(value=(pose or {}).get("strength", 0.9))
        self.hidden = set()
        self.hands = sp.clean_hands((pose or {}).get("hands")) or \
            sp.clean_hands(sp.DEFAULT_HANDS)
        if pose and sp.clean(pose.get("points")):
            pts = sp.refit(pose["points"], (pose.get("width") or w, pose.get("height") or h),
                           (w, h))
            self.hidden = set(pose.get("hidden") or ())
            self.points = self._whole(pts)
        else:
            self._preset("standing", draw=False)
        self.undo = []
        self.drag = None
        self.seeing = tk.StringVar(value="figure")
        k = o.px(self.VIEW) / max(w, h)
        self.vw, self.vh = int(w * k), int(h * k)

        win = self.win = tk.Toplevel(host)
        win.title("Pose")
        win.transient(host)
        host._skin(win, bg="bg")
        win.resizable(False, False)
        left = o.frame(win)
        left.pack(side="left", padx=o.px(12), pady=o.px(12))
        tabs = o.frame(left)
        tabs.pack(side="top", fill="x", pady=(0, o.px(6)))
        self.view_pills = {}
        for key, label in (("figure", "Figure"), ("model", "What the model sees")):
            pill = o.button(tabs, label, lambda k=key: self._see(k),
                            kind="accent" if key == "figure" else "quiet")
            pill.pack(side="left", padx=(0, o.px(4)))
            self.view_pills[key] = pill
        self.cv = tk.Canvas(left, width=self.vw, height=self.vh, bd=0,
                            highlightthickness=1, cursor="hand2", bg=self.BG,
                            highlightbackground="#3a3a3a")
        self.cv.pack(side="top")
        self.hover = o.label(left, "%d × %d picture" % (w, h), "faint", host.f_small)
        self.hover.pack(side="top", fill="x", pady=(o.px(4), 0))

        right = o.frame(win)
        right.pack(side="left", fill="y", pady=o.px(12), padx=(0, o.px(12)))
        o.cap(right, "Start from")
        grid = o.frame(right)
        grid.pack(side="top", fill="x")
        for i, (key, label, _) in enumerate(sp.PRESETS):
            o.button(grid, label, lambda k=key: self._preset(k)).grid(
                row=i // 2, column=i % 2, sticky="we", padx=(0, o.px(4)), pady=(0, o.px(4)))
        o.cap(right, "Hands")
        self.hand_pills, self.back_pills = {}, {}
        shapes = [(k, label) for k, label, _ in sp.HAND_SHAPES]
        for side in ("right", "left"):
            row = o.frame(right)
            row.pack(side="top", fill="x", pady=(0, o.px(4)))
            o.label(row, side.capitalize(), "muted", width=6).pack(side="left")
            pill = o.choice(row, shapes, self.hands[side]["shape"],
                            lambda v, s=side: self._shape(s, v))
            pill.pack(side="left")
            self.hand_pills[side] = pill
            back = o.button(row, "", lambda s=side: self._flip(s), kind="ghost")
            back.pack(side="left", padx=(o.px(4), 0))
            self.back_pills[side] = back
        self._label_hands()
        o.cap(right, "Change")
        row = o.frame(right)
        row.pack(side="top", fill="x")
        o.button(row, "Mirror", self._mirror).pack(side="left")
        o.button(row, "Undo", self._undo, kind="ghost").pack(side="left", padx=(o.px(4), 0))
        o.cap(right, "How closely to follow it")
        o.slider(right, self.strength, 0.3, 1.2).pack(side="top", fill="x")
        o.label(right, "0.9 follows the figure; lower lets the model move the person "
                "more freely.", "faint", host.f_small, wraplength=o.px(240)).pack(
            side="top", fill="x")
        o.label(right, "Drag a joint to move it and what hangs off it; Shift-drag "
                "moves it alone. Click a hand for its next shape. Drag the empty frame "
                "to move the figure, scroll to resize it. Right-click a joint to hide "
                "or show it. Ctrl+Z undoes. Right and left are the person's own.",
                "muted", host.f_small, wraplength=o.px(240)).pack(
            side="top", fill="x", pady=(o.px(12), 0))
        foot = o.frame(right)
        foot.pack(side="bottom", fill="x", pady=(o.px(12), 0))
        o.button(foot, "Use pose", self._use, kind="accent").pack(side="right")
        o.button(foot, "Cancel", win.destroy, kind="ghost").pack(side="right",
                                                                 padx=(0, o.px(6)))

        self.cv.bind("<ButtonPress-1>", self._press)
        self.cv.bind("<B1-Motion>", self._move)
        self.cv.bind("<ButtonRelease-1>", self._release)
        self.cv.bind("<Button-3>", self._toggle)
        self.cv.bind("<Motion>", self._hover)
        self.cv.bind("<MouseWheel>", self._wheel)
        win.bind("<Control-z>", lambda ev: self._undo())
        win.bind("<Escape>", lambda ev: win.destroy())
        self._draw()

    # ---------------------------------------------------------------- state
    def _whole(self, pts):
        """Points with a place for every joint: a joint a preset leaves out
        is put beside its other side, hidden, so it can be shown again."""
        out = [list(p) if p else None for p in pts]
        for i, p in enumerate(out):
            if p is None:
                twin = out[sp.MIRROR.get(i, 1)] or [0.5, 0.5]
                out[i] = [twin[0] + 0.01, twin[1]]
                self.hidden.add(i)
        return out

    def _keep(self):
        self.undo.append(([list(p) for p in self.points], set(self.hidden),
                          {s: dict(h) for s, h in self.hands.items()}))
        del self.undo[:-50]

    def _preset(self, key, draw=True):
        if draw:
            self._keep()
        self.hidden = set()
        self.points = self._whole(sp.preset(key, *self.size))
        if draw:
            self._draw()

    def _mirror(self):
        self._keep()
        self.hidden = {sp.MIRROR.get(i, i) for i in self.hidden}
        self.points = sp.mirror(self.points)
        self.hands = sp.mirror_hands(self.hands)
        self._label_hands()
        self._draw()

    def _undo(self):
        if self.undo:
            self.points, self.hidden, self.hands = self.undo.pop()
            self._label_hands()
            self._draw()

    def _shape(self, side, shape, keep=True):
        if keep:
            self._keep()
        self.hands[side]["shape"] = shape
        self._label_hands()
        self._draw()

    def _cycle(self, side):
        names = sp.HAND_SHAPE_NAMES
        now = self.hands[side]["shape"]
        self._shape(side, names[(names.index(now) + 1) % len(names)], keep=False)

    def _flip(self, side):
        self._keep()
        self.hands[side]["back"] = not self.hands[side]["back"]
        self._label_hands()
        self._draw()

    def _label_hands(self):
        names = dict((k, label) for k, label, _ in sp.HAND_SHAPES)
        for side, pill in self.hand_pills.items():
            pill.set(text=names[self.hands[side]["shape"]] + "  ▾")
            self.back_pills[side].set(text="Back" if self.hands[side]["back"] else "Palm")

    def _see(self, which):
        self.seeing.set(which)
        host = self.owner.host
        for key, pill in self.view_pills.items():
            pill.roles = host.PILL_ROLES["accent" if key == which else "quiet"]
            pill.paint(host.C)
        self._draw()

    # ---------------------------------------------------------------- mouse
    def _near(self, ev):
        return sp.near(self.points, ev.x, ev.y, self.vw, self.vh, self.owner.px(self.REACH))

    def _hand_at(self, ev):
        """The side whose hand (fingers and palm) is under the pointer."""
        best, dist = None, None
        for side, pts in self._hand_points().items():
            cx = sum(p[0] for p in pts) / len(pts) * self.vw
            cy = sum(p[1] for p in pts) / len(pts) * self.vh
            reach = max(math.hypot(p[0] * self.vw - cx, p[1] * self.vh - cy) for p in pts)
            d = math.hypot(ev.x - cx, ev.y - cy)
            if d <= reach * 0.9 + self.owner.px(4) and (dist is None or d < dist):
                best, dist = side, d
        return best

    def _hand_points(self):
        return sp.hand_points(self._seen(), self.vw, self.vh, self.hands)

    def _seen(self):
        return [None if i in self.hidden else p for i, p in enumerate(self.points)]

    def _press(self, ev):
        self._keep()
        j = self._near(ev)
        side = None
        if j is None:
            side = self._hand_at(ev)
        if side:
            moving = sp.carried(sp.HAND_OF[side][0])
        elif j in (4, 7) and not ev.state & 0x0001:
            side = "right" if j == 4 else "left"
            moving = [j]
        elif j is None:
            moving = list(range(len(self.points)))
        elif ev.state & 0x0001:                 # Shift: the joint alone
            moving = [j]
        else:
            moving = sp.carried(j)
        self.drag = (moving, ev.x, ev.y)
        self.click = (side, ev.x, ev.y)

    def _move(self, ev):
        if not self.drag:
            return
        moving, x0, y0 = self.drag
        dx, dy = (ev.x - x0) / self.vw, (ev.y - y0) / self.vh
        for i in moving:
            p = self.points[i]
            p[0] = min(1.2, max(-0.2, p[0] + dx))
            p[1] = min(1.2, max(-0.2, p[1] + dy))
        self.drag = (moving, ev.x, ev.y)
        self._draw()

    def _release(self, ev):
        """A click on a hand that did not move it: its next shape."""
        side, x0, y0 = getattr(self, "click", (None, 0, 0))
        self.drag, self.click = None, (None, 0, 0)
        if side and math.hypot(ev.x - x0, ev.y - y0) < self.owner.px(3):
            self._cycle(side)
            self._hover(ev)

    def _wheel(self, ev):
        self._keep()
        k = 1.05 ** (ev.delta / 120)
        seen = [p for i, p in enumerate(self.points) if i not in self.hidden] or self.points
        cx = sum(p[0] for p in seen) / len(seen)
        cy = sum(p[1] for p in seen) / len(seen)
        for p in self.points:
            p[0], p[1] = cx + (p[0] - cx) * k, cy + (p[1] - cy) * k
        self._draw()

    def _toggle(self, ev):
        j = self._near(ev)
        if j is None:
            return
        self._keep()
        self.hidden ^= {j}
        if len(self.hidden) == len(self.points):
            self.hidden.discard(j)
        self._draw()
        self._hover(ev)

    def _hover(self, ev):
        j = self._near(ev)
        side = self._hand_at(ev) if j is None else ("right" if j == 4 else
                                                    "left" if j == 7 else None)
        w, h = self.size
        if side:
            names = dict((k, label) for k, label, _ in sp.HAND_SHAPES)
            text = "%s hand: %s - click for the next shape" % (
                side.capitalize(), names[self.hands[side]["shape"]])
        elif j is not None:
            text = sp.JOINTS[j] + (" (hidden)" if j in self.hidden else "")
        else:
            text = "%d × %d picture" % (w, h)
        self.hover.config(text=text)

    # ---------------------------------------------------------------- draw
    def _draw(self):
        self.cv.delete("all")
        if self.seeing.get() == "model":
            self.cv.config(bg="#000000")
            self._draw_skeleton()
        else:
            self.cv.config(bg=self.BG)
            self._draw_figure()
        self._draw_handles()

    def _limb(self, a, b, ra, rb, fill):
        """A tapered limb from a to b (view px) with a ball at each end."""
        cv = self.cv
        dx, dy = b[0] - a[0], b[1] - a[1]
        n = math.hypot(dx, dy) or 1e-6
        nx, ny = -dy / n, dx / n
        cv.create_polygon(a[0] + nx * ra, a[1] + ny * ra, b[0] + nx * rb, b[1] + ny * rb,
                          b[0] - nx * rb, b[1] - ny * rb, a[0] - nx * ra, a[1] - ny * ra,
                          fill=fill, outline=self.LINE)
        for (x, y), r in ((a, ra), (b, rb)):
            cv.create_oval(x - r, y - r, x + r, y + r, fill=fill, outline=self.LINE)

    def _draw_figure(self):
        cv = self.cv
        seen = self._seen()
        at = [None if p is None else (p[0] * self.vw, p[1] * self.vh) for p in seen]
        hips = [at[i] for i in (8, 11) if at[i]]
        neck = at[1]
        if neck and hips:
            mid = (sum(p[0] for p in hips) / len(hips), sum(p[1] for p in hips) / len(hips))
            torso = math.hypot(mid[0] - neck[0], mid[1] - neck[1])
        else:
            mid, torso = None, self.vh * 0.28
        s = max(torso, self.owner.px(20))
        wood = self.WOOD

        def limb(a, b, side):
            if at[a] and at[b]:
                ra, rb = self.LIMB_R[(a, b)]
                self._limb(at[a], at[b], ra * s, rb * s, wood[side])

        limb(11, 12, "left")
        limb(12, 13, "left")
        limb(8, 9, "right")
        limb(9, 10, "right")
        # The torso: shoulders to hips, a little wider than the joints.
        shoulders = [at[i] for i in (2, 5)]
        if neck and all(shoulders) and len(hips) == 2:
            (rx, ry), (lx, ly) = shoulders
            cx = (rx + lx) / 2
            out = 0.08 * s
            pts = [rx - out if rx < cx else rx + out, ry, cx, (ry + ly) / 2 - 0.04 * s,
                   lx + out if lx > cx else lx - out, ly, at[11][0], at[11][1] - 0.05 * s,
                   at[11][0], at[11][1] + 0.1 * s, at[8][0], at[8][1] + 0.1 * s,
                   at[8][0], at[8][1] - 0.05 * s]
            cv.create_polygon(*pts, smooth=True, fill=wood["torso"], outline=self.LINE)
        elif neck and mid:
            self._limb(neck, mid, 0.24 * s, 0.2 * s, wood["torso"])
        self._draw_head(at, s)
        limb(5, 6, "left")
        limb(6, 7, "left")
        limb(2, 3, "right")
        limb(3, 4, "right")
        for side, pts in self._hand_points().items():
            self._draw_hand([(x * self.vw, y * self.vh) for x, y in pts], s,
                            wood[side], self.hands[side]["back"])

    def _draw_head(self, at, s):
        cv = self.cv
        nose, neck = at[0], at[1]
        eyes = [at[i] for i in (14, 15) if at[i]]
        ears = [at[i] for i in (16, 17) if at[i]]
        if not (nose or eyes):
            return
        centre = (eyes if eyes else [nose])
        cx = sum(p[0] for p in centre) / len(centre)
        cy = sum(p[1] for p in centre) / len(centre)
        if len(ears) == 2:
            rx = math.hypot(ears[0][0] - ears[1][0], ears[0][1] - ears[1][1]) / 2 * 1.15
        else:
            rx = 0.2 * s
        rx = max(rx, 0.12 * s)
        ry = rx * 1.3
        # "Up" for the head is away from the neck.
        ux, uy = (0.0, -1.0)
        if neck:
            d = math.hypot(cx - neck[0], cy - neck[1]) or 1e-6
            ux, uy = (cx - neck[0]) / d, (cy - neck[1]) / d
            self._limb(neck, (cx - ux * ry * 0.6, cy - uy * ry * 0.6), 0.08 * s, 0.08 * s,
                       self.WOOD["torso"])
        px_, py_ = -uy, ux
        # The eye line sits a little under the middle of the head.
        hx, hy = cx + ux * ry * 0.12, cy + uy * ry * 0.12
        pts = []
        for i in range(28):
            t = 2 * math.pi * i / 28
            pts += [hx + px_ * rx * math.cos(t) + ux * ry * math.sin(t),
                    hy + py_ * rx * math.cos(t) + uy * ry * math.sin(t)]
        cv.create_polygon(*pts, smooth=True, fill=self.WOOD["head"], outline=self.LINE)
        dot = max(2, s * 0.025)
        for x, y in eyes:
            cv.create_oval(x - dot, y - dot, x + dot, y + dot, fill=self.LINE, outline="")
        if nose:
            cv.create_oval(nose[0] - dot * 1.3, nose[1] - dot * 1.3, nose[0] + dot * 1.3,
                           nose[1] + dot * 1.3, fill="#8a5a33", outline=self.LINE)

    def _draw_hand(self, pts, s, fill, back):
        cv = self.cv
        palm = [pts[i] for i in (0, 1, 5, 9, 13, 17)]
        cv.create_polygon(*[c for p in palm for c in p], smooth=True, fill=fill,
                          outline=self.LINE)
        r = max(1.5, 0.028 * s)
        for base in (1, 5, 9, 13, 17):
            chain = [pts[0] if base == 1 else pts[base]] + [pts[base + k] for k in range(1, 4)]
            if base == 1:
                chain = [pts[1], pts[2], pts[3], pts[4]]
            for (a, b) in zip(chain, chain[1:]):
                self._limb(a, b, r * (1.15 if base == 1 else 1), r * 0.85, fill)
        if back:                                  # knuckles, to tell the back from the palm
            for i in (5, 9, 13, 17):
                x, y = pts[i]
                cv.create_line(x - r, y, x + r, y, fill=self.LINE)

    def _draw_skeleton(self):
        """What the ControlNet reads, drawn as studio_pose.render draws it."""
        cv, px = self.cv, self.owner.px
        seen = self._seen()
        stick = max(3, px(5))
        for n, (a, b) in enumerate(sp.LIMBS):
            if seen[a] and seen[b]:
                cv.create_line(seen[a][0] * self.vw, seen[a][1] * self.vh,
                               seen[b][0] * self.vw, seen[b][1] * self.vh, width=stick,
                               capstyle="round", fill="#%02x%02x%02x" % tuple(
                                   int(c * 0.6) for c in sp.COLOURS[n]))
        for i, p in enumerate(seen):
            if p:
                x, y, r = p[0] * self.vw, p[1] * self.vh, stick / 2 + 1
                cv.create_oval(x - r, y - r, x + r, y + r, outline="",
                               fill="#%02x%02x%02x" % sp.COLOURS[i])
        for x, y in sp.face_points(seen, self.vw, self.vh):
            x, y = x * self.vw, y * self.vh
            cv.create_oval(x - 1, y - 1, x + 1, y + 1, fill="#ffffff", outline="")
        for pts in self._hand_points().values():
            at = [(x * self.vw, y * self.vh) for x, y in pts]
            for n, (a, b) in enumerate(sp.HAND_EDGES):
                rgb = colorsys.hsv_to_rgb(n / len(sp.HAND_EDGES), 1, 1)
                cv.create_line(*at[a], *at[b], width=2, fill="#%02x%02x%02x" % tuple(
                    int(c * 255) for c in rgb))
            for x, y in at:
                cv.create_oval(x - 2, y - 2, x + 2, y + 2, fill="#0000ff", outline="")

    def _draw_handles(self):
        """The joints you can grab, drawn over either view."""
        cv, r = self.cv, max(3, self.owner.px(4))
        for i, p in enumerate(self.points):
            x, y = p[0] * self.vw, p[1] * self.vh
            if i in self.hidden:
                cv.create_oval(x - r, y - r, x + r, y + r, outline="#8a8a8a", dash=(2, 2))
            elif i not in (14, 15, 16, 17) or self.seeing.get() == "model":
                cv.create_oval(x - r, y - r, x + r, y + r, outline="#f2f2f2")

    # ---------------------------------------------------------------- done
    def _use(self):
        w, h = self.size
        self.owner.use_pose({"points": [[round(p[0], 4), round(p[1], 4)] for p in self.points],
                             "hidden": sorted(self.hidden), "width": w, "height": h,
                             "hands": {s: dict(v) for s, v in self.hands.items()},
                             "strength": round(self.strength.get(), 2)})
        self.win.destroy()

class ModelSourceSettings:
    """Provider settings and reviewable model / upstream-code discoveries."""

    def __init__(self, owner, source):
        self.owner, self.source = owner, source
        self.busy = False
        name, domain, env = model_sources.SOURCES[source]
        self.win = win = tk.Toplevel(owner.host)
        win.title(name)
        win.transient(owner.host)
        owner.skin(win, bg="bg")
        win.geometry("%dx%d" % (owner.px(780), owner.px(780)))
        win.minsize(owner.px(600), owner.px(650))
        body = owner.frame(win)
        body.pack(fill="both", expand=True, padx=owner.px(16), pady=owner.px(16))
        data = model_sources.load(owner.studio.lib.root, source)
        owner.label(body, "Model links — one per line", "muted").pack(anchor="w")
        self.links = tk.Text(body, height=3, wrap="word", bd=0,
                             font=owner.host.f_ui, highlightthickness=0)
        owner.skin(self.links, bg="card", fg="text", insertbackground="accent")
        self.links.pack(fill="x", pady=(owner.px(6), owner.px(12)))
        self.links.insert("1.0", "\n".join(data["links"]))
        owner.label(body, "API key", "muted").pack(anchor="w")
        self.token = tk.StringVar(value=data["token"])
        entry = owner.host._entry(body, self.token)
        entry.config(show="•")
        entry.master.pack(fill="x", pady=(owner.px(4), owner.px(8)))
        if os.environ.get(env):
            entry.config(state="readonly")
            note = "API key supplied by %s in your environment." % env
        else:
            note = "Saved locally beside your library. The API key is stored unencrypted."
        owner.label(body, note, "faint", owner.host.f_small,
                    wraplength=owner.px(540)).pack(fill="x")
        self.message = owner.label(body, "", "muted", owner.host.f_small,
                                   wraplength=owner.px(540))
        self.message.pack(fill="x", pady=owner.px(8))
        row = owner.frame(body)
        row.pack(fill="x")
        owner.button(row, "Save", self.save, kind="accent").pack(side="right")
        owner.button(row, "Close", win.destroy, kind="ghost").pack(side="right", padx=owner.px(6))
        if source == "civitai":
            owner.button(row, "Import LoRAs", self.import_loras).pack(side="left")
        owner.label(body, "Discover models and code improvements", font=owner.host.f_ui).pack(
            anchor="w", pady=(owner.px(14), owner.px(4)))
        owner.label(body, "Checks when opened; successful results are cached for 24 hours. "
                    "Candidates need review before use.", "faint", owner.host.f_small,
                    wraplength=owner.px(710)).pack(fill="x")
        controls = owner.frame(body)
        controls.pack(fill="x", pady=owner.px(6))
        self.refresh_button = owner.button(controls, "Refresh now",
                                           lambda: self.refresh_discoveries(force=True))
        self.refresh_button.pack(side="right")
        self.discovery_status = owner.label(controls, "", "muted", owner.host.f_small,
                                            wraplength=owner.px(530))
        self.discovery_status.pack(side="left", fill="x", expand=True)
        scroll, self.results = owner.scrolled(body)
        scroll.pack(fill="both", expand=True)
        self.show_discoveries(discovery.cached(owner.studio.lib.root, source))

    def refresh_discoveries(self, force=False):
        if self.busy:
            return
        self.busy = True
        self.refresh_button.set(state="disabled")
        self.discovery_status.config(text="Finding models and code releases" + ELLIPSIS)
        root, source, token = self.owner.studio.lib.root, self.source, self.token.get().strip()

        def work():
            try:
                result = discovery.discover(root, source, token, force=force)
            except Exception:
                result = {"items": [], "checked": 0, "errors": ["Discovery failed. Try Refresh now."]}
            self.owner._post("call", lambda: self.show_discoveries(result))

        self.owner.host._spawn(self.owner.s.event_id, work)

    def show_discoveries(self, result):
        if not self.win.winfo_exists():
            return
        self.busy = False
        self.refresh_button.set(state="normal")
        for widget in self.results.winfo_children():
            widget.destroy()
        checked = result.get("checked", 0)
        status = ("Checked " + time.strftime("%b %d, %H:%M", time.localtime(checked))) if checked else "No results yet."
        errors = result.get("errors") or []
        if errors:
            status += " · Some feeds unavailable."
        self.discovery_status.config(text=status)
        o = self.owner
        for error in errors:
            o.label(self.results, error, "warn", o.host.f_small,
                    wraplength=o.px(680)).pack(fill="x", pady=o.px(4))
        for item in addons.candidates() + result.get("items", []):
            card = o.frame(self.results)
            card.pack(fill="x", pady=o.px(8))
            title = item["kind"] + " · " + item["title"]
            if item.get("stale"):
                title += " (previous result; refresh failed)"
            o.label(card, title, wraplength=o.px(680)).pack(fill="x")
            if item["kind"] in ("Code / library", "Supported add-on"):
                for label, value in (
                    ("What it does", item.get("purpose", item["why"])),
                    ("What it can improve", item.get("improves", item["why"])),
                    ("How it fits the app", item.get("integration", "Refresh for integration details.")),
                ):
                    o.label(card, label + ": " + value, "muted", o.host.f_small,
                            wraplength=o.px(680)).pack(fill="x", pady=(o.px(3), 0))
                changes = item.get("highlights") or []
                detail = ("Publisher's release changes:\n" + "\n".join("• " + c for c in changes)
                          if changes else "No release summary supplied. Review the source for version-specific changes.")
                if item["kind"] == "Code / library":
                    o.label(card, detail, "faint", o.host.f_small,
                            wraplength=o.px(680)).pack(fill="x", pady=o.px(6))
            else:
                o.label(card, item["why"], "muted", o.host.f_small,
                        wraplength=o.px(680)).pack(fill="x")
                o.label(card, item["details"], "faint", o.host.f_small,
                        wraplength=o.px(680)).pack(fill="x", pady=o.px(4))
            actions = o.frame(card)
            actions.pack(fill="x")
            if item.get("url"):
                o.button(actions, "Review source", lambda item=item: self.review_source(item)).pack(side="left")
            if item["kind"] == "Supported add-on":
                button = o.button(actions, "Add to app", lambda: None, kind="accent")
                button.command = lambda item=item, button=button: self.install_addon(item, button)
                button.pack(side="left")
            elif item["kind"] == "Code / library":
                o.label(actions, "Review only — no supported installer yet", "faint", o.host.f_small).pack(
                    side="left", padx=o.px(6))
            else:
                o.button(actions, "Keep link", lambda item=item: self.keep_link(item)).pack(
                    side="left", padx=o.px(6))
                if item.get("importable"):
                    o.button(actions, "Import LoRA", lambda item=item: self.import_candidate(item)).pack(side="left")

    def install_addon(self, item, button):
        folder = filedialog.askdirectory(parent=self.win,
                                         title="Choose ComfyUI folder (contains main.py)", mustexist=True)
        if not folder:
            return
        button.set(state="disabled", text="Installing" + ELLIPSIS)
        self.message.config(text="Installing " + item["title"] + ELLIPSIS)

        def work():
            try:
                result = addons.install(item["addon_id"], folder)
                note = ("Already installed" if result["unchanged"] else "Installed and file verified")
                note += ": " + result["path"] + ". Restart ComfyUI when its queue is idle, then Check connections."
                if result["backup"]:
                    note += " Previous file saved at " + result["backup"]
                note += " Node loading and required models are checked by ComfyUI after restart."
                success = True
            except (OSError, ValueError, SyntaxError) as error:
                note, success = "Could not install: " + str(error), False
            self.owner._post("call", lambda: finished(note, success))

        def finished(note, success):
            self.owner.say(note, "ok" if success else "err")
            if self.win.winfo_exists():
                self.message.config(text=note)
                if button.winfo_exists():
                    button.set(text="Installed" if success else "Add to app", state="normal")

        self.owner.host._spawn(self.owner.s.event_id, work)

    def review_source(self, item):
        # Discovery builds these URLs from fixed origins and validated IDs.
        from urllib.parse import urlsplit
        url = urlsplit(item["url"])
        if url.scheme == "https" and url.netloc in ("huggingface.co", "civitai.com", "github.com"):
            webbrowser.open(item["url"])

    def keep_link(self, item):
        links = self.links.get("1.0", "end").splitlines()
        if item["url"] not in links:
            self.links.insert("end", "\n" + item["url"])
        self.save()

    def import_candidate(self, item):
        if not self.save():
            return
        dialog = LoraImport(self.owner)
        dialog.links.delete("1.0", "end")
        dialog.links.insert("1.0", item["url"])

    def save(self):
        try:
            model_sources.save(self.owner.studio.lib.root, self.source,
                               self.links.get("1.0", "end").splitlines(), self.token.get())
        except (OSError, ValueError):
            domain = model_sources.SOURCES[self.source][1]
            self.message.config(text="Could not save. Check that every link starts with "
                                "https://%s/ and the library folder is writable." % domain)
            return False
        self.message.config(text="Saved.")
        return True

    def import_loras(self):
        if self.save():
            dialog = LoraImport(self.owner)
            dialog.links.delete("1.0", "end")
            dialog.links.insert("1.0", self.links.get("1.0", "end").strip())
            self.win.destroy()


class LoraImport:
    """The LoRA library's import window: CivitAI links pasted one per line,
    and/or .safetensors files picked from disk, each made into a library
    record by `studio_civitai` on a worker thread. The file itself can be
    put into a backend's LoRA folder on this PC (a download for a link, a
    copy for a file); a folder on the other machine is never assumed."""

    def __init__(self, owner, editor=None):
        self.owner, self.editor = owner, editor
        o, host = owner, owner.host
        self.files = []
        self.stop = None
        self.busy = False
        win = self.win = tk.Toplevel(editor.win if editor else host)
        win.title("Import LoRAs")
        win.transient(editor.win if editor else host)
        host._skin(win, bg="bg")
        win.geometry("%dx%d" % (host._px(600), host._px(440)))
        win.protocol("WM_DELETE_WINDOW", self.close)
        body = o.frame(win)
        body.pack(side="top", fill="both", expand=True, padx=o.px(14), pady=o.px(12))
        o.label(body, "CivitAI links, one per line: a model page, a download link or "
                "an AIR (urn:air:…)", "muted", host.f_small).pack(side="top", fill="x")
        self.links = tk.Text(body, height=4, wrap="none", bd=0, highlightthickness=0,
                             font=host.f_ui, padx=o.px(6), pady=o.px(4))
        o.skin(self.links, bg="card", fg="text", insertbackground="accent")
        self.links.pack(side="top", fill="x", pady=(o.px(2), 0))
        pasted = self._clipboard_link()
        if pasted:
            self.links.insert("1.0", pasted + "\n")

        o.label(body, "…and/or LoRA files on this PC", "muted", host.f_small).pack(
            side="top", fill="x", pady=(o.px(10), o.px(2)))
        row = o.frame(body)
        row.pack(side="top", fill="x")
        o.button(row, "Add .safetensors files…", self.add_files).pack(side="left")
        o.button(row, "Clear", self.clear_files, kind="ghost").pack(
            side="left", padx=(o.px(4), 0))
        self.file_list = o.label(body, "No files.", "faint", host.f_small,
                                 wraplength=o.px(560))
        self.file_list.pack(side="top", fill="x", pady=(o.px(2), 0))

        o.label(body, "Put the LoRA file in", "muted", host.f_small).pack(
            side="top", fill="x", pady=(o.px(10), o.px(2)))
        folders = owner.lora_folders()
        self.dest = folders[0][0] if folders else ""
        choices = folders + [("", "Nowhere - only add the profile (the file is already "
                                  "on the backend)")]
        o.choice(body, choices, self.dest, self._set_dest).pack(side="top", anchor="w")
        if not folders:
            o.label(body, "No backend has a LoRA folder on this PC (Backends… → LoRA "
                    "folder), so files cannot be placed; profiles still can.", "faint",
                    host.f_small, wraplength=o.px(560)).pack(
                side="top", fill="x")

        self.lookup = tk.BooleanVar(value=True)
        cb = tk.Checkbutton(body, text="Look files up on CivitAI by their hash",
                            variable=self.lookup, anchor="w", font=host.f_ui, bd=0,
                            highlightthickness=0)
        o.skin(cb, bg="bg", fg="text", activebackground="bg", selectcolor="card",
               activeforeground="text")
        cb.pack(side="top", fill="x", pady=(o.px(10), 0))

        o.label(body, "CivitAI API key (only some downloads need one; kept beside the "
                "library)", "muted", host.f_small).pack(
            side="top", fill="x", pady=(o.px(10), o.px(2)))
        self.token = tk.StringVar(value=civitai.load_token(owner.studio.lib.root))
        e = host._entry(body, self.token)
        e.config(show="•")
        e.master.pack(side="top", fill="x")

        foot = o.frame(win)
        foot.pack(side="bottom", fill="x", padx=o.px(14), pady=(0, o.px(12)))
        self.go = o.button(foot, "Import", self.start, kind="accent")
        self.go.pack(side="right")
        self.msg = o.label(foot, "", "muted", host.f_small, wraplength=o.px(440))
        self.msg.pack(side="left", fill="x", expand=True)

    @staticmethod
    def _one_line(text):
        return " ".join(str(text).split())

    def _clipboard_link(self):
        try:
            text = self.win.clipboard_get()
        except tk.TclError:
            return ""
        lines = [ln.strip() for ln in text.splitlines()
                 if ln.strip() and not ln.strip().isdigit() and civitai.parse_link(ln)]
        return "\n".join(lines[:20])

    def _set_dest(self, value):
        self.dest = value

    def add_files(self):
        paths = filedialog.askopenfilenames(parent=self.win, filetypes=[
            ("LoRA files", "*.safetensors"), ("All files", "*.*")])
        for p in paths:
            if p not in self.files:
                self.files.append(p)
        self._show_files()

    def clear_files(self):
        self.files = []
        self._show_files()

    def _show_files(self):
        self.file_list.config(text="\n".join(os.path.basename(p) for p in self.files)
                              or "No files.")

    def status(self, text, role="muted"):
        self.msg.config(text=text)
        self.owner.skin(self.msg, bg="bg", fg=role)

    def finished(self, text, role):
        self.busy = False
        self.status(text, role)

    def close(self):
        if self.stop is not None:
            self.stop.set()
        self.win.destroy()

    def start(self):
        if self.busy:
            return
        links = [ln.strip() for ln in self.links.get("1.0", "end").splitlines()
                 if ln.strip()]
        bad = [ln for ln in links if civitai.parse_link(ln) is None]
        if bad:
            self.status("Not a CivitAI link: %s" % bad[0], "err")
            return
        if not links and not self.files:
            self.status("Paste a link or add a file first.", "warn")
            return
        lib = self.owner.studio.lib
        token = self.token.get().strip()
        if token != civitai.load_token(lib.root) and not os.environ.get(civitai.TOKEN_ENV):
            try:
                civitai.save_token(lib.root, token)
            except OSError as e:
                self.status("Could not keep the API key: %s" % e, "warn")
        folder = ""
        if self.dest:
            b = self.owner.studio.backend(self.dest)
            folder = b.get("lora_dir", "") if b else ""
        dirs = [f for f in (b.get("lora_dir") for b in self.owner.studio.backends()) if f]
        self.stop = threading.Event()
        self.busy = True
        self.status("Importing" + ELLIPSIS, "accent")
        self.owner.host._spawn(self.owner.s.event_id, self._work, links, list(self.files),
                               folder, dirs, token, self.lookup.get(), self.stop)

    def _work(self, links, files, folder, dirs, token, lookup, stop):
        """Off the UI thread: one item at a time, each failure said and the
        rest still done."""
        o = self.owner
        lib = o.studio.lib
        client = civitai.Client(token)
        post = o.host.q.put
        ev = o.s.event_id
        last = ""

        def say(text, role="muted"):
            post(("images", ev, ("import-said", (self, self._one_line(text), role))))

        added = updated = 0
        failed = []
        items = [("link", x) for x in links] + [("file", x) for x in files]
        for kind, item in items:
            if stop.is_set():
                break
            try:
                if kind == "link":
                    rec, new = civitai.import_link(lib, client, item, folder, say, stop)
                else:
                    rec, new = civitai.import_file(lib, client, item, folder, dirs, lookup,
                                                   say)
            except Exception as e:          # said, and the next item still runs
                why = e if isinstance(e, civitai.CivitAIError) else "%s: %s" % (
                    type(e).__name__, e)
                failed.append("%s: %s" % (os.path.basename(item) if kind == "file" else item,
                                          why))
                say(failed[-1], "err")
                continue
            last = rec["id"]
            added, updated = added + new, updated + (not new)
        bits = []
        if added:
            bits.append("%d added" % added)
        if updated:
            bits.append("%d already there, empty fields filled" % updated)
        if failed:
            bits.append("%d failed - %s" % (len(failed), "; ".join(failed)))
        text = ("Done: " + ", ".join(bits) + ".") if bits else "Nothing imported."
        role = "err" if failed and not (added or updated) else "warn" if failed else "ok"
        o._post("said", ("LoRA import: " + text, role))
        post(("images", ev, ("import-done", (self, self.editor, last, text, role))))


class AddonsWindow:
    """Add-ons: the LoRAs for the models the user has, like a media server's
    plugin catalog where the "server version" is the model. A pill per model
    (and one for LoRAs that fit none of them) above two tabs:

    - **Installed** - the library's LoRAs that work with that model, then
      those whose model is not set. Each can be turned off (kept, not
      offered on the form), made Always on, told which model it is for, or
      uninstalled (its file to the Recycle Bin: `studio_catalog.uninstall`).
    - **Catalog** - CivitAI's LoRAs for that model's own base models, by
      downloads, rating or date, with a search. Install downloads the file
      into a LoRA folder on this PC and files it in the library.

    Network and file work is on worker threads; results come back through
    the tab's queue (`_post("call", ...)`), dropped if the window closed or
    a newer search replaced them (`gen`)."""

    TABS = (("installed", "Installed"), ("catalog", "CivitAI"),
            ("huggingface", "Hugging Face"), ("github", "GitHub plugins"))
    OTHER = "_other"              # the pill for LoRAs that fit none of the models
    CONFIRM_MS = 4000             # how long "Click again to uninstall" waits

    def __init__(self, owner, tab="installed", model=None):
        self.owner = o = owner
        host = owner.host
        self.tab = tab if tab in self.TABS else "installed"
        self.page = 1                 # GitHub's next page
        self.model = model or owner.settings["model"]
        self.sort = civitai.SORTS[0]
        self.query = tk.StringVar()
        self.cursor = ""
        self.gen = 0                  # bumped by each search; late answers are dropped
        self.busy = False
        self.stop = threading.Event()
        self.images = {}              # key -> PhotoImage, kept or Tk drops them
        self.pics = {}                # key -> the Label that shows it
        self.cards = []               # the catalog cards shown, in order
        self.armed = None             # the record id whose Uninstall was clicked once
        win = self.win = tk.Toplevel(host)
        win.title("Add-ons")
        win.transient(host)
        o.skin(win, bg="bg")
        win.geometry("%dx%d" % (o.px(820), o.px(760)))
        win.minsize(o.px(620), o.px(520))
        win.protocol("WM_DELETE_WINDOW", self.close)
        head = o.frame(win)
        head.pack(side="top", fill="x", padx=o.px(16), pady=(o.px(14), 0))
        o.label(head, "Add-ons", font=host.f_title).pack(side="top", anchor="w")
        o.label(head, "LoRAs for the models you have. The form only offers a LoRA with a "
                "model it works with.", "muted", host.f_small,
                wraplength=o.px(760)).pack(side="top", fill="x")
        self.model_row = o.frame(win)
        self.model_row.pack(side="top", fill="x", padx=o.px(16), pady=(o.px(10), 0))
        self.tab_row = o.frame(win)
        self.tab_row.pack(side="top", fill="x", padx=o.px(16), pady=(o.px(8), 0))
        self.tools = o.frame(win)
        self.tools.pack(side="top", fill="x", padx=o.px(16), pady=(o.px(8), 0))
        foot = o.frame(win)
        foot.pack(side="bottom", fill="x", padx=o.px(16), pady=o.px(12))
        o.button(foot, "Close", self.close, kind="ghost").pack(side="right")
        o.button(foot, "CivitAI key…", self.civitai_key, kind="ghost").pack(
            side="right", padx=(0, o.px(6)))
        o.button(foot, "Library…", owner.edit_loras, kind="ghost").pack(
            side="right", padx=(0, o.px(6)))
        self.msg = o.label(foot, "", "muted", host.f_small, wraplength=o.px(470))
        self.msg.pack(side="left", fill="x", expand=True)
        scroll, self.list = o.scrolled(win)
        scroll.pack(side="top", fill="both", expand=True, padx=o.px(16), pady=(o.px(8), 0))
        self.show()

    # ------------------------------------------------------------ plumbing
    def alive(self):
        try:
            return bool(self.win.winfo_exists())
        except tk.TclError:
            return False

    def close(self):
        self.stop.set()
        self.win.destroy()

    def status(self, text, role="muted"):
        if self.alive():
            self.msg.config(text=text)
            self.owner.skin(self.msg, bg="bg", fg=role)

    def call(self, fn):
        """From a worker: run `fn` on the UI thread if the window is open."""
        self.owner._post("call", lambda: fn() if self.alive() else None)

    def spawn(self, fn, *args):
        self.owner.host._spawn(self.owner.s.event_id, fn, *args)

    def client(self):
        return civitai.Client(civitai.load_token(self.owner.studio.lib.root))

    def thumbs_dir(self):
        return os.path.join(self.owner.studio.lib.root, "addon-thumbs")

    def model_rec(self):
        return self.owner.studio.lib.get("models", self.model)

    def library_changed(self):
        """The form rebuilds from the library (rows, menus, presets)."""
        self.owner._post("library")

    # -------------------------------------------------------------- layout
    def show(self):
        o, lib = self.owner, self.owner.studio.lib
        models = lib.all("models")
        if self.model != self.OTHER and not lib.get("models", self.model):
            self.model = models[0]["id"] if models else self.OTHER
        for w in self.model_row.winfo_children() + self.tab_row.winfo_children() + \
                self.tools.winfo_children() + self.list.winfo_children():
            w.destroy()
        self.images, self.pics, self.cards, self.armed = {}, {}, [], None
        self.box = self.list          # where cards go
        o.label(self.model_row, "For", "muted", o.host.f_small).pack(side="left")
        for m in models + [{"id": self.OTHER, "label": "Other models"}]:
            o.button(self.model_row, m["label"], lambda mid=m["id"]: self.pick_model(mid),
                     kind="accent" if m["id"] == self.model else "quiet").pack(
                side="left", padx=(o.px(6), 0))
        for key, text in self.TABS:
            o.button(self.tab_row, text, lambda k=key: self.pick_tab(k),
                     kind="accent" if key == self.tab else "ghost").pack(
                side="left", padx=(0, o.px(6)))
        self.list.canvas.yview_moveto(0)
        if self.tab == "installed":
            self.show_installed()
        elif self.tab == "huggingface":
            self.show_hf()
        elif self.tab == "github":
            self.show_github()
        else:
            self.show_catalog()

    def pick_model(self, mid):
        self.model = mid
        self.gen += 1
        self.show()

    def pick_tab(self, key):
        self.tab = key
        self.gen += 1
        self.query.set("")
        self.show()

    def heading(self, text, about=""):
        o = self.owner
        o.cap(self.list, text)
        if about:
            o.label(self.list, about, "faint", o.host.f_small,
                    wraplength=o.px(740)).pack(side="top", fill="x")

    def card_frame(self):
        """A card: picture on the left, words in the middle, buttons right."""
        o = self.owner
        card = o.frame(self.box, bg="card")
        card.pack(side="top", fill="x", pady=(o.px(6), 0))
        holder = o.frame(card, bg="card")      # a fixed square, whatever the picture's shape
        holder.config(width=o.px(92), height=o.px(92))
        holder.pack_propagate(False)
        holder.pack(side="left", padx=o.px(8), pady=o.px(8), anchor="n")
        pic = o.label(holder, "", "faint", o.host.f_small, bg="card")
        pic.config(anchor="center", justify="center")
        pic.pack(fill="both", expand=True)
        right = o.frame(card, bg="card")
        right.pack(side="right", padx=o.px(8), pady=o.px(8), anchor="n")
        mid = o.frame(card, bg="card")
        mid.pack(side="left", fill="x", expand=True, pady=o.px(8))
        return card, pic, mid, right

    def set_pictures(self, paths):
        """{key: PNG path} into the cards' picture labels."""
        side = self.owner.px(84)
        for key, path in paths.items():
            lbl = self.pics.get(key)
            if lbl is None or not lbl.winfo_exists():
                continue
            img = photo_at(path, side, self.win)
            if img is not None:
                self.images[key] = img
                lbl.config(image=img, text="")
                if not hasattr(lbl, "_peek"):
                    peek(lbl, lambda l=lbl: l.peek_path)
                lbl.peek_path = path

    # ----------------------------------------------------------- installed
    def show_installed(self):
        o, lib = self.owner, self.owner.studio.lib
        if self.model == self.OTHER:
            recs = catalog.fits_none(lib)
            self.heading("Fits none of your models (%d)" % len(recs),
                         "Made for a model you do not generate with, so the form never offers "
                         "them. Uninstall to free the space, or set the model if it is wrong.")
            for rec in recs:
                self.installed_card(rec)
            if not recs:
                o.label(self.list, "None.", "muted").pack(side="top", anchor="w")
        else:
            model = self.model_rec()
            fits, unknown = catalog.sorted_for(lib, model)
            fam = ig.FAMILIES.get(model["family"], model["family"])
            self.heading("Works with %s (%d)" % (model["label"], len(fits)),
                         "Made for %s. Turned off, a LoRA stays installed but is not offered "
                         "on the form." % fam)
            for rec in fits:
                self.installed_card(rec)
            if not fits:
                row = o.frame(self.list)
                row.pack(side="top", fill="x", pady=o.px(6))
                o.label(row, "Nothing installed for %s yet." % model["label"], "muted").pack(
                    side="left")
                o.button(row, "Browse the catalog", lambda: self.pick_tab("catalog"),
                         kind="accent").pack(side="left", padx=o.px(8))
            if unknown:
                self.heading("Model not set (%d)" % len(unknown),
                             "Offered with every model, marked “may not work”, until "
                             "you say which model each was made for.")
                for rec in unknown:
                    self.installed_card(rec)
        shown = [r for r in lib.all("loras") if r["id"] in self.pics]
        folder = self.thumbs_dir()

        def work():
            pics = catalog.previews(shown, folder)
            self.call(lambda: self.set_pictures(pics))
        self.spawn(work)

    def installed_card(self, rec):
        o, host, studio = self.owner, self.owner.host, self.owner.studio
        card, pic, mid, right = self.card_frame()
        self.pics[rec["id"]] = pic
        on = rec.get("enabled", True)
        o.label(mid, rec["name"], "text" if on else "faint", host.f_bold, bg="card",
                wraplength=o.px(440)).pack(side="top", fill="x")
        bits = [rec["category"], "strength %.2g" % rec["strength"]]
        if rec.get("always"):
            bits.append("always on")
        if not on:
            bits.insert(0, "off")
        o.label(mid, " · ".join(bits), "muted", host.f_small, bg="card").pack(
            side="top", fill="x")
        if rec.get("trigger"):
            o.label(mid, "Trigger: " + rec["trigger"], "faint", host.f_small, bg="card",
                    wraplength=o.px(440)).pack(side="top", fill="x")
        where_on, where_off = catalog.where_installed(rec, studio.backends(),
                                                      studio.inventories)
        if where_on or where_off:
            text = ("On " + ", ".join(where_on) if where_on else "") + (
                ("; " if where_on else "") + "not on " + ", ".join(where_off)
                if where_off else "")
            o.label(mid, text, "muted" if where_on else "warn", host.f_small,
                    bg="card").pack(side="top", fill="x")
        fams = [("", "Model not set")] + list(ig.FAMILIES.items())
        o.choice(mid, [(k, "Made for " + v if k else v) for k, v in fams], rec["family"],
                 lambda v, r=rec: self.set_family(r, v), bg="card").pack(
            side="top", anchor="w", pady=(o.px(4), 0))
        def switch_row(text, key, val):
            row = o.frame(right, bg="card")
            row.pack(side="top", fill="x", pady=(0, o.px(4)))
            var = tk.BooleanVar(value=bool(val))
            o.switch(row, var, lambda r=rec, k=key: self.flip(r, k), bg="card").pack(
                side="left")
            o.label(row, text, "text", host.f_small, bg="card").pack(
                side="left", padx=(o.px(6), 0))
        switch_row("On" if on else "Off", "enabled", on)
        switch_row("Always on", "always", rec.get("always"))
        if rec.get("source", "").startswith("https://civitai.com/"):
            o.button(right, "Page", lambda u=rec["source"]: webbrowser.open(u),
                     kind="option", bg="card").pack(side="top", fill="x", pady=(o.px(4), 0))
        b = o.button(right, "Uninstall", lambda: None, kind="option", bg="card")
        b.command = lambda r=rec, b=b: self.uninstall(r, b)
        b.pack(side="top", fill="x", pady=(o.px(4), 0))
        return card

    def save_library(self, what):
        try:
            self.owner.studio.lib.save("loras")
        except OSError as e:
            self.status("Could not save the library: %s" % e, "err")
            return False
        self.library_changed()
        self.status(what, "ok")
        return True

    def flip(self, rec, key):
        rec[key] = not rec.get(key, key == "enabled")
        if key == "enabled":
            what = "%s is %s." % (rec["name"], "on" if rec[key] else
                                  "off: installed, but not offered on the form")
        else:
            what = "%s is %s." % (rec["name"], "always on for the models it suits"
                                  if rec[key] else "no longer always on")
        if self.save_library(what):
            self.show()

    def set_family(self, rec, fam):
        rec["family"] = fam
        what = "%s: made for %s." % (rec["name"], ig.FAMILIES.get(fam, "no model set"))
        if self.save_library(what):
            self.show()

    def uninstall(self, rec, button):
        """First click arms it; the second, within CONFIRM_MS, uninstalls. A
        LoRA with no file on this PC is turned off instead."""
        studio = self.owner.studio
        if self.armed != rec["id"]:
            self.armed = rec["id"]
            button.set(text="Click again")
            used = catalog.users_of(studio.lib, rec)
            self.status("Uninstall %s? Its file goes to the Recycle Bin.%s" % (
                rec["name"], " Used by " + ", ".join(used) + "." if used else ""), "warn")
            self.win.after(self.CONFIRM_MS, lambda: self.disarm(rec["id"], button))
            return
        self.armed = None
        try:
            paths = catalog.uninstall(studio.lib, rec, studio.backends())
        except ValueError:
            rec["enabled"] = False
            if self.save_library("%s's file is not on this PC (only on another machine), "
                                 "so it is turned off instead." % rec["name"]):
                self.show()
            return
        except OSError as e:
            self.status("Could not uninstall %s: %s" % (rec["name"], e), "err")
            return
        self.library_changed()
        self.owner.refresh_backends()
        self.status("Uninstalled %s: %s in the Recycle Bin." % (
            rec["name"], ", ".join(os.path.basename(p) for p in paths)), "ok")
        self.show()

    def disarm(self, rid, button):
        if self.armed == rid and self.alive() and button.winfo_exists():
            self.armed = None
            button.set(text="Uninstall")
            self.status("")

    # ------------------------------------------------------------- catalog
    def show_catalog(self):
        o = self.owner
        model = self.model_rec() if self.model != self.OTHER else None
        if model is None:
            o.label(self.list, "Pick one of your models above to browse LoRAs made for it.",
                    "muted").pack(side="top", anchor="w", pady=o.px(8))
            return
        bases = catalog.bases_for(model)
        if not bases:
            o.label(self.list, "CivitAI files no LoRAs under %s's model family (%s)." % (
                model["label"], model["family"] or "not set"), "muted",
                wraplength=o.px(740)).pack(side="top", anchor="w", pady=o.px(8))
            return
        e = o.host._entry(self.tools, self.query)
        e.master.pack(side="left", fill="x", expand=True)
        e.bind("<Return>", lambda ev: self.search())
        o.choice(self.tools, [(s, s) for s in civitai.SORTS], self.sort, self.set_sort).pack(
            side="left", padx=(o.px(6), 0))
        o.button(self.tools, "Search", self.search, kind="accent").pack(
            side="left", padx=(o.px(6), 0))
        self.heading("CivitAI LoRAs for %s" % model["label"],
                     "Listed under %s. Only pictures CivitAI rates PG or PG-13 are shown; "
                     "check a LoRA's page before installing." % ", ".join(bases))
        self.box = o.frame(self.list)
        self.box.pack(side="top", fill="x")
        self.search()

    def set_sort(self, value):
        self.sort = value
        self.search()

    def search(self, more=False):
        """A page of the catalog, on a worker. `more` appends the next page."""
        model = self.model_rec()
        if model is None:
            return
        self.gen += 1
        gen, cursor = self.gen, self.cursor if more else ""
        if not more:
            self.cursor = ""
            for w in self.box.winfo_children():
                w.destroy()
            self.cards = []
        self.busy = True
        self.status("Asking CivitAI" + ELLIPSIS, "accent")
        client, query, sort = self.client(), self.query.get(), self.sort
        folder, stop = self.thumbs_dir(), self.stop

        def work():
            try:
                cards, nxt = catalog.search(client, model, query, sort, cursor)
            except civitai.CivitAIError as e:
                self.call(lambda: self.searched(gen, [], "", str(e)))
                return
            self.call(lambda: self.searched(gen, cards, nxt, ""))
            pics = catalog.thumbnails(client, {c["version_id"]: c["preview_url"]
                                               for c in cards}, folder, stop)
            self.call(lambda: gen == self.gen and self.set_pictures(pics))
        self.spawn(work)

    def searched(self, gen, cards, nxt, error):
        if gen != self.gen:
            return
        self.busy = False
        o = self.owner
        for w in self.box.winfo_children():
            if getattr(w, "more", False):
                w.destroy()
        if error:
            self.status(error, "err")
            return
        self.cursor = nxt
        self.cards += cards
        for c in cards:
            self.catalog_card(c)
        if not self.cards:
            o.label(self.box, "Nothing found.", "muted").pack(side="top", anchor="w",
                                                                pady=o.px(8))
        if nxt:
            b = o.button(self.box, "More", lambda: self.search(more=True), kind="ghost")
            b.more = True
            b.pack(side="top", pady=o.px(10))
        self.status("%d LoRAs for %s." % (len(self.cards), self.model_rec()["label"]))

    def catalog_card(self, c):
        o, host = self.owner, self.owner.host
        card, pic, mid, right = self.card_frame()
        pic.config(text="no safe\npreview" if not c["preview_url"] else "")
        self.pics[c["version_id"]] = pic
        name = c["name"] + (" · " + c["version"] if c["version"] else "")
        o.label(mid, name, "text", host.f_bold, bg="card",
                wraplength=o.px(440)).pack(side="top", fill="x")
        bits = [x for x in ("by " + c["creator"] if c["creator"] else "",
                            "↓ " + catalog.human_count(c["downloads"]),
                            c["base_model"], c["category"],
                            "%.0f MB" % (c["size"] / 1048576.0) if c["size"] else "") if x]
        o.label(mid, " · ".join(bits), "muted", host.f_small, bg="card").pack(
            side="top", fill="x")
        if c["about"]:
            o.label(mid, c["about"], "faint", host.f_small, bg="card",
                    wraplength=o.px(440)).pack(side="top", fill="x", pady=(o.px(2), 0))
        if c["trigger"]:
            o.label(mid, "Trigger: " + c["trigger"], "faint", host.f_small, bg="card",
                    wraplength=o.px(440)).pack(side="top", fill="x")
        have = catalog.installed_as(self.owner.studio.lib, c)
        b = o.button(right, "Installed" if have else "Install", lambda: None,
                     kind="option" if have else "accent", bg="card")
        b.command = lambda c=c, b=b: self.install(c, b)
        if have:
            b.set(state="disabled")
        b.pack(side="top", fill="x")
        o.button(right, "Page", lambda u=c["link"]: webbrowser.open(u), kind="option",
                 bg="card").pack(side="top", fill="x", pady=(o.px(4), 0))
        return card

    def install_folder(self):
        """(backend id, LoRA folder on this PC) to install into: a backend
        that has this model ready first, else any with a folder here."""
        studio = self.owner.studio
        folders = [bid for bid, _ in self.owner.lora_folders()]
        if not folders:
            return None
        ready = studio.readiness(self.model_rec()) if self.model_rec() else {}
        bid = next((b for b in folders if ready.get(b, ("",))[0] == "ready"), folders[0])
        return bid, studio.backend(bid)["lora_dir"]

    def install(self, c, button):
        where = self.install_folder()
        if where is None:
            self.status("No backend has a LoRA folder on this PC, so there is nowhere to put "
                        "the file. Set one in Backends… (LoRA folder).", "err")
            return
        bid, folder = where
        button.set(state="disabled", text="Installing" + ELLIPSIS)
        lib, client, stop = self.owner.studio.lib, self.client(), self.stop

        def say(text):
            self.call(lambda: self.status(" ".join(str(text).split()), "accent"))

        def work():
            try:
                rec, _new = catalog.install(lib, client, c, folder, say, stop)
            except Exception as e:           # said; the window stays usable
                why = str(e) if isinstance(e, civitai.CivitAIError) else "%s: %s" % (
                    type(e).__name__, e)
                self.call(lambda: self.installed(c, button, bid, None, why))
                return
            self.call(lambda: self.installed(c, button, bid, rec, ""))
        self.spawn(work)

    def installed(self, c, button, bid, rec, error):
        if button.winfo_exists():
            button.set(state="normal" if error else "disabled",
                       text="Install" if error else "Installed")
        if error:
            key = "API key" in error or "401" in error or "403" in error
            self.status("Could not install %s: %s%s" % (
                c["name"], error, " Paste one under CivitAI key…" if key else ""), "err")
            return
        self.library_changed()
        self.owner.refresh_backends()     # so the form knows the file is there
        self.status("Installed %s into %s. It is on the form's Add LoRA menu for %s." % (
            rec["name"], self.owner.studio.backend(bid)["name"],
            self.model_rec()["label"]), "ok")

    def civitai_key(self):
        return ModelSourceSettings(self.owner, "civitai")

    # -------------------------------------------------------- Hugging Face
    def search_row(self, run):
        o = self.owner
        e = o.host._entry(self.tools, self.query)
        e.master.pack(side="left", fill="x", expand=True)
        e.bind("<Return>", lambda ev: run())
        o.button(self.tools, "Search", run, kind="accent").pack(side="left", padx=(o.px(6), 0))

    def show_hf(self):
        o = self.owner
        model = self.model_rec() if self.model != self.OTHER else None
        if model is None:
            o.label(self.list, "Pick one of your models above to browse LoRAs made for it.",
                    "muted").pack(side="top", anchor="w", pady=o.px(8))
            return
        repos = hub.repos_for(model)
        if not repos:
            o.label(self.list, "Hugging Face has no base repo on file for %s's model family "
                    "(%s)." % (model["label"], model["family"] or "not set"), "muted",
                    wraplength=o.px(740)).pack(side="top", anchor="w", pady=o.px(8))
            return
        self.search_row(self.search_hf)
        self.heading("Hugging Face LoRAs for %s" % model["label"],
                     "Adapters of %s, most downloaded first. Check a LoRA's page and "
                     "license before installing." % ", ".join(repos))
        self.box = o.frame(self.list)
        self.box.pack(side="top", fill="x")
        self.search_hf()

    def hf_token(self):
        return model_sources.load(self.owner.studio.lib.root, "huggingface")["token"]

    def search_hf(self):
        model = self.model_rec()
        if model is None:
            return
        self.gen += 1
        gen = self.gen
        for w in self.box.winfo_children():
            w.destroy()
        self.status("Asking Hugging Face" + ELLIPSIS, "accent")
        query, token = self.query.get(), self.hf_token()

        def work():
            try:
                cards, error = hub.hf_search(model, query, token), ""
            except hub.HubError as e:
                cards, error = [], str(e)
            self.call(lambda: self.hub_found(gen, cards, error, self.hf_card, 0))
        self.spawn(work)

    def hub_found(self, gen, cards, error, make, more):
        if gen != self.gen:
            return
        o = self.owner
        for w in self.box.winfo_children():
            if getattr(w, "more", False):
                w.destroy()
        if error:
            self.status(error, "err")
            return
        for c in cards:
            make(c)
        shown = len([w for w in self.box.winfo_children() if not getattr(w, "more", False)])
        if not shown:
            o.label(self.box, "Nothing found.", "muted").pack(side="top", anchor="w",
                                                                pady=o.px(8))
        self.page = more
        if more:
            b = o.button(self.box, "More", lambda: self.search_github(more=True), kind="ghost")
            b.more = True
            b.pack(side="top", pady=o.px(10))
        self.status("%d found." % shown)

    def hf_card(self, c):
        o, host = self.owner, self.owner.host
        card, pic, mid, right = self.card_frame()
        pic.config(text="Hugging\nFace")
        o.label(mid, c["name"], "text", host.f_bold, bg="card",
                wraplength=o.px(440)).pack(side="top", fill="x")
        bits = [x for x in ("by " + c["creator"], "↓ " + catalog.human_count(c["downloads"]),
                            "♥ %d" % c["likes"], c["license"]) if x]
        o.label(mid, " · ".join(bits), "muted", host.f_small, bg="card").pack(
            side="top", fill="x")
        have = any(r.get("source") == c["link"] for r in self.owner.studio.lib.all("loras"))
        b = o.button(right, "Installed" if have else "Install", lambda: None,
                     kind="option" if have else "accent", bg="card")
        b.command = lambda c=c, b=b: self.install_hf(c, b)
        if have:
            b.set(state="disabled")
        b.pack(side="top", fill="x")
        o.button(right, "Page", lambda u=c["link"]: webbrowser.open(u), kind="option",
                 bg="card").pack(side="top", fill="x", pady=(o.px(4), 0))

    def install_hf(self, c, button):
        where = self.install_folder()
        if where is None:
            self.status("No backend has a LoRA folder on this PC, so there is nowhere to put "
                        "the file. Set one in Backends… (LoRA folder).", "err")
            return
        bid, folder = where
        button.set(state="disabled", text="Installing" + ELLIPSIS)
        lib, stop, token = self.owner.studio.lib, self.stop, self.hf_token()
        model = self.model_rec()

        def say(text):
            self.call(lambda: self.status(" ".join(str(text).split()), "accent"))

        def work():
            try:
                rec, _new = hub.hf_install(lib, c, folder, model["family"], token, say, stop)
            except Exception as e:           # said; the window stays usable
                why = str(e) if isinstance(e, (hub.HubError, civitai.CivitAIError)) else \
                    "%s: %s" % (type(e).__name__, e)
                self.call(lambda: self.installed(c, button, bid, None, why))
                return
            self.call(lambda: self.installed(c, button, bid, rec, ""))
        self.spawn(work)

    # -------------------------------------------------------------- GitHub
    def show_github(self):
        o = self.owner
        self.search_row(self.search_github)
        folder = hub.load_comfy_folder(self.owner.studio.lib.root)
        row = o.frame(self.list)
        row.pack(side="top", fill="x", pady=(o.px(4), 0))
        o.label(row, "ComfyUI folder: " + (folder or "not chosen yet"), "muted",
                o.host.f_small, wraplength=o.px(560)).pack(side="left")
        o.button(row, "Choose…", self.choose_comfy, kind="ghost").pack(side="right")
        self.heading("ComfyUI plugins on GitHub",
                     "Custom nodes tagged %s, most starred first. A plugin is someone "
                     "else's code that ComfyUI runs: install only what you trust. Its "
                     "Python requirements are not installed for you; restart ComfyUI "
                     "after installing." % hub.GH_TOPIC)
        self.box = o.frame(self.list)
        self.box.pack(side="top", fill="x")
        self.search_github()

    def choose_comfy(self):
        folder = filedialog.askdirectory(parent=self.win, mustexist=True,
                                         title="Choose ComfyUI folder (contains main.py)")
        if not folder:
            return None
        try:
            hub.custom_nodes(folder)
            hub.save_comfy_folder(self.owner.studio.lib.root, folder)
        except (hub.HubError, OSError) as e:
            self.status(str(e), "err")
            return None
        if self.tab == "github":
            self.show()
        return folder

    def search_github(self, more=False):
        self.gen += 1
        gen, page = self.gen, (self.page or 1) if more else 1
        if not more:
            for w in self.box.winfo_children():
                w.destroy()
        self.status("Asking GitHub" + ELLIPSIS, "accent")
        query = self.query.get()

        def work():
            try:
                cards, nxt = hub.gh_search(query, page)
                error = ""
            except hub.HubError as e:
                cards, nxt, error = [], 0, str(e)
            self.call(lambda: self.hub_found(gen, cards, error, self.gh_card, nxt))
        self.spawn(work)

    def gh_card(self, c):
        o, host = self.owner, self.owner.host
        card, pic, mid, right = self.card_frame()
        pic.config(text="GitHub")
        o.label(mid, c["id"], "text", host.f_bold, bg="card",
                wraplength=o.px(440)).pack(side="top", fill="x")
        bits = [x for x in ("★ " + catalog.human_count(c["stars"]),
                            "updated " + c["updated"] if c["updated"] else "",
                            c["license"]) if x]
        o.label(mid, " · ".join(bits), "muted", host.f_small, bg="card").pack(
            side="top", fill="x")
        if c["about"]:
            o.label(mid, c["about"], "faint", host.f_small, bg="card",
                    wraplength=o.px(440)).pack(side="top", fill="x", pady=(o.px(2), 0))
        have = hub.gh_installed(hub.load_comfy_folder(self.owner.studio.lib.root), c)
        b = o.button(right, "Installed" if have else "Install", lambda: None,
                     kind="option" if have else "accent", bg="card")
        b.command = lambda c=c, b=b: self.install_github(c, b)
        if have:
            b.set(state="disabled")
        b.pack(side="top", fill="x")
        o.button(right, "Page", lambda u=c["link"]: webbrowser.open(u), kind="option",
                 bg="card").pack(side="top", fill="x", pady=(o.px(4), 0))

    def install_github(self, c, button):
        folder = hub.load_comfy_folder(self.owner.studio.lib.root) or self.choose_comfy()
        if not folder:
            return
        if not messagebox.askyesno(
                "Install plugin", "Install %s into ComfyUI's custom_nodes?\n\nIt is "
                "third-party code that ComfyUI will run when it next starts. Its Python "
                "requirements are not installed." % c["id"], parent=self.win):
            return
        button.set(state="disabled", text="Installing" + ELLIPSIS)

        def say(text):
            self.call(lambda: self.status(text, "accent"))

        def work():
            try:
                path, error = hub.gh_install(c, folder, say=say), ""
            except Exception as e:           # said; the window stays usable
                path = ""
                error = str(e) if isinstance(e, hub.HubError) else "%s: %s" % (
                    type(e).__name__, e)
            self.call(lambda: done(path, error))

        def done(path, error):
            if button.winfo_exists():
                button.set(state="normal" if error else "disabled",
                           text="Install" if error else "Installed")
            if error:
                self.status("Could not install %s: %s" % (c["id"], error), "err")
                return
            reqs = os.path.isfile(os.path.join(path, "requirements.txt"))
            self.status("Installed %s into %s. Restart ComfyUI to load it.%s" % (
                c["id"], path, " It lists Python requirements (requirements.txt); install "
                "them into ComfyUI's Python first." if reqs else ""), "ok")
        self.spawn(work)
