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
import subprocess
import sys
import time
import tkinter as tk
from tkinter import filedialog

import studio_imagegen as ig
import studio_pose as sp
import studio_scene_ui

THUMB = 72                    # px, before the display's scale
STYLE_TILE = 104              # px, before the display's scale; the examples are 208
HISTORY_PAGE = 40
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
               "refining": "accent", "complete": "ok", "failed": "err", "cancelled": "faint"}
# Where each status sits on the Queued -> ... -> Complete strip.
STAGE_AT = {"queued": 0, "uploading": 0, "loading": 1, "running": 2, "sampling": 2,
            "refining": 2, "decoding": 3, "complete": 4}
READY_MARK = {"ready": "✓", "missing": "✗", "offline": "○", "disabled": "–",
              "unchecked": "?"}


def open_path(path, select=False):
    if sys.platform == "win32":
        if select:
            subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])
        else:
            os.startfile(path)            # noqa - Windows only, like the rest of the app


def photo(path, box):
    """A PhotoImage of `path` shrunk by a whole factor to fit `box` px, or None.
    Tk reads PNG and GIF itself; anything else has no preview."""
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
        self.refs = {}                # kind -> local path
        self.pose = None              # the drawn stick figure (PoseEditor), or None
        self.planned_size = (1024, 1024)   # the picture's size, as last composed
        self.adv = {}                 # setting -> StringVar
        self.text = {}                # look slot and camera setting -> StringVar
        self.sliders = {}             # weight, muscle, stature -> IntVar
        self.item_refs = {}           # item -> picture, from the character
        self.look_section = ig.LOOKS[0][0]
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
        self.adv_open = False
        self.scene_builder = None     # the Scene Builder window, while it is open
        self._build(session.frame)
        for problem in self.studio.lib.problems:
            self.say(problem, "warn")

    def start(self):
        """First view of the tab (`Chat._ensure`): ask the backends how they
        are. Not at construction - a tab is built before it is looked at,
        and building must not reach the network."""
        self.refresh_backends()

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

    def choice(self, parent, items, current, on_pick, bg="bg", width=None):
        """A dropdown: a Pill that posts a menu of (value, label) pairs. The
        window has no ttk and wants none (palette roles, Pills everywhere)."""
        labels = dict(items)
        pill = self.button(parent, labels.get(current, current or "Choose") + "  ▾",
                           lambda: None, kind="option", bg=bg)

        def post():
            menu = tk.Menu(pill, tearoff=0)
            self.skin(menu, bg="card", fg="text", activebackground="sel",
                      activeforeground="text")
            for value, text in items:
                if value is None:
                    menu.add_separator()
                else:
                    menu.add_command(label=text, command=lambda v=value: pick(v))
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
        self.button(head, "Backends…", self.edit_backends).pack(side="right")
        self.button(head, "Models…", self.edit_models).pack(side="right",
                                                                 padx=(0, self.px(6)))
        self.button(head, "Check", self.refresh_backends).pack(side="right",
                                                               padx=(0, self.px(6)))
        self.health_row = self.frame(head)
        self.health_row.pack(side="left", fill="x", expand=True)

        self.note = self.label(root, "", "muted")
        self.note.pack(side="top", fill="x", padx=self.px(20), pady=(0, self.px(4)))

        body = self.frame(root)
        body.pack(side="top", fill="both", expand=True, padx=self.px(14),
                  pady=(0, self.px(14)))
        # Fixed before expanding (AGENTS.md: pack order).
        left, self.form = self.scrolled(body)
        left.config(width=self.px(390))
        left.pack_propagate(False)
        left.pack(side="left", fill="y")
        right = self.frame(body)
        right.pack(side="left", fill="both", expand=True, padx=(self.px(14), 0))
        self._build_form(self.form)
        self._build_right(right)

    # -------------------------------------------------------------- the form
    def _build_form(self, f):
        pad = {"padx": (self.px(6), self.px(10))}
        self.cap(f, "Preset").pack(**pad)
        row = self.frame(f)
        row.pack(side="top", fill="x", **pad)
        self.preset_pill = self.choice(
            row, [(k, ig.PRESETS[k]["label"]) for k in ig.PRESET_ORDER],
            self.settings["preset"], self._set_preset)
        self.preset_pill.pack(side="left")
        self.preset_about = self.label(f, ig.PRESETS["standard"]["about"], "faint",
                                       self.host.f_small, wraplength=self.px(380))
        self.preset_about.pack(side="top", fill="x", **pad)

        self.cap(f, "Model and backend").pack(**pad)
        self.model_row = self.frame(f)
        self.model_row.pack(side="top", fill="x", **pad)

        self.cap(f, "Scene").pack(**pad)
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
        self.button(srow, "Scene Builder…", self.build_scene).pack(side="top", fill="x")
        self.label(srow, "Stage people and props in 3D, or make a scene from a picture; "
                   "the frame becomes the source picture.", "faint", self.host.f_small,
                   wraplength=self.px(380)).pack(side="top", fill="x",
                                                 pady=(self.px(3), 0))

        prow = self.frame(f)
        prow.pack(side="top", fill="x", pady=(self.px(12), 0), **pad)
        self.pc_pill = self.button(prow, "Person and camera  ▸",
                                   self._toggle_person, kind="ghost")
        self.pc_pill.pack(side="left")
        # Hidden until asked for; what is in it still goes into the prompt.
        self.pc_open = False
        self.pc_box = pb = self.frame(f)
        self.cap(pb, "Person").pack(**pad)
        crow = self.frame(pb)
        crow.pack(side="top", fill="x", **pad)
        self.char_row = self.frame(crow)
        self.char_row.pack(side="left")
        self.button(crow, "Creator…", lambda: self.edit_characters(), kind="ghost").pack(
            side="left", padx=(self.px(6), 0))
        self.button(crow, "Save as…", self.save_as_character, kind="ghost").pack(
            side="left", padx=(self.px(4), 0))
        self.person_box = self.frame(pb)
        self.person_box.pack(side="top", fill="x", pady=(self.px(4), 0), **pad)
        self.label(pb, "Who, and what they look like. Any of these can stay blank; a "
                   "character fills all but the expression.", "faint", self.host.f_small,
                   wraplength=self.px(380)).pack(side="top", fill="x",
                                                 pady=(self.px(6), 0), **pad)
        for key in ig.SLOTS:
            self.text[key] = tk.StringVar()
        for key in ig.SLIDER_KEYS:
            self.sliders[key] = tk.IntVar(value=0)
        self.look_tabs = self.frame(pb)
        self.look_tabs.pack(side="top", fill="x", **pad)
        self.look_box = self.frame(pb)
        self.look_box.pack(side="top", fill="x", **pad)
        self._show_looks(self.look_section)
        b = tk.Checkbutton(pb, text="Anatomy constants", variable=self.anatomy, anchor="w",
                           font=self.host.f_ui, bd=0, highlightthickness=0,
                           command=self._recheck)
        self.skin(b, bg="bg", fg="text", activebackground="bg", selectcolor="card",
                  activeforeground="text")
        b.pack(side="top", fill="x", pady=(self.px(8), 0), **pad)
        self.label(pb, "Every person: " + ig._and([pos for _, pos, _ in ig.ANATOMY]) + ".",
                   "faint", self.host.f_small, wraplength=self.px(380)).pack(
            side="top", fill="x", **pad)
        self.label(f, "Pictures of the clothes, hair and accessories (Clothes, Hair and "
                   "Accessories tabs) go into the picture itself: it is then made with "
                   "FLUX Kontext, which draws from them.", "faint", self.host.f_small,
                   wraplength=self.px(380)).pack(side="top", fill="x",
                                                 pady=(self.px(6), 0), **pad)

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

        self.cap(f, "References").pack(**pad)
        self.ref_box = self.frame(f)
        self.ref_box.pack(side="top", fill="x", **pad)
        self._build_refs()

        self.cap(f, "LoRAs").pack(**pad)
        self.lora_box = self.frame(f)
        self.lora_box.pack(side="top", fill="x", **pad)
        lrow = self.frame(f)
        lrow.pack(side="top", fill="x", pady=(self.px(4), 0), **pad)
        self.add_lora_pill = self.button(lrow, "Add LoRA  ▾", self._post_lora_menu)
        self.add_lora_pill.pack(side="left")
        self.button(lrow, "Library…", self.edit_loras, kind="ghost").pack(
            side="left", padx=(self.px(6), 0))

        arow = self.frame(f)
        arow.pack(side="top", fill="x", pady=(self.px(12), 0), **pad)
        self.adv_pill = self.button(arow, "Advanced  ▸", self._toggle_advanced,
                                    kind="ghost")
        self.adv_pill.pack(side="left")
        self.adv_box = self.frame(f)
        self._build_advanced(self.adv_box)

        self.warn = self.label(f, "", "warn", self.host.f_small, wraplength=self.px(380))
        self.warn.pack(side="top", fill="x", pady=(self.px(10), 0), **pad)
        # Where the job will go, and why, before Generate (plan_route).
        self.route_note = self.label(f, "", "muted", self.host.f_small,
                                     wraplength=self.px(380))
        self.route_note.pack(side="top", fill="x", pady=(self.px(6), 0), **pad)
        grow = self.frame(f)
        grow.pack(side="top", fill="x", pady=(self.px(10), self.px(18)), **pad)
        self.go = self.button(grow, "Generate", self.generate, kind="accent")
        self.go.pack(side="left")
        self.label(grow, "Ctrl+Enter in the scene", "faint", self.host.f_small).pack(
            side="left", padx=(self.px(10), 0))
        # The Visual Critic (studio_critic): the vision model checks the
        # picture and the faults it finds are redrawn, up to three passes.
        b = tk.Checkbutton(f, text="Automatic refinement (a vision model checks the picture "
                           "and fixes what is wrong)", variable=self.auto_refine,
                           anchor="w", font=self.host.f_small, bd=0, highlightthickness=0,
                           wraplength=self.px(380), justify="left")
        self.skin(b, bg="bg", fg="muted", activebackground="bg", selectcolor="card",
                  activeforeground="text")
        b.pack(side="top", fill="x", pady=(0, self.px(12)), before=grow, **pad)
        self._rebuild_choices()

    def _rebuild_choices(self):
        """Everything drawn from the library: models, backends, people, styles,
        LoRA rows. Called again after any editor saves."""
        lib = self.studio.lib
        self._rebuild_models()
        for w in self.char_row.winfo_children():
            w.destroy()
        chars = [("", "No character")] + [(c["id"], c["name"]) for c in lib.all("characters")]
        if self.settings["character"] not in dict(chars):
            self.settings["character"] = ""
        self.choice(self.char_row, chars, self.settings["character"],
                    self._set_character).pack(side="left")
        for w in self.person_box.winfo_children():
            w.destroy()
        old = {k: (b.get(), s.get()) for k, (b, s, _) in self.idents.items()}
        self.idents = {}
        for ident in lib.all("identities"):
            on, st = old.get(ident["id"], (False, ident["strength"]))
            bvar, svar = tk.BooleanVar(value=on), tk.DoubleVar(value=st)
            row = self.frame(self.person_box)
            row.pack(side="top", fill="x")
            box = tk.Checkbutton(row, text=ident["name"], variable=bvar, font=self.host.f_ui,
                                 anchor="w", bd=0, highlightthickness=0,
                                 command=lambda i=ident["id"]: self._toggle_ident(i))
            self.skin(box, bg="bg", fg="text", activebackground="bg", selectcolor="card",
                      activeforeground="text")
            box.pack(side="left")
            sc = self.slider(row, svar, 0.0, 1.5, command=lambda _v: self._recheck())
            self.idents[ident["id"]] = (bvar, svar, sc)
            if on:
                sc.pack(side="right", fill="x", expand=True, padx=(self.px(8), 0))
        if not self.idents:
            self.label(self.person_box, "No identities yet.", "faint").pack(side="left")
        self.button(self.person_box, "Identities…", self.edit_identities,
                    kind="ghost").pack(side="top", anchor="w", pady=(self.px(4), 0))

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
                    lambda v: self._set("model", v)).pack(side="top", anchor="w")
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

    def _set_preset(self, key):
        self.settings["preset"] = key
        self.preset_about.config(text=ig.PRESETS[key]["about"])
        if not self.refine_set:
            self.refine.set(bool(ig.PRESETS[key]["values"].get("refine")))
        if not self.faces_set:
            self.faces.set(bool(ig.PRESETS[key]["values"].get("face_detail")))
        self._recheck()

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
                pic = tk.Label(tile, image=img, bd=0)
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
        bvar, _, sc = self.idents[iid]
        if bvar.get():
            sc.pack(side="right", fill="x", expand=True, padx=(self.px(8), 0))
        else:
            sc.pack_forget()
        self._recheck()

    # ------------------------------------------------------------- the look
    def _set_character(self, cid):
        """Put a character's look on the form: every slot and slider it keeps
        (blank where it has none, so the last one's beard does not stay),
        its item pictures, and its identity ticked. The expression is the
        picture's, and stays."""
        self.settings["character"] = cid
        rec = self.studio.lib.get("characters", cid) if cid else None
        if rec is not None:
            for key in ig.CHARACTER_KEYS:
                if key in self.sliders:
                    self.sliders[key].set(int(rec["looks"].get(key, 0)))
                else:
                    self.text[key].set(rec["looks"].get(key, ""))
            self.item_refs = dict(rec["item_refs"])
            if rec["identity"] in self.idents:
                bvar = self.idents[rec["identity"]][0]
                if not bvar.get():
                    bvar.set(True)
                    self._toggle_ident(rec["identity"])
            self._show_looks(self.look_section)
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
        """Weight, muscle and height as the game's sliders: -3..3, a word
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
        tabs are."""
        self.look_section = section
        for box in (self.look_tabs, self.look_box):
            for w in box.winfo_children():
                w.destroy()
        for name, _ in ig.LOOKS:
            self.button(self.look_tabs, name, lambda n=name: self._show_looks(n),
                        kind="accent" if name == section else "quiet",
                        font=self.host.f_small, padx=self.px(6), pady=self.px(2)).pack(
                side="left", padx=(0, self.px(2)), pady=(self.px(6), self.px(2)))
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
                           self._set_item)

    def item_rows(self, p, items, refs, choose, drop):
        """A row per item worn - its picture, name and file, Picture… and ×:
        what the glasses or the necklace actually look like. The form's look
        tabs and the creator's both show these; Generate draws from them."""
        o, host = self, self.host
        o.label(p, "ITEM PICTURES", "faint", host.f_small).pack(
            side="top", fill="x", pady=(o.px(12), o.px(2)))
        if not items:
            o.label(p, "Choose an item above to give it a picture.", "faint",
                    host.f_small).pack(side="top", fill="x")
            return
        for item in items:
            row = o.frame(p)
            row.pack(side="top", fill="x", pady=(0, o.px(3)))
            path = refs.get(item)
            img = photo(path, o.px(40)) if path and os.path.isfile(path) else None
            if img is not None:
                o.keep.append(img)
                tk.Label(row, image=img, bd=0).pack(side="left", padx=(0, o.px(6)))
            o.label(row, item, "text", width=18).pack(side="left")
            o.label(row, os.path.basename(path) if path else "no picture", "faint",
                    host.f_small).pack(side="left", fill="x", expand=True)
            if path:
                o.button(row, "×", lambda i=item: drop(i, None), kind="ghost").pack(
                    side="right")
            o.button(row, "Picture…", lambda i=item: choose(i)).pack(
                side="right", padx=(o.px(4), 0))

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
            row.pack(side="top", fill="x", pady=(0, self.px(2)))
            self.label(row, label, "text", width=13).pack(side="left")
            clear = self.button(row, "×", lambda k=kind: self._clear_ref(k),
                                kind="ghost")
            clear.pack(side="right")
            self.button(row, "Choose…", lambda k=kind, a=about: self._pick_ref(k, a),
                        kind="quiet").pack(side="right", padx=(self.px(4), 0))
            if kind == "pose":
                self.button(row, "Draw…", self.edit_pose, kind="quiet").pack(
                    side="right", padx=(self.px(4), 0))
            name = self.label(row, "—", "faint", self.host.f_small)
            name.pack(side="left", fill="x", expand=True)
            self.ref_labels[kind] = name

    def _pick_ref(self, kind, about):
        path = filedialog.askopenfilename(
            parent=self.host, title=about,
            filetypes=[("Pictures", "*.png *.jpg *.jpeg *.webp *.bmp"), ("All files", "*.*")])
        if path:
            if kind == "pose":
                self.pose = None      # a picture of its own replaces the drawn one
            self._set_ref(kind, path)

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
    def _post_lora_menu(self):
        menu = tk.Menu(self.add_lora_pill, tearoff=0)
        self.skin(menu, bg="card", fg="text", activebackground="sel", activeforeground="text")
        by_cat = {}
        for r in self.studio.lib.all("loras"):
            by_cat.setdefault(r["category"], []).append(r)
        if not by_cat:
            menu.add_command(label="The library is empty - open Library… and Scan",
                             state="disabled")
        for cat in ig.CATEGORIES:
            if cat not in by_cat:
                continue
            sub = tk.Menu(menu, tearoff=0)
            self.skin(sub, bg="card", fg="text", activebackground="sel",
                      activeforeground="text")
            for r in sorted(by_cat[cat], key=lambda r: r["name"].lower()):
                fam = ig.FAMILIES.get(r["family"], r["family"])
                sub.add_command(label="%s%s" % (r["name"], "  (%s)" % fam if fam else ""),
                                command=lambda i=r["id"]: self._add_lora(i))
            menu.add_cascade(label=cat, menu=sub)
        p = self.add_lora_pill
        menu.tk_popup(p.winfo_rootx(), p.winfo_rooty() + p.winfo_height())

    def _add_lora(self, lid, strength=None, recheck=True):
        rec = self.studio.lib.get("loras", lid)
        if rec is None or any(r["id"] == lid for r in self.loras):
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
        self.pc_open = not self.pc_open
        self.pc_pill.set(text="Person and camera  " + ("▾" if self.pc_open else "▸"))
        if self.pc_open:
            self.pc_box.pack(side="top", fill="x", after=self.pc_pill.master)
        else:
            self.pc_box.pack_forget()

    def _toggle_advanced(self):
        self.adv_open = not self.adv_open
        self.adv_pill.set(text="Advanced  " + ("▾" if self.adv_open else "▸"))
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
        chosen = {d["id"]: d.get("strength") for d in s.get("identities") or []
                  if isinstance(d, dict)}
        for iid, (bv, sv, _) in self.idents.items():
            bv.set(iid in chosen)
            if chosen.get(iid) is not None:
                sv.set(chosen[iid])
        self.scene.delete("1.0", "end")
        self.scene.insert("1.0", s.get("scene") or "")
        for key, var in self.text.items():
            var.set(s.get(key) or "")
        for key, var in self.sliders.items():
            var.set(int(s.get(key) or 0))
        self.item_refs = ig.clean_item_refs(s.get("item_refs"))
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
        for r in list(self.loras):
            r["row"].destroy()
        self.loras = []
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
        self.preset_pill.set(text=ig.PRESETS.get(self.settings["preset"], ig.PRESETS[
            "standard"])["label"] + "  ▾")
        if not self.adv_open:
            self._toggle_advanced()
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
                return
            v = self.studio.preview(s, b).values
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
        return self.scene_builder

    def _submit(self, s):
        """Off the UI thread: routing may check a backend's health."""
        try:
            jobs = self.studio.submit(s)
        except ig.ComfyError as e:
            self._post("said", (str(e), "err"))
            self._post("health")
            return
        names = sorted({j.backend["name"] for j in jobs})
        self._post("submitted", jobs)
        self._post("said", ("Queued %d job%s on %s." % (len(jobs), "" if len(jobs) == 1
                                                        else "s", " and ".join(names)),
                            "muted"))
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
        self.act_again = self.button(acts, "Generate again", self._again_selected, bg="card")
        self.act_vary = self.button(acts, "New seed", lambda: self._again_selected(True),
                                    bg="card")
        self.act_reuse = self.button(acts, "Reuse settings", self._reuse_selected, bg="card")
        for p in (self.act_again, self.act_vary, self.act_reuse):
            p.pack(side="left", padx=(0, self.px(6)))
            p.set(state="disabled")
        self.caption = self.label(top, "", "muted", self.host.f_small, bg="card")
        self.caption.pack(side="bottom", fill="x", padx=self.px(12))
        self.wrap(self.caption, top, self.px(24))
        self.preview = tk.Label(top, bd=0, highlightthickness=0, text="Nothing yet",
                                font=self.host.f_ui, cursor="hand2")
        self.skin(self.preview, bg="card", fg="faint")
        self.preview.pack(side="top", fill="both", expand=True, pady=self.px(8))
        self.preview.bind("<Double-Button-1>", lambda ev: self._open_selected())
        self.preview.bind("<Button-3>", self._picture_menu)
        self.preview.bind("<Configure>", lambda ev: self._repaint_preview())

        tabs = self.frame(right)
        tabs.grid(row=1, column=0, sticky="ew", pady=(self.px(10), self.px(4)))
        self.tab_queue = self.button(tabs, "Queue", lambda: self._show_list("queue"),
                                     kind="option")
        self.tab_hist = self.button(tabs, "History", lambda: self._show_list("history"),
                                    kind="ghost")
        self.tab_queue.pack(side="left")
        self.tab_hist.pack(side="left", padx=(self.px(6), 0))
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
        menu.add_command(label="Show in folder", command=lambda: self._open_selected(True))
        menu.add_command(label="Copy path", command=lambda: (
            self.host.clipboard_clear(), self.host.clipboard_append(path)))
        menu.tk_popup(ev.x_root, ev.y_root)

    def _show_list(self, which):
        self.view = which
        self.tab_queue.roles = self.host.PILL_ROLES["option" if which == "queue" else "ghost"]
        self.tab_hist.roles = self.host.PILL_ROLES["option" if which == "history" else "ghost"]
        for p in (self.tab_queue, self.tab_hist):
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
        self.set_thumb(box, path)
        return box

    def set_thumb(self, box, path):
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
        preset = ig.PRESETS.get(s.get("preset"), {}).get("label", s.get("preset"))
        meta = self.label(right, "", "faint", self.host.f_small, bg="card")
        meta.pack(side="top", fill="x")
        self.wrap(meta, right, self.px(12))
        strip = self.frame(right, "card")
        strip.pack(side="top", fill="x", pady=(self.px(3), 0))
        stages = []
        for i, name in enumerate(ig.STAGES):
            if i:
                self.label(strip, "→", "faint", self.host.f_small, bg="card").pack(
                    side="left", padx=self.px(3))
            lbl = self.label(strip, name.capitalize(), "faint", self.host.f_small, bg="card")
            lbl.pack(side="left")
            stages.append(lbl)
        bar = tk.Canvas(right, height=self.px(4), highlightthickness=0, bd=0)
        self.skin(bar, bg="card")
        bar.pack(side="top", fill="x", pady=(self.px(4), 0), padx=(0, self.px(8)))
        detail = self.label(right, "", "faint", self.host.f_small, bg="card")
        detail.pack(side="top", fill="x")
        self.wrap(detail, right, self.px(12))
        widgets = {"row": row, "thumb": thumb, "status": status, "elapsed": elapsed,
                   "bar": bar, "detail": detail, "cancel": cancel, "meta": meta,
                   "stages": stages, "strip": strip,
                   "base": "%s · %s · %s · seed %s" % (
                       preset, model.get("label", s.get("model")), job.backend["name"],
                       s.get("seed")) if s.get("mode") != "dress" else
                   "Try On · %s · seed %s" % (job.backend["name"], s.get("seed"))}
        for w in (row, right, thumb, thumb.img, status, meta, detail):
            w.bind("<Button-1>", lambda ev: self._select(("job", job)))
        self.rows[job.id] = widgets
        self._paint_job(job)

    def _paint_job(self, job):
        w = self.rows.get(job.id)
        if w is None:
            return
        w["status"].config(text=job.status.capitalize())
        self.skin(w["status"], bg="card", fg=STATUS_ROLE.get(job.status, "muted"))
        loras = ", ".join("%s %.2f" % (l["name"], l["strength"])
                          for l in (job.plan.lora_meta if job.plan else []))
        w["meta"].config(text=w["base"] + (" · " + loras if loras else ""))
        w["detail"].config(text=job.detail or "")
        self.skin(w["detail"], bg="card", fg="err" if job.status == "failed" else "faint")
        at = STAGE_AT.get(job.status)
        if at is None:                    # failed or cancelled: the strip has said its piece
            w["strip"].pack_forget()
        else:
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
        if job.status == "complete" and job.outputs and job.id in self.rows:
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
        if job.status in ig.FINISHED and job.settings.get("scene_texture"):
            sb = self.scene_builder
            if sb is not None and sb.win.winfo_exists():
                sb.texture_done(job)
        if job.status in ig.FINISHED:
            self._paint_health()
            if job.status == "failed":
                self.say("A job on %s failed: %s" % (job.backend["name"], job.detail), "err")
                if self.selected is None or self.selected[0] == "job":
                    self._select(("job", job))
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
        self.button(btns, "Again", lambda: self._again(rec), bg="card").pack(
            side="right", padx=(0, self.px(6)))
        self.button(btns, "Reuse", lambda: self.reuse(rec["settings"]), bg="card").pack(
            side="right", padx=(0, self.px(4)))
        self.label(btns, rec.get("created", "")[5:16].replace("T", " "), "muted",
                   self.host.f_small, bg="card").pack(side="left")
        prompt = (rec.get("prompt") or "").replace("\n", " ")
        for text, role, font in ((prompt[:160] + ("\u2026" if len(prompt) > 160 else ""),
                                  "text", None), (self.describe(rec), "faint",
                                                  self.host.f_small)):
            lbl = self.label(right, text, role, font, bg="card")
            lbl.pack(side="top", fill="x")
            self.wrap(lbl, right, self.px(12))
        for w in (row, right, thumb, thumb.img):
            w.bind("<Button-1>", lambda ev: self._select(("record", rec)))

    def describe(self, rec):
        bits = [(rec.get("model") or {}).get("label") or "?",
                (rec.get("backend") or {}).get("name") or "?",
                "seed %s" % rec.get("seed"),
                "%sx%s" % (rec.get("width"), rec.get("height")),
                "%s steps" % rec.get("steps"),
                "%s/%s" % (rec.get("sampler"), rec.get("scheduler"))]
        if rec.get("guidance") is not None:
            bits.append("guidance %s" % rec["guidance"])
        if rec.get("refine"):
            bits.append("refined x%s at %s" % (rec["refine"].get("upscale"),
                                               rec["refine"].get("denoise")))
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
        path, text = None, ""
        if item[0] == "job":
            job = item[1]
            path = job.outputs[0] if job.outputs else None
            if job.record:
                text = self.clip(job.record["prompt"]) + "\n" + self.describe(job.record)
            else:
                text = (job.plan.prompt if job.plan else ig.summary(job.settings)) \
                    + "\n" + job.status.capitalize() + (": " + job.detail if job.detail
                                                        else "")
        else:
            rec = item[1]
            path = (rec.get("images") or [None])[0]
            text = self.clip(rec.get("prompt", "")) + "\n" + self.describe(rec)
            warn = rec.get("warnings") or []
            if warn:
                text += "\n" + "; ".join(warn)
        self.pending_preview = path
        self._repaint_preview()
        self.caption.config(text=text)
        rec = self._selected_record()
        for p in (self.act_again, self.act_vary, self.act_reuse):
            p.set(state="normal" if rec or item[0] == "job" else "disabled")

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
        """The same picture again (its seed, values and backend), or with
        `new_seed` the same settings on a fresh seed."""
        s = ig.again(rec, new_seed)
        self.say(("Same settings, new seed" if new_seed else
                  "Generating again with seed %s" % s.get("seed")) + ELLIPSIS, "muted")
        self.host._spawn(self.s.event_id, self._submit, s)

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
        self.apply(settings)

    def _open_selected(self, select=False):
        path = self.pending_preview
        if path and os.path.exists(path):
            open_path(path, select)

    # =============================================================== editors
    def _saved(self, kind):
        self._rebuild_choices()
        if kind == "backends":
            self.studio.clients.clear()
            self._paint_health()
            self.refresh_backends()
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
            ("always", "Always on (every picture from a model it suits)", "bool"),
            ("family", "Trained for", ("choice", [("", "unknown")] + list(ig.FAMILIES.items()))),
            ("preview", "Preview image", "path"),
            ("notes", "Notes", "long"),
        ], template={"file": "new_lora.safetensors", "category": "Other"},
            extra=("Scan backends", self._scan_loras),
            label=lambda r: "%s — %s%s" % (r["category"], r["name"],
                                            "  (always on)" if r.get("always") else ""))

    def _scan_loras(self, editor):
        editor.status("Scanning" + ELLIPSIS)

        def work():
            self.studio.check_all()
            n = self.studio.scan_loras()
            self._post("library")
            self._post("said", ("LoRA scan: %d new in the library." % n, "muted"))
            self.host.q.put(("images", self.s.event_id, ("editor-reload", editor)))
        self.host._spawn(self.s.event_id, work)

    def edit_identities(self):
        loras = [("", "none")] + [(r["id"], "%s (%s)" % (r["name"], r["category"]))
                                  for r in self.studio.lib.all("loras")]
        return RecordEditor(self, "identities", "Identities", [
            ("name", "Name", "text"),
            ("lora", "Identity LoRA", ("choice", loras)),
            ("trigger", "Trigger token", "text"),
            ("strength", "Default LoRA strength", "number"),
            ("references", "Reference photos", "paths"),
            ("use_references", "Use the first photo as the face reference", "bool"),
            ("reference_strength", "Face reference strength", "number"),
            ("notes", "Notes", "long"),
        ], template={"name": "New person", "strength": 0.85},
            extra=("Pick person…", self._pick_person))

    # ------------------------------------------------------- person cut-out
    def _pick_person(self, editor):
        """The selected reference photo (else the first) cut down to one
        person on white: SAM3 finds everyone; with more than one, a click
        says who. The cut-out takes the photo's place in the list, so it is
        the face reference when the photo was; the photo stays after it."""
        w = editor.widgets.get("references")
        pics = w[1] if w else None
        if not pics or not pics["paths"]:
            editor.status("Add a reference photo first.", "warn")
            return
        i = min(pics["sel"]) if pics["sel"] else 0
        path = pics["paths"][i]
        rec = editor.records[editor.current]
        editor.status("Finding the people in the photo" + ELLIPSIS)

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

        def find():
            try:
                found = self.studio.find_people(path)
            except (ig.ComfyError, OSError) as e:
                later(lambda: editor.status(str(e), "err"))
                return
            boxes = found["boxes"]
            if not boxes:
                later(lambda: editor.status("SAM3 found nobody in that photo.", "warn"))
            elif len(boxes) == 1:
                later(lambda: editor.status("One person; cutting them out" + ELLIPSIS))
                cut(found, boxes[0])
            else:
                later(lambda: self._choose_person(editor, found, lambda b: (
                    editor.status("Cutting them out" + ELLIPSIS),
                    self.host._spawn(self.s.event_id, cut, found, b))))
        self.host._spawn(self.s.event_id, find)

    def _choose_person(self, editor, found, then):
        """A window with the photo and a numbered box round each person; a
        click picks the one under it."""
        top = tk.Toplevel(editor.win)
        top.title("Who is it?")
        top.transient(editor.win)
        self.skin(top, bg="bg")
        self.label(top, "%d people. Click the one this identity is." % len(found["boxes"]),
                   "text").pack(side="top", fill="x", padx=self.px(12), pady=(self.px(10), 0))
        img = None
        if found["preview"]:
            tmp = os.path.join(self.studio.lib.root, "references", "_people.png")
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
        cv = tk.Canvas(top, width=cw, height=ch, bd=0, highlightthickness=0, cursor="hand2")
        self.skin(cv, bg="card")
        cv.pack(side="top", padx=self.px(12), pady=self.px(10))
        if img:
            top._img = img
            cv.create_image(0, 0, image=img, anchor="nw")
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

    def release(self):
        """On the UI thread, before the tab's frame goes (`Chat._close_tab`,
        `Chat._quit`): the Scene Builder writes into this form, so it cannot
        outlive it. `close()` runs on a worker thread and must not touch Tk."""
        sb, self.scene_builder = self.scene_builder, None
        try:
            if sb is not None and sb.win.winfo_exists():
                sb.close(final=True)
        except tk.TclError:
            pass

    def close(self):
        self.studio.close()


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
        if extra:
            owner.button(left, extra[0], lambda: extra[1](self)).pack(
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

    def reload(self):
        self.records = [dict(r) for r in self.owner.studio.lib.all(self.kind)]
        self._reload_list(0 if self.records else None)
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
        for key, label, kind in self.fields:
            o.label(self.form, label, "muted", host.f_small).pack(
                side="top", fill="x", pady=(o.px(8), o.px(2)))
            val = rec.get(key)
            if kind in ("text", "number", "path"):
                var = tk.StringVar(value="" if val is None else str(val))
                row = o.frame(self.form)
                row.pack(side="top", fill="x")
                if kind == "path":
                    o.button(row, "Choose…", lambda v=var: self._choose(v)).pack(
                        side="right", padx=(o.px(6), 0))
                e = host._entry(row, var)
                e.master.pack(side="left", fill="x", expand=True)
                self.widgets[key] = (kind, var)
                if kind == "path" and val and os.path.isfile(val):
                    img = photo(val, o.px(160))
                    if img is not None:
                        o.keep.append(img)
                        tk.Label(self.form, image=img, bd=0).pack(side="top", anchor="w",
                                                                  pady=o.px(4))
            elif kind == "long":
                t = tk.Text(self.form, height=3, wrap="word", bd=0, highlightthickness=0,
                            font=host.f_ui, padx=o.px(6), pady=o.px(4))
                o.skin(t, bg="card", fg="text", insertbackground="accent")
                t.insert("1.0", val or "")
                t.pack(side="top", fill="x")
                self.widgets[key] = (kind, t)
            elif kind == "bool":
                var = tk.BooleanVar(value=bool(val))
                b = tk.Checkbutton(self.form, text="yes", variable=var, anchor="w",
                                   font=host.f_ui, bd=0, highlightthickness=0)
                o.skin(b, bg="bg", fg="text", activebackground="bg", selectcolor="card",
                       activeforeground="text")
                b.pack(side="top", fill="x")
                self.widgets[key] = (kind, var)
            elif isinstance(kind, tuple) and kind[0] == "choice":
                var = tk.StringVar(value=val or "")
                o.choice(self.form, kind[1], val or "", var.set).pack(side="top", anchor="w")
                self.widgets[key] = ("choice", var)
            elif isinstance(kind, tuple) and kind[0] == "multi":
                have, vars_ = set(val or []), {}
                grid = o.frame(self.form)
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
                pics = {"paths": list(val or []), "sel": set(), "grid": o.frame(self.form)}
                pics["grid"].pack(side="top", fill="x")
                self._draw_paths(pics)
                row = o.frame(self.form)
                row.pack(side="top", fill="x", pady=(o.px(4), 0))
                o.button(row, "Add photos…", lambda p=pics: self._add_paths(p)).pack(
                    side="left")
                o.button(row, "Remove", lambda p=pics: self._remove_paths(p),
                         kind="ghost").pack(side="left", padx=(o.px(4), 0))
                self.widgets[key] = ("paths", pics)
            elif kind == "kv":
                t = tk.Text(self.form, height=5, wrap="none", bd=0, highlightthickness=0,
                            font=host.f_mono, padx=o.px(6), pady=o.px(4))
                o.skin(t, bg="card", fg="text", insertbackground="accent")
                t.insert("1.0", "\n".join("%s = %s" % kv for kv in (val or {}).items()))
                t.pack(side="top", fill="x")
                self.widgets[key] = ("kv", t)
            elif kind in ("per_backend", "per_backend_file"):
                per = {}
                for b in backends:
                    row = o.frame(self.form)
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
            img = photo(p, side) if os.path.isfile(p) else None
            if img is not None:
                o.keep.append(img)
                lbl = tk.Label(tile, image=img, bd=0, width=side, height=side)
            else:
                name = os.path.basename(p) + ("" if os.path.isfile(p) else "\n(missing)")
                lbl = tk.Label(tile, text=name, bd=0, font=host.f_small,
                               wraplength=side - o.px(8), width=12, height=6)
            o.skin(lbl, bg="card", fg="muted")
            lbl.pack()
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
        rec = self.records[self.current]
        for p in paths:
            try:
                kept = self.owner.studio.lib.keep_reference(p, rec.get("name") or "person")
            except OSError as e:
                self.status("Could not copy %s: %s" % (p, e), "err")
                continue
            pics["paths"].append(kept)
        self._draw_paths(pics)

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


class CharacterCreator:
    """The character creator, laid out like a video game's: the characters on
    the left; on the right a name, the identity that carries the face, and
    tabs - Body (with the sliders), Face, Hair, Clothes, Accessories - of
    picks to click, each slot also taking free text; a picture per item worn;
    Randomize; and the character sheet, the prompt text it makes, underneath.
    The anatomy constants have a tab of their own, locked: every character
    has them. Expressions are not here - they are the picture's, on the form."""

    SECTIONS = [name for name, _ in ig.LOOKS if name != "Expression"] + ["Constants"]

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
        win = self.win = tk.Toplevel(host)
        win.title("Character creator")
        win.transient(host)
        host._skin(win, bg="bg")
        win.geometry("%dx%d" % (host._px(940), host._px(680)))

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
        self.owner.item_rows(p, items, self.item_refs, self._choose_item, self._set_item)

    def _choose_item(self, item):
        path = filedialog.askopenfilename(parent=self.win, title="A picture of the " + item,
                                          filetypes=[("Pictures", "*.png *.jpg *.jpeg *.webp"),
                                                     ("All files", "*.*")])
        if path:
            self._set_item(item, path)

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
        """A roll of the dice for the tab shown (every look tab, from Constants)."""
        tab = None if self.section == "Constants" else [self.section]
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
