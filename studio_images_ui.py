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

import os
import subprocess
import sys
import time
import tkinter as tk
from tkinter import filedialog

import studio_imagegen as ig

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


class ImageStudio:
    def __init__(self, host, session):
        self.host, self.s = host, session
        self.studio = ig.Studio(notify=self._notify, make_room=host._images_make_room)
        self.settings = ig.default_settings()
        self.idents = {}              # identity id -> (BooleanVar, DoubleVar, scale row)
        self.loras = []               # [{"id", "var", "row"}]
        self.refs = {}                # kind -> local path
        self.adv = {}                 # setting -> StringVar
        self.text = {}                # person and camera setting -> StringVar
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
        self.adv_open = False
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

    def button(self, parent, text, command, kind="quiet", bg="bg"):
        return self.host._button(parent, text, command, kind=kind, bg=bg)

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

        self.cap(f, "Person").pack(**pad)
        self.person_box = self.frame(f)
        self.person_box.pack(side="top", fill="x", **pad)
        self.label(f, "Who, and what they look like. Any of these can stay blank.",
                   "faint", self.host.f_small, wraplength=self.px(380)).pack(
            side="top", fill="x", pady=(self.px(6), 0), **pad)
        grid = self.frame(f)
        grid.pack(side="top", fill="x", **pad)
        for i, (key, label) in enumerate([("subject", "Who")]
                                         + [(k, lab) for k, lab, _ in ig.PERSON_FIELDS]):
            self.label(grid, label, "muted").grid(row=i, column=0, sticky="w",
                                                  pady=self.px(1))
            var = tk.StringVar()
            entry = self.host._entry(grid, var)
            entry.master.grid(row=i, column=1, sticky="we", padx=(self.px(8), 0))
            entry.bind("<KeyRelease>", lambda ev: self._recheck())
            self.text[key] = var
        grid.columnconfigure(1, weight=1)

        self.cap(f, "Camera").pack(**pad)
        self.text["camera"] = tk.StringVar()
        e = self.host._entry(f, self.text["camera"])
        e.master.pack(side="top", fill="x", **pad)
        e.bind("<KeyRelease>", lambda ev: self._recheck())
        self.label(f, "Lens, angle, light, film: \u201c85mm, shallow depth of field, "
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
        self._rebuild_choices()

    def _rebuild_choices(self):
        """Everything drawn from the library: models, backends, people, styles,
        LoRA rows. Called again after any editor saves."""
        lib = self.studio.lib
        self._rebuild_models()
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

    # ------------------------------------------------------------ references
    def _build_refs(self):
        self.ref_labels = {}
        for kind, label, about in ig.REFERENCE_KINDS:
            row = self.frame(self.ref_box)
            row.pack(side="top", fill="x", pady=(0, self.px(2)))
            self.label(row, label, "text", width=13).pack(side="left")
            clear = self.button(row, "×", lambda k=kind: self._set_ref(k, None),
                                kind="ghost")
            clear.pack(side="right")
            self.button(row, "Choose…", lambda k=kind, a=about: self._pick_ref(k, a),
                        kind="quiet").pack(side="right", padx=(self.px(4), 0))
            name = self.label(row, "—", "faint", self.host.f_small)
            name.pack(side="left", fill="x", expand=True)
            self.ref_labels[kind] = name

    def _pick_ref(self, kind, about):
        path = filedialog.askopenfilename(
            parent=self.host, title=about,
            filetypes=[("Pictures", "*.png *.jpg *.jpeg *.webp *.bmp"), ("All files", "*.*")])
        if path:
            self._set_ref(kind, path)

    def _set_ref(self, kind, path):
        if path:
            self.refs[kind] = path
        else:
            self.refs.pop(kind, None)
        self.ref_labels[kind].config(text=os.path.basename(path) if path else "—")
        self._recheck()

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
        s["negative"] = self.neg.get().strip()
        s["identities"] = [{"id": iid, "strength": round(sv.get(), 3)}
                           for iid, (bv, sv, _) in self.idents.items() if bv.get()]
        st = self.studio.lib.get("styles", s["style"])
        s["style_strength"] = (round(self.style_strength.get(), 3)
                               if st and st["lora"] else None)
        s["loras"] = [{"id": r["id"], "strength": round(r["var"].get(), 3)}
                      for r in self.loras]
        s["references"] = dict(self.refs)
        s["refine"] = bool(self.refine.get())
        s["face_detail"] = bool(self.faces.get())
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
        self.neg.set(s.get("negative") or "")
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
        for key, _, _ in ADVANCED:
            val = v.get(key)
            if key == "seed":
                val = "random" if self.random_seed.get() else ""
            elif key == "batch":
                val = 1
            self.hints[key].config(text="" if val in (None, "") else "default %s" % val)

    # ============================================================== generate
    def generate(self):
        """Refuses, in words, what cannot run where it would go: a missing
        file or node, no backend able to take it. Routing that has not heard
        from the backends yet is left to submit(), which asks them."""
        s = self.collect()
        b, why = self.studio.plan_route(s)
        known = all(bk["id"] in self.studio.health for bk in self.studio.backends()
                    if bk["enabled"])
        if b is None and known:
            self.say(why, "err")
            return
        if b is not None:
            p = self.studio.preview(s, b)
            if p is not None and p.errors:
                self.say("Not sent. " + " ".join(p.errors), "err")
                return
        self.say("Routing" + ELLIPSIS, "muted")
        self.host._spawn(self.s.event_id, self._submit, s)

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
                       s.get("seed"))}
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
        self.button(btns, "Reuse", lambda: self.apply(rec["settings"]), bg="card").pack(
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
            self.apply(s)

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
        wfs = [(w["id"], w.get("label", w["id"])) for w in ig.list_workflows()]
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
            ("family", "Trained for", ("choice", [("", "unknown")] + list(ig.FAMILIES.items()))),
            ("preview", "Preview image", "path"),
            ("notes", "Notes", "long"),
        ], template={"file": "new_lora.safetensors", "category": "Other"},
            extra=("Scan backends", self._scan_loras),
            label=lambda r: "%s — %s" % (r["category"], r["name"]))

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
        ], template={"name": "New person", "strength": 0.85})

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
                lb = tk.Listbox(self.form, height=4, bd=0, highlightthickness=0,
                                font=host.f_small)
                o.skin(lb, bg="card", fg="text", selectbackground="sel")
                for p in val or []:
                    lb.insert("end", p)
                lb.pack(side="top", fill="x")
                row = o.frame(self.form)
                row.pack(side="top", fill="x", pady=(o.px(4), 0))
                o.button(row, "Add photos…", lambda l=lb: self._add_paths(l)).pack(
                    side="left")
                o.button(row, "Remove", lambda l=lb: [l.delete(i) for i in
                                                      reversed(l.curselection())],
                         kind="ghost").pack(side="left", padx=(o.px(4), 0))
                self.widgets[key] = ("paths", lb)
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

    def _add_paths(self, lb):
        paths = filedialog.askopenfilenames(parent=self.win, filetypes=[
            ("Pictures", "*.png *.jpg *.jpeg *.webp"), ("All files", "*.*")])
        rec = self.records[self.current]
        for p in paths:
            try:
                kept = self.owner.studio.lib.keep_reference(p, rec.get("name") or "person")
            except OSError as e:
                self.status("Could not copy %s: %s" % (p, e), "err")
                continue
            lb.insert("end", kept)

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
                rec[key] = list(w.get(0, "end"))
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
